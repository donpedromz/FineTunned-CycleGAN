"""CycleGAN training loop with frozen-encoder fine-tuning.

Implements the training entry point ``train_cyclegan`` plus the loss functions
and the history-buffer ``ImagePool`` used to stabilise discriminator training.

Loss functions are injectable (dependency inversion): ``train_cyclegan`` accepts
callables for the GAN, cycle, and identity losses, defaulting to the standard
LSGAN (MSE) / L1 implementations below. This keeps the loop open for extension
without modification.

Usage:
    history = train_cyclegan(gen_ab, gen_ba, d_a, d_b, dl, config, reg, run_id, device)
    # or: python src/training.py  (runs a 5-epoch smoke test on synthetic data)
"""

from __future__ import annotations

import logging
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.model import PatchGANDiscriminator, ResNetGenerator
from src.registry import ModelRegistry

logger = logging.getLogger(__name__)


def lsa_gan_loss(
    pred: torch.Tensor, is_real: bool, criterion: nn.Module = nn.MSELoss()
) -> torch.Tensor:
    """Least-squares GAN loss (no sigmoid). Target is 1 for real, 0 for fake."""
    target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
    return criterion(pred, target)


def cycle_loss(
    reconstructed: torch.Tensor, real: torch.Tensor, criterion: nn.Module = nn.L1Loss()
) -> torch.Tensor:
    """L1 cycle-consistency loss: ||G_BA(G_AB(A)) - A||."""
    return criterion(reconstructed, real)


def identity_loss(
    same_domain: torch.Tensor, real: torch.Tensor, criterion: nn.Module = nn.L1Loss()
) -> torch.Tensor:
    """L1 identity loss: G_AB(B) should equal B when B is already in target domain."""
    return criterion(same_domain, real)


class ImagePool:
    """History buffer that returns a stored image with 50% probability.

    Reduces discriminator oscillation by mixing in previously generated fakes.
    """

    def __init__(self, size: int = 50) -> None:
        self.size = size
        self.images: list[torch.Tensor] = []

    def query(self, image: torch.Tensor) -> torch.Tensor:
        if len(self.images) < self.size:
            self.images.append(image)
            return image
        if random.random() < 0.5:
            idx = random.randrange(self.size)
            out = self.images[idx].clone()
            self.images[idx] = image
            return out
        self.images.append(image)
        if len(self.images) > self.size:
            self.images.pop(0)
        return image


def train_cyclegan(
    gen_ab: nn.Module,
    gen_ba: nn.Module,
    d_a: nn.Module,
    d_b: nn.Module,
    dl_train: DataLoader,
    config: dict,
    registry: ModelRegistry,
    run_id: str,
    device: torch.device,
    gan_loss_fn=lsa_gan_loss,
    cycle_loss_fn=cycle_loss,
    identity_loss_fn=identity_loss,
    progress: bool = True,
) -> dict[str, list[float]]:
    """Train a CycleGAN pair with frozen encoders and registry checkpointing.

    Returns a dict of per-epoch loss curves (lists, one value per epoch).
    """
    total_epochs = int(config.get("epochs", 100))
    lr = float(config.get("lr", 2e-4))
    betas = tuple(config.get("betas", (0.5, 0.999)))
    lambda_cycle = float(config.get("lambda_cycle", 10.0))
    lambda_identity = float(config.get("lambda_identity", 0.5))
    pool_size = int(config.get("pool_size", 50))
    checkpoint_interval = int(config.get("checkpoint_interval", 10))
    decay_epochs = int(config.get("decay_epochs", total_epochs))

    gen_ab.freeze_encoder()
    gen_ba.freeze_encoder()
    for m in (gen_ab, gen_ba, d_a, d_b):
        m.to(device).train()

    opt_g = torch.optim.Adam(
        gen_ab.trainable_parameters() + gen_ba.trainable_parameters(),
        lr=lr,
        betas=betas,
    )
    opt_d = torch.optim.Adam(
        list(d_a.parameters()) + list(d_b.parameters()), lr=lr, betas=betas
    )

    pool_a, pool_b = ImagePool(pool_size), ImagePool(pool_size)
    gan, l1 = nn.MSELoss(), nn.L1Loss()
    history: dict[str, list[float]] = {
        k: []
        for k in (
            "G_total",
            "D_total",
            "cycle",
            "identity",
            "g_adv",
            "d_a",
            "d_b",
            "lr",
        )
    }

    for epoch in range(1, total_epochs + 1):
        n_decay = max(1, total_epochs - decay_epochs)
        lr_factor = 1.0 - max(0, epoch - decay_epochs) / n_decay
        for pg in (*opt_g.param_groups, *opt_d.param_groups):
            pg["lr"] = lr * lr_factor

        epoch_bar = tqdm(
            dl_train,
            desc=f"Epoch {epoch}/{total_epochs}",
            unit="batch",
            leave=True,
            disable=not progress,
        )
        lr_now = lr * lr_factor
        run = {
            "G_total": 0.0,
            "D_total": 0.0,
            "cycle": 0.0,
            "identity": 0.0,
            "g_adv": 0.0,
            "d_a": 0.0,
            "d_b": 0.0,
            "lr": lr_now,
            "n": 0,
        }
        for real_a, real_b in epoch_bar:
            real_a, real_b = real_a.to(device), real_b.to(device)

            fake_b, fake_a = gen_ab(real_a), gen_ba(real_b)
            rec_a, rec_b = gen_ba(fake_b), gen_ab(fake_a)
            id_b, id_a = gen_ab(real_b), gen_ba(real_a)

            g_adv = gan_loss_fn(d_a(fake_a), True, gan) + gan_loss_fn(
                d_b(fake_b), True, gan
            )
            cyc = lambda_cycle * (
                cycle_loss_fn(rec_a, real_a, l1) + cycle_loss_fn(rec_b, real_b, l1)
            )
            idt = lambda_identity * (
                identity_loss_fn(id_b, real_b, l1) + identity_loss_fn(id_a, real_a, l1)
            )
            g_loss = g_adv + cyc + idt

            opt_g.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                gen_ab.trainable_parameters() + gen_ba.trainable_parameters(),
                max_norm=1.0,
            )
            opt_g.step()

            fake_a_pool = pool_a.query(fake_a.detach())
            fake_b_pool = pool_b.query(fake_b.detach())
            d_a_loss = 0.5 * (
                gan_loss_fn(d_a(real_a), True, gan)
                + gan_loss_fn(d_a(fake_a_pool), False, gan)
            )
            d_b_loss = 0.5 * (
                gan_loss_fn(d_b(real_b), True, gan)
                + gan_loss_fn(d_b(fake_b_pool), False, gan)
            )
            d_loss = d_a_loss + d_b_loss

            opt_d.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(d_a.parameters()) + list(d_b.parameters()), max_norm=1.0
            )
            opt_d.step()

            run["G_total"] += g_loss.item()
            run["D_total"] += d_loss.item()
            run["cycle"] += cyc.item()
            run["identity"] += idt.item()
            run["g_adv"] += g_adv.item()
            run["d_a"] += d_a_loss.item()
            run["d_b"] += d_b_loss.item()
            run["n"] += 1
            epoch_bar.set_postfix(G=f"{g_loss.item():.3f}", D=f"{d_loss.item():.3f}")
        epoch_bar.close()

        for k in ("G_total", "D_total", "cycle", "identity", "g_adv", "d_a", "d_b"):
            history[k].append(run[k] / max(1, run["n"]))
        history["lr"].append(run["lr"])
        logger.info(
            "epoch %d/%d  lr=%.1e  G=%.4f D=%.4f g_adv=%.4f d_a=%.4f d_b=%.4f cyc=%.4f id=%.4f",
            epoch,
            total_epochs,
            run["lr"],
            history["G_total"][-1],
            history["D_total"][-1],
            history["g_adv"][-1],
            history["d_a"][-1],
            history["d_b"][-1],
            history["cycle"][-1],
            history["identity"][-1],
        )

        if epoch % checkpoint_interval == 0 or epoch == total_epochs:
            registry.save(
                gen_ab,
                gen_ba,
                run_id=run_id,
                config=config,
                training_results={
                    "epoch": epoch,
                    "cycle_loss": float(history["cycle"][-1]),
                    "g_loss": float(history["G_total"][-1]),
                    "d_loss": float(history["D_total"][-1]),
                },
                base_checkpoint=config.get("base_checkpoint"),
            )
            logger.info("Checkpoint saved at epoch %d", epoch)

    test_dir_a = config.get("test_dir_a")
    test_dir_b = config.get("test_dir_b")
    if (
        test_dir_a
        and test_dir_b
        and Path(test_dir_a).is_dir()
        and Path(test_dir_b).is_dir()
    ):
        from src.evaluation import compute_fid, compute_lpips, generate_translations

        logger.info("Running post-training evaluation …")
        gen_ab.eval(), gen_ba.eval()
        with torch.no_grad():
            tmp_a = Path(test_dir_a).parent / "_eval_fake_a"
            tmp_b = Path(test_dir_b).parent / "_eval_fake_b"
            generate_translations(gen_ab, test_dir_a, tmp_a, str(device))
            generate_translations(gen_ba, test_dir_b, tmp_b, str(device))
            fid_ab = compute_fid(test_dir_b, tmp_a, device=str(device))
            fid_ba = compute_fid(test_dir_a, tmp_b, device=str(device))
            lpips_ab = compute_lpips(test_dir_a, tmp_a, device=str(device))
            lpips_ba = compute_lpips(test_dir_b, tmp_b, device=str(device))
        registry.update_meta(
            run_id,
            evaluation={
                "fid_ab": fid_ab,
                "fid_ba": fid_ba,
                "lpips_ab": lpips_ab,
                "lpips_ba": lpips_ba,
            },
        )
        history["fid_ab"] = [fid_ab]
        history["fid_ba"] = [fid_ba]
        history["lpips_ab"] = [lpips_ab]
        history["lpips_ba"] = [lpips_ba]
        logger.info(
            "Eval done — FID: %.2f/%.2f, LPIPS: %.3f/%.3f",
            fid_ab,
            fid_ba,
            lpips_ab,
            lpips_ba,
        )

    return history


def _select_device() -> torch.device:
    """Use CUDA only when it has headroom; otherwise fall back to CPU."""
    if torch.cuda.is_available():
        try:
            if torch.cuda.mem_get_free_memory() > 1_000_000_000:  # > ~1 GiB free
                return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


def _smoke_test() -> None:
    torch.manual_seed(0)
    device = _select_device()
    n, bs = 4, 2
    dl = DataLoader(
        TensorDataset(torch.randn(n, 3, 256, 256), torch.randn(n, 3, 256, 256)),
        batch_size=bs,
        shuffle=True,
    )
    config = {
        "epochs": 3,
        "lr": 2e-4,
        "decay_epochs": 100,
        "lambda_cycle": 10.0,
        "lambda_identity": 0.5,
        "checkpoint_interval": 10,
    }

    def _run(dev: torch.device) -> dict[str, list[float]]:
        gen_ab, gen_ba = ResNetGenerator().to(dev), ResNetGenerator().to(dev)
        d_a, d_b = PatchGANDiscriminator().to(dev), PatchGANDiscriminator().to(dev)
        with tempfile.TemporaryDirectory() as tmp:
            reg = ModelRegistry(tmp)
            hist = train_cyclegan(
                gen_ab,
                gen_ba,
                d_a,
                d_b,
                dl,
                config,
                reg,
                "smoke001",
                dev,
                progress=False,
            )
            run_dir = Path(tmp) / "smoke001"
            assert (run_dir / "gen_AB.pth").exists() and (
                run_dir / "meta.json"
            ).exists(), "Final checkpoint not saved"
        return hist

    try:
        history = _run(device)
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA OOM — retrying smoke test on CPU")
        history = _run(torch.device("cpu"))

    assert min(history["G_total"]) < history["G_total"][0], (
        f"G loss never decreased from start: {history['G_total'][0]:.4f} -> {history['G_total']}"
    )
    print(
        f"training self-check PASSED ({device.type}) — "
        f"G loss {history['G_total'][0]:.4f} -> {min(history['G_total']):.4f}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _smoke_test()

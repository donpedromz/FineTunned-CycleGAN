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

from src.model import PatchGANDiscriminator, ResNetGenerator
from src.registry import ModelRegistry

logger = logging.getLogger(__name__)


# ── Losses (injectable) ────────────────────────────────────────────────────────


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


# ── Image pool ─────────────────────────────────────────────────────────────────


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


# ── Training loop ──────────────────────────────────────────────────────────────


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
        k: [] for k in ("G_total", "D_total", "cycle", "identity")
    }

    for epoch in range(1, total_epochs + 1):
        lr_factor = 1.0 - max(0, epoch - 1) / max(1, decay_epochs - 1)
        for pg in (*opt_g.param_groups, *opt_d.param_groups):
            pg["lr"] = lr * lr_factor

        run = {"G_total": 0.0, "D_total": 0.0, "cycle": 0.0, "identity": 0.0, "n": 0}
        for real_a, real_b in dl_train:
            real_a, real_b = real_a.to(device), real_b.to(device)

            # ── Generators ──
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
            opt_g.step()

            # ── Discriminators (history buffer) ──
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
            opt_d.step()

            run["G_total"] += g_loss.item()
            run["D_total"] += d_loss.item()
            run["cycle"] += cyc.item()
            run["identity"] += idt.item()
            run["n"] += 1

        for k in ("G_total", "D_total", "cycle", "identity"):
            history[k].append(run[k] / max(1, run["n"]))
        logger.info(
            "epoch %d/%d  G=%.4f D=%.4f cyc=%.4f id=%.4f",
            epoch,
            total_epochs,
            history["G_total"][-1],
            history["D_total"][-1],
            history["cycle"][-1],
            history["identity"][-1],
        )

        if epoch % checkpoint_interval == 0 or epoch == total_epochs:
            registry.save(
                gen_ab,
                gen_ba,
                run_id=run_id,
                epoch=epoch,
                cycle_loss=float(history["cycle"][-1]),
                approach=config.get("approach", "frozen-encoder"),
                lr=lr,
                lambda_cycle=lambda_cycle,
                lambda_identity=lambda_identity,
            )
            logger.info("Checkpoint saved at epoch %d", epoch)

    return history


# ── Self-check ─────────────────────────────────────────────────────────────────


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
                gen_ab, gen_ba, d_a, d_b, dl, config, reg, "smoke001", dev
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

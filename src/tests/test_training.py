import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import PatchGANDiscriminator, ResNetGenerator
from src.registry import ModelRegistry
from src.training import train_cyclegan


def test_smoke():
    torch.manual_seed(0)
    device = torch.device("cpu")
    n, bs = 4, 2
    dl = DataLoader(
        TensorDataset(torch.randn(n, 3, 256, 256), torch.randn(n, 3, 256, 256)),
        batch_size=bs,
        shuffle=True,
    )
    gen_ab = ResNetGenerator()
    gen_ba = ResNetGenerator()
    d_a = PatchGANDiscriminator()
    d_b = PatchGANDiscriminator()
    config = {
        "epochs": 3,
        "lr": 2e-4,
        "decay_epochs": 100,
        "lambda_cycle": 10.0,
        "lambda_identity": 0.5,
        "checkpoint_interval": 10,
    }

    with tempfile.TemporaryDirectory() as tmp:
        reg = ModelRegistry(tmp)
        hist = train_cyclegan(
            gen_ab, gen_ba, d_a, d_b, dl, config, reg, "test001", device,
            progress=False,
        )

        for k in ("G_total", "D_total", "cycle", "lr"):
            assert k in hist, f"Missing key in history: {k}"
            assert len(hist[k]) == 3, f"Expected 3 values for {k}, got {len(hist[k])}"

        assert min(hist["G_total"]) < hist["G_total"][0], (
            f"G loss never decreased: {hist['G_total'][0]:.4f} -> {min(hist['G_total']):.4f}"
        )

        last_ckpt = "test001-e03"
        run_dir = Path(tmp) / last_ckpt
        assert (run_dir / "gen_AB.pth").exists(), "Final gen_AB.pth not saved"
        assert (run_dir / "meta.json").exists(), "Final meta.json not saved"

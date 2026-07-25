import math
import tempfile
from pathlib import Path

import torch

from src.evaluation import _save_tensor, compute_fid, compute_lpips, create_visual_grid
from src.model import ResNetGenerator


def test_fid_and_lpips():
    torch.manual_seed(0)
    device = "cpu"
    gen_ab = ResNetGenerator().to(device)
    gen_ba = ResNetGenerator().to(device)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        real_dir = root / "real"
        fake_dir = root / "fake"
        real_dir.mkdir()
        fake_dir.mkdir()
        for i in range(8):
            _save_tensor(
                gen_ab(torch.randn(1, 3, 256, 256))[0], real_dir / f"img_{i}.png"
            )
            _save_tensor(
                gen_ba(torch.randn(1, 3, 256, 256))[0], fake_dir / f"img_{i}.png"
            )

        fid_score = compute_fid(real_dir, fake_dir, device=device)
        lpips_score = compute_lpips(real_dir, fake_dir, device=device)

        assert math.isfinite(fid_score), f"FID not finite: {fid_score}"
        assert math.isfinite(lpips_score), f"LPIPS not finite: {lpips_score}"


def test_visual_grid():
    torch.manual_seed(0)
    device = "cpu"
    gen_ab = ResNetGenerator().to(device)
    gen_ba = ResNetGenerator().to(device)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "lion").mkdir()
        (root / "cheetah").mkdir()
        for i in range(8):
            _save_tensor(
                gen_ab(torch.randn(1, 3, 256, 256))[0], root / "lion" / f"l_{i}.png"
            )
            _save_tensor(
                gen_ba(torch.randn(1, 3, 256, 256))[0], root / "cheetah" / f"c_{i}.png"
            )

        grid = root / "grid.png"
        create_visual_grid(gen_ab, gen_ba, root, grid, n=4, device=device)

        assert grid.exists()
        assert grid.stat().st_size > 0

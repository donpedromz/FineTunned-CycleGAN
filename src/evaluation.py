"""Model evaluation: FID, LPIPS, and visual translation grids.

Thin wrappers around ``torchmetrics`` plus a matplotlib grid renderer.
Every metric accepts directories of images.

Usage:
    fid = compute_fid("data/test/cheetah", "results/lion2cheetah")
    lpips = compute_lpips("data/test/lion", "results/lion2cheetah")
    create_visual_grid(gen_ab, gen_ba, "data/test", "grid.png")
    # or: python src/evaluation.py  (synthetic self-check)
"""

from __future__ import annotations

import logging
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image

from src.model import ResNetGenerator

logger = logging.getLogger(__name__)


def _load_tensor(path: str | Path, size: int = 256) -> torch.Tensor:
    """Load an image file as a (1,3,H,W) tensor normalised to [-1, 1]."""
    img = (
        Image.open(path)
        .convert("RGBA")
        .convert("RGB")
        .resize((size, size), Image.Resampling.BILINEAR)
    )
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0


def _save_tensor(tensor: torch.Tensor, path: str | Path) -> None:
    """Save a (1,3,H,W) or (3,H,W) tensor in [-1, 1] as a PNG."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = ((tensor.clamp(-1, 1) + 1.0) / 2.0 * 255.0).permute(1, 2, 0).byte().numpy()
    Image.fromarray(arr).save(path)


def generate_translations(
    gen: torch.nn.Module,
    source_dir: str | Path,
    out_dir: str | Path,
    device: str = "cpu",
) -> list[Path]:
    """Run ``gen`` over every image in ``source_dir`` and write results to ``out_dir``."""
    src, out = Path(source_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(
        p for p in src.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    gen.to(device).eval()
    saved: list[Path] = []
    with torch.no_grad():
        for p in files:
            x = _load_tensor(p).to(device)
            y = gen(x)[0].cpu()
            dst = out / p.name
            _save_tensor(y, dst)
            saved.append(dst)
    return saved


def _image_files(directory: str | Path) -> list[Path]:
    _IMG_EXT = {".png", ".jpg", ".jpeg"}
    return sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in _IMG_EXT)


def compute_fid(
    real_dir: str | Path, fake_dir: str | Path, device: str = "cpu"
) -> float:
    """Fréchet Inception Distance between two image directories (torchmetrics)."""
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=2048).to(device)
    batch_size = 32
    reals, fakes = _image_files(real_dir), _image_files(fake_dir)
    with torch.no_grad():
        for paths, split in [(reals, "real"), (fakes, "fake")]:
            for i in range(0, len(paths), batch_size):
                batch = torch.cat(
                    [_load_tensor(p) for p in paths[i : i + batch_size]], dim=0
                ).to(device)
                batch = (batch.clamp(-1, 1) + 1.0) / 2.0
                batch = (batch * 255.0).clamp(0, 255).to(torch.uint8)
                fid.update(batch, real=(split == "real"))
    return float(fid.compute().item())


def compute_lpips(
    real_dir: str | Path, fake_dir: str | Path, device: str = "cpu"
) -> float:
    """Mean LPIPS (AlexNet) between index-aligned image pairs (torchmetrics)."""
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    reals, fakes = _image_files(real_dir), _image_files(fake_dir)
    n = min(len(reals), len(fakes))
    if n == 0:
        raise ValueError("No paired images found for LPIPS")
    batch_size = 32
    vals: list[float] = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            r_batch = torch.cat(
                [_load_tensor(reals[j]) for j in range(i, min(i + batch_size, n))],
                dim=0,
            ).to(device)
            f_batch = torch.cat(
                [_load_tensor(fakes[j]) for j in range(i, min(i + batch_size, n))],
                dim=0,
            ).to(device)
            r_batch = (r_batch.clamp(-1, 1) + 1.0) / 2.0
            f_batch = (f_batch.clamp(-1, 1) + 1.0) / 2.0
            vals.append(metric(r_batch, f_batch).item())
    return float(np.mean(vals))


def create_visual_grid(
    gen_ab: torch.nn.Module,
    gen_ba: torch.nn.Module,
    test_root: str | Path,
    out_path: str | Path = "translation_grid.png",
    n: int = 4,
    device: str = "cpu",
) -> Path:
    """Render a 2×n grid: row 0 lion→cheetah, row 1 cheetah→lion. Returns grid path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(test_root)
    _IMG_EXT = {".png", ".jpg", ".jpeg"}
    lions = sorted(
        p for p in (root / "lion").iterdir() if p.suffix.lower() in _IMG_EXT
    )[:n]
    cheetahs = sorted(
        p for p in (root / "cheetah").iterdir() if p.suffix.lower() in _IMG_EXT
    )[:n]
    gen_ab.to(device).eval()
    gen_ba.to(device).eval()

    # Render row 0 = source domain, row 1 = translated domain for each direction.
    def _to_img(t: torch.Tensor) -> np.ndarray:
        arr = t.detach().clamp(-1, 1).float().cpu()
        if arr.dim() == 4:
            arr = arr[0]
        return arr.permute(1, 2, 0).numpy() * 0.5 + 0.5

    fig, axes = plt.subplots(2, 2 * n, figsize=(3 * 2 * n, 6))
    with torch.no_grad():
        for j, p in enumerate(lions):
            axes[0, j].imshow(_to_img(_load_tensor(p)))
            axes[1, j].imshow(_to_img(gen_ab(_load_tensor(p).to(device))))
        for j, p in enumerate(cheetahs):
            col = j + n
            axes[0, col].imshow(_to_img(_load_tensor(p)))
            axes[1, col].imshow(_to_img(gen_ba(_load_tensor(p).to(device))))
    for ax in axes.flat:
        ax.axis("off")
    axes[0, 0].set_ylabel("source", fontsize=10)
    axes[1, 0].set_ylabel("translated", fontsize=10)
    if n > 0 and len(lions) > 0:
        axes[0, n - 1].set_title("lion→cheetah", fontsize=9)
    if n > 0 and len(cheetahs) > 0:
        axes[0, n].set_title("cheetah→lion", fontsize=9)
    fig.suptitle("lion→cheetah (left) · cheetah→lion (right)", fontsize=10)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return Path(out_path)


def _smoke_test() -> None:
    torch.manual_seed(0)
    device = "cpu"
    gen_ab, gen_ba = ResNetGenerator().to(device), ResNetGenerator().to(device)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        real_dir, fake_dir = root / "real", root / "fake"
        real_dir.mkdir(), fake_dir.mkdir()
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

        (root / "lion").mkdir(), (root / "cheetah").mkdir()
        for i in range(8):
            _save_tensor(
                gen_ab(torch.randn(1, 3, 256, 256))[0], root / "lion" / f"l_{i}.png"
            )
            _save_tensor(
                gen_ba(torch.randn(1, 3, 256, 256))[0], root / "cheetah" / f"c_{i}.png"
            )
        grid = root / "grid.png"
        create_visual_grid(gen_ab, gen_ba, root, grid, n=4, device=device)
        assert grid.exists() and grid.stat().st_size > 0
        grid_size = grid.stat().st_size

    print(f"evaluation self-check PASSED — FID={fid_score:.3f} LPIPS={lpips_score:.3f}")
    print(f"grid written, {grid_size} bytes")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _smoke_test()

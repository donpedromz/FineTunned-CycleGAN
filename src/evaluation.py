"""Model evaluation: FID, LPIPS, and visual translation grids.

Thin wrappers around ``clean-fid`` and ``lpips`` (lazy-imported) plus a
matplotlib grid renderer. Every metric accepts directories of images, keeping
the module open for extension (new metrics just need a directory-based
``compute_*`` function).

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


# ── I/O helpers ────────────────────────────────────────────────────────────────


def _load_tensor(path: str | Path, size: int = 256) -> torch.Tensor:
    """Load an image file as a (1,3,H,W) tensor normalised to [-1, 1]."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0


def _save_tensor(tensor: torch.Tensor, path: str | Path) -> None:
    """Save a (1,3,H,W) or (3,H,W) tensor in [-1, 1] as a PNG."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = ((tensor.clamp(-1, 1) + 1.0) / 2.0 * 255.0).permute(1, 2, 0).byte().numpy()
    Image.fromarray(arr).save(path)


# ── Metrics ────────────────────────────────────────────────────────────────────


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


def frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    """Standard Fréchet distance (scipy.linalg.sqrtm — numpy 2.x dropped it)."""
    import scipy.linalg as sla

    diff = mu1 - mu2
    covmean = np.real(sla.sqrtm(sigma1 @ sigma2))
    if not np.isfinite(covmean).all():
        covmean = np.zeros_like(covmean)
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_fid(
    real_dir: str | Path, fake_dir: str | Path, device: str = "cpu"
) -> float:
    """Fréchet Inception Distance between two image directories.

    Uses ``clean-fid`` for InceptionV3 feature extraction and computes the
    Fréchet distance directly (clean-fid 0.1.35's bundled sqrtm call is
    incompatible with scipy >= 1.13).
    """
    from cleanfid.features import build_feature_extractor
    from cleanfid.fid import get_folder_features

    model = build_feature_extractor("clean", device=device)
    f1 = get_folder_features(str(real_dir), model=model, device=device, batch_size=32)
    f2 = get_folder_features(str(fake_dir), model=model, device=device, batch_size=32)
    mu1, sigma1 = f1.mean(axis=0), np.cov(f1, rowvar=False)
    mu2, sigma2 = f2.mean(axis=0), np.cov(f2, rowvar=False)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


def compute_lpips(
    real_dir: str | Path, fake_dir: str | Path, device: str = "cpu"
) -> float:
    """Mean LPIPS (AlexNet) between index-aligned image pairs (lpips library)."""
    import lpips

    loss_fn = lpips.LPIPS(net="alex").to(device)
    reals = sorted(
        p
        for p in Path(real_dir).iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    fakes = sorted(
        p
        for p in Path(fake_dir).iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    n = min(len(reals), len(fakes))
    if n == 0:
        raise ValueError("No paired images found for LPIPS")
    vals: list[float] = []
    with torch.no_grad():
        for i in range(n):
            r = _load_tensor(reals[i]).to(device)
            f = _load_tensor(fakes[i]).to(device)
            vals.append(loss_fn(r, f).item())
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
    lions = sorted((root / "lion").glob("*"))[:n]
    cheetahs = sorted((root / "cheetah").glob("*"))[:n]
    gen_ab.to(device).eval()
    gen_ba.to(device).eval()

    # Render row 0 = source domain, row 1 = translated domain for each direction.
    def _to_img(t: torch.Tensor) -> np.ndarray:
        arr = t.detach().clamp(-1, 1).float().cpu()
        if arr.dim() == 4:
            arr = arr[0]
        return arr.permute(1, 2, 0).numpy() * 0.5 + 0.5

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    with torch.no_grad():
        for j, p in enumerate(lions or []):
            axes[0, j].imshow(_to_img(_load_tensor(p)))
            axes[1, j].imshow(_to_img(gen_ab(_load_tensor(p).to(device))))
        if lions and cheetahs:
            for j, p in enumerate(cheetahs):
                axes[0, j].imshow(_to_img(_load_tensor(p)))
                axes[1, j].imshow(_to_img(gen_ba(_load_tensor(p).to(device))))
    for ax in axes.flat:
        ax.axis("off")
    axes[0, 0].set_ylabel("source", fontsize=10)
    axes[1, 0].set_ylabel("translated", fontsize=10)
    fig.suptitle("lion→cheetah (top half) · cheetah→lion (bottom half)", fontsize=10)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return Path(out_path)


# ── Self-check ─────────────────────────────────────────────────────────────────


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

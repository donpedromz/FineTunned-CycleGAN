"""
Inference pipeline for pre-trained CycleGAN generators.

Handles device detection, image loading, checkpoint downloading,
bidirectional translation, and side-by-side visualisation.

Usage:
    engine = CycleGANInference()
    engine.download_checkpoints()
    engine.load_generators()
    original, translated = engine.translate(image_path, direction="AB")
    display_translations([original], [translated], "Lion -> Cheetah")
"""

from __future__ import annotations

import logging
import urllib.request
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from src.model import ResNetGenerator

logger = logging.getLogger(__name__)

# Checkpoint source (horse2zebra)
_CHECKPOINT_URLS: dict[str, str] = {
    "gen_AB.pth": "https://huggingface.co/johko/cyclegan-horse2zebra/resolve/main/gen_AB.pth",
    "gen_BA.pth": "https://huggingface.co/johko/cyclegan-horse2zebra/resolve/main/gen_BA.pth",
}
_DEFAULT_CHECKPOINT_DIR = Path("checkpoints") / "horse2zebra"


class InferenceError(Exception):
    """Base exception for inference pipeline errors."""


class DownloadError(InferenceError):
    """Raised when a checkpoint download fails."""


class ImageLoadError(InferenceError):
    """Raised when an image cannot be loaded or decoded."""


class GeneratorNotLoadedError(InferenceError):
    """Raised when attempting inference before loading generators."""


class CycleGANInference:
    """Orchestrates CycleGAN inference with pre-trained generators.

    Encapsulates device detection, checkpoint management, image preprocessing,
    and bidirectional translation. Follows the Single Responsibility Principle
    by separating inference orchestration from model architecture.

    Usage::

        engine = CycleGANInference()
        engine.download_checkpoints()
        engine.load_generators()
        translated_np = engine.translate("data/test/lion/0001.png", direction="AB")
    """

    def __init__(self, checkpoint_dir: str | Path = _DEFAULT_CHECKPOINT_DIR) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)
        self._device = self._detect_device()
        self._gen_ab: ResNetGenerator | None = None
        self._gen_ba: ResNetGenerator | None = None
        self._transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    # ── Device ────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_device() -> torch.device:
        """Detect and return the best available device."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
        else:
            warnings.warn(
                "CUDA not available — falling back to CPU.",
                stacklevel=2,
            )
            device = torch.device("cpu")
        return device

    @property
    def device(self) -> torch.device:
        return self._device

    # ── Checkpoint Management ─────────────────────────────────────────────

    def download_checkpoints(self) -> None:
        """Download both generator checkpoints, skipping existing files.

        Raises:
            DownloadError: If any download fails.
        """
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for name, url in _CHECKPOINT_URLS.items():
            dest = self._checkpoint_dir / name
            if dest.exists():
                logger.info("Checkpoint already exists: %s — skipping.", dest.name)
                continue
            try:
                logger.info("Downloading %s …", url)
                urllib.request.urlretrieve(url, dest)
                size_mb = dest.stat().st_size / (1024 * 1024)
                logger.info("Downloaded %s (%.1f MB).", dest.name, size_mb)
            except Exception as exc:
                raise DownloadError(f"Failed to download {name}: {exc}") from exc

    def load_generators(self) -> None:
        """Load both generators from downloaded checkpoints.

        Raises:
            GeneratorNotLoadedError: If checkpoint files are missing.
        """
        ckpt_ab = self._checkpoint_dir / "gen_AB.pth"
        ckpt_ba = self._checkpoint_dir / "gen_BA.pth"

        missing = [p for p in (ckpt_ab, ckpt_ba) if not p.exists()]
        if missing:
            names = ", ".join(p.name for p in missing)
            raise GeneratorNotLoadedError(
                f"Checkpoints not found: {names}. Call download_checkpoints() first."
            )

        self._gen_ab = ResNetGenerator.from_checkpoint(ckpt_ab, self._device)
        self._gen_ba = ResNetGenerator.from_checkpoint(ckpt_ba, self._device)

    # ── Image Processing ──────────────────────────────────────────────────

    def load_image(self, path: str | Path) -> torch.Tensor:
        """Load, normalise to [-1, 1], and add batch dimension.

        Args:
            path: Path to a 256x256 RGB PNG.

        Returns:
            Tensor of shape ``(1, 3, 256, 256)`` on the target device.

        Raises:
            ImageLoadError: If the file is missing, corrupt, or wrong format.
        """
        img_path = Path(path)
        if not img_path.exists():
            raise ImageLoadError(f"Image not found: {img_path}")

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            raise ImageLoadError(f"Cannot decode image {img_path}: {exc}") from exc

        tensor = self._transform(img).unsqueeze(0).to(self._device)
        return tensor

    @staticmethod
    def denormalize(tensor: torch.Tensor) -> np.ndarray:
        """Convert a [-1, 1] tensor to a [0, 1] numpy array in HWC format.

        Args:
            tensor: Shape ``(C, H, W)`` or ``(1, C, H, W)`` with values in [-1, 1].

        Returns:
            Numpy array ``(H, W, C)`` with dtype float32 in [0, 1].
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        arr = (tensor * 0.5 + 0.5).clamp(0, 1).cpu().detach().permute(1, 2, 0).numpy()
        return arr.astype(np.float32)

    # ── Translation ───────────────────────────────────────────────────────

    def translate(self, image_path: str | Path, direction: str = "AB") -> np.ndarray:
        """Translate a single image in the specified direction.

        Args:
            image_path: Path to a 256x256 RGB image.
            direction: ``"AB"`` for horse2zebra (lion->cheetah) or
                       ``"BA"`` for zebra2horse (cheetah->lion).

        Returns:
            Translated image as ``(H, W, C)`` numpy array in [0, 1].

        Raises:
            GeneratorNotLoadedError: If generators have not been loaded.
            ImageLoadError: If the image cannot be loaded.
        """
        generator = self._get_generator(direction)
        tensor = self.load_image(image_path)
        with torch.no_grad():
            output = generator(tensor)
        return self.denormalize(output)

    def translate_batch(
        self, image_paths: list[Path], direction: str = "AB"
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Translate multiple images, returning originals + translations.

        Args:
            image_paths: List of paths to 256x256 RGB images.
            direction: ``"AB"`` for lion->cheetah, ``"BA"`` for cheetah->lion.

        Returns:
            Tuple of ``(originals, translations)`` as lists of numpy arrays.
        """
        originals: list[np.ndarray] = []
        translations: list[np.ndarray] = []

        for p in image_paths:
            img_np = self._load_original_np(p)
            originals.append(img_np)
            translated = self.translate(p, direction)
            translations.append(translated)

        return originals, translations

    def _get_generator(self, direction: str) -> ResNetGenerator:
        if direction == "AB":
            if self._gen_ab is None:
                raise GeneratorNotLoadedError(
                    "gen_AB not loaded. Call load_generators() first."
                )
            return self._gen_ab
        elif direction == "BA":
            if self._gen_ba is None:
                raise GeneratorNotLoadedError(
                    "gen_BA not loaded. Call load_generators() first."
                )
            return self._gen_ba
        else:
            raise ValueError(f"Invalid direction: {direction!r}. Use 'AB' or 'BA'.")

    @staticmethod
    def _load_original_np(path: Path) -> np.ndarray:
        """Load an image as a [0, 1] numpy array for side-by-side display."""
        img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
        return img


# ── Visualisation ────────────────────────────────────────────────────────────


def display_translations(
    originals: list[np.ndarray],
    translated: list[np.ndarray],
    title: str,
) -> None:
    """Display original and translated images side by side.

    Args:
        originals: List of original images as ``(H, W, C)`` arrays.
        translated: List of translated (generated) images.
        title: Super-title for the figure.
    """
    n = min(len(originals), len(translated))
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    fig.suptitle(title, fontsize=14, y=1.02)

    for i in range(n):
        axes[i, 0].imshow(originals[i])
        axes[i, 0].set_title("Original", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(translated[i])
        axes[i, 1].set_title("Translated", fontsize=10)
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


# ── Self-check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    logging.basicConfig(level=logging.INFO)

    engine = CycleGANInference()
    engine.download_checkpoints()
    engine.load_generators()

    test_lion = list(Path("data/test/lion").glob("*.png"))
    test_cheetah = list(Path("data/test/cheetah").glob("*.png"))

    if test_lion:
        result = engine.translate(test_lion[0], "AB")
        assert result.shape == (256, 256, 3), (
            f"Expected (256,256,3), got {result.shape}"
        )
        print(f"Lion->Cheetah: {result.shape} OK")

    if test_cheetah:
        result = engine.translate(test_cheetah[0], "BA")
        assert result.shape == (256, 256, 3), (
            f"Expected (256,256,3), got {result.shape}"
        )
        print(f"Cheetah->Lion: {result.shape} OK")

    print("self-check OK")

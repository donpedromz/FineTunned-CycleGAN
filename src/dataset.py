"""Unpaired dataset loader for CycleGAN fine-tuning.

Loads two unaligned image directories (domain A and domain B) and yields
paired tensors for training. Applies standard CycleGAN augmentation:
resize to 286, random crop 256, horizontal flip, normalize to [-1, 1].

Usage:
    loader = get_dataloaders("data/train", batch_size=4)
    for real_A, real_B in loader:
        ...
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class DatasetError(Exception):
    """Raised when dataset directories are invalid."""


class UnpairedDataset(Dataset):
    """Dataset yielding (real_A, real_B) from two unaligned directories.

    Both domains are loaded independently; the smaller domain is cycled
    to match the larger domain's length.
    """

    def __init__(
        self,
        root: str | Path,
        subfolder_a: str = "lion",
        subfolder_b: str = "cheetah",
        img_size: int = 256,
        load_size: int = 286,
    ) -> None:
        self.root = Path(root)
        dir_a = self.root / subfolder_a
        dir_b = self.root / subfolder_b

        if not dir_a.is_dir():
            raise DatasetError(f"Domain A directory not found: {dir_a}")
        if not dir_b.is_dir():
            raise DatasetError(f"Domain B directory not found: {dir_b}")

        self.paths_a = sorted(
            p for p in dir_a.iterdir() if p.suffix.lower() in IMG_EXTENSIONS
        )
        self.paths_b = sorted(
            p for p in dir_b.iterdir() if p.suffix.lower() in IMG_EXTENSIONS
        )

        if not self.paths_a:
            raise DatasetError(f"No images found in {dir_a}")
        if not self.paths_b:
            raise DatasetError(f"No images found in {dir_b}")

        self.length = max(len(self.paths_a), len(self.paths_b))
        self.img_size = img_size
        self.load_size = load_size

        logger.info(
            "UnpairedDataset: %d A images, %d B images, epoch length %d",
            len(self.paths_a),
            len(self.paths_b),
            self.length,
        )

    def __len__(self) -> int:
        return self.length

    def _load_image(self, idx: int, paths: list[Path]) -> torch.Tensor:
        path = paths[idx % len(paths)]
        img = Image.open(path).convert("RGBA").convert("RGB")

        # resize > random crop > h-flip
        img = img.resize((self.load_size, self.load_size), Image.Resampling.BILINEAR)
        w, h = img.size
        th, tw = self.img_size, self.img_size
        top = random.randint(0, h - th)
        left = random.randint(0, w - tw)
        img = img.crop((left, top, left + tw, top + th))

        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        import numpy as np

        arr = (
            2.0
            * torch.from_numpy(np.array(img, dtype=np.float32)).permute(2, 0, 1)
            / 255.0
            - 1.0
        )
        return arr  # already (3, H, W)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._load_image(idx, self.paths_a), self._load_image(idx, self.paths_b)


def get_dataloaders(
    root: str | Path,
    batch_size: int = 1,
    num_workers: int = 0,
) -> DataLoader:
    """Factory: create a DataLoader for the unpaired dataset.

    Args:
        root: Directory containing domain subfolders.
        batch_size: Batch size.
        num_workers: DataLoader workers (0 for debug).

    Returns:
        DataLoader yielding (real_A, real_B) tensors.
    """
    dataset = UnpairedDataset(root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )


if __name__ == "__main__":
    import tempfile
    import os

    logging.basicConfig(level=logging.INFO)

    # Create dummy images for self-check
    with tempfile.TemporaryDirectory() as tmpdir:
        for sub in ("lion", "cheetah"):
            d = os.path.join(tmpdir, "train", sub)
            os.makedirs(d, exist_ok=True)
            for i in range(3):
                img = Image.fromarray(
                    __import__("numpy", fromlist=["random"]).random.randint(
                        0, 255, (300, 300, 3), dtype="uint8"
                    )
                )
                img.save(os.path.join(d, f"img_{i}.png"))

        ds = UnpairedDataset(os.path.join(tmpdir, "train"))
        a, b = ds[0]
        assert a.shape == (3, 256, 256), f"A shape: {a.shape}"
        assert b.shape == (3, 256, 256), f"B shape: {b.shape}"
        assert a.min() >= -1.0 and a.max() <= 1.0, f"A range: [{a.min()}, {a.max()}]"
        assert b.min() >= -1.0 and b.max() <= 1.0, f"B range: [{b.min()}, {b.max()}]"
        assert len(ds) == 3  # max of both dirs
        print(f"Dataset OK — {len(ds)} samples, shapes (3,256,256), range [-1,1]")

    print("All self-checks passed.")

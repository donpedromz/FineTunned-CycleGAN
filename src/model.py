"""
CycleGAN model architecture — ResNet-9 block generator.

Matches the junyanz/pytorch-CycleGAN generator architecture for loading
pre-trained horse2zebra checkpoints from HuggingFace.

Usage:
    model = ResNetGenerator.from_checkpoint("gen_AB.pth", device)
    out = model(torch.randn(1, 3, 256, 256))
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class ModelError(Exception):
    """Base exception for model-related errors."""


class CheckpointKeyError(ModelError):
    """Raised when checkpoint keys cannot be mapped to the model."""


class CheckpointNotFoundError(ModelError):
    """Raised when a checkpoint file does not exist."""


def _build_key_map() -> dict[str, str]:
    """Build the complete checkpoint-to-model key mapping."""
    mapping: dict[str, str] = {
        "upfeature.conv.weight": "enc1.1.weight",
        "upfeature.conv.bias": "enc1.1.bias",
        "encoder1.conv1.weight": "enc2.0.weight",
        "encoder1.conv1.bias": "enc2.0.bias",
        "encoder2.conv1.weight": "enc3.0.weight",
        "encoder2.conv1.bias": "enc3.0.bias",
        "decoder1.conv1.weight": "dec1.0.weight",
        "decoder1.conv1.bias": "dec1.0.bias",
        "decoder2.conv1.weight": "dec2.0.weight",
        "decoder2.conv1.bias": "dec2.0.bias",
        "downfeature.conv.weight": "dec3.1.weight",
        "downfeature.conv.bias": "dec3.1.bias",
    }
    for i in range(9):
        mapping[f"res{i}.conv1.weight"] = f"res_blocks.{i}.block.1.weight"
        mapping[f"res{i}.conv1.bias"] = f"res_blocks.{i}.block.1.bias"
        mapping[f"res{i}.conv2.weight"] = f"res_blocks.{i}.block.5.weight"
        mapping[f"res{i}.conv2.bias"] = f"res_blocks.{i}.block.5.bias"
    return mapping


class CheckpointMapper:
    """Maps checkpoint key names to ``ResNetGenerator.state_dict()`` keys.

    Encapsulates all key-remapping logic in one place, making it reusable
    across different checkpoint sources without modifying the model class.
    Follows the Single Responsibility Principle.
    """

    CKPT_KEY_MAP: dict[str, str] = _build_key_map()

    @classmethod
    def remap(cls, ckpt: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Rename checkpoint keys to match ``ResNetGenerator.state_dict()``.

        Args:
            ckpt: Raw checkpoint state dict.

        Returns:
            Remapped state dict with recognisable keys.

        Raises:
            CheckpointKeyError: If no keys match after remapping.
        """
        remapped: dict[str, torch.Tensor] = {}
        unmatched: list[str] = []

        for ckpt_key, tensor in ckpt.items():
            if ckpt_key in cls.CKPT_KEY_MAP:
                remapped[cls.CKPT_KEY_MAP[ckpt_key]] = tensor
            else:
                unmatched.append(ckpt_key)

        if unmatched:
            logger.warning(
                "Unrecognised checkpoint keys (%d): %s", len(unmatched), unmatched[:5]
            )

        if not remapped:
            raise CheckpointKeyError(
                f"No checkpoint keys could be mapped. "
                f"Sample keys: {list(ckpt.keys())[:3]}"
            )

        return remapped


class ResNetBlock(nn.Module):
    """Residual block with skip connection: Conv → IN → ReLU → Conv → IN."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=True),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=True),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """CycleGAN 9-block ResNet generator.

    Encoder: 3 × Conv → IN → ReLU (downsampling)
    Transformer: 9 × ResNetBlock
    Decoder: 3 × TransposeConv → IN → ReLU → Tanh (upsampling)
    """

    def __init__(self) -> None:
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(*[ResNetBlock(256) for _ in range(9)])

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=True,
            ),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, bias=True),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.res_blocks(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        return x.clamp(-1, 1)

    @staticmethod
    def from_checkpoint(path: str | Path, device: torch.device) -> ResNetGenerator:
        """Factory method: load pre-trained weights and return model instance.

        Handles ``module.`` DataParallel prefix stripping and HuggingFace
        key-remapping via :class:`CheckpointMapper`.

        Args:
            path: Path to the ``.pth`` checkpoint file.
            device: Target device (cuda or cpu).

        Returns:
            ``ResNetGenerator`` on the target device with loaded weights.

        Raises:
            CheckpointNotFoundError: If the checkpoint file does not exist.
            CheckpointKeyError: If checkpoint keys cannot be mapped to the model.
        """
        ckpt_path = Path(path)

        if not ckpt_path.exists():
            raise CheckpointNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model = ResNetGenerator()
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        except Exception as exc:
            raise CheckpointKeyError(
                f"Checkpoint {ckpt_path} appears corrupted. "
                f"Delete the file and re-download."
            ) from exc

        if any(k.startswith("module.") for k in checkpoint):
            logger.info("Stripping 'module.' prefix from checkpoint keys …")
            checkpoint = {k.removeprefix("module."): v for k, v in checkpoint.items()}

        checkpoint = CheckpointMapper.remap(checkpoint)

        model_keys = list(model.state_dict().keys())
        logger.info("Model keys (first 5): %s …", model_keys[:5])

        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        if missing:
            logger.warning("Missing keys in checkpoint: %s", missing[:5])
        if unexpected:
            logger.warning("Unexpected keys in checkpoint: %s", unexpected[:5])

        return model.to(device)


# ── Self-check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model = ResNetGenerator()
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 3, 256, 256)
    out = model(x)

    assert out.shape == (1, 3, 256, 256), f"Expected (1,3,256,256), got {out.shape}"
    assert out.min() >= -1.0 and out.max() <= 1.0, "Output not in [-1, 1]"

    print(
        f"self-check OK — {n_params:,} params, output {tuple(out.shape)} in [{out.min():.2f}, {out.max():.2f}]"
    )

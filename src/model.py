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


class PatchGANDiscriminator(nn.Module):
    """70×70 PatchGAN discriminator (InstanceNorm, no sigmoid).

    Output is an (B, 1, N, N) patch map where each element classifies a
    70×70 receptive field as real/fake. Used with LSGAN (MSE) loss.
    """

    def __init__(self, input_nc: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),  # N/2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # N/4
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1),  # N/4
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),  # N/4 - 2
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),  # N/4 - 3
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


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

    def freeze_encoder(self) -> None:
        """Freeze encoder layers (enc1/enc2/enc3) for fine-tuning."""
        for name, param in self.named_parameters():
            if name.startswith(("enc1.", "enc2.", "enc3.")):
                param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only parameters with requires_grad=True."""
        return [p for p in self.parameters() if p.requires_grad]

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

    # Generator self-check
    model = ResNetGenerator()
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 3, 256, 256)
    out = model(x)

    assert out.shape == (1, 3, 256, 256), f"Expected (1,3,256,256), got {out.shape}"
    assert out.min() >= -1.0 and out.max() <= 1.0, "Output not in [-1, 1]"
    print(
        f"Generator OK — {n_params:,} params, output {tuple(out.shape)} "
        f"in [{out.min():.2f}, {out.max():.2f}]"
    )

    # Freeze/unfreeze check
    model.freeze_encoder()
    enc_frozen = sum(
        1
        for n, p in model.named_parameters()
        if n.startswith(("enc1.", "enc2.", "enc3.")) and not p.requires_grad
    )
    dec_trainable = sum(
        1
        for n, p in model.named_parameters()
        if not n.startswith(("enc1.", "enc2.", "enc3.")) and p.requires_grad
    )
    assert enc_frozen > 0, "Encoder params should be frozen"
    assert dec_trainable > 0, "Decoder/ResNet params should still be trainable"
    assert len(model.trainable_parameters()) == dec_trainable
    print(f"Freeze OK — {enc_frozen} encoder params frozen, {dec_trainable} trainable")

    model.unfreeze()
    assert all(p.requires_grad for p in model.parameters()), (
        "Unfreeze should restore all grads"
    )

    # Discriminator self-check
    disc = PatchGANDiscriminator()
    d_params = sum(p.numel() for p in disc.parameters())
    d_out = disc(x)
    assert d_out.shape[0] == 1 and d_out.shape[1] == 1, (
        f"Unexpected disc output: {d_out.shape}"
    )
    print(f"Discriminator OK — {d_params:,} params, output {tuple(d_out.shape)}")
    print("All self-checks passed.")

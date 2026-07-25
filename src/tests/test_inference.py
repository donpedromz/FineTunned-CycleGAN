import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.inference import CycleGANInference
from src.model import ResNetGenerator


def test_load_image():
    engine = CycleGANInference()
    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "test.png"
        Image.fromarray((np.random.rand(256, 256, 3) * 255).astype("uint8")).save(
            img_path
        )

        tensor = engine.load_image(img_path)
        assert tensor.shape == (1, 3, 256, 256)
        assert tensor.min() >= -1.0 and tensor.max() <= 1.0


def test_translate_with_manual_models():
    engine = CycleGANInference()
    device = torch.device("cpu")
    engine._gen_ab = ResNetGenerator().to(device).eval()
    engine._gen_ba = ResNetGenerator().to(device).eval()
    engine._device = device

    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "test.png"
        Image.fromarray((np.random.rand(256, 256, 3) * 255).astype("uint8")).save(
            img_path
        )

        result_ab = engine.translate(img_path, "AB")
        assert result_ab.shape == (256, 256, 3)
        assert result_ab.min() >= -1.0 and result_ab.max() <= 1.0

        result_ba = engine.translate(img_path, "BA")
        assert result_ba.shape == (256, 256, 3)


def _has_internet() -> bool:
    import socket

    try:
        socket.create_connection(("huggingface.co", 443), timeout=2)
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _has_internet(), reason="Requires internet for HF download")
def test_download_checkpoints():
    engine = CycleGANInference()
    engine.download_checkpoints()
    assert (Path("checkpoints") / "horse2zebra" / "gen_AB.pth").exists()
    assert (Path("checkpoints") / "horse2zebra" / "gen_BA.pth").exists()

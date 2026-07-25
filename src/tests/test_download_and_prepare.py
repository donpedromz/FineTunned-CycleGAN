import os
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from src.download_and_prepare import (
    ImageProcessingError,
    filter_classes,
    resize_and_export,
    split_data,
    validate_images,
)


def _dummy_image(path: str | Path, size: tuple[int, int] = (300, 300)) -> None:
    img = Image.new("RGB", size, (128, 128, 128))
    img.save(path)


def test_validate_images_keeps_correct_size():
    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "good.png"
        _dummy_image(good, (300, 300))

        valid = validate_images([good], "test")
        assert len(valid) == 1


def test_validate_images_rejects_wrong_size():
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.png"
        _dummy_image(bad, (200, 200))

        with pytest.warns(UserWarning, match="discarding"):
            with pytest.raises(ImageProcessingError):
                validate_images([bad], "test")


def test_validate_images_raises_on_empty():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            validate_images([], "test")
            assert False, "Should have raised"
        except ImageProcessingError:
            pass


def test_split_data_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i in range(10):
            p = Path(tmp) / f"img_{i}.png"
            _dummy_image(p)
            paths.append(p)

        train, test = split_data(paths, "test")
        assert len(train) == 9
        assert len(test) == 1


def test_split_data_raises_on_empty():
    try:
        split_data([], "test")
        assert False, "Should have raised"
    except ImageProcessingError:
        pass


def test_resize_and_export():
    with tempfile.TemporaryDirectory() as tmp:
        src_dir = Path(tmp) / "src"
        src_dir.mkdir()
        for i in range(3):
            _dummy_image(src_dir / f"img_{i}.png", (300, 300))

        count = resize_and_export(list(src_dir.iterdir()), "lion", "train", (256, 256))
        assert count == 3
        for i in range(3):
            out_path = Path("data") / "train" / "lion" / f"lion_{i:04d}.png"
            assert out_path.exists()
            img = Image.open(out_path)
            assert img.size == (256, 256)
            out_path.unlink()
        # cleanup
        import shutil

        shutil.rmtree(Path("data") / "train" / "lion", ignore_errors=True)


def test_filter_classes_missing_dir_raises():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            filter_classes(Path(tmp))
            assert False, "Should have raised"
        except Exception:
            pass

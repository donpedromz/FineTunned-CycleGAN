import os
import tempfile

import numpy as np
from PIL import Image

from src.dataset import UnpairedDataset


def test_unpaired_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        for sub in ("lion", "cheetah"):
            d = os.path.join(tmpdir, "train", sub)
            os.makedirs(d, exist_ok=True)
            for i in range(3):
                img = Image.fromarray(
                    np.random.randint(0, 255, (300, 300, 3), dtype="uint8")
                )
                img.save(os.path.join(d, f"img_{i}.png"))

        ds = UnpairedDataset(os.path.join(tmpdir, "train"))
        a, b = ds[0]

        assert a.shape == (3, 256, 256), f"A shape: {a.shape}"
        assert b.shape == (3, 256, 256), f"B shape: {b.shape}"
        assert a.min() >= -1.0 and a.max() <= 1.0, f"A range: [{a.min()}, {a.max()}]"
        assert b.min() >= -1.0 and b.max() <= 1.0, f"B range: [{b.min()}, {b.max()}]"
        assert len(ds) == 3

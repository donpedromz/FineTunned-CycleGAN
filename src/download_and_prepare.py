"""
Dataset preparation for CycleGAN fine-tuning.

Downloads Wildlife Animals Images from Kaggle, filters for lion/cheetah at 300×300,
validates actual dimensions, splits 90/10 train/test, and resizes to 256×256.

Usage:
    from src.download_and_prepare import prepare_dataset
    prepare_dataset()
"""

import os
import random
import warnings
from pathlib import Path

import kagglehub
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError


class DatasetPreparationError(Exception):
    """Base exception for dataset preparation errors."""


class KaggleAuthError(DatasetPreparationError):
    """Raised when Kaggle API token is missing or invalid."""


class DatasetStructureError(DatasetPreparationError):
    """Raised when expected dataset directories or files are missing."""


class ImageProcessingError(DatasetPreparationError):
    """Raised when image validation or transformation fails."""


# ── Auth ────────────────────────────────────────────────────────────────────


def check_kaggle_auth() -> None:
    """Verify Kaggle credentials are available.

    Checks, in priority order (mirrors kagglehub's discovery):
    1. ``KAGGLE_API_TOKEN`` environment variable (modern token auth)
    2. ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` environment variables (legacy)
    3. ``~/.kaggle/kaggle.json`` credentials file

    Raises KaggleAuthError if none are found.
    """
    load_dotenv()
    if "KAGGLE_API_TOKEN" in os.environ:
        return

    if "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ:
        return

    kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_path.exists():
        return

    token_path = Path.home() / ".kaggle" / "access_token"
    if token_path.exists():
        token = token_path.read_text().strip()
        if token:
            # Set it so kagglehub picks it up automatically
            os.environ["KAGGLE_API_TOKEN"] = token
            return

    raise KaggleAuthError(
        "Kaggle credentials not found.\n\n"
        "Option A — Set the environment variable (recommended):\n"
        "  export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)\n\n"
        "Option B — Download kaggle.json from:\n"
        "  https://www.kaggle.com/settings/account\n"
        "  and place it in ~/.kaggle/kaggle.json"
    )


# ── Download ────────────────────────────────────────────────────────────────


def download_dataset() -> Path:
    """Download the wildlife dataset. Returns the local path."""
    print("Downloading wildlife animals dataset via kagglehub...")
    try:
        path = kagglehub.dataset_download("anshulmehtakaggl/wildlife-animals-images")
    except Exception as exc:
        raise DatasetPreparationError(f"Failed to download dataset: {exc}") from exc
    result = Path(path)
    print(f"Dataset downloaded to: {result}")
    return result


def count_all_images(dataset_path: Path) -> int:
    """Count total image files across the entire dataset."""
    total = len(list(dataset_path.rglob("*.[jJ][pP][gG]")))
    total += len(list(dataset_path.rglob("*.[jJ][pP][eE][gG]")))
    total += len(list(dataset_path.rglob("*.[pP][nN][gG]")))
    print(f"Total image files found: {total}")
    return total


# ── Filter ──────────────────────────────────────────────────────────────────


def filter_classes(
    dataset_path: Path,
    resolution: str = "300",
    classes: tuple[str, str] = ("lion", "cheetah"),
) -> tuple[list[Path], list[Path]]:
    """Filter images by class and resolution.

    The dataset uses ``{class}-resize-{resolution}`` directory naming
    (e.g. ``lion-resize-300``, ``cheetah-resize-512``).

    Returns (class1_images, class2_images).
    Raises DatasetStructureError if expected directories are missing.
    """
    lion_dir = dataset_path / f"{classes[0]}-resize-{resolution}"
    cheetah_dir = dataset_path / f"{classes[1]}-resize-{resolution}"

    for name, d in [("lion", lion_dir), ("cheetah", cheetah_dir)]:
        if not d.is_dir():
            raise DatasetStructureError(
                f"Expected '{d.name}' directory not found: {d}\n"
                f"Dataset structure is: {{class}}-resize-{{resolution}}"
            )

    lion_raw = sorted(lion_dir.rglob("*.png"))
    cheetah_raw = sorted(cheetah_dir.rglob("*.png"))

    print(f"  Images in lion-resize-{resolution}:  {len(lion_raw)}")
    print(f"  Images in cheetah-resize-{resolution}: {len(cheetah_raw)}")
    print(f"  Total after class filter:             {len(lion_raw) + len(cheetah_raw)}")
    return lion_raw, cheetah_raw


# ── Validate ────────────────────────────────────────────────────────────────


def validate_images(paths: list[Path], class_name: str) -> list[Path]:
    """Keep only images that are exactly 300×300 pixels.

    Corrupted or incorrectly-sized images are reported via warnings and
    excluded. Raises ImageProcessingError if no valid images remain.
    """
    valid: list[Path] = []
    for p in paths:
        try:
            with Image.open(p) as img:
                if img.size == (300, 300):
                    valid.append(p)
                else:
                    warnings.warn(
                        f"{class_name}/{p.name}: expected 300×300, got "
                        f"{img.size} — discarding"
                    )
        except (OSError, UnidentifiedImageError) as exc:
            warnings.warn(f"{class_name}/{p.name}: corrupted ({exc}) — skipping")

    if not valid:
        raise ImageProcessingError(
            f"No valid 300×300 images found for '{class_name}'. "
            f"All {len(paths)} candidate(s) were discarded."
        )

    return valid


# ── Split ───────────────────────────────────────────────────────────────────


def split_data(
    paths: list[Path], class_name: str, seed: int = 42, split: float = 0.9
) -> tuple[list[Path], list[Path]]:
    """Split images into train/test sets.

    Returns (train_paths, test_paths).
    Raises ImageProcessingError if there aren't enough images for both splits.
    """
    if len(paths) == 0:
        raise ImageProcessingError(f"Cannot split '{class_name}': 0 images.")

    if len(paths) < 2:
        raise ImageProcessingError(
            f"Cannot split '{class_name}' into train/test: only {len(paths)} "
            f"image(s). Need at least 2."
        )

    random.seed(seed)
    shuffled = paths[:]
    random.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * split))
    train_set = shuffled[:split_idx]
    test_set = shuffled[split_idx:]

    print(
        f"  {class_name:>8}: {len(train_set):>3} train, {len(test_set):>3} test "
        f"(total {len(shuffled)})"
    )

    if len(shuffled) < 5:
        print(
            f"    NOTE: Only {len(shuffled)} images — split is functional "
            f"but uneven. Train={len(train_set)}, Test={len(test_set)}"
        )

    return train_set, test_set


# ── Export ──────────────────────────────────────────────────────────────────


def resize_and_export(
    paths: list[Path],
    class_name: str,
    split_name: str,
    size: tuple[int, int] = (256, 256),
) -> int:
    """Resize images and export as PNG.

    Returns the count of successfully exported images.
    """
    out_dir = Path("data") / split_name / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    errors = 0

    for i, src_path in enumerate(paths):
        try:
            with Image.open(src_path) as img:
                resized = img.resize(size, Image.LANCZOS)
                out_path = out_dir / f"{class_name}_{i:04d}.png"
                resized.save(out_path, "PNG")
                count += 1
        except Exception as exc:
            warnings.warn(f"Failed to resize/export {src_path.name}: {exc}")
            errors += 1

    print(f"  {split_name}/{class_name}: {count} exported, {errors} errors → {out_dir}")
    return count


# ── Main pipeline ───────────────────────────────────────────────────────────


def prepare_dataset() -> dict:
    """Run the full dataset preparation pipeline.

    Returns a summary dict with counts per class and split.
    Callers should catch DatasetPreparationError for error handling.
    """
    check_kaggle_auth()

    dataset_path = download_dataset()
    total_images = count_all_images(dataset_path)

    print("\n--- Stage: Class & Resolution Filter ---")
    lion_raw, cheetah_raw = filter_classes(dataset_path)

    print("\n--- Stage: Dimension Validation ---")
    lion_valid = validate_images(lion_raw, "lion")
    cheetah_valid = validate_images(cheetah_raw, "cheetah")
    print(f"  Lion valid (300×300):    {len(lion_valid)}")
    print(f"  Cheetah valid (300×300): {len(cheetah_valid)}")
    print(f"  Total after validation:  {len(lion_valid) + len(cheetah_valid)}")

    print("\n--- Stage: Train/Test Split (90/10, seed=42) ---")
    lion_train, lion_test = split_data(lion_valid, "lion")
    cheetah_train, cheetah_test = split_data(cheetah_valid, "cheetah")

    print("\n--- Stage: Resize (256×256, LANCZOS) & Export ---")
    n_lion_train = resize_and_export(lion_train, "lion", "train")
    n_lion_test = resize_and_export(lion_test, "lion", "test")
    n_cheetah_train = resize_and_export(cheetah_train, "cheetah", "train")
    n_cheetah_test = resize_and_export(cheetah_test, "cheetah", "test")

    # ── Final Report ────────────────────────────────────────────────────────
    train_total = n_lion_train + n_cheetah_train
    test_total = n_lion_test + n_cheetah_test
    grand_total = train_total + test_total

    print("\n" + "=" * 58)
    print("  DATASET PREPARATION — COMPLETE")
    print("=" * 58)
    print(f"  {'Stage':<35} {'Count':<8}")
    print(f"  {'-' * 43}")
    print(f"  {'Total images downloaded':<35} {total_images:<8}")
    print(
        f"  {'After class filter (lion+cheetah)':<35} "
        f"{len(lion_raw) + len(cheetah_raw):<8}"
    )
    print(
        f"  {'After dimension validation (300×300)':<35} "
        f"{len(lion_valid) + len(cheetah_valid):<8}"
    )
    print()
    print(f"  {'Split':<12} {'Lion':<8} {'Cheetah':<8} {'Total':<8}")
    print(f"  {'-' * 36}")
    print(f"  {'Train':<12} {n_lion_train:<8} {n_cheetah_train:<8} {train_total:<8}")
    print(f"  {'Test':<12} {n_lion_test:<8} {n_cheetah_test:<8} {test_total:<8}")
    print(f"  {'-' * 36}")
    print(
        f"  {'Total':<12} {n_lion_train + n_lion_test:<8} "
        f"{n_cheetah_train + n_cheetah_test:<8} {grand_total:<8}"
    )
    print()
    print("  All images resized to 256×256 PNG via LANCZOS.")
    print(f"  Output: data/train/{'{lion,cheetah}'}/, data/test/{'{lion,cheetah}'}/")
    print("=" * 58)

    return {
        "lion": {"train": n_lion_train, "test": n_lion_test},
        "cheetah": {"train": n_cheetah_train, "test": n_cheetah_test},
    }


if __name__ == "__main__":
    prepare_dataset()

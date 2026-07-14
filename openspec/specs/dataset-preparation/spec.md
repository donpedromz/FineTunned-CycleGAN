# Dataset Preparation Specification

## Purpose

Download the Wildlife Animals Images Kaggle dataset, filter for lion and cheetah at 300×300px, validate dimensions, split 90/10 train/test, resize to 256×256px, and visualize samples — producing a clean dataset ready for CycleGAN training.

## Requirements

### Requirement: Kaggle Download

The system MUST download the full dataset via `kagglehub.dataset_download('anshulmehtakaggl/wildlife-animals-images')`. It MUST check for a valid Kaggle API token at `~/.kaggle/kaggle.json` before download.

#### Scenario: Successful download
- GIVEN a valid Kaggle API token and internet connectivity
- WHEN the download function is called
- THEN the full dataset is downloaded and a local path is returned

#### Scenario: Missing API token
- GIVEN no Kaggle API token at `~/.kaggle/kaggle.json`
- WHEN the download function is called
- THEN the system MUST raise a clear error instructing the user to create a token at kaggle.com

### Requirement: Class Filtering

The system MUST filter the downloaded dataset to retain only `lion` and `cheetah` images from the `300x300` resolution folder, discarding all other classes and resolutions.

#### Scenario: Happy path
- GIVEN a downloaded dataset with 6 animal classes across 3 resolutions
- WHEN the class filter is applied
- THEN only `300x300/lion/` and `300x300/cheetah/` images are retained

### Requirement: Dimension Validation

The system MUST verify that every retained image has exactly 300×300 pixel dimensions. Images deviating from this MUST be discarded with a warning logged.

#### Scenario: All images valid
- GIVEN 200 lion and 180 cheetah images all exactly 300×300px
- WHEN dimension validation runs
- THEN all images pass validation and proceed unchanged

#### Scenario: Deviant dimensions found
- GIVEN an image that is 299×300 or 301×301
- WHEN dimension validation runs
- THEN the deviant image is discarded and a warning is logged with its filename

### Requirement: Train/Test Split

The system MUST split each class independently into 90% training and 10% testing using a deterministic random seed.

#### Scenario: Happy path
- GIVEN 100 lion images and 100 cheetah images
- WHEN the split function is applied with `random_seed=42`
- THEN 90 lion and 90 cheetah go to training, 10 lion and 10 cheetah go to testing

#### Scenario: Repeatable split
- GIVEN the same set of images
- WHEN the split runs twice with the same seed
- THEN the same images appear in train and test on both runs

### Requirement: Resize

The system MUST resize all images to exactly 256×256 pixels using Pillow with `LANCZOS` resampling and save as PNG.

#### Scenario: Happy path
- GIVEN a valid 300×300 RGB image
- WHEN resize is applied
- THEN the output is exactly 256×256 PNG with identical aspect ratio preserved

### Requirement: Output Structure

The system MUST write resized images to `data/train/{lion,cheetah}/` and `data/test/{lion,cheetah}/` directories as PNG files.

#### Scenario: Happy path
- GIVEN 90 train lion images exist
- WHEN output is written
- THEN `data/train/lion/` contains 90 PNG files named sequentially

### Requirement: Reporting

The system MUST log image counts at each stage: downloaded total, after class filter, after dimension validation, after train/test split (per class/split).

#### Scenario: Happy path
- GIVEN the full pipeline runs
- WHEN all stages complete
- THEN a report displays counts at each stage including the final per-class per-split totals

### Requirement: Notebook Visualization

The notebook MUST display at least 4 sample images per class per split with shape, pixel range, and source metadata. It MUST confirm total counts per split/class and report final image dimensions.

#### Scenario: Happy path
- GIVEN the preparation script completed successfully
- WHEN the notebook visualization cell runs
- THEN ≥4 images per class per split are displayed with metadata below each image

#### Scenario: Empty split
- GIVEN a class has zero images in a split after preparation
- WHEN the visualization cell runs
- THEN a message states "No images available for {class}/{split}" without crashing

### Requirement: Error Handling

The system MUST handle edge cases gracefully: empty class directories, very few images (less than 5 total across both classes), and corrupted image files.

#### Scenario: Corrupted image
- GIVEN an image file that cannot be opened by Pillow
- WHEN the system attempts to load it
- THEN the image is skipped, a warning is logged, and the pipeline continues without crashing

#### Scenario: Very few images
- GIVEN only 3 total lion and cheetah images after filtering
- WHEN the split function runs
- THEN the system MUST still produce a valid train/test split (e.g., 2 train + 1 test) or raise a clear actionable error if 90/10 is impossible

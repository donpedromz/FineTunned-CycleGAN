# Proposal: Dataset Notebook — First Iteration

## Intent

Create a data-preparation script and a root-level notebook that downloads the Kaggle wildlife dataset, filters for lion and cheetah images at 300×300px, validates actual image dimensions, splits 90/10 train/test, resizes to 256×256px, and visualizes sample images to confirm the dataset is ready. This produces a clean dataset ready for CycleGAN fine-tuning on lion↔cheetah translation in a future iteration.

## Scope

### In Scope
- Download full dataset via kagglehub
- Filter: only lion and cheetah from 300×300 folder
- Validate: keep only images exactly 300×300px
- Split: 90% training, 10% testing
- Resize: all to 256×256px via Pillow
- Output: `data/train/{lion,cheetah}/`, `data/test/{lion,cheetah}/` as PNG
- Root `.ipynb` notebook that imports and runs the preparation script
- Visualization: sample images from each split showing size/origin metadata
- Basic confirmation: report image counts, shapes, split ratios
- Add `pillow` and `matplotlib` dependencies to pyproject.toml
- Kaggle auth check with clear failure message

### Out of Scope
- CycleGAN fine-tuning (future iteration)
- Data augmentation
- Any model/training code
- Cloud/Kaggle deployment
- Other animal classes (fox, hyena, tiger, wolf)

## Capabilities

### New Capabilities
- `dataset-preparation`: downloading, filtering, splitting, and resizing wildlife images for CycleGAN training

### Modified Capabilities
None

## Approach

Two deliverables:
1. **`src/download_and_prepare.py`** — Dataset preparation script with `# %%` cell markers. Sequential cells: (1) install deps, (2) download dataset via kagglehub, (3) filter for `300x300/{lion,cheetah}/`, (4) validate actual image dimensions, (5) 90/10 random split, (6) resize to 256×256 and export as PNG. Kaggle auth checked early with graceful error on missing token. Report image counts at each stage.
2. **Root `.ipynb` notebook** — Imports and executes the preparation script, then displays sample images from each split with shape/size metadata, confirms counts and split ratios, and saves the prepared dataset summary.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/download_and_prepare.py` | New | Dataset preparation script with `# %%` cells |
| `src/__init__.py` | New | Package init for `src` |
| `fine_tuning_experiment.ipynb` | New | Root notebook that runs preparation and visualizes results |
| `pyproject.toml` | Modified | Add `pillow` and `matplotlib` dependencies |
| `data/` | New | Output directory (add to .gitignore) |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Kaggle API token missing | Medium | Early check with clear error and instructions |
| Image count unknown | Low | Report counts, handle zero/too-few gracefully |
| Full download ~1GB+ | Medium | One-time cost, report progress via kagglehub |

## Rollback

Delete `src/download_and_prepare.py`, `src/__init__.py`, and root `.ipynb` notebook. Revert `pyproject.toml`. Delete `data/` directory.

## Dependencies

- `kagglehub>=1.0.2` (already installed)
- `pillow` (to be added)
- `matplotlib` (to be added)
- `ipykernel` (to be added, for .ipynb support)
- Kaggle API token at `~/.kaggle/kaggle.json`

## Success Criteria

- [ ] Script runs end-to-end without errors
- [ ] Notebook runs without errors and displays sample images from each split
- [ ] Output: `data/train/lion/`, `data/train/cheetah/`, `data/test/lion/`, `data/test/cheetah/`
- [ ] All output images are 256×256 PNGs
- [ ] Train/test split is approximately 90/10
- [ ] Only images that were exactly 300×300px survive filtering
- [ ] Visualization shows at least 4 sample images per class with correct dimensions

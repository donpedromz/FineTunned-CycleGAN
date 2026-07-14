# Tasks: Dataset Notebook — First Iteration

## Review Workload Forecast

| Metric | Value |
|--------|-------|
| Estimated lines changed | ~100–120 |
| Budget risk | Low — well under 400-line threshold |
| Chained PRs recommended | No — single PR |
| Delivery strategy | `ask-on-risk` |
| Decision needed | Approve task breakdown, then implement in order |

No structural changes to existing code — only new files and a trivial dependency bump. Review effort is minimal.

---

## Phase 1: Foundation

**Focus**: Dependencies, directory scaffold, gitignore

| # | Task | File(s) | What to do |
|---|------|---------|------------|
| [x] 1.1 | Add dependencies | `pyproject.toml` | Add `pillow`, `matplotlib`, `ipykernel` to `[project] dependencies` list. Run `uv lock` to update lockfile. |
| [x] 1.2 | Create package init | `src/__init__.py` | Create empty init so `src` is a Python package (importable). |
| [x] 1.3 | Gitignore data dir | `.gitignore` | Append `data/` to `.gitignore` so generated dataset outputs are never tracked. |

---

## Phase 2: Core Script

**Focus**: `src/download_and_prepare.py` with `# %%` cell markers covering the full pipeline

| # | Task | File(s) | What to do |
|---|------|---------|------------|
| [x] 2.1 | Kaggle auth guard | `src/download_and_prepare.py` | Before download, check `~/.kaggle/kaggle.json` exists. If missing, raise a clear error with instructions to create a token at kaggle.com. |
| [x] 2.2 | Download via kagglehub | `src/download_and_prepare.py` | Call `kagglehub.dataset_download('anshulmehtakaggl/wildlife-animals-images')` and store the returned local path. |
| [x] 2.3 | Filter lion/cheetah from 300×300 | `src/download_and_prepare.py` | Traverse the downloaded dataset tree. Retain only files under `300x300/lion/` and `300x300/cheetah/`. Discard all other classes and resolutions. |
| [x] 2.4 | Validate image dimensions | `src/download_and_prepare.py` | Open each candidate image with Pillow. Keep only those that are exactly 300×300 pixels. Log a warning (with filename) for any that deviate. |
| [x] 2.5 | Corrupted-image resilience | `src/download_and_prepare.py` | Wrap each Pillow `open()` in try/except. On `OSError`/`UnidentifiedImageError`, log a warning with filename and skip (don't crash). |
| [x] 2.6 | Train/test split | `src/download_and_prepare.py` | Per class, shuffle with `random.seed(42)` then split 90/10. Handle tiny edge cases (e.g., <5 images: split what exists; 0 images: clear warning). |
| [x] 2.7 | Resize and export | `src/download_and_prepare.py` | Resize each image to 256×256 using Pillow `LANCZOS`. Save as PNG to `data/train/{lion,cheetah}/` and `data/test/{lion,cheetah}/`. Create output dirs with `os.makedirs(exist_ok=True)`. |
| [x] 2.8 | Stage-reporting printout | `src/download_and_prepare.py` | At each stage (downloaded, after filter, after validation, after split) print a human-readable count summary. Final line shows per-class per-split counts. |
| [x] 2.9 | Cell markers | `src/download_and_prepare.py` | Insert `# %%` markers to split the script into logical VS Code Interactive cells: imports, auth, download, filter, validate, split, resize, report. |

---

## Phase 3: Notebook

**Focus**: `fine_tuning_experiment.ipynb` that consumes the script and visualises results

| # | Task | File(s) | What to do |
|---|------|---------|------------|
| [x] 3.1 | Notebook scaffold and script runner | `fine_tuning_experiment.ipynb` | First cell: import and execute `download_and_prepare.py` (via `import src.download_and_prepare` or `%run`). Display the script's console output inline. |
| [x] 3.2 | Sample-image grid (≥4 per class/split) | `fine_tuning_experiment.ipynb` | Using matplotlib, display a grid of ≥4 images per class per split (lion/train, lion/test, cheetah/train, cheetah/test). Show shape, pixel range (min/max), and class/split label below each image. |
| [x] 3.3 | Empty-split guard | `fine_tuning_experiment.ipynb` | Before rendering each class/split subplot, check if images exist. If zero, display "No images available for {class}/{split}" in the cell output without crashing. |
| [x] 3.4 | Final confirmation summary | `fine_tuning_experiment.ipynb` | Final cell: print total count per split per class, total images overall, and confirm-all dimensions are 256×256. |

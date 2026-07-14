# Exploration: dataset-notebook-first-iteration

## 1. Current Project State

### Tech Stack
- **Python**: 3.12 (`.python-version`)
- **Package manager**: `uv` (v0.11.16) with lockfile
- **Dependencies**: `kagglehub>=1.0.2` (v1.0.2 installed in `.venv`)
- **No other deps**: No Pillow, no torch/torchvision, no OpenCV, no jupyter/notebook
- **Testing**: None configured
- **CI**: None

### Files
- `main.py` — skeleton (prints "Hello")
- `pyproject.toml` — single dep kagglehub
- `openspec/` — SDD config exists, no specs or changes yet
- `.gitignore` — basic Python ignores
- `README.md` — minimal
- **No `.ipynb` files, no `notebooks/` directory, no Jupyter config anywhere**

### Authentication
- `kagglehub` is installed and ready; requires a Kaggle API token (`~/.kaggle/kaggle.json`) for downloads. The notebook will need to instruct the user on this setup.

---

## 2. Dataset Structure Findings

**Dataset**: Wildlife Animals Images by anshulmehtakaggl
**URL**: https://www.kaggle.com/datasets/anshulmehtakaggl/wildlife-animals-images

Based on research (Medium articles referencing this dataset):

### Classes (6 total)
1. cheetah
2. fox
3. hyena
4. lion
5. tiger
6. wolf

### Resolutions available
- 224×224
- **300×300** (target size)
- 512×512

### Inferred folder structure
```
wildlife-animals-images/
├── 300x300/
│   ├── cheetah/
│   ├── fox/
│   ├── hyena/
│   ├── lion/
│   ├── tiger/
│   └── wolf/
├── 512x512/
│   └── (same classes)
└── 224x224/
    └── (same classes)
```

### KaggleHub download
```python
import kagglehub
path = kagglehub.dataset_download('anshulmehtakaggl/wildlife-animals-images')
```
- Downloads full dataset to `~/.cache/kagglehub/datasets/anshulmehtakaggl/wildlife-animals-images/versions/<n>`
- Returns the local path string
- Supports `path=` for single-file download (but NOT folder-level filtering — confirmed open GitHub issue #285)

---

## 3. Option Analysis

### 3.1 Notebook Tooling

| Option | Pros | Cons |
|--------|------|------|
| **Raw Jupyter (.ipynb)** | Standard, shareable, visual outputs, markdown interleaved | Adds jupyter/notebook dep, heavier, Git-unfriendly diffs |
| **VS Code Jupyter** | Built-in with Python extension, no extra install, native .ipynb support | VS Code-specific, same Git-unfriendly format |
| **Jupyter Lab** | Full IDE experience, good for exploration | Heavy dependency tree, overkill for a data-prep notebook |
| **papermill + .ipynb** | Parameterized execution, good for pipelines | Overkill for a single exploration notebook |
| **.py scripts** | Lightest, git-friendly diffs, no extra deps | No visual outputs, no inline charts/markdown, harder to explore iteratively |
| **VS Code Interactive (.py with # %%)** | Lightweight (no jupyter install), inline outputs, git-friendly .py files | VS Code-specific cells |

### 3.2 Image Processing

| Option | Pros | Cons |
|--------|------|------|
| **Pillow (PIL)** | Lightweight, std for image I/O, resize via `Image.resize()` | No tensor integration, manual batching |
| **OpenCV (cv2)** | Fast, resize via `cv2.resize()`, good array support | Heavy dep, complex install, overkill for resize-only |
| **torchvision.transforms** | Direct tensor output, composable transforms, `Resize(256)` one-liner | Requires torch/torchvision (heavy), pulls in CUDA deps |
| **Pillow + numpy** | Lightweight, resize in PIL → convert to numpy array | Two deps, slightly more code |
| **scikit-image** | Good for scientific image processing | Overkill, slow for batch resize |

### 3.3 Output Directory Structure

**Recommended structure** (standard ImageFolder format for torchvision / keras):
```
data/
├── train/
│   ├── lion/
│   ├── cheetah/
├── test/
│   ├── lion/
│   ├── cheetah/
```

Alternative (flat with CSV metadata):
```
data/
├── images/
│   ├── lion_001.jpg
│   ├── cheetah_001.jpg
├── metadata.csv
```

**Recommendation**: ImageFolder structure — directly compatible with `torchvision.datasets.ImageFolder` and `keras.utils.image_dataset_from_directory`, no metadata file needed.

---

## 4. Key Constraints & Risks

### kagglehub limitation (risk)
**kagglehub does not support folder-level download** (GitHub issue #285). The dataset is organized by resolution (300x300/) then by class. The notebook must:
1. Download the full dataset
2. Filter locally for `300x300/` directory
3. Filter for only `lion/` and `cheetah/` directories

### Image count unknown (risk)
Not verified — the number of lion and cheetah images at 300×300 is unknown. Need to handle edge case of empty directories or insufficient images for 90/10 split.

### kaggle authentication (constraint)
kagglehub needs API token at `~/.kaggle/kaggle.json`. Notebook must document this. The notebook will fail gracefully if unauthenticated.

### No ML framework yet (opportunity)
This notebook is the right place to add the first ML dependencies. Pillow alone is enough for this step; torchvision can wait for the CycleGAN model.

---

## 5. Recommendations

### Notebook approach
**VS Code Interactive Python (.py with `# %%` cells)**
- Zero new dependencies (no jupyter install)
- Git-friendly diffs (plain .py, not JSON .ipynb)
- Inline image output via `IPython.display`
- Easy transition to .ipynb later if needed

### Image processing
**Pillow** (`PIL`) — add `pillow` to pyproject.toml
- Lightest option that does the job
- `Image.open()` + `Image.resize((256, 256))` + `Image.save()`
- Can report original size before resize

### Output format
**ImageFolder structure**: `data/train/{lion,cheetah}/`, `data/test/{lion,cheetah}/`

---

## 6. Affected Areas

| File | Action |
|------|--------|
| `notebooks/download_and_prepare.py` | **Create** — main data preparation script with `# %%` cells |
| `pyproject.toml` | **Update** — add `pillow` dependency |
| `uv.lock` | **Auto-update** via `uv lock` |
| `data/` | **Create** — output directory (gitignored) |
| `.gitignore` | **Optionally update** — add `data/` |

---

## 7. Ready for Proposal?

**Yes** — the scope is well-defined, risks are understood, and the approach is clear. The exploration confirms:
- Dataset structure is predictable (resolution → class folders)
- kagglehub is already installed and functional
- Pillow is the right minimal dep for image operations
- VS Code Interactive cells give the best tradeoff for a first notebook
- The output ImageFolder structure is standard and future-proof for CycleGAN training

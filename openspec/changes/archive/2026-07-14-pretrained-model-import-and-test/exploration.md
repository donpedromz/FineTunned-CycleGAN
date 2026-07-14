# Exploration: Pretrained Model Import and Test (horse2zebra)

## Current State

### Project Structure
- **Python 3.12** with `uv` package manager (`.venv` at project root)
- **Dependencies installed**: `kagglehub`, `pillow`, `matplotlib`, `ipykernel`, `numpy`, `pandas`, `tqdm`
- **NO PyTorch, NO torchvision, NO ML framework installed yet**
- **Dataset**: 636 images (264 lion train, 307 cheetah train, 30 lion test, 35 cheetah test) as 256×256 PNGs in standard ImageFolder layout at `data/{train,test}/{lion,cheetah}/`
- **Existing notebook**: `fine_tuning_experiment.ipynb` — Jupyter notebook with dataset preparation cells already executed

### Model Architecture (CycleGAN — ResNet-based Generator)
The CycleGAN horse2zebra model uses a **ResNet-9 block generator** architecture:
- **Generator**: Encoder (Conv → InstanceNorm → ReLU × 3) → Transformer (9× ResNet blocks) → Decoder (ConvTranspose → InstanceNorm → ReLU × 3 → Tanh). Input/output: 3 channels (RGB), 256×256 pixels.
- **Discriminator**: PatchGAN (70×70) with 5 Conv layers
- The official checkpoint only ships the **generator** (`latest_net_G.pth`), which is all that's needed for inference/testing

### Checkpoint Sources Found
Two viable sources for the pretrained horse2zebra checkpoint:

1. **Official junyanz repo**: Download via `bash scripts/download_cyclegan_model.sh horse2zebra` from `http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/horse2zebra.pth` — single `latest_net_G.pth` file (~45 MB)
2. **HuggingFace (johko/cyclegan-horse2zebra)**: Full set of 4 checkpoint files (gen_AB.pth, gen_BA.pth, disc_A.pth, disc_B.pth — 113 MB total). Uses the same state_dict format, compatible with junyanz's architecture.

---

## Affected Areas

| File | Action |
|------|--------|
| `pyproject.toml` | **Update** — add `torch`, `torchvision` dependencies |
| `uv.lock` | **Auto-update** — via `uv lock` |
| `checkpoints/horse2zebra/gen_AB.pth` | **Create** — downloaded generator checkpoint (A→B) |
| `checkpoints/horse2zebra/gen_BA.pth` | **Create** — downloaded generator checkpoint (B→A) |
| `checkpoints/horse2zebra/disc_A.pth` | **Create** — downloaded discriminator checkpoint (only if using HF source) |
| `checkpoints/horse2zebra/disc_B.pth` | **Create** — downloaded discriminator checkpoint (only if using HF source) |
| `src/model.py` | **Create** — CycleGAN model definition (ResNetGenerator + PatchGAN discriminator) matching the pretrained architecture |
| `src/inference.py` | **Create** — inference/visualization utility: load images, run through generator, display results |
| `fine_tuning_experiment.ipynb` | **Update** — add new cells for model loading and testing section |
| `openspec/changes/pretrained-model-import-and-test/` | **Create** — SDD change folder for this iteration |

---

## Approaches

### Approach A: HuggingFace Hub (`johko/cyclegan-horse2zebra`)

Import the model directly from HuggingFace using `torch.hub.load()` or manual download + `torch.load()`.

**Steps**:
1. Install `torch` + `torchvision` via uv
2. Write a `ResNetGenerator` model class matching junyanz's architecture (9 blocks, 64 base filters, instance norm)
3. Download checkpoint files from HF: `gen_AB.pth` (45.5 MB) + `gen_BA.pth` (45.5 MB) using `huggingface_hub` or direct HTTP
4. Load state dicts into the model
5. Write inference loop: load dataset images → normalize to [-1,1] → forward pass → denormalize → display/save
6. Add notebook cells for testing

**Pros**:
- Checkpoint is already split into generators and discriminators (no need to extract from a combined file)
- `huggingface_hub` is lightweight and well-documented
- Easy to extend: can swap to better checkpoints later
- ~113 MB total download (but only need the two generators = ~91 MB)

**Cons**:
- Another dependency beyond torch/torchvision (`huggingface_hub`)
- Need to write the model architecture code by hand (no standard CycleGAN in torchvision)
- Model binary format uses pickle (inherent security risk, but standard for PyTorch)
- Older checkpoint (over 2 years old)

**Effort**: Medium

---

### Approach B: Official junyanz PyTorch CycleGAN Repo

Clone the official repo and use its test script + download mechanism.

**Steps**:
1. Clone `https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix` as a dependency or submodule
2. Run `bash scripts/download_cyclegan_model.sh horse2zebra` to download `latest_net_G.pth` (~45 MB)
3. Adapt the repo's `test.py` or import its model definitions directly
4. Point it at our lion/cheetah dataset for testing

**Pros**:
- **No need to write model code** — the repo provides `models/networks.py` with exact ResNetGenerator + PatchDiscriminator
- Single authoritative source (berkeley.edu), same checkpoints used in the CycleGAN paper
- Script already handles G vs D split automatically during export
- Well-tested by thousands of users
- Only need the generator (~45 MB) for inference

**Cons**:
- Cloning the full repo is ~20 MB (but unnecessary — only need `models/` directory)
- Their test.py expects `--dataroot` with specific directory conventions (testA/testB)
- Mixing the repo's CLI tools with a notebook workflow is awkward
- The repo has its own `options/` and `data/` infra that conflicts with our project structure
- Would need to import specific modules rather than run their scripts directly
- ~3× the code to read and understand vs. writing a focused 80-line model class

**Effort**: Medium-High

---

## Recommendation

**Approach A (HuggingFace Hub)** is the better choice, with a twist: don't use `huggingface_hub` at all — just download the two generator `.pth` files via direct HTTP and load them manually.

**Why**:
- We only need generators for inference. The HF repo provides `gen_AB.pth` (horse→zebra) and `gen_BA.pth` (zebra→horse) as separate files — perfect for our bidirectional lion↔cheetah task.
- Writing a ~80-line `ResNetGenerator` class is **simpler and cleaner** than importing and adapting the entire junyanz codebase. The architecture is well-known: Conv → IN → ReLU blocks, 9× residual blocks, transpose conv → IN → ReLU decoder, final Tanh. This is ~1 hour of work.
- Direct HTTP download of the `.pth` files is trivial (no `huggingface_hub` dep needed):
  ```
  wget https://huggingface.co/johko/cyclegan-horse2zebra/resolve/main/gen_AB.pth
  ```
- The **real work** isn't the download — it's adding `torch`/`torchvision`, writing the model class, the data loader, the inference pipeline, and the notebook integration.

### Specific Implementation Plan
1. `uv add torch torchvision` — add PyTorch to the project
2. Create `src/model.py` with `ResNetGenerator` (matching the 9-block architecture)
3. Create `src/inference.py` with a `generate()` function: load image → normalize → forward → denormalize → save/display
4. Download checkpoints to `checkpoints/horse2zebra/{gen_AB,gen_BA}.pth`
5. Test on a handful of lion and cheetah images from `data/test/`
6. Add notebook cells that:
   - Load the model and checkpoints
   - Run inference on sample images
   - Display lion→cheetah and cheetah→lion translations side by side
   - Record observations about translation quality

---

## Risks

1. **GPU vs CPU**: PyTorch with CUDA is ~2 GB. On a CPU-only machine, inference on 256×256 images through a ResNet-9 generator will be **slow** (~1-5 seconds per image). The project should default to CPU and gracefully handle missing CUDA. For 10 test images this is acceptable; for the full 571 training images it's not.

2. **Large checkpoint downloads**: `gen_AB.pth` and `gen_BA.pth` are ~45 MB each (~91 MB total). This is manageable, but the download location and management should be documented. The `checkpoints/` directory should be gitignored.

3. **State dict key mismatches**: The HF checkpoint might use slightly different naming conventions than our model implementation. This is the most common source of errors when loading pretrained models. Mitigation: log state dict keys and our model's keys before loading.

4. **Pickle format**: PyTorch `.pth` files are pickle-based. While standard practice, the HF model has pickle imports (`torch.FloatStorage`, `collections.OrderedDict`) that might trigger security scanners. No real risk here — it's the standard PyTorch format — but worth noting.

5. **Project scope creep**: Adding PyTorch changes the dependency footprint significantly. From a lightweight data-prep project (~5 deps, ~200 MB venv) to a full ML project (~1.5-3 GB venv with CUDA). This is expected and necessary, but the user should be aware.

6. **No existing tests**: The project has no test infrastructure (SDD config confirms: `testing.layers.unit.available: false`). Testing model loading and inference will need to rely on notebook-based verification.

---

## Ready for Proposal

**Yes** — the exploration is complete. The approach is clear, risks are understood, and the scope is well-defined. Move to the Proposal phase to formalize the change intent, scope, and concrete tasks.

Key decision for the Proposal phase:
- **Checkpoint source**: HuggingFace `johko/cyclegan-horse2zebra` (recommended)
- **Download method**: Direct HTTP (`wget`), no `huggingface_hub` dependency needed
- **Model architecture**: Write focused `ResNetGenerator` class (not clone the full junyanz repo)
- **Inference**: Notebook cells with side-by-side visualization of lion↔cheetah translations

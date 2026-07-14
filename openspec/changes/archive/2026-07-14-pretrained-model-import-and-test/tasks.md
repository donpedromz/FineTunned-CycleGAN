# Tasks: Pretrained Model Import and Test

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~220–260 |
| 400-line budget risk | Low |
| Chained PRs recommended | No |
| Suggested split | Single PR |
| Delivery strategy | ask-on-risk |

Decision needed before apply: No
Chained PRs recommended: No
Chain strategy: pending
400-line budget risk: Low

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | Foundation + Model + Inference + Notebook | PR 1 (single PR) | All tasks are tightly coupled — single coherent PR under 400 lines |

## Phase 1: Foundation

- [x] **1.1 Add PyTorch dependencies** — Run `uv add torch torchvision` to add PyTorch with CUDA support. Modifies `pyproject.toml` and auto-updates `uv.lock`.

## Phase 2: Model Architecture

- [x] **2.1 Create `ResNetBlock` helper** — In `src/model.py`: `Conv2d→InstanceNorm→ReLU→Conv2d→InstanceNorm` with skip connection. Reflect padding on first conv.
- [x] **2.2 Create `ResNetGenerator`** — In `src/model.py`: encoder (3× Conv→IN→ReLU, c64→c128→c256), 9× `ResNetBlock`, decoder (3× TransposeConv→IN→ReLU → TanH, c256→c128→c64→c3).
- [x] **2.3 Add `load_state_dict_with_prefix`** — Log model + checkpoint keys before loading, strip `module.` prefix from checkpoint keys, load with `strict=False`, log mismatched keys at WARNING.
- [x] **2.4 Add model self-check** — In `__main__`: forward pass on `(1,3,256,256)` random tensor, assert output shape and values in `[-1,1]`.

## Phase 3: Inference Pipeline

- [x] **3.1 Create `get_device()` helper** — In `src/inference.py`: return `cuda:0` if available else `cpu` with logged warning.
- [x] **3.2 Create `download_checkpoint()`** — `urllib.request.urlretrieve` from HuggingFace `johko/cyclegan-horse2zebra`, skip if file exists at `checkpoints/horse2zebra/gen_{AB,BA}.pth`.
- [x] **3.3 Create `load_image()`** — `PIL.Image.open` → `transforms.ToTensor` → `Normalize(0.5, 0.5)` → `[1,3,256,256]` tensor.
- [x] **3.4 Create `denormalize()`** — unsqueeze to batch → `.to(device)` → forward → squeeze → denormalize `(out*0.5+0.5).clamp(0,1)` → permute to `(H,W,3)` → numpy float32.
- [x] **3.5 Add inference self-check** — In `__main__`: download checkpoints, smoke-test one image from `data/test/lion/`, verify output is `(256,256,3)` float32.

## Phase 4: Notebook Integration

- [x] **4.1 Add setup cells** — Import `ResNetGenerator`, `download_checkpoint`, `get_device`, `translate`. Download checkpoints, detect device, instantiate both generators.
- [x] **4.2 Add translation cells** — Loop test images from `data/test/{lion,cheetah}/` through both generators (lion→cheetah with gen_AB, cheetah→lion with gen_BA).
- [x] **4.3 Add visualization cells** — Matplotlib side-by-side grid: original | translated with `axis off` and direction captions. ≥4 per translation direction.

# Model Inference Specification

**Purpose**: Requirements for loading pre-trained CycleGAN generators (horse2zebra) and running inference on lion/cheetah to establish a pre-fine-tuning baseline.

## Requirements

### GPU-Aware Device Detection

The system MUST detect CUDA via `torch.cuda.is_available()`. If available, models and tensors MUST be moved to GPU. Otherwise, MUST fall back to CPU with a logged warning.

| Scenario | GIVEN | WHEN | THEN |
|----------|-------|------|------|
| GPU available | `torch.cuda.is_available()` returns True | pipeline initializes | model and inputs move to `cuda:0`, inference runs on GPU |
| CPU fallback | `torch.cuda.is_available()` returns False | pipeline initializes | warning logged, inference completes on CPU without error |

### ResNetGenerator Architecture

The system MUST implement a `ResNetGenerator` with 9 ResNet blocks matching junyanz/pytorch-CycleGAN: encoder (3× Conv→IN→ReLU), 9× ResBlock (Conv→IN→ReLU→Conv→IN, skip), decoder (3× TransposeConv→IN→ReLU → Tanh). Input/output SHALL be 3-channel 256×256 normalized to [-1, 1].

| Scenario | GIVEN | WHEN | THEN |
|----------|-------|------|------|
| Correct state dict | `ResNetGenerator()` on selected device | `state_dict().keys()` inspected | keys match CycleGAN structure (e.g., `model.0.weight`) and forward pass on `(1,3,256,256)` produces `(1,3,256,256)` in [-1,1] |

### Checkpoint Download

The system MUST download `gen_AB.pth` and `gen_BA.pth` from `johko/cyclegan-horse2zebra` (HuggingFace, direct HTTP) into `checkpoints/horse2zebra/`. If a file exists, MUST skip re-download.

| Scenario | GIVEN | WHEN | THEN |
|----------|-------|------|------|
| First download | no `checkpoints/horse2zebra/` | download runs | both `.pth` files exist with non-zero size, `torch.load()` succeeds |
| Already cached | both files exist in `checkpoints/horse2zebra/` | download runs | no HTTP request, existing files used as-is |
| Corrupted checkpoint | `torch.load()` raises exception (truncated download, unexpected EOF) | pipeline attempts load | informative error with file path, user instructed to delete and re-download |

### State Dict Loading

Before loading, the system SHALL log both checkpoint and model keys. On mismatch, SHALL load with `strict=False`, log differences at WARNING, and load matching keys only.

| Scenario | GIVEN | WHEN | THEN |
|----------|-------|------|------|
| Exact match | checkpoint keys match model keys exactly | `load_state_dict(strict=True)` | no warning, weights update correctly |
| Partial mismatch | checkpoint keys partially differ | `load_state_dict(strict=False)` | mismatched keys logged, matching keys load, model produces valid output |

### Inference Pipeline

The pipeline MUST: load image (PIL), normalize to [-1,1], unsqueeze for batch dim, move to device, run forward, squeeze, denormalize to [0,1], return display-ready float32 HWC format.

| Scenario | GIVEN | WHEN | THEN |
|----------|-------|------|------|
| Single image | one 256x256 RGB PNG at `data/test/lion/` | passed through pipeline | output is 256x256 RGB float32 in [0,1], no batch dim errors |
| Empty test set | `data/test/lion/` or `data/test/cheetah/` has no PNGs | pipeline lists images | inference skipped, directory-empty message displayed |

### Side-by-Side Visualization

The system MUST display original and translated images side by side via matplotlib, with translation direction captions. At least 4 examples per direction (lion→cheetah, cheetah→lion) MUST be shown in the notebook.

| Scenario | GIVEN | WHEN | THEN |
|----------|-------|------|------|
| Normal display | list of original → translated pairs | notebook cell renders | each pair shown side by side with translation caption |

## Constraints

| Constraint | Value |
|-----------|-------|
| Input / output size | 256×256 pixels |
| Channels | RGB (3) |
| Normalization | [-1, 1] (model), [0, 1] (display) |
| Batch size | 1 |
| Output format | [0, 1] float32 RGB (HWC) |
| Framework | PyTorch + torchvision |
| Checkpoint source | HuggingFace `johko/cyclegan-horse2zebra` |
| Validation | Visual inspection only (no metrics) |

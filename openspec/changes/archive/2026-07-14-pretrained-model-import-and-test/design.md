# Design: Pre-trained Model Import and Test

## Technical Approach

Implement a `ResNetGenerator` matching the junyanz/pytorch-CycleGAN architecture (9-block variant), download pre-trained horse2zebra checkpoints from HuggingFace, and run inference on lion/cheetah test images. The pipeline is GPU-aware with graceful CPU fallback. Visualization lives in the existing notebook via matplotlib side-by-side grids.

## Architecture Decisions

### Decision: Model File Location

| Option | Tradeoff | Decision |
|--------|----------|----------|
| Inline in notebook | Quick but not importable | |
| `src/model.py` | Importable, follows existing `src/` pattern | **Chosen** |

**Rationale**: The existing codebase already uses `src/download_and_prepare.py` and imports from it in the notebook. `ResNetGenerator` is needed in both the notebook and future training scripts — keeping it importable avoids duplication.

### Decision: Checkpoint Download Strategy

| Option | Tradeoff | Decision |
|--------|----------|----------|
| `huggingface_hub` SDK | Extra dependency | |
| Direct HTTP via `requests` | Another dependency | |
| `urllib` from stdlib | Zero new deps, supports skip-if-exists | **Chosen** |

**Rationale**: No new dependency needed. `urllib.request.urlretrieve` with `os.path.exists()` guard covers the two download scenarios from the spec. Simpler than adding `requests` or the HF hub SDK for two fixed URLs.

### Decision: State Dict Loading

| Option | Tradeoff | Decision |
|--------|----------|----------|
| `strict=True` | Fails on any key mismatch | |
| `strict=False` + key logging | Resilient, debuggable | **Chosen** |

**Rationale**: Checkpoints may carry `module.` prefix from DataParallel training or extra keys. Logging both key sets before load and using `strict=False` with a warning matches the spec's "handle prefix mismatches" requirement. The `load_state_dict` helper strips `module.` from checkpoint keys when model keys lack it.

### Decision: Visualization

| Option | Tradeoff | Decision |
|--------|----------|----------|
| Save to file | Adds a step for the user | |
| Inline matplotlib in notebook | Immediate visual feedback | **Chosen** |

**Rationale**: The existing notebook already uses `matplotlib` for sample display. Reusing the same pattern (subplots, `imshow`, `axis off`) keeps the codebase consistent. Side-by-side pairs match the "original | translated" spec requirement.

## Data Flow

```
Image (256×256 PNG, [0,255]) 
  → PIL.Image.open(path)
  → transforms.ToTensor() ([0,1])
  → transforms.Normalize(0.5, 0.5) ([-1,1])
  → .unsqueeze(0).to(device) ([1,3,256,256])
  → ResNetGenerator.forward()
  → output.squeeze(0).cpu() ([3,256,256], [-1,1])
  → denormalize: (out * 0.5 + 0.5).clamp(0,1)
  → .permute(1,2,0).numpy() → display via matplotlib imshow
```

## Module Relationships

```
src/model.py             ResNetGenerator, ResNetBlock
       ↓ imports
src/inference.py         load_checkpoint(), load_image(), translate()
       ↓ imports
fine_tuning_experiment.ipynb  (calls translate, displays results)
       ↑
checkpoints/horse2zebra/gen_{AB,BA}.pth   (downloaded externally)
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/model.py` | Create | `ResNetGenerator` with 9 ResNet blocks, `ResNetBlock` helper, `load_state_dict` with `module.` prefix stripping |
| `src/inference.py` | Create | `download_checkpoint()`, `load_checkpoint()`, `load_image()`, `translate()` — full inference pipeline |
| `pyproject.toml` | Modify | Add `torch` + `torchvision` dependencies |
| `fine_tuning_experiment.ipynb` | Modify | Add cells for model loading, checkpoint download, 4+ side-by-side translations per direction |
| `checkpoints/horse2zebra/` | New | Downloaded `gen_AB.pth` + `gen_BA.pth` |

## Interfaces / Contracts

```python
# src/model.py
class ResNetGenerator(nn.Module):
    def __init__(self, n_res_blocks: int = 9): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def load_state_dict_with_prefix(self, state_dict: dict) -> None: ...

# src/inference.py
def download_checkpoint(url: str, dest: Path) -> None: ...
def load_image(path: Path | str) -> torch.Tensor: ...
def translate(model: nn.Module, image_tensor: torch.Tensor,
              device: torch.device) -> np.ndarray: ...
```

`torch.Tensor` shape contract: input `(1, 3, 256, 256)`, output `(1, 3, 256, 256)`, values in `[-1, 1]`.

## Testing Strategy

| Layer | What to Test | Approach |
|-------|-------------|----------|
| Unit | `ResNetGenerator` output shape + value range | Forward pass on random `(1,3,256,256)` tensor, assert output shape and `[-1,1]` range |
| Integration | Download → load → translate on real image | Smoke test: one lion image through pipeline, verify numpy output is `(256,256,3)` uint8 |
| E2E | Notebook cell execution | Manual: run all cells, inspect 8 side-by-side panels for visual plausibility |

No automated test framework exists (`testing.available: false` in config). Tests will be assert-based self-checks in `__main__` blocks.

## Migration / Rollout

No migration required. Checkpoints are downloaded on first inference run. If a checkpoint is corrupted, user deletes the file and re-runs.

## Open Questions

- None

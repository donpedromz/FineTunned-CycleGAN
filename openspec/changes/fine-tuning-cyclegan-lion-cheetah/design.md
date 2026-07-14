# Design: Fine-Tune CycleGAN for Lion↔Cheetah Translation

## Technical Approach

Frozen-encoder fine-tuning of pre-trained horse2zebra ResNet generators for lion↔cheetah translation. The encoder (enc1–enc3) preserves generic animal features from the base model; 9 ResNet blocks + decoder learn domain-specific mappings. New PatchGANDiscriminators (70×70) trained from scratch provide adversarial signal. All experiments tracked via a lightweight JSON-based ModelRegistry.

## Architecture Decisions

### Decision: PatchGANDiscriminator Architecture

| Option | Tradeoff | Decision |
|--------|----------|----------|
| 70×70 PatchGAN (standard) | Proven in CycleGAN papers, ~2.8M params | ✅ Selected |
| 1×1 PixelGAN | Too coarse, loses spatial context | Rejected |
| Multi-scale (Pix2PixHD) | Overkill for 256×256, more params | Rejected |

**Rationale**: Standard CycleGAN design. InstanceNorm (not BatchNorm) for style invariance. No sigmoid on output — LSGAN uses raw MSE against target labels 0/1. No spectral normalization needed for this dataset size.

### Decision: Freeze Mechanism

| Option | Tradeoff | Decision |
|--------|----------|----------|
| `freeze_encoder()` method + `trainable_parameters` property | Transparent to forward pass; optimizer picks up trainable params only | ✅ Selected |
| Wrapper class around generator | Extra indirection, complicates checkpoint loading | Rejected |
| Manual param filtering at train loop | Scattered logic, easy to forget | Rejected |

**Rationale**: Methods live on `ResNetGenerator` itself — no wrapper, no changes to `forward()`. `requires_grad=False` on enc1/enc2/enc3 is all PyTorch needs. `unfreeze()` for full fine-tuning fallback.

### Decision: Training Loop Structure

| Option | Tradeoff | Decision |
|--------|----------|----------|
| Single `train_cyclegan()` function | One entry point, easy to call from notebook | ✅ Selected |
| Per-component functions (train_G, train_D) | More modular but scattered state | Rejected |
| Callback/event system | Over-engineered for single experiment notebook | Rejected |

**Rationale**: One function, one return value (dict of per-epoch losses). Image pool (size 50, p=0.5 replace). Linear LR decay from 2e-4 → 0. Registry save every 10 epochs + final.

### Decision: Loss Functions

| Loss | Weight | Rationale |
|------|--------|-----------|
| LSGAN (MSE real/fake) | 1.0 | Standard CycleGAN, more stable than original GAN loss |
| Cycle consistency (L1) | 10.0 | Prevents content drift — A→B→A ≈ A |
| Identity (L1) | 0.5 | Preserves color when input already in target domain |

**Rationale**: Matches CycleGAN paper hyperparameters. L1 for cycle/identity (sharper than L2). No perceptual loss — added complexity for uncertain gain on this dataset.

### Decision: Model Registry Storage

| Option | Tradeoff | Decision |
|--------|----------|----------|
| JSON files in `checkpoints/experiments/` | Zero dependencies, git-trackable metadata, pandas for listing | ✅ Selected |
| SQLite | Adds dependency, overkill for <100 experiments | Rejected |
| External tracker (MLflow) | Not in scope, heavy dependency | Rejected |

**Rationale**: `run_id` = `{YYYYMMDD}-{NNN}` (auto-incrementing per day). Each run dir: `meta.json` + `gen_AB.pth` + `gen_BA.pth`. `list()` returns pandas DataFrame for notebook selection.

### Decision: Evaluation Pipeline

| Metric | Tool | Sample Size |
|--------|------|-------------|
| FID | clean-fid | 50 translations per direction |
| LPIPS | lpips (AlexNet) | Same 50 pairs |
| Visual grid | matplotlib | 4 random test images per direction |

**Rationale**: 50 samples balances fidelity vs. speed on RTX 3050. Same images for both metrics avoids redundant generation. Metrics saved to registry `meta.json` on evaluation call.

## Data Flow

```
data/train/{lion,cheetah}/
        │
        ▼
  UnpairedDataLoader ──→ (real_A, real_B) tensors
        │
        ▼
  ┌─────────────────────────────────────────┐
  │           train_cyclegan() loop         │
  │                                         │
  │  real_A ──→ G_A2B ──→ fake_B           │
  │  fake_B ──→ G_B2A ──→ rec_A   (cycle)  │
  │  real_B ──→ G_B2A ──→ fake_A           │
  │  fake_A ──→ G_A2B ──→ rec_B   (cycle)  │
  │  real_A ──→ G_A2B ──→ id_B     (identity)│
  │  real_B ──→ G_B2A ──→ id_A     (identity)│
  │                                         │
  │  D_A(real_A) + D_A(fake_A_pool) → D_A loss│
  │  D_B(real_B) + D_B(fake_B_pool) → D_B loss│
  │                                         │
  │  G_loss = adv + λ_cyc*(L_rec) + λ_id*(L_id)│
  └─────────┬───────────────────┬───────────┘
            │                   │
            ▼                   ▼
  ModelRegistry.save()   per-epoch loss dict
  (every 10 epochs)
            │
            ▼
  checkpoints/experiments/{run_id}/
      ├── meta.json
      ├── gen_AB.pth
      └── gen_BA.pth
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/model.py` | Modify | Add `PatchGANDiscriminator` class; add `freeze_encoder()`, `unfreeze()`, `trainable_parameters` to `ResNetGenerator` |
| `src/dataset.py` | Create | `UnpairedDataLoader` with random crop, resize→286→crop 256, hflip, [-1,1] norm |
| `src/training.py` | Create | `train_cyclegan()` — full training loop with G/D losses, image pool, LR schedule |
| `src/evaluation.py` | Create | `compute_fid()`, `compute_lpips()`, `visual_grid()` |
| `src/registry.py` | Create | `ModelRegistry` — save, list, load, load_best, compare |
| `pyproject.toml` | Modify | Add `lpips`, `clean-fid`, `scipy` |
| `fine_tuning_experiment.ipynb` | Modify | Training + evaluation + model selection cells |

## Interfaces / Contracts

```python
# src/model.py — PatchGANDiscriminator
class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_nc: int = 3) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...  # (B,1,N,N)

# src/model.py — ResNetGenerator additions
class ResNetGenerator(nn.Module):
    def freeze_encoder(self) -> None: ...     # enc1/2/3 requires_grad=False
    def unfreeze(self) -> None: ...            # all requires_grad=True
    def trainable_parameters(self) -> list[nn.Parameter]: ...  # filter requires_grad

# src/registry.py
class ModelRegistry:
    def __init__(self, base_dir: str | Path) -> None: ...
    def save(self, gen_ab: ResNetGenerator, gen_ba: ResNetGenerator,
             epoch: int, fid: float, lpips: float, cycle_loss: float,
             config: dict, notes: str = "") -> str: ...  # returns run_id
    def list(self) -> pd.DataFrame: ...
    def load(self, run_id: str) -> tuple[ResNetGenerator, ResNetGenerator, dict]: ...
    def load_best(self, metric: str, ascending: bool = True) -> tuple[...]: ...

# src/training.py
def train_cyclegan(
    gen_ab: ResNetGenerator, gen_ba: ResNetGenerator,
    d_a: PatchGANDiscriminator, d_b: PatchGANDiscriminator,
    dl_train: DataLoader, config: dict,
    registry: ModelRegistry, run_id: str,
    device: torch.device,
) -> dict[str, list[float]]: ...  # per-epoch loss curves
```

## Testing Strategy

| Layer | What to Test | Approach |
|-------|-------------|----------|
| Unit | PatchGANDiscriminator output shape | Forward pass with dummy tensor, assert `(1,1,N,N)` |
| Unit | freeze_encoder sets correct grads | Count frozen vs trainable params |
| Unit | ModelRegistry save/load roundtrip | Save → load → assert weights match |
| Unit | UnpairedDataLoader yields correct shapes | One batch, assert `(B,3,256,256)` in [-1,1] |
| Integration | Loss computation correctness | Known inputs → expected loss values |
| Integration | Registry + training integration | 2-epoch train → verify checkpoint exists |

Note: No testing infrastructure exists yet. Unit tests as `assert`-based `__main__` blocks per the existing codebase pattern.

## Threat Matrix

N/A — no routing, shell, subprocess, VCS/PR automation, executable-file classification, or process-integration boundary.

## Migration / Rollout

No migration required. New files are additive; `model.py` changes are backward-compatible (new methods don't alter existing behavior). Rollback: delete new files, revert model.py and pyproject.toml via git.

## Open Questions

None — all decisions are within standard CycleGAN territory for this dataset/hardware combo.

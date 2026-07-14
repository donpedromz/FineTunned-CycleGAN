# Proposal: Fine-Tune CycleGAN for Lion↔Cheetah Translation

## Intent

Adapt pre-trained horse2zebra CycleGAN generators to lion↔cheetah translation via frozen-encoder fine-tuning. The pre-trained generators already capture generic animal texture/shape features; freezing the encoder preserves these while trainable ResNet blocks + decoder learn domain-specific lion/cheetah mappings. A new PatchGANDiscriminator (70×70) trained from scratch provides adversarial signal for the new domain. A custom ModelRegistry tracks all experiment checkpoints with metadata for selection and comparison in the notebook.

## Scope

### In Scope
- `src/dataset.py`: unaligned DataLoader with random crop, resize→256, horizontal flip, [-1,1] normalization
- `src/training.py`: CycleGAN training loop (G_A2B, G_B2A, D_A, D_B) with LSGAN + cycle (λ=10) + identity (λ=0.5) losses, integrates ModelRegistry for checkpointing
- `src/evaluation.py`: FID (clean-fid), LPIPS (lpips), cycle loss curve, visual grid
- `src/model.py`: add `PatchGANDiscriminator` (70×70), add freeze/unfreeze helpers for encoder layers
- `src/registry.py`: custom ModelRegistry — save, list, load, compare checkpoints with metadata (JSON-based, no external dependencies)
- `pyproject.toml`: add lpips, clean-fid, scipy
- `fine_tuning_experiment.ipynb`: training + evaluation + model selection notebook cells

### Out of Scope
- Training generators from scratch
- Paired metrics (PSNR/SSIM), Inception Score
- Multi-GPU or mixed-precision
- Web UI or production deployment
- External experiment trackers (MLflow, Aim, Weights & Biases)

## Capabilities

### New Capabilities
- `dataset-loading`: Unaligned image-to-image DataLoader with augmentation for CycleGAN training
- `cyclegan-training`: Full CycleGAN training loop with frozen-encoder fine-tuning strategy
- `model-evaluation`: FID, LPIPS, and cycle loss metrics for unpaired image translation
- `model-registry`: Custom lightweight experiment registry for checkpoint management with metadata

### Modified Capabilities
- `model-inference`: Add PatchGANDiscriminator to model.py; add freeze/unfreeze mechanism to ResNetGenerator

## Approach

### Model Registry (`src/registry.py`)

Custom JSON-based ModelRegistry — no external dependencies. Each experiment run gets:
- A unique `run_id` (timestamp-based, e.g. `20260714-001`)
- A directory at `checkpoints/experiments/{run_id}/` containing:
  - `gen_AB.pth`, `gen_BA.pth` — generator checkpoints
  - `meta.json` — all metadata (hyperparams, metrics, approach, timestamp)

API:
```python
registry = ModelRegistry("checkpoints/experiments")

# Save a checkpoint with full metadata
registry.save(
    gen_ab=gen_ab, gen_ba=gen_ba,
    epoch=50, fid=45.2, lpips=0.32,
    config={"lr": 2e-4, "approach": "frozen_encoder", "lambda_cycle": 10}
)

# List all experiments as a pandas DataFrame
df = registry.list()  # run_id, epoch, fid, lpips, approach, lr, created_at

# Load best model by metric
model = registry.load_best(metric="fid", ascending=True)

# Load specific run
model = registry.load(run_id="20260714-001")
```

Notebook integration: a cell that lists all experiments in a DataFrame and lets the user select which run to load for inference.

### Fine-Tuning Strategy

1. Generator fine-tuning: Load horse2zebra ResNetGenerator weights. Freeze encoder (enc1, enc2, enc3). Train 9 ResNet blocks + decoder. Preserves low-level features while adapting translation.
2. Discriminator from scratch: PatchGANDiscriminator (70×70), both D_A and D_B randomly initialized.
3. Losses: LSGAN + cycle consistency (λ=10) + identity (λ=0.5).
4. Optimizer: Adam (β1=0.5), LR=2e-4, linear decay.
5. Training: batch=1, 50-100 epochs, image pool=50, FID every 10 epochs.
6. Checkpointing: registry.save() every 10 epochs with current FID + LPIPS.
7. Evaluation: FID (clean-fid), LPIPS (lpips), cycle loss, visual grid.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| src/model.py | Modified | Add PatchGANDiscriminator, freeze/unfreeze helpers |
| src/dataset.py | New | Unaligned DataLoader with augmentation |
| src/training.py | New | CycleGAN training loop with registry integration |
| src/evaluation.py | New | FID, LPIPS, visual evaluation |
| src/registry.py | New | Custom ModelRegistry (JSON-based, no deps) |
| pyproject.toml | Modified | Add lpips, clean-fid, scipy |
| fine_tuning_experiment.ipynb | Modified | Training + eval + model selection cells |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Small dataset (571 train) → overfitting | High | Frozen encoder reduces params; augmentation; monitor FID gap |
| 4GB VRAM limits batch to 1 | Known | CycleGAN designed for batch=1 |
| Domain gap too large | Med | Visual checkpointing; fallback to full fine-tune |
| Mode collapse | Med | Image pool (50); identity loss |

## Rollback Plan

Delete new files. Revert model.py and pyproject.toml via git. Delete checkpoints/experiments/ manually.

## Dependencies

- PyTorch 2.13+cu130 (installed)
- clean-fid, lpips, scipy (to add)
- Pre-trained horse2zebra checkpoints (downloaded)

## Success Criteria

- [ ] FID < 150 on test set
- [ ] LPIPS < 0.4 on test set
- [ ] Visual translations show plausible fur pattern adaptation
- [ ] No mode collapse (D loss > 0.1, G loss stable)
- [ ] Training completes within 4-6 hours on RTX 3050 for 100 epochs
- [ ] ModelRegistry: can list, compare, and load any saved experiment

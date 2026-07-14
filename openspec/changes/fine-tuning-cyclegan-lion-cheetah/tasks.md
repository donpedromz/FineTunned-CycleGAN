# Tasks: Fine-Tune CycleGAN for Lion↔Cheetah Translation

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~550-650 |
| 400-line budget risk | High |
| Chained PRs recommended | Yes |
| Suggested split | PR 1 (Foundation: model + dataset + registry ~250 lines) → PR 2 (Training + Eval + Notebook ~350 lines) |
| Delivery strategy | ask-on-risk |
| Chain strategy | feature-branch-chain |

Decision needed before apply: No (PR1 foundation merged to main; PR2 chain strategy = feature-branch-chain)
Chained PRs recommended: Yes
Chain strategy: feature-branch-chain
400-line budget risk: High (whole change ~550-650); PR2 alone ~350 lines < 400 budget

### Suggested Work Units

| Unit | Goal | Likely PR | Focused test command | Runtime harness | Rollback boundary |
|------|------|-----------|----------------------|-----------------|-------------------|
| 1 | Foundation (model.py mods, dataset.py, registry.py) | PR 1 | `python src/model.py && python src/dataset.py && python src/registry.py` | __main__ self-checks per module | Delete dataset.py, registry.py; revert model.py |
| 2 | Training loop + evaluation + notebook integration | PR 2 | `python src/training.py --smoke-test` + `python src/evaluation.py` | 2-epoch smoke on small subset | Delete training.py, evaluation.py; revert notebook + pyproject.toml |

## Phase 1: Foundation — Model (src/model.py)

- [x] 1.1 Add `PatchGANDiscriminator` class (70×70, InstanceNorm, no sigmoid) — ~40 lines
- [x] 1.2 Add `freeze_encoder()`, `unfreeze()`, `trainable_parameters()` to `ResNetGenerator` — ~20 lines
- [x] 1.3 Self-check: `__main__` block — forward pass shape (B,1,N,N), freeze sets requires_grad=False on enc1-3 only

## Phase 2: Foundation — Data (src/dataset.py)

- [x] 2.1 Create `UnpairedDataset` with resize→286→random crop 256, h-flip, normalize [-1,1] — ~50 lines
- [x] 2.2 Create `get_dataloaders(root, batch_size)` factory — ~15 lines
- [x] 2.3 Self-check: load one batch, verify (B,3,256,256) shapes and [-1,1] range

## Phase 3: Foundation — Registry (src/registry.py)

- [x] 3.1 Create `ModelRegistry(base_dir)` with `save()`, `list()`, `load_best()`, `load()` — ~80 lines
- [x] 3.2 Self-check: save dummy models, list as DataFrame, load_best by metric, load by run_id

## Phase 4: Core — Training (src/training.py)

- [ ] 4.1 Implement `lsa_gan_loss()` (MSE), `cycle_loss()` (L1), `identity_loss()` (L1) — ~25 lines
- [ ] 4.2 Implement `train_cyclegan()` with image pool (size 50), linear LR decay, registry checkpointing every 10 epochs — ~120 lines
- [ ] 4.3 Smoke test: train 2 epochs on small subset, verify losses decrease and checkpoint saved

## Phase 5: Core — Evaluation (src/evaluation.py)

- [ ] 5.1 Implement `compute_fid()`, `compute_lpips()`, `create_visual_grid()` — ~80 lines
- [ ] 5.2 Self-check: generate translations, compute metrics, render grid

## Phase 6: Notebook Integration

- [ ] 6.1 Add training cell (config + run train_cyclegan)
- [ ] 6.2 Add evaluation cell (FID + LPIPS + visual grid)
- [ ] 6.3 Add model selection cell (list experiments DataFrame, pick run_id, load for inference)

## Phase 7: Dependencies

- [ ] 7.1 Update `pyproject.toml` with lpips, clean-fid, scipy; run `uv sync`

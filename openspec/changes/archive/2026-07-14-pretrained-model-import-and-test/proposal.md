# Proposal: Import Pre-trained CycleGAN (horse2zebra) and Test on Lion/Cheetah

## Intent

Verify what the base horse2zebra model produces on lion and cheetah images before investing in fine-tuning. This pre-fine-tuning signal tells us whether the existing feature space is already useful or needs full retraining — saving compute if it works, establishing a baseline if it doesn't.

## Scope

### In Scope
- Add `torch` + `torchvision` to dependencies
- Write `ResNetGenerator` model class (9-block, matching junyanz architecture)
- Download `gen_AB.pth` + `gen_BA.pth` from HuggingFace via direct HTTP
- Create `src/inference.py` — load checkpoint → forward pass → side-by-side visualization
- Update `fine_tuning_experiment.ipynb` with model loading and test cells
- Run inference on lion→cheetah and cheetah→lion from `data/test/`

### Out of Scope
- Training, fine-tuning, or optimizer setup
- Discriminator model (not needed for inference)
- Metrics or quantitative evaluation (visual inspection only)
- Batch inference or production pipeline

## Capabilities

### New Capabilities
- `pretrained-model-inference`: Load a pre-trained CycleGAN generator checkpoint, run inference on ImageFolder-style datasets, and produce side-by-side original→translated visualizations.

### Modified Capabilities
- None

## Approach

1. `uv add torch torchvision` — install PyTorch with CUDA support
2. Write `src/model.py` with `ResNetGenerator` class (encoder: 3 Conv→IN→ReLU, 9× ResNet blocks, decoder: 3 TransposeConv→IN→ReLU → Tanh)
3. Write `src/inference.py`: auto-detect CUDA (`torch.cuda.is_available()`), move model to GPU, `load_image(path)` → normalize to [-1,1] → `model(image.unsqueeze(0).to(device))` → denormalize → display side by side via matplotlib
4. Download checkpoints: `wget https://huggingface.co/johko/cyclegan-horse2zebra/resolve/main/gen_{AB,BA}.pth` into `checkpoints/horse2zebra/`
5. Add notebook cells that load both generators (auto device detection), run translations on test samples, and display results with captions

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `pyproject.toml` | Modified | Add `torch` + `torchvision` |
| `uv.lock` | Auto-update | Via `uv lock` |
| `checkpoints/horse2zebra/` | New | Downloaded `.pth` checkpoint files |
| `src/model.py` | New | ResNetGenerator model class |
| `src/inference.py` | New | Inference + visualization pipeline |
| `fine_tuning_experiment.ipynb` | Modified | Test cells for model loading + translation display |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| PyTorch adds 1.5-3 GB to venv | High | Expected; document in README |
| State dict key mismatches | Medium | Log both key sets before loading; rename dict keys if needed |
| CUDA not available on target machine | Medium | Auto-detect with `torch.cuda.is_available()`, fallback to CPU gracefully |
| HuggingFace checkpoint unavailable | Low | Fallback to official `download_cyclegan_model.sh` from junyanz repo |
| No existing test infra | High | Notebook-based verification only; manual visual check of outputs |

## Rollback Plan

Revert each change independently:
1. `uv remove torch torchvision` to remove PyTorch
2. `rm -rf checkpoints/horse2zebra/` to delete downloaded checkpoints
3. `git checkout -- src/model.py src/inference.py` to remove new source files
4. Revert notebook via `git checkout fine_tuning_experiment.ipynb`
5. `git clean -f src/` if files are untracked

## Dependencies

- `torch` (PyTorch)
- `torchvision`
- HuggingFace checkpoint files (public, no auth needed)

## Success Criteria

- [ ] `uv run python -c "import torch; print(torch.cuda.is_available())"` confirms CUDA availability
- [ ] `uv run python -c "from src.model import ResNetGenerator; m = ResNetGenerator().cuda(); print(m)"` loads on GPU without error
- [ ] Both `gen_AB.pth` and `gen_BA.pth` download and load into model state dicts without key mismatch errors
- [ ] Notebook displays ≥4 side-by-side lion→cheetah translations from `data/test/`
- [ ] Notebook displays ≥4 side-by-side cheetah→lion translations from `data/test/`
- [ ] All translations produce valid 256×256 RGB output images (no artifacts, NaNs, or crashes)

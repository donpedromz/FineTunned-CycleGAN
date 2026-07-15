# Model Registry Specification

## Purpose

Lightweight JSON-based experiment registry for saving, listing, comparing, and loading CycleGAN checkpoints with metadata.

## Requirements

### Requirement: Checkpoint Saving

The system MUST save generator state dicts and metadata to `checkpoints/experiments/{run_id}/` with a unique timestamp-based `run_id`. Each save MUST create `gen_AB.pth`, `gen_BA.pth`, and `meta.json`.

#### Scenario: Save checkpoint

- GIVEN a `ModelRegistry("checkpoints/experiments")` and two generators
- WHEN `registry.save(gen_ab=G_A2B, gen_ba=G_B2A, epoch=50, fid=45.2, lpips=0.32, config={...})` is called
- THEN `checkpoints/experiments/{run_id}/` contains `gen_AB.pth`, `gen_BA.pth`, and `meta.json` with all fields populated

#### Scenario: Run ID uniqueness

- GIVEN two saves within the same second
- WHEN both calls complete
- THEN each gets a unique `run_id` (collision avoidance)

### Requirement: Experiment Listing

The system MUST list all experiments as a pandas DataFrame with columns: run_id, epoch, fid, lpips, cycle_loss, approach, lr, created_at.

#### Scenario: List experiments

- GIVEN 3 saved experiments in `checkpoints/experiments/`
- WHEN `registry.list()` is called
- THEN a DataFrame with 3 rows and all required columns is returned

#### Scenario: No experiments

- GIVEN empty `checkpoints/experiments/` directory
- WHEN `registry.list()` is called
- THEN an empty DataFrame with correct columns is returned

### Requirement: Load Best Model

The system MUST load generators from the experiment with the best value for a specified metric (FID or LPIPS).

#### Scenario: Load best by FID (lower is better)

- GIVEN experiments with FID values [45.2, 78.1, 32.5]
- WHEN `registry.load_best(metric="fid", ascending=True)` is called
- THEN generators from the experiment with FID=32.5 are returned

#### Scenario: Load best by LPIPS (ascending)

- GIVEN experiments with LPIPS values [0.32, 0.18, 0.41]
- WHEN `registry.load_best(metric="lpips", ascending=True)` is called
- THEN generators from the experiment with LPIPS=0.18 are returned

### Requirement: Load Specific Run

The system MUST load generators from a specific experiment identified by `run_id`.

#### Scenario: Load by run_id

- GIVEN experiment `20260714-001` exists
- WHEN `registry.load(run_id="20260714-001")` is called
- THEN `gen_AB.pth` and `gen_BA.pth` are loaded and returned as a tuple

#### Scenario: Non-existent run_id

- GIVEN no experiment with `run_id="9999"`
- WHEN `registry.load(run_id="9999")` is called
- THEN a `KeyError` is raised with a descriptive message

### Requirement: Meta.json Schema

Each `meta.json` MUST include: run_id, epoch, fid, lpips, cycle_loss, approach, lr, lambda_cycle, lambda_identity, base_checkpoint, created_at.

## Constraints

| Constraint | Value |
|-----------|-------|
| Storage | JSON files, no external dependencies |
| Run ID format | timestamp-based (e.g., `20260714-001`) |
| Checkpoint directory | `checkpoints/experiments/{run_id}/` |
| Listing format | pandas DataFrame |

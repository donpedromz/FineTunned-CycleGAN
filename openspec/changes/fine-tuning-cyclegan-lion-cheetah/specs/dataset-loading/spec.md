# Dataset Loading Specification

## Purpose

Unaligned image-to-image DataLoader with augmentation for CycleGAN training on lion/cheetah domains.

## Requirements

### Requirement: Unpaired DataLoader

The system MUST provide an `UnalignedDataLoader` that loads lion (domain A) and cheetah (domain B) images as separate, unpaired datasets with augmentation.

#### Scenario: Normal load with augmentation

- GIVEN directories `data/train/lion/` and `data/train/cheetah/` containing PNG/JPG images
- WHEN `UnpairedDataLoader(root="data/train")` is created and iterated
- THEN each iteration yields a batch tuple `(real_A, real_B)` where each tensor is `(1, 3, 256, 256)` normalized to [-1, 1]

#### Scenario: Augmentation pipeline

- GIVEN a raw image of any size loaded from disk
- WHEN the image passes through the transform pipeline
- THEN random horizontal flip (p=0.5), random crop to 256x256 (after resize to 286x286), and normalization to [-1, 1] are applied in order

#### Scenario: Empty domain directory

- GIVEN `data/train/cheetah/` is empty or missing
- WHEN `UnalignedDataLoader(root="data/train")` is created
- THEN a descriptive error is raised naming the missing/empty directory — no silent empty batches

#### Scenario: Class balance

- GIVEN domain A has 300 images and domain B has 271 images
- WHEN dataloader is exhausted
- THEN every image from the smaller domain is yielded exactly once, and the larger domain is cycled to match length

## Constraints

| Constraint | Value |
|-----------|-------|
| Image size | 256x256 (after resize→286 then crop) |
| Normalization | [-1, 1] |
| Channels | RGB (3) |
| Batch size | 1 |
| Shuffle | Yes |

# Model Evaluation Specification

## Purpose

FID, LPIPS, and visual metrics for unpaired lion/cheetah image translation quality assessment.

## Requirements

### Requirement: FID Computation

The system MUST compute Fréchet Inception Distance between real and generated image sets using the `clean-fid` library. Both domains MUST be evaluated (lion→cheetah and cheetah→lion).

#### Scenario: Normal FID evaluation

- GIVEN trained generators and test image directories
- WHEN `compute_fid(generator, domain="lion")` is called
- THEN FID score is a float representing distance between real cheetah test set and generated (lion→cheetah) images, using InceptionV3 features

#### Scenario: Small test set

- GIVEN fewer than 50 test images in a domain
- WHEN FID is computed
- THEN computation proceeds without error (clean-fid handles small sample sizes with a warning)

### Requirement: LPIPS Computation

The system MUST compute Learned Perceptual Image Patch Similarity using the `lpips` library. Average LPIPS across 4+ random pairs MUST be reported.

#### Scenario: Normal LPIPS evaluation

- GIVEN trained generators and paired (real, generated) image pairs
- WHEN `compute_lpips(real_set, generated_set)` is called
- THEN average LPIPS score (float) is returned, lower is better

### Requirement: Visual Translation Grid

The system MUST generate a visual grid showing 4 source→translated examples for each direction (lion→cheetah, cheetah→lion) using matplotlib.

#### Scenario: Grid generation

- GIVEN trained generators and test images
- WHEN `visual_grid(gen_AB, gen_BA)` is called
- THEN a 2×4 grid is displayed showing real and translated pairs for both directions

#### Scenario: No test images

- GIVEN empty test directory for a domain
- WHEN `visual_grid()` is called
- THEN a warning is logged and only available direction is shown

### Requirement: Metrics Logging

The system MUST persist FID and LPIPS to `ModelRegistry` via `meta.json` so they are available for experiment comparison.

#### Scenario: Metrics in meta.json

- GIVEN a checkpoint saved after evaluation
- WHEN `meta.json` is read
- THEN `fid` and `lpips` fields are present with float values

## Constraints

| Constraint | Value |
|-----------|-------|
| FID library | clean-fid |
| LPIPS library | lpips (AlexNet backbone) |
| Visual grid | 4 examples per direction |
| Metrics storage | ModelRegistry meta.json |

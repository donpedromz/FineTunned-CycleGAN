# CycleGAN Training Specification

## Purpose

Full CycleGAN training loop with frozen-encoder fine-tuning, adversarial + cycle + identity losses, and registry-based checkpointing.

## Requirements

### Requirement: Training Loop

The system MUST implement a `train_cyclegan()` function that trains two generators (G_A2B, G_B2A) and two discriminators (D_A, D_B) on unpaired lion/cheetah data with frozen encoder on generators.

#### Scenario: Single training epoch

- GIVEN dataloader, two generators with frozen encoders, two randomly-initialized discriminators, and all optimizers
- WHEN one epoch completes
- THEN D_A and D_B are updated on real/fake batches; G_A2B and G_B2A are updated on adversarial + cycle + identity losses

#### Scenario: Loss composition

- GIVEN generator forward pass producing fake_B = G_A2B(real_A) and rec_A = G_B2A(fake_B)
- WHEN total generator loss is computed
- THEN loss = LSGAN_loss + λ_cycle * (||rec_A - real_A|| + ||rec_B - real_B||) + λ_identity * (||G_A2B(real_B) - real_B|| + ||G_B2A(real_A) - real_A||) where λ_cycle=10, λ_identity=0.5

#### Scenario: LSGAN loss

- GIVEN discriminator D_A and a batch of real_A and fake_A
- WHEN discriminator loss is computed
- THEN loss = MSE(D_A(real_A), 1) + MSE(D_A(fake_A), 0) — NOT BCE

### Requirement: Optimizer Configuration

The system MUST use Adam optimizer with β1=0.5, β2=0.999, LR=2e-4 for all trainable parameters. Linear LR decay to 0 MUST be applied over total epochs.

#### Scenario: LR schedule

- GIVEN total_epochs=100, start_lr=2e-4
- WHEN epoch 50 is reached
- THEN effective LR is approximately 1e-4 (linear midpoint)

#### Scenario: Optimizer only sees trainable params

- GIVEN generators with frozen encoders
- WHEN optimizers are created
- THEN encoder parameters do NOT appear in any optimizer param group

### Requirement: Image Pool

The system MUST maintain an image pool of size 50 for each discriminator to reduce oscillation. Previously generated images MUST replace current ones with probability 0.5.

#### Scenario: Pool behavior

- GIVEN pool with 50 images, a new fake image arrives
- WHEN pool replacement logic executes
- THEN there is a 50% chance the pool image is returned instead of the new fake, and the returned image (whichever) replaces a random pool slot

### Requirement: Checkpointing

The system MUST save checkpoints to `ModelRegistry` every 10 epochs with current FID, LPIPS, cycle loss, and epoch number.

#### Scenario: Checkpoint trigger

- GIVEN training is in epoch 20
- WHEN epoch 20 completes
- THEN `registry.save()` is called with generators, current epoch, and all tracked metrics

#### Scenario: Final epoch

- GIVEN training reaches the final epoch (e.g., 100)
- WHEN the epoch completes
- THEN a final checkpoint is saved regardless of epoch divisibility by 10

## Constraints

| Constraint | Value |
|-----------|-------|
| Batch size | 1 |
| LR | 2e-4 |
| β1 | 0.5 |
| λ_cycle | 10 |
| λ_identity | 0.5 |
| Image pool | 50 |
| Checkpoint interval | 10 epochs |
| LR schedule | Linear decay to 0 |

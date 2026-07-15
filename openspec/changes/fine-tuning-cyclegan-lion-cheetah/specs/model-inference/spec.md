# Delta for Model Inference

## ADDED Requirements

### Requirement: PatchGANDiscriminator

The system MUST implement a `PatchGANDiscriminator` (70x70 receptive field) following the standard CycleGAN discriminator architecture: 4 downsampling Conv→LeakyReLU blocks with spectral normalization, final Conv→Sigmoid producing a single-channel real/fake prediction map.

#### Scenario: Forward pass

- GIVEN a `PatchGANDiscriminator()` instance on device
- WHEN a tensor of shape `(1, 3, 256, 256)` is passed through forward
- THEN output is `(1, 1, N, N)` where N > 1 (patch map), values in [0, 1]

#### Scenario: Parameter count

- GIVEN `PatchGANDiscriminator()` is instantiated
- WHEN total parameters are counted
- THEN count is approximately 2.8M (standard 70x70 PatchGAN)

### Requirement: Generator Freeze Mechanism

The system MUST provide `freeze_encoder()` and `unfreeze()` methods on `ResNetGenerator`. `freeze_encoder()` MUST set `requires_grad=False` on encoder layers (enc1, enc2, enc3). `unfreeze()` MUST restore `requires_grad=True` on all parameters.

#### Scenario: Freeze encoder

- GIVEN a `ResNetGenerator` with all parameters having `requires_grad=True`
- WHEN `freeze_encoder()` is called
- THEN only encoder layer parameters (enc1, enc2, enc3) have `requires_grad=False`; ResNet blocks and decoder remain `True`

#### Scenario: Unfreeze all

- GIVEN a `ResNetGenerator` with frozen encoder
- WHEN `unfreeze()` is called
- THEN all parameters have `requires_grad=True`

#### Scenario: Optimizer respects freeze

- GIVEN a `ResNetGenerator` with frozen encoder
- WHEN an Adam optimizer is created with `model.parameters()`
- THEN only trainable (ResNet + decoder) parameters appear in optimizer param groups — encoder parameters have zero gradient

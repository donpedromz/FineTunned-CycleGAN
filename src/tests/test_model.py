import torch

from src.model import PatchGANDiscriminator, ResNetGenerator


def test_generator_forward():
    model = ResNetGenerator()
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 3, 256, 256)
    out = model(x)

    assert out.shape == (1, 3, 256, 256), f"Expected (1,3,256,256), got {out.shape}"
    assert out.min() >= -1.0 and out.max() <= 1.0, "Output not in [-1, 1]"
    assert n_params > 0


def test_freeze_encoder():
    model = ResNetGenerator()
    model.freeze_encoder()

    enc_frozen = sum(
        1
        for n, p in model.named_parameters()
        if n.startswith(("enc1.", "enc2.", "enc3.")) and not p.requires_grad
    )
    dec_trainable = sum(
        1
        for n, p in model.named_parameters()
        if not n.startswith(("enc1.", "enc2.", "enc3.")) and p.requires_grad
    )

    assert enc_frozen > 0, "Encoder params should be frozen"
    assert dec_trainable > 0, "Decoder/ResNet params should still be trainable"
    assert len(model.trainable_parameters()) == dec_trainable


def test_unfreeze():
    model = ResNetGenerator()
    model.freeze_encoder()
    model.unfreeze()

    assert all(p.requires_grad for p in model.parameters()), (
        "Unfreeze should restore all grads"
    )


def test_discriminator_forward():
    disc = PatchGANDiscriminator()
    x = torch.randn(1, 3, 256, 256)
    out = disc(x)

    assert out.shape[0] == 1 and out.shape[1] == 1, (
        f"Unexpected disc output: {out.shape}"
    )

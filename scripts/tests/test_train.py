import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.train import _build_config, _build_parser, _resolve_device


def test_parse_args_defaults():
    p = _build_parser()
    args = p.parse_args([])

    assert args.epochs == 15
    assert args.lr == 2e-4
    assert args.betas == [0.5, 0.999]
    assert args.lambda_cycle == 10.0
    assert args.lambda_identity == 0.5
    assert args.pool_size == 50
    assert args.batch_size == 1
    assert args.img_size == 256
    assert args.load_size == 286
    assert args.checkpoint_interval == 10
    assert args.decay_epochs == 5
    assert args.approach == "frozen-encoder"
    assert args.data_dir == "data/train"
    assert args.test_dir_a == "data/test/lion"
    assert args.test_dir_b == "data/test/cheetah"
    assert args.registry_dir == "checkpoints/experiments"
    assert args.base_checkpoint_dir == "checkpoints/horse2zebra"
    assert args.gpu == "auto"
    assert args.progress is False
    assert args.prepare is False


def test_parse_args_custom():
    p = _build_parser()
    args = p.parse_args([
        "--epochs", "30",
        "--lr", "1e-4",
        "--betas", "0.0", "0.9",
        "--approach", "full",
        "--gpu", "cpu",
        "--progress",
        "--prepare",
    ])

    assert args.epochs == 30
    assert args.lr == 1e-4
    assert args.betas == [0.0, 0.9]
    assert args.approach == "full"
    assert args.gpu == "cpu"
    assert args.progress is True
    assert args.prepare is True

    assert args.decay_epochs == 5
    assert args.batch_size == 1


def test_build_config_keys():
    p = _build_parser()
    args = p.parse_args([])
    config = _build_config(args)

    expected_keys = {
        "epochs", "lr", "betas", "lambda_cycle", "lambda_identity",
        "pool_size", "batch_size", "img_size", "load_size",
        "checkpoint_interval", "decay_epochs", "approach",
        "test_dir_a", "test_dir_b",
    }
    assert set(config.keys()) == expected_keys, (
        f"Missing or extra keys: {set(config.keys()) ^ expected_keys}"
    )


def test_build_config_betas_tuple():
    p = _build_parser()
    args = p.parse_args([])
    config = _build_config(args)

    assert isinstance(config["betas"], tuple), f"betas should be tuple, got {type(config['betas'])}"
    assert config["betas"] == (0.5, 0.999)


def test_resolve_device():
    assert _resolve_device("cpu") == torch.device("cpu")
    assert _resolve_device("cuda") == torch.device("cuda")


def test_smoke():
    torch.manual_seed(0)
    device = torch.device("cpu")

    from src.model import PatchGANDiscriminator, ResNetGenerator
    from src.registry import ModelRegistry
    from src.training import train_cyclegan

    n, bs = 4, 2
    dl = DataLoader(
        TensorDataset(torch.randn(n, 3, 256, 256), torch.randn(n, 3, 256, 256)),
        batch_size=bs,
        shuffle=True,
    )
    gen_ab = ResNetGenerator()
    gen_ba = ResNetGenerator()
    d_a = PatchGANDiscriminator()
    d_b = PatchGANDiscriminator()

    config = {
        "epochs": 1,
        "lr": 2e-4,
        "betas": (0.5, 0.999),
        "lambda_cycle": 10.0,
        "lambda_identity": 0.5,
        "pool_size": 50,
        "batch_size": bs,
        "img_size": 256,
        "load_size": 286,
        "checkpoint_interval": 10,
        "decay_epochs": 100,
        "approach": "frozen-encoder",
        "test_dir_a": None,
        "test_dir_b": None,
    }

    with tempfile.TemporaryDirectory() as tmp:
        reg = ModelRegistry(tmp)
        history = train_cyclegan(
            gen_ab, gen_ba, d_a, d_b, dl, config, reg,
            "clitest001", device, progress=False,
        )

        for k in ("G_total", "D_total", "cycle", "lr"):
            assert k in history, f"Missing key: {k}"
            assert len(history[k]) == 1

        run_dir = Path(tmp) / "clitest001-e01"
        assert (run_dir / "gen_AB.pth").exists()
        assert (run_dir / "meta.json").exists()

import tempfile
from pathlib import Path

import pytest

from src.model import ResNetGenerator
from src.registry import ModelRegistry, RegistryError, RunNotFoundError


def _make_registry() -> tuple[ModelRegistry, ResNetGenerator, ResNetGenerator, dict]:
    reg = ModelRegistry(tempfile.mkdtemp())
    g_ab = ResNetGenerator()
    g_ba = ResNetGenerator()
    config = {
        "epochs": 10,
        "lr": 2e-4,
        "lambda_cycle": 10.0,
        "approach": "frozen-encoder",
    }
    return reg, g_ab, g_ba, config


def test_save_and_list():
    reg, g_ab, g_ba, config = _make_registry()

    reg.save(g_ab, g_ba, run_id="test001", config=config, training_results={
        "epoch": 10, "cycle_loss": 1.2, "g_loss": 4.5, "d_loss": 0.8,
    })
    reg.save(g_ab, g_ba, run_id="test002", config=config, training_results={
        "epoch": 20, "cycle_loss": 0.9, "g_loss": 3.8, "d_loss": 0.6,
    })

    records = reg.list()
    assert len(records) == 2
    assert records[0]["run_id"] == "test002"


def test_update_meta():
    reg, g_ab, g_ba, config = _make_registry()
    reg.save(g_ab, g_ba, run_id="test001", config=config, training_results={
        "epoch": 10, "cycle_loss": 1.2, "g_loss": 4.5, "d_loss": 0.8,
    })

    reg.update_meta("test001", evaluation={"fid_ab": 45.2, "lpips_ab": 0.31})
    records = reg.list()
    assert records[0]["evaluation"]["fid_ab"] == 45.2


def test_load_best():
    reg, g_ab, g_ba, config = _make_registry()
    reg.save(g_ab, g_ba, run_id="test001", config=config, training_results={
        "epoch": 10, "cycle_loss": 1.2, "g_loss": 4.5, "d_loss": 0.8,
    })
    reg.save(g_ab, g_ba, run_id="test002", config=config, training_results={
        "epoch": 20, "cycle_loss": 0.9, "g_loss": 3.8, "d_loss": 0.6,
    })
    reg.update_meta("test001", evaluation={"fid_ab": 45.2})
    reg.update_meta("test002", evaluation={"fid_ab": 38.7})

    g_best, _ = reg.load_best("fid_ab", ascending=True)
    assert g_best is not None


def test_load_specific():
    reg, g_ab, g_ba, config = _make_registry()
    reg.save(g_ab, g_ba, run_id="test001", config=config, training_results={
        "epoch": 10, "cycle_loss": 1.2, "g_loss": 4.5, "d_loss": 0.8,
    })

    loaded_ab, loaded_ba = reg.load("test001")
    assert loaded_ab is not None


def test_load_nonexistent_raises():
    reg, g_ab, g_ba, config = _make_registry()
    with pytest.raises(RunNotFoundError):
        reg.load("nonexistent")


def test_load_best_nonexistent_raises():
    reg, g_ab, g_ba, config = _make_registry()
    with pytest.raises(RegistryError):
        reg.load_best("nonexistent_metric")

"""Model registry for CycleGAN experiment tracking.

Stores checkpoints and metadata as JSON files on disk — zero external
dependencies. Provides save, list (as DataFrame), load_best, and load
operations.

Usage:
    reg = ModelRegistry("checkpoints/experiments")
    reg.save(gen_ab, gen_ba, run_id="exp001", config=config, training_results={...})
    df = reg.list()
    reg.load_best("fid_ab", ascending=True)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


class RegistryError(Exception):
    """Base exception for registry errors."""


class RunNotFoundError(RegistryError):
    """Raised when a run_id does not exist."""


_META_SECTIONS = {"config", "training", "evaluation"}


class ModelRegistry:
    """JSON-file-based experiment registry for generator checkpoints.

    Directory layout:
        base_dir/{run_id}/
            gen_AB.pth
            gen_BA.pth
            meta.json

    meta.json structure:
        {
            "run_id", "created_at", "approach", "base_checkpoint",
            "config":      { full training config },
            "training":    { epoch, cycle_loss, g_loss, d_loss },
            "evaluation":   { fid_ab, fid_ba, lpips_ab, lpips_ba }
        }
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def _validate_meta(self, meta: dict) -> None:
        required_top = {
            "run_id",
            "created_at",
            "approach",
            "config",
            "training",
            "evaluation",
        }
        missing = required_top - set(meta.keys())
        if missing:
            logger.warning("meta.json missing top-level keys: %s", missing)
        for section in ("training", "evaluation"):
            if not isinstance(meta.get(section), dict):
                logger.warning("meta.json section '%s' is not a dict", section)

    def save(
        self,
        gen_AB: nn.Module,
        gen_BA: nn.Module,
        run_id: str,
        config: dict,
        training_results: dict,
        base_checkpoint: str | None = None,
    ) -> None:
        """Save generators and metadata for an experiment run.

        Args:
            gen_AB: Generator A→B.
            gen_BA: Generator B→A.
            run_id: Unique experiment identifier.
            config: Full training config dict (serializable values kept).
            training_results: {epoch, cycle_loss, g_loss, d_loss}.
            base_checkpoint: Path to the pre-trained checkpoint (if any).
        """
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        torch.save(gen_AB.state_dict(), run_dir / "gen_AB.pth")
        torch.save(gen_BA.state_dict(), run_dir / "gen_BA.pth")

        safe_config = {
            k: v
            for k, v in config.items()
            if isinstance(v, (bool, int, float, str, list, type(None)))
        }

        meta = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "approach": config.get("approach", "frozen-encoder"),
            "base_checkpoint": base_checkpoint,
            "config": safe_config,
            "training": {
                "epoch": training_results["epoch"],
                "cycle_loss": float(training_results["cycle_loss"]),
                "g_loss": float(training_results["g_loss"]),
                "d_loss": float(training_results["d_loss"]),
            },
            "evaluation": {
                "fid_ab": None,
                "fid_ba": None,
                "lpips_ab": None,
                "lpips_ba": None,
            },
        }
        self._validate_meta(meta)
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        logger.info("Saved run %s to %s", run_id, run_dir)

    def list(self) -> list[dict]:
        """List all experiments as a list of metadata dicts.

        Returns:
            List of meta dicts sorted by created_at descending.
            Empty list if no experiments exist.
        """
        if not self.base_dir.exists():
            return []

        records: list[dict] = []
        for run_dir in sorted(self.base_dir.iterdir()):
            meta_path = run_dir / "meta.json"
            if meta_path.exists():
                try:
                    records.append(json.loads(meta_path.read_text()))
                except json.JSONDecodeError:
                    logger.warning("Corrupt meta.json in %s", run_dir)
        records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return records

    def update_meta(self, run_id: str, **kwargs) -> None:
        """Patch metadata fields on an existing run.

        Supports two patterns:
            update_meta(run_id, evaluation={"fid_ab": 42.0})  → deep-merges into section
            update_meta(run_id, approach="new-value")          → updates top-level

        Raises:
            RunNotFoundError: run_id does not exist.
        """
        run_dir = self.base_dir / run_id
        if not run_dir.is_dir():
            raise RunNotFoundError(f"Run not found: {run_id}")
        meta_path = run_dir / "meta.json"
        meta = json.loads(meta_path.read_text())
        for key, value in kwargs.items():
            if (
                isinstance(value, dict)
                and key in _META_SECTIONS
                and isinstance(meta.get(key), dict)
            ):
                meta[key].update(value)
            else:
                meta[key] = value
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.info("Updated meta for run %s: %s", run_id, list(kwargs.keys()))

    def _flatten_meta(self, meta: dict) -> dict:
        """Flatten nested sections into a single lookup dict.

        Section keys are promoted directly so that e.g. ``evaluation.fid_ab``
        is accessible as ``flat["fid_ab"]``. Top-level keys take precedence.
        """
        flat: dict = {}
        for k, v in meta.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    flat.setdefault(sk, sv)
            else:
                flat[k] = v
        return flat

    def load_best(
        self,
        metric: str,
        ascending: bool = True,
        gen_AB: nn.Module | None = None,
        gen_BA: nn.Module | None = None,
    ) -> tuple[nn.Module, nn.Module]:
        """Load generators from the best experiment by metric.

        Args:
            metric: Meta key to optimize (e.g. "fid_ab", "lpips_ab", "cycle_loss").
                    Searches nested sections automatically.
            ascending: True → minimize (lowest wins), False → maximize.
            gen_AB: Optional generator instance to load weights into.
            gen_BA: Optional generator instance to load weights into.

        Returns:
            Tuple of (gen_AB, gen_BA) with loaded weights.

        Raises:
            RegistryError: No experiments exist or metric missing.
        """
        records = self.list()
        valid = []
        for r in records:
            flat = self._flatten_meta(r)
            if flat.get(metric) is not None:
                valid.append((r, flat[metric]))
        if not valid:
            all_keys: set[str] = set()
            for r in records:
                all_keys.update(self._flatten_meta(r).keys())
            raise RegistryError(
                f"No experiments with metric '{metric}'. Saved metrics: "
                f"{all_keys if all_keys else 'none'}"
            )

        best = (
            min(valid, key=lambda pair: pair[1])
            if ascending
            else max(valid, key=lambda pair: pair[1])
        )
        best_record = best[0]
        logger.info("Best run by %s (%s): %s", metric, best[1], best_record["run_id"])

        return self.load(best_record["run_id"], gen_AB, gen_BA)

    def load(
        self,
        run_id: str,
        gen_AB: nn.Module | None = None,
        gen_BA: nn.Module | None = None,
    ) -> tuple[nn.Module, nn.Module]:
        """Load generators from a specific run.

        Args:
            run_id: Experiment identifier.
            gen_AB: Optional generator to load weights into (else created fresh).
            gen_BA: Optional generator to load weights into (else created fresh).

        Returns:
            Tuple of (gen_AB, gen_BA) with loaded weights.

        Raises:
            RunNotFoundError: run_id does not exist.
        """
        import sys

        _parent = str(Path(__file__).resolve().parent.parent)
        if _parent not in sys.path:
            sys.path.insert(0, _parent)
        from src.model import ResNetGenerator

        run_dir = self.base_dir / run_id
        if not run_dir.is_dir():
            raise RunNotFoundError(f"Run not found: {run_id}")

        if gen_AB is None:
            gen_AB = ResNetGenerator()
        if gen_BA is None:
            gen_BA = ResNetGenerator()

        ckpt_AB = torch.load(run_dir / "gen_AB.pth", weights_only=True)
        ckpt_BA = torch.load(run_dir / "gen_BA.pth", weights_only=True)
        gen_AB.load_state_dict(ckpt_AB)
        gen_BA.load_state_dict(ckpt_BA)

        logger.info("Loaded run %s from %s", run_id, run_dir)
        return gen_AB, gen_BA


# ── Self-check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    logging.basicConfig(level=logging.INFO)

    from src.model import ResNetGenerator

    with tempfile.TemporaryDirectory() as tmpdir:
        reg = ModelRegistry(tmpdir)

        g_ab = ResNetGenerator()
        g_ba = ResNetGenerator()

        config = {
            "epochs": 10,
            "lr": 2e-4,
            "lambda_cycle": 10.0,
            "approach": "frozen-encoder",
        }
        reg.save(
            g_ab,
            g_ba,
            run_id="test001",
            config=config,
            training_results={
                "epoch": 10,
                "cycle_loss": 1.2,
                "g_loss": 4.5,
                "d_loss": 0.8,
            },
        )
        reg.save(
            g_ab,
            g_ba,
            run_id="test002",
            config=config,
            training_results={
                "epoch": 20,
                "cycle_loss": 0.9,
                "g_loss": 3.8,
                "d_loss": 0.6,
            },
        )

        # Update evaluation section
        reg.update_meta("test001", evaluation={"fid_ab": 45.2, "lpips_ab": 0.31})
        reg.update_meta("test002", evaluation={"fid_ab": 38.7, "lpips_ab": 0.25})

        records = reg.list()
        assert len(records) == 2, f"Expected 2 records, got {len(records)}"
        assert records[0]["run_id"] == "test002"
        assert records[0]["evaluation"]["fid_ab"] == 38.7
        print(f"List OK — {len(records)} experiments")

        # Load best by fid_ab (nested metric, ascending → lowest wins)
        g_loaded_ab, g_loaded_ba = reg.load_best("fid_ab", ascending=True)
        assert g_loaded_ab is not None
        print("Load best (nested fid_ab) OK — loaded by lowest FID")

        # Load best by cycle_loss (nested training metric)
        g2_ab, _ = reg.load_best("cycle_loss", ascending=True)
        assert g2_ab is not None
        print("Load best (nested cycle_loss) OK")

        # Load specific run
        g_spec_ab, g_spec_ba = reg.load("test001")
        assert g_spec_ab is not None
        print("Load specific OK — loaded test001")

        # Error cases
        try:
            reg.load("nonexistent")
            assert False, "Should have raised RunNotFoundError"
        except RunNotFoundError:
            print("RunNotFoundError OK")

        try:
            reg.load_best("nonexistent_metric")
            assert False, "Should have raised RegistryError"
        except RegistryError:
            print("RegistryError OK")

    print("All self-checks passed.")

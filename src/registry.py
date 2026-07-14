"""Model registry for CycleGAN experiment tracking.

Stores checkpoints and metadata as JSON files on disk — zero external
dependencies. Provides save, list (as DataFrame), load_best, and load
operations.

Usage:
    reg = ModelRegistry("checkpoints/experiments")
    reg.save(gen_ab, gen_ba, run_id="exp001", epoch=10, fid=45.2)
    df = reg.list()
    reg.load_best("fid", ascending=True)
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


class ModelRegistry:
    """JSON-file-based experiment registry for generator checkpoints.

    Directory layout:
        base_dir/{run_id}/
            gen_AB.pth
            gen_BA.pth
            meta.json
    """

    REQUIRED_META_KEYS = {
        "run_id", "epoch", "fid", "lpips", "cycle_loss",
        "approach", "lr", "lambda_cycle", "lambda_identity",
        "created_at",
    }

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def save(
        self,
        gen_AB: nn.Module,
        gen_BA: nn.Module,
        run_id: str,
        epoch: int = 0,
        fid: float | None = None,
        lpips: float | None = None,
        cycle_loss: float | None = None,
        approach: str = "frozen-encoder",
        lr: float = 2e-4,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 0.5,
        base_checkpoint: str | None = None,
    ) -> None:
        """Save generators and metadata for an experiment run."""
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        torch.save(gen_AB.state_dict(), run_dir / "gen_AB.pth")
        torch.save(gen_BA.state_dict(), run_dir / "gen_BA.pth")

        meta = {
            "run_id": run_id,
            "epoch": epoch,
            "fid": fid,
            "lpips": lpips,
            "cycle_loss": cycle_loss,
            "approach": approach,
            "lr": lr,
            "lambda_cycle": lambda_cycle,
            "lambda_identity": lambda_identity,
            "base_checkpoint": base_checkpoint,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        logger.info("Saved run %s (epoch %d) to %s", run_id, epoch, run_dir)

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

    def load_best(
        self,
        metric: str,
        ascending: bool = True,
        gen_AB: nn.Module | None = None,
        gen_BA: nn.Module | None = None,
    ) -> tuple[nn.Module, nn.Module]:
        """Load generators from the best experiment by metric.

        Args:
            metric: Meta key to optimize (e.g. "fid", "lpips").
            ascending: True → minimize (lowest wins), False → maximize.
            gen_AB: Optional generator instance to load weights into.
            gen_BA: Optional generator instance to load weights into.

        Returns:
            Tuple of (gen_AB, gen_BA) with loaded weights.

        Raises:
            RegistryError: No experiments exist or metric missing.
        """
        records = self.list()
        valid = [r for r in records if r.get(metric) is not None]
        if not valid:
            raise RegistryError(
                f"No experiments with metric '{metric}'. Saved metrics: "
                f"{set().union(*(r.keys() for r in records)) if records else 'none'}"
            )

        best = min(valid, key=lambda r: r[metric]) if ascending else max(valid, key=lambda r: r[metric])
        logger.info("Best run by %s (%s): %s", metric, best[metric], best["run_id"])

        return self.load(best["run_id"], gen_AB, gen_BA)

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

        # Create dummy generators
        g_ab = ResNetGenerator()
        g_ba = ResNetGenerator()

        # Save
        reg.save(g_ab, g_ba, run_id="test001", epoch=10, fid=45.2, lpips=0.31, cycle_loss=1.2)
        reg.save(g_ab, g_ba, run_id="test002", epoch=20, fid=38.7, lpips=0.25, cycle_loss=0.9)

        # List
        records = reg.list()
        assert len(records) == 2, f"Expected 2 records, got {len(records)}"
        assert records[0]["run_id"] == "test002"  # newest first
        print(f"List OK — {len(records)} experiments")

        # Load best by fid (ascending → lowest wins)
        g_loaded_ab, g_loaded_ba = reg.load_best("fid", ascending=True)
        assert g_loaded_ab is not None
        print("Load best OK — loaded by lowest FID")

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

    print("All self-checks passed.")

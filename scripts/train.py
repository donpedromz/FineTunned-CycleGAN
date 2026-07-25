"""Train CycleGAN fine-tuning from CLI. Replaces the notebook training cell."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch

from src.dataset import get_dataloaders
from src.download_and_prepare import DatasetPreparationError, prepare_dataset
from src.inference import CycleGANInference
from src.model import PatchGANDiscriminator, ResNetGenerator
from src.registry import ModelRegistry
from src.training import _select_device, train_cyclegan

logger = logging.getLogger("train")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fine-tune CycleGAN horse2zebra -> lion/cheetah"
    )

    g_data = p.add_argument_group("Dataset")
    g_data.add_argument(
        "--prepare",
        action="store_true",
        help="Run dataset preparation before training",
    )
    g_data.add_argument(
        "--data-dir",
        default="data/train",
        help="Training data directory with {lion,cheetah} subdirs (default: %(default)s)",
    )
    g_data.add_argument(
        "--test-dir-a",
        default="data/test/lion",
        help="Test images for domain A / lion (default: %(default)s)",
    )
    g_data.add_argument(
        "--test-dir-b",
        default="data/test/cheetah",
        help="Test images for domain B / cheetah (default: %(default)s)",
    )

    g_hp = p.add_argument_group("Hyperparameters")
    g_hp.add_argument(
        "--epochs", type=int, default=15,
        help="Total training epochs (default: %(default)s)",
    )
    g_hp.add_argument(
        "--lr", type=float, default=2e-4,
        help="Initial learning rate for Adam (default: %(default)s)",
    )
    g_hp.add_argument(
        "--betas", type=float, nargs=2, default=[0.5, 0.999],
        help="Adam beta coefficients (beta1 beta2, default: %(default)s)",
    )
    g_hp.add_argument(
        "--lambda-cycle", type=float, default=10.0,
        help="Cycle-consistency loss weight (default: %(default)s)",
    )
    g_hp.add_argument(
        "--lambda-identity", type=float, default=0.5,
        help="Identity loss weight (default: %(default)s)",
    )
    g_hp.add_argument(
        "--pool-size", type=int, default=50,
        help="Image history buffer size for discriminator (default: %(default)s)",
    )
    g_hp.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size (default: %(default)s)",
    )
    g_hp.add_argument(
        "--img-size", type=int, default=256,
        help="Final crop size for training images (default: %(default)s)",
    )
    g_hp.add_argument(
        "--load-size", type=int, default=286,
        help="Resize before random crop (default: %(default)s)",
    )
    g_hp.add_argument(
        "--checkpoint-interval", type=int, default=10,
        help="Save a checkpoint every N epochs (default: %(default)s)",
    )
    g_hp.add_argument(
        "--decay-epochs",
        type=int,
        default=5,
        help="Epochs with constant LR before linear decay starts (default: %(default)s)",
    )
    g_hp.add_argument(
        "--approach",
        default="frozen-encoder",
        choices=["frozen-encoder", "full"],
        help="frozen-encoder: train decoder only; full: train all weights (default: %(default)s)",
    )

    g_path = p.add_argument_group("Paths")
    g_path.add_argument(
        "--registry-dir",
        default="checkpoints/experiments",
        help="ModelRegistry output directory (default: %(default)s)",
    )
    g_path.add_argument(
        "--base-checkpoint-dir",
        default="checkpoints/horse2zebra",
        help="Directory with gen_AB.pth / gen_BA.pth (default: %(default)s)",
    )

    g_run = p.add_argument_group("Runtime")
    g_run.add_argument(
        "--gpu",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device selection (default: %(default)s)",
    )
    g_run.add_argument(
        "--progress",
        action="store_true",
        help="Show tqdm progress bars (default: off for log-friendly output)",
    )

    return p


def _resolve_device(choice: str) -> torch.device:
    if choice == "cuda":
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return _select_device()


def _build_config(args: argparse.Namespace) -> dict:
    return {
        "epochs": args.epochs,
        "lr": args.lr,
        "betas": tuple(args.betas),
        "lambda_cycle": args.lambda_cycle,
        "lambda_identity": args.lambda_identity,
        "pool_size": args.pool_size,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "load_size": args.load_size,
        "checkpoint_interval": args.checkpoint_interval,
        "decay_epochs": args.decay_epochs,
        "approach": args.approach,
        "test_dir_a": args.test_dir_a,
        "test_dir_b": args.test_dir_b,
    }


def main() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    device = _resolve_device(args.gpu)
    logger.info("Device: %s", device)

    if args.prepare:
        logger.info("Starting dataset preparation ...")
        try:
            prepare_dataset()
            logger.info("Dataset preparation complete.")
        except DatasetPreparationError as e:
            logger.error("Dataset preparation failed: %s", e)
            sys.exit(1)

    logger.info("Ensuring pre-trained horse2zebra checkpoints ...")
    CycleGANInference().download_checkpoints()

    ckpt_dir = Path(args.base_checkpoint_dir)
    logger.info("Loading generators from %s ...", ckpt_dir)
    gen_ab = ResNetGenerator.from_checkpoint(str(ckpt_dir / "gen_AB.pth"), device)
    gen_ba = ResNetGenerator.from_checkpoint(str(ckpt_dir / "gen_BA.pth"), device)
    base_checkpoint = str(ckpt_dir)

    d_a = PatchGANDiscriminator()
    d_b = PatchGANDiscriminator()

    dl_train = get_dataloaders(args.data_dir, batch_size=args.batch_size)

    registry = ModelRegistry(args.registry_dir)

    today = datetime.now().strftime("%Y%m%d")
    seq = len([r for r in registry.list() if r["run_id"].startswith(today)]) + 1
    run_id = f"{today}-{seq:03d}"

    config = _build_config(args)

    logger.info(
        "run_id=%s | epochs=%d | lr=%.1e | device=%s",
        run_id,
        args.epochs,
        args.lr,
        device,
    )

    history = train_cyclegan(
        gen_ab,
        gen_ba,
        d_a,
        d_b,
        dl_train,
        config,
        registry,
        run_id,
        device,
        initial_base_checkpoint=base_checkpoint,
        progress=args.progress,
    )

    logger.info(
        "Training complete — G_total: %.4f -> %.4f",
        history["G_total"][0],
        history["G_total"][-1],
    )

    if "fid_ab" in history:
        logger.info(
            "FID   lion->cheetah: %.2f | cheetah->lion: %.2f",
            history["fid_ab"][-1],
            history["fid_ba"][-1],
        )
        logger.info(
            "LPIPS lion->cheetah: %.3f | cheetah->lion: %.3f",
            history["lpips_ab"][-1],
            history["lpips_ba"][-1],
        )


if __name__ == "__main__":
    main()

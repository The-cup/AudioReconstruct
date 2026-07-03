from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config.paths import build_dir_path
from datasets.audio_dataset import GanDataset
from models.registry import get_model
from tasks.gan.test import evaluate_gan
from tasks.gan.train import gan_collate_fn, train_gan_model


LOGGER = logging.getLogger(__name__)

DEFAULT_EPOCHS = 32
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_VALIDATE_EVERY_N_EPOCHS = 50
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_SEED = 42
DEFAULT_NUM_WORKERS = 1
DEFAULT_EMBEDDING_VECTOR_NAME = "embedded_vector.pt"


def _resolve_device(device: str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _get_weights_path() -> Path:
    from config.paths import CHECKPOINTS_DIR
    return CHECKPOINTS_DIR / "voice_expand_gan.pth"


def _get_log_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    from config.paths import LOGS_DIR
    return LOGS_DIR / f"voice_expand_gan_{timestamp}"


def _load_oss_datasets() -> tuple[dict, dict, dict]:
    try:
        from oss.oss_dataset import oss_test_dataset, oss_train_dataset, oss_val_dataset
    except Exception as exc:
        raise RuntimeError("Failed to import OSS GAN datasets from src/oss/oss_dataset.py.") from exc
    return oss_train_dataset, oss_val_dataset, oss_test_dataset


def _validate_oss_dataset(name: str, dataset_obj) -> None:
    if dataset_obj is None:
        raise ValueError(f"{name} dataset is None.")
    required_keys = ("sample", "low_freq", "embedding")
    for key in required_keys:
        if key not in dataset_obj:
            raise KeyError(f"{name} dataset is missing required key: {key}")
    try:
        dataset_length = len(dataset_obj["sample"])
    except Exception as exc:
        raise RuntimeError(f"Failed to determine length for {name} dataset.") from exc
    if dataset_length <= 0:
        raise RuntimeError(f"{name} dataset is empty.")


def _build_dataloaders(
    *,
    train_dataset_obj,
    val_dataset_obj,
    test_dataset_obj,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = GanDataset(oss_dataset=train_dataset_obj)
    val_dataset = GanDataset(oss_dataset=val_dataset_obj)
    test_dataset = GanDataset(oss_dataset=test_dataset_obj)

    _validate_oss_dataset("train", train_dataset_obj)
    _validate_oss_dataset("validation", val_dataset_obj)
    _validate_oss_dataset("test", test_dataset_obj)

    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=gan_collate_fn,
        drop_last=False,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=gan_collate_fn,
        drop_last=False,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=gan_collate_fn,
        drop_last=False,
    )

    if len(train_dataloader) <= 0:
        raise RuntimeError("GAN train dataloader is empty.")
    if len(val_dataloader) <= 0:
        raise RuntimeError("GAN validation dataloader is empty.")
    if len(test_dataloader) <= 0:
        raise RuntimeError("GAN test dataloader is empty.")

    return train_dataloader, val_dataloader, test_dataloader


def train_and_evaluate(
    *,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    validate_every_n_epochs: int = DEFAULT_VALIDATE_EVERY_N_EPOCHS,
    seed: int = DEFAULT_SEED,
    num_workers: int = DEFAULT_NUM_WORKERS,
    device: str | None = None,
    weights_path: Path | None = None,
    log_dir: Path | None = None,
) -> dict[str, float | int]:
    resolved_device = _resolve_device(device)

    LOGGER.info("Loading GAN OSS datasets...")
    oss_train_dataset, oss_val_dataset, oss_test_dataset = _load_oss_datasets()

    LOGGER.info("Building GAN dataloaders...")
    train_dataloader, val_dataloader, test_dataloader = _build_dataloaders(
        train_dataset_obj=oss_train_dataset,
        val_dataset_obj=oss_val_dataset,
        test_dataset_obj=oss_test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=resolved_device,
    )

    LOGGER.info("Getting GAN model...")
    model = get_model("voice_expand_gan")

    resolved_weights_path = Path(weights_path) if weights_path is not None else _get_weights_path()
    resolved_log_dir = Path(log_dir) if log_dir is not None else _get_log_dir()
    resolved_log_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting GAN training...")
    model = train_gan_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=resolved_device,
        log_dir=resolved_log_dir,
        weights_path=resolved_weights_path,
        seed=seed,
        validate_every_n_epochs=validate_every_n_epochs,
    )

    LOGGER.info("Running GAN test evaluation...")
    if len(test_dataloader) == 0:
        LOGGER.warning("Test dataloader is empty; skipping GAN test evaluation.")
        return {}

    test_metrics = evaluate_gan(
        model=model,
        test_dataloader=test_dataloader,
        device=resolved_device,
        seed=seed,
    )
    LOGGER.info("GAN test metrics: %s", test_metrics)
    return test_metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the GAN voice expansion model.")
    parser.add_argument("--project_root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate.")
    parser.add_argument(
        "--validate_every_n_epochs",
        type=int,
        default=DEFAULT_VALIDATE_EVERY_N_EPOCHS,
        help="Validation interval in epochs.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="DataLoader workers.")
    parser.add_argument("--device", type=str, default=None, help="Target device, e.g. cpu or cuda.")
    parser.add_argument("--weights_path", type=Path, default=None, help="Checkpoint output path.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    project_root = args.project_root or Path(__file__).resolve().parents[3]
    try:
        build_dir_path(project_root)
    except Exception as exc:
        raise RuntimeError(f"Failed to build project directory paths from {project_root}.") from exc
    logging.basicConfig(level=logging.INFO)

    from config.paths import LOGS_DIR
    train_and_evaluate(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validate_every_n_epochs=args.validate_every_n_epochs,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        weights_path=args.weights_path,
        log_dir=LOGS_DIR,
    )


if __name__ == "__main__":
    main()

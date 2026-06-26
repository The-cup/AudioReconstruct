from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.audio_dataset import GanDataset
from datasets.dataset_builder import build_gan_dataset_split
from models.registry import get_model
from tasks.gan.test import evaluate_gan
from tasks.gan.train import gan_collate_fn, train_gan_model


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDED_VECTOR_DIR = DATA_DIR / "selected"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOW_FREQ_DATA_DIR = DATA_DIR / "low_freq"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
LOG_DIR = ARTIFACTS_DIR / "logs"

DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_VALIDATE_EVERY_N_EPOCHS = 50
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_SEED = 42
DEFAULT_NUM_WORKERS = 1


def _resolve_device(device: str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _get_weights_path() -> Path:
    return CHECKPOINT_DIR / "voice_expand_gan.pth"


def _get_log_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"voice_expand_gan_{timestamp}"


def _build_dataloaders(
    dataset: GanDataset,
    *,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader | None, DataLoader]:
    train_dataset, val_dataset, test_dataset = build_gan_dataset_split(
        dataset=dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

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
    val_dataloader = None
    if len(val_dataset) > 0:
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
    return train_dataloader, val_dataloader, test_dataloader


def train_and_evaluate(
    *,
    data_dir: Path = PROCESSED_DATA_DIR,
    low_freq_data_dir: Path = LOW_FREQ_DATA_DIR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    validate_every_n_epochs: int = DEFAULT_VALIDATE_EVERY_N_EPOCHS,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
    num_workers: int = DEFAULT_NUM_WORKERS,
    device: str | None = None,
    weights_path: Path | None = None,
    log_dir: Path | None = None,
) -> dict[str, float | int]:
    resolved_device = _resolve_device(device)

    LOGGER.info("Loading GAN dataset from %s and %s", data_dir, low_freq_data_dir)
    dataset = GanDataset(
        embedded_vector_dir=EMBEDDED_VECTOR_DIR,
        processed_dataset_dir=data_dir,
        low_freq_dataset_dir=low_freq_data_dir,
        randomize=True,
    )
    if len(dataset) == 0:
        raise RuntimeError("GAN dataset is empty. Please prepare processed and low-frequency samples first.")

    LOGGER.info("Building GAN dataloaders...")
    train_dataloader, val_dataloader, test_dataloader = _build_dataloaders(
        dataset,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
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
    parser.add_argument("--data_dir", type=Path, default=PROCESSED_DATA_DIR, help="Processed mel-spectrogram directory.")
    parser.add_argument(
        "--low_freq_data_dir",
        type=Path,
        default=LOW_FREQ_DATA_DIR,
        help="Low-frequency mel-spectrogram directory.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate.")
    parser.add_argument(
        "--validate_every_n_epochs",
        type=int,
        default=DEFAULT_VALIDATE_EVERY_N_EPOCHS,
        help="Validation interval in epochs.",
    )
    parser.add_argument("--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO, help="Training split ratio.")
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="DataLoader workers.")
    parser.add_argument("--device", type=str, default=None, help="Target device, e.g. cpu or cuda.")
    parser.add_argument("--weights_path", type=Path, default=None, help="Checkpoint output path.")
    parser.add_argument("--log_dir", type=Path, default=None, help="TensorBoard log directory.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    train_and_evaluate(
        data_dir=args.data_dir,
        low_freq_data_dir=args.low_freq_data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validate_every_n_epochs=args.validate_every_n_epochs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        weights_path=args.weights_path,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()

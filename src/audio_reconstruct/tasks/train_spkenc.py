import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from audio_reconstruct.data.load_data import load_raw_data
from audio_reconstruct.data.preprocess.pipeline import run_spkenc_preprocessing_pipeline
from audio_reconstruct.datasets.dataset_builder import build_dataset
from audio_reconstruct.ml.train import train
from audio_reconstruct.models.registry import get_model, get_loss_function

# directory to save the processed data
DATA_BASE_DIR = Path("D:\\projects\\python\\AudioReconstruct\\data")
DATA_RAW_DIR = DATA_BASE_DIR / "raw"
DATA_PROCESSED_DIR = DATA_BASE_DIR / "processed"

ARTIFACTS_BASE_DIR = Path("D:\\projects\\python\\AudioReconstruct\\artifacts")
LOG_DIR = ARTIFACTS_BASE_DIR / "logs"

# logger
LOGGER = logging.getLogger(__name__)

# model parameters
EPOCHS = 100
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOSS_FUNCTION = "GE2E"


def _get_dataset():
    return load_raw_data(dataset_name="LibriSpeech", dataset_sub_name="train-clean-100", base_dir=DATA_RAW_DIR)


def _get_model():
    return get_model("spkenc")

def train_and_evaluate():
    # Load the dataset
    LOGGER.info("Loading the dataset...")
    dataset = _get_dataset()

    # Run the preprocessing pipeline
    LOGGER.info("Running the preprocessing pipeline...")
    preprocessed_dataset = run_spkenc_preprocessing_pipeline(dataset=dataset, save_dir=DATA_PROCESSED_DIR)

    # Split the dataset into train/val/test subsets
    LOGGER.info("Splitting the dataset into train/val/test subsets...")
    train_dataset, val_dataset, test_dataset = build_dataset(dataset=preprocessed_dataset, train_ratio=0.8, val_ratio=0.1)

    # Get Dataloader
    LOGGER.info("Getting Dataloader...")
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    # Get model
    LOGGER.info("Getting model...")
    model = _get_model()

    loss_fn = get_loss_function(LOSS_FUNCTION)

    train(
        model=model,
        train_dataset=train_dataloader,
        validation_dataset=val_dataloader,
        epochs=EPOCHS,
        device=DEVICE,
        log_dir=LOG_DIR,
        learning_rate=LEARNING_RATE,
        loss_fn=loss_fn
    )


if __name__ == "__main__":
    pass

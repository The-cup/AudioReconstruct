from __future__ import annotations

import logging
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.load_data import load_raw_data
from data.preprocess.pipeline import run_preprocessing_pipeline
from datasets.dataset_builder import build_spk_dataset_split
from tasks.spkenc.test import test_spkenc
from tasks.spkenc.train import train_spkenc
from models.custom.ge2e_sampler import (
    GE2ETestBatchSampler,
    GE2ETrainBatchSampler,
    GE2EValidationBatchSampler,
    ge2e_collate,
)
from models.registry import get_loss_function, get_model


DATA_BASE_DIR = Path("D:\\projects\\python\\AudioReconstruct\\data")
DATA_RAW_DIR = DATA_BASE_DIR / "raw"
DATA_PROCESSED_DIR = DATA_BASE_DIR / "processed"

ARTIFACTS_BASE_DIR = Path("D:\\projects\\python\\AudioReconstruct\\artifacts")
LOG_DIR = ARTIFACTS_BASE_DIR / "logs"
WEIGHTS_DIR = ARTIFACTS_BASE_DIR / "checkpoints"

LOGGER = logging.getLogger(__name__)

EPOCHS = 100
NUM_SPEAKERS_PER_BATCH = 4
UTTERANCES_PER_SPEAKER = 10
TRAIN_CHUNKS_PER_SPEAKER_PER_EPOCH = 2
VALIDATE_EVERY_N_EPOCHS = 50
NUM_WORKERS = 1
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOSS_FUNCTION = "GE2E"
RANDOM_SEED = 42


def _get_dataset():
    return load_raw_data(
        dataset_name="LibriSpeech",
        dataset_sub_name="train-clean-100",
        base_dir=DATA_RAW_DIR,
    )


def _get_model():
    return get_model("spkenc")


def _get_loss_function():
    return get_loss_function(LOSS_FUNCTION)


def _get_dataloader(train_dataset, val_dataset, test_dataset):
    collate_fn = partial(
        ge2e_collate,
        num_speakers_per_batch=NUM_SPEAKERS_PER_BATCH,
        num_utterances_per_speaker=UTTERANCES_PER_SPEAKER,
    )

    train_sampler = GE2ETrainBatchSampler(
        dataset=train_dataset,
        num_speakers_per_batch=NUM_SPEAKERS_PER_BATCH,
        num_utterances_per_speaker=UTTERANCES_PER_SPEAKER,
        chunks_per_speaker_per_epoch=TRAIN_CHUNKS_PER_SPEAKER_PER_EPOCH,
        shuffle_speakers=True,
        seed=RANDOM_SEED,
    )
    val_sampler = GE2EValidationBatchSampler(
        dataset=val_dataset,
        num_speakers_per_batch=NUM_SPEAKERS_PER_BATCH,
        num_utterances_per_speaker=UTTERANCES_PER_SPEAKER,
        shuffle_speakers=False,
        seed=RANDOM_SEED,
    )
    test_sampler = GE2ETestBatchSampler(
        dataset=test_dataset,
        num_speakers_per_batch=NUM_SPEAKERS_PER_BATCH,
        num_utterances_per_speaker=UTTERANCES_PER_SPEAKER,
        shuffle_speakers=False,
        seed=RANDOM_SEED,
    )

    return {
        "train": DataLoader(
            dataset=train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        ),
        "validation": DataLoader(
            dataset=val_dataset,
            batch_sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        ),
        "test": DataLoader(
            dataset=test_dataset,
            batch_sampler=test_sampler,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        ),
    }


def _get_weights_path():
    now = datetime.now()
    now_str = now.strftime("%y-%m-%d-%H-%M-%S")
    return WEIGHTS_DIR / f"spkenc_{now_str}.pth"


def _get_log_dir():
    now = datetime.now()
    now_str = now.strftime("%y-%m-%d-%H-%M-%S")
    log_dir = LOG_DIR / f"spkenc_{now_str}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def train_and_evaluate():
    LOGGER.info("Loading the dataset...")
    dataset = _get_dataset()

    LOGGER.info("Running the preprocessing pipeline...")
    preprocessed_dataset = run_preprocessing_pipeline(dataset=dataset, save_dir=DATA_PROCESSED_DIR)
    LOGGER.info("Preprocessing done. Dataset size: %s", len(preprocessed_dataset))

    LOGGER.info("Splitting the dataset into train/val/test subsets...")
    train_dataset, val_dataset, test_dataset = build_spk_dataset_split(
        dataset=preprocessed_dataset,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=RANDOM_SEED,
        min_utt_per_spk=UTTERANCES_PER_SPEAKER,
    )

    LOGGER.info("Getting loss function %s...", LOSS_FUNCTION)
    loss_fn = _get_loss_function()

    LOGGER.info("Getting dataloaders...")
    all_dataloader = _get_dataloader(train_dataset, val_dataset, test_dataset)

    LOGGER.info("Getting model...")
    model = _get_model()

    weight_save_path = _get_weights_path()
    log_dir = _get_log_dir()

    model = train_spkenc(
        model=model,
        train_dataloader=all_dataloader["train"],
        val_dataloader=all_dataloader["validation"],
        epochs=EPOCHS,
        device=DEVICE,
        log_dir=log_dir,
        learning_rate=LEARNING_RATE,
        loss_fn=loss_fn,
        weights_path=weight_save_path,
        validate_every_n_epochs=VALIDATE_EVERY_N_EPOCHS,
        steps_per_epoch=None,
        max_grad_norm=1.0,
    )

    test_spkenc(
        model=model,
        test_dataloader=all_dataloader["test"],
        loss_fn=loss_fn,
        device=DEVICE,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_and_evaluate()

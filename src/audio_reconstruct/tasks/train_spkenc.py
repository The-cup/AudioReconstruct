from __future__ import annotations

import logging
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from audio_reconstruct.data.load_data import load_raw_data
from audio_reconstruct.data.preprocess.pipeline import run_spkenc_preprocessing_pipeline
from audio_reconstruct.datasets.dataset_builder import build_spk_dataset_split
from audio_reconstruct.ml.train import train
from audio_reconstruct.models.custom.ge2e_sampler import GE2ESampler, ge2e_collate
from audio_reconstruct.models.registry import get_loss_function, get_model


DATA_BASE_DIR = Path("D:\\projects\\python\\AudioReconstruct\\data")
DATA_RAW_DIR = DATA_BASE_DIR / "raw"
DATA_PROCESSED_DIR = DATA_BASE_DIR / "processed"

ARTIFACTS_BASE_DIR = Path("D:\\projects\\python\\AudioReconstruct\\artifacts")
LOG_DIR = ARTIFACTS_BASE_DIR / "logs"
WEIGHTS_DIR = ARTIFACTS_BASE_DIR / "checkpoints"

LOGGER = logging.getLogger(__name__)

EPOCHS = 10
NUM_SPEAKERS_PER_BATCH = 2
UTTERANCES_PER_SPEAKER = 10
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

    train_sampler = GE2ESampler(
        dataset=train_dataset,
        num_speakers_per_batch=NUM_SPEAKERS_PER_BATCH,
        num_utterances_per_speaker=UTTERANCES_PER_SPEAKER,
        shuffle=True,
        seed=RANDOM_SEED,
    )
    val_sampler = GE2ESampler(
        dataset=val_dataset,
        num_speakers_per_batch=NUM_SPEAKERS_PER_BATCH,
        num_utterances_per_speaker=UTTERANCES_PER_SPEAKER,
        shuffle=False,
        seed=RANDOM_SEED,
    )
    test_sampler = GE2ESampler(
        dataset=test_dataset,
        num_speakers_per_batch=NUM_SPEAKERS_PER_BATCH,
        num_utterances_per_speaker=UTTERANCES_PER_SPEAKER,
        shuffle=False,
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


def train_and_evaluate():
    LOGGER.info("Loading the dataset...")
    dataset = _get_dataset()

    LOGGER.info("Running the preprocessing pipeline...")
    preprocessed_dataset = run_spkenc_preprocessing_pipeline(dataset=dataset, save_dir=DATA_PROCESSED_DIR)
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

    train(
        model=model,
        train_dataloader=all_dataloader["train"],
        val_dataloader=all_dataloader["validation"],
        epochs=EPOCHS,
        device=DEVICE,
        log_dir=LOG_DIR,
        learning_rate=LEARNING_RATE,
        loss_fn=loss_fn,
        weights_path=weight_save_path,
        steps_per_epoch=None,
        max_grad_norm=1.0,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_and_evaluate()

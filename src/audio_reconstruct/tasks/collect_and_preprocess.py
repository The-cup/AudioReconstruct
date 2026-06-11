from __future__ import annotations

from pathlib import Path

from audio_reconstruct.data.load_data import load_raw_data
from audio_reconstruct.data.preprocess.pipeline import run_preprocessing_pipeline
from audio_reconstruct.datasets.audio_dataset import AudioReconstructionDataset
from audio_reconstruct.datasets.dataset_builder import build_dataset


def collect_data(dataset_name: str = "LibriSpeech", dataset_sub_name: str = "train-clean-100") -> AudioReconstructionDataset:
    """Load the raw dataset by name.

    Args:
        dataset_name: Name of the dataset to load.
        dataset_sub_name: Subset name within the dataset.

    Returns:
        A dataset object that inherits from AudioReconstructionDataset.
    """
    dataset = load_raw_data(dataset_name=dataset_name, dataset_sub_name=dataset_sub_name)
    if not isinstance(dataset, AudioReconstructionDataset):
        raise TypeError(
            "load_raw_data() must return an AudioReconstructionDataset-compatible object."
        )
    return dataset


def preprocess_data(
    dataset: AudioReconstructionDataset,
    save_dir: Path | None = None,
) -> AudioReconstructionDataset:
    """Run the full preprocessing pipeline on a raw dataset.

    Args:
        dataset: Raw dataset instance.
        save_dir: Reserved output directory parameter. Kept for future use.

    Returns:
        The processed dataset returned by the preprocessing pipeline.
    """
    processed_dataset = run_preprocessing_pipeline(dataset, save_dir)
    if not isinstance(processed_dataset, AudioReconstructionDataset):
        raise TypeError(
            "run_preprocessing_pipeline() must return an AudioReconstructionDataset-compatible object."
        )
    return processed_dataset


def run_collect_and_preprocess(
    dataset_name: str = "LibriSpeech",
    save_dir: Path | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[AudioReconstructionDataset, AudioReconstructionDataset, AudioReconstructionDataset]:
    """Run the full data pipeline: collect, preprocess, and split.

    Args:
        dataset_name: Name of the dataset to load.
        save_dir: Reserved output directory parameter for preprocessing.
        train_ratio: Ratio assigned to the train split.
        val_ratio: Ratio assigned to the validation split.

    Returns:
        A tuple of (train_dataset, val_dataset, test_dataset).
    """
    raw_dataset = collect_data(dataset_name=dataset_name)
    processed_dataset = preprocess_data(dataset=raw_dataset, save_dir=save_dir)
    train_dataset, val_dataset, test_dataset = build_dataset(
        dataset=processed_dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    return train_dataset, val_dataset, test_dataset

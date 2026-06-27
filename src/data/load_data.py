from __future__ import annotations
from pathlib import Path

from datasets.audio_dataset import (
    AudioReconstructionDataset,
    LibriSpeechRawDataset,
)

from config.paths import RAW_DATA_DIR


def load_raw_data(
    dataset_name: str = "LibriSpeech",
    dataset_sub_name: str = "train-clean-100",
    base_dir: str | Path = RAW_DATA_DIR,
) -> AudioReconstructionDataset:
    """Load a raw dataset by name.

    Args:
        dataset_name: Dataset name to load.
        dataset_sub_name: LibriSpeech subset name.
        base_dir: Root directory that contains the raw dataset folders.

    Returns:
        A raw dataset object inheriting from AudioReconstructionDataset.
    """
    dataset_name = dataset_name.strip()
    if dataset_name != "LibriSpeech":
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    base_path = Path(base_dir)
    dataset_root = base_path
    if not dataset_root.exists():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {dataset_root}")

    raw_dataset = LibriSpeechRawDataset(base_dir=dataset_root, dataset_sub_name=dataset_sub_name)
    return raw_dataset


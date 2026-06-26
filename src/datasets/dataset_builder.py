from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterator

from .audio_dataset import AudioReconstructionDataset, GanDataset, SpkEncDataset, create_spk_subset, get_all_speakers


DEFAULT_SEED = 42


@dataclass(slots=True)
class _DatasetSubset(AudioReconstructionDataset):
    """Lightweight dataset wrapper for deterministic train/val/test splits."""

    dataset: AudioReconstructionDataset
    indices: list[int]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]

    def __iter__(self) -> Iterator:
        for index in self.indices:
            yield self.dataset[index]


def _validate_ratios(train_ratio: float, val_ratio: float) -> None:
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio must not exceed 1.0.")


def build_dataset(
    dataset: AudioReconstructionDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[AudioReconstructionDataset, AudioReconstructionDataset, AudioReconstructionDataset]:
    """Split a dataset into train/validation/test subsets."""
    _validate_ratios(train_ratio, val_ratio)

    total_size = len(dataset)
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0.0:
        raise ValueError("The resulting test_ratio must not be negative.")

    indices = list(range(total_size))
    random.Random(DEFAULT_SEED).shuffle(indices)

    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    if total_size > 0 and test_ratio > 0.0 and test_size == 0:
        if val_size > 0:
            val_size -= 1
        elif train_size > 0:
            train_size -= 1

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    return (
        _DatasetSubset(dataset=dataset, indices=train_indices),
        _DatasetSubset(dataset=dataset, indices=val_indices),
        _DatasetSubset(dataset=dataset, indices=test_indices),
    )


def build_spk_dataset_split(
    dataset: SpkEncDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = DEFAULT_SEED,
    min_utt_per_spk: int = 2,
) -> tuple[SpkEncDataset, SpkEncDataset, SpkEncDataset]:
    """Split a speaker embedding dataset at the speaker level."""
    _validate_ratios(train_ratio, val_ratio)

    all_valid_spks = get_all_speakers(dataset, min_utt_per_spk=min_utt_per_spk)
    if not all_valid_spks:
        raise RuntimeError("No valid speakers for GE2E split.")

    rng = random.Random(seed)
    rng.shuffle(all_valid_spks)

    total_spk = len(all_valid_spks)
    train_spk_num = int(total_spk * train_ratio)
    val_spk_num = int(total_spk * val_ratio)

    train_spks = all_valid_spks[:train_spk_num]
    val_spks = all_valid_spks[train_spk_num : train_spk_num + val_spk_num]
    test_spks = all_valid_spks[train_spk_num + val_spk_num :]

    return (
        create_spk_subset(dataset, train_spks),
        create_spk_subset(dataset, val_spks),
        create_spk_subset(dataset, test_spks),
    )


def _get_gan_speakers(dataset: GanDataset) -> list[str]:
    speaker_ids = sorted({item.speaker_id for item in dataset._data_files})
    return speaker_ids


def _create_gan_subset(source_ds: GanDataset, selected_spks: list[str]) -> GanDataset:
    selected_speakers = set(selected_spks)
    new_ds = GanDataset(
        embedded_vector_dir=None,
        processed_dataset_dir=None,
        low_freq_dataset_dir=None,
        randomize=source_ds.randomize,
    )
    new_ds.processed_dataset_dir = source_ds.processed_dataset_dir
    new_ds.low_freq_dataset_dir = source_ds.low_freq_dataset_dir
    new_ds._data_files = [item for item in source_ds._data_files if item.speaker_id in selected_speakers]
    return new_ds


def build_gan_dataset_split(
    dataset: GanDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> tuple[GanDataset, GanDataset, GanDataset]:
    """Split a GAN dataset at the speaker level."""
    _validate_ratios(train_ratio, val_ratio)

    all_speakers = _get_gan_speakers(dataset)
    if not all_speakers:
        raise RuntimeError("No valid speakers for GAN split.")

    rng = random.Random(seed)
    rng.shuffle(all_speakers)

    total_spk = len(all_speakers)
    train_spk_num = int(total_spk * train_ratio)
    val_spk_num = int(total_spk * val_ratio)
    if total_spk > 0:
        train_spk_num = max(1, train_spk_num)

    if total_spk >= 3:
        val_spk_num = max(1, val_spk_num)
        if train_spk_num + val_spk_num >= total_spk:
            overflow = train_spk_num + val_spk_num - (total_spk - 1)
            if overflow > 0:
                reducible_train = min(overflow, max(0, train_spk_num - 1))
                train_spk_num -= reducible_train
                overflow -= reducible_train
            if overflow > 0:
                val_spk_num -= min(overflow, max(0, val_spk_num - 1))
        test_spk_num = total_spk - train_spk_num - val_spk_num
    else:
        test_spk_num = total_spk - train_spk_num - val_spk_num

    train_spks = all_speakers[:train_spk_num]
    val_spks = all_speakers[train_spk_num : train_spk_num + val_spk_num]
    test_spks = all_speakers[train_spk_num + val_spk_num :]

    return (
        _create_gan_subset(dataset, train_spks),
        _create_gan_subset(dataset, val_spks),
        _create_gan_subset(dataset, test_spks),
    )

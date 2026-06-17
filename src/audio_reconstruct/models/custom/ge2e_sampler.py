from __future__ import annotations

import random
from typing import Iterator, List

import torch
from torch.utils.data import Sampler

from audio_reconstruct.datasets.audio_dataset import (
    SpkEncDataset,
    build_spk_to_items,
    get_data_files,
)


class GE2ESampler(Sampler[List[int]]):
    """Batch sampler that groups speaker embeddings for GE2E training."""

    def __init__(
        self,
        dataset: SpkEncDataset,
        num_speakers_per_batch: int = 4,
        num_utterances_per_speaker: int = 10,
        shuffle: bool = True,
        seed: int = 42,
        min_utterances_per_speaker: int | None = None,
    ) -> None:
        super().__init__(data_source=dataset)
        if num_speakers_per_batch <= 0:
            raise ValueError("num_speakers_per_batch must be greater than 0.")
        if num_utterances_per_speaker <= 0:
            raise ValueError("num_utterances_per_speaker must be greater than 0.")

        self.dataset = dataset
        self.num_speakers_per_batch = num_speakers_per_batch
        self.num_utterances_per_speaker = num_utterances_per_speaker
        self.shuffle = shuffle
        self.seed = seed
        self.min_utterances_per_speaker = (
            min_utterances_per_speaker if min_utterances_per_speaker is not None else num_utterances_per_speaker
        )

        speaker_to_items = build_spk_to_items(dataset)
        self.speaker_to_items = {
            speaker_id: items
            for speaker_id, items in speaker_to_items.items()
            if len(items) >= self.min_utterances_per_speaker
        }
        self.valid_speakers = list(self.speaker_to_items.keys())
        if len(self.valid_speakers) < self.num_speakers_per_batch:
            raise ValueError(
                "Not enough valid speakers to build a GE2E batch "
                f"({len(self.valid_speakers)} < {self.num_speakers_per_batch})."
            )

        self.item_to_idx = {
            item.file_path: idx
            for idx, item in enumerate(get_data_files(dataset))
        }

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
        speakers = list(self.valid_speakers)
        if self.shuffle:
            rng.shuffle(speakers)

        batches: list[list[int]] = []
        for start in range(0, len(speakers), self.num_speakers_per_batch):
            selected_speakers = speakers[start : start + self.num_speakers_per_batch]
            if len(selected_speakers) < self.num_speakers_per_batch:
                break

            group_indices: list[int] = []
            for speaker_id in selected_speakers:
                utter_items = list(self.speaker_to_items[speaker_id])
                if self.shuffle:
                    rng.shuffle(utter_items)
                if len(utter_items) >= self.num_utterances_per_speaker:
                    selected_utts = utter_items[: self.num_utterances_per_speaker]
                else:
                    selected_utts = [rng.choice(utter_items) for _ in range(self.num_utterances_per_speaker)]
                group_indices.extend(self.item_to_idx[item.file_path] for item in selected_utts)

            batches.append(group_indices)

        if self.shuffle:
            rng.shuffle(batches)

        for batch_indices in batches:
            yield batch_indices

    def __len__(self) -> int:
        return len(self.valid_speakers) // self.num_speakers_per_batch


def ge2e_collate(
    batch: List[dict[str, torch.Tensor]],
    num_speakers_per_batch: int,
    num_utterances_per_speaker: int,
) -> torch.Tensor:
    """Stack utterance tensors into GE2E speaker groups."""
    expected_batch_size = num_speakers_per_batch * num_utterances_per_speaker
    if len(batch) != expected_batch_size:
        raise ValueError(
            f"GE2E collate expected {expected_batch_size} samples, received {len(batch)}."
        )

    mel_tensors = [item["file"] for item in batch]
    stacked = torch.stack(mel_tensors, dim=0)
    return stacked.reshape(
        num_speakers_per_batch,
        num_utterances_per_speaker,
        *stacked.shape[1:],
    )

from __future__ import annotations

import math
import random
from typing import Iterator, List, Sequence

import torch
from torch.utils.data import Sampler

from datasets import (
    SpkEncDataset,
    build_spk_to_items,
    get_data_files,
)


def _cyclic_slice(indices: Sequence[int], start: int, size: int) -> list[int]:
    if not indices:
        raise ValueError("Cannot sample utterances from an empty speaker bucket.")
    length = len(indices)
    return [indices[(start + offset) % length] for offset in range(size)]


def _pad_speakers(speakers: list[str], target_size: int) -> list[str]:
    if not speakers:
        raise ValueError("Cannot build a GE2E batch without any valid speakers.")
    if len(speakers) >= target_size:
        return speakers[:target_size]

    padded = list(speakers)
    while len(padded) < target_size:
        padded.append(speakers[len(padded) % len(speakers)])
    return padded[:target_size]


class _BaseGE2EBatchSampler(Sampler[List[int]]):
    """Common utilities for GE2E batch samplers."""

    def __init__(
        self,
        dataset: SpkEncDataset,
        num_speakers_per_batch: int = 4,
        num_utterances_per_speaker: int = 10,
        shuffle_speakers: bool = True,
        seed: int = 42,
        min_utterances_per_speaker: int = 2,
    ) -> None:
        super().__init__(data_source=dataset)
        if num_speakers_per_batch <= 0:
            raise ValueError("num_speakers_per_batch must be greater than 0.")
        if num_utterances_per_speaker <= 0:
            raise ValueError("num_utterances_per_speaker must be greater than 0.")
        if min_utterances_per_speaker <= 0:
            raise ValueError("min_utterances_per_speaker must be greater than 0.")

        self.dataset = dataset
        self.num_speakers_per_batch = num_speakers_per_batch
        self.num_utterances_per_speaker = num_utterances_per_speaker
        self.shuffle_speakers = shuffle_speakers
        self.seed = seed
        self.min_utterances_per_speaker = min_utterances_per_speaker
        self._epoch = 0

        self._speaker_to_items = {
            speaker_id: items
            for speaker_id, items in build_spk_to_items(dataset).items()
            if len(items) >= self.min_utterances_per_speaker
        }
        self._speaker_ids = sorted(self._speaker_to_items.keys())
        if len(self._speaker_ids) < self.num_speakers_per_batch:
            raise ValueError(
                "Not enough valid speakers to construct a GE2E batch "
                f"({len(self._speaker_ids)} < {self.num_speakers_per_batch})."
            )

        self._item_to_idx = {
            item.file_path: index
            for index, item in enumerate(get_data_files(dataset))
        }
        self._speaker_groups = math.ceil(len(self._speaker_ids) / self.num_speakers_per_batch)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = max(0, int(epoch))

    def _speaker_order(self, rng: random.Random) -> list[str]:
        speakers = list(self._speaker_ids)
        if self.shuffle_speakers:
            rng.shuffle(speakers)
        return speakers

    def _speaker_batches(self, speakers: list[str]) -> list[list[str]]:
        batches: list[list[str]] = []
        for start in range(0, len(speakers), self.num_speakers_per_batch):
            batch_speakers = speakers[start : start + self.num_speakers_per_batch]
            batches.append(_pad_speakers(batch_speakers, self.num_speakers_per_batch))
        return batches

    def _utterance_indices_for_round(
        self,
        speaker_id: str,
        round_index: int,
        cycle_span: int,
        rng: random.Random,
    ) -> list[int]:
        utterance_items = self._speaker_to_items[speaker_id]
        utterance_indices = [self._item_to_idx[item.file_path] for item in utterance_items]
        if not utterance_indices:
            raise ValueError(f"Speaker {speaker_id!r} does not have any utterances.")

        start = ((self._epoch * cycle_span) + round_index) * self.num_utterances_per_speaker
        start %= len(utterance_indices)
        return _cyclic_slice(utterance_indices, start, self.num_utterances_per_speaker)

    def _build_batches(
        self,
        speakers: list[str],
        round_count: int,
        rng: random.Random,
    ) -> Iterator[List[int]]:
        speaker_batches = self._speaker_batches(speakers)
        for round_index in range(round_count):
            for batch_speakers in speaker_batches:
                batch_indices: list[int] = []
                for speaker_id in batch_speakers:
                    batch_indices.extend(
                        self._utterance_indices_for_round(
                            speaker_id=speaker_id,
                            round_index=round_index,
                            cycle_span=round_count,
                            rng=rng,
                        )
                    )
                yield batch_indices


class GE2ETrainBatchSampler(_BaseGE2EBatchSampler):
    """Training sampler that rotates through multiple utterance chunks per epoch."""

    def __init__(
        self,
        dataset: SpkEncDataset,
        num_speakers_per_batch: int = 4,
        num_utterances_per_speaker: int = 10,
        chunks_per_speaker_per_epoch: int = 2,
        shuffle_speakers: bool = True,
        seed: int = 42,
        min_utterances_per_speaker: int = 2,
    ) -> None:
        if chunks_per_speaker_per_epoch <= 0:
            raise ValueError("chunks_per_speaker_per_epoch must be greater than 0.")
        super().__init__(
            dataset=dataset,
            num_speakers_per_batch=num_speakers_per_batch,
            num_utterances_per_speaker=num_utterances_per_speaker,
            shuffle_speakers=shuffle_speakers,
            seed=seed,
            min_utterances_per_speaker=min_utterances_per_speaker,
        )
        self.chunks_per_speaker_per_epoch = chunks_per_speaker_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self._epoch)
        speakers = self._speaker_order(rng)
        yield from self._build_batches(
            speakers=speakers,
            round_count=self.chunks_per_speaker_per_epoch,
            rng=rng,
        )

    def __len__(self) -> int:
        return self._speaker_groups * self.chunks_per_speaker_per_epoch


class GE2EValidationBatchSampler(_BaseGE2EBatchSampler):
    """Validation sampler that keeps evaluation light and deterministic."""

    def __init__(
        self,
        dataset: SpkEncDataset,
        num_speakers_per_batch: int = 4,
        num_utterances_per_speaker: int = 10,
        shuffle_speakers: bool = False,
        seed: int = 42,
        min_utterances_per_speaker: int = 2,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_speakers_per_batch=num_speakers_per_batch,
            num_utterances_per_speaker=num_utterances_per_speaker,
            shuffle_speakers=shuffle_speakers,
            seed=seed,
            min_utterances_per_speaker=min_utterances_per_speaker,
        )

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self._epoch)
        speakers = self._speaker_order(rng)
        yield from self._build_batches(
            speakers=speakers,
            round_count=1,
            rng=rng,
        )

    def __len__(self) -> int:
        return self._speaker_groups


class GE2ETestBatchSampler(_BaseGE2EBatchSampler):
    """Test sampler that traverses all available utterances."""

    def __init__(
        self,
        dataset: SpkEncDataset,
        num_speakers_per_batch: int = 4,
        num_utterances_per_speaker: int = 10,
        shuffle_speakers: bool = False,
        seed: int = 42,
        min_utterances_per_speaker: int = 2,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_speakers_per_batch=num_speakers_per_batch,
            num_utterances_per_speaker=num_utterances_per_speaker,
            shuffle_speakers=shuffle_speakers,
            seed=seed,
            min_utterances_per_speaker=min_utterances_per_speaker,
        )
        self._speaker_chunk_count = max(
            1,
            max(
                math.ceil(len(items) / self.num_utterances_per_speaker)
                for items in self._speaker_to_items.values()
            ),
        )

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self._epoch)
        speakers = self._speaker_order(rng)
        yield from self._build_batches(
            speakers=speakers,
            round_count=self._speaker_chunk_count,
            rng=rng,
        )

    def __len__(self) -> int:
        return self._speaker_groups * self._speaker_chunk_count


# Backward-compatible alias used by existing code.
GE2ESampler = GE2ETrainBatchSampler


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

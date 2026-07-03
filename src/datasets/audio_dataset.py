from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset
from config.paths import RAW_DATA_DIR

LOGGER = logging.getLogger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = (
    ".flac",
    ".wav",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
)


@dataclass(slots=True)
class LibriSpeechRawItem:
    """Metadata for a single raw LibriSpeech utterance."""

    speaker_id: str
    chapter_id: str
    label: str
    file: str
    text: str = ""


@dataclass(slots=True)
class LibriSpeechProcessedItem:
    """Metadata for a single processed LibriSpeech feature file."""

    label: str
    file: str
    text: str


@dataclass(slots=True)
class SpkEncDatasetItem:
    """Metadata for a single processed speaker embedding sample."""

    speaker_id: str
    file_path: Path


class AudioReconstructionDataset(Dataset):
    """Base dataset for the audio reconstruction project."""

    _data_files: list[Any]

    def __iter__(self) -> Iterator[Any]:
        for index in range(len(self)):
            yield self[index]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={len(self)})"

    def __len__(self) -> int:
        return len(self._data_files)

    def __getitem__(self, index: int) -> Any: ...


def _find_audio_file(directory: Path, stem: str) -> Path | None:
    for extension in SUPPORTED_AUDIO_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.exists():
            return candidate

    for candidate in directory.glob(f"{stem}.*"):
        if candidate.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            return candidate
    return None


def build_spk_to_items(dataset: "SpkEncDataset") -> dict[str, list[SpkEncDatasetItem]]:
    speaker_to_items: dict[str, list[SpkEncDatasetItem]] = defaultdict(list)
    for item in dataset._data_files:
        speaker_to_items[item.speaker_id].append(item)
    return dict(speaker_to_items)


def get_data_files(dataset: "SpkEncDataset") -> list[SpkEncDatasetItem]:
    return list(dataset._data_files)


def get_speaker_utterances(dataset: "SpkEncDataset", speaker_id: str) -> list[SpkEncDatasetItem]:
    speaker_to_items = build_spk_to_items(dataset)
    return list(speaker_to_items.get(speaker_id, []))


def get_all_speakers(dataset: "SpkEncDataset", min_utt_per_spk: int = 2) -> list[str]:
    speaker_to_items = build_spk_to_items(dataset)
    return [speaker_id for speaker_id, items in speaker_to_items.items() if len(items) >= min_utt_per_spk]


def create_spk_subset(source_ds: "SpkEncDataset", selected_spks: list[str]) -> "SpkEncDataset":
    selected_speakers = set(selected_spks)
    new_ds = SpkEncDataset(randomize=source_ds.randomize)
    for item in source_ds._data_files:
        if item.speaker_id in selected_speakers:
            new_ds.add_item(item.speaker_id, item.file_path)
    return new_ds


class LibriSpeechRawDataset(AudioReconstructionDataset):
    """Raw LibriSpeech dataset reader."""

    def __init__(
        self,
        base_dir: str | Path = RAW_DATA_DIR,
        dataset_sub_name: str = "train-clean-100",
        dataset_length_limit: int | None = None,
    ) -> None:
        super().__init__()
        self._dataset_name = "LibriSpeech"
        self._dataset_subset_name = dataset_sub_name
        self._dataset_subset_dir = Path(base_dir) / self._dataset_name / dataset_sub_name
        self._data_files: list[LibriSpeechRawItem] = []

        if not self._dataset_subset_dir.exists():
            LOGGER.warning("Raw directory does not exist: %s", self._dataset_subset_dir)
            return

        item_count = 0
        transcript_files = sorted(self._dataset_subset_dir.rglob("*.trans.txt"))
        for transcript_file in transcript_files:
            chapter_dir = transcript_file.parent
            if chapter_dir == self._dataset_subset_dir:
                continue

            speaker_id = chapter_dir.parent.name
            chapter_id = chapter_dir.name

            with transcript_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    label, text = line.split(" ", maxsplit=1)
                    audio_file = _find_audio_file(chapter_dir, label)
                    if audio_file is None:
                        LOGGER.warning("Audio file not found for label %s in %s", label, chapter_dir)
                        continue
                    self._data_files.append(
                        LibriSpeechRawItem(
                            speaker_id=speaker_id,
                            chapter_id=chapter_id,
                            label=label,
                            file=str(audio_file),
                            text=text,
                        )
                    )
                    item_count += 1

            if dataset_length_limit is not None and item_count >= dataset_length_limit:
                break

        LOGGER.info("Found %d raw audio files", len(self._data_files))

    def __len__(self) -> int:
        return len(self._data_files)

    def __getitem__(self, idx: int) -> LibriSpeechRawItem:
        return self._data_files[idx]


class SpkEncDataset(AudioReconstructionDataset):
    """Dataset for speaker embedding training."""

    def __init__(
        self,
        processed_dataset_dir: Path | None = None,
        randomize: bool = True,
    ) -> None:
        super().__init__()
        self.randomize = randomize
        self._data_files: list[SpkEncDatasetItem] = []

        if processed_dataset_dir is not None:
            self.build_from_dir(processed_dataset_dir)

    def build_from_dir(self, processed_dataset_dir: Path) -> None:
        for speaker_dir in sorted(processed_dataset_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker_id = speaker_dir.name
            for file_path in sorted(speaker_dir.iterdir()):
                if file_path.suffix != ".pt":
                    continue
                self.add_item(speaker_id, file_path)

    def add_item(self, speaker_id: str, file_path: Path) -> None:
        self._data_files.append(SpkEncDatasetItem(speaker_id=speaker_id, file_path=file_path))

    def __len__(self) -> int:
        return len(self._data_files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self._data_files[idx]
        file = torch.load(item.file_path, map_location="cpu", weights_only=True)
        return {
            "speaker_id": item.speaker_id,
            "file": file,
            "file_path": item.file_path,
        }

    def get_all_speakers(self, min_utt_per_spk: int = 2) -> list[str]:
        return get_all_speakers(self, min_utt_per_spk=min_utt_per_spk)

    def get_speaker_utterances(self, speaker_id: str) -> list[SpkEncDatasetItem]:
        return get_speaker_utterances(self, speaker_id)

    def get_data_files(self) -> list[SpkEncDatasetItem]:
        return get_data_files(self)


@dataclass
class GanDatasetItem:
    speaker_id: str
    embedded_path: Path
    sample_path: Path
    low_freq_sample_path: Path


class GanDataset(AudioReconstructionDataset):
    """Dataset for GAN training."""

    def __init__(
        self,
        embedded_vector_dir: Path | None = None,
        embedded_vector_name: str = "embedded_vector.pt",
        processed_dataset_dir: Path | None = None,
        low_freq_dataset_dir: Path | None = None,
        randomize: bool = True,
        oss_dataset = None,
    ) -> None:
        super().__init__()
        self.randomize = randomize
        self.embedded_vector_dir = embedded_vector_dir
        self.embedded_vector_name = embedded_vector_name
        self.processed_dataset_dir = processed_dataset_dir
        self.low_freq_dataset_dir = low_freq_dataset_dir
        self._data_files: list[GanDatasetItem] = []
        self.oss_dataset = oss_dataset
        # if processed_dataset_dir is not None:
        #     self.build_from_dir(
        #         processed_dataset_dir,
        #         low_freq_dataset_dir=low_freq_dataset_dir,
        #         embedded_vector_dir=embedded_vector_dir,
        #     )
        if oss_dataset is not None:
            self.build_from_oss_dataset(oss_dataset)
        else:
            raise ValueError("Either 'processed_dataset_dir' or 'oss_dataset' must be provided.")


    def build_from_dir(
        self,
        processed_dataset_dir: Path,
        low_freq_dataset_dir: Path | None = None,
        embedded_vector_dir: Path | None = None,
        num_workers: int | None = None,
    ) -> None:
        self._data_files = []

        processed_dataset_dir = Path(processed_dataset_dir)
        low_freq_dataset_dir = Path(low_freq_dataset_dir) if low_freq_dataset_dir is not None else None
        embedded_vector_dir = Path(embedded_vector_dir) if embedded_vector_dir is not None else processed_dataset_dir

        if not processed_dataset_dir.exists():
            LOGGER.warning("Processed GAN dataset directory does not exist: %s", processed_dataset_dir)
            return
        if low_freq_dataset_dir is not None and not low_freq_dataset_dir.exists():
            LOGGER.warning("Low-frequency GAN dataset directory does not exist: %s", low_freq_dataset_dir)
            return
        if embedded_vector_dir is not None and not embedded_vector_dir.exists():
            LOGGER.warning("Embedded vector directory does not exist: %s", embedded_vector_dir)
            return

        speaker_dirs = [speaker_dir for speaker_dir in sorted(processed_dataset_dir.iterdir()) if speaker_dir.is_dir()]
        if not speaker_dirs:
            LOGGER.warning("No speaker directories found in processed GAN dataset directory: %s", processed_dataset_dir)
            return

        def _build_items_for_speaker(speaker_dir: Path) -> list[GanDatasetItem]:
            speaker_id = speaker_dir.name
            embedded_path = embedded_vector_dir / speaker_id / self.embedded_vector_name
            if not embedded_path.exists():
                LOGGER.warning("Missing embedded vector for speaker %s: %s", speaker_id, embedded_path)
                return []

            sample_root_dir = speaker_dir
            low_freq_root_dir = low_freq_dataset_dir / speaker_id if low_freq_dataset_dir is not None else speaker_dir
            if low_freq_dataset_dir is not None and not low_freq_root_dir.exists():
                LOGGER.warning("Missing low-frequency directory for speaker %s: %s", speaker_id, low_freq_root_dir)
                return []

            sample_files = [
                file_path
                for file_path in sorted(sample_root_dir.iterdir())
                if file_path.is_file()
                and file_path.suffix == ".pt"
                and file_path.name != self.embedded_vector_name
                and not file_path.name.endswith("_low.pt")
            ]

            speaker_items: list[GanDatasetItem] = []
            for sample_path in sample_files:
                low_freq_sample_path = low_freq_root_dir / f"{sample_path.stem}_low.pt"
                if not low_freq_sample_path.exists():
                    LOGGER.warning(
                        "Missing low-frequency sample for speaker %s, sample %s: %s",
                        speaker_id,
                        sample_path.name,
                        low_freq_sample_path,
                    )
                    continue
                speaker_items.append(
                    GanDatasetItem(
                        speaker_id=speaker_id,
                        embedded_path=embedded_path,
                        sample_path=sample_path,
                        low_freq_sample_path=low_freq_sample_path,
                    )
                )
            return speaker_items

        cpu_count = os.cpu_count() or 1
        max_workers = num_workers if num_workers is not None else min(32, max(1, cpu_count))
        max_workers = min(max_workers, max(1, len(speaker_dirs)))

        if max_workers == 1:
            for speaker_dir in speaker_dirs:
                self._data_files.extend(_build_items_for_speaker(speaker_dir))
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for speaker_items in executor.map(_build_items_for_speaker, speaker_dirs):
                self._data_files.extend(speaker_items)


    def build_from_oss_dataset(self, oss_dataset_obj=None):
        if oss_dataset_obj is None:
            raise ValueError("OSS dataset object is None")

        required_keys = ("sample", "low_freq", "embedding")
        for key in required_keys:
            if key not in oss_dataset_obj:
                raise KeyError(f"OSS GAN dataset is missing required key: {key}")

        sample_length = len(oss_dataset_obj["sample"])
        low_freq_length = len(oss_dataset_obj["low_freq"])
        embedding_length = len(oss_dataset_obj["embedding"])
        if sample_length <= 0:
            raise RuntimeError("OSS GAN sample dataset is empty.")
        if low_freq_length <= 0:
            raise RuntimeError("OSS GAN low-frequency dataset is empty.")
        if embedding_length <= 0:
            raise RuntimeError("OSS GAN embedding dataset is empty.")
        if sample_length != low_freq_length or sample_length != embedding_length:
            raise ValueError(
                "OSS GAN dataset lengths do not match across sample, low_freq, and embedding.")

        self.oss_dataset = oss_dataset_obj
        return self


    def __len__(self) -> int:
        if self.oss_dataset is not None:
            try:
                return len(self.oss_dataset["sample"])
            except KeyError as exc:
                raise KeyError("OSS GAN dataset is missing required key: sample") from exc
        return len(self._data_files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.oss_dataset is not None:
            try:
                return {
                    "embedding": self.oss_dataset["embedding"][idx],
                    "sample": self.oss_dataset["sample"][idx],
                    "low_freq": self.oss_dataset["low_freq"][idx],
                }
            except KeyError as exc:
                raise KeyError(f"OSS GAN dataset is missing required key: {exc.args[0]}") from exc

        if not self._data_files:
            raise RuntimeError("GanDataset is empty and has no OSS dataset backing.")

        item = self._data_files[idx]
        return {
            "embedding": torch.load(item.embedded_path, map_location="cpu"),
            "sample": torch.load(item.sample_path, map_location="cpu"),
            "low_freq": torch.load(item.low_freq_sample_path, map_location="cpu"),
        }

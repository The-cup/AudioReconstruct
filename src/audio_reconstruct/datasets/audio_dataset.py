from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_MODULE_DIR = Path(__file__).resolve().parent
RAW_DATASETS_DIR = DATASET_MODULE_DIR / "raw"
PROCESSED_DATASETS_DIR = DATASET_MODULE_DIR / "processed"
# RAW_LIBRISPEECH_DIR = PROJECT_ROOT / "data" / "raw" / "LibriSpeech"
# PROCESSED_LIBRISPEECH_DIR = PROJECT_ROOT / "data" / "processed" / "LibriSpeech"
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


class LibriSpeechRawDataset(AudioReconstructionDataset):
    """Raw LibriSpeech dataset reader.

    Each item exposes the original utterance metadata so the preprocessing
    pipeline can derive log-mel features from the source audio.
    """

    def __init__(
        self,
        base_dir: str | Path = RAW_DATASETS_DIR,
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


class LibriSpeechDataset(AudioReconstructionDataset):
    """Processed LibriSpeech dataset.

    Items are stored as log-mel feature tensors on disk and indexed by a
    metadata file for efficient loading.
    """

    def __init__(
        self,
        base_dir: str | Path = PROCESSED_DATASETS_DIR,
        dataset_sub_name: str | None = None,
    ) -> None:
        super().__init__()
        if dataset_sub_name is not None:
            self._dataset_dir = Path(base_dir) / dataset_sub_name
        else:
            self._dataset_dir = Path(base_dir)

        self._data_files = []
        if self._dataset_dir.exists():
            self._data_files = self._scan_for_tensor_files()

    def _scan_for_tensor_files(self) -> list[LibriSpeechProcessedItem]:
        data_files: list[LibriSpeechProcessedItem] = []
        with (self._dataset_dir / "transcript.txt").open("r", encoding="utf-8") as f:
            for line in f:
                label, text = line.strip().split(" ", maxsplit=1)
                data_files.append(
                    LibriSpeechProcessedItem(
                        label=label,
                        file=os.path.join(self._dataset_dir, f"{label}.pt"),
                        text=text,
                    )
                )
        return data_files

    def __len__(self) -> int:
        return len(self._data_files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self._data_files[idx]
        file = torch.load(item.file, map_location="cpu")
        return {
            "label": item.label,    # 文件名（无后缀）
            "file": file,           # 文件内容，torch格式
            "text": item.text,      # 文件内容对应文本
            "path": Path(item.file).absolute(),  # 文件路径
        }


@dataclass(slots=True)
class SpkEncDatasetItem:
    speaker_id: str
    file_path: Path


class SpkEncDataset(AudioReconstructionDataset):
    """Dataset for speaker embedding training."""
    def __init__(
        self,
        processed_dataset_dir: Path | None = None,
        randomize: bool = True,
    ) -> None:
        super().__init__()
        self._randomize = randomize
        self._data_files: [SpkEncDatasetItem] = []

        if processed_dataset_dir is not None:
            self.build_from_dir(processed_dataset_dir)

    def build_from_dir(self, processed_dataset_dir: Path):
        for speaker_dir in processed_dataset_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
            speaker_id = speaker_dir.name
            for file_path in speaker_dir.iterdir():
                if file_path.suffix != ".pt":
                    continue
                self._data_files.append(SpkEncDatasetItem(speaker_id=speaker_id, file_path=file_path))

    def add_item(self, speaker_id: str, file_path: Path):
        self._data_files.append(SpkEncDatasetItem(speaker_id=speaker_id, file_path=file_path))

    def __len__(self) -> int:
        return len(self._data_files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self._data_files[idx]
        file = torch.load(item.file_path, map_location="cpu")
        return {
            "speaker_id": item.speaker_id,
            "file": file,
        }

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
RAW_LIBRISPEECH_DIR = PROJECT_ROOT / "data" / "raw" / "LibriSpeech"
PROCESSED_LIBRISPEECH_DIR = PROJECT_ROOT / "data" / "dataset" / "LibriSpeech"
SUPPORTED_AUDIO_EXTENSIONS = (
    ".flac",
    ".wav",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
)
METADATA_FILENAME = "metadata.jsonl"


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


class AudioReconstructionDataset(Dataset):
    """Base dataset for the audio reconstruction project."""

    _data_files: list[Any]

    def __iter__(self) -> Iterator[Any]:
        for index in range(len(self)):
            yield self[index]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={len(self)})"


def _find_audio_file(directory: Path, stem: str) -> Path | None:
    for extension in SUPPORTED_AUDIO_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.exists():
            return candidate

    for candidate in directory.glob(f"{stem}.*"):
        if candidate.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            return candidate
    return None


def _read_metadata_index(index_file: Path) -> list[LibriSpeechProcessedItem]:
    data_files: list[LibriSpeechProcessedItem] = []
    if not index_file.exists():
        return data_files

    with index_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            data_files.append(
                LibriSpeechProcessedItem(
                    label=record["label"],
                    file=record["file"],
                )
            )
    return data_files


class LibriSpeechRawDataset(AudioReconstructionDataset):
    """Raw LibriSpeech dataset reader.

    Each item exposes the original utterance metadata so the preprocessing
    pipeline can derive log-mel features from the source audio.
    """

    def __init__(
        self,
        base_dir: str | Path = RAW_LIBRISPEECH_DIR,
        dataset_sub_name: str = "train-clean-100",
    ) -> None:
        super().__init__()
        self._dataset_dir = Path(base_dir) / dataset_sub_name
        self._data_files: list[LibriSpeechRawItem] = []

        if not self._dataset_dir.exists():
            LOGGER.warning("Raw LibriSpeech directory does not exist: %s", self._dataset_dir)
            return

        transcript_files = sorted(self._dataset_dir.rglob("*.trans.txt"))
        for transcript_file in transcript_files:
            chapter_dir = transcript_file.parent
            if chapter_dir == self._dataset_dir:
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
        base_dir: str | Path = PROCESSED_LIBRISPEECH_DIR,
        dataset_sub_name: str = "train-clean-100",
    ) -> None:
        super().__init__()
        self._dataset_dir = Path(base_dir) / dataset_sub_name
        self._index_file = self._dataset_dir / METADATA_FILENAME

        self._data_files = _read_metadata_index(self._index_file)
        if not self._data_files and self._dataset_dir.exists():
            self._data_files = self._scan_for_tensor_files()

    def _scan_for_tensor_files(self) -> list[LibriSpeechProcessedItem]:
        data_files: list[LibriSpeechProcessedItem] = []
        for tensor_file in sorted(self._dataset_dir.rglob("*.pt")):
            relative_file = os.path.relpath(tensor_file, DATASET_MODULE_DIR)
            data_files.append(
                LibriSpeechProcessedItem(
                    label=tensor_file.stem.split("__", maxsplit=1)[0],
                    file=relative_file,
                )
            )
        return data_files

    def __len__(self) -> int:
        return len(self._data_files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self._data_files[idx]
        feature_path = (DATASET_MODULE_DIR / item.file).resolve()
        feature = torch.load(feature_path, map_location="cpu")
        return {
            "feature": feature,
            "label": item.label,
            "file": item.file,
            "path": str(feature_path),
        }

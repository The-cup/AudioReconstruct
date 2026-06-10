from __future__ import annotations

from array import array
import math
from pathlib import Path
import wave

import pytest


SAMPLE_RATE = 16_000
DURATION_SECONDS = 5
TOTAL_SAMPLES = SAMPLE_RATE * DURATION_SECONDS
LABELS = ("group0", "group1", "group2")
OCTAVE_STARTS = {
    "group0": 48,  # C3-B3
    "group1": 60,  # C4-B4
    "group2": 72,  # C5-B5
}


def _midi_to_frequency(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def _write_discrete_tone_wav(path: Path, midi_start: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    note_lengths = [TOTAL_SAMPLES // 12] * 12
    for index in range(TOTAL_SAMPLES % 12):
        note_lengths[index] += 1

    amplitude = int(0.45 * 32767)
    samples = array("h")
    sample_index = 0

    for note_offset, note_length in enumerate(note_lengths):
        frequency = _midi_to_frequency(midi_start + note_offset)
        for _ in range(note_length):
            value = int(
                amplitude
                * math.sin(2.0 * math.pi * frequency * (sample_index / SAMPLE_RATE))
            )
            samples.append(max(-32767, min(32767, value)))
            sample_index += 1

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(samples.tobytes())


def _build_fake_librispeech_corpus(tmp_path: Path) -> tuple[Path, dict[str, dict[str, Path | str]]]:
    raw_dataset_root = tmp_path / "data" / "raw" / "LibriSpeech" / "train-clean-100"
    expected: dict[str, dict[str, Path | str]] = {}

    for index, label in enumerate(LABELS, start=1):
        speaker_id = f"{index:04d}"
        chapter_id = f"{index:04d}"
        chapter_dir = raw_dataset_root / speaker_id / chapter_id
        audio_path = chapter_dir / f"{label}.wav"
        transcript_path = chapter_dir / f"{speaker_id}-{chapter_id}.trans.txt"
        text = "synthetic tone sequence"

        _write_discrete_tone_wav(audio_path, OCTAVE_STARTS[label])
        transcript_path.write_text(f"{label} {text}\n", encoding="utf-8")

        expected[label] = {
            "speaker_id": speaker_id,
            "chapter_id": chapter_id,
            "file": audio_path,
            "text": text,
        }

    return raw_dataset_root.parent, expected


@pytest.fixture()
def fake_librispeech_corpus(tmp_path: Path) -> tuple[Path, dict[str, dict[str, Path | str]]]:
    """Create a synthetic LibriSpeech-style corpus for dataset tests."""
    return _build_fake_librispeech_corpus(tmp_path)


@pytest.fixture()
def prepared_librispeech_datasets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    """Create raw and processed LibriSpeech datasets for model and dataset tests."""
    pytest.importorskip("torch")
    pytest.importorskip("torchaudio")

    from audio_reconstruct.data.preprocess import pipeline as preprocess_pipeline
    from audio_reconstruct.datasets.audio_dataset import (
        DATASET_MODULE_DIR,
        LibriSpeechDataset,
        LibriSpeechRawDataset,
    )
    from audio_reconstruct.tasks.collect_and_preprocess import preprocess_data

    raw_base_dir, expected = _build_fake_librispeech_corpus(tmp_path)
    processed_base_dir = tmp_path / "data" / "dataset" / "LibriSpeech"
    processed_dataset_dir = processed_base_dir / "train-clean-100"

    monkeypatch.setattr(preprocess_pipeline, "PROCESSED_LIBRISPEECH_ROOT", processed_base_dir)
    monkeypatch.setattr(preprocess_pipeline, "DEFAULT_DATASET_SUB_NAME", "train-clean-100")

    raw_dataset = LibriSpeechRawDataset(base_dir=raw_base_dir, dataset_sub_name="train-clean-100")
    processed_dataset = preprocess_data(
        dataset=raw_dataset,
        save_dir=processed_dataset_dir,
    )
    reloaded_processed_dataset = LibriSpeechDataset(
        base_dir=processed_base_dir,
        dataset_sub_name="train-clean-100",
    )
    return {
        "raw_dataset": raw_dataset,
        "processed_dataset": processed_dataset,
        "reloaded_processed_dataset": reloaded_processed_dataset,
        "raw_base_dir": raw_base_dir,
        "processed_base_dir": processed_base_dir,
        "processed_dataset_dir": processed_dataset_dir,
        "expected": expected,
        "dataset_module_dir": DATASET_MODULE_DIR,
    }


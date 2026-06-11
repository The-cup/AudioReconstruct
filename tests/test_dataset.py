from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from audio_reconstruct.datasets import AudioReconstructionDataset


def test_librispeech_raw_dataset_reads_fake_audio(
    fake_librispeech_corpus,
) -> None:
    pytest.importorskip("torch")

    from audio_reconstruct.datasets.audio_dataset import LibriSpeechRawDataset

    raw_base_dir, expected = fake_librispeech_corpus
    dataset = LibriSpeechRawDataset(base_dir=raw_base_dir, dataset_sub_name="train-clean-100")

    assert len(dataset) == 3

    items_by_label = {item.label: item for item in dataset}
    assert set(items_by_label) == {"group0", "group1", "group2"}

    for label, expected_item in expected.items():
        item = items_by_label[label]
        assert item.speaker_id == expected_item["speaker_id"]
        assert item.chapter_id == expected_item["chapter_id"]
        assert item.file == str(expected_item["file"])
        assert item.text == expected_item["text"]
        assert Path(item.file).exists()


def test_librispeech_dataset_processing_and_save(
    prepared_librispeech_datasets,
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchaudio")

    from audio_reconstruct.datasets.audio_dataset import DATASET_MODULE_DIR, LibriSpeechDataset

    processed_dataset = prepared_librispeech_datasets["processed_dataset"]
    reloaded_processed_dataset = prepared_librispeech_datasets["reloaded_processed_dataset"]
    processed_base_dir = prepared_librispeech_datasets["processed_base_dir"]
    expected = prepared_librispeech_datasets["expected"]

    assert isinstance(processed_dataset, LibriSpeechDataset)
    assert processed_dataset._dataset_dir == processed_base_dir
    assert isinstance(processed_base_dir, Path)
    assert processed_base_dir.exists()
    assert (processed_base_dir / "transcript.txt").exists()
    assert len(list(processed_base_dir.rglob("*.pt"))) == 18
    assert len(processed_dataset) == 18

    first_item = processed_dataset[0]
    assert first_item["file"].shape == (160, 40)
    assert Path(first_item["path"]).exists()

    for item in processed_dataset:
        assert item["file"].shape == (160, 40)
        assert Path(item["path"]).exists()

    assert isinstance(reloaded_processed_dataset, AudioReconstructionDataset)
    assert len(reloaded_processed_dataset) == len(processed_dataset)
    assert [item["path"] for item in reloaded_processed_dataset] == [
        item["path"] for item in processed_dataset
    ]

    for label, expected_item in expected.items():
        assert expected_item["file"].exists()


def test_collect_data_wrapper_uses_raw_loader(
    fake_librispeech_corpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")

    from audio_reconstruct.datasets.audio_dataset import LibriSpeechRawDataset
    from audio_reconstruct.tasks import collect_and_preprocess as collect_task

    raw_base_dir, _ = fake_librispeech_corpus
    raw_dataset = LibriSpeechRawDataset(base_dir=raw_base_dir, dataset_sub_name="train-clean-100")
    monkeypatch.setattr(
        collect_task,
        "load_raw_data",
        lambda dataset_name="LibriSpeech", dataset_sub_name="train-clean-100": raw_dataset
    )

    loaded_dataset = collect_task.collect_data()
    assert loaded_dataset is raw_dataset


from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest


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
    processed_dataset_dir = prepared_librispeech_datasets["processed_dataset_dir"]
    expected = prepared_librispeech_datasets["expected"]

    assert isinstance(processed_dataset, LibriSpeechDataset)
    assert processed_dataset._dataset_dir == processed_dataset_dir
    assert processed_dataset_dir.exists()
    assert (processed_dataset_dir / "metadata.jsonl").exists()
    assert len(list(processed_dataset_dir.rglob("*.pt"))) == 18
    assert len(processed_dataset) == 18

    label_counts = Counter(item["label"] for item in processed_dataset)
    assert label_counts == Counter({"group0": 6, "group1": 6, "group2": 6})

    first_item = processed_dataset[0]
    assert first_item["feature"].shape == (160, 40)
    assert Path(first_item["path"]).exists()
    assert (DATASET_MODULE_DIR / first_item["file"]).resolve() == Path(first_item["path"]).resolve()

    for item in processed_dataset:
        assert item["label"] in {"group0", "group1", "group2"}
        assert item["feature"].shape == (160, 40)
        assert Path(item["path"]).exists()
        assert (DATASET_MODULE_DIR / item["file"]).resolve() == Path(item["path"]).resolve()

    assert len(reloaded_processed_dataset) == len(processed_dataset)
    assert [item["file"] for item in reloaded_processed_dataset] == [
        item["file"] for item in processed_dataset
    ]
    assert {item["label"] for item in reloaded_processed_dataset} == {"group0", "group1", "group2"}

    for label in expected:
        expected_file_name = f"{label}__seg0000.pt"
        assert any(expected_file_name in item["file"] for item in processed_dataset)

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
        lambda dataset_name="LibriSpeech": raw_dataset,
    )

    loaded_dataset = collect_task.collect_data()
    assert loaded_dataset is raw_dataset


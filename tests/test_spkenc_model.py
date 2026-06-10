from __future__ import annotations

import pytest


def test_spkenc_model_registry_returns_encoder() -> None:
    torch = pytest.importorskip("torch")
    _ = torch

    from audio_reconstruct.models.custom.spkenc import SpkEnc
    from audio_reconstruct.models.registry import get_model

    model = get_model("spkenc")
    assert isinstance(model, SpkEnc)
    assert model.hidden_dim == 256
    assert model.embedding_dim == 256
    assert model.num_layers == 6


def test_spkenc_reads_processed_dataset_and_embeds(
    prepared_librispeech_datasets,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")

    from audio_reconstruct.datasets.audio_dataset import LibriSpeechDataset
    from audio_reconstruct.models.registry import get_model

    reloaded_processed_dataset = prepared_librispeech_datasets["reloaded_processed_dataset"]
    processed_dataset_dir = prepared_librispeech_datasets["processed_dataset_dir"]

    assert isinstance(reloaded_processed_dataset, LibriSpeechDataset)
    assert len(reloaded_processed_dataset) == 18

    sample = reloaded_processed_dataset[0]
    assert sample["feature"].shape == (160, 40)
    assert sample["label"] in {"group0", "group1", "group2"}
    assert sample["path"].startswith(str(processed_dataset_dir))

    model = get_model("spkenc")
    model.eval()
    with torch.no_grad():
        embedding = model(sample["feature"].unsqueeze(0))

    assert embedding.shape == (1, 256)
    assert torch.allclose(embedding.norm(dim=-1), torch.ones(1), atol=1e-5)


def test_spkenc_training_validation_and_test_pipeline(
    prepared_librispeech_datasets,
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    pytest.importorskip("tqdm")
    pytest.importorskip("tensorboard")

    from audio_reconstruct.datasets.dataset_builder import build_dataset
    from audio_reconstruct.models.registry import get_model
    from audio_reconstruct.ml.test import test as run_test
    from audio_reconstruct.ml.train import train
    from audio_reconstruct.ml.validate import validate

    processed_dataset = prepared_librispeech_datasets["processed_dataset"]

    train_dataset, val_dataset, test_dataset = build_dataset(
        processed_dataset,
        train_ratio=0.8,
        val_ratio=0.1,
    )

    model = get_model("spkenc")
    weights_path = tmp_path / "artifacts" / "checkpoints" / "spkenc.pt"
    log_dir = tmp_path / "artifacts" / "tensorboard"

    trained_model = train(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        weights_path=weights_path,
        epochs=1,
        batch_size=3,
        utterances_per_speaker=2,
        steps_per_epoch=1,
        learning_rate=1e-4,
        device="cpu",
        log_dir=log_dir,
    )

    assert trained_model is model
    assert weights_path.exists()

    validation_result = validate(
        trained_model,
        val_dataset,
        batch_size=3,
        utterances_per_speaker=2,
        device="cpu",
    )
    test_result = run_test(
        trained_model,
        test_dataset,
        batch_size=3,
        utterances_per_speaker=2,
        device="cpu",
    )

    assert validation_result["num_batches"] > 0
    assert test_result["num_batches"] > 0
    assert validation_result["average_loss"] >= 0
    assert test_result["average_loss"] >= 0


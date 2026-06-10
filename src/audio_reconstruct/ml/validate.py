from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import Dataset

from audio_reconstruct.ml.train import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_UTTERANCES_PER_SPEAKER,
    _evaluate_dataset,
    _evaluate_voice_expand_gan,
    _is_voice_expand_gan,
)
from audio_reconstruct.models.custom.ge2e_loss import GE2ELoss


__test__ = False


def validate(
    model: nn.Module,
    validation_dataset: Dataset,
    loss_fn: nn.Module | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    utterances_per_speaker: int = DEFAULT_UTTERANCES_PER_SPEAKER,
    device: torch.device | str | None = None,
) -> dict[str, float | int]:
    """Evaluate a model on a validation dataset.

    Args:
        model: PyTorch model to evaluate.
        validation_dataset: Validation dataset inheriting from Dataset.
        loss_fn: Loss function, defaults to GE2ELoss.
        batch_size: Number of speakers per batch.
        utterances_per_speaker: Number of utterances per speaker in each batch.
        device: Device used for evaluation.

    Returns:
        Aggregated validation metrics.
    """
    if loss_fn is None:
        loss_fn = GE2ELoss()
    if not isinstance(validation_dataset, Dataset):
        raise TypeError("validation_dataset must inherit from torch.utils.data.Dataset.")

    resolved_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(resolved_device)
    loss_fn = loss_fn.to(resolved_device)

    if _is_voice_expand_gan(model):
        return _evaluate_voice_expand_gan(
            model=model,
            dataset=validation_dataset,
            batch_size=batch_size,
            device=resolved_device,
        )

    return _evaluate_dataset(
        model=model,
        dataset=validation_dataset,
        loss_fn=loss_fn,
        speakers_per_batch=batch_size,
        utterances_per_speaker=utterances_per_speaker,
        device=resolved_device,
    )


def validate_model(
    model: nn.Module,
    validation_dataset: Dataset,
    loss_fn: nn.Module | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    utterances_per_speaker: int = DEFAULT_UTTERANCES_PER_SPEAKER,
    device: torch.device | str | None = None,
) -> dict[str, float | int]:
    """Backward-compatible alias for validate()."""
    return validate(
        model=model,
        validation_dataset=validation_dataset,
        loss_fn=loss_fn,
        batch_size=batch_size,
        utterances_per_speaker=utterances_per_speaker,
        device=device,
    )

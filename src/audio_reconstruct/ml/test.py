from __future__ import annotations

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

from audio_reconstruct.ml.train import _evaluate_dataset, _evaluate_voice_expand_gan, _is_voice_expand_gan
from audio_reconstruct.models.custom.ge2e_loss import GE2ELoss


__test__ = False


LOGGER = logging.getLogger(__name__)


def _resolve_device(device: torch.device | str | None) -> torch.device:
    return torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def _validate_test_inputs(test_dataloader: DataLoader) -> None:
    if not isinstance(test_dataloader, DataLoader):
        raise TypeError("test_dataloader must be a torch.utils.data.DataLoader.")
    if len(test_dataloader) == 0:
        raise ValueError("test_dataloader must not be empty.")


def _set_dataloader_epoch(dataloader: DataLoader, epoch: int) -> None:
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)


def _format_metric_value(value: float | int) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _log_metrics(title: str, metrics: dict[str, float | int]) -> None:
    preferred_order = (
        "average_loss",
        "average_generator_loss",
        "average_discriminator_loss",
        "num_batches",
        "num_speakers",
        "num_utterances",
        "num_samples",
    )
    ordered_keys = [key for key in preferred_order if key in metrics]
    ordered_keys.extend(key for key in metrics.keys() if key not in ordered_keys)
    ordered_items = ", ".join(
        f"{key}={_format_metric_value(metrics[key])}" for key in ordered_keys
    )
    message = f"{title}: {ordered_items}"
    LOGGER.info(message)
    print(message)


def test(
    model: nn.Module,
    test_dataloader: DataLoader,
    loss_fn: nn.Module | None = None,
    device: torch.device | str | None = None,
) -> dict[str, float | int]:
    """Evaluate a model on a test dataloader."""
    if loss_fn is None:
        loss_fn = GE2ELoss()

    _validate_test_inputs(test_dataloader)

    resolved_device = _resolve_device(device)
    _set_dataloader_epoch(test_dataloader, 0)
    model = model.to(resolved_device)
    loss_fn = loss_fn.to(resolved_device)

    if _is_voice_expand_gan(model):
        metrics = _evaluate_voice_expand_gan(
            model=model,
            dataset=test_dataloader,
            device=resolved_device,
        )
        _log_metrics("Test results", metrics)
        return metrics

    metrics = _evaluate_dataset(
        model=model,
        dataset=test_dataloader,
        loss_fn=loss_fn,
        device=resolved_device,
    )
    _log_metrics("Test results", metrics)
    return metrics


def test_model(
    model: nn.Module,
    test_dataloader: DataLoader,
    loss_fn: nn.Module | None = None,
    device: torch.device | str | None = None,
) -> dict[str, float | int]:
    """Backward-compatible alias for test()."""
    return test(
        model=model,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        device=device,
    )

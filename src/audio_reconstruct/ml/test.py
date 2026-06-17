from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from audio_reconstruct.ml.train import _evaluate_dataset, _evaluate_voice_expand_gan, _is_voice_expand_gan
from audio_reconstruct.models.custom.ge2e_loss import GE2ELoss


__test__ = False


def test(
    model: nn.Module,
    test_dataloader: DataLoader,
    loss_fn: nn.Module | None = None,
    device: torch.device | str | None = None,
) -> dict[str, float | int]:
    """Evaluate a model on a test dataloader."""
    if loss_fn is None:
        loss_fn = GE2ELoss()
    if not isinstance(test_dataloader, DataLoader):
        raise TypeError("test_dataloader must be a torch.utils.data.DataLoader.")

    resolved_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(resolved_device)
    loss_fn = loss_fn.to(resolved_device)

    if _is_voice_expand_gan(model):
        return _evaluate_voice_expand_gan(
            model=model,
            dataset=test_dataloader,
            device=resolved_device,
        )

    return _evaluate_dataset(
        model=model,
        dataset=test_dataloader,
        loss_fn=loss_fn,
        device=resolved_device,
    )


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

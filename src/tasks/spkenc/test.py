from __future__ import annotations

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

from tasks.spkenc.train import evaluate_spkenc
from models.custom.ge2e_loss import GE2ELoss


LOGGER = logging.getLogger(__name__)


def test_spkenc(
    model: nn.Module,
    test_dataloader: DataLoader,
    loss_fn: nn.Module | None = None,
    device: torch.device | str | None = None,
) -> dict[str, float | int]:
    if loss_fn is None:
        loss_fn = GE2ELoss()

    metrics = evaluate_spkenc(
        model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        device=device,
    )
    LOGGER.info("Test results: %s", metrics)
    return metrics

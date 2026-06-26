from __future__ import annotations

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

from tasks.gan.train import evaluate_gan_model


LOGGER = logging.getLogger(__name__)


def evaluate_gan(
    model: nn.Module,
    test_dataloader: DataLoader,
    device: torch.device | str | None = None,
    seed: int = 42,
) -> dict[str, float | int]:
    """Evaluate a GAN model on the test set and return output statistics."""
    metrics = evaluate_gan_model(
        model=model,
        dataloader=test_dataloader,
        device=device,
        seed=seed,
    )
    LOGGER.info("Test results: %s", metrics)
    return metrics

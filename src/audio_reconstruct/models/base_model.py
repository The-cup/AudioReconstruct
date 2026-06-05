from torch import nn


class BaseAudioReconstructionModel(nn.Module):
    """Base class for custom PyTorch models."""

    def forward(self, x):  # type: ignore[override]
        raise NotImplementedError("Model forward pass is not implemented yet.")


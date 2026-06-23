from __future__ import annotations

from torch import Tensor, nn
import torch.nn.functional as F

from models.base_model import BaseAudioReconstructionModel


class SpkEnc(BaseAudioReconstructionModel):
    """Speaker encoder for 40-band mel spectrogram inputs.

    Expected input shape:
        (batch_size, 160, 40)
    """

    def __init__(
        self,
        input_dim: int = 40,
        hidden_dim: int = 256,
        num_layers: int = 6,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_dim, embedding_dim)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """
        Args:
            x: Mel spectrogram tensor with shape (batch, 160, 40).

        Returns:
            L2-normalized speaker embedding with shape (batch, 256).
        """
        _, (hidden_state, _) = self.encoder(x)
        last_layer_hidden = hidden_state[-1]
        embedding = self.projection(last_layer_hidden)
        embedding = self.activation(embedding)
        return F.normalize(embedding, p=2, dim=-1)


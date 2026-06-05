from __future__ import annotations

from torch import Tensor, nn
import torch
import torch.nn.functional as F


class GE2ELoss(nn.Module):
    """Generalized End-to-End (GE2E) speaker verification loss.

    Expected input shape:
        embeddings: (num_speakers, num_utterances, embedding_dim)

    The implementation follows the GE2E idea from the reference code, but is
    vectorized, device-agnostic, and uses learnable similarity scale/bias.
    """

    def __init__(self, init_w: float = 10.0, init_b: float = -5.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor(float(init_w)))
        self.b = nn.Parameter(torch.tensor(float(init_b)))
        self.eps = eps

    def _validate_input(self, embeddings: Tensor) -> tuple[int, int, int]:
        if embeddings.dim() != 3:
            raise ValueError(
                "GE2ELoss expects embeddings with shape "
                "(num_speakers, num_utterances, embedding_dim)."
            )

        num_speakers, num_utterances, embedding_dim = embeddings.shape
        if num_speakers < 1:
            raise ValueError("num_speakers must be at least 1.")
        if num_utterances < 2:
            raise ValueError(
                "GE2ELoss requires at least 2 utterances per speaker to compute "
                "exclusive centroids."
            )
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1.")
        return num_speakers, num_utterances, embedding_dim

    def compute_similarity_matrix(self, embeddings: Tensor) -> Tensor:
        """Build the GE2E similarity matrix.

        Returns:
            Tensor of shape (num_speakers, num_utterances, num_speakers)
        """
        num_speakers, num_utterances, _ = self._validate_input(embeddings)

        # Normalize the input embeddings first for stable cosine similarity.
        embeddings = F.normalize(embeddings, p=2, dim=-1, eps=self.eps)

        # Inclusive centroids: mean over all utterances for each speaker.
        centroids = embeddings.mean(dim=1, keepdim=True)
        centroids = F.normalize(centroids, p=2, dim=-1, eps=self.eps)

        # Exclusive centroids: leave-one-out mean for the same speaker.
        summed = embeddings.sum(dim=1, keepdim=True)
        exclusive_centroids = (summed - embeddings) / float(num_utterances - 1)
        exclusive_centroids = F.normalize(
            exclusive_centroids, p=2, dim=-1, eps=self.eps
        )

        # Cosine similarity between each utterance and every speaker centroid.
        # Shape: (S, U, S)
        similarity = torch.einsum("sud,td->sut", embeddings, centroids.squeeze(1))

        # Replace diagonal entries with the leave-one-out similarity.
        speaker_ids = torch.arange(num_speakers, device=embeddings.device)
        diagonal_similarity = (embeddings * exclusive_centroids).sum(dim=-1)
        similarity[speaker_ids, :, speaker_ids] = diagonal_similarity

        # Learnable scale and bias from the original GE2E formulation.
        scale = torch.clamp(self.w, min=self.eps)
        similarity = similarity * scale + self.b
        return similarity

    def forward(self, embeddings: Tensor) -> Tensor:  # type: ignore[override]
        """Compute the GE2E loss.

        Args:
            embeddings: Tensor with shape (S, U, D).

        Returns:
            Scalar loss tensor.
        """
        similarity = self.compute_similarity_matrix(embeddings)
        num_speakers, num_utterances, _ = similarity.shape

        # Each utterance should classify to its own speaker index.
        logits = similarity.reshape(num_speakers * num_utterances, num_speakers)
        targets = torch.arange(num_speakers, device=embeddings.device).repeat_interleave(
            num_utterances
        )
        return F.cross_entropy(logits, targets)


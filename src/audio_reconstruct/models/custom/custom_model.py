from audio_reconstruct.models.base_model import BaseAudioReconstructionModel


class CustomAudioReconstructionModel(BaseAudioReconstructionModel):
    """Template for user-defined model structures."""

    def forward(self, x):  # type: ignore[override]
        raise NotImplementedError("Custom model structure is not implemented yet.")

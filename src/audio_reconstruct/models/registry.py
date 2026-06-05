"""Model registry for selecting model implementations."""

from audio_reconstruct.models.custom import SpkEnc


def get_model(name: str):
    """Return a model instance by name."""
    normalized_name = name.strip().lower()
    if normalized_name == "spkenc":
        return SpkEnc()
    raise ValueError(f"Unknown model name: {name}")

"""Model registry for selecting model implementations."""

from audio_reconstruct.models.custom import SpkEnc, VoiceExpandGAN


def get_model(name: str):
    """Return a model instance by name."""
    normalized_name = name.strip().lower()
    if normalized_name == "spkenc":
        return SpkEnc()
    elif normalized_name == "voice_expand_gan":
        return VoiceExpandGAN()
    raise ValueError(f"Unknown model name: {name}")

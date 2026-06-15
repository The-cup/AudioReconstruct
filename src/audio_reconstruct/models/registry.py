"""Model registry for selecting model implementations."""

def get_model(name: str):
    """Return a model instance by name."""
    normalized_name = name.strip().lower()
    if normalized_name.lower() == "spkenc":
        from audio_reconstruct.models.custom import SpkEnc
        return SpkEnc()
    elif normalized_name.lower() == "voice_expand_gan":
        from audio_reconstruct.models.custom import VoiceExpandGAN
        return VoiceExpandGAN()
    raise ValueError(f"Unknown model name: {name}")

def get_loss_function(name: str):
    """Return a loss function instance by name."""
    normalized_name = name.strip().lower()
    if normalized_name.lower() == "ge2e":
        from audio_reconstruct.models.custom.ge2e_loss import GE2ELoss
        return GE2ELoss()
    raise ValueError(f"Unknown loss function name: {name}")

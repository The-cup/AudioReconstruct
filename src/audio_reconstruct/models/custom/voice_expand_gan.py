from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from audio_reconstruct.models.base_model import BaseAudioReconstructionModel


def _match_spatial_size(source: Tensor, reference: Tensor) -> Tensor:
    """Match source spatial size to the reference tensor.

    This keeps the generator robust when the input spectrogram dimensions are
    not perfectly divisible by 2^n.
    """
    if source.shape[-2:] == reference.shape[-2:]:
        return source
    return F.interpolate(source, size=reference.shape[-2:], mode="bilinear", align_corners=False)


def _stack_condition_and_target(condition: Tensor, target: Tensor | None = None) -> Tensor:
    """Build a 2-channel discriminator input."""
    if target is None:
        if condition.dim() != 4 or condition.shape[1] != 2:
            raise ValueError("Expected a 2-channel tensor when target is not provided.")
        return condition

    if condition.dim() != 4 or target.dim() != 4:
        raise ValueError("Condition and target tensors must be 4D tensors.")
    if condition.shape[0] != target.shape[0]:
        raise ValueError("Condition and target batch sizes must match.")
    if condition.shape[-2:] != target.shape[-2:]:
        raise ValueError("Condition and target spatial sizes must match.")
    return torch.cat([condition, target], dim=1)


class ConvBlk(nn.Module):
    """A compact convolutional block used by the generator."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.block(x)


class FTB(nn.Module):
    """Frequency-time block.

    A stable default implementation using asymmetric convolutions that keeps the
    spatial size unchanged while changing the channel width.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.block(x)


class UpConv(nn.Module):
    """Upsampling block used by the generator decoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.block(x)


class VoiceExpandGenerator(BaseAudioReconstructionModel):
    """U-Net style generator for speech spectrogram expansion."""

    def __init__(self) -> None:
        super().__init__()

        # Encoder
        self.blk_0 = ConvBlk(2, 64)
        self.blk_1_ftb = FTB(64, 128)
        self.blk_1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.blk_1_conv = ConvBlk(128, 128)

        self.blk_2_ftb = FTB(128, 64)
        self.blk_2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.blk_2_conv = ConvBlk(64, 256)

        self.blk_3_ftb = FTB(256, 32)
        self.blk_3_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.blk_3_conv = ConvBlk(32, 512)

        self.blk_4_ftb = FTB(512, 16)
        self.blk_4_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.blk_4_conv = ConvBlk(16, 1024)

        # Decoder
        self.blk_5_up = UpConv(1024, 512)
        self.blk_5_conv = ConvBlk(1024, 512)

        self.blk_6_up = UpConv(512, 256)
        self.blk_6_conv = ConvBlk(512, 256)

        self.blk_7_up = UpConv(256, 128)
        self.blk_7_conv = ConvBlk(256, 128)

        self.blk_8_up = UpConv(128, 64)
        self.blk_8_conv = ConvBlk(128, 64)

        self.blk_9 = nn.Conv2d(64, 1, kernel_size=1)

    def _encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        skip_0 = self.blk_0(x)
        x = skip_0

        x = self.blk_1_ftb(x)
        x = self.blk_1_pool(x)
        skip_1 = self.blk_1_conv(x)
        x = skip_1

        x = self.blk_2_ftb(x)
        x = self.blk_2_pool(x)
        skip_2 = self.blk_2_conv(x)
        x = skip_2

        x = self.blk_3_ftb(x)
        x = self.blk_3_pool(x)
        skip_3 = self.blk_3_conv(x)
        x = skip_3

        x = self.blk_4_ftb(x)
        x = self.blk_4_pool(x)
        x = self.blk_4_conv(x)
        return x, skip_0, skip_1, skip_2, skip_3

    def _decode(self, x: Tensor, skip_0: Tensor, skip_1: Tensor, skip_2: Tensor, skip_3: Tensor) -> Tensor:
        x = self.blk_5_up(x)
        x = _match_spatial_size(x, skip_3)
        x = self.blk_5_conv(torch.cat([x, skip_3], dim=1))

        x = self.blk_6_up(x)
        x = _match_spatial_size(x, skip_2)
        x = self.blk_6_conv(torch.cat([x, skip_2], dim=1))

        x = self.blk_7_up(x)
        x = _match_spatial_size(x, skip_1)
        x = self.blk_7_conv(torch.cat([x, skip_1], dim=1))

        x = self.blk_8_up(x)
        x = _match_spatial_size(x, skip_0)
        x = self.blk_8_conv(torch.cat([x, skip_0], dim=1))

        return self.blk_9(x)

    def forward(self, x: Tensor, noise: Tensor | None = None) -> Tensor:  # type: ignore[override]
        """Generate an expanded mel spectrogram.

        Args:
            x: Either a 2-channel stacked tensor or the low-frequency mel spectrogram.
            noise: Optional noise mel spectrogram to be stacked with x.

        Returns:
            Generated mel spectrogram with shape (batch, 1, height, width).
        """
        if noise is not None:
            x = _stack_condition_and_target(x, noise)
        elif x.dim() == 4 and x.shape[1] == 2:
            pass
        else:
            raise ValueError("Generator expects either a 2-channel tensor or (condition, noise) inputs.")

        x, skip_0, skip_1, skip_2, skip_3 = self._encode(x)
        return self._decode(x, skip_0, skip_1, skip_2, skip_3)


class VoiceExpandDiscriminator(BaseAudioReconstructionModel):
    """Patch-style discriminator for conditioned spectrogram pairs."""

    def __init__(self) -> None:
        super().__init__()
        self.blk_0 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.blk_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.blk_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.blk_3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.blk_4 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor, target: Tensor | None = None) -> Tensor:  # type: ignore[override]
        """Score a conditioned spectrogram pair."""
        x = _stack_condition_and_target(x, target)
        x = self.blk_0(x)
        x = self.blk_1(x)
        x = self.blk_2(x)
        x = self.blk_3(x)
        return self.blk_4(x)


class VoiceExpandDiscriminatorLoss(nn.Module):
    """Binary cross-entropy loss for the discriminator."""

    def __init__(self, discriminator: VoiceExpandDiscriminator | None = None) -> None:
        super().__init__()
        self.discriminator = discriminator
        self.criterion = nn.BCELoss()

    def forward(self, m_re: Tensor, m_nb: Tensor, m_gt: Tensor) -> Tensor:  # type: ignore[override]
        if self.discriminator is None:
            raise ValueError("VoiceExpandDiscriminatorLoss requires a discriminator instance.")

        fake_pred = self.discriminator(m_nb, m_re)
        real_pred = self.discriminator(m_nb, m_gt)
        fake_target = torch.zeros_like(fake_pred)
        real_target = torch.ones_like(real_pred)
        return self.criterion(fake_pred, fake_target) + self.criterion(real_pred, real_target)


class VoiceExpandGeneratorLoss(nn.Module):
    """Adversarial + mel reconstruction loss for the generator."""

    def __init__(self, discriminator: VoiceExpandDiscriminator | None = None, lambda_mel: float = 0.5) -> None:
        super().__init__()
        self.discriminator = discriminator
        self.lambda_mel = lambda_mel
        self.bce = nn.BCELoss()
        self.l1 = nn.L1Loss()

    def forward(self, m_re: Tensor, m_nb: Tensor, m_gt: Tensor) -> Tensor:  # type: ignore[override]
        if self.discriminator is None:
            raise ValueError("VoiceExpandGeneratorLoss requires a discriminator instance.")

        fake_pred = self.discriminator(m_nb, m_re)
        adv_target = torch.ones_like(fake_pred)
        l_adv = self.bce(fake_pred, adv_target)
        l_mel = self.l1(m_re, m_gt)
        return l_adv + self.lambda_mel * l_mel


@dataclass
class VoiceExpandGANConfig:
    """Configurable defaults for the GAN wrapper."""

    lambda_mel: float = 0.5


class VoiceExpandGAN(BaseAudioReconstructionModel):
    """GAN wrapper composed of generator and discriminator."""

    def __init__(self, config: VoiceExpandGANConfig | None = None) -> None:
        super().__init__()
        self.config = config or VoiceExpandGANConfig()
        self.generator = VoiceExpandGenerator()
        self.discriminator = VoiceExpandDiscriminator()
        self.generator_loss_fn = VoiceExpandGeneratorLoss(self.discriminator, self.config.lambda_mel)
        self.discriminator_loss_fn = VoiceExpandDiscriminatorLoss(self.discriminator)

    def generate(self, m_nb: Tensor, noise: Tensor | None = None) -> Tensor:
        return self.generator(m_nb, noise)

    def discriminate(self, m_nb: Tensor, m_target: Tensor) -> Tensor:
        return self.discriminator(m_nb, m_target)

    def generator_loss(self, m_re: Tensor, m_nb: Tensor, m_gt: Tensor) -> Tensor:
        return self.generator_loss_fn(m_re, m_nb, m_gt)

    def discriminator_loss(self, m_re: Tensor, m_nb: Tensor, m_gt: Tensor) -> Tensor:
        return self.discriminator_loss_fn(m_re, m_nb, m_gt)

    def forward(self, m_nb: Tensor, noise: Tensor | None = None) -> Tensor:  # type: ignore[override]
        """Generate a reconstructed mel spectrogram."""
        return self.generate(m_nb, noise)

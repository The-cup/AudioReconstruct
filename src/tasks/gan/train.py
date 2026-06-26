from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from config.paths import LOGS_DIR

LOGGER = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_VALIDATE_EVERY_N_EPOCHS = 50
DEFAULT_ADAM_BETAS = (0.5, 0.999)


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _validate_dataloader(name: str, dataloader: DataLoader | None) -> None:
    if dataloader is None:
        return
    if not isinstance(dataloader, DataLoader):
        raise TypeError(f"{name} must be a torch.utils.data.DataLoader instance.")
    if len(dataloader) == 0:
        raise ValueError(f"{name} must not be empty.")


def _set_dataloader_epoch(dataloader: DataLoader | None, epoch: int) -> None:
    if dataloader is None:
        return
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def _stack_tensors(items: list[Any]) -> Tensor:
    tensors: list[Tensor] = []
    for item in items:
        if torch.is_tensor(item):
            tensors.append(item.float())
        else:
            tensors.append(torch.as_tensor(item, dtype=torch.float32))
    return torch.stack(tensors, dim=0)


def gan_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate GAN samples into a batched dictionary.

    The dataset already returns loaded tensors. This collate function keeps
    path metadata as lists and stacks tensor fields for model consumption.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    speaker_ids = [item["speaker_id"] for item in batch]
    embedded_paths = [item["embedded_path"] for item in batch]
    sample_paths = [item["sample_path"] for item in batch]
    low_freq_sample_paths = [item["low_freq_sample_path"] for item in batch]

    collated = {
        "speaker_id": speaker_ids,
        "embedded_path": embedded_paths,
        "sample_path": sample_paths,
        "low_freq_sample_path": low_freq_sample_paths,
        "embedded": _stack_tensors([item["embedded"] for item in batch]),
        "sample": _stack_tensors([item["sample"] for item in batch]),
        "low_freq_sample": _stack_tensors([item["low_freq_sample"] for item in batch]),
    }
    return collated


def _prepare_spectrogram_batch(batch: Tensor) -> Tensor:
    if batch.ndim == 3:
        return batch.unsqueeze(1)
    if batch.ndim == 4:
        return batch
    raise ValueError(
        "GAN spectrogram batches must have shape (batch, time, mel) or (batch, 1, time, mel)."
    )


def _freeze_module(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def _log_metrics(title: str, metrics: dict[str, float | int]) -> None:
    preferred_order = (
        "average_generator_loss",
        "average_discriminator_loss",
        "generated_mean",
        "generated_std",
        "generated_min",
        "generated_max",
        "num_batches",
        "num_samples",
    )
    ordered_keys = [key for key in preferred_order if key in metrics]
    ordered_keys.extend(key for key in metrics if key not in ordered_keys)
    message = ", ".join(
        f"{key}={metrics[key]:.4f}" if isinstance(metrics[key], float) else f"{key}={metrics[key]}"
        for key in ordered_keys
    )
    LOGGER.info("%s: %s", title, message)
    print(f"{title}: {message}")


def evaluate_gan_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, float | int]:
    """Evaluate a GAN model on a dataloader and return loss/statistics."""
    _validate_dataloader("dataloader", dataloader)
    resolved_device = _resolve_device(device)

    model_was_training = model.training
    model.eval()

    total_generator_loss = 0.0
    total_discriminator_loss = 0.0
    total_batches = 0
    total_samples = 0
    generated_sum = 0.0
    generated_sq_sum = 0.0
    generated_count = 0
    generated_min = math.inf
    generated_max = -math.inf

    noise_generator = torch.Generator(device=resolved_device.type)
    noise_generator.manual_seed(seed)

    with torch.no_grad():
        progress = tqdm(dataloader, desc="Evaluating GAN", leave=False)
        for batch in progress:
            low_freq_sample = _prepare_spectrogram_batch(batch["low_freq_sample"].to(resolved_device))
            target_sample = _prepare_spectrogram_batch(batch["sample"].to(resolved_device))
            noise = torch.randn(
                low_freq_sample.shape,
                device=resolved_device,
                dtype=low_freq_sample.dtype,
                generator=noise_generator,
            )

            generated = model.generate(low_freq_sample, noise)
            discriminator_loss = model.discriminator_loss(generated, low_freq_sample, target_sample)
            generator_loss = model.generator_loss(generated, low_freq_sample, target_sample)

            generated_cpu = generated.detach().float().cpu()
            flattened = generated_cpu.reshape(-1)
            if flattened.numel() > 0:
                generated_sum += float(flattened.sum().item())
                generated_sq_sum += float((flattened * flattened).sum().item())
                generated_count += int(flattened.numel())
                generated_min = min(generated_min, float(flattened.min().item()))
                generated_max = max(generated_max, float(flattened.max().item()))

            total_generator_loss += float(generator_loss.detach().cpu().item())
            total_discriminator_loss += float(discriminator_loss.detach().cpu().item())
            total_batches += 1
            total_samples += int(low_freq_sample.shape[0])
            progress.set_postfix(
                generator_loss=f"{float(generator_loss.detach().cpu().item()):.4f}",
                discriminator_loss=f"{float(discriminator_loss.detach().cpu().item()):.4f}",
            )

    if model_was_training:
        model.train()

    if generated_count > 0:
        generated_mean = generated_sum / generated_count
        variance = max(generated_sq_sum / generated_count - generated_mean * generated_mean, 0.0)
        generated_std = math.sqrt(variance)
    else:
        generated_mean = 0.0
        generated_std = 0.0
        generated_min = 0.0
        generated_max = 0.0

    metrics: dict[str, float | int] = {
        "average_generator_loss": total_generator_loss / total_batches if total_batches > 0 else 0.0,
        "average_discriminator_loss": total_discriminator_loss / total_batches if total_batches > 0 else 0.0,
        "generated_mean": float(generated_mean),
        "generated_std": float(generated_std),
        "generated_min": float(generated_min),
        "generated_max": float(generated_max),
        "num_batches": total_batches,
        "num_samples": total_samples,
    }
    return metrics


def train_gan_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader | None = None,
    *,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    device: torch.device | str | None = None,
    log_dir: Path | None = None,
    weights_path: Path | None = None,
    seed: int = DEFAULT_SEED,
    validate_every_n_epochs: int = DEFAULT_VALIDATE_EVERY_N_EPOCHS,
    max_grad_norm: float | None = None,
    adam_betas: tuple[float, float] = DEFAULT_ADAM_BETAS,
) -> nn.Module:
    """Train a GAN model using separate generator and discriminator updates."""
    _validate_dataloader("train_dataloader", train_dataloader)
    _validate_dataloader("val_dataloader", val_dataloader)
    if epochs <= 0:
        raise ValueError("epochs must be greater than 0.")
    if validate_every_n_epochs <= 0:
        raise ValueError("validate_every_n_epochs must be greater than 0.")

    resolved_device = _resolve_device(device)
    model = model.to(resolved_device)
    model.train()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generator_parameters = list(model.generator.parameters())
    discriminator_parameters = list(model.discriminator.parameters())
    generator_optimizer = Adam(
        generator_parameters,
        lr=learning_rate,
        betas=adam_betas,
        weight_decay=weight_decay,
    )
    discriminator_optimizer = Adam(
        discriminator_parameters,
        lr=learning_rate,
        betas=adam_betas,
        weight_decay=weight_decay,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer_dir = Path(log_dir) if log_dir is not None else LOGS_DIR / f"gan_train_{timestamp}"
    writer_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(writer_dir))

    if weights_path is not None:
        weights_path = Path(weights_path)
        weights_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = math.inf
    best_state: dict[str, Any] | None = None
    global_step = 0
    noise_generator = torch.Generator(device=resolved_device.type)
    noise_generator.manual_seed(seed)

    try:
        for epoch in range(epochs):
            _set_dataloader_epoch(train_dataloader, epoch)
            if val_dataloader is not None:
                _set_dataloader_epoch(val_dataloader, epoch)

            model.train()
            epoch_generator_loss = 0.0
            epoch_discriminator_loss = 0.0
            epoch_samples = 0
            epoch_batches = 0

            progress = tqdm(
                train_dataloader,
                desc=f"GAN Epoch {epoch + 1}/{epochs}",
                leave=False,
            )
            for batch in progress:
                low_freq_sample = _prepare_spectrogram_batch(batch["low_freq_sample"].to(resolved_device))
                target_sample = _prepare_spectrogram_batch(batch["sample"].to(resolved_device))
                noise = torch.randn(
                    low_freq_sample.shape,
                    device=resolved_device,
                    dtype=low_freq_sample.dtype,
                    generator=noise_generator,
                )

                # Update discriminator.
                _freeze_module(model.discriminator, True)
                discriminator_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    generated_for_discriminator = model.generate(low_freq_sample, noise)
                discriminator_loss = model.discriminator_loss(
                    generated_for_discriminator,
                    low_freq_sample,
                    target_sample,
                )
                discriminator_loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(discriminator_parameters, max_grad_norm)
                discriminator_optimizer.step()

                # Update generator.
                _freeze_module(model.discriminator, False)
                generator_optimizer.zero_grad(set_to_none=True)
                generated_for_generator = model.generate(low_freq_sample, noise)
                generator_loss = model.generator_loss(
                    generated_for_generator,
                    low_freq_sample,
                    target_sample,
                )
                generator_loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(generator_parameters, max_grad_norm)
                generator_optimizer.step()

                loss_g_value = float(generator_loss.detach().cpu().item())
                loss_d_value = float(discriminator_loss.detach().cpu().item())
                epoch_generator_loss += loss_g_value
                epoch_discriminator_loss += loss_d_value
                epoch_samples += int(low_freq_sample.shape[0])
                epoch_batches += 1
                global_step += 1

                writer.add_scalar("train/generator_batch_loss", loss_g_value, global_step)
                writer.add_scalar("train/discriminator_batch_loss", loss_d_value, global_step)
                progress.set_postfix(
                    generator_loss=f"{loss_g_value:.4f}",
                    discriminator_loss=f"{loss_d_value:.4f}",
                )

            average_generator_loss = epoch_generator_loss / max(1, epoch_batches)
            average_discriminator_loss = epoch_discriminator_loss / max(1, epoch_batches)
            _log_metrics(
                f"Epoch {epoch + 1}/{epochs} training results",
                {
                    "average_generator_loss": average_generator_loss,
                    "average_discriminator_loss": average_discriminator_loss,
                    "num_batches": epoch_batches,
                    "num_samples": epoch_samples,
                },
            )
            writer.add_scalar("train/generator_epoch_loss", average_generator_loss, epoch + 1)
            writer.add_scalar("train/discriminator_epoch_loss", average_discriminator_loss, epoch + 1)
            writer.add_scalar("train/epoch_samples", epoch_samples, epoch + 1)

            should_validate = (
                val_dataloader is not None
                and ((epoch + 1) % validate_every_n_epochs == 0 or (epoch + 1) == epochs)
            )
            if should_validate:
                validation_result = evaluate_gan_model(
                    model=model,
                    dataloader=val_dataloader,
                    device=resolved_device,
                    seed=seed + epoch + 1,
                )
                validation_loss = float(validation_result["average_generator_loss"])
                _log_metrics("Validation results", validation_result)
                writer.add_scalar("val/generator_epoch_loss", validation_loss, epoch + 1)
                writer.add_scalar(
                    "val/discriminator_epoch_loss",
                    float(validation_result["average_discriminator_loss"]),
                    epoch + 1,
                )
                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    best_state = {
                        "model_state_dict": deepcopy(model.state_dict()),
                        "generator_optimizer_state_dict": deepcopy(generator_optimizer.state_dict()),
                        "discriminator_optimizer_state_dict": deepcopy(discriminator_optimizer.state_dict()),
                        "epoch": epoch + 1,
                        "best_val_loss": best_val_loss,
                    }
            elif val_dataloader is None:
                best_state = {
                    "model_state_dict": deepcopy(model.state_dict()),
                    "generator_optimizer_state_dict": deepcopy(generator_optimizer.state_dict()),
                    "discriminator_optimizer_state_dict": deepcopy(discriminator_optimizer.state_dict()),
                    "epoch": epoch + 1,
                    "best_val_loss": average_generator_loss,
                }

        if best_state is not None:
            model.load_state_dict(best_state["model_state_dict"])

        if weights_path is not None:
            checkpoint = best_state if best_state is not None else {
                "model_state_dict": model.state_dict(),
                "generator_optimizer_state_dict": generator_optimizer.state_dict(),
                "discriminator_optimizer_state_dict": discriminator_optimizer.state_dict(),
                "epoch": epochs,
                "best_val_loss": best_val_loss if best_val_loss != math.inf else None,
            }
            torch.save(checkpoint, weights_path)
    finally:
        writer.close()

    return model

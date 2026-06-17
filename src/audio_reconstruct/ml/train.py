from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from audio_reconstruct.models.custom.ge2e_loss import GE2ELoss
from audio_reconstruct.models.custom.voice_expand_gan import VoiceExpandGAN


ARTIFACTS_DIR = Path(__file__).resolve().parents[3] / "artifacts"
TENSORBOARD_DIR = ARTIFACTS_DIR / "tensorboard"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 4
DEFAULT_UTTERANCES_PER_SPEAKER = 4
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 0.0


def _resolve_device(device: torch.device | str | None) -> torch.device:
    return torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def _is_data_loader(source: Any) -> bool:
    return isinstance(source, DataLoader)


def _unwrap_batch(batch: Any) -> Any:
    if isinstance(batch, dict):
        for key in ("file", "feature", "features", "input", "inputs", "audio", "mel", "mels"):
            value = batch.get(key)
            if value is not None:
                return value
        return next(iter(batch.values()))
    if isinstance(batch, (tuple, list)):
        if not batch:
            raise ValueError("Received an empty batch.")
        return batch[0]
    return batch


def _batch_to_tensor(batch: Any) -> Tensor:
    batch = _unwrap_batch(batch)
    if not torch.is_tensor(batch):
        batch = torch.as_tensor(batch)
    return batch.float()


def _prepare_ge2e_loader_batch(batch: Any) -> Tensor:
    tensor = _batch_to_tensor(batch)
    if tensor.ndim < 3:
        raise ValueError(
            "GE2E DataLoader batches must contain grouped speaker data with shape "
            "(speakers, utterances, ...)."
        )
    return tensor


def _prepare_gan_loader_batch(batch: Any) -> Tensor:
    tensor = _batch_to_tensor(batch)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 4:
        raise ValueError("GAN DataLoader batches must be 4D tensors with shape (B, 1, H, W).")
    if tensor.shape[1] != 1:
        raise ValueError("GAN DataLoader batches must have a single mel channel.")
    return tensor


def _iter_batches(source: DataLoader, steps_per_epoch: int | None = None):
    if steps_per_epoch is None:
        yield from source
        return

    iterator = iter(source)
    for _ in range(steps_per_epoch):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(source)
            batch = next(iterator)
        yield batch


def _validate_training_inputs(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader | None,
    epochs: int,
    steps_per_epoch: int | None,
) -> None:
    if not isinstance(train_dataloader, DataLoader):
        raise TypeError("train_dataloader must be a torch.utils.data.DataLoader.")
    if val_dataloader is not None and not isinstance(val_dataloader, DataLoader):
        raise TypeError("val_dataloader must be a torch.utils.data.DataLoader.")
    if epochs <= 0:
        raise ValueError("epochs must be greater than 0.")
    if steps_per_epoch is not None and steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be greater than 0 when provided.")


def _is_voice_expand_gan(model: nn.Module) -> bool:
    return isinstance(model, VoiceExpandGAN) or (
        hasattr(model, "generator")
        and hasattr(model, "discriminator")
        and hasattr(model, "generator_loss")
        and hasattr(model, "discriminator_loss")
    )


def _set_module_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def _prepare_gan_inputs(batch: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if batch.ndim != 4:
        raise ValueError("GAN batches must be 4D tensors with shape (B, 1, H, W).")
    if batch.shape[1] != 1:
        raise ValueError("GAN batches must have a single mel channel.")

    m_gt = batch.float()
    width = m_gt.shape[-1]
    low_freq_bins = max(1, width // 2)
    mask = torch.zeros_like(m_gt)
    mask[..., :low_freq_bins] = 1.0
    m_nb = m_gt * mask
    noise = torch.randn_like(m_gt)
    return m_nb, noise, m_gt


def _evaluate_dataset(
    model: nn.Module,
    dataset: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float | int]:
    model_was_training = model.training
    loss_was_training = loss_fn.training
    model.eval()
    loss_fn.eval()

    total_loss = 0.0
    total_batches = 0
    total_speakers = 0
    total_utterances = 0

    with torch.no_grad():
        progress = tqdm(dataset, desc="Evaluating", leave=False)
        for batch in progress:
            batch_tensor = _prepare_ge2e_loader_batch(batch).to(device)
            num_speakers, num_utterances = batch_tensor.shape[:2]
            flattened_batch = batch_tensor.reshape(num_speakers * num_utterances, *batch_tensor.shape[2:])
            embeddings = model(flattened_batch).reshape(num_speakers, num_utterances, -1)
            loss = loss_fn(embeddings)

            loss_value = float(loss.detach().cpu().item())
            total_loss += loss_value
            total_batches += 1
            total_speakers += num_speakers
            total_utterances += num_speakers * num_utterances
            progress.set_postfix(loss=f"{loss_value:.4f}")

    if model_was_training:
        model.train()
    if loss_was_training:
        loss_fn.train()

    average_loss = total_loss / total_batches if total_batches > 0 else 0.0
    return {
        "average_loss": average_loss,
        "num_batches": total_batches,
        "num_speakers": total_speakers,
        "num_utterances": total_utterances,
    }


def _evaluate_voice_expand_gan(
    model: nn.Module,
    dataset: DataLoader,
    device: torch.device,
) -> dict[str, float | int]:
    if not _is_voice_expand_gan(model):
        raise TypeError("GAN evaluation requires a VoiceExpandGAN-compatible model.")

    model_was_training = model.training
    model.eval()

    with torch.no_grad():
        total_g_loss = 0.0
        total_d_loss = 0.0
        total_batches = 0
        total_samples = 0
        progress = tqdm(dataset, desc="Evaluating GAN", leave=False)
        for batch in progress:
            batch_tensor = _prepare_gan_loader_batch(batch).to(device)
            m_nb, noise, m_gt = _prepare_gan_inputs(batch_tensor)
            m_re = model.generate(m_nb, noise)
            d_loss = model.discriminator_loss(m_re, m_nb, m_gt)
            g_loss = model.generator_loss(m_re, m_nb, m_gt)

            d_loss_value = float(d_loss.detach().cpu().item())
            g_loss_value = float(g_loss.detach().cpu().item())
            total_d_loss += d_loss_value
            total_g_loss += g_loss_value
            total_batches += 1
            total_samples += batch_tensor.shape[0]
            progress.set_postfix(g_loss=f"{g_loss_value:.4f}", d_loss=f"{d_loss_value:.4f}")

    if model_was_training:
        model.train()

    average_g_loss = total_g_loss / total_batches if total_batches > 0 else 0.0
    average_d_loss = total_d_loss / total_batches if total_batches > 0 else 0.0
    return {
        "average_generator_loss": average_g_loss,
        "average_discriminator_loss": average_d_loss,
        "average_loss": average_g_loss + average_d_loss,
        "num_batches": total_batches,
        "num_samples": total_samples,
    }


def _train_voice_expand_gan(
    model: nn.Module,
    train_dataloader: DataLoader,
    *,
    weights_path: Path | None,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    val_dataloader: DataLoader | None,
    device: torch.device,
    log_dir: Path | None,
    seed: int,
    steps_per_epoch: int | None,
    max_grad_norm: float | None,
) -> nn.Module:
    if not _is_voice_expand_gan(model):
        raise TypeError("GAN training requires a VoiceExpandGAN-compatible model.")
    if len(train_dataloader) == 0:
        raise ValueError("train_dataloader must not be empty.")

    model = model.to(device)
    model.train()
    generator = model.generator
    discriminator = model.discriminator
    g_optimizer = Adam(generator.parameters(), lr=learning_rate, weight_decay=weight_decay)
    d_optimizer = Adam(discriminator.parameters(), lr=learning_rate, weight_decay=weight_decay)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    effective_steps_per_epoch = steps_per_epoch if steps_per_epoch is not None else max(1, len(train_dataloader))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer_dir = Path(log_dir) if log_dir is not None else TENSORBOARD_DIR / f"gan_train_{timestamp}"
    writer_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(writer_dir))

    if weights_path is not None:
        weights_path = Path(weights_path)
        weights_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = math.inf
    best_state: dict[str, Any] | None = None
    global_step = 0

    try:
        for epoch in range(epochs):
            batches = _iter_batches(train_dataloader, effective_steps_per_epoch)
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_steps = 0

            progress = tqdm(batches, desc=f"GAN Epoch {epoch + 1}/{epochs}", leave=False)
            for batch_data in progress:
                batch = _prepare_gan_loader_batch(batch_data).to(device)
                m_nb, noise, m_gt = _prepare_gan_inputs(batch)

                _set_module_requires_grad(generator, False)
                _set_module_requires_grad(discriminator, True)
                d_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    m_re_detached = model.generate(m_nb, noise)
                d_loss = model.discriminator_loss(m_re_detached, m_nb, m_gt)
                d_loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_grad_norm)
                d_optimizer.step()

                _set_module_requires_grad(discriminator, False)
                _set_module_requires_grad(generator, True)
                g_optimizer.zero_grad(set_to_none=True)
                m_re = model.generate(m_nb, noise)
                g_loss = model.generator_loss(m_re, m_nb, m_gt)
                g_loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_grad_norm)
                g_optimizer.step()
                _set_module_requires_grad(discriminator, True)

                d_loss_value = float(d_loss.detach().cpu().item())
                g_loss_value = float(g_loss.detach().cpu().item())
                epoch_d_loss += d_loss_value
                epoch_g_loss += g_loss_value
                epoch_steps += 1
                global_step += 1
                writer.add_scalar("gan/train/discriminator_loss", d_loss_value, global_step)
                writer.add_scalar("gan/train/generator_loss", g_loss_value, global_step)
                progress.set_postfix(g_loss=f"{g_loss_value:.4f}", d_loss=f"{d_loss_value:.4f}")

            average_g_loss = epoch_g_loss / max(1, epoch_steps)
            average_d_loss = epoch_d_loss / max(1, epoch_steps)
            writer.add_scalar("gan/train/epoch_generator_loss", average_g_loss, epoch + 1)
            writer.add_scalar("gan/train/epoch_discriminator_loss", average_d_loss, epoch + 1)

            current_state = {
                "generator_state_dict": deepcopy(generator.state_dict()),
                "discriminator_state_dict": deepcopy(discriminator.state_dict()),
                "generator_optimizer_state_dict": deepcopy(g_optimizer.state_dict()),
                "discriminator_optimizer_state_dict": deepcopy(d_optimizer.state_dict()),
                "epoch": epoch + 1,
                "best_val_loss": average_g_loss + average_d_loss,
            }

            if val_dataloader is not None:
                validation_result = _evaluate_voice_expand_gan(model=model, dataset=val_dataloader, device=device)
                validation_loss = float(validation_result["average_loss"])
                writer.add_scalar("gan/val/generator_loss", float(validation_result["average_generator_loss"]), epoch + 1)
                writer.add_scalar("gan/val/discriminator_loss", float(validation_result["average_discriminator_loss"]), epoch + 1)
                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    best_state = {
                        **current_state,
                        "best_val_loss": best_val_loss,
                    }
            else:
                best_state = current_state

        if weights_path is not None:
            checkpoint = best_state if best_state is not None else {
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "generator_optimizer_state_dict": g_optimizer.state_dict(),
                "discriminator_optimizer_state_dict": d_optimizer.state_dict(),
                "epoch": epochs,
                "best_val_loss": best_val_loss if best_val_loss != math.inf else None,
            }
            torch.save(checkpoint, weights_path)
    finally:
        _set_module_requires_grad(generator, True)
        _set_module_requires_grad(discriminator, True)
        writer.close()

    return model


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    loss_fn: nn.Module | None = None,
    weights_path: Path | None = None,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    val_dataloader: DataLoader | None = None,
    device: torch.device | str | None = None,
    log_dir: Path | None = None,
    seed: int = DEFAULT_SEED,
    steps_per_epoch: int | None = None,
    max_grad_norm: float | None = None,
) -> nn.Module:
    """Train a non-GAN model with GE2E-style batches."""
    if loss_fn is None:
        loss_fn = GE2ELoss()

    _validate_training_inputs(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )

    resolved_device = _resolve_device(device)
    model = model.to(resolved_device)
    loss_fn = loss_fn.to(resolved_device)
    model.train()
    loss_fn.train()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    parameters = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = Adam(parameters, lr=learning_rate, weight_decay=weight_decay)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer_dir = Path(log_dir) if log_dir is not None else TENSORBOARD_DIR / f"train_{timestamp}"
    writer_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(writer_dir))

    if weights_path is not None:
        weights_path = Path(weights_path)
        weights_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = math.inf
    best_state = None
    global_step = 0
    if len(train_dataloader) == 0:
        raise ValueError("train_dataloader must not be empty.")
    if val_dataloader is not None and len(val_dataloader) == 0:
        raise ValueError("val_dataloader must not be empty if provided.")

    effective_steps_per_epoch = steps_per_epoch if steps_per_epoch is not None else max(1, len(train_dataloader))

    try:
        for epoch in range(epochs):
            model.train()
            loss_fn.train()
            batches = _iter_batches(train_dataloader, effective_steps_per_epoch)

            epoch_loss = 0.0
            epoch_steps = 0
            progress = tqdm(batches, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for batch_data in progress:
                batch = _prepare_ge2e_loader_batch(batch_data).to(resolved_device)
                num_speakers, num_utterances = batch.shape[:2]
                flattened_batch = batch.reshape(num_speakers * num_utterances, *batch.shape[2:])

                optimizer.zero_grad(set_to_none=True)
                embeddings = model(flattened_batch).reshape(num_speakers, num_utterances, -1)
                loss = loss_fn(embeddings)
                loss.backward()

                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)

                optimizer.step()

                loss_value = float(loss.detach().cpu().item())
                epoch_loss += loss_value
                epoch_steps += 1
                global_step += 1
                writer.add_scalar("train/batch_loss", loss_value, global_step)
                progress.set_postfix(loss=f"{loss_value:.4f}")

            average_train_loss = epoch_loss / max(1, epoch_steps)
            writer.add_scalar("train/epoch_loss", average_train_loss, epoch + 1)

            if val_dataloader is not None:
                validation_result = _evaluate_dataset(
                    model=model,
                    dataset=val_dataloader,
                    loss_fn=loss_fn,
                    device=resolved_device,
                )
                validation_loss = float(validation_result["average_loss"])
                writer.add_scalar("val/epoch_loss", validation_loss, epoch + 1)
                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    best_state = {
                        "model_state_dict": deepcopy(model.state_dict()),
                        "loss_state_dict": deepcopy(loss_fn.state_dict()),
                        "optimizer_state_dict": deepcopy(optimizer.state_dict()),
                        "epoch": epoch + 1,
                        "best_val_loss": best_val_loss,
                    }
            else:
                best_state = {
                    "model_state_dict": deepcopy(model.state_dict()),
                    "loss_state_dict": deepcopy(loss_fn.state_dict()),
                    "optimizer_state_dict": deepcopy(optimizer.state_dict()),
                    "epoch": epoch + 1,
                    "best_val_loss": average_train_loss,
                }

        if weights_path is not None:
            checkpoint = best_state if best_state is not None else {
                "model_state_dict": model.state_dict(),
                "loss_state_dict": loss_fn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epochs,
                "best_val_loss": best_val_loss if best_val_loss != math.inf else None,
            }
            torch.save(checkpoint, weights_path)
    finally:
        writer.close()

    return model


def train_gan_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    weights_path: Path | None = None,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    val_dataloader: DataLoader | None = None,
    device: torch.device | str | None = None,
    log_dir: Path | None = None,
    seed: int = DEFAULT_SEED,
    steps_per_epoch: int | None = None,
    max_grad_norm: float | None = None,
) -> nn.Module:
    """Train a VoiceExpandGAN model with alternating generator/discriminator updates."""
    _validate_training_inputs(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )
    return _train_voice_expand_gan(
        model=model,
        train_dataloader=train_dataloader,
        weights_path=weights_path,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        val_dataloader=val_dataloader,
        device=_resolve_device(device),
        log_dir=log_dir,
        seed=seed,
        steps_per_epoch=steps_per_epoch,
        max_grad_norm=max_grad_norm,
    )


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader | None = None,
    loss_fn: nn.Module | None = None,
    weights_path: Path | None = None,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    device: torch.device | str | None = None,
    log_dir: Path | None = None,
    seed: int = DEFAULT_SEED,
    steps_per_epoch: int | None = None,
    max_grad_norm: float | None = None,
) -> nn.Module:
    """Dispatch to the appropriate training routine based on model type."""
    _validate_training_inputs(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )

    if _is_voice_expand_gan(model):
        return train_gan_model(
            model=model,
            train_dataloader=train_dataloader,
            weights_path=weights_path,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            val_dataloader=val_dataloader,
            device=device,
            log_dir=log_dir,
            seed=seed,
            steps_per_epoch=steps_per_epoch,
            max_grad_norm=max_grad_norm,
        )

    return train_model(
        model=model,
        train_dataloader=train_dataloader,
        loss_fn=loss_fn,
        weights_path=weights_path,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        val_dataloader=val_dataloader,
        device=device,
        log_dir=log_dir,
        seed=seed,
        steps_per_epoch=steps_per_epoch,
        max_grad_norm=max_grad_norm,
    )

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import Dataset
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


def _normalize_string(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value).strip()


def _extract_speaker_id(sample: Any) -> str:
    if isinstance(sample, dict):
        if sample.get("speaker_id") is not None:
            return _normalize_string(sample["speaker_id"])
        label = sample.get("label")
    elif isinstance(sample, (tuple, list)):
        label = sample[1] if len(sample) > 1 else None
    else:
        speaker_id = getattr(sample, "speaker_id", None)
        if speaker_id is not None:
            return _normalize_string(speaker_id)
        label = getattr(sample, "label", None)

    label_str = _normalize_string(label)
    if "-" in label_str:
        return label_str.split("-", maxsplit=1)[0]
    return label_str


def _extract_feature(sample: Any) -> Tensor:
    if isinstance(sample, dict):
        feature = sample.get("feature")
        if feature is None:
            feature = sample.get("input")
    elif isinstance(sample, (tuple, list)):
        feature = sample[0]
    else:
        feature = getattr(sample, "feature", None)

    if feature is None:
        raise ValueError("Dataset item does not contain a feature tensor.")

    if not torch.is_tensor(feature):
        feature = torch.as_tensor(feature)

    feature = feature.float()
    if feature.ndim == 3 and feature.shape[0] == 1:
        feature = feature.squeeze(0)
    if feature.ndim == 2 and feature.shape[0] == 40 and feature.shape[1] != 40:
        feature = feature.transpose(0, 1)
    return feature


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


def _materialize_mel_batch(dataset: Dataset, batch_indices: list[int]) -> Tensor:
    features = [_extract_feature(dataset[index]) for index in batch_indices]
    normalized_features: list[Tensor] = []
    for feature in features:
        if feature.ndim == 2:
            feature = feature.unsqueeze(0)
        elif feature.ndim == 3 and feature.shape[0] != 1 and feature.shape[-1] == 40:
            feature = feature.unsqueeze(0)
        normalized_features.append(feature)
    batch = torch.stack(normalized_features, dim=0)
    if batch.ndim == 4 and batch.shape[1] != 1:
        raise ValueError("Expected mel batches to have a single channel.")
    return batch.float()


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


def _build_linear_batches(
    dataset_size: int,
    batch_size: int,
    rng: random.Random,
) -> list[list[int]]:
    indices = list(range(dataset_size))
    rng.shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, dataset_size, batch_size)]


def _build_random_batches(
    dataset_size: int,
    batch_size: int,
    steps_per_epoch: int,
    rng: random.Random,
) -> list[list[int]]:
    indices = list(range(dataset_size))
    if dataset_size == 0:
        raise ValueError("The dataset is empty.")

    batches: list[list[int]] = []
    for _ in range(steps_per_epoch):
        if dataset_size >= batch_size:
            batch_indices = rng.sample(indices, k=batch_size)
        else:
            batch_indices = [rng.choice(indices) for _ in range(batch_size)]
        batches.append(batch_indices)
    return batches


def _build_speaker_groups(dataset: Dataset) -> dict[str, list[int]]:
    speaker_to_indices: dict[str, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        sample = dataset[index]
        speaker_id = _extract_speaker_id(sample)
        speaker_to_indices[speaker_id].append(index)
    if not speaker_to_indices:
        raise ValueError("The dataset is empty or no valid speaker labels were found.")
    return speaker_to_indices


def _sample_utterance_indices(
    indices: list[int],
    utterances_per_speaker: int,
    rng: random.Random,
) -> list[int]:
    if len(indices) >= utterances_per_speaker:
        return rng.sample(indices, k=utterances_per_speaker)
    return [rng.choice(indices) for _ in range(utterances_per_speaker)]


def _build_training_batches(
    speaker_to_indices: dict[str, list[int]],
    speakers_per_batch: int,
    utterances_per_speaker: int,
    steps_per_epoch: int,
    rng: random.Random,
) -> list[list[list[int]]]:
    speaker_ids = list(speaker_to_indices.keys())
    if len(speaker_ids) == 0:
        raise ValueError("No speakers were found in the dataset.")

    batches: list[list[list[int]]] = []
    for _ in range(steps_per_epoch):
        if len(speaker_ids) >= speakers_per_batch:
            selected_speakers = rng.sample(speaker_ids, k=speakers_per_batch)
        else:
            selected_speakers = [rng.choice(speaker_ids) for _ in range(speakers_per_batch)]

        batch_indices = [
            _sample_utterance_indices(speaker_to_indices[speaker_id], utterances_per_speaker, rng)
            for speaker_id in selected_speakers
        ]
        batches.append(batch_indices)
    return batches


def _build_evaluation_batches(
    speaker_to_indices: dict[str, list[int]],
    speakers_per_batch: int,
    utterances_per_speaker: int,
) -> list[list[list[int]]]:
    speaker_ids = sorted(speaker_to_indices.keys())
    if len(speaker_ids) == 0:
        raise ValueError("No speakers were found in the dataset.")

    batches: list[list[list[int]]] = []
    for start in range(0, len(speaker_ids), max(1, speakers_per_batch)):
        selected_speakers = speaker_ids[start : start + max(1, speakers_per_batch)]
        batch_indices = [
            _sample_utterance_indices(
                speaker_to_indices[speaker_id],
                utterances_per_speaker,
                random.Random(DEFAULT_SEED),
            )
            for speaker_id in selected_speakers
        ]
        batches.append(batch_indices)
    return batches


def _materialize_ge2e_batch(dataset: Dataset, batch_indices: list[list[int]]) -> Tensor:
    speaker_batches: list[Tensor] = []
    for speaker_indices in batch_indices:
        utterances = [_extract_feature(dataset[index]) for index in speaker_indices]
        speaker_batches.append(torch.stack(utterances, dim=0))
    return torch.stack(speaker_batches, dim=0)


def _evaluate_dataset(
    model: nn.Module,
    dataset: Dataset,
    loss_fn: nn.Module,
    *,
    speakers_per_batch: int,
    utterances_per_speaker: int,
    device: torch.device,
) -> dict[str, float | int]:
    model_was_training = model.training
    loss_was_training = loss_fn.training
    model.eval()
    loss_fn.eval()

    speaker_to_indices = _build_speaker_groups(dataset)
    batches = _build_evaluation_batches(
        speaker_to_indices=speaker_to_indices,
        speakers_per_batch=speakers_per_batch,
        utterances_per_speaker=utterances_per_speaker,
    )

    total_loss = 0.0
    total_batches = 0
    total_speakers = 0
    total_utterances = 0

    with torch.no_grad():
        progress = tqdm(batches, desc="Evaluating", leave=False)
        for batch_indices in progress:
            batch = _materialize_ge2e_batch(dataset, batch_indices).to(device)
            num_speakers, num_utterances = batch.shape[:2]
            flattened_batch = batch.reshape(num_speakers * num_utterances, *batch.shape[2:])
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
    dataset: Dataset,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    if not _is_voice_expand_gan(model):
        raise TypeError("GAN evaluation requires a VoiceExpandGAN-compatible model.")

    model_was_training = model.training
    model.eval()

    total_g_loss = 0.0
    total_d_loss = 0.0
    total_batches = 0
    total_samples = 0
    batches = _build_linear_batches(len(dataset), max(1, batch_size), random.Random(DEFAULT_SEED))

    with torch.no_grad():
        progress = tqdm(batches, desc="Evaluating GAN", leave=False)
        for batch_indices in progress:
            batch = _materialize_mel_batch(dataset, batch_indices).to(device)
            m_nb, noise, m_gt = _prepare_gan_inputs(batch)
            m_re = model.generate(m_nb, noise)
            d_loss = model.discriminator_loss(m_re, m_nb, m_gt)
            g_loss = model.generator_loss(m_re, m_nb, m_gt)

            d_loss_value = float(d_loss.detach().cpu().item())
            g_loss_value = float(g_loss.detach().cpu().item())
            total_d_loss += d_loss_value
            total_g_loss += g_loss_value
            total_batches += 1
            total_samples += batch.shape[0]
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
    train_dataset: Dataset,
    *,
    weights_path: Path | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    validation_dataset: Dataset | None,
    device: torch.device,
    log_dir: Path | None,
    seed: int,
    steps_per_epoch: int | None,
    max_grad_norm: float | None,
) -> nn.Module:
    if not _is_voice_expand_gan(model):
        raise TypeError("GAN training requires a VoiceExpandGAN-compatible model.")
    if len(train_dataset) == 0:
        raise ValueError("train_dataset must not be empty.")

    model = model.to(device)
    model.train()
    generator = model.generator
    discriminator = model.discriminator
    g_optimizer = Adam(generator.parameters(), lr=learning_rate, weight_decay=weight_decay)
    d_optimizer = Adam(discriminator.parameters(), lr=learning_rate, weight_decay=weight_decay)

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    effective_steps_per_epoch = steps_per_epoch
    if effective_steps_per_epoch is None:
        effective_steps_per_epoch = max(1, math.ceil(len(train_dataset) / max(1, batch_size)))

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
            epoch_rng = random.Random(seed + epoch)
            batches = _build_random_batches(
                dataset_size=len(train_dataset),
                batch_size=batch_size,
                steps_per_epoch=effective_steps_per_epoch,
                rng=epoch_rng,
            )

            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            progress = tqdm(batches, desc=f"GAN Epoch {epoch + 1}/{epochs}", leave=False)
            for batch_indices in progress:
                batch = _materialize_mel_batch(train_dataset, batch_indices).to(device)
                m_nb, noise, m_gt = _prepare_gan_inputs(batch)

                # Update discriminator.
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

                # Update generator.
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
                global_step += 1
                writer.add_scalar("gan/train/discriminator_loss", d_loss_value, global_step)
                writer.add_scalar("gan/train/generator_loss", g_loss_value, global_step)
                progress.set_postfix(g_loss=f"{g_loss_value:.4f}", d_loss=f"{d_loss_value:.4f}")

            average_g_loss = epoch_g_loss / max(1, len(batches))
            average_d_loss = epoch_d_loss / max(1, len(batches))
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

            if validation_dataset is not None:
                validation_result = _evaluate_voice_expand_gan(
                    model=model,
                    dataset=validation_dataset,
                    batch_size=batch_size,
                    device=device,
                )
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


def train(
    model: nn.Module,
    train_dataset: Dataset,
    loss_fn: nn.Module | None = None,
    weights_path: Path | None = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    utterances_per_speaker: int = DEFAULT_UTTERANCES_PER_SPEAKER,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    validation_dataset: Dataset | None = None,
    device: torch.device | str | None = None,
    log_dir: Path | None = None,
    seed: int = DEFAULT_SEED,
    steps_per_epoch: int | None = None,
    max_grad_norm: float | None = None,
) -> nn.Module:
    """Train a speaker encoder model with GE2E-style batches.

    Args:
        model: PyTorch model to train.
        train_dataset: Training dataset inheriting from Dataset.
        loss_fn: Loss function, defaults to GE2ELoss.
        weights_path: Optional checkpoint file path. When provided, the final
            checkpoint is saved there.
        epochs: Number of training epochs.
        batch_size: Number of speakers per batch.
        utterances_per_speaker: Number of utterances per speaker in each batch.
        learning_rate: Optimizer learning rate.
        weight_decay: Optimizer weight decay.
        validation_dataset: Optional validation dataset.
        device: Device for training, defaults to CUDA when available.
        log_dir: TensorBoard log directory. Defaults under artifacts/.
        seed: Random seed for batch sampling.
        steps_per_epoch: Optional number of batches per epoch.
        max_grad_norm: Optional gradient clipping threshold.

    Returns:
        The trained model instance.
    """
    if loss_fn is None:
        loss_fn = GE2ELoss()

    if not isinstance(train_dataset, Dataset):
        raise TypeError("train_dataset must inherit from torch.utils.data.Dataset.")
    if validation_dataset is not None and not isinstance(validation_dataset, Dataset):
        raise TypeError("validation_dataset must inherit from torch.utils.data.Dataset.")
    if epochs <= 0:
        raise ValueError("epochs must be greater than 0.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if utterances_per_speaker <= 0:
        raise ValueError("utterances_per_speaker must be greater than 0.")
    if steps_per_epoch is not None and steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be greater than 0 when provided.")

    resolved_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if _is_voice_expand_gan(model):
        return _train_voice_expand_gan(
            model=model,
            train_dataset=train_dataset,
            weights_path=weights_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            validation_dataset=validation_dataset,
            device=resolved_device,
            log_dir=log_dir,
            seed=seed,
            steps_per_epoch=steps_per_epoch,
            max_grad_norm=max_grad_norm,
        )

    model = model.to(resolved_device)
    loss_fn = loss_fn.to(resolved_device)

    model.train()
    loss_fn.train()

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    speaker_to_indices = _build_speaker_groups(train_dataset)
    effective_steps_per_epoch = steps_per_epoch
    if effective_steps_per_epoch is None:
        effective_steps_per_epoch = max(
            1,
            math.ceil(len(speaker_to_indices) / max(1, batch_size)),
        )

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

    try:
        for epoch in range(epochs):
            model.train()
            loss_fn.train()
            epoch_rng = random.Random(seed + epoch)
            batches = _build_training_batches(
                speaker_to_indices=speaker_to_indices,
                speakers_per_batch=batch_size,
                utterances_per_speaker=utterances_per_speaker,
                steps_per_epoch=effective_steps_per_epoch,
                rng=epoch_rng,
            )

            epoch_loss = 0.0
            progress = tqdm(batches, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for batch_indices in progress:
                batch = _materialize_ge2e_batch(train_dataset, batch_indices).to(resolved_device)
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
                global_step += 1
                writer.add_scalar("train/batch_loss", loss_value, global_step)
                progress.set_postfix(loss=f"{loss_value:.4f}")

            average_train_loss = epoch_loss / max(1, len(batches))
            writer.add_scalar("train/epoch_loss", average_train_loss, epoch + 1)

            if validation_dataset is not None:
                validation_result = _evaluate_dataset(
                    model=model,
                    dataset=validation_dataset,
                    loss_fn=loss_fn,
                    speakers_per_batch=batch_size,
                    utterances_per_speaker=utterances_per_speaker,
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


def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    loss_fn: nn.Module | None = None,
    weights_path: Path | None = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    utterances_per_speaker: int = DEFAULT_UTTERANCES_PER_SPEAKER,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    validation_dataset: Dataset | None = None,
    device: torch.device | str | None = None,
    log_dir: Path | None = None,
    seed: int = DEFAULT_SEED,
    steps_per_epoch: int | None = None,
    max_grad_norm: float | None = None,
) -> nn.Module:
    """Backward-compatible alias for train()."""
    return train(
        model=model,
        train_dataset=train_dataset,
        loss_fn=loss_fn,
        weights_path=weights_path,
        epochs=epochs,
        batch_size=batch_size,
        utterances_per_speaker=utterances_per_speaker,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        validation_dataset=validation_dataset,
        device=device,
        log_dir=log_dir,
        seed=seed,
        steps_per_epoch=steps_per_epoch,
        max_grad_norm=max_grad_norm,
    )

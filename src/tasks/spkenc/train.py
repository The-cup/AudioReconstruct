from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
import logging
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from config.paths import LOGS_DIR
from models.custom.ge2e_loss import GE2ELoss


LOGGER = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_VALIDATE_EVERY_N_EPOCHS = 50


def _resolve_device(device: torch.device | str | None) -> torch.device:
    return torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def _set_dataloader_epoch(dataloader: DataLoader | None, epoch: int) -> None:
    if dataloader is None:
        return
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)


def _validate_inputs(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader | None,
    epochs: int,
    steps_per_epoch: int | None,
    validate_every_n_epochs: int,
) -> None:
    if not isinstance(train_dataloader, DataLoader):
        raise TypeError("train_dataloader must be a torch.utils.data.DataLoader.")
    if val_dataloader is not None and not isinstance(val_dataloader, DataLoader):
        raise TypeError("val_dataloader must be a torch.utils.data.DataLoader.")
    if len(train_dataloader) == 0:
        raise ValueError("train_dataloader must not be empty.")
    if val_dataloader is not None and len(val_dataloader) == 0:
        raise ValueError("val_dataloader must not be empty if provided.")
    if epochs <= 0:
        raise ValueError("epochs must be greater than 0.")
    if steps_per_epoch is not None and steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be greater than 0 when provided.")
    if validate_every_n_epochs <= 0:
        raise ValueError("validate_every_n_epochs must be greater than 0.")


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


def _prepare_spkenc_batch(batch: Any) -> Tensor:
    tensor = _batch_to_tensor(batch)
    if tensor.ndim < 3:
        raise ValueError(
            "SpkEnc batches must contain grouped speaker data with shape "
            "(speakers, utterances, ...)."
        )
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


def _format_metric_value(value: float | int) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _log_metrics(title: str, metrics: dict[str, float | int]) -> None:
    preferred_order = ("average_loss", "num_batches", "num_speakers", "num_utterances")
    ordered_keys = [key for key in preferred_order if key in metrics]
    ordered_keys.extend(key for key in metrics.keys() if key not in ordered_keys)
    message = ", ".join(f"{key}={_format_metric_value(metrics[key])}" for key in ordered_keys)
    LOGGER.info("%s: %s", title, message)
    print(f"{title}: {message}")


def evaluate_spkenc(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device | str | None = None,
) -> dict[str, float | int]:
    resolved_device = _resolve_device(device)

    model_was_training = model.training
    loss_was_training = loss_fn.training
    model.eval()
    loss_fn.eval()

    total_loss = 0.0
    total_batches = 0
    total_speakers = 0
    total_utterances = 0

    with torch.no_grad():
        progress = tqdm(dataloader, desc="Evaluating SpkEnc", leave=False)
        for batch in progress:
            batch_tensor = _prepare_spkenc_batch(batch).to(resolved_device)
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

    metrics = {
        "average_loss": total_loss / total_batches if total_batches > 0 else 0.0,
        "num_batches": total_batches,
        "num_speakers": total_speakers,
        "num_utterances": total_utterances,
    }
    return metrics


def train_spkenc(
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
    validate_every_n_epochs: int = DEFAULT_VALIDATE_EVERY_N_EPOCHS,
) -> nn.Module:
    if loss_fn is None:
        loss_fn = GE2ELoss()

    _validate_inputs(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validate_every_n_epochs=validate_every_n_epochs,
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
    writer_dir = Path(log_dir) if log_dir is not None else LOGS_DIR / f"spkenc_train_{timestamp}"
    writer_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(writer_dir))

    if weights_path is not None:
        weights_path = Path(weights_path)
        weights_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = math.inf
    best_state: dict[str, Any] | None = None
    global_step = 0
    effective_steps_per_epoch = steps_per_epoch if steps_per_epoch is not None else max(1, len(train_dataloader))

    try:
        for epoch in range(epochs):
            model.train()
            loss_fn.train()
            _set_dataloader_epoch(train_dataloader, epoch)

            epoch_loss = 0.0
            epoch_steps = 0
            progress = tqdm(_iter_batches(train_dataloader, effective_steps_per_epoch), desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for batch_data in progress:
                batch = _prepare_spkenc_batch(batch_data).to(resolved_device)
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

            should_validate = (
                val_dataloader is not None
                and ((epoch + 1) % validate_every_n_epochs == 0 or (epoch + 1) == epochs)
            )
            if should_validate:
                _set_dataloader_epoch(val_dataloader, epoch + 1)
                validation_result = evaluate_spkenc(
                    model=model,
                    dataloader=val_dataloader,
                    loss_fn=loss_fn,
                    device=resolved_device,
                )
                validation_loss = float(validation_result["average_loss"])
                _log_metrics("Validation results", validation_result)
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
            elif val_dataloader is None:
                best_state = {
                    "model_state_dict": deepcopy(model.state_dict()),
                    "loss_state_dict": deepcopy(loss_fn.state_dict()),
                    "optimizer_state_dict": deepcopy(optimizer.state_dict()),
                    "epoch": epoch + 1,
                    "best_val_loss": average_train_loss,
                }

        if best_state is not None:
            model.load_state_dict(best_state["model_state_dict"])
            loss_fn.load_state_dict(best_state["loss_state_dict"])

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

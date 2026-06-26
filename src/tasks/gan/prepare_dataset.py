from __future__ import annotations

from config.paths import DATA_DIR, SELECTED_EMBEDDED_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, \
    build_dir_path, CHECKPOINTS_DIR

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import math
import logging
import random
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

from models.registry import get_model


LOGGER = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "LibriSpeech"
DEFAULT_DATASET_SUB_NAME = "train-clean-100"
DEFAULT_STATE_DICT_PATH = CHECKPOINTS_DIR / "spkenc_26-06-18-18-08-07.pth"
DEFAULT_MIN_UTT_PER_SPK = 20
DEFAULT_EMBED_EXTRACT_UTT_PER_SPK = 20
DEFAULT_EMBEDDING_FILENAME = "embedded_vector.pt"
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 32
DEFAULT_LOW_FREQ_CUTOFF_HZ = 600.0
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_N_MELS = 40


def _hz_to_mel(frequency_hz: float) -> float:
    return 2595.0 * math.log10(1.0 + frequency_hz / 700.0)


def _mel_to_hz(mel_value: float) -> float:
    return 700.0 * (10.0 ** (mel_value / 2595.0) - 1.0)


def _build_low_freq_mask(
    num_mel_bins: int,
    cutoff_hz: float = DEFAULT_LOW_FREQ_CUTOFF_HZ,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Tensor:
    mel_min = _hz_to_mel(0.0)
    mel_max = _hz_to_mel(sample_rate / 2.0)
    mel_centers = torch.linspace(mel_min, mel_max, steps=num_mel_bins)
    hz_centers = torch.tensor([_mel_to_hz(float(value)) for value in mel_centers], dtype=torch.float32)
    return (hz_centers <= cutoff_hz).to(dtype=torch.float32)


def _apply_low_freq_mask(
    mel_tensor: Tensor,
    cutoff_hz: float = DEFAULT_LOW_FREQ_CUTOFF_HZ,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Tensor:
    if mel_tensor.ndim != 2:
        raise ValueError(f"Expected a 2D mel tensor with shape (time, mel_bins), got {tuple(mel_tensor.shape)}")

    num_mel_bins = mel_tensor.shape[-1]
    mask = _build_low_freq_mask(num_mel_bins=num_mel_bins, cutoff_hz=cutoff_hz, sample_rate=sample_rate)
    return mel_tensor * mask.to(device=mel_tensor.device).view(1, -1)


def _resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _is_directory_empty(directory: Path) -> bool:
    return next(directory.iterdir(), None) is None


def _ensure_empty_directory(directory: Path) -> None:
    if directory.exists():
        if not directory.is_dir():
            raise NotADirectoryError(f"{directory} exists but is not a directory.")
        if not _is_directory_empty(directory):
            raise FileExistsError(f"{directory} already exists and is not empty.")
    else:
        directory.mkdir(parents=True, exist_ok=True)


def _load_tensor(file_path: Path) -> Tensor | None:
    try:
        tensor = torch.load(file_path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - runtime IO safeguard
        LOGGER.warning("Failed to load %s: %s", file_path, exc)
        return None

    if not isinstance(tensor, torch.Tensor):
        LOGGER.warning("Skipping non-tensor file %s", file_path)
        return None

    return tensor


def _save_tensor(tensor: Tensor, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.to(dtype=torch.float32).cpu(), file_path)


def _load_model_state_dict(state_dict_path: Path, device: torch.device) -> dict[str, Tensor]:
    checkpoint = torch.load(state_dict_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format in {state_dict_path}.")

    if state_dict and all(isinstance(key, str) and key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

    return state_dict


def _resolve_state_dict_path(state_dict_path: Path) -> Path:
    if state_dict_path.exists():
        return state_dict_path

    checkpoint_dir = state_dict_path.parent
    if checkpoint_dir.exists():
        candidates = sorted(
            checkpoint_dir.glob("spkenc_*.pth"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            LOGGER.info("Fallback to latest checkpoint: %s", candidates[0])
            return candidates[0]

    raise FileNotFoundError(f"state_dict_path does not exist and no fallback checkpoint was found: {state_dict_path}")


def _speaker_files(directory: Path, embedding_filename: str) -> list[Path]:
    return [
        file_path
        for file_path in sorted(directory.iterdir())
        if file_path.is_file()
        and file_path.suffix == ".pt"
        and file_path.name != embedding_filename
    ]


def _move_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    return Path(shutil.move(str(source), str(destination)))


def _extract_embeddings_for_speaker(
    model: torch.nn.Module,
    speaker_files: list[Path],
    device: torch.device,
    batch_size: int,
) -> list[Tensor]:
    embeddings: list[Tensor] = []
    for start in range(0, len(speaker_files), batch_size):
        batch_paths = speaker_files[start : start + batch_size]
        batch_tensors: list[Tensor] = []
        for file_path in batch_paths:
            tensor = _load_tensor(file_path)
            if tensor is None:
                continue
            batch_tensors.append(tensor.to(device=device, dtype=torch.float32))

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors, dim=0)
        with torch.inference_mode():
            batch_embeddings = model(batch)
        embeddings.extend(batch_embeddings.detach().cpu())

    return embeddings


def preprocess_dataset(
        dataset_name="LibriSpeech",
        dataset_sub_name="train-clean-100",
        raw_data_dir=RAW_DATA_DIR,
        save_dir=PROCESSED_DATA_DIR,
        load_data=False
):
    from tasks.preprocess_dataset import get_dataset, preprocess_dataset
    dataset = get_dataset(
        dataset_name=dataset_name,
        dataset_sub_name=dataset_sub_name,
        base_dir=raw_data_dir,
    )
    return preprocess_dataset(
        dataset=dataset,
        save_dir=save_dir,
        load_data=load_data
    )


def select_embedded_samples(
    data_dir: str | Path = PROCESSED_DATA_DIR,
    selected_embedded_dir: str | Path = SELECTED_EMBEDDED_DIR,
    *,
    min_utt_per_spk: int = DEFAULT_MIN_UTT_PER_SPK,
    embed_extract_utt_per_spk: int = DEFAULT_EMBED_EXTRACT_UTT_PER_SPK,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    selected_embedded_dir = Path(selected_embedded_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")
    _ensure_empty_directory(selected_embedded_dir)

    selected_speaker_count = 0
    skipped_speakers: list[str] = []
    moved_file_count = 0

    speaker_dirs = [speaker_dir for speaker_dir in sorted(data_dir.iterdir()) if speaker_dir.is_dir()]
    progress = tqdm(speaker_dirs, desc="Selecting embedded samples", leave=False)
    for speaker_dir in progress:
        speaker_id = speaker_dir.name
        speaker_files = _speaker_files(speaker_dir, DEFAULT_EMBEDDING_FILENAME)

        if len(speaker_files) < min_utt_per_spk:
            skipped_speakers.append(speaker_id)
            LOGGER.warning(
                "Skipping speaker %s because it has %d files, fewer than min_utt_per_spk=%d",
                speaker_id,
                len(speaker_files),
                min_utt_per_spk,
            )
            continue

        sample_size = min(embed_extract_utt_per_spk, len(speaker_files))
        rng = random.Random(f"{seed}:{speaker_id}")
        selected_files = rng.sample(speaker_files, k=sample_size)

        destination_dir = selected_embedded_dir / speaker_id
        for source_path in selected_files:
            _move_file(source_path, destination_dir / source_path.name)
            moved_file_count += 1

        selected_speaker_count += 1
        progress.set_postfix(speaker_id=speaker_id, selected=sample_size)

    LOGGER.info(
        "Selected embedded samples complete: selected_speakers=%d, skipped_speakers=%d, moved_files=%d",
        selected_speaker_count,
        len(skipped_speakers),
        moved_file_count,
    )
    if skipped_speakers:
        LOGGER.info("Skipped speaker_ids: %s", skipped_speakers)

    return {
        "selected_speaker_count": selected_speaker_count,
        "skipped_speaker_count": len(skipped_speakers),
        "moved_file_count": moved_file_count,
        "skipped_speakers": skipped_speakers,
    }


def extract_embedded(
    selected_embedded_dir: str | Path = SELECTED_EMBEDDED_DIR,
    state_dict_path: str | Path = None,
    *,
    output_filename: str = DEFAULT_EMBEDDING_FILENAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
) -> dict[str, Any]:
    selected_embedded_dir = Path(selected_embedded_dir)
    state_dict_path = Path(state_dict_path)
    resolved_device = _resolve_device(device)

    if not selected_embedded_dir.exists():
        raise FileNotFoundError(f"selected_embedded_dir does not exist: {selected_embedded_dir}")
    state_dict_path = _resolve_state_dict_path(state_dict_path)

    model = get_model("spkenc").to(resolved_device)
    model.load_state_dict(_load_model_state_dict(state_dict_path, resolved_device))
    model.eval()

    generated_speaker_count = 0
    skipped_speakers: list[str] = []

    speaker_dirs = [speaker_dir for speaker_dir in sorted(selected_embedded_dir.iterdir()) if speaker_dir.is_dir()]
    progress = tqdm(speaker_dirs, desc="Extracting speaker embeddings", leave=False)
    for speaker_dir in progress:
        speaker_id = speaker_dir.name
        speaker_files = _speaker_files(speaker_dir, output_filename)

        if not speaker_files:
            skipped_speakers.append(speaker_id)
            LOGGER.warning("Skipping speaker %s because no valid mel files were found.", speaker_id)
            continue

        embeddings = _extract_embeddings_for_speaker(
            model=model,
            speaker_files=speaker_files,
            device=resolved_device,
            batch_size=batch_size,
        )

        if not embeddings:
            skipped_speakers.append(speaker_id)
            LOGGER.warning("Skipping speaker %s because embedding extraction failed.", speaker_id)
            continue

        speaker_embedding = torch.stack(embeddings, dim=0).mean(dim=0)
        speaker_embedding = F.normalize(speaker_embedding, p=2, dim=0)
        speaker_embedding = speaker_embedding.to(dtype=torch.float32).cpu()

        embedding_path = speaker_dir / output_filename
        torch.save(speaker_embedding, embedding_path)
        generated_speaker_count += 1
        progress.set_postfix(speaker_id=speaker_id, generated=generated_speaker_count)

    LOGGER.info(
        "Extract embedded complete: generated_speakers=%d, skipped_speakers=%d",
        generated_speaker_count,
        len(skipped_speakers),
    )
    if skipped_speakers:
        LOGGER.info("Skipped speaker_ids: %s", skipped_speakers)

    return {
        "generated_speaker_count": generated_speaker_count,
        "skipped_speaker_count": len(skipped_speakers),
        "skipped_speakers": skipped_speakers,
    }


def reset_embedded_samples(
    selected_embedded_dir: str | Path = SELECTED_EMBEDDED_DIR,
    data_dir: str | Path = PROCESSED_DATA_DIR,
    *,
    output_filename: str = DEFAULT_EMBEDDING_FILENAME,
) -> dict[str, Any]:
    selected_embedded_dir = Path(selected_embedded_dir)
    data_dir = Path(data_dir)

    if not selected_embedded_dir.exists():
        LOGGER.info("selected_embedded_dir does not exist, nothing to reset: %s", selected_embedded_dir)
        return {"restored_file_count": 0}
    if not selected_embedded_dir.is_dir():
        raise NotADirectoryError(f"selected_embedded_dir is not a directory: {selected_embedded_dir}")
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    restored_file_count = 0
    speaker_dirs = [speaker_dir for speaker_dir in sorted(selected_embedded_dir.iterdir()) if speaker_dir.is_dir()]
    progress = tqdm(speaker_dirs, desc="Resetting embedded samples", leave=False)
    for speaker_dir in progress:
        speaker_id = speaker_dir.name
        destination_dir = data_dir / speaker_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        for source_file in sorted(speaker_dir.iterdir()):
            if not source_file.is_file() or source_file.name == output_filename:
                continue
            destination_file = destination_dir / source_file.name
            if destination_file.exists():
                destination_file.unlink()
            shutil.copy2(source_file, destination_file)
            restored_file_count += 1
        progress.set_postfix(speaker_id=speaker_id, restored=restored_file_count)

    shutil.rmtree(selected_embedded_dir)
    LOGGER.info("Reset embedded samples complete: restored_files=%d", restored_file_count)
    return {"restored_file_count": restored_file_count}


def prepare_low_freq_sample(
    data_dir: str | Path,
    low_freq_data_dir: str | Path,
    *,
    cutoff_hz: float = DEFAULT_LOW_FREQ_CUTOFF_HZ,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    low_freq_data_dir = Path(low_freq_data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    speaker_dirs = [speaker_dir for speaker_dir in sorted(data_dir.iterdir()) if speaker_dir.is_dir()]
    if not speaker_dirs:
        LOGGER.warning("No speaker directories found under %s", data_dir)
        low_freq_data_dir.mkdir(parents=True, exist_ok=True)
        return {"processed_count": 0, "skipped_count": 0}

    processed_count = 0
    skipped_count = 0

    progress = tqdm(speaker_dirs, desc="Preparing low-frequency samples", leave=False)
    for speaker_dir in progress:
        speaker_id = speaker_dir.name
        target_speaker_dir = low_freq_data_dir / speaker_id
        target_speaker_dir.mkdir(parents=True, exist_ok=True)

        speaker_files = [
            file_path
            for file_path in sorted(speaker_dir.iterdir())
            if file_path.is_file() and file_path.suffix == ".pt"
        ]
        for source_path in speaker_files:
            target_path = target_speaker_dir / f"{source_path.stem}_low.pt"
            if target_path.exists():
                skipped_count += 1
                continue

            try:
                mel_tensor = _load_tensor(source_path)
                if mel_tensor is None:
                    skipped_count += 1
                    continue

                low_freq_tensor = _apply_low_freq_mask(
                    mel_tensor=mel_tensor,
                    cutoff_hz=cutoff_hz,
                    sample_rate=sample_rate,
                )
                _save_tensor(low_freq_tensor, target_path)
                processed_count += 1
            except Exception as exc:  # pragma: no cover - runtime IO safeguard
                skipped_count += 1
                LOGGER.warning("Failed to prepare low-frequency sample for %s: %s", source_path, exc)

        progress.set_postfix(speaker_id=speaker_id, processed=processed_count)

    LOGGER.info(
        "Low-frequency sample preparation complete: processed=%d, skipped=%d, output_dir=%s",
        processed_count,
        skipped_count,
        low_freq_data_dir,
    )
    return {
        "processed_count": processed_count,
        "skipped_count": skipped_count,
        "output_dir": str(low_freq_data_dir),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage embedded speaker sample preparation.")
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)

    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess dataset.")
    preprocess_parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME)
    preprocess_parser.add_argument("--dataset_sub_name", type=str, default=DEFAULT_DATASET_SUB_NAME)
    preprocess_parser.add_argument("--load_data", type=bool, default=False)

    select_parser = subparsers.add_parser("select", help="Select embedded samples.")
    select_parser.add_argument("--min_utt_per_spk", type=int, default=DEFAULT_MIN_UTT_PER_SPK)
    select_parser.add_argument("--embed_extract_utt_per_spk", type=int, default=DEFAULT_EMBED_EXTRACT_UTT_PER_SPK)
    select_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    extract_parser = subparsers.add_parser("extract", help="Extract speaker embeddings.")
    extract_parser.add_argument("--state_dict_path", type=Path, default=DEFAULT_STATE_DICT_PATH)
    extract_parser.add_argument("--output_filename", type=str, default=DEFAULT_EMBEDDING_FILENAME)
    extract_parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    extract_parser.add_argument("--device", type=str, default=None)

    reset_parser = subparsers.add_parser("reset", help="Reset embedded samples.")
    reset_parser.add_argument("--output_filename", type=str, default=DEFAULT_EMBEDDING_FILENAME)

    low_freq_parser = subparsers.add_parser("prepare_low_freq", help="Prepare low-frequency mel samples.")
    low_freq_parser.add_argument("--low_freq_data_dir", type=Path, default=PROJECT_ROOT / "data" / "low_freq")
    low_freq_parser.add_argument("--cutoff_hz", type=float, default=DEFAULT_LOW_FREQ_CUTOFF_HZ)
    low_freq_parser.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE)

    all_parser = subparsers.add_parser("all", help="Run preprocess, select, extract, and low-frequency preparation.")
    all_parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME)
    all_parser.add_argument("--dataset_sub_name", type=str, default=DEFAULT_DATASET_SUB_NAME)
    all_parser.add_argument("--load_data", type=bool, default=False)
    all_parser.add_argument("--min_utt_per_spk", type=int, default=DEFAULT_MIN_UTT_PER_SPK)
    all_parser.add_argument("--embed_extract_utt_per_spk", type=int, default=DEFAULT_EMBED_EXTRACT_UTT_PER_SPK)
    all_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    all_parser.add_argument("--state_dict_path", type=Path, default=DEFAULT_STATE_DICT_PATH)
    all_parser.add_argument("--output_filename", type=str, default=DEFAULT_EMBEDDING_FILENAME)
    all_parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    all_parser.add_argument("--device", type=str, default=None)
    all_parser.add_argument("--low_freq_data_dir", type=Path, default=PROJECT_ROOT / "data" / "low_freq")
    all_parser.add_argument("--cutoff_hz", type=float, default=DEFAULT_LOW_FREQ_CUTOFF_HZ)
    all_parser.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = build_arg_parser()
    args = parser.parse_args()

    build_dir_path(args.project_root)

    if args.command == "select":
        select_embedded_samples(
            min_utt_per_spk=args.min_utt_per_spk,
            embed_extract_utt_per_spk=args.embed_extract_utt_per_spk,
            seed=args.seed,
        )
    elif args.command == "extract":
        extract_embedded(
            state_dict_path=args.state_dict_path,
            output_filename=args.output_filename,
            batch_size=args.batch_size,
            device=args.device,
        )
    elif args.command == "reset":
        reset_embedded_samples(
            output_filename=args.output_filename,
        )
    elif args.command == "prepare_low_freq":
        prepare_low_freq_sample(
            cutoff_hz=args.cutoff_hz,
            sample_rate=args.sample_rate,
        )
    elif args.command == "preprocess":
        preprocess_dataset(
            dataset_name=args.dataset_name,
            dataset_sub_name=args.dataset_sub_name,
            save_dir=args.save_dir,
        )
    elif args.command == "all":
        preprocess_dataset(
            dataset_name=args.dataset_name,
            dataset_sub_name=args.dataset_sub_name,
            save_dir=args.save_dir,
        )
        select_embedded_samples(
            min_utt_per_spk=args.min_utt_per_spk,
            embed_extract_utt_per_spk=args.embed_extract_utt_per_spk,
            seed=args.seed,
        )
        extract_embedded(
            state_dict_path=args.state_dict_path,
            output_filename=args.output_filename,
            batch_size=args.batch_size,
            device=args.device,
        )
        prepare_low_freq_sample(
            data_dir=args.data_dir,
            low_freq_data_dir=args.low_freq_data_dir,
            cutoff_hz=args.cutoff_hz,
            sample_rate=args.sample_rate,
        )


if __name__ == "__main__":
    main()

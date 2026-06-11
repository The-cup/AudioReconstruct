from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

from audio_reconstruct.data.preprocess.utils import (
    ensure_directory,
    load_audio_file,
    resample_audio,
    slice_waveform_into_segments,
    waveform_to_log_mel,
    TARGET_SAMPLE_RATE,
)
from audio_reconstruct.datasets.audio_dataset import (
    DATASET_MODULE_DIR,
    LibriSpeechDataset,
    AudioReconstructionDataset,
)


LOGGER = logging.getLogger(__name__)

PROCESSED_DATASETS_ROOT = Path(__file__).resolve().parents[4] / "data" / "processed"
DEFAULT_DATASET_SUB_NAME = "train-clean-100"


def _get_item_value(item: Any, key: str, default: Any = None) -> Any:
    if hasattr(item, key):
        return getattr(item, key)
    if isinstance(item, dict):
        return item.get(key, default)
    if isinstance(item, (tuple, list)):
        index_map = {"file": 0, "label": 1}
        if key in index_map and len(item) > index_map[key]:
            return item[index_map[key]]
    return default


def _write_metadata(index_file: Path, processed_items: list[dict[str, str]]) -> None:
    ensure_directory(index_file.parent)
    with index_file.open("w", encoding="utf-8") as handle:
        for item in processed_items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def run_preprocessing_pipeline(
    dataset: AudioReconstructionDataset,
    save_dir: Path | None = None,
) -> LibriSpeechDataset:
    """Process raw LibriSpeech audio into 40-band log-mel features.

    Args:
        dataset: Raw LibriSpeech dataset object.
        save_dir: Directory to save processed tensors and metadata. If None,
            the pipeline only computes features in memory and skips persistence.

    Returns:
        A processed LibriSpeechDataset instance indexed by metadata.
    """
    output_root = Path(save_dir) if save_dir is not None else None

    processed_items: list[dict[str, str]] = []
    total_source_items = len(dataset)
    LOGGER.info("Starting preprocessing for %d raw utterances", total_source_items)

    for raw_item in dataset:
        audio_path = Path(_get_item_value(raw_item, "file"))
        label = str(_get_item_value(raw_item, "label"))

        try:
            waveform, sample_rate = load_audio_file(audio_path)
        except Exception as exc:  # pragma: no cover - runtime IO safeguard
            LOGGER.error("Failed to read %s: %s", audio_path, exc)
            continue

        if sample_rate is None:
            LOGGER.warning("Missing sample rate for %s, defaulting to %d Hz", audio_path, TARGET_SAMPLE_RATE)
            sample_rate = TARGET_SAMPLE_RATE

        if sample_rate < TARGET_SAMPLE_RATE:
            LOGGER.error(
                "Skipping %s because sample rate %d Hz is below %d Hz",
                audio_path,
                sample_rate,
                TARGET_SAMPLE_RATE,
            )
            continue

        waveform = waveform.float()
        waveform = resample_audio(waveform, sample_rate, TARGET_SAMPLE_RATE)
        waveform = waveform.cpu()

        segments = slice_waveform_into_segments(waveform)
        for segment_index, segment in enumerate(segments):
            mel_feature = waveform_to_log_mel(segment, sample_rate=TARGET_SAMPLE_RATE)

            if output_root is not None:
                # tensor_path = _build_processed_file_path(
                #     output_dir=output_root,
                #     label=label,
                #     segment_index=segment_index,
                # )
                tensor_path = output_root / f"{label}_seg{segment_index:04d}.pt"
                ensure_directory(tensor_path.parent)
                torch.save(mel_feature.to(torch.float32), tensor_path)

                processed_items.append(
                    {
                        "label": tensor_path.stem,
                        "file": str(tensor_path),
                        "text": _get_item_value(raw_item, "text")
                    }
                )
    transcript_path = output_root / "transcript.txt"
    with transcript_path.open("w", encoding="utf-8") as f:
        for item in processed_items:
            f.write(f'{item["label"]} {item["text"]}\n')


    LOGGER.info("Preprocessing finished without persistence because save_dir is None.")
    return LibriSpeechDataset(base_dir=PROCESSED_DATASETS_ROOT, dataset_sub_name=None)

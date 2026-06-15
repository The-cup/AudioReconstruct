from __future__ import annotations

import json
import logging
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
    AudioReconstructionDataset, SpkEncDataset, PROCESSED_DATASETS_DIR,
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


def _is_empty_dir(p: Path) -> bool:
    # 先判断是不是目录，防止传入文件报错
    if not p.is_dir():
        raise NotADirectoryError(f"{p} 不是文件夹")
    # 遍历第一个元素，无元素则为空
    return next(p.iterdir(), None) is None


def run_spkenc_preprocessing_pipeline(
    dataset: AudioReconstructionDataset,
    save_dir: Path = PROCESSED_DATASETS_DIR,
) -> SpkEncDataset:
    """Process raw LibriSpeech audio into 40-band log-mel features.

    Args:
        dataset: Raw LibriSpeech dataset object.
        save_dir: Directory to save processed tensors and metadata. If None,
            the pipeline only computes features in memory and skips persistence.

    Returns:
        A processed LibriSpeechDataset instance indexed by metadata.
    """

    spkenc_dataset = SpkEncDataset()

    ensure_directory(save_dir)
    if not _is_empty_dir(save_dir):
        LOGGER.info("Loading processed dataset from %s", save_dir)
        spkenc_dataset.build_from_dir(save_dir)
        return spkenc_dataset

    LOGGER.info("Starting preprocessing for %d raw utterances", len(dataset))

    for raw_item in dataset:
        audio_path = Path(_get_item_value(raw_item, "file"))
        label = str(_get_item_value(raw_item, "label"))
        speaker_id = str(_get_item_value(raw_item, "speaker_id"))

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
            tensor_path = save_dir / speaker_id / f"{label}_seg{segment_index:04d}.pt"
            torch.save(mel_feature.to(torch.float32), tensor_path)

            spkenc_dataset.add_item(speaker_id=speaker_id, file_path=tensor_path)

    LOGGER.info(f"Preprocessing complete, {len(spkenc_dataset)} items saved at {save_dir}")
    return spkenc_dataset

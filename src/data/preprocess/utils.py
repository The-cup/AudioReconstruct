from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
from pydub import AudioSegment

try:  # pragma: no cover - optional fallback path
    import soundfile as sf
except ImportError:  # pragma: no cover - optional fallback path
    sf = None


TARGET_SAMPLE_RATE = 16_000
TARGET_SEGMENT_SECONDS = 1.6
TARGET_SEGMENT_SAMPLES = int(TARGET_SAMPLE_RATE * TARGET_SEGMENT_SECONDS)
TARGET_OVERLAP_SECONDS = 0.8
TARGET_OVERLAP_SAMPLES = int(TARGET_SAMPLE_RATE * TARGET_OVERLAP_SECONDS)
TARGET_FRAME_COUNT = 160
TARGET_FRAME_LENGTH_MS = 25
TARGET_FRAME_SHIFT_MS = 10
TARGET_N_MELS = 40
TARGET_N_FFT = 512
EPSILON = 1e-10


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_audio_file(file_path: str | Path) -> tuple[torch.Tensor, int]:
    """
    加载音频文件，兼容 WAV, FLAC, M4A, MP3, AAC 等所有主流格式
    返回格式：(waveform: [Channels, Samples], sample_rate: int)
    与 torchaudio.load 输出完全对齐，保证后续代码零修改
    """
    file_path = str(file_path)

    # ========== 原有逻辑 1：torchaudio 加载（支持wav/flac）==========
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        return waveform, int(sample_rate or TARGET_SAMPLE_RATE)
    except Exception:
        pass

    # ========== 原有逻辑 2：soundfile 兜底加载 ==========
    try:
        if sf is not None:
            data, sample_rate = sf.read(file_path, always_2d=True)
            waveform = torch.from_numpy(data.T).float()
            return waveform, int(sample_rate or TARGET_SAMPLE_RATE)
    except Exception:
        pass

    # ========== 新增逻辑 3：pydub + ffmpeg 兜底（支持m4a/mp3/aac）==========
    try:
        # 用 ffmpeg 读取所有压缩音频格式
        audio = AudioSegment.from_file(file_path)
        # 转换为 numpy 数组，并归一化到 [-1, 1]（和 torchaudio 格式完全一致）
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / (2 ** (8 * audio.sample_width - 1))  # 归一化

        # 处理通道维度：[channels, length]
        if audio.channels == 1:
            waveform = torch.from_numpy(samples).unsqueeze(0)
        else:
            # 双声道：重塑为 [2, length]
            waveform = torch.from_numpy(samples.reshape(audio.channels, -1))

        return waveform, audio.frame_rate

    # 所有方式都失败，抛出明确错误
    except Exception as e:
        raise RuntimeError(
            f"Failed to load audio file: {file_path}\n"
            f"Supported format: wav, flac, m4a, mp3, aac, ogg\n"
            f"Error: {str(e)}"
        ) from e


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        return waveform.unsqueeze(0)
    if waveform.shape[0] == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def resample_audio(waveform: torch.Tensor, sample_rate: int, target_sample_rate: int = TARGET_SAMPLE_RATE) -> torch.Tensor:
    if sample_rate == target_sample_rate:
        return waveform
    return F.resample(waveform, orig_freq=sample_rate, new_freq=target_sample_rate)


def pad_or_trim_waveform(
    waveform: torch.Tensor,
    target_samples: int = TARGET_SEGMENT_SAMPLES,
) -> torch.Tensor:
    current_samples = waveform.shape[-1]
    if current_samples == target_samples:
        return waveform
    if current_samples > target_samples:
        return waveform[..., :target_samples]

    pad_left = target_samples - current_samples
    return torch.nn.functional.pad(waveform, (pad_left, 0))


def slice_waveform_into_segments(
    waveform: torch.Tensor,
    segment_samples: int = TARGET_SEGMENT_SAMPLES,
    overlap_samples: int = TARGET_OVERLAP_SAMPLES,
) -> list[torch.Tensor]:
    total_samples = waveform.shape[-1]
    if total_samples <= segment_samples:
        return [pad_or_trim_waveform(waveform, segment_samples)]

    hop_samples = max(segment_samples - overlap_samples, 1)
    starts = list(range(0, total_samples - segment_samples + 1, hop_samples))
    final_start = total_samples - segment_samples
    if not starts or starts[-1] != final_start:
        starts.append(final_start)

    segments = [waveform[..., start : start + segment_samples] for start in starts]
    return segments


def waveform_to_log_mel(
    waveform: torch.Tensor,
    sample_rate: int = TARGET_SAMPLE_RATE,
    target_frames: int = TARGET_FRAME_COUNT,
    n_mels: int = TARGET_N_MELS,
    n_fft: int = TARGET_N_FFT,
    win_length_ms: int = TARGET_FRAME_LENGTH_MS,
    hop_length_ms: int = TARGET_FRAME_SHIFT_MS,
) -> torch.Tensor:
    waveform = to_mono(waveform)
    win_length = int(sample_rate * win_length_ms / 1000)
    hop_length = int(sample_rate * hop_length_ms / 1000)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
        center=True,
    )
    mel = mel_transform(waveform)
    mel = mel.squeeze(0).transpose(0, 1)
    mel = ensure_frame_count(mel, target_frames=target_frames)
    return torch.log(mel.clamp_min(EPSILON))


def ensure_frame_count(feature: torch.Tensor, target_frames: int = TARGET_FRAME_COUNT) -> torch.Tensor:
    current_frames = feature.shape[0]
    if current_frames == target_frames:
        return feature
    if current_frames > target_frames:
        return feature[:target_frames]

    pad_frames = target_frames - current_frames
    pad = torch.zeros((pad_frames, feature.shape[1]), dtype=feature.dtype, device=feature.device)
    return torch.cat([feature, pad], dim=0)

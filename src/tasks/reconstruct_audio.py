"""
reconstruct_audio.py
====================
推理阶段工具：将受损（低频）音频恢复为完整频谱并重建波形。

公开函数
--------
reconstruct_spec              — 单段梅尔频谱重建（调用 GAN 生成器）
concat_spec                   — 将推理段拼接为完整梅尔频谱
reconstruct_audio_spec        — 端到端处理一段原始音频，生成重建后的梅尔谱文件
reconstruct_audio_from_spec   — 从梅尔谱 .pt 文件重建音频波形
reconstruct_audio_from_folder — 批量处理 RECONSTRUCTED_RAW_DIR 下所有说话人的音频
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Literal, Optional, Dict

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from torch import Tensor
import torch.nn.functional as F

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from data.preprocess.utils import (
    load_audio_file,
    resample_audio,
    slice_waveform_into_segments,
    waveform_to_log_mel,
    TARGET_SAMPLE_RATE,
    TARGET_N_MELS,
    TARGET_N_FFT,
    TARGET_FRAME_LENGTH_MS,
    TARGET_FRAME_SHIFT_MS,
)
from models.custom import VoiceExpandGenerator
from models.custom.voice_expand_gan import VoiceExpandGAN
from models.registry import get_model

LOGGER = logging.getLogger(__name__)

# 梅尔谱参数（与预处理管线保持一致）
_WIN_LENGTH: int = int(TARGET_SAMPLE_RATE * TARGET_FRAME_LENGTH_MS / 1000)  # 400 samples
_HOP_LENGTH: int = int(TARGET_SAMPLE_RATE * TARGET_FRAME_SHIFT_MS / 1000)   # 160 samples
_N_STFT: int = TARGET_N_FFT // 2 + 1                                        # 257

BlendMode = Literal["linear", "l2_weighted"]

# ---------------------------------------------------------------------------
# 目标 1 — 单段频谱重建
# ---------------------------------------------------------------------------

def reconstruct_spec(
    model: VoiceExpandGenerator | None = None,
    speaker_embedding: torch.Tensor | None = None,
    low_freq_mel_spec: torch.Tensor | None = None,
    spec_path: Path | None = None,
) -> torch.Tensor:
    """用加载好参数的 VoiceExpandGAN 将单段低频梅尔谱重建为完整梅尔谱。

    参数
    ----
    model             : 已加载权重的 VoiceExpandGAN；为 None 时抛出 ValueError。
    speaker_embedding : 形状 (256,) 的说话人嵌入向量；为 None 时使用随机向量。
    low_freq_mel_spec : 形状 (160, 40) 的低频梅尔谱；为 None 时抛出 ValueError。
    spec_path         : 若非 None，将结果以 .pt 格式保存至该路径。

    返回
    ----
    形状 (160, 40) 的重建梅尔谱张量（CPU，float32）。
    """
    if model is None:
        raise ValueError("'model' must not be None.")
    if low_freq_mel_spec is None:
        raise ValueError("'low_freq_mel_spec' must not be None.")

    # 说话人嵌入：None 时使用随机向量
    if speaker_embedding is None:
        LOGGER.warning("'speaker_embedding' is None — using a random embedding vector.")
        speaker_embedding = torch.randn(256)

    # 推断模型所在设备
    device = next(model.parameters()).device

    # 维度适配：(160, 40) → (1, 1, 160, 40)
    lf = low_freq_mel_spec.float().unsqueeze(0).unsqueeze(0).to(device)   # (1,1,160,40)
    emb = speaker_embedding.float().unsqueeze(0).to(device)               # (1, 256)

    # GAN 推理
    model.eval()
    with torch.no_grad():
        generated = model(lf, emb)   # (1, 1, 160, 40)

    # 维度还原：(1, 1, 160, 40) → (160, 40)，移回 CPU
    result = generated.squeeze(0).squeeze(0).cpu()   # (160, 40)

    # 可选：保存
    if spec_path is not None:
        spec_path = Path(spec_path)
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, spec_path)
        LOGGER.info("Reconstructed spec saved to %s", spec_path)

    return result


# ---------------------------------------------------------------------------
# 目标 2 — 梅尔频谱拼接
# ---------------------------------------------------------------------------

def concat_spec(
    spec: torch.Tensor,
    other_spec: torch.Tensor,
    window_size: int = 25,
    overlap: int = 10,
    blend_mode: BlendMode = "linear",
) -> torch.Tensor:
    """将推理段 other_spec 拼接到已有频谱 spec 末尾，边界处进行平滑混合。

    参数
    ----
    spec        : 已拼接的频谱，形状 (T, n_mels)。
    other_spec  : 待追加的频谱，形状 (T', n_mels)。
    window_size : STFT 帧长（ms），默认 25。
    overlap     : STFT 帧间重叠（ms），默认 10。
    blend_mode  : 边界混合方式：
                  "linear"     — 线性淡出 / 淡入（默认）；
                  "l2_weighted"— L2 范数加权混合。

    返回
    ----
    拼接后的频谱，形状 (T + T' - n_blend, n_mels)。

    注：混合帧数计算
    ----
    hop_ms         = window_size - overlap  = 15 ms
    n_blend_frames = round(overlap / hop_ms) = round(10/15) = 1
    若 n_blend_frames == 0，则直接拼接，不做混合。
    """
    # 混合帧数：边界处重叠的帧数
    hop_ms = window_size - overlap           # 15 ms
    n_blend = max(0, round(overlap / hop_ms)) if hop_ms > 0 else 0  # 1 frame

    if n_blend == 0 or spec.shape[0] < n_blend or other_spec.shape[0] < n_blend:
        return torch.cat([spec, other_spec], dim=0)

    # 边界区域：spec 末尾 n_blend 帧 & other_spec 开头 n_blend 帧
    tail = spec[-n_blend:]          # (n_blend, n_mels)
    head = other_spec[:n_blend]     # (n_blend, n_mels)

    if blend_mode == "l2_weighted":
        # L2 范数加权：范数大的片段权重更低，保留较弱的一方（融合两侧信息）
        w_tail = tail.norm(dim=1, keepdim=True).clamp_min(1e-8)   # (n_blend, 1)
        w_head = head.norm(dim=1, keepdim=True).clamp_min(1e-8)   # (n_blend, 1)
        blended = (w_head * tail + w_tail * head) / (w_tail + w_head)
    else:
        # 线性淡出 / 淡入：α 从 0 → 1 控制 other_spec 的权重
        alpha = torch.linspace(0.0, 1.0, n_blend, dtype=spec.dtype).unsqueeze(1)  # (n_blend, 1)
        blended = (1.0 - alpha) * tail + alpha * head

    return torch.cat([spec[:-n_blend], blended, other_spec[n_blend:]], dim=0)


# ---------------------------------------------------------------------------
# 目标 3 — 完整音频 → 重建梅尔谱
# ---------------------------------------------------------------------------

def reconstruct_audio_spec(
    model: VoiceExpandGAN | None = None,
    speaker_embedding: torch.Tensor | None = None,
    audio_path: Path | None = None,
    spec_path: Path | None = None,
    blend_mode: BlendMode = "linear",
) -> torch.Tensor | None:
    """将一段原始音频拆分为帧段，逐段经 GAN 推理后拼接为完整重建梅尔谱。

    参数
    ----
    model             : 已加载权重的 VoiceExpandGAN；为 None 时抛出 ValueError。
    speaker_embedding : 形状 (256,) 的说话人嵌入向量；为 None 时每段均使用随机向量。
    audio_path        : 原始音频文件路径；为 None 时抛出 ValueError。
    spec_path         : 若非 None，覆盖自动生成的保存路径。

    返回
    ----
    拼接后的完整重建梅尔谱（float32，CPU），或在不满足采样率要求时返回 None。

    自动输出路径
    ----------
    RECONSTRUCTED_SPEC_DIR / <audio 相对 RAW_DATA_DIR 的路径>.parent / <stem>.pt
    若 RAW_DATA_DIR 为 None 或无法计算相对路径，则保存至 RECONSTRUCTED_SPEC_DIR / <stem>.pt。
    """
    if model is None:
        raise ValueError("'model' must not be None.")
    if audio_path is None:
        raise ValueError("'audio_path' must not be None.")

    audio_path = Path(audio_path)

    # ------------------------------------------------------------------
    # 1. 加载音频，统一采样率到 16 kHz（与 pipeline.py 处理逻辑对齐）
    # ------------------------------------------------------------------
    waveform, sr = load_audio_file(audio_path)

    if sr is None:
        LOGGER.warning("Missing sample rate for %s, defaulting to %d Hz", audio_path, TARGET_SAMPLE_RATE)
        sr = TARGET_SAMPLE_RATE

    if sr < TARGET_SAMPLE_RATE:
        LOGGER.error(
            "Skipping %s because sample rate %d Hz is below %d Hz",
            audio_path, sr, TARGET_SAMPLE_RATE,
        )
        return None

    waveform = waveform.float()
    waveform = resample_audio(waveform, sr, TARGET_SAMPLE_RATE)
    waveform = waveform.cpu()

    # ------------------------------------------------------------------
    # 2. 切分波形为 1.6s 段（默认参数与 pipeline.py 保持一致）
    # ------------------------------------------------------------------
    segments = slice_waveform_into_segments(waveform)
    LOGGER.info("Audio '%s' split into %d segment(s).", audio_path.name, len(segments))

    # ------------------------------------------------------------------
    # 3. 逐段: 波形 → log-mel (160×40) → GAN 推理 → 拼接
    # ------------------------------------------------------------------
    full_spec: torch.Tensor | None = None

    for idx, segment in enumerate(segments):
        # 波形段 → log-mel (160, 40)，参数与 pipeline.py 一致
        log_mel = waveform_to_log_mel(segment, sample_rate=TARGET_SAMPLE_RATE)

        # GAN 推理（spec_path=None：不单独保存每段结果）
        reconstructed = reconstruct_spec(
            model=model,
            speaker_embedding=speaker_embedding,
            low_freq_mel_spec=log_mel,
        )

        # 拼接
        if full_spec is None:
            full_spec = reconstructed
        else:
            full_spec = concat_spec(full_spec, reconstructed, window_size=25, overlap=10, blend_mode=blend_mode)

        LOGGER.debug("Segment %d/%d processed.", idx + 1, len(segments))

    if full_spec is None:
        LOGGER.warning("No segments produced for '%s'.", audio_path)
        return None

    # ------------------------------------------------------------------
    # 4. 确定输出路径并保存
    # ------------------------------------------------------------------
    if spec_path is not None:
        target_path = Path(spec_path)
    else:
        # 自动构建路径：RECONSTRUCTED_SPEC_DIR / <相对路径>.parent / stem.pt
        try:
            from config.paths import RAW_DATA_DIR
            if RAW_DATA_DIR is not None:
                rel = audio_path.relative_to(RAW_DATA_DIR)
                rel_dir = rel.parent
            else:
                raise ValueError("RAW_DATA_DIR is None")
        except ValueError:
            rel_dir = Path(".")

        from config.paths import RECONSTRUCTED_SPEC_DIR
        base_dir = RECONSTRUCTED_SPEC_DIR if RECONSTRUCTED_SPEC_DIR is not None else Path(".")
        target_path = base_dir / rel_dir / f"{audio_path.stem}.pt"

    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(full_spec, target_path)
    LOGGER.info("Full reconstructed spec saved to %s  (shape %s).", target_path, tuple(full_spec.shape))

    return full_spec


# ---------------------------------------------------------------------------
# 目标 4 — 梅尔谱 → 重建音频波形
# ---------------------------------------------------------------------------

def reconstruct_audio_from_spec(
    spec_path: Path | None = None,
    reconstructed_audio_path: Path | None = None,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> torch.Tensor | None:
    """从保存的对数梅尔谱 .pt 文件重建音频波形并保存。

    参数
    ----
    spec_path                : 梅尔谱 .pt 文件路径；为 None 时抛出 ValueError。
    reconstructed_audio_path : 重建音频的保存路径（.wav）；为 None 时抛出 ValueError。
    sample_rate              : 目标采样率，默认 16000 Hz。

    返回
    ----
    形状 (1, samples) 的音频波形张量（CPU，float32），或失败时返回 None。

    逆变换链
    --------
    .pt (log-mel, (T, 40))
    → torch.exp()                           → linear mel (T, 40)
    → transpose + unsqueeze                 → (1, 40, T)
    → InverseMelScale(n_stft=257, n_mels=40)→ linear STFT magnitude (1, 257, T)
    → GriffinLim(n_fft=512, hop=160, win=400)→ waveform (1, samples)
    → torchaudio.save()
    """
    if spec_path is None:
        raise ValueError("'spec_path' must not be None.")
    if reconstructed_audio_path is None:
        raise ValueError("'reconstructed_audio_path' must not be None.")

    spec_path = Path(spec_path)
    reconstructed_audio_path = Path(reconstructed_audio_path)

    if not spec_path.exists():
        LOGGER.error("Spec file not found: %s", spec_path)
        return None

    # ------------------------------------------------------------------
    # 1. 加载对数梅尔谱并转回线性域
    # ------------------------------------------------------------------
    log_mel: torch.Tensor = torch.load(spec_path, map_location="cpu", weights_only=True)
    # 期望形状 (T, n_mels)；若存的是 (160, 40) 则 T=160
    if log_mel.ndim != 2:
        LOGGER.error(
            "Expected a 2-D mel spectrogram tensor from '%s', got shape %s.",
            spec_path, tuple(log_mel.shape),
        )
        return None

    # log → linear mel
    linear_mel = torch.exp(log_mel)                    # (T, 40)
    linear_mel = linear_mel.transpose(0, 1).unsqueeze(0).float()  # (1, 40, T)

    # ------------------------------------------------------------------
    # 2. InverseMelScale: linear mel (1, 40, T) → linear STFT magnitude (1, 257, T)
    # ------------------------------------------------------------------
    inverse_mel = T.InverseMelScale(
        n_stft=_N_STFT,          # 257 = TARGET_N_FFT // 2 + 1
        n_mels=TARGET_N_MELS,    # 40
        sample_rate=sample_rate,
    )
    linear_stft = inverse_mel(linear_mel)   # (1, 257, T)

    # ------------------------------------------------------------------
    # 3. Griffin-Lim: linear STFT magnitude (1, 257, T) → waveform (1, samples)
    # ------------------------------------------------------------------
    griffin_lim = T.GriffinLim(
        n_fft=TARGET_N_FFT,      # 512
        win_length=_WIN_LENGTH,  # 400
        hop_length=_HOP_LENGTH,  # 160
        power=2.0,               # 与 MelSpectrogram(power=2.0) 一致
    )
    waveform = griffin_lim(linear_stft)   # (1, samples)

    # ------------------------------------------------------------------
    # 4. 保存音频
    # ------------------------------------------------------------------
    reconstructed_audio_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(
        str(reconstructed_audio_path),
        waveform,
        sample_rate,
    )
    LOGGER.info(
        "Reconstructed audio saved to %s  (%d samples, %.2f s).",
        reconstructed_audio_path,
        waveform.shape[-1],
        waveform.shape[-1] / sample_rate,
    )

    return waveform


# ---------------------------------------------------------------------------
# 目标 5 — 批量处理文件夹（全量推理入口）
# ---------------------------------------------------------------------------

_EMBEDDED_VECTOR_FILENAME = "embedded_vector.pt"
_SUPPORTED_AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".wma"})


def reconstruct_audio_from_folder(generator: VoiceExpandGAN, blend_mode: BlendMode) -> None:
    """批量处理 RECONSTRUCTED_RAW_DIR 下所有说话人的音频，完成频谱重建与波形还原。

    目录结构约定
    ----------
    RECONSTRUCTED_RAW_DIR/
    └── {speaker_id}/
        └── {audio_name}.wav          ← 待重建的低频/受损音频

    RECONSTRUCTED_EMBEDDING_DIR/
    └── {speaker_id}/
        └── embedded_vector.pt        ← 说话人嵌入向量（可选，缺失时使用随机向量）

    输出目录（由 config/paths.py 配置）
    ----------
    RECONSTRUCTED_SPEC_DIR/{speaker_id}/{audio_name}.pt    ← 重建梅尔谱
    RECONSTRUCTED_AUDIO_DIR/{speaker_id}/{audio_name}.wav  ← 重建音频

    参数
    ----
    generator : 已加载权重的 VoiceExpandGAN。
    """
    from config.paths import RECONSTRUCTED_RAW_DIR
    if RECONSTRUCTED_RAW_DIR is None:
        raise ValueError("RECONSTRUCTED_RAW_DIR is None — call build_dir_path() first.")
    if not RECONSTRUCTED_RAW_DIR.exists():
        LOGGER.error("RECONSTRUCTED_RAW_DIR does not exist: %s", RECONSTRUCTED_RAW_DIR)
        return

    speaker_dirs = sorted(p for p in RECONSTRUCTED_RAW_DIR.iterdir() if p.is_dir())
    if not speaker_dirs:
        LOGGER.warning("No speaker directories found in %s", RECONSTRUCTED_RAW_DIR)
        return

    LOGGER.info("Found %d speaker(s) in %s", len(speaker_dirs), RECONSTRUCTED_RAW_DIR)

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name

        if speaker_id not in ["zkh"]:
            continue

        # ------------------------------------------------------------------
        # 加载说话人嵌入向量；路径不存在时与 None 同等处理（使用随机向量）
        # ------------------------------------------------------------------
        from config.paths import RECONSTRUCTED_EMBEDDING_DIR
        embedding_path = (
            RECONSTRUCTED_EMBEDDING_DIR / speaker_id / _EMBEDDED_VECTOR_FILENAME
            if RECONSTRUCTED_EMBEDDING_DIR is not None
            else None
        )
        if embedding_path is not None and embedding_path.exists():
            speaker_embedding: torch.Tensor | None = torch.load(
                embedding_path, map_location="cpu", weights_only=True
            )
            LOGGER.info("[%s] Loaded speaker embedding from %s", speaker_id, embedding_path)
        else:
            speaker_embedding = None
            LOGGER.warning(
                "[%s] Embedding not found at %s — will use random vector.", speaker_id, embedding_path
            )

        # ------------------------------------------------------------------
        # 枚举该说话人下的所有音频文件
        # ------------------------------------------------------------------
        audio_files = []
        audio_label_set = set()
        for f in speaker_dir.iterdir():
            if f.is_file() and f.suffix.lower() in _SUPPORTED_AUDIO_SUFFIXES \
                and f.name.lower() not in audio_label_set:
                audio_files.append(f)
                audio_label_set.add(f.name.lower())

        # audio_files = sorted(
        #     f for f in speaker_dir.iterdir()
        #     if f.is_file() and f.suffix.lower() in _SUPPORTED_AUDIO_SUFFIXES
        # )
        if not audio_files:
            LOGGER.warning("[%s] No audio files found in %s", speaker_id, speaker_dir)
            continue

        LOGGER.info("[%s] Processing %d audio file(s).", speaker_id, len(audio_files))

        for audio_file in audio_files:
            audio_name = audio_file.stem
            from config.paths import RECONSTRUCTED_SPEC_DIR
            spec_out = (
                RECONSTRUCTED_SPEC_DIR / speaker_id / f"{audio_name}.pt"
                if RECONSTRUCTED_SPEC_DIR is not None
                else None
            )
            from config.paths import RECONSTRUCTED_AUDIO_DIR
            audio_out = (
                RECONSTRUCTED_AUDIO_DIR / speaker_id / f"{audio_name}.wav"
                if RECONSTRUCTED_AUDIO_DIR is not None
                else None
            )

            # ---- 步骤1：音频 → 重建梅尔谱 --------------------------------
            LOGGER.info("[%s] reconstruct_audio_spec: %s", speaker_id, audio_file.name)
            reconstruct_audio_spec(
                model=generator,
                speaker_embedding=speaker_embedding,
                audio_path=audio_file,
                spec_path=spec_out,
                blend_mode=blend_mode
            )

            # ---- 步骤2：重建梅尔谱 → 音频波形 ----------------------------
            if spec_out is None or not spec_out.exists():
                LOGGER.warning(
                    "[%s] Spec file missing after reconstruct_audio_spec, skipping audio export: %s",
                    speaker_id, audio_file.name,
                )
                continue

            LOGGER.info("[%s] reconstruct_audio_from_spec: %s", speaker_id, audio_file.name)
            reconstruct_audio_from_spec(
                spec_path=spec_out,
                reconstructed_audio_path=audio_out,
                sample_rate=TARGET_SAMPLE_RATE,
            )

    LOGGER.info("reconstruct_audio_from_folder complete.")


def plot_mel_spec_from_dir() -> None:
    """
    遍历 RECONSTRUCTED_SPEC_DIR 下所有说话人目录的梅尔谱 .pt 文件，绘制频谱图并保存至 RECONSTRUCTED_PLOT_DIR 对应路径。

    目录映射规则
    ----------1
    输入: RECONSTRUCTED_SPEC_DIR/{speaker_id}/{audio_name}.pt
    输出: RECONSTRUCTED_PLOT_DIR/{speaker_id}/{audio_name}.png
    """
    # 检查输入目录
    from config.paths import RECONSTRUCTED_SPEC_DIR, RECONSTRUCTED_PLOT_DIR
    if RECONSTRUCTED_SPEC_DIR is None:
        raise ValueError("RECONSTRUCTED_SPEC_DIR is None — call build_dir_path() first.")
    if not RECONSTRUCTED_SPEC_DIR.exists():
        LOGGER.error("RECONSTRUCTED_SPEC_DIR does not exist: %s", RECONSTRUCTED_SPEC_DIR)
        return

    # 检查输出目录配置
    if RECONSTRUCTED_PLOT_DIR is None:
        raise ValueError("RECONSTRUCTED_PLOT_DIR is None — check config/paths.py")
    RECONSTRUCTED_PLOT_DIR = Path(RECONSTRUCTED_PLOT_DIR)

    # 遍历所有说话人目录
    speaker_dirs = sorted(p for p in RECONSTRUCTED_SPEC_DIR.iterdir() if p.is_dir())
    if not speaker_dirs:
        LOGGER.warning("No speaker directories found in %s", RECONSTRUCTED_SPEC_DIR)
        return

    LOGGER.info("Found %d speaker(s) in %s", len(speaker_dirs), RECONSTRUCTED_SPEC_DIR)

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name
        LOGGER.info("[%s] Starting to plot mel spectrograms", speaker_id)

        # 遍历当前说话人下的所有 .pt 梅尔谱文件
        spec_files = sorted(
            f for f in speaker_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".pt"
        )
        if not spec_files:
            LOGGER.warning("[%s] No mel spec .pt files found in %s", speaker_id, speaker_dir)
            continue

        # 创建当前说话人的输出目录
        speaker_plot_dir = RECONSTRUCTED_PLOT_DIR / speaker_id
        speaker_plot_dir.mkdir(parents=True, exist_ok=True)

        for spec_file in spec_files:
            audio_name = spec_file.stem
            plot_save_path = speaker_plot_dir / f"{audio_name}.png"

            try:
                # 加载梅尔谱张量
                mel_spec: torch.Tensor = torch.load(
                    spec_file, map_location="cpu", weights_only=True
                )

                # 校验张量维度 (T, n_mels)
                if mel_spec.ndim != 2:
                    LOGGER.error(
                        "[%s] Invalid tensor shape for %s: expected 2D (T, n_mels), got %s",
                        speaker_id, spec_file.name, tuple(mel_spec.shape)
                    )
                    continue

                # 转换为 numpy 数组（便于绘图）
                mel_spec_np = mel_spec.numpy()

                # 计算时间轴（单位：秒）
                n_frames = mel_spec_np.shape[0]
                time_axis = np.arange(n_frames) * _HOP_LENGTH / TARGET_SAMPLE_RATE

                # 绘制梅尔谱图
                plt.figure(figsize=(12, 6))
                plt.imshow(
                    mel_spec_np.T,  # 转置为 (n_mels, T) 以匹配 imshow 维度
                    aspect="auto",
                    origin="lower",
                    cmap="viridis",
                    extent=[time_axis[0], time_axis[-1], 0, mel_spec_np.shape[1]]
                )
                plt.colorbar(label="Log Mel Spectrogram (dB)")
                plt.xlabel("Time (s)")
                plt.ylabel("Mel Band")
                plt.title(f"Reconstructed Mel Spectrogram - {speaker_id}/{audio_name}")
                plt.tight_layout()

                # 保存图片并关闭画布（避免内存泄漏）
                plt.savefig(plot_save_path, dpi=150, bbox_inches="tight")
                plt.close()

                LOGGER.info(
                    "[%s] Saved mel spec plot to %s (shape: %s)",
                    speaker_id, plot_save_path, tuple(mel_spec.shape)
                )

            except Exception as e:
                LOGGER.error(
                    "[%s] Failed to plot %s: %s",
                    speaker_id, spec_file.name, str(e), exc_info=True
                )
                continue

    LOGGER.info("plot_mel_spec_from_dir complete. All plots saved to %s", RECONSTRUCTED_PLOT_DIR)


DEFAULT_MIN_UTT_PER_SPK = 10
DEFAULT_EMBED_EXTRACT_UTT_PER_SPK = 20
DEFAULT_SEED = 42


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
            tensor = file_path
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


def generate_speaker_embedding():
    min_utt_per_spk = 20
    spkenc_model = get_model("spkenc")
    from config.paths import CHECKPOINTS_DIR
    spkenc_model.load_state_dict(torch.load(CHECKPOINTS_DIR / "spkenc.pth")["model_state_dict"], wights_only=True)

    from config.paths import RECONSTRUCTED_EMBEDDING_DIR
    for speaker_id in Path.iterdir(RECONSTRUCTED_EMBEDDING_DIR):
        # 1. 处理所有音频文件
        processed_audios = []
        for audio_file in Path.iterdir(speaker_id):
            if audio_file.suffix.lower() not in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
                continue

            waveform, sample_rate = load_audio_file(audio_file)

            if sample_rate is None:
                LOGGER.warning("Missing sample rate for %s, defaulting to %d Hz", str(audio_file), TARGET_SAMPLE_RATE)
                sample_rate = TARGET_SAMPLE_RATE

            if sample_rate < TARGET_SAMPLE_RATE:
                LOGGER.error(
                    "Skipping %s because sample rate %d Hz is below %d Hz",
                    str(audio_file),
                    sample_rate,
                    TARGET_SAMPLE_RATE,
                )
                continue

            waveform = waveform.float()
            waveform = resample_audio(waveform, sample_rate, TARGET_SAMPLE_RATE)
            waveform = waveform.cpu()
            segments = slice_waveform_into_segments(waveform)
            for segment in segments:
                mel_feature = waveform_to_log_mel(segment, sample_rate=TARGET_SAMPLE_RATE)
                processed_audios.append(mel_feature)

        # 2. 随机抽取min_utt_per_spk个片段作为训练集
        if len(processed_audios) < min_utt_per_spk:
            selected_specs = processed_audios
        else:
            selected_specs = random.sample(processed_audios, k=min_utt_per_spk)

        embeddings = _extract_embeddings_for_speaker(
            model=spkenc_model,
            speaker_files=selected_specs,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            batch_size=1,
        )

        speaker_embedding = torch.stack(embeddings, dim=0).mean(dim=0)
        speaker_embedding = F.normalize(speaker_embedding, p=2, dim=0)
        speaker_embedding = speaker_embedding.to(dtype=torch.float32).cpu()
        embedding_path = speaker_id / _EMBEDDED_VECTOR_FILENAME
        torch.save(speaker_embedding, embedding_path)


def main():
    # 初始化路径
    from config.paths import build_dir_path
    build_dir_path(Path("D:\\projects\\python\\AudioReconstruct"))

    # 初始化模型和生成器
    from config.paths import CHECKPOINTS_DIR
    model = get_model("voice_expand_gan")
    model_state_dict = torch.load(CHECKPOINTS_DIR / "voice_expand_gan.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(model_state_dict["model_state_dict"])
    generator = model.generator
    generator.eval()
    with torch.no_grad():
        reconstruct_audio_from_folder(generator, blend_mode="l2_weighted")


if __name__ == "__main__":
    from config.paths import build_dir_path, RECONSTRUCTED_EMBEDDING_DIR, CHECKPOINTS_DIR
    build_dir_path(Path("D:\\projects\\python\\AudioReconstruct"))
    # main()
    plot_mel_spec_from_dir()
    # generate_speaker_embedding()
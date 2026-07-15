import numpy as np
from pydub import AudioSegment
from pathlib import Path

# 与合并端完全一致的标记参数，必须对齐
MARKER_FREQ = 7500
MARKER_DURATION_MS = 50
MARKER_WORD_GAP_MS = 200
TARGET_SAMPLE_RATE = 16000
# 检测阈值（相对最大能量的比例，环境噪音大就调低）
DETECT_THRESHOLD_RATIO = 0.3
# 标记最小间隔（过滤回声误检，单位ms）
MIN_MARKER_GAP_MS = 300


def detect_marker_positions(audio: AudioSegment, target_freq: int = MARKER_FREQ) -> list[int]:
    """检测音频中所有标记的起始时间点，返回毫秒级位置列表"""
    # 统一为单声道16kHz
    audio = audio.set_channels(1).set_frame_rate(TARGET_SAMPLE_RATE)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32767.0
    sr = TARGET_SAMPLE_RATE

    # 滑动窗口参数：窗口=标记时长，步长=10ms
    win_len = int(sr * MARKER_DURATION_MS / 1000)
    hop_len = int(sr * 0.01)
    n_frames = (len(samples) - win_len) // hop_len + 1

    # 预计算目标频率的FFT索引
    freq_bin = int(target_freq * win_len / sr)
    energies = []

    # 滑动窗口计算目标频率能量
    for i in range(n_frames):
        start = i * hop_len
        frame = samples[start:start + win_len] * np.hanning(win_len)  # 加窗减少频谱泄漏
        fft = np.abs(np.fft.rfft(frame))
        target_energy = fft[freq_bin]
        energies.append(target_energy)

    # 自适应阈值
    max_energy = max(energies) if energies else 0
    threshold = max_energy * DETECT_THRESHOLD_RATIO

    # 提取峰值位置，过滤过近的重复检测
    positions = []
    last_pos = -MIN_MARKER_GAP_MS
    for i, e in enumerate(energies):
        if e > threshold:
            pos_ms = i * 10  # 步长10ms
            if pos_ms - last_pos > MIN_MARKER_GAP_MS:
                positions.append(pos_ms)
                last_pos = pos_ms
    return positions


def split_audio_by_markers(
        recorded_audio_path: str | Path,
        output_dir: str | Path,
        word_order_file: str | Path | None = None
):
    """根据微标记自动切分录制的音频，输出纯净单词音频"""
    recorded_audio_path = Path(recorded_audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载录制音频
    audio = AudioSegment.from_file(str(recorded_audio_path))
    print(f"加载录制音频，总时长 {len(audio) / 1000:.2f}s")

    # 2. 检测所有标记位置
    all_markers = detect_marker_positions(audio)
    print(f"检测到标记 {len(all_markers)} 个")

    # 3. 配对起始/结束标记（两两配对，第1个起始、第2个结束、第3个起始...）
    if len(all_markers) % 2 != 0:
        print(f"警告：标记数量为奇数，最后一个未配对标记将被忽略")

    word_segments = []
    for i in range(0, len(all_markers) - 1, 2):
        start_marker_pos = all_markers[i]
        end_marker_pos = all_markers[i + 1]
        # 有效音频范围：起始标记后0.2s → 结束标记前0.2s，完全避开标记
        word_start = start_marker_pos + MARKER_DURATION_MS + MARKER_WORD_GAP_MS
        word_end = end_marker_pos - MARKER_WORD_GAP_MS
        if word_end <= word_start:
            print(f"跳过第 {i // 2 + 1} 个无效片段：时长为负")
            continue
        word_segments.append(audio[word_start:word_end])

    # 4. 加载单词顺序（可选，自动对应命名）
    word_names = []
    if word_order_file and Path(word_order_file).exists():
        with open(word_order_file, "r", encoding="utf-8") as f:
            word_names = [line.strip() for line in f.readlines() if line.strip()]

    # 5. 导出切分结果
    for idx, seg in enumerate(word_segments):
        if word_names and idx < len(word_names):
            filename = f"{idx + 1:03d}_{word_names[idx]}.wav"
        else:
            filename = f"word_{idx + 1:03d}.wav"
        save_path = output_dir / filename
        seg.export(str(save_path), format="wav")

    print(f"切分完成，共导出 {len(word_segments)} 个单词音频，保存至 {output_dir}")


if __name__ == "__main__":
    split_audio_by_markers(
        recorded_audio_path="录制后的音频路径.wav",
        output_dir="./split_words",
        word_order_file="./word_order.txt"  # 合并时生成的顺序表，可自动命名
    )
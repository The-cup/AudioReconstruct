from pydub import AudioSegment
from pathlib import Path
from typing import Dict, List
import numpy as np

# ========== 可修改配置 ==========
# 单词音频所在文件夹（一级目录为单词名，二级为该单词的多个发音）
INPUT_DIR = Path(r"D:\projects\python\AudioReconstruct\data\raw\WordSelected")
# 拼接后音频输出路径
OUTPUT_PATH = Path(r"D:\projects\python\AudioReconstruct\data\raw\merged_words.wav")
# 统一采样率（和你项目现有参数保持一致）
TARGET_SAMPLE_RATE = 16000
# 音频最前端留空时长（10秒）
FRONT_SILENCE_MS = 10 * 1000

# 原有固定槽配置（可选启用）
SLOT_DURATION_MS = 5 * 1000
USE_FIXED_SLOT = False  # 关闭则用紧凑间隔模式，大幅缩短总时长
WORD_INTERVAL_MS = 1000  # 紧凑模式下，相邻单词单元的间隔（1s，匹配你说的有效音频间隔）

# ========== 微标记专属配置 ==========
MARKER_FREQ = 7500  # 标记频率（Hz），高频不刺耳，与人声区分度高
MARKER_DURATION_MS = 50  # 单条标记时长（ms），短且易检测
MARKER_VOLUME_DB = -20  # 标记音量（dBFS），调低避免刺耳
MARKER_WORD_GAP_MS = 200  # 标记与单词前后的间隔（0.2s，完全匹配你的需求）

_SUPPORTED_AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"})


def generate_tone_marker(
        freq: int = MARKER_FREQ,
        duration_ms: int = MARKER_DURATION_MS,
        volume_db: int = MARKER_VOLUME_DB,
        sample_rate: int = TARGET_SAMPLE_RATE
) -> AudioSegment:
    """生成单频正弦波微标记：人耳感知轻微，频谱特征稳定易检测"""
    # 生成时间轴与正弦波
    sample_count = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, sample_count, endpoint=False)
    amplitude = 10 ** (volume_db / 20)  # 按分贝换算幅值
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    # 转为16位PCM格式的音频段
    wave_int16 = np.int16(wave * 32767)
    return AudioSegment(
        wave_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )


def load_word_audio_dataset(
        root_dir: str | Path,
        max_per_word: int | None = None,
) -> Dict[str, List[Path]]:
    """
    读取分层单词音频数据集
    目录结构：root_dir/{单词名}/xxx.wav（每个单词文件夹下包含多条发音）
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"单词音频根目录不存在: {root_dir}")

    word_dataset: Dict[str, List[Path]] = {}
    word_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])

    for word_dir in word_dirs:
        word = word_dir.name
        audio_files = sorted([
            f for f in word_dir.iterdir()
            if f.is_file() and f.suffix.lower() in _SUPPORTED_AUDIO_SUFFIXES
        ])
        if not audio_files:
            continue
        if max_per_word is not None:
            audio_files = audio_files[:max_per_word]
        word_dataset[word] = audio_files

    total_count = sum(len(v) for v in word_dataset.values())
    print(f"读取完成：共{len(word_dataset)}个单词，总发音数{total_count}条")
    return word_dataset


def merge_words_with_time_slots():
    # 1. 读取分层单词数据集，展平为有序列表（同单词发音连续排列）
    word_dataset = load_word_audio_dataset(INPUT_DIR)
    word_order = []  # 记录顺序，方便切分后对应标注
    word_path_list = []
    for word, paths in word_dataset.items():
        for p in paths:
            word_order.append(word)
            word_path_list.append(p)

    if not word_path_list:
        raise ValueError(f"输入目录 {INPUT_DIR} 下未找到有效音频文件")
    print(f"共待拼接 {len(word_path_list)} 条单词发音")

    # 2. 预生成标记音频
    start_marker = generate_tone_marker()
    end_marker = generate_tone_marker()

    # 3. 初始化最终音频 + 累计时间变量（精准记录当前拼接位置）
    final_audio = AudioSegment.silent(duration=FRONT_SILENCE_MS, frame_rate=TARGET_SAMPLE_RATE)
    current_time_ms = FRONT_SILENCE_MS  # 新增：实时累计的当前时间，单位ms

    # 4. 逐个构造单词单元
    for idx, word_path in enumerate(word_path_list):
        # 加载并标准化单词音频
        word_audio = AudioSegment.from_file(str(word_path))
        word_audio = word_audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(1)

        # 超长截断保护
        max_word_len = SLOT_DURATION_MS - 2 * MARKER_DURATION_MS - 2 * MARKER_WORD_GAP_MS
        if len(word_audio) > max_word_len:
            print(f"警告：{word_path.name} 时长 {len(word_audio) / 1000:.2f}s 超限，截断到 {max_word_len / 1000:.2f}s")
            word_audio = word_audio[:max_word_len]

        # 构造单单词单元：起始标记 + 0.2s静音 + 单词 + 0.2s静音 + 结束标记
        unit = start_marker
        unit += AudioSegment.silent(duration=MARKER_WORD_GAP_MS, frame_rate=TARGET_SAMPLE_RATE)
        unit += word_audio
        unit += AudioSegment.silent(duration=MARKER_WORD_GAP_MS, frame_rate=TARGET_SAMPLE_RATE)
        unit += end_marker

        # 补全到固定槽长 / 追加间隔静音
        if USE_FIXED_SLOT:
            tail_silence = SLOT_DURATION_MS - len(unit)
            if tail_silence > 0:
                unit += AudioSegment.silent(duration=tail_silence, frame_rate=TARGET_SAMPLE_RATE)
        else:
            # 紧凑模式：单词单元后加1s间隔（最后一个单词不加）
            if idx != len(word_path_list) - 1:
                unit += AudioSegment.silent(duration=WORD_INTERVAL_MS, frame_rate=TARGET_SAMPLE_RATE)

        # ========== 修正后的打印逻辑：直接用累计时间，完全准确 ==========
        slot_start_sec = current_time_ms / 1000
        print(
            f"已处理第 {idx + 1}/{len(word_path_list)} 个：{word_order[idx]} - {word_path.name}，起始位置 {slot_start_sec:.2f}s")

        # 追加音频 + 更新累计时间
        final_audio += unit
        current_time_ms += len(unit)

    # 5. 导出结果 + 保存单词顺序列表（方便切分后自动命名）
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_audio.export(str(OUTPUT_PATH), format="wav")

    # 同步保存单词顺序，切分后可直接对应命名
    order_save_path = OUTPUT_PATH.with_name("word_order.txt")
    with open(order_save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(word_order))

    print(f"\n拼接完成，总时长 {len(final_audio) / 1000:.2f}s")
    print(f"合并音频输出：{OUTPUT_PATH}")
    print(f"单词顺序表输出：{order_save_path}")


if __name__ == "__main__":
    merge_words_with_time_slots()
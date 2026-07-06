import torch
from osstorchconnector import OssMapDataset
import io
import logging
from typing import Iterable, Tuple
from pathlib import Path

ENDPOINT = "http://oss-cn-shanghai-internal.aliyuncs.com"
REGION = "cn-shanghai"
MNT_PATH = Path("/root/code/src")
MNT_OSS_PATH = MNT_PATH / "oss"
CONFIG_PATH = str(MNT_OSS_PATH / "config.json")
CRED_PATH = str(MNT_OSS_PATH / "credentials.json")
# OSS_BASE_URI = "oss://oss-pai-t6plj4go0d9wk1etgd-cn-shanghai/draft-cqwks9jp2ss6evicim/AudioReconstruct/"
OSS_BASE_URI = ""

OSS_MNT_PATH = Path("/mnt/data/AudioReconstruct/")
# 使用oss connector读取所有processed数据，读取所有的processed对应的嵌入向量
MANIFEST_SAMPLE_TRAIN = str(OSS_MNT_PATH / "data" / "manifest-sample-train.txt")
MANIFEST_EMBEDDING_TRAIN = str(OSS_MNT_PATH / "data" / "manifest-embedding-train.txt")
MANIFEST_LOW_FREQ_TRAIN = str(OSS_MNT_PATH / "data" / "manifest-low-freq-train.txt")
MANIFEST_SAMPLE_TEST = str(OSS_MNT_PATH / "data" / "manifest-sample-test.txt")
MANIFEST_EMBEDDING_TEST = str(OSS_MNT_PATH / "data" / "manifest-embedding-test.txt")
MANIFEST_LOW_FREQ_TEST = str(OSS_MNT_PATH / "data" / "manifest-low-freq-test.txt")
MANIFEST_SAMPLE_VAL = str(OSS_MNT_PATH / "data" / "manifest-sample-val.txt")
MANIFEST_EMBEDDING_VAL = str(OSS_MNT_PATH / "data" / "manifest-embedding-val.txt")
MANIFEST_LOW_FREQ_VAL = str(OSS_MNT_PATH / "data" / "manifest-low-freq-val.txt")


def transform(object):
    try:
        raw_bytes = object.read()
        tensor = torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=True)
    except Exception as e:
        raise e
    return tensor

def get_oss_dataset(manifest_file_path):
    return OssMapDataset.from_manifest_file(
        manifest_file_path=manifest_file_path,
        manifest_parser=imagenet_manifest_parser,
        transform=transform,
        oss_base_uri=OSS_BASE_URI,
        endpoint=ENDPOINT,
        cred_path=CRED_PATH,
        config_path=CONFIG_PATH,
        region=REGION
    )

import io
import logging
from typing import Iterable, Tuple

def clean_ascii_edge_chars(s: str) -> str:
    """
    清除字符串首尾 ASCII≥128 的字符，中间保留；
    自动统一去掉 \r，消除 \n / \r\n 差异
    """
    # 1. 统一剔除 \r，抹平换行符差异
    s = s.replace("\r", "")
    # 2. 去掉首尾空白（空格、制表、换行）
    s_stripped = s.strip()
    if not s_stripped:
        return ""

    # 从头部截取：只保留连续 ASCII < 128 的前缀
    head_idx = 0
    while head_idx < len(s_stripped) and ord(s_stripped[head_idx]) < 128:
        head_idx += 1
    head_clean = s_stripped[:head_idx]

    # 从尾部反向截取：去掉末尾≥128的字符
    tail_idx = len(head_clean)
    while tail_idx > 0 and ord(head_clean[tail_idx - 1]) >= 128:
        tail_idx -= 1
    final = head_clean[:tail_idx]
    return final


def imagenet_manifest_parser(reader: io.IOBase) -> Iterable[Tuple[str, str]]:
    # 读取全部文本，统一移除 \r，换行统一按 \n 分割
    raw_text = reader.read().decode("utf-8")
    raw_text = raw_text.replace("\r", "")  # 消除 \r\n 残留
    lines = raw_text.split("\n")

    for line_num, raw_line in enumerate(lines):
        try:
            # 清洗首尾非ASCII(≥128)字符
            clean_line = clean_ascii_edge_chars(raw_line)
            if not clean_line:
                continue  # 空行跳过

            # 这里根据你的真实文件格式修改分割逻辑
            # 示例：假设每行用空格/制表分割成 key, label，按需替换分隔符
            items = clean_line.split(maxsplit=1)
            if len(items) == 1:
                key = items[0]
                label = ""
                yield key, label
            elif len(items) == 2:
                key, label = items
                yield key, label
            else:
                raise ValueError(f"Line split error, expecting 1 or 2 segments, got {len(items)}")

        except ValueError as e:
            logging.error(f"Line {line_num} parse error: {e}, raw content: {repr(raw_line)}")

oss_train_dataset = {
    "sample": get_oss_dataset(MANIFEST_SAMPLE_TRAIN),
    "low_freq": get_oss_dataset(MANIFEST_LOW_FREQ_TRAIN),
    "embedding": get_oss_dataset(MANIFEST_EMBEDDING_TRAIN)
}
oss_test_dataset = {
    "sample": get_oss_dataset(MANIFEST_SAMPLE_TEST),
    "low_freq": get_oss_dataset(MANIFEST_LOW_FREQ_TEST),
    "embedding": get_oss_dataset(MANIFEST_EMBEDDING_TEST)
}
oss_val_dataset = {
    "sample": get_oss_dataset(MANIFEST_SAMPLE_VAL),
    "low_freq": get_oss_dataset(MANIFEST_LOW_FREQ_VAL),
    "embedding": get_oss_dataset(MANIFEST_EMBEDDING_VAL)
}
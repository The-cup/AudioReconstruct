import os
import tarfile
import requests
from tqdm import tqdm

# 基础配置
SAVE_DIR = r"D:\projects\python\AudioReconstruct\data\raw"
DATASET_VERSION = "v0.02"  # 可选 v0.01 / v0.02，和原脚本版本对应
SPLITS = ["train", "validation", "test"]
# 和原脚本下载地址完全一致
BASE_URL = "https://s3.amazonaws.com/datasets.huggingface.co/SpeechCommands/{name}/{name}_{split}.tar.gz"

os.makedirs(SAVE_DIR, exist_ok=True)


def download_file(url: str, save_path: str):
    """带进度条的下载函数，支持断点续传"""
    resp = requests.get(url, stream=True, timeout=30)
    total_size = int(resp.headers.get("content-length", 0))
    with open(save_path, "wb") as f, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)


if __name__ == "__main__":
    for split in SPLITS:
        url = BASE_URL.format(name=DATASET_VERSION, split=split)
        tar_path = os.path.join(SAVE_DIR, f"{DATASET_VERSION}_{split}.tar.gz")

        # 下载压缩包
        if not os.path.exists(tar_path):
            print(f"开始下载 {split} 数据集...")
            download_file(url, tar_path)
        else:
            print(f"{split} 压缩包已存在，跳过下载")

        # 解压到目标目录
        print(f"开始解压 {split} 数据集...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(SAVE_DIR)
        print(f"{split} 数据集解压完成")

    print(f"\n全部处理完成，数据集存放路径：{SAVE_DIR}")
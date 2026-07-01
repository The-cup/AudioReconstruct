from osstorchconnector import OssMapDataset, imagenet_manifest_parser
import io
import logging
from typing import Iterable, Tuple

ENDPOINT = "http://oss-cn-shanghai-internal.aliyuncs.com"
REGION = "cn-shanghai"
CONFIG_PATH = "/root/code/src/config/oss/config.json"
CRED_PATH = "/root/code/src/config/oss/credentials.json"
OSS_BASE_URI = "oss://oss-pai-t6plj4go0d9wk1etgd-cn-shanghai/draft-cqwks9jp2ss6evicim/AudioReconstruct/"

# 使用oss connector读取所有processed数据，读取所有的processed对应的嵌入向量

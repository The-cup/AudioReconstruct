# AudioReconstruct

语音重建项目骨架，面向低频变形人声音频到可听声音的重建任务。

## 目录约定

- `data/raw/`: 原始数据
- `data/interim/`: 中间处理数据
- `data/processed/`: 处理完成的数据
- `data/dataset/`: 训练用数据集文件
- `data/external/`: 外部数据
- `artifacts/checkpoints/`: 模型权重与检查点
- `artifacts/logs/`: 训练和推理日志
- `artifacts/reports/`: 评估结果与报表
- `src/audio_reconstruct/data/preprocess/`: 数据处理工具与主流程
- `src/audio_reconstruct/datasets/`: 数据集定义
- `src/audio_reconstruct/models/`: 自定义模型结构
- `src/audio_reconstruct/ml/`: 训练、验证、测试、评估、推理
- `src/audio_reconstruct/tasks/`: 实际任务编排

## 环境

优先使用 Conda:

```bash
conda env create -f environment.yml
conda activate audio-reconstruct
```

如果只想补齐 pip 依赖，可使用:

```bash
pip install -r requirements.txt
```

## 入口

主入口为 `main.py`，后续可在各任务模块完成实现后串起完整流程。


# HAPA 细菌分割

[English README](README_EN.md)

本仓库公开 HAPA（Hydrophobic Agent Permeability Assay）研究中细菌分割部分的 PyTorch U-Net 实现，包括图像预处理、模型训练、预测代码、预训练权重，以及部分训练图像和对应的人工标注。当前公开的训练数据为 20 对 TIFF 图像与标注；后续可继续补充数据。

## 仓库内容

| 路径 | 用途 |
| --- | --- |
| [`training_data/train/`](training_data/train/) | 当前公开的训练图像和人工标注，按图像名称分文件夹存放。 |
| `model.py` | 与预训练权重兼容的单通道 U-Net。 |
| `image_io.py`、`dataset.py` | TIFF 读取、图像与标注配对、裁剪、图像块提取和 CLAHE 预处理。 |
| `grouping.py`、`make_group_split.py` | 按图像序列划分训练集和测试集，并检查两组是否重叠。 |
| `loss.py`、`metrics.py` | 损失函数，以及 ACC、F1、MOC、精确率和召回率计算。 |
| `train.py`、`plot_curves.py` | 训练模型并生成逐轮指标记录与曲线。 |
| `predict.py` | 使用预训练模型预测细菌掩膜。 |

预训练权重 [`best_model.pth`](https://github.com/xiaokongkong-ld/HAPA-bacterial-segmentation/releases/download/v1/best_model.pth) 在 [v1 Release](https://github.com/xiaokongkong-ld/HAPA-bacterial-segmentation/releases/tag/v1) 中，下载后放在仓库根目录即可使用。Release 同时提供包含代码、权重和当前训练数据的完整压缩包。

## 安装

建议使用 Python 3.10 或更新版本，并安装适合本机 CPU 或 CUDA 的 PyTorch：

```bash
pip install -r requirements.txt
```

## 数据结构

当前训练图像及人工标注已直接放在仓库的 `training_data/train/` 中，无需解压。每个文件夹包含一张显微图像和一张对应的 `_col.tif` 标注，标注中非零像素表示细菌。例如：

```text
training_data/
└── train/
    └── Image_8_0_e/
        ├── Image_8_0_e.tif
        └── Image_8_0_e_col.tif
```

训练并绘制训练集、测试集两条曲线时，`--data-dir` 指向同时包含 `train/` 和 `test/` 的目录。请按图像序列分组，避免同一序列同时进入两组；仓库中的 `make_group_split.py` 提供了划分和检查方法。公开仓库目前提供训练集部分数据，测试数据可在本地按相同文件夹结构准备。

## 模型训练与曲线

```bash
python train.py --data-dir /path/to/data --output-dir runs/hapa --epochs 200
```

默认参数包括 batch size 6、Adam 学习率 `1e-5`、`256 × 256` 图像块，以及 Dice、IoU 和 BCE 组合损失。训练过程中每轮更新 `training_metrics.csv`、五项指标的训练/测试 TXT 和 JPG 曲线、五联图 `segmentation_training_curves.png`，并保存模型检查点。

如需训练 2000 轮：

```bash
python -u train.py --data-dir /path/to/data --output-dir runs/hapa_2000 --epochs 2000 --checkpoint-every 25
```

中断后在相同命令末尾加 `--resume`。已有 `training_metrics.csv` 时，可单独重绘曲线：

```bash
python plot_curves.py runs/hapa/training_metrics.csv
```

## 使用预训练模型预测

先从 [v1 Release](https://github.com/xiaokongkong-ld/HAPA-bacterial-segmentation/releases/tag/v1) 下载 `best_model.pth`，然后运行：

```bash
python predict.py --input /path/to/image.tif --output /path/to/mask.tif --weights best_model.pth
```

输出为 8 位 TIFF 掩膜：背景为 0，细菌为 255。预训练权重的 SHA-256 为 `c6a2068c53eb34a2b6a95a462cdb0954c1cea21207b9eea3a2120d6feaf93ab5`。

本仓库对应论文中的细菌分割部分。训练脚本生成的是运行时的新曲线；历史论文图的逐轮原始数值未包含在当前版本中。


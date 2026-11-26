# HAPA 细菌分割

[English](README.md)

本仓库提供 HAPA 研究中细菌分割使用的 PyTorch U-Net 代码，包括图像预处理、训练、预测、第 550 轮模型权重，以及部分显微图像和配对标注。目前公开的是完整训练数据中的一部分，共 36 对图像与掩膜；其他训练数据尚未公开，后续可继续增加。其中 31 份为已有标注、5 份为经复核认可的模型预测标注；来源及文件校验值见 `training_data/manifest.csv`。

主要脚本：`model_v2.py` 为模型，`train_final.py` 为训练，`predict_v2.py` 为预测，`checkpoint_epoch0550.pth` 为配套权重。

安装适合本机的 PyTorch 后执行 `pip install -r requirements.txt`。预测示例：

```bash
python predict_v2.py --input image.tif --output mask.tif --weights checkpoint_epoch0550.pth
```

继续训练时，将配对的 TIFF 图像与 `_col.tif` 掩膜放入 `DATA_DIR/train/`，并自行准备独立的 `DATA_DIR/test/`：

```bash
python train_final.py --data-dir DATA_DIR --initial-weights checkpoint_epoch0550.pth --epoch-offset 550 --output-dir runs/continue --epochs 50 --test-each-epoch
```

训练过程逐轮保存指标、曲线和模型。这 36 对只是部分公开训练数据，仓库未包含当前测试图像。随仓库提供的第 550 轮权重早于本次数据更新。

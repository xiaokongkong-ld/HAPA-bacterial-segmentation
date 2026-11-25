# HAPA 细菌分割

[English](README.md)

本仓库提供 HAPA 研究中细菌分割使用的 PyTorch U-Net 代码，包括图像预处理、训练、预测、第 550 轮模型权重，以及部分显微图像和人工标注。公开训练数据目前为 20 对图像与掩膜，后续可继续增加。

主要脚本：`model_v2.py` 为模型，`train_final.py` 为训练，`predict_v2.py` 为预测，`checkpoint_epoch0550.pth` 为配套权重。

安装适合本机的 PyTorch 后执行 `pip install -r requirements.txt`。预测示例：

```bash
python predict_v2.py --input image.tif --output mask.tif --weights checkpoint_epoch0550.pth
```

继续训练时，将配对的 TIFF 图像与 `_col.tif` 掩膜放入 `DATA_DIR/train/`，并自行准备独立的 `DATA_DIR/test/`：

```bash
python train_final.py --data-dir DATA_DIR --initial-weights checkpoint_epoch0550.pth --epoch-offset 550 --output-dir runs/continue --epochs 50 --test-each-epoch
```

训练过程逐轮保存指标、曲线和模型。仓库仅公开部分训练数据，未包含测试图像。

# HAPA 细菌分割

[English](README.md)

本仓库提供 HAPA 研究中细菌分割使用的 PyTorch U-Net 代码，包括图像预处理、训练、预测、训练好的模型权重，以及部分训练图像和配对标注。

主要脚本：`model_v2.py` 为模型，`train_final.py` 为训练，`predict_v2.py` 为预测，`trained_model.pth` 为训练好的模型权重。

安装适合本机的 PyTorch 后执行 `pip install -r requirements.txt`。预测示例：

```bash
python predict_v2.py --input image.tif --output mask.tif --weights trained_model.pth
```

继续训练时，将配对的 TIFF 图像与 `_col.tif` 掩膜放入 `DATA_DIR/train/`，并自行准备独立的 `DATA_DIR/test/`：

```bash
python train_final.py --data-dir DATA_DIR --initial-weights trained_model.pth --output-dir runs/continue --epochs 50 --test-each-epoch
```

训练过程逐轮保存指标、曲线和模型。

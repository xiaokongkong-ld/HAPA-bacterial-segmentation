# HAPA bacterial segmentation

[中文版说明](README_ZH.md)

This repository provides the PyTorch U-Net implementation for bacterial segmentation in the Hydrophobic Agent Permeability Assay (HAPA) study. It includes image preprocessing, model training and inference code, pretrained weights, and a portion of the training images with their manual annotations. The current public training set contains 20 microscopy TIFFs and 20 corresponding masks; additional data may be added later.

## Repository contents

| Path | Description |
| --- | --- |
| [`training_data/train/`](training_data/train/) | Public training images and manual masks, organized in paired folders. |
| `model.py` | Single-channel U-Net compatible with the pretrained weights. |
| `image_io.py`, `dataset.py` | TIFF loading, image–mask pairing, cropping, patch extraction, and CLAHE preprocessing. |
| `grouping.py`, `make_group_split.py` | Group-disjoint train/test splitting and overlap checks. |
| `loss.py`, `metrics.py` | Loss functions and ACC, F1, MOC, precision, and recall metrics. |
| `train.py`, `plot_curves.py` | Model training, epoch-by-epoch metric logs, and training curves. |
| `predict.py` | Bacterial mask prediction with the pretrained model. |

The pretrained [`best_model.pth`](https://github.com/xiaokongkong-ld/HAPA-bacterial-segmentation/releases/download/v1/best_model.pth) is available in the [v1 release](https://github.com/xiaokongkong-ld/HAPA-bacterial-segmentation/releases/tag/v1). Download it to the repository root before running inference. The release also includes a complete archive of the code, weights, and current training data.

## Installation

Use Python 3.10 or newer and install a PyTorch build suitable for your CPU or CUDA environment:

```bash
pip install -r requirements.txt
```

## Data layout

The current training images and manual masks are directly available in [`training_data/train/`](training_data/train/); no extraction is needed. Each folder contains one microscopy image and its matching `_col.tif` mask. Nonzero mask pixels denote bacteria. For example:

```text
training_data/
└── train/
    └── Image_8_0_e/
        ├── Image_8_0_e.tif
        └── Image_8_0_e_col.tif
```

To train the model and plot both training and test curves, point `--data-dir` to a directory containing both `train/` and `test/`. Keep image series disjoint across the two splits; `make_group_split.py` provides a splitting and overlap-checking example. This repository currently provides a portion of the training data. Test images can be prepared locally using the same folder structure.

## Training and metric curves

```bash
python train.py --data-dir /path/to/data --output-dir runs/hapa --epochs 200
```

Default settings include batch size 6, Adam learning rate `1e-5`, `256 × 256` patches, and a combined Dice, IoU, and BCE loss. After each epoch, the script updates `training_metrics.csv`, train/test TXT and JPG series for five metrics, the five-panel `segmentation_training_curves.png`, and model checkpoints.

For 2,000 epochs:

```bash
python -u train.py --data-dir /path/to/data --output-dir runs/hapa_2000 --epochs 2000 --checkpoint-every 25
```

Append `--resume` to the same command to continue an interrupted run. To redraw curves from an existing metrics CSV:

```bash
python plot_curves.py runs/hapa/training_metrics.csv
```

## Inference with the pretrained model

Download `best_model.pth` from the [v1 release](https://github.com/xiaokongkong-ld/HAPA-bacterial-segmentation/releases/tag/v1), then run:

```bash
python predict.py --input /path/to/image.tif --output /path/to/mask.tif --weights best_model.pth
```

The output is an 8-bit TIFF mask: 0 for background and 255 for bacteria. The pretrained checkpoint has SHA-256 `c6a2068c53eb34a2b6a95a462cdb0954c1cea21207b9eea3a2120d6feaf93ab5`.

This repository covers the bacterial segmentation component of the study. Training generates new metric curves; the epoch-by-epoch values underlying the historical manuscript figure are not included in this release.


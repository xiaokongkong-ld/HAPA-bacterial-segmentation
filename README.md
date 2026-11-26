# HAPA bacterial segmentation

[中文说明](README_ZH.md)

PyTorch U-Net code for bacterial segmentation in the HAPA study. This repository includes image preprocessing, training, prediction, a trained model, and a public subset of the microscopy training data with paired masks.

## Contents

- `model_v2.py`: U-Net model
- `dataset.py`, `image_io.py`: paired TIFF loading, preprocessing, and augmentation
- `train_final.py`, `plot_train.py`: training and metric curves
- `predict_v2.py`: segmentation of new images
- `trained_model.pth`: trained U-Net weights
- `training_data/train/`: a public subset of 36 image and mask pairs; mask origins are recorded in `training_data/manifest.csv`

## Use

Install Python 3.10 or newer and a suitable PyTorch build, then run `pip install -r requirements.txt`.

```bash
python predict_v2.py --input image.tif --output mask.tif --weights trained_model.pth
```

To continue training, arrange paired TIFFs in `DATA_DIR/train/` and `DATA_DIR/test/`. Each folder should contain `<name>.tif` and `<name>_col.tif`. Keep images from the same acquisition series in one split.

```bash
python train_final.py --data-dir DATA_DIR --initial-weights trained_model.pth --output-dir runs/continue --epochs 50 --test-each-epoch
```

Training saves metrics, curves, and checkpoints after every epoch.

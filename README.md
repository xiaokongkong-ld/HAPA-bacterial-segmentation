# HAPA bacterial segmentation

[中文说明](README_ZH.md)

PyTorch U-Net code for bacterial segmentation in the HAPA study. This repository includes image preprocessing, training, prediction, a trained model, and a public subset of the microscopy training data with paired masks.

## Contents

- `model_v2.py`: U-Net model
- `dataset.py`, `image_io.py`: paired TIFF loading, preprocessing, and augmentation
- `train_final.py`, `plot_train.py`: training and metric curves
- `predict_v2.py`: segmentation of new images
- `checkpoint_epoch0550.pth`: compatible model weights from training epoch 550
- `training_data/train/`: a public subset of 36 image and mask pairs; mask origins are recorded in `training_data/manifest.csv`

## Use

Install Python 3.10 or newer and a suitable PyTorch build, then run `pip install -r requirements.txt`.

```bash
python predict_v2.py --input image.tif --output mask.tif --weights checkpoint_epoch0550.pth
```

To continue training, arrange paired TIFFs in `DATA_DIR/train/` and `DATA_DIR/test/`. Each folder should contain `<name>.tif` and `<name>_col.tif`. Keep images from the same acquisition series in one split.

```bash
python train_final.py --data-dir DATA_DIR --initial-weights checkpoint_epoch0550.pth --epoch-offset 550 --output-dir runs/continue --epochs 50 --test-each-epoch
```

Training saves metrics, curves, and checkpoints after every epoch. The 36 public pairs are part of a larger training dataset; additional training images and annotations are not included in this repository. The current test images are also not included. The included epoch-550 checkpoint predates this data update.

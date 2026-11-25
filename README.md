# HAPA bacterial segmentation and training curves

This repository contains the PyTorch U-Net code used for the bacterial segmentation component of the Hydrophobic Agent Permeability Assay (HAPA). It includes the trained segmentation checkpoint, the current 20-image training split with manual masks, and a cleaned training pipeline that records accuracy (ACC), F1, MOC, precision, and recall.

Download `best_model.pth` from the [v1 release](https://github.com/xiaokongkong-ld/HAPA-bacterial-segmentation/releases/tag/v1) and extract `training_data.zip` into the repository root before using the examples below. The release also provides a complete source-and-data ZIP.

The training script generates **new** curves from the data supplied to it. The original epoch-by-epoch TXT files behind the published figure were not found in the available project directory or its two RAR archives. This repository does not claim to recreate those exact numeric series.

## Files

| File | Function |
| --- | --- |
| `model.py` | Checkpoint-compatible single-channel U-Net. |
| `image_io.py`, `dataset.py` | Grayscale TIFF loading, manual-mask pairing, centre cropping, 256-pixel patches, and CLAHE. |
| `grouping.py`, `make_group_split.py` | Group-disjoint train/test split and overlap guard. |
| `loss.py`, `metrics.py` | Combined Dice/IoU/BCE loss and pixel metrics. |
| `train.py` | Training and metric logging. |
| `plot_curves.py` | Five individual JPG plots, ten TXT series, and one five-panel PNG. |
| `predict.py` | Run the supplied checkpoint on an unlabeled TIFF. |
| `best_model.pth` (release asset) | Original trained segmentation weights (30,531,377 bytes). |
| `training_data.zip` | Current 20 training images and corresponding manual masks. |

## Installation

Use Python 3.10 or newer and a PyTorch build suitable for your CPU or CUDA installation:

```bash
pip install -r requirements.txt
```

The code was smoke tested with Python and a CUDA-enabled PyTorch environment on Windows. The exact versions used in the historical training run were not recorded.

## Training data layout

Extract `training_data.zip` to obtain the current training images and manual masks in `training_data/train/`. Supply a validation/test split in the following structure to run training and plot both curves:

```text
DATA_DIR/
├── train/
│   └── image_001/
│       ├── image_001.tif
│       └── image_001_col.tif
└── test/
    └── image_002/
        ├── image_002.tif
        └── image_002_col.tif
```

Each image must be a single-channel 8-bit or 16-bit TIFF. The matching mask must be a single-channel TIFF with bacteria represented by nonzero pixels. The loader checks image/mask sizes and rejects images smaller than the requested crop.

This first release includes 20 training image/mask pairs. Additional microscopy images can be added in later releases.

### Reorganize the locally available HAPA data

The original split has image series `8`, `12`, `18`, and `1451` in both train and test. To create a new split while leaving the source untouched:

```bash
python make_group_split.py --source /path/to/original_data --destination /path/to/private_group_split --test-groups 14 1427 3528
```

This assigns all images from three complete series (four image/mask pairs) to test and the remaining 20 pairs to train. It copies only the paired image and `_col.tif` mask files and writes a private `split_manifest.csv` with file hashes. Keep the destination and manifest outside a public repository. The training script refuses to run if a group identifier appears in both splits. Grouping is based on the series number in `Image_<series>_<frame>_e`; other folder names are treated as their own groups.

## Generate the five curves

```bash
python train.py --data-dir /path/to/private_group_split --output-dir runs/hapa --epochs 200
```

Defaults reflect the original training settings where possible: batch size 6, Adam learning rate 1e-5, the same U-Net architecture, 256 × 256 patches, CLAHE with clip limit 2.0 and 8 × 8 tiles, and equal 0.33 weights on Dice, IoU and binary cross-entropy losses. Training images use a 512 × 512 centre crop and patch stride 40. The `test/` images use a 512 × 512 centre crop and stride 256.

The command updates `training_metrics.csv`, ten `*_train.txt` / `*_test.txt` files, five `*_plot.jpg` files, `segmentation_training_curves.png`, and `latest_epoch.txt` after every epoch. It also saves `trained_final.pth`, `trained_best_validation.pth`, and `checkpoint_latest.pth`. The TXT files contain one metric value per completed epoch. The best-validation weights are selected using the split named `test/` and must not be treated as an unbiased final test result.

For a long run, specify the total target epoch count. A full 2,000-epoch run saves a resumable model and optimizer checkpoint every 25 epochs by default:

```bash
python -u train.py --data-dir /path/to/private_group_split --output-dir runs/hapa_2000 --epochs 2000 --checkpoint-every 25
```

If interrupted, run the same command with `--resume`. The script restores the latest checkpoint and rewinds any CSV rows written after that checkpoint. It does not stop early: `--epochs 2000` runs through epoch 2000 unless interrupted.

The code was also verified with a separate 30-epoch run on 20 training and four test images. The run outputs are not included in this first release.

To redraw the figures from an existing `training_metrics.csv`:

```bash
python plot_curves.py runs/hapa/training_metrics.csv
```

The `test/` split is evaluated after **every epoch** to draw its curve. If model choices are made from these values, this is a validation split; a separate held-out test set is needed for an independent final estimate.

The group check prevents a filename series from crossing splits. Available metadata still does not establish independence at the specimen or experiment level. Confirm those identifiers with the imaging team before interpreting test metrics as a fully independent estimate.

## Use the provided checkpoint

```bash
python predict.py --input /path/to/image.tif --output /path/to/mask.tif --weights best_model.pth
```

The output mask is an 8-bit TIFF (0 = background, 255 = bacterium). The checkpoint SHA-256 is `c6a2068c53eb34a2b6a95a462cdb0954c1cea21207b9eea3a2120d6feaf93ab5`. The checkpoint is compatible with `model.py`; it is not used to initialize a new run of `train.py`.

## Differences from the historical scripts

The historical `train.py` requested a 1024 × 1024 centre crop for the test split, but the available four test TIFFs are 512 × 512. The cleaned loader uses a valid 512 × 512 crop and raises an error for an undersized image instead of silently slicing a different region. A fixed random seed and sorted folder order were added to make new runs repeatable. Results from this code may therefore differ from the published training figure.

The original curve code calls `intersection / sqrt(predicted_area × target_area)` MOC; `moc_legacy` preserves that formula for the plotted MOC. The current manuscript describes an overlap-over-smaller-area metric. `moc_min_area_patch` is recorded separately in the CSV, but it is a patch-level diagnostic, not a reproduction of the manuscript's instance-level MOC.

This release includes the current 20 training images and manual annotations. The repository covers the bacterial segmentation stage only.

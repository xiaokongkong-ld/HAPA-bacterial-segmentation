# Training data

This directory contains 36 grayscale microscopy TIFFs in the current public HAPA bacterial-segmentation training split and their paired masks (`_col.tif`). Nonzero mask pixels indicate bacteria. Thirty-one masks are existing annotations; five are reviewed and accepted model predictions. Their origins and file checksums are listed in `manifest.csv`.

Each image and mask pair is stored in a folder with the same image name. The current three-image test split is separate and is not included. Additional microscopy images may be added in later releas
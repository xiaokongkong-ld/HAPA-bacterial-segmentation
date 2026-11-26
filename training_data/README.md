# Training data

This directory contains a public subset of 36 grayscale microscopy TIFFs from the HAPA bacterial-segmentation training data and their paired masks (`_col.tif`). Nonzero mask pixels indicate bacteria. Thirty-one masks are existing annotations; five are reviewed and accepted model predictions. Their origins and file checksums are listed in `manifest.csv`.

Each image and mask pair is stored in a folder with the same image name. The current three-image test split is separate and is not included. Other training images and annotations are not included in this release and may be added later.

"""Image loading and preprocessing shared by training and inference."""

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    """Read one grayscale TIFF as uint16; expand 8-bit images to 16-bit range."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read TIFF image: {path}")
    if image.ndim != 2 or image.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"Expected a 2D uint8/uint16 TIFF: {path}")
    if image.dtype == np.uint8:
        image = image.astype(np.uint16) * 257
    return image


def read_mask(path: str | Path) -> np.ndarray:
    """Read a binary mask, treating every nonzero pixel as foreground."""
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read TIFF mask: {path}")
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D TIFF mask: {path}")
    return (mask > 0).astype(np.uint8)


def preprocess_patch(patch: np.ndarray) -> np.ndarray:
    """Contrast-stretch and lightly Gaussian-filter the microscopy patch."""
    if patch.dtype != np.uint16 or patch.ndim != 2:
        raise ValueError("Expected a 2D uint16 image patch")
    lower, upper = np.percentile(patch, (1, 99))
    if upper <= lower:
        return np.zeros(patch.shape, dtype=np.float32)
    stretched = np.clip((patch.astype(np.float32) - lower) /
                        (upper - lower), 0, 1)
    smoothed = cv2.GaussianBlur(stretched, (0, 0), sigmaX=0.7)
    return (smoothed * 65535.0).astype(np.float32)

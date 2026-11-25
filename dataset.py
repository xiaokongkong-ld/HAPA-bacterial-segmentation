"""Paired microscopy-image and binary-mask patch dataset."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from image_io import preprocess_patch, read_image, read_mask
from grouping import group_id


class PairedPatchDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        crop_size: int = 512,
        patch_size: int = 256,
        stride: int = 256,
    ) -> None:
        if crop_size < patch_size or stride < 1:
            raise ValueError("crop_size must be >= patch_size and stride must be positive")
        split_dir = Path(data_dir) / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.patch_size = patch_size
        self.images: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.image_names: list[str] = []
        self.group_ids: set[str] = set()
        self.index: list[tuple[int, int, int]] = []

        folders = sorted(path for path in split_dir.iterdir() if path.is_dir())
        if not folders:
            raise ValueError(f"No image subfolders found in {split_dir}")
        for folder in folders:
            image = read_image(folder / f"{folder.name}.tif")
            mask = read_mask(folder / f"{folder.name}_col.tif")
            if image.shape != mask.shape:
                raise ValueError(f"Image and mask shapes differ in {folder}")
            height, width = image.shape
            if height < crop_size or width < crop_size:
                raise ValueError(
                    f"{folder.name} is {width}x{height}, smaller than requested "
                    f"{crop_size}x{crop_size} crop"
                )

            y0 = (height - crop_size) // 2
            x0 = (width - crop_size) // 2
            image = image[y0:y0 + crop_size, x0:x0 + crop_size]
            mask = mask[y0:y0 + crop_size, x0:x0 + crop_size]
            image_id = len(self.images)
            self.images.append(image)
            self.masks.append(mask)
            self.image_names.append(folder.name)
            self.group_ids.add(group_id(folder.name))
            for y in range(0, crop_size - patch_size + 1, stride):
                for x in range(0, crop_size - patch_size + 1, stride):
                    self.index.append((image_id, y, x))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id, y, x = self.index[index]
        size = self.patch_size
        image_patch = self.images[image_id][y:y + size, x:x + size]
        mask_patch = self.masks[image_id][y:y + size, x:x + size]
        image = torch.from_numpy(preprocess_patch(image_patch))[None]
        mask = torch.from_numpy(mask_patch.astype(np.float32))[None]
        return image, mask

"""Paired microscopy-image and binary-mask patch dataset."""

from pathlib import Path

import cv2
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
        augment: bool | str = False,
    ) -> None:
        if crop_size < patch_size or stride < 1:
            raise ValueError("crop_size must be >= patch_size and stride must be positive")
        split_dir = Path(data_dir) / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.patch_size = patch_size
        self.augment = augment
        self.random_crop = split == "train"
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

            if not self.random_crop:
                y0 = (height - crop_size) // 2
                x0 = (width - crop_size) // 2
                image = image[y0:y0 + crop_size, x0:x0 + crop_size]
                mask = mask[y0:y0 + crop_size, x0:x0 + crop_size]
            image_id = len(self.images)
            self.images.append(image)
            self.masks.append(mask)
            self.image_names.append(folder.name)
            self.group_ids.add(group_id(folder.name))
            if self.random_crop:
                samples_per_image = len(range(0, crop_size - patch_size + 1, stride)) ** 2
                self.index.extend((image_id, 0, 0) for _ in range(samples_per_image))
            else:
                for y in range(0, crop_size - patch_size + 1, stride):
                    for x in range(0, crop_size - patch_size + 1, stride):
                        self.index.append((image_id, y, x))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id, y, x = self.index[index]
        size = self.patch_size
        if self.random_crop:
            height, width = self.images[image_id].shape
            y = int(np.random.randint(0, height - size + 1))
            x = int(np.random.randint(0, width - size + 1))
        image_patch = self.images[image_id][y:y + size, x:x + size]
        mask_patch = self.masks[image_id][y:y + size, x:x + size]
        if self.augment:
            quarter_turns = int(np.random.randint(4))
            image_patch = np.rot90(image_patch, quarter_turns)
            mask_patch = np.rot90(mask_patch, quarter_turns)
            if np.random.random() < 0.5:
                image_patch = np.fliplr(image_patch)
                mask_patch = np.fliplr(mask_patch)
            image_patch = np.ascontiguousarray(image_patch)
            mask_patch = np.ascontiguousarray(mask_patch)
            if augment_mode_is_full(self.augment) and np.random.random() < 0.6:
                center = ((size - 1) / 2, (size - 1) / 2)
                matrix = cv2.getRotationMatrix2D(
                    center, float(np.random.uniform(-15, 15)), float(np.random.uniform(0.9, 1.1))
                )
                matrix[:, 2] += np.random.uniform(-8, 8, size=2)
                image_patch = cv2.warpAffine(
                    image_patch, matrix, (size, size), flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                mask_patch = cv2.warpAffine(
                    mask_patch, matrix, (size, size), flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                )
            if augment_mode_is_full(self.augment) and np.random.random() < 0.2:
                grid_x, grid_y = np.meshgrid(np.arange(size), np.arange(size))
                displacement_x = cv2.resize(
                    np.random.uniform(-2, 2, (9, 9)).astype(np.float32),
                    (size, size), interpolation=cv2.INTER_CUBIC,
                )
                displacement_y = cv2.resize(
                    np.random.uniform(-2, 2, (9, 9)).astype(np.float32),
                    (size, size), interpolation=cv2.INTER_CUBIC,
                )
                map_x = grid_x.astype(np.float32) + displacement_x
                map_y = grid_y.astype(np.float32) + displacement_y
                image_patch = cv2.remap(
                    image_patch, map_x, map_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                mask_patch = cv2.remap(
                    mask_patch, map_x, map_y, cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                )
        image = preprocess_patch(image_patch)
        if augment_mode_is_full(self.augment):
            if np.random.random() < 0.7:
                image = image * np.random.uniform(0.85, 1.15) + np.random.uniform(-2000, 2000)
            if np.random.random() < 0.3:
                gamma = np.random.uniform(0.8, 1.2)
                image = np.power(np.clip(image / 65535.0, 0, 1), gamma) * 65535.0
            if np.random.random() < 0.35:
                image = image + np.random.normal(0, np.random.uniform(200, 1000), image.shape)
            if np.random.random() < 0.25:
                image = cv2.GaussianBlur(image.astype(np.float32), (0, 0),
                                         float(np.random.uniform(0.3, 1.0)))
            image = np.clip(image, 0, 65535).astype(np.float32)
        image = torch.from_numpy(np.ascontiguousarray(image))[None]
        mask = torch.from_numpy(mask_patch.astype(np.float32))[None]
        return image, mask


def augment_mode_is_full(augment: bool | str) -> bool:
    return augment is True or augment == "full"

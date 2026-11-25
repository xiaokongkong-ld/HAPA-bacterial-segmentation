"""Generate a binary bacterial mask from one grayscale TIFF image."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from image_io import preprocess_patch, read_image
from model import U_Net


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("best_model.pth"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()

    image = read_image(args.input)
    original_height, original_width = image.shape
    patch_size = 256
    pad_height = (-original_height) % patch_size
    pad_width = (-original_width) % patch_size
    if pad_height or pad_width:
        image = np.pad(image, ((0, pad_height), (0, pad_width)), mode="edge")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = U_Net(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()

    mask = np.zeros(image.shape, dtype=np.uint8)
    with torch.inference_mode():
        for y in range(0, image.shape[0], patch_size):
            for x in range(0, image.shape[1], patch_size):
                patch = image[y:y + patch_size, x:x + patch_size]
                tensor = torch.from_numpy(preprocess_patch(patch))[None, None].to(device)
                logits = model(tensor)
                binary = (logits > 0.5).to(torch.uint8).squeeze().cpu().numpy()
                mask[y:y + patch_size, x:x + patch_size] = binary * 255

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), mask[:original_height, :original_width]):
        raise OSError(f"Could not write mask: {args.output}")
    print(f"Saved {original_width}x{original_height} binary mask: {args.output}")


if __name__ == "__main__":
    main()

"""Copy paired TIFFs into group-disjoint train/test folders without changing originals."""

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

from grouping import group_id


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original directory with train/ and test/")
    parser.add_argument("--destination", type=Path, required=True, help="New, empty private data directory")
    parser.add_argument("--test-groups", nargs="+", default=["14", "1427", "3528"])
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    if destination == source or source in destination.parents:
        parser.error("Destination must be outside the source directory")
    if destination.exists() and any(destination.iterdir()):
        parser.error(f"Destination already contains files: {destination}")

    test_groups = set(args.test_groups)
    if not test_groups:
        parser.error("Specify at least one test group")
    image_rows = []
    seen_names = set()
    for original_split in ("train", "test"):
        split_dir = source / original_split
        if not split_dir.is_dir():
            parser.error(f"Missing source split: {split_dir}")
        for folder in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            name = folder.name
            if name in seen_names:
                parser.error(f"Duplicate image folder name across original splits: {name}")
            seen_names.add(name)
            image = folder / f"{name}.tif"
            mask = folder / f"{name}_col.tif"
            if not image.is_file() or not mask.is_file():
                parser.error(f"Missing image or paired mask in {folder}")
            group = group_id(name)
            image_rows.append((original_split, name, group, image, mask))

    available_groups = {row[2] for row in image_rows}
    if not test_groups <= available_groups:
        parser.error(f"Unknown test groups: {', '.join(sorted(test_groups - available_groups))}")
    if test_groups == available_groups:
        parser.error("The test groups would leave no training images")

    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / "split_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image_name", "group_id", "original_split", "new_split", "image_sha256", "mask_sha256"))
        for original_split, name, group, image, mask in image_rows:
            new_split = "test" if group in test_groups else "train"
            target = destination / new_split / name
            target.mkdir(parents=True, exist_ok=False)
            shutil.copy2(image, target / image.name)
            shutil.copy2(mask, target / mask.name)
            writer.writerow((name, group, original_split, new_split, sha256(image), sha256(mask)))

    for split in ("train", "test"):
        count = sum((group in test_groups) == (split == "test") for _, _, group, _, _ in image_rows)
        print(f"{split}: {count} image/mask pairs")
    print(f"Private group-disjoint split created at {destination}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()

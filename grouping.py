"""Identify acquisition series so related images stay in one split."""

import re


def group_id(image_name: str) -> str:
    """Return the series number in Image_<series>_<frame>_e names."""
    match = re.fullmatch(r"Image_(\d+)_\d+_e", image_name)
    return match.group(1) if match else image_name

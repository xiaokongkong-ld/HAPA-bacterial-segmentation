"""Identify acquisition series so related images stay in one split."""

import re


def group_id(image_name: str) -> str:
    """Return the series number in Image_<series>_<frame>_e names."""
    if image_name.startswith("Predict_"):
        # Source acquisition metadata are unavailable; keep the entire batch together.
        return "predict_batch_3185_3213"
    match = re.fullmatch(r"Image_(\d+)_\d+_e", image_name)
    return match.group(1) if match else image_name

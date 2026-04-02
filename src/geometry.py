"""Bounding-box and spatial utilities for OCR word processing."""

import logging
from typing import List, Tuple

from .config import SPATIAL_BUFFER

log = logging.getLogger(__name__)


def parse_word_bounding_box(word) -> Tuple[int, int, int, int]:
    """Parse word bounding box into (x1, y1, x2, y2)."""
    bbox_str = word.bounding_box
    vals = [float(v) for v in bbox_str.split(",")] if isinstance(bbox_str, str) else list(bbox_str)
    if len(vals) == 4:
        x1, y1, w, h = vals
        return int(x1), int(y1), int(x1 + w), int(y1 + h)
    if len(vals) >= 6:
        return int(vals[0]), int(vals[1]), int(vals[4]), int(vals[5])
    log.warning("Malformed bounding box with %d values: %s", len(vals), bbox_str)
    raise ValueError(f"Bounding box must have 4 or 6+ values, got {len(vals)}: {bbox_str}")


def is_bounding_box_horizontal(bbox: List[int]) -> bool:
    """Check if bounding box is wider than tall."""
    return (bbox[2] - bbox[0]) > (bbox[5] - bbox[1])


def get_bounding_box_half_min_extent(bbox: List[int]) -> int:
    """Return half the shorter dimension of the bounding box."""
    if is_bounding_box_horizontal(bbox):
        return (bbox[5] - bbox[1]) // 2
    return (bbox[2] - bbox[0]) // 2


def are_coordinates_within_distance(co1: int, co2: int, distance: int = SPATIAL_BUFFER) -> bool:
    """Check if two coordinates are within a given distance of each other."""
    return co1 - distance < co2 < co1 + distance

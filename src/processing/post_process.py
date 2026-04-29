"""Post-processing utilities for extraction results."""

import logging
from typing import List, Optional

import cv2
import numpy as np
from ..utils.config import CROP_PADDING
from ..utils.validation import validate_weight_consistency

from .extraction import ContainerResult

log = logging.getLogger(__name__)


def get_dominant_color(image: np.ndarray) -> List[int]:
    """Get most dominant color [B, G, R] from image."""
    colors, count = np.unique(image.reshape(-1, image.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()].tolist()


def extract_dominant_color_from_image_bytes(
    image_bytes: bytes,
    crop_zone: Optional[List[int]] = None,
) -> List[int]:
    """Extract dominant container color from image bytes.

    Uses centre quarter of image if crop_zone is None or invalid.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes – unsupported format or corrupted data")

    h, w = img.shape[:2]

    if not crop_zone or crop_zone == [0, 0, 0, 0]:
        x1, y1, x2, y2 = w // 4, h // 4, 3 * w // 4, 3 * h // 4
        log.debug("No crop zone provided – using centre region (%d,%d,%d,%d)", x1, y1, x2, y2)
    else:
        x1, y1, x2, y2 = crop_zone

    crop_x1 = max(0, x1 - CROP_PADDING)
    crop_y1 = max(0, y1 - CROP_PADDING)
    crop_x2 = min(w, x2 + CROP_PADDING)
    crop_y2 = min(h, y2 + CROP_PADDING)

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        log.warning(
            "Crop zone resulted in empty region (x=%d-%d, y=%d-%d) on %dx%d image",
            crop_x1, crop_x2, crop_y1, crop_y2, w, h,
        )
        raise ValueError(f"Crop zone {crop_zone} produced empty region on {w}x{h} image")

    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
    log.debug("Cropped image to region (%d, %d, %d, %d) from %dx%d", crop_x1, crop_y1, crop_x2, crop_y2, w, h)
    return get_dominant_color(cropped)


def post_process_result(
    result: ContainerResult,
    image_bytes: bytes,
) -> ContainerResult:
    """Post-process extraction result to ensure consistent output.

    Fills in container_color when missing or zero-default.
    """
    if not result.container_number:
        return result

    if result.container_color == [0, 0, 0] or not result.container_color:
        crop_zone: Optional[list] = None
        if result.bounding_box and result.bounding_box != [0, 0, 0, 0]:
            crop_zone = result.bounding_box
        try:
            color = extract_dominant_color_from_image_bytes(image_bytes, crop_zone)
            result.container_color = color
            if crop_zone:
                result.bounding_box = crop_zone
            log.debug("Extracted color: %s", color)
        except ValueError as e:
            log.warning("Could not extract container color: %s", e)

    # Validate weight consistency if all weight values are present
    if result.weights is not None:
        tare = result.weights.tare_weight
        payload = result.weights.payload_weight
        max_gross = result.weights.maximum_gross_weight
        # Check that all values are present (not None)
        if None not in (tare.kilograms, tare.pounds, payload.kilograms, payload.pounds, max_gross.kilograms, max_gross.pounds):
            is_valid, detailed_reason = validate_weight_consistency(tare.kilograms, tare.pounds, payload.kilograms, payload.pounds, max_gross.kilograms, max_gross.pounds)
            if not is_valid:
                result.valid = False
                result.reason = detailed_reason

    return result
"""Image preprocessing utilities."""

import logging

import numpy as np
import cv2
from typing import List, Optional
from .config import DOWNSCALE_THRESHOLD, CROP_PADDING

log = logging.getLogger(__name__)


def downscale_image(image: np.ndarray) -> np.ndarray:
    """Downscale image if width >= threshold."""
    h, w = image.shape[:2]
    if w >= DOWNSCALE_THRESHOLD:
        scale = DOWNSCALE_THRESHOLD / w
        log.debug("Downscaling image from %dx%d to %dx%d", w, h, DOWNSCALE_THRESHOLD, int(h * scale))
        return cv2.resize(image, (DOWNSCALE_THRESHOLD, int(h * scale)))
    return image


def get_dominant_color(image: np.ndarray) -> List[int]:
    """Get the most dominant color [B, G, R] from the image."""
    colors, count = np.unique(image.reshape(-1, image.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()].tolist()


def extract_dominant_color_from_image_bytes(
    image_bytes: bytes,
    crop_zone: Optional[List[int]] = None,
) -> List[int]:
    """Extract dominant container color from byte image with optional cropping.

    If *crop_zone* is ``None`` or ``[0, 0, 0, 0]`` the centre quarter of the
    image is used as a sensible fallback so callers never have to duplicate
    that logic.
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

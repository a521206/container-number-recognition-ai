"""Image preprocessing utilities."""

import logging

import numpy as np
import cv2
from ..utils.config import DOWNSCALE_THRESHOLD

log = logging.getLogger(__name__)


def downscale_image(image: np.ndarray) -> np.ndarray:
    """Downscale image if width >= threshold."""
    h, w = image.shape[:2]
    if w >= DOWNSCALE_THRESHOLD:
        scale = DOWNSCALE_THRESHOLD / w
        log.debug("Downscaling image from %dx%d to %dx%d", w, h, DOWNSCALE_THRESHOLD, int(h * scale))
        return cv2.resize(image, (DOWNSCALE_THRESHOLD, int(h * scale)))
    return image




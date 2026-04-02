"""Image preprocessing utilities."""

import numpy as np
import cv2
from typing import Tuple, List
from .config import DOWNSCALE_THRESHOLD, CROP_PADDING


def downscale_image(image: np.ndarray) -> np.ndarray:
    """
    Downscale image if width >= threshold.

    Args:
        image: Input image

    Returns:
        Resized image or original if no downscaling needed
    """
    h, w = image.shape[:2]
    if w >= DOWNSCALE_THRESHOLD:
        scale = DOWNSCALE_THRESHOLD / w
        new_w = DOWNSCALE_THRESHOLD
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h))
    return image


def get_container_color(image: np.ndarray) -> List[int]:
    """
    Get the most dominant color from the image.

    Args:
        image: Input image

    Returns:
        [B, G, R] of dominant color
    """
    colors, count = np.unique(image.reshape(-1, image.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()].tolist()


def get_container_color_from_bytes(image_bytes: bytes, crop_zone: List[int]) -> List[int]:
    """
    Extract container color from byte image with cropping.

    Args:
        image_bytes: Image as bytes
        crop_zone: [x1, y1, x2, y2] bounding box

    Returns:
        [B, G, R] color
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes – unsupported format or corrupted data")

    h, w = img.shape[:2]
    x1, y1, x2, y2 = crop_zone

    # Crop with padding, clamped to image bounds
    crop_x1 = max(0, x1 - CROP_PADDING)
    crop_y1 = max(0, y1 - CROP_PADDING)
    crop_x2 = min(w, x2 + CROP_PADDING)
    crop_y2 = min(h, y2 + CROP_PADDING)

    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
    return get_container_color(cropped)
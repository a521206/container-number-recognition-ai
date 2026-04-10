"""Unified post-processing for extraction results."""

import logging
from typing import Optional, Tuple

from .extraction import ContainerResult
from .preprocessing import extract_dominant_color_from_image_bytes

log = logging.getLogger(__name__)


def post_process_result(
    result: ContainerResult,
    image_bytes: bytes,
    bounding_box: Optional[Tuple[int, int, int, int]] = None,
) -> ContainerResult:
    """Post-process an extraction result to ensure consistent output.

    Fills in ``container_color`` when it is missing or the zero-default,
    delegating all color-extraction and centre-region fallback logic to
    :func:`preprocessing.extract_dominant_color_from_image_bytes`.
    """
    if not result.container_number:
        return result

    if result.container_color == [0, 0, 0] or not result.container_color:
        crop_zone: Optional[list] = None
        if bounding_box and bounding_box != (0, 0, 0, 0):
            crop_zone = list(bounding_box)
        elif result.bounding_box and result.bounding_box != [0, 0, 0, 0]:
            crop_zone = result.bounding_box
        # If crop_zone is still None, extract_dominant_color_from_image_bytes
        # will fall back to the centre region automatically.
        try:
            color = extract_dominant_color_from_image_bytes(image_bytes, crop_zone)
            result.container_color = color
            if crop_zone:
                result.bounding_box = crop_zone
            log.debug("Extracted color: %s", color)
        except ValueError as e:
            log.warning("Could not extract container color: %s", e)

    return result
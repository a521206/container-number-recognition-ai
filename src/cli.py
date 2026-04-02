"""Command-line interface for container detection."""

import os
import cv2
import numpy as np
from .config import DATA_DIR
from .ocr_client import OCRClient
from .extraction import extract_container_regex, extract_container_location
from .preprocessing import get_container_color_from_bytes, downscale_image
from .models import ContainerResult


def process_image(image_path: str, ocr_client: OCRClient) -> ContainerResult:
    """
    Process a single image file.

    The caller is responsible for creating and passing the ``OCRClient`` so
    that the client is initialised only once across multiple images.

    The image is read once from disk, encoded to JPEG bytes, and those same
    bytes are reused for both OCR and dominant-colour detection, avoiding a
    second file-read.

    Args:
        image_path: Path to image
        ocr_client: Shared OCR client instance

    Returns:
        Detection result
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return ContainerResult(error=f"Could not read image: {image_path}")

        img = downscale_image(img)
        _, encoded = cv2.imencode('.JPG', img)
        img_bytes = encoded.tobytes()

        ocr_result = ocr_client.recognize_printed_text(img_bytes)

        if not ocr_result or not hasattr(ocr_result, 'regions') or not ocr_result.regions:
            return ContainerResult(error="No text detected")

        # Try regex extraction first; fall back to location extraction only for
        # missing fields (mirrors api.py – no redundant second call).
        result = extract_container_regex(ocr_result)
        if not result.container_number or not result.container_type:
            location_result = extract_container_location(ocr_result)
            if not result.container_number:
                result.container_number = location_result.container_number
                result.bounding_box = location_result.bounding_box
            if not result.container_type:
                result.container_type = location_result.container_type

        # Reuse the already-encoded bytes for colour detection
        if result.bounding_box != [0, 0, 0, 0]:
            result.container_color = get_container_color_from_bytes(img_bytes, result.bounding_box)

        return result

    except Exception as e:
        return ContainerResult(error=str(e))


def main():
    """Main CLI function."""
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} not found")
        return

    ocr_client = OCRClient()

    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.isfile(filepath) and filename.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
            print(f"Processing {filename}...")
            result = process_image(filepath, ocr_client)
            print(f"Result: {result}")

            if result.error or not result.container_number:
                print(f"Skipped: {result.error or 'No container detected'}")
                continue



if __name__ == "__main__":
    main()
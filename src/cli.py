"""Command-line interface for container detection."""

import json
import logging
import os

import cv2
from .config import DATA_DIR
from .ocr_client import OCRClient
from .extraction import ContainerResult, run_extraction_pipeline
from .preprocessing import downscale_image

log = logging.getLogger(__name__)


def process_image(image_path: str, ocr_client: OCRClient) -> ContainerResult:
    """Process a single image file and return detection results."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return ContainerResult(error=f"Could not read image: {image_path}")

        img = downscale_image(img)
        _, encoded = cv2.imencode('.JPG', img)
        img_bytes = encoded.tobytes()

        ocr_result = ocr_client.recognize_printed_text(img_bytes)
        if not ocr_result.regions:
            return ContainerResult(error="No text detected")

        return run_extraction_pipeline(ocr_result, img_bytes)

    except ValueError as e:
        log.error("Validation error processing %s: %s", image_path, e)
        return ContainerResult(error=str(e))
    except TimeoutError as e:
        log.error("OCR timeout processing %s: %s", image_path, e)
        return ContainerResult(error=str(e))
    except Exception as e:
        log.exception("Unexpected error processing %s", image_path)
        return ContainerResult(error=f"{type(e).__name__}: {e}")


def main():
    """Main CLI function."""
    if not os.path.exists(DATA_DIR):
        log.error("Data directory '%s' not found", DATA_DIR)
        return

    ocr_client = OCRClient()

    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.isfile(filepath) and filename.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
            result = process_image(filepath, ocr_client)

            if result.error or not result.container_number:
                log.warning("Skipped %s: %s", filename, result.error or "No container detected")
            else:
                print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

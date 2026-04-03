"""Command-line interface for container detection."""

import json
import logging
import os
import sys

import cv2
from .config import DATA_DIR
from .ocr_client import OCRClient
from .document_intelligence_client import DocumentIntelligenceClient
from .extraction import ContainerResult, run_extraction_pipeline
from .preprocessing import downscale_image

from dotenv import load_dotenv
load_dotenv()

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


def process_image_llama_extract(image_path: str) -> ContainerResult:
    """Process a single image file using Llama Extract."""
    from .llama_extract_client import LlamaExtractClient

    try:
        client = LlamaExtractClient()
        return client.extract_from_file(image_path)
    except ValueError as e:
        log.error("Configuration error: %s", e)
        return ContainerResult(error=str(e))
    except Exception as e:
        log.exception("Unexpected error processing %s with Llama Extract", image_path)
        return ContainerResult(error=f"{type(e).__name__}: {e}")


def process_image_document_intelligence(image_path: str) -> ContainerResult:
    """Process a single image file using Azure Document Intelligence."""
    try:
        client = DocumentIntelligenceClient()
        return client.extract_from_file(image_path)
    except ValueError as e:
        log.error("Configuration error: %s", e)
        return ContainerResult(error=str(e))
    except Exception as e:
        log.exception("Unexpected error processing %s with Document Intelligence", image_path)
        return ContainerResult(error=f"{type(e).__name__}: {e}")


def main():
    """Main CLI function."""
    use_llama_extract = "--llama-extract" in sys.argv
    use_doc_intelligence = "--doc-intelligence" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    # Check if a specific file path was provided
    file_paths = []
    for arg in args:
        if os.path.isfile(arg):
            file_paths.append(arg)

    if file_paths:
        # Process specific file(s)
        if use_llama_extract:
            for filepath in file_paths:
                result = process_image_llama_extract(filepath)
                if result.error or not result.container_number:
                    log.warning("Skipped %s: %s", filepath, result.error or "No container detected")
                else:
                    print(json.dumps(result.to_dict(), indent=2))
        elif use_doc_intelligence:
            for filepath in file_paths:
                result = process_image_document_intelligence(filepath)
                if result.error or not result.container_number:
                    log.warning("Skipped %s: %s", filepath, result.error or "No container detected")
                else:
                    print(json.dumps(result.to_dict(), indent=2))
        else:
            ocr_client = OCRClient()
            for filepath in file_paths:
                result = process_image(filepath, ocr_client)
                if result.error or not result.container_number:
                    log.warning("Skipped %s: %s", filepath, result.error or "No container detected")
                else:
                    print(json.dumps(result.to_dict(), indent=2))
    elif not os.path.exists(DATA_DIR):
        log.error("Data directory '%s' not found", DATA_DIR)
        return
    else:
        # Process all files in data directory
        if use_llama_extract:
            log.info("Using Llama Extract for extraction")
            for filename in os.listdir(DATA_DIR):
                filepath = os.path.join(DATA_DIR, filename)
                if os.path.isfile(filepath) and filename.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
                    result = process_image_llama_extract(filepath)

                    if result.error or not result.container_number:
                        log.warning("Skipped %s: %s", filename, result.error or "No container detected")
                    else:
                        print(json.dumps(result.to_dict(), indent=2))
        elif use_doc_intelligence:
            log.info("Using Azure Document Intelligence for extraction")
            for filename in os.listdir(DATA_DIR):
                filepath = os.path.join(DATA_DIR, filename)
                if os.path.isfile(filepath) and filename.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
                    result = process_image_document_intelligence(filepath)

                    if result.error or not result.container_number:
                        log.warning("Skipped %s: %s", filename, result.error or "No container detected")
                    else:
                        print(json.dumps(result.to_dict(), indent=2))
        else:
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

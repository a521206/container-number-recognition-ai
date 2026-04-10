"""OCR extraction client using Azure Computer Vision."""

import logging
import time
import types
from io import BytesIO
from typing import Optional

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

from ..config import VISION_ENDPOINT, VISION_KEY
from ..extraction import ContainerResult
from ..preprocessing import downscale_image
from .base import ExtractionClient

log = logging.getLogger(__name__)

_OCR_POLL_TIMEOUT = 60
_OCR_POLL_INTERVAL = 1


class OCRClient:
    """Wrapper for Azure Computer Vision Read API."""

    def __init__(self):
        if not VISION_ENDPOINT or not VISION_KEY:
            raise ValueError("VISION_ENDPOINT and VISION_KEY must be set in environment")
        credentials = CognitiveServicesCredentials(VISION_KEY)
        self.client = ComputerVisionClient(VISION_ENDPOINT, credentials)
        log.info("OCRClient initialized with endpoint %s", VISION_ENDPOINT)

    @property
    def name(self) -> str:
        return "ocr"

    def recognize_printed_text(self, image_bytes: bytes):
        """
        Recognise text from image bytes using the Azure Read API.
        Polls until the operation completes or timeout is reached.
        """
        log.debug("Sending image (%d bytes) to Azure Read API", len(image_bytes))
        read_response = self.client.read_in_stream(image=BytesIO(image_bytes), raw=True)
        operation_id = read_response.headers["Operation-Location"].split("/")[-1]
        log.debug("OCR operation started: %s", operation_id)

        deadline = time.monotonic() + _OCR_POLL_TIMEOUT
        while True:
            read_result = self.client.get_read_result(operation_id)
            status = read_result.status.lower()
            if status not in ("notstarted", "running"):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Azure Read API did not complete within {_OCR_POLL_TIMEOUT}s "
                    f"(operation_id={operation_id})"
                )
            time.sleep(_OCR_POLL_INTERVAL)

        if status == "succeeded":
            pages = read_result.analyze_result.read_results
            log.debug("OCR operation %s succeeded with %d page(s)", operation_id, len(pages))
        else:
            log.warning("OCR operation %s finished with unexpected status: %s", operation_id, status)
            pages = []

        return types.SimpleNamespace(regions=pages)

    def extract_from_file(self, file_path: str) -> ContainerResult:
        """Extract container data from a file path using OCR."""
        import cv2
        
        try:
            img = cv2.imread(file_path)
            if img is None:
                return ContainerResult(error=f"Could not read image: {file_path}")

            img = downscale_image(img)
            _, encoded = cv2.imencode('.JPG', img)
            img_bytes = encoded.tobytes()

            ocr_result = self.recognize_printed_text(img_bytes)
            if not ocr_result.regions:
                return ContainerResult(error="No text detected")

            from ..extraction import run_extraction_pipeline
            return run_extraction_pipeline(ocr_result)

        except ValueError as e:
            log.error("Validation error processing %s: %s", file_path, e)
            return ContainerResult(error=str(e))
        except TimeoutError as e:
            log.error("OCR timeout processing %s: %s", file_path, e)
            return ContainerResult(error=str(e))
        except Exception as e:
            log.exception("Unexpected error processing %s", file_path)
            return ContainerResult(error=f"{type(e).__name__}: {e}")

    def extract_from_bytes(self, data: bytes, filename: str = "image.jpg") -> ContainerResult:
        """Extract container data from image bytes using OCR."""
        import cv2
        import numpy as np
        
        try:
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return ContainerResult(error="Could not decode image bytes")

            img = downscale_image(img)
            _, encoded = cv2.imencode('.JPG', img)
            img_bytes = encoded.tobytes()

            ocr_result = self.recognize_printed_text(img_bytes)
            if not ocr_result.regions:
                return ContainerResult(error="No text detected")

            from ..extraction import run_extraction_pipeline
            return run_extraction_pipeline(ocr_result)

        except ValueError as e:
            log.error("Validation error processing bytes: %s", e)
            return ContainerResult(error=str(e))
        except TimeoutError as e:
            log.error("OCR timeout processing bytes: %s", e)
            return ContainerResult(error=str(e))
        except Exception as e:
            log.exception("Unexpected error processing bytes")
            return ContainerResult(error=f"{type(e).__name__}: {e}")
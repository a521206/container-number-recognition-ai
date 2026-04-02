"""Azure Computer Vision OCR client."""

import logging
import time
import types
from io import BytesIO
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from .config import VISION_ENDPOINT, VISION_KEY

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

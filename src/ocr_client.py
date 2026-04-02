"""Azure Computer Vision OCR client."""

import time
import types
from io import BytesIO
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from .config import VISION_ENDPOINT, VISION_KEY

_OCR_POLL_TIMEOUT = 60
_OCR_POLL_INTERVAL = 1


class OCRClient:
    """Wrapper for Azure Computer Vision Read API."""

    def __init__(self):
        if not VISION_ENDPOINT or not VISION_KEY:
            raise ValueError("VISION_ENDPOINT and VISION_KEY must be set in environment")
        credentials = CognitiveServicesCredentials(VISION_KEY)
        self.client = ComputerVisionClient(VISION_ENDPOINT, credentials)

    def recognize_printed_text(self, image_bytes: bytes):
        """
        Recognise text from image bytes using the Azure Read API.
        Polls until the operation completes or timeout is reached.
        """
        read_response = self.client.read_in_stream(image=BytesIO(image_bytes), raw=True)
        operation_id = read_response.headers["Operation-Location"].split("/")[-1]

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

        pages = read_result.analyze_result.read_results if status == "succeeded" else []
        return types.SimpleNamespace(regions=pages)

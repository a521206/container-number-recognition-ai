"""Azure Computer Vision OCR client."""

import time
from io import BytesIO
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from .config import VISION_ENDPOINT, VISION_KEY

# Maximum seconds to wait for the Read API to finish before raising.
_OCR_POLL_TIMEOUT = 60
_OCR_POLL_INTERVAL = 1


# ---------------------------------------------------------------------------
# Lightweight data-transfer objects that mirror the shape expected by the
# extraction helpers.  Defined at module level so they are created once, not
# redefined on every call to recognize_printed_text.
# ---------------------------------------------------------------------------

class _OCRWord:
    """Minimal representation of a recognised word."""
    __slots__ = ("text", "bounding_box")

    def __init__(self, text: str, bounding_box: str) -> None:
        self.text = text
        self.bounding_box = bounding_box


class _OCRLine:
    """Minimal representation of a recognised text line."""
    __slots__ = ("words", "bounding_box")

    def __init__(self, words: list, bounding_box: str) -> None:
        self.words = words
        self.bounding_box = bounding_box


class _OCRRegion:
    """Minimal representation of a recognised text region."""
    __slots__ = ("lines",)

    def __init__(self, lines: list) -> None:
        self.lines = lines


class _OCRResult:
    """Top-level OCR result carrying a list of regions."""
    __slots__ = ("regions",)

    def __init__(self, regions: list) -> None:
        self.regions = regions


def _polygon_to_bbox_str(polygon: list) -> str:
    """Convert an 8-value polygon list to a 'x,y,w,h' bounding-box string."""
    if len(polygon) >= 4:
        x_coords = polygon[::2]
        y_coords = polygon[1::2]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        return f"{min_x},{min_y},{max_x - min_x},{max_y - min_y}"
    return ",".join(map(str, polygon))


class OCRClient:
    """Wrapper for Azure Computer Vision Read API."""

    def __init__(self):
        if not VISION_ENDPOINT or not VISION_KEY:
            raise ValueError("VISION_ENDPOINT and VISION_KEY must be set in environment")
        credentials = CognitiveServicesCredentials(VISION_KEY)
        self.client = ComputerVisionClient(VISION_ENDPOINT, credentials)

    def recognize_printed_text(self, image_bytes: bytes) -> _OCRResult:
        """
        Recognise text from image bytes using the Azure Read API.

        Polls until the operation completes or ``_OCR_POLL_TIMEOUT`` seconds
        have elapsed, raising ``TimeoutError`` if the API does not respond
        within that window.

        Args:
            image_bytes: Raw image data

        Returns:
            _OCRResult with regions/lines/words matching the extraction API
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

        regions: list = []
        if status == "succeeded":
            for page in read_result.analyze_result.read_results:
                lines = []
                for line in page.lines:
                    words = [
                        _OCRWord(word.text, _polygon_to_bbox_str(word.bounding_box))
                        for word in line.words
                    ]
                    lines.append(_OCRLine(words, _polygon_to_bbox_str(line.bounding_box)))
                if lines:
                    regions.append(_OCRRegion(lines))

        return _OCRResult(regions)
"""Combined extraction pipeline using OCR and Llama Extract together."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Tuple

from ..clients import OCRClient, LlamaExtractClient
from ..clients.base import ExtractionClient
from ..processing.extraction import ContainerResult
from ..processing.post_process import post_process_result
from ..utils.validation import validate_iso6346

log = logging.getLogger(__name__)

_client_cache: dict = {}


def clear_client_cache() -> None:
    """Clear the client cache. Useful for testing."""
    _client_cache.clear()


class ExtractionMethod:
    """Available extraction methods."""
    OCR = "ocr"
    LLAMA_EXTRACT = "llama_extract"


def get_client(method: str) -> Optional[ExtractionClient]:
    """Get the appropriate client for the extraction method (cached)."""
    if method in _client_cache:
        return _client_cache[method]
    
    if method == ExtractionMethod.LLAMA_EXTRACT:
        client = LlamaExtractClient()
    else:
        client = OCRClient()
    
    _client_cache[method] = client
    return client


def _run_clients(
    ocr_fn: Callable[[], ContainerResult],
    llama_fn: Callable[[], ContainerResult],
    image_bytes: bytes,
) -> Tuple[ContainerResult, str]:
    """Invoke both clients in parallel, tolerate individual failures, combine."""
    ocr_result = None
    llama_result = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        ocr_future = executor.submit(ocr_fn)
        llama_future = executor.submit(llama_fn)

        try:
            ocr_result = ocr_future.result()
        except Exception as e:
            log.warning("OCR extraction failed: %s", e)

        try:
            llama_result = llama_future.result()
        except Exception as e:
            log.warning("Llama Extract failed: %s", e)

    return combine_results(ocr_result, llama_result, image_bytes)


def run_combined_extraction(
    image_path: str,
    image_bytes: Optional[bytes] = None,
) -> Tuple[ContainerResult, str]:
    """Run both OCR and Llama Extract from a file path, then combine results.

    Strategy:
    1. Read image file as bytes
    2. Delegate to run_combined_extraction_from_bytes
    """
    img_bytes = image_bytes or _read_image_bytes(image_path)
    filename = os.path.basename(image_path) if image_path else "image.jpg"
    return run_combined_extraction_from_bytes(img_bytes, filename)


def run_combined_extraction_from_bytes(
    image_bytes: bytes,
    filename: str = "image.jpg",
) -> Tuple[ContainerResult, str]:
    """Run combined extraction directly from image bytes.

    Both clients are invoked via their ``extract_from_bytes`` methods so no
    combined-pipeline temp file is created here.  Each client manages its own
    temporary resources internally as needed.
    """
    ocr_client = get_client(ExtractionMethod.OCR)
    llama_client = get_client(ExtractionMethod.LLAMA_EXTRACT)
    return _run_clients(
        lambda: ocr_client.extract_from_bytes(image_bytes, filename),
        lambda: llama_client.extract_from_bytes(image_bytes, filename),
        image_bytes,
    )


def _finish(result: ContainerResult, method: str, image_bytes: bytes) -> Tuple[ContainerResult, str]:
    """Post-process *result* and set valid flag if not already decided."""
    result = post_process_result(result, image_bytes)
    if result.container_number and result.valid is None:
        result.valid = validate_iso6346(result.container_number)
        if result.valid and len(result.container_number) == 10:
            result.reason = "Check digit not visible but container number is valid"
        elif not result.valid:
            result.reason = "Invalid container number format"
    return result, method


def combine_results(
    ocr_result: Optional[ContainerResult],
    llama_result: Optional[ContainerResult],
    image_bytes: bytes,
) -> Tuple[ContainerResult, str]:
    """Combine results from OCR and Llama Extract.

    Priority:
    1. Both match → validate, use combined if valid, else invalid
    2. Both differ, one valid → use the valid one
    3. Both differ, both valid → prefer OCR (has bbox)
    4. Both differ, both invalid → prefer Llama (more fields)
    5. Only one has container_number → use that

    The returned ContainerResult always has ``method_used`` set.
    """
    if not ocr_result and not llama_result:
        return ContainerResult(error="All extraction methods failed"), "none"

    # Save error messages before nulling results.
    ocr_error = ocr_result.error if ocr_result else None
    llama_error = llama_result.error if llama_result else None

    if ocr_result and ocr_result.error:
        ocr_result = None
    if llama_result and llama_result.error:
        llama_result = None

    has_ocr = ocr_result and ocr_result.container_number
    has_llama = llama_result and llama_result.container_number

    if has_ocr and has_llama:
        ocr_num = ocr_result.container_number.upper()
        llama_num = llama_result.container_number.upper()

        if ocr_num == llama_num:
            log.info("Both methods returned matching container number: %s", ocr_num)
            if validate_iso6346(ocr_num):
                return _finish(_merge_results(ocr_result, llama_result), "combined", image_bytes)
            log.warning("Matched container number %s failed validation", ocr_num)
            result = _merge_results(ocr_result, llama_result)
            result.valid = False
            return _finish(result, "combined", image_bytes)

        log.warning("Container number mismatch – OCR: %s, Llama: %s", ocr_num, llama_num)
        ocr_valid = validate_iso6346(ocr_num)
        llama_valid = validate_iso6346(llama_num)

        if ocr_valid and not llama_valid:
            log.info("Using OCR result (valid container number)")
            return _finish(ocr_result, "ocr", image_bytes)
        if llama_valid and not ocr_valid:
            log.info("Using Llama result (valid container number)")
            return _finish(_merge_results(llama_result, ocr_result), "llama_extract", image_bytes)
        if ocr_valid and llama_valid:
            log.warning("Both valid but different – using OCR (has bbox)")
            return _finish(ocr_result, "ocr", image_bytes)
        log.warning("Both invalid – using Llama (more fields)")
        result = _merge_results(llama_result, ocr_result)
        result.valid = False
        return _finish(result, "llama_extract", image_bytes)

    if has_ocr:
        log.info("Using OCR result (Llama failed or no result)")
        return _finish(ocr_result, "ocr", image_bytes)

    if has_llama:
        log.info("Using Llama result (OCR failed or no result)")
        return _finish(llama_result, "llama_extract", image_bytes)

    error_parts = []
    if ocr_error:
        error_parts.append(f"OCR: {ocr_error}")
    if llama_error:
        error_parts.append(f"Llama: {llama_error}")
    return ContainerResult(valid=False, reason=f"All methods failed: {'; '.join(error_parts)}"), "none"


def _merge_results(
    primary: ContainerResult,
    secondary: Optional[ContainerResult],
) -> ContainerResult:
    """Merge two results with primary taking precedence for a superset of fields.

    Color extraction / post-processing is intentionally **not** done here;
    callers must pass the merged result through :func:`_finish`.
    """
    result = ContainerResult()

    result.container_number = primary.container_number

    if primary.container_type:
        result.container_type = primary.container_type
    elif secondary and secondary.container_type:
        result.container_type = secondary.container_type

    if primary.bounding_box and primary.bounding_box != [0, 0, 0, 0]:
        result.bounding_box = primary.bounding_box
    elif secondary and secondary.bounding_box and secondary.bounding_box != [0, 0, 0, 0]:
        result.bounding_box = secondary.bounding_box

    if primary.weights:
        result.weights = primary.weights
    elif secondary and secondary.weights:
        result.weights = secondary.weights

    if primary.owner_operator:
        result.owner_operator = primary.owner_operator
    elif secondary and secondary.owner_operator:
        result.owner_operator = secondary.owner_operator

    return result


def _read_image_bytes(image_path: str) -> bytes:
    """Read image file as bytes."""
    with open(image_path, "rb") as f:
        return f.read()
"""Unified processing pipeline for container extraction."""

import logging
from typing import Optional

from .clients import OCRClient, LlamaExtractClient
from .clients.base import ExtractionClient

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



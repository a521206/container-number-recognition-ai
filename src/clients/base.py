"""Base extraction client protocol."""

import logging
from typing import Protocol, Optional

from ..processing.extraction import ContainerResult

log = logging.getLogger(__name__)


class ExtractionClient(Protocol):
    """Protocol defining the interface for extraction clients."""

    @property
    def name(self) -> str:
        """Return the name of the extraction method."""
        ...

    def extract_from_file(self, file_path: str) -> ContainerResult:
        """Extract container data from a file path."""
        ...

    def extract_from_bytes(self, data: bytes, filename: str = "image.jpg") -> ContainerResult:
        """Extract container data from image bytes."""
        ...
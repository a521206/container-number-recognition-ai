"""Data models for container number recognition."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ContainerResult:
    """Result of container detection."""
    container_number: str = ""
    container_type: str = ""
    bounding_box: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    container_color: List[int] = field(default_factory=lambda: [0, 0, 0])
    error: Optional[str] = None
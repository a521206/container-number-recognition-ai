"""Data models for container processing."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WeightValue:
    """Weight in pounds and kilograms."""
    pounds: Optional[int] = None
    kilograms: Optional[int] = None

    def to_dict(self) -> Optional[dict]:
        if self.pounds is None and self.kilograms is None:
            return None
        return {"pounds": self.pounds, "kilograms": self.kilograms}


@dataclass
class OwnerOperator:
    """Container owner/operator details."""
    name: Optional[str] = None
    location: Optional[str] = None

    def to_dict(self) -> Optional[dict]:
        if self.name is None and self.location is None:
            return None
        return {"name": self.name, "location": self.location}


@dataclass
class Weights:
    """Container weight specifications."""
    tare_weight: WeightValue = field(default_factory=WeightValue)
    payload_weight: WeightValue = field(default_factory=WeightValue)
    maximum_gross_weight: WeightValue = field(default_factory=WeightValue)

    def to_dict(self) -> Optional[dict]:
        d = {
            "tare_weight": self.tare_weight.to_dict(),
            "payload_weight": self.payload_weight.to_dict(),
            "maximum_gross_weight": self.maximum_gross_weight.to_dict(),
        }
        if all(v is None for v in d.values()):
            return None
        return d


@dataclass
class ContainerResult:
    """Result of container detection."""
    file_name: str = ""
    container_number: str = ""
    container_type: str = ""
    bounding_box: list = field(default_factory=lambda: [0, 0, 0, 0])
    container_color: list = field(default_factory=lambda: [0, 0, 0])
    error: Optional[str] = None
    status: Optional[str] = None
    weights: Optional[Weights] = None
    owner_code: Optional[str] = None
    container_id: Optional[str] = None
    serial_number: Optional[str] = None
    owner_operator: Optional[OwnerOperator] = None
    container_type_code: Optional[str] = None
    method_used: Optional[str] = None
    valid: Optional[bool] = None
    reason: Optional[str] = None

    def _derive_fields(self) -> None:
        """Populate structured fields from container_number and container_type (idempotent)."""
        if self.owner_code is not None:
            return
        if not self.container_number or len(self.container_number) != 11:
            return
        self.owner_code = self.container_number[:4]
        self.serial_number = self.container_number[4:]
        self.container_id = f"{self.owner_code} {self.serial_number[:6]} {self.serial_number[6]}"
        if self.container_type:
            self.container_type_code = self.container_type

    def to_dict(self) -> dict:
        self._derive_fields()
        d: dict = {}
        if self.file_name:
            d["file_name"] = self.file_name
        if self.error:
            d["error"] = self.error
            return d
        if self.container_number:
            d["container_number"] = self.container_number
        if self.container_type:
            d["container_type"] = self.container_type
        if self.bounding_box and self.bounding_box != [0, 0, 0, 0]:
            d["bounding_box"] = self.bounding_box
        if self.container_color and self.container_color != [0, 0, 0]:
            d["container_color"] = self.container_color
        if self.status is not None:
            d["status"] = self.status
        if self.weights is not None:
            w = self.weights.to_dict()
            if w is not None:
                d["weights"] = w
        if self.owner_code is not None:
            d["owner_code"] = self.owner_code
        if self.container_id is not None:
            d["container_id"] = self.container_id
        if self.serial_number is not None:
            d["serial_number"] = self.serial_number
        if self.owner_operator is not None:
            oo = self.owner_operator.to_dict()
            if oo is not None:
                d["owner_operator"] = oo
        if self.container_type_code is not None:
            d["container_type_code"] = self.container_type_code
        if self.method_used:
            d["method_used"] = self.method_used
        if self.valid is not None:
            d["valid"] = self.valid
        if self.reason is not None:
            d["reason"] = self.reason
        return d


class MergedOCRLine:
    """Synthetic line produced by merging vertically-overlapping OCR lines."""
    __slots__ = ("words", "bounding_box")

    def __init__(self, words: list) -> None:
        self.words = words
        self.bounding_box = self._compute_bounding_box()

    def _compute_bounding_box(self) -> str:
        """Compute combined bounding box of all words."""
        if not self.words:
            return ""
        x1, y1, x2, y2 = None, None, None, None
        from ..utils.geometry import parse_word_bounding_box
        for w in self.words:
            wx1, wy1, wx2, wy2 = parse_word_bounding_box(w)
            if x1 is None:
                x1, y1, x2, y2 = wx1, wy1, wx2, wy2
            else:
                x1 = min(x1, wx1)
                y1 = min(y1, wy1)
                x2 = max(x2, wx2)
                y2 = max(y2, wy2)
        return f"{x1},{y1},{x2},{y2}"
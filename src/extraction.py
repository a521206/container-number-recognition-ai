"""Container number extraction logic."""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from .config import (
    CARRIER_PREFIXES, CARRIER_PREFIX_SET, CONTAINER_TYPE_PREFIXES,
    SPATIAL_BUFFER, CONTAINER_NUMBER_LENGTH, PREFIX_FUZZY_MAP,
)
from .geometry import (
    parse_word_bounding_box, is_bounding_box_horizontal,
    get_bounding_box_half_min_extent, are_coordinates_within_distance,
)
from .preprocessing import extract_dominant_color_from_image_bytes
from .validation import validate_iso6346_check_digit

log = logging.getLogger(__name__)


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
    container_number: str = ""
    container_type: str = ""
    bounding_box: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    container_color: List[int] = field(default_factory=lambda: [0, 0, 0])
    error: Optional[str] = None
    status: Optional[str] = None
    weights: Optional[Weights] = None
    owner_code: Optional[str] = None
    container_id: Optional[str] = None
    serial_number: Optional[str] = None
    owner_operator: Optional[OwnerOperator] = None
    container_type_code: Optional[str] = None

    def _derive_fields(self) -> None:
        """Populate structured fields from container_number and container_type."""
        if not self.container_number or len(self.container_number) != 11:
            return
        self.owner_code = self.container_number[:4]
        self.serial_number = self.container_number[4:]
        self.container_id = (
            f"{self.owner_code} "
            f"{self.serial_number[:6]} "
            f"{self.serial_number[6]}"
        )
        if self.container_type:
            self.container_type_code = self.container_type

    def to_dict(self) -> dict:
        """Return the result as a JSON-serialisable dict matching the target schema."""
        self._derive_fields()
        d: dict = {}
        if self.error:
            d["error"] = self.error
            return d
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
        return d


def _find_matching_carrier_prefix(text: str) -> Optional[str]:
    """Match first 4 chars against carrier prefixes, allowing OCR confusions."""
    candidate = text[:4]
    if candidate in CARRIER_PREFIX_SET:
        return candidate
    matches = PREFIX_FUZZY_MAP.get(candidate)
    if matches:
        return matches[0]
    return None


class MergedOCRLine:
    """Synthetic line produced by merging vertically-overlapping OCR lines."""
    __slots__ = ("words", "bounding_box")

    def __init__(self, words: list, bounding_box: str = "") -> None:
        self.words = words
        self.bounding_box = bounding_box


def _get_line_vertical_bounds(line) -> Tuple[float, float]:
    """Return the (min_y, max_y) vertical extent of a line's words."""
    y_vals: List[float] = []
    for w in line.words:
        vals = [float(v) for v in w.bounding_box.split(",")] if isinstance(w.bounding_box, str) else list(w.bounding_box)
        if len(vals) == 4:
            y_vals.extend([vals[1], vals[1] + vals[3]])
        else:
            y_vals.extend(vals[1::2])
    return min(y_vals), max(y_vals)


def _merge_vertically_overlapping_lines(lines) -> list:
    """Merge OCR lines that share vertical overlap."""
    if not lines:
        return []

    merged = []
    current_words = list(lines[0].words)
    current_y_min, current_y_max = _get_line_vertical_bounds(lines[0])

    for line in lines[1:]:
        y_min, y_max = _get_line_vertical_bounds(line)
        if y_min < current_y_max and y_max > current_y_min:
            current_words.extend(line.words)
            current_y_min = min(current_y_min, y_min)
            current_y_max = max(current_y_max, y_max)
        else:
            merged.append(MergedOCRLine(current_words))
            current_words = list(line.words)
            current_y_min = y_min
            current_y_max = y_max

    merged.append(MergedOCRLine(current_words))
    return merged


def _extract_container_from_words(words, result: ContainerResult, type_regex: str) -> bool:
    """Extract container number and type from a flat list of words using regex. Returns True if both found."""
    prefix_pattern = "|".join(CARRIER_PREFIXES)
    regex_pattern = f"({prefix_pattern})(\\d{{7}})"

    word_spans: List[Tuple[int, int]] = []
    line_text = ""
    for w in words:
        wt = str(w.text).strip().replace(" ", "").upper()
        start = len(line_text)
        line_text += wt
        word_spans.append((start, len(line_text)))

    # Exact regex match
    if not result.container_number:
        for m in re.finditer(regex_pattern, line_text):
            candidate = m.group(0)
            if not validate_iso6346_check_digit(candidate):
                log.debug("Regex match '%s' failed ISO 6346 check digit validation", candidate)
                continue
            result.container_number = candidate
            match_start, match_end = m.start(), m.end()
            bb: List[int] = [0, 0, 0, 0]
            bb_set = False
            for idx, w in enumerate(words):
                ws, we = word_spans[idx]
                if we <= match_start or ws >= match_end:
                    continue
                wx1, wy1, wx2, wy2 = parse_word_bounding_box(w)
                if not bb_set:
                    bb = [wx1, wy1, wx2, wy2]
                    bb_set = True
                else:
                    bb[2] = max(bb[2], wx2)
                    bb[3] = max(bb[3], wy2)
            result.bounding_box = bb
            log.info("Regex extraction found container number: %s", candidate)
            break

    # Fuzzy prefix fallback
    if not result.container_number:
        for idx, w in enumerate(words):
            wt = str(w.text).strip().replace(" ", "").upper()
            matched_prefix = _find_matching_carrier_prefix(wt)
            if matched_prefix is None:
                continue
            serial_digits = ""
            wx1, wy1, wx2, wy2 = parse_word_bounding_box(w)
            bb = [wx1, wy1, wx2, wy2]
            for w2 in words[idx + 1:]:
                w2t = str(w2.text).strip().replace(" ", "").upper()
                digits_only = "".join(ch for ch in w2t if ch.isdigit())
                if not digits_only:
                    break
                serial_digits += digits_only
                _, _, w2x2, w2y2 = parse_word_bounding_box(w2)
                bb[2] = max(bb[2], w2x2)
                bb[3] = max(bb[3], w2y2)
                if len(serial_digits) >= 7:
                    break
            if len(serial_digits) >= 7:
                candidate = matched_prefix + serial_digits[:7]
                if validate_iso6346_check_digit(candidate):
                    result.container_number = candidate
                    result.bounding_box = bb
                    log.info("Fuzzy prefix extraction found container number: %s", candidate)
                    break
                else:
                    log.debug("Fuzzy match '%s' failed ISO 6346 check digit validation", candidate)

    # Container type
    if not result.container_type:
        mt = re.search(type_regex, line_text)
        if mt:
            result.container_type = mt.group(0)
            log.debug("Found container type: %s", result.container_type)

    return bool(result.container_number) and bool(result.container_type)


def extract_container_regex(ocr_result) -> ContainerResult:
    """Regex-based extraction with ISO 6346 check-digit validation and fuzzy prefix fallback."""
    result = ContainerResult()

    type_pattern = "|".join(CONTAINER_TYPE_PREFIXES)
    type_regex = f"\\d{{2}}({type_pattern})"

    for region in ocr_result.regions:
        if not region.lines:
            continue

        merged_lines = _merge_vertically_overlapping_lines(region.lines)

        for line in merged_lines:
            words = line.words
            if not words:
                continue

            done = _extract_container_from_words(words, result, type_regex)
            if done:
                break
        if result.container_number and result.container_type:
            break

    if result.container_number:
        log.debug("Regex extraction result: %s type=%s", result.container_number, result.container_type or "not found")
    else:
        log.debug("Regex extraction found no container number")

    return result


def extract_container_location(ocr_result) -> ContainerResult:
    """Location-based extraction using spatial adjacency of OCR words."""
    result = ContainerResult()
    bound_block = [0, 0, 0, 0]
    orientation_horizontal = True
    last_xy: List[int] = []
    allowable_buffer = SPATIAL_BUFFER

    type_regex = f"\\d{{2}}({'|'.join(CONTAINER_TYPE_PREFIXES)})"

    for region in ocr_result.regions:
        if not region.lines:
            continue
        for line in region.lines:
            for word in line.words:
                if len(result.container_number) >= CONTAINER_NUMBER_LENGTH and result.container_type:
                    break

                clean_text = str(word.text).strip().replace(" ", "").upper()
                x1, y1, x2, y2 = parse_word_bounding_box(word)
                bbox8 = [x1, y1, x2, y1, x2, y2, x1, y2]

                if not result.container_number:
                    matched_prefix = _find_matching_carrier_prefix(clean_text)
                    if matched_prefix:
                        result.container_number = matched_prefix
                        orientation_horizontal = is_bounding_box_horizontal(bbox8)
                        last_xy = [x2, y1] if orientation_horizontal else [x1, y2]
                        allowable_buffer = get_bounding_box_half_min_extent(bbox8) * 3 or SPATIAL_BUFFER
                        bound_block = [x1, y1, x2, y2]

                elif 4 <= len(result.container_number) < CONTAINER_NUMBER_LENGTH:
                    if orientation_horizontal:
                        crit_met = (x1 >= last_xy[0] and
                                    are_coordinates_within_distance(last_xy[1], y1, allowable_buffer))
                    else:
                        crit_met = (y1 >= last_xy[1] and
                                    are_coordinates_within_distance(last_xy[0], x1, allowable_buffer))

                    if crit_met:
                        result.container_number += clean_text
                        last_xy = [x2, y1] if orientation_horizontal else [x1, y2]
                        bound_block[2] = max(bound_block[2], x2)
                        bound_block[3] = max(bound_block[3], y2)

                if not result.container_type and re.search(type_regex, clean_text):
                    result.container_type = clean_text

                allowable_buffer = get_bounding_box_half_min_extent(bbox8) * 3 or allowable_buffer

    result.bounding_box = bound_block

    if result.container_number:
        log.debug("Location extraction result: %s type=%s", result.container_number, result.container_type or "not found")
    else:
        log.debug("Location extraction found no container number")

    return result


def run_extraction_pipeline(ocr_result, image_bytes: bytes) -> ContainerResult:
    """Run the full pipeline: regex extraction → location fallback → color detection."""
    result = extract_container_regex(ocr_result)

    if not result.container_number or not result.container_type:
        log.info("Regex extraction incomplete (number=%s, type=%s), falling back to location extraction",
                 result.container_number or "none", result.container_type or "none")
        loc = extract_container_location(ocr_result)
        if not result.container_number:
            result.container_number = loc.container_number
            result.bounding_box = loc.bounding_box
        if not result.container_type:
            result.container_type = loc.container_type

    if result.bounding_box != [0, 0, 0, 0]:
        try:
            result.container_color = extract_dominant_color_from_image_bytes(image_bytes, result.bounding_box)
        except ValueError as e:
            log.warning("Could not extract container color: %s", e)
            result.container_color = [0, 0, 0]

    if not result.container_number:
        log.info("No container number found in image")
    else:
        log.info("Extracted container: %s, type: %s", result.container_number, result.container_type or "unknown")

    return result

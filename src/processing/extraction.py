"""Container number extraction logic."""

import logging
import re
from typing import List, Tuple, Optional
from .models import ContainerResult, MergedOCRLine
from ..utils.config import (
    CARRIER_PREFIXES, CARRIER_PREFIX_SET, CONTAINER_TYPE_PREFIXES,
    SPATIAL_BUFFER, CONTAINER_NUMBER_LENGTH, PREFIX_FUZZY_MAP,
)
from ..utils.geometry import (
    parse_word_bounding_box, is_bounding_box_horizontal,
    get_bounding_box_half_min_extent, are_coordinates_within_distance,
)
from ..utils.validation import validate_iso6346_check_digit

log = logging.getLogger(__name__)

_PREFIX_PATTERN = None
_TYPE_PATTERN_CACHE = {}
_LOCATION_TYPE_REGEX = None


def _get_prefix_pattern():
    global _PREFIX_PATTERN
    if _PREFIX_PATTERN is None:
        prefix_pattern = "|".join(CARRIER_PREFIXES)
        _PREFIX_PATTERN = re.compile(f"({prefix_pattern})(\\d{{7}})")
    return _PREFIX_PATTERN


def _get_type_pattern(type_regex: str):
    if type_regex not in _TYPE_PATTERN_CACHE:
        _TYPE_PATTERN_CACHE[type_regex] = re.compile(type_regex)
    return _TYPE_PATTERN_CACHE[type_regex]


def _get_location_type_regex():
    global _LOCATION_TYPE_REGEX
    if _LOCATION_TYPE_REGEX is None:
        _LOCATION_TYPE_REGEX = re.compile(f"\\d{{2}}({'|'.join(CONTAINER_TYPE_PREFIXES)})")
    return _LOCATION_TYPE_REGEX


def _find_matching_carrier_prefix(text: str) -> Optional[str]:
    """Match first 4 chars against carrier prefixes, allowing OCR confusions."""
    candidate = text[:4]
    if candidate in CARRIER_PREFIX_SET:
        return candidate
    matches = PREFIX_FUZZY_MAP.get(candidate)
    if matches:
        return matches[0]
    return None


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
    cleaned_words = [str(w.text).strip().replace(" ", "").upper() for w in words]

    word_spans: List[Tuple[int, int]] = []
    line_text = ""
    for wt in cleaned_words:
        start = len(line_text)
        line_text += wt
        word_spans.append((start, len(line_text)))

    prefix_pattern = _get_prefix_pattern()
    type_pattern = _get_type_pattern(type_regex)

    if not result.container_number:
        for m in prefix_pattern.finditer(line_text):
            candidate = m.group(0)
            if not validate_iso6346_check_digit(candidate):
                continue
            result.container_number = candidate
            match_start, match_end = m.start(), m.end()
            bb: List[int] = [0, 0, 0, 0]
            bb_set = False
            for idx, (ws, we) in enumerate(word_spans):
                if we <= match_start or ws >= match_end:
                    continue
                wx1, wy1, wx2, wy2 = parse_word_bounding_box(words[idx])
                if not bb_set:
                    bb = [wx1, wy1, wx2, wy2]
                    bb_set = True
                else:
                    bb[2] = max(bb[2], wx2)
                    bb[3] = max(bb[3], wy2)
            result.bounding_box = bb
            break

    if not result.container_number:
        for idx, wt in enumerate(cleaned_words):
            matched_prefix = _find_matching_carrier_prefix(wt)
            if matched_prefix is None:
                continue
            serial_digits = ""
            wx1, wy1, wx2, wy2 = parse_word_bounding_box(words[idx])
            bb = [wx1, wy1, wx2, wy2]
            for j in range(idx + 1, len(cleaned_words)):
                w2t = cleaned_words[j]
                digits_only = "".join(ch for ch in w2t if ch.isdigit())
                if not digits_only:
                    break
                serial_digits += digits_only
                _, _, w2x2, w2y2 = parse_word_bounding_box(words[j])
                bb[2] = max(bb[2], w2x2)
                bb[3] = max(bb[3], w2y2)
                if len(serial_digits) >= 7:
                    break
            if len(serial_digits) >= 7:
                candidate = matched_prefix + serial_digits[:7]
                if validate_iso6346_check_digit(candidate):
                    result.container_number = candidate
                    result.bounding_box = bb
                    break

    if not result.container_type:
        mt = type_pattern.search(line_text)
        if mt:
            result.container_type = mt.group(0)

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

    return result


def extract_container_location(ocr_result) -> ContainerResult:
    """Location-based extraction using spatial adjacency of OCR words."""
    result = ContainerResult()
    bound_block = [0, 0, 0, 0]
    orientation_horizontal = True
    last_xy: List[int] = []
    allowable_buffer = SPATIAL_BUFFER

    location_type_regex = _get_location_type_regex()

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
                        crit_met = (x1 >= last_xy[0] and are_coordinates_within_distance(last_xy[1], y1, allowable_buffer))
                    else:
                        crit_met = (y1 >= last_xy[1] and are_coordinates_within_distance(last_xy[0], x1, allowable_buffer))

                    if crit_met:
                        result.container_number += clean_text
                        last_xy = [x2, y1] if orientation_horizontal else [x1, y2]
                        bound_block[2] = max(bound_block[2], x2)
                        bound_block[3] = max(bound_block[3], y2)

                if not result.container_type and location_type_regex.search(clean_text):
                    result.container_type = clean_text

                allowable_buffer = get_bounding_box_half_min_extent(bbox8) * 3 or allowable_buffer

    result.bounding_box = bound_block
    return result


def run_extraction_pipeline(ocr_result) -> ContainerResult:
    """Run the full pipeline: regex extraction -> location fallback.
    """
    result = extract_container_regex(ocr_result)

    if not result.container_number or not result.container_type:
        loc = extract_container_location(ocr_result)
        if not result.container_number:
            result.container_number = loc.container_number
            result.bounding_box = loc.bounding_box
        if not result.container_type:
            result.container_type = loc.container_type

    if not result.container_number:
        log.info("No container number found in image")
    else:
        log.info("Extracted container: %s, type: %s", result.container_number, result.container_type or "unknown")

    return result
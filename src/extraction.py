"""Container number extraction logic."""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from .config import CARRIER_PREFIXES, CARRIER_PREFIX_SET, CONTAINER_TYPE_PREFIXES, SPATIAL_BUFFER, CONTAINER_NUMBER_LENGTH
from .preprocessing import get_container_color_from_bytes
from .validation import validate_iso6346_check_digit


@dataclass
class ContainerResult:
    """Result of container detection."""
    container_number: str = ""
    container_type: str = ""
    bounding_box: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    container_color: List[int] = field(default_factory=lambda: [0, 0, 0])
    error: Optional[str] = None


# OCR character confusions for fuzzy prefix matching (e.g. O↔D, I↔1).
_OCR_CHAR_CONFUSIONS = [
    ("O", ["D", "0"]),
    ("D", ["O"]),
    ("0", ["O"]),
    ("I", ["1"]),
    ("1", ["I"]),
    ("S", ["5"]),
    ("5", ["S"]),
    ("B", ["8"]),
    ("8", ["B"]),
    ("G", ["6"]),
    ("6", ["G"]),
    ("Z", ["2"]),
    ("2", ["Z"]),
    ("T", ["7"]),
    ("7", ["T"]),
]

_CHAR_ALTERNATIVES: dict = {}
for _correct, _confused_list in _OCR_CHAR_CONFUSIONS:
    for _alt in _confused_list:
        _CHAR_ALTERNATIVES.setdefault(_correct, set()).add(_alt)
        _CHAR_ALTERNATIVES.setdefault(_alt, set()).add(_correct)

_PREFIX_FUZZY_MAP: dict = {}
for _prefix in CARRIER_PREFIXES:
    for _pos in range(len(_prefix)):
        _orig_ch = _prefix[_pos]
        for _alt in _CHAR_ALTERNATIVES.get(_orig_ch, ()):
            _confused = list(_prefix)
            _confused[_pos] = _alt
            _key = "".join(_confused)
            if _key != _prefix:
                _PREFIX_FUZZY_MAP.setdefault(_key, []).append(_prefix)


def _fuzzy_prefix_match(text: str) -> Optional[str]:
    """Match first 4 chars against carrier prefixes, allowing OCR confusions."""
    candidate = text[:4]
    if candidate in CARRIER_PREFIX_SET:
        return candidate
    matches = _PREFIX_FUZZY_MAP.get(candidate)
    if matches:
        return matches[0]
    return None


class _MergedLine:
    """Synthetic line produced by merging vertically-overlapping OCR lines."""
    __slots__ = ("words", "bounding_box")

    def __init__(self, words: list, bounding_box: str = "") -> None:
        self.words = words
        self.bounding_box = bounding_box


def _get_line_y_range(line) -> Tuple[float, float]:
    """Return the (min_y, max_y) vertical extent of a line's words."""
    y_vals: List[float] = []
    for w in line.words:
        vals = [float(v) for v in w.bounding_box.split(",")] if isinstance(w.bounding_box, str) else list(w.bounding_box)
        if len(vals) == 4:
            y_vals.extend([vals[1], vals[1] + vals[3]])
        else:
            y_vals.extend(vals[1::2])
    return min(y_vals), max(y_vals)


def _merge_overlapping_lines(lines) -> list:
    """Merge OCR lines that share vertical overlap."""
    if not lines:
        return []

    merged = []
    current_words = list(lines[0].words)
    current_y_min, current_y_max = _get_line_y_range(lines[0])

    for line in lines[1:]:
        y_min, y_max = _get_line_y_range(line)
        if y_min < current_y_max and y_max > current_y_min:
            current_words.extend(line.words)
            current_y_min = min(current_y_min, y_min)
            current_y_max = max(current_y_max, y_max)
        else:
            merged.append(_MergedLine(current_words))
            current_words = list(line.words)
            current_y_min = y_min
            current_y_max = y_max

    merged.append(_MergedLine(current_words))
    return merged


def _parse_word_bbox(word) -> Tuple[int, int, int, int]:
    """Parse word bounding box into (x1, y1, x2, y2)."""
    bbox_str = word.bounding_box
    vals = [float(v) for v in bbox_str.split(",")] if isinstance(bbox_str, str) else list(bbox_str)
    if len(vals) == 4:
        x1, y1, w, h = vals
        return int(x1), int(y1), int(x1 + w), int(y1 + h)
    return int(vals[0]), int(vals[1]), int(vals[4]), int(vals[5])


def _check_orientation_horizontal(bbox: List[int]) -> bool:
    return (bbox[2] - bbox[0]) > (bbox[5] - bbox[1])


def _get_label_angle(bbox: List[int]) -> int:
    if _check_orientation_horizontal(bbox):
        return (bbox[5] - bbox[1]) // 2
    return (bbox[2] - bbox[0]) // 2


def _within_buffer(co1: int, co2: int, buffer: int = SPATIAL_BUFFER) -> bool:
    return co1 - buffer < co2 < co1 + buffer


def _try_regex_on_words(words, result: ContainerResult, type_regex: str) -> bool:
    """Try regex-based extraction on a flat list of words. Returns True if both fields found."""
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
                continue
            result.container_number = candidate
            match_start, match_end = m.start(), m.end()
            bb: List[int] = [0, 0, 0, 0]
            bb_set = False
            for idx, w in enumerate(words):
                ws, we = word_spans[idx]
                if we <= match_start or ws >= match_end:
                    continue
                wx1, wy1, wx2, wy2 = _parse_word_bbox(w)
                if not bb_set:
                    bb = [wx1, wy1, wx2, wy2]
                    bb_set = True
                else:
                    bb[2] = max(bb[2], wx2)
                    bb[3] = max(bb[3], wy2)
            result.bounding_box = bb
            break

    # Fuzzy prefix fallback
    if not result.container_number:
        for idx, w in enumerate(words):
            wt = str(w.text).strip().replace(" ", "").upper()
            matched_prefix = _fuzzy_prefix_match(wt)
            if matched_prefix is None:
                continue
            serial_digits = ""
            wx1, wy1, wx2, wy2 = _parse_word_bbox(w)
            bb = [wx1, wy1, wx2, wy2]
            for w2 in words[idx + 1:]:
                w2t = str(w2.text).strip().replace(" ", "").upper()
                digits_only = "".join(ch for ch in w2t if ch.isdigit())
                if not digits_only:
                    break
                serial_digits += digits_only
                _, _, w2x2, w2y2 = _parse_word_bbox(w2)
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

    # Container type
    if not result.container_type:
        mt = re.search(type_regex, line_text)
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

        merged_lines = _merge_overlapping_lines(region.lines)

        for line in merged_lines:
            words = line.words
            if not words:
                continue

            done = _try_regex_on_words(words, result, type_regex)
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

    type_regex = f"\\d{{2}}({'|'.join(CONTAINER_TYPE_PREFIXES)})"

    for region in ocr_result.regions:
        if not region.lines:
            continue
        for line in region.lines:
            for word in line.words:
                if len(result.container_number) >= CONTAINER_NUMBER_LENGTH and result.container_type:
                    break

                clean_text = str(word.text).strip().replace(" ", "").upper()
                x1, y1, x2, y2 = _parse_word_bbox(word)
                bbox8 = [x1, y1, x2, y1, x2, y2, x1, y2]

                if not result.container_number:
                    matched_prefix = _fuzzy_prefix_match(clean_text)
                    if matched_prefix:
                        result.container_number = matched_prefix
                        orientation_horizontal = _check_orientation_horizontal(bbox8)
                        last_xy = [x2, y1] if orientation_horizontal else [x1, y2]
                        allowable_buffer = _get_label_angle(bbox8) * 3 or SPATIAL_BUFFER
                        bound_block = [x1, y1, x2, y2]

                elif CONTAINER_NUMBER_LENGTH > len(result.container_number) >= 4:
                    if orientation_horizontal:
                        crit_met = (x1 >= last_xy[0] and
                                    _within_buffer(last_xy[1], y1, allowable_buffer))
                    else:
                        crit_met = (y1 >= last_xy[1] and
                                    _within_buffer(last_xy[0], x1, allowable_buffer))

                    if crit_met:
                        result.container_number += clean_text
                        last_xy = [x2, y1] if orientation_horizontal else [x1, y2]
                        bound_block[2] = max(bound_block[2], x2)
                        bound_block[3] = max(bound_block[3], y2)

                if not result.container_type and re.search(type_regex, clean_text):
                    result.container_type = clean_text

                allowable_buffer = _get_label_angle(bbox8) * 3 or allowable_buffer

    result.bounding_box = bound_block
    return result


def run_extraction_pipeline(ocr_result, image_bytes: bytes) -> ContainerResult:
    """Run the full pipeline: regex extraction → location fallback → color detection."""
    result = extract_container_regex(ocr_result)

    if not result.container_number or not result.container_type:
        loc = extract_container_location(ocr_result)
        if not result.container_number:
            result.container_number = loc.container_number
            result.bounding_box = loc.bounding_box
        if not result.container_type:
            result.container_type = loc.container_type

    if result.bounding_box != [0, 0, 0, 0]:
        result.container_color = get_container_color_from_bytes(image_bytes, result.bounding_box)

    return result

"""Container number extraction logic."""

import re
from typing import List, Tuple, Optional, FrozenSet
from .config import CARRIER_PREFIXES, CARRIER_PREFIX_SET, CONTAINER_TYPE_PREFIXES, SPATIAL_BUFFER, CONTAINER_NUMBER_LENGTH
from .models import ContainerResult
from .validation import validate_iso6346_check_digit

# Common OCR character confusions that affect carrier prefix matching.
# Each tuple is (correct_char, [list of characters OCR might confuse it with]).
# For each known prefix we generate variants by replacing each character with
# every commonly-confused alternative.
_OCR_CHAR_CONFUSIONS = [
    ("O", ["D", "0"]),   # O <-> D (primary issue in 1.png) and O <-> 0
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

# Reverse index: for a given character, what alternatives might OCR show?
# e.g. _CHAR_ALTERNATIVES['O'] = {'D', '0'}
_CHAR_ALTERNATIVES: dict = {}
for _correct, _confused_list in _OCR_CHAR_CONFUSIONS:
    for _alt in _confused_list:
        _CHAR_ALTERNATIVES.setdefault(_correct, set()).add(_alt)
        _CHAR_ALTERNATIVES.setdefault(_alt, set()).add(_correct)

# Build a lookup: OCR-seen prefix → canonical prefix(s) from the known set.
# This is computed once at import time.
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
    """
    Try to match the first 4 characters of *text* against carrier prefixes,
    allowing for common OCR character confusions.

    Returns the canonical prefix string (e.g. ``"OOLU"``) if a match is
    found, otherwise ``None``.
    """
    candidate = text[:4]
    if candidate in CARRIER_PREFIX_SET:
        return candidate
    matches = _PREFIX_FUZZY_MAP.get(candidate)
    if matches:
        return matches[0]
    return None


def _get_line_y_range(line) -> Tuple[float, float]:
    """Return the (min_y, max_y) vertical extent of a line's words."""
    y_vals: List[float] = []
    for w in line.words:
        vals = [float(v) for v in w.bounding_box.split(",")] if isinstance(w.bounding_box, str) else list(w.bounding_box)
        y_vals.extend([vals[1], vals[1] + vals[3]])
    return min(y_vals), max(y_vals)


def _merge_overlapping_lines(lines) -> list:
    """
    Merge OCR lines that share vertical overlap.

    Azure's Read API sometimes splits a single visual line of text into
    multiple line groups (e.g. when a word is offset vertically).
    This function groups lines whose y-ranges overlap and concatenates
    their words so downstream regex matching can see the full line.
    """
    if not lines:
        return []

    merged = []
    current_words = list(lines[0].words)
    current_y_min, current_y_max = _get_line_y_range(lines[0])

    for line in lines[1:]:
        y_min, y_max = _get_line_y_range(line)
        if y_min < current_y_max and y_max > current_y_min:
            # Overlapping – merge words from this line
            current_words.extend(line.words)
            current_y_min = min(current_y_min, y_min)
            current_y_max = max(current_y_max, y_max)
        else:
            # No overlap – flush current group
            merged.append(type(lines[0])(current_words, ""))
            current_words = list(line.words)
            current_y_min = y_min
            current_y_max = y_max

    merged.append(type(lines[0])(current_words, ""))
    return merged


def _parse_word_bbox(word) -> Tuple[int, int, int, int]:
    """
    Parse word bounding box.

    Args:
        word: OCR word object

    Returns:
        (x1, y1, x2, y2)
    """
    bbox_str = word.bounding_box
    vals = [float(v) for v in bbox_str.split(",")] if isinstance(bbox_str, str) else list(bbox_str)
    if len(vals) == 4:
        x1, y1, w, h = vals
        return int(x1), int(y1), int(x1 + w), int(y1 + h)
    # 8-point polygon
    return int(vals[0]), int(vals[1]), int(vals[4]), int(vals[5])


def _check_orientation_horizontal(bbox: List[int]) -> bool:
    """Check if text is horizontal."""
    width = bbox[2] - bbox[0]
    height = bbox[5] - bbox[1]
    return width > height


def _get_label_angle(bbox: List[int]) -> int:
    """Get buffer for adjacency checks."""
    is_horizontal = _check_orientation_horizontal(bbox)
    if is_horizontal:
        return (bbox[5] - bbox[1]) // 2
    return (bbox[2] - bbox[0]) // 2


def _within_buffer(co1: int, co2: int, buffer: int = SPATIAL_BUFFER) -> bool:
    """Check if co2 is within buffer of co1."""
    return co1 - buffer < co2 < co1 + buffer


def _try_regex_on_words(words, result: ContainerResult, type_regex: str) -> bool:
    """
    Try regex-based extraction on a flat list of words.

    Builds a concatenated text from words, then applies the carrier-prefix
    regex and the container-type regex.  Uses fuzzy prefix matching as a
    fallback when exact regex matching fails (handles common OCR misreads
    like O/D).

    Returns True if both container number and type were found.
    """
    # Carrier-prefix regex (exact match)
    prefix_pattern = "|".join(CARRIER_PREFIXES)
    regex_pattern = f"({prefix_pattern})(\\d{{7}})"

    word_spans: List[Tuple[int, int]] = []
    line_text = ""
    for w in words:
        wt = str(w.text).strip().replace(" ", "").upper()
        start = len(line_text)
        line_text += wt
        word_spans.append((start, len(line_text)))

    # --- Container number (exact regex) ---
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

    # --- Container number (fuzzy prefix fallback) ---
    if not result.container_number:
        for idx, w in enumerate(words):
            wt = str(w.text).strip().replace(" ", "").upper()
            matched_prefix = _fuzzy_prefix_match(wt)
            if matched_prefix is None:
                continue
            # Collect digits from subsequent adjacent words
            serial_digits = ""
            bb: List[int] = [0, 0, 0, 0]
            bb_set = False
            wx1, wy1, wx2, wy2 = _parse_word_bbox(w)
            bb = [wx1, wy1, wx2, wy2]
            bb_set = True
            for w2 in words[idx + 1:]:
                w2t = str(w2.text).strip().replace(" ", "").upper()
                digits_only = "".join(ch for ch in w2t if ch.isdigit())
                if not digits_only:
                    break
                serial_digits += digits_only
                w2x1, w2y1, w2x2, w2y2 = _parse_word_bbox(w2)
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

    # --- Container type ---
    if not result.container_type:
        mt = re.search(type_regex, line_text)
        if mt:
            result.container_type = mt.group(0)

    return bool(result.container_number) and bool(result.container_type)


def extract_container_regex(ocr_result) -> ContainerResult:
    """
    Regex-based container extraction.

    Processes each OCR line independently to avoid cross-region false matches.
    Uses ``re.finditer`` so every candidate on a line is tested against the
    ISO 6346 check-digit rule before being accepted.  The regex enforces
    exactly 11 characters (4-letter owner code + 7 digits) to match the
    ISO 6346 standard.

    Lines that share vertical overlap are merged before matching so that
    container numbers split across OCR line groups (a common artefact) are
    still detected.

    A fuzzy-prefix fallback handles common OCR misreads (e.g. O/D) when
    the exact regex match fails.

    Args:
        ocr_result: Azure OCR result

    Returns:
        ContainerResult
    """
    result = ContainerResult()

    type_pattern = "|".join(CONTAINER_TYPE_PREFIXES)
    type_regex = f"\\d{{2}}({type_pattern})"

    for region in ocr_result.regions:
        if not region.lines:
            continue

        # Merge vertically-overlapping lines so prefix + serial are visible
        # to the regex even when the OCR groups them separately.
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
    """
    Location-based container extraction using spatial adjacency of OCR words.

    Uses ``CARRIER_PREFIX_SET`` (frozenset) for O(1) prefix lookup.
    The reference coordinate ``last_xy`` is set according to the detected
    text orientation:
      - Horizontal text → [x2, y1] (next word must start to the right)
      - Vertical text   → [x1, y2] (next word must start below)

    Args:
        ocr_result: Azure OCR result

    Returns:
        ContainerResult
    """
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

                # Container prefix – O(1) lookup via frozenset, with fuzzy
                # fallback for common OCR misreads (e.g. DOLU → OOLU).
                if not result.container_number:
                    matched_prefix = _fuzzy_prefix_match(clean_text)
                    if matched_prefix:
                        result.container_number = matched_prefix
                        orientation_horizontal = _check_orientation_horizontal(bbox8)
                        # Reference coordinate depends on orientation so that the
                        # adjacency test for the next word is directionally correct.
                        last_xy = [x2, y1] if orientation_horizontal else [x1, y2]
                        allowable_buffer = _get_label_angle(bbox8) * 3 or SPATIAL_BUFFER
                        bound_block = [x1, y1, x2, y2]

                # Container serial
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

                # Container type
                if not result.container_type and re.search(type_regex, clean_text):
                    result.container_type = clean_text

                allowable_buffer = _get_label_angle(bbox8) * 3 or allowable_buffer

    result.bounding_box = bound_block
    return result
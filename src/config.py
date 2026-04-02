"""Configuration constants for container number recognition."""

import logging
import os

log = logging.getLogger(__name__)

CONTAINER_NUMBER_LENGTH = 11
DOWNSCALE_THRESHOLD = 4000
CROP_PADDING = 100
SPATIAL_BUFFER = 50

DATA_DIR = "./data"
PREFIX_FILE = "./container_prefix.txt"

VISION_ENDPOINT = os.getenv("VISION_ENDPOINT", "")
VISION_KEY = os.getenv("VISION_KEY", "")


def _load_carrier_prefixes_from_file():
    """Load carrier prefixes from file."""
    if not os.path.exists(PREFIX_FILE):
        log.warning("Prefix file '%s' not found — carrier prefix matching disabled", PREFIX_FILE)
        return ()
    with open(PREFIX_FILE, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
        if lines:
            prefixes = tuple(lines[0].split(","))
            log.info("Loaded %d carrier prefixes from '%s'", len(prefixes), PREFIX_FILE)
            return prefixes
    log.warning("Prefix file '%s' is empty — carrier prefix matching disabled", PREFIX_FILE)
    return ()


CARRIER_PREFIXES = _load_carrier_prefixes_from_file()
CARRIER_PREFIX_SET = frozenset(CARRIER_PREFIXES)
CONTAINER_TYPE_PREFIXES = ("G1", "R1", "U1", "P1", "T1")

# OCR character confusions for fuzzy prefix matching (e.g. O↔D, I↔1).
OCR_CHAR_CONFUSIONS = [
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

CHAR_ALTERNATIVES: dict = {}
for _correct, _confused_list in OCR_CHAR_CONFUSIONS:
    for _alt in _confused_list:
        CHAR_ALTERNATIVES.setdefault(_correct, set()).add(_alt)
        CHAR_ALTERNATIVES.setdefault(_alt, set()).add(_correct)

PREFIX_FUZZY_MAP: dict = {}
for _prefix in CARRIER_PREFIXES:
    for _pos in range(len(_prefix)):
        _orig_ch = _prefix[_pos]
        for _alt in CHAR_ALTERNATIVES.get(_orig_ch, ()):
            _confused = list(_prefix)
            _confused[_pos] = _alt
            _key = "".join(_confused)
            if _key != _prefix:
                PREFIX_FUZZY_MAP.setdefault(_key, []).append(_prefix)

log.info("Built fuzzy prefix map with %d entries from %d prefixes", len(PREFIX_FUZZY_MAP), len(CARRIER_PREFIXES))

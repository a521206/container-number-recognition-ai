"""Configuration constants for container number recognition."""

import os
from typing import FrozenSet, Tuple

# Container standards
CONTAINER_NUMBER_LENGTH = 11

# Image processing
DOWNSCALE_THRESHOLD = 4000
CROP_PADDING = 100
SPATIAL_BUFFER = 50

# File paths
DATA_DIR = "./data"
PREFIX_FILE = "./container_prefix.txt"

# Azure Vision
VISION_ENDPOINT = os.getenv("VISION_ENDPOINT", "")
VISION_KEY = os.getenv("VISION_KEY", "")

# Validation
ISO6346_CHECK_DIGIT_MULTIPLIERS = tuple(2 ** i for i in range(10))

# OCR and extraction
def _load_carrier_prefixes() -> Tuple[str, ...]:
    """Load carrier prefixes from file."""
    if not os.path.exists(PREFIX_FILE):
        return ()
    with open(PREFIX_FILE, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
        if lines:
            return tuple(lines[0].split(","))
    return ()

CARRIER_PREFIXES: Tuple[str, ...] = _load_carrier_prefixes()
# Set for O(1) membership tests in hot extraction loops
CARRIER_PREFIX_SET: FrozenSet[str] = frozenset(CARRIER_PREFIXES)

CONTAINER_TYPE_PREFIXES: Tuple[str, ...] = ("G1", "R1", "U1", "P1", "T1")
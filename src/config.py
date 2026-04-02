"""Configuration constants for container number recognition."""

import os

CONTAINER_NUMBER_LENGTH = 11
DOWNSCALE_THRESHOLD = 4000
CROP_PADDING = 100
SPATIAL_BUFFER = 50

DATA_DIR = "./data"
PREFIX_FILE = "./container_prefix.txt"

VISION_ENDPOINT = os.getenv("VISION_ENDPOINT", "")
VISION_KEY = os.getenv("VISION_KEY", "")


def _load_carrier_prefixes():
    """Load carrier prefixes from file."""
    if not os.path.exists(PREFIX_FILE):
        return ()
    with open(PREFIX_FILE, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
        if lines:
            return tuple(lines[0].split(","))
    return ()


CARRIER_PREFIXES = _load_carrier_prefixes()
CARRIER_PREFIX_SET = frozenset(CARRIER_PREFIXES)
CONTAINER_TYPE_PREFIXES = ("G1", "R1", "U1", "P1", "T1")

"""Validation functions for container numbers."""

from .config import ISO6346_CHECK_DIGIT_MULTIPLIERS

# ISO 6346 check-digit validation
_ISO6346_LETTER_VALUES = {
    'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17, 'H': 18,
    'I': 19, 'J': 20, 'K': 21, 'L': 23, 'M': 24, 'N': 25, 'O': 26, 'P': 27,
    'Q': 28, 'R': 29, 'S': 30, 'T': 31, 'U': 32, 'V': 34, 'W': 35, 'X': 36,
    'Y': 37, 'Z': 38,
}


def validate_iso6346_check_digit(container_number: str) -> bool:
    """
    Validate the ISO 6346 check digit for a container number.

    Args:
        container_number: 11-character string, e.g. "MSCU1234567"

    Returns:
        True if the check digit matches, False otherwise
    """
    if len(container_number) != 11:
        return False

    total = 0
    for i, ch in enumerate(container_number[:10]):
        if ch.isalpha():
            val = _ISO6346_LETTER_VALUES.get(ch.upper(), -1)
            if val < 0:
                return False
        elif ch.isdigit():
            val = int(ch)
        else:
            return False
        total += val * ISO6346_CHECK_DIGIT_MULTIPLIERS[i]

    remainder = total % 11
    expected_check = 0 if remainder == 10 else remainder

    try:
        return int(container_number[10]) == expected_check
    except ValueError:
        return False
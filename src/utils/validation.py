"""Validation functions for container numbers."""

from typing import Tuple

_ISO6346_LETTER_VALUES = {
    'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17, 'H': 18,
    'I': 19, 'J': 20, 'K': 21, 'L': 23, 'M': 24, 'N': 25, 'O': 26, 'P': 27,
    'Q': 28, 'R': 29, 'S': 30, 'T': 31, 'U': 32, 'V': 34, 'W': 35, 'X': 36,
    'Y': 37, 'Z': 38,
}


def validate_iso6346_check_digit(container_number: str) -> bool:
    """
    Validate the ISO 6346 check digit for a container number.
    Letters map to values 10-38, skipping multiples of 11 (11, 22, 33).
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
        total += val * 2 ** i

    remainder = total % 11
    expected_check = 0 if remainder == 10 else remainder

    try:
        return int(container_number[10]) == expected_check
    except ValueError:
        return False


def validate_weight_consistency(
    tare_weight_kg: float,
    tare_weight_lbs: float,
    payload_weight_kg: float,
    payload_weight_lbs: float,
    max_gross_weight_kg: float,
    max_gross_weight_lbs: float,
    margin: float = 11.0
) -> Tuple[bool, str]:
    """
    Validate consistency between kilogram and pound values for weight metrics.
    Converts lbs to kgs, rounds to nearest 10kgs, and allows margin of ±11kgs.

    Args:
        tare_weight_kg: Tare weight in kilograms
        tare_weight_lbs: Tare weight in pounds
        payload_weight_kg: Payload weight in kilograms
        payload_weight_lbs: Payload weight in pounds
        max_gross_weight_kg: Max gross weight in kilograms
        max_gross_weight_lbs: Max gross weight in pounds
        margin: Acceptable margin in kilograms (default 11.0 kg)

    Returns:
        Tuple of (is_valid: bool, detailed_reason: str)
        detailed_reason is empty string if valid, or describes inconsistencies if invalid
    """
    LBS_TO_KG = 1 / 2.20462
    reasons = []

    # Check tare weight conversion
    expected_tare_kg = tare_weight_lbs * LBS_TO_KG
    rounded_tare_kg = round(expected_tare_kg / 10) * 10
    diff_tare = abs(rounded_tare_kg - tare_weight_kg)
    if diff_tare > margin:
        reasons.append(f"Tare weight: {tare_weight_lbs}lbs converts to ~{rounded_tare_kg}kg (calculated: {expected_tare_kg:.1f}kg), but provided {tare_weight_kg}kg (diff: {diff_tare:.1f}kg)")

    # Check payload weight conversion
    expected_payload_kg = payload_weight_lbs * LBS_TO_KG
    rounded_payload_kg = round(expected_payload_kg / 10) * 10
    diff_payload = abs(rounded_payload_kg - payload_weight_kg)
    if diff_payload > margin:
        reasons.append(f"Payload weight: {payload_weight_lbs}lbs converts to ~{rounded_payload_kg}kg (calculated: {expected_payload_kg:.1f}kg), but provided {payload_weight_kg}kg (diff: {diff_payload:.1f}kg)")

    # Check max gross weight conversion
    expected_max_gross_kg = max_gross_weight_lbs * LBS_TO_KG
    rounded_max_gross_kg = round(expected_max_gross_kg / 10) * 10
    diff_max_gross = abs(rounded_max_gross_kg - max_gross_weight_kg)
    if diff_max_gross > margin:
        reasons.append(f"Max gross weight: {max_gross_weight_lbs}lbs converts to ~{rounded_max_gross_kg}kg (calculated: {expected_max_gross_kg:.1f}kg), but provided {max_gross_weight_kg}kg (diff: {diff_max_gross:.1f}kg)")

    is_valid = len(reasons) == 0
    return is_valid, "; ".join(reasons)
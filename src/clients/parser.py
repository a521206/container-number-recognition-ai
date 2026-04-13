"""Shared result parsing logic for extraction clients."""

import logging
from typing import Any, Dict, Optional

from ..processing.models import ContainerResult, Weights, WeightValue, OwnerOperator
from ..utils.validation import validate_iso6346_check_digit

log = logging.getLogger(__name__)


def _safe_int(value: Any) -> Optional[int]:
    """Safely convert value to int, returning None on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None


def parse_extracted_data(data: Dict[str, Any]) -> ContainerResult:
    """
    Parse extracted data from any client (Llama Extract, Document Intelligence, etc.)
    into a ContainerResult. This function centralizes the parsing logic that was
    previously duplicated in each client.

    Args:
        data: Dictionary containing extracted fields from the API response.
              Expected keys: owner_code, serial_number, container_id, container_number,
              container_type, container_type_code, weights, owner_operator

    Returns:
        ContainerResult with parsed fields populated.
    """
    result = ContainerResult()

    if not data:
        log.warning("No data provided to parser")
        result.error = "No data extracted"
        return result

    _container_number_sources = []

    owner_code = data.get("owner_code")
    serial_number = data.get("serial_number")

    if owner_code and serial_number:
        sn = str(serial_number).strip().replace(" ", "")
        result.container_number = f"{owner_code}{sn}"
        _container_number_sources.append("owner_code+serial_number")
        log.debug("Parsed container number from owner_code + serial_number: %s", result.container_number)

    container_id = data.get("container_id")
    if container_id and not result.container_number:
        cid_str = str(container_id).strip()
        parts = cid_str.split()
        if len(parts) >= 2:
            result.container_number = f"{parts[0]}{''.join(parts[1:])}"
        else:
            result.container_number = cid_str.replace(" ", "")
        _container_number_sources.append("container_id")
        log.debug("Parsed container number from container_id: %s", result.container_number)

    container_number = data.get("container_number")
    if container_number and not result.container_number:
        result.container_number = str(container_number).strip().upper().replace(" ", "")
        _container_number_sources.append("container_number")
        log.debug("Parsed container number from container_number field: %s", result.container_number)

    source = _container_number_sources[-1] if _container_number_sources else None
    result.source = source

    _container_type_sources = []
    container_type = data.get("container_type")
    if container_type:
        result.container_type = str(container_type).strip().upper().replace(" ", "")
        _container_type_sources.append("container_type")
        log.debug("Parsed container type: %s", result.container_type)

    container_type_code = data.get("container_type_code")
    if container_type_code and not result.container_type:
        result.container_type = str(container_type_code).upper()
        _container_type_sources.append("container_type_code")
        log.debug("Parsed container type from container_type_code: %s", result.container_type)

    type_source = _container_type_sources[-1] if _container_type_sources else None
    if type_source == "container_type_code":
        result.raw_container_type = str(data.get("container_type_code")).upper()

    weights_data = data.get("weights")
    if weights_data and isinstance(weights_data, dict):
        result.weights = _parse_weights(weights_data)

    owner_op = data.get("owner_operator")
    if owner_op and isinstance(owner_op, dict):
        result.owner_operator = _parse_owner_operator(owner_op)

    if result.container_number and len(result.container_number) == 11:
        if not validate_iso6346_check_digit(result.container_number):
            log.warning("Container number %s failed ISO 6346 check digit validation", result.container_number)
            result.valid = False
            result.reason = "Invalid check digit"
            result.raw_container_number = result.container_number

    log.debug(
        "Parsed result: number=%s, type=%s",
        result.container_number or "none",
        result.container_type or "none",
    )
    return result


def _parse_weights(weights_data: Dict[str, Any]) -> Optional[Weights]:
    """Parse weights data into a Weights object."""
    tare = weights_data.get("tare_weight", {})
    payload = weights_data.get("payload_weight", {})
    max_gross = weights_data.get("maximum_gross_weight", {})

    return Weights(
        tare_weight=WeightValue(
            pounds=_safe_int(tare.get("pounds")),
            kilograms=_safe_int(tare.get("kilograms")),
        ),
        payload_weight=WeightValue(
            pounds=_safe_int(payload.get("pounds")),
            kilograms=_safe_int(payload.get("kilograms")),
        ),
        maximum_gross_weight=WeightValue(
            pounds=_safe_int(max_gross.get("pounds")),
            kilograms=_safe_int(max_gross.get("kilograms")),
        ),
    )


def _parse_owner_operator(owner_op: Dict[str, Any]) -> Optional[OwnerOperator]:
    """Parse owner/operator data into an OwnerOperator object."""
    name = owner_op.get("name")
    location = owner_op.get("location")
    return OwnerOperator(
        name=str(name) if name else None,
        location=str(location) if location else None,
    )
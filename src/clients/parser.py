"""Shared result parsing logic for extraction clients."""

import logging
from typing import Any, Dict, Optional

from ..processing.models import ContainerResult, Weights, WeightValue, OwnerOperator
from ..utils.validation import validate_iso6346_check_digit

log = logging.getLogger(__name__)


def parse_extracted_data(data: Dict[str, Any]) -> ContainerResult:
    """
    Parse extracted data from any client (Llama Extract, Document Intelligence, etc.)
    into a ContainerResult. This function centralizes the parsing logic that was
    previously duplicated in each client.
    
    Args:
        data: Dictionary containing extracted fields from the API response.
              Expected keys: owner_code, serial_number, container_id, container_number,
              container_type, status, container_type_code, weights, owner_operator
    
    Returns:
        ContainerResult with parsed fields populated.
    """
    result = ContainerResult()

    if not data:
        log.warning("No data provided to parser")
        result.error = "No data extracted"
        return result

    owner_code = data.get("owner_code")
    serial_number = data.get("serial_number")

    if owner_code and serial_number:
        sn = str(serial_number).strip().replace(" ", "")
        result.container_number = f"{owner_code}{sn}"
        result.owner_code = str(owner_code).upper()
        result.serial_number = sn
        log.debug("Parsed container number from owner_code + serial_number: %s", result.container_number)

    container_id = data.get("container_id")
    if container_id and not result.container_number:
        cid_str = str(container_id).strip()
        parts = cid_str.split()
        if len(parts) >= 2:
            result.container_number = f"{parts[0]}{''.join(parts[1:])}"
            result.owner_code = parts[0].upper()
            result.serial_number = "".join(parts[1:])
        else:
            result.container_number = cid_str.replace(" ", "")
        log.debug("Parsed container number from container_id: %s", result.container_number)

    container_number = data.get("container_number")
    if container_number and not result.container_number:
        result.container_number = str(container_number).strip().upper().replace(" ", "")
        log.debug("Parsed container number from container_number field: %s", result.container_number)

    container_type = data.get("container_type")
    if container_type:
        result.container_type = str(container_type).strip().upper().replace(" ", "")
        log.debug("Parsed container type: %s", result.container_type)

    status = data.get("status")
    if status:
        result.status = str(status)

    container_type_code = data.get("container_type_code")
    if container_type_code:
        result.container_type_code = str(container_type_code).upper()

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
            result.container_number = ""

    log.debug(
        "Parsed result: number=%s, type=%s, status=%s",
        result.container_number or "none",
        result.container_type or "none",
        result.status or "none",
    )
    return result


def _parse_weights(weights_data: Dict[str, Any]) -> Optional[Weights]:
    """Parse weights data into a Weights object."""
    tare = weights_data.get("tare_weight", {})
    payload = weights_data.get("payload_weight", {})
    max_gross = weights_data.get("maximum_gross_weight", {})

    return Weights(
        tare_weight=WeightValue(
            pounds=int(float(tare.get("pounds"))) if tare.get("pounds") else None,
            kilograms=int(float(tare.get("kilograms"))) if tare.get("kilograms") else None,
        ),
        payload_weight=WeightValue(
            pounds=int(float(payload.get("pounds"))) if payload.get("pounds") else None,
            kilograms=int(float(payload.get("kilograms"))) if payload.get("kilograms") else None,
        ),
        maximum_gross_weight=WeightValue(
            pounds=int(float(max_gross.get("pounds"))) if max_gross.get("pounds") else None,
            kilograms=int(float(max_gross.get("kilograms"))) if max_gross.get("kilograms") else None,
        ),
    )


def _parse_owner_operator(owner_op: Dict[str, Any]) -> Optional[OwnerOperator]:
    """Parse owner/operator data into an OwnerOperator object."""
    return OwnerOperator(
        name=str(owner_op.get("name")) if owner_op.get("name") else None,
        location=str(owner_op.get("location")) if owner_op.get("location") else None,
    )
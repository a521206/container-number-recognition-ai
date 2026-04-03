"""Azure Document Intelligence client for container data extraction."""

import logging
import os
import tempfile
import time
from typing import Optional

from azure.ai.documentintelligence import DocumentIntelligenceClient as DIClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential

from .config import (
    DOCUMENT_INTELLIGENCE_ENDPOINT,
    DOCUMENT_INTELLIGENCE_KEY,
    DOCUMENT_INTELLIGENCE_MODEL_ID,
)
from .extraction import ContainerResult, Weights, WeightValue, OwnerOperator

log = logging.getLogger(__name__)

_DOC_INTELLIGENCE_POLL_INTERVAL = 2
_DOC_INTELLIGENCE_POLL_TIMEOUT = 120
_MAX_RETRIES = 3
_INITIAL_RETRY_DELAY = 0.5
_MAX_RETRY_DELAY = 8


class DocumentIntelligenceClient:
    """Client for Azure Document Intelligence API."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        self.endpoint = endpoint or DOCUMENT_INTELLIGENCE_ENDPOINT
        self.api_key = api_key or DOCUMENT_INTELLIGENCE_KEY
        self.model_id = model_id or DOCUMENT_INTELLIGENCE_MODEL_ID

        if not self.endpoint:
            raise ValueError("DOCUMENT_INTELLIGENCE_ENDPOINT must be set in environment")
        if not self.api_key:
            raise ValueError("DOCUMENT_INTELLIGENCE_KEY must be set in environment")

        self._client = None
        log.info("DocumentIntelligenceClient initialized with model_id=%s", self.model_id)

    @property
    def client(self) -> DIClient:
        """Lazy initialization of Document Intelligence client."""
        if self._client is None:
            self._client = DIClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key),
            )
        return self._client

    @staticmethod
    def _is_transient_error(e: Exception) -> bool:
        """Check if error is transient (network/API issues)."""
        transient_types = (ConnectionError, TimeoutError, OSError)
        transient_messages = ("connection", "timeout", "network", "temporarily unavailable", "429", "503")
        if isinstance(e, transient_types):
            return True
        error_str = str(e).lower()
        return any(msg in error_str for msg in transient_messages)

    def _with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic and exponential backoff."""
        last_error = None
        for attempt in range(_MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if not self._is_transient_error(e):
                    raise
                if attempt < _MAX_RETRIES - 1:
                    delay = min(_INITIAL_RETRY_DELAY * (2 ** attempt), _MAX_RETRY_DELAY)
                    import random
                    delay += random.uniform(0, 0.5)
                    log.warning("Attempt %d failed: %s, retrying in %.1fs", attempt + 1, e, delay)
                    time.sleep(delay)
        if last_error:
            raise last_error
        raise RuntimeError("Retry logic failed without capturing an error")

    def _wait_for_analyze_result(self, poller):
        """Poll for analysis completion with retry logic."""
        deadline = time.monotonic() + _DOC_INTELLIGENCE_POLL_TIMEOUT
        while True:
            try:
                result = poller.result()
                return result
            except Exception as e:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Document Intelligence analysis timed out after {_DOC_INTELLIGENCE_POLL_TIMEOUT}s"
                    )
                # Check if still running
                if "not started" in str(e).lower() or "still running" in str(e).lower():
                    time.sleep(_DOC_INTELLIGENCE_POLL_INTERVAL)
                    continue
                raise

    def extract_from_file(self, file_path: str) -> ContainerResult:
        """Extract container data from a file path using Azure Document Intelligence."""
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()

            return self._analyze_document(file_bytes, os.path.basename(file_path))

        except Exception as e:
            log.exception("Document Intelligence error for %s", file_path)
            return ContainerResult(error=f"DocumentIntelligenceError: {e}")

    def extract_from_bytes(self, data: bytes, filename: str = "image.jpg") -> ContainerResult:
        """Extract container data from image bytes using Azure Document Intelligence."""
        try:
            return self._analyze_document(data, filename)
        except Exception as e:
            log.exception("Document Intelligence error for bytes (filename=%s)", filename)
            return ContainerResult(error=f"DocumentIntelligenceError: {e}")

    def _analyze_document(self, file_bytes: bytes, filename: str) -> ContainerResult:
        """Analyze document with Azure Document Intelligence."""
        try:
            # Create the analysis request with bytes source
            analyze_request = AnalyzeDocumentRequest(bytes_source=file_bytes)

            # Call the API
            poller = self.client.begin_analyze_document(
                model_id=self.model_id,
                body=analyze_request,
            )

            log.debug("Document analysis started for %s", filename)
            analyze_result = poller.result()

            log.info("Document analysis completed for %s", filename)
            return self._parse_analyze_result(analyze_result)

        except TimeoutError as e:
            log.error("Document Intelligence timeout: %s", e)
            return ContainerResult(error=str(e))
        except Exception as e:
            log.exception("Document Intelligence error for %s", filename)
            return ContainerResult(error=f"DocumentIntelligenceError: {e}")

    def _parse_analyze_result(self, analyze_result) -> ContainerResult:
        """Parse Azure Document Intelligence result into a ContainerResult."""
        result = ContainerResult()

        if analyze_result is None:
            log.warning("Document Intelligence returned empty result")
            result.error = "No data extracted"
            return result

        # Get documents from result
        documents = getattr(analyze_result, "documents", None)
        if not documents or not isinstance(documents, list) or len(documents) == 0:
            log.warning("Document Intelligence returned no documents")
            result.error = "No data extracted"
            return result

        doc = documents[0]
        fields = getattr(doc, "fields", None)
        if not fields:
            log.warning("Document Intelligence returned no fields")
            result.error = "No fields extracted"
            return result

        # Parse container number fields (matching Llama Extract output structure)
        owner_code = self._get_field_value(fields, "owner_code")
        serial_number = self._get_field_value(fields, "serial_number")

        if owner_code and serial_number:
            sn = str(serial_number).strip().replace(" ", "")
            result.container_number = f"{owner_code}{sn}"
            result.owner_code = str(owner_code).upper()
            result.serial_number = sn

        # Try container_id if not found
        container_id = self._get_field_value(fields, "container_id")
        if container_id and not result.container_number:
            cid_str = str(container_id).strip()
            parts = cid_str.split()
            if len(parts) >= 2:
                result.container_number = f"{parts[0]}{''.join(parts[1:])}"
                result.owner_code = parts[0].upper()
                result.serial_number = "".join(parts[1:])
            else:
                result.container_number = cid_str.replace(" ", "")

        # Container number as fallback
        container_number = self._get_field_value(fields, "container_number")
        if container_number and not result.container_number:
            result.container_number = str(container_number).strip().upper().replace(" ", "")

        # Container type
        container_type = self._get_field_value(fields, "container_type")
        if container_type:
            result.container_type = str(container_type).strip().upper().replace(" ", "")

        # Status
        status = self._get_field_value(fields, "status")
        if status:
            result.status = str(status)

        # Container type code
        container_type_code = self._get_field_value(fields, "container_type_code")
        if container_type_code:
            result.container_type_code = str(container_type_code).upper()

        # Weights
        weights_data = fields.get("weights")
        if weights_data and hasattr(weights_data, "value"):
            weights_dict = weights_data.value
            if weights_dict and isinstance(weights_dict, dict):
                tare = weights_dict.get("tare_weight", {})
                payload = weights_dict.get("payload_weight", {})
                max_gross = weights_dict.get("maximum_gross_weight", {})

                result.weights = Weights(
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
        elif weights_data and isinstance(weights_data, dict):
            # Handle raw dict format
            tare = weights_data.get("tare_weight", {})
            payload = weights_data.get("payload_weight", {})
            max_gross = weights_data.get("maximum_gross_weight", {})

            if tare:
                result.weights = Weights(
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

        # Owner/Operator
        owner_op = fields.get("owner_operator")
        if owner_op and hasattr(owner_op, "value"):
            owner_op_dict = owner_op.value
            if owner_op_dict and isinstance(owner_op_dict, dict):
                result.owner_operator = OwnerOperator(
                    name=str(owner_op_dict.get("name")) if owner_op_dict.get("name") else None,
                    location=str(owner_op_dict.get("location")) if owner_op_dict.get("location") else None,
                )
        elif owner_op and isinstance(owner_op, dict):
            result.owner_operator = OwnerOperator(
                name=str(owner_op.get("name")) if owner_op.get("name") else None,
                location=str(owner_op.get("location")) if owner_op.get("location") else None,
            )

        log.debug(
            "Parsed result: number=%s, type=%s, status=%s",
            result.container_number or "none",
            result.container_type or "none",
            result.status or "none",
        )
        return result

    def _get_field_value(self, fields, field_name: str):
        """Get field value from Document Intelligence fields."""
        field = fields.get(field_name)
        if field is None:
            return None
        # Document Intelligence returns fields with 'value' attribute for typed fields
        if hasattr(field, "value"):
            return field.value
        # Or direct value
        if hasattr(field, "content"):
            return field.content
        return str(field)
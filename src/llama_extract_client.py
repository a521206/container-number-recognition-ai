"""Llama Extract client for container number extraction using official SDK."""

import logging
import os
import tempfile
import time
from typing import Optional

from llama_cloud import LlamaCloud

from .config import LLAMA_CLOUD_API_KEY, LLAMA_EXTRACT_CONFIG_ID
from .extraction import ContainerResult, Weights, WeightValue, OwnerOperator

log = logging.getLogger(__name__)

_EXTRACT_POLL_INTERVAL = 2
_EXTRACT_POLL_TIMEOUT = 120
_MAX_RETRIES = 3
_INITIAL_RETRY_DELAY = 0.5
_MAX_RETRY_DELAY = 8


class LlamaExtractClient:
    """Client for Llama Extract API using the official llama-cloud SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        config_id: Optional[str] = None,
    ):
        self.api_key = api_key or LLAMA_CLOUD_API_KEY
        self.config_id = config_id or LLAMA_EXTRACT_CONFIG_ID

        if not self.api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY must be set in environment")
        if not self.config_id:
            raise ValueError("LLAMA_EXTRACT_CONFIG_ID must be set in environment")

        self._client = None
        log.info("LlamaExtractClient initialized with config_id=%s", self.config_id)

    @property
    def client(self) -> LlamaCloud:
        """Lazy initialization of LlamaCloud client."""
        if self._client is None:
            self._client = LlamaCloud(api_key=self.api_key)
        return self._client

    @staticmethod
    def _is_transient_error(e: Exception) -> bool:
        """Check if error is transient (network/API issues)."""
        transient_types = (
            ConnectionError,
            TimeoutError,
            OSError,
        )
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

    def _wait_for_job(self, job):
        """Poll for job completion with retry logic."""
        deadline = time.monotonic() + _EXTRACT_POLL_TIMEOUT
        while True:
            try:
                status = str(job.status) if hasattr(job, "status") else str(job)
                if status in ("COMPLETED", "FAILED", "CANCELLED"):
                    break
            except Exception:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Job {getattr(job, 'id', 'unknown')} timed out after {_EXTRACT_POLL_TIMEOUT}s"
                )
            time.sleep(_EXTRACT_POLL_INTERVAL)
            job = self._with_retry(lambda: self.client.extract.get(job.id))
        return job

    def extract_from_file(self, file_path: str) -> ContainerResult:
        """Extract container data from a file path using Llama Extract."""
        file_obj = None
        try:
            file_obj = self._with_retry(
                lambda: self.client.files.create(
                    file=file_path,
                    purpose="extract",
                )
            )
            log.debug("File uploaded: %s", file_obj.id)

            job = self._with_retry(
                lambda: self.client.extract.create(
                    file_input=file_obj.id,
                    configuration_id=self.config_id,
                )
            )
            log.debug("Extraction job created: %s (status=%s)", job.id, job.status)

            job = self._wait_for_job(job)

            if job.status != "COMPLETED":
                error_msg = getattr(job, "error_message", f"Job finished with status: {job.status}")
                log.error("Llama Extract job %s failed: %s", job.id, error_msg)
                return ContainerResult(error=error_msg)

            log.info("Llama Extract job %s completed successfully", job.id)
            return self._parse_extract_result(job.extract_result)

        except TimeoutError as e:
            log.error("Llama Extract timeout: %s", e)
            return ContainerResult(error=str(e))
        except Exception as e:
            log.exception("Llama Extract error for %s", file_path)
            return ContainerResult(error=f"LlamaExtractError: {e}")
        finally:
            if file_obj:
                try:
                    self.client.files.delete(file_obj.id)
                    log.debug("Deleted uploaded file: %s", file_obj.id)
                except Exception as cleanup_err:
                    log.warning("Failed to delete uploaded file %s: %s", file_obj.id, cleanup_err)

    def extract_from_bytes(self, data: bytes, filename: str = "image.jpg") -> ContainerResult:
        """Extract container data from image bytes using Llama Extract."""
        temp_path = None
        try:
            suffix = os.path.splitext(filename)[1] or ".jpg"
            fd, temp_path = tempfile.mkstemp(suffix=suffix)
            try:
                os.write(fd, data)
            finally:
                os.close(fd)
            return self.extract_from_file(temp_path)
        except Exception as e:
            log.exception("Llama Extract error for bytes (filename=%s)", filename)
            return ContainerResult(error=f"LlamaExtractError: {e}")
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as cleanup_err:
                    log.warning("Failed to delete temp file %s: %s", temp_path, cleanup_err)

    def _parse_extract_result(self, extract_result) -> ContainerResult:
        """Parse Llama Extract result into a ContainerResult."""
        result = ContainerResult()

        if extract_result is None:
            log.warning("Llama Extract returned empty result")
            result.error = "No data extracted"
            return result

        if isinstance(extract_result, list):
            if not extract_result:
                result.error = "No data extracted"
                return result
            extract_result = extract_result[0]

        if not isinstance(extract_result, dict):
            extract_result = extract_result.__dict__ if hasattr(extract_result, "__dict__") else {}

        owner_code = extract_result.get("owner_code")
        serial_number = extract_result.get("serial_number")

        if owner_code and serial_number:
            sn = str(serial_number).strip().replace(" ", "")
            result.container_number = f"{owner_code}{sn}"
            result.owner_code = str(owner_code).upper()
            result.serial_number = sn

        container_id = extract_result.get("container_id")
        if container_id and not result.container_number:
            cid_str = str(container_id).strip()
            parts = cid_str.split()
            if len(parts) >= 2:
                result.container_number = f"{parts[0]}{''.join(parts[1:])}"
                result.owner_code = parts[0].upper()
                result.serial_number = "".join(parts[1:])
            else:
                result.container_number = cid_str.replace(" ", "")

        container_number = extract_result.get("container_number")
        if container_number and not result.container_number:
            result.container_number = str(container_number).strip().upper().replace(" ", "")

        container_type = extract_result.get("container_type", "")
        if container_type:
            result.container_type = str(container_type).strip().upper().replace(" ", "")

        status = extract_result.get("status")
        if status:
            result.status = str(status)

        container_type_code = extract_result.get("container_type_code")
        if container_type_code:
            result.container_type_code = str(container_type_code).upper()

        weights_data = extract_result.get("weights")
        if weights_data and isinstance(weights_data, dict):
            tare = weights_data.get("tare_weight", {})
            payload = weights_data.get("payload_weight", {})
            max_gross = weights_data.get("maximum_gross_weight", {})

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

        owner_op = extract_result.get("owner_operator")
        if owner_op and isinstance(owner_op, dict):
            result.owner_operator = OwnerOperator(
                name=str(owner_op.get("name")) if owner_op.get("name") else None,
                location=str(owner_op.get("location")) if owner_op.get("location") else None,
            )

        log.debug(
            "Parsed result: number=%s, type=%s",
            result.container_number or "none",
            result.container_type or "none",
        )
        return result
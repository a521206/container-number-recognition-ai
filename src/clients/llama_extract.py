"""Llama Extract client for container number extraction using official SDK."""

import logging
import os
import random
import tempfile
import time
from typing import Optional

from llama_cloud import LlamaCloud

from ..utils.config import LLAMA_CLOUD_API_KEY, LLAMA_EXTRACT_CONFIG_ID
from ..processing.extraction import ContainerResult
from .base import ExtractionClient
from .parser import parse_extracted_data

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
    def name(self) -> str:
        return "llama_extract"

    @property
    def client(self) -> LlamaCloud:
        """Lazy initialization of LlamaCloud client."""
        if self._client is None:
            self._client = LlamaCloud(api_key=self.api_key)
        return self._client

    @staticmethod
    def _normalize_status(raw) -> str:
        """Return a plain uppercase status string regardless of whether the
        SDK gives us a string (``"COMPLETED"``), an enum with a ``.name``
        attribute (``ExtractionStatus.COMPLETED``), or a dotted string
        (``"ExtractionStatus.COMPLETED"``).
        """
        if hasattr(raw, "name"):          # proper Python enum
            return raw.name.upper()
        if hasattr(raw, "value"):         # enum-like with a value attribute
            return str(raw.value).upper()
        s = str(raw).upper()
        return s.split(".")[-1] if "." in s else s  # "FOO.COMPLETED" → "COMPLETED"

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
                raw = job.status if hasattr(job, "status") else job
                status = self._normalize_status(raw)
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
        temp_bytes = None
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

            if self._normalize_status(job.status) != "COMPLETED":
                error_msg = getattr(job, "error_message", f"Job finished with status: {job.status}")
                log.error("Llama Extract job %s failed: %s", job.id, error_msg)
                return ContainerResult(error=error_msg)

            log.info("Llama Extract job %s completed successfully", job.id)
            
            result = self._parse_extract_result(job.extract_result)
            return result

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
        """Parse Llama Extract result into a ContainerResult using shared parser."""
        if extract_result is None:
            log.warning("Llama Extract returned empty result")
            return ContainerResult(error="No data extracted")

        if isinstance(extract_result, list):
            if not extract_result:
                return ContainerResult(error="No data extracted")
            extract_result = extract_result[0]

        if not isinstance(extract_result, dict):
            extract_result = extract_result.__dict__ if hasattr(extract_result, "__dict__") else {}

        return parse_extracted_data(extract_result)
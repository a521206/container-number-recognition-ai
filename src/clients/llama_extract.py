"""Llama Extract client for container number extraction using official SDK."""

import logging
import mimetypes
from typing import Optional

from llama_cloud import LlamaCloud

from ..utils.config import LLAMA_CLOUD_API_KEY, LLAMA_EXTRACT_CONFIG_ID
from ..processing.extraction import ContainerResult
from .parser import parse_extracted_data

log = logging.getLogger(__name__)

_EXTRACT_POLL_TIMEOUT = 120


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

    def extract_from_bytes(self, data: bytes, filename: str = "image.jpg") -> ContainerResult:
        """Extract container data from image bytes using Llama Extract."""
        file_obj = None
        try:
            mime_type = mimetypes.guess_type(filename)[0] or "image/jpeg"
            file_obj = self.client.files.create(
                file=(filename, data, mime_type),
                purpose="extract",
            )
            log.debug("File uploaded: %s", file_obj.id)

            job = self.client.extract.run(
                file_input=file_obj.id,
                configuration_id=self.config_id,
                polling_timeout=_EXTRACT_POLL_TIMEOUT,
            )
            log.debug("Extraction job created: %s (status=%s)", job.id, job.status)

            if job.status.upper() != "COMPLETED":
                log.error("Llama Extract job %s failed: %s", job.id, job.error_message)
                return ContainerResult(error=job.error_message or f"Job finished with status: {job.status}")

            log.info("Llama Extract job %s completed successfully", job.id)
            return self._parse_extract_result(job.extract_result)

        except TimeoutError as e:
            log.error("Llama Extract timeout: %s", e)
            return ContainerResult(error=str(e))
        except Exception as e:
            log.exception("Llama Extract error for bytes (filename=%s)", filename)
            return ContainerResult(error=f"LlamaExtractError: {e}")
        finally:
            if file_obj:
                try:
                    self.client.files.delete(file_obj.id)
                    log.debug("Deleted uploaded file: %s", file_obj.id)
                except Exception as cleanup_err:
                    log.warning("Failed to delete uploaded file %s: %s", file_obj.id, cleanup_err)

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
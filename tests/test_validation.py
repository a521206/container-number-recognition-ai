"""Tests for container number recognition."""

import pytest
from src.validation import validate_iso6346_check_digit


class TestValidation:
    """Test ISO 6346 validation."""

    def test_valid_container_number(self):
        """Test valid container number."""
        assert validate_iso6346_check_digit("CSQU3054383") is True

    def test_invalid_container_number(self):
        """Test invalid container number."""
        assert validate_iso6346_check_digit("CSQU3054384") is False

    def test_wrong_length(self):
        """Test wrong length."""
        assert validate_iso6346_check_digit("CSQU305438") is False
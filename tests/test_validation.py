"""Tests for container number recognition."""

import pytest
from src.utils.validation import validate_iso6346_check_digit, validate_weight_consistency


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


class TestWeightValidation:
    """Test weight validation."""

    def test_valid_weights(self):
        """Test valid weight conversions within margin."""
        # Tare weight: 4410 lbs converts to ~2000kg (rounded to nearest 10kg)
        is_valid, reason = validate_weight_consistency(2000, 4410, 10000, 22050, 12000, 26460)
        assert is_valid is True
        assert reason == ""

    def test_invalid_tare_weight(self):
        """Test invalid tare weight conversion."""
        # Tare weight: 4000 lbs converts to ~1814kg (rounded to 1810kg), but we provide 2000 kg (> 11 kg margin)
        is_valid, reason = validate_weight_consistency(2000, 4000, 10000, 22050, 12000, 26460)
        assert is_valid is False
        assert "Tare weight: 4000lbs converts to ~1810kg" in reason

    def test_invalid_payload_weight(self):
        """Test invalid payload weight conversion."""
        # Payload weight: 20000 lbs converts to ~9072kg (rounded to 9070kg), but we provide 10000 kg (> 11 kg margin)
        is_valid, reason = validate_weight_consistency(2000, 4410, 10000, 20000, 12000, 26460)
        assert is_valid is False
        assert "Payload weight: 20000lbs converts to ~9070kg" in reason

    def test_invalid_max_gross_weight(self):
        """Test invalid max gross weight conversion."""
        # Max gross weight: 24000 lbs converts to ~10886kg (rounded to 10890kg), but we provide 12000 kg (> 11 kg margin)
        is_valid, reason = validate_weight_consistency(2000, 4410, 10000, 22050, 12000, 24000)
        assert is_valid is False
        assert "Max gross weight: 24000lbs converts to ~10890kg" in reason
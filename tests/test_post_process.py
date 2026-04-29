"""Tests for post-processing utilities."""

import pytest
import cv2
from src.processing.post_process import post_process_result, extract_dominant_color_from_image_bytes
from src.processing.models import ContainerResult
import numpy as np


class TestPostProcess:
    """Test post-processing functions."""

    def test_extract_dominant_color_valid_image(self):
        """Test dominant color extraction from valid image bytes."""
        # Create a simple test image (blue square)
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :] = [255, 0, 0]  # Blue color in BGR format
        success, encoded_image = cv2.imencode('.png', test_image)
        assert success, "Failed to encode test image"
        image_bytes = encoded_image.tobytes()

        color = extract_dominant_color_from_image_bytes(image_bytes)
        # Should extract some color (exact value may vary due to compression)
        assert isinstance(color, list)
        assert len(color) == 3
        assert all(isinstance(c, int) and 0 <= c <= 255 for c in color)

    def test_extract_dominant_color_corrupted_image(self):
        """Test dominant color extraction from corrupted/invalid image bytes."""
        corrupted_bytes = b"not an image"

        with pytest.raises(ValueError, match="Could not decode image bytes"):
            extract_dominant_color_from_image_bytes(corrupted_bytes)

    def test_post_process_adds_color(self):
        """Test that post-processing adds container color when missing."""
        result = ContainerResult(
            container_number="CSQU3054383",
            container_color=None
        )

        # Create a test image (red square)
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :] = [0, 0, 255]  # Red color in BGR format
        success, encoded_image = cv2.imencode('.png', test_image)
        assert success, "Failed to encode test image"
        image_bytes = encoded_image.tobytes()

        processed = post_process_result(result, image_bytes)

        # Should have added a color
        assert processed.container_color is not None
        assert isinstance(processed.container_color, list)
        assert len(processed.container_color) == 3

    def test_post_process_no_container_number(self):
        """Test that post-processing skips when no container number."""
        result = ContainerResult(container_number="")

        # Create a test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        image_bytes = test_image.tobytes()

        processed = post_process_result(result, image_bytes)

        # Should return unchanged (no color processing)
        assert processed == result
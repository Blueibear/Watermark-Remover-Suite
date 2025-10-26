"""Pytest configuration and fixtures for the Watermark Remover Suite tests.

This module provides shared test fixtures and configuration.

For lightweight test assets, use the helpers in tests/helpers.py:
- create_synthetic_sample(): Creates a small gradient image with watermark text
- create_test_video_clip(): Generates a short synthetic video clip

These helpers generate minimal in-memory test data without requiring:
- Network downloads
- Large binary fixtures
- Heavy model weights
- GPU resources

All CPU-only tests should use these helpers to remain hermetic and fast.
"""

import pytest


# Example: Add shared fixtures here if needed
# @pytest.fixture
# def temp_config():
#     """Temporary configuration for tests."""
#     return {...}

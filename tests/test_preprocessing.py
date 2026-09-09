"""Tests for the chest X-ray preprocessing pipeline.

Verifies the preprocessor turns a real chest X-ray into the exact tensor the
TorchXRayVision DenseNet expects: shape (1, 1, 224, 224), float32, values within
xrv's ~[-1024, 1024] normalization range.
"""
from pathlib import Path

import pytest
import torch

from app.services.preprocessing import ChestXrayPreprocessor, Preprocessor, TARGET_SIZE

SAMPLE = Path("data/sample_cxr.jpg")


@pytest.fixture
def preprocessor() -> ChestXrayPreprocessor:
    return ChestXrayPreprocessor()


def test_is_a_preprocessor(preprocessor):
    # The concrete class honours the pluggable interface (D-11).
    assert isinstance(preprocessor, Preprocessor)


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image not present")
def test_output_shape(preprocessor):
    t = preprocessor.process(SAMPLE)
    assert t.shape == (1, 1, TARGET_SIZE, TARGET_SIZE)


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image not present")
def test_output_dtype(preprocessor):
    t = preprocessor.process(SAMPLE)
    assert t.dtype == torch.float32


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image not present")
def test_output_range(preprocessor):
    # xrv's convention scales to roughly [-1024, 1024]; allow a small margin.
    t = preprocessor.process(SAMPLE)
    assert t.min() >= -1100.0
    assert t.max() <= 1100.0


def test_missing_file_raises(preprocessor):
    with pytest.raises(FileNotFoundError):
        preprocessor.process("data/does_not_exist.jpg")

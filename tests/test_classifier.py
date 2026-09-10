"""Tests for the chest X-ray classifier.

Verifies the classifier turns a preprocessed tensor into a structured
ClassificationResult: 18 findings, per-finding op-threshold flags (D-07), and
captured activations for GradCAM (Phase 3 groundwork).
"""
from pathlib import Path

import pytest
import torch

from app.services.classifier import (
    ChestXrayClassifier,
    Classifier,
    ClassificationResult,
    Finding,
)
from app.services.preprocessing import ChestXrayPreprocessor

SAMPLE = Path("data/sample_cxr.jpg")

# Load the model once for all tests in this module (it's slow to construct).
_clf = ChestXrayClassifier()


def test_is_a_classifier():
    # Honours the pluggable interface (D-11).
    assert isinstance(_clf, Classifier)


def test_has_18_pathologies():
    assert len(_clf.pathologies) == 18


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image not present")
def test_predict_returns_result():
    tensor = ChestXrayPreprocessor().process(SAMPLE)
    result = _clf.predict(tensor)
    assert isinstance(result, ClassificationResult)
    assert len(result.findings) == 18
    assert all(isinstance(f, Finding) for f in result.findings)


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image not present")
def test_findings_have_valid_probabilities():
    tensor = ChestXrayPreprocessor().process(SAMPLE)
    result = _clf.predict(tensor)
    for f in result.findings:
        assert 0.0 <= f.probability <= 1.0
        # present flag must agree with prob-vs-threshold
        assert f.present == (f.probability > f.threshold)


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image not present")
def test_activations_captured_for_gradcam():
    # The forward hook must capture the DenseNet feature map (Phase 3 needs it).
    tensor = ChestXrayPreprocessor().process(SAMPLE)
    result = _clf.predict(tensor)
    assert result.activations is not None
    # DenseNet121 backbone: (batch, 1024 channels, H, W)
    assert result.activations.shape[0] == 1
    assert result.activations.shape[1] == 1024


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image not present")
def test_present_property_sorted_descending():
    tensor = ChestXrayPreprocessor().process(SAMPLE)
    result = _clf.predict(tensor)
    probs = [f.probability for f in result.present]
    assert probs == sorted(probs, reverse=True)
    assert all(f.present for f in result.present)

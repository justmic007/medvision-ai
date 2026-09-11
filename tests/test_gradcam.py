"""Tests for GradCAM explainability.

Verifies the heatmap for a finding is a normalized 2D map matching the input
size, that an unknown finding is rejected, and that the overlay renderer writes
a PNG.
"""
from pathlib import Path

import numpy as np
import pytest

from app.services.gradcam import Explainer, GradCAM, render_overlay
from app.services.preprocessing import ChestXrayPreprocessor

SAMPLE = (Path(__file__).parent.parent / "data" / "sample_cxr.jpg").resolve()

_cam = GradCAM()


def test_is_an_explainer():
    assert isinstance(_cam, Explainer)


def test_has_18_pathologies():
    assert len(_cam.pathologies) == 18


def test_unknown_finding_rejected():
    import torch
    dummy = torch.zeros(1, 1, 224, 224)
    with pytest.raises(ValueError):
        _cam.heatmap(dummy, "NotARealFinding")


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image not present")
def test_heatmap_shape_and_range():
    tensor = ChestXrayPreprocessor().process(SAMPLE)
    hm = _cam.heatmap(tensor, "Nodule")
    assert hm.ndim == 2
    assert hm.shape == tuple(tensor.shape[-2:])
    assert hm.min() >= 0.0
    assert hm.max() <= 1.0


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image not present")
def test_heatmap_has_variation():
    # A meaningful heatmap localizes — it isn't a flat constant.
    tensor = ChestXrayPreprocessor().process(SAMPLE)
    hm = _cam.heatmap(tensor, "Nodule")
    assert hm.std() > 0.0


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image not present")
def test_overlay_writes_png(tmp_path):
    tensor = ChestXrayPreprocessor().process(SAMPLE)
    hm = _cam.heatmap(tensor, "Nodule")
    out = tmp_path / "overlay.png"
    render_overlay(tensor, hm, str(out))
    assert out.exists()
    assert out.stat().st_size > 0

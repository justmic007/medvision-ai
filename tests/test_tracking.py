"""Tests for MLflow inference-run tracking.

Verifies a ClassificationResult can be logged as an MLflow run and returns a
run id. Redirects MLflow's tracking URI to a temp store so tests never pollute
the real ./mlruns.
"""
from pathlib import Path

import mlflow
import pytest

from app.services.classifier import ChestXrayClassifier
from app.services.preprocessing import ChestXrayPreprocessor
from app.services.tracking import log_inference

# Absolute path — resolved now, so it survives any working-directory change.
SAMPLE = (Path(__file__).parent.parent / "data" / "sample_cxr.jpg").resolve()


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image not present")
def test_log_inference_returns_run_id(tmp_path):
    # Point MLflow at a temp store so the test doesn't touch the real ./mlruns.
    # No chdir — we redirect via the tracking URI instead, keeping relative
    # paths (like the sample image) valid.
    mlflow.set_tracking_uri(f"file:{tmp_path}/mlruns")

    clf = ChestXrayClassifier()
    result = clf.predict(ChestXrayPreprocessor().process(SAMPLE))

    run_id = log_inference(result, image_id="test_image.jpg")
    assert isinstance(run_id, str)
    assert len(run_id) > 0

    run = mlflow.get_run(run_id)
    assert run.data.params["model_name"] == result.model_name
    assert run.data.params["image_id"] == "test_image.jpg"
    assert run.data.metrics["num_present"] == len(result.present)

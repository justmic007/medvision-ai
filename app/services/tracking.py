"""Inference-run provenance tracking via MLflow.

Records each inference run for reproducibility: which model, which thresholds,
when, and a summary of the result. This is NOT training-experiment tracking —
MedVision does no training (D-07). It logs *inference* provenance, which is the
responsible-AI reason it exists: any result can be traced to exactly how it was
produced.

Decoupled by design: the classifier does inference and knows nothing about
logging. This helper records a ClassificationResult after the fact, so tracking
is optional plumbing around a pure classifier, not baked into it.

Runs are written to the local ./mlruns store (gitignored, D-03).
"""
from __future__ import annotations

from datetime import datetime, timezone

import mlflow

from app.services.classifier import ClassificationResult

EXPERIMENT_NAME = "medvision-inference"


def log_inference(
    result: ClassificationResult,
    image_id: str,
    experiment_name: str = EXPERIMENT_NAME,
) -> str:
    """Log one inference run to MLflow. Returns the MLflow run id.

    Params: model name, image identifier, count of pathologies.
    Metrics: per-finding probability, and number flagged present.
    Tags: UTC timestamp, project phase.
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_param("model_name", result.model_name)
        mlflow.log_param("image_id", image_id)
        mlflow.log_param("num_pathologies", len(result.findings))

        mlflow.log_metric("num_present", len(result.present))
        for f in result.findings:
            # MLflow metric keys can't contain some chars; normalize the label.
            key = "prob_" + f.name.replace(" ", "_").replace("/", "_")
            mlflow.log_metric(key, f.probability)

        mlflow.set_tag("timestamp_utc", datetime.now(timezone.utc).isoformat())
        mlflow.set_tag("phase", "2-inference")

        return run.info.run_id

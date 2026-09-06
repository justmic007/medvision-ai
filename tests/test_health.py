"""Smoke tests for the Phase 0 app skeleton."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["app"] == "MedVision AI"
    assert "version" in body


def test_root_carries_disclaimer():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    # The non-diagnostic disclaimer is a project invariant; assert it is surfaced.
    assert "not a" in body["disclaimer"].lower()
    assert "diagnostic" in body["disclaimer"].lower()

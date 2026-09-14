# MedVision AI

A decision-support research prototype for chest X-ray analysis.

MedVision AI is designed with low-resource settings in mind — contexts where
radiologist access is scarce and a single specialist may serve a large
population, delaying interpretation of routine chest films. It pairs a
multi-pathology chest X-ray classifier with visual explainability (showing
*where* a finding is grounded in the image) and cited literature retrieval
(linking each detected finding to peer-reviewed evidence).

The system is explicitly **non-diagnostic** and keeps a **clinician in the
loop**: it surfaces, the clinician decides. It is a research/educational
prototype, not a validated clinical tool.

> **Disclaimer.** MedVision AI is a research/educational prototype and is NOT a
> diagnostic tool. Outputs are not a substitute for evaluation by a qualified
> clinician.

## System

Frontal chest X-ray in (DICOM or PNG/JPG) → calibrated multi-pathology
predictions + per-finding GradCAM heatmap overlay + cited PubMed literature.
One API, containerized. See `ARCHITECTURE.md` for the full design and
`DECISIONS.md` for the decision log.

## Status

Phase 0 — scaffolding. Runnable containerized FastAPI skeleton with a real
`/health` endpoint and a passing test suite. Imaging, explainability, and
literature layers arrive in Phases 1–5.

## Sample data

Test chest X-rays are public images, not committed to the repo (D-03). Fetch
them into `data/` with:

```bash
bash scripts/fetch_sample_data.sh
```

This downloads a few public, de-identified chest X-rays (from the
TorchXRayVision test set) for local development and the demo.

## Quickstart

### With Docker (primary — D-01)

```bash
docker compose up --build
# in another terminal:
curl http://localhost:8000/health
curl http://localhost:8000/
# interactive docs at http://localhost:8000/docs
```

### Local (without Docker)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Tests

```bash
pip install -r requirements.txt   # includes pytest + httpx
pytest -q
```

## Project structure

medvision-ai/
├── app/
│ ├── main.py # FastAPI entrypoint
│ ├── api/ # routers (thin) — health.py; imaging/rag added later
│ ├── core/ # config, cross-cutting concerns
│ ├── services/ # inference / explainability / RAG logic (later phases)
│ └── schemas/ # pydantic request/response models
├── tests/ # pytest suite
├── data/ # gitignored — public/synthetic only, never committed (D-03)
├── weights/ # gitignored — cached model weights (D-03)
├── mlruns/ # gitignored — local MLflow store (Phase 2)
├── notebooks/ # exploratory / Colab work
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── ARCHITECTURE.md
└── DECISIONS.md


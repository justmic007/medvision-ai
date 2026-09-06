"""MedVision AI — FastAPI application entrypoint.

Phase 0: app skeleton with a real /health endpoint and the project-wide
non-diagnostic disclaimer surfaced at the root. Imaging, explainability,
and literature layers are added in later phases as additional routers under
app/api/, backed by logic in app/services/.
"""
from fastapi import FastAPI

from app.api import health
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Decision-support research prototype for chest X-ray analysis. "
        "Non-diagnostic; clinician-in-the-loop."
    ),
)

app.include_router(health.router)


@app.get("/", tags=["root"])
def root() -> dict:
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "disclaimer": settings.disclaimer,
        "docs": "/docs",
    }

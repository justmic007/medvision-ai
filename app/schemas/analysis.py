"""Response schema for the /analyze endpoint.

This is the cross-repo API contract: the separate Next.js frontend builds
against these shapes. Kept deliberately explicit so the JSON is self-describing.
"""
from pydantic import BaseModel


class ArticleOut(BaseModel):
    pmid: str
    title: str
    journal: str
    year: str
    citation: str
    url: str


class FindingOut(BaseModel):
    name: str
    probability: float
    threshold: float
    # Base64-encoded PNG heatmap overlay, present only for top-N findings.
    # Demo choice (D-14 era): self-contained response, no storage layer.
    # Production would return a storage URL instead (D-10, clinical-workflow phase).
    heatmap_base64: str | None = None
    articles: list[ArticleOut] = []


class AnalysisResponse(BaseModel):
    model_name: str
    num_present: int
    findings: list[FindingOut]
    disclaimer: str

"""Literature grounding: finding -> cited PubMed abstracts.

Takes the *label* of a detected finding and returns real, cited PubMed
references (title, PubMed ID, citation). Retrieval only — no generation (D-14,
D-04). Loosely coupled: needs only the finding name, never the model or image.

Retrieval uses NCBI E-utilities directly (esearch -> IDs, efetch -> abstracts).
Each finding maps to a disambiguated search term via FINDING_QUERIES, since a
bare label ("Effusion") is too vague to search well.

Design (D-11 / D-14): retrieval is behind the LiteratureRetriever interface, so a
vector-store implementation could replace PubMedRetriever later without changing
callers.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from xml.etree import ElementTree as ET

import requests

# Each of the 18 model pathologies -> a focused PubMed query. The "chest
# radiograph"/"chest x-ray" qualifiers keep results in the imaging context and
# disambiguate (e.g. effusion -> pleural, not pericardial).
FINDING_QUERIES: dict[str, str] = {
    "Atelectasis": "atelectasis chest radiograph",
    "Consolidation": "pulmonary consolidation chest radiograph",
    "Infiltration": "\"pulmonary infiltrate\" chest radiograph",
    "Pneumothorax": "pneumothorax chest radiograph",
    "Edema": "pulmonary edema chest radiograph",
    "Emphysema": "emphysema chest radiograph",
    "Fibrosis": "pulmonary fibrosis chest radiograph",
    "Effusion": "pleural effusion chest radiograph",
    "Pneumonia": "pneumonia chest radiograph",
    "Pleural_Thickening": "pleural thickening chest radiograph",
    "Cardiomegaly": "cardiomegaly chest radiograph",
    "Nodule": "pulmonary nodule chest radiograph",
    "Mass": "\"lung mass\" chest radiograph diagnosis",
    "Hernia": "diaphragmatic hernia chest radiograph",
    "Lung Lesion": "\"lung lesion\" chest radiograph diagnosis",
    "Fracture": "rib fracture chest radiograph",
    "Lung Opacity": "\"lung opacity\" chest radiograph diagnosis",
    "Enlarged Cardiomediastinum": "enlarged cardiomediastinum chest radiograph",
}

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@dataclass
class Article:
    """One retrieved reference."""
    pmid: str
    title: str
    journal: str
    year: str

    @property
    def citation(self) -> str:
        return f"{self.title} {self.journal} ({self.year}). PMID: {self.pmid}"

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"


class LiteratureRetriever(ABC):
    """Returns cited references for a finding label."""

    @abstractmethod
    def retrieve(self, finding: str, max_results: int = 3) -> list[Article]:
        ...


class PubMedRetriever(LiteratureRetriever):
    """Direct PubMed retrieval via NCBI E-utilities (D-14).

    Optional api_key raises the NCBI rate limit; keyless works for dev.
    """

    def __init__(self, api_key: str | None = None, timeout: float = 10.0) -> None:
        self._api_key = api_key
        self._timeout = timeout

    def _params(self, **kw) -> dict:
        if self._api_key:
            kw["api_key"] = self._api_key
        return kw

    def _esearch(self, query: str, max_results: int) -> list[str]:
        r = requests.get(
            f"{EUTILS_BASE}/esearch.fcgi",
            params=self._params(
                db="pubmed", term=query, retmax=max_results,
                retmode="json", sort="relevance",
            ),
            timeout=self._timeout,
        )
        r.raise_for_status()
        return r.json().get("esearchresult", {}).get("idlist", [])

    def _efetch(self, pmids: list[str]) -> list[Article]:
        if not pmids:
            return []
        r = requests.get(
            f"{EUTILS_BASE}/efetch.fcgi",
            params=self._params(
                db="pubmed", id=",".join(pmids), retmode="xml"
            ),
            timeout=self._timeout,
        )
        r.raise_for_status()
        return self._parse(r.text)

    @staticmethod
    def _parse(xml_text: str) -> list[Article]:
        root = ET.fromstring(xml_text)
        articles: list[Article] = []
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//PMID") or ""
            title = art.findtext(".//ArticleTitle") or "(no title)"
            journal = art.findtext(".//Journal/Title") or ""
            year = art.findtext(".//PubDate/Year") or ""
            articles.append(
                Article(pmid=pmid, title=title, journal=journal, year=year)
            )
        return articles

    def retrieve(self, finding: str, max_results: int = 3) -> list[Article]:
        query = FINDING_QUERIES.get(finding, f"{finding} chest radiograph")
        pmids = self._esearch(query, max_results)
        # Be polite to NCBI between the two calls.
        time.sleep(0.34)
        return self._efetch(pmids)


class CachedPubMedRetriever(PubMedRetriever):
    """PubMedRetriever with an in-memory cache keyed by (finding, max_results).

    Serves retrieval directly (avoids re-hitting NCBI for a finding already
    fetched, respecting rate limits) and — per D-14 — the accumulated cache is
    the corpus a future vector-store retriever (Option B) would embed.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cache: dict[tuple[str, int], list[Article]] = {}

    def retrieve(self, finding: str, max_results: int = 3) -> list[Article]:
        key = (finding, max_results)
        if key not in self._cache:
            self._cache[key] = super().retrieve(finding, max_results)
        return self._cache[key]

    @property
    def cached_findings(self) -> list[str]:
        return sorted({finding for finding, _ in self._cache})


def ground_findings(
    finding_names: list[str],
    retriever: LiteratureRetriever | None = None,
    max_results: int = 3,
) -> dict[str, list[Article]]:
    """Retrieve cited literature for each finding name.

    Takes the *labels* of present findings (e.g. from a ClassificationResult's
    .present) and returns {finding: [Article, ...]}. This is the loosely-coupled
    grounding step — it never touches the model or image, only names.
    """
    retriever = retriever or CachedPubMedRetriever()
    return {name: retriever.retrieve(name, max_results) for name in finding_names}

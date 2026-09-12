"""Tests for literature grounding.

Network-independent: the NCBI HTTP calls are mocked so tests are deterministic
and don't depend on PubMed being reachable. One optional live test is marked to
run only when explicitly enabled.
"""
import os
from unittest.mock import patch

import pytest

from app.services.literature import (
    Article,
    CachedPubMedRetriever,
    FINDING_QUERIES,
    LiteratureRetriever,
    PubMedRetriever,
    ground_findings,
)

# A minimal fake PubMed efetch XML response with two articles.
FAKE_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>11111111</PMID>
      <Article>
        <ArticleTitle>Pleural effusion on chest radiography</ArticleTitle>
        <Journal><Title>Test Journal</Title></Journal>
      </Article>
    </MedlineCitation>
    <PubmedData><History><PubMedPubDate><Year>2020</Year></PubMedPubDate></History></PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>22222222</PMID>
      <Article>
        <ArticleTitle>Second effusion study</ArticleTitle>
        <Journal><Title>Another Journal</Title></Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


def test_all_18_findings_have_queries():
    # Every model pathology maps to a search term.
    assert len(FINDING_QUERIES) == 18


def test_article_citation_and_url():
    a = Article(pmid="12345", title="A study", journal="J Rad", year="2021")
    assert "12345" in a.citation
    assert a.url == "https://pubmed.ncbi.nlm.nih.gov/12345/"


def test_retriever_is_interface():
    assert isinstance(PubMedRetriever(), LiteratureRetriever)


def test_retrieve_parses_mocked_response():
    r = PubMedRetriever()
    with patch.object(r, "_esearch", return_value=["11111111", "22222222"]), \
         patch("app.services.literature.requests.get") as mock_get:
        mock_get.return_value.text = FAKE_XML
        mock_get.return_value.raise_for_status = lambda: None
        articles = r.retrieve("Effusion", max_results=2)

    assert len(articles) == 2
    assert articles[0].pmid == "11111111"
    assert "effusion" in articles[0].title.lower()


def test_cache_avoids_second_fetch():
    r = CachedPubMedRetriever()
    fake = [Article(pmid="1", title="t", journal="j", year="2020")]
    with patch.object(PubMedRetriever, "retrieve", return_value=fake) as spy:
        r.retrieve("Nodule")
        r.retrieve("Nodule")  # second call should hit cache, not PubMed
    assert spy.call_count == 1
    assert "Nodule" in r.cached_findings


def test_ground_findings_maps_each():
    fake = [Article(pmid="1", title="t", journal="j", year="2020")]
    with patch.object(CachedPubMedRetriever, "retrieve", return_value=fake):
        grounded = ground_findings(["Effusion", "Nodule"])
    assert set(grounded.keys()) == {"Effusion", "Nodule"}
    assert all(len(v) == 1 for v in grounded.values())


@pytest.mark.skipif(
    os.environ.get("MEDVISION_LIVE_TESTS") != "1",
    reason="live PubMed test; set MEDVISION_LIVE_TESTS=1 to run",
)
def test_live_pubmed_retrieval():
    articles = PubMedRetriever().retrieve("Effusion", max_results=2)
    assert len(articles) >= 1
    assert all(a.pmid for a in articles)

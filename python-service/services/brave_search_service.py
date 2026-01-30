"""Brave Search API service for literature review search.

Maps Brave Web Search results to a structured paper-like format compatible
with the frontend SearchPaper type. Enriches with authors, year, journal
via Crossref (DOI), PubMed (PMID), and arXiv APIs. Always includes `url`
so every paper has a link.
"""
import hashlib
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import httpx
import structlog

logger = structlog.get_logger()

BRAVE_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
CROSSREF_WORKS_URL = "https://api.crossref.org/works"
PUBMED_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
ARXIV_QUERY_URL = "http://export.arxiv.org/api/query"

# XML namespaces for arXiv Atom response
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}

# Timeout and concurrency for enrichment
ENRICH_TIMEOUT = 5.0
ENRICH_MAX_WORKERS = 5


def _get_api_key() -> Optional[str]:
    """Return Brave Search API key from environment."""
    import os
    return (os.getenv("BRAVE_SEARCH_API_KEY") or "").strip() or None


def _infer_source(url: str) -> str:
    """Infer paper source from URL for display."""
    url_lower = url.lower()
    if "pubmed" in url_lower or "ncbi.nlm.nih.gov" in url_lower:
        return "PubMed"
    if "biorxiv" in url_lower:
        return "BioRxiv"
    if "medrxiv" in url_lower:
        return "MedRxiv"
    if "arxiv" in url_lower or "doi.org" in url_lower or ".pdf" in url_lower:
        return "Preprint"
    return "Preprint"


def _extract_doi(url: str) -> Optional[str]:
    """Extract DOI from URL if present."""
    m = re.search(r"doi\.org/(10\.\S+)", url, re.IGNORECASE)
    if m:
        return m.group(1).rstrip("/")
    return None


def _extract_pmid(url: str) -> Optional[str]:
    """Extract PMID from PubMed URL if present."""
    m = re.search(r"pubmed.*?[/\-](\d+)", url, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"ncbi\.nlm\.nih\.gov/pubmed/(\d+)", url, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _extract_arxiv_id(url: str) -> Optional[str]:
    """Extract arXiv id from URL (e.g. 1706.03762)."""
    url_lower = url.lower()
    # arxiv.org/abs/1706.03762 or arxiv.org/pdf/1706.03762.pdf
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?", url_lower)
    if m:
        return m.group(1)
    return None


def _extract_year_from_text(text: str) -> int:
    """Try to extract a 4-digit year from snippet text."""
    if not text:
        return 0
    matches = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    return int(matches[0]) if matches else 0


def _strip_html(text: str) -> str:
    """Remove simple HTML tags and entities."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&quot;", '"').replace("&#27;", "'").replace("&amp;", "&")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:5000]


def _enrich_via_crossref(doi: str, client: httpx.Client) -> Optional[Dict[str, Any]]:
    """Fetch authors, year, journal, title from Crossref by DOI. Returns partial update dict or None."""
    try:
        # Crossref expects DOI without url encoding of the path part
        url = f"{CROSSREF_WORKS_URL}/{doi}"
        resp = client.get(url, timeout=ENRICH_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message") or {}
        authors = []
        for a in msg.get("author") or []:
            given = (a.get("given") or "").strip()
            family = (a.get("family") or "").strip()
            name = f"{given} {family}".strip() or family or given
            if name:
                authors.append(name)
        year = 0
        for part in (msg.get("published") or msg.get("published-print") or {}).get("date-parts") or []:
            if part:
                year = int(part[0]) if isinstance(part[0], (int, float)) else 0
                break
        container = msg.get("container-title") or []
        journal = (container[0] or "").strip() if container else ""
        title_list = msg.get("title") or []
        title = (title_list[0] or "").strip() if title_list else ""
        abstract = (msg.get("abstract") or "").strip()
        if abstract:
            abstract = _strip_html(abstract)
        return {
            "authors": authors,
            "year": year,
            "journal": journal,
            "title": title or None,
            "abstract": abstract or None,
        }
    except Exception as e:
        logger.debug("crossref_enrich_failed", doi=doi, error=str(e))
        return None


def _enrich_via_pubmed(pmid: str, client: httpx.Client) -> Optional[Dict[str, Any]]:
    """Fetch authors, year, journal from PubMed E-utilities. Returns partial update dict or None."""
    try:
        params = {"db": "pubmed", "id": pmid, "retmode": "json"}
        resp = client.get(PUBMED_ESUMMARY_URL, params=params, timeout=ENRICH_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        result = (data.get("result") or {}).get(pmid) or {}
        authors = []
        for a in result.get("authors") or []:
            name = (a.get("name") or "").strip()
            if name:
                authors.append(name)
        pubdate = (result.get("pubdate") or "").strip()
        year = 0
        if pubdate:
            m = re.search(r"(\d{4})", pubdate)
            if m:
                year = int(m.group(1))
        journal = (result.get("fulljournalname") or result.get("source") or "").strip()
        title = (result.get("title") or "").strip()
        return {
            "authors": authors,
            "year": year,
            "journal": journal,
            "title": title or None,
        }
    except Exception as e:
        logger.debug("pubmed_enrich_failed", pmid=pmid, error=str(e))
        return None


def _enrich_via_arxiv(arxiv_id: str, client: httpx.Client) -> Optional[Dict[str, Any]]:
    """Fetch authors, year, title from arXiv API. Returns partial update dict or None."""
    try:
        params = {"id_list": arxiv_id}
        resp = client.get(ARXIV_QUERY_URL, params=params, timeout=ENRICH_TIMEOUT)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        authors = []
        year = 0
        title = ""
        # Atom feed: entry with authors and title
        for entry in root.findall(".//atom:entry", ARXIV_NS):
            tit = entry.find("atom:title", ARXIV_NS)
            if tit is not None and tit.text:
                title = _strip_html(tit.text.replace("\n", " ").strip())
            published = entry.find("atom:published", ARXIV_NS)
            if published is not None and published.text:
                m = re.search(r"(\d{4})", published.text)
                if m:
                    year = int(m.group(1))
            for author in entry.findall("atom:author", ARXIV_NS):
                name_el = author.find("atom:name", ARXIV_NS)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())
            break
        return {
            "authors": authors,
            "year": year,
            "title": title or None,
        }
    except Exception as e:
        logger.debug("arxiv_enrich_failed", arxiv_id=arxiv_id, error=str(e))
        return None


def _enrich_paper(paper: Dict[str, Any], client: httpx.Client) -> Dict[str, Any]:
    """Enrich a single paper with authors, year, journal from DOI, PMID, or arXiv. Mutates and returns paper."""
    url = paper.get("url") or ""
    doi = paper.get("doi")
    pmid = paper.get("pmid")
    arxiv_id = _extract_arxiv_id(url) if url else None

    # Prefer DOI (often has best metadata), then PMID, then arXiv
    if doi:
        meta = _enrich_via_crossref(doi, client)
        if meta:
            if meta.get("authors"):
                paper["authors"] = meta["authors"]
            if meta.get("year"):
                paper["year"] = meta["year"]
            if meta.get("journal"):
                paper["journal"] = meta["journal"]
            if meta.get("title"):
                paper["title"] = meta["title"]
            if meta.get("abstract"):
                paper["abstract"] = meta["abstract"]
            return paper

    if pmid:
        meta = _enrich_via_pubmed(pmid, client)
        if meta:
            if meta.get("authors"):
                paper["authors"] = meta["authors"]
            if meta.get("year"):
                paper["year"] = meta["year"]
            if meta.get("journal"):
                paper["journal"] = meta["journal"]
            if meta.get("title"):
                paper["title"] = meta["title"]
            return paper

    if arxiv_id:
        meta = _enrich_via_arxiv(arxiv_id, client)
        if meta:
            if meta.get("authors"):
                paper["authors"] = meta["authors"]
            if meta.get("year"):
                paper["year"] = meta["year"]
            if meta.get("title"):
                paper["title"] = meta["title"]
            # arXiv PDF link
            if not paper.get("pdfUrl"):
                paper["pdfUrl"] = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            return paper

    return paper


def _brave_result_to_paper(result: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Map a single Brave web result to the structured paper format."""
    url = result.get("url") or ""
    title = result.get("title") or "Untitled"
    description = result.get("description") or ""
    extra_snippets = result.get("extra_snippets") or []
    full_abstract = description
    if extra_snippets:
        full_abstract = description + "\n\n" + "\n\n".join(extra_snippets[:3])
    full_abstract = _strip_html(full_abstract)[:5000] if full_abstract else ""

    raw = url if url else f"{title}_{index}"
    id_str = hashlib.sha256(raw.encode()).hexdigest()[:16]

    source = _infer_source(url)
    doi = _extract_doi(url)
    pmid = _extract_pmid(url)
    arxiv_id = _extract_arxiv_id(url)
    year = _extract_year_from_text(full_abstract)

    pdf_url: Optional[str] = None
    if ".pdf" in url.lower():
        pdf_url = url
    elif arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    return {
        "id": id_str,
        "title": title,
        "authors": [],
        "year": year,
        "journal": "",
        "abstract": full_abstract,
        "isOpenAccess": True,
        "doi": doi,
        "pmid": pmid,
        "pdfUrl": pdf_url,
        "url": url,
        "source": source,
    }


def search_literature(
    query: str,
    count: int = 20,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search Brave Web Search for literature-related results and return
    a list of items in the structured paper format. Each paper includes
    `url` (link to the result). Authors, year, and journal are enriched
    from Crossref (DOI), PubMed (PMID), or arXiv when available.

    Returns:
        List of paper-like dicts with id, title, authors, year, journal,
        abstract, isOpenAccess, doi, pmid, pdfUrl, url, source.
    """
    key = api_key or _get_api_key()
    if not key:
        raise ValueError(
            "Brave Search API key is not configured. "
            "Set BRAVE_SEARCH_API_KEY in your environment."
        )

    search_q = query.strip()
    if not search_q:
        return []

    params: Dict[str, Any] = {
        "q": search_q,
        "count": min(count, 20),
    }
    headers = {
        "X-Subscription-Token": key,
        "Accept": "application/json",
    }

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(BRAVE_WEB_SEARCH_URL, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.error("brave_search_http_error", status=e.response.status_code, body=e.response.text[:500])
        raise ValueError(f"Brave Search API error: {e.response.status_code}") from e
    except httpx.RequestError as e:
        logger.error("brave_search_request_error", error=str(e))
        raise ValueError(f"Brave Search request failed: {str(e)}") from e

    web = data.get("web") or {}
    results = web.get("results") or []
    papers = [_brave_result_to_paper(r, i) for i, r in enumerate(results)]

    # Enrich with authors, year, journal from Crossref / PubMed / arXiv
    with httpx.Client(timeout=ENRICH_TIMEOUT) as enrich_client:
        with ThreadPoolExecutor(max_workers=ENRICH_MAX_WORKERS) as executor:
            futures = {executor.submit(_enrich_paper, p, enrich_client): p for p in papers}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.debug("enrich_task_error", error=str(e))

    logger.info("brave_search_done", query=query[:80], result_count=len(papers))
    return papers

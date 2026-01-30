"""Literature review search API.

Exposes an endpoint for literature search using Brave Search API.
Returns results in a structured JSON format compatible with the frontend
SearchPaper type.
"""
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from services.brave_search_service import search_literature

router = APIRouter(prefix="/literature", tags=["literature"])


class LiteratureSearchResponse(BaseModel):
    """Response schema for literature search."""

    papers: List[Dict[str, Any]] = Field(..., description="List of paper-like results")
    totalCount: int = Field(..., description="Number of results returned")
    source: str = Field(default="brave", description="Search source identifier")


@router.get("/search", response_model=LiteratureSearchResponse)
async def literature_search(
    q: str = Query(..., min_length=1, description="Search query for literature topic"),
    limit: int = Query(default=20, ge=1, le=50, description="Max number of results (1–50)"),
) -> LiteratureSearchResponse:
    """
    Search for literature related to a topic using Brave Search.

    Returns a list of results in structured JSON format with id, title,
    authors, year, journal, abstract, isOpenAccess, doi, pmid, pdfUrl, source.
    """
    try:
        papers = search_literature(query=q, count=limit)
        return LiteratureSearchResponse(
            papers=papers,
            totalCount=len(papers),
            source="brave",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Literature search failed: {str(e)}")

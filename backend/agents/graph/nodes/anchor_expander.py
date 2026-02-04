"""
Anchor expansion node for hybrid flow: SQL first, then use SQL results to drive targeted RAG.

Hybrid = SQL (anchors) → expand with targeted retrieval → summarizer.
RAG is never queried blindly when SQL already returned concrete entities.
"""
import time
import structlog
from typing import Dict, Any, List, Optional, Set
from agents.graph.state import AgentState
from agents.graph.nodes.normalize import get_llm_client
from agents.constants import (
    TOOL_SQL,
    TOOL_RAG,
    SQL_ANALYZER_MAX_IDS,
    MAX_ENRICHMENT_QUERIES,
    MAX_ANCHOR_QUERIES_TOTAL,
    ENRICHMENT_MATCH_COUNT,
    ENRICHMENT_THRESHOLD,
    ENRICHMENT_CONTENT_TRUNCATE,
)
from services.rag import RAGService
from services.embedder import EmbeddingService
from services.trace_service import TraceService
from services.config import get_app_config

logger = structlog.get_logger()

_rag_service: Optional[RAGService] = None
_embedder: Optional[EmbeddingService] = None
_trace_service: Optional[TraceService] = None


def _get_rag() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def _get_embedder() -> EmbeddingService:
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingService()
    return _embedder


def get_trace_service() -> TraceService:
    global _trace_service
    if _trace_service is None:
        _trace_service = TraceService()
    return _trace_service


def _analyze_sql_result(sql_result: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze SQL result for anchor expansion: project/experiment IDs, counts, description flags."""
    data = sql_result.get("data") if isinstance(sql_result, dict) else []
    if not data or not isinstance(data, list):
        return {
            "project_ids": [], "experiment_ids": [], "experiment_count": 0, "project_count": 0,
            "has_only_names": True, "has_descriptions": False, "row_count": 0, "column_names": [],
        }
    rows = [r for r in data if isinstance(r, dict)]
    all_keys = set()
    for r in rows:
        all_keys.update(k for k in r if k)
    key_lower_set = {k.lower() for k in all_keys}
    project_id_keys = {k for k in key_lower_set if "project" in k and "id" in k}
    experiment_id_keys = {k for k in key_lower_set if "experiment" in k and "id" in k}
    desc_keys = {k for k in key_lower_set if any(d in k for d in ("description", "content", "abstract", "notes", "hypothesis"))}

    def _collect(rows: List[Dict], candidate_keys: Set[str]) -> List[str]:
        out: Set[str] = set()
        for row in rows:
            for k, v in row.items():
                if k and v is not None and k.lower() in candidate_keys:
                    s = str(v).strip()
                    if s:
                        out.add(s)
        return list(out)

    def _has_desc(rows: List[Dict], keys_lower: Set[str]) -> bool:
        for row in rows:
            for k, v in row.items():
                if k and k.lower() in keys_lower and v is not None and str(v).strip():
                    return True
        return False

    project_ids = _collect(rows, project_id_keys) if project_id_keys else []
    experiment_ids = _collect(rows, experiment_id_keys) if experiment_id_keys else []
    has_descriptions = _has_desc(rows, desc_keys) if desc_keys else False
    return {
        "project_ids": project_ids[:SQL_ANALYZER_MAX_IDS],
        "experiment_ids": experiment_ids[:SQL_ANALYZER_MAX_IDS],
        "experiment_count": len(experiment_ids),
        "project_count": len(project_ids),
        "has_only_names": not has_descriptions,
        "has_descriptions": has_descriptions,
        "row_count": len(rows),
        "column_names": sorted(all_keys),
    }


def _decide_enrichment_needed(anchors: Dict[str, Any], normalized) -> bool:
    """Need targeted RAG when we have project/experiment IDs so we can fetch notes and protocols.
    We always enrich in hybrid when IDs exist, even if SQL returned description columns,
    so the user gets lab notes and literature context, not just DB descriptions."""
    if not anchors:
        return False
    has_ids = (anchors.get("experiment_count", 0) > 0 or anchors.get("project_count", 0) > 0)
    return bool(has_ids)


def _generate_enrichment_queries(
    anchors: Dict[str, Any],
    user_query: str,
    llm_client,
) -> List[Dict[str, Any]]:
    """LLM-assisted but constrained: generate follow-up queries with allowed IDs."""
    project_ids = anchors.get("project_ids", [])[:5]
    experiment_ids = anchors.get("experiment_ids", [])[:5]
    if not project_ids and not experiment_ids:
        return []
    prompt = f"""The user asked: "{user_query}"

We have SQL results with these project and experiment identifiers. We need to fetch lab notes, protocols, and literature for follow-up summaries.
- project_ids: {project_ids}
- experiment_ids: {experiment_ids}

Generate 1 to {MAX_ENRICHMENT_QUERIES} short follow-up queries to fetch meaningful context (notes, protocols, experiment details) for these entities. Each query should be targeted, e.g. "notes and protocols for experiment", "lab notes and summary for project", "experiment methods and results".

Return a JSON array of objects. Each object must have:
- "query_text": string (short search phrase)
- "experiment_id": null or one UUID from {experiment_ids[:3]}
- "project_id": null or one UUID from {project_ids[:3]}

Use at most {MAX_ENRICHMENT_QUERIES} objects. Prefer experiment_id when both apply. Return ONLY the JSON array, no markdown."""

    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "query_text": {"type": "string"},
                "experiment_id": {"type": ["string", "null"]},
                "project_id": {"type": ["string", "null"]},
            },
            "required": ["query_text"],
        },
    }
    try:
        result = llm_client.complete_json(prompt, schema, temperature=0.2)
        if isinstance(result, list):
            queries = result[:MAX_ENRICHMENT_QUERIES]
        elif isinstance(result, dict) and "queries" in result:
            queries = result["queries"][:MAX_ENRICHMENT_QUERIES]
        else:
            queries = []
    except Exception as e:
        logger.warning("anchor_expander: LLM enrichment queries failed", error=str(e))
        queries = []
    # Ensure we have at least one RAG fetch per experiment_id and project_id so textual content is always requested
    seen_exp: set = set()
    seen_proj: set = set()
    for q in queries:
        if q.get("experiment_id"):
            seen_exp.add(str(q["experiment_id"]))
        if q.get("project_id"):
            seen_proj.add(str(q["project_id"]))
    for eid in experiment_ids:
        if eid and str(eid) not in seen_exp:
            queries.append({"query_text": "lab notes and experiment content", "experiment_id": eid, "project_id": None})
            seen_exp.add(str(eid))
    for pid in project_ids:
        if pid and str(pid) not in seen_proj:
            queries.append({"query_text": "lab notes and project summary", "project_id": pid, "experiment_id": None})
            seen_proj.add(str(pid))
    return queries[:MAX_ANCHOR_QUERIES_TOTAL]


def anchor_expander_node(state: AgentState) -> AgentState:
    """
    After SQL in hybrid flow: analyze result, optionally run targeted RAG, then continue to summarizer.
    """
    start_time = time.time()
    sql_result = state.get("sql_result")
    router = state.get("router_decision")
    normalized = state.get("normalized_query")
    request = state.get("request")
    run_id = state.get("run_id")
    trace = get_trace_service()

    tools = router.tools if router and hasattr(router, "tools") else []
    is_hybrid = TOOL_SQL in tools and TOOL_RAG in tools
    has_sql_data = (
        sql_result
        and isinstance(sql_result, dict)
        and not sql_result.get("error")
        and (sql_result.get("row_count", 0) or 0) > 0
        and sql_result.get("data")
    )

    if not has_sql_data:
        state["sql_anchors"] = _analyze_sql_result(sql_result or {})
        state["enriched_context"] = []
        return state

    anchors = _analyze_sql_result(sql_result)
    state["sql_anchors"] = anchors
    state["enriched_context"] = []

    if not is_hybrid:
        return state

    enrichment_needed = _decide_enrichment_needed(anchors, normalized)
    if not enrichment_needed:
        logger.info("anchor_expander: enrichment not needed", run_id=run_id, has_descriptions=anchors.get("has_descriptions"))
        return state

    user_id = request.get("user_id", "") if isinstance(request, dict) else getattr(request, "user_id", "")
    user_query = request.get("query", "") if isinstance(request, dict) else getattr(request, "query", "")
    if not user_id:
        return state

    llm_client = get_llm_client()
    queries = _generate_enrichment_queries(anchors, user_query, llm_client)
    if not queries:
        return state

    rag_service = _get_rag()
    embedder = _get_embedder()
    app_cfg = get_app_config()
    threshold = getattr(app_cfg, "agent_enrichment_threshold", ENRICHMENT_THRESHOLD)
    enriched: List[Dict[str, Any]] = []
    seen_keys: set = set()

    for q in queries:
        query_text = q.get("query_text") or "notes and context"
        exp_id = q.get("experiment_id") if q.get("experiment_id") else None
        proj_id = q.get("project_id") if q.get("project_id") else None
        try:
            emb = embedder.embed_text(query_text)
            if not emb:
                continue
            chunks = rag_service.search_chunks(
                query_embedding=emb,
                user_id=user_id,
                project_id=proj_id,
                experiment_id=exp_id,
                match_threshold=threshold,
                match_count=ENRICHMENT_MATCH_COUNT,
                return_below_threshold_for_entity=True,
            )
            for c in chunks:
                key = (c.get("source_id"), c.get("chunk_index"))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                enriched.append({
                    "chunk_id": c.get("id"),
                    "source_type": c.get("source_type"),
                    "source_id": c.get("source_id"),
                    "experiment_id": c.get("experiment_id"),
                    "content": (c.get("content") or "")[:ENRICHMENT_CONTENT_TRUNCATE],
                    "similarity": c.get("similarity", 0.0),
                    "metadata": c.get("metadata", {}),
                    "from_anchor_expansion": True,
                })
        except Exception as e:
            logger.warning("anchor_expander: RAG call failed", error=str(e), query=query_text[:50])

    state["enriched_context"] = enriched
    existing_rag = state.get("rag_result") or []
    state["rag_result"] = existing_rag + enriched
    if enriched:
        attempted = state.get("attempted_tools") or []
        if TOOL_RAG not in attempted:
            attempted = list(attempted) + [TOOL_RAG]
            state["attempted_tools"] = attempted

    latency_ms = int((time.time() - start_time) * 1000)
    logger.info(
        "anchor_expander completed",
        run_id=run_id,
        enrichment_queries=len(queries),
        enriched_chunks=len(enriched),
        latency_ms=latency_ms,
    )
    if run_id:
        try:
            trace.log_event(
                run_id=run_id,
                node_name="anchor_expander",
                event_type="output",
                payload={"enrichment_queries": len(queries), "enriched_chunks": len(enriched)},
                latency_ms=latency_ms,
            )
        except Exception:
            pass
    return state

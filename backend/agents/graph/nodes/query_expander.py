"""Query expander node for multi-query RAG.

Given the normalized query and (optional) history summary, generate a small set
of alternate phrasings or sub-questions that can improve recall in RAG. These
are stored in state["expanded_queries"] and consumed by the RAG node.
"""
import time
from typing import List, Dict, Any
import structlog

from agents.graph.state import AgentState
from agents.graph.stream_utils import emit_stream_event
from agents.graph.nodes.normalize import get_llm_client
from services.trace_service import TraceService
from agents.services.thinking_logger import get_thinking_logger

logger = structlog.get_logger()

_trace_service: TraceService = None


def get_trace_service() -> TraceService:
    """Get or create trace service singleton."""
    global _trace_service
    if _trace_service is None:
        _trace_service = TraceService()
    return _trace_service


def query_expander_node(state: AgentState) -> AgentState:
    """Optionally expand the normalized query into a few related queries for multi-query retrieval."""
    emit_stream_event(
        state,
        "thinking",
        {"node": "query_expander", "status": "started", "message": "Exploring related angles..."},
    )
    start_time = time.time()
    normalized = state.get("normalized_query")
    request = state.get("request")
    run_id = state.get("run_id")
    trace_service = get_trace_service()

    if not normalized:
        # Nothing to expand; continue without changes
        state["expanded_queries"] = None
        return state

    base_query = normalized.normalized_query
    history_summary = getattr(normalized, "history_summary", None)

    intent = getattr(normalized, "intent", "search")
    # Only expand for search/hybrid/detail intents; aggregate doesn't benefit much.
    if intent not in ("search", "hybrid", "detail"):
        state["expanded_queries"] = None
        return state

    logger.info(
        "query_expander_node started",
        agent_node="query_expander",
        run_id=run_id,
        normalized_query=base_query[:100],
        intent=intent,
    )

    if run_id:
        try:
            trace_service.log_event(
                run_id=run_id,
                node_name="query_expander",
                event_type="input",
                payload={"normalized_query": base_query[:200], "intent": intent},
            )
        except Exception:
            pass

    try:
        llm_client = get_llm_client()
        prompt = f"""You are helping a retrieval system for a scientific lab assistant.

Base normalized query:
\"\"\"{base_query}\"\"\"

Conversation context summary (optional):
{history_summary or "None"}

Your task:
- Generate up to 3 alternate phrasings or sub-questions that, if used as additional search queries, would help find more relevant documents for this user.
- Focus on variations that:
  - Expand abbreviations or acronyms.
  - Include key related entities (projects, experiments, methods).
  - Capture different ways a scientist might describe the same question.
- Do NOT drift outside the user’s intent.
- When the base query is already very specific, you may return an empty list.

Return ONLY valid JSON:
{{
  "queries": ["alt query 1", "alt query 2"]
}}"""

        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "queries": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["queries"],
        }

        result = llm_client.complete_json(prompt, schema)
        queries: List[str] = []
        raw = result.get("queries") if isinstance(result, dict) else None
        if isinstance(raw, list):
            for q in raw:
                if isinstance(q, str):
                    q_clean = q.strip()
                    if q_clean and q_clean.lower() != base_query.lower():
                        queries.append(q_clean)

        # Hard cap to avoid excessive fan-out
        queries = queries[:3]
        state["expanded_queries"] = queries or None

        thinking_logger = get_thinking_logger()
        if run_id and queries:
            thinking_logger.log_analysis(
                run_id=run_id,
                node_name="query_expander",
                analysis=f"Generated {len(queries)} expanded queries for RAG",
                data_summary={"expanded_queries": queries},
                insights=[],
            )

        latency_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "query_expander_node completed",
            agent_node="query_expander",
            run_id=run_id,
            expanded_queries=len(queries),
            latency_ms=round(latency_ms, 2),
        )
        if run_id:
            try:
                trace_service.log_event(
                    run_id=run_id,
                    node_name="query_expander",
                    event_type="output",
                    payload={"expanded_queries": queries},
                    latency_ms=latency_ms,
                )
            except Exception:
                pass

        request_dict: Dict[str, Any] = (
            request if isinstance(request, dict) else getattr(request, "__dict__", {})
        )
        options = request_dict.get("options", {})
        if isinstance(options, dict) and options.get("debug"):
            trace = state.get("trace") or []
            trace.append(
                {
                    "node": "query_expander",
                    "input": {"normalized_query": base_query[:200]},
                    "output": {"expanded_queries": queries},
                    "latency_ms": round(latency_ms, 2),
                }
            )
            state["trace"] = trace

        return state

    except Exception as e:
        logger.warning("query_expander_node failed", run_id=run_id, error=str(e))
        state["expanded_queries"] = None
        if run_id:
            try:
                trace_service.log_event(
                    run_id=run_id,
                    node_name="query_expander",
                    event_type="error",
                    payload={"error": str(e)},
                )
            except Exception:
                pass
        return state


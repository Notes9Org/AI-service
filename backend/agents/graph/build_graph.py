"""Build LangGraph agent graph."""
from langgraph.graph import StateGraph, END
import structlog

from agents.graph.state import AgentState
from agents.constants import (
    TOOL_SQL,
    TOOL_RAG,
    ROUTE_OUT_OF_SCOPE,
    VERDICT_PASS,
    DEFAULT_MAX_RETRIES,
    RECURSION_LIMIT,
    ENTITY_KEYS_FOR_SQL_FALLBACK,
)
from services.config import get_app_config
from agents.graph.nodes.normalize import normalize_node
from agents.graph.nodes.router import router_node
from agents.graph.nodes.sql import sql_node
from agents.graph.nodes.rag import rag_node
from agents.graph.nodes.summarizer import summarizer_node
from agents.graph.nodes.judge import judge_node
from agents.graph.nodes.retry import retry_node
from agents.graph.nodes.final import final_node
from agents.graph.nodes.anchor_expander import anchor_expander_node

logger = structlog.get_logger()


def should_retry(state: AgentState) -> str:
    """Conditional edge: should retry or go to final?"""
    # Early exit if final_response already set (error case)
    if state.get("final_response"):
        return "final"
    
    judge = state.get("judge_result")
    retry_count = state.get("retry_count", 0)
    request = state.get("request", {})
    
    # Handle both dict and object access
    if isinstance(request, dict):
        options = request.get("options", {})
    else:
        options = getattr(request, "options", {}) if hasattr(request, "options") else {}
    
    app_cfg = get_app_config()
    max_retries = options.get("max_retries", getattr(app_cfg, "agent_max_retries", DEFAULT_MAX_RETRIES)) if isinstance(options, dict) else getattr(options, "max_retries", getattr(app_cfg, "agent_max_retries", DEFAULT_MAX_RETRIES))
    
    # Skipped judge (last-attempt shortcut): go to final with current summary
    if judge is None and state.get("summary"):
        return "final"
    
    # If judge passed, go to final
    if judge and isinstance(judge, dict) and judge.get("verdict") == VERDICT_PASS:
        return "final"
    
    # If max retries reached, go to final
    if retry_count >= max_retries:
        return "final"
    
    # Otherwise, retry (go back to router)
    return "router"


def build_agent_graph() -> StateGraph:
    """
    Build and compile LangGraph agent graph.
    
    Structure:
    START → normalize → router → [sql | rag | summarizer] → (anchor_expander if enabled) → summarizer
           → [judge | retry] → retry → [router | final] → END
    
    Data flow (only relevant information with complete context to the LLM):
    - Normalize: query → intent, normalized_query, entities.
    - Router: routes using intent and normalized_query.
    - SQL: runs with query + schema; appends each run to sql_runs (query, generated_sql, data, row_count).
    - RAG: when state has project_id/experiment_id from SQL (initial or retry), fetches by UUID filter first (so
      descriptions for those entities are retrieved); then merges with semantic search. Appends chunks to rag_chunks_all.
    - On retry: SQL/RAG append to sql_runs / rag_chunks_all (never replace).
    - Summarizer: reads sql_runs (merged facts) and rag_chunks_all (relevance-sorted excerpts), plus enriched_context;
      sends one prompt with user query + all relevant facts + relevant excerpts so the LLM has complete context.
    
    Production levers (config):
    - AGENT_JUDGE_ENABLED=false: skip judge node (1 fewer LLM call, faster).
    - AGENT_ANCHOR_EXPANSION_ENABLED=false: skip anchor_expander in hybrid (faster when SQL has data).
    - Entity keys for SQL fallback: agents.constants.ENTITY_KEYS_FOR_SQL_FALLBACK (single source of truth).
    """
    logger.info("Building agent graph")
    
    # Create graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("normalize", normalize_node)
    graph.add_node("router", router_node)
    graph.add_node("sql", sql_node)
    graph.add_node("rag", rag_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("judge", judge_node)
    graph.add_node("anchor_expander", anchor_expander_node)
    graph.add_node("retry", retry_node)
    graph.add_node("final", final_node)
    
    # Add edges
    graph.set_entry_point("normalize")
    
    # normalize → final (if out-of-scope response set) or router
    def route_after_normalize(state: AgentState) -> str:
        """If normalize set final_response (e.g. out-of-scope), go to final; else router."""
        if state.get("final_response"):
            return "final"
        return "router"
    
    graph.add_conditional_edges(
        "normalize",
        route_after_normalize,
        {"final": "final", "router": "router"}
    )
    
    # router → final (out_of_scope) or tools (conditional routing)
    def route_after_router(state: AgentState) -> str:
        """Route after router: final if out_of_scope (response already set), or sql/rag/summarizer."""
        if state.get("final_response"):
            return "final"
        router = state.get("router_decision")
        if not router:
            return "summarizer"
        route = getattr(router, "route", None) if hasattr(router, "route") else (router.get("route") if isinstance(router, dict) else None)
        if route == ROUTE_OUT_OF_SCOPE:
            return "final"
        tools = router.tools if hasattr(router, "tools") else (router.get("tools", []) if isinstance(router, dict) else [])
        if TOOL_SQL in tools:
            return "sql"
        if TOOL_RAG in tools:
            return "rag"
        return "summarizer"
    
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "final": "final",
            "sql": "sql",
            "rag": "rag",
            "summarizer": "summarizer"
        }
    )
    
    # SQL → RAG when in tools (so semantic search runs), else anchor_expander or summarizer
    def route_after_sql(state: AgentState) -> str:
        """After SQL: sql_empty → rag; when RAG in tools → rag; else anchor_expander or summarizer."""
        router = state.get("router_decision")
        flags = state.get("flags") or {}
        if flags.get("sql_empty"):
            return "rag"
        tools = router.tools if router and hasattr(router, "tools") else (router.get("tools", []) if isinstance(router, dict) else [])
        has_rag = TOOL_RAG in tools
        has_sql = TOOL_SQL in tools
        sql_result = state.get("sql_result")
        has_data = (
            sql_result
            and not sql_result.get("error")
            and (sql_result.get("row_count", 0) or 0) > 0
        )
        anchor_enabled = getattr(get_app_config(), "agent_anchor_expansion_enabled", True)
        # When router selected RAG (detail/hybrid), run RAG so semantic search is used
        if has_rag:
            return "rag"
        # When only SQL was used and we have data, optionally expand anchors
        if has_sql and has_data and anchor_enabled:
            return "anchor_expander"
        return "summarizer"
    
    graph.add_conditional_edges(
        "sql",
        route_after_sql,
        {
            "anchor_expander": "anchor_expander",
            "rag": "rag",
            "summarizer": "summarizer"
        }
    )
    
    graph.add_edge("anchor_expander", "summarizer")
    
    # RAG → summarizer, or (if rag_weak) SQL fallback
    def route_after_rag(state: AgentState) -> str:
        """After RAG: if rag_weak and SQL not tried yet and has entities → sql; else summarizer.
        Avoid infinite loop: if we already ran SQL (from prior RAG-weak fallback), go to summarizer."""
        flags = state.get("flags") or {}
        if not flags.get("rag_weak"):
            return "summarizer"
        # Already tried SQL fallback? Don't loop — go to summarizer with what we have
        sql_runs = state.get("sql_runs") or []
        sql_result = state.get("sql_result")
        has_sql_data = (
            (sql_result and not sql_result.get("error") and (sql_result.get("row_count", 0) or 0) > 0)
            or (len(sql_runs) > 0 and any(not r.get("error") and (r.get("row_count", 0) or 0) > 0 for r in sql_runs if isinstance(r, dict)))
        )
        if has_sql_data:
            return "summarizer"
        # Named document + rag_weak → fetch full content via SQL (fast path)
        normalized = state.get("normalized_query")
        entities = getattr(normalized, "entities", {}) if normalized else {}
        if isinstance(entities, dict) and (entities.get("lab_note_titles") or entities.get("protocol_names")):
            # Upgrade intent to aggregate so SQL fetches full content
            if normalized and normalized.intent == "search":
                try:
                    copy_fn = getattr(normalized, "model_copy", None) or getattr(normalized, "copy", None)
                    if copy_fn:
                        state["normalized_query"] = copy_fn(update={
                            "intent": "aggregate",
                            "context": {**(normalized.context or {}), "requires_aggregation": True},
                        })
                except Exception:
                    pass
            return "sql"
        # General entity fallback
        router = state.get("router_decision")
        tools = router.tools if router and hasattr(router, "tools") else (router.get("tools", []) if isinstance(router, dict) else [])
        if TOOL_SQL not in tools:
            if isinstance(entities, dict) and any(entities.get(k) for k in ENTITY_KEYS_FOR_SQL_FALLBACK):
                return "sql"
        return "summarizer"
    
    graph.add_conditional_edges(
        "rag",
        route_after_rag,
        {
            "sql": "sql",
            "summarizer": "summarizer"
        }
    )
    
    # summarizer → judge or retry (skip judge when disabled, or on last attempt to save LLM call)
    def route_after_summarizer(state: AgentState) -> str:
        """If judge disabled: go to retry. Else on last attempt skip judge; else go to judge."""
        app_cfg = get_app_config()
        if not getattr(app_cfg, "agent_judge_enabled", True):
            return "retry"
        request = state.get("request", {})
        if isinstance(request, dict):
            options = request.get("options", {})
        else:
            options = getattr(request, "options", {}) if hasattr(request, "options") else {}
        max_retries = options.get("max_retries", getattr(app_cfg, "agent_max_retries", DEFAULT_MAX_RETRIES)) if isinstance(options, dict) else getattr(options, "max_retries", getattr(app_cfg, "agent_max_retries", DEFAULT_MAX_RETRIES))
        retry_count = state.get("retry_count", 0)
        if retry_count >= max_retries - 1:
            return "retry"
        return "judge"
    
    graph.add_conditional_edges(
        "summarizer",
        route_after_summarizer,
        {"judge": "judge", "retry": "retry"}
    )
    
    # judge → retry (always)
    graph.add_edge("judge", "retry")
    
    # retry → router or final (conditional)
    graph.add_conditional_edges(
        "retry",
        should_retry,
        {
            "router": "router",
            "final": "final"
        }
    )
    
    # final → END (always)
    graph.add_edge("final", END)
    
    # Compile graph with recursion limit
    compiled_graph = graph.compile()
    
    # Set recursion limit to prevent infinite loops
    # This prevents the graph from running indefinitely if nodes keep failing
    compiled_graph = compiled_graph.with_config({"recursion_limit": getattr(get_app_config(), "agent_recursion_limit", RECURSION_LIMIT)})
    
    logger.info("Agent graph built and compiled")
    
    return compiled_graph
"""Agent state TypedDict for LangGraph."""
from typing import TypedDict, Optional, List, Dict, Any, Callable
from agents.contracts.request import AgentRequest
from agents.contracts.normalized import NormalizedQuery
from agents.contracts.router import RouterDecision
from agents.contracts.response import FinalResponse


def _noop_stream_callback(event_type: str, data: Dict[str, Any]) -> None:
    """Default no-op when streaming is disabled."""
    pass


class AgentState(TypedDict):
    """State passed between LangGraph nodes."""
    # Trace tracking
    run_id: str  # UUID string for trace correlation
    
    # Input
    request: AgentRequest
    
    # Normalization
    normalized_query: Optional[NormalizedQuery]
    # Optional expanded queries for multi-query retrieval (derived from normalized_query/history)
    expanded_queries: Optional[List[str]]
    
    # Routing
    router_decision: Optional[RouterDecision]
    
    # Tool results (latest run; used by routing and flags)
    sql_result: Optional[Dict[str, Any]]
    rag_result: Optional[List[Dict[str, Any]]]
    
    # Accumulated across attempts: every SQL run and every RAG chunk (append on retry)
    sql_runs: Optional[List[Dict[str, Any]]]  # [{query, normalized_query, generated_sql, data, row_count, ...}]
    rag_chunks_all: Optional[List[Dict[str, Any]]]  # all RAG chunks from all runs, for complete context
    
    # Hybrid: SQL anchors (programmatic analysis) and targeted enrichment
    sql_anchors: Optional[Dict[str, Any]]  # project_ids, experiment_ids, has_only_names, etc.
    enriched_context: Optional[List[Dict[str, Any]]]  # targeted RAG/SQL follow-up results
    
    # Synthesis
    summary: Optional[Dict[str, Any]]  # {answer, citations}
    
    # Validation
    judge_result: Optional[Dict[str, Any]]  # {verdict, confidence, issues, suggested_revision}
    
    # Retry control
    retry_count: int
    
    # Best answer across retries (never cleared); used by final when returning after max retries
    best_summary: Optional[Dict[str, Any]]
    best_judge_result: Optional[Dict[str, Any]]
    best_tool_used: Optional[str]  # "sql" | "rag" | "hybrid"
    
    # Strategy-aware retry: what we tried and why we failed
    attempted_tools: Optional[List[str]]  # e.g. ["sql", "rag"] so we don't ping-pong
    retry_context: Optional[Dict[str, Any]]  # attempt, failure_reason, rewrite_hint
    
    # Post-tool quality gates (fallback routing)
    flags: Optional[Dict[str, Any]]  # e.g. sql_empty, rag_weak
    
    # Output
    final_response: Optional[FinalResponse]
    
    # Run-wide: all phases and citations for this run_id (including retries)
    run_process_log: Optional[List[Dict[str, Any]]]  # [{phase, attempt, ...}, ...]
    run_citations: Optional[List[Dict[str, Any]]]  # [{source_type, source_id, relevance, ...}, ...]
    
    # Debug trace
    trace: List[Dict[str, Any]]  # Execution trace for debugging

    # Streaming: optional callback(event_type, data) for SSE; called by nodes during execution
    stream_callback: Optional[Callable[[str, Dict[str, Any]], None]]
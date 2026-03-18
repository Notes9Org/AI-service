"""SQL execution node with dynamic SQL generation."""
import time
import structlog
from agents.graph.state import AgentState
from agents.graph.stream_utils import emit_stream_event
from agents.constants import TOOL_SQL, ENTITY_KEYS_FOR_SQL_FALLBACK
from agents.services.sql_service import SQLService
from services.trace_service import TraceService
from agents.services.thinking_logger import get_thinking_logger

logger = structlog.get_logger()

# Singleton services
_sql_service: SQLService = None
_trace_service: TraceService = None


def get_sql_service() -> SQLService:
    """Get or create SQL service singleton."""
    global _sql_service
    if _sql_service is None:
        _sql_service = SQLService()
    return _sql_service


def get_trace_service() -> TraceService:
    """Get or create trace service singleton."""
    global _trace_service
    if _trace_service is None:
        _trace_service = TraceService()
    return _trace_service


def _has_structured_entities(normalized) -> bool:
    """True if normalized query has entities that could support a SQL fallback."""
    if not normalized:
        return False
    entities = getattr(normalized, "entities", {}) or {}
    if not isinstance(entities, dict):
        return False
    return any(entities.get(k) for k in ENTITY_KEYS_FOR_SQL_FALLBACK)


def sql_node(state: AgentState) -> AgentState:
    """Execute SQL tool: generate and execute SQL queries using LLM."""
    emit_stream_event(state, "thinking", {"node": "sql", "status": "started", "message": "Querying database..."})
    start_time = time.time()
    router = state.get("router_decision")
    normalized = state.get("normalized_query")
    request = state["request"]
    run_id = state.get("run_id")
    trace_service = get_trace_service()
    flags = state.get("flags") or {}
    
    # Run if router selected SQL, or if RAG-weak fallback (we're retrying with SQL)
    run_sql = router and TOOL_SQL in router.tools
    if not run_sql and flags.get("rag_weak") and _has_structured_entities(normalized):
        run_sql = True
        logger.info("sql_node: running as RAG-weak fallback", run_id=run_id)
    if not run_sql:
        return state
    
    logger.info(
        "sql_node started",
        agent_node="sql",
        run_id=run_id,
        intent=normalized.intent if normalized else None,
        normalized_query=normalized.normalized_query[:100] if normalized else None,
        payload={
            "input_query": request.get("query", "") if isinstance(request, dict) else getattr(request, "query", ""),
            "input_intent": normalized.intent if normalized else None,
            "input_normalized_query": normalized.normalized_query if normalized else None,
            "input_entities": {k: (str(v)[:100] if isinstance(v, (list, dict)) else v) for k, v in list((normalized.entities if normalized else {}).items())[:5]}
        }
    )
    
    # Log input event
    if run_id:
        try:
            trace_service.log_event(
                run_id=run_id,
                node_name="sql",
                event_type="input",
                payload={
                    "query": request.get("query", "")[:200],
                    "normalized_query": normalized.normalized_query[:200] if normalized else None
                }
            )
        except Exception:
            pass
    
    try:
        sql_service = get_sql_service()
        
        original_query = request.get("query", "") if isinstance(request, dict) else getattr(request, "query", "")
        normalized_query_text = normalized.normalized_query if normalized else original_query
        entities = normalized.entities if normalized else {}
        user_id = request.get("user_id", "") if isinstance(request, dict) else getattr(request, "user_id", "")
        
        if not user_id:
            logger.error("SQL generation: user_id required for data isolation", run_id=run_id)
            state["sql_result"] = {
                "data": [], "row_count": 0, "error": "user_id required for data isolation", "execution_time_ms": 0
            }
            return state
        
        result = sql_service.generate_and_execute(
            query=original_query, user_id=user_id, normalized_query=normalized_query_text,
            entities=entities, scope={}
        )
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Extract generated SQL for debugging
        generated_sql = result.get("generated_sql", "")
        sql_preview = generated_sql[:200] + "..." if len(generated_sql) > 200 else generated_sql
        
        logger.info("sql_node completed", agent_node="sql", run_id=run_id,
                   row_count=result.get("row_count", 0), latency_ms=round(latency_ms, 2),
                   payload={
                       "input_query": original_query[:200], 
                       "input_intent": normalized.intent if normalized else None,
                       "output_row_count": result.get("row_count", 0), 
                       "output_has_error": "error" in result,
                       "output_execution_time_ms": result.get("execution_time_ms", 0),
                       "output_generated_sql": generated_sql  # Full SQL for debugging
                   },
                   sql_preview=sql_preview,
                   sql_full=generated_sql)
        state["sql_result"] = result
        # Emit SQL and completed thinking for streaming UI (show query + status while loading)
        if generated_sql:
            emit_stream_event(state, "thinking", {
                "node": "sql",
                "status": "completed",
                "message": "Query executed",
            })
        # Accumulate this run for summarizer (complete context across retries)
        sql_runs = state.get("sql_runs") or []
        sql_runs = list(sql_runs) + [{
            "query": original_query,
            "normalized_query": normalized_query_text,
            "generated_sql": result.get("generated_sql", ""),
            "data": result.get("data", []),
            "row_count": result.get("row_count", 0),
            "error": result.get("error"),
            "execution_time_ms": result.get("execution_time_ms", 0),
        }]
        state["sql_runs"] = sql_runs
        attempted = state.get("attempted_tools") or []
        if TOOL_SQL not in attempted:
            attempted.append(TOOL_SQL)
            state["attempted_tools"] = attempted

        run_log = state.get("run_process_log") or []
        run_log.append({
            "phase": "sql",
            "attempt": state.get("retry_count", 0),
            "row_count": result.get("row_count", 0),
            "has_error": "error" in result,
            "latency_ms": latency_ms,
        })
        state["run_process_log"] = run_log
        run_cits = state.get("run_citations") or []
        if result.get("row_count", 0) > 0 and not result.get("error"):
            run_cits.append({"source_type": "sql", "source_id": run_id or "query", "relevance": 1.0, "chunk_id": None, "excerpt": None})
        state["run_citations"] = run_cits

        # SQL-empty fallback: only SQL was selected, 0 rows, no error → try RAG next
        tools = router.tools if hasattr(router, "tools") else []
        if (
            tools == [TOOL_SQL]
            and result.get("row_count", 0) == 0
            and not result.get("error")
        ):
            flags = state.get("flags") or {}
            flags["sql_empty"] = True
            state["flags"] = flags
            logger.info("sql_node: sql_empty flag set, will fallback to RAG", run_id=run_id)
        
        thinking_logger = get_thinking_logger()
        if run_id:
            has_error = "error" in result
            thinking_logger.log_analysis(
                run_id=run_id, node_name="sql",
                analysis=f"Executed SQL query: {'Success' if not has_error else 'Failed'}",
                data_summary={"row_count": result.get("row_count", 0), "has_error": has_error},
                insights=[f"Query returned {result.get('row_count', 0)} rows"] if not has_error else []
            )
        
        if run_id:
            try:
                trace_service.log_event(
                    run_id=run_id, node_name="sql", event_type="output",
                    payload={"row_count": result.get("row_count", 0), "has_error": "error" in result},
                    latency_ms=latency_ms
                )
            except Exception:
                pass
        
        options = request.get("options", {}) if isinstance(request, dict) else getattr(request, "options", {}) if hasattr(request, "options") else {}
        if (isinstance(options, dict) and options.get("debug")) or (hasattr(options, "debug") and getattr(options, "debug", False)):
            state["trace"].append({
                "node": "sql", "input": {"query": original_query[:200]},
                "output": {"row_count": result.get("row_count", 0)}, "latency_ms": round(latency_ms, 2)
            })
        
        return state
        
    except Exception as e:
        logger.error("sql_node failed", run_id=run_id, error=str(e))
        
        if run_id:
            try:
                trace_service.log_event(run_id=run_id, node_name="sql", event_type="error",
                                      payload={"error": str(e)})
            except Exception:
                pass
        
        state["sql_result"] = {"data": [], "row_count": 0, "error": str(e), "execution_time_ms": 0}
        return state
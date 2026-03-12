"""Retry node for ReAct-style refinement. Handles query rewrite inline when judge suggested entities/clarity."""
import structlog
from typing import Dict, List
from agents.graph.state import AgentState
from agents.constants import (
    TOOL_SQL,
    TOOL_RAG,
    TOOL_HYBRID,
    VERDICT_PASS,
    DEFAULT_MAX_RETRIES,
)
from services.trace_service import TraceService
from services.config import get_app_config
from agents.services.thinking_logger import get_thinking_logger
from agents.graph.nodes.normalize import get_llm_client

logger = structlog.get_logger()

# Columns whose values we treat as entities for query enrichment (e.g. project_name, experiment_name)
SQL_ENTITY_COLUMNS = (
    "project_name",
    "experiment_name",
    "name",
    "title",
    "project_id",
    "experiment_id",
)

# Singleton trace service
_trace_service: TraceService = None


def get_trace_service() -> TraceService:
    """Get or create trace service singleton."""
    global _trace_service
    if _trace_service is None:
        _trace_service = TraceService()
    return _trace_service


def _build_failure_reason(judge: dict, flags: dict) -> dict:
    """Build structured failure_reason from judge result and post-tool flags."""
    issues = judge.get("issues") or [] if judge else []
    issues_text = " ".join(issues).lower() if isinstance(issues, (list, tuple)) else str(issues).lower()
    return {
        "judge_verdict": judge.get("verdict") if judge else None,
        "judge_issues": issues,
        "sql_empty": bool(flags.get("sql_empty")),
        "rag_weak": bool(flags.get("rag_weak")),
        "wrong_intent": "wrong intent" in issues_text or "intent" in issues_text and "match" in issues_text,
        "entities_missing": "entit" in issues_text or "missing" in issues_text or "extract" in issues_text,
    }


def _build_rewrite_hint(judge: dict, failure_reason: dict) -> str:
    """Build rewrite_hint for query_rewrite when judge suggests entities or clarity."""
    if not failure_reason.get("entities_missing"):
        return ""
    issues = failure_reason.get("judge_issues") or []
    suggested = judge.get("suggested_revision") if judge else None
    parts = []
    if issues:
        parts.append("Judge issues: " + "; ".join(issues[:3]))
    if suggested:
        parts.append("Suggested: " + (suggested[:200] if isinstance(suggested, str) else ""))
    return " ".join(parts) if parts else "Add explicit project name, time range, or experiment if relevant."


def _extract_entities_from_run(state: AgentState) -> Dict[str, List[str]]:
    """Extract entity values from SQL result and optionally RAG for use in query rewrite.
    Returns e.g. {'project_name': ['Cancer drug'], 'experiment_name': ['Basic experiment on mice']}.
    """
    out: Dict[str, List[str]] = {}
    sql_result = state.get("sql_result")
    if not sql_result or not isinstance(sql_result, dict):
        return out
    data = sql_result.get("data") or []
    if not data:
        return out
    for row in data:
        if not isinstance(row, dict):
            continue
        for col in SQL_ENTITY_COLUMNS:
            if col not in row:
                continue
            val = row[col]
            if val is None or (isinstance(val, str) and not val.strip()):
                continue
            s = str(val).strip()
            if not s or len(s) > 500:
                continue
            if col not in out:
                out[col] = []
            if s not in out[col]:
                out[col].append(s)
    # Cap list sizes to avoid huge prompts
    for col in list(out.keys()):
        out[col] = out[col][:20]
    return out


def _format_extracted_for_prompt(entities: Dict[str, List[str]]) -> str:
    """Format extracted entities as a short bullet list for the rewrite prompt."""
    if not entities:
        return ""
    lines = []
    for col, values in entities.items():
        lines.append(f"  - {col}: {', '.join(values[:10])}")
    return "Extracted from previous attempt (use these to make the query more specific):\n" + "\n".join(lines)


def _do_query_rewrite(state: AgentState) -> None:
    """Rewrite query in-place using retry_context.rewrite_hint and extracted output (SQL/RAG) for a richer query."""
    request = state.get("request")
    normalized = state.get("normalized_query")
    retry_ctx = state.get("retry_context") or {}
    rewrite_hint = (retry_ctx.get("rewrite_hint") or "").strip()
    query_text = request.get("query", "") if isinstance(request, dict) else getattr(request, "query", "")
    # Always consider rewrite when we have extracted output to enrich the query (not only when rewrite_hint is set)
    extracted = _extract_entities_from_run(state)
    extracted_block = _format_extracted_for_prompt(extracted)
    if not rewrite_hint and not extracted_block:
        return
    prompt_parts = [
        "The following user query should be rewritten to be more specific so the system can answer it better.",
        "",
        f"Original query: {query_text}",
    ]
    if rewrite_hint:
        prompt_parts.append("")
        prompt_parts.append(f"Hint from judge: {rewrite_hint}")
    if extracted_block:
        prompt_parts.append("")
        prompt_parts.append(extracted_block)
    prompt_parts.extend([
        "",
        "Rewrite the query to include concrete entities where relevant (e.g. project name, experiment name from the extracted data). Keep it one sentence. If the original is already clear and no enrichment applies, return it unchanged. Return ONLY the rewritten query text, no JSON, no explanation.",
    ])
    prompt = "\n".join(prompt_parts)
    try:
        llm_client = get_llm_client()
        rewritten = llm_client.complete_text(prompt=prompt, temperature=0.2)
        rewritten = (rewritten or query_text).strip() or query_text
    except Exception as e:
        logger.warning("Retry query rewrite failed, keeping original", error=str(e), run_id=state.get("run_id"))
        return
    if isinstance(request, dict):
        state["request"] = {**request, "query": rewritten}
    else:
        state["request"] = request
    if normalized:
        try:
            copy_fn = getattr(normalized, "model_copy", None) or getattr(normalized, "copy", None)
            if copy_fn:
                state["normalized_query"] = copy_fn(update={"normalized_query": rewritten})
        except Exception:
            pass
    state["attempted_tools"] = []


def _current_tool_used(router) -> str:
    """Derive tool_used from router decision."""
    if not router:
        return TOOL_RAG
    tools = router.tools if hasattr(router, "tools") else (router.get("tools", []) if isinstance(router, dict) else [])
    if isinstance(tools, list):
        if TOOL_SQL in tools and TOOL_RAG in tools:
            return TOOL_HYBRID
        if TOOL_SQL in tools:
            return TOOL_SQL
        if TOOL_RAG in tools:
            return TOOL_RAG
    return TOOL_RAG


def _is_better_than(current_summary: dict, current_judge: dict, current_tool: str,
                    best_summary: dict, best_judge: dict, best_tool: str) -> bool:
    """True if current response is better than the stored best (for same-fail or first store)."""
    if not best_summary:
        return True
    cur_pass = current_judge and current_judge.get("verdict") == VERDICT_PASS
    best_pass = best_judge and best_judge.get("verdict") == VERDICT_PASS
    if cur_pass and not best_pass:
        return True
    if best_pass and not cur_pass:
        return False
    if cur_pass and best_pass:
        return len(current_summary.get("answer", "")) >= len(best_summary.get("answer", ""))
    # Both failed: prefer SQL/hybrid over RAG, then longer answer
    cur_prefer = current_tool in (TOOL_SQL, TOOL_HYBRID)
    best_prefer = best_tool in (TOOL_SQL, TOOL_HYBRID)
    if cur_prefer and not best_prefer:
        return True
    if best_prefer and not cur_prefer:
        return False
    return len(current_summary.get("answer", "")) >= len(best_summary.get("answer", ""))


def retry_node(state: AgentState) -> AgentState:
    """Handle retries: build retry_context from judge + flags, reset tool state, keep attempted_tools."""
    # Skipped-judge path (summarizer → retry on last attempt): pass through so should_retry → final
    if state.get("judge_result") is None and state.get("summary"):
        return state

    judge = state.get("judge_result")
    summary = state.get("summary")

    # Use judge's suggested_revision directly when available — skip full pipeline re-run
    if judge and judge.get("verdict") != VERDICT_PASS and summary:
        suggested = judge.get("suggested_revision")
        if suggested and isinstance(suggested, str) and suggested.strip():
            # Use the improved answer; keep existing citations (they remain valid)
            revised_answer = suggested.strip()
            citations = summary.get("citations") or []
            state["summary"] = {"answer": revised_answer, "citations": citations}
            # Mark as pass so should_retry routes to final (no retry loop)
            state["judge_result"] = {
                **judge,
                "verdict": VERDICT_PASS,
                "confidence": min(judge.get("confidence", 0.5) + 0.1, 1.0),
                "issues": [],
                "suggested_revision": None,
            }
            logger.info(
                "retry_node: using judge suggested_revision directly",
                run_id=state.get("run_id"),
                answer_length=len(revised_answer),
            )
            return state

    retry_count = state.get("retry_count", 0)
    request = state["request"]
    run_id = state.get("run_id")
    trace_service = get_trace_service()
    flags = state.get("flags") or {}
    attempted_tools = state.get("attempted_tools") or []
    
    options = request.get("options", {}) if isinstance(request, dict) else getattr(request, "options", {}) if hasattr(request, "options") else {}
    app_cfg = get_app_config()
    default_max = getattr(app_cfg, "agent_max_retries", DEFAULT_MAX_RETRIES)
    max_retries = options.get("max_retries", default_max) if isinstance(options, dict) else getattr(options, "max_retries", default_max)
    
    if judge and judge.get("verdict") == VERDICT_PASS:
        return state
    
    if retry_count >= max_retries:
        if run_id:
            try:
                trace_service.log_event(run_id=run_id, node_name="retry", event_type="output",
                                       payload={"action": "max_retries_reached", "retry_count": retry_count})
            except Exception:
                pass
        return state
    
    new_retry_count = retry_count + 1
    state["retry_count"] = new_retry_count
    
    failure_reason = _build_failure_reason(judge, flags)
    
    # When we've exhausted retries: update best from current (if better), then clear summary/judge
    # so final node uses best_summary instead of the last failed attempt.
    if new_retry_count >= max_retries:
        summary_for_best = state.get("summary")
        if summary_for_best and isinstance(summary_for_best, dict):
            router = state.get("router_decision")
            current_tool = _current_tool_used(router)
            best_summary = state.get("best_summary")
            best_judge = state.get("best_judge_result")
            best_tool = state.get("best_tool_used") or TOOL_RAG
            if _is_better_than(summary_for_best, judge or {}, current_tool, best_summary or {}, best_judge or {}, best_tool):
                state["best_summary"] = summary_for_best
                state["best_judge_result"] = judge
                state["best_tool_used"] = current_tool
        state["summary"] = None
        state["judge_result"] = None
        if run_id:
            try:
                trace_service.log_event(run_id=run_id, node_name="retry", event_type="output",
                                        payload={"action": "max_retries_reached", "retry_count": new_retry_count})
            except Exception:
                pass
        return state
    
    rewrite_hint = _build_rewrite_hint(judge, failure_reason)
    state["retry_context"] = {
        "attempt": new_retry_count,
        "attempted_tools": list(attempted_tools),
        "failure_reason": failure_reason,
        "rewrite_hint": rewrite_hint,
    }
    
    # Inline query rewrite when judge suggested entities/clarity or we have extracted output to enrich the query
    if rewrite_hint or _extract_entities_from_run(state):
        _do_query_rewrite(state)
    
    # Update best answer before clearing (so final can use it after max retries)
    summary = state.get("summary")
    if summary and isinstance(summary, dict):
        router = state.get("router_decision")
        current_tool = _current_tool_used(router)
        best_summary = state.get("best_summary")
        best_judge = state.get("best_judge_result")
        best_tool = state.get("best_tool_used") or TOOL_RAG
        if _is_better_than(summary, judge or {}, current_tool, best_summary or {}, best_judge or {}, best_tool):
            state["best_summary"] = summary
            state["best_judge_result"] = judge
            state["best_tool_used"] = current_tool
    
    thinking_logger = get_thinking_logger()
    if run_id:
        thinking_logger.log_reasoning(
            run_id=run_id, node_name="retry",
            reasoning=f"Retry attempt {new_retry_count}/{max_retries}",
            factors=[
                f"Judge verdict: {judge.get('verdict') if judge else 'unknown'}",
                f"Attempted tools: {attempted_tools}",
                f"sql_empty={failure_reason.get('sql_empty')}, rag_weak={failure_reason.get('rag_weak')}",
            ],
            conclusion="Strategy-aware retry with different tools or query rewrite"
        )
    
    state["router_decision"] = None
    # Keep sql_result and rag_result so the next attempt's summarizer can still use
    # first-attempt tool outputs (e.g. SQL had 1 row, retry runs RAG only → we still have SQL data).
    state["summary"] = None
    state["judge_result"] = None
    state["flags"] = None
    state["sql_anchors"] = None
    state["enriched_context"] = []

    run_log = state.get("run_process_log") or []
    run_log.append({
        "phase": "retry",
        "attempt": new_retry_count,
        "reason": failure_reason,
        "attempted_tools": list(attempted_tools),
        "rewrite_hint": bool(rewrite_hint),
    })
    state["run_process_log"] = run_log

    if run_id:
        try:
            trace_service.log_event(run_id=run_id, node_name="retry", event_type="output",
                                   payload={"action": "retry_initiated", "retry_count": new_retry_count,
                                            "attempted_tools": attempted_tools, "rewrite_hint": bool(rewrite_hint)})
        except Exception:
            pass

    return state
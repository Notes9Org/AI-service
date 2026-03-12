"""Summarizer node for answer synthesis. Keeps prompt bounded so the LLM can handle it."""
import json
import re
import time
import structlog
from typing import Dict, Any
from agents.graph.state import AgentState
from agents.graph.stream_utils import emit_stream_event
from agents.graph.nodes.normalize import get_llm_client
from agents.constants import (
    TOOL_SQL,
    SUMMARIZER_TEMPERATURE,
    SUMMARIZER_ENRICHED_PREVIEW_COUNT,
    SUMMARIZER_ENRICHED_CONTENT_LEN,
    SUMMARIZER_RAG_CONTENT_LEN,
    SUMMARIZER_ANSWER_PREVIEW_LEN,
    SUMMARIZER_SQL_MAX_ROWS,
    SUMMARIZER_SQL_MAX_CELL_LEN,
    SUMMARIZER_RAG_MAX_CHUNKS,
    SUMMARIZER_PROMPT_MAX_CHARS,
)
from services.trace_service import TraceService
from agents.services.thinking_logger import get_thinking_logger

logger = structlog.get_logger()

# Strip control chars and nulls that can break LLM APIs
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# UUID pattern (8-4-4-4-12 hex) and "source_type (uuid):" so we never show raw UUIDs in the answer
_UUID_PATTERN = re.compile(
    r"[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}"
)
_CITATION_LINE_IN_ANSWER = re.compile(
    r"\s*\[\d+\]\s*\w+\s*\([a-fA-F0-9-]{36}\)\s*:\s*",  # [1] lab_note (uuid):
)


def _strip_uuids_from_answer(text: str) -> str:
    """Remove UUIDs and citation lines (source_type (uuid):) from answer so end users never see them."""
    if not text or not isinstance(text, str):
        return text
    # Remove patterns like "[1] lab_note (72a5ffeb-...):" from the answer (LLM sometimes copies this)
    s = _CITATION_LINE_IN_ANSWER.sub(" ", text)
    # Remove any remaining standalone UUIDs (e.g. leaked in prose)
    s = _UUID_PATTERN.sub("", s)
    # Clean " (): " left after UUID removal so we don't show empty parens
    s = re.sub(r"\s*\(\s*\)\s*:\s*", " ", s)
    return " ".join(s.split())


def _safe_str(value: Any, max_len: int = 0) -> str:
    """Coerce to string, strip control characters, truncate. Ensures LLM receives safe text."""
    if value is None:
        return ""
    s = str(value)
    s = _CONTROL_CHAR_RE.sub(" ", s)
    s = " ".join(s.split())
    if max_len and len(s) > max_len:
        s = s[: max_len - 3].rstrip() + "..."
    return s

# User-facing message when synthesis or LLM fails (do not expose RetryError/LLMError)
SUMMARIZER_ERROR_MESSAGE = "We couldn't generate a summary right now. Please try again in a moment."

# Singleton trace service
_trace_service: TraceService = None


def get_trace_service() -> TraceService:
    """Get or create trace service singleton."""
    global _trace_service
    if _trace_service is None:
        _trace_service = TraceService()
    return _trace_service


def _merged_sql_facts_from_runs(sql_runs: list, max_rows: int, max_cell_len: int) -> str:
    """Build one summary-ready facts string from all SQL runs (complete context across retries)."""
    if not sql_runs or not isinstance(sql_runs, list):
        return "No data from database."
    merged_data = []
    last_error = None
    for run in sql_runs:
        if not isinstance(run, dict):
            continue
        if run.get("error"):
            last_error = run.get("error")
            continue
        data = run.get("data") or []
        merged_data.extend(data)
    if not merged_data:
        if last_error:
            return f"Error: {_safe_str(last_error, 500)}"
        return "No data from database."
    combined = {"data": merged_data, "row_count": len(merged_data)}
    return _sql_result_to_summary_facts(combined, max_rows, max_cell_len)


def summarizer_node(state: AgentState) -> AgentState:
    """Synthesize answer from all accumulated SQL and RAG context (complete, relevant only)."""
    emit_stream_event(state, "thinking", {"node": "summarizer", "status": "started", "message": "Synthesizing answer..."})
    start_time = time.time()
    sql_result = state.get("sql_result")
    sql_runs = state.get("sql_runs") or []
    rag_result = state.get("rag_result")
    rag_chunks_all = state.get("rag_chunks_all") or []
    # Use accumulated lists for complete context; fall back to latest run if empty
    rag_for_context = rag_chunks_all if rag_chunks_all else (rag_result or [])
    if rag_result is None and not rag_chunks_all:
        rag_result = []
    sql_anchors = state.get("sql_anchors")
    enriched_context = state.get("enriched_context") or []
    normalized = state.get("normalized_query")
    request = state["request"]
    run_id = state.get("run_id")
    trace_service = get_trace_service()
    total_sql_rows = sum(r.get("row_count", 0) for r in sql_runs if isinstance(r, dict) and not r.get("error"))

    logger.info(
        "summarizer_node started",
        agent_node="summarizer",
        run_id=run_id,
        sql_runs_count=len(sql_runs),
        rag_chunks_total=len(rag_for_context),
        enriched_chunks=len(enriched_context),
        payload={
            "input_sql_runs": len(sql_runs),
            "input_sql_rows_total": total_sql_rows,
            "input_rag_chunks": len(rag_for_context),
            "input_enriched_chunks": len(enriched_context),
        }
    )
    
    if run_id:
        try:
            trace_service.log_event(run_id=run_id, node_name="summarizer", event_type="input",
                                   payload={"sql_runs": len(sql_runs), "sql_rows_total": total_sql_rows,
                                           "rag_chunks": len(rag_for_context)})
        except Exception:
            pass
    
    try:
        llm_client = get_llm_client()
        
        # Data flow: query + all relevant facts (from every SQL run) + all relevant excerpts (from every RAG run)
        query_text = request.get("query", "") if isinstance(request, dict) else getattr(request, "query", "")
        original_query = _safe_str(normalized.normalized_query if normalized else query_text, 2000)
        history_summary = _safe_str(getattr(normalized, "history_summary", None) if normalized else None, 500)

        if sql_runs:
            facts_from_db = _merged_sql_facts_from_runs(
                sql_runs, SUMMARIZER_SQL_MAX_ROWS, SUMMARIZER_SQL_MAX_CELL_LEN
            )
        else:
            facts_from_db = "None available."
            if sql_result and isinstance(sql_result, dict):
                if sql_result.get("data") and not sql_result.get("error"):
                    facts_from_db = _sql_result_to_summary_facts(
                        sql_result, SUMMARIZER_SQL_MAX_ROWS, SUMMARIZER_SQL_MAX_CELL_LEN
                    )
                elif sql_result.get("error"):
                    facts_from_db = f"Error: {_safe_str(sql_result.get('error', 'Unknown error'), 500)}"
                else:
                    facts_from_db = "No data from database."

        relevant_excerpts = _rag_to_relevant_excerpts(
            rag_for_context,
            max_chunks=SUMMARIZER_RAG_MAX_CHUNKS,
            max_content_len=SUMMARIZER_RAG_CONTENT_LEN,
        )
        relevant_followup = _enriched_to_relevant_followup(
            enriched_context,
            max_chunks=SUMMARIZER_ENRICHED_PREVIEW_COUNT,
            max_content_len=SUMMARIZER_ENRICHED_CONTENT_LEN,
        )

        # Keep total prompt within model context to avoid LLMError/RetryError
        prompt_body = f"""You are a knowledgeable lab assistant. Synthesize a comprehensive, insightful answer from the data below.

**Approach (think step by step):**
1. First, analyze the data: What does the database show? What do the documents say?
2. Identify key findings, relationships, and context that matter for the user's question.
3. Then write a clear, well-structured answer that explains the significance of findings—not just raw facts.

**User query:** {original_query}
"""
        if history_summary:
            prompt_body += f"""
**Conversation context:** {history_summary}

"""
        prompt_body += f"""

**Facts (from database — use for counts, names, status, dates, assignments):**
{facts_from_db}

**Relevant excerpts (from documents — cite as [1], [2], etc. for details):**
{relevant_excerpts}"""
        if relevant_followup:
            prompt_body += f"""

**Relevant follow-up (lab notes, protocols — cite by source):**
{relevant_followup}"""
        if len(prompt_body) > SUMMARIZER_PROMPT_MAX_CHARS:
            logger.warning(
                "Summarizer prompt truncated to fit context",
                run_id=run_id,
                original_len=len(prompt_body),
                max_chars=SUMMARIZER_PROMPT_MAX_CHARS,
            )
            prompt_body = prompt_body[: SUMMARIZER_PROMPT_MAX_CHARS - 200].rstrip() + "\n\n[Context truncated for length.]"
        
        thin_note = ""
        if sql_anchors and not relevant_followup:
            thin_note = " If there is no follow-up (lab notes/protocols), say so in one short sentence."
        rules_and_json = f"""
**Rules:**
1. Be specific, informative, and natural. Explain the significance of findings—not just list them.
2. Use bullet points or numbered lists for multiple items. Use headers for complex responses.
3. Use Facts (from database) for counts, names, status, dates; do not invent data.
4. Refer to projects and experiments by name only. Never put UUIDs, source_id, or "source_type (uuid):" in the answer text—readers must not see raw IDs.
5. In the answer text use only [1], [2], etc. as citation markers. Do not copy the excerpt line into the answer. Put the excerpt only in the citations array.
6. When you use content from "Relevant excerpts" or "Relevant follow-up", add a citation with the exact source_type and source_id in the citations array. Include every document you use.
7. No hallucination: only state what is supported by the data above.{thin_note}

Return JSON with:
{{
  "answer": "Comprehensive, well-structured answer with only [1], [2] as citation markers. No UUIDs, no source_type (uuid):, no excerpt text in the answer body.",
  "citations": [
    {{
      "source_type": "lab_note|protocol|report|experiment_summary|sql",
      "source_id": "UUID or identifier",
      "chunk_id": "UUID or null",
      "relevance": 0.0-1.0,
      "excerpt": "Relevant excerpt or null"
    }}
  ]
}}

Citations must reference only sources from the Facts and excerpts above."""
        prompt = prompt_body + rules_and_json

        # Define schema
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_type": {"type": "string"},
                            "source_id": {"type": "string"},
                            "chunk_id": {"type": ["string", "null"]},
                            "relevance": {"type": "number"},
                            "excerpt": {"type": ["string", "null"]}
                        },
                        "required": ["source_type", "source_id", "relevance"]
                    }
                }
            },
            "required": ["answer", "citations"]
        }
        
        try:
            # Use summary-specific model when configured (e.g. Bedrock BEDROCK_CHAT_MODEL_ID_SUMMARY).
            model = getattr(llm_client, "chat_model_id_summary", None) or llm_client.default_deployment
            stream_cb = state.get("stream_callback")
            if stream_cb and callable(stream_cb) and hasattr(llm_client, "complete_text_stream"):
                # Stream tokens when callback is set
                content_parts = []
                for token in llm_client.complete_text_stream(prompt, model=model, temperature=SUMMARIZER_TEMPERATURE):
                    content_parts.append(token)
                    try:
                        stream_cb("token", {"text": token})
                    except Exception:
                        pass
                content = "".join(content_parts).strip()
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
                    content = re.sub(r"\s*```$", "", content)
                content = content.strip()
                result = json.loads(content)
            else:
                result = llm_client.complete_json(prompt, schema, model=model, temperature=SUMMARIZER_TEMPERATURE)
        except Exception as e:
            logger.error("Summarizer LLM call failed", error=str(e), run_id=run_id)
            state["summary"] = {"answer": SUMMARIZER_ERROR_MESSAGE, "citations": []}
            _lat = int((time.time() - start_time) * 1000)
            logger.info("summarizer_node completed", agent_node="summarizer", run_id=run_id,
                       answer_length=0, citations_count=0, latency_ms=round(_lat, 2),
                       payload={"input_sql_rows": total_sql_rows, "input_rag_chunks": len(rag_for_context),
                                "output_answer_length": 0, "output_citations_count": 0, "output_error": str(e)[:100]})
            return state

        if not result or not isinstance(result, dict):
            logger.error("Invalid LLM result in summarizer", run_id=run_id)
            state["summary"] = {"answer": SUMMARIZER_ERROR_MESSAGE, "citations": []}
            _lat = int((time.time() - start_time) * 1000)
            logger.info("summarizer_node completed", agent_node="summarizer", run_id=run_id,
                       answer_length=0, citations_count=0, latency_ms=round(_lat, 2),
                       payload={"input_sql_rows": total_sql_rows, "input_rag_chunks": len(rag_for_context),
                                "output_answer_length": 0, "output_citations_count": 0, "output_error": "Invalid LLM result"})
            return state
        
        validated_citations = []
        rag_source_map = {}  # (source_type_lower, source_id) -> chunk
        rag_by_source_id = {}  # source_id -> chunk (first seen), so we can accept "rag"/"document" citations by id
        chunks_for_validation = rag_for_context if rag_for_context else (rag_result or [])
        if chunks_for_validation and isinstance(chunks_for_validation, list):
            for chunk in chunks_for_validation:
                if not isinstance(chunk, dict):
                    continue
                source_type = chunk.get("source_type")
                source_id = chunk.get("source_id")
                if source_id is None:
                    continue
                sid = str(source_id).strip()
                if source_type is not None:
                    key = (str(source_type).strip().lower(), sid)
                    rag_source_map[key] = chunk
                if sid and sid not in rag_by_source_id:
                    rag_by_source_id[sid] = chunk
        # Enriched context is separate from RAG; allow citations that match enriched chunks
        enriched_source_map = {}
        if enriched_context and isinstance(enriched_context, list):
            for chunk in enriched_context:
                if not isinstance(chunk, dict):
                    continue
                st = chunk.get("source_type")
                sid = chunk.get("source_id")
                if sid is not None and st is not None:
                    key = (str(st).strip().lower(), str(sid).strip())
                    enriched_source_map[key] = chunk
                if sid and str(sid).strip() not in rag_by_source_id:
                    rag_by_source_id[str(sid).strip()] = chunk  # allow resolution by source_id for enriched too

        for citation in result.get("citations", []):
            st = citation.get("source_type")
            sid = str(citation.get("source_id") or "").strip()
            st_normalized = (str(st).strip().lower() if st is not None else "")
            source_key = (st_normalized, sid)
            is_sql = st_normalized == TOOL_SQL
            if source_key in rag_source_map or source_key in enriched_source_map or is_sql:
                validated_citations.append(citation)
                continue
            # Accept RAG/enriched citation when LLM used generic label (e.g. "rag", "document") but source_id matches a chunk
            if sid and st_normalized not in (TOOL_SQL,) and sid in rag_by_source_id:
                chunk = rag_by_source_id[sid]
                resolved = {
                    "source_type": chunk.get("source_type") or st or "rag",
                    "source_id": sid,
                    "chunk_id": citation.get("chunk_id") or chunk.get("chunk_id"),
                    "relevance": citation.get("relevance", chunk.get("similarity", 0.8)),
                    "excerpt": citation.get("excerpt") or (chunk.get("content") or "")[:200],
                }
                validated_citations.append(resolved)
                logger.info("Citation resolved from source_id", run_id=run_id, source_id=sid, resolved_type=resolved.get("source_type"))
            else:
                logger.warning("Invalid citation filtered out", source_type=st, source_id=sid or None)

        has_sql_data = total_sql_rows > 0 or (sql_result and isinstance(sql_result, dict) and (sql_result.get("row_count", 0) or 0) > 0 and not sql_result.get("error"))
        if has_sql_data:
            has_sql_citation = any(
                (str(c.get("source_type") or "").strip().lower() == TOOL_SQL for c in validated_citations)
            )
            if not has_sql_citation:
                validated_citations.insert(0, {"source_type": TOOL_SQL, "source_id": run_id or "query", "relevance": 1.0, "chunk_id": None, "excerpt": None})

        answer_text = result.get("answer", "")
        answer_text = _strip_uuids_from_answer(answer_text)
        summary = {"answer": answer_text, "citations": validated_citations}
        
        thinking_logger = get_thinking_logger()
        if run_id:
            thinking_logger.log_analysis(
                run_id=run_id, node_name="summarizer",
                analysis=f"Synthesized answer from {len(validated_citations)} citations",
                data_summary={"answer_length": len(summary["answer"]), "citations_count": len(validated_citations)},
                insights=[f"Validated {len(validated_citations)} citations"]
            )
        
        latency_ms = int((time.time() - start_time) * 1000)
        logger.info("summarizer_node completed", agent_node="summarizer", run_id=run_id,
                   answer_length=len(summary["answer"]), citations_count=len(validated_citations),
                   latency_ms=round(latency_ms, 2),
                   payload={"input_sql_rows": total_sql_rows, "input_rag_chunks": len(rag_for_context),
                           "output_answer_length": len(summary["answer"]), "output_citations_count": len(validated_citations),
                           "output_answer_preview": summary["answer"][:SUMMARIZER_ANSWER_PREVIEW_LEN]})
        state["summary"] = summary
        # Stream thinking for UI: answer generated
        emit_stream_event(state, "thinking", {
            "node": "summarizer",
            "status": "completed",
            "message": "Answer generated",
        })
        
        run_log = state.get("run_process_log") or []
        run_log.append({
            "phase": "summarizer",
            "attempt": state.get("retry_count", 0),
            "answer_length": len(summary["answer"]),
            "citations_count": len(validated_citations),
            "latency_ms": latency_ms,
        })
        state["run_process_log"] = run_log

        if run_id:
            try:
                trace_service.log_event(run_id=run_id, node_name="summarizer", event_type="output",
                                       payload={"answer_length": len(summary["answer"]),
                                               "citations_count": len(validated_citations)}, latency_ms=latency_ms)
            except Exception:
                pass
        
        options = request.get("options", {}) if isinstance(request, dict) else getattr(request, "options", {}) if hasattr(request, "options") else {}
        if (isinstance(options, dict) and options.get("debug")) or (hasattr(options, "debug") and getattr(options, "debug", False)):
            state["trace"].append({
                "node": "summarizer", "input": {"sql_runs": len(sql_runs), "sql_rows_total": total_sql_rows, "rag_chunks": len(rag_for_context)},
                "output": {"answer_length": len(summary["answer"])}, "latency_ms": round(latency_ms, 2)
            })
        
        return state
        
    except Exception as e:
        logger.error("summarizer_node failed", run_id=run_id, error=str(e))
        _lat = int((time.time() - start_time) * 1000)
        logger.info("summarizer_node completed", agent_node="summarizer", run_id=run_id,
                   answer_length=0, citations_count=0, latency_ms=round(_lat, 2),
                   payload={"input_sql_rows": 0, "input_rag_chunks": 0,
                            "output_answer_length": 0, "output_citations_count": 0, "output_error": str(e)[:100]})
        if run_id:
            try:
                trace_service.log_event(run_id=run_id, node_name="summarizer", event_type="error",
                                       payload={"error": str(e)})
            except Exception:
                pass
        state["summary"] = {"answer": SUMMARIZER_ERROR_MESSAGE, "citations": []}
        return state


# Columns to HIDE from the LLM (binary/embedding data - not useful for narrative)
SQL_HIDDEN_KEYS = frozenset({"embedding", "fts", "chunk_index"})


def _sql_result_to_summary_facts(
    sql_result: Dict,
    max_rows: int = 50,
    max_cell_len: int = 400,
) -> str:
    """Build summary-ready facts from SQL result: pass ALL columns (except binary data) so LLM has complete context."""
    if not sql_result or not sql_result.get("data"):
        return "No data from database."
    data = sql_result["data"][:max_rows]
    lines = []
    for i, row in enumerate(data, 1):
        if not isinstance(row, dict):
            continue
        parts = []
        for k, v in sorted(row.items()):
            if not k or k.lower() in SQL_HIDDEN_KEYS:
                continue
            if v is None or (isinstance(v, str) and not v.strip()):
                continue
            label = k.replace("_", " ").title()
            parts.append(f"{label}: {_safe_str(v, max_cell_len)}")
        if parts:
            lines.append(f"  {i}. " + "; ".join(parts))
    if not lines:
        return "No data from database."
    if len(sql_result["data"]) > max_rows:
        lines.append(f"  ... and {len(sql_result['data']) - max_rows} more (truncated).")
    return "\n".join(lines)


def _rag_to_relevant_excerpts(
    rag_result: list,
    max_chunks: int = 12,
    max_content_len: int = 500,
) -> str:
    """Format RAG chunks as relevance-ordered excerpts for the answer. Only content + source for citation."""
    if not rag_result or not isinstance(rag_result, list):
        return "No relevant documents."
    # Sort by similarity descending so the LLM sees most relevant first
    sorted_chunks = sorted(
        (c for c in rag_result if isinstance(c, dict) and c.get("content")),
        key=lambda c: float(c.get("similarity") or 0),
        reverse=True,
    )[:max_chunks]
    if not sorted_chunks:
        return "No relevant documents."
    lines = []
    for i, chunk in enumerate(sorted_chunks, 1):
        raw_type = chunk.get("source_type")
        source_type = (str(raw_type).strip().lower() if raw_type else "document")
        source_type = _safe_str(source_type, 50)
        source_id = _safe_str(chunk.get("source_id"), 80)
        content = _safe_str(chunk.get("content"), max_content_len)
        lines.append(f"[{i}] {source_type} ({source_id}): {content}")
    return "\n".join(lines)


def _enriched_to_relevant_followup(enriched_context: list, max_chunks: int = 10, max_content_len: int = 400) -> str:
    """Format enriched context as relevant follow-up for lab notes/protocols. Content + source for citation."""
    if not enriched_context or not isinstance(enriched_context, list):
        return ""
    lines = []
    for i, chunk in enumerate(enriched_context[:max_chunks], 1):
        if not isinstance(chunk, dict):
            continue
        source_type = _safe_str(chunk.get("source_type"), 50)
        source_id = _safe_str(chunk.get("source_id"), 80)
        content = _safe_str(chunk.get("content"), max_content_len)
        if content:
            lines.append(f"[{i}] {source_type} ({source_id}): {content}")
    return "\n".join(lines) if lines else ""
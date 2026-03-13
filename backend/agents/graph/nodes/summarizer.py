"""Summarizer node for answer synthesis."""
import time
import structlog
from typing import Dict
from agents.graph.state import AgentState
from agents.services.llm_client import LLMClient, LLMError
from agents.graph.nodes.normalize import get_llm_client
from services.trace_service import TraceService
from services.db import SupabaseService
from agents.services.thinking_logger import get_thinking_logger

logger = structlog.get_logger()

# Singleton trace service
_trace_service: TraceService = None


def get_trace_service() -> TraceService:
    """Get or create trace service singleton."""
    global _trace_service
    if _trace_service is None:
        _trace_service = TraceService()
    return _trace_service


def summarizer_node(state: AgentState) -> AgentState:
    """Synthesize answer from SQL facts and RAG evidence with citations."""
    start_time = time.time()
    sql_result = state.get("sql_result")
    rag_result = state.get("rag_result")  # Can be None if RAG was skipped
    if rag_result is None:
        rag_result = []  # Default to empty list if None
    normalized = state.get("normalized_query")
    request = state["request"]
    run_id = state.get("run_id")
    trace_service = get_trace_service()
    
    logger.info(
        "summarizer_node started",
        agent_node="summarizer",
        run_id=run_id,
        has_sql=sql_result is not None,
        rag_chunks=len(rag_result),
        payload={
            "input_sql_row_count": sql_result.get("row_count", 0) if sql_result else 0,
            "input_rag_chunks": len(rag_result),
            "input_has_sql": sql_result is not None,
            "input_has_rag": len(rag_result) > 0
        }
    )
    
    if run_id:
        try:
            trace_service.log_event(run_id=run_id, node_name="summarizer", event_type="input",
                                   payload={"sql_rows": sql_result.get("row_count", 0) if sql_result else 0,
                                           "rag_chunks": len(rag_result)})
        except Exception:
            pass
    
    try:
        llm_client = get_llm_client()
        
        sql_context = ""
        if sql_result and isinstance(sql_result, dict):
            if sql_result.get("data") and not sql_result.get("error"):
                sql_context = f"Data from records:\n{_format_sql_result(sql_result)}"
            elif sql_result.get("error"):
                sql_context = f"Data: Error occurred - {sql_result.get('error', 'Unknown error')}"
            else:
                sql_context = "Data: No records returned"
        else:
            sql_context = "Data: None available"
        
        source_name_map = {}
        if rag_result and isinstance(rag_result, list) and len(rag_result) > 0:
            try:
                db = SupabaseService()
                source_name_map = db.get_source_display_names(rag_result)
            except Exception as e:
                logger.warning("Could not resolve source display names", error=str(e))
        rag_context = ""
        if rag_result and isinstance(rag_result, list) and len(rag_result) > 0:
            rag_context = "Retrieved documents (in your answer refer to these by the names below, e.g. \"according to [1] Lab note: PCR Protocol\" or citation [1], [2]; never mention IDs):\n"
            for i, chunk in enumerate(rag_result, 1):
                if isinstance(chunk, dict):
                    st = chunk.get("source_type", "unknown")
                    sid = chunk.get("source_id")
                    display_name = source_name_map.get((st, str(sid))) if sid else None
                    if display_name:
                        label = f"{st.replace('_', ' ').title()}: {display_name}"
                    else:
                        label = st.replace("_", " ").title()
                    rag_context += f"\n[{i}] {label}\n"
                    rag_context += f"Relevance: {chunk.get('similarity', 0.0):.3f}\n"
                    rag_context += f"Content: {chunk.get('content', '')[:500]}\n"
        else:
            rag_context = "Retrieved documents: None available"
        
        query_text = request.get("query", "") if isinstance(request, dict) else getattr(request, "query", "")
        original_query = normalized.normalized_query if normalized else query_text
        
        prompt = f"""Synthesize a scientific answer from the following data for a lab management system.

User Query: {original_query}

{sql_context}

{rag_context}

Requirements:
1. Answer must be factual and cite sources by index [1], [2], etc. Never mention source_id, chunk_id, or any UUID in the answer text. When describing a source, use a short label (e.g. "the lab note", "the protocol") or the citation number only.
2. The data from records and retrieved documents are authoritative - use them directly. Never say "based on SQL facts", "according to RAG evidence", "the database shows", or "from the RAG results". Refer directly to the information: e.g. "The records show...", "According to the lab note...", "The data indicates...".
3. All claims must have citations [1], [2] referencing the numbered documents above.
4. Use scientific terminology appropriate for lab management.
5. If data from records and retrieved documents conflict, prefer the former for numbers and the latter for context.
6. If no relevant data, say so clearly.

Answer formatting (critical for UI): (1) Start directly with the answer; no filler like "Certainly!" or "Great question!". (2) Numbered items on one line only: "Number. Topic — explanation" e.g. "1. Lung Cancer — Smoking is the leading cause." Use em dash (—) after topic; never break topic and explanation onto separate lines. (3) Bullets allowed but flat only; no nested sub-bullets. (4) Headings OK for long answers; keep them short. (5) Bold/italic sparingly; do not bold whole sentences. (6) One line break between list items, no excessive blank lines. (7) Conversational and direct; not formal report style unless asked. (8) Simple answers in plain prose; use lists only when structure is needed. (9) Do not repeat the user's question before answering. (10) End naturally; no "I hope this helps!" or similar.

Return JSON with:
{{
  "answer": "Complete answer text with citations in [1], [2] format",
  "citations": [
    {{ "index": 1 }},
    {{ "index": 2 }}
  ]
}}

Citations: For each document you used from the list above, add one object with "index": N (N = 1 for first, 2 for second, etc.). You must include at least one citation. If the answer uses only the "Data from records" section and no retrieved documents, use exactly one citation: {{ "index": 0 }}. Use the same numbers [1], [2] in your answer text."""

        # Schema: citations by index (1-based = document [1],[2]...; 0 = data from records only)
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer", "description": "Document number 1, 2, ... or 0 for data from records"},
                            "source_type": {"type": "string"},
                            "source_id": {"type": "string"},
                            "relevance": {"type": "number"},
                            "excerpt": {"type": ["string", "null"]}
                        }
                    }
                }
            },
            "required": ["answer", "citations"]
        }
        
        try:
            # Use summary-specific model when configured (e.g. Bedrock BEDROCK_CHAT_MODEL_ID_SUMMARY).
            model = getattr(llm_client, "chat_model_id_summary", None) or llm_client.default_deployment
            result = llm_client.complete_json(prompt, schema, model=model, temperature=0.3)
        except Exception as e:
            logger.error("Summarizer LLM call failed", error=str(e), run_id=run_id)
            state["summary"] = {"answer": f"Error synthesizing answer: {str(e)}", "citations": []}
            return state
        
        if not result or not isinstance(result, dict):
            logger.error("Invalid LLM result in summarizer", run_id=run_id)
            state["summary"] = {"answer": "Error: Invalid response from synthesis", "citations": []}
            return state
        
        validated_citations = []
        rag_source_map = {}
        if rag_result and isinstance(rag_result, list) and len(rag_result) > 0:
            for chunk in rag_result:
                if isinstance(chunk, dict):
                    source_type = chunk.get("source_type")
                    source_id = chunk.get("source_id")
                    if source_type and source_id:
                        rag_source_map[(source_type, str(source_id))] = chunk

        for citation in result.get("citations", []):
            if not isinstance(citation, dict):
                continue
            raw_idx = citation.get("index")
            idx = None
            if raw_idx is not None:
                try:
                    idx = int(raw_idx) if not isinstance(raw_idx, int) else raw_idx
                except (TypeError, ValueError):
                    pass
            if idx is not None:
                if idx == 0:
                    validated_citations.append({
                        "display_label": "Data from records",
                        "source_type": "sql",
                        "source_id": "",
                        "source_name": "Data from records",
                        "chunk_id": None,
                        "relevance": 1.0,
                        "excerpt": citation.get("excerpt"),
                    })
                elif 1 <= idx <= len(rag_result):
                    chunk = rag_result[idx - 1]
                    if isinstance(chunk, dict):
                        st = chunk.get("source_type", "unknown")
                        sid = chunk.get("source_id", "")
                        name = source_name_map.get((st, str(sid)))
                        label = f"{st.replace('_', ' ').title()}: {name}" if name else st.replace("_", " ").title()
                        validated_citations.append({
                            "display_label": label,
                            "source_type": st,
                            "source_id": str(sid) if sid else "",
                            "source_name": name,
                            "chunk_id": chunk.get("chunk_id"),
                            "relevance": float(chunk.get("similarity", 0.0)),
                            "excerpt": citation.get("excerpt") or (chunk.get("content", "")[:300] if chunk.get("content") else None),
                        })
                else:
                    logger.warning("Citation index out of range", index=idx, rag_len=len(rag_result))
            else:
                source_key = (citation.get("source_type"), str(citation.get("source_id") or ""))
                if source_key in rag_source_map:
                    chunk = rag_source_map[source_key]
                    st, sid = citation.get("source_type"), citation.get("source_id")
                    name = source_name_map.get((st, str(sid)))
                    label = f"{st.replace('_', ' ').title()}: {name}" if name else st.replace("_", " ").title()
                    validated_citations.append({
                        "display_label": label,
                        "source_type": st,
                        "source_id": str(sid),
                        "source_name": name,
                        "chunk_id": chunk.get("chunk_id"),
                        "relevance": float(citation.get("relevance", chunk.get("similarity", 0.0))),
                        "excerpt": citation.get("excerpt"),
                    })
                elif citation.get("source_type") == "sql":
                    validated_citations.append({
                        "display_label": "Data from records",
                        "source_type": "sql",
                        "source_id": "",
                        "source_name": "Data from records",
                        "chunk_id": None,
                        "relevance": float(citation.get("relevance", 1.0)),
                        "excerpt": citation.get("excerpt"),
                    })
                else:
                    logger.warning("Citation skipped (no index and not in rag_source_map)", citation=citation)
        
        summary = {"answer": result.get("answer", ""), "citations": validated_citations}
        
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
                   payload={"input_sql_rows": sql_result.get("row_count", 0) if sql_result else 0,
                           "input_rag_chunks": len(rag_result),
                           "output_answer_length": len(summary["answer"]), "output_citations_count": len(validated_citations),
                           "output_answer_preview": summary["answer"][:200]})
        state["summary"] = summary
        
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
                "node": "summarizer", "input": {"sql_rows": sql_result.get("row_count", 0) if sql_result else 0},
                "output": {"answer_length": len(summary["answer"])}, "latency_ms": round(latency_ms, 2)
            })
        
        return state
        
    except Exception as e:
        logger.error("summarizer_node failed", run_id=run_id, error=str(e))
        
        if run_id:
            try:
                trace_service.log_event(run_id=run_id, node_name="summarizer", event_type="error",
                                       payload={"error": str(e)})
            except Exception:
                pass
        
        state["summary"] = {"answer": f"Error synthesizing answer: {str(e)}", "citations": []}
        return state


def _format_sql_result(sql_result: Dict) -> str:
    """Format SQL result for prompt."""
    if not sql_result or not sql_result.get("data"):
        return "No data"
    
    lines = []
    for row in sql_result["data"]:
        line = ", ".join([f"{k}={v}" for k, v in row.items()])
        lines.append(line)
    
    return "\n".join(lines)
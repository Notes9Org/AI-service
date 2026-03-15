"""ClarifyEngine: LLM-based query analysis for intelligent clarifying questions."""
import asyncio
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger()

CLARIFY_SYSTEM = """You are a biomedical query analyst. Your job is to determine if a user's biomedical research query is clear enough to execute, or if it needs clarification.

Consider these cases that typically need clarification:
- Compound/chemical queries: missing compound name, SMILES, or structure
- Gene/protein queries: missing organism, gene symbol, or identifier
- CRISPR/screen queries: missing target, cell type, or delivery method
- ADMET/drug queries: missing compound or dosage context
- Analysis queries: missing file path, data format, or sample info
- Vague terms: "this compound", "the gene", "that experiment" without context
- Ambiguous task type: could mean multiple different analyses

If the query is clear and has enough context to execute, return needs_clarification: false.
If clarification would help, return needs_clarification: true with a concise, helpful question.
Optionally provide 2-4 suggested options (e.g. common choices) in the "options" array.

Respond with JSON only: {"needs_clarification": bool, "question": str|null, "options": [str]|null}"""


@dataclass
class ClarifyResult:
    """Result of query clarification analysis."""

    needs_clarification: bool
    question: Optional[str] = None
    options: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "needs_clarification": self.needs_clarification,
            "question": self.question,
            "options": self.options,
        }


def _parse_clarify_response(raw: Dict[str, Any]) -> ClarifyResult:
    """Parse LLM response into ClarifyResult."""
    needs = raw.get("needs_clarification", False)
    question = raw.get("question")
    options = raw.get("options")
    if isinstance(options, list):
        options = [str(o) for o in options if o]
    else:
        options = None
    if needs and not question:
        question = "Could you provide more details about your query?"
    return ClarifyResult(
        needs_clarification=bool(needs),
        question=str(question) if question else None,
        options=options,
    )


async def evaluate_clarification(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> ClarifyResult:
    """
    Evaluate if a biomedical query needs clarification before execution.

    Args:
        query: The user's natural language query.
        history: Previous clarification Q&A: [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        ClarifyResult with needs_clarification, question, and optional options.
    """
    history = history or []
    history_text = ""
    if history:
        history_text = "\n\nPrevious clarification exchange:\n"
        for h in history[-4:]:  # last 4 turns
            role = h.get("role", "user")
            content = h.get("content", "")
            history_text += f"{role}: {content}\n"

    user_prompt = f"""Analyze this biomedical query:

"{query}"
{history_text}

Does this query need clarification before a biomedical AI agent can execute it? Return JSON only."""

    def _run_llm() -> ClarifyResult:
        from agents.services.llm_client import LLMClient, parse_llm_json

        llm = LLMClient()
        content = llm._converse_bedrock(
            model=llm.default_deployment,
            system=CLARIFY_SYSTEM,
            user=user_prompt + "\n\nIMPORTANT: Return ONLY valid JSON.",
            temperature=0.3,
        )
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content).strip()
        raw = parse_llm_json(content)
        return _parse_clarify_response(raw)

    try:
        return await asyncio.to_thread(_run_llm)
    except Exception as e:
        logger.warning("clarify_evaluation_failed", error=str(e))
        return ClarifyResult(needs_clarification=False)

"""LLM client abstraction with retry and JSON validation (AWS Bedrock only)."""

import json
import time
import re
from typing import Dict, Any, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

from services.config import get_bedrock_config

logger = structlog.get_logger()


def _extract_text_from_content(content: List[Dict[str, Any]]) -> str:
    """Extract and join all text from Bedrock Converse content blocks. Preserves newlines."""
    if not content:
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and "text" in block:
            parts.append(block["text"] or "")
    return "".join(parts)


class LLMError(Exception):
    """Custom exception for LLM errors."""

    pass


class LLMClient:
    """Abstracted LLM client with structured output support (AWS Bedrock)."""

    def __init__(self):
        self.config = get_bedrock_config()
        self.client = self.config.create_bedrock_runtime_client()
        self.default_deployment = self.config.get_chat_model_id()
        self.default_model = self.default_deployment
        self.chat_model_id_sql = self.config.get_chat_model_id_sql()
        self.chat_model_id_summary = self.config.get_chat_model_id_summary()
        self.max_completion_tokens = self.config.max_completion_tokens
        self.default_temperature = self.config.default_temperature
        logger.info(
            "LLM client initialized",
            model=self.default_model,
            provider="AWS Bedrock",
            max_completion_tokens=self.max_completion_tokens,
            default_temperature=self.default_temperature,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def complete_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output from prompt using Bedrock Converse API.
        """
        model = model or self.default_deployment
        use_temperature = (
            temperature if temperature is not None else self.default_temperature
        )
        start_time = time.time()

        try:
            content = self._converse_bedrock(
                model=model,
                system=(
                    "You are a helpful assistant that returns valid JSON. "
                    "Always respond with JSON only, no markdown, no explanations."
                ),
                user=(
                    f"{prompt}\n\nIMPORTANT: Return ONLY valid JSON, "
                    "no markdown code blocks, no explanations."
                ),
                temperature=use_temperature,
            )

            content = content.strip()
            if content.startswith("```"):
                content = re.sub(
                    r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE
                )
            content = re.sub(r"\s*```$", "", content).strip()

            result = json.loads(content)
            latency_ms = (time.time() - start_time) * 1000
            logger.info(
                "LLM JSON completion",
                model=model,
                latency_ms=round(latency_ms, 2),
                response_length=len(content),
            )
            return result
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON from LLM", error=str(e), content=content[:200])
            raise LLMError(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_str = str(e)
            logger.error(
                "LLM JSON completion failed",
                error=error_str,
                model=model,
                latency_ms=round(latency_ms, 2),
            )
            raise LLMError(f"Bedrock LLM failed (model={model}): {error_str}") from e

    def _converse_bedrock(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.0,
    ) -> str:
        """Call Bedrock Converse API and return assistant text."""

        messages = [{"role": "user", "content": [{"text": user}]}]
        system_content = [{"text": system}]
        inference_config: Dict[str, Any] = {
            "maxTokens": self.max_completion_tokens,
        }
        if temperature != 1.0:
            inference_config["temperature"] = temperature

        response = self.client.converse(
            modelId=model,
            messages=messages,
            system=system_content,
            inferenceConfig=inference_config,
        )
        out = response.get("output", {}).get("message", {}).get("content", [])
        text = _extract_text_from_content(out)
        if not text:
            raise LLMError("Empty or invalid response from Bedrock")
        return text

    def chat(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Multi-turn chat: pass list of {"role": "user"|"assistant", "content": "..."}.
        Returns the assistant reply text.
        """
        model = model or self.default_deployment

        converse_messages = []
        for m in messages:
            role = (m.get("role") or "user").lower()
            if role not in ("user", "assistant"):
                role = "user"
            content = m.get("content") or ""
            converse_messages.append({"role": role, "content": [{"text": content}]})

        system_content = [{"text": system}] if system else []
        inference_config: Dict[str, Any] = {
            "maxTokens": self.max_completion_tokens,
        }
        if temperature != 1.0:
            inference_config["temperature"] = temperature

        response = self.client.converse(
            modelId=model,
            messages=converse_messages,
            system=system_content if system_content else None,
            inferenceConfig=inference_config,
        )
        out = response.get("output", {}).get("message", {}).get("content", [])
        text = _extract_text_from_content(out)
        if not text:
            raise LLMError("Empty or invalid response from Bedrock")
        return text.strip()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def complete_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate text output from prompt using Bedrock Converse API.
        """
        model = model or self.default_deployment
        start_time = time.time()

        try:
            content = self._converse_bedrock(
                model=model,
                system="You are a helpful assistant.",
                user=prompt,
                temperature=temperature,
            )
            latency_ms = (time.time() - start_time) * 1000
            logger.info(
                "LLM text completion",
                model=model,
                latency_ms=round(latency_ms, 2),
                response_length=len(content),
            )
            return content.strip()
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(
                "LLM text completion failed",
                error=str(e),
                model=model,
                latency_ms=round(latency_ms, 2),
            )
            raise LLMError(f"LLM completion failed: {str(e)}")


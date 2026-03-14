"""LLM client abstraction with retry and JSON validation (AWS Bedrock only)."""

import json
import time
import re
from typing import Dict, Any, Optional, Iterator, List
import structlog

from services.config import get_bedrock_config
from tenacity import retry, stop_after_attempt, wait_exponential

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


def _sanitize_json_control_chars(raw: str) -> str:
    """
    Escape unescaped control characters (\\n, \\r, \\t) inside JSON double-quoted strings.
    LLMs sometimes return literal newlines inside string values, which is invalid JSON.
    """
    result = []
    i = 0
    in_string = False
    escape_next = False
    while i < len(raw):
        c = raw[i]
        if escape_next:
            result.append(c)
            escape_next = False
            i += 1
            continue
        if in_string:
            if c == "\\":
                result.append(c)
                escape_next = True
                i += 1
                continue
            if c == '"':
                in_string = False
                result.append(c)
                i += 1
                continue
            if c == "\n":
                result.append("\\n")
            elif c == "\r":
                result.append("\\r")
            elif c == "\t":
                result.append("\\t")
            elif ord(c) < 32:
                result.append(" ")
            else:
                result.append(c)
            i += 1
            continue
        if c == '"':
            in_string = True
        result.append(c)
        i += 1
    return "".join(result)


def parse_llm_json(content: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM response. If parsing fails due to invalid control characters
    (e.g. literal newlines inside strings), sanitize and retry once.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        if "control character" in str(e).lower() or "invalid" in str(e).lower():
            sanitized = _sanitize_json_control_chars(content)
            return json.loads(sanitized)
        raise


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

            result = parse_llm_json(content)
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

    def complete_text_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """
        Generate text output from prompt, yielding tokens as they arrive.
        Supports Bedrock converse_stream and Azure streaming.
        """
        model = model or self.default_deployment
        if self._provider == "bedrock":
            yield from self._converse_stream_bedrock(
                model=model,
                system="You are a helpful assistant.",
                user=prompt,
                temperature=temperature,
            )
        else:
            yield from self._complete_text_stream_azure(model, prompt, temperature)

    def _converse_stream_bedrock(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Stream text from Bedrock ConverseStream API."""
        messages = [{"role": "user", "content": [{"text": user}]}]
        system_content = [{"text": system}]
        inference_config = {"maxTokens": self.max_completion_tokens}
        if temperature != 1.0:
            inference_config["temperature"] = temperature
        try:
            response = self.client.converse_stream(
                modelId=model,
                messages=messages,
                system=system_content,
                inferenceConfig=inference_config,
            )
            stream = response.get("stream")
            if stream:
                for event in stream:
                    if "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"].get("delta", {})
                        text = delta.get("text", "")
                        if text:
                            yield text
        except Exception as e:
            logger.error("Bedrock stream failed", error=str(e), model=model)
            raise LLMError(f"Stream failed: {str(e)}") from e

    def _complete_text_stream_azure(
        self,
        model: str,
        prompt: str,
        temperature: float,
    ) -> Iterator[str]:
        """Stream text from Azure OpenAI chat completions."""
        request_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "timeout": 60.0,
        }
        if temperature != 1.0:
            request_params["temperature"] = temperature
        try:
            request_params["max_completion_tokens"] = self.max_completion_tokens
        except Exception:
            pass
        try:
            stream = self.client.chat.completions.create(**request_params)
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        yield delta.content
        except Exception as e:
            logger.error("Azure stream failed", error=str(e), model=model)
            raise LLMError(f"Stream failed: {str(e)}") from e

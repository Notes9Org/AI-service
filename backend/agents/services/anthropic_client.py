import os
from typing import List, Dict, Any, Iterator

import anthropic
import httpx
import structlog

from agents.services.llm_client import LLMError


logger = structlog.get_logger()

# Single shared Anthropic client to avoid repeated construction/GC issues
_anthropic_client: anthropic.Anthropic | None = None


def get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set in environment")
        _anthropic_client = anthropic.Anthropic(
            api_key=api_key,
            http_client=httpx.Client(),
        )
    return _anthropic_client


class AnthropicChatClient:
    """
    Thin wrapper around Anthropic's Messages API for chat and streaming.

    This is used for general chat (internet-enabled Claude) while the rest
    of the system can continue to use Bedrock via LLMClient.
    """

    def __init__(self):
        self.client = get_anthropic_client()
        self.model = os.getenv("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-6")

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert our simple {'role','content'} list to Anthropic message format."""
        converted: List[Dict[str, Any]] = []
        for m in messages:
            role = (m.get("role") or "user").lower()
            if role not in ("user", "assistant"):
                role = "user"
            content = m.get("content") or ""
            converted.append({"role": role, "content": content})
        return converted

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        system: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        use_web: bool = True,
    ) -> Iterator[Dict[str, Any]]:
        """
        Streaming chat. Yields dicts: {"type": "token", "text": "..."} or {"type": "source", "url": "...", "title": "..."}.
        When use_web=True, enables web search; sources are extracted from stream events when available.
        """
        anthropic_messages = self._convert_messages(messages)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": anthropic_messages,
        }
        if use_web:
            kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]
            kwargs["tool_choice"] = {"type": "auto"}

        try:
            with self.client.messages.stream(**kwargs) as stream:
                sources_seen: set = set()

                for event in stream:
                    ev_type = getattr(event, "type", None) or (event.get("type") if isinstance(event, dict) else None)
                    if ev_type == "content_block_delta":
                        delta = getattr(event, "delta", None) or (event.get("delta") if isinstance(event, dict) else None)
                        if delta:
                            d_type = getattr(delta, "type", None) or (delta.get("type") if isinstance(delta, dict) else None)
                            if d_type == "text_delta":
                                text = getattr(delta, "text", None) or (delta.get("text") if isinstance(delta, dict) else "") or ""
                                if text:
                                    yield {"type": "token", "text": text}
                    elif ev_type == "content_block_start":
                        cb = getattr(event, "content_block", None) or (event.get("content_block") if isinstance(event, dict) else None)
                        if cb:
                            cb_type = getattr(cb, "type", None) or (cb.get("type") if isinstance(cb, dict) else None)
                            if cb_type == "web_search_tool_result":
                                content = getattr(cb, "content", None) or (cb.get("content") if isinstance(cb, dict) else []) or []
                                for item in content:
                                    url = None
                                    title = None
                                    if isinstance(item, dict):
                                        url = item.get("url") or (item.get("source") or {}).get("url")
                                        title = item.get("title") or (item.get("source") or {}).get("title")
                                    elif hasattr(item, "url"):
                                        url = getattr(item, "url", None)
                                        title = getattr(item, "title", None)
                                    if url and url not in sources_seen:
                                        sources_seen.add(url)
                                        yield {"type": "source", "url": url, "title": title or url}
        except anthropic.APIError as e:
            logger.error("Anthropic stream API error", status_code=e.status_code, error=str(e))
            raise LLMError(f"Anthropic stream error ({e.status_code}): {str(e)}") from e
        except Exception as e:
            logger.error("Anthropic chat stream failed", error=str(e), exc_info=True)
            raise LLMError(f"Anthropic chat stream failed: {str(e)}") from e

    def chat(
        self,
        messages: List[Dict[str, Any]],
        system: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        use_web: bool = True,
    ) -> Dict[str, Any]:
        """
        Non-streaming chat. Returns dict with content, role, and optionally sources.

        When use_web=True, enables Anthropic's server-side web_search tool
        (type: web_search_20250305) with automatic tool choice.
        Extracts sources from text block citations and web_search_tool_result blocks.
        """
        anthropic_messages = self._convert_messages(messages)
        try:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": anthropic_messages,
            }
            if use_web:
                kwargs["tools"] = [
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                    }
                ]
                kwargs["tool_choice"] = {"type": "auto"}

            logger.info(
                "anthropic_chat_request",
                model=self.model,
                use_web=use_web,
                message_count=len(anthropic_messages),
            )
            resp = self.client.messages.create(**kwargs)
            logger.info(
                "anthropic_chat_response",
                model=resp.model,
                stop_reason=resp.stop_reason,
                usage_input=resp.usage.input_tokens,
                usage_output=resp.usage.output_tokens,
            )

            parts: List[str] = []
            sources_seen: set = set()
            sources_order: List[Dict[str, str]] = []

            for block in getattr(resp, "content", []) or []:
                b_type = getattr(block, "type", None) or (
                    block.get("type") if isinstance(block, dict) else None
                )
                if b_type == "text":
                    text = getattr(block, "text", None) or (
                        block.get("text") if isinstance(block, dict) else ""
                    ) or ""
                    parts.append(text)
                    # Extract citations (web_search_result_location) for sources
                    citations = getattr(block, "citations", None) or (
                        block.get("citations") if isinstance(block, dict) else []
                    ) or []
                    for c in citations:
                        c_type = getattr(c, "type", None) or (
                            c.get("type") if isinstance(c, dict) else None
                        )
                        if c_type == "web_search_result_location":
                            url = getattr(c, "url", None) or (
                                c.get("url") if isinstance(c, dict) else None
                            )
                            title = getattr(c, "title", None) or (
                                c.get("title") if isinstance(c, dict) else None
                            ) or url or ""
                            if url and url not in sources_seen:
                                sources_seen.add(url)
                                sources_order.append({"url": url, "title": title})
                elif b_type == "web_search_tool_result":
                    content = getattr(block, "content", None) or (
                        block.get("content") if isinstance(block, dict) else []
                    ) or []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "web_search_result":
                            url = item.get("url") or (item.get("source") or {}).get("url")
                            title = item.get("title") or (item.get("source") or {}).get("title") or url or ""
                            if url and url not in sources_seen:
                                sources_seen.add(url)
                                sources_order.append({"url": url, "title": title})
                        elif hasattr(item, "url"):
                            url = getattr(item, "url", None)
                            title = getattr(item, "title", None) or url or ""
                            if url and url not in sources_seen:
                                sources_seen.add(url)
                                sources_order.append({"url": url, "title": title})

            result: Dict[str, Any] = {
                "content": "".join(parts).strip(),
                "role": "assistant",
            }
            if sources_order:
                result["sources"] = sources_order
                result["searched_web"] = True
            return result
        except anthropic.APIError as e:
            logger.error("Anthropic API error", status_code=e.status_code, error=str(e))
            raise LLMError(f"Anthropic API error ({e.status_code}): {str(e)}") from e
        except Exception as e:
            logger.error("Anthropic chat failed", error=str(e), exc_info=True)
            raise LLMError(f"Anthropic chat failed: {str(e)}") from e


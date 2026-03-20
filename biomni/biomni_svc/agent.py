"""Biomni agent factory, task runner, and streaming. Imports from pip package 'biomni'."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, Iterator, List, Optional

import structlog

from biomni_svc.config import ensure_bedrock_env, get_biomni_config

logger = structlog.get_logger()

_agent_instance: Optional[Any] = None
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="biomni")
_concurrency_limit = threading.Semaphore(2)


def _is_rate_limit_error(e: Exception) -> bool:
    """Check if an exception is a rate limit / throttling error (Bedrock or Anthropic)."""
    msg = str(e).lower()
    return "throttlingexception" in msg or "rate_limit" in msg or "too many requests" in msg


def get_biomni_agent():
    """Lazily create and cache Biomni A1 agent instance."""
    global _agent_instance
    if _agent_instance is not None:
        return _agent_instance

    ensure_bedrock_env()
    cfg = get_biomni_config()

    try:
        from biomni.agent import A1
        from biomni.config import default_config
    except ImportError as e:
        raise RuntimeError(
            "Biomni package not installed. Run: pip install biomni langchain-aws"
        ) from e

    # Set default_config BEFORE creating A1 so internal tools use correct settings
    default_config.path = cfg.path
    default_config.source = cfg.source
    default_config.llm = cfg.llm
    default_config.timeout_seconds = cfg.timeout_seconds
    default_config.temperature = cfg.temperature
    if hasattr(default_config, "commercial_mode"):
        default_config.commercial_mode = cfg.commercial_mode

    init_kwargs = {
        "path": cfg.path,
        "llm": cfg.llm,
        "source": cfg.source,
        "commercial_mode": cfg.commercial_mode,
    }
    if cfg.skip_datalake:
        init_kwargs["expected_data_lake_files"] = []

    agent = A1(**init_kwargs)

    # Attach MCP servers if configured
    try:
        from biomni_svc.mcp import attach_mcp_to_agent
        attach_mcp_to_agent(agent)
    except Exception:
        pass

    _agent_instance = agent
    logger.info(
        "biomni_agent_initialized",
        path=cfg.path,
        llm=cfg.llm,
        source=cfg.source,
        commercial_mode=cfg.commercial_mode,
        timeout_seconds=cfg.timeout_seconds,
    )
    return agent


def _clear_plots() -> None:
    """Clear any previously captured plots from the biomni execution environment."""
    try:
        from biomni.tool.support_tools import clear_captured_plots
        clear_captured_plots()
    except Exception:
        pass


def _get_captured_images() -> List[str]:
    """Retrieve captured plot images as base64 data URIs."""
    try:
        from biomni.tool.support_tools import get_captured_plots
        plots = get_captured_plots()
        return list(plots) if plots else []
    except Exception:
        return []


def _parse_go_result(raw: Any) -> tuple[List[str], str]:
    """Parse agent.go() return value. Biomni returns (log_steps, answer) or just answer."""
    if raw is None:
        return [], ""
    if isinstance(raw, tuple) and len(raw) >= 2:
        steps = list(raw[0]) if raw[0] else []
        answer = str(raw[1]) if raw[1] else ""
        return steps, answer
    return [], str(raw)


def run_biomni_task(
    query: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    max_retries: int = 3,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run a Biomni task and return result with metadata.

    Returns:
        Dict with keys: result (str), success (bool), error (optional str),
        steps (list), images (list of base64 data URIs).
    """
    start = time.time()
    cfg = get_biomni_config()
    logger.info(
        "biomni_task_started",
        user_id=user_id,
        session_id=session_id,
        query_len=len(query),
        timeout_seconds=cfg.timeout_seconds,
    )

    def _run() -> tuple[List[str], str, List[str]]:
        with _concurrency_limit:
            agent = get_biomni_agent()
            _clear_plots()
            for attempt in range(max_retries + 1):
                try:
                    raw = agent.go(query, **kwargs)
                    steps, answer = _parse_go_result(raw)
                    images = _get_captured_images()
                    return steps, answer, images
                except Exception as e:
                    if _is_rate_limit_error(e) and attempt < max_retries:
                        wait = min(120, 15 * 2 ** attempt)
                        logger.warning(
                            "biomni_throttled",
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            wait_seconds=wait,
                        )
                        time.sleep(wait)
                        continue
                    raise

    try:
        future = _executor.submit(_run)
        steps, result, images = future.result(timeout=cfg.timeout_seconds)
        latency_ms = (time.time() - start) * 1000
        logger.info(
            "biomni_task_completed",
            user_id=user_id,
            session_id=session_id,
            latency_ms=round(latency_ms, 2),
            result_len=len(result),
            steps_count=len(steps),
            images_count=len(images),
        )
        return {
            "result": result,
            "success": True,
            "steps": [str(s) for s in steps],
            "images": images,
        }
    except FuturesTimeoutError:
        latency_ms = (time.time() - start) * 1000
        logger.error(
            "biomni_task_timeout",
            user_id=user_id,
            session_id=session_id,
            timeout_seconds=cfg.timeout_seconds,
            latency_ms=round(latency_ms, 2),
        )
        return {
            "result": "",
            "success": False,
            "error": f"Task exceeded timeout of {cfg.timeout_seconds} seconds",
            "steps": [],
            "images": [],
        }
    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        logger.error(
            "biomni_task_failed",
            user_id=user_id,
            session_id=session_id,
            error=str(e),
            latency_ms=round(latency_ms, 2),
            exc_info=True,
        )
        return {
            "result": "",
            "success": False,
            "error": str(e),
            "steps": [],
            "images": [],
        }


def stream_biomni_task(
    query: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    max_retries: int = 3,
) -> Iterator[Dict[str, Any]]:
    """
    Stream BiomNI task execution using agent.go_stream().

    Yields dicts with 'type' key:
      {"type": "step",   "index": int, "content": str}
      {"type": "image",  "data": str}        -- base64 PNG data URI
      {"type": "result", "answer": str, "success": bool, "images": list}
      {"type": "error",  "error": str}
    """
    cfg = get_biomni_config()
    logger.info(
        "biomni_stream_started",
        user_id=user_id,
        session_id=session_id,
        query_len=len(query),
    )

    with _concurrency_limit:
        agent = get_biomni_agent()
        _clear_plots()

        step_index = 0
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                for chunk in agent.go_stream(query):
                    text = ""
                    if isinstance(chunk, dict):
                        text = chunk.get("output", "")
                    elif isinstance(chunk, str):
                        text = chunk
                    if text:
                        yield {"type": "step", "index": step_index, "content": text}
                        step_index += 1

                # Stream completed — capture images
                images = _get_captured_images()
                for img_data in images:
                    yield {"type": "image", "data": img_data}

                # Extract final answer from the agent's last output
                answer = _extract_final_answer(agent)
                yield {
                    "type": "result",
                    "answer": answer,
                    "success": True,
                    "images": images,
                }
                return

            except Exception as e:
                last_error = e
                if _is_rate_limit_error(e) and attempt < max_retries:
                    wait = min(120, 15 * 2 ** attempt)
                    logger.warning(
                        "biomni_stream_throttled",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        wait_seconds=wait,
                    )
                    time.sleep(wait)
                    _clear_plots()
                    step_index = 0
                    continue
                break

        error_msg = str(last_error) if last_error else "Unknown error"
        logger.error("biomni_stream_failed", error=error_msg, exc_info=True)
        yield {"type": "error", "error": error_msg}


def _extract_final_answer(agent: Any) -> str:
    """Extract the final answer from agent's conversation state after go_stream()."""
    # Try _conversation_state (populated by go_stream)
    if hasattr(agent, "_conversation_state") and agent._conversation_state:
        messages = agent._conversation_state.get("messages", [])
        for msg in reversed(messages):
            role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if role == "assistant" and content:
                return str(content)

    # Fallback: try agent.messages or similar attributes
    for attr in ("messages", "_messages", "conversation_history"):
        msgs = getattr(agent, attr, None)
        if msgs and isinstance(msgs, list):
            for msg in reversed(msgs):
                role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
                content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                if role == "assistant" and content:
                    return str(content)

    return ""

"""Biomni agent factory and task runner. Imports from pip package 'biomni'."""
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional

import structlog

from biomni_svc.config import ensure_bedrock_env, get_biomni_config

logger = structlog.get_logger()

_agent_instance: Optional[Any] = None
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="biomni")


def get_biomni_agent():
    """Lazily create and cache Biomni A1 agent instance."""
    global _agent_instance
    if _agent_instance is not None:
        return _agent_instance

    ensure_bedrock_env()
    cfg = get_biomni_config()

    if cfg.path.startswith("s3://"):
        import s3fs  # noqa: F401 - registers fsspec handler for s3://

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

    init_kwargs = {"path": cfg.path, "llm": cfg.llm, "source": cfg.source}
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
        timeout_seconds=cfg.timeout_seconds,
    )
    return agent


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

    Args:
        query: Natural language prompt for the biomedical task.
        user_id: Optional user ID for logging.
        session_id: Optional session ID for logging.
        max_retries: Max retries on Bedrock throttling errors (default 3).
        **kwargs: Additional arguments passed to agent.go().

    Returns:
        Dict with keys: result (str), success (bool), error (optional str), steps (list).
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

    def _run() -> tuple[List[str], str]:
        agent = get_biomni_agent()
        for attempt in range(max_retries + 1):
            try:
                raw = agent.go(query, **kwargs)
                return _parse_go_result(raw)
            except Exception as e:
                if "ThrottlingException" in str(e) and attempt < max_retries:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        "biomni_bedrock_throttled",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        wait_seconds=wait,
                    )
                    time.sleep(wait)
                    continue
                raise

    try:
        future = _executor.submit(_run)
        steps, result = future.result(timeout=cfg.timeout_seconds)
        latency_ms = (time.time() - start) * 1000
        logger.info(
            "biomni_task_completed",
            user_id=user_id,
            session_id=session_id,
            latency_ms=round(latency_ms, 2),
            result_len=len(result),
            steps_count=len(steps),
        )
        return {
            "result": result,
            "success": True,
            "steps": [str(s) for s in steps],
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
        }

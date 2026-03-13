"""Biomni agent factory and task runner. Imports from pip package 'biomni'."""
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, Optional

import structlog

from biomni_svc.config import get_biomni_config

logger = structlog.get_logger()

_agent_instance: Optional[Any] = None
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="biomni")


def get_biomni_agent():
    """Lazily create and cache Biomni A1 agent instance."""
    global _agent_instance
    if _agent_instance is not None:
        return _agent_instance

    cfg = get_biomni_config()

    if cfg.path.startswith("s3://"):
        import s3fs  # noqa: F401 - registers fsspec handler for s3://

    try:
        from biomni.agent import A1
    except ImportError as e:
        raise RuntimeError(
            "Biomni package not installed. Run: pip install biomni langchain-aws"
        ) from e

    init_kwargs = {"path": cfg.path, "llm": cfg.llm, "source": cfg.source}
    if cfg.skip_datalake:
        init_kwargs["expected_data_lake_files"] = []

    agent = A1(**init_kwargs)

    try:
        from biomni.config import default_config
        default_config.timeout_seconds = cfg.timeout_seconds
        default_config.temperature = cfg.temperature
    except ImportError:
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


def run_biomni_task(
    query: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run a Biomni task and return result with metadata.

    Args:
        query: Natural language prompt for the biomedical task.
        user_id: Optional user ID for logging.
        session_id: Optional session ID for logging.
        **kwargs: Additional arguments passed to agent.go().

    Returns:
        Dict with keys: result (str), success (bool), error (optional str).
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

    def _run():
        agent = get_biomni_agent()
        return agent.go(query, **kwargs)

    try:
        future = _executor.submit(_run)
        result = future.result(timeout=cfg.timeout_seconds)
        latency_ms = (time.time() - start) * 1000
        logger.info(
            "biomni_task_completed",
            user_id=user_id,
            session_id=session_id,
            latency_ms=round(latency_ms, 2),
            result_len=len(str(result)) if result else 0,
        )
        return {
            "result": str(result) if result else "",
            "success": True,
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
        }

"""Node-to-node debug printing (stdout, does not depend on structlog)."""
import os

# Set AGENT_DEBUG=0 or LOG_FORMAT=json to disable
_ENABLED = os.getenv("AGENT_DEBUG", "1").strip().lower() not in ("0", "false", "no") and os.getenv("LOG_FORMAT", "").lower() != "json"


def node_start(node_name: str) -> None:
    if _ENABLED:
        print(f"[NODE] {node_name.upper()} → start", flush=True)


def node_end(node_name: str, latency_ms: int = None, extra: str = "") -> None:
    if _ENABLED:
        msg = f"[NODE] {node_name.upper()} → completed"
        if latency_ms is not None:
            msg += f" ({latency_ms}ms)"
        if extra:
            msg += f" {extra}"
        print(msg, flush=True)

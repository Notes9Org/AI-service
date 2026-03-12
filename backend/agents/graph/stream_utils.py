"""Utilities for streaming events from agent nodes."""
from typing import Dict, Any
from agents.graph.state import AgentState


def emit_stream_event(state: AgentState, event_type: str, data: Dict[str, Any]) -> None:
    """Emit a streaming event if stream_callback is set. Safe to call from any node."""
    cb = state.get("stream_callback")
    if cb and callable(cb):
        try:
            cb(event_type, data)
        except Exception:
            pass

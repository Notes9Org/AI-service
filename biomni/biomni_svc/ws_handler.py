"""WebSocket message handler for Biomni streaming."""
import asyncio
import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog
from fastapi import WebSocket, WebSocketDisconnect

from biomni_svc.agent import run_biomni_task
from biomni_svc.clarify import evaluate_clarification
from biomni_svc.storage import upload_biomni_result_to_s3
from services.auth import verify_token_for_websocket

logger = structlog.get_logger()


async def _send(websocket: WebSocket, msg: Dict[str, Any]) -> None:
    """Send a JSON message over the WebSocket."""
    await websocket.send_text(json.dumps(msg))


async def handle_biomni_websocket(
    websocket: WebSocket,
    token: Optional[str] = None,
) -> None:
    """
    Handle WebSocket connection for Biomni streaming.

    Auth: token from query param (?token=JWT) or first message {"type": "auth", "token": "..."}.
    Client sends {"type": "run", "query": "...", "session_id": "..."}.
    Server sends: connected, started, step, result, done.
    """
    await websocket.accept()

    # Resolve token: query param or wait for auth message
    if not token:
        try:
            data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            msg = json.loads(data)
            if msg.get("type") == "auth":
                token = msg.get("token")
            else:
                await _send(websocket, {"type": "error", "error": "Send auth message first: {\"type\": \"auth\", \"token\": \"...\"}"})
                await websocket.close(code=4001)
                return
        except asyncio.TimeoutError:
            await _send(websocket, {"type": "error", "error": "Auth timeout"})
            await websocket.close(code=4001)
            return
        except json.JSONDecodeError:
            await _send(websocket, {"type": "error", "error": "Invalid JSON"})
            await websocket.close(code=4001)
            return

    try:
        current_user = verify_token_for_websocket(token)
    except Exception as e:
        logger.warning("biomni_ws_auth_failed", error=str(e))
        await _send(websocket, {"type": "error", "error": "Invalid token"})
        await websocket.close(code=4001)
        return

    user_id = current_user.user_id  # Always from auth (token); never from client message
    session_id = str(uuid4())
    await _send(websocket, {"type": "connected", "session_id": session_id, "user_id": user_id})

    while True:
        try:
            data = await websocket.receive_text()
        except WebSocketDisconnect:
            break

        try:
            msg = json.loads(data)
        except json.JSONDecodeError:
            await _send(websocket, {"type": "error", "error": "Invalid JSON"})
            continue

        msg_type = msg.get("type")
        if msg_type == "run":
            query = (msg.get("query") or "").strip()
            sess_id = msg.get("session_id") or session_id
            max_retries = msg.get("max_retries", 3)
            history: List[Dict[str, str]] = msg.get("history", [])
            skip_clarify = msg.get("options", {}).get("skip_clarify", False)
            max_clarify_rounds = msg.get("options", {}).get("max_clarify_rounds", 2)
            if not query:
                await _send(websocket, {"type": "error", "error": "Missing query"})
                continue

            run_id = str(uuid4())
            await _send(websocket, {"type": "started", "run_id": run_id})

            # Clarification check
            if not skip_clarify and len(history) < max_clarify_rounds:
                clarify_result = await evaluate_clarification(query, history)
                if clarify_result.needs_clarification and clarify_result.question:
                    await _send(
                        websocket,
                        {
                            "type": "clarify",
                            "question": clarify_result.question,
                            "options": clarify_result.options,
                        },
                    )
                    try:
                        clarify_data = await asyncio.wait_for(websocket.receive_text(), timeout=120.0)
                        clarify_msg = json.loads(clarify_data)
                        if clarify_msg.get("type") == "clarify_response":
                            answer = clarify_msg.get("answer", "")
                            history = history + [
                                {"role": "assistant", "content": clarify_result.question},
                                {"role": "user", "content": answer},
                            ]
                            query = f"{query}\n\nUser clarification: {answer}"
                    except asyncio.TimeoutError:
                        await _send(websocket, {"type": "error", "error": "Clarification timeout"})
                        await _send(websocket, {"type": "done"})
                        continue

            def _run() -> Dict[str, Any]:
                return run_biomni_task(
                    query=query,
                    user_id=user_id,
                    session_id=sess_id,
                    max_retries=max_retries,
                )

            try:
                outcome = await asyncio.to_thread(_run)
            except Exception as e:
                logger.error("biomni_ws_run_failed", run_id=run_id, error=str(e), exc_info=True)
                await _send(websocket, {"type": "error", "error": str(e)})
                await _send(websocket, {"type": "done"})
                continue

            steps = outcome.get("steps", [])
            for i, content in enumerate(steps):
                await _send(websocket, {"type": "step", "index": i, "content": content})

            artifact_url = None
            if outcome.get("success") and outcome.get("result"):
                artifact_url = upload_biomni_result_to_s3(
                    result=outcome["result"],
                    session_id=sess_id,
                    user_id=user_id,
                    metadata={"query": query[:500], "run_id": run_id},
                )

            await _send(
                websocket,
                {
                    "type": "result",
                    "answer": outcome.get("result", ""),
                    "steps": steps,
                    "success": outcome.get("success", False),
                    "artifact_url": artifact_url,
                    "error": outcome.get("error"),
                },
            )
            await _send(websocket, {"type": "done"})

        elif msg_type == "ping":
            await _send(websocket, {"type": "pong"})
        else:
            await _send(websocket, {"type": "error", "error": f"Unknown message type: {msg_type}"})

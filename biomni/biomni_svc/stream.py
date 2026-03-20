"""SSE streaming for Biomni tasks using native go_stream()."""
import asyncio
import json
import time
from queue import Empty, Queue
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import uuid4

import structlog

from biomni_svc.clarify import evaluate_clarification
from biomni_svc.pdf import generate_and_upload_run_pdf
from biomni_svc.storage import upload_biomni_result_to_s3

logger = structlog.get_logger()

PING_INTERVAL_SEC = 15.0


def _format_sse(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def stream_biomni_events(
    query: str,
    user_id: str,
    session_id: str = "",
    max_retries: int = 3,
    history: Optional[List[dict]] = None,
    skip_clarify: bool = False,
    max_clarify_rounds: int = 2,
    generate_pdf: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Async generator that runs Biomni via go_stream() and yields SSE events.

    Event types: started, step, image, clarify, result, error, ping, done
    """
    queue: Queue = Queue()
    run_id = str(uuid4())
    history = history or []

    if not skip_clarify and len(history) < max_clarify_rounds:
        clarify_result = await evaluate_clarification(query, history)
        if clarify_result.needs_clarification and clarify_result.question:
            yield _format_sse("started", {"query": query[:200], "session_id": session_id, "run_id": run_id})
            yield _format_sse("clarify", clarify_result.to_dict())
            yield _format_sse("done", {})
            return

    yield _format_sse("started", {"query": query[:200], "session_id": session_id, "run_id": run_id})

    def _stream_to_queue() -> None:
        try:
            from biomni_svc.agent import stream_biomni_task

            for event in stream_biomni_task(
                query=query,
                user_id=user_id,
                session_id=session_id,
                max_retries=max_retries,
            ):
                queue.put(event)
        except Exception as e:
            queue.put({"type": "error", "error": str(e)})
        finally:
            queue.put(None)  # sentinel

    task = asyncio.to_thread(_stream_to_queue)
    run_task = asyncio.create_task(task)
    last_ping = 0.0
    images_collected: List[str] = []
    steps_collected: List[str] = []

    try:
        while True:
            try:
                event = queue.get(timeout=0.5)
            except Empty:
                if run_task.done():
                    break
                now = time.time()
                if now - last_ping >= PING_INTERVAL_SEC:
                    last_ping = now
                    yield _format_sse("ping", {"ts": now})
                continue

            if event is None:
                break

            event_type = event.get("type", "")

            if event_type == "step":
                steps_collected.append(event["content"])
                yield _format_sse("step", {"index": event["index"], "content": event["content"]})

            elif event_type == "image":
                images_collected.append(event["data"])
                yield _format_sse("image", {"data": event["data"]})

            elif event_type == "result":
                answer = event.get("answer", "")
                success = event.get("success", False)

                artifact_url: Optional[str] = None
                if success and answer:
                    artifact_url = upload_biomni_result_to_s3(
                        result=answer,
                        session_id=session_id,
                        user_id=user_id,
                        metadata={"query": query[:500], "run_id": run_id},
                    )

                pdf_url: Optional[str] = None
                if generate_pdf and success and answer:
                    pdf_url = generate_and_upload_run_pdf(
                        query=query,
                        result=answer,
                        steps=steps_collected,
                        session_id=session_id,
                        user_id=user_id,
                        run_id=run_id,
                    )

                result_payload: Dict[str, Any] = {
                    "answer": answer,
                    "steps": steps_collected,
                    "success": success,
                    "images": images_collected,
                    "artifact_url": artifact_url,
                    "pdf_url": pdf_url,
                }
                yield _format_sse("result", result_payload)

            elif event_type == "error":
                yield _format_sse("error", {"error": event.get("error", "Unknown error")})
                break

        await run_task
        yield _format_sse("done", {})

    except Exception as e:
        logger.error("biomni_stream_failed", run_id=run_id, error=str(e), exc_info=True)
        yield _format_sse("error", {"error": str(e)})

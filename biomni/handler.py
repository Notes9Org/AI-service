"""Production Lambda handler for Biomni Agent. Uses Mangum to adapt FastAPI for API Gateway HTTP API."""
import structlog
from mangum import Mangum
from main import app


class RequestAwareMangum(Mangum):
    """Mangum adapter that binds Lambda request ID to structlog for tracing."""

    def __call__(self, event, context):
        structlog.contextvars.clear_contextvars()
        request_id = (
            getattr(context, "aws_request_id", None)
            or (event.get("requestContext") or {}).get("requestId")
        )
        structlog.contextvars.bind_contextvars(request_id=request_id or "")
        return super().__call__(event, context)


handler = RequestAwareMangum(app, lifespan="on")

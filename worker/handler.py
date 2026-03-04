"""Lambda handler for chunk worker. Supports EventBridge (polling) and Lambda Function URL (Supabase webhook)."""
import json


def handler(event, context):
    from worker import ChunkWorker

    worker = ChunkWorker()
    processed = worker.run_once()

    # Lambda Function URL / API Gateway expect HTTP response format
    if _is_http_invocation(event):
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"processed": processed}),
        }
    # EventBridge / direct invoke
    return {"processed": processed}


def _is_http_invocation(event):
    """Detect Lambda Function URL or API Gateway HTTP invocation."""
    return event.get("version") == "2.0" or "requestContext" in event
"""FastAPI application for Notes9 Agent Chat Service."""
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from api import agent_router, aws_transcribe_router, chat_router
from dotenv import load_dotenv
import structlog
from mangum import Mangum
import uvicorn

# Ensure INFO logs are emitted (structlog may use stdlib underneath)
logging.basicConfig(level=logging.INFO)
# Reduce noise (trace/Supabase best-effort; AWS credentials once)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

try:
    from services.websockets_patch import *  # noqa: F401, F403
except ImportError:
    pass

load_dotenv()

# LangGraph: enable strict msgpack deserialization (CVE mitigation for checkpoint stores)
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

_fallback_renderer = structlog.dev.ConsoleRenderer()

def console_renderer(logger, name, event_dict):
    """Console renderer for agent node events - shows every node transition (start + completed).
    Non-agent messages fall through to the standard console renderer so errors stay visible."""
    if "agent_node" not in event_dict:
        return _fallback_renderer(logger, name, event_dict)
    if "error" in event_dict or "thinking_type" in event_dict:
        return ""
    event = (event_dict.get("event") or event_dict.get("message") or "")
    if not isinstance(event, str):
        event = str(event)
    node = event_dict.get("agent_node", "unknown").upper()
    payload = event_dict.get("payload") or {}
    latency_ms = event_dict.get("latency_ms")
    # Show "started" as a single line so we see node-to-node flow
    if "started" in event.lower():
        print(f"[NODE] {node} → start", flush=True)
        return ""
    # Full block for any "completed" event (including "router_node completed (out_of_scope)" etc.)
    if "completed" not in event.lower():
        return ""
    input_items = {k.replace("input_", ""): v for k, v in payload.items() if isinstance(k, str) and k.startswith("input_")}
    output_items = {k.replace("output_", ""): v for k, v in payload.items() if isinstance(k, str) and k.startswith("output_")}
    # Fallback: pull from event_dict so we always have INPUT/OUTPUT for every node
    if not input_items:
        in_candidates = {}
        for key in ("query", "intent", "normalized_query", "answer_length", "citations_count", "has_summary", "has_judge"):
            if key in event_dict and event_dict[key] is not None:
                in_candidates[key] = event_dict[key]
        if in_candidates:
            input_items = in_candidates
    if not output_items:
        out_candidates = {}
        for key in (
            "intent", "tools", "confidence", "reasoning", "row_count", "has_error", "chunks_found",
            "avg_similarity", "answer_length", "citations_count", "verdict", "answer_preview",
            "tool_used", "route",
        ):
            if key in event_dict and event_dict[key] is not None:
                out_candidates[key] = event_dict[key]
        if "output_generated_sql" in payload:
            out_candidates["generated_sql"] = payload.get("output_generated_sql", "")
        if out_candidates:
            output_items = out_candidates
    # Special handling for SQL node - show generated SQL
    if node == "SQL" and ("output_generated_sql" in payload or "generated_sql" in event_dict):
        if "generated_sql" not in output_items:
            output_items["generated_sql"] = payload.get("output_generated_sql") or event_dict.get("generated_sql", "")
    
    print("-" * 8)
    print(f"🤖 {node}")
    print("-" * 8)
    
    if input_items:
        print("📥 INPUT:")
        for key, value in input_items.items():
            if isinstance(value, str) and len(value) > 250:
                value = value[:250] + "..."
            elif isinstance(value, (list, dict)):
                value_str = str(value)
                if len(value_str) > 250:
                    value = value_str[:250] + "..."
            print(f"   • {key}: {value}")
    
    if output_items:
        print("📤 OUTPUT:")
        for key, value in output_items.items():
            # Special formatting for SQL queries
            if key == "generated_sql" and isinstance(value, str):
                # Show full SQL query for debugging
                print(f"   • {key}:")
                # Print SQL with indentation for readability
                sql_lines = value.split('\n')
                for line in sql_lines:
                    print(f"      {line}")
            elif isinstance(value, str) and len(value) > 250:
                value = value[:250] + "..."
            elif isinstance(value, (list, dict)):
                value_str = str(value)
                if len(value_str) > 250:
                    value = value_str[:250] + "..."
            else:
                print(f"   • {key}: {value}")
    
    if latency_ms is None and isinstance(payload.get("latency_ms"), (int, float)):
        latency_ms = payload.get("latency_ms")
    if latency_ms is not None:
        print(f"⏱️  Latency: {latency_ms}ms")
    print("=" * 80)
    return ""

from services.config import get_app_config, get_llm_provider, get_bedrock_config, get_supabase_config

app_config = get_app_config()
if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    app_config.log_format = "json"
use_json = app_config.log_format == "json"

_processors = [
    structlog.contextvars.merge_contextvars,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.add_log_level,
    structlog.processors.StackInfoRenderer(),
]
if use_json:
    _processors.append(structlog.processors.format_exc_info)
_processors.append(console_renderer if not use_json else structlog.processors.JSONRenderer())
structlog.configure(processors=_processors)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    app_config = get_app_config()
    logger.info("service_starting", service="notes9-agent-chat", version="1.0.0")
    
    try:
        get_supabase_config()
        logger.info("Supabase service: available")
    except Exception as e:
        logger.error("Supabase service: not available", error=str(e))
    
    try:
        # Only Bedrock is supported as LLM provider.
        provider = get_llm_provider()
        if provider != "bedrock":
            logger.warning("Unsupported LLM_PROVIDER; defaulting to 'bedrock'", provider=provider)
        cfg = get_bedrock_config()
        logger.info(
            "LLM: AWS Bedrock",
            provider="bedrock",
            chat_model=cfg.get_chat_model_id(),
            embedding_model=cfg.get_embedding_model(),
            region=cfg.region,
        )
        # Pre-warm embedding service in background (don't block startup — Bedrock cold start can take 1–2 min)
        def _prewarm():
            try:
                from services.embedder import EmbeddingService
                _t0 = time.time()
                EmbeddingService().embed_text("warm")
                logger.info("Embedding service pre-warmed", latency_ms=int((time.time() - _t0) * 1000))
            except Exception as ew:
                logger.warning("Embedding pre-warm skipped", error=str(ew))
        import threading
        threading.Thread(target=_prewarm, daemon=True).start()
    except Exception as e:
        logger.error("LLM/embedding service: not available", error=str(e))
    
    yield
    logger.info("service_shutting_down", service="notes9-agent-chat")


app = FastAPI(
    title="Notes9 Agent Chat Service",
    description="AI-powered chat service for Notes9 scientific lab documentation platform.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in app_config.cors_origins],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests and responses."""
    start_time = time.time()
    logger.info("request_received", method=request.method, path=request.url.path)
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info("request_completed", method=request.method, path=request.url.path, 
                   status_code=response.status_code, process_time_ms=round(process_time * 1000, 2))
        response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
        return response
    except Exception as e:
        logger.error("request_failed", method=request.method, path=request.url.path, error=str(e))
        raise


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning("http_exception", status_code=exc.status_code, detail=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.warning("validation_error", errors=exc.errors())
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "details": exc.errors()},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error("unexpected_error", error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


app.include_router(agent_router)
app.include_router(chat_router)
app.include_router(aws_transcribe_router)

@app.get("/health", tags=["monitoring"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for ECS/Docker health probes."""
    return {"status": "healthy", "service": "notes9-agent-chat", "version": "1.0.0"}


@app.get("/health/ready", tags=["monitoring"])
async def readiness_check() -> Dict[str, Any]:
    """Readiness check for Kubernetes/Docker health probes."""
    checks = {"database": False, "embeddings": False}
    
    try:
        from services.db import SupabaseService
        db = SupabaseService()
        checks["database"] = db.client is not None
    except Exception as e:
        logger.warning("database_check_failed", error=str(e))
    
    try:
        from services.embedder import EmbeddingService
        embedder = EmbeddingService()
        checks["embeddings"] = embedder.client is not None
    except Exception as e:
        logger.warning("embeddings_check_failed", error=str(e))
    
    all_ready = all(checks.values())
    return JSONResponse(
        status_code=200 if all_ready else 503,
        content={"status": "ready" if all_ready else "not_ready", "checks": checks},
    )


@app.get("/", tags=["info"])
async def root() -> Dict[str, Any]:
    """Root endpoint with service information."""
    return {
        "service": "Notes9 Agent Chat Service",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "chat": {
                "stream": "POST /chat/stream",
                "description": "Direct Claude/LLM chat via SSE stream. Bearer token required.",
            },
            "notes9": {
                "stream": "POST /notes9/stream",
                "description": "Full agent pipeline (normalize → router → SQL/RAG → summarizer) via SSE stream. Bearer token required.",
            },
            "AWS_transcribe": {
                "createSession": "POST /AWS_transcribe",
                "description": "Create a Transcribe session for streaming dictation. Returns stream_url for WebSocket. Bearer token required.",
            },
            "monitoring": {"health": "GET /health", "readiness": "GET /health/ready"},
            "documentation": {"swagger": "/docs", "redoc": "/redoc"},
        },
    }

handler = Mangum(app)

if __name__ == "__main__":
    app_config = get_app_config()
    logger.info("starting_server", host=app_config.host, port=app_config.port)
    
    uvicorn.run(
        "main:app",
        host=app_config.host,
        port=app_config.port,
        workers=app_config.workers if not app_config.reload else 1,
        log_level=app_config.log_level,
        reload=app_config.reload,
        access_log=True,
    )
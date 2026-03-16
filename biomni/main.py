"""FastAPI application for Biomni biomedical AI agent."""
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import biomni_router

load_dotenv()

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

use_json = os.environ.get("LOG_FORMAT", "console").lower() == "json"
if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    use_json = True

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer() if use_json else structlog.dev.ConsoleRenderer(),
    ]
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("service_starting", service="biomni-agent", version="1.0.0")
    try:
        from services.config import get_supabase_config
        get_supabase_config()
        logger.info("Supabase auth: available")
    except Exception as e:
        logger.error("Supabase auth: not available", error=str(e))
    try:
        from biomni_svc.config import get_biomni_config
        cfg = get_biomni_config()
        logger.info("Biomni config", path=cfg.path, llm=cfg.llm)
    except Exception as e:
        logger.error("Biomni config: not available", error=str(e))
    yield
    logger.info("service_shutting_down", service="biomni-agent")


app = FastAPI(
    title="Biomni Biomedical AI Agent",
    description="Biomedical research AI agent with MCP tools.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(biomni_router)


@app.get("/health")
async def health():
    """Basic health check."""
    return {"status": "ok", "service": "biomni"}

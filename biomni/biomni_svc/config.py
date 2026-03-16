"""Biomni configuration from environment, aligned with existing Bedrock config."""
import os
from pathlib import Path
from typing import Optional

import structlog
from dotenv import load_dotenv

load_dotenv()

logger = structlog.get_logger()
_biomni_config: Optional["BiomniConfig"] = None

_backend_root = Path(__file__).resolve().parent.parent


def _is_path_writable(path: str) -> bool:
    """Check if path (or its parent) is writable. Returns False on read-only filesystem."""
    try:
        p = Path(path)
        # Test write to parent if path may not exist
        test_dir = p.parent if not p.exists() else p
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file = test_dir / ".biomni_write_test"
        test_file.write_text("")
        test_file.unlink()
        return True
    except OSError:
        return False


def _get_writable_fallback_path() -> Path:
    """Return a writable path for biomni data. Use /tmp in Lambda (read-only /var/task)."""
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return Path("/tmp/biomni")
    return (_backend_root / "data" / "biomni").resolve()


def ensure_bedrock_env() -> None:
    """Set Biomni env vars for Bedrock. Uses existing AWS credentials from boto3 chain."""
    os.environ.setdefault("LLM_SOURCE", "Bedrock")
    if "AWS_REGION" not in os.environ:
        try:
            from services.aws_config import get_bedrock_config
            cfg = get_bedrock_config()
            os.environ["AWS_REGION"] = cfg.region
        except Exception:
            os.environ.setdefault("AWS_REGION", "us-east-1")


def get_biomni_config() -> "BiomniConfig":
    """Get or create Biomni configuration."""
    global _biomni_config
    if _biomni_config is None:
        _biomni_config = BiomniConfig()
    return _biomni_config


class BiomniConfig:
    """Biomni agent configuration from environment variables."""

    def __init__(self):
        # BIOMNI_PATH (EFS/local) or fallback to writable path
        path = os.getenv("BIOMNI_PATH") or os.getenv("BIOMNI_DATA_PATH")
        if path:
            resolved = str(Path(path).resolve())
            # If path is read-only (e.g. /mnt EFS not mounted, or /var/task in Lambda), use writable fallback
            if not _is_path_writable(resolved):
                fallback = _get_writable_fallback_path()
                fallback.mkdir(parents=True, exist_ok=True)
                self.path = str(fallback)
                logger.warning(
                    "biomni_path_fallback",
                    configured=resolved,
                    fallback=str(fallback),
                    reason="Read-only filesystem",
                )
            else:
                self.path = resolved
        else:
            default = _get_writable_fallback_path()
            default.mkdir(parents=True, exist_ok=True)
            self.path = str(default)
        self.llm = os.getenv("BIOMNI_LLM") or os.getenv("BEDROCK_CHAT_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
        self.timeout_seconds = int(os.getenv("BIOMNI_TIMEOUT_SECONDS", "600"))
        self.temperature = float(os.getenv("BIOMNI_TEMPERATURE", "0.7"))
        self.source = os.getenv("BIOMNI_SOURCE", "Bedrock")
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.skip_datalake = os.getenv("BIOMNI_SKIP_DATALAKE", "false").lower() == "true"

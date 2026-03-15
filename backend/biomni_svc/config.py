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
        # BIOMNI_PATH (EFS/local) takes precedence over S3 when set
        path = os.getenv("BIOMNI_PATH") or os.getenv("BIOMNI_DATA_PATH")
        if path:
            resolved = str(Path(path).resolve())
            # If path is under /mnt and read-only, fall back to local storage
            if resolved.startswith("/mnt") and not _is_path_writable(resolved):
                local_fallback = (_backend_root / "data" / "biomni").resolve()
                local_fallback.mkdir(parents=True, exist_ok=True)
                self.path = str(local_fallback)
                logger.warning(
                    "biomni_path_fallback",
                    configured=resolved,
                    fallback=self.path,
                    reason="Read-only filesystem under /mnt",
                )
            else:
                self.path = resolved
        else:
            bucket = os.getenv("BIOMNI_S3_DATALAKE_BUCKET", "").strip()
            prefix = os.getenv("BIOMNI_S3_DATALAKE_PREFIX", "biomni/datalake").strip().rstrip("/")
            if bucket:
                self.path = f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}"
            else:
                default = (_backend_root / "data" / "biomni").resolve()
                self.path = str(default)
        self.llm = os.getenv("BIOMNI_LLM") or os.getenv("BEDROCK_CHAT_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
        self.timeout_seconds = int(os.getenv("BIOMNI_TIMEOUT_SECONDS", "600"))
        self.temperature = float(os.getenv("BIOMNI_TEMPERATURE", "0.7"))
        self.source = os.getenv("BIOMNI_SOURCE", "Bedrock")
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.skip_datalake = os.getenv("BIOMNI_SKIP_DATALAKE", "false").lower() == "true"

"""Biomni configuration from environment, aligned with existing Bedrock config."""
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

_biomni_config: Optional["BiomniConfig"] = None


def get_biomni_config() -> "BiomniConfig":
    """Get or create Biomni configuration."""
    global _biomni_config
    if _biomni_config is None:
        _biomni_config = BiomniConfig()
    return _biomni_config


class BiomniConfig:
    """Biomni agent configuration from environment variables."""

    def __init__(self):
        bucket = os.getenv("BIOMNI_S3_DATALAKE_BUCKET", "").strip()
        prefix = os.getenv("BIOMNI_S3_DATALAKE_PREFIX", "biomni/datalake").strip().rstrip("/")
        if bucket:
            self.path = f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}"
        else:
            self.path = os.getenv("BIOMNI_PATH", os.path.join(os.getcwd(), "data", "biomni"))
        self.llm = os.getenv("BIOMNI_LLM") or os.getenv("BEDROCK_CHAT_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
        self.timeout_seconds = int(os.getenv("BIOMNI_TIMEOUT_SECONDS", "600"))
        self.temperature = float(os.getenv("BIOMNI_TEMPERATURE", "0.7"))
        self.source = os.getenv("BIOMNI_SOURCE", "Bedrock")
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.skip_datalake = os.getenv("BIOMNI_SKIP_DATALAKE", "false").lower() == "true"

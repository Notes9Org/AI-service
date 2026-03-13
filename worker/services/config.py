"""Worker configuration. Supabase, embeddings (Bedrock/Azure), and worker settings."""
import os
from typing import Optional
from dotenv import load_dotenv
import structlog

from services.config_errors import ConfigurationError
from services.aws_config import BedrockConfig, get_bedrock_config
from services.azure_config import AzureOpenAIConfig, get_azure_openai_config

# Patch websockets before importing supabase (must be at module level)
try:
    from services.websockets_patch import *  # noqa: F401, F403
except ImportError:
    pass  # Patch not critical if websockets not installed

load_dotenv()

logger = structlog.get_logger()


class SupabaseConfig:
    """Supabase REST API configuration."""

    def __init__(self):
        """Initialize Supabase configuration."""
        self.url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        self.service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.jwt_secret = os.getenv("SUPABASE_JWT_SECRET", "").strip() or None

        if not self.url or not self.service_key:
            raise ConfigurationError(
                "Supabase service is not available: NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set.\n\n"
                "To fix this:\n"
                "1. Get your Supabase URL and service role key from:\n"
                "   - Go to https://supabase.com/dashboard\n"
                "   - Select your project\n"
                "   - Go to Settings → API\n"
                "   - Copy 'Project URL' and 'service_role' key\n\n"
                "2. Add to your .env file:\n"
                "   NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co\n"
                "   SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here"
            )

    def get_client(self):
        """Get Supabase client (imported here to avoid circular imports)."""
        try:
            from supabase import create_client, Client
            return create_client(self.url, self.service_key)
        except Exception as e:
            raise ConfigurationError(
                f"Supabase service is not available: Failed to create client. Error: {str(e)}"
            ) from e


def get_llm_provider() -> str:
    """Return LLM provider: 'azure' or 'bedrock'. Default is 'azure'."""
    return (os.getenv("LLM_PROVIDER") or "azure").strip().lower()


class AppConfig:
    """Application-level configuration."""

    def __init__(self):
        """Initialize application configuration."""
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.log_format = os.getenv("LOG_FORMAT", "console").lower()
        self.log_level = os.getenv("LOG_LEVEL", "info").lower()

        # Worker configuration
        self.chunk_worker_batch_size = int(os.getenv("CHUNK_WORKER_BATCH_SIZE", "10"))
        self.chunk_worker_poll_interval = int(os.getenv("CHUNK_WORKER_POLL_INTERVAL", "5"))
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

        # Chunking strategy and versioning for semantic chunks
        self.chunking_strategy = os.getenv("CHUNKING_STRATEGY", "semantic").lower()
        # Version tag written into semantic_chunks.metadata.chunk_version
        self.chunk_version = int(os.getenv("CHUNK_VERSION", "2"))


# Global configuration instances (lazy initialization)
_supabase_config: Optional[SupabaseConfig] = None
_app_config: Optional[AppConfig] = None


def get_supabase_config() -> SupabaseConfig:
    """Get or create Supabase configuration."""
    global _supabase_config
    if _supabase_config is None:
        _supabase_config = SupabaseConfig()
    return _supabase_config


def get_app_config() -> AppConfig:
    """Get or create application configuration."""
    global _app_config
    if _app_config is None:
        _app_config = AppConfig()
    return _app_config

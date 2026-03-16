"""Minimal configuration for Biomni agent (auth, Bedrock)."""
import os
from typing import Optional

from dotenv import load_dotenv
from services.config_errors import ConfigurationError

load_dotenv()


class SupabaseConfig:
    """Supabase Auth configuration (JWT verification only)."""

    def __init__(self):
        self.url = os.getenv("NEXT_PUBLIC_SUPABASE_URL", "")
        self.service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        self.jwt_secret = os.getenv("SUPABASE_JWT_SECRET", "").strip() or None
        if not self.jwt_secret:
            raise ConfigurationError(
                "Authentication is not configured: SUPABASE_JWT_SECRET is required for Biomni."
            )


_supabase_config: Optional[SupabaseConfig] = None


def get_supabase_config() -> SupabaseConfig:
    """Get or create Supabase configuration."""
    global _supabase_config
    if _supabase_config is None:
        _supabase_config = SupabaseConfig()
    return _supabase_config

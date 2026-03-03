"""Shared configuration exception (used by config and aws_config)."""


class ConfigurationError(Exception):
    """Raised when configuration is invalid or service is unavailable."""

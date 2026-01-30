"""Shared configuration exception (used by config, aws_config, azure_config)."""


class ConfigurationError(Exception):
    """Raised when configuration is invalid or service is unavailable."""

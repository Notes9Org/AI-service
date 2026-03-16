"""MCP (Model Context Protocol) integration for Biomni."""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger()

_mcp_servers_cache: Optional[List[Dict[str, Any]]] = None


def get_mcp_config_path() -> Optional[str]:
    """Return path to MCP config YAML if BIOMNI_MCP_CONFIG_PATH is set."""
    path = os.getenv("BIOMNI_MCP_CONFIG_PATH", "").strip()
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning("biomni_mcp_config_not_found", path=path)
        return None
    return str(p.resolve())


def attach_mcp_to_agent(agent: Any) -> bool:
    """
    Attach MCP servers to the Biomni agent if config is set.

    Returns True if MCP was attached, False otherwise.
    """
    config_path = get_mcp_config_path()
    if not config_path:
        return False
    try:
        agent.add_mcp(config_path=config_path)
        logger.info("biomni_mcp_attached", config_path=config_path)
        return True
    except Exception as e:
        logger.error("biomni_mcp_attach_failed", config_path=config_path, error=str(e), exc_info=True)
        return False


def list_mcp_servers() -> List[Dict[str, Any]]:
    """
    List registered MCP servers and their tools.

    Returns list of {"name": str, "tools": [...]} or [] if agent not initialized.
    """
    try:
        from biomni_svc.agent import get_biomni_agent

        agent = get_biomni_agent()
        if not hasattr(agent, "list_mcp_servers"):
            return []
        servers = agent.list_mcp_servers()
        if isinstance(servers, dict):
            return [
                {"name": k, "tools": v if isinstance(v, list) else []}
                for k, v in servers.items()
            ]
        if isinstance(servers, list):
            return [
                {"name": s, "tools": []} if isinstance(s, str) else s
                for s in servers
            ]
        return []
    except Exception as e:
        logger.warning("biomni_mcp_list_failed", error=str(e))
        return []


def test_mcp_connection() -> Dict[str, Any]:
    """
    Test MCP server connections.

    Returns {"ok": bool, "servers": [...], "errors": [...]}.
    """
    config_path = get_mcp_config_path()
    if not config_path:
        return {"ok": False, "servers": [], "errors": ["BIOMNI_MCP_CONFIG_PATH not set"]}
    try:
        from biomni_svc.agent import get_biomni_agent

        agent = get_biomni_agent()
        if not hasattr(agent, "test_mcp_connection"):
            return {"ok": True, "servers": [], "errors": [], "message": "test_mcp_connection not available"}
        results = agent.test_mcp_connection(config_path)
        if isinstance(results, dict):
            errors = [v for k, v in results.items() if not v.get("ok", True)]
            return {"ok": len(errors) == 0, "servers": list(results.keys()), "errors": errors}
        return {"ok": True, "servers": [], "errors": []}
    except Exception as e:
        logger.warning("biomni_mcp_test_failed", error=str(e))
        return {"ok": False, "servers": [], "errors": [str(e)]}

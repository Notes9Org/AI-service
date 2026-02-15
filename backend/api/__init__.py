"""API routes - agent, chat, literature."""
from api.agent import router as agent_router
from api.chat import router as chat_router
from api.literature import router as literature_router

__all__ = ["agent_router", "chat_router", "literature_router"]

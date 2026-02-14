"""API routes - agent, chat, transcribe, literature."""
from api.agent import router as agent_router
from api.chat import router as chat_router
from api.transcribe import router as transcribe_router
from api.literature import router as literature_router

__all__ = ["agent_router", "chat_router", "transcribe_router", "literature_router"]

"""CopilotLibrary - Python client for GitHub Copilot language-server.

This library provides a clean API for interacting with the copilot-language-server
binary that JetBrains IDEs use, enabling chat, multi-turn conversations, and more.

Example usage:

    from copilotlibrary import CopilotClient

    # Simple one-shot chat
    with CopilotClient() as client:
        response = client.chat("Explain binary search")
        print(response.content)

    # Multi-turn conversation
    with CopilotClient() as client:
        conv = client.create_conversation("You are a Python expert.")
        
        response = client.send_message(conv, "What is a decorator?")
        print(response.content)
        
        response = client.send_message(conv, "Show me an example")
        print(response.content)
"""

from .client import CopilotClient
from .embeddings import CopilotEmbeddings, EmbeddingResult, find_copilot_token, get_embedding, get_embeddings_batch
from .models import (
    Agent,
    AuthStatus,
    ChatMessage,
    ChatResponse,
    Conversation,
    ConversationMode,
    CopilotModel,
    Embedding,
    FeatureFlags,
    Mode,
    Model,
    ServerVersion,
    SignInInfo,
    StatusResult,
    Template,
    TokenUsage,
)

__version__ = "0.2.0"

__all__ = [
    # Main client
    "CopilotClient",
    # Remote embeddings (auto-auth via apps.json)
    "CopilotEmbeddings",
    "EmbeddingResult",
    "find_copilot_token",
    "get_embedding",
    "get_embeddings_batch",
    # Data models
    "Agent",
    "AuthStatus",
    "ChatMessage",
    "ChatResponse",
    "Conversation",
    "ConversationMode",
    "CopilotModel",
    "Embedding",
    "FeatureFlags",
    "Mode",
    "Model",
    "ServerVersion",
    "SignInInfo",
    "StatusResult",
    "Template",
    "TokenUsage",
    # Version
    "__version__",
]

"""Data models for CopilotLibrary API responses.

These dataclasses provide clean, typed representations of the copilot-language-server
responses, making the API easier to use and more predictable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AuthStatus(Enum):
    """Authentication status returned by checkStatus."""
    OK = "OK"
    ALREADY_SIGNED_IN = "AlreadySignedIn"
    NOT_SIGNED_IN = "NotSignedIn"
    UNKNOWN = "Unknown"

    @classmethod
    def from_str(cls, value: str) -> AuthStatus:
        """Convert string status to enum value."""
        mapping = {
            "OK": cls.OK,
            "AlreadySignedIn": cls.ALREADY_SIGNED_IN,
            "NotSignedIn": cls.NOT_SIGNED_IN,
        }
        return mapping.get(value, cls.UNKNOWN)


@dataclass
class StatusResult:
    """Result of a checkStatus request."""
    status: AuthStatus
    user: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_authenticated(self) -> bool:
        """Check if the user is authenticated."""
        return self.status in (AuthStatus.OK, AuthStatus.ALREADY_SIGNED_IN)


@dataclass
class Model:
    """Represents an available AI model."""
    id: str
    name: str
    vendor: str | None = None
    version: str | None = None
    family: str | None = None
    supports_chat: bool = True
    supports_embeddings: bool = False
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class ChatMessage:
    """A single message in a conversation."""
    role: str  # "user", "assistant", or "system"
    content: str
    name: str | None = None  # Optional name for multi-agent scenarios

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format for API calls."""
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class ChatResponse:
    """Response from a chat request."""
    content: str
    conversation_id: str | None = None
    turn_id: str | None = None
    model_name: str | None = None
    suggested_title: str | None = None
    finish_reason: str | None = None
    token_usage: TokenUsage | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class TokenUsage:
    """Token usage statistics for a request."""
    total_tokens: int = 0
    system_prompt_tokens: int = 0
    user_messages_tokens: int = 0
    assistant_messages_tokens: int = 0
    total_token_limit: int = 0
    utilization_percentage: float = 0.0

    @classmethod
    def from_context_size(cls, data: dict[str, Any]) -> TokenUsage:
        """Create from a contextSize dictionary in progress notifications."""
        return cls(
            total_tokens=data.get("totalUsedTokens", 0),
            system_prompt_tokens=data.get("systemPromptTokens", 0),
            user_messages_tokens=data.get("userMessagesTokens", 0),
            assistant_messages_tokens=data.get("assistantMessagesTokens", 0),
            total_token_limit=data.get("totalTokenLimit", 0),
            utilization_percentage=data.get("utilizationPercentage", 0.0),
        )


@dataclass
class Embedding:
    """An embedding vector result."""
    vector: list[float]
    model: str
    input_text: str
    dimensions: int = 0

    def __post_init__(self) -> None:
        if not self.dimensions:
            self.dimensions = len(self.vector)


@dataclass
class Conversation:
    """Represents an ongoing conversation with history."""
    id: str | None = None
    system_prompt: str = "You are a helpful coding assistant."
    messages: list[ChatMessage] = field(default_factory=list)
    model: str | None = None
    mode: str = "Ask"  # Ask, Edit, Agent, Plan
    agent: str | None = None  # github, project, etc.
    title: str | None = None

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append(ChatMessage(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation."""
        self.messages.append(ChatMessage(role="assistant", content=content))

    def get_turns(self) -> list[dict[str, str]]:
        """Convert messages to turns format for API."""
        turns = []
        i = 0
        while i < len(self.messages):
            msg = self.messages[i]
            if msg.role == "user":
                # Find the corresponding assistant response if it exists
                response = ""
                if i + 1 < len(self.messages) and self.messages[i + 1].role == "assistant":
                    response = self.messages[i + 1].content
                    i += 1
                turns.append({"request": msg.content, "response": response})
            i += 1
        return turns

    def clear(self) -> None:
        """Clear conversation history but keep system prompt."""
        self.messages.clear()
        self.id = None
        self.title = None


@dataclass 
class SignInInfo:
    """Information needed for device flow sign-in."""
    verification_uri: str
    user_code: str
    device_code: str | None = None
    expires_in: int | None = None
    interval: int | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class FeatureFlags:
    """Feature flags from the copilot server."""
    chat_enabled: bool = False
    agent_as_default: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureFlags:
        return cls(
            chat_enabled=data.get("chat", False),
            agent_as_default=data.get("agent_as_default", False),
            raw=data,
        )


class ConversationMode(Enum):
    """Available conversation modes."""
    ASK = "Ask"
    EDIT = "Edit"
    AGENT = "Agent"
    PLAN = "Plan"

    @classmethod
    def from_str(cls, value: str) -> ConversationMode:
        """Convert string to mode enum."""
        for mode in cls:
            if mode.value.lower() == value.lower():
                return mode
        return cls.ASK


@dataclass
class Mode:
    """Represents a conversation mode (Ask, Edit, Agent, etc.)."""
    id: str
    name: str
    kind: str
    description: str = ""
    is_builtin: bool = True
    uri: str | None = None
    custom_tools: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Mode:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            kind=data.get("kind", ""),
            description=data.get("description", ""),
            is_builtin=data.get("isBuiltIn", True),
            uri=data.get("uri"),
            custom_tools=data.get("customTools", []),
            raw=data,
        )


@dataclass
class Agent:
    """Represents a Copilot agent (github, project, etc.)."""
    slug: str
    name: str
    description: str = ""
    avatar_url: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Agent:
        return cls(
            slug=data.get("slug", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            avatar_url=data.get("avatarUrl"),
            raw=data,
        )


@dataclass
class Template:
    """Represents a conversation template (tests, fix, explain, etc.)."""
    id: str
    description: str
    short_description: str = ""
    scopes: list[str] = field(default_factory=list)
    source: str = "builtin"
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Template:
        return cls(
            id=data.get("id", ""),
            description=data.get("description", ""),
            short_description=data.get("shortDescription", ""),
            scopes=data.get("scopes", []),
            source=data.get("source", "builtin"),
            raw=data,
        )


@dataclass
class CopilotModel:
    """Detailed model info from copilot/models API."""
    id: str
    name: str
    family: str
    scopes: list[str] = field(default_factory=list)
    is_preview: bool = False
    is_chat_default: bool = False
    is_premium: bool = False
    premium_multiplier: float = 1.0
    supports_vision: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CopilotModel:
        capabilities = data.get("capabilities", {})
        supports = capabilities.get("supports", {})
        billing = data.get("billing", {})
        return cls(
            id=data.get("id", ""),
            name=data.get("modelName", data.get("id", "")),
            family=data.get("modelFamily", ""),
            scopes=data.get("scopes", []),
            is_preview=data.get("preview", False),
            is_chat_default=data.get("isChatDefault", False),
            is_premium=billing.get("isPremium", False),
            premium_multiplier=billing.get("multiplier", 1.0),
            supports_vision=supports.get("vision", False),
            raw=data,
        )

    @property
    def supports_chat(self) -> bool:
        return "chat-panel" in self.scopes

    @property
    def supports_edit(self) -> bool:
        return "edit-panel" in self.scopes

    @property
    def supports_agent(self) -> bool:
        return "agent-panel" in self.scopes

    @property
    def supports_inline(self) -> bool:
        return "inline" in self.scopes


@dataclass
class ServerVersion:
    """Server version information."""
    version: str
    build_type: str = ""
    runtime_version: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ServerVersion:
        result = data.get("result", data)
        return cls(
            version=result.get("version", ""),
            build_type=result.get("buildType", ""),
            runtime_version=result.get("runtimeVersion", ""),
            raw=data,
        )



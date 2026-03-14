# CopilotLibrary API Documentation

A Python client for the GitHub Copilot language-server. Enables chat, multi-turn conversations, and more via the local copilot-language-server binary that JetBrains IDEs use.

## Installation

```bash
# From the repository root
pip install -e .

# Or install dependencies manually
pip install .
```

## Quick Start

```python
from copilotlibrary import CopilotClient

# Simple one-shot chat
with CopilotClient() as client:
    response = client.chat("Explain binary search in Python")
    print(response.content)
```

## Prerequisites

- Python 3.11+
- IntelliJ IDEA (or other JetBrains IDE) with the GitHub Copilot plugin installed
- Signed in to GitHub Copilot in the IDE

The library automatically detects the copilot-language-server binary from your JetBrains IDE installation.

---

## Core Classes

### CopilotClient

The main client for interacting with the Copilot language-server.

#### Constructor

```python
CopilotClient(
    *,
    agent_path: str | None = None,
    on_progress: Callable[[dict], None] | None = None,
)
```

**Parameters:**
- `agent_path`: Path to the copilot-language-server binary. Optional - auto-detected from IntelliJ plugin or `COPILOT_AGENT_PATH` environment variable.
- `on_progress`: Optional callback function for streaming progress notifications.

#### Usage

```python
# Using context manager (recommended)
with CopilotClient() as client:
    response = client.chat("Hello!")
    print(response.content)

# Manual lifecycle management
client = CopilotClient()
client.start()
try:
    response = client.chat("Hello!")
finally:
    client.stop()
```

---

## API Reference

### Authentication & Status

#### `check_status() -> StatusResult`

Check authentication status with GitHub Copilot.

```python
with CopilotClient() as client:
    status = client.check_status()
    print(f"User: {status.user}")
    print(f"Authenticated: {status.is_authenticated}")
    print(f"Status: {status.status.value}")
```

**Returns:** `StatusResult` with:
- `status`: `AuthStatus` enum (OK, ALREADY_SIGNED_IN, NOT_SIGNED_IN, UNKNOWN)
- `user`: Username string or None
- `is_authenticated`: Boolean property
- `raw`: Raw response dictionary

#### `sign_in_initiate() -> SignInInfo`

Start device-flow sign-in process.

```python
sign_in_info = client.sign_in_initiate()
print(f"Go to: {sign_in_info.verification_uri}")
print(f"Enter code: {sign_in_info.user_code}")
```

**Returns:** `SignInInfo` with:
- `verification_uri`: URL to open in browser
- `user_code`: Code to enter on the verification page
- `device_code`: Device code for confirmation
- `expires_in`: Seconds until expiration

#### `sign_in_confirm(user_code: str) -> StatusResult`

Complete device-flow sign-in after user authorization.

```python
status = client.sign_in_confirm(sign_in_info.user_code)
if status.is_authenticated:
    print("Successfully signed in!")
```

#### `sign_out() -> StatusResult`

Sign out from GitHub Copilot.

---

### Chat (Simple)

#### `chat(prompt, *, system=None, model=None, doc_uri=None) -> ChatResponse`

Send a single chat prompt and get a response.

```python
with CopilotClient() as client:
    # Basic usage
    response = client.chat("What is Python?")
    print(response.content)
    
    # With custom system prompt
    response = client.chat(
        "Optimize this code",
        system="You are a performance optimization expert.",
    )
    
    # With model selection (if supported)
    response = client.chat("Hello", model="gpt-4o")
```

**Parameters:**
- `prompt`: The user's message
- `system`: System prompt to set assistant behavior (default: "You are a helpful coding assistant.")
- `model`: Model ID to use (server may ignore if not supported)
- `doc_uri`: Document URI context (default: "file:///untitled")

**Returns:** `ChatResponse` with:
- `content`: The assistant's reply text
- `conversation_id`: Unique conversation identifier
- `turn_id`: Unique turn identifier
- `model_name`: Model used (e.g., "GPT-4o")
- `suggested_title`: Auto-generated title for the conversation
- `token_usage`: TokenUsage object with token statistics
- `raw`: Raw response dictionary

---

### Multi-turn Conversations

#### `create_conversation(system_prompt=None, model=None, mode="Ask", agent=None) -> Conversation`

Create a new conversation for multi-turn chat.

```python
# Simple conversation
conv = client.create_conversation("You are a Python expert.")

# With specific mode and model
conv = client.create_conversation(
    mode="Agent",
    model="claude-sonnet-4.6",
)

# With agent
conv = client.create_conversation(
    mode="Agent",
    agent="github",
)
```

**Parameters:**
- `system_prompt`: System prompt (default: "You are a helpful coding assistant.")
- `model`: Model ID (e.g., "gpt-4o", "claude-sonnet-4.6")
- `mode`: Conversation mode - "Ask", "Edit", "Agent", or "Plan"
- `agent`: Agent to use (e.g., "github", "project")

**Returns:** `Conversation` object

#### `send_message(conversation, message, *, doc_uri=None) -> ChatResponse`

Send a message in an existing conversation.

```python
with CopilotClient() as client:
    # Create conversation with Agent mode
    conv = client.create_conversation(mode="Agent")
    
    # First message
    response1 = client.send_message(conv, "What are decorators?")
    print(response1.content)
    
    # Follow-up (context is maintained)
    response2 = client.send_message(conv, "Show me an example")
    print(response2.content)
    
    # Check message history
    print(f"Messages: {len(conv.messages)}")
    for msg in conv.messages:
        print(f"  {msg.role}: {msg.content[:50]}...")
    
    # Access conversation history
    for msg in conv.messages:
        print(f"{msg.role}: {msg.content[:50]}...")
```

**Parameters:**
- `conversation`: Conversation object from create_conversation()
- `message`: The user's message
- `doc_uri`: Document URI context

**Returns:** `ChatResponse`

#### `get_conversation(conversation_id: str) -> Conversation | None`

Retrieve a conversation by ID.

#### `list_conversations() -> list[Conversation]`

List all active conversations in this session.

---

### Models

#### `get_models() -> list[Model]`

Get available models. Returns known defaults if server API unavailable.

```python
models = client.get_models()
for model in models:
    print(f"{model.id}: {model.name}")
    print(f"  Vendor: {model.vendor}")
    print(f"  Chat: {model.supports_chat}")
    print(f"  Embeddings: {model.supports_embeddings}")
```

**Returns:** List of `Model` objects with:
- `id`: Model identifier
- `name`: Display name
- `vendor`: Model vendor (e.g., "OpenAI")
- `supports_chat`: Whether model supports chat
- `supports_embeddings`: Whether model supports embeddings
- `max_input_tokens`: Maximum input token limit
- `max_output_tokens`: Maximum output token limit

#### `get_chat_models() -> list[Model]`

Get models that support chat completions.

#### `get_embedding_models() -> list[Model]`

Get models that support embeddings.

---

### Server Discovery

#### `get_version() -> ServerVersion`

Get server version information.

```python
version = client.get_version()
print(f"Version: {version.version}")
print(f"Build: {version.build_type}")
print(f"Runtime: {version.runtime_version}")
```

#### `get_modes() -> list[Mode]`

Get available conversation modes (Ask, Edit, Agent, Plan).

```python
for mode in client.get_modes():
    print(f"{mode.name} ({mode.kind}): {mode.description}")
```

**Returns:** List of `Mode` objects with:
- `id`: Mode identifier
- `name`: Display name
- `kind`: Mode type (Ask, Edit, Agent)
- `description`: Mode description
- `is_builtin`: Whether it's a built-in mode
- `custom_tools`: List of tools available in this mode

#### `get_agents() -> list[Agent]`

Get available agents (github, project, etc.).

```python
for agent in client.get_agents():
    print(f"{agent.slug}: {agent.name}")
    print(f"  {agent.description}")
```

**Returns:** List of `Agent` objects with:
- `slug`: Agent identifier (e.g., "github", "project")
- `name`: Display name
- `description`: What the agent does
- `avatar_url`: Avatar image URL

#### `get_templates() -> list[Template]`

Get available conversation templates (tests, fix, explain, etc.).

```python
for template in client.get_templates():
    print(f"/{template.id}: {template.short_description}")
    print(f"  {template.description}")
```

**Returns:** List of `Template` objects with:
- `id`: Template identifier (tests, fix, explain, doc, simplify)
- `description`: Full description
- `short_description`: Brief label
- `scopes`: Where template can be used

#### `get_copilot_models() -> list[CopilotModel]`

Get detailed model information from the Copilot server.

```python
for model in client.get_copilot_models():
    print(f"{model.id}: {model.name}")
    print(f"  Family: {model.family}")
    print(f"  Premium: {model.is_premium} (x{model.premium_multiplier})")
    print(f"  Vision: {model.supports_vision}")
    print(f"  Scopes: {', '.join(model.scopes)}")
```

**Returns:** List of `CopilotModel` objects with:
- `id`: Model identifier
- `name`: Display name
- `family`: Model family
- `scopes`: Supported scopes (chat-panel, edit-panel, agent-panel, inline)
- `is_premium`: Whether it's a premium model
- `premium_multiplier`: Token cost multiplier
- `supports_vision`: Whether model supports vision/images
- `is_preview`: Whether it's a preview model

#### `get_preconditions() -> dict`

Check conversation preconditions (authentication, chat enabled, etc.).

```python
pre = client.get_preconditions()
for result in pre.get("results", []):
    print(f"{result['type']}: {result['status']}")
```

---

### Embeddings

#### `get_embedding(text, *, model=None) -> list[float]`

Get embedding vector for text.

> **Note:** The copilot-language-server may not support direct embeddings. This method will raise `NotImplementedError` if unavailable.

```python
try:
    embedding = client.get_embedding("Hello world")
    print(f"Dimensions: {len(embedding)}")
except NotImplementedError:
    print("Embeddings not available - use OpenAI API directly")
```

**Parameters:**
- `text`: Text to embed
- `model`: Embedding model to use (default: "text-embedding-3-small")

**Returns:** List of floats representing the embedding vector

**Raises:** `NotImplementedError` if embeddings API is not available

#### `get_embeddings_batch(texts, *, model=None) -> list[list[float]]`

Get embedding vectors for multiple texts.

---

### Lifecycle Management

#### `start() -> None`

Launch the language-server and perform LSP handshake.

```python
client = CopilotClient()
client.start()
# ... use client ...
client.stop()
```

#### `stop() -> None`

Terminate the language-server process.

#### `is_running() -> bool`

Check if the language-server process is running.

---

## Data Models

### ChatResponse

```python
@dataclass
class ChatResponse:
    content: str                    # The assistant's reply
    conversation_id: str | None     # Conversation ID
    turn_id: str | None             # Turn ID
    model_name: str | None          # Model used
    suggested_title: str | None     # Auto-generated title
    finish_reason: str | None       # Why generation stopped
    token_usage: TokenUsage | None  # Token statistics
    raw: dict                       # Raw response
```

### Conversation

```python
@dataclass
class Conversation:
    id: str | None                  # Server-assigned conversation ID
    system_prompt: str              # System prompt
    messages: list[ChatMessage]     # Message history
    model: str | None               # Model preference
    mode: str = "Ask"               # Mode: Ask, Edit, Agent, Plan
    agent: str | None               # Agent: github, project, etc.
    title: str | None               # Auto-generated title
    
    def add_user_message(content: str) -> None
    def add_assistant_message(content: str) -> None
    def get_turns() -> list[dict[str, str]]
    def clear() -> None
```

### ChatMessage

```python
@dataclass
class ChatMessage:
    role: str       # "user", "assistant", or "system"
    content: str    # Message content
    name: str | None  # Optional name
```

### TokenUsage

```python
@dataclass
class TokenUsage:
    total_tokens: int
    system_prompt_tokens: int
    user_messages_tokens: int
    assistant_messages_tokens: int
    total_token_limit: int
    utilization_percentage: float
```

### Model

```python
@dataclass
class Model:
    id: str                         # Model ID
    name: str                       # Display name
    vendor: str | None              # Vendor (e.g., "OpenAI")
    version: str | None             # Version
    family: str | None              # Model family
    supports_chat: bool = True      # Supports chat
    supports_embeddings: bool = False  # Supports embeddings
    max_input_tokens: int | None    # Input token limit
    max_output_tokens: int | None   # Output token limit
```

### StatusResult

```python
@dataclass
class StatusResult:
    status: AuthStatus              # Auth status enum
    user: str | None                # Username
    raw: dict                       # Raw response
    
    @property
    def is_authenticated(self) -> bool
```

### AuthStatus (Enum)

```python
class AuthStatus(Enum):
    OK = "OK"
    ALREADY_SIGNED_IN = "AlreadySignedIn"
    NOT_SIGNED_IN = "NotSignedIn"
    UNKNOWN = "Unknown"
```

---

## CLI Usage

```bash
# One-shot prompt
copilotlibrary "Explain binary search"

# Or with words (automatically joined)
copilotlibrary Hello how are you

# Check status
copilotlibrary status

# List models
copilotlibrary models
copilotlibrary models --filter chat
copilotlibrary models --filter embeddings

# Chat with verbose output
copilotlibrary chat "Hello world" -v

# Interactive mode (no arguments)
copilotlibrary
```

### Interactive Mode Commands

```
/help     - Show help
/new      - Start a new conversation
/history  - Show conversation history
/system <prompt> - Set new system prompt
/exit, /quit     - Exit
```

---

## REPL Tool

An interactive REPL for exploring all Copilot features:

```bash
# Run the REPL
copilot-repl

# Or directly
python -m copilotlibrary.repl
```

### REPL Commands

| Category | Command | Description |
|----------|---------|-------------|
| **Chat** | `chat <msg>` | Send message (or just type) |
| | `new` | Start new conversation |
| | `history` | Show conversation history |
| | `clear` | Clear conversation |
| **Config** | `mode [name]` | Get/set mode (Ask, Edit, Agent, Plan) |
| | `model [name]` | Get/set model |
| | `agent [name]` | Get/set agent |
| | `system [text]` | Get/set system prompt |
| | `verbose [on\|off]` | Toggle verbose mode |
| **Discovery** | `status` | Show auth status |
| | `version` | Show server version |
| | `models` | List available models |
| | `modes` | List conversation modes |
| | `agents` | List available agents |
| | `templates` | List templates |
| | `capabilities` | Show server capabilities |
| **Raw** | `raw <method> [json]` | Send raw JSON-RPC |

### Example REPL Session

```
$ copilot-repl

╔══════════════════════════════════════════════════════════════════╗
║           CopilotLibrary REPL - Interactive Explorer             ║
╚══════════════════════════════════════════════════════════════════╝

Connecting to Copilot server...
✓ Connected as: username
✓ Server version: 1.442.0

🤖 copilot> mode Agent
✓ Mode set to: Agent

🤖 copilot> model claude-sonnet-4.6
✓ Model set to: claude-sonnet-4.6

🤖 copilot> What is Python?
📝 Python is a high-level, interpreted programming language...

🤖 copilot> models
🤖 Available Models (10)
──────────────────────────────────────────────────────────
  claude-opus-4.6 💎
    Name: Claude Opus 4.6
    Scopes: chat-panel, edit-panel, agent-panel
    Premium multiplier: 3x

🤖 copilot> quit
Goodbye!
```

---

## Progress Callbacks

For streaming responses, you can register a callback:

```python
def on_progress(msg: dict):
    value = msg.get("params", {}).get("value", {})
    if value.get("kind") == "report":
        # Extract partial reply
        reply = value.get("reply") or ""
        edit_rounds = value.get("editAgentRounds", [])
        if edit_rounds:
            reply = edit_rounds[0].get("reply", "")
        print(reply, end="", flush=True)

with CopilotClient(on_progress=on_progress) as client:
    response = client.chat("Write a poem")
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `COPILOT_AGENT_PATH` | Path to copilot-language-server binary |

---

## Binary Path Resolution

The library looks for the copilot-language-server binary in this order:

1. `agent_path` parameter passed to `CopilotClient()`
2. `COPILOT_AGENT_PATH` environment variable
3. Auto-detected from JetBrains IDE plugins:

| OS | JetBrains Root | Architecture |
|----|----------------|--------------|
| macOS | `~/Library/Application Support/JetBrains/` | darwin-arm64 / darwin-x64 |
| Windows | `%APPDATA%\JetBrains\` | win32-arm64 / win32-x64 |
| Linux | `~/.config/JetBrains/` | linux-arm64 / linux-x64 |

---

## Error Handling

```python
from copilotlibrary import CopilotClient

try:
    with CopilotClient() as client:
        response = client.chat("Hello")
except FileNotFoundError as e:
    print(f"Binary not found: {e}")
    print("Set COPILOT_AGENT_PATH or install IntelliJ Copilot plugin")
except RuntimeError as e:
    print(f"API error: {e}")
```

---

## Examples

### Basic Chat

```python
from copilotlibrary import CopilotClient

with CopilotClient() as client:
    response = client.chat("Write a Python function to reverse a string")
    print(response.content)
    print(f"Model: {response.model_name}")
```

### Multi-turn Code Review

```python
from copilotlibrary import CopilotClient

code = '''
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
'''

with CopilotClient() as client:
    conv = client.create_conversation("You are a code review expert.")
    
    r1 = client.send_message(conv, f"Review this code:\n```python\n{code}\n```")
    print("Review:", r1.content)
    
    r2 = client.send_message(conv, "How can I optimize it?")
    print("Optimization:", r2.content)
    
    r3 = client.send_message(conv, "Show me the improved code")
    print("Improved:", r3.content)
```

### Check Authentication

```python
from copilotlibrary import CopilotClient

with CopilotClient() as client:
    status = client.check_status()
    
    if not status.is_authenticated:
        print("Not signed in. Starting device flow...")
        info = client.sign_in_initiate()
        print(f"Go to: {info.verification_uri}")
        print(f"Code: {info.user_code}")
        input("Press Enter after signing in...")
        
        status = client.sign_in_confirm(info.user_code)
        if status.is_authenticated:
            print(f"Signed in as {status.user}")
    else:
        print(f"Already signed in as {status.user}")
```

### List Available Models

```python
from copilotlibrary import CopilotClient

with CopilotClient() as client:
    print("Chat Models:")
    for model in client.get_chat_models():
        print(f"  - {model.id}: {model.name}")
    
    print("\nEmbedding Models:")
    for model in client.get_embedding_models():
        print(f"  - {model.id}: {model.name}")
```

---

## Version History

### 0.2.0
- Added proper response extraction from `editAgentRounds`
- Multi-turn conversation support
- Typed data models (ChatResponse, Conversation, Model, etc.)
- Progress callback support
- Model listing API
- Embedding API (placeholder)
- Improved CLI with subcommands
- Comprehensive error handling

### 0.1.0
- Initial release
- Basic chat functionality
- LSP/JSON-RPC communication


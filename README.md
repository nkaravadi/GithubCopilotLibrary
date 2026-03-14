# CopilotLibrary

A Python client for the GitHub Copilot language-server. Enables chat, multi-turn conversations, and more via the local copilot-language-server binary that JetBrains IDEs use.

**No API token needed** - uses your existing IDE authentication!

## Features

- ✅ **Simple chat** - one-shot prompts
- ✅ **Multi-turn conversations** - context-aware follow-ups
- ✅ **Typed responses** - clean dataclasses (ChatResponse, Conversation, etc.)
- ✅ **Progress callbacks** - streaming support
- ✅ **Model listing** - discover available models
- ✅ **CLI** - command-line interface with interactive mode
- ✅ **Zero runtime dependencies** - stdlib only

## Quick Start

```python
from copilotlibrary import CopilotClient

# Simple one-shot chat
with CopilotClient() as client:
    response = client.chat("Explain binary search in Python")
    print(response.content)
```

## Prerequisites

1. **Python 3.11+**
2. **JetBrains IDE** (IntelliJ IDEA, PyCharm, etc.) with GitHub Copilot plugin installed
3. **Signed in** to GitHub Copilot in the IDE

The library automatically detects the copilot-language-server binary from your JetBrains IDE installation.

## Installation

```bash
# Clone and install
git clone <repo-url>
cd CopilotLibrary
pip install -e .
```

## Usage Examples

### Simple Chat

```python
from copilotlibrary import CopilotClient

with CopilotClient() as client:
    response = client.chat("What is a decorator in Python?")
    print(response.content)
    print(f"Model: {response.model_name}")
    print(f"Tokens: {response.token_usage.total_tokens}")
```

### Multi-turn Conversation

```python
from copilotlibrary import CopilotClient

with CopilotClient() as client:
    # Create a conversation with a custom system prompt
    conv = client.create_conversation("You are a Python expert.")
    
    # First message
    r1 = client.send_message(conv, "What are list comprehensions?")
    print(r1.content)
    
    # Follow-up (context is maintained)
    r2 = client.send_message(conv, "Show me an example")
    print(r2.content)
    
    # View conversation history
    for msg in conv.messages:
        print(f"{msg.role}: {msg.content[:50]}...")
```

### Check Authentication

```python
from copilotlibrary import CopilotClient

with CopilotClient() as client:
    status = client.check_status()
    print(f"Authenticated: {status.is_authenticated}")
    print(f"User: {status.user}")
```

### List Models

```python
from copilotlibrary import CopilotClient

with CopilotClient() as client:
    for model in client.get_chat_models():
        print(f"{model.id}: {model.name}")
```

## CLI Usage

```bash
# One-shot prompt
copilotlibrary "Explain quicksort"

# Check auth status
copilotlibrary status

# List models
copilotlibrary models

# Interactive mode
copilotlibrary
```

### Interactive Mode Commands

```
/help     - Show help
/new      - Start new conversation
/history  - Show history
/system   - Set system prompt
/exit     - Exit
```

## Configuration

### Binary Path

The library auto-detects the copilot-language-server binary from your JetBrains installation. To override:

```bash
# Environment variable
export COPILOT_AGENT_PATH=/path/to/copilot-language-server

# Or pass directly
client = CopilotClient(agent_path="/path/to/copilot-language-server")
```

### Default Locations

| OS | Path |
|----|------|
| macOS | `~/Library/Application Support/JetBrains/IntelliJIdea*/plugins/github-copilot-intellij/copilot-agent/native/darwin-arm64/copilot-language-server` |
| Windows | `%APPDATA%\JetBrains\IntelliJIdea*\plugins\github-copilot-intellij\copilot-agent\native\win32-x64\copilot-language-server.exe` |
| Linux | `~/.config/JetBrains/IntelliJIdea*/plugins/github-copilot-intellij/copilot-agent/native/linux-x64/copilot-language-server` |

## API Documentation

See [API.md](API.md) for complete API documentation including:
- All methods and parameters
- Data models (ChatResponse, Conversation, Model, etc.)
- Examples and error handling

## Run Tests

```bash
# Unit tests
python -m unittest discover -s tests

# Manual integration test (requires binary)
python tests/manual_test.py
python tests/manual_test.py --test-all
```

## Architecture

```
CLI (cli.py)  →  CopilotClient (client.py)  →  copilot-language-server (subprocess, stdio)
```

The client communicates with the binary using **LSP/JSON-RPC 2.0** over stdin/stdout, the same protocol that VS Code and IntelliJ use.

## License

MIT

# Integrating CopilotLibrary into a Python Program

## Table of Contents

1. [Installation](#installation)
2. [What Happens When the User Is Not Logged In](#what-happens-when-the-user-is-not-logged-in)
3. [Authentication Flow](#authentication-flow)
4. [Basic Integration Pattern](#basic-integration-pattern)
5. [Error Handling Reference](#error-handling-reference)
6. [Common Integration Recipes](#common-integration-recipes)
7. [Full Working Example](#full-working-example)

---

## Installation

```bash
# From the repo root (editable mode for development)
pip install -e /path/to/CopilotLibrary

# Or install from the directory directly
pip install /path/to/CopilotLibrary
```

No runtime dependencies — the package uses the stdlib only.

**Pre-requisite:** The user must have a JetBrains IDE (IntelliJ IDEA, PyCharm, etc.)
with the GitHub Copilot plugin installed. The library talks to the **local
`copilot-language-server` binary** the plugin ships; no separate API token is needed.

---

## What Happens When the User Is Not Logged In

### Behaviour at each stage

| Action | Logged in | Not logged in |
|--------|-----------|---------------|
| `CopilotClient()` | Object created; no subprocess yet | Same |
| `client.start()` | Spawns the binary, LSP handshake succeeds | Same — handshake is auth-independent |
| `client.check_status()` | `StatusResult(status=OK, is_authenticated=True)` | `StatusResult(status=NOT_SIGNED_IN, is_authenticated=False)` — **no exception** |
| `client.chat(…)` | Returns `ChatResponse` | **Raises `RuntimeError`** (server sends a JSON-RPC error) |
| `client.send_message(…)` | Returns `ChatResponse` | **Raises `RuntimeError`** |
| `client.sign_in_initiate()` | Works (re-auth or no-op) | Works — safe to call, returns URL + user code |
| `client.sign_in_confirm(code)` | Returns new `StatusResult` | Returns `StatusResult` after device flow completes |

### Key rule

**Always call `check_status()` before making API calls**, or wrap calls in a
`try/except RuntimeError` block. See [Error Handling Reference](#error-handling-reference).

---

## Authentication Flow

The library uses GitHub's **device-flow** OAuth. This is a two-step process that
lets your program display a short code to the user without needing a browser
redirect back to your app.

```
Your program          GitHub website         copilot-language-server
─────────────         ──────────────         ───────────────────────
sign_in_initiate() ──> returns URL + code
display URL & code ──> user opens URL,
                        enters code on
                        github.com
sign_in_confirm(code) ────────────────────────────────────────────>
                                               exchanges code for token
                   <── StatusResult(OK, user="alice")
```

### Code example

```python
import webbrowser
from copilotlibrary import CopilotClient
from copilotlibrary.models import AuthStatus

def ensure_authenticated(client: CopilotClient) -> bool:
    """Return True if already authenticated, or guide the user through sign-in."""
    status = client.check_status()
    if status.is_authenticated:
        return True

    # Start device flow
    info = client.sign_in_initiate()

    print("──────────────────────────────────────────────")
    print("GitHub Copilot sign-in required.")
    print(f"  1. Open:  {info.verification_uri}")
    print(f"  2. Enter: {info.user_code}")
    print("──────────────────────────────────────────────")

    # Optionally open the browser automatically
    webbrowser.open(info.verification_uri)

    input("Press Enter once you have authorised in the browser... ")

    result = client.sign_in_confirm(info.user_code)
    if result.is_authenticated:
        print(f"Signed in as {result.user}")
        return True

    print(f"Sign-in failed (status={result.status.value})")
    return False


with CopilotClient() as client:
    if ensure_authenticated(client):
        response = client.chat("Hello!")
        print(response.content)
```

---

## Basic Integration Pattern

The recommended pattern for any program that embeds CopilotLibrary:

```python
from copilotlibrary import CopilotClient
from copilotlibrary.models import AuthStatus

class MyCopilotIntegration:
    def __init__(self):
        # agent_path is optional — auto-detected from JetBrains plugin
        self._client = CopilotClient()

    def __enter__(self):
        self._client.start()
        return self

    def __exit__(self, *_):
        self._client.stop()

    def ensure_auth(self) -> bool:
        """Check auth and prompt sign-in if needed. Returns True if ready."""
        status = self._client.check_status()
        if status.is_authenticated:
            return True
        # ... call sign_in_initiate / sign_in_confirm here
        return False

    def ask(self, question: str) -> str:
        """Ask Copilot something. Raises RuntimeError if not authenticated."""
        status = self._client.check_status()
        if not status.is_authenticated:
            raise RuntimeError(
                "Not signed in to GitHub Copilot. Call ensure_auth() first."
            )
        return self._client.chat(question).content


with MyCopilotIntegration() as copilot:
    if copilot.ensure_auth():
        print(copilot.ask("What is a monad?"))
```

---

## Error Handling Reference

### `FileNotFoundError`

**When:** `client.start()` (or the first API call, which calls `start()` lazily)  
**Why:** The `copilot-language-server` binary could not be found.  
**Fix:** Install the GitHub Copilot plugin in IntelliJ/JetBrains, or set
`COPILOT_AGENT_PATH` to the binary's absolute path.

```python
import os
from copilotlibrary import CopilotClient

try:
    client = CopilotClient()
    client.start()
except FileNotFoundError as e:
    print(f"Binary not found: {e}")
    print("Tip: set COPILOT_AGENT_PATH=/path/to/copilot-language-server")
```

You can also pass the path directly:

```python
client = CopilotClient(agent_path="/path/to/copilot-language-server")
```

Or via environment variable (useful in CI / Docker):

```bash
export COPILOT_AGENT_PATH=/path/to/copilot-language-server
```

---

### `RuntimeError` — not authenticated

**When:** `client.chat()`, `client.send_message()`, or any method that calls the
Copilot API when the user is not signed in.  
**Message:** `"JSON-RPC error <code> from 'conversation/create': <message>"`

```python
from copilotlibrary import CopilotClient
from copilotlibrary.models import AuthStatus

with CopilotClient() as client:
    status = client.check_status()
    if not status.is_authenticated:
        # Handle gracefully instead of letting RuntimeError propagate
        print("Please sign in to GitHub Copilot first.")
    else:
        try:
            response = client.chat("Hello!")
            print(response.content)
        except RuntimeError as e:
            # Session may have expired mid-run
            print(f"Copilot API error: {e}")
```

---

### `RuntimeError` — server-side error

**When:** The server rejects any request (quota exceeded, unsupported model, etc.)  
**Pattern:**

```python
try:
    response = client.chat("Generate 10,000 words", model="gpt-4o")
except RuntimeError as e:
    print(f"Server error: {e}")
```

---

### Checking auth status without raising

`check_status()` **never raises**. It is always safe to call and returns a
`StatusResult` with `.is_authenticated` bool and `.status` enum:

```python
from copilotlibrary.models import AuthStatus

status = client.check_status()

match status.status:
    case AuthStatus.OK:
        print(f"Ready — signed in as {status.user}")
    case AuthStatus.ALREADY_SIGNED_IN:
        print(f"Already signed in as {status.user}")
    case AuthStatus.NOT_SIGNED_IN:
        print("Not signed in — starting device flow…")
    case AuthStatus.UNKNOWN:
        print("Unknown status — the server may be starting up")
```

---

## Common Integration Recipes

### 1. One-shot question answering

```python
from copilotlibrary import CopilotClient

def ask_copilot(question: str, *, system: str | None = None) -> str:
    """Ask a single question and return the answer as a string."""
    with CopilotClient() as client:
        status = client.check_status()
        if not status.is_authenticated:
            raise RuntimeError("Not signed in to GitHub Copilot.")
        return client.chat(question, system=system).content


# Usage
answer = ask_copilot("Explain async/await in Python in one paragraph.")
print(answer)
```

---

### 2. Multi-turn chatbot

```python
from copilotlibrary import CopilotClient

def run_chatbot(system_prompt: str = "You are a helpful assistant.") -> None:
    with CopilotClient() as client:
        if not client.check_status().is_authenticated:
            print("Error: not signed in to GitHub Copilot.")
            return

        conv = client.create_conversation(system_prompt)
        print("Chat started. Type 'quit' to exit.\n")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"quit", "exit"}:
                break
            try:
                resp = client.send_message(conv, user_input)
                print(f"Copilot: {resp.content}\n")
            except RuntimeError as e:
                print(f"Error: {e}")
                break


run_chatbot("You are a Python tutor. Give short, practical answers.")
```

---

### 3. Streaming with a progress callback

```python
from copilotlibrary import CopilotClient

chunks_received = []

def on_chunk(notification: dict) -> None:
    value = notification.get("params", {}).get("value", {})
    if text := value.get("reply"):
        chunks_received.append(text)
        print(text, end="", flush=True)

with CopilotClient(on_progress=on_chunk) as client:
    if client.check_status().is_authenticated:
        client.chat("Write a haiku about Python.")
        print()  # final newline
```

---

### 4. Choosing a model

```python
from copilotlibrary import CopilotClient

with CopilotClient() as client:
    # List detailed model info (scopes, premium flag, etc.)
    models = client.get_copilot_models()
    for m in models:
        flag = "⭐" if m.is_premium else "  "
        print(f"{flag} {m.id}: {m.name}")

    # Use a specific model
    response = client.chat(
        "Review this code for bugs",
        model="claude-sonnet-4.6",
    )
    print(response.content)
```

---

### 5. Checking auth silently at startup (recommended for long-running programs)

```python
import sys
from copilotlibrary import CopilotClient
from copilotlibrary.models import AuthStatus

def check_copilot_ready(agent_path: str | None = None) -> CopilotClient | None:
    """
    Start the client and verify auth.
    Returns a started CopilotClient on success, or None if not ready.
    The caller is responsible for calling client.stop() when done.
    """
    client = CopilotClient(agent_path=agent_path)
    try:
        client.start()
    except FileNotFoundError as e:
        print(f"[copilot] Binary not found: {e}", file=sys.stderr)
        return None

    status = client.check_status()
    if not status.is_authenticated:
        print(
            "[copilot] Not signed in. Open IntelliJ and sign in to GitHub Copilot.",
            file=sys.stderr,
        )
        client.stop()
        return None

    return client


# In your app's startup:
copilot = check_copilot_ready()
if copilot is None:
    print("Copilot unavailable — running in offline mode.")
else:
    try:
        response = copilot.chat("Hello!")
        print(response.content)
    finally:
        copilot.stop()
```

---

### 6. Using from a web server / long-lived process

Keep a single `CopilotClient` alive for the process lifetime rather than
re-spawning the binary per request:

```python
from contextlib import asynccontextmanager
from copilotlibrary import CopilotClient

# --- FastAPI example ---
# pip install fastapi uvicorn

from fastapi import FastAPI, HTTPException

_copilot: CopilotClient | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _copilot
    _copilot = CopilotClient()
    _copilot.start()
    if not _copilot.check_status().is_authenticated:
        raise RuntimeError("Copilot not authenticated — cannot start server.")
    yield
    _copilot.stop()

app = FastAPI(lifespan=lifespan)

@app.post("/ask")
def ask(prompt: str) -> dict:
    if _copilot is None:
        raise HTTPException(503, "Copilot not ready")
    try:
        resp = _copilot.chat(prompt)
        return {"answer": resp.content, "model": resp.model_name}
    except RuntimeError as e:
        raise HTTPException(502, str(e))
```

> **Thread safety:** `CopilotClient` is **not thread-safe** — it uses a single
> subprocess with a shared read/write cursor. Protect concurrent calls with a
> `threading.Lock` or route all calls through a single thread/event loop.

---

## Full Working Example

Save this as `my_app.py` and run with `python my_app.py`:

```python
"""
Minimal standalone integration — covers binary detection, auth check,
device-flow sign-in, and a multi-turn chat loop.
"""
import sys
import webbrowser
from copilotlibrary import CopilotClient
from copilotlibrary.models import AuthStatus


def ensure_signed_in(client: CopilotClient) -> bool:
    """Return True when the client is authenticated, prompting sign-in if needed."""
    status = client.check_status()
    if status.is_authenticated:
        print(f"Signed in as {status.user}")
        return True

    print("GitHub Copilot is not signed in.")
    answer = input("Start device-flow sign-in? [y/N] ").strip().lower()
    if answer != "y":
        return False

    info = client.sign_in_initiate()
    print(f"\n  Open: {info.verification_uri}")
    print(f"  Code: {info.user_code}\n")
    webbrowser.open(info.verification_uri)

    input("Press Enter after completing authorisation in the browser...")

    result = client.sign_in_confirm(info.user_code)
    if result.is_authenticated:
        print(f"Signed in as {result.user}\n")
        return True

    print(f"Sign-in failed: {result.status.value}")
    return False


def main() -> int:
    try:
        client = CopilotClient()
        client.start()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(
            "Install the GitHub Copilot plugin in IntelliJ, or set "
            "COPILOT_AGENT_PATH=/path/to/copilot-language-server",
            file=sys.stderr,
        )
        return 1

    try:
        if not ensure_signed_in(client):
            print("Cannot continue without authentication.")
            return 1

        conv = client.create_conversation("You are a helpful assistant.")
        print("Chat started. Type 'quit' to exit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return 0

            if user_input.lower() in {"quit", "exit"}:
                return 0

            if not user_input:
                continue

            try:
                resp = client.send_message(conv, user_input)
                print(f"Copilot: {resp.content}\n")
            except RuntimeError as e:
                print(f"Error from Copilot: {e}")
                # Check if it's an auth error and offer re-auth
                status = client.check_status()
                if not status.is_authenticated:
                    print("Session expired. Please re-authenticate.")
                    if not ensure_signed_in(client):
                        return 1
    finally:
        client.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## Summary of Exceptions to Handle

| Exception | From | Cause | Action |
|-----------|------|-------|--------|
| `FileNotFoundError` | `start()` | Binary not found | Check IDE install / set `COPILOT_AGENT_PATH` |
| `RuntimeError` | Any API method | Not signed in, quota, unsupported model, network error | Call `check_status()` first; re-auth if needed |
| `AssertionError` | Internal | `start()` was never called and `_ensure_started()` was not triggered | Should not happen in normal use; always use context manager or call `start()` |


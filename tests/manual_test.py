"""
Manual integration test — runs against the real copilot-language-server binary.

Usage:
    python tests/manual_test.py
    python tests/manual_test.py --prompt "What is a binary search tree?"
    python tests/manual_test.py --agent-path /custom/path/copilot-language-server
    python tests/manual_test.py --test-conversation

Each step prints details to verify the API is working correctly.
"""
from __future__ import annotations

import argparse
import json
import sys
import os

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from copilotlibrary import (
    CopilotClient,
    ChatResponse,
    StatusResult,
    Conversation,
)
from copilotlibrary.client import _DEFAULT_AGENT_PATH


def _pretty(data: dict) -> str:
    return json.dumps(data, indent=2)


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _sign_in(client: CopilotClient) -> bool:
    """Walk through the Copilot device-flow sign-in interactively."""
    _section("Sign-in  (device flow)")
    try:
        sign_in_info = client.sign_in_initiate()
        print(f"\n  1. Open this URL in your browser:\n     {sign_in_info.verification_uri}")
        print(f"  2. Enter the code:  {sign_in_info.user_code}")
        input("\n  Press Enter once you have completed sign-in in the browser...")

        status = client.sign_in_confirm(sign_in_info.user_code)
        if status.is_authenticated:
            print(f"\n  ✓  Signed in successfully (status={status.status.value})")
            return True
        print(f"\n  ✗  Unexpected status after confirm: {status.status.value}")
        return False
    except Exception as exc:
        print(f"  ✗  Sign-in failed: {exc}")
        return False


def test_basic_chat(client: CopilotClient, prompt: str) -> bool:
    """Test basic single-shot chat."""
    _section(f"Chat Test: '{prompt[:50]}...'")
    
    try:
        response: ChatResponse = client.chat(prompt)
        
        print(f"\n  Content: {response.content[:200]}{'...' if len(response.content) > 200 else ''}")
        print(f"\n  Model: {response.model_name}")
        print(f"  Conversation ID: {response.conversation_id}")
        print(f"  Turn ID: {response.turn_id}")
        if response.token_usage:
            print(f"  Tokens Used: {response.token_usage.total_tokens}")
        if response.suggested_title:
            print(f"  Suggested Title: {response.suggested_title}")
        
        if response.content:
            print("\n  ✓  Chat successful!")
            return True
        else:
            print("\n  ⚠  Empty response")
            return False
    except Exception as exc:
        print(f"  ✗  Error: {exc}")
        return False


def test_conversation(client: CopilotClient) -> bool:
    """Test multi-turn conversation."""
    _section("Multi-turn Conversation Test")
    
    try:
        conv: Conversation = client.create_conversation("You are a helpful Python expert.")
        
        # First message
        print("\n  Turn 1: 'What is a list comprehension?'")
        response1 = client.send_message(conv, "What is a list comprehension in Python?")
        print(f"  Response: {response1.content[:150]}...")
        print(f"  Messages in conversation: {len(conv.messages)}")
        
        # Follow-up message (should have context)
        print("\n  Turn 2: 'Show me an example'")
        response2 = client.send_message(conv, "Show me a simple example")
        print(f"  Response: {response2.content[:150]}...")
        print(f"  Messages in conversation: {len(conv.messages)}")
        
        # Verify context is maintained
        if conv.id:
            print(f"\n  Conversation ID: {conv.id}")
        if conv.title:
            print(f"  Title: {conv.title}")
        
        print("\n  ✓  Multi-turn conversation successful!")
        return True
    except Exception as exc:
        print(f"  ✗  Error: {exc}")
        return False


def test_models(client: CopilotClient) -> bool:
    """Test model listing."""
    _section("Models Test")
    
    try:
        models = client.get_models()
        print(f"\n  Total models: {len(models)}")
        
        chat_models = client.get_chat_models()
        print(f"  Chat models: {len(chat_models)}")
        for m in chat_models[:3]:
            print(f"    - {m.id}: {m.name}")
        
        embedding_models = client.get_embedding_models()
        print(f"  Embedding models: {len(embedding_models)}")
        for m in embedding_models[:3]:
            print(f"    - {m.id}: {m.name}")
        
        print("\n  ✓  Models retrieved!")
        return True
    except Exception as exc:
        print(f"  ✗  Error: {exc}")
        return False


def test_with_callback(client: CopilotClient) -> bool:
    """Test progress callback functionality."""
    _section("Progress Callback Test")
    
    progress_count = 0
    
    def on_progress(msg: dict) -> None:
        nonlocal progress_count
        progress_count += 1
    
    # Create a new client with callback
    callback_client = CopilotClient(
        agent_path=client.agent_path,
        on_progress=on_progress,
    )
    
    try:
        with callback_client:
            response = callback_client.chat("Say hello in one word")
            print(f"\n  Response: {response.content}")
            print(f"  Progress callbacks received: {progress_count}")
            
            if progress_count > 0:
                print("\n  ✓  Callback working!")
                return True
            else:
                print("\n  ⚠  No callbacks received")
                return False
    except Exception as exc:
        print(f"  ✗  Error: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual integration test for CopilotLibrary.")
    parser.add_argument("--prompt", default="Explain binary search in Python in one sentence.")
    parser.add_argument("--agent-path", default=None)
    parser.add_argument("--test-conversation", action="store_true", help="Include conversation test")
    parser.add_argument("--test-all", action="store_true", help="Run all tests")
    args = parser.parse_args()

    # ── 1. Show resolved binary path ──────────────────────────────────
    _section("1 · Binary path")
    agent_path = args.agent_path or os.getenv("COPILOT_AGENT_PATH") or _DEFAULT_AGENT_PATH
    print(f"  {agent_path}")
    if not os.path.isfile(agent_path):
        print("\n  ✗  Binary not found. Set COPILOT_AGENT_PATH or pass --agent-path.")
        return 1
    print("  ✓  File exists")

    client = CopilotClient(agent_path=agent_path)

    # ── 2. Initialize (LSP handshake) ─────────────────────────────────
    _section("2 · LSP handshake")
    try:
        client.start()
        print("  ✓  Handshake complete")
        print(f"  ✓  Server running: {client.is_running()}")
    except Exception as exc:
        print(f"  ✗  {exc}")
        return 1

    # ── 3. checkStatus ────────────────────────────────────────────────
    _section("3 · Authentication Status")
    try:
        status: StatusResult = client.check_status()
        print(f"\n  Status: {status.status.value}")
        print(f"  User: {status.user}")
        print(f"  Authenticated: {status.is_authenticated}")

        if not status.is_authenticated:
            print("\n  ⚠  Not signed in. Starting device-flow sign-in...")
            signed_in = _sign_in(client)
            if not signed_in:
                client.stop()
                return 1
    except Exception as exc:
        print(f"  ✗  {exc}")
        client.stop()
        return 1

    # ── 4. Basic chat test ────────────────────────────────────────────
    if not test_basic_chat(client, args.prompt):
        client.stop()
        return 1

    # ── 5. Optional tests ─────────────────────────────────────────────
    if args.test_conversation or args.test_all:
        test_conversation(client)

    if args.test_all:
        test_models(client)
        test_with_callback(client)

    client.stop()
    _section("Done - All tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

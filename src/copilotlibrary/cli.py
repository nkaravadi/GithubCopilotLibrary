from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .client import CopilotClient


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the CLI."""
    if argv is None:
        argv = sys.argv[1:]
    
    # Quick check for subcommands
    subcommands = {"status", "models", "chat"}
    first_arg = argv[0] if argv else None
    
    # If first arg is a known subcommand or starts with -, use argparse
    if first_arg in subcommands or (first_arg and first_arg.startswith("-")):
        return _run_with_parser(list(argv))
    
    # Otherwise, treat all args as a prompt
    if argv:
        return _run_prompt(" ".join(argv))
    
    # No args - interactive mode
    return _run_interactive()


def _run_with_parser(argv: list[str]) -> int:
    """Run with full argparse when subcommands are used."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    client = CopilotClient(agent_path=args.agent_path)

    if args.command == "status":
        return _cmd_status(client)
    elif args.command == "models":
        return _cmd_models(client, args)
    elif args.command == "chat":
        return _cmd_chat(client, args)
    
    # No command - interactive mode
    print("Interactive mode. Type 'exit' or 'quit' to stop. Use /help for commands.")
    return _interactive_mode(client, args.system)


def _run_prompt(prompt: str, agent_path: str | None = None, system: str = "You are a helpful coding assistant.") -> int:
    """Run a single prompt."""
    client = CopilotClient(agent_path=agent_path)
    response = client.chat(prompt, system=system)
    print(response.content)
    return 0


def _run_interactive(agent_path: str | None = None, system: str = "You are a helpful coding assistant.") -> int:
    """Run interactive mode."""
    client = CopilotClient(agent_path=agent_path)
    print("Interactive mode. Type 'exit' or 'quit' to stop. Use /help for commands.")
    return _interactive_mode(client, system)


def _cmd_status(client: CopilotClient) -> int:
    """Show authentication status."""
    status = client.check_status()
    print(f"Status: {status.status.value}")
    if status.user:
        print(f"User: {status.user}")
    print(f"Authenticated: {status.is_authenticated}")
    return 0


def _cmd_models(client: CopilotClient, args: argparse.Namespace) -> int:
    """List available models."""
    models = client.get_models()
    
    if args.filter == "chat":
        models = [m for m in models if m.supports_chat]
        print("Chat Models:")
    elif args.filter == "embeddings":
        models = [m for m in models if m.supports_embeddings]
        print("Embedding Models:")
    else:
        print("All Models:")
    
    for model in models:
        print(f"  - {model.id}: {model.name}")
        if model.vendor:
            print(f"      Vendor: {model.vendor}")
        if model.max_input_tokens:
            print(f"      Max Input Tokens: {model.max_input_tokens}")
    return 0


def _cmd_chat(client: CopilotClient, args: argparse.Namespace) -> int:
    """Single chat message."""
    response = client.chat(args.message, system=args.system)
    print(response.content)
    
    if args.verbose:
        print("\n--- Metadata ---")
        print(f"Model: {response.model_name}")
        print(f"Conversation ID: {response.conversation_id}")
        if response.token_usage:
            print(f"Tokens Used: {response.token_usage.total_tokens}")
    return 0


def _interactive_mode(client: CopilotClient, system: str) -> int:
    """Run interactive chat mode with multi-turn support."""
    conversation = client.create_conversation(system)
    
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() in {"exit", "quit", "/exit", "/quit"}:
            return 0
        
        if user_input == "/help":
            _print_help()
            continue
        
        if user_input == "/new":
            conversation = client.create_conversation(system)
            print("Started new conversation.")
            continue
        
        if user_input == "/history":
            if not conversation.messages:
                print("No messages yet.")
            else:
                for msg in conversation.messages:
                    role = "You" if msg.role == "user" else "Copilot"
                    print(f"{role}: {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
            continue
        
        if user_input.startswith("/system "):
            new_system = user_input[8:].strip()
            conversation = client.create_conversation(new_system)
            print(f"Updated system prompt: {new_system[:50]}...")
            continue
        
        # Send message
        try:
            response = client.send_message(conversation, user_input)
            print(f"copilot> {response.content}")
        except Exception as e:
            print(f"Error: {e}")
    
    return 0


def _print_help() -> None:
    """Print interactive mode help."""
    print("""
Commands:
  /help     - Show this help
  /new      - Start a new conversation
  /history  - Show conversation history
  /system <prompt> - Set new system prompt and start fresh
  /exit, /quit, exit, quit - Exit interactive mode
""")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="copilotlibrary",
        description="Chat with GitHub Copilot via the local language-server binary.",
    )
    parser.add_argument(
        "--agent-path",
        default=None,
        help="Absolute path to the copilot-language-server binary.",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful coding assistant.",
        help="System prompt to set assistant behavior.",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # status command
    subparsers.add_parser("status", help="Check authentication status")
    
    # models command
    models_parser = subparsers.add_parser("models", help="List available models")
    models_parser.add_argument(
        "--filter",
        choices=["chat", "embeddings", "all"],
        default="all",
        help="Filter models by capability",
    )
    
    # chat command
    chat_parser = subparsers.add_parser("chat", help="Send a chat message")
    chat_parser.add_argument("message", help="The message to send")
    chat_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show response metadata",
    )
    
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

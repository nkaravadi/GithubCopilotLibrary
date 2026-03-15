#!/usr/bin/env python3
"""
CopilotLibrary REPL - Interactive tool to explore all Copilot features.

Usage:
    python -m copilotlibrary.repl
    python tests/repl.py

Commands available in REPL - type 'help' for full list.
"""
from __future__ import annotations

import cmd
import json
import sys
import os
import textwrap
from http import client
from typing import Any

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from copilotlibrary import (
    CopilotClient,
    Conversation,
    ChatResponse,
)


class CopilotREPL(cmd.Cmd):
    """Interactive REPL for exploring CopilotLibrary features."""
    
    intro = textwrap.dedent("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║           CopilotLibrary REPL - Interactive Explorer             ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Type 'help' for commands, 'quit' to exit                        ║
    ║  Type any text to chat directly                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    prompt = "\n🤖 copilot> "
    
    def __init__(self):
        super().__init__()
        self.client: CopilotClient | None = None
        self.conversation: Conversation | None = None
        self.current_mode = "Ask"
        self.current_model: str | None = None
        self.current_agent: str | None = None
        self.system_prompt = "You are a helpful coding assistant."
        self.verbose = False
        
    def preloop(self):
        """Initialize client before starting REPL."""
        print("Connecting to Copilot server...")
        try:
            self.client = CopilotClient()
            self.client.start()
            
            # Check status
            status = self.client.check_status()
            if status.is_authenticated:
                print(f"✓ Connected as: {status.user}")
                version = self.client.get_version()
                print(f"✓ Server version: {version.version}")
            else:
                print("⚠ Not authenticated. Use 'signin' command.")
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            sys.exit(1)
    
    def postloop(self):
        """Cleanup on exit."""
        if self.client:
            self.client.stop()
        print("\nGoodbye!")
    
    def default(self, line: str):
        """Handle unknown commands as chat messages."""
        if not line.strip():
            return
        self._chat(line)
    
    def emptyline(self):
        """Don't repeat last command on empty line."""
        pass
    
    # ══════════════════════════════════════════════════════════════════
    # HELP
    # ══════════════════════════════════════════════════════════════════
    
    def do_help(self, arg: str):
        """Show help for commands."""
        if arg:
            super().do_help(arg)
        else:
            print(textwrap.dedent("""
            ═══════════════════════════════════════════════════════════════════
            CHAT COMMANDS
            ═══════════════════════════════════════════════════════════════════
              chat <msg>      Send a chat message (or just type the message)
              new             Start a new conversation
              history         Show conversation history
              clear           Clear conversation history
              embed <text>    Get embedding vector for text (via Copilot REST API)
              localembed <text>  Get embedding vector using a local on-device model
            
            ═══════════════════════════════════════════════════════════════════
            CONFIGURATION
            ═══════════════════════════════════════════════════════════════════
              mode [name]     Get/set mode: Ask, Edit, Agent, Plan
              model [name]    Get/set model (e.g., gpt-4o, claude-sonnet-4.6)
              agent [name]    Get/set agent (e.g., github, project)
              system [text]   Get/set system prompt
              verbose [on|off] Toggle verbose mode
            
            ═══════════════════════════════════════════════════════════════════
            DISCOVERY
            ═══════════════════════════════════════════════════════════════════
              status          Show authentication status
              version         Show server version
              models          List available models (detailed)
              modes           List available conversation modes
              agents          List available agents
              templates       List available templates
              capabilities    Show server capabilities
              preconditions   Check conversation preconditions
            
            ═══════════════════════════════════════════════════════════════════
            RAW API
            ═══════════════════════════════════════════════════════════════════
              raw <method> [json]   Send raw JSON-RPC request
              
            ═══════════════════════════════════════════════════════════════════
            OTHER
            ═══════════════════════════════════════════════════════════════════
              signin
              signout
              quit / exit     Exit the REPL
            """))
    
    # ══════════════════════════════════════════════════════════════════
    # CHAT COMMANDS
    # ══════════════════════════════════════════════════════════════════
    
    def do_chat(self, arg: str):
        """Send a chat message: chat <message>"""
        if not arg.strip():
            print("Usage: chat <message>")
            return
        self._chat(arg)
    
    def _chat(self, message: str):
        """Internal method to send chat."""
        if not self.client:
            print("Not connected.")
            return
        
        try:
            # Create conversation if needed
            if not self.conversation:
                self.conversation = self.client.create_conversation(
                    system_prompt=self.system_prompt,
                    model=self.current_model,
                    mode=self.current_mode,
                    agent=self.current_agent,
                )
            
            # Send message
            response = self.client.send_message(self.conversation, message)
            
            # Print response
            print(f"\n📝 {response.content}")
            
            if self.verbose:
                print(f"\n  ├─ Model: {response.model_name}")
                print(f"  ├─ Conv ID: {response.conversation_id}")
                print(f"  ├─ Turn ID: {response.turn_id}")
                if response.token_usage:
                    print(f"  ├─ Tokens: {response.token_usage.total_tokens}")
                if response.suggested_title:
                    print(f"  └─ Title: {response.suggested_title}")
                    
        except Exception as e:
            print(f"✗ Error: {e}")
    
    def do_new(self, arg: str):
        """Start a new conversation."""
        self.conversation = None
        print("✓ Started new conversation")
        print(f"  Mode: {self.current_mode}")
        if self.current_model:
            print(f"  Model: {self.current_model}")
        if self.current_agent:
            print(f"  Agent: {self.current_agent}")
    
    def do_history(self, arg: str):
        """Show conversation history."""
        if not self.conversation or not self.conversation.messages:
            print("No conversation history.")
            return
        
        print(f"\n📜 Conversation History ({len(self.conversation.messages)} messages)")
        if self.conversation.title:
            print(f"   Title: {self.conversation.title}")
        print("─" * 60)
        
        for i, msg in enumerate(self.conversation.messages, 1):
            role = "👤 You" if msg.role == "user" else "🤖 Copilot"
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            print(f"\n{i}. {role}:")
            print(f"   {content}")
    
    def do_clear(self, arg: str):
        """Clear conversation history."""
        if self.conversation:
            self.conversation.clear()
        print("✓ Conversation cleared")
    
    # ══════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ══════════════════════════════════════════════════════════════════
    
    def do_mode(self, arg: str):
        """Get or set conversation mode: mode [Ask|Edit|Agent|Plan]"""
        if not arg.strip():
            print(f"Current mode: {self.current_mode}")
            print("Available: Ask, Edit, Agent, Plan")
            return
        
        mode = arg.strip().capitalize()
        if mode in ("Ask", "Edit", "Agent", "Plan"):
            self.current_mode = mode
            self.conversation = None  # Reset conversation
            print(f"✓ Mode set to: {mode}")
        else:
            print(f"✗ Unknown mode: {arg}")
            print("Available: Ask, Edit, Agent, Plan")
    
    def do_model(self, arg: str):
        """Get or set model: model [model-id]"""
        if not arg.strip():
            print(f"Current model: {self.current_model or '(default)'}")
            print("\nUse 'models' to see available models")
            return
        
        self.current_model = arg.strip()
        self.conversation = None  # Reset conversation
        print(f"✓ Model set to: {self.current_model}")
    
    def do_agent(self, arg: str):
        """Get or set agent: agent [github|project|...]"""
        if not arg.strip():
            print(f"Current agent: {self.current_agent or '(none)'}")
            print("\nUse 'agents' to see available agents")
            return
        
        if arg.lower() == "none":
            self.current_agent = None
        else:
            self.current_agent = arg.strip()
        self.conversation = None
        print(f"✓ Agent set to: {self.current_agent or '(none)'}")
    
    def do_system(self, arg: str):
        """Get or set system prompt: system [prompt]"""
        if not arg.strip():
            print(f"Current system prompt: {self.system_prompt}")
            return
        
        self.system_prompt = arg.strip()
        self.conversation = None
        print(f"✓ System prompt updated")
    
    def do_verbose(self, arg: str):
        """Toggle verbose mode: verbose [on|off]"""
        if not arg.strip():
            print(f"Verbose: {'on' if self.verbose else 'off'}")
            return
        
        self.verbose = arg.strip().lower() in ("on", "true", "1", "yes")
        print(f"✓ Verbose: {'on' if self.verbose else 'off'}")
    
    # ══════════════════════════════════════════════════════════════════
    # DISCOVERY
    # ══════════════════════════════════════════════════════════════════

    def do_signin(self, arg: str):
        """Sign in to GitHub Copilot using device flow."""
        if not self.client:
            print("Not connected.")
            return

        status = client.check_status()
        if status.is_authenticated:
            print(f"✓ Already signed in as: {status.user}")
            return

        try:
            info =  self.client.sign_in_initiate()
            print(f"\n🔑 Sign in to GitHub Copilot")
            print("─" * 40)
            print(f"1. Visit: {info.verification_uri}")
            print(f"2. Enter code: {info.user_code}")
            print()
            input("Press Enter after completing authorization in the browser...")

            result = self.client.sign_in_confirm(info.user_code)
            if result.is_authenticated:
                print(f"✓ Sign-in successful! Authenticated as: {result.user}")
            else:
                print(f"✗ Sign-in failed. Status code: {result.status_code}")
        except Exception as e:
            print(f"✗ Sign-in error: {e}")

        def do_signout(self, arg: str):
            if not self.client:
                print("Not connected.")
                return

            try:
                result = self.client.sign_out()
                print(f"✓ Signed out successfully.")
            except Exception as e:
                print(f"✗ Sign-out error: {e}")



    def do_status(self, arg: str):
        """Show authentication status."""
        if not self.client:
            print("Not connected.")
            return
        
        status = self.client.check_status()
        print(f"\n📊 Authentication Status")
        print("─" * 40)
        print(f"  Status: {status.status.value}")
        print(f"  User: {status.user or 'N/A'}")
        print(f"  Authenticated: {'✓' if status.is_authenticated else '✗'}")
    
    def do_version(self, arg: str):
        """Show server version."""
        if not self.client:
            print("Not connected.")
            return
        
        version = self.client.get_version()
        print(f"\n📦 Server Version")
        print("─" * 40)
        print(f"  Version: {version.version}")
        print(f"  Build: {version.build_type}")
        print(f"  Runtime: {version.runtime_version}")
    
    def do_models(self, arg: str):
        """List available models with details."""
        if not self.client:
            print("Not connected.")
            return
        
        models = self.client.get_copilot_models()
        if not models:
            print("No models available or API not supported.")
            return
        
        print(f"\n🤖 Available Models ({len(models)})")
        print("─" * 60)
        
        for m in models:
            premium = " 💎" if m.is_premium else ""
            default = " ⭐" if m.is_chat_default else ""
            vision = " 👁" if m.supports_vision else ""
            
            print(f"\n  {m.id}{premium}{default}{vision}")
            print(f"    Name: {m.name}")
            print(f"    Family: {m.family}")
            print(f"    Scopes: {', '.join(m.scopes)}")
            if m.is_premium:
                print(f"    Premium multiplier: {m.premium_multiplier}x")
    
    def do_modes(self, arg: str):
        """List available conversation modes."""
        if not self.client:
            print("Not connected.")
            return
        
        modes = self.client.get_modes()
        print(f"\n🎯 Available Modes ({len(modes)})")
        print("─" * 60)
        
        for m in modes:
            current = " ← current" if m.id == self.current_mode else ""
            print(f"\n  {m.name} ({m.kind}){current}")
            print(f"    {m.description}")
            if m.custom_tools:
                print(f"    Tools: {', '.join(m.custom_tools[:5])}...")
    
    def do_agents(self, arg: str):
        """List available agents."""
        if not self.client:
            print("Not connected.")
            return
        
        agents = self.client.get_agents()
        if not agents:
            print("No agents available.")
            return
        
        print(f"\n🤖 Available Agents ({len(agents)})")
        print("─" * 60)
        
        for a in agents:
            current = " ← current" if a.slug == self.current_agent else ""
            print(f"\n  {a.slug}: {a.name}{current}")
            print(f"    {a.description}")
    
    def do_templates(self, arg: str):
        """List available templates."""
        if not self.client:
            print("Not connected.")
            return
        
        templates = self.client.get_templates()
        if not templates:
            print("No templates available.")
            return
        
        print(f"\n📋 Available Templates ({len(templates)})")
        print("─" * 60)
        
        for t in templates:
            print(f"\n  /{t.id}: {t.short_description}")
            print(f"    {t.description}")
            print(f"    Scopes: {', '.join(t.scopes)}")
    
    def do_capabilities(self, arg: str):
        """Show server capabilities."""
        if not self.client:
            print("Not connected.")
            return
        
        caps = self.client.get_capabilities()
        print(f"\n⚡ Server Capabilities")
        print("─" * 60)
        print(json.dumps(caps, indent=2))
    
    def do_preconditions(self, arg: str):
        """Check conversation preconditions."""
        if not self.client:
            print("Not connected.")
            return
        
        pre = self.client.get_preconditions()
        print(f"\n✅ Preconditions")
        print("─" * 60)
        
        results = pre.get("results", [])
        for r in results:
            status_icon = "✓" if r.get("status") == "ok" else "✗"
            print(f"  {status_icon} {r.get('type')}: {r.get('status')}")
        
        overall = pre.get("status", "unknown")
        print(f"\n  Overall: {overall}")
    
    def do_embed(self, arg: str):
        """Get the embedding vector for text via Copilot REST API: embed <text>"""
        if not arg.strip():
            print("Usage: embed <text>")
            return
        try:
            vector = self.client.get_embedding(arg.strip())
            print(f"\n🔢 Copilot Embedding (REST API)")
            print("─" * 40)
            print(f"  Model:      text-embedding-3-small")
            print(f"  Dimensions: {len(vector)}")
            print(f"  First 8:    {[round(v, 4) for v in vector[:8]]}")
        except Exception as e:
            print(f"✗ {e}")


    # ══════════════════════════════════════════════════════════════════
    # RAW API
    # ══════════════════════════════════════════════════════════════════
    
    def do_raw(self, arg: str):
        """Send raw JSON-RPC request: raw <method> [json-params]"""
        if not self.client:
            print("Not connected.")
            return
        
        parts = arg.strip().split(None, 1)
        if not parts:
            print("Usage: raw <method> [json-params]")
            print("Example: raw getVersion {}")
            print("Example: raw conversation/agents {}")
            return
        
        method = parts[0]
        params = {}
        if len(parts) > 1:
            try:
                params = json.loads(parts[1])
            except json.JSONDecodeError as e:
                print(f"✗ Invalid JSON: {e}")
                return
        
        try:
            result = self.client._send_request(method, params)
            print(f"\n📤 Response:")
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # ══════════════════════════════════════════════════════════════════
    # EXIT
    # ══════════════════════════════════════════════════════════════════
    
    def do_quit(self, arg: str):
        """Exit the REPL."""
        return True
    
    def do_exit(self, arg: str):
        """Exit the REPL."""
        return True
    
    def do_EOF(self, arg: str):
        """Handle Ctrl+D."""
        print()
        return True


def main():
    """Run the REPL."""
    try:
        repl = CopilotREPL()
        repl.cmdloop()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()


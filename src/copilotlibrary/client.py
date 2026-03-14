"""
CopilotClient - A Python client for the GitHub Copilot language-server.

This module provides a clean, well-structured API for interacting with the
copilot-language-server binary that IntelliJ/JetBrains IDEs use.

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
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from typing import Any, Callable

from .models import (
    Agent,
    AuthStatus,
    ChatMessage,
    ChatResponse,
    Conversation,
    ConversationMode,
    CopilotModel,
    FeatureFlags,
    Mode,
    Model,
    ServerVersion,
    SignInInfo,
    StatusResult,
    Template,
    TokenUsage,
)


def _idea_version_key(dir_name: str) -> tuple[int, int]:
    """Parse the numeric version out of a directory name like ``IntelliJIdea2025.3``.

    Returns ``(major, minor)`` so the sort is purely numeric and handles
    double-digit minor versions correctly (``2025.10 > 2025.9``).
    """
    m = re.search(r"(\d+)\.(\d+)", dir_name)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _find_agent_path() -> str:
    """Locate the copilot-language-server binary installed by the IntelliJ Copilot plugin.

    Searches the newest ``IntelliJIdea*`` plugin directory for the binary
    that matches the current OS and CPU architecture.

    Platform layout:
        macOS   – ``~/Library/Application Support/JetBrains/``  arch: ``darwin-arm64`` / ``darwin-x64``
        Windows – ``%APPDATA%\\JetBrains\\``                    arch: ``win32-arm64``   / ``win32-x64``
        Linux   – ``~/.config/JetBrains/``                      arch: ``linux-arm64``  / ``linux-x64``

    Returns an empty string (→ clear FileNotFoundError in ``start()``) when
    nothing is found.
    """
    system = platform.system()          # 'Darwin', 'Windows', 'Linux'
    machine = platform.machine().lower()  # 'arm64', 'aarch64', 'x86_64', 'amd64'
    is_arm = machine in ("arm64", "aarch64")

    if system == "Darwin":
        jetbrains_root = os.path.expanduser("~/Library/Application Support/JetBrains")
        arch = "darwin-arm64" if is_arm else "darwin-x64"
        binary_name = "copilot-language-server"
    elif system == "Windows":
        jetbrains_root = os.path.join(os.environ.get("APPDATA", ""), "JetBrains")
        arch = "win32-arm64" if is_arm else "win32-x64"
        binary_name = "copilot-language-server.exe"
    else:  # Linux / other
        jetbrains_root = os.path.expanduser("~/.config/JetBrains")
        arch = "linux-arm64" if is_arm else "linux-x64"
        binary_name = "copilot-language-server"

    try:
        entries = os.listdir(jetbrains_root)
    except OSError:
        return ""

    # Sort newest-first by numeric (major, minor) version — handles 2025.10 > 2025.9
    # correctly, unlike plain lexicographic order.
    idea_dirs = sorted(
        (e for e in entries if e.startswith("IntelliJIdea")),
        key=_idea_version_key,
        reverse=True,
    )

    for idea_dir in idea_dirs:
        candidate = os.path.join(
            jetbrains_root,
            idea_dir,
            "plugins",
            "github-copilot-intellij",
            "copilot-agent",
            "native",
            arch,
            binary_name,
        )
        if os.path.isfile(candidate):
            return candidate

    return ""


_DEFAULT_AGENT_PATH: str = _find_agent_path()


class CopilotClient:
    """Communicates with the local GitHub Copilot language-server binary via JSON-RPC/LSP.

    Launches the same binary that IntelliJ uses, so no separate token setup is
    required beyond being signed in inside the IDE.

    Agent path lookup order:
        1. ``agent_path`` keyword argument
        2. ``COPILOT_AGENT_PATH`` environment variable
        3. Auto-detected IntelliJ plugin path (see ``_DEFAULT_AGENT_PATH``)

    Attributes:
        agent_path: Path to the copilot-language-server binary.
        
    Example::

        # Using context manager (recommended)
        with CopilotClient() as client:
            status = client.check_status()
            if status.is_authenticated:
                response = client.chat("Hello!")
                print(response.content)

        # Manual lifecycle management
        client = CopilotClient()
        client.start()
        try:
            print(client.chat("Hello!").content)
        finally:
            client.stop()
    """

    def __init__(
        self,
        *,
        agent_path: str | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Initialize the CopilotClient.

        Args:
            agent_path: Path to copilot-language-server binary. Falls back to
                COPILOT_AGENT_PATH env var, then auto-detection.
            on_progress: Optional callback for progress notifications during chat.
                Receives the raw progress notification dict.
        """
        self.agent_path = (
            agent_path
            or os.getenv("COPILOT_AGENT_PATH")
            or _DEFAULT_AGENT_PATH
        )
        self._req_id = 0
        self._process: subprocess.Popen[str] | None = None
        self._on_progress = on_progress
        self._feature_flags: FeatureFlags | None = None
        self._conversations: dict[str, Conversation] = {}
        self._server_capabilities: dict[str, Any] = {}
        self._embeddings: Any | None = None   # lazy CopilotEmbeddings, shared session token

    # ------------------------------------------------------------------ #
    # Process lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Launch the Copilot language-server and perform the LSP handshake.

        Raises:
            FileNotFoundError: If the agent binary doesn't exist.
        """
        if self._process is not None and self._process.poll() is None:
            return  # Already running

        if not os.path.isfile(self.agent_path):
            raise FileNotFoundError(
                f"Copilot agent binary not found: {self.agent_path}\n"
                "Set COPILOT_AGENT_PATH to the correct path, or pass agent_path=..."
            )

        self._process = subprocess.Popen(
            [self.agent_path, "--stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._initialize()

    def stop(self) -> None:
        """Terminate the language-server process."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None

    def is_running(self) -> bool:
        """Check if the language-server process is running."""
        return self._process is not None and self._process.poll() is None

    def __enter__(self) -> CopilotClient:
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    # Public API - Status & Authentication
    # ------------------------------------------------------------------ #

    def check_status(self) -> StatusResult:
        """Return the Copilot auth / subscription status from the agent.

        Returns:
            StatusResult with authentication status and user info.
        """
        self._ensure_started()
        response = self._send_request("checkStatus", {})
        result = response.get("result", {})
        return StatusResult(
            status=AuthStatus.from_str(result.get("status", "")),
            user=result.get("user"),
            raw=response,
        )

    def sign_in_initiate(self) -> SignInInfo:
        """Start the device-flow sign-in process.

        Returns:
            SignInInfo with the verification URL and user code to display.
        """
        self._ensure_started()
        response = self._send_request("signInInitiate", {})
        result = response.get("result", {})
        return SignInInfo(
            verification_uri=result.get("verificationUri", result.get("user_code_url", "")),
            user_code=result.get("userCode", result.get("user_code", "")),
            device_code=result.get("deviceCode"),
            expires_in=result.get("expiresIn"),
            interval=result.get("interval"),
            raw=response,
        )

    def sign_in_confirm(self, user_code: str) -> StatusResult:
        """Complete the device-flow sign-in after user authorization.

        Args:
            user_code: The user code from sign_in_initiate.

        Returns:
            StatusResult indicating whether sign-in succeeded.
        """
        self._ensure_started()
        response = self._send_request("signInConfirm", {"userCode": user_code})
        result = response.get("result", {})
        return StatusResult(
            status=AuthStatus.from_str(result.get("status", "")),
            user=result.get("user"),
            raw=response,
        )

    def sign_out(self) -> StatusResult:
        """Sign out from GitHub Copilot.

        Returns:
            StatusResult indicating the new status.
        """
        self._ensure_started()
        response = self._send_request("signOut", {})
        result = response.get("result", {})
        return StatusResult(
            status=AuthStatus.from_str(result.get("status", "")),
            user=result.get("user"),
            raw=response,
        )

    def ensure_authenticated(
        self,
        *,
        prompt_callback: Callable[[SignInInfo], None] | None = None,
        confirm_callback: Callable[[], str] | None = None,
    ) -> StatusResult:
        """Ensure the user is authenticated, running device-flow sign-in if needed.

        Call this once at startup before making any API calls.  It is safe to
        call on an already-authenticated client — it returns immediately.

        Args:
            prompt_callback: Optional callable invoked with a ``SignInInfo``
                when sign-in is required.  Use it to display the verification
                URL and user code to the end user.  If omitted, nothing is
                printed and ``confirm_callback`` is called right away (useful
                when the caller handles UI itself).
            confirm_callback: Optional callable that is invoked after
                ``prompt_callback`` and must return the user code string once
                the user has authorised in the browser.  If omitted, uses
                ``input()`` to block on a keypress (interactive terminal only).

        Returns:
            The final ``StatusResult``.  Check ``.is_authenticated`` to confirm
            success.

        Raises:
            FileNotFoundError: If the agent binary doesn't exist (from ``start()``).

        Example::

            import webbrowser

            def show_prompt(info):
                webbrowser.open(info.verification_uri)
                print(f"Enter code {info.user_code} at {info.verification_uri}")

            def wait_for_user():
                input("Press Enter after authorising in the browser...")
                return ""   # user_code already submitted; any string is fine

            with CopilotClient() as client:
                status = client.ensure_authenticated(
                    prompt_callback=show_prompt,
                    confirm_callback=wait_for_user,
                )
                if status.is_authenticated:
                    print(f"Ready — signed in as {status.user}")
        """
        self._ensure_started()
        status = self.check_status()
        if status.is_authenticated:
            return status

        info = self.sign_in_initiate()

        if prompt_callback is not None:
            prompt_callback(info)

        if confirm_callback is not None:
            confirm_callback()
        else:
            # Default: block on terminal input
            print(f"\nGitHub Copilot sign-in required.")
            print(f"  Open:  {info.verification_uri}")
            print(f"  Enter: {info.user_code}\n")
            input("Press Enter after completing authorisation in the browser... ")

        return self.sign_in_confirm(info.user_code)

    # ------------------------------------------------------------------ #
    # Public API - Models
    # ------------------------------------------------------------------ #

    def get_models(self) -> list[Model]:
        """Get available models from the Copilot server.

        Note: The copilot-language-server may not expose a direct model listing API.
        This method attempts to retrieve available models and returns known defaults
        if the API is not available.

        Returns:
            List of available Model objects.
        """
        self._ensure_started()
        
        # Try to get models from the server
        try:
            response = self._send_request("getModels", {})
            result = response.get("result", {})
            models_data = result.get("models", [])
            
            if models_data:
                return [
                    Model(
                        id=m.get("id", ""),
                        name=m.get("name", m.get("id", "")),
                        vendor=m.get("vendor"),
                        version=m.get("version"),
                        family=m.get("family"),
                        supports_chat=m.get("supportsChat", True),
                        supports_embeddings=m.get("supportsEmbeddings", False),
                        max_input_tokens=m.get("maxInputTokens"),
                        max_output_tokens=m.get("maxOutputTokens"),
                        raw=m,
                    )
                    for m in models_data
                ]
        except Exception:
            pass
        
        # Server doesn't support getModels - return known defaults
        return self._get_default_models()

    def _get_default_models(self) -> list[Model]:
        """Return known default models when API isn't available."""
        return [
            Model(
                id="gpt-4o",
                name="GPT-4o",
                vendor="OpenAI",
                supports_chat=True,
                supports_embeddings=False,
            ),
            Model(
                id="gpt-4",
                name="GPT-4",
                vendor="OpenAI",
                supports_chat=True,
                supports_embeddings=False,
            ),
            Model(
                id="gpt-3.5-turbo",
                name="GPT-3.5 Turbo",
                vendor="OpenAI",
                supports_chat=True,
                supports_embeddings=False,
            ),
            Model(
                id="text-embedding-ada-002",
                name="Ada 002 Embeddings",
                vendor="OpenAI",
                supports_chat=False,
                supports_embeddings=True,
                max_input_tokens=8191,
            ),
            Model(
                id="text-embedding-3-small",
                name="Embedding 3 Small",
                vendor="OpenAI",
                supports_chat=False,
                supports_embeddings=True,
                max_input_tokens=8191,
            ),
            Model(
                id="text-embedding-3-large",
                name="Embedding 3 Large",
                vendor="OpenAI",
                supports_chat=False,
                supports_embeddings=True,
                max_input_tokens=8191,
            ),
        ]

    def get_chat_models(self) -> list[Model]:
        """Get models that support chat completions.

        Returns:
            List of Model objects that support chat.
        """
        return [m for m in self.get_models() if m.supports_chat]

    def get_embedding_models(self) -> list[Model]:
        """Get models that support embeddings.

        Returns:
            List of Model objects that support embeddings.
        """
        return [m for m in self.get_models() if m.supports_embeddings]

    # ------------------------------------------------------------------ #
    # Public API - Server Discovery
    # ------------------------------------------------------------------ #

    def get_version(self) -> ServerVersion:
        """Get the server version information.

        Returns:
            ServerVersion with version, build type, and runtime info.
        """
        self._ensure_started()
        response = self._send_request("getVersion", {})
        return ServerVersion.from_dict(response)

    def get_modes(self) -> list[Mode]:
        """Get available conversation modes (Ask, Edit, Agent, Plan).

        Returns:
            List of Mode objects.
        """
        self._ensure_started()
        try:
            response = self._send_request("conversation/modes", {})
            result = response.get("result", [])
            return [Mode.from_dict(m) for m in result]
        except RuntimeError:
            # Return defaults if not supported
            return [
                Mode(id="Ask", name="Ask", kind="Ask", description="General purpose chat"),
                Mode(id="Edit", name="Edit", kind="Edit", description="Code editing mode"),
                Mode(id="Agent", name="Agent", kind="Agent", description="Advanced agent mode"),
            ]

    def get_agents(self) -> list[Agent]:
        """Get available agents (github, project, etc.).

        Returns:
            List of Agent objects.
        """
        self._ensure_started()
        try:
            response = self._send_request("conversation/agents", {})
            result = response.get("result", [])
            return [Agent.from_dict(a) for a in result]
        except RuntimeError:
            return []

    def get_templates(self) -> list[Template]:
        """Get available conversation templates (tests, fix, explain, etc.).

        Returns:
            List of Template objects.
        """
        self._ensure_started()
        try:
            response = self._send_request("conversation/templates", {})
            result = response.get("result", [])
            return [Template.from_dict(t) for t in result]
        except RuntimeError:
            return []

    def get_copilot_models(self) -> list[CopilotModel]:
        """Get detailed model information from the Copilot server.

        This returns more detailed information than get_models(), including
        billing info, capabilities, and supported scopes.

        Returns:
            List of CopilotModel objects with detailed info.
        """
        self._ensure_started()
        try:
            response = self._send_request("copilot/models", {})
            result = response.get("result", [])
            return [CopilotModel.from_dict(m) for m in result]
        except RuntimeError:
            return []

    def get_preconditions(self) -> dict[str, Any]:
        """Check conversation preconditions (auth, chat enabled, etc.).

        Returns:
            Dictionary with precondition results.
        """
        self._ensure_started()
        try:
            response = self._send_request("conversation/preconditions", {})
            return response.get("result", {})
        except RuntimeError:
            return {}

    # ------------------------------------------------------------------ #
    # Public API - Chat (Simple)
    # ------------------------------------------------------------------ #

    def chat(
        self,
        prompt: str,
        *,
        system: str | None = "You are a helpful coding assistant.",
        model: str | None = None,
        doc_uri: str = "file:///untitled",
    ) -> ChatResponse:
        """Send a single chat prompt and return the response.

        This is the simplest way to interact with Copilot. For multi-turn
        conversations, use create_conversation() and send_message().

        Args:
            prompt: The user's message.
            system: Optional system prompt to set assistant behavior.
            model: Optional model ID to use (server may ignore if not supported).
            doc_uri: Document URI context for the chat.

        Returns:
            ChatResponse with the assistant's reply and metadata.

        Example::

            response = client.chat("What is Python?")
            print(response.content)
        """
        self._ensure_started()

        req_id = self._next_id()
        token = str(req_id)
        params: dict[str, Any] = {
            "workDoneToken": token,
            "turns": [{"request": prompt, "response": ""}],
            "doc": {"uri": doc_uri, "version": 0, "languageId": "plaintext"},
        }
        if system:
            params["systemPrompt"] = system
        if model:
            params["model"] = model

        self._write_rpc("conversation/create", params, req_id)
        return self._collect_chat_response(token=token, req_id=req_id)

    # ------------------------------------------------------------------ #
    # Public API - Multi-turn Conversations
    # ------------------------------------------------------------------ #

    def create_conversation(
        self,
        system_prompt: str = "You are a helpful coding assistant.",
        model: str | None = None,
        mode: str = "Ask",
        agent: str | None = None,
    ) -> Conversation:
        """Create a new conversation for multi-turn chat.

        Args:
            system_prompt: The system prompt to set assistant behavior.
            model: Optional model ID to use (e.g., "gpt-4o", "claude-sonnet-4.6").
            mode: Conversation mode - "Ask", "Edit", "Agent", or "Plan".
            agent: Optional agent to use (e.g., "github", "project").

        Returns:
            A Conversation object to use with send_message().

        Example::

            # Simple conversation
            conv = client.create_conversation("You are a Python expert.")
            
            # With mode and model
            conv = client.create_conversation(
                mode="Agent",
                model="claude-sonnet-4.6",
            )
            
            response = client.send_message(conv, "What are decorators?")
            print(response.content)
        """
        conv = Conversation(
            system_prompt=system_prompt,
            model=model,
            mode=mode,
            agent=agent,
        )
        return conv

    def send_message(
        self,
        conversation: Conversation,
        message: str,
        *,
        doc_uri: str = "file:///untitled",
    ) -> ChatResponse:
        """Send a message in an existing conversation.

        Args:
            conversation: The Conversation object from create_conversation().
            message: The user's message.
            doc_uri: Document URI context.

        Returns:
            ChatResponse with the assistant's reply.
        """
        self._ensure_started()

        # Add the new user message
        conversation.add_user_message(message)

        req_id = self._next_id()
        token = str(req_id)
        
        # For continuation, only send the new turn with conversationId
        # For first message, send without conversationId
        if conversation.id:
            # Continuation: send just the new message with conversation context
            params: dict[str, Any] = {
                "workDoneToken": token,
                "conversationId": conversation.id,
                "turns": [{"request": message, "response": ""}],
                "doc": {"uri": doc_uri, "version": 0, "languageId": "plaintext"},
            }
        else:
            # First message: send full turns history
            params = {
                "workDoneToken": token,
                "turns": conversation.get_turns(),
                "doc": {"uri": doc_uri, "version": 0, "languageId": "plaintext"},
            }
        
        if conversation.system_prompt:
            params["systemPrompt"] = conversation.system_prompt
        if conversation.model:
            params["model"] = conversation.model
        if conversation.mode:
            params["mode"] = conversation.mode
        if conversation.agent:
            params["agent"] = conversation.agent

        self._write_rpc("conversation/create", params, req_id)
        response = self._collect_chat_response(token=token, req_id=req_id)

        # Update conversation with response
        conversation.add_assistant_message(response.content)
        if response.conversation_id:
            conversation.id = response.conversation_id
        if response.suggested_title:
            conversation.title = response.suggested_title

        # Store conversation for reference
        if conversation.id:
            self._conversations[conversation.id] = conversation

        return response

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Retrieve a conversation by ID.

        Args:
            conversation_id: The conversation ID.

        Returns:
            The Conversation object or None if not found.
        """
        return self._conversations.get(conversation_id)

    def list_conversations(self) -> list[Conversation]:
        """List all active conversations.

        Returns:
            List of Conversation objects.
        """
        return list(self._conversations.values())

    # ------------------------------------------------------------------ #
    # Public API - Embeddings
    # ------------------------------------------------------------------ #
    # Public API - Embeddings
    # ------------------------------------------------------------------ #

    def _get_embeddings_client(self, model: str = "text-embedding-3-small") -> Any:
        """Return a lazily-created, shared CopilotEmbeddings instance.

        The OAuth token is read once from apps.json (same source the language
        server uses) and the session token exchange is cached for ~30 min, so
        repeated embedding calls within the same CopilotClient session pay the
        exchange cost only once.
        """
        from .embeddings import CopilotEmbeddings, find_copilot_token
        if self._embeddings is None or self._embeddings.model != model:
            self._embeddings = CopilotEmbeddings(model=model)
        return self._embeddings

    def get_embedding(
        self,
        text: str,
        *,
        model: str = "text-embedding-3-small",
    ) -> list[float]:
        """Get an embedding vector for *text* via the Copilot REST API.

        The language server does not expose embeddings over JSON-RPC; instead
        this method uses the same OAuth token the language server stores in
        ``~/.config/github-copilot/apps.json`` and makes the REST call directly,
        with the session-token exchange shared across all calls on this client.

        Args:
            text:  Text to embed.
            model: Embedding model — ``"text-embedding-3-small"`` (default,
                   1536 dims) or ``"text-embedding-ada-002"`` (1536 dims).

        Returns:
            Embedding vector as a list of floats.
        """
        return self._get_embeddings_client(model).embed(text).vector

    def get_embeddings_batch(
        self,
        texts: list[str],
        *,
        model: str = "text-embedding-3-small",
    ) -> list[list[float]]:
        """Get embedding vectors for multiple texts in a single REST call.

        Args:
            texts: Texts to embed.
            model: Embedding model.

        Returns:
            List of embedding vectors, one per input text.
        """
        return [
            r.vector
            for r in self._get_embeddings_client(model).embed_batch(texts)
        ]


    # ------------------------------------------------------------------ #
    # Public API - Feature Detection
    # ------------------------------------------------------------------ #

    def get_feature_flags(self) -> FeatureFlags:
        """Get feature flags from the server.

        Returns:
            FeatureFlags object with enabled features.
        """
        if self._feature_flags:
            return self._feature_flags
        
        # Feature flags are sent via notifications, so we need to trigger a request
        # and capture them. For now, return cached or default.
        return FeatureFlags()

    def get_capabilities(self) -> dict[str, Any]:
        """Get server capabilities from initialization.

        Returns:
            Dictionary of server capabilities.
        """
        return self._server_capabilities

    # ------------------------------------------------------------------ #
    # JSON-RPC / LSP helpers
    # ------------------------------------------------------------------ #

    def _ensure_started(self) -> None:
        """Ensure the client is started."""
        if not self.is_running():
            self.start()

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        req_id = self._next_id()
        self._write_rpc(method, params, req_id)
        response = self._read_response()
        if "error" in response:
            err = response["error"]
            raise RuntimeError(
                f"JSON-RPC error {err.get('code')} from '{method}': {err.get('message')}"
            )
        return response

    def _write_rpc(self, method: str, params: dict[str, Any], req_id: int | None) -> None:
        body: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params}
        if req_id is not None:
            body["id"] = req_id

        msg = json.dumps(body)
        # Content-Length is measured in bytes (LSP spec).
        payload = f"Content-Length: {len(msg.encode())}\r\n\r\n{msg}"

        assert self._process and self._process.stdin
        self._process.stdin.write(payload)
        self._process.stdin.flush()

    def _read_frame(self) -> dict[str, Any]:
        """Read exactly one LSP frame from stdout and return the parsed JSON."""
        assert self._process and self._process.stdout
        stdout = self._process.stdout

        headers: dict[str, str] = {}
        while True:
            line = stdout.readline()
            if line in ("\r\n", "\n", ""):
                break
            if ":" in line:
                key, _, value = line.partition(":")
                headers[key.strip().lower()] = value.strip()

        content_length = int(headers.get("content-length", 0))
        body = stdout.read(content_length)
        
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            # Some responses may have extra data - try to parse just the first object
            # This can happen with malformed server responses
            import re
            # Find first complete JSON object
            match = re.match(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', body)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            # Return an empty notification if we can't parse
            return {"jsonrpc": "2.0", "method": "_parse_error", "params": {"error": str(e)}}

    def _read_response(self) -> dict[str, Any]:
        """Read frames, discarding notifications, until a response with ``id`` arrives."""
        while True:
            msg = self._read_frame()
            if "id" in msg:
                return msg

    def _collect_chat_response(self, *, token: str, req_id: int) -> ChatResponse:
        """Drive the ``conversation/create`` streaming protocol.

        The server sends the actual reply as a sequence of ``$/progress``
        notifications tagged with *token*, then a final one with
        ``value.kind == "end"``.  The JSON-RPC response (matching *req_id*)
        carries only metadata and may arrive before, during, or after the
        progress stream.

        Reads frames until both the response and the ``end`` notification
        have been received.
        """
        chunks: list[str] = []
        got_response = False
        got_end = False
        response_data: dict[str, Any] = {}
        token_usage: TokenUsage | None = None
        conversation_id: str | None = None
        turn_id: str | None = None
        model_name: str | None = None
        suggested_title: str | None = None
        finish_reason: str | None = None

        while not (got_response and got_end):
            msg = self._read_frame()

            # ── JSON-RPC response for our request ──────────────────────
            if msg.get("id") == req_id:
                if "error" in msg:
                    err = msg["error"]
                    raise RuntimeError(
                        f"JSON-RPC error {err.get('code')} from "
                        f"'conversation/create': {err.get('message')}"
                    )
                got_response = True
                response_data = msg.get("result", {})
                conversation_id = response_data.get("conversationId")
                turn_id = response_data.get("turnId")
                model_name = response_data.get("modelName")

            # ── Streaming progress notification ─────────────────────────
            elif msg.get("method") == "$/progress":
                params = msg.get("params", {})
                if str(params.get("token")) == token:
                    value = params.get("value", {})
                    
                    # Notify callback if set
                    if self._on_progress:
                        self._on_progress(msg)
                    
                    # Extract text from various possible locations
                    text = self._extract_progress_text(value)
                    if text:
                        chunks.append(text)
                    
                    # Extract token usage
                    if "contextSize" in value:
                        token_usage = TokenUsage.from_context_size(value["contextSize"])
                    
                    # Extract IDs
                    if value.get("conversationId"):
                        conversation_id = value["conversationId"]
                    if value.get("turnId"):
                        turn_id = value["turnId"]
                    
                    # Handle end notification
                    if value.get("kind") == "end":
                        got_end = True
                        suggested_title = value.get("suggestedTitle")

            # ── Feature flags notification ──────────────────────────────
            elif msg.get("method") == "featureFlagsNotification":
                params = msg.get("params", {})
                self._feature_flags = FeatureFlags.from_dict(params)

            # ── Log messages (for finish_reason) ────────────────────────
            elif msg.get("method") == "window/logMessage":
                params = msg.get("params", {})
                log_msg = params.get("message", "")
                if "finish reason:" in log_msg.lower():
                    # Extract finish reason from log message
                    match = re.search(r"finish reason:\s*\[(\w+)\]", log_msg, re.IGNORECASE)
                    if match:
                        finish_reason = match.group(1)

        return ChatResponse(
            content="".join(chunks),
            conversation_id=conversation_id,
            turn_id=turn_id,
            model_name=model_name,
            suggested_title=suggested_title,
            finish_reason=finish_reason,
            token_usage=token_usage,
            raw=response_data,
        )

    def _extract_progress_text(self, value: dict[str, Any]) -> str:
        """Extract reply text from a progress notification value.

        The copilot-language-server sends text in different structures:
        - value.reply (direct reply)
        - value.message (message content)
        - value.editAgentRounds[].reply (agent mode responses)
        """
        # Direct reply field
        if value.get("reply"):
            return str(value["reply"])
        
        # Message field
        if value.get("message"):
            return str(value["message"])
        
        # Agent rounds (most common in recent versions)
        edit_rounds = value.get("editAgentRounds", [])
        if edit_rounds:
            replies = []
            for round_data in edit_rounds:
                if isinstance(round_data, dict) and round_data.get("reply"):
                    replies.append(str(round_data["reply"]))
            if replies:
                return "\n".join(replies)
        
        # Completions array
        completions = value.get("completions", [])
        if completions and isinstance(completions, list):
            texts = []
            for comp in completions:
                if isinstance(comp, dict):
                    text = comp.get("displayText") or comp.get("text") or ""
                    if text:
                        texts.append(text)
            if texts:
                return "\n".join(texts)
        
        return ""

    def _initialize(self) -> None:
        """Run the LSP initialize / initialized handshake."""
        self._write_rpc(
            "initialize",
            {
                "capabilities": {},
                "clientInfo": {"name": "copilotlibrary", "version": "0.2.0"},
                "initializationOptions": {
                    "editorInfo": {"name": "copilotlibrary", "version": "0.2.0"},
                    "editorPluginInfo": {"name": "copilotlibrary", "version": "0.2.0"},
                },
            },
            self._next_id(),
        )
        response = self._read_response()
        self._server_capabilities = response.get("result", {}).get("capabilities", {})

        # Send the 'initialized' notification — no response is expected.
        self._write_rpc("initialized", {}, req_id=None)


# ------------------------------------------------------------------ #
# Module-level helpers (importable for tests)
# ------------------------------------------------------------------ #

def _build_messages(
    *,
    prompt: str,
    system: str | None,
) -> list[dict[str, str]]:
    """Build a list of messages for chat completion.
    
    Args:
        prompt: The user's message.
        system: Optional system prompt.
        
    Returns:
        List of message dictionaries with role and content.
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _extract_reply(response: dict[str, Any]) -> str:
    """Pull the assistant text out of a ``conversation/create`` response.

    This handles various response shapes from the copilot-language-server.
    
    Args:
        response: The JSON-RPC response dictionary.
        
    Returns:
        The extracted reply text, or JSON string of result if format unknown.
    """
    result = response.get("result") or {}

    if isinstance(result, dict):
        # Shape: { "result": { "reply": "..." } }
        if "reply" in result:
            return str(result["reply"])
        # Shape: { "result": { "message": { "content": "..." } } }
        message = result.get("message") or {}
        if isinstance(message, dict) and "content" in message:
            return str(message["content"])
        # Shape: { "result": { "completions": [{ "displayText": "..." }] } }
        completions = result.get("completions") or []
        if completions and isinstance(completions, list):
            return str(completions[0].get("displayText", ""))
        # Shape with editAgentRounds
        edit_rounds = result.get("editAgentRounds") or []
        if edit_rounds and isinstance(edit_rounds, list):
            replies = [r.get("reply", "") for r in edit_rounds if isinstance(r, dict)]
            return "\n".join(filter(None, replies))

    return json.dumps(result)


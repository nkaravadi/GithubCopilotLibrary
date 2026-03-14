import io
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from copilotlibrary.client import (
    CopilotClient,
    _build_messages,
    _extract_reply,
)
from copilotlibrary.embeddings import (
    CopilotEmbeddings,
    EmbeddingResult,
    find_copilot_token,
    _apps_json_path,
    _hosts_json_path,
)
from copilotlibrary.models import (
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


def _lsp_frame(payload: dict) -> str:
    """Encode a dict as an LSP Content-Length frame (text mode)."""
    body = json.dumps(payload)
    return f"Content-Length: {len(body.encode())}\r\n\r\n{body}"


class BuildMessagesTests(unittest.TestCase):
    def test_includes_system_and_prompt(self):
        msgs = _build_messages(prompt="Hello", system="Be concise.")
        self.assertEqual(msgs[0], {"role": "system", "content": "Be concise."})
        self.assertEqual(msgs[1], {"role": "user", "content": "Hello"})

    def test_no_system(self):
        msgs = _build_messages(prompt="Hello", system=None)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")


class ExtractReplyTests(unittest.TestCase):
    def test_reply_key(self):
        self.assertEqual(_extract_reply({"result": {"reply": "hi"}}), "hi")

    def test_message_content_key(self):
        self.assertEqual(
            _extract_reply({"result": {"message": {"content": "hello"}}}), "hello"
        )

    def test_completions_key(self):
        self.assertEqual(
            _extract_reply({"result": {"completions": [{"displayText": "world"}]}}),
            "world",
        )

    def test_edit_agent_rounds_key(self):
        """Test extraction from editAgentRounds format."""
        self.assertEqual(
            _extract_reply({
                "result": {
                    "editAgentRounds": [
                        {"roundId": 1, "reply": "Hello from agent"}
                    ]
                }
            }),
            "Hello from agent",
        )

    def test_fallback_to_json(self):
        result = _extract_reply({"result": {"unknown": 42}})
        self.assertIn("unknown", result)


class ModelTests(unittest.TestCase):
    def test_model_creation(self):
        model = Model(
            id="gpt-4o",
            name="GPT-4o",
            vendor="OpenAI",
            supports_chat=True,
            supports_embeddings=False,
        )
        self.assertEqual(model.id, "gpt-4o")
        self.assertEqual(model.name, "GPT-4o")
        self.assertTrue(model.supports_chat)
        self.assertFalse(model.supports_embeddings)


class StatusResultTests(unittest.TestCase):
    def test_is_authenticated_ok(self):
        status = StatusResult(status=AuthStatus.OK, user="testuser")
        self.assertTrue(status.is_authenticated)

    def test_is_authenticated_already_signed_in(self):
        status = StatusResult(status=AuthStatus.ALREADY_SIGNED_IN, user="testuser")
        self.assertTrue(status.is_authenticated)

    def test_is_not_authenticated(self):
        status = StatusResult(status=AuthStatus.NOT_SIGNED_IN)
        self.assertFalse(status.is_authenticated)


class ConversationTests(unittest.TestCase):
    def test_add_messages(self):
        conv = Conversation()
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi there!")
        
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.messages[0].role, "user")
        self.assertEqual(conv.messages[1].role, "assistant")

    def test_get_turns(self):
        conv = Conversation()
        conv.add_user_message("What is Python?")
        conv.add_assistant_message("A programming language")
        conv.add_user_message("Show me code")
        
        turns = conv.get_turns()
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0]["request"], "What is Python?")
        self.assertEqual(turns[0]["response"], "A programming language")
        self.assertEqual(turns[1]["request"], "Show me code")
        self.assertEqual(turns[1]["response"], "")

    def test_clear(self):
        conv = Conversation(id="test-123")
        conv.add_user_message("Hello")
        conv.clear()
        
        self.assertEqual(len(conv.messages), 0)
        self.assertIsNone(conv.id)


class TokenUsageTests(unittest.TestCase):
    def test_from_context_size(self):
        data = {
            "totalUsedTokens": 100,
            "systemPromptTokens": 20,
            "userMessagesTokens": 30,
            "totalTokenLimit": 1000,
            "utilizationPercentage": 10.0,
        }
        usage = TokenUsage.from_context_size(data)
        
        self.assertEqual(usage.total_tokens, 100)
        self.assertEqual(usage.system_prompt_tokens, 20)
        self.assertEqual(usage.user_messages_tokens, 30)
        self.assertEqual(usage.total_token_limit, 1000)
        self.assertEqual(usage.utilization_percentage, 10.0)


class CopilotClientTests(unittest.TestCase):
    def _make_fake_process(self, *response_dicts):
        """Return a mock Popen whose stdout plays back the given LSP frames."""
        frames = "".join(_lsp_frame(r) for r in response_dicts)
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.stdin = MagicMock()
        fake_proc.stdout = io.StringIO(frames)
        return fake_proc

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_start_runs_initialize_handshake(self, mock_popen, _mock_isfile):
        init_response = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        mock_popen.return_value = self._make_fake_process(init_response)

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        client.start()

        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        self.assertIn("--stdio", args)

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_chat_returns_chat_response(self, mock_popen, _mock_isfile):
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        # conversation/create returns only metadata
        chat_resp = {"jsonrpc": "2.0", "id": 2, "result": {
            "conversationId": "abc", "turnId": "xyz", "modelName": "GPT-4o"
        }}
        # The actual text arrives via $/progress notifications
        progress_report = {"jsonrpc": "2.0", "method": "$/progress", "params": {
            "token": "2",
            "value": {"kind": "report", "reply": "Use a for-loop."},
        }}
        progress_end = {"jsonrpc": "2.0", "method": "$/progress", "params": {
            "token": "2",
            "value": {"kind": "end"},
        }}
        mock_popen.return_value = self._make_fake_process(
            init_resp, chat_resp, progress_report, progress_end
        )

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        response = client.chat("How do I iterate a list?")

        self.assertIsInstance(response, ChatResponse)
        self.assertEqual(response.content, "Use a for-loop.")
        self.assertEqual(response.conversation_id, "abc")
        self.assertEqual(response.model_name, "GPT-4o")

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_chat_extracts_from_edit_agent_rounds(self, mock_popen, _mock_isfile):
        """Test extraction from editAgentRounds format (newer server versions)."""
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        chat_resp = {"jsonrpc": "2.0", "id": 2, "result": {
            "conversationId": "abc", "turnId": "xyz", "modelName": "GPT-4o"
        }}
        progress_report = {"jsonrpc": "2.0", "method": "$/progress", "params": {
            "token": "2",
            "value": {
                "kind": "report",
                "editAgentRounds": [
                    {"roundId": 1, "reply": "Binary search divides the search space in half."}
                ]
            },
        }}
        progress_end = {"jsonrpc": "2.0", "method": "$/progress", "params": {
            "token": "2",
            "value": {"kind": "end"},
        }}
        mock_popen.return_value = self._make_fake_process(
            init_resp, chat_resp, progress_report, progress_end
        )

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        response = client.chat("What is binary search?")

        self.assertEqual(response.content, "Binary search divides the search space in half.")

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_check_status_returns_status_result(self, mock_popen, _mock_isfile):
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        status_resp = {"jsonrpc": "2.0", "id": 2, "result": {
            "status": "OK", "user": "testuser"
        }}
        mock_popen.return_value = self._make_fake_process(init_resp, status_resp)

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        client.start()
        status = client.check_status()

        self.assertIsInstance(status, StatusResult)
        self.assertEqual(status.status, AuthStatus.OK)
        self.assertEqual(status.user, "testuser")
        self.assertTrue(status.is_authenticated)

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_progress_callback_invoked(self, mock_popen, _mock_isfile):
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        chat_resp = {"jsonrpc": "2.0", "id": 2, "result": {
            "conversationId": "abc", "turnId": "xyz", "modelName": "GPT-4o"
        }}
        progress_report = {"jsonrpc": "2.0", "method": "$/progress", "params": {
            "token": "2",
            "value": {"kind": "report", "reply": "Hello"},
        }}
        progress_end = {"jsonrpc": "2.0", "method": "$/progress", "params": {
            "token": "2",
            "value": {"kind": "end"},
        }}
        mock_popen.return_value = self._make_fake_process(
            init_resp, chat_resp, progress_report, progress_end
        )

        callback_messages = []
        def on_progress(msg):
            callback_messages.append(msg)

        client = CopilotClient(
            agent_path="/fake/copilot-language-server",
            on_progress=on_progress,
        )
        client.chat("Hi")

        self.assertEqual(len(callback_messages), 2)  # report + end

    def test_start_raises_when_binary_missing(self):
        client = CopilotClient(agent_path="/nonexistent/binary")
        with self.assertRaises(FileNotFoundError):
            client.start()

    def test_is_running_false_before_start(self):
        client = CopilotClient(agent_path="/fake/binary")
        self.assertFalse(client.is_running())

    def test_get_default_models(self):
        client = CopilotClient(agent_path="/fake/binary")
        models = client._get_default_models()
        
        self.assertGreater(len(models), 0)
        chat_models = [m for m in models if m.supports_chat]
        embed_models = [m for m in models if m.supports_embeddings]
        
        self.assertGreater(len(chat_models), 0)
        self.assertGreater(len(embed_models), 0)


class EmbeddingTokenTests(unittest.TestCase):
    """Unit tests for token-discovery logic (no network calls)."""

    def test_env_var_copilot_token_takes_precedence(self):
        with patch.dict(os.environ, {"COPILOT_TOKEN": "tok_copilot", "GITHUB_TOKEN": "tok_gh"}):
            self.assertEqual(find_copilot_token(), "tok_copilot")

    def test_env_var_github_token_fallback(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("COPILOT_TOKEN", "GITHUB_TOKEN")}
        env["GITHUB_TOKEN"] = "tok_gh"
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(find_copilot_token(), "tok_gh")

    def test_apps_json_copilot_app_preferred(self):
        apps = {
            "github.com:Ov23liXXXX": {"oauth_token": "gho_copilot", "githubAppId": "Ov23liXXXX", "user": "u"},
            "github.com:other":       {"oauth_token": "gho_other",   "githubAppId": "other",      "user": "u"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(apps, f)
            tmp = f.name
        try:
            with patch("copilotlibrary.embeddings._apps_json_path", return_value=tmp), \
                 patch.dict(os.environ, {}, clear=True):
                tok = find_copilot_token()
            self.assertEqual(tok, "gho_copilot")
        finally:
            os.unlink(tmp)

    def test_apps_json_fallback_when_no_copilot_app(self):
        apps = {
            "github.com:other": {"oauth_token": "gho_only", "githubAppId": "other", "user": "u"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(apps, f)
            tmp = f.name
        try:
            with patch("copilotlibrary.embeddings._apps_json_path", return_value=tmp), \
                 patch.dict(os.environ, {}, clear=True):
                tok = find_copilot_token()
            self.assertEqual(tok, "gho_only")
        finally:
            os.unlink(tmp)

    def test_hosts_json_fallback(self):
        hosts = {"github.com": {"oauth_token": "gho_hosts"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(hosts, f)
            tmp = f.name
        try:
            # apps.json does not exist → fall through to hosts.json
            with patch("copilotlibrary.embeddings._apps_json_path", return_value="/nonexistent.json"), \
                 patch("copilotlibrary.embeddings._hosts_json_path", return_value=tmp), \
                 patch.dict(os.environ, {}, clear=True):
                tok = find_copilot_token()
            self.assertEqual(tok, "gho_hosts")
        finally:
            os.unlink(tmp)

    def test_no_token_raises_value_error(self):
        with patch("copilotlibrary.embeddings._apps_json_path", return_value="/nonexistent.json"), \
             patch("copilotlibrary.embeddings._hosts_json_path", return_value="/nonexistent.json"), \
             patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                CopilotEmbeddings()

    def test_embedding_result_dimensions(self):
        result = EmbeddingResult(
            vector=[0.1, 0.2, 0.3, 0.4],
            model="text-embedding-3-small",
            input_text="hello",
        )
        self.assertEqual(result.dimensions, 4)

    def test_list_models(self):
        with patch("copilotlibrary.embeddings.find_copilot_token", return_value="fake"):
            emb = CopilotEmbeddings()
        models = emb.list_models()
        self.assertIn("text-embedding-3-small", models)
        self.assertIn("text-embedding-ada-002", models)


class ChatMessageTests(unittest.TestCase):
    def test_to_dict_basic(self):
        msg = ChatMessage(role="user", content="Hello")
        d = msg.to_dict()
        self.assertEqual(d, {"role": "user", "content": "Hello"})

    def test_to_dict_with_name(self):
        msg = ChatMessage(role="assistant", content="Hi", name="agent-1")
        d = msg.to_dict()
        self.assertEqual(d["name"], "agent-1")

    def test_to_dict_no_name_key_when_absent(self):
        msg = ChatMessage(role="user", content="Hello")
        self.assertNotIn("name", msg.to_dict())


class SignInInfoTests(unittest.TestCase):
    def test_fields(self):
        info = SignInInfo(
            verification_uri="https://github.com/login/device",
            user_code="ABCD-1234",
            device_code="dev_abc",
            expires_in=900,
            interval=5,
        )
        self.assertEqual(info.verification_uri, "https://github.com/login/device")
        self.assertEqual(info.user_code, "ABCD-1234")
        self.assertEqual(info.device_code, "dev_abc")
        self.assertEqual(info.expires_in, 900)
        self.assertEqual(info.interval, 5)

    def test_defaults(self):
        info = SignInInfo(
            verification_uri="https://example.com",
            user_code="CODE",
        )
        self.assertIsNone(info.device_code)
        self.assertIsNone(info.expires_in)
        self.assertIsNone(info.interval)


class FeatureFlagsTests(unittest.TestCase):
    def test_defaults(self):
        flags = FeatureFlags()
        self.assertFalse(flags.chat_enabled)
        self.assertFalse(flags.agent_as_default)

    def test_from_dict(self):
        flags = FeatureFlags.from_dict({"chat": True, "agent_as_default": True})
        self.assertTrue(flags.chat_enabled)
        self.assertTrue(flags.agent_as_default)

    def test_from_dict_partial(self):
        flags = FeatureFlags.from_dict({"chat": True})
        self.assertTrue(flags.chat_enabled)
        self.assertFalse(flags.agent_as_default)

    def test_from_dict_empty(self):
        flags = FeatureFlags.from_dict({})
        self.assertFalse(flags.chat_enabled)


class ConversationModeTests(unittest.TestCase):
    def test_from_str_ask(self):
        self.assertEqual(ConversationMode.from_str("Ask"), ConversationMode.ASK)

    def test_from_str_edit(self):
        self.assertEqual(ConversationMode.from_str("Edit"), ConversationMode.EDIT)

    def test_from_str_agent(self):
        self.assertEqual(ConversationMode.from_str("Agent"), ConversationMode.AGENT)

    def test_from_str_plan(self):
        self.assertEqual(ConversationMode.from_str("Plan"), ConversationMode.PLAN)

    def test_from_str_case_insensitive(self):
        self.assertEqual(ConversationMode.from_str("ask"), ConversationMode.ASK)
        self.assertEqual(ConversationMode.from_str("AGENT"), ConversationMode.AGENT)

    def test_from_str_unknown_defaults_to_ask(self):
        self.assertEqual(ConversationMode.from_str("Unknown"), ConversationMode.ASK)

    def test_enum_values(self):
        self.assertEqual(ConversationMode.ASK.value, "Ask")
        self.assertEqual(ConversationMode.EDIT.value, "Edit")
        self.assertEqual(ConversationMode.AGENT.value, "Agent")
        self.assertEqual(ConversationMode.PLAN.value, "Plan")


class ModeTests(unittest.TestCase):
    def test_from_dict(self):
        data = {
            "id": "Ask",
            "name": "Ask",
            "kind": "Ask",
            "description": "General purpose chat",
            "isBuiltIn": True,
        }
        mode = Mode.from_dict(data)
        self.assertEqual(mode.id, "Ask")
        self.assertEqual(mode.kind, "Ask")
        self.assertTrue(mode.is_builtin)
        self.assertEqual(mode.description, "General purpose chat")

    def test_from_dict_with_uri(self):
        data = {
            "id": "custom",
            "name": "Custom",
            "kind": "Custom",
            "uri": "file:///custom.yaml",
            "isBuiltIn": False,
        }
        mode = Mode.from_dict(data)
        self.assertFalse(mode.is_builtin)
        self.assertEqual(mode.uri, "file:///custom.yaml")

    def test_from_dict_defaults(self):
        mode = Mode.from_dict({})
        self.assertEqual(mode.id, "")
        self.assertEqual(mode.description, "")
        self.assertEqual(mode.custom_tools, [])


class AgentTests(unittest.TestCase):
    def test_from_dict(self):
        data = {
            "slug": "github",
            "name": "GitHub",
            "description": "GitHub agent",
            "avatarUrl": "https://github.com/avatar.png",
        }
        agent = Agent.from_dict(data)
        self.assertEqual(agent.slug, "github")
        self.assertEqual(agent.name, "GitHub")
        self.assertEqual(agent.avatar_url, "https://github.com/avatar.png")

    def test_from_dict_no_avatar(self):
        agent = Agent.from_dict({"slug": "project", "name": "Project"})
        self.assertIsNone(agent.avatar_url)

    def test_from_dict_defaults(self):
        agent = Agent.from_dict({})
        self.assertEqual(agent.slug, "")
        self.assertEqual(agent.description, "")


class TemplateTests(unittest.TestCase):
    def test_from_dict(self):
        data = {
            "id": "tests",
            "description": "Generate Tests",
            "shortDescription": "Tests",
            "scopes": ["chat-panel"],
            "source": "builtin",
        }
        template = Template.from_dict(data)
        self.assertEqual(template.id, "tests")
        self.assertEqual(template.description, "Generate Tests")
        self.assertEqual(template.short_description, "Tests")
        self.assertIn("chat-panel", template.scopes)
        self.assertEqual(template.source, "builtin")

    def test_from_dict_defaults(self):
        template = Template.from_dict({})
        self.assertEqual(template.id, "")
        self.assertEqual(template.scopes, [])
        self.assertEqual(template.source, "builtin")


class CopilotModelTests(unittest.TestCase):
    def test_from_dict(self):
        data = {
            "id": "gpt-4o",
            "modelName": "GPT-4o",
            "modelFamily": "gpt-4",
            "scopes": ["chat-panel", "inline", "agent-panel"],
            "preview": False,
            "isChatDefault": True,
            "billing": {"isPremium": False, "multiplier": 1.0},
            "capabilities": {"supports": {"vision": True}},
        }
        model = CopilotModel.from_dict(data)
        self.assertEqual(model.id, "gpt-4o")
        self.assertEqual(model.name, "GPT-4o")
        self.assertEqual(model.family, "gpt-4")
        self.assertTrue(model.is_chat_default)
        self.assertFalse(model.is_premium)
        self.assertTrue(model.supports_vision)

    def test_scope_properties(self):
        data = {
            "id": "claude",
            "modelName": "Claude",
            "modelFamily": "claude",
            "scopes": ["chat-panel", "edit-panel", "agent-panel"],
        }
        model = CopilotModel.from_dict(data)
        self.assertTrue(model.supports_chat)
        self.assertTrue(model.supports_edit)
        self.assertTrue(model.supports_agent)
        self.assertFalse(model.supports_inline)

    def test_inline_scope(self):
        model = CopilotModel.from_dict({"id": "m", "modelName": "M", "modelFamily": "f",
                                        "scopes": ["inline"]})
        self.assertFalse(model.supports_chat)
        self.assertTrue(model.supports_inline)

    def test_premium_model(self):
        data = {
            "id": "claude-opus",
            "modelName": "Claude Opus",
            "modelFamily": "claude",
            "billing": {"isPremium": True, "multiplier": 2.5},
        }
        model = CopilotModel.from_dict(data)
        self.assertTrue(model.is_premium)
        self.assertEqual(model.premium_multiplier, 2.5)

    def test_from_dict_defaults(self):
        model = CopilotModel.from_dict({"id": "x", "modelName": "X", "modelFamily": "x"})
        self.assertFalse(model.is_premium)
        self.assertFalse(model.supports_vision)
        self.assertEqual(model.premium_multiplier, 1.0)


class ServerVersionTests(unittest.TestCase):
    def test_from_dict_with_result_key(self):
        data = {
            "result": {
                "version": "1.442.0",
                "buildType": "release",
                "runtimeVersion": "node-20",
            }
        }
        sv = ServerVersion.from_dict(data)
        self.assertEqual(sv.version, "1.442.0")
        self.assertEqual(sv.build_type, "release")
        self.assertEqual(sv.runtime_version, "node-20")

    def test_from_dict_flat(self):
        data = {"version": "1.0.0", "buildType": "dev"}
        sv = ServerVersion.from_dict(data)
        self.assertEqual(sv.version, "1.0.0")
        self.assertEqual(sv.build_type, "dev")

    def test_from_dict_empty(self):
        sv = ServerVersion.from_dict({})
        self.assertEqual(sv.version, "")
        self.assertEqual(sv.build_type, "")


class ConversationExtraTests(unittest.TestCase):
    def test_get_turns_multiple(self):
        conv = Conversation()
        conv.add_user_message("Q1")
        conv.add_assistant_message("A1")
        conv.add_user_message("Q2")
        conv.add_assistant_message("A2")

        turns = conv.get_turns()
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0], {"request": "Q1", "response": "A1"})
        self.assertEqual(turns[1], {"request": "Q2", "response": "A2"})

    def test_get_turns_unanswered(self):
        conv = Conversation()
        conv.add_user_message("Q1")
        conv.add_assistant_message("A1")
        conv.add_user_message("Q2")  # no reply yet

        turns = conv.get_turns()
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[1]["response"], "")

    def test_model_and_mode_stored(self):
        conv = Conversation(model="gpt-4o", mode="Agent", agent="github")
        self.assertEqual(conv.model, "gpt-4o")
        self.assertEqual(conv.mode, "Agent")
        self.assertEqual(conv.agent, "github")

    def test_clear_resets_id_and_title(self):
        conv = Conversation(id="conv-1", title="My Chat")
        conv.add_user_message("Hello")
        conv.clear()
        self.assertIsNone(conv.id)
        self.assertIsNone(conv.title)
        self.assertEqual(conv.messages, [])


class CopilotClientMultiTurnTests(unittest.TestCase):
    def _make_fake_process(self, *response_dicts):
        frames = "".join(_lsp_frame(r) for r in response_dicts)
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.stdin = MagicMock()
        fake_proc.stdout = io.StringIO(frames)
        return fake_proc

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_create_conversation_returns_conversation(self, mock_popen, _mock_isfile):
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        mock_popen.return_value = self._make_fake_process(init_resp)

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        client.start()

        conv = client.create_conversation("You are a test assistant.")
        self.assertIsInstance(conv, Conversation)
        self.assertEqual(conv.system_prompt, "You are a test assistant.")
        self.assertIsNone(conv.id)
        self.assertEqual(conv.messages, [])

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_create_conversation_with_model_and_mode(self, mock_popen, _mock_isfile):
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        mock_popen.return_value = self._make_fake_process(init_resp)

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        client.start()

        conv = client.create_conversation(model="gpt-4o", mode="Agent", agent="github")
        self.assertEqual(conv.model, "gpt-4o")
        self.assertEqual(conv.mode, "Agent")
        self.assertEqual(conv.agent, "github")

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_send_message_updates_conversation(self, mock_popen, _mock_isfile):
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        chat_resp = {"jsonrpc": "2.0", "id": 2, "result": {
            "conversationId": "conv-123", "turnId": "turn-1", "modelName": "GPT-4o"
        }}
        progress_report = {"jsonrpc": "2.0", "method": "$/progress", "params": {
            "token": "2",
            "value": {"kind": "report", "reply": "Hello from assistant"},
        }}
        progress_end = {"jsonrpc": "2.0", "method": "$/progress", "params": {
            "token": "2",
            "value": {"kind": "end"},
        }}
        mock_popen.return_value = self._make_fake_process(
            init_resp, chat_resp, progress_report, progress_end
        )

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        conv = client.create_conversation()
        response = client.send_message(conv, "Hello")

        self.assertIsInstance(response, ChatResponse)
        self.assertEqual(response.content, "Hello from assistant")
        self.assertEqual(response.conversation_id, "conv-123")

        # Conversation should be updated
        self.assertEqual(conv.id, "conv-123")
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.messages[0].role, "user")
        self.assertEqual(conv.messages[0].content, "Hello")
        self.assertEqual(conv.messages[1].role, "assistant")
        self.assertEqual(conv.messages[1].content, "Hello from assistant")

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_list_and_get_conversation(self, mock_popen, _mock_isfile):
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        chat_resp = {"jsonrpc": "2.0", "id": 2, "result": {
            "conversationId": "conv-xyz", "turnId": "t1", "modelName": "GPT-4o"
        }}
        progress_report = {"jsonrpc": "2.0", "method": "$/progress", "params": {
            "token": "2", "value": {"kind": "report", "reply": "Pong"},
        }}
        progress_end = {"jsonrpc": "2.0", "method": "$/progress", "params": {
            "token": "2", "value": {"kind": "end"},
        }}
        mock_popen.return_value = self._make_fake_process(
            init_resp, chat_resp, progress_report, progress_end
        )

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        conv = client.create_conversation()
        client.send_message(conv, "Ping")

        self.assertEqual(len(client.list_conversations()), 1)
        retrieved = client.get_conversation("conv-xyz")
        self.assertIs(retrieved, conv)

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_get_conversation_missing_returns_none(self, mock_popen, _mock_isfile):
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        mock_popen.return_value = self._make_fake_process(init_resp)

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        client.start()
        self.assertIsNone(client.get_conversation("nonexistent"))

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_get_models_uses_server_response(self, mock_popen, _mock_isfile):
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        models_resp = {"jsonrpc": "2.0", "id": 2, "result": {
            "models": [
                {"id": "gpt-5", "name": "GPT-5", "vendor": "OpenAI",
                 "supportsChat": True, "supportsEmbeddings": False},
            ]
        }}
        mock_popen.return_value = self._make_fake_process(init_resp, models_resp)

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        client.start()
        models = client.get_models()

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].id, "gpt-5")
        self.assertEqual(models[0].vendor, "OpenAI")
        self.assertTrue(models[0].supports_chat)

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_get_models_falls_back_to_defaults_on_empty(self, mock_popen, _mock_isfile):
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        # Server returns empty list
        models_resp = {"jsonrpc": "2.0", "id": 2, "result": {"models": []}}
        mock_popen.return_value = self._make_fake_process(init_resp, models_resp)

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        client.start()
        models = client.get_models()

        self.assertGreater(len(models), 0)

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_get_chat_models_filters_correctly(self, mock_popen, _mock_isfile):
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        # getModels not supported → triggers fallback to default list
        getmodels_err = {"jsonrpc": "2.0", "id": 2,
                         "error": {"code": -32601, "message": "Method not found"}}
        mock_popen.return_value = self._make_fake_process(init_resp, getmodels_err)

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        client.start()
        chat_models = client.get_chat_models()

        self.assertTrue(all(m.supports_chat for m in chat_models))
        self.assertGreater(len(chat_models), 0)

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_get_embedding_models_filters_correctly(self, mock_popen, _mock_isfile):
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        # getModels not supported → triggers fallback to default list
        getmodels_err = {"jsonrpc": "2.0", "id": 2,
                         "error": {"code": -32601, "message": "Method not found"}}
        mock_popen.return_value = self._make_fake_process(init_resp, getmodels_err)

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        client.start()
        embed_models = client.get_embedding_models()

        self.assertTrue(all(m.supports_embeddings for m in embed_models))
        self.assertGreater(len(embed_models), 0)

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_is_running_true_after_start(self, mock_popen, _mock_isfile):
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        mock_popen.return_value = self._make_fake_process(init_resp)

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        client.start()
        self.assertTrue(client.is_running())

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_is_running_false_after_stop(self, mock_popen, _mock_isfile):
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        mock_popen.return_value = self._make_fake_process(init_resp)

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        client.start()
        client.stop()
        self.assertFalse(client.is_running())

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_context_manager_starts_and_stops(self, mock_popen, _mock_isfile):
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        mock_popen.return_value = self._make_fake_process(init_resp)

        with CopilotClient(agent_path="/fake/copilot-language-server") as client:
            self.assertTrue(client.is_running())
        self.assertFalse(client.is_running())

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_feature_flags_captured_from_notification(self, mock_popen, _mock_isfile):
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        chat_resp = {"jsonrpc": "2.0", "id": 2, "result": {
            "conversationId": "c1", "turnId": "t1", "modelName": "GPT-4o"
        }}
        feature_flags_notif = {
            "jsonrpc": "2.0",
            "method": "featureFlagsNotification",
            "params": {"chat": True, "agent_as_default": False},
        }
        progress_report = {"jsonrpc": "2.0", "method": "$/progress", "params": {
            "token": "2", "value": {"kind": "report", "reply": "Hi"},
        }}
        progress_end = {"jsonrpc": "2.0", "method": "$/progress", "params": {
            "token": "2", "value": {"kind": "end"},
        }}
        mock_popen.return_value = self._make_fake_process(
            init_resp, chat_resp, feature_flags_notif, progress_report, progress_end
        )

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        client.chat("hello")

        flags = client.get_feature_flags()
        self.assertTrue(flags.chat_enabled)

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_suggested_title_extracted(self, mock_popen, _mock_isfile):
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        chat_resp = {"jsonrpc": "2.0", "id": 2, "result": {
            "conversationId": "c1", "turnId": "t1", "modelName": "GPT-4o"
        }}
        progress_report = {"jsonrpc": "2.0", "method": "$/progress", "params": {
            "token": "2", "value": {"kind": "report", "reply": "Binary search is O(log n)."},
        }}
        progress_end = {"jsonrpc": "2.0", "method": "$/progress", "params": {
            "token": "2", "value": {"kind": "end", "suggestedTitle": "Binary Search"},
        }}
        mock_popen.return_value = self._make_fake_process(
            init_resp, chat_resp, progress_report, progress_end
        )

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        response = client.chat("Explain binary search")

        self.assertEqual(response.suggested_title, "Binary Search")

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_token_usage_extracted(self, mock_popen, _mock_isfile):
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        chat_resp = {"jsonrpc": "2.0", "id": 2, "result": {
            "conversationId": "c1", "turnId": "t1", "modelName": "GPT-4o"
        }}
        progress_report = {"jsonrpc": "2.0", "method": "$/progress", "params": {
            "token": "2",
            "value": {
                "kind": "report",
                "reply": "Done",
                "contextSize": {
                    "totalUsedTokens": 200,
                    "systemPromptTokens": 50,
                    "userMessagesTokens": 100,
                    "totalTokenLimit": 4096,
                    "utilizationPercentage": 4.88,
                },
            },
        }}
        progress_end = {"jsonrpc": "2.0", "method": "$/progress", "params": {
            "token": "2", "value": {"kind": "end"},
        }}
        mock_popen.return_value = self._make_fake_process(
            init_resp, chat_resp, progress_report, progress_end
        )

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        response = client.chat("Tell me something")

        self.assertIsNotNone(response.token_usage)
        self.assertEqual(response.token_usage.total_tokens, 200)
        self.assertEqual(response.token_usage.system_prompt_tokens, 50)
        self.assertEqual(response.token_usage.total_token_limit, 4096)

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_sign_in_initiate_returns_sign_in_info(self, mock_popen, _mock_isfile):
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        sign_in_resp = {"jsonrpc": "2.0", "id": 2, "result": {
            "verificationUri": "https://github.com/login/device",
            "userCode": "ABCD-1234",
            "deviceCode": "dev_abc123",
            "expiresIn": 900,
            "interval": 5,
        }}
        mock_popen.return_value = self._make_fake_process(init_resp, sign_in_resp)

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        client.start()
        info = client.sign_in_initiate()

        self.assertIsInstance(info, SignInInfo)
        self.assertEqual(info.verification_uri, "https://github.com/login/device")
        self.assertEqual(info.user_code, "ABCD-1234")
        self.assertEqual(info.device_code, "dev_abc123")
        self.assertEqual(info.expires_in, 900)

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_sign_in_confirm_returns_status(self, mock_popen, _mock_isfile):
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        confirm_resp = {"jsonrpc": "2.0", "id": 2, "result": {
            "status": "OK", "user": "newuser"
        }}
        mock_popen.return_value = self._make_fake_process(init_resp, confirm_resp)

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        client.start()
        status = client.sign_in_confirm("ABCD-1234")

        self.assertIsInstance(status, StatusResult)
        self.assertTrue(status.is_authenticated)
        self.assertEqual(status.user, "newuser")

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_sign_out_returns_status(self, mock_popen, _mock_isfile):
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        sign_out_resp = {"jsonrpc": "2.0", "id": 2, "result": {"status": "NotSignedIn"}}
        mock_popen.return_value = self._make_fake_process(init_resp, sign_out_resp)

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        client.start()
        status = client.sign_out()

        self.assertIsInstance(status, StatusResult)
        self.assertFalse(status.is_authenticated)
        self.assertEqual(status.status, AuthStatus.NOT_SIGNED_IN)


class CLIExtraTests(unittest.TestCase):
    @patch("copilotlibrary.cli.CopilotClient")
    def test_agent_path_forwarded_to_client(self, mock_client_cls):
        from copilotlibrary import cli
        from copilotlibrary.models import ChatResponse

        mock_client = mock_client_cls.return_value
        mock_client.chat.return_value = ChatResponse(content="ok", conversation_id="1")

        cli.main(["--agent-path", "/custom/binary", "chat", "hello"])

        _, kwargs = mock_client_cls.call_args
        self.assertEqual(kwargs.get("agent_path"), "/custom/binary")

    @patch("copilotlibrary.cli.CopilotClient")
    def test_models_filter_chat(self, mock_client_cls):
        from copilotlibrary import cli
        from copilotlibrary.models import Model

        mock_client = mock_client_cls.return_value
        mock_client.get_models.return_value = [
            Model(id="gpt-4o", name="GPT-4o", vendor="OpenAI", supports_chat=True),
            Model(id="ada-002", name="Ada", vendor="OpenAI", supports_embeddings=True,
                  supports_chat=False),
        ]

        exit_code = cli.main(["models", "--filter", "chat"])
        self.assertEqual(exit_code, 0)

    @patch("copilotlibrary.cli.CopilotClient")
    def test_models_filter_embeddings(self, mock_client_cls):
        from copilotlibrary import cli
        from copilotlibrary.models import Model

        mock_client = mock_client_cls.return_value
        mock_client.get_models.return_value = [
            Model(id="ada-002", name="Ada", vendor="OpenAI", supports_embeddings=True,
                  supports_chat=False),
        ]

        exit_code = cli.main(["models", "--filter", "embeddings"])
        self.assertEqual(exit_code, 0)

    @patch("copilotlibrary.cli.CopilotClient")
    def test_chat_verbose_flag(self, mock_client_cls):
        from copilotlibrary import cli
        from copilotlibrary.models import ChatResponse, TokenUsage

        mock_client = mock_client_cls.return_value
        mock_client.chat.return_value = ChatResponse(
            content="Hello!",
            conversation_id="c1",
            model_name="GPT-4o",
            token_usage=TokenUsage(total_tokens=42),
        )

        exit_code = cli.main(["chat", "--verbose", "Hello world"])
        self.assertEqual(exit_code, 0)


class EnsureAuthenticatedTests(unittest.TestCase):
    """Tests for CopilotClient.ensure_authenticated()."""

    def _make_fake_process(self, *response_dicts):
        frames = "".join(_lsp_frame(r) for r in response_dicts)
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.stdin = MagicMock()
        fake_proc.stdout = io.StringIO(frames)
        return fake_proc

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_already_authenticated_returns_immediately(self, mock_popen, _mock_isfile):
        """ensure_authenticated returns right away when already signed in."""
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        # First check_status call (inside ensure_authenticated)
        status_resp = {"jsonrpc": "2.0", "id": 2, "result": {
            "status": "OK", "user": "alice"
        }}
        mock_popen.return_value = self._make_fake_process(init_resp, status_resp)

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        client.start()

        prompt_cb = MagicMock()
        result = client.ensure_authenticated(prompt_callback=prompt_cb)

        self.assertTrue(result.is_authenticated)
        self.assertEqual(result.user, "alice")
        # prompt_callback should NOT have been called
        prompt_cb.assert_not_called()

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_not_signed_in_calls_prompt_and_confirm_callbacks(self, mock_popen, _mock_isfile):
        """When not signed in, prompt_callback is called then sign-in completes."""
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        # check_status → NOT_SIGNED_IN
        status_resp = {"jsonrpc": "2.0", "id": 2, "result": {"status": "NotSignedIn"}}
        # signInInitiate
        initiate_resp = {"jsonrpc": "2.0", "id": 3, "result": {
            "verificationUri": "https://github.com/login/device",
            "userCode": "ABCD-1234",
            "deviceCode": "dev_xyz",
            "expiresIn": 900,
        }}
        # signInConfirm
        confirm_resp = {"jsonrpc": "2.0", "id": 4, "result": {
            "status": "OK", "user": "bob"
        }}
        mock_popen.return_value = self._make_fake_process(
            init_resp, status_resp, initiate_resp, confirm_resp
        )

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        client.start()

        prompted_info = []
        confirmed = []

        def my_prompt(info):
            prompted_info.append(info)

        def my_confirm():
            confirmed.append(True)
            return ""

        result = client.ensure_authenticated(
            prompt_callback=my_prompt,
            confirm_callback=my_confirm,
        )

        self.assertTrue(result.is_authenticated)
        self.assertEqual(result.user, "bob")
        # prompt_callback was called with SignInInfo
        self.assertEqual(len(prompted_info), 1)
        self.assertEqual(prompted_info[0].user_code, "ABCD-1234")
        self.assertEqual(prompted_info[0].verification_uri, "https://github.com/login/device")
        # confirm_callback was called
        self.assertEqual(len(confirmed), 1)

    @patch("os.path.isfile", return_value=True)
    @patch("subprocess.Popen")
    def test_sign_in_fails_returns_not_authenticated(self, mock_popen, _mock_isfile):
        """If sign-in fails (wrong code, timeout), StatusResult.is_authenticated is False."""
        init_resp = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
        status_resp = {"jsonrpc": "2.0", "id": 2, "result": {"status": "NotSignedIn"}}
        initiate_resp = {"jsonrpc": "2.0", "id": 3, "result": {
            "verificationUri": "https://github.com/login/device",
            "userCode": "WXYZ-9999",
        }}
        # Server still returns NotSignedIn after confirm (e.g. code expired)
        confirm_resp = {"jsonrpc": "2.0", "id": 4, "result": {"status": "NotSignedIn"}}
        mock_popen.return_value = self._make_fake_process(
            init_resp, status_resp, initiate_resp, confirm_resp
        )

        client = CopilotClient(agent_path="/fake/copilot-language-server")
        client.start()

        result = client.ensure_authenticated(
            prompt_callback=lambda _: None,
            confirm_callback=lambda: "",
        )

        self.assertFalse(result.is_authenticated)


if __name__ == "__main__":
    unittest.main()


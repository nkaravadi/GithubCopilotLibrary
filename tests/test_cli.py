import unittest
from unittest.mock import patch, MagicMock

from copilotlibrary import cli
from copilotlibrary.models import ChatResponse


class CLITests(unittest.TestCase):
    @patch("copilotlibrary.cli.CopilotClient")
    def test_one_shot_prompt_calls_chat(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_response = ChatResponse(content="done", conversation_id="123")
        mock_client.chat.return_value = mock_response

        exit_code = cli.main(["Hello", "there"]) 

        self.assertEqual(exit_code, 0)
        mock_client.chat.assert_called_once()
        # Check the prompt was joined correctly
        call_args = mock_client.chat.call_args
        self.assertIn("Hello there", call_args[0][0])

    @patch("copilotlibrary.cli.CopilotClient")
    def test_status_command(self, mock_client_cls):
        from copilotlibrary.models import StatusResult, AuthStatus
        
        mock_client = mock_client_cls.return_value
        mock_client.check_status.return_value = StatusResult(
            status=AuthStatus.OK,
            user="testuser",
        )

        exit_code = cli.main(["status"])

        self.assertEqual(exit_code, 0)
        mock_client.check_status.assert_called_once()

    @patch("copilotlibrary.cli.CopilotClient")
    def test_models_command(self, mock_client_cls):
        from copilotlibrary.models import Model
        
        mock_client = mock_client_cls.return_value
        mock_client.get_models.return_value = [
            Model(id="gpt-4o", name="GPT-4o", vendor="OpenAI"),
        ]

        exit_code = cli.main(["models"])

        self.assertEqual(exit_code, 0)
        mock_client.get_models.assert_called_once()

    @patch("copilotlibrary.cli.CopilotClient")
    def test_chat_subcommand(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_response = ChatResponse(content="Hello!", conversation_id="123")
        mock_client.chat.return_value = mock_response

        exit_code = cli.main(["chat", "Hello world"])

        self.assertEqual(exit_code, 0)
        mock_client.chat.assert_called_once()


if __name__ == "__main__":
    unittest.main()


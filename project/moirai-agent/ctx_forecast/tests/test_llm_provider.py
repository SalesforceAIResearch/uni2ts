"""Unit tests for the LLM provider abstraction."""

import json
import os
import sys
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

# Adjust path so imports work from the test directory
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "src"),
)

from ctx_forecast.llm_provider import (
    PROVIDER_PRESETS,
    ChatCompletionsProvider,
    LLMResponse,
    OpenAIResponsesProvider,
    OutputFunctionCall,
    OutputMessage,
    OutputReasoning,
    create_llm_provider,
)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestCreateLLMProvider(unittest.TestCase):
    """Tests for the ``create_llm_provider`` factory."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_default_provider_is_openai(self):
        config = {"llm": {"model_name": "gpt-5.1", "model_params_type": {}}}
        provider = create_llm_provider(config)
        self.assertIsInstance(provider, OpenAIResponsesProvider)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_explicit_openai_provider(self):
        config = {
            "llm": {"provider": "openai", "model_name": "gpt-5.1", "model_params_type": {}}
        }
        provider = create_llm_provider(config)
        self.assertIsInstance(provider, OpenAIResponsesProvider)

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    def test_minimax_provider(self):
        config = {
            "llm": {
                "provider": "minimax",
                "model_name": "MiniMax-M2.7",
                "model_params_type": {},
            }
        }
        provider = create_llm_provider(config)
        self.assertIsInstance(provider, ChatCompletionsProvider)

    @patch.dict(os.environ, {"OPENAI_COMPATIBLE_API_KEY": "test-key"})
    def test_openai_compatible_provider(self):
        config = {
            "llm": {
                "provider": "openai_compatible",
                "model_name": "custom-model",
                "base_url": "https://api.example.com/v1",
                "model_params_type": {},
            }
        }
        provider = create_llm_provider(config)
        self.assertIsInstance(provider, ChatCompletionsProvider)

    def test_unknown_provider_raises(self):
        config = {
            "llm": {
                "provider": "nonexistent_provider",
                "model_name": "x",
                "model_params_type": {},
            }
        }
        with self.assertRaises(ValueError):
            create_llm_provider(config)


# ---------------------------------------------------------------------------
# Message conversion tests
# ---------------------------------------------------------------------------


class TestChatCompletionsMessageConversion(unittest.TestCase):
    """Tests for ``ChatCompletionsProvider._to_chat_messages``."""

    def _make_provider(self):
        """Create a provider instance with mocked client."""
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            config = {
                "llm": {
                    "provider": "minimax",
                    "model_name": "MiniMax-M2.7",
                    "model_params_type": {},
                }
            }
            return ChatCompletionsProvider(config)

    def test_system_instruction(self):
        provider = self._make_provider()
        msgs = provider._to_chat_messages("You are helpful.", [])
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], "You are helpful.")

    def test_simple_user_message(self):
        provider = self._make_provider()
        input_items = [{"role": "user", "content": "Hello"}]
        msgs = provider._to_chat_messages("sys", input_items)
        self.assertEqual(len(msgs), 2)  # system + user
        self.assertEqual(msgs[1]["role"], "user")
        self.assertEqual(msgs[1]["content"], "Hello")

    def test_image_message_conversion(self):
        provider = self._make_provider()
        input_items = [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "data:image/png;base64,abc"},
                    {"type": "input_text", "text": "Describe this"},
                ],
            }
        ]
        msgs = provider._to_chat_messages(None, input_items)
        self.assertEqual(len(msgs), 1)
        content = msgs[0]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertEqual(
            content[0]["image_url"]["url"], "data:image/png;base64,abc"
        )
        self.assertEqual(content[1]["type"], "text")
        self.assertEqual(content[1]["text"], "Describe this")

    def test_function_call_output_conversion(self):
        provider = self._make_provider()
        input_items = [
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "result text",
            }
        ]
        msgs = provider._to_chat_messages(None, input_items)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "tool")
        self.assertEqual(msgs[0]["tool_call_id"], "call_123")
        self.assertEqual(msgs[0]["content"], "result text")

    def test_output_objects_grouped_into_assistant_message(self):
        """Consecutive output objects should merge into a single assistant msg."""
        provider = self._make_provider()
        input_items = [
            OutputMessage(content="I'll call the tool"),
            OutputFunctionCall(
                call_id="c1", name="forecast", arguments='{"x": 1}'
            ),
            OutputFunctionCall(
                call_id="c2", name="sandbox", arguments='{"code": "print(1)"}'
            ),
        ]
        msgs = provider._to_chat_messages(None, input_items)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "assistant")
        self.assertEqual(msgs[0]["content"], "I'll call the tool")
        self.assertEqual(len(msgs[0]["tool_calls"]), 2)
        self.assertEqual(msgs[0]["tool_calls"][0]["id"], "c1")
        self.assertEqual(msgs[0]["tool_calls"][1]["id"], "c2")

    def test_reasoning_output_skipped(self):
        provider = self._make_provider()
        input_items = [
            OutputReasoning(summary="thinking..."),
            OutputMessage(content="Done"),
        ]
        msgs = provider._to_chat_messages(None, input_items)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["content"], "Done")

    def test_multi_round_context(self):
        """Simulate a two-round conversation context."""
        provider = self._make_provider()
        input_items = [
            # Round 1: user + assistant + tool result
            {"role": "user", "content": "step: 1"},
            {"role": "user", "content": "What is 2+2?"},
            OutputMessage(content="Let me calculate"),
            OutputFunctionCall(
                call_id="c1", name="calc", arguments='{"expr": "2+2"}'
            ),
            {"type": "function_call_output", "call_id": "c1", "output": "4"},
            # Round 2: user step marker
            {"role": "user", "content": "step: 2"},
        ]
        msgs = provider._to_chat_messages("You are helpful.", input_items)
        # system, user(step1), user(query), assistant(content+tool), tool, user(step2)
        self.assertEqual(len(msgs), 6)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertEqual(msgs[2]["role"], "user")
        self.assertEqual(msgs[3]["role"], "assistant")
        self.assertEqual(msgs[4]["role"], "tool")
        self.assertEqual(msgs[5]["role"], "user")


# ---------------------------------------------------------------------------
# Tool conversion tests
# ---------------------------------------------------------------------------


class TestChatCompletionsToolConversion(unittest.TestCase):
    def test_tools_converted(self):
        tools = [
            {
                "type": "function",
                "name": "forecast",
                "description": "Run forecast",
                "parameters": {"type": "object", "properties": {"x": {"type": "number"}}},
            }
        ]
        result = ChatCompletionsProvider._to_chat_tools(tools)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "function")
        self.assertEqual(result[0]["function"]["name"], "forecast")
        self.assertEqual(result[0]["function"]["description"], "Run forecast")

    def test_empty_tools(self):
        self.assertEqual(ChatCompletionsProvider._to_chat_tools([]), [])
        self.assertEqual(ChatCompletionsProvider._to_chat_tools(None), [])


# ---------------------------------------------------------------------------
# Response conversion tests
# ---------------------------------------------------------------------------


class TestChatCompletionsResponseConversion(unittest.TestCase):
    def _mock_response(self, content=None, tool_calls=None):
        """Create a mock Chat Completions response."""
        message = MagicMock()
        message.content = content
        message.tool_calls = tool_calls

        choice = MagicMock()
        choice.message = message

        response = MagicMock()
        response.choices = [choice]
        return response

    def test_text_only_response(self):
        resp = self._mock_response(content="The answer is 42")
        result = ChatCompletionsProvider._to_llm_response(resp)
        self.assertIsInstance(result, LLMResponse)
        self.assertEqual(result.output_text, "The answer is 42")
        self.assertEqual(len(result.output), 1)
        self.assertIsInstance(result.output[0], OutputMessage)

    def test_tool_call_response(self):
        tc = MagicMock()
        tc.id = "call_abc"
        tc.function.name = "forecast"
        tc.function.arguments = '{"x": 1}'

        resp = self._mock_response(content=None, tool_calls=[tc])
        result = ChatCompletionsProvider._to_llm_response(resp)
        self.assertEqual(result.output_text, "")
        self.assertEqual(len(result.output), 1)
        self.assertIsInstance(result.output[0], OutputFunctionCall)
        self.assertEqual(result.output[0].call_id, "call_abc")
        self.assertEqual(result.output[0].name, "forecast")

    def test_text_with_tool_calls(self):
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "calc"
        tc.function.arguments = '{"a": 1}'

        resp = self._mock_response(content="Calling calc", tool_calls=[tc])
        result = ChatCompletionsProvider._to_llm_response(resp)
        self.assertEqual(result.output_text, "Calling calc")
        self.assertEqual(len(result.output), 2)  # message + function_call

    def test_think_tag_stripping(self):
        resp = self._mock_response(
            content="<think>Let me think about this...</think>\nThe answer is 42"
        )
        result = ChatCompletionsProvider._to_llm_response(resp)
        self.assertEqual(result.output_text, "The answer is 42")


# ---------------------------------------------------------------------------
# Parameter handling tests
# ---------------------------------------------------------------------------


class TestParameterHandling(unittest.TestCase):
    def _make_provider(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            config = {
                "llm": {
                    "provider": "minimax",
                    "model_name": "MiniMax-M2.7",
                    "model_params_type": {},
                }
            }
            return ChatCompletionsProvider(config)

    @patch("ctx_forecast.llm_provider.ChatCompletionsProvider._to_llm_response")
    def test_reasoning_param_filtered(self, mock_resp):
        provider = self._make_provider()
        mock_resp.return_value = LLMResponse(output=[], output_text="")
        provider.client = MagicMock()
        provider.client.chat.completions.create.return_value = MagicMock()

        provider.create(
            model="MiniMax-M2.7",
            instructions="sys",
            input=[],
            tools=[],
            tool_choice="auto",
            reasoning={"effort": "medium"},
            temperature=0.7,
        )

        call_kwargs = provider.client.chat.completions.create.call_args[1]
        self.assertNotIn("reasoning", call_kwargs)
        self.assertNotIn("reasoning_effort", call_kwargs)

    @patch("ctx_forecast.llm_provider.ChatCompletionsProvider._to_llm_response")
    def test_max_output_tokens_converted(self, mock_resp):
        provider = self._make_provider()
        mock_resp.return_value = LLMResponse(output=[], output_text="")
        provider.client = MagicMock()
        provider.client.chat.completions.create.return_value = MagicMock()

        provider.create(
            model="MiniMax-M2.7",
            instructions="sys",
            input=[],
            tools=[],
            tool_choice="auto",
            max_output_tokens=4096,
        )

        call_kwargs = provider.client.chat.completions.create.call_args[1]
        self.assertNotIn("max_output_tokens", call_kwargs)
        self.assertEqual(call_kwargs["max_tokens"], 4096)

    @patch("ctx_forecast.llm_provider.ChatCompletionsProvider._to_llm_response")
    def test_temperature_clamped(self, mock_resp):
        provider = self._make_provider()
        mock_resp.return_value = LLMResponse(output=[], output_text="")
        provider.client = MagicMock()
        provider.client.chat.completions.create.return_value = MagicMock()

        provider.create(
            model="MiniMax-M2.7",
            instructions="sys",
            input=[],
            tools=[],
            tool_choice="auto",
            temperature=1.5,
        )

        call_kwargs = provider.client.chat.completions.create.call_args[1]
        self.assertLessEqual(call_kwargs["temperature"], 1.0)


# ---------------------------------------------------------------------------
# Provider presets tests
# ---------------------------------------------------------------------------


class TestProviderPresets(unittest.TestCase):
    def test_minimax_preset_exists(self):
        self.assertIn("minimax", PROVIDER_PRESETS)

    def test_minimax_base_url(self):
        self.assertEqual(
            PROVIDER_PRESETS["minimax"]["base_url"],
            "https://api.minimax.io/v1",
        )

    def test_minimax_models(self):
        models = PROVIDER_PRESETS["minimax"]["models"]
        self.assertIn("MiniMax-M2.7", models)
        self.assertIn("MiniMax-M2.5-highspeed", models)

    def test_minimax_env_key(self):
        self.assertEqual(PROVIDER_PRESETS["minimax"]["env_key"], "MINIMAX_API_KEY")


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestResponseDataclasses(unittest.TestCase):
    def test_output_message_attributes(self):
        msg = OutputMessage(content="hello")
        self.assertEqual(msg.type, "message")
        self.assertEqual(msg.content, "hello")

    def test_output_function_call_attributes(self):
        fc = OutputFunctionCall(call_id="c1", name="fn", arguments='{"a":1}')
        self.assertEqual(fc.type, "function_call")
        self.assertEqual(fc.call_id, "c1")
        self.assertEqual(fc.name, "fn")
        self.assertEqual(fc.arguments, '{"a":1}')

    def test_output_reasoning_attributes(self):
        r = OutputReasoning(summary="thinking")
        self.assertEqual(r.type, "reasoning")
        self.assertEqual(r.summary, "thinking")

    def test_llm_response_attributes(self):
        resp = LLMResponse(
            output=[OutputMessage(content="hi")],
            output_text="hi",
        )
        self.assertEqual(len(resp.output), 1)
        self.assertEqual(resp.output_text, "hi")

    def test_llm_response_iteration(self):
        """Verify output items can be iterated and type-checked like the OpenAI SDK."""
        items = [
            OutputReasoning(summary="step 1"),
            OutputMessage(content="result"),
            OutputFunctionCall(call_id="c1", name="fn", arguments="{}"),
        ]
        resp = LLMResponse(output=items, output_text="result")

        reasoning = [r for r in resp.output if r.type == "reasoning"]
        self.assertEqual(len(reasoning), 1)
        self.assertEqual(reasoning[0].summary, "step 1")

        tool_calls = [r for r in resp.output if r.type == "function_call"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "fn")


if __name__ == "__main__":
    unittest.main()

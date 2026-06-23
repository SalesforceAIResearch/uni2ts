"""
Integration tests for MiniMax LLM provider.

These tests require a valid MINIMAX_API_KEY environment variable.
Skip if the key is not set.
"""

import json
import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "src"),
)

from ctx_forecast.llm_provider import (
    ChatCompletionsProvider,
    LLMResponse,
    OutputFunctionCall,
    OutputMessage,
    create_llm_provider,
)

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
SKIP_REASON = "MINIMAX_API_KEY not set"


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMiniMaxBasicCompletion(unittest.TestCase):
    """Test basic text completion with MiniMax."""

    def setUp(self):
        self.config = {
            "llm": {
                "provider": "minimax",
                "model_name": "MiniMax-M2.5-highspeed",
                "model_params_type": {
                    "temperature": 0.1,
                    "max_output_tokens": 256,
                },
            }
        }
        self.provider = create_llm_provider(self.config)

    def test_simple_text_response(self):
        result = self.provider.create(
            model="MiniMax-M2.5-highspeed",
            instructions="You are a helpful assistant. Reply concisely.",
            input=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
            tools=[],
            tool_choice="auto",
            temperature=0.1,
            max_output_tokens=64,
        )
        self.assertIsInstance(result, LLMResponse)
        self.assertIn("4", result.output_text)
        self.assertTrue(len(result.output) > 0)

    def test_response_output_iteration(self):
        """Verify the response output can be iterated and type-checked."""
        result = self.provider.create(
            model="MiniMax-M2.5-highspeed",
            instructions="Reply briefly.",
            input=[{"role": "user", "content": "Say hi"}],
            tools=[],
            tool_choice="auto",
            temperature=0.1,
            max_output_tokens=32,
        )
        messages = [r for r in result.output if r.type == "message"]
        self.assertTrue(len(messages) >= 1)
        self.assertTrue(len(messages[0].content) > 0)


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMiniMaxToolCalling(unittest.TestCase):
    """Test function/tool calling with MiniMax."""

    def setUp(self):
        self.provider = create_llm_provider(
            {
                "llm": {
                    "provider": "minimax",
                    "model_name": "MiniMax-M2.5-highspeed",
                    "model_params_type": {},
                }
            }
        )
        self.tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            }
        ]

    def test_tool_call_triggered(self):
        result = self.provider.create(
            model="MiniMax-M2.5-highspeed",
            instructions="You have access to tools. Use them when appropriate.",
            input=[
                {"role": "user", "content": "What is the weather in Tokyo?"},
            ],
            tools=self.tools,
            tool_choice="auto",
            temperature=0.1,
            max_output_tokens=256,
        )
        self.assertIsInstance(result, LLMResponse)
        tool_calls = [r for r in result.output if r.type == "function_call"]
        self.assertTrue(len(tool_calls) >= 1, "Expected at least one tool call")
        tc = tool_calls[0]
        self.assertEqual(tc.name, "get_weather")
        args = json.loads(tc.arguments)
        self.assertIn("city", args)


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMiniMaxMultiTurn(unittest.TestCase):
    """Test multi-turn conversation with tool results fed back."""

    def setUp(self):
        self.provider = create_llm_provider(
            {
                "llm": {
                    "provider": "minimax",
                    "model_name": "MiniMax-M2.5-highspeed",
                    "model_params_type": {},
                }
            }
        )

    def test_multi_turn_with_tool_result(self):
        tools = [
            {
                "type": "function",
                "name": "calculate",
                "description": "Evaluate a math expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression",
                        }
                    },
                    "required": ["expression"],
                },
            }
        ]

        # Round 1: ask question, expect tool call
        r1 = self.provider.create(
            model="MiniMax-M2.5-highspeed",
            instructions="Use the calculate tool when asked math questions.",
            input=[
                {"role": "user", "content": "What is 123 * 456?"},
            ],
            tools=tools,
            tool_choice="auto",
            temperature=0.1,
            max_output_tokens=256,
        )

        tc_items = [r for r in r1.output if r.type == "function_call"]
        if not tc_items:
            # Model answered directly – still valid
            self.assertIn("56088", r1.output_text)
            return

        # Build context for round 2
        context = [
            {"role": "user", "content": "What is 123 * 456?"},
        ]
        context += r1.output  # add assistant output objects
        context.append(
            {
                "type": "function_call_output",
                "call_id": tc_items[0].call_id,
                "output": "56088",
            }
        )

        # Round 2: feed tool result back
        r2 = self.provider.create(
            model="MiniMax-M2.5-highspeed",
            instructions="Use the calculate tool when asked math questions.",
            input=context,
            tools=tools,
            tool_choice="auto",
            temperature=0.1,
            max_output_tokens=256,
        )
        # Model may format the number with commas (e.g., "56,088")
        normalized = r2.output_text.replace(",", "")
        self.assertIn("56088", normalized)


if __name__ == "__main__":
    unittest.main()

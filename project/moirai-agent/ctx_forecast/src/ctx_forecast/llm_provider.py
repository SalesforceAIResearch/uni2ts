"""
LLM Provider abstraction for Moirai Agent.

Supports OpenAI (native Responses API) and OpenAI-compatible providers
(e.g., MiniMax) via Chat Completions API with automatic format translation.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Lightweight response dataclasses – attribute-compatible with the objects
# returned by OpenAI's Responses API so that existing MCPClient code works
# unchanged.
# ---------------------------------------------------------------------------


@dataclass
class OutputMessage:
    """Text message output item."""

    type: str = "message"
    content: str = ""


@dataclass
class OutputFunctionCall:
    """Function/tool call output item."""

    type: str = "function_call"
    call_id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class OutputReasoning:
    """Reasoning trace output item (only populated by OpenAI reasoning models)."""

    type: str = "reasoning"
    summary: str = ""


@dataclass
class LLMResponse:
    """Unified response envelope – drop-in replacement for ``openai.Response``."""

    output: list = field(default_factory=list)
    output_text: str = ""


# ---------------------------------------------------------------------------
# Provider presets
# ---------------------------------------------------------------------------

PROVIDER_PRESETS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "base_url": None,
    },
    "minimax": {
        "env_key": "MINIMAX_API_KEY",
        "base_url": "https://api.minimax.io/v1",
        "default_model": "MiniMax-M2.7",
        "models": [
            "MiniMax-M2.7",
            "MiniMax-M2.7-highspeed",
            "MiniMax-M2.5",
            "MiniMax-M2.5-highspeed",
        ],
    },
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_llm_provider(config: dict):
    """Instantiate the appropriate LLM provider based on ``config["llm"]["provider"]``."""

    provider_name = config.get("llm", {}).get("provider", "openai")

    if provider_name == "openai":
        return OpenAIResponsesProvider(config)

    if provider_name in PROVIDER_PRESETS or provider_name == "openai_compatible":
        return ChatCompletionsProvider(config)

    raise ValueError(
        f"Unknown provider: {provider_name!r}. "
        f"Supported: {sorted(PROVIDER_PRESETS.keys())}"
    )


# ---------------------------------------------------------------------------
# OpenAI – native Responses API (no translation needed)
# ---------------------------------------------------------------------------


class OpenAIResponsesProvider:
    """Thin wrapper around the OpenAI Responses API."""

    def __init__(self, config: dict):
        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def create(self, model, instructions, input, tools, tool_choice, **kwargs):
        return self.client.responses.create(
            model=model,
            instructions=instructions,
            input=input,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Chat Completions – for MiniMax and other OpenAI-compatible providers
# ---------------------------------------------------------------------------


class ChatCompletionsProvider:
    """Translates between Responses-API semantics and Chat Completions."""

    # Parameters specific to OpenAI Responses API that are not supported
    # by the standard Chat Completions endpoint.
    _UNSUPPORTED_PARAMS = frozenset({"reasoning", "reasoning_effort", "summary"})

    def __init__(self, config: dict):
        from openai import OpenAI

        provider_name = config.get("llm", {}).get("provider", "openai_compatible")
        preset = PROVIDER_PRESETS.get(provider_name, {})

        env_key = preset.get("env_key", f"{provider_name.upper()}_API_KEY")
        api_key = os.environ.get(env_key, "")
        base_url = config.get("llm", {}).get(
            "base_url", preset.get("base_url", "")
        )

        if not api_key:
            raise ValueError(
                f"API key not found. Set the {env_key} environment variable."
            )
        if not base_url:
            raise ValueError(
                f"base_url is required for provider {provider_name!r}. "
                "Set it in config['llm']['base_url']."
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.provider_name = provider_name

    # ---- public interface --------------------------------------------------

    def create(self, model, instructions, input, tools, tool_choice, **kwargs):
        messages = self._to_chat_messages(instructions, input)
        chat_tools = self._to_chat_tools(tools)

        clean_kwargs = {}
        for k, v in kwargs.items():
            if k in self._UNSUPPORTED_PARAMS:
                continue
            if k == "temperature":
                v = max(0.0, min(float(v), 1.0))
            if k == "max_output_tokens":
                clean_kwargs["max_tokens"] = v
                continue
            clean_kwargs[k] = v

        create_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            **clean_kwargs,
        }
        if chat_tools:
            create_kwargs["tools"] = chat_tools
            create_kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**create_kwargs)
        return self._to_llm_response(response)

    # ---- input conversion --------------------------------------------------

    def _to_chat_messages(self, instructions, input_items):
        """Convert Responses-API ``input`` list to Chat Completions ``messages``."""

        messages: List[Dict[str, Any]] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})

        i = 0
        while i < len(input_items):
            item = input_items[i]

            # --- dict-based items ---
            if isinstance(item, dict):
                if item.get("type") == "function_call_output":
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": item["call_id"],
                            "content": str(item.get("output", "")),
                        }
                    )
                elif "role" in item:
                    content = item.get("content", "")
                    if isinstance(content, list):
                        converted = []
                        for part in content:
                            ptype = part.get("type", "")
                            if ptype == "input_image":
                                converted.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": part["image_url"]},
                                    }
                                )
                            elif ptype == "input_text":
                                converted.append(
                                    {"type": "text", "text": part["text"]}
                                )
                            else:
                                converted.append(part)
                        messages.append({"role": item["role"], "content": converted})
                    else:
                        messages.append({"role": item["role"], "content": content})
                i += 1

            # --- output objects from a previous LLM response ---
            elif hasattr(item, "type"):
                text_content = ""
                tool_calls: List[Dict[str, Any]] = []

                # Group consecutive output objects into a single assistant msg
                while i < len(input_items) and hasattr(input_items[i], "type"):
                    curr = input_items[i]
                    if curr.type == "function_call":
                        tool_calls.append(
                            {
                                "id": curr.call_id,
                                "type": "function",
                                "function": {
                                    "name": curr.name,
                                    "arguments": curr.arguments,
                                },
                            }
                        )
                    elif curr.type == "message":
                        text_content = getattr(curr, "content", "")
                    # reasoning items are silently skipped
                    i += 1

                msg: Dict[str, Any] = {"role": "assistant"}
                if text_content:
                    msg["content"] = text_content
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                if text_content or tool_calls:
                    messages.append(msg)
            else:
                i += 1

        return messages

    # ---- tools conversion --------------------------------------------------

    @staticmethod
    def _to_chat_tools(tools):
        """Convert Responses-API tool dicts to Chat Completions format."""

        if not tools:
            return []
        chat_tools = []
        for tool in tools:
            if isinstance(tool, dict) and tool.get("type") == "function":
                chat_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {}),
                        },
                    }
                )
        return chat_tools

    # ---- response conversion -----------------------------------------------

    @staticmethod
    def _to_llm_response(response):
        """Convert a Chat Completions response to the ``LLMResponse`` envelope."""

        message = response.choices[0].message
        output: list = []
        output_text = message.content or ""

        # Strip <think>…</think> blocks that some models emit
        if output_text:
            output_text = re.sub(
                r"<think>.*?</think>\s*", "", output_text, flags=re.DOTALL
            ).strip()

        if output_text:
            output.append(OutputMessage(type="message", content=output_text))

        if message.tool_calls:
            for tc in message.tool_calls:
                output.append(
                    OutputFunctionCall(
                        type="function_call",
                        call_id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                )

        return LLMResponse(output=output, output_text=output_text)

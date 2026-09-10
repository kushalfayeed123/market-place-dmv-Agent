"""Anthropic model backend — swap path.

Uses the Anthropic Messages API with tool-calling support. This is the
documented swap path for when we move away from NVIDIA.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from ..provider import ModelBackend


class AnthropicBackend(ModelBackend):
    """Anthropic Claude backend using the Messages API."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 60.0,
    ):
        self._model = model
        self._api_key = api_key
        self._base_url = (base_url or "https://api.anthropic.com").rstrip("/")
        self._timeout = timeout

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    async def classify(
        self,
        message: str,
        system_prompt: str,
        valid_intents: list[str],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Classify message intent using Claude."""
        try:
            import anthropic
        except ImportError:
            return {"intent": "info", "confidence": 0.0, "reasoning": "", "error": "anthropic package not installed"}

        client = anthropic.AsyncAnthropic(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
        )

        try:
            resp = await client.messages.create(
                model=self._model,
                system=system_prompt,
                messages=[{"role": "user", "content": message}],
                temperature=temperature,
                max_tokens=256,
            )
            content = ""
            for block in resp.content:
                if block.type == "text":
                    content += block.text
            return _parse_classification(content, valid_intents)
        except Exception as e:
            return {"intent": "info", "confidence": 0.0, "reasoning": "", "error": str(e)}

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a Messages API response with tool-calling."""
        try:
            import anthropic
        except ImportError:
            yield {"type": "error", "error": "anthropic package not installed. Run: pip install anthropic"}
            return

        client = anthropic.AsyncAnthropic(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
        )

        # Convert OpenAI-format tools to Anthropic format
        anthropic_tools = _convert_tools_to_anthropic(tools) if tools else None

        try:
            async with client.messages.stream(
                model=self._model,
                system=system_prompt,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=anthropic_tools,
            ) as stream:
                async for event in stream:
                    event_type = getattr(event, "type", None)

                    if event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block and getattr(block, "type", None) == "tool_use":
                            yield {
                                "type": "tool_call",
                                "tool_call": {
                                    "id": block.id,
                                    "name": block.name,
                                    "arguments": {},  # Will be filled by delta
                                },
                            }

                    elif event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta:
                            delta_type = getattr(delta, "type", None)
                            if delta_type == "text_delta":
                                yield {"type": "delta", "content": delta.text}
                            elif delta_type == "input_json_delta":
                                # Partial JSON for tool input
                                yield {
                                    "type": "tool_call_delta",
                                    "tool_call_delta": {
                                        "id": getattr(event, "content_block", {}).get("id", ""),
                                        "partial_json": delta.partial_json,
                                    },
                                }

                    elif event_type == "message_delta":
                        delta = getattr(event, "delta", None)
                        if delta:
                            yield {"type": "finish", "reason": getattr(delta, "stop_reason", "end_turn")}

                    elif event_type == "message_stop":
                        # Emit usage
                        msg = stream.get_final_message() if hasattr(stream, "get_final_message") else None
                        if msg and hasattr(msg, "usage"):
                            yield {
                                "type": "usage",
                                "usage": {
                                    "input_tokens": getattr(msg.usage, "input_tokens", 0),
                                    "output_tokens": getattr(msg.usage, "output_tokens", 0),
                                },
                            }

        except Exception as e:
            yield {"type": "error", "error": f"Anthropic API error: {str(e)}"}


def _parse_classification(content: str, valid_intents: list[str]) -> dict[str, Any]:
    """Parse a classification response from the model."""
    import json
    # Try to parse as JSON first
    try:
        parsed = json.loads(content)
        intent = parsed.get("intent", "info")
        if intent not in valid_intents:
            intent = "info"
        return {
            "intent": intent,
            "confidence": float(parsed.get("confidence", 0.5)),
            "reasoning": str(parsed.get("reasoning", "")),
            "error": None,
        }
    except json.JSONDecodeError:
        pass
    # Fallback: search for intent keywords in the text
    content_lower = content.lower().strip()
    for intent in valid_intents:
        if intent in content_lower:
            return {"intent": intent, "confidence": 0.7, "reasoning": content.strip(), "error": None}
    # Default fallback
    return {"intent": "info", "confidence": 0.5, "reasoning": content.strip(), "error": None}


def _convert_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-format tools to Anthropic format."""
    anthropic_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            fn = tool.get("function", {})
            anthropic_tools.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
    return anthropic_tools

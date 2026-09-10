"""NVIDIA model backend — default provider.

Uses the OpenAI-compatible chat completions API at integrate.api.nvidia.com.
NVIDIA's API supports tool-calling via the OpenAI `tools` format.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from ..provider import ModelBackend


class NvidiaBackend(ModelBackend):
    """NVIDIA NIM backend using OpenAI-compatible chat completions."""

    BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 60.0,
    ):
        self._model = model
        self._api_key = api_key
        self._base_url = (base_url or self.BASE_URL).rstrip("/")
        self._timeout = timeout

    @property
    def provider_name(self) -> str:
        return "nvidia"

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
        """Classify message intent using a non-streaming chat completion."""
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            "temperature": temperature,
            "max_tokens": 256,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code != 200:
                    return {"intent": "info", "confidence": 0.0, "reasoning": "", "error": f"API error {resp.status_code}"}
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
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
        """Stream a chat completion with tool-calling support."""
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        yield {
                            "type": "error",
                            "error": f"NVIDIA API error {response.status_code}: {body.decode()[:500]}",
                        }
                        return

                    async for event in _parse_sse(response, tools):
                        yield event

        except httpx.TimeoutException:
            yield {"type": "error", "error": "NVIDIA API request timed out"}


async def _parse_sse(response: httpx.Response, tools: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    """Parse NVIDIA's SSE stream into structured events."""
    tool_calls_buffer: dict[int, dict[str, Any]] = {}

    async for line in response.aiter_lines():
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            break

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices", [])
        if not choices:
            continue

        choice = choices[0]
        delta = choice.get("delta", {})

        # Text content
        if delta.get("content"):
            yield {"type": "delta", "content": delta["content"]}

        # Tool calls (streamed in chunks)
        if delta.get("tool_calls"):
            for tc in delta["tool_calls"]:
                idx = tc.get("index", 0)
                if idx not in tool_calls_buffer:
                    tool_calls_buffer[idx] = {
                        "id": tc.get("id", ""),
                        "function": {"name": "", "arguments": ""},
                    }
                buf = tool_calls_buffer[idx]
                if tc.get("id"):
                    buf["id"] = tc["id"]
                fn = tc.get("function", {})
                if fn.get("name"):
                    buf["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    buf["function"]["arguments"] += fn["arguments"]

        # Finish reason
        finish = choice.get("finish_reason")
        if finish:
            # Emit any completed tool calls
            if tool_calls_buffer:
                tool_calls = []
                for idx in sorted(tool_calls_buffer):
                    buf = tool_calls_buffer[idx]
                    try:
                        args = json.loads(buf["function"]["arguments"])
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append({
                        "id": buf["id"],
                        "name": buf["function"]["name"],
                        "arguments": args,
                    })
                yield {"type": "tool_calls", "tool_calls": tool_calls}

            # Usage
            usage = chunk.get("usage")
            if usage:
                yield {"type": "usage", "usage": usage}

            yield {"type": "finish", "reason": finish}


def _parse_classification(content: str, valid_intents: list[str]) -> dict[str, Any]:
    """Parse a classification response from the model."""
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

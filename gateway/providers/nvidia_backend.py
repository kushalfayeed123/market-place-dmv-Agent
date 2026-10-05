"""NVIDIA model backend — default provider.

Uses the OpenAI-compatible chat completions API at integrate.api.nvidia.com.
NVIDIA's API supports tool-calling via the OpenAI `tools` format.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from gateway.provider import ModelBackend

# HTTP status codes that are safe to retry (transient errors)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
BASE_RETRY_DELAY = 1.0  # seconds


def _is_rate_limit_error(status_code: int, response_text: str) -> bool:
    """Check if the error is a rate limit or resource exhaustion error."""
    if status_code == 429:
        return True
    if status_code == 503:
        # Check for resource exhaustion messages
        lower_text = response_text.lower()
        return (
            "resourceexhausted" in lower_text
            or "rate limit" in lower_text
            or "request limit" in lower_text
        )
    return False


async def _retry_with_backoff(
    func,
    *args,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_RETRY_DELAY,
    **kwargs,
) -> Any:
    """Execute a function with exponential backoff retry for transient errors."""
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            result = await func(*args, **kwargs)
            # If result is a dict with error, check if it's retryable
            if isinstance(result, dict) and result.get("error"):
                error_msg = result.get("error", "")
                # Check if error message indicates a retryable error
                if (
                    any(
                        code in error_msg
                        for code in ["429", "500", "502", "503", "504"]
                    )
                    and attempt < max_retries
                ):
                    delay = base_delay * (2**attempt)
                    print(
                        f"[NVIDIA] Retryable error detected, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
            return result
        except httpx.TimeoutException:
            if attempt < max_retries:
                delay = base_delay * (2**attempt)
                print(
                    f"[NVIDIA] Request timed out, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(delay)
                continue
            raise
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2**attempt)
                print(
                    f"[NVIDIA] Error: {e}, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(delay)
                continue
            raise
    return result


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

        # Log the request payload
        print(
            f"[NVIDIA] Classification payload: model={self._model}, message={message[:100]}..."
        )

        async def _do_classify() -> dict[str, Any]:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code != 200:
                    error_msg = f"API error {resp.status_code}: {resp.text[:200]}"
                    print(f"[NVIDIA] Classification API error: {error_msg}")
                    return {
                        "intent": "info",
                        "confidence": 0.0,
                        "reasoning": "",
                        "error": error_msg,
                    }
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                print(f"[NVIDIA] Classification response: {content[:200]}")
                return _parse_classification(content, valid_intents)

        try:
            return await _retry_with_backoff(_do_classify)
        except Exception as e:
            error_msg = f"Exception: {e!s}"
            print(f"[NVIDIA] Classification error: {error_msg}")
            return {
                "intent": "info",
                "confidence": 0.0,
                "reasoning": "",
                "error": error_msg,
            }

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

        # Log the request payload
        user_msg = messages[0].get("content", "")[:100] if messages else ""
        print(
            f"[NVIDIA] Stream payload: model={self._model}, tools={len(tools)}, messages={len(messages)}, user_message={user_msg}..."
        )

        # Retry loop for rate limiting
        for attempt in range(MAX_RETRIES + 1):
            print(
                f"[NVIDIA] Streaming request to {self._model} with {len(tools)} tools (attempt {attempt + 1})"
            )
            try:
                async with (
                    httpx.AsyncClient(timeout=self._timeout) as client,
                    client.stream(
                        "POST",
                        f"{self._base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    ) as response,
                ):
                    if response.status_code != 200:
                        body = await response.aread()
                        error_msg = f"NVIDIA API error {response.status_code}: {body.decode()[:500]}"
                        print(f"[NVIDIA] Stream error: {error_msg}")

                        # Check if this is a retryable error
                        if (
                            _is_rate_limit_error(response.status_code, body.decode())
                            and attempt < MAX_RETRIES
                        ):
                            delay = BASE_RETRY_DELAY * (2**attempt)
                            print(f"[NVIDIA] Rate limited, retrying in {delay:.1f}s...")
                            await asyncio.sleep(delay)
                            continue

                        yield {
                            "type": "error",
                            "error": error_msg,
                        }
                        return

                    event_count = 0
                    async for event in _parse_sse(response, tools):
                        event_count += 1
                        if event.get("type") == "tool_calls":
                            tc_names = [
                                (tc.get('function') or {}).get('name') or tc.get('name')
                                for tc in event.get('tool_calls', [])
                            ]
                            print(f"[NVIDIA] Stream yielded tool_calls event: {tc_names}")
                        yield event
                    print(f"[NVIDIA] Stream completed with {event_count} events")
                    return  # Success, exit retry loop

            except httpx.TimeoutException:
                error_msg = "NVIDIA API request timed out"
                print(f"[NVIDIA] {error_msg}")
                if attempt < MAX_RETRIES:
                    delay = BASE_RETRY_DELAY * (2**attempt)
                    print(f"[NVIDIA] Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                yield {"type": "error", "error": error_msg}
                return
            except Exception as e:
                error_msg = f"Unexpected error: {e!s}"
                print(f"[NVIDIA] {error_msg}")
                if attempt < MAX_RETRIES:
                    delay = BASE_RETRY_DELAY * (2**attempt)
                    print(f"[NVIDIA] Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                yield {"type": "error", "error": error_msg}
                return


async def _parse_sse(
    response: httpx.Response, tools: list[dict[str, Any]]
) -> AsyncIterator[dict[str, Any]]:
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
                    # Keep arguments as JSON string for API compatibility
                    args_str = buf["function"]["arguments"]
                    try:
                        json.loads(args_str)  # Validate it's valid JSON
                    except json.JSONDecodeError:
                        args_str = "{}"
                    tool_calls.append(
                        {
                            "id": buf["id"],
                            "type": "function",
                            "function": {
                                "name": buf["function"]["name"],
                                "arguments": args_str,
                            },
                        }
                    )
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
            return {
                "intent": intent,
                "confidence": 0.7,
                "reasoning": content.strip(),
                "error": None,
            }
    # Default fallback
    return {
        "intent": "info",
        "confidence": 0.5,
        "reasoning": content.strip(),
        "error": None,
    }

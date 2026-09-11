"""Model router — failover across multiple NVIDIA NIM models.

When one model is rate-limited or unavailable, automatically switches
to the next model in the list to ensure high availability.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .provider import ModelBackend
from .providers.nvidia_backend import NvidiaBackend


@dataclass
class ModelSpec:
    """Specification for a model available for failover."""
    name: str
    context: int = 131_000
    supports_tools: bool = True
    rpm: int = 40  # requests per minute


# Default model pool — order matters: first model is primary, rest are fallbacks
NIM_MODELS = [
    ModelSpec("nvidia/nemotron-3-super-120b-a12b", context=262_000, supports_tools=True, rpm=40),
    ModelSpec("nvidia/nemotron-3-ultra-550b-a55b", context=1_000_000, supports_tools=True, rpm=40),
    ModelSpec("minimaxai/minimax-m3", context=1_000_000, supports_tools=True, rpm=40),
    ModelSpec("moonshotai/kimi-k2.6", context=262_000, supports_tools=True, rpm=40),
    ModelSpec("openai/gpt-oss-120b", context=131_000, supports_tools=True, rpm=40),
    ModelSpec("meta/llama-3.3-70b-instruct", context=131_000, supports_tools=True, rpm=40),
    ModelSpec("deepseek-ai/deepseek-v4-flash-0731", context=1_000_000, supports_tools=True, rpm=40),
]

# Cooldown period for rate-limited models (seconds)
RATE_LIMIT_COOLDOWN = 60.0


@dataclass
class _ModelEntry:
    """Internal tracking for a model in the pool."""
    spec: ModelSpec
    backend: NvidiaBackend
    cooldown_until: float = 0.0
    failure_count: int = 0

    @property
    def is_available(self) -> bool:
        return time.time() >= self.cooldown_until

    def mark_rate_limited(self) -> None:
        self.cooldown_until = time.time() + RATE_LIMIT_COOLDOWN
        self.failure_count += 1

    def mark_failed(self) -> None:
        self.failure_count += 1

    def reset(self) -> None:
        self.cooldown_until = 0.0
        self.failure_count = 0


class FailoverBackend(ModelBackend):
    """Model backend that automatically fails over to alternate models.

    Wraps multiple NvidiaBackend instances and switches between them
    when one returns rate limit or service unavailable errors.
    """

    def __init__(
        self,
        models: list[ModelSpec],
        api_key: str,
        base_url: str | None = None,
        timeout: float = 60.0,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._models: list[_ModelEntry] = []
        self._active_index: int = 0
        self._initialize_models(models)

    def _initialize_models(self, models: list[ModelSpec]) -> None:
        """Create backend instances for each model spec."""
        for spec in models:
            backend = NvidiaBackend(
                model=spec.name,
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
            )
            self._models.append(_ModelEntry(spec=spec, backend=backend))

    @property
    def provider_name(self) -> str:
        return "nvidia-failover"

    @property
    def model_name(self) -> str:
        return self._models[self._active_index].spec.name

    @property
    def active_model(self) -> str:
        """Return the name of the currently active model."""
        return self._models[self._active_index].spec.name

    def _get_available_model(self) -> int:
        """Get the index of an available model, preferring the primary."""
        # First, try to find an available model starting from index 0
    def _switch_to_next_model(self) -> None:
        """Switch to the next available model."""
        self._active_index = self._get_available_model()
        print(f"[FAILOVER] Switched to model: {self.active_model}")

    async def classify(
        self,
        message: str,
        system_prompt: str,
        valid_intents: list[str],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Classify intent with automatic failover."""
        last_error: dict[str, Any] | None = None

        for _ in range(len(self._models)):
            entry = self._models[self._active_index]
            if not entry.is_available:
                self._switch_to_next_model()
                continue

            try:
                result = await entry.backend.classify(
                    message=message,
                    system_prompt=system_prompt,
                    valid_intents=valid_intents,
                    temperature=temperature,
                )
                # Check if result indicates a rate limit error
                if result.get("error"):
                    error_msg = result.get("error", "").lower()
                    if any(indicator in error_msg for indicator in ["429", "503", "rate limit", "resourceexhausted"]):
                        print(f"[FAILOVER] Model {entry.spec.name} rate limited: {result['error'][:100]}")
                        entry.mark_rate_limited()
                        self._switch_to_next_model()
                        last_error = result
                        continue
                    # Non-rate-limit error — return it
                    return result
                # Success — reset failure count on this model
                entry.reset()
                return result
            except Exception as e:
                print(f"[FAILOVER] Model {entry.spec.name} exception: {e}")
                entry.mark_failed()
                last_error = {"intent": "info", "confidence": 0.0, "reasoning": "", "error": str(e)}
                self._switch_to_next_model()
                continue

        # All models exhausted
        return last_error or {"intent": "info", "confidence": 0.0, "reasoning": "", "error": "All models unavailable"}

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream response with automatic failover."""
        for attempt in range(len(self._models)):
            entry = self._models[self._active_index]
            if not entry.is_available:
                self._switch_to_next_model()
                continue

            print(f"[FAILOVER] Using model: {entry.spec.name}")
            event_count = 0
            has_error = False

            try:
                async for event in entry.backend.stream(
                    messages=messages,
                    tools=tools,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    event_count += 1
                    if event.get("type") == "error":
                        has_error = True
                        error_msg = event.get("error", "").lower()
                        # Check if this is a rate limit error
                        if any(indicator in error_msg for indicator in ["429", "503", "rate limit", "resourceexhausted"]):
                            print(f"[FAILOVER] Model {entry.spec.name} rate limited during stream")
                            entry.mark_rate_limited()
                            self._switch_to_next_model()
                            break
                    yield event

                if has_error:
                    # This model had a rate limit error, try next
                    continue

                # Success — reset failure count
                entry.reset()
                return

            except Exception as e:
                print(f"[FAILOVER] Model {entry.spec.name} exception during stream: {e}")
                entry.mark_failed()
                self._switch_to_next_model()
                continue

        # All models exhausted — yield error
        yield {
            "type": "error",
            "error": "I apologize, but all AI models are currently experiencing high demand. "
                     "Please wait a moment and try again.",
        }

    def _switch_to_next_model(self) -> None:
        """Switch to the next available model."""
        self._active_index = self._get_available_model()
        print(f"[FAILOVER] Switched to model: {self.active_model}")
    backend: NvidiaBackend
    cooldown_until: float = 0.0
    failure_count: int = 0

    @property
    def is_available(self) -> bool:
        return time.time() >= self.cooldown_until

    def mark_rate_limited(self) -> None:
        self.cooldown_until = time.time() + RATE_LIMIT_COOLDOWN
        self.failure_count += 1

    def mark_failed(self) -> None:
        self.failure_count += 1

    def reset(self) -> None:
        self.cooldown_until = 0.0
        self.failure_count = 0

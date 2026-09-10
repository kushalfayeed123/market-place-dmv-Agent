"""ModelBackend port — provider-agnostic interface for LLM calls.

The Gateway talks to models through this interface so the provider can be
swapped (NVIDIA, Anthropic, etc.) without touching tools or flows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional


class ModelBackend(ABC):
    """Abstract interface for a model backend that supports tool-calling."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name, e.g. 'nvidia', 'anthropic'."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The specific model being used."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a model response.

        Yields event dicts:
        - {"type": "delta", "content": "..."} — text delta
        - {"type": "tool_call", "tool_call": {"id": "...", "name": "...", "arguments": {...}}}
        - {"type": "tool_calls", "tool_calls": [...]} — batch of tool calls
        - {"type": "usage", "usage": {"input_tokens": N, "output_tokens": N}}
        - {"type": "finish", "reason": "end_turn"|"tool_use"|"max_tokens"}
        - {"type": "error", "error": "..."}
        """
        ...

    @abstractmethod
    async def classify(
        self,
        message: str,
        system_prompt: str,
        valid_intents: list[str],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Classify a message into one of the valid intents.

        Returns a dict: {"intent": str, "confidence": float, "reasoning": str, "error": str|None}
        """
        ...


def get_backend(provider: str, model: str, api_key: str, **kwargs: Any) -> ModelBackend:
    """Factory: create a ModelBackend for the given provider."""
    provider = provider.lower()
    if provider == "nvidia":
        from .providers.nvidia_backend import NvidiaBackend
        return NvidiaBackend(model=model, api_key=api_key, **kwargs)
    if provider == "anthropic":
        from .providers.anthropic_backend import AnthropicBackend
        return AnthropicBackend(model=model, api_key=api_key, **kwargs)
    raise ValueError(f"Unknown model provider: {provider!r}. Supported: 'nvidia', 'anthropic'")

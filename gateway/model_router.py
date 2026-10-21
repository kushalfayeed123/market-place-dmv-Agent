"""Model router — failover across multiple NVIDIA NIM models.

When one model is rate-limited or unavailable, automatically switches
to the next model in the list to ensure high availability.

Error classification rules (per probe results 2026-10-20):
  * permanent (404, 410, 401, 403) → mark model dead for 10 min, switch.
  * transient (429, 5xx, timeouts) → let NvidiaBackend retry internally;
    if still failing, mark model rate-limited for 60 s, switch.
  * tool rejection (400 on /tools) → mark model no_tools, switch to
    a tool-capable model.
"""
from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from .provider import ModelBackend
from .providers.nvidia_backend import NvidiaBackend

logger = logging.getLogger("gateway.model_router")

# Default model pool — order matters: first is primary, rest are fallbacks.
# Updated per probe results: these three returned 200 AND emitted tool_calls.
_DEFAULT_NVIDIA_CHAT_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "meta/muse-glimmer-30b",
]


def _parse_chat_models(env_val: str | None) -> list[str]:
    if env_val:
        return [m.strip() for m in env_val.split(",") if m.strip()]
    return list(_DEFAULT_NVIDIA_CHAT_MODELS)


def get_default_chat_models() -> list[str]:
    return _parse_chat_models(os.getenv("NVIDIA_CHAT_MODELS"))

    return _parse_chat_models(os.getenv("NVIDIA_CHAT_MODELS"))


# ── User-facing error messages ──────────────────────────────────────────────

_PERMANENT_MSG = (
    "AI model '{model}' is not available for this account (HTTP {code}). "
    "The failover pool will skip it for the next {ttl:.0f} minutes."
)
_RATE_LIMIT_MSG = (
    "AI model '{model}' is rate-limited (HTTP 429). "
    "Switching to another model."
)
_ALL_EXHAUSTED_MSG = (
    "All AI models are currently unavailable. "
    "The models we tried were all rate-limited or returned errors. "
    "Please wait a moment and try again."
)
_NO_MODELS_MSG = (
    "No AI models are configured. "
    "Set the NVIDIA_CHAT_MODELS environment variable."
)
_ALL_DEAD_MSG = (
    "All AI models are currently unavailable due to permanent errors "
    "(404/410/403). Please try again later."
)
_TOOL_REJECTION_MSG = (
    "AI model '{model}' does not support the requested tools. "
    "Switching to a different model."
)


@dataclass
class _ModelEntry:
    """Internal tracking for a model in the failover pool."""
    spec: ModelSpec
    backend: NvidiaBackend
    dead_until: float = 0.0
    rate_limited_until: float = 0.0
    failure_count: int = 0
    no_tools: bool = False  # model rejected the tool schema (HTTP 400)

    @property
    def is_available(self) -> bool:
        """True if not in a permanent or rate-limit cooldown."""
        now = time.time()
        return now >= self.dead_until and now >= self.rate_limited_until

    def mark_permanent_failure(self) -> None:
        self.dead_until = time.time() + DEAD_MODEL_TTL
        self.failure_count += 1
        logger.warning(
            "Model %s marked DEAD (permanent error) for %.0fs",
            self.spec.name, DEAD_MODEL_TTL,
        )

    def mark_rate_limited(self) -> None:
        self.rate_limited_until = time.time() + RATE_LIMIT_COOLDOWN
        self.failure_count += 1

    def reset(self) -> None:
        self.dead_until = 0.0
        self.rate_limited_until = 0.0
        self.failure_count = 0


@dataclass
class ModelSpec:
    """Specification for a model available for failover."""
    name: str
    context: int = 262_000
    supports_tools: bool = True
    rpm: int = 40


# Build the default model pool from env or hardcoded defaults.
NIM_MODELS: list[ModelSpec] = [
    ModelSpec(name=name) for name in _parse_chat_models(os.getenv("NVIDIA_CHAT_MODELS"))
]

# Status codes that mean the model is permanently unavailable.
PERMANENT_STATUS_CODES: set[int] = {401, 403, 404, 410}
# Status codes that are transient / retryable.
TRANSIENT_STATUS_CODES: set[int] = {408, 429, 500, 502, 503, 504}

# Cooldown periods.
DEAD_MODEL_TTL = 600.0       # 10 minutes — permanent errors keep model dead this long.
RATE_LIMIT_COOLDOWN = 60.0  # 1 minute — rate-limited models cool down faster.


class FailoverBackend(ModelBackend):
    """Model backend with automatic failover across multiple NVIDIA NIM models."""

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
        """Create a NvidiaBackend per model spec."""
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
        return self._models[self._active_index].spec.name

    @property
    def all_models(self) -> list[str]:
        return [e.spec.name for e in self._models]

    @property
    def available_models(self) -> list[str]:
        return [e.spec.name for e in self._models if e.is_available]

    # ── Model selection ──────────────────────────────────────────────────

    def _get_available_model(self) -> int:
        """Find the index of the next available model after _active_index.

        Skips models on cooldown (permanent or rate-limit).
        Returns the current index if no better option exists.
        """
        n = len(self._models)
        if n == 0:
            return 0
        start = self._active_index
        for i in range(1, n + 1):
            idx = (start + i) % n
            entry = self._models[idx]
            if entry.is_available:
                return idx
        return self._active_index

    def _switch_to_next_model(self) -> bool:
        """Switch to the next available model.

        Returns True if the active index changed, False if all dead.
        """
        old = self._active_index
        self._active_index = self._get_available_model()
        if self._active_index != old:
            logger.info(
                "[FAILOVER] Switched model: %s -> %s",
                self._models[old].spec.name,
                self._models[self._active_index].spec.name,
            )
            return True
        return False

    # ── Error classification ───────────────────────────────────────────

    @staticmethod
    def _classify_error(error_event: dict[str, Any]) -> tuple[str, int | None]:
        """Classify an error event from the backend.

        Returns (classification, status_code) where classification is one of:
          'permanent', 'transient', 'tool_rejected', 'unknown'
        """
        status_code = error_event.get("status_code")
        error_msg = str(error_event.get("error", ""))

        if isinstance(status_code, int):
            if status_code in PERMANENT_STATUS_CODES:
                return "permanent", status_code
            if status_code == 400:
                return "tool_rejected", status_code
            if status_code in TRANSIENT_STATUS_CODES:
                return "transient", status_code

        # Fallback: pattern-match HTTP codes embedded in the error string.
        msg_lower = error_msg.lower()
        for code in sorted(PERMANENT_STATUS_CODES):
            if f" {code}" in msg_lower or f" {code}:" in msg_lower:
                return "permanent", code
        for code in sorted(TRANSIENT_STATUS_CODES):
            if f" {code}" in msg_lower or f" {code}:" in msg_lower:
                return "transient", code

        return "unknown", status_code

    # ── Classification (non-streaming intent routing) ──────────────────

    async def classify(
        self,
        message: str,
        system_prompt: str,
        valid_intents: list[str],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Classify message intent with failover across models."""
        last_error: dict[str, Any] | None = None
        active_backup = self._active_index

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
                if result.get("error"):
                    classification, _ = self._classify_error(
                        {"error": result.get("error", ""), "status_code": result.get("status_code")}
                    )
                    if classification == "permanent":
                        entry.mark_permanent_failure()
                        last_error = result
                        self._switch_to_next_model()
                        continue
                    if classification == "transient":
                        entry.mark_rate_limited()
                        last_error = result
                        self._switch_to_next_model()
                        continue
                    # Unknown error — return it
                    return result

                entry.reset()
                return result

            except Exception as e:
                logger.exception(
                    "classify: model %s exception: %s", entry.spec.name, e
                )
                entry.mark_permanent_failure()
                last_error = {
                    "intent": "info",
                    "confidence": 0.0,
                    "reasoning": "",
                    "error": f"Model {entry.spec.name} failed: {e}",
                }
                self._switch_to_next_model()
                continue

        # All models exhausted
        if last_error:
            return last_error
        return {
            "intent": "info",
            "confidence": 0.0,
            "reasoning": "",
            "error": "All AI models are currently unavailable.",
        }

    # ── Streaming ──────────────────────────────────────────────────────────

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a model response with automatic failover.

        Failover is transparent: errors that occur before any token is
        emitted cause a silent switch to the next model.  Mid-stream
        errors cannot be recovered and are yielded as-is.
        """
        n = len(self._models)
        if n == 0:
            yield {"type": "error", "error": _NO_MODELS_MSG}
            return

        tried: set[str] = set()
        last_error: dict[str, Any] | None = None

        for attempt in range(n):
            entry = self._models[self._active_index]

            # Skip models on cooldown.
            if not entry.is_available:
                logger.debug("Model %s on cooldown, skipping", entry.spec.name)
                if self._switch_to_next_model():
                    entry = self._models[self._active_index]
                    if entry.spec.name in tried:
                        continue
                else:
                    continue

            # Skip models that reject tools when tools are required.
            if tools and entry.no_tools:
                logger.debug("Model %s rejects tools, skipping", entry.spec.name)
                if self._switch_to_next_model():
                    entry = self._models[self._active_index]
                    if entry.spec.name in tried:
                        continue
                else:
                    continue

            # Don't try the same model twice.
            if entry.spec.name in tried:
                continue

            tried.add(entry.spec.name)
            event_count = 0
            error_event: dict[str, Any] | None = None

            logger.info(
                "[FAILOVER] Attempt %d/%d — model: %s",
                attempt + 1, n, entry.spec.name,
            )

            try:
                async for event in entry.backend.stream(
                    messages=messages,
                    tools=tools,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    if event.get("type") == "error":
                        error_event = event
                        classification, status_code = self._classify_error(event)

                        # No tokens emitted yet → failover transparently.
                        if event_count == 0:
                            if classification == "permanent":
                                entry.mark_permanent_failure()
                            elif classification == "transient":
                                entry.mark_rate_limited()
                            elif classification == "tool_rejected":
                                entry.no_tools = True

                            self._switch_to_next_model()
                            break  # try next model

                        # Tokens already sent → must yield error, can't failover.
                        yield event
                        return

                    else:
                        event_count += 1
                        yield event

                else:
                    # Stream completed without error — success.
                    entry.reset()
                    return

            except Exception as e:
                logger.exception(
                    "Unexpected error on model %s: %s", entry.spec.name, e
                )
                if event_count == 0:
                    entry.mark_permanent_failure()
                    last_error = {
                        "type": "error",
                        "error": f"Unexpected error on {entry.spec.name}: {e}",
                    }
                    self._switch_to_next_model()
                    continue
                else:
                    yield {"type": "error", "error": f"Streaming error: {e}"}
                    return

            if error_event:
                last_error = error_event

        # All models exhausted.
        if last_error:
            yield last_error
        else:
            if all(e.dead_until > time.time() for e in self._models):
                yield {"type": "error", "error": _ALL_DEAD_MSG}
            else:
                yield {"type": "error", "error": _ALL_EXHAUSTED_MSG}





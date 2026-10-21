"""Tests for the model-router failover engine."""
from __future__ import annotations

import sys, os
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, AGENT_ROOT)

from gateway.model_router import (
    FailoverBackend,
    ModelSpec,
    _ModelEntry,
)


# ── Test helpers ───────────────────────────────────────────────────────────

class _FakeBackend:
    """Stand-in for NvidiaBackend that yields pre-configured events."""

    def __init__(self, events: list[dict]):
        self._events = events
        self.call_count = 0

    async def stream(self, **kwargs) -> AsyncIterator[dict[str, Any]]:
        self.call_count += 1
        for ev in self._events:
            yield ev

    async def classify(self, **kwargs) -> dict[str, Any]:
        self.call_count += 1
        return {"intent": "info", "confidence": 0.5, "reasoning": "", "error": None}


def _make_backend(specs, backends):
    """Create a FailoverBackend with pre-injected backends for testing."""
    fb = FailoverBackend.__new__(FailoverBackend)
    fb._api_key = "test-key"
    fb._base_url = None
    fb._timeout = 60.0
    fb._models = []
    fb._active_index = 0
    for spec, backend in zip(specs, backends):
        fb._models.append(_ModelEntry(spec=spec, backend=backend))
    return fb


# ── Error classification tests ────────────────────────────────────────────

class TestErrorClassification:

    def test_404_is_permanent(self):
        event = {"type": "error", "error": "NVIDIA API error 404: Not found", "status_code": 404}
        cls, code = FailoverBackend._classify_error(event)
        assert cls == "permanent"
        assert code == 404

    def test_410_is_permanent(self):
        event = {"type": "error", "error": "NVIDIA API error 410: Gone", "status_code": 410}
        cls, code = FailoverBackend._classify_error(event)
        assert cls == "permanent"
        assert code == 410

    def test_429_is_transient(self):
        event = {"type": "error", "error": "NVIDIA API error 429: rate limited", "status_code": 429}
        cls, code = FailoverBackend._classify_error(event)
        assert cls == "transient"
        assert code == 429

    def test_400_is_tool_rejected(self):
        event = {"type": "error", "error": "NVIDIA API error 400: tool not supported", "status_code": 400}
        cls, code = FailoverBackend._classify_error(event)
        assert cls == "tool_rejected"
        assert code == 400

    def test_fallback_pattern_matching(self):
        """Without status_code, classify via string pattern."""
        event = {"type": "error", "error": "API error 404: Not Found", "status_code": None}


class TestModelSelection:
    """_get_available_model should skip dead and rate-limited models."""

    def test_skip_dead_models(self):
        specs = [ModelSpec("m1"), ModelSpec("m2"), ModelSpec("m3")]
        backends = [MagicMock(), MagicMock(), MagicMock()]
        fb = _make_backend(specs, backends)
        fb._models[0].dead_until = 9999999999.0
        fb._active_index = 0
        assert fb._get_available_model() == 1

    def test_skip_rate_limited(self):
        specs = [ModelSpec("m1"), ModelSpec("m2"), ModelSpec("m3")]
        backends = [MagicMock(), MagicMock(), MagicMock()]
        fb = _make_backend(specs, backends)
        fb._models[0].rate_limited_until = 9999999999.0
        fb._active_index = 0
        assert fb._get_available_model() == 1

    def test_all_dead_returns_current(self):
        specs = [ModelSpec("m1"), ModelSpec("m2")]
        backends = [MagicMock(), MagicMock()]
        fb = _make_backend(specs, backends)
        for entry in fb._models:
            entry.dead_until = 9999999999.0
        assert fb._get_available_model() == 0

    def test_model_properties(self):
        specs = [ModelSpec("nvidia/a"), ModelSpec("nvidia/b")]
        fb = _make_backend(specs, [MagicMock(), MagicMock()])
        assert fb.model_name == "nvidia/a"
        assert fb.all_models == ["nvidia/a", "nvidia/b"]
        assert fb.available_models == ["nvidia/a", "nvidia/b"]

class TestStreamFailover:
    @pytest.mark.asyncio
    async def test_failover_on_404(self):
        specs = [ModelSpec('bad-404'), ModelSpec('good-200')]
        backends = [
            _FakeBackend([{'type': 'error', 'error': 'NVIDIA API error 404: Not found', 'status_code': 404}]),
            _FakeBackend([{'type': 'delta', 'content': 'hello'}]),
        ]
        fb = _make_backend(specs, backends)
        events = [ev async for ev in fb.stream(messages=[], tools=[], system_prompt='test')]
        assert len(events) == 1
        assert events[0]['type'] == 'delta'
        assert events[0]['content'] == 'hello'
        assert fb._models[0].dead_until > 0

    @pytest.mark.asyncio
    async def test_failover_on_410(self):
        specs = [ModelSpec('retired'), ModelSpec('working')]
        backends = [
            _FakeBackend([{'type': 'error', 'error': '410 Gone', 'status_code': 410}]),
            _FakeBackend([{'type': 'delta', 'content': 'ok'}]),
        ]
        fb = _make_backend(specs, backends)
        events = [ev async for ev in fb.stream(messages=[], tools=[], system_prompt='test')]
        assert events[0]['content'] == 'ok'
        assert fb._models[0].dead_until > 0

    @pytest.mark.asyncio
    async def test_all_models_404(self):
        specs = [ModelSpec('a'), ModelSpec('b'), ModelSpec('c')]
        backends = [
            _FakeBackend([{'type': 'error', 'error': '404 Not found', 'status_code': 404}]),
            _FakeBackend([{'type': 'error', 'error': '404 Not found', 'status_code': 404}]),
            _FakeBackend([{'type': 'error', 'error': '404 Not found', 'status_code': 404}]),
        ]
        fb = _make_backend(specs, backends)
        events = [ev async for ev in fb.stream(messages=[], tools=[], system_prompt='test')]
        assert len(events) == 1
        assert events[0]['type'] == 'error'

    @pytest.mark.asyncio
    async def test_successful_stream(self):
        specs = [ModelSpec('good')]
        backends = [_FakeBackend([{'type': 'delta', 'content': 'a'}, {'type': 'delta', 'content': 'b'}])]
        fb = _make_backend(specs, backends)
        events = [ev async for ev in fb.stream(messages=[], tools=[], system_prompt='test')]
        assert len(events) == 2
        assert fb._models[0].dead_until == 0

    @pytest.mark.asyncio
    async def test_no_retry_on_404(self):
        specs = [ModelSpec('m1')]
        backend = _FakeBackend([{'type': 'error', 'error': '404 Not found', 'status_code': 404}])
        fb = _make_backend(specs, [backend])
        async for _ in fb.stream(messages=[], tools=[], system_prompt='test'):
            pass
        assert backend.call_count == 1

    @pytest.mark.asyncio
    async def test_dead_model_not_retried_next_call(self):
        specs = [ModelSpec('was-bad'), ModelSpec('good')]
        backends = [
            _FakeBackend([{'type': 'error', 'error': '404', 'status_code': 404}]),
            _FakeBackend([{'type': 'delta', 'content': 'ok'}]),
        ]
        fb = _make_backend(specs, backends)
        events = [ev async for ev in fb.stream(messages=[], tools=[], system_prompt='t')]
        assert events[0]['content'] == 'ok'
        assert fb._models[0].dead_until > 0
        events = [ev async for ev in fb.stream(messages=[], tools=[], system_prompt='t')]
        assert events[0]['content'] == 'ok'
        assert backends[0].call_count == 1

    @pytest.mark.asyncio
    async def test_mid_stream_error_yeielded(self):
        specs = [ModelSpec('m1')]
        backends = [_FakeBackend([
            {'type': 'delta', 'content': 'partial'},
            {'type': 'error', 'error': '503 failed', 'status_code': 503},
        ])]
        fb = _make_backend(specs, backends)
        events = [ev async for ev in fb.stream(messages=[], tools=[], system_prompt='test')]
        assert len(events) == 2
        assert events[0]['content'] == 'partial'
        assert events[1]['type'] == 'error'

#!/usr/bin/env python3
"""Probe NVIDIA NIM /chat/completions and /embeddings for model health.

Reads NVIDIA_API_KEY from the environment (or .env) and NEVER prints the key.
Usage:
    python scripts/probe_nvidia.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx

BASE_URL = "https://integrate.api.nvidia.com/v1"
TIMEOUT = 30.0
SLEEP_BETWEEN = 2.0  # free tier is ~40 req/min

CHAT_CANDIDATES = [
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-3-super-120b-a12b",       # control: known 404 for this account
    "openai/gpt-oss-20b",
    "poolside/laguna-xs-2.1",
    "meta/muse-glimmer-30b",
    "meta/llama-3.2-90b-vision-instruct",
    "z-ai/glm-5.2",
    "deepseek-ai/deepseek-v4-pro",
    "minimaxai/minimax-m3",
    "moonshotai/kimi-k2.6",                    # reported flaky: 404 or hangs
]

EMBEDDING_CANDIDATES = [
    "nvidia/llama-nemotron-embed-1b-v2",      # 2048 dims, Matryoshka
    "nvidia/llama-3.2-nv-embedqa-1b-v2",     # old name
    "nvidia/nv-embedqa-e5-v5",                # currently used by code
]

# Single trivial tool definition for tool-calling probe
_TOOL = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "A no-op that always succeeds.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _load_key() -> str:
    """Load NVIDIA_API_KEY from env or .env; never print it."""
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
                key = os.environ.get("NVIDIA_API_KEY", "")
            except Exception:
                pass
    if not key:
        print("ERROR: NVIDIA_API_KEY not set in environment.", file=sys.stderr)
        sys.exit(1)
    return key


_API_KEY = _load_key()


def _auth_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


@dataclass
class ChatResult:
    model: str
    status_code: int | None
    latency_ms: int
    tool_calls_accepted: bool
    error: str | None
    detail: str | None


@dataclass
class EmbedResult:
    model: str
    status_code: int | None
    dim_returned: int | None
    dim_requested: int | None
    dim_honored: bool | None
    error: str | None


def fetch_models() -> list[str]:
    """GET /v1/models and return model ids."""
    resp = httpx.get(f"{BASE_URL}/models", headers=_auth_headers(), timeout=TIMEOUT)
    if resp.status_code != 200:
        print(f"[models] GET /v1/models -> {resp.status_code}: {resp.text[:300]}")
        return []
    data = resp.json()
    ids = [m["id"] for m in data.get("data", [])]
    print(f"[models] {len(ids)} models returned by /v1/models:")
    for mid in ids:
        in_candidates = mid in CHAT_CANDIDATES or mid in EMBEDDING_CANDIDATES
        flag = "  <-- candidate" if in_candidates else ""
        print(f"  - {mid}{flag}")
    return ids

    return ids


def probe_chat(model: str) -> ChatResult:
    """Test basic chat + tool calling for one model."""
    # 1. Basic chat
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 8,
        "stream": False,
    }
    t0 = time.monotonic()
    status = None
    error = None
    detail = None
    try:
        resp = httpx.post(
            f"{BASE_URL}/chat/completions",
            json=payload,
            headers=_auth_headers(),
            timeout=TIMEOUT,
        )
        latency = int((time.monotonic() - t0) * 1000)
        status = resp.status_code
        if status != 200:
            detail = resp.text[:500]
            error = f"HTTP {status}"
    except httpx.HTTPStatusError as e:
        latency = int((time.monotonic() - t0) * 1000)
        status = e.response.status_code if e.response else None
        detail = e.response.text[:500] if e.response else str(e)
        error = f"HTTP {status}" if status else str(e)
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        latency = int((time.monotonic() - t0) * 1000)
        error = f"connection error: {type(e).__name__}"
        detail = str(e)[:300]
    except Exception as e:
        latency = int((time.monotonic() - t0) * 1000)
        error = str(e)[:300]
        detail = str(e)[:500]

    # 2. Tool-calling test (skip if basic chat already failed)
    tool_accepted = False
    if status == 200:
        time.sleep(SLEEP_BETWEEN)
        tool_payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Call the ping tool."}],
            "max_tokens": 8,
            "stream": False,
            "tools": [_TOOL],
            "tool_choice": "required",
        }
        t0 = time.monotonic()
        try:
            resp = httpx.post(
                f"{BASE_URL}/chat/completions",
                json=tool_payload,
                headers=_auth_headers(),
                timeout=TIMEOUT,
            )
            latency = int((time.monotonic() - t0) * 1000)
            status = resp.status_code
            if status == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices and (choices[0].get("message", {}).get("tool_calls")):
                    tool_accepted = True
            elif status != 200:
                detail = (detail or "") + f" | tool: HTTP {resp.status_code}: {resp.text[:300]}"
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            latency = int((time.monotonic() - t0) * 1000)
            error = f"{error or ''} | tool: {type(e).__name__}"
        except Exception as e:
            latency = int((time.monotonic() - t0) * 1000)
            error = f"{error or ''} | tool: {str(e)[:200]}"
    else:
        latency = int((time.monotonic() - t0) * 1000)

    return ChatResult(
        model=model,
        status_code=status,
        latency_ms=latency,
        tool_calls_accepted=tool_accepted,
        error=error,
        detail=detail,
    )


def probe_embedding(model: str) -> EmbedResult:
    """Test embedding generation for one model."""
    # 1. Default request (no dimensions param)
    payload = {
        "model": model,
        "input": ["hello"],
        "input_type": "query",
        "encoding_format": "float",
        "truncate": "END",
    }
    t0 = time.monotonic()
    status = None
    dim_returned = None
    error = None
    try:
        resp = httpx.post(
            f"{BASE_URL}/embeddings",
            json=payload,
            headers=_auth_headers(),
            timeout=TIMEOUT,
        )
        latency = int((time.monotonic() - t0) * 1000)
        status = resp.status_code
        if status == 200:
            data = resp.json()
            if data.get("data"):
                dim_returned = len(data["data"][0]["embedding"])
        else:
            error = f"HTTP {status}: {resp.text[:300]}"
    except Exception as e:
        latency = int((time.monotonic() - t0) * 1000)
        error = str(e)[:300]

    # 2. Request with dimensions param (Matryoshka / dimensionality reduction)
    dim_requested = None
    dim_honored = None
    if status == 200 and dim_returned and dim_returned >= 512:
        time.sleep(SLEEP_BETWEEN)
        dim_requested = 512
        dim_payload = {
            "model": model,
            "input": ["hello"],
            "input_type": "query",
            "encoding_format": "float",
            "truncate": "END",
            "dimensions": dim_requested,
        }
        t0 = time.monotonic()
        try:
            resp = httpx.post(
                f"{BASE_URL}/embeddings",
                json=dim_payload,
                headers=_auth_headers(),
                timeout=TIMEOUT,
            )
            latency = int((time.monotonic() - t0) * 1000)
            status = resp.status_code
            if status == 200:
                data = resp.json()
                new_dim = len(data["data"][0]["embedding"]) if data.get("data") else None
                dim_honored = (new_dim == dim_requested)
            else:
                dim_honored = False
                error = f"HTTP {status}: {resp.text[:300]}"
        except Exception as e:
            dim_honored = False
            error = f"{error or ''} | dim test: {str(e)[:200]}"

    return EmbedResult(
        model=model,
        status_code=status,
        dim_returned=dim_returned,
        dim_requested=dim_requested,
        dim_honored=dim_honored,
        error=error,
    )


def print_chat_table(results: list[ChatResult]) -> None:
    print("\n=== CHAT PROBE RESULTS ===")
    print(f"{'Model':<45} {'Status':>6} {'Latency':>8} {'Tools':>6} {'Error'}")
    print("-" * 100)
    for r in results:
        status_str = str(r.status_code) if r.status_code else "N/A"
        tool_str = "yes" if r.tool_calls_accepted else ("no" if r.status_code == 200 else "-")
        err = (r.error or "")[:40]
        print(f"{r.model:<45} {status_str:>6} {r.latency_ms:>6}ms {tool_str:>6} {err}")


def print_embed_table(results: list[EmbedResult]) -> None:
    print("\n=== EMBEDDING PROBE RESULTS ===")
    print(f"{'Model':<45} {'Status':>6} {'DimOut':>8} {'DimReq':>8} {'Honored':>8} {'Error'}")
    print("-" * 100)
    for r in results:
        status_str = str(r.status_code) if r.status_code else "N/A"
        dim_out = str(r.dim_returned) if r.dim_returned else "-"
        dim_req = str(r.dim_requested) if r.dim_requested else "-"
        honored = str(r.dim_honored) if r.dim_honored is not None else "-"
        err = (r.error or "")[:40]
        print(f"{r.model:<45} {status_str:>6} {dim_out:>8} {dim_req:>8} {honored:>8} {err}")


def main() -> None:
    print("=" * 100)
    print("NVIDIA PROBE — models, chat, tools, embeddings")
    print("=" * 100)

    available = fetch_models()
    print()

    chat_results: list[ChatResult] = []
    for model in CHAT_CANDIDATES:
        print(f"\n[chat] probing {model}...")
        in_list = "yes" if model in available else "not in /v1/models"
        r = probe_chat(model)
        r.error = f"{r.error or ''} [{in_list}]"
        chat_results.append(r)
        print(f"  -> status={r.status_code} latency={r.latency_ms}ms tools={r.tool_calls_accepted} error={r.error}")
        time.sleep(SLEEP_BETWEEN)

    embed_results: list[EmbedResult] = []
    for model in EMBEDDING_CANDIDATES:
        print(f"\n[embed] probing {model}...")
        r = probe_embedding(model)
        embed_results.append(r)
        print(f"  -> status={r.status_code} dim={r.dim_returned} honored={r.dim_honored} error={r.error}")
        time.sleep(SLEEP_BETWEEN)

    print_chat_table(chat_results)
    print_embed_table(embed_results)

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "available_models": available,
        "chat_results": [asdict(r) for r in chat_results],
        "embedding_results": [asdict(r) for r in embed_results],
    }
    out_path = os.path.join(os.path.dirname(__file__), "nvidia_probe_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_path}")

    # Summary verdict
    print("\n=== SUMMARY ===")
    chat_ok = [r for r in chat_results if r.status_code == 200]
    chat_404 = [r for r in chat_results if r.status_code == 404]
    embed_ok = [r for r in embed_results if r.status_code == 200]
    embed_410 = [r for r in embed_results if r.status_code == 410]
    print(f"Chat models that WORK:   {len(chat_ok)}/{len(chat_results)}")
    for r in chat_ok:
        print(f"  + {r.model}  (tools={r.tool_calls_accepted})")
    print(f"Chat models 404:         {len(chat_404)}")
    for r in chat_404:
        print(f"  - {r.model}")
    print(f"Embedding models that WORK: {len(embed_ok)}/{len(embed_results)}")
    for r in embed_ok:
        print(f"  + {r.model}  (dim={r.dim_returned})")
    print(f"Embedding models 410:      {len(embed_410)}")
    for r in embed_410:
        print(f"  - {r.model}")

    if not chat_ok:
        print("\n*** CRITICAL: Every chat model returned non-200. ***")
        print("*** The NVIDIA org may lack the 'Public API Endpoints' entitlement. ***")
        print("*** Fix: request it on forums.developer.nvidia.com, ***")
        print("*** or generate a new key from build.nvidia.com under a different org. ***")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Additional probe for models that appeared in /v1/models but weren't in the
original candidate lists.  Focus: find a working embedding model + chat models
that actually accept tool calls.
"""
from __future__ import annotations

import json, os, sys, time
import httpx

BASE_URL = "https://integrate.api.nvidia.com/v1"
TIMEOUT = 30.0
SLEEP = 2.0

EXTRA_CHAT = [
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/nemotron-4-340b-instruct",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "mistralai/mistral-large",
    "databricks/dbrx-instruct",
]
EXTRA_EMBED = [
    "nvidia/embed-qa-4",
    "snowflake/arctic-embed-l",
    "nvidia/llama-3.2-nv-embedqa-1b-v1",
]

_TOOL = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "A no-op that always succeeds.",
        "parameters": {"type": "object", "properties": {}},
    },
}

def _key():
    k = os.environ.get("NVIDIA_API_KEY", "")
    if not k:
        from dotenv import load_dotenv
        load_dotenv("c:/sandbox/market-place/Agent/.env")
        k = os.environ.get("NVIDIA_API_KEY", "")
    if not k:
        print("ERROR: NVIDIA_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    return k

def _hdr():
    return {"Authorization": f"Bearer {_key()}", "Content-Type": "application/json", "Accept": "application/json"}

def chat(model: str) -> dict:
    """Return details about chat + tool call attempt."""
    r = {"model": model, "basic_status": None, "tool_status": None, "tool_calls": False, "error": None}
    payload = {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 8, "stream": False}
    try:
        resp = httpx.post(f"{BASE_URL}/chat/completions", json=payload, headers=_hdr(), timeout=TIMEOUT)
        r["basic_status"] = resp.status_code
        if resp.status_code != 200:
            r["error"] = f"basic: HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        r["error"] = f"basic: {type(e).__name__}"
        return r

    if r["basic_status"] == 200:
        time.sleep(SLEEP)
        tool_payload = {"model": model, "messages": [{"role": "user", "content": "Call the ping tool."}], "max_tokens": 8, "stream": False, "tools": [_TOOL], "tool_choice": "required"}
        try:
            resp = httpx.post(f"{BASE_URL}/chat/completions", json=tool_payload, headers=_hdr(), timeout=TIMEOUT)
            r["tool_status"] = resp.status_code
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                msg = choices[0].get("message", {}) if choices else {}
                tc = msg.get("tool_calls")
                r["tool_calls"] = bool(tc)
                if not tc:
                    r["tool_response"] = msg.get("content", "")[:200]
            else:
                r["error"] = f"tool: HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            r["error"] = f"tool: {type(e).__name__}"
    return r

def embed(model: str) -> dict:
    r = {"model": model, "status": None, "dim": None, "error": None}
    payload = {"model": model, "input": ["hello"], "input_type": "query", "encoding_format": "float", "truncate": "END"}
    try:
        resp = httpx.post(f"{BASE_URL}/embeddings", json=payload, headers=_hdr(), timeout=TIMEOUT)
        r["status"] = resp.status_code
        if resp.status_code == 200:
            data = resp.json()
            if data.get("data"):
                r["dim"] = len(data["data"][0]["embedding"])
        else:
            r["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return r

def main():
    print("=== EXTRA CHAT PROBES ===")
    for m in EXTRA_CHAT:
        print(f"\n[chat] {m}")
        r = chat(m)
        print(f"  basic={r.get('basic_status')} tool={r.get('tool_status')} tool_calls={r.get('tool_calls')} err={r.get('error')}")
        if r.get("tool_response"):
            print(f"  model text-responded (no tool): {r['tool_response'][:100]}")
        time.sleep(SLEEP)

    print("\n=== EXTRA EMBEDDING PROBES ===")
    for m in EXTRA_EMBED:
        print(f"\n[embed] {m}")
        r = embed(m)
        print(f"  status={r['status']} dim={r['dim']} err={r.get('error')}")
        time.sleep(SLEEP)

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "extra_chat": [], "extra_embed": []}
    print("\nDone.")


if __name__ == "__main__":
    main()

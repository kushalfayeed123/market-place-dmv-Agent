#!/usr/bin/env python3
"""Probe nvidia/nemotron-3-embed-1b for dimension support + input_type."""
from __future__ import annotations
import os, time, httpx
BASE = "https://integrate.api.nvidia.com/v1"
SLEEP = 2.0
MODEL = "nvidia/nemotron-3-embed-1b"
def key():
    k = os.environ.get("NVIDIA_API_KEY","")
    if not k:
        from dotenv import load_dotenv
        load_dotenv("c:/sandbox/market-place/Agent/.env")
        k = os.environ.get("NVIDIA_API_KEY","")
    return k
def hdr():
    return {"Authorization": f"Bearer {key()}", "Content-Type":"application/json","Accept":"application/json"}

def test_embed(input_type, dimensions=None):
    payload = {"model": MODEL, "input": ["hello", "world"], "input_type": input_type, "encoding_format": "float", "truncate": "END"}
    if dimensions:
        payload["dimensions"] = dimensions
    try:
        r = httpx.post(f"{BASE}/embeddings", json=payload, headers=hdr(), timeout=30)
        print(f"  input_type={input_type} dims={dimensions}: status={r.status_code}", end="")
        if r.status_code == 200:
            data = r.json()
            vec = data["data"][0]["embedding"]
            print(f" dim={len(vec)}")
        else:
            print(f" body={r.text[:200]}")
    except Exception as e:
        print(f"  ERROR: {e}")
    time.sleep(SLEEP)

print(f"=== {MODEL} dimension tests ===")
test_embed("query")
test_embed("passage")
test_embed("query", 1024)
test_embed("query", 768)
test_embed("query", 512)
test_embed("query", 384)
test_embed("query", 256)  # below Matryoshka minimum?
print("\nDone.")

if __name__ == "__main__":
    pass  # calls are at module level above


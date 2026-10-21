#!/usr/bin/env python3
"""Quick syntax check for all modified files."""
import ast, sys

files = [
    "gateway/model_router.py",
    "gateway/providers/nvidia_backend.py",
    "mcp_server/knowledge/config.py",
    "mcp_server/knowledge/embedders.py",
    "mcp_server/vector_store.py",
    "mcp_server/knowledge/service.py",
    "mcp_server/knowledge/indexer.py",
]

base = "c:/sandbox/market-place/Agent/"
all_ok = True
for f in files:
    path = base + f
    try:
        ast.parse(open(path).read())
        print(f"OK: {f}")
    except SyntaxError as e:
        print(f"FAIL: {f} - {e}")
        all_ok = False

if not all_ok:
    sys.exit(1)
print("All files OK!")

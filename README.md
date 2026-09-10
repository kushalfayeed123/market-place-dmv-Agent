# Agent Layer — Marketplace Platform

The AI agent layer sits on top of the backend. It reads the OpenAPI docs and calls
available endpoints. It serves as an autonomous operating system that infers user
intentions and carries out actions to achieve user goals.

## Architecture

```
Browser ──SSE──► Agent Gateway ──MCP──► MCP Server ──HTTPS+JWT──► Backend
                  (FastAPI,              (thin typed      (AI-unaware,
                   session, cart,        HTTP wrappers,    unchanged)
                   idempotency,          auth, guard,
                   confirmation          role filter)
                   tokens)
```

## Key Design Principles

1. **Backend is never modified to be AI-aware.** The agent is an additive layer.
2. **The real authorization boundary stays in the backend.** RBAC, rate limits, and
   idempotency in the backend are the actual containment.
3. **The model never sees credentials.** Tokens live only in the MCP server's
   execution context.
4. **Confirmation is enforced server-side.** A model cannot skip the confirmation
   step for financial actions — the MCP server refuses the call without a token
   that only a genuine user click can produce.
5. **Idempotency keys are derived server-side.** The model never generates or
   manages them.

## Tech Stack

- **Language:** Python 3.12 (same as backend)
- **Package manager:** uv
- **Gateway:** FastAPI with SSE streaming
- **MCP Server:** mcp SDK
- **Model provider:** NVIDIA (default), swappable to Anthropic
- **Session store:** Redis (same Upstash instance as backend, namespaced)
- **Schemas:** Pydantic v2 (reuses backend schemas via `backend_schemas.py`)

## Quick Start

```bash
cd Agent
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your values

# Run tests
uv run pytest -q

# Run gateway locally
PYTHONPATH=../backend python -m gateway.main
```

## Project Structure

```
Agent/
├── docs/                          # Documentation
│   ├── backend_gap_analysis.md    # Part A: backend gaps to resolve
│   ├── agent_development_plan.md  # Part B: this dev plan
│   └── ci_cd.md                   # CI/CD guide (Phase 6)
├── schemas/                       # Canonical view DTOs (= frontend Zod)
├── mcp_server/                    # MCP Server (typed HTTP wrappers)
│   ├── tools_catalog.py           # Tool metadata + risk tiers
│   ├── guards.py                  # Confirmation gate
│   ├── idempotency.py             # Idempotency key derivation
│   ├── backend_client.py          # Backend HTTP client
│   ├── server.py                  # MCP server factory
│   └── tools/                     # Tool implementations by domain
├── gateway/                       # Agent Gateway (FastAPI)
│   ├── main.py                    # FastAPI app + SSE endpoints
│   ├── session.py                 # Session store (Redis)
│   ├── agent.py                   # Turn loop
│   ├── debug.py                   # Debug tracer
│   ├── provider.py                # ModelBackend port
│   └── providers/                 # NVIDIA + Anthropic backends
├── prompts/                       # System prompt + intent classifier
├── tests/                         # All tests (Phase 5)
├── backend_schemas.py             # Re-export of backend Pydantic schemas
├── pyproject.toml
└── .env.example
```

## Debug/Internal Processing

The agent exposes its internal processing for debugging:

- `GET /debug/sessions/{id}/trace` — full structured trace (postmortem)
- `GET /debug/sessions/{id}/events` — live SSE stream of agent events

Events include: turn start, intent classification, tool selection, idempotency
key derivation, backend calls, confirmation gate results, directive emission,
model usage, and errors.

**Safety:** Debug traces never contain access tokens, card data, or full PII.

## Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | ✅ | Scaffold — project structure, env, deps |
| 1 | ✅ | MCP Server — tools, guards, idempotency, backend client |
| 2 | ✅ | Agent Gateway — SSE, session, model-agnostic provider |
| 3 | ✅ | Debug/observability — DebugTracer + SSE debug channel |
| 4 | ✅ | Frontend contract — view DTOs = component prop schemas |
| 5 | ✅ | Tests — contract, security, adversarial, auth, directives, fallback |
| 6 | ✅ | CI/CD doc — deployment guide for human operator |

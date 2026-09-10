# CI/CD Guide — Agent Layer

**For:** Human operator to follow manually.
**Scope:** Build, test, and deploy the Agent layer (MCP Server + Agent Gateway).

---

## 1. Prerequisites

Before starting, ensure you have:

- Python 3.12+ installed
- `uv` package manager (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Redis running locally (or Upstash Redis URL for production)
- Backend running locally at `http://localhost:8000` (for integration tests)
- NVIDIA API key (or Anthropic key if using that provider)
- Render account (for deployment)

---

## 2. Local Development Setup

```bash
# Navigate to the Agent directory
cd Agent

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Copy environment file and fill in your values
cp .env.example .env
# Edit .env with your NVIDIA_API_KEY, BACKEND_API_URL, REDIS_URL, etc.

# Verify setup
python -c "from gateway.main import create_app; print('OK')"
```

---

## 3. Running Tests

```bash
# Run all tests
uv run pytest -q

# Run specific test categories
uv run pytest tests/contract/ -v      # Tool contract tests
uv run pytest tests/security/ -v      # Confirmation gate + idempotency
uv run pytest tests/adversarial/ -v    # Prompt injection tests
uv run pytest tests/auth/ -v           # Anonymous session tests
uv run pytest tests/directives/ -v     # Directive safety tests
uv run pytest tests/fallback/ -v       # Classic fallback tests
```

### Test Requirements

- **Unit tests** (contract, idempotency, directives): No external services needed.
- **Security tests** (confirmation gate): Require Redis on `localhost:6379/15`.
- **Auth tests**: No external services needed.
- **Adversarial tests**: No external services needed.
- **Fallback tests**: Require the `backend/` directory to exist.

---

## 4. Running the Gateway Locally

```bash
# Terminal 1: Start the Agent Gateway
cd Agent
PYTHONPATH=../backend python -m gateway.main

# Health check: curl http://localhost:8001/health

# Terminal 2: Start the MCP Server (if running separately)
cd Agent
PYTHONPATH=../backend python -m mcp_server.main
```

### Testing the SSE Stream

The `/sse` endpoint accepts a JSON body with the following fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | The user's message to the agent |
| `session_id` | string | No | An existing session ID to resume a conversation. If omitted or invalid, a new session is created. |
| `confirmed_token` | string | No | A token from a prior `user_confirmed` directive, used to authorize a destructive action (e.g., checkout). |

**Minimal request (new session):**

```bash
curl -X POST http://localhost:8001/sse \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me products"}'
```

**Resume an existing session:**

```bash
curl -X POST http://localhost:8001/sse \
  -H "Content-Type: application/json" \
  -d '{"message": "Add the first one to my cart", "session_id": "abc123"}'
```

**Submit a confirmed action (e.g., after user clicks confirm button):**

```bash
curl -X POST http://localhost:8001/sse \
  -H "Content-Type: application/json" \
  -d '{"message": "Yes, proceed", "session_id": "abc123", "confirmed_token": "tok_abc123"}'
```

### Testing Debug Endpoints

```bash
curl http://localhost:8001/debug/sessions/{session_id}/trace
curl http://localhost:8001/debug/sessions/{session_id}/events
```

## 6. Deployment to Render

### Option A: Single Service (Free Tier)

1. **Create Web Service**
   - Connect your repository
   - Root directory: `Agent`
   - Build command: `uv pip install -e .`
   - Start command: `PYTHONPATH=../backend python -m gateway.main`
   - Environment: Python 3.12

2. **Environment Variables**
   ```
   AGENT_MODEL_PROVIDER=nvidia
   AGENT_MODEL=nvidia/llama-3.1-nemotron-70b-instruct
   NVIDIA_API_KEY=your_key_here
   BACKEND_API_URL=https://your-backend.onrender.com
   REDIS_URL=redis://your-upstash-url
   AGENT_DEBUG=0
   CONFIRMED_TOKEN_TTL=300
   ```

3. **Health Check Path:** `/health`

### Option B: Two Services (Recommended)

**Service 1: Agent Gateway** (port 8001)
- Start: `PYTHONPATH=../backend python -m gateway.main`

**Service 2: MCP Server** (port 8002)
- Start: `PYTHONPATH=../backend python -m mcp_server.main`

---

## 7. CI Pipeline (GitHub Actions)

Create `.github/workflows/agent-ci.yml`:

```yaml
name: Agent CI

on:
  push:
    paths: ['Agent/**']
  pull_request:
    paths: ['Agent/**']

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
        ports: [6379:6379]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with: {python-version: '3.12'}

      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Install dependencies
        working-directory: ./Agent
        run: |
          uv venv
          uv pip install -e ".[dev]"

      - name: Run tests
        working-directory: ./Agent
        run: uv run pytest -q
        env:
          REDIS_URL: redis://localhost:6379/15
          BACKEND_API_URL: http://localhost:8000
          NVIDIA_API_KEY: ${{ secrets.NVIDIA_API_KEY }}
```

---

## 8. Monitoring & Observability

### Structured Logs

The agent emits JSON logs with: `timestamp`, `session_id`, `turn_id`, `correlation_id`, `event_type`.

### Key Metrics

- Tool call latency
- Confirmation rejection rate
- Idempotency replay rate
- Error rate by tool
- Session length (turns per session)

### Alerting

- Confirmation gate failures (potential attack)
- Backend error rate > 5%
- Redis connection failures
- Model API timeouts

---

## 9. Rollback Procedure

1. **Immediate**: Set `AGENT_DEBUG=1` and check `/debug/sessions/{id}/trace`
2. **Quick fix**: Revert to previous Render deploy
3. **Full rollback**: Classic frontend continues working
4. **Data safety**: No agent-layer data is critical; sessions are ephemeral

---

## 10. Security Checklist

- [ ] API keys in Render env vars, not in code
- [ ] `AGENT_DEBUG=0` in production
- [ ] Redis credentials secure
- [ ] CORS restricted to frontend origin
- [ ] Rate limiting enabled on backend
- [ ] Confirmation gate enforced
- [ ] No tokens in debug traces

---

## 11. Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: app.schemas` | Set `PYTHONPATH=../backend` |
| Redis connection refused | Start Redis or check `REDIS_URL` |
| NVIDIA API 401 | Check `NVIDIA_API_KEY` |
| Tests fail on gate | Ensure Redis on `localhost:6379/15` |
| SSE stream hangs | Check backend is reachable |
| Debug endpoints 403 | Set `AGENT_DEBUG=1` |

---

## 12. Future Improvements

- Prometheus metrics endpoint
- OpenTelemetry tracing
- Conversation summarization for long sessions
- A/B testing framework for model providers
- Tool call caching for read-only tools
- Agent-layer rate limiting (in addition to backend)

---

## 5. Pre-Deployment Checklist

- [ ] All tests pass: `uv run pytest -q`
- [ ] `.env` configured with production values (never commit `.env`)
- [ ] `AGENT_DEBUG=0` in production
- [ ] Redis URL points to production Upstash instance
- [ ] Backend API URL points to production backend
- [ ] NVIDIA_API_KEY is set
- [ ] `CONFIRMED_TOKEN_TTL=300`

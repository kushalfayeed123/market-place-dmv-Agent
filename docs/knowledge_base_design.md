# Knowledge Base & Support — Design & Development Plan (Agent Layer)

**Status:** Approved (decisions locked with product owner)
**Companion to:** `agent_mcp_system_design.md`, `../backend_gap_analysis.md`

## 0. Locked Decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Storage | Durable, human-facing state (KB source documents + support tickets) lives in the **backend relational DB** as plain, AI-unaware modules. The agent owns derived artifacts (chunks, embeddings, vector index in Redis). Redis-only was explicitly rejected. |
| 2 | Embeddings | **NVIDIA NIM embeddings now** (`nvidia/nv-embedqa-e5-v5`, 1024-dim), behind a pluggable `Embedder` port. TF-IDF remains a dev/test fallback. Index namespace carries model+dimension so a model change is a build-new-then-swap reindex. |
| 3 | Ticket flow | **KB-first with triage fallback.** Answer from the KB when possible; otherwise run structured triage, show a summary, get explicit user agreement, then create the ticket. |
| 4 | Human pickup | **Backend admin API + notifications.** `platform_admin`-scoped REST endpoints; webhook/email notification on ticket create and status change. No admin UI this phase. |

## 1. Current State (verified)

| Component | Where | State |
|-----------|-------|-------|
| Vector store | `Agent/mcp_server/vector_store.py` | Redis hashes + full-scan cosine in Python; no metadata filters, no KNN index |
| Embedder | `TfidfEmbedder` (256-dim, refit per restart); unused `OpenAIEmbedder` | Not semantic |
| Corpus | `Agent/mcp_server/ingest_vectors.py` — 5 hardcoded `SEED_DOCS` + shallow product scrape, manual CLI | Not organization-rooted |
| KB tool | `semantic_search` (`tools/knowledge.py`, `ToolMeta(method="VECTOR")`) | **Broken in live path** — `Agent._execute_tool` has no `VECTOR` branch; would call `BackendClient.call("VECTOR", "")` and fail |
| Support tickets | Nothing anywhere | Absent |
| KB source of truth | None — policies are Python string literals | Absent |

## 2. Gap Analysis → Bridge

| Gap | Bridge |
|-----|--------|
| G1 KB unreachable at runtime | `VECTOR` branch in `Agent._execute_tool`; KB service injected into the gateway Agent; MCP registration kept in parity |
| G2 Corpus not org-rooted | Backend `knowledge_documents` registry + collectors for published docs and catalog |
| G3 No policy source of truth | Versioned `knowledge_documents` table + admin CRUD (`platform_admin`) |
| G4 No support tickets | Backend `support_tickets`/`support_ticket_events` tables, REST, RBAC, notifications; agent triage + ticket tools |
| G5 Retrieval quality | NVIDIA embeddings via port; versioned index; score threshold; keyword fallback |
| G6 No freshness | Checksummed chunk IDs, per-source checkpoints, incremental + nightly sync |
| G7 No KB answer rails | Grounding/citation rules in prompts; anti-injection scrub; PII ban on corpus |
| G8 No human pickup | Admin ticket endpoints + signed webhook/email notifications |
| G9 No evaluation | Golden Q→doc set; recall@k gate in CI with deterministic fake embedder |

## 3. Architecture

```
                ┌──────────────────────── BACKEND (AI-unaware) ────────────────────────┐
                │  knowledge module                support module                       │
                │  ├─ knowledge_documents          ├─ support_tickets                   │
                │  ├─ admin CRUD (platform_admin)  ├─ support_ticket_events (append-only)│
                │  ├─ public read (published)      ├─ REST (create/own, admin pick-up)  │
                │  └─ (existing catalog/orders/…)  └─ notify worker → signed webhook/email│
                └───────────────▲──────────────────────────────▲───────────────────────┘
                                │ HTTPS (public/API)           │ HTTPS (user token)
                ┌───────────────┴──────────────── AGENT LAYER ─┴───────────────────────┐
                │  Ingestion pipeline (schedule + CLI)    Runtime (turn loop)           │
                │  collectors → chunker → indexer         search_knowledge_base (VECTOR)│
                │  sync state/checkpoints (Redis)         get_knowledge_document        │
                │  Embedder port (NVIDIA / TF-IDF / fake) triage + ticket tools         │
                │         Vector index (Redis, versioned ns — derived, rebuildable)    │
                └───────────────────────────────────────────────────────────────────────┘
```

**Data ownership principle:** durable, human-facing state in the backend; derived index in the agent. The vector index is a rebuildable cache.

### 3.1 Backend — `knowledge` module
`knowledge_documents`: `doc_type` (policy|faq|guide|announcement), `title`, `body` (markdown), `topic`, `audience` (buyer|merchant|all), `status` (draft|published|archived), `version`, `effective_from/to`, `created_by`.

- `POST/PATCH/DELETE /knowledge/documents` — `platform_admin` only. PATCH bumps `version` when title/body change. DELETE soft-archives.
- `GET /knowledge/documents`, `GET /knowledge/documents/{id}` — **published only** (customer-facing, same trust level as the public catalog).

### 3.2 Backend — `support` module
`support_tickets`: `ticket_number` (`TCK-YYYYMMDD-XXXXXXXX`), `status` (new→triaged→in_progress→resolved→closed, reopened), `priority` (low|medium|high|urgent), `category` (order_issue|payment|refund|product|merchant|account|other), `subject`, `description`, `requester_user_id?`, `contact_email?` (mandatory for anonymous), `order_id?`, `payment_id?`, `merchant_id?`, `session_id?`, `correlation_id?`, `assigned_to?`, `resolution_notes?`.
`support_ticket_events`: append-only audit trail (created | status_changed | assigned | note), actor = agent|requester|admin|system.

- `POST /support/tickets` — authenticated users (requester attached server-side) **and anonymous** (mandatory contact email, per-IP rate limited, Idempotency-Key supported).
- `GET /support/tickets/mine`, `GET /support/tickets/{id}` — requester-scoped.
- `GET /support/tickets?status=&category=&priority=`, `PATCH /support/tickets/{id}` — `platform_admin` only.
- Notifications: on create and on requester-visible status change, a `webhook_events` row (`provider="support"`) is written in the same transaction; `app/workers/support_notify_worker.py` delivers signed webhooks (HMAC-SHA256, `SUPPORT_WEBHOOK_SECRET`) with retry/backoff, idempotent by `(provider, event_id)`.

### 3.3 Agent — knowledge pipeline (`Agent/mcp_server/knowledge/`)

```
mcp_server/knowledge/
├── config.py            # env-driven settings
├── embedders.py         # Embedder port + NvidiaEmbedder + TfidfEmbedder + FakeEmbedder (tests)
├── collectors/          # base protocol; documents (published KB docs); catalog (products/categories)
├── chunking.py          # heading-aware ~512-token chunks, overlap, deterministic IDs
├── indexer.py           # batch embed → upsert/delete; checksum change detection
├── sync_state.py        # per-source checkpoints + index version (Redis)
├── sync.py              # full|incremental orchestration + CLI
└── service.py           # query side: hybrid search, thresholds, filters, citations
```

- Chunk IDs: `{source}:{doc_id}#chunk{n:04d}`; content checksum in metadata → idempotent re-runs; changed/archived sources deleted by prefix.
- Index namespace: `{ns}:kb:v{model}:{dim}[-{KB_INDEX_VERSION}]` — embedding-model change never corrupts a live index.
- Search: vector top-k + lexical-overlap fallback, weighted merge; **minimum score threshold** → "no confident match" (triggers ticket path instead of hallucination); results carry provenance (`title`, `version`, `doc_id`).
- Runtime: `gateway/main.py` builds the KB service; `Agent._execute_tool` handles `method == "VECTOR"`; empty/unavailable index → typed `kb_unavailable` fallback (never crashes a turn).
- Replaces `ingest_vectors.py`; `SEED_DOCS` migrate into `knowledge_documents` as the first content drop, then the literals are deleted.

### 3.4 Agent — support tools & prompts
Tools (risk tier LOW — no money moves; triage summary = conversational confirmation):

| Tool | Type |
|------|------|
| `search_knowledge_base(query, topic?, kind?, top_k?)` | VECTOR |
| `get_knowledge_document(doc_id)` | READ |
| `create_support_ticket(subject, description, category, priority, order_id?, payment_id?, merchant_id?, contact_email?)` | MUTATING |
| `get_ticket_status(ticket_number)` / `list_my_tickets()` | READ |

Prompts: `support` intent added to classifier; `system.md` gains "Support & Knowledge" grounding rules (cite title+version; personal order/payment questions → live tools, never KB; below-threshold → triage; explicit user agreement before `create_support_ticket`). `TicketCreatedView` DTO + allowlist component added (frontend rendering out of scope this phase).

### 3.5 Security
- No PII in the corpus (personal data answered via authenticated live tools; `agent_mcp_system_design.md §10.2` preserved).
- Agent → backend uses the same authenticated-call discipline (correlation IDs propagated to tickets).
- Anti-injection scrub on indexed content; product descriptions remain untrusted data.
- Anonymous ticket spam: per-IP rate limit (existing middleware) + per-session open-ticket cap; webhooks signed.
- Debug traces never contain PII (existing rule).

### 3.6 Observability & evaluation
- Sync metrics (sources seen/changed/failed, chunks up/deleted, index version, `kb_sync_age` on health).
- Retrieval telemetry: query → scores → resolved-by-KB vs escalated-to-ticket (content-gap signal).
- Golden-set recall@k gate in CI with deterministic fake embedder (no API cost).

## 4. Phases & Acceptance

| Phase | Scope | Acceptance |
|-------|-------|------------|
| 0 Foundations | Design doc, config, embedder port, `.env.example` | Unit tests for embedder port green |
| 1 Backend modules | knowledge + support models/schemas/routers/services, migration, notify worker, tests | Migration up/down clean; RBAC enforced; webhook event written on create |
| 2 Ingestion | Collectors, chunker, indexer, sync CLI + schedule, seed-doc migration script | Idempotent re-run; incremental picks up edits; archived docs removed |
| 3 Runtime KB | VECTOR branch, KB service injection, KB tools, kb_unavailable fallback | Policy question answered with citation live; KB-down graceful |
| 4 Ticket flow | Support tools, support intent, prompt rules, DTO | E2E fake-model test: unanswerable → triage → confirmed → ticket in DB |
| 5 Tests | Unit/contract/security/E2E + golden set | `uv run pytest -q` green without live model/API |
| 6 Rollout | Deploy order backend → migration → agent; content drop; reindex; webhook config; runbook | Health shows fresh `kb_sync_age`; real ticket picked up via admin API |

## 5. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Upstash Redis lacks RediSearch | Exact cosine fine to ~tens of thousands of chunks; documented threshold to move to Redis Stack KNN; never block the turn on index errors |
| Embedding API limits/latency | Batching, retries, sync is offline; query-time = one embed call |
| Model/dimension change | Version-scoped namespace; build-new-then-swap reindex |
| Stale/wrong answers | Published-only ingestion, versioned citations, score threshold, ticket fallback, golden-set CI |
| Anonymous ticket spam | Rate limit, per-session cap, signed webhooks |
| Backend backlog coupling | Collectors behind a protocol; agent can develop against a stub endpoint |

## 6. Assumptions
- Admins author initial policy/FAQ content in `knowledge_documents` (existing seed text migrates in).
- Anonymous ticket creation allowed (mandatory contact email + rate limits). One-line policy change if sign-in is later required.
- No frontend ticket rendering this phase; DTO defined for later mirroring.




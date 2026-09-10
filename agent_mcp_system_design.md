# AI Agent & MCP Server Design — Marketplace Platform

**Companion to:** `marketplace_system_design.md`, `backend_system_design.md`, `frontend_system_design.md`

**Core constraint driving this whole document:** the backend (from `backend_system_design.md`) is **never modified to be AI-aware**. It stays a plain, RBAC-enforced, rate-limited, idempotent REST API that would work perfectly well with a conventional frontend and no LLM at all. Everything here is an **additive layer on top** — an agent and an MCP server that talk to the backend using the exact same authenticated HTTP calls a human-driven frontend would make. If this whole layer were deleted, the backend and a classic frontend would still be a fully working marketplace.

## 1. Architecture Overview

The agent layer follows a **Retrieval-Augmented Generation (RAG)** pattern to provide knowledge-based assistance while maintaining the backward-compatible design principle.

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Interaction                           │
│  (natural language requests → agent → tool calls → backend API)   │
└───────────────────────────────────────┬─────────────────────────┘
                                        │
                              ┌────────────────────┐
                              │   MCP Server       │
                              │  (FastMCP + tools) │
                              └───────┬────────────┘
                                      │
                              ┌────────────────────┐
                              │   Agent Gateway    │
                              │  (session + context)│
                              └───────┬────────────┘
                                      │
                              ┌────────────────────┐
                              │   Knowledge Base   │
                              │   (Redis Vector +   │
                              │    TF-IDF embedder) │
                              └───────┬────────────┘
                                      │
                              ┌────────────────────┐
                              │   Tool Modules     │
                              │   (catalog, cart,  │
                              │    orders, payments)│
                              └────────────────────┘
```

## 2. RAG Pipeline Workflow

The agent follows a structured RAG pipeline for knowledge-based queries:

### 2.1 Query Processing
1. **User Input**: Natural language request from the user
2. **Intent Classification**: Determine if the request requires:
   - Pure tool execution (e.g., "add product to cart")
   - Knowledge-based tool execution (e.g., "find products in category X")
   - Pure knowledge query (e.g., "what's our refund policy")
   - Hybrid (e.g., "find cheap red shirts and add to cart")

3. **Knowledge Retrieval** (when needed):
   - Extract key entities (product names, categories, prices, etc.)
   - Perform semantic search against the vector store
   - Return top-k relevant documents with similarity scores

### 2.2 Context Enhancement
- Retrieved knowledge is injected as context into the tool call parameters
- Ensures tool calls are informed by the latest product information, policies, etc.
- Maintains separation between knowledge data and backend data

### 2.3 Tool Execution
- Tools proceed as normal, but with enhanced context from the RAG step
- All tool calls still go through the same authenticated backend API
- No backend modifications needed - the agent layer handles all knowledge integration

### 2.4 Response Synthesis
- Tool results are combined with knowledge base information
- Response is generated that's both data-driven (from backend) and context-aware (from KB)
- Maintains the principle that frontend renders backend-confirmed state, not agent-asserted state

## 3. Knowledge Base Integration

### 3.1 Vector Store Setup
The knowledge base uses Redis with TF-IDF embedding for semantic search:

```python
from mcp_server.vector_store import VectorStore, TfidfEmbedder

vector_store = VectorStore(
    redis_client=redis_client, 
    embedder=TfidfEmbedder(), 
    namespace="agent"
)
```

### 3.2 Knowledge Tools
New tools registered for knowledge queries:

```python
@server.tool(
    name="semantic_search",
    description="Search the knowledge base semantically for products, policies, or FAQs. Returns relevant documents ranked by similarity.",
)
async def semantic_search(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    results = await vector_store.search(query, top_k=top_k)
    return [{"text": r["text"], "score": r["score"], "metadata": r.get("metadata", {})} for r in results]
```

### 3.3 FAQ Integration
FAQ entries are stored in the vector store with proper categorization:
- Product FAQs: linked to specific products/variants
- Policy FAQs: refunds, KYC, payouts, etc.
- General questions: onboarding, features, etc.

## 4. Agent Design Enhancements

### 4.1 Intent Aware Tool Registration
Tools are now tagged with their knowledge dependency level:
- **Level 0**: Pure tool execution (no knowledge needed)
- **Level 1**: Tool execution with KB context (e.g., search with filters)
- **Level 2**: Pure knowledge queries (e.g., "what's the return policy")
- **Level 3**: Hybrid queries (tool + knowledge)

### 4.2 Token Management for Knowledge
- Knowledge retrieval adds minimal overhead (vector search in Redis)
- Tokens are only consumed for actual tool calls, not knowledge queries
- Confirmed tokens from frontend interactions still apply to financial tools

### 4.3 Session Context Preservation
- Knowledge search results are scoped to the current session/turn
- Enables follow-up questions within the same context
- Turn counter still used for idempotency key derivation

## 5. Backend Integration (Unchanged)

The backend remains **completely unaware** of the agent layer:

- All tool calls use the same `BackendClient` with authenticated HTTP calls
- No new backend endpoints are required
- Idempotency, rate limiting, and RBAC all work as designed
- The agent layer is 100% additive - deleting it leaves a fully working marketplace

**Key design principle maintained**: `frontend_system_design.md` §6 - "State shown is always backend-confirmed, not agent-asserted." The agent never asserts payment success from its own reasoning - fresh tool call results always reflect the backend's webhook-verified state.

## 6. Tool Updates for RAG Support

### 6.1 Modified Tools
Several tools now optionally accept knowledge-enhanced parameters:

**catalog tools** - `search_products` can use KB-filtered results:
- Accepts optional `knowledge_context` parameter for product filter enhancements
- Search results can be supplemented with KB semantic search

**cart tools** - `add_to_cart` can reference KB product info:
- Validates product availability against KB stock data
- Provides better error messages with KB context

**orders tools** - `initiate_checkout` uses KB for:
- Cart validation with up-to-date product info
- Price confirmation from KB if backend is slow to respond

### 6.2 New Knowledge-Enhanced Tools
- `get_product_recommendations` - KB-filtered product suggestions
- `check_policy_compliance` - Verify cart/checkout against company policies from KB
- `get_merchant_context` - Get merchant-specific knowledge for admin tools

## 7. Error Handling & Fallbacks

### 7.1 Knowledge Search Failures
- If KB search fails or returns no results, tools proceed with default/backend behavior
- Fallback to backend-only behavior ensures no disruption
- Error messages are informative but don't expose KB internals

### 7.2 Tool Fallback Chain
1. Try knowledge-enhanced execution
2. If KB unavailable or error → try backend-only execution
3. If both fail → return appropriate error to user
4. Never silently degrade - user always gets explicit feedback

### 7.3 Token Expiry Handling
- Knowledge searches don't require confirmed tokens (read-only)
- Financial tools still require confirmation per `frontend_system_design.md` §3
- Token management unchanged from original design

## 8. Performance Considerations

### 8.1 Latency
- Vector search in Redis: ~10-50ms typical
- Cached results for frequently queried terms
- Async search doesn't block tool execution

### 8.2 Memory
- Vector store scoped per namespace ("agent")
- TTL-based cleanup for old entries
- Regular pruning of low-frequency queries

### 8.3 Scaling
- Redis vector store scales horizontally
- Multiple agent instances share same knowledge base
- No backend sharding required - all knowledge is in the agent layer

## 9. Migration Path from Existing Design

If upgrading from a version without RAG:

1. **Add vector store setup** to `gateway/session.py` or a new `knowledge.py`
2. **Register knowledge tools** in `server.py` `create_server()`
3. **Update tool modules** to optionally accept knowledge context
4. **Add intent classification** layer at the agent gateway level
5. **Test with KB-empty state** - ensure no regressions
6. **Gradually enable** knowledge enhancement on a per-tool basis

**Backward compatibility**: All existing tool calls work exactly as before when no knowledge context is provided. The RAG pipeline is opt-in per tool, not a requirement.

## 10. Security & Compliance

### 10.1 Knowledge Access Control
- Same token-based access control as all agent tools
- Knowledge namespace is isolated per session/user role
- No sensitive backend data exposed in knowledge base

### 10.2 Data Privacy
- KB contains only public product info, policies, FAQs
- No customer PII or merchant secrets in vector store
- Redis configuration follows same security as session store

### 10.3 Auditability
- All tool calls (knowledge-enhanced or not) are logged with correlation IDs
- Knowledge search results are traceable to the originating query
- Backend state always overrides agent assertions per design

---

## 11. Summary of Changes

| Area | Before | After |
|------|--------|-------|
| **Knowledge Integration** | None | RAG pipeline with vector store |
| **Tool Enhancement** | Backend-only | Optional KB context |
| **Intent Handling** | Simple dispatch | Intent-aware with KB |
| **Response Quality** | Backend data only | Backend + KB context |
| **Backend Impact** | N/A | None (additive layer) |
| **Backward Compatibility** | Full | Full (opt-in KB) |

**Core principle maintained**: The agent layer is an additive UX layer on top of the backend. All tool calls go through the same authenticated backend API. The backend remains fully functional without the agent layer. Knowledge integration happens entirely in the agent/MCP server layer.
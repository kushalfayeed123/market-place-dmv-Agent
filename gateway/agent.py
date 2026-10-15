"""Agent: the turn loop that orchestrates model calls, tool execution and directive emission."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from gateway.money import format_money, llm_view
from gateway.search_pipeline import search_catalog
from gateway.tokens import normalize_token

from gateway.debug import DebugTracer
from gateway.provider import ModelBackend
from gateway.session import Session
from mcp_server.backend_client import BackendClient
from mcp_server.guards import ConfirmationGate
from schemas.components import ComponentName, UIDirective

logger = logging.getLogger("gateway.agent")

# --------------------------------------------------------------------------------------
# Anonymous-user gating
#
# The model's intent classifier is slow (a whole extra LLM round trip) and unreliable, so
# deterministic rules decide first, and the model is asked only when no rule matches.
# Rules match WHOLE words/phrases: the old substring rules treated 'recorder' as 'order',
# 'display' ... and 'Update my cart' as an admin action, which sent shoppers to a sign-in
# prompt instead of searching.
# --------------------------------------------------------------------------------------


def _phrases(items: list[str]) -> re.Pattern[str]:
    parts = [re.escape(i).replace(r"\ ", r"\s+") for i in items]
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(parts) + r")(?![a-z0-9])", re.IGNORECASE)


_ANON_SAFE = _phrases([
    "show me", "show", "find", "search", "browse", "looking for", "what is", "what are", "what's", "whats",
    "how many", "how much", "how to", "how do", "how can", "how should", "price", "prices", "cost", "stock",
    "in stock", "available", "rating", "ratings", "review", "reviews", "detail", "details", "category",
    "categories", "info", "information", "describe", "tell me", "can i", "is there", "do you have",
    "where is", "where are", "product", "products", "catalog", "item", "items", "popular",
    "add to cart", "add to my cart", "view cart", "see my cart", "my cart", "clear cart", "empty cart",
    "remove from cart", "delete from cart", "remove cart item", "delete cart item",
])

# Financial, order and account actions: always require sign-in.
_AUTH_SPECIFIC = _phrases([
    "buy", "purchase", "checkout", "check out", "pay", "payment", "payments", "refund", "refunds", "payout",
    "payouts", "withdraw", "transfer money", "place an order", "place order", "my order", "my orders",
    "order status", "order number", "order #", "track my order", "track order", "where is my order",
    "cancel my order", "cancel order", "order history", "my profile", "my account", "sign up", "register",
    "onboard", "approve", "reject", "kyc", "verify kyc", "review kyc", "fulfill", "fulfillment", "ledger",
    "payout account",
])
# Generic verbs that are only admin actions when they are not about the cart.
_AUTH_GENERIC = _phrases(["ship", "create", "update", "delete", "cancel"])

_HOW_TO_PREFIXES = ("how to", "how do", "how can", "how should", "how much", "how many")
_CART_OP = re.compile(
    r"^\s*(?:please\s+)?(?:add|put|remove|delete|update|change|set|clear|empty|show|view|see)\b[^.]*"
    r"\b(?:cart|basket)\b",
    re.IGNORECASE,
)


def _is_anon_safe_query(message: str) -> bool:
    return bool(_ANON_SAFE.search(message.lower().strip()))


def _is_cart_operation(message: str) -> bool:
    """'Add X to my cart', 'Update my cart: ...', 'Show my cart': safe for anonymous users."""
    return bool(_CART_OP.search(message))


def _requires_auth_action(message: str) -> bool:
    """True when the message asks for something that needs a signed-in user."""
    m = message.lower().strip()
    if _is_cart_operation(m):
        return False
    return bool(_AUTH_SPECIFIC.search(m) or _AUTH_GENERIC.search(m))


def _is_how_to_query(message: str) -> bool:
    return message.lower().strip().startswith(_HOW_TO_PREFIXES)


# Questions a live catalog search cannot answer (they belong to the knowledge base).
_POLICY_QUERY = re.compile(r"\b(policy|policies|return|refund|shipping|delivery|payment method|faq|track|how (do|to|can))\b", re.I)

_MERCHANT_ROLES = {"merchant_owner", "merchant_staff", "platform_admin"}


class Agent:
    """The agent turn loop."""

    def __init__(
        self,
        backend_client: BackendClient,
        gate: ConfirmationGate,
        model_backend: ModelBackend,
        debug_tracer: DebugTracer,
        system_prompt: str,
        redis_client: Any | None = None,
        vector_namespace: str = "agent",
        vector_store: Any | None = None,
        knowledge_service: Any | None = None,
    ):
        self._client = backend_client
        self._gate = gate
        self._model = model_backend
        self._debug = debug_tracer
        self._system_prompt = system_prompt
        # KnowledgeService: the versioned Redis vector index. Used for policies, FAQs and
        # as a recall helper; product questions are always answered from the live catalog.
        self._knowledge_service = knowledge_service
        self._redis = redis_client
        self._vector_namespace = vector_namespace
        self._vector_store = vector_store

    # ------------------------------------------------------------------ turn loop

    async def run_turn(
        self,
        session: Session,
        user_message: str,
        confirmed_token: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run one turn: bounded, tool-aware loop (model -> tools -> model ...)."""
        turn_id = session.next_turn()
        session_id = session.session_id
        self._debug.log_turn_start(session_id, turn_id, user_message)

        intent, confidence = await self._decide_intent(user_message, session)
        self._debug.log_intent(session_id, turn_id, intent, confidence)

        if intent == "action" and not session.token:
            self._debug.log("auth.required", {"reason": "unauthenticated_action"}, session_id, turn_id)
            yield {
                "type": "ui_directive",
                "directive": UIDirective(
                    component=ComponentName.SignInPrompt,
                    props={"message": "Please sign in to perform this action."},
                    correlation_id=session_id,
                ).model_dump(),
            }
            return

        from mcp_server.tools_catalog import tools_for_role

        tool_defs = [_tool_to_openai_format(t) for t in tools_for_role(session.role)]
        history = [m for m in (getattr(session, "history", None) or []) if m.get("content")]
        messages: list[dict[str, Any]] = [*history[-8:], {"role": "user", "content": user_message}]

        self._debug.log_model_call(session_id, turn_id, self._model.provider_name, self._model.model_name)

        model_parts: list[str] = []
        tool_parts: list[str] = []
        try:
            max_iterations = 5
            iteration = 0
            executed_signatures: set[tuple[str, str]] = set()
            turn_summary_buffer: list[tuple[str, str, dict[str, Any], bool]] = []
            while True:
                iteration += 1
                if iteration > max_iterations:
                    self._debug.log_error(session_id, turn_id, "max_iterations", f"agent loop exceeded {max_iterations} iterations")
                    yield {"type": "error", "error": "The assistant took too many steps and stopped."}
                    for summary in _flush_summaries(turn_summary_buffer):
                        tool_parts.append(summary)
                        yield {"type": "text", "content": summary, "origin": "tool"}
                    return

                tool_calls_this_stream: list[dict[str, Any]] = []
                tool_results: list[dict[str, Any]] = []

                async for event in self._model.stream(
                    messages=messages, tools=tool_defs, system_prompt=self._system_prompt,
                ):
                    async for output in self._handle_event(
                        event, session, turn_id, confirmed_token,
                        tool_results, executed_signatures, turn_summary_buffer,
                    ):
                        if output.get("type") == "text":
                            (tool_parts if output.get("origin") == "tool" else model_parts).append(output.get("content", ""))
                        yield output
                    if event.get("type") == "tool_calls":
                        tool_calls_this_stream.extend(event.get("tool_calls", []))
                    elif event.get("type") == "tool_call":
                        single = event.get("tool_call")
                        if single:
                            tool_calls_this_stream.append(single)

                if any(r.get("status") == "confirmation_required" for r in tool_results):
                    for summary in _flush_summaries(turn_summary_buffer):
                        tool_parts.append(summary)
                        yield {"type": "text", "content": summary, "origin": "tool"}
                    return

                if tool_calls_this_stream and tool_results:
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": tc.get("name", "") or (tc.get("function") or {}).get("name", ""),
                                    "arguments": json.dumps(tc.get("arguments", {}) or _args_of(tc)),
                                },
                            }
                            for tc in tool_calls_this_stream
                        ],
                    })
                    for r in tool_results:
                        messages.append({"role": "tool", "tool_call_id": r["tool_call_id"], "content": r["content"]})
                    continue

                for summary in _flush_summaries(turn_summary_buffer):
                    tool_parts.append(summary)
                    yield {"type": "text", "content": summary, "origin": "tool"}
                return

        except Exception as e:
            logger.exception("agent turn failed session=%s turn=%s", session_id, turn_id)
            self._debug.log_error(session_id, turn_id, "agent_error", str(e))
            yield {"type": "error", "error": f"Agent error: {e!s}"}
        finally:
            remember = getattr(session, "append_exchange", None)
            if callable(remember):
                remember(user_message, ("".join(model_parts) or " ".join(tool_parts)).strip())

    # ------------------------------------------------------------------ intent

    _INTENT_CLASSIFICATION_PROMPT = (
        "You are an intent classifier for a marketplace assistant. "
        "Classify the user's message into exactly one of these intents:\n"
        "- 'info': The user is asking a question, browsing, searching, reading reviews, "
        "checking order status, or seeking information. No state will be modified.\n"
        "- 'action': The user wants to perform an operation that modifies state: "
        "buying, checking out, paying, placing an order, purchasing, requesting a refund or payout, "
        "creating/updating/deleting a resource, approving KYC, shipping, or fulfilling. "
        "Adding items to the cart or changing the cart is NOT an action here.\n\n"
        "Be precise: if the user is merely asking *about* an action (e.g. 'how do I refund?') "
        "without requesting it, classify as 'info'.\n\n"
        "Respond with JSON only: {\"intent\": \"info\"|\"action\", \"confidence\": 0.0-1.0, \"reasoning\": \"brief explanation\"}"
    )

    async def _decide_intent(self, message: str, session: Session) -> tuple[str, float]:
        """Signed-in users need no gate. Anonymous users: rules first, model only as a last resort."""
        if session.token:
            return "info", 1.0
        if _is_how_to_query(message) or _is_cart_operation(message):
            return "info", 1.0
        if _requires_auth_action(message):
            return "action", 1.0
        if _is_anon_safe_query(message):
            return "info", 1.0
        return await self._classify_intent(message, session)

    async def _classify_intent(self, message: str, session: Session) -> tuple[str, float]:
        try:
            result = await asyncio.wait_for(
                self._model.classify(
                    message=message,
                    system_prompt=self._INTENT_CLASSIFICATION_PROMPT,
                    valid_intents=["info", "action"],
                ),
                timeout=8.0,
            )
        except Exception as exc:  # a slow or failing classifier must never block a search
            logger.warning("intent classification failed, defaulting to info: %r", exc)
            return "info", 0.5
        if result.get("error"):
            self._debug.log_error(session.session_id, session.current_turn_id, "intent_classification_error", result["error"])
        return result.get("intent", "info"), result.get("confidence", 0.5)

    # ------------------------------------------------------------------ events

    async def _handle_event(
        self,
        event: dict[str, Any],
        session: Session,
        turn_id: int,
        confirmed_token: str | None,
        tool_results: list[dict[str, Any]],
        executed_signatures: set | None = None,
        summary_buffer: list[tuple[str, str, dict[str, Any], bool]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        event_type = event.get("type")
        session_id = session.session_id

        if event_type == "delta":
            yield {"type": "text", "content": event.get("content", "")}

        elif event_type in ("tool_calls", "tool_call"):
            if event_type == "tool_calls":
                tool_list = event.get("tool_calls") or []
            else:
                single = event.get("tool_call")
                tool_list = [single] if single else []
            if isinstance(tool_list, dict):
                tool_list = [tool_list]
            for tc in tool_list or []:
                fn = tc.get("function") if isinstance(tc, dict) else None
                if isinstance(fn, dict):
                    tool_name = fn.get("name", "")
                    raw_args = fn.get("arguments", {})
                    tool_call_id = tc.get("id", "")
                else:
                    tool_name = tc.get("name", "") if isinstance(tc, dict) else ""
                    raw_args = tc.get("arguments", {}) if isinstance(tc, dict) else {}
                    tool_call_id = tc.get("id", "") if isinstance(tc, dict) else ""
                if isinstance(raw_args, str):
                    try:
                        tool_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        tool_args = {}
                elif isinstance(raw_args, dict):
                    tool_args = dict(raw_args)
                else:
                    tool_args = {}
                sig = (tool_name, json.dumps(tool_args, sort_keys=True))
                if executed_signatures is not None and sig in executed_signatures:
                    self._debug.log("dedup.duplicate_tool_call", {"tool_name": tool_name, "args": _summarize_args(tool_args)}, session_id, turn_id)
                    continue
                if executed_signatures is not None:
                    executed_signatures.add(sig)
                self._debug.log_tool_selected(session_id, turn_id, tool_name, _summarize_args(tool_args))

                from mcp_server.tools_catalog import is_mutating_financial

                if is_mutating_financial(tool_name) and confirmed_token:
                    tool_args["user_confirmed_token"] = confirmed_token

                result = await self._execute_tool(tool_name, tool_args, session, turn_id)

                if result.get("status") == "confirmation_required":
                    self._debug.log_confirmation(session_id, turn_id, tool_name, True, False)
                    yield {
                        "type": "awaiting_confirmation",
                        "tool_name": tool_name,
                        "message": result.get("message", "Please confirm this action."),
                    }
                    yield {
                        "type": "ui_directive",
                        "directive": UIDirective(
                            component=ComponentName.ConfirmationDialog,
                            props={
                                "tool_name": tool_name,
                                "title": "Confirm to proceed",
                                "message": result.get("message", "Please confirm this action."),
                            },
                            correlation_id=session_id,
                        ).model_dump(),
                    }
                    tool_results.append({
                        "tool_call_id": tool_call_id,
                        "status": "confirmation_required",
                        "content": json.dumps({"status": "confirmation_required", "message": result.get("message", "")}),
                    })
                elif result.get("status") == "error":
                    message = result.get("message", "An error occurred.")
                    self._debug.log_error(session_id, turn_id, result.get("error_kind", "tool_error"), message)
                    yield {"type": "error", "error": message}
                    # The model must be told WHY the tool failed, otherwise it reports 'no results'.
                    tool_results.append({
                        "tool_call_id": tool_call_id,
                        "status": "error",
                        "content": json.dumps({
                            "status": "error",
                            "tool_name": tool_name,
                            "error_kind": result.get("error_kind", "unknown"),
                            "message": message,
                            "instruction": "Tell the user this action is temporarily unavailable. Do not say no products exist.",
                        }),
                    })
                else:
                    self._debug.log_tool_result(session_id, turn_id, tool_name, _summarize_result(result))
                    text_summary = _result_to_text(tool_name, result)
                    if text_summary:
                        if summary_buffer is not None:
                            summary_buffer.append((tool_name, text_summary, result, True))
                        else:
                            yield {"type": "text", "content": text_summary, "origin": "tool"}
                    directive = self._result_to_directive(tool_name, result, session_id)
                    if directive:
                        self._debug.log_directive(session_id, turn_id, directive["component"], directive.get("props", {}))
                        yield {"type": "ui_directive", "directive": directive}
                    tool_results.append({
                        "tool_call_id": tool_call_id,
                        "status": "success",
                        "content": json.dumps({
                            "status": "success",
                            "tool_name": tool_name,
                            "data": llm_view(_strip_sensitive(result.get("data", {}))),
                        }),
                    })

        elif event_type == "usage":
            self._debug.log_model_usage(session_id, turn_id, event.get("usage", {}))

        elif event_type == "finish":
            self._debug.log_turn_end(session_id, turn_id, event.get("reason", "unknown"))

        elif event_type == "error":
            self._debug.log_error(session_id, turn_id, "model_error", event.get("error", "Unknown"))
            yield {"type": "error", "error": event.get("error", "Model error")}

    # ------------------------------------------------------------------ tools

    async def _execute_tool(
        self, tool_name: str, args: dict[str, Any], session: Session, turn_id: int,
    ) -> dict[str, Any]:
        """Run one tool. `session.token` goes to the backend only as an Authorization header."""
        from mcp_server.backend_client import ToolError
        from mcp_server.tools_catalog import BY_NAME, is_mutating_financial

        meta = BY_NAME.get(tool_name)
        if not meta:
            if tool_name in {c.value for c in ComponentName}:
                return {
                    "status": "error",
                    "tool_name": tool_name,
                    "message": (
                        f"\"{tool_name}\" is a UI component, not a tool. Call a tool "
                        f"(e.g. search_products, get_product_detail) and the {tool_name} component is rendered automatically."
                    ),
                }
            return {"status": "error", "tool_name": tool_name, "message": f"Unknown tool: {tool_name}"}

        provided_token = args.get("user_confirmed_token")
        if is_mutating_financial(tool_name):
            if not provided_token:
                self._debug.log_confirmation(session.session_id, turn_id, tool_name, True, False)
                return {"status": "confirmation_required", "tool_name": tool_name,
                        "message": f"This action ({tool_name}) requires your confirmation."}
            try:
                await self._gate.check_confirmation(session.session_id, turn_id, tool_name, provided_token)
            except Exception as e:
                self._debug.log_confirmation(session.session_id, turn_id, tool_name, True, False)
                return {"status": "confirmation_required", "tool_name": tool_name,
                        "message": str(e), "error_kind": "confirmation_required"}
            self._debug.log_confirmation(session.session_id, turn_id, tool_name, True, True)

        reserved = {"token", "user_confirmed_token", "confirmed_token"}
        # Public catalog routes get no token: an expired token makes some backends reject a public call.
        token = None if meta.path.startswith("/catalog") else normalize_token(session.token)
        idempotency_key = f"{session.session_id}:{turn_id}:{tool_name}" if meta.method != "GET" else None
        if idempotency_key:
            self._debug.log_idempotency_key(session.session_id, turn_id, tool_name, idempotency_key)

        start = time.perf_counter()
        try:
            if tool_name == "search_products":
                data = await self._search_products(args, session)
            elif meta.method == "SESSION":
                if tool_name == "add_to_cart":
                    args = await self._enrich_add_to_cart(args, session)
                data = _exec_session_tool(tool_name, args, session)
            elif meta.method == "VECTOR":
                data = await self._exec_vector_tool(tool_name, args, session)
            else:
                params, path_params, json_body = _build_call_args(meta, args, reserved)
                body = await self._client.call(
                    meta.method, meta.path, token=token, params=params,
                    path_params=path_params, json_body=json_body,
                    idempotency_key=idempotency_key, correlation_id=session.session_id,
                )
                data = _map_tool_result(tool_name, body, args)
        except ToolError as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("tool %s failed kind=%s status=%s in %.0fms: %s", tool_name, e.error_kind, e.status_code, elapsed, e.message)
            self._debug.log_backend_call(session.session_id, turn_id, meta.method, meta.path, e.status_code, elapsed)
            return {"status": "error", "tool_name": tool_name, "message": e.message, "error_kind": e.error_kind}
        except Exception as e:
            logger.exception("tool %s crashed", tool_name)
            self._debug.log_error(session.session_id, turn_id, "tool_exec_error", str(e))
            return {"status": "error", "tool_name": tool_name, "message": f"{tool_name} failed: {e}", "error_kind": "unknown"}

        elapsed = (time.perf_counter() - start) * 1000
        logger.info("tool %s ok in %.0fms", tool_name, elapsed)
        self._debug.log_backend_call(session.session_id, turn_id, meta.method, meta.path, None, elapsed)
        return {"status": "success", "tool_name": tool_name, "data": _strip_sensitive(data)}

    async def _fetch_catalog_page(self, params: dict[str, Any], session: Session) -> list[dict[str, Any]]:
        from mcp_server.backend_client import ToolError

        try:
            body = await self._client.call("GET", "/catalog/products", token=None, params=params, correlation_id=session.session_id)
        except ToolError as exc:
            # Some backends cap `limit` below our page size and answer 422; retry once with a small page.
            if exc.error_kind == "validation_error" and int(params.get("limit", 0)) > 20:
                body = await self._client.call(
                    "GET", "/catalog/products", token=None, params={**params, "limit": 20}, correlation_id=session.session_id
                )
            else:
                raise
        items = body.get("items", body) if isinstance(body, dict) else body
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            raise ToolError("Unexpected response from the catalog service.", error_kind="backend_error")
        return [i for i in items if isinstance(i, dict)]

    async def _search_products(self, args: dict[str, Any], session: Session) -> dict[str, Any]:
        from mcp_server.tools.catalog import _map_product_card

        async def fetch(params: dict[str, Any]) -> list[dict[str, Any]]:
            return await self._fetch_catalog_page(params, session)

        outcome = await search_catalog(
            fetch,
            query=args.get("q"),
            category_id=args.get("category_id"),
            merchant_id=args.get("merchant_id"),
            price_min=args.get("price_min"),
            price_max=args.get("price_max"),
            currency=args.get("currency"),
            limit=args.get("limit") or 20,
        )
        items = [_map_product_card(p).model_dump() for p in outcome.items]
        result: dict[str, Any] = {
            "items": items,
            "query": outcome.query,
            "count": len(items),
            "total": outcome.total,
            "strategy": outcome.strategy,
        }
        for key in ("category_id", "merchant_id"):
            if args.get(key):
                result[key] = args[key]
        return result

    async def _enrich_add_to_cart(self, args: dict[str, Any], session: Session) -> dict[str, Any]:
        """Price, stock and names come from the catalog, never from the model."""
        from mcp_server.backend_client import ToolError
        from mcp_server.tools.catalog import _map_product_detail

        product_id = args.get("product_id")
        variant_id = args.get("variant_id")
        if not product_id and not variant_id:
            raise ToolError("add_to_cart needs a product_id or a variant_id.", error_kind="validation_error")
        if not product_id:
            return args  # variant only: nothing to verify against
        body = await self._client.call(
            "GET", "/catalog/products/{product_id}", token=None,
            path_params={"product_id": product_id}, correlation_id=session.session_id,
        )
        detail = _map_product_detail(body).model_dump()
        if str(detail.get("status", "active")).lower() != "active":
            raise ToolError("This product is not available right now.", error_kind="validation_error")
        variants = detail.get("variants") or []
        variant = next((v for v in variants if v.get("id") == variant_id), None) if variant_id else (variants[0] if variants else None)
        if variant is None:
            raise ToolError(
                "That variant was not found for this product." if variant_id else "This product has no purchasable variant yet.",
                error_kind="not_found" if variant_id else "validation_error",
            )
        stock = variant.get("quantity_available")
        raw_variant = next(
            (v for v in (body.get("variants") or []) if isinstance(v, dict) and str(v.get("id")) == str(variant["id"])), {}
        ) if isinstance(body, dict) else {}
        policy = raw_variant.get("inventory_policy", "tracked")
        if policy == "tracked" and stock is not None and int(stock) <= 0:
            raise ToolError("This item is out of stock.", error_kind="validation_error")
        price = variant.get("price") if isinstance(variant.get("price"), dict) and variant["price"].get("amount") else detail.get("price") or {}
        return {
            **args,
            "variant_id": variant["id"],
            "product_id": product_id,
            "product_name": detail.get("name", ""),
            "variant_name": variant.get("name", ""),
            "sku": variant.get("sku", ""),
            "unit_price_amount": int(price.get("amount", 0) or 0),
            "unit_price_currency": price.get("currency", "NGN"),
            "quantity_available": int(stock or 0),
        }

    async def _exec_vector_tool(self, tool_name: str, args: dict[str, Any], session: Session) -> dict[str, Any]:
        """semantic_search: knowledge base first, then the LIVE catalog for product questions.

        A stale or empty knowledge index must never make products unfindable, so when the
        index returns no product documents the query is answered from the catalog.
        """
        if tool_name != "semantic_search":
            return {"items": []}

        query = (args.get("query", "") or "").strip()
        try:
            top_k = max(1, min(int(args.get("top_k") or 5), 20))
        except (TypeError, ValueError):
            top_k = 5

        docs: list[dict[str, Any]] = []
        if self._knowledge_service is not None:
            try:
                kb = await self._knowledge_service.search(query, top_k=top_k)
                if kb.status == "ok":
                    docs = [
                        {"id": r.doc_id, "text": r.text, "score": r.score, "kind": r.kind or "", "topic": r.topic or "",
                         "title": r.title or "", "metadata": r.metadata or {}}
                        for r in kb.results
                    ]
                else:
                    logger.warning("knowledge search status=%s query=%r", kb.status, query)
            except Exception:
                logger.exception("knowledge search failed query=%r", query)
        elif self._vector_store is not None:
            try:
                docs = await self._vector_store.search(query, top_k=top_k)
            except Exception:
                logger.exception("vector store search failed")
        elif self._redis is not None:
            try:
                from mcp_server.vector_store import TfidfEmbedder, VectorStore

                store = VectorStore(redis_client=self._redis, embedder=TfidfEmbedder(), namespace=self._vector_namespace)
                docs = await store.search(query, top_k=top_k)
            except Exception:
                logger.exception("legacy vector search failed")

        def is_product_doc(d: dict[str, Any]) -> bool:
            meta = d.get("metadata") or {}
            return d.get("kind") == "product" or meta.get("kind") == "product" or bool(meta.get("product_id"))

        if query and not any(is_product_doc(d) for d in docs) and not _POLICY_QUERY.search(query):
            try:
                live = await self._search_products({"q": query, "limit": top_k * 2}, session)
                if live["items"]:
                    live["strategy"] = "live_catalog:" + live["strategy"]
                    return live
            except Exception:
                logger.exception("live catalog fallback failed query=%r", query)

        if not docs:
            return {"items": [], "query": query, "count": 0, "sources": []}
        return {"items": docs, "query": query, "count": len(docs), "sources": [_kb_doc_to_source(d) for d in docs]}

    # ------------------------------------------------------------------ directives

    def _result_to_directive(self, tool_name: str, result: dict[str, Any], session_id: str) -> dict[str, Any] | None:
        component_map = {
            "search_products": ComponentName.ProductGrid,
            "get_product_detail": ComponentName.ProductDetail,
            "get_categories": ComponentName.CategoryList,
            "semantic_search": ComponentName.ProductGrid,
            "view_cart": ComponentName.CartSummary,
            "add_to_cart": ComponentName.CartSummary,
            "update_cart_items": ComponentName.CartSummary,
            "remove_from_cart": ComponentName.CartSummary,
            "initiate_checkout": ComponentName.PaymentCapturePanel,
            "process_payment": ComponentName.PaymentCapturePanel,
            "get_order_status": ComponentName.OrderDetail,
            "list_orders": ComponentName.OrderList,
            "get_merchant_balance": ComponentName.MerchantBalanceCard,
            "get_merchant_ledger": ComponentName.LedgerTable,
            "get_fulfillment_status": ComponentName.FulfillmentTracker,
        }
        component = component_map.get(tool_name)
        if not component:
            return None
        props = _strip_sensitive(result.get("data", {}))
        if component == ComponentName.ProductGrid:
            props = _grid_props(props)
        return UIDirective(component=component, props=props, correlation_id=session_id).model_dump()


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------


def _args_of(tc: dict[str, Any]) -> dict[str, Any]:
    fn = tc.get("function") if isinstance(tc, dict) else None
    raw = (fn or {}).get("arguments", {}) if isinstance(fn, dict) else {}
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _grid_props(data: dict[str, Any]) -> dict[str, Any]:
    """Add a correct, instant summary and badges, computed from real data (not by the model)."""
    props = dict(data)
    items = [i for i in props.get("items", []) if isinstance(i, dict) and i.get("name") and isinstance(i.get("price"), dict)]
    if not items:
        return props
    props.pop("sources", None)
    amounts = [i["price"].get("amount") for i in items if i["price"].get("amount") is not None]
    if not amounts:
        return props
    currency = items[0]["price"].get("currency", "NGN")
    total = int(props.get("total") or len(items))
    query = props.get("query") or ""
    lo, hi = min(amounts), max(amounts)
    summary = f"{total} result{'s' if total != 1 else ''}" + (f" for \u201c{query}\u201d" if query else "")
    summary += f", {format_money(lo, currency)}" + (f" to {format_money(hi, currency)}" if hi != lo else "")
    props["summary"] = summary
    if len(items) > 1 and lo != hi:
        cheapest = min(items, key=lambda i: i["price"].get("amount") or 0)
        props["highlights"] = {str(cheapest["id"]): "Lowest price"}
    return props


def _tool_to_openai_format(tool_meta: Any) -> dict[str, Any]:
    from mcp_server.tools_catalog import parameters_for

    return {
        "type": "function",
        "function": {
            "name": tool_meta.name,
            "description": tool_meta.description,
            "parameters": parameters_for(tool_meta.name),
        },
    }


def _summarize_args(args: dict[str, Any]) -> dict[str, Any]:
    safe_keys = {"product_id", "order_id", "merchant_id", "variant_id", "quantity", "q", "query", "status"}
    return {k: v for k, v in args.items() if k in safe_keys}


def _summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    return {"status": result.get("status"), "tool_name": result.get("tool_name"), "has_data": "data" in result}


def _kb_doc_to_source(doc: dict[str, Any]) -> dict[str, Any]:
    meta = doc.get("metadata") or {}
    return {
        "doc_id": doc.get("id", ""),
        "title": doc.get("title", "") or meta.get("title", ""),
        "text": doc.get("text", ""),
        "score": float(doc.get("score", 0.0) or 0.0),
        "kind": doc.get("kind", "") or meta.get("kind", "text"),
        "topic": doc.get("topic", "") or meta.get("topic", ""),
        "metadata": doc.get("metadata", {}) or {},
    }


def _result_to_text(tool_name: str, result: dict[str, Any]) -> str | None:
    """Human-readable fallback summary. Shown to the shopper only if the model wrote no text itself."""
    data = result.get("data", {})
    if isinstance(result, list):
        data = result

    if tool_name in ("search_products", "semantic_search"):
        items = data.get("items", data) if isinstance(data, dict) else data
        query = data.get("query", "") if isinstance(data, dict) else ""
        if isinstance(items, list):
            count = len(items)
            if count == 0:
                return f"I couldn't find any results for '{query}'." if query else "I couldn't find any results."
            total = int(data.get("total") or count) if isinstance(data, dict) else count
            noun = "product" if tool_name == "search_products" or _looks_like_products(items) else "result"
            noun += "" if total == 1 else "s"
            return f"Found {total} {noun}" + (f" for '{query}'." if query else ".")
        return "Here are the products."

    if tool_name == "get_product_detail":
        name = data.get("name", "") if isinstance(data, dict) else ""
        return f"Here are the details for {name}." if name else "Here are the product details."
    if tool_name == "get_categories":
        items = data.get("items", data) if isinstance(data, dict) else data
        if isinstance(items, list):
            return f"Found {len(items)} categor{'ies' if len(items) != 1 else 'y'}."
        return "Here are the categories."
    if tool_name in ("view_cart",):
        items = data.get("items", []) if isinstance(data, dict) else []
        return f"Your cart has {len(items)} item{'s' if len(items) != 1 else ''}." if items else "Your cart is empty."
    if tool_name == "add_to_cart":
        return "Item added to your cart."
    if tool_name == "update_cart_items":
        return "Your cart is updated."
    if tool_name == "initiate_checkout":
        return "Checkout initiated. Here are your order details."
    if tool_name == "process_payment":
        status = data.get("status", "") if isinstance(data, dict) else ""
        if status == "initialized":
            return "Payment initialized. Please complete payment at the provider's page."
        if status == "success":
            return "Payment processed successfully."
        return f"Payment status: {status or 'unknown'}." if status else "Payment processed."
    if tool_name == "get_order_status":
        status = data.get("status", "") if isinstance(data, dict) else ""
        order_id = data.get("order_id", "") if isinstance(data, dict) else ""
        return f"Order {order_id} status: {status}." if status and order_id else "Here is the order status."
    if tool_name == "list_orders":
        if isinstance(data, list):
            return f"Found {len(data)} order{'s' if len(data) != 1 else ''}."
        return "Here are your orders."
    if tool_name == "get_merchant_balance":
        balance = data.get("balance", "") if isinstance(data, dict) else ""
        return f"Current merchant balance: {balance}." if balance else "Here is the merchant balance."
    if tool_name == "get_merchant_ledger":
        if isinstance(data, list):
            return f"Found {len(data)} ledger entr{'ies' if len(data) != 1 else 'y'}."
        return "Here is the merchant ledger."
    if tool_name == "get_fulfillment_status":
        status = data.get("status", "") if isinstance(data, dict) else ""
        return f"Fulfillment status: {status}." if status else "Here is the fulfillment status."
    return None


def _looks_like_products(items: list) -> bool:
    return bool(items) and isinstance(items[0], dict) and "price" in items[0] and "name" in items[0]


def _result_has_items(result: dict[str, Any]) -> bool:
    data = result.get("data") if isinstance(result, dict) else None
    if isinstance(data, dict):
        items = data.get("items")
        if isinstance(items, list):
            return len(items) > 0
        return bool(data.get("count"))
    if isinstance(data, list):
        return len(data) > 0
    return False


def _is_deferred_summary(tool_name: str, result: dict[str, Any]) -> bool:
    return tool_name == "semantic_search" and not _result_has_items(result)


def _flush_summaries(summary_buffer: list[tuple[str, str, dict[str, Any], bool]]) -> list[str]:
    """Emit deferred summaries, suppressing a 'couldn't find' miss when a sibling search had hits."""
    search_tools = {"search_products", "semantic_search"}
    has_hit = any(tn in search_tools and _result_has_items(result) for tn, _t, result, _d in summary_buffer)
    out: list[str] = []
    for tool_name, text, result, deferred in summary_buffer:
        if not deferred:
            continue
        if has_hit and tool_name in search_tools and not _result_has_items(result):
            continue
        out.append(text)
    return out


def _build_call_args(meta: Any, args: dict[str, Any], reserved: set[str]):
    params: dict[str, Any] = {}
    path_params: dict[str, str] = {}
    json_body: dict[str, Any] | None = None
    placeholders = set(re.findall(r"\{(\w+)\}", meta.path))
    for key, value in args.items():
        if key in reserved or value is None:
            continue
        if key in placeholders:
            path_params[key] = str(value)
        elif meta.method == "GET":
            params[key] = value
        else:
            json_body = json_body or {}
            json_body[key] = value
    return params or None, path_params or None, json_body


def _map_tool_result(tool_name: str, body: Any, args: dict[str, Any]) -> dict[str, Any]:
    """Map a raw backend response to the view DTOs the frontend expects (search_products has its own pipeline)."""
    if tool_name == "get_product_detail":
        from mcp_server.tools.catalog import _map_product_detail

        return _map_product_detail(body).model_dump()
    if tool_name == "get_categories":
        from mcp_server.tools.catalog import _map_category

        items = body.get("items", body) if isinstance(body, dict) else body
        if not isinstance(items, list):
            items = [items] if isinstance(items, dict) else []
        return {"items": [_map_category(c).model_dump() for c in items]}
    return body


def _exec_session_tool(tool_name: str, args: dict[str, Any], session: Session) -> dict[str, Any]:
    """Cart tools operate on the session cart, not the backend."""
    from mcp_server.tools.cart import _build_cart_view

    cart = session.get_cart()
    if tool_name == "view_cart":
        return _build_cart_view(cart).model_dump()
    if tool_name == "clear_cart":
        session.clear_cart()
        return _build_cart_view(session.get_cart()).model_dump()
    if tool_name == "remove_from_cart":
        variant_id = args.get("variant_id")
        cart["items"] = [i for i in cart["items"] if i.get("variant_id") != variant_id]
        session.set_cart(cart)
        return _build_cart_view(cart).model_dump()
    if tool_name == "update_cart_items":
        # Absolute quantities (idempotent): {variant_id, quantity}; 0 removes the line.
        wanted: dict[str, int] = {}
        for u in args.get("items") or []:
            if isinstance(u, dict) and u.get("variant_id"):
                try:
                    wanted[str(u["variant_id"])] = max(0, min(int(u.get("quantity", 0)), 99))
                except (TypeError, ValueError):
                    continue
        kept = []
        for item in cart["items"]:
            vid = str(item.get("variant_id"))
            if vid in wanted:
                qty = wanted[vid]
                if qty == 0:
                    continue
                item["quantity"] = qty
                unit = item["unit_price"]
                item["line_total"] = {"amount": qty * int(unit["amount"]), "currency": unit["currency"]}
            kept.append(item)
        cart["items"] = kept
        session.set_cart(cart)
        return _build_cart_view(cart).model_dump()
    if tool_name == "add_to_cart":
        variant_id = args.get("variant_id")
        quantity = max(1, min(int(args.get("quantity", 1) or 1), 99))
        for item in cart["items"]:
            if item.get("variant_id") == variant_id:
                item["quantity"] = min(int(item.get("quantity", 1)) + quantity, 99)
                item["line_total"]["amount"] = item["quantity"] * item["unit_price"]["amount"]
                session.set_cart(cart)
                return _build_cart_view(cart).model_dump()
        unit_amount = int(args.get("unit_price_amount", 0) or 0)
        currency = args.get("unit_price_currency", "NGN")
        cart["items"].append({
            "variant_id": variant_id, "product_id": args.get("product_id", ""),
            "product_name": args.get("product_name", ""), "variant_name": args.get("variant_name", ""),
            "sku": args.get("sku", ""), "quantity": quantity,
            "unit_price": {"amount": unit_amount, "currency": currency},
            "line_total": {"amount": quantity * unit_amount, "currency": currency},
            "quantity_available": int(args.get("quantity_available", 0) or 0),
        })
        session.set_cart(cart)
        return _build_cart_view(cart).model_dump()
    return {}


_SENSITIVE_KEYS = {
    "token", "access_token", "user_token", "user_confirmed_token",
    "confirmed_token", "password", "secret", "card_number", "cvv",
}


def _strip_sensitive(data: Any) -> Any:
    """Recursively remove sensitive keys so tokens never reach the model or the frontend."""
    if isinstance(data, dict):
        return {k: _strip_sensitive(v) for k, v in data.items() if k.lower() not in _SENSITIVE_KEYS}
    if isinstance(data, list):
        return [_strip_sensitive(x) for x in data]
    return data

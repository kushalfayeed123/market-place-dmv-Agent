"""Agent — the turn loop that orchestrates model calls, tool execution, and directive emission."""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any, Optional

from gateway.debug import DebugTracer
from gateway.provider import ModelBackend
from gateway.session import Session
from mcp_server.backend_client import BackendClient
from mcp_server.guards import ConfirmationGate
from schemas.components import ComponentName, UIDirective


class Agent:
    """The agent turn loop."""

    def __init__(
        self,
        backend_client: BackendClient,
        gate: ConfirmationGate,
        model_backend: ModelBackend,
        debug_tracer: DebugTracer,
        system_prompt: str,
        redis_client: Optional[Any] = None,
        vector_namespace: str = "agent",
        vector_store: Optional[Any] = None,
    ):
        self._client = backend_client
        self._gate = gate
        self._model = model_backend
        self._debug = debug_tracer
        self._system_prompt = system_prompt
        # Redis-backed vector store for semantic_search (knowledge retrieval).
        # Optional so the agent still functions — with an empty knowledge result
        # — when Redis is unavailable or not wired (e.g. inside unit tests).
        self._redis = redis_client
        self._vector_namespace = vector_namespace
        # When provided (e.g. by tests), `vector_store` is used directly instead
        # of building one from `redis_client`.
        self._vector_store = vector_store

    async def run_turn(
        self,
        session: Session,
        user_message: str,
        confirmed_token: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run a single turn of the agent loop.

        Bounded, tool-aware loop: the model may emit tool calls, the agent
        executes them and feeds the results back so the model can either deliver
        its final answer or make further (combinations of) tool calls for the
        same request — e.g. ``semantic_search`` -> ``get_product_detail``.
        The loop stops when the model stops calling tools, when a tool requires
        frontend confirmation (hand back via ``confirmed_token``), or when the
        iteration cap is reached.
        """
        turn_id = session.next_turn()
        session_id = session.session_id
        self._debug.log_turn_start(session_id, turn_id, user_message)

        intent, confidence = await self._classify_intent(user_message, session)
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
        available_tools = tools_for_role(session.role)
        tool_defs = [_tool_to_openai_format(t) for t in available_tools]
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

        self._debug.log_model_call(session_id, turn_id, self._model.provider_name, self._model.model_name)

        try:
            max_iterations = 5
            iteration = 0
            while True:
                iteration += 1
                if iteration > max_iterations:
                    self._debug.log_error(session_id, turn_id, "max_iterations", f"agent loop exceeded {max_iterations} iterations")
                    yield {"type": "error", "error": "The assistant took too many steps and stopped."}
                    return

                # Tool calls emitted by the model during THIS stream, plus the
                # executed results that get fed back as tool-role messages.
                tool_calls_this_stream: list[dict[str, Any]] = []
                tool_results: list[dict[str, Any]] = []

                async for event in self._model.stream(
                    messages=messages, tools=tool_defs, system_prompt=self._system_prompt,
                ):
                    async for output in self._handle_event(event, session, turn_id, confirmed_token, tool_results):
                        yield output
                    if event.get("type") == "tool_calls":
                        tool_calls_this_stream.extend(event.get("tool_calls", []))
                    elif event.get("type") == "tool_call":
                        single = event.get("tool_call")
                        if single:
                            tool_calls_this_stream.append(single)

                # A confirmation_required tool ends the turn; the frontend will
                # re-invoke /sse with a confirmed_token rather than the agent loop.
                if any(r.get("status") == "confirmation_required" for r in tool_results):
                    return

                # The model made tool calls: feed the results back and let it
                # decide the next step (this is what enables tool combinations).
                if tool_calls_this_stream and tool_results:
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": tc.get("name", ""),
                                    "arguments": json.dumps(tc.get("arguments", {})),
                                },
                            }
                            for tc in tool_calls_this_stream
                        ],
                    })
                    for r in tool_results:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": r["tool_call_id"],
                            "content": r["content"],
                        })
                    continue

                return
        except Exception as e:
            self._debug.log_error(session_id, turn_id, "agent_error", str(e))
            yield {"type": "error", "error": f"Agent error: {e!s}"}

    async def _handle_event(
        self,
        event: dict[str, Any],
        session: Session,
        turn_id: int,
        confirmed_token: str | None,
        tool_results: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        """Handle a single model event.

        ``tool_results`` accumulates the outcome of every tool call during this
        stream so the agent loop can feed the results back to the model for the
        next round (enabling combinations of tools per request).
        """
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
                tool_name = tc.get("name", "")
                tool_args = tc.get("arguments", {})
                tool_call_id = tc.get("id", "")
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
                            props={"tool_name": tool_name, "message": result.get("message", "Please confirm.")},
                            correlation_id=session_id,
                        ).model_dump(),
                    }
                    tool_results.append({
                        "tool_call_id": tool_call_id,
                        "status": "confirmation_required",
                        "content": json.dumps({"status": "confirmation_required", "message": result.get("message", "")}),
                    })
                elif result.get("status") == "error":
                    self._debug.log_error(session_id, turn_id, result.get("error_kind", "tool_error"), result.get("message", "Unknown"))
                    yield {"type": "error", "error": result.get("message", "An error occurred.")}
                    tool_results.append({
                        "tool_call_id": tool_call_id,
                        "status": "error",
                        "content": json.dumps(_summarize_result(result)),
                    })
                else:
                    self._debug.log_tool_result(session_id, turn_id, tool_name, _summarize_result(result))
                    text_summary = _result_to_text(tool_name, result)
                    if text_summary:
                        yield {"type": "text", "content": text_summary}
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
                            "data": _strip_sensitive(result.get("data", {})),
                        }),
                    })

        elif event_type == "usage":
            self._debug.log_model_usage(session_id, turn_id, event.get("usage", {}))

        elif event_type == "finish":
            self._debug.log_turn_end(session_id, turn_id, event.get("reason", "unknown"))

        elif event_type == "error":
            self._debug.log_error(session_id, turn_id, "model_error", event.get("error", "Unknown"))
            yield {"type": "error", "error": event.get("error", "Model error")}

    async def _execute_tool(
        self, tool_name: str, args: dict[str, Any], session: Session, turn_id: int,
    ) -> dict[str, Any]:
        """Execute a tool call against the backend, the session cart, or the vector store.

        ``session.token`` is forwarded to the backend only as an Authorization header
        and is never placed in the result data seen by the model or the frontend.
        """
        from mcp_server.backend_client import ToolError
        from mcp_server.tools_catalog import BY_NAME, is_mutating_financial

        meta = BY_NAME.get(tool_name)
        if not meta:
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

        # Reserved keys are control-flow / auth and are never forwarded to the backend.
        reserved = {"token", "user_confirmed_token", "confirmed_token"}
        token = session.token
        idempotency_key = f"{session.session_id}:{turn_id}:{tool_name}" if meta.method != "GET" else None
        if idempotency_key:
            self._debug.log_idempotency_key(session.session_id, turn_id, tool_name, idempotency_key)

        start = time.perf_counter()
        try:
            if meta.method == "SESSION":
                data = _exec_session_tool(tool_name, args, session)
            elif meta.method == "VECTOR":
                data = await self._exec_vector_tool(tool_name, args)
            else:
                params, path_params, json_body = _build_call_args(meta, args, reserved)
                body = await self._client.call(
                    meta.method, meta.path, token=token, params=params,
                    path_params=path_params, json_body=json_body,
                    idempotency_key=idempotency_key, correlation_id=session.session_id,
                )
                data = _map_tool_result(tool_name, body, args)
        except ToolError as e:
            self._debug.log_backend_call(session.session_id, turn_id, meta.method, meta.path, e.status_code, (time.perf_counter() - start) * 1000)
            return {"status": "error", "tool_name": tool_name,
                    "message": e.message if hasattr(e, "message") else str(e),
                    "error_kind": getattr(e, "error_kind", "unknown")}
        except Exception as e:
            self._debug.log_error(session.session_id, turn_id, "tool_exec_error", str(e))
            return {"status": "error", "tool_name": tool_name, "message": str(e), "error_kind": "unknown"}

        self._debug.log_backend_call(session.session_id, turn_id, meta.method, meta.path, None, (time.perf_counter() - start) * 1000)
        return {"status": "success", "tool_name": tool_name, "data": _strip_sensitive(data)}

    async def _exec_vector_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute a VECTOR tool (currently only ``semantic_search``).

        Backed by the Redis vector store when a Redis client or an injected
        vector store is available; otherwise returns an empty result so the turn
        degrades gracefully (no crash, no 500). The session token is never used
        here, and results are stripped of sensitive keys before emission.
        """
        if tool_name != "semantic_search":
            return {"items": []}
        vector_store = self._vector_store
        if vector_store is None and self._redis is not None:
            from mcp_server.vector_store import TfidfEmbedder, VectorStore
            vector_store = VectorStore(
                redis_client=self._redis,
                embedder=TfidfEmbedder(),
                namespace=self._vector_namespace,
            )
        if vector_store is None:
            return {"items": []}
        query = args.get("query", "") or ""
        top_k = args.get("top_k")
        try:
            top_k = int(top_k) if top_k is not None else 5
        except (TypeError, ValueError):
            top_k = 5
        try:
            results = await vector_store.search(query, top_k=max(1, min(top_k, 20)))
        except Exception:
            return {"items": []}
        # VectorStore.search returns {id, text, metadata, score}; project to the
        # MCP knowledge-tool shape ({text, score, metadata}) expected downstream.
        items = [
            {
                "text": r.get("text", ""),
                "score": r.get("score", 0.0),
                "metadata": r.get("metadata", {}) or {},
            }
            for r in results
        ]
        return {"items": items}

    _INTENT_CLASSIFICATION_PROMPT = (
        "You are an intent classifier for a marketplace assistant. "
        "Classify the user's message into exactly one of these intents:\n"
        "- 'info': The user is asking a question, browsing, searching, reading reviews, "
        "checking order status, or seeking information. No state will be modified.\n"
        "- 'action': The user wants to perform an operation that modifies state — "
        "buying, checking out, paying, placing an order, purchasing, adding to cart, "
        "requesting a refund or payout, creating/updating/deleting a resource, "
        "approving KYC, shipping, or fulfilling.\n\n"
        "Be precise: if the user is merely asking *about* an action (e.g. 'how do I refund?') "
        "without requesting it, classify as 'info'. Only classify as 'action' when the user "
        "is requesting the operation itself.\n\n"
        "Respond with JSON only: {\"intent\": \"info\"|\"action\", \"confidence\": 0.0-1.0, \"reasoning\": \"brief explanation\"}"
    )

    async def _classify_intent(self, message: str, session: Session) -> tuple[str, float]:
        """Classify user intent using the model. Returns (intent, confidence)."""
        result = await self._model.classify(
            message=message,
            system_prompt=self._INTENT_CLASSIFICATION_PROMPT,
            valid_intents=["info", "action"],
        )
        intent = result.get("intent", "info")
        confidence = result.get("confidence", 0.5)
        if result.get("error"):
            self._debug.log_error(session.session_id, session.current_turn_id, "intent_classification_error", result["error"])
        return intent, confidence

    def _result_to_directive(self, tool_name: str, result: dict[str, Any], session_id: str) -> dict[str, Any] | None:
        """Convert a tool result to a UI directive."""
        component_map = {
            "search_products": ComponentName.ProductGrid,
            "get_product_detail": ComponentName.ProductDetail,
            "get_categories": ComponentName.CategoryList,
            "semantic_search": ComponentName.ProductGrid,
            "view_cart": ComponentName.CartSummary,
            "add_to_cart": ComponentName.CartSummary,
            "initiate_checkout": ComponentName.OrderConfirmation,
            "get_order_status": ComponentName.OrderDetail,
            "list_orders": ComponentName.OrderList,
            "get_merchant_balance": ComponentName.MerchantBalanceCard,
            "get_merchant_ledger": ComponentName.LedgerTable,
            "get_fulfillment_status": ComponentName.FulfillmentTracker,
        }
        component = component_map.get(tool_name)
        if not component:
            return None
        return UIDirective(component=component, props=_strip_sensitive(result.get("data", {})), correlation_id=session_id).model_dump()


def _tool_to_openai_format(tool_meta: Any) -> dict[str, Any]:
    """Convert a ToolMeta to OpenAI function-calling format.

    Parameter schemas come from ``mcp_server.tools_catalog.parameters_for``,
    which mirrors each tool's real signature (minus reserved auth keys), so the
    model can populate arguments accurately rather than guessing.
    """
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
    safe_keys = {"product_id", "order_id", "merchant_id", "variant_id", "quantity", "q", "status"}
    return {k: v for k, v in args.items() if k in safe_keys}


def _summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    return {"status": result.get("status"), "tool_name": result.get("tool_name"), "has_data": "data" in result}


def _result_to_text(tool_name: str, result: dict[str, Any]) -> str | None:
    """Generate a human-readable text summary of a tool result."""
    data = result.get("data", {})
    if isinstance(result, list):
        data = result

    if tool_name == "search_products":
        items = data.get("items", data) if isinstance(data, dict) else data
        if isinstance(items, list):
            count = len(items)
            query = data.get("q", "") if isinstance(data, dict) else ""
            if query:
                return f"Found {count} product{'s' if count != 1 else ''} matching '{query}'."
            return f"Found {count} product{'s' if count != 1 else ''}."
        return "Here are the products."

    if tool_name == "get_product_detail":
        name = data.get("name", "") if isinstance(data, dict) else ""
        if name:
            return f"Here's the details for {name}."
        return "Here are the product details."

    if tool_name == "get_categories":
        if isinstance(data, list):
            count = len(data)
            return f"Found {count} categor{'ies' if count != 1 else 'y'}."
        return "Here are the categories."

    if tool_name == "semantic_search":
        items = data.get("items", data) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    if isinstance(items, list):
        count = len(items)
        if count == 0:
            return "I couldn't find anything relevant in the knowledge base."
        return f"Found {count} relevant result{'s' if count != 1 else ''} from the knowledge base."
    return "Here's what I found."

    if tool_name == "view_cart":
        items = data.get("items", []) if isinstance(data, dict) else []
        if isinstance(items, list) and items:
            count = len(items)
            return f"Your cart has {count} item{'s' if count != 1 else ''}."
        return "Your cart is empty."

    if tool_name == "add_to_cart":
        return "Item added to your cart."

    if tool_name == "initiate_checkout":
        return "Checkout initiated. Here are your order details."

    if tool_name == "get_order_status":
        status = data.get("status", "") if isinstance(data, dict) else ""
        order_id = data.get("order_id", "") if isinstance(data, dict) else ""
        if status and order_id:
            return f"Order {order_id} status: {status}."
        return "Here's the order status."

    if tool_name == "list_orders":
        if isinstance(data, list):
            count = len(data)
            return f"Found {count} order{'s' if count != 1 else ''}."
        return "Here are your orders."

    if tool_name == "get_merchant_balance":
        balance = data.get("balance", "") if isinstance(data, dict) else ""
        if balance:
            return f"Current merchant balance: {balance}."
        return "Here's the merchant balance."

    if tool_name == "get_merchant_ledger":
        if isinstance(data, list):
            count = len(data)
            return f"Found {count} ledger entr{'ies' if count != 1 else 'y'}."
        return "Here's the merchant ledger."

    if tool_name == "get_fulfillment_status":
        status = data.get("status", "") if isinstance(data, dict) else ""
        if status:
            return f"Fulfillment status: {status}."
        return "Here's the fulfillment status."

    return None

def _build_call_args(meta: Any, args: dict[str, Any], reserved: set[str]):
    """Split tool args into backend query params / path params / JSON body."""
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
    """Map a raw backend response to the view DTOs the frontend expects.

    Catalog tools reuse the MCP server's mappers (single source of truth for shapes);
    unknown tools pass the raw backend body through.
    """
    if tool_name == "search_products":
        from mcp_server.tools.catalog import _map_product_card
        items = body.get("items", body) if isinstance(body, dict) else body
        if not isinstance(items, list):
            items = [items] if isinstance(items, dict) else []
        result: dict[str, Any] = {"items": [_map_product_card(p).model_dump() for p in items]}
        for key in ("q", "category_id", "merchant_id"):
            if args.get(key):
                result[key] = args[key]
        return result
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


def _exec_session_tool(tool_name: str, args: dict[str, Any], session: "Session") -> dict[str, Any]:
    """Cart tools operate on the session cart (Redis), not the backend."""
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
    if tool_name == "add_to_cart":
        variant_id = args.get("variant_id")
        for item in cart["items"]:
            if item.get("variant_id") == variant_id:
                item["quantity"] = int(item.get("quantity", 1)) + int(args.get("quantity", 1))
                item["line_total"]["amount"] = item["quantity"] * item["unit_price"]["amount"]
                session.set_cart(cart)
                return _build_cart_view(cart).model_dump()
        quantity = int(args.get("quantity", 1))
        unit_amount = float(args.get("unit_price_amount", 0))
        currency = args.get("unit_price_currency", "NGN")
        cart["items"].append({
            "variant_id": variant_id, "product_id": args.get("product_id", ""),
            "product_name": args.get("product_name", ""), "variant_name": args.get("variant_name", ""),
            "sku": args.get("sku", ""), "quantity": quantity,
            "unit_price": {"amount": unit_amount, "currency": currency},
            "line_total": {"amount": quantity * unit_amount, "currency": currency},
            "quantity_available": int(args.get("quantity_available", 0)),
        })
        session.set_cart(cart)
        return _build_cart_view(cart).model_dump()
    return {}


_SENSITIVE_KEYS = {
    "token", "access_token", "user_token", "user_confirmed_token",
    "confirmed_token", "password", "secret", "card_number", "cvv",
}


def _strip_sensitive(data: Any) -> Any:
    """Recursively remove sensitive keys so tokens never reach the frontend."""
    if isinstance(data, dict):
        return {k: _strip_sensitive(v) for k, v in data.items() if k.lower() not in _SENSITIVE_KEYS}
    if isinstance(data, list):
        return [_strip_sensitive(x) for x in data]
    return data

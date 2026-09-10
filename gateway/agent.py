"""Agent — the turn loop that orchestrates model calls, tool execution, and directive emission."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

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
    ):
        self._client = backend_client
        self._gate = gate
        self._model = model_backend
        self._debug = debug_tracer
        self._system_prompt = system_prompt

    async def run_turn(
        self,
        session: Session,
        user_message: str,
        confirmed_token: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run a single turn of the agent loop."""
        turn_id = session.next_turn()
        session_id = session.session_id
        self._debug.log_turn_start(session_id, turn_id, user_message)

        intent, confidence = await self._classify_intent(user_message, session)
        self._debug.log_intent(session_id, turn_id, intent, confidence)

        if intent == "action" and not session.user_token:
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
        messages = [{"role": "user", "content": user_message}]

        self._debug.log_model_call(session_id, turn_id, self._model.provider_name, self._model.model_name)

        try:
            async for event in self._model.stream(
                messages=messages, tools=tool_defs, system_prompt=self._system_prompt,
            ):
                async for output in self._handle_event(event, session, turn_id, confirmed_token):
                    yield output
        except Exception as e:
            self._debug.log_error(session_id, turn_id, "agent_error", str(e))
            yield {"type": "error", "error": f"Agent error: {e!s}"}

    async def _handle_event(
        self,
        event: dict[str, Any],
        session: Session,
        turn_id: int,
        confirmed_token: str | None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Handle a single model event."""
        event_type = event.get("type")
        session_id = session.session_id

        if event_type == "delta":
            yield {"type": "text", "content": event.get("content", "")}

        elif event_type == "tool_calls":
            for tc in event.get("tool_calls", []):
                tool_name = tc.get("name", "")
                tool_args = tc.get("arguments", {})
                self._debug.log_tool_selected(session_id, turn_id, tool_name, tool_args)

                from mcp_server.tools_catalog import is_mutating_financial
                if is_mutating_financial(tool_name) and confirmed_token:
                    tool_args["user_confirmed_token"] = confirmed_token

                result = await self._execute_tool(tool_name, tool_args, session, turn_id)

                if result.get("status") == "confirmation_required":
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
                elif result.get("status") == "error":
                    self._debug.log_error(session_id, turn_id, result.get("error_kind", "tool_error"), result.get("message", "Unknown"))
                    yield {"type": "error", "error": result.get("message", "An error occurred.")}
                else:
                    self._debug.log_tool_result(session_id, turn_id, tool_name, _summarize_result(result))
                    text_summary = _result_to_text(tool_name, result)
                    if text_summary:
                        yield {"type": "text", "content": text_summary}
                    directive = self._result_to_directive(tool_name, result, session_id)
                    if directive:
                        self._debug.log_directive(session_id, turn_id, directive["component"], directive.get("props", {}))
                        yield {"type": "ui_directive", "directive": directive}

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
        """Execute a tool call."""
        from mcp_server.tools_catalog import BY_NAME, is_mutating_financial
        meta = BY_NAME.get(tool_name)
        if not meta:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}

        args["token"] = session.user_token

        if is_mutating_financial(tool_name):
            provided_token = args.get("user_confirmed_token")
            if not provided_token:
                return {"status": "confirmation_required", "message": f"This action ({tool_name}) requires your confirmation."}
            try:
                await self._gate.check_confirmation(session.session_id, turn_id, tool_name, provided_token)
            except Exception as e:
                return {"status": "confirmation_required", "message": str(e), "error_kind": "confirmation_required"}

        return {"status": "success", "tool_name": tool_name, "data": args}

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
        return UIDirective(component=component, props=result.get("data", {}), correlation_id=session_id).model_dump()


def _tool_to_openai_format(tool_meta: Any) -> dict[str, Any]:
    """Convert a ToolMeta to OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {"name": tool_meta.name, "description": tool_meta.description, "parameters": {"type": "object", "properties": {}}},
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
        if isinstance(data, list):
            count = len(data)
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

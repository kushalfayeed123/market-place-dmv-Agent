"""Agent — the turn loop that orchestrates model calls, tool execution, and directive emission."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

from gateway.debug import DebugTracer
from gateway.provider import ModelBackend
from gateway.session import Session
from mcp_server.backend_client import BackendClient, ToolError
from mcp_server.guards import ConfirmationGate, ConfirmationRequiredError
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
        """Run a single turn of the agent loop.

        Flow per turn: model stream → tool calls → directives → repeat,
        until the model emits a final text response or MAX_ITERATIONS is hit.

        Whenever a successful tool call returns displayable data, a ``ui_directive``
        is emitted alongside the model-generated text summary so the frontend can
        render the data on screen.
        """
        turn_id = session.next_turn()
        session_id = session.session_id
        self._debug.log_turn_start(session_id, turn_id, user_message)
        print(f"[AGENT] Turn {turn_id} started for session {session_id[:8]}...")
        print(f"[AGENT] User message: {user_message[:100]}")

        # Check if user is providing a token for sign-in
        token_match = re.search(r"(?:token|api[._-]?key|access[._-]?key)\s*[:=]\s*(\S+)", user_message, re.IGNORECASE)
        if token_match:
            new_token = token_match.group(1)
            session.user_token = new_token
            print("[AGENT] Token received, session authenticated")
            yield {
                "type": "text",
                "content": "You're now signed in! I've saved your token for this session. You can now perform actions like adding to cart or checking out.",
            }
            return

        intent, confidence = await self._classify_intent(user_message, session)
        self._debug.log_intent(session_id, turn_id, intent, confidence)
        print(f"[AGENT] Intent classified: {intent} (confidence: {confidence:.2f})")

        if intent == "action" and not session.user_token:
            self._debug.log("auth.required", {"reason": "unauthenticated_action"}, session_id, turn_id)
            print("[AGENT] Action requires authentication, prompting sign-in")
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

        # Build conversation history for the agent loop
        messages = [{"role": "user", "content": user_message}]

        self._debug.log_model_call(session_id, turn_id, self._model.provider_name, self._model.model_name)
        print(f"[AGENT] Starting agent loop with {len(tool_defs)} tools available")

        # Agent loop: continue until model generates a final response (no more tool calls)
        max_iterations = 5  # Prevent infinite loops
        for iteration in range(max_iterations):
            print(f"[AGENT] Agent loop iteration {iteration + 1}")
            event_count = 0
            tool_calls_made = False
            error_events: list[str] = []

            async for event in self._model.stream(
                messages=messages, tools=tool_defs, system_prompt=self._system_prompt,
            ):
                event_count += 1
                event_type = event.get("type")
                print(f"[AGENT] Model event #{event_count}: type={event_type}")

                if event.get("type") == "error":
                    error_events.append(event.get("error", "Unknown error"))

                # Handle tool calls - execute them and add results to conversation
                if event_type == "tool_calls":
                    tool_calls_made = True
                    # Add the assistant message with tool calls to conversation
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": event.get("tool_calls", []),
                    })
                    # Execute each tool call and add results
                    for tc in event.get("tool_calls", []):
                        # Extract tool name and arguments from the function wrapper
                        function_info = tc.get("function", {})
                        tool_name = function_info.get("name", "")
                        # Arguments come as JSON string, parse them
                        args_str = function_info.get("arguments", "{}")
                        try:
                            tool_args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        except json.JSONDecodeError:
                            tool_args = {}
                        print(f"[AGENT] Tool call: {tool_name} with args {tool_args}")

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
                            return
                        elif result.get("status") == "error":
                            self._debug.log_error(session_id, turn_id, result.get("error_kind", "tool_error"), result.get("message", "Unknown"))
                            print(f"[AGENT] Tool error: {tool_name} - {result.get('message', 'Unknown error')}")
                            yield {"type": "error", "error": result.get("message", "An error occurred.")}
                        else:
                            # Add tool result to conversation history so the model
                            # can dynamically summarize the real DB data.
                            tool_result_content = self._format_tool_result_for_context(result)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.get("id", ""),
                                "content": tool_result_content,
                            })
                            print(f"[AGENT] Tool success: {tool_name} - result type: {type(result.get('data')).__name__}")
                            # Emit a ui_directive alongside the (later) model text
                            # summary so the frontend can render the DB data.
                            directive = self._result_to_directive(tool_name, result, session_id)
                            if directive:
                                self._debug.log_directive(session_id, turn_id, directive["component"], directive.get("props", {}))
                                print(f"[AGENT] UI directive: {directive['component']}")
                                yield {"type": "ui_directive", "directive": directive}
                    # Continue the loop to send results back to model
                    continue

                # For non-tool-call events, yield to user
                async for output in self._handle_event_simple(event, session, turn_id):
                    yield output

            print(f"[AGENT] Model stream completed with {event_count} events")

            # Handle empty stream case
            if event_count == 0:
                print("[AGENT] WARNING: Model stream returned no events")
                yield {
                    "type": "error",
                    "error": "I apologize, but I'm having trouble processing your request right now. "
                             "The AI service appears to be temporarily unavailable. Please try again in a moment.",
                }
                return

            # If no tool calls were made, the model generated a final response - exit loop
            if not tool_calls_made:
                print("[AGENT] Agent loop complete - model generated final response")
                return

            # If all events were errors, exit loop
            if error_events and event_count == len(error_events):
                print(f"[AGENT] WARNING: All model events were errors: {error_events}")
                rate_limit_indicators = ["rate limit", "resourceexhausted", "request limit", "429", "503"]
                is_rate_limit = any(
                    indicator in error.lower()
                    for error in error_events
                    for indicator in rate_limit_indicators
                )
                if is_rate_limit:
                    yield {
                        "type": "error",
                        "error": "I apologize, but the AI service is currently experiencing high demand. "
                                 "Please wait a moment and try again.",
                    }
                else:
                    yield {
                        "type": "error",
                        "error": "I apologize, but I encountered an error while processing your request. "
                                 "Please try again.",
                    }
                return

        print(f"[AGENT] Agent loop reached max iterations ({max_iterations})")

    def _format_tool_result_for_context(self, result: dict[str, Any]) -> str:
        """Format tool result as a string for the model context."""
        import json
        data = result.get("data", {})
        if isinstance(data, list):
            # For lists, summarize if too long
            if len(data) > 10:
                return json.dumps({
                    "count": len(data),
                    "items": data[:5],
                    "note": f"Showing 5 of {len(data)} items"
                }, default=str)
            return json.dumps(data, default=str)
        elif isinstance(data, dict):
            return json.dumps(data, default=str)
        return str(data)

    async def _handle_event_simple(
        self,
        event: dict[str, Any],
        session: Session,
        turn_id: int,
    ) -> AsyncIterator[dict[str, Any]]:
        """Handle a single model event (simplified version without tool call handling)."""
        event_type = event.get("type")
        session_id = session.session_id

        if event_type == "delta":
            content = event.get("content", "")
            if content:
                self._debug.log("model.text_delta", {"content": content[:200]}, session_id, turn_id)
                yield {"type": "text", "content": content}

        elif event_type == "usage":
            self._debug.log_model_usage(session_id, turn_id, event.get("usage", {}))

        elif event_type == "finish":
            self._debug.log_turn_end(session_id, turn_id, event.get("reason", "unknown"))

        elif event_type == "error":
            self._debug.log_error(session_id, turn_id, "model_error", event.get("error", "Unknown"))
            yield {"type": "error", "error": event.get("error", "Model error")}

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
            content = event.get("content", "")
            if content:
                self._debug.log("model.text_delta", {"content": content[:200]}, session_id, turn_id)
                yield {"type": "text", "content": content}

        elif event_type == "tool_calls":
            for tc in event.get("tool_calls", []):
                tool_name = tc.get("name", "")
                tool_args = tc.get("arguments", {})
                self._debug.log_tool_selected(session_id, turn_id, tool_name, tool_args)
                print(f"[AGENT] Tool call: {tool_name} with args {tool_args}")

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
                    print(f"[AGENT] Tool error: {tool_name} - {result.get('message', 'Unknown error')}")
                    yield {"type": "error", "error": result.get("message", "An error occurred.")}
                else:
                    self._debug.log_tool_result(session_id, turn_id, tool_name, _summarize_result(result))
                    print(f"[AGENT] Tool success: {tool_name} - result type: {type(result.get('data')).__name__}")
                    text_summary = _result_to_text(tool_name, result)
                    if text_summary:
                        print(f"[AGENT] Text summary: {text_summary}")
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
        """Execute a tool call against the backend API."""
        from mcp_server.tools_catalog import BY_NAME, is_mutating_financial

        session_id = session.session_id
        meta = BY_NAME.get(tool_name)
        if not meta:
            self._debug.log_error(session_id, turn_id, "unknown_tool", f"Unknown tool: {tool_name}")
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}

        self._debug.log("tool.execute_start", {"tool_name": tool_name, "args": _summarize_args(args)}, session_id, turn_id)

        # Check confirmation for mutating financial tools
        if is_mutating_financial(tool_name):
            provided_token = args.get("user_confirmed_token")
            if not provided_token:
                return {"status": "confirmation_required", "message": f"This action ({tool_name}) requires your confirmation."}
            try:
                await self._gate.check_confirmation(session_id, turn_id, tool_name, provided_token)
            except ConfirmationRequiredError as e:
                return {"status": "confirmation_required", "message": str(e), "error_kind": "confirmation_required"}

        # Call the backend API
        token = session.user_token
        try:
            self._debug.log_backend_call(session_id, turn_id, meta.method, meta.path, None, 0)
            # Build full URL for logging
            full_url = f"{self._client._base}{self._client._prefix}{meta.path}"
            # Extract path params (keys like {product_id} in the path)
            path_keys = re.findall(r"\{(\w+)\}", meta.path)
            path_params = {k: args[k] for k in path_keys if k in args} if path_keys else None

            # Build the payload for logging
            request_payload = {}
            if meta.method == "GET":
                request_payload = {"params": args}
            elif meta.method in ("POST", "PUT", "PATCH"):
                request_payload = {"json_body": args}
            if path_params:
                request_payload["path_params"] = path_params

            print(f"[AGENT] Backend request: {meta.method} {full_url}")
            print(f"[AGENT] Backend payload: {request_payload}")

            result = await self._client.call(
                meta.method,
                meta.path,
                token=token,
                params=args if meta.method == "GET" else None,
                json_body=args if meta.method in ("POST", "PUT", "PATCH") else None,
                path_params=path_params,
            )
            self._debug.log("tool.execute_success", {"tool_name": tool_name, "result_type": type(result).__name__}, session_id, turn_id)
            print(f"[AGENT] Backend response: {type(result).__name__} - {str(result)[:200]}")
            return {"status": "success", "tool_name": tool_name, "data": result}
        except ToolError as e:
            self._debug.log_error(session_id, turn_id, "tool_execution_error", f"{tool_name}: {e}")
            print(f"[AGENT] Backend error: {e}")
            return {"status": "error", "message": f"Tool {tool_name} failed: {e}", "error_kind": "tool_execution_error"}

    _INTENT_CLASSIFICATION_PROMPT = (
        "You are an intent classifier for a marketplace assistant. "
        "Classify the user\'s message into exactly one of these intents:\n"
        "- \'info\': The user is browsing, searching, asking questions, reading reviews, "
        "checking order status, expressing general interest in products, or seeking information. "
        "No state will be modified. This includes phrases like \'I am looking to buy X\', "
        "\'show me X\', \'find me X\', \'I want to see X\', \'do you have X\', \'how much is X\', etc.\n"
        "- \'action\': The user explicitly requests an operation that modifies state \u2014 "
        "adding to cart (\'add this to my cart\'), checking out (\'checkout now\'), "
        "paying (\'pay for this\'), placing an order (\'place the order\'), "
        "requesting a refund (\'refund this order\'), requesting a payout, "
        "creating/updating/deleting a resource, approving KYC, shipping, or fulfilling.\n\n"
        "Be precise: browsing and searching are ALWAYS \'info\'. "
        "Only classify as \'action\' when the user explicitly requests a state-changing operation. "
        "If the user is just expressing interest or looking around, classify as \'info\'.\n\n"
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
        """Convert a successful tool result to a frontend UI directive.

        Whenever a tool returns displayable database data, emit a matching
        allowlisted component with validated DTO props. This travels with the
        model-generated dynamic text summary: the frontend renders ``props``
        visually while the model text explains the same result.

        If normalization fails (e.g., unexpected data shape), fall back to
        passing the raw unwrapped data so the frontend still has something
        to render rather than silently dropping the result.
        """
        component_map = {
            "search_products": ComponentName.ProductGrid,
            "semantic_search": ComponentName.ProductGrid,
            "get_product_detail": ComponentName.ProductDetail,
            "get_categories": ComponentName.CategoryList,
            "view_cart": ComponentName.CartSummary,
            "add_to_cart": ComponentName.CartSummary,
            "remove_from_cart": ComponentName.CartSummary,
            "clear_cart": ComponentName.CartSummary,
            "initiate_checkout": ComponentName.OrderConfirmation,
            "get_order_status": ComponentName.OrderDetail,
            "list_orders": ComponentName.OrderList,
            "get_merchant_balance": ComponentName.MerchantBalanceCard,
            "get_merchant_ledger": ComponentName.LedgerTable,
            "get_merchant_profile": ComponentName.MerchantProfile,
            "get_fulfillment_status": ComponentName.FulfillmentTracker,
        }
        component = component_map.get(tool_name)
        data = _unwrap_tool_data(result.get("data", {}))
        if not component:
            # Fallback for unknown tools: emit directive with raw data so the
            # frontend can display something instead of nothing.
            if data is not None:
                return UIDirective(
                    component=ComponentName.ErrorMessage,
                    props={"message": "Data received, but no specific component mapping available.", "raw_data": data},
                    correlation_id=session_id,
                ).model_dump()
            # No data available – still emit a minimal directive with summary info
            return UIDirective(
                component=ComponentName.ErrorMessage,
                props={"message": "No data available from the backend.", "raw_data": None},
                correlation_id=session_id,
            ).model_dump()
        normalized = _normalize_directive_props(component, data)
        if normalized is not None:
            return UIDirective(component=component, props=normalized, correlation_id=session_id).model_dump()
        # Normalization fell back to None – emit the raw unwrapped data so the
        # frontend can still display something instead of nothing.
        # Return None when there is nothing displayable — the caller then
        # skips emitting a directive and relies on the model text alone.
        if data is None:
            return None
        return UIDirective(
            component=component,
            props={"raw_data": data} if data is not None else {},
            correlation_id=session_id,
        ).model_dump()


def _tool_to_openai_format(tool_meta: Any) -> dict[str, Any]:
    """Convert a ToolMeta to OpenAI function-calling format with proper parameters."""
    # Define parameters for each tool based on its purpose
    tool_parameters = _get_tool_parameters(tool_meta.name)
    return {
        "type": "function",
        "function": {
            "name": tool_meta.name,
            "description": tool_meta.description,
            "parameters": tool_parameters,
        },
    }


def _get_tool_parameters(tool_name: str) -> dict[str, Any]:
    """Return the parameter schema for a given tool name."""
    # Tool parameter definitions for the OpenAI function-calling format
    tool_schemas: dict[str, dict[str, Any]] = {
        "search_products": {
            "type": "object",
            "properties": {
                "q": {"type": "string", "description": "Search query string (e.g., 'hp laptop', 'iphone 15')"},
                "category": {"type": "string", "description": "Optional category filter"},
                "min_price": {"type": "number", "description": "Optional minimum price filter"},
                "max_price": {"type": "number", "description": "Optional maximum price filter"},
                "limit": {"type": "integer", "description": "Maximum number of results to return (default 20)"},
            },
            "required": ["q"],
        },
        "get_product_detail": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "The unique identifier of the product"},
            },
            "required": ["product_id"],
        },
        "get_categories": {"type": "object", "properties": {}},
        "semantic_search": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "limit": {"type": "integer", "description": "Maximum number of results (default 10)"},
            },
            "required": ["query"],
        },
        "get_user_profile": {"type": "object", "properties": {}},
        "add_to_cart": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "The product ID to add to cart"},
                "variant_id": {"type": "string", "description": "Optional variant ID (e.g., color, size)"},
                "quantity": {"type": "integer", "description": "Quantity to add (default 1)"},
            },
            "required": ["product_id"],
        },
        "view_cart": {"type": "object", "properties": {}},
        "remove_from_cart": {
            "type": "object",
            "properties": {
                "item_id": {"type": "string", "description": "The cart item ID to remove"},
            },
            "required": ["item_id"],
        },
        "clear_cart": {"type": "object", "properties": {}},
        "initiate_checkout": {
            "type": "object",
            "properties": {
                "user_confirmed_token": {"type": "string", "description": "Confirmation token from the user"},
            },
            "required": ["user_confirmed_token"],
        },
        "get_order_status": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID to check"},
            },
            "required": ["order_id"],
        },
        "list_orders": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "Optional status filter (e.g., 'pending', 'shipped')"},
                "limit": {"type": "integer", "description": "Maximum number of orders to return"},
            },
        },
        "process_payment": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID to process payment for"},
                "payment_method": {"type": "string", "description": "Payment method (e.g., 'credit_card', 'paypal')"},
                "user_confirmed_token": {"type": "string", "description": "Confirmation token from the user"},
            },
            "required": ["order_id", "user_confirmed_token"],
        },
        "get_payment_status": {
            "type": "object",
            "properties": {
                "payment_id": {"type": "string", "description": "The payment ID to check"},
            },
            "required": ["payment_id"],
        },
        "request_refund": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID to request refund for"},
                "reason": {"type": "string", "description": "Reason for the refund"},
                "user_confirmed_token": {"type": "string", "description": "Confirmation token from the user"},
            },
            "required": ["order_id", "user_confirmed_token"],
        },
        "get_merchant_balance": {
            "type": "object",
            "properties": {
                "merchant_id": {"type": "string", "description": "The merchant ID"},
            },
            "required": ["merchant_id"],
        },
        "get_merchant_ledger": {
            "type": "object",
            "properties": {
                "merchant_id": {"type": "string", "description": "The merchant ID"},
                "limit": {"type": "integer", "description": "Maximum number of entries to return"},
            },
            "required": ["merchant_id"],
        },
        "get_fulfillment_status": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID to check fulfillment for"},
            },
            "required": ["order_id"],
        },
        "list_fulfillments": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "Optional status filter"},
                "limit": {"type": "integer", "description": "Maximum number to return"},
            },
        },
        "create_fulfillment": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID to create fulfillment for"},
                "tracking_number": {"type": "string", "description": "Optional tracking number"},
            },
            "required": ["order_id"],
        },
        "update_fulfillment_status": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID"},
                "status": {"type": "string", "description": "New fulfillment status"},
            },
            "required": ["order_id", "status"],
        },
        "get_merchant_profile": {
            "type": "object",
            "properties": {
                "merchant_id": {"type": "string", "description": "The merchant ID"},
            },
            "required": ["merchant_id"],
        },
        "list_merchants": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Maximum number to return"},
            },
        },
        "create_product": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Product name"},
                "description": {"type": "string", "description": "Product description"},
                "price": {"type": "number", "description": "Product price"},
                "category": {"type": "string", "description": "Product category"},
            },
            "required": ["name", "price"],
        },
        "update_inventory": {
            "type": "object",
            "properties": {
                "variant_id": {"type": "string", "description": "The variant ID"},
                "quantity": {"type": "integer", "description": "New quantity"},
            },
            "required": ["variant_id", "quantity"],
        },
        "review_kyc": {
            "type": "object",
            "properties": {
                "merchant_id": {"type": "string", "description": "The merchant ID"},
                "decision": {"type": "string", "description": "KYC decision: 'approve' or 'reject'"},
                "user_confirmed_token": {"type": "string", "description": "Confirmation token from the user"},
            },
            "required": ["merchant_id", "decision", "user_confirmed_token"],
        },
        "request_payout": {
            "type": "object",
            "properties": {
                "merchant_id": {"type": "string", "description": "The merchant ID"},
                "amount": {"type": "number", "description": "Payout amount"},
                "user_confirmed_token": {"type": "string", "description": "Confirmation token from the user"},
            },
            "required": ["merchant_id", "amount", "user_confirmed_token"],
        },
    }
    return tool_schemas.get(tool_name, {"type": "object", "properties": {}})


def _summarize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Summarize tool args for debug logging (redact secrets)."""
    redacted_keys = {"user_confirmed_token", "token", "api_key", "access_key"}
    summary: dict[str, Any] = {}
    for key, value in (args or {}).items():
        if key in redacted_keys:
            summary[key] = "***redacted***"
            continue
        summary[key] = value if len(str(value)) <= 100 else f"{str(value)[:100]}..."
    return summary


def _coerce_product_card(item: dict[str, Any]) -> dict[str, Any] | None:
    """Map a raw product dict onto ``ProductCardView``-compatible props."""
    product_id = item.get("id") or item.get("product_id") or item.get("uuid")
    name = item.get("name") or item.get("title") or item.get("product_name")
    if not product_id or not name:
        return None
    price = _coerce_price(item.get("price", item)) or {"amount": 0.0, "currency": "NGN"}
    # Map all available fields from the backend response
    attributes = item.get("attributes", {})
    if not isinstance(attributes, dict):
        attributes = {}
    variants = item.get("variants", [])
    if not isinstance(variants, list):
        variants = []
    return {
        "id": str(product_id),
        "name": str(name),
        "merchant_id": str(item.get("merchant_id", "unknown")),
        "merchant_name": str(item.get("merchant_name", "Unknown merchant")),
        "category_id": item.get("category_id"),
        "category_name": item.get("category_name"),
        "price": price,
        "image_url": item.get("image_url", item.get("image")),
        "quantity_available": item.get("quantity_available", item.get("stock")),
        "status": str(item.get("status", "active") or "active"),
        # Additional backend fields
        "store_id": item.get("store_id"),
        "title": str(item.get("title", name)),
        "slug": item.get("slug"),
        "description": item.get("description"),
        "fulfillment_type": item.get("fulfillment_type"),
        "base_price_amount": price.get("amount") if isinstance(price, dict) else None,
        "base_price_currency": price.get("currency") if isinstance(price, dict) else "NGN",
        # Attributes
        "brand": attributes.get("brand"),
        "model": attributes.get("model"),
        "processor": attributes.get("processor"),
        "ram": attributes.get("ram"),
        "storage": attributes.get("storage"),
        "display_size": attributes.get("display_size"),
        "display_resolution": attributes.get("display_resolution"),
        "graphics": attributes.get("graphics"),
        "battery_life": attributes.get("battery_life"),
        "operating_system": attributes.get("operating_system"),
        "weight": attributes.get("weight"),
        "color": attributes.get("color"),
        "warranty": attributes.get("warranty"),
        # Variants summary
        "variants": [
            {
                "id": v.get("id"),
                "sku": v.get("sku"),
                "attributes": v.get("attributes", {}),
                "price_override_amount": v.get("price_override_amount"),
                "price_override_currency": v.get("price_override_currency"),
                "inventory_policy": v.get("inventory_policy"),
                "quantity_available": v.get("quantity_available"),
            }
            for v in variants
        ] if variants else None,
        # Timestamps
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
    }


def _extract_list(data: Any) -> list[Any]:
    """Extract a list payload from common backend envelopes."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("items", "results", "products", "orders", "entries", "categories", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    return []


def _coerce_price(value: Any, default_currency: str = "NGN") -> dict[str, Any] | None:
    """Coerce backend price shapes into ``PriceView``-compatible props."""
    if value is None:
        return None
    if isinstance(value, dict):
        if "amount" in value:
            try:
                return {
                    "amount": float(value.get("amount", 0) or 0),
                    "currency": str(value.get("currency", default_currency) or default_currency),
                }
            except (TypeError, ValueError):
                return None
        for key in ("price", "price_amount", "unit_price", "total", "balance", "available", "subtotal"):
            if key in value and isinstance(value[key], (int, float)):
                return {"amount": float(value[key]), "currency": str(value.get("currency", default_currency))}
        for key in ("price_minor", "amount_minor", "amount_cents"):
            if key in value and isinstance(value[key], (int, float)):
                return {"amount": float(value[key]) / 100.0, "currency": str(value.get("currency", default_currency))}
        return None
    if isinstance(value, (int, float)):
        return {"amount": float(value), "currency": default_currency}
    return None


def _unwrap_tool_data(data: Any) -> Any:
    """Unwrap common backend response envelopes into displayable payload."""
    if data is None:
        return None
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "result", "payload"):
            value = data.get(key)
            if isinstance(value, (list, dict)) and value:
                return value
        return data
    return data


def _coerce_product_detail(item: dict[str, Any]) -> dict[str, Any]:
    """Map a raw product dict onto ``ProductDetailView``-compatible props."""
    card = _coerce_product_card(item) or {}
    detail: dict[str, Any] = dict(card) if card else {
        "id": str(item.get("id", item.get("product_id", "unknown"))),
        "name": str(item.get("name", item.get("title", "Unknown Product"))),
        "merchant_id": str(item.get("merchant_id", "unknown")),
        "merchant_name": str(item.get("merchant_name", "Unknown merchant")),
        "price": _coerce_price(item.get("price", item)) or {"amount": 0.0, "currency": "NGN"},
        "status": str(item.get("status", "active") or "active"),
    }
    detail["description"] = item.get("description")
    variants = item.get("variants", item.get("product_variants", []))
    detail["variants"] = variants if isinstance(variants, list) else []
    attributes = item.get("attributes", {})
    detail["attributes"] = attributes if isinstance(attributes, dict) else {}
    detail["created_at"] = item.get("created_at")
    detail["updated_at"] = item.get("updated_at")
    return detail


def _coerce_category(item: Any) -> dict[str, Any] | None:
    """Map a raw category dict onto ``CategoryView``-compatible props."""
    if not isinstance(item, dict):
        return None
    category_id = item.get("id") or item.get("category_id") or item.get("slug")
    name = item.get("name") or item.get("title")
    if not category_id or not name:
        return None
    return {
        "id": str(category_id),
        "name": str(name),
        "slug": str(item.get("slug", category_id)),
        "parent_id": item.get("parent_id"),
        "product_count": item.get("product_count", item.get("count")),
    }


def _coerce_order_summary(item: Any) -> dict[str, Any] | None:
    """Map a raw order dict onto ``OrderSummaryView``-compatible props."""
    if not isinstance(item, dict):
        return None
    order_id = item.get("id") or item.get("order_id")
    if not order_id:
        return None
    total = _coerce_price(item.get("total", item)) or {"amount": 0.0, "currency": "NGN"}
    return {
        "id": str(order_id),
        "status": str(item.get("status", "unknown") or "unknown"),
        "total": total,
        "item_count": int(item.get("item_count", 0) or 0),
        "created_at": str(item.get("created_at", "") or ""),
        "merchant_id": str(item.get("merchant_id", "unknown")),
        "merchant_name": str(item.get("merchant_name", "Unknown merchant")),
    }


def _coerce_order_item(item: Any) -> dict[str, Any] | None:
    """Map a raw order line onto ``OrderItemView``-compatible props."""
    if not isinstance(item, dict):
        return None
    variant_id = item.get("variant_id") or item.get("sku") or item.get("id")
    product_id = item.get("product_id") or item.get("id")
    name = item.get("product_name") or item.get("name")
    if not variant_id or not product_id or not name:
        return None
    unit = _coerce_price(item.get("unit_price", item.get("price", item))) or {"amount": 0.0, "currency": "NGN"}
    return {
        "variant_id": str(variant_id),
        "product_id": str(product_id),
        "product_name": str(name),
        "variant_name": str(item.get("variant_name", name)),
        "sku": str(item.get("sku", variant_id)),
        "quantity": int(item.get("quantity", 1) or 1),
        "unit_price": unit,
        "line_total": _coerce_price(item.get("line_total")) or unit,
    }


def _coerce_order_detail(item: dict[str, Any]) -> dict[str, Any] | None:
    """Map a raw order dict onto ``OrderDetailView``-compatible props."""
    summary = _coerce_order_summary(item)
    if summary is None:
        return None
    items = [
        c for raw in (item.get("items", []) or [])
        if (c := _coerce_order_item(raw)) is not None
    ]
    return {
        **summary,
        "items": items,
        "subtotal": _coerce_price(item.get("subtotal")) or summary["total"],
        "shipping_address": item.get("shipping_address"),
        "payment_status": item.get("payment_status"),
        "fulfillment_status": item.get("fulfillment_status"),
        "updated_at": item.get("updated_at"),
    }


def _coerce_balance(item: dict[str, Any]) -> dict[str, Any] | None:
    """Map a raw balance dict onto ``BalanceView``-compatible props."""
    merchant_id = item.get("merchant_id")
    if not merchant_id:
        return None
    avail = _coerce_price(item.get("available", item)) or {"amount": 0.0, "currency": "NGN"}
    pend = _coerce_price(item.get("pending")) or {"amount": 0.0, "currency": avail["currency"]}
    return {
        "merchant_id": str(merchant_id),
        "merchant_name": str(item.get("merchant_name", "Unknown merchant")),
        "available": avail,
        "pending": pend,
        "currency": str(item.get("currency", avail["currency"])),
        "updated_at": item.get("updated_at"),
    }


def _coerce_ledger_entry(item: Any) -> dict[str, Any] | None:
    """Map a raw ledger dict onto ``LedgerEntryView``-compatible props."""
    if not isinstance(item, dict):
        return None
    entry_id = item.get("id") or item.get("entry_id")
    if not entry_id:
        return None
    amount = _coerce_price(item.get("amount", item)) or {"amount": 0.0, "currency": "NGN"}
    return {
        "id": str(entry_id),
        "entry_type": str(item.get("entry_type", item.get("type", "unknown"))),
        "amount": amount,
        "balance_after": _coerce_price(item.get("balance_after")),
        "order_id": item.get("order_id"),
        "payment_id": item.get("payment_id"),
        "description": item.get("description"),
        "created_at": str(item.get("created_at", "") or ""),
        "entry_group_id": item.get("entry_group_id"),
    }


def _coerce_cart(data: Any) -> dict[str, Any]:
    """Map tool data onto ``CartView``-compatible props."""
    if not isinstance(data, dict):
        return None
    product_id = data.get("id") or data.get("product_id") or data.get("uuid")
    name = data.get("name") or data.get("title") or data.get("product_name")
    if not product_id or not name:
        return None
    price = _coerce_price(data.get("price", data))
    currency = (price or {}).get("currency", data.get("currency", "NGN"))
    if price is None:
        price = {"amount": 0.0, "currency": str(currency or "NGN")}
    return {
        "id": str(product_id),
        "name": str(name),
        "merchant_id": str(data.get("merchant_id", data.get("merchantId", "unknown"))),
        "merchant_name": str(data.get("merchant_name", data.get("merchantName", data.get("store_name", "Unknown merchant")))),
        "category_id": data.get("category_id") or data.get("categoryId"),
        "category_name": data.get("category_name", data.get("categoryName", data.get("category"))),
        "price": price,
        "image_url": data.get("image_url", data.get("imageUrl", data.get("image", data.get("thumbnail")))),
        "quantity_available": data.get("quantity_available", data.get("quantityAvailable", data.get("stock", data.get("quantity")))),
        "status": str(data.get("status", "active") or "active"),
    }


def _normalize_directive_props(component: Any, data: Any) -> dict[str, Any] | None:
    """Build validated props for a UI directive component. Returns None when there is nothing displayable (no DB data), so the
    caller skips emitting a directive. Otherwise returns a dict that validates
    against the matching canonical view DTO — the frontend mirrors these as
    Zod schemas, so props must be DTO-compatible, never raw backend shapes."""
    from schemas.components import ComponentName

    comp = str(component.value if isinstance(component, ComponentName) else component)

    if data is None:
        return None
    if isinstance(data, list) and len(data) == 0:
        # Empty result sets still render (empty grid / list), except for
        # single-object components where empty means "nothing to show".
        if comp in (ComponentName.ProductDetail, ComponentName.OrderDetail,
                    ComponentName.OrderConfirmation, ComponentName.MerchantBalanceCard):
            return None
        return {"items": []}
    if isinstance(data, dict) and not data:
        return None

    # ── Catalog: ProductGrid ← list of products ──
    if comp == ComponentName.ProductGrid:
        items_raw = data if isinstance(data, list) else _extract_list(data)
        cards: list[dict[str, Any]] = []
        for item in items_raw:
            if isinstance(item, dict):
                card = _coerce_product_card(item)
                if card is not None:
                    cards.append(card)
        if not cards and not items_raw:
            return {"items": []}
        # If coercion dropped everything (unexpected shape), fall back to raw
        # items so the frontend still has something to render/debug.
        if not cards and items_raw:
            return {"items": items_raw}
        return {"items": cards, "total": len(cards)}

    # ── Catalog: ProductDetail ← single product ──
    if comp == ComponentName.ProductDetail:
        obj = data[0] if isinstance(data, list) and data else data
        if not isinstance(obj, dict):
            return None
        detail = _coerce_product_detail(obj)
        return detail

    # ── Catalog: CategoryList ← list of categories ──
    if comp == ComponentName.CategoryList:
        items_raw = data if isinstance(data, list) else _extract_list(data)
        cats: list[dict[str, Any]] = []
        for item in items_raw:
            coerced = _coerce_category(item)
            if coerced is not None:
                cats.append(coerced)
        if not cats and items_raw:
            return {"items": items_raw}
        return {"items": cats}

    # ── Cart: CartSummary ← CartView dict ──
    if comp == ComponentName.CartSummary:
        cart = _coerce_cart(data)
        return cart

    # ── Orders: OrderList ← list of orders ──
    if comp == ComponentName.OrderList:
        items_raw = data if isinstance(data, list) else _extract_list(data)
        orders: list[dict[str, Any]] = []
        for item in items_raw:
            coerced = _coerce_order_summary(item)
            orders.append(coerced if coerced is not None else item)
        # Provide both `items` (generic) and `orders` (explicit) keys so the
        # frontend can consume either without breaking the contract.
        return {"items": orders, "orders": orders, "total": len(orders)}

    # ── Orders: OrderDetail / OrderConfirmation ← single order dict ──
    if comp in (ComponentName.OrderDetail, ComponentName.OrderConfirmation):
        obj = data[0] if isinstance(data, list) and data else data
        if not isinstance(obj, dict):
            return None
        coerced = _coerce_order_detail(obj)
        return coerced if coerced is not None else obj

    # ── Merchant: BalanceCard ← BalanceView dict ──
    if comp == ComponentName.MerchantBalanceCard:
        if not isinstance(data, dict):
            return None
        coerced = _coerce_balance(data)
        return coerced if coerced is not None else data

    # ── Merchant: LedgerTable ← list of ledger entries ──
    if comp == ComponentName.LedgerTable:
        items_raw = data if isinstance(data, list) else _extract_list(data)
        entries: list[dict[str, Any]] = []
        for item in items_raw:
            coerced = _coerce_ledger_entry(item)
            entries.append(coerced if coerced is not None else item)
        return {"items": entries, "entries": entries, "total": len(entries)}

    # ── Fallback: MerchantProfile / FulfillmentTracker / others ──
    # Pass through the raw payload wrapped appropriately. These components
    # have no strict view DTO in schemas/, so raw data is the best we can do.
    if isinstance(data, list):
        return {"items": data}
    if isinstance(data, dict):
        return data
    return {"value": data}


def _summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    return {"status": result.get("status"), "tool_name": result.get("tool_name"), "has_data": "data" in result}


# Mapping of identifier types to the tools that can use them
_IDENTIFIER_TOOL_MAP = {
    "merchant_id": "get_merchant_profile",
    "store_id": "get_merchant_profile",
    "product_id": "get_product_detail",
    "order_id": "get_order_status",
    "payment_id": "get_payment_status",
    "category_id": "get_categories",
}


def _extract_identifiers(data: Any) -> dict[str, str]:
    """Extract identifier key-value pairs from tool result data.
    
    Returns a dict of identifier type -> value found in the data.
    Only extracts known identifier types and ignores them from user-facing output.
    """
    identifiers = {}
    if not isinstance(data, dict):
        return identifiers
    
    # Check for known identifier keys
    identifier_keys = {
        "merchant_id", "store_id", "product_id", "order_id",
        "payment_id", "category_id", "variant_id"
    }
    
    for key in identifier_keys:
        if key in data and data[key]:
            identifiers[key] = str(data[key])
    
    # Also check nested attributes dict
    attributes = data.get("attributes", {})
    if isinstance(attributes, dict):
        for key in identifier_keys:
            if key in attributes and attributes[key]:
                identifiers[key] = str(attributes[key])
    
    # Check variants for product IDs
    variants = data.get("variants", [])
    if isinstance(variants, list) and variants:
        for variant in variants:
            if isinstance(variant, dict) and "product_id" in variant and variant["product_id"]:
                identifiers["product_id"] = str(variant["product_id"])
                break
    
    return identifiers


def _get_tool_for_identifier(identifier_type: str) -> str | None:
    """Map an identifier type to the appropriate tool name."""
    return _IDENTIFIER_TOOL_MAP.get(identifier_type)


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
            if count == 0:
                if query:
                    return f"I couldn't find any products matching '{query}'. Try a different search term."
                return "I couldn't find any products. Try a different search."
            # Build a more detailed summary with product details
            product_details = []
            for item in items[:5]:  # Limit to first 5 products
                # Try multiple fields for product name
                name = (
                    item.get("name")
                    or item.get("title")
                    or item.get("product_name")
                    or (f"{item.get('category_name', '')} product" if item.get("category_name") else None)
                    or "Unknown Product"
                )
                # Get price info
                price_amount = item.get("price", {}).get("amount") if isinstance(item.get("price"), dict) else None
                price_currency = item.get("price", {}).get("currency", "") if isinstance(item.get("price"), dict) else ""
                price_str = f"{price_currency} {price_amount}" if price_amount is not None else ""
                # Get additional details
                merchant_name = item.get("merchant_name", "")
                category_name = item.get("category_name", "")
                quantity = item.get("quantity_available")
                # Build description parts
                desc_parts = [name]
                if price_str:
                    desc_parts.append(f"Price: {price_str}")
                if merchant_name:
                    desc_parts.append(f"by {merchant_name}")
                if category_name:
                    desc_parts.append(f"Category: {category_name}")
                if quantity is not None:
                    desc_parts.append(f"Stock: {quantity}")
                product_details.append(" | ".join(desc_parts))
            summary_parts = [f"Found {count} product{'s' if count != 1 else ''}"]
            if query:
                summary_parts.append(f"matching '{query}'")
            summary_parts.append(":")
            summary_parts.append("; ".join(product_details))
            if count > 5:
                summary_parts.append(f"... and {count - 5} more products")
            return " ".join(summary_parts)
        return "Here are the products."

    if tool_name == "get_product_detail":
        if isinstance(data, dict):
            name = data.get("name", "")
            price_amount = data.get("price", {}).get("amount") if isinstance(data.get("price"), dict) else None
            price_currency = data.get("price", {}).get("currency", "") if isinstance(data.get("price"), dict) else ""
            description = data.get("description", "")
            variants = data.get("variants", [])
            details_parts = [f"Here's the details for {name}"]
            if price_amount is not None:
                details_parts.append(f"- Price: {price_currency} {price_amount}")
            if description:
                # Truncate long descriptions
                desc_short = description[:100] + "..." if len(description) > 100 else description
                details_parts.append(f"Description: {desc_short}")
            if variants:
                details_parts.append(f"Available in {len(variants)} variant{'s' if len(variants) != 1 else ''}")
            return ". ".join(details_parts) + "."
        return "Here are the product details."

    if tool_name == "get_categories":
        if isinstance(data, list):
            count = len(data)
            if count == 0:
                return "No categories found."
            # Include category names for a more informative summary
            category_names = [c.get("name", "Unknown") for c in data[:5] if isinstance(c, dict)]
            summary = f"Found {count} categor{'ies' if count != 1 else 'y'}"
            if category_names:
                summary += ": " + ", ".join(category_names)
                if count > 5:
                    summary += f", and {count - 5} more"
            return summary + "."
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
            # Calculate total if available
            total_amount = data.get("total", {}).get("amount") if isinstance(data.get("total"), dict) else None
            total_currency = data.get("total", {}).get("currency", "") if isinstance(data.get("total"), dict) else ""
            item_names = [i.get("name", i.get("product_name", "Item")) for i in items[:3] if isinstance(i, dict)]
            summary = f"Your cart has {count} item{'s' if count != 1 else ''}"
            if item_names:
                summary += ": " + ", ".join(item_names)
                if count > 3:
                    summary += f", and {count - 3} more"
            if total_amount is not None:
                summary += f". Total: {total_currency} {total_amount}"
            return summary + "."
        return "Your cart is empty."

    if tool_name == "add_to_cart":
        product_name = data.get("name", data.get("product_name", "")) if isinstance(data, dict) else ""
        if product_name:
            return f"Added '{product_name}' to your cart."
        return "Item added to your cart."

    if tool_name == "initiate_checkout":
        order_id = data.get("order_id", "") if isinstance(data, dict) else ""
        total = data.get("total", {}).get("amount") if isinstance(data.get("total"), dict) else None
        if order_id and total:
            return f"Checkout initiated for order {order_id}. Total amount: {total}."
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
            if count == 0:
                return "You have no orders yet."
            # Include order IDs and statuses for a more informative summary
            order_details = []
            for order in data[:5]:
                if isinstance(order, dict):
                    order_id = order.get("order_id", order.get("id", ""))
                    status = order.get("status", "")
                    if order_id and status:
                        order_details.append(f"{order_id} ({status})")
                    elif order_id:
                        order_details.append(str(order_id))
            summary = f"Found {count} order{'s' if count != 1 else ''}"
            if order_details:
                summary += ": " + "; ".join(order_details)
                if count > 5:
                    summary += f"... and {count - 5} more"
            return summary + "."
        return "Here are your orders."

    if tool_name == "get_merchant_balance":
        balance = data.get("balance", "") if isinstance(data, dict) else ""
        currency = data.get("currency", "") if isinstance(data, dict) else ""
        if balance:
            return f"Current merchant balance: {currency} {balance}."
        return "Here's the merchant balance."

    if tool_name == "get_merchant_ledger":
        if isinstance(data, list):
            count = len(data)
            if count == 0:
                return "No ledger entries found."
            return f"Found {count} ledger entr{'ies' if count != 1 else 'y'}."
        return "Here's the merchant ledger."

    if tool_name == "get_fulfillment_status":
        status = data.get("status", "") if isinstance(data, dict) else ""
        order_id = data.get("order_id", "") if isinstance(data, dict) else ""
        if status and order_id:
            return f"Fulfillment status for order {order_id}: {status}."
        if status:
            return f"Fulfillment status: {status}."
        return "Here's the fulfillment status."

    return None

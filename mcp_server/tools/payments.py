"""Payment tools — process_payment, get_payment_status, request_refund."""

from __future__ import annotations

from typing import Any

from mcp_server.backend_client import BackendClient, ToolError
from mcp_server.guards import ConfirmationGate, ConfirmationRequiredError
from mcp_server.idempotency import derive_idempotency_key
from schemas.payments import PaymentView, PriceView, RefundView


def _map_price(amount: float | None, currency: str | None) -> PriceView:
    return PriceView(amount=float(amount or 0), currency=currency or "NGN")


def _map_payment(p: dict) -> PaymentView:
    return PaymentView(
        id=str(p.get("id", "")),
        order_id=str(p.get("order_id", "")),
        status=p.get("status", ""),
        amount=_map_price(p.get("amount"), p.get("currency")),
        provider=p.get("provider"),
        provider_reference=p.get("provider_reference"),
        created_at=p.get("created_at"),
        updated_at=p.get("updated_at"),
    )


def _map_refund(r: dict) -> RefundView:
    return RefundView(
        id=str(r.get("id", "")),
        payment_id=str(r.get("payment_id", "")),
        order_id=str(r.get("order_id", "")),
        status=r.get("status", ""),
        amount=_map_price(r.get("amount"), r.get("currency")),
        reason=r.get("reason"),
        created_at=r.get("created_at"),
    )


def register_payments_tools(
    server: Any,
    client: BackendClient,
    gate: ConfirmationGate,
    session: Any,
) -> None:
    """Register payment tools."""

    @server.tool(
        name="process_payment",
        description="Process payment for an order. HIGH risk — financial. Requires user confirmation.",
    )
    async def process_payment(
        order_id: str,
        user_confirmed_token: str | None = None,
        token: str | None = None,
    ) -> dict:
        """Process payment for an order."""
        session_id = session.session_id
        turn_id = session.current_turn_id
        try:
            await gate.check_confirmation(session_id, turn_id, "process_payment", user_confirmed_token)
        except ConfirmationRequiredError as e:
            raise ToolError(str(e), error_kind="confirmation_required") from e

        payment_body: dict[str, Any] = {"order_id": order_id}
        idempotency_key = derive_idempotency_key(session.user_id, turn_id, "process_payment", payment_body)

        body = await client.call(
            "POST", "/payments/process",
            token=token,
            json_body=payment_body,
            idempotency_key=idempotency_key,
        )
        return _map_payment(body).model_dump()

    @server.tool(
        name="get_payment_status",
        description="Get payment status. Read-only.",
    )
    async def get_payment_status(
        payment_id: str,
        token: str | None = None,
    ) -> dict:
        """Get payment status by ID."""
        body = await client.call(
            "GET", "/payments/{payment_id}",
            token=token, path_params={"payment_id": payment_id},
        )
        return _map_payment(body).model_dump()

    @server.tool(
        name="list_payments",
        description="List payments with optional filtering. Read-only.",
    )
    async def list_payments(
        skip: int = 0,
        limit: int = 50,
        order_id: Optional[str] = None,
        status: Optional[str] = None,
        token: str | None = None,
    ) -> list[dict]:
        """List payments, optionally filtered by order ID or status."""
        params: dict[str, Any] = {"skip": skip, "limit": limit}
        if order_id:
            params["order_id"] = order_id
        if status:
            params["status"] = status
        body = await client.call(
            "GET", "/payments/", token=token, params=params,
        )
        items = body.get("items", body) if isinstance(body, dict) else body
        if not isinstance(items, list):
            items = [items] if isinstance(items, dict) else []
        return [_map_payment(p).model_dump() for p in items]

    @server.tool(
        name="request_refund",
        description="Request a refund. HIGH risk — financial. Requires user confirmation.",
    )
    async def request_refund(
        payment_id: str,
        amount: float | None = None,
        reason: str | None = None,
        user_confirmed_token: str | None = None,
        token: str | None = None,
    ) -> dict:
        """Request a refund for a payment."""
        session_id = session.session_id
        turn_id = session.current_turn_id
        try:
            await gate.check_confirmation(session_id, turn_id, "request_refund", user_confirmed_token)
        except ConfirmationRequiredError as e:
            raise ToolError(str(e), error_kind="confirmation_required") from e

        refund_body: dict[str, Any] = {}
        if amount is not None:
            refund_body["amount"] = amount
        if reason:
            refund_body["reason"] = reason

        idempotency_key = derive_idempotency_key(session.user_id, turn_id, "request_refund", refund_body)

        body = await client.call(
            "POST", "/payments/{payment_id}/refund",
            token=token,
            json_body=refund_body,
            path_params={"payment_id": payment_id},
            idempotency_key=idempotency_key,
        )
        return _map_refund(body).model_dump()

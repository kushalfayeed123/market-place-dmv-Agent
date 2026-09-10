"""Admin tools — review_kyc, request_payout. Both HIGH risk, require confirmation."""

from __future__ import annotations

from typing import Any

from mcp_server.backend_client import BackendClient, ToolError
from mcp_server.guards import ConfirmationGate, ConfirmationRequiredError
from mcp_server.idempotency import derive_idempotency_key


def register_admin_tools(
    server: Any,
    client: BackendClient,
    gate: ConfirmationGate,
    session: Any,
) -> None:
    """Register admin-only tools."""

    @server.tool(
        name="review_kyc",
        description="Review/approve/reject merchant KYC. HIGH risk — admin-only. Requires confirmation.",
    )
    async def review_kyc(
        merchant_id: str,
        kyc_status: str,
        reason: str | None = None,
        user_confirmed_token: str | None = None,
        token: str | None = None,
    ) -> dict:
        """Review a merchant's KYC status (admin)."""
        try:
            await gate.check_confirmation(
                session.session_id, session.current_turn_id, "review_kyc", user_confirmed_token,
            )
        except ConfirmationRequiredError as e:
            raise ToolError(str(e), error_kind="confirmation_required") from e

        body_data: dict[str, Any] = {"kyc_status": kyc_status}
        if reason:
            body_data["reason"] = reason

        idempotency_key = derive_idempotency_key(
            session.user_id, session.current_turn_id, "review_kyc", body_data,
        )
        return await client.call(
            "PATCH", "/admin/merchants/{merchant_id}/kyc",
            token=token,
            json_body=body_data,
            path_params={"merchant_id": merchant_id},
            idempotency_key=idempotency_key,
        )

    @server.tool(
        name="request_payout",
        description="Request a payout. HIGH risk — financial. Requires confirmation + MFA step-up.",
    )
    async def request_payout(
        merchant_id: str,
        amount: float,
        currency: str = "NGN",
        user_confirmed_token: str | None = None,
        token: str | None = None,
    ) -> dict:
        """Request a payout (merchant owner / admin)."""
        try:
            await gate.check_confirmation(
                session.session_id, session.current_turn_id, "request_payout", user_confirmed_token,
            )
        except ConfirmationRequiredError as e:
            raise ToolError(str(e), error_kind="confirmation_required") from e

        body_data = {"amount": amount, "currency": currency}
        idempotency_key = derive_idempotency_key(
            session.user_id, session.current_turn_id, "request_payout", body_data,
        )
        return await client.call(
            "POST", "/merchants/{merchant_id}/payouts",
            token=token,
            json_body=body_data,
            path_params={"merchant_id": merchant_id},
            idempotency_key=idempotency_key,
        )

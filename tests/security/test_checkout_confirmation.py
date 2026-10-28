import os

import fakeredis.aioredis as fakeredis
import pytest

from gateway.agent import Agent
from gateway.debug import DebugTracer
from gateway.session import Session
from mcp_server.guards import ConfirmationGate

HERE = os.path.dirname(__file__)
SYSTEM_PROMPT = open(
    os.path.join(HERE, "..", "..", "prompts", "system.md"),
    encoding="utf-8",
).read()


class NoCallModel:
    provider_name = "test"
    model_name = "test"

    async def stream(self, **_kwargs):
        raise AssertionError("A confirmed checkout should not need another model call")
        yield  # Make this an async generator.

    async def classify(self, **_kwargs):
        raise AssertionError("Authenticated confirmations should not be classified")


class CheckoutBackend:
    def __init__(self):
        self.calls = []

    async def call(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        return {
            "id": "order-1",
            "order_number": "ORD-001",
            "status": "pending",
            "currency": "NGN",
            "total_amount": 12000,
            "item_count": 2,
            "items": [],
        }


@pytest.mark.asyncio
async def test_confirm_replays_checkout_from_session_cart():
    redis_client = fakeredis.FakeRedis(decode_responses=True)
    backend = CheckoutBackend()
    agent = Agent(
        backend_client=backend,
        gate=ConfirmationGate(redis_client, namespace="test"),
        model_backend=NoCallModel(),
        debug_tracer=DebugTracer(redis_client=None, namespace="test", enabled=False),
        system_prompt=SYSTEM_PROMPT,
    )
    session = Session(
        session_id="checkout-confirmation",
        user_token="buyer-token",
        user_id="buyer-1",
        role="buyer",
    )
    session.set_cart({
        "items": [
            {"variant_id": "variant-1", "quantity": 2},
        ],
    })

    first_events = [
        event
        async for event in agent._handle_event(
            {
                "type": "tool_call",
                "tool_call": {
                    "id": "checkout-call",
                    "name": "initiate_checkout",
                    "arguments": {
                        "items": [{"variant_id": "incorrect-model-item", "quantity": 99}],
                        "payment_method": "pay_on_delivery",
                    },
                },
            },
            session,
            1,
            None,
            [],
        )
    ]
    assert any(event.get("type") == "awaiting_confirmation" for event in first_events)
    assert session.pending_confirmation == {
        "tool_name": "initiate_checkout",
        "arguments": {
            "items": [{"variant_id": "incorrect-model-item", "quantity": 99}],
            "payment_method": "pay_on_delivery",
        },
    }
    assert backend.calls == []

    restored_session = Session.from_dict(session.to_dict())
    confirmed_events = [
        event
        async for event in agent.run_turn(
            restored_session,
            "Confirmed",
            confirmed_token="confirmed-by-click",
        )
    ]

    assert len(backend.calls) == 1
    method, path, kwargs = backend.calls[0]
    assert (method, path) == ("POST", "/orders/checkout")
    assert kwargs["json_body"]["items"] == [{"variant_id": "variant-1", "quantity": 2}]
    assert kwargs["json_body"]["payment_method"] == "pay_on_delivery"
    assert restored_session.pending_confirmation is None
    assert restored_session.get_cart()["items"] == []
    assert any(event.get("type") == "text" for event in confirmed_events)
    assert any(
        event.get("type") == "ui_directive"
        and event["directive"]["component"] == "PaymentCapturePanel"
        for event in confirmed_events
    )

    await redis_client.aclose()

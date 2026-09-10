"""Idempotency replay tests — same tool call twice with the same derived key = one backend effect."""

import pytest
from mcp_server.idempotency import derive_idempotency_key


class TestIdempotencyKey:

    def test_same_request_same_key(self):
        """Same user + turn + tool + body = same key."""
        key1 = derive_idempotency_key("user-1", 5, "initiate_checkout", {"items": [{"variant_id": "v1", "quantity": 2}]})
        key2 = derive_idempotency_key("user-1", 5, "initiate_checkout", {"items": [{"variant_id": "v1", "quantity": 2}]})
        assert key1 == key2

    def test_different_turn_different_key(self):
        """Different turn = different key (new transaction)."""
        key1 = derive_idempotency_key("user-1", 5, "initiate_checkout", {"items": [{"variant_id": "v1", "quantity": 2}]})
        key2 = derive_idempotency_key("user-1", 6, "initiate_checkout", {"items": [{"variant_id": "v1", "quantity": 2}]})
        assert key1 != key2

    def test_different_user_different_key(self):
        """Different user = different key."""
        key1 = derive_idempotency_key("user-1", 5, "initiate_checkout", {"items": [{"variant_id": "v1", "quantity": 2}]})
        key2 = derive_idempotency_key("user-2", 5, "initiate_checkout", {"items": [{"variant_id": "v1", "quantity": 2}]})
        assert key1 != key2

    def test_different_body_different_key(self):
        """Different body = different key."""
        key1 = derive_idempotency_key("user-1", 5, "initiate_checkout", {"items": [{"variant_id": "v1", "quantity": 2}]})
        key2 = derive_idempotency_key("user-1", 5, "initiate_checkout", {"items": [{"variant_id": "v1", "quantity": 3}]})
        assert key1 != key2

    def test_key_is_sha256_hex(self):
        """Key is a 64-character hex string (SHA-256)."""
        key = derive_idempotency_key("user-1", 1, "initiate_checkout", {"a": 1})
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_key_stable_across_key_order(self):
        """Canonicalization means key order doesn't matter."""
        body1 = {"b": 2, "a": 1}
        body2 = {"a": 1, "b": 2}
        key1 = derive_idempotency_key("user-1", 1, "initiate_checkout", body1)
        key2 = derive_idempotency_key("user-1", 1, "initiate_checkout", body2)
        assert key1 == key2

    def test_none_body_handled(self):
        """None body is handled gracefully."""
        key = derive_idempotency_key("user-1", 1, "initiate_checkout", None)
        assert len(key) == 64

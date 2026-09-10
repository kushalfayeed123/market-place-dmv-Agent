"""Directive tests — unknown components / malformed props must fall back safely."""

import pytest
from schemas.components import UIDirective, ComponentName


class TestDirectiveSafety:

    def test_unknown_component_rejected(self):
        """An unknown component name must be rejected by validation."""
        with pytest.raises(ValueError):
            UIDirective(
                component="NotARealComponent",  # type: ignore
                props={},
                correlation_id="abc",
            )

    def test_empty_props_allowed(self):
        """A directive with empty props is valid (some components need no props)."""
        d = UIDirective(
            component=ComponentName.ErrorMessage,
            props={},
            correlation_id="abc",
        )
        assert d.props == {}

    def test_directive_type_field_fixed(self):
        """The type field must always be 'ui_directive'."""
        d = UIDirective(component=ComponentName.SignInPrompt, props={}, correlation_id="x")
        assert d.type == "ui_directive"

    def test_directive_version_is_1(self):
        """The version field must be 1."""
        d = UIDirective(component=ComponentName.SignInPrompt, props={}, correlation_id="x")
        assert d.version == 1

    def test_correlation_id_required(self):
        """A correlation_id is required for tracing."""
        with pytest.raises(Exception):
            UIDirective(component=ComponentName.SignInPrompt, props={}, correlation_id="")  # type: ignore

    def test_all_components_have_valid_names(self):
        """Every ComponentName must be a valid string."""
        for name in ComponentName:
            assert isinstance(name.value, str)
            assert len(name.value) > 0

    def test_directive_serializes_to_json(self):
        """A directive must serialize to valid JSON."""
        import json
        d = UIDirective(
            component=ComponentName.CartSummary,
            props={"items": [], "total": {"amount": 0, "currency": "NGN"}},
            correlation_id="test-123",
        )
        serialized = json.dumps(d.model_dump())
        assert "ui_directive" in serialized
        assert "CartSummary" in serialized

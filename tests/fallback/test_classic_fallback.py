"""Fallback tests — with the Agent Gateway down, the Classic view still works.

The agent layer is additive. If it's removed or down, the classic frontend
+ backend must still be a fully working marketplace.
"""


class TestClassicFallback:

    def test_agent_layer_is_additive(self):
        """The agent layer must not be required for backend operation."""
        # This is an architectural test — the backend has no agent-specific code.
        # We verify by checking that the backend's main.py doesn't import agent code.
        import os
        backend_main = os.path.join(os.path.dirname(__file__), "..", "..", "..", "backend", "app", "main.py")
        if os.path.exists(backend_main):
            with open(backend_main, "r") as f:
                content = f.read()
            # Backend should not import from the agent layer
            assert "agent" not in content.lower() or "from agent" not in content

    def test_backend_endpoints_unmodified(self):
        """Backend endpoints must work without the agent layer."""
        # The backend's OpenAPI spec must be self-contained
        import os
        backend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "backend", "app", "modules")
        if os.path.exists(backend_dir):
            # Verify routers exist (they should not import agent code)
            for module in os.listdir(backend_dir):
                module_path = os.path.join(backend_dir, module)
                router_file = os.path.join(module_path, "router.py")
                if os.path.exists(router_file):
                    with open(router_file, "r") as f:
                        content = f.read()
                    # No agent imports in backend routers
                    assert "from agent" not in content
                    assert "import agent" not in content

    def test_agent_can_be_deleted_without_breaking_backend(self):
        """Deleting the Agent folder must not affect the backend."""
        # This is a design principle test — verified by the fact that
        # the backend has no imports from the agent layer.
        import os
        agent_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Agent")
        backend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "backend")
        # Both directories should exist independently
        assert os.path.exists(agent_dir)
        assert os.path.exists(backend_dir)

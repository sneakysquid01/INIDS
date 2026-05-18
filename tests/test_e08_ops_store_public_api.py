"""E-08: OpsStore.list_pending_actions() public method + api_actions_pending uses it."""
import pytest
import inspect

from src.ops_store import OpsStore


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    return OpsStore(db_path)


class TestOpsStorePublicAPI:
    def test_list_pending_actions_exists(self):
        assert hasattr(OpsStore, "list_pending_actions")
        assert callable(OpsStore.list_pending_actions)

    def test_list_pending_actions_is_public(self):
        """Method name must not start with underscore."""
        assert not "list_pending_actions".startswith("_")

    def test_list_pending_actions_returns_list(self, store):
        result = store.list_pending_actions()
        assert isinstance(result, list)

    def test_list_pending_actions_empty_db(self, store):
        result = store.list_pending_actions()
        assert result == []

    def test_list_pending_actions_with_pending_action(self, store):
        store.save_action({
            "action": "block",
            "target": "5.6.7.8",
            "reason": "test pending",
            "status": "pending_approval",
            "action_type": "approve_block",
        })
        results = store.list_pending_actions()
        assert len(results) == 1
        assert results[0]["status"] == "pending_approval"

    def test_list_pending_actions_excludes_active(self, store):
        store.save_action({
            "action": "block",
            "target": "1.2.3.4",
            "reason": "active action",
            "status": "active",
            "action_type": "block",
        })
        results = store.list_pending_actions()
        assert results == []

    def test_list_pending_actions_limit_respected(self, store):
        # Insert 5 pending_approval actions with unique targets
        for i in range(5):
            store._execute(
                "INSERT INTO actions (action, target, reason, action_id, ip, action_type, status, created_at) "
                "VALUES ('block', :t, 'r', :aid, :t, 'approve_block', 'pending_approval', '2026-01-01T00:00:00')",
                {"t": f"192.0.2.{i}", "aid": f"aid_{i}"},
            )
        results = store.list_pending_actions(limit=3)
        assert len(results) == 3

    def test_list_pending_actions_default_limit_200(self):
        sig = inspect.signature(OpsStore.list_pending_actions)
        params = sig.parameters
        assert "limit" in params
        assert params["limit"].default == 50

    def test_api_actions_pending_does_not_call_fetchall_directly(self):
        """api_actions_pending must call list_pending_actions, not _fetchall (searches app.py and blueprints)."""
        import ast
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        search_files = [root / "web_app" / "app.py"]
        bp_dir = root / "web_app" / "blueprints"
        if bp_dir.is_dir():
            search_files.extend(bp_dir.glob("*.py"))

        for src_path in search_files:
            source = src_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "api_actions_pending":
                    func_source = ast.get_source_segment(source, node) or ""
                    assert "_fetchall" not in func_source, (
                        "api_actions_pending still calls ops_store._fetchall directly"
                    )
                    assert "list_pending_actions" in func_source, (
                        "api_actions_pending must call ops_store.list_pending_actions()"
                    )
                    return
        pytest.fail("api_actions_pending function not found in app.py or any blueprint")

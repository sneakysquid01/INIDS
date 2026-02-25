from src.model_registry import ModelRegistry


def test_model_registry_register_and_list(tmp_path):
    registry = ModelRegistry(str(tmp_path / "model_registry.json"))
    registry.register("rf", "models/rf.pkl", 0.99, 0.98, 12.3)
    rows = registry.list_entries(limit=10)
    assert len(rows) == 1
    assert rows[0]["name"] == "rf"

from pathlib import Path

from sage.core.config import load_config


def test_load_yaml_config(tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("a: 1\nb: test\n", encoding="utf-8")
    data = load_config(cfg)
    assert data["a"] == 1
    assert data["b"] == "test"


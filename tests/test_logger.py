from pathlib import Path

from sage.monitoring.logger import _build_log_path


def test_build_log_path_creates_module_dir(tmp_path: Path):
    path = _build_log_path(tmp_path, "trend")
    assert path.parent.exists()
    assert path.name.endswith(".log")


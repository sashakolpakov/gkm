from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest

from bongard.runtime_source_snapshot import (
    RuntimeSourceSnapshotError,
    verify_loaded_source,
)


def _module_source(module_name: str, value: int) -> str:
    return (
        "from bongard.runtime_source_snapshot import capture_loaded_source\n"
        f"SNAPSHOT = capture_loaded_source({module_name!r}, __file__)\n"
        f"VALUE = {value}\n"
    )


def test_import_snapshot_rejects_source_changed_after_import(tmp_path: Path) -> None:
    module_name = "bongard_runtime_snapshot_after_import_test"
    path = tmp_path / f"{module_name}.py"
    original = _module_source(module_name, 1)
    path.write_text(original, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        expected = hashlib.sha256(original.encode("utf-8")).hexdigest()
        assert module.SNAPSHOT == expected
        assert verify_loaded_source(
            module_name, expected_source_sha256=expected
        ) == expected

        path.write_text(_module_source(module_name, 2), encoding="utf-8")
        with pytest.raises(RuntimeSourceSnapshotError, match="changed after import"):
            verify_loaded_source(module_name)
    finally:
        sys.modules.pop(module_name, None)


def test_import_snapshot_rejects_disk_code_different_from_executing_code(
    tmp_path: Path,
) -> None:
    module_name = "bongard_runtime_snapshot_import_race_test"
    path = tmp_path / f"{module_name}.py"
    imported_source = _module_source(module_name, 1)
    path.write_text(_module_source(module_name, 2), encoding="utf-8")
    imported_code = compile(imported_source, str(path), "exec")

    namespace = {"__name__": module_name, "__file__": str(path)}
    with pytest.raises(RuntimeSourceSnapshotError, match="differs from imported code"):
        exec(imported_code, namespace)

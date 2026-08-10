from __future__ import annotations

from pathlib import Path

import pytest

from tools.audited_pytest import ProjectWriteAudit


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_write_auditor_accepts_project_artifact_path() -> None:
    audit = ProjectWriteAudit(PROJECT_ROOT)
    audit._record("test", PROJECT_ROOT / "artifacts" / "not-created")
    assert audit.events == [("test", "artifacts/not-created")]


def test_write_auditor_rejects_parent_repository_path_without_writing() -> None:
    audit = ProjectWriteAudit(PROJECT_ROOT)
    forbidden = PROJECT_ROOT.parent / "arc" / "not-created-by-roboarm"
    with pytest.raises(PermissionError, match="outside roboarm"):
        audit._record("adversarial-test", forbidden)
    assert not forbidden.exists()

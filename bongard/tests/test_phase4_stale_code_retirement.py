from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from bongard.historical_exposure import (
    DEFAULT_SEED_PATH,
    _RepositoryEvidenceReader,
    load_historical_exposure,
)
from bongard.panel_retired_pipeline_archive import (
    load_retired_pipeline_source_archive,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPOSITORY_ROOT / "bongard"
GIT_PREIMAGE_COMMIT = "a35cf269e418241da8db4fef6fb72ede20e5780f"

REMOVED_MODULES = (
    "bongard.panel_action_count_cnn_preregister",
    "bongard.panel_action_count_cnn_preregister_v2",
    "bongard.panel_action_count_spatial_dev_command",
    "bongard.run_abstraction_emergence",
    "bongard.run_bongard_logo_adapter",
    "bongard.run_bongard_overcapacity_ablation",
    "bongard.run_bongard_sparse_classifier",
    "bongard.run_bongard_symbolic_baseline",
)

REMOVED_PATHS = (
    *(module.replace(".", "/") + ".py" for module in REMOVED_MODULES),
    "bongard/test_abstraction_emergence.py",
    "bongard/test_bongard_sparse_classifier.py",
    "bongard/tests/test_panel_action_count_cnn_preregistration.py",
    "bongard/tests/test_panel_action_count_cnn_preregistration_v2.py",
    "bongard/tests/test_panel_action_count_spatial_dev_command.py",
)

ARCHIVED_PANEL_SOURCE_SHA256 = {
    "bongard.panel_action_count_cnn_preregister": (
        "37ffe6c2fc05398d12f67d48d27072b45343507abd41c895ca7c9ace11862722"
    ),
    "bongard.panel_action_count_cnn_preregister_v2": (
        "a57b685cdcdb312585c96b0fe46ffb0424612d235f4fe2d9c67c11f49673296c"
    ),
    "bongard.panel_action_count_spatial_dev_command": (
        "3a5dcf6a707132badc2706187236135a43bb5abe57d3ab80045949d71311b838"
    ),
}

DIRTY_FILES_OUTSIDE_THIS_RETIREMENT = {
    PACKAGE_ROOT / "crack_lab/codex_proposer.py",
    PACKAGE_ROOT / "tests/test_prototype_run_verification.py",
    PACKAGE_ROOT / "tests/test_semantic_gated_dev_validation.py",
}


def _imported_local_modules(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
            if node.module == "bongard":
                imported.update(f"bongard.{alias.name}" for alias in node.names)
    return imported


def test_phase4_removed_source_is_physically_absent() -> None:
    assert len(REMOVED_PATHS) == 13
    assert all(not (REPOSITORY_ROOT / path).exists() for path in REMOVED_PATHS)


def test_retained_python_has_no_import_of_phase4_modules() -> None:
    sources = (
        *PACKAGE_ROOT.glob("*.py"),
        *(PACKAGE_ROOT / "tests").glob("*.py"),
        *(PACKAGE_ROOT / "crack_lab").glob("*.py"),
    )
    removed_short_names = {module.rpartition(".")[2] for module in REMOVED_MODULES}
    for source in sources:
        if source in DIRTY_FILES_OUTSIDE_THIS_RETIREMENT:
            continue
        imported = _imported_local_modules(source)
        assert set(REMOVED_MODULES).isdisjoint(imported), source
        assert removed_short_names.isdisjoint(imported), source


@pytest.mark.parametrize("module", REMOVED_MODULES)
def test_phase4_python_m_surfaces_are_absent(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--definitely-not-a-real-option"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode != 0
    assert f"No module named {module}" in result.stderr


def test_retired_panel_sources_have_exact_authenticated_preimages() -> None:
    archive = load_retired_pipeline_source_archive()
    for module, expected_sha256 in ARCHIVED_PANEL_SOURCE_SHA256.items():
        snapshot_id = f"{module}@sha256:{expected_sha256}"
        entry = archive.entries[snapshot_id]
        source = archive.source_for(module, expected_sha256)
        assert hashlib.sha256(source).hexdigest() == expected_sha256
        assert entry["module"] == module
        assert entry["source_sha256"] == expected_sha256
        assert entry["relative_path"] == module.replace(".", "/") + ".py"
        assert entry["artifact_bindings"]


@pytest.mark.parametrize("relative_path", REMOVED_PATHS)
def test_every_phase4_source_and_test_has_a_pinned_git_preimage(
    relative_path: str,
) -> None:
    result = subprocess.run(
        ["git", "show", f"{GIT_PREIMAGE_COMMIT}:{relative_path}"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, relative_path
    assert result.stdout


def test_retired_logo_evidence_uses_the_exact_pinned_git_fallback() -> None:
    relative_path = "bongard/run_bongard_logo_adapter.py"
    assert not (REPOSITORY_ROOT / relative_path).exists()
    seed = load_historical_exposure(DEFAULT_SEED_PATH)
    reader = _RepositoryEvidenceReader(
        REPOSITORY_ROOT,
        expected_digests=dict(seed.evidence_files),
    )
    expected = dict(seed.evidence_files)[relative_path]
    fallback = reader._fallback_bytes(relative_path)
    assert reader.read_bytes(relative_path) == fallback
    assert "sha256:" + hashlib.sha256(fallback).hexdigest() == expected

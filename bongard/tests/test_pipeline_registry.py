from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest

import bongard.cli as cli
from bongard.pipeline_registry import (
    CANONICAL_PIPELINE_REGISTRY,
    LAST_DEVELOPMENT_PIPELINE_ID,
    PipelineLifecycle,
    RetiredPipelineExecutionError,
    TERMINAL_CUSTODY_GAP_PIPELINE_ID,
    pipeline_registration,
    pipeline_registry_data,
    require_new_pipeline_execution,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

PHASE_3_REMOVED_MODULES = (
    "bongard.panel_action_count_phase_command",
    "bongard.panel_action_count_multiview_adapter",
    "bongard.panel_action_count_multiview_fit_command",
    "bongard.panel_action_decomposition_threeview_adapter",
    "bongard.panel_action_decomposition_fit_ablation_command",
    "bongard.panel_soft_engineering_campaign_command",
    "bongard.panel_soft_engineering_task_runner",
    "bongard.panel_soft_observer",
    "bongard.panel_soft_predicate",
    "bongard.panel_soft_proposer",
    "bongard.panel_soft_ranker",
)
PHASE_3_REMOVED_SOURCE = (
    *(
        "bongard/" + module.rpartition(".")[2] + ".py"
        for module in PHASE_3_REMOVED_MODULES
    ),
    "bongard/tests/test_panel_action_count_phase_command.py",
    "bongard/tests/test_panel_action_count_multiview_fit_command.py",
    "bongard/tests/test_panel_action_decomposition_fit_ablation_command.py",
    "bongard/tests/test_panel_soft_engineering_campaign_command.py",
    "bongard/tests/test_panel_soft_engineering_task_runner.py",
    "bongard/tests/test_panel_soft_observer.py",
    "bongard/tests/test_panel_soft_predicate.py",
    "bongard/tests/test_panel_soft_proposer.py",
    "bongard/tests/test_panel_soft_ranker.py",
)


def _imported_modules(source_module: str) -> set[str]:
    source = PACKAGE_ROOT / (source_module.rpartition(".")[2] + ".py")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    imported.update(
        f"{node.module}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "bongard"
        for alias in node.names
    )
    return imported


def test_registry_has_terminal_gap_and_no_active_execution() -> None:
    former_successor = pipeline_registration(LAST_DEVELOPMENT_PIPELINE_ID)
    terminal = pipeline_registration(TERMINAL_CUSTODY_GAP_PIPELINE_ID)
    assert former_successor.lifecycle is PipelineLifecycle.AUDIT_ONLY
    assert former_successor.new_execution_authorized is False
    assert former_successor.successor_pipeline_id == terminal.pipeline_id
    assert terminal.lifecycle is PipelineLifecycle.TERMINATED_GAP
    assert terminal.new_execution_authorized is False
    assert terminal.entrypoints == ()
    assert all(
        registration.lifecycle is not PipelineLifecycle.ACTIVE_DEVELOPMENT
        and registration.new_execution_authorized is False
        for registration in CANONICAL_PIPELINE_REGISTRY.values()
    )
    assert {
        "bongard.panel_action_count_skeleton_graph_calibration_prereg",
        "bongard.panel_action_count_skeleton_graph_custody_incident",
        "bongard.panel_action_count_skeleton_graph_custody_incident_persistence",
        "bongard.panel_action_count_skeleton_graph_custody_gap",
    }.issubset(terminal.source_modules)

    retired = tuple(
        item
        for item in CANONICAL_PIPELINE_REGISTRY.values()
        if item.lifecycle is PipelineLifecycle.RETIRED
    )
    assert retired
    assert all(item.new_execution_authorized is False for item in retired)
    assert all(
        item.successor_pipeline_id == terminal.pipeline_id for item in retired
    )
    assert all(item.retained_for for item in retired)

    removed_by_id = {
        "panel-feature-exposed-support-smoke-v1": (
            "bongard.panel_feature_exposed_support_smoke_command",
        ),
        "panel-positive-prose-exposed-probe-v1": (
            "bongard.panel_positive_prose_exposed_probe_command",
        ),
        "panel-positive-contextual-typed-count-probe-v1": (
            "bongard.panel_positive_contextual_typed_count_probe_command",
        ),
        "panel-positive-atom-slate-exposed-probe-v1": (
            "bongard.panel_positive_atom_slate_exposed_probe_command",
        ),
        "panel-hierarchical-exposed-support-smoke-v1": (
            "bongard.panel_hierarchical_exposed_support_smoke_command",
        ),
        "panel-soft-exact-unused-campaign-v1": PHASE_3_REMOVED_MODULES[5:],
        "panel-action-count-prompt-development-v1": PHASE_3_REMOVED_MODULES[:5],
    }
    for pipeline_id, modules in removed_by_id.items():
        removed = pipeline_registration(pipeline_id)
        assert removed.source_modules == ()
        assert removed.removed_source_modules == modules
        assert removed.removal_blockers == ()


def test_registry_report_names_removal_blockers_and_unlean_authority() -> None:
    report = pipeline_registry_data()
    assert report["schema"] == "gkm.bongard-pipeline-lifecycle-registry.v2"
    assert report["active_successor_pipeline_id"] is None
    assert report["last_development_pipeline_id"] == LAST_DEVELOPMENT_PIPELINE_ID
    assert report["terminal_pipeline_id"] == TERMINAL_CUSTODY_GAP_PIPELINE_ID
    assert report["new_execution_authorized"] is False
    assert report["registry_is_execution_enforcement_boundary"] is False
    assert report["direct_module_execution_requires_independent_custody"] is True
    assert report["python_is_canonical_authority"] is True
    assert report["lean_present"] is False
    assert report["lean_required"] is False
    assert report["lean_removable"] is True
    retirement = report["physical_retirement_plan"]
    assert len(retirement["phase_2_removed_source"]) == 8
    assert retirement["phase_2_neutral_successors"] == {
        "bounded_custody": "bongard.panel_probe_custody",
        "named_image_transport": "bongard.panel_probe_transport",
        "retired_source_decoder": "bongard.panel_retired_probe_source_archive",
        "retired_source_snapshot": (
            "bongard/data/panel_retired_probe_source_snapshot_20260810_v1.json"
        ),
    }
    assert tuple(retirement["phase_3_removed_source"]) == PHASE_3_REMOVED_SOURCE
    assert retirement["phase_3_neutral_successors"] == {
        "retired_source_decoder": "bongard.panel_retired_pipeline_archive",
        "retired_source_snapshot": (
            "bongard/data/panel_retired_pipeline_source_snapshot_20260810_v1.json"
        ),
    }
    assert retirement["phase_3_test_preimage_commit"] == (
        "a35cf269e418241da8db4fef6fb72ede20e5780f"
    )
    assert len(retirement["phase_4_removed_source"]) == 13
    assert retirement["phase_4_panel_source_preimage_archive"] == (
        "bongard/data/panel_retired_pipeline_source_snapshot_20260810_v1.json"
    )
    assert retirement["phase_4_git_source_and_test_preimage_commit"] == (
        "a35cf269e418241da8db4fef6fb72ede20e5780f"
    )
    assert (
        "bongard/data/panel_retired_pipeline_source_snapshot_20260810_v1.json"
        in retirement["audit_artifact_policy"]["immutable_compact_records_to_retain"]
    )
    by_id = {item["pipeline_id"]: item for item in report["pipelines"]}
    assert by_id["panel-feature-exposed-support-smoke-v1"]["removal_blockers"] == []
    assert by_id["panel-soft-exact-unused-campaign-v1"]["removal_blockers"] == []
    assert by_id["panel-action-count-prompt-development-v1"][
        "removal_blockers"
    ] == []
    assert by_id["panel-positive-prose-exposed-probe-v1"]["removal_blockers"] == []
    audit = by_id["completed-support-diagnostic-artifacts-v1"]
    assert "bongard.panel_retired_pipeline_archive" in audit["source_modules"]


def test_audit_only_typed_axis_sources_exclude_retired_action_count_executors() -> None:
    retired_modules = {
        "bongard.panel_feature_exposed_support_smoke_command",
        "bongard.panel_positive_prose_exposed_probe_command",
        "bongard.panel_positive_contextual_typed_count_probe_command",
        "bongard.panel_positive_atom_slate_exposed_probe_command",
        "bongard.panel_action_count_phase_command",
        "bongard.panel_action_count_multiview_fit_command",
        "bongard.panel_action_decomposition_fit_ablation_command",
        "bongard.panel_action_count_cnn_train_command",
        "bongard.panel_action_count_spatial_dev_command",
    }
    successor = pipeline_registration(LAST_DEVELOPMENT_PIPELINE_ID)
    assert successor.lifecycle is PipelineLifecycle.AUDIT_ONLY
    assert successor.new_execution_authorized is False
    assert successor.entrypoints == ()
    assert set(successor.source_modules).isdisjoint(retired_modules)
    assert {
        "bongard.panel_typed_axis_slate_v2",
        "bongard.panel_typed_axis_task_runner",
        "bongard.panel_feature_extracted_release_gate",
        "bongard.python_predicate_authority",
    }.issubset(successor.source_modules)


def test_failed_action_observers_are_registered_by_physical_status() -> None:
    soft = pipeline_registration("panel-soft-exact-unused-campaign-v1")
    prompt = pipeline_registration("panel-action-count-prompt-development-v1")
    cnn = pipeline_registration(
        "panel-action-count-global-spatial-cnn-development-v1"
    )
    tiny = pipeline_registration("panel-action-count-tiny-query-set-development-v1")
    assert soft.lifecycle is PipelineLifecycle.RETIRED
    assert prompt.lifecycle is PipelineLifecycle.RETIRED
    assert cnn.lifecycle is PipelineLifecycle.RETIRED
    assert tiny.lifecycle is PipelineLifecycle.RETIRED
    assert soft.new_execution_authorized is False
    assert prompt.new_execution_authorized is False
    assert cnn.new_execution_authorized is False
    assert tiny.new_execution_authorized is False
    assert soft.source_modules == ()
    assert soft.removed_source_modules == PHASE_3_REMOVED_MODULES[5:]
    assert prompt.source_modules == ()
    assert prompt.removed_source_modules == PHASE_3_REMOVED_MODULES[:5]
    assert "bongard.panel_action_count_spatial_dev_command" in (
        cnn.removed_source_modules
    )
    assert {
        "bongard.panel_action_count_cnn_preregister",
        "bongard.panel_action_count_cnn_preregister_v2",
    }.issubset(cnn.removed_source_modules)
    assert soft.removal_blockers == ()
    assert prompt.removal_blockers == ()
    assert cnn.removal_blockers
    assert "bongard.panel_action_count_tiny_local_failure_forensics" in (
        tiny.source_modules
    )
    assert tiny.removal_blockers


@pytest.mark.parametrize(
    "pipeline_id",
    (LAST_DEVELOPMENT_PIPELINE_ID, TERMINAL_CUSTODY_GAP_PIPELINE_ID),
)
def test_superseded_and_terminal_pipelines_fail_closed(pipeline_id: str) -> None:
    with pytest.raises(RetiredPipelineExecutionError, match="cannot start"):
        require_new_pipeline_execution(pipeline_id)


@pytest.mark.parametrize("relative_path", PHASE_3_REMOVED_SOURCE)
def test_phase_3_removed_source_is_physically_absent(relative_path: str) -> None:
    assert not (PACKAGE_ROOT.parent / relative_path).exists()


def test_removed_launcher_is_gone_but_immutable_failure_record_remains() -> None:
    assert not (
        PACKAGE_ROOT / "panel_hierarchical_exposed_support_smoke_command.py"
    ).exists()
    assert not (
        PACKAGE_ROOT
        / "tests/test_panel_hierarchical_exposed_support_smoke_command.py"
    ).exists()
    assert (
        PACKAGE_ROOT
        / "data/panel_hierarchical_exposed_support_smoke_20260809_v1.failure.json"
    ).is_file()


@pytest.mark.parametrize(
    "module",
    (
        "bongard.panel_feature_exposed_support_smoke_command",
        "bongard.panel_positive_prose_exposed_probe_command",
        "bongard.panel_positive_contextual_typed_count_probe_command",
        "bongard.panel_positive_atom_slate_exposed_probe_command",
    ),
)
def test_physically_retired_python_m_surfaces_fail_closed(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--definitely-not-a-real-option"],
        cwd=PACKAGE_ROOT.parent,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode != 0
    assert f"No module named {module}" in result.stderr


@pytest.mark.parametrize(
    "module",
    (
        "bongard.panel_soft_engineering_campaign_command",
        "bongard.panel_action_count_phase_command",
        "bongard.panel_action_count_multiview_fit_command",
        "bongard.panel_action_decomposition_fit_ablation_command",
    ),
)
def test_phase_3_removed_python_m_surfaces_fail_closed(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--definitely-not-a-real-option"],
        cwd=PACKAGE_ROOT.parent,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode != 0
    assert f"No module named {module}" in result.stderr


@pytest.mark.parametrize(
    "pipeline_id",
    (
        "legacy-two-query-episode-cli-v1",
        "legacy-visual-semantic-calibration-cli-v1",
        "panel-soft-exact-unused-campaign-v1",
        "panel-action-count-prompt-development-v1",
        "panel-action-count-global-spatial-cnn-development-v1",
        "panel-action-count-tiny-query-set-development-v1",
        "legacy-july-symbolic-scaffolds-v1",
        "panel-feature-exposed-support-smoke-v1",
        "panel-positive-prose-exposed-probe-v1",
        "panel-positive-contextual-typed-count-probe-v1",
        "panel-positive-atom-slate-exposed-probe-v1",
        "panel-hierarchical-exposed-support-smoke-v1",
    ),
)
def test_retired_pipeline_guard_is_fail_closed(pipeline_id: str) -> None:
    with pytest.raises(RetiredPipelineExecutionError, match="cannot start"):
        require_new_pipeline_execution(pipeline_id)


@pytest.mark.parametrize(
    ("command", "pipeline_id"),
    (
        ("run", "legacy-two-query-episode-cli-v1"),
        (
            "calibrate-semantic-stage-a",
            "legacy-visual-semantic-calibration-cli-v1",
        ),
        (
            "validate-semantic-stage-b",
            "legacy-visual-semantic-calibration-cli-v1",
        ),
    ),
)
def test_legacy_canonical_cli_handlers_are_retirement_guards(
    command: str, pipeline_id: str
) -> None:
    subparser_action = next(
        action
        for action in cli.build_parser()._actions
        if action.dest == "command"
    )
    parser = subparser_action.choices[command]
    assert parser.get_default("handler") is cli._retired_pipeline_command
    assert parser.get_default("retired_pipeline_id") == pipeline_id

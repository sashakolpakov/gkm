from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest

import bongard.cli as cli
from bongard.pipeline_registry import (
    ACTIVE_SUCCESSOR_PIPELINE_ID,
    CANONICAL_PIPELINE_REGISTRY,
    PipelineLifecycle,
    RetiredPipelineExecutionError,
    pipeline_registration,
    pipeline_registry_data,
    require_new_pipeline_execution,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


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
    return imported


def test_registry_has_one_python_successor_and_fail_closed_retirements() -> None:
    successor = pipeline_registration(ACTIVE_SUCCESSOR_PIPELINE_ID)
    assert successor.lifecycle is PipelineLifecycle.ACTIVE_DEVELOPMENT
    assert successor.new_execution_authorized is True
    assert "python" in successor.pipeline_id

    retired = tuple(
        item
        for item in CANONICAL_PIPELINE_REGISTRY.values()
        if item.lifecycle is PipelineLifecycle.RETIRED
    )
    assert retired
    assert all(item.new_execution_authorized is False for item in retired)
    assert all(item.successor_pipeline_id == successor.pipeline_id for item in retired)
    assert all(item.retained_for for item in retired)

    removed_by_id = {
        "panel-feature-exposed-support-smoke-v1": (
            "bongard.panel_feature_exposed_support_smoke_command"
        ),
        "panel-positive-prose-exposed-probe-v1": (
            "bongard.panel_positive_prose_exposed_probe_command"
        ),
        "panel-positive-contextual-typed-count-probe-v1": (
            "bongard.panel_positive_contextual_typed_count_probe_command"
        ),
        "panel-positive-atom-slate-exposed-probe-v1": (
            "bongard.panel_positive_atom_slate_exposed_probe_command"
        ),
        "panel-hierarchical-exposed-support-smoke-v1": (
            "bongard.panel_hierarchical_exposed_support_smoke_command"
        ),
    }
    for pipeline_id, module in removed_by_id.items():
        removed = pipeline_registration(pipeline_id)
        assert removed.source_modules == ()
        assert removed.removed_source_modules == (module,)
        assert removed.removal_blockers == ()


def test_registry_report_names_removal_blockers_and_unlean_authority() -> None:
    report = pipeline_registry_data()
    assert report["active_successor_pipeline_id"] == ACTIVE_SUCCESSOR_PIPELINE_ID
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
    by_id = {item["pipeline_id"]: item for item in report["pipelines"]}
    assert by_id["panel-feature-exposed-support-smoke-v1"]["removal_blockers"] == []
    assert by_id["panel-soft-exact-unused-campaign-v1"][
        "removal_blockers"
    ]
    assert by_id["panel-positive-prose-exposed-probe-v1"]["removal_blockers"] == []


def test_active_typed_axis_sources_exclude_retired_action_count_executors() -> None:
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
    successor = pipeline_registration(ACTIVE_SUCCESSOR_PIPELINE_ID)
    assert successor.entrypoints == ()
    assert set(successor.source_modules).isdisjoint(retired_modules)
    assert {
        "bongard.panel_typed_axis_slate_v2",
        "bongard.panel_typed_axis_task_runner",
        "bongard.panel_feature_extracted_release_gate",
        "bongard.python_predicate_authority",
    }.issubset(successor.source_modules)


def test_failed_action_observers_are_registered_retired() -> None:
    prompt = pipeline_registration("panel-action-count-prompt-development-v1")
    cnn = pipeline_registration(
        "panel-action-count-global-spatial-cnn-development-v1"
    )
    assert prompt.lifecycle is PipelineLifecycle.RETIRED
    assert cnn.lifecycle is PipelineLifecycle.RETIRED
    assert prompt.new_execution_authorized is False
    assert cnn.new_execution_authorized is False
    assert "bongard.panel_action_count_multiview_fit_command" in prompt.source_modules
    assert "bongard.panel_action_count_spatial_dev_command" in cnn.source_modules
    assert prompt.removal_blockers
    assert cnn.removal_blockers


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
    "pipeline_id",
    (
        "legacy-two-query-episode-cli-v1",
        "legacy-visual-semantic-calibration-cli-v1",
        "panel-soft-exact-unused-campaign-v1",
        "panel-action-count-prompt-development-v1",
        "panel-action-count-global-spatial-cnn-development-v1",
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

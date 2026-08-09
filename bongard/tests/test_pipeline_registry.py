from __future__ import annotations

import ast
from pathlib import Path

import pytest

import bongard.cli as cli
import bongard.panel_feature_exposed_support_smoke_command as feature_smoke
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


def _imports_from(source_module: str, imported_module: str) -> list[str]:
    source = PACKAGE_ROOT / (source_module.rpartition(".")[2] + ".py")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    return [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == imported_module
        for alias in node.names
    ]


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

    removed = pipeline_registration(
        "panel-hierarchical-exposed-support-smoke-v1"
    )
    assert removed.source_modules == ()
    assert removed.removed_source_modules == (
        "bongard.panel_hierarchical_exposed_support_smoke_command",
    )
    assert removed.removal_blockers == ()


def test_registry_report_names_removal_blockers_and_unlean_authority() -> None:
    report = pipeline_registry_data()
    assert report["active_successor_pipeline_id"] == ACTIVE_SUCCESSOR_PIPELINE_ID
    assert report["python_is_canonical_authority"] is True
    assert report["lean_present"] is False
    assert report["lean_required"] is False
    assert report["lean_removable"] is True
    retirement = report["physical_retirement_plan"]
    phase_2 = retirement["phase_2_extract_shared_helpers_then_remove"]
    assert phase_2[
        "helper_imports_from_panel_positive_prose_exposed_probe_command"
    ]["bongard.panel_action_count_phase_command"] == ["_call"]
    assert len(phase_2["remove_source_and_exclusive_tests"]) == 8
    by_id = {item["pipeline_id"]: item for item in report["pipelines"]}
    assert by_id["panel-feature-exposed-support-smoke-v1"][
        "removal_blockers"
    ]
    assert by_id["panel-soft-exact-unused-campaign-v1"][
        "removal_blockers"
    ]
    assert by_id["panel-positive-prose-exposed-probe-v1"][
        "removal_blockers"
    ]


def test_phase_2_helper_map_is_the_exact_current_import_graph() -> None:
    retirement = pipeline_registry_data()["physical_retirement_plan"]
    phase_2 = retirement["phase_2_extract_shared_helpers_then_remove"]
    maps = (
        (
            "bongard.panel_feature_exposed_support_smoke_command",
            phase_2[
                "helper_imports_from_panel_feature_exposed_support_smoke_command"
            ],
        ),
        (
            "bongard.panel_positive_prose_exposed_probe_command",
            phase_2[
                "helper_imports_from_panel_positive_prose_exposed_probe_command"
            ],
        ),
    )
    for imported_module, expected_by_source in maps:
        for source_module, expected_names in expected_by_source.items():
            assert _imports_from(source_module, imported_module) == expected_names


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
    "pipeline_id",
    (
        "legacy-two-query-episode-cli-v1",
        "legacy-visual-semantic-calibration-cli-v1",
        "panel-soft-exact-unused-campaign-v1",
        "panel-feature-exposed-support-smoke-v1",
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


@pytest.mark.parametrize(
    ("entrypoint", "pipeline_id"),
    (
        (feature_smoke.main, "panel-feature-exposed-support-smoke-v1"),
    ),
)
def test_retired_standalone_commands_fail_before_argument_or_io_access(
    entrypoint, pipeline_id: str
) -> None:
    with pytest.raises(
        RetiredPipelineExecutionError,
        match=repr(pipeline_id),
    ):
        entrypoint(["--definitely-not-a-real-option"])

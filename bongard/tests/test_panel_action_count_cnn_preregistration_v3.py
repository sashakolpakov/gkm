from __future__ import annotations

import ast
import json
from pathlib import Path

from bongard.canonical import canonical_digest, canonical_json
from bongard.panel_action_count_cnn_preregister_v3 import (
    EXPECTED_CALIBRATION_PANEL_IDS_DIGEST,
    EXPECTED_CALIBRATION_TASK_IDS_DIGEST,
    EXPECTED_EVALUATION_PANEL_IDS_DIGEST,
    EXPECTED_EVALUATION_TASK_IDS_DIGEST,
    build_v3_preregistration,
)


BONGARD = Path(__file__).resolve().parents[1]
ROOT = BONGARD.parent
CORPUS = ROOT / "downloads/ShapeBongard_V2_full/ShapeBongard_V2"
DATA = BONGARD / "data"
SOURCE = BONGARD / "panel_action_count_cnn_preregister_v3.py"
POSTPREDICTION_SOURCE = (
    BONGARD / "panel_action_count_cnn_postprediction_labels_v3.py"
)
PLAN = DATA / "panel_action_count_cnn_preregistration_20260810_v3.json"
DEVELOPMENT = DATA / "panel_action_count_cnn_development_panels_20260810_v3.json"
CALIBRATION = DATA / "panel_action_count_cnn_calibration_panels_20260810_v3.json"
EVALUATION = DATA / "panel_action_count_cnn_evaluation_panels_20260810_v3.json"
V2 = DATA / "panel_action_count_cnn_preregistration_20260810_v2.json"
ACTION_COUNT_PLAN = (
    DATA / "panel_action_count_calibration_preregistration_20260809_v1.json"
)
FAMILY_PLAN = DATA / "panel_convex_four_lines_same_family_train_drill_20260809_v1.json"
HISTORICAL = DATA / "historical_exposure_v1.json"
LEDGER = (
    ROOT
    / "downloads/ShapeBongard_V2_full/panel_soft_exact_unused_train_20260809_ranked_v1"
    / "research-exposure-successors"
    / "6995ea9cfda2f384cb0ba1b1cdc3611c965227c60fdb281d1e2e56fffa357b56.exposure.json"
)
SPLIT = CORPUS / "ShapeBongard_V2_split.json"


def _record(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == canonical_json(value) + b"\n"
    body = dict(value)
    found = body.pop("record_digest")
    assert found == "sha256:" + canonical_digest(body)
    return value


def _rebuild() -> tuple[dict[str, object], ...]:
    return build_v3_preregistration(
        repository_root=ROOT,
        authority_source_path=SOURCE,
        postprediction_authority_source_path=POSTPREDICTION_SOURCE,
        v2_plan_path=V2,
        action_count_plan_path=ACTION_COUNT_PLAN,
        family_plan_path=FAMILY_PLAN,
        historical_exposure_path=HISTORICAL,
        cumulative_exposure_ledger_path=LEDGER,
        split_path=SPLIT,
        development_output_path=DEVELOPMENT,
        calibration_output_path=CALIBRATION,
        evaluation_output_path=EVALUATION,
    )


def test_v3_replays_exactly_with_a_metadata_only_read_boundary(monkeypatch) -> None:
    archived = tuple(_record(path) for path in (PLAN, DEVELOPMENT, CALIBRATION, EVALUATION))
    assert [value["record_digest"] for value in archived] == [
        "sha256:bb4524a0958cd21f2d4d49bc6a9caa964ccb96c67fbf7c6192185f7b2f363dcb",
        "sha256:ee02e48ea3e07dd4804ad24e5c1c9228addc4a0fe658efe821993451bc749fde",
        "sha256:17088e6b72544a12829b255b4ada9f3b50e03423595c295185dbcfb02f9f515f",
        "sha256:6e0e17a91b48547a83706968d58fbc1ef8c61bbe3f082d8986d9b6bff33678cd",
    ]
    allowed = {
        SOURCE.resolve(),
        POSTPREDICTION_SOURCE.resolve(),
        V2.resolve(),
        ACTION_COUNT_PLAN.resolve(),
        FAMILY_PLAN.resolve(),
        HISTORICAL.resolve(),
        LEDGER.resolve(),
        SPLIT.resolve(),
    }
    original = Path.read_bytes
    opened: list[Path] = []

    def guard(path: Path) -> bytes:
        resolved = path.resolve()
        opened.append(resolved)
        if resolved not in allowed:
            raise AssertionError(f"V3 opened a non-metadata input: {resolved}")
        lowered = resolved.name.lower()
        if resolved.suffix.lower() == ".png":
            raise AssertionError("V3 opened a panel PNG")
        if resolved.suffix.lower() == ".json" and any(
            token in lowered
            for token in ("action_program", "catalog", "label", "development_labels")
        ):
            raise AssertionError("V3 opened an action/catalog/target artifact")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guard)
    assert _rebuild() == archived
    assert set(opened) == allowed


def test_v3_source_ast_has_no_action_catalog_or_target_import_boundary() -> None:
    tree = ast.parse(SOURCE.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(
        token in module
        for module in imported
        for token in (
            "panel_action_count_cnn_preregister_v2",
            "panel_convexity_catalog",
            "action_program",
            "label",
        )
    )
    builder = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "build_v3_preregistration"
    )
    argument_names = {argument.arg for argument in builder.args.kwonlyargs}
    assert not any(
        token in name
        for name in argument_names
        for token in ("program", "catalog", "label_manifest", "label_path")
    )


def test_v3_keeps_v2_development_and_excludes_all_v2_tasks_from_fresh_roles() -> None:
    plan = _record(PLAN)
    v2 = _record(V2)
    assert plan["cohorts"]["train"]["task_ids"] == v2["cohorts"]["train"]["task_ids"]
    assert plan["cohorts"]["validation"]["task_ids"] == (
        v2["cohorts"]["validation"]["task_ids"]
    )
    old = set(v2["oracle_taint_record"]["selected_task_ids"])
    calibration = plan["cohorts"]["calibration"]
    evaluation = plan["cohorts"]["evaluation"]
    assert calibration["rank_slice"] == [1100, 1200]
    assert evaluation["rank_slice"] == [1200, 1300]
    assert not old.intersection(calibration["task_ids"])
    assert not old.intersection(evaluation["task_ids"])
    assert not set(calibration["task_ids"]).intersection(evaluation["task_ids"])
    taint = plan["old_v2_design_taint"]
    assert taint["permanent"] is True
    assert taint["reuse_allowed"] is False
    assert taint["all_1100_v2_tasks_excluded_from_fresh_selection"] is True
    assert taint["old_tainted_task_count"] == 200
    assert taint["old_calibration_and_evaluation_plaintext_targets_materialized"] is True
    assert taint["old_calibration_and_evaluation_panel_png_bytes_read"] == 0
    assert set(taint["old_calibration_task_ids"]) == set(
        v2["cohorts"]["calibration"]["task_ids"]
    )
    assert set(taint["old_evaluation_task_ids"]) == set(
        v2["cohorts"]["evaluation"]["task_ids"]
    )


def test_v3_fresh_cohort_digests_and_identifier_only_manifests_are_exact() -> None:
    plan = _record(PLAN)
    calibration = plan["cohorts"]["calibration"]
    evaluation = plan["cohorts"]["evaluation"]
    assert calibration["task_ids_digest"] == EXPECTED_CALIBRATION_TASK_IDS_DIGEST
    assert calibration["panel_ids_digest"] == EXPECTED_CALIBRATION_PANEL_IDS_DIGEST
    assert evaluation["task_ids_digest"] == EXPECTED_EVALUATION_TASK_IDS_DIGEST
    assert evaluation["panel_ids_digest"] == EXPECTED_EVALUATION_PANEL_IDS_DIGEST
    for manifest_path in (DEVELOPMENT, CALIBRATION, EVALUATION):
        manifest = _record(manifest_path)
        for cohort in manifest["cohorts"].values():
            assert set(cohort) == {"panel_ids", "task_ids"}
            assert len(cohort["panel_ids"]) == len(cohort["task_ids"]) * 14
        serialized = canonical_json(manifest).decode("utf-8").lower()
        for forbidden in (
            "action_program",
            '"rows"',
            '"summary"',
            "label_value",
            "label_digest",
            "target_digest",
            "catalog_import",
        ):
            assert forbidden not in serialized
    assert plan["current_state"] == {
        "fresh_action_program_or_target_rows_read": 0,
        "fresh_calibration_panel_png_bytes_read": 0,
        "fresh_evaluation_panel_png_bytes_read": 0,
        "fresh_plaintext_targets_materialized": False,
        "model_training_started": False,
        "selected_png_bytes_read_by_v3_authority": 0,
    }

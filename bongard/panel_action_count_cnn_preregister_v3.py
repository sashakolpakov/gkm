"""Clean pre-pixel V3 cohort and custody authority.

Selection reads only official split task IDs and metadata exclusion records.
It never reads panel PNGs, action programs, catalog data, or label artifacts.
V2 train/validation are retained; V2 calibration/evaluation are permanently
design-tainted because plaintext targets were materialized before prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.exposure import ExposureLedger


SCHEMA = "gkm.bongard-action-count-catalog-cnn-preregistration.v3"
SELECTION_SEED = "gkm-panel-action-count-cnn-supervised-train-dev-20260810-v1"
TRAIN_TASK_COUNT = 800
VALIDATION_TASK_COUNT = 100
OLD_DESIGN_TAINT_TASK_COUNT = 200
CALIBRATION_TASK_COUNT = 100
EVALUATION_TASK_COUNT = 100
PANELS_PER_TASK = 14
CALIBRATION_RANK_SLICE = (1100, 1200)
EVALUATION_RANK_SLICE = (1200, 1300)
EXPECTED_CALIBRATION_TASK_IDS_DIGEST = (
    "sha256:01ae13699706ff67f524241fe257224a6f9136b80d2f8b857e2b98ff758f82c9"
)
EXPECTED_CALIBRATION_PANEL_IDS_DIGEST = (
    "sha256:5a82a07cee92dbeae57c10b48f080afc9d490dd5514c89e3ef3cb1eebadca5df"
)
EXPECTED_EVALUATION_TASK_IDS_DIGEST = (
    "sha256:0fb782b78f709850801e5669e844591728d0f167b4e9965f87d302324f71f1c3"
)
EXPECTED_EVALUATION_PANEL_IDS_DIGEST = (
    "sha256:550a0403574665a0a215650b2712738eb450af0e3bb2ace1bb6cec97bde689ed"
)


class ActionCountCNNV3PreregistrationError(RuntimeError):
    """V3 metadata selection or custody bindings differ."""


def _address(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _read_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ActionCountCNNV3PreregistrationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ActionCountCNNV3PreregistrationError(f"{label} is not an object")
    return value, raw


def _canonical_record(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    value, raw = _read_object(path, label)
    if raw != canonical_json(value) + b"\n":
        raise ActionCountCNNV3PreregistrationError(f"{label} is not canonical")
    body = dict(value)
    found = body.pop("record_digest", None)
    expected = "sha256:" + canonical_digest(body)
    if found != expected:
        raise ActionCountCNNV3PreregistrationError(f"{label} digest differs")
    return value, raw


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ActionCountCNNV3PreregistrationError(f"{label} is not a string list")
    if len(value) != len(set(value)):
        raise ActionCountCNNV3PreregistrationError(f"{label} contains duplicates")
    return list(value)


def _task_ids_from_action_plan(plan: Mapping[str, Any]) -> set[str]:
    cohorts = plan.get("cohorts")
    if not isinstance(cohorts, dict):
        raise ActionCountCNNV3PreregistrationError("action-count cohorts are invalid")
    result: list[str] = []
    for name, cohort in cohorts.items():
        if not isinstance(cohort, dict):
            raise ActionCountCNNV3PreregistrationError(
                f"action-count cohort {name} is invalid"
            )
        result.extend(_string_list(cohort.get("task_ids"), f"action-count {name}"))
    if len(result) != len(set(result)):
        raise ActionCountCNNV3PreregistrationError("action-count cohorts overlap")
    return set(result)


def _task_ids_from_family_plan(plan: Mapping[str, Any]) -> set[str]:
    partition = plan.get("frozen_partition")
    if not isinstance(partition, dict):
        raise ActionCountCNNV3PreregistrationError("family partition is invalid")
    result: list[str] = []
    for name, task_ids in partition.items():
        result.extend(_string_list(task_ids, f"family partition {name}"))
    if len(result) != len(set(result)):
        raise ActionCountCNNV3PreregistrationError("family partition overlaps")
    return set(result)


def _task_ids_from_historical(value: Mapping[str, Any]) -> set[str]:
    seed = value.get("seed")
    exact = seed.get("exact_official_exposure") if isinstance(seed, dict) else None
    rows = exact.get("task_ids") if isinstance(exact, dict) else None
    if not isinstance(rows, list):
        raise ActionCountCNNV3PreregistrationError("historical task rows are invalid")
    result: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("task_id"), str):
            raise ActionCountCNNV3PreregistrationError("historical task row is invalid")
        result.append(row["task_id"])
    if len(result) != len(set(result)):
        raise ActionCountCNNV3PreregistrationError("historical tasks overlap")
    return set(result)


def _panel_ids(task_ids: Sequence[str]) -> list[str]:
    return [
        f"hd/{task_id}/{folder}/{panel_index}.png"
        for task_id in task_ids
        for folder in (1, 0)
        for panel_index in range(7)
    ]


def _ids_manifest(
    schema: str, claim: str, cohorts: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "claim": claim,
        "cohorts": {
            name: {"panel_ids": _panel_ids(task_ids), "task_ids": list(task_ids)}
            for name, task_ids in cohorts.items()
        },
        "schema": schema,
    }
    return {**body, "record_digest": "sha256:" + canonical_digest(body)}


def _manifest_binding(
    repository_root: Path, path: Path, value: Mapping[str, Any]
) -> dict[str, Any]:
    raw = canonical_json(value) + b"\n"
    return {
        "path": str(path.relative_to(repository_root)),
        "record_digest": value["record_digest"],
        "source_sha256": _address(raw),
    }


def build_v3_preregistration(
    *,
    repository_root: Path,
    authority_source_path: Path,
    postprediction_authority_source_path: Path,
    v2_plan_path: Path,
    action_count_plan_path: Path,
    family_plan_path: Path,
    historical_exposure_path: Path,
    cumulative_exposure_ledger_path: Path,
    split_path: Path,
    development_output_path: Path,
    calibration_output_path: Path,
    evaluation_output_path: Path,
) -> tuple[dict[str, Any], ...]:
    """Build V3 using metadata only; selected pixels and targets remain unopened."""

    split, split_raw = _read_object(split_path, "official split")
    v2, v2_raw = _canonical_record(v2_plan_path, "V2 plan")
    prior, prior_raw = _canonical_record(action_count_plan_path, "action-count plan")
    family, family_raw = _canonical_record(family_plan_path, "family plan")
    historical, historical_raw = _read_object(
        historical_exposure_path, "historical exposure"
    )
    cumulative_value, cumulative_raw = _read_object(
        cumulative_exposure_ledger_path, "cumulative exposure ledger"
    )
    cumulative = ExposureLedger.from_dict(cumulative_value)

    official_train = _string_list(split.get("train"), "official TRAIN task IDs")
    hd_train = sorted(task_id for task_id in official_train if task_id.startswith("hd_"))
    semantic_excluded = {
        task_id
        for task_id in hd_train
        if "convex" in task_id or "has_four_straight_lines" in task_id
    }
    metadata_excluded = (
        _task_ids_from_action_plan(prior)
        | _task_ids_from_family_plan(family)
        | _task_ids_from_historical(historical)
        | {
            task_id
            for event in cumulative.events
            for task_id in event.task_ids
        }
    )
    eligible = sorted(set(hd_train) - semantic_excluded - metadata_excluded)
    ranked = sorted(
        eligible,
        key=lambda task_id: (
            hashlib.sha256((SELECTION_SEED + "\0" + task_id).encode()).hexdigest(),
            task_id,
        ),
    )
    if len(ranked) < EVALUATION_RANK_SLICE[1]:
        raise ActionCountCNNV3PreregistrationError("not enough eligible HD TRAIN tasks")

    v2_taint = v2.get("oracle_taint_record")
    v2_cohorts = v2.get("cohorts")
    if not isinstance(v2_taint, dict) or not isinstance(v2_cohorts, dict):
        raise ActionCountCNNV3PreregistrationError("V2 selection records are invalid")
    old_v2_tasks = _string_list(v2_taint.get("selected_task_ids"), "V2 selected tasks")
    if len(old_v2_tasks) != 1100 or old_v2_tasks != ranked[:1100]:
        raise ActionCountCNNV3PreregistrationError("V2 is not the frozen rank prefix")
    train_tasks = ranked[:TRAIN_TASK_COUNT]
    validation_tasks = ranked[TRAIN_TASK_COUNT : TRAIN_TASK_COUNT + VALIDATION_TASK_COUNT]
    old_calibration_tasks = ranked[900:1000]
    old_evaluation_tasks = ranked[1000:1100]
    if train_tasks != _string_list(v2_cohorts["train"].get("task_ids"), "V2 train"):
        raise ActionCountCNNV3PreregistrationError("V2 train changed")
    if validation_tasks != _string_list(
        v2_cohorts["validation"].get("task_ids"), "V2 validation"
    ):
        raise ActionCountCNNV3PreregistrationError("V2 validation changed")
    if old_calibration_tasks != _string_list(
        v2_cohorts["calibration"].get("task_ids"), "V2 calibration"
    ):
        raise ActionCountCNNV3PreregistrationError("V2 calibration changed")
    if old_evaluation_tasks != _string_list(
        v2_cohorts["evaluation"].get("task_ids"), "V2 evaluation"
    ):
        raise ActionCountCNNV3PreregistrationError("V2 evaluation changed")

    old_v2_set = set(old_v2_tasks)
    fresh_ranked = [task_id for task_id in ranked if task_id not in old_v2_set]
    calibration_tasks = fresh_ranked[:CALIBRATION_TASK_COUNT]
    evaluation_tasks = fresh_ranked[
        CALIBRATION_TASK_COUNT : CALIBRATION_TASK_COUNT + EVALUATION_TASK_COUNT
    ]
    if calibration_tasks != ranked[slice(*CALIBRATION_RANK_SLICE)]:
        raise ActionCountCNNV3PreregistrationError("fresh calibration rank differs")
    if evaluation_tasks != ranked[slice(*EVALUATION_RANK_SLICE)]:
        raise ActionCountCNNV3PreregistrationError("fresh evaluation rank differs")
    calibration_panels = _panel_ids(calibration_tasks)
    evaluation_panels = _panel_ids(evaluation_tasks)
    expected = {
        "calibration task IDs": (
            "sha256:" + canonical_digest(calibration_tasks),
            EXPECTED_CALIBRATION_TASK_IDS_DIGEST,
        ),
        "calibration panel IDs": (
            "sha256:" + canonical_digest(calibration_panels),
            EXPECTED_CALIBRATION_PANEL_IDS_DIGEST,
        ),
        "evaluation task IDs": (
            "sha256:" + canonical_digest(evaluation_tasks),
            EXPECTED_EVALUATION_TASK_IDS_DIGEST,
        ),
        "evaluation panel IDs": (
            "sha256:" + canonical_digest(evaluation_panels),
            EXPECTED_EVALUATION_PANEL_IDS_DIGEST,
        ),
    }
    for label, (found, wanted) in expected.items():
        if found != wanted:
            raise ActionCountCNNV3PreregistrationError(f"{label} digest differs")

    development = _ids_manifest(
        "gkm.bongard-action-count-cnn-development-panel-ids.v3",
        "exact-v2-train-and-validation-panel-identifiers-only",
        {"train": train_tasks, "validation": validation_tasks},
    )
    calibration = _ids_manifest(
        "gkm.bongard-action-count-cnn-calibration-panel-ids.v3",
        "fresh-label-free-calibration-identifiers-rank-1100-through-1199",
        {"calibration": calibration_tasks},
    )
    evaluation = _ids_manifest(
        "gkm.bongard-action-count-cnn-evaluation-panel-ids.v3",
        "fresh-label-free-evaluation-identifiers-rank-1200-through-1299",
        {"evaluation": evaluation_tasks},
    )
    manifests = {
        "development_panel_ids": (development_output_path, development),
        "calibration_panel_ids": (calibration_output_path, calibration),
        "evaluation_panel_ids": (evaluation_output_path, evaluation),
    }
    source_raw = authority_source_path.read_bytes()
    postprediction_raw = postprediction_authority_source_path.read_bytes()
    current = v2.get("current_state")
    v2_authority_bindings = v2.get("dataset_and_authority_bindings")
    if not isinstance(v2_authority_bindings, dict):
        raise ActionCountCNNV3PreregistrationError("V2 authority bindings are invalid")
    frozen_label_source_bindings = {
        key: v2_authority_bindings.get(key)
        for key in (
            "catalog_algorithm_digest",
            "catalog_audit_record_digest",
            "catalog_authority_source_sha256",
            "hd_action_program_raw_sha256",
        )
    }
    catalog_source = frozen_label_source_bindings["catalog_authority_source_sha256"]
    if (
        isinstance(catalog_source, str)
        and len(catalog_source) == 64
        and all(character in "0123456789abcdef" for character in catalog_source)
    ):
        frozen_label_source_bindings["catalog_authority_source_sha256"] = (
            "sha256:" + catalog_source
        )
    if any(
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        for value in frozen_label_source_bindings.values()
    ):
        raise ActionCountCNNV3PreregistrationError(
            "V2 frozen label-source binding is invalid"
        )
    if not isinstance(current, dict) or any(
        current.get(key) != 0
        for key in (
            "selected_panel_png_bytes_read",
            "calibration_panel_png_bytes_read",
            "evaluation_panel_png_bytes_read",
        )
    ):
        raise ActionCountCNNV3PreregistrationError("V2 pixel state is not zero")
    plan_body: dict[str, Any] = {
        "claim": (
            "metadata-only-v3-cohort-repair;_no_selected-pixel-or-fresh-target-access"
        ),
        "chronology": [
            "commit_and_replay_v3_authority_and_identifier-only_manifests",
            "retain_exact_v2_train_800_and_validation_100",
            "never_use_v2_calibration_or_evaluation_200_for_calibration_evaluation_or_selection",
            "fit_and_validate_under_the_existing_frozen_v2_training_protocol",
            "on_validation_failure_stop_with_fresh_calibration_and_evaluation_pixels_unopened",
            "after_validation_pass_precommit_then_infer_fresh_calibration_panels",
            "fsync_and_reload_complete_calibration_predictions_before_delayed_target_derivation",
            "freeze_joint_q_then_precommit_and_infer_fresh_evaluation_panels",
            "fsync_and_reload_complete_evaluation_predictions_and_sets_before_delayed_target_derivation",
        ],
        "cohorts": {
            "train": {
                "panel_count": len(train_tasks) * PANELS_PER_TASK,
                "rank_slice": [0, 800],
                "task_count": len(train_tasks),
                "task_ids": train_tasks,
                "task_ids_digest": "sha256:" + canonical_digest(train_tasks),
            },
            "validation": {
                "panel_count": len(validation_tasks) * PANELS_PER_TASK,
                "rank_slice": [800, 900],
                "task_count": len(validation_tasks),
                "task_ids": validation_tasks,
                "task_ids_digest": "sha256:" + canonical_digest(validation_tasks),
            },
            "calibration": {
                "panel_count": len(calibration_panels),
                "panel_ids_digest": EXPECTED_CALIBRATION_PANEL_IDS_DIGEST,
                "rank_slice": list(CALIBRATION_RANK_SLICE),
                "task_count": len(calibration_tasks),
                "task_ids": calibration_tasks,
                "task_ids_digest": EXPECTED_CALIBRATION_TASK_IDS_DIGEST,
            },
            "evaluation": {
                "panel_count": len(evaluation_panels),
                "panel_ids_digest": EXPECTED_EVALUATION_PANEL_IDS_DIGEST,
                "rank_slice": list(EVALUATION_RANK_SLICE),
                "task_count": len(evaluation_tasks),
                "task_ids": evaluation_tasks,
                "task_ids_digest": EXPECTED_EVALUATION_TASK_IDS_DIGEST,
            },
        },
        "current_state": {
            "fresh_action_program_or_target_rows_read": 0,
            "fresh_calibration_panel_png_bytes_read": 0,
            "fresh_evaluation_panel_png_bytes_read": 0,
            "fresh_plaintext_targets_materialized": False,
            "model_training_started": False,
            "selected_png_bytes_read_by_v3_authority": 0,
        },
        "exclusion_and_selection": {
            "eligible_hd_train_task_count": len(eligible),
            "eligible_task_ids_digest": "sha256:" + canonical_digest(eligible),
            "fresh_selection_excludes_every_v2_task_before_take": True,
            "hash_order": "sha256_utf8_seed_NUL_task_id_then_task_id",
            "metadata_excluded_task_count": len(metadata_excluded),
            "selection_seed": SELECTION_SEED,
            "semantic_exclusion": "task_id_contains_convex_or_has_four_straight_lines",
            "source_fields": "official_split_task_ids_and_metadata_exclusion_records_only",
        },
        "identifier_manifest_bindings": {
            name: _manifest_binding(repository_root, path, value)
            for name, (path, value) in manifests.items()
        },
        "metadata_source_bindings": {
            "action_count_plan_record_digest": prior["record_digest"],
            "action_count_plan_source_sha256": _address(prior_raw),
            "cumulative_exposure_ledger_digest": cumulative.digest,
            "cumulative_exposure_source_sha256": _address(cumulative_raw),
            "family_plan_record_digest": family["record_digest"],
            "family_plan_source_sha256": _address(family_raw),
            "historical_exposure_source_sha256": _address(historical_raw),
            "official_split_source_sha256": _address(split_raw),
            "v2_plan_record_digest": v2["record_digest"],
            "v2_plan_source_sha256": _address(v2_raw),
        },
        "old_v2_design_taint": {
            "all_1100_v2_tasks_excluded_from_fresh_selection": True,
            "old_calibration_and_evaluation_panel_png_bytes_read": 0,
            "old_calibration_and_evaluation_plaintext_targets_materialized": True,
            "old_calibration_task_ids": old_calibration_tasks,
            "old_evaluation_task_ids": old_evaluation_tasks,
            "old_tainted_task_count": OLD_DESIGN_TAINT_TASK_COUNT,
            "permanent": True,
            "reason": (
                "target_rows_were_materialized_before_predictions;_file_sealing_claims_"
                "cannot_restore_design_blindness"
            ),
            "reuse_allowed": False,
        },
        "postprediction_target_authority": {
            "catalog_typed_projection": {
                "axis": "catalog_convexity",
                "any_set_containing_catalog_unresolved": "whole-axis-GAP",
                "catalog_class_1_value": "catalog_nonconvex",
                "catalog_class_2_value": "catalog_convex",
                "geometric_turning_axis_used": False,
                "not_applicable_used_for_catalog_unresolved": False,
            },
            "contract_digest_function": "contract_digest",
            "entrypoint": "derive_labels_after_durable_predictions",
            "frozen_label_source_bindings": frozen_label_source_bindings,
            "prediction_schema": (
                "gkm.bongard-action-count-catalog-cnn-prelabel-predictions.v3"
            ),
            "source_path": str(
                postprediction_authority_source_path.relative_to(repository_root)
            ),
            "source_sha256": _address(postprediction_raw),
            "source_loader_invoked_only_after_fsync_and_byte-identical_reload": True,
        },
        "preregistration_authority": {
            "source_path": str(authority_source_path.relative_to(repository_root)),
            "source_sha256": _address(source_raw),
        },
        "schema": SCHEMA,
        "supersession": {
            "retained_exactly": ["v2_train_800", "v2_validation_100"],
            "v2_calibration_100_status": "permanently_design_tainted",
            "v2_evaluation_100_status": "permanently_design_tainted",
            "v2_plan_record_digest": v2["record_digest"],
        },
    }
    plan = {**plan_body, "record_digest": "sha256:" + canonical_digest(plan_body)}
    return plan, development, calibration, evaluation


def write_outputs(paths: Sequence[Path], values: Sequence[Mapping[str, Any]]) -> None:
    if len(paths) != len(values):
        raise ActionCountCNNV3PreregistrationError("output cardinality differs")
    for path, value in zip(paths, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json(value) + b"\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repository-root",
        "v2-plan",
        "action-count-plan",
        "family-plan",
        "historical-exposure",
        "cumulative-exposure-ledger",
        "split",
        "plan-output",
        "development-output",
        "calibration-output",
        "evaluation-output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = args.repository_root.resolve()
    values = build_v3_preregistration(
        repository_root=repository_root,
        authority_source_path=Path(__file__).resolve(),
        postprediction_authority_source_path=(
            repository_root
            / "bongard/panel_action_count_cnn_postprediction_labels_v3.py"
        ),
        v2_plan_path=args.v2_plan.resolve(),
        action_count_plan_path=args.action_count_plan.resolve(),
        family_plan_path=args.family_plan.resolve(),
        historical_exposure_path=args.historical_exposure.resolve(),
        cumulative_exposure_ledger_path=args.cumulative_exposure_ledger.resolve(),
        split_path=args.split.resolve(),
        development_output_path=args.development_output.resolve(),
        calibration_output_path=args.calibration_output.resolve(),
        evaluation_output_path=args.evaluation_output.resolve(),
    )
    paths = (
        args.plan_output.resolve(),
        args.development_output.resolve(),
        args.calibration_output.resolve(),
        args.evaluation_output.resolve(),
    )
    write_outputs(paths, values)
    print(json.dumps([value["record_digest"] for value in values]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

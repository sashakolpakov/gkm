from __future__ import annotations

import json
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from bongard.canonical import canonical_digest
from bongard.panel_convexity_catalog_audit import (
    CatalogBinding,
    ConvexityCatalogError,
    RAW_LABEL_TO_CLASS,
    audit_cohorts,
    build_catalog_binding,
    catalog_label_for_actions,
    convexity_catalog_algorithm_digest,
    convexity_catalog_source_digest,
    signature_from_actions,
    signature_from_shape_row,
)


def _panel(action: str) -> list[list[str]]:
    return [[action]]


def _task(positive: str, negative: str) -> list[list[list[list[str]]]]:
    return [
        [_panel(positive) for _ in range(7)],
        [_panel(negative) for _ in range(7)],
    ]


def _rows(alias_label: str = "-1") -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    shapes = [
        {
            "shape function name": "direct_shape",
            "set of base actions": "line_1.0",
            "turn angles": "L0",
        },
        {
            "shape function name": "stale_shape",
            "set of base actions": "line_2.0",
            "turn angles": "L0",
        },
    ]
    attrs = [
        {"shape function name": "direct_shape", "convex": "1"},
        {"shape function name": "stale_shape", "convex": alias_label},
    ]
    return shapes, attrs


def test_signatures_remove_only_style() -> None:
    row = {
        "shape function name": "shape",
        "set of base actions": "line_1.0, arc_0.5_120",
        "turn angles": "L0--R60",
    }
    expected = ("line:1.000:0.500", "arc:0.500:0.667:0.333")
    assert signature_from_shape_row(row) == expected
    assert signature_from_actions(
        ["line_triangle_1.000-0.500", "arc_zigzag_0.500_0.667-0.333"]
    ) == expected


def test_bd_singleton_alias_is_exact_fail_closed_and_auditable() -> None:
    shapes, attrs = _rows()
    direct = "line_circle_1.000-0.500"
    stale_release = "line_zigzag_3.000-0.500"
    hd = {"hd_task_0000": _task(stale_release, direct)}
    bd = {"bd_stale_shape_0000": _task(stale_release, direct)}
    binding = build_catalog_binding(
        shape_rows=shapes,
        attribute_rows=attrs,
        hd_programs=hd,
        bd_programs=bd,
    )

    assert len(binding.direct_by_signature) == 2
    assert len(binding.alias_by_signature) == 1
    assert len(binding.alias_proofs) == 1
    alias_label = catalog_label_for_actions([stale_release], binding)
    assert alias_label.shape_function_name == "stale_shape"
    assert alias_label.raw_label == "-1"
    assert alias_label.supervised_class == "catalog_unresolved"
    assert alias_label.match_kind == "bd_singleton_compatibility_alias"

    audit = audit_cohorts(
        programs=hd,
        cohorts={"train": ["hd_task_0000"]},
        binding=binding,
    )["train"]
    assert audit["task_count"] == 1
    assert audit["panel_count"] == 14
    assert audit["all_14_direct_exact_signature_task_count"] == 0
    assert audit["all_14_catalog_labelled_with_compatibility_task_count"] == 1
    assert audit["all_14_binary_0_or_1_task_count"] == 0
    assert audit["label_counts"] == {"catalog_unresolved": 7, "convex": 7}

    with pytest.raises(TypeError):
        binding.raw_label_by_name["direct_shape"] = "0"  # type: ignore[index]
    with pytest.raises(TypeError):
        alias_signature = next(iter(binding.alias_by_signature))
        binding.alias_by_signature[alias_signature] = "direct_shape"  # type: ignore[index]
    with pytest.raises(TypeError):
        binding.alias_proofs[0]["raw_convexity_label"] = "0"  # type: ignore[index]
    with pytest.raises(TypeError):
        RAW_LABEL_TO_CLASS["-1"] = "nonconvex"  # type: ignore[index]
    with pytest.raises(ConvexityCatalogError, match="must be reconstructed"):
        replace(binding, raw_label_by_name={"direct_shape": "0", "stale_shape": "-1"})


def test_direct_binding_construction_is_rejected() -> None:
    with pytest.raises(ConvexityCatalogError, match="must be reconstructed"):
        CatalogBinding(
            direct_by_signature={("line:1.000:0.500",): "forged"},
            raw_label_by_name={"forged": "1"},
            alias_by_signature={},
            alias_proofs=(),
            hd_missing_signature_counts={},
        )


def test_alias_cannot_turn_a_version_mismatch_into_binary_truth() -> None:
    shapes, attrs = _rows(alias_label="0")
    direct = "line_normal_1.000-0.500"
    stale_release = "line_normal_3.000-0.500"
    with pytest.raises(ConvexityCatalogError, match="target only raw -1"):
        build_catalog_binding(
            shape_rows=shapes,
            attribute_rows=attrs,
            hd_programs={"hd_task_0000": _task(stale_release, direct)},
            bd_programs={"bd_stale_shape_0000": _task(stale_release, direct)},
        )


def test_missing_signature_requires_one_named_singleton_proof() -> None:
    shapes, attrs = _rows()
    direct = "line_normal_1.000-0.500"
    stale_release = "line_normal_3.000-0.500"
    with pytest.raises(ConvexityCatalogError, match="exactly one BD singleton"):
        build_catalog_binding(
            shape_rows=shapes,
            attribute_rows=attrs,
            hd_programs={"hd_task_0000": _task(stale_release, direct)},
            bd_programs={},
        )


def test_checked_in_live_audit_is_self_consistent_and_source_bound() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "data/panel_convexity_catalog_audit_20260810_v1.json"
    )
    value = json.loads(path.read_bytes())
    record_digest = value.pop("record_digest")
    assert record_digest == "sha256:" + canonical_digest(value)
    assert value["algorithm"]["source_sha256"] == convexity_catalog_source_digest()
    assert value["algorithm"]["algorithm_digest"] == (
        convexity_catalog_algorithm_digest()
    )
    assert value["inventory"]["direct_signature_count"] == 627
    assert value["inventory"]["hd_panel_count"] == 61_600
    assert value["inventory"]["hd_panel_count_with_compatibility"] == 61_600
    assert value["compatibility"]["alias_count"] == 4
    assert {
        row["shape_function_name"] for row in value["compatibility"]["alias_proofs"]
    } == {
        "exist_triangle_three_lines4",
        "open_symm_bridge",
        "thin_rec_down_right_triangle",
        "thin_rec_right_triangle",
    }
    assert value["claim_limits"]["catalog_unresolved_downstream_disposition"] == "GAP"


def test_v2_cohort_audit_is_bound_without_opening_sealed_labels() -> None:
    repository = Path(__file__).resolve().parents[2]
    path = (
        repository
        / "bongard/data/panel_convexity_catalog_v2_cohort_audit_20260810_v1.json"
    )
    value = json.loads(path.read_bytes())
    record_digest = value.pop("record_digest")
    assert record_digest == "sha256:" + canonical_digest(value)

    plan_path = repository / value["v2_preregistration_binding"]["path"]
    plan_raw = plan_path.read_bytes()
    assert value["v2_preregistration_binding"]["raw_sha256"] == (
        "sha256:" + hashlib.sha256(plan_raw).hexdigest()
    )
    plan = json.loads(plan_raw)
    assert value["v2_preregistration_binding"]["record_digest"] == plan["record_digest"]
    for cohort_name, audit in value["cohorts"].items():
        assert audit["combined_rows_digest_matches_v2_plan"] is True
        assert audit["combined_action_and_catalog_rows_digest"] == plan["cohorts"][
            cohort_name
        ]["action_and_catalog_label_rows_digest"]
        assert audit["all_14_catalog_labelled_with_compatibility_task_count"] == audit[
            "task_count"
        ]
    assert value["custody"] == {
        "calibration_or_evaluation_sealed_label_payloads_read_for_this_independent_audit": False,
        "new_panel_png_bytes_read": 0,
        "official_validation_or_test_pixels_read": False,
        "target_family_pixels_read": False,
        "v2_selected_panel_png_bytes_read": 0,
    }

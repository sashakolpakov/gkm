from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


DATA = Path(__file__).resolve().parents[1] / "data"
RECORD = DATA / "salience_cap_telemetry_exposure_20260809_v1.json"

TASK_IDS = (
    "bd_open_arc_line_arc1_0000",
    "bd_bird7-square_dagger5_0000",
    "bd_exist_quadrangle_five_lines3-necked_six_lines2_0000",
    "bd_irregular_jar_triangle2-necked_six_lines4_0000",
    "bd_open_s1-thin_sym_nail3_0000",
    "bd_two_updown_obtuse_triangles-open_triangle3_0000",
    "bd_sector330-symmetric_unbala_x_0000",
    "bd_band_three_arcs1-necked_six_lines4_0000",
    "bd_quasi_sector270-symmetric_bala_x_0000",
    "bd_unbala_three_intersect_circles1-thin_seven_lines6_0000",
    "ff_nact7_0043",
    "ff_nact2_5_0208",
    "ff_nact4_4_0133",
    "ff_nact6_0112",
    "ff_nact3_5_0015",
    "ff_nact5_0205",
    "ff_nact7_0231",
    "ff_nact4_4_0155",
    "ff_nact8_0186",
    "ff_nact6_0268",
    "hd_symmetric-exist_triangle_0017",
    "hd_has_three_straight_lines-has_four_straight_lines_0016",
    "hd_has_three_straight_lines-has_obtuse_angle_0005",
    "hd_has_five_straight_lines-has_seven_straight_lines_0017",
    "hd_has_six_straight_lines-exist_sector_0001",
    "hd_unbalanced_two_0010",
    "hd_convex-has_three_straight_lines_0000",
    "hd_has_curve-symmetric_0001",
    "hd_has_curve-has_obtuse_angle_0012",
    "hd_has_curve-symmetric_0011",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _lf_digest(values: tuple[str, ...]) -> str:
    return "sha256:" + hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def test_salience_cap_telemetry_exposure_record_is_strict_and_bound() -> None:
    raw = RECORD.read_bytes()
    record = json.loads(raw)
    assert raw == _canonical_json(record) + b"\n"

    assert set(record) == {
        "access_boundary",
        "counts",
        "exposure_effect",
        "panel_membership",
        "predecessor_exposure_ledger_digest",
        "provenance",
        "record_digest",
        "record_digest_policy",
        "schema",
        "task_membership",
        "telemetry_cap_results",
    }
    assert record["schema"] == "gkm.bongard-salience-cap-telemetry-exposure.v1"
    assert record["predecessor_exposure_ledger_digest"] == (
        "sha256:71489b47cfedcef9ab220bef740f307832c4bfd22363610eabdce3e3ec86bd6d"
    )
    digest_input = dict(record)
    declared_record_digest = digest_input.pop("record_digest")
    assert record["record_digest_policy"] == (
        "sha256(canonical JSON of this object with record_digest omitted)"
    )
    assert declared_record_digest == (
        "sha256:" + hashlib.sha256(_canonical_json(digest_input)).hexdigest()
    ) == "sha256:74d768982ed30149929840662b5bb156fce4819a1e5fa243afd7559c6b46c0af"

    task_membership = record["task_membership"]
    assert set(task_membership) == {
        "digest",
        "digest_algorithm",
        "family_task_counts",
        "reported_telemetry_cohort_digest",
        "reported_telemetry_cohort_digest_is_membership_binding",
        "reported_telemetry_cohort_digest_note",
        "task_ids",
    }
    task_ids = tuple(task_membership["task_ids"])
    assert task_ids == TASK_IDS
    assert len(task_ids) == len(set(task_ids)) == 30
    assert Counter(task[:2] for task in task_ids) == {"bd": 10, "ff": 10, "hd": 10}
    assert task_membership["family_task_counts"] == {"bd": 10, "ff": 10, "hd": 10}
    assert task_membership["digest_algorithm"] == (
        "sha256(UTF-8(task_ids joined by LF with no terminal LF))"
    )
    assert _lf_digest(task_ids) == task_membership["digest"] == (
        "sha256:8a33630b7132fdaff8c3a1b366a473b6b389b481f483a25fd3128861bb5f5bb0"
    )
    assert task_membership["reported_telemetry_cohort_digest"] == (
        "sha256:8867d9e3feb673831a06aac8dc5e5499cb58d22063659bb8fd21d121c345c64b"
    )
    assert task_membership["reported_telemetry_cohort_digest_is_membership_binding"] is False
    assert task_membership["reported_telemetry_cohort_digest"] != task_membership["digest"]
    assert "does not equal" in task_membership["reported_telemetry_cohort_digest_note"]

    panel_membership = record["panel_membership"]
    assert set(panel_membership) == {
        "derivation_order",
        "digest",
        "digest_algorithm",
        "family_rule",
        "id_template",
        "panel_ids",
        "side_indices",
        "support_indices",
    }
    assert panel_membership["side_indices"] == [0, 1]
    assert panel_membership["support_indices"] == [1, 2, 3, 4, 5, 6]
    expected_panel_ids = tuple(
        f"{task[:2]}/{task}/{side}/{index}.png"
        for task in TASK_IDS
        for side in (0, 1)
        for index in range(1, 7)
    )
    panel_ids = tuple(panel_membership["panel_ids"])
    assert panel_ids == expected_panel_ids
    assert len(panel_ids) == len(set(panel_ids)) == 360
    assert panel_membership["digest_algorithm"] == (
        "sha256(UTF-8(panel_ids joined by LF with no terminal LF))"
    )
    assert _lf_digest(panel_ids) == panel_membership["digest"] == (
        "sha256:3a136ea28327302be38c19136ff241df3d5c39091f33c587d1982196a236abb0"
    )
    for panel_id in panel_ids:
        family, task_id, side, filename = panel_id.split("/")
        assert family == task_id[:2] in {"bd", "ff", "hd"}
        assert side in {"0", "1"}
        assert filename in {f"{index}.png" for index in range(1, 7)}
        assert filename != "0.png"

    access = record["access_boundary"]
    assert set(access) == {
        "declared_split",
        "official_test_accessed",
        "official_test_panels_accessed",
        "query_index",
        "query_panels_accessed",
        "support_indices",
        "support_only",
    }
    assert access == {
        "declared_split": "train",
        "official_test_accessed": False,
        "official_test_panels_accessed": 0,
        "query_index": 0,
        "query_panels_accessed": 0,
        "support_indices": [1, 2, 3, 4, 5, 6],
        "support_only": True,
    }
    assert record["exposure_effect"] == {
        "exclude_tasks_from_future_unused_evaluation": True,
        "research_exposed": True,
        "scope": "task-level-conservative-from-support-panel-access",
    }

    counts = record["counts"]
    assert set(counts) == {
        "panels_with_resource_cap_exceeded",
        "proposal_states",
        "proposals",
        "resource_cap_exceeded_proposals",
        "semantic_cap_exceeded_proposals",
        "support_panels",
        "tasks",
        "tasks_with_resource_cap_exceeded",
        "zero_proposal_panels",
    }
    assert counts == {
        "panels_with_resource_cap_exceeded": 12,
        "proposal_states": {"clean": 666, "error": 0, "exception": 0, "indeterminate": 34},
        "proposals": 700,
        "resource_cap_exceeded_proposals": 13,
        "semantic_cap_exceeded_proposals": 21,
        "support_panels": 360,
        "tasks": 30,
        "tasks_with_resource_cap_exceeded": 8,
        "zero_proposal_panels": 76,
    }
    assert sum(counts["proposal_states"].values()) == counts["proposals"]

    telemetry = record["telemetry_cap_results"]
    assert set(telemetry) == {
        "counterfactuals_replayed_from_observed_work_bounds",
        "maximum_observed_estimated_work_units",
        "resource_cap_results",
        "work_bound",
    }
    assert telemetry["counterfactuals_replayed_from_observed_work_bounds"] is True
    assert telemetry["maximum_observed_estimated_work_units"] == 515_172_840
    assert telemetry["resource_cap_results"] == [
        {
            "estimated_work_unit_cap": 2**28,
            "mode": "executed",
            "resource_cap_exceeded_proposals": 13,
        },
        {
            "estimated_work_unit_cap": 2**29,
            "mode": "counterfactual",
            "resource_cap_exceeded_proposals": 0,
        },
        {
            "estimated_work_unit_cap": 2**30,
            "mode": "counterfactual",
            "resource_cap_exceeded_proposals": 0,
        },
    ]

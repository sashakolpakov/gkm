"""Exact metadata-only validation for the convex/four-lines family drill."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bongard.canonical import canonical_digest, canonical_json
from bongard.corpus import SplitIndex
from bongard.release import load_official_release


SEMANTIC = "hd_convex-has_four_straight_lines"
BONGARD_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = BONGARD_ROOT.parent
PLAN_PATH = (
    BONGARD_ROOT
    / "data/panel_convex_four_lines_same_family_train_drill_20260809_v1.json"
)


def _task_ids(first: int, last: int) -> list[str]:
    return [f"{SEMANTIC}_{index:04d}" for index in range(first, last + 1)]


def _support(task_id: str) -> list[str]:
    return [
        f"hd/{task_id}/{side}/{index}.png"
        for side in (1, 0)
        for index in (0, 1, 2, 3, 5, 6)
    ]


def _query(task_id: str) -> list[str]:
    return [f"hd/{task_id}/{side}/4.png" for side in (1, 0)]


def test_same_family_drill_plan_is_exact_canonical_and_fail_closed() -> None:
    raw = PLAN_PATH.read_bytes()
    plan = json.loads(raw)
    assert raw == canonical_json(plan) + b"\n"
    body = dict(plan)
    record_digest = body.pop("record_digest")
    assert record_digest == "sha256:" + canonical_digest(body)
    assert record_digest == (
        "sha256:a8a94af6b430018ce9f80550a7bce910c8095c781d4fa51113abdb505a1e3cd7"
    )

    partition = plan["frozen_partition"]
    assert partition == {
        "development_train_task_ids": _task_ids(2, 13),
        "exposed_diagnostic_task_ids": _task_ids(1, 1),
        "heldout_engineering_validation_train_task_ids": _task_ids(14, 17),
        "reserved_untouched_target_train_task_ids": _task_ids(0, 0),
        "sealed_validation_task_ids": _task_ids(18, 19),
    }
    for name, first, last in (
        ("development", 2, 13),
        ("heldout_engineering_validation", 14, 17),
    ):
        cohort = plan["cohorts"][name]
        expected_tasks = _task_ids(first, last)
        expected_support = [panel for task in expected_tasks for panel in _support(task)]
        expected_query = [panel for task in expected_tasks for panel in _query(task)]
        assert cohort["task_ids"] == expected_tasks
        assert cohort["support_panel_ids"] == expected_support
        assert cohort["query_panel_ids"] == expected_query
        assert cohort["support_panel_count"] == 12 * len(expected_tasks)
        assert cohort["query_panel_count"] == 2 * len(expected_tasks)
        assert not set(expected_support).intersection(expected_query)
    authorized_panels = {
        panel
        for cohort in plan["cohorts"].values()
        for field in ("support_panel_ids", "query_panel_ids")
        for panel in cohort[field]
    }
    assert len(authorized_panels) == 16 * 14
    assert all(f"/{SEMANTIC}_0000/" not in panel for panel in authorized_panels)
    assert all(f"/{SEMANTIC}_0001/" not in panel for panel in authorized_panels)
    assert all(f"/{SEMANTIC}_0018/" not in panel for panel in authorized_panels)
    assert all(f"/{SEMANTIC}_0019/" not in panel for panel in authorized_panels)

    release = load_official_release()
    bindings = plan["dataset_bindings"]
    assert bindings["split_source_digest"] == release.split_sha256
    assert bindings["task_inventory_digest"] == release.task_ids_sha256
    assert bindings["corpus_manifest_digest"] == release.corpus_manifest_sha256
    split_path = (
        REPOSITORY_ROOT
        / "downloads/ShapeBongard_V2_full/ShapeBongard_V2/ShapeBongard_V2_split.json"
    )
    if split_path.is_file():
        split = SplitIndex.load(split_path)
        assert "sha256:" + canonical_digest(split.to_manifest_dict()) == bindings[
            "split_manifest_digest"
        ]
        assert all(split.assignment(task).split == "train" for task in _task_ids(0, 17))
        assert all(split.assignment(task).split == "val" for task in _task_ids(18, 19))

    stack = plan["protocol_stack"]
    for name in ("component_observation", "zoom"):
        binding = stack[name]
        artifact_raw = (REPOSITORY_ROOT / binding["artifact_path"]).read_bytes()
        artifact = json.loads(artifact_raw)
        artifact_body = dict(artifact)
        artifact_digest = artifact_body.pop("record_digest")
        assert artifact_digest == "sha256:" + canonical_digest(artifact_body)
        assert artifact_digest == binding["protocol_digest"]
        assert "sha256:" + hashlib.sha256(artifact_raw).hexdigest() == binding[
            "artifact_source_sha256"
        ]
    atom = stack["atom_slate"]
    atom_source = (REPOSITORY_ROOT / atom["source_path"]).read_bytes()
    assert hashlib.sha256(atom_source).hexdigest() == atom["source_sha256"]
    atom_contract = {
        "schema": "gkm.bongard-positive-atom-slate-protocol-binding.v1",
        "protocol_id": atom["protocol_id"],
        "source_sha256": atom["source_sha256"],
        "atom_count": atom["atom_count"],
        "enumerated_formula_count": atom["enumerated_formula_count"],
        "formula_order": atom["formula_order"],
        "present_when_lower_at_least": atom["present_when_lower_at_least"],
        "certified_absent_when_upper_at_most": atom[
            "certified_absent_when_upper_at_most"
        ],
        "minimum_decisive_per_side": atom["minimum_decisive_per_side"],
        "logical_negation_or_polarity_operator_present": False,
        "model_selected_formula": False,
        "python_is_canonical_authority": True,
        "lean_present": False,
    }
    assert atom["protocol_digest"] == "sha256:" + canonical_digest(atom_contract)
    stack_contract = {
        "schema": "gkm.bongard-convex-four-lines-drill-protocol-stack.v1",
        "component_protocol_digest": stack["component_observation"]["protocol_digest"],
        "zoom_protocol_digest": stack["zoom"]["protocol_digest"],
        "atom_slate_protocol_digest": atom["protocol_digest"],
    }
    assert stack["protocol_stack_digest"] == "sha256:" + canonical_digest(
        stack_contract
    )

    assert plan["metadata_only_preregistration"] is True
    assert plan["new_panel_pixels_read_before_commit"] is False
    assert plan["action_programs_authorized"] is False
    assert plan["query_identities_committed_before_support_pixels"] is True
    assert plan["query_pixels_opened_before_formula_freeze"] is False
    assert plan["target_0000_panel_pixels_authorized"] is False
    assert plan["validation_0018_0019_panel_pixels_authorized"] is False
    assert plan["official_test_authorized"] is False
    assert plan["python_is_canonical_authority"] is True
    assert plan["lean_present"] is False
    assert plan["lean_required"] is False
    assert plan["lean_removable"] is True
    before = plan["exposure_accounting"]["before"]
    after = plan["exposure_accounting"]["maximum_after_authorized_cohorts"]
    assert (before["exposed_train"], before["exposed_validation"], before["exposed_official_test"]) == (290, 24, 0)
    assert (after["exposed_train"], after["exposed_validation"], after["exposed_official_test"]) == (306, 24, 0)
    assert (after["exact_unused_train"], after["exact_unused_validation"], after["sealed_official_test"]) == (8994, 876, 1800)

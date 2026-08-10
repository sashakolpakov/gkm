from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bongard.canonical import canonical_digest, canonical_json
from bongard.panel_action_local_duplicate_conflict_audit import (
    AUDIT_SCHEMA,
    DuplicateTargetAuditError,
    _summarize_cohort,
    verify_duplicate_target_conflict_audit,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT
    / "bongard/data/panel_action_local_duplicate_conflict_audit_20260810_v1.json"
)
LIVE_PRECOMMIT = (
    ROOT
    / "downloads/ShapeBongard_V2_full/panel_action_count_cnn_fit_20260810_v3/"
    "fit_pixel_precommit.json"
)
SOURCE = ROOT / "bongard/panel_action_local_duplicate_conflict_audit.py"


def _address(character: str) -> str:
    return "sha256:" + character * 64


def _row(panel: str, target: str, label=(4, 1, -1)) -> dict[str, object]:
    return {
        "fit_cohort": "train",
        "label_triple": list(label),
        "panel_id": panel,
        "png_sha256": _address("a"),
        "pose_free_target_digest": target,
    }


def test_target_conflict_is_ineligible_even_when_count_and_catalog_agree():
    summary = _summarize_cohort(
        [_row("p0", _address("1")), _row("p1", _address("2"))]
    )
    assert summary["pose_free_target_conflicts"]["group_count"] == 1
    assert summary["pose_free_target_conflicts"]["occurrence_count"] == 2
    assert summary["action_count_conflicts"]["straight"]["group_count"] == 0
    assert summary["action_count_conflicts"]["arc"]["group_count"] == 0
    assert summary["action_count_conflicts"]["straight_arc_pair"]["group_count"] == 0
    assert summary["catalog_convexity_conflicts"]["group_count"] == 0
    assert summary["descriptor_loss_eligibility"]["eligible_group_count"] == 0
    assert summary["descriptor_loss_eligibility"]["ineligible_group_count"] == 1
    assert summary["descriptor_loss_eligibility"]["ineligible_occurrence_count"] == 2


def test_action_count_and_catalog_conflicts_are_reported_separately():
    rows = [
        _row("p0", _address("1"), label=(4, 1, -1)),
        _row("p1", _address("1"), label=(5, 1, 1)),
    ]
    summary = _summarize_cohort(rows)
    assert summary["pose_free_target_conflicts"]["group_count"] == 0
    assert summary["action_count_conflicts"]["straight"]["group_count"] == 1
    assert summary["action_count_conflicts"]["arc"]["group_count"] == 0
    assert summary["action_count_conflicts"]["straight_arc_pair"]["group_count"] == 1
    assert summary["catalog_convexity_conflicts"]["group_count"] == 1
    # Descriptor eligibility is about the pose-free target only.  A consumer
    # must still apply the separately reported classification-label gates.
    assert summary["descriptor_loss_eligibility"]["eligible_group_count"] == 1


@pytest.mark.skipif(not LIVE_PRECOMMIT.exists(), reason="frozen local fit absent")
def test_committed_audit_is_canonical_and_exactly_replays_without_png_bytes():
    raw = ARTIFACT.read_bytes()
    artifact = json.loads(raw)
    assert raw == canonical_json(artifact) + b"\n"
    body = dict(artifact)
    digest = body.pop("record_digest")
    assert digest == "sha256:" + canonical_digest(body)
    assert artifact["schema"] == AUDIT_SCHEMA
    assert artifact["bindings"]["audit_source_sha256"] == (
        "sha256:" + hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    )
    verify_duplicate_target_conflict_audit(artifact, repository_root=ROOT)

    assert artifact["custody"] == {
        "action_program_scope": "frozen_v3_train_and_validation_only",
        "calibration_evaluation_family_or_target_identifiers_opened": 0,
        "fit_precommit_exact_png_digest_metadata_read": 12_600,
        "label_source": "label_triples_already_frozen_in_fit_precommit",
        "png_bytes_read": 0,
        "pose_free_target_source": "committed_development_authority_8fd4de9a",
        "removed_validation_occurrences_not_joined": 8,
    }
    assert artifact["result"] == {
        "all_effective_png_groups_descriptor_loss_eligible": True,
        "catalog_convexity_conflict_group_count": 0,
        "pose_free_target_conflict_group_count": 0,
        "scope": "training_and_effective_validation_only",
        "straight_arc_pair_conflict_group_count": 0,
    }
    train = artifact["cohorts"]["train"]
    assert (
        train["occurrence_count"],
        train["png_digest_group_count"],
        train["duplicate_png_digest_group_count"],
        train["duplicate_png_occurrence_count"],
    ) == (11_200, 11_143, 57, 114)
    assert train["pose_free_target_cardinality_histogram_by_png_group"] == {
        "1": 11_143
    }
    assert train["pose_free_target_conflicts"]["group_count"] == 0
    assert train["descriptor_loss_eligibility"]["eligible_group_count"] == 11_143
    assert train["descriptor_loss_eligibility"]["eligible_occurrence_count"] == 11_200
    assert train["descriptor_loss_eligibility"]["ineligible_group_count"] == 0

    validation = artifact["cohorts"]["validation"]
    assert (
        validation["occurrence_count"],
        validation["png_digest_group_count"],
        validation["duplicate_png_digest_group_count"],
        validation["duplicate_png_occurrence_count"],
    ) == (1_392, 1_392, 0, 0)
    assert validation["pose_free_target_cardinality_histogram_by_png_group"] == {
        "1": 1_392
    }
    assert validation["pose_free_target_conflicts"]["group_count"] == 0
    assert validation["descriptor_loss_eligibility"]["eligible_group_count"] == 1_392
    assert validation["descriptor_loss_eligibility"]["ineligible_group_count"] == 0

    tampered = json.loads(raw)
    tampered["result"]["pose_free_target_conflict_group_count"] = 1
    with pytest.raises(DuplicateTargetAuditError, match="record digest"):
        verify_duplicate_target_conflict_audit(tampered, repository_root=ROOT)

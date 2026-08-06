from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bongard import load_historical_exposure, load_official_release
from bongard.artifacts import canonical_json
from bongard.prototype_calibration import PrototypeCalibrationRecord


DATA = Path(__file__).resolve().parents[1] / "data"


def _canonical_record(name: str) -> dict[str, object]:
    payload = (DATA / name).read_bytes()
    value = json.loads(payload)
    assert isinstance(value, dict)
    assert payload == canonical_json(value) + b"\n"
    return value


def test_checked_in_complete_corpus_cohort_summary_is_bound_to_frozen_inputs() -> None:
    record = _canonical_record("shape_bongard_v2_cohort_summary_v1.json")
    release = load_official_release()
    historical = load_historical_exposure()

    assert record["schema"] == "gkm.shape-bongard-v2-cohort-summary.v1"
    assert record["release_descriptor_digest"] == release.digest
    assert record["historical_seed_digest"] == historical.seed_digest
    counts = record["counts"]
    assert isinstance(counts, dict)
    assert counts["tasks"] == 12_000
    assert counts["ff"] + counts["bd"] + counts["hd"] == 12_000
    assert counts["historically_clean"] == (
        counts["drill"]
        + counts["dev"]
        + counts["sealed"]
    )
    assert (counts["drill"], counts["dev"], counts["sealed"]) == (
        2_769,
        542,
        557,
    )
    assert record["cohort_report_digest"] == (
        "sha256:55de04a582ffa3a4fbf26466ab88f265ddd7839ae10004210cca4d9ffa4f8e9d"
    )
    assert record["membership_digests"] == {
        "all": "sha256:4503ae6b40dc7b34520eb5b8a4cca6ff8153635df0f42db5f6715cc349602dd0",
        "dev": "sha256:ea0334e538cfe3b6fed58fa4d575f85ac077afdb650fcb8d8043d0337f8d3f74",
        "drill": "sha256:15c95adcffe7e858b8007a3b7f20df4acb5c6fdaa7a994f2a32a5aa595abdbe9",
        "exact_task_recorded": (
            "sha256:4c9b9236c62d2e2b8e6f43dbe7297aeab380f45f473061bc80bcc7fd525b7477"
        ),
        "historically_clean": (
            "sha256:0c1bc85f24bf7491a3882c4994e57aaf50e0b66cf31fd92476d15341399b0336"
        ),
        "sealed": "sha256:e130e2281ce0209ee35ce292e1d7abd7c184f469699f99710f3fd82230ae30d0",
    }
    assert "do not certify" in str(record["qualification"])


def test_checked_in_complete_corpus_image_audit_is_strict_and_bound() -> None:
    record = _canonical_record("shape_bongard_v2_image_audit_v1.json")
    release = load_official_release()

    assert record["schema"] == "gkm.shape-bongard-image-audit.v1"
    assert record["corpus_manifest_digest"] == release.corpus_manifest_sha256
    assert record["task_count"] == 12_000
    assert record["panel_count"] == 168_000
    assert record["family_task_counts"] == {"bd": 4_000, "ff": 3_600, "hd": 4_400}
    assert record["family_panel_counts"] == {
        "bd": 56_000,
        "ff": 50_400,
        "hd": 61_600,
    }
    assert record["format_counts"] == {"PNG": 168_000}
    assert record["mode_counts"] == {"RGB": 168_000}
    assert record["size_counts"] == [
        {"count": 168_000, "height": 512, "width": 512}
    ]
    assert record["info_key_set_counts"] == [
        {"count": 168_000, "info_keys": []}
    ]
    assert record["frame_count_counts"] == [
        {"count": 168_000, "frame_count": 1}
    ]
    assert record["expectations"] == {
        "frame_count": 1,
        "height": 512,
        "info_keys": [],
        "mode": "RGB",
        "width": 512,
    }
    assert record["require_expected_properties"] is True
    assert record["anomaly_count"] == 0
    assert record["anomalies"] == []
    assert record["anomalies_truncated"] is False
    assert record["digest"] == (
        "sha256:d3485ada3605d708db82fbcfe6ecfc73506ce51ed85fcd1ce6ccd798e3bff9f8"
    )


def test_checked_in_support_prototype_calibration_is_canonical_and_bound() -> None:
    data = _canonical_record("support_prototype_calibration_v1.json")
    record = PrototypeCalibrationRecord.from_data(data)

    assert record.digest() == (
        "cf02d58ab57fe1b44201c67d06f00faf06e77374b762c81ff5f61ef20aef93b6"
    )
    assert record.to_freeze_policy().digest() == (
        "8bb04e21b2ac59c2391105c1a0a729e87842e956f3116323f2228d291d8f119e"
    )
    assert data["seed"] == "prototype-calibration-v1"
    assert data["candidate_margin_grid"] == [1e-9, 1e-6, 1e-4, 1e-3, 1e-2]
    assert data["task_ids"] == [
        "bd_asymmetric_clamp_0000",
        "bd_inverse_trap_arc180_0000",
        "bd_open_square_right_triangle_0000",
        "bd_open_symm_trans_arc_lamp_0000",
        "bd_symm_unbala_goldfish_0000",
        "bd_two_symm_bala_quadrangles-open_band_four_arcs1_0000",
        "hd_has_five_straight_lines-has_seven_straight_lines_0018",
        "hd_has_four_straight_lines-closed_shape_0013",
        "hd_has_line_crossing-exist_regular_0018",
        "hd_has_seven_straight_lines-has_line_crossing_0019",
        "hd_has_six_straight_lines-symmetric_transposed_0005",
        "hd_unbalanced_two-exist_triangle_0012",
    ]
    assert len(data["tasks"]) == 12
    assert all(task["declared_split"] != "test" for task in data["tasks"])


def test_checked_in_support_prototype_drill_plan_is_canonical_and_bound() -> None:
    plan = _canonical_record("support_prototype_drill_plan_v1.json")
    content = dict(plan)
    declared_digest = content.pop("digest")
    computed_digest = "sha256:" + hashlib.sha256(
        canonical_json(content) + b"\n"
    ).hexdigest()
    calibration = PrototypeCalibrationRecord.from_data(
        _canonical_record("support_prototype_calibration_v1.json")
    )
    release = load_official_release()

    assert declared_digest == computed_digest == (
        "sha256:f04dbccc9b3518f0df69c1fa4566d98653de6354c48e50f4ccc80365b8c9c67b"
    )
    assert plan["schema"] == "gkm.bongard-support-prototype-drill-plan.v1"
    assert plan["calibration_record_digest"] == "sha256:" + calibration.digest()
    assert plan["predicate_policy_digest"] == (
        "sha256:" + calibration.to_freeze_policy().digest()
    )
    assert plan["official_release_descriptor_digest"] == release.digest
    assert plan["corpus_manifest_digest"] == release.corpus_manifest_sha256
    assert plan["split_source_digest"] == release.split_sha256
    assert plan["exposure_ledger_head_digest"] == (
        "sha256:da01c133c87c551e01b581578b55d40283be0c62cbb23dddc18c5dc873b1ec9a"
    )
    assert plan["exposure_ledger_head_event_count"] == 16
    assert len(plan["task_ids"]) == len(set(plan["task_ids"])) == 12
    assert plan["first_task_id"] == plan["task_ids"][0]
    assert plan["selection_boundary"] == {
        "allowed_primary_splits": ["train", "val"],
        "expected_semantic_cohort": "drill",
        "official_test_pixels_authorized": False,
        "require_live_exact_and_semantic_unseen": True,
    }
    assert plan["scoring"] == {
        "abstention_is_wrong": True,
        "error_is_wrong": True,
        "polarity_flip_allowed": False,
        "support_gate_must_be_exactly_aligned": True,
    }


def test_checked_in_support_prototype_drill_result_is_canonical_and_bound() -> None:
    result = _canonical_record("support_prototype_drill_result_v1.json")
    content = dict(result)
    declared_digest = content.pop("digest")
    computed_digest = "sha256:" + hashlib.sha256(
        canonical_json(content) + b"\n"
    ).hexdigest()
    plan = _canonical_record("support_prototype_drill_plan_v1.json")

    assert declared_digest == computed_digest == (
        "sha256:38a89b3f78afa7c89f2f9dc881d209fce7b791ef3a346e54ee9ee3abaffa7fca"
    )
    assert result["schema"] == "gkm.bongard-support-prototype-drill-result.v1"
    assert result["campaign_id"] == plan["campaign_id"]
    assert result["campaign_plan_digest"] == plan["digest"]
    assert result["calibration_record_digest"] == plan["calibration_record_digest"]
    assert result["predicate_policy_digest"] == plan["predicate_policy_digest"]
    assert result["corpus_manifest_digest"] == plan["corpus_manifest_digest"]
    assert result["split_source_digest"] == plan["split_source_digest"]
    assert result["initial_exposure_ledger_digest"] == (
        plan["exposure_ledger_head_digest"]
    )
    assert result["episode_count"] == 12
    assert result["status_counts"] == {
        "complete": 0,
        "proposal_error": 1,
        "support_rejected": 11,
    }
    assert result["support_gate_passes"] == 0
    assert result["query_panels_released"] == 0
    assert result["executable_support_panels"] == 132
    assert result["support_forward_matches"] == 46
    assert result["support_reverse_matches"] == 10
    assert result["support_indeterminate"] == 76

    episodes = result["episodes"]
    assert isinstance(episodes, list)
    assert [episode["task_id"] for episode in episodes] == plan["task_ids"]
    repository = DATA.parents[1]
    for episode in episodes:
        run_payload = (repository / episode["run_file"]).read_bytes()
        assert hashlib.sha256(run_payload).hexdigest() == (
            episode["run_file_sha256"]
        )
        run = json.loads(run_payload)
        assert canonical_json(run) == run_payload
        assert run["record_digest"] == episode["record_digest"]
        assert run["episode"]["status"] == episode["status"]
        assert not episode["query_released"]

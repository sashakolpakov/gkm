from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bongard import (
    a3_closed_language_gate,
    load_historical_exposure,
    load_official_release,
    relational_library_ablation,
)
from bongard.artifacts import canonical_digest, canonical_json
from bongard.closed_visual_predicates import (
    closed_visual_predicate_evaluator_digest,
    closed_visual_predicate_source_digest,
    freeze_complete_closed_predicate_library,
)
from bongard.composite_visual_packet import (
    composite_visual_packet_source_digest,
    exact_panel_witness_extractor_digest,
)
from bongard.loop_scene_witnesses import loop_scene_extractor_digest
from bongard.prototype_calibration import PrototypeCalibrationRecord
from bongard.relational_visual_query import (
    Rational,
    RelationalVisualQuery,
    enumerate_factorized_shape_ratio_queries,
    relational_query_algorithm_digest,
)


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


def test_checked_in_support_prototype_drill_verification_is_bound() -> None:
    verification = _canonical_record(
        "support_prototype_drill_verification_v1.json"
    )
    content = dict(verification)
    declared_digest = content.pop("digest")
    computed_digest = "sha256:" + hashlib.sha256(
        canonical_json(content) + b"\n"
    ).hexdigest()
    result = _canonical_record("support_prototype_drill_result_v1.json")

    assert declared_digest == computed_digest == (
        "sha256:2fdfd965916450cf4165201464e68369dd523ed44806caa5994e7e5ddaa07729"
    )
    assert verification["schema"] == (
        "gkm.bongard-support-prototype-drill-verification.v1"
    )
    assert verification["campaign_result_digest"] == result["digest"]
    assert verification["all_records_verified"] is True
    assert verification["verified_record_count"] == result["episode_count"] == 12
    assert verification["status_counts"] == result["status_counts"]
    assert verification["verified_blob_preimages"] == 144
    assert verification["missing_blob_preimages"] == 0
    assert verification["neutral_extraction_replays"] == 276
    assert verification["query_panel_replays"] == 0
    assert [item["task_id"] for item in verification["records"]] == [
        item["task_id"] for item in result["episodes"]
    ]
    assert [item["run_file_sha256"] for item in verification["records"]] == [
        item["run_file_sha256"] for item in result["episodes"]
    ]


def test_checked_in_semantic_calibration_a3_outcome_is_canonical_and_bound() -> None:
    outcome = _canonical_record("semantic_calibration_stage_a3_outcome_v1.json")
    content = dict(outcome)
    declared_digest = content.pop("record_digest")
    computed_digest = "sha256:" + hashlib.sha256(
        canonical_json(content) + b"\n"
    ).hexdigest()

    assert declared_digest == computed_digest == (
        "sha256:3bca00d3bfac8d92292c73649cfcf1fe5adb48799eed972d07d43793f49a391f"
    )
    assert outcome["schema"] == (
        "gkm.bongard-semantic-calibration-stage-a3-outcome.v1"
    )
    release = load_official_release()
    assert outcome["identities"]["archive_sha256"] == release.archive_sha256
    assert (
        outcome["identities"]["corpus_manifest_digest"]
        == release.corpus_manifest_sha256
    )
    assert outcome["identities"]["split_source_digest"] == release.split_sha256
    assert outcome["funnel"] == {
        "candidate_count": 22,
        "direct_only_attrition": 6,
        "proposer_transport_failed": 0,
        "soft_claim_accepted": 15,
        "typed_parser_rejected": 1,
    }
    assert sum(
        outcome["funnel"][name]
        for name in (
            "direct_only_attrition",
            "proposer_transport_failed",
            "soft_claim_accepted",
            "typed_parser_rejected",
        )
    ) == outcome["funnel"]["candidate_count"]
    assert outcome["scoring"]["score_counts"] == {
        "0.0": 8,
        "0.5": 1,
        "1.0": 6,
    }
    assert [item["n"] for item in outcome["scoring"]["bins"]] == [9, 6]
    assert outcome["terminal"]["exact_reason"] == (
        "calibration score bins are underpopulated: 1"
    )
    assert outcome["terminal"]["reason_digest"] == (
        "42e086d19d4f6a3c4c75a9f6e01de3964bf5126995a984393f75081c2df343a7"
    )
    assert outcome["terminal"]["stage_b_authorized"] is False
    assert outcome["corpus"]["sealed_or_test_touched"] is False
    assert outcome["diagnostic_only"]["negation_won"] is False
    assert outcome["selection"]["scoreable_task_labels_opened"] == 15
    assert outcome["selection"]["unscoreable_task_labels_opened"] is False
    assert outcome["authority"]["python_predicate_authoritative"] is True
    assert outcome["authority"]["optional_checker_may_affect_result"] is False
    assert outcome["capacity_after_a3"]["drill"] == {
        "availability_by_family": {"bd": 0, "hd": 0},
        "certificate_digest": (
            "sha256:48fba29c8a33a5fd773baed373694ac32d91a6f456b17ede563113eeeecd18b1"
        ),
        "eligible_group_count": 0,
        "eligible_task_count": 0,
        "maximum_capacity": 0,
    }
    assert outcome["capacity_after_a3"]["dev"]["maximum_capacity"] == 16
    assert outcome["capacity_after_a3"]["dev"]["availability_by_family"] == {
        "bd": 16,
        "hd": 0,
    }
    assert outcome["forensics"]["panel_descriptions"] == {
        "audit_only": True,
        "distinct": 258,
        "total": 264,
    }
    assert outcome["forensics"]["complete_synthesized_formulas_evaluated"] == 0
    assert outcome["forensics"]["supported_cue_citations"][
        "semantic_part_style_axis_curvature_or_gestalt"
    ] == 0
    assert outcome["transport"][
        "a3_exact_native_client_bytes_authenticated"
    ] is False
    assert outcome["transport"]["posthoc_current_native_audit"] == {
        "causal_for_a3": False,
        "sha256": (
            "sha256:ae1d3ffe6d48aec6a4dc3f50e7eb8e0d11962485a6a9406c5a7012139383da02"
        ),
        "size_bytes": 271056976,
    }
    assert outcome["corpus"]["exact_unused_before_a3"]["total"] - 22 == (
        outcome["corpus"]["exact_unused_after_a3"]["total"]
    )


def test_attempt3_relational_forensics_binds_source_panels_and_support_roles() -> None:
    record = _canonical_record(
        "atomic_smoke_attempt3_relational_forensics_v1.json"
    )
    content = dict(record)
    declared_digest = content.pop("record_digest")

    assert declared_digest == canonical_digest(content) == (
        "0487edf805fda6de40ecfc42add1d8bf95e435e0f6912f6e2fd8d2a25e89eb2a"
    )
    assert record["schema"] == (
        "gkm.bongard-atomic-smoke-attempt3-relational-forensics.v1"
    )
    release = load_official_release()
    source = record["source_binding"]
    assert source["corpus_manifest_digest"] == release.corpus_manifest_sha256
    assert source["split_source_digest"] == release.split_sha256
    assert source["split"] == "train"
    assert source["task_id"] == "bd_mismatch_triangle_rec6_0000"
    assert source["label_to_archive_side"] == {"false": "0", "true": "1"}

    algorithms = record["algorithms"]
    assert algorithms["loop_scene_extractor_digest"] == (
        loop_scene_extractor_digest()
    )
    assert algorithms["relational_query_algorithm_digest"] == (
        relational_query_algorithm_digest()
    )
    query = RelationalVisualQuery.from_data(record["base_query"])
    assert query.digest() == record["base_query_digest"]

    panels = record["panels"]
    assert len(panels) == 14
    assert len({item["png_sha256"] for item in panels}) == 14
    support = [item for item in panels if item["role"] == "support"]
    heldout = [item for item in panels if item["role"] == "heldout"]
    assert len(support) == 12
    assert {(item["label"], item["source_index"]) for item in heldout} == {
        (True, 4),
        (False, 5),
    }
    assert {
        (item["label"], item["source_index"]) for item in support
    } == {
        *((True, index) for index in (0, 1, 2, 3, 5, 6)),
        *((False, index) for index in (0, 1, 2, 3, 4, 6)),
    }
    journal = record["journal_support_panel_sequence"]
    assert [item["panel_id"] for item in journal] == [
        f"support-panel-{index:02d}" for index in range(12)
    ]
    assert {item["png_sha256"] for item in journal} == {
        item["png_sha256"] for item in support
    }
    assert not ({item["png_sha256"] for item in journal} & {
        item["png_sha256"] for item in heldout
    })

    assert [item["base_disposition"] for item in support if item["label"]] == [
        "present"
    ] * 6
    assert [
        item["base_disposition"] for item in support if not item["label"]
    ] == ["certified_absent"] * 6
    assert [item["base_disposition"] for item in heldout] == [
        "indeterminate",
        "indeterminate",
    ]

    library = record["library"]
    assert library["member_count"] == len(
        enumerate_factorized_shape_ratio_queries()
    ) == 2520
    assert library["support_exact_separator_count"] == 4
    assert library["full_fourteen_exact_separator_count"] == 0
    assert library["polarity_flip_allowed"] is False
    expected_separator_digests = {
        RelationalVisualQuery.factorized_shape_ratio(
            numerator_side_count=3,
            denominator_side_count=4,
            ratio=Rational(1, denominator),
            denominator_obliqueness_millidegrees=obliqueness,
            require_point_contact=False,
        ).digest()
        for denominator in (12, 8)
        for obliqueness in (None, 5000)
    }
    assert {item["query_digest"] for item in library["separators"]} == (
        expected_separator_digests
    )
    assert all(
        item["heldout_positive_disposition"] == "indeterminate"
        and item["require_point_contact"] is False
        for item in library["separators"]
    )
    assert record["claim_boundary"] == {
        "benchmark_claim_authorized": False,
        "new_pixels_opened_for_this_record": False,
        "official_test_authorized": False,
        "purpose": (
            "post-hoc relational forensics over the already exposed "
            "attempt-three train task"
        ),
        "replaces_historical_soft_atom_record": False,
    }


def test_rejected_relational_dev_plan_is_compact_and_not_runnable() -> None:
    record = _canonical_record(
        "relational_headless_full_current_dev_20260807.rejection.json"
    )
    content = dict(record)
    declared_digest = content.pop("digest")
    computed_digest = "sha256:" + hashlib.sha256(
        canonical_json(content) + b"\n"
    ).hexdigest()

    assert declared_digest == computed_digest == (
        "sha256:df695aa7e5ced4e4dd9ba2df1e694754c2b8aa9e5b9397a62ee5045a28177569"
    )
    assert record["schema"] == (
        "gkm.bongard-relational-headless-plan-rejection.v1"
    )
    assert record["rejection"]["disposition"] == (
        "permanently_rejected_do_not_execute"
    )
    assert record["rejection"]["schedule_commitment_audit"] == {
        "commitment_hiding": False,
        "construction": "unkeyed digest of selected support indices",
        "preimage_count_per_task": 49,
        "recovered_task_schedules": 15,
        "task_count": 15,
    }
    assert record["rejection"]["language_audit"][
        "full_intended_concept_expressible_count"
    ] == 0
    assert record["rejection"]["language_audit"]["task_count"] == 15
    assert record["execution"] == {
        "active_ledger_digest": (
            "sha256:651a43af02c29aedf2276aaf20c60f621954d95b2bbfa1ec827101ab2500fb57"
        ),
        "dev_pixels_opened": False,
        "exposure_successor_written": False,
        "plan_executed": False,
        "proposer_calls": 0,
        "query_labels_materialized": False,
    }
    assert len(record["invalid_plan"]["task_ids"]) == 15
    assert not (DATA / "relational_headless_full_current_dev_20260807.plan.json").exists()


def test_relational_library_ablation_outcome_is_compact_strict_and_bound() -> None:
    name = "relational_library_ablation_24task_outcome_v1.json"
    record = _canonical_record(name)
    assert (DATA / name).stat().st_size < 12_000
    assert set(record) == {
        "algorithm_identities",
        "qualification",
        "query_library",
        "record_digest",
        "restrictions",
        "results",
        "schema",
        "source",
    }
    content = dict(record)
    declared_digest = content.pop("record_digest")
    assert declared_digest == "sha256:" + canonical_digest(content) == (
        "sha256:ea6ee897513c22f1db8e656570e6572f2955855bbadb5caa39d8dc5dc8d423cd"
    )
    assert record["schema"] == (
        "gkm.bongard-relational-library-ablation-outcome.v1"
    )

    source = record["source"]
    assert set(source) == {
        "ablation_input_digest",
        "ablation_output_digest",
        "coverage_output_digest",
        "coverage_selection_digest",
        "exposure_successor_digest",
        "full_report_file_sha256",
        "full_report_relative_path",
        "full_report_size_bytes",
        "selected_png_manifest_digest",
        "source_corpus_manifest_digest",
    }
    assert source == {
        "ablation_input_digest": (
            "sha256:2b9010c0706fbaa8217c7eb3fa551fed41ac41efe9cae0ab7a112fab3c8608d4"
        ),
        "ablation_output_digest": (
            "sha256:0a4b601ffc794a640175d2afda4f4b0d7f57fc980700bafbf09848ea4768c59b"
        ),
        "coverage_output_digest": (
            "sha256:f78626c51b0af34cb0ccd96ed56041a51bcaeb453d3f26b10ea1ed1377542ae0"
        ),
        "coverage_selection_digest": (
            "sha256:ccd8cb65a0d3524da354b7a1e448638f7a0f2a14c5e68eb469b68e09e9feae67"
        ),
        "exposure_successor_digest": (
            "sha256:651a43af02c29aedf2276aaf20c60f621954d95b2bbfa1ec827101ab2500fb57"
        ),
        "full_report_file_sha256": (
            "34529736da21666d775853868586ef3f326dbc7a5cfd4f7ce5145ddd8a231f9a"
        ),
        "full_report_relative_path": (
            "downloads/ShapeBongard_V2_full/relational_library_ablation_v1/"
            "0a4b601ffc794a640175d2afda4f4b0d7f57fc980700bafbf09848ea4768c59b."
            "ablation.json"
        ),
        "full_report_size_bytes": 1_716_919,
        "selected_png_manifest_digest": (
            "sha256:acdd8069fc3a5cf2341eed4d39dd37a40fd1381921882c4caac72e1124b761c9"
        ),
        "source_corpus_manifest_digest": (
            "sha256:6fa51548520190a412812ba8f872dc3c7a7a2b2c47c0e42a4d9f6df351dce138"
        ),
    }
    assert source["source_corpus_manifest_digest"] == (
        load_official_release().corpus_manifest_sha256
    )

    algorithms = record["algorithm_identities"]
    assert algorithms["ablation_algorithm_id"] == (
        "bongard.relational-library-ablation/complete-v3-library-v1"
    )
    assert algorithms["ablation_python_source_digest"] == hashlib.sha256(
        Path(relational_library_ablation.__file__).read_bytes()
    ).hexdigest()
    assert algorithms["loop_scene_extractor_digest"] == (
        loop_scene_extractor_digest()
    )
    assert algorithms["relational_query_algorithm_digest"] == (
        relational_query_algorithm_digest()
    )
    assert algorithms["relational_query_algorithm_id"] == (
        "bongard.relational-visual-query/python-v3"
    )
    assert algorithms["canonical_equivalence_sample_indices"] == [0, 1, 8, 2519]
    assert algorithms["reference_execution"] == "python-canonical/v1"

    queries = enumerate_factorized_shape_ratio_queries()
    query_library = record["query_library"]
    assert len(queries) == query_library["count"] == 2_520
    assert query_library == {
        "canonical_equivalence_check_count": 1_344,
        "canonical_equivalence_checks_per_unique_packet": 4,
        "count": 2_520,
        "inventory_digest": (
            "sha256:f4675201aec95031214f7b93ad9947c56352ab6832a0dcca7acaed4f43ff2697"
        ),
        "query_algorithm_digest": relational_query_algorithm_digest(),
        "unique_reextracted_packet_count": 336,
    }
    assert query_library["inventory_digest"] == "sha256:" + hashlib.sha256(
        canonical_json([query.digest() for query in queries])
    ).hexdigest()

    assert record["restrictions"] == {
        "action_program_json_authorized": False,
        "allowed_splits": ["train", "val"],
        "candidate_dependent_extraction_authorized": False,
        "negation_rescue_authorized": False,
        "new_exposure_event_created": False,
        "official_benchmark_or_generalization_claim_authorized": False,
        "official_test_pixels_authorized": False,
        "polarity_flip_authorized": False,
        "proposer_or_model_authorized": False,
        "selected_manifest_only_png_replay": True,
    }
    assert record["qualification"]["evaluation_kind"] == (
        "resubstitution/library-coverage"
    )
    assert record["qualification"]["benchmark_or_generalization_result"] is False
    assert record["qualification"]["candidate_pixel_access"] is False
    assert record["qualification"]["downstream_exposure_delta"] == 0

    results = record["results"]
    assert set(results) == {
        "by_family",
        "by_split",
        "finding",
        "global",
        "maximum_best_forward_correct_panels",
        "task_query_evaluations",
        "tasks_at_maximum_best_forward_correct_panels",
    }
    global_counts = results["global"]
    assert global_counts == {
        "all_positive_present_queries": 6,
        "best_error_panel_histogram": {"0": 24},
        "best_forward_correct_panel_histogram": {
            "2": 2,
            "4": 4,
            "5": 4,
            "6": 5,
            "7": 8,
            "8": 1,
        },
        "best_indeterminate_panel_histogram": {
            "0": 5,
            "1": 2,
            "10": 1,
            "11": 1,
            "2": 5,
            "3": 3,
            "4": 1,
            "5": 2,
            "6": 2,
            "8": 2,
        },
        "extractor_failure_panels": 0,
        "fit_separator_occurrences_across_folds": 0,
        "folds_with_any_fit_separator": 0,
        "folds_with_any_heldout_forward_correct_separator": 0,
        "full_7_plus_7_exact_forward_separators": 0,
        "heldout_forward_correct_separator_occurrences_across_folds": 0,
        "paired_leave_one_out_folds": 168,
        "tasks": 24,
        "tasks_with_any_all_positive_present_query": 1,
        "tasks_with_any_full_7_plus_7_exact_forward_separator": 0,
    }
    assert set(results["by_split"]) == {"train", "val"}
    assert set(results["by_family"]) == {"bd", "ff", "hd"}
    assert sum(item["tasks"] for item in results["by_split"].values()) == 24
    assert sum(item["tasks"] for item in results["by_family"].values()) == 24
    assert all(
        item["full_7_plus_7_exact_forward_separators"] == 0
        and item["fit_separator_occurrences_across_folds"] == 0
        and item["folds_with_any_heldout_forward_correct_separator"] == 0
        for grouping in (results["by_split"], results["by_family"])
        for item in grouping.values()
    )
    assert results["task_query_evaluations"] == 24 * 2_520 == 60_480
    assert results["maximum_best_forward_correct_panels"] == 8
    assert results["tasks_at_maximum_best_forward_correct_panels"] == 1
    assert "0/24" in results["finding"]
    assert "0/168" in results["finding"]


def test_a3_closed_language_gate_result_is_strict_current_and_support_only() -> None:
    name = "a3_closed_language_gate_result_v2.json"
    path = DATA / name
    payload = path.read_bytes()
    record = _canonical_record(name)

    assert not (DATA / "atomic_smoke_attempt3_no_exact_separator_v1.json").exists()
    assert not (DATA / "a3_closed_language_gate_result_v1.json").exists()
    assert len(payload) == 7_588
    assert hashlib.sha256(payload).hexdigest() == (
        "adffdaa4eb4208125d273c47732cb606b570ce443c5ef2bacd641e57aa4b52a2"
    )
    assert set(record) == {
        "algorithm_id",
        "algorithm_identities",
        "claim_boundary",
        "frozen_library",
        "oracle",
        "record_digest",
        "schema",
        "source",
        "support",
    }
    content = dict(record)
    declared_digest = content.pop("record_digest")
    assert declared_digest == "sha256:" + canonical_digest(content) == (
        "sha256:f9b6373df4dbe5d63807cf7e21be931db7ec0e9dfba106917df73d0e170a52d6"
    )
    assert record["schema"] == (
        "gkm.bongard-a3-closed-language-gate-result.v2"
    )
    assert record["algorithm_id"] == (
        "bongard.a3-closed-language-gate/support-only-v2"
    )

    algorithms = record["algorithm_identities"]
    assert algorithms == {
        "closed_predicate_evaluator_digest": (
            closed_visual_predicate_evaluator_digest()
        ),
        "closed_predicate_source_digest": (
            closed_visual_predicate_source_digest()
        ),
        "composite_packet_source_digest": (
            composite_visual_packet_source_digest()
        ),
        "exact_composite_extractor_digest": (
            exact_panel_witness_extractor_digest()
        ),
        "gate_python_source_digest": hashlib.sha256(
            Path(a3_closed_language_gate.__file__).read_bytes()
        ).hexdigest(),
        "lean_required": False,
        "oracle_algorithm_id": (
            "bongard.support-only-expressibility-oracle/v1"
        ),
        "python_is_canonical": True,
    }

    frozen_record = record["frozen_library"]
    assert frozen_record == {
        "construction_id": "complete-proposer-reachable-closed-union/v2",
        "freeze_preceded_any_png_read": True,
        "library_digest": (
            "4d1db17ce37a46fb7c220ce57557b44e4883175617347a01bb4ef1bb5871df35"
        ),
        "member_count": 65_678,
        "member_counts_by_tagged_kind": {
            "direct_counts": 64_400,
            "relational": 1_260,
            "symmetry": 18,
        },
    }
    current_library = freeze_complete_closed_predicate_library()
    assert current_library.construction_id == frozen_record["construction_id"]
    assert len(current_library.members) == frozen_record["member_count"]
    assert current_library.digest == frozen_record["library_digest"]

    source = record["source"]
    assert source == {
        "corpus_manifest_digest": (
            "sha256:6fa51548520190a412812ba8f872dc3c7a7a2b2c47c0e42a4d9f6df351dce138"
        ),
        "forbidden_heldout_authority_digest": (
            "bc0b281d9060c5a303868ce66f347d5118b360cca7c7184d33026aaeb6f2baa7"
        ),
        "forensics_file_sha256": (
            "a674869ce98575733b86c9a6ba9e2c32a6f5784a7134d59d8b1cd3db651ab46c"
        ),
        "forensics_record_digest": (
            "0487edf805fda6de40ecfc42add1d8bf95e435e0f6912f6e2fd8d2a25e89eb2a"
        ),
        "forensics_schema": (
            "gkm.bongard-atomic-smoke-attempt3-relational-forensics.v1"
        ),
        "split": "train",
        "split_source_digest": (
            "sha256:ebb9cd474478e0776dff539951070db2c96b9b312c4b0b073689d20792ed7230"
        ),
        "support_mapping_digest": (
            "d91190a336e7eb0b3725ba51b309dcedd6cb5f9daee2d523788fd8b9cae81834"
        ),
        "task_id": "bd_mismatch_triangle_rec6_0000",
    }
    assert source["corpus_manifest_digest"] == (
        load_official_release().corpus_manifest_sha256
    )

    support = record["support"]
    assert set(support) == {
        "count",
        "exact_receipts_digest",
        "negative_count",
        "panels",
        "positive_count",
    }
    assert support["count"] == 12
    assert support["positive_count"] == support["negative_count"] == 6
    panels = support["panels"]
    assert len(panels) == 12
    assert len({item["relative_path"] for item in panels}) == 12
    assert len({item["png_sha256"] for item in panels}) == 12
    assert len({item["exact_composite_packet_digest"] for item in panels}) == 12
    assert [(item["label"], item["source_index"]) for item in panels] == [
        *((True, index) for index in (0, 1, 2, 3, 5, 6)),
        *((False, index) for index in (0, 1, 2, 3, 4, 6)),
    ]
    assert all("/1/" in item["relative_path"] for item in panels[:6])
    assert all("/0/" in item["relative_path"] for item in panels[6:])
    assert support["exact_receipts_digest"] == canonical_digest(
        {
            "schema": "gkm.bongard-a3-exact-support-receipts.v1",
            "support_mapping_digest": source["support_mapping_digest"],
            "receipts": panels,
        }
    )

    assert record["oracle"] == {
        "diagnosis": "language_separator_exists_no_model_proposal",
        "evaluation_matrix_digest": (
            "eff0dad0476110deabc93f08b778e675763843ae9228e51b59f9a85cc773cc7f"
        ),
        "exact_forward_separator_count": 4,
        "model_is_exact_separator": None,
        "model_predicate_digest": None,
        "result_digest": (
            "38b5f796b3e470c26b8ac2bedd062b7409ee3e3796d8bcaed98c3e19fccd7f4a"
        ),
        "separator_counts_by_tagged_kind": {
            "direct_counts": 0,
            "relational": 4,
            "symmetry": 0,
        },
        "separator_inventory_digest": (
            "6ff59a246404d50b0d25970a332126c7632c940c98be3ff87579512664f02fdb"
        ),
    }
    assert record["claim_boundary"] == {
        "action_program_json_authorized": False,
        "benchmark_or_generalization_claim_authorized": False,
        "canonical_attempt3_support_mapping_only": True,
        "evaluation_kind": (
            "already-exposed-support-only-closed-language-coverage"
        ),
        "heldout_pixels_read": False,
        "model_or_proposer_called": False,
        "negation_rescue_authorized": False,
        "new_exposure_event_created": False,
        "new_pixels_opened": False,
        "official_test_pixels_read": False,
        "polarity_flip_authorized": False,
        "query_pixels_read": False,
    }

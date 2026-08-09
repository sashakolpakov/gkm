"""Validation for the two final exposed support-only diagnostics of 2026-08-09."""

from __future__ import annotations

import json
from pathlib import Path

from bongard.artifacts import canonical_digest, canonical_json


SUMMARY = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "panel_positive_live_support_diagnostic_summary_20260809_v1.json"
)

ATOMS = [
    "The latent carrier forms a closed contour.",
    "The enclosed region is convex.",
    "The latent carrier is a simple Jordan curve.",
    "The drawing has one connected carrier.",
    "Visible marks follow a common perimeter.",
    "The carrier bounds a single compact interior region.",
    "The perimeter turns with a consistent rotational sense.",
    "The silhouette is elongated along a dominant axis.",
]


def _summary() -> dict[str, object]:
    payload = SUMMARY.read_bytes()
    record = json.loads(payload)
    assert isinstance(record, dict)
    assert payload == canonical_json(record) + b"\n"
    return record


def test_summary_is_canonical_and_self_addressed() -> None:
    record = _summary()
    assert set(record) == {
        "authority",
        "conclusions",
        "custody",
        "record_digest",
        "record_digest_policy",
        "runs",
        "schema",
        "summary_construction",
    }
    assert record["schema"] == (
        "gkm.bongard-panel-positive-live-support-diagnostic-summary.v1"
    )
    content = dict(record)
    declared_digest = content.pop("record_digest")
    assert record["record_digest_policy"] == (
        "sha256(canonical JSON of this object with record_digest omitted)"
    )
    assert declared_digest == "sha256:" + canonical_digest(content) == (
        "sha256:0fc11ce76fc5cba16fa14aeddf0d3a5b779120d38877fbfec14076aec3666f97"
    )


def test_contextual_count_completion_is_exactly_bound() -> None:
    run = _summary()["runs"]["contextual_typed_count"]
    assert run["completion_artifact"] == {
        "byte_count": 12_968,
        "raw_sha256": (
            "5401e05ed3fe7ad4e4b09c52f7f213c4b2411bea64bbbbec8823f730115a13d6"
        ),
        "record_digest": (
            "sha256:fadb6206e64eba90c40a40d393058117ca935794390998a392f1d8005ab40935"
        ),
        "relative_path": (
            "downloads/ShapeBongard_V2_full/"
            "panel_positive_contextual_typed_count_probe_20260809_v1/"
            "completion.json"
        ),
        "schema": "gkm.bongard-contextual-typed-count-support-completion.v1",
    }
    assert run["digests"] == {
        "authorization": (
            "sha256:bd067e9088d185b8df6a0c97a10956cf3d8f952b0cbbc7fcc2b21b2e7a71852e"
        ),
        "command_source": (
            "07da149634fb336747154725e4731b3b9953081bcaa7e1b6d117eec8f514dd94"
        ),
        "context_policy_record": (
            "sha256:f4e8f4e3d44a91cf96ce64ccdaf6bbbb951e48a02eb50e19a9a5a47f9075591f"
        ),
        "execution_precommit": (
            "sha256:fb3f9356587bf31b907f302a57d09236fbf210a40d0e8fce4b9c3dd2a7fe6714"
        ),
        "ink_zoom_policy_record": (
            "sha256:25a602455aab02b0d5cbcb05f18bd283e9b7ce43e88c343933ab2a4b2798d564"
        ),
        "output_schema": (
            "be2e5e31bbc134e64495fa460b401ef9ae2958f961eb17be7701d6f96cab2403"
        ),
        "prompt_sha256": (
            "1bf2c748ef0e0f0f8796c91e95c0cc17a64b10e2514b16ba3358931e0450796c"
        ),
        "receipt": (
            "4fac9720cfd78dcddb18ddc8678cd29d7762f04a995a17e49e122b323c8e4fa9"
        ),
        "runtime_evidence": (
            "sha256:09ce79d7cceecb316ab422b1eb059df959078f70a11f65f2d62e9b07956842e0"
        ),
        "typed_count_policy_record": (
            "sha256:3ac70952b2fa0c94a4b4afe87e0dce7448f86cef21bf70f6952cd33e68164bae"
        ),
    }
    assert run["calls"] == {
        "physical_model_calls": 1,
        "query_observer_calls": 0,
        "query_release_calls": 0,
        "support_context_batch_observer_calls": 1,
    }
    assert run["disposition_counts"] == {
        "contrast": {
            "certified_absent": 6,
            "error": 0,
            "indeterminate": 0,
            "present": 0,
        },
        "primary": {
            "certified_absent": 4,
            "error": 0,
            "indeterminate": 0,
            "present": 2,
        },
    }
    assert run["component_disposition_counts"] == {
        "contrast_convexity": {"certified_absent": 3, "present": 3},
        "contrast_straight_count_four": {"certified_absent": 4, "present": 2},
        "primary_convexity": {"certified_absent": 0, "present": 6},
        "primary_straight_count_four": {"certified_absent": 4, "present": 2},
    }
    assert run["status"] == "support_gap"
    assert run["support_consistent"] is False
    assert run["gap_reasons"] == [
        "primary_present_below_five",
        "primary_certified_absent_contradiction",
    ]


def test_atom_slate_completion_binds_every_atom_and_collapse_count() -> None:
    run = _summary()["runs"]["positive_atom_slate"]
    assert run["completion_artifact"] == {
        "byte_count": 49_187,
        "raw_sha256": (
            "772ba0909a149a64a57c99a07e5dae37f07f08d0dacef81bd8f872e5ffc90bf4"
        ),
        "record_digest": (
            "sha256:df17e204782a7251891452bf9e5ffb87b3085ed5b15ad4f32da45f28e5f63b6d"
        ),
        "relative_path": (
            "downloads/ShapeBongard_V2_full/"
            "panel_positive_atom_slate_exposed_probe_20260809_v1/"
            "completion.json"
        ),
        "schema": "gkm.bongard-positive-atom-slate-exposed-completion.v1",
    }
    assert run["digests"] == {
        "atom_core_source": (
            "a245e73826ef04640e439843bc9f606264085924f06f2303a1c60a5fac137d55"
        ),
        "authorization": (
            "sha256:053b9ff346eba8531c14a809e26665342d546024d09a1efd430944e2c96b87c2"
        ),
        "command_source": (
            "42ad8195809ebb226a18d9133c9dcb80a585594086652064e16eda287813f9c9"
        ),
        "execution_precommit": (
            "sha256:cae66ba51ad2817a1e3124a8b1be0e7bb93a395cd870c982427d140f60de3576"
        ),
        "proposer_artifact": (
            "d5cfeb6179a5f199d8d704fe2c73b06ba60145a12ee14a26b50e3d26c4196c78"
        ),
        "runtime_evidence": (
            "sha256:7195854f075d0efe62401817e0910ab6d4a02b42b277f3d11f77b2bd45a1c14e"
        ),
        "support_inventory": (
            "cad1e4b325f44434bf325caf162ec3e6255deda41b0a9a4c5d9947ceb8448550"
        ),
    }
    assert run["calls"] == {
        "physical_model_calls": 13,
        "proposer_model_calls": 1,
        "query_observer_calls": 0,
        "query_release_calls": 0,
        "support_observer_model_calls": 12,
    }
    slate = run["slate"]
    assert slate["atom_count"] == 8
    assert slate["atom_ids"] == [f"atom_{index:02d}" for index in range(8)]
    assert slate["atoms"] == ATOMS
    assert slate["slate_digest"] == (
        "285217eb60f5d4ad3153ac5b15a0eda30eeafe3e70d05bad48f7af89a1dd4f7e"
    )
    assert run["formula_search"] == {
        "admitted_formula_count": 0,
        "distinct_disposition_signature_count": 3,
        "enumerated_formula_count": 36,
        "formula_counts_by_signature_descending": [15, 13, 8],
        "formula_order": "eight_singletons_then_twenty_eight_lexicographic_pairs",
        "maximum_correct_support_rows": 7,
        "negative_formula_present": False,
        "support_formula_admitted": False,
        "support_row_count": 12,
    }
    assert sum(run["formula_search"]["formula_counts_by_signature_descending"]) == 36
    assert run["status"] == "support_gap"
    assert run["gap"] == {
        "code": "no_admissible_affirmative_singleton_or_pair",
        "error_row_ordinals": [],
        "formula_count": 36,
        "query_release_allowed": False,
        "row_count": 12,
        "schema": "gkm.bongard-positive-atom-gap.v1",
    }


def test_summary_preserves_custody_authority_and_explicit_diagnoses() -> None:
    record = _summary()
    assert record["authority"] == {
        "benchmark_or_generalization_claim_authorized": False,
        "engineering_only": True,
        "lean_present": False,
        "lean_removable": True,
        "lean_required": False,
        "python_is_canonical_authority": True,
        "scientific_benchmark": False,
    }
    assert record["custody"] == {
        "diagnostic_split": "train",
        "diagnostic_support_panel_count": 12,
        "diagnostic_task_id": "hd_convex-has_four_straight_lines_0001",
        "official_test_panel_pixels_accessed": False,
        "query_observer_calls": 0,
        "query_release_calls": 0,
        "reserved_target_panel_pixels_accessed": False,
        "reserved_target_task_id": "hd_convex-has_four_straight_lines_0000",
        "sealed_validation_task_ids": [
            "hd_convex-has_four_straight_lines_0018",
            "hd_convex-has_four_straight_lines_0019",
        ],
        "validation_panel_pixels_accessed": False,
    }
    assert record["summary_construction"] == {
        "completion_json_files_read": 2,
        "model_calls_made": 0,
        "panel_pixels_read": False,
        "query_calls_made": 0,
        "query_releases_made": 0,
    }
    runs = record["runs"].values()
    assert sum(run["calls"]["query_observer_calls"] for run in runs) == 0
    assert sum(run["calls"]["query_release_calls"] for run in runs) == 0
    assert record["conclusions"] == {
        "action_segmentation_failure": {
            "asserted": True,
            "evidence": (
                "all six primary rows were convex-present, but straight-count-four "
                "was present for only two and certified absent for four"
            ),
        },
        "proposer_semantic_redundancy": {
            "asserted": True,
            "evidence": (
                "thirty-six singleton-or-pair formulas over eight exact proposed "
                "atoms collapsed to three support signatures with multiplicities "
                "15, 13, and 8; none was admitted"
            ),
        },
        "release_disposition": "both_support_gaps_queries_remain_sealed",
    }

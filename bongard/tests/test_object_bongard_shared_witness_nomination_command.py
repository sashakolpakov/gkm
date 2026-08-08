"""Offline launch/replay test for the structured historical nomination."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import bongard.object_bongard_semantics as old_semantics
from bongard.object_bongard_shared_witness_nomination_command import (
    ARTIFACT_FILENAME,
    AUTHORIZATION_FILENAME,
    PRECOMMIT_FILENAME,
    ObjectBongardSharedWitnessNominationCommandError,
    run_object_bongard_shared_witness_nomination,
    verify_object_bongard_shared_witness_nomination,
)
from bongard.object_bongard_shared_witness_semantics import (
    object_bongard_shared_witness_semantics_output_schema,
    object_bongard_shared_witness_semantics_prompt,
)
from bongard.tests.no_tools_fixture import canonical_no_tools_runtime
from bongard.tests.test_object_bongard_panel_rubric_calibration import SOURCE_ROOT
from bongard.tests.test_prototype_scene_observer import _receipt
from bongard.transport import (
    PINNED_CODEX_CLI_VERSION,
    CloudPolicyCacheSnapshot,
    CodexStructuredResult,
)


LAUNCHER_DIGEST = "b" * 64


def _payload() -> dict[str, object]:
    return {
        "proposal_0": {
            "shared_anchor": "decorated figure",
            "visual_axis": "closed loop topology",
            "group_0_endpoint": "two loops touching at one vertex",
            "group_1_endpoint": "single loop with dangling branch",
        },
        "proposal_1": {
            "shared_anchor": "central outlined figure",
            "visual_axis": "junction angle profile",
            "group_0_endpoint": "four oblique rays meeting centrally",
            "group_1_endpoint": "three acute rays meeting centrally",
        },
    }


def test_one_fresh_structured_call_then_model_free_cold_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def old_path_forbidden(*_args, **_kwargs):
        raise AssertionError("historical free-cue semantic proposer was called")

    monkeypatch.setattr(
        old_semantics, "describe_object_bongard_support", old_path_forbidden
    )
    cache = CloudPolicyCacheSnapshot(None)
    catalog, attestation = canonical_no_tools_runtime(LAUNCHER_DIGEST)
    preflight: list[str] = []

    def cache_snapshotter():
        preflight.append("cache")
        return cache

    def catalog_snapshotter():
        preflight.append("catalog")
        return catalog

    def fingerprinter(executable, *, expected_launcher_digest):
        assert executable == "codex"
        assert expected_launcher_digest == LAUNCHER_DIGEST
        preflight.append("fingerprint")
        return {
            "version": PINNED_CODEX_CLI_VERSION,
            "launcher_digest": LAUNCHER_DIGEST,
        }

    def attester(**kwargs):
        assert kwargs["cloud_policy_cache_snapshot"] is cache
        assert kwargs["model_catalog_snapshot"] is catalog
        preflight.append("attestation")
        return attestation

    root = tmp_path / "structured_nomination"
    calls = 0

    def transport(prompt, paths, names, schema, **_kwargs):
        nonlocal calls
        calls += 1
        assert (root / AUTHORIZATION_FILENAME).is_file()
        assert (root / PRECOMMIT_FILENAME).is_file()
        assert not (root / ARTIFACT_FILENAME).exists()
        assert prompt == object_bongard_shared_witness_semantics_prompt()
        assert schema == object_bongard_shared_witness_semantics_output_schema()
        assert len(paths) == len(names) == 12
        assert names[0] == "group_0_ref_00.png"
        assert names[-1] == "group_1_ref_05.png"
        payload = _payload()
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    launched = run_object_bongard_shared_witness_nomination(
        root,
        source_root=SOURCE_ROOT,
        expected_launcher_sha256=LAUNCHER_DIGEST,
        cache_snapshotter=cache_snapshotter,
        catalog_snapshotter=catalog_snapshotter,
        launcher_fingerprinter=fingerprinter,
        runtime_attester=attester,
        visual_transport=transport,
    )
    assert preflight == ["cache", "catalog", "fingerprint", "attestation"]
    assert calls == 1
    assert launched.accepted is True
    assert len(launched.artifact.contrast_candidates) == 2
    assert launched.artifact.model_payload == _payload()
    assert verify_object_bongard_shared_witness_nomination(
        root, source_root=SOURCE_ROOT
    ) == launched
    assert calls == 1

    authorization = json.loads((root / AUTHORIZATION_FILENAME).read_text())
    assert authorization["historical_exposed_panel_count"] == 12
    assert authorization["physical_model_call_count"] == 1
    assert authorization["support_roles_visible_to_model"] is False
    assert authorization["query_pixels_used"] is False
    assert authorization["fresh_broad_cohort_pixels_used"] is False
    assert authorization["official_test_pixels_used"] is False
    assert authorization["calibration_authorized_by_this_command"] is False
    assert len(authorization["neutral_groups"]) == 2
    assert [len(item["panel_ids"]) for item in authorization["neutral_groups"]] == [
        6,
        6,
    ]
    assert (
        authorization["semantic_protocol_digest"]
        == launched.artifact.protocol_digest
    )
    assert '"lean_present":false' in (root / "result.json").read_text("utf-8")

    with pytest.raises(
        ObjectBongardSharedWitnessNominationCommandError, match="fresh"
    ):
        run_object_bongard_shared_witness_nomination(
            root,
            source_root=SOURCE_ROOT,
            expected_launcher_sha256=LAUNCHER_DIGEST,
            cache_snapshotter=cache_snapshotter,
            catalog_snapshotter=catalog_snapshotter,
            launcher_fingerprinter=fingerprinter,
            runtime_attester=attester,
            visual_transport=transport,
        )
    assert calls == 1

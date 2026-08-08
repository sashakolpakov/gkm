from __future__ import annotations

import pytest
from types import SimpleNamespace

import bongard.object_bongard_rubric_campaign_command as command
from bongard.object_bongard_rubric_campaign_command import (
    ObjectBongardRubricCampaignAuthorization,
    ObjectBongardRubricCampaignCommandError,
    ObjectBongardRubricCampaignRuntimePrecommit,
    run_object_bongard_rubric_campaign_command,
)
from bongard.tests.no_tools_fixture import canonical_no_tools_runtime
from bongard.transport import CloudPolicyCacheSnapshot, PINNED_CODEX_CLI_VERSION


LAUNCHER_DIGEST = "7" * 64


def _authorization() -> ObjectBongardRubricCampaignAuthorization:
    values = {
        "preregistration_digest": "sha256:" + "1" * 64,
        "batch_plan_digest": "sha256:" + "2" * 64,
        "release_descriptor_digest": "sha256:" + "3" * 64,
        "exposure_predecessor_digest": "sha256:" + "4" * 64,
        "calibration_assessment_digest": "5" * 64,
        "calibration_replay_digest": "sha256:" + "6" * 64,
        "calibration_observation_inventory_digest": "sha256:" + "8" * 64,
        "campaign_source_bindings": tuple(
            sorted(command.object_bongard_rubric_campaign_source_bindings().items())
        ),
        "command_source_digest": (
            command.object_bongard_rubric_campaign_command_source_digest()
        ),
        "minutes": 15,
        "parallel_workers": 4,
        "max_physical_model_calls": 5_400,
        "executable": "codex",
        "expected_launcher_sha256": LAUNCHER_DIGEST,
        "exposure_observed_at": "2026-08-08T12:00:00Z",
    }
    provisional = object.__new__(ObjectBongardRubricCampaignAuthorization)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    return ObjectBongardRubricCampaignAuthorization(
        **values,
        authorization_digest=command._address(
            command._authorization_content(provisional)
        ),
    )


def test_runtime_precommit_round_trip_binds_both_modalities() -> None:
    authorization = _authorization()
    catalog, attestation = canonical_no_tools_runtime(LAUNCHER_DIGEST)
    precommit = ObjectBongardRubricCampaignRuntimePrecommit.seal(
        authorization,
        cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
        model_catalog_snapshot=catalog,
        no_tools_attestation=attestation,
        launcher_fingerprint={
            "version": PINNED_CODEX_CLI_VERSION,
            "launcher_digest": LAUNCHER_DIGEST,
        },
    )

    loaded = ObjectBongardRubricCampaignRuntimePrecommit.from_data(
        precommit.to_data()
    )

    assert loaded == precommit
    assert loaded.runtime.visual.model == command.MODEL
    assert loaded.runtime.rank.model == command.MODEL
    assert loaded.runtime.visual.reasoning_effort == command.REASONING_EFFORT
    assert loaded.runtime.rank.reasoning_effort == command.REASONING_EFFORT
    command._assert_runtime_precommit_matches_authorization(
        loaded, authorization
    )


def test_launch_fails_at_calibration_gate_before_runtime_or_archive(
    tmp_path,
) -> None:
    runtime_snapshot_called = False

    def forbidden_snapshot():
        nonlocal runtime_snapshot_called
        runtime_snapshot_called = True
        raise AssertionError("runtime snapshot crossed calibration gate")

    with pytest.raises(
        ObjectBongardRubricCampaignCommandError,
        match="closed until calibration cold replay is accepted",
    ):
        run_object_bongard_rubric_campaign_command(
            tmp_path / "campaign",
            calibration_root=tmp_path / "failed-calibration",
            calibration_verifier=lambda _root: object(),
            cloud_policy_cache_snapshotter=forbidden_snapshot,
        )

    assert runtime_snapshot_called is False


def test_authorization_requires_addressed_calibration_replay() -> None:
    raw = _authorization().to_data()
    raw["calibration_replay_digest"] = "6" * 64
    raw["authorization_digest"] = "sha256:" + "0" * 64

    with pytest.raises(
        ObjectBongardRubricCampaignCommandError,
        match="calibration_replay_digest must be a sha256: address",
    ):
        ObjectBongardRubricCampaignAuthorization.from_data(raw)


def test_launch_gate_rejects_any_configuration_tamper() -> None:
    authorization = _authorization()
    catalog, attestation = canonical_no_tools_runtime(LAUNCHER_DIGEST)
    precommit = ObjectBongardRubricCampaignRuntimePrecommit.seal(
        authorization,
        cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
        model_catalog_snapshot=catalog,
        no_tools_attestation=attestation,
        launcher_fingerprint={
            "version": PINNED_CODEX_CLI_VERSION,
            "launcher_digest": LAUNCHER_DIGEST,
        },
    )
    configuration = {
        "campaign_id": command.CAMPAIGN_ID,
        "preregistration_digest": authorization.preregistration_digest,
        "runtime_binding_digest": precommit.runtime.binding_digest,
        "max_workers": precommit.runtime.max_workers,
        "max_physical_model_calls": precommit.runtime.max_physical_model_calls,
        "headless": True,
        "pure_python_predicates": True,
        "lean_required": False,
        "fixed_query_denominator": 24,
        "launch_authorization_digest": authorization.authorization_digest,
        "campaign_runtime_precommit_digest": precommit.precommit_digest,
        "calibration_assessment_digest": (
            authorization.calibration_assessment_digest
        ),
        "calibration_replay_digest": authorization.calibration_replay_digest,
        "calibration_observation_inventory_digest": (
            authorization.calibration_observation_inventory_digest
        ),
    }

    def result_for(value):
        archive = SimpleNamespace(
            execution_precommit=SimpleNamespace(configuration=tuple(value.items())),
            runtime_binding_digest=precommit.runtime.binding_digest,
        )
        return SimpleNamespace(campaign=SimpleNamespace(archive=archive))

    command._assert_campaign_launch_gate(
        result_for(configuration), authorization, precommit
    )
    configuration["uncommitted_escape_hatch"] = True
    with pytest.raises(
        ObjectBongardRubricCampaignCommandError,
        match="not bound to the accepted calibration",
    ):
        command._assert_campaign_launch_gate(
            result_for(configuration), authorization, precommit
        )

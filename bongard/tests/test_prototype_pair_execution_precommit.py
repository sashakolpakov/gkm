from __future__ import annotations

import ast
import builtins
from dataclasses import fields, replace
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from bongard.canonical import canonical_digest
from bongard.grounded_multimodal_predicates import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.prototype_pair_cohort import PrototypePairCohortPlan
from bongard.prototype_pair_execution_precommit import (
    PHASE_ORDER,
    REQUIRED_RUNTIME_SOURCE_ROLES,
    PrototypePairExecutionIdentities,
    PrototypePairExecutionPrecommit,
    PrototypePairExecutionPrecommitError,
    prepare_prototype_pair_execution_precommit,
    verify_prototype_pair_execution_precommit,
)
from bongard.prototype_scene_calibration import (
    PrototypeSceneTagThreshold,
    calibration_algorithm_digest,
    threshold_commitment,
)


_PLAN_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "prototype_pair_targeted_engineering_20260807.plan.json"
)
_POLICY_CACHE = (
    "sha256:d89c1d515983bdb19c39330b8789b9266b3a8f6969145878e5dc06c3a1c9da14"
)
_LAUNCHER = "19c4f144c5226a9f17c58e6f0fa854843b0f77a6eb420f40e2745a12f10f5d37"


def _raw(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _address(label: str) -> str:
    return "sha256:" + canonical_digest(label)


def _plan() -> PrototypePairCohortPlan:
    return PrototypePairCohortPlan.from_data(json.loads(_PLAN_PATH.read_bytes()))


def _identities(
    plan: PrototypePairCohortPlan,
    *,
    execution_configuration_digest: str | None = None,
) -> PrototypePairExecutionIdentities:
    thresholds = (
        PrototypeSceneTagThreshold("opaque_visual_tag_0", 250_000, 750_000),
        PrototypeSceneTagThreshold("opaque_visual_tag_1", 250_000, 750_000),
    )
    source_digests = {
        role: _raw(f"source:{role}") for role in REQUIRED_RUNTIME_SOURCE_ROLES
    }
    source_digests["observer"] = _raw("observer-final-a1df9d")
    source_digests["transport"] = _raw("transport-final-0848ee")
    return PrototypePairExecutionIdentities.create(
        exposure_predecessor_digest=plan.exposure_predecessor_digest,
        execution_configuration_digest=(
            execution_configuration_digest or _address("execution configuration")
        ),
        thresholds=thresholds,
        threshold_commitment=threshold_commitment(thresholds),
        calibration_algorithm_digest=calibration_algorithm_digest(),
        observer_protocol_id=(
            "bongard.prototype-scene-observer/two-phase-neutral-prototypes-v1"
        ),
        observer_description_protocol_digest=_raw("description protocol"),
        observer_scoring_protocol_digest=_raw("scoring protocol"),
        observer_environment_digest=_raw("environment"),
        observer_model_id="gpt-5.6-sol",
        observer_reasoning_effort="medium",
        observer_model_identity_digest=_raw("observer:gpt-5.6-sol:medium"),
        ranker_model_id="gpt-5.6-sol",
        ranker_reasoning_effort="medium",
        ranker_model_identity_digest=_raw("ranker:gpt-5.6-sol:medium"),
        runner_protocol_id="bongard.prototype-pair-headless-runner/v1",
        runner_algorithm_digest=_address("runner algorithm"),
        codex_cli_version="codex-test-pinned",
        codex_launcher_sha256=_LAUNCHER,
        cloud_policy_cache_binding=_POLICY_CACHE,
        python_runtime_id="cpython-test-pinned",
        python_runtime_identity_digest=_raw("python runtime"),
        runtime_source_digests=source_digests,
    )


def _prepare(
    plan: PrototypePairCohortPlan,
    identities: PrototypePairExecutionIdentities,
) -> PrototypePairExecutionPrecommit:
    return prepare_prototype_pair_execution_precommit(
        cohort_plan=plan,
        identities=identities,
        expected_cohort_plan_digest=plan.record_digest,
        expected_identity_bundle_digest=identities.record_digest,
        expected_exposure_predecessor_digest=plan.exposure_predecessor_digest,
    )


def test_deterministic_split_is_held_out_disjoint_complete_and_cold_replays() -> None:
    plan = _plan()
    identities = _identities(plan)
    first = _prepare(plan, identities)
    second = _prepare(plan, identities)
    assert first == second
    assert len(first.support_roles) == 12
    assert len(first.query_roles) == 2
    assert {
        side: sum(item.opaque_side_id == side for item in first.support_roles)
        for side in ("side_0", "side_1")
    } == {"side_0": 6, "side_1": 6}
    assert {
        side: sum(item.opaque_side_id == side for item in first.query_roles)
        for side in ("side_0", "side_1")
    } == {"side_0": 1, "side_1": 1}
    support = set(first.support_panel_ids)
    query = set(first.query_panel_ids)
    expected = set(plan.drill.positive_panel_ids + plan.drill.negative_panel_ids)
    assert support.isdisjoint(query)
    assert support | query == expected
    assert len(support) == 12 and len(query) == 2
    assert all("positive" not in item.role_id for item in first.query_roles)
    assert all("negative" not in item.role_id for item in first.query_roles)

    archived = first.to_data()
    assert PrototypePairExecutionPrecommit.from_data(archived) == first
    assert verify_prototype_pair_execution_precommit(
        archived,
        cohort_plan=plan.to_data(),
        identities=identities.to_data(),
        expected_precommit_digest=first.record_digest,
        expected_cohort_plan_digest=plan.record_digest,
        expected_identity_bundle_digest=identities.record_digest,
        expected_exposure_predecessor_digest=plan.exposure_predecessor_digest,
    ) == first


def test_query_is_sealed_before_candidate_freeze_with_exact_call_budgets() -> None:
    precommit = _prepare(_plan(), _identities(_plan()))
    data = precommit.to_data()
    execution = data["execution"]
    assert execution["phase_order"] == list(PHASE_ORDER) == [
        "execution_precommit_persisted",
        "exposure_successor_persisted",
        "six_prototype_pixels_released",
        "prototype_description_observed",
        "calibration_plan_frozen",
        "twenty_eight_calibration_scenes_released_and_observed",
        "calibration_family_and_predicate_library_frozen",
        "twelve_support_scenes_released_and_observed",
        "support_version_space_constructed",
        "headless_codex_candidate_ranked",
        "durable_python_candidate_frozen",
        "two_query_scenes_released_and_observed",
        "python_query_evaluation",
        "model_free_tamper_detecting_replay",
    ]
    assert execution["durable_candidate_freeze_phase_ordinal"] == 10
    assert execution["query_pixel_release_phase_ordinal"] == 11
    assert execution["query_roles_sealed_in_precommit"] is True
    assert execution["query_pixels_released_after_durable_candidate_freeze"] is True
    assert execution["formula_frozen_before_query_observation"] is True
    budgets = {item["phase_id"]: item for item in execution["call_budgets"]}
    assert budgets["prototype_description_observed"]["calls_when_condition_true"] == 1
    assert budgets[
        "twenty_eight_calibration_scenes_released_and_observed"
    ]["calls_when_condition_true"] == 28
    assert budgets[
        "twelve_support_scenes_released_and_observed"
    ]["calls_when_condition_true"] == 12
    assert budgets[
        "two_query_scenes_released_and_observed"
    ]["calls_when_condition_true"] == 2
    assert budgets[
        "two_query_scenes_released_and_observed"
    ]["condition"] == "durable_python_candidate_frozen"
    assert budgets["headless_codex_candidate_ranked"] == {
        "phase_id": "headless_codex_candidate_ranked",
        "actor": "headless_codex_ranker",
        "condition": "verified_survivor_set_nonempty",
        "calls_when_condition_true": 1,
        "calls_when_condition_false": 0,
    }
    assert execution["maximum_model_calls"] == 44
    assert execution["model_calls_on_complete_candidate_and_query_branch"] == 44
    assert execution["model_calls_on_no_verified_support_survivor_branch"] == 41
    assert execution["model_calls_on_calibration_family_rejected_branch"] == 29
    assert execution["model_calls_on_ranker_error_branch"] == 42


def test_exact_threshold_model_cli_policy_runtime_and_source_identities_are_bound() -> None:
    plan = _plan()
    identities = _identities(plan)
    precommit = _prepare(plan, identities)
    assert tuple(
        (item.absent_upper_ppm, item.present_lower_ppm)
        for item in precommit.identities.thresholds
    ) == ((250_000, 750_000), (250_000, 750_000))
    assert precommit.identities.threshold_commitment == threshold_commitment(
        precommit.identities.thresholds
    )
    assert precommit.identities.observer_model_id == "gpt-5.6-sol"
    assert precommit.identities.observer_reasoning_effort == "medium"
    assert precommit.identities.ranker_model_id == "gpt-5.6-sol"
    assert precommit.identities.ranker_reasoning_effort == "medium"
    assert precommit.identities.codex_launcher_sha256 == _LAUNCHER
    assert precommit.identities.cloud_policy_cache_binding == _POLICY_CACHE
    assert precommit.identities.execution_configuration_digest == _address(
        "execution configuration"
    )
    assert REQUIRED_RUNTIME_SOURCE_ROLES <= set(
        dict(precommit.identities.runtime_source_digests)
    )
    serialized_text = json.dumps(precommit.identities.to_data(), sort_keys=True)
    for outcome_dependent_name in (
        "calibration_plan_digest",
        "calibration_family_digest",
        "observer_catalog_digest",
        "observer_reference_digest",
    ):
        assert outcome_dependent_name not in serialized_text

    drifted = replace(identities, ranker_reasoning_effort="high")
    with pytest.raises(PrototypePairExecutionPrecommitError, match="identities"):
        prepare_prototype_pair_execution_precommit(
            cohort_plan=plan,
            identities=drifted,
            expected_cohort_plan_digest=plan.record_digest,
            expected_identity_bundle_digest=identities.record_digest,
            expected_exposure_predecessor_digest=plan.exposure_predecessor_digest,
        )
    with pytest.raises(PrototypePairExecutionPrecommitError, match="missing"):
        PrototypePairExecutionIdentities.create(
            **{
                field.name: getattr(identities, field.name)
                for field in fields(identities)
                if field.name != "runtime_source_digests"
            },
            runtime_source_digests={"observer": _raw("observer")},
        )


def test_serialized_tamper_and_external_pin_drift_fail_closed() -> None:
    plan = _plan()
    identities = _identities(plan)
    precommit = _prepare(plan, identities)
    tampered = json.loads(json.dumps(precommit.to_data()))
    tampered["execution"]["query_pixel_release_phase_ordinal"] = 5
    with pytest.raises(PrototypePairExecutionPrecommitError):
        PrototypePairExecutionPrecommit.from_data(tampered)
    tampered = json.loads(json.dumps(precommit.to_data()))
    tampered["roles"]["query"][0]["source_panel_id"] = "bd/forged/1/0.png"
    with pytest.raises(PrototypePairExecutionPrecommitError):
        PrototypePairExecutionPrecommit.from_data(tampered)
    with pytest.raises(PrototypePairExecutionPrecommitError, match="exposure"):
        prepare_prototype_pair_execution_precommit(
            cohort_plan=plan,
            identities=identities,
            expected_cohort_plan_digest=plan.record_digest,
            expected_identity_bundle_digest=identities.record_digest,
            expected_exposure_predecessor_digest="sha256:" + "0" * 64,
        )


def test_constructor_is_metadata_only_and_python_authoritative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    identities = _identities(plan)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("precommit attempted file or model access")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    precommit = _prepare(plan, identities)
    boundary = precommit.to_data()["construction_boundary"]
    assert boundary == {
        "accepted_inputs": "verified-plan-and-explicit-digests-only",
        "panel_bytes_read": False,
        "panel_paths_resolved": False,
        "action_program_json_read": False,
        "model_calls_made": False,
        "exposure_ledger_read": False,
        "exposure_ledger_mutated": False,
    }
    claim = precommit.to_data()["claim_scope"]
    assert claim == {
        "split": "train",
        "exact_task_unused": True,
        "drill_semantics_reused": True,
        "targeted_engineering_claim": True,
        "benchmark_claim_authorized": False,
        "unseen_claim_authorized": False,
        "validation_claim_authorized": False,
        "official_test_authorized": False,
    }
    authority = precommit.to_data()["runtime_authority"]
    assert authority["predicate_authority_id"] == PYTHON_PREDICATE_AUTHORITY_ID
    assert authority["python_is_canonical_authority"] is True
    assert authority["lean_required"] is False
    assert authority["lean_removable"] is True
    assert authority["lean_defines_artifact_identity"] is False
    assert authority["lean_affects_selection_or_decision"] is False
    assert authority["lean_required_for_replay"] is False

    parameters = set(
        inspect.signature(prepare_prototype_pair_execution_precommit).parameters
    )
    assert parameters == {
        "cohort_plan",
        "identities",
        "expected_cohort_plan_digest",
        "expected_identity_bundle_digest",
        "expected_exposure_predecessor_digest",
    }
    module = inspect.getmodule(prepare_prototype_pair_execution_precommit)
    assert module is not None
    tree = ast.parse(inspect.getsource(module))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("lean" in name.lower() for name in imported)
    assert not any("transport" in name.lower() for name in imported)
    assert "ExposureLedger" not in imported

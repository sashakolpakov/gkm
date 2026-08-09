"""Offline task/role custody tests for the closed-catalog inventory."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from bongard.canonical import canonical_digest
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.object_bongard_release_gate import (
    ObjectBongardExecutionPrecommit,
    ObjectBongardReleaseStore,
    _precommit_content,
    persist_object_bongard_task_commit,
    persist_object_bongard_task_freeze,
)
from bongard.official_extracted_panel_archive import (
    OfficialExtractedPanelReceipt,
    ReleasedOfficialExtractedPanel,
    _released_content,
)
from bongard.panel_batched_typed_codex_observer import (
    BatchedFeatureAxisRequest,
    complete_whole_panel_feature_axes,
    observe_typed_panel_axes_batched,
)
from bongard.panel_feature_closed_catalog_inventory import (
    ClosedCatalogSupportInventory,
)
from bongard.panel_feature_evidence_bundle import (
    PanelFeatureEvidenceBundle,
    PanelFeatureEvidencePanel,
    PanelFeatureEvidencePhase,
)
from bongard.panel_feature_task_bound_inventory import (
    TaskBoundClosedCatalogInventory,
    TaskBoundClosedCatalogInventoryError,
    cold_replay_task_bound_closed_catalog_inventory,
)
from bongard.panel_feature_primary_task_runner import (
    PrimaryFormulaQueryDecision,
    PrimaryFormulaSupportPhase,
    PrimaryFormulaSupportStatus,
    PrimaryFormulaTaskFreeze,
    PrimaryFormulaTaskFreezeCommit,
    PrimaryFormulaTaskRunnerError,
    classify_primary_formula_survivor_count,
    cold_replay_primary_formula_query_decision,
    verify_primary_formula_task_commit,
    verify_primary_formula_task_freeze,
)
from bongard.panel_feature_predicate import (
    EngineeringDisposition,
    EngineeringQueryOutcome,
)
from bongard.panel_soft_ontology import (
    FeatureFamily,
    GestaltKind,
    NativeOrientation,
    SymmetryKind,
)
from bongard.panel_typed_codex_observer import build_panel_only_observation_context
from bongard.tests.test_panel_batched_typed_codex_observer import (
    _transport as _batched_transport,
)
from bongard.tests.test_panel_feature_evidence_bundle import (
    _proposer,
)
from bongard.tests.test_panel_feature_proposer import _payload as _proposer_payload
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _png,
)


def _task() -> ObjectBongardTaskPlan:
    return ObjectBongardTaskPlan.create(
        "hd_task_bound_inventory_fixture_0000",
        seed_digest="2" * 64,
    )


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _precommit(task: ObjectBongardTaskPlan) -> ObjectBongardExecutionPrecommit:
    values = {
        "batch_plan_digest": _address({"batch": task.record_digest}),
        "batch_algorithm_digest": _address({"batch_algorithm": 1}),
        "batch_source_digest": _address({"batch_source": 1}),
        "release_gate_source_digest": _address({"release_gate_source": 1}),
        "release_descriptor_digest": _address({"release": 1}),
        "archive_record_digest": _address({"archive_record": 1}),
        "archive_digest": _address({"archive": 1}),
        "archive_central_directory_digest": _address({"central": 1}),
        "corpus_digest": _address({"corpus": 1}),
        "exposure_predecessor_digest": _address({"predecessor": 1}),
        "task_inventory_digest": _address({"inventory": [task.task_id]}),
        "train_task_ids_digest": _address([task.task_id]),
        "exact_used_task_ids_digest": _address([]),
        "selected_task_ids": (task.task_id,),
        "authorized_support_panel_ids": tuple(
            sorted(task.side_0_support_panel_ids + task.side_1_support_panel_ids)
        ),
        "sealed_query_panel_ids": tuple(
            sorted((task.side_0_query_panel_id, task.side_1_query_panel_id))
        ),
        "runtime_source_bindings": (("runner_source", _address({"source": 1})),),
        "configuration": (("headless", True),),
        "exposure_observed_at": "2026-08-09T12:00:00Z",
        "exposure_actor": "synthetic-test",
        "exposure_purpose": "primary-formula-task-runner",
        "exposure_source": "offline-synthetic",
    }
    provisional = object.__new__(ObjectBongardExecutionPrecommit)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardExecutionPrecommit(
        **values,
        record_digest=_address(_precommit_content(provisional)),
    )


def _variant_alias(item, kind: object) -> str:
    matches = tuple(
        variant.alias
        for variant in item.view.variants
        if getattr(variant.spec.parameters, "kind", None) is kind
    )
    assert len(matches) == 1
    return matches[0]


def _support_payload(
    request: BatchedFeatureAxisRequest,
    index: int,
    *,
    negative_mode: str = "heterogeneous",
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for item in request.aliases:
        assert len(item.view.bindings) == 1
        binding = item.view.bindings[0]
        family = item.view.axis.family
        if family is FeatureFamily.GESTALT_RESEMBLANCE:
            kind = (
                GestaltKind.BIRD_LIKE
                if index < 6
                or (negative_mode == "heterogeneous" and index < 9)
                else GestaltKind.ANIMAL_LIKE
            )
            row = {
                "resolution": "complete",
                "variant_evidence": [
                    {
                        "variant_alias": _variant_alias(item, kind),
                        "evidence_x": binding.search_region.minimum.x,
                        "evidence_y": binding.search_region.minimum.y,
                    }
                ],
                "issue": "none",
            }
        elif family is FeatureFamily.SYMMETRY:
            kind = (
                SymmetryKind.REFLECTIONAL
                if index < 6
                or (negative_mode == "heterogeneous" and index >= 9)
                else SymmetryKind.HALF_TURN
            )
            row = {
                "resolution": "complete",
                "variant_evidence": [
                    {
                        "variant_alias": _variant_alias(item, kind),
                        "evidence_x": binding.search_region.minimum.x,
                        "evidence_y": binding.search_region.minimum.y,
                    }
                ],
                "issue": "none",
            }
        elif family is FeatureFamily.STRAIGHT_SEGMENT_COUNT:
            row = {
                "resolution": "unclear",
                "straight_segment_evidence": [],
                "issue": "missing_straightness_evidence",
            }
        elif family is FeatureFamily.CONVEXITY:
            row = {
                "resolution": "unclear",
                "outer_boundary_vertices": [],
                "issue": "missing_boundary_evidence",
            }
        else:
            row = {
                "resolution": "unclear",
                "variant_evidence": [],
                "issue": "ambiguous_geometry",
            }
        payload[item.alias] = {binding.alias: row}
    return payload


def _bundle_and_calls(
    *, negative_mode: str = "heterogeneous"
) -> tuple[
    ObjectBongardTaskPlan,
    PanelFeatureEvidenceBundle,
    list[dict[str, object]],
]:
    task = _task()
    support_ids = task.side_0_support_panel_ids + task.side_1_support_panel_ids
    panels = tuple(_png(2100 + index) for index in range(12))
    proposer_artifact, proposer_result = _proposer(panels, _proposer_payload())
    axes = complete_whole_panel_feature_axes()
    calls: list[dict[str, object]] = []
    rows = []
    for index, (panel_id, panel) in enumerate(zip(support_ids, panels, strict=True)):
        context = build_panel_only_observation_context(
            panel,
            model=MODEL,
            reasoning_effort=EFFORT,
            expected_launcher_digest=LAUNCHER_DIGEST,
            **NO_TOOLS_KWARGS,
        )
        request = BatchedFeatureAxisRequest.build(context, axes)
        artifact = observe_typed_panel_axes_batched(
            panel,
            axes=axes,
            panel_only_context=context,
            model=MODEL,
            reasoning_effort=EFFORT,
            expected_launcher_digest=LAUNCHER_DIGEST,
            **NO_TOOLS_KWARGS,
            transport=_batched_transport(
                _support_payload(
                    request, index, negative_mode=negative_mode
                ),
                panel,
                calls,
            ),
        )
        rows.append(
            PanelFeatureEvidencePanel.derive_from_batched_artifact(
                phase=PanelFeatureEvidencePhase.SUPPORT,
                phase_index=index,
                panel_id=panel_id,
                panel_png=panel,
                batched_axis_artifact=artifact,
            )
        )
    return (
        task,
        PanelFeatureEvidenceBundle.create(
            proposer_artifact=proposer_artifact,
            proposer_result=proposer_result,
            observer_axes=axes,
            panels=rows,
        ),
        calls,
    )


@pytest.fixture(scope="module")
def bound_fixture():
    task, bundle, calls = _bundle_and_calls()
    inventory = ClosedCatalogSupportInventory.create(
        bundle.proposer_result,
        bundle.observation_sets_for_phase(PanelFeatureEvidencePhase.SUPPORT),
        primary_orientation=NativeOrientation.SIDE0_POSITIVE,
    )
    bound = TaskBoundClosedCatalogInventory.bind(task, bundle, inventory)
    return task, bundle, inventory, bound, calls


def test_exact_task_evidence_inventory_binding_and_zero_call_replay(
    bound_fixture,
) -> None:
    task, bundle, inventory, bound, calls = bound_fixture
    assert len(calls) == 12
    assert bound.task_plan == task
    assert bound.evidence_bundle == bundle
    assert bound.inventory == inventory
    assert tuple(item[0] for item in bound.support_panel_bindings) == (
        task.side_0_support_panel_ids + task.side_1_support_panel_ids
    )
    assert bound.sealed_query_panel_ids == (
        task.side_0_query_panel_id,
        task.side_1_query_panel_id,
    )
    assert bound.to_data()["bare_observation_sequence_accepted"] is False
    assert bound.to_data()["query_panel_count"] == 0
    assert bound.to_data()["lean_required"] is False
    assert TaskBoundClosedCatalogInventory.from_data(bound.to_data()) == bound
    assert (
        cold_replay_task_bound_closed_catalog_inventory(
            bound, expected_artifact_address=bound.artifact_address
        )
        == bound
    )
    assert len(calls) == 12


def test_role_swap_wrong_task_and_bare_inventory_fail_closed(bound_fixture) -> None:
    task, bundle, inventory, _bound, _calls = bound_fixture
    first, second, *rest = bundle.panels
    swapped = (
        PanelFeatureEvidencePanel.create(
            phase=first.phase,
            phase_index=first.phase_index,
            panel_id=second.panel_id,
            panel_png=first.panel_png,
            owner_artifact=first.owner_artifact,
            axis_artifacts=first.axis_artifacts,
            batched_axis_artifact=first.batched_axis_artifact,
            observation_set=first.observation_set,
        ),
        PanelFeatureEvidencePanel.create(
            phase=second.phase,
            phase_index=second.phase_index,
            panel_id=first.panel_id,
            panel_png=second.panel_png,
            owner_artifact=second.owner_artifact,
            axis_artifacts=second.axis_artifacts,
            batched_axis_artifact=second.batched_axis_artifact,
            observation_set=second.observation_set,
        ),
        *rest,
    )
    swapped_bundle = PanelFeatureEvidenceBundle.create(
        proposer_artifact=bundle.proposer_artifact,
        proposer_result=bundle.proposer_result,
        observer_axes=bundle.observer_axes,
        panels=swapped,
    )
    with pytest.raises(TaskBoundClosedCatalogInventoryError):
        TaskBoundClosedCatalogInventory.bind(task, swapped_bundle, inventory)

    other_task = ObjectBongardTaskPlan.create(
        "hd_task_bound_inventory_fixture_0001", seed_digest="2" * 64
    )
    with pytest.raises(TaskBoundClosedCatalogInventoryError):
        TaskBoundClosedCatalogInventory.bind(other_task, bundle, inventory)
    with pytest.raises(TypeError):
        TaskBoundClosedCatalogInventory.bind(  # type: ignore[arg-type]
            task, object(), inventory
        )


def test_observation_order_primary_orientation_and_tamper_fail_closed(
    bound_fixture,
) -> None:
    task, bundle, inventory, bound, _calls = bound_fixture
    reversed_inventory = ClosedCatalogSupportInventory.create(
        bundle.proposer_result,
        tuple(reversed(inventory.support_observations)),
        primary_orientation=NativeOrientation.SIDE0_POSITIVE,
    )
    with pytest.raises(TaskBoundClosedCatalogInventoryError):
        TaskBoundClosedCatalogInventory.bind(task, bundle, reversed_inventory)

    wrong_orientation = ClosedCatalogSupportInventory.create(
        bundle.proposer_result,
        inventory.support_observations,
        primary_orientation=NativeOrientation.SIDE1_POSITIVE,
    )
    with pytest.raises(TaskBoundClosedCatalogInventoryError):
        TaskBoundClosedCatalogInventory.bind(task, bundle, wrong_orientation)

    tampered = deepcopy(bound.to_data())
    tampered["support_panel_bindings"][0][0] = task.side_0_query_panel_id
    with pytest.raises(TaskBoundClosedCatalogInventoryError):
        TaskBoundClosedCatalogInventory.from_data(tampered)


def _query_evidence(
    *,
    panel_id: str,
    phase_index: int,
    payload_index: int,
) -> PanelFeatureEvidencePanel:
    panel = _png(3100 + phase_index)
    axes = complete_whole_panel_feature_axes()
    context = build_panel_only_observation_context(
        panel,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
    )
    request = BatchedFeatureAxisRequest.build(context, axes)
    calls: list[dict[str, object]] = []
    artifact = observe_typed_panel_axes_batched(
        panel,
        axes=axes,
        panel_only_context=context,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=_batched_transport(
            _support_payload(request, payload_index), panel, calls
        ),
    )
    assert len(calls) == 1
    return PanelFeatureEvidencePanel.derive_from_batched_artifact(
        phase=PanelFeatureEvidencePhase.QUERY,
        phase_index=phase_index,
        panel_id=panel_id,
        panel_png=panel,
        batched_axis_artifact=artifact,
    )


def _released_query(
    evidence: PanelFeatureEvidencePanel,
    precommit: ObjectBongardExecutionPrecommit,
) -> ReleasedOfficialExtractedPanel:
    family, task_id, side, filename = evidence.panel_id.split("/")
    receipt = OfficialExtractedPanelReceipt.seal(
        panel_id=evidence.panel_id,
        relative_path=f"{family}/images/{task_id}/{side}/{filename}",
        payload=evidence.panel_png,
        task_manifest_digest=_address({"task_manifest": task_id}),
        corpus_manifest_digest=_address({"corpus_manifest": 1}),
        release_descriptor_digest=_address({"release_descriptor": 1}),
        extracted_archive_digest=_address({"extracted_archive": 1}),
    )
    values = {
        "panel_id": evidence.panel_id,
        "exact_png_bytes": evidence.panel_png,
        "exact_png_digest": receipt.sha256,
        "release_receipt": receipt,
        "execution_precommit_digest": precommit.record_digest,
        "exposure_successor_digest": _address({"exposure_successor": 1}),
    }
    provisional = object.__new__(ReleasedOfficialExtractedPanel)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ReleasedOfficialExtractedPanel(
        **values,
        record_digest=_address(_released_content(provisional)),
    )


def test_unique_primary_freeze_commit_query_mapping_and_zero_call_replay(
    bound_fixture,
    tmp_path: Path,
) -> None:
    task, _bundle, _inventory, bound, support_calls = bound_fixture
    phase = PrimaryFormulaSupportPhase.create(bound)
    assert phase.status is PrimaryFormulaSupportStatus.UNIQUE_PRIMARY_SURVIVOR
    assert phase.primary_survivor_count == 1
    assert phase.gap is None
    assert PrimaryFormulaSupportPhase.from_data(phase.to_data()) == phase

    precommit = _precommit(task)
    freeze = PrimaryFormulaTaskFreeze.seal(
        support_phase=phase,
        execution_precommit=precommit,
    )
    assert freeze.selection_mode == "unique_primary_support_survivor"
    assert freeze.rank_artifact is None
    assert freeze.rank_journal_terminal is None
    assert freeze.resolve_selected_all_of() == (
        bound.inventory.primary_version_space.survivor_formulas[0]
    )
    assert (
        verify_primary_formula_task_freeze(
            freeze, expected_record_digest=freeze.record_digest
        )
        == freeze
    )

    store = ObjectBongardReleaseStore((tmp_path / "store").resolve())
    freeze_receipt = persist_object_bongard_task_freeze(
        store=store, freeze=freeze
    )
    commit = PrimaryFormulaTaskFreezeCommit.seal(freeze, freeze_receipt)
    commit_receipt = persist_object_bongard_task_commit(store=store, commit=commit)
    assert (
        verify_primary_formula_task_commit(
            commit,
            expected_record_digest=commit.record_digest,
            task_commit_store_receipt=commit_receipt,
        )
        == commit
    )

    positive_evidence = _query_evidence(
        panel_id=task.side_0_query_panel_id,
        phase_index=0,
        payload_index=0,
    )
    positive_release = _released_query(positive_evidence, precommit)
    positive_receipt = store.persist(
        object_kind="released-query-panel",
        object_digest=positive_release.record_digest,
        data=positive_release.to_data(),
    )
    positive = PrimaryFormulaQueryDecision.create(
        freeze,
        released_query_panel=positive_release,
        query_release_store_receipt=positive_receipt,
        query_evidence_panel=positive_evidence,
    )
    assert positive.formula_disposition is EngineeringDisposition.MATCH
    assert positive.outcome is EngineeringQueryOutcome.SIDE0
    assert (
        cold_replay_primary_formula_query_decision(
            positive,
            freeze=freeze,
            expected_artifact_address=positive.artifact_address,
        )
        == positive
    )

    negative_evidence = _query_evidence(
        panel_id=task.side_1_query_panel_id,
        phase_index=1,
        payload_index=10,
    )
    negative_release = _released_query(negative_evidence, precommit)
    negative_receipt = store.persist(
        object_kind="released-query-panel",
        object_digest=negative_release.record_digest,
        data=negative_release.to_data(),
    )
    negative = PrimaryFormulaQueryDecision.create(
        freeze,
        released_query_panel=negative_release,
        query_release_store_receipt=negative_receipt,
        query_evidence_panel=negative_evidence,
    )
    assert negative.formula_disposition is EngineeringDisposition.NONMATCH
    assert negative.outcome is EngineeringQueryOutcome.SIDE1
    assert len(support_calls) == 12


def test_query_id_evidence_swap_bare_observation_and_freeze_tamper_fail_closed(
    bound_fixture,
    tmp_path: Path,
) -> None:
    task, _bundle, _inventory, bound, _support_calls = bound_fixture
    phase = PrimaryFormulaSupportPhase.create(bound)
    precommit = _precommit(task)
    freeze = PrimaryFormulaTaskFreeze.seal(
        support_phase=phase,
        execution_precommit=precommit,
    )
    evidence = _query_evidence(
        panel_id=task.side_0_query_panel_id,
        phase_index=1,
        payload_index=0,
    )
    released = _released_query(evidence, precommit)
    store = ObjectBongardReleaseStore((tmp_path / "swap-store").resolve())
    receipt = store.persist(
        object_kind="released-query-panel",
        object_digest=released.record_digest,
        data=released.to_data(),
    )
    with pytest.raises(PrimaryFormulaTaskRunnerError, match="swapped"):
        PrimaryFormulaQueryDecision.create(
            freeze,
            released_query_panel=released,
            query_release_store_receipt=receipt,
            query_evidence_panel=evidence,
        )
    with pytest.raises(TypeError):
        PrimaryFormulaQueryDecision.create(  # type: ignore[call-arg]
            freeze,
            query_panel_id=task.side_0_query_panel_id,
            observation=evidence.observation_set,
        )

    tampered = deepcopy(freeze.to_data())
    tampered["selected_formula_digest"] = "0" * 64
    with pytest.raises(PrimaryFormulaTaskRunnerError):
        PrimaryFormulaTaskFreeze.from_data(tampered)


def test_rank_capacity_is_a_typed_closed_phase_before_rank_input_construction() -> None:
    status, kind = classify_primary_formula_survivor_count(257)
    assert status is PrimaryFormulaSupportStatus.RANK_CAPACITY_GAP
    assert kind is not None
    assert kind.value == "primary_survivor_count_exceeds_rank_capacity"
    assert classify_primary_formula_survivor_count(256) == (
        PrimaryFormulaSupportStatus.RANK_REQUIRED,
        None,
    )
    assert classify_primary_formula_survivor_count(0)[0] is (
        PrimaryFormulaSupportStatus.PRIMARY_SUPPORT_GAP
    )


def test_multiple_primary_survivors_require_rank_artifact_and_exact_journal() -> None:
    task, bundle, _calls = _bundle_and_calls(negative_mode="both_absent")
    inventory = ClosedCatalogSupportInventory.create(
        bundle.proposer_result,
        bundle.observation_sets_for_phase(PanelFeatureEvidencePhase.SUPPORT),
        primary_orientation=NativeOrientation.SIDE0_POSITIVE,
    )
    bound = TaskBoundClosedCatalogInventory.bind(task, bundle, inventory)
    phase = PrimaryFormulaSupportPhase.create(bound)
    assert phase.status is PrimaryFormulaSupportStatus.RANK_REQUIRED
    assert phase.primary_survivor_count == 3
    with pytest.raises(PrimaryFormulaTaskRunnerError, match="exact rank artifact"):
        PrimaryFormulaTaskFreeze.seal(
            support_phase=phase,
            execution_precommit=_precommit(task),
        )

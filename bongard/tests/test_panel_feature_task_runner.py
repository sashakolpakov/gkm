from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from bongard.canonical import canonical_digest
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.panel_feature_observation import (
    EngineeringFeatureDisposition,
    FeatureAxis,
    PanelFeatureObservationSet,
    derive_inventory_count_observation,
)
from bongard.panel_feature_predicate import (
    EngineeringDisposition,
    EngineeringQueryOutcome,
    FrozenEngineeringFeaturePredicatePair,
)
from bongard.panel_feature_proposer import (
    PANEL_FEATURE_BLOCKS,
    PANEL_FEATURE_NONE,
    PANEL_FEATURE_PRESENTATION_NAMES,
    PANEL_FEATURE_SLOTS_PER_DIRECTION,
    panel_feature_spec_to_wire,
    parse_panel_feature_proposer_payload,
)
from bongard.panel_feature_task_runner import (
    PanelFeatureTaskArchive,
    PanelFeatureTaskRunStatus,
    PanelFeatureTaskRunnerError,
    cold_replay_panel_feature_task,
    engineering_disposition_from_observation,
    run_panel_feature_task,
    run_panel_feature_task_with_support_callbacks,
)
from bongard.panel_soft_ontology import (
    ClosedCount,
    ComponentCountParameters,
    EnumerationResolution,
    FeatureFamily,
    NativeOrientation,
    OwnerId,
    OwnerInventory,
    OwnerKind,
    PanelFeatureSpec,
    PanelLocalOwner,
    QuantizedPoint,
    QuantizedRegion,
    ReferenceFrame,
    SubjectScope,
)


_TASK_SEED = "sha256:" + "7" * 64
_RECEIPT = "8" * 64
_CONTRACT = "9" * 64
_PROTOCOL = "a" * 64


def _task() -> ObjectBongardTaskPlan:
    return ObjectBongardTaskPlan.create(
        "bd_panel_feature_runner_0000", seed_digest=_TASK_SEED
    )


def _png(index: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + f"panel-{index:04d}".encode("ascii")


def _count_spec(count: ClosedCount) -> PanelFeatureSpec:
    return PanelFeatureSpec(
        FeatureFamily.COMPONENT_COUNT,
        SubjectScope.WHOLE_PANEL,
        ReferenceFrame.NONE,
        ComponentCountParameters(count),
    )


def _candidate(
    spec: PanelFeatureSpec, *, native_block: str, suffix: str
) -> dict[str, object]:
    result: dict[str, object] = {
        "candidate_kind": "registered_feature",
        **panel_feature_spec_to_wire(spec),
        "language_gap_kind": PANEL_FEATURE_NONE,
        "archival_summary": f"A visible exact component count {suffix}",
        "archival_indicator_a": f"Complete component grouping {suffix}",
        "archival_indicator_b": f"Panel-wide count witness {suffix}",
    }
    for block in PANEL_FEATURE_BLOCKS:
        for index in range(6):
            result[f"{block}_panel_{index:03d}_estimate"] = (
                "supports" if block == native_block else "does_not_support"
            )
    return result


def _payload() -> dict[str, object]:
    result: dict[str, object] = {}
    for block_index, block in enumerate(PANEL_FEATURE_BLOCKS):
        for slot in range(PANEL_FEATURE_SLOTS_PER_DIRECTION):
            count = ClosedCount.ONE if block_index == 0 else ClosedCount.FIVE
            result[f"{block}_candidate_{slot}"] = _candidate(
                _count_spec(count),
                native_block=block,
                suffix=f"{block}-{slot}",
            )
    return result


def _presentation_digest(pngs: tuple[bytes, ...]) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-panel-feature-proposer-presentation.v1",
            "images": [
                {"name": name, "sha256": hashlib.sha256(panel).hexdigest()}
                for name, panel in zip(
                    PANEL_FEATURE_PRESENTATION_NAMES, pngs, strict=True
                )
            ],
        }
    )


def _proposer(task: ObjectBongardTaskPlan, pngs: tuple[bytes, ...]):
    return parse_panel_feature_proposer_payload(
        _payload(),
        proposer_receipt_digest=_RECEIPT,
        support_set_digest=_presentation_digest(pngs),
        task_context_digest=task.record_digest.split(":", 1)[1],
    )


def _inventory(panel: bytes, count: int, *, complete: bool = True) -> OwnerInventory:
    region = QuantizedRegion(QuantizedPoint(0, 0), QuantizedPoint(15, 15))
    owners = tuple(
        PanelLocalOwner(
            OwnerId(f"owner_{index:04d}"), OwnerKind.FIGURE, region
        )
        for index in range(1, count + 1)
    )
    return OwnerInventory(
        hashlib.sha256(panel).hexdigest(),
        "b" * 64,
        EnumerationResolution.GRID16_FULL_PANEL,
        canonical_digest(
            {
                "panel_digest": hashlib.sha256(panel).hexdigest(),
                "count": count,
                "complete": complete,
            }
        ),
        complete,
        owners,
    )


def _observation(
    panel: bytes,
    count: int,
    *,
    complete: bool = True,
    contract: str = _CONTRACT,
    protocol: str = _PROTOCOL,
) -> PanelFeatureObservationSet:
    inventory = _inventory(panel, count, complete=complete)
    axis = FeatureAxis.for_spec(_count_spec(ClosedCount.ONE))
    observation = derive_inventory_count_observation(
        inventory,
        axis,
        observer_contract_digest=contract,
        measurement_protocol_digest=protocol,
    )
    return PanelFeatureObservationSet(
        inventory.panel_digest, contract, protocol, (observation,)
    )


def _fixture(*, incomplete: bool = False):
    task = _task()
    support_pngs = tuple(_png(index) for index in range(12))
    proposer = _proposer(task, support_pngs)
    observations = tuple(
        _observation(
            panel,
            1 if index < 6 else 5,
            complete=not incomplete,
        )
        for index, panel in enumerate(support_pngs)
    )
    return task, support_pngs, proposer, observations


def test_complete_run_freezes_before_query_and_cold_replays_without_callbacks() -> None:
    task, support_pngs, proposer, observations = _fixture()
    events: list[str] = []
    frozen_bytes: list[bytes] = []

    def persist_and_reload(value: bytes) -> bytes:
        events.append("persist_reload")
        frozen_bytes.append(value)
        return value

    def query(pair: FrozenEngineeringFeaturePredicatePair):
        events.append("query")
        assert frozen_bytes
        assert json.loads(frozen_bytes[0]) == pair.to_data()
        panels = (_png(100), _png(101))
        return {
            task.side_0_query_panel_id: (
                panels[0],
                _observation(panels[0], 1),
            ),
            task.side_1_query_panel_id: (
                panels[1],
                _observation(panels[1], 5),
            ),
        }

    archive = run_panel_feature_task(
        task,
        support_pngs,
        proposer,
        observations,
        persist_and_reload=persist_and_reload,
        query_callback=query,
    )

    assert events == ["persist_reload", "query"]
    assert archive.status is PanelFeatureTaskRunStatus.COMPLETE
    assert archive.persist_reload_callback_invocations == 1
    assert archive.query_callback_invocations == 1
    assert tuple(item.outcome for item in archive.query_decisions) == (
        EngineeringQueryOutcome.SIDE0,
        EngineeringQueryOutcome.SIDE1,
    )
    assert archive.archive_address == "sha256:" + archive.record_digest
    assert PanelFeatureTaskArchive.from_data(archive.to_data()) == archive
    assert (
        cold_replay_panel_feature_task(
            archive, expected_archive_address=archive.archive_address
        )
        == archive
    )


def test_empty_version_space_returns_typed_gap_without_touching_callbacks() -> None:
    task, support_pngs, proposer, _ = _fixture()
    observations = tuple(
        _observation(panel, 1 if index < 6 else 5, complete=False)
        for index, panel in enumerate(support_pngs)
    )

    def forbidden(_value):
        raise AssertionError("support gap invoked a forbidden callback")

    archive = run_panel_feature_task(
        task,
        support_pngs,
        proposer,
        observations,
        persist_and_reload=forbidden,
        query_callback=forbidden,
    )
    assert archive.status is PanelFeatureTaskRunStatus.SUPPORT_GAP
    assert archive.support_gap is not None
    assert archive.support_gap.missing_orientations == tuple(NativeOrientation)
    assert archive.persist_reload_callback_invocations == 0
    assert archive.query_callback_invocations == 0
    assert archive.query_decisions == ()
    assert cold_replay_panel_feature_task(
        archive, expected_archive_address=archive.archive_address
    ) == archive


def test_proposer_must_bind_task_presentation_and_semantic_orientations() -> None:
    task, support_pngs, _, observations = _fixture()
    wrong_task = parse_panel_feature_proposer_payload(
        _payload(),
        proposer_receipt_digest=_RECEIPT,
        support_set_digest=_presentation_digest(support_pngs),
        task_context_digest="f" * 64,
    )
    with pytest.raises(PanelFeatureTaskRunnerError, match="provenance"):
        run_panel_feature_task(
            task,
            support_pngs,
            wrong_task,
            observations,
            persist_and_reload=None,
            query_callback=None,
        )

    swapped = parse_panel_feature_proposer_payload(
        _payload(),
        proposer_receipt_digest=_RECEIPT,
        support_set_digest=_presentation_digest(support_pngs),
        task_context_digest=task.record_digest.split(":", 1)[1],
        block_orientations=(
            NativeOrientation.SIDE1_POSITIVE,
            NativeOrientation.SIDE0_POSITIVE,
        ),
    )
    with pytest.raises(PanelFeatureTaskRunnerError, match="orientation"):
        run_panel_feature_task(
            task,
            support_pngs,
            swapped,
            observations,
            persist_and_reload=None,
            query_callback=None,
        )


def test_observation_matrix_requires_exact_axes_pixels_and_shared_protocol() -> None:
    task, support_pngs, proposer, observations = _fixture()
    missing_axis = PanelFeatureObservationSet(
        observations[0].panel_digest,
        observations[0].observer_contract_digest,
        observations[0].measurement_protocol_digest,
        (),
    )
    with pytest.raises(PanelFeatureTaskRunnerError, match="exact complete"):
        run_panel_feature_task(
            task,
            support_pngs,
            proposer,
            (missing_axis, *observations[1:]),
            persist_and_reload=None,
            query_callback=None,
        )

    foreign_protocol = _observation(support_pngs[0], 1, protocol="c" * 64)
    with pytest.raises(PanelFeatureTaskRunnerError, match="share one"):
        run_panel_feature_task(
            task,
            support_pngs,
            proposer,
            (foreign_protocol, *observations[1:]),
            persist_and_reload=None,
            query_callback=None,
        )

    with pytest.raises(PanelFeatureTaskRunnerError, match="provenance|different PNG"):
        run_panel_feature_task(
            task,
            (*support_pngs[1:], support_pngs[0]),
            proposer,
            observations,
            persist_and_reload=None,
            query_callback=None,
        )


def test_freeze_byte_change_fails_before_query_callback() -> None:
    task, support_pngs, proposer, observations = _fixture()
    query_calls = 0

    def query(_pair):
        nonlocal query_calls
        query_calls += 1
        raise AssertionError("query ran after a bad freeze reload")

    with pytest.raises(PanelFeatureTaskRunnerError, match="bytes changed"):
        run_panel_feature_task(
            task,
            support_pngs,
            proposer,
            observations,
            persist_and_reload=lambda value: value + b"\n",
            query_callback=query,
        )
    assert query_calls == 0


def test_query_requires_same_vocabulary_protocol_and_exact_two_sides() -> None:
    task, support_pngs, proposer, observations = _fixture()
    query_panel = _png(200)

    def bad_query(_pair):
        return {
            task.side_0_query_panel_id: (
                query_panel,
                _observation(query_panel, 1, protocol="d" * 64),
            ),
            task.side_1_query_panel_id: (
                _png(201),
                _observation(_png(201), 5),
            ),
        }

    with pytest.raises(PanelFeatureTaskRunnerError, match="protocol"):
        run_panel_feature_task(
            task,
            support_pngs,
            proposer,
            observations,
            persist_and_reload=lambda value: value,
            query_callback=bad_query,
        )


def test_disposition_mapping_is_total_and_never_turns_uncertainty_negative() -> None:
    assert engineering_disposition_from_observation(
        EngineeringFeatureDisposition.MATCH
    ) is EngineeringDisposition.MATCH
    assert engineering_disposition_from_observation(
        EngineeringFeatureDisposition.NONMATCH
    ) is EngineeringDisposition.NONMATCH
    assert engineering_disposition_from_observation(
        EngineeringFeatureDisposition.INDETERMINATE
    ) is EngineeringDisposition.INDETERMINATE
    assert engineering_disposition_from_observation(
        EngineeringFeatureDisposition.ERROR
    ) is EngineeringDisposition.ERROR
    with pytest.raises(TypeError):
        engineering_disposition_from_observation(
            EngineeringDisposition.NONMATCH  # type: ignore[arg-type]
        )


def test_live_support_adapter_uses_neutral_names_and_defers_query_until_reload() -> None:
    task, support_pngs, proposer, _ = _fixture()
    observed_names: list[str] = []
    events: list[str] = []

    def propose(panels: tuple[bytes, ...], task_context_digest: str):
        assert panels == support_pngs
        assert task_context_digest == task.record_digest.split(":", 1)[1]
        return proposer

    def observe(name: str, panel: bytes, specs: tuple[PanelFeatureSpec, ...]):
        observed_names.append(name)
        assert specs == proposer.observer_vocabulary.specs
        index = support_pngs.index(panel)
        return _observation(panel, 1 if index < 6 else 5)

    def persist(value: bytes) -> bytes:
        events.append("persist")
        return value

    def query(_pair):
        events.append("query")
        panels = (_png(300), _png(301))
        return {
            task.side_0_query_panel_id: (
                panels[0],
                _observation(panels[0], 1),
            ),
            task.side_1_query_panel_id: (
                panels[1],
                _observation(panels[1], 5),
            ),
        }

    archive = run_panel_feature_task_with_support_callbacks(
        task,
        support_pngs,
        proposer_callback=propose,
        observation_callback=observe,
        persist_and_reload=persist,
        query_callback=query,
    )
    assert observed_names == list(PANEL_FEATURE_PRESENTATION_NAMES)
    assert all("side" not in name for name in observed_names)
    assert events == ["persist", "query"]
    assert archive.status is PanelFeatureTaskRunStatus.COMPLETE


def test_archive_policy_is_python_engineering_only_and_tamper_evident() -> None:
    task, support_pngs, proposer, observations = _fixture(incomplete=True)
    archive = run_panel_feature_task(
        task,
        support_pngs,
        proposer,
        observations,
        persist_and_reload=None,
        query_callback=None,
    )
    rendered = json.dumps(archive.to_data(), sort_keys=True).lower()
    assert '"implementation_language": "python"' in rendered
    assert '"engineering_only": true' in rendered
    assert '"uncalibrated": true' in rendered
    assert '"lean_present": false' in rendered
    assert '"lean_required": false' in rendered

    tampered = deepcopy(archive.to_data())
    tampered["cold_replay_model_calls"] = 1
    with pytest.raises(PanelFeatureTaskRunnerError):
        PanelFeatureTaskArchive.from_data(tampered)

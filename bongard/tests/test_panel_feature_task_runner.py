from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.object_bongard_release_gate import (
    ObjectBongardReleaseStore,
    ObjectBongardTaskCommitProtocol,
    ObjectBongardTaskFreezeProtocol,
    persist_object_bongard_task_commit,
    persist_object_bongard_task_freeze,
)
from bongard.official_panel_archive import (
    OfficialPanelReceipt,
    ReleasedOfficialPanel,
)
from bongard.panel_feature_observation import (
    BindingFeatureObservation,
    BindingResolution,
    EligibleDomainGap,
    EngineeringFeatureDisposition,
    FeatureAxis,
    ObservationIssue,
    PanelAxisObservation,
    PanelFeatureObservationSet,
    derive_inventory_count_observation,
    eligible_axis_bindings,
)
from bongard.panel_feature_predicate import (
    EngineeringDisposition,
    EngineeringQueryOutcome,
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
    PanelFeatureTaskFreeze,
    PanelFeatureTaskFreezeCommit,
    PanelFeatureTaskRunStatus,
    PanelFeatureTaskRunnerError,
    cold_replay_panel_feature_task,
    engineering_disposition_from_observation,
    panel_feature_axis_catalog,
    run_panel_feature_task,
    run_panel_feature_task_with_support_callbacks,
)
from bongard.panel_soft_ontology import (
    ClosedCount,
    ComponentCountParameters,
    EnumerationResolution,
    FeatureFamily,
    GestaltKind,
    GestaltResemblanceParameters,
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
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


_TASK_SEED = "sha256:" + "7" * 64
_RECEIPT = "8" * 64
_CONTRACT = "9" * 64
_PROTOCOL = "a" * 64
_PRECOMMIT = "sha256:" + "b" * 64
_EXPOSURE = "sha256:" + "c" * 64
_RELEASE_DESCRIPTOR = "sha256:" + "d" * 64
_ARCHIVE = "sha256:" + "e" * 64
_CENTRAL = "sha256:" + "f" * 64
_RANK = "1" * 64


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


def _gestalt_spec(kind: GestaltKind) -> PanelFeatureSpec:
    return PanelFeatureSpec(
        FeatureFamily.GESTALT_RESEMBLANCE,
        SubjectScope.WHOLE_PANEL,
        ReferenceFrame.NONE,
        GestaltResemblanceParameters(kind),
    )


def _candidate(
    spec: PanelFeatureSpec, *, native_block: str, suffix: str
) -> dict[str, object]:
    result: dict[str, object] = {
        "candidate_kind": "registered_feature",
        **panel_feature_spec_to_wire(spec),
        "language_gap_kind": PANEL_FEATURE_NONE,
        "archival_summary": f"A visible registered feature {suffix}",
        "archival_indicator_a": f"Complete closed witness {suffix}",
        "archival_indicator_b": f"Panel-local evidence {suffix}",
    }
    for block in PANEL_FEATURE_BLOCKS:
        for index in range(6):
            result[f"{block}_panel_{index:03d}_estimate"] = (
                "supports" if block == native_block else "does_not_support"
            )
    return result


def _payload(*, multiple: bool = False) -> dict[str, object]:
    result: dict[str, object] = {}
    for block_index, block in enumerate(PANEL_FEATURE_BLOCKS):
        count = ClosedCount.ONE if block_index == 0 else ClosedCount.FIVE
        gestalt = GestaltKind.BIRD_LIKE if block_index == 0 else GestaltKind.ANIMAL_LIKE
        for slot in range(PANEL_FEATURE_SLOTS_PER_DIRECTION):
            spec = (
                _gestalt_spec(gestalt)
                if multiple and slot % 2
                else _count_spec(count)
            )
            result[f"{block}_candidate_{slot}"] = _candidate(
                spec,
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


def _proposer(
    task: ObjectBongardTaskPlan,
    pngs: tuple[bytes, ...],
    *,
    multiple: bool = False,
):
    return parse_panel_feature_proposer_payload(
        _payload(multiple=multiple),
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
        "2" * 64,
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


def _unclear_axis(
    inventory: OwnerInventory,
    axis: FeatureAxis,
    *,
    observed_gestalt: GestaltKind | None,
) -> PanelAxisObservation:
    bindings = eligible_axis_bindings(axis, inventory)
    if not bindings:
        return PanelAxisObservation(
            inventory,
            axis,
            _CONTRACT,
            _PROTOCOL,
            (),
            EligibleDomainGap.unverified_empty(inventory, axis),
        )
    rows: list[BindingFeatureObservation] = []
    for binding in bindings:
        if (
            axis.family is FeatureFamily.GESTALT_RESEMBLANCE
            and axis.subject_scope is SubjectScope.WHOLE_PANEL
        ):
            specs = (
                ()
                if observed_gestalt is None
                else (_gestalt_spec(observed_gestalt),)
            )
            points = () if not specs else (QuantizedPoint(8, 8),)
            rows.append(
                BindingFeatureObservation(
                    axis.axis_digest,
                    binding,
                    BindingResolution.COMPLETE,
                    specs,
                    points,
                    None,
                    canonical_digest(
                        {"axis": axis.axis_digest, "binding": binding.binding_digest}
                    ),
                )
            )
        else:
            rows.append(
                BindingFeatureObservation(
                    axis.axis_digest,
                    binding,
                    BindingResolution.UNCLEAR,
                    (),
                    (),
                    ObservationIssue.AMBIGUOUS_GEOMETRY,
                    canonical_digest(
                        {"axis": axis.axis_digest, "binding": binding.binding_digest}
                    ),
                )
            )
    return PanelAxisObservation(
        inventory,
        axis,
        _CONTRACT,
        _PROTOCOL,
        tuple(rows),
    )


def _observation(
    panel: bytes,
    count: int,
    *,
    complete: bool = True,
    contract: str = _CONTRACT,
    protocol: str = _PROTOCOL,
    observed_gestalt: GestaltKind | None = None,
) -> PanelFeatureObservationSet:
    inventory = _inventory(panel, count, complete=complete)
    rows: list[PanelAxisObservation] = []
    for axis in panel_feature_axis_catalog():
        if axis.family in {
            FeatureFamily.COMPONENT_COUNT,
            FeatureFamily.EXACT_SEGMENT_COUNT,
        }:
            row = derive_inventory_count_observation(
                inventory,
                axis,
                observer_contract_digest=contract,
                measurement_protocol_digest=protocol,
            )
        else:
            row = _unclear_axis(
                inventory, axis, observed_gestalt=observed_gestalt
            )
            if contract != _CONTRACT or protocol != _PROTOCOL:
                row = PanelAxisObservation(
                    row.inventory,
                    row.axis,
                    contract,
                    protocol,
                    row.binding_observations,
                    row.domain_gap,
                )
        rows.append(row)
    return PanelFeatureObservationSet(
        inventory.panel_digest, contract, protocol, tuple(rows)
    )


def _released_panel(panel_id: str, panel: bytes) -> ReleasedOfficialPanel:
    family, tail = panel_id.split("/", 1)
    member = f"ShapeBongard_V2/{family}/images/{tail}"
    receipt = OfficialPanelReceipt.seal(
        panel_id=panel_id,
        payload=panel,
        archive_member=member,
        zip_crc32=123,
        release_descriptor_digest=_RELEASE_DESCRIPTOR,
        archive_digest=_ARCHIVE,
        central_directory_digest=_CENTRAL,
    )
    content = {
        "schema": "gkm.bongard-released-panel.v1",
        "panel_id": panel_id,
        "exact_png_base64": base64.b64encode(panel).decode("ascii"),
        "exact_png_digest": receipt.sha256,
        "release_receipt": receipt.to_data(),
        "execution_precommit_digest": _PRECOMMIT,
        "exposure_successor_digest": _EXPOSURE,
        "released_after_durable_exposure": True,
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_defines_identity_or_decision": False,
        "lean_required_for_replay": False,
        "lean_removal_changes_decision": False,
    }
    return ReleasedOfficialPanel(
        panel_id,
        panel,
        receipt.sha256,
        receipt,
        _PRECOMMIT,
        _EXPOSURE,
        "sha256:" + canonical_digest(content),
    )


def _release_row(
    store: ObjectBongardReleaseStore,
    panel_id: str,
    panel: bytes,
    *,
    kind: str,
):
    released = _released_panel(panel_id, panel)
    receipt = store.persist(
        object_kind=kind,
        object_digest=released.record_digest,
        data=released.to_data(),
    )
    return released, receipt


def _fixture(tmp_path: Path, *, incomplete: bool = False, multiple: bool = False):
    task = _task()
    support_pngs = tuple(_png(index) for index in range(12))
    proposer = _proposer(task, support_pngs, multiple=multiple)
    observations = tuple(
        _observation(
            panel,
            1 if index < 6 else 5,
            complete=not incomplete,
            observed_gestalt=(
                GestaltKind.BIRD_LIKE if index < 6 else GestaltKind.ANIMAL_LIKE
            )
            if multiple
            else None,
        )
        for index, panel in enumerate(support_pngs)
    )
    store = ObjectBongardReleaseStore((tmp_path / "store").absolute())
    releases = tuple(
        _release_row(
            store,
            panel_id,
            panel,
            kind="released-support-panel",
        )
        for panel_id, panel in zip(
            (*task.side_0_support_panel_ids, *task.side_1_support_panel_ids),
            support_pngs,
            strict=True,
        )
    )
    return task, support_pngs, proposer, observations, store, releases


def _freeze_callback(store: ObjectBongardReleaseStore, events: list[str]):
    def callback(freeze: PanelFeatureTaskFreeze):
        events.append("freeze")
        freeze_receipt = persist_object_bongard_task_freeze(
            store=store, freeze=freeze
        )
        commit = PanelFeatureTaskFreezeCommit.seal(freeze, freeze_receipt)
        commit_receipt = persist_object_bongard_task_commit(
            store=store, commit=commit
        )
        return commit, freeze_receipt, commit_receipt

    return callback


def _query_callbacks(
    task: ObjectBongardTaskPlan,
    store: ObjectBongardReleaseStore,
    events: list[str],
):
    panels = (_png(100), _png(101))

    def release():
        events.append("release")
        return {
            task.side_0_query_panel_id: _release_row(
                store,
                task.side_0_query_panel_id,
                panels[0],
                kind="released-query-panel",
            ),
            task.side_1_query_panel_id: _release_row(
                store,
                task.side_1_query_panel_id,
                panels[1],
                kind="released-query-panel",
            ),
        }

    def observe(token: str, panel: bytes, axes: tuple[FeatureAxis, ...]):
        events.append("observe")
        assert token.startswith("panel_")
        assert axes == panel_feature_axis_catalog()
        assert all(type(item) is FeatureAxis for item in axes)
        return _observation(panel, 1 if panel == panels[0] else 5)

    return release, observe


def _run_complete(tmp_path: Path):
    task, support_pngs, proposer, observations, store, releases = _fixture(tmp_path)
    events: list[str] = []
    release, observe = _query_callbacks(task, store, events)
    archive = run_panel_feature_task(
        task,
        releases,
        proposer,
        observations,
        execution_precommit_digest=_PRECOMMIT,
        exposure_successor_digest=_EXPOSURE,
        rank_response_digest=_RANK,
        freeze_persist_reload=_freeze_callback(store, events),
        query_release_callback=release,
        query_observation_callback=observe,
    )
    return archive, events, task, support_pngs, store, releases


def test_complete_run_freezes_before_separate_query_release_and_observation(
    tmp_path: Path,
) -> None:
    archive, events, _, _, _, _ = _run_complete(tmp_path)
    assert events == ["freeze", "release", "observe", "observe"]
    assert archive.status is PanelFeatureTaskRunStatus.COMPLETE
    assert archive.freeze_persist_reload_invocations == 1
    assert archive.query_release_invocations == 1
    assert archive.query_observer_invocations == 2
    assert isinstance(archive.task_freeze, ObjectBongardTaskFreezeProtocol)
    assert isinstance(archive.task_freeze_commit, ObjectBongardTaskCommitProtocol)
    assert tuple(item.outcome for item in archive.query_decisions) == (
        EngineeringQueryOutcome.SIDE0,
        EngineeringQueryOutcome.SIDE1,
    )
    assert PanelFeatureTaskArchive.from_data(archive.to_data()) == archive
    assert cold_replay_panel_feature_task(
        archive, expected_archive_address=archive.archive_address
    ) == archive


def test_empty_version_space_returns_gap_without_callbacks(tmp_path: Path) -> None:
    task, _, proposer, observations, _, releases = _fixture(
        tmp_path, incomplete=True
    )

    def forbidden(*_args):
        raise AssertionError("gap invoked a forbidden callback")

    archive = run_panel_feature_task(
        task,
        releases,
        proposer,
        observations,
        execution_precommit_digest=_PRECOMMIT,
        exposure_successor_digest=_EXPOSURE,
        rank_response_digest=_RANK,
        freeze_persist_reload=forbidden,
        query_release_callback=forbidden,
        query_observation_callback=forbidden,
    )
    assert archive.status is PanelFeatureTaskRunStatus.SUPPORT_GAP
    assert archive.support_gap is not None
    assert archive.freeze_persist_reload_invocations == 0
    assert archive.query_release_invocations == 0
    assert archive.query_observer_invocations == 0


def test_multiple_survivors_are_selection_gap_not_digest_order_choice(
    tmp_path: Path,
) -> None:
    task, _, proposer, observations, _, releases = _fixture(
        tmp_path, multiple=True
    )

    def forbidden(*_args):
        raise AssertionError("selection gap invoked a forbidden callback")

    archive = run_panel_feature_task(
        task,
        releases,
        proposer,
        observations,
        execution_precommit_digest=_PRECOMMIT,
        exposure_successor_digest=_EXPOSURE,
        rank_response_digest=_RANK,
        freeze_persist_reload=forbidden,
        query_release_callback=forbidden,
        query_observation_callback=forbidden,
    )
    assert archive.status is PanelFeatureTaskRunStatus.SELECTION_GAP
    assert archive.selection_gap is not None
    assert archive.selection_gap.survivor_counts_by_orientation[0] > 1
    assert archive.predicate_pair is None


def test_support_observer_gets_no_candidate_specs_labels_or_positions(
    tmp_path: Path,
) -> None:
    task, support_pngs, proposer, _, store, releases = _fixture(tmp_path)
    calls: list[tuple[str, tuple[FeatureAxis, ...]]] = []
    events: list[str] = []

    def propose(panels: tuple[bytes, ...], task_digest: str):
        assert panels == support_pngs
        assert task_digest == task.record_digest.split(":", 1)[1]
        return proposer

    def observe(token: str, panel: bytes, axes: tuple[FeatureAxis, ...]):
        calls.append((token, axes))
        index = support_pngs.index(panel)
        return _observation(panel, 1 if index < 6 else 5)

    release, query_observer = _query_callbacks(task, store, events)
    archive = run_panel_feature_task_with_support_callbacks(
        task,
        releases,
        proposer_callback=propose,
        observation_callback=observe,
        execution_precommit_digest=_PRECOMMIT,
        exposure_successor_digest=_EXPOSURE,
        rank_response_digest=_RANK,
        freeze_persist_reload=_freeze_callback(store, events),
        query_release_callback=release,
        query_observation_callback=query_observer,
    )
    assert archive.status is PanelFeatureTaskRunStatus.COMPLETE
    assert [item[0] for item in calls] == [f"panel_{index:03d}" for index in range(12)]
    assert all("side" not in token and "block" not in token for token, _ in calls)
    assert all(axes == panel_feature_axis_catalog() for _, axes in calls)
    assert all(not isinstance(axis, PanelFeatureSpec) for _, axes in calls for axis in axes)


def test_query_observer_api_never_receives_frozen_formula(tmp_path: Path) -> None:
    task, _, proposer, observations, store, releases = _fixture(tmp_path)
    events: list[str] = []
    release, _ = _query_callbacks(task, store, events)
    observed_argument_types: list[tuple[type, type, type]] = []

    def observer(token: str, panel: bytes, axes: tuple[FeatureAxis, ...]):
        observed_argument_types.append((type(token), type(panel), type(axes)))
        return _observation(panel, 1 if panel == _png(100) else 5)

    archive = run_panel_feature_task(
        task,
        releases,
        proposer,
        observations,
        execution_precommit_digest=_PRECOMMIT,
        exposure_successor_digest=_EXPOSURE,
        rank_response_digest=_RANK,
        freeze_persist_reload=_freeze_callback(store, events),
        query_release_callback=release,
        query_observation_callback=observer,
    )
    assert archive.status is PanelFeatureTaskRunStatus.COMPLETE
    assert observed_argument_types == [(str, bytes, tuple), (str, bytes, tuple)]


def test_wrong_freeze_receipt_rejects_before_query_release(tmp_path: Path) -> None:
    task, _, proposer, observations, store, releases = _fixture(tmp_path)
    query_calls = 0

    def bad_freeze(freeze: PanelFeatureTaskFreeze):
        freeze_receipt = persist_object_bongard_task_freeze(
            store=store, freeze=freeze
        )
        commit = PanelFeatureTaskFreezeCommit.seal(freeze, freeze_receipt)
        commit_receipt = persist_object_bongard_task_commit(
            store=store, commit=commit
        )
        return commit, freeze_receipt, freeze_receipt

    def query_release():
        nonlocal query_calls
        query_calls += 1
        raise AssertionError("query release ran after bad durable freeze")

    with pytest.raises(PanelFeatureTaskRunnerError, match="durable freeze"):
        run_panel_feature_task(
            task,
            releases,
            proposer,
            observations,
            execution_precommit_digest=_PRECOMMIT,
            exposure_successor_digest=_EXPOSURE,
            rank_response_digest=_RANK,
            freeze_persist_reload=bad_freeze,
            query_release_callback=query_release,
            query_observation_callback=lambda *_args: None,  # type: ignore[arg-type]
        )
    assert query_calls == 0


def test_mislabeled_pixels_and_release_receipts_reject(tmp_path: Path) -> None:
    task, _, proposer, observations, store, releases = _fixture(tmp_path)
    swapped = (releases[1], releases[0], *releases[2:])
    with pytest.raises(PanelFeatureTaskRunnerError, match="release identity"):
        run_panel_feature_task(
            task,
            swapped,
            proposer,
            observations,
            execution_precommit_digest=_PRECOMMIT,
            exposure_successor_digest=_EXPOSURE,
            rank_response_digest=_RANK,
            freeze_persist_reload=None,
            query_release_callback=None,
            query_observation_callback=None,
        )

    mismatched_receipt = ((releases[0][0], releases[1][1]), *releases[1:])
    with pytest.raises(PanelFeatureTaskRunnerError, match="durable receipt"):
        run_panel_feature_task(
            task,
            mismatched_receipt,
            proposer,
            observations,
            execution_precommit_digest=_PRECOMMIT,
            exposure_successor_digest=_EXPOSURE,
            rank_response_digest=_RANK,
            freeze_persist_reload=None,
            query_release_callback=None,
            query_observation_callback=None,
        )

    raw = deepcopy(releases[0][0].to_data())
    raw["exact_png_base64"] = base64.b64encode(_png(999)).decode("ascii")
    with pytest.raises(Exception):
        ReleasedOfficialPanel.from_data(raw)


def test_observation_requires_fixed_full_axis_catalog(tmp_path: Path) -> None:
    task, _, proposer, observations, _, releases = _fixture(tmp_path)
    missing = PanelFeatureObservationSet(
        observations[0].panel_digest,
        observations[0].observer_contract_digest,
        observations[0].measurement_protocol_digest,
        observations[0].axis_observations[1:],
    )
    with pytest.raises(PanelFeatureTaskRunnerError, match="fixed full-axis"):
        run_panel_feature_task(
            task,
            releases,
            proposer,
            (missing, *observations[1:]),
            execution_precommit_digest=_PRECOMMIT,
            exposure_successor_digest=_EXPOSURE,
            rank_response_digest=_RANK,
            freeze_persist_reload=None,
            query_release_callback=None,
            query_observation_callback=None,
        )


def test_proposer_provenance_and_disposition_mapping_remain_fail_closed(
    tmp_path: Path,
) -> None:
    task, support_pngs, _, observations, _, releases = _fixture(tmp_path)
    wrong_task = parse_panel_feature_proposer_payload(
        _payload(),
        proposer_receipt_digest=_RECEIPT,
        support_set_digest=_presentation_digest(support_pngs),
        task_context_digest="f" * 64,
    )
    with pytest.raises(PanelFeatureTaskRunnerError, match="provenance"):
        run_panel_feature_task(
            task,
            releases,
            wrong_task,
            observations,
            execution_precommit_digest=_PRECOMMIT,
            exposure_successor_digest=_EXPOSURE,
            rank_response_digest=_RANK,
            freeze_persist_reload=None,
            query_release_callback=None,
            query_observation_callback=None,
        )
    assert engineering_disposition_from_observation(
        EngineeringFeatureDisposition.INDETERMINATE
    ) is EngineeringDisposition.INDETERMINATE
    with pytest.raises(TypeError):
        engineering_disposition_from_observation(
            EngineeringDisposition.NONMATCH  # type: ignore[arg-type]
        )


def test_archive_is_python_only_and_tamper_evident(tmp_path: Path) -> None:
    archive, _, _, _, _, _ = _run_complete(tmp_path)
    rendered = json.dumps(archive.to_data(), sort_keys=True).lower()
    assert '"implementation_language": "python"' in rendered
    assert '"lean_required": false' in rendered
    assert '"query_observer_receives_predicate_or_formula": false' in rendered

    tampered = deepcopy(archive.to_data())
    tampered["cold_replay_model_calls"] = 1
    with pytest.raises(PanelFeatureTaskRunnerError):
        PanelFeatureTaskArchive.from_data(tampered)

    assert archive.task_freeze is not None
    exact = canonical_json(archive.task_freeze.to_data()) + b"\n"
    assert archive.task_freeze_store_receipt is not None
    assert archive.task_freeze_store_receipt.payload_digest == (
        "sha256:" + hashlib.sha256(exact).hexdigest()
    )

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any
import zipfile

import pytest

from bongard import panel_program_official_task as official_task
from bongard.canonical import canonical_digest
from bongard.canonical import canonical_json
from bongard.exposure import ExposureLedger
from bongard.object_bongard_batch import (
    FAMILIES,
    object_bongard_task_inventory_digest,
    plan_object_bongard_batch,
)
from bongard.object_bongard_release_gate import (
    ObjectBongardReleaseStore,
    ObjectBongardReleaseGateError,
    PreparedObjectBongardRelease,
    create_object_bongard_execution_precommit,
    persist_object_bongard_task_commit,
    persist_object_bongard_task_freeze,
    prepare_object_bongard_release,
    release_object_bongard_query_panel,
    release_object_bongard_support_panel,
)
from bongard.official_panel_archive import OfficialPanelArchive
from bongard.panel_action_count_connected_synthetic import (
    primitive_catalog,
    render_catalog_program,
)
from bongard.panel_program_observation import (
    PanelProgramObservation,
    observe_authenticated_program_png,
)
from bongard.panel_program_official_task import (
    PanelProgramOfficialSupportRuntime,
    PanelProgramOfficialQueryResult,
    PanelProgramOfficialSupportArtifact,
    PanelProgramOfficialTaskError,
    PanelProgramOfficialTaskCommit,
    PanelProgramOfficialTaskFreeze,
    build_panel_program_official_support,
    commit_panel_program_official_task_decision,
    freeze_panel_program_official_task_decision,
    panel_program_required_precommit_bindings,
    persist_panel_program_official_task_decision,
    release_and_evaluate_panel_program_official_query,
)
from bongard.panel_program_predicate import evaluate_frozen_program_rule
from bongard.release import OfficialReleaseDescriptor


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _contains_bytes(value: object) -> bool:
    if isinstance(value, bytes):
        return True
    if isinstance(value, dict):
        return any(_contains_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_bytes(item) for item in value)
    return False


@dataclass(frozen=True)
class _Fixture:
    archive: OfficialPanelArchive
    prepared: PreparedObjectBongardRelease
    positive_png: bytes
    contrast_png: bytes


def _fixture(tmp_path: Path, *, include_program_bindings: bool = True) -> _Fixture:
    catalog = primitive_catalog()
    line = next(item for item in catalog if item.kind == "line")
    arc = next(item for item in catalog if item.kind == "arc")
    positive_png = render_catalog_program((line.primitive_id,))
    contrast_png = render_catalog_program((arc.primitive_id,))

    inventory = tuple(
        sorted(
            f"{family}_task{index:02d}"
            for family in FAMILIES
            for index in range(3)
        )
    )
    inventory_digest = object_bongard_task_inventory_digest(inventory)
    corpus_digest = _address({"synthetic": "panel-program-official-custody"})
    predecessor = ExposureLedger.create(corpus_digest)
    archive_path = tmp_path / "ShapeBongard_V2.zip"
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as bundle:
        for task_id in inventory:
            family = task_id.split("_", 1)[0]
            for directory, png in (("1", positive_png), ("0", contrast_png)):
                for index in range(7):
                    bundle.writestr(
                        f"ShapeBongard_V2/{family}/images/"
                        f"{task_id}/{directory}/{index}.png",
                        png,
                    )
    archive_bytes = archive_path.read_bytes()
    descriptor = OfficialReleaseDescriptor(
        release_id="ShapeBongard_V2-panel-program-custody-test",
        archive_filename=archive_path.name,
        archive_sha256="sha256:" + hashlib.sha256(archive_bytes).hexdigest(),
        archive_size_bytes=len(archive_bytes),
        split_filename="ShapeBongard_V2_split.json",
        split_sha256=_address({"split": "synthetic-train"}),
        split_size_bytes=1,
        upstream_repository="https://github.com/NVlabs/Bongard-LOGO",
        upstream_commit="2" * 40,
        family_counts=tuple((family, 3) for family in FAMILIES),
        primary_split_counts=(("test", 0), ("train", len(inventory)), ("val", 0)),
        regime_counts=(("BA", 0), ("CM", 0), ("FF", 0), ("NV", 0)),
        task_ids_sha256=inventory_digest,
        corpus_manifest_sha256=corpus_digest,
    )
    plan = plan_object_bongard_batch(
        task_ids=inventory,
        train_task_ids=inventory,
        exact_used_task_ids=tuple(
            sorted(f"{family}_task00" for family in FAMILIES)
        ),
        selection_seed="panel-program-official-custody-test",
        requested_per_family=1,
        release_descriptor_digest=descriptor.digest,
        split_source_digest=descriptor.split_sha256,
        task_inventory_digest=inventory_digest,
        exposure_predecessor_digest=predecessor.digest,
        historical_exposure_digest=_address({"historical": []}),
    )
    archive = OfficialPanelArchive.load(
        descriptor,
        archive_path,
        expected_release_descriptor_digest=descriptor.digest,
    )
    runtime_bindings = {"runner_source": _address({"runner": "synthetic"})}
    if include_program_bindings:
        runtime_bindings.update(panel_program_required_precommit_bindings())
    precommit = create_object_bongard_execution_precommit(
        plan=plan,
        predecessor=predecessor,
        descriptor=descriptor,
        archive=archive,
        task_ids=inventory,
        train_task_ids=inventory,
        exact_used_task_ids=tuple(
            sorted(f"{family}_task00" for family in FAMILIES)
        ),
        runtime_source_bindings=runtime_bindings,
        configuration={"observer": "fixed-catalog-test", "headless": True},
        exposure_observed_at="2026-08-13T08:00:00Z",
    )
    store = ObjectBongardReleaseStore((tmp_path / "release-store").absolute())
    prepared = prepare_object_bongard_release(
        store=store, plan=plan, precommit=precommit, predecessor=predecessor
    )
    return _Fixture(archive, prepared, positive_png, contrast_png)


def _release_support(
    fixture: _Fixture, task: object
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    released = []
    receipts = []
    for panel_id in (
        *task.side_0_support_panel_ids,  # type: ignore[attr-defined]
        *task.side_1_support_panel_ids,  # type: ignore[attr-defined]
    ):
        panel, receipt = release_object_bongard_support_panel(
            prepared=fixture.prepared, archive=fixture.archive, panel_id=panel_id
        )
        released.append(panel)
        receipts.append(receipt)
    return tuple(released), tuple(receipts)


def _readdress(data: dict[str, Any], field: str) -> dict[str, Any]:
    content = deepcopy(data)
    content.pop(field, None)
    data[field] = _address(content)
    return data


def _persist_direct_decision(
    fixture: _Fixture, freeze: PanelProgramOfficialTaskFreeze
) -> tuple[PanelProgramOfficialTaskCommit, object, object]:
    freeze_receipt = persist_object_bongard_task_freeze(
        store=fixture.prepared.store, freeze=freeze
    )
    exact_payload = canonical_json(freeze.to_data()) + b"\n"
    commit = commit_panel_program_official_task_decision(
        freeze=freeze,
        exact_freeze_payload=exact_payload,
        task_freeze_store_receipt=freeze_receipt,
    )
    commit_receipt = persist_object_bongard_task_commit(
        store=fixture.prepared.store, commit=commit
    )
    return commit, freeze_receipt, commit_receipt


def test_exact_support_freeze_commit_then_query_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    task = fixture.prepared.plan.tasks[0]
    support_ids = (
        *task.side_0_support_panel_ids,
        *task.side_1_support_panel_ids,
    )
    released_support = []
    support_store_receipts = []
    for panel_id in support_ids:
        released, receipt = release_object_bongard_support_panel(
            prepared=fixture.prepared,
            archive=fixture.archive,
            panel_id=panel_id,
        )
        released_support.append(released)
        support_store_receipts.append(receipt)

    support = build_panel_program_official_support(
        task=task,
        prepared=fixture.prepared,
        released_panels=tuple(released_support),
        released_panel_store_receipts=tuple(support_store_receipts),
        observe_program=observe_authenticated_program_png,
    )
    assert all(
        receipt.object_kind == "released-support-panel"
        for receipt in support_store_receipts
    )
    assert not _contains_bytes(support.artifact.to_data())
    assert (
        PanelProgramOfficialSupportArtifact.from_data(support.artifact.to_data())
        == support.artifact
    )
    stale_algorithm = support.artifact.to_data()
    stale_algorithm["adapter_algorithm_digest"] = "sha256:" + "f" * 64
    with pytest.raises(
        PanelProgramOfficialTaskError,
        match="support artifact inventory differs",
    ):
        PanelProgramOfficialSupportArtifact.from_data(
            _readdress(stale_algorithm, "record_digest")
        )

    freeze = freeze_panel_program_official_task_decision(support=support)
    assert PanelProgramOfficialTaskFreeze.from_data(freeze.to_data()) == freeze
    assert freeze.support_panel_ids == support_ids
    assert freeze.sealed_query_panel_ids == (
        task.side_0_query_panel_id,
        task.side_1_query_panel_id,
    )
    assert freeze.selected_rule.formula.to_data() == {
        "schema": "gkm.panel-program-count-rule-language.v1",
        "operator": "all_of",
        "atoms": [
            {
                "axis": "arc_count",
                "expected": 0,
                "atom_digest": (
                    "sha256:042d0670ae003f4c9a4ecb944f1ec9d02ad9c7655a73d5d9f0f03bc3d362ac9f"
                ),
            }
        ],
        "positive_only": True,
        "formula_digest": (
            "sha256:3f0d56997717bd1175c80ac085253cd6656892c83eb199549bdb2cc63569825c"
        ),
    }
    assert not _contains_bytes(freeze.to_data())

    archive_reads: list[str] = []
    original_read = OfficialPanelArchive.read_panel

    def tracked_read(archive: OfficialPanelArchive, panel_id: str):
        archive_reads.append(panel_id)
        return original_read(archive, panel_id)

    monkeypatch.setattr(OfficialPanelArchive, "read_panel", tracked_read)
    with pytest.raises(TypeError, match="durable_decision"):
        release_and_evaluate_panel_program_official_query(
            task=task,
            prepared=fixture.prepared,
            archive=fixture.archive,
            panel_id=task.side_0_query_panel_id,
            durable_decision=object(),  # type: ignore[arg-type]
            observe_program=observe_authenticated_program_png,
        )
    assert archive_reads == []

    durable = persist_panel_program_official_task_decision(
        prepared=fixture.prepared, support_runtime=support, freeze=freeze
    )
    assert PanelProgramOfficialTaskCommit.from_data(
        durable.commit.to_data()
    ) == durable.commit
    assert durable.freeze_receipt.object_kind == "task-freeze"
    assert durable.commit_receipt.object_kind == "task-decision-commit"
    assert archive_reads == []

    query = release_and_evaluate_panel_program_official_query(
        task=task,
        prepared=fixture.prepared,
        archive=fixture.archive,
        panel_id=task.side_0_query_panel_id,
        durable_decision=durable,
        observe_program=observe_authenticated_program_png,
    )
    assert archive_reads == [task.side_0_query_panel_id]
    assert query.released_panel.exact_png_bytes == fixture.positive_png
    assert query.result.decision == evaluate_frozen_program_rule(
        freeze.selected_rule, query.result.observation
    )
    assert query.result.decision.prediction == "positive"
    assert PanelProgramOfficialQueryResult.from_data(
        query.result.to_data()
    ) == query.result
    assert not _contains_bytes(query.result.to_data())
    assert query.released_panel_store_receipt.object_kind == "released-query-panel"
    assert query.result_store_receipt.object_kind == "panel-program-query-result"

    contrast_query = release_and_evaluate_panel_program_official_query(
        task=task,
        prepared=fixture.prepared,
        archive=fixture.archive,
        panel_id=task.side_1_query_panel_id,
        durable_decision=durable,
        observe_program=observe_authenticated_program_png,
    )
    assert archive_reads == [
        task.side_0_query_panel_id,
        task.side_1_query_panel_id,
    ]
    assert contrast_query.released_panel.exact_png_bytes == fixture.contrast_png
    assert contrast_query.result.decision == evaluate_frozen_program_rule(
        freeze.selected_rule, contrast_query.result.observation
    )
    assert contrast_query.result.decision.prediction == "contrast"
    assert PanelProgramOfficialQueryResult.from_data(
        contrast_query.result.to_data()
    ) == contrast_query.result


def test_observer_and_precommit_must_match_before_any_support_callback(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    task = fixture.prepared.plan.tasks[0]
    released = []
    receipts = []
    for panel_id in (*task.side_0_support_panel_ids, *task.side_1_support_panel_ids):
        panel, receipt = release_object_bongard_support_panel(
            prepared=fixture.prepared, archive=fixture.archive, panel_id=panel_id
        )
        released.append(panel)
        receipts.append(receipt)

    calls = 0

    def counterfeit(raw: bytes) -> PanelProgramObservation:
        nonlocal calls
        calls += 1
        return observe_authenticated_program_png(raw)

    with pytest.raises(PanelProgramOfficialTaskError, match="exact precommitted"):
        build_panel_program_official_support(
            task=task,
            prepared=fixture.prepared,
            released_panels=tuple(released),
            released_panel_store_receipts=tuple(receipts),
            observe_program=counterfeit,
        )
    assert calls == 0


def test_missing_precommit_bindings_reject_before_support_observer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, include_program_bindings=False)
    task = fixture.prepared.plan.tasks[0]
    released, receipts = _release_support(fixture, task)
    calls = 0

    def forbidden(_raw: bytes) -> PanelProgramObservation:
        nonlocal calls
        calls += 1
        raise AssertionError("observer ran before its precommit was checked")

    monkeypatch.setattr(
        official_task, "observe_authenticated_program_png", forbidden
    )
    with pytest.raises(PanelProgramOfficialTaskError, match="not frozen"):
        build_panel_program_official_support(
            task=task,
            prepared=fixture.prepared,
            released_panels=released,  # type: ignore[arg-type]
            released_panel_store_receipts=receipts,  # type: ignore[arg-type]
            observe_program=forbidden,
        )
    assert calls == 0


def test_generic_gate_rejects_cross_task_support_transplant_before_query_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    target_task, donor_task = fixture.prepared.plan.tasks[:2]
    released, receipts = _release_support(fixture, donor_task)
    donor = build_panel_program_official_support(
        task=donor_task,
        prepared=fixture.prepared,
        released_panels=released,  # type: ignore[arg-type]
        released_panel_store_receipts=receipts,  # type: ignore[arg-type]
        observe_program=observe_authenticated_program_png,
    )
    forged_data = donor.artifact.to_data()
    forged_data["task_id"] = target_task.task_id
    forged_data["task_plan_digest"] = target_task.record_digest
    forged_data["sealed_query_panel_ids"] = [
        target_task.side_0_query_panel_id,
        target_task.side_1_query_panel_id,
    ]
    forged_artifact = PanelProgramOfficialSupportArtifact.from_data(
        _readdress(forged_data, "record_digest")
    )
    forged_runtime = PanelProgramOfficialSupportRuntime(
        forged_artifact, donor.released_panels
    )
    freeze = freeze_panel_program_official_task_decision(support=forged_runtime)
    commit, freeze_receipt, commit_receipt = _persist_direct_decision(
        fixture, freeze
    )
    reads: list[str] = []
    original = OfficialPanelArchive.read_panel

    def tracked(archive: OfficialPanelArchive, panel_id: str):
        reads.append(panel_id)
        return original(archive, panel_id)

    monkeypatch.setattr(OfficialPanelArchive, "read_panel", tracked)
    with pytest.raises(
        ObjectBongardReleaseGateError,
        match="panel-program support release authority differs",
    ):
        release_object_bongard_query_panel(
            prepared=fixture.prepared,
            archive=fixture.archive,
            panel_id=target_task.side_0_query_panel_id,
            task_freeze=freeze,
            task_commit=commit,
            task_freeze_receipt=freeze_receipt,  # type: ignore[arg-type]
            task_commit_receipt=commit_receipt,  # type: ignore[arg-type]
        )
    assert reads == []


def test_generic_gate_cold_rejects_forged_support_semantics_before_query_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    task = fixture.prepared.plan.tasks[0]
    released, receipts = _release_support(fixture, task)
    support = build_panel_program_official_support(
        task=task,
        prepared=fixture.prepared,
        released_panels=released,  # type: ignore[arg-type]
        released_panel_store_receipts=receipts,  # type: ignore[arg-type]
        observe_program=observe_authenticated_program_png,
    )
    forged_data = support.artifact.to_data()
    fake_source = "sha256:" + "f" * 64
    for panel in forged_data["support_panels"]:
        observation = panel["observation"]
        observation["observer_source_digest"] = fake_source
        _readdress(observation, "observation_digest")
        panel["observation_digest"] = observation["observation_digest"]
        _readdress(panel, "record_digest")
    forged_artifact = PanelProgramOfficialSupportArtifact.from_data(
        _readdress(forged_data, "record_digest")
    )
    forged_runtime = PanelProgramOfficialSupportRuntime(
        forged_artifact, support.released_panels
    )
    freeze = freeze_panel_program_official_task_decision(support=forged_runtime)
    commit, freeze_receipt, commit_receipt = _persist_direct_decision(
        fixture, freeze
    )
    reads: list[str] = []
    original = OfficialPanelArchive.read_panel

    def tracked(archive: OfficialPanelArchive, panel_id: str):
        reads.append(panel_id)
        return original(archive, panel_id)

    monkeypatch.setattr(OfficialPanelArchive, "read_panel", tracked)
    with pytest.raises(
        ObjectBongardReleaseGateError,
        match="panel-program support release authority differs",
    ):
        release_object_bongard_query_panel(
            prepared=fixture.prepared,
            archive=fixture.archive,
            panel_id=task.side_0_query_panel_id,
            task_freeze=freeze,
            task_commit=commit,
            task_freeze_receipt=freeze_receipt,  # type: ignore[arg-type]
            task_commit_receipt=commit_receipt,  # type: ignore[arg-type]
        )
    assert reads == []

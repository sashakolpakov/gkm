from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Any, Mapping
import zipfile

import pytest

from bongard.canonical import canonical_digest
from bongard.exposure import ExposureLedger
from bongard.object_bongard_batch import (
    FAMILIES,
    object_bongard_task_inventory_digest,
    plan_object_bongard_batch,
)
from bongard.object_bongard_drill_batch import (
    object_bongard_drill_batch_algorithm_digest,
    object_bongard_drill_batch_source_digest,
    plan_object_bongard_drill_batch,
)
from bongard.object_bongard_release_gate import (
    ObjectBongardExecutionPrecommit,
    ObjectBongardReleaseAuthorization,
    ObjectBongardReleaseGateError,
    ObjectBongardReleaseStore,
    create_object_bongard_execution_precommit,
    persist_object_bongard_task_commit,
    persist_object_bongard_task_freeze,
    prepare_object_bongard_release,
    release_object_bongard_query_panel,
    release_object_bongard_support_panel,
    verify_prepared_object_bongard_release,
)
from bongard.official_panel_archive import (
    OfficialPanelArchive,
    OfficialPanelArchiveError,
)
from bongard.release import OfficialReleaseDescriptor


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _record_data(content: Mapping[str, Any]) -> dict[str, Any]:
    return {**content, "record_digest": _address(content)}


@dataclass(frozen=True)
class _Freeze:
    task_id: str
    task_plan_digest: str
    execution_precommit_digest: str
    version_space_digest: str
    support_version_space_digest: str
    rank_response_digest: str
    selected_predicate_digest: str
    record_digest: str

    def to_data(self) -> Mapping[str, Any]:
        return {
            "schema": "test.object-task-freeze.v1",
            "task_id": self.task_id,
            "task_plan_digest": self.task_plan_digest,
            "execution_precommit_digest": self.execution_precommit_digest,
            "version_space_digest": self.version_space_digest,
            "support_version_space_digest": self.support_version_space_digest,
            "rank_response_digest": self.rank_response_digest,
            "selected_predicate_digest": self.selected_predicate_digest,
            "record_digest": self.record_digest,
        }


@dataclass(frozen=True)
class _Commit(_Freeze):
    task_freeze_digest: str
    exact_freeze_payload_digest: str
    task_freeze_store_receipt_digest: str

    def to_data(self) -> Mapping[str, Any]:
        return {
            "schema": "test.object-task-commit.v1",
            "task_id": self.task_id,
            "task_plan_digest": self.task_plan_digest,
            "execution_precommit_digest": self.execution_precommit_digest,
            "version_space_digest": self.version_space_digest,
            "support_version_space_digest": self.support_version_space_digest,
            "rank_response_digest": self.rank_response_digest,
            "selected_predicate_digest": self.selected_predicate_digest,
            "task_freeze_digest": self.task_freeze_digest,
            "exact_freeze_payload_digest": self.exact_freeze_payload_digest,
            "task_freeze_store_receipt_digest": self.task_freeze_store_receipt_digest,
            "record_digest": self.record_digest,
        }


def _freeze(task_id: str, task_plan_digest: str, precommit_digest: str) -> _Freeze:
    content = {
        "schema": "test.object-task-freeze.v1",
        "task_id": task_id,
        "task_plan_digest": task_plan_digest,
        "execution_precommit_digest": precommit_digest,
        "version_space_digest": "a" * 64,
        "support_version_space_digest": "a" * 64,
        "rank_response_digest": "b" * 64,
        "selected_predicate_digest": "c" * 64,
    }
    return _Freeze(**{key: value for key, value in content.items() if key != "schema"}, record_digest=_address(content))


def _commit(freeze: _Freeze, freeze_payload: str, freeze_receipt: str) -> _Commit:
    content = {
        "schema": "test.object-task-commit.v1",
        "task_id": freeze.task_id,
        "task_plan_digest": freeze.task_plan_digest,
        "execution_precommit_digest": freeze.execution_precommit_digest,
        "version_space_digest": freeze.version_space_digest,
        "support_version_space_digest": freeze.support_version_space_digest,
        "rank_response_digest": freeze.rank_response_digest,
        "selected_predicate_digest": freeze.selected_predicate_digest,
        "task_freeze_digest": freeze.record_digest,
        "exact_freeze_payload_digest": freeze_payload,
        "task_freeze_store_receipt_digest": freeze_receipt,
    }
    return _Commit(**{key: value for key, value in content.items() if key != "schema"}, record_digest=_address(content))


@dataclass(frozen=True)
class _Fixture:
    descriptor: OfficialReleaseDescriptor
    archive: OfficialPanelArchive
    predecessor: ExposureLedger
    plan: Any
    precommit: Any
    store: ObjectBongardReleaseStore


def _fixture(tmp_path: Path) -> _Fixture:
    inventory = tuple(sorted(f"{family}_task{index:02d}" for family in FAMILIES for index in range(3)))
    train = inventory
    used = tuple(sorted(f"{family}_task00" for family in FAMILIES))
    inventory_digest = object_bongard_task_inventory_digest(inventory)
    corpus_digest = _address({"synthetic": "corpus"})
    predecessor = ExposureLedger.create(corpus_digest)
    archive_path = tmp_path / "ShapeBongard_V2.zip"
    png = b"\x89PNG\r\n\x1a\nsynthetic-bounded-panel"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for task_id in inventory:
            family = task_id.split("_", 1)[0]
            for side in ("0", "1"):
                for index in range(7):
                    bundle.writestr(
                        f"ShapeBongard_V2/{family}/images/{task_id}/{side}/{index}.png",
                        png,
                    )
    archive_bytes = archive_path.read_bytes()
    descriptor = OfficialReleaseDescriptor(
        release_id="ShapeBongard_V2-object-release-gate-test",
        archive_filename=archive_path.name,
        archive_sha256="sha256:" + hashlib.sha256(archive_bytes).hexdigest(),
        archive_size_bytes=len(archive_bytes),
        split_filename="ShapeBongard_V2_split.json",
        split_sha256=_address({"split": "train"}),
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
        train_task_ids=train,
        exact_used_task_ids=used,
        selection_seed="release-gate-cross-family-test",
        requested_per_family=1,
        release_descriptor_digest=descriptor.digest,
        split_source_digest=descriptor.split_sha256,
        task_inventory_digest=inventory_digest,
        exposure_predecessor_digest=predecessor.digest,
        historical_exposure_digest=_address({"historical": []}),
    )
    archive = OfficialPanelArchive.load(
        descriptor, archive_path, expected_release_descriptor_digest=descriptor.digest
    )
    precommit = create_object_bongard_execution_precommit(
        plan=plan,
        predecessor=predecessor,
        descriptor=descriptor,
        archive=archive,
        task_ids=inventory,
        train_task_ids=train,
        exact_used_task_ids=used,
        runtime_source_bindings={"runner_source": _address({"runner": 1})},
        configuration={"model": "gpt-5", "minutes": 15, "headless": True},
        exposure_observed_at="2026-08-08T12:00:00Z",
    )
    store = ObjectBongardReleaseStore((tmp_path / "release-store").absolute())
    return _Fixture(descriptor, archive, predecessor, plan, precommit, store)


def test_prepare_records_one_cross_family_exposure_before_any_panel_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    calls: list[str] = []
    original = OfficialPanelArchive.read_panel

    def checked_read(archive: OfficialPanelArchive, panel_id: str):
        calls.append(panel_id)
        assert prepared.exposure_receipt is not None
        assert (fixture.store.root / prepared.exposure_receipt.relative_path).is_file()
        assert prepared.successor.events[-1].task_ids == tuple(
            task.task_id for task in fixture.plan.tasks
        )
        return original(archive, panel_id)

    monkeypatch.setattr(OfficialPanelArchive, "read_panel", checked_read)
    prepared = prepare_object_bongard_release(
        store=fixture.store,
        plan=fixture.plan,
        precommit=fixture.precommit,
        predecessor=fixture.predecessor,
    )
    assert calls == []
    assert len(prepared.successor.events) == 1
    assert {task.family for task in fixture.plan.tasks} == set(FAMILIES)
    assert ObjectBongardExecutionPrecommit.from_data(fixture.precommit.to_data()) == fixture.precommit
    assert ObjectBongardReleaseAuthorization.from_data(prepared.authorization.to_data()) == prepared.authorization
    released, receipt = release_object_bongard_support_panel(
        prepared=prepared,
        archive=fixture.archive,
        panel_id=fixture.plan.tasks[0].side_0_support_panel_ids[0],
    )
    assert calls == [released.panel_id]
    assert receipt.object_kind == "released-support-panel"


def test_execution_precommit_accepts_the_strict_drill_plan_type(
    tmp_path: Path,
) -> None:
    task_ids = tuple(
        sorted(
            (
                "bd_asymm_trap_bridge_0000",
                "bd_asymm_unbala_goldfish-regular_x_0000",
                "bd_inverse_trapez_parallel_0000",
                "bd_symmetric_clamp-irregular_arc_cup_0000",
                "bd_thin_rec_down_right_triangle_0000",
                "bd_three_mismatch_sectors2-mismatch_triangle_rec3_0000",
                "hd_exist_quadrangle-symmetric_transposed_0016",
                "hd_exist_regular-exist_triangle_0014",
                "hd_has_five_straight_lines-thin_shape_0013",
                "hd_has_obtuse_angle-has_line_crossing_0011",
                "hd_has_six_straight_lines-has_acute_angle_0002",
                "hd_unbalanced_two-exist_sector_0012",
            )
        )
    )
    inventory_digest = object_bongard_task_inventory_digest(task_ids)
    corpus_digest = _address({"synthetic": "strict-drill-corpus"})
    predecessor = ExposureLedger.create(corpus_digest)
    archive_path = tmp_path / "ShapeBongard_V2-drill.zip"
    png = b"\x89PNG\r\n\x1a\nsynthetic-strict-drill-panel"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for task_id in task_ids:
            family = task_id.split("_", 1)[0]
            for side in ("0", "1"):
                for index in range(7):
                    bundle.writestr(
                        f"ShapeBongard_V2/{family}/images/{task_id}/{side}/{index}.png",
                        png,
                    )
    archive_bytes = archive_path.read_bytes()
    descriptor = OfficialReleaseDescriptor(
        release_id="ShapeBongard_V2-strict-drill-release-gate-test",
        archive_filename=archive_path.name,
        archive_sha256="sha256:" + hashlib.sha256(archive_bytes).hexdigest(),
        archive_size_bytes=len(archive_bytes),
        split_filename="ShapeBongard_V2_split.json",
        split_sha256=_address({"split": "strict-drill-train"}),
        split_size_bytes=1,
        upstream_repository="https://github.com/NVlabs/Bongard-LOGO",
        upstream_commit="2" * 40,
        family_counts=(("bd", 6), ("ff", 0), ("hd", 6)),
        primary_split_counts=(("test", 0), ("train", 12), ("val", 0)),
        regime_counts=(("BA", 0), ("CM", 0), ("FF", 0), ("NV", 0)),
        task_ids_sha256=inventory_digest,
        corpus_manifest_sha256=corpus_digest,
    )
    plan = plan_object_bongard_drill_batch(
        task_ids=task_ids,
        train_task_ids=task_ids,
        predecessor=predecessor,
        selection_seed="release-gate-strict-drill-test",
        requested_per_family=6,
        release_descriptor_digest=descriptor.digest,
        split_source_digest=descriptor.split_sha256,
        task_inventory_digest=inventory_digest,
    )
    archive = OfficialPanelArchive.load(
        descriptor,
        archive_path,
        expected_release_descriptor_digest=descriptor.digest,
    )
    precommit = create_object_bongard_execution_precommit(
        plan=plan,
        predecessor=predecessor,
        descriptor=descriptor,
        archive=archive,
        task_ids=task_ids,
        train_task_ids=task_ids,
        exact_used_task_ids=(),
        runtime_source_bindings={"runner_source": _address({"runner": 2})},
        configuration={"model": "gpt-5", "minutes": 15, "headless": True},
        exposure_observed_at="2026-08-08T12:00:00Z",
    )
    assert precommit.batch_algorithm_digest == (
        object_bongard_drill_batch_algorithm_digest()
    )
    assert precommit.batch_source_digest == (
        "sha256:" + object_bongard_drill_batch_source_digest()
    )


def test_query_rejects_counterfeit_protocol_pair_before_archive_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    prepared = prepare_object_bongard_release(
        store=fixture.store, plan=fixture.plan, precommit=fixture.precommit,
        predecessor=fixture.predecessor,
    )
    task = fixture.plan.tasks[0]
    freeze = _freeze(task.task_id, task.record_digest, fixture.precommit.record_digest)
    freeze_receipt = persist_object_bongard_task_freeze(store=fixture.store, freeze=freeze)
    commit = _commit(freeze, freeze_receipt.payload_digest, freeze_receipt.record_digest)
    commit_receipt = persist_object_bongard_task_commit(store=fixture.store, commit=commit)

    reads = 0

    def forbidden_read(*args: object, **kwargs: object) -> object:
        nonlocal reads
        reads += 1
        raise AssertionError("counterfeit decision reached archive.read_panel")

    monkeypatch.setattr(OfficialPanelArchive, "read_panel", forbidden_read)
    with pytest.raises(ObjectBongardReleaseGateError, match="exact production pair"):
        release_object_bongard_query_panel(
            prepared=prepared, archive=fixture.archive,
            panel_id=task.side_0_query_panel_id,
            task_freeze=freeze, task_commit=commit,
            task_freeze_receipt=freeze_receipt, task_commit_receipt=commit_receipt,
        )
    assert reads == 0


def test_write_once_store_and_cold_replay_reject_tamper(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    prepared = prepare_object_bongard_release(
        store=fixture.store, plan=fixture.plan, precommit=fixture.precommit,
        predecessor=fixture.predecessor,
    )
    verify_prepared_object_bongard_release(prepared)
    # Idempotent preparation reuses the exact content-addressed bytes.
    replayed = prepare_object_bongard_release(
        store=fixture.store, plan=fixture.plan, precommit=fixture.precommit,
        predecessor=fixture.predecessor,
    )
    assert replayed.authorization == prepared.authorization
    path = fixture.store.root / prepared.precommit_receipt.relative_path
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ObjectBongardReleaseGateError, match="payload differs|collision|tamper"):
        verify_prepared_object_bongard_release(prepared)


def test_exact_unused_and_nonvisual_precommit_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    selected = fixture.plan.tasks[0].task_id
    exposed = fixture.predecessor.record(
        phase="prior", actor="tester", purpose="prior view", task_ids=(selected,),
        observed_at="2026-08-08T11:00:00Z",
    )
    with pytest.raises(ObjectBongardReleaseGateError, match="differ|exact-unused"):
        create_object_bongard_execution_precommit(
            plan=fixture.plan, predecessor=exposed, descriptor=fixture.descriptor,
            archive=fixture.archive,
            task_ids=tuple(sorted(f"{family}_task{index:02d}" for family in FAMILIES for index in range(3))),
            train_task_ids=tuple(sorted(f"{family}_task{index:02d}" for family in FAMILIES for index in range(3))),
            exact_used_task_ids=tuple(sorted(f"{family}_task00" for family in FAMILIES)),
            runtime_source_bindings={"runner_source": _address({"runner": 1})},
            configuration={"model": "gpt-5"}, exposure_observed_at="2026-08-08T12:00:00Z",
        )
    with pytest.raises(ObjectBongardReleaseGateError, match="visual/action"):
        create_object_bongard_execution_precommit(
            plan=fixture.plan, predecessor=fixture.predecessor, descriptor=fixture.descriptor,
            archive=fixture.archive,
            task_ids=tuple(sorted(f"{family}_task{index:02d}" for family in FAMILIES for index in range(3))),
            train_task_ids=tuple(sorted(f"{family}_task{index:02d}" for family in FAMILIES for index in range(3))),
            exact_used_task_ids=tuple(sorted(f"{family}_task00" for family in FAMILIES)),
            runtime_source_bindings={"runner_source": _address({"runner": 1})},
            configuration={"pixel_bytes": "forbidden"}, exposure_observed_at="2026-08-08T12:00:00Z",
        )


def test_release_gates_reject_virtual_store_and_archive_before_any_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    prepared = prepare_object_bongard_release(
        store=fixture.store,
        plan=fixture.plan,
        precommit=fixture.precommit,
        predecessor=fixture.predecessor,
    )
    task = fixture.plan.tasks[0]

    class VirtualStore(ObjectBongardReleaseStore):
        verify_calls = 0

        def verify(self, *args: object, **kwargs: object) -> Mapping[str, Any]:
            self.verify_calls += 1
            raise AssertionError("virtual store verification ran")

    virtual_store = VirtualStore(fixture.store.root)
    virtual_prepared = replace(prepared, store=virtual_store)
    archive_reads = 0

    def forbidden_archive_read(*args: object, **kwargs: object) -> object:
        nonlocal archive_reads
        archive_reads += 1
        raise AssertionError("archive read ran")

    monkeypatch.setattr(OfficialPanelArchive, "read_panel", forbidden_archive_read)
    with pytest.raises(TypeError, match="exact ObjectBongardReleaseStore"):
        release_object_bongard_support_panel(
            prepared=virtual_prepared,
            archive=fixture.archive,
            panel_id=task.side_0_support_panel_ids[0],
        )
    with pytest.raises(TypeError, match="exact ObjectBongardReleaseStore"):
        release_object_bongard_query_panel(
            prepared=virtual_prepared,
            archive=fixture.archive,
            panel_id=task.side_0_query_panel_id,
            task_freeze=object(),  # type: ignore[arg-type]
            task_commit=object(),  # type: ignore[arg-type]
            task_freeze_receipt=object(),  # type: ignore[arg-type]
            task_commit_receipt=object(),  # type: ignore[arg-type]
        )
    assert virtual_store.verify_calls == 0
    assert archive_reads == 0

    class VirtualArchive(OfficialPanelArchive):
        read_calls = 0

        def read_panel(self, panel_id: str) -> object:
            self.read_calls += 1
            raise AssertionError("virtual archive read ran")

    virtual_archive = object.__new__(VirtualArchive)
    for field_name in OfficialPanelArchive.__slots__:
        object.__setattr__(
            virtual_archive,
            field_name,
            getattr(fixture.archive, field_name),
        )
    OfficialPanelArchive.__post_init__(virtual_archive)
    store_verifies = 0
    original_verify = ObjectBongardReleaseStore.verify

    def tracked_store_verify(
        store: ObjectBongardReleaseStore,
        *args: object,
        **kwargs: object,
    ) -> Mapping[str, Any]:
        nonlocal store_verifies
        store_verifies += 1
        return original_verify(store, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ObjectBongardReleaseStore, "verify", tracked_store_verify)
    with pytest.raises(TypeError, match="exact OfficialPanelArchive"):
        release_object_bongard_support_panel(
            prepared=prepared,
            archive=virtual_archive,
            panel_id=task.side_0_support_panel_ids[0],
        )
    with pytest.raises(TypeError, match="exact OfficialPanelArchive"):
        release_object_bongard_query_panel(
            prepared=prepared,
            archive=virtual_archive,
            panel_id=task.side_0_query_panel_id,
            task_freeze=object(),  # type: ignore[arg-type]
            task_commit=object(),  # type: ignore[arg-type]
            task_freeze_receipt=object(),  # type: ignore[arg-type]
            task_commit_receipt=object(),  # type: ignore[arg-type]
        )
    assert virtual_archive.read_calls == 0
    assert store_verifies == 0

    exact_forged_archive = object.__new__(OfficialPanelArchive)
    for field_name in OfficialPanelArchive.__slots__:
        object.__setattr__(
            exact_forged_archive,
            field_name,
            getattr(fixture.archive, field_name),
        )
    object.__setattr__(
        exact_forged_archive,
        "archive_digest",
        "sha256:" + "f" * 64,
    )
    assert type(exact_forged_archive) is OfficialPanelArchive
    with pytest.raises(
        OfficialPanelArchiveError, match="official archive binding differs"
    ):
        release_object_bongard_support_panel(
            prepared=prepared,
            archive=exact_forged_archive,
            panel_id=task.side_0_support_panel_ids[0],
        )
    assert archive_reads == 0


def test_release_gates_reject_virtual_precommit_before_any_store_or_archive_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    prepared = prepare_object_bongard_release(
        store=fixture.store,
        plan=fixture.plan,
        precommit=fixture.precommit,
        predecessor=fixture.predecessor,
    )

    class VirtualPrecommit(ObjectBongardExecutionPrecommit):
        @property
        def runtime_source_bindings(self) -> tuple[tuple[str, str], ...]:
            return (("forged_binding", "sha256:" + "f" * 64),)

    virtual_precommit = object.__new__(VirtualPrecommit)
    for field_name in ObjectBongardExecutionPrecommit.__slots__:
        if field_name != "runtime_source_bindings":
            object.__setattr__(
                virtual_precommit,
                field_name,
                getattr(prepared.precommit, field_name),
            )
    virtual_prepared = replace(prepared, precommit=virtual_precommit)
    store_verifies = 0
    archive_reads = 0
    original_verify = ObjectBongardReleaseStore.verify

    def tracked_store_verify(
        store: ObjectBongardReleaseStore,
        *args: object,
        **kwargs: object,
    ) -> Mapping[str, Any]:
        nonlocal store_verifies
        store_verifies += 1
        return original_verify(store, *args, **kwargs)  # type: ignore[arg-type]

    def forbidden_archive_read(*args: object, **kwargs: object) -> object:
        nonlocal archive_reads
        archive_reads += 1
        raise AssertionError("archive read ran")

    monkeypatch.setattr(ObjectBongardReleaseStore, "verify", tracked_store_verify)
    monkeypatch.setattr(OfficialPanelArchive, "read_panel", forbidden_archive_read)
    task = fixture.plan.tasks[0]
    with pytest.raises(TypeError, match="exact ObjectBongardExecutionPrecommit"):
        release_object_bongard_support_panel(
            prepared=virtual_prepared,
            archive=fixture.archive,
            panel_id=task.side_0_support_panel_ids[0],
        )
    with pytest.raises(TypeError, match="exact ObjectBongardExecutionPrecommit"):
        release_object_bongard_query_panel(
            prepared=virtual_prepared,
            archive=fixture.archive,
            panel_id=task.side_0_query_panel_id,
            task_freeze=object(),  # type: ignore[arg-type]
            task_commit=object(),  # type: ignore[arg-type]
            task_freeze_receipt=object(),  # type: ignore[arg-type]
            task_commit_receipt=object(),  # type: ignore[arg-type]
        )
    assert store_verifies == 0
    assert archive_reads == 0

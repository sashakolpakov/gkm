from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import pytest

from bongard.canonical import canonical_digest
from bongard.corpus import ShapeBongardCorpus, SplitIndex
from bongard.exposure import ExposureLedger
from bongard.object_bongard_batch import object_bongard_task_inventory_digest
from bongard.object_bongard_release_gate import (
    ObjectBongardReleaseStore,
    ObjectBongardWriteOnceReceipt,
    persist_object_bongard_task_commit,
    persist_object_bongard_task_freeze,
)
from bongard.official_extracted_panel_archive import OfficialExtractedPanelArchive
from bongard.panel_feature_extracted_release_gate import (
    PanelFeatureExtractedExecutionPrecommit,
    PanelFeatureExtractedReleaseAuthorization,
    PanelFeatureExtractedReleaseGateError,
    PreparedPanelFeatureExtractedRelease,
    create_panel_feature_extracted_execution_precommit,
    prepare_panel_feature_extracted_release,
    release_panel_feature_extracted_query_panel,
    release_panel_feature_extracted_support_panel,
    verify_prepared_panel_feature_extracted_release,
)
from bongard.panel_feature_predicate import (
    EngineeringDisposition,
    EngineeringFeatureVersionSpace,
    EngineeringSupportTable,
    FeatureVocabulary,
)
from bongard.panel_feature_proposer import PanelFeatureProposerResult
from bongard.panel_feature_task_runner import (
    PanelFeatureTaskFreeze,
    PanelFeatureTaskFreezeCommit,
)
from bongard.panel_feature_targeted_drill_plan import (
    PanelFeatureTargetedDrillPlan,
    plan_panel_feature_targeted_drill,
)
from bongard.panel_soft_ontology import (
    ClosedCount,
    ComponentCountParameters,
    FeatureFamily,
    NativeOrientation,
    PanelFeatureSpec,
    ReferenceFrame,
    SubjectScope,
)
from bongard.release import OfficialReleaseDescriptor


SEMANTIC = "hd_convex-has_four_straight_lines"
TASK_IDS = (f"{SEMANTIC}_0000", f"{SEMANTIC}_0001")
SEED = "extracted-release-gate-focused-test"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _bytes_address(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


@dataclass(frozen=True, slots=True)
class _Fixture:
    root: Path
    split_path: Path
    split: SplitIndex
    descriptor: OfficialReleaseDescriptor
    archive: OfficialExtractedPanelArchive
    predecessor: ExposureLedger
    plan: PanelFeatureTargetedDrillPlan
    store: ObjectBongardReleaseStore


def _fixture(tmp_path: Path) -> _Fixture:
    root = (tmp_path / "ShapeBongard_V2").resolve()
    for task_id in TASK_IDS:
        for side in ("0", "1"):
            directory = root / "hd" / "images" / task_id / side
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(7):
                payload = (
                    PNG_SIGNATURE
                    + f"{task_id}:{side}:{index}:authenticated".encode("ascii")
                )
                (directory / f"{index}.png").write_bytes(payload)

    split_path = root / "ShapeBongard_V2_split.json"
    split_payload = (
        json.dumps(
            {"test": [], "train": list(TASK_IDS), "val": []},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    split_path.write_bytes(split_payload)
    split = SplitIndex.load(split_path)
    corpus = ShapeBongardCorpus.from_root(root, split_file=split_path)
    manifest = corpus.build_manifest()
    inventory_digest = object_bongard_task_inventory_digest(TASK_IDS)
    descriptor = OfficialReleaseDescriptor(
        release_id="synthetic-extracted-panel-feature-release-gate",
        archive_filename="ShapeBongard_V2.zip",
        archive_sha256="sha256:" + "1" * 64,
        archive_size_bytes=1,
        split_filename=split_path.name,
        split_sha256=_bytes_address(split_payload),
        split_size_bytes=len(split_payload),
        upstream_repository="https://example.invalid/no-zip-used",
        upstream_commit="2" * 40,
        family_counts=tuple(sorted(corpus.family_counts.items())),
        primary_split_counts=(("test", 0), ("train", 2), ("val", 0)),
        regime_counts=(("BA", 0), ("CM", 0), ("FF", 0), ("NV", 0)),
        task_ids_sha256=inventory_digest,
        corpus_manifest_sha256=manifest.digest,
    )
    archive = OfficialExtractedPanelArchive._from_verified_manifest(
        descriptor,
        root,
        manifest,
    )
    predecessor = ExposureLedger.create(manifest.digest).record(
        phase="prior-engineering-drill",
        actor="fixture",
        purpose="disclose-the-generator-semantic",
        task_ids=(TASK_IDS[1],),
        source="focused-test",
        observed_at="2026-08-09T00:00:00Z",
        known_task_ids=TASK_IDS,
    )
    plan = plan_panel_feature_targeted_drill(
        task_ids=TASK_IDS,
        train_task_ids=TASK_IDS,
        predecessor=predecessor,
        target_semantic_key=SEMANTIC,
        selection_seed=SEED,
        requested_task_count=1,
        release_descriptor_digest=descriptor.digest,
        split_source_digest=descriptor.split_sha256,
        task_inventory_digest=inventory_digest,
    )
    assert plan.tasks[0].task_id == TASK_IDS[0]
    return _Fixture(
        root,
        split_path,
        split,
        descriptor,
        archive,
        predecessor,
        plan,
        ObjectBongardReleaseStore((tmp_path / "release-store").resolve()),
    )


def _precommit(fixture: _Fixture):
    return create_panel_feature_extracted_execution_precommit(
        plan=fixture.plan,
        expected_plan_digest=fixture.plan.record_digest,
        selection_seed=SEED,
        predecessor=fixture.predecessor,
        descriptor=fixture.descriptor,
        expected_release_descriptor_digest=fixture.descriptor.digest,
        archive=fixture.archive,
        split=fixture.split,
        task_ids=TASK_IDS,
        runtime_source_bindings={"runner_source": _address({"runner": 1})},
        configuration={"headless": True, "model": "gpt-5.6-sol"},
        exposure_observed_at="2026-08-09T01:00:00Z",
    )


def _prepare(fixture: _Fixture) -> PreparedPanelFeatureExtractedRelease:
    return prepare_panel_feature_extracted_release(
        store=fixture.store,
        plan=fixture.plan,
        precommit=_precommit(fixture),
        predecessor=fixture.predecessor,
    )


def _freeze(
    prepared: PreparedPanelFeatureExtractedRelease,
    fixture: _Fixture,
) -> PanelFeatureTaskFreeze:
    task = prepared.plan.tasks[0]
    released_by_id = {}
    for panel_id in prepared.authorization.authorized_support_panel_ids:
        released, _receipt = release_panel_feature_extracted_support_panel(
            prepared=prepared,
            archive=fixture.archive,
            panel_id=panel_id,
        )
        released_by_id[panel_id] = released

    side0 = tuple(
        hashlib.sha256(released_by_id[panel_id].exact_png_bytes).hexdigest()
        for panel_id in task.side_0_support_panel_ids
    )
    side1 = tuple(
        hashlib.sha256(released_by_id[panel_id].exact_png_bytes).hexdigest()
        for panel_id in task.side_1_support_panel_ids
    )
    one = PanelFeatureSpec(
        FeatureFamily.COMPONENT_COUNT,
        SubjectScope.WHOLE_PANEL,
        ReferenceFrame.NONE,
        ComponentCountParameters(ClosedCount.ONE),
    )
    five = PanelFeatureSpec(
        FeatureFamily.COMPONENT_COUNT,
        SubjectScope.WHOLE_PANEL,
        ReferenceFrame.NONE,
        ComponentCountParameters(ClosedCount.FIVE),
    )
    vocabulary = FeatureVocabulary.create(
        side0_specs=(one,),
        side1_specs=(five,),
    )
    values = {}
    for panel_digest in side0:
        values[(panel_digest, one.spec_digest)] = EngineeringDisposition.MATCH
        values[(panel_digest, five.spec_digest)] = EngineeringDisposition.NONMATCH
    for panel_digest in side1:
        values[(panel_digest, one.spec_digest)] = EngineeringDisposition.NONMATCH
        values[(panel_digest, five.spec_digest)] = EngineeringDisposition.MATCH
    table = EngineeringSupportTable.create(
        vocabulary,
        (*side0, *side1),
        values,
    )
    side0_space = EngineeringFeatureVersionSpace.create(
        table,
        NativeOrientation.SIDE0_POSITIVE,
        side0,
        side1,
    )
    side1_space = EngineeringFeatureVersionSpace.create(
        table,
        NativeOrientation.SIDE1_POSITIVE,
        side0,
        side1,
    )
    assert len(side0_space.survivor_formula_digests) == 1
    assert len(side1_space.survivor_formula_digests) == 1
    proposer = PanelFeatureProposerResult(
        "d" * 64,
        "e" * 64,
        (),
        (),
        (),
        None,
    )
    return PanelFeatureTaskFreeze.seal(
        task=task,
        execution_precommit_digest=prepared.precommit.record_digest,
        proposer=proposer,
        side0_space=side0_space,
        side1_space=side1_space,
        rank_artifact=None,
    )


def _commit(
    freeze: PanelFeatureTaskFreeze,
    receipt: ObjectBongardWriteOnceReceipt,
) -> PanelFeatureTaskFreezeCommit:
    return PanelFeatureTaskFreezeCommit.seal(freeze, receipt)


def test_support_waits_for_durable_exposure_and_never_claims_zip_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    reads: list[str] = []
    original = OfficialExtractedPanelArchive.read_panel
    prepared: PreparedPanelFeatureExtractedRelease | None = None

    def checked_read(archive: OfficialExtractedPanelArchive, panel_id: str):
        reads.append(panel_id)
        assert prepared is not None
        exposure_path = fixture.store.root / prepared.exposure_receipt.relative_path
        assert exposure_path.is_file()
        assert fixture.store.verify(
            prepared.exposure_receipt,
            expected_data=prepared.successor.to_dict(),
        ) == prepared.successor.to_dict()
        return original(archive, panel_id)

    monkeypatch.setattr(OfficialExtractedPanelArchive, "read_panel", checked_read)
    precommit = _precommit(fixture)
    assert reads == []
    prepared = prepare_panel_feature_extracted_release(
        store=fixture.store,
        plan=fixture.plan,
        precommit=precommit,
        predecessor=fixture.predecessor,
    )
    assert reads == []
    assert len(prepared.successor.events) == len(fixture.predecessor.events) + 1
    assert PanelFeatureExtractedExecutionPrecommit.from_data(
        precommit.to_data()
    ) == precommit
    assert PanelFeatureExtractedReleaseAuthorization.from_data(
        prepared.authorization.to_data()
    ) == prepared.authorization
    assert precommit.to_data()["release_source_authority"] == (
        "authenticated-extracted-tree-manifest"
    )
    assert precommit.to_data()["zip_archive_opened_or_required"] is False
    assert precommit.to_data()["zip_central_directory_custody_claimed"] is False
    assert not (fixture.root / fixture.descriptor.archive_filename).exists()

    panel_id = fixture.plan.tasks[0].side_0_support_panel_ids[0]
    released, receipt = release_panel_feature_extracted_support_panel(
        prepared=prepared,
        archive=fixture.archive,
        panel_id=panel_id,
    )
    assert reads == [panel_id]
    assert released.panel_id == panel_id
    assert receipt.object_kind == "released-extracted-support-panel"


def test_both_queries_require_the_exact_durable_python_freeze_and_commit(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    prepared = _prepare(fixture)
    task = fixture.plan.tasks[0]
    freeze = _freeze(prepared, fixture)
    freeze_receipt = persist_object_bongard_task_freeze(
        store=fixture.store,
        freeze=freeze,
    )
    commit = _commit(freeze, freeze_receipt)

    with pytest.raises(TypeError, match="exact PanelFeatureTaskFreeze"):
        release_panel_feature_extracted_query_panel(
            prepared=prepared,
            archive=fixture.archive,
            panel_id=task.side_0_query_panel_id,
            task_freeze=object(),  # type: ignore[arg-type]
            task_commit=commit,
            task_freeze_receipt=freeze_receipt,
            task_commit_receipt=prepared.precommit_receipt,
        )

    with pytest.raises(RuntimeError, match="payload differs"):
        release_panel_feature_extracted_query_panel(
            prepared=prepared,
            archive=fixture.archive,
            panel_id=task.side_0_query_panel_id,
            task_freeze=freeze,
            task_commit=commit,
            task_freeze_receipt=freeze_receipt,
            task_commit_receipt=prepared.precommit_receipt,
        )

    commit_receipt = persist_object_bongard_task_commit(
        store=fixture.store,
        commit=commit,
    )
    released_ids = []
    for panel_id in (
        task.side_0_query_panel_id,
        task.side_1_query_panel_id,
    ):
        released, receipt = release_panel_feature_extracted_query_panel(
            prepared=prepared,
            archive=fixture.archive,
            panel_id=panel_id,
            task_freeze=freeze,
            task_commit=commit,
            task_freeze_receipt=freeze_receipt,
            task_commit_receipt=commit_receipt,
        )
        released_ids.append(released.panel_id)
        assert receipt.object_kind == "released-extracted-query-panel"
    assert released_ids == [
        task.side_0_query_panel_id,
        task.side_1_query_panel_id,
    ]

    freeze_path = fixture.store.root / freeze_receipt.relative_path
    freeze_path.write_bytes(freeze_path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="payload differs"):
        release_panel_feature_extracted_query_panel(
            prepared=prepared,
            archive=fixture.archive,
            panel_id=task.side_0_query_panel_id,
            task_freeze=freeze,
            task_commit=commit,
            task_freeze_receipt=freeze_receipt,
            task_commit_receipt=commit_receipt,
        )


@pytest.mark.parametrize(
    "receipt_name",
    ("plan_receipt", "precommit_receipt", "exposure_receipt"),
)
def test_plan_precommit_and_ledger_store_tamper_fail_closed(
    tmp_path: Path,
    receipt_name: str,
) -> None:
    fixture = _fixture(tmp_path)
    prepared = _prepare(fixture)
    receipt = getattr(prepared, receipt_name)
    path = fixture.store.root / receipt.relative_path
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(Exception, match="payload differs|tamper|collision"):
        verify_prepared_panel_feature_extracted_release(prepared)
    with pytest.raises(Exception, match="payload differs|tamper|collision"):
        release_panel_feature_extracted_support_panel(
            prepared=prepared,
            archive=fixture.archive,
            panel_id=fixture.plan.tasks[0].side_0_support_panel_ids[0],
        )


def test_split_plan_ledger_descriptor_and_archive_digest_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    with pytest.raises(PanelFeatureExtractedReleaseGateError, match="plan differs"):
        create_panel_feature_extracted_execution_precommit(
            plan=fixture.plan,
            expected_plan_digest="sha256:" + "f" * 64,
            selection_seed=SEED,
            predecessor=fixture.predecessor,
            descriptor=fixture.descriptor,
            expected_release_descriptor_digest=fixture.descriptor.digest,
            archive=fixture.archive,
            split=fixture.split,
            task_ids=TASK_IDS,
            runtime_source_bindings={"runner": _address({"runner": 1})},
            configuration={},
            exposure_observed_at="2026-08-09T01:00:00Z",
        )

    fixture.split_path.write_bytes(fixture.split_path.read_bytes() + b" ")
    with pytest.raises(PanelFeatureExtractedReleaseGateError, match="split source"):
        _precommit(fixture)

    fixture = _fixture(tmp_path / "ledger")
    wrong_ledger = ExposureLedger.create(fixture.predecessor.corpus_digest)
    with pytest.raises(PanelFeatureExtractedReleaseGateError, match="plan metadata"):
        create_panel_feature_extracted_execution_precommit(
            plan=fixture.plan,
            expected_plan_digest=fixture.plan.record_digest,
            selection_seed=SEED,
            predecessor=wrong_ledger,
            descriptor=fixture.descriptor,
            expected_release_descriptor_digest=fixture.descriptor.digest,
            archive=fixture.archive,
            split=fixture.split,
            task_ids=TASK_IDS,
            runtime_source_bindings={"runner": _address({"runner": 1})},
            configuration={},
            exposure_observed_at="2026-08-09T01:00:00Z",
        )

    with pytest.raises(
        PanelFeatureExtractedReleaseGateError,
        match="external commitment",
    ):
        create_panel_feature_extracted_execution_precommit(
            plan=fixture.plan,
            expected_plan_digest=fixture.plan.record_digest,
            selection_seed=SEED,
            predecessor=fixture.predecessor,
            descriptor=fixture.descriptor,
            expected_release_descriptor_digest="sha256:" + "e" * 64,
            archive=fixture.archive,
            split=fixture.split,
            task_ids=TASK_IDS,
            runtime_source_bindings={"runner": _address({"runner": 1})},
            configuration={},
            exposure_observed_at="2026-08-09T01:00:00Z",
        )

    object.__setattr__(fixture.archive, "record_digest", "sha256:" + "d" * 64)
    with pytest.raises(
        PanelFeatureExtractedReleaseGateError,
        match="tree, manifest, descriptor",
    ):
        _precommit(fixture)


def test_path_symlink_and_same_size_panel_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    prepared = _prepare(fixture)
    panel_id = fixture.plan.tasks[0].side_0_support_panel_ids[0]
    row = fixture.archive.panel_by_id[panel_id]
    original = row.path.read_bytes()
    row.path.write_bytes(PNG_SIGNATURE + b"x" * (len(original) - len(PNG_SIGNATURE)))
    with pytest.raises(Exception, match="verified corpus manifest"):
        release_panel_feature_extracted_support_panel(
            prepared=prepared,
            archive=fixture.archive,
            panel_id=panel_id,
        )

    fixture = _fixture(tmp_path / "symlink")
    prepared = _prepare(fixture)
    panel_id = fixture.plan.tasks[0].side_0_support_panel_ids[0]
    path = fixture.archive.panel_by_id[panel_id].path
    target = path.with_name("replacement.png")
    target.write_bytes(path.read_bytes())
    path.unlink()
    os.symlink(target, path)
    with pytest.raises(Exception, match="safely|regular|path"):
        release_panel_feature_extracted_support_panel(
            prepared=prepared,
            archive=fixture.archive,
            panel_id=panel_id,
        )


def test_extracted_root_symlink_is_rejected_before_precommit(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    alias = tmp_path / "corpus-alias"
    os.symlink(fixture.root, alias, target_is_directory=True)
    forged = OfficialExtractedPanelArchive._from_verified_manifest(
        fixture.descriptor,
        alias.absolute(),
        ShapeBongardCorpus.from_root(
            fixture.root,
            split_file=fixture.split_path,
        ).build_manifest(),
    )
    with pytest.raises(
        PanelFeatureExtractedReleaseGateError,
        match="tree, manifest, descriptor|manifest panel row",
    ):
        create_panel_feature_extracted_execution_precommit(
            plan=fixture.plan,
            expected_plan_digest=fixture.plan.record_digest,
            selection_seed=SEED,
            predecessor=fixture.predecessor,
            descriptor=fixture.descriptor,
            expected_release_descriptor_digest=fixture.descriptor.digest,
            archive=forged,
            split=fixture.split,
            task_ids=TASK_IDS,
            runtime_source_bindings={"runner": _address({"runner": 1})},
            configuration={},
            exposure_observed_at="2026-08-09T01:00:00Z",
        )

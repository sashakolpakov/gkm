from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bongard.canonical import canonical_digest
from bongard.corpus import CorpusManifest, PanelManifest, SplitIndex, TaskManifest
from bongard.official_extracted_panel_archive import (
    OfficialExtractedPanelArchive,
    OfficialExtractedPanelArchiveError,
    OfficialExtractedPanelReceipt,
    ReleasedOfficialExtractedPanel,
)
from bongard.release import OfficialReleaseDescriptor


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _fixture(tmp_path: Path) -> tuple[
    OfficialExtractedPanelArchive,
    str,
    bytes,
]:
    root = tmp_path.resolve() / "ShapeBongard_V2"
    panel_id = "ff/ff_fixture_0000/1/0.png"
    path = root / "ff/images/ff_fixture_0000/1/0.png"
    path.parent.mkdir(parents=True)
    payload = b"\x89PNG\r\n\x1a\nfixture-payload"
    path.write_bytes(payload)
    panel = PanelManifest(
        panel_id=panel_id,
        task_id="ff_fixture_0000",
        family="ff",
        polarity="positive",
        index=0,
        filename="0.png",
        path=path,
        sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )
    task_body = {
        "schema": "gkm.shape-bongard-task.v1",
        "task_id": "ff_fixture_0000",
        "family": "ff",
        "panels": [panel.to_dict()],
    }
    task = TaskManifest(
        "ff_fixture_0000", "ff", (panel,), _address(task_body)
    )
    provisional_manifest = CorpusManifest(
        layout="archive",
        family_counts=(("ff", 1),),
        tasks=(task,),
        split=SplitIndex.empty(),
        digest="sha256:" + "0" * 64,
    )
    manifest = CorpusManifest(
        layout=provisional_manifest.layout,
        family_counts=provisional_manifest.family_counts,
        tasks=provisional_manifest.tasks,
        split=provisional_manifest.split,
        digest=_address(provisional_manifest.content_dict()),
    )
    descriptor = OfficialReleaseDescriptor(
        release_id="fixture",
        archive_filename="ShapeBongard_V2.zip",
        archive_sha256="sha256:" + "1" * 64,
        archive_size_bytes=1,
        split_filename="ShapeBongard_V2_split.json",
        split_sha256="sha256:" + "2" * 64,
        split_size_bytes=1,
        upstream_repository="https://example.invalid/fixture",
        upstream_commit="3" * 40,
        family_counts=(("ff", 1),),
        primary_split_counts=(),
        regime_counts=(),
        task_ids_sha256="sha256:" + "4" * 64,
        corpus_manifest_sha256=manifest.digest,
    )
    archive = OfficialExtractedPanelArchive._from_verified_manifest(
        descriptor, root, manifest
    )
    return archive, panel_id, payload


def test_release_round_trip_and_cold_verification(tmp_path: Path) -> None:
    archive, panel_id, payload = _fixture(tmp_path)
    precommit = "sha256:" + "a" * 64
    exposure = "sha256:" + "b" * 64
    released = ReleasedOfficialExtractedPanel.release(
        archive,
        panel_id,
        execution_precommit_digest=precommit,
        exposure_successor_digest=exposure,
        expected_execution_precommit_digest=precommit,
        expected_exposure_successor_digest=exposure,
    )

    assert released.exact_png_bytes == payload
    assert released.release_receipt.corpus_manifest_digest == (
        archive.corpus_manifest_digest
    )
    assert OfficialExtractedPanelReceipt.from_data(
        released.release_receipt.to_data()
    ) == released.release_receipt
    restored = ReleasedOfficialExtractedPanel.from_data(released.to_data())
    assert restored == released
    restored.cold_verify(
        archive,
        expected_execution_precommit_digest=precommit,
        expected_exposure_successor_digest=exposure,
    )


def test_tampered_extracted_panel_fails_against_verified_manifest(
    tmp_path: Path,
) -> None:
    archive, panel_id, _payload = _fixture(tmp_path)
    path = tmp_path.resolve() / (
        "ShapeBongard_V2/ff/images/ff_fixture_0000/1/0.png"
    )
    path.write_bytes(b"\x89PNG\r\n\x1a\nchanged")

    with pytest.raises(
        OfficialExtractedPanelArchiveError,
        match="verified corpus manifest",
    ):
        archive.read_panel(panel_id)


def test_release_commitments_are_not_self_asserting(tmp_path: Path) -> None:
    archive, panel_id, _payload = _fixture(tmp_path)
    with pytest.raises(
        OfficialExtractedPanelArchiveError,
        match="precommit differs",
    ):
        ReleasedOfficialExtractedPanel.release(
            archive,
            panel_id,
            execution_precommit_digest="sha256:" + "a" * 64,
            exposure_successor_digest="sha256:" + "b" * 64,
            expected_execution_precommit_digest="sha256:" + "c" * 64,
            expected_exposure_successor_digest="sha256:" + "b" * 64,
        )


def test_panel_outside_verified_manifest_is_rejected(tmp_path: Path) -> None:
    archive, _panel_id, _payload = _fixture(tmp_path)
    with pytest.raises(
        OfficialExtractedPanelArchiveError,
        match="absent from the verified corpus manifest",
    ):
        archive.read_panel("ff/ff_fixture_0000/1/1.png")

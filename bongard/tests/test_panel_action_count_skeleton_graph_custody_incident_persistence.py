from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import pytest

from bongard.canonical import canonical_digest, canonical_json
from bongard.exposure import ExposureLedger
from bongard import panel_action_count_skeleton_graph_calibration_runner as core
from bongard import panel_action_count_skeleton_graph_custody_incident as incident_api
from bongard import panel_action_count_skeleton_graph_custody_incident_persistence as subject


ROOT = Path(__file__).resolve().parents[2]
REAL_PREDECESSOR = (
    ROOT
    / subject.AUTHORITY_RELATIVE_DIRECTORY
    / subject.PREDECESSOR_FILENAME
)
REAL_INCIDENT = ROOT / subject.INCIDENT_RELATIVE_PATH
OBSERVED_AT = "2026-08-10T15:15:00Z"


def _prepared_root(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    authority = root / subject.AUTHORITY_RELATIVE_DIRECTORY
    authority.mkdir(parents=True)
    incident_path = root / subject.INCIDENT_RELATIVE_PATH
    incident_path.parent.mkdir(parents=True)
    (authority / subject.PREDECESSOR_FILENAME).write_bytes(
        REAL_PREDECESSOR.read_bytes()
    )
    incident_path.write_bytes(REAL_INCIDENT.read_bytes())
    return root


def _inventory(root: Path) -> dict[str, bytes]:
    authority = root / subject.AUTHORITY_RELATIVE_DIRECTORY
    return {
        path.name: path.read_bytes()
        for path in sorted(authority.iterdir())
        if path.is_file()
    }


def _persist(root: Path, *, observed_at: str | None = None):
    return subject._persist_at_repository_root(root, observed_at=observed_at)


def _verify(root: Path):
    return subject._verify_at_repository_root(root)


def test_persist_and_fresh_verify_exact_tombstone(tmp_path: Path) -> None:
    root = _prepared_root(tmp_path)
    receipt = _persist(root, observed_at=OBSERVED_AT)
    restored = _verify(root)
    assert restored == receipt
    assert receipt.persistence_completed is True
    assert receipt.serialized_receipt_is_authority is False
    assert receipt.fresh_store_verification_required is True
    assert receipt.calibration_pixels_authorized is False
    assert receipt.action_program_or_label_reads_authorized is False
    assert receipt.target_query_support_test_pixels_authorized is False
    assert receipt.benchmark_claim_authorized is False
    assert receipt.incident_event_observed_at == OBSERVED_AT
    assert receipt.predecessor_event_count == 158
    assert receipt.incident_event_sequence == 158
    assert receipt.successor_event_count == 159
    assert set(_inventory(root)) == {
        subject.PREDECESSOR_FILENAME,
        incident_api.CAMPAIGN_INTENT_FILENAME,
        receipt.successor_filename,
        subject.RECEIPT_FILENAME,
    }
    claim = json.loads(
        _inventory(root)[incident_api.CAMPAIGN_INTENT_FILENAME]
    )
    assert claim["schema"] == incident_api.TOMBSTONE_CLAIM_SCHEMA
    assert claim["schema"] != subject.PINNED_CORE_INTENT_SCHEMA


@pytest.mark.parametrize("crash_stage", ["claim", "successor"])
def test_crash_recovery_is_idempotent_and_never_forks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_stage: str,
) -> None:
    root = _prepared_root(tmp_path)

    def crash(stage: str) -> None:
        if stage == crash_stage:
            raise RuntimeError(f"synthetic crash after {stage}")

    monkeypatch.setattr(subject, "_stage_hook", crash)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        _persist(root, observed_at=OBSERVED_AT)
    after_crash = _inventory(root)
    assert incident_api.CAMPAIGN_INTENT_FILENAME in after_crash
    with pytest.raises(subject.SkeletonGraphCustodyIncidentPersistenceError):
        _persist(root, observed_at="2026-08-10T15:15:01Z")
    monkeypatch.setattr(subject, "_stage_hook", lambda _stage: None)
    receipt = _persist(root)
    assert receipt.incident_event_observed_at == OBSERVED_AT
    final = _inventory(root)
    assert all("pending" not in name for name in final)
    assert _persist(root) == receipt
    assert _inventory(root) == final


def test_verifier_is_zero_write_and_rejects_receipt_tamper(tmp_path: Path) -> None:
    root = _prepared_root(tmp_path)
    receipt = _persist(root, observed_at=OBSERVED_AT)
    before = _inventory(root)
    assert _verify(root) == receipt
    assert _inventory(root) == before

    receipt_path = root / subject.AUTHORITY_RELATIVE_DIRECTORY / subject.RECEIPT_FILENAME
    data = receipt.to_data()
    data["benchmark_claim_authorized"] = True
    content = {name: value for name, value in data.items() if name != "record_digest"}
    data["record_digest"] = "sha256:" + canonical_digest(content)
    receipt_path.write_bytes(canonical_json(data) + b"\n")
    with pytest.raises(subject.SkeletonGraphCustodyIncidentPersistenceError):
        _verify(root)


def test_claim_collision_and_symlink_roots_fail_before_publication(
    tmp_path: Path,
) -> None:
    root = _prepared_root(tmp_path)
    authority = root / subject.AUTHORITY_RELATIVE_DIRECTORY
    (authority / incident_api.CAMPAIGN_INTENT_FILENAME).write_text(
        '{"schema":"different"}\n', encoding="utf-8"
    )
    with pytest.raises(subject.SkeletonGraphCustodyIncidentPersistenceError):
        _persist(root, observed_at=OBSERVED_AT)
    assert subject.RECEIPT_FILENAME not in _inventory(root)

    real = _prepared_root(tmp_path / "second")
    alias = tmp_path / "repository-alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(subject.SkeletonGraphCustodyIncidentPersistenceError):
        _persist(alias, observed_at=OBSERVED_AT)


def test_receipt_parser_rejects_numeric_bool_and_subclass_aliases(
    tmp_path: Path,
) -> None:
    root = _prepared_root(tmp_path)
    receipt = _persist(root, observed_at=OBSERVED_AT)
    for field, replacement in (
        ("persistence_completed", 1),
        ("successor_event_count", 159.0),
        ("benchmark_claim_authorized", True),
    ):
        changed = receipt.to_data()
        changed[field] = replacement
        content = {
            name: value for name, value in changed.items() if name != "record_digest"
        }
        changed["record_digest"] = "sha256:" + canonical_digest(content)
        with pytest.raises(subject.SkeletonGraphCustodyIncidentPersistenceError):
            subject.SkeletonGraphCustodyIncidentPersistenceReceipt.from_data(changed)

    class ReceiptSubclass(subject.SkeletonGraphCustodyIncidentPersistenceReceipt):
        pass

    with pytest.raises(subject.SkeletonGraphCustodyIncidentPersistenceError):
        ReceiptSubclass.from_data(receipt.to_data())


def test_receipt_binds_exact_store_and_only_verified_capability_is_authority(
    tmp_path: Path,
) -> None:
    first_root = _prepared_root(tmp_path / "first")
    second_root = _prepared_root(tmp_path / "second")
    first = subject._persist_at_repository_root(
        first_root, observed_at=OBSERVED_AT
    )
    second = subject._persist_at_repository_root(
        second_root, observed_at=OBSERVED_AT
    )
    assert first.record_digest != second.record_digest
    assert first.repository_root_absolute_path == str(first_root)
    assert second.repository_root_absolute_path == str(second_root)
    assert first.authority_directory_st_ino != second.authority_directory_st_ino
    parsed = subject.SkeletonGraphCustodyIncidentPersistenceReceipt.from_data(
        first.to_data()
    )
    with pytest.raises(subject.SkeletonGraphCustodyIncidentPersistenceError):
        subject._validate_verified_persistence(parsed)  # type: ignore[arg-type]
    with pytest.raises(subject.SkeletonGraphCustodyIncidentPersistenceError):
        subject.SkeletonGraphVerifiedCustodyIncidentPersistence._from_fresh_store(
            issuance_token=object()
        )
    assert not hasattr(subject.SkeletonGraphVerifiedCustodyIncidentPersistence, "_issue")
    unissued = subject.SkeletonGraphVerifiedCustodyIncidentPersistence()
    with pytest.raises(subject.SkeletonGraphCustodyIncidentPersistenceError):
        subject._validate_verified_persistence(unissued)


def test_fresh_claim_rejects_preexisting_successor_and_receipt(
    tmp_path: Path,
) -> None:
    root = _prepared_root(tmp_path / "successor")
    incident = incident_api.load_incident_record(root / subject.INCIDENT_RELATIVE_PATH)
    predecessor = ExposureLedger.from_dict(
        json.loads(
            (
                root
                / subject.AUTHORITY_RELATIVE_DIRECTORY
                / subject.PREDECESSOR_FILENAME
            ).read_bytes()
        )
    )
    successor, _claim = incident_api.build_incident_exposure_tombstone(
        predecessor, incident=incident, observed_at=OBSERVED_AT
    )
    successor_path = (
        root
        / subject.AUTHORITY_RELATIVE_DIRECTORY
        / (successor.digest.removeprefix("sha256:") + ".exposure.json")
    )
    successor_path.write_text(successor.to_json(), encoding="utf-8")
    with pytest.raises(
        subject.SkeletonGraphCustodyIncidentPersistenceError,
        match="predates the fixed claim",
    ):
        _persist(root, observed_at=OBSERVED_AT)
    assert incident_api.CAMPAIGN_INTENT_FILENAME not in _inventory(root)

    alternate = _prepared_root(tmp_path / "alternate-successor")
    alternate_incident = incident_api.load_incident_record(
        alternate / subject.INCIDENT_RELATIVE_PATH
    )
    alternate_predecessor = ExposureLedger.from_dict(
        json.loads(
            (
                alternate
                / subject.AUTHORITY_RELATIVE_DIRECTORY
                / subject.PREDECESSOR_FILENAME
            ).read_bytes()
        )
    )
    alternate_successor, _alternate_claim = (
        incident_api.build_incident_exposure_tombstone(
            alternate_predecessor,
            incident=alternate_incident,
            observed_at=OBSERVED_AT,
        )
    )
    alternate_authority = alternate / subject.AUTHORITY_RELATIVE_DIRECTORY
    (
        alternate_authority
        / (
            alternate_successor.digest.removeprefix("sha256:")
            + ".exposure.json"
        )
    ).write_text(alternate_successor.to_json(), encoding="utf-8")
    with pytest.raises(
        subject.SkeletonGraphCustodyIncidentPersistenceError,
        match="predates the fixed claim",
    ):
        _persist(alternate, observed_at="2026-08-10T15:15:01Z")
    assert incident_api.CAMPAIGN_INTENT_FILENAME not in _inventory(alternate)

    other = _prepared_root(tmp_path / "receipt")
    authority = other / subject.AUTHORITY_RELATIVE_DIRECTORY
    (authority / subject.RECEIPT_FILENAME).write_text(
        '{"not":"a receipt"}\n', encoding="utf-8"
    )
    with pytest.raises(
        subject.SkeletonGraphCustodyIncidentPersistenceError,
        match="predates the fixed claim",
    ):
        _persist(other, observed_at=OBSERVED_AT)
    assert incident_api.CAMPAIGN_INTENT_FILENAME not in _inventory(other)


def test_real_timestamp_and_public_fixed_root_surface(tmp_path: Path) -> None:
    root = _prepared_root(tmp_path)
    for invalid in (
        "2026-99-99T99:99:99Z",
        "2026-02-30T12:00:00Z",
        "2026-08-10T12:00:00+00:00",
    ):
        with pytest.raises(subject.SkeletonGraphCustodyIncidentPersistenceError):
            _persist(root, observed_at=invalid)
    assert not inspect.signature(
        subject.persist_custody_incident_tombstone
    ).parameters
    assert not _inventory(root).keys() - {subject.PREDECESSOR_FILENAME}

    seeded = _prepared_root(tmp_path / "seeded")
    seeded_incident = incident_api.load_incident_record(
        seeded / subject.INCIDENT_RELATIVE_PATH
    )
    seeded_predecessor = ExposureLedger.from_dict(
        json.loads(
            (
                seeded
                / subject.AUTHORITY_RELATIVE_DIRECTORY
                / subject.PREDECESSOR_FILENAME
            ).read_bytes()
        )
    )
    _bad_successor, bad_claim = incident_api.build_incident_exposure_tombstone(
        seeded_predecessor,
        incident=seeded_incident,
        observed_at="2026-02-30T12:00:00Z",
    )
    seeded_authority = seeded / subject.AUTHORITY_RELATIVE_DIRECTORY
    (seeded_authority / incident_api.CAMPAIGN_INTENT_FILENAME).write_bytes(
        incident_api.tombstone_claim_bytes(bad_claim)
    )
    before = _inventory(seeded)
    with pytest.raises(
        subject.SkeletonGraphCustodyIncidentPersistenceError,
        match="real UTC timestamp",
    ):
        _persist(seeded)
    assert _inventory(seeded) == before


def test_leaf_rebinding_and_private_temp_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _prepared_root(tmp_path)
    _persist(root, observed_at=OBSERVED_AT)
    authority = root / subject.AUTHORITY_RELATIVE_DIRECTORY
    receipt_path = authority / subject.RECEIPT_FILENAME
    original_reader = core._read_dirfd_bytes
    changed = False

    def replace_after_read(*args, **kwargs):
        nonlocal changed
        raw = original_reader(*args, **kwargs)
        if kwargs.get("label") == "incident persistence receipt" and not changed:
            changed = True
            replacement = authority / "replacement.json"
            replacement.write_text('{"tampered":true}\n', encoding="utf-8")
            os.replace(replacement, receipt_path)
        return raw

    monkeypatch.setattr(core, "_read_dirfd_bytes", replace_after_read)
    with pytest.raises(subject.SkeletonGraphCustodyIncidentPersistenceError):
        subject._verify_at_repository_root(root)

    recovery_root = _prepared_root(tmp_path / "recovery")
    recovery_authority = recovery_root / subject.AUTHORITY_RELATIVE_DIRECTORY
    pending = recovery_authority / (
        f".{incident_api.CAMPAIGN_INTENT_FILENAME}.pending.1.2."
        + "0" * 32
    )
    pending.write_bytes(b"complete but unpublished")
    pending.chmod(0o600)
    _persist(recovery_root, observed_at=OBSERVED_AT)
    assert not pending.exists()
    assert all(".pending." not in name for name in _inventory(recovery_root))

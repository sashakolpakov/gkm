from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from bongard.canonical import canonical_digest, canonical_json
from bongard.exposure import ExposureLedger
from bongard import panel_action_count_skeleton_graph_calibration_runner as runner
from bongard import panel_action_count_skeleton_graph_custody_incident as subject


ROOT = Path(__file__).resolve().parents[2]
PREDECESSOR = (
    ROOT
    / "downloads/ShapeBongard_V2_full"
    / "panel_soft_exact_unused_train_20260809_ranked_v1"
    / "research-exposure-successors"
    / "6995ea9cfda2f384cb0ba1b1cdc3611c965227c60fdb281d1e2e56fffa357b56.exposure.json"
)
INCIDENT = ROOT / subject.INCIDENT_RECORD_PATH
OBSERVED_AT = "2026-08-10T14:00:00Z"


def test_checked_in_incident_is_exact_canonical_and_fail_closed() -> None:
    expected = subject.build_overbroad_rg_incident()
    loaded = subject.load_incident_record(INCIDENT)
    assert loaded == expected
    assert INCIDENT.read_bytes() == canonical_json(expected.to_data()) + b"\n"
    assert loaded.record_digest == (
        "sha256:c647b0929a524a3fec64f74afbda1d1f469e6cf4ba1b8d6da1de788f0af2801f"
    )
    assert loaded.official_program_task_count == 4400
    assert loaded.target_family_task_ids == subject.TARGET_FAMILY_TASK_IDS
    assert loaded.same_family_calibration_task_ids == subject.TARGET_FAMILY_TASK_IDS[2:18]
    assert loaded.calibration_live_launch_authorized is False
    assert loaded.target_query_authorized is False
    assert loaded.benchmark_claim_authorized is False
    assert loaded.required_terminal_outcome == "typed_custody_gap"
    assert loaded.no_png_bytes_opened is True
    assert loaded.no_zip_or_archive_bytes_opened is True
    assert loaded.no_model_or_prediction_execution is True
    assert loaded.no_files_written_by_incident_command is True


def test_incident_parser_rejects_tamper_and_numeric_bool_aliases(tmp_path: Path) -> None:
    data = subject.build_overbroad_rg_incident().to_data()
    for field, replacement in (
        ("record_digest", "sha256:" + "0" * 64),
        ("target_query_authorized", True),
        ("official_program_size_bytes", 10_311_475.0),
        ("no_png_bytes_opened", 1),
        ("renderer_omitted_byte_count", 9_773_906),
    ):
        changed = dict(data)
        changed[field] = replacement
        path = tmp_path / f"{field}.json"
        path.write_bytes(canonical_json(changed) + b"\n")
        with pytest.raises(subject.SkeletonGraphCustodyIncidentError):
            subject.load_incident_record(path)


def test_tombstone_is_exact_one_event_child_and_blocks_core_schema() -> None:
    predecessor = ExposureLedger.load(PREDECESSOR)
    incident = subject.load_incident_record(INCIDENT)
    successor, claim = subject.build_incident_exposure_tombstone(
        predecessor,
        incident=incident,
        observed_at=OBSERVED_AT,
    )
    assert len(predecessor.events) == 158
    assert successor.events[:-1] == predecessor.events
    assert len(successor.events) == 159
    event = successor.events[-1]
    assert event.sequence == 158
    assert event.previous_digest == predecessor.events[-1].digest
    assert event.task_ids == subject.TARGET_FAMILY_TASK_IDS
    assert event.panel_ids == ()
    assert event.source == incident.record_digest
    assert claim.successor_ledger_digest == successor.digest
    assert claim.fixed_campaign_intent_filename == subject.CAMPAIGN_INTENT_FILENAME
    assert claim.write_once_persistence_required is True
    assert claim.calibration_pixels_authorized is False
    raw = subject.tombstone_claim_bytes(claim)
    decoded = json.loads(raw)
    assert decoded["schema"] == subject.TOMBSTONE_CLAIM_SCHEMA
    assert decoded["schema"] != runner.CAMPAIGN_ATTEMPT_AUTHORITY_SCHEMA
    restored = subject.SkeletonGraphIncidentTombstoneClaim.from_data(decoded)
    assert restored == claim
    subject.verify_incident_tombstone_claim(
        predecessor,
        successor=successor,
        claim=restored,
        incident=incident,
    )


def test_tombstone_verifier_rejects_every_authority_mutation() -> None:
    predecessor = ExposureLedger.load(PREDECESSOR)
    incident = subject.load_incident_record(INCIDENT)
    successor, claim = subject.build_incident_exposure_tombstone(
        predecessor,
        incident=incident,
        observed_at=OBSERVED_AT,
    )
    changes = (
        {"calibration_pixels_authorized": True},
        {"benchmark_claim_authorized": True},
        {"write_once_persistence_required": False},
        {"affected_task_ids": claim.affected_task_ids[:-1]},
        {"successor_file_sha256": "sha256:" + "1" * 64},
        {"incident_event_observed_at": "2026-08-10T14:00:01Z"},
    )
    for change in changes:
        bad = replace(claim, **change)
        with pytest.raises(subject.SkeletonGraphCustodyIncidentError):
            subject.verify_incident_tombstone_claim(
                predecessor,
                successor=successor,
                claim=bad,
                incident=incident,
            )


def test_resealed_incident_and_wrong_corpus_child_are_rejected() -> None:
    predecessor = ExposureLedger.load(PREDECESSOR)
    incident = subject.load_incident_record(INCIDENT)
    changed_incident = replace(incident, shell_command="rg --wrong")
    changed_incident = replace(
        changed_incident,
        record_digest="sha256:" + canonical_digest(changed_incident.content_data()),
    )
    with pytest.raises(subject.SkeletonGraphCustodyIncidentError):
        subject.build_incident_exposure_tombstone(
            predecessor,
            incident=changed_incident,
            observed_at=OBSERVED_AT,
        )

    successor, claim = subject.build_incident_exposure_tombstone(
        predecessor,
        incident=incident,
        observed_at=OBSERVED_AT,
    )
    wrong_successor = ExposureLedger(
        corpus_digest="sha256:" + "0" * 64,
        events=successor.events,
    )
    wrong_raw = wrong_successor.to_json().encode("utf-8")
    wrong_claim = replace(
        claim,
        successor_ledger_digest=wrong_successor.digest,
        successor_file_sha256="sha256:" + hashlib.sha256(wrong_raw).hexdigest(),
        successor_filename=wrong_successor.digest.removeprefix("sha256:")
        + ".exposure.json",
    )
    wrong_claim = replace(
        wrong_claim,
        record_digest="sha256:" + canonical_digest(wrong_claim.content_data()),
    )
    with pytest.raises(subject.SkeletonGraphCustodyIncidentError):
        subject.verify_incident_tombstone_claim(
            predecessor,
            successor=wrong_successor,
            claim=wrong_claim,
            incident=incident,
        )


def test_claim_parser_and_serializer_reject_resealed_policy_reversal() -> None:
    predecessor = ExposureLedger.load(PREDECESSOR)
    incident = subject.load_incident_record(INCIDENT)
    _successor, claim = subject.build_incident_exposure_tombstone(
        predecessor,
        incident=incident,
        observed_at=OBSERVED_AT,
    )
    changed = dict(claim.to_data())
    changed["calibration_pixels_authorized"] = True
    content = {name: value for name, value in changed.items() if name != "record_digest"}
    changed["record_digest"] = "sha256:" + canonical_digest(content)
    with pytest.raises(subject.SkeletonGraphCustodyIncidentError):
        subject.SkeletonGraphIncidentTombstoneClaim.from_data(changed)

    altered = replace(claim, calibration_pixels_authorized=True)
    altered = replace(
        altered,
        record_digest="sha256:" + canonical_digest(altered.content_data()),
    )
    with pytest.raises(subject.SkeletonGraphCustodyIncidentError):
        subject.tombstone_claim_bytes(altered)


def test_direct_verifier_rejects_forged_incident_and_fully_resealed_chain() -> None:
    predecessor = ExposureLedger.load(PREDECESSOR)
    incident = subject.load_incident_record(INCIDENT)
    _successor, claim = subject.build_incident_exposure_tombstone(
        predecessor,
        incident=incident,
        observed_at=OBSERVED_AT,
    )
    forged = replace(incident, shell_command="false factual transcript")
    forged = replace(
        forged,
        record_digest="sha256:" + canonical_digest(forged.content_data()),
    )
    forged_successor = predecessor.record(
        phase="custody-incident",
        actor="gkm-codex-custody-incident-recorder-v1",
        purpose="official-programs-preexposed-no-launch",
        task_ids=subject.TARGET_FAMILY_TASK_IDS,
        source=forged.record_digest,
        observed_at=OBSERVED_AT,
        allow_sealed=True,
    )
    forged_raw = forged_successor.to_json().encode("utf-8")
    forged_event = forged_successor.events[-1]
    forged_claim = replace(
        claim,
        incident_record_digest=forged.record_digest,
        successor_ledger_digest=forged_successor.digest,
        successor_file_sha256="sha256:" + hashlib.sha256(forged_raw).hexdigest(),
        successor_filename=forged_successor.digest.removeprefix("sha256:")
        + ".exposure.json",
        incident_event_digest=forged_event.digest,
    )
    forged_claim = replace(
        forged_claim,
        record_digest="sha256:" + canonical_digest(forged_claim.content_data()),
    )
    with pytest.raises(subject.SkeletonGraphCustodyIncidentError):
        subject.verify_incident_tombstone_claim(
            predecessor,
            successor=forged_successor,
            claim=forged_claim,
            incident=forged,
        )


def test_claim_serializer_rejects_subclass_content_override() -> None:
    predecessor = ExposureLedger.load(PREDECESSOR)
    incident = subject.load_incident_record(INCIDENT)
    _successor, claim = subject.build_incident_exposure_tombstone(
        predecessor,
        incident=incident,
        observed_at=OBSERVED_AT,
    )

    class ForgedClaim(subject.SkeletonGraphIncidentTombstoneClaim):
        def content_data(self) -> dict[str, object]:
            data = super().content_data()
            data["calibration_pixels_authorized"] = True
            return data

    values = {
        name: getattr(claim, name)
        for name in subject.SkeletonGraphIncidentTombstoneClaim.__dataclass_fields__
    }
    forged = ForgedClaim(**values)
    object.__setattr__(
        forged,
        "record_digest",
        "sha256:" + canonical_digest(forged.content_data()),
    )
    with pytest.raises(subject.SkeletonGraphCustodyIncidentError):
        subject.tombstone_claim_bytes(forged)


def test_verifier_rejects_ledger_subclass_byte_override() -> None:
    predecessor = ExposureLedger.load(PREDECESSOR)
    incident = subject.load_incident_record(INCIDENT)
    successor, claim = subject.build_incident_exposure_tombstone(
        predecessor,
        incident=incident,
        observed_at=OBSERVED_AT,
    )

    class ForgedLedger(ExposureLedger):
        def to_json(self) -> str:
            return '{"calibration_pixels_authorized":true}\n'

    forged = ForgedLedger(
        corpus_digest=successor.corpus_digest,
        events=successor.events,
    )
    forged_raw = forged.to_json().encode("utf-8")
    forged_claim = replace(
        claim,
        successor_file_sha256="sha256:" + hashlib.sha256(forged_raw).hexdigest(),
    )
    forged_claim = replace(
        forged_claim,
        record_digest="sha256:" + canonical_digest(forged_claim.content_data()),
    )
    with pytest.raises(subject.SkeletonGraphCustodyIncidentError):
        subject.verify_incident_tombstone_claim(
            predecessor,
            successor=forged,
            claim=forged_claim,
            incident=incident,
        )


def test_exact_record_and_string_leaf_types_are_mandatory() -> None:
    incident = subject.load_incident_record(INCIDENT)

    class IncidentSubclass(subject.SkeletonGraphCustodyIncident):
        pass

    with pytest.raises(subject.SkeletonGraphCustodyIncidentError):
        IncidentSubclass.from_data(incident.to_data())

    class Text(str):
        pass

    changed_incident = replace(incident, incident_id=Text(incident.incident_id))
    changed_incident = replace(
        changed_incident,
        record_digest="sha256:" + canonical_digest(changed_incident.content_data()),
    )
    predecessor = ExposureLedger.load(PREDECESSOR)
    successor, claim = subject.build_incident_exposure_tombstone(
        predecessor,
        incident=incident,
        observed_at=OBSERVED_AT,
    )
    with pytest.raises(subject.SkeletonGraphCustodyIncidentError):
        subject.verify_incident_tombstone_claim(
            predecessor,
            successor=successor,
            claim=claim,
            incident=changed_incident,
        )

    changed_claim = replace(
        claim,
        successor_filename=Text(claim.successor_filename),
    )
    changed_claim = replace(
        changed_claim,
        record_digest="sha256:" + canonical_digest(changed_claim.content_data()),
    )
    with pytest.raises(subject.SkeletonGraphCustodyIncidentError):
        subject.tombstone_claim_bytes(changed_claim)


def test_claim_parser_rejects_unrelated_resealed_successor_filename() -> None:
    predecessor = ExposureLedger.load(PREDECESSOR)
    incident = subject.load_incident_record(INCIDENT)
    _successor, claim = subject.build_incident_exposure_tombstone(
        predecessor,
        incident=incident,
        observed_at=OBSERVED_AT,
    )
    changed = claim.to_data()
    changed["successor_filename"] = "unrelated.exposure.json"
    content = {name: value for name, value in changed.items() if name != "record_digest"}
    changed["record_digest"] = "sha256:" + canonical_digest(content)
    with pytest.raises(subject.SkeletonGraphCustodyIncidentError):
        subject.SkeletonGraphIncidentTombstoneClaim.from_data(changed)

    changed_digest = claim.to_data()
    changed_digest["incident_record_digest"] = "sha256:" + "0" * 64
    content = {
        name: value
        for name, value in changed_digest.items()
        if name != "record_digest"
    }
    changed_digest["record_digest"] = "sha256:" + canonical_digest(content)
    with pytest.raises(subject.SkeletonGraphCustodyIncidentError):
        subject.SkeletonGraphIncidentTombstoneClaim.from_data(changed_digest)


def test_parsers_reject_string_subclass_schema_values_and_keys() -> None:
    incident = subject.load_incident_record(INCIDENT)

    class Text(str):
        pass

    for changed in (
        {**incident.to_data(), "schema": Text(subject.INCIDENT_SCHEMA)},
        {
            **{
                key: value
                for key, value in incident.to_data().items()
                if key != "schema"
            },
            Text("schema"): subject.INCIDENT_SCHEMA,
        },
    ):
        with pytest.raises(subject.SkeletonGraphCustodyIncidentError):
            subject.SkeletonGraphCustodyIncident.from_data(changed)

    predecessor = ExposureLedger.load(PREDECESSOR)
    _successor, claim = subject.build_incident_exposure_tombstone(
        predecessor,
        incident=incident,
        observed_at=OBSERVED_AT,
    )
    for changed in (
        {**claim.to_data(), "schema": Text(subject.TOMBSTONE_CLAIM_SCHEMA)},
        {
            **{
                key: value for key, value in claim.to_data().items() if key != "schema"
            },
            Text("schema"): subject.TOMBSTONE_CLAIM_SCHEMA,
        },
    ):
        with pytest.raises(subject.SkeletonGraphCustodyIncidentError):
            subject.SkeletonGraphIncidentTombstoneClaim.from_data(changed)


def test_verifier_rejects_ledger_leaf_and_container_subclasses() -> None:
    predecessor = ExposureLedger.load(PREDECESSOR)
    incident = subject.load_incident_record(INCIDENT)
    successor, claim = subject.build_incident_exposure_tombstone(
        predecessor,
        incident=incident,
        observed_at=OBSERVED_AT,
    )

    class Text(str):
        pass

    changed_predecessor = ExposureLedger(
        corpus_digest=Text(predecessor.corpus_digest),
        events=predecessor.events,
    )
    with pytest.raises(subject.SkeletonGraphCustodyIncidentError):
        subject.verify_incident_tombstone_claim(
            changed_predecessor,
            successor=successor,
            claim=claim,
            incident=incident,
        )

    changed_successor = ExposureLedger(
        corpus_digest=successor.corpus_digest,
        events=list(successor.events),  # type: ignore[arg-type]
    )
    with pytest.raises(subject.SkeletonGraphCustodyIncidentError):
        subject.verify_incident_tombstone_claim(
            predecessor,
            successor=changed_successor,
            claim=claim,
            incident=incident,
        )


def test_builder_performs_no_path_or_official_data_io(monkeypatch: pytest.MonkeyPatch) -> None:
    predecessor = ExposureLedger.load(PREDECESSOR)
    incident = subject.load_incident_record(INCIDENT)
    original_read_bytes = Path.read_bytes

    def source_only(path: Path) -> bytes:
        if path.resolve() != Path(subject.__file__).resolve():
            raise AssertionError(f"builder attempted non-source I/O: {path}")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", source_only)
    successor, claim = subject.build_incident_exposure_tombstone(
        predecessor,
        incident=incident,
        observed_at=OBSERVED_AT,
    )
    assert successor.digest == claim.successor_ledger_digest

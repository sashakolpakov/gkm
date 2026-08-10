from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from bongard.canonical import canonical_digest, canonical_json
from bongard import (
    panel_action_count_skeleton_graph_custody_gap as subject,
)
from bongard import (
    panel_action_count_skeleton_graph_custody_incident_persistence
    as persistence_api,
)


ROOT = Path(__file__).resolve().parents[2]
SUBJECT = ROOT / "bongard/panel_action_count_skeleton_graph_custody_gap.py"
RECORD = (
    ROOT
    / "bongard/data/panel_action_count_skeleton_graph_custody_gap_20260810_v1.json"
)


def _live_receipt() -> persistence_api.SkeletonGraphCustodyIncidentPersistenceReceipt:
    receipt = persistence_api.SkeletonGraphCustodyIncidentPersistenceReceipt(
        repository_root_absolute_path="/Users/sasha/gkm",
        repository_root_st_dev=16_777_234,
        repository_root_st_ino=7_230_619,
        repository_root_st_mode=16_877,
        authority_directory_relative_path=(
            persistence_api.AUTHORITY_RELATIVE_DIRECTORY.as_posix()
        ),
        authority_directory_absolute_path=(
            "/Users/sasha/gkm/"
            + persistence_api.AUTHORITY_RELATIVE_DIRECTORY.as_posix()
        ),
        authority_directory_st_dev=16_777_234,
        authority_directory_st_ino=36_368_499,
        authority_directory_st_mode=16_832,
        incident_commit=persistence_api.INCIDENT_COMMIT,
        incident_source_sha256=persistence_api.PINNED_INCIDENT_SOURCE_SHA256,
        incident_file_sha256=subject.PINNED_INCIDENT_FILE_SHA256,
        incident_record_digest=subject.PINNED_INCIDENT_RECORD_DIGEST,
        predecessor_filename=persistence_api.PREDECESSOR_FILENAME,
        predecessor_file_sha256=subject.PINNED_PREDECESSOR_FILE_SHA256,
        predecessor_ledger_digest=subject.PINNED_PREDECESSOR_LEDGER_DIGEST,
        predecessor_corpus_digest=subject.PINNED_PREDECESSOR_CORPUS_DIGEST,
        predecessor_event_count=158,
        claim_filename=persistence_api.incident_api.CAMPAIGN_INTENT_FILENAME,
        claim_schema=persistence_api.incident_api.TOMBSTONE_CLAIM_SCHEMA,
        claim_file_sha256=subject.PINNED_CLAIM_FILE_SHA256,
        claim_record_digest=subject.PINNED_CLAIM_RECORD_DIGEST,
        incident_event_digest=subject.PINNED_INCIDENT_EVENT_DIGEST,
        incident_event_observed_at=subject.PINNED_INCIDENT_EVENT_OBSERVED_AT,
        incident_event_sequence=158,
        successor_filename=(
            subject.PINNED_SUCCESSOR_LEDGER_DIGEST.removeprefix("sha256:")
            + ".exposure.json"
        ),
        successor_file_sha256=subject.PINNED_SUCCESSOR_FILE_SHA256,
        successor_ledger_digest=subject.PINNED_SUCCESSOR_LEDGER_DIGEST,
        successor_event_count=159,
        core_commit=persistence_api.PINNED_CORE_COMMIT,
        core_source_sha256=persistence_api.PINNED_CORE_SOURCE_SHA256,
        core_expected_intent_schema=persistence_api.PINNED_CORE_INTENT_SCHEMA,
        canonical_source_sha256=(
            persistence_api.PINNED_CANONICAL_SOURCE_SHA256
        ),
        exposure_source_sha256=persistence_api.PINNED_EXPOSURE_SOURCE_SHA256,
        runtime_source_snapshot_sha256=(
            persistence_api.PINNED_RUNTIME_SOURCE_SNAPSHOT_SHA256
        ),
        calibration_prereg_source_sha256=(
            persistence_api.PINNED_CALIBRATION_PREREG_SOURCE_SHA256
        ),
        core_claim_schema_rejected=True,
        persistence_completed=True,
        serialized_receipt_is_authority=False,
        fresh_store_verification_required=True,
        calibration_pixels_authorized=False,
        action_program_or_label_reads_authorized=False,
        target_query_support_test_pixels_authorized=False,
        benchmark_claim_authorized=False,
        persistence_source_sha256=(
            "sha256:" + subject.PINNED_PERSISTENCE_SOURCE_SHA256
        ),
        record_digest=subject.PINNED_PERSISTENCE_RECEIPT_RECORD_DIGEST,
    )
    assert receipt.file_sha256 == subject.PINNED_PERSISTENCE_RECEIPT_FILE_SHA256
    return receipt


def _verified_capability(
) -> persistence_api.SkeletonGraphVerifiedCustodyIncidentPersistence:
    result = object.__new__(
        persistence_api.SkeletonGraphVerifiedCustodyIncidentPersistence
    )
    object.__setattr__(result, "receipt", _live_receipt())
    object.__setattr__(
        result,
        "_issuance_token",
        persistence_api._VERIFIED_ISSUANCE_TOKEN,
    )
    persistence_api._validate_verified_persistence(result)
    return result


def _reseal(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    content = dict(result)
    content.pop("record_digest", None)
    result["record_digest"] = "sha256:" + canonical_digest(content)
    return result


def test_checked_record_is_exact_canonical_custody_only_gap() -> None:
    gap = subject.build_typed_custody_gap(_verified_capability())
    loaded = subject.load_typed_custody_gap(RECORD)
    assert loaded == gap
    assert RECORD.read_bytes() == subject.typed_custody_gap_bytes(gap)
    data = loaded.to_data()
    assert data["schema"] == subject.GAP_SCHEMA
    assert data["outcome"] == "typed_custody_gap"
    assert data["gap_domain"] == "custody"
    assert data["terminal_stage"] == (
        "after_tombstone_persistence_before_campaign_execution"
    )
    assert data["historical_official_program_exposure_acknowledged"] is True
    assert data["persistence_receipt_record_digest"] == (
        subject.PINNED_PERSISTENCE_RECEIPT_RECORD_DIGEST
    )
    assert data["persistence_receipt_file_sha256"] == (
        subject.PINNED_PERSISTENCE_RECEIPT_FILE_SHA256
    )
    assert data["serialized_persistence_receipt_is_authority"] is False
    assert data["fresh_verified_persistence_capability_required"] is True


def test_gap_explicitly_terminates_before_semantic_version_space() -> None:
    data = subject.build_typed_custody_gap(_verified_capability()).to_data()
    assert data["support_matrix_constructed"] is False
    assert data["typed_axis_inventory_constructed"] is False
    assert data["version_space_not_constructed"] is True
    assert data["version_space_digest"] is None
    assert data["evaluated_formula_count"] is None
    assert data["survivor_count"] is None
    assert data["semantic_empty_space_evidence"] is False
    assert data["semantic_empty_gap"] is None
    assert data["semantic_empty_gap_schema"] is None
    assert data["semantic_empty_reason_code"] is None
    assert data["frozen_python_predicate_constructed"] is False
    assert data["schema"] not in {
        "gkm.bongard-typed-axis-empty-gap.v5",
        "gkm.bongard-typed-axis-task-gap.v1",
        "gkm.bongard-typed-axis-task-gap.v2",
    }
    assert "inventory_address" not in data
    assert "support_matrix_address" not in data
    assert "admitted_formula_ids" not in data
    assert "rejected_formula_ids" not in data
    assert data.get("reason_code") != "no_formula_admitted"


def test_every_post_tombstone_call_and_authority_is_exactly_zero_or_false() -> None:
    data = subject.build_typed_custody_gap(_verified_capability()).to_data()
    assert data["call_counter_scope"] == (
        "post_tombstone_terminalization_before_campaign_execution"
    )
    for name in (
        "model_calls",
        "pixel_calls",
        "label_calls",
        "rank_calls",
        "query_calls",
        "formula_evaluation_calls",
    ):
        assert type(data[name]) is int
        assert data[name] == 0
    for name in (
        "calibration_pixels_authorized",
        "support_pixels_authorized",
        "target_pixels_authorized",
        "query_pixels_authorized",
        "official_test_pixels_authorized",
        "action_program_or_label_reads_authorized",
        "rank_authorized",
        "query_evaluation_authorized",
        "production_adapter_authorized",
        "benchmark_claim_authorized",
    ):
        assert type(data[name]) is bool
        assert data[name] is False


def test_serialized_receipt_or_unissued_value_cannot_build_gap() -> None:
    receipt = _live_receipt()
    with pytest.raises(
        subject.SkeletonGraphTypedCustodyGapError,
        match="verified persistence capability",
    ):
        subject.build_typed_custody_gap(receipt)  # type: ignore[arg-type]

    unissued = persistence_api.SkeletonGraphVerifiedCustodyIncidentPersistence()
    with pytest.raises(
        subject.SkeletonGraphTypedCustodyGapError,
        match="verified persistence capability",
    ):
        subject.build_typed_custody_gap(unissued)

    class CapabilitySubclass(
        persistence_api.SkeletonGraphVerifiedCustodyIncidentPersistence
    ):
        pass

    forged = object.__new__(CapabilitySubclass)
    object.__setattr__(forged, "receipt", receipt)
    object.__setattr__(
        forged,
        "_issuance_token",
        persistence_api._VERIFIED_ISSUANCE_TOKEN,
    )
    with pytest.raises(
        subject.SkeletonGraphTypedCustodyGapError,
        match="verified persistence capability",
    ):
        subject.build_typed_custody_gap(forged)


def test_fresh_token_cannot_convert_a_fully_resealed_receipt_fork() -> None:
    changed = _live_receipt().to_data()
    changed["claim_record_digest"] = "sha256:" + "9" * 64
    content = dict(changed)
    content.pop("record_digest")
    changed["record_digest"] = "sha256:" + canonical_digest(content)
    forked_receipt = (
        persistence_api.SkeletonGraphCustodyIncidentPersistenceReceipt.from_data(
            changed
        )
    )
    forged = object.__new__(
        persistence_api.SkeletonGraphVerifiedCustodyIncidentPersistence
    )
    object.__setattr__(forged, "receipt", forked_receipt)
    object.__setattr__(
        forged,
        "_issuance_token",
        persistence_api._VERIFIED_ISSUANCE_TOKEN,
    )
    with pytest.raises(
        subject.SkeletonGraphTypedCustodyGapError,
        match="fixed live tombstone",
    ):
        subject.build_typed_custody_gap(forged)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("model_calls", False),
        ("pixel_calls", 0.0),
        ("label_calls", 1),
        ("rank_calls", False),
        ("query_calls", 1),
        ("version_space_not_constructed", False),
        ("version_space_digest", "sha256:" + "0" * 64),
        ("evaluated_formula_count", 0),
        ("survivor_count", 0),
        ("semantic_empty_space_evidence", True),
        ("semantic_empty_gap", {}),
        ("target_pixels_authorized", True),
        ("successor_ledger_digest", "sha256:" + "1" * 64),
        ("persistence_receipt_record_digest", "sha256:" + "2" * 64),
    ),
)
def test_parser_rejects_fully_resealed_policy_and_binding_mutations(
    field: str,
    replacement: object,
) -> None:
    value = subject.build_typed_custody_gap(_verified_capability()).to_data()
    value[field] = replacement
    with pytest.raises(subject.SkeletonGraphTypedCustodyGapError):
        subject.SkeletonGraphTypedCustodyGap.from_data(_reseal(value))


def test_parser_and_verifier_reject_subclasses_and_noncanonical_files(
    tmp_path: Path,
) -> None:
    gap = subject.build_typed_custody_gap(_verified_capability())

    class Text(str):
        pass

    changed = gap.to_data()
    changed["schema"] = Text(subject.GAP_SCHEMA)
    with pytest.raises(subject.SkeletonGraphTypedCustodyGapError):
        subject.SkeletonGraphTypedCustodyGap.from_data(changed)

    class Integer(int):
        pass

    for field, replacement in (
        ("reason_code", Text(gap.to_data()["reason_code"])),
        ("incident_event_sequence", Integer(158)),
        ("model_calls", Integer(0)),
    ):
        changed_leaf = gap.to_data()
        changed_leaf[field] = replacement
        with pytest.raises(
            subject.SkeletonGraphTypedCustodyGapError,
            match="leaf types",
        ):
            subject.SkeletonGraphTypedCustodyGap.from_data(changed_leaf)

    changed_key = {
        (Text(key) if key == "schema" else key): value
        for key, value in gap.to_data().items()
    }
    with pytest.raises(subject.SkeletonGraphTypedCustodyGapError):
        subject.SkeletonGraphTypedCustodyGap.from_data(changed_key)

    class GapSubclass(subject.SkeletonGraphTypedCustodyGap):
        pass

    with pytest.raises(subject.SkeletonGraphTypedCustodyGapError):
        GapSubclass(record_digest=gap.record_digest)
    with pytest.raises(subject.SkeletonGraphTypedCustodyGapError):
        GapSubclass.from_data(gap.to_data())

    with pytest.raises(FrozenInstanceError):
        gap.record_digest = "sha256:" + "0" * 64  # type: ignore[misc]

    noncanonical = tmp_path / "gap.json"
    noncanonical.write_text(
        json.dumps(gap.to_data(), indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        subject.SkeletonGraphTypedCustodyGapError,
        match="not canonical",
    ):
        subject.load_typed_custody_gap(noncanonical)


def test_cold_replay_never_calls_store_or_campaign_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = _verified_capability()
    gap = subject.build_typed_custody_gap(capability)
    calls: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        calls.append("forbidden")
        raise AssertionError("cold replay touched the persistence store")

    monkeypatch.setattr(persistence_api, "_verify_at_repository_root", forbidden)
    monkeypatch.setattr(persistence_api, "_persist_at_repository_root", forbidden)
    monkeypatch.setattr(
        persistence_api,
        "verify_persisted_custody_incident_tombstone",
        forbidden,
    )
    monkeypatch.setattr(
        persistence_api,
        "persist_custody_incident_tombstone",
        forbidden,
    )
    replayed = subject.cold_replay_typed_custody_gap(
        gap,
        verified_persistence=capability,
        expected_record_digest=gap.record_digest,
    )
    assert replayed == gap
    assert calls == []
    with pytest.raises(
        subject.SkeletonGraphTypedCustodyGapError,
        match="cold replay differs",
    ):
        subject.cold_replay_typed_custody_gap(
            gap,
            verified_persistence=capability,
            expected_record_digest="sha256:" + "f" * 64,
        )


def test_gap_module_has_no_semantic_or_campaign_execution_dependency() -> None:
    source = SUBJECT.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SUBJECT))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    forbidden = {
        "bongard.panel_typed_axis_slate_v2",
        "bongard.panel_typed_axis_custody_v2",
        "bongard.panel_typed_axis_task_runner",
        "bongard.panel_typed_axis_task_runner_v2",
        "bongard.panel_action_count_skeleton_graph_inference_custody",
        "bongard.panel_action_count_skeleton_graph_calibration_runner",
    }
    assert imported.isdisjoint(forbidden)
    assert "TypedEmptyGap" not in source
    assert "TypedAxisTaskGap" not in source
    assert "TypedAxisInventory" not in source
    assert "rank_transport" not in inspect.signature(
        subject.cold_replay_typed_custody_gap
    ).parameters
    assert "path" not in inspect.signature(
        subject.cold_replay_typed_custody_gap
    ).parameters


def test_gap_record_digest_and_bytes_are_stable_under_replay() -> None:
    capability = _verified_capability()
    gap = subject.build_typed_custody_gap(capability)
    restored = subject.SkeletonGraphTypedCustodyGap.from_data(
        json.loads(subject.typed_custody_gap_bytes(gap))
    )
    subject.verify_typed_custody_gap(
        restored,
        verified_persistence=capability,
    )
    assert restored == gap
    assert restored.file_sha256 == (
        "sha256:"
        + hashlib.sha256(
            canonical_json(restored.to_data()) + b"\n"
        ).hexdigest()
    )

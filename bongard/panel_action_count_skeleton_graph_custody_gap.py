"""Terminal typed custody GAP for the abandoned skeleton calibration campaign.

This is a custody outcome, not semantic empty-version-space evidence.  It can
be issued only from the fresh, process-local capability returned after the
fixed incident tombstone has been verified in its write-once store.  Building
and cold-replaying the GAP use canonical metadata only: no official archive,
PNG, action program, label, model, ranker, or query evaluator is accepted or
opened here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Final, Mapping

from bongard.canonical import canonical_digest, canonical_json
from bongard.runtime_source_snapshot import capture_loaded_source
from bongard import (
    panel_action_count_skeleton_graph_custody_incident_persistence
    as persistence_api,
)


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

GAP_SCHEMA: Final = "gkm.bongard-skeleton-graph-typed-custody-gap.v1"
GAP_RECORD_PATH: Final = (
    Path(__file__).resolve().parent
    / "data/panel_action_count_skeleton_graph_custody_gap_20260810_v1.json"
)
PERSISTENCE_COMMIT: Final = "d0bd18a2b2a30501ebaffda4af1c2c65b56322e5"
PINNED_PERSISTENCE_SOURCE_SHA256: Final = (
    "620d528d5c937a941857208faabd799d23ad62fee4f19a287a35e1b346248c10"
)
PINNED_PERSISTENCE_RECEIPT_RECORD_DIGEST: Final = (
    "sha256:b9275ea06c71b0967b5929d4631b61f94a252493d6dc1bfda5a7f392318854cd"
)
PINNED_PERSISTENCE_RECEIPT_FILE_SHA256: Final = (
    "sha256:e5b6ed77ff3f36ba03253753e96e55f658cb0a5b3de073398c21b462823f1836"
)
PINNED_INCIDENT_RECORD_DIGEST: Final = (
    "sha256:c647b0929a524a3fec64f74afbda1d1f469e6cf4ba1b8d6da1de788f0af2801f"
)
PINNED_INCIDENT_FILE_SHA256: Final = (
    "sha256:0f076190b70cf320f999a959640c20aa2bd8fda89131a36b175a0c80d62dcd7b"
)
PINNED_PREDECESSOR_LEDGER_DIGEST: Final = (
    "sha256:6995ea9cfda2f384cb0ba1b1cdc3611c965227c60fdb281d1e2e56fffa357b56"
)
PINNED_PREDECESSOR_FILE_SHA256: Final = (
    "sha256:8c5034e77f769a67b1bc16b41881e14887592e070e730d062049ea33e1467ff8"
)
PINNED_PREDECESSOR_CORPUS_DIGEST: Final = (
    "sha256:6fa51548520190a412812ba8f872dc3c7a7a2b2c47c0e42a4d9f6df351dce138"
)
PINNED_CLAIM_RECORD_DIGEST: Final = (
    "sha256:a29a4e239b3bc77deac01e550d746524599257d0bf139241905bfa5c923d23f7"
)
PINNED_CLAIM_FILE_SHA256: Final = (
    "sha256:aaf72f92eddd88b5547b46055193ea853142d8fb916898f80fc188725fa76855"
)
PINNED_SUCCESSOR_LEDGER_DIGEST: Final = (
    "sha256:63f3a24b32191985f0733af2934493165d2838fe110a1d412d6d779a017dcb03"
)
PINNED_SUCCESSOR_FILE_SHA256: Final = (
    "sha256:de301f9b9cbcedc4f3e7420cf2c0fd43a9e73584ac71321308c47cbc93e794f5"
)
PINNED_INCIDENT_EVENT_DIGEST: Final = (
    "sha256:5eb6ac37571e3b6883e88c1e890f70236eeebc0e8b1263fcf135e8f154cb6e53"
)
PINNED_INCIDENT_EVENT_OBSERVED_AT: Final = "2026-08-10T14:13:42Z"

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_GAP_RECORD_BYTES: Final = 64 << 10


class SkeletonGraphTypedCustodyGapError(RuntimeError):
    """The terminal custody evidence or its verified persistence differs."""


def source_sha256() -> str:
    current = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if current != _LOADED_SOURCE_SHA256:
        raise SkeletonGraphTypedCustodyGapError(
            "typed-custody-gap source changed after import"
        )
    return current


def _address(value: object, *, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise SkeletonGraphTypedCustodyGapError(
            f"{label} is not a SHA-256 address"
        )
    return value


def _record_digest(content: Mapping[str, Any]) -> str:
    return "sha256:" + canonical_digest(content)


def _canonical_record_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json(value) + b"\n"


def _exact_typed_equal(actual: object, expected: object) -> bool:
    """Compare canonical leaves without allowing subclass aliases."""

    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        actual_dict = actual
        expected_dict = expected
        if (
            any(type(key) is not str for key in actual_dict)
            or set(actual_dict) != set(expected_dict)
        ):
            return False
        return all(
            _exact_typed_equal(actual_dict[key], expected_dict[key])
            for key in expected_dict
        )
    if type(expected) in (list, tuple):
        return len(actual) == len(expected) and all(
            _exact_typed_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _frozen_content() -> dict[str, Any]:
    return {
        "schema": GAP_SCHEMA,
        "outcome": "typed_custody_gap",
        "gap_domain": "custody",
        "reason_code": (
            "official_program_authority_preexposed_before_prediction_barrier"
        ),
        "terminal_stage": (
            "after_tombstone_persistence_before_campaign_execution"
        ),
        "historical_official_program_exposure_acknowledged": True,
        "incident_record_digest": PINNED_INCIDENT_RECORD_DIGEST,
        "incident_file_sha256": PINNED_INCIDENT_FILE_SHA256,
        "predecessor_ledger_digest": PINNED_PREDECESSOR_LEDGER_DIGEST,
        "predecessor_file_sha256": PINNED_PREDECESSOR_FILE_SHA256,
        "predecessor_corpus_digest": PINNED_PREDECESSOR_CORPUS_DIGEST,
        "tombstone_claim_schema": persistence_api.incident_api.TOMBSTONE_CLAIM_SCHEMA,
        "tombstone_claim_filename": (
            persistence_api.incident_api.CAMPAIGN_INTENT_FILENAME
        ),
        "tombstone_claim_record_digest": PINNED_CLAIM_RECORD_DIGEST,
        "tombstone_claim_file_sha256": PINNED_CLAIM_FILE_SHA256,
        "incident_event_digest": PINNED_INCIDENT_EVENT_DIGEST,
        "incident_event_observed_at": PINNED_INCIDENT_EVENT_OBSERVED_AT,
        "incident_event_sequence": 158,
        "successor_ledger_digest": PINNED_SUCCESSOR_LEDGER_DIGEST,
        "successor_file_sha256": PINNED_SUCCESSOR_FILE_SHA256,
        "successor_event_count": 159,
        "persistence_commit": PERSISTENCE_COMMIT,
        "persistence_source_sha256": (
            "sha256:" + PINNED_PERSISTENCE_SOURCE_SHA256
        ),
        "persistence_receipt_schema": persistence_api.RECEIPT_SCHEMA,
        "persistence_receipt_filename": persistence_api.RECEIPT_FILENAME,
        "persistence_receipt_record_digest": (
            PINNED_PERSISTENCE_RECEIPT_RECORD_DIGEST
        ),
        "persistence_receipt_file_sha256": (
            PINNED_PERSISTENCE_RECEIPT_FILE_SHA256
        ),
        "serialized_persistence_receipt_is_authority": False,
        "fresh_verified_persistence_capability_required": True,
        "call_counter_scope": (
            "post_tombstone_terminalization_before_campaign_execution"
        ),
        "model_calls": 0,
        "pixel_calls": 0,
        "label_calls": 0,
        "rank_calls": 0,
        "query_calls": 0,
        "formula_evaluation_calls": 0,
        "support_matrix_constructed": False,
        "typed_axis_inventory_constructed": False,
        "version_space_not_constructed": True,
        "version_space_digest": None,
        "evaluated_formula_count": None,
        "survivor_count": None,
        "semantic_empty_space_evidence": False,
        "semantic_empty_gap": None,
        "semantic_empty_gap_schema": None,
        "semantic_empty_reason_code": None,
        "frozen_python_predicate_constructed": False,
        "calibration_pixels_authorized": False,
        "support_pixels_authorized": False,
        "target_pixels_authorized": False,
        "query_pixels_authorized": False,
        "official_test_pixels_authorized": False,
        "action_program_or_label_reads_authorized": False,
        "rank_authorized": False,
        "query_evaluation_authorized": False,
        "production_adapter_authorized": False,
        "benchmark_claim_authorized": False,
        "gap_source_sha256": "sha256:" + source_sha256(),
    }


@dataclass(frozen=True, slots=True)
class SkeletonGraphTypedCustodyGap:
    """A terminal custody witness carrying no observation or formula claim."""

    record_digest: str

    def __post_init__(self) -> None:
        if type(self) is not SkeletonGraphTypedCustodyGap:
            raise SkeletonGraphTypedCustodyGapError(
                "typed custody GAP must have the exact record type"
            )
        _address(self.record_digest, label="typed custody GAP")
        if self.record_digest != _record_digest(_frozen_content()):
            raise SkeletonGraphTypedCustodyGapError(
                "typed custody GAP digest differs"
            )

    def content_data(self) -> dict[str, Any]:
        if type(self) is not SkeletonGraphTypedCustodyGap:
            raise SkeletonGraphTypedCustodyGapError(
                "typed custody GAP must have the exact record type"
            )
        return _frozen_content()

    def to_data(self) -> dict[str, Any]:
        return {**self.content_data(), "record_digest": self.record_digest}

    @property
    def file_sha256(self) -> str:
        return "sha256:" + hashlib.sha256(
            _canonical_record_bytes(self.to_data())
        ).hexdigest()

    @classmethod
    def from_data(cls, raw: object) -> "SkeletonGraphTypedCustodyGap":
        if cls is not SkeletonGraphTypedCustodyGap:
            raise SkeletonGraphTypedCustodyGapError(
                "typed custody GAP parser requires the exact record class"
            )
        expected_content = _frozen_content()
        expected_fields = set(expected_content) | {"record_digest"}
        if (
            type(raw) is not dict
            or any(type(key) is not str for key in raw)
            or set(raw) != expected_fields
            or type(raw.get("schema")) is not str
            or raw.get("schema") != GAP_SCHEMA
            or type(raw.get("record_digest")) is not str
        ):
            raise SkeletonGraphTypedCustodyGapError(
                "typed custody GAP fields differ"
            )
        expected = {
            **expected_content,
            "record_digest": _record_digest(expected_content),
        }
        if not _exact_typed_equal(raw, expected):
            raise SkeletonGraphTypedCustodyGapError(
                "typed custody GAP leaf types or values differ"
            )
        try:
            differs = canonical_json(raw) != canonical_json(expected)
        except (TypeError, ValueError) as exc:
            raise SkeletonGraphTypedCustodyGapError(
                "typed custody GAP is not canonical JSON"
            ) from exc
        if differs:
            raise SkeletonGraphTypedCustodyGapError(
                "typed custody GAP differs from the frozen terminal outcome"
            )
        return cls(record_digest=raw["record_digest"])


def _verified_live_receipt(
    verified_persistence: (
        persistence_api.SkeletonGraphVerifiedCustodyIncidentPersistence
    ),
) -> persistence_api.SkeletonGraphCustodyIncidentPersistenceReceipt:
    if type(verified_persistence) is not (
        persistence_api.SkeletonGraphVerifiedCustodyIncidentPersistence
    ):
        raise SkeletonGraphTypedCustodyGapError(
            "fresh exact verified persistence capability is required"
        )
    try:
        persistence_api._validate_verified_persistence(verified_persistence)
    except persistence_api.SkeletonGraphCustodyIncidentPersistenceError as exc:
        raise SkeletonGraphTypedCustodyGapError(
            "verified persistence capability differs"
        ) from exc
    receipt = verified_persistence.receipt
    if (
        type(receipt)
        is not persistence_api.SkeletonGraphCustodyIncidentPersistenceReceipt
        or receipt.record_digest != PINNED_PERSISTENCE_RECEIPT_RECORD_DIGEST
        or receipt.file_sha256 != PINNED_PERSISTENCE_RECEIPT_FILE_SHA256
        or receipt.incident_record_digest != PINNED_INCIDENT_RECORD_DIGEST
        or receipt.incident_file_sha256 != PINNED_INCIDENT_FILE_SHA256
        or receipt.predecessor_ledger_digest
        != PINNED_PREDECESSOR_LEDGER_DIGEST
        or receipt.predecessor_file_sha256 != PINNED_PREDECESSOR_FILE_SHA256
        or receipt.predecessor_corpus_digest != PINNED_PREDECESSOR_CORPUS_DIGEST
        or receipt.claim_record_digest != PINNED_CLAIM_RECORD_DIGEST
        or receipt.claim_file_sha256 != PINNED_CLAIM_FILE_SHA256
        or receipt.incident_event_digest != PINNED_INCIDENT_EVENT_DIGEST
        or receipt.incident_event_observed_at
        != PINNED_INCIDENT_EVENT_OBSERVED_AT
        or receipt.incident_event_sequence != 158
        or receipt.successor_ledger_digest != PINNED_SUCCESSOR_LEDGER_DIGEST
        or receipt.successor_file_sha256 != PINNED_SUCCESSOR_FILE_SHA256
        or receipt.successor_event_count != 159
        or receipt.persistence_source_sha256
        != "sha256:" + PINNED_PERSISTENCE_SOURCE_SHA256
        or receipt.serialized_receipt_is_authority is not False
        or receipt.fresh_store_verification_required is not True
        or receipt.calibration_pixels_authorized is not False
        or receipt.action_program_or_label_reads_authorized is not False
        or receipt.target_query_support_test_pixels_authorized is not False
        or receipt.benchmark_claim_authorized is not False
    ):
        raise SkeletonGraphTypedCustodyGapError(
            "verified persistence is not the fixed live tombstone"
        )
    return receipt


def _frozen_gap() -> SkeletonGraphTypedCustodyGap:
    content = _frozen_content()
    return SkeletonGraphTypedCustodyGap(
        record_digest=_record_digest(content)
    )


def build_typed_custody_gap(
    verified_persistence: (
        persistence_api.SkeletonGraphVerifiedCustodyIncidentPersistence
    ),
) -> SkeletonGraphTypedCustodyGap:
    """Issue the fixed terminal GAP only from the fresh verified capability."""

    _verified_live_receipt(verified_persistence)
    return _frozen_gap()


def verify_typed_custody_gap(
    gap: SkeletonGraphTypedCustodyGap,
    *,
    verified_persistence: (
        persistence_api.SkeletonGraphVerifiedCustodyIncidentPersistence
    ),
) -> None:
    if type(gap) is not SkeletonGraphTypedCustodyGap:
        raise SkeletonGraphTypedCustodyGapError(
            "typed custody GAP has the wrong exact type"
        )
    _verified_live_receipt(verified_persistence)
    restored = SkeletonGraphTypedCustodyGap.from_data(gap.to_data())
    if gap != restored or gap != _frozen_gap():
        raise SkeletonGraphTypedCustodyGapError(
            "typed custody GAP verification differs"
        )


def cold_replay_typed_custody_gap(
    gap: SkeletonGraphTypedCustodyGap,
    *,
    verified_persistence: (
        persistence_api.SkeletonGraphVerifiedCustodyIncidentPersistence
    ),
    expected_record_digest: str,
) -> SkeletonGraphTypedCustodyGap:
    """Rebuild the terminal metadata with zero campaign or store calls."""

    expected = _address(expected_record_digest, label="expected typed custody GAP")
    verify_typed_custody_gap(
        gap,
        verified_persistence=verified_persistence,
    )
    restored = SkeletonGraphTypedCustodyGap.from_data(gap.to_data())
    rebuilt = build_typed_custody_gap(verified_persistence)
    if (
        restored != gap
        or rebuilt != gap
        or gap.record_digest != expected
    ):
        raise SkeletonGraphTypedCustodyGapError(
            "typed custody GAP cold replay differs"
        )
    return restored


def typed_custody_gap_bytes(gap: SkeletonGraphTypedCustodyGap) -> bytes:
    if type(gap) is not SkeletonGraphTypedCustodyGap:
        raise SkeletonGraphTypedCustodyGapError(
            "typed custody GAP has the wrong exact type"
        )
    return _canonical_record_bytes(gap.to_data())


def load_typed_custody_gap(
    path: Path = GAP_RECORD_PATH,
) -> SkeletonGraphTypedCustodyGap:
    raw = Path(path).read_bytes()
    if not 0 < len(raw) <= _MAX_GAP_RECORD_BYTES:
        raise SkeletonGraphTypedCustodyGapError(
            "typed custody GAP file is not bounded"
        )
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkeletonGraphTypedCustodyGapError(
            "typed custody GAP file is invalid JSON"
        ) from exc
    gap = SkeletonGraphTypedCustodyGap.from_data(decoded)
    if raw != typed_custody_gap_bytes(gap):
        raise SkeletonGraphTypedCustodyGapError(
            "typed custody GAP file is not canonical"
        )
    return gap


__all__ = (
    "GAP_RECORD_PATH",
    "GAP_SCHEMA",
    "PINNED_PERSISTENCE_RECEIPT_FILE_SHA256",
    "PINNED_PERSISTENCE_RECEIPT_RECORD_DIGEST",
    "SkeletonGraphTypedCustodyGap",
    "SkeletonGraphTypedCustodyGapError",
    "build_typed_custody_gap",
    "cold_replay_typed_custody_gap",
    "load_typed_custody_gap",
    "source_sha256",
    "typed_custody_gap_bytes",
    "verify_typed_custody_gap",
)

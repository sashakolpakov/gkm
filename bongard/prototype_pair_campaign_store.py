"""Durable, write-once state for the prototype-pair engineering campaign.

This module owns persistence and call admission only.  It does not open panel
files, interpret visual data, or execute a model.  The canonical decision
authority is Python; Lean is neither imported nor required for replay.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import time
from typing import TYPE_CHECKING, Any, Iterator, Mapping

from bongard.canonical import canonical_digest, canonical_json
from bongard.exposure import ExposureLedger
from bongard.prototype_pair_cohort import PrototypePairCohortPlan
from bongard.prototype_pair_execution_precommit import (
    PrototypePairExecutionPrecommit,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID

if TYPE_CHECKING:
    from bongard.prototype_scene_headless_runner import (
        PrototypeSceneFreezeCommitReceipt,
    )


STORED_OBJECT_SCHEMA = "gkm.bongard-prototype-pair-stored-object.v1"
RELEASE_AUTHORIZATION_SCHEMA = (
    "gkm.bongard-prototype-pair-release-authorization.v1"
)
CALL_CLAIM_SCHEMA = "gkm.bongard-prototype-pair-call-claim.v1"
CALL_OUTCOME_SCHEMA = "gkm.bongard-prototype-pair-call-outcome.v1"
CALL_JOURNAL_SEAL_SCHEMA = "gkm.bongard-prototype-pair-call-journal-seal.v1"
STORE_PROTOCOL = (
    "private-exclusive-create-fsync-atomic-link-no-replace-"
    "fsync-directory-reload-exact-journal-seal/v3"
)
RELEASE_PHASE = "prototype_pair_selected_task_release"
RELEASE_PURPOSE = (
    "release exactly the 31 preselected exact-unused TRAIN task identities "
    "for prototype-conditioned targeted engineering"
)

MODEL_CALL_PHASES = frozenset(
    {
        "prototype_description_observed",
        "twenty_eight_calibration_scenes_released_and_observed",
        "twelve_support_scenes_released_and_observed",
        "headless_codex_candidate_ranked",
        "two_query_scenes_released_and_observed",
    }
)
TERMINAL_STATUSES = frozenset(
    {"success", "parser_error", "transport_error", "error"}
)

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_KIND = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}\Z")
_MAX_CANONICAL_BYTES = 64 * 1024 * 1024
_MAX_PRECOMMIT_BYTES = 8 * 1024 * 1024
_PUBLICATION_READ_ATTEMPTS = 100
_PUBLICATION_RETRY_SECONDS = 0.001

STORE_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class PrototypePairCampaignStoreError(RuntimeError):
    """A persistence, replay, or one-shot admission invariant failed."""


class PrototypePairCallAlreadyFinished(PrototypePairCampaignStoreError):
    """A one-shot claim already has its unique terminal outcome."""


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _bytes_address(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _require_address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise PrototypePairCampaignStoreError(f"{label} must be a sha256: address")
    return value


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or (
        _ADDRESS.fullmatch(value) is None and _RAW_DIGEST.fullmatch(value) is None
    ):
        raise PrototypePairCampaignStoreError(
            f"{label} must be lowercase SHA-256, raw or sha256:-addressed"
        )
    return value


def _require_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise PrototypePairCampaignStoreError(f"{label} is not a bounded identifier")
    return value


def _require_kind(value: object) -> str:
    if not isinstance(value, str) or _KIND.fullmatch(value) is None:
        raise PrototypePairCampaignStoreError("object kind is not a bounded slug")
    return value


def _require_text(value: object, label: str, *, limit: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8", errors="strict")) > limit
    ):
        raise PrototypePairCampaignStoreError(f"{label} must be bounded exact text")
    return value


def _mapping(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise PrototypePairCampaignStoreError(f"{label} must be an object")
    if set(value) != fields:
        raise PrototypePairCampaignStoreError(f"{label} fields differ from schema")
    return value


def _clone(value: object, label: str) -> Any:
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrototypePairCampaignStoreError(
            f"{label} is not finite canonical JSON"
        ) from exc


def _authority() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_defines_identity_or_decision": False,
        "lean_required_for_replay": False,
        "lean_removal_changes_decision": False,
    }


def campaign_store_algorithm_digest() -> str:
    return _address(
        {
            "schema": "gkm.bongard-prototype-pair-campaign-store-algorithm.v2",
            "source_sha256": STORE_SOURCE_SHA256,
            "persistence_protocol": STORE_PROTOCOL,
            "model_call_phases": sorted(MODEL_CALL_PHASES),
            "terminal_statuses": sorted(TERMINAL_STATUSES),
            "selected_task_count": 31,
            "release_phase": RELEASE_PHASE,
            "release_purpose": RELEASE_PURPOSE,
            "journal_policy": (
                "authorization-scoped-exclusive-lock;seal-exact-terminal-key-set;"
                "reject-new-claims-after-seal;enumerate-all-claims-and-outcomes"
            ),
            **_authority(),
        }
    )


def _record_content(data: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key != digest_field}


def _validate_external_record_digest(
    data: Mapping[str, Any], expected_record_digest: str
) -> None:
    expected = _require_digest(expected_record_digest, "expected object record digest")
    candidates = [
        key
        for key, item in data.items()
        if key.endswith("_digest") and item == expected
    ]
    valid: list[str] = []
    for field in candidates:
        computed_raw = canonical_digest(_record_content(data, field))
        computed = (
            "sha256:" + computed_raw if expected.startswith("sha256:") else computed_raw
        )
        if computed == expected:
            valid.append(field)
    if len(valid) != 1:
        raise PrototypePairCampaignStoreError(
            "canonical object does not expose one exact self-authenticating digest"
        )


@dataclass(frozen=True, slots=True)
class PrototypePairStoredObjectReceipt:
    object_kind: str
    object_record_digest: str
    canonical_bytes_digest: str
    canonical_byte_count: int
    relative_path: str
    persistence_protocol: str
    store_source_sha256: str
    record_digest: str

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": STORED_OBJECT_SCHEMA,
            "object_kind": self.object_kind,
            "object_record_digest": self.object_record_digest,
            "canonical_bytes_digest": self.canonical_bytes_digest,
            "canonical_byte_count": self.canonical_byte_count,
            "relative_path": self.relative_path,
            "persistence_protocol": self.persistence_protocol,
            "store_source_sha256": self.store_source_sha256,
            "store_algorithm_digest": campaign_store_algorithm_digest(),
            **_authority(),
        }

    def __post_init__(self) -> None:
        kind = _require_kind(self.object_kind)
        _require_digest(self.object_record_digest, "stored object record digest")
        content = _require_address(
            self.canonical_bytes_digest, "stored canonical bytes digest"
        )
        if (
            type(self.canonical_byte_count) is not int
            or not 0 < self.canonical_byte_count <= _MAX_CANONICAL_BYTES
            or self.relative_path
            != f"objects/{kind}/{content.removeprefix('sha256:')}.json"
            or self.persistence_protocol != STORE_PROTOCOL
            or self.store_source_sha256 != STORE_SOURCE_SHA256
            or self.record_digest != _address(self.content_dict())
        ):
            raise PrototypePairCampaignStoreError("stored object receipt differs")

    @classmethod
    def seal(
        cls, *, object_kind: str, object_record_digest: str, canonical_bytes: bytes
    ) -> "PrototypePairStoredObjectReceipt":
        kind = _require_kind(object_kind)
        digest = _bytes_address(canonical_bytes)
        values: dict[str, object] = {
            "object_kind": kind,
            "object_record_digest": _require_digest(
                object_record_digest, "stored object record digest"
            ),
            "canonical_bytes_digest": digest,
            "canonical_byte_count": len(canonical_bytes),
            "relative_path": (
                f"objects/{kind}/{digest.removeprefix('sha256:')}.json"
            ),
            "persistence_protocol": STORE_PROTOCOL,
            "store_source_sha256": STORE_SOURCE_SHA256,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, record_digest=_address(provisional.content_dict()))  # type: ignore[arg-type]

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypePairStoredObjectReceipt":
        raw = _mapping(
            value,
            {
                "schema", "object_kind", "object_record_digest",
                "canonical_bytes_digest", "canonical_byte_count", "relative_path",
                "persistence_protocol", "store_source_sha256",
                "store_algorithm_digest", *_authority(), "record_digest",
            },
            "stored object receipt",
        )
        if (
            raw["schema"] != STORED_OBJECT_SCHEMA
            or raw["store_algorithm_digest"] != campaign_store_algorithm_digest()
            or any(raw[key] != item for key, item in _authority().items())
        ):
            raise PrototypePairCampaignStoreError("stored object authority differs")
        result = cls(
            raw["object_kind"], raw["object_record_digest"],
            raw["canonical_bytes_digest"], raw["canonical_byte_count"],
            raw["relative_path"], raw["persistence_protocol"],
            raw["store_source_sha256"], raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise PrototypePairCampaignStoreError("stored object receipt is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class PrototypePairReleaseAuthorizationReceipt:
    plan_digest: str
    execution_precommit_digest: str
    execution_precommit_receipt: PrototypePairStoredObjectReceipt
    exposure_predecessor_digest: str
    exposure_successor_digest: str
    exposure_successor_receipt: PrototypePairStoredObjectReceipt
    exposure_event_digest: str
    selected_task_ids: tuple[str, ...]
    actor: str
    observed_at: str
    phase: str
    purpose: str
    record_digest: str

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": RELEASE_AUTHORIZATION_SCHEMA,
            "plan_digest": self.plan_digest,
            "execution_precommit_digest": self.execution_precommit_digest,
            "execution_precommit_receipt": self.execution_precommit_receipt.to_data(),
            "exposure_predecessor_digest": self.exposure_predecessor_digest,
            "exposure_successor_digest": self.exposure_successor_digest,
            "exposure_successor_receipt": self.exposure_successor_receipt.to_data(),
            "exposure_event_digest": self.exposure_event_digest,
            "selected_task_ids": list(self.selected_task_ids),
            "selected_task_count": 31,
            "actor": self.actor,
            "observed_at": self.observed_at,
            "phase": self.phase,
            "purpose": self.purpose,
            "predecessor_exact_unused_verified": True,
            "exactly_one_successor_event": True,
            "successor_persisted_and_reloaded_before_authorization": True,
            "store_source_sha256": STORE_SOURCE_SHA256,
            "store_algorithm_digest": campaign_store_algorithm_digest(),
            **_authority(),
        }

    def __post_init__(self) -> None:
        for name in (
            "plan_digest", "execution_precommit_digest",
            "exposure_predecessor_digest", "exposure_successor_digest",
            "exposure_event_digest", "record_digest",
        ):
            _require_address(getattr(self, name), name)
        _require_identifier(self.actor, "release actor")
        _require_text(self.observed_at, "release observed_at", limit=128)
        tasks = self.selected_task_ids
        if (
            len(tasks) != 31
            or tasks != tuple(sorted(set(tasks)))
            or any(_IDENTIFIER.fullmatch(item) is None for item in tasks)
            or self.phase != RELEASE_PHASE
            or self.purpose != RELEASE_PURPOSE
            or self.execution_precommit_receipt.object_kind
            != "execution_precommit"
            or self.execution_precommit_receipt.object_record_digest
            != self.execution_precommit_digest
            or self.exposure_successor_receipt.object_kind != "exposure_successor"
            or self.exposure_successor_receipt.object_record_digest
            != self.exposure_successor_digest
            or self.record_digest != _address(self.content_dict())
        ):
            raise PrototypePairCampaignStoreError("release authorization differs")

    @classmethod
    def seal(
        cls,
        *,
        plan_digest: str,
        execution_precommit_receipt: PrototypePairStoredObjectReceipt,
        exposure_predecessor_digest: str,
        exposure_successor_receipt: PrototypePairStoredObjectReceipt,
        exposure_event_digest: str,
        selected_task_ids: tuple[str, ...],
        actor: str,
        observed_at: str,
    ) -> "PrototypePairReleaseAuthorizationReceipt":
        values: dict[str, object] = {
            "plan_digest": plan_digest,
            "execution_precommit_digest": (
                execution_precommit_receipt.object_record_digest
            ),
            "execution_precommit_receipt": execution_precommit_receipt,
            "exposure_predecessor_digest": exposure_predecessor_digest,
            "exposure_successor_digest": (
                exposure_successor_receipt.object_record_digest
            ),
            "exposure_successor_receipt": exposure_successor_receipt,
            "exposure_event_digest": exposure_event_digest,
            "selected_task_ids": selected_task_ids,
            "actor": actor,
            "observed_at": observed_at,
            "phase": RELEASE_PHASE,
            "purpose": RELEASE_PURPOSE,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, record_digest=_address(provisional.content_dict()))  # type: ignore[arg-type]

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "PrototypePairReleaseAuthorizationReceipt":
        raw = _mapping(
            value,
            {
                "schema", "plan_digest", "execution_precommit_digest",
                "execution_precommit_receipt", "exposure_predecessor_digest",
                "exposure_successor_digest", "exposure_successor_receipt",
                "exposure_event_digest", "selected_task_ids",
                "selected_task_count", "actor", "observed_at", "phase",
                "purpose", "predecessor_exact_unused_verified",
                "exactly_one_successor_event",
                "successor_persisted_and_reloaded_before_authorization",
                "store_source_sha256", "store_algorithm_digest", *_authority(),
                "record_digest",
            },
            "release authorization",
        )
        if (
            raw["schema"] != RELEASE_AUTHORIZATION_SCHEMA
            or raw["selected_task_count"] != 31
            or raw["predecessor_exact_unused_verified"] is not True
            or raw["exactly_one_successor_event"] is not True
            or raw["successor_persisted_and_reloaded_before_authorization"] is not True
            or raw["store_source_sha256"] != STORE_SOURCE_SHA256
            or raw["store_algorithm_digest"] != campaign_store_algorithm_digest()
            or any(raw[key] != item for key, item in _authority().items())
            or not isinstance(raw["execution_precommit_receipt"], Mapping)
            or not isinstance(raw["exposure_successor_receipt"], Mapping)
            or not isinstance(raw["selected_task_ids"], list)
        ):
            raise PrototypePairCampaignStoreError("release authorization authority differs")
        result = cls(
            raw["plan_digest"], raw["execution_precommit_digest"],
            PrototypePairStoredObjectReceipt.from_data(
                raw["execution_precommit_receipt"]
            ),
            raw["exposure_predecessor_digest"], raw["exposure_successor_digest"],
            PrototypePairStoredObjectReceipt.from_data(
                raw["exposure_successor_receipt"]
            ),
            raw["exposure_event_digest"], tuple(raw["selected_task_ids"]),
            raw["actor"], raw["observed_at"], raw["phase"], raw["purpose"],
            raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise PrototypePairCampaignStoreError("release authorization is not canonical")
        return result


def _call_key_content(
    *, authorization_digest: str, phase: str, subject_id: str, context_digest: str
) -> dict[str, object]:
    return {
        "schema": "gkm.bongard-prototype-pair-call-key.v1",
        "authorization_digest": authorization_digest,
        "phase": phase,
        "subject_id": subject_id,
        "context_digest": context_digest,
    }


def _authorization_key_content(
    *, plan_digest: str, predecessor_digest: str
) -> dict[str, object]:
    return {
        "schema": "gkm.bongard-prototype-pair-authorization-root.v2",
        "plan_digest": plan_digest,
        "exposure_predecessor_digest": predecessor_digest,
        "exclusive_across_precommits_configurations_and_actors": True,
    }


@dataclass(frozen=True, slots=True)
class PrototypePairCallClaim:
    authorization_digest: str
    phase: str
    subject_id: str
    context_digest: str
    key_digest: str
    claimed_at: str
    record_digest: str

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": CALL_CLAIM_SCHEMA,
            "authorization_digest": self.authorization_digest,
            "phase": self.phase,
            "subject_id": self.subject_id,
            "context_digest": self.context_digest,
            "key_digest": self.key_digest,
            "claimed_at": self.claimed_at,
            "state": "claimed_nonterminal",
            "exclusive_before_transport": True,
            "store_source_sha256": STORE_SOURCE_SHA256,
            "store_algorithm_digest": campaign_store_algorithm_digest(),
            **_authority(),
        }

    def __post_init__(self) -> None:
        _require_address(self.authorization_digest, "claim authorization digest")
        _require_address(self.context_digest, "claim context digest")
        _require_address(self.key_digest, "call key digest")
        _require_address(self.record_digest, "call claim record digest")
        _require_identifier(self.subject_id, "call subject")
        _require_text(self.claimed_at, "claim timestamp", limit=128)
        if (
            self.phase not in MODEL_CALL_PHASES
            or self.key_digest
            != _address(
                _call_key_content(
                    authorization_digest=self.authorization_digest,
                    phase=self.phase,
                    subject_id=self.subject_id,
                    context_digest=self.context_digest,
                )
            )
            or self.record_digest != _address(self.content_dict())
        ):
            raise PrototypePairCampaignStoreError("call claim differs")

    @classmethod
    def seal(
        cls, *, authorization_digest: str, phase: str, subject_id: str,
        context_digest: str, claimed_at: str
    ) -> "PrototypePairCallClaim":
        key = _address(
            _call_key_content(
                authorization_digest=authorization_digest, phase=phase,
                subject_id=subject_id, context_digest=context_digest,
            )
        )
        values: dict[str, object] = {
            "authorization_digest": authorization_digest, "phase": phase,
            "subject_id": subject_id, "context_digest": context_digest,
            "key_digest": key, "claimed_at": claimed_at,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, record_digest=_address(provisional.content_dict()))  # type: ignore[arg-type]

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypePairCallClaim":
        fields = {
            "schema", "authorization_digest", "phase", "subject_id",
            "context_digest", "key_digest", "claimed_at", "state",
            "exclusive_before_transport", "store_source_sha256",
            "store_algorithm_digest", *_authority(), "record_digest",
        }
        raw = _mapping(value, fields, "call claim")
        if (
            raw["schema"] != CALL_CLAIM_SCHEMA
            or raw["state"] != "claimed_nonterminal"
            or raw["exclusive_before_transport"] is not True
            or raw["store_source_sha256"] != STORE_SOURCE_SHA256
            or raw["store_algorithm_digest"] != campaign_store_algorithm_digest()
            or any(raw[key] != item for key, item in _authority().items())
        ):
            raise PrototypePairCampaignStoreError("call claim authority differs")
        result = cls(
            raw["authorization_digest"], raw["phase"], raw["subject_id"],
            raw["context_digest"], raw["key_digest"], raw["claimed_at"],
            raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise PrototypePairCampaignStoreError("call claim is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class PrototypePairCallOutcome:
    claim_digest: str
    key_digest: str
    phase: str
    subject_id: str
    context_digest: str
    terminal_status: str
    result_digest: str
    result_receipt: PrototypePairStoredObjectReceipt
    finished_at: str
    record_digest: str

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": CALL_OUTCOME_SCHEMA,
            "claim_digest": self.claim_digest,
            "key_digest": self.key_digest,
            "phase": self.phase,
            "subject_id": self.subject_id,
            "context_digest": self.context_digest,
            "terminal_status": self.terminal_status,
            "result_digest": self.result_digest,
            "result_receipt": self.result_receipt.to_data(),
            "finished_at": self.finished_at,
            "state": "terminal",
            "exactly_one_terminal_outcome": True,
            "result_persisted_and_reloaded_before_terminal": True,
            "store_source_sha256": STORE_SOURCE_SHA256,
            "store_algorithm_digest": campaign_store_algorithm_digest(),
            **_authority(),
        }

    def __post_init__(self) -> None:
        for value, label in (
            (self.claim_digest, "outcome claim digest"),
            (self.key_digest, "outcome key digest"),
            (self.context_digest, "outcome context digest"),
            (self.record_digest, "outcome record digest"),
        ):
            _require_address(value, label)
        _require_digest(self.result_digest, "outcome result digest")
        _require_identifier(self.subject_id, "outcome subject")
        _require_text(self.finished_at, "outcome timestamp", limit=128)
        if (
            self.phase not in MODEL_CALL_PHASES
            or self.terminal_status not in TERMINAL_STATUSES
            or self.result_receipt.object_record_digest != self.result_digest
            or self.record_digest != _address(self.content_dict())
        ):
            raise PrototypePairCampaignStoreError("call outcome differs")

    @classmethod
    def seal(
        cls, *, claim: PrototypePairCallClaim, terminal_status: str,
        result_receipt: PrototypePairStoredObjectReceipt, finished_at: str
    ) -> "PrototypePairCallOutcome":
        values: dict[str, object] = {
            "claim_digest": claim.record_digest,
            "key_digest": claim.key_digest,
            "phase": claim.phase,
            "subject_id": claim.subject_id,
            "context_digest": claim.context_digest,
            "terminal_status": terminal_status,
            "result_digest": result_receipt.object_record_digest,
            "result_receipt": result_receipt,
            "finished_at": finished_at,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, record_digest=_address(provisional.content_dict()))  # type: ignore[arg-type]

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypePairCallOutcome":
        raw = _mapping(
            value,
            {
                "schema", "claim_digest", "key_digest", "phase", "subject_id",
                "context_digest", "terminal_status", "result_digest",
                "result_receipt", "finished_at", "state",
                "exactly_one_terminal_outcome",
                "result_persisted_and_reloaded_before_terminal",
                "store_source_sha256", "store_algorithm_digest", *_authority(),
                "record_digest",
            },
            "call outcome",
        )
        if (
            raw["schema"] != CALL_OUTCOME_SCHEMA
            or raw["state"] != "terminal"
            or raw["exactly_one_terminal_outcome"] is not True
            or raw["result_persisted_and_reloaded_before_terminal"] is not True
            or raw["store_source_sha256"] != STORE_SOURCE_SHA256
            or raw["store_algorithm_digest"] != campaign_store_algorithm_digest()
            or any(raw[key] != item for key, item in _authority().items())
            or not isinstance(raw["result_receipt"], Mapping)
        ):
            raise PrototypePairCampaignStoreError("call outcome authority differs")
        result = cls(
            raw["claim_digest"], raw["key_digest"], raw["phase"],
            raw["subject_id"], raw["context_digest"], raw["terminal_status"],
            raw["result_digest"],
            PrototypePairStoredObjectReceipt.from_data(raw["result_receipt"]),
            raw["finished_at"], raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise PrototypePairCampaignStoreError("call outcome is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class PrototypePairCallJournalSeal:
    authorization_digest: str
    entries: tuple[tuple[str, str, str], ...]
    sealed_at: str
    record_digest: str

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": CALL_JOURNAL_SEAL_SCHEMA,
            "authorization_digest": self.authorization_digest,
            "entries": [
                {
                    "key_digest": key_digest,
                    "claim_digest": claim_digest,
                    "outcome_digest": outcome_digest,
                }
                for key_digest, claim_digest, outcome_digest in self.entries
            ],
            "terminal_key_count": len(self.entries),
            "sealed_at": self.sealed_at,
            "state": "terminal",
            "prevents_new_claims": True,
            "all_authorized_claims_enumerated": True,
            "all_enumerated_claims_terminal": True,
            "store_source_sha256": STORE_SOURCE_SHA256,
            "store_algorithm_digest": campaign_store_algorithm_digest(),
            **_authority(),
        }

    def __post_init__(self) -> None:
        _require_address(self.authorization_digest, "journal authorization digest")
        _require_address(self.record_digest, "journal seal digest")
        _require_text(self.sealed_at, "journal seal timestamp", limit=128)
        if not isinstance(self.entries, tuple) or not self.entries:
            raise PrototypePairCampaignStoreError("journal seal entries differ")
        keys: list[str] = []
        for entry in self.entries:
            if not isinstance(entry, tuple) or len(entry) != 3:
                raise PrototypePairCampaignStoreError("journal seal entry differs")
            for value, label in zip(
                entry,
                ("journal key digest", "journal claim digest", "journal outcome digest"),
                strict=True,
            ):
                _require_address(value, label)
            keys.append(entry[0])
        if (
            tuple(keys) != tuple(sorted(set(keys)))
            or self.record_digest != _address(self.content_dict())
        ):
            raise PrototypePairCampaignStoreError("journal seal differs")

    @classmethod
    def seal(
        cls,
        *,
        authorization_digest: str,
        entries: tuple[tuple[str, str, str], ...],
        sealed_at: str,
    ) -> "PrototypePairCallJournalSeal":
        values: dict[str, object] = {
            "authorization_digest": authorization_digest,
            "entries": entries,
            "sealed_at": sealed_at,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, record_digest=_address(provisional.content_dict()))  # type: ignore[arg-type]

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "PrototypePairCallJournalSeal":
        raw = _mapping(
            value,
            {
                "schema", "authorization_digest", "entries",
                "terminal_key_count", "sealed_at", "state",
                "prevents_new_claims", "all_authorized_claims_enumerated",
                "all_enumerated_claims_terminal", "store_source_sha256",
                "store_algorithm_digest", *_authority(), "record_digest",
            },
            "call journal seal",
        )
        rows = raw["entries"]
        if not isinstance(rows, list):
            raise PrototypePairCampaignStoreError("journal seal entries differ")
        entries: list[tuple[str, str, str]] = []
        for row in rows:
            item = _mapping(
                row,
                {"key_digest", "claim_digest", "outcome_digest"},
                "journal seal entry",
            )
            entries.append(
                (item["key_digest"], item["claim_digest"], item["outcome_digest"])
            )
        if (
            raw["schema"] != CALL_JOURNAL_SEAL_SCHEMA
            or raw["terminal_key_count"] != len(entries)
            or raw["state"] != "terminal"
            or raw["prevents_new_claims"] is not True
            or raw["all_authorized_claims_enumerated"] is not True
            or raw["all_enumerated_claims_terminal"] is not True
            or raw["store_source_sha256"] != STORE_SOURCE_SHA256
            or raw["store_algorithm_digest"] != campaign_store_algorithm_digest()
            or any(raw[key] != item for key, item in _authority().items())
        ):
            raise PrototypePairCampaignStoreError("call journal seal authority differs")
        result = cls(
            raw["authorization_digest"],
            tuple(entries),
            raw["sealed_at"],
            raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise PrototypePairCampaignStoreError("call journal seal is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class PrototypePairCallAdmission:
    model_eligible: bool
    reason: str
    claim: PrototypePairCallClaim
    terminal_outcome: PrototypePairCallOutcome | None

    def __post_init__(self) -> None:
        allowed = {
            "new_exclusive_claim",
            "preexisting_nonterminal_claim",
            "preexisting_terminal_outcome",
        }
        if self.reason not in allowed:
            raise PrototypePairCampaignStoreError("call admission reason differs")
        if self.model_eligible is not (self.reason == "new_exclusive_claim"):
            raise PrototypePairCampaignStoreError("call eligibility differs")
        if (self.terminal_outcome is None) is not (
            self.reason != "preexisting_terminal_outcome"
        ):
            raise PrototypePairCampaignStoreError("call terminal state differs")
        if self.terminal_outcome is not None and (
            self.terminal_outcome.claim_digest != self.claim.record_digest
            or self.terminal_outcome.key_digest != self.claim.key_digest
        ):
            raise PrototypePairCampaignStoreError("call admission lineage differs")


def _descriptor_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
        value.st_size, value.st_mtime_ns, value.st_ctime_ns,
    )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISDIR(opened.st_mode):
                raise PrototypePairCampaignStoreError("store directory is not a directory")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise PrototypePairCampaignStoreError(
            f"cannot fsync store directory {path}"
        ) from exc


def _ensure_directory(path: Path) -> None:
    try:
        created = False
        try:
            os.mkdir(path, 0o700)
            created = True
        except FileExistsError:
            pass
        current = path.lstat()
        if stat.S_ISLNK(current.st_mode) or not stat.S_ISDIR(current.st_mode):
            raise PrototypePairCampaignStoreError(
                f"store path is not a real directory: {path}"
            )
        if created:
            _fsync_directory(path.parent)
        _fsync_directory(path)
    except OSError as exc:
        raise PrototypePairCampaignStoreError(
            f"cannot initialize store directory {path}"
        ) from exc


def _stable_read_once(
    path: Path, *, byte_limit: int = _MAX_CANONICAL_BYTES
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PrototypePairCampaignStoreError(f"cannot open stored object {path}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= byte_limit
        ):
            raise PrototypePairCampaignStoreError("stored object is not a bounded private file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise PrototypePairCampaignStoreError("stored object was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise PrototypePairCampaignStoreError("stored object grew while reading")
        after = os.fstat(descriptor)
        if _descriptor_identity(before) != _descriptor_identity(after):
            raise PrototypePairCampaignStoreError("stored object changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _stable_read(path: Path, *, byte_limit: int = _MAX_CANONICAL_BYTES) -> bytes:
    """Read a published file, tolerating only the atomic-link handoff window."""

    last_error: PrototypePairCampaignStoreError | None = None
    for attempt in range(_PUBLICATION_READ_ATTEMPTS):
        try:
            return _stable_read_once(path, byte_limit=byte_limit)
        except PrototypePairCampaignStoreError as exc:
            last_error = exc
            if not path.exists() and not path.is_symlink():
                break
            if attempt + 1 == _PUBLICATION_READ_ATTEMPTS:
                break
            time.sleep(_PUBLICATION_RETRY_SECONDS)
    assert last_error is not None
    raise last_error


def _write_once(path: Path, payload: bytes, *, allow_identical: bool) -> bool:
    if not 0 < len(payload) <= _MAX_CANONICAL_BYTES:
        raise PrototypePairCampaignStoreError("stored payload is outside its byte bound")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    temporary = path.parent / (
        f".{path.name}.{os.getpid()}.{secrets.token_hex(16)}.tmp"
    )
    try:
        descriptor = os.open(temporary, flags, 0o600)
    except OSError as exc:
        raise PrototypePairCampaignStoreError(
            f"cannot create private publication file for {path}"
        ) from exc
    try:
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise PrototypePairCampaignStoreError("short durable write")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, path, follow_symlinks=False)
            created = True
        except FileExistsError:
            created = False
        except OSError as exc:
            raise PrototypePairCampaignStoreError(
                f"cannot atomically publish stored object {path}"
            ) from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as exc:
            raise PrototypePairCampaignStoreError(
                "cannot remove private publication link"
            ) from exc
    _fsync_directory(path.parent)
    if not created and not allow_identical:
        raise FileExistsError(os.fspath(path))
    existing = _stable_read(path)
    if existing != payload:
        label = "stored object reload" if created else "content-addressed collision"
        raise PrototypePairCampaignStoreError(f"{label} differs")
    if not created:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(path.parent)
    return created


def _decode_canonical(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PrototypePairCampaignStoreError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, Mapping) or payload != canonical_json(value) + b"\n":
        raise PrototypePairCampaignStoreError(f"{label} bytes are not exact canonical JSON")
    return value


@dataclass(frozen=True, slots=True)
class PrototypePairCampaignStore:
    root: Path

    def __post_init__(self) -> None:
        if (
            not isinstance(self.root, Path)
            or not self.root.is_absolute()
            or self.root.is_symlink()
        ):
            raise PrototypePairCampaignStoreError("store root must be an absolute Path")
        for name in (
            "objects", "authorizations", "claims", "outcomes", "freeze_commits",
            "journal_seals", "journal_locks",
        ):
            path = self.root / name
            if not path.is_dir() or path.is_symlink():
                raise PrototypePairCampaignStoreError("store layout is incomplete")

    @classmethod
    def open(cls, root: str | Path) -> "PrototypePairCampaignStore":
        candidate = Path(os.path.abspath(os.fspath(root)))
        if candidate.is_symlink():
            raise PrototypePairCampaignStoreError("store root cannot be a symlink")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        _ensure_directory(candidate)
        for name in (
            "objects", "authorizations", "claims", "outcomes", "freeze_commits",
            "journal_seals", "journal_locks",
        ):
            _ensure_directory(candidate / name)
        return cls(candidate)

    def _object_directory(self, kind: str) -> Path:
        _ensure_directory(self.root)
        _ensure_directory(self.root / "objects")
        directory = self.root / "objects" / _require_kind(kind)
        _ensure_directory(directory)
        return directory

    def _fixed_directory(self, name: str) -> Path:
        if name not in {
            "authorizations", "claims", "outcomes", "freeze_commits",
            "journal_seals", "journal_locks",
        }:
            raise PrototypePairCampaignStoreError("unknown fixed store directory")
        _ensure_directory(self.root)
        directory = self.root / name
        _ensure_directory(directory)
        return directory

    def persist_canonical_object(
        self,
        kind: str,
        data: Mapping[str, Any],
        expected_record_digest: str,
    ) -> PrototypePairStoredObjectReceipt:
        """Persist one self-authenticating canonical object and cold-reload it."""

        cloned = _clone(data, "canonical object")
        if not isinstance(cloned, Mapping):
            raise PrototypePairCampaignStoreError("canonical object must be an object")
        _validate_external_record_digest(cloned, expected_record_digest)
        payload = canonical_json(cloned) + b"\n"
        receipt = PrototypePairStoredObjectReceipt.seal(
            object_kind=kind,
            object_record_digest=expected_record_digest,
            canonical_bytes=payload,
        )
        path = self.root / receipt.relative_path
        directory = self._object_directory(kind)
        if path.parent != directory:
            raise PrototypePairCampaignStoreError("stored object path escaped its kind")
        _write_once(path, payload, allow_identical=True)
        if self.load_canonical_object(receipt, expected_record_digest) != dict(cloned):
            raise PrototypePairCampaignStoreError("canonical object cold reload differs")
        return receipt

    def verify_stored_object_bytes(
        self,
        receipt: PrototypePairStoredObjectReceipt | Mapping[str, Any],
        expected_record_digest: str,
    ) -> bytes:
        restored = (
            receipt
            if isinstance(receipt, PrototypePairStoredObjectReceipt)
            else PrototypePairStoredObjectReceipt.from_data(receipt)
        )
        expected = _require_digest(expected_record_digest, "expected object record digest")
        if restored.object_record_digest != expected:
            raise PrototypePairCampaignStoreError("stored object external digest differs")
        directory = self._object_directory(restored.object_kind)
        path = self.root / restored.relative_path
        if path.parent != directory:
            raise PrototypePairCampaignStoreError("stored object path escaped its kind")
        payload = _stable_read(path)
        if (
            len(payload) != restored.canonical_byte_count
            or _bytes_address(payload) != restored.canonical_bytes_digest
        ):
            raise PrototypePairCampaignStoreError("stored canonical bytes differ from receipt")
        data = _decode_canonical(payload, "stored object")
        _validate_external_record_digest(data, expected)
        return payload

    def load_canonical_object(
        self,
        receipt: PrototypePairStoredObjectReceipt | Mapping[str, Any],
        expected_record_digest: str,
    ) -> Mapping[str, Any]:
        return _decode_canonical(
            self.verify_stored_object_bytes(receipt, expected_record_digest),
            "stored object",
        )

    def persist_artifact(
        self, kind: str, data: Mapping[str, Any], expected_record_digest: str
    ) -> PrototypePairStoredObjectReceipt:
        return self.persist_canonical_object(kind, data, expected_record_digest)

    def persist_released_panel(
        self, data: Mapping[str, Any], expected_record_digest: str
    ) -> PrototypePairStoredObjectReceipt:
        return self.persist_canonical_object(
            "released_panel", data, expected_record_digest
        )

    def persist_candidate_freeze(
        self, freeze_bytes: bytes, expected_record_digest: str | None = None
    ) -> "PrototypeSceneFreezeCommitReceipt":
        if not isinstance(freeze_bytes, bytes):
            raise TypeError("freeze_bytes must be bytes")
        data = _decode_canonical(freeze_bytes, "candidate freeze")
        from bongard.prototype_scene_headless_runner import (
            PrototypeSceneCandidateFreeze,
            PrototypeSceneFreezeCommitReceipt,
        )

        freeze = PrototypeSceneCandidateFreeze.from_data(data)
        expected = freeze.record_digest
        if expected_record_digest is not None and expected != _require_address(
            expected_record_digest, "expected candidate freeze digest"
        ):
            raise PrototypePairCampaignStoreError(
                "candidate freeze differs from external commitment"
            )
        freeze_receipt = self.persist_canonical_object(
            "candidate_freeze", data, expected
        )
        if self.verify_stored_object_bytes(freeze_receipt, expected) != freeze_bytes:
            raise PrototypePairCampaignStoreError("candidate freeze exact bytes differ")
        commit = PrototypeSceneFreezeCommitReceipt.seal(
            freeze,
            freeze_bytes,
            storage_id=freeze_receipt.record_digest,
        )
        commit_receipt = self.persist_canonical_object(
            "candidate_freeze_commit", commit.to_data(), commit.record_digest
        )
        bundle_body = {
            "schema": "gkm.bongard-prototype-pair-freeze-persistence.v1",
            "freeze_digest": freeze.record_digest,
            "freeze_storage_receipt": freeze_receipt.to_data(),
            "commit_digest": commit.record_digest,
            "commit_storage_receipt": commit_receipt.to_data(),
        }
        bundle = {**bundle_body, "record_digest": _address(bundle_body)}
        bundle_path = self._fixed_directory("freeze_commits") / (
            commit.record_digest.removeprefix("sha256:") + ".json"
        )
        _write_once(
            bundle_path, canonical_json(bundle) + b"\n", allow_identical=True
        )
        replayed, replay_freeze_receipt, replay_commit_receipt = (
            self.load_candidate_freeze_commit(commit.record_digest)
        )
        if (
            replayed != commit
            or replay_freeze_receipt != freeze_receipt
            or replay_commit_receipt != commit_receipt
        ):
            raise PrototypePairCampaignStoreError(
                "candidate freeze persistence cold replay differs"
            )
        return replayed

    def load_candidate_freeze_commit(
        self, expected_commit_digest: str
    ) -> tuple[
        "PrototypeSceneFreezeCommitReceipt",
        PrototypePairStoredObjectReceipt,
        PrototypePairStoredObjectReceipt,
    ]:
        from bongard.prototype_scene_headless_runner import (
            PrototypeSceneCandidateFreeze,
            PrototypeSceneFreezeCommitReceipt,
        )

        expected = _require_address(
            expected_commit_digest, "expected candidate freeze commit digest"
        )
        path = self._fixed_directory("freeze_commits") / (
            expected.removeprefix("sha256:") + ".json"
        )
        raw = _mapping(
            _decode_canonical(_stable_read(path), "freeze persistence"),
            {
                "schema", "freeze_digest", "freeze_storage_receipt",
                "commit_digest", "commit_storage_receipt", "record_digest",
            },
            "freeze persistence",
        )
        body = {key: value for key, value in raw.items() if key != "record_digest"}
        if (
            raw["schema"] != "gkm.bongard-prototype-pair-freeze-persistence.v1"
            or raw["commit_digest"] != expected
            or raw["record_digest"] != _address(body)
            or not isinstance(raw["freeze_storage_receipt"], Mapping)
            or not isinstance(raw["commit_storage_receipt"], Mapping)
        ):
            raise PrototypePairCampaignStoreError("freeze persistence differs")
        freeze_receipt = PrototypePairStoredObjectReceipt.from_data(
            raw["freeze_storage_receipt"]
        )
        commit_receipt = PrototypePairStoredObjectReceipt.from_data(
            raw["commit_storage_receipt"]
        )
        if (
            freeze_receipt.object_kind != "candidate_freeze"
            or freeze_receipt.object_record_digest != raw["freeze_digest"]
            or commit_receipt.object_kind != "candidate_freeze_commit"
            or commit_receipt.object_record_digest != expected
        ):
            raise PrototypePairCampaignStoreError("freeze storage receipts differ")
        freeze_bytes = self.verify_stored_object_bytes(
            freeze_receipt, raw["freeze_digest"]
        )
        freeze = PrototypeSceneCandidateFreeze.from_data(
            _decode_canonical(freeze_bytes, "candidate freeze")
        )
        commit = PrototypeSceneFreezeCommitReceipt.from_data(
            self.load_canonical_object(commit_receipt, expected)
        )
        if commit.storage_id != freeze_receipt.record_digest:
            raise PrototypePairCampaignStoreError(
                "freeze commit storage ID does not bind its durable receipt"
            )
        commit.assert_matches(freeze, freeze_bytes)
        return commit, freeze_receipt, commit_receipt

    def load_candidate_freeze_persistence(
        self, expected_commit_digest: str
    ) -> tuple[
        "PrototypeSceneFreezeCommitReceipt",
        PrototypePairStoredObjectReceipt,
        PrototypePairStoredObjectReceipt,
    ]:
        return self.load_candidate_freeze_commit(expected_commit_digest)

    def persist_execution_precommit(
        self, precommit_bytes: bytes, expected_precommit_digest: str
    ) -> PrototypePairStoredObjectReceipt:
        if (
            not isinstance(precommit_bytes, bytes)
            or not 0 < len(precommit_bytes) <= _MAX_PRECOMMIT_BYTES
        ):
            raise PrototypePairCampaignStoreError(
                "execution precommit bytes are outside their bound"
            )
        expected = _require_address(
            expected_precommit_digest, "expected execution precommit digest"
        )
        data = _decode_canonical(precommit_bytes, "execution precommit")
        restored = PrototypePairExecutionPrecommit.from_data(data)
        if restored.record_digest != expected:
            raise PrototypePairCampaignStoreError(
                "execution precommit differs from external commitment"
            )
        receipt = self.persist_canonical_object(
            "execution_precommit", data, expected
        )
        if self.verify_execution_precommit(receipt, expected) != precommit_bytes:
            raise PrototypePairCampaignStoreError(
                "execution precommit exact reload differs"
            )
        return receipt

    def verify_execution_precommit(
        self,
        receipt: PrototypePairStoredObjectReceipt | Mapping[str, Any],
        expected_precommit_digest: str,
        expected_bytes: bytes | None = None,
    ) -> bytes:
        restored = (
            receipt
            if isinstance(receipt, PrototypePairStoredObjectReceipt)
            else PrototypePairStoredObjectReceipt.from_data(receipt)
        )
        if restored.object_kind != "execution_precommit":
            raise PrototypePairCampaignStoreError(
                "receipt does not name an execution precommit"
            )
        expected = _require_address(
            expected_precommit_digest, "expected execution precommit digest"
        )
        payload = self.verify_stored_object_bytes(restored, expected)
        parsed = PrototypePairExecutionPrecommit.from_data(
            _decode_canonical(payload, "execution precommit")
        )
        if parsed.record_digest != expected:
            raise PrototypePairCampaignStoreError(
                "stored execution precommit identity differs"
            )
        if expected_bytes is not None and payload != expected_bytes:
            raise PrototypePairCampaignStoreError(
                "stored execution precommit bytes differ from external bytes"
            )
        return payload

    def authorize_release(
        self,
        plan: PrototypePairCohortPlan,
        predecessor: ExposureLedger,
        execution_precommit_receipt: PrototypePairStoredObjectReceipt,
        *,
        expected_plan_digest: str,
        expected_execution_precommit_digest: str,
        expected_exposure_predecessor_digest: str,
        actor: str,
        observed_at: str,
    ) -> PrototypePairReleaseAuthorizationReceipt:
        """Persist/reload the sole 31-task successor before authorizing release."""

        if not isinstance(plan, PrototypePairCohortPlan):
            raise TypeError("plan must be PrototypePairCohortPlan")
        if not isinstance(predecessor, ExposureLedger):
            raise TypeError("predecessor must be ExposureLedger")
        plan_pin = _require_address(expected_plan_digest, "expected plan digest")
        precommit_pin = _require_address(
            expected_execution_precommit_digest,
            "expected execution precommit digest",
        )
        predecessor_pin = _require_address(
            expected_exposure_predecessor_digest,
            "expected exposure predecessor digest",
        )
        if (
            plan.record_digest != plan_pin
            or predecessor.digest != predecessor_pin
            or plan.exposure_predecessor_digest != predecessor_pin
        ):
            raise PrototypePairCampaignStoreError(
                "plan or predecessor differs from external commitment"
            )
        precommit_bytes = self.verify_execution_precommit(
            execution_precommit_receipt, precommit_pin
        )
        authorization_key_content = _authorization_key_content(
            plan_digest=plan_pin,
            predecessor_digest=predecessor_pin,
        )
        authorization_key = _address(authorization_key_content)
        key_claim_body = {
            **authorization_key_content,
            "authorization_key_digest": authorization_key,
            "state": "claimed_nonterminal",
            "exclusive_before_successor": True,
            "store_source_sha256": STORE_SOURCE_SHA256,
            "store_algorithm_digest": campaign_store_algorithm_digest(),
            **_authority(),
        }
        key_claim = {**key_claim_body, "record_digest": _address(key_claim_body)}
        key_claim_payload = canonical_json(key_claim) + b"\n"
        authorization_directory = self._fixed_directory("authorizations")
        key_claim_path = authorization_directory / (
            authorization_key.removeprefix("sha256:") + ".claim.json"
        )
        try:
            _write_once(key_claim_path, key_claim_payload, allow_identical=False)
        except FileExistsError:
            if _stable_read(key_claim_path) != key_claim_payload:
                raise PrototypePairCampaignStoreError(
                    "authorization key claim differs"
                )
            completion_path = authorization_directory / (
                authorization_key.removeprefix("sha256:") + ".complete.json"
            )
            if not completion_path.exists():
                raise PrototypePairCampaignStoreError(
                    "authorization key has a nonterminal durable claim"
                )
            completion = _decode_canonical(
                _stable_read(completion_path), "authorization completion"
            )
            completion_raw = _mapping(
                completion,
                {
                    "schema", "authorization_key_digest",
                    "authorization_claim_digest", "authorization_digest",
                    "state", "record_digest",
                },
                "authorization completion",
            )
            completion_body = {
                key: value for key, value in completion_raw.items()
                if key != "record_digest"
            }
            if (
                completion_raw["schema"]
                != "gkm.bongard-prototype-pair-authorization-completion.v1"
                or completion_raw["authorization_key_digest"] != authorization_key
                or completion_raw["authorization_claim_digest"]
                != key_claim["record_digest"]
                or completion_raw["state"] != "terminal"
                or completion_raw["record_digest"] != _address(completion_body)
            ):
                raise PrototypePairCampaignStoreError(
                    "authorization completion differs"
                )
            existing = self.load_release_authorization(
                completion_raw["authorization_digest"]
            )
            if (
                existing.plan_digest != plan_pin
                or existing.exposure_predecessor_digest != predecessor_pin
                or existing.execution_precommit_digest != precommit_pin
                or existing.actor != actor
            ):
                raise PrototypePairCampaignStoreError(
                    "plan/predecessor already authorized another precommit, "
                    "configuration, or actor"
                )
            return existing
        precommit = PrototypePairExecutionPrecommit.from_data(
            _decode_canonical(precommit_bytes, "execution precommit")
        )
        if (
            precommit.cohort_plan_digest != plan_pin
            or precommit.corpus_manifest_digest != plan.corpus_manifest_digest
            or precommit.identities.exposure_predecessor_digest != predecessor_pin
        ):
            raise PrototypePairCampaignStoreError(
                "execution precommit does not bind the pinned release inputs"
            )
        tasks = plan.selected_task_ids
        if len(tasks) != 31 or tasks != tuple(sorted(set(tasks))):
            raise PrototypePairCampaignStoreError(
                "release plan does not select exactly 31 unique tasks"
            )
        predecessor.assert_corpus(plan.corpus_manifest_digest)
        predecessor.assert_unseen(task_ids=tasks)
        successor = predecessor.record(
            phase=RELEASE_PHASE,
            actor=_require_identifier(actor, "release actor"),
            purpose=RELEASE_PURPOSE,
            task_ids=tasks,
            panel_ids=(),
            source=precommit_pin,
            observed_at=_require_text(observed_at, "release observed_at", limit=128),
            require_unseen=True,
        )
        if len(successor.events) != len(predecessor.events) + 1:
            raise PrototypePairCampaignStoreError("successor did not append exactly one event")
        event = successor.events[-1]
        expected_previous = predecessor.events[-1].digest if predecessor.events else None
        if (
            event.sequence != len(predecessor.events)
            or event.previous_digest != expected_previous
            or event.task_ids != tasks
            or event.panel_ids
            or event.phase != RELEASE_PHASE
            or event.purpose != RELEASE_PURPOSE
            or event.actor != actor
            or event.observed_at != observed_at
            or event.source != precommit_pin
        ):
            raise PrototypePairCampaignStoreError("successor exposure event differs")
        successor_receipt = self.persist_canonical_object(
            "exposure_successor", successor.to_dict(), successor.digest
        )
        reloaded_successor = ExposureLedger.from_dict(
            self.load_canonical_object(successor_receipt, successor.digest)
        )
        if reloaded_successor != successor or reloaded_successor.digest != successor.digest:
            raise PrototypePairCampaignStoreError("exposure successor cold reload differs")
        authorization = PrototypePairReleaseAuthorizationReceipt.seal(
            plan_digest=plan_pin,
            execution_precommit_receipt=execution_precommit_receipt,
            exposure_predecessor_digest=predecessor_pin,
            exposure_successor_receipt=successor_receipt,
            exposure_event_digest=event.digest,
            selected_task_ids=tasks,
            actor=actor,
            observed_at=observed_at,
        )
        payload = canonical_json(authorization.to_data()) + b"\n"
        path = authorization_directory / (
            authorization.record_digest.removeprefix("sha256:") + ".json"
        )
        _write_once(path, payload, allow_identical=True)
        reloaded = self.load_release_authorization(authorization.record_digest)
        if reloaded != authorization:
            raise PrototypePairCampaignStoreError(
                "release authorization cold reload differs"
            )
        completion_body = {
            "schema": "gkm.bongard-prototype-pair-authorization-completion.v1",
            "authorization_key_digest": authorization_key,
            "authorization_claim_digest": key_claim["record_digest"],
            "authorization_digest": reloaded.record_digest,
            "state": "terminal",
        }
        completion = {
            **completion_body,
            "record_digest": _address(completion_body),
        }
        completion_path = authorization_directory / (
            authorization_key.removeprefix("sha256:") + ".complete.json"
        )
        _write_once(
            completion_path,
            canonical_json(completion) + b"\n",
            allow_identical=False,
        )
        if _decode_canonical(
            _stable_read(completion_path), "authorization completion"
        ) != completion:
            raise PrototypePairCampaignStoreError(
                "authorization completion cold reload differs"
            )
        return reloaded

    def load_release_authorization(
        self, expected_authorization_digest: str
    ) -> PrototypePairReleaseAuthorizationReceipt:
        expected = _require_address(
            expected_authorization_digest, "expected authorization digest"
        )
        path = self._fixed_directory("authorizations") / (
            expected.removeprefix("sha256:") + ".json"
        )
        data = _decode_canonical(_stable_read(path), "release authorization")
        result = PrototypePairReleaseAuthorizationReceipt.from_data(data)
        if result.record_digest != expected:
            raise PrototypePairCampaignStoreError(
                "release authorization differs from external commitment"
            )
        self.verify_execution_precommit(
            result.execution_precommit_receipt,
            result.execution_precommit_digest,
        )
        successor = ExposureLedger.from_dict(
            self.load_canonical_object(
                result.exposure_successor_receipt,
                result.exposure_successor_digest,
            )
        )
        prefix = ExposureLedger(
            corpus_digest=successor.corpus_digest,
            events=successor.events[:-1],
        )
        if (
            successor.digest != result.exposure_successor_digest
            or not successor.events
            or prefix.digest != result.exposure_predecessor_digest
            or successor.events[-1].digest != result.exposure_event_digest
            or successor.events[-1].task_ids != result.selected_task_ids
            or successor.events[-1].panel_ids
            or successor.events[-1].phase != RELEASE_PHASE
            or successor.events[-1].purpose != RELEASE_PURPOSE
            or successor.events[-1].actor != result.actor
            or successor.events[-1].observed_at != result.observed_at
            or successor.events[-1].source != result.execution_precommit_digest
        ):
            raise PrototypePairCampaignStoreError(
                "release authorization successor evidence differs"
            )
        return result

    def _claim_path(self, key_digest: str) -> Path:
        return self._fixed_directory("claims") / (
            _require_address(key_digest, "call key digest").removeprefix("sha256:")
            + ".claim.json"
        )

    def _outcome_path(self, key_digest: str) -> Path:
        return self._fixed_directory("outcomes") / (
            _require_address(key_digest, "call key digest").removeprefix("sha256:")
            + ".outcome.json"
        )

    def _journal_seal_path(self, authorization_digest: str) -> Path:
        return self._fixed_directory("journal_seals") / (
            _require_address(
                authorization_digest, "journal authorization digest"
            ).removeprefix("sha256:")
            + ".journal-seal.json"
        )

    @contextmanager
    def _call_journal_lock(
        self, authorization_digest: str
    ) -> Iterator[None]:
        authorization = _require_address(
            authorization_digest, "journal authorization digest"
        )
        directory = self._fixed_directory("journal_locks")
        path = directory / (
            authorization.removeprefix("sha256:") + ".journal.lock"
        )
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise PrototypePairCampaignStoreError(
                "cannot open authorization journal lock"
            ) from exc
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                raise PrototypePairCampaignStoreError(
                    "authorization journal lock is not a private regular file"
                )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            _fsync_directory(directory)
            yield
        except OSError as exc:
            raise PrototypePairCampaignStoreError(
                "authorization journal lock failed"
            ) from exc
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _enumerate_call_journal_unlocked(
        self, authorization_digest: str
    ) -> tuple[tuple[PrototypePairCallClaim, PrototypePairCallOutcome | None], ...]:
        authorization = _require_address(
            authorization_digest, "journal authorization digest"
        )
        claim_by_key: dict[str, PrototypePairCallClaim] = {}
        claim_directory = self._fixed_directory("claims")
        for path in sorted(claim_directory.iterdir(), key=lambda item: item.name):
            match = re.fullmatch(r"([0-9a-f]{64})\.claim\.json", path.name)
            if match is None or path.is_symlink() or not path.is_file():
                raise PrototypePairCampaignStoreError(
                    "call claim directory contains an invalid entry"
                )
            key = "sha256:" + match.group(1)
            claim = self._load_claim(key)
            if key in claim_by_key:
                raise PrototypePairCampaignStoreError("call claim key repeats")
            claim_by_key[key] = claim

        outcome_keys: set[str] = set()
        outcome_directory = self._fixed_directory("outcomes")
        for path in sorted(outcome_directory.iterdir(), key=lambda item: item.name):
            match = re.fullmatch(r"([0-9a-f]{64})\.outcome\.json", path.name)
            if match is None or path.is_symlink() or not path.is_file():
                raise PrototypePairCampaignStoreError(
                    "call outcome directory contains an invalid entry"
                )
            key = "sha256:" + match.group(1)
            claim = claim_by_key.get(key)
            if claim is None:
                raise PrototypePairCampaignStoreError(
                    "call outcome has no durable claim"
                )
            if self.load_call_outcome(claim) is None:
                raise PrototypePairCampaignStoreError(
                    "enumerated call outcome disappeared"
                )
            outcome_keys.add(key)

        return tuple(
            (claim, self.load_call_outcome(claim))
            for key, claim in sorted(claim_by_key.items())
            if claim.authorization_digest == authorization
        )

    def enumerate_call_journal(
        self, authorization_digest: str
    ) -> tuple[tuple[PrototypePairCallClaim, PrototypePairCallOutcome | None], ...]:
        """Enumerate every durable claim and outcome for one authorization."""

        with self._call_journal_lock(authorization_digest):
            return self._enumerate_call_journal_unlocked(authorization_digest)

    def _load_claim(self, key_digest: str) -> PrototypePairCallClaim:
        data = _decode_canonical(_stable_read(self._claim_path(key_digest)), "call claim")
        result = PrototypePairCallClaim.from_data(data)
        if result.key_digest != key_digest:
            raise PrototypePairCampaignStoreError("call claim key differs from path")
        return result

    def load_call_journal(
        self,
        expected_key_digest: str,
        *,
        expected_authorization_digest: str | None = None,
    ) -> tuple[PrototypePairCallClaim, PrototypePairCallOutcome | None]:
        """Cold-load one exact claim and its optional unique terminal outcome."""

        key = _require_address(expected_key_digest, "expected call key digest")
        claim = self._load_claim(key)
        if expected_authorization_digest is not None and (
            claim.authorization_digest
            != _require_address(
                expected_authorization_digest,
                "expected call authorization digest",
            )
        ):
            raise PrototypePairCampaignStoreError(
                "call journal authorization differs"
            )
        return claim, self.load_call_outcome(claim)

    def _load_call_journal_seal_unlocked(
        self, authorization_digest: str
    ) -> PrototypePairCallJournalSeal | None:
        path = self._journal_seal_path(authorization_digest)
        try:
            payload = _stable_read(path)
        except PrototypePairCampaignStoreError as exc:
            if not path.exists():
                return None
            raise exc
        result = PrototypePairCallJournalSeal.from_data(
            _decode_canonical(payload, "call journal seal")
        )
        if result.authorization_digest != authorization_digest:
            raise PrototypePairCampaignStoreError(
                "call journal seal authorization differs"
            )
        return result

    @staticmethod
    def _expected_journal_keys(
        expected_terminal_key_digests: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not isinstance(expected_terminal_key_digests, tuple):
            raise PrototypePairCampaignStoreError(
                "expected terminal journal keys must be a tuple"
            )
        keys = tuple(
            _require_address(item, "expected terminal journal key")
            for item in expected_terminal_key_digests
        )
        if not keys or keys != tuple(sorted(set(keys))):
            raise PrototypePairCampaignStoreError(
                "expected terminal journal keys must be nonempty, unique, and sorted"
            )
        return keys

    def _verified_journal_entries_unlocked(
        self,
        authorization_digest: str,
        expected_terminal_key_digests: tuple[str, ...],
    ) -> tuple[tuple[str, str, str], ...]:
        expected = self._expected_journal_keys(expected_terminal_key_digests)
        journal = self._enumerate_call_journal_unlocked(authorization_digest)
        actual = tuple(claim.key_digest for claim, _outcome in journal)
        if actual != expected:
            raise PrototypePairCampaignStoreError(
                "authorization call journal key set differs"
            )
        if any(outcome is None for _claim, outcome in journal):
            raise PrototypePairCampaignStoreError(
                "authorization call journal contains a nonterminal claim"
            )
        return tuple(
            (claim.key_digest, claim.record_digest, outcome.record_digest)
            for claim, outcome in journal
            if outcome is not None
        )

    def seal_call_journal(
        self,
        authorization_digest: str,
        *,
        expected_terminal_key_digests: tuple[str, ...],
        sealed_at: str,
    ) -> PrototypePairCallJournalSeal:
        """Freeze the exact terminal journal and prevent every new call key."""

        authorization = _require_address(
            authorization_digest, "journal authorization digest"
        )
        self.load_release_authorization(authorization)
        with self._call_journal_lock(authorization):
            entries = self._verified_journal_entries_unlocked(
                authorization, expected_terminal_key_digests
            )
            existing = self._load_call_journal_seal_unlocked(authorization)
            if existing is not None:
                if existing.entries != entries:
                    raise PrototypePairCampaignStoreError(
                        "call journal differs from its terminal seal"
                    )
                return existing
            seal = PrototypePairCallJournalSeal.seal(
                authorization_digest=authorization,
                entries=entries,
                sealed_at=sealed_at,
            )
            _write_once(
                self._journal_seal_path(authorization),
                canonical_json(seal.to_data()) + b"\n",
                allow_identical=False,
            )
            reloaded = self._load_call_journal_seal_unlocked(authorization)
            if reloaded != seal:
                raise PrototypePairCampaignStoreError(
                    "call journal seal reload differs"
                )
            return seal

    def verify_call_journal_seal(
        self,
        authorization_digest: str,
        *,
        expected_terminal_key_digests: tuple[str, ...],
    ) -> PrototypePairCallJournalSeal:
        """Cold-verify a sealed journal against every store claim and outcome."""

        authorization = _require_address(
            authorization_digest, "journal authorization digest"
        )
        with self._call_journal_lock(authorization):
            seal = self._load_call_journal_seal_unlocked(authorization)
            if seal is None:
                raise PrototypePairCampaignStoreError(
                    "authorization call journal is not sealed"
                )
            entries = self._verified_journal_entries_unlocked(
                authorization, expected_terminal_key_digests
            )
            if seal.entries != entries:
                raise PrototypePairCampaignStoreError(
                    "call journal differs from its terminal seal"
                )
            return seal

    def claim_call(
        self,
        authorization: PrototypePairReleaseAuthorizationReceipt,
        *,
        phase: str,
        subject_id: str,
        context_digest: str,
        claimed_at: str,
    ) -> PrototypePairCallAdmission:
        """Acquire the sole durable claim before a caller may invoke transport."""

        if not isinstance(authorization, PrototypePairReleaseAuthorizationReceipt):
            raise TypeError("authorization must be a release authorization receipt")
        if self.load_release_authorization(authorization.record_digest) != authorization:
            raise PrototypePairCampaignStoreError("authorization cold replay differs")
        claim = PrototypePairCallClaim.seal(
            authorization_digest=authorization.record_digest,
            phase=phase,
            subject_id=subject_id,
            context_digest=context_digest,
            claimed_at=claimed_at,
        )
        payload = canonical_json(claim.to_data()) + b"\n"
        with self._call_journal_lock(authorization.record_digest):
            claim_path = self._claim_path(claim.key_digest)
            if claim_path.exists():
                existing = self._load_claim(claim.key_digest)
                if (
                    existing.authorization_digest != authorization.record_digest
                    or existing.phase != phase
                    or existing.subject_id != subject_id
                    or existing.context_digest != context_digest
                ):
                    raise PrototypePairCampaignStoreError(
                        "preexisting call claim key collides"
                    )
                outcome = self.load_call_outcome(existing)
                return PrototypePairCallAdmission(
                    model_eligible=False,
                    reason=(
                        "preexisting_nonterminal_claim"
                        if outcome is None
                        else "preexisting_terminal_outcome"
                    ),
                    claim=existing,
                    terminal_outcome=outcome,
                )
            if self._load_call_journal_seal_unlocked(
                authorization.record_digest
            ) is not None:
                raise PrototypePairCampaignStoreError(
                    "authorization call journal is sealed; new call key is forbidden"
                )
            try:
                _write_once(claim_path, payload, allow_identical=False)
            except FileExistsError as exc:
                raise PrototypePairCampaignStoreError(
                    "call claim appeared outside the authorization journal lock"
                ) from exc
            reloaded = self._load_claim(claim.key_digest)
            if reloaded != claim:
                raise PrototypePairCampaignStoreError(
                    "exclusive call claim reload differs"
                )
            return PrototypePairCallAdmission(
                model_eligible=True,
                reason="new_exclusive_claim",
                claim=reloaded,
                terminal_outcome=None,
            )

    def load_call_outcome(
        self, claim: PrototypePairCallClaim
    ) -> PrototypePairCallOutcome | None:
        if not isinstance(claim, PrototypePairCallClaim):
            raise TypeError("claim must be PrototypePairCallClaim")
        path = self._outcome_path(claim.key_digest)
        try:
            payload = _stable_read(path)
        except PrototypePairCampaignStoreError as exc:
            if not path.exists():
                return None
            raise exc
        result = PrototypePairCallOutcome.from_data(
            _decode_canonical(payload, "call outcome")
        )
        if (
            result.key_digest != claim.key_digest
            or result.claim_digest != claim.record_digest
            or result.phase != claim.phase
            or result.subject_id != claim.subject_id
            or result.context_digest != claim.context_digest
        ):
            raise PrototypePairCampaignStoreError("call outcome lineage differs")
        self.verify_stored_object_bytes(
            result.result_receipt, result.result_digest
        )
        return result

    def finish_call(
        self,
        claim: PrototypePairCallClaim,
        *,
        terminal_status: str,
        result_receipt: PrototypePairStoredObjectReceipt,
        finished_at: str,
        result_digest: str | None = None,
    ) -> PrototypePairCallOutcome:
        """Close one claim once, after its exact result is already durable."""

        if not isinstance(claim, PrototypePairCallClaim):
            raise TypeError("claim must be PrototypePairCallClaim")
        if not isinstance(result_receipt, PrototypePairStoredObjectReceipt):
            raise TypeError("result_receipt must be PrototypePairStoredObjectReceipt")
        if result_digest is not None and _require_digest(
            result_digest, "expected terminal result digest"
        ) != result_receipt.object_record_digest:
            raise PrototypePairCampaignStoreError(
                "terminal result receipt differs from explicit result digest"
            )
        with self._call_journal_lock(claim.authorization_digest):
            persisted_claim = self._load_claim(claim.key_digest)
            if persisted_claim != claim:
                raise PrototypePairCampaignStoreError(
                    "call claim differs from durable bytes"
                )
            if self._load_call_journal_seal_unlocked(
                claim.authorization_digest
            ) is not None:
                existing = self.load_call_outcome(claim)
                if existing is not None:
                    raise PrototypePairCallAlreadyFinished(
                        "call claim already has a terminal outcome; rerun is forbidden"
                    )
                raise PrototypePairCampaignStoreError(
                    "sealed authorization journal contains a nonterminal claim"
                )
            self.verify_stored_object_bytes(
                result_receipt, result_receipt.object_record_digest
            )
            outcome = PrototypePairCallOutcome.seal(
                claim=claim,
                terminal_status=terminal_status,
                result_receipt=result_receipt,
                finished_at=finished_at,
            )
            payload = canonical_json(outcome.to_data()) + b"\n"
            try:
                _write_once(
                    self._outcome_path(claim.key_digest), payload,
                    allow_identical=False,
                )
            except FileExistsError as exc:
                raise PrototypePairCallAlreadyFinished(
                    "call claim already has a terminal outcome; rerun is forbidden"
                ) from exc
            reloaded = self.load_call_outcome(claim)
            if reloaded != outcome:
                raise PrototypePairCampaignStoreError(
                    "terminal call outcome reload differs"
                )
            return outcome


__all__ = (
    "MODEL_CALL_PHASES",
    "RELEASE_PHASE",
    "RELEASE_PURPOSE",
    "STORE_PROTOCOL",
    "STORE_SOURCE_SHA256",
    "TERMINAL_STATUSES",
    "PrototypePairCallAdmission",
    "PrototypePairCallAlreadyFinished",
    "PrototypePairCallClaim",
    "PrototypePairCallJournalSeal",
    "PrototypePairCallOutcome",
    "PrototypePairCampaignStore",
    "PrototypePairCampaignStoreError",
    "PrototypePairReleaseAuthorizationReceipt",
    "PrototypePairStoredObjectReceipt",
    "campaign_store_algorithm_digest",
)

"""Headless packet runner for the calibrated two-tag prototype scene language.

The runner consumes complete, already-observed scene panel records.  It never
calls the scene observer itself.  A required external ``artifact_verifier``
must re-authenticate each archived observer binding against the exact PNG and
raw-artifact archive before support or cold replay is accepted.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.prototype_scene_calibration import PrototypeSceneCalibrationFamily
from bongard.prototype_scene_predicates import (
    PrototypeScenePanelEvaluation,
    PrototypeScenePredicateLibrary,
    PrototypeSceneVerifiedObserverBinding,
)
from bongard.prototype_scene_support_version_space import (
    PrototypeSceneCandidateResult,
    PrototypeSceneGapKind,
    PrototypeSceneSupportCandidate,
    PrototypeSceneSupportVersionSpace,
    PrototypeSceneVerifiedRanking,
    build_prototype_scene_support_version_space,
    complete_prototype_scene_candidates,
    evaluate_prototype_scene_candidate,
    rank_prototype_scene_survivors,
)


RUNNER_ID = "bongard.prototype-scene-headless/packet-core-v1"
RANK_RESPONSE_SCHEMA = "gkm.bongard-prototype-scene-rank-response.v1"
FREEZE_SCHEMA = "gkm.bongard-prototype-scene-candidate-freeze.v1"
FREEZE_COMMIT_SCHEMA = "gkm.bongard-prototype-scene-freeze-commit.v1"
ARCHIVE_SCHEMA = "gkm.bongard-prototype-scene-headless-archive.v1"

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}\Z")


class PrototypeSceneHeadlessError(RuntimeError):
    """A prototype-scene headless phase or archive failed closed."""


class PrototypeSceneHeadlessStatus(str, Enum):
    COMPLETE = "complete"
    LANGUAGE_GAP = "language_gap"
    WITNESS_GAP = "witness_gap"


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _require_address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise PrototypeSceneHeadlessError(f"{label} must be a sha256: address")
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise PrototypeSceneHeadlessError(f"{label} must be a bounded identifier")
    return value


def _fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PrototypeSceneHeadlessError(f"{label} fields differ from schema")


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_defines_identity_or_decision": False,
        "lean_required_for_replay": False,
        "lean_removal_changes_decision": False,
    }


def prototype_scene_runner_source_digest() -> str:
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def prototype_scene_rank_input_digest(
    *,
    library_digest: str,
    version_space_digest: str,
    survivor_candidate_ids: Sequence[str],
) -> str:
    return _address(
        {
            "schema": "gkm.bongard-prototype-scene-rank-input.v1",
            "library_digest": library_digest,
            "version_space_digest": version_space_digest,
            "survivor_candidate_ids": list(survivor_candidate_ids),
            "query_material_included": False,
            **_authority_data(),
        }
    )


def _rank_output_digest(ordered_ids: Sequence[str]) -> str:
    return _address(
        {
            "schema": "gkm.bongard-prototype-scene-rank-output.v1",
            "ordered_candidate_ids": list(ordered_ids),
        }
    )


@dataclass(frozen=True, slots=True)
class PrototypeSceneRankResponse:
    """Complete survivor permutation plus strict headless proposer provenance."""

    ordered_candidate_ids: tuple[str, ...]
    ranker_protocol_id: str
    ranker_protocol_digest: str
    model_id: str
    model_identity_digest: str
    environment_digest: str
    input_digest: str
    output_digest: str
    receipt: Mapping[str, Any]
    receipt_digest: str
    record_digest: str

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": RANK_RESPONSE_SCHEMA,
            "ordered_candidate_ids": list(self.ordered_candidate_ids),
            "ranker_protocol_id": self.ranker_protocol_id,
            "ranker_protocol_digest": self.ranker_protocol_digest,
            "model_id": self.model_id,
            "model_identity_digest": self.model_identity_digest,
            "environment_digest": self.environment_digest,
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
            "receipt": dict(self.receipt),
            "receipt_digest": self.receipt_digest,
            "complete_survivor_permutation_claimed": True,
            **_authority_data(),
        }

    def __post_init__(self) -> None:
        if not self.ordered_candidate_ids or len(set(self.ordered_candidate_ids)) != len(
            self.ordered_candidate_ids
        ):
            raise PrototypeSceneHeadlessError("rank response is not a permutation")
        _identifier(self.ranker_protocol_id, "ranker protocol_id")
        _identifier(self.model_id, "ranker model_id")
        for name in (
            "ranker_protocol_digest",
            "model_identity_digest",
            "environment_digest",
            "input_digest",
            "output_digest",
            "receipt_digest",
            "record_digest",
        ):
            _require_address(getattr(self, name), name)
        if not isinstance(self.receipt, Mapping) or any(
            not isinstance(key, str) for key in self.receipt
        ):
            raise PrototypeSceneHeadlessError("rank receipt must be an object")
        canonical_receipt = dict(self.receipt)
        object.__setattr__(self, "receipt", canonical_receipt)
        if set(canonical_receipt) != {"proposer_binding", "transport_receipt"}:
            raise PrototypeSceneHeadlessError("rank receipt envelope fields differ")
        binding = canonical_receipt["proposer_binding"]
        transport_receipt = canonical_receipt["transport_receipt"]
        expected_binding = {
            "ranker_protocol_id": self.ranker_protocol_id,
            "ranker_protocol_digest": self.ranker_protocol_digest,
            "model_id": self.model_id,
            "model_identity_digest": self.model_identity_digest,
            "environment_digest": self.environment_digest,
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
        }
        if (
            not isinstance(binding, Mapping)
            or dict(binding) != expected_binding
            or not isinstance(transport_receipt, Mapping)
            or not transport_receipt
        ):
            raise PrototypeSceneHeadlessError(
                "rank transport receipt does not bind proposer invocation"
            )
        if (
            self.output_digest != _rank_output_digest(self.ordered_candidate_ids)
            or self.receipt_digest
            != _address(
                {
                    "schema": "gkm.bongard-prototype-scene-rank-receipt.v1",
                    "receipt": canonical_receipt,
                }
            )
            or self.record_digest != _address(self.content_dict())
        ):
            raise PrototypeSceneHeadlessError("rank response provenance differs")

    @classmethod
    def seal(
        cls,
        *,
        ordered_candidate_ids: Sequence[str],
        ranker_protocol_id: str,
        ranker_protocol_digest: str,
        model_id: str,
        model_identity_digest: str,
        environment_digest: str,
        input_digest: str,
        receipt: Mapping[str, Any],
    ) -> "PrototypeSceneRankResponse":
        ordered = tuple(ordered_candidate_ids)
        output_digest = _rank_output_digest(ordered)
        receipt_envelope = {
            "proposer_binding": {
                "ranker_protocol_id": ranker_protocol_id,
                "ranker_protocol_digest": ranker_protocol_digest,
                "model_id": model_id,
                "model_identity_digest": model_identity_digest,
                "environment_digest": environment_digest,
                "input_digest": input_digest,
                "output_digest": output_digest,
            },
            "transport_receipt": dict(receipt),
        }
        receipt_digest = _address(
            {
                "schema": "gkm.bongard-prototype-scene-rank-receipt.v1",
                "receipt": receipt_envelope,
            }
        )
        values: dict[str, object] = {
            "ordered_candidate_ids": ordered,
            "ranker_protocol_id": ranker_protocol_id,
            "ranker_protocol_digest": ranker_protocol_digest,
            "model_id": model_id,
            "model_identity_digest": model_identity_digest,
            "environment_digest": environment_digest,
            "input_digest": input_digest,
            "output_digest": output_digest,
            "receipt": receipt_envelope,
            "receipt_digest": receipt_digest,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=_address(provisional.content_dict()),
        )

    def assert_matches(
        self,
        *,
        expected_input_digest: str,
        survivor_candidate_ids: Sequence[str],
    ) -> None:
        survivors = tuple(survivor_candidate_ids)
        if (
            self.input_digest != expected_input_digest
            or len(self.ordered_candidate_ids) != len(survivors)
            or set(self.ordered_candidate_ids) != set(survivors)
        ):
            raise PrototypeSceneHeadlessError(
                "rank response must be an exact survivor permutation"
            )

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypeSceneRankResponse":
        expected = {
            "schema",
            "ordered_candidate_ids",
            "ranker_protocol_id",
            "ranker_protocol_digest",
            "model_id",
            "model_identity_digest",
            "environment_digest",
            "input_digest",
            "output_digest",
            "receipt",
            "receipt_digest",
            "complete_survivor_permutation_claimed",
            *_authority_data(),
            "record_digest",
        }
        _fields(value, expected, "prototype-scene rank response")
        if not isinstance(value["receipt"], Mapping):
            raise PrototypeSceneHeadlessError("rank receipt is malformed")
        result = cls(
            ordered_candidate_ids=tuple(value["ordered_candidate_ids"]),
            ranker_protocol_id=value["ranker_protocol_id"],
            ranker_protocol_digest=value["ranker_protocol_digest"],
            model_id=value["model_id"],
            model_identity_digest=value["model_identity_digest"],
            environment_digest=value["environment_digest"],
            input_digest=value["input_digest"],
            output_digest=value["output_digest"],
            receipt=dict(value["receipt"]),
            receipt_digest=value["receipt_digest"],
            record_digest=value["record_digest"],
        )
        if result.to_data() != dict(value):
            raise PrototypeSceneHeadlessError("rank response is not canonical")
        return result


def _freeze_content(value: "PrototypeSceneCandidateFreeze") -> dict[str, object]:
    return {
        "schema": FREEZE_SCHEMA,
        "runner_id": RUNNER_ID,
        "runner_source_digest": value.runner_source_digest,
        "library_digest": value.library_digest,
        "calibration_family_digest": value.calibration_family_digest,
        "support_digest": value.support_digest,
        "version_space_digest": value.version_space_digest,
        "ranking_digest": value.ranking_digest,
        "rank_response_digest": value.rank_response_digest,
        "selected_candidate": value.selected_candidate.to_data(),
        "selected_candidate_digest": value.selected_candidate.record_digest,
        "query_panels_accepted": False,
        "negation_allowed": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PrototypeSceneCandidateFreeze:
    runner_source_digest: str
    library_digest: str
    calibration_family_digest: str
    support_digest: str
    version_space_digest: str
    ranking_digest: str
    rank_response_digest: str
    selected_candidate: PrototypeSceneSupportCandidate
    record_digest: str

    def __post_init__(self) -> None:
        for name in (
            "runner_source_digest",
            "library_digest",
            "calibration_family_digest",
            "support_digest",
            "version_space_digest",
            "ranking_digest",
            "rank_response_digest",
            "record_digest",
        ):
            _require_address(getattr(self, name), name)
        if (
            self.runner_source_digest != prototype_scene_runner_source_digest()
            or self.record_digest != _address(_freeze_content(self))
        ):
            raise PrototypeSceneHeadlessError("candidate freeze differs")

    @classmethod
    def seal(
        cls,
        *,
        library: PrototypeScenePredicateLibrary,
        family: PrototypeSceneCalibrationFamily,
        support_digest: str,
        version: PrototypeSceneSupportVersionSpace,
        ranking: PrototypeSceneVerifiedRanking,
        rank_response: PrototypeSceneRankResponse,
        selected_candidate: PrototypeSceneSupportCandidate,
    ) -> "PrototypeSceneCandidateFreeze":
        values: dict[str, object] = {
            "runner_source_digest": prototype_scene_runner_source_digest(),
            "library_digest": library.record_digest,
            "calibration_family_digest": family.record_digest,
            "support_digest": support_digest,
            "version_space_digest": version.record_digest,
            "ranking_digest": ranking.record_digest,
            "rank_response_digest": rank_response.record_digest,
            "selected_candidate": selected_candidate,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=_address(_freeze_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_freeze_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypeSceneCandidateFreeze":
        expected = {
            "schema",
            "runner_id",
            "runner_source_digest",
            "library_digest",
            "calibration_family_digest",
            "support_digest",
            "version_space_digest",
            "ranking_digest",
            "rank_response_digest",
            "selected_candidate",
            "selected_candidate_digest",
            "query_panels_accepted",
            "negation_allowed",
            *_authority_data(),
            "record_digest",
        }
        _fields(value, expected, "prototype-scene candidate freeze")
        if not isinstance(value["selected_candidate"], Mapping):
            raise PrototypeSceneHeadlessError("freeze candidate is malformed")
        candidate = PrototypeSceneSupportCandidate.from_data(value["selected_candidate"])
        result = cls(
            value["runner_source_digest"],
            value["library_digest"],
            value["calibration_family_digest"],
            value["support_digest"],
            value["version_space_digest"],
            value["ranking_digest"],
            value["rank_response_digest"],
            candidate,
            value["record_digest"],
        )
        if result.to_data() != dict(value):
            raise PrototypeSceneHeadlessError("candidate freeze is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class PrototypeSceneFreezeCommitReceipt:
    freeze_digest: str
    canonical_bytes_digest: str
    canonical_byte_count: int
    storage_id: str
    record_digest: str

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": FREEZE_COMMIT_SCHEMA,
            "freeze_digest": self.freeze_digest,
            "canonical_bytes_digest": self.canonical_bytes_digest,
            "canonical_byte_count": self.canonical_byte_count,
            "storage_id": self.storage_id,
            "durable_before_query": True,
            **_authority_data(),
        }

    def __post_init__(self) -> None:
        _require_address(self.freeze_digest, "freeze_digest")
        _require_address(self.canonical_bytes_digest, "canonical_bytes_digest")
        _identifier(self.storage_id, "storage_id")
        if (
            isinstance(self.canonical_byte_count, bool)
            or not isinstance(self.canonical_byte_count, int)
            or self.canonical_byte_count <= 0
            or self.record_digest != _address(self.content_dict())
        ):
            raise PrototypeSceneHeadlessError("freeze commit receipt differs")

    @classmethod
    def seal(
        cls,
        freeze: PrototypeSceneCandidateFreeze,
        canonical_bytes: bytes,
        *,
        storage_id: str,
    ) -> "PrototypeSceneFreezeCommitReceipt":
        expected = canonical_json(freeze.to_data()) + b"\n"
        if canonical_bytes != expected:
            raise PrototypeSceneHeadlessError("freeze commit bytes are not canonical")
        values: dict[str, object] = {
            "freeze_digest": freeze.record_digest,
            "canonical_bytes_digest": "sha256:"
            + hashlib.sha256(expected).hexdigest(),
            "canonical_byte_count": len(expected),
            "storage_id": storage_id,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=_address(provisional.content_dict()),
        )

    def assert_matches(
        self, freeze: PrototypeSceneCandidateFreeze, canonical_bytes: bytes
    ) -> None:
        if PrototypeSceneFreezeCommitReceipt.seal(
            freeze, canonical_bytes, storage_id=self.storage_id
        ) != self:
            raise PrototypeSceneHeadlessError("freeze commit replay differs")

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "PrototypeSceneFreezeCommitReceipt":
        expected = {
            "schema",
            "freeze_digest",
            "canonical_bytes_digest",
            "canonical_byte_count",
            "storage_id",
            "durable_before_query",
            *_authority_data(),
            "record_digest",
        }
        _fields(value, expected, "prototype-scene freeze commit")
        result = cls(
            value["freeze_digest"],
            value["canonical_bytes_digest"],
            value["canonical_byte_count"],
            value["storage_id"],
            value["record_digest"],
        )
        if result.to_data() != dict(value):
            raise PrototypeSceneHeadlessError("freeze commit is not canonical")
        return result


def _support_digest(panels: Sequence[PrototypeScenePanelEvaluation]) -> str:
    return _address(
        {
            "schema": "gkm.bongard-prototype-scene-headless-support.v1",
            "panels": [item.to_data() for item in panels],
            "sides": ["positive"] * 6 + ["negative"] * 6,
            **_authority_data(),
        }
    )


def _query_digest(
    panels: Sequence[PrototypeScenePanelEvaluation],
    results: Sequence[PrototypeSceneCandidateResult],
) -> str:
    return _address(
        {
            "schema": "gkm.bongard-prototype-scene-headless-query.v1",
            "panels": [item.to_data() for item in panels],
            "sides": ["positive", "negative"],
            "results": [item.to_data() for item in results],
            **_authority_data(),
        }
    )


def _candidate(
    library: PrototypeScenePredicateLibrary, candidate_id: str
) -> PrototypeSceneSupportCandidate:
    matches = tuple(
        item
        for item in complete_prototype_scene_candidates(library)
        if item.candidate_id == candidate_id
    )
    if len(matches) != 1:
        raise PrototypeSceneHeadlessError("selected candidate is outside inventory")
    return matches[0]


def _archive_content(value: "PrototypeSceneHeadlessArchive") -> dict[str, object]:
    return {
        "schema": ARCHIVE_SCHEMA,
        "runner_id": RUNNER_ID,
        "status": value.status.value,
        "runner_source_digest": value.runner_source_digest,
        "family": value.family.to_data(),
        "family_digest": value.family_digest,
        "library": value.library.to_data(),
        "library_digest": value.library_digest,
        "support_panels": [item.to_data() for item in value.support_panels],
        "support_digest": value.support_digest,
        "version_space": value.version_space.to_data(),
        "version_space_digest": value.version_space_digest,
        "rank_response": (
            None if value.rank_response is None else value.rank_response.to_data()
        ),
        "rank_response_digest": value.rank_response_digest,
        "ranking": None if value.ranking is None else value.ranking.to_data(),
        "ranking_digest": value.ranking_digest,
        "freeze": None if value.freeze is None else value.freeze.to_data(),
        "freeze_digest": value.freeze_digest,
        "freeze_commit": (
            None if value.freeze_commit is None else value.freeze_commit.to_data()
        ),
        "freeze_commit_digest": value.freeze_commit_digest,
        "query_panels": [item.to_data() for item in value.query_panels],
        "query_sides": ["positive", "negative"] if value.query_panels else [],
        "query_results": [item.to_data() for item in value.query_results],
        "query_digest": value.query_digest,
        "rank_calls_made": value.rank_calls_made,
        "artifact_verification_calls_made": value.artifact_verification_calls_made,
        "query_source_calls_made": value.query_source_calls_made,
        "typed_geometry_is_nondecisional": True,
        "external_observer_artifact_verifier_required": True,
        "full_campaign_must_reverify_archived_observer_artifacts": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PrototypeSceneHeadlessArchive:
    status: PrototypeSceneHeadlessStatus
    runner_source_digest: str
    family: PrototypeSceneCalibrationFamily
    family_digest: str
    library: PrototypeScenePredicateLibrary
    library_digest: str
    support_panels: tuple[PrototypeScenePanelEvaluation, ...]
    support_digest: str
    version_space: PrototypeSceneSupportVersionSpace
    version_space_digest: str
    rank_response: PrototypeSceneRankResponse | None
    rank_response_digest: str | None
    ranking: PrototypeSceneVerifiedRanking | None
    ranking_digest: str | None
    freeze: PrototypeSceneCandidateFreeze | None
    freeze_digest: str | None
    freeze_commit: PrototypeSceneFreezeCommitReceipt | None
    freeze_commit_digest: str | None
    query_panels: tuple[PrototypeScenePanelEvaluation, ...]
    query_results: tuple[PrototypeSceneCandidateResult, ...]
    query_digest: str | None
    rank_calls_made: int
    artifact_verification_calls_made: int
    query_source_calls_made: int
    record_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, PrototypeSceneHeadlessStatus):
            raise TypeError("status must be PrototypeSceneHeadlessStatus")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (
                self.rank_calls_made,
                self.artifact_verification_calls_made,
                self.query_source_calls_made,
            )
        ):
            raise PrototypeSceneHeadlessError("archive call counts must be integers")
        for name in (
            "runner_source_digest",
            "family_digest",
            "library_digest",
            "support_digest",
            "version_space_digest",
            "record_digest",
        ):
            _require_address(getattr(self, name), name)
        if (
            self.runner_source_digest != prototype_scene_runner_source_digest()
            or self.family_digest != self.family.record_digest
            or self.library_digest != self.library.record_digest
            or len(self.support_panels) != 12
            or len({item.panel_id for item in self.support_panels}) != 12
            or self.support_digest != _support_digest(self.support_panels)
        ):
            raise PrototypeSceneHeadlessError("archive support parents differ")
        self.library.assert_matches_family(self.family)
        for panel in self.support_panels:
            panel.assert_matches(self.family)
        self.version_space.assert_matches(
            self.library,
            self.family,
            self.support_panels[:6],
            self.support_panels[6:],
        )
        if self.version_space_digest != self.version_space.record_digest:
            raise PrototypeSceneHeadlessError("archive version digest differs")
        is_gap = not self.version_space.survivor_candidate_ids
        if is_gap:
            assert self.version_space.gap is not None
            expected_status = (
                PrototypeSceneHeadlessStatus.LANGUAGE_GAP
                if self.version_space.gap.kind is PrototypeSceneGapKind.LANGUAGE_GAP
                else PrototypeSceneHeadlessStatus.WITNESS_GAP
            )
            if (
                self.status is not expected_status
                or any(
                    item is not None
                    for item in (
                        self.rank_response,
                        self.rank_response_digest,
                        self.ranking,
                        self.ranking_digest,
                        self.freeze,
                        self.freeze_digest,
                        self.freeze_commit,
                        self.freeze_commit_digest,
                        self.query_digest,
                    )
                )
                or self.query_panels
                or self.query_results
                or self.rank_calls_made != 0
                or self.artifact_verification_calls_made != 12
                or self.query_source_calls_made != 0
            ):
                raise PrototypeSceneHeadlessError("gap archive phase fields differ")
        else:
            if (
                self.status is not PrototypeSceneHeadlessStatus.COMPLETE
                or not isinstance(self.rank_response, PrototypeSceneRankResponse)
                or not isinstance(self.ranking, PrototypeSceneVerifiedRanking)
                or not isinstance(self.freeze, PrototypeSceneCandidateFreeze)
                or not isinstance(self.freeze_commit, PrototypeSceneFreezeCommitReceipt)
                or len(self.query_panels) != 2
                or len(self.query_results) != 2
                or self.rank_calls_made != 1
                or self.artifact_verification_calls_made != 14
                or self.query_source_calls_made != 1
            ):
                raise PrototypeSceneHeadlessError("complete archive phase fields differ")
            rank_input = prototype_scene_rank_input_digest(
                library_digest=self.library_digest,
                version_space_digest=self.version_space_digest,
                survivor_candidate_ids=self.version_space.survivor_candidate_ids,
            )
            self.rank_response.assert_matches(
                expected_input_digest=rank_input,
                survivor_candidate_ids=self.version_space.survivor_candidate_ids,
            )
            self.ranking.assert_matches(self.version_space, self.library)
            selected = _candidate(self.library, self.ranking.ordered_candidate_ids[0])
            if (
                self.rank_response_digest != self.rank_response.record_digest
                or self.ranking_digest != self.ranking.record_digest
                or self.ranking.ordered_candidate_ids
                != self.rank_response.ordered_candidate_ids
                or selected.candidate_id
                not in self.version_space.survivor_candidate_ids
                or self.freeze.selected_candidate != selected
                or self.freeze.runner_source_digest != self.runner_source_digest
                or self.freeze.library_digest != self.library_digest
                or self.freeze.calibration_family_digest != self.family_digest
                or self.freeze.rank_response_digest != self.rank_response_digest
                or self.freeze.ranking_digest != self.ranking_digest
                or self.freeze.version_space_digest != self.version_space_digest
                or self.freeze.support_digest != self.support_digest
                or self.freeze_digest != self.freeze.record_digest
            ):
                raise PrototypeSceneHeadlessError("archive ranking/freeze differs")
            freeze_bytes = canonical_json(self.freeze.to_data()) + b"\n"
            self.freeze_commit.assert_matches(self.freeze, freeze_bytes)
            if self.freeze_commit_digest != self.freeze_commit.record_digest:
                raise PrototypeSceneHeadlessError("freeze commit digest differs")
            support_ids = {item.panel_id for item in self.support_panels}
            query_ids = {item.panel_id for item in self.query_panels}
            if len(query_ids) != 2 or query_ids.intersection(support_ids):
                raise PrototypeSceneHeadlessError("query IDs repeat or overlap support")
            for panel, result in zip(self.query_panels, self.query_results, strict=True):
                panel.assert_matches(self.family)
                if evaluate_prototype_scene_candidate(
                    selected, self.library, self.family, panel
                ) != result:
                    raise PrototypeSceneHeadlessError("query result replay differs")
            if self.query_digest != _query_digest(self.query_panels, self.query_results):
                raise PrototypeSceneHeadlessError("query digest differs")
        if self.record_digest != _address(_archive_content(self)):
            raise PrototypeSceneHeadlessError("archive digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_archive_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypeSceneHeadlessArchive":
        expected = {
            "schema",
            "runner_id",
            "status",
            "runner_source_digest",
            "family",
            "family_digest",
            "library",
            "library_digest",
            "support_panels",
            "support_digest",
            "version_space",
            "version_space_digest",
            "rank_response",
            "rank_response_digest",
            "ranking",
            "ranking_digest",
            "freeze",
            "freeze_digest",
            "freeze_commit",
            "freeze_commit_digest",
            "query_panels",
            "query_sides",
            "query_results",
            "query_digest",
            "rank_calls_made",
            "artifact_verification_calls_made",
            "query_source_calls_made",
            "typed_geometry_is_nondecisional",
            "external_observer_artifact_verifier_required",
            "full_campaign_must_reverify_archived_observer_artifacts",
            *_authority_data(),
            "record_digest",
        }
        _fields(value, expected, "prototype-scene archive")
        for name in ("family", "library", "version_space"):
            if not isinstance(value[name], Mapping):
                raise PrototypeSceneHeadlessError(f"archive {name} is malformed")
        optional_mappings = ("rank_response", "ranking", "freeze", "freeze_commit")
        if any(
            value[name] is not None and not isinstance(value[name], Mapping)
            for name in optional_mappings
        ):
            raise PrototypeSceneHeadlessError("archive optional record is malformed")
        result = cls(
            status=PrototypeSceneHeadlessStatus(value["status"]),
            runner_source_digest=value["runner_source_digest"],
            family=PrototypeSceneCalibrationFamily.from_data(value["family"]),
            family_digest=value["family_digest"],
            library=PrototypeScenePredicateLibrary.from_data(value["library"]),
            library_digest=value["library_digest"],
            support_panels=tuple(
                PrototypeScenePanelEvaluation.from_data(item)
                for item in value["support_panels"]
            ),
            support_digest=value["support_digest"],
            version_space=PrototypeSceneSupportVersionSpace.from_data(
                value["version_space"]
            ),
            version_space_digest=value["version_space_digest"],
            rank_response=(
                None
                if value["rank_response"] is None
                else PrototypeSceneRankResponse.from_data(value["rank_response"])
            ),
            rank_response_digest=value["rank_response_digest"],
            ranking=(
                None
                if value["ranking"] is None
                else PrototypeSceneVerifiedRanking.from_data(value["ranking"])
            ),
            ranking_digest=value["ranking_digest"],
            freeze=(
                None
                if value["freeze"] is None
                else PrototypeSceneCandidateFreeze.from_data(value["freeze"])
            ),
            freeze_digest=value["freeze_digest"],
            freeze_commit=(
                None
                if value["freeze_commit"] is None
                else PrototypeSceneFreezeCommitReceipt.from_data(
                    value["freeze_commit"]
                )
            ),
            freeze_commit_digest=value["freeze_commit_digest"],
            query_panels=tuple(
                PrototypeScenePanelEvaluation.from_data(item)
                for item in value["query_panels"]
            ),
            query_results=tuple(
                PrototypeSceneCandidateResult.from_data(item)
                for item in value["query_results"]
            ),
            query_digest=value["query_digest"],
            rank_calls_made=value["rank_calls_made"],
            artifact_verification_calls_made=value[
                "artifact_verification_calls_made"
            ],
            query_source_calls_made=value["query_source_calls_made"],
            record_digest=value["record_digest"],
        )
        if result.to_data() != dict(value):
            raise PrototypeSceneHeadlessError("archive is not canonical")
        return result


Ranker = Callable[[tuple[str, ...], str], PrototypeSceneRankResponse | Mapping[str, Any]]
FreezeCommitter = Callable[
    [bytes], PrototypeSceneFreezeCommitReceipt | Mapping[str, Any]
]
QuerySource = Callable[
    [Mapping[str, object]], Mapping[str, PrototypeScenePanelEvaluation]
]
ArtifactVerifier = Callable[
    [PrototypeSceneVerifiedObserverBinding, bytes], None
]


def _verify_panel_artifact(
    panel: PrototypeScenePanelEvaluation, verifier: ArtifactVerifier
) -> None:
    if not callable(verifier):
        raise TypeError("external artifact_verifier is required")
    verifier(panel.observer_binding, panel.exact_png_bytes)
    # Re-check the binding after the external hook returns; a hook cannot
    # replace or mutate the archived authority record.
    panel.observer_binding.assert_matches(
        panel_id=panel.panel_id,
        exact_png_bytes=panel.exact_png_bytes,
        scores=panel.scores,
        context=panel.context,
    )


def _make_archive(**values: object) -> PrototypeSceneHeadlessArchive:
    provisional = object.__new__(PrototypeSceneHeadlessArchive)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return PrototypeSceneHeadlessArchive(
        **values,  # type: ignore[arg-type]
        record_digest=_address(_archive_content(provisional)),
    )


def run_prototype_scene_headless(
    family: PrototypeSceneCalibrationFamily,
    library: PrototypeScenePredicateLibrary,
    positive_support: Sequence[PrototypeScenePanelEvaluation],
    negative_support: Sequence[PrototypeScenePanelEvaluation],
    *,
    artifact_verifier: ArtifactVerifier,
    ranker: Ranker,
    freeze_committer: FreezeCommitter,
    query_source: QuerySource,
) -> PrototypeSceneHeadlessArchive:
    """Build all three rows, rank once, durably freeze, then admit 1+1 queries."""

    library.assert_matches_family(family)
    positives = tuple(positive_support)
    negatives = tuple(negative_support)
    if len(positives) != 6 or len(negatives) != 6:
        raise PrototypeSceneHeadlessError("runner requires exact 6+6 support")
    support = positives + negatives
    if len({item.panel_id for item in support}) != 12:
        raise PrototypeSceneHeadlessError("support panel IDs must be unique")
    for panel in support:
        _verify_panel_artifact(panel, artifact_verifier)
        panel.assert_matches(family)
    version = build_prototype_scene_support_version_space(
        library, family, positives, negatives
    )
    support_digest = _support_digest(support)
    common: dict[str, object] = {
        "runner_source_digest": prototype_scene_runner_source_digest(),
        "family": family,
        "family_digest": family.record_digest,
        "library": library,
        "library_digest": library.record_digest,
        "support_panels": support,
        "support_digest": support_digest,
        "version_space": version,
        "version_space_digest": version.record_digest,
    }
    if not version.survivor_candidate_ids:
        assert version.gap is not None
        status = (
            PrototypeSceneHeadlessStatus.LANGUAGE_GAP
            if version.gap.kind is PrototypeSceneGapKind.LANGUAGE_GAP
            else PrototypeSceneHeadlessStatus.WITNESS_GAP
        )
        return _make_archive(
            status=status,
            **common,
            rank_response=None,
            rank_response_digest=None,
            ranking=None,
            ranking_digest=None,
            freeze=None,
            freeze_digest=None,
            freeze_commit=None,
            freeze_commit_digest=None,
            query_panels=(),
            query_results=(),
            query_digest=None,
            rank_calls_made=0,
            artifact_verification_calls_made=12,
            query_source_calls_made=0,
        )
    rank_input = prototype_scene_rank_input_digest(
        library_digest=library.record_digest,
        version_space_digest=version.record_digest,
        survivor_candidate_ids=version.survivor_candidate_ids,
    )
    raw_response = ranker(version.survivor_candidate_ids, rank_input)
    response = (
        raw_response
        if isinstance(raw_response, PrototypeSceneRankResponse)
        else PrototypeSceneRankResponse.from_data(raw_response)
    )
    response.assert_matches(
        expected_input_digest=rank_input,
        survivor_candidate_ids=version.survivor_candidate_ids,
    )
    ranking = rank_prototype_scene_survivors(
        version, library, response.ordered_candidate_ids
    )
    selected = _candidate(library, ranking.ordered_candidate_ids[0])
    freeze = PrototypeSceneCandidateFreeze.seal(
        library=library,
        family=family,
        support_digest=support_digest,
        version=version,
        ranking=ranking,
        rank_response=response,
        selected_candidate=selected,
    )
    freeze_data = PrototypeSceneCandidateFreeze.from_data(freeze.to_data()).to_data()
    freeze_bytes = canonical_json(freeze_data) + b"\n"
    raw_commit = freeze_committer(freeze_bytes)
    commit = (
        raw_commit
        if isinstance(raw_commit, PrototypeSceneFreezeCommitReceipt)
        else PrototypeSceneFreezeCommitReceipt.from_data(raw_commit)
    )
    commit.assert_matches(freeze, freeze_bytes)
    raw_queries = query_source(freeze_data)
    if not isinstance(raw_queries, Mapping) or set(raw_queries) != {
        "positive",
        "negative",
    }:
        raise PrototypeSceneHeadlessError("query source must return positive+negative")
    queries = (raw_queries["positive"], raw_queries["negative"])
    if any(not isinstance(item, PrototypeScenePanelEvaluation) for item in queries):
        raise TypeError("queries must be PrototypeScenePanelEvaluation")
    support_ids = {item.panel_id for item in support}
    if len({item.panel_id for item in queries}) != 2 or any(
        item.panel_id in support_ids for item in queries
    ):
        raise PrototypeSceneHeadlessError("query IDs repeat or overlap support")
    for panel in queries:
        _verify_panel_artifact(panel, artifact_verifier)
        panel.assert_matches(family)
    query_results = tuple(
        evaluate_prototype_scene_candidate(selected, library, family, panel)
        for panel in queries
    )
    return _make_archive(
        status=PrototypeSceneHeadlessStatus.COMPLETE,
        **common,
        rank_response=response,
        rank_response_digest=response.record_digest,
        ranking=ranking,
        ranking_digest=ranking.record_digest,
        freeze=freeze,
        freeze_digest=freeze.record_digest,
        freeze_commit=commit,
        freeze_commit_digest=commit.record_digest,
        query_panels=queries,
        query_results=query_results,
        query_digest=_query_digest(queries, query_results),
        rank_calls_made=1,
        artifact_verification_calls_made=14,
        query_source_calls_made=1,
    )


def cold_replay_prototype_scene_headless_run(
    archive: PrototypeSceneHeadlessArchive | Mapping[str, Any],
    *,
    expected_archive_digest: str,
    artifact_verifier: ArtifactVerifier,
) -> PrototypeSceneHeadlessArchive:
    """Externally anchored replay with no observer, ranker, or query-source call."""

    expected = _require_address(expected_archive_digest, "expected_archive_digest")
    supplied = (
        archive.record_digest
        if isinstance(archive, PrototypeSceneHeadlessArchive)
        else archive.get("record_digest")
    )
    if supplied != expected:
        raise PrototypeSceneHeadlessError("archive differs from external commitment")
    restored = (
        archive
        if isinstance(archive, PrototypeSceneHeadlessArchive)
        else PrototypeSceneHeadlessArchive.from_data(archive)
    )
    restored = PrototypeSceneHeadlessArchive.from_data(restored.to_data())
    if restored.record_digest != expected:
        raise PrototypeSceneHeadlessError("cold archive digest differs")
    for panel in (*restored.support_panels, *restored.query_panels):
        _verify_panel_artifact(panel, artifact_verifier)
        panel.assert_matches(restored.family)
    replayed_version = build_prototype_scene_support_version_space(
        restored.library,
        restored.family,
        restored.support_panels[:6],
        restored.support_panels[6:],
    )
    if replayed_version != restored.version_space:
        raise PrototypeSceneHeadlessError("cold version space differs")
    if restored.status is not PrototypeSceneHeadlessStatus.COMPLETE:
        return restored
    assert restored.rank_response is not None and restored.ranking is not None
    rank_input = prototype_scene_rank_input_digest(
        library_digest=restored.library_digest,
        version_space_digest=restored.version_space_digest,
        survivor_candidate_ids=restored.version_space.survivor_candidate_ids,
    )
    restored.rank_response.assert_matches(
        expected_input_digest=rank_input,
        survivor_candidate_ids=restored.version_space.survivor_candidate_ids,
    )
    restored.ranking.assert_matches(restored.version_space, restored.library)
    return restored


__all__ = [
    "ARCHIVE_SCHEMA",
    "FREEZE_COMMIT_SCHEMA",
    "FREEZE_SCHEMA",
    "RANK_RESPONSE_SCHEMA",
    "RUNNER_ID",
    "PrototypeSceneCandidateFreeze",
    "PrototypeSceneFreezeCommitReceipt",
    "PrototypeSceneHeadlessArchive",
    "PrototypeSceneHeadlessError",
    "PrototypeSceneHeadlessStatus",
    "PrototypeSceneRankResponse",
    "cold_replay_prototype_scene_headless_run",
    "prototype_scene_rank_input_digest",
    "prototype_scene_runner_source_digest",
    "run_prototype_scene_headless",
]

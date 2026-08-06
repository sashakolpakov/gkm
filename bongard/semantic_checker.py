"""Non-authoritative checker boundary for Python semantic episodes.

The benchmark's executable meaning is the closed Python IR, its frozen leg
registry, and the Python cold replay.  A proof assistant or any other checker
may inspect an already completed episode, but it is downstream of every
authoritative commitment and cannot replace, repair, or reinterpret one.

Checker output is deliberately returned as a separate sidecar.  It is never
inserted into :meth:`VisualSemanticEpisode.artifact_data`, an
:class:`~bongard.benchmark.EpisodeResult`, or a
:class:`~bongard.artifacts.RunArtifactBundle`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Callable, Mapping

from bongard.artifacts import canonical_digest
from bongard.benchmark import EpisodeResult, EpisodeStatus
from bongard.ir import formula_digest
from bongard.semantic_commitment import REFERENCE_EXECUTION_SEMANTICS
from bongard.semantic_episode import VisualSemanticEpisode


PYTHON_SEMANTIC_AUTHORITY_SCHEMA = "gkm.bongard-python-semantic-authority.v1"
OPTIONAL_CHECKER_REQUEST_SCHEMA = "gkm.bongard-optional-semantic-checker-request.v1"
OPTIONAL_CHECKER_RESPONSE_SCHEMA = "gkm.bongard-optional-semantic-checker-response.v1"
OPTIONAL_CHECKER_SIDECAR_SCHEMA = "gkm.bongard-optional-semantic-checker-sidecar.v1"

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class SemanticCheckerError(RuntimeError):
    """Base class for violations at the optional-checker boundary."""


class SemanticCheckerProtocolError(SemanticCheckerError):
    """A checker returned malformed or incorrectly bound output."""


class SemanticCheckerAuthorityMutation(SemanticCheckerError):
    """A checker changed Python-owned state while it was running."""


class OptionalCheckerUnavailable(SemanticCheckerError):
    """An optional checker adapter could not run in this environment."""


class OptionalCheckerDisagreement(SemanticCheckerError):
    """An optional checker explicitly disagreed with the Python replay."""

    def __init__(self, sidecar: "OptionalCheckerSidecar") -> None:
        self.sidecar = sidecar
        super().__init__(
            "optional checker disagreed with the authoritative Python result"
        )


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SemanticCheckerProtocolError(f"invalid {label} {value!r}")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SemanticCheckerProtocolError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _mapping(
    value: object, fields: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise SemanticCheckerProtocolError(
            f"{label} fields differ from the closed schema"
        )
    return value


@dataclass(frozen=True, slots=True)
class PythonSemanticAuthority:
    """Digest projection of every Python-owned semantic result.

    This object has no checker identity.  Equality of two instances therefore
    means that adding, removing, or replacing an optional checker did not
    change the benchmark semantics or archive.
    """

    task_id: str
    run_id: str
    pre_observation_commitment_digest: str
    lowering_archive_digest: str
    compiled_formula_digest: str
    registry_digest: str
    attachment_contract_digest: str
    proposal_freeze_digest: str
    support_gate_digest: str
    prediction_commitment_digest: str
    run_artifact_chain_digest: str
    run_archive_digest: str
    semantic_archive_digest: str
    episode_result_digest: str

    def __post_init__(self) -> None:
        _identifier(self.task_id, "authority task_id")
        _identifier(self.run_id, "authority run_id")
        for name in (
            "pre_observation_commitment_digest",
            "lowering_archive_digest",
            "compiled_formula_digest",
            "registry_digest",
            "attachment_contract_digest",
            "proposal_freeze_digest",
            "support_gate_digest",
            "prediction_commitment_digest",
            "run_artifact_chain_digest",
            "run_archive_digest",
            "semantic_archive_digest",
            "episode_result_digest",
        ):
            _digest(getattr(self, name), name)

    def content_data(self) -> dict[str, object]:
        return {
            "schema": PYTHON_SEMANTIC_AUTHORITY_SCHEMA,
            "reference_execution_semantics": REFERENCE_EXECUTION_SEMANTICS,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "pre_observation_commitment_digest": (
                self.pre_observation_commitment_digest
            ),
            "lowering_archive_digest": self.lowering_archive_digest,
            "compiled_formula_digest": self.compiled_formula_digest,
            "registry_digest": self.registry_digest,
            "attachment_contract_digest": self.attachment_contract_digest,
            "proposal_freeze_digest": self.proposal_freeze_digest,
            "support_gate_digest": self.support_gate_digest,
            "prediction_commitment_digest": self.prediction_commitment_digest,
            "run_artifact_chain_digest": self.run_artifact_chain_digest,
            "run_archive_digest": self.run_archive_digest,
            "semantic_archive_digest": self.semantic_archive_digest,
            "episode_result_digest": self.episode_result_digest,
            "checker_identity_participates": False,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "authority_digest": self.digest}


def capture_python_semantic_authority(
    episode: VisualSemanticEpisode,
    result: EpisodeResult,
) -> PythonSemanticAuthority:
    """Validate and snapshot a completed episode's Python-owned identities."""

    if not isinstance(episode, VisualSemanticEpisode):
        raise TypeError("episode must be VisualSemanticEpisode")
    if not isinstance(result, EpisodeResult):
        raise TypeError("result must be EpisodeResult")
    if result.status is not EpisodeStatus.COMPLETE or result.bundle is None:
        raise SemanticCheckerProtocolError(
            "optional checking requires a completed Python episode"
        )
    compiled = episode.compiled
    precommit = episode.pre_observation_commitment
    proposed = episode.proposed_rule
    gate = result.support_gate
    freeze = result.proposal_freeze
    if any(item is None for item in (compiled, precommit, proposed, gate, freeze)):
        raise SemanticCheckerProtocolError(
            "completed semantic episode lacks an authoritative component"
        )
    assert compiled is not None
    assert precommit is not None
    assert proposed is not None
    assert gate is not None
    assert freeze is not None
    bundle = result.bundle

    precommit.assert_untampered()
    replay = bundle.verify()
    if not replay.predictions_match:
        raise SemanticCheckerProtocolError("Python cold replay does not reproduce predictions")

    compiled_formula_digest = formula_digest(compiled.formula)
    joins = (
        (
            result.task_id,
            episode.task_id,
            "episode result belongs to another semantic task",
        ),
        (result.run_id, bundle.support.run_id, "result and artifact run IDs differ"),
        (
            compiled_formula_digest,
            formula_digest(freeze.formula),
            "compiled and frozen formulas differ",
        ),
        (
            compiled.registry.digest(),
            freeze.registry_digest,
            "compiled and frozen registries differ",
        ),
        (
            compiled.attachment_contract.digest(),
            freeze.attachment_contract_digest,
            "compiled and frozen attachments differ",
        ),
        (
            bundle.freeze.digest(),
            freeze.digest(),
            "result and bundled freezes differ",
        ),
        (
            gate.digest,
            freeze.support_gate_digest,
            "support gate is not the parent of the freeze",
        ),
        (
            bundle.predictions.proposal_freeze_digest,
            freeze.digest(),
            "predictions do not descend from the freeze",
        ),
        (
            proposed.proposer_digest,
            precommit.digest,
            "proposed rule does not descend from the Python precommitment",
        ),
        (
            compiled.lowering_archive.digest,
            precommit.identity_data()["lowering_archive_digest"],
            "lowering archive differs from the Python precommitment",
        ),
    )
    for actual, expected, reason in joins:
        if actual != expected:
            raise SemanticCheckerProtocolError(reason)

    run_archive = bundle.to_archive_data()
    run_archive_digest = _digest(run_archive.get("archive_digest"), "run archive digest")
    semantic_archive = episode.artifact_data()
    # Output from an optional checker has no admitted field in this object.
    forbidden = {"checker_output", "checker_response", "checker_sidecar"}
    if forbidden & set(semantic_archive):
        raise SemanticCheckerProtocolError(
            "optional checker output entered the authoritative semantic archive"
        )
    return PythonSemanticAuthority(
        task_id=result.task_id,
        run_id=result.run_id,
        pre_observation_commitment_digest=precommit.digest,
        lowering_archive_digest=compiled.lowering_archive.digest,
        compiled_formula_digest=compiled_formula_digest,
        registry_digest=compiled.registry.digest(),
        attachment_contract_digest=compiled.attachment_contract.digest(),
        proposal_freeze_digest=freeze.digest(),
        support_gate_digest=gate.digest,
        prediction_commitment_digest=bundle.predictions.digest(),
        run_artifact_chain_digest=bundle.digest(),
        run_archive_digest=run_archive_digest,
        semantic_archive_digest=canonical_digest(semantic_archive),
        episode_result_digest=canonical_digest(result.to_data()),
    )


@dataclass(frozen=True, slots=True)
class OptionalCheckerRequest:
    """Complete, read-only replay input offered to an optional checker."""

    authority: PythonSemanticAuthority
    payload: Mapping[str, Any]

    @classmethod
    def capture(
        cls,
        episode: VisualSemanticEpisode,
        result: EpisodeResult,
    ) -> "OptionalCheckerRequest":
        authority = capture_python_semantic_authority(episode, result)
        assert episode.compiled is not None
        assert result.bundle is not None
        assert result.support_gate is not None
        assert result.proposal_freeze is not None
        compiled = episode.compiled
        payload = {
            "lowering_archive": compiled.lowering_archive.to_data(),
            "compiled_formula": compiled.formula.to_data(),
            "registry_snapshot": compiled.registry.snapshot().to_data(),
            "attachment_contract": compiled.attachment_contract.to_data(),
            "proposal_freeze": result.proposal_freeze.to_data(),
            "support_gate": result.support_gate.to_data(),
            "cold_replay_inputs": result.bundle.cold_inputs.to_data(),
            "prediction_commitment": result.bundle.predictions.to_data(),
        }
        # Detach even though every source is nominally frozen: checker adapters
        # receive ordinary JSON, never live benchmark objects.
        return cls(authority=authority, payload=copy.deepcopy(payload))

    def content_data(self) -> dict[str, object]:
        return {
            "schema": OPTIONAL_CHECKER_REQUEST_SCHEMA,
            "authoritative_semantics": REFERENCE_EXECUTION_SEMANTICS,
            "checker_may_affect_result": False,
            "authority": self.authority.to_data(),
            "replay_payload": copy.deepcopy(dict(self.payload)),
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "request_digest": self.digest}


@dataclass(frozen=True, slots=True)
class OptionalCheckerResponse:
    """Strict response bound to one checker request and Python authority."""

    checker_id: str
    checker_version: str
    request_digest: str
    authority_digest: str
    agrees: bool
    detail: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.checker_id, "checker_id")
        _identifier(self.checker_version, "checker_version")
        _digest(self.request_digest, "checker request_digest")
        _digest(self.authority_digest, "checker authority_digest")
        if not isinstance(self.agrees, bool):
            raise SemanticCheckerProtocolError("checker agrees must be Boolean")
        if self.detail is not None and (
            not isinstance(self.detail, str) or len(self.detail) > 4096
        ):
            raise SemanticCheckerProtocolError(
                "checker detail must be null or at most 4096 characters"
            )

    def content_data(self) -> dict[str, object]:
        return {
            "schema": OPTIONAL_CHECKER_RESPONSE_SCHEMA,
            "checker_id": self.checker_id,
            "checker_version": self.checker_version,
            "request_digest": self.request_digest,
            "authority_digest": self.authority_digest,
            "agrees": self.agrees,
            "detail": self.detail,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "response_digest": self.digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "OptionalCheckerResponse":
        data = _mapping(
            value,
            frozenset(
                {
                    "schema",
                    "checker_id",
                    "checker_version",
                    "request_digest",
                    "authority_digest",
                    "agrees",
                    "detail",
                    "response_digest",
                }
            ),
            "optional checker response",
        )
        if data["schema"] != OPTIONAL_CHECKER_RESPONSE_SCHEMA:
            raise SemanticCheckerProtocolError(
                "unsupported optional checker response schema"
            )
        result = cls(
            checker_id=data["checker_id"],
            checker_version=data["checker_version"],
            request_digest=data["request_digest"],
            authority_digest=data["authority_digest"],
            agrees=data["agrees"],
            detail=data["detail"],
        )
        if result.digest != data["response_digest"] or result.to_data() != dict(value):
            raise SemanticCheckerProtocolError(
                "optional checker response digest or representation differs"
            )
        return result


class OptionalCheckerStatus(str, Enum):
    ABSENT = "absent"
    AGREED = "agreed"
    UNAVAILABLE = "unavailable"
    DISAGREED = "disagreed"


@dataclass(frozen=True, slots=True)
class OptionalCheckerSidecar:
    """Non-authoritative record intentionally excluded from run archives."""

    status: OptionalCheckerStatus
    authority_digest: str
    request_digest: str
    checker_id: str | None
    checker_version: str | None
    response: OptionalCheckerResponse | None
    unavailability_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, OptionalCheckerStatus):
            raise TypeError("status must be OptionalCheckerStatus")
        _digest(self.authority_digest, "sidecar authority_digest")
        _digest(self.request_digest, "sidecar request_digest")
        if self.status is OptionalCheckerStatus.ABSENT:
            if any(
                item is not None
                for item in (
                    self.checker_id,
                    self.checker_version,
                    self.response,
                    self.unavailability_reason,
                )
            ):
                raise SemanticCheckerProtocolError(
                    "absent checker sidecar cannot name or contain a checker"
                )
        else:
            _identifier(self.checker_id, "sidecar checker_id")
            _identifier(self.checker_version, "sidecar checker_version")
        if self.status in {OptionalCheckerStatus.AGREED, OptionalCheckerStatus.DISAGREED}:
            if self.response is None or self.unavailability_reason is not None:
                raise SemanticCheckerProtocolError(
                    "completed checker sidecar requires exactly one response"
                )
            if (
                self.response.checker_id != self.checker_id
                or self.response.checker_version != self.checker_version
                or self.response.authority_digest != self.authority_digest
                or self.response.request_digest != self.request_digest
            ):
                raise SemanticCheckerProtocolError(
                    "checker sidecar response differs from its bound identities"
                )
            if self.response.agrees is not (
                self.status is OptionalCheckerStatus.AGREED
            ):
                raise SemanticCheckerProtocolError(
                    "checker response and sidecar status disagree"
                )
        elif self.status is OptionalCheckerStatus.UNAVAILABLE:
            if (
                self.response is not None
                or not isinstance(self.unavailability_reason, str)
                or not self.unavailability_reason
                or len(self.unavailability_reason) > 4096
            ):
                raise SemanticCheckerProtocolError(
                    "unavailable checker sidecar requires a reason and no response"
                )

    def content_data(self) -> dict[str, object]:
        return {
            "schema": OPTIONAL_CHECKER_SIDECAR_SCHEMA,
            "status": self.status.value,
            "non_authoritative": True,
            "may_affect_python_result": False,
            "python_authority_unchanged": True,
            "authority_digest": self.authority_digest,
            "request_digest": self.request_digest,
            "checker": (
                None
                if self.checker_id is None
                else {"id": self.checker_id, "version": self.checker_version}
            ),
            "response": None if self.response is None else self.response.to_data(),
            "unavailability_reason": self.unavailability_reason,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "sidecar_digest": self.digest}


CheckerCallable = Callable[
    [Mapping[str, Any]], OptionalCheckerResponse | Mapping[str, Any]
]


def audit_optional_semantic_checker(
    episode: VisualSemanticEpisode,
    result: EpisodeResult,
    *,
    checker: CheckerCallable | None = None,
    checker_id: str | None = None,
    checker_version: str | None = None,
) -> OptionalCheckerSidecar:
    """Run a downstream checker without admitting it into Python semantics.

    Absence and declared unavailability are ordinary sidecar statuses.  A
    malformed response, mutation of Python state, or semantic disagreement is
    an explicit exception; disagreement carries its diagnostic sidecar.
    """

    request = OptionalCheckerRequest.capture(episode, result)
    authority = request.authority
    if checker is None:
        if checker_id is not None or checker_version is not None:
            raise SemanticCheckerProtocolError(
                "absent checker cannot have an ID or version"
            )
        after = capture_python_semantic_authority(episode, result)
        if after != authority:
            raise SemanticCheckerAuthorityMutation(
                "Python authority changed while recording absent checker"
            )
        return OptionalCheckerSidecar(
            OptionalCheckerStatus.ABSENT,
            authority.digest,
            request.digest,
            None,
            None,
            None,
        )

    if not callable(checker):
        raise TypeError("checker must be callable or None")
    checked_id = _identifier(checker_id, "checker_id")
    checked_version = _identifier(checker_version, "checker_version")
    raw_response: OptionalCheckerResponse | Mapping[str, Any]
    try:
        raw_response = checker(copy.deepcopy(request.to_data()))
    except OptionalCheckerUnavailable as exc:
        after = capture_python_semantic_authority(episode, result)
        if after != authority:
            raise SemanticCheckerAuthorityMutation(
                "unavailable checker changed Python authority"
            ) from exc
        reason = str(exc).strip() or "optional checker unavailable"
        if len(reason) > 4096:
            reason = reason[:4096]
        return OptionalCheckerSidecar(
            OptionalCheckerStatus.UNAVAILABLE,
            authority.digest,
            request.digest,
            checked_id,
            checked_version,
            None,
            reason,
        )

    after = capture_python_semantic_authority(episode, result)
    if after != authority:
        raise SemanticCheckerAuthorityMutation(
            "optional checker changed authoritative Python state"
        )
    if isinstance(raw_response, OptionalCheckerResponse):
        response = OptionalCheckerResponse.from_data(raw_response.to_data())
    elif isinstance(raw_response, Mapping):
        response = OptionalCheckerResponse.from_data(raw_response)
    else:
        raise SemanticCheckerProtocolError(
            "optional checker must return a strict response object"
        )
    if response.checker_id != checked_id or response.checker_version != checked_version:
        raise SemanticCheckerProtocolError(
            "optional checker response identity differs from invoked checker"
        )
    if response.request_digest != request.digest:
        raise SemanticCheckerProtocolError(
            "optional checker response belongs to another replay request"
        )
    if response.authority_digest != authority.digest:
        raise SemanticCheckerProtocolError(
            "optional checker response belongs to another Python authority"
        )
    status = (
        OptionalCheckerStatus.AGREED
        if response.agrees
        else OptionalCheckerStatus.DISAGREED
    )
    sidecar = OptionalCheckerSidecar(
        status,
        authority.digest,
        request.digest,
        checked_id,
        checked_version,
        response,
    )
    if not response.agrees:
        raise OptionalCheckerDisagreement(sidecar)
    return sidecar


__all__ = [
    "OPTIONAL_CHECKER_REQUEST_SCHEMA",
    "OPTIONAL_CHECKER_RESPONSE_SCHEMA",
    "OPTIONAL_CHECKER_SIDECAR_SCHEMA",
    "PYTHON_SEMANTIC_AUTHORITY_SCHEMA",
    "CheckerCallable",
    "OptionalCheckerDisagreement",
    "OptionalCheckerRequest",
    "OptionalCheckerResponse",
    "OptionalCheckerSidecar",
    "OptionalCheckerStatus",
    "OptionalCheckerUnavailable",
    "PythonSemanticAuthority",
    "SemanticCheckerAuthorityMutation",
    "SemanticCheckerError",
    "SemanticCheckerProtocolError",
    "audit_optional_semantic_checker",
    "capture_python_semantic_authority",
]

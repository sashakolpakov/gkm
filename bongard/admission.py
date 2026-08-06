"""Verifier-owned attachment, replay, novelty, and promotion admission.

Admission is deliberately a pure compare-and-swap transition.  Every gate is
evaluated first; only one accepted :class:`PromotionDecision` contains a new
archive contract.  A failed decision cannot partially mutate the accepted
archive.

Candidate source is tied to the exact registered implementation it promotes,
and the comparison source is tied to a verifier-precommitted digest.  This
prevents a candidate from submitting harmless decoy text to obtain a cheap AST
novelty charge while replaying different code.

This module checks data contracts; it does not execute evaluation suites and
``issued_by`` is a namespace, not a cryptographic signature.  A production
caller must construct these receipts inside a verifier-owned process (or
authenticate them externally) before calling :class:`AdmissionVerifier`.
Candidate code must never be allowed to manufacture its own evidence DTOs.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable, Mapping

from bongard.ir import Formula, atoms, formula_digest, validate_formula
from bongard.legs.contracts import (
    LegReference,
    LegRegistry,
    RegistrySnapshot,
    Transform,
    ValueType,
)


def _canonical_digest(data: object) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check_sha256(value: str, label: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase sha256")


def _check_rate_counts(successes: int, total: int, label: str) -> None:
    if (
        isinstance(successes, bool)
        or isinstance(total, bool)
        or not isinstance(successes, int)
        or not isinstance(total, int)
        or total <= 0
        or successes < 0
        or successes > total
    ):
        raise ValueError(f"{label} counts require 0 <= successes <= total and total > 0")


@dataclass(frozen=True)
class TypedAttachmentContract:
    """Exact type/registry boundary issued and owned by the verifier."""

    issued_by: str
    registry_digest: str
    registry_snapshot: RegistrySnapshot
    boundary_types: tuple[tuple[str, ValueType], ...]
    allowed_legs: tuple[LegReference, ...]
    ir_version: str = "positive-ir/v2"

    def __post_init__(self) -> None:
        if not self.issued_by.strip():
            raise ValueError("attachment contract issuer must be non-empty")
        _check_sha256(self.registry_digest, "registry_digest")
        if not isinstance(self.registry_snapshot, RegistrySnapshot):
            raise TypeError("attachment registry_snapshot is malformed")
        if self.registry_snapshot.digest() != self.registry_digest:
            raise ValueError("registry snapshot differs from registry_digest")
        names = [name for name, _ in self.boundary_types]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("boundary types must be unique and sorted")
        if any(not name.strip() for name in names):
            raise ValueError("boundary names must be non-empty")
        if tuple(sorted(self.allowed_legs)) != self.allowed_legs:
            raise ValueError("allowed leg references must be sorted")
        if len(self.allowed_legs) != len(set(self.allowed_legs)):
            raise ValueError("allowed leg references must be unique")
        for reference in self.allowed_legs:
            self.registry_snapshot.resolve(reference)
        if self.ir_version != "positive-ir/v2":
            raise ValueError("only the closed positive IR v2 is admitted")

    @classmethod
    def issue(
        cls,
        *,
        issued_by: str,
        registry: LegRegistry,
        boundary_types: Mapping[str, ValueType],
        allowed_legs: Iterable[LegReference] | None = None,
    ) -> "TypedAttachmentContract":
        if not registry.frozen:
            raise ValueError("verifier must freeze the leg registry before issuing")
        references = (
            tuple(allowed_legs)
            if allowed_legs is not None
            else tuple(
                registry.reference(contract.name, contract.version)
                for contract in registry.contracts()
            )
        )
        for reference in references:
            registry.resolve(reference)
        snapshot = registry.snapshot()
        return cls(
            issued_by=issued_by,
            registry_digest=snapshot.digest(),
            registry_snapshot=snapshot,
            boundary_types=tuple(sorted(boundary_types.items())),
            allowed_legs=tuple(sorted(references)),
        )

    def to_data(self) -> dict[str, object]:
        return {
            "issued_by": self.issued_by,
            "registry_digest": self.registry_digest,
            "registry_snapshot": self.registry_snapshot.to_data(),
            "boundary_types": [
                [name, value_type.to_data()]
                for name, value_type in self.boundary_types
            ],
            "allowed_legs": [reference.to_data() for reference in self.allowed_legs],
            "ir_version": self.ir_version,
        }

    def digest(self) -> str:
        return _canonical_digest(self.to_data())

    def validate(self, formula: Formula, registry: LegRegistry) -> None:
        if not registry.frozen:
            raise ValueError("admission requires a frozen leg registry")
        if registry.digest() != self.registry_digest:
            raise ValueError("leg registry differs from attachment contract")
        if registry.snapshot() != self.registry_snapshot:
            raise ValueError("live leg contracts differ from archived registry snapshot")
        self.validate_static(formula)

    def validate_static(self, formula: Formula) -> None:
        """Validate a formula using only the embedded non-executable registry."""

        allowed = set(self.allowed_legs)
        for atom in atoms(formula):
            if atom.call.leg not in allowed:
                raise ValueError(
                    f"leg {atom.call.leg.name}@{atom.call.leg.version} is not "
                    "admitted by the verifier"
                )
        validate_formula(
            formula, self.registry_snapshot, dict(self.boundary_types)
        )


@dataclass(frozen=True, order=True)
class ArchiveEntry:
    attachment_id: str
    candidate_digest: str
    formula_digest: str
    replay_digest: str
    case_count: int

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.attachment_id):
            raise ValueError(f"invalid attachment id {self.attachment_id!r}")
        for label, value in (
            ("candidate_digest", self.candidate_digest),
            ("formula_digest", self.formula_digest),
            ("replay_digest", self.replay_digest),
        ):
            _check_sha256(value, label)
        if isinstance(self.case_count, bool) or self.case_count <= 0:
            raise ValueError("archive entry case_count must be positive")


@dataclass(frozen=True)
class ArchiveReplayReceipt:
    """Verifier result for one previously accepted attachment."""

    issued_by: str
    replay_suite_digest: str
    attachment_id: str
    candidate_digest: str
    formula_digest: str
    replay_digest: str
    case_count: int
    passed: bool

    def __post_init__(self) -> None:
        if not self.issued_by.strip():
            raise ValueError("replay receipt issuer must be non-empty")
        for label, value in (
            ("replay_suite_digest", self.replay_suite_digest),
            ("candidate_digest", self.candidate_digest),
            ("formula_digest", self.formula_digest),
            ("replay_digest", self.replay_digest),
        ):
            _check_sha256(value, label)
        if isinstance(self.case_count, bool) or self.case_count <= 0:
            raise ValueError("replay receipt case_count must be positive")


@dataclass(frozen=True)
class CandidateReplayReceipt:
    """Result issued by an external verifier for one frozen replay suite.

    ``AdmissionVerifier`` checks every binding and the explicit pass bit.  It
    cannot establish who created an ordinary Python value; process isolation
    or an external signature remains the caller's responsibility.
    """

    issued_by: str
    replay_suite_digest: str
    candidate_digest: str
    formula_digest: str
    replay_digest: str
    case_count: int
    passed: bool

    def __post_init__(self) -> None:
        if not self.issued_by.strip():
            raise ValueError("candidate replay issuer must be non-empty")
        for label, value in (
            ("replay_suite_digest", self.replay_suite_digest),
            ("candidate_digest", self.candidate_digest),
            ("formula_digest", self.formula_digest),
            ("replay_digest", self.replay_digest),
        ):
            _check_sha256(value, label)
        if isinstance(self.case_count, bool) or self.case_count <= 0:
            raise ValueError("candidate replay case_count must be positive")
        if not isinstance(self.passed, bool):
            raise ValueError("candidate replay passed must be Boolean")


@dataclass(frozen=True)
class ArchivePreservationContract:
    """The complete accepted archive that must replay before promotion."""

    issued_by: str
    replay_suite_digest: str
    entries: tuple[ArchiveEntry, ...] = ()
    version: str = "archive-preservation/v1"

    def __post_init__(self) -> None:
        if not self.issued_by.strip():
            raise ValueError("archive contract issuer must be non-empty")
        _check_sha256(self.replay_suite_digest, "replay_suite_digest")
        if tuple(sorted(self.entries)) != self.entries:
            raise ValueError("archive entries must be sorted")
        ids = [entry.attachment_id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("archive attachment ids must be unique")
        if self.version != "archive-preservation/v1":
            raise ValueError("unsupported archive preservation contract version")

    def to_data(self) -> dict[str, object]:
        return {
            "issued_by": self.issued_by,
            "replay_suite_digest": self.replay_suite_digest,
            "entries": [asdict(entry) for entry in self.entries],
            "version": self.version,
        }

    def digest(self) -> str:
        return _canonical_digest(self.to_data())

    def verify_full_replay(
        self, receipts: tuple[ArchiveReplayReceipt, ...]
    ) -> tuple[bool, str]:
        expected = {entry.attachment_id: entry for entry in self.entries}
        observed: dict[str, ArchiveReplayReceipt] = {}
        for receipt in receipts:
            if receipt.attachment_id in observed:
                return False, f"duplicate replay receipt {receipt.attachment_id}"
            observed[receipt.attachment_id] = receipt
        if set(observed) != set(expected):
            missing = sorted(set(expected) - set(observed))
            extra = sorted(set(observed) - set(expected))
            return False, f"archive replay coverage mismatch: missing={missing}, extra={extra}"
        for attachment_id, entry in expected.items():
            receipt = observed[attachment_id]
            if receipt.issued_by != self.issued_by:
                return False, f"{attachment_id}: replay issuer mismatch"
            if receipt.replay_suite_digest != self.replay_suite_digest:
                return False, f"{attachment_id}: replay suite mismatch"
            if not receipt.passed:
                return False, f"{attachment_id}: replay failed"
            if (
                receipt.candidate_digest != entry.candidate_digest
                or receipt.formula_digest != entry.formula_digest
                or receipt.replay_digest != entry.replay_digest
                or receipt.case_count != entry.case_count
            ):
                return False, f"{attachment_id}: replay result differs from archive"
        return True, f"replayed all {len(expected)} accepted attachments"

    def with_entry(self, entry: ArchiveEntry) -> "ArchivePreservationContract":
        if any(old.attachment_id == entry.attachment_id for old in self.entries):
            raise ValueError(f"archive already contains {entry.attachment_id}")
        return ArchivePreservationContract(
            issued_by=self.issued_by,
            replay_suite_digest=self.replay_suite_digest,
            entries=tuple(sorted((*self.entries, entry))),
            version=self.version,
        )


@dataclass(frozen=True)
class CandidateAttachment:
    attachment_id: str
    formula: Formula
    candidate_leg: LegReference
    incumbent_source: str
    candidate_source: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.attachment_id):
            raise ValueError(f"invalid attachment id {self.attachment_id!r}")
        if not isinstance(self.candidate_leg, LegReference):
            raise TypeError("candidate_leg must be an exact static leg reference")
        if not self.incumbent_source.strip() or not self.candidate_source.strip():
            raise ValueError("candidate and incumbent sources must be non-empty")

    def digest(self) -> str:
        return _canonical_digest(
            {
                "attachment_id": self.attachment_id,
                "formula_digest": formula_digest(self.formula),
                "candidate_leg": self.candidate_leg.to_data(),
                "incumbent_source_sha256": _sha256_text(self.incumbent_source),
                "candidate_source_sha256": _sha256_text(self.candidate_source),
            }
        )


@dataclass(frozen=True)
class NuisanceTrial:
    case_id: str
    transform: Transform
    preserved: bool

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("nuisance case id must be non-empty")


@dataclass(frozen=True)
class NuisanceEvaluation:
    issued_by: str
    candidate_digest: str
    trials: tuple[NuisanceTrial, ...]

    def __post_init__(self) -> None:
        _check_sha256(self.candidate_digest, "nuisance candidate_digest")
        if not self.trials:
            raise ValueError("nuisance evaluation requires trials")


@dataclass(frozen=True)
class CalibrationEvaluation:
    issued_by: str
    candidate_digest: str
    candidate_brier: float
    baseline_brier: float
    sample_count: int

    def __post_init__(self) -> None:
        _check_sha256(self.candidate_digest, "calibration candidate_digest")
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in (self.candidate_brier, self.baseline_brier)
        ):
            raise ValueError("Brier scores must lie in [0, 1]")
        if isinstance(self.sample_count, bool) or self.sample_count <= 0:
            raise ValueError("calibration sample_count must be positive")


@dataclass(frozen=True)
class NearMissEvaluation:
    issued_by: str
    candidate_digest: str
    successes: int
    total: int

    def __post_init__(self) -> None:
        _check_sha256(self.candidate_digest, "near-miss candidate_digest")
        _check_rate_counts(self.successes, self.total, "near-miss")


@dataclass(frozen=True)
class AntiMemorizationEvaluation:
    issued_by: str
    candidate_digest: str
    challenge_successes: int
    challenge_total: int
    train_query_overlap: int
    forbidden_identifier_hits: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _check_sha256(self.candidate_digest, "anti-memorization candidate_digest")
        _check_rate_counts(
            self.challenge_successes, self.challenge_total, "anti-memorization"
        )
        if isinstance(self.train_query_overlap, bool) or self.train_query_overlap < 0:
            raise ValueError("train_query_overlap must be a non-negative integer")
        if tuple(sorted(self.forbidden_identifier_hits)) != self.forbidden_identifier_hits:
            raise ValueError("forbidden identifier hits must be sorted")
        if len(self.forbidden_identifier_hits) != len(
            set(self.forbidden_identifier_hits)
        ):
            raise ValueError("forbidden identifier hits must be unique")


@dataclass(frozen=True)
class AdmissionEvidence:
    nuisance: NuisanceEvaluation
    calibration: CalibrationEvaluation
    near_miss: NearMissEvaluation
    anti_memorization: AntiMemorizationEvaluation
    archive_replay: tuple[ArchiveReplayReceipt, ...]
    candidate_replay: CandidateReplayReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_replay, CandidateReplayReceipt):
            raise TypeError("candidate_replay must be a verifier receipt")


@dataclass(frozen=True)
class AdmissionPolicy:
    required_nuisances: frozenset[Transform]
    candidate_replay_suite_digest: str
    incumbent_source_sha256: str
    minimum_nuisance_rate: float = 1.0
    minimum_calibration_gain: float = 0.0
    minimum_calibration_samples: int = 1
    minimum_near_miss_rate: float = 1.0
    minimum_anti_memorization_rate: float = 1.0
    maximum_ast_novelty: int | None = None

    def __post_init__(self) -> None:
        _check_sha256(
            self.candidate_replay_suite_digest,
            "candidate_replay_suite_digest",
        )
        _check_sha256(self.incumbent_source_sha256, "incumbent_source_sha256")
        for label, value in (
            ("minimum_nuisance_rate", self.minimum_nuisance_rate),
            ("minimum_near_miss_rate", self.minimum_near_miss_rate),
            ("minimum_anti_memorization_rate", self.minimum_anti_memorization_rate),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must lie in [0, 1]")
        if not math.isfinite(self.minimum_calibration_gain):
            raise ValueError("minimum_calibration_gain must be finite")
        if (
            isinstance(self.minimum_calibration_samples, bool)
            or self.minimum_calibration_samples <= 0
        ):
            raise ValueError("minimum_calibration_samples must be positive")
        if self.maximum_ast_novelty is not None and (
            isinstance(self.maximum_ast_novelty, bool)
            or self.maximum_ast_novelty < 0
        ):
            raise ValueError("maximum_ast_novelty must be non-negative")


@dataclass(frozen=True)
class AstNoveltyCharge:
    parsed: bool
    added_fingerprints: int
    removed_fingerprints: int
    incumbent_nodes: int
    candidate_nodes: int
    same_size_rewrite: bool
    charge: int | None
    error: str | None = None


def _ast_fingerprints(source: str) -> tuple[Counter[str], int, str]:
    tree = ast.parse(source)
    fingerprints = Counter(
        ast.dump(node, annotate_fields=True, include_attributes=False)
        for node in ast.walk(tree)
    )
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return fingerprints, sum(fingerprints.values()), _sha256_text(normalized)


def charge_ast_novelty(incumbent_source: str, candidate_source: str) -> AstNoveltyCharge:
    """Charge new AST structure, including replacements with zero size delta."""

    try:
        incumbent, incumbent_nodes, incumbent_digest = _ast_fingerprints(
            incumbent_source
        )
        candidate, candidate_nodes, candidate_digest = _ast_fingerprints(
            candidate_source
        )
    except SyntaxError as exc:
        return AstNoveltyCharge(
            parsed=False,
            added_fingerprints=0,
            removed_fingerprints=0,
            incumbent_nodes=0,
            candidate_nodes=0,
            same_size_rewrite=False,
            charge=None,
            error=f"{exc.msg} at line {exc.lineno}",
        )
    added = sum((candidate - incumbent).values())
    removed = sum((incumbent - candidate).values())
    same_size = (
        incumbent_nodes == candidate_nodes and incumbent_digest != candidate_digest
    )
    # Charging additions, rather than max(0, node-count delta), is what makes
    # an operator/name replacement at fixed AST size non-free.
    return AstNoveltyCharge(
        parsed=True,
        added_fingerprints=added,
        removed_fingerprints=removed,
        incumbent_nodes=incumbent_nodes,
        candidate_nodes=candidate_nodes,
        same_size_rewrite=same_size,
        charge=added,
    )


class GateName(str, Enum):
    TYPED_ATTACHMENT = "typed_attachment"
    SOURCE_BINDING = "source_binding"
    NUISANCE = "nuisance"
    CALIBRATION = "calibration_vs_baseline"
    NEAR_MISS = "near_miss_contrast"
    ANTI_MEMORIZATION = "anti_memorization"
    ARCHIVE_REPLAY = "full_archive_replay"
    CANDIDATE_REPLAY = "candidate_replay"
    NOVELTY = "ast_novelty"


@dataclass(frozen=True)
class GateResult:
    gate: GateName
    passed: bool
    detail: str


@dataclass(frozen=True)
class PromotionDecision:
    """One indivisible verdict and its compare-and-swap archive transition."""

    accepted: bool
    candidate_id: str
    candidate_digest: str
    attachment_contract_digest: str
    archive_before_digest: str
    archive_after: ArchivePreservationContract | None
    gates: tuple[GateResult, ...]
    novelty: AstNoveltyCharge

    def __post_init__(self) -> None:
        for label, value in (
            ("candidate_digest", self.candidate_digest),
            ("attachment_contract_digest", self.attachment_contract_digest),
            ("archive_before_digest", self.archive_before_digest),
        ):
            _check_sha256(value, label)
        expected_gates = tuple(GateName)
        if tuple(result.gate for result in self.gates) != expected_gates:
            raise ValueError("promotion decision must contain every gate in order")
        if self.accepted != all(result.passed for result in self.gates):
            raise ValueError("promotion verdict differs from its gates")
        if self.accepted and self.archive_after is None:
            raise ValueError("accepted decision requires a complete next archive")
        if not self.accepted and self.archive_after is not None:
            raise ValueError("rejected decision cannot contain an archive mutation")

    def digest(self) -> str:
        return _canonical_digest(
            {
                "accepted": self.accepted,
                "candidate_id": self.candidate_id,
                "candidate_digest": self.candidate_digest,
                "attachment_contract_digest": self.attachment_contract_digest,
                "archive_before_digest": self.archive_before_digest,
                "archive_after_digest": (
                    self.archive_after.digest() if self.archive_after else None
                ),
                "gates": [
                    {
                        "gate": result.gate.value,
                        "passed": result.passed,
                        "detail": result.detail,
                    }
                    for result in self.gates
                ],
                "novelty": asdict(self.novelty),
            }
        )


class AdmissionVerifier:
    """Evaluate all gates and construct one atomic promotion decision."""

    def __init__(
        self,
        *,
        verifier_id: str,
        registry: LegRegistry,
        attachment_contract: TypedAttachmentContract,
        archive_contract: ArchivePreservationContract,
        policy: AdmissionPolicy,
    ) -> None:
        if not verifier_id.strip():
            raise ValueError("verifier id must be non-empty")
        if attachment_contract.issued_by != verifier_id:
            raise ValueError("attachment contract is not owned by this verifier")
        if archive_contract.issued_by != verifier_id:
            raise ValueError("archive contract is not owned by this verifier")
        self.verifier_id = verifier_id
        self.registry = registry
        self.attachment_contract = attachment_contract
        self.archive_contract = archive_contract
        self.policy = policy

    def _receipt_identity(
        self, candidate_digest: str, evidence: AdmissionEvidence
    ) -> tuple[bool, str]:
        evaluations = (
            evidence.nuisance,
            evidence.calibration,
            evidence.near_miss,
            evidence.anti_memorization,
        )
        for evaluation in evaluations:
            if evaluation.issued_by != self.verifier_id:
                return False, "evaluation receipt issuer mismatch"
            if evaluation.candidate_digest != candidate_digest:
                return False, "evaluation receipt candidate mismatch"
        return True, "all evaluation receipts bind this candidate and verifier"

    def _source_binding(
        self, candidate: CandidateAttachment
    ) -> tuple[bool, str]:
        """Bind novelty text to verifier-owned incumbent and executable code."""

        incumbent_digest = _sha256_text(candidate.incumbent_source)
        if incumbent_digest != self.policy.incumbent_source_sha256:
            return False, "incumbent source differs from the verifier precommit"
        try:
            contract = self.registry.resolve(candidate.candidate_leg)
        except (KeyError, ValueError) as exc:
            return False, f"candidate leg is not the registered executable: {exc}"
        formula_legs = tuple(atom.call.leg for atom in atoms(candidate.formula))
        direct_calls = sum(
            1 for reference in formula_legs if reference == candidate.candidate_leg
        )
        if direct_calls == 0:
            return False, "candidate formula never directly calls candidate_leg"
        submitted_digest = _sha256_text(candidate.candidate_source.strip())
        if submitted_digest != contract.source_digest:
            return False, (
                "candidate source differs from the registered executable "
                f"for {candidate.candidate_leg.name}@{candidate.candidate_leg.version}"
            )
        return True, (
            "incumbent precommit and candidate executable source match; "
            f"direct_calls={direct_calls}"
        )

    def decide(
        self, candidate: CandidateAttachment, evidence: AdmissionEvidence
    ) -> PromotionDecision:
        candidate_digest = candidate.digest()
        identity_ok, identity_detail = self._receipt_identity(
            candidate_digest, evidence
        )

        try:
            self.attachment_contract.validate(candidate.formula, self.registry)
            typed_ok = identity_ok
            typed_detail = (
                "closed formula is typed and all receipts are identity-bound"
                if typed_ok
                else identity_detail
            )
        except (TypeError, ValueError) as exc:
            typed_ok = False
            typed_detail = str(exc)

        source_ok, source_detail = self._source_binding(candidate)
        source_ok = source_ok and identity_ok

        transforms = {trial.transform for trial in evidence.nuisance.trials}
        nuisance_successes = sum(
            1 for trial in evidence.nuisance.trials if trial.preserved
        )
        nuisance_rate = nuisance_successes / len(evidence.nuisance.trials)
        missing_transforms = self.policy.required_nuisances - transforms
        nuisance_ok = (
            identity_ok
            and not missing_transforms
            and nuisance_rate >= self.policy.minimum_nuisance_rate
        )
        nuisance_detail = (
            f"rate={nuisance_rate:.6f}; missing="
            f"{sorted(item.value for item in missing_transforms)}"
        )

        calibration_gain = (
            evidence.calibration.baseline_brier
            - evidence.calibration.candidate_brier
        )
        calibration_ok = (
            identity_ok
            and evidence.calibration.sample_count
            >= self.policy.minimum_calibration_samples
            and calibration_gain >= self.policy.minimum_calibration_gain
        )
        calibration_detail = (
            f"Brier gain={calibration_gain:.6f}; "
            f"n={evidence.calibration.sample_count}"
        )

        near_miss_rate = evidence.near_miss.successes / evidence.near_miss.total
        near_miss_ok = (
            identity_ok
            and near_miss_rate >= self.policy.minimum_near_miss_rate
        )
        near_miss_detail = f"contrast accuracy={near_miss_rate:.6f}"

        anti_rate = (
            evidence.anti_memorization.challenge_successes
            / evidence.anti_memorization.challenge_total
        )
        anti_ok = (
            identity_ok
            and anti_rate >= self.policy.minimum_anti_memorization_rate
            and evidence.anti_memorization.train_query_overlap == 0
            and not evidence.anti_memorization.forbidden_identifier_hits
        )
        anti_detail = (
            f"challenge accuracy={anti_rate:.6f}; "
            f"overlap={evidence.anti_memorization.train_query_overlap}; "
            f"identifier_hits={list(evidence.anti_memorization.forbidden_identifier_hits)}"
        )

        archive_ok, archive_detail = self.archive_contract.verify_full_replay(
            evidence.archive_replay
        )
        archive_ok = archive_ok and identity_ok

        candidate_replay = evidence.candidate_replay
        expected_formula_digest = formula_digest(candidate.formula)
        candidate_replay_ok = (
            identity_ok
            and candidate_replay.issued_by == self.verifier_id
            and candidate_replay.replay_suite_digest
            == self.policy.candidate_replay_suite_digest
            and candidate_replay.candidate_digest == candidate_digest
            and candidate_replay.formula_digest == expected_formula_digest
            and candidate_replay.passed
        )
        candidate_replay_detail = (
            f"passed={candidate_replay.passed}; cases={candidate_replay.case_count}; "
            f"suite_match={candidate_replay.replay_suite_digest == self.policy.candidate_replay_suite_digest}; "
            f"candidate_match={candidate_replay.candidate_digest == candidate_digest}; "
            f"formula_match={candidate_replay.formula_digest == expected_formula_digest}; "
            f"issuer_match={candidate_replay.issued_by == self.verifier_id}"
        )

        novelty = charge_ast_novelty(
            candidate.incumbent_source, candidate.candidate_source
        )
        novelty_ok = novelty.parsed and (
            self.policy.maximum_ast_novelty is None
            or (
                novelty.charge is not None
                and novelty.charge <= self.policy.maximum_ast_novelty
            )
        )
        novelty_detail = (
            f"charge={novelty.charge}; same_size_rewrite="
            f"{novelty.same_size_rewrite}"
            if novelty.parsed
            else f"unparseable source: {novelty.error}"
        )

        gates = (
            GateResult(GateName.TYPED_ATTACHMENT, typed_ok, typed_detail),
            GateResult(GateName.SOURCE_BINDING, source_ok, source_detail),
            GateResult(GateName.NUISANCE, nuisance_ok, nuisance_detail),
            GateResult(GateName.CALIBRATION, calibration_ok, calibration_detail),
            GateResult(GateName.NEAR_MISS, near_miss_ok, near_miss_detail),
            GateResult(GateName.ANTI_MEMORIZATION, anti_ok, anti_detail),
            GateResult(GateName.ARCHIVE_REPLAY, archive_ok, archive_detail),
            GateResult(
                GateName.CANDIDATE_REPLAY,
                candidate_replay_ok,
                candidate_replay_detail,
            ),
            GateResult(GateName.NOVELTY, novelty_ok, novelty_detail),
        )
        accepted = all(result.passed for result in gates)
        archive_after = None
        if accepted:
            archive_after = self.archive_contract.with_entry(
                ArchiveEntry(
                    attachment_id=candidate.attachment_id,
                    candidate_digest=candidate_digest,
                    formula_digest=expected_formula_digest,
                    replay_digest=candidate_replay.replay_digest,
                    case_count=candidate_replay.case_count,
                )
            )
        return PromotionDecision(
            accepted=accepted,
            candidate_id=candidate.attachment_id,
            candidate_digest=candidate_digest,
            attachment_contract_digest=self.attachment_contract.digest(),
            archive_before_digest=self.archive_contract.digest(),
            archive_after=archive_after,
            gates=gates,
            novelty=novelty,
        )


def apply_promotion(
    decision: PromotionDecision,
    current_archive: ArchivePreservationContract,
) -> ArchivePreservationContract:
    """Apply an accepted decision only to the exact archive it inspected."""

    if not decision.accepted or decision.archive_after is None:
        raise ValueError("cannot apply a rejected promotion decision")
    if current_archive.digest() != decision.archive_before_digest:
        raise ValueError("accepted archive changed after admission decision")
    return decision.archive_after

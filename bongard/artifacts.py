"""Immutable freeze/commit/reveal artifacts for canonical Bongard runs.

The dependency chain is the protocol:

``support -> proposal freeze -> two-query release -> cold inputs -> predictions
-> labels``.

Every arrow contains the digest of its parent.  Query labels occur only in the
last artifact.  Cold replay uses committed atom-level truth dispositions and
the closed Boolean IR; it never calls a vision model or a registered leg.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, NoReturn

from bongard.admission import TypedAttachmentContract
from bongard.evidence import Disposition, Evidence, Provenance, Uncertainty
from bongard.ir import (
    AllOf,
    AnyOf,
    Atom,
    Formula,
    formula_digest,
    formula_from_data,
)
from bongard.legs import LegReference, LegRegistry, Unit, ValueType
from bongard.legs.contracts import RegistrySnapshot


ARCHIVE_SCHEMA = "bongard-run-artifact-archive/v3"


class ArtifactTamperError(ValueError):
    """An immutable artifact chain no longer matches its commitments."""


def _validate_json_value(value: object, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite float")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path}: canonical JSON object keys must be strings")
        for key, item in value.items():
            _validate_json_value(item, f"{path}.{key}")
        return
    raise ValueError(f"{path}: unsupported canonical JSON value {type(value).__name__}")


def canonical_json(data: object) -> bytes:
    """Return the sole canonical JSON encoding used by protocol digests."""

    try:
        _validate_json_value(data)
        text = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"artifact is not canonical-JSON encodable: {exc}") from exc
    return text.encode("utf-8")


def canonical_digest(data: object) -> str:
    return hashlib.sha256(canonical_json(data)).hexdigest()


def _check_digest(value: str, label: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase sha256")


def _check_identifier(value: str, label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"invalid {label} {value!r}")


@dataclass(frozen=True, order=True)
class BlobRef:
    """Commitment to exact bytes stored outside the JSON control plane."""

    blob_id: str
    sha256: str
    byte_count: int
    media_type: str

    def __post_init__(self) -> None:
        _check_identifier(self.blob_id, "blob id")
        _check_digest(self.sha256, "blob sha256")
        if isinstance(self.byte_count, bool) or self.byte_count <= 0:
            raise ValueError("blob byte_count must be positive")
        if not self.media_type.strip() or "/" not in self.media_type:
            raise ValueError("blob media_type must be a MIME type")

    @classmethod
    def from_bytes(cls, blob_id: str, payload: bytes, media_type: str) -> "BlobRef":
        if not payload:
            raise ValueError("committed blob cannot be empty")
        return cls(
            blob_id,
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            media_type,
        )

    def verify_bytes(self, payload: bytes) -> None:
        if len(payload) != self.byte_count:
            raise ArtifactTamperError(f"{self.blob_id}: byte count changed")
        if hashlib.sha256(payload).hexdigest() != self.sha256:
            raise ArtifactTamperError(f"{self.blob_id}: bytes changed")

    def to_data(self) -> dict[str, object]:
        return {
            "blob_id": self.blob_id,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "media_type": self.media_type,
        }


@dataclass(frozen=True, order=True)
class SupportExample:
    panel: BlobRef
    positive: bool

    def to_data(self) -> dict[str, object]:
        return {"panel": self.panel.to_data(), "positive": self.positive}


@dataclass(frozen=True)
class SupportCommitment:
    run_id: str
    issued_by: str
    corpus_digest: str
    support: tuple[SupportExample, ...]
    verifier_nonce: str
    version: str = "support-commitment/v1"

    def __post_init__(self) -> None:
        _check_identifier(self.run_id, "run id")
        if not self.issued_by.strip():
            raise ValueError("support issuer must be non-empty")
        _check_digest(self.corpus_digest, "corpus_digest")
        if not self.verifier_nonce.strip():
            raise ValueError("support verifier nonce must be non-empty")
        if len(self.support) < 2:
            raise ValueError("support must contain examples from both classes")
        panel_ids = [item.panel.blob_id for item in self.support]
        if panel_ids != sorted(panel_ids) or len(panel_ids) != len(set(panel_ids)):
            raise ValueError("support panel ids must be unique and sorted")
        labels = {item.positive for item in self.support}
        if labels != {False, True}:
            raise ValueError("support must commit both positive and negative panels")
        if self.version != "support-commitment/v1":
            raise ValueError("unsupported support commitment version")

    def to_data(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "issued_by": self.issued_by,
            "corpus_digest": self.corpus_digest,
            "support": [item.to_data() for item in self.support],
            "verifier_nonce": self.verifier_nonce,
            "version": self.version,
        }

    def digest(self) -> str:
        return canonical_digest(self.to_data())


@dataclass(frozen=True)
class ProposalFreeze:
    """Formula commitment whose contents contain no query reference."""

    run_id: str
    proposal_id: str
    support_commitment_digest: str
    attachment_contract_digest: str
    registry_digest: str
    formula: Formula
    proposer_digest: str
    support_gate_digest: str
    verifier_nonce: str
    version: str = "proposal-freeze/v2"

    def __post_init__(self) -> None:
        _check_identifier(self.run_id, "run id")
        _check_identifier(self.proposal_id, "proposal id")
        for label, value in (
            ("support_commitment_digest", self.support_commitment_digest),
            ("attachment_contract_digest", self.attachment_contract_digest),
            ("registry_digest", self.registry_digest),
            ("proposer_digest", self.proposer_digest),
            ("support_gate_digest", self.support_gate_digest),
        ):
            _check_digest(value, label)
        if not self.verifier_nonce.strip():
            raise ValueError("proposal freeze verifier nonce must be non-empty")
        if self.version != "proposal-freeze/v2":
            raise ValueError("unsupported proposal freeze version")

    @classmethod
    def create(
        cls,
        *,
        support: SupportCommitment,
        proposal_id: str,
        formula: Formula,
        proposer_digest: str,
        attachment_contract: TypedAttachmentContract,
        registry: LegRegistry,
        support_gate_digest: str,
        verifier_nonce: str,
    ) -> "ProposalFreeze":
        if support.issued_by != attachment_contract.issued_by:
            raise ValueError("support and attachment contract issuers differ")
        attachment_contract.validate(formula, registry)
        return cls(
            run_id=support.run_id,
            proposal_id=proposal_id,
            support_commitment_digest=support.digest(),
            attachment_contract_digest=attachment_contract.digest(),
            registry_digest=registry.digest(),
            formula=formula,
            proposer_digest=proposer_digest,
            support_gate_digest=support_gate_digest,
            verifier_nonce=verifier_nonce,
        )

    def to_data(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "proposal_id": self.proposal_id,
            "support_commitment_digest": self.support_commitment_digest,
            "attachment_contract_digest": self.attachment_contract_digest,
            "registry_digest": self.registry_digest,
            "formula": self.formula.to_data(),
            "formula_digest": formula_digest(self.formula),
            "proposer_digest": self.proposer_digest,
            "support_gate_digest": self.support_gate_digest,
            "verifier_nonce": self.verifier_nonce,
            "version": self.version,
        }

    def digest(self) -> str:
        return canonical_digest(self.to_data())


@dataclass(frozen=True, order=True)
class QueryPanel:
    query_id: str
    panel: BlobRef

    def __post_init__(self) -> None:
        _check_identifier(self.query_id, "query id")

    def to_data(self) -> dict[str, object]:
        return {"query_id": self.query_id, "panel": self.panel.to_data()}


@dataclass(frozen=True)
class QueryRelease:
    """Exactly two unlabeled query panels released against a prior freeze."""

    run_id: str
    proposal_freeze_digest: str
    queries: tuple[QueryPanel, QueryPanel]
    verifier_nonce: str
    version: str = "two-query-release/v1"

    def __post_init__(self) -> None:
        _check_identifier(self.run_id, "run id")
        _check_digest(self.proposal_freeze_digest, "proposal_freeze_digest")
        if len(self.queries) != 2:
            raise ValueError("canonical release requires exactly two queries")
        ids = [query.query_id for query in self.queries]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("query ids must be unique and sorted")
        if len({query.panel.sha256 for query in self.queries}) != 2:
            raise ValueError("the two queries must commit distinct panel bytes")
        if not self.verifier_nonce.strip():
            raise ValueError("query release verifier nonce must be non-empty")
        if self.version != "two-query-release/v1":
            raise ValueError("unsupported query release version")

    @classmethod
    def create(
        cls,
        freeze: ProposalFreeze,
        queries: tuple[QueryPanel, QueryPanel],
        *,
        verifier_nonce: str,
    ) -> "QueryRelease":
        return cls(
            run_id=freeze.run_id,
            proposal_freeze_digest=freeze.digest(),
            queries=tuple(sorted(queries)),  # type: ignore[arg-type]
            verifier_nonce=verifier_nonce,
        )

    def to_data(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "proposal_freeze_digest": self.proposal_freeze_digest,
            "queries": [query.to_data() for query in self.queries],
            "verifier_nonce": self.verifier_nonce,
            "version": self.version,
        }

    def digest(self) -> str:
        return canonical_digest(self.to_data())


@dataclass(frozen=True)
class TruthEvidenceRecord:
    """Canonical JSON projection of an atom/final ``Evidence[bool]``."""

    disposition: Disposition
    provenance: Provenance
    value: bool | None = None
    uncertainty: Uncertainty | None = None
    certificate: str | None = None
    reason: str | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        # Reuse Evidence's exhaustive invariant checker.
        evidence = self.to_evidence()
        if evidence.disposition is Disposition.PRESENT and evidence.unwrap() is not True:
            raise ValueError("present truth evidence must contain True")

    @classmethod
    def from_evidence(cls, evidence: Evidence[bool]) -> "TruthEvidenceRecord":
        if evidence.disposition is Disposition.PRESENT and evidence.unwrap() is not True:
            raise ValueError("formula truth evidence can only present True")
        return cls(
            disposition=evidence.disposition,
            provenance=evidence.provenance,
            value=evidence.value,
            uncertainty=evidence.uncertainty,
            certificate=evidence.certificate,
            reason=evidence.reason,
            error_type=evidence.error_type,
        )

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "TruthEvidenceRecord":
        required = {
            "disposition",
            "provenance",
            "value",
            "uncertainty",
            "certificate",
            "reason",
            "error_type",
        }
        if set(data) != required:
            raise ValueError("truth evidence JSON has missing or unknown fields")
        provenance_data = data["provenance"]
        if not isinstance(provenance_data, Mapping):
            raise ValueError("truth evidence provenance must be an object")
        provenance_required = {
            "producer",
            "version",
            "method",
            "input_digests",
            "artifact_digest",
            "run_id",
            "details",
        }
        if set(provenance_data) != provenance_required:
            raise ValueError("provenance JSON has missing or unknown fields")
        uncertainty_data = data["uncertainty"]
        uncertainty = None
        if uncertainty_data is not None:
            if not isinstance(uncertainty_data, Mapping) or set(uncertainty_data) != {
                "lower",
                "upper",
                "confidence_level",
                "causes",
            }:
                raise ValueError("uncertainty JSON has missing or unknown fields")
            for field in ("lower", "upper"):
                if type(uncertainty_data[field]) is not float:
                    raise ValueError(
                        f"uncertainty {field} must be a literal canonical float"
                    )
            if (
                uncertainty_data["confidence_level"] is not None
                and type(uncertainty_data["confidence_level"]) is not float
            ):
                raise ValueError(
                    "uncertainty confidence_level must be a literal canonical float"
                )
            if not isinstance(uncertainty_data["causes"], list) or any(
                not isinstance(item, str) for item in uncertainty_data["causes"]
            ):
                raise ValueError("uncertainty causes must be a list of strings")
            uncertainty = Uncertainty(
                uncertainty_data["lower"],
                uncertainty_data["upper"],
                uncertainty_data["confidence_level"],
                tuple(uncertainty_data["causes"]),
            )
        value = data["value"]
        if value not in (None, True):
            raise ValueError("truth evidence value must be true or null")
        return cls(
            disposition=Disposition(str(data["disposition"])),
            provenance=Provenance(
                producer=str(provenance_data["producer"]),
                version=str(provenance_data["version"]),
                method=str(provenance_data["method"]),
                input_digests=tuple(
                    str(item) for item in provenance_data["input_digests"]
                ),
                artifact_digest=(
                    str(provenance_data["artifact_digest"])
                    if provenance_data["artifact_digest"] is not None
                    else None
                ),
                run_id=(
                    str(provenance_data["run_id"])
                    if provenance_data["run_id"] is not None
                    else None
                ),
                details=tuple(
                    (str(item[0]), str(item[1]))
                    for item in provenance_data["details"]
                ),
            ),
            value=value,
            uncertainty=uncertainty,
            certificate=(
                str(data["certificate"])
                if data["certificate"] is not None
                else None
            ),
            reason=str(data["reason"]) if data["reason"] is not None else None,
            error_type=(
                str(data["error_type"])
                if data["error_type"] is not None
                else None
            ),
        )

    def to_evidence(self) -> Evidence[bool]:
        return Evidence(
            disposition=self.disposition,
            provenance=self.provenance,
            value=self.value,
            uncertainty=self.uncertainty,
            certificate=self.certificate,
            reason=self.reason,
            error_type=self.error_type,
        )

    def to_data(self) -> dict[str, object]:
        return {
            "disposition": self.disposition.value,
            "provenance": self.provenance.canonical_data(),
            "value": self.value,
            "uncertainty": (
                {
                    "lower": self.uncertainty.lower,
                    "upper": self.uncertainty.upper,
                    "confidence_level": self.uncertainty.confidence_level,
                    "causes": list(self.uncertainty.causes),
                }
                if self.uncertainty is not None
                else None
            ),
            "certificate": self.certificate,
            "reason": self.reason,
            "error_type": self.error_type,
        }

    def digest(self) -> str:
        return canonical_digest(self.to_data())

    def __bool__(self) -> NoReturn:
        raise TypeError("truth evidence record cannot be coerced to bool")


AtomPath = tuple[int, ...]


def atom_paths(formula: Formula, prefix: AtomPath = ()) -> tuple[AtomPath, ...]:
    if isinstance(formula, Atom):
        return (prefix,)
    return tuple(
        path
        for index, term in enumerate(formula.terms)
        for path in atom_paths(term, (*prefix, index))
    )


@dataclass(frozen=True, order=True)
class AtomReplayInput:
    path: AtomPath
    evidence: TruthEvidenceRecord

    def __post_init__(self) -> None:
        if any(isinstance(index, bool) or index < 0 for index in self.path):
            raise ValueError("atom path indices must be non-negative integers")

    def to_data(self) -> dict[str, object]:
        return {"path": list(self.path), "evidence": self.evidence.to_data()}


@dataclass(frozen=True, order=True)
class QueryReplayInput:
    query_id: str
    panel_digest: str
    atom_inputs: tuple[AtomReplayInput, ...]

    def __post_init__(self) -> None:
        _check_identifier(self.query_id, "query id")
        _check_digest(self.panel_digest, "query panel_digest")
        paths = [item.path for item in self.atom_inputs]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("atom replay paths must be unique and sorted")

    def to_data(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "panel_digest": self.panel_digest,
            "atom_inputs": [item.to_data() for item in self.atom_inputs],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "QueryReplayInput":
        if set(data) != {"query_id", "panel_digest", "atom_inputs"}:
            raise ValueError("query replay JSON has missing or unknown fields")
        raw_inputs = data["atom_inputs"]
        if not isinstance(raw_inputs, list):
            raise ValueError("query atom_inputs must be a list")
        parsed: list[AtomReplayInput] = []
        for raw in raw_inputs:
            if not isinstance(raw, Mapping) or set(raw) != {"path", "evidence"}:
                raise ValueError("atom replay JSON has missing or unknown fields")
            evidence_data = raw["evidence"]
            if not isinstance(evidence_data, Mapping):
                raise ValueError("atom replay evidence must be an object")
            raw_path = raw["path"]
            if not isinstance(raw_path, list) or any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in raw_path
            ):
                raise ValueError("atom replay path must be a list of integers")
            parsed.append(
                AtomReplayInput(
                    tuple(raw_path),
                    TruthEvidenceRecord.from_data(evidence_data),
                )
            )
        return cls(
            str(data["query_id"]),
            str(data["panel_digest"]),
            tuple(parsed),
        )


@dataclass(frozen=True)
class ColdReplayInputs:
    """All empirical atom decisions needed for replay without a model."""

    proposal_freeze_digest: str
    query_release_digest: str
    formula_digest: str
    registry_digest: str
    queries: tuple[QueryReplayInput, QueryReplayInput]
    version: str = "cold-model-free-inputs/v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("proposal_freeze_digest", self.proposal_freeze_digest),
            ("query_release_digest", self.query_release_digest),
            ("formula_digest", self.formula_digest),
            ("registry_digest", self.registry_digest),
        ):
            _check_digest(value, label)
        if len(self.queries) != 2:
            raise ValueError("cold replay requires exactly two query inputs")
        ids = [query.query_id for query in self.queries]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("cold replay query ids must be unique and sorted")
        if self.version != "cold-model-free-inputs/v1":
            raise ValueError("unsupported cold replay input version")

    @classmethod
    def capture(
        cls,
        *,
        freeze: ProposalFreeze,
        release: QueryRelease,
        atom_evidence: Mapping[str, Mapping[AtomPath, Evidence[bool]]],
    ) -> "ColdReplayInputs":
        if release.proposal_freeze_digest != freeze.digest():
            raise ArtifactTamperError("query release is not bound to proposal freeze")
        expected_paths = set(atom_paths(freeze.formula))
        expected_ids = {query.query_id for query in release.queries}
        if set(atom_evidence) != expected_ids:
            raise ValueError("atom evidence must cover exactly both released queries")
        queries: list[QueryReplayInput] = []
        for query in release.queries:
            observed = atom_evidence[query.query_id]
            if set(observed) != expected_paths:
                raise ValueError(
                    f"{query.query_id}: atom evidence paths differ from frozen formula"
                )
            queries.append(
                QueryReplayInput(
                    query.query_id,
                    query.panel.sha256,
                    tuple(
                        AtomReplayInput(path, TruthEvidenceRecord.from_evidence(observed[path]))
                        for path in sorted(observed)
                    ),
                )
            )
        return cls(
            proposal_freeze_digest=freeze.digest(),
            query_release_digest=release.digest(),
            formula_digest=formula_digest(freeze.formula),
            registry_digest=freeze.registry_digest,
            queries=tuple(queries),  # type: ignore[arg-type]
        )

    def to_data(self) -> dict[str, object]:
        return {
            "proposal_freeze_digest": self.proposal_freeze_digest,
            "query_release_digest": self.query_release_digest,
            "formula_digest": self.formula_digest,
            "registry_digest": self.registry_digest,
            "queries": [query.to_data() for query in self.queries],
            "version": self.version,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ColdReplayInputs":
        required = {
            "proposal_freeze_digest",
            "query_release_digest",
            "formula_digest",
            "registry_digest",
            "queries",
            "version",
        }
        if set(data) != required:
            raise ValueError("cold replay JSON has missing or unknown fields")
        raw_queries = data["queries"]
        if not isinstance(raw_queries, list) or any(
            not isinstance(query, Mapping) for query in raw_queries
        ):
            raise ValueError("cold replay queries must be a list of objects")
        return cls(
            proposal_freeze_digest=str(data["proposal_freeze_digest"]),
            query_release_digest=str(data["query_release_digest"]),
            formula_digest=str(data["formula_digest"]),
            registry_digest=str(data["registry_digest"]),
            queries=tuple(
                QueryReplayInput.from_data(query) for query in raw_queries
            ),  # type: ignore[arg-type]
            version=str(data["version"]),
        )

    def digest(self) -> str:
        return canonical_digest(self.to_data())


def _composed_provenance(
    method: str, evidence: tuple[Evidence[bool], ...]
) -> Provenance:
    return Provenance.composed(
        "bongard.closed_ir",
        "1",
        method,
        tuple(item.provenance for item in evidence),
    )


def _replay_composite(
    formula: AllOf | AnyOf, results: tuple[Evidence[bool], ...]
) -> Evidence[bool]:
    method = "and" if isinstance(formula, AllOf) else "or"
    provenance = _composed_provenance(method, results)
    errors = [item for item in results if item.disposition is Disposition.ERROR]
    if errors:
        first = errors[0]
        return Evidence.error(
            provenance,
            first.error_type or "ChildError",
            first.reason or f"{method} child failed",
        )
    if isinstance(formula, AllOf):
        absent = [
            item
            for item in results
            if item.disposition is Disposition.CERTIFIED_ABSENT
        ]
        if absent:
            return Evidence.certified_absent(
                provenance,
                "conjunct certified absent: "
                + (absent[0].certificate or "unspecified certificate"),
            )
        if any(item.disposition is Disposition.INDETERMINATE for item in results):
            return Evidence.indeterminate(
                provenance, "one or more conjuncts are indeterminate"
            )
        return Evidence.present(True, provenance)
    if any(item.disposition is Disposition.PRESENT for item in results):
        return Evidence.present(True, provenance)
    if any(item.disposition is Disposition.INDETERMINATE for item in results):
        return Evidence.indeterminate(
            provenance, "no disjunct is present and at least one is indeterminate"
        )
    return Evidence.certified_absent(provenance, "every disjunct is certified absent")


def replay_query(
    formula: Formula, query: QueryReplayInput, path: AtomPath = ()
) -> Evidence[bool]:
    """Replay the Boolean cone strictly from committed atom decisions."""

    by_path = {item.path: item.evidence for item in query.atom_inputs}

    def visit(node: Formula, current_path: AtomPath) -> Evidence[bool]:
        if isinstance(node, Atom):
            try:
                return by_path[current_path].to_evidence()
            except KeyError as exc:
                raise ArtifactTamperError(
                    f"{query.query_id}: missing atom path {current_path}"
                ) from exc
        children = tuple(
            visit(term, (*current_path, index))
            for index, term in enumerate(node.terms)
        )
        return _replay_composite(node, children)

    expected = set(atom_paths(formula, path))
    if set(by_path) != expected:
        raise ArtifactTamperError(
            f"{query.query_id}: cold inputs do not match frozen formula atoms"
        )
    return visit(formula, path)


def replay_cold_payload(
    formula_data: Mapping[str, Any], cold_inputs_data: Mapping[str, Any]
) -> tuple[tuple[str, TruthEvidenceRecord], ...]:
    """Replay a decoded on-disk payload with no registry, leg, or model.

    The caller may obtain both mappings with ``json.loads`` from independently
    stored canonical JSON bytes.  Exact formula and cold-input digests are
    checked before any result is returned.
    """

    formula = formula_from_data(formula_data)
    cold_inputs = ColdReplayInputs.from_data(cold_inputs_data)
    if formula_digest(formula) != cold_inputs.formula_digest:
        raise ArtifactTamperError("cold payload formula digest mismatch")
    expected_paths = set(atom_paths(formula))
    results: list[tuple[str, TruthEvidenceRecord]] = []
    for query in cold_inputs.queries:
        if {item.path for item in query.atom_inputs} != expected_paths:
            raise ArtifactTamperError(
                f"{query.query_id}: cold payload atom paths mismatch"
            )
        results.append(
            (
                query.query_id,
                TruthEvidenceRecord.from_evidence(replay_query(formula, query)),
            )
        )
    return tuple(results)


@dataclass(frozen=True, order=True)
class PredictionRecord:
    query_id: str
    positive: bool | None
    disposition: Disposition
    evidence_digest: str

    def __post_init__(self) -> None:
        _check_identifier(self.query_id, "query id")
        _check_digest(self.evidence_digest, "prediction evidence_digest")
        if self.disposition is Disposition.PRESENT and self.positive is not True:
            raise ValueError("present formula evidence predicts positive")
        if (
            self.disposition is Disposition.CERTIFIED_ABSENT
            and self.positive is not False
        ):
            raise ValueError("certified absence predicts negative")
        if self.disposition in (Disposition.INDETERMINATE, Disposition.ERROR) and (
            self.positive is not None
        ):
            raise ValueError("indeterminate/error prediction must abstain")

    @classmethod
    def from_evidence(
        cls, query_id: str, evidence: Evidence[bool]
    ) -> "PredictionRecord":
        record = TruthEvidenceRecord.from_evidence(evidence)
        positive: bool | None
        if evidence.disposition is Disposition.PRESENT:
            positive = True
        elif evidence.disposition is Disposition.CERTIFIED_ABSENT:
            positive = False
        else:
            positive = None
        return cls(query_id, positive, evidence.disposition, record.digest())

    def to_data(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "positive": self.positive,
            "disposition": self.disposition.value,
            "evidence_digest": self.evidence_digest,
        }


@dataclass(frozen=True)
class PredictionCommitment:
    """Both query predictions committed while labels remain unavailable."""

    run_id: str
    proposal_freeze_digest: str
    query_release_digest: str
    cold_replay_inputs_digest: str
    predictions: tuple[PredictionRecord, PredictionRecord]
    verifier_nonce: str
    version: str = "two-query-prediction-commitment/v1"

    def __post_init__(self) -> None:
        _check_identifier(self.run_id, "run id")
        for label, value in (
            ("proposal_freeze_digest", self.proposal_freeze_digest),
            ("query_release_digest", self.query_release_digest),
            ("cold_replay_inputs_digest", self.cold_replay_inputs_digest),
        ):
            _check_digest(value, label)
        if len(self.predictions) != 2:
            raise ValueError("prediction commitment requires exactly two predictions")
        ids = [prediction.query_id for prediction in self.predictions]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("prediction query ids must be unique and sorted")
        if not self.verifier_nonce.strip():
            raise ValueError("prediction verifier nonce must be non-empty")
        if self.version != "two-query-prediction-commitment/v1":
            raise ValueError("unsupported prediction commitment version")

    @classmethod
    def create(
        cls,
        *,
        freeze: ProposalFreeze,
        release: QueryRelease,
        cold_inputs: ColdReplayInputs,
        verifier_nonce: str,
    ) -> "PredictionCommitment":
        if release.proposal_freeze_digest != freeze.digest():
            raise ArtifactTamperError("release does not descend from proposal freeze")
        if cold_inputs.proposal_freeze_digest != freeze.digest():
            raise ArtifactTamperError("cold inputs do not descend from proposal freeze")
        if cold_inputs.query_release_digest != release.digest():
            raise ArtifactTamperError("cold inputs do not bind query release")
        query_map = {query.query_id: query for query in cold_inputs.queries}
        predictions = tuple(
            PredictionRecord.from_evidence(
                query.query_id,
                replay_query(freeze.formula, query_map[query.query_id]),
            )
            for query in release.queries
        )
        return cls(
            run_id=freeze.run_id,
            proposal_freeze_digest=freeze.digest(),
            query_release_digest=release.digest(),
            cold_replay_inputs_digest=cold_inputs.digest(),
            predictions=predictions,  # type: ignore[arg-type]
            verifier_nonce=verifier_nonce,
        )

    def to_data(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "proposal_freeze_digest": self.proposal_freeze_digest,
            "query_release_digest": self.query_release_digest,
            "cold_replay_inputs_digest": self.cold_replay_inputs_digest,
            "predictions": [prediction.to_data() for prediction in self.predictions],
            "verifier_nonce": self.verifier_nonce,
            "version": self.version,
        }

    def digest(self) -> str:
        return canonical_digest(self.to_data())


@dataclass(frozen=True, order=True)
class RevealedLabel:
    query_id: str
    positive: bool

    def __post_init__(self) -> None:
        _check_identifier(self.query_id, "query id")

    def to_data(self) -> dict[str, object]:
        return {"query_id": self.query_id, "positive": self.positive}


@dataclass(frozen=True)
class LabelReveal:
    """The only protocol object permitted to contain query labels."""

    run_id: str
    prediction_commitment_digest: str
    labels: tuple[RevealedLabel, RevealedLabel]
    verifier_nonce: str
    version: str = "label-reveal/v1"

    def __post_init__(self) -> None:
        _check_identifier(self.run_id, "run id")
        _check_digest(
            self.prediction_commitment_digest, "prediction_commitment_digest"
        )
        if len(self.labels) != 2:
            raise ValueError("label reveal requires exactly two labels")
        ids = [label.query_id for label in self.labels]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("revealed label ids must be unique and sorted")
        if not self.verifier_nonce.strip():
            raise ValueError("label reveal verifier nonce must be non-empty")
        if self.version != "label-reveal/v1":
            raise ValueError("unsupported label reveal version")

    @classmethod
    def create(
        cls,
        commitment: PredictionCommitment,
        labels: tuple[RevealedLabel, RevealedLabel],
        *,
        verifier_nonce: str,
    ) -> "LabelReveal":
        expected = {prediction.query_id for prediction in commitment.predictions}
        if {label.query_id for label in labels} != expected:
            raise ValueError("labels must cover exactly the committed predictions")
        return cls(
            run_id=commitment.run_id,
            prediction_commitment_digest=commitment.digest(),
            labels=tuple(sorted(labels)),  # type: ignore[arg-type]
            verifier_nonce=verifier_nonce,
        )

    def to_data(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "prediction_commitment_digest": self.prediction_commitment_digest,
            "labels": [label.to_data() for label in self.labels],
            "verifier_nonce": self.verifier_nonce,
            "version": self.version,
        }

    def digest(self) -> str:
        return canonical_digest(self.to_data())


@dataclass(frozen=True)
class ModelFreeReplayReceipt:
    run_id: str
    chain_digest: str
    predictions_match: bool
    determinate_correct: int
    determinate_total: int
    abstentions: int
    query_evidence_digests: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _check_identifier(self.run_id, "run id")
        _check_digest(self.chain_digest, "chain_digest")
        if min(self.determinate_correct, self.determinate_total, self.abstentions) < 0:
            raise ValueError("replay counts must be non-negative")
        if self.determinate_correct > self.determinate_total:
            raise ValueError("correct replay count exceeds determinate total")

    def to_data(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "chain_digest": self.chain_digest,
            "predictions_match": self.predictions_match,
            "determinate_correct": self.determinate_correct,
            "determinate_total": self.determinate_total,
            "abstentions": self.abstentions,
            "query_evidence_digests": [list(item) for item in self.query_evidence_digests],
        }

    def digest(self) -> str:
        return canonical_digest(self.to_data())


def _expect_fields(
    data: Mapping[str, Any], expected: set[str], label: str
) -> None:
    missing = expected - set(data)
    extra = set(data) - expected
    if missing or extra:
        raise ArtifactTamperError(
            f"{label} fields differ from schema: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ArtifactTamperError(f"{label} must be a JSON object")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ArtifactTamperError(f"{label} must be a JSON list")
    return value


def _blob_from_data(value: object) -> BlobRef:
    data = _mapping(value, "blob reference")
    _expect_fields(data, {"blob_id", "sha256", "byte_count", "media_type"}, "blob")
    try:
        return BlobRef(
            blob_id=str(data["blob_id"]),
            sha256=str(data["sha256"]),
            byte_count=data["byte_count"],
            media_type=str(data["media_type"]),
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactTamperError(f"invalid blob reference: {exc}") from exc


def _support_from_data(value: object) -> SupportCommitment:
    data = _mapping(value, "support commitment")
    _expect_fields(
        data,
        {
            "run_id",
            "issued_by",
            "corpus_digest",
            "support",
            "verifier_nonce",
            "version",
        },
        "support commitment",
    )
    support: list[SupportExample] = []
    for raw in _list(data["support"], "support examples"):
        item = _mapping(raw, "support example")
        _expect_fields(item, {"panel", "positive"}, "support example")
        if not isinstance(item["positive"], bool):
            raise ArtifactTamperError("support label must be boolean")
        support.append(SupportExample(_blob_from_data(item["panel"]), item["positive"]))
    try:
        return SupportCommitment(
            run_id=str(data["run_id"]),
            issued_by=str(data["issued_by"]),
            corpus_digest=str(data["corpus_digest"]),
            support=tuple(support),
            verifier_nonce=str(data["verifier_nonce"]),
            version=str(data["version"]),
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactTamperError(f"invalid support commitment: {exc}") from exc


def _value_type_from_data(value: object) -> ValueType:
    data = _mapping(value, "value type")
    _expect_fields(data, {"name", "unit"}, "value type")
    try:
        return ValueType.from_data(data)
    except (TypeError, ValueError) as exc:
        raise ArtifactTamperError(f"invalid value type: {exc}") from exc


def _leg_reference_from_data(value: object) -> LegReference:
    data = _mapping(value, "leg reference")
    _expect_fields(data, {"name", "version", "contract_digest"}, "leg reference")
    try:
        return LegReference(
            str(data["name"]),
            str(data["version"]),
            str(data["contract_digest"]),
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactTamperError(f"invalid leg reference: {exc}") from exc


def _attachment_contract_from_data(value: object) -> TypedAttachmentContract:
    """Decode the static contract without constructing executable legs."""

    data = _mapping(value, "typed attachment contract")
    _expect_fields(
        data,
        {
            "issued_by",
            "registry_digest",
            "registry_snapshot",
            "boundary_types",
            "allowed_legs",
            "ir_version",
        },
        "typed attachment contract",
    )
    boundary: list[tuple[str, ValueType]] = []
    for raw in _list(data["boundary_types"], "attachment boundary types"):
        if not isinstance(raw, list) or len(raw) != 2:
            raise ArtifactTamperError("attachment boundary entry must be [name, type]")
        boundary.append((str(raw[0]), _value_type_from_data(raw[1])))
    allowed = tuple(
        _leg_reference_from_data(raw)
        for raw in _list(data["allowed_legs"], "attachment allowed legs")
    )
    try:
        snapshot = RegistrySnapshot.from_data(data["registry_snapshot"])
    except (TypeError, ValueError) as exc:
        raise ArtifactTamperError(f"invalid registry snapshot: {exc}") from exc
    try:
        return TypedAttachmentContract(
            issued_by=str(data["issued_by"]),
            registry_digest=str(data["registry_digest"]),
            registry_snapshot=snapshot,
            boundary_types=tuple(boundary),
            allowed_legs=allowed,
            ir_version=str(data["ir_version"]),
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactTamperError(f"invalid typed attachment contract: {exc}") from exc


def _freeze_from_data(value: object) -> ProposalFreeze:
    data = _mapping(value, "proposal freeze")
    _expect_fields(
        data,
        {
            "run_id",
            "proposal_id",
            "support_commitment_digest",
            "attachment_contract_digest",
            "registry_digest",
            "formula",
            "formula_digest",
            "proposer_digest",
            "support_gate_digest",
            "verifier_nonce",
            "version",
        },
        "proposal freeze",
    )
    formula_data = _mapping(data["formula"], "frozen formula")
    try:
        formula = formula_from_data(formula_data)
    except (TypeError, ValueError) as exc:
        raise ArtifactTamperError(f"invalid frozen formula: {exc}") from exc
    if str(data["formula_digest"]) != formula_digest(formula):
        raise ArtifactTamperError("frozen formula content differs from formula_digest")
    try:
        return ProposalFreeze(
            run_id=str(data["run_id"]),
            proposal_id=str(data["proposal_id"]),
            support_commitment_digest=str(data["support_commitment_digest"]),
            attachment_contract_digest=str(data["attachment_contract_digest"]),
            registry_digest=str(data["registry_digest"]),
            formula=formula,
            proposer_digest=str(data["proposer_digest"]),
            support_gate_digest=str(data["support_gate_digest"]),
            verifier_nonce=str(data["verifier_nonce"]),
            version=str(data["version"]),
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactTamperError(f"invalid proposal freeze: {exc}") from exc


def verify_support_commitment_data(value: object) -> SupportCommitment:
    """Decode one support commitment without weakening its JSON types.

    Full run archives validate this object as part of their hash chain.  An
    episode that stops before query release has no full archive, so the outer
    protocol persists this same nonce-bearing preimage directly.  The exact
    round-trip check prevents the decoder's defensive string conversions from
    accepting a differently typed JSON object as the committed value.
    """

    support = _support_from_data(value)
    data = _mapping(value, "support commitment")
    if support.to_data() != dict(data):
        raise ArtifactTamperError(
            "support commitment does not reproduce from its archived fields"
        )
    return support


def verify_proposal_freeze_data(value: object) -> ProposalFreeze:
    """Decode and exactly reproduce a standalone pre-query proposal freeze."""

    freeze = _freeze_from_data(value)
    data = _mapping(value, "proposal freeze")
    if freeze.to_data() != dict(data):
        raise ArtifactTamperError(
            "proposal freeze does not reproduce from its archived fields"
        )
    return freeze


def _query_panel_from_data(value: object) -> QueryPanel:
    data = _mapping(value, "query panel")
    _expect_fields(data, {"query_id", "panel"}, "query panel")
    try:
        return QueryPanel(str(data["query_id"]), _blob_from_data(data["panel"]))
    except (TypeError, ValueError) as exc:
        raise ArtifactTamperError(f"invalid query panel: {exc}") from exc


def _release_from_data(value: object) -> QueryRelease:
    data = _mapping(value, "query release")
    _expect_fields(
        data,
        {
            "run_id",
            "proposal_freeze_digest",
            "queries",
            "verifier_nonce",
            "version",
        },
        "query release",
    )
    queries = tuple(
        _query_panel_from_data(raw)
        for raw in _list(data["queries"], "released queries")
    )
    try:
        return QueryRelease(
            run_id=str(data["run_id"]),
            proposal_freeze_digest=str(data["proposal_freeze_digest"]),
            queries=queries,  # type: ignore[arg-type]
            verifier_nonce=str(data["verifier_nonce"]),
            version=str(data["version"]),
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactTamperError(f"invalid query release: {exc}") from exc


def _prediction_record_from_data(value: object) -> PredictionRecord:
    data = _mapping(value, "prediction")
    _expect_fields(
        data,
        {"query_id", "positive", "disposition", "evidence_digest"},
        "prediction",
    )
    positive = data["positive"]
    if positive is not None and not isinstance(positive, bool):
        raise ArtifactTamperError("prediction positive must be boolean or null")
    try:
        return PredictionRecord(
            query_id=str(data["query_id"]),
            positive=positive,
            disposition=Disposition(str(data["disposition"])),
            evidence_digest=str(data["evidence_digest"]),
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactTamperError(f"invalid prediction: {exc}") from exc


def _predictions_from_data(value: object) -> PredictionCommitment:
    data = _mapping(value, "prediction commitment")
    _expect_fields(
        data,
        {
            "run_id",
            "proposal_freeze_digest",
            "query_release_digest",
            "cold_replay_inputs_digest",
            "predictions",
            "verifier_nonce",
            "version",
        },
        "prediction commitment",
    )
    predictions = tuple(
        _prediction_record_from_data(raw)
        for raw in _list(data["predictions"], "predictions")
    )
    try:
        return PredictionCommitment(
            run_id=str(data["run_id"]),
            proposal_freeze_digest=str(data["proposal_freeze_digest"]),
            query_release_digest=str(data["query_release_digest"]),
            cold_replay_inputs_digest=str(data["cold_replay_inputs_digest"]),
            predictions=predictions,  # type: ignore[arg-type]
            verifier_nonce=str(data["verifier_nonce"]),
            version=str(data["version"]),
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactTamperError(f"invalid prediction commitment: {exc}") from exc


def _labels_from_data(value: object) -> LabelReveal:
    data = _mapping(value, "label reveal")
    _expect_fields(
        data,
        {
            "run_id",
            "prediction_commitment_digest",
            "labels",
            "verifier_nonce",
            "version",
        },
        "label reveal",
    )
    labels: list[RevealedLabel] = []
    for raw in _list(data["labels"], "revealed labels"):
        item = _mapping(raw, "revealed label")
        _expect_fields(item, {"query_id", "positive"}, "revealed label")
        if not isinstance(item["positive"], bool):
            raise ArtifactTamperError("revealed label must be boolean")
        labels.append(RevealedLabel(str(item["query_id"]), item["positive"]))
    try:
        return LabelReveal(
            run_id=str(data["run_id"]),
            prediction_commitment_digest=str(data["prediction_commitment_digest"]),
            labels=tuple(labels),  # type: ignore[arg-type]
            verifier_nonce=str(data["verifier_nonce"]),
            version=str(data["version"]),
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactTamperError(f"invalid label reveal: {exc}") from exc


def _receipt_from_data(value: object) -> ModelFreeReplayReceipt:
    data = _mapping(value, "model-free replay receipt")
    _expect_fields(
        data,
        {
            "run_id",
            "chain_digest",
            "predictions_match",
            "determinate_correct",
            "determinate_total",
            "abstentions",
            "query_evidence_digests",
        },
        "model-free replay receipt",
    )
    if not isinstance(data["predictions_match"], bool):
        raise ArtifactTamperError("receipt predictions_match must be boolean")
    counts = (
        data["determinate_correct"],
        data["determinate_total"],
        data["abstentions"],
    )
    if any(isinstance(item, bool) or not isinstance(item, int) for item in counts):
        raise ArtifactTamperError("receipt counts must be integers")
    evidence_digests: list[tuple[str, str]] = []
    for raw in _list(data["query_evidence_digests"], "receipt evidence digests"):
        if not isinstance(raw, list) or len(raw) != 2:
            raise ArtifactTamperError("receipt evidence entry must be [query_id, digest]")
        _check_identifier(str(raw[0]), "receipt query id")
        _check_digest(str(raw[1]), "receipt evidence digest")
        evidence_digests.append((str(raw[0]), str(raw[1])))
    try:
        return ModelFreeReplayReceipt(
            run_id=str(data["run_id"]),
            chain_digest=str(data["chain_digest"]),
            predictions_match=data["predictions_match"],
            determinate_correct=counts[0],
            determinate_total=counts[1],
            abstentions=counts[2],
            query_evidence_digests=tuple(evidence_digests),
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactTamperError(f"invalid model-free replay receipt: {exc}") from exc


@dataclass(frozen=True)
class RunArtifactBundle:
    support: SupportCommitment
    attachment_contract: TypedAttachmentContract
    freeze: ProposalFreeze
    release: QueryRelease
    cold_inputs: ColdReplayInputs
    predictions: PredictionCommitment
    labels: LabelReveal

    def chain_data(self) -> dict[str, str]:
        return {
            "support": self.support.digest(),
            "attachment_contract": self.attachment_contract.digest(),
            "freeze": self.freeze.digest(),
            "release": self.release.digest(),
            "cold_inputs": self.cold_inputs.digest(),
            "predictions": self.predictions.digest(),
            "labels": self.labels.digest(),
        }

    def digest(self) -> str:
        return canonical_digest(self.chain_data())

    def to_archive_data(self) -> dict[str, object]:
        """Serialize every frozen component plus the verified replay receipt."""

        receipt = self.verify()
        content: dict[str, object] = {
            "schema": ARCHIVE_SCHEMA,
            "support": self.support.to_data(),
            "attachment_contract": self.attachment_contract.to_data(),
            "proposal_freeze": self.freeze.to_data(),
            "query_release": self.release.to_data(),
            "cold_replay_inputs": self.cold_inputs.to_data(),
            "prediction_commitment": self.predictions.to_data(),
            "label_reveal": self.labels.to_data(),
            "chain": self.chain_data(),
            "chain_digest": self.digest(),
            "model_free_replay_receipt": receipt.to_data(),
        }
        return {**content, "archive_digest": canonical_digest(content)}

    def to_archive_bytes(self) -> bytes:
        """Return the exact canonical UTF-8 JSON persistence payload."""

        return canonical_json(self.to_archive_data())

    def to_archive_json(self) -> str:
        return self.to_archive_bytes().decode("utf-8")

    def verify(self) -> ModelFreeReplayReceipt:
        """Verify the full chain and replay without invoking any model/leg."""

        run_ids = {
            self.support.run_id,
            self.freeze.run_id,
            self.release.run_id,
            self.predictions.run_id,
            self.labels.run_id,
        }
        if len(run_ids) != 1:
            raise ArtifactTamperError("run ids differ across artifact chain")
        if self.freeze.support_commitment_digest != self.support.digest():
            raise ArtifactTamperError("proposal freeze does not bind support")
        if self.freeze.attachment_contract_digest != self.attachment_contract.digest():
            raise ArtifactTamperError("proposal freeze does not bind attachment contract")
        if self.freeze.registry_digest != self.attachment_contract.registry_digest:
            raise ArtifactTamperError("proposal freeze registry digest changed")
        try:
            self.attachment_contract.validate_static(self.freeze.formula)
        except (TypeError, ValueError) as exc:
            raise ArtifactTamperError(
                f"frozen formula fails offline static validation: {exc}"
            ) from exc
        if self.release.proposal_freeze_digest != self.freeze.digest():
            raise ArtifactTamperError("query was not released against frozen proposal")
        if self.cold_inputs.proposal_freeze_digest != self.freeze.digest():
            raise ArtifactTamperError("cold inputs use a different proposal")
        if self.cold_inputs.query_release_digest != self.release.digest():
            raise ArtifactTamperError("cold inputs use a different query release")
        if self.cold_inputs.formula_digest != formula_digest(self.freeze.formula):
            raise ArtifactTamperError("cold inputs use a different formula")
        if self.cold_inputs.registry_digest != self.freeze.registry_digest:
            raise ArtifactTamperError("cold inputs use a different registry")
        if self.predictions.proposal_freeze_digest != self.freeze.digest():
            raise ArtifactTamperError("predictions use a different proposal")
        if self.predictions.query_release_digest != self.release.digest():
            raise ArtifactTamperError("predictions use a different query release")
        if self.predictions.cold_replay_inputs_digest != self.cold_inputs.digest():
            raise ArtifactTamperError("cold replay inputs changed after prediction")
        if self.labels.prediction_commitment_digest != self.predictions.digest():
            raise ArtifactTamperError("labels were not revealed after predictions")

        support_digests = {item.panel.sha256 for item in self.support.support}
        if any(query.panel.sha256 in support_digests for query in self.release.queries):
            raise ArtifactTamperError("query bytes overlap committed support")
        released = {query.query_id: query.panel.sha256 for query in self.release.queries}
        replayed = {query.query_id: query.panel_digest for query in self.cold_inputs.queries}
        if replayed != released:
            raise ArtifactTamperError("cold inputs do not bind exact query panel bytes")

        prediction_map = {
            prediction.query_id: prediction for prediction in self.predictions.predictions
        }
        evidence_digests: list[tuple[str, str]] = []
        predictions_match = True
        for query in self.cold_inputs.queries:
            evidence = replay_query(self.freeze.formula, query)
            observed = PredictionRecord.from_evidence(query.query_id, evidence)
            evidence_digests.append((query.query_id, observed.evidence_digest))
            if observed != prediction_map.get(query.query_id):
                predictions_match = False
        if not predictions_match:
            raise ArtifactTamperError("committed predictions do not cold-replay")

        label_map = {label.query_id: label.positive for label in self.labels.labels}
        if set(label_map) != set(prediction_map):
            raise ArtifactTamperError("revealed labels do not cover predictions")
        determinate = [
            prediction
            for prediction in self.predictions.predictions
            if prediction.positive is not None
        ]
        correct = sum(
            prediction.positive == label_map[prediction.query_id]
            for prediction in determinate
        )
        return ModelFreeReplayReceipt(
            run_id=self.support.run_id,
            chain_digest=self.digest(),
            predictions_match=True,
            determinate_correct=correct,
            determinate_total=len(determinate),
            abstentions=2 - len(determinate),
            query_evidence_digests=tuple(evidence_digests),
        )


@dataclass(frozen=True)
class VerifiedRunArchive:
    """A decoded archive whose complete chain and cold replay have passed."""

    bundle: RunArtifactBundle
    replay_receipt: ModelFreeReplayReceipt
    archive_digest: str

    def __post_init__(self) -> None:
        _check_digest(self.archive_digest, "archive_digest")


_ARCHIVE_FIELDS = {
    "schema",
    "support",
    "attachment_contract",
    "proposal_freeze",
    "query_release",
    "cold_replay_inputs",
    "prediction_commitment",
    "label_reveal",
    "chain",
    "chain_digest",
    "model_free_replay_receipt",
    "archive_digest",
}


def verify_archive_data(data: Mapping[str, Any]) -> VerifiedRunArchive:
    """Verify a decoded full archive without a model or executable registry."""

    archive = _mapping(data, "run artifact archive")
    _expect_fields(archive, _ARCHIVE_FIELDS, "run artifact archive")
    if archive["schema"] != ARCHIVE_SCHEMA:
        raise ArtifactTamperError(
            f"unsupported run artifact archive schema {archive['schema']!r}"
        )
    archived_digest = str(archive["archive_digest"])
    _check_digest(archived_digest, "archive_digest")
    content = {key: value for key, value in archive.items() if key != "archive_digest"}
    if canonical_digest(content) != archived_digest:
        raise ArtifactTamperError("run artifact archive digest mismatch")

    support = _support_from_data(archive["support"])
    attachment = _attachment_contract_from_data(archive["attachment_contract"])
    freeze = _freeze_from_data(archive["proposal_freeze"])
    release = _release_from_data(archive["query_release"])
    try:
        cold_inputs = ColdReplayInputs.from_data(
            _mapping(archive["cold_replay_inputs"], "cold replay inputs")
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ArtifactTamperError):
            raise
        raise ArtifactTamperError(f"invalid cold replay inputs: {exc}") from exc
    predictions = _predictions_from_data(archive["prediction_commitment"])
    labels = _labels_from_data(archive["label_reveal"])
    bundle = RunArtifactBundle(
        support=support,
        attachment_contract=attachment,
        freeze=freeze,
        release=release,
        cold_inputs=cold_inputs,
        predictions=predictions,
        labels=labels,
    )

    # Static contract checking does not depend on any parent-link digest.  Run
    # it directly on the decoded archive so an independently resealed formula
    # still has to satisfy the archived signed/type/unit contract.
    try:
        attachment.validate_static(freeze.formula)
    except (TypeError, ValueError) as exc:
        raise ArtifactTamperError(
            f"frozen formula fails offline static validation: {exc}"
        ) from exc

    chain = _mapping(archive["chain"], "artifact digest chain")
    expected_chain = bundle.chain_data()
    _expect_fields(chain, set(expected_chain), "artifact digest chain")
    if dict(chain) != expected_chain:
        raise ArtifactTamperError("archived component digests differ from full components")
    chain_digest = str(archive["chain_digest"])
    _check_digest(chain_digest, "chain_digest")
    if chain_digest != bundle.digest():
        raise ArtifactTamperError("archived chain digest differs from component chain")

    # This performs every parent-link, support/query disjointness, formula,
    # cold-evidence, prediction, and label-coverage check without invoking a
    # registry or model.
    replay_receipt = bundle.verify()
    archived_receipt = _receipt_from_data(archive["model_free_replay_receipt"])
    if archived_receipt != replay_receipt:
        raise ArtifactTamperError("archived replay receipt does not reproduce")
    return VerifiedRunArchive(bundle, replay_receipt, archived_digest)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactTamperError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def verify_archive_bytes(payload: bytes | str) -> VerifiedRunArchive:
    """Decode and verify exact canonical archive bytes.

    Unlike :func:`verify_archive_data`, this entry point can also reject
    duplicate object keys and non-canonical whitespace/key/number encodings.
    """

    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    try:
        decoded = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
    except ArtifactTamperError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ArtifactTamperError(f"cannot decode run artifact archive: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise ArtifactTamperError("run artifact archive root must be a JSON object")
    try:
        expected = canonical_json(decoded)
    except ValueError as exc:
        raise ArtifactTamperError(str(exc)) from exc
    if raw != expected:
        raise ArtifactTamperError("run artifact archive bytes are not canonical JSON")
    return verify_archive_data(decoded)

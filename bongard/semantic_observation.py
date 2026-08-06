"""One-panel execution and cold replay for compiled visual semantics.

The compiler freezes a canonical Python closed formula with at most two runtime
boundaries: a deterministic :class:`VisualWitnessBundle` and, when the
proposal contains a soft claim, one :class:`BlindSoftScoreRecord`.  This module
materializes those boundaries from one exact neutral PNG, evaluates every
outer atom with the pure-Python reference backend, and archives the atom-level
and composed formula evidence.

Task, side, role, and label metadata are never inputs to extraction or witness
summarization.  The verifier-only scorer context is retained in the blind
transport artifact but the scorer transport itself enforces that none of it
enters the model-visible prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Callable, Mapping

from bongard.artifacts import (
    AtomPath,
    AtomReplayInput,
    QueryReplayInput,
    TruthEvidenceRecord,
    atom_paths,
    canonical_digest,
    canonical_json,
)
from bongard.blind_soft_transport import (
    BlindSoftScoreTransportArtifact,
    BlindSoftVerifierContext,
    VerifierWitnessSummary,
    canonical_witness_summaries,
    score_blind_soft_panel,
)
from bongard.evidence import Evidence, Provenance
from bongard.ir import Atom, Formula, formula_digest, formula_from_data
from bongard.legs import FROZEN_VISUAL_SCORE, TypedValue
from bongard.predicate_backend import PYTHON_PREDICATE_BACKEND
from bongard.semantic_synthesis import (
    DIRECT_BOUNDARY_NAME,
    SOFT_BOUNDARY_NAME,
    CompiledVisualSemanticProposal,
)
from bongard.soft_predicates import SoftScorerProtocol
from bongard.transport import (
    MAX_PANEL_PNG_BYTES,
    CloudPolicyCacheSnapshot,
    run_codex_named_images_structured,
)
from bongard.visual_witness_summaries import (
    visual_joint_soft_witness_interface_digest,
    visual_witness_summaries,
)
from bongard.visual_witness_bundle import (
    VISUAL_WITNESS_BUNDLE,
    VisualWitnessBundle,
    extract_visual_witness_bundle,
    verify_visual_witness_bundle,
)


SEMANTIC_OBSERVATION_SCHEMA = "gkm.bongard-visual-semantic-observation.v2"
SEMANTIC_ATOM_EVIDENCE_SCHEMA = "gkm.bongard-semantic-atom-evidence.v1"
SEMANTIC_OBSERVATION_ALGORITHM_ID = "python-compiled-visual-semantics/v2"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

StructuredTransport = Callable[..., Any]
PanelInput = str | Path | bytes


class SemanticObservationError(ValueError):
    """A compiled proposal, panel, protocol, or archive binding is invalid."""


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SemanticObservationError(f"{label} must be a lowercase SHA-256")
    return value


def _canonical_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise SemanticObservationError(f"{label} must be a JSON object")
    try:
        decoded = json.loads(canonical_json(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise SemanticObservationError(f"{label} is not canonical JSON") from exc
    if not isinstance(decoded, dict):  # pragma: no cover - Mapping guarantees it.
        raise SemanticObservationError(f"{label} must decode as an object")
    return decoded


def _read_panel_bytes(value: PanelInput) -> bytes:
    if isinstance(value, bytes):
        payload = value
    else:
        try:
            path = Path(value).resolve(strict=True)
            before = path.stat()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise SemanticObservationError("panel PNG does not exist") from exc
        if not stat.S_ISREG(before.st_mode):
            raise SemanticObservationError("panel PNG must be a regular file")
        try:
            payload = path.read_bytes()
            after = path.stat()
        except OSError as exc:
            raise SemanticObservationError("panel PNG could not be read") from exc
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity:
            raise SemanticObservationError("panel PNG changed while it was read")
    if not 0 < len(payload) <= MAX_PANEL_PNG_BYTES:
        raise SemanticObservationError("panel PNG exceeds the fixed byte guard")
    if not payload.startswith(_PNG_SIGNATURE):
        raise SemanticObservationError("panel input is not a PNG")
    return payload


def _atom_at_path(formula: Formula, path: AtomPath) -> Atom:
    node = formula
    for index in path:
        if isinstance(node, Atom) or index >= len(node.terms):
            raise SemanticObservationError("atom path is outside the frozen formula")
        node = node.terms[index]
    if not isinstance(node, Atom):
        raise SemanticObservationError("atom path does not terminate at an atom")
    return node


def semantic_atom_at_path(formula: Formula, path: AtomPath) -> Atom:
    """Resolve one atom in closed IR using the archive's canonical path rules."""

    return _atom_at_path(formula, path)


def _evaluation_error(
    *,
    panel_digest: str,
    compiled: CompiledVisualSemanticProposal,
    path: AtomPath,
    error: Exception,
) -> Evidence[bool]:
    provenance = Provenance(
        producer="bongard.semantic_observation",
        version="1",
        method="python_atom_evaluation_error",
        input_digests=(
            panel_digest,
            compiled.lowering_archive.digest,
            formula_digest(compiled.formula),
        ),
        details=(("atom_path", ".".join(str(item) for item in path) or "root"),),
    )
    return Evidence.error(
        provenance,
        type(error).__name__,
        str(error) or repr(error),
    )


def _semantic_runtime_bindings(
    compiled: CompiledVisualSemanticProposal,
    witness_bundle: VisualWitnessBundle,
    scorer_artifact: BlindSoftScoreTransportArtifact | None,
) -> dict[str, TypedValue]:
    bindings: dict[str, TypedValue] = {}
    if DIRECT_BOUNDARY_NAME in compiled.boundary_types:
        bindings[DIRECT_BOUNDARY_NAME] = TypedValue(
            VISUAL_WITNESS_BUNDLE, witness_bundle
        )
    if SOFT_BOUNDARY_NAME in compiled.boundary_types:
        if scorer_artifact is None:
            raise SemanticObservationError(
                "compiled soft boundary has no blind score artifact"
            )
        bindings[SOFT_BOUNDARY_NAME] = TypedValue(
            FROZEN_VISUAL_SCORE, scorer_artifact.record
        )
    if set(bindings) != set(compiled.boundary_types):
        raise SemanticObservationError(
            "compiled proposal contains an unsupported runtime boundary"
        )
    return bindings


@dataclass(frozen=True, order=True, slots=True)
class SemanticAtomEvidence:
    """Canonical evidence for one exact path in the compiled outer formula."""

    path: AtomPath
    evidence: TruthEvidenceRecord

    def __post_init__(self) -> None:
        if not isinstance(self.path, tuple) or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.path
        ):
            raise SemanticObservationError(
                "semantic atom path must contain non-negative integers"
            )
        if not isinstance(self.evidence, TruthEvidenceRecord):
            raise TypeError("semantic atom evidence must be TruthEvidenceRecord")

    @classmethod
    def from_evidence(
        cls, path: AtomPath, evidence: Evidence[bool]
    ) -> "SemanticAtomEvidence":
        return cls(path, TruthEvidenceRecord.from_evidence(evidence))

    def to_data(self) -> dict[str, object]:
        return {
            "schema": SEMANTIC_ATOM_EVIDENCE_SCHEMA,
            "path": list(self.path),
            "evidence": self.evidence.to_data(),
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "SemanticAtomEvidence":
        if not isinstance(value, Mapping) or set(value) != {
            "schema",
            "path",
            "evidence",
        }:
            raise SemanticObservationError(
                "semantic atom evidence fields differ from the static schema"
            )
        if value["schema"] != SEMANTIC_ATOM_EVIDENCE_SCHEMA:
            raise SemanticObservationError("unsupported semantic atom evidence")
        raw_path = value["path"]
        raw_evidence = value["evidence"]
        if not isinstance(raw_path, list) or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in raw_path
        ):
            raise SemanticObservationError("semantic atom path must be an integer list")
        if not isinstance(raw_evidence, Mapping):
            raise SemanticObservationError("semantic atom evidence must be an object")
        return cls(tuple(raw_path), TruthEvidenceRecord.from_data(raw_evidence))


@dataclass(frozen=True)
class VisualSemanticObservationArtifact:
    """Content-addressed bundle, score, atom, and formula observation archive."""

    panel_digest: str
    panel_byte_count: int
    pre_observation_commitment_digest: str
    proposal_digest: str
    policy_digest: str
    prospective_protocol_digest: str
    scorer_family_digest: str
    family_development_manifest_digest: str
    lowering_archive_digest: str
    formula_data: Mapping[str, Any]
    compiled_formula_digest: str
    registry_digest: str
    attachment_digest: str
    witness_bundle: VisualWitnessBundle
    witness_summaries: tuple[VerifierWitnessSummary, ...]
    scorer_artifact: BlindSoftScoreTransportArtifact | None
    atom_evidence: tuple[SemanticAtomEvidence, ...]
    formula_evidence: TruthEvidenceRecord
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "panel_digest",
            "pre_observation_commitment_digest",
            "proposal_digest",
            "policy_digest",
            "prospective_protocol_digest",
            "scorer_family_digest",
            "family_development_manifest_digest",
            "lowering_archive_digest",
            "compiled_formula_digest",
            "registry_digest",
            "attachment_digest",
        ):
            _digest(getattr(self, name), name)
        if (
            isinstance(self.panel_byte_count, bool)
            or not isinstance(self.panel_byte_count, int)
            or self.panel_byte_count <= 0
            or self.panel_byte_count > MAX_PANEL_PNG_BYTES
        ):
            raise SemanticObservationError("panel_byte_count is outside the guard")
        if not isinstance(self.witness_bundle, VisualWitnessBundle):
            raise TypeError("witness_bundle must be VisualWitnessBundle")
        verified = verify_visual_witness_bundle(self.witness_bundle)
        if verified.panel_digest != self.panel_digest:
            raise SemanticObservationError("witness bundle belongs to another panel")
        if not isinstance(self.witness_summaries, tuple) or any(
            not isinstance(item, VerifierWitnessSummary)
            for item in self.witness_summaries
        ):
            raise TypeError("witness_summaries must be a typed tuple")
        canonical_summaries = canonical_witness_summaries(self.witness_summaries)
        if canonical_summaries != self.witness_summaries:
            raise SemanticObservationError("witness summaries are not canonical")
        formula_data = _canonical_object(self.formula_data, "formula_data")
        formula = formula_from_data(formula_data)
        if formula.to_data() != formula_data:
            raise SemanticObservationError("formula data is not canonical closed IR")
        if formula_digest(formula) != self.compiled_formula_digest:
            raise SemanticObservationError("compiled formula digest differs")
        object.__setattr__(self, "formula_data", formula_data)
        formula_atoms = tuple(
            _atom_at_path(formula, path) for path in atom_paths(formula)
        )
        boundary_names = {
            name for atom in formula_atoms for name in atom.call.arguments
        }
        if not boundary_names <= {DIRECT_BOUNDARY_NAME, SOFT_BOUNDARY_NAME}:
            raise SemanticObservationError(
                "semantic observation formula names an unknown runtime boundary"
            )
        soft_boundary_required = SOFT_BOUNDARY_NAME in boundary_names
        if (self.scorer_artifact is not None) != soft_boundary_required:
            raise SemanticObservationError(
                "soft scorer artifact does not match the frozen formula boundary"
            )
        if not isinstance(self.atom_evidence, tuple) or any(
            not isinstance(item, SemanticAtomEvidence) for item in self.atom_evidence
        ):
            raise TypeError("atom_evidence must be a typed tuple")
        paths = tuple(item.path for item in self.atom_evidence)
        if paths != atom_paths(formula):
            raise SemanticObservationError(
                "atom evidence does not cover the frozen formula in canonical order"
            )
        if not isinstance(self.formula_evidence, TruthEvidenceRecord):
            raise TypeError("formula_evidence must be TruthEvidenceRecord")
        replay_input = QueryReplayInput(
            "panel",
            self.panel_digest,
            tuple(
                AtomReplayInput(item.path, item.evidence)
                for item in self.atom_evidence
            ),
        )
        replayed = TruthEvidenceRecord.from_evidence(
            PYTHON_PREDICATE_BACKEND.replay_query(formula, replay_input)
        )
        if replayed != self.formula_evidence:
            raise SemanticObservationError(
                "formula evidence differs from Python atom replay"
            )
        if self.scorer_artifact is not None:
            if not isinstance(
                self.scorer_artifact, BlindSoftScoreTransportArtifact
            ):
                raise TypeError(
                    "scorer_artifact must be BlindSoftScoreTransportArtifact or null"
                )
            self.scorer_artifact.assert_untampered()
            if self.scorer_artifact.protocol_digest != (
                self.prospective_protocol_digest
            ):
                raise SemanticObservationError(
                    "scorer artifact belongs to another protocol"
                )
            if self.scorer_artifact.panel.content_digest != self.panel_digest:
                raise SemanticObservationError(
                    "scorer artifact belongs to another panel"
                )
            if self.scorer_artifact.witness_packet_digest != (
                self.witness_bundle.digest()
            ):
                raise SemanticObservationError(
                    "scorer artifact belongs to another witness bundle"
                )
            if self.scorer_artifact.witness_summaries != self.witness_summaries:
                raise SemanticObservationError(
                    "scorer artifact witness summaries differ"
                )
            if self.scorer_artifact.context.pre_observation_commitment_digest != (
                self.pre_observation_commitment_digest
            ):
                raise SemanticObservationError(
                    "scorer artifact belongs to another pre-observation commitment"
                )
        object.__setattr__(self, "_sealed_digest", self.digest)

    @property
    def transport_attempted(self) -> bool:
        """True exactly when the compiled proposal required soft scoring."""

        return self.scorer_artifact is not None

    @property
    def witness_bundle_digest(self) -> str:
        return self.witness_bundle.digest()

    @property
    def witness_summaries_digest(self) -> str:
        return canonical_digest(
            [item.to_data() for item in self.witness_summaries]
        )

    @property
    def scorer_artifact_digest(self) -> str | None:
        return None if self.scorer_artifact is None else self.scorer_artifact.digest

    def evidence_by_path(self) -> dict[AtomPath, Evidence[bool]]:
        return {
            item.path: item.evidence.to_evidence() for item in self.atom_evidence
        }

    def content_data(self) -> dict[str, object]:
        return {
            "schema": SEMANTIC_OBSERVATION_SCHEMA,
            "algorithm_id": SEMANTIC_OBSERVATION_ALGORITHM_ID,
            "backend_semantics": "python_closed_ir_atom_evaluation_and_replay",
            "panel": {
                "neutral_name": "query.png",
                "media_type": "image/png",
                "byte_count": self.panel_byte_count,
                "content_digest": self.panel_digest,
            },
            "pre_observation_commitment_digest": (
                self.pre_observation_commitment_digest
            ),
            "proposal_digest": self.proposal_digest,
            "policy_digest": self.policy_digest,
            "prospective_protocol_digest": self.prospective_protocol_digest,
            "scorer_family_digest": self.scorer_family_digest,
            "family_development_manifest_digest": (
                self.family_development_manifest_digest
            ),
            "lowering_archive_digest": self.lowering_archive_digest,
            "formula": dict(self.formula_data),
            "compiled_formula_digest": self.compiled_formula_digest,
            "registry_digest": self.registry_digest,
            "attachment_digest": self.attachment_digest,
            "witness_interface_digest": visual_joint_soft_witness_interface_digest(),
            "witness_bundle": self.witness_bundle.to_data(),
            "witness_bundle_digest": self.witness_bundle_digest,
            "witness_summaries": [
                item.to_data() for item in self.witness_summaries
            ],
            "witness_summaries_digest": self.witness_summaries_digest,
            "soft_scoring_attempted": self.transport_attempted,
            "scorer_artifact": (
                None
                if self.scorer_artifact is None
                else self.scorer_artifact.to_data()
            ),
            "scorer_artifact_digest": self.scorer_artifact_digest,
            "atom_evidence": [item.to_data() for item in self.atom_evidence],
            "formula_evidence": self.formula_evidence.to_data(),
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "artifact_digest": self.digest}

    def assert_untampered(self) -> None:
        verify_visual_witness_bundle(self.witness_bundle)
        if self.scorer_artifact is not None:
            self.scorer_artifact.assert_untampered()
        if self.digest != self._sealed_digest:
            raise SemanticObservationError(
                "visual semantic observation changed after sealing"
            )

    def assert_matches(
        self,
        compiled: CompiledVisualSemanticProposal,
        protocol: SoftScorerProtocol,
    ) -> None:
        _validate_compiled_protocol(compiled, protocol)
        expected = {
            "proposal_digest": compiled.proposal.digest,
            "policy_digest": compiled.policy.digest(),
            "prospective_protocol_digest": protocol.digest(),
            "scorer_family_digest": compiled.family.digest(),
            "family_development_manifest_digest": (
                compiled.family.development_manifest_digest
            ),
            "lowering_archive_digest": compiled.lowering_archive.digest,
            "compiled_formula_digest": formula_digest(compiled.formula),
            "registry_digest": compiled.registry.digest(),
            "attachment_digest": compiled.attachment_contract.digest(),
        }
        for name, wanted in expected.items():
            if getattr(self, name) != wanted:
                raise SemanticObservationError(
                    f"observation {name} differs from compiled proposal"
                )
        if self.formula_data != compiled.formula.to_data():
            raise SemanticObservationError(
                "observation formula differs from compiled proposal"
            )
        soft_required = compiled.lowering_archive.soft_lowering is not None
        if self.transport_attempted is not soft_required:
            raise SemanticObservationError(
                "soft scorer attempt differs from compiled proposal"
            )
        if soft_required:
            assert self.scorer_artifact is not None
            claim = compiled.lowering_archive.soft_lowering
            assert claim is not None
            if self.scorer_artifact.claim != claim.claim:
                raise SemanticObservationError(
                    "scorer artifact claim differs from compiled proposal"
                )
        self.assert_untampered()

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        compiled: CompiledVisualSemanticProposal,
        protocol: SoftScorerProtocol,
        expected_digest: str | None = None,
        panel_png: PanelInput | None = None,
    ) -> "VisualSemanticObservationArtifact":
        """Cold-decode, replay atom evidence, and optionally replay exact pixels."""

        fields = {
            "schema",
            "algorithm_id",
            "backend_semantics",
            "panel",
            "pre_observation_commitment_digest",
            "proposal_digest",
            "policy_digest",
            "prospective_protocol_digest",
            "scorer_family_digest",
            "family_development_manifest_digest",
            "lowering_archive_digest",
            "formula",
            "compiled_formula_digest",
            "registry_digest",
            "attachment_digest",
            "witness_interface_digest",
            "witness_bundle",
            "witness_bundle_digest",
            "witness_summaries",
            "witness_summaries_digest",
            "soft_scoring_attempted",
            "scorer_artifact",
            "scorer_artifact_digest",
            "atom_evidence",
            "formula_evidence",
            "artifact_digest",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise SemanticObservationError(
                "semantic observation fields differ from the static schema"
            )
        data = dict(value)
        if data["schema"] != SEMANTIC_OBSERVATION_SCHEMA or data[
            "algorithm_id"
        ] != SEMANTIC_OBSERVATION_ALGORITHM_ID:
            raise SemanticObservationError("unsupported semantic observation")
        if data["backend_semantics"] != (
            "python_closed_ir_atom_evaluation_and_replay"
        ):
            raise SemanticObservationError("semantic observation backend drift")
        if data["witness_interface_digest"] != (
            visual_joint_soft_witness_interface_digest()
        ):
            raise SemanticObservationError("visual witness interface drift")
        panel = data["panel"]
        if not isinstance(panel, Mapping) or set(panel) != {
            "neutral_name",
            "media_type",
            "byte_count",
            "content_digest",
        }:
            raise SemanticObservationError("semantic panel identity is malformed")
        if panel["neutral_name"] != "query.png" or panel["media_type"] != (
            "image/png"
        ):
            raise SemanticObservationError("semantic panel presentation drift")
        raw_bundle = data["witness_bundle"]
        if not isinstance(raw_bundle, Mapping):
            raise SemanticObservationError("witness_bundle must be an object")
        bundle = VisualWitnessBundle.from_data(raw_bundle)
        if bundle.digest() != _digest(
            data["witness_bundle_digest"], "witness_bundle_digest"
        ):
            raise SemanticObservationError("witness bundle digest differs")
        raw_summaries = data["witness_summaries"]
        if not isinstance(raw_summaries, list) or any(
            not isinstance(item, Mapping) for item in raw_summaries
        ):
            raise SemanticObservationError(
                "witness_summaries must be an object list"
            )
        summaries = tuple(
            VerifierWitnessSummary.from_data(item) for item in raw_summaries
        )
        if canonical_digest([item.to_data() for item in summaries]) != _digest(
            data["witness_summaries_digest"], "witness_summaries_digest"
        ):
            raise SemanticObservationError("witness summaries digest differs")
        attempted = data["soft_scoring_attempted"]
        if not isinstance(attempted, bool):
            raise SemanticObservationError("soft_scoring_attempted must be Boolean")
        raw_scorer = data["scorer_artifact"]
        archived_scorer_digest = data["scorer_artifact_digest"]
        if attempted:
            if not isinstance(raw_scorer, Mapping):
                raise SemanticObservationError(
                    "attempted soft scoring requires a scorer artifact"
                )
            scorer_digest = _digest(
                archived_scorer_digest, "scorer_artifact_digest"
            )
            scorer = BlindSoftScoreTransportArtifact.from_data(
                raw_scorer,
                expected_digest=scorer_digest,
                expected_protocol_digest=protocol.digest(),
            )
        else:
            if raw_scorer is not None or archived_scorer_digest is not None:
                raise SemanticObservationError(
                    "unattempted soft scoring cannot carry a scorer artifact"
                )
            scorer = None
        raw_atoms = data["atom_evidence"]
        if not isinstance(raw_atoms, list) or any(
            not isinstance(item, Mapping) for item in raw_atoms
        ):
            raise SemanticObservationError("atom_evidence must be an object list")
        raw_formula_evidence = data["formula_evidence"]
        if not isinstance(raw_formula_evidence, Mapping):
            raise SemanticObservationError("formula_evidence must be an object")
        raw_formula = data["formula"]
        if not isinstance(raw_formula, Mapping):
            raise SemanticObservationError("formula must be an object")
        result = cls(
            panel_digest=panel["content_digest"],
            panel_byte_count=panel["byte_count"],
            pre_observation_commitment_digest=(
                data["pre_observation_commitment_digest"]
            ),
            proposal_digest=data["proposal_digest"],
            policy_digest=data["policy_digest"],
            prospective_protocol_digest=data["prospective_protocol_digest"],
            scorer_family_digest=data["scorer_family_digest"],
            family_development_manifest_digest=(
                data["family_development_manifest_digest"]
            ),
            lowering_archive_digest=data["lowering_archive_digest"],
            formula_data=raw_formula,
            compiled_formula_digest=data["compiled_formula_digest"],
            registry_digest=data["registry_digest"],
            attachment_digest=data["attachment_digest"],
            witness_bundle=bundle,
            witness_summaries=summaries,
            scorer_artifact=scorer,
            atom_evidence=tuple(
                SemanticAtomEvidence.from_data(item) for item in raw_atoms
            ),
            formula_evidence=TruthEvidenceRecord.from_data(raw_formula_evidence),
        )
        result.assert_matches(compiled, protocol)
        archived_digest = _digest(data["artifact_digest"], "artifact_digest")
        if result.digest != archived_digest:
            raise SemanticObservationError("semantic observation digest differs")
        if expected_digest is not None and result.digest != _digest(
            expected_digest, "expected observation digest"
        ):
            raise SemanticObservationError(
                "semantic observation differs from expected digest"
            )
        if canonical_json(result.to_data()) != canonical_json(data):
            raise SemanticObservationError(
                "semantic observation is not canonically represented"
            )
        if panel_png is not None:
            exact_bytes = _read_panel_bytes(panel_png)
            if len(exact_bytes) != result.panel_byte_count or hashlib.sha256(
                exact_bytes
            ).hexdigest() != result.panel_digest:
                raise SemanticObservationError(
                    "exact panel bytes differ from observation identity"
                )
            verify_visual_witness_bundle(bundle, expected_png_bytes=exact_bytes)
            expected_summaries = canonical_witness_summaries(
                visual_witness_summaries(bundle, expected_png_bytes=exact_bytes)
            )
            if expected_summaries != summaries:
                raise SemanticObservationError(
                    "witness summaries differ from exact panel replay"
                )
        return result

    def to_support_gate_measurement(self):
        """Adapt the formula result to the benchmark support-gate interface."""

        from bongard.benchmark import SupportGateMeasurement

        return SupportGateMeasurement(
            evidence=self.formula_evidence.to_evidence(),
            observer_artifact=self.to_data(),
            transport_attempted=self.transport_attempted,
        )


def semantic_boundary_bindings(
    artifact: VisualSemanticObservationArtifact,
    compiled: CompiledVisualSemanticProposal,
) -> dict[str, TypedValue]:
    """Rebuild the authoritative Python runtime boundary map from an archive."""

    if not isinstance(artifact, VisualSemanticObservationArtifact):
        raise TypeError("artifact must be VisualSemanticObservationArtifact")
    artifact.assert_matches(compiled, compiled.family.protocol)
    return _semantic_runtime_bindings(
        compiled,
        artifact.witness_bundle,
        artifact.scorer_artifact,
    )


def replay_semantic_atom_evidence(
    artifact: VisualSemanticObservationArtifact,
    compiled: CompiledVisualSemanticProposal,
) -> tuple[SemanticAtomEvidence, ...]:
    """Re-evaluate every registered atom with no model call or proof assistant."""

    bindings = semantic_boundary_bindings(artifact, compiled)
    replayed: list[SemanticAtomEvidence] = []
    for path in atom_paths(compiled.formula):
        atom = semantic_atom_at_path(compiled.formula, path)
        atom_bindings = {name: bindings[name] for name in atom.call.arguments}
        try:
            evidence = PYTHON_PREDICATE_BACKEND.evaluate(
                atom, compiled.registry, atom_bindings
            )
        except Exception as exc:  # Preserve the same fail-closed observation rule.
            evidence = _evaluation_error(
                panel_digest=artifact.panel_digest,
                compiled=compiled,
                path=path,
                error=exc,
            )
        replayed.append(SemanticAtomEvidence.from_evidence(path, evidence))
    return tuple(replayed)


def _validate_compiled_protocol(
    compiled: CompiledVisualSemanticProposal,
    protocol: SoftScorerProtocol,
) -> None:
    if not isinstance(compiled, CompiledVisualSemanticProposal):
        raise TypeError("compiled must be CompiledVisualSemanticProposal")
    if not isinstance(protocol, SoftScorerProtocol):
        raise TypeError("protocol must be SoftScorerProtocol")
    protocol.assert_untampered()
    compiled.family.assert_untampered()
    compiled.family.verify_calibration()
    if protocol.digest() != compiled.family.protocol_digest:
        raise SemanticObservationError(
            "prospective protocol differs from compiled scorer family"
        )
    expected = (
        (compiled.policy.soft_scorer_protocol_digest, protocol.digest()),
        (compiled.policy.soft_scorer_family_digest, compiled.family.digest()),
        (
            compiled.policy.soft_family_development_manifest_digest,
            compiled.family.development_manifest_digest,
        ),
        (compiled.lowering_archive.scorer_protocol_digest, protocol.digest()),
        (compiled.lowering_archive.scorer_family_digest, compiled.family.digest()),
    )
    if any(actual != wanted for actual, wanted in expected):
        raise SemanticObservationError(
            "compiled semantic dependencies differ from the prospective protocol"
        )
    compiled.attachment_contract.validate(compiled.formula, compiled.registry)
    PYTHON_PREDICATE_BACKEND.validate(
        compiled.formula, compiled.registry, compiled.boundary_types
    )


def observe_visual_semantic_panel(
    panel_png: PanelInput,
    compiled: CompiledVisualSemanticProposal,
    *,
    protocol: SoftScorerProtocol,
    context: BlindSoftVerifierContext,
    pre_observation_commitment_digest: str,
    model: str | None = None,
    reasoning_effort: str | None = None,
    minutes: int = 10,
    verbose: bool = False,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    transport: StructuredTransport = run_codex_named_images_structured,
) -> VisualSemanticObservationArtifact:
    """Observe one exact neutral panel under one already-compiled proposal."""

    _validate_compiled_protocol(compiled, protocol)
    if not isinstance(context, BlindSoftVerifierContext):
        raise TypeError("context must be BlindSoftVerifierContext")
    precommit = _digest(
        pre_observation_commitment_digest,
        "pre_observation_commitment_digest",
    )
    if context.pre_observation_commitment_digest != precommit:
        raise SemanticObservationError(
            "verifier context belongs to another pre-observation commitment"
        )
    panel_bytes = _read_panel_bytes(panel_png)
    panel_digest = hashlib.sha256(panel_bytes).hexdigest()

    # The extractor and summarizer receive exact bytes/bundle only.  They have
    # no API slots through which task, side, role, or label metadata can enter.
    bundle = extract_visual_witness_bundle(panel_bytes)
    verify_visual_witness_bundle(bundle, expected_png_bytes=panel_bytes)
    summary_pairs = visual_witness_summaries(
        bundle, expected_png_bytes=panel_bytes
    )
    summaries = canonical_witness_summaries(summary_pairs)

    scorer_artifact: BlindSoftScoreTransportArtifact | None = None
    soft_lowering = compiled.lowering_archive.soft_lowering
    if soft_lowering is not None:
        # Always restage the already-read bytes under the neutral name.  Thus
        # extraction and scoring consume the same immutable byte preimage even
        # when the caller supplied a metadata-bearing source path.
        with tempfile.TemporaryDirectory(
            prefix="bongard-semantic-observation-"
        ) as raw_dir:
            neutral_path = Path(raw_dir) / "query.png"
            neutral_path.write_bytes(panel_bytes)
            scorer_artifact = score_blind_soft_panel(
                neutral_path,
                soft_lowering.claim,
                protocol=protocol,
                witness_packet_digest=bundle.digest(),
                witness_summaries=summaries,
                context=context,
                model=model,
                reasoning_effort=reasoning_effort,
                minutes=minutes,
                verbose=verbose,
                executable=executable,
                cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
                transport=transport,
            )
        if scorer_artifact.panel.content_digest != panel_digest:
            raise SemanticObservationError(
                "blind scorer did not consume the extracted panel bytes"
            )

    bindings = _semantic_runtime_bindings(compiled, bundle, scorer_artifact)

    observed_atoms: list[SemanticAtomEvidence] = []
    for path in atom_paths(compiled.formula):
        atom = semantic_atom_at_path(compiled.formula, path)
        atom_bindings = {name: bindings[name] for name in atom.call.arguments}
        try:
            evidence = PYTHON_PREDICATE_BACKEND.evaluate(
                atom, compiled.registry, atom_bindings
            )
        except Exception as exc:  # Fail closed at the observation boundary.
            evidence = _evaluation_error(
                panel_digest=panel_digest,
                compiled=compiled,
                path=path,
                error=exc,
            )
        observed_atoms.append(SemanticAtomEvidence.from_evidence(path, evidence))

    replay_input = QueryReplayInput(
        "panel",
        panel_digest,
        tuple(
            AtomReplayInput(item.path, item.evidence) for item in observed_atoms
        ),
    )
    formula_evidence = TruthEvidenceRecord.from_evidence(
        PYTHON_PREDICATE_BACKEND.replay_query(compiled.formula, replay_input)
    )
    result = VisualSemanticObservationArtifact(
        panel_digest=panel_digest,
        panel_byte_count=len(panel_bytes),
        pre_observation_commitment_digest=precommit,
        proposal_digest=compiled.proposal.digest,
        policy_digest=compiled.policy.digest(),
        prospective_protocol_digest=protocol.digest(),
        scorer_family_digest=compiled.family.digest(),
        family_development_manifest_digest=(
            compiled.family.development_manifest_digest
        ),
        lowering_archive_digest=compiled.lowering_archive.digest,
        formula_data=compiled.formula.to_data(),
        compiled_formula_digest=formula_digest(compiled.formula),
        registry_digest=compiled.registry.digest(),
        attachment_digest=compiled.attachment_contract.digest(),
        witness_bundle=bundle,
        witness_summaries=summaries,
        scorer_artifact=scorer_artifact,
        atom_evidence=tuple(observed_atoms),
        formula_evidence=formula_evidence,
    )
    result.assert_matches(compiled, protocol)
    return result


def visual_semantic_support_gate_measurement(
    artifact: VisualSemanticObservationArtifact,
):
    if not isinstance(artifact, VisualSemanticObservationArtifact):
        raise TypeError("artifact must be VisualSemanticObservationArtifact")
    return artifact.to_support_gate_measurement()


# Short aliases for callers that already name this layer "semantic".
observe_semantic_panel = observe_visual_semantic_panel
SemanticObservationArtifact = VisualSemanticObservationArtifact


__all__ = [
    "SEMANTIC_ATOM_EVIDENCE_SCHEMA",
    "SEMANTIC_OBSERVATION_ALGORITHM_ID",
    "SEMANTIC_OBSERVATION_SCHEMA",
    "SemanticAtomEvidence",
    "SemanticObservationArtifact",
    "SemanticObservationError",
    "VisualSemanticObservationArtifact",
    "observe_semantic_panel",
    "observe_visual_semantic_panel",
    "replay_semantic_atom_evidence",
    "semantic_atom_at_path",
    "semantic_boundary_bindings",
    "visual_semantic_support_gate_measurement",
]

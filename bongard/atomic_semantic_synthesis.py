"""Python-native synthesis of positive atomic visual-semantic predicates.

The proposer supplies affirmative, one-cue atoms grounded in its frozen panel
descriptions.  A scorer supplies one observation for every atom/support-panel
pair.  An uncalibrated operational nonmatch has its own atomic record type and
projects to truth-lattice indeterminacy; it is never semantic certified
absence.  This module then performs an exact, support-only search for a small
positive conjunction under an explicit operational or calibrated scope.  It
has no operation that reverses an atom's polarity.

``one-cue`` receives a deliberately conservative lexical guard.  This is not
a semantic proof of atomicity: prose still needs a constrained
proposer/scorer protocol and calibration for semantic truth.

Every persistent value is canonical JSON with a content digest.  Consequently
the selected formula and its support evidence can be decoded and replayed
without a proposer, scorer, or execution backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import re
import unicodedata
from typing import Any, Mapping, Sequence

from bongard.artifacts import TruthEvidenceRecord, canonical_digest, canonical_json
from bongard.evidence import Disposition, Evidence, Provenance, Uncertainty
from bongard.typed_visual_proposal import (
    TypedSoftCue,
    TypedVisualProposalError,
)


PANEL_DESCRIPTION_BINDING_SCHEMA = "gkm.bongard-panel-description-binding.v2"
ATOMIC_EVIDENCE_BINDING_SCHEMA = "gkm.bongard-atomic-evidence-binding.v2"
ATOMIC_SOFT_PREDICATE_SCHEMA = "gkm.bongard-atomic-soft-predicate.v2"
ATOMIC_SUPPORT_CELL_SCHEMA = "gkm.bongard-atomic-support-cell.v2"
ATOMIC_SUPPORT_MATRIX_SCHEMA = "gkm.bongard-atomic-support-matrix.v2"
ATOMIC_SELECTION_ARCHIVE_SCHEMA = "gkm.bongard-atomic-selection-archive.v2"
NO_SEPARATOR_DIAGNOSTICS_SCHEMA = "gkm.bongard-no-exact-separator.v2"
MAX_ATOMIC_CONJUNCTION_SIZE = 4

OPERATIONAL_SELECTION_SCOPE = "operational-observer"
CALIBRATED_SELECTION_SCOPE = "calibrated-semantic"
_SELECTION_SCOPES = frozenset(
    {OPERATIONAL_SELECTION_SCOPE, CALIBRATED_SELECTION_SCOPE}
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PANEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_DISJUNCTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("or", re.compile(r"\bor\b")),
    ("either", re.compile(r"\beither\b")),
    (
        "one of",
        re.compile(r"\bone(?:\s*[-/\u2010\u2011\u2012\u2013\u2014\u2015]\s*|\s+)of\b"),
    ),
    ("alternative", re.compile(r"\balternat(?:ely|ively)\b")),
    ("slash alternative", re.compile(r"/")),
    ("otherwise", re.compile(r"\botherwise\b")),
    ("failing that", re.compile(r"\bfailing\s+that\b")),
    ("whichever", re.compile(r"\bwhichever\b")),
    ("any among", re.compile(r"\bany\s+among\b")),
    ("versus", re.compile(r"\bversus\b|\bvs\.?\b")),
)

_NEGATION_LAUNDERING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("avoid", re.compile(r"\bavoid(?:s|ed|ing)?\b")),
    ("fails to", re.compile(r"\bfail(?:s|ed|ing)?\s+to\b")),
    ("other than", re.compile(r"\bother\s+than\b")),
    ("instead of", re.compile(r"\binstead\s+of\b")),
    ("hyphen-free", re.compile(r"\b[a-z][a-z0-9-]*-free\b")),
)

_BUNDLING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("and", re.compile(r"\band\b")),
    ("also", re.compile(r"\balso\b")),
    ("both", re.compile(r"\bboth\b")),
    ("as well as", re.compile(r"\bas\s+well\s+as\b")),
    ("additionally", re.compile(r"\badditionally\b")),
    ("together with", re.compile(r"\btogether\s+with\b")),
    ("respectively", re.compile(r"\brespectively\b")),
    ("semicolon", re.compile(r";")),
)

ATOMIC_AFFIRMATIVE_SURFACE_POLICY_SCHEMA = (
    "gkm.bongard-atomic-affirmative-surface-policy.v1"
)
_ATOMIC_AFFIRMATIVE_PATTERN_FAMILIES: tuple[
    tuple[str, tuple[tuple[str, re.Pattern[str]], ...]], ...
] = (
    ("disjunction", _DISJUNCTION_PATTERNS),
    ("negation-laundering", _NEGATION_LAUNDERING_PATTERNS),
    ("bundling", _BUNDLING_PATTERNS),
)


def atomic_affirmative_surface_policy_data() -> dict[str, object]:
    """Return the exact closed regex policy used by ``_atomic_affirmative``."""

    return {
        "schema": ATOMIC_AFFIRMATIVE_SURFACE_POLICY_SCHEMA,
        "matching_normalization": "NFKC-then-casefold",
        "closed_families": [
            {
                "family": family,
                "patterns": [
                    {
                        "name": name,
                        "regex": pattern.pattern,
                        "flags": pattern.flags,
                    }
                    for name, pattern in patterns
                ],
            }
            for family, patterns in _ATOMIC_AFFIRMATIVE_PATTERN_FAMILIES
        ],
    }


def atomic_affirmative_surface_policy_description() -> str:
    """Describe every closed forbidden family for model-visible contracts."""

    families = "; ".join(
        family + " [" + ", ".join(name for name, _ in patterns) + "]"
        for family, patterns in _ATOMIC_AFFIRMATIVE_PATTERN_FAMILIES
    )
    return (
        "After NFKC casefold matching, the exact closed forbidden atomicity "
        "surface families are "
        + families
        + "."
    )

_DESCRIPTION_ROLE_LEAK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "support label",
        re.compile(
            r"\b(?:positive|negative)(?:[ -]+support)?[ -]+"
            r"(?:panel|example|image|side|set|class)s?\b"
        ),
    ),
    ("hidden label", re.compile(r"\bhidden[ -]+label\b")),
    ("label assertion", re.compile(r"\blabel\s+(?:is|equals|:)\b")),
    ("query role", re.compile(r"\b(?:released[ -]+)?query(?:[ -]+panel)?\b")),
)


class AtomicSemanticSynthesisError(ValueError):
    """An atomic predicate, evidence matrix, or archive is invalid."""


def _same_canonical_json(left: object, right: object) -> bool:
    """Compare decoded and in-memory projections at the JSON boundary."""

    return canonical_json(left) == canonical_json(right)


def _mapping(
    value: object, fields: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise AtomicSemanticSynthesisError(
            f"{label} fields differ from the static schema"
        )
    if any(not isinstance(key, str) for key in value):
        raise AtomicSemanticSynthesisError(f"{label} keys must be strings")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise AtomicSemanticSynthesisError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _panel_id(value: object, label: str = "panel_id") -> str:
    if not isinstance(value, str) or _PANEL_ID.fullmatch(value) is None:
        raise AtomicSemanticSynthesisError(f"invalid {label} {value!r}")
    return value


def _selection_scope(value: object, label: str = "selection_scope") -> str:
    if not isinstance(value, str) or value not in _SELECTION_SCOPES:
        raise AtomicSemanticSynthesisError(
            f"{label} must be one of {sorted(_SELECTION_SCOPES)}"
        )
    return value


def _literal_ordinal(value: object, label: str) -> int:
    if type(value) is not int or not 1 <= value <= 1_000_000:
        raise AtomicSemanticSynthesisError(
            f"{label} must be a literal integer in 1..1000000"
        )
    return value


def _identity_text(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or unicodedata.normalize("NFKC", value) != value
        or len(value.encode("utf-8")) > 128
        or any(unicodedata.category(character) in {"Cc", "Cf"} for character in value)
    ):
        raise AtomicSemanticSynthesisError(
            f"{label} must be bounded canonical exact text"
        )
    return value


def _exact_description(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise AtomicSemanticSynthesisError(
            f"{label} must be non-empty exact text without outer whitespace"
        )
    if unicodedata.normalize("NFKC", value) != value:
        raise AtomicSemanticSynthesisError(f"{label} must use canonical NFKC text")
    if "\x00" in value or any(
        unicodedata.category(character) in {"Cc", "Cf"} for character in value
    ):
        raise AtomicSemanticSynthesisError(
            f"{label} contains a forbidden control character"
        )
    if len(value.encode("utf-8")) > 384:
        raise AtomicSemanticSynthesisError(f"{label} exceeds 384 UTF-8 bytes")
    normalised = value.casefold()
    for name, pattern in _DESCRIPTION_ROLE_LEAK_PATTERNS:
        if pattern.search(normalised) is not None:
            raise AtomicSemanticSynthesisError(
                f"{label} contains forbidden run-role leak {name!r}"
            )
    return value


def _bounded_reason(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or unicodedata.normalize("NFKC", value) != value
        or len(value.encode("utf-8")) > 2048
        or "\x00" in value
        or any(
            unicodedata.category(character) in {"Cc", "Cf"}
            for character in value
        )
    ):
        raise AtomicSemanticSynthesisError(
            f"{label} must be bounded canonical exact text"
        )
    return value


def _atomic_affirmative(value: object, label: str) -> str:
    """Apply a conservative surface guard to one observer phrase.

    This is deliberately *not* a semantic proof of atomicity.  It removes
    obvious polarity, alternative, and bundled constructions so that the
    remaining phrase can define one opaque operational observer question.
    Scientific meaning still requires an independently pinned calibration.
    """

    if not isinstance(value, str):
        raise AtomicSemanticSynthesisError(f"{label} must be a string")
    normalised = unicodedata.normalize("NFKC", value).casefold()
    for family, patterns in _ATOMIC_AFFIRMATIVE_PATTERN_FAMILIES:
        for name, pattern in patterns:
            if pattern.search(normalised) is not None:
                raise AtomicSemanticSynthesisError(
                    f"{label} contains non-atomic {family} surface token "
                    f"{name!r}"
                )
    try:
        TypedSoftCue("cue-00", value)
    except TypedVisualProposalError as exc:
        raise AtomicSemanticSynthesisError(f"invalid {label}: {exc}") from exc
    return value


def validate_atomic_affirmative_surface(
    value: object, label: str = "observer phrase"
) -> str:
    """Apply the public, exact closed atomicity surface validator."""

    return _atomic_affirmative(value, label)


@dataclass(frozen=True, slots=True)
class PanelDescriptionBinding:
    """Attested panel bytes and neutral vision prose from one scheduled call."""

    panel_id: str
    panel_digest: str
    description: str
    phase: str
    description_protocol_digest: str
    validated_receipt_digest: str
    run_commitment_digest: str
    call_ordinal: int
    description_digest: str

    def __post_init__(self) -> None:
        _panel_id(self.panel_id)
        _digest(self.panel_digest, "panel_digest")
        _exact_description(self.description, "panel description")
        if self.phase not in {"support", "query"}:
            raise AtomicSemanticSynthesisError(
                "panel description phase must be support or query"
            )
        _digest(
            self.description_protocol_digest,
            "description_protocol_digest",
        )
        _digest(self.validated_receipt_digest, "validated_receipt_digest")
        _digest(self.run_commitment_digest, "run_commitment_digest")
        _literal_ordinal(self.call_ordinal, "description call_ordinal")
        _digest(self.description_digest, "description_digest")
        expected = canonical_digest(self.content_data())
        if self.description_digest != expected:
            raise AtomicSemanticSynthesisError(
                "description_digest differs from its exact panel/prose binding"
            )

    @classmethod
    def create(
        cls,
        panel_id: str,
        panel_digest: str,
        description: str,
        *,
        phase: str,
        description_protocol_digest: str,
        validated_receipt_digest: str,
        run_commitment_digest: str,
        call_ordinal: int,
    ) -> "PanelDescriptionBinding":
        content = {
            "schema": PANEL_DESCRIPTION_BINDING_SCHEMA,
            "panel_id": panel_id,
            "panel_digest": panel_digest,
            "description": description,
            "phase": phase,
            "description_protocol_digest": description_protocol_digest,
            "validated_receipt_digest": validated_receipt_digest,
            "run_commitment_digest": run_commitment_digest,
            "call_ordinal": call_ordinal,
        }
        return cls(**{  # type: ignore[arg-type]
            key: value for key, value in content.items() if key != "schema"
        }, description_digest=canonical_digest(content))

    def content_data(self) -> dict[str, object]:
        return {
            "schema": PANEL_DESCRIPTION_BINDING_SCHEMA,
            "panel_id": self.panel_id,
            "panel_digest": self.panel_digest,
            "description": self.description,
            "phase": self.phase,
            "description_protocol_digest": self.description_protocol_digest,
            "validated_receipt_digest": self.validated_receipt_digest,
            "run_commitment_digest": self.run_commitment_digest,
            "call_ordinal": self.call_ordinal,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "description_digest": self.description_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PanelDescriptionBinding":
        data = _mapping(
            value,
            frozenset(
                {
                    "schema",
                    "panel_id",
                    "panel_digest",
                    "description",
                    "phase",
                    "description_protocol_digest",
                    "validated_receipt_digest",
                    "run_commitment_digest",
                    "call_ordinal",
                    "description_digest",
                }
            ),
            "panel description binding",
        )
        if data["schema"] != PANEL_DESCRIPTION_BINDING_SCHEMA:
            raise AtomicSemanticSynthesisError(
                "unsupported panel description binding schema"
            )
        result = cls(
            panel_id=data["panel_id"],
            panel_digest=data["panel_digest"],
            description=data["description"],
            phase=data["phase"],
            description_protocol_digest=data["description_protocol_digest"],
            validated_receipt_digest=data["validated_receipt_digest"],
            run_commitment_digest=data["run_commitment_digest"],
            call_ordinal=data["call_ordinal"],
            description_digest=data["description_digest"],
        )
        if not _same_canonical_json(result.to_data(), value):
            raise AtomicSemanticSynthesisError(
                "panel description binding is not canonical"
            )
        return result


def _panel_descriptions_digest(
    bindings: tuple[PanelDescriptionBinding, ...],
) -> str:
    return canonical_digest([binding.to_data() for binding in bindings])


@dataclass(frozen=True, slots=True)
class AtomicSoftPredicate:
    """One affirmative claim with exactly one independently scored cue."""

    source_proposal_digest: str
    scorer_protocol_digest: str
    positive_description: str
    cue_description: str
    panel_descriptions: tuple[PanelDescriptionBinding, ...]
    panel_descriptions_digest: str
    atom_digest: str

    def __post_init__(self) -> None:
        _digest(self.source_proposal_digest, "source_proposal_digest")
        _digest(self.scorer_protocol_digest, "scorer_protocol_digest")
        _atomic_affirmative(self.positive_description, "positive_description")
        _atomic_affirmative(self.cue_description, "cue_description")
        if self.positive_description != self.cue_description:
            raise AtomicSemanticSynthesisError(
                "positive_description and cue_description must be the same "
                "single canonical observer phrase"
            )
        if (
            not isinstance(self.panel_descriptions, tuple)
            or not self.panel_descriptions
        ):
            raise AtomicSemanticSynthesisError(
                "panel_descriptions must be a non-empty immutable tuple"
            )
        if not all(
            isinstance(item, PanelDescriptionBinding)
            for item in self.panel_descriptions
        ):
            raise TypeError(
                "panel_descriptions must contain PanelDescriptionBinding values"
            )
        panel_ids = tuple(item.panel_id for item in self.panel_descriptions)
        if panel_ids != tuple(sorted(panel_ids)) or len(panel_ids) != len(
            set(panel_ids)
        ):
            raise AtomicSemanticSynthesisError(
                "panel description bindings must have unique sorted panel IDs"
            )
        if any(item.phase != "support" for item in self.panel_descriptions):
            raise AtomicSemanticSynthesisError(
                "atomic predicates may bind support descriptions only"
            )
        description_contexts = {
            (
                item.description_protocol_digest,
                item.run_commitment_digest,
            )
            for item in self.panel_descriptions
        }
        if len(description_contexts) != 1:
            raise AtomicSemanticSynthesisError(
                "atomic panel descriptions must share one protocol and run commitment"
            )
        receipt_digests = tuple(
            item.validated_receipt_digest for item in self.panel_descriptions
        )
        call_ordinals = tuple(item.call_ordinal for item in self.panel_descriptions)
        if len(receipt_digests) != len(set(receipt_digests)) or len(
            call_ordinals
        ) != len(set(call_ordinals)):
            raise AtomicSemanticSynthesisError(
                "support descriptions must come from distinct receipts and "
                "call ordinals"
            )
        _digest(self.panel_descriptions_digest, "panel_descriptions_digest")
        if self.panel_descriptions_digest != _panel_descriptions_digest(
            self.panel_descriptions
        ):
            raise AtomicSemanticSynthesisError(
                "panel_descriptions_digest differs from its bindings"
            )
        _digest(self.atom_digest, "atom_digest")
        if self.atom_digest != canonical_digest(self.content_data()):
            raise AtomicSemanticSynthesisError(
                "atom_digest differs from the atomic predicate preimage"
            )

    @classmethod
    def create(
        cls,
        *,
        source_proposal_digest: str,
        scorer_protocol_digest: str,
        positive_description: str,
        cue_description: str,
        panel_descriptions: Sequence[PanelDescriptionBinding],
    ) -> "AtomicSoftPredicate":
        bindings = tuple(sorted(panel_descriptions, key=lambda item: item.panel_id))
        content = {
            "schema": ATOMIC_SOFT_PREDICATE_SCHEMA,
            "source_proposal_digest": source_proposal_digest,
            "scorer_protocol_digest": scorer_protocol_digest,
            "positive_description": positive_description,
            "cue_description": cue_description,
            "panel_descriptions": [binding.to_data() for binding in bindings],
            "panel_descriptions_digest": _panel_descriptions_digest(bindings),
        }
        return cls(
            source_proposal_digest=source_proposal_digest,
            scorer_protocol_digest=scorer_protocol_digest,
            positive_description=positive_description,
            cue_description=cue_description,
            panel_descriptions=bindings,
            panel_descriptions_digest=content["panel_descriptions_digest"],
            atom_digest=canonical_digest(content),
        )

    @property
    def atom_id(self) -> str:
        """The full canonical atom digest, used directly in frozen formulae."""

        return self.atom_digest

    @property
    def description_utf8_bytes(self) -> int:
        """Frozen MDL proxy used after minimum conjunction cardinality."""

        return len(self.positive_description.encode("utf-8"))

    def panel_binding(self, panel_id: str) -> PanelDescriptionBinding:
        for binding in self.panel_descriptions:
            if binding.panel_id == panel_id:
                return binding
        raise AtomicSemanticSynthesisError(
            f"atom {self.atom_id} has no panel binding {panel_id!r}"
        )

    def content_data(self) -> dict[str, object]:
        return {
            "schema": ATOMIC_SOFT_PREDICATE_SCHEMA,
            "source_proposal_digest": self.source_proposal_digest,
            "scorer_protocol_digest": self.scorer_protocol_digest,
            "positive_description": self.positive_description,
            "cue_description": self.cue_description,
            "panel_descriptions": [
                binding.to_data() for binding in self.panel_descriptions
            ],
            "panel_descriptions_digest": self.panel_descriptions_digest,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "atom_digest": self.atom_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "AtomicSoftPredicate":
        data = _mapping(
            value,
            frozenset(
                {
                    "schema",
                    "source_proposal_digest",
                    "scorer_protocol_digest",
                    "positive_description",
                    "cue_description",
                    "panel_descriptions",
                    "panel_descriptions_digest",
                    "atom_digest",
                }
            ),
            "atomic soft predicate",
        )
        if data["schema"] != ATOMIC_SOFT_PREDICATE_SCHEMA:
            raise AtomicSemanticSynthesisError(
                "unsupported atomic soft predicate schema"
            )
        raw_bindings = data["panel_descriptions"]
        if not isinstance(raw_bindings, list):
            raise AtomicSemanticSynthesisError(
                "atomic panel_descriptions must be a list"
            )
        result = cls(
            source_proposal_digest=data["source_proposal_digest"],
            scorer_protocol_digest=data["scorer_protocol_digest"],
            positive_description=data["positive_description"],
            cue_description=data["cue_description"],
            panel_descriptions=tuple(
                PanelDescriptionBinding.from_data(item) for item in raw_bindings
            ),
            panel_descriptions_digest=data["panel_descriptions_digest"],
            atom_digest=data["atom_digest"],
        )
        if not _same_canonical_json(result.to_data(), value):
            raise AtomicSemanticSynthesisError(
                "atomic soft predicate is not canonically represented"
            )
        return result


@dataclass(frozen=True, slots=True)
class AtomicEvidenceBinding:
    """Frozen context that one scorer observation must name exactly.

    Support cells derive this context from their atom/panel bindings.  Query
    callers must supply it explicitly because query pixels and their vision
    descriptions are created only after the support formula has been frozen.
    """

    atom_id: str
    panel_digest: str
    panel_description_digest: str
    scorer_protocol_digest: str
    run_commitment_digest: str
    scorer_producer: str
    scorer_version: str
    scorer_method: str
    scorer_run_id: str
    scorer_receipt_digest: str
    scorer_output_digest: str
    scorer_call_digest: str
    scorer_call_ordinal: int
    observation_scope: str
    calibration_digest: str | None = None

    def __post_init__(self) -> None:
        _digest(self.atom_id, "evidence binding atom_id")
        _digest(self.panel_digest, "evidence binding panel_digest")
        _digest(
            self.panel_description_digest,
            "evidence binding panel_description_digest",
        )
        _digest(
            self.scorer_protocol_digest,
            "evidence binding scorer_protocol_digest",
        )
        _digest(
            self.run_commitment_digest,
            "evidence binding run_commitment_digest",
        )
        _identity_text(self.scorer_producer, "authorized scorer producer")
        _identity_text(self.scorer_version, "authorized scorer version")
        _identity_text(self.scorer_method, "authorized scorer method")
        _panel_id(self.scorer_run_id, "authorized scorer run_id")
        _digest(self.scorer_receipt_digest, "authorized scorer receipt digest")
        _digest(self.scorer_output_digest, "authorized scorer output digest")
        _digest(self.scorer_call_digest, "authorized scorer call digest")
        _literal_ordinal(self.scorer_call_ordinal, "scorer call_ordinal")
        _selection_scope(self.observation_scope, "observation_scope")
        if self.observation_scope == OPERATIONAL_SELECTION_SCOPE:
            if self.calibration_digest is not None:
                raise AtomicSemanticSynthesisError(
                    "operational observations cannot claim a calibration digest"
                )
        elif self.calibration_digest is None:
            raise AtomicSemanticSynthesisError(
                "calibrated semantic observations require an externally pinned "
                "calibration digest"
            )
        if self.calibration_digest is not None:
            _digest(self.calibration_digest, "calibration_digest")

    @property
    def call_ordinal_digest(self) -> str:
        return canonical_digest(
            {"scorer_call_ordinal": self.scorer_call_ordinal}
        )

    @property
    def calibration_binding_digest(self) -> str:
        return canonical_digest(
            {"calibration_digest": self.calibration_digest}
        )

    @property
    def input_digests(self) -> tuple[str, ...]:
        """The only accepted canonical order in scorer provenance."""

        return (
            self.atom_id,
            self.panel_digest,
            self.panel_description_digest,
            self.scorer_protocol_digest,
            self.run_commitment_digest,
            self.scorer_receipt_digest,
            self.scorer_output_digest,
            self.scorer_call_digest,
            self.call_ordinal_digest,
            self.calibration_binding_digest,
        )

    @property
    def provenance_details(self) -> tuple[tuple[str, str], ...]:
        return (
            (
                "calibration_digest",
                self.calibration_digest
                if self.calibration_digest is not None
                else "none",
            ),
            ("call_digest", self.scorer_call_digest),
            ("call_ordinal", str(self.scorer_call_ordinal)),
            ("output_digest", self.scorer_output_digest),
            ("semantic_scope", self.observation_scope),
        )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": ATOMIC_EVIDENCE_BINDING_SCHEMA,
            "atom_id": self.atom_id,
            "panel_digest": self.panel_digest,
            "panel_description_digest": self.panel_description_digest,
            "scorer_protocol_digest": self.scorer_protocol_digest,
            "run_commitment_digest": self.run_commitment_digest,
            "scorer_producer": self.scorer_producer,
            "scorer_version": self.scorer_version,
            "scorer_method": self.scorer_method,
            "scorer_run_id": self.scorer_run_id,
            "scorer_receipt_digest": self.scorer_receipt_digest,
            "scorer_output_digest": self.scorer_output_digest,
            "scorer_call_digest": self.scorer_call_digest,
            "scorer_call_ordinal": self.scorer_call_ordinal,
            "observation_scope": self.observation_scope,
            "calibration_digest": self.calibration_digest,
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "AtomicEvidenceBinding":
        fields = frozenset(
            {
                "schema",
                "atom_id",
                "panel_digest",
                "panel_description_digest",
                "scorer_protocol_digest",
                "run_commitment_digest",
                "scorer_producer",
                "scorer_version",
                "scorer_method",
                "scorer_run_id",
                "scorer_receipt_digest",
                "scorer_output_digest",
                "scorer_call_digest",
                "scorer_call_ordinal",
                "observation_scope",
                "calibration_digest",
            }
        )
        data = _mapping(value, fields, "atomic evidence binding")
        if data["schema"] != ATOMIC_EVIDENCE_BINDING_SCHEMA:
            raise AtomicSemanticSynthesisError(
                "unsupported atomic evidence binding schema"
            )
        result = cls(  # type: ignore[arg-type]
            **{key: data[key] for key in fields if key != "schema"}
        )
        if not _same_canonical_json(result.to_data(), value):
            raise AtomicSemanticSynthesisError(
                "atomic evidence binding is not canonically represented"
            )
        return result


def _validate_evidence_provenance_binding(
    provenance: object,
    binding: AtomicEvidenceBinding,
    label: str,
) -> None:
    if not isinstance(provenance, Provenance):
        raise TypeError(f"{label} provenance must be Provenance")
    if provenance.input_digests != binding.input_digests:
        raise AtomicSemanticSynthesisError(
            f"{label} provenance input_digests must exactly bind "
            "the atom, panel, description, scorer protocol, run commitment, "
            "validated receipt, output, call record, call ordinal, and "
            "calibration state in canonical order"
        )
    if (
        provenance.producer != binding.scorer_producer
        or provenance.version != binding.scorer_version
        or provenance.method != binding.scorer_method
        or provenance.run_id != binding.scorer_run_id
        or provenance.artifact_digest != binding.scorer_receipt_digest
        or provenance.details != binding.provenance_details
    ):
        raise AtomicSemanticSynthesisError(
            f"{label} provenance identity differs from its exact authorized "
            "producer, run, receipt, output, scope, or call ordinal"
        )


@dataclass(frozen=True, slots=True)
class OperationalNonmatchRecord:
    """Atomic-only persisted outcome for an uncalibrated observer nonmatch.

    It serializes with its own disposition and can cover negatives only in an
    explicitly operational selection archive.  Conversion to the general
    evidence lattice is an ``INDETERMINATE`` abstention, never certified
    semantic absence.
    """

    provenance: Provenance
    reason: str
    uncertainty: Uncertainty | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, Provenance):
            raise TypeError("operational nonmatch provenance must be Provenance")
        _bounded_reason(self.reason, "operational nonmatch reason")
        if self.uncertainty is not None and not isinstance(
            self.uncertainty, Uncertainty
        ):
            raise TypeError("operational nonmatch uncertainty must be Uncertainty")

    @property
    def disposition(self) -> Disposition:
        """Its conservative truth-lattice projection."""

        return Disposition.INDETERMINATE

    @classmethod
    def from_evidence(
        cls, evidence: Evidence[bool]
    ) -> "OperationalNonmatchRecord":
        if not isinstance(evidence, Evidence) or not evidence.is_operational_nonmatch:
            raise AtomicSemanticSynthesisError(
                "operational nonmatch record requires explicit operational evidence"
            )
        assert evidence.reason is not None
        prefix = "operational nonmatch (uncalibrated): "
        return cls(
            provenance=evidence.provenance,
            reason=evidence.reason[len(prefix) :],
            uncertainty=evidence.uncertainty,
        )

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "OperationalNonmatchRecord":
        raw = dict(value)
        if raw.get("disposition") != "operational_nonmatch":
            raise AtomicSemanticSynthesisError(
                "operational nonmatch record has the wrong disposition"
            )
        raw["disposition"] = Disposition.INDETERMINATE.value
        record = TruthEvidenceRecord.from_data(raw)
        if record.reason is None:
            raise AtomicSemanticSynthesisError(
                "operational nonmatch record requires a reason"
            )
        result = cls(record.provenance, record.reason, record.uncertainty)
        if not _same_canonical_json(result.to_data(), value):
            raise AtomicSemanticSynthesisError(
                "operational nonmatch record is not canonical"
            )
        return result

    def to_evidence(self) -> Evidence[bool]:
        return Evidence.operational_nonmatch(
            self.provenance, self.reason, self.uncertainty
        )

    def to_data(self) -> dict[str, object]:
        data = TruthEvidenceRecord.from_evidence(
            Evidence.indeterminate(
                self.provenance, self.reason, self.uncertainty
            )
        ).to_data()
        data["disposition"] = "operational_nonmatch"
        return data

    def digest(self) -> str:
        return canonical_digest(self.to_data())


@dataclass(frozen=True, slots=True)
class AtomicSupportCell:
    """One cold-verifiable atom/panel evidence preimage and its commitments."""

    atom_id: str
    panel_id: str
    panel_digest: str
    panel_description_digest: str
    scorer_protocol_digest: str
    evidence_binding: AtomicEvidenceBinding
    evidence: TruthEvidenceRecord | OperationalNonmatchRecord
    evidence_digest: str
    cell_digest: str

    def __post_init__(self) -> None:
        _digest(self.atom_id, "cell atom_id")
        _panel_id(self.panel_id)
        _digest(self.panel_digest, "cell panel_digest")
        _digest(self.panel_description_digest, "cell panel_description_digest")
        _digest(self.scorer_protocol_digest, "cell scorer_protocol_digest")
        if not isinstance(self.evidence_binding, AtomicEvidenceBinding):
            raise TypeError("cell evidence_binding must be AtomicEvidenceBinding")
        if not isinstance(
            self.evidence, (TruthEvidenceRecord, OperationalNonmatchRecord)
        ):
            raise TypeError("cell evidence must be an atomic evidence record")
        if (
            self.evidence_binding.atom_id != self.atom_id
            or self.evidence_binding.panel_digest != self.panel_digest
            or self.evidence_binding.panel_description_digest
            != self.panel_description_digest
            or self.evidence_binding.scorer_protocol_digest
            != self.scorer_protocol_digest
        ):
            raise AtomicSemanticSynthesisError(
                "cell evidence authorization differs from its atom/panel context"
            )
        _validate_evidence_provenance_binding(
            self.evidence.provenance,
            self.evidence_binding,
            "cell evidence",
        )
        if isinstance(self.evidence, OperationalNonmatchRecord):
            if self.evidence_binding.observation_scope != OPERATIONAL_SELECTION_SCOPE:
                raise AtomicSemanticSynthesisError(
                    "operational nonmatch requires operational observer authorization"
                )
        elif (
            self.evidence.disposition is Disposition.CERTIFIED_ABSENT
        ):
            raise AtomicSemanticSynthesisError(
                "calibrated semantic absence is disabled until the core can "
                "validate a typed calibration artifact and interval rule"
            )
        if (
            self.evidence.disposition is Disposition.PRESENT
            and self.evidence.to_evidence().unwrap() is not True
        ):
            raise AtomicSemanticSynthesisError("present cell evidence must mean True")
        _digest(self.evidence_digest, "cell evidence_digest")
        if self.evidence_digest != self.evidence.digest():
            raise AtomicSemanticSynthesisError(
                "cell evidence_digest differs from the evidence preimage"
            )
        _digest(self.cell_digest, "cell_digest")
        if self.cell_digest != canonical_digest(self.content_data()):
            raise AtomicSemanticSynthesisError(
                "cell_digest differs from its evidence/binding preimage"
            )

    @classmethod
    def capture(
        cls,
        atom: AtomicSoftPredicate,
        panel_id: str,
        evidence: Evidence[bool],
        *,
        evidence_binding: AtomicEvidenceBinding,
    ) -> "AtomicSupportCell":
        if not isinstance(atom, AtomicSoftPredicate):
            raise TypeError("atom must be an AtomicSoftPredicate")
        if not isinstance(evidence, Evidence):
            raise TypeError("cell evidence must be Evidence[bool]")
        binding = atom.panel_binding(panel_id)
        if not isinstance(evidence_binding, AtomicEvidenceBinding):
            raise TypeError("evidence_binding must be AtomicEvidenceBinding")
        expected_context = (
            atom.atom_id,
            binding.panel_digest,
            binding.description_digest,
            atom.scorer_protocol_digest,
            binding.run_commitment_digest,
        )
        actual_context = (
            evidence_binding.atom_id,
            evidence_binding.panel_digest,
            evidence_binding.panel_description_digest,
            evidence_binding.scorer_protocol_digest,
            evidence_binding.run_commitment_digest,
        )
        if actual_context != expected_context:
            raise AtomicSemanticSynthesisError(
                "evidence authorization differs from atom/panel/run context"
            )
        _validate_evidence_provenance_binding(
            evidence.provenance,
            evidence_binding,
            "captured evidence",
        )
        record: TruthEvidenceRecord | OperationalNonmatchRecord
        if evidence.is_operational_nonmatch:
            record = OperationalNonmatchRecord.from_evidence(evidence)
        else:
            record = TruthEvidenceRecord.from_evidence(evidence)
        if (
            isinstance(record, OperationalNonmatchRecord)
            and evidence_binding.observation_scope != OPERATIONAL_SELECTION_SCOPE
        ):
            raise AtomicSemanticSynthesisError(
                "operational nonmatch requires operational observer authorization"
            )
        if (
            isinstance(record, TruthEvidenceRecord)
            and record.disposition is Disposition.CERTIFIED_ABSENT
        ):
            raise AtomicSemanticSynthesisError(
                "calibrated semantic absence is disabled until the core can "
                "validate a typed calibration artifact and interval rule"
            )
        content = {
            "schema": ATOMIC_SUPPORT_CELL_SCHEMA,
            "atom_id": atom.atom_id,
            "panel_id": panel_id,
            "panel_digest": binding.panel_digest,
            "panel_description_digest": binding.description_digest,
            "scorer_protocol_digest": atom.scorer_protocol_digest,
            "evidence_binding": evidence_binding.to_data(),
            "evidence": record.to_data(),
            "evidence_digest": record.digest(),
        }
        return cls(
            atom_id=atom.atom_id,
            panel_id=panel_id,
            panel_digest=binding.panel_digest,
            panel_description_digest=binding.description_digest,
            scorer_protocol_digest=atom.scorer_protocol_digest,
            evidence_binding=evidence_binding,
            evidence=record,
            evidence_digest=record.digest(),
            cell_digest=canonical_digest(content),
        )

    def content_data(self) -> dict[str, object]:
        return {
            "schema": ATOMIC_SUPPORT_CELL_SCHEMA,
            "atom_id": self.atom_id,
            "panel_id": self.panel_id,
            "panel_digest": self.panel_digest,
            "panel_description_digest": self.panel_description_digest,
            "scorer_protocol_digest": self.scorer_protocol_digest,
            "evidence_binding": self.evidence_binding.to_data(),
            "evidence": self.evidence.to_data(),
            "evidence_digest": self.evidence_digest,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "cell_digest": self.cell_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "AtomicSupportCell":
        data = _mapping(
            value,
            frozenset(
                {
                    "schema",
                    "atom_id",
                    "panel_id",
                    "panel_digest",
                    "panel_description_digest",
                    "scorer_protocol_digest",
                    "evidence_binding",
                    "evidence",
                    "evidence_digest",
                    "cell_digest",
                }
            ),
            "atomic support cell",
        )
        if data["schema"] != ATOMIC_SUPPORT_CELL_SCHEMA:
            raise AtomicSemanticSynthesisError("unsupported support cell schema")
        if not isinstance(data["evidence"], Mapping) or not isinstance(
            data["evidence_binding"], Mapping
        ):
            raise AtomicSemanticSynthesisError(
                "cell evidence and authorization must be objects"
            )
        raw_evidence = data["evidence"]
        record: TruthEvidenceRecord | OperationalNonmatchRecord
        if raw_evidence.get("disposition") == "operational_nonmatch":
            record = OperationalNonmatchRecord.from_data(raw_evidence)
        else:
            record = TruthEvidenceRecord.from_data(raw_evidence)
        result = cls(
            atom_id=data["atom_id"],
            panel_id=data["panel_id"],
            panel_digest=data["panel_digest"],
            panel_description_digest=data["panel_description_digest"],
            scorer_protocol_digest=data["scorer_protocol_digest"],
            evidence_binding=AtomicEvidenceBinding.from_data(
                data["evidence_binding"]
            ),
            evidence=record,
            evidence_digest=data["evidence_digest"],
            cell_digest=data["cell_digest"],
        )
        if not _same_canonical_json(result.to_data(), value):
            raise AtomicSemanticSynthesisError(
                "atomic support cell is not canonically represented"
            )
        return result


@dataclass(frozen=True, slots=True)
class AtomicSupportMatrix:
    """The exact atom by support-panel Cartesian evidence matrix."""

    atoms: tuple[AtomicSoftPredicate, ...]
    panel_ids: tuple[str, ...]
    cells: tuple[AtomicSupportCell, ...]
    source_proposal_digest: str
    scorer_protocol_digest: str
    panel_descriptions_digest: str
    matrix_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.atoms, tuple) or not self.atoms:
            raise AtomicSemanticSynthesisError(
                "support matrix atoms must be a non-empty tuple"
            )
        if not all(isinstance(atom, AtomicSoftPredicate) for atom in self.atoms):
            raise TypeError("matrix atoms must be AtomicSoftPredicate values")
        atom_ids = tuple(atom.atom_id for atom in self.atoms)
        if atom_ids != tuple(sorted(atom_ids)) or len(atom_ids) != len(set(atom_ids)):
            raise AtomicSemanticSynthesisError(
                "matrix atoms must have unique sorted canonical digests"
            )
        if not isinstance(self.panel_ids, tuple) or not self.panel_ids:
            raise AtomicSemanticSynthesisError(
                "support matrix panel_ids must be a non-empty tuple"
            )
        for value in self.panel_ids:
            _panel_id(value)
        if self.panel_ids != tuple(sorted(self.panel_ids)) or len(
            self.panel_ids
        ) != len(set(self.panel_ids)):
            raise AtomicSemanticSynthesisError(
                "support matrix panel IDs must be unique and sorted"
            )
        _digest(self.source_proposal_digest, "matrix source_proposal_digest")
        _digest(self.scorer_protocol_digest, "matrix scorer_protocol_digest")
        _digest(self.panel_descriptions_digest, "matrix panel_descriptions_digest")
        for atom in self.atoms:
            if atom.source_proposal_digest != self.source_proposal_digest:
                raise AtomicSemanticSynthesisError(
                    "matrix atoms bind different source proposals"
                )
            if atom.scorer_protocol_digest != self.scorer_protocol_digest:
                raise AtomicSemanticSynthesisError(
                    "matrix atoms bind different scorer protocols"
                )
            if atom.panel_descriptions_digest != self.panel_descriptions_digest:
                raise AtomicSemanticSynthesisError(
                    "matrix atoms bind different panel descriptions"
                )
            if (
                tuple(item.panel_id for item in atom.panel_descriptions)
                != self.panel_ids
            ):
                raise AtomicSemanticSynthesisError(
                    "matrix atom panel bindings differ from panel_ids"
                )
        if not isinstance(self.cells, tuple) or not all(
            isinstance(cell, AtomicSupportCell) for cell in self.cells
        ):
            raise TypeError("matrix cells must be AtomicSupportCell values")
        expected = tuple(
            (atom.atom_id, panel_id)
            for atom in self.atoms
            for panel_id in self.panel_ids
        )
        actual = tuple((cell.atom_id, cell.panel_id) for cell in self.cells)
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            raise AtomicSemanticSynthesisError(
                "matrix cells do not equal the atom x panel Cartesian product: "
                f"missing={missing}, extra={extra}"
            )
        atoms_by_id = {atom.atom_id: atom for atom in self.atoms}
        for cell in self.cells:
            atom = atoms_by_id[cell.atom_id]
            binding = atom.panel_binding(cell.panel_id)
            if (
                cell.panel_digest != binding.panel_digest
                or cell.panel_description_digest != binding.description_digest
                or cell.scorer_protocol_digest != atom.scorer_protocol_digest
            ):
                raise AtomicSemanticSynthesisError(
                    "matrix cell binding differs from its atom/panel context"
                )
            if (
                cell.evidence_binding.run_commitment_digest
                != binding.run_commitment_digest
            ):
                raise AtomicSemanticSynthesisError(
                    "matrix scorer authorization differs from description run "
                    "commitment"
                )
        authorization_contexts = {
            (
                cell.evidence_binding.scorer_producer,
                cell.evidence_binding.scorer_version,
                cell.evidence_binding.scorer_method,
                cell.evidence_binding.scorer_run_id,
                cell.evidence_binding.run_commitment_digest,
                cell.evidence_binding.observation_scope,
                cell.evidence_binding.calibration_digest,
            )
            for cell in self.cells
        }
        if len(authorization_contexts) != 1:
            raise AtomicSemanticSynthesisError(
                "matrix cells must share one authorized scorer/run/scope"
            )
        calls_by_panel: dict[str, tuple[object, ...]] = {}
        for cell in self.cells:
            authorization = cell.evidence_binding
            call_context = (
                authorization.scorer_receipt_digest,
                authorization.scorer_output_digest,
                authorization.scorer_call_digest,
                authorization.scorer_call_ordinal,
            )
            previous = calls_by_panel.setdefault(cell.panel_id, call_context)
            if previous != call_context:
                raise AtomicSemanticSynthesisError(
                    "all atoms for one panel must come from one exact scorer call"
                )
        call_digests = tuple(value[2] for value in calls_by_panel.values())
        call_ordinals = tuple(value[3] for value in calls_by_panel.values())
        receipt_digests = tuple(value[0] for value in calls_by_panel.values())
        if (
            len(call_digests) != len(set(call_digests))
            or len(call_ordinals) != len(set(call_ordinals))
            or len(receipt_digests) != len(set(receipt_digests))
        ):
            raise AtomicSemanticSynthesisError(
                "different support panels must use distinct scorer calls"
            )
        _digest(self.matrix_digest, "matrix_digest")
        if self.matrix_digest != canonical_digest(self.content_data()):
            raise AtomicSemanticSynthesisError(
                "matrix_digest differs from its exact Cartesian preimage"
            )

    @classmethod
    def create(
        cls,
        atoms: Sequence[AtomicSoftPredicate],
        cells: Sequence[AtomicSupportCell],
    ) -> "AtomicSupportMatrix":
        ordered_atoms = tuple(sorted(atoms, key=lambda atom: atom.atom_id))
        if not ordered_atoms:
            raise AtomicSemanticSynthesisError("support matrix requires atoms")
        panel_ids = tuple(
            item.panel_id for item in ordered_atoms[0].panel_descriptions
        )
        ordered_cells = tuple(
            sorted(cells, key=lambda cell: (cell.atom_id, cell.panel_id))
        )
        content = {
            "schema": ATOMIC_SUPPORT_MATRIX_SCHEMA,
            "atoms": [atom.to_data() for atom in ordered_atoms],
            "panel_ids": list(panel_ids),
            "cells": [cell.to_data() for cell in ordered_cells],
            "source_proposal_digest": ordered_atoms[0].source_proposal_digest,
            "scorer_protocol_digest": ordered_atoms[0].scorer_protocol_digest,
            "panel_descriptions_digest": ordered_atoms[0].panel_descriptions_digest,
        }
        return cls(
            atoms=ordered_atoms,
            panel_ids=panel_ids,
            cells=ordered_cells,
            source_proposal_digest=ordered_atoms[0].source_proposal_digest,
            scorer_protocol_digest=ordered_atoms[0].scorer_protocol_digest,
            panel_descriptions_digest=ordered_atoms[0].panel_descriptions_digest,
            matrix_digest=canonical_digest(content),
        )

    def content_data(self) -> dict[str, object]:
        return {
            "schema": ATOMIC_SUPPORT_MATRIX_SCHEMA,
            "atoms": [atom.to_data() for atom in self.atoms],
            "panel_ids": list(self.panel_ids),
            "cells": [cell.to_data() for cell in self.cells],
            "source_proposal_digest": self.source_proposal_digest,
            "scorer_protocol_digest": self.scorer_protocol_digest,
            "panel_descriptions_digest": self.panel_descriptions_digest,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "matrix_digest": self.matrix_digest}

    def cell(self, atom_id: str, panel_id: str) -> AtomicSupportCell:
        for cell in self.cells:
            if cell.atom_id == atom_id and cell.panel_id == panel_id:
                return cell
        raise AtomicSemanticSynthesisError(
            f"missing matrix cell atom={atom_id!r}, panel={panel_id!r}"
        )

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "AtomicSupportMatrix":
        data = _mapping(
            value,
            frozenset(
                {
                    "schema",
                    "atoms",
                    "panel_ids",
                    "cells",
                    "source_proposal_digest",
                    "scorer_protocol_digest",
                    "panel_descriptions_digest",
                    "matrix_digest",
                }
            ),
            "atomic support matrix",
        )
        if data["schema"] != ATOMIC_SUPPORT_MATRIX_SCHEMA:
            raise AtomicSemanticSynthesisError("unsupported support matrix schema")
        raw_atoms = data["atoms"]
        raw_panels = data["panel_ids"]
        raw_cells = data["cells"]
        if (
            not isinstance(raw_atoms, list)
            or not isinstance(raw_panels, list)
            or not isinstance(raw_cells, list)
        ):
            raise AtomicSemanticSynthesisError(
                "matrix atoms, panel_ids, and cells must be lists"
            )
        result = cls(
            atoms=tuple(AtomicSoftPredicate.from_data(item) for item in raw_atoms),
            panel_ids=tuple(raw_panels),
            cells=tuple(AtomicSupportCell.from_data(item) for item in raw_cells),
            source_proposal_digest=data["source_proposal_digest"],
            scorer_protocol_digest=data["scorer_protocol_digest"],
            panel_descriptions_digest=data["panel_descriptions_digest"],
            matrix_digest=data["matrix_digest"],
        )
        if not _same_canonical_json(result.to_data(), value):
            raise AtomicSemanticSynthesisError(
                "atomic support matrix is not canonically represented"
            )
        return result


@dataclass(frozen=True, slots=True)
class AtomEligibilityDiagnostic:
    atom_id: str
    rejection_reasons: tuple[str, ...]
    covered_negative_panel_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _digest(self.atom_id, "diagnostic atom_id")
        if (
            self.rejection_reasons != tuple(sorted(self.rejection_reasons))
            or len(self.rejection_reasons) != len(set(self.rejection_reasons))
            or any(not reason.strip() for reason in self.rejection_reasons)
        ):
            raise AtomicSemanticSynthesisError(
                "diagnostic rejection reasons must be non-empty, unique, and sorted"
            )
        for panel_id in self.covered_negative_panel_ids:
            _panel_id(panel_id, "covered negative panel_id")
        if (
            self.covered_negative_panel_ids
            != tuple(sorted(self.covered_negative_panel_ids))
            or len(self.covered_negative_panel_ids)
            != len(set(self.covered_negative_panel_ids))
        ):
            raise AtomicSemanticSynthesisError(
                "diagnostic negative coverage must be unique and sorted"
            )

    @property
    def eligible(self) -> bool:
        return not self.rejection_reasons

    def to_data(self) -> dict[str, object]:
        return {
            "atom_id": self.atom_id,
            "eligible": self.eligible,
            "rejection_reasons": list(self.rejection_reasons),
            "covered_negative_panel_ids": list(
                self.covered_negative_panel_ids
            ),
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "AtomEligibilityDiagnostic":
        data = _mapping(
            value,
            frozenset(
                {
                    "atom_id",
                    "eligible",
                    "rejection_reasons",
                    "covered_negative_panel_ids",
                }
            ),
            "atom eligibility diagnostic",
        )
        reasons = data["rejection_reasons"]
        covered = data["covered_negative_panel_ids"]
        if not isinstance(reasons, list) or not isinstance(covered, list):
            raise AtomicSemanticSynthesisError(
                "diagnostic reasons and coverage must be lists"
            )
        result = cls(
            atom_id=data["atom_id"],
            rejection_reasons=tuple(reasons),
            covered_negative_panel_ids=tuple(covered),
        )
        if (
            type(data["eligible"]) is not bool
            or data["eligible"] is not result.eligible
        ):
            raise AtomicSemanticSynthesisError(
                "diagnostic eligible flag differs from rejection reasons"
            )
        if not _same_canonical_json(result.to_data(), value):
            raise AtomicSemanticSynthesisError(
                "atom eligibility diagnostic is not canonical"
            )
        return result


@dataclass(frozen=True, slots=True)
class NoExactSeparatorDiagnostics:
    matrix_digest: str
    support_labels_digest: str
    max_atoms: int
    selection_scope: str
    positive_panel_ids: tuple[str, ...]
    negative_panel_ids: tuple[str, ...]
    atom_diagnostics: tuple[AtomEligibilityDiagnostic, ...]
    best_attempt_atom_ids: tuple[str, ...]
    best_attempt_covered_negative_panel_ids: tuple[str, ...]
    uncovered_by_any_eligible_atom: tuple[str, ...]
    reason: str
    diagnostic_digest: str

    def __post_init__(self) -> None:
        _digest(self.matrix_digest, "diagnostic matrix_digest")
        _digest(self.support_labels_digest, "diagnostic support_labels_digest")
        if (
            isinstance(self.max_atoms, bool)
            or not isinstance(self.max_atoms, int)
            or not 1 <= self.max_atoms <= MAX_ATOMIC_CONJUNCTION_SIZE
        ):
            raise AtomicSemanticSynthesisError(
                "diagnostic max_atoms lies outside the frozen small-conjunction bound"
            )
        if _selection_scope(self.selection_scope) == CALIBRATED_SELECTION_SCOPE:
            raise AtomicSemanticSynthesisError(
                "calibrated-semantic diagnostics are disabled until calibration "
                "is independently cold-validated"
            )
        for panel_id in (*self.positive_panel_ids, *self.negative_panel_ids):
            _panel_id(panel_id, "diagnostic panel_id")
        if (
            not self.positive_panel_ids
            or not self.negative_panel_ids
            or self.positive_panel_ids != tuple(sorted(self.positive_panel_ids))
            or self.negative_panel_ids != tuple(sorted(self.negative_panel_ids))
            or len(self.positive_panel_ids) != len(set(self.positive_panel_ids))
            or len(self.negative_panel_ids) != len(set(self.negative_panel_ids))
            or set(self.positive_panel_ids) & set(self.negative_panel_ids)
        ):
            raise AtomicSemanticSynthesisError(
                "diagnostic class panel IDs must be non-empty, disjoint, and sorted"
            )
        diagnostic_ids = tuple(item.atom_id for item in self.atom_diagnostics)
        if diagnostic_ids != tuple(sorted(diagnostic_ids)) or len(
            diagnostic_ids
        ) != len(set(diagnostic_ids)):
            raise AtomicSemanticSynthesisError(
                "atom diagnostics must have unique canonical digest order"
            )
        negative_set = set(self.negative_panel_ids)
        if any(
            not set(item.covered_negative_panel_ids).issubset(negative_set)
            for item in self.atom_diagnostics
        ):
            raise AtomicSemanticSynthesisError(
                "atom diagnostic coverage contains a non-negative panel"
            )
        if (
            self.best_attempt_atom_ids
            != tuple(sorted(self.best_attempt_atom_ids))
            or len(self.best_attempt_atom_ids)
            != len(set(self.best_attempt_atom_ids))
        ):
            raise AtomicSemanticSynthesisError(
                "best-attempt atom IDs must be in canonical digest order"
            )
        if len(self.best_attempt_atom_ids) > self.max_atoms or not set(
            self.best_attempt_atom_ids
        ).issubset({item.atom_id for item in self.atom_diagnostics if item.eligible}):
            raise AtomicSemanticSynthesisError(
                "best-attempt atom IDs differ from eligible bounded candidates"
            )
        for values, label in (
            (
                self.best_attempt_covered_negative_panel_ids,
                "best-attempt coverage",
            ),
            (self.uncovered_by_any_eligible_atom, "globally uncovered panels"),
        ):
            if (
                values != tuple(sorted(values))
                or len(values) != len(set(values))
                or not set(values).issubset(negative_set)
            ):
                raise AtomicSemanticSynthesisError(
                    f"diagnostic {label} must be unique, sorted negative panel IDs"
                )
        diagnostic_by_id = {item.atom_id: item for item in self.atom_diagnostics}
        expected_best_covered = tuple(
            sorted(
                {
                    panel_id
                    for atom_id in self.best_attempt_atom_ids
                    for panel_id in diagnostic_by_id[
                        atom_id
                    ].covered_negative_panel_ids
                }
            )
        )
        if self.best_attempt_covered_negative_panel_ids != expected_best_covered:
            raise AtomicSemanticSynthesisError(
                "best-attempt coverage differs from its selected atom diagnostics"
            )
        eligible = tuple(item for item in self.atom_diagnostics if item.eligible)
        covered_by_any = {
            panel_id
            for item in eligible
            for panel_id in item.covered_negative_panel_ids
        }
        expected_uncovered = tuple(sorted(negative_set - covered_by_any))
        if self.uncovered_by_any_eligible_atom != expected_uncovered:
            raise AtomicSemanticSynthesisError(
                "globally uncovered panels differ from eligible atom diagnostics"
            )
        expected_reason = (
            "no atom is total and present on every positive support panel"
            if not eligible
            else "some negative support panels have no authorized nonmatch witness"
            if expected_uncovered
            else "no exact positive conjunction exists within max_atoms"
        )
        if self.reason != expected_reason:
            raise AtomicSemanticSynthesisError(
                "diagnostic reason differs from its recomputed local state"
            )
        _digest(self.diagnostic_digest, "diagnostic_digest")
        if self.diagnostic_digest != canonical_digest(self.content_data()):
            raise AtomicSemanticSynthesisError(
                "diagnostic_digest differs from the no-separator preimage"
            )

    def content_data(self) -> dict[str, object]:
        return {
            "schema": NO_SEPARATOR_DIAGNOSTICS_SCHEMA,
            "matrix_digest": self.matrix_digest,
            "support_labels_digest": self.support_labels_digest,
            "max_atoms": self.max_atoms,
            "selection_scope": self.selection_scope,
            "positive_panel_ids": list(self.positive_panel_ids),
            "negative_panel_ids": list(self.negative_panel_ids),
            "atom_diagnostics": [item.to_data() for item in self.atom_diagnostics],
            "best_attempt_atom_ids": list(self.best_attempt_atom_ids),
            "best_attempt_covered_negative_panel_ids": list(
                self.best_attempt_covered_negative_panel_ids
            ),
            "uncovered_by_any_eligible_atom": list(
                self.uncovered_by_any_eligible_atom
            ),
            "reason": self.reason,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "diagnostic_digest": self.diagnostic_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "NoExactSeparatorDiagnostics":
        data = _mapping(
            value,
            frozenset(
                {
                    "schema",
                    "matrix_digest",
                    "support_labels_digest",
                    "max_atoms",
                    "selection_scope",
                    "positive_panel_ids",
                    "negative_panel_ids",
                    "atom_diagnostics",
                    "best_attempt_atom_ids",
                    "best_attempt_covered_negative_panel_ids",
                    "uncovered_by_any_eligible_atom",
                    "reason",
                    "diagnostic_digest",
                }
            ),
            "no-separator diagnostics",
        )
        if data["schema"] != NO_SEPARATOR_DIAGNOSTICS_SCHEMA:
            raise AtomicSemanticSynthesisError(
                "unsupported no-separator diagnostic schema"
            )
        list_fields = (
            "positive_panel_ids",
            "negative_panel_ids",
            "atom_diagnostics",
            "best_attempt_atom_ids",
            "best_attempt_covered_negative_panel_ids",
            "uncovered_by_any_eligible_atom",
        )
        if any(not isinstance(data[field], list) for field in list_fields):
            raise AtomicSemanticSynthesisError(
                "no-separator diagnostic sequence fields must be lists"
            )
        result = cls(
            matrix_digest=data["matrix_digest"],
            support_labels_digest=data["support_labels_digest"],
            max_atoms=data["max_atoms"],
            selection_scope=data["selection_scope"],
            positive_panel_ids=tuple(data["positive_panel_ids"]),
            negative_panel_ids=tuple(data["negative_panel_ids"]),
            atom_diagnostics=tuple(
                AtomEligibilityDiagnostic.from_data(item)
                for item in data["atom_diagnostics"]
            ),
            best_attempt_atom_ids=tuple(data["best_attempt_atom_ids"]),
            best_attempt_covered_negative_panel_ids=tuple(
                data["best_attempt_covered_negative_panel_ids"]
            ),
            uncovered_by_any_eligible_atom=tuple(
                data["uncovered_by_any_eligible_atom"]
            ),
            reason=data["reason"],
            diagnostic_digest=data["diagnostic_digest"],
        )
        if not _same_canonical_json(result.to_data(), value):
            raise AtomicSemanticSynthesisError(
                "no-separator diagnostics are not canonically represented"
            )
        return result


class NoExactSeparatorError(AtomicSemanticSynthesisError):
    """No admissible small positive conjunction separates the support set."""

    def __init__(self, diagnostics: NoExactSeparatorDiagnostics) -> None:
        self.diagnostics = diagnostics
        super().__init__(
            f"{diagnostics.reason}; diagnostic_digest="
            f"{diagnostics.diagnostic_digest}"
        )


def _canonical_labels(
    panel_ids: tuple[str, ...], support_labels: Mapping[str, bool]
) -> tuple[tuple[str, bool], ...]:
    if not isinstance(support_labels, Mapping) or any(
        not isinstance(key, str) for key in support_labels
    ):
        raise AtomicSemanticSynthesisError("support_labels must be a string mapping")
    if set(support_labels) != set(panel_ids):
        missing = sorted(set(panel_ids) - set(support_labels))
        extra = sorted(set(support_labels) - set(panel_ids))
        raise AtomicSemanticSynthesisError(
            "support labels differ from matrix panels: "
            f"missing={missing}, extra={extra}"
        )
    if any(type(value) is not bool for value in support_labels.values()):
        raise AtomicSemanticSynthesisError("support labels must be literal booleans")
    result = tuple((panel_id, support_labels[panel_id]) for panel_id in panel_ids)
    if {positive for _, positive in result} != {False, True}:
        raise AtomicSemanticSynthesisError(
            "support labels must contain positive and negative panels"
        )
    return result


def _labels_data(labels: tuple[tuple[str, bool], ...]) -> list[dict[str, object]]:
    return [
        {"panel_id": panel_id, "positive": positive}
        for panel_id, positive in labels
    ]


def _analyse_atoms(
    matrix: AtomicSupportMatrix,
    labels: tuple[tuple[str, bool], ...],
    selection_scope: str,
) -> tuple[AtomEligibilityDiagnostic, ...]:
    positives = tuple(panel_id for panel_id, positive in labels if positive)
    negatives = tuple(panel_id for panel_id, positive in labels if not positive)
    result: list[AtomEligibilityDiagnostic] = []
    for atom in matrix.atoms:
        reasons: list[str] = []
        covered: list[str] = []
        for panel_id in matrix.panel_ids:
            cell = matrix.cell(atom.atom_id, panel_id)
            disposition = cell.evidence.disposition
            if cell.evidence_binding.observation_scope != selection_scope:
                reasons.append(
                    f"scope_mismatch:{panel_id}:"
                    f"{cell.evidence_binding.observation_scope}"
                )
            if (
                selection_scope == CALIBRATED_SELECTION_SCOPE
                and cell.evidence_binding.calibration_digest is None
            ):
                reasons.append(f"uncalibrated:{panel_id}")
            if isinstance(cell.evidence, OperationalNonmatchRecord):
                if selection_scope != OPERATIONAL_SELECTION_SCOPE:
                    reasons.append(f"non_total:{panel_id}:operational_nonmatch")
            elif disposition in {Disposition.INDETERMINATE, Disposition.ERROR}:
                reasons.append(f"non_total:{panel_id}:{disposition.value}")
        for panel_id in positives:
            evidence = matrix.cell(atom.atom_id, panel_id).evidence
            disposition = evidence.disposition
            if disposition is not Disposition.PRESENT or isinstance(
                evidence, OperationalNonmatchRecord
            ):
                disposition_name = (
                    "operational_nonmatch"
                    if isinstance(evidence, OperationalNonmatchRecord)
                    else disposition.value
                )
                reasons.append(
                    f"positive_not_present:{panel_id}:{disposition_name}"
                )
        for panel_id in negatives:
            evidence = matrix.cell(atom.atom_id, panel_id).evidence
            if selection_scope == OPERATIONAL_SELECTION_SCOPE and isinstance(
                evidence, OperationalNonmatchRecord
            ):
                covered.append(panel_id)
            elif (
                selection_scope == CALIBRATED_SELECTION_SCOPE
                and evidence.disposition is Disposition.CERTIFIED_ABSENT
            ):
                covered.append(panel_id)
        result.append(
            AtomEligibilityDiagnostic(
                atom_id=atom.atom_id,
                rejection_reasons=tuple(sorted(reasons)),
                covered_negative_panel_ids=tuple(sorted(covered)),
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class _Selection:
    atom_ids: tuple[str, ...]
    description_utf8_bytes: int
    negative_coverage: tuple[tuple[str, tuple[str, ...]], ...]
    load_bearing: tuple[tuple[str, tuple[str, ...]], ...]
    eligible_atom_ids: tuple[str, ...]


def _selection_or_diagnostics(
    matrix: AtomicSupportMatrix,
    labels: tuple[tuple[str, bool], ...],
    max_atoms: int,
    selection_scope: str,
) -> _Selection | NoExactSeparatorDiagnostics:
    if (
        isinstance(max_atoms, bool)
        or not isinstance(max_atoms, int)
        or not 1 <= max_atoms <= MAX_ATOMIC_CONJUNCTION_SIZE
    ):
        raise AtomicSemanticSynthesisError(
            "max_atoms lies outside the frozen 1..4 small-conjunction bound"
        )
    scope = _selection_scope(selection_scope)
    if scope == CALIBRATED_SELECTION_SCOPE:
        raise AtomicSemanticSynthesisError(
            "calibrated-semantic selection is disabled until a typed calibration "
            "artifact and interval rule are independently cold-validated"
        )
    diagnostics = _analyse_atoms(matrix, labels, scope)
    eligible = tuple(item for item in diagnostics if item.eligible)
    eligible_ids = tuple(item.atom_id for item in eligible)
    negative_ids = tuple(panel_id for panel_id, positive in labels if not positive)
    negative_set = frozenset(negative_ids)
    coverage = {
        item.atom_id: frozenset(item.covered_negative_panel_ids)
        for item in eligible
    }
    atoms_by_id = {atom.atom_id: atom for atom in matrix.atoms}

    chosen: tuple[str, ...] | None = None
    chosen_mdl = 0
    for size in range(1, min(max_atoms, len(eligible_ids)) + 1):
        options: list[tuple[int, tuple[str, ...]]] = []
        for atom_ids in combinations(eligible_ids, size):
            covered = frozenset().union(*(coverage[item] for item in atom_ids))
            if covered == negative_set:
                mdl = sum(
                    atoms_by_id[item].description_utf8_bytes for item in atom_ids
                )
                options.append((mdl, atom_ids))
        if options:
            chosen_mdl, chosen = min(options, key=lambda item: (item[0], item[1]))
            break

    if chosen is not None:
        negative_coverage = tuple(
            (
                panel_id,
                tuple(atom_id for atom_id in chosen if panel_id in coverage[atom_id]),
            )
            for panel_id in negative_ids
        )
        load_bearing = tuple(
            (
                atom_id,
                tuple(
                    panel_id
                    for panel_id in negative_ids
                    if panel_id in coverage[atom_id]
                    and all(
                        panel_id not in coverage[other]
                        for other in chosen
                        if other != atom_id
                    )
                ),
            )
            for atom_id in chosen
        )
        if any(not panel_ids for _, panel_ids in load_bearing):
            raise AtomicSemanticSynthesisError(
                "minimum-cardinality selection contains a non-load-bearing atom"
            )
        return _Selection(
            atom_ids=chosen,
            description_utf8_bytes=chosen_mdl,
            negative_coverage=negative_coverage,
            load_bearing=load_bearing,
            eligible_atom_ids=eligible_ids,
        )

    best_ids: tuple[str, ...] = ()
    best_covered: frozenset[str] = frozenset()
    best_key: tuple[int, int, int, tuple[str, ...]] | None = None
    for size in range(1, min(max_atoms, len(eligible_ids)) + 1):
        for atom_ids in combinations(eligible_ids, size):
            covered = frozenset().union(*(coverage[item] for item in atom_ids))
            mdl = sum(
                atoms_by_id[item].description_utf8_bytes for item in atom_ids
            )
            key = (-len(covered), size, mdl, atom_ids)
            if best_key is None or key < best_key:
                best_key = key
                best_ids = atom_ids
                best_covered = covered
    covered_by_any = frozenset().union(
        *(coverage[item] for item in eligible_ids)
    ) if eligible_ids else frozenset()
    uncovered_by_any = tuple(sorted(negative_set - covered_by_any))
    if not eligible_ids:
        reason = "no atom is total and present on every positive support panel"
    elif uncovered_by_any:
        reason = "some negative support panels have no authorized nonmatch witness"
    else:
        reason = "no exact positive conjunction exists within max_atoms"
    labels_digest = canonical_digest(_labels_data(labels))
    content = {
        "schema": NO_SEPARATOR_DIAGNOSTICS_SCHEMA,
        "matrix_digest": matrix.matrix_digest,
        "support_labels_digest": labels_digest,
        "max_atoms": max_atoms,
        "selection_scope": scope,
        "positive_panel_ids": [
            panel_id for panel_id, positive in labels if positive
        ],
        "negative_panel_ids": list(negative_ids),
        "atom_diagnostics": [item.to_data() for item in diagnostics],
        "best_attempt_atom_ids": list(best_ids),
        "best_attempt_covered_negative_panel_ids": sorted(best_covered),
        "uncovered_by_any_eligible_atom": list(uncovered_by_any),
        "reason": reason,
    }
    return NoExactSeparatorDiagnostics(
        matrix_digest=matrix.matrix_digest,
        support_labels_digest=labels_digest,
        max_atoms=max_atoms,
        selection_scope=scope,
        positive_panel_ids=tuple(
            panel_id for panel_id, positive in labels if positive
        ),
        negative_panel_ids=negative_ids,
        atom_diagnostics=diagnostics,
        best_attempt_atom_ids=best_ids,
        best_attempt_covered_negative_panel_ids=tuple(sorted(best_covered)),
        uncovered_by_any_eligible_atom=uncovered_by_any,
        reason=reason,
        diagnostic_digest=canonical_digest(content),
    )


def _formula_data(atom_ids: tuple[str, ...]) -> dict[str, object]:
    return {"kind": "all", "atom_ids": list(atom_ids)}


def _archive_content(
    *,
    matrix: AtomicSupportMatrix,
    labels: tuple[tuple[str, bool], ...],
    labels_digest: str,
    max_atoms: int,
    selection_scope: str,
    selection: _Selection,
) -> dict[str, object]:
    operational = selection_scope == OPERATIONAL_SELECTION_SCOPE
    return {
        "schema": ATOMIC_SELECTION_ARCHIVE_SCHEMA,
        "matrix": matrix.to_data(),
        "support_labels": _labels_data(labels),
        "support_labels_digest": labels_digest,
        "max_atoms": max_atoms,
        "selection_scope": selection_scope,
        "claim_authority": {
            "calibration_authorized": not operational,
            "benchmark_claim_authorized": False,
            "semantic_truth_claim": not operational,
        },
        "formula": _formula_data(selection.atom_ids),
        "selection_objective": {
            "atom_count": len(selection.atom_ids),
            "description_utf8_bytes": selection.description_utf8_bytes,
            "atom_digest_tuple": list(selection.atom_ids),
        },
        "eligible_atom_ids": list(selection.eligible_atom_ids),
        "negative_coverage": [
            {"panel_id": panel_id, "atom_ids": list(atom_ids)}
            for panel_id, atom_ids in selection.negative_coverage
        ],
        "load_bearing": [
            {
                "atom_id": atom_id,
                "unique_negative_panel_ids": list(panel_ids),
            }
            for atom_id, panel_ids in selection.load_bearing
        ],
    }


@dataclass(frozen=True, slots=True)
class AtomicSelectionArchive:
    """Content-addressed exact separator and all support-only inputs."""

    matrix: AtomicSupportMatrix
    support_labels: tuple[tuple[str, bool], ...]
    support_labels_digest: str
    max_atoms: int
    selection_scope: str
    selected_atom_ids: tuple[str, ...]
    description_utf8_bytes: int
    eligible_atom_ids: tuple[str, ...]
    negative_coverage: tuple[tuple[str, tuple[str, ...]], ...]
    load_bearing: tuple[tuple[str, tuple[str, ...]], ...]
    archive_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.matrix, AtomicSupportMatrix):
            raise TypeError("matrix must be an AtomicSupportMatrix")
        scope = _selection_scope(self.selection_scope)
        if (
            type(self.description_utf8_bytes) is not int
            or self.description_utf8_bytes < 0
        ):
            raise AtomicSemanticSynthesisError(
                "description_utf8_bytes must be a literal non-negative integer"
            )
        labels = _canonical_labels(
            self.matrix.panel_ids, dict(self.support_labels)
        )
        if labels != self.support_labels:
            raise AtomicSemanticSynthesisError(
                "archive support labels are not canonical"
            )
        _digest(self.support_labels_digest, "support_labels_digest")
        if self.support_labels_digest != canonical_digest(_labels_data(labels)):
            raise AtomicSemanticSynthesisError(
                "support_labels_digest differs from support labels"
            )
        expected = _selection_or_diagnostics(
            self.matrix, labels, self.max_atoms, scope
        )
        if isinstance(expected, NoExactSeparatorDiagnostics):
            raise AtomicSemanticSynthesisError(
                "selection archive claims a separator where none exists"
            )
        if (
            self.selected_atom_ids != expected.atom_ids
            or self.description_utf8_bytes != expected.description_utf8_bytes
            or self.eligible_atom_ids != expected.eligible_atom_ids
            or self.negative_coverage != expected.negative_coverage
            or self.load_bearing != expected.load_bearing
        ):
            raise AtomicSemanticSynthesisError(
                "selection archive differs from exact deterministic synthesis"
            )
        _digest(self.archive_digest, "archive_digest")
        if self.archive_digest != canonical_digest(self.content_data()):
            raise AtomicSemanticSynthesisError(
                "archive_digest differs from the selection preimage"
            )
        # Replay is part of construction: an archive is valid only if its
        # formula reproduces every support label in its declared scope.
        self.replay_support()

    @property
    def formula(self) -> dict[str, object]:
        return _formula_data(self.selected_atom_ids)

    def content_data(self) -> dict[str, object]:
        selection = _Selection(
            atom_ids=self.selected_atom_ids,
            description_utf8_bytes=self.description_utf8_bytes,
            negative_coverage=self.negative_coverage,
            load_bearing=self.load_bearing,
            eligible_atom_ids=self.eligible_atom_ids,
        )
        return _archive_content(
            matrix=self.matrix,
            labels=self.support_labels,
            labels_digest=self.support_labels_digest,
            max_atoms=self.max_atoms,
            selection_scope=self.selection_scope,
            selection=selection,
        )

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "archive_digest": self.archive_digest}

    def replay_support(
        self,
    ) -> tuple[
        tuple[str, TruthEvidenceRecord | OperationalNonmatchRecord], ...
    ]:
        """Replay the frozen conjunction from archived cells only."""

        result: list[
            tuple[str, TruthEvidenceRecord | OperationalNonmatchRecord]
        ] = []
        for panel_id, positive in self.support_labels:
            cells = {
                atom_id: self.matrix.cell(atom_id, panel_id)
                for atom_id in self.selected_atom_ids
            }
            evidence = {
                atom_id: cell.evidence.to_evidence()
                for atom_id, cell in cells.items()
            }
            bindings = {
                atom_id: cell.evidence_binding
                for atom_id, cell in cells.items()
            }
            combined = evaluate_atomic_formula(
                self.formula,
                evidence,
                provenance_bindings=bindings,
                selection_scope=self.selection_scope,
            )
            if positive and combined.disposition is not Disposition.PRESENT:
                raise AtomicSemanticSynthesisError(
                    f"selected formula does not present positive panel {panel_id}"
                )
            negative_rejected = (
                combined.is_operational_nonmatch
                if self.selection_scope == OPERATIONAL_SELECTION_SCOPE
                else combined.disposition is Disposition.CERTIFIED_ABSENT
            )
            if not positive and not negative_rejected:
                raise AtomicSemanticSynthesisError(
                    f"selected formula does not reject negative panel {panel_id}"
                )
            replay_record: TruthEvidenceRecord | OperationalNonmatchRecord
            if combined.is_operational_nonmatch:
                replay_record = OperationalNonmatchRecord.from_evidence(combined)
            else:
                replay_record = TruthEvidenceRecord.from_evidence(combined)
            result.append((panel_id, replay_record))
        return tuple(result)

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "AtomicSelectionArchive":
        data = _mapping(
            value,
            frozenset(
                {
                    "schema",
                    "matrix",
                    "support_labels",
                    "support_labels_digest",
                    "max_atoms",
                    "selection_scope",
                    "claim_authority",
                    "formula",
                    "selection_objective",
                    "eligible_atom_ids",
                    "negative_coverage",
                    "load_bearing",
                    "archive_digest",
                }
            ),
            "atomic selection archive",
        )
        if data["schema"] != ATOMIC_SELECTION_ARCHIVE_SCHEMA:
            raise AtomicSemanticSynthesisError("unsupported selection archive schema")
        if not isinstance(data["matrix"], Mapping):
            raise AtomicSemanticSynthesisError("archive matrix must be an object")
        matrix = AtomicSupportMatrix.from_data(data["matrix"])
        scope = _selection_scope(data["selection_scope"])
        authority = _mapping(
            data["claim_authority"],
            frozenset(
                {
                    "calibration_authorized",
                    "benchmark_claim_authorized",
                    "semantic_truth_claim",
                }
            ),
            "selection claim authority",
        )
        expected_authority = {
            "calibration_authorized": scope == CALIBRATED_SELECTION_SCOPE,
            "benchmark_claim_authorized": False,
            "semantic_truth_claim": scope == CALIBRATED_SELECTION_SCOPE,
        }
        if (
            any(type(item) is not bool for item in authority.values())
            or dict(authority) != expected_authority
        ):
            raise AtomicSemanticSynthesisError(
                "selection claim authority differs from its frozen scope"
            )
        raw_labels = data["support_labels"]
        if not isinstance(raw_labels, list):
            raise AtomicSemanticSynthesisError("support_labels must be a list")
        labels: list[tuple[str, bool]] = []
        for item in raw_labels:
            raw = _mapping(
                item,
                frozenset({"panel_id", "positive"}),
                "support label",
            )
            if type(raw["positive"]) is not bool:
                raise AtomicSemanticSynthesisError(
                    "archived support label must be a literal boolean"
                )
            labels.append((_panel_id(raw["panel_id"]), raw["positive"]))
        formula = _mapping(
            data["formula"],
            frozenset({"kind", "atom_ids"}),
            "frozen atomic formula",
        )
        raw_selected = formula["atom_ids"]
        if formula["kind"] != "all" or not isinstance(raw_selected, list):
            raise AtomicSemanticSynthesisError(
                "frozen atomic formula must be an all/atom_ids object"
            )
        objective = _mapping(
            data["selection_objective"],
            frozenset(
                {"atom_count", "description_utf8_bytes", "atom_digest_tuple"}
            ),
            "selection objective",
        )
        if (
            type(objective["atom_count"]) is not int
            or objective["atom_count"] < 1
            or objective["atom_count"] > MAX_ATOMIC_CONJUNCTION_SIZE
        ):
            raise AtomicSemanticSynthesisError(
                "selection objective atom_count must be a literal integer in 1..4"
            )
        if (
            type(objective["description_utf8_bytes"]) is not int
            or objective["description_utf8_bytes"] < 0
        ):
            raise AtomicSemanticSynthesisError(
                "selection objective description_utf8_bytes must be a literal "
                "non-negative integer"
            )
        if objective["atom_digest_tuple"] != raw_selected or objective[
            "atom_count"
        ] != len(raw_selected):
            raise AtomicSemanticSynthesisError(
                "selection objective differs from frozen formula"
            )
        raw_eligible = data["eligible_atom_ids"]
        raw_coverage = data["negative_coverage"]
        raw_load = data["load_bearing"]
        if (
            not isinstance(raw_eligible, list)
            or not isinstance(raw_coverage, list)
            or not isinstance(raw_load, list)
        ):
            raise AtomicSemanticSynthesisError(
                "eligible, coverage, and load-bearing values must be lists"
            )
        coverage: list[tuple[str, tuple[str, ...]]] = []
        for item in raw_coverage:
            raw = _mapping(
                item,
                frozenset({"panel_id", "atom_ids"}),
                "negative coverage",
            )
            if not isinstance(raw["atom_ids"], list):
                raise AtomicSemanticSynthesisError(
                    "negative coverage atom_ids must be a list"
                )
            coverage.append((raw["panel_id"], tuple(raw["atom_ids"])))
        load: list[tuple[str, tuple[str, ...]]] = []
        for item in raw_load:
            raw = _mapping(
                item,
                frozenset({"atom_id", "unique_negative_panel_ids"}),
                "load-bearing record",
            )
            if not isinstance(raw["unique_negative_panel_ids"], list):
                raise AtomicSemanticSynthesisError(
                    "unique_negative_panel_ids must be a list"
                )
            load.append(
                (raw["atom_id"], tuple(raw["unique_negative_panel_ids"]))
            )
        result = cls(
            matrix=matrix,
            support_labels=tuple(labels),
            support_labels_digest=data["support_labels_digest"],
            max_atoms=data["max_atoms"],
            selection_scope=scope,
            selected_atom_ids=tuple(raw_selected),
            description_utf8_bytes=objective["description_utf8_bytes"],
            eligible_atom_ids=tuple(raw_eligible),
            negative_coverage=tuple(coverage),
            load_bearing=tuple(load),
            archive_digest=data["archive_digest"],
        )
        if not _same_canonical_json(result.to_data(), value):
            raise AtomicSemanticSynthesisError(
                "selection archive is not canonically represented"
            )
        return result


def synthesize_atomic_conjunction(
    matrix: AtomicSupportMatrix,
    support_labels: Mapping[str, bool],
    *,
    selection_scope: str,
    max_atoms: int = 4,
) -> AtomicSelectionArchive:
    """Select the exact positive support separator under the frozen tie-break.

    The objective is minimum atom count, then minimum total UTF-8 bytes across
    selected claim/cue descriptions, then the canonical atom-digest tuple.
    """

    if not isinstance(matrix, AtomicSupportMatrix):
        raise TypeError("matrix must be an AtomicSupportMatrix")
    scope = _selection_scope(selection_scope)
    labels = _canonical_labels(matrix.panel_ids, support_labels)
    selected = _selection_or_diagnostics(matrix, labels, max_atoms, scope)
    if isinstance(selected, NoExactSeparatorDiagnostics):
        raise NoExactSeparatorError(selected)
    labels_digest = canonical_digest(_labels_data(labels))
    content = _archive_content(
        matrix=matrix,
        labels=labels,
        labels_digest=labels_digest,
        max_atoms=max_atoms,
        selection_scope=scope,
        selection=selected,
    )
    return AtomicSelectionArchive(
        matrix=matrix,
        support_labels=labels,
        support_labels_digest=labels_digest,
        max_atoms=max_atoms,
        selection_scope=scope,
        selected_atom_ids=selected.atom_ids,
        description_utf8_bytes=selected.description_utf8_bytes,
        eligible_atom_ids=selected.eligible_atom_ids,
        negative_coverage=selected.negative_coverage,
        load_bearing=selected.load_bearing,
        archive_digest=canonical_digest(content),
    )


def _formula_atom_ids(formula: Mapping[str, Any]) -> tuple[str, ...]:
    data = _mapping(
        formula,
        frozenset({"kind", "atom_ids"}),
        "frozen atomic formula",
    )
    raw_ids = data["atom_ids"]
    if data["kind"] != "all" or not isinstance(raw_ids, list) or not raw_ids:
        raise AtomicSemanticSynthesisError(
            "frozen atomic formula must be a non-empty all/atom_ids object"
        )
    atom_ids = tuple(_digest(item, "formula atom_id") for item in raw_ids)
    if len(atom_ids) > MAX_ATOMIC_CONJUNCTION_SIZE:
        raise AtomicSemanticSynthesisError(
            "frozen atomic formula lies outside the 1..4 atom bound"
        )
    if atom_ids != tuple(sorted(atom_ids)) or len(atom_ids) != len(set(atom_ids)):
        raise AtomicSemanticSynthesisError(
            "formula atom IDs must be unique canonical digest order"
        )
    return atom_ids


def evaluate_atomic_formula(
    formula: Mapping[str, Any],
    evidence_by_atom: Mapping[str, Evidence[bool]],
    *,
    provenance_bindings: Mapping[str, AtomicEvidenceBinding],
    selection_scope: str,
) -> Evidence[bool]:
    """Evaluate a frozen conjunction under explicit query/support bindings.

    ``provenance_bindings`` is deliberately independent of the evidence
    objects: every child observation must name the frozen atom, query pixels,
    query description, and scorer protocol in that exact canonical order.
    """

    scope = _selection_scope(selection_scope)
    if scope == CALIBRATED_SELECTION_SCOPE:
        raise AtomicSemanticSynthesisError(
            "calibrated-semantic evaluation is disabled until a typed calibration "
            "artifact and interval rule are independently cold-validated"
        )
    atom_ids = _formula_atom_ids(formula)
    if not isinstance(evidence_by_atom, Mapping) or set(evidence_by_atom) != set(
        atom_ids
    ):
        raise AtomicSemanticSynthesisError(
            "evidence keys must exactly equal frozen formula atom IDs"
        )
    if not isinstance(provenance_bindings, Mapping) or set(
        provenance_bindings
    ) != set(atom_ids):
        raise AtomicSemanticSynthesisError(
            "provenance binding keys must exactly equal frozen formula atom IDs"
        )
    evidence = tuple(evidence_by_atom[atom_id] for atom_id in atom_ids)
    if not all(isinstance(item, Evidence) for item in evidence):
        raise TypeError("formula evidence values must be Evidence[bool]")
    bindings = tuple(provenance_bindings[atom_id] for atom_id in atom_ids)
    if not all(isinstance(item, AtomicEvidenceBinding) for item in bindings):
        raise TypeError(
            "formula provenance bindings must be AtomicEvidenceBinding values"
        )
    shared_contexts = {
        (
            binding.panel_digest,
            binding.panel_description_digest,
            binding.scorer_protocol_digest,
            binding.run_commitment_digest,
            binding.scorer_producer,
            binding.scorer_version,
            binding.scorer_method,
            binding.scorer_run_id,
            binding.scorer_receipt_digest,
            binding.scorer_output_digest,
            binding.scorer_call_digest,
            binding.scorer_call_ordinal,
            binding.observation_scope,
            binding.calibration_digest,
        )
        for binding in bindings
    }
    if len(shared_contexts) != 1:
        raise AtomicSemanticSynthesisError(
            "formula provenance bindings must share one frozen panel, "
            "description, and scorer context"
        )
    for atom_id, item, binding in zip(atom_ids, evidence, bindings, strict=True):
        if binding.atom_id != atom_id:
            raise AtomicSemanticSynthesisError(
                "provenance binding atom_id differs from its mapping key"
            )
        _validate_evidence_provenance_binding(
            item.provenance, binding, f"formula atom {atom_id} evidence"
        )
        TruthEvidenceRecord.from_evidence(item)
        if binding.observation_scope != scope:
            raise AtomicSemanticSynthesisError(
                "formula evidence authorization differs from selection_scope"
            )
        if (
            item.disposition is Disposition.CERTIFIED_ABSENT
            and scope != CALIBRATED_SELECTION_SCOPE
        ):
            raise AtomicSemanticSynthesisError(
                "operational formula evidence cannot claim certified absence"
            )
        if item.is_operational_nonmatch and scope != OPERATIONAL_SELECTION_SCOPE:
            raise AtomicSemanticSynthesisError(
                "operational nonmatch cannot enter calibrated semantic evaluation"
            )
    provenance = Provenance.composed(
        producer="atomic-semantic-synthesis",
        version="1",
        method="positive-conjunction",
        parents=tuple(item.provenance for item in evidence),
        details=(("formula_digest", canonical_digest(dict(formula))),),
    )
    errors = [item for item in evidence if item.disposition is Disposition.ERROR]
    if errors:
        first = errors[0]
        return Evidence.error(
            provenance,
            first.error_type or "AtomicChildError",
            first.reason or "atomic child failed",
        )
    operational_nonmatches = [
        item for item in evidence if item.is_operational_nonmatch
    ]
    if operational_nonmatches:
        first = operational_nonmatches[0]
        reason = first.reason or "one conjunct operationally did not match"
        prefix = "operational nonmatch (uncalibrated): "
        if reason.startswith(prefix):
            reason = reason[len(prefix) :]
        return Evidence.operational_nonmatch(
            provenance,
            reason,
            first.uncertainty,
        )
    absent = [
        item
        for item in evidence
        if item.disposition is Disposition.CERTIFIED_ABSENT
    ]
    if absent:
        return Evidence.certified_absent(
            provenance,
            "conjunct certified absent: "
            + (absent[0].certificate or "unspecified certificate"),
        )
    if any(item.disposition is Disposition.INDETERMINATE for item in evidence):
        return Evidence.indeterminate(
            provenance, "one or more atomic conjuncts are indeterminate"
        )
    return Evidence.present(True, provenance)


def cold_decode_and_replay_atomic_selection(
    value: Mapping[str, Any],
    *,
    expected_archive_digest: str,
) -> tuple[
    tuple[str, TruthEvidenceRecord | OperationalNonmatchRecord], ...
]:
    """Decode and replay an archive only under an external digest pin."""

    expected = _digest(expected_archive_digest, "expected_archive_digest")
    archive = AtomicSelectionArchive.from_data(value)
    if archive.archive_digest != expected:
        raise AtomicSemanticSynthesisError(
            "decoded selection archive differs from expected_archive_digest"
        )
    return archive.replay_support()


def cold_decode_and_recompute_no_exact_separator(
    value: Mapping[str, Any],
    *,
    expected_diagnostic_digest: str,
    matrix: AtomicSupportMatrix,
    support_labels: Mapping[str, bool],
    max_atoms: int,
    selection_scope: str,
) -> NoExactSeparatorDiagnostics:
    """Authenticate and recompute a persisted no-separator result.

    A self-hash is not evidence that the claimed reason or coverage follows
    from a matrix.  This boundary therefore requires the independently pinned
    matrix/labels/search scope and reruns the exact analysis.
    """

    expected_digest = _digest(
        expected_diagnostic_digest, "expected_diagnostic_digest"
    )
    if not isinstance(matrix, AtomicSupportMatrix):
        raise TypeError("matrix must be an AtomicSupportMatrix")
    decoded = NoExactSeparatorDiagnostics.from_data(value)
    if decoded.diagnostic_digest != expected_digest:
        raise AtomicSemanticSynthesisError(
            "decoded diagnostic differs from expected_diagnostic_digest"
        )
    labels = _canonical_labels(matrix.panel_ids, support_labels)
    recomputed = _selection_or_diagnostics(
        matrix,
        labels,
        max_atoms,
        _selection_scope(selection_scope),
    )
    if isinstance(recomputed, _Selection):
        raise AtomicSemanticSynthesisError(
            "diagnostic claims no separator where exact synthesis succeeds"
        )
    if recomputed != decoded:
        raise AtomicSemanticSynthesisError(
            "decoded diagnostic differs from exact recomputation"
        )
    return decoded


__all__ = [
    "ATOMIC_AFFIRMATIVE_SURFACE_POLICY_SCHEMA",
    "ATOMIC_SELECTION_ARCHIVE_SCHEMA",
    "ATOMIC_EVIDENCE_BINDING_SCHEMA",
    "ATOMIC_SOFT_PREDICATE_SCHEMA",
    "ATOMIC_SUPPORT_CELL_SCHEMA",
    "ATOMIC_SUPPORT_MATRIX_SCHEMA",
    "MAX_ATOMIC_CONJUNCTION_SIZE",
    "NO_SEPARATOR_DIAGNOSTICS_SCHEMA",
    "PANEL_DESCRIPTION_BINDING_SCHEMA",
    "OPERATIONAL_SELECTION_SCOPE",
    "CALIBRATED_SELECTION_SCOPE",
    "AtomEligibilityDiagnostic",
    "AtomicEvidenceBinding",
    "AtomicSelectionArchive",
    "AtomicSemanticSynthesisError",
    "AtomicSoftPredicate",
    "AtomicSupportCell",
    "AtomicSupportMatrix",
    "NoExactSeparatorDiagnostics",
    "NoExactSeparatorError",
    "OperationalNonmatchRecord",
    "PanelDescriptionBinding",
    "atomic_affirmative_surface_policy_data",
    "atomic_affirmative_surface_policy_description",
    "cold_decode_and_replay_atomic_selection",
    "cold_decode_and_recompute_no_exact_separator",
    "evaluate_atomic_formula",
    "synthesize_atomic_conjunction",
    "validate_atomic_affirmative_surface",
]

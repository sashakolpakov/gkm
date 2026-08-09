"""Closed Python semantics for direct whole-panel prose predicates.

Vision supplies repeatable observations of affirmative prose atoms.  This
module never executes that prose.  It binds the proposed observation
instrument, records a non-scientific repeatability diagnostic, enumerates only
positive conjunctions, and evaluates them deterministically.

This scaffold does not implement a calibration authority or a sealed observer
receipt boundary.  Therefore neither repeated ``present`` nor repeated
``mismatch`` is scientific evidence: both project to ``indeterminate``.
Transport or observer failures project to ``error``.  A future calibrated
implementation must introduce and verify a typed receipt boundary rather than
turning an arbitrary caller-supplied digest into scientific authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_bongard_soft_cues import (
    ObjectBongardSoftCue,
    object_bongard_soft_cue_grammar_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

PANEL_SOFT_ATOM_SCHEMA = "gkm.bongard-panel-soft-atom.v2"
PANEL_SOFT_ATOM_TEXT_GRAMMAR_ID = (
    "bongard.panel-soft-atom/lexically-filtered-visible-text-v1"
)
PANEL_SOFT_VOCABULARY_SCHEMA = "gkm.bongard-panel-soft-vocabulary.v1"
PANEL_SOFT_OBSERVER_CONTRACT_SCHEMA = (
    "gkm.bongard-panel-soft-observer-contract.v1"
)
PANEL_SOFT_OBSERVATION_CELL_SCHEMA = (
    "gkm.bongard-panel-soft-observation-cell.v1"
)
PANEL_SOFT_OBSERVATION_TABLE_SCHEMA = (
    "gkm.bongard-panel-soft-observation-table.v1"
)
PANEL_SOFT_FORMULA_SCHEMA = "gkm.bongard-panel-soft-formula.v1"
PANEL_SOFT_VERSION_SPACE_SCHEMA = "gkm.bongard-panel-soft-version-space.v1"
PANEL_SOFT_ENGINEERING_VERSION_SPACE_SCHEMA = (
    "gkm.bongard-panel-soft-engineering-version-space.v1"
)
PANEL_SOFT_ENGINEERING_PREDICATE_PAIR_SCHEMA = (
    "gkm.bongard-panel-soft-engineering-predicate-pair.v1"
)
PANEL_SOFT_ENGINEERING_QUERY_DECISION_SCHEMA = (
    "gkm.bongard-panel-soft-engineering-query-decision.v1"
)
PANEL_SOFT_ALGORITHM_ID = (
    "bongard.panel-soft-predicate/diagnostic-consensus-no-calibration-v1"
)
PANEL_SOFT_ENGINEERING_ALGORITHM_ID = (
    "bongard.panel-soft-predicate/engineering-only-positive-consensus-v1"
)

PANEL_SOFT_ORIENTATIONS = ("side0_positive", "side1_positive")
PANEL_SOFT_RAW_VERDICTS = ("present", "mismatch", "indeterminate", "error")
PANEL_SOFT_MAX_ATOMS = 16
PANEL_SOFT_MAX_WITNESSES = 4
PANEL_SOFT_MAX_CONJUNCTION = 4
PANEL_SOFT_SUPPORTS_PER_SIDE = 6

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ATOM_ID = re.compile(r"atom_[0-9]{4}\Z")
_PANEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}\Z")
_PANEL_SOFT_TEXT_SHAPE = re.compile(r"[A-Za-z][A-Za-z ,.'-]{7,239}\Z")
_PANEL_SOFT_PROMPT_CONTROL = re.compile(
    r"\b(?:"
    r"ignore|disregard|forget|override|bypass|obey|follow|pretend|assume|"
    r"inspect|evaluate|consider|read|return|output|answer|respond|response|"
    r"emit|write|say|choose|select|classify|rate|score|assign|"
    r"instruction|prompt|system|developer|assistant|user|model|tool|code|"
    r"python|lean|schema|json|alias|criterion|verdict|"
    r"present|mismatch|indeterminate|error|previous|always"
    r")s?\b",
    re.IGNORECASE,
)
_PANEL_SOFT_COVERT_NEGATION = re.compile(
    r"(?:"
    r"\b(?:avoid|avoids|avoided|avoiding|devoid|empty|blank|bare|"
    r"unmarked|undecorated|unfilled|unable|incapable|"
    r"fail|fails|failed|failing|instead|rather|only|solely|merely|zero)\b|"
    r"\b[A-Za-z]+(?:-?less)\b|"
    r"\b[A-Za-z]+(?:[- ]free)\b|"
    r"\bfree[ ]+of\b|"
    r"\b(?:non|un|dis)-?(?:"
    r"connected|closed|filled|decorated|marked|curved|rounded|straight|"
    r"symmetric|regular|touching|overlapping|intersecting|broken"
    r")\b|"
    r"\b[A-Za-z]+n't\b"
    r")",
    re.IGNORECASE,
)


class PanelSoftPredicateError(ValueError):
    """A panel atom, observation, formula, or version space is not canonical."""


class PanelSoftAtomTextRejected(PanelSoftPredicateError):
    """One prose row failed the closed panel-specific lexical filter."""


class _EngineeringOnlyEnum:
    """Machine-readable limits shared by uncalibrated engineering enums."""

    @property
    def engineering_only(self) -> bool:
        return True

    @property
    def uncalibrated(self) -> bool:
        return True

    @property
    def scientific_evidence(self) -> bool:
        return False

    @property
    def benchmark_authoritative(self) -> bool:
        return False


class PanelSoftOperationalConsensus(_EngineeringOnlyEnum, str, Enum):
    """Non-scientific summary of two same-instrument raw verdicts.

    This diagnostic is useful for an engineering drill, but it is deliberately
    not a :class:`Disposition` and cannot support a scientific survivor.
    """

    REPEATED_PRESENT = "repeated_present"
    REPEATED_MISMATCH = "repeated_mismatch"
    REPEATED_INDETERMINATE = "repeated_indeterminate"
    DISAGREEMENT = "disagreement"
    ERROR = "error"


class PanelSoftOperationalFormulaResult(_EngineeringOnlyEnum, str, Enum):
    """Engineering-only, uncalibrated result of a positive conjunction."""

    MATCH = "match"
    NONMATCH = "nonmatch"
    INDETERMINATE = "indeterminate"
    ERROR = "error"


class PanelSoftEngineeringQueryOutcome(_EngineeringOnlyEnum, str, Enum):
    """Engineering-only query outcome; never a scientific benchmark label."""

    SIDE0 = "side0"
    SIDE1 = "side1"
    ABSTAIN = "abstain"
    ERROR = "error"


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "prose_is_observed_not_executed": True,
        "arbitrary_code_allowed": False,
        "negation_allowed": False,
        "formula_negation_operator_allowed": False,
        "polarity_flip_allowed": False,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_checker_optional": True,
        "lean_affects_identity_or_decision": False,
    }


def _engineering_only_data() -> dict[str, object]:
    """Closed warning labels for every operational artifact in this module."""

    return {
        "engineering_only": True,
        "uncalibrated": True,
        "scientific_evidence": False,
        "benchmark_authoritative": False,
        "freeze_before_query_chronology_verified": False,
        "sealed_observer_receipts_verified": False,
    }


def panel_soft_predicate_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def panel_soft_atom_text_grammar_digest() -> str:
    """Identity of the panel-specific syntactic prose filter."""

    return canonical_digest(
        {
            "schema": "gkm.bongard-panel-soft-atom-text-grammar.v1",
            "grammar_id": PANEL_SOFT_ATOM_TEXT_GRAMMAR_ID,
            "predicate_source_digest": panel_soft_predicate_source_digest(),
            "upstream_soft_cue_grammar_digest": (
                object_bongard_soft_cue_grammar_digest()
            ),
            "allowed_text_pattern": _PANEL_SOFT_TEXT_SHAPE.pattern,
            "prompt_control_pattern": _PANEL_SOFT_PROMPT_CONTROL.pattern,
            "covert_negation_pattern": _PANEL_SOFT_COVERT_NEGATION.pattern,
            "lexical_prompt_control_filter_applied": True,
            "forbidden_negative_construction_filter_applied": True,
            "open_prose_instruction_safety_proved": False,
            "open_prose_semantic_positivity_proved": False,
            "observer_requires_inert_structured_rendering": True,
            **_authority_data(),
        }
    )


def validate_panel_soft_atom_text(value: object) -> str:
    """Apply exact lexical controls; this does not prove open-prose semantics."""

    if (
        not isinstance(value, str)
        or value != value.strip()
        or _PANEL_SOFT_TEXT_SHAPE.fullmatch(value) is None
        or "  " in value
        or _PANEL_SOFT_PROMPT_CONTROL.search(value) is not None
        or _PANEL_SOFT_COVERT_NEGATION.search(value) is not None
    ):
        raise PanelSoftAtomTextRejected(
            "panel atom text violates the panel-specific lexical filter"
        )
    return value


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise PanelSoftPredicateError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise PanelSoftPredicateError(f"{label} must be a lowercase SHA-256")
    return value


def _panel_id(value: object) -> str:
    if not isinstance(value, str) or _PANEL_ID.fullmatch(value) is None:
        raise PanelSoftPredicateError("panel ID differs")
    return value


def _raw_verdict(value: object) -> str:
    if value not in PANEL_SOFT_RAW_VERDICTS:
        raise PanelSoftPredicateError("raw panel verdict differs")
    return value  # type: ignore[return-value]


def _disposition(value: object) -> Disposition:
    try:
        return Disposition(value)
    except (TypeError, ValueError) as exc:
        raise PanelSoftPredicateError("panel disposition differs") from exc


def _consensus(value: object) -> PanelSoftOperationalConsensus:
    try:
        return PanelSoftOperationalConsensus(value)
    except (TypeError, ValueError) as exc:
        raise PanelSoftPredicateError("operational consensus differs") from exc


def _operational_formula_result(value: object) -> PanelSoftOperationalFormulaResult:
    try:
        return PanelSoftOperationalFormulaResult(value)
    except (TypeError, ValueError) as exc:
        raise PanelSoftPredicateError("operational formula result differs") from exc


def _engineering_query_outcome(value: object) -> PanelSoftEngineeringQueryOutcome:
    try:
        return PanelSoftEngineeringQueryOutcome(value)
    except (TypeError, ValueError) as exc:
        raise PanelSoftPredicateError("engineering query outcome differs") from exc


def _operational_consensus(
    raw_verdicts: tuple[str, str],
) -> PanelSoftOperationalConsensus:
    if "error" in raw_verdicts:
        return PanelSoftOperationalConsensus.ERROR
    if raw_verdicts == ("present", "present"):
        return PanelSoftOperationalConsensus.REPEATED_PRESENT
    if raw_verdicts == ("mismatch", "mismatch"):
        return PanelSoftOperationalConsensus.REPEATED_MISMATCH
    if raw_verdicts == ("indeterminate", "indeterminate"):
        return PanelSoftOperationalConsensus.REPEATED_INDETERMINATE
    return PanelSoftOperationalConsensus.DISAGREEMENT


def _project_repeats(raw_verdicts: tuple[str, str]) -> Disposition:
    """Project raw repeats without claiming unimplemented calibration."""

    if "error" in raw_verdicts:
        return Disposition.ERROR
    return Disposition.INDETERMINATE


def _and_dispositions(values: Sequence[Disposition]) -> Disposition:
    row = tuple(values)
    if not row:
        raise PanelSoftPredicateError("a positive conjunction cannot be empty")
    if Disposition.ERROR in row:
        return Disposition.ERROR
    if Disposition.CERTIFIED_ABSENT in row:
        return Disposition.CERTIFIED_ABSENT
    if all(item is Disposition.PRESENT for item in row):
        return Disposition.PRESENT
    return Disposition.INDETERMINATE


def _atom_content(value: "PanelSoftAtom") -> dict[str, object]:
    return {
        "schema": PANEL_SOFT_ATOM_SCHEMA,
        "text_grammar_id": PANEL_SOFT_ATOM_TEXT_GRAMMAR_ID,
        "text_grammar_digest": panel_soft_atom_text_grammar_digest(),
        "atom_id": value.atom_id,
        "orientation": value.orientation,
        "phrase": value.phrase.to_data(),
        "witnesses": [item.to_data() for item in value.witnesses],
        "witness_order": "cue-digest-ascending",
        "proposer_artifact_digest": value.proposer_artifact_digest,
        "scope": "complete_panel",
        "lexical_prompt_control_filter_applied": True,
        "forbidden_negative_construction_filter_applied": True,
        "open_prose_instruction_safety_proved": False,
        "open_prose_semantic_positivity_proved": False,
        **_authority_data(),
    }


@dataclass(frozen=True, order=True, slots=True)
class PanelSoftAtom:
    """One affirmative prose identity scoped to the complete raw panel."""

    atom_id: str
    orientation: str
    phrase: ObjectBongardSoftCue
    witnesses: tuple[ObjectBongardSoftCue, ...]
    proposer_artifact_digest: str
    atom_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.atom_id, str) or _ATOM_ID.fullmatch(self.atom_id) is None:
            raise PanelSoftPredicateError("panel atom ID differs")
        if self.orientation not in PANEL_SOFT_ORIENTATIONS:
            raise PanelSoftPredicateError("panel atom orientation differs")
        if not isinstance(self.phrase, ObjectBongardSoftCue):
            raise TypeError("panel atom phrase has the wrong type")
        validate_panel_soft_atom_text(self.phrase.text)
        if (
            type(self.witnesses) is not tuple
            or not 1 <= len(self.witnesses) <= PANEL_SOFT_MAX_WITNESSES
            or any(not isinstance(item, ObjectBongardSoftCue) for item in self.witnesses)
            or len({item.cue_digest for item in self.witnesses}) != len(self.witnesses)
            or tuple(item.cue_digest for item in self.witnesses)
            != tuple(sorted(item.cue_digest for item in self.witnesses))
            or self.phrase.cue_digest in {item.cue_digest for item in self.witnesses}
        ):
            raise PanelSoftPredicateError("panel atom witness bundle differs")
        for item in self.witnesses:
            validate_panel_soft_atom_text(item.text)
        _digest(self.proposer_artifact_digest, "proposer artifact digest")
        _digest(self.atom_digest, "panel atom digest")
        if self.atom_digest != canonical_digest(_atom_content(self)):
            raise PanelSoftPredicateError("panel atom digest differs")

    @classmethod
    def create(
        cls,
        *,
        atom_id: str,
        orientation: str,
        phrase: str | ObjectBongardSoftCue,
        witnesses: Sequence[str | ObjectBongardSoftCue],
        proposer_artifact_digest: str,
    ) -> "PanelSoftAtom":
        cue = phrase if isinstance(phrase, ObjectBongardSoftCue) else ObjectBongardSoftCue.create(phrase)
        witness_row = tuple(
            sorted(
                (
                    item
                    if isinstance(item, ObjectBongardSoftCue)
                    else ObjectBongardSoftCue.create(item)
                    for item in witnesses
                ),
                key=lambda item: item.cue_digest,
            )
        )
        values = {
            "atom_id": atom_id,
            "orientation": orientation,
            "phrase": cue,
            "witnesses": witness_row,
            "proposer_artifact_digest": proposer_artifact_digest,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, atom_digest=canonical_digest(_atom_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_atom_content(self), "atom_digest": self.atom_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelSoftAtom":
        raw = _fields(
            value,
            {
                "schema", "text_grammar_id", "text_grammar_digest", "atom_id",
                "orientation", "phrase", "witnesses", "witness_order",
                "proposer_artifact_digest",
                "scope", "lexical_prompt_control_filter_applied",
                "forbidden_negative_construction_filter_applied",
                "open_prose_instruction_safety_proved",
                "open_prose_semantic_positivity_proved", *_authority_data(),
                "atom_digest",
            },
            "panel atom",
        )
        if (
            raw["schema"] != PANEL_SOFT_ATOM_SCHEMA
            or raw["text_grammar_id"] != PANEL_SOFT_ATOM_TEXT_GRAMMAR_ID
            or raw["text_grammar_digest"] != panel_soft_atom_text_grammar_digest()
            or raw["scope"] != "complete_panel"
            or raw["witness_order"] != "cue-digest-ascending"
            or raw["lexical_prompt_control_filter_applied"] is not True
            or raw["forbidden_negative_construction_filter_applied"] is not True
            or raw["open_prose_instruction_safety_proved"] is not False
            or raw["open_prose_semantic_positivity_proved"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["witnesses"], list)
        ):
            raise PanelSoftPredicateError("panel atom policy differs")
        result = cls(
            raw["atom_id"],
            raw["orientation"],
            ObjectBongardSoftCue.from_data(raw["phrase"]),
            tuple(ObjectBongardSoftCue.from_data(item) for item in raw["witnesses"]),
            raw["proposer_artifact_digest"],
            raw["atom_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelSoftPredicateError("panel atom is not canonical")
        return result


def _vocabulary_content(value: "PanelSoftVocabulary") -> dict[str, object]:
    return {
        "schema": PANEL_SOFT_VOCABULARY_SCHEMA,
        "proposer_artifact_digest": value.proposer_artifact_digest,
        "atoms": [item.to_data() for item in value.atoms],
        "orientation_order": list(PANEL_SOFT_ORIENTATIONS),
        "complete_atom_vector_required": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelSoftVocabulary:
    proposer_artifact_digest: str
    atoms: tuple[PanelSoftAtom, ...]
    vocabulary_digest: str

    def __post_init__(self) -> None:
        _digest(self.proposer_artifact_digest, "vocabulary proposer artifact digest")
        if (
            type(self.atoms) is not tuple
            or not 2 <= len(self.atoms) <= PANEL_SOFT_MAX_ATOMS
            or any(not isinstance(item, PanelSoftAtom) for item in self.atoms)
            or tuple(item.atom_id for item in self.atoms)
            != tuple(f"atom_{index:04d}" for index in range(len(self.atoms)))
            or any(item.proposer_artifact_digest != self.proposer_artifact_digest for item in self.atoms)
            or len({item.atom_digest for item in self.atoms}) != len(self.atoms)
            or tuple(item.orientation for item in self.atoms)
            != tuple(sorted((item.orientation for item in self.atoms), key=PANEL_SOFT_ORIENTATIONS.index))
            or {item.orientation for item in self.atoms} != set(PANEL_SOFT_ORIENTATIONS)
        ):
            raise PanelSoftPredicateError("panel atom vocabulary differs")
        _digest(self.vocabulary_digest, "panel vocabulary digest")
        if self.vocabulary_digest != canonical_digest(_vocabulary_content(self)):
            raise PanelSoftPredicateError("panel vocabulary digest differs")

    @classmethod
    def create(cls, atoms: Sequence[PanelSoftAtom]) -> "PanelSoftVocabulary":
        row = tuple(atoms)
        if not row:
            raise PanelSoftPredicateError("panel vocabulary cannot be empty")
        proposer = row[0].proposer_artifact_digest
        values = {"proposer_artifact_digest": proposer, "atoms": row}
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, vocabulary_digest=canonical_digest(_vocabulary_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_vocabulary_content(self), "vocabulary_digest": self.vocabulary_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelSoftVocabulary":
        raw = _fields(
            value,
            {
                "schema", "proposer_artifact_digest", "atoms", "orientation_order",
                "complete_atom_vector_required", *_authority_data(), "vocabulary_digest",
            },
            "panel vocabulary",
        )
        if (
            raw["schema"] != PANEL_SOFT_VOCABULARY_SCHEMA
            or raw["orientation_order"] != list(PANEL_SOFT_ORIENTATIONS)
            or raw["complete_atom_vector_required"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["atoms"], list)
        ):
            raise PanelSoftPredicateError("panel vocabulary policy differs")
        result = cls(
            raw["proposer_artifact_digest"],
            tuple(PanelSoftAtom.from_data(item) for item in raw["atoms"]),
            raw["vocabulary_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelSoftPredicateError("panel vocabulary is not canonical")
        return result


def _contract_content(value: "PanelSoftObserverContract") -> dict[str, object]:
    return {
        "schema": PANEL_SOFT_OBSERVER_CONTRACT_SCHEMA,
        "protocol_digest": value.protocol_digest,
        "model_runtime_digest": value.model_runtime_digest,
        "prompt_digest": value.prompt_digest,
        "output_schema_digest": value.output_schema_digest,
        "presentation_digest": value.presentation_digest,
        "vocabulary_digest": value.vocabulary_digest,
        "repeat_count": 2,
        "panels_per_call": 1,
        "complete_ordered_atom_vector_per_call": True,
        "support_query_protocol_identical": True,
        "same_model_repeats_are_independent_evidence": False,
        "scientific_calibration_receipt_boundary_implemented": False,
        "scientific_present_enabled": False,
        "scientific_absence_enabled": False,
        "repeated_present_projects_to": Disposition.INDETERMINATE.value,
        "repeated_mismatch_projects_to": Disposition.INDETERMINATE.value,
        "operational_consensus_is_scientific_evidence": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelSoftObserverContract:
    protocol_digest: str
    model_runtime_digest: str
    prompt_digest: str
    output_schema_digest: str
    presentation_digest: str
    vocabulary_digest: str
    contract_digest: str

    def __post_init__(self) -> None:
        for label, item in (
            ("observer protocol digest", self.protocol_digest),
            ("observer model runtime digest", self.model_runtime_digest),
            ("observer prompt digest", self.prompt_digest),
            ("observer output schema digest", self.output_schema_digest),
            ("observer presentation digest", self.presentation_digest),
            ("observer vocabulary digest", self.vocabulary_digest),
        ):
            _digest(item, label)
        _digest(self.contract_digest, "observer contract digest")
        if self.contract_digest != canonical_digest(_contract_content(self)):
            raise PanelSoftPredicateError("observer contract digest differs")

    @classmethod
    def create(
        cls,
        *,
        protocol_digest: str,
        model_runtime_digest: str,
        prompt_digest: str,
        output_schema_digest: str,
        presentation_digest: str,
        vocabulary_digest: str,
    ) -> "PanelSoftObserverContract":
        values = {
            "protocol_digest": protocol_digest,
            "model_runtime_digest": model_runtime_digest,
            "prompt_digest": prompt_digest,
            "output_schema_digest": output_schema_digest,
            "presentation_digest": presentation_digest,
            "vocabulary_digest": vocabulary_digest,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, contract_digest=canonical_digest(_contract_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_contract_content(self), "contract_digest": self.contract_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelSoftObserverContract":
        raw = _fields(
            value,
            {
                "schema", "protocol_digest", "model_runtime_digest", "prompt_digest",
                "output_schema_digest", "presentation_digest", "vocabulary_digest",
                "repeat_count", "panels_per_call", "complete_ordered_atom_vector_per_call",
                "support_query_protocol_identical", "same_model_repeats_are_independent_evidence",
                "scientific_calibration_receipt_boundary_implemented",
                "scientific_present_enabled", "scientific_absence_enabled",
                "repeated_present_projects_to", "repeated_mismatch_projects_to",
                "operational_consensus_is_scientific_evidence",
                *_authority_data(), "contract_digest",
            },
            "panel observer contract",
        )
        if (
            raw["schema"] != PANEL_SOFT_OBSERVER_CONTRACT_SCHEMA
            or raw["repeat_count"] != 2
            or raw["panels_per_call"] != 1
            or raw["complete_ordered_atom_vector_per_call"] is not True
            or raw["support_query_protocol_identical"] is not True
            or raw["same_model_repeats_are_independent_evidence"] is not False
            or raw["scientific_calibration_receipt_boundary_implemented"] is not False
            or raw["scientific_present_enabled"] is not False
            or raw["scientific_absence_enabled"] is not False
            or raw["repeated_present_projects_to"] != Disposition.INDETERMINATE.value
            or raw["repeated_mismatch_projects_to"] != Disposition.INDETERMINATE.value
            or raw["operational_consensus_is_scientific_evidence"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise PanelSoftPredicateError("panel observer contract policy differs")
        result = cls(
            raw["protocol_digest"], raw["model_runtime_digest"], raw["prompt_digest"],
            raw["output_schema_digest"], raw["presentation_digest"],
            raw["vocabulary_digest"], raw["contract_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelSoftPredicateError("panel observer contract is not canonical")
        return result


def _cell_content(value: "PanelSoftObservationCell") -> dict[str, object]:
    return {
        "schema": PANEL_SOFT_OBSERVATION_CELL_SCHEMA,
        "panel_id": value.panel_id,
        "panel_png_digest": value.panel_png_digest,
        "atom_digest": value.atom_digest,
        "contract_digest": value.contract_digest,
        "raw_verdicts": list(value.raw_verdicts),
        "operational_consensus": value.operational_consensus.value,
        "operational_consensus_is_scientific_evidence": False,
        "disposition": value.disposition.value,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelSoftObservationCell:
    panel_id: str
    panel_png_digest: str
    atom_digest: str
    contract_digest: str
    raw_verdicts: tuple[str, str]
    operational_consensus: PanelSoftOperationalConsensus
    disposition: Disposition
    cell_digest: str

    def __post_init__(self) -> None:
        _panel_id(self.panel_id)
        for label, item in (
            ("panel PNG digest", self.panel_png_digest),
            ("observation atom digest", self.atom_digest),
            ("observation contract digest", self.contract_digest),
        ):
            _digest(item, label)
        if type(self.raw_verdicts) is not tuple or len(self.raw_verdicts) != 2:
            raise PanelSoftPredicateError("raw repeat verdicts differ")
        verdicts = tuple(_raw_verdict(item) for item in self.raw_verdicts)
        if self.operational_consensus is not _operational_consensus(verdicts):
            raise PanelSoftPredicateError("cell operational consensus differs")
        if self.disposition is not _project_repeats(verdicts):
            raise PanelSoftPredicateError("cell disposition projection differs")
        _digest(self.cell_digest, "panel observation cell digest")
        if self.cell_digest != canonical_digest(_cell_content(self)):
            raise PanelSoftPredicateError("panel observation cell digest differs")

    @classmethod
    def create(
        cls,
        *,
        panel_id: str,
        panel_png_digest: str,
        atom_digest: str,
        contract: PanelSoftObserverContract,
        raw_verdicts: tuple[str, str],
    ) -> "PanelSoftObservationCell":
        if not isinstance(contract, PanelSoftObserverContract):
            raise TypeError("contract must be a panel observer contract")
        verdicts = tuple(_raw_verdict(item) for item in raw_verdicts)
        if len(verdicts) != 2:
            raise PanelSoftPredicateError("raw repeat verdicts differ")
        values = {
            "panel_id": _panel_id(panel_id),
            "panel_png_digest": _digest(panel_png_digest, "panel PNG digest"),
            "atom_digest": _digest(atom_digest, "observation atom digest"),
            "contract_digest": contract.contract_digest,
            "raw_verdicts": verdicts,
            "operational_consensus": _operational_consensus(verdicts),
            "disposition": _project_repeats(verdicts),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, cell_digest=canonical_digest(_cell_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_cell_content(self), "cell_digest": self.cell_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelSoftObservationCell":
        raw = _fields(
            value,
            {
                "schema", "panel_id", "panel_png_digest", "atom_digest",
                "contract_digest", "raw_verdicts", "operational_consensus",
                "operational_consensus_is_scientific_evidence",
                "disposition", *_authority_data(), "cell_digest",
            },
            "panel observation cell",
        )
        if (
            raw["schema"] != PANEL_SOFT_OBSERVATION_CELL_SCHEMA
            or any(raw[key] != item for key, item in _authority_data().items())
            or raw["operational_consensus_is_scientific_evidence"] is not False
            or not isinstance(raw["raw_verdicts"], list)
        ):
            raise PanelSoftPredicateError("panel observation cell policy differs")
        result = cls(
            raw["panel_id"], raw["panel_png_digest"], raw["atom_digest"],
            raw["contract_digest"], tuple(raw["raw_verdicts"]),
            _consensus(raw["operational_consensus"]),
            _disposition(raw["disposition"]),
            raw["cell_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelSoftPredicateError("panel observation cell is not canonical")
        return result


def _table_content(value: "PanelSoftObservationTable") -> dict[str, object]:
    return {
        "schema": PANEL_SOFT_OBSERVATION_TABLE_SCHEMA,
        "vocabulary": value.vocabulary.to_data(),
        "contract": value.contract.to_data(),
        "panel_ids": list(value.panel_ids),
        "panel_png_digests": list(value.panel_png_digests),
        "cells": [item.to_data() for item in value.cells],
        "cell_order": "panel-major-then-complete-vocabulary-order",
        "one_panel_per_observer_call": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelSoftObservationTable:
    vocabulary: PanelSoftVocabulary
    contract: PanelSoftObserverContract
    panel_ids: tuple[str, ...]
    panel_png_digests: tuple[str, ...]
    cells: tuple[PanelSoftObservationCell, ...]
    table_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.vocabulary, PanelSoftVocabulary):
            raise TypeError("table vocabulary has the wrong type")
        if not isinstance(self.contract, PanelSoftObserverContract):
            raise TypeError("table contract has the wrong type")
        if self.contract.vocabulary_digest != self.vocabulary.vocabulary_digest:
            raise PanelSoftPredicateError("table observer vocabulary differs")
        if (
            type(self.panel_ids) is not tuple
            or not self.panel_ids
            or any(_panel_id(item) != item for item in self.panel_ids)
            or len(set(self.panel_ids)) != len(self.panel_ids)
            or type(self.panel_png_digests) is not tuple
            or len(self.panel_png_digests) != len(self.panel_ids)
        ):
            raise PanelSoftPredicateError("table panel inventory differs")
        for item in self.panel_png_digests:
            _digest(item, "table panel PNG digest")
        if type(self.cells) is not tuple or any(
            not isinstance(item, PanelSoftObservationCell) for item in self.cells
        ):
            raise PanelSoftPredicateError("table cells have the wrong type")
        expected_pairs = tuple(
            (panel_id, panel_digest, atom.atom_digest)
            for panel_id, panel_digest in zip(
                self.panel_ids, self.panel_png_digests, strict=True
            )
            for atom in self.vocabulary.atoms
        )
        actual_pairs = tuple(
            (item.panel_id, item.panel_png_digest, item.atom_digest) for item in self.cells
        )
        if (
            actual_pairs != expected_pairs
            or any(item.contract_digest != self.contract.contract_digest for item in self.cells)
        ):
            raise PanelSoftPredicateError("table is not a complete ordered panel-atom matrix")
        _digest(self.table_digest, "panel observation table digest")
        if self.table_digest != canonical_digest(_table_content(self)):
            raise PanelSoftPredicateError("panel observation table digest differs")

    @classmethod
    def create(
        cls,
        *,
        vocabulary: PanelSoftVocabulary,
        contract: PanelSoftObserverContract,
        panels: Sequence[tuple[str, str]],
        raw_verdict_rows: Sequence[Sequence[tuple[str, str]]],
    ) -> "PanelSoftObservationTable":
        panel_row = tuple(panels)
        verdict_matrix = tuple(tuple(row) for row in raw_verdict_rows)
        if (
            len(verdict_matrix) != len(panel_row)
            or any(len(row) != len(vocabulary.atoms) for row in verdict_matrix)
        ):
            raise PanelSoftPredicateError("raw verdict matrix shape differs")
        panel_ids = tuple(_panel_id(item[0]) for item in panel_row)
        panel_digests = tuple(_digest(item[1], "panel PNG digest") for item in panel_row)
        cells = tuple(
            PanelSoftObservationCell.create(
                panel_id=panel_id,
                panel_png_digest=panel_digest,
                atom_digest=atom.atom_digest,
                contract=contract,
                raw_verdicts=verdicts,
            )
            for (panel_id, panel_digest), verdict_row in zip(
                panel_row, verdict_matrix, strict=True
            )
            for atom, verdicts in zip(vocabulary.atoms, verdict_row, strict=True)
        )
        values = {
            "vocabulary": vocabulary,
            "contract": contract,
            "panel_ids": panel_ids,
            "panel_png_digests": panel_digests,
            "cells": cells,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, table_digest=canonical_digest(_table_content(provisional)))

    @property
    def cell_by_panel_and_atom(self) -> dict[tuple[str, str], PanelSoftObservationCell]:
        return {(item.panel_id, item.atom_digest): item for item in self.cells}

    def to_data(self) -> dict[str, object]:
        return {**_table_content(self), "table_digest": self.table_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelSoftObservationTable":
        raw = _fields(
            value,
            {
                "schema", "vocabulary", "contract", "panel_ids",
                "panel_png_digests", "cells", "cell_order",
                "one_panel_per_observer_call", *_authority_data(), "table_digest",
            },
            "panel observation table",
        )
        if (
            raw["schema"] != PANEL_SOFT_OBSERVATION_TABLE_SCHEMA
            or raw["cell_order"] != "panel-major-then-complete-vocabulary-order"
            or raw["one_panel_per_observer_call"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["panel_ids"], list)
            or not isinstance(raw["panel_png_digests"], list)
            or not isinstance(raw["cells"], list)
        ):
            raise PanelSoftPredicateError("panel observation table policy differs")
        result = cls(
            PanelSoftVocabulary.from_data(raw["vocabulary"]),
            PanelSoftObserverContract.from_data(raw["contract"]),
            tuple(raw["panel_ids"]), tuple(raw["panel_png_digests"]),
            tuple(PanelSoftObservationCell.from_data(item) for item in raw["cells"]),
            raw["table_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelSoftPredicateError("panel observation table is not canonical")
        return result


def _formula_content(value: "PanelSoftFormula") -> dict[str, object]:
    return {
        "schema": PANEL_SOFT_FORMULA_SCHEMA,
        "vocabulary_digest": value.vocabulary_digest,
        "observer_contract_digest": value.observer_contract_digest,
        "orientation": value.orientation,
        "atom_digests": list(value.atom_digests),
        "operator": "all_of",
        "formula_language": "positive-atoms-and-conjunction-only",
        **_authority_data(),
    }


@dataclass(frozen=True, order=True, slots=True)
class PanelSoftFormula:
    vocabulary_digest: str
    observer_contract_digest: str
    orientation: str
    atom_digests: tuple[str, ...]
    formula_digest: str

    def __post_init__(self) -> None:
        _digest(self.vocabulary_digest, "formula vocabulary digest")
        _digest(self.observer_contract_digest, "formula observer contract digest")
        if self.orientation not in PANEL_SOFT_ORIENTATIONS:
            raise PanelSoftPredicateError("formula orientation differs")
        if (
            type(self.atom_digests) is not tuple
            or not 1 <= len(self.atom_digests) <= PANEL_SOFT_MAX_CONJUNCTION
            or len(set(self.atom_digests)) != len(self.atom_digests)
        ):
            raise PanelSoftPredicateError("formula atom conjunction differs")
        for item in self.atom_digests:
            _digest(item, "formula atom digest")
        _digest(self.formula_digest, "panel formula digest")
        if self.formula_digest != canonical_digest(_formula_content(self)):
            raise PanelSoftPredicateError("panel formula digest differs")

    @classmethod
    def create(
        cls,
        vocabulary: PanelSoftVocabulary,
        contract: PanelSoftObserverContract,
        orientation: str,
        atom_digests: Sequence[str],
    ) -> "PanelSoftFormula":
        if not isinstance(vocabulary, PanelSoftVocabulary):
            raise TypeError("vocabulary must be a panel soft vocabulary")
        if not isinstance(contract, PanelSoftObserverContract):
            raise TypeError("contract must be a panel soft observer contract")
        if contract.vocabulary_digest != vocabulary.vocabulary_digest:
            raise PanelSoftPredicateError("formula observer vocabulary differs")
        row = tuple(atom_digests)
        ordered = tuple(
            item.atom_digest for item in vocabulary.atoms if item.atom_digest in set(row)
        )
        by_digest = {item.atom_digest: item for item in vocabulary.atoms}
        if (
            row != ordered
            or any(item not in by_digest for item in row)
            or any(by_digest[item].orientation != orientation for item in row)
        ):
            raise PanelSoftPredicateError("formula atoms are not native ordered orientation atoms")
        values = {
            "vocabulary_digest": vocabulary.vocabulary_digest,
            "observer_contract_digest": contract.contract_digest,
            "orientation": orientation,
            "atom_digests": row,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, formula_digest=canonical_digest(_formula_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_formula_content(self), "formula_digest": self.formula_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelSoftFormula":
        raw = _fields(
            value,
            {
                "schema", "vocabulary_digest", "orientation", "atom_digests",
                "observer_contract_digest",
                "operator", "formula_language", *_authority_data(), "formula_digest",
            },
            "panel soft formula",
        )
        if (
            raw["schema"] != PANEL_SOFT_FORMULA_SCHEMA
            or raw["operator"] != "all_of"
            or raw["formula_language"] != "positive-atoms-and-conjunction-only"
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["atom_digests"], list)
        ):
            raise PanelSoftPredicateError("panel formula policy differs")
        result = cls(
            raw["vocabulary_digest"], raw["observer_contract_digest"], raw["orientation"],
            tuple(raw["atom_digests"]), raw["formula_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelSoftPredicateError("panel formula is not canonical")
        return result


def evaluate_panel_soft_formula(
    formula: PanelSoftFormula,
    table: PanelSoftObservationTable,
    panel_id: str,
) -> Disposition:
    """Interpret one frozen positive conjunction on one observed panel."""

    if not isinstance(formula, PanelSoftFormula):
        raise TypeError("formula must be a panel soft formula")
    if not isinstance(table, PanelSoftObservationTable):
        raise TypeError("table must be a panel soft observation table")
    panel = _panel_id(panel_id)
    if formula.vocabulary_digest != table.vocabulary.vocabulary_digest:
        raise PanelSoftPredicateError("formula/table vocabulary differs")
    if formula.observer_contract_digest != table.contract.contract_digest:
        raise PanelSoftPredicateError("formula/table observer contract differs")
    atoms = {item.atom_digest: item for item in table.vocabulary.atoms}
    if any(
        item not in atoms or atoms[item].orientation != formula.orientation
        for item in formula.atom_digests
    ):
        raise PanelSoftPredicateError("formula atoms differ from table vocabulary")
    cells = table.cell_by_panel_and_atom
    try:
        dispositions = tuple(cells[(panel, atom)].disposition for atom in formula.atom_digests)
    except KeyError as exc:
        raise PanelSoftPredicateError("panel is absent from the complete observation table") from exc
    return _and_dispositions(dispositions)


def enumerate_panel_soft_formulas(
    vocabulary: PanelSoftVocabulary,
    contract: PanelSoftObserverContract,
) -> tuple[PanelSoftFormula, ...]:
    """Enumerate the closed native-orientation positive language."""

    if not isinstance(vocabulary, PanelSoftVocabulary):
        raise TypeError("vocabulary must be a panel soft vocabulary")
    if not isinstance(contract, PanelSoftObserverContract):
        raise TypeError("contract must be a panel soft observer contract")
    if contract.vocabulary_digest != vocabulary.vocabulary_digest:
        raise PanelSoftPredicateError("formula inventory observer vocabulary differs")
    result: list[PanelSoftFormula] = []
    for orientation in PANEL_SOFT_ORIENTATIONS:
        atoms = tuple(
            item.atom_digest for item in vocabulary.atoms if item.orientation == orientation
        )
        for size in range(1, min(PANEL_SOFT_MAX_CONJUNCTION, len(atoms)) + 1):
            result.extend(
                PanelSoftFormula.create(vocabulary, contract, orientation, row)
                for row in combinations(atoms, size)
            )
    return tuple(result)


def _version_space_gap_kind(
    support_table: PanelSoftObservationTable,
    survivor_formula_digests: tuple[str, ...],
) -> str | None:
    if survivor_formula_digests:
        return None
    if any(item.disposition is Disposition.ERROR for item in support_table.cells):
        return "observer_error_gap"
    if any(
        item.operational_consensus is PanelSoftOperationalConsensus.DISAGREEMENT
        for item in support_table.cells
    ):
        return "observer_disagreement_gap"
    if any(
        item.operational_consensus
        is PanelSoftOperationalConsensus.REPEATED_INDETERMINATE
        for item in support_table.cells
    ):
        return "observer_indeterminate_gap"
    return "calibration_authority_gap"


def _version_space_content(value: "PanelSoftVersionSpace") -> dict[str, object]:
    return {
        "schema": PANEL_SOFT_VERSION_SPACE_SCHEMA,
        "algorithm_id": PANEL_SOFT_ALGORITHM_ID,
        "support_table": value.support_table.to_data(),
        "side0_panel_ids": list(value.side0_panel_ids),
        "side1_panel_ids": list(value.side1_panel_ids),
        "ordered_formulas": [item.to_data() for item in value.ordered_formulas],
        "survivor_formula_digests": list(value.survivor_formula_digests),
        "gap_kind": value.gap_kind,
        "support_rule": "six-native-present-and-six-opposite-certified-absent",
        "failed_or_indeterminate_is_absence": False,
        "operational_consensus_can_create_survivors": False,
        "scientific_calibration_receipt_boundary_implemented": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelSoftVersionSpace:
    support_table: PanelSoftObservationTable
    side0_panel_ids: tuple[str, ...]
    side1_panel_ids: tuple[str, ...]
    ordered_formulas: tuple[PanelSoftFormula, ...]
    survivor_formula_digests: tuple[str, ...]
    gap_kind: str | None
    version_space_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.support_table, PanelSoftObservationTable):
            raise TypeError("version-space support table has the wrong type")
        for row in (self.side0_panel_ids, self.side1_panel_ids):
            if (
                type(row) is not tuple
                or len(row) != PANEL_SOFT_SUPPORTS_PER_SIDE
                or len(set(row)) != len(row)
                or any(_panel_id(item) != item for item in row)
            ):
                raise PanelSoftPredicateError("version-space support side differs")
        if (
            set(self.side0_panel_ids).intersection(self.side1_panel_ids)
            or self.support_table.panel_ids != self.side0_panel_ids + self.side1_panel_ids
        ):
            raise PanelSoftPredicateError("version-space support order differs")
        expected_formulas = enumerate_panel_soft_formulas(
            self.support_table.vocabulary, self.support_table.contract
        )
        if self.ordered_formulas != expected_formulas:
            raise PanelSoftPredicateError("version-space formula inventory differs")
        expected_survivors = tuple(
            formula.formula_digest
            for formula in expected_formulas
            if _formula_supports_exactly(
                formula, self.support_table, self.side0_panel_ids, self.side1_panel_ids
            )
        )
        if self.survivor_formula_digests != expected_survivors:
            raise PanelSoftPredicateError("version-space survivors differ")
        expected_gap = _version_space_gap_kind(
            self.support_table, expected_survivors
        )
        if self.gap_kind != expected_gap:
            raise PanelSoftPredicateError("version-space gap kind differs")
        _digest(self.version_space_digest, "panel version-space digest")
        if self.version_space_digest != canonical_digest(_version_space_content(self)):
            raise PanelSoftPredicateError("panel version-space digest differs")

    @classmethod
    def create(
        cls,
        support_table: PanelSoftObservationTable,
        side0_panel_ids: Sequence[str],
        side1_panel_ids: Sequence[str],
    ) -> "PanelSoftVersionSpace":
        side0 = tuple(side0_panel_ids)
        side1 = tuple(side1_panel_ids)
        formulas = enumerate_panel_soft_formulas(
            support_table.vocabulary, support_table.contract
        )
        survivors = tuple(
            formula.formula_digest
            for formula in formulas
            if _formula_supports_exactly(formula, support_table, side0, side1)
        )
        gap = _version_space_gap_kind(support_table, survivors)
        values = {
            "support_table": support_table,
            "side0_panel_ids": side0,
            "side1_panel_ids": side1,
            "ordered_formulas": formulas,
            "survivor_formula_digests": survivors,
            "gap_kind": gap,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, version_space_digest=canonical_digest(_version_space_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_version_space_content(self), "version_space_digest": self.version_space_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelSoftVersionSpace":
        raw = _fields(
            value,
            {
                "schema", "algorithm_id", "support_table", "side0_panel_ids",
                "side1_panel_ids", "ordered_formulas", "survivor_formula_digests",
                "gap_kind", "support_rule", "failed_or_indeterminate_is_absence",
                "operational_consensus_can_create_survivors",
                "scientific_calibration_receipt_boundary_implemented",
                *_authority_data(), "version_space_digest",
            },
            "panel soft version space",
        )
        if (
            raw["schema"] != PANEL_SOFT_VERSION_SPACE_SCHEMA
            or raw["algorithm_id"] != PANEL_SOFT_ALGORITHM_ID
            or raw["support_rule"] != "six-native-present-and-six-opposite-certified-absent"
            or raw["failed_or_indeterminate_is_absence"] is not False
            or raw["operational_consensus_can_create_survivors"] is not False
            or raw["scientific_calibration_receipt_boundary_implemented"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["side0_panel_ids"], list)
            or not isinstance(raw["side1_panel_ids"], list)
            or not isinstance(raw["ordered_formulas"], list)
            or not isinstance(raw["survivor_formula_digests"], list)
        ):
            raise PanelSoftPredicateError("panel version-space policy differs")
        result = cls(
            PanelSoftObservationTable.from_data(raw["support_table"]),
            tuple(raw["side0_panel_ids"]), tuple(raw["side1_panel_ids"]),
            tuple(PanelSoftFormula.from_data(item) for item in raw["ordered_formulas"]),
            tuple(raw["survivor_formula_digests"]), raw["gap_kind"],
            raw["version_space_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelSoftPredicateError("panel version space is not canonical")
        return result


def _formula_supports_exactly(
    formula: PanelSoftFormula,
    table: PanelSoftObservationTable,
    side0: Sequence[str],
    side1: Sequence[str],
) -> bool:
    native, opposite = (
        (tuple(side0), tuple(side1))
        if formula.orientation == "side0_positive"
        else (tuple(side1), tuple(side0))
    )
    return all(
        evaluate_panel_soft_formula(formula, table, panel) is Disposition.PRESENT
        for panel in native
    ) and all(
        evaluate_panel_soft_formula(formula, table, panel)
        is Disposition.CERTIFIED_ABSENT
        for panel in opposite
    )


def evaluate_panel_soft_formula_operationally(
    formula: PanelSoftFormula,
    table: PanelSoftObservationTable,
    panel_id: str,
) -> PanelSoftOperationalFormulaResult:
    """Evaluate ``all_of`` using repeatability diagnostics only.

    This is a deliberately separate engineering path.  It does not inspect or
    alter :class:`Disposition`, and its result is uncalibrated and cannot be
    cited as scientific evidence or an authoritative benchmark decision.
    """

    if not isinstance(formula, PanelSoftFormula):
        raise TypeError("formula must be a panel soft formula")
    if not isinstance(table, PanelSoftObservationTable):
        raise TypeError("table must be a panel soft observation table")
    panel = _panel_id(panel_id)
    if formula.vocabulary_digest != table.vocabulary.vocabulary_digest:
        raise PanelSoftPredicateError("formula/table vocabulary differs")
    if formula.observer_contract_digest != table.contract.contract_digest:
        raise PanelSoftPredicateError("formula/table observer contract differs")
    atoms = {item.atom_digest: item for item in table.vocabulary.atoms}
    if any(
        item not in atoms or atoms[item].orientation != formula.orientation
        for item in formula.atom_digests
    ):
        raise PanelSoftPredicateError("formula atoms differ from table vocabulary")
    cells = table.cell_by_panel_and_atom
    try:
        consensuses = tuple(
            cells[(panel, atom)].operational_consensus
            for atom in formula.atom_digests
        )
    except KeyError as exc:
        raise PanelSoftPredicateError(
            "panel is absent from the complete observation table"
        ) from exc
    if PanelSoftOperationalConsensus.ERROR in consensuses:
        return PanelSoftOperationalFormulaResult.ERROR
    if any(
        item
        in {
            PanelSoftOperationalConsensus.DISAGREEMENT,
            PanelSoftOperationalConsensus.REPEATED_INDETERMINATE,
        }
        for item in consensuses
    ):
        return PanelSoftOperationalFormulaResult.INDETERMINATE
    if all(
        item is PanelSoftOperationalConsensus.REPEATED_PRESENT
        for item in consensuses
    ):
        return PanelSoftOperationalFormulaResult.MATCH
    if PanelSoftOperationalConsensus.REPEATED_MISMATCH in consensuses:
        return PanelSoftOperationalFormulaResult.NONMATCH
    raise PanelSoftPredicateError("operational formula consensus is not closed")


def _engineering_formula_supports_exactly(
    formula: PanelSoftFormula,
    table: PanelSoftObservationTable,
    side0: Sequence[str],
    side1: Sequence[str],
) -> bool:
    native, contrast = (
        (tuple(side0), tuple(side1))
        if formula.orientation == "side0_positive"
        else (tuple(side1), tuple(side0))
    )
    return all(
        evaluate_panel_soft_formula_operationally(formula, table, panel)
        is PanelSoftOperationalFormulaResult.MATCH
        for panel in native
    ) and all(
        evaluate_panel_soft_formula_operationally(formula, table, panel)
        is PanelSoftOperationalFormulaResult.NONMATCH
        for panel in contrast
    )


def _engineering_version_space_content(
    value: "PanelSoftEngineeringVersionSpace",
) -> dict[str, object]:
    return {
        "schema": PANEL_SOFT_ENGINEERING_VERSION_SPACE_SCHEMA,
        "algorithm_id": PANEL_SOFT_ENGINEERING_ALGORITHM_ID,
        "support_table": value.support_table.to_data(),
        "side0_panel_ids": list(value.side0_panel_ids),
        "side1_panel_ids": list(value.side1_panel_ids),
        "ordered_formulas": [item.to_data() for item in value.ordered_formulas],
        "survivor_formula_digests": list(value.survivor_formula_digests),
        "support_rule": "match-all-native-and-nonmatch-all-contrast",
        "result_semantics": "positive-all-of-over-operational-consensus",
        "uses_scientific_dispositions": False,
        "failed_or_indeterminate_is_nonmatch": False,
        "negation_rescue_allowed": False,
        **_engineering_only_data(),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelSoftEngineeringVersionSpace:
    """Uncalibrated support-consistent formulas for engineering drills only."""

    support_table: PanelSoftObservationTable
    side0_panel_ids: tuple[str, ...]
    side1_panel_ids: tuple[str, ...]
    ordered_formulas: tuple[PanelSoftFormula, ...]
    survivor_formula_digests: tuple[str, ...]
    engineering_version_space_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.support_table, PanelSoftObservationTable):
            raise TypeError("engineering version-space support table has the wrong type")
        for row in (self.side0_panel_ids, self.side1_panel_ids):
            if (
                type(row) is not tuple
                or len(row) != PANEL_SOFT_SUPPORTS_PER_SIDE
                or len(set(row)) != len(row)
                or any(_panel_id(item) != item for item in row)
            ):
                raise PanelSoftPredicateError(
                    "engineering version-space support side differs"
                )
        if (
            set(self.side0_panel_ids).intersection(self.side1_panel_ids)
            or self.support_table.panel_ids
            != self.side0_panel_ids + self.side1_panel_ids
        ):
            raise PanelSoftPredicateError(
                "engineering version-space support order differs"
            )
        expected_formulas = enumerate_panel_soft_formulas(
            self.support_table.vocabulary, self.support_table.contract
        )
        if self.ordered_formulas != expected_formulas:
            raise PanelSoftPredicateError(
                "engineering version-space formula inventory differs"
            )
        expected_survivors = tuple(
            formula.formula_digest
            for formula in expected_formulas
            if _engineering_formula_supports_exactly(
                formula,
                self.support_table,
                self.side0_panel_ids,
                self.side1_panel_ids,
            )
        )
        if self.survivor_formula_digests != expected_survivors:
            raise PanelSoftPredicateError(
                "engineering version-space survivors differ"
            )
        _digest(
            self.engineering_version_space_digest,
            "engineering version-space digest",
        )
        if self.engineering_version_space_digest != canonical_digest(
            _engineering_version_space_content(self)
        ):
            raise PanelSoftPredicateError(
                "engineering version-space digest differs"
            )

    @classmethod
    def create(
        cls,
        support_table: PanelSoftObservationTable,
        side0_panel_ids: Sequence[str],
        side1_panel_ids: Sequence[str],
    ) -> "PanelSoftEngineeringVersionSpace":
        if not isinstance(support_table, PanelSoftObservationTable):
            raise TypeError("support table must be a panel soft observation table")
        side0 = tuple(side0_panel_ids)
        side1 = tuple(side1_panel_ids)
        formulas = enumerate_panel_soft_formulas(
            support_table.vocabulary, support_table.contract
        )
        survivors = tuple(
            formula.formula_digest
            for formula in formulas
            if _engineering_formula_supports_exactly(
                formula, support_table, side0, side1
            )
        )
        values = {
            "support_table": support_table,
            "side0_panel_ids": side0,
            "side1_panel_ids": side1,
            "ordered_formulas": formulas,
            "survivor_formula_digests": survivors,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            engineering_version_space_digest=canonical_digest(
                _engineering_version_space_content(provisional)
            ),
        )

    @property
    def survivor_formulas(self) -> tuple[PanelSoftFormula, ...]:
        survivors = set(self.survivor_formula_digests)
        return tuple(
            item for item in self.ordered_formulas if item.formula_digest in survivors
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_engineering_version_space_content(self),
            "engineering_version_space_digest": self.engineering_version_space_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "PanelSoftEngineeringVersionSpace":
        raw = _fields(
            value,
            {
                "schema", "algorithm_id", "support_table", "side0_panel_ids",
                "side1_panel_ids", "ordered_formulas", "survivor_formula_digests",
                "support_rule", "result_semantics", "uses_scientific_dispositions",
                "failed_or_indeterminate_is_nonmatch", "negation_rescue_allowed",
                *_engineering_only_data(), *_authority_data(),
                "engineering_version_space_digest",
            },
            "panel soft engineering version space",
        )
        if (
            raw["schema"] != PANEL_SOFT_ENGINEERING_VERSION_SPACE_SCHEMA
            or raw["algorithm_id"] != PANEL_SOFT_ENGINEERING_ALGORITHM_ID
            or raw["support_rule"]
            != "match-all-native-and-nonmatch-all-contrast"
            or raw["result_semantics"]
            != "positive-all-of-over-operational-consensus"
            or raw["uses_scientific_dispositions"] is not False
            or raw["failed_or_indeterminate_is_nonmatch"] is not False
            or raw["negation_rescue_allowed"] is not False
            or any(raw[key] != item for key, item in _engineering_only_data().items())
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["side0_panel_ids"], list)
            or not isinstance(raw["side1_panel_ids"], list)
            or not isinstance(raw["ordered_formulas"], list)
            or not isinstance(raw["survivor_formula_digests"], list)
        ):
            raise PanelSoftPredicateError(
                "panel soft engineering version-space policy differs"
            )
        result = cls(
            PanelSoftObservationTable.from_data(raw["support_table"]),
            tuple(raw["side0_panel_ids"]),
            tuple(raw["side1_panel_ids"]),
            tuple(PanelSoftFormula.from_data(item) for item in raw["ordered_formulas"]),
            tuple(raw["survivor_formula_digests"]),
            raw["engineering_version_space_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelSoftPredicateError(
                "panel soft engineering version space is not canonical"
            )
        return result


def _selected_engineering_formula(
    version_space: PanelSoftEngineeringVersionSpace,
    orientation: str,
) -> PanelSoftFormula:
    candidates = tuple(
        formula
        for formula in version_space.survivor_formulas
        if formula.orientation == orientation
    )
    if not candidates:
        raise PanelSoftPredicateError(
            f"engineering version space has no {orientation} survivor"
        )
    return min(candidates, key=lambda item: (len(item.atom_digests), item.formula_digest))


def _engineering_predicate_pair_content(
    value: "PanelSoftEngineeringPredicatePair",
) -> dict[str, object]:
    return {
        "schema": PANEL_SOFT_ENGINEERING_PREDICATE_PAIR_SCHEMA,
        "engineering_version_space": value.engineering_version_space.to_data(),
        "engineering_version_space_digest": (
            value.engineering_version_space.engineering_version_space_digest
        ),
        "side0_formula_digest": value.side0_formula_digest,
        "side1_formula_digest": value.side1_formula_digest,
        "selected_formula_count_by_orientation": {
            "side0_positive": 1,
            "side1_positive": 1,
        },
        "selection_rule": "minimum-atom-count-then-formula-digest",
        "selected_formulas_must_be_support_survivors": True,
        "uses_scientific_dispositions": False,
        **_engineering_only_data(),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelSoftEngineeringPredicatePair:
    """Deterministically selected, uncalibrated formula pair for a drill."""

    engineering_version_space: PanelSoftEngineeringVersionSpace
    side0_formula_digest: str
    side1_formula_digest: str
    predicate_pair_digest: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.engineering_version_space, PanelSoftEngineeringVersionSpace
        ):
            raise TypeError("predicate pair version space has the wrong type")
        _digest(self.side0_formula_digest, "selected side0 formula digest")
        _digest(self.side1_formula_digest, "selected side1 formula digest")
        expected_side0 = _selected_engineering_formula(
            self.engineering_version_space, "side0_positive"
        )
        expected_side1 = _selected_engineering_formula(
            self.engineering_version_space, "side1_positive"
        )
        if (
            self.side0_formula_digest != expected_side0.formula_digest
            or self.side1_formula_digest != expected_side1.formula_digest
        ):
            raise PanelSoftPredicateError(
                "engineering predicate pair selection differs"
            )
        _digest(self.predicate_pair_digest, "engineering predicate pair digest")
        if self.predicate_pair_digest != canonical_digest(
            _engineering_predicate_pair_content(self)
        ):
            raise PanelSoftPredicateError("engineering predicate pair digest differs")

    @classmethod
    def create(
        cls,
        engineering_version_space: PanelSoftEngineeringVersionSpace,
    ) -> "PanelSoftEngineeringPredicatePair":
        if not isinstance(
            engineering_version_space, PanelSoftEngineeringVersionSpace
        ):
            raise TypeError("version space must be an engineering version space")
        side0 = _selected_engineering_formula(
            engineering_version_space, "side0_positive"
        )
        side1 = _selected_engineering_formula(
            engineering_version_space, "side1_positive"
        )
        values = {
            "engineering_version_space": engineering_version_space,
            "side0_formula_digest": side0.formula_digest,
            "side1_formula_digest": side1.formula_digest,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            predicate_pair_digest=canonical_digest(
                _engineering_predicate_pair_content(provisional)
            ),
        )

    @property
    def selected_formulas(self) -> tuple[PanelSoftFormula, PanelSoftFormula]:
        by_digest = {
            item.formula_digest: item
            for item in self.engineering_version_space.ordered_formulas
        }
        return (
            by_digest[self.side0_formula_digest],
            by_digest[self.side1_formula_digest],
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_engineering_predicate_pair_content(self),
            "predicate_pair_digest": self.predicate_pair_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "PanelSoftEngineeringPredicatePair":
        raw = _fields(
            value,
            {
                "schema", "engineering_version_space",
                "engineering_version_space_digest", "side0_formula_digest",
                "side1_formula_digest", "selected_formula_count_by_orientation",
                "selection_rule", "selected_formulas_must_be_support_survivors",
                "uses_scientific_dispositions", *_engineering_only_data(),
                *_authority_data(), "predicate_pair_digest",
            },
            "panel soft engineering predicate pair",
        )
        version_space = PanelSoftEngineeringVersionSpace.from_data(
            raw["engineering_version_space"]
        )
        if (
            raw["schema"] != PANEL_SOFT_ENGINEERING_PREDICATE_PAIR_SCHEMA
            or raw["engineering_version_space_digest"]
            != version_space.engineering_version_space_digest
            or raw["selected_formula_count_by_orientation"]
            != {"side0_positive": 1, "side1_positive": 1}
            or raw["selection_rule"]
            != "minimum-atom-count-then-formula-digest"
            or raw["selected_formulas_must_be_support_survivors"] is not True
            or raw["uses_scientific_dispositions"] is not False
            or any(raw[key] != item for key, item in _engineering_only_data().items())
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise PanelSoftPredicateError(
                "panel soft engineering predicate-pair policy differs"
            )
        result = cls(
            version_space,
            raw["side0_formula_digest"],
            raw["side1_formula_digest"],
            raw["predicate_pair_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelSoftPredicateError(
                "panel soft engineering predicate pair is not canonical"
            )
        return result


def _derive_engineering_query_outcome(
    side0_result: PanelSoftOperationalFormulaResult,
    side1_result: PanelSoftOperationalFormulaResult,
) -> PanelSoftEngineeringQueryOutcome:
    if PanelSoftOperationalFormulaResult.ERROR in (side0_result, side1_result):
        return PanelSoftEngineeringQueryOutcome.ERROR
    if (
        side0_result is PanelSoftOperationalFormulaResult.MATCH
        and side1_result is PanelSoftOperationalFormulaResult.NONMATCH
    ):
        return PanelSoftEngineeringQueryOutcome.SIDE0
    if (
        side1_result is PanelSoftOperationalFormulaResult.MATCH
        and side0_result is PanelSoftOperationalFormulaResult.NONMATCH
    ):
        return PanelSoftEngineeringQueryOutcome.SIDE1
    return PanelSoftEngineeringQueryOutcome.ABSTAIN


def _engineering_query_decision_content(
    value: "PanelSoftEngineeringQueryDecision",
) -> dict[str, object]:
    return {
        "schema": PANEL_SOFT_ENGINEERING_QUERY_DECISION_SCHEMA,
        "predicate_pair": value.predicate_pair.to_data(),
        "predicate_pair_digest": value.predicate_pair.predicate_pair_digest,
        "engineering_version_space_digest": (
            value.predicate_pair.engineering_version_space.engineering_version_space_digest
        ),
        "query_table": value.query_table.to_data(),
        "query_table_digest": value.query_table.table_digest,
        "panel_id": value.panel_id,
        "panel_png_digest": value.panel_png_digest,
        "side0_formula_result": value.side0_formula_result.value,
        "side1_formula_result": value.side1_formula_result.value,
        "outcome": value.outcome.value,
        "decision_rule": "one-match-and-other-nonmatch",
        "nonmatch_alone_predicts_the_opposite": False,
        "conflict_or_indeterminate_policy": "abstain",
        "observer_error_policy": "error",
        "uses_scientific_dispositions": False,
        **_engineering_only_data(),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelSoftEngineeringQueryDecision:
    """Uncalibrated two-formula query decision for engineering drills only."""

    predicate_pair: PanelSoftEngineeringPredicatePair
    query_table: PanelSoftObservationTable
    panel_id: str
    panel_png_digest: str
    side0_formula_result: PanelSoftOperationalFormulaResult
    side1_formula_result: PanelSoftOperationalFormulaResult
    outcome: PanelSoftEngineeringQueryOutcome
    decision_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.predicate_pair, PanelSoftEngineeringPredicatePair):
            raise TypeError("query predicate pair has the wrong type")
        if not isinstance(self.query_table, PanelSoftObservationTable):
            raise TypeError("query observation table has the wrong type")
        panel = _panel_id(self.panel_id)
        _digest(self.panel_png_digest, "query panel PNG digest")
        support_table = self.predicate_pair.engineering_version_space.support_table
        if (
            self.query_table.vocabulary.vocabulary_digest
            != support_table.vocabulary.vocabulary_digest
            or self.query_table.contract.contract_digest
            != support_table.contract.contract_digest
        ):
            raise PanelSoftPredicateError(
                "engineering query uses a different observer instrument"
            )
        try:
            panel_index = self.query_table.panel_ids.index(panel)
        except ValueError as exc:
            raise PanelSoftPredicateError(
                "query panel is absent from the complete observation table"
            ) from exc
        if self.panel_png_digest != self.query_table.panel_png_digests[panel_index]:
            raise PanelSoftPredicateError("query panel PNG digest differs")
        side0_formula, side1_formula = self.predicate_pair.selected_formulas
        expected_side0 = evaluate_panel_soft_formula_operationally(
            side0_formula, self.query_table, panel
        )
        expected_side1 = evaluate_panel_soft_formula_operationally(
            side1_formula, self.query_table, panel
        )
        if (
            self.side0_formula_result is not expected_side0
            or self.side1_formula_result is not expected_side1
        ):
            raise PanelSoftPredicateError(
                "engineering query formula result differs"
            )
        expected_outcome = _derive_engineering_query_outcome(
            expected_side0, expected_side1
        )
        if self.outcome is not expected_outcome:
            raise PanelSoftPredicateError("engineering query outcome differs")
        _digest(self.decision_digest, "engineering query decision digest")
        if self.decision_digest != canonical_digest(
            _engineering_query_decision_content(self)
        ):
            raise PanelSoftPredicateError("engineering query decision digest differs")

    @classmethod
    def create(
        cls,
        predicate_pair: PanelSoftEngineeringPredicatePair,
        query_table: PanelSoftObservationTable,
        panel_id: str,
    ) -> "PanelSoftEngineeringQueryDecision":
        if not isinstance(predicate_pair, PanelSoftEngineeringPredicatePair):
            raise TypeError("predicate pair must be an engineering predicate pair")
        if not isinstance(query_table, PanelSoftObservationTable):
            raise TypeError("query table must be a panel soft observation table")
        panel = _panel_id(panel_id)
        try:
            panel_index = query_table.panel_ids.index(panel)
        except ValueError as exc:
            raise PanelSoftPredicateError(
                "query panel is absent from the complete observation table"
            ) from exc
        side0_formula, side1_formula = predicate_pair.selected_formulas
        side0_result = evaluate_panel_soft_formula_operationally(
            side0_formula, query_table, panel
        )
        side1_result = evaluate_panel_soft_formula_operationally(
            side1_formula, query_table, panel
        )
        values = {
            "predicate_pair": predicate_pair,
            "query_table": query_table,
            "panel_id": panel,
            "panel_png_digest": query_table.panel_png_digests[panel_index],
            "side0_formula_result": side0_result,
            "side1_formula_result": side1_result,
            "outcome": _derive_engineering_query_outcome(
                side0_result, side1_result
            ),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            decision_digest=canonical_digest(
                _engineering_query_decision_content(provisional)
            ),
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_engineering_query_decision_content(self),
            "decision_digest": self.decision_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "PanelSoftEngineeringQueryDecision":
        raw = _fields(
            value,
            {
                "schema", "predicate_pair", "predicate_pair_digest",
                "engineering_version_space_digest", "query_table",
                "query_table_digest", "panel_id", "panel_png_digest",
                "side0_formula_result", "side1_formula_result", "outcome",
                "decision_rule", "nonmatch_alone_predicts_the_opposite",
                "conflict_or_indeterminate_policy", "observer_error_policy",
                "uses_scientific_dispositions", *_engineering_only_data(),
                *_authority_data(), "decision_digest",
            },
            "panel soft engineering query decision",
        )
        predicate_pair = PanelSoftEngineeringPredicatePair.from_data(
            raw["predicate_pair"]
        )
        query_table = PanelSoftObservationTable.from_data(raw["query_table"])
        if (
            raw["schema"] != PANEL_SOFT_ENGINEERING_QUERY_DECISION_SCHEMA
            or raw["predicate_pair_digest"] != predicate_pair.predicate_pair_digest
            or raw["engineering_version_space_digest"]
            != predicate_pair.engineering_version_space.engineering_version_space_digest
            or raw["query_table_digest"] != query_table.table_digest
            or raw["decision_rule"] != "one-match-and-other-nonmatch"
            or raw["nonmatch_alone_predicts_the_opposite"] is not False
            or raw["conflict_or_indeterminate_policy"] != "abstain"
            or raw["observer_error_policy"] != "error"
            or raw["uses_scientific_dispositions"] is not False
            or any(raw[key] != item for key, item in _engineering_only_data().items())
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise PanelSoftPredicateError(
                "panel soft engineering query-decision policy differs"
            )
        result = cls(
            predicate_pair,
            query_table,
            raw["panel_id"],
            raw["panel_png_digest"],
            _operational_formula_result(raw["side0_formula_result"]),
            _operational_formula_result(raw["side1_formula_result"]),
            _engineering_query_outcome(raw["outcome"]),
            raw["decision_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelSoftPredicateError(
                "panel soft engineering query decision is not canonical"
            )
        return result


__all__ = (
    "PANEL_SOFT_ALGORITHM_ID",
    "PANEL_SOFT_ATOM_TEXT_GRAMMAR_ID",
    "PANEL_SOFT_ENGINEERING_ALGORITHM_ID",
    "PANEL_SOFT_MAX_ATOMS",
    "PANEL_SOFT_MAX_CONJUNCTION",
    "PANEL_SOFT_ORIENTATIONS",
    "PanelSoftAtom",
    "PanelSoftAtomTextRejected",
    "PanelSoftEngineeringPredicatePair",
    "PanelSoftEngineeringQueryDecision",
    "PanelSoftEngineeringQueryOutcome",
    "PanelSoftEngineeringVersionSpace",
    "PanelSoftFormula",
    "PanelSoftObservationCell",
    "PanelSoftObservationTable",
    "PanelSoftOperationalConsensus",
    "PanelSoftOperationalFormulaResult",
    "PanelSoftObserverContract",
    "PanelSoftPredicateError",
    "PanelSoftVersionSpace",
    "PanelSoftVocabulary",
    "enumerate_panel_soft_formulas",
    "evaluate_panel_soft_formula",
    "evaluate_panel_soft_formula_operationally",
    "panel_soft_atom_text_grammar_digest",
    "panel_soft_predicate_source_digest",
    "validate_panel_soft_atom_text",
)

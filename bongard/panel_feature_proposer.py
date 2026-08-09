"""Contrastive, typed feature nomination for twelve Bongard support panels.

This is an additive vNext component.  It does not parse prose into executable
semantics: the model emits a closed wire representation which Python converts
directly to :class:`bongard.panel_soft_ontology.PanelFeatureSpec`.  Narration is
archival only.  Native orientation lives in proposal provenance, while the
observer vocabulary contains only globally deduplicated feature specs.

The module deliberately stops at an injectable receipted call boundary.  It
does not know about a particular CLI, journal, model, or image staging helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.panel_soft_ontology import (
    FEATURE_SPEC_SCHEMA,
    FeatureFamily,
    LanguageGapArtifact,
    LanguageGapKind,
    NativeFeatureProposal,
    NativeOrientation,
    NativeProposalProvenance,
    PanelFeatureNarration,
    PanelFeatureSpec,
    PanelSoftOntologyError,
    ReferenceFrame,
    SubjectScope,
    feature_catalog_data,
    feature_catalog_digest,
)
from bongard.transport import validate_codex_strict_output_schema


PANEL_FEATURE_PROPOSER_PROTOCOL_ID = (
    "bongard.panel-feature-proposer/two-neutral-blocks-contrastive-v2"
)
PANEL_FEATURE_PROPOSER_RESULT_SCHEMA = (
    "gkm.bongard-panel-feature-proposer-result.v2"
)
PANEL_FEATURE_PROPOSER_NOMINATION_SCHEMA = (
    "gkm.bongard-panel-feature-nomination.v2"
)
PANEL_FEATURE_PROPOSER_NOMINATION_GAP_SCHEMA = (
    "gkm.bongard-panel-feature-nomination-gap.v2"
)
PANEL_FEATURE_OBSERVER_VOCABULARY_SCHEMA = (
    "gkm.bongard-panel-feature-observer-vocabulary.v1"
)
PANEL_FEATURE_PROPOSER_CALL_RESULT_SCHEMA = (
    "gkm.bongard-panel-feature-proposer-call-result.v1"
)

PANEL_FEATURE_BLOCKS = ("block_a", "block_b")
PANEL_FEATURE_SLOTS_PER_DIRECTION = 4
PANEL_FEATURE_PANELS_PER_BLOCK = 6
PANEL_FEATURE_PRESENTATION_NAMES = tuple(
    f"{block}_panel_{index:03d}.png"
    for block in PANEL_FEATURE_BLOCKS
    for index in range(PANEL_FEATURE_PANELS_PER_BLOCK)
)
PANEL_FEATURE_ESTIMATES = ("supports", "does_not_support", "unclear")
PANEL_FEATURE_NONE = "unset"
PANEL_FEATURE_MIN_NATIVE_SUPPORT = 5
PANEL_FEATURE_MAX_NATIVE_UNCLEAR = 1
PANEL_FEATURE_MIN_CONTRAST_NONSUPPORT = 5
PANEL_FEATURE_MAX_CONTRAST_SUPPORT = 1
PANEL_FEATURE_MAX_CONTRAST_UNCLEAR = 1
PANEL_FEATURE_MIN_MARGIN = 3

_DIGEST_CHARS = frozenset("0123456789abcdef")


class PanelFeatureProposerError(ValueError):
    """A proposer payload, boundary receipt, or canonical result is invalid."""


class PanelFeatureCandidateKind(str, Enum):
    REGISTERED_FEATURE = "registered_feature"
    LANGUAGE_GAP = "language_gap"


class PanelFeatureNominationGapCode(str, Enum):
    SHARED_SALIENCE_REJECTED = "shared_salience_rejected"
    CONTRASTIVE_ADMISSION_REJECTED = "contrastive_admission_rejected"
    INVALID_ARCHIVAL_NARRATION = "invalid_archival_narration"
    DUPLICATE_NATIVE_SPEC = "duplicate_native_spec"
    GLOBAL_SPEC_CONTRADICTION = "global_spec_contradiction"
    MISSING_NATIVE_ORIENTATION = "missing_native_orientation"


_PARAMETER_FIELDS: Mapping[FeatureFamily, tuple[str, ...]] = {
    FeatureFamily.COMPONENT_COUNT: ("count",),
    FeatureFamily.EXACT_SEGMENT_COUNT: ("count",),
    FeatureFamily.STRAIGHT_SEGMENT_COUNT: ("count",),
    FeatureFamily.MARKER_PATTERN: ("primitive", "repetition", "arrangement"),
    FeatureFamily.GESTALT_RESEMBLANCE: ("kind",),
    FeatureFamily.SEGMENT_ORIENTATION: ("orientation", "aggregation"),
    FeatureFamily.CORNER_ANGLE: ("angle_class", "aggregation"),
    FeatureFamily.TURN_PROFILE: ("profile",),
    FeatureFamily.OPEN_TRACE: ("kind",),
    FeatureFamily.CLOSED_LOOP: ("kind",),
    FeatureFamily.POINT_CONTACT: ("kind",),
    FeatureFamily.VISIBLE_GAP: ("kind",),
    FeatureFamily.ENCLOSURE: ("kind",),
    FeatureFamily.SYMMETRY: ("kind",),
    FeatureFamily.SHARED_BOUNDARY_ADJACENCY: ("kind",),
    FeatureFamily.ASPECT_RATIO: ("aspect_class",),
    FeatureFamily.TEXTURE_COMPOSITION: ("composition",),
}

if set(_PARAMETER_FIELDS) != set(FeatureFamily):  # pragma: no cover - import guard
    raise RuntimeError("feature proposer parameter table is incomplete")


def _digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(item not in _DIGEST_CHARS for item in value)
    ):
        raise PanelFeatureProposerError(f"{label} must be a lowercase SHA-256")
    return value


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise PanelFeatureProposerError(f"{label} fields differ")
    return value


def _canonical_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise PanelFeatureProposerError("proposer payload must be a JSON object")
    try:
        result = json.loads(canonical_json(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise PanelFeatureProposerError("proposer payload is not canonical JSON") from exc
    if type(result) is not dict:
        raise PanelFeatureProposerError("proposer payload must be a JSON object")
    return result


def _candidate_field(block: str, slot: int) -> str:
    return f"{block}_candidate_{slot}"


def _estimate_field(block: str, panel_index: int) -> str:
    return f"{block}_panel_{panel_index:03d}_estimate"


def _candidate_fields() -> set[str]:
    return {
        "candidate_kind",
        "feature_family",
        "subject_scope",
        "reference_frame",
        "parameter_a",
        "parameter_b",
        "parameter_c",
        "language_gap_kind",
        "archival_summary",
        "archival_indicator_a",
        "archival_indicator_b",
        *(
            _estimate_field(block, index)
            for block in PANEL_FEATURE_BLOCKS
            for index in range(PANEL_FEATURE_PANELS_PER_BLOCK)
        ),
    }


def panel_feature_proposer_output_schema() -> dict[str, object]:
    """Return the fixed eight-slot strict output schema."""

    candidate_properties: dict[str, object] = {
        "candidate_kind": {
            "type": "string",
            "enum": [item.value for item in PanelFeatureCandidateKind],
        },
        "feature_family": {
            "type": "string",
            "enum": [PANEL_FEATURE_NONE, *(item.value for item in FeatureFamily)],
        },
        "subject_scope": {
            "type": "string",
            "enum": [PANEL_FEATURE_NONE, *(item.value for item in SubjectScope)],
        },
        "reference_frame": {
            "type": "string",
            "enum": [PANEL_FEATURE_NONE, *(item.value for item in ReferenceFrame)],
        },
        # These strings are decoded through the closed family-specific table;
        # no string is interpreted as prose or executable source.
        "parameter_a": {"type": "string"},
        "parameter_b": {"type": "string"},
        "parameter_c": {"type": "string"},
        "language_gap_kind": {
            "type": "string",
            "enum": [PANEL_FEATURE_NONE, *(item.value for item in LanguageGapKind)],
        },
        "archival_summary": {"type": "string"},
        "archival_indicator_a": {"type": "string"},
        "archival_indicator_b": {"type": "string"},
    }
    for block in PANEL_FEATURE_BLOCKS:
        for index in range(PANEL_FEATURE_PANELS_PER_BLOCK):
            candidate_properties[_estimate_field(block, index)] = {
                "type": "string",
                "enum": list(PANEL_FEATURE_ESTIMATES),
            }
    candidate_schema: dict[str, object] = {
        "type": "object",
        "properties": candidate_properties,
        "required": list(candidate_properties),
        "additionalProperties": False,
    }
    properties = {
        _candidate_field(block, slot): candidate_schema
        for block in PANEL_FEATURE_BLOCKS
        for slot in range(PANEL_FEATURE_SLOTS_PER_DIRECTION)
    }
    schema: dict[str, object] = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(schema)
    return schema


def _wire_catalog() -> dict[str, object]:
    feature_catalog = feature_catalog_data()
    catalog_by_family = {
        row["family"]: row for row in feature_catalog["families"]  # type: ignore[index]
    }
    count_membership_rules = feature_catalog["count_membership_rules"]
    return {
        "catalog_digest": feature_catalog_digest(),
        "unused_parameter_token": PANEL_FEATURE_NONE,
        "families": [
            {
                "family": family.value,
                "parameter_positions": list(_PARAMETER_FIELDS[family]),
                "parameter_schema": catalog_by_family[family.value]["parameter_schema"],
                "allowed_scope_frames": catalog_by_family[family.value][
                    "allowed_scope_frames"
                ],
                "count_membership_rule_id": count_membership_rules.get(  # type: ignore[union-attr]
                    family.value, PANEL_FEATURE_NONE
                ),
            }
            for family in sorted(FeatureFamily, key=lambda item: item.value)
        ],
    }


def panel_feature_proposer_prompt() -> str:
    """Symmetric contrastive nomination prompt with neutral block names."""

    catalog = canonical_json(_wire_catalog()).decode("utf-8")
    return (
        "Inspect exactly twelve complete drawings. The six block_a images form one "
        "neutral block and the six block_b images form the other. Work symmetrically: "
        "the block_a slots nominate registered affirmative visual features recurring "
        "in block_a and uncommon in block_b; the block_b slots do the reverse. Inspect "
        "all twelve images for every slot. Prefer a candidate estimated as supported by "
        "at least five images in its native block and explicitly does_not_support on at "
        "least five images in the other block. Use unclear for missing evidence; unclear "
        "does not count as does_not_support, and at most one image per block may be unclear. "
        "Do not encode comparison, absence, a complement, a task label, a class label, "
        "a path, or an image role in the feature fields. Select semantics only from the "
        "closed catalog. exact_segment_count counts every registered segment owner, "
        "whether straight or curved; straight_segment_count counts only visibly straight "
        "structural contour or boundary segments backed by explicit line evidence; it "
        "excludes marker strokes, hatching, and texture lines and must never be inferred "
        "from the segment-owner count. If the needed visual concept is outside the catalog, emit a "
        "language_gap slot with all feature and parameter fields set to unset. Archival "
        "summary and indicator fields are non-executable narration; phrase them as "
        "affirmative visible descriptions. Fill exactly four slots per block and all "
        "twelve estimate fields in every slot using supports, does_not_support, or "
        "unclear. Put family parameter values in parameter_a through parameter_c in the "
        "catalog order and fill unused parameter positions with unset.\n\n"
        "BEGIN_CLOSED_FEATURE_CATALOG\n"
        + catalog
        + "\nEND_CLOSED_FEATURE_CATALOG"
    )


def panel_feature_proposer_contract_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-panel-feature-proposer-contract.v2",
            "protocol_id": PANEL_FEATURE_PROPOSER_PROTOCOL_ID,
            "feature_catalog_digest": feature_catalog_digest(),
            "prompt_digest": hashlib.sha256(
                panel_feature_proposer_prompt().encode("utf-8")
            ).hexdigest(),
            "output_schema_digest": canonical_digest(
                panel_feature_proposer_output_schema()
            ),
            "presentation_names": list(PANEL_FEATURE_PRESENTATION_NAMES),
            "blocks": list(PANEL_FEATURE_BLOCKS),
            "slots_per_direction": PANEL_FEATURE_SLOTS_PER_DIRECTION,
            "estimates_per_slot": len(PANEL_FEATURE_PRESENTATION_NAMES),
            "admission": {
                "minimum_native_support": PANEL_FEATURE_MIN_NATIVE_SUPPORT,
                "maximum_native_unclear": PANEL_FEATURE_MAX_NATIVE_UNCLEAR,
                "minimum_contrast_does_not_support": (
                    PANEL_FEATURE_MIN_CONTRAST_NONSUPPORT
                ),
                "maximum_contrast_support": PANEL_FEATURE_MAX_CONTRAST_SUPPORT,
                "maximum_contrast_unclear": PANEL_FEATURE_MAX_CONTRAST_UNCLEAR,
                "minimum_margin": PANEL_FEATURE_MIN_MARGIN,
            },
            "typed_specs_only": True,
            "narration_executable": False,
            "observer_vocabulary_contains_provenance": False,
        }
    )


def panel_feature_spec_to_wire(spec: PanelFeatureSpec) -> dict[str, str]:
    """Encode one typed spec into the three fixed parameter positions."""

    if type(spec) is not PanelFeatureSpec:
        raise TypeError("spec must be PanelFeatureSpec")
    names = _PARAMETER_FIELDS[spec.family]
    parameter_data = spec.parameters.to_data()
    values = [parameter_data[name] for name in names]
    if any(type(item) is not str for item in values):
        raise PanelFeatureProposerError("feature parameter is not a closed string")
    padded = [*values, *([PANEL_FEATURE_NONE] * (3 - len(values)))]
    return {
        "feature_family": spec.family.value,
        "subject_scope": spec.subject_scope.value,
        "reference_frame": spec.reference_frame.value,
        "parameter_a": padded[0],
        "parameter_b": padded[1],
        "parameter_c": padded[2],
    }


def panel_feature_spec_from_wire(value: Mapping[str, Any]) -> PanelFeatureSpec:
    """Parse a closed wire tuple directly; prose is never consulted."""

    raw = _fields(
        value,
        {
            "feature_family",
            "subject_scope",
            "reference_frame",
            "parameter_a",
            "parameter_b",
            "parameter_c",
        },
        "feature-spec wire tuple",
    )
    try:
        family = FeatureFamily(raw["feature_family"])
        scope = SubjectScope(raw["subject_scope"])
        frame = ReferenceFrame(raw["reference_frame"])
    except (TypeError, ValueError) as exc:
        raise PanelFeatureProposerError("feature-spec wire enum differs") from exc
    parameter_names = _PARAMETER_FIELDS[family]
    wire_values = (raw["parameter_a"], raw["parameter_b"], raw["parameter_c"])
    if any(type(item) is not str for item in wire_values):
        raise PanelFeatureProposerError("feature parameter wire values must be strings")
    if any(item != PANEL_FEATURE_NONE for item in wire_values[len(parameter_names) :]):
        raise PanelFeatureProposerError("unused feature parameter position differs")
    parameter_data = dict(zip(parameter_names, wire_values, strict=False))
    spec_data = {
        "schema": FEATURE_SPEC_SCHEMA,
        "catalog_digest": feature_catalog_digest(),
        "family": family.value,
        "subject_scope": scope.value,
        "reference_frame": frame.value,
        "parameters": parameter_data,
    }
    try:
        result = PanelFeatureSpec.from_data(spec_data)
    except (PanelSoftOntologyError, TypeError, ValueError) as exc:
        raise PanelFeatureProposerError("feature-spec wire tuple is unregistered") from exc
    if panel_feature_spec_to_wire(result) != dict(raw):
        raise PanelFeatureProposerError("feature-spec wire tuple is not canonical")
    return result


@dataclass(frozen=True, slots=True)
class PanelFeatureEstimateVector:
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.values) is not tuple
            or len(self.values) != len(PANEL_FEATURE_PRESENTATION_NAMES)
            or any(item not in PANEL_FEATURE_ESTIMATES for item in self.values)
        ):
            raise PanelFeatureProposerError("candidate estimate vector differs")

    def counts_for_block(self, block: str) -> tuple[int, int, int, int, int, int]:
        if block not in PANEL_FEATURE_BLOCKS:
            raise PanelFeatureProposerError("candidate block differs")
        split = PANEL_FEATURE_PANELS_PER_BLOCK
        native = self.values[:split] if block == "block_a" else self.values[split:]
        contrast = self.values[split:] if block == "block_a" else self.values[:split]
        native_support = native.count("supports")
        native_unclear = native.count("unclear")
        contrast_support = contrast.count("supports")
        contrast_nonsupport = contrast.count("does_not_support")
        contrast_unclear = contrast.count("unclear")
        return (
            native_support,
            native_unclear,
            contrast_support,
            contrast_nonsupport,
            contrast_unclear,
            native_support - contrast_support,
        )

    def to_data(self) -> list[str]:
        return list(self.values)


@dataclass(frozen=True, order=True, slots=True)
class PanelFeatureNominationGap:
    native_orientation: NativeOrientation
    raw_slot: int
    code: PanelFeatureNominationGapCode
    candidate_payload_digest: str

    def __post_init__(self) -> None:
        if type(self.native_orientation) is not NativeOrientation:
            raise TypeError("gap orientation has the wrong type")
        if type(self.raw_slot) is not int or self.raw_slot not in range(4):
            raise PanelFeatureProposerError("gap slot differs")
        if type(self.code) is not PanelFeatureNominationGapCode:
            raise TypeError("gap code has the wrong type")
        _digest(self.candidate_payload_digest, "gap candidate payload digest")

    @property
    def gap_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": PANEL_FEATURE_PROPOSER_NOMINATION_GAP_SCHEMA,
            "native_orientation": self.native_orientation.value,
            "raw_slot": self.raw_slot,
            "code": self.code.value,
            "candidate_payload_digest": self.candidate_payload_digest,
        }


@dataclass(frozen=True, slots=True)
class PanelFeatureNomination:
    source_block: str
    raw_slot: int
    proposal: NativeFeatureProposal
    estimates: PanelFeatureEstimateVector
    native_support_count: int
    native_unclear_count: int
    contrast_support_count: int
    contrast_does_not_support_count: int
    contrast_unclear_count: int
    support_margin: int

    def __post_init__(self) -> None:
        if self.source_block not in PANEL_FEATURE_BLOCKS:
            raise PanelFeatureProposerError("nomination block differs")
        if type(self.raw_slot) is not int or self.raw_slot not in range(4):
            raise PanelFeatureProposerError("nomination slot differs")
        if type(self.proposal) is not NativeFeatureProposal:
            raise TypeError("nomination proposal has the wrong type")
        if type(self.estimates) is not PanelFeatureEstimateVector:
            raise TypeError("nomination estimates have the wrong type")
        expected = self.estimates.counts_for_block(self.source_block)
        if expected != (
            self.native_support_count,
            self.native_unclear_count,
            self.contrast_support_count,
            self.contrast_does_not_support_count,
            self.contrast_unclear_count,
            self.support_margin,
        ):
            raise PanelFeatureProposerError("nomination admission counts differ")
        if not (
            self.native_support_count >= PANEL_FEATURE_MIN_NATIVE_SUPPORT
            and self.native_unclear_count <= PANEL_FEATURE_MAX_NATIVE_UNCLEAR
            and self.contrast_support_count <= PANEL_FEATURE_MAX_CONTRAST_SUPPORT
            and self.contrast_does_not_support_count
            >= PANEL_FEATURE_MIN_CONTRAST_NONSUPPORT
            and self.contrast_unclear_count <= PANEL_FEATURE_MAX_CONTRAST_UNCLEAR
            and self.support_margin >= PANEL_FEATURE_MIN_MARGIN
        ):
            raise PanelFeatureProposerError("unadmitted candidate became a nomination")

    @property
    def spec(self) -> PanelFeatureSpec:
        return self.proposal.spec

    @property
    def native_orientation(self) -> NativeOrientation:
        return self.proposal.provenance.native_orientation

    @property
    def nomination_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": PANEL_FEATURE_PROPOSER_NOMINATION_SCHEMA,
            "source_block": self.source_block,
            "raw_slot": self.raw_slot,
            "proposal": self.proposal.to_data(),
            "estimates_in_presentation_order": self.estimates.to_data(),
            "native_support_count": self.native_support_count,
            "native_unclear_count": self.native_unclear_count,
            "contrast_support_count": self.contrast_support_count,
            "contrast_does_not_support_count": self.contrast_does_not_support_count,
            "contrast_unclear_count": self.contrast_unclear_count,
            "support_margin": self.support_margin,
            "admission_rule": (
                "native-support-at-least-five-native-unclear-at-most-one-"
                "contrast-does-not-support-at-least-five-contrast-support-at-most-one-"
                "contrast-unclear-at-most-one-margin-at-least-three"
            ),
            "narration_executable": False,
        }


@dataclass(frozen=True, slots=True)
class PanelFeatureObserverVocabulary:
    specs: tuple[PanelFeatureSpec, ...]

    def __post_init__(self) -> None:
        if type(self.specs) is not tuple or not self.specs:
            raise PanelFeatureProposerError("observer vocabulary must be nonempty")
        if any(type(item) is not PanelFeatureSpec for item in self.specs):
            raise TypeError("observer vocabulary contains a non-spec")
        expected = tuple(sorted(self.specs, key=lambda item: item.spec_digest))
        if self.specs != expected or len({item.spec_digest for item in self.specs}) != len(self.specs):
            raise PanelFeatureProposerError("observer vocabulary is not globally deduplicated")

    @property
    def vocabulary_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": PANEL_FEATURE_OBSERVER_VOCABULARY_SCHEMA,
            "catalog_digest": feature_catalog_digest(),
            "specs": [item.to_data() for item in self.specs],
            "spec_order": "spec-digest-ascending",
            "provenance_included": False,
            "narration_included": False,
        }


@dataclass(frozen=True, slots=True)
class PanelFeatureProposerResult:
    payload_digest: str
    receipt_digest: str
    nominations: tuple[PanelFeatureNomination, ...]
    language_gaps: tuple[LanguageGapArtifact, ...]
    nomination_gaps: tuple[PanelFeatureNominationGap, ...]
    observer_vocabulary: PanelFeatureObserverVocabulary | None

    def __post_init__(self) -> None:
        _digest(self.payload_digest, "proposer payload digest")
        _digest(self.receipt_digest, "proposer receipt digest")
        if type(self.nominations) is not tuple or any(
            type(item) is not PanelFeatureNomination for item in self.nominations
        ):
            raise TypeError("proposer nominations have the wrong type")
        if type(self.language_gaps) is not tuple or any(
            type(item) is not LanguageGapArtifact for item in self.language_gaps
        ):
            raise TypeError("proposer language gaps have the wrong type")
        if type(self.nomination_gaps) is not tuple or any(
            type(item) is not PanelFeatureNominationGap for item in self.nomination_gaps
        ):
            raise TypeError("proposer nomination gaps have the wrong type")
        nomination_spec_digests = tuple(
            item.spec.spec_digest for item in self.nominations
        )
        if len(nomination_spec_digests) != len(set(nomination_spec_digests)):
            raise PanelFeatureProposerError(
                "proposer result retains a globally duplicated feature spec"
            )
        if self.observer_vocabulary is None:
            if self.nominations:
                raise PanelFeatureProposerError("nominations lack an observer vocabulary")
        else:
            expected = {
                item.spec.spec_digest: item.spec for item in self.nominations
            }
            if tuple(sorted(expected.values(), key=lambda item: item.spec_digest)) != self.observer_vocabulary.specs:
                raise PanelFeatureProposerError("observer vocabulary differs from nominations")

    @property
    def result_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": PANEL_FEATURE_PROPOSER_RESULT_SCHEMA,
            "protocol_id": PANEL_FEATURE_PROPOSER_PROTOCOL_ID,
            "contract_digest": panel_feature_proposer_contract_digest(),
            "payload_digest": self.payload_digest,
            "receipt_digest": self.receipt_digest,
            "nominations": [item.to_data() for item in self.nominations],
            "language_gaps": [item.to_data() for item in self.language_gaps],
            "nomination_gaps": [item.to_data() for item in self.nomination_gaps],
            "observer_vocabulary": (
                None if self.observer_vocabulary is None else self.observer_vocabulary.to_data()
            ),
            "typed_feature_specs_only": True,
            "narration_executable": False,
            "global_spec_deduplication": True,
        }


def _estimates(raw: Mapping[str, Any]) -> PanelFeatureEstimateVector:
    return PanelFeatureEstimateVector(
        tuple(
            raw[_estimate_field(block, index)]
            for block in PANEL_FEATURE_BLOCKS
            for index in range(PANEL_FEATURE_PANELS_PER_BLOCK)
        )
    )


def _narration_data(raw: Mapping[str, Any]) -> dict[str, object]:
    return {
        "summary": raw["archival_summary"],
        "visible_indicators": [
            raw["archival_indicator_a"],
            raw["archival_indicator_b"],
        ],
    }


def parse_panel_feature_proposer_payload(
    payload: Mapping[str, Any],
    *,
    proposer_receipt_digest: str,
    support_set_digest: str,
    task_context_digest: str,
    block_orientations: tuple[NativeOrientation, NativeOrientation] = (
        NativeOrientation.SIDE0_POSITIVE,
        NativeOrientation.SIDE1_POSITIVE,
    ),
) -> PanelFeatureProposerResult:
    """Parse, admit, and globally deduplicate one fixed structured payload."""

    receipt_digest = _digest(proposer_receipt_digest, "proposer receipt digest")
    support_digest = _digest(support_set_digest, "support set digest")
    task_digest = _digest(task_context_digest, "task context digest")
    if (
        type(block_orientations) is not tuple
        or len(block_orientations) != 2
        or set(block_orientations) != set(NativeOrientation)
        or any(type(item) is not NativeOrientation for item in block_orientations)
    ):
        raise PanelFeatureProposerError("block orientations must be an exact permutation")
    frozen = _canonical_payload(payload)
    expected_root = {
        _candidate_field(block, slot)
        for block in PANEL_FEATURE_BLOCKS
        for slot in range(PANEL_FEATURE_SLOTS_PER_DIRECTION)
    }
    root = _fields(frozen, expected_root, "panel feature proposer payload")
    contract_digest = panel_feature_proposer_contract_digest()
    provisional_nominations: list[tuple[PanelFeatureNomination, str]] = []
    language_gaps: list[LanguageGapArtifact] = []
    nomination_gaps: list[PanelFeatureNominationGap] = []

    for block_index, block in enumerate(PANEL_FEATURE_BLOCKS):
        orientation = block_orientations[block_index]
        for slot in range(PANEL_FEATURE_SLOTS_PER_DIRECTION):
            candidate = _fields(
                root[_candidate_field(block, slot)],
                _candidate_fields(),
                "panel feature candidate",
            )
            candidate_digest = canonical_digest(dict(candidate))
            estimates = _estimates(candidate)
            narration_data = _narration_data(candidate)
            if candidate["candidate_kind"] == PanelFeatureCandidateKind.LANGUAGE_GAP.value:
                if (
                    candidate["feature_family"] != PANEL_FEATURE_NONE
                    or candidate["subject_scope"] != PANEL_FEATURE_NONE
                    or candidate["reference_frame"] != PANEL_FEATURE_NONE
                    or any(candidate[name] != PANEL_FEATURE_NONE for name in ("parameter_a", "parameter_b", "parameter_c"))
                    or candidate["language_gap_kind"] == PANEL_FEATURE_NONE
                ):
                    raise PanelFeatureProposerError("language-gap candidate sentinels differ")
                try:
                    gap_kind = LanguageGapKind(candidate["language_gap_kind"])
                except (TypeError, ValueError) as exc:
                    raise PanelFeatureProposerError("language-gap kind differs") from exc
                language_gaps.append(
                    LanguageGapArtifact(
                        gap_kind,
                        canonical_digest(
                            {
                                "schema": "gkm.bongard-panel-feature-gap-narration.v1",
                                **narration_data,
                            }
                        ),
                        receipt_digest,
                        candidate_digest,
                    )
                )
                continue
            if candidate["candidate_kind"] != PanelFeatureCandidateKind.REGISTERED_FEATURE.value:
                raise PanelFeatureProposerError("candidate kind differs")
            if candidate["language_gap_kind"] != PANEL_FEATURE_NONE:
                raise PanelFeatureProposerError("registered candidate carries a language gap")
            wire = {
                name: candidate[name]
                for name in (
                    "feature_family",
                    "subject_scope",
                    "reference_frame",
                    "parameter_a",
                    "parameter_b",
                    "parameter_c",
                )
            }
            try:
                spec = panel_feature_spec_from_wire(wire)
            except PanelFeatureProposerError:
                language_gaps.append(
                    LanguageGapArtifact(
                        LanguageGapKind.AMBIGUOUS_FAMILY,
                        canonical_digest(
                            {
                                "schema": "gkm.bongard-panel-feature-gap-narration.v1",
                                **narration_data,
                            }
                        ),
                        receipt_digest,
                        candidate_digest,
                    )
                )
                continue
            try:
                narration = PanelFeatureNarration(
                    spec.spec_digest,
                    candidate["archival_summary"],
                    (
                        candidate["archival_indicator_a"],
                        candidate["archival_indicator_b"],
                    ),
                )
            except (PanelSoftOntologyError, TypeError, ValueError):
                nomination_gaps.append(
                    PanelFeatureNominationGap(
                        orientation,
                        slot,
                        PanelFeatureNominationGapCode.INVALID_ARCHIVAL_NARRATION,
                        candidate_digest,
                    )
                )
                continue
            (
                native_count,
                native_unclear_count,
                contrast_count,
                contrast_nonsupport_count,
                contrast_unclear_count,
                margin,
            ) = estimates.counts_for_block(block)
            if not (
                native_count >= PANEL_FEATURE_MIN_NATIVE_SUPPORT
                and native_unclear_count <= PANEL_FEATURE_MAX_NATIVE_UNCLEAR
                and contrast_count <= PANEL_FEATURE_MAX_CONTRAST_SUPPORT
                and contrast_nonsupport_count
                >= PANEL_FEATURE_MIN_CONTRAST_NONSUPPORT
                and contrast_unclear_count <= PANEL_FEATURE_MAX_CONTRAST_UNCLEAR
                and margin >= PANEL_FEATURE_MIN_MARGIN
            ):
                code = (
                    PanelFeatureNominationGapCode.SHARED_SALIENCE_REJECTED
                    if native_count == 6 and contrast_count == 6
                    else PanelFeatureNominationGapCode.CONTRASTIVE_ADMISSION_REJECTED
                )
                nomination_gaps.append(
                    PanelFeatureNominationGap(orientation, slot, code, candidate_digest)
                )
                continue
            provenance = NativeProposalProvenance(
                orientation,
                contract_digest,
                receipt_digest,
                support_digest,
                task_digest,
            )
            provisional_nominations.append(
                (
                    PanelFeatureNomination(
                        block,
                        slot,
                        NativeFeatureProposal(spec, narration, provenance),
                        estimates,
                        native_count,
                        native_unclear_count,
                        contrast_count,
                        contrast_nonsupport_count,
                        contrast_unclear_count,
                        margin,
                    ),
                    candidate_digest,
                )
            )

    grouped: dict[str, list[tuple[PanelFeatureNomination, str]]] = {}
    for item in provisional_nominations:
        grouped.setdefault(item[0].spec.spec_digest, []).append(item)
    nominations: list[PanelFeatureNomination] = []
    for spec_digest in sorted(grouped):
        group = sorted(
            grouped[spec_digest],
            key=lambda item: (
                PANEL_FEATURE_BLOCKS.index(item[0].source_block),
                item[0].raw_slot,
            ),
        )
        orientations = {item.native_orientation for item, _ in group}
        estimate_vectors = {item.estimates.values for item, _ in group}
        if len(orientations) > 1 or len(estimate_vectors) > 1:
            nomination_gaps.extend(
                PanelFeatureNominationGap(
                    item.native_orientation,
                    item.raw_slot,
                    PanelFeatureNominationGapCode.GLOBAL_SPEC_CONTRADICTION,
                    candidate_digest,
                )
                for item, candidate_digest in group
            )
            continue
        nominations.append(group[0][0])
        nomination_gaps.extend(
            PanelFeatureNominationGap(
                item.native_orientation,
                item.raw_slot,
                PanelFeatureNominationGapCode.DUPLICATE_NATIVE_SPEC,
                candidate_digest,
            )
            for item, candidate_digest in group[1:]
        )

    present_orientations = {item.native_orientation for item in nominations}
    absent_payload_digest = canonical_digest(
        {"schema": "gkm.bongard-panel-feature-missing-orientation.v1"}
    )
    for orientation in NativeOrientation:
        if orientation not in present_orientations:
            nomination_gaps.append(
                PanelFeatureNominationGap(
                    orientation,
                    0,
                    PanelFeatureNominationGapCode.MISSING_NATIVE_ORIENTATION,
                    absent_payload_digest,
                )
            )
    orientation_order = {item: index for index, item in enumerate(NativeOrientation)}
    ordered_nominations = tuple(
        sorted(
            nominations,
            key=lambda item: (orientation_order[item.native_orientation], item.raw_slot),
        )
    )
    specs_by_digest = {item.spec.spec_digest: item.spec for item in ordered_nominations}
    vocabulary = (
        None
        if not specs_by_digest
        else PanelFeatureObserverVocabulary(
            tuple(sorted(specs_by_digest.values(), key=lambda item: item.spec_digest))
        )
    )
    return PanelFeatureProposerResult(
        canonical_digest(frozen),
        receipt_digest,
        ordered_nominations,
        tuple(sorted(language_gaps, key=lambda item: item.gap_digest)),
        tuple(sorted(nomination_gaps)),
        vocabulary,
    )


@dataclass(frozen=True, slots=True)
class PanelFeatureProposerCallResult:
    """Minimal externally supplied receipt envelope for an injected call."""

    payload: Mapping[str, Any]
    prompt_digest: str
    output_schema_digest: str
    presentation_digest: str
    structured_output_digest: str
    external_receipt_digest: str

    def __post_init__(self) -> None:
        frozen = _canonical_payload(self.payload)
        object.__setattr__(self, "payload", frozen)
        for label, item in (
            ("call prompt digest", self.prompt_digest),
            ("call output schema digest", self.output_schema_digest),
            ("call presentation digest", self.presentation_digest),
            ("call structured output digest", self.structured_output_digest),
            ("call external receipt digest", self.external_receipt_digest),
        ):
            _digest(item, label)
        if self.structured_output_digest != canonical_digest(frozen):
            raise PanelFeatureProposerError("call structured output digest differs")

    @classmethod
    def seal(
        cls,
        payload: Mapping[str, Any],
        *,
        prompt: str,
        output_schema: Mapping[str, Any],
        presentation_digest: str,
        external_receipt_digest: str,
    ) -> "PanelFeatureProposerCallResult":
        frozen = _canonical_payload(payload)
        return cls(
            frozen,
            hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            canonical_digest(dict(output_schema)),
            presentation_digest,
            canonical_digest(frozen),
            external_receipt_digest,
        )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": PANEL_FEATURE_PROPOSER_CALL_RESULT_SCHEMA,
            "payload": dict(self.payload),
            "prompt_digest": self.prompt_digest,
            "output_schema_digest": self.output_schema_digest,
            "presentation_digest": self.presentation_digest,
            "structured_output_digest": self.structured_output_digest,
            "external_receipt_digest": self.external_receipt_digest,
        }


PanelFeatureReceiptedCall = Callable[
    [tuple[tuple[str, bytes], ...], str, Mapping[str, Any]],
    PanelFeatureProposerCallResult,
]


def invoke_panel_feature_proposer(
    support_pngs: Sequence[bytes],
    *,
    task_context_digest: str,
    call: PanelFeatureReceiptedCall,
    block_orientations: tuple[NativeOrientation, NativeOrientation] = (
        NativeOrientation.SIDE0_POSITIVE,
        NativeOrientation.SIDE1_POSITIVE,
    ),
) -> PanelFeatureProposerResult:
    """Invoke one injected boundary and verify its exact receipt commitments."""

    if isinstance(support_pngs, (bytes, str)) or len(support_pngs) != 12:
        raise PanelFeatureProposerError("exactly twelve support panels are required")
    if not callable(call):
        raise TypeError("call must be callable")
    presentation: list[tuple[str, bytes]] = []
    identities: list[dict[str, str]] = []
    for name, raw in zip(PANEL_FEATURE_PRESENTATION_NAMES, support_pngs, strict=True):
        if type(raw) is not bytes or not raw:
            raise PanelFeatureProposerError("support panel must be nonempty exact bytes")
        presentation.append((name, raw))
        identities.append({"name": name, "sha256": hashlib.sha256(raw).hexdigest()})
    presentation_digest = canonical_digest(
        {
            "schema": "gkm.bongard-panel-feature-proposer-presentation.v1",
            "images": identities,
        }
    )
    prompt = panel_feature_proposer_prompt()
    schema = panel_feature_proposer_output_schema()
    result = call(tuple(presentation), prompt, schema)
    if type(result) is not PanelFeatureProposerCallResult:
        raise PanelFeatureProposerError("injected call returned no receipted result")
    if (
        result.prompt_digest != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        or result.output_schema_digest != canonical_digest(schema)
        or result.presentation_digest != presentation_digest
    ):
        raise PanelFeatureProposerError("injected call receipt does not bind the request")
    return parse_panel_feature_proposer_payload(
        result.payload,
        proposer_receipt_digest=result.external_receipt_digest,
        support_set_digest=presentation_digest,
        task_context_digest=task_context_digest,
        block_orientations=block_orientations,
    )


__all__ = (
    "PANEL_FEATURE_BLOCKS",
    "PANEL_FEATURE_ESTIMATES",
    "PANEL_FEATURE_MAX_CONTRAST_SUPPORT",
    "PANEL_FEATURE_MAX_CONTRAST_UNCLEAR",
    "PANEL_FEATURE_MAX_NATIVE_UNCLEAR",
    "PANEL_FEATURE_MIN_CONTRAST_NONSUPPORT",
    "PANEL_FEATURE_MIN_MARGIN",
    "PANEL_FEATURE_MIN_NATIVE_SUPPORT",
    "PANEL_FEATURE_NONE",
    "PANEL_FEATURE_PANELS_PER_BLOCK",
    "PANEL_FEATURE_PRESENTATION_NAMES",
    "PANEL_FEATURE_PROPOSER_PROTOCOL_ID",
    "PANEL_FEATURE_SLOTS_PER_DIRECTION",
    "PanelFeatureCandidateKind",
    "PanelFeatureEstimateVector",
    "PanelFeatureNomination",
    "PanelFeatureNominationGap",
    "PanelFeatureNominationGapCode",
    "PanelFeatureObserverVocabulary",
    "PanelFeatureProposerCallResult",
    "PanelFeatureProposerError",
    "PanelFeatureProposerResult",
    "invoke_panel_feature_proposer",
    "panel_feature_proposer_contract_digest",
    "panel_feature_proposer_output_schema",
    "panel_feature_proposer_prompt",
    "panel_feature_spec_from_wire",
    "panel_feature_spec_to_wire",
    "parse_panel_feature_proposer_payload",
)

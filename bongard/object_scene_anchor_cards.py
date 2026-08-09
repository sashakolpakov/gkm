"""Canonical affirmative soft-predicate cards over selected object anchors."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_scene_anchor_bindings import (
    ObjectSceneAnchorBindingCatalog,
    ObjectSceneAnchorBindingSpec,
    ObjectSceneAnchorWitnessSpec,
    ObjectSceneResolvedAnchorBinding,
    build_object_scene_anchor_binding_catalog,
)
from bongard.object_scene_anchor_panel_manifest import (
    ObjectSceneAnchorPanelDecisionManifest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


OBJECT_SCENE_ANCHOR_CARD_WITNESS_SCHEMA = (
    "gkm.object-scene-anchor-card-witness.v1"
)
OBJECT_SCENE_ANCHOR_CARD_CITATION_SCHEMA = (
    "gkm.object-scene-anchor-card-positive-citation.v1"
)
OBJECT_SCENE_ANCHOR_PREDICATE_CARD_SCHEMA = (
    "gkm.object-scene-anchor-predicate-card.v1"
)
OBJECT_SCENE_DROPPED_ANCHOR_CARD_SCHEMA = (
    "gkm.object-scene-anchor-dropped-card.v1"
)
OBJECT_SCENE_ANCHOR_CARD_PROPOSAL_SCHEMA = (
    "gkm.object-scene-anchor-card-proposal.v1"
)
OBJECT_SCENE_ANCHOR_CARD_ORIENTATIONS = (
    "side0_positive",
    "side1_positive",
)
OBJECT_SCENE_ANCHOR_CARD_WITNESS_KINDS = (
    "shape_appearance",
    "marking_pattern",
    "spatial_relation",
    "part_topology",
)
OBJECT_SCENE_ANCHOR_CARD_DROP_REASON_CODES = (
    "malformed_card",
    "binding_spec_policy",
    "phrase_policy",
    "witness_policy",
    "variant_policy",
    "citation_policy",
    "foreign_panel",
    "foreign_object",
    "binding_mismatch",
    "duplicate_card",
)
OBJECT_SCENE_ANCHOR_MAX_CARDS_PER_ORIENTATION = 4
OBJECT_SCENE_ANCHOR_MAX_WITNESSES_PER_CARD = 4
OBJECT_SCENE_ANCHOR_MAX_UNION_WITNESSES = 32
OBJECT_SCENE_ANCHOR_POSITIVE_CITATION_COUNT = 6
OBJECT_SCENE_ANCHOR_MAX_VARIANTS = 4
OBJECT_SCENE_ANCHOR_MAX_NEAR_MISS_BOUNDARIES = 4

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_CARD_ID = re.compile(r"card_[0-9]{4}\Z")
_WITNESS_ID = re.compile(r"witness_[0-9]{2}\Z")
_PANEL_ALIAS = re.compile(r"panel_[0-9]{3}\Z")
_OBJECT_ID = re.compile(r"object_[0-9]{4}\Z")
_ANCHOR_ID = re.compile(r"(?:entity|part-[0-9]{8}|compact-[0-9]{8}|frame-[0-9]{8})\Z")
_PHRASE = re.compile(r"[a-z][a-z' -]{1,79}\Z")
_CLAUSE = re.compile(r"[a-z][a-z' -]{7,159}\Z")
_VARIANT = re.compile(r"[a-z][a-z', -]{7,159}\Z")
_BAD_COMMA = re.compile(r"\s,|,(?:\S| {2,})|,\Z")
_NEGATION = re.compile(
    r"\b(?:no|not|none|neither|nor|never|without|absent|absence|"
    r"lack|lacks|lacked|lacking|missing|omits?|omitted|excluding|except|unless)\b"
)
_NONATOMIC = re.compile(r"\b(?:or|either|versus|unlike)\b")
_FORBIDDEN_POLICY = re.compile(
    r"\b(?:panel|scene|canvas|support|positive|negative|side0|side1|"
    r"orientation|polarity|flip|complement|label|role|bucket|truth|"
    r"disposition|predicate|formula|query|prompt|python|lean|code|script|"
    r"function|lambda|import|return|eval|theorem|proof)\b"
)
_COMPARATIVE_POLICY = re.compile(
    r"\b(?:more|less|fewer|most|least|common|frequent|usually|typically|always)\b"
)
_BOUND_OBJECT_LOCAL_LOWER_BOUND = re.compile(
    r"\bthe bound (?:form|shape|object|entity|figure|path|outline|contour|marking|stroke) "
    r"(?:has|contains|carries|shows|includes|exhibits|forms|makes) at least "
    r"(?:"
    r"one (?:(?:clearly|visually|distinct|pronounced|visible|separate|internal|"
    r"external|rounded|sharp|angular|curved|straight|short|long|small|large|"
    r"broad|narrow|connected|disconnected) ){0,4}"
    r"(?:bend|corner|arm|mark|stroke|loop|turn|hole|lobe|tip|point|prong|branch|"
    r"segment|notch|cusp|endpoint|intersection|dot|stripe|spot)"
    r"|(?:two|three|four|five|six|seven|eight) "
    r"(?:(?:clearly|visually|distinct|pronounced|visible|separate|internal|"
    r"external|rounded|sharp|angular|curved|straight|short|long|small|large|"
    r"broad|narrow|connected|disconnected) ){0,4}"
    r"(?:bends|corners|arms|marks|strokes|loops|turns|holes|lobes|tips|points|"
    r"prongs|branches|segments|notches|cusps|endpoints|intersections|dots|"
    r"stripes|spots)"
    r")\b"
)
_PANEL_GLOBAL_RELATION = re.compile(
    r"\b(?:between|among|across) (?:objects|entities|figures)\b|"
    r"\banother (?:object|entity|figure)\b"
)
_BOUNDARY_EXCLUSION = re.compile(
    r"\b(?:does not qualify|do not qualify|is excluded|are excluded|"
    r"falls outside|fall outside)\b"
)


class ObjectSceneAnchorCardError(ValueError):
    """An anchor card, citation, or proposal is not canonical."""


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
    }


def _exact_fields(
    value: object, expected: set[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != set(expected)
    ):
        raise ObjectSceneAnchorCardError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectSceneAnchorCardError(f"{label} must be a lowercase SHA-256")
    return value


def _sequence(value: object, label: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ObjectSceneAnchorCardError(f"{label} must be a finite sequence")
    return tuple(value)


def _policy_local(value: str, label: str) -> str:
    comparative_view = _BOUND_OBJECT_LOCAL_LOWER_BOUND.sub(
        "the bound form has several local features", value
    )
    if (
        _FORBIDDEN_POLICY.search(value)
        or _COMPARATIVE_POLICY.search(comparative_view)
        or _PANEL_GLOBAL_RELATION.search(value)
    ):
        raise ObjectSceneAnchorCardError(f"{label} is not anchor-local prose")
    return value


def _phrase_text(value: object) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or "  " in value
        or _PHRASE.fullmatch(value) is None
        or _NEGATION.search(value)
        or _NONATOMIC.search(value)
    ):
        raise ObjectSceneAnchorCardError("card phrase is not bounded affirmative prose")
    return _policy_local(value, "card phrase")


def _witness_text(value: object) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or "  " in value
        or _CLAUSE.fullmatch(value) is None
        or _NEGATION.search(value)
        or _NONATOMIC.search(value)
    ):
        raise ObjectSceneAnchorCardError(
            "witness statement is not affirmative atomic prose"
        )
    return _policy_local(value, "witness statement")


def _variant_text(value: object) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or "  " in value
        or _VARIANT.fullmatch(value) is None
        or _BAD_COMMA.search(value)
        or _NEGATION.search(value)
    ):
        raise ObjectSceneAnchorCardError(
            "accepted variant is not affirmative inclusion prose"
        )
    return _policy_local(value, "accepted variant")


def _boundary_text(value: object) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or "  " in value
        or _CLAUSE.fullmatch(value) is None
    ):
        raise ObjectSceneAnchorCardError("near-miss boundary prose differs")
    exclusions = tuple(_BOUNDARY_EXCLUSION.finditer(value))
    if len(exclusions) != 1:
        raise ObjectSceneAnchorCardError(
            "near-miss boundary lacks one controlled exclusion"
        )
    scrubbed = (
        value[: exclusions[0].start()]
        + "qualifies"
        + value[exclusions[0].end() :]
    )
    if _NEGATION.search(scrubbed) or _NONATOMIC.search(scrubbed):
        raise ObjectSceneAnchorCardError(
            "near-miss boundary antecedent is not affirmative"
        )
    _policy_local(scrubbed, "near-miss boundary")
    return value


def _witness_semantic_content(
    value: "ObjectSceneAnchorCardWitness",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_CARD_WITNESS_SCHEMA,
        "kind": value.kind,
        "statement": value.statement,
    }


@dataclass(frozen=True, order=True, slots=True)
class ObjectSceneAnchorCardWitness:
    witness_id: str
    kind: str
    statement: str
    witness_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.witness_id, str) or _WITNESS_ID.fullmatch(
            self.witness_id
        ) is None:
            raise ObjectSceneAnchorCardError("card witness ID differs")
        if self.kind not in OBJECT_SCENE_ANCHOR_CARD_WITNESS_KINDS:
            raise ObjectSceneAnchorCardError("card witness kind differs")
        if _witness_text(self.statement) != self.statement:
            raise ObjectSceneAnchorCardError("card witness statement is not canonical")
        _digest(self.witness_digest, "card witness digest")
        if self.witness_digest != canonical_digest(_witness_semantic_content(self)):
            raise ObjectSceneAnchorCardError("card witness digest differs")

    @classmethod
    def create(
        cls, witness_id: str, kind: str, statement: object
    ) -> "ObjectSceneAnchorCardWitness":
        normalized = _witness_text(statement)
        provisional = object.__new__(cls)
        object.__setattr__(provisional, "kind", kind)
        object.__setattr__(provisional, "statement", normalized)
        return cls(
            witness_id,
            kind,
            normalized,
            canonical_digest(_witness_semantic_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {
            "witness_id": self.witness_id,
            **_witness_semantic_content(self),
            "witness_digest": self.witness_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorCardWitness":
        raw = _exact_fields(
            value,
            {
                "witness_id",
                "schema",
                "kind",
                "statement",
                "witness_digest",
            },
            "anchor card witness",
        )
        if raw["schema"] != OBJECT_SCENE_ANCHOR_CARD_WITNESS_SCHEMA:
            raise ObjectSceneAnchorCardError("card witness schema differs")
        result = cls(
            raw["witness_id"],
            raw["kind"],
            raw["statement"],
            raw["witness_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorCardError("card witness is not canonical")
        return result


def _citation_content(
    value: "ObjectSceneAnchorPositiveSupportCitation",
) -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_CARD_CITATION_SCHEMA,
        "panel_alias": value.panel_alias,
        "panel_manifest_digest": value.panel_manifest_digest,
        "binding_catalogs_digest": value.binding_catalogs_digest,
        "object_id": value.object_id,
        "anchor_id": value.anchor_id,
        "binding_alias": value.binding_alias,
        "binding_digest": value.binding_digest,
        "resolved_binding": value.resolved_binding.to_data(),
        **_authority_data(),
    }


@dataclass(frozen=True, order=True, slots=True)
class ObjectSceneAnchorPositiveSupportCitation:
    panel_alias: str
    panel_manifest_digest: str
    binding_catalogs_digest: str
    object_id: str
    anchor_id: str
    binding_alias: str
    binding_digest: str
    resolved_binding: ObjectSceneResolvedAnchorBinding
    citation_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.panel_alias, str) or _PANEL_ALIAS.fullmatch(
            self.panel_alias
        ) is None:
            raise ObjectSceneAnchorCardError("positive citation panel alias differs")
        for label, item in (
            ("panel manifest digest", self.panel_manifest_digest),
            ("binding catalogs digest", self.binding_catalogs_digest),
            ("resolved binding digest", self.binding_digest),
            ("positive citation digest", self.citation_digest),
        ):
            _digest(item, label)
        if (
            not isinstance(self.object_id, str)
            or _OBJECT_ID.fullmatch(self.object_id) is None
            or not isinstance(self.anchor_id, str)
            or _ANCHOR_ID.fullmatch(self.anchor_id) is None
            or type(self.resolved_binding) is not ObjectSceneResolvedAnchorBinding
            or self.object_id != self.resolved_binding.object_id
            or self.anchor_id != self.resolved_binding.anchor_id
            or self.binding_alias != self.resolved_binding.binding_alias
            or self.binding_digest != self.resolved_binding.binding_digest
        ):
            raise ObjectSceneAnchorCardError("positive citation binding differs")
        if self.citation_digest != canonical_digest(_citation_content(self)):
            raise ObjectSceneAnchorCardError("positive citation digest differs")

    @classmethod
    def create(
        cls,
        panel_alias: str,
        panel_manifest_digest: str,
        binding_catalogs_digest: str,
        resolved_binding: ObjectSceneResolvedAnchorBinding,
    ) -> "ObjectSceneAnchorPositiveSupportCitation":
        if type(resolved_binding) is not ObjectSceneResolvedAnchorBinding:
            raise TypeError("resolved_binding must be exact ObjectSceneResolvedAnchorBinding")
        values = {
            "panel_alias": panel_alias,
            "panel_manifest_digest": panel_manifest_digest,
            "binding_catalogs_digest": binding_catalogs_digest,
            "object_id": resolved_binding.object_id,
            "anchor_id": resolved_binding.anchor_id,
            "binding_alias": resolved_binding.binding_alias,
            "binding_digest": resolved_binding.binding_digest,
            "resolved_binding": resolved_binding,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            citation_digest=canonical_digest(_citation_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_citation_content(self), "citation_digest": self.citation_digest}

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectSceneAnchorPositiveSupportCitation":
        raw = _exact_fields(
            value,
            {
                "schema",
                "panel_alias",
                "panel_manifest_digest",
                "binding_catalogs_digest",
                "object_id",
                "anchor_id",
                "binding_alias",
                "binding_digest",
                "resolved_binding",
                *tuple(_authority_data()),
                "citation_digest",
            },
            "positive support citation",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_CARD_CITATION_SCHEMA
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["resolved_binding"], Mapping)
        ):
            raise ObjectSceneAnchorCardError("positive citation policy differs")
        result = cls(
            panel_alias=raw["panel_alias"],
            panel_manifest_digest=raw["panel_manifest_digest"],
            binding_catalogs_digest=raw["binding_catalogs_digest"],
            object_id=raw["object_id"],
            anchor_id=raw["anchor_id"],
            binding_alias=raw["binding_alias"],
            binding_digest=raw["binding_digest"],
            resolved_binding=ObjectSceneResolvedAnchorBinding.from_data(
                raw["resolved_binding"]
            ),
            citation_digest=raw["citation_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorCardError("positive citation is not canonical")
        return result


def _canonical_witnesses(value: object) -> tuple[ObjectSceneAnchorCardWitness, ...]:
    pairs: list[tuple[str, str]] = []
    for item in _sequence(value, "required witnesses"):
        if type(item) is ObjectSceneAnchorCardWitness:
            kind, statement = item.kind, item.statement
        else:
            raw = _exact_fields(
                item, {"kind", "statement"}, "required witness specification"
            )
            kind, statement = raw["kind"], raw["statement"]
        if kind not in OBJECT_SCENE_ANCHOR_CARD_WITNESS_KINDS:
            raise ObjectSceneAnchorCardError("card witness kind differs")
        pairs.append((kind, _witness_text(statement)))
    canonical = tuple(sorted(set(pairs)))
    if (
        not 1 <= len(canonical) <= OBJECT_SCENE_ANCHOR_MAX_WITNESSES_PER_CARD
        or len(canonical) != len(pairs)
    ):
        raise ObjectSceneAnchorCardError("card witness bounds or uniqueness differ")
    return tuple(
        ObjectSceneAnchorCardWitness.create(f"witness_{index:02d}", kind, statement)
        for index, (kind, statement) in enumerate(canonical)
    )


def _canonical_clauses(
    value: object,
    *,
    label: str,
    maximum: int,
    normalizer: Any,
) -> tuple[str, ...]:
    normalized = tuple(normalizer(item) for item in _sequence(value, label))
    canonical = tuple(sorted(set(normalized)))
    if len(canonical) > maximum or len(canonical) != len(normalized):
        raise ObjectSceneAnchorCardError(f"{label} bounds or uniqueness differ")
    return canonical


def _canonical_citations(
    value: object,
) -> tuple[ObjectSceneAnchorPositiveSupportCitation, ...]:
    items = _sequence(value, "positive support citations")
    if any(type(item) is not ObjectSceneAnchorPositiveSupportCitation for item in items):
        raise ObjectSceneAnchorCardError("positive support citation type differs")
    result = tuple(sorted(items, key=lambda item: item.panel_alias))
    if (
        len(result) != OBJECT_SCENE_ANCHOR_POSITIVE_CITATION_COUNT
        or len({item.panel_alias for item in result}) != len(result)
    ):
        raise ObjectSceneAnchorCardError(
            "positive support citations are not exact and unique"
        )
    return result


def _card_content(value: "ObjectSceneAnchorPredicateCard") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_PREDICATE_CARD_SCHEMA,
        "card_id": value.card_id,
        "orientation": value.orientation,
        "phrase": value.phrase,
        "binding_spec": value.binding_spec.to_data(),
        "required_witnesses": [item.to_data() for item in value.required_witnesses],
        "accepted_variants": list(value.accepted_variants),
        "near_miss_boundaries": list(value.near_miss_boundaries),
        "positive_support_citations": [
            item.to_data() for item in value.positive_support_citations
        ],
        "truth_assignment_present": False,
        "polarity_flip_authorized": False,
        "panel_global_relation_authorized": False,
        "free_code_authorized": False,
        "accepted_variants_compile_to_atoms": False,
        "near_miss_boundaries_compile_to_atoms": False,
        **_authority_data(),
    }


@dataclass(frozen=True, order=True, slots=True)
class ObjectSceneAnchorPredicateCard:
    card_id: str
    orientation: str
    phrase: str
    binding_spec: ObjectSceneAnchorBindingSpec
    required_witnesses: tuple[ObjectSceneAnchorCardWitness, ...]
    accepted_variants: tuple[str, ...]
    near_miss_boundaries: tuple[str, ...]
    positive_support_citations: tuple[
        ObjectSceneAnchorPositiveSupportCitation, ...
    ]
    card_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.card_id, str) or _CARD_ID.fullmatch(self.card_id) is None:
            raise ObjectSceneAnchorCardError("anchor card ID differs")
        if self.orientation not in OBJECT_SCENE_ANCHOR_CARD_ORIENTATIONS:
            raise ObjectSceneAnchorCardError("anchor card orientation differs")
        if _phrase_text(self.phrase) != self.phrase:
            raise ObjectSceneAnchorCardError("anchor card phrase is not canonical")
        if type(self.binding_spec) is not ObjectSceneAnchorBindingSpec:
            raise TypeError("binding_spec must be exact ObjectSceneAnchorBindingSpec")
        if ObjectSceneAnchorBindingSpec.from_data(self.binding_spec.to_data()) != self.binding_spec:
            raise ObjectSceneAnchorCardError("anchor card binding spec differs")
        witnesses = _canonical_witnesses(self.required_witnesses)
        variants = _canonical_clauses(
            self.accepted_variants,
            label="accepted variants",
            maximum=OBJECT_SCENE_ANCHOR_MAX_VARIANTS,
            normalizer=_variant_text,
        )
        boundaries = _canonical_clauses(
            self.near_miss_boundaries,
            label="near-miss boundaries",
            maximum=OBJECT_SCENE_ANCHOR_MAX_NEAR_MISS_BOUNDARIES,
            normalizer=_boundary_text,
        )
        citations = _canonical_citations(self.positive_support_citations)
        if (
            witnesses != self.required_witnesses
            or variants != self.accepted_variants
            or boundaries != self.near_miss_boundaries
            or citations != self.positive_support_citations
            or any(
                item.resolved_binding.spec_digest != self.binding_spec.spec_digest
                or item.resolved_binding.anchor_kind != self.binding_spec.anchor_kind
                for item in citations
            )
        ):
            raise ObjectSceneAnchorCardError(
                "anchor card criteria or shared binding spec differ"
            )
        _digest(self.card_digest, "anchor card digest")
        if self.card_digest != canonical_digest(_card_content(self)):
            raise ObjectSceneAnchorCardError("anchor card digest differs")

    @classmethod
    def create(
        cls,
        card_id: str,
        orientation: str,
        phrase: object,
        binding_spec: ObjectSceneAnchorBindingSpec,
        required_witnesses: object,
        accepted_variants: object,
        near_miss_boundaries: object,
        positive_support_citations: object,
    ) -> "ObjectSceneAnchorPredicateCard":
        if type(binding_spec) is not ObjectSceneAnchorBindingSpec:
            raise TypeError("binding_spec must be exact ObjectSceneAnchorBindingSpec")
        values = {
            "card_id": card_id,
            "orientation": orientation,
            "phrase": _phrase_text(phrase),
            "binding_spec": ObjectSceneAnchorBindingSpec.from_data(
                binding_spec.to_data()
            ),
            "required_witnesses": _canonical_witnesses(required_witnesses),
            "accepted_variants": _canonical_clauses(
                accepted_variants,
                label="accepted variants",
                maximum=OBJECT_SCENE_ANCHOR_MAX_VARIANTS,
                normalizer=_variant_text,
            ),
            "near_miss_boundaries": _canonical_clauses(
                near_miss_boundaries,
                label="near-miss boundaries",
                maximum=OBJECT_SCENE_ANCHOR_MAX_NEAR_MISS_BOUNDARIES,
                normalizer=_boundary_text,
            ),
            "positive_support_citations": _canonical_citations(
                positive_support_citations
            ),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            card_digest=canonical_digest(_card_content(provisional)),
        )

    @property
    def binding_witness_specs(self) -> tuple[ObjectSceneAnchorWitnessSpec, ...]:
        """Project only actual affirmative witnesses into the binding evaluator."""

        return tuple(
            ObjectSceneAnchorWitnessSpec(item.witness_id, item.witness_digest)
            for item in self.required_witnesses
        )

    def to_data(self) -> dict[str, object]:
        return {**_card_content(self), "card_digest": self.card_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorPredicateCard":
        raw = _exact_fields(
            value,
            {
                "schema",
                "card_id",
                "orientation",
                "phrase",
                "binding_spec",
                "required_witnesses",
                "accepted_variants",
                "near_miss_boundaries",
                "positive_support_citations",
                "truth_assignment_present",
                "polarity_flip_authorized",
                "panel_global_relation_authorized",
                "free_code_authorized",
                "accepted_variants_compile_to_atoms",
                "near_miss_boundaries_compile_to_atoms",
                *tuple(_authority_data()),
                "card_digest",
            },
            "anchor predicate card",
        )
        false_fields = (
            "truth_assignment_present",
            "polarity_flip_authorized",
            "panel_global_relation_authorized",
            "free_code_authorized",
            "accepted_variants_compile_to_atoms",
            "near_miss_boundaries_compile_to_atoms",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_PREDICATE_CARD_SCHEMA
            or any(raw[key] is not False for key in false_fields)
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["binding_spec"], Mapping)
            or not isinstance(raw["required_witnesses"], list)
            or not isinstance(raw["accepted_variants"], list)
            or not isinstance(raw["near_miss_boundaries"], list)
            or not isinstance(raw["positive_support_citations"], list)
        ):
            raise ObjectSceneAnchorCardError("anchor predicate card policy differs")
        result = cls(
            card_id=raw["card_id"],
            orientation=raw["orientation"],
            phrase=raw["phrase"],
            binding_spec=ObjectSceneAnchorBindingSpec.from_data(raw["binding_spec"]),
            required_witnesses=tuple(
                ObjectSceneAnchorCardWitness.from_data(item)
                for item in raw["required_witnesses"]
            ),
            accepted_variants=tuple(raw["accepted_variants"]),
            near_miss_boundaries=tuple(raw["near_miss_boundaries"]),
            positive_support_citations=tuple(
                ObjectSceneAnchorPositiveSupportCitation.from_data(item)
                for item in raw["positive_support_citations"]
            ),
            card_digest=raw["card_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorCardError("anchor predicate card is not canonical")
        return result


def _drop_content(value: "ObjectSceneDroppedAnchorCard") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_DROPPED_ANCHOR_CARD_SCHEMA,
        "orientation": value.orientation,
        "input_index": value.input_index,
        "reason_code": value.reason_code,
        "rejected_payload_or_prose_persisted": False,
        "truth_assignment_present": False,
    }


@dataclass(frozen=True, order=True, slots=True)
class ObjectSceneDroppedAnchorCard:
    orientation: str
    input_index: int
    reason_code: str
    drop_digest: str

    def __post_init__(self) -> None:
        if (
            self.orientation not in OBJECT_SCENE_ANCHOR_CARD_ORIENTATIONS
            or type(self.input_index) is not int
            or self.input_index < 0
            or self.reason_code not in OBJECT_SCENE_ANCHOR_CARD_DROP_REASON_CODES
        ):
            raise ObjectSceneAnchorCardError("dropped anchor card record differs")
        _digest(self.drop_digest, "dropped anchor card digest")
        if self.drop_digest != canonical_digest(_drop_content(self)):
            raise ObjectSceneAnchorCardError("dropped anchor card digest differs")

    @classmethod
    def create(
        cls, orientation: str, input_index: int, reason_code: str
    ) -> "ObjectSceneDroppedAnchorCard":
        provisional = object.__new__(cls)
        object.__setattr__(provisional, "orientation", orientation)
        object.__setattr__(provisional, "input_index", input_index)
        object.__setattr__(provisional, "reason_code", reason_code)
        return cls(
            orientation,
            input_index,
            reason_code,
            canonical_digest(_drop_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_drop_content(self), "drop_digest": self.drop_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneDroppedAnchorCard":
        raw = _exact_fields(
            value,
            {
                "schema",
                "orientation",
                "input_index",
                "reason_code",
                "rejected_payload_or_prose_persisted",
                "truth_assignment_present",
                "drop_digest",
            },
            "dropped anchor card",
        )
        if (
            raw["schema"] != OBJECT_SCENE_DROPPED_ANCHOR_CARD_SCHEMA
            or raw["rejected_payload_or_prose_persisted"] is not False
            or raw["truth_assignment_present"] is not False
        ):
            raise ObjectSceneAnchorCardError("dropped anchor card policy differs")
        result = cls(
            raw["orientation"],
            raw["input_index"],
            raw["reason_code"],
            raw["drop_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorCardError("dropped anchor card is not canonical")
        return result


def _card_sort_key(value: ObjectSceneAnchorPredicateCard) -> tuple[object, ...]:
    return (
        value.phrase,
        value.binding_spec.spec_digest,
        tuple(item.witness_digest for item in value.required_witnesses),
        value.accepted_variants,
        value.near_miss_boundaries,
        tuple(item.citation_digest for item in value.positive_support_citations),
    )


def _proposal_content(value: "ObjectSceneAnchorCardProposal") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_CARD_PROPOSAL_SCHEMA,
        "side0_positive": [item.to_data() for item in value.side0_positive],
        "side1_positive": [item.to_data() for item in value.side1_positive],
        "dropped_cards": [item.to_data() for item in value.dropped_cards],
        "truth_assignment_present": False,
        "automatic_polarity_flip_authorized": False,
        "model_payload_or_receipt_persisted": False,
        "only_required_witnesses_compile_to_atoms": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorCardProposal:
    side0_positive: tuple[ObjectSceneAnchorPredicateCard, ...]
    side1_positive: tuple[ObjectSceneAnchorPredicateCard, ...]
    dropped_cards: tuple[ObjectSceneDroppedAnchorCard, ...]
    proposal_digest: str

    def __post_init__(self) -> None:
        if any(
            type(items) is not tuple
            for items in (self.side0_positive, self.side1_positive, self.dropped_cards)
        ):
            raise TypeError("anchor proposal inventories must be exact tuples")
        cards = (*self.side0_positive, *self.side1_positive)
        if (
            not 1 <= len(self.side0_positive) <= OBJECT_SCENE_ANCHOR_MAX_CARDS_PER_ORIENTATION
            or not 1 <= len(self.side1_positive) <= OBJECT_SCENE_ANCHOR_MAX_CARDS_PER_ORIENTATION
            or any(type(item) is not ObjectSceneAnchorPredicateCard for item in cards)
            or any(type(item) is not ObjectSceneDroppedAnchorCard for item in self.dropped_cards)
            or tuple(item.orientation for item in self.side0_positive)
            != ("side0_positive",) * len(self.side0_positive)
            or tuple(item.orientation for item in self.side1_positive)
            != ("side1_positive",) * len(self.side1_positive)
            or tuple(item.card_id for item in cards)
            != tuple(f"card_{index:04d}" for index in range(len(cards)))
            or tuple(_card_sort_key(item) for item in self.side0_positive)
            != tuple(sorted(_card_sort_key(item) for item in self.side0_positive))
            or tuple(_card_sort_key(item) for item in self.side1_positive)
            != tuple(sorted(_card_sort_key(item) for item in self.side1_positive))
            or sum(len(item.required_witnesses) for item in cards)
            > OBJECT_SCENE_ANCHOR_MAX_UNION_WITNESSES
            or self.dropped_cards
            != tuple(
                sorted(
                    self.dropped_cards,
                    key=lambda item: (item.orientation, item.input_index),
                )
            )
            or len({(item.orientation, item.input_index) for item in self.dropped_cards})
            != len(self.dropped_cards)
        ):
            raise ObjectSceneAnchorCardError("anchor card proposal inventory differs")
        _digest(self.proposal_digest, "anchor card proposal digest")
        if self.proposal_digest != canonical_digest(_proposal_content(self)):
            raise ObjectSceneAnchorCardError("anchor card proposal digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_proposal_content(self), "proposal_digest": self.proposal_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorCardProposal":
        raw = _exact_fields(
            value,
            {
                "schema",
                "side0_positive",
                "side1_positive",
                "dropped_cards",
                "truth_assignment_present",
                "automatic_polarity_flip_authorized",
                "model_payload_or_receipt_persisted",
                "only_required_witnesses_compile_to_atoms",
                *tuple(_authority_data()),
                "proposal_digest",
            },
            "anchor card proposal",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_CARD_PROPOSAL_SCHEMA
            or raw["truth_assignment_present"] is not False
            or raw["automatic_polarity_flip_authorized"] is not False
            or raw["model_payload_or_receipt_persisted"] is not False
            or raw["only_required_witnesses_compile_to_atoms"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or any(
                not isinstance(raw[key], list)
                for key in ("side0_positive", "side1_positive", "dropped_cards")
            )
        ):
            raise ObjectSceneAnchorCardError("anchor card proposal policy differs")
        result = cls(
            side0_positive=tuple(
                ObjectSceneAnchorPredicateCard.from_data(item)
                for item in raw["side0_positive"]
            ),
            side1_positive=tuple(
                ObjectSceneAnchorPredicateCard.from_data(item)
                for item in raw["side1_positive"]
            ),
            dropped_cards=tuple(
                ObjectSceneDroppedAnchorCard.from_data(item)
                for item in raw["dropped_cards"]
            ),
            proposal_digest=raw["proposal_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorCardError("anchor card proposal is not canonical")
        return result


def _normalized_panel_manifests(
    value: object, label: str
) -> dict[str, ObjectSceneAnchorPanelDecisionManifest]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or _PANEL_ALIAS.fullmatch(key) is None
        for key in value
    ):
        raise ObjectSceneAnchorCardError(f"{label} must map exact panel aliases")
    if len(value) != OBJECT_SCENE_ANCHOR_POSITIVE_CITATION_COUNT:
        raise ObjectSceneAnchorCardError(f"{label} must contain exactly six panels")
    result: dict[str, ObjectSceneAnchorPanelDecisionManifest] = {}
    for alias in sorted(value):
        manifest = value[alias]
        if type(manifest) is not ObjectSceneAnchorPanelDecisionManifest:
            raise TypeError("panel manifests must be exact decision manifests")
        result[alias] = ObjectSceneAnchorPanelDecisionManifest.from_data(
            manifest.to_data()
        )
    if len({item.panel_digest for item in result.values()}) != len(result):
        raise ObjectSceneAnchorCardError(f"{label} repeats an exact panel")
    return result


def _binding_catalogs_digest(
    panel_manifest: ObjectSceneAnchorPanelDecisionManifest,
    binding_spec: ObjectSceneAnchorBindingSpec,
    catalogs: tuple[ObjectSceneAnchorBindingCatalog, ...],
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-card-panel-binding-catalogs.v1",
            "panel_manifest_digest": panel_manifest.manifest_digest,
            "binding_spec_digest": binding_spec.spec_digest,
            "object_ids": list(panel_manifest.object_ids),
            "catalogs": [item.to_data() for item in catalogs],
            "complete_object_inventory_required": True,
        }
    )


def _catalogs_for_panel(
    panel_manifest: ObjectSceneAnchorPanelDecisionManifest,
    binding_spec: ObjectSceneAnchorBindingSpec,
) -> tuple[tuple[ObjectSceneAnchorBindingCatalog, ...], str]:
    catalogs = tuple(
        build_object_scene_anchor_binding_catalog(
            decision,
            binding_spec,
            expected_object_id=object_id,
        )
        for object_id, decision in zip(
            panel_manifest.object_ids,
            panel_manifest.object_decisions,
            strict=True,
        )
    )
    if (
        tuple(item.object_id for item in catalogs) != panel_manifest.object_ids
        or any(item.binding_spec != binding_spec for item in catalogs)
    ):
        raise ObjectSceneAnchorCardError(
            "ordered per-object binding catalogs differ from panel manifest"
        )
    return catalogs, _binding_catalogs_digest(panel_manifest, binding_spec, catalogs)


class _CitationDrop(Exception):
    def __init__(self, reason_code: str):
        self.reason_code = reason_code


def _resolved_citations(
    value: object,
    panel_manifests: Mapping[str, ObjectSceneAnchorPanelDecisionManifest],
    binding_spec: ObjectSceneAnchorBindingSpec,
) -> tuple[ObjectSceneAnchorPositiveSupportCitation, ...]:
    try:
        rows = _sequence(value, "positive citation specifications")
    except ObjectSceneAnchorCardError as exc:
        raise _CitationDrop("citation_policy") from exc
    expected_aliases = tuple(sorted(panel_manifests))
    if len(rows) != OBJECT_SCENE_ANCHOR_POSITIVE_CITATION_COUNT:
        raise _CitationDrop("citation_policy")
    parsed: list[tuple[str, str, str]] = []
    for item in rows:
        try:
            raw = _exact_fields(
                item,
                {"panel_alias", "object_id", "anchor_id"},
                "positive citation specification",
            )
        except ObjectSceneAnchorCardError as exc:
            raise _CitationDrop("citation_policy") from exc
        panel_alias, object_id, anchor_id = (
            raw["panel_alias"],
            raw["object_id"],
            raw["anchor_id"],
        )
        if not isinstance(panel_alias, str) or _PANEL_ALIAS.fullmatch(panel_alias) is None:
            raise _CitationDrop("citation_policy")
        if not isinstance(object_id, str) or _OBJECT_ID.fullmatch(object_id) is None:
            raise _CitationDrop("citation_policy")
        if not isinstance(anchor_id, str) or _ANCHOR_ID.fullmatch(anchor_id) is None:
            raise _CitationDrop("citation_policy")
        parsed.append((panel_alias, object_id, anchor_id))
    if tuple(item[0] for item in parsed) != expected_aliases:
        if any(item[0] not in panel_manifests for item in parsed):
            raise _CitationDrop("foreign_panel")
        raise _CitationDrop("citation_policy")

    cache: dict[str, tuple[tuple[ObjectSceneAnchorBindingCatalog, ...], str]] = {}
    result: list[ObjectSceneAnchorPositiveSupportCitation] = []
    for panel_alias, object_id, anchor_id in parsed:
        manifest = panel_manifests[panel_alias]
        if object_id not in manifest.by_object_id:
            raise _CitationDrop("foreign_object")
        catalogs, catalogs_digest = cache.setdefault(
            panel_alias, _catalogs_for_panel(manifest, binding_spec)
        )
        catalog = catalogs[manifest.object_ids.index(object_id)]
        if catalog.hard_disposition is not Disposition.PRESENT:
            raise _CitationDrop("binding_mismatch")
        matches = tuple(
            item for item in catalog.bindings if item.anchor_id == anchor_id
        )
        if len(matches) != 1:
            raise _CitationDrop("binding_mismatch")
        result.append(
            ObjectSceneAnchorPositiveSupportCitation.create(
                panel_alias,
                manifest.manifest_digest,
                catalogs_digest,
                matches[0],
            )
        )
    return tuple(result)


def _candidate_signature(card: ObjectSceneAnchorPredicateCard) -> tuple[object, ...]:
    return (
        card.phrase,
        card.binding_spec.spec_digest,
        tuple((item.kind, item.statement) for item in card.required_witnesses),
        card.accepted_variants,
        card.near_miss_boundaries,
    )


def _reidentify_card(
    value: ObjectSceneAnchorPredicateCard, card_id: str
) -> ObjectSceneAnchorPredicateCard:
    return ObjectSceneAnchorPredicateCard.create(
        card_id,
        value.orientation,
        value.phrase,
        value.binding_spec,
        value.required_witnesses,
        value.accepted_variants,
        value.near_miss_boundaries,
        value.positive_support_citations,
    )


def _construct_proposal(
    side0: tuple[ObjectSceneAnchorPredicateCard, ...],
    side1: tuple[ObjectSceneAnchorPredicateCard, ...],
    dropped: tuple[ObjectSceneDroppedAnchorCard, ...],
) -> ObjectSceneAnchorCardProposal:
    values = {
        "side0_positive": side0,
        "side1_positive": side1,
        "dropped_cards": dropped,
    }
    provisional = object.__new__(ObjectSceneAnchorCardProposal)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorCardProposal(
        **values,
        proposal_digest=canonical_digest(_proposal_content(provisional)),
    )


def build_object_scene_anchor_card_proposal(
    payload: Mapping[str, Any],
    *,
    side0_panel_manifests: Mapping[
        str, ObjectSceneAnchorPanelDecisionManifest
    ],
    side1_panel_manifests: Mapping[
        str, ObjectSceneAnchorPanelDecisionManifest
    ],
) -> ObjectSceneAnchorCardProposal:
    """Parse cards and resolve every positive citation through complete panels."""

    raw = _exact_fields(
        payload,
        {"side0_positive", "side1_positive"},
        "anchor card proposal payload",
    )
    buckets_raw: dict[str, tuple[object, ...]] = {}
    for orientation in OBJECT_SCENE_ANCHOR_CARD_ORIENTATIONS:
        rows = _sequence(raw[orientation], f"{orientation} cards")
        if not 1 <= len(rows) <= OBJECT_SCENE_ANCHOR_MAX_CARDS_PER_ORIENTATION:
            raise ObjectSceneAnchorCardError(
                "anchor card proposal bucket capacity differs"
            )
        buckets_raw[orientation] = rows
    manifests = {
        "side0_positive": _normalized_panel_manifests(
            side0_panel_manifests, "side0 panel manifests"
        ),
        "side1_positive": _normalized_panel_manifests(
            side1_panel_manifests, "side1 panel manifests"
        ),
    }
    if (
        set(manifests["side0_positive"]) & set(manifests["side1_positive"])
        or len(
            {
                item.panel_digest
                for bucket in manifests.values()
                for item in bucket.values()
            }
        )
        != 2 * OBJECT_SCENE_ANCHOR_POSITIVE_CITATION_COUNT
    ):
        raise ObjectSceneAnchorCardError(
            "anchor card orientation panel inventories overlap"
        )

    accepted: dict[str, list[tuple[int, ObjectSceneAnchorPredicateCard]]] = {
        orientation: [] for orientation in OBJECT_SCENE_ANCHOR_CARD_ORIENTATIONS
    }
    dropped: list[ObjectSceneDroppedAnchorCard] = []
    seen_signatures: set[tuple[object, ...]] = set()
    card_fields = {
        "phrase",
        "binding_spec",
        "required_witnesses",
        "accepted_variants",
        "near_miss_boundaries",
        "positive_support_citations",
    }
    for orientation in OBJECT_SCENE_ANCHOR_CARD_ORIENTATIONS:
        for input_index, item in enumerate(buckets_raw[orientation]):
            try:
                item_raw = _exact_fields(item, card_fields, "anchor card payload")
            except ObjectSceneAnchorCardError:
                dropped.append(
                    ObjectSceneDroppedAnchorCard.create(
                        orientation, input_index, "malformed_card"
                    )
                )
                continue
            try:
                if not isinstance(item_raw["binding_spec"], Mapping):
                    raise ObjectSceneAnchorCardError("binding spec payload differs")
                binding_spec = ObjectSceneAnchorBindingSpec.from_data(
                    item_raw["binding_spec"]
                )
            except Exception:
                dropped.append(
                    ObjectSceneDroppedAnchorCard.create(
                        orientation, input_index, "binding_spec_policy"
                    )
                )
                continue
            try:
                phrase = _phrase_text(item_raw["phrase"])
            except ObjectSceneAnchorCardError:
                dropped.append(
                    ObjectSceneDroppedAnchorCard.create(
                        orientation, input_index, "phrase_policy"
                    )
                )
                continue
            try:
                witnesses = _canonical_witnesses(item_raw["required_witnesses"])
            except ObjectSceneAnchorCardError:
                dropped.append(
                    ObjectSceneDroppedAnchorCard.create(
                        orientation, input_index, "witness_policy"
                    )
                )
                continue
            try:
                variants = _canonical_clauses(
                    item_raw["accepted_variants"],
                    label="accepted variants",
                    maximum=OBJECT_SCENE_ANCHOR_MAX_VARIANTS,
                    normalizer=_variant_text,
                )
                boundaries = _canonical_clauses(
                    item_raw["near_miss_boundaries"],
                    label="near-miss boundaries",
                    maximum=OBJECT_SCENE_ANCHOR_MAX_NEAR_MISS_BOUNDARIES,
                    normalizer=_boundary_text,
                )
            except ObjectSceneAnchorCardError:
                dropped.append(
                    ObjectSceneDroppedAnchorCard.create(
                        orientation, input_index, "variant_policy"
                    )
                )
                continue
            try:
                citations = _resolved_citations(
                    item_raw["positive_support_citations"],
                    manifests[orientation],
                    binding_spec,
                )
            except _CitationDrop as exc:
                dropped.append(
                    ObjectSceneDroppedAnchorCard.create(
                        orientation, input_index, exc.reason_code
                    )
                )
                continue
            candidate = ObjectSceneAnchorPredicateCard.create(
                "card_0000",
                orientation,
                phrase,
                binding_spec,
                witnesses,
                variants,
                boundaries,
                citations,
            )
            signature = _candidate_signature(candidate)
            if signature in seen_signatures:
                dropped.append(
                    ObjectSceneDroppedAnchorCard.create(
                        orientation, input_index, "duplicate_card"
                    )
                )
                continue
            seen_signatures.add(signature)
            accepted[orientation].append((input_index, candidate))

    if any(not accepted[orientation] for orientation in OBJECT_SCENE_ANCHOR_CARD_ORIENTATIONS):
        raise ObjectSceneAnchorCardError(
            "anchor card proposal has no usable card in one or both orientations"
        )
    ordered: dict[str, list[ObjectSceneAnchorPredicateCard]] = {}
    next_id = 0
    for orientation in OBJECT_SCENE_ANCHOR_CARD_ORIENTATIONS:
        cards = sorted(
            (item for _, item in accepted[orientation]), key=_card_sort_key
        )
        ordered[orientation] = []
        for card in cards:
            ordered[orientation].append(
                _reidentify_card(card, f"card_{next_id:04d}")
            )
            next_id += 1
    drops = tuple(
        sorted(dropped, key=lambda item: (item.orientation, item.input_index))
    )
    proposal = _construct_proposal(
        tuple(ordered["side0_positive"]),
        tuple(ordered["side1_positive"]),
        drops,
    )
    return verify_object_scene_anchor_card_proposal(
        proposal,
        side0_panel_manifests=manifests["side0_positive"],
        side1_panel_manifests=manifests["side1_positive"],
    )


def verify_object_scene_anchor_card_proposal(
    proposal: ObjectSceneAnchorCardProposal,
    *,
    side0_panel_manifests: Mapping[
        str, ObjectSceneAnchorPanelDecisionManifest
    ],
    side1_panel_manifests: Mapping[
        str, ObjectSceneAnchorPanelDecisionManifest
    ],
) -> ObjectSceneAnchorCardProposal:
    """Strictly restore and re-resolve all six citations for every card."""

    if type(proposal) is not ObjectSceneAnchorCardProposal:
        raise TypeError("proposal must be exact ObjectSceneAnchorCardProposal")
    restored = ObjectSceneAnchorCardProposal.from_data(proposal.to_data())
    manifests = {
        "side0_positive": _normalized_panel_manifests(
            side0_panel_manifests, "side0 panel manifests"
        ),
        "side1_positive": _normalized_panel_manifests(
            side1_panel_manifests, "side1 panel manifests"
        ),
    }
    if set(manifests["side0_positive"]) & set(manifests["side1_positive"]):
        raise ObjectSceneAnchorCardError("card verification panel aliases overlap")
    for orientation, cards in (
        ("side0_positive", restored.side0_positive),
        ("side1_positive", restored.side1_positive),
    ):
        expected_aliases = tuple(sorted(manifests[orientation]))
        for card in cards:
            if tuple(
                item.panel_alias for item in card.positive_support_citations
            ) != expected_aliases:
                raise ObjectSceneAnchorCardError(
                    "card citations differ from their frozen orientation"
                )
            for citation in card.positive_support_citations:
                manifest = manifests[orientation][citation.panel_alias]
                catalogs, catalogs_digest = _catalogs_for_panel(
                    manifest, card.binding_spec
                )
                if (
                    citation.panel_manifest_digest != manifest.manifest_digest
                    or citation.binding_catalogs_digest != catalogs_digest
                    or citation.object_id not in manifest.by_object_id
                ):
                    raise ObjectSceneAnchorCardError(
                        "card citation panel inventory binding differs"
                    )
                catalog = catalogs[manifest.object_ids.index(citation.object_id)]
                matches = tuple(
                    item
                    for item in catalog.bindings
                    if item.anchor_id == citation.anchor_id
                )
                if len(matches) != 1 or matches[0] != citation.resolved_binding:
                    raise ObjectSceneAnchorCardError(
                        "card citation resolved binding differs"
                    )
    return restored


parse_object_scene_anchor_card_proposal = build_object_scene_anchor_card_proposal


__all__ = (
    "OBJECT_SCENE_ANCHOR_CARD_CITATION_SCHEMA",
    "OBJECT_SCENE_ANCHOR_CARD_DROP_REASON_CODES",
    "OBJECT_SCENE_ANCHOR_CARD_ORIENTATIONS",
    "OBJECT_SCENE_ANCHOR_CARD_PROPOSAL_SCHEMA",
    "OBJECT_SCENE_ANCHOR_CARD_WITNESS_KINDS",
    "OBJECT_SCENE_ANCHOR_CARD_WITNESS_SCHEMA",
    "OBJECT_SCENE_ANCHOR_MAX_CARDS_PER_ORIENTATION",
    "OBJECT_SCENE_ANCHOR_MAX_UNION_WITNESSES",
    "OBJECT_SCENE_ANCHOR_MAX_WITNESSES_PER_CARD",
    "OBJECT_SCENE_ANCHOR_POSITIVE_CITATION_COUNT",
    "OBJECT_SCENE_ANCHOR_PREDICATE_CARD_SCHEMA",
    "OBJECT_SCENE_DROPPED_ANCHOR_CARD_SCHEMA",
    "ObjectSceneAnchorCardError",
    "ObjectSceneAnchorCardProposal",
    "ObjectSceneAnchorCardWitness",
    "ObjectSceneAnchorPositiveSupportCitation",
    "ObjectSceneAnchorPredicateCard",
    "ObjectSceneDroppedAnchorCard",
    "build_object_scene_anchor_card_proposal",
    "parse_object_scene_anchor_card_proposal",
    "verify_object_scene_anchor_card_proposal",
)

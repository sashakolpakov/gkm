"""Closed pure-Python scene predicates over repeated visual transcripts.

This module is the decision authority between the neutral visual frontend and
query evaluation.  It merges two registered-evaluation transcripts without
turning disagreement or failure into absence, enumerates a finite typed
affirmative language in both orientations before roles are used, and never
repairs a failed candidate with a post-hoc Not or polarity flip.  Natural
affirmative categories may be logical complements.  Lean is absent.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_scene_visual_frontend import (
    OBJECT_SCENE_COUNT_OBSERVABLE_IDS,
    OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS,
    ObjectSceneProposalInventory,
    ObjectSceneSoftTagRegistry,
    ObjectSceneTranscriptArtifact,
    ObjectSceneTranscriptMode,
    verify_object_scene_soft_tag_registry,
)
from bongard.prototype_scene_observer import PrototypeSceneObserverStatus
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


SCENE_OBSERVATION_SCHEMA = "gkm.object-bongard-scene-observation.v1"
SCENE_ATOM_SCHEMA = "gkm.object-bongard-scene-atom.v1"
SCENE_FORMULA_SCHEMA = "gkm.object-bongard-scene-formula.v1"
SCENE_CANDIDATE_SCHEMA = "gkm.object-bongard-scene-candidate.v1"
SCENE_VERSION_SPACE_SCHEMA = "gkm.object-bongard-scene-version-space.v1"
SCENE_VERSION_SPACES_SCHEMA = "gkm.object-bongard-scene-version-spaces.v1"
SCENE_LANGUAGE_SCHEMA = "gkm.object-bongard-scene-language.v1"
SCENE_CALIBRATION_BUNDLE_SCHEMA = "gkm.bongard-scene-predicate-calibration-ir-bundle.v1"
SCENE_ALGORITHM_ID = "bongard.scene-predicate/typed-positive-version-space-v1"
SCENE_MAX_RANK_SLATE = 64
SCENE_MAX_ENUMERATED_FORMULAS = 250_000

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}\Z")
_TAG = re.compile(r"tag_[0-9]{4}\Z")

_QUALITATIVE_PHRASES = {
    "triangle_like": "resembles a triangle",
    "quadrilateral_like": "resembles a quadrilateral",
    "sector_like": "resembles a circular sector",
    "bird_like": "resembles a bird or flying bird silhouette",
    "open_contour": "has an open contour",
    "closed_boundary": "has a closed boundary",
    "pointed": "has a pointed part",
    "thin_elongated": "has a thin elongated part",
    "necked": "has a narrow neck",
    "mismatched_parts": "has visibly mismatched parts",
    "unequal_part_sizes": "has parts of unequal size",
    "unequal_edge_lengths": "has unequal edge lengths",
    "reflection_symmetry": "has reflection symmetry",
    "bilateral_layout": "has a bilateral layout",
    "oblique": "contains oblique directions",
    "parallel": "contains parallel directions",
    "perpendicular": "contains perpendicular directions",
    "crossing": "contains a crossing",
    "internal_marks": "contains internal marks",
    "paired_sector_mismatch": "contains a mismatched pair of sector-like parts",
    "triangle_with_three_lines": "contains a triangle associated with three lines",
}
if set(_QUALITATIVE_PHRASES) != set(OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS):
    raise RuntimeError("fixed qualitative phrase catalog differs")


class ObjectBongardScenePredicateIRError(ValueError):
    """A repeated observation, formula, or version space is malformed."""


class SceneScope(str, Enum):
    ENTITY = "entity"
    PAIR = "pair"
    PANEL = "panel"


class SceneAtomKind(str, Enum):
    QUALITATIVE = "qualitative"
    REGISTERED_TAG = "registered_tag"
    COUNT = "count"
    GEOMETRY = "geometry"
    PAIR_RELATION = "pair_relation"
    PANEL_COUNT = "panel_count"


class SceneNumericUnit(str, Enum):
    COUNT = "count"
    PIXEL_AREA = "pixel_area"
    PIXEL_LENGTH = "pixel_length"
    Q16_COORDINATE = "q16_coordinate"


class SceneComparison(str, Enum):
    EQUAL = "equal"
    AT_LEAST = "at_least"
    AT_MOST = "at_most"


class SceneGeometryPreset(str, Enum):
    SINGLE_COMPONENT = "single_component"
    MULTIPLE_COMPONENTS = "multiple_components"
    WIDER_THAN_TALL = "wider_than_tall"
    TALLER_THAN_WIDE = "taller_than_wide"


class ScenePairRelation(str, Enum):
    BBOX_INTERSECTS = "bbox_intersects"
    BBOX_DISJOINT = "bbox_disjoint"
    HORIZONTALLY_SEPARATED_BBOXES = "horizontally_separated_bboxes"
    VERTICALLY_SEPARATED_BBOXES = "vertically_separated_bboxes"


class SceneFormulaNode(str, Enum):
    ATOM = "atom"
    AND = "and"
    QUANTIFIED = "quantified"


class SceneQuantifier(str, Enum):
    EXISTS = "exists"
    ALL = "all"
    COUNT = "count"


class SceneOrientation(str, Enum):
    GROUP0_POSITIVE = "group0_positive"
    GROUP1_POSITIVE = "group1_positive"


class SceneGapKind(str, Enum):
    LANGUAGE_GAP = "language_gap"
    WITNESS_GAP = "witness_gap"


class SceneLanguageCapacityGap(ObjectBongardScenePredicateIRError):
    """The declared complete finite grammar exceeds its explicit resource cap."""

    kind = SceneGapKind.LANGUAGE_GAP

    def __init__(self, prospective_formula_count: int) -> None:
        self.prospective_formula_count = prospective_formula_count
        super().__init__(f"complete formula inventory {prospective_formula_count} exceeds cap {SCENE_MAX_ENUMERATED_FORMULAS}")


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_or_decision": False,
        "lean_required_for_replay": False,
        "syntactic_negation_operator": False,
        "post_hoc_negation_repair": False,
        "post_hoc_polarity_flip": False,
        "natural_affirmative_complement_equivalents_predeclared": True,
        "both_orientations_predeclared_before_roles": True,
        "arbitrary_code": False,
    }


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value) or set(value) != expected:
        raise ObjectBongardScenePredicateIRError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectBongardScenePredicateIRError(f"{label} must be a raw SHA-256")
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ObjectBongardScenePredicateIRError(f"{label} is not a bounded identifier")
    return value


def object_bongard_scene_predicate_ir_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def scene_and(values: Sequence[Disposition]) -> Disposition:
    """Error-dominant Strong-Kleene conjunction."""
    row = tuple(values)
    if not row:
        return Disposition.PRESENT
    if Disposition.ERROR in row:
        return Disposition.ERROR
    if Disposition.CERTIFIED_ABSENT in row:
        return Disposition.CERTIFIED_ABSENT
    if all(item is Disposition.PRESENT for item in row):
        return Disposition.PRESENT
    return Disposition.INDETERMINATE


def scene_or(values: Sequence[Disposition]) -> Disposition:
    """Error-dominant Strong-Kleene disjunction."""
    row = tuple(values)
    if not row:
        return Disposition.CERTIFIED_ABSENT
    if Disposition.ERROR in row:
        return Disposition.ERROR
    if Disposition.PRESENT in row:
        return Disposition.PRESENT
    if all(item is Disposition.CERTIFIED_ABSENT for item in row):
        return Disposition.CERTIFIED_ABSENT
    return Disposition.INDETERMINATE


def merge_repeated_disposition(first: Disposition | None, second: Disposition | None) -> Disposition:
    """P/P=P, A/A=A, disagreements/indeterminacy=I, missing/error=E."""
    if first is None or second is None or Disposition.ERROR in (first, second):
        return Disposition.ERROR
    if first is second is Disposition.PRESENT:
        return Disposition.PRESENT
    if first is second is Disposition.CERTIFIED_ABSENT:
        return Disposition.CERTIFIED_ABSENT
    return Disposition.INDETERMINATE


@dataclass(frozen=True, order=True, slots=True)
class SceneNumericInterval:
    unit: SceneNumericUnit
    lower: int
    upper: int

    def __post_init__(self) -> None:
        if not isinstance(self.unit, SceneNumericUnit) or type(self.lower) is not int or type(self.upper) is not int or self.lower < 0 or self.lower > self.upper:
            raise ObjectBongardScenePredicateIRError("numeric interval differs")

    def to_data(self) -> dict[str, object]:
        return {"unit": self.unit.value, "lower": self.lower, "upper": self.upper}

    @classmethod
    def from_data(cls, value: object) -> "SceneNumericInterval":
        raw = _fields(value, {"unit", "lower", "upper"}, "numeric interval")
        return cls(SceneNumericUnit(raw["unit"]), raw["lower"], raw["upper"])


def merge_repeated_interval(first: SceneNumericInterval | None, second: SceneNumericInterval | None) -> tuple[Disposition, SceneNumericInterval | None]:
    """Intersect repeated typed intervals; empty overlap is indeterminate."""
    if first is None or second is None:
        return Disposition.INDETERMINATE, None
    if first.unit is not second.unit:
        return Disposition.ERROR, None
    lower, upper = max(first.lower, second.lower), min(first.upper, second.upper)
    if lower > upper:
        return Disposition.INDETERMINATE, None
    return Disposition.PRESENT, SceneNumericInterval(first.unit, lower, upper)


@dataclass(frozen=True, order=True, slots=True)
class SceneMergedCell:
    observable_id: str
    disposition: Disposition
    interval: SceneNumericInterval | None
    source_cell_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.observable_id, "observable ID")
        if not isinstance(self.disposition, Disposition):
            raise TypeError("cell disposition differs")
        if self.interval is not None and self.disposition is not Disposition.PRESENT:
            raise ObjectBongardScenePredicateIRError("only a present numeric cell carries an interval")
        if self.source_cell_digests != tuple(sorted(set(self.source_cell_digests))) or any(_DIGEST.fullmatch(item) is None for item in self.source_cell_digests):
            raise ObjectBongardScenePredicateIRError("source cell commitments differ")

    def to_data(self) -> dict[str, object]:
        return {"observable_id": self.observable_id, "disposition": self.disposition.value, "interval": None if self.interval is None else self.interval.to_data(), "source_cell_digests": list(self.source_cell_digests)}

    @classmethod
    def from_data(cls, value: object) -> "SceneMergedCell":
        raw = _fields(value, {"observable_id", "disposition", "interval", "source_cell_digests"}, "merged cell")
        if not isinstance(raw["source_cell_digests"], list):
            raise ObjectBongardScenePredicateIRError("merged cell source digests differ")
        return cls(raw["observable_id"], Disposition(raw["disposition"]), None if raw["interval"] is None else SceneNumericInterval.from_data(raw["interval"]), tuple(raw["source_cell_digests"]))


@dataclass(frozen=True, slots=True)
class SceneEntityObservation:
    object_id: str
    crop_receipt_digest: str
    bbox_q16: tuple[int, int, int, int]
    area_pixels: int
    component_count: int
    emergence_gap_pixels: int
    overlap_object_ids: tuple[str, ...]
    qualitative_cells: tuple[SceneMergedCell, ...]
    count_cells: tuple[SceneMergedCell, ...]
    registered_tag_cells: tuple[SceneMergedCell, ...]

    def __post_init__(self) -> None:
        _identifier(self.object_id, "object ID")
        _digest(self.crop_receipt_digest, "crop receipt digest")
        if len(self.bbox_q16) != 4 or any(type(item) is not int or not 0 <= item <= 65535 for item in self.bbox_q16) or self.bbox_q16[0] >= self.bbox_q16[2] or self.bbox_q16[1] >= self.bbox_q16[3]:
            raise ObjectBongardScenePredicateIRError("entity Q16 geometry differs")
        if any(type(item) is not int or item < 0 for item in (self.area_pixels, self.component_count, self.emergence_gap_pixels)) or self.area_pixels < 1 or self.component_count < 1:
            raise ObjectBongardScenePredicateIRError("entity exact geometry differs")
        if self.overlap_object_ids != tuple(sorted(set(self.overlap_object_ids))) or self.object_id in self.overlap_object_ids:
            raise ObjectBongardScenePredicateIRError("entity overlap graph differs")
        expected = (OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS, OBJECT_SCENE_COUNT_OBSERVABLE_IDS)
        if tuple(item.observable_id for item in self.qualitative_cells) != expected[0] or tuple(item.observable_id for item in self.count_cells) != expected[1]:
            raise ObjectBongardScenePredicateIRError("entity fixed cell catalog differs")
        if tuple(item.observable_id for item in self.registered_tag_cells) != tuple(f"tag_{i:04d}" for i in range(len(self.registered_tag_cells))):
            raise ObjectBongardScenePredicateIRError("entity registered tag catalog differs")

    def to_data(self) -> dict[str, object]:
        return {"object_id": self.object_id, "crop_receipt_digest": self.crop_receipt_digest, "bbox_q16": list(self.bbox_q16), "area_pixels": self.area_pixels, "component_count": self.component_count, "emergence_gap_pixels": self.emergence_gap_pixels, "overlap_object_ids": list(self.overlap_object_ids), "qualitative_cells": [x.to_data() for x in self.qualitative_cells], "count_cells": [x.to_data() for x in self.count_cells], "registered_tag_cells": [x.to_data() for x in self.registered_tag_cells]}

    @classmethod
    def from_data(cls, value: object) -> "SceneEntityObservation":
        raw = _fields(value, {"object_id", "crop_receipt_digest", "bbox_q16", "area_pixels", "component_count", "emergence_gap_pixels", "overlap_object_ids", "qualitative_cells", "count_cells", "registered_tag_cells"}, "entity observation")
        if any(not isinstance(raw[k], list) for k in ("bbox_q16", "overlap_object_ids", "qualitative_cells", "count_cells", "registered_tag_cells")):
            raise ObjectBongardScenePredicateIRError("entity observation arrays differ")
        return cls(raw["object_id"], raw["crop_receipt_digest"], tuple(raw["bbox_q16"]), raw["area_pixels"], raw["component_count"], raw["emergence_gap_pixels"], tuple(raw["overlap_object_ids"]), tuple(SceneMergedCell.from_data(x) for x in raw["qualitative_cells"]), tuple(SceneMergedCell.from_data(x) for x in raw["count_cells"]), tuple(SceneMergedCell.from_data(x) for x in raw["registered_tag_cells"]))


def _observation_content(value: "ScenePanelObservation") -> dict[str, object]:
    return {
        "schema": SCENE_OBSERVATION_SCHEMA,
        "panel_id": value.panel_id,
        "panel_digest": value.panel_digest,
        "inventory_digest": value.inventory_digest,
        "registry_digest": value.registry_digest,
        "observation_mode": value.observation_mode,
        "source_artifact_digests": list(value.source_artifact_digests),
        "source_transcript_digests": list(value.source_transcript_digests),
        "disposition": value.disposition.value,
        "entities": [item.to_data() for item in value.entities],
        "merge_policy": "two-registered-calls-exact-agreement-or-indeterminate" if value.observation_mode == "repeated_registered_merge" else "one-registered-call-query-or-repeatability-leg",
        "geometry_binding": "stable-object-ids-crop-receipts-q16-overlap-graph",
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ScenePanelObservation:
    panel_id: str
    panel_digest: str
    inventory_digest: str
    registry_digest: str
    observation_mode: str
    source_artifact_digests: tuple[str, ...]
    source_transcript_digests: tuple[str, ...]
    disposition: Disposition
    entities: tuple[SceneEntityObservation, ...]
    observation_digest: str

    def __post_init__(self) -> None:
        _identifier(self.panel_id, "panel ID")
        for value, label in ((self.panel_digest, "panel digest"), (self.inventory_digest, "inventory digest"), (self.registry_digest, "registry digest"), (self.observation_digest, "observation digest")):
            _digest(value, label)
        expected_artifacts = {"single_registered": 1, "repeated_registered_merge": 2}.get(self.observation_mode)
        if expected_artifacts is None or len(self.source_artifact_digests) != expected_artifacts or len(set(self.source_artifact_digests)) != expected_artifacts or self.source_artifact_digests != tuple(sorted(self.source_artifact_digests)):
            raise ObjectBongardScenePredicateIRError("panel registered-call commitments differ")
        if self.source_transcript_digests != tuple(sorted(self.source_transcript_digests)):
            raise ObjectBongardScenePredicateIRError("transcript commitments differ")
        for value in (*self.source_artifact_digests, *self.source_transcript_digests):
            _digest(value, "source digest")
        if self.disposition not in (Disposition.PRESENT, Disposition.ERROR):
            raise ObjectBongardScenePredicateIRError("panel merge disposition differs")
        if tuple(item.object_id for item in self.entities) != tuple(f"object_{index:04d}" for index in range(len(self.entities))):
            raise ObjectBongardScenePredicateIRError("stable object IDs differ")
        by_id = {item.object_id: item for item in self.entities}
        for item in self.entities:
            for other in item.overlap_object_ids:
                if other not in by_id or item.object_id not in by_id[other].overlap_object_ids:
                    raise ObjectBongardScenePredicateIRError("overlap graph is not symmetric")
        if self.disposition is Disposition.ERROR:
            cells = (cell for entity in self.entities for cells in (entity.qualitative_cells, entity.count_cells, entity.registered_tag_cells) for cell in cells)
            if any(cell.disposition is not Disposition.ERROR for cell in cells):
                raise ObjectBongardScenePredicateIRError("failed panel contains decisive visual cells")
        if self.observation_digest != canonical_digest(_observation_content(self)):
            raise ObjectBongardScenePredicateIRError("panel observation digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_observation_content(self), "observation_digest": self.observation_digest}

    @classmethod
    def from_data(cls, value: object) -> "ScenePanelObservation":
        expected = {"schema", "panel_id", "panel_digest", "inventory_digest", "registry_digest", "observation_mode", "source_artifact_digests", "source_transcript_digests", "disposition", "entities", "merge_policy", "geometry_binding", *_authority_data(), "observation_digest"}
        raw = _fields(value, expected, "scene panel observation")
        merge_policy = "two-registered-calls-exact-agreement-or-indeterminate" if raw.get("observation_mode") == "repeated_registered_merge" else "one-registered-call-query-or-repeatability-leg"
        if raw["schema"] != SCENE_OBSERVATION_SCHEMA or raw["merge_policy"] != merge_policy or raw["geometry_binding"] != "stable-object-ids-crop-receipts-q16-overlap-graph" or any(raw[key] != val for key, val in _authority_data().items()) or any(not isinstance(raw[key], list) for key in ("source_artifact_digests", "source_transcript_digests", "entities")):
            raise ObjectBongardScenePredicateIRError("scene panel observation policy differs")
        result = cls(raw["panel_id"], raw["panel_digest"], raw["inventory_digest"], raw["registry_digest"], raw["observation_mode"], tuple(raw["source_artifact_digests"]), tuple(raw["source_transcript_digests"]), Disposition(raw["disposition"]), tuple(SceneEntityObservation.from_data(item) for item in raw["entities"]), raw["observation_digest"])
        if result.to_data() != dict(raw):
            raise ObjectBongardScenePredicateIRError("scene panel observation is not canonical")
        return result


def _merge_visual_cell(first: object | None, second: object | None, *, observable_id: str, numeric: bool) -> SceneMergedCell:
    digests = tuple(sorted({getattr(item, "cell_digest") for item in (first, second) if isinstance(getattr(item, "cell_digest", None), str)}))
    if first is None or second is None:
        return SceneMergedCell(observable_id, Disposition.ERROR, None, digests)
    if numeric:
        if getattr(first, "state", None) != "measured" or getattr(second, "state", None) != "measured":
            return SceneMergedCell(observable_id, Disposition.INDETERMINATE, None, digests)
        first_raw, second_raw = getattr(first, "interval", None), getattr(second, "interval", None)
        if first_raw is None or second_raw is None:
            return SceneMergedCell(observable_id, Disposition.ERROR, None, digests)
        state, interval = merge_repeated_interval(SceneNumericInterval(SceneNumericUnit.COUNT, first_raw.lower, first_raw.upper), SceneNumericInterval(SceneNumericUnit.COUNT, second_raw.lower, second_raw.upper))
        return SceneMergedCell(observable_id, state, interval, digests)
    state = merge_repeated_disposition(getattr(first, "disposition", None), getattr(second, "disposition", None))
    return SceneMergedCell(observable_id, state, None, digests)


def _error_entity(receipt: object, tag_ids: tuple[str, ...]) -> SceneEntityObservation:
    error = lambda observable: SceneMergedCell(observable, Disposition.ERROR, None, ())
    bbox = receipt.bbox_q16
    return SceneEntityObservation(receipt.object_id, receipt.receipt_digest, (bbox.x0, bbox.y0, bbox.x1, bbox.y1), receipt.union_area_pixels, receipt.component_count, receipt.emergence_gap_pixels, receipt.overlap_object_ids, tuple(error(item) for item in OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS), tuple(error(item) for item in OBJECT_SCENE_COUNT_OBSERVABLE_IDS), tuple(error(item) for item in tag_ids))


def adapt_object_scene_registered_pair(
    panel_id: str,
    first: ObjectSceneTranscriptArtifact,
    second: ObjectSceneTranscriptArtifact,
) -> ScenePanelObservation:
    """Merge two frozen registered-evaluation artifacts for one exact panel."""

    if not isinstance(first, ObjectSceneTranscriptArtifact) or not isinstance(second, ObjectSceneTranscriptArtifact):
        raise TypeError("registered pair must contain transcript artifacts")
    first.assert_untampered()
    second.assert_untampered()
    if first.artifact_digest == second.artifact_digest or first.observation_context_digest == second.observation_context_digest:
        raise ObjectBongardScenePredicateIRError("repeatability requires two distinct registered calls")
    if first.mode is not ObjectSceneTranscriptMode.REGISTERED_EVALUATION or second.mode is not ObjectSceneTranscriptMode.REGISTERED_EVALUATION:
        raise ObjectBongardScenePredicateIRError("decisive pair must be registered evaluation")
    if first.panel_digest != second.panel_digest or first.inventory_digest != second.inventory_digest or first.registry_digest != second.registry_digest or first.inventory != second.inventory:
        raise ObjectBongardScenePredicateIRError("registered passes do not bind one frozen panel inventory")
    if not isinstance(first.registry, ObjectSceneSoftTagRegistry) or first.registry != second.registry:
        raise ObjectBongardScenePredicateIRError("registered passes do not bind one frozen registry")
    inventory = first.inventory
    tag_ids = tuple(item.tag_id for item in first.registry.tags)
    successful = first.status is PrototypeSceneObserverStatus.SUCCESS and second.status is PrototypeSceneObserverStatus.SUCCESS
    entities: list[SceneEntityObservation] = []
    if not inventory.objects:
        disposition = Disposition.ERROR
        transcript_digests = tuple(sorted(item.transcript.transcript_digest for item in (first, second) if item.transcript is not None))
    elif not successful:
        entities = [_error_entity(item, tag_ids) for item in inventory.objects]
        disposition = Disposition.ERROR
        transcript_digests: tuple[str, ...] = tuple(sorted(item.transcript.transcript_digest for item in (first, second) if item.transcript is not None))
    else:
        assert first.transcript is not None and second.transcript is not None
        first_rows = {item.object_id: item for item in first.transcript.objects}
        second_rows = {item.object_id: item for item in second.transcript.objects}
        for receipt in inventory.objects:
            row_a, row_b = first_rows.get(receipt.object_id), second_rows.get(receipt.object_id)
            if row_a is None or row_b is None or row_a.crop_receipt_digest != receipt.receipt_digest or row_b.crop_receipt_digest != receipt.receipt_digest:
                entities.append(_error_entity(receipt, tag_ids))
                continue
            qa, qb = {x.observable_id: x for x in row_a.qualitative_cells}, {x.observable_id: x for x in row_b.qualitative_cells}
            ca, cb = {x.observable_id: x for x in row_a.count_cells}, {x.observable_id: x for x in row_b.count_cells}
            ta, tb = {x.tag_id: x for x in row_a.registered_tag_cells}, {x.tag_id: x for x in row_b.registered_tag_cells}
            bbox = receipt.bbox_q16
            entities.append(SceneEntityObservation(receipt.object_id, receipt.receipt_digest, (bbox.x0, bbox.y0, bbox.x1, bbox.y1), receipt.union_area_pixels, receipt.component_count, receipt.emergence_gap_pixels, receipt.overlap_object_ids, tuple(_merge_visual_cell(qa.get(item), qb.get(item), observable_id=item, numeric=False) for item in OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS), tuple(_merge_visual_cell(ca.get(item), cb.get(item), observable_id=item, numeric=True) for item in OBJECT_SCENE_COUNT_OBSERVABLE_IDS), tuple(_merge_visual_cell(ta.get(item), tb.get(item), observable_id=item, numeric=False) for item in tag_ids)))
        disposition = Disposition.ERROR if any(cell.disposition is Disposition.ERROR for entity in entities for cells in (entity.qualitative_cells, entity.count_cells, entity.registered_tag_cells) for cell in cells) else Disposition.PRESENT
        if disposition is Disposition.ERROR:
            entities = [_error_entity(item, tag_ids) for item in inventory.objects]
        transcript_digests = tuple(sorted((first.transcript.transcript_digest, second.transcript.transcript_digest)))
    values = {"panel_id": _identifier(panel_id, "panel ID"), "panel_digest": inventory.panel_digest, "inventory_digest": inventory.inventory_digest, "registry_digest": first.registry.registry_digest, "observation_mode": "repeated_registered_merge", "source_artifact_digests": tuple(sorted((first.artifact_digest, second.artifact_digest))), "source_transcript_digests": transcript_digests, "disposition": disposition, "entities": tuple(entities)}
    provisional = object.__new__(ScenePanelObservation)
    for key, item in values.items():
        object.__setattr__(provisional, key, item)
    return ScenePanelObservation(**values, observation_digest=canonical_digest(_observation_content(provisional)))


def _single_visual_cell(value: object | None, *, observable_id: str, numeric: bool) -> SceneMergedCell:
    digest = getattr(value, "cell_digest", None)
    sources = () if not isinstance(digest, str) else (digest,)
    if value is None:
        return SceneMergedCell(observable_id, Disposition.ERROR, None, sources)
    if numeric:
        if getattr(value, "state", None) == "indeterminate":
            return SceneMergedCell(observable_id, Disposition.INDETERMINATE, None, sources)
        interval = getattr(value, "interval", None)
        if getattr(value, "state", None) != "measured" or interval is None:
            return SceneMergedCell(observable_id, Disposition.ERROR, None, sources)
        return SceneMergedCell(observable_id, Disposition.PRESENT, SceneNumericInterval(SceneNumericUnit.COUNT, interval.lower, interval.upper), sources)
    disposition = getattr(value, "disposition", None)
    if not isinstance(disposition, Disposition):
        return SceneMergedCell(observable_id, Disposition.ERROR, None, sources)
    return SceneMergedCell(observable_id, disposition, None, sources)


def adapt_object_scene_registered_single(panel_id: str, artifact: ObjectSceneTranscriptArtifact) -> ScenePanelObservation:
    """Adapt one registered query/repeatability call without faking a repeat."""
    if not isinstance(artifact, ObjectSceneTranscriptArtifact): raise TypeError("registered single must be a transcript artifact")
    artifact.assert_untampered()
    if artifact.mode is not ObjectSceneTranscriptMode.REGISTERED_EVALUATION or not isinstance(artifact.registry, ObjectSceneSoftTagRegistry): raise ObjectBongardScenePredicateIRError("single decisive call must be registered evaluation")
    inventory, registry = artifact.inventory, artifact.registry
    tag_ids = tuple(item.tag_id for item in registry.tags)
    entities: list[SceneEntityObservation] = []
    if not inventory.objects:
        disposition = Disposition.ERROR
        transcript_digests = () if artifact.transcript is None else (artifact.transcript.transcript_digest,)
    elif artifact.status is not PrototypeSceneObserverStatus.SUCCESS or artifact.transcript is None:
        entities = [_error_entity(item, tag_ids) for item in inventory.objects]
        disposition = Disposition.ERROR
        transcript_digests: tuple[str, ...] = ()
    else:
        rows = {item.object_id: item for item in artifact.transcript.objects}
        for receipt in inventory.objects:
            row = rows.get(receipt.object_id)
            if row is None or row.crop_receipt_digest != receipt.receipt_digest:
                entities.append(_error_entity(receipt, tag_ids)); continue
            q = {item.observable_id: item for item in row.qualitative_cells}; c = {item.observable_id: item for item in row.count_cells}; t = {item.tag_id: item for item in row.registered_tag_cells}
            bbox = receipt.bbox_q16
            entities.append(SceneEntityObservation(receipt.object_id, receipt.receipt_digest, (bbox.x0, bbox.y0, bbox.x1, bbox.y1), receipt.union_area_pixels, receipt.component_count, receipt.emergence_gap_pixels, receipt.overlap_object_ids, tuple(_single_visual_cell(q.get(item), observable_id=item, numeric=False) for item in OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS), tuple(_single_visual_cell(c.get(item), observable_id=item, numeric=True) for item in OBJECT_SCENE_COUNT_OBSERVABLE_IDS), tuple(_single_visual_cell(t.get(item), observable_id=item, numeric=False) for item in tag_ids)))
        disposition = Disposition.ERROR if any(cell.disposition is Disposition.ERROR for entity in entities for cells in (entity.qualitative_cells, entity.count_cells, entity.registered_tag_cells) for cell in cells) else Disposition.PRESENT
        if disposition is Disposition.ERROR: entities = [_error_entity(item, tag_ids) for item in inventory.objects]
        transcript_digests = (artifact.transcript.transcript_digest,)
    values = {"panel_id": _identifier(panel_id, "panel ID"), "panel_digest": inventory.panel_digest, "inventory_digest": inventory.inventory_digest, "registry_digest": registry.registry_digest, "observation_mode": "single_registered", "source_artifact_digests": (artifact.artifact_digest,), "source_transcript_digests": transcript_digests, "disposition": disposition, "entities": tuple(entities)}
    provisional = object.__new__(ScenePanelObservation)
    for key, item in values.items(): object.__setattr__(provisional, key, item)
    return ScenePanelObservation(**values, observation_digest=canonical_digest(_observation_content(provisional)))


@dataclass(frozen=True, order=True, slots=True)
class SceneNumericBoundary:
    boundary_id: str
    observable_id: str
    unit: SceneNumericUnit
    comparison: SceneComparison
    value: int
    source_observation_digests: tuple[str, ...]
    boundary_digest: str

    def __post_init__(self) -> None:
        _identifier(self.boundary_id, "boundary ID")
        _identifier(self.observable_id, "boundary observable ID")
        if not isinstance(self.unit, SceneNumericUnit) or not isinstance(self.comparison, SceneComparison) or type(self.value) is not int or self.value < 0:
            raise ObjectBongardScenePredicateIRError("typed boundary differs")
        allowed = set(OBJECT_SCENE_COUNT_OBSERVABLE_IDS) | {"entity_count", "matching_entity_count", "matching_pair_count"}
        if self.observable_id not in allowed or self.unit is not SceneNumericUnit.COUNT:
            raise ObjectBongardScenePredicateIRError("boundary observable/unit combination differs")
        if self.source_observation_digests != tuple(sorted(set(self.source_observation_digests))) or not self.source_observation_digests:
            raise ObjectBongardScenePredicateIRError("boundary provenance differs")
        for item in self.source_observation_digests:
            _digest(item, "boundary source observation digest")
        if self.boundary_digest != canonical_digest(self.content_data()):
            raise ObjectBongardScenePredicateIRError("boundary digest differs")

    def content_data(self) -> dict[str, object]:
        return {"boundary_id": self.boundary_id, "observable_id": self.observable_id, "unit": self.unit.value, "comparison": self.comparison.value, "value": self.value, "source_observation_digests": list(self.source_observation_digests), "threshold_origin": "finite-support-derived-boundary"}

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "boundary_digest": self.boundary_digest}

    @classmethod
    def create(cls, boundary_id: str, observable_id: str, unit: SceneNumericUnit, comparison: SceneComparison, value: int, sources: Sequence[str]) -> "SceneNumericBoundary":
        values = {"boundary_id": boundary_id, "observable_id": observable_id, "unit": unit, "comparison": comparison, "value": value, "source_observation_digests": tuple(sorted(set(sources)))}
        provisional = object.__new__(cls)
        for key, item in values.items(): object.__setattr__(provisional, key, item)
        return cls(**values, boundary_digest=canonical_digest(provisional.content_data()))

    @classmethod
    def from_data(cls, value: object) -> "SceneNumericBoundary":
        raw = _fields(value, {"boundary_id", "observable_id", "unit", "comparison", "value", "source_observation_digests", "threshold_origin", "boundary_digest"}, "numeric boundary")
        if raw["threshold_origin"] != "finite-support-derived-boundary" or not isinstance(raw["source_observation_digests"], list):
            raise ObjectBongardScenePredicateIRError("numeric boundary policy differs")
        result = cls(raw["boundary_id"], raw["observable_id"], SceneNumericUnit(raw["unit"]), SceneComparison(raw["comparison"]), raw["value"], tuple(raw["source_observation_digests"]), raw["boundary_digest"])
        if result.to_data() != dict(raw): raise ObjectBongardScenePredicateIRError("numeric boundary is not canonical")
        return result


def _atom_content(value: "SceneAtom") -> dict[str, object]:
    return {"schema": SCENE_ATOM_SCHEMA, "scope": value.scope.value, "kind": value.kind.value, "observable_id": value.observable_id, "boundary_id": value.boundary_id, "typed_affirmative_atom": True, "literal_threshold": False, "syntactically_negated": False, **_authority_data()}


@dataclass(frozen=True, order=True, slots=True)
class SceneAtom:
    scope: SceneScope
    kind: SceneAtomKind
    observable_id: str
    boundary_id: str | None
    atom_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.scope, SceneScope) or not isinstance(self.kind, SceneAtomKind): raise TypeError("atom enum differs")
        _identifier(self.observable_id, "atom observable ID")
        expected_scope = {
            SceneAtomKind.QUALITATIVE: SceneScope.ENTITY,
            SceneAtomKind.REGISTERED_TAG: SceneScope.ENTITY,
            SceneAtomKind.COUNT: SceneScope.ENTITY,
            SceneAtomKind.GEOMETRY: SceneScope.ENTITY,
            SceneAtomKind.PAIR_RELATION: SceneScope.PAIR,
            SceneAtomKind.PANEL_COUNT: SceneScope.PANEL,
        }[self.kind]
        if self.scope is not expected_scope: raise ObjectBongardScenePredicateIRError("atom scope/kind differ")
        if self.kind is SceneAtomKind.QUALITATIVE and self.observable_id not in OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS: raise ObjectBongardScenePredicateIRError("qualitative atom is unregistered")
        if self.kind is SceneAtomKind.REGISTERED_TAG and _TAG.fullmatch(self.observable_id) is None: raise ObjectBongardScenePredicateIRError("tag atom is unregistered")
        if self.kind is SceneAtomKind.COUNT and self.observable_id not in OBJECT_SCENE_COUNT_OBSERVABLE_IDS: raise ObjectBongardScenePredicateIRError("count atom is unregistered")
        if self.kind is SceneAtomKind.GEOMETRY:
            try: SceneGeometryPreset(self.observable_id)
            except ValueError as exc: raise ObjectBongardScenePredicateIRError("geometry atom is not a preset") from exc
        if self.kind is SceneAtomKind.PAIR_RELATION:
            try: ScenePairRelation(self.observable_id)
            except ValueError as exc: raise ObjectBongardScenePredicateIRError("pair atom is not a preset") from exc
        if self.kind is SceneAtomKind.PANEL_COUNT and self.observable_id != "entity_count":
            raise ObjectBongardScenePredicateIRError("panel count atom is not a preset")
        needs_boundary = self.kind in (SceneAtomKind.COUNT, SceneAtomKind.PANEL_COUNT)
        if needs_boundary != isinstance(self.boundary_id, str): raise ObjectBongardScenePredicateIRError("atom boundary binding differs")
        if self.boundary_id is not None: _identifier(self.boundary_id, "atom boundary ID")
        if self.atom_digest != canonical_digest(_atom_content(self)): raise ObjectBongardScenePredicateIRError("atom digest differs")

    @classmethod
    def create(cls, scope: SceneScope, kind: SceneAtomKind, observable_id: str, boundary_id: str | None = None) -> "SceneAtom":
        provisional = object.__new__(cls)
        for key, item in (("scope", scope), ("kind", kind), ("observable_id", observable_id), ("boundary_id", boundary_id)): object.__setattr__(provisional, key, item)
        return cls(scope, kind, observable_id, boundary_id, canonical_digest(_atom_content(provisional)))

    def to_data(self) -> dict[str, object]: return {**_atom_content(self), "atom_digest": self.atom_digest}

    @classmethod
    def from_data(cls, value: object) -> "SceneAtom":
        expected = {"schema", "scope", "kind", "observable_id", "boundary_id", "typed_affirmative_atom", "literal_threshold", "syntactically_negated", *_authority_data(), "atom_digest"}
        raw = _fields(value, expected, "scene atom")
        if raw["schema"] != SCENE_ATOM_SCHEMA or raw["typed_affirmative_atom"] is not True or raw["literal_threshold"] is not False or raw["syntactically_negated"] is not False or any(raw[k] != v for k, v in _authority_data().items()): raise ObjectBongardScenePredicateIRError("scene atom policy differs")
        result = cls(SceneScope(raw["scope"]), SceneAtomKind(raw["kind"]), raw["observable_id"], raw["boundary_id"], raw["atom_digest"])
        if result.to_data() != dict(raw): raise ObjectBongardScenePredicateIRError("scene atom is not canonical")
        return result


def _language_content(value: "ScenePredicateLanguage") -> dict[str, object]:
    return {
        "schema": SCENE_LANGUAGE_SCHEMA,
        "algorithm_id": SCENE_ALGORITHM_ID,
        "registry_digest": value.registry_digest,
        "registered_tag_ids": list(value.registered_tag_ids),
        "support_observation_digests": list(value.support_observation_digests),
        "boundaries": [item.to_data() for item in value.boundaries],
        "grammar": {
            "entity_atoms": ["fixed_qualitative", "registered_soft_tag", "typed_count_boundary", "exact_geometry_preset"],
            "pair_atoms": ["exact_pair_relation_preset"],
            "panel_atoms": ["typed_entity_count_boundary"],
            "quantifiers": ["exists", "all", "count"],
            "count_quantifier_body": "one-positive-atom",
            "same_entity_conjunction": "exactly-two-distinct-qualitative-tag-or-geometry-atoms",
            "connectives": ["and"],
            "or_is_only_exists_aggregation": True,
            "syntactic_not_operator": False,
            "post_hoc_polarity_flip": False,
            "both_orientations_predeclared_before_roles": True,
            "natural_affirmative_complement_equivalents_may_coexist": True,
            "literal_thresholds": False,
            "candidate_numeric_comparisons": ["at_least_positive", "equal_positive"],
            "at_most_or_zero_threshold_candidate": False,
            "arbitrary_code": False,
        },
        "candidate_order": "complexity-then-canonical-formula-digest-then-orientation",
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ScenePredicateLanguage:
    registry_digest: str
    registered_tag_ids: tuple[str, ...]
    support_observation_digests: tuple[str, ...]
    boundaries: tuple[SceneNumericBoundary, ...]
    language_digest: str

    def __post_init__(self) -> None:
        _digest(self.registry_digest, "language registry digest")
        if self.registered_tag_ids != tuple(f"tag_{index:04d}" for index in range(len(self.registered_tag_ids))): raise ObjectBongardScenePredicateIRError("language tag catalog differs")
        if self.support_observation_digests != tuple(sorted(set(self.support_observation_digests))) or not self.support_observation_digests: raise ObjectBongardScenePredicateIRError("language support commitments differ")
        for item in self.support_observation_digests: _digest(item, "language support digest")
        if tuple(item.boundary_id for item in self.boundaries) != tuple(f"boundary_{index:05d}" for index in range(len(self.boundaries))): raise ObjectBongardScenePredicateIRError("language boundary IDs differ")
        keys = tuple((item.observable_id, item.unit.value, item.comparison.value, item.value) for item in self.boundaries)
        if keys != tuple(sorted(set(keys))): raise ObjectBongardScenePredicateIRError("language boundaries are not canonical")
        if any(not set(item.source_observation_digests).issubset(self.support_observation_digests) for item in self.boundaries): raise ObjectBongardScenePredicateIRError("boundary source escapes support")
        if self.language_digest != canonical_digest(_language_content(self)): raise ObjectBongardScenePredicateIRError("language digest differs")

    def to_data(self) -> dict[str, object]: return {**_language_content(self), "language_digest": self.language_digest}

    @classmethod
    def from_data(cls, value: object) -> "ScenePredicateLanguage":
        expected = {"schema", "algorithm_id", "registry_digest", "registered_tag_ids", "support_observation_digests", "boundaries", "grammar", "candidate_order", *_authority_data(), "language_digest"}
        raw = _fields(value, expected, "scene predicate language")
        if raw["schema"] != SCENE_LANGUAGE_SCHEMA or raw["algorithm_id"] != SCENE_ALGORITHM_ID or raw["candidate_order"] != "complexity-then-canonical-formula-digest-then-orientation" or any(raw[k] != v for k, v in _authority_data().items()) or any(not isinstance(raw[k], list) for k in ("registered_tag_ids", "support_observation_digests", "boundaries")):
            raise ObjectBongardScenePredicateIRError("scene predicate language policy differs")
        result = cls(raw["registry_digest"], tuple(raw["registered_tag_ids"]), tuple(raw["support_observation_digests"]), tuple(SceneNumericBoundary.from_data(item) for item in raw["boundaries"]), raw["language_digest"])
        if result.to_data() != dict(raw): raise ObjectBongardScenePredicateIRError("scene predicate language is not canonical")
        return result

    def boundary(self, boundary_id: str) -> SceneNumericBoundary:
        try: result = self.boundaries[int(boundary_id.removeprefix("boundary_"))]
        except (ValueError, IndexError): raise ObjectBongardScenePredicateIRError("formula names an absent boundary") from None
        if result.boundary_id != boundary_id:
            raise ObjectBongardScenePredicateIRError("formula names a noncanonical boundary alias")
        return result


def freeze_object_scene_predicate_language(registry: ObjectSceneSoftTagRegistry, observations: Sequence[ScenePanelObservation]) -> ScenePredicateLanguage:
    if not isinstance(registry, ObjectSceneSoftTagRegistry): raise TypeError("registry must be ObjectSceneSoftTagRegistry")
    panels = tuple(sorted((ScenePanelObservation.from_data(item.to_data()) for item in observations), key=lambda item: item.panel_id))
    if not panels or len({item.panel_id for item in panels}) != len(panels) or any(item.registry_digest != registry.registry_digest or item.observation_mode != "repeated_registered_merge" for item in panels): raise ObjectBongardScenePredicateIRError("language requires unique repeated observations under one registry")
    sources = tuple(item.observation_digest for item in panels)
    values_by_key: dict[tuple[str, SceneNumericUnit, int], set[str]] = {}
    for panel in panels:
        entity_count = len(panel.entities)
        values_by_key.setdefault(("entity_count", SceneNumericUnit.COUNT, entity_count), set()).add(panel.observation_digest)
        for value in range(entity_count + 1):
            values_by_key.setdefault(("matching_entity_count", SceneNumericUnit.COUNT, value), set()).add(panel.observation_digest)
        pair_count = entity_count * (entity_count - 1) // 2
        for value in range(pair_count + 1):
            values_by_key.setdefault(("matching_pair_count", SceneNumericUnit.COUNT, value), set()).add(panel.observation_digest)
        for entity in panel.entities:
            for cell in entity.count_cells:
                if cell.interval is not None:
                    for value in (cell.interval.lower, cell.interval.upper): values_by_key.setdefault((cell.observable_id, cell.interval.unit, value), set()).add(panel.observation_digest)
    rows = sorted((observable, unit, comparison, value, tuple(sorted(found))) for (observable, unit, value), found in values_by_key.items() for comparison in SceneComparison)
    boundaries = tuple(SceneNumericBoundary.create(f"boundary_{index:05d}", observable, unit, comparison, value, found) for index, (observable, unit, comparison, value, found) in enumerate(rows))
    values = {"registry_digest": registry.registry_digest, "registered_tag_ids": tuple(item.tag_id for item in registry.tags), "support_observation_digests": tuple(sorted(sources)), "boundaries": boundaries}
    provisional = object.__new__(ScenePredicateLanguage)
    for key, item in values.items(): object.__setattr__(provisional, key, item)
    return ScenePredicateLanguage(**values, language_digest=canonical_digest(_language_content(provisional)))


def _formula_content(value: "SceneFormula") -> dict[str, object]:
    return {"schema": SCENE_FORMULA_SCHEMA, "node": value.node.value, "scope": value.scope.value, "atom": None if value.atom is None else value.atom.to_data(), "children": [item.to_data() for item in value.children], "quantifier": None if value.quantifier is None else value.quantifier.value, "count_boundary_id": value.count_boundary_id, "same_binding_for_conjunction": value.node is SceneFormulaNode.AND, "allowed_connectives": ["and"], "syntactic_not_allowed": True, "post_hoc_polarity_operator_allowed": False, "arbitrary_expression_allowed": False, **_authority_data()}


@dataclass(frozen=True, slots=True)
class SceneFormula:
    node: SceneFormulaNode
    scope: SceneScope
    atom: SceneAtom | None
    children: tuple["SceneFormula", ...]
    quantifier: SceneQuantifier | None
    count_boundary_id: str | None
    formula_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.node, SceneFormulaNode) or not isinstance(self.scope, SceneScope): raise TypeError("formula enum differs")
        if self.node is SceneFormulaNode.ATOM:
            if not isinstance(self.atom, SceneAtom) or self.atom.scope is not self.scope or self.children or self.quantifier is not None or self.count_boundary_id is not None: raise ObjectBongardScenePredicateIRError("atom formula differs")
        elif self.node is SceneFormulaNode.AND:
            eligible = {SceneAtomKind.QUALITATIVE, SceneAtomKind.REGISTERED_TAG, SceneAtomKind.GEOMETRY}
            if self.scope is not SceneScope.ENTITY or self.atom is not None or len(self.children) != 2 or self.children != tuple(sorted(self.children, key=lambda x: x.formula_digest)) or len({x.formula_digest for x in self.children}) != 2 or any(x.node is not SceneFormulaNode.ATOM or x.scope is not self.scope or x.atom is None or x.atom.kind not in eligible for x in self.children) or self.quantifier is not None or self.count_boundary_id is not None: raise ObjectBongardScenePredicateIRError("same-binding conjunction differs")
        else:
            if self.scope is SceneScope.PANEL or self.atom is not None or len(self.children) != 1 or self.children[0].scope is not self.scope or self.children[0].node not in (SceneFormulaNode.ATOM, SceneFormulaNode.AND) or not isinstance(self.quantifier, SceneQuantifier): raise ObjectBongardScenePredicateIRError("quantified formula differs")
            if (self.quantifier is SceneQuantifier.COUNT) != isinstance(self.count_boundary_id, str): raise ObjectBongardScenePredicateIRError("COUNT boundary differs")
            if self.quantifier is SceneQuantifier.COUNT and self.children[0].node is not SceneFormulaNode.ATOM:
                raise ObjectBongardScenePredicateIRError("COUNT body must be one positive atom")
            if self.count_boundary_id is not None: _identifier(self.count_boundary_id, "COUNT boundary ID")
        if self.formula_digest != canonical_digest(_formula_content(self)): raise ObjectBongardScenePredicateIRError("formula digest differs")

    @classmethod
    def atom_formula(cls, atom: SceneAtom) -> "SceneFormula":
        return cls._seal(SceneFormulaNode.ATOM, atom.scope, atom=atom)

    @classmethod
    def conjunction(cls, first: "SceneFormula", second: "SceneFormula") -> "SceneFormula":
        if first.scope is not second.scope: raise ObjectBongardScenePredicateIRError("conjunction crosses bindings")
        return cls._seal(SceneFormulaNode.AND, first.scope, children=tuple(sorted((first, second), key=lambda x: x.formula_digest)))

    @classmethod
    def quantified(cls, scope: SceneScope, quantifier: SceneQuantifier, body: "SceneFormula", count_boundary_id: str | None = None) -> "SceneFormula":
        return cls._seal(SceneFormulaNode.QUANTIFIED, scope, children=(body,), quantifier=quantifier, count_boundary_id=count_boundary_id)

    @classmethod
    def _seal(cls, node: SceneFormulaNode, scope: SceneScope, *, atom: SceneAtom | None = None, children: tuple["SceneFormula", ...] = (), quantifier: SceneQuantifier | None = None, count_boundary_id: str | None = None) -> "SceneFormula":
        provisional = object.__new__(cls)
        for key, item in (("node", node), ("scope", scope), ("atom", atom), ("children", children), ("quantifier", quantifier), ("count_boundary_id", count_boundary_id)): object.__setattr__(provisional, key, item)
        return cls(node, scope, atom, children, quantifier, count_boundary_id, canonical_digest(_formula_content(provisional)))

    @property
    def complexity(self) -> int: return 1 + sum(item.complexity for item in self.children)

    def to_data(self) -> dict[str, object]: return {**_formula_content(self), "formula_digest": self.formula_digest}

    @classmethod
    def from_data(cls, value: object) -> "SceneFormula":
        expected = {"schema", "node", "scope", "atom", "children", "quantifier", "count_boundary_id", "same_binding_for_conjunction", "allowed_connectives", "syntactic_not_allowed", "post_hoc_polarity_operator_allowed", "arbitrary_expression_allowed", *_authority_data(), "formula_digest"}
        raw = _fields(value, expected, "scene formula")
        if raw["schema"] != SCENE_FORMULA_SCHEMA or raw["allowed_connectives"] != ["and"] or raw["syntactic_not_allowed"] is not True or raw["post_hoc_polarity_operator_allowed"] is not False or raw["arbitrary_expression_allowed"] is not False or any(raw[k] != v for k, v in _authority_data().items()) or not isinstance(raw["children"], list): raise ObjectBongardScenePredicateIRError("scene formula policy differs")
        result = cls(SceneFormulaNode(raw["node"]), SceneScope(raw["scope"]), None if raw["atom"] is None else SceneAtom.from_data(raw["atom"]), tuple(cls.from_data(item) for item in raw["children"]), None if raw["quantifier"] is None else SceneQuantifier(raw["quantifier"]), raw["count_boundary_id"], raw["formula_digest"])
        if result.to_data() != dict(raw): raise ObjectBongardScenePredicateIRError("scene formula is not canonical")
        return result


def validate_scene_formula(language: ScenePredicateLanguage, formula: SceneFormula, *, candidate_root: bool = False) -> SceneFormula:
    """Bind every atom and numeric comparison to the frozen language."""
    if not isinstance(language, ScenePredicateLanguage) or not isinstance(formula, SceneFormula): raise TypeError("language/formula type differs")
    if formula.node is SceneFormulaNode.ATOM:
        assert formula.atom is not None
        atom = formula.atom
        if atom.kind is SceneAtomKind.REGISTERED_TAG and atom.observable_id not in language.registered_tag_ids: raise ObjectBongardScenePredicateIRError("formula names a tag outside the frozen registry")
        if atom.boundary_id is not None:
            boundary = language.boundary(atom.boundary_id)
            if boundary.value < 1 or boundary.comparison is SceneComparison.AT_MOST:
                raise ObjectBongardScenePredicateIRError("downward or zero numeric predicate is outside the positive language")
            expected = "entity_count" if atom.kind is SceneAtomKind.PANEL_COUNT else atom.observable_id
            if boundary.observable_id != expected or boundary.unit is not SceneNumericUnit.COUNT: raise ObjectBongardScenePredicateIRError("atom boundary type differs")
        if candidate_root and atom.scope is not SceneScope.PANEL: raise ObjectBongardScenePredicateIRError("unquantified candidate atom is not panel scoped")
    elif formula.node is SceneFormulaNode.AND:
        for child in formula.children: validate_scene_formula(language, child)
        if candidate_root: raise ObjectBongardScenePredicateIRError("candidate conjunction lacks an explicit quantifier")
    else:
        body = formula.children[0]
        validate_scene_formula(language, body)
        if formula.quantifier is SceneQuantifier.COUNT:
            assert formula.count_boundary_id is not None
            boundary = language.boundary(formula.count_boundary_id)
            if boundary.value < 1 or boundary.comparison is SceneComparison.AT_MOST:
                raise ObjectBongardScenePredicateIRError("COUNT cannot encode absence or a zero tautology")
            expected = "matching_entity_count" if formula.scope is SceneScope.ENTITY else "matching_pair_count"
            if boundary.observable_id != expected or boundary.unit is not SceneNumericUnit.COUNT: raise ObjectBongardScenePredicateIRError("COUNT boundary type/scope differ")
    return formula


def _candidate_content(value: "ScenePredicateCandidate") -> dict[str, object]:
    return {"schema": SCENE_CANDIDATE_SCHEMA, "language_digest": value.language_digest, "orientation": value.orientation.value, "formula": value.formula.to_data(), "complexity": value.complexity, "affirmative_atoms_only": True, "same_language_both_orientations": True, "post_hoc_complement_synthesized": False, "natural_affirmative_complement_equivalents_may_coexist": True, **_authority_data()}


@dataclass(frozen=True, slots=True)
class ScenePredicateCandidate:
    language_digest: str
    orientation: SceneOrientation
    formula: SceneFormula
    complexity: int
    candidate_digest: str

    def __post_init__(self) -> None:
        _digest(self.language_digest, "candidate language digest")
        if not isinstance(self.orientation, SceneOrientation) or not isinstance(self.formula, SceneFormula): raise TypeError("candidate type differs")
        if type(self.complexity) is not int or self.complexity != self.formula.complexity: raise ObjectBongardScenePredicateIRError("candidate complexity differs")
        _digest(self.candidate_digest, "candidate digest")
        if self.candidate_digest != canonical_digest(_candidate_content(self)): raise ObjectBongardScenePredicateIRError("candidate digest differs")

    @classmethod
    def create(cls, language: ScenePredicateLanguage, orientation: SceneOrientation, formula: SceneFormula) -> "ScenePredicateCandidate":
        validate_scene_formula(language, formula, candidate_root=True)
        provisional = object.__new__(cls)
        values = {"language_digest": language.language_digest, "orientation": orientation, "formula": formula, "complexity": formula.complexity}
        for key, item in values.items(): object.__setattr__(provisional, key, item)
        return cls(**values, candidate_digest=canonical_digest(_candidate_content(provisional)))

    def to_data(self) -> dict[str, object]: return {**_candidate_content(self), "candidate_digest": self.candidate_digest}

    @classmethod
    def from_data(cls, value: object, *, language: ScenePredicateLanguage | None = None) -> "ScenePredicateCandidate":
        expected = {"schema", "language_digest", "orientation", "formula", "complexity", "affirmative_atoms_only", "same_language_both_orientations", "post_hoc_complement_synthesized", "natural_affirmative_complement_equivalents_may_coexist", *_authority_data(), "candidate_digest"}
        raw = _fields(value, expected, "scene predicate candidate")
        if raw["schema"] != SCENE_CANDIDATE_SCHEMA or raw["affirmative_atoms_only"] is not True or raw["same_language_both_orientations"] is not True or raw["post_hoc_complement_synthesized"] is not False or raw["natural_affirmative_complement_equivalents_may_coexist"] is not True or any(raw[k] != v for k, v in _authority_data().items()): raise ObjectBongardScenePredicateIRError("scene candidate policy differs")
        result = cls(raw["language_digest"], SceneOrientation(raw["orientation"]), SceneFormula.from_data(raw["formula"]), raw["complexity"], raw["candidate_digest"])
        if language is not None:
            if result.language_digest != language.language_digest: raise ObjectBongardScenePredicateIRError("candidate belongs to another language")
            validate_scene_formula(language, result.formula, candidate_root=True)
        if result.to_data() != dict(raw): raise ObjectBongardScenePredicateIRError("scene candidate is not canonical")
        return result


def _boundary_groups(language: ScenePredicateLanguage) -> dict[str, tuple[SceneNumericBoundary, ...]]:
    result: dict[str, list[SceneNumericBoundary]] = {}
    for item in language.boundaries: result.setdefault(item.observable_id, []).append(item)
    return {key: tuple(value) for key, value in result.items()}


def enumerate_object_scene_formulas(language: ScenePredicateLanguage) -> tuple[SceneFormula, ...]:
    """Enumerate the complete label-blind finite grammar before orientation."""
    if not isinstance(language, ScenePredicateLanguage): raise TypeError("language must be ScenePredicateLanguage")
    boundaries = {key: tuple(item for item in values if item.value >= 1 and item.comparison is not SceneComparison.AT_MOST) for key, values in _boundary_groups(language).items()}
    entity_atoms: list[SceneAtom] = [SceneAtom.create(SceneScope.ENTITY, SceneAtomKind.QUALITATIVE, item) for item in OBJECT_SCENE_QUALITATIVE_OBSERVABLE_IDS]
    entity_atoms.extend(SceneAtom.create(SceneScope.ENTITY, SceneAtomKind.REGISTERED_TAG, item) for item in language.registered_tag_ids)
    entity_atoms.extend(SceneAtom.create(SceneScope.ENTITY, SceneAtomKind.GEOMETRY, item.value) for item in SceneGeometryPreset)
    for observable in OBJECT_SCENE_COUNT_OBSERVABLE_IDS:
        entity_atoms.extend(SceneAtom.create(SceneScope.ENTITY, SceneAtomKind.COUNT, observable, item.boundary_id) for item in boundaries.get(observable, ()))
    pair_atoms = [SceneAtom.create(SceneScope.PAIR, SceneAtomKind.PAIR_RELATION, item.value) for item in ScenePairRelation]
    panel_atoms = [SceneAtom.create(SceneScope.PANEL, SceneAtomKind.PANEL_COUNT, "entity_count", item.boundary_id) for item in boundaries.get("entity_count", ())]
    entity_local = [SceneFormula.atom_formula(item) for item in entity_atoms]
    pair_local = [SceneFormula.atom_formula(item) for item in pair_atoms]
    eligible_atom_count = sum(item.kind in (SceneAtomKind.QUALITATIVE, SceneAtomKind.REGISTERED_TAG, SceneAtomKind.GEOMETRY) for item in entity_atoms)
    prospective = len(panel_atoms) + len(entity_atoms) * (2 + len(boundaries.get("matching_entity_count", ()))) + eligible_atom_count * (eligible_atom_count - 1) + len(pair_atoms) * (2 + len(boundaries.get("matching_pair_count", ())))
    if prospective > SCENE_MAX_ENUMERATED_FORMULAS:
        raise SceneLanguageCapacityGap(prospective)
    formulas: list[SceneFormula] = [SceneFormula.atom_formula(item) for item in panel_atoms]
    for body in entity_local:
        formulas.extend(SceneFormula.quantified(SceneScope.ENTITY, quantifier, body) for quantifier in (SceneQuantifier.EXISTS, SceneQuantifier.ALL))
        formulas.extend(SceneFormula.quantified(SceneScope.ENTITY, SceneQuantifier.COUNT, body, boundary.boundary_id) for boundary in boundaries.get("matching_entity_count", ()))
    eligible = [body for body in entity_local if body.atom is not None and body.atom.kind in (SceneAtomKind.QUALITATIVE, SceneAtomKind.REGISTERED_TAG, SceneAtomKind.GEOMETRY)]
    for index, first in enumerate(eligible):
        for second in eligible[index + 1:]:
            body = SceneFormula.conjunction(first, second)
            formulas.extend(SceneFormula.quantified(SceneScope.ENTITY, quantifier, body) for quantifier in (SceneQuantifier.EXISTS, SceneQuantifier.ALL))
    for body in pair_local:
        formulas.extend(SceneFormula.quantified(SceneScope.PAIR, quantifier, body) for quantifier in (SceneQuantifier.EXISTS, SceneQuantifier.ALL))
        formulas.extend(SceneFormula.quantified(SceneScope.PAIR, SceneQuantifier.COUNT, body, boundary.boundary_id) for boundary in boundaries.get("matching_pair_count", ()))
    unique = {item.formula_digest: item for item in formulas}
    if len(unique) != prospective:
        raise ObjectBongardScenePredicateIRError("prospective complete formula count differs")
    return tuple(sorted(unique.values(), key=lambda item: (item.complexity, item.formula_digest)))


def enumerate_object_scene_candidates(language: ScenePredicateLanguage) -> tuple[ScenePredicateCandidate, ...]:
    formulas = enumerate_object_scene_formulas(language)
    candidates = tuple(ScenePredicateCandidate.create(language, orientation, formula) for formula in formulas for orientation in SceneOrientation)
    return tuple(sorted(candidates, key=lambda item: (item.complexity, item.formula.formula_digest, item.orientation.value)))


def _compare_interval(interval: SceneNumericInterval, boundary: SceneNumericBoundary) -> Disposition:
    if interval.unit is not boundary.unit: return Disposition.ERROR
    if boundary.comparison is SceneComparison.AT_LEAST:
        if interval.lower >= boundary.value: return Disposition.PRESENT
        if interval.upper < boundary.value: return Disposition.CERTIFIED_ABSENT
    elif boundary.comparison is SceneComparison.AT_MOST:
        if interval.upper <= boundary.value: return Disposition.PRESENT
        if interval.lower > boundary.value: return Disposition.CERTIFIED_ABSENT
    else:
        if interval.lower == interval.upper == boundary.value: return Disposition.PRESENT
        if boundary.value < interval.lower or boundary.value > interval.upper: return Disposition.CERTIFIED_ABSENT
    return Disposition.INDETERMINATE


def _find_cell(cells: Sequence[SceneMergedCell], observable_id: str) -> SceneMergedCell | None:
    return next((item for item in cells if item.observable_id == observable_id), None)


def _evaluate_atom(atom: SceneAtom, language: ScenePredicateLanguage, panel: ScenePanelObservation, binding: object) -> Disposition:
    if panel.disposition is Disposition.ERROR: return Disposition.ERROR
    if atom.scope is SceneScope.ENTITY:
        if not isinstance(binding, SceneEntityObservation): return Disposition.ERROR
        if atom.kind is SceneAtomKind.QUALITATIVE: cell = _find_cell(binding.qualitative_cells, atom.observable_id)
        elif atom.kind is SceneAtomKind.REGISTERED_TAG: cell = _find_cell(binding.registered_tag_cells, atom.observable_id)
        elif atom.kind is SceneAtomKind.COUNT:
            cell = _find_cell(binding.count_cells, atom.observable_id)
            if cell is None or cell.disposition is Disposition.ERROR: return Disposition.ERROR
            if cell.interval is None: return cell.disposition
            assert atom.boundary_id is not None
            return _compare_interval(cell.interval, language.boundary(atom.boundary_id))
        else:
            preset = SceneGeometryPreset(atom.observable_id)
            width, height = binding.bbox_q16[2] - binding.bbox_q16[0], binding.bbox_q16[3] - binding.bbox_q16[1]
            truth = {SceneGeometryPreset.SINGLE_COMPONENT: binding.component_count == 1, SceneGeometryPreset.MULTIPLE_COMPONENTS: binding.component_count >= 2, SceneGeometryPreset.WIDER_THAN_TALL: width > height, SceneGeometryPreset.TALLER_THAN_WIDE: height > width}[preset]
            return Disposition.PRESENT if truth else Disposition.CERTIFIED_ABSENT
        return Disposition.ERROR if cell is None else cell.disposition
    if atom.scope is SceneScope.PAIR:
        if not isinstance(binding, tuple) or len(binding) != 2 or any(not isinstance(item, SceneEntityObservation) for item in binding): return Disposition.ERROR
        first, second = binding
        relation = ScenePairRelation(atom.observable_id)
        ax0, ay0, ax1, ay1 = first.bbox_q16
        bx0, by0, bx1, by1 = second.bbox_q16
        horizontal = ax1 <= bx0 or bx1 <= ax0
        vertical = ay1 <= by0 or by1 <= ay0
        if relation is ScenePairRelation.BBOX_INTERSECTS: truth = not horizontal and not vertical
        elif relation is ScenePairRelation.BBOX_DISJOINT: truth = horizontal or vertical
        elif relation is ScenePairRelation.HORIZONTALLY_SEPARATED_BBOXES: truth = horizontal
        else: truth = vertical
        return Disposition.PRESENT if truth else Disposition.CERTIFIED_ABSENT
    assert atom.boundary_id is not None
    return _compare_interval(SceneNumericInterval(SceneNumericUnit.COUNT, len(panel.entities), len(panel.entities)), language.boundary(atom.boundary_id))


def _evaluate_local(formula: SceneFormula, language: ScenePredicateLanguage, panel: ScenePanelObservation, binding: object) -> Disposition:
    if formula.node is SceneFormulaNode.ATOM:
        assert formula.atom is not None
        return _evaluate_atom(formula.atom, language, panel, binding)
    if formula.node is SceneFormulaNode.AND:
        return scene_and(tuple(_evaluate_local(item, language, panel, binding) for item in formula.children))
    return Disposition.ERROR


def evaluate_object_scene_formula(formula: SceneFormula, language: ScenePredicateLanguage, panel: ScenePanelObservation) -> Disposition:
    validate_scene_formula(language, formula, candidate_root=True)
    if panel.registry_digest != language.registry_digest:
        raise ObjectBongardScenePredicateIRError("panel was observed under a different soft-tag registry")
    if panel.disposition is Disposition.ERROR: return Disposition.ERROR
    if formula.node is SceneFormulaNode.ATOM: return _evaluate_atom(formula.atom, language, panel, panel)  # type: ignore[arg-type]
    body = formula.children[0]
    bindings: tuple[object, ...]
    if formula.scope is SceneScope.ENTITY: bindings = tuple(panel.entities)
    else: bindings = tuple((panel.entities[i], panel.entities[j]) for i in range(len(panel.entities)) for j in range(i + 1, len(panel.entities)))
    row = tuple(_evaluate_local(body, language, panel, item) for item in bindings)
    if not row and formula.quantifier is SceneQuantifier.ALL:
        return Disposition.INDETERMINATE
    if formula.quantifier is SceneQuantifier.EXISTS: return scene_or(row)
    if formula.quantifier is SceneQuantifier.ALL: return scene_and(row)
    if Disposition.ERROR in row: return Disposition.ERROR
    lower = sum(item is Disposition.PRESENT for item in row)
    upper = lower + sum(item is Disposition.INDETERMINATE for item in row)
    assert formula.count_boundary_id is not None
    return _compare_interval(SceneNumericInterval(SceneNumericUnit.COUNT, lower, upper), language.boundary(formula.count_boundary_id))


def evaluate_object_scene_candidate(candidate: ScenePredicateCandidate, language: ScenePredicateLanguage, panel: ScenePanelObservation) -> Disposition:
    if candidate.language_digest != language.language_digest: raise ObjectBongardScenePredicateIRError("candidate/language binding differs")
    return evaluate_object_scene_formula(candidate.formula, language, panel)


@dataclass(frozen=True, slots=True)
class SceneCandidateEvaluation:
    candidate_digest: str
    panel_ids: tuple[str, ...]
    panel_observation_digests: tuple[str, ...]
    dispositions: tuple[Disposition, ...]
    evaluation_digest: str

    def __post_init__(self) -> None:
        _digest(self.candidate_digest, "evaluation candidate digest")
        if not self.panel_ids or len(self.panel_ids) != len(self.panel_observation_digests) or len(self.panel_ids) != len(self.dispositions) or len(set(self.panel_ids)) != len(self.panel_ids): raise ObjectBongardScenePredicateIRError("candidate evaluation width differs")
        for item in self.panel_ids: _identifier(item, "evaluation panel ID")
        for item in self.panel_observation_digests: _digest(item, "evaluation observation digest")
        if any(not isinstance(item, Disposition) for item in self.dispositions): raise TypeError("evaluation disposition differs")
        if self.evaluation_digest != canonical_digest(self.content_data()): raise ObjectBongardScenePredicateIRError("evaluation digest differs")

    def content_data(self) -> dict[str, object]: return {"candidate_digest": self.candidate_digest, "panel_ids": list(self.panel_ids), "panel_observation_digests": list(self.panel_observation_digests), "dispositions": [item.value for item in self.dispositions], "error_never_masked": True}
    def to_data(self) -> dict[str, object]: return {**self.content_data(), "evaluation_digest": self.evaluation_digest}

    @classmethod
    def create(cls, candidate: ScenePredicateCandidate, language: ScenePredicateLanguage, panels: Sequence[ScenePanelObservation]) -> "SceneCandidateEvaluation":
        values = {"candidate_digest": candidate.candidate_digest, "panel_ids": tuple(item.panel_id for item in panels), "panel_observation_digests": tuple(item.observation_digest for item in panels), "dispositions": tuple(evaluate_object_scene_candidate(candidate, language, item) for item in panels)}
        provisional = object.__new__(cls)
        for key, item in values.items(): object.__setattr__(provisional, key, item)
        return cls(**values, evaluation_digest=canonical_digest(provisional.content_data()))

    @classmethod
    def from_data(cls, value: object) -> "SceneCandidateEvaluation":
        raw = _fields(value, {"candidate_digest", "panel_ids", "panel_observation_digests", "dispositions", "error_never_masked", "evaluation_digest"}, "candidate evaluation")
        if raw["error_never_masked"] is not True or any(not isinstance(raw[k], list) for k in ("panel_ids", "panel_observation_digests", "dispositions")): raise ObjectBongardScenePredicateIRError("candidate evaluation policy differs")
        result = cls(raw["candidate_digest"], tuple(raw["panel_ids"]), tuple(raw["panel_observation_digests"]), tuple(Disposition(item) for item in raw["dispositions"]), raw["evaluation_digest"])
        if result.to_data() != dict(raw): raise ObjectBongardScenePredicateIRError("candidate evaluation is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class SceneGateRecord:
    gate_id: str
    passed: bool
    eligible_candidate_count: int
    joint_survivor_count: int
    support_panel_count: int
    gate_digest: str

    def __post_init__(self) -> None:
        if self.gate_id not in ("coverage", "selectivity", "repeatability") or type(self.passed) is not bool or any(type(item) is not int or item < 0 for item in (self.eligible_candidate_count, self.joint_survivor_count, self.support_panel_count)) or self.support_panel_count < 1: raise ObjectBongardScenePredicateIRError("gate record differs")
        if self.passed is not (self.eligible_candidate_count > 0): raise ObjectBongardScenePredicateIRError("gate decision differs from eligible candidate count")
        if self.gate_digest != canonical_digest(self.content_data()): raise ObjectBongardScenePredicateIRError("gate digest differs")

    def content_data(self) -> dict[str, object]: return {"gate_id": self.gate_id, "passed": self.passed, "eligible_candidate_count": self.eligible_candidate_count, "joint_survivor_count": self.joint_survivor_count, "support_panel_count": self.support_panel_count, "decision_rule": "at-least-one-gate-eligible-candidate"}
    def to_data(self) -> dict[str, object]: return {**self.content_data(), "gate_digest": self.gate_digest}
    @classmethod
    def create(cls, gate_id: str, eligible: int, survivors: int, panels: int) -> "SceneGateRecord":
        provisional = object.__new__(cls)
        values = {"gate_id": gate_id, "passed": eligible > 0, "eligible_candidate_count": eligible, "joint_survivor_count": survivors, "support_panel_count": panels}
        for key, item in values.items(): object.__setattr__(provisional, key, item)
        return cls(**values, gate_digest=canonical_digest(provisional.content_data()))
    @classmethod
    def from_data(cls, value: object) -> "SceneGateRecord":
        raw = _fields(value, {"gate_id", "passed", "eligible_candidate_count", "joint_survivor_count", "support_panel_count", "decision_rule", "gate_digest"}, "gate record")
        if raw["decision_rule"] != "at-least-one-gate-eligible-candidate": raise ObjectBongardScenePredicateIRError("gate policy differs")
        result = cls(raw["gate_id"], raw["passed"], raw["eligible_candidate_count"], raw["joint_survivor_count"], raw["support_panel_count"], raw["gate_digest"])
        if result.to_data() != dict(raw): raise ObjectBongardScenePredicateIRError("gate is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class SceneSupportGap:
    kind: SceneGapKind
    candidate_count: int
    witness_compatible_candidate_count: int
    indeterminate_evaluation_count: int
    error_evaluation_count: int
    gap_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SceneGapKind) or any(type(item) is not int or item < 0 for item in (self.candidate_count, self.witness_compatible_candidate_count, self.indeterminate_evaluation_count, self.error_evaluation_count)): raise ObjectBongardScenePredicateIRError("support gap differs")
        expected = SceneGapKind.WITNESS_GAP if self.witness_compatible_candidate_count > 0 else SceneGapKind.LANGUAGE_GAP
        if self.kind is not expected or self.gap_digest != canonical_digest(self.content_data()): raise ObjectBongardScenePredicateIRError("support gap diagnosis differs")
    def content_data(self) -> dict[str, object]: return {"kind": self.kind.value, "candidate_count": self.candidate_count, "witness_compatible_candidate_count": self.witness_compatible_candidate_count, "indeterminate_evaluation_count": self.indeterminate_evaluation_count, "error_evaluation_count": self.error_evaluation_count}
    def to_data(self) -> dict[str, object]: return {**self.content_data(), "gap_digest": self.gap_digest}
    @classmethod
    def create(cls, candidate_count: int, compatible: int, indeterminate: int, error: int) -> "SceneSupportGap":
        kind = SceneGapKind.WITNESS_GAP if compatible > 0 else SceneGapKind.LANGUAGE_GAP
        provisional = object.__new__(cls)
        values = {"kind": kind, "candidate_count": candidate_count, "witness_compatible_candidate_count": compatible, "indeterminate_evaluation_count": indeterminate, "error_evaluation_count": error}
        for key, item in values.items(): object.__setattr__(provisional, key, item)
        return cls(**values, gap_digest=canonical_digest(provisional.content_data()))
    @classmethod
    def from_data(cls, value: object) -> "SceneSupportGap":
        raw = _fields(value, {"kind", "candidate_count", "witness_compatible_candidate_count", "indeterminate_evaluation_count", "error_evaluation_count", "gap_digest"}, "support gap")
        result = cls(SceneGapKind(raw["kind"]), raw["candidate_count"], raw["witness_compatible_candidate_count"], raw["indeterminate_evaluation_count"], raw["error_evaluation_count"], raw["gap_digest"])
        if result.to_data() != dict(raw): raise ObjectBongardScenePredicateIRError("support gap is not canonical")
        return result


def _expected_row(orientation: SceneOrientation, group0_count: int, width: int) -> tuple[Disposition, ...]:
    if not 0 < group0_count < width: raise ObjectBongardScenePredicateIRError("support requires both groups")
    if orientation is SceneOrientation.GROUP0_POSITIVE:
        return (Disposition.PRESENT,) * group0_count + (Disposition.CERTIFIED_ABSENT,) * (width - group0_count)
    return (Disposition.CERTIFIED_ABSENT,) * group0_count + (Disposition.PRESENT,) * (width - group0_count)


def _row_metrics(row: Sequence[Disposition], expected: Sequence[Disposition]) -> tuple[bool, bool, bool, bool]:
    positive_indexes = tuple(i for i, item in enumerate(expected) if item is Disposition.PRESENT)
    negative_indexes = tuple(i for i, item in enumerate(expected) if item is Disposition.CERTIFIED_ABSENT)
    coverage = all(row[i] is Disposition.PRESENT for i in positive_indexes)
    selectivity = all(row[i] is Disposition.CERTIFIED_ABSENT for i in negative_indexes)
    repeatability = all(item not in (Disposition.INDETERMINATE, Disposition.ERROR) for item in row)
    compatible = all(row[i] is not Disposition.CERTIFIED_ABSENT for i in positive_indexes) and all(row[i] is not Disposition.PRESENT for i in negative_indexes)
    return coverage, selectivity, repeatability, compatible


def _version_content(value: "SceneVersionSpace") -> dict[str, object]:
    return {"schema": SCENE_VERSION_SPACE_SCHEMA, "algorithm_digest": value.algorithm_digest, "language_digest": value.language_digest, "orientation": value.orientation.value, "group0_count": value.group0_count, "support_panel_ids": list(value.support_panel_ids), "support_observation_digests": list(value.support_observation_digests), "pass_a_observation_digests": list(value.pass_a_observation_digests), "pass_b_observation_digests": list(value.pass_b_observation_digests), "candidates": [item.to_data() for item in value.candidates], "evaluations": [item.to_data() for item in value.evaluations], "pass_a_evaluations": [item.to_data() for item in value.pass_a_evaluations], "pass_b_evaluations": [item.to_data() for item in value.pass_b_evaluations], "survivor_candidate_digests": list(value.survivor_candidate_digests), "gap": None if value.gap is None else value.gap.to_data(), "coverage_gate": value.coverage_gate.to_data(), "selectivity_gate": value.selectivity_gate.to_data(), "repeatability_gate": value.repeatability_gate.to_data(), "complete_enumeration": True, "orientation_filters_same_label_blind_formula_inventory": True, "repeatability_rule": "same-frozen-candidate-exact-on-pass-a-pass-b-and-conservative-merge", **_authority_data()}


@dataclass(frozen=True, slots=True)
class SceneVersionSpace:
    algorithm_digest: str
    language_digest: str
    orientation: SceneOrientation
    group0_count: int
    support_panel_ids: tuple[str, ...]
    support_observation_digests: tuple[str, ...]
    pass_a_observation_digests: tuple[str, ...]
    pass_b_observation_digests: tuple[str, ...]
    candidates: tuple[ScenePredicateCandidate, ...]
    evaluations: tuple[SceneCandidateEvaluation, ...]
    pass_a_evaluations: tuple[SceneCandidateEvaluation, ...]
    pass_b_evaluations: tuple[SceneCandidateEvaluation, ...]
    survivor_candidate_digests: tuple[str, ...]
    gap: SceneSupportGap | None
    coverage_gate: SceneGateRecord
    selectivity_gate: SceneGateRecord
    repeatability_gate: SceneGateRecord
    version_space_digest: str

    def __post_init__(self) -> None:
        _digest(self.algorithm_digest, "version algorithm digest"); _digest(self.language_digest, "version language digest"); _digest(self.version_space_digest, "version-space digest")
        if not isinstance(self.orientation, SceneOrientation) or type(self.group0_count) is not int or not 0 < self.group0_count < len(self.support_panel_ids) or any(len(self.support_panel_ids) != len(item) for item in (self.support_observation_digests, self.pass_a_observation_digests, self.pass_b_observation_digests)): raise ObjectBongardScenePredicateIRError("version support geometry differs")
        if len(set(self.support_panel_ids)) != len(self.support_panel_ids): raise ObjectBongardScenePredicateIRError("version repeats a support panel")
        if any(item.orientation is not self.orientation or item.language_digest != self.language_digest for item in self.candidates) or tuple((x.complexity, x.formula.formula_digest, x.orientation.value) for x in self.candidates) != tuple(sorted((x.complexity, x.formula.formula_digest, x.orientation.value) for x in self.candidates)): raise ObjectBongardScenePredicateIRError("version candidate inventory differs")
        matrices = ((self.evaluations, self.support_observation_digests), (self.pass_a_evaluations, self.pass_a_observation_digests), (self.pass_b_evaluations, self.pass_b_observation_digests))
        if any(len(matrix) != len(self.candidates) or any(row.candidate_digest != candidate.candidate_digest or row.panel_ids != self.support_panel_ids or row.panel_observation_digests != digests for candidate, row in zip(self.candidates, matrix, strict=True)) for matrix, digests in matrices): raise ObjectBongardScenePredicateIRError("version evaluation matrices differ")
        expected = _expected_row(self.orientation, self.group0_count, len(self.support_panel_ids))
        survivors = tuple(candidate.candidate_digest for candidate, merged, first, second in zip(self.candidates, self.evaluations, self.pass_a_evaluations, self.pass_b_evaluations, strict=True) if merged.dispositions == first.dispositions == second.dispositions == expected)
        if survivors != self.survivor_candidate_digests or (self.gap is None) is not bool(survivors): raise ObjectBongardScenePredicateIRError("version survivor accounting differs")
        metrics_a = tuple(_row_metrics(row.dispositions, expected) for row in self.pass_a_evaluations)
        coverage = sum(item[0] for item in metrics_a)
        selectivity = sum(item[0] and item[1] for item in metrics_a)
        repeatability = sum(first.dispositions == second.dispositions == merged.dispositions == expected for first, second, merged in zip(self.pass_a_evaluations, self.pass_b_evaluations, self.evaluations, strict=True))
        if not coverage >= selectivity >= repeatability == len(survivors): raise ObjectBongardScenePredicateIRError("gate funnel differs")
        gates = (SceneGateRecord.create("coverage", coverage, len(survivors), len(expected)), SceneGateRecord.create("selectivity", selectivity, len(survivors), len(expected)), SceneGateRecord.create("repeatability", repeatability, len(survivors), len(expected)))
        if gates != (self.coverage_gate, self.selectivity_gate, self.repeatability_gate): raise ObjectBongardScenePredicateIRError("version gates differ")
        compatible = sum(all(_row_metrics(row.dispositions, expected)[3] for row in rows) and not all(row.dispositions == expected for row in rows) for rows in zip(self.evaluations, self.pass_a_evaluations, self.pass_b_evaluations, strict=True))
        indeterminate = sum(item is Disposition.INDETERMINATE for matrix in (self.evaluations, self.pass_a_evaluations, self.pass_b_evaluations) for row in matrix for item in row.dispositions)
        errors = sum(item is Disposition.ERROR for matrix in (self.evaluations, self.pass_a_evaluations, self.pass_b_evaluations) for row in matrix for item in row.dispositions)
        if not survivors and self.gap != SceneSupportGap.create(len(self.candidates), compatible, indeterminate, errors): raise ObjectBongardScenePredicateIRError("version typed gap differs")
        if self.version_space_digest != canonical_digest(_version_content(self)): raise ObjectBongardScenePredicateIRError("version-space digest differs")

    def to_data(self) -> dict[str, object]: return {**_version_content(self), "version_space_digest": self.version_space_digest}

    @classmethod
    def from_data(cls, value: object, *, language: ScenePredicateLanguage) -> "SceneVersionSpace":
        if not isinstance(language, ScenePredicateLanguage): raise TypeError("version-space decoding requires its frozen language")
        expected = {"schema", "algorithm_digest", "language_digest", "orientation", "group0_count", "support_panel_ids", "support_observation_digests", "pass_a_observation_digests", "pass_b_observation_digests", "candidates", "evaluations", "pass_a_evaluations", "pass_b_evaluations", "survivor_candidate_digests", "gap", "coverage_gate", "selectivity_gate", "repeatability_gate", "complete_enumeration", "orientation_filters_same_label_blind_formula_inventory", "repeatability_rule", *_authority_data(), "version_space_digest"}
        raw = _fields(value, expected, "scene version space")
        if raw["schema"] != SCENE_VERSION_SPACE_SCHEMA or raw["complete_enumeration"] is not True or raw["orientation_filters_same_label_blind_formula_inventory"] is not True or raw["repeatability_rule"] != "same-frozen-candidate-exact-on-pass-a-pass-b-and-conservative-merge" or any(raw[k] != v for k, v in _authority_data().items()) or any(not isinstance(raw[k], list) for k in ("support_panel_ids", "support_observation_digests", "pass_a_observation_digests", "pass_b_observation_digests", "candidates", "evaluations", "pass_a_evaluations", "pass_b_evaluations", "survivor_candidate_digests")): raise ObjectBongardScenePredicateIRError("scene version policy differs")
        candidates = tuple(ScenePredicateCandidate.from_data(item, language=language) for item in raw["candidates"])
        try:
            complete_for_orientation = tuple(item for item in enumerate_object_scene_candidates(language) if item.orientation is SceneOrientation(raw["orientation"]))
        except SceneLanguageCapacityGap as exc:
            raise ObjectBongardScenePredicateIRError("a materialized orientation space cannot stand in for a typed capacity gap") from exc
        if candidates != complete_for_orientation:
            raise ObjectBongardScenePredicateIRError("orientation space omits or changes the complete candidate inventory")
        result = cls(raw["algorithm_digest"], raw["language_digest"], SceneOrientation(raw["orientation"]), raw["group0_count"], tuple(raw["support_panel_ids"]), tuple(raw["support_observation_digests"]), tuple(raw["pass_a_observation_digests"]), tuple(raw["pass_b_observation_digests"]), candidates, tuple(SceneCandidateEvaluation.from_data(item) for item in raw["evaluations"]), tuple(SceneCandidateEvaluation.from_data(item) for item in raw["pass_a_evaluations"]), tuple(SceneCandidateEvaluation.from_data(item) for item in raw["pass_b_evaluations"]), tuple(raw["survivor_candidate_digests"]), None if raw["gap"] is None else SceneSupportGap.from_data(raw["gap"]), SceneGateRecord.from_data(raw["coverage_gate"]), SceneGateRecord.from_data(raw["selectivity_gate"]), SceneGateRecord.from_data(raw["repeatability_gate"]), raw["version_space_digest"])
        if result.to_data() != dict(raw): raise ObjectBongardScenePredicateIRError("scene version is not canonical")
        return result


def object_bongard_scene_predicate_algorithm_digest(language: ScenePredicateLanguage) -> str:
    return canonical_digest({"schema": "gkm.object-bongard-scene-predicate-algorithm.v1", "algorithm_id": SCENE_ALGORITHM_ID, "implementation_source_sha256": object_bongard_scene_predicate_ir_source_digest(), "language_digest": language.language_digest, "repeat_merge": "P/P=P;A/A=A;flip-or-I=I;missing-or-artifact-failure=E", "numeric_merge": "typed-interval-intersection-else-I", "logic": "error-dominant-strong-kleene-affirmative-and", "quantifiers": ["exists", "all", "count"], "numeric_candidate_boundaries": "at-least-k-or-equal-k-with-k>=1;never-at-most;never-zero", "empty_all": "indeterminate", "two_orientations_predeclared_before_roles": True, "post_hoc_complement_after_failure": False, "absence_count_aliases": False, "natural_affirmative_complement_equivalents_may_coexist": True, **_authority_data()})


def _build_orientation_space(language: ScenePredicateLanguage, algorithm_digest: str, candidates: Sequence[ScenePredicateCandidate], panels: Sequence[ScenePanelObservation], pass_a_panels: Sequence[ScenePanelObservation], pass_b_panels: Sequence[ScenePanelObservation], group0_count: int, orientation: SceneOrientation) -> SceneVersionSpace:
    panels, pass_a_panels, pass_b_panels = tuple(panels), tuple(pass_a_panels), tuple(pass_b_panels)
    if any(len(item) != len(panels) for item in (pass_a_panels, pass_b_panels)) or tuple(item.panel_id for item in pass_a_panels) != tuple(item.panel_id for item in panels) or tuple(item.panel_id for item in pass_b_panels) != tuple(item.panel_id for item in panels) or any(item.observation_mode != "repeated_registered_merge" for item in panels) or any(item.observation_mode != "single_registered" for item in (*pass_a_panels, *pass_b_panels)):
        raise ObjectBongardScenePredicateIRError("orientation support pass alignment differs")
    selected = tuple(item for item in candidates if item.orientation is orientation)
    evaluations = tuple(SceneCandidateEvaluation.create(item, language, panels) for item in selected)
    pass_a_evaluations = tuple(SceneCandidateEvaluation.create(item, language, pass_a_panels) for item in selected)
    pass_b_evaluations = tuple(SceneCandidateEvaluation.create(item, language, pass_b_panels) for item in selected)
    expected = _expected_row(orientation, group0_count, len(panels))
    survivors = tuple(item.candidate_digest for item, merged, first, second in zip(selected, evaluations, pass_a_evaluations, pass_b_evaluations, strict=True) if merged.dispositions == first.dispositions == second.dispositions == expected)
    metrics_a = tuple(_row_metrics(row.dispositions, expected) for row in pass_a_evaluations)
    coverage = sum(item[0] for item in metrics_a)
    selectivity = sum(item[0] and item[1] for item in metrics_a)
    repeatability = len(survivors)
    compatible = sum(all(_row_metrics(row.dispositions, expected)[3] for row in rows) and not all(row.dispositions == expected for row in rows) for rows in zip(evaluations, pass_a_evaluations, pass_b_evaluations, strict=True))
    indeterminate = sum(item is Disposition.INDETERMINATE for matrix in (evaluations, pass_a_evaluations, pass_b_evaluations) for row in matrix for item in row.dispositions)
    errors = sum(item is Disposition.ERROR for matrix in (evaluations, pass_a_evaluations, pass_b_evaluations) for row in matrix for item in row.dispositions)
    values = {"algorithm_digest": algorithm_digest, "language_digest": language.language_digest, "orientation": orientation, "group0_count": group0_count, "support_panel_ids": tuple(item.panel_id for item in panels), "support_observation_digests": tuple(item.observation_digest for item in panels), "pass_a_observation_digests": tuple(item.observation_digest for item in pass_a_panels), "pass_b_observation_digests": tuple(item.observation_digest for item in pass_b_panels), "candidates": selected, "evaluations": evaluations, "pass_a_evaluations": pass_a_evaluations, "pass_b_evaluations": pass_b_evaluations, "survivor_candidate_digests": survivors, "gap": None if survivors else SceneSupportGap.create(len(selected), compatible, indeterminate, errors), "coverage_gate": SceneGateRecord.create("coverage", coverage, len(survivors), len(panels)), "selectivity_gate": SceneGateRecord.create("selectivity", selectivity, len(survivors), len(panels)), "repeatability_gate": SceneGateRecord.create("repeatability", repeatability, len(survivors), len(panels))}
    provisional = object.__new__(SceneVersionSpace)
    for key, item in values.items(): object.__setattr__(provisional, key, item)
    return SceneVersionSpace(**values, version_space_digest=canonical_digest(_version_content(provisional)))


def _semantic_atom_view(atom: SceneAtom, language: ScenePredicateLanguage, registry: ObjectSceneSoftTagRegistry) -> dict[str, object]:
    view: dict[str, object] = {"scope": atom.scope.value, "kind": atom.kind.value}
    if atom.kind is SceneAtomKind.REGISTERED_TAG:
        phrases = {item.tag_id: item.tag for item in registry.tags}
        if atom.observable_id not in phrases: raise ObjectBongardScenePredicateIRError("rank view tag is absent from registry")
        view.update({"tag_id": atom.observable_id, "affirmative_phrase": phrases[atom.observable_id]})
    elif atom.boundary_id is not None:
        boundary = language.boundary(atom.boundary_id)
        view.update({"observable": atom.observable_id.replace("_", " "), "comparison": boundary.comparison.value, "value": boundary.value, "unit": boundary.unit.value})
    elif atom.kind is SceneAtomKind.QUALITATIVE:
        view["affirmative_phrase"] = _QUALITATIVE_PHRASES[atom.observable_id]
    else:
        view["affirmative_phrase"] = atom.observable_id.replace("_", " ")
    return view


def _semantic_formula_view(formula: SceneFormula, language: ScenePredicateLanguage, registry: ObjectSceneSoftTagRegistry) -> dict[str, object]:
    if formula.node is SceneFormulaNode.ATOM:
        assert formula.atom is not None
        return {"node": "positive_atom", **_semantic_atom_view(formula.atom, language, registry)}
    if formula.node is SceneFormulaNode.AND:
        return {"node": "same_entity_positive_conjunction", "terms": [_semantic_formula_view(item, language, registry) for item in formula.children]}
    result: dict[str, object] = {"node": "quantified", "binding_scope": formula.scope.value, "quantifier": formula.quantifier.value, "body": _semantic_formula_view(formula.children[0], language, registry)}  # type: ignore[union-attr]
    if formula.count_boundary_id is not None:
        boundary = language.boundary(formula.count_boundary_id)
        result["count_comparison"] = {"comparison": boundary.comparison.value, "value": boundary.value, "unit": boundary.unit.value}
    return result


def _ranker_view(candidate: ScenePredicateCandidate, language: ScenePredicateLanguage, registry: ObjectSceneSoftTagRegistry, evaluation: SceneCandidateEvaluation, *, coverage: bool, selectivity: bool, repeatability: bool) -> dict[str, object]:
    counts = {item.value: evaluation.dispositions.count(item) for item in Disposition}
    return {"candidate_digest": candidate.candidate_digest, "orientation": candidate.orientation.value, "complexity": candidate.complexity, "formula": _semantic_formula_view(candidate.formula, language, registry), "merged_support_summary": counts, "gate_summary": {"coverage": coverage, "selectivity": selectivity, "repeatability": repeatability}, "formula_is_frozen": True, "ranker_can_only_select": True}


def _candidate_rank_family(candidate: ScenePredicateCandidate) -> str:
    body = candidate.formula.children[0] if candidate.formula.node is SceneFormulaNode.QUANTIFIED else candidate.formula
    if body.node is SceneFormulaNode.AND:
        kinds = tuple(sorted(item.atom.kind.value for item in body.children if item.atom is not None))
        return "same_entity_conjunction:" + "+".join(kinds)
    if body.atom is None: raise ObjectBongardScenePredicateIRError("rank candidate lacks an atomic semantic family")
    return body.atom.kind.value


def _candidate_rank_stratum(candidate: ScenePredicateCandidate) -> tuple[object, ...]:
    body = candidate.formula.children[0] if candidate.formula.node is SceneFormulaNode.QUANTIFIED else candidate.formula
    leaves = body.children if body.node is SceneFormulaNode.AND else (body,)
    identities = tuple(sorted((item.atom.kind.value, item.atom.observable_id) for item in leaves if item.atom is not None))
    quantifier = "panel" if candidate.formula.quantifier is None else candidate.formula.quantifier.value
    return (candidate.complexity, quantifier, identities, candidate.orientation.value)


def _round_robin_rank_strata(candidates: Sequence[ScenePredicateCandidate]) -> tuple[ScenePredicateCandidate, ...]:
    strata: dict[tuple[object, ...], list[ScenePredicateCandidate]] = {}
    for item in sorted(candidates, key=lambda value: (value.complexity, value.formula.formula_digest, value.orientation.value)):
        strata.setdefault(_candidate_rank_stratum(item), []).append(item)
    queues = [strata[key] for key in sorted(strata)]
    result: list[ScenePredicateCandidate] = []
    depth = 0
    while True:
        added = False
        for queue in queues:
            if depth < len(queue): result.append(queue[depth]); added = True
        if not added: return tuple(result)
        depth += 1


def _semantically_stratified_rank_selection(candidates: Sequence[ScenePredicateCandidate]) -> tuple[ScenePredicateCandidate, ...]:
    """Round-robin semantic families, then their quantifier/identity strata."""
    families: dict[str, list[ScenePredicateCandidate]] = {}
    for item in candidates: families.setdefault(_candidate_rank_family(item), []).append(item)
    queues = {key: _round_robin_rank_strata(value) for key, value in families.items()}
    indexes = {key: 0 for key in queues}
    selected: list[ScenePredicateCandidate] = []
    while len(selected) < SCENE_MAX_RANK_SLATE:
        added = False
        for family in sorted(queues):
            index = indexes[family]
            if index < len(queues[family]):
                selected.append(queues[family][index]); indexes[family] = index + 1; added = True
                if len(selected) == SCENE_MAX_RANK_SLATE: break
        if not added: break
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class ScenePredicateCalibrationBundle:
    ir_source_digest: str
    algorithm_digest: str
    registry_digest: str
    coverage_gate: SceneGateRecord
    selectivity_gate: SceneGateRecord
    repeatability_gate: SceneGateRecord
    version_space: Mapping[str, object]
    candidates: tuple[ScenePredicateCandidate, ...]
    complete_survivor_digests: tuple[str, ...]
    ranker_slate: tuple[Mapping[str, object], ...]
    omitted_survivors: tuple[Mapping[str, object], ...]
    bundle_digest: str

    def __post_init__(self) -> None:
        for item, label in ((self.ir_source_digest, "IR source digest"), (self.algorithm_digest, "algorithm digest"), (self.registry_digest, "bundle registry digest"), (self.bundle_digest, "bundle digest")): _digest(item, label)
        if tuple(item.candidate_digest for item in self.candidates) != tuple(dict.fromkeys(item.candidate_digest for item in self.candidates)): raise ObjectBongardScenePredicateIRError("bundle candidate inventory repeats")
        if len(self.ranker_slate) > SCENE_MAX_RANK_SLATE: raise ObjectBongardScenePredicateIRError("ranker slate exceeds capacity")
        survivors = set(self.complete_survivor_digests)
        slate = {item.get("candidate_digest") for item in self.ranker_slate}; omitted = {item.get("candidate_digest") for item in self.omitted_survivors}
        if slate & omitted or slate | omitted != survivors or not survivors.issubset({item.candidate_digest for item in self.candidates}): raise ObjectBongardScenePredicateIRError("bundle survivor accounting differs")
        if bool(survivors) is not all(item.passed for item in (self.coverage_gate, self.selectivity_gate, self.repeatability_gate)): raise ObjectBongardScenePredicateIRError("bundle gate funnel differs")
        if self.bundle_digest != canonical_digest(self.content_data()): raise ObjectBongardScenePredicateIRError("calibration bundle digest differs")

    def content_data(self) -> dict[str, object]:
        return {"schema": SCENE_CALIBRATION_BUNDLE_SCHEMA, "ir_source_digest": self.ir_source_digest, "algorithm_digest": self.algorithm_digest, "registry_digest": self.registry_digest, "coverage_gate": self.coverage_gate.to_data(), "selectivity_gate": self.selectivity_gate.to_data(), "repeatability_gate": self.repeatability_gate.to_data(), "version_space": dict(self.version_space), "candidates": [item.to_data() for item in self.candidates], "complete_survivor_digests": list(self.complete_survivor_digests), "ranker_slate": [dict(item) for item in self.ranker_slate], "omitted_survivors": [dict(item) for item in self.omitted_survivors]}
    def to_data(self) -> dict[str, object]: return {**self.content_data(), "bundle_digest": self.bundle_digest}

    @classmethod
    def from_data(cls, value: object) -> "ScenePredicateCalibrationBundle":
        expected = {"schema", "ir_source_digest", "algorithm_digest", "registry_digest", "coverage_gate", "selectivity_gate", "repeatability_gate", "version_space", "candidates", "complete_survivor_digests", "ranker_slate", "omitted_survivors", "bundle_digest"}
        raw = _fields(value, expected, "scene calibration bundle")
        if raw["schema"] != SCENE_CALIBRATION_BUNDLE_SCHEMA or not isinstance(raw["version_space"], Mapping) or any(not isinstance(raw[key], list) for key in ("candidates", "complete_survivor_digests", "ranker_slate", "omitted_survivors")):
            raise ObjectBongardScenePredicateIRError("scene calibration bundle policy differs")
        version = _fields(raw["version_space"], {"schema", "algorithm_digest", "language", "support_observations", "pass_a_observations", "pass_b_observations", "group0_count", "discovery_artifact_digests", "registered_a_artifact_digests", "registered_b_artifact_digests", "orientation_spaces", "resource_gap", "model_calls_during_build", "full_candidate_space_persisted", "complete_space_accounted_by_typed_capacity_gap", "candidate_enumeration_was_truncated", *_authority_data()}, "bundle embedded version space")
        if version["schema"] != SCENE_VERSION_SPACES_SCHEMA or not isinstance(version["orientation_spaces"], list): raise ObjectBongardScenePredicateIRError("bundle embedded version-space policy differs")
        language = ScenePredicateLanguage.from_data(version["language"])
        if language.registry_digest != raw["registry_digest"] or version["algorithm_digest"] != raw["algorithm_digest"]:
            raise ObjectBongardScenePredicateIRError("bundle language parent binding differs")
        candidates = tuple(ScenePredicateCandidate.from_data(item, language=language) for item in raw["candidates"])
        try:
            complete = enumerate_object_scene_candidates(language)
        except SceneLanguageCapacityGap as exc:
            complete = ()
            expected_resource: object = {"kind": SceneGapKind.LANGUAGE_GAP.value, "reason": "complete_formula_inventory_exceeds_resource_cap", "prospective_formula_count": exc.prospective_formula_count, "maximum_formula_count": SCENE_MAX_ENUMERATED_FORMULAS}
        else:
            expected_resource = None
        if version["resource_gap"] != expected_resource or version["full_candidate_space_persisted"] is not (expected_resource is None) or version["complete_space_accounted_by_typed_capacity_gap"] is not (expected_resource is not None) or version["candidate_enumeration_was_truncated"] is not False:
            raise ObjectBongardScenePredicateIRError("bundle typed capacity accounting differs")
        if candidates != complete:
            raise ObjectBongardScenePredicateIRError("bundle omits or changes the complete candidate inventory")
        spaces = tuple(SceneVersionSpace.from_data(item, language=language) for item in version["orientation_spaces"])
        expected_spaces = () if expected_resource is not None else tuple(SceneOrientation)
        if tuple(item.orientation for item in spaces) != expected_spaces or any(space.candidates != tuple(item for item in candidates if item.orientation is space.orientation) for space in spaces):
            raise ObjectBongardScenePredicateIRError("bundle orientation inventory is not complete")
        result = cls(raw["ir_source_digest"], raw["algorithm_digest"], raw["registry_digest"], SceneGateRecord.from_data(raw["coverage_gate"]), SceneGateRecord.from_data(raw["selectivity_gate"]), SceneGateRecord.from_data(raw["repeatability_gate"]), dict(raw["version_space"]), candidates, tuple(raw["complete_survivor_digests"]), tuple(dict(item) for item in raw["ranker_slate"]), tuple(dict(item) for item in raw["omitted_survivors"]), raw["bundle_digest"])
        if result.to_data() != dict(raw): raise ObjectBongardScenePredicateIRError("scene calibration bundle is not canonical")
        return result


def _aggregate_gate(name: str, spaces: Sequence[SceneVersionSpace], candidate_count: int, panel_count: int) -> SceneGateRecord:
    eligible = sum(getattr(item, f"{name}_gate").eligible_candidate_count for item in spaces)
    survivors = sum(len(item.survivor_candidate_digests) for item in spaces)
    return SceneGateRecord.create(name, eligible, survivors, panel_count)


def build_object_bongard_scene_predicate_calibration_bundle(
    registry: ObjectSceneSoftTagRegistry,
    discovery_artifacts: Sequence[ObjectSceneTranscriptArtifact],
    registered_a_artifacts: Sequence[ObjectSceneTranscriptArtifact],
    registered_b_artifacts: Sequence[ObjectSceneTranscriptArtifact],
    role_rows: Sequence[Mapping[str, object]],
) -> ScenePredicateCalibrationBundle:
    """Build the complete two-orientation Python calibration result."""
    discovery, pass_a, pass_b, roles = tuple(discovery_artifacts), tuple(registered_a_artifacts), tuple(registered_b_artifacts), tuple(role_rows)
    if not isinstance(registry, ObjectSceneSoftTagRegistry) or not discovery or not (len(discovery) == len(pass_a) == len(pass_b) == len(roles)):
        raise ObjectBongardScenePredicateIRError("calibration bundle inputs differ")
    merged_rows: list[tuple[int, ScenePanelObservation]] = []
    pass_a_rows: list[tuple[int, ScenePanelObservation]] = []
    pass_b_rows: list[tuple[int, ScenePanelObservation]] = []
    discovery_digests: list[str] = []
    pass_a_digests: list[str] = []
    pass_b_digests: list[str] = []
    for index, (discover, first, second, role) in enumerate(zip(discovery, pass_a, pass_b, roles, strict=True)):
        raw = _fields(role, {"ordinal", "neutral_panel_digest", "historical_role", "blind_panel_id"}, "role row")
        discover.assert_untampered(); first.assert_untampered(); second.assert_untampered()
        if discover.mode is not ObjectSceneTranscriptMode.DISCOVERY or discover.registry is not None or type(raw["ordinal"]) is not int or raw["ordinal"] < 0 or not isinstance(raw["neutral_panel_digest"], str) or _DIGEST.fullmatch(raw["neutral_panel_digest"]) is None or raw["blind_panel_id"] != discover.scene_id or raw["blind_panel_id"] != first.scene_id or raw["blind_panel_id"] != second.scene_id or raw["historical_role"] not in (0, 1): raise ObjectBongardScenePredicateIRError("role row/artifact identity differs")
        if discover.panel_digest != first.panel_digest or discover.panel_digest != second.panel_digest or discover.inventory_digest != first.inventory_digest or discover.inventory != first.inventory or first.inventory != second.inventory: raise ObjectBongardScenePredicateIRError("discovery/A/B panel inventory binding differs")
        if first.registry != registry or second.registry != registry: raise ObjectBongardScenePredicateIRError("registered artifact uses another frozen registry")
        merged_rows.append((raw["historical_role"], adapt_object_scene_registered_pair(raw["blind_panel_id"], first, second)))
        pass_a_rows.append((raw["historical_role"], adapt_object_scene_registered_single(raw["blind_panel_id"], first)))
        pass_b_rows.append((raw["historical_role"], adapt_object_scene_registered_single(raw["blind_panel_id"], second)))
        discovery_digests.append(discover.artifact_digest); pass_a_digests.append(first.artifact_digest); pass_b_digests.append(second.artifact_digest)
    if any(len({row[key] for row in roles}) != len(roles) for key in ("ordinal", "neutral_panel_digest", "blind_panel_id")):
        raise ObjectBongardScenePredicateIRError("role reveal identities are not unique")
    verify_object_scene_soft_tag_registry(
        registry,
        tuple(item.transcript for item in discovery if item.transcript is not None),
        expected_registry_digest=registry.registry_digest,
    )
    group0 = tuple(item for role, item in merged_rows if role == 0); group1 = tuple(item for role, item in merged_rows if role == 1)
    pass_a_group0 = tuple(item for role, item in pass_a_rows if role == 0); pass_a_group1 = tuple(item for role, item in pass_a_rows if role == 1)
    pass_b_group0 = tuple(item for role, item in pass_b_rows if role == 0); pass_b_group1 = tuple(item for role, item in pass_b_rows if role == 1)
    if not group0 or not group1: raise ObjectBongardScenePredicateIRError("calibration requires both revealed roles")
    panels = group0 + group1
    pass_a_panels = pass_a_group0 + pass_a_group1
    pass_b_panels = pass_b_group0 + pass_b_group1
    language = freeze_object_scene_predicate_language(registry, panels)
    algorithm_digest = object_bongard_scene_predicate_algorithm_digest(language)
    resource_gap: dict[str, object] | None = None
    try:
        candidates = enumerate_object_scene_candidates(language)
    except SceneLanguageCapacityGap as exc:
        candidates = ()
        resource_gap = {"kind": SceneGapKind.LANGUAGE_GAP.value, "reason": "complete_formula_inventory_exceeds_resource_cap", "prospective_formula_count": exc.prospective_formula_count, "maximum_formula_count": SCENE_MAX_ENUMERATED_FORMULAS}
    spaces = () if resource_gap is not None else tuple(_build_orientation_space(language, algorithm_digest, candidates, panels, pass_a_panels, pass_b_panels, len(group0), orientation) for orientation in SceneOrientation)
    survivors = tuple(item for space in spaces for item in space.survivor_candidate_digests)
    coverage = _aggregate_gate("coverage", spaces, len(candidates), len(panels)); selectivity = _aggregate_gate("selectivity", spaces, len(candidates), len(panels)); repeatability = _aggregate_gate("repeatability", spaces, len(candidates), len(panels))
    by_digest = {item.candidate_digest: item for item in candidates}; evaluations = {row.candidate_digest: row for space in spaces for row in space.evaluations}
    survivor_candidates = sorted((by_digest[item] for item in survivors), key=lambda item: (item.complexity, item.formula.formula_digest, item.orientation.value))
    admitted = _semantically_stratified_rank_selection(survivor_candidates)
    slate = tuple(_ranker_view(item, language, registry, evaluations[item.candidate_digest], coverage=True, selectivity=True, repeatability=True) for item in admitted)
    admitted_set = {item.candidate_digest for item in admitted}
    omitted = tuple({"candidate_digest": item.candidate_digest, "reason": "semantic_stratified_rank_slate_capacity_64_exceeded"} for item in survivor_candidates if item.candidate_digest not in admitted_set)
    version_space = {"schema": SCENE_VERSION_SPACES_SCHEMA, "algorithm_digest": algorithm_digest, "language": language.to_data(), "support_observations": [item.to_data() for item in panels], "pass_a_observations": [item.to_data() for item in pass_a_panels], "pass_b_observations": [item.to_data() for item in pass_b_panels], "group0_count": len(group0), "discovery_artifact_digests": sorted(discovery_digests), "registered_a_artifact_digests": sorted(pass_a_digests), "registered_b_artifact_digests": sorted(pass_b_digests), "orientation_spaces": [item.to_data() for item in spaces], "resource_gap": resource_gap, "model_calls_during_build": 0, "full_candidate_space_persisted": resource_gap is None, "complete_space_accounted_by_typed_capacity_gap": resource_gap is not None, "candidate_enumeration_was_truncated": False, **_authority_data()}
    provisional = object.__new__(ScenePredicateCalibrationBundle)
    values = {"ir_source_digest": object_bongard_scene_predicate_ir_source_digest(), "algorithm_digest": algorithm_digest, "registry_digest": registry.registry_digest, "coverage_gate": coverage, "selectivity_gate": selectivity, "repeatability_gate": repeatability, "version_space": version_space, "candidates": tuple(candidates), "complete_survivor_digests": survivors, "ranker_slate": slate, "omitted_survivors": omitted}
    for key, item in values.items(): object.__setattr__(provisional, key, item)
    return ScenePredicateCalibrationBundle(**values, bundle_digest=canonical_digest(provisional.content_data()))


def cold_replay_object_bongard_scene_predicate_calibration_bundle(
    value: ScenePredicateCalibrationBundle | Mapping[str, object],
    registry: ObjectSceneSoftTagRegistry,
) -> ScenePredicateCalibrationBundle:
    """Rebuild the complete grammar, three evaluation matrices, gates, and slate."""
    bundle = ScenePredicateCalibrationBundle.from_data(value.to_data() if isinstance(value, ScenePredicateCalibrationBundle) else value)
    if not isinstance(registry, ObjectSceneSoftTagRegistry) or bundle.registry_digest != registry.registry_digest:
        raise ObjectBongardScenePredicateIRError("cold replay registry differs")
    version = _fields(bundle.version_space, {"schema", "algorithm_digest", "language", "support_observations", "pass_a_observations", "pass_b_observations", "group0_count", "discovery_artifact_digests", "registered_a_artifact_digests", "registered_b_artifact_digests", "orientation_spaces", "resource_gap", "model_calls_during_build", "full_candidate_space_persisted", "complete_space_accounted_by_typed_capacity_gap", "candidate_enumeration_was_truncated", *_authority_data()}, "combined version space")
    if version["schema"] != SCENE_VERSION_SPACES_SCHEMA or version["model_calls_during_build"] != 0 or version["candidate_enumeration_was_truncated"] is not False or any(version[key] != item for key, item in _authority_data().items()) or any(not isinstance(version[key], list) for key in ("support_observations", "pass_a_observations", "pass_b_observations", "discovery_artifact_digests", "registered_a_artifact_digests", "registered_b_artifact_digests", "orientation_spaces")):
        raise ObjectBongardScenePredicateIRError("combined version-space policy differs")
    panels = tuple(ScenePanelObservation.from_data(item) for item in version["support_observations"])
    pass_a = tuple(ScenePanelObservation.from_data(item) for item in version["pass_a_observations"])
    pass_b = tuple(ScenePanelObservation.from_data(item) for item in version["pass_b_observations"])
    language = ScenePredicateLanguage.from_data(version["language"])
    if language != freeze_object_scene_predicate_language(registry, panels) or bundle.ir_source_digest != object_bongard_scene_predicate_ir_source_digest() or bundle.algorithm_digest != object_bongard_scene_predicate_algorithm_digest(language) or version["algorithm_digest"] != bundle.algorithm_digest:
        raise ObjectBongardScenePredicateIRError("cold replay language/algorithm differs")
    group0_count = version["group0_count"]
    if type(group0_count) is not int or not 0 < group0_count < len(panels): raise ObjectBongardScenePredicateIRError("cold replay support grouping differs")
    resource_gap = version["resource_gap"]
    try:
        candidates = enumerate_object_scene_candidates(language)
    except SceneLanguageCapacityGap as exc:
        candidates = ()
        expected_resource: object = {"kind": SceneGapKind.LANGUAGE_GAP.value, "reason": "complete_formula_inventory_exceeds_resource_cap", "prospective_formula_count": exc.prospective_formula_count, "maximum_formula_count": SCENE_MAX_ENUMERATED_FORMULAS}
    else:
        expected_resource = None
    if resource_gap != expected_resource or version["full_candidate_space_persisted"] is not (resource_gap is None) or version["complete_space_accounted_by_typed_capacity_gap"] is not (resource_gap is not None) or tuple(item.to_data() for item in candidates) != tuple(item.to_data() for item in bundle.candidates):
        raise ObjectBongardScenePredicateIRError("cold replay complete candidate inventory differs")
    spaces = () if resource_gap is not None else tuple(_build_orientation_space(language, bundle.algorithm_digest, candidates, panels, pass_a, pass_b, group0_count, orientation) for orientation in SceneOrientation)
    if [item.to_data() for item in spaces] != version["orientation_spaces"]:
        raise ObjectBongardScenePredicateIRError("cold replay orientation spaces differ")
    survivors = tuple(item for space in spaces for item in space.survivor_candidate_digests)
    gates = tuple(_aggregate_gate(name, spaces, len(candidates), len(panels)) for name in ("coverage", "selectivity", "repeatability"))
    if survivors != bundle.complete_survivor_digests or gates != (bundle.coverage_gate, bundle.selectivity_gate, bundle.repeatability_gate):
        raise ObjectBongardScenePredicateIRError("cold replay survivor/gate result differs")
    by_digest = {item.candidate_digest: item for item in candidates}; evaluations = {row.candidate_digest: row for space in spaces for row in space.evaluations}
    survivor_candidates = sorted((by_digest[item] for item in survivors), key=lambda item: (item.complexity, item.formula.formula_digest, item.orientation.value))
    admitted = _semantically_stratified_rank_selection(survivor_candidates)
    slate = tuple(_ranker_view(item, language, registry, evaluations[item.candidate_digest], coverage=True, selectivity=True, repeatability=True) for item in admitted)
    admitted_set = {item.candidate_digest for item in admitted}
    omitted = tuple({"candidate_digest": item.candidate_digest, "reason": "semantic_stratified_rank_slate_capacity_64_exceeded"} for item in survivor_candidates if item.candidate_digest not in admitted_set)
    if slate != bundle.ranker_slate or omitted != bundle.omitted_survivors:
        raise ObjectBongardScenePredicateIRError("cold replay rank slate differs")
    return bundle

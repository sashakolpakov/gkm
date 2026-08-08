"""Prose-grounded ordinal visual observations for generic Bongard panels.

The model does not write a predicate.  It applies one frozen prose rubric to
candidate-independent atlas cells and to the complete scene, returning only
inclusive intervals on a fixed five-level ordinal scale.  Pure Python then
projects those raw rows onto reciprocal-stable object lineages, conservative
unresolved possible-object rows, and one canonical whole-scene witness.

All predicate identity and decisions remain Python-authoritative.  Lean is
neither imported nor required and may be removed without changing replay.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard import prototype_object_observer_protocol as _object_protocol
from bongard import prototype_object_scene_observer as _object_observer
from bongard import prototype_scene_observer as _legacy
from bongard.prototype_object_hypotheses import (
    ObjectHypothesis,
    ObjectHypothesisAtlasSheet,
    ObjectHypothesisPacket,
    render_object_hypothesis_atlas,
    verify_object_hypothesis_packet,
)
from bongard.prototype_object_lineages import (
    ObjectLineage,
    ObjectLineagePacket,
    verify_object_lineage_packet,
)
from bongard.prototype_object_profiles import OBJECT_FEATURE_IDS
from bongard.prototype_scene_observer import (
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexNoToolsAttestation,
    PrototypeImageIdentity,
    PrototypeSceneObserverStatus,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.transport import (
    CodexReceipt,
    run_codex_named_images_structured,
    validate_codex_strict_output_schema,
)


RUBRIC_SPEC_SCHEMA = "gkm.bongard-object-rubric-spec.v1"
RUBRIC_OBSERVATION_SCHEMA = "gkm.bongard-object-rubric-observation.v1"
RUBRIC_SHARD_SCHEMA = "gkm.bongard-object-rubric-observer-shard.v1"
RUBRIC_OBSERVER_ARTIFACT_SCHEMA = "gkm.bongard-object-rubric-observer-artifact.v1"
RUBRIC_OBSERVER_PROTOCOL_ID = "bongard.object-rubric-observer/ordinal-atlas-scene-v1"

# These strings are the operational meaning of every threshold in the closed
# downstream language.  They are model-visible, content-addressed, and shared
# verbatim with the text-only survivor ranker.
RUBRIC_ORDINAL_LEVEL_ANCHORS: tuple[tuple[int, str], ...] = (
    (0, "No visible match, or the visible form contradicts the description."),
    (1, "One weak or incidental visible cue matches the description."),
    (2, "A plausible but partial or ambiguous visible match."),
    (3, "A clear match showing most defining visible cues."),
    (4, "An unmistakable prototypical visible match."),
)

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PANEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}\Z")
_OBSERVATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}\Z")
_REASON = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


class ObjectBongardRubricObserverError(ValueError):
    """A rubric, visual payload, provenance binding, or replay is invalid."""


class RubricScope(str, Enum):
    OBJECT = "object"
    SCENE = "scene"


class RubricObservationState(str, Enum):
    SCORED = "scored"
    INDETERMINATE = "indeterminate"
    ERROR = "error"


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_defines_identity_or_decision": False,
        "lean_required_for_replay": False,
        "lean_removal_changes_decision": False,
    }


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectBongardRubricObserverError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectBongardRubricObserverError(f"{label} must be a raw SHA-256")
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectBongardRubricObserverError(f"{label} must be a sha256: address")
    return value


def _panel_id(value: object) -> str:
    if not isinstance(value, str) or _PANEL_ID.fullmatch(value) is None:
        raise ObjectBongardRubricObserverError("panel ID is invalid")
    return value


def object_bongard_rubric_observer_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def object_bongard_rubric_ordinal_scale_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-object-rubric-ordinal-scale.v1",
            "anchors": [list(item) for item in RUBRIC_ORDINAL_LEVEL_ANCHORS],
            "interval_semantics": "inclusive-lower-upper-narrowest-honest-range",
        }
    )


def object_bongard_rubric_observer_catalog_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-object-rubric-observer-catalog.v1",
            "protocol_id": RUBRIC_OBSERVER_PROTOCOL_ID,
            "source_digest": object_bongard_rubric_observer_source_digest(),
            "ordinal_scale_digest": object_bongard_rubric_ordinal_scale_digest(),
            "hypothesis_extractor_digest": (
                _object_observer.object_hypothesis_extractor_artifact_digest()
            ),
            "lineage_policy": "reciprocal-stable-objects-plus-eligible-unresolved-blockers",
            "scene_policy": "one-all-members-whole-scene-lineage",
            "physical_calls": "one-per-canonical-atlas-sheet",
            **_authority_data(),
        }
    )


def _spec_content(value: "ObjectBongardRubricSpec") -> dict[str, object]:
    return {
        "schema": RUBRIC_SPEC_SCHEMA,
        "semantic_artifact_digest": value.semantic_artifact_digest,
        "rubric": value.rubric,
        "feature_nominations": list(value.feature_nominations),
        "ordinal_scale_digest": object_bongard_rubric_ordinal_scale_digest(),
        "prose_is_observed_not_executable": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricSpec:
    semantic_artifact_digest: str
    rubric: str
    feature_nominations: tuple[str, ...]
    spec_digest: str

    def __post_init__(self) -> None:
        _digest(self.semantic_artifact_digest, "semantic artifact digest")
        try:
            prose = _object_protocol._audit_prose(self.rubric, "rubric")
        except (TypeError, ValueError) as exc:
            raise ObjectBongardRubricObserverError("rubric prose is invalid") from exc
        if prose != self.rubric:
            raise ObjectBongardRubricObserverError("rubric prose differs")
        if (
            not isinstance(self.feature_nominations, tuple)
            or not self.feature_nominations
            or self.feature_nominations
            != tuple(sorted(set(self.feature_nominations), key=OBJECT_FEATURE_IDS.index))
            or any(item not in OBJECT_FEATURE_IDS for item in self.feature_nominations)
        ):
            raise ObjectBongardRubricObserverError("feature nominations are invalid")
        _digest(self.spec_digest, "rubric spec digest")
        if self.spec_digest != canonical_digest(_spec_content(self)):
            raise ObjectBongardRubricObserverError("rubric spec digest differs")

    @classmethod
    def create(
        cls,
        semantic_artifact_digest: str,
        rubric: str,
        feature_nominations: Sequence[str],
    ) -> "ObjectBongardRubricSpec":
        values = {
            "semantic_artifact_digest": semantic_artifact_digest,
            "rubric": rubric,
            "feature_nominations": tuple(feature_nominations),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            spec_digest=canonical_digest(_spec_content(provisional)),
        )

    @classmethod
    def from_semantic_artifact(
        cls, artifact: object, *, expected_artifact_digest: str
    ) -> "ObjectBongardRubricSpec":
        expected = _digest(expected_artifact_digest, "expected semantic digest")
        if (
            getattr(artifact, "artifact_digest", None) != expected
            or len(getattr(artifact, "rubrics", ())) != 2
            or len(getattr(artifact, "feature_families", ())) != 2
        ):
            raise ObjectBongardRubricObserverError("semantic artifact differs")
        return cls.create(expected, artifact.rubrics[0], artifact.feature_families[0])

    def to_data(self) -> dict[str, object]:
        return {**_spec_content(self), "spec_digest": self.spec_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardRubricSpec":
        raw = _fields(
            value,
            {
                "schema", "semantic_artifact_digest", "rubric",
                "feature_nominations", "ordinal_scale_digest",
                "prose_is_observed_not_executable", *_authority_data(), "spec_digest",
            },
            "rubric spec",
        )
        if (
            raw["schema"] != RUBRIC_SPEC_SCHEMA
            or raw["ordinal_scale_digest"] != object_bongard_rubric_ordinal_scale_digest()
            or raw["prose_is_observed_not_executable"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["feature_nominations"], list)
        ):
            raise ObjectBongardRubricObserverError("rubric spec policy differs")
        result = cls(
            raw["semantic_artifact_digest"], raw["rubric"],
            tuple(raw["feature_nominations"]), raw["spec_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricObserverError("rubric spec is not canonical")
        return result


@dataclass(frozen=True, order=True, slots=True)
class OrdinalLevelInterval:
    lower: int
    upper: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.lower, bool)
            or isinstance(self.upper, bool)
            or not isinstance(self.lower, int)
            or not isinstance(self.upper, int)
            or not 0 <= self.lower <= self.upper <= 4
        ):
            raise ObjectBongardRubricObserverError("ordinal interval must lie in 0..4")

    def to_data(self) -> dict[str, int]:
        return {"lower": self.lower, "upper": self.upper}

    @classmethod
    def from_data(cls, value: object) -> "OrdinalLevelInterval":
        raw = _fields(value, {"lower", "upper"}, "ordinal interval")
        return cls(raw["lower"], raw["upper"])


def _observation_content(value: "RubricScopeObservation") -> dict[str, object]:
    return {
        "schema": RUBRIC_OBSERVATION_SCHEMA,
        "rubric_spec_digest": value.rubric_spec_digest,
        "scope": value.scope.value,
        "observation_id": value.observation_id,
        "member_hypothesis_ids": list(value.member_hypothesis_ids),
        "geometry_digest": value.geometry_digest,
        "state": value.state.value,
        "interval": None if value.interval is None else value.interval.to_data(),
        "reason": value.reason,
        "error_type": value.error_type,
        "ordinal_scale_digest": object_bongard_rubric_ordinal_scale_digest(),
    }


@dataclass(frozen=True, slots=True)
class RubricScopeObservation:
    rubric_spec_digest: str
    scope: RubricScope
    observation_id: str
    member_hypothesis_ids: tuple[str, ...]
    geometry_digest: str
    state: RubricObservationState
    interval: OrdinalLevelInterval | None
    reason: str | None
    error_type: str | None
    observation_digest: str

    def __post_init__(self) -> None:
        _digest(self.rubric_spec_digest, "observation rubric spec digest")
        if not isinstance(self.scope, RubricScope):
            raise TypeError("observation scope must be RubricScope")
        if not isinstance(self.observation_id, str) or _OBSERVATION_ID.fullmatch(self.observation_id) is None:
            raise ObjectBongardRubricObserverError("observation ID is invalid")
        if (
            not isinstance(self.member_hypothesis_ids, tuple)
            or any(not isinstance(item, str) or not item for item in self.member_hypothesis_ids)
        ):
            raise ObjectBongardRubricObserverError("observation member IDs are invalid")
        _digest(self.geometry_digest, "observation geometry digest")
        if not isinstance(self.state, RubricObservationState):
            raise TypeError("observation state must be RubricObservationState")
        if self.state is RubricObservationState.SCORED:
            if not isinstance(self.interval, OrdinalLevelInterval) or self.reason is not None or self.error_type is not None:
                raise ObjectBongardRubricObserverError("scored observation differs")
        elif self.state is RubricObservationState.INDETERMINATE:
            if self.interval is not None or not isinstance(self.reason, str) or _REASON.fullmatch(self.reason) is None or self.error_type is not None:
                raise ObjectBongardRubricObserverError("indeterminate observation differs")
        elif self.interval is not None or not isinstance(self.reason, str) or _REASON.fullmatch(self.reason) is None or not isinstance(self.error_type, str) or not self.error_type:
            raise ObjectBongardRubricObserverError("error observation differs")
        _digest(self.observation_digest, "observation digest")
        if self.observation_digest != canonical_digest(_observation_content(self)):
            raise ObjectBongardRubricObserverError("observation digest differs")

    @classmethod
    def create(
        cls,
        *,
        rubric_spec_digest: str,
        scope: RubricScope,
        observation_id: str,
        member_hypothesis_ids: Sequence[str],
        geometry_digest: str,
        state: RubricObservationState,
        interval: OrdinalLevelInterval | None = None,
        reason: str | None = None,
        error_type: str | None = None,
    ) -> "RubricScopeObservation":
        values = {
            "rubric_spec_digest": rubric_spec_digest,
            "scope": scope,
            "observation_id": observation_id,
            "member_hypothesis_ids": tuple(member_hypothesis_ids),
            "geometry_digest": geometry_digest,
            "state": state,
            "interval": interval,
            "reason": reason,
            "error_type": error_type,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, observation_digest=canonical_digest(_observation_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_observation_content(self), "observation_digest": self.observation_digest}

    @classmethod
    def from_data(cls, value: object) -> "RubricScopeObservation":
        raw = _fields(
            value,
            {
                "schema", "rubric_spec_digest", "scope", "observation_id",
                "member_hypothesis_ids", "geometry_digest", "state", "interval",
                "reason", "error_type", "ordinal_scale_digest", "observation_digest",
            },
            "rubric observation",
        )
        if raw["schema"] != RUBRIC_OBSERVATION_SCHEMA or raw["ordinal_scale_digest"] != object_bongard_rubric_ordinal_scale_digest() or not isinstance(raw["member_hypothesis_ids"], list):
            raise ObjectBongardRubricObserverError("rubric observation policy differs")
        try:
            scope = RubricScope(raw["scope"])
            state = RubricObservationState(raw["state"])
        except (TypeError, ValueError) as exc:
            raise ObjectBongardRubricObserverError("rubric observation enum differs") from exc
        result = cls(
            raw["rubric_spec_digest"], scope, raw["observation_id"],
            tuple(raw["member_hypothesis_ids"]), raw["geometry_digest"], state,
            None if raw["interval"] is None else OrdinalLevelInterval.from_data(raw["interval"]),
            raw["reason"], raw["error_type"], raw["observation_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricObserverError("rubric observation is not canonical")
        return result


def _score_content(value: "ObjectBongardRubricSlotScore") -> dict[str, object]:
    return {
        "slot_id": value.slot_id,
        "state": value.state.value,
        "interval": None if value.interval is None else value.interval.to_data(),
        "reason": value.reason,
        "error_type": value.error_type,
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricSlotScore:
    """One raw model row, before any object-lineage decision is made."""

    slot_id: str
    state: RubricObservationState
    interval: OrdinalLevelInterval | None
    reason: str | None
    error_type: str | None
    score_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.slot_id, str) or _OBSERVATION_ID.fullmatch(self.slot_id) is None:
            raise ObjectBongardRubricObserverError("raw score slot ID is invalid")
        if not isinstance(self.state, RubricObservationState):
            raise TypeError("raw score state has the wrong type")
        if self.state is RubricObservationState.SCORED:
            if not isinstance(self.interval, OrdinalLevelInterval) or self.reason is not None or self.error_type is not None:
                raise ObjectBongardRubricObserverError("scored raw row differs")
        elif self.state is RubricObservationState.INDETERMINATE:
            if self.interval is not None or not isinstance(self.reason, str) or _REASON.fullmatch(self.reason) is None or self.error_type is not None:
                raise ObjectBongardRubricObserverError("indeterminate raw row differs")
        elif self.interval is not None or not isinstance(self.reason, str) or _REASON.fullmatch(self.reason) is None or not isinstance(self.error_type, str) or not self.error_type:
            raise ObjectBongardRubricObserverError("error raw row differs")
        _digest(self.score_digest, "raw score digest")
        if self.score_digest != canonical_digest(_score_content(self)):
            raise ObjectBongardRubricObserverError("raw score digest differs")

    @classmethod
    def scored(cls, slot_id: str, interval: OrdinalLevelInterval) -> "ObjectBongardRubricSlotScore":
        values = {
            "slot_id": slot_id,
            "state": RubricObservationState.SCORED,
            "interval": interval,
            "reason": None,
            "error_type": None,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, score_digest=canonical_digest(_score_content(provisional)))

    @classmethod
    def failure(cls, slot_id: str, reason: str, error_type: str) -> "ObjectBongardRubricSlotScore":
        values = {
            "slot_id": slot_id,
            "state": RubricObservationState.ERROR,
            "interval": None,
            "reason": reason,
            "error_type": error_type,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, score_digest=canonical_digest(_score_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_score_content(self), "score_digest": self.score_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardRubricSlotScore":
        raw = _fields(value, {"slot_id", "state", "interval", "reason", "error_type", "score_digest"}, "raw rubric score")
        try:
            state = RubricObservationState(raw["state"])
        except (TypeError, ValueError) as exc:
            raise ObjectBongardRubricObserverError("raw score state is unknown") from exc
        result = cls(
            raw["slot_id"], state,
            None if raw["interval"] is None else OrdinalLevelInterval.from_data(raw["interval"]),
            raw["reason"], raw["error_type"], raw["score_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricObserverError("raw score is not canonical")
        return result


def _receipt_data(value: object | None) -> object:
    return None if value is None else value.to_dict()  # type: ignore[union-attr]


def _receipt_from_data(value: object) -> CodexReceipt | None:
    if value is None:
        return None
    result = _legacy._receipt_from_data(value)
    if not isinstance(result, CodexReceipt):
        raise ObjectBongardRubricObserverError("shard receipt has the wrong type")
    return result


def _canonical_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ObjectBongardRubricObserverError("observer payload must be an object")
    try:
        decoded = json.loads(canonical_json(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardRubricObserverError("observer payload is not canonical JSON") from exc
    if not isinstance(decoded, dict):
        raise ObjectBongardRubricObserverError("observer payload must be an object")
    return decoded


def _shard_content(value: "ObjectBongardRubricObserverShard") -> dict[str, object]:
    return {
        "schema": RUBRIC_SHARD_SCHEMA,
        "status": value.status.value,
        "sheet_index": value.sheet_index,
        "sheet_name": value.sheet_name,
        "slot_ids": list(value.slot_ids),
        "presentation": [item.to_data() for item in value.presentation],
        "prompt_digest": value.prompt_digest,
        "output_schema_digest": value.output_schema_digest,
        "model_payload": value.model_payload,
        "receipt": _receipt_data(value.receipt),
        "slot_scores": [item.to_data() for item in value.slot_scores],
        "scene_score": value.scene_score.to_data(),
        "failure_code": value.failure_code,
        "failure_type": value.failure_type,
        "one_physical_call": True,
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricObserverShard:
    status: PrototypeSceneObserverStatus
    sheet_index: int
    sheet_name: str
    slot_ids: tuple[str, ...]
    presentation: tuple[PrototypeImageIdentity, ...]
    prompt_digest: str
    output_schema_digest: str
    model_payload: Mapping[str, Any] | None
    receipt: CodexReceipt | None
    slot_scores: tuple[ObjectBongardRubricSlotScore, ...]
    scene_score: ObjectBongardRubricSlotScore
    failure_code: str | None
    failure_type: str | None
    shard_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, PrototypeSceneObserverStatus):
            raise TypeError("shard status has the wrong type")
        if isinstance(self.sheet_index, bool) or not isinstance(self.sheet_index, int) or self.sheet_index < 0:
            raise ObjectBongardRubricObserverError("sheet index is invalid")
        if self.sheet_name != f"sheet_{self.sheet_index:03d}.png":
            raise ObjectBongardRubricObserverError("sheet name differs")
        if not isinstance(self.slot_ids, tuple) or len(set(self.slot_ids)) != len(self.slot_ids):
            raise ObjectBongardRubricObserverError("shard slot inventory differs")
        if tuple(item.slot_id for item in self.slot_scores) != self.slot_ids:
            raise ObjectBongardRubricObserverError("shard scores do not exhaust slots")
        if self.scene_score.slot_id != "scene":
            raise ObjectBongardRubricObserverError("shard scene score differs")
        if (
            not isinstance(self.presentation, tuple)
            or len(self.presentation) != 2
            or tuple(item.name for item in self.presentation) != ("scene.png", self.sheet_name)
        ):
            raise ObjectBongardRubricObserverError("shard presentation differs")
        _digest(self.prompt_digest, "shard prompt digest")
        _digest(self.output_schema_digest, "shard schema digest")
        success = self.status is PrototypeSceneObserverStatus.SUCCESS
        if success:
            if self.model_payload is None or self.receipt is None or self.failure_code is not None or self.failure_type is not None or any(item.state is not RubricObservationState.SCORED for item in (*self.slot_scores, self.scene_score)):
                raise ObjectBongardRubricObserverError("successful shard differs")
        elif self.failure_code is None or self.failure_type is None or any(item.state is not RubricObservationState.ERROR for item in (*self.slot_scores, self.scene_score)):
            raise ObjectBongardRubricObserverError("failed shard differs")
        if self.model_payload is not None:
            object.__setattr__(self, "model_payload", _canonical_payload(self.model_payload))
        _digest(self.shard_digest, "shard digest")
        if self.shard_digest != canonical_digest(_shard_content(self)):
            raise ObjectBongardRubricObserverError("shard digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_shard_content(self), "shard_digest": self.shard_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardRubricObserverShard":
        raw = _fields(
            value,
            {
                "schema", "status", "sheet_index", "sheet_name", "slot_ids",
                "presentation", "prompt_digest", "output_schema_digest",
                "model_payload", "receipt", "slot_scores", "scene_score",
                "failure_code", "failure_type", "one_physical_call", "shard_digest",
            },
            "rubric observer shard",
        )
        if raw["schema"] != RUBRIC_SHARD_SCHEMA or raw["one_physical_call"] is not True or any(not isinstance(raw[name], list) for name in ("slot_ids", "presentation", "slot_scores")):
            raise ObjectBongardRubricObserverError("rubric shard policy differs")
        try:
            status = PrototypeSceneObserverStatus(raw["status"])
        except (TypeError, ValueError) as exc:
            raise ObjectBongardRubricObserverError("shard status is unknown") from exc
        result = cls(
            status, raw["sheet_index"], raw["sheet_name"], tuple(raw["slot_ids"]),
            tuple(PrototypeImageIdentity.from_data(item) for item in raw["presentation"]),
            raw["prompt_digest"], raw["output_schema_digest"], raw["model_payload"],
            _receipt_from_data(raw["receipt"]),
            tuple(ObjectBongardRubricSlotScore.from_data(item) for item in raw["slot_scores"]),
            ObjectBongardRubricSlotScore.from_data(raw["scene_score"]),
            raw["failure_code"], raw["failure_type"], raw["shard_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricObserverError("rubric shard is not canonical")
        return result


def object_bongard_rubric_observer_model_digest(model: str, reasoning_effort: str) -> str:
    if not isinstance(model, str) or not model or not isinstance(reasoning_effort, str) or not reasoning_effort:
        raise ObjectBongardRubricObserverError("observer model request is invalid")
    return canonical_digest({"schema": "gkm.bongard-object-rubric-model.v1", "model": model, "reasoning_effort": reasoning_effort})


def _runtime_identity_digest(
    *, model: str, reasoning_effort: str, expected_launcher_digest: str | None,
    cloud_policy_cache_binding: str, model_catalog_digest: str,
    no_tools_attestation_digest: str,
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-object-rubric-runtime.v1",
            "model_digest": object_bongard_rubric_observer_model_digest(model, reasoning_effort),
            "expected_launcher_digest": expected_launcher_digest,
            "cloud_policy_cache_binding": cloud_policy_cache_binding,
            "model_catalog_digest": model_catalog_digest,
            "no_tools_attestation_digest": no_tools_attestation_digest,
            "transport_source_digest": _object_observer.prototype_scene_transport_source_digest(),
        }
    )


def object_bongard_rubric_observer_output_schema() -> dict[str, object]:
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "scene": {
                "type": "object",
                "properties": {
                    "lower": {"type": "integer"},
                    "upper": {"type": "integer"},
                },
                "required": ["lower", "upper"],
                "additionalProperties": False,
            },
            "slots": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "slot_id": {"type": "string"},
                        "lower": {"type": "integer"},
                        "upper": {"type": "integer"},
                    },
                    "required": ["slot_id", "lower", "upper"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["scene", "slots"],
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(schema)
    return schema


def object_bongard_rubric_observer_prompt(
    rubric_spec: ObjectBongardRubricSpec,
    sheet: ObjectHypothesisAtlasSheet,
) -> str:
    if not isinstance(rubric_spec, ObjectBongardRubricSpec):
        raise TypeError("rubric_spec must be ObjectBongardRubricSpec")
    if not isinstance(sheet, ObjectHypothesisAtlasSheet):
        raise TypeError("sheet must be ObjectHypothesisAtlasSheet")
    anchors = "\n".join(
        f"{level}: {meaning}" for level, meaning in RUBRIC_ORDINAL_LEVEL_ANCHORS
    )
    positions = "\n".join(
        f"- {slot.slot_id}: row {slot.row_index + 1}, column {slot.column_index + 1}"
        for slot in sheet.slots
    )
    return (
        "Inspect scene.png and the occupied cells of the four-by-four grid in "
        f"{sheet.name}. Apply this exact visible-description rubric:\n"
        f"{rubric_spec.rubric}\n\n"
        "Use only this fixed ordinal scale:\n"
        f"{anchors}\n\n"
        "Return the narrowest honest inclusive lower and upper level for the "
        "complete drawing and for every occupied grid cell. A wide interval "
        "expresses genuine visual uncertainty. Do not omit, merge, or add a cell. "
        "Use the following opaque row-major identities exactly and in this order:\n"
        f"{positions}"
    )


def object_bongard_rubric_observer_protocol_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-object-rubric-observer-protocol.v1",
            "protocol_id": RUBRIC_OBSERVER_PROTOCOL_ID,
            "source_digest": object_bongard_rubric_observer_source_digest(),
            "catalog_digest": object_bongard_rubric_observer_catalog_digest(),
            "ordinal_scale_digest": object_bongard_rubric_ordinal_scale_digest(),
            "output_schema": object_bongard_rubric_observer_output_schema(),
            "prompt_policy": "exact-rubric-fixed-anchors-exhaustive-row-major-slots",
            **_authority_data(),
        }
    )


def _environment_digest(*, runtime_identity_digest: str) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-object-rubric-environment.v1",
            "protocol_digest": object_bongard_rubric_observer_protocol_digest(),
            "catalog_digest": object_bongard_rubric_observer_catalog_digest(),
            "runtime_identity_digest": runtime_identity_digest,
        }
    )


def _score_map(
    shards: Sequence[ObjectBongardRubricObserverShard],
) -> tuple[dict[str, ObjectBongardRubricSlotScore], tuple[ObjectBongardRubricSlotScore, ...]]:
    by_slot: dict[str, ObjectBongardRubricSlotScore] = {}
    scene: list[ObjectBongardRubricSlotScore] = []
    for shard in shards:
        for score in shard.slot_scores:
            if score.slot_id in by_slot:
                raise ObjectBongardRubricObserverError("slot score is duplicated across shards")
            by_slot[score.slot_id] = score
        scene.append(shard.scene_score)
    return by_slot, tuple(scene)


def _slot_by_hypothesis(
    packet: ObjectHypothesisPacket,
) -> dict[tuple[str, str], str]:
    return {
        (slot.scenario_id, slot.hypothesis_id): slot.slot_id
        for sheet in packet.atlas_sheets
        for slot in sheet.slots
    }


def _observation_from_scores(
    *,
    rubric_spec: ObjectBongardRubricSpec,
    scope: RubricScope,
    observation_id: str,
    member_hypothesis_ids: Sequence[str],
    geometry_digest: str,
    scores: Sequence[ObjectBongardRubricSlotScore],
) -> RubricScopeObservation:
    frozen = tuple(scores)
    if not frozen:
        return RubricScopeObservation.create(
            rubric_spec_digest=rubric_spec.spec_digest,
            scope=scope,
            observation_id=observation_id,
            member_hypothesis_ids=member_hypothesis_ids,
            geometry_digest=geometry_digest,
            state=RubricObservationState.INDETERMINATE,
            reason="missing_observation",
        )
    if any(item.state is RubricObservationState.ERROR for item in frozen):
        return RubricScopeObservation.create(
            rubric_spec_digest=rubric_spec.spec_digest,
            scope=scope,
            observation_id=observation_id,
            member_hypothesis_ids=member_hypothesis_ids,
            geometry_digest=geometry_digest,
            state=RubricObservationState.ERROR,
            reason="observer_member_error",
            error_type="RubricObservationError",
        )
    if any(item.state is RubricObservationState.INDETERMINATE for item in frozen):
        return RubricScopeObservation.create(
            rubric_spec_digest=rubric_spec.spec_digest,
            scope=scope,
            observation_id=observation_id,
            member_hypothesis_ids=member_hypothesis_ids,
            geometry_digest=geometry_digest,
            state=RubricObservationState.INDETERMINATE,
            reason="observer_member_indeterminate",
        )
    intervals = tuple(item.interval for item in frozen)
    if any(item is None for item in intervals):
        raise ObjectBongardRubricObserverError("scored row lacks interval")
    return RubricScopeObservation.create(
        rubric_spec_digest=rubric_spec.spec_digest,
        scope=scope,
        observation_id=observation_id,
        member_hypothesis_ids=member_hypothesis_ids,
        geometry_digest=geometry_digest,
        state=RubricObservationState.SCORED,
        interval=OrdinalLevelInterval(
            min(item.lower for item in intervals if item is not None),
            max(item.upper for item in intervals if item is not None),
        ),
    )


def _hypothesis_is_whole_scene(
    hypothesis: ObjectHypothesis, scenario_hypotheses: Sequence[ObjectHypothesis]
) -> bool:
    singleton_count = sum(len(item.source_component_ids) == 1 for item in scenario_hypotheses)
    return len(hypothesis.source_component_ids) == singleton_count and singleton_count > 1


def _hypothesis_is_eligible_unresolved(
    hypothesis: ObjectHypothesis, scenario_hypotheses: Sequence[ObjectHypothesis]
) -> bool:
    if len(hypothesis.source_component_ids) == 1:
        return True
    if _hypothesis_is_whole_scene(hypothesis, scenario_hypotheses):
        return False
    owned = set(hypothesis.source_component_ids)
    deaths = tuple(
        item.emergence_gap_pixels
        for item in scenario_hypotheses
        if owned < set(item.source_component_ids)
    )
    return bool(deaths) and hypothesis.emergence_gap_pixels < min(deaths)


def _project_observations(
    rubric_spec: ObjectBongardRubricSpec,
    hypothesis_packet: ObjectHypothesisPacket,
    lineage_packet: ObjectLineagePacket,
    shards: Sequence[ObjectBongardRubricObserverShard],
) -> tuple[
    tuple[RubricScopeObservation, ...],
    tuple[RubricScopeObservation, ...],
    RubricScopeObservation | None,
]:
    by_slot, scene_scores = _score_map(shards)
    slot_by_key = _slot_by_hypothesis(hypothesis_packet)
    if set(by_slot) != set(slot_by_key.values()):
        raise ObjectBongardRubricObserverError("shards do not exhaust hypothesis atlas")

    stable: list[RubricScopeObservation] = []
    stable_keys: set[tuple[str, str]] = set()
    for lineage in lineage_packet.lineages:
        if not lineage.eligible_for_aggregation:
            continue
        keys = tuple((item.scenario_id, item.hypothesis_id) for item in lineage.members)
        stable_keys.update(keys)
        stable.append(
            _observation_from_scores(
                rubric_spec=rubric_spec,
                scope=RubricScope.OBJECT,
                observation_id=lineage.lineage_id,
                member_hypothesis_ids=tuple(item.hypothesis_id for item in lineage.members),
                geometry_digest=canonical_digest(lineage.to_data()),
                scores=tuple(by_slot[slot_by_key[key]] for key in keys),
            )
        )

    unresolved: list[RubricScopeObservation] = []
    for scenario in hypothesis_packet.scenarios:
        for hypothesis in scenario.hypotheses:
            key = (scenario.scenario_id, hypothesis.hypothesis_id)
            if key in stable_keys or not _hypothesis_is_eligible_unresolved(
                hypothesis, scenario.hypotheses
            ):
                continue
            unresolved.append(
                _observation_from_scores(
                    rubric_spec=rubric_spec,
                    scope=RubricScope.OBJECT,
                    observation_id=f"unresolved/{scenario.scenario_id}/{hypothesis.hypothesis_id}",
                    member_hypothesis_ids=(hypothesis.hypothesis_id,),
                    geometry_digest=hypothesis.digest(),
                    scores=(by_slot[slot_by_key[key]],),
                )
            )

    component_counts = {
        scenario.scenario_id: sum(
            len(item.source_component_ids) == 1 for item in scenario.hypotheses
        )
        for scenario in hypothesis_packet.scenarios
    }
    whole_lineages = tuple(
        lineage
        for lineage in lineage_packet.lineages
        if all(
            len(member.source_component_ids) == component_counts[member.scenario_id]
            for member in lineage.members
        )
    )
    canonical_scene = None
    if len(whole_lineages) == 1:
        lineage = whole_lineages[0]
        canonical_scene = _observation_from_scores(
            rubric_spec=rubric_spec,
            scope=RubricScope.SCENE,
            observation_id="canonical-scene",
            member_hypothesis_ids=tuple(item.hypothesis_id for item in lineage.members),
            geometry_digest=canonical_digest(lineage.to_data()),
            scores=scene_scores,
        )
    return tuple(stable), tuple(unresolved), canonical_scene


def _artifact_content(value: "ObjectBongardRubricObserverArtifact") -> dict[str, object]:
    return {
        "schema": RUBRIC_OBSERVER_ARTIFACT_SCHEMA,
        "panel_id": value.panel_id,
        "panel_digest": value.panel_digest,
        "observation_context_digest": value.observation_context_digest,
        "rubric_spec": value.rubric_spec.to_data(),
        "hypothesis_packet": value.hypothesis_packet.to_data(),
        "lineage_packet": value.lineage_packet.to_data(),
        "catalog_digest": value.catalog_digest,
        "hypothesis_packet_digest": value.hypothesis_packet_digest,
        "lineage_packet_digest": value.lineage_packet_digest,
        "source_digest": value.source_digest,
        "protocol_digest": value.protocol_digest,
        "transport_source_digest": value.transport_source_digest,
        "model": value.model,
        "reasoning_effort": value.reasoning_effort,
        "model_digest": value.model_digest,
        "expected_launcher_digest": value.expected_launcher_digest,
        "cloud_policy_cache_binding": value.cloud_policy_cache_binding,
        "model_catalog_digest": value.model_catalog_digest,
        "no_tools_attestation_digest": value.no_tools_attestation_digest,
        "environment_digest": value.environment_digest,
        "runtime_identity_digest": value.runtime_identity_digest,
        "presentation": [item.to_data() for item in value.presentation],
        "shards": [item.to_data() for item in value.shards],
        "physical_call_count": value.physical_call_count,
        "object_observations": [item.to_data() for item in value.object_observations],
        "unresolved_object_observations": [item.to_data() for item in value.unresolved_object_observations],
        "canonical_scene_observation": None if value.canonical_scene_observation is None else value.canonical_scene_observation.to_data(),
        "unresolved_object_witness_possible": value.unresolved_object_witness_possible,
        "raw_rows_are_candidate_independent": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricObserverArtifact:
    panel_id: str
    panel_digest: str
    observation_context_digest: str
    rubric_spec: ObjectBongardRubricSpec
    hypothesis_packet: ObjectHypothesisPacket
    lineage_packet: ObjectLineagePacket
    catalog_digest: str
    hypothesis_packet_digest: str
    lineage_packet_digest: str
    source_digest: str
    protocol_digest: str
    transport_source_digest: str
    model: str
    reasoning_effort: str
    model_digest: str
    expected_launcher_digest: str | None
    cloud_policy_cache_binding: str
    model_catalog_digest: str
    no_tools_attestation_digest: str
    environment_digest: str
    runtime_identity_digest: str
    presentation: tuple[PrototypeImageIdentity, ...]
    shards: tuple[ObjectBongardRubricObserverShard, ...]
    physical_call_count: int
    object_observations: tuple[RubricScopeObservation, ...]
    unresolved_object_observations: tuple[RubricScopeObservation, ...]
    canonical_scene_observation: RubricScopeObservation | None
    unresolved_object_witness_possible: bool
    artifact_digest: str

    def __post_init__(self) -> None:
        _panel_id(self.panel_id)
        _digest(self.panel_digest, "panel digest")
        _address(self.observation_context_digest, "observation context digest")
        if not isinstance(self.rubric_spec, ObjectBongardRubricSpec):
            raise TypeError("rubric_spec has the wrong type")
        if not isinstance(self.hypothesis_packet, ObjectHypothesisPacket) or not isinstance(self.lineage_packet, ObjectLineagePacket):
            raise TypeError("artifact geometry packets have the wrong type")
        for name in (
            "catalog_digest", "hypothesis_packet_digest", "lineage_packet_digest",
            "source_digest", "protocol_digest", "transport_source_digest",
            "model_digest", "model_catalog_digest", "no_tools_attestation_digest",
            "environment_digest", "runtime_identity_digest", "artifact_digest",
        ):
            _digest(getattr(self, name), name)
        expected_runtime = _runtime_identity_digest(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            expected_launcher_digest=self.expected_launcher_digest,
            cloud_policy_cache_binding=self.cloud_policy_cache_binding,
            model_catalog_digest=self.model_catalog_digest,
            no_tools_attestation_digest=self.no_tools_attestation_digest,
        )
        if (
            self.catalog_digest != object_bongard_rubric_observer_catalog_digest()
            or self.source_digest != object_bongard_rubric_observer_source_digest()
            or self.protocol_digest != object_bongard_rubric_observer_protocol_digest()
            or self.transport_source_digest != _object_observer.prototype_scene_transport_source_digest()
            or self.model_digest != object_bongard_rubric_observer_model_digest(self.model, self.reasoning_effort)
            or self.runtime_identity_digest != expected_runtime
            or self.environment_digest != _environment_digest(runtime_identity_digest=expected_runtime)
        ):
            raise ObjectBongardRubricObserverError("artifact protocol or source binding differs")
        if (
            self.hypothesis_packet_digest != self.hypothesis_packet.digest()
            or self.lineage_packet_digest != self.lineage_packet.digest()
            or self.hypothesis_packet.panel_digest != self.panel_digest
            or self.lineage_packet.panel_digest != self.panel_digest
            or self.lineage_packet.hypothesis_packet_digest != self.hypothesis_packet_digest
        ):
            raise ObjectBongardRubricObserverError("artifact geometry packet binding differs")
        if not isinstance(self.shards, tuple) or tuple(item.sheet_index for item in self.shards) != tuple(range(len(self.shards))) or self.physical_call_count != len(self.shards) or self.physical_call_count < 1:
            raise ObjectBongardRubricObserverError("artifact shard/call inventory differs")
        expected_presentation = tuple(image for shard in self.shards for image in shard.presentation)
        if self.presentation != expected_presentation:
            raise ObjectBongardRubricObserverError("artifact presentation differs")
        schema = object_bongard_rubric_observer_output_schema()
        schema_digest = canonical_digest(schema)
        for sheet, shard in zip(self.hypothesis_packet.atlas_sheets, self.shards, strict=True):
            prompt = object_bongard_rubric_observer_prompt(self.rubric_spec, sheet)
            if (
                shard.sheet_name != sheet.name
                or shard.slot_ids != tuple(item.slot_id for item in sheet.slots)
                or shard.prompt_digest != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                or shard.output_schema_digest != schema_digest
                or shard.presentation[0].content_digest != self.panel_digest
                or shard.presentation[1].byte_count != sheet.png_byte_count
                or shard.presentation[1].content_digest != sheet.png_digest
            ):
                raise ObjectBongardRubricObserverError("artifact shard input binding differs")
            if shard.status is PrototypeSceneObserverStatus.SUCCESS:
                assert shard.model_payload is not None and shard.receipt is not None
                parsed_slots, parsed_scene = _parse_observer_payload(
                    shard.model_payload, shard.slot_ids
                )
                if shard.slot_scores != parsed_slots or shard.scene_score != parsed_scene:
                    raise ObjectBongardRubricObserverError("shard scores differ from model payload")
                receipt = shard.receipt
                if (
                    receipt.prompt_digest != shard.prompt_digest
                    or receipt.output_schema_digest != shard.output_schema_digest
                    or receipt.structured_output_digest != canonical_digest(dict(shard.model_payload))
                    or receipt.requested_model != self.model
                    or receipt.requested_reasoning_effort != self.reasoning_effort
                    or (
                        self.expected_launcher_digest is not None
                        and receipt.codex_launcher_digest != self.expected_launcher_digest
                    )
                    or receipt.cloud_config_bundle_cache_binding != self.cloud_policy_cache_binding
                    or receipt.model_catalog_digest != self.model_catalog_digest
                    or receipt.tool_surface_attestation_digest != self.no_tools_attestation_digest
                ):
                    raise ObjectBongardRubricObserverError("shard receipt binding differs")
        expected_objects, expected_unresolved, expected_scene = _project_observations(
            self.rubric_spec, self.hypothesis_packet, self.lineage_packet, self.shards
        )
        if (
            self.object_observations != expected_objects
            or self.unresolved_object_observations != expected_unresolved
            or self.canonical_scene_observation != expected_scene
        ):
            raise ObjectBongardRubricObserverError("artifact observations differ from raw shard projection")
        for values, scope in ((self.object_observations, RubricScope.OBJECT), (self.unresolved_object_observations, RubricScope.OBJECT)):
            if not isinstance(values, tuple) or any(item.scope is not scope or item.rubric_spec_digest != self.rubric_spec.spec_digest for item in values):
                raise ObjectBongardRubricObserverError("artifact object observation inventory differs")
        if self.canonical_scene_observation is not None and (self.canonical_scene_observation.scope is not RubricScope.SCENE or self.canonical_scene_observation.rubric_spec_digest != self.rubric_spec.spec_digest):
            raise ObjectBongardRubricObserverError("canonical scene observation differs")
        ids = tuple(item.observation_id for item in (*self.object_observations, *self.unresolved_object_observations))
        if len(ids) != len(set(ids)) or self.unresolved_object_witness_possible is not bool(self.unresolved_object_observations):
            raise ObjectBongardRubricObserverError("artifact unresolved inventory differs")
        if self.artifact_digest != canonical_digest(_artifact_content(self)):
            raise ObjectBongardRubricObserverError("observer artifact digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_artifact_content(self), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardRubricObserverArtifact":
        raw = _fields(
            value,
            {
                "schema", "panel_id", "panel_digest", "observation_context_digest",
                "rubric_spec", "hypothesis_packet", "lineage_packet", "catalog_digest", "hypothesis_packet_digest",
                "lineage_packet_digest", "source_digest", "protocol_digest",
                "transport_source_digest", "model", "reasoning_effort", "model_digest",
                "expected_launcher_digest", "cloud_policy_cache_binding",
                "model_catalog_digest", "no_tools_attestation_digest",
                "environment_digest", "runtime_identity_digest", "presentation",
                "shards", "physical_call_count", "object_observations",
                "unresolved_object_observations", "canonical_scene_observation",
                "unresolved_object_witness_possible", "raw_rows_are_candidate_independent",
                *_authority_data(), "artifact_digest",
            },
            "rubric observer artifact",
        )
        if raw["schema"] != RUBRIC_OBSERVER_ARTIFACT_SCHEMA or raw["raw_rows_are_candidate_independent"] is not True or any(raw[key] != item for key, item in _authority_data().items()) or any(not isinstance(raw[name], list) for name in ("presentation", "shards", "object_observations", "unresolved_object_observations")):
            raise ObjectBongardRubricObserverError("rubric artifact policy differs")
        result = cls(
            raw["panel_id"], raw["panel_digest"], raw["observation_context_digest"],
            ObjectBongardRubricSpec.from_data(raw["rubric_spec"]),
            ObjectHypothesisPacket.from_data(raw["hypothesis_packet"]),
            ObjectLineagePacket.from_data(raw["lineage_packet"]),
            raw["catalog_digest"],
            raw["hypothesis_packet_digest"], raw["lineage_packet_digest"],
            raw["source_digest"], raw["protocol_digest"], raw["transport_source_digest"],
            raw["model"], raw["reasoning_effort"], raw["model_digest"],
            raw["expected_launcher_digest"], raw["cloud_policy_cache_binding"],
            raw["model_catalog_digest"], raw["no_tools_attestation_digest"],
            raw["environment_digest"], raw["runtime_identity_digest"],
            tuple(PrototypeImageIdentity.from_data(item) for item in raw["presentation"]),
            tuple(ObjectBongardRubricObserverShard.from_data(item) for item in raw["shards"]),
            raw["physical_call_count"],
            tuple(RubricScopeObservation.from_data(item) for item in raw["object_observations"]),
            tuple(RubricScopeObservation.from_data(item) for item in raw["unresolved_object_observations"]),
            None if raw["canonical_scene_observation"] is None else RubricScopeObservation.from_data(raw["canonical_scene_observation"]),
            raw["unresolved_object_witness_possible"], raw["artifact_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricObserverError("rubric artifact is not canonical")
        return result


def _parse_interval(value: object, label: str) -> OrdinalLevelInterval:
    raw = _fields(value, {"lower", "upper"}, label)
    return OrdinalLevelInterval(raw["lower"], raw["upper"])


def _parse_observer_payload(
    payload: object, slot_ids: Sequence[str]
) -> tuple[tuple[ObjectBongardRubricSlotScore, ...], ObjectBongardRubricSlotScore]:
    raw = _fields(payload, {"scene", "slots"}, "rubric observer payload")
    rows = raw["slots"]
    if not isinstance(rows, list) or len(rows) != len(slot_ids):
        raise ObjectBongardRubricObserverError("payload does not exhaust atlas slots")
    scores: list[ObjectBongardRubricSlotScore] = []
    for expected_id, row in zip(slot_ids, rows, strict=True):
        item = _fields(row, {"slot_id", "lower", "upper"}, "rubric slot row")
        if item["slot_id"] != expected_id:
            raise ObjectBongardRubricObserverError("payload slot order or identity differs")
        scores.append(
            ObjectBongardRubricSlotScore.scored(
                expected_id, OrdinalLevelInterval(item["lower"], item["upper"])
            )
        )
    scene = ObjectBongardRubricSlotScore.scored(
        "scene", _parse_interval(raw["scene"], "scene interval")
    )
    return tuple(scores), scene


def _seal_shard(
    *,
    status: PrototypeSceneObserverStatus,
    sheet: ObjectHypothesisAtlasSheet,
    presentation: tuple[PrototypeImageIdentity, ...],
    prompt: str,
    schema: Mapping[str, Any],
    payload: Mapping[str, Any] | None,
    receipt: CodexReceipt | None,
    slot_scores: Sequence[ObjectBongardRubricSlotScore],
    scene_score: ObjectBongardRubricSlotScore,
    failure_code: str | None,
    failure_type: str | None,
) -> ObjectBongardRubricObserverShard:
    values = {
        "status": status,
        "sheet_index": sheet.sheet_index,
        "sheet_name": sheet.name,
        "slot_ids": tuple(item.slot_id for item in sheet.slots),
        "presentation": presentation,
        "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "output_schema_digest": canonical_digest(dict(schema)),
        "model_payload": None if payload is None else _canonical_payload(payload),
        "receipt": receipt,
        "slot_scores": tuple(slot_scores),
        "scene_score": scene_score,
        "failure_code": failure_code,
        "failure_type": failure_type,
    }
    provisional = object.__new__(ObjectBongardRubricObserverShard)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardRubricObserverShard(
        **values,
        shard_digest=canonical_digest(_shard_content(provisional)),
    )


def _failure_scores(
    sheet: ObjectHypothesisAtlasSheet, reason: str, error_type: str
) -> tuple[tuple[ObjectBongardRubricSlotScore, ...], ObjectBongardRubricSlotScore]:
    return (
        tuple(
            ObjectBongardRubricSlotScore.failure(slot.slot_id, reason, error_type)
            for slot in sheet.slots
        ),
        ObjectBongardRubricSlotScore.failure("scene", reason, error_type),
    )


def _seal_artifact(
    *,
    panel_id: str,
    panel_digest: str,
    observation_context_digest: str,
    rubric_spec: ObjectBongardRubricSpec,
    hypothesis_packet: ObjectHypothesisPacket,
    lineage_packet: ObjectLineagePacket,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str | None,
    cloud_policy_cache_binding: str,
    model_catalog_digest: str,
    no_tools_attestation_digest: str,
    shards: Sequence[ObjectBongardRubricObserverShard],
) -> ObjectBongardRubricObserverArtifact:
    frozen_shards = tuple(shards)
    objects, unresolved, scene = _project_observations(
        rubric_spec, hypothesis_packet, lineage_packet, frozen_shards
    )
    runtime = _runtime_identity_digest(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=cloud_policy_cache_binding,
        model_catalog_digest=model_catalog_digest,
        no_tools_attestation_digest=no_tools_attestation_digest,
    )
    values = {
        "panel_id": panel_id,
        "panel_digest": panel_digest,
        "observation_context_digest": observation_context_digest,
        "rubric_spec": rubric_spec,
        "hypothesis_packet": hypothesis_packet,
        "lineage_packet": lineage_packet,
        "catalog_digest": object_bongard_rubric_observer_catalog_digest(),
        "hypothesis_packet_digest": hypothesis_packet.digest(),
        "lineage_packet_digest": lineage_packet.digest(),
        "source_digest": object_bongard_rubric_observer_source_digest(),
        "protocol_digest": object_bongard_rubric_observer_protocol_digest(),
        "transport_source_digest": _object_observer.prototype_scene_transport_source_digest(),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "model_digest": object_bongard_rubric_observer_model_digest(model, reasoning_effort),
        "expected_launcher_digest": expected_launcher_digest,
        "cloud_policy_cache_binding": cloud_policy_cache_binding,
        "model_catalog_digest": model_catalog_digest,
        "no_tools_attestation_digest": no_tools_attestation_digest,
        "environment_digest": _environment_digest(runtime_identity_digest=runtime),
        "runtime_identity_digest": runtime,
        "presentation": tuple(image for shard in frozen_shards for image in shard.presentation),
        "shards": frozen_shards,
        "physical_call_count": len(frozen_shards),
        "object_observations": objects,
        "unresolved_object_observations": unresolved,
        "canonical_scene_observation": scene,
        "unresolved_object_witness_possible": bool(unresolved),
    }
    provisional = object.__new__(ObjectBongardRubricObserverArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardRubricObserverArtifact(
        **values,
        artifact_digest=canonical_digest(_artifact_content(provisional)),
    )


def observe_object_bongard_rubric(
    png_bytes: bytes,
    *,
    panel_id: str,
    rubric_spec: ObjectBongardRubricSpec,
    hypothesis_packet: ObjectHypothesisPacket,
    lineage_packet: ObjectLineagePacket,
    expected_scene_sha256: str,
    expected_rubric_spec_digest: str,
    expected_hypothesis_packet_digest: str,
    expected_lineage_packet_digest: str,
    model: str,
    reasoning_effort: str = "medium",
    minutes: int = 15,
    verbose: bool = False,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    expected_launcher_digest: str | None = None,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    transport=run_codex_named_images_structured,
    observation_context_digest: str | None = None,
) -> ObjectBongardRubricObserverArtifact:
    """Make exactly one receipt-attested visual call per canonical atlas sheet."""

    scene = _legacy._validate_exact_png(png_bytes, "scene")
    panel = _panel_id(panel_id)
    scene_digest = hashlib.sha256(scene).hexdigest()
    if scene_digest != _digest(expected_scene_sha256, "expected scene digest"):
        raise ObjectBongardRubricObserverError("scene bytes differ from commitment")
    if not isinstance(rubric_spec, ObjectBongardRubricSpec) or rubric_spec.spec_digest != _digest(expected_rubric_spec_digest, "expected rubric spec digest"):
        raise ObjectBongardRubricObserverError("rubric spec differs from commitment")
    if not isinstance(hypothesis_packet, ObjectHypothesisPacket) or hypothesis_packet.digest() != _digest(expected_hypothesis_packet_digest, "expected hypothesis packet digest"):
        raise ObjectBongardRubricObserverError("hypothesis packet differs from commitment")
    if not isinstance(lineage_packet, ObjectLineagePacket) or lineage_packet.digest() != _digest(expected_lineage_packet_digest, "expected lineage packet digest"):
        raise ObjectBongardRubricObserverError("lineage packet differs from commitment")
    rendered = render_object_hypothesis_atlas(hypothesis_packet, scene)
    verify_object_hypothesis_packet(
        hypothesis_packet,
        expected_png_bytes=scene,
        expected_atlas_png_by_name=dict(rendered),
    )
    verify_object_lineage_packet(lineage_packet, scene)
    if lineage_packet.hypothesis_packet_digest != hypothesis_packet.digest():
        raise ObjectBongardRubricObserverError("lineage/hypothesis packet binding differs")
    if not callable(transport):
        raise TypeError("transport must be callable")

    policy, model_catalog_digest, no_tools_digest = _object_observer._runtime_identities(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    context = (
        observation_context_digest
        if observation_context_digest is not None
        else "sha256:"
        + canonical_digest(
            {
                "schema": "gkm.bongard-object-rubric-observation-context.v1",
                "panel_id": panel,
                "panel_digest": scene_digest,
                "rubric_spec_digest": rubric_spec.spec_digest,
                "hypothesis_packet_digest": hypothesis_packet.digest(),
                "lineage_packet_digest": lineage_packet.digest(),
                "model": model,
                "reasoning_effort": reasoning_effort,
            }
        )
    )
    _address(context, "observation context digest")
    schema = object_bongard_rubric_observer_output_schema()
    by_name = dict(rendered)
    shards: list[ObjectBongardRubricObserverShard] = []
    for sheet in hypothesis_packet.atlas_sheets:
        atlas = by_name[sheet.name]
        presentation_bytes = (("scene.png", scene), (sheet.name, atlas))
        identities = _legacy._image_identities(presentation_bytes)
        prompt = object_bongard_rubric_observer_prompt(rubric_spec, sheet)
        _legacy._assert_model_visible_boundary(
            prompt,
            schema,
            tuple(name for name, _ in presentation_bytes),
            hidden_values=(
                panel,
                scene_digest,
                rubric_spec.spec_digest,
                hypothesis_packet.digest(),
                lineage_packet.digest(),
            ),
        )
        try:
            payload, receipt = _legacy._stage_and_call(
                presentation_bytes,
                prompt=prompt,
                schema=schema,
                model=model,
                reasoning_effort=reasoning_effort,
                minutes=minutes,
                verbose=verbose,
                executable=executable,
                cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
                expected_launcher_digest=expected_launcher_digest,
                model_catalog_snapshot=model_catalog_snapshot,
                no_tools_attestation=no_tools_attestation,
                transport=transport,
            )
        except Exception as exc:
            error_type = _legacy._exception_type(exc)
            slot_scores, scene_score = _failure_scores(
                sheet, "observer_transport_failed", error_type
            )
            shards.append(
                _seal_shard(
                    status=PrototypeSceneObserverStatus.TRANSPORT_ERROR,
                    sheet=sheet,
                    presentation=identities,
                    prompt=prompt,
                    schema=schema,
                    payload=None,
                    receipt=None,
                    slot_scores=slot_scores,
                    scene_score=scene_score,
                    failure_code="observer_transport_failed",
                    failure_type=error_type,
                )
            )
            continue
        try:
            slot_scores, scene_score = _parse_observer_payload(
                payload, tuple(item.slot_id for item in sheet.slots)
            )
        except Exception as exc:
            error_type = _legacy._exception_type(exc)
            slot_scores, scene_score = _failure_scores(
                sheet, "observer_payload_rejected", error_type
            )
            shards.append(
                _seal_shard(
                    status=PrototypeSceneObserverStatus.PARSER_ERROR,
                    sheet=sheet,
                    presentation=identities,
                    prompt=prompt,
                    schema=schema,
                    payload=payload,
                    receipt=receipt,
                    slot_scores=slot_scores,
                    scene_score=scene_score,
                    failure_code="observer_payload_rejected",
                    failure_type=error_type,
                )
            )
            continue
        shards.append(
            _seal_shard(
                status=PrototypeSceneObserverStatus.SUCCESS,
                sheet=sheet,
                presentation=identities,
                prompt=prompt,
                schema=schema,
                payload=payload,
                receipt=receipt,
                slot_scores=slot_scores,
                scene_score=scene_score,
                failure_code=None,
                failure_type=None,
            )
        )
    return _seal_artifact(
        panel_id=panel,
        panel_digest=scene_digest,
        observation_context_digest=context,
        rubric_spec=rubric_spec,
        hypothesis_packet=hypothesis_packet,
        lineage_packet=lineage_packet,
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=policy,
        model_catalog_digest=model_catalog_digest,
        no_tools_attestation_digest=no_tools_digest,
        shards=shards,
    )


def verify_object_bongard_rubric_observer_artifact(
    artifact: ObjectBongardRubricObserverArtifact,
    png_bytes: bytes,
    *,
    panel_id: str,
    rubric_spec: ObjectBongardRubricSpec,
    hypothesis_packet: ObjectHypothesisPacket,
    lineage_packet: ObjectLineagePacket,
    expected_artifact_digest: str,
    expected_runtime_identity_digest: str | None = None,
) -> ObjectBongardRubricObserverArtifact:
    """Cold-replay pixels, packets, payloads, receipts, and projections."""

    restored = ObjectBongardRubricObserverArtifact.from_data(artifact.to_data())
    if restored.artifact_digest != _digest(expected_artifact_digest, "expected artifact digest"):
        raise ObjectBongardRubricObserverError("observer artifact differs from commitment")
    if expected_runtime_identity_digest is not None and restored.runtime_identity_digest != _digest(expected_runtime_identity_digest, "expected runtime digest"):
        raise ObjectBongardRubricObserverError("observer runtime differs from commitment")
    scene = _legacy._validate_exact_png(png_bytes, "scene")
    if (
        restored.panel_id != _panel_id(panel_id)
        or restored.panel_digest != hashlib.sha256(scene).hexdigest()
        or restored.rubric_spec != ObjectBongardRubricSpec.from_data(rubric_spec.to_data())
        or restored.hypothesis_packet != ObjectHypothesisPacket.from_data(hypothesis_packet.to_data())
        or restored.lineage_packet != ObjectLineagePacket.from_data(lineage_packet.to_data())
    ):
        raise ObjectBongardRubricObserverError("observer replay inputs differ")
    rendered = render_object_hypothesis_atlas(hypothesis_packet, scene)
    verify_object_hypothesis_packet(
        hypothesis_packet,
        expected_png_bytes=scene,
        expected_atlas_png_by_name=dict(rendered),
    )
    verify_object_lineage_packet(lineage_packet, scene)
    schema = object_bongard_rubric_observer_output_schema()
    by_name = dict(rendered)
    for sheet, shard in zip(hypothesis_packet.atlas_sheets, restored.shards, strict=True):
        if shard.status is not PrototypeSceneObserverStatus.SUCCESS:
            continue
        assert shard.receipt is not None and shard.model_payload is not None
        prompt = object_bongard_rubric_observer_prompt(rubric_spec, sheet)
        with tempfile.TemporaryDirectory(prefix="bongard-rubric-replay-") as raw:
            directory = Path(raw)
            presentation = (("scene.png", scene), (sheet.name, by_name[sheet.name]))
            paths: list[str] = []
            for name, data in presentation:
                target = directory / name
                target.write_bytes(data)
                paths.append(str(target.resolve()))
            _legacy.validate_codex_named_image_receipt(
                shard.receipt,
                prompt,
                tuple(paths),
                tuple(name for name, _ in presentation),
                schema,
                dict(shard.model_payload),
            )
    return restored


__all__ = (
    "ObjectBongardRubricObserverArtifact",
    "ObjectBongardRubricObserverError",
    "ObjectBongardRubricObserverShard",
    "ObjectBongardRubricSlotScore",
    "ObjectBongardRubricSpec",
    "OrdinalLevelInterval",
    "RUBRIC_OBSERVER_PROTOCOL_ID",
    "RUBRIC_ORDINAL_LEVEL_ANCHORS",
    "RubricObservationState",
    "RubricScope",
    "RubricScopeObservation",
    "object_bongard_rubric_observer_catalog_digest",
    "object_bongard_rubric_observer_model_digest",
    "object_bongard_rubric_observer_output_schema",
    "object_bongard_rubric_observer_prompt",
    "object_bongard_rubric_observer_protocol_digest",
    "object_bongard_rubric_observer_source_digest",
    "object_bongard_rubric_ordinal_scale_digest",
    "observe_object_bongard_rubric",
    "verify_object_bongard_rubric_observer_artifact",
)

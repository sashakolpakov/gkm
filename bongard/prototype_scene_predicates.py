"""Finite Python predicates over calibrated prototype-scene tag results.

The calibrated observer vocabulary is exactly the two opaque tags frozen by
the prototype-pair cohort.  This module adds no prose concepts, negation,
polarity bit, or executable predicate callback.  Optional triangle geometry
is archived as accompanying evidence and is excluded from every decision.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import base64
from dataclasses import dataclass, field
import hashlib
import importlib
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.prototype_pair_cohort import OPAQUE_TAG_IDS
from bongard.prototype_scene_calibration import (
    PrototypeSceneCalibratedResult,
    PrototypeSceneCalibrationFamily,
    PrototypeSceneEvaluationContext,
    PrototypeSceneTagScore,
    evaluate_prototype_scene_score,
)


PREDICATE_SCHEMA = "gkm.bongard-prototype-scene-predicate.v1"
LIBRARY_SCHEMA = "gkm.bongard-prototype-scene-predicate-library.v1"
PANEL_SCHEMA = "gkm.bongard-prototype-scene-panel-evaluation.v1"
OBSERVER_BINDING_SCHEMA = (
    "gkm.bongard-prototype-scene-verified-observer-binding.v1"
)
EVALUATOR_ID = "bongard.prototype-scene-predicates/python-v1"

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}\Z")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_OBSERVER_PNG_BYTES = 4_000_000


class PrototypeScenePredicateError(ValueError):
    """A prototype-scene predicate, library, or panel is not canonical."""


def _triangle_geometry_packet_type() -> type:
    """Load the optional, self-authenticating geometry sidecar only on use.

    The active prototype-scene campaign archives no geometry packet.  Keeping
    this import lazy prevents an absent, nondecisional sidecar from expanding
    the campaign's executable authority closure.  A non-null packet still
    validates through its own source-bound canonical class.
    """

    module = importlib.import_module("bongard.triangle_geometry")
    packet_type = getattr(module, "TriangleGeometryPacket", None)
    if not isinstance(packet_type, type):
        raise PrototypeScenePredicateError(
            "optional triangle geometry authority is unavailable"
        )
    return packet_type


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _require_address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise PrototypeScenePredicateError(f"{label} must be a sha256: address")
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise PrototypeScenePredicateError(f"{label} must be a bounded identifier")
    return value


def _fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PrototypeScenePredicateError(f"{label} fields differ from schema")


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_defines_identity_or_decision": False,
        "lean_required_for_replay": False,
        "lean_removal_changes_decision": False,
    }


def prototype_scene_evaluator_digest() -> str:
    return _address(
        {
            "schema": "gkm.bongard-prototype-scene-evaluator-algorithm.v1",
            "evaluator_id": EVALUATOR_ID,
            "source_sha256": _source_sha256(),
            "opaque_tag_ids": list(OPAQUE_TAG_IDS),
            "atoms": "exactly-one-positive-atom-per-opaque-tag",
            "geometry": "canonical-accompanying-evidence-only-nondecisional",
            "negation": False,
            "polarity_flip": False,
            **_authority_data(),
        }
    )


@dataclass(frozen=True, order=True, slots=True)
class PrototypeScenePredicate:
    tag_id: str

    def __post_init__(self) -> None:
        if self.tag_id not in OPAQUE_TAG_IDS:
            raise PrototypeScenePredicateError("predicate tag is outside opaque pair")

    @property
    def predicate_id(self) -> str:
        return f"prototype-scene:positive:{self.tag_id}"

    @property
    def rubric(self) -> str:
        return (
            "the complete calibrated scene result for opaque tag "
            f"{self.tag_id} is calibrated_present"
        )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": PREDICATE_SCHEMA,
            "predicate_id": self.predicate_id,
            "tag_id": self.tag_id,
            "rubric": self.rubric,
            "positive_only": True,
            "negation": False,
            **_authority_data(),
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypeScenePredicate":
        expected = {
            "schema",
            "predicate_id",
            "tag_id",
            "rubric",
            "positive_only",
            "negation",
            *_authority_data(),
        }
        _fields(value, expected, "prototype-scene predicate")
        result = cls(value["tag_id"])
        if result.to_data() != dict(value):
            raise PrototypeScenePredicateError("predicate is not canonical")
        return result


def _library_content(value: "PrototypeScenePredicateLibrary") -> dict[str, object]:
    return {
        "schema": LIBRARY_SCHEMA,
        "evaluator_id": EVALUATOR_ID,
        "evaluator_digest": value.evaluator_digest,
        "evaluator_source_sha256": value.evaluator_source_sha256,
        "calibration_family_digest": value.calibration_family_digest,
        "cohort_plan_digest": value.cohort_plan_digest,
        "predicates": [item.to_data() for item in value.predicates],
        "closed_before_support": True,
        "complete_positive_atom_inventory": True,
        "negation_available": False,
        "polarity_flip_available": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PrototypeScenePredicateLibrary:
    evaluator_digest: str
    evaluator_source_sha256: str
    calibration_family_digest: str
    cohort_plan_digest: str
    predicates: tuple[PrototypeScenePredicate, PrototypeScenePredicate]
    record_digest: str

    def __post_init__(self) -> None:
        for name in (
            "evaluator_digest",
            "calibration_family_digest",
            "cohort_plan_digest",
            "record_digest",
        ):
            _require_address(getattr(self, name), name)
        if (
            self.evaluator_source_sha256 != _source_sha256()
            or self.evaluator_digest != prototype_scene_evaluator_digest()
            or tuple(item.tag_id for item in self.predicates) != OPAQUE_TAG_IDS
            or self.record_digest != _address(_library_content(self))
        ):
            raise PrototypeScenePredicateError("predicate library identity differs")

    @classmethod
    def freeze(
        cls, family: PrototypeSceneCalibrationFamily
    ) -> "PrototypeScenePredicateLibrary":
        if not isinstance(family, PrototypeSceneCalibrationFamily):
            raise TypeError("family must be PrototypeSceneCalibrationFamily")
        if PrototypeSceneCalibrationFamily.from_data(family.to_data()) != family:
            raise PrototypeScenePredicateError("calibration family is not canonical")
        values: dict[str, object] = {
            "evaluator_digest": prototype_scene_evaluator_digest(),
            "evaluator_source_sha256": _source_sha256(),
            "calibration_family_digest": family.record_digest,
            "cohort_plan_digest": family.cohort_plan_digest,
            "predicates": tuple(PrototypeScenePredicate(tag) for tag in OPAQUE_TAG_IDS),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=_address(_library_content(provisional)),
        )

    def assert_matches_family(
        self, family: PrototypeSceneCalibrationFamily
    ) -> None:
        if PrototypeScenePredicateLibrary.from_data(self.to_data()) != self:
            raise PrototypeScenePredicateError("predicate library changed")
        if (
            not isinstance(family, PrototypeSceneCalibrationFamily)
            or PrototypeSceneCalibrationFamily.from_data(family.to_data()) != family
            or family.record_digest != self.calibration_family_digest
            or family.cohort_plan_digest != self.cohort_plan_digest
        ):
            raise PrototypeScenePredicateError("library calibration family differs")

    def to_data(self) -> dict[str, object]:
        return {**_library_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "PrototypeScenePredicateLibrary":
        expected = {
            "schema",
            "evaluator_id",
            "evaluator_digest",
            "evaluator_source_sha256",
            "calibration_family_digest",
            "cohort_plan_digest",
            "predicates",
            "closed_before_support",
            "complete_positive_atom_inventory",
            "negation_available",
            "polarity_flip_available",
            *_authority_data(),
            "record_digest",
        }
        _fields(value, expected, "prototype-scene library")
        raw = value["predicates"]
        if not isinstance(raw, list) or any(not isinstance(item, Mapping) for item in raw):
            raise PrototypeScenePredicateError("library predicates are malformed")
        result = cls(
            evaluator_digest=value["evaluator_digest"],
            evaluator_source_sha256=value["evaluator_source_sha256"],
            calibration_family_digest=value["calibration_family_digest"],
            cohort_plan_digest=value["cohort_plan_digest"],
            predicates=tuple(PrototypeScenePredicate.from_data(item) for item in raw),
            record_digest=value["record_digest"],
        )
        if result.to_data() != dict(value):
            raise PrototypeScenePredicateError("library is not canonical")
        return result


def _context_digest(context: PrototypeSceneEvaluationContext) -> str:
    return _address(
        {
            "schema": "gkm.bongard-prototype-scene-evaluation-context.v1",
            **context.to_data(),
        }
    )


@dataclass(frozen=True, slots=True)
class PrototypeSceneVerifiedObserverBinding:
    """External verification attestation for the archived score provenance."""

    panel_id: str
    exact_png_digest: str
    observer_artifact_schema: str
    observer_artifact_digest: str
    verifier_id: str
    verifier_digest: str
    score_digests: tuple[str, str]
    context_digest: str
    record_digest: str

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": OBSERVER_BINDING_SCHEMA,
            "panel_id": self.panel_id,
            "exact_png_digest": self.exact_png_digest,
            "observer_artifact_schema": self.observer_artifact_schema,
            "observer_artifact_digest": self.observer_artifact_digest,
            "verifier_id": self.verifier_id,
            "verifier_digest": self.verifier_digest,
            "score_digests": list(self.score_digests),
            "context_digest": self.context_digest,
            "observer_artifact_verified": True,
            "artifact_payload_archived_externally": True,
            **_authority_data(),
        }

    def __post_init__(self) -> None:
        _identifier(self.panel_id, "observer binding panel_id")
        _identifier(self.observer_artifact_schema, "observer artifact schema")
        _identifier(self.verifier_id, "observer verifier_id")
        for name in (
            "exact_png_digest",
            "observer_artifact_digest",
            "verifier_digest",
            "context_digest",
            "record_digest",
        ):
            _require_address(getattr(self, name), name)
        if (
            len(self.score_digests) != 2
            or any(_ADDRESS.fullmatch(item) is None for item in self.score_digests)
            or self.record_digest != _address(self.content_dict())
        ):
            raise PrototypeScenePredicateError("observer binding identity differs")

    @classmethod
    def seal_verified(
        cls,
        *,
        panel_id: str,
        exact_png_bytes: bytes,
        observer_artifact_schema: str,
        observer_artifact_digest: str,
        verifier_id: str,
        verifier_digest: str,
        scores: Sequence[PrototypeSceneTagScore],
        context: PrototypeSceneEvaluationContext,
    ) -> "PrototypeSceneVerifiedObserverBinding":
        frozen_scores = tuple(scores)
        if (
            not isinstance(exact_png_bytes, bytes)
            or not exact_png_bytes.startswith(_PNG_SIGNATURE)
            or not 0 < len(exact_png_bytes) <= _MAX_OBSERVER_PNG_BYTES
            or len(frozen_scores) != 2
        ):
            raise PrototypeScenePredicateError("observer binding inputs differ")
        values: dict[str, object] = {
            "panel_id": panel_id,
            "exact_png_digest": "sha256:"
            + hashlib.sha256(exact_png_bytes).hexdigest(),
            "observer_artifact_schema": observer_artifact_schema,
            "observer_artifact_digest": observer_artifact_digest,
            "verifier_id": verifier_id,
            "verifier_digest": verifier_digest,
            "score_digests": tuple(item.record_digest for item in frozen_scores),
            "context_digest": _context_digest(context),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=_address(provisional.content_dict()),
        )

    def assert_matches(
        self,
        *,
        panel_id: str,
        exact_png_bytes: bytes,
        scores: Sequence[PrototypeSceneTagScore],
        context: PrototypeSceneEvaluationContext,
    ) -> None:
        replay = PrototypeSceneVerifiedObserverBinding.seal_verified(
            panel_id=panel_id,
            exact_png_bytes=exact_png_bytes,
            observer_artifact_schema=self.observer_artifact_schema,
            observer_artifact_digest=self.observer_artifact_digest,
            verifier_id=self.verifier_id,
            verifier_digest=self.verifier_digest,
            scores=scores,
            context=context,
        )
        if replay != self:
            raise PrototypeScenePredicateError("observer binding parents differ")

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "PrototypeSceneVerifiedObserverBinding":
        expected = {
            "schema",
            "panel_id",
            "exact_png_digest",
            "observer_artifact_schema",
            "observer_artifact_digest",
            "verifier_id",
            "verifier_digest",
            "score_digests",
            "context_digest",
            "observer_artifact_verified",
            "artifact_payload_archived_externally",
            *_authority_data(),
            "record_digest",
        }
        _fields(value, expected, "verified observer binding")
        result = cls(
            panel_id=value["panel_id"],
            exact_png_digest=value["exact_png_digest"],
            observer_artifact_schema=value["observer_artifact_schema"],
            observer_artifact_digest=value["observer_artifact_digest"],
            verifier_id=value["verifier_id"],
            verifier_digest=value["verifier_digest"],
            score_digests=tuple(value["score_digests"]),
            context_digest=value["context_digest"],
            record_digest=value["record_digest"],
        )
        if result.to_data() != dict(value):
            raise PrototypeScenePredicateError("observer binding is not canonical")
        return result


def _panel_content(value: "PrototypeScenePanelEvaluation") -> dict[str, object]:
    return {
        "schema": PANEL_SCHEMA,
        "panel_id": value.panel_id,
        "exact_png_base64": base64.b64encode(value.exact_png_bytes).decode("ascii"),
        "exact_png_digest": value.exact_png_digest,
        "observer_binding": value.observer_binding.to_data(),
        "calibration_family_digest": value.calibration_family_digest,
        "context": value.context.to_data(),
        "scores": [item.to_data() for item in value.scores],
        "results": [item.to_data() for item in value.results],
        "typed_geometry": (
            None if value.typed_geometry is None else value.typed_geometry.to_data()
        ),
        "typed_geometry_digest": value.typed_geometry_digest,
        "typed_geometry_is_nondecisional": True,
        "decision_inputs": ["calibration_family", "context", "both_opaque_tag_scores"],
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PrototypeScenePanelEvaluation:
    """Both opaque scores and their complete calibrated scene results."""

    panel_id: str
    exact_png_bytes: bytes = field(repr=False)
    exact_png_digest: str
    observer_binding: PrototypeSceneVerifiedObserverBinding
    calibration_family_digest: str
    context: PrototypeSceneEvaluationContext
    scores: tuple[PrototypeSceneTagScore, PrototypeSceneTagScore]
    results: tuple[PrototypeSceneCalibratedResult, PrototypeSceneCalibratedResult]
    typed_geometry: object | None
    typed_geometry_digest: str | None
    record_digest: str

    def __post_init__(self) -> None:
        _identifier(self.panel_id, "panel_id")
        if (
            not isinstance(self.exact_png_bytes, bytes)
            or not self.exact_png_bytes.startswith(_PNG_SIGNATURE)
            or not 0 < len(self.exact_png_bytes) <= _MAX_OBSERVER_PNG_BYTES
            or self.exact_png_digest
            != "sha256:" + hashlib.sha256(self.exact_png_bytes).hexdigest()
        ):
            raise PrototypeScenePredicateError("panel exact PNG bytes differ")
        if not isinstance(
            self.observer_binding, PrototypeSceneVerifiedObserverBinding
        ):
            raise TypeError("observer_binding must be verified")
        _require_address(self.calibration_family_digest, "calibration_family_digest")
        _require_address(self.record_digest, "panel record_digest")
        if not isinstance(self.context, PrototypeSceneEvaluationContext):
            raise TypeError("context must be PrototypeSceneEvaluationContext")
        if tuple(item.tag_id for item in self.scores) != OPAQUE_TAG_IDS or tuple(
            item.tag_id for item in self.results
        ) != OPAQUE_TAG_IDS:
            raise PrototypeScenePredicateError("panel must contain both tags in order")
        if any(not isinstance(item, PrototypeSceneTagScore) for item in self.scores) or any(
            not isinstance(item, PrototypeSceneCalibratedResult) for item in self.results
        ):
            raise TypeError("panel scores/results must be typed")
        for score, result in zip(self.scores, self.results, strict=True):
            if (
                result.family_digest != self.calibration_family_digest
                or result.score_digest != score.record_digest
                or result.context_digest != _context_digest(self.context)
            ):
                raise PrototypeScenePredicateError("panel result parent differs")
        self.observer_binding.assert_matches(
            panel_id=self.panel_id,
            exact_png_bytes=self.exact_png_bytes,
            scores=self.scores,
            context=self.context,
        )
        if self.typed_geometry is None:
            if self.typed_geometry_digest is not None:
                raise PrototypeScenePredicateError("absent geometry has a digest")
        else:
            packet_type = _triangle_geometry_packet_type()
            if (
                not isinstance(self.typed_geometry, packet_type)
                or packet_type.from_data(self.typed_geometry.to_data())
                != self.typed_geometry
                or self.typed_geometry_digest != self.typed_geometry.digest
            ):
                raise PrototypeScenePredicateError("typed geometry evidence differs")
        if self.record_digest != _address(_panel_content(self)):
            raise PrototypeScenePredicateError("panel evaluation digest differs")

    @classmethod
    def seal(
        cls,
        *,
        panel_id: str,
        exact_png_bytes: bytes,
        observer_binding: PrototypeSceneVerifiedObserverBinding,
        family: PrototypeSceneCalibrationFamily,
        context: PrototypeSceneEvaluationContext,
        scores: Sequence[PrototypeSceneTagScore],
        typed_geometry: object | None = None,
    ) -> "PrototypeScenePanelEvaluation":
        frozen_scores = tuple(scores)
        if len(frozen_scores) != 2:
            raise PrototypeScenePredicateError("panel requires exactly two scores")
        results = tuple(
            evaluate_prototype_scene_score(family, score, context)
            for score in frozen_scores
        )
        values: dict[str, object] = {
            "panel_id": _identifier(panel_id, "panel_id"),
            "exact_png_bytes": exact_png_bytes,
            "exact_png_digest": "sha256:"
            + hashlib.sha256(exact_png_bytes).hexdigest(),
            "observer_binding": observer_binding,
            "calibration_family_digest": family.record_digest,
            "context": context,
            "scores": frozen_scores,
            "results": results,
            "typed_geometry": typed_geometry,
            "typed_geometry_digest": (
                None if typed_geometry is None else typed_geometry.digest
            ),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=_address(_panel_content(provisional)),
        )

    def assert_matches(self, family: PrototypeSceneCalibrationFamily) -> None:
        replay = PrototypeScenePanelEvaluation.seal(
            panel_id=self.panel_id,
            exact_png_bytes=self.exact_png_bytes,
            observer_binding=self.observer_binding,
            family=family,
            context=self.context,
            scores=self.scores,
            typed_geometry=self.typed_geometry,
        )
        if replay != self:
            raise PrototypeScenePredicateError("panel cold evaluation differs")

    def result(self, tag_id: str) -> PrototypeSceneCalibratedResult:
        matches = tuple(item for item in self.results if item.tag_id == tag_id)
        if len(matches) != 1:
            raise PrototypeScenePredicateError("panel result inventory differs")
        return matches[0]

    def to_data(self) -> dict[str, object]:
        return {**_panel_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "PrototypeScenePanelEvaluation":
        expected = {
            "schema",
            "panel_id",
            "exact_png_base64",
            "exact_png_digest",
            "observer_binding",
            "calibration_family_digest",
            "context",
            "scores",
            "results",
            "typed_geometry",
            "typed_geometry_digest",
            "typed_geometry_is_nondecisional",
            "decision_inputs",
            *_authority_data(),
            "record_digest",
        }
        _fields(value, expected, "prototype-scene panel")
        if not isinstance(value["context"], Mapping) or not isinstance(
            value["observer_binding"], Mapping
        ):
            raise PrototypeScenePredicateError("panel context is malformed")
        raw_scores = value["scores"]
        raw_results = value["results"]
        raw_geometry = value["typed_geometry"]
        if (
            not isinstance(raw_scores, list)
            or not isinstance(raw_results, list)
            or any(not isinstance(item, Mapping) for item in (*raw_scores, *raw_results))
            or (raw_geometry is not None and not isinstance(raw_geometry, Mapping))
        ):
            raise PrototypeScenePredicateError("panel children are malformed")
        if not isinstance(value["exact_png_base64"], str):
            raise PrototypeScenePredicateError("panel exact PNG is malformed")
        try:
            exact_png_bytes = base64.b64decode(
                value["exact_png_base64"], validate=True
            )
        except (TypeError, ValueError) as exc:
            raise PrototypeScenePredicateError("panel exact PNG is malformed") from exc
        result = cls(
            panel_id=value["panel_id"],
            exact_png_bytes=exact_png_bytes,
            exact_png_digest=value["exact_png_digest"],
            observer_binding=PrototypeSceneVerifiedObserverBinding.from_data(
                value["observer_binding"]
            ),
            calibration_family_digest=value["calibration_family_digest"],
            context=PrototypeSceneEvaluationContext.from_data(value["context"]),
            scores=tuple(PrototypeSceneTagScore.from_data(item) for item in raw_scores),
            results=tuple(
                PrototypeSceneCalibratedResult.from_data(item) for item in raw_results
            ),
            typed_geometry=(
                None
                if raw_geometry is None
                else _triangle_geometry_packet_type().from_data(raw_geometry)
            ),
            typed_geometry_digest=value["typed_geometry_digest"],
            record_digest=value["record_digest"],
        )
        if result.to_data() != dict(value):
            raise PrototypeScenePredicateError("panel is not canonical")
        return result


__all__ = [
    "EVALUATOR_ID",
    "LIBRARY_SCHEMA",
    "PANEL_SCHEMA",
    "PREDICATE_SCHEMA",
    "PrototypeScenePanelEvaluation",
    "PrototypeScenePredicate",
    "PrototypeScenePredicateError",
    "PrototypeScenePredicateLibrary",
    "PrototypeSceneVerifiedObserverBinding",
    "prototype_scene_evaluator_digest",
]

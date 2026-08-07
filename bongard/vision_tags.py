"""Typed, neutral vision tags bound to an exact loop-scene packet.

This module is the deliberately narrow bridge between free-form vision and
closed Python predicates.  A vision observer may describe a loop as
``bird-like`` and attach a score interval, but that prose is audit/display
text only: it is never executed, never fed to a predicate, and never
constitutes a fact about the depicted world.  Its only rigorous meaning is
operational: the sealed record contains this integer interval for this exact
scenario-qualified loop.

The layer performs no model calls.  A transport supplies the description,
score records, and digests; :func:`seal_vision_tag_output` checks and seals
them against exact PNG bytes and a :class:`LoopScenePacket`.  Receipt, prompt,
model, protocol, and provenance digests in this v1 envelope are opaque,
caller-declared content bindings.  Syntax and content addressing do not prove
that a named model executed; a future transport artifact must validate that
causal chain.  The finite tag catalog is candidate-independent and the record
schema has no task, side, label, candidate, formula, or filesystem-path field.

The prose blacklist below is defense in depth against obvious experiment
metadata and prompt-control text.  It is not an information-flow proof.  The
v1 safety contract therefore requires consumers to keep description prose out
of synthesis and executable inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.loop_scene_witnesses import (
    LoopScenePacket,
    verify_loop_scene_packet,
)


VISION_TAG_OUTPUT_SCHEMA = "gkm.bongard-vision-tag-output.v1"
VISION_TAG_SCORE_SCHEMA = "gkm.bongard-vision-tag-score.v1"
VISION_TAG_INTERVAL_SCHEMA = "gkm.bongard-vision-tag-interval-ppm.v1"
VISION_TAG_CALIBRATION_SCHEMA = "gkm.bongard-vision-tag-calibration.v1"
VISION_TAG_PREDICATE_SCHEMA = "gkm.bongard-closed-vision-tag-predicate.v1"
VISION_TAG_PREDICATE_RESULT_SCHEMA = (
    "gkm.bongard-vision-tag-predicate-result.v1"
)
VISION_TAG_PREDICATE_ALGORITHM_ID = (
    "bongard.vision-tags/calibrated-interval-comparison-v1"
)

# Definitions are operational observer rubrics, not claims of category truth.
VISION_TAG_CATALOG = (
    (
        "geometry.oblique_edges",
        "observer support that the bound loop visibly has non-axis-aligned edges",
    ),
    (
        "gestalt.bird_like",
        "observer support that the bound loop has a bird-like visual gestalt",
    ),
)
VISION_TAG_IDS = tuple(item[0] for item in VISION_TAG_CATALOG)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_OBJECT_ID = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/loop/loop-[0-9]{8}\Z"
)
_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_DESCRIPTION = re.compile(r"[\x20-\x7e]{1,512}\Z")
_MAX_PPM = 1_000_000

# Defense only.  The authoritative boundary is that description prose is not
# consumed by the evaluator or a v1 synthesizer.  These patterns reject the
# most obvious accidental metadata/control leaks without pretending a
# blacklist can prevent covert communication.
_FORBIDDEN_DESCRIPTION_PROSE = (
    re.compile(r"\b(?:pos|neg)[_-][0-9]+\b", re.IGNORECASE),
    re.compile(
        r"\b(?:positive|negative)[ -]+(?:support|example|panel|side|class|label)s?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bsupport[ -]+(?:set|label|side|panel|example|position|index|id)s?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\btask(?:[ -]+(?:id|label|side|role|name|number|index))?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:source|file)[ -]+path\b", re.IGNORECASE),
    re.compile(r"\b(?:candidate|formula)s?\b", re.IGNORECASE),
    re.compile(r"(?:https?://|file://)", re.IGNORECASE),
    re.compile(r"(?:^|\s)(?:~?/|\.\.?/|[A-Za-z]:\\)"),
)
_FORBIDDEN_DESCRIPTION_CONTROL = (
    re.compile(
        r"(?:^|\s)(?:system|developer|assistant|user|tool)\s*:",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:ignore|disregard|override|bypass|forget)\b.{0,48}"
        r"\b(?:instruction|prompt|policy|schema|rule|message)s?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:new|previous|prior|above|following|hidden|system|developer|"
        r"assistant|user|tool)[ -]+(?:instruction|prompt|message|role)s?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:return|output|emit|respond|reply|write)\b.{0,40}"
        r"\b(?:json|schema|score|tag|formula|predicate)s?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:follow|obey|execute)\b.{0,32}"
        r"\b(?:instruction|prompt|message|command)s?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:act as|you are now|switch (?:to )?role)\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?:<\|[^>]{0,64}\|>|\[/?INST\]|```)", re.IGNORECASE),
)


class VisionTagIntegrityError(ValueError):
    """A typed vision record or one of its content bindings is invalid."""


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise VisionTagIntegrityError(
            f"{label} fields differ from the static schema"
        )


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise VisionTagIntegrityError(f"{label} must be a lowercase sha256")
    return value


def _integer(
    value: object, label: str, *, minimum: int = 0, maximum: int = _MAX_PPM
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VisionTagIntegrityError(f"{label} must be an integer")
    if not minimum <= value <= maximum:
        raise VisionTagIntegrityError(
            f"{label} must lie in [{minimum}, {maximum}]"
        )
    return value


def _code(value: object, label: str) -> str:
    if not isinstance(value, str) or _CODE.fullmatch(value) is None:
        raise VisionTagIntegrityError(f"{label} must be a bounded identifier")
    return value


def _object_id(value: object) -> str:
    if not isinstance(value, str) or _OBJECT_ID.fullmatch(value) is None:
        raise VisionTagIntegrityError(
            "vision object_id must be a scenario-qualified loop identity"
        )
    return value


def _description_text(value: object) -> str:
    if (
        not isinstance(value, str)
        or _DESCRIPTION.fullmatch(value) is None
        or value != value.strip()
    ):
        raise VisionTagIntegrityError(
            "vision description must be 1..512 stripped printable ASCII characters"
        )
    if any(pattern.search(value) for pattern in _FORBIDDEN_DESCRIPTION_PROSE):
        raise VisionTagIntegrityError(
            "vision description contains experiment metadata or a source path"
        )
    if any(pattern.search(value) for pattern in _FORBIDDEN_DESCRIPTION_CONTROL):
        raise VisionTagIntegrityError(
            "vision description contains prompt/control-language text"
        )
    return value


def vision_tag_catalog_digest() -> str:
    """Return the content identity of the only admitted tag vocabulary."""

    return canonical_digest(
        {
            "schema": "gkm.bongard-vision-tag-catalog.v1",
            "score_unit": "integer-parts-per-million",
            "score_bounds": [0, _MAX_PPM],
            "semantics": "frozen-observer-support-not-world-truth",
            "tags": [
                {"tag_id": tag_id, "operational_rubric": rubric}
                for tag_id, rubric in VISION_TAG_CATALOG
            ],
        }
    )


def _source_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def vision_tag_predicate_algorithm_digest() -> str:
    """Bind the pure-Python evaluator and its finite comparison semantics."""

    return canonical_digest(
        {
            "algorithm_id": VISION_TAG_PREDICATE_ALGORITHM_ID,
            "source_digest": _source_digest(),
            "tag_catalog_digest": vision_tag_catalog_digest(),
            "present": "score_lower_ppm >= threshold_ppm",
            "certified_absent": "unavailable-in-v1",
            "score_upper_below_threshold": (
                "indeterminate:soft_absence_not_certifiable_v1"
            ),
            "score_interval_overlaps_threshold": "indeterminate",
            "calibration_record_required_for_every_result": True,
            "reference_execution": "python-canonical/v1",
        }
    )


def vision_tag_object_ids(packet: LoopScenePacket) -> tuple[str, ...]:
    """Enumerate every neutral loop identity in the packet, canonically."""

    if not isinstance(packet, LoopScenePacket):
        raise TypeError("packet must be a LoopScenePacket")
    return tuple(
        sorted(
            f"{scenario.scenario_id}/loop/{loop.loop_id}"
            for scenario in packet.scenarios
            for loop in scenario.loops
        )
    )


class VisionTagScoreState(str, Enum):
    """Raw observer state; raw scores can never certify semantic absence."""

    SCORED = "scored"
    INDETERMINATE = "indeterminate"
    ERROR = "error"


@dataclass(frozen=True, order=True, slots=True)
class VisionTagInterval:
    lower_ppm: int
    upper_ppm: int

    def __post_init__(self) -> None:
        lower = _integer(self.lower_ppm, "score lower_ppm")
        upper = _integer(self.upper_ppm, "score upper_ppm")
        if lower > upper:
            raise VisionTagIntegrityError("score lower_ppm exceeds upper_ppm")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": VISION_TAG_INTERVAL_SCHEMA,
            "lower_ppm": self.lower_ppm,
            "upper_ppm": self.upper_ppm,
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "VisionTagInterval":
        _exact_fields(
            value,
            frozenset({"schema", "lower_ppm", "upper_ppm"}),
            "vision tag interval",
        )
        if value["schema"] != VISION_TAG_INTERVAL_SCHEMA:
            raise VisionTagIntegrityError("unsupported vision tag interval")
        result = cls(value["lower_ppm"], value["upper_ppm"])
        if result.to_data() != dict(value):
            raise VisionTagIntegrityError(
                "vision tag interval is not canonically represented"
            )
        return result


@dataclass(frozen=True, order=True, slots=True)
class VisionTagScore:
    """One exhaustive object/tag cell in a neutral observer response."""

    object_id: str
    tag_id: str
    state: VisionTagScoreState
    interval: VisionTagInterval | None = None
    reason_code: str | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        _object_id(self.object_id)
        if self.tag_id not in VISION_TAG_IDS:
            raise VisionTagIntegrityError("vision score uses a tag outside the catalog")
        if not isinstance(self.state, VisionTagScoreState):
            raise TypeError("vision score state must be VisionTagScoreState")
        if self.state is VisionTagScoreState.SCORED:
            if not isinstance(self.interval, VisionTagInterval):
                raise VisionTagIntegrityError("scored tag requires an integer interval")
            if self.reason_code is not None or self.error_type is not None:
                raise VisionTagIntegrityError(
                    "scored tag cannot carry failure metadata"
                )
        elif self.state is VisionTagScoreState.INDETERMINATE:
            if self.interval is not None or self.error_type is not None:
                raise VisionTagIntegrityError(
                    "indeterminate tag cannot carry an interval or error type"
                )
            _code(self.reason_code, "indeterminate reason_code")
        else:
            if self.interval is not None:
                raise VisionTagIntegrityError("error tag cannot carry an interval")
            _code(self.reason_code, "error reason_code")
            _code(self.error_type, "error_type")

    @classmethod
    def scored(
        cls, object_id: str, tag_id: str, lower_ppm: int, upper_ppm: int
    ) -> "VisionTagScore":
        return cls(
            object_id,
            tag_id,
            VisionTagScoreState.SCORED,
            VisionTagInterval(lower_ppm, upper_ppm),
        )

    @classmethod
    def indeterminate(
        cls, object_id: str, tag_id: str, reason_code: str
    ) -> "VisionTagScore":
        return cls(
            object_id,
            tag_id,
            VisionTagScoreState.INDETERMINATE,
            reason_code=reason_code,
        )

    @classmethod
    def error(
        cls, object_id: str, tag_id: str, reason_code: str, error_type: str
    ) -> "VisionTagScore":
        return cls(
            object_id,
            tag_id,
            VisionTagScoreState.ERROR,
            reason_code=reason_code,
            error_type=error_type,
        )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": VISION_TAG_SCORE_SCHEMA,
            "object_id": self.object_id,
            "tag_id": self.tag_id,
            "state": self.state.value,
            "interval": None if self.interval is None else self.interval.to_data(),
            "reason_code": self.reason_code,
            "error_type": self.error_type,
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "VisionTagScore":
        _exact_fields(
            value,
            frozenset(
                {
                    "schema",
                    "object_id",
                    "tag_id",
                    "state",
                    "interval",
                    "reason_code",
                    "error_type",
                }
            ),
            "vision tag score",
        )
        if value["schema"] != VISION_TAG_SCORE_SCHEMA:
            raise VisionTagIntegrityError("unsupported vision tag score")
        raw_interval = value["interval"]
        if raw_interval is not None and not isinstance(raw_interval, Mapping):
            raise VisionTagIntegrityError("vision score interval must be an object or null")
        try:
            state = VisionTagScoreState(value["state"])
        except (TypeError, ValueError) as exc:
            raise VisionTagIntegrityError("unknown vision score state") from exc
        result = cls(
            object_id=value["object_id"],
            tag_id=value["tag_id"],
            state=state,
            interval=(
                None
                if raw_interval is None
                else VisionTagInterval.from_data(raw_interval)
            ),
            reason_code=value["reason_code"],
            error_type=value["error_type"],
        )
        if result.to_data() != dict(value):
            raise VisionTagIntegrityError(
                "vision tag score is not canonically represented"
            )
        return result


def _inventory_digest(object_ids: tuple[str, ...]) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-vision-object-inventory.v1",
            "entity_type": "scenario-qualified-loop",
            "object_ids": list(object_ids),
        }
    )


def _output_preimage(output: "VisionTagOutput") -> dict[str, object]:
    return {
        "schema": VISION_TAG_OUTPUT_SCHEMA,
        "panel_digest": output.panel_digest,
        "loop_scene_packet_digest": output.loop_scene_packet_digest,
        "tag_catalog_digest": output.tag_catalog_digest,
        "object_inventory_digest": output.object_inventory_digest,
        "object_ids": list(output.object_ids),
        "description": output.description,
        "scores": [item.to_data() for item in output.scores],
        "receipt_digest": output.receipt_digest,
        "prompt_digest": output.prompt_digest,
        "model_digest": output.model_digest,
        "protocol_digest": output.protocol_digest,
        "provenance_digest": output.provenance_digest,
    }


@dataclass(frozen=True, slots=True)
class VisionTagOutput:
    """A sealed, exhaustive, exact-panel-bound neutral vision response."""

    panel_digest: str
    loop_scene_packet_digest: str
    tag_catalog_digest: str
    object_inventory_digest: str
    object_ids: tuple[str, ...]
    description: str
    scores: tuple[VisionTagScore, ...]
    receipt_digest: str
    prompt_digest: str
    model_digest: str
    protocol_digest: str
    provenance_digest: str
    record_digest: str
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for label in (
            "panel_digest",
            "loop_scene_packet_digest",
            "tag_catalog_digest",
            "object_inventory_digest",
            "receipt_digest",
            "prompt_digest",
            "model_digest",
            "protocol_digest",
            "provenance_digest",
            "record_digest",
        ):
            _digest(getattr(self, label), label)
        if self.tag_catalog_digest != vision_tag_catalog_digest():
            raise VisionTagIntegrityError("vision output uses another tag catalog")
        if not isinstance(self.object_ids, tuple):
            raise TypeError("vision object_ids must be an immutable tuple")
        for item in self.object_ids:
            _object_id(item)
        if self.object_ids != tuple(sorted(set(self.object_ids))):
            raise VisionTagIntegrityError("vision object_ids must be unique and sorted")
        if self.object_inventory_digest != _inventory_digest(self.object_ids):
            raise VisionTagIntegrityError("vision object inventory digest differs")
        _description_text(self.description)
        if not isinstance(self.scores, tuple) or any(
            not isinstance(item, VisionTagScore) for item in self.scores
        ):
            raise TypeError("vision scores must be an immutable typed tuple")
        expected_pairs = tuple(
            (object_id, tag_id)
            for object_id in self.object_ids
            for tag_id in VISION_TAG_IDS
        )
        actual_pairs = tuple((item.object_id, item.tag_id) for item in self.scores)
        if actual_pairs != expected_pairs:
            raise VisionTagIntegrityError(
                "vision scores must exhaust object x finite-tag inventory in order"
            )
        computed = canonical_digest(_output_preimage(self))
        if self.record_digest != computed:
            raise VisionTagIntegrityError("vision output content digest differs")
        object.__setattr__(self, "_sealed_digest", computed)

    def to_data(self) -> dict[str, object]:
        return {**_output_preimage(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "VisionTagOutput":
        _exact_fields(
            value,
            frozenset(
                {
                    "schema",
                    "panel_digest",
                    "loop_scene_packet_digest",
                    "tag_catalog_digest",
                    "object_inventory_digest",
                    "object_ids",
                    "description",
                    "scores",
                    "receipt_digest",
                    "prompt_digest",
                    "model_digest",
                    "protocol_digest",
                    "provenance_digest",
                    "record_digest",
                }
            ),
            "vision tag output",
        )
        if value["schema"] != VISION_TAG_OUTPUT_SCHEMA:
            raise VisionTagIntegrityError("unsupported vision tag output")
        raw_object_ids = value["object_ids"]
        raw_scores = value["scores"]
        if not isinstance(raw_object_ids, list):
            raise VisionTagIntegrityError("vision object_ids must be a list")
        if not isinstance(raw_scores, list) or any(
            not isinstance(item, Mapping) for item in raw_scores
        ):
            raise VisionTagIntegrityError("vision scores must be an object list")
        result = cls(
            panel_digest=value["panel_digest"],
            loop_scene_packet_digest=value["loop_scene_packet_digest"],
            tag_catalog_digest=value["tag_catalog_digest"],
            object_inventory_digest=value["object_inventory_digest"],
            object_ids=tuple(raw_object_ids),
            description=value["description"],
            scores=tuple(VisionTagScore.from_data(item) for item in raw_scores),
            receipt_digest=value["receipt_digest"],
            prompt_digest=value["prompt_digest"],
            model_digest=value["model_digest"],
            protocol_digest=value["protocol_digest"],
            provenance_digest=value["provenance_digest"],
            record_digest=value["record_digest"],
        )
        if result.to_data() != dict(value):
            raise VisionTagIntegrityError(
                "vision tag output is not canonically represented"
            )
        return result

    def assert_untampered(self) -> None:
        if (
            canonical_digest(_output_preimage(self)) != self.record_digest
            or self.record_digest != self._sealed_digest
        ):
            raise VisionTagIntegrityError("vision tag output changed after sealing")


def seal_vision_tag_output(
    *,
    exact_png_bytes: bytes,
    loop_scene_packet: LoopScenePacket,
    description: str,
    scores: Sequence[VisionTagScore],
    receipt_digest: str,
    prompt_digest: str,
    model_digest: str,
    protocol_digest: str,
    provenance_digest: str,
) -> VisionTagOutput:
    """Seal supplied observations and opaque digest claims; never call a model.

    This function validates exact pixels, loop identities, schemas, and
    content hashes.  It does not validate that ``receipt_digest`` came from a
    model execution.  A future typed transport artifact owns that boundary.
    """

    if not isinstance(exact_png_bytes, bytes):
        raise TypeError("exact_png_bytes must be bytes")
    verify_loop_scene_packet(
        loop_scene_packet, expected_png_bytes=exact_png_bytes
    )
    if isinstance(scores, (str, bytes)) or not isinstance(scores, Sequence):
        raise TypeError("scores must be a sequence")
    ordered_scores = tuple(
        sorted(scores, key=lambda item: (item.object_id, item.tag_id))
    )
    object_ids = vision_tag_object_ids(loop_scene_packet)
    provisional = object.__new__(VisionTagOutput)
    values: dict[str, object] = {
        "panel_digest": hashlib.sha256(exact_png_bytes).hexdigest(),
        "loop_scene_packet_digest": loop_scene_packet.digest(),
        "tag_catalog_digest": vision_tag_catalog_digest(),
        "object_inventory_digest": _inventory_digest(object_ids),
        "object_ids": object_ids,
        "description": description,
        "scores": ordered_scores,
        "receipt_digest": receipt_digest,
        "prompt_digest": prompt_digest,
        "model_digest": model_digest,
        "protocol_digest": protocol_digest,
        "provenance_digest": provenance_digest,
    }
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    record_digest = canonical_digest(_output_preimage(provisional))
    return VisionTagOutput(**values, record_digest=record_digest)


def verify_vision_tag_output(
    output: VisionTagOutput,
    *,
    expected_png_bytes: bytes,
    expected_loop_scene_packet: LoopScenePacket,
    expected_record_digest: str | None = None,
) -> VisionTagOutput:
    """Cold-replay all exact-panel, object-inventory, and content bindings."""

    if not isinstance(output, VisionTagOutput):
        raise TypeError("output must be a VisionTagOutput")
    output.assert_untampered()
    if not isinstance(expected_png_bytes, bytes):
        raise TypeError("expected_png_bytes must be bytes")
    verify_loop_scene_packet(
        expected_loop_scene_packet, expected_png_bytes=expected_png_bytes
    )
    if output.panel_digest != hashlib.sha256(expected_png_bytes).hexdigest():
        raise VisionTagIntegrityError("vision output names another exact panel")
    if output.loop_scene_packet_digest != expected_loop_scene_packet.digest():
        raise VisionTagIntegrityError("vision output names another loop scene packet")
    expected_ids = vision_tag_object_ids(expected_loop_scene_packet)
    if output.object_ids != expected_ids:
        raise VisionTagIntegrityError("vision output names objects absent from packet")
    if expected_record_digest is not None and output.record_digest != _digest(
        expected_record_digest, "expected_record_digest"
    ):
        raise VisionTagIntegrityError("vision output differs from committed digest")
    return output


def _calibration_preimage(
    calibration: "VisionTagCalibration",
) -> dict[str, object]:
    return {
        "schema": VISION_TAG_CALIBRATION_SCHEMA,
        "tag_id": calibration.tag_id,
        "threshold_ppm": calibration.threshold_ppm,
        "tag_catalog_digest": calibration.tag_catalog_digest,
        "prompt_digest": calibration.prompt_digest,
        "model_digest": calibration.model_digest,
        "protocol_digest": calibration.protocol_digest,
        "development_manifest_digest": calibration.development_manifest_digest,
        "calibration_method_digest": calibration.calibration_method_digest,
        "calibration_receipt_digest": calibration.calibration_receipt_digest,
        "provenance_digest": calibration.provenance_digest,
        "absence_authorized": calibration.absence_authorized,
        "absence_authorization_digest": calibration.absence_authorization_digest,
    }


@dataclass(frozen=True, slots=True)
class VisionTagCalibration:
    """Frozen observer threshold; v1 has no certified-absence authority.

    Digest fields are opaque content bindings in this transport-neutral
    module.  In particular, an arbitrary digest cannot authorize absence.
    """

    tag_id: str
    threshold_ppm: int
    tag_catalog_digest: str
    prompt_digest: str
    model_digest: str
    protocol_digest: str
    development_manifest_digest: str
    calibration_method_digest: str
    calibration_receipt_digest: str
    provenance_digest: str
    absence_authorized: bool
    absence_authorization_digest: str | None
    record_digest: str
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.tag_id not in VISION_TAG_IDS:
            raise VisionTagIntegrityError("calibration tag is outside the catalog")
        _integer(self.threshold_ppm, "threshold_ppm")
        for label in (
            "tag_catalog_digest",
            "prompt_digest",
            "model_digest",
            "protocol_digest",
            "development_manifest_digest",
            "calibration_method_digest",
            "calibration_receipt_digest",
            "provenance_digest",
            "record_digest",
        ):
            _digest(getattr(self, label), label)
        if self.tag_catalog_digest != vision_tag_catalog_digest():
            raise VisionTagIntegrityError("calibration uses another tag catalog")
        if not isinstance(self.absence_authorized, bool):
            raise TypeError("absence_authorized must be boolean")
        if self.absence_authorized:
            raise VisionTagIntegrityError(
                "vision-tag v1 cannot authorize certified absence"
            )
        if self.absence_authorization_digest is not None:
            raise VisionTagIntegrityError(
                "vision-tag v1 cannot carry an absence authorization digest"
            )
        computed = canonical_digest(_calibration_preimage(self))
        if self.record_digest != computed:
            raise VisionTagIntegrityError("calibration content digest differs")
        object.__setattr__(self, "_sealed_digest", computed)

    @classmethod
    def create(
        cls,
        *,
        tag_id: str,
        threshold_ppm: int,
        prompt_digest: str,
        model_digest: str,
        protocol_digest: str,
        development_manifest_digest: str,
        calibration_method_digest: str,
        calibration_receipt_digest: str,
        provenance_digest: str,
        absence_authorized: bool,
        absence_authorization_digest: str | None = None,
    ) -> "VisionTagCalibration":
        values: dict[str, object] = {
            "tag_id": tag_id,
            "threshold_ppm": threshold_ppm,
            "tag_catalog_digest": vision_tag_catalog_digest(),
            "prompt_digest": prompt_digest,
            "model_digest": model_digest,
            "protocol_digest": protocol_digest,
            "development_manifest_digest": development_manifest_digest,
            "calibration_method_digest": calibration_method_digest,
            "calibration_receipt_digest": calibration_receipt_digest,
            "provenance_digest": provenance_digest,
            "absence_authorized": absence_authorized,
            "absence_authorization_digest": absence_authorization_digest,
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        record_digest = canonical_digest(_calibration_preimage(provisional))
        return cls(**values, record_digest=record_digest)

    def to_data(self) -> dict[str, object]:
        return {**_calibration_preimage(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "VisionTagCalibration":
        _exact_fields(
            value,
            frozenset(
                {
                    "schema",
                    "tag_id",
                    "threshold_ppm",
                    "tag_catalog_digest",
                    "prompt_digest",
                    "model_digest",
                    "protocol_digest",
                    "development_manifest_digest",
                    "calibration_method_digest",
                    "calibration_receipt_digest",
                    "provenance_digest",
                    "absence_authorized",
                    "absence_authorization_digest",
                    "record_digest",
                }
            ),
            "vision tag calibration",
        )
        if value["schema"] != VISION_TAG_CALIBRATION_SCHEMA:
            raise VisionTagIntegrityError("unsupported vision tag calibration")
        result = cls(
            tag_id=value["tag_id"],
            threshold_ppm=value["threshold_ppm"],
            tag_catalog_digest=value["tag_catalog_digest"],
            prompt_digest=value["prompt_digest"],
            model_digest=value["model_digest"],
            protocol_digest=value["protocol_digest"],
            development_manifest_digest=value["development_manifest_digest"],
            calibration_method_digest=value["calibration_method_digest"],
            calibration_receipt_digest=value["calibration_receipt_digest"],
            provenance_digest=value["provenance_digest"],
            absence_authorized=value["absence_authorized"],
            absence_authorization_digest=value["absence_authorization_digest"],
            record_digest=value["record_digest"],
        )
        if result.to_data() != dict(value):
            raise VisionTagIntegrityError("calibration is not canonically represented")
        return result

    def assert_untampered(self) -> None:
        if (
            canonical_digest(_calibration_preimage(self)) != self.record_digest
            or self.record_digest != self._sealed_digest
        ):
            raise VisionTagIntegrityError("calibration changed after sealing")


def _predicate_preimage(
    predicate: "ClosedVisionTagPredicate",
) -> dict[str, object]:
    return {
        "schema": VISION_TAG_PREDICATE_SCHEMA,
        "tag_id": predicate.tag_id,
        "threshold_ppm": predicate.threshold_ppm,
        "tag_catalog_digest": predicate.tag_catalog_digest,
        "prompt_digest": predicate.prompt_digest,
        "model_digest": predicate.model_digest,
        "protocol_digest": predicate.protocol_digest,
        "calibration_digest": predicate.calibration_digest,
        "evaluation_algorithm_digest": predicate.evaluation_algorithm_digest,
    }


@dataclass(frozen=True, slots=True)
class ClosedVisionTagPredicate:
    """One finite tag/threshold operation with no prose or arbitrary code."""

    tag_id: str
    threshold_ppm: int
    tag_catalog_digest: str
    prompt_digest: str
    model_digest: str
    protocol_digest: str
    calibration_digest: str
    evaluation_algorithm_digest: str
    record_digest: str
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.tag_id not in VISION_TAG_IDS:
            raise VisionTagIntegrityError("predicate tag is outside the catalog")
        _integer(self.threshold_ppm, "predicate threshold_ppm")
        for label in (
            "tag_catalog_digest",
            "prompt_digest",
            "model_digest",
            "protocol_digest",
            "calibration_digest",
            "evaluation_algorithm_digest",
            "record_digest",
        ):
            _digest(getattr(self, label), label)
        if self.tag_catalog_digest != vision_tag_catalog_digest():
            raise VisionTagIntegrityError("predicate uses another tag catalog")
        if self.evaluation_algorithm_digest != vision_tag_predicate_algorithm_digest():
            raise VisionTagIntegrityError("predicate names another evaluator")
        computed = canonical_digest(_predicate_preimage(self))
        if self.record_digest != computed:
            raise VisionTagIntegrityError("predicate content digest differs")
        object.__setattr__(self, "_sealed_digest", computed)

    @classmethod
    def freeze(
        cls, calibration: VisionTagCalibration
    ) -> "ClosedVisionTagPredicate":
        if not isinstance(calibration, VisionTagCalibration):
            raise TypeError("calibration must be a VisionTagCalibration")
        calibration.assert_untampered()
        values: dict[str, object] = {
            "tag_id": calibration.tag_id,
            "threshold_ppm": calibration.threshold_ppm,
            "tag_catalog_digest": calibration.tag_catalog_digest,
            "prompt_digest": calibration.prompt_digest,
            "model_digest": calibration.model_digest,
            "protocol_digest": calibration.protocol_digest,
            "calibration_digest": calibration.record_digest,
            "evaluation_algorithm_digest": vision_tag_predicate_algorithm_digest(),
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        record_digest = canonical_digest(_predicate_preimage(provisional))
        return cls(**values, record_digest=record_digest)

    def to_data(self) -> dict[str, object]:
        return {**_predicate_preimage(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "ClosedVisionTagPredicate":
        _exact_fields(
            value,
            frozenset(
                {
                    "schema",
                    "tag_id",
                    "threshold_ppm",
                    "tag_catalog_digest",
                    "prompt_digest",
                    "model_digest",
                    "protocol_digest",
                    "calibration_digest",
                    "evaluation_algorithm_digest",
                    "record_digest",
                }
            ),
            "closed vision tag predicate",
        )
        if value["schema"] != VISION_TAG_PREDICATE_SCHEMA:
            raise VisionTagIntegrityError("unsupported closed vision tag predicate")
        result = cls(
            tag_id=value["tag_id"],
            threshold_ppm=value["threshold_ppm"],
            tag_catalog_digest=value["tag_catalog_digest"],
            prompt_digest=value["prompt_digest"],
            model_digest=value["model_digest"],
            protocol_digest=value["protocol_digest"],
            calibration_digest=value["calibration_digest"],
            evaluation_algorithm_digest=value["evaluation_algorithm_digest"],
            record_digest=value["record_digest"],
        )
        if result.to_data() != dict(value):
            raise VisionTagIntegrityError("predicate is not canonically represented")
        return result

    def assert_untampered(self) -> None:
        if (
            canonical_digest(_predicate_preimage(self)) != self.record_digest
            or self.record_digest != self._sealed_digest
        ):
            raise VisionTagIntegrityError("predicate changed after sealing")


def _result_preimage(result: "VisionTagPredicateResult") -> dict[str, object]:
    return {
        "schema": VISION_TAG_PREDICATE_RESULT_SCHEMA,
        "disposition": result.disposition.value,
        "object_id": result.object_id,
        "tag_id": result.tag_id,
        "threshold_ppm": result.threshold_ppm,
        "score_interval": (
            None
            if result.score_interval is None
            else result.score_interval.to_data()
        ),
        "output_digest": result.output_digest,
        "predicate_digest": result.predicate_digest,
        "calibration_digest": result.calibration_digest,
        "certificate": result.certificate,
        "reason_code": result.reason_code,
        "error_type": result.error_type,
    }


@dataclass(frozen=True, slots=True)
class VisionTagPredicateResult:
    """Content-addressed v1 result of one closed object/tag comparison."""

    disposition: Disposition
    object_id: str
    tag_id: str
    threshold_ppm: int
    score_interval: VisionTagInterval | None
    output_digest: str
    predicate_digest: str
    calibration_digest: str
    record_digest: str
    certificate: str | None = None
    reason_code: str | None = None
    error_type: str | None = None
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, Disposition):
            raise TypeError("predicate result disposition must be Disposition")
        if self.disposition is Disposition.CERTIFIED_ABSENT:
            raise VisionTagIntegrityError(
                "vision-tag v1 cannot emit certified absence"
            )
        _object_id(self.object_id)
        if self.tag_id not in VISION_TAG_IDS:
            raise VisionTagIntegrityError("result tag is outside the catalog")
        _integer(self.threshold_ppm, "result threshold_ppm")
        _digest(self.output_digest, "result output_digest")
        _digest(self.predicate_digest, "result predicate_digest")
        _digest(self.record_digest, "result record_digest")
        _digest(self.calibration_digest, "result calibration_digest")
        if self.disposition is Disposition.PRESENT:
            if self.score_interval is None or any(
                item is not None
                for item in (self.certificate, self.reason_code, self.error_type)
            ):
                raise VisionTagIntegrityError("present result fields are inconsistent")
        elif self.disposition is Disposition.INDETERMINATE:
            _code(self.reason_code, "result reason_code")
            if self.certificate is not None or self.error_type is not None:
                raise VisionTagIntegrityError(
                    "indeterminate result fields are inconsistent"
                )
        else:
            _code(self.reason_code, "result reason_code")
            _code(self.error_type, "result error_type")
            if self.score_interval is not None or self.certificate is not None:
                raise VisionTagIntegrityError("error result fields are inconsistent")
        computed = canonical_digest(_result_preimage(self))
        if self.record_digest != computed:
            raise VisionTagIntegrityError("predicate result content digest differs")
        object.__setattr__(self, "_sealed_digest", computed)

    @classmethod
    def create(
        cls,
        *,
        disposition: Disposition,
        object_id: str,
        tag_id: str,
        threshold_ppm: int,
        score_interval: VisionTagInterval | None,
        output_digest: str,
        predicate_digest: str,
        calibration_digest: str,
        certificate: str | None = None,
        reason_code: str | None = None,
        error_type: str | None = None,
    ) -> "VisionTagPredicateResult":
        values: dict[str, object] = {
            "disposition": disposition,
            "object_id": object_id,
            "tag_id": tag_id,
            "threshold_ppm": threshold_ppm,
            "score_interval": score_interval,
            "output_digest": output_digest,
            "predicate_digest": predicate_digest,
            "calibration_digest": calibration_digest,
            "certificate": certificate,
            "reason_code": reason_code,
            "error_type": error_type,
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        record_digest = canonical_digest(_result_preimage(provisional))
        return cls(**values, record_digest=record_digest)

    def to_data(self) -> dict[str, object]:
        return {**_result_preimage(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "VisionTagPredicateResult":
        _exact_fields(
            value,
            frozenset(
                {
                    "schema",
                    "disposition",
                    "object_id",
                    "tag_id",
                    "threshold_ppm",
                    "score_interval",
                    "output_digest",
                    "predicate_digest",
                    "calibration_digest",
                    "record_digest",
                    "certificate",
                    "reason_code",
                    "error_type",
                }
            ),
            "vision tag predicate result",
        )
        if value["schema"] != VISION_TAG_PREDICATE_RESULT_SCHEMA:
            raise VisionTagIntegrityError("unsupported vision tag predicate result")
        raw_interval = value["score_interval"]
        if raw_interval is not None and not isinstance(raw_interval, Mapping):
            raise VisionTagIntegrityError(
                "predicate result score_interval must be an object or null"
            )
        try:
            disposition = Disposition(value["disposition"])
        except (TypeError, ValueError) as exc:
            raise VisionTagIntegrityError(
                "unknown predicate result disposition"
            ) from exc
        result = cls(
            disposition=disposition,
            object_id=value["object_id"],
            tag_id=value["tag_id"],
            threshold_ppm=value["threshold_ppm"],
            score_interval=(
                None
                if raw_interval is None
                else VisionTagInterval.from_data(raw_interval)
            ),
            output_digest=value["output_digest"],
            predicate_digest=value["predicate_digest"],
            calibration_digest=value["calibration_digest"],
            record_digest=value["record_digest"],
            certificate=value["certificate"],
            reason_code=value["reason_code"],
            error_type=value["error_type"],
        )
        if result.to_data() != dict(value):
            raise VisionTagIntegrityError(
                "predicate result is not canonically represented"
            )
        return result

    def assert_untampered(self) -> None:
        if (
            canonical_digest(_result_preimage(self)) != self.record_digest
            or self.record_digest != self._sealed_digest
        ):
            raise VisionTagIntegrityError(
                "predicate result changed after sealing"
            )


def _verify_calibration_binding(
    predicate: ClosedVisionTagPredicate,
    calibration: VisionTagCalibration,
) -> None:
    calibration.assert_untampered()
    if calibration.record_digest != predicate.calibration_digest:
        raise VisionTagIntegrityError("supplied calibration differs from predicate")
    if (
        calibration.tag_id,
        calibration.threshold_ppm,
        calibration.tag_catalog_digest,
        calibration.prompt_digest,
        calibration.model_digest,
        calibration.protocol_digest,
    ) != (
        predicate.tag_id,
        predicate.threshold_ppm,
        predicate.tag_catalog_digest,
        predicate.prompt_digest,
        predicate.model_digest,
        predicate.protocol_digest,
    ):
        raise VisionTagIntegrityError("predicate and calibration bindings differ")


def evaluate_vision_tag_predicate(
    output: VisionTagOutput,
    predicate: ClosedVisionTagPredicate,
    *,
    object_id: str,
    expected_png_bytes: bytes,
    expected_loop_scene_packet: LoopScenePacket,
    expected_output_digest: str,
    calibration: VisionTagCalibration,
) -> VisionTagPredicateResult:
    """Evaluate one committed score interval conservatively.

    ``expected_output_digest`` is mandatory so a caller cannot silently
    re-seal different scores.  It is still only a content commitment; this v1
    module does not claim the opaque receipt fields authenticate a model run.
    """

    verify_vision_tag_output(
        output,
        expected_png_bytes=expected_png_bytes,
        expected_loop_scene_packet=expected_loop_scene_packet,
        expected_record_digest=expected_output_digest,
    )
    if not isinstance(predicate, ClosedVisionTagPredicate):
        raise TypeError("predicate must be a ClosedVisionTagPredicate")
    predicate.assert_untampered()
    object_id = _object_id(object_id)
    if object_id not in output.object_ids:
        raise VisionTagIntegrityError("predicate object is absent from vision output")
    if (
        output.tag_catalog_digest,
        output.prompt_digest,
        output.model_digest,
        output.protocol_digest,
    ) != (
        predicate.tag_catalog_digest,
        predicate.prompt_digest,
        predicate.model_digest,
        predicate.protocol_digest,
    ):
        raise VisionTagIntegrityError("vision output and predicate observer differ")
    if not isinstance(calibration, VisionTagCalibration):
        raise TypeError("calibration must be a VisionTagCalibration")
    _verify_calibration_binding(predicate, calibration)

    score = next(
        item
        for item in output.scores
        if item.object_id == object_id and item.tag_id == predicate.tag_id
    )
    common: dict[str, object] = {
        "object_id": object_id,
        "tag_id": predicate.tag_id,
        "threshold_ppm": predicate.threshold_ppm,
        "output_digest": output.record_digest,
        "predicate_digest": predicate.record_digest,
        "calibration_digest": calibration.record_digest,
    }
    if score.state is VisionTagScoreState.ERROR:
        return VisionTagPredicateResult.create(
            disposition=Disposition.ERROR,
            score_interval=None,
            reason_code=score.reason_code,
            error_type=score.error_type,
            **common,
        )
    if score.state is VisionTagScoreState.INDETERMINATE:
        return VisionTagPredicateResult.create(
            disposition=Disposition.INDETERMINATE,
            score_interval=None,
            reason_code=score.reason_code,
            **common,
        )
    interval = score.interval
    assert interval is not None
    if interval.lower_ppm >= predicate.threshold_ppm:
        return VisionTagPredicateResult.create(
            disposition=Disposition.PRESENT,
            score_interval=interval,
            **common,
        )
    if interval.upper_ppm < predicate.threshold_ppm:
        return VisionTagPredicateResult.create(
            disposition=Disposition.INDETERMINATE,
            score_interval=interval,
            reason_code="soft_absence_not_certifiable_v1",
            **common,
        )
    return VisionTagPredicateResult.create(
        disposition=Disposition.INDETERMINATE,
        score_interval=interval,
        reason_code="score_interval_overlaps_threshold",
        **common,
    )


def verify_vision_tag_predicate_result(
    result: VisionTagPredicateResult,
    *,
    output: VisionTagOutput,
    predicate: ClosedVisionTagPredicate,
    expected_png_bytes: bytes,
    expected_loop_scene_packet: LoopScenePacket,
    expected_output_digest: str,
    calibration: VisionTagCalibration,
    expected_result_digest: str | None = None,
) -> VisionTagPredicateResult:
    """Cold-replay a serialized result from all exact committed parents."""

    if not isinstance(result, VisionTagPredicateResult):
        raise TypeError("result must be a VisionTagPredicateResult")
    result.assert_untampered()
    if expected_result_digest is not None and result.record_digest != _digest(
        expected_result_digest, "expected_result_digest"
    ):
        raise VisionTagIntegrityError(
            "predicate result differs from committed digest"
        )
    replay = evaluate_vision_tag_predicate(
        output,
        predicate,
        object_id=result.object_id,
        expected_png_bytes=expected_png_bytes,
        expected_loop_scene_packet=expected_loop_scene_packet,
        expected_output_digest=expected_output_digest,
        calibration=calibration,
    )
    if replay != result:
        raise VisionTagIntegrityError(
            "predicate result differs from exact committed replay"
        )
    return result

"""Pure-Python fixed-threshold calibration for prototype-conditioned scenes.

The calibration plan is frozen after the six prototype references exist but
before any of the 28 scheduled calibration scenes is opened or scored.  Every
scheduled Basic-pair scene contributes one task-level cluster and is scored
once for both opaque tags.  Technical failures stay in the applicable
direction's denominator and count as errors; a valid numerical score interval
between the two frozen thresholds is an abstention, not a false decision.  An
abstention nevertheless fails the precommitted coverage gate: every one of the
four tag/direction cells must make a decisive call on all 14 clusters before a
calibration family can be certified.

The statistical claim is deliberately narrow.  It is conditional on the
calibration and selected drill scenes being rendered by the same pinned Basic
renderer and on the precommitted observer environment remaining identical.
It is targeted engineering evidence, not an official benchmark or a general
claim that generator identity proves arbitrary prose semantics.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from bongard.canonical import canonical_digest
from bongard.cluster_binomial import (
    familywise_clopper_pearson_upper_ppm,
    fixed_threshold_cluster_algorithm_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.prototype_pair_cohort import (
    CALIBRATION_CLUSTERS_PER_TAG,
    CONFIDENCE_LEVEL_PPM,
    HYPOTHESIS_COUNT,
    OPAQUE_TAG_IDS,
    TARGETED_ENGINEERING_TOLERANCE_PPM,
    PrototypePairCohortPlan,
)


PPM_SCALE = 1_000_000
# Audit the bounded physical work used to produce one scene-level observation;
# this count is never a statistical sample size or cluster multiplier.
MAX_PHYSICAL_OBSERVER_CALLS_PER_SCENE = 4_096
PLAN_SCHEMA = "gkm.bongard-prototype-scene-calibration-plan.v1"
THRESHOLD_SCHEMA = "gkm.bongard-prototype-scene-threshold.v1"
SCENE_SCHEMA = "gkm.bongard-prototype-scene-calibration-scene.v1"
TAG_SCORE_SCHEMA = "gkm.bongard-prototype-scene-tag-score.v1"
OBSERVATION_SCHEMA = "gkm.bongard-prototype-scene-calibration-observation.v1"
BOUND_SCHEMA = "gkm.bongard-prototype-scene-direction-bound.v2"
ASSESSMENT_SCHEMA = "gkm.bongard-prototype-scene-calibration-assessment.v2"
FAMILY_SCHEMA = "gkm.bongard-prototype-scene-calibration-family.v2"
RESULT_SCHEMA = "gkm.bongard-prototype-scene-calibrated-result.v1"
THRESHOLD_COMMITMENT_SCHEMA = (
    "gkm.bongard-prototype-scene-threshold-commitment.v1"
)
CALIBRATION_ALGORITHM_ID = (
    "bongard.prototype-scene/frozen-cross-calibration-cluster-cp-coverage-v2"
)
OBSERVER_ADAPTER_PROTOCOL_ID = (
    "bongard.prototype-scene-observer/calibration-adapter-v1"
)

SAME_BASIC_RENDERER_CONDITIONAL_ASSUMPTION = (
    "Conditional targeted-engineering transport only: the 28 positive-side "
    "calibration scenes and selected drill scenes must come from the same "
    "hash-pinned ShapeBongard release, Basic sampler, painter/renderer, "
    "description catalog, prototype reference set, observer protocol, model, "
    "and runtime environment.  This does not establish exchangeability with "
    "other renderers, datasets, prose concepts, validation, or official test."
)

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}\Z")

# The source identity is computed in the importing process.  A cold process
# therefore gives edited calibration code a different algorithm identity.
CALIBRATION_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class PrototypeSceneCalibrationError(ValueError):
    """A freeze, observation, fit, evaluator, or replay invariant failed."""


class PrototypeSceneScoreStatus(str, Enum):
    SCORE = "score"
    PARSER_ERROR = "parser_error"
    TRANSPORT_ERROR = "transport_error"
    MISSING = "missing"
    INDETERMINATE = "indeterminate"
    ERROR = "error"


class PrototypeSceneDisposition(str, Enum):
    CALIBRATED_PRESENT = "calibrated_present"
    CALIBRATED_ABSENT = "calibrated_absent"
    INDETERMINATE = "indeterminate"
    ERROR = "error"


class CalibrationDirection(str, Enum):
    FALSE_PRESENT = "false_present_on_expected_absent"
    FALSE_ABSENT = "false_absent_on_expected_present"


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _require_address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise PrototypeSceneCalibrationError(f"{label} must be a sha256: address")
    return value


def _require_raw_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_SHA256.fullmatch(value) is None:
        raise PrototypeSceneCalibrationError(f"{label} must be lowercase SHA-256")
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise PrototypeSceneCalibrationError(f"{label} must be a bounded identifier")
    return value


def _text(value: object, label: str, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8", errors="strict")) > maximum
    ):
        raise PrototypeSceneCalibrationError(f"{label} must be bounded text")
    return value


def _integer(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise PrototypeSceneCalibrationError(f"{label} is outside its integer domain")
    return value


def _object(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise PrototypeSceneCalibrationError(f"{label} must be an object")
    if set(value) != fields:
        raise PrototypeSceneCalibrationError(
            f"{label} fields differ: missing={sorted(fields - set(value))}, "
            f"extra={sorted(set(value) - fields)}"
        )
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PrototypeSceneCalibrationError(f"{label} must be a list")
    return value


def _verify_digest(raw: Mapping[str, Any], label: str) -> None:
    digest = _require_address(raw["record_digest"], f"{label} digest")
    body = {key: value for key, value in raw.items() if key != "record_digest"}
    if digest != _address(body):
        raise PrototypeSceneCalibrationError(f"{label} digest differs")


def calibration_algorithm_digest() -> str:
    """Bind exact source bytes and every decision/statistical convention."""

    return _address(
        {
            "schema": "gkm.bongard-prototype-scene-calibration-algorithm.v1",
            "source_sha256": CALIBRATION_SOURCE_SHA256,
            "algorithm_id": CALIBRATION_ALGORITHM_ID,
            "schemas": [
                PLAN_SCHEMA,
                THRESHOLD_SCHEMA,
                SCENE_SCHEMA,
                TAG_SCORE_SCHEMA,
                OBSERVATION_SCHEMA,
                BOUND_SCHEMA,
                ASSESSMENT_SCHEMA,
                FAMILY_SCHEMA,
                RESULT_SCHEMA,
            ],
            "observer_adapter_protocol_id": OBSERVER_ADAPTER_PROTOCOL_ID,
            "tag_ids": list(OPAQUE_TAG_IDS),
            "threshold_rule": "absent_upper_ppm < present_lower_ppm",
            "present_decision": "score.lower_ppm >= present_lower_ppm",
            "absent_decision": "score.upper_ppm <= absent_upper_ppm",
            "between_thresholds": "indeterminate-abstention-not-error",
            "coverage_gate": (
                "each-tag-direction-abstention-cluster-count-equals-zero"
            ),
            "abstention_blocks_family_certification": True,
            "technical_score_states_count_as_errors": [
                status.value
                for status in PrototypeSceneScoreStatus
                if status is not PrototypeSceneScoreStatus.SCORE
            ],
            "cluster_unit": "official-basic-task-id",
            "clusters_per_direction": CALIBRATION_CLUSTERS_PER_TAG,
            "hypothesis_count": HYPOTHESIS_COUNT,
            "confidence_level_ppm": CONFIDENCE_LEVEL_PPM,
            "maximum_conditional_error_ppm": (
                TARGETED_ENGINEERING_TOLERANCE_PPM
            ),
            "bound_algorithm_digest": fixed_threshold_cluster_algorithm_digest(),
            "subset_selection": False,
            "posthoc_threshold_selection": False,
            "posthoc_tag_selection": False,
            "population_assumption": SAME_BASIC_RENDERER_CONDITIONAL_ASSUMPTION,
            "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
            "python_is_authority": True,
            "lean_required": False,
            "lean_affects_identity_or_decision": False,
        }
    )


@dataclass(frozen=True, slots=True)
class PrototypeSceneTagThreshold:
    tag_id: str
    absent_upper_ppm: int
    present_lower_ppm: int

    def __post_init__(self) -> None:
        if self.tag_id not in OPAQUE_TAG_IDS:
            raise PrototypeSceneCalibrationError("threshold tag is outside cohort")
        absent = _integer(
            self.absent_upper_ppm, "absent upper threshold", maximum=PPM_SCALE
        )
        present = _integer(
            self.present_lower_ppm, "present lower threshold", maximum=PPM_SCALE
        )
        if absent >= present:
            raise PrototypeSceneCalibrationError(
                "thresholds require absent_upper_ppm < present_lower_ppm"
            )

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": THRESHOLD_SCHEMA,
            "tag_id": self.tag_id,
            "absent_upper_ppm": self.absent_upper_ppm,
            "present_lower_ppm": self.present_lower_ppm,
        }

    @property
    def record_digest(self) -> str:
        return _address(self.content_dict())

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypeSceneTagThreshold":
        raw = _object(
            value,
            {
                "schema",
                "tag_id",
                "absent_upper_ppm",
                "present_lower_ppm",
                "record_digest",
            },
            "tag threshold",
        )
        _verify_digest(raw, "tag threshold")
        if raw["schema"] != THRESHOLD_SCHEMA:
            raise PrototypeSceneCalibrationError("threshold schema differs")
        result = cls(
            tag_id=raw["tag_id"],
            absent_upper_ppm=raw["absent_upper_ppm"],
            present_lower_ppm=raw["present_lower_ppm"],
        )
        if result.to_data() != dict(raw):
            raise PrototypeSceneCalibrationError("threshold is not canonical")
        return result


def threshold_commitment(
    thresholds: Sequence[PrototypeSceneTagThreshold],
) -> str:
    values = tuple(thresholds)
    if (
        tuple(item.tag_id for item in values) != OPAQUE_TAG_IDS
        or len(values) != len(OPAQUE_TAG_IDS)
    ):
        raise PrototypeSceneCalibrationError(
            "thresholds must cover both opaque tags in frozen order"
        )
    return _address(
        {
            "schema": THRESHOLD_COMMITMENT_SCHEMA,
            "thresholds": [item.to_data() for item in values],
        }
    )


@dataclass(frozen=True, slots=True)
class PrototypeSceneCalibrationScene:
    ordinal: int
    task_id: str
    cluster_id: str
    panel_id: str
    panel_index: int
    expected_tag_states: tuple[tuple[str, str], tuple[str, str]]

    def __post_init__(self) -> None:
        _integer(self.ordinal, "scene ordinal")
        _identifier(self.task_id, "scene task ID")
        if self.cluster_id != self.task_id:
            raise PrototypeSceneCalibrationError("task ID must be the cluster ID")
        _text(self.panel_id, "scene panel ID")
        _integer(self.panel_index, "scene panel index", maximum=6)
        if (
            tuple(tag_id for tag_id, _state in self.expected_tag_states)
            != OPAQUE_TAG_IDS
            or sorted(state for _tag_id, state in self.expected_tag_states)
            != ["absent", "present"]
        ):
            raise PrototypeSceneCalibrationError("scene expected states differ")

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": SCENE_SCHEMA,
            "ordinal": self.ordinal,
            "task_id": self.task_id,
            "cluster_id": self.cluster_id,
            "panel_id": self.panel_id,
            "panel_index": self.panel_index,
            "expected_tag_states": [
                {"tag_id": tag_id, "state": state}
                for tag_id, state in self.expected_tag_states
            ],
        }

    @property
    def record_digest(self) -> str:
        return _address(self.content_dict())

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "PrototypeSceneCalibrationScene":
        raw = _object(
            value,
            {
                "schema",
                "ordinal",
                "task_id",
                "cluster_id",
                "panel_id",
                "panel_index",
                "expected_tag_states",
                "record_digest",
            },
            "calibration scene",
        )
        _verify_digest(raw, "calibration scene")
        if raw["schema"] != SCENE_SCHEMA:
            raise PrototypeSceneCalibrationError("scene schema differs")
        states: list[tuple[str, str]] = []
        for row in _list(raw["expected_tag_states"], "expected states"):
            item = _object(row, {"tag_id", "state"}, "expected state")
            states.append((item["tag_id"], item["state"]))
        result = cls(
            ordinal=raw["ordinal"],
            task_id=raw["task_id"],
            cluster_id=raw["cluster_id"],
            panel_id=raw["panel_id"],
            panel_index=raw["panel_index"],
            expected_tag_states=tuple(states),  # type: ignore[arg-type]
        )
        if result.to_data() != dict(raw):
            raise PrototypeSceneCalibrationError("scene is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class PrototypeSceneCalibrationPlan:
    cohort_plan_digest: str
    cohort_planner_algorithm_digest: str
    threshold_commitment: str
    thresholds: tuple[PrototypeSceneTagThreshold, PrototypeSceneTagThreshold]
    scenes: tuple[PrototypeSceneCalibrationScene, ...]
    description_catalog_digest: str
    prototype_reference_digest: str
    observer_protocol_id: str
    observer_protocol_digest: str
    model_id: str
    model_identity_digest: str
    environment_digest: str
    release_descriptor_digest: str
    corpus_manifest_digest: str
    basic_sampler_sha256: str
    basic_generator_sha256: str
    same_basic_renderer_population_valid: bool
    conditional_transport_assumption_accepted: bool
    observer_environment_valid: bool
    general_transport_validated: bool
    created_before_calibration_observations: bool
    hypothesis_count: int
    confidence_level_ppm: int
    maximum_conditional_error_ppm: int
    posthoc_subset_selection_allowed: bool
    posthoc_threshold_selection_allowed: bool
    posthoc_tag_selection_allowed: bool
    population_assumption: str
    calibration_source_sha256: str
    calibration_algorithm_digest: str
    predicate_authority_id: str
    python_is_canonical_authority: bool
    lean_required: bool
    lean_defines_identity_or_decision: bool
    lean_required_for_replay: bool

    def __post_init__(self) -> None:
        for name in (
            "cohort_plan_digest",
            "cohort_planner_algorithm_digest",
            "threshold_commitment",
            "description_catalog_digest",
            "prototype_reference_digest",
            "observer_protocol_digest",
            "model_identity_digest",
            "environment_digest",
            "release_descriptor_digest",
            "corpus_manifest_digest",
            "calibration_algorithm_digest",
        ):
            _require_address(getattr(self, name), name)
        _require_raw_sha(self.basic_sampler_sha256, "Basic sampler SHA-256")
        _require_raw_sha(self.basic_generator_sha256, "Basic generator SHA-256")
        _require_raw_sha(self.calibration_source_sha256, "calibration source SHA-256")
        _identifier(self.observer_protocol_id, "observer protocol ID")
        _text(self.model_id, "model ID")
        if tuple(item.tag_id for item in self.thresholds) != OPAQUE_TAG_IDS:
            raise PrototypeSceneCalibrationError("threshold inventory differs")
        if self.threshold_commitment != threshold_commitment(self.thresholds):
            raise PrototypeSceneCalibrationError("threshold commitment differs")
        if (
            len(self.scenes) != 2 * CALIBRATION_CLUSTERS_PER_TAG
            or tuple(item.ordinal for item in self.scenes)
            != tuple(range(2 * CALIBRATION_CLUSTERS_PER_TAG))
            or len({item.task_id for item in self.scenes}) != len(self.scenes)
            or len({item.panel_id for item in self.scenes}) != len(self.scenes)
        ):
            raise PrototypeSceneCalibrationError("calibration scene inventory differs")
        for tag_id in OPAQUE_TAG_IDS:
            states = [dict(item.expected_tag_states)[tag_id] for item in self.scenes]
            if states.count("present") != 14 or states.count("absent") != 14:
                raise PrototypeSceneCalibrationError(
                    "each tag requires 14 present and 14 absent clusters"
                )
        if (
            self.same_basic_renderer_population_valid is not True
            or self.conditional_transport_assumption_accepted is not True
            or self.observer_environment_valid is not True
            or self.general_transport_validated is not False
            or self.created_before_calibration_observations is not True
            or self.hypothesis_count != HYPOTHESIS_COUNT
            or self.confidence_level_ppm != CONFIDENCE_LEVEL_PPM
            or self.maximum_conditional_error_ppm
            != TARGETED_ENGINEERING_TOLERANCE_PPM
            or self.posthoc_subset_selection_allowed is not False
            or self.posthoc_threshold_selection_allowed is not False
            or self.posthoc_tag_selection_allowed is not False
            or self.population_assumption
            != SAME_BASIC_RENDERER_CONDITIONAL_ASSUMPTION
            or self.calibration_source_sha256 != CALIBRATION_SOURCE_SHA256
            or self.calibration_algorithm_digest != calibration_algorithm_digest()
            or self.predicate_authority_id != PYTHON_PREDICATE_AUTHORITY_ID
            or self.python_is_canonical_authority is not True
            or self.lean_required is not False
            or self.lean_defines_identity_or_decision is not False
            or self.lean_required_for_replay is not False
        ):
            raise PrototypeSceneCalibrationError(
                "calibration scientific or runtime authority differs"
            )

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": PLAN_SCHEMA,
            "cohort": {
                "plan_digest": self.cohort_plan_digest,
                "planner_algorithm_digest": self.cohort_planner_algorithm_digest,
                "release_descriptor_digest": self.release_descriptor_digest,
                "corpus_manifest_digest": self.corpus_manifest_digest,
                "basic_sampler_sha256": self.basic_sampler_sha256,
                "basic_generator_sha256": self.basic_generator_sha256,
            },
            "threshold_commitment": self.threshold_commitment,
            "thresholds": [item.to_data() for item in self.thresholds],
            "scenes": [item.to_data() for item in self.scenes],
            "observer_identity": {
                "description_catalog_digest": self.description_catalog_digest,
                "prototype_reference_digest": self.prototype_reference_digest,
                "observer_protocol_id": self.observer_protocol_id,
                "observer_protocol_digest": self.observer_protocol_digest,
                "model_id": self.model_id,
                "model_identity_digest": self.model_identity_digest,
                "environment_digest": self.environment_digest,
            },
            "validity": {
                "same_basic_renderer_population_valid": (
                    self.same_basic_renderer_population_valid
                ),
                "conditional_transport_assumption_accepted": (
                    self.conditional_transport_assumption_accepted
                ),
                "observer_environment_valid": self.observer_environment_valid,
                "general_transport_validated": self.general_transport_validated,
                "created_before_calibration_observations": (
                    self.created_before_calibration_observations
                ),
                "population_assumption": self.population_assumption,
            },
            "statistics": {
                "hypothesis_count": self.hypothesis_count,
                "confidence_level_ppm": self.confidence_level_ppm,
                "maximum_conditional_error_ppm": (
                    self.maximum_conditional_error_ppm
                ),
                "clusters_per_direction": CALIBRATION_CLUSTERS_PER_TAG,
                "posthoc_subset_selection_allowed": (
                    self.posthoc_subset_selection_allowed
                ),
                "posthoc_threshold_selection_allowed": (
                    self.posthoc_threshold_selection_allowed
                ),
                "posthoc_tag_selection_allowed": (
                    self.posthoc_tag_selection_allowed
                ),
            },
            "runtime_authority": {
                "calibration_source_sha256": self.calibration_source_sha256,
                "calibration_algorithm_digest": self.calibration_algorithm_digest,
                "predicate_authority_id": self.predicate_authority_id,
                "python_is_canonical_authority": self.python_is_canonical_authority,
                "lean_required": self.lean_required,
                "lean_defines_identity_or_decision": (
                    self.lean_defines_identity_or_decision
                ),
                "lean_required_for_replay": self.lean_required_for_replay,
            },
        }

    @property
    def record_digest(self) -> str:
        return _address(self.content_dict())

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypeSceneCalibrationPlan":
        raw = _object(
            value,
            {
                "schema",
                "cohort",
                "threshold_commitment",
                "thresholds",
                "scenes",
                "observer_identity",
                "validity",
                "statistics",
                "runtime_authority",
                "record_digest",
            },
            "calibration plan",
        )
        _verify_digest(raw, "calibration plan")
        if raw["schema"] != PLAN_SCHEMA:
            raise PrototypeSceneCalibrationError("calibration plan schema differs")
        cohort = _object(
            raw["cohort"],
            {
                "plan_digest",
                "planner_algorithm_digest",
                "release_descriptor_digest",
                "corpus_manifest_digest",
                "basic_sampler_sha256",
                "basic_generator_sha256",
            },
            "calibration cohort identity",
        )
        observer = _object(
            raw["observer_identity"],
            {
                "description_catalog_digest",
                "prototype_reference_digest",
                "observer_protocol_id",
                "observer_protocol_digest",
                "model_id",
                "model_identity_digest",
                "environment_digest",
            },
            "calibration observer identity",
        )
        validity = _object(
            raw["validity"],
            {
                "same_basic_renderer_population_valid",
                "conditional_transport_assumption_accepted",
                "observer_environment_valid",
                "general_transport_validated",
                "created_before_calibration_observations",
                "population_assumption",
            },
            "calibration validity",
        )
        statistics = _object(
            raw["statistics"],
            {
                "hypothesis_count",
                "confidence_level_ppm",
                "maximum_conditional_error_ppm",
                "clusters_per_direction",
                "posthoc_subset_selection_allowed",
                "posthoc_threshold_selection_allowed",
                "posthoc_tag_selection_allowed",
            },
            "calibration statistics",
        )
        runtime = _object(
            raw["runtime_authority"],
            {
                "calibration_source_sha256",
                "calibration_algorithm_digest",
                "predicate_authority_id",
                "python_is_canonical_authority",
                "lean_required",
                "lean_defines_identity_or_decision",
                "lean_required_for_replay",
            },
            "calibration runtime authority",
        )
        result = cls(
            cohort_plan_digest=cohort["plan_digest"],
            cohort_planner_algorithm_digest=cohort["planner_algorithm_digest"],
            threshold_commitment=raw["threshold_commitment"],
            thresholds=tuple(
                PrototypeSceneTagThreshold.from_data(item)
                for item in _list(raw["thresholds"], "calibration thresholds")
            ),  # type: ignore[arg-type]
            scenes=tuple(
                PrototypeSceneCalibrationScene.from_data(item)
                for item in _list(raw["scenes"], "calibration scenes")
            ),
            description_catalog_digest=observer["description_catalog_digest"],
            prototype_reference_digest=observer["prototype_reference_digest"],
            observer_protocol_id=observer["observer_protocol_id"],
            observer_protocol_digest=observer["observer_protocol_digest"],
            model_id=observer["model_id"],
            model_identity_digest=observer["model_identity_digest"],
            environment_digest=observer["environment_digest"],
            release_descriptor_digest=cohort["release_descriptor_digest"],
            corpus_manifest_digest=cohort["corpus_manifest_digest"],
            basic_sampler_sha256=cohort["basic_sampler_sha256"],
            basic_generator_sha256=cohort["basic_generator_sha256"],
            same_basic_renderer_population_valid=validity[
                "same_basic_renderer_population_valid"
            ],
            conditional_transport_assumption_accepted=validity[
                "conditional_transport_assumption_accepted"
            ],
            observer_environment_valid=validity["observer_environment_valid"],
            general_transport_validated=validity["general_transport_validated"],
            created_before_calibration_observations=validity[
                "created_before_calibration_observations"
            ],
            hypothesis_count=statistics["hypothesis_count"],
            confidence_level_ppm=statistics["confidence_level_ppm"],
            maximum_conditional_error_ppm=statistics[
                "maximum_conditional_error_ppm"
            ],
            posthoc_subset_selection_allowed=statistics[
                "posthoc_subset_selection_allowed"
            ],
            posthoc_threshold_selection_allowed=statistics[
                "posthoc_threshold_selection_allowed"
            ],
            posthoc_tag_selection_allowed=statistics[
                "posthoc_tag_selection_allowed"
            ],
            population_assumption=validity["population_assumption"],
            calibration_source_sha256=runtime["calibration_source_sha256"],
            calibration_algorithm_digest=runtime[
                "calibration_algorithm_digest"
            ],
            predicate_authority_id=runtime["predicate_authority_id"],
            python_is_canonical_authority=runtime[
                "python_is_canonical_authority"
            ],
            lean_required=runtime["lean_required"],
            lean_defines_identity_or_decision=runtime[
                "lean_defines_identity_or_decision"
            ],
            lean_required_for_replay=runtime["lean_required_for_replay"],
        )
        if statistics["clusters_per_direction"] != CALIBRATION_CLUSTERS_PER_TAG:
            raise PrototypeSceneCalibrationError("cluster count policy differs")
        if result.to_data() != dict(raw):
            raise PrototypeSceneCalibrationError("calibration plan is not canonical")
        return result


def create_prototype_scene_calibration_plan(
    *,
    cohort_plan: PrototypePairCohortPlan | Mapping[str, Any],
    thresholds: Sequence[PrototypeSceneTagThreshold],
    description_catalog_digest: str,
    prototype_reference_digest: str,
    observer_protocol_id: str,
    observer_protocol_digest: str,
    model_id: str,
    model_identity_digest: str,
    environment_digest: str,
    expected_cohort_plan_digest: str,
    expected_threshold_commitment: str,
    expected_description_catalog_digest: str,
    expected_prototype_reference_digest: str,
    expected_observer_protocol_digest: str,
    expected_model_identity_digest: str,
    expected_environment_digest: str,
) -> PrototypeSceneCalibrationPlan:
    """Freeze thresholds, identities, and all scene clusters before scoring."""

    cohort = (
        cohort_plan
        if isinstance(cohort_plan, PrototypePairCohortPlan)
        else PrototypePairCohortPlan.from_data(cohort_plan)
    )
    if cohort.record_digest != _require_address(
        expected_cohort_plan_digest, "expected cohort plan digest"
    ):
        raise PrototypeSceneCalibrationError("cohort plan differs from commitment")
    threshold_values = tuple(thresholds)
    threshold_pin = _require_address(
        expected_threshold_commitment, "expected threshold commitment"
    )
    if threshold_commitment(threshold_values) != threshold_pin:
        raise PrototypeSceneCalibrationError("thresholds differ from commitment")
    identity_pairs = (
        (
            description_catalog_digest,
            expected_description_catalog_digest,
            "description catalog",
        ),
        (
            prototype_reference_digest,
            expected_prototype_reference_digest,
            "prototype reference",
        ),
        (
            observer_protocol_digest,
            expected_observer_protocol_digest,
            "observer protocol",
        ),
        (
            model_identity_digest,
            expected_model_identity_digest,
            "model identity",
        ),
        (environment_digest, expected_environment_digest, "environment"),
    )
    for actual, expected, label in identity_pairs:
        if _require_address(actual, label) != _require_address(
            expected, f"expected {label}"
        ):
            raise PrototypeSceneCalibrationError(f"{label} differs from commitment")
    _identifier(observer_protocol_id, "observer protocol ID")
    _text(model_id, "model ID")
    scenes = tuple(
        PrototypeSceneCalibrationScene(
            ordinal=ordinal,
            task_id=item.task_id,
            cluster_id=item.cluster_id,
            panel_id=item.panel_id,
            panel_index=item.panel_index,
            expected_tag_states=item.expected_tag_states,
        )
        for ordinal, item in enumerate(cohort.calibration_clusters)
    )
    return PrototypeSceneCalibrationPlan(
        cohort_plan_digest=cohort.record_digest,
        cohort_planner_algorithm_digest=cohort.planner_algorithm_digest,
        threshold_commitment=threshold_pin,
        thresholds=threshold_values,  # type: ignore[arg-type]
        scenes=scenes,
        description_catalog_digest=description_catalog_digest,
        prototype_reference_digest=prototype_reference_digest,
        observer_protocol_id=observer_protocol_id,
        observer_protocol_digest=observer_protocol_digest,
        model_id=model_id,
        model_identity_digest=model_identity_digest,
        environment_digest=environment_digest,
        release_descriptor_digest=cohort.release_descriptor_digest,
        corpus_manifest_digest=cohort.corpus_manifest_digest,
        basic_sampler_sha256=cohort.basic_sampler_sha256,
        basic_generator_sha256=cohort.basic_generator_sha256,
        same_basic_renderer_population_valid=True,
        conditional_transport_assumption_accepted=True,
        observer_environment_valid=True,
        general_transport_validated=False,
        created_before_calibration_observations=True,
        hypothesis_count=HYPOTHESIS_COUNT,
        confidence_level_ppm=CONFIDENCE_LEVEL_PPM,
        maximum_conditional_error_ppm=TARGETED_ENGINEERING_TOLERANCE_PPM,
        posthoc_subset_selection_allowed=False,
        posthoc_threshold_selection_allowed=False,
        posthoc_tag_selection_allowed=False,
        population_assumption=SAME_BASIC_RENDERER_CONDITIONAL_ASSUMPTION,
        calibration_source_sha256=CALIBRATION_SOURCE_SHA256,
        calibration_algorithm_digest=calibration_algorithm_digest(),
        predicate_authority_id=PYTHON_PREDICATE_AUTHORITY_ID,
        python_is_canonical_authority=True,
        lean_required=False,
        lean_defines_identity_or_decision=False,
        lean_required_for_replay=False,
    )


@dataclass(frozen=True, slots=True)
class PrototypeSceneTagScore:
    tag_id: str
    status: PrototypeSceneScoreStatus
    lower_ppm: int | None
    upper_ppm: int | None
    reason_code: str
    error_type: str | None

    def __post_init__(self) -> None:
        if self.tag_id not in OPAQUE_TAG_IDS:
            raise PrototypeSceneCalibrationError("score tag is outside cohort")
        if not isinstance(self.status, PrototypeSceneScoreStatus):
            raise TypeError("score status must be typed")
        _identifier(self.reason_code, "score reason code")
        if self.status is PrototypeSceneScoreStatus.SCORE:
            lower = _integer(self.lower_ppm, "score lower ppm", maximum=PPM_SCALE)
            upper = _integer(self.upper_ppm, "score upper ppm", maximum=PPM_SCALE)
            if lower > upper or self.error_type is not None:
                raise PrototypeSceneCalibrationError("scored interval is malformed")
        elif (
            self.lower_ppm is not None
            or self.upper_ppm is not None
            or not isinstance(self.error_type, str)
            or not self.error_type
        ):
            raise PrototypeSceneCalibrationError(
                "technical score state requires null interval and error_type"
            )

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": TAG_SCORE_SCHEMA,
            "tag_id": self.tag_id,
            "status": self.status.value,
            "lower_ppm": self.lower_ppm,
            "upper_ppm": self.upper_ppm,
            "reason_code": self.reason_code,
            "error_type": self.error_type,
        }

    @property
    def record_digest(self) -> str:
        return _address(self.content_dict())

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypeSceneTagScore":
        raw = _object(
            value,
            {
                "schema",
                "tag_id",
                "status",
                "lower_ppm",
                "upper_ppm",
                "reason_code",
                "error_type",
                "record_digest",
            },
            "prototype scene tag score",
        )
        _verify_digest(raw, "prototype scene tag score")
        if raw["schema"] != TAG_SCORE_SCHEMA:
            raise PrototypeSceneCalibrationError("tag score schema differs")
        try:
            status = PrototypeSceneScoreStatus(raw["status"])
        except (TypeError, ValueError) as exc:
            raise PrototypeSceneCalibrationError("tag score status differs") from exc
        result = cls(
            tag_id=raw["tag_id"],
            status=status,
            lower_ppm=raw["lower_ppm"],
            upper_ppm=raw["upper_ppm"],
            reason_code=raw["reason_code"],
            error_type=raw["error_type"],
        )
        if result.to_data() != dict(raw):
            raise PrototypeSceneCalibrationError("tag score is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class PrototypeSceneCalibrationObservation:
    calibration_plan_digest: str
    cohort_plan_digest: str
    task_id: str
    panel_id: str
    observer_artifact_digest: str
    observer_artifact_schema: str
    description_catalog_digest: str
    prototype_reference_digest: str
    observer_protocol_id: str
    observer_protocol_digest: str
    model_id: str
    model_identity_digest: str
    environment_digest: str
    observer_call_count: int
    scores: tuple[PrototypeSceneTagScore, PrototypeSceneTagScore]
    adapter_protocol_id: str

    def __post_init__(self) -> None:
        for name in (
            "calibration_plan_digest",
            "cohort_plan_digest",
            "observer_artifact_digest",
            "description_catalog_digest",
            "prototype_reference_digest",
            "observer_protocol_digest",
            "model_identity_digest",
            "environment_digest",
        ):
            _require_address(getattr(self, name), name)
        _identifier(self.task_id, "observation task ID")
        _text(self.panel_id, "observation panel ID")
        _identifier(self.observer_artifact_schema, "observer artifact schema")
        _identifier(self.observer_protocol_id, "observer protocol ID")
        _text(self.model_id, "observer model ID")
        _integer(
            self.observer_call_count,
            "physical observer call count",
            minimum=1,
            maximum=MAX_PHYSICAL_OBSERVER_CALLS_PER_SCENE,
        )
        if (
            tuple(item.tag_id for item in self.scores) != OPAQUE_TAG_IDS
            or any(not isinstance(item, PrototypeSceneTagScore) for item in self.scores)
            or self.adapter_protocol_id != OBSERVER_ADAPTER_PROTOCOL_ID
        ):
            raise PrototypeSceneCalibrationError(
                "observation must score both tags once in frozen order"
            )

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": OBSERVATION_SCHEMA,
            "calibration_plan_digest": self.calibration_plan_digest,
            "cohort_plan_digest": self.cohort_plan_digest,
            "task_id": self.task_id,
            "panel_id": self.panel_id,
            "observer_artifact_digest": self.observer_artifact_digest,
            "observer_artifact_schema": self.observer_artifact_schema,
            "description_catalog_digest": self.description_catalog_digest,
            "prototype_reference_digest": self.prototype_reference_digest,
            "observer_protocol_id": self.observer_protocol_id,
            "observer_protocol_digest": self.observer_protocol_digest,
            "model_id": self.model_id,
            "model_identity_digest": self.model_identity_digest,
            "environment_digest": self.environment_digest,
            "observer_call_count": self.observer_call_count,
            "scores": [item.to_data() for item in self.scores],
            "adapter_protocol_id": self.adapter_protocol_id,
        }

    @property
    def record_digest(self) -> str:
        return _address(self.content_dict())

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "PrototypeSceneCalibrationObservation":
        raw = _object(
            value,
            {
                "schema",
                "calibration_plan_digest",
                "cohort_plan_digest",
                "task_id",
                "panel_id",
                "observer_artifact_digest",
                "observer_artifact_schema",
                "description_catalog_digest",
                "prototype_reference_digest",
                "observer_protocol_id",
                "observer_protocol_digest",
                "model_id",
                "model_identity_digest",
                "environment_digest",
                "observer_call_count",
                "scores",
                "adapter_protocol_id",
                "record_digest",
            },
            "calibration observation",
        )
        _verify_digest(raw, "calibration observation")
        if raw["schema"] != OBSERVATION_SCHEMA:
            raise PrototypeSceneCalibrationError("observation schema differs")
        result = cls(
            calibration_plan_digest=raw["calibration_plan_digest"],
            cohort_plan_digest=raw["cohort_plan_digest"],
            task_id=raw["task_id"],
            panel_id=raw["panel_id"],
            observer_artifact_digest=raw["observer_artifact_digest"],
            observer_artifact_schema=raw["observer_artifact_schema"],
            description_catalog_digest=raw["description_catalog_digest"],
            prototype_reference_digest=raw["prototype_reference_digest"],
            observer_protocol_id=raw["observer_protocol_id"],
            observer_protocol_digest=raw["observer_protocol_digest"],
            model_id=raw["model_id"],
            model_identity_digest=raw["model_identity_digest"],
            environment_digest=raw["environment_digest"],
            observer_call_count=raw["observer_call_count"],
            scores=tuple(
                PrototypeSceneTagScore.from_data(item)
                for item in _list(raw["scores"], "observation scores")
            ),  # type: ignore[arg-type]
            adapter_protocol_id=raw["adapter_protocol_id"],
        )
        if result.to_data() != dict(raw):
            raise PrototypeSceneCalibrationError("observation is not canonical")
        return result


@runtime_checkable
class PrototypeSceneCalibrationArtifactAdapter(Protocol):
    """Temporary exact hook for the separately versioned observer artifact."""

    def to_calibration_observation_data(
        self, *, calibration_plan_digest: str
    ) -> Mapping[str, Any]: ...


def adapt_prototype_scene_observation(
    value: PrototypeSceneCalibrationObservation
    | Mapping[str, Any]
    | PrototypeSceneCalibrationArtifactAdapter,
    *,
    calibration_plan_digest: str,
) -> PrototypeSceneCalibrationObservation:
    """Adapt a verified observer artifact through one explicit typed hook.

    The forthcoming observer module must expose
    ``to_calibration_observation_data`` and, critically, its scene artifact
    must already bind ``calibration_plan_digest``.  Adding that digest only in
    this adapter after model output would not prove pre-observation freezing
    and is rejected by the reconstructed observation's equality checks.
    """

    expected = _require_address(calibration_plan_digest, "calibration plan digest")
    if isinstance(value, PrototypeSceneCalibrationObservation):
        result = value
    elif isinstance(value, Mapping):
        result = PrototypeSceneCalibrationObservation.from_data(value)
    elif isinstance(value, PrototypeSceneCalibrationArtifactAdapter):
        raw = value.to_calibration_observation_data(
            calibration_plan_digest=expected
        )
        result = PrototypeSceneCalibrationObservation.from_data(raw)
    else:
        raise PrototypeSceneCalibrationError(
            "observer artifact lacks the frozen calibration adapter hook"
        )
    if result.calibration_plan_digest != expected:
        raise PrototypeSceneCalibrationError(
            "observation does not bind the frozen calibration plan"
        )
    return result


def _threshold_map(
    plan: PrototypeSceneCalibrationPlan,
) -> dict[str, PrototypeSceneTagThreshold]:
    return {item.tag_id: item for item in plan.thresholds}


def _score_disposition(
    score: PrototypeSceneTagScore,
    threshold: PrototypeSceneTagThreshold,
) -> PrototypeSceneDisposition:
    if score.status is PrototypeSceneScoreStatus.INDETERMINATE:
        return PrototypeSceneDisposition.INDETERMINATE
    if score.status is not PrototypeSceneScoreStatus.SCORE:
        return PrototypeSceneDisposition.ERROR
    assert score.lower_ppm is not None and score.upper_ppm is not None
    if score.lower_ppm >= threshold.present_lower_ppm:
        return PrototypeSceneDisposition.CALIBRATED_PRESENT
    if score.upper_ppm <= threshold.absent_upper_ppm:
        return PrototypeSceneDisposition.CALIBRATED_ABSENT
    return PrototypeSceneDisposition.INDETERMINATE


@dataclass(frozen=True, slots=True)
class PrototypeSceneDirectionBound:
    tag_id: str
    direction: CalibrationDirection
    cluster_count: int
    error_cluster_count: int
    abstention_cluster_count: int
    correct_decision_cluster_count: int
    conditional_error_upper_ppm: int
    confidence_level_ppm: int
    hypothesis_count: int
    maximum_conditional_error_ppm: int
    coverage_gate_accepted: bool
    accepted: bool
    cluster_ids: tuple[str, ...]
    error_cluster_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.tag_id not in OPAQUE_TAG_IDS or not isinstance(
            self.direction, CalibrationDirection
        ):
            raise PrototypeSceneCalibrationError("direction bound identity differs")
        n = _integer(self.cluster_count, "cluster count", minimum=1)
        errors = _integer(
            self.error_cluster_count, "error count", maximum=n
        )
        abstentions = _integer(
            self.abstention_cluster_count, "abstention count", maximum=n
        )
        correct = _integer(
            self.correct_decision_cluster_count, "correct count", maximum=n
        )
        if errors + abstentions + correct != n:
            raise PrototypeSceneCalibrationError(
                "direction disposition counts do not exhaust denominator"
            )
        expected_upper = familywise_clopper_pearson_upper_ppm(
            cluster_count=n,
            error_cluster_count=errors,
            confidence_level_ppm=self.confidence_level_ppm,
            hypothesis_count=self.hypothesis_count,
        )
        if (
            n != CALIBRATION_CLUSTERS_PER_TAG
            or self.conditional_error_upper_ppm != expected_upper
            or self.confidence_level_ppm != CONFIDENCE_LEVEL_PPM
            or self.hypothesis_count != HYPOTHESIS_COUNT
            or self.maximum_conditional_error_ppm
            != TARGETED_ENGINEERING_TOLERANCE_PPM
            or self.coverage_gate_accepted is not (abstentions == 0)
            or self.accepted
            is not (
                expected_upper <= self.maximum_conditional_error_ppm
                and self.coverage_gate_accepted
            )
            or len(self.cluster_ids) != n
            or len(set(self.cluster_ids)) != n
            or self.error_cluster_ids
            != tuple(cluster_id for cluster_id in self.cluster_ids if cluster_id in set(self.error_cluster_ids))
            or len(self.error_cluster_ids) != errors
        ):
            raise PrototypeSceneCalibrationError("direction bound differs")

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": BOUND_SCHEMA,
            "tag_id": self.tag_id,
            "direction": self.direction.value,
            "cluster_count": self.cluster_count,
            "error_cluster_count": self.error_cluster_count,
            "abstention_cluster_count": self.abstention_cluster_count,
            "correct_decision_cluster_count": self.correct_decision_cluster_count,
            "conditional_error_upper_ppm": self.conditional_error_upper_ppm,
            "confidence_level_ppm": self.confidence_level_ppm,
            "hypothesis_count": self.hypothesis_count,
            "maximum_conditional_error_ppm": self.maximum_conditional_error_ppm,
            "coverage_gate_accepted": self.coverage_gate_accepted,
            "accepted": self.accepted,
            "cluster_ids": list(self.cluster_ids),
            "error_cluster_ids": list(self.error_cluster_ids),
        }

    @property
    def record_digest(self) -> str:
        return _address(self.content_dict())

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypeSceneDirectionBound":
        raw = _object(
            value,
            {
                "schema",
                "tag_id",
                "direction",
                "cluster_count",
                "error_cluster_count",
                "abstention_cluster_count",
                "correct_decision_cluster_count",
                "conditional_error_upper_ppm",
                "confidence_level_ppm",
                "hypothesis_count",
                "maximum_conditional_error_ppm",
                "coverage_gate_accepted",
                "accepted",
                "cluster_ids",
                "error_cluster_ids",
                "record_digest",
            },
            "direction bound",
        )
        _verify_digest(raw, "direction bound")
        if raw["schema"] != BOUND_SCHEMA:
            raise PrototypeSceneCalibrationError("bound schema differs")
        try:
            direction = CalibrationDirection(raw["direction"])
        except (TypeError, ValueError) as exc:
            raise PrototypeSceneCalibrationError("bound direction differs") from exc
        result = cls(
            tag_id=raw["tag_id"],
            direction=direction,
            cluster_count=raw["cluster_count"],
            error_cluster_count=raw["error_cluster_count"],
            abstention_cluster_count=raw["abstention_cluster_count"],
            correct_decision_cluster_count=raw[
                "correct_decision_cluster_count"
            ],
            conditional_error_upper_ppm=raw["conditional_error_upper_ppm"],
            confidence_level_ppm=raw["confidence_level_ppm"],
            hypothesis_count=raw["hypothesis_count"],
            maximum_conditional_error_ppm=raw[
                "maximum_conditional_error_ppm"
            ],
            coverage_gate_accepted=raw["coverage_gate_accepted"],
            accepted=raw["accepted"],
            cluster_ids=tuple(_list(raw["cluster_ids"], "bound cluster IDs")),
            error_cluster_ids=tuple(
                _list(raw["error_cluster_ids"], "bound error cluster IDs")
            ),
        )
        if result.to_data() != dict(raw):
            raise PrototypeSceneCalibrationError("bound is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class PrototypeSceneCalibrationAssessment:
    calibration_plan_digest: str
    observation_set_digest: str
    observation_count: int
    observation_record_digests: tuple[str, ...]
    bounds: tuple[PrototypeSceneDirectionBound, ...]
    all_four_coverage_gates_accepted: bool
    all_four_bounds_accepted: bool
    complete_observation_set: bool
    every_scene_scored_once_for_both_tags: bool

    def __post_init__(self) -> None:
        _require_address(self.calibration_plan_digest, "assessment plan digest")
        _require_address(self.observation_set_digest, "observation set digest")
        if (
            self.observation_count != 2 * CALIBRATION_CLUSTERS_PER_TAG
            or len(self.observation_record_digests) != self.observation_count
            or any(
                _ADDRESS.fullmatch(value) is None
                for value in self.observation_record_digests
            )
            or len(self.bounds) != HYPOTHESIS_COUNT
            or tuple((item.tag_id, item.direction) for item in self.bounds)
            != tuple(
                (tag_id, direction)
                for tag_id in OPAQUE_TAG_IDS
                for direction in (
                    CalibrationDirection.FALSE_PRESENT,
                    CalibrationDirection.FALSE_ABSENT,
                )
            )
            or self.all_four_coverage_gates_accepted
            is not all(item.coverage_gate_accepted for item in self.bounds)
            or self.all_four_bounds_accepted
            is not all(item.accepted for item in self.bounds)
            or self.complete_observation_set is not True
            or self.every_scene_scored_once_for_both_tags is not True
        ):
            raise PrototypeSceneCalibrationError("calibration assessment differs")

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": ASSESSMENT_SCHEMA,
            "calibration_plan_digest": self.calibration_plan_digest,
            "observation_set_digest": self.observation_set_digest,
            "observation_count": self.observation_count,
            "observation_record_digests": list(self.observation_record_digests),
            "bounds": [item.to_data() for item in self.bounds],
            "all_four_coverage_gates_accepted": (
                self.all_four_coverage_gates_accepted
            ),
            "all_four_bounds_accepted": self.all_four_bounds_accepted,
            "complete_observation_set": self.complete_observation_set,
            "every_scene_scored_once_for_both_tags": (
                self.every_scene_scored_once_for_both_tags
            ),
        }

    @property
    def record_digest(self) -> str:
        return _address(self.content_dict())

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "PrototypeSceneCalibrationAssessment":
        raw = _object(
            value,
            {
                "schema",
                "calibration_plan_digest",
                "observation_set_digest",
                "observation_count",
                "observation_record_digests",
                "bounds",
                "all_four_coverage_gates_accepted",
                "all_four_bounds_accepted",
                "complete_observation_set",
                "every_scene_scored_once_for_both_tags",
                "record_digest",
            },
            "calibration assessment",
        )
        _verify_digest(raw, "calibration assessment")
        if raw["schema"] != ASSESSMENT_SCHEMA:
            raise PrototypeSceneCalibrationError("assessment schema differs")
        result = cls(
            calibration_plan_digest=raw["calibration_plan_digest"],
            observation_set_digest=raw["observation_set_digest"],
            observation_count=raw["observation_count"],
            observation_record_digests=tuple(
                _list(
                    raw["observation_record_digests"],
                    "observation record digests",
                )
            ),
            bounds=tuple(
                PrototypeSceneDirectionBound.from_data(item)
                for item in _list(raw["bounds"], "assessment bounds")
            ),
            all_four_coverage_gates_accepted=raw[
                "all_four_coverage_gates_accepted"
            ],
            all_four_bounds_accepted=raw["all_four_bounds_accepted"],
            complete_observation_set=raw["complete_observation_set"],
            every_scene_scored_once_for_both_tags=raw[
                "every_scene_scored_once_for_both_tags"
            ],
        )
        if result.to_data() != dict(raw):
            raise PrototypeSceneCalibrationError("assessment is not canonical")
        return result


def assess_prototype_scene_calibration(
    plan: PrototypeSceneCalibrationPlan | Mapping[str, Any],
    observations: Sequence[
        PrototypeSceneCalibrationObservation
        | Mapping[str, Any]
        | PrototypeSceneCalibrationArtifactAdapter
    ],
    *,
    expected_calibration_plan_digest: str,
) -> PrototypeSceneCalibrationAssessment:
    """Evaluate the complete frozen 28-cluster matrix without fitting a subset."""

    frozen = (
        plan
        if isinstance(plan, PrototypeSceneCalibrationPlan)
        else PrototypeSceneCalibrationPlan.from_data(plan)
    )
    expected = _require_address(
        expected_calibration_plan_digest, "expected calibration plan digest"
    )
    if frozen.record_digest != expected:
        raise PrototypeSceneCalibrationError(
            "calibration plan differs from external commitment"
        )
    adapted = tuple(
        adapt_prototype_scene_observation(
            item, calibration_plan_digest=frozen.record_digest
        )
        for item in observations
    )
    if len(adapted) != len(frozen.scenes):
        raise PrototypeSceneCalibrationError(
            "complete observation set requires all 28 scheduled scenes"
        )
    by_key: dict[tuple[str, str], PrototypeSceneCalibrationObservation] = {}
    for observation in adapted:
        key = (observation.task_id, observation.panel_id)
        if key in by_key:
            raise PrototypeSceneCalibrationError(
                "a calibration scene was observed more than once"
            )
        by_key[key] = observation
    scheduled_keys = {(item.task_id, item.panel_id) for item in frozen.scenes}
    if set(by_key) != scheduled_keys:
        missing = sorted(scheduled_keys - set(by_key))
        extra = sorted(set(by_key) - scheduled_keys)
        raise PrototypeSceneCalibrationError(
            f"observation schedule differs: missing={missing}, extra={extra}"
        )
    ordered = tuple(by_key[(item.task_id, item.panel_id)] for item in frozen.scenes)
    for observation in ordered:
        identity = (
            observation.cohort_plan_digest == frozen.cohort_plan_digest
            and observation.description_catalog_digest
            == frozen.description_catalog_digest
            and observation.prototype_reference_digest
            == frozen.prototype_reference_digest
            and observation.observer_protocol_id == frozen.observer_protocol_id
            and observation.observer_protocol_digest
            == frozen.observer_protocol_digest
            and observation.model_id == frozen.model_id
            and observation.model_identity_digest == frozen.model_identity_digest
            and observation.environment_digest == frozen.environment_digest
        )
        if not identity:
            raise PrototypeSceneCalibrationError(
                "observation cohort/catalog/reference/protocol/model/environment drift"
            )
    thresholds = _threshold_map(frozen)
    bounds: list[PrototypeSceneDirectionBound] = []
    for tag_id in OPAQUE_TAG_IDS:
        for direction, expected_state in (
            (CalibrationDirection.FALSE_PRESENT, "absent"),
            (CalibrationDirection.FALSE_ABSENT, "present"),
        ):
            cluster_ids: list[str] = []
            error_ids: list[str] = []
            abstentions = 0
            correct = 0
            for scene, observation in zip(frozen.scenes, ordered, strict=True):
                if dict(scene.expected_tag_states)[tag_id] != expected_state:
                    continue
                cluster_ids.append(scene.cluster_id)
                score = next(item for item in observation.scores if item.tag_id == tag_id)
                disposition = _score_disposition(score, thresholds[tag_id])
                is_error = disposition is PrototypeSceneDisposition.ERROR or (
                    direction is CalibrationDirection.FALSE_PRESENT
                    and disposition is PrototypeSceneDisposition.CALIBRATED_PRESENT
                ) or (
                    direction is CalibrationDirection.FALSE_ABSENT
                    and disposition is PrototypeSceneDisposition.CALIBRATED_ABSENT
                )
                if is_error:
                    error_ids.append(scene.cluster_id)
                elif disposition is PrototypeSceneDisposition.INDETERMINATE:
                    abstentions += 1
                else:
                    correct += 1
            n = len(cluster_ids)
            upper = familywise_clopper_pearson_upper_ppm(
                cluster_count=n,
                error_cluster_count=len(error_ids),
                confidence_level_ppm=frozen.confidence_level_ppm,
                hypothesis_count=frozen.hypothesis_count,
            )
            coverage_gate_accepted = abstentions == 0
            bounds.append(
                PrototypeSceneDirectionBound(
                    tag_id=tag_id,
                    direction=direction,
                    cluster_count=n,
                    error_cluster_count=len(error_ids),
                    abstention_cluster_count=abstentions,
                    correct_decision_cluster_count=correct,
                    conditional_error_upper_ppm=upper,
                    confidence_level_ppm=frozen.confidence_level_ppm,
                    hypothesis_count=frozen.hypothesis_count,
                    maximum_conditional_error_ppm=(
                        frozen.maximum_conditional_error_ppm
                    ),
                    coverage_gate_accepted=coverage_gate_accepted,
                    accepted=(
                        upper <= frozen.maximum_conditional_error_ppm
                        and coverage_gate_accepted
                    ),
                    cluster_ids=tuple(cluster_ids),
                    error_cluster_ids=tuple(error_ids),
                )
            )
    observation_digests = tuple(item.record_digest for item in ordered)
    return PrototypeSceneCalibrationAssessment(
        calibration_plan_digest=frozen.record_digest,
        observation_set_digest=_address(
            {
                "schema": "gkm.bongard-prototype-scene-observation-set.v1",
                "calibration_plan_digest": frozen.record_digest,
                "observation_record_digests": list(observation_digests),
            }
        ),
        observation_count=len(ordered),
        observation_record_digests=observation_digests,
        bounds=tuple(bounds),
        all_four_coverage_gates_accepted=all(
            item.coverage_gate_accepted for item in bounds
        ),
        all_four_bounds_accepted=all(item.accepted for item in bounds),
        complete_observation_set=True,
        every_scene_scored_once_for_both_tags=True,
    )


@dataclass(frozen=True, slots=True)
class PrototypeSceneCalibrationFamily:
    calibration_plan_digest: str
    cohort_plan_digest: str
    thresholds: tuple[PrototypeSceneTagThreshold, PrototypeSceneTagThreshold]
    description_catalog_digest: str
    prototype_reference_digest: str
    observer_protocol_id: str
    observer_protocol_digest: str
    model_id: str
    model_identity_digest: str
    environment_digest: str
    population_assumption: str
    conditional_target_population_valid: bool
    general_transport_validated: bool
    assessment_digest: str
    observation_set_digest: str
    bounds: tuple[PrototypeSceneDirectionBound, ...]
    coverage_gate_accepted: bool
    hypothesis_count: int
    confidence_level_ppm: int
    maximum_conditional_error_ppm: int
    calibration_source_sha256: str
    calibration_algorithm_digest: str
    predicate_authority_id: str
    python_is_canonical_authority: bool
    lean_required: bool
    lean_defines_identity_or_decision: bool
    lean_required_for_replay: bool

    def __post_init__(self) -> None:
        for name in (
            "calibration_plan_digest",
            "cohort_plan_digest",
            "description_catalog_digest",
            "prototype_reference_digest",
            "observer_protocol_digest",
            "model_identity_digest",
            "environment_digest",
            "assessment_digest",
            "observation_set_digest",
            "calibration_algorithm_digest",
        ):
            _require_address(getattr(self, name), name)
        _require_raw_sha(self.calibration_source_sha256, "calibration source SHA-256")
        if (
            tuple(item.tag_id for item in self.thresholds) != OPAQUE_TAG_IDS
            or len(self.bounds) != HYPOTHESIS_COUNT
            or tuple((item.tag_id, item.direction) for item in self.bounds)
            != tuple(
                (tag_id, direction)
                for tag_id in OPAQUE_TAG_IDS
                for direction in (
                    CalibrationDirection.FALSE_PRESENT,
                    CalibrationDirection.FALSE_ABSENT,
                )
            )
            or not all(item.accepted for item in self.bounds)
            or not all(item.coverage_gate_accepted for item in self.bounds)
            or self.coverage_gate_accepted is not True
            or self.population_assumption
            != SAME_BASIC_RENDERER_CONDITIONAL_ASSUMPTION
            or self.conditional_target_population_valid is not True
            or self.general_transport_validated is not False
            or self.hypothesis_count != HYPOTHESIS_COUNT
            or self.confidence_level_ppm != CONFIDENCE_LEVEL_PPM
            or self.maximum_conditional_error_ppm
            != TARGETED_ENGINEERING_TOLERANCE_PPM
            or self.calibration_source_sha256 != CALIBRATION_SOURCE_SHA256
            or self.calibration_algorithm_digest != calibration_algorithm_digest()
            or self.predicate_authority_id != PYTHON_PREDICATE_AUTHORITY_ID
            or self.python_is_canonical_authority is not True
            or self.lean_required is not False
            or self.lean_defines_identity_or_decision is not False
            or self.lean_required_for_replay is not False
        ):
            raise PrototypeSceneCalibrationError("calibration family authority differs")

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": FAMILY_SCHEMA,
            "calibration_plan_digest": self.calibration_plan_digest,
            "cohort_plan_digest": self.cohort_plan_digest,
            "thresholds": [item.to_data() for item in self.thresholds],
            "observer_identity": {
                "description_catalog_digest": self.description_catalog_digest,
                "prototype_reference_digest": self.prototype_reference_digest,
                "observer_protocol_id": self.observer_protocol_id,
                "observer_protocol_digest": self.observer_protocol_digest,
                "model_id": self.model_id,
                "model_identity_digest": self.model_identity_digest,
                "environment_digest": self.environment_digest,
            },
            "population": {
                "assumption": self.population_assumption,
                "conditional_target_population_valid": (
                    self.conditional_target_population_valid
                ),
                "general_transport_validated": self.general_transport_validated,
            },
            "assessment_digest": self.assessment_digest,
            "observation_set_digest": self.observation_set_digest,
            "bounds": [item.to_data() for item in self.bounds],
            "statistics": {
                "coverage_gate_accepted": self.coverage_gate_accepted,
                "hypothesis_count": self.hypothesis_count,
                "confidence_level_ppm": self.confidence_level_ppm,
                "maximum_conditional_error_ppm": (
                    self.maximum_conditional_error_ppm
                ),
            },
            "runtime_authority": {
                "calibration_source_sha256": self.calibration_source_sha256,
                "calibration_algorithm_digest": self.calibration_algorithm_digest,
                "predicate_authority_id": self.predicate_authority_id,
                "python_is_canonical_authority": self.python_is_canonical_authority,
                "lean_required": self.lean_required,
                "lean_defines_identity_or_decision": (
                    self.lean_defines_identity_or_decision
                ),
                "lean_required_for_replay": self.lean_required_for_replay,
            },
        }

    @property
    def record_digest(self) -> str:
        return _address(self.content_dict())

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypeSceneCalibrationFamily":
        raw = _object(
            value,
            {
                "schema",
                "calibration_plan_digest",
                "cohort_plan_digest",
                "thresholds",
                "observer_identity",
                "population",
                "assessment_digest",
                "observation_set_digest",
                "bounds",
                "statistics",
                "runtime_authority",
                "record_digest",
            },
            "calibration family",
        )
        _verify_digest(raw, "calibration family")
        if raw["schema"] != FAMILY_SCHEMA:
            raise PrototypeSceneCalibrationError("family schema differs")
        observer = _object(
            raw["observer_identity"],
            {
                "description_catalog_digest",
                "prototype_reference_digest",
                "observer_protocol_id",
                "observer_protocol_digest",
                "model_id",
                "model_identity_digest",
                "environment_digest",
            },
            "family observer identity",
        )
        population = _object(
            raw["population"],
            {
                "assumption",
                "conditional_target_population_valid",
                "general_transport_validated",
            },
            "family population",
        )
        statistics = _object(
            raw["statistics"],
            {
                "coverage_gate_accepted",
                "hypothesis_count",
                "confidence_level_ppm",
                "maximum_conditional_error_ppm",
            },
            "family statistics",
        )
        runtime = _object(
            raw["runtime_authority"],
            {
                "calibration_source_sha256",
                "calibration_algorithm_digest",
                "predicate_authority_id",
                "python_is_canonical_authority",
                "lean_required",
                "lean_defines_identity_or_decision",
                "lean_required_for_replay",
            },
            "family runtime authority",
        )
        result = cls(
            calibration_plan_digest=raw["calibration_plan_digest"],
            cohort_plan_digest=raw["cohort_plan_digest"],
            thresholds=tuple(
                PrototypeSceneTagThreshold.from_data(item)
                for item in _list(raw["thresholds"], "family thresholds")
            ),  # type: ignore[arg-type]
            description_catalog_digest=observer["description_catalog_digest"],
            prototype_reference_digest=observer["prototype_reference_digest"],
            observer_protocol_id=observer["observer_protocol_id"],
            observer_protocol_digest=observer["observer_protocol_digest"],
            model_id=observer["model_id"],
            model_identity_digest=observer["model_identity_digest"],
            environment_digest=observer["environment_digest"],
            population_assumption=population["assumption"],
            conditional_target_population_valid=population[
                "conditional_target_population_valid"
            ],
            general_transport_validated=population["general_transport_validated"],
            assessment_digest=raw["assessment_digest"],
            observation_set_digest=raw["observation_set_digest"],
            bounds=tuple(
                PrototypeSceneDirectionBound.from_data(item)
                for item in _list(raw["bounds"], "family bounds")
            ),
            coverage_gate_accepted=statistics["coverage_gate_accepted"],
            hypothesis_count=statistics["hypothesis_count"],
            confidence_level_ppm=statistics["confidence_level_ppm"],
            maximum_conditional_error_ppm=statistics[
                "maximum_conditional_error_ppm"
            ],
            calibration_source_sha256=runtime["calibration_source_sha256"],
            calibration_algorithm_digest=runtime[
                "calibration_algorithm_digest"
            ],
            predicate_authority_id=runtime["predicate_authority_id"],
            python_is_canonical_authority=runtime[
                "python_is_canonical_authority"
            ],
            lean_required=runtime["lean_required"],
            lean_defines_identity_or_decision=runtime[
                "lean_defines_identity_or_decision"
            ],
            lean_required_for_replay=runtime["lean_required_for_replay"],
        )
        if result.to_data() != dict(raw):
            raise PrototypeSceneCalibrationError("family is not canonical")
        return result


def fit_prototype_scene_calibration_family(
    plan: PrototypeSceneCalibrationPlan | Mapping[str, Any],
    observations: Sequence[
        PrototypeSceneCalibrationObservation
        | Mapping[str, Any]
        | PrototypeSceneCalibrationArtifactAdapter
    ],
    *,
    expected_calibration_plan_digest: str,
) -> PrototypeSceneCalibrationFamily:
    frozen = (
        plan
        if isinstance(plan, PrototypeSceneCalibrationPlan)
        else PrototypeSceneCalibrationPlan.from_data(plan)
    )
    assessment = assess_prototype_scene_calibration(
        frozen,
        observations,
        expected_calibration_plan_digest=expected_calibration_plan_digest,
    )
    if not assessment.all_four_bounds_accepted:
        failed = [
            (
                item.tag_id,
                item.direction.value,
                item.error_cluster_count,
                item.abstention_cluster_count,
                item.conditional_error_upper_ppm,
                item.coverage_gate_accepted,
            )
            for item in assessment.bounds
            if not item.accepted
        ]
        raise PrototypeSceneCalibrationError(
            "calibration exceeds 300000 ppm targeted tolerance or fails "
            f"zero-abstention coverage: {failed}"
        )
    return PrototypeSceneCalibrationFamily(
        calibration_plan_digest=frozen.record_digest,
        cohort_plan_digest=frozen.cohort_plan_digest,
        thresholds=frozen.thresholds,
        description_catalog_digest=frozen.description_catalog_digest,
        prototype_reference_digest=frozen.prototype_reference_digest,
        observer_protocol_id=frozen.observer_protocol_id,
        observer_protocol_digest=frozen.observer_protocol_digest,
        model_id=frozen.model_id,
        model_identity_digest=frozen.model_identity_digest,
        environment_digest=frozen.environment_digest,
        population_assumption=frozen.population_assumption,
        conditional_target_population_valid=(
            frozen.same_basic_renderer_population_valid
            and frozen.conditional_transport_assumption_accepted
            and frozen.observer_environment_valid
        ),
        general_transport_validated=frozen.general_transport_validated,
        assessment_digest=assessment.record_digest,
        observation_set_digest=assessment.observation_set_digest,
        bounds=assessment.bounds,
        coverage_gate_accepted=assessment.all_four_coverage_gates_accepted,
        hypothesis_count=frozen.hypothesis_count,
        confidence_level_ppm=frozen.confidence_level_ppm,
        maximum_conditional_error_ppm=frozen.maximum_conditional_error_ppm,
        calibration_source_sha256=CALIBRATION_SOURCE_SHA256,
        calibration_algorithm_digest=calibration_algorithm_digest(),
        predicate_authority_id=PYTHON_PREDICATE_AUTHORITY_ID,
        python_is_canonical_authority=True,
        lean_required=False,
        lean_defines_identity_or_decision=False,
        lean_required_for_replay=False,
    )


@dataclass(frozen=True, slots=True)
class PrototypeSceneEvaluationContext:
    cohort_plan_digest: str
    description_catalog_digest: str
    prototype_reference_digest: str
    observer_protocol_id: str
    observer_protocol_digest: str
    model_id: str
    model_identity_digest: str
    environment_digest: str
    same_basic_renderer_population_valid: bool
    conditional_transport_assumption_accepted: bool
    observer_environment_valid: bool

    def __post_init__(self) -> None:
        for name in (
            "cohort_plan_digest",
            "description_catalog_digest",
            "prototype_reference_digest",
            "observer_protocol_digest",
            "model_identity_digest",
            "environment_digest",
        ):
            _require_address(getattr(self, name), name)
        _identifier(self.observer_protocol_id, "evaluation observer protocol ID")
        _text(self.model_id, "evaluation model ID")
        for name in (
            "same_basic_renderer_population_valid",
            "conditional_transport_assumption_accepted",
            "observer_environment_valid",
        ):
            if not isinstance(getattr(self, name), bool):
                raise PrototypeSceneCalibrationError(
                    "evaluation validity flags must be Boolean"
                )

    def to_data(self) -> dict[str, object]:
        return {
            "cohort_plan_digest": self.cohort_plan_digest,
            "description_catalog_digest": self.description_catalog_digest,
            "prototype_reference_digest": self.prototype_reference_digest,
            "observer_protocol_id": self.observer_protocol_id,
            "observer_protocol_digest": self.observer_protocol_digest,
            "model_id": self.model_id,
            "model_identity_digest": self.model_identity_digest,
            "environment_digest": self.environment_digest,
            "same_basic_renderer_population_valid": (
                self.same_basic_renderer_population_valid
            ),
            "conditional_transport_assumption_accepted": (
                self.conditional_transport_assumption_accepted
            ),
            "observer_environment_valid": self.observer_environment_valid,
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypeSceneEvaluationContext":
        fields = {
            "cohort_plan_digest",
            "description_catalog_digest",
            "prototype_reference_digest",
            "observer_protocol_id",
            "observer_protocol_digest",
            "model_id",
            "model_identity_digest",
            "environment_digest",
            "same_basic_renderer_population_valid",
            "conditional_transport_assumption_accepted",
            "observer_environment_valid",
        }
        raw = _object(value, fields, "evaluation context")
        result = cls(**raw)  # type: ignore[arg-type]
        if result.to_data() != dict(raw):
            raise PrototypeSceneCalibrationError("evaluation context is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class PrototypeSceneCalibratedResult:
    family_digest: str
    tag_id: str
    score_digest: str
    context_digest: str
    disposition: PrototypeSceneDisposition
    reason_code: str
    identity_valid: bool
    conditional_target_population_valid: bool
    predicate_authority_id: str
    python_is_canonical_authority: bool
    lean_required: bool

    def __post_init__(self) -> None:
        for name in ("family_digest", "score_digest", "context_digest"):
            _require_address(getattr(self, name), name)
        if self.tag_id not in OPAQUE_TAG_IDS or not isinstance(
            self.disposition, PrototypeSceneDisposition
        ):
            raise PrototypeSceneCalibrationError("calibrated result identity differs")
        _identifier(self.reason_code, "calibrated result reason")
        if (
            not isinstance(self.identity_valid, bool)
            or not isinstance(self.conditional_target_population_valid, bool)
            or self.predicate_authority_id != PYTHON_PREDICATE_AUTHORITY_ID
            or self.python_is_canonical_authority is not True
            or self.lean_required is not False
            or (
                self.disposition is not PrototypeSceneDisposition.ERROR
                and (
                    not self.identity_valid
                    or not self.conditional_target_population_valid
                )
            )
        ):
            raise PrototypeSceneCalibrationError("calibrated result authority differs")

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": RESULT_SCHEMA,
            "family_digest": self.family_digest,
            "tag_id": self.tag_id,
            "score_digest": self.score_digest,
            "context_digest": self.context_digest,
            "disposition": self.disposition.value,
            "reason_code": self.reason_code,
            "identity_valid": self.identity_valid,
            "conditional_target_population_valid": (
                self.conditional_target_population_valid
            ),
            "predicate_authority_id": self.predicate_authority_id,
            "python_is_canonical_authority": self.python_is_canonical_authority,
            "lean_required": self.lean_required,
        }

    @property
    def record_digest(self) -> str:
        return _address(self.content_dict())

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypeSceneCalibratedResult":
        raw = _object(
            value,
            {
                "schema",
                "family_digest",
                "tag_id",
                "score_digest",
                "context_digest",
                "disposition",
                "reason_code",
                "identity_valid",
                "conditional_target_population_valid",
                "predicate_authority_id",
                "python_is_canonical_authority",
                "lean_required",
                "record_digest",
            },
            "calibrated result",
        )
        _verify_digest(raw, "calibrated result")
        if raw["schema"] != RESULT_SCHEMA:
            raise PrototypeSceneCalibrationError("result schema differs")
        try:
            disposition = PrototypeSceneDisposition(raw["disposition"])
        except (TypeError, ValueError) as exc:
            raise PrototypeSceneCalibrationError("result disposition differs") from exc
        result = cls(
            family_digest=raw["family_digest"],
            tag_id=raw["tag_id"],
            score_digest=raw["score_digest"],
            context_digest=raw["context_digest"],
            disposition=disposition,
            reason_code=raw["reason_code"],
            identity_valid=raw["identity_valid"],
            conditional_target_population_valid=raw[
                "conditional_target_population_valid"
            ],
            predicate_authority_id=raw["predicate_authority_id"],
            python_is_canonical_authority=raw[
                "python_is_canonical_authority"
            ],
            lean_required=raw["lean_required"],
        )
        if result.to_data() != dict(raw):
            raise PrototypeSceneCalibrationError("result is not canonical")
        return result


def evaluate_prototype_scene_score(
    family: PrototypeSceneCalibrationFamily | Mapping[str, Any],
    score: PrototypeSceneTagScore | Mapping[str, Any],
    context: PrototypeSceneEvaluationContext | Mapping[str, Any],
) -> PrototypeSceneCalibratedResult:
    """Map a score interval through the accepted dynamic two-tag family."""

    calibrated = (
        family
        if isinstance(family, PrototypeSceneCalibrationFamily)
        else PrototypeSceneCalibrationFamily.from_data(family)
    )
    tag_score = (
        score
        if isinstance(score, PrototypeSceneTagScore)
        else PrototypeSceneTagScore.from_data(score)
    )
    runtime = (
        context
        if isinstance(context, PrototypeSceneEvaluationContext)
        else PrototypeSceneEvaluationContext.from_data(context)
    )
    identity_valid = (
        runtime.cohort_plan_digest == calibrated.cohort_plan_digest
        and runtime.description_catalog_digest
        == calibrated.description_catalog_digest
        and runtime.prototype_reference_digest
        == calibrated.prototype_reference_digest
        and runtime.observer_protocol_id == calibrated.observer_protocol_id
        and runtime.observer_protocol_digest
        == calibrated.observer_protocol_digest
        and runtime.model_id == calibrated.model_id
        and runtime.model_identity_digest == calibrated.model_identity_digest
        and runtime.environment_digest == calibrated.environment_digest
    )
    target_valid = (
        calibrated.conditional_target_population_valid
        and runtime.same_basic_renderer_population_valid
        and runtime.conditional_transport_assumption_accepted
        and runtime.observer_environment_valid
    )
    if not identity_valid:
        disposition = PrototypeSceneDisposition.ERROR
        reason = "identity_drift"
    elif not target_valid:
        disposition = PrototypeSceneDisposition.ERROR
        reason = "population_transport_or_environment_invalid"
    else:
        threshold = next(
            item for item in calibrated.thresholds if item.tag_id == tag_score.tag_id
        )
        disposition = _score_disposition(tag_score, threshold)
        reason = {
            PrototypeSceneDisposition.CALIBRATED_PRESENT: "at_or_above_present_threshold",
            PrototypeSceneDisposition.CALIBRATED_ABSENT: "at_or_below_absent_threshold",
            PrototypeSceneDisposition.INDETERMINATE: "between_or_overlapping_thresholds",
            PrototypeSceneDisposition.ERROR: "observer_score_state_failure",
        }[disposition]
    context_digest = _address(
        {
            "schema": "gkm.bongard-prototype-scene-evaluation-context.v1",
            **runtime.to_data(),
        }
    )
    return PrototypeSceneCalibratedResult(
        family_digest=calibrated.record_digest,
        tag_id=tag_score.tag_id,
        score_digest=tag_score.record_digest,
        context_digest=context_digest,
        disposition=disposition,
        reason_code=reason,
        identity_valid=identity_valid,
        conditional_target_population_valid=target_valid,
        predicate_authority_id=PYTHON_PREDICATE_AUTHORITY_ID,
        python_is_canonical_authority=True,
        lean_required=False,
    )


def verify_prototype_scene_calibration_plan(
    plan: PrototypeSceneCalibrationPlan | Mapping[str, Any],
    *,
    cohort_plan: PrototypePairCohortPlan | Mapping[str, Any],
    expected_calibration_plan_digest: str,
    expected_cohort_plan_digest: str,
) -> PrototypeSceneCalibrationPlan:
    """Cold-recompute the freeze from its cohort and committed identities."""

    frozen = (
        plan
        if isinstance(plan, PrototypeSceneCalibrationPlan)
        else PrototypeSceneCalibrationPlan.from_data(plan)
    )
    if frozen.record_digest != _require_address(
        expected_calibration_plan_digest, "expected calibration plan digest"
    ):
        raise PrototypeSceneCalibrationError("calibration plan commitment differs")
    cohort = (
        cohort_plan
        if isinstance(cohort_plan, PrototypePairCohortPlan)
        else PrototypePairCohortPlan.from_data(cohort_plan)
    )
    replay = create_prototype_scene_calibration_plan(
        cohort_plan=cohort,
        thresholds=frozen.thresholds,
        description_catalog_digest=frozen.description_catalog_digest,
        prototype_reference_digest=frozen.prototype_reference_digest,
        observer_protocol_id=frozen.observer_protocol_id,
        observer_protocol_digest=frozen.observer_protocol_digest,
        model_id=frozen.model_id,
        model_identity_digest=frozen.model_identity_digest,
        environment_digest=frozen.environment_digest,
        expected_cohort_plan_digest=expected_cohort_plan_digest,
        expected_threshold_commitment=frozen.threshold_commitment,
        expected_description_catalog_digest=frozen.description_catalog_digest,
        expected_prototype_reference_digest=frozen.prototype_reference_digest,
        expected_observer_protocol_digest=frozen.observer_protocol_digest,
        expected_model_identity_digest=frozen.model_identity_digest,
        expected_environment_digest=frozen.environment_digest,
    )
    if replay != frozen or replay.record_digest != frozen.record_digest:
        raise PrototypeSceneCalibrationError(
            "cold-recomputed calibration plan differs"
        )
    return frozen


def verify_prototype_scene_calibration_family(
    family: PrototypeSceneCalibrationFamily | Mapping[str, Any],
    *,
    calibration_plan: PrototypeSceneCalibrationPlan | Mapping[str, Any],
    cohort_plan: PrototypePairCohortPlan | Mapping[str, Any],
    observations: Sequence[
        PrototypeSceneCalibrationObservation | Mapping[str, Any]
    ],
    expected_family_digest: str,
    expected_calibration_plan_digest: str,
    expected_cohort_plan_digest: str,
) -> PrototypeSceneCalibrationFamily:
    """Model-free cold plan verification, refit, and family comparison."""

    archived = (
        family
        if isinstance(family, PrototypeSceneCalibrationFamily)
        else PrototypeSceneCalibrationFamily.from_data(family)
    )
    if archived.record_digest != _require_address(
        expected_family_digest, "expected family digest"
    ):
        raise PrototypeSceneCalibrationError("family commitment differs")
    frozen = verify_prototype_scene_calibration_plan(
        calibration_plan,
        cohort_plan=cohort_plan,
        expected_calibration_plan_digest=expected_calibration_plan_digest,
        expected_cohort_plan_digest=expected_cohort_plan_digest,
    )
    replay = fit_prototype_scene_calibration_family(
        frozen,
        observations,
        expected_calibration_plan_digest=expected_calibration_plan_digest,
    )
    if replay != archived or replay.record_digest != archived.record_digest:
        raise PrototypeSceneCalibrationError("cold-refit family differs")
    return archived


__all__ = [
    "CALIBRATION_ALGORITHM_ID",
    "CALIBRATION_SOURCE_SHA256",
    "CalibrationDirection",
    "MAX_PHYSICAL_OBSERVER_CALLS_PER_SCENE",
    "OBSERVER_ADAPTER_PROTOCOL_ID",
    "PrototypeSceneCalibratedResult",
    "PrototypeSceneCalibrationArtifactAdapter",
    "PrototypeSceneCalibrationAssessment",
    "PrototypeSceneCalibrationError",
    "PrototypeSceneCalibrationFamily",
    "PrototypeSceneCalibrationObservation",
    "PrototypeSceneCalibrationPlan",
    "PrototypeSceneCalibrationScene",
    "PrototypeSceneDirectionBound",
    "PrototypeSceneDisposition",
    "PrototypeSceneEvaluationContext",
    "PrototypeSceneScoreStatus",
    "PrototypeSceneTagScore",
    "PrototypeSceneTagThreshold",
    "SAME_BASIC_RENDERER_CONDITIONAL_ASSUMPTION",
    "adapt_prototype_scene_observation",
    "assess_prototype_scene_calibration",
    "calibration_algorithm_digest",
    "create_prototype_scene_calibration_plan",
    "evaluate_prototype_scene_score",
    "fit_prototype_scene_calibration_family",
    "threshold_commitment",
    "verify_prototype_scene_calibration_family",
    "verify_prototype_scene_calibration_plan",
]

"""Offline byte archive and adapter for calibrated prototype-scene panels.

This module is the only bridge from a frozen whole-scene observer artifact to
the finite prototype-scene predicate evaluator.  It performs no model call.
Every materialization and every runner verifier invocation reparses the exact
archived bytes, replays :func:`verify_prototype_scene_observer_artifact`, and
then reconstructs the two typed scores and their precommitted evaluation
context.

The archive stores bytes and immutable tuples rather than caller-owned
mappings.  Its runner hook therefore authenticates a panel against an
independent raw-artifact archive; a panel's own observer binding is never
accepted as evidence for itself.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
import threading
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.prototype_scene_calibration import (
    PrototypeSceneCalibrationObservation,
    PrototypeSceneCalibrationFamily,
    PrototypeSceneCalibrationPlan,
    PrototypeSceneEvaluationContext,
    PrototypeSceneScoreStatus,
    PrototypeSceneTagScore,
    adapt_prototype_scene_observation,
)
from bongard.prototype_object_scene_observer import (
    PROTOTYPE_SCENE_OBSERVER_ARTIFACT_SCHEMA,
    PROTOTYPE_SCENE_OBSERVER_PROTOCOL_ID,
    PrototypeReferenceCatalog,
    PrototypeRubricDescriptionArtifact,
    PrototypeSceneObserverArtifact,
    PrototypeSceneObserverStatus,
    PrototypeSceneScore,
    PrototypeSceneScoreState,
    verify_prototype_scene_observer_artifact,
)
from bongard.prototype_scene_predicates import (
    PrototypeScenePanelEvaluation,
    PrototypeSceneVerifiedObserverBinding,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


RUNTIME_ADAPTER_ID = "bongard.prototype-scene-runtime/offline-python-v1"
RUNTIME_ARCHIVE_SCHEMA = "gkm.bongard-prototype-scene-runtime-archive.v1"
RUNTIME_ARCHIVE_ENTRY_SCHEMA = (
    "gkm.bongard-prototype-scene-runtime-archive-entry.v1"
)
RUNTIME_REFERENCE_SCHEMA = (
    "gkm.bongard-prototype-scene-runtime-reference-bytes.v1"
)
RUNTIME_VERIFIER_SCHEMA = "gkm.bongard-prototype-scene-runtime-verifier.v1"
EVALUATION_CONTEXT_SCHEMA = (
    "gkm.bongard-prototype-scene-evaluation-context.v1"
)

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}\Z")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_PNG_BYTES = 4_000_000
_MAX_ARCHIVE_SCENES = 64
_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
_ADAPTER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class PrototypeSceneRuntimeAdapterError(ValueError):
    """The offline archive, its trust anchor, or an adaptation is invalid."""


class PrototypeSceneArtifactPurpose(str, Enum):
    """Causal role precommitted for one archived observer turn."""

    RUNTIME_EVALUATION = "runtime_evaluation"
    CALIBRATION = "calibration"


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _require_address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise PrototypeSceneRuntimeAdapterError(
            f"{label} must be a sha256: address"
        )
    return value


def _require_raw_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_SHA256.fullmatch(value) is None:
        raise PrototypeSceneRuntimeAdapterError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise PrototypeSceneRuntimeAdapterError(
            f"{label} must be a bounded identifier"
        )
    return value


def _bounded_bytes(value: object, label: str, maximum: int) -> bytes:
    if (
        not isinstance(value, bytes)
        or not value
        or len(value) > maximum
    ):
        raise PrototypeSceneRuntimeAdapterError(
            f"{label} must be 1..{maximum} exact bytes"
        )
    return value


def _committed_bytes(
    value: object, expected_sha256: object, label: str, maximum: int
) -> bytes:
    data = _bounded_bytes(value, label, maximum)
    expected = _require_raw_sha256(expected_sha256, f"expected {label} SHA-256")
    if hashlib.sha256(data).hexdigest() != expected:
        raise PrototypeSceneRuntimeAdapterError(
            f"{label} differs from external byte commitment"
        )
    return data


def _committed_png(
    value: object, expected_sha256: object, label: str
) -> bytes:
    data = _committed_bytes(value, expected_sha256, label, _MAX_PNG_BYTES)
    if not data.startswith(_PNG_SIGNATURE):
        raise PrototypeSceneRuntimeAdapterError(f"{label} is not a PNG")
    return data


def _json_object(data: bytes, label: str) -> Mapping[str, Any]:
    """Decode one bounded JSON object while rejecting duplicate keys."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise PrototypeSceneRuntimeAdapterError(
                    f"{label} contains a duplicate JSON key"
                )
            result[key] = item
        return result

    def reject_nonfinite(token: str) -> object:
        raise PrototypeSceneRuntimeAdapterError(
            f"{label} contains non-finite JSON number {token}"
        )

    try:
        decoded = data.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=unique_object,
            parse_constant=reject_nonfinite,
        )
    except PrototypeSceneRuntimeAdapterError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise PrototypeSceneRuntimeAdapterError(
            f"{label} is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, Mapping):
        raise PrototypeSceneRuntimeAdapterError(f"{label} must encode an object")
    return value


def prototype_scene_runtime_adapter_source_digest() -> str:
    """Content address of the active pure-Python adapter source."""

    try:
        current = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError as exc:
        raise PrototypeSceneRuntimeAdapterError(
            "adapter source bytes are unavailable"
        ) from exc
    if current != _ADAPTER_SOURCE_SHA256:
        raise PrototypeSceneRuntimeAdapterError(
            "adapter source changed after this Python module was loaded"
        )
    return "sha256:" + _ADAPTER_SOURCE_SHA256


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


def prototype_scene_evaluation_context_digest(
    context: PrototypeSceneEvaluationContext,
) -> str:
    """Return the context address precommitted before scene observation."""

    if not isinstance(context, PrototypeSceneEvaluationContext):
        raise TypeError("context must be PrototypeSceneEvaluationContext")
    if PrototypeSceneEvaluationContext.from_data(context.to_data()) != context:
        raise PrototypeSceneRuntimeAdapterError("evaluation context is not canonical")
    return _address({"schema": EVALUATION_CONTEXT_SCHEMA, **context.to_data()})


def _require_family_context_identity(
    family: PrototypeSceneCalibrationFamily,
    context: PrototypeSceneEvaluationContext,
) -> None:
    if (
        context.cohort_plan_digest != family.cohort_plan_digest
        or context.description_catalog_digest
        != family.description_catalog_digest
        or context.prototype_reference_digest
        != family.prototype_reference_digest
        or context.observer_protocol_id != family.observer_protocol_id
        or context.observer_protocol_digest
        != family.observer_protocol_digest
        or context.model_id != family.model_id
        or context.model_identity_digest != family.model_identity_digest
        or context.environment_digest != family.environment_digest
    ):
        raise PrototypeSceneRuntimeAdapterError(
            "archived evaluation context differs from calibration family"
        )


@dataclass(frozen=True, slots=True)
class PrototypeSceneRuntimeArtifactInput:
    """Caller-supplied raw scene material plus external byte commitments."""

    scene_task_id: str
    panel_id: str
    expected_observation_context_digest: str
    exact_scene_png_bytes: bytes = field(repr=False)
    expected_scene_sha256: str
    observer_artifact_json_bytes: bytes = field(repr=False)
    expected_observer_artifact_json_sha256: str
    expected_observer_artifact_digest: str
    purpose: PrototypeSceneArtifactPurpose = (
        PrototypeSceneArtifactPurpose.RUNTIME_EVALUATION
    )

    def __post_init__(self) -> None:
        if not isinstance(self.purpose, PrototypeSceneArtifactPurpose):
            raise TypeError("scene purpose must be PrototypeSceneArtifactPurpose")
        _identifier(self.scene_task_id, "scene task_id")
        _identifier(self.panel_id, "scene panel_id")
        _require_address(
            self.expected_observation_context_digest,
            "expected observation context digest",
        )
        _committed_png(
            self.exact_scene_png_bytes,
            self.expected_scene_sha256,
            "scene PNG",
        )
        _committed_bytes(
            self.observer_artifact_json_bytes,
            self.expected_observer_artifact_json_sha256,
            "observer artifact JSON",
            _MAX_JSON_BYTES,
        )
        _require_raw_sha256(
            self.expected_observer_artifact_digest,
            "expected observer artifact digest",
        )


@dataclass(frozen=True, slots=True)
class _ArchivedReference:
    panel_id: str
    exact_png_bytes: bytes = field(repr=False)
    expected_png_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.panel_id, "reference panel_id")
        _committed_png(
            self.exact_png_bytes,
            self.expected_png_sha256,
            f"reference PNG {self.panel_id}",
        )

    def commitment_data(self) -> dict[str, object]:
        return {
            "schema": RUNTIME_REFERENCE_SCHEMA,
            "panel_id": self.panel_id,
            "byte_count": len(self.exact_png_bytes),
            "byte_sha256": self.expected_png_sha256,
        }


@dataclass(frozen=True, slots=True)
class _ArchivedScene:
    scene_task_id: str
    panel_id: str
    expected_observation_context_digest: str
    exact_scene_png_bytes: bytes = field(repr=False)
    expected_scene_sha256: str
    observer_artifact_json_bytes: bytes = field(repr=False)
    expected_observer_artifact_json_sha256: str
    expected_observer_artifact_digest: str
    purpose: PrototypeSceneArtifactPurpose

    @classmethod
    def from_input(
        cls, value: PrototypeSceneRuntimeArtifactInput
    ) -> "_ArchivedScene":
        if not isinstance(value, PrototypeSceneRuntimeArtifactInput):
            raise TypeError("scene must be PrototypeSceneRuntimeArtifactInput")
        return cls(
            scene_task_id=value.scene_task_id,
            panel_id=value.panel_id,
            expected_observation_context_digest=(
                value.expected_observation_context_digest
            ),
            exact_scene_png_bytes=value.exact_scene_png_bytes,
            expected_scene_sha256=value.expected_scene_sha256,
            observer_artifact_json_bytes=value.observer_artifact_json_bytes,
            expected_observer_artifact_json_sha256=(
                value.expected_observer_artifact_json_sha256
            ),
            expected_observer_artifact_digest=(
                value.expected_observer_artifact_digest
            ),
            purpose=value.purpose,
        )

    def __post_init__(self) -> None:
        PrototypeSceneRuntimeArtifactInput(
            scene_task_id=self.scene_task_id,
            panel_id=self.panel_id,
            expected_observation_context_digest=(
                self.expected_observation_context_digest
            ),
            exact_scene_png_bytes=self.exact_scene_png_bytes,
            expected_scene_sha256=self.expected_scene_sha256,
            observer_artifact_json_bytes=self.observer_artifact_json_bytes,
            expected_observer_artifact_json_sha256=(
                self.expected_observer_artifact_json_sha256
            ),
            expected_observer_artifact_digest=(
                self.expected_observer_artifact_digest
            ),
            purpose=self.purpose,
        )

    def commitment_data(self) -> dict[str, object]:
        return {
            "schema": RUNTIME_ARCHIVE_ENTRY_SCHEMA,
            "scene_task_id": self.scene_task_id,
            "panel_id": self.panel_id,
            "observation_context_digest": (
                self.expected_observation_context_digest
            ),
            "scene_png_byte_count": len(self.exact_scene_png_bytes),
            "scene_png_sha256": self.expected_scene_sha256,
            "observer_artifact_json_byte_count": len(
                self.observer_artifact_json_bytes
            ),
            "observer_artifact_json_sha256": (
                self.expected_observer_artifact_json_sha256
            ),
            "observer_artifact_digest": self.expected_observer_artifact_digest,
            "purpose": self.purpose.value,
        }


def _map_verified_score(
    artifact_status: PrototypeSceneObserverStatus,
    score: PrototypeSceneScore,
) -> PrototypeSceneTagScore:
    """Exhaustively preserve observer score/failure semantics."""

    if artifact_status is PrototypeSceneObserverStatus.PARSER_ERROR:
        status = PrototypeSceneScoreStatus.PARSER_ERROR
    elif artifact_status is PrototypeSceneObserverStatus.TRANSPORT_ERROR:
        status = PrototypeSceneScoreStatus.TRANSPORT_ERROR
    elif artifact_status is PrototypeSceneObserverStatus.PREREQUISITE_ERROR:
        status = PrototypeSceneScoreStatus.ERROR
    elif artifact_status is PrototypeSceneObserverStatus.INTERNAL_ERROR:
        status = PrototypeSceneScoreStatus.ERROR
    elif artifact_status is PrototypeSceneObserverStatus.SUCCESS:
        if score.state is PrototypeSceneScoreState.SCORED:
            status = PrototypeSceneScoreStatus.SCORE
        elif score.state is PrototypeSceneScoreState.INDETERMINATE:
            status = PrototypeSceneScoreStatus.INDETERMINATE
        elif score.state is PrototypeSceneScoreState.ERROR:
            status = PrototypeSceneScoreStatus.ERROR
        else:  # pragma: no cover - Enum construction prevents this branch.
            raise PrototypeSceneRuntimeAdapterError(
                "verified observer score state is not exhaustive"
            )
    else:  # pragma: no cover - Enum construction prevents this branch.
        raise PrototypeSceneRuntimeAdapterError(
            "verified observer artifact status is not exhaustive"
        )

    if status is PrototypeSceneScoreStatus.SCORE:
        if score.lower_ppm is None or score.upper_ppm is None:
            raise PrototypeSceneRuntimeAdapterError(
                "verified scored cell has no interval"
            )
        return PrototypeSceneTagScore(
            tag_id=score.tag_id,
            status=status,
            lower_ppm=score.lower_ppm,
            upper_ppm=score.upper_ppm,
            reason_code="scored",
            error_type=None,
        )

    # Calibration's typed score record requires an error_type for every
    # non-numerical state.  Observer indeterminacy deliberately has none, so
    # the adapter supplies one typed marker rather than fabricating a low score.
    return PrototypeSceneTagScore(
        tag_id=score.tag_id,
        status=status,
        lower_ppm=None,
        upper_ppm=None,
        reason_code=score.reason_code or "observer_indeterminate",
        error_type=score.error_type or "PrototypeSceneIndeterminate",
    )


@dataclass(frozen=True, slots=True)
class _ColdMaterial:
    artifact: PrototypeSceneObserverArtifact
    context: PrototypeSceneEvaluationContext
    scores: tuple[PrototypeSceneTagScore, PrototypeSceneTagScore]


@dataclass(frozen=True, slots=True, init=False)
class PrototypeSceneRuntimeArtifactArchive:
    """Immutable exact-byte archive used by materialization and cold replay."""

    archive_source_id: str
    verifier_id: str
    adapter_source_digest: str
    catalog_json_bytes: bytes = field(repr=False)
    expected_catalog_json_sha256: str
    expected_catalog_digest: str
    rubric_artifact_json_bytes: bytes = field(repr=False)
    expected_rubric_artifact_json_sha256: str
    expected_rubric_artifact_digest: str
    references: tuple[_ArchivedReference, ...] = field(repr=False)
    scenes: tuple[_ArchivedScene, ...] = field(repr=False)
    same_basic_renderer_population_valid: bool
    conditional_transport_assumption_accepted: bool
    observer_environment_valid: bool
    record_digest: str
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    @classmethod
    def seal_external(
        cls,
        *,
        archive_source_id: str,
        verifier_id: str,
        catalog_json_bytes: bytes,
        expected_catalog_json_sha256: str,
        expected_catalog_digest: str,
        rubric_artifact_json_bytes: bytes,
        expected_rubric_artifact_json_sha256: str,
        expected_rubric_artifact_digest: str,
        prototype_reference_png_by_panel_id: Mapping[str, bytes],
        expected_reference_sha256: Mapping[str, str],
        scenes: Sequence[PrototypeSceneRuntimeArtifactInput],
        same_basic_renderer_population_valid: bool,
        conditional_transport_assumption_accepted: bool,
        observer_environment_valid: bool,
    ) -> "PrototypeSceneRuntimeArtifactArchive":
        """Copy and cold-verify externally committed raw artifacts.

        The byte SHA-256 values and logical artifact digests are caller-owned
        commitments.  They are checked before any record can enter the archive.
        """

        _identifier(archive_source_id, "archive source_id")
        _identifier(verifier_id, "archive verifier_id")
        catalog_bytes = _committed_bytes(
            catalog_json_bytes,
            expected_catalog_json_sha256,
            "catalog JSON",
            _MAX_JSON_BYTES,
        )
        rubric_bytes = _committed_bytes(
            rubric_artifact_json_bytes,
            expected_rubric_artifact_json_sha256,
            "rubric artifact JSON",
            _MAX_JSON_BYTES,
        )
        catalog_digest = _require_raw_sha256(
            expected_catalog_digest, "expected catalog digest"
        )
        rubric_digest = _require_raw_sha256(
            expected_rubric_artifact_digest,
            "expected rubric artifact digest",
        )
        catalog = PrototypeReferenceCatalog.from_data(
            _json_object(catalog_bytes, "catalog JSON"),
            expected_catalog_digest=catalog_digest,
        )
        PrototypeRubricDescriptionArtifact.from_data(
            _json_object(rubric_bytes, "rubric artifact JSON"),
            expected_artifact_digest=rubric_digest,
        )

        if (
            not isinstance(prototype_reference_png_by_panel_id, Mapping)
            or not isinstance(expected_reference_sha256, Mapping)
            or any(
                not isinstance(key, str)
                for key in (*prototype_reference_png_by_panel_id, *expected_reference_sha256)
            )
        ):
            raise PrototypeSceneRuntimeAdapterError(
                "reference byte commitments must be mappings"
            )
        reference_ids = tuple(item.source_panel_id for item in catalog.bindings)
        if (
            set(prototype_reference_png_by_panel_id) != set(reference_ids)
            or set(expected_reference_sha256) != set(reference_ids)
        ):
            raise PrototypeSceneRuntimeAdapterError(
                "reference byte commitment keys differ from catalog"
            )
        references = tuple(
            _ArchivedReference(
                panel_id=panel_id,
                exact_png_bytes=prototype_reference_png_by_panel_id[panel_id],
                expected_png_sha256=expected_reference_sha256[panel_id],
            )
            for panel_id in reference_ids
        )
        archived_scenes = tuple(_ArchivedScene.from_input(item) for item in scenes)
        if (
            not 1 <= len(archived_scenes) <= _MAX_ARCHIVE_SCENES
            or len({item.panel_id for item in archived_scenes})
            != len(archived_scenes)
        ):
            raise PrototypeSceneRuntimeAdapterError(
                "archive requires 1..64 uniquely identified scenes"
            )
        for name, flag in (
            (
                "same Basic renderer population validity",
                same_basic_renderer_population_valid,
            ),
            (
                "conditional transport assumption acceptance",
                conditional_transport_assumption_accepted,
            ),
            ("observer environment validity", observer_environment_valid),
        ):
            if not isinstance(flag, bool):
                raise PrototypeSceneRuntimeAdapterError(f"{name} must be Boolean")

        values: dict[str, object] = {
            "archive_source_id": archive_source_id,
            "verifier_id": verifier_id,
            "adapter_source_digest": (
                prototype_scene_runtime_adapter_source_digest()
            ),
            "catalog_json_bytes": bytes(catalog_bytes),
            "expected_catalog_json_sha256": expected_catalog_json_sha256,
            "expected_catalog_digest": catalog_digest,
            "rubric_artifact_json_bytes": bytes(rubric_bytes),
            "expected_rubric_artifact_json_sha256": (
                expected_rubric_artifact_json_sha256
            ),
            "expected_rubric_artifact_digest": rubric_digest,
            "references": references,
            "scenes": archived_scenes,
            "same_basic_renderer_population_valid": (
                same_basic_renderer_population_valid
            ),
            "conditional_transport_assumption_accepted": (
                conditional_transport_assumption_accepted
            ),
            "observer_environment_valid": observer_environment_valid,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        digest = _address(provisional.commitment_data())
        object.__setattr__(provisional, "record_digest", digest)
        provisional.__post_init__()

        # Seal only after every scene has passed the actual observer verifier.
        for scene in provisional.scenes:
            if scene.purpose is PrototypeSceneArtifactPurpose.RUNTIME_EVALUATION:
                provisional._cold_material(scene)
            else:
                provisional._cold_verified_artifact(scene)
        return provisional

    def __post_init__(self) -> None:
        _identifier(self.archive_source_id, "archive source_id")
        _identifier(self.verifier_id, "archive verifier_id")
        if self.adapter_source_digest != (
            prototype_scene_runtime_adapter_source_digest()
        ):
            raise PrototypeSceneRuntimeAdapterError("adapter source identity differs")
        if len(self.references) != 6 or not 1 <= len(self.scenes) <= (
            _MAX_ARCHIVE_SCENES
        ):
            raise PrototypeSceneRuntimeAdapterError("archive cardinality differs")
        if len({item.panel_id for item in self.scenes}) != len(self.scenes):
            raise PrototypeSceneRuntimeAdapterError("scene panel IDs are not unique")
        for flag in (
            self.same_basic_renderer_population_valid,
            self.conditional_transport_assumption_accepted,
            self.observer_environment_valid,
        ):
            if not isinstance(flag, bool):
                raise PrototypeSceneRuntimeAdapterError(
                    "archive validity flags must be Boolean"
                )
        total_bytes = (
            len(self.catalog_json_bytes)
            + len(self.rubric_artifact_json_bytes)
            + sum(len(item.exact_png_bytes) for item in self.references)
            + sum(
                len(item.exact_scene_png_bytes)
                + len(item.observer_artifact_json_bytes)
                for item in self.scenes
            )
        )
        if total_bytes > _MAX_ARCHIVE_BYTES:
            raise PrototypeSceneRuntimeAdapterError("archive byte budget exceeded")
        if self.record_digest != _address(self.commitment_data()):
            raise PrototypeSceneRuntimeAdapterError("archive digest differs")
        object.__setattr__(self, "_sealed_digest", self.record_digest)

    def commitment_data(self) -> dict[str, object]:
        return {
            "schema": RUNTIME_ARCHIVE_SCHEMA,
            "adapter_id": RUNTIME_ADAPTER_ID,
            "adapter_source_digest": self.adapter_source_digest,
            "archive_source_id": self.archive_source_id,
            "verifier_id": self.verifier_id,
            "catalog": {
                "json_byte_count": len(self.catalog_json_bytes),
                "json_sha256": self.expected_catalog_json_sha256,
                "catalog_digest": self.expected_catalog_digest,
            },
            "rubric_artifact": {
                "json_byte_count": len(self.rubric_artifact_json_bytes),
                "json_sha256": self.expected_rubric_artifact_json_sha256,
                "artifact_digest": self.expected_rubric_artifact_digest,
            },
            "references": [item.commitment_data() for item in self.references],
            "scenes": [item.commitment_data() for item in self.scenes],
            "evaluation_validity": {
                "same_basic_renderer_population_valid": (
                    self.same_basic_renderer_population_valid
                ),
                "conditional_transport_assumption_accepted": (
                    self.conditional_transport_assumption_accepted
                ),
                "observer_environment_valid": self.observer_environment_valid,
            },
            "raw_artifact_bytes_archived": True,
            "live_observer_calls_allowed": False,
            "observer_verification_replayed_per_use": True,
            **_authority_data(),
        }

    @property
    def verifier_digest(self) -> str:
        return _address(
            {
                "schema": RUNTIME_VERIFIER_SCHEMA,
                "adapter_id": RUNTIME_ADAPTER_ID,
                "adapter_source_digest": self.adapter_source_digest,
                "archive_source_id": self.archive_source_id,
                "archive_digest": self.record_digest,
                "verifier_id": self.verifier_id,
                "verification": (
                    "cold-reparse-all-parent-bytes-and-replay-observer-verifier"
                ),
                **_authority_data(),
            }
        )

    def assert_untampered(self, *, expected_archive_digest: str) -> None:
        expected = _require_address(
            expected_archive_digest, "expected runtime archive digest"
        )
        if self.record_digest != expected or self._sealed_digest != expected:
            raise PrototypeSceneRuntimeAdapterError(
                "archive differs from external commitment"
            )
        _committed_bytes(
            self.catalog_json_bytes,
            self.expected_catalog_json_sha256,
            "catalog JSON",
            _MAX_JSON_BYTES,
        )
        _committed_bytes(
            self.rubric_artifact_json_bytes,
            self.expected_rubric_artifact_json_sha256,
            "rubric artifact JSON",
            _MAX_JSON_BYTES,
        )
        for item in self.references:
            item.__post_init__()
        for item in self.scenes:
            item.__post_init__()
        if _address(self.commitment_data()) != expected:
            raise PrototypeSceneRuntimeAdapterError("archive content changed")

    def _scene(self, panel_id: str) -> _ArchivedScene:
        matches = tuple(item for item in self.scenes if item.panel_id == panel_id)
        if len(matches) != 1:
            raise PrototypeSceneRuntimeAdapterError(
                "panel is absent from immutable artifact archive"
            )
        return matches[0]

    def _cold_verified_artifact(
        self, scene: _ArchivedScene
    ) -> PrototypeSceneObserverArtifact:
        catalog = PrototypeReferenceCatalog.from_data(
            _json_object(self.catalog_json_bytes, "catalog JSON"),
            expected_catalog_digest=self.expected_catalog_digest,
        )
        rubric = PrototypeRubricDescriptionArtifact.from_data(
            _json_object(self.rubric_artifact_json_bytes, "rubric artifact JSON"),
            expected_artifact_digest=self.expected_rubric_artifact_digest,
        )
        artifact = PrototypeSceneObserverArtifact.from_data(
            _json_object(
                scene.observer_artifact_json_bytes,
                "observer artifact JSON",
            ),
            expected_artifact_digest=scene.expected_observer_artifact_digest,
        )
        references = {
            item.panel_id: item.exact_png_bytes for item in self.references
        }
        return verify_prototype_scene_observer_artifact(
            artifact,
            scene.exact_scene_png_bytes,
            expected_scene_task_id=scene.scene_task_id,
            expected_scene_panel_id=scene.panel_id,
            expected_observation_context_digest=(
                scene.expected_observation_context_digest
            ),
            expected_scene_sha256=scene.expected_scene_sha256,
            catalog=catalog,
            prototype_png_by_panel_id=references,
            expected_catalog_digest=self.expected_catalog_digest,
            rubric_artifact=rubric,
            expected_rubric_artifact_digest=(
                self.expected_rubric_artifact_digest
            ),
            expected_artifact_digest=scene.expected_observer_artifact_digest,
        )

    def _cold_material(self, scene: _ArchivedScene) -> _ColdMaterial:
        if scene.purpose is not PrototypeSceneArtifactPurpose.RUNTIME_EVALUATION:
            raise PrototypeSceneRuntimeAdapterError(
                "calibration artifact cannot be used as a runtime panel"
            )
        verified = self._cold_verified_artifact(scene)
        context = PrototypeSceneEvaluationContext(
            cohort_plan_digest=verified.plan_digest,
            description_catalog_digest=(
                "sha256:" + verified.rubric_description_digest
            ),
            prototype_reference_digest="sha256:" + verified.catalog_digest,
            observer_protocol_id=PROTOTYPE_SCENE_OBSERVER_PROTOCOL_ID,
            observer_protocol_digest="sha256:" + verified.protocol_digest,
            model_id=verified.model,
            model_identity_digest="sha256:" + verified.model_digest,
            environment_digest="sha256:" + verified.environment_digest,
            same_basic_renderer_population_valid=(
                self.same_basic_renderer_population_valid
            ),
            conditional_transport_assumption_accepted=(
                self.conditional_transport_assumption_accepted
            ),
            observer_environment_valid=self.observer_environment_valid,
        )
        if prototype_scene_evaluation_context_digest(context) != (
            scene.expected_observation_context_digest
        ):
            raise PrototypeSceneRuntimeAdapterError(
                "scene did not precommit its reconstructed evaluation context"
            )
        scores = tuple(
            _map_verified_score(verified.status, item) for item in verified.scores
        )
        if len(scores) != 2:
            raise PrototypeSceneRuntimeAdapterError(
                "verified observer artifact did not exhaust both scores"
            )
        return _ColdMaterial(
            artifact=verified,
            context=context,
            scores=scores,  # type: ignore[arg-type]
        )

    def artifact_verifier(
        self, *, expected_archive_digest: str
    ) -> "PrototypeSceneRuntimeArtifactVerifier":
        return PrototypeSceneRuntimeArtifactVerifier(
            archive=self,
            expected_archive_digest=expected_archive_digest,
        )


@dataclass(frozen=True, slots=True)
class PrototypeSceneRuntimeArtifactVerifier:
    """Runner-compatible verifier rooted in one externally pinned archive."""

    archive: PrototypeSceneRuntimeArtifactArchive = field(repr=False)
    expected_archive_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.archive, PrototypeSceneRuntimeArtifactArchive):
            raise TypeError("archive must be PrototypeSceneRuntimeArtifactArchive")
        self.archive.assert_untampered(
            expected_archive_digest=self.expected_archive_digest
        )

    @property
    def verifier_id(self) -> str:
        return self.archive.verifier_id

    @property
    def verifier_digest(self) -> str:
        return self.archive.verifier_digest

    def __call__(
        self,
        binding: PrototypeSceneVerifiedObserverBinding,
        exact_png_bytes: bytes,
    ) -> None:
        if not isinstance(binding, PrototypeSceneVerifiedObserverBinding):
            raise TypeError("binding must be PrototypeSceneVerifiedObserverBinding")
        self.archive.assert_untampered(
            expected_archive_digest=self.expected_archive_digest
        )
        scene = self.archive._scene(binding.panel_id)
        if exact_png_bytes != scene.exact_scene_png_bytes:
            raise PrototypeSceneRuntimeAdapterError(
                "runner PNG differs from immutable scene archive"
            )
        material = self.archive._cold_material(scene)
        expected = PrototypeSceneVerifiedObserverBinding.seal_verified(
            panel_id=scene.panel_id,
            exact_png_bytes=scene.exact_scene_png_bytes,
            observer_artifact_schema=PROTOTYPE_SCENE_OBSERVER_ARTIFACT_SCHEMA,
            observer_artifact_digest=(
                "sha256:" + material.artifact.artifact_digest
            ),
            verifier_id=self.verifier_id,
            verifier_digest=self.verifier_digest,
            scores=material.scores,
            context=material.context,
        )
        if binding != expected:
            raise PrototypeSceneRuntimeAdapterError(
                "observer binding differs from archive reconstruction"
            )


@dataclass(slots=True, init=False)
class PrototypeScenePhasedArtifactVerifier:
    """One runner hook with a causal support-to-query archive transition.

    The object initially owns only a pinned, immutable twelve-panel support
    archive.  Query material cannot be attached until the caller supplies the
    typed durable candidate-freeze receipt.  Attachment is one-shot.  Cold
    replay uses :meth:`from_pinned_archives_for_cold_replay` after both
    archives and the freeze receipt already exist.
    """

    _support: PrototypeSceneRuntimeArtifactVerifier
    _support_digest: str
    _family_digest: str
    _query: PrototypeSceneRuntimeArtifactVerifier | None
    _freeze_commit_digest: str | None
    _lock: threading.Lock

    @staticmethod
    def _require_runtime_archive_cardinality(
        archive: PrototypeSceneRuntimeArtifactArchive,
        expected: int,
        label: str,
    ) -> None:
        if (
            len(archive.scenes) != expected
            or any(
                item.purpose
                is not PrototypeSceneArtifactPurpose.RUNTIME_EVALUATION
                for item in archive.scenes
            )
        ):
            raise PrototypeSceneRuntimeAdapterError(
                f"{label} archive must contain exactly {expected} runtime scenes"
            )

    @classmethod
    def for_support(
        cls,
        support_archive: PrototypeSceneRuntimeArtifactArchive,
        *,
        expected_support_archive_digest: str,
        family: PrototypeSceneCalibrationFamily,
        support_panels: Sequence[PrototypeScenePanelEvaluation],
    ) -> "PrototypeScenePhasedArtifactVerifier":
        if not isinstance(support_archive, PrototypeSceneRuntimeArtifactArchive):
            raise TypeError(
                "support_archive must be PrototypeSceneRuntimeArtifactArchive"
            )
        cls._require_runtime_archive_cardinality(
            support_archive, 12, "support"
        )
        support = support_archive.artifact_verifier(
            expected_archive_digest=expected_support_archive_digest
        )
        if not isinstance(family, PrototypeSceneCalibrationFamily) or (
            PrototypeSceneCalibrationFamily.from_data(family.to_data()) != family
        ):
            raise PrototypeSceneRuntimeAdapterError(
                "phased verifier calibration family is not canonical"
            )
        panels = tuple(support_panels)
        if (
            len(panels) != 12
            or any(
                not isinstance(item, PrototypeScenePanelEvaluation)
                for item in panels
            )
            or tuple(item.panel_id for item in panels)
            != tuple(item.panel_id for item in support_archive.scenes)
        ):
            raise PrototypeSceneRuntimeAdapterError(
                "support panels differ from pinned archive order"
            )
        for panel in panels:
            support(panel.observer_binding, panel.exact_png_bytes)
            panel.assert_matches(family)
        support_digest = _address(
            {
                "schema": "gkm.bongard-prototype-scene-headless-support.v1",
                "panels": [item.to_data() for item in panels],
                "sides": ["positive"] * 6 + ["negative"] * 6,
                **_authority_data(),
            }
        )
        result = object.__new__(cls)
        result._support = support
        result._support_digest = support_digest
        result._family_digest = family.record_digest
        result._query = None
        result._freeze_commit_digest = None
        result._lock = threading.Lock()
        return result

    @staticmethod
    def _freeze_receipt(
        freeze: object,
        freeze_commit: object,
        expected_freeze_commit_digest: str,
    ) -> tuple[object, str]:
        # Local import avoids making the byte archive a dependency of the
        # headless runner while still validating the runner's exact types.
        from bongard.prototype_scene_headless_runner import (
            PrototypeSceneCandidateFreeze,
            PrototypeSceneFreezeCommitReceipt,
        )

        frozen = (
            freeze
            if isinstance(freeze, PrototypeSceneCandidateFreeze)
            else PrototypeSceneCandidateFreeze.from_data(freeze)  # type: ignore[arg-type]
        )
        receipt = (
            freeze_commit
            if isinstance(freeze_commit, PrototypeSceneFreezeCommitReceipt)
            else PrototypeSceneFreezeCommitReceipt.from_data(  # type: ignore[arg-type]
                freeze_commit
            )
        )
        expected = _require_address(
            expected_freeze_commit_digest,
            "expected freeze commit digest",
        )
        if receipt.record_digest != expected:
            raise PrototypeSceneRuntimeAdapterError(
                "freeze receipt differs from external commitment"
            )
        receipt.assert_matches(
            frozen, canonical_json(frozen.to_data()) + b"\n"
        )
        return frozen, receipt.record_digest

    def attach_query_archive_after_freeze(
        self,
        query_archive: PrototypeSceneRuntimeArtifactArchive,
        *,
        expected_query_archive_digest: str,
        freeze: object,
        freeze_commit: object,
        expected_freeze_commit_digest: str,
    ) -> None:
        """Attach exactly one pinned two-panel query archive after commit."""

        if not isinstance(query_archive, PrototypeSceneRuntimeArtifactArchive):
            raise TypeError(
                "query_archive must be PrototypeSceneRuntimeArtifactArchive"
            )
        self._require_runtime_archive_cardinality(query_archive, 2, "query")
        frozen, receipt_digest = self._freeze_receipt(
            freeze, freeze_commit, expected_freeze_commit_digest
        )
        if (
            frozen.support_digest != self._support_digest  # type: ignore[attr-defined]
            or frozen.calibration_family_digest  # type: ignore[attr-defined]
            != self._family_digest
        ):
            raise PrototypeSceneRuntimeAdapterError(
                "candidate freeze belongs to another support archive or family"
            )
        query = query_archive.artifact_verifier(
            expected_archive_digest=expected_query_archive_digest
        )
        support_ids = {item.panel_id for item in self._support.archive.scenes}
        query_ids = {item.panel_id for item in query.archive.scenes}
        if support_ids & query_ids:
            raise PrototypeSceneRuntimeAdapterError(
                "query archive overlaps immutable support archive"
            )
        if (
            query.archive.record_digest == self._support.archive.record_digest
            or (
                query.verifier_id == self._support.verifier_id
                and query.verifier_digest == self._support.verifier_digest
            )
        ):
            raise PrototypeSceneRuntimeAdapterError(
                "query archive does not define a distinct verifier domain"
            )
        with self._lock:
            if self._query is not None or self._freeze_commit_digest is not None:
                raise PrototypeSceneRuntimeAdapterError(
                    "query archive attachment is one-shot"
                )
            self._query = query
            self._freeze_commit_digest = receipt_digest

    @classmethod
    def from_pinned_archives_for_cold_replay(
        cls,
        support_archive: PrototypeSceneRuntimeArtifactArchive,
        query_archive: PrototypeSceneRuntimeArtifactArchive,
        *,
        expected_support_archive_digest: str,
        expected_query_archive_digest: str,
        family: PrototypeSceneCalibrationFamily,
        support_panels: Sequence[PrototypeScenePanelEvaluation],
        freeze: object,
        freeze_commit: object,
        expected_freeze_commit_digest: str,
    ) -> "PrototypeScenePhasedArtifactVerifier":
        result = cls.for_support(
            support_archive,
            expected_support_archive_digest=expected_support_archive_digest,
            family=family,
            support_panels=support_panels,
        )
        result.attach_query_archive_after_freeze(
            query_archive,
            expected_query_archive_digest=expected_query_archive_digest,
            freeze=freeze,
            freeze_commit=freeze_commit,
            expected_freeze_commit_digest=expected_freeze_commit_digest,
        )
        return result

    @property
    def query_archive_attached(self) -> bool:
        with self._lock:
            return self._query is not None

    def __call__(
        self,
        binding: PrototypeSceneVerifiedObserverBinding,
        exact_png_bytes: bytes,
    ) -> None:
        if not isinstance(binding, PrototypeSceneVerifiedObserverBinding):
            raise TypeError("binding must be PrototypeSceneVerifiedObserverBinding")
        claimed = (binding.verifier_id, binding.verifier_digest)
        support_identity = (
            self._support.verifier_id,
            self._support.verifier_digest,
        )
        if claimed == support_identity:
            self._support(binding, exact_png_bytes)
            return
        with self._lock:
            query = self._query
        if query is None:
            raise PrototypeSceneRuntimeAdapterError(
                "query archive is not attached and verifier identity is unknown"
            )
        query_identity = (query.verifier_id, query.verifier_digest)
        if claimed != query_identity:
            raise PrototypeSceneRuntimeAdapterError(
                "binding verifier identity is outside both pinned archives"
            )
        query(binding, exact_png_bytes)


def materialize_prototype_scene_calibration_observation(
    archive: PrototypeSceneRuntimeArtifactArchive,
    calibration_plan: PrototypeSceneCalibrationPlan,
    task_id: str,
    panel_id: str,
    *,
    expected_archive_digest: str,
) -> PrototypeSceneCalibrationObservation:
    """Cold-verify one archived calibration turn before adapting its scores."""

    if not isinstance(archive, PrototypeSceneRuntimeArtifactArchive):
        raise TypeError("archive must be PrototypeSceneRuntimeArtifactArchive")
    if not isinstance(calibration_plan, PrototypeSceneCalibrationPlan):
        raise TypeError("calibration_plan must be PrototypeSceneCalibrationPlan")
    if PrototypeSceneCalibrationPlan.from_data(
        calibration_plan.to_data()
    ) != calibration_plan:
        raise PrototypeSceneRuntimeAdapterError("calibration plan is not canonical")
    archive.assert_untampered(expected_archive_digest=expected_archive_digest)
    scene = archive._scene(_identifier(panel_id, "calibration panel_id"))
    if (
        scene.purpose is not PrototypeSceneArtifactPurpose.CALIBRATION
        or scene.scene_task_id != _identifier(task_id, "calibration task_id")
        or scene.expected_observation_context_digest
        != calibration_plan.record_digest
    ):
        raise PrototypeSceneRuntimeAdapterError(
            "calibration archive role, task, or precommitted plan differs"
        )
    scheduled = tuple(
        item
        for item in calibration_plan.scenes
        if item.task_id == task_id and item.panel_id == panel_id
    )
    if len(scheduled) != 1:
        raise PrototypeSceneRuntimeAdapterError(
            "calibration artifact is outside the frozen scene schedule"
        )
    artifact = archive._cold_verified_artifact(scene)
    observation = adapt_prototype_scene_observation(
        artifact,
        calibration_plan_digest=calibration_plan.record_digest,
    )
    if (
        observation.task_id != task_id
        or observation.panel_id != panel_id
        or observation.cohort_plan_digest
        != calibration_plan.cohort_plan_digest
        or observation.description_catalog_digest
        != calibration_plan.description_catalog_digest
        or observation.prototype_reference_digest
        != calibration_plan.prototype_reference_digest
        or observation.observer_protocol_id
        != calibration_plan.observer_protocol_id
        or observation.observer_protocol_digest
        != calibration_plan.observer_protocol_digest
        or observation.model_id != calibration_plan.model_id
        or observation.model_identity_digest
        != calibration_plan.model_identity_digest
        or observation.environment_digest != calibration_plan.environment_digest
    ):
        raise PrototypeSceneRuntimeAdapterError(
            "cold-verified calibration observation identity differs from plan"
        )
    return observation


def materialize_prototype_scene_calibration_observations(
    archive: PrototypeSceneRuntimeArtifactArchive,
    calibration_plan: PrototypeSceneCalibrationPlan,
    *,
    expected_archive_digest: str,
) -> tuple[PrototypeSceneCalibrationObservation, ...]:
    """Cold-adapt the complete frozen calibration schedule in plan order."""

    if not isinstance(archive, PrototypeSceneRuntimeArtifactArchive):
        raise TypeError("archive must be PrototypeSceneRuntimeArtifactArchive")
    if not isinstance(calibration_plan, PrototypeSceneCalibrationPlan):
        raise TypeError("calibration_plan must be PrototypeSceneCalibrationPlan")
    archive.assert_untampered(expected_archive_digest=expected_archive_digest)
    expected_keys = tuple(
        (item.task_id, item.panel_id) for item in calibration_plan.scenes
    )
    archived_keys = tuple(
        (item.scene_task_id, item.panel_id) for item in archive.scenes
    )
    if (
        len(archive.scenes) != len(calibration_plan.scenes)
        or set(archived_keys) != set(expected_keys)
        or len(set(archived_keys)) != len(archived_keys)
        or any(
            item.purpose is not PrototypeSceneArtifactPurpose.CALIBRATION
            for item in archive.scenes
        )
    ):
        raise PrototypeSceneRuntimeAdapterError(
            "calibration archive differs from the complete frozen schedule"
        )
    return tuple(
        materialize_prototype_scene_calibration_observation(
            archive,
            calibration_plan,
            task_id,
            panel_id,
            expected_archive_digest=expected_archive_digest,
        )
        for task_id, panel_id in expected_keys
    )


def materialize_prototype_scene_panel(
    archive: PrototypeSceneRuntimeArtifactArchive,
    family: PrototypeSceneCalibrationFamily,
    panel_id: str,
    *,
    expected_archive_digest: str,
    typed_geometry: object | None = None,
) -> PrototypeScenePanelEvaluation:
    """Cold-verify archived bytes and seal one finite Python panel record."""

    if not isinstance(archive, PrototypeSceneRuntimeArtifactArchive):
        raise TypeError("archive must be PrototypeSceneRuntimeArtifactArchive")
    if not isinstance(family, PrototypeSceneCalibrationFamily):
        raise TypeError("family must be PrototypeSceneCalibrationFamily")
    if PrototypeSceneCalibrationFamily.from_data(family.to_data()) != family:
        raise PrototypeSceneRuntimeAdapterError("calibration family is not canonical")
    verifier = archive.artifact_verifier(
        expected_archive_digest=expected_archive_digest
    )
    scene = archive._scene(_identifier(panel_id, "panel_id"))
    material = archive._cold_material(scene)
    _require_family_context_identity(family, material.context)
    binding = PrototypeSceneVerifiedObserverBinding.seal_verified(
        panel_id=scene.panel_id,
        exact_png_bytes=scene.exact_scene_png_bytes,
        observer_artifact_schema=PROTOTYPE_SCENE_OBSERVER_ARTIFACT_SCHEMA,
        observer_artifact_digest="sha256:" + material.artifact.artifact_digest,
        verifier_id=verifier.verifier_id,
        verifier_digest=verifier.verifier_digest,
        scores=material.scores,
        context=material.context,
    )
    verifier(binding, scene.exact_scene_png_bytes)
    return PrototypeScenePanelEvaluation.seal(
        panel_id=scene.panel_id,
        exact_png_bytes=scene.exact_scene_png_bytes,
        observer_binding=binding,
        family=family,
        context=material.context,
        scores=material.scores,
        typed_geometry=typed_geometry,
    )


__all__ = [
    "RUNTIME_ADAPTER_ID",
    "RUNTIME_ARCHIVE_SCHEMA",
    "PrototypeSceneArtifactPurpose",
    "PrototypeScenePhasedArtifactVerifier",
    "PrototypeSceneRuntimeAdapterError",
    "PrototypeSceneRuntimeArtifactArchive",
    "PrototypeSceneRuntimeArtifactInput",
    "PrototypeSceneRuntimeArtifactVerifier",
    "materialize_prototype_scene_calibration_observation",
    "materialize_prototype_scene_calibration_observations",
    "materialize_prototype_scene_panel",
    "prototype_scene_evaluation_context_digest",
    "prototype_scene_runtime_adapter_source_digest",
]

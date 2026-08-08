"""One-call whole-panel observer for prose-grounded Bongard rubrics.

The vision model sees exactly one neutrally named panel and one frozen ordered
rubric.  It returns only an inclusive interval on a fixed five-level scale.
Pure Python maps that interval to present, certified absence, indeterminate,
or error.  Lean is neither imported nor involved in identity or replay.
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
from typing import Any, Mapping

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_rubric_observer import (
    ObjectBongardRubricSpec,
    OrdinalLevelInterval,
    object_bongard_rubric_observer_source_digest,
)
from bongard import prototype_object_scene_observer as _runtime
from bongard import prototype_scene_observer as _legacy
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
    NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
    run_codex_named_images_structured,
    validate_codex_named_image_receipt,
    validate_codex_strict_output_schema,
)


PANEL_RUBRIC_ARTIFACT_SCHEMA = "gkm.bongard-panel-rubric-observer-artifact.v1"
PANEL_RUBRIC_OBSERVATION_SCHEMA = "gkm.bongard-panel-rubric-observation.v1"
PANEL_RUBRIC_PROTOCOL_ID = "bongard.panel-rubric-observer/one-panel-signed-ordinal-v1"
PANEL_RUBRIC_ORDINAL_LEVEL_ANCHORS: tuple[tuple[int, str], ...] = (
    (0, "The complete panel clearly matches the foil description more aptly than the target description."),
    (1, "The complete panel matches the foil description slightly more aptly than the target description."),
    (2, "The complete panel matches both descriptions equally, matches neither description, or the comparison is genuinely uncertain."),
    (3, "The complete panel matches the target description slightly more aptly than the foil description."),
    (4, "The complete panel clearly matches the target description more aptly than the foil description."),
)

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PANEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}\Z")
_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


class ObjectBongardPanelRubricObserverError(ValueError):
    """A panel observation, commitment, or cold replay is invalid."""


class PanelRubricDisposition(str, Enum):
    PRESENT = "present"
    CERTIFIED_ABSENCE = "certified_absence"
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
        "negation_allowed": False,
        "polarity_flip_allowed": False,
        "threshold_selection_allowed": False,
        "model_selection_allowed": False,
    }


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value) or set(value) != expected:
        raise ObjectBongardPanelRubricObserverError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectBongardPanelRubricObserverError(f"{label} must be a raw SHA-256")
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectBongardPanelRubricObserverError(f"{label} must be a sha256: address")
    return value


def _panel_id(value: object) -> str:
    if not isinstance(value, str) or _PANEL_ID.fullmatch(value) is None:
        raise ObjectBongardPanelRubricObserverError("panel ID is invalid")
    return value


def _canonical_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ObjectBongardPanelRubricObserverError("panel rubric payload must be an object")
    try:
        decoded = json.loads(canonical_json(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardPanelRubricObserverError("panel rubric payload is not canonical JSON") from exc
    if not isinstance(decoded, dict):
        raise ObjectBongardPanelRubricObserverError("panel rubric payload must be an object")
    return decoded


def _receipt_data(value: CodexReceipt | None) -> object:
    return None if value is None else value.to_dict()


def _receipt_from_data(value: object) -> CodexReceipt | None:
    if value is None:
        return None
    result = _legacy._receipt_from_data(value)
    if not isinstance(result, CodexReceipt):
        raise ObjectBongardPanelRubricObserverError("panel rubric receipt has the wrong type")
    return result


def object_bongard_panel_rubric_observer_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def object_bongard_panel_rubric_ordinal_scale_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-panel-rubric-ordinal-scale.v1",
            "anchors": [list(item) for item in PANEL_RUBRIC_ORDINAL_LEVEL_ANCHORS],
            "interval_semantics": "inclusive-narrowest-honest-range",
            "present_rule": "lower-greater-than-or-equal-to-three",
            "certified_absence_rule": "upper-less-than-or-equal-to-one",
            "indeterminate_rule": "contains-deadband-two-or-crosses-decision-regions",
            "error_rule": "transport-or-parser-errors-are-never-absence",
        }
    )


def object_bongard_panel_rubric_output_schema() -> dict[str, object]:
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "lower": {"type": "integer"},
            "upper": {"type": "integer"},
        },
        "required": ["lower", "upper"],
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(schema)
    return schema


def object_bongard_panel_rubric_prompt(rubric_spec: ObjectBongardRubricSpec) -> str:
    if not isinstance(rubric_spec, ObjectBongardRubricSpec):
        raise TypeError("rubric_spec must be ObjectBongardRubricSpec")
    anchors = "\n".join(f"{level}: {meaning}" for level, meaning in PANEL_RUBRIC_ORDINAL_LEVEL_ANCHORS)
    return (
        "Inspect panel.png as one complete drawing. Treat every visible mark together; "
        "the single supplied image is the entire visual evidence. Apply this exact "
        "ordered target-versus-foil comparison to the complete panel:\n"
        f"{rubric_spec.rubric}\n\n"
        "Use only this fixed ordinal scale:\n"
        f"{anchors}\n\n"
        "Level 4 is reserved for a complete panel where the target description is "
        "clearly more apt. Level 0 is reserved for a complete panel where the foil "
        "description is clearly more apt. Level 2 covers both, neither, a tie, and "
        "genuine uncertainty. Return the narrowest honest inclusive lower and upper "
        "levels for the complete panel."
    )


def object_bongard_panel_rubric_protocol_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-panel-rubric-observer-protocol.v1",
            "protocol_id": PANEL_RUBRIC_PROTOCOL_ID,
            "source_digest": object_bongard_panel_rubric_observer_source_digest(),
            "rubric_spec_authority_source_digest": object_bongard_rubric_observer_source_digest(),
            "runtime_helper_source_digest": _runtime.prototype_scene_observer_source_digest(),
            "transport_source_digest": _runtime.prototype_scene_transport_source_digest(),
            "ordinal_scale_digest": object_bongard_panel_rubric_ordinal_scale_digest(),
            "output_schema": object_bongard_panel_rubric_output_schema(),
            "ordered_names": ["panel.png"],
            "physical_calls": 1,
            "whole_panel_only": True,
            **_authority_data(),
        }
    )


def _model_digest(model: str, reasoning_effort: str) -> str:
    if not isinstance(model, str) or not model or not isinstance(reasoning_effort, str) or not reasoning_effort:
        raise ObjectBongardPanelRubricObserverError("model request is invalid")
    return canonical_digest(
        {
            "schema": "gkm.bongard-panel-rubric-model.v1",
            "model": model,
            "reasoning_effort": reasoning_effort,
        }
    )


def _runtime_identity_digest(
    *,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str | None,
    cloud_policy_cache_binding: str,
    model_catalog_digest: str,
    no_tools_attestation_digest: str,
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-panel-rubric-runtime.v1",
            "model_digest": _model_digest(model, reasoning_effort),
            "expected_launcher_digest": expected_launcher_digest,
            "cloud_policy_cache_binding": cloud_policy_cache_binding,
            "model_catalog_digest": model_catalog_digest,
            "no_tools_attestation_digest": no_tools_attestation_digest,
            "runtime_helper_source_digest": _runtime.prototype_scene_observer_source_digest(),
            "transport_source_digest": _runtime.prototype_scene_transport_source_digest(),
        }
    )


def classify_panel_rubric_interval(interval: OrdinalLevelInterval) -> PanelRubricDisposition:
    if not isinstance(interval, OrdinalLevelInterval):
        raise TypeError("interval must be OrdinalLevelInterval")
    if interval.lower >= 3:
        return PanelRubricDisposition.PRESENT
    if interval.upper <= 1:
        return PanelRubricDisposition.CERTIFIED_ABSENCE
    return PanelRubricDisposition.INDETERMINATE


def _observation_content(value: "ObjectBongardPanelRubricObservation") -> dict[str, object]:
    return {
        "schema": PANEL_RUBRIC_OBSERVATION_SCHEMA,
        "rubric_spec_digest": value.rubric_spec_digest,
        "disposition": value.disposition.value,
        "interval": None if value.interval is None else value.interval.to_data(),
        "error_code": value.error_code,
        "error_type": value.error_type,
        "ordinal_scale_digest": object_bongard_panel_rubric_ordinal_scale_digest(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardPanelRubricObservation:
    rubric_spec_digest: str
    disposition: PanelRubricDisposition
    interval: OrdinalLevelInterval | None
    error_code: str | None
    error_type: str | None
    observation_digest: str

    def __post_init__(self) -> None:
        _digest(self.rubric_spec_digest, "observation rubric spec digest")
        if not isinstance(self.disposition, PanelRubricDisposition):
            raise TypeError("disposition has the wrong type")
        if self.disposition is PanelRubricDisposition.ERROR:
            if self.interval is not None or not isinstance(self.error_code, str) or _CODE.fullmatch(self.error_code) is None or not isinstance(self.error_type, str) or _CODE.fullmatch(self.error_type) is None:
                raise ObjectBongardPanelRubricObserverError("error observation differs")
        elif not isinstance(self.interval, OrdinalLevelInterval) or self.error_code is not None or self.error_type is not None or classify_panel_rubric_interval(self.interval) is not self.disposition:
            raise ObjectBongardPanelRubricObserverError("scored observation disposition differs")
        _digest(self.observation_digest, "observation digest")
        if self.observation_digest != canonical_digest(_observation_content(self)):
            raise ObjectBongardPanelRubricObserverError("observation digest differs")

    @classmethod
    def from_interval(cls, rubric_spec_digest: str, interval: OrdinalLevelInterval) -> "ObjectBongardPanelRubricObservation":
        values = {
            "rubric_spec_digest": rubric_spec_digest,
            "disposition": classify_panel_rubric_interval(interval),
            "interval": interval,
            "error_code": None,
            "error_type": None,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, observation_digest=canonical_digest(_observation_content(provisional)))

    @classmethod
    def error(cls, rubric_spec_digest: str, error_code: str, error_type: str) -> "ObjectBongardPanelRubricObservation":
        values = {
            "rubric_spec_digest": rubric_spec_digest,
            "disposition": PanelRubricDisposition.ERROR,
            "interval": None,
            "error_code": error_code,
            "error_type": error_type,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, observation_digest=canonical_digest(_observation_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_observation_content(self), "observation_digest": self.observation_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardPanelRubricObservation":
        raw = _fields(
            value,
            {"schema", "rubric_spec_digest", "disposition", "interval", "error_code", "error_type", "ordinal_scale_digest", "observation_digest"},
            "panel rubric observation",
        )
        if raw["schema"] != PANEL_RUBRIC_OBSERVATION_SCHEMA or raw["ordinal_scale_digest"] != object_bongard_panel_rubric_ordinal_scale_digest():
            raise ObjectBongardPanelRubricObserverError("observation policy differs")
        try:
            disposition = PanelRubricDisposition(raw["disposition"])
        except (TypeError, ValueError) as exc:
            raise ObjectBongardPanelRubricObserverError("observation disposition is unknown") from exc
        result = cls(
            raw["rubric_spec_digest"],
            disposition,
            None if raw["interval"] is None else OrdinalLevelInterval.from_data(raw["interval"]),
            raw["error_code"],
            raw["error_type"],
            raw["observation_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardPanelRubricObserverError("observation is not canonical")
        return result


def _parse_payload(value: object, rubric_spec_digest: str) -> ObjectBongardPanelRubricObservation:
    raw = _fields(value, {"lower", "upper"}, "panel rubric payload")
    return ObjectBongardPanelRubricObservation.from_interval(
        rubric_spec_digest, OrdinalLevelInterval(raw["lower"], raw["upper"])
    )


def _artifact_content(value: "ObjectBongardPanelRubricArtifact") -> dict[str, object]:
    return {
        "schema": PANEL_RUBRIC_ARTIFACT_SCHEMA,
        "panel_id": value.panel_id,
        "panel_digest": value.panel_digest,
        "observation_context_digest": value.observation_context_digest,
        "rubric_spec": value.rubric_spec.to_data(),
        "rubric_spec_digest": value.rubric_spec_digest,
        "source_digest": value.source_digest,
        "rubric_spec_authority_source_digest": value.rubric_spec_authority_source_digest,
        "protocol_digest": value.protocol_digest,
        "transport_source_digest": value.transport_source_digest,
        "prompt_digest": value.prompt_digest,
        "output_schema_digest": value.output_schema_digest,
        "model": value.model,
        "reasoning_effort": value.reasoning_effort,
        "model_digest": value.model_digest,
        "expected_launcher_digest": value.expected_launcher_digest,
        "cloud_policy_cache_binding": value.cloud_policy_cache_binding,
        "model_catalog_digest": value.model_catalog_digest,
        "no_tools_attestation_digest": value.no_tools_attestation_digest,
        "runtime_identity_digest": value.runtime_identity_digest,
        "presentation": [item.to_data() for item in value.presentation],
        "physical_call_count": value.physical_call_count,
        "status": value.status.value,
        "model_payload": value.model_payload,
        "receipt": _receipt_data(value.receipt),
        "observation": value.observation.to_data(),
        "failure_code": value.failure_code,
        "failure_type": value.failure_type,
        "whole_panel_only": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardPanelRubricArtifact:
    panel_id: str
    panel_digest: str
    observation_context_digest: str
    rubric_spec: ObjectBongardRubricSpec
    rubric_spec_digest: str
    source_digest: str
    rubric_spec_authority_source_digest: str
    protocol_digest: str
    transport_source_digest: str
    prompt_digest: str
    output_schema_digest: str
    model: str
    reasoning_effort: str
    model_digest: str
    expected_launcher_digest: str | None
    cloud_policy_cache_binding: str
    model_catalog_digest: str
    no_tools_attestation_digest: str
    runtime_identity_digest: str
    presentation: tuple[PrototypeImageIdentity, ...]
    physical_call_count: int
    status: PrototypeSceneObserverStatus
    model_payload: Mapping[str, Any] | None
    receipt: CodexReceipt | None
    observation: ObjectBongardPanelRubricObservation
    failure_code: str | None
    failure_type: str | None
    artifact_digest: str

    def __post_init__(self) -> None:
        _panel_id(self.panel_id)
        _digest(self.panel_digest, "panel digest")
        _address(self.observation_context_digest, "observation context digest")
        if not isinstance(self.rubric_spec, ObjectBongardRubricSpec):
            raise TypeError("rubric spec has the wrong type")
        for name in (
            "rubric_spec_digest", "source_digest", "rubric_spec_authority_source_digest",
            "protocol_digest", "transport_source_digest", "prompt_digest",
            "output_schema_digest", "model_digest", "model_catalog_digest",
            "no_tools_attestation_digest", "runtime_identity_digest", "artifact_digest",
        ):
            _digest(getattr(self, name), name)
        if self.expected_launcher_digest is None:
            raise ObjectBongardPanelRubricObserverError(
                "artifact lacks the required launcher commitment"
            )
        _digest(self.expected_launcher_digest, "expected launcher digest")
        if self.cloud_policy_cache_binding != "absent":
            _address(self.cloud_policy_cache_binding, "cloud policy cache binding")
        prompt = object_bongard_panel_rubric_prompt(self.rubric_spec)
        schema = object_bongard_panel_rubric_output_schema()
        expected_runtime = _runtime_identity_digest(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            expected_launcher_digest=self.expected_launcher_digest,
            cloud_policy_cache_binding=self.cloud_policy_cache_binding,
            model_catalog_digest=self.model_catalog_digest,
            no_tools_attestation_digest=self.no_tools_attestation_digest,
        )
        if (
            self.rubric_spec_digest != self.rubric_spec.spec_digest
            or self.source_digest != object_bongard_panel_rubric_observer_source_digest()
            or self.rubric_spec_authority_source_digest != object_bongard_rubric_observer_source_digest()
            or self.protocol_digest != object_bongard_panel_rubric_protocol_digest()
            or self.transport_source_digest != _runtime.prototype_scene_transport_source_digest()
            or self.prompt_digest != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or self.output_schema_digest != canonical_digest(schema)
            or self.model_digest != _model_digest(self.model, self.reasoning_effort)
            or self.runtime_identity_digest != expected_runtime
        ):
            raise ObjectBongardPanelRubricObserverError("artifact source, protocol, or runtime binding differs")
        if (
            not isinstance(self.presentation, tuple)
            or len(self.presentation) != 1
            or self.presentation[0].name != "panel.png"
            or self.presentation[0].content_digest != self.panel_digest
            or self.physical_call_count != 1
        ):
            raise ObjectBongardPanelRubricObserverError("artifact must bind one complete panel image and one call")
        if not isinstance(self.status, PrototypeSceneObserverStatus) or not isinstance(self.observation, ObjectBongardPanelRubricObservation) or self.observation.rubric_spec_digest != self.rubric_spec_digest:
            raise ObjectBongardPanelRubricObserverError("artifact observation binding differs")
        if self.model_payload is not None:
            object.__setattr__(self, "model_payload", _canonical_payload(self.model_payload))
        if self.status is PrototypeSceneObserverStatus.SUCCESS:
            if self.model_payload is None or self.receipt is None or self.failure_code is not None or self.failure_type is not None or self.observation != _parse_payload(self.model_payload, self.rubric_spec_digest):
                raise ObjectBongardPanelRubricObserverError("successful artifact differs from payload")
        elif self.status in (PrototypeSceneObserverStatus.PARSER_ERROR, PrototypeSceneObserverStatus.TRANSPORT_ERROR):
            if self.observation.disposition is not PanelRubricDisposition.ERROR or not isinstance(self.failure_code, str) or _CODE.fullmatch(self.failure_code) is None or not isinstance(self.failure_type, str) or _CODE.fullmatch(self.failure_type) is None:
                raise ObjectBongardPanelRubricObserverError("failed artifact lacks typed error evidence")
            if (
                self.observation.error_code != self.failure_code
                or self.observation.error_type != self.failure_type
            ):
                raise ObjectBongardPanelRubricObserverError(
                    "artifact error observation differs from its failure"
                )
            if self.status is PrototypeSceneObserverStatus.PARSER_ERROR and (self.model_payload is None or self.receipt is None):
                raise ObjectBongardPanelRubricObserverError("parser error lacks receipted payload")
            if self.status is PrototypeSceneObserverStatus.TRANSPORT_ERROR and (self.model_payload is not None or self.receipt is not None):
                raise ObjectBongardPanelRubricObserverError("transport error contains a payload")
        else:
            raise ObjectBongardPanelRubricObserverError("artifact status is unsupported")
        if self.receipt is not None:
            receipt = self.receipt
            view = [item.to_data() for item in self.presentation]
            expected_set = "sha256:" + canonical_digest({"schema": NAMED_IMAGE_INPUT_DIGEST_SCHEMA, "images": view})
            if (
                receipt.prompt_digest != self.prompt_digest
                or receipt.output_schema_digest != self.output_schema_digest
                or receipt.structured_output_digest != canonical_digest(dict(self.model_payload or {}))
                or receipt.panel_view_digest != canonical_digest(view)
                or receipt.panel_set_digest != expected_set
                or receipt.requested_model != self.model
                or receipt.requested_reasoning_effort != self.reasoning_effort
                or (self.expected_launcher_digest is not None and receipt.codex_launcher_digest != self.expected_launcher_digest)
                or receipt.cloud_config_bundle_cache_binding != self.cloud_policy_cache_binding
                or receipt.model_catalog_digest != self.model_catalog_digest
                or receipt.tool_surface_attestation_digest != self.no_tools_attestation_digest
            ):
                raise ObjectBongardPanelRubricObserverError("artifact receipt binding differs")
        if self.artifact_digest != canonical_digest(_artifact_content(self)):
            raise ObjectBongardPanelRubricObserverError("artifact digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_artifact_content(self), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardPanelRubricArtifact":
        raw = _fields(value, set(_artifact_content_fields()) | {"artifact_digest"}, "panel rubric artifact")
        if raw["schema"] != PANEL_RUBRIC_ARTIFACT_SCHEMA or raw["whole_panel_only"] is not True or any(raw[key] != item for key, item in _authority_data().items()) or not isinstance(raw["presentation"], list):
            raise ObjectBongardPanelRubricObserverError("artifact policy differs")
        try:
            status = PrototypeSceneObserverStatus(raw["status"])
        except (TypeError, ValueError) as exc:
            raise ObjectBongardPanelRubricObserverError("artifact status is unknown") from exc
        result = cls(
            raw["panel_id"], raw["panel_digest"], raw["observation_context_digest"],
            ObjectBongardRubricSpec.from_data(raw["rubric_spec"]), raw["rubric_spec_digest"],
            raw["source_digest"], raw["rubric_spec_authority_source_digest"], raw["protocol_digest"],
            raw["transport_source_digest"], raw["prompt_digest"], raw["output_schema_digest"],
            raw["model"], raw["reasoning_effort"], raw["model_digest"], raw["expected_launcher_digest"],
            raw["cloud_policy_cache_binding"], raw["model_catalog_digest"], raw["no_tools_attestation_digest"],
            raw["runtime_identity_digest"], tuple(PrototypeImageIdentity.from_data(item) for item in raw["presentation"]),
            raw["physical_call_count"], status, raw["model_payload"], _receipt_from_data(raw["receipt"]),
            ObjectBongardPanelRubricObservation.from_data(raw["observation"]), raw["failure_code"],
            raw["failure_type"], raw["artifact_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardPanelRubricObserverError("artifact is not canonical")
        return result


def _artifact_content_fields() -> tuple[str, ...]:
    return (
        "schema", "panel_id", "panel_digest", "observation_context_digest", "rubric_spec",
        "rubric_spec_digest", "source_digest", "rubric_spec_authority_source_digest",
        "protocol_digest", "transport_source_digest", "prompt_digest", "output_schema_digest",
        "model", "reasoning_effort", "model_digest", "expected_launcher_digest",
        "cloud_policy_cache_binding", "model_catalog_digest", "no_tools_attestation_digest",
        "runtime_identity_digest", "presentation", "physical_call_count", "status",
        "model_payload", "receipt", "observation", "failure_code", "failure_type",
        "whole_panel_only", *_authority_data(),
    )


def _seal_artifact(
    *,
    panel_id: str,
    panel_digest: str,
    context: str,
    rubric_spec: ObjectBongardRubricSpec,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str | None,
    cloud_policy_cache_binding: str,
    model_catalog_digest: str,
    no_tools_attestation_digest: str,
    presentation: tuple[PrototypeImageIdentity, ...],
    status: PrototypeSceneObserverStatus,
    payload: Mapping[str, Any] | None,
    receipt: CodexReceipt | None,
    observation: ObjectBongardPanelRubricObservation,
    failure_code: str | None,
    failure_type: str | None,
) -> ObjectBongardPanelRubricArtifact:
    prompt = object_bongard_panel_rubric_prompt(rubric_spec)
    schema = object_bongard_panel_rubric_output_schema()
    values = {
        "panel_id": panel_id,
        "panel_digest": panel_digest,
        "observation_context_digest": context,
        "rubric_spec": rubric_spec,
        "rubric_spec_digest": rubric_spec.spec_digest,
        "source_digest": object_bongard_panel_rubric_observer_source_digest(),
        "rubric_spec_authority_source_digest": object_bongard_rubric_observer_source_digest(),
        "protocol_digest": object_bongard_panel_rubric_protocol_digest(),
        "transport_source_digest": _runtime.prototype_scene_transport_source_digest(),
        "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "output_schema_digest": canonical_digest(schema),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "model_digest": _model_digest(model, reasoning_effort),
        "expected_launcher_digest": expected_launcher_digest,
        "cloud_policy_cache_binding": cloud_policy_cache_binding,
        "model_catalog_digest": model_catalog_digest,
        "no_tools_attestation_digest": no_tools_attestation_digest,
        "runtime_identity_digest": _runtime_identity_digest(
            model=model, reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=cloud_policy_cache_binding,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_attestation_digest,
        ),
        "presentation": presentation,
        "physical_call_count": 1,
        "status": status,
        "model_payload": None if payload is None else _canonical_payload(payload),
        "receipt": receipt,
        "observation": observation,
        "failure_code": failure_code,
        "failure_type": failure_type,
    }
    provisional = object.__new__(ObjectBongardPanelRubricArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardPanelRubricArtifact(**values, artifact_digest=canonical_digest(_artifact_content(provisional)))


def observe_object_bongard_panel_rubric(
    png_bytes: bytes,
    *,
    panel_id: str,
    rubric_spec: ObjectBongardRubricSpec,
    expected_panel_sha256: str,
    expected_rubric_spec_digest: str,
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
) -> ObjectBongardPanelRubricArtifact:
    """Observe one complete panel with exactly one no-tools vision call."""

    panel = _legacy._validate_exact_png(png_bytes, "panel")
    identity = _panel_id(panel_id)
    panel_digest = hashlib.sha256(panel).hexdigest()
    if panel_digest != _digest(expected_panel_sha256, "expected panel digest"):
        raise ObjectBongardPanelRubricObserverError("panel bytes differ from commitment")
    if not isinstance(rubric_spec, ObjectBongardRubricSpec) or rubric_spec.spec_digest != _digest(expected_rubric_spec_digest, "expected rubric spec digest"):
        raise ObjectBongardPanelRubricObserverError("rubric spec differs from commitment")
    if not callable(transport):
        raise TypeError("transport must be callable")
    policy, model_catalog_digest, no_tools_digest = _runtime._runtime_identities(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    context = observation_context_digest or "sha256:" + canonical_digest(
        {
            "schema": "gkm.bongard-panel-rubric-observation-context.v1",
            "panel_id": identity,
            "panel_digest": panel_digest,
            "rubric_spec_digest": rubric_spec.spec_digest,
            "model": model,
            "reasoning_effort": reasoning_effort,
        }
    )
    _address(context, "observation context digest")
    prompt = object_bongard_panel_rubric_prompt(rubric_spec)
    schema = object_bongard_panel_rubric_output_schema()
    presentation_bytes = (("panel.png", panel),)
    presentation = _legacy._image_identities(presentation_bytes)
    _legacy._assert_model_visible_boundary(
        prompt,
        schema,
        ("panel.png",),
        hidden_values=(identity, panel_digest, rubric_spec.spec_digest, context),
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
        return _seal_artifact(
            panel_id=identity, panel_digest=panel_digest, context=context,
            rubric_spec=rubric_spec, model=model, reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=policy, model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_digest, presentation=presentation,
            status=PrototypeSceneObserverStatus.TRANSPORT_ERROR, payload=None, receipt=None,
            observation=ObjectBongardPanelRubricObservation.error(
                rubric_spec.spec_digest, "observer_transport_failed", error_type
            ),
            failure_code="observer_transport_failed", failure_type=error_type,
        )
    try:
        observation = _parse_payload(payload, rubric_spec.spec_digest)
    except Exception as exc:
        error_type = _legacy._exception_type(exc)
        return _seal_artifact(
            panel_id=identity, panel_digest=panel_digest, context=context,
            rubric_spec=rubric_spec, model=model, reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=policy, model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_digest, presentation=presentation,
            status=PrototypeSceneObserverStatus.PARSER_ERROR, payload=payload, receipt=receipt,
            observation=ObjectBongardPanelRubricObservation.error(
                rubric_spec.spec_digest, "observer_payload_rejected", error_type
            ),
            failure_code="observer_payload_rejected", failure_type=error_type,
        )
    return _seal_artifact(
        panel_id=identity, panel_digest=panel_digest, context=context,
        rubric_spec=rubric_spec, model=model, reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=policy, model_catalog_digest=model_catalog_digest,
        no_tools_attestation_digest=no_tools_digest, presentation=presentation,
        status=PrototypeSceneObserverStatus.SUCCESS, payload=payload, receipt=receipt,
        observation=observation, failure_code=None, failure_type=None,
    )


def verify_object_bongard_panel_rubric_artifact(
    artifact: ObjectBongardPanelRubricArtifact,
    png_bytes: bytes,
    *,
    panel_id: str,
    rubric_spec: ObjectBongardRubricSpec,
    expected_artifact_digest: str,
    expected_runtime_identity_digest: str | None = None,
) -> ObjectBongardPanelRubricArtifact:
    """Cold-replay exact pixels, prompt, schema, payload, receipt, and projection."""

    if not isinstance(artifact, ObjectBongardPanelRubricArtifact):
        raise TypeError("artifact must be ObjectBongardPanelRubricArtifact")
    restored = ObjectBongardPanelRubricArtifact.from_data(artifact.to_data())
    if restored.artifact_digest != _digest(expected_artifact_digest, "expected artifact digest"):
        raise ObjectBongardPanelRubricObserverError("artifact differs from commitment")
    if expected_runtime_identity_digest is not None and restored.runtime_identity_digest != _digest(expected_runtime_identity_digest, "expected runtime digest"):
        raise ObjectBongardPanelRubricObserverError("runtime differs from commitment")
    panel = _legacy._validate_exact_png(png_bytes, "panel")
    if (
        restored.panel_id != _panel_id(panel_id)
        or restored.panel_digest != hashlib.sha256(panel).hexdigest()
        or restored.presentation[0].byte_count != len(panel)
        or restored.rubric_spec != ObjectBongardRubricSpec.from_data(rubric_spec.to_data())
    ):
        raise ObjectBongardPanelRubricObserverError("cold replay inputs differ")
    if restored.receipt is not None:
        assert restored.model_payload is not None
        prompt = object_bongard_panel_rubric_prompt(restored.rubric_spec)
        schema = object_bongard_panel_rubric_output_schema()
        with tempfile.TemporaryDirectory(prefix="bongard-panel-rubric-replay-") as raw:
            target = Path(raw) / "panel.png"
            target.write_bytes(panel)
            validate_codex_named_image_receipt(
                restored.receipt,
                prompt,
                (str(target.resolve()),),
                ("panel.png",),
                schema,
                dict(restored.model_payload),
            )
            if target.read_bytes() != panel:
                raise ObjectBongardPanelRubricObserverError("cold replay panel changed")
    return restored


__all__ = (
    "ObjectBongardPanelRubricArtifact",
    "ObjectBongardPanelRubricObservation",
    "ObjectBongardPanelRubricObserverError",
    "PANEL_RUBRIC_ORDINAL_LEVEL_ANCHORS",
    "PANEL_RUBRIC_PROTOCOL_ID",
    "PanelRubricDisposition",
    "classify_panel_rubric_interval",
    "object_bongard_panel_rubric_observer_source_digest",
    "object_bongard_panel_rubric_ordinal_scale_digest",
    "object_bongard_panel_rubric_output_schema",
    "object_bongard_panel_rubric_prompt",
    "object_bongard_panel_rubric_protocol_digest",
    "observe_object_bongard_panel_rubric",
    "verify_object_bongard_panel_rubric_artifact",
)

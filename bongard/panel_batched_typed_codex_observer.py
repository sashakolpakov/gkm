"""One-call, candidate-independent typed observation of a whole panel.

The production single-axis observer is intentionally strict, but paying one
vision invocation for every axis is unnecessary when all axes share the same
raw panel and panel-only context.  This adapter batches the *complete*
registered whole-panel axis catalog into one neutral named-image call.  Python
still builds each per-axis view, schema, and observation with the existing
single-axis protocol.

Only ``panel.png`` and the closed axis views are model-visible.  Axis aliases
are opaque and canonical.  Task ids, support roles, semantic-side labels,
query identity, formulas, and selected feature specifications are not inputs
to this API.  The response is engineering-only and uncalibrated.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from bongard import prototype_scene_observer as _scene_runtime
from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import CodexNoToolsAttestation
from bongard.panel_feature_observation import (
    FeatureAxis,
    PanelFeatureObservationSet,
)
from bongard.panel_feature_observer_protocol import (
    FeatureAxisObservationView,
    feature_axis_observer_output_schema,
    feature_axis_observer_prompt,
    parse_feature_axis_observer_payload,
)
from bongard.panel_owner_inventory import PANEL_OWNER_NEUTRAL_IMAGE_NAME
from bongard.panel_soft_ontology import FAMILY_CONTRACTS, SubjectScope
from bongard.panel_typed_codex_observer import (
    PanelOnlyObservationContext,
    PanelTypedCodexObserverError,
    TypedCodexRuntimeBinding,
    _bind_runtime,
    _canonical_payload,
    _digest,
    _exact_png,
    _receipt_from_data,
    _validate_receipt_binding,
    typed_codex_observer_contract_digest,
    typed_measurement_protocol_digest,
)
from bongard.prototype_scene_observer import PrototypeImageIdentity
from bongard.transport import (
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexReceipt,
    run_codex_named_images_structured,
    validate_codex_named_image_receipt,
    validate_codex_strict_output_schema,
)


BATCHED_AXIS_ALIAS_SCHEMA = "gkm.bongard-batched-feature-axis-alias.v1"
BATCHED_AXIS_REQUEST_SCHEMA = "gkm.bongard-batched-feature-axis-request.v1"
BATCHED_AXIS_MODEL_VIEW_SCHEMA = "gkm.bongard-batched-feature-axis-model-view.v1"
BATCHED_AXIS_ARTIFACT_SCHEMA = "gkm.bongard-batched-typed-axis-artifact.v1"
BATCHED_AXIS_CONTRACT_SCHEMA = "gkm.bongard-batched-typed-axis-contract.v1"
BATCHED_AXIS_PROTOCOL_ID = (
    "bongard.panel-batched-typed-codex-observer/"
    "one-panel-complete-whole-panel-catalog-v1"
)

# These are transport-envelope limits, not claims about visual validity.
MAX_BATCHED_AXES = 32
MAX_BATCHED_PROMPT_BYTES = 256 * 1024
MAX_BATCHED_OUTPUT_SCHEMA_BYTES = 256 * 1024
MAX_BATCHED_RESPONSE_BYTES = 256 * 1024

_AXIS_ALIAS = re.compile(r"axis_[0-9]{4}\Z")


class PanelBatchedTypedCodexObserverError(PanelTypedCodexObserverError):
    """A batched request, receipt, payload, or cold replay is invalid."""


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise PanelBatchedTypedCodexObserverError(f"{label} fields differ")
    return value


def panel_batched_typed_codex_observer_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def complete_whole_panel_feature_axes() -> tuple[FeatureAxis, ...]:
    """Return every registered whole-panel axis in content-addressed order."""

    axes = tuple(
        FeatureAxis(family, scope, frame)
        for family, contract in FAMILY_CONTRACTS.items()
        for scope, frame in contract.allowed_scope_frames
        if scope is SubjectScope.WHOLE_PANEL
    )
    ordered = tuple(sorted(axes, key=lambda item: item.axis_digest))
    if not ordered or len(ordered) > MAX_BATCHED_AXES:
        raise PanelBatchedTypedCodexObserverError(
            "complete whole-panel axis catalog exceeds the fixed batch capacity"
        )
    if len({item.axis_digest for item in ordered}) != len(ordered):
        raise PanelBatchedTypedCodexObserverError(
            "complete whole-panel axis catalog is not unique"
        )
    return ordered


@dataclass(frozen=True, slots=True)
class BatchedFeatureAxisAlias:
    """Internal custody map from an opaque alias to one complete axis view."""

    alias: str
    view: FeatureAxisObservationView

    def __post_init__(self) -> None:
        if type(self.alias) is not str or _AXIS_ALIAS.fullmatch(self.alias) is None:
            raise PanelBatchedTypedCodexObserverError("batched axis alias differs")
        if type(self.view) is not FeatureAxisObservationView:
            raise TypeError("batched axis alias needs FeatureAxisObservationView")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": BATCHED_AXIS_ALIAS_SCHEMA,
            "alias": self.alias,
            "view": self.view.to_data(),
        }

    @classmethod
    def from_data(cls, value: object) -> "BatchedFeatureAxisAlias":
        raw = _fields(value, {"schema", "alias", "view"}, "batched axis alias")
        if raw["schema"] != BATCHED_AXIS_ALIAS_SCHEMA:
            raise PanelBatchedTypedCodexObserverError(
                "batched axis alias schema differs"
            )
        result = cls(
            raw["alias"], FeatureAxisObservationView.from_data(raw["view"])
        )
        if result.to_data() != dict(raw):
            raise PanelBatchedTypedCodexObserverError(
                "batched axis alias is not canonical"
            )
        return result


@dataclass(frozen=True, slots=True)
class BatchedFeatureAxisRequest:
    """Frozen caller request; v1 accepts only the complete whole-panel catalog."""

    panel_only_context: PanelOnlyObservationContext
    axes: tuple[FeatureAxis, ...]
    aliases: tuple[BatchedFeatureAxisAlias, ...]

    def __post_init__(self) -> None:
        if type(self.panel_only_context) is not PanelOnlyObservationContext:
            raise TypeError("batched request needs PanelOnlyObservationContext")
        if type(self.axes) is not tuple or any(
            type(item) is not FeatureAxis for item in self.axes
        ):
            raise TypeError("batched request axes must be a FeatureAxis tuple")
        complete = complete_whole_panel_feature_axes()
        if self.axes != complete:
            raise PanelBatchedTypedCodexObserverError(
                "batched request must use the exact canonical complete whole-panel axis tuple"
            )
        context = self.panel_only_context.to_observation_context()
        expected_aliases = tuple(
            BatchedFeatureAxisAlias(
                f"axis_{index:04d}", FeatureAxisObservationView.build(context, axis)
            )
            for index, axis in enumerate(complete)
        )
        if type(self.aliases) is not tuple or self.aliases != expected_aliases:
            raise PanelBatchedTypedCodexObserverError(
                "batched request alias/view map differs"
            )

    @classmethod
    def build(
        cls,
        panel_only_context: PanelOnlyObservationContext,
        axes: tuple[FeatureAxis, ...],
    ) -> "BatchedFeatureAxisRequest":
        if type(panel_only_context) is not PanelOnlyObservationContext:
            raise TypeError("batched request needs PanelOnlyObservationContext")
        if type(axes) is not tuple or any(type(item) is not FeatureAxis for item in axes):
            raise TypeError("batched request axes must be a FeatureAxis tuple")
        if axes != complete_whole_panel_feature_axes():
            raise PanelBatchedTypedCodexObserverError(
                "batched request must use the exact canonical complete whole-panel axis tuple"
            )
        context = panel_only_context.to_observation_context()
        aliases = tuple(
            BatchedFeatureAxisAlias(
                f"axis_{index:04d}", FeatureAxisObservationView.build(context, axis)
            )
            for index, axis in enumerate(axes)
        )
        return cls(panel_only_context, axes, aliases)

    @property
    def request_digest(self) -> str:
        return canonical_digest(self.to_data())

    @property
    def axis_set_digest(self) -> str:
        return canonical_digest(
            {
                "schema": "gkm.bongard-batched-feature-axis-set.v1",
                "axis_digests": [item.axis_digest for item in self.axes],
            }
        )

    def model_data(self) -> dict[str, object]:
        """Return the entire and only inert data embedded in the model prompt."""

        return {
            "schema": BATCHED_AXIS_MODEL_VIEW_SCHEMA,
            "panel_name": PANEL_OWNER_NEUTRAL_IMAGE_NAME,
            "axes": [
                {
                    "axis_alias": item.alias,
                    "axis_measurement": item.view.model_data(),
                }
                for item in self.aliases
            ],
        }

    def to_data(self) -> dict[str, object]:
        return {
            "schema": BATCHED_AXIS_REQUEST_SCHEMA,
            "protocol_id": BATCHED_AXIS_PROTOCOL_ID,
            "panel_only_context": self.panel_only_context.to_data(),
            "axes": [item.to_data() for item in self.axes],
            "aliases": [item.to_data() for item in self.aliases],
            "axis_set_digest": self.axis_set_digest,
            "model_view_digest": canonical_digest(self.model_data()),
            "axis_subset_permitted": False,
            "caller_axis_order_model_visible": False,
            "task_metadata_model_visible": False,
            "selected_candidate_specs_model_visible": False,
            "native_task_orientation_model_visible": False,
            "support_or_query_role_model_visible": False,
            "frozen_formula_model_visible": False,
            "python_is_canonical_authority": True,
            "engineering_only": True,
            "scientific_calibration_supplied": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "BatchedFeatureAxisRequest":
        raw = _fields(
            value,
            {
                "schema",
                "protocol_id",
                "panel_only_context",
                "axes",
                "aliases",
                "axis_set_digest",
                "model_view_digest",
                "axis_subset_permitted",
                "caller_axis_order_model_visible",
                "task_metadata_model_visible",
                "selected_candidate_specs_model_visible",
                "native_task_orientation_model_visible",
                "support_or_query_role_model_visible",
                "frozen_formula_model_visible",
                "python_is_canonical_authority",
                "engineering_only",
                "scientific_calibration_supplied",
            },
            "batched feature axis request",
        )
        if (
            raw["schema"] != BATCHED_AXIS_REQUEST_SCHEMA
            or raw["protocol_id"] != BATCHED_AXIS_PROTOCOL_ID
            or type(raw["axes"]) is not list
            or type(raw["aliases"]) is not list
            or raw["axis_subset_permitted"] is not False
            or raw["caller_axis_order_model_visible"] is not False
            or raw["task_metadata_model_visible"] is not False
            or raw["selected_candidate_specs_model_visible"] is not False
            or raw["native_task_orientation_model_visible"] is not False
            or raw["support_or_query_role_model_visible"] is not False
            or raw["frozen_formula_model_visible"] is not False
            or raw["python_is_canonical_authority"] is not True
            or raw["engineering_only"] is not True
            or raw["scientific_calibration_supplied"] is not False
        ):
            raise PanelBatchedTypedCodexObserverError(
                "batched feature axis request policy differs"
            )
        result = cls(
            PanelOnlyObservationContext.from_data(raw["panel_only_context"]),
            tuple(FeatureAxis.from_data(item) for item in raw["axes"]),
            tuple(BatchedFeatureAxisAlias.from_data(item) for item in raw["aliases"]),
        )
        if (
            raw["axis_set_digest"] != result.axis_set_digest
            or raw["model_view_digest"] != canonical_digest(result.model_data())
            or result.to_data() != dict(raw)
        ):
            raise PanelBatchedTypedCodexObserverError(
                "batched feature axis request digest differs"
            )
        return result


def batched_feature_axis_output_schema(
    request: BatchedFeatureAxisRequest,
) -> dict[str, object]:
    """Nest each existing strict per-axis schema under its opaque alias."""

    if type(request) is not BatchedFeatureAxisRequest:
        raise TypeError("batched schema needs BatchedFeatureAxisRequest")
    schema = {
        "type": "object",
        "properties": {
            item.alias: feature_axis_observer_output_schema(item.view)
            for item in request.aliases
        },
        "required": [item.alias for item in request.aliases],
        "additionalProperties": False,
    }
    encoded = canonical_json(schema)
    if len(encoded) > MAX_BATCHED_OUTPUT_SCHEMA_BYTES:
        raise PanelBatchedTypedCodexObserverError(
            "batched output schema exceeds the fixed byte capacity"
        )
    validate_codex_strict_output_schema(schema)
    return schema


def batched_feature_axis_prompt(request: BatchedFeatureAxisRequest) -> str:
    """Wrap every existing per-axis prompt under its canonical opaque alias."""

    if type(request) is not BatchedFeatureAxisRequest:
        raise TypeError("batched prompt needs BatchedFeatureAxisRequest")
    axis_protocols = "\n\n".join(
        (
            f"BEGIN_AXIS_PROTOCOL {item.alias}\n"
            f"{feature_axis_observer_prompt(item.view)}\n"
            f"END_AXIS_PROTOCOL {item.alias}"
        )
        for item in request.aliases
    )
    prompt = (
        "Inspect the one neutral image panel.png once. The independent protocols below "
        "cover the complete fixed whole-panel axis catalog. Follow every protocol and "
        "return exactly one top-level result under each opaque axis alias. Each nested "
        "result must have exactly the binding fields required by that protocol. Do not "
        "compare this drawing with another drawing and do not infer a preferred task "
        "answer.\n\n"
        f"{axis_protocols}"
    )
    if len(prompt.encode("utf-8")) > MAX_BATCHED_PROMPT_BYTES:
        raise PanelBatchedTypedCodexObserverError(
            "batched prompt exceeds the fixed byte capacity"
        )
    return prompt


def parse_batched_feature_axis_payload(
    request: BatchedFeatureAxisRequest,
    payload: object,
    *,
    observer_contract_digest: str,
    measurement_protocol_digest: str,
    observation_receipt_digest: str,
) -> PanelFeatureObservationSet:
    """Strictly replay every nested result with the existing per-axis parser."""

    if type(request) is not BatchedFeatureAxisRequest:
        raise TypeError("batched parser needs BatchedFeatureAxisRequest")
    for label, value in (
        ("observer contract digest", observer_contract_digest),
        ("measurement protocol digest", measurement_protocol_digest),
        ("observation receipt digest", observation_receipt_digest),
    ):
        _digest(value, label)
    frozen = _canonical_payload(payload, "batched model payload")
    if len(canonical_json(frozen)) > MAX_BATCHED_RESPONSE_BYTES:
        raise PanelBatchedTypedCodexObserverError(
            "batched response exceeds the fixed byte capacity"
        )
    expected_aliases = {item.alias for item in request.aliases}
    raw = _fields(frozen, expected_aliases, "batched axis payload")
    observations = tuple(
        sorted(
            (
                parse_feature_axis_observer_payload(
                    item.view,
                    raw[item.alias],
                    observer_contract_digest=observer_contract_digest,
                    measurement_protocol_digest=measurement_protocol_digest,
                    observation_receipt_digest=observation_receipt_digest,
                )
                for item in request.aliases
            ),
            key=lambda item: item.axis.axis_digest,
        )
    )
    return PanelFeatureObservationSet(
        request.panel_only_context.panel_png_digest,
        observer_contract_digest,
        measurement_protocol_digest,
        observations,
    )


def batched_typed_codex_observer_contract_data(
    runtime: TypedCodexRuntimeBinding,
) -> dict[str, object]:
    if type(runtime) is not TypedCodexRuntimeBinding:
        raise TypeError("batched contract needs TypedCodexRuntimeBinding")
    axes = complete_whole_panel_feature_axes()
    return {
        "schema": BATCHED_AXIS_CONTRACT_SCHEMA,
        "protocol_id": BATCHED_AXIS_PROTOCOL_ID,
        "adapter_source_digest": panel_batched_typed_codex_observer_source_digest(),
        "base_observer_contract_digest": typed_codex_observer_contract_digest(runtime),
        "runtime": runtime.to_data(),
        "neutral_panel_name": PANEL_OWNER_NEUTRAL_IMAGE_NAME,
        "axis_digests": [item.axis_digest for item in axes],
        "axis_count": len(axes),
        "model_calls_per_panel": 1,
        "max_batched_axes": MAX_BATCHED_AXES,
        "max_prompt_bytes": MAX_BATCHED_PROMPT_BYTES,
        "max_output_schema_bytes": MAX_BATCHED_OUTPUT_SCHEMA_BYTES,
        "max_response_bytes": MAX_BATCHED_RESPONSE_BYTES,
        "axis_subset_permitted": False,
        "selected_candidate_specs_model_visible": False,
        "native_task_orientation_model_visible": False,
        "support_or_query_role_model_visible": False,
        "frozen_formula_model_visible": False,
        "python_is_canonical_authority": True,
        "engineering_only": True,
        "scientific_calibration_supplied": False,
    }


def batched_typed_codex_observer_contract_digest(
    runtime: TypedCodexRuntimeBinding,
) -> str:
    return canonical_digest(batched_typed_codex_observer_contract_data(runtime))


@dataclass(frozen=True, slots=True)
class TypedBatchedAxisCodexArtifact:
    """One complete receipted call and its deterministic Python observation set."""

    panel_png_digest: str
    panel_png_byte_count: int
    runtime: TypedCodexRuntimeBinding
    panel_only_context: PanelOnlyObservationContext
    request: BatchedFeatureAxisRequest
    request_digest: str
    batch_contract_digest: str
    observer_contract_digest: str
    measurement_protocol_digest: str
    prompt_digest: str
    output_schema_digest: str
    payload_digest: str
    model_payload: Mapping[str, Any]
    codex_receipt: CodexReceipt
    observation_set: PanelFeatureObservationSet

    def __post_init__(self) -> None:
        _digest(self.panel_png_digest, "batched artifact panel digest")
        if type(self.panel_png_byte_count) is not int or self.panel_png_byte_count <= 0:
            raise PanelBatchedTypedCodexObserverError(
                "batched artifact panel byte count differs"
            )
        if type(self.runtime) is not TypedCodexRuntimeBinding:
            raise TypeError("batched artifact needs TypedCodexRuntimeBinding")
        if type(self.panel_only_context) is not PanelOnlyObservationContext:
            raise TypeError("batched artifact needs PanelOnlyObservationContext")
        if type(self.request) is not BatchedFeatureAxisRequest:
            raise TypeError("batched artifact needs BatchedFeatureAxisRequest")
        for label, value in (
            ("batched request digest", self.request_digest),
            ("batched contract digest", self.batch_contract_digest),
            ("batched observer contract digest", self.observer_contract_digest),
            ("batched measurement protocol digest", self.measurement_protocol_digest),
            ("batched prompt digest", self.prompt_digest),
            ("batched output schema digest", self.output_schema_digest),
            ("batched payload digest", self.payload_digest),
        ):
            _digest(value, label)
        if (
            self.panel_only_context.runtime != self.runtime
            or self.panel_only_context.panel_png_digest != self.panel_png_digest
            or self.panel_only_context.panel_png_byte_count != self.panel_png_byte_count
            or self.request.panel_only_context != self.panel_only_context
        ):
            raise PanelBatchedTypedCodexObserverError(
                "batched panel/context/runtime custody differs"
            )
        payload = _canonical_payload(self.model_payload, "batched model payload")
        object.__setattr__(self, "model_payload", payload)
        prompt = batched_feature_axis_prompt(self.request)
        schema = batched_feature_axis_output_schema(self.request)
        if (
            self.request_digest != self.request.request_digest
            or self.batch_contract_digest
            != batched_typed_codex_observer_contract_digest(self.runtime)
            or self.observer_contract_digest
            != typed_codex_observer_contract_digest(self.runtime)
            or self.measurement_protocol_digest
            != typed_measurement_protocol_digest(self.runtime)
            or self.prompt_digest != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or self.output_schema_digest != canonical_digest(schema)
            or self.payload_digest != canonical_digest(payload)
        ):
            raise PanelBatchedTypedCodexObserverError(
                "batched frozen invocation envelope differs"
            )
        _validate_receipt_binding(
            self.codex_receipt,
            runtime=self.runtime,
            prompt_digest=self.prompt_digest,
            output_schema_digest=self.output_schema_digest,
            payload_digest=self.payload_digest,
            presentation=(
                PrototypeImageIdentity(
                    PANEL_OWNER_NEUTRAL_IMAGE_NAME,
                    self.panel_png_byte_count,
                    self.panel_png_digest,
                ),
            ),
        )
        if type(self.observation_set) is not PanelFeatureObservationSet:
            raise TypeError("batched artifact needs PanelFeatureObservationSet")
        replayed = parse_batched_feature_axis_payload(
            self.request,
            payload,
            observer_contract_digest=self.observer_contract_digest,
            measurement_protocol_digest=self.measurement_protocol_digest,
            observation_receipt_digest=self.codex_receipt.receipt_digest,
        )
        if replayed != self.observation_set:
            raise PanelBatchedTypedCodexObserverError(
                "batched observation-set parser replay differs"
            )

    @property
    def artifact_digest(self) -> str:
        return canonical_digest(self.content_data())

    def content_data(self) -> dict[str, object]:
        return {
            "schema": BATCHED_AXIS_ARTIFACT_SCHEMA,
            "panel_png_digest": self.panel_png_digest,
            "panel_png_byte_count": self.panel_png_byte_count,
            "runtime": self.runtime.to_data(),
            "panel_only_context": self.panel_only_context.to_data(),
            "request": self.request.to_data(),
            "request_digest": self.request_digest,
            "batch_contract_digest": self.batch_contract_digest,
            "observer_contract_digest": self.observer_contract_digest,
            "measurement_protocol_digest": self.measurement_protocol_digest,
            "prompt_digest": self.prompt_digest,
            "output_schema_digest": self.output_schema_digest,
            "payload_digest": self.payload_digest,
            "model_payload": dict(self.model_payload),
            "codex_receipt": self.codex_receipt.to_dict(),
            "codex_receipt_digest": self.codex_receipt.receipt_digest,
            "observation_set": self.observation_set.to_data(),
            "observation_set_digest": self.observation_set.observation_set_digest,
            "axis_observation_digests": {
                item.alias: observation.observation_digest
                for item, observation in zip(
                    self.request.aliases,
                    self.observation_set.axis_observations,
                    strict=True,
                )
            },
            "model_call_count": 1,
            "model_visible_image_names": [PANEL_OWNER_NEUTRAL_IMAGE_NAME],
            "task_metadata_model_visible": False,
            "selected_candidate_specs_model_visible": False,
            "native_task_orientation_model_visible": False,
            "support_or_query_role_model_visible": False,
            "frozen_formula_model_visible": False,
            "python_is_canonical_authority": True,
            "engineering_only": True,
            "scientific_calibration_supplied": False,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, value: object) -> "TypedBatchedAxisCodexArtifact":
        raw = _fields(
            value,
            {
                "schema",
                "panel_png_digest",
                "panel_png_byte_count",
                "runtime",
                "panel_only_context",
                "request",
                "request_digest",
                "batch_contract_digest",
                "observer_contract_digest",
                "measurement_protocol_digest",
                "prompt_digest",
                "output_schema_digest",
                "payload_digest",
                "model_payload",
                "codex_receipt",
                "codex_receipt_digest",
                "observation_set",
                "observation_set_digest",
                "axis_observation_digests",
                "model_call_count",
                "model_visible_image_names",
                "task_metadata_model_visible",
                "selected_candidate_specs_model_visible",
                "native_task_orientation_model_visible",
                "support_or_query_role_model_visible",
                "frozen_formula_model_visible",
                "python_is_canonical_authority",
                "engineering_only",
                "scientific_calibration_supplied",
                "artifact_digest",
            },
            "batched typed Codex artifact",
        )
        if (
            raw["schema"] != BATCHED_AXIS_ARTIFACT_SCHEMA
            or raw["model_call_count"] != 1
            or raw["model_visible_image_names"] != [PANEL_OWNER_NEUTRAL_IMAGE_NAME]
            or raw["task_metadata_model_visible"] is not False
            or raw["selected_candidate_specs_model_visible"] is not False
            or raw["native_task_orientation_model_visible"] is not False
            or raw["support_or_query_role_model_visible"] is not False
            or raw["frozen_formula_model_visible"] is not False
            or raw["python_is_canonical_authority"] is not True
            or raw["engineering_only"] is not True
            or raw["scientific_calibration_supplied"] is not False
        ):
            raise PanelBatchedTypedCodexObserverError(
                "batched typed Codex artifact policy differs"
            )
        result = cls(
            panel_png_digest=raw["panel_png_digest"],
            panel_png_byte_count=raw["panel_png_byte_count"],
            runtime=TypedCodexRuntimeBinding.from_data(raw["runtime"]),
            panel_only_context=PanelOnlyObservationContext.from_data(
                raw["panel_only_context"]
            ),
            request=BatchedFeatureAxisRequest.from_data(raw["request"]),
            request_digest=raw["request_digest"],
            batch_contract_digest=raw["batch_contract_digest"],
            observer_contract_digest=raw["observer_contract_digest"],
            measurement_protocol_digest=raw["measurement_protocol_digest"],
            prompt_digest=raw["prompt_digest"],
            output_schema_digest=raw["output_schema_digest"],
            payload_digest=raw["payload_digest"],
            model_payload=_canonical_payload(
                raw["model_payload"], "archived batched model payload"
            ),
            codex_receipt=_receipt_from_data(raw["codex_receipt"]),
            observation_set=PanelFeatureObservationSet.from_data(
                raw["observation_set"]
            ),
        )
        if (
            raw["codex_receipt_digest"] != result.codex_receipt.receipt_digest
            or raw["observation_set_digest"]
            != result.observation_set.observation_set_digest
            or raw["axis_observation_digests"]
            != result.content_data()["axis_observation_digests"]
            or raw["artifact_digest"] != result.artifact_digest
            or result.to_data() != dict(raw)
        ):
            raise PanelBatchedTypedCodexObserverError(
                "batched typed Codex artifact digest differs"
            )
        return result


def observe_typed_panel_axes_batched(
    panel_png: bytes,
    *,
    axes: tuple[FeatureAxis, ...],
    panel_only_context: PanelOnlyObservationContext,
    model: str,
    reasoning_effort: str,
    minutes: int = 15,
    verbose: bool = False,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    expected_launcher_digest: str,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    transport=run_codex_named_images_structured,
) -> TypedBatchedAxisCodexArtifact:
    """Observe every whole-panel axis with exactly one neutral Codex call."""

    panel = _exact_png(panel_png)
    if type(panel_only_context) is not PanelOnlyObservationContext:
        raise TypeError("batched observation needs PanelOnlyObservationContext")
    if not callable(transport):
        raise TypeError("batched transport must be callable")
    runtime = _bind_runtime(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    panel_digest = hashlib.sha256(panel).hexdigest()
    if (
        panel_only_context.runtime != runtime
        or panel_only_context.panel_png_digest != panel_digest
        or panel_only_context.panel_png_byte_count != len(panel)
    ):
        raise PanelBatchedTypedCodexObserverError(
            "batched panel-only context belongs to another panel or runtime"
        )
    request = BatchedFeatureAxisRequest.build(panel_only_context, axes)
    prompt = batched_feature_axis_prompt(request)
    schema = batched_feature_axis_output_schema(request)
    observer_contract = typed_codex_observer_contract_digest(runtime)
    measurement_protocol = typed_measurement_protocol_digest(runtime)
    try:
        payload, receipt = _scene_runtime._stage_and_call(
            ((PANEL_OWNER_NEUTRAL_IMAGE_NAME, panel),),
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
        frozen = _canonical_payload(payload, "batched model payload")
        observations = parse_batched_feature_axis_payload(
            request,
            frozen,
            observer_contract_digest=observer_contract,
            measurement_protocol_digest=measurement_protocol,
            observation_receipt_digest=receipt.receipt_digest,
        )
        return TypedBatchedAxisCodexArtifact(
            panel_png_digest=panel_digest,
            panel_png_byte_count=len(panel),
            runtime=runtime,
            panel_only_context=panel_only_context,
            request=request,
            request_digest=request.request_digest,
            batch_contract_digest=batched_typed_codex_observer_contract_digest(runtime),
            observer_contract_digest=observer_contract,
            measurement_protocol_digest=measurement_protocol,
            prompt_digest=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            output_schema_digest=canonical_digest(schema),
            payload_digest=canonical_digest(frozen),
            model_payload=frozen,
            codex_receipt=receipt,
            observation_set=observations,
        )
    except PanelBatchedTypedCodexObserverError:
        raise
    except Exception as exc:
        raise PanelBatchedTypedCodexObserverError(
            "batched typed call failed closed; no observation artifact was produced"
        ) from exc


def verify_typed_batched_axis_codex_artifact(
    artifact: TypedBatchedAxisCodexArtifact,
    panel_png: bytes,
    *,
    expected_artifact_digest: str,
) -> TypedBatchedAxisCodexArtifact:
    """Cold replay exact pixels, request, full receipt, and every axis parser."""

    if type(artifact) is not TypedBatchedAxisCodexArtifact:
        raise TypeError("batched cold replay needs TypedBatchedAxisCodexArtifact")
    expected = _digest(expected_artifact_digest, "expected batched artifact digest")
    restored = TypedBatchedAxisCodexArtifact.from_data(artifact.to_data())
    if restored.artifact_digest != expected:
        raise PanelBatchedTypedCodexObserverError(
            "batched artifact differs from commitment"
        )
    panel = _exact_png(panel_png)
    if (
        restored.panel_png_digest != hashlib.sha256(panel).hexdigest()
        or restored.panel_png_byte_count != len(panel)
    ):
        raise PanelBatchedTypedCodexObserverError(
            "batched artifact panel bytes differ"
        )
    rebuilt = BatchedFeatureAxisRequest.build(
        restored.panel_only_context, restored.request.axes
    )
    if rebuilt != restored.request:
        raise PanelBatchedTypedCodexObserverError(
            "batched cold-replay request differs"
        )
    prompt = batched_feature_axis_prompt(rebuilt)
    schema = batched_feature_axis_output_schema(rebuilt)
    with tempfile.TemporaryDirectory(prefix="bongard-batched-axis-replay-") as raw:
        target = Path(raw) / PANEL_OWNER_NEUTRAL_IMAGE_NAME
        target.write_bytes(panel)
        try:
            validate_codex_named_image_receipt(
                restored.codex_receipt,
                prompt,
                (str(target.resolve()),),
                (PANEL_OWNER_NEUTRAL_IMAGE_NAME,),
                schema,
                dict(restored.model_payload),
            )
        except Exception as exc:
            raise PanelBatchedTypedCodexObserverError(
                "batched receipt cold replay failed"
            ) from exc
        if target.read_bytes() != panel:
            raise PanelBatchedTypedCodexObserverError(
                "batched cold-replay panel changed"
            )
    replayed = parse_batched_feature_axis_payload(
        rebuilt,
        dict(restored.model_payload),
        observer_contract_digest=restored.observer_contract_digest,
        measurement_protocol_digest=restored.measurement_protocol_digest,
        observation_receipt_digest=restored.codex_receipt.receipt_digest,
    )
    if replayed != restored.observation_set:
        raise PanelBatchedTypedCodexObserverError(
            "batched typed cold replay differs"
        )
    return restored


__all__ = (
    "BatchedFeatureAxisAlias",
    "BatchedFeatureAxisRequest",
    "MAX_BATCHED_AXES",
    "MAX_BATCHED_OUTPUT_SCHEMA_BYTES",
    "MAX_BATCHED_PROMPT_BYTES",
    "MAX_BATCHED_RESPONSE_BYTES",
    "PanelBatchedTypedCodexObserverError",
    "TypedBatchedAxisCodexArtifact",
    "batched_feature_axis_output_schema",
    "batched_feature_axis_prompt",
    "batched_typed_codex_observer_contract_data",
    "batched_typed_codex_observer_contract_digest",
    "complete_whole_panel_feature_axes",
    "observe_typed_panel_axes_batched",
    "panel_batched_typed_codex_observer_source_digest",
    "parse_batched_feature_axis_payload",
    "verify_typed_batched_axis_codex_artifact",
)

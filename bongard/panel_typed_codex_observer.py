"""Production no-tools Codex custody for typed panel vision.

This module is the byte/receipt boundary missing from the transport-agnostic
typed panel frontend.  It stages only exact PNG bytes under frozen neutral
names, uses the pinned no-tools named-image runtime, retains every
``CodexReceipt`` field, and produces content-addressed Python artifacts that
can be replayed cold without invoking a model.

There are three independent call shapes:

* one ``panel.png`` owner-inventory call for owner-local feature axes;
* one ``panel.png`` complete-axis call, either over that receipted inventory
  or over a no-enumeration whole-panel context; and
* one twelve-image contrastive proposer call implementing the existing
  :class:`PanelFeatureReceiptedCall` boundary.

All resulting observations are engineering-only and uncalibrated.  Transport
or parser failures raise and produce no truth-valued artifact.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from bongard import prototype_scene_observer as _scene_runtime
from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import CodexNoToolsAttestation
from bongard.panel_feature_observation import (
    FEATURE_OBSERVATION_PROTOCOL_ID,
    FeatureAxis,
    PanelAxisObservation,
)
from bongard.panel_feature_observer_protocol import (
    FEATURE_AXIS_VIEW_PROTOCOL_ID,
    MAX_BINDINGS_PER_AXIS_CALL,
    FeatureAxisObservationView,
    feature_axis_observer_output_schema,
    feature_axis_observer_prompt,
    parse_feature_axis_observer_payload,
)
from bongard.panel_feature_proposer import (
    PANEL_FEATURE_PRESENTATION_NAMES,
    PanelFeatureProposerCallResult,
    PanelFeatureProposerResult,
    invoke_panel_feature_proposer,
    panel_feature_proposer_contract_digest,
    panel_feature_proposer_output_schema,
    panel_feature_proposer_prompt,
)
from bongard.panel_owner_inventory import (
    PANEL_OWNER_NEUTRAL_IMAGE_NAME,
    InventoryStatus,
    InventoryTransportKind,
    PanelOwnerInventoryArtifact,
    bind_panel_owner_inventory_receipt,
    build_panel_owner_inventory_artifact,
    panel_owner_inventory_contract_digest,
    panel_owner_inventory_model_view,
    panel_owner_inventory_output_schema,
    panel_owner_inventory_prompt,
)
from bongard.panel_soft_ontology import (
    EnumerationResolution,
    NativeOrientation,
    OwnerInventory,
    SubjectScope,
    feature_catalog_digest,
)
from bongard.prototype_scene_observer import PrototypeImageIdentity
from bongard.transport import (
    CODEX_EFFECTIVE_TOOL_MODE,
    CODEX_ISOLATION_POLICY,
    CODEX_RECEIPT_SCHEMA,
    CODEX_TOOL_SURFACE_DIGEST,
    CODEX_TRANSPORT_POLICY_DIGEST,
    NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexProposerFailure,
    CodexReceipt,
    run_codex_named_images_structured,
    validate_codex_named_image_receipt,
    validate_codex_receipt,
    validate_codex_strict_output_schema,
)


TYPED_CODEX_RUNTIME_SCHEMA = "gkm.bongard-panel-typed-codex-runtime.v1"
TYPED_CODEX_OBSERVER_CONTRACT_SCHEMA = (
    "gkm.bongard-panel-typed-codex-observer-contract.v1"
)
TYPED_MEASUREMENT_PROTOCOL_SCHEMA = (
    "gkm.bongard-panel-typed-measurement-protocol.v1"
)
TYPED_PANEL_ONLY_CONTEXT_SCHEMA = "gkm.bongard-panel-only-context.v1"
TYPED_OWNER_CODEX_ARTIFACT_SCHEMA = "gkm.bongard-typed-owner-codex-artifact.v1"
TYPED_AXIS_CODEX_ARTIFACT_SCHEMA = "gkm.bongard-typed-axis-codex-artifact.v1"
TYPED_PROPOSER_CODEX_ARTIFACT_SCHEMA = (
    "gkm.bongard-typed-proposer-codex-call-artifact.v1"
)
TYPED_CODEX_OBSERVER_PROTOCOL_ID = (
    "bongard.panel-typed-codex-observer/neutral-raw-panel-v1"
)
TYPED_PANEL_ONLY_PROTOCOL_ID = (
    "bongard.panel-typed-codex-observer/whole-panel-no-enumeration-v1"
)

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


class PanelTypedCodexObserverError(ValueError):
    """A pinned runtime, receipted call, or cold replay is invalid."""


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise PanelTypedCodexObserverError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise PanelTypedCodexObserverError(f"{label} must be a lowercase SHA-256")
    return value


def _code(value: object, label: str) -> str:
    if type(value) is not str or _CODE.fullmatch(value) is None:
        raise PanelTypedCodexObserverError(f"{label} must be a bounded safe code")
    return value


def _byte_count(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise PanelTypedCodexObserverError(f"{label} must be a positive exact integer")
    return value


def _canonical_payload(value: object, label: str = "model payload") -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise PanelTypedCodexObserverError(f"{label} must be a JSON object")
    try:
        decoded = json.loads(canonical_json(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise PanelTypedCodexObserverError(
            f"{label} is not canonical finite JSON"
        ) from exc
    if type(decoded) is not dict:
        raise PanelTypedCodexObserverError(f"{label} must be a JSON object")
    return decoded


def _receipt_from_data(value: object) -> CodexReceipt:
    try:
        receipt = _scene_runtime._receipt_from_data(value)
    except Exception as exc:
        raise PanelTypedCodexObserverError("archived Codex receipt is invalid") from exc
    if type(receipt) is not CodexReceipt:
        raise PanelTypedCodexObserverError("archived Codex receipt type differs")
    return receipt


def _exact_png(value: object, label: str = "panel") -> bytes:
    try:
        result = _scene_runtime._validate_exact_png(value, label)
    except Exception as exc:
        raise PanelTypedCodexObserverError(f"{label} is not an exact bounded PNG") from exc
    if type(result) is not bytes:
        raise PanelTypedCodexObserverError(f"{label} PNG validator returned wrong type")
    return result


def panel_typed_codex_observer_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


@dataclass(frozen=True, slots=True)
class TypedCodexRuntimeBinding:
    """The exact precommitted no-tools environment shared by all typed calls."""

    model: str
    reasoning_effort: str
    model_request_digest: str
    expected_launcher_digest: str
    cloud_policy_cache_binding: str
    transport_policy_digest: str
    model_catalog_digest: str
    no_tools_attestation_digest: str

    def __post_init__(self) -> None:
        _code(self.model, "runtime model")
        _code(self.reasoning_effort, "runtime reasoning effort")
        for label, item in (
            ("runtime model request digest", self.model_request_digest),
            ("runtime launcher digest", self.expected_launcher_digest),
            ("runtime transport policy digest", self.transport_policy_digest),
            ("runtime model catalog digest", self.model_catalog_digest),
            ("runtime no-tools attestation digest", self.no_tools_attestation_digest),
        ):
            _digest(item, label)
        if self.cloud_policy_cache_binding != "absent" and (
            type(self.cloud_policy_cache_binding) is not str
            or _ADDRESS.fullmatch(self.cloud_policy_cache_binding) is None
        ):
            raise PanelTypedCodexObserverError("runtime cloud-policy binding differs")
        if self.model_request_digest != _scene_runtime.prototype_scene_observer_model_digest(
            self.model, self.reasoning_effort
        ):
            raise PanelTypedCodexObserverError("runtime model request digest differs")
        if self.transport_policy_digest != CODEX_TRANSPORT_POLICY_DIGEST:
            raise PanelTypedCodexObserverError("runtime transport policy differs")

    @property
    def runtime_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": TYPED_CODEX_RUNTIME_SCHEMA,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "model_request_digest": self.model_request_digest,
            "expected_launcher_digest": self.expected_launcher_digest,
            "cloud_policy_cache_binding": self.cloud_policy_cache_binding,
            "transport_policy_digest": self.transport_policy_digest,
            "model_catalog_digest": self.model_catalog_digest,
            "no_tools_attestation_digest": self.no_tools_attestation_digest,
            "receipt_schema": CODEX_RECEIPT_SCHEMA,
            "input_digest_schema": NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
            "isolation_policy": CODEX_ISOLATION_POLICY,
            "effective_tool_mode": CODEX_EFFECTIVE_TOOL_MODE,
            "tool_surface_digest": CODEX_TOOL_SURFACE_DIGEST,
        }

    @classmethod
    def from_data(cls, value: object) -> "TypedCodexRuntimeBinding":
        raw = _fields(
            value,
            {
                "schema",
                "model",
                "reasoning_effort",
                "model_request_digest",
                "expected_launcher_digest",
                "cloud_policy_cache_binding",
                "transport_policy_digest",
                "model_catalog_digest",
                "no_tools_attestation_digest",
                "receipt_schema",
                "input_digest_schema",
                "isolation_policy",
                "effective_tool_mode",
                "tool_surface_digest",
            },
            "typed Codex runtime",
        )
        if (
            raw["schema"] != TYPED_CODEX_RUNTIME_SCHEMA
            or raw["receipt_schema"] != CODEX_RECEIPT_SCHEMA
            or raw["input_digest_schema"] != NAMED_IMAGE_INPUT_DIGEST_SCHEMA
            or raw["isolation_policy"] != CODEX_ISOLATION_POLICY
            or raw["effective_tool_mode"] != CODEX_EFFECTIVE_TOOL_MODE
            or raw["tool_surface_digest"] != CODEX_TOOL_SURFACE_DIGEST
        ):
            raise PanelTypedCodexObserverError("typed Codex runtime policy differs")
        result = cls(
            raw["model"],
            raw["reasoning_effort"],
            raw["model_request_digest"],
            raw["expected_launcher_digest"],
            raw["cloud_policy_cache_binding"],
            raw["transport_policy_digest"],
            raw["model_catalog_digest"],
            raw["no_tools_attestation_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelTypedCodexObserverError("typed Codex runtime is not canonical")
        return result


def _bind_runtime(
    *,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
) -> TypedCodexRuntimeBinding:
    """Derive runtime identity from exact snapshots, never caller-supplied digests."""

    policy = _scene_runtime._policy_cache_binding(cloud_policy_cache_snapshot)
    try:
        catalog_digest, no_tools_digest = _scene_runtime._validate_no_tools_runtime(
            model_catalog_snapshot=model_catalog_snapshot,
            no_tools_attestation=no_tools_attestation,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=policy,
        )
    except Exception as exc:
        raise PanelTypedCodexObserverError("pinned no-tools runtime validation failed") from exc
    return TypedCodexRuntimeBinding(
        model,
        reasoning_effort,
        _scene_runtime.prototype_scene_observer_model_digest(model, reasoning_effort),
        expected_launcher_digest,
        policy,
        CODEX_TRANSPORT_POLICY_DIGEST,
        catalog_digest,
        no_tools_digest,
    )


def typed_codex_observer_contract_data(
    runtime: TypedCodexRuntimeBinding,
) -> dict[str, object]:
    """Return the Python-derived two-stage visual instrument contract."""

    if type(runtime) is not TypedCodexRuntimeBinding:
        raise TypeError("observer contract requires TypedCodexRuntimeBinding")
    owner_view = panel_owner_inventory_model_view()
    return {
        "schema": TYPED_CODEX_OBSERVER_CONTRACT_SCHEMA,
        "protocol_id": TYPED_CODEX_OBSERVER_PROTOCOL_ID,
        "adapter_source_digest": panel_typed_codex_observer_source_digest(),
        "runtime": runtime.to_data(),
        "neutral_panel_name": PANEL_OWNER_NEUTRAL_IMAGE_NAME,
        "owner_inventory_contract_digest": panel_owner_inventory_contract_digest(),
        "owner_view_digest": canonical_digest(owner_view),
        "owner_prompt_digest": hashlib.sha256(
            panel_owner_inventory_prompt().encode("utf-8")
        ).hexdigest(),
        "owner_output_schema_digest": canonical_digest(
            panel_owner_inventory_output_schema()
        ),
        "feature_axis_view_protocol_id": FEATURE_AXIS_VIEW_PROTOCOL_ID,
        "feature_observation_protocol_id": FEATURE_OBSERVATION_PROTOCOL_ID,
        "feature_catalog_digest": feature_catalog_digest(),
        "max_bindings_per_axis_call": MAX_BINDINGS_PER_AXIS_CALL,
        "python_is_canonical_authority": True,
        "engineering_only": True,
        "scientific_calibration_supplied": False,
    }


def typed_codex_observer_contract_digest(runtime: TypedCodexRuntimeBinding) -> str:
    return canonical_digest(typed_codex_observer_contract_data(runtime))


def typed_measurement_protocol_data(
    runtime: TypedCodexRuntimeBinding,
) -> dict[str, object]:
    """Return the closed-axis measurement protocol, derived only in Python."""

    return {
        "schema": TYPED_MEASUREMENT_PROTOCOL_SCHEMA,
        "observer_contract_digest": typed_codex_observer_contract_digest(runtime),
        "feature_axis_view_protocol_id": FEATURE_AXIS_VIEW_PROTOCOL_ID,
        "feature_observation_protocol_id": FEATURE_OBSERVATION_PROTOCOL_ID,
        "feature_catalog_digest": feature_catalog_digest(),
        "owner_inventory_contract_digest": panel_owner_inventory_contract_digest(),
        "complete_registered_axis_shown": True,
        "target_parameter_designated": False,
        "support_role_shown": False,
        "empty_complete_row_is_scientific_absence": False,
        "python_is_canonical_authority": True,
        "engineering_only": True,
        "scientific_calibration_supplied": False,
    }


def typed_measurement_protocol_digest(runtime: TypedCodexRuntimeBinding) -> str:
    return canonical_digest(typed_measurement_protocol_data(runtime))


def _panel_identity(panel: bytes) -> PrototypeImageIdentity:
    return PrototypeImageIdentity(
        PANEL_OWNER_NEUTRAL_IMAGE_NAME,
        len(panel),
        hashlib.sha256(panel).hexdigest(),
    )


def _validate_receipt_binding(
    receipt: CodexReceipt,
    *,
    runtime: TypedCodexRuntimeBinding,
    prompt_digest: str,
    output_schema_digest: str,
    payload_digest: str,
    presentation: Sequence[PrototypeImageIdentity],
) -> None:
    if type(receipt) is not CodexReceipt:
        raise PanelTypedCodexObserverError("call did not return a full CodexReceipt")
    try:
        validate_codex_receipt(receipt.to_dict())
    except (CodexProposerFailure, TypeError, ValueError) as exc:
        raise PanelTypedCodexObserverError("Codex receipt validation failed") from exc
    identities = [item.to_data() for item in presentation]
    image_view_digest = canonical_digest(identities)
    image_set_digest = "sha256:" + canonical_digest(
        {"schema": NAMED_IMAGE_INPUT_DIGEST_SCHEMA, "images": identities}
    )
    expected = {
        "requested_model": runtime.model,
        "requested_reasoning_effort": runtime.reasoning_effort,
        "codex_launcher_digest": runtime.expected_launcher_digest,
        "cloud_config_bundle_cache_binding": runtime.cloud_policy_cache_binding,
        "transport_policy_digest": runtime.transport_policy_digest,
        "model_catalog_digest": runtime.model_catalog_digest,
        "tool_surface_attestation_digest": runtime.no_tools_attestation_digest,
        "effective_tool_mode": CODEX_EFFECTIVE_TOOL_MODE,
        "tool_surface_digest": CODEX_TOOL_SURFACE_DIGEST,
        "isolation_policy": CODEX_ISOLATION_POLICY,
        "input_digest_schema": NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
        "task_digest": prompt_digest,
        "prompt_digest": prompt_digest,
        "output_schema_digest": output_schema_digest,
        "structured_output_digest": payload_digest,
        "panel_view_digest": image_view_digest,
        "panel_set_digest": image_set_digest,
    }
    for field, wanted in expected.items():
        if getattr(receipt, field) != wanted:
            raise PanelTypedCodexObserverError(f"Codex receipt {field} differs")


@dataclass(frozen=True, slots=True)
class PanelOnlyObservationContext:
    """Exact panel/runtime context for a whole-panel axis, with no owner call."""

    panel_png_digest: str
    panel_png_byte_count: int
    runtime: TypedCodexRuntimeBinding
    observer_contract_digest: str
    measurement_protocol_digest: str

    def __post_init__(self) -> None:
        _digest(self.panel_png_digest, "panel-only PNG digest")
        _byte_count(self.panel_png_byte_count, "panel-only PNG byte count")
        if type(self.runtime) is not TypedCodexRuntimeBinding:
            raise TypeError("panel-only context needs TypedCodexRuntimeBinding")
        if self.observer_contract_digest != typed_codex_observer_contract_digest(
            self.runtime
        ):
            raise PanelTypedCodexObserverError("panel-only observer contract differs")
        if self.measurement_protocol_digest != typed_measurement_protocol_digest(
            self.runtime
        ):
            raise PanelTypedCodexObserverError("panel-only measurement protocol differs")

    @property
    def context_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": TYPED_PANEL_ONLY_CONTEXT_SCHEMA,
            "protocol_id": TYPED_PANEL_ONLY_PROTOCOL_ID,
            "panel_png_digest": self.panel_png_digest,
            "panel_png_byte_count": self.panel_png_byte_count,
            "runtime": self.runtime.to_data(),
            "observer_contract_digest": self.observer_contract_digest,
            "measurement_protocol_digest": self.measurement_protocol_digest,
            "owner_enumeration_call_made": False,
            "enumeration_complete": False,
            "owners": [],
            "python_is_canonical_authority": True,
            "engineering_only": True,
            "scientific_calibration_supplied": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "PanelOnlyObservationContext":
        raw = _fields(
            value,
            {
                "schema",
                "protocol_id",
                "panel_png_digest",
                "panel_png_byte_count",
                "runtime",
                "observer_contract_digest",
                "measurement_protocol_digest",
                "owner_enumeration_call_made",
                "enumeration_complete",
                "owners",
                "python_is_canonical_authority",
                "engineering_only",
                "scientific_calibration_supplied",
            },
            "panel-only observation context",
        )
        if (
            raw["schema"] != TYPED_PANEL_ONLY_CONTEXT_SCHEMA
            or raw["protocol_id"] != TYPED_PANEL_ONLY_PROTOCOL_ID
            or raw["owner_enumeration_call_made"] is not False
            or raw["enumeration_complete"] is not False
            or raw["owners"] != []
            or raw["python_is_canonical_authority"] is not True
            or raw["engineering_only"] is not True
            or raw["scientific_calibration_supplied"] is not False
        ):
            raise PanelTypedCodexObserverError("panel-only context policy differs")
        result = cls(
            raw["panel_png_digest"],
            raw["panel_png_byte_count"],
            TypedCodexRuntimeBinding.from_data(raw["runtime"]),
            raw["observer_contract_digest"],
            raw["measurement_protocol_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelTypedCodexObserverError("panel-only context is not canonical")
        return result

    def to_owner_inventory(self) -> OwnerInventory:
        """Supply only the panel binding; never certify an owner enumeration."""

        protocol_digest = canonical_digest(
            {
                "schema": "gkm.bongard-panel-only-enumeration-placeholder.v1",
                "protocol_id": TYPED_PANEL_ONLY_PROTOCOL_ID,
                "observer_contract_digest": self.observer_contract_digest,
                "measurement_protocol_digest": self.measurement_protocol_digest,
                "owner_enumeration_call_made": False,
            }
        )
        return OwnerInventory(
            self.panel_png_digest,
            protocol_digest,
            EnumerationResolution.GRID16_FULL_PANEL,
            self.context_digest,
            False,
            (),
        )


def build_panel_only_observation_context(
    panel_png: bytes,
    *,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
) -> PanelOnlyObservationContext:
    panel = _exact_png(panel_png)
    runtime = _bind_runtime(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    return PanelOnlyObservationContext(
        hashlib.sha256(panel).hexdigest(),
        len(panel),
        runtime,
        typed_codex_observer_contract_digest(runtime),
        typed_measurement_protocol_digest(runtime),
    )


@dataclass(frozen=True, slots=True)
class TypedOwnerCodexArtifact:
    """One response-bound owner inventory plus the complete executed receipt."""

    panel_png_digest: str
    panel_png_byte_count: int
    runtime: TypedCodexRuntimeBinding
    observer_contract_digest: str
    measurement_protocol_digest: str
    owner_view_digest: str
    prompt_digest: str
    output_schema_digest: str
    payload_digest: str
    model_payload: Mapping[str, Any]
    codex_receipt: CodexReceipt
    inventory_artifact: PanelOwnerInventoryArtifact

    def __post_init__(self) -> None:
        _digest(self.panel_png_digest, "typed owner panel digest")
        _byte_count(self.panel_png_byte_count, "typed owner panel byte count")
        if type(self.runtime) is not TypedCodexRuntimeBinding:
            raise TypeError("typed owner artifact needs TypedCodexRuntimeBinding")
        for label, item in (
            ("typed owner contract digest", self.observer_contract_digest),
            ("typed owner measurement protocol digest", self.measurement_protocol_digest),
            ("typed owner view digest", self.owner_view_digest),
            ("typed owner prompt digest", self.prompt_digest),
            ("typed owner output schema digest", self.output_schema_digest),
            ("typed owner payload digest", self.payload_digest),
        ):
            _digest(item, label)
        payload = _canonical_payload(self.model_payload, "typed owner model payload")
        object.__setattr__(self, "model_payload", payload)
        expected_prompt = panel_owner_inventory_prompt()
        expected_schema = panel_owner_inventory_output_schema()
        expected_view = panel_owner_inventory_model_view()
        if (
            self.observer_contract_digest
            != typed_codex_observer_contract_digest(self.runtime)
            or self.measurement_protocol_digest
            != typed_measurement_protocol_digest(self.runtime)
            or self.owner_view_digest != canonical_digest(expected_view)
            or self.prompt_digest
            != hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest()
            or self.output_schema_digest != canonical_digest(expected_schema)
            or self.payload_digest != canonical_digest(payload)
        ):
            raise PanelTypedCodexObserverError("typed owner frozen envelope differs")
        presentation = (
            PrototypeImageIdentity(
                PANEL_OWNER_NEUTRAL_IMAGE_NAME,
                self.panel_png_byte_count,
                self.panel_png_digest,
            ),
        )
        _validate_receipt_binding(
            self.codex_receipt,
            runtime=self.runtime,
            prompt_digest=self.prompt_digest,
            output_schema_digest=self.output_schema_digest,
            payload_digest=self.payload_digest,
            presentation=presentation,
        )
        if type(self.inventory_artifact) is not PanelOwnerInventoryArtifact:
            raise TypeError("typed owner artifact needs PanelOwnerInventoryArtifact")
        bound = self.inventory_artifact
        if (
            bound.panel_png_digest != self.panel_png_digest
            or bound.panel_png_byte_count != self.panel_png_byte_count
            or bound.observer_contract_digest != self.observer_contract_digest
            or bound.receipt.transport_kind is not InventoryTransportKind.CODEX_NAMED_IMAGE
            or bound.receipt.model_id != self.runtime.model
            or bound.receipt.transport_receipt_digest
            != self.codex_receipt.receipt_digest
            or bound.receipt.response_digest != self.payload_digest
            or bound.response.to_data() != payload
        ):
            raise PanelTypedCodexObserverError(
                "typed owner response receipt and Codex receipt differ"
            )

    @property
    def artifact_digest(self) -> str:
        return canonical_digest(self.content_data())

    def content_data(self) -> dict[str, object]:
        return {
            "schema": TYPED_OWNER_CODEX_ARTIFACT_SCHEMA,
            "panel_png_digest": self.panel_png_digest,
            "panel_png_byte_count": self.panel_png_byte_count,
            "runtime": self.runtime.to_data(),
            "observer_contract_digest": self.observer_contract_digest,
            "measurement_protocol_digest": self.measurement_protocol_digest,
            "owner_view_digest": self.owner_view_digest,
            "prompt_digest": self.prompt_digest,
            "output_schema_digest": self.output_schema_digest,
            "payload_digest": self.payload_digest,
            "model_payload": dict(self.model_payload),
            "codex_receipt": self.codex_receipt.to_dict(),
            "codex_receipt_digest": self.codex_receipt.receipt_digest,
            "response_bound_inventory_receipt_digest": (
                self.inventory_artifact.receipt.receipt_digest
            ),
            "inventory_artifact": self.inventory_artifact.to_data(),
            "python_is_canonical_authority": True,
            "engineering_only": True,
            "scientific_calibration_supplied": False,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, value: object) -> "TypedOwnerCodexArtifact":
        raw = _fields(
            value,
            {
                "schema",
                "panel_png_digest",
                "panel_png_byte_count",
                "runtime",
                "observer_contract_digest",
                "measurement_protocol_digest",
                "owner_view_digest",
                "prompt_digest",
                "output_schema_digest",
                "payload_digest",
                "model_payload",
                "codex_receipt",
                "codex_receipt_digest",
                "response_bound_inventory_receipt_digest",
                "inventory_artifact",
                "python_is_canonical_authority",
                "engineering_only",
                "scientific_calibration_supplied",
                "artifact_digest",
            },
            "typed owner Codex artifact",
        )
        if (
            raw["schema"] != TYPED_OWNER_CODEX_ARTIFACT_SCHEMA
            or raw["python_is_canonical_authority"] is not True
            or raw["engineering_only"] is not True
            or raw["scientific_calibration_supplied"] is not False
        ):
            raise PanelTypedCodexObserverError("typed owner artifact policy differs")
        result = cls(
            raw["panel_png_digest"],
            raw["panel_png_byte_count"],
            TypedCodexRuntimeBinding.from_data(raw["runtime"]),
            raw["observer_contract_digest"],
            raw["measurement_protocol_digest"],
            raw["owner_view_digest"],
            raw["prompt_digest"],
            raw["output_schema_digest"],
            raw["payload_digest"],
            _canonical_payload(raw["model_payload"], "archived owner payload"),
            _receipt_from_data(raw["codex_receipt"]),
            PanelOwnerInventoryArtifact.from_data(raw["inventory_artifact"]),
        )
        if (
            raw["codex_receipt_digest"] != result.codex_receipt.receipt_digest
            or raw["response_bound_inventory_receipt_digest"]
            != result.inventory_artifact.receipt.receipt_digest
            or raw["artifact_digest"] != result.artifact_digest
            or result.to_data() != dict(raw)
        ):
            raise PanelTypedCodexObserverError("typed owner artifact digest differs")
        return result

    def to_owner_inventory(self) -> OwnerInventory:
        if self.inventory_artifact.status is not InventoryStatus.COMPLETE:
            raise PanelTypedCodexObserverError(
                "owner-local feature observation requires a complete inventory"
            )
        return self.inventory_artifact.to_owner_inventory()


def observe_typed_panel_owners(
    panel_png: bytes,
    *,
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
) -> TypedOwnerCodexArtifact:
    """Run one neutral owner enumeration and close its complete receipt chain."""

    panel = _exact_png(panel_png)
    if not callable(transport):
        raise TypeError("typed owner transport must be callable")
    runtime = _bind_runtime(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    observer_contract = typed_codex_observer_contract_digest(runtime)
    measurement_protocol = typed_measurement_protocol_digest(runtime)
    prompt = panel_owner_inventory_prompt()
    schema = panel_owner_inventory_output_schema()
    view_digest = canonical_digest(panel_owner_inventory_model_view())
    try:
        payload, codex_receipt = _scene_runtime._stage_and_call(
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
        frozen_payload = _canonical_payload(payload, "owner model payload")
        response_receipt = bind_panel_owner_inventory_receipt(
            panel_png=panel,
            observer_contract_digest=observer_contract,
            payload=frozen_payload,
            transport_kind=InventoryTransportKind.CODEX_NAMED_IMAGE,
            model_id=model,
            transport_receipt_digest=codex_receipt.receipt_digest,
        )
        inventory_artifact = build_panel_owner_inventory_artifact(
            panel_png=panel,
            observer_contract_digest=observer_contract,
            payload=frozen_payload,
            receipt=response_receipt,
        )
        return TypedOwnerCodexArtifact(
            hashlib.sha256(panel).hexdigest(),
            len(panel),
            runtime,
            observer_contract,
            measurement_protocol,
            view_digest,
            hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            canonical_digest(schema),
            canonical_digest(frozen_payload),
            frozen_payload,
            codex_receipt,
            inventory_artifact,
        )
    except PanelTypedCodexObserverError:
        raise
    except Exception as exc:
        raise PanelTypedCodexObserverError(
            "typed owner call failed closed; no inventory artifact was produced"
        ) from exc


def verify_typed_owner_codex_artifact(
    artifact: TypedOwnerCodexArtifact,
    panel_png: bytes,
    *,
    expected_artifact_digest: str,
) -> TypedOwnerCodexArtifact:
    """Cold replay an owner call against exact supplied PNG bytes."""

    if type(artifact) is not TypedOwnerCodexArtifact:
        raise TypeError("owner cold replay requires TypedOwnerCodexArtifact")
    expected = _digest(expected_artifact_digest, "expected owner artifact digest")
    restored = TypedOwnerCodexArtifact.from_data(artifact.to_data())
    if restored.artifact_digest != expected:
        raise PanelTypedCodexObserverError("owner artifact differs from commitment")
    panel = _exact_png(panel_png)
    if (
        restored.panel_png_digest != hashlib.sha256(panel).hexdigest()
        or restored.panel_png_byte_count != len(panel)
    ):
        raise PanelTypedCodexObserverError("owner artifact panel bytes differ")
    prompt = panel_owner_inventory_prompt()
    schema = panel_owner_inventory_output_schema()
    with tempfile.TemporaryDirectory(prefix="bongard-typed-owner-replay-") as raw:
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
            raise PanelTypedCodexObserverError("owner receipt cold replay failed") from exc
        if target.read_bytes() != panel:
            raise PanelTypedCodexObserverError("owner cold-replay panel changed")
    replayed_receipt = bind_panel_owner_inventory_receipt(
        panel_png=panel,
        observer_contract_digest=restored.observer_contract_digest,
        payload=dict(restored.model_payload),
        transport_kind=InventoryTransportKind.CODEX_NAMED_IMAGE,
        model_id=restored.runtime.model,
        transport_receipt_digest=restored.codex_receipt.receipt_digest,
    )
    replayed_inventory = build_panel_owner_inventory_artifact(
        panel_png=panel,
        observer_contract_digest=restored.observer_contract_digest,
        payload=dict(restored.model_payload),
        receipt=replayed_receipt,
    )
    if replayed_inventory != restored.inventory_artifact:
        raise PanelTypedCodexObserverError("owner typed replay differs")
    return restored


@dataclass(frozen=True, slots=True)
class TypedAxisCodexArtifact:
    """One exact complete-axis call and its Python-replayed observation."""

    panel_png_digest: str
    panel_png_byte_count: int
    runtime: TypedCodexRuntimeBinding
    source_kind: str
    source_artifact_digest: str
    panel_only_context: PanelOnlyObservationContext | None
    view: FeatureAxisObservationView
    view_digest: str
    observer_contract_digest: str
    measurement_protocol_digest: str
    prompt_digest: str
    output_schema_digest: str
    payload_digest: str
    model_payload: Mapping[str, Any]
    codex_receipt: CodexReceipt
    row_receipt_digests: tuple[str, ...]
    observation: PanelAxisObservation

    def __post_init__(self) -> None:
        _digest(self.panel_png_digest, "typed axis panel digest")
        _byte_count(self.panel_png_byte_count, "typed axis panel byte count")
        if type(self.runtime) is not TypedCodexRuntimeBinding:
            raise TypeError("typed axis artifact needs TypedCodexRuntimeBinding")
        if self.source_kind not in {"receipted_owner_inventory", "panel_only"}:
            raise PanelTypedCodexObserverError("typed axis source kind differs")
        _digest(self.source_artifact_digest, "typed axis source artifact digest")
        if type(self.view) is not FeatureAxisObservationView:
            raise TypeError("typed axis artifact needs FeatureAxisObservationView")
        canonical_view = FeatureAxisObservationView.build(
            self.view.inventory, self.view.axis
        )
        if canonical_view != self.view:
            raise PanelTypedCodexObserverError("typed axis exact view differs")
        if self.source_kind == "panel_only":
            if type(self.panel_only_context) is not PanelOnlyObservationContext:
                raise PanelTypedCodexObserverError(
                    "whole-panel axis needs its exact panel-only context"
                )
            context = self.panel_only_context
            if (
                self.source_artifact_digest != context.context_digest
                or self.view.axis.subject_scope is not SubjectScope.WHOLE_PANEL
                or self.view.inventory != context.to_owner_inventory()
                or context.runtime != self.runtime
            ):
                raise PanelTypedCodexObserverError("panel-only axis source differs")
        else:
            if self.panel_only_context is not None:
                raise PanelTypedCodexObserverError(
                    "owner-local axis cannot carry a panel-only context"
                )
            if (
                self.view.axis.subject_scope is SubjectScope.WHOLE_PANEL
                or not self.view.inventory.enumeration_complete
            ):
                raise PanelTypedCodexObserverError(
                    "owner-local axis needs a complete owner inventory"
                )
        if (
            self.view.inventory.panel_digest != self.panel_png_digest
            or (
                self.panel_only_context is not None
                and self.panel_only_context.panel_png_byte_count
                != self.panel_png_byte_count
            )
        ):
            raise PanelTypedCodexObserverError("typed axis panel custody differs")
        for label, item in (
            ("typed axis view digest", self.view_digest),
            ("typed axis observer contract digest", self.observer_contract_digest),
            (
                "typed axis measurement protocol digest",
                self.measurement_protocol_digest,
            ),
            ("typed axis prompt digest", self.prompt_digest),
            ("typed axis output schema digest", self.output_schema_digest),
            ("typed axis payload digest", self.payload_digest),
        ):
            _digest(item, label)
        payload = _canonical_payload(self.model_payload, "typed axis model payload")
        object.__setattr__(self, "model_payload", payload)
        prompt = feature_axis_observer_prompt(self.view)
        schema = feature_axis_observer_output_schema(self.view)
        validate_codex_strict_output_schema(schema)
        if (
            self.view_digest != self.view.view_digest
            or self.observer_contract_digest
            != typed_codex_observer_contract_digest(self.runtime)
            or self.measurement_protocol_digest
            != typed_measurement_protocol_digest(self.runtime)
            or self.prompt_digest != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or self.output_schema_digest != canonical_digest(schema)
            or self.payload_digest != canonical_digest(payload)
        ):
            raise PanelTypedCodexObserverError("typed axis frozen envelope differs")
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
        if type(self.row_receipt_digests) is not tuple:
            raise TypeError("typed axis row receipt digests must be a tuple")
        for index, item in enumerate(self.row_receipt_digests):
            _digest(item, f"typed axis row receipt digest {index}")
        if type(self.observation) is not PanelAxisObservation:
            raise TypeError("typed axis artifact needs PanelAxisObservation")
        row_digests = tuple(
            item.observation_receipt_digest
            for item in self.observation.binding_observations
        )
        if (
            self.observation.inventory != self.view.inventory
            or self.observation.axis != self.view.axis
            or self.observation.observer_contract_digest
            != self.observer_contract_digest
            or self.observation.measurement_protocol_digest
            != self.measurement_protocol_digest
            or self.row_receipt_digests != row_digests
            or any(item != self.codex_receipt.receipt_digest for item in row_digests)
        ):
            raise PanelTypedCodexObserverError(
                "typed axis observation/row receipt custody differs"
            )
        replayed = parse_feature_axis_observer_payload(
            self.view,
            payload,
            observer_contract_digest=self.observer_contract_digest,
            measurement_protocol_digest=self.measurement_protocol_digest,
            observation_receipt_digest=self.codex_receipt.receipt_digest,
        )
        if replayed != self.observation:
            raise PanelTypedCodexObserverError("typed axis parser replay differs")

    @property
    def artifact_digest(self) -> str:
        return canonical_digest(self.content_data())

    def content_data(self) -> dict[str, object]:
        return {
            "schema": TYPED_AXIS_CODEX_ARTIFACT_SCHEMA,
            "panel_png_digest": self.panel_png_digest,
            "panel_png_byte_count": self.panel_png_byte_count,
            "runtime": self.runtime.to_data(),
            "source_kind": self.source_kind,
            "source_artifact_digest": self.source_artifact_digest,
            "panel_only_context": (
                None
                if self.panel_only_context is None
                else self.panel_only_context.to_data()
            ),
            "view": self.view.to_data(),
            "view_digest": self.view_digest,
            "observer_contract_digest": self.observer_contract_digest,
            "measurement_protocol_digest": self.measurement_protocol_digest,
            "prompt_digest": self.prompt_digest,
            "output_schema_digest": self.output_schema_digest,
            "payload_digest": self.payload_digest,
            "model_payload": dict(self.model_payload),
            "codex_receipt": self.codex_receipt.to_dict(),
            "codex_receipt_digest": self.codex_receipt.receipt_digest,
            "row_receipt_digests": list(self.row_receipt_digests),
            "observation": self.observation.to_data(),
            "observation_digest": self.observation.observation_digest,
            "python_is_canonical_authority": True,
            "engineering_only": True,
            "scientific_calibration_supplied": False,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, value: object) -> "TypedAxisCodexArtifact":
        raw = _fields(
            value,
            {
                "schema",
                "panel_png_digest",
                "panel_png_byte_count",
                "runtime",
                "source_kind",
                "source_artifact_digest",
                "panel_only_context",
                "view",
                "view_digest",
                "observer_contract_digest",
                "measurement_protocol_digest",
                "prompt_digest",
                "output_schema_digest",
                "payload_digest",
                "model_payload",
                "codex_receipt",
                "codex_receipt_digest",
                "row_receipt_digests",
                "observation",
                "observation_digest",
                "python_is_canonical_authority",
                "engineering_only",
                "scientific_calibration_supplied",
                "artifact_digest",
            },
            "typed axis Codex artifact",
        )
        if (
            raw["schema"] != TYPED_AXIS_CODEX_ARTIFACT_SCHEMA
            or raw["python_is_canonical_authority"] is not True
            or raw["engineering_only"] is not True
            or raw["scientific_calibration_supplied"] is not False
            or type(raw["row_receipt_digests"]) is not list
        ):
            raise PanelTypedCodexObserverError("typed axis artifact policy differs")
        result = cls(
            raw["panel_png_digest"],
            raw["panel_png_byte_count"],
            TypedCodexRuntimeBinding.from_data(raw["runtime"]),
            raw["source_kind"],
            raw["source_artifact_digest"],
            (
                None
                if raw["panel_only_context"] is None
                else PanelOnlyObservationContext.from_data(raw["panel_only_context"])
            ),
            FeatureAxisObservationView.from_data(raw["view"]),
            raw["view_digest"],
            raw["observer_contract_digest"],
            raw["measurement_protocol_digest"],
            raw["prompt_digest"],
            raw["output_schema_digest"],
            raw["payload_digest"],
            _canonical_payload(raw["model_payload"], "archived axis payload"),
            _receipt_from_data(raw["codex_receipt"]),
            tuple(raw["row_receipt_digests"]),
            PanelAxisObservation.from_data(raw["observation"]),
        )
        if (
            raw["codex_receipt_digest"] != result.codex_receipt.receipt_digest
            or raw["observation_digest"] != result.observation.observation_digest
            or raw["artifact_digest"] != result.artifact_digest
            or result.to_data() != dict(raw)
        ):
            raise PanelTypedCodexObserverError("typed axis artifact digest differs")
        return result


def observe_typed_panel_axis(
    panel_png: bytes,
    *,
    axis: FeatureAxis,
    owner_artifact: TypedOwnerCodexArtifact | None = None,
    panel_only_context: PanelOnlyObservationContext | None = None,
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
) -> TypedAxisCodexArtifact:
    """Run one complete axis over a receipted owner source or whole panel."""

    panel = _exact_png(panel_png)
    if type(axis) is not FeatureAxis:
        raise TypeError("typed axis call requires FeatureAxis")
    if not callable(transport):
        raise TypeError("typed axis transport must be callable")
    runtime = _bind_runtime(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    if axis.subject_scope is SubjectScope.WHOLE_PANEL:
        if type(panel_only_context) is not PanelOnlyObservationContext or (
            owner_artifact is not None
        ):
            raise PanelTypedCodexObserverError(
                "whole-panel axis requires exactly one panel-only context"
            )
        source_kind = "panel_only"
        source_digest = panel_only_context.context_digest
        inventory = panel_only_context.to_owner_inventory()
        if panel_only_context.runtime != runtime:
            raise PanelTypedCodexObserverError("panel-only context runtime differs")
    else:
        if type(owner_artifact) is not TypedOwnerCodexArtifact or (
            panel_only_context is not None
        ):
            raise PanelTypedCodexObserverError(
                "owner-local axis requires exactly one receipted owner artifact"
            )
        source_kind = "receipted_owner_inventory"
        source_digest = owner_artifact.artifact_digest
        inventory = owner_artifact.to_owner_inventory()
        if owner_artifact.runtime != runtime:
            raise PanelTypedCodexObserverError("owner artifact runtime differs")
    panel_digest = hashlib.sha256(panel).hexdigest()
    if (
        inventory.panel_digest != panel_digest
        or (
            owner_artifact is not None
            and owner_artifact.panel_png_byte_count != len(panel)
        )
        or (
            panel_only_context is not None
            and panel_only_context.panel_png_byte_count != len(panel)
        )
    ):
        raise PanelTypedCodexObserverError("axis source belongs to another panel")
    view = FeatureAxisObservationView.build(inventory, axis)
    prompt = feature_axis_observer_prompt(view)
    schema = feature_axis_observer_output_schema(view)
    validate_codex_strict_output_schema(schema)
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
        frozen_payload = _canonical_payload(payload, "axis model payload")
        observation = parse_feature_axis_observer_payload(
            view,
            frozen_payload,
            observer_contract_digest=observer_contract,
            measurement_protocol_digest=measurement_protocol,
            observation_receipt_digest=receipt.receipt_digest,
        )
        row_receipts = tuple(
            item.observation_receipt_digest
            for item in observation.binding_observations
        )
        return TypedAxisCodexArtifact(
            panel_digest,
            len(panel),
            runtime,
            source_kind,
            source_digest,
            panel_only_context,
            view,
            view.view_digest,
            observer_contract,
            measurement_protocol,
            hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            canonical_digest(schema),
            canonical_digest(frozen_payload),
            frozen_payload,
            receipt,
            row_receipts,
            observation,
        )
    except PanelTypedCodexObserverError:
        raise
    except Exception as exc:
        raise PanelTypedCodexObserverError(
            "typed axis call failed closed; no observation artifact was produced"
        ) from exc


def verify_typed_axis_codex_artifact(
    artifact: TypedAxisCodexArtifact,
    panel_png: bytes,
    *,
    expected_artifact_digest: str,
    owner_artifact: TypedOwnerCodexArtifact | None = None,
) -> TypedAxisCodexArtifact:
    """Cold replay pixels, source inventory, view, parser, and full receipt."""

    if type(artifact) is not TypedAxisCodexArtifact:
        raise TypeError("axis cold replay requires TypedAxisCodexArtifact")
    expected = _digest(expected_artifact_digest, "expected axis artifact digest")
    restored = TypedAxisCodexArtifact.from_data(artifact.to_data())
    if restored.artifact_digest != expected:
        raise PanelTypedCodexObserverError("axis artifact differs from commitment")
    panel = _exact_png(panel_png)
    if (
        restored.panel_png_digest != hashlib.sha256(panel).hexdigest()
        or restored.panel_png_byte_count != len(panel)
    ):
        raise PanelTypedCodexObserverError("axis artifact panel bytes differ")
    if restored.source_kind == "receipted_owner_inventory":
        if type(owner_artifact) is not TypedOwnerCodexArtifact:
            raise PanelTypedCodexObserverError(
                "owner-local axis replay needs its owner artifact"
            )
        verified_owner = verify_typed_owner_codex_artifact(
            owner_artifact,
            panel,
            expected_artifact_digest=restored.source_artifact_digest,
        )
        inventory = verified_owner.to_owner_inventory()
    else:
        if owner_artifact is not None or restored.panel_only_context is None:
            raise PanelTypedCodexObserverError("panel-only axis replay source differs")
        inventory = restored.panel_only_context.to_owner_inventory()
    rebuilt_view = FeatureAxisObservationView.build(inventory, restored.view.axis)
    if rebuilt_view != restored.view:
        raise PanelTypedCodexObserverError("axis cold-replay view differs")
    prompt = feature_axis_observer_prompt(rebuilt_view)
    schema = feature_axis_observer_output_schema(rebuilt_view)
    validate_codex_strict_output_schema(schema)
    with tempfile.TemporaryDirectory(prefix="bongard-typed-axis-replay-") as raw:
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
            raise PanelTypedCodexObserverError("axis receipt cold replay failed") from exc
        if target.read_bytes() != panel:
            raise PanelTypedCodexObserverError("axis cold-replay panel changed")
    replayed = parse_feature_axis_observer_payload(
        rebuilt_view,
        dict(restored.model_payload),
        observer_contract_digest=restored.observer_contract_digest,
        measurement_protocol_digest=restored.measurement_protocol_digest,
        observation_receipt_digest=restored.codex_receipt.receipt_digest,
    )
    if replayed != restored.observation:
        raise PanelTypedCodexObserverError("axis typed cold replay differs")
    return restored


def _orientation_manifest_data(
    block_orientations: tuple[NativeOrientation, NativeOrientation],
) -> dict[str, object]:
    if (
        type(block_orientations) is not tuple
        or len(block_orientations) != 2
        or any(type(item) is not NativeOrientation for item in block_orientations)
        or set(block_orientations) != set(NativeOrientation)
    ):
        raise PanelTypedCodexObserverError(
            "proposer block orientations must be the exact semantic-side permutation"
        )
    return {
        "schema": "gkm.bongard-panel-feature-block-orientation-manifest.v1",
        "blocks": [
            {"block": "block_a", "native_orientation": block_orientations[0].value},
            {"block": "block_b", "native_orientation": block_orientations[1].value},
        ],
        "model_visible": False,
        "derived_by_production_invoker": True,
    }


def typed_proposer_transport_contract_data(
    runtime: TypedCodexRuntimeBinding,
    *,
    task_context_digest: str,
    block_orientations: tuple[NativeOrientation, NativeOrientation],
) -> dict[str, object]:
    _digest(task_context_digest, "proposer task context digest")
    manifest = _orientation_manifest_data(block_orientations)
    return {
        "schema": "gkm.bongard-panel-feature-proposer-transport-contract.v1",
        "adapter_source_digest": panel_typed_codex_observer_source_digest(),
        "core_proposer_contract_digest": panel_feature_proposer_contract_digest(),
        "runtime": runtime.to_data(),
        "task_context_digest": task_context_digest,
        "block_orientation_manifest": manifest,
        "block_orientation_manifest_digest": canonical_digest(manifest),
        "presentation_names": list(PANEL_FEATURE_PRESENTATION_NAMES),
        "python_is_canonical_authority": True,
        "narration_executable": False,
    }


def typed_proposer_transport_contract_digest(
    runtime: TypedCodexRuntimeBinding,
    *,
    task_context_digest: str,
    block_orientations: tuple[NativeOrientation, NativeOrientation],
) -> str:
    return canonical_digest(
        typed_proposer_transport_contract_data(
            runtime,
            task_context_digest=task_context_digest,
            block_orientations=block_orientations,
        )
    )


def _proposer_presentation_digest(
    presentation: Sequence[PrototypeImageIdentity],
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-panel-feature-proposer-presentation.v1",
            "images": [
                {"name": item.name, "sha256": item.content_digest}
                for item in presentation
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class TypedProposerCodexCallArtifact:
    """Full response-bound custody for one twelve-panel nomination call."""

    runtime: TypedCodexRuntimeBinding
    task_context_digest: str
    block_orientations: tuple[NativeOrientation, NativeOrientation]
    block_orientation_manifest_digest: str
    transport_contract_digest: str
    presentation: tuple[PrototypeImageIdentity, ...]
    presentation_digest: str
    prompt_digest: str
    output_schema_digest: str
    payload_digest: str
    model_payload: Mapping[str, Any]
    codex_receipt: CodexReceipt

    def __post_init__(self) -> None:
        if type(self.runtime) is not TypedCodexRuntimeBinding:
            raise TypeError("typed proposer artifact needs TypedCodexRuntimeBinding")
        _digest(self.task_context_digest, "typed proposer task context digest")
        manifest = _orientation_manifest_data(self.block_orientations)
        for label, item in (
            (
                "typed proposer orientation manifest digest",
                self.block_orientation_manifest_digest,
            ),
            ("typed proposer transport contract digest", self.transport_contract_digest),
            ("typed proposer presentation digest", self.presentation_digest),
            ("typed proposer prompt digest", self.prompt_digest),
            ("typed proposer output schema digest", self.output_schema_digest),
            ("typed proposer payload digest", self.payload_digest),
        ):
            _digest(item, label)
        if (
            type(self.presentation) is not tuple
            or any(type(item) is not PrototypeImageIdentity for item in self.presentation)
            or tuple(item.name for item in self.presentation)
            != PANEL_FEATURE_PRESENTATION_NAMES
        ):
            raise PanelTypedCodexObserverError(
                "typed proposer exact presentation differs"
            )
        payload = _canonical_payload(self.model_payload, "typed proposer model payload")
        object.__setattr__(self, "model_payload", payload)
        prompt = panel_feature_proposer_prompt()
        schema = panel_feature_proposer_output_schema()
        if (
            self.block_orientation_manifest_digest != canonical_digest(manifest)
            or self.transport_contract_digest
            != typed_proposer_transport_contract_digest(
                self.runtime,
                task_context_digest=self.task_context_digest,
                block_orientations=self.block_orientations,
            )
            or self.presentation_digest
            != _proposer_presentation_digest(self.presentation)
            or self.prompt_digest != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or self.output_schema_digest != canonical_digest(schema)
            or self.payload_digest != canonical_digest(payload)
        ):
            raise PanelTypedCodexObserverError("typed proposer frozen envelope differs")
        _validate_receipt_binding(
            self.codex_receipt,
            runtime=self.runtime,
            prompt_digest=self.prompt_digest,
            output_schema_digest=self.output_schema_digest,
            payload_digest=self.payload_digest,
            presentation=self.presentation,
        )

    @property
    def artifact_digest(self) -> str:
        return canonical_digest(self.content_data())

    def content_data(self) -> dict[str, object]:
        manifest = _orientation_manifest_data(self.block_orientations)
        return {
            "schema": TYPED_PROPOSER_CODEX_ARTIFACT_SCHEMA,
            "runtime": self.runtime.to_data(),
            "task_context_digest": self.task_context_digest,
            "block_orientation_manifest": manifest,
            "block_orientation_manifest_digest": self.block_orientation_manifest_digest,
            "transport_contract_digest": self.transport_contract_digest,
            "presentation": [item.to_data() for item in self.presentation],
            "presentation_digest": self.presentation_digest,
            "prompt_digest": self.prompt_digest,
            "output_schema_digest": self.output_schema_digest,
            "payload_digest": self.payload_digest,
            "model_payload": dict(self.model_payload),
            "codex_receipt": self.codex_receipt.to_dict(),
            "codex_receipt_digest": self.codex_receipt.receipt_digest,
            "python_is_canonical_authority": True,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, value: object) -> "TypedProposerCodexCallArtifact":
        raw = _fields(
            value,
            {
                "schema",
                "runtime",
                "task_context_digest",
                "block_orientation_manifest",
                "block_orientation_manifest_digest",
                "transport_contract_digest",
                "presentation",
                "presentation_digest",
                "prompt_digest",
                "output_schema_digest",
                "payload_digest",
                "model_payload",
                "codex_receipt",
                "codex_receipt_digest",
                "python_is_canonical_authority",
                "artifact_digest",
            },
            "typed proposer Codex artifact",
        )
        if (
            raw["schema"] != TYPED_PROPOSER_CODEX_ARTIFACT_SCHEMA
            or raw["python_is_canonical_authority"] is not True
            or type(raw["presentation"]) is not list
        ):
            raise PanelTypedCodexObserverError("typed proposer artifact policy differs")
        manifest = _fields(
            raw["block_orientation_manifest"],
            {"schema", "blocks", "model_visible", "derived_by_production_invoker"},
            "block orientation manifest",
        )
        blocks = manifest["blocks"]
        if type(blocks) is not list or len(blocks) != 2:
            raise PanelTypedCodexObserverError("block orientation manifest differs")
        parsed_orientations: list[NativeOrientation] = []
        for index, (expected_block, row) in enumerate(
            zip(("block_a", "block_b"), blocks, strict=True)
        ):
            item = _fields(
                row,
                {"block", "native_orientation"},
                f"block orientation manifest row {index}",
            )
            if item["block"] != expected_block:
                raise PanelTypedCodexObserverError("block orientation order differs")
            try:
                parsed_orientations.append(NativeOrientation(item["native_orientation"]))
            except (TypeError, ValueError) as exc:
                raise PanelTypedCodexObserverError(
                    "block orientation value differs"
                ) from exc
        orientations = tuple(parsed_orientations)
        if len(orientations) != 2:
            raise PanelTypedCodexObserverError("block orientation arity differs")
        result = cls(
            TypedCodexRuntimeBinding.from_data(raw["runtime"]),
            raw["task_context_digest"],
            orientations,  # type: ignore[arg-type]
            raw["block_orientation_manifest_digest"],
            raw["transport_contract_digest"],
            tuple(PrototypeImageIdentity.from_data(item) for item in raw["presentation"]),
            raw["presentation_digest"],
            raw["prompt_digest"],
            raw["output_schema_digest"],
            raw["payload_digest"],
            _canonical_payload(raw["model_payload"], "archived proposer payload"),
            _receipt_from_data(raw["codex_receipt"]),
        )
        if (
            dict(manifest) != _orientation_manifest_data(result.block_orientations)
            or raw["codex_receipt_digest"] != result.codex_receipt.receipt_digest
            or raw["artifact_digest"] != result.artifact_digest
            or result.to_data() != dict(raw)
        ):
            raise PanelTypedCodexObserverError("typed proposer artifact digest differs")
        return result


class HeadlessCodexPanelFeatureReceiptedCall:
    """One-shot production implementation of ``PanelFeatureReceiptedCall``.

    The returned legacy-sized envelope uses the digest of the full artifact as
    ``external_receipt_digest``.  Consequently proposal provenance commits the
    exact images, full Codex receipt, payload, task context, and frozen
    block-to-semantic-side manifest rather than a self-asserted opaque string.
    """

    def __init__(
        self,
        *,
        task_context_digest: str,
        block_orientations: tuple[NativeOrientation, NativeOrientation],
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
    ) -> None:
        _digest(task_context_digest, "proposer task context digest")
        _orientation_manifest_data(block_orientations)
        if type(minutes) is not int or minutes <= 0:
            raise PanelTypedCodexObserverError(
                "proposer call timeout must be a positive exact integer"
            )
        if not callable(transport):
            raise TypeError("typed proposer transport must be callable")
        self.task_context_digest = task_context_digest
        self.block_orientations = block_orientations
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.minutes = minutes
        self.verbose = verbose
        self.executable = executable
        self.cloud_policy_cache_snapshot = cloud_policy_cache_snapshot
        self.expected_launcher_digest = expected_launcher_digest
        self.model_catalog_snapshot = model_catalog_snapshot
        self.no_tools_attestation = no_tools_attestation
        self.transport = transport
        self.runtime = _bind_runtime(
            model=model,
            reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
            model_catalog_snapshot=model_catalog_snapshot,
            no_tools_attestation=no_tools_attestation,
        )
        self._artifact: TypedProposerCodexCallArtifact | None = None

    @property
    def artifact(self) -> TypedProposerCodexCallArtifact:
        if self._artifact is None:
            raise PanelTypedCodexObserverError("proposer call has not completed")
        return self._artifact

    def __call__(
        self,
        presentation: tuple[tuple[str, bytes], ...],
        prompt: str,
        output_schema: Mapping[str, Any],
    ) -> PanelFeatureProposerCallResult:
        if self._artifact is not None:
            raise PanelTypedCodexObserverError("proposer call adapter is exactly-once")
        if (
            type(presentation) is not tuple
            or len(presentation) != len(PANEL_FEATURE_PRESENTATION_NAMES)
            or tuple(item[0] for item in presentation)
            != PANEL_FEATURE_PRESENTATION_NAMES
        ):
            raise PanelTypedCodexObserverError("proposer presentation names differ")
        if prompt != panel_feature_proposer_prompt() or dict(
            output_schema
        ) != panel_feature_proposer_output_schema():
            raise PanelTypedCodexObserverError("proposer prompt or schema differs")
        frozen: list[tuple[str, bytes]] = []
        identities: list[PrototypeImageIdentity] = []
        for index, (name, raw) in enumerate(presentation):
            panel = _exact_png(raw, f"proposer panel {index}")
            frozen.append((name, panel))
            identities.append(
                PrototypeImageIdentity(
                    name, len(panel), hashlib.sha256(panel).hexdigest()
                )
            )
        try:
            payload, receipt = _scene_runtime._stage_and_call(
                tuple(frozen),
                prompt=prompt,
                schema=output_schema,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                minutes=self.minutes,
                verbose=self.verbose,
                executable=self.executable,
                cloud_policy_cache_snapshot=self.cloud_policy_cache_snapshot,
                expected_launcher_digest=self.expected_launcher_digest,
                model_catalog_snapshot=self.model_catalog_snapshot,
                no_tools_attestation=self.no_tools_attestation,
                transport=self.transport,
            )
            frozen_payload = _canonical_payload(payload, "proposer model payload")
            artifact = TypedProposerCodexCallArtifact(
                self.runtime,
                self.task_context_digest,
                self.block_orientations,
                canonical_digest(_orientation_manifest_data(self.block_orientations)),
                typed_proposer_transport_contract_digest(
                    self.runtime,
                    task_context_digest=self.task_context_digest,
                    block_orientations=self.block_orientations,
                ),
                tuple(identities),
                _proposer_presentation_digest(identities),
                hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                canonical_digest(dict(output_schema)),
                canonical_digest(frozen_payload),
                frozen_payload,
                receipt,
            )
            self._artifact = artifact
            return PanelFeatureProposerCallResult.seal(
                frozen_payload,
                prompt=prompt,
                output_schema=output_schema,
                presentation_digest=artifact.presentation_digest,
                external_receipt_digest=artifact.artifact_digest,
            )
        except PanelTypedCodexObserverError:
            raise
        except Exception as exc:
            raise PanelTypedCodexObserverError(
                "typed proposer call failed closed; no call artifact was produced"
            ) from exc


def invoke_receipted_panel_feature_proposer(
    support_pngs: Sequence[bytes],
    *,
    call: HeadlessCodexPanelFeatureReceiptedCall,
) -> PanelFeatureProposerResult:
    """Invoke core parsing with the call's already-frozen hidden manifest."""

    if type(call) is not HeadlessCodexPanelFeatureReceiptedCall:
        raise TypeError("production proposer invocation needs the headless adapter")
    return invoke_panel_feature_proposer(
        support_pngs,
        task_context_digest=call.task_context_digest,
        call=call,
        block_orientations=call.block_orientations,
    )


def verify_typed_proposer_codex_artifact(
    artifact: TypedProposerCodexCallArtifact,
    support_pngs: Sequence[bytes],
    *,
    expected_artifact_digest: str,
) -> TypedProposerCodexCallArtifact:
    """Cold replay one proposer receipt against the exact twelve PNG bytes."""

    if type(artifact) is not TypedProposerCodexCallArtifact:
        raise TypeError("proposer cold replay needs TypedProposerCodexCallArtifact")
    if isinstance(support_pngs, (bytes, str)) or len(support_pngs) != 12:
        raise PanelTypedCodexObserverError("proposer replay requires twelve PNGs")
    expected = _digest(expected_artifact_digest, "expected proposer artifact digest")
    restored = TypedProposerCodexCallArtifact.from_data(artifact.to_data())
    if restored.artifact_digest != expected:
        raise PanelTypedCodexObserverError("proposer artifact differs from commitment")
    frozen = tuple(
        (name, _exact_png(raw, f"proposer replay panel {index}"))
        for index, (name, raw) in enumerate(
            zip(PANEL_FEATURE_PRESENTATION_NAMES, support_pngs, strict=True)
        )
    )
    identities = tuple(
        PrototypeImageIdentity(name, len(raw), hashlib.sha256(raw).hexdigest())
        for name, raw in frozen
    )
    if identities != restored.presentation:
        raise PanelTypedCodexObserverError("proposer artifact image bytes differ")
    prompt = panel_feature_proposer_prompt()
    schema = panel_feature_proposer_output_schema()
    with tempfile.TemporaryDirectory(prefix="bongard-typed-proposer-replay-") as raw:
        directory = Path(raw)
        paths: list[str] = []
        names: list[str] = []
        for name, data in frozen:
            target = directory / name
            target.write_bytes(data)
            paths.append(str(target.resolve()))
            names.append(name)
        try:
            validate_codex_named_image_receipt(
                restored.codex_receipt,
                prompt,
                tuple(paths),
                tuple(names),
                schema,
                dict(restored.model_payload),
            )
        except Exception as exc:
            raise PanelTypedCodexObserverError(
                "proposer receipt cold replay failed"
            ) from exc
        for path, (_, expected_bytes) in zip(paths, frozen, strict=True):
            if Path(path).read_bytes() != expected_bytes:
                raise PanelTypedCodexObserverError(
                    "proposer cold-replay presentation changed"
                )
    return restored


__all__ = (
    "HeadlessCodexPanelFeatureReceiptedCall",
    "PanelOnlyObservationContext",
    "PanelTypedCodexObserverError",
    "TypedAxisCodexArtifact",
    "TypedCodexRuntimeBinding",
    "TypedOwnerCodexArtifact",
    "TypedProposerCodexCallArtifact",
    "build_panel_only_observation_context",
    "invoke_receipted_panel_feature_proposer",
    "observe_typed_panel_axis",
    "observe_typed_panel_owners",
    "panel_typed_codex_observer_source_digest",
    "typed_codex_observer_contract_data",
    "typed_codex_observer_contract_digest",
    "typed_measurement_protocol_data",
    "typed_measurement_protocol_digest",
    "typed_proposer_transport_contract_data",
    "typed_proposer_transport_contract_digest",
    "verify_typed_axis_codex_artifact",
    "verify_typed_owner_codex_artifact",
    "verify_typed_proposer_codex_artifact",
)

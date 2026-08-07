"""Profile-free visual prose and feature nominations for one Bongard task.

The model sees two neutral groups of six support images.  It may emit prose
and nominate identifiers from the frozen feature catalog, but it cannot pick
operators, thresholds, polarity, formulas, or executable code.  A later
Python-only version-space stage is the sole authority for operationalizing
the nominated feature family.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.prototype_object_observer_protocol import (
    OBJECT_FEATURE_IDS,
    parse_prototype_object_description_payload,
    prototype_object_description_output_schema,
)
from bongard.prototype_object_profiles import (
    OBJECT_FEATURE_CATALOG,
    OBJECT_FEATURE_CATALOG_DIGEST,
)
from bongard.prototype_object_scene_observer import (
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexNoToolsAttestation,
    PrototypeImageIdentity,
    PrototypeSceneObserverError,
    PrototypeSceneObserverStatus,
)
from bongard import prototype_object_scene_observer as _observer
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.transport import run_codex_named_images_structured


SEMANTIC_ARTIFACT_SCHEMA = "gkm.bongard-object-task-semantics.v1"
SEMANTIC_PROTOCOL_ID = "bongard.object-task-semantics/two-neutral-groups-v1"
GROUP_IDS = ("group_0", "group_1")
GROUP_SIZE = 6

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_TASK_ID = re.compile(r"(?:bd|ff|hd)_[A-Za-z0-9_.-]+\Z")


class ObjectBongardSemanticsError(ValueError):
    """A semantic turn, artifact, or replay commitment is malformed."""


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "model_can_nominate_feature_ids_only": True,
        "model_can_choose_operator_threshold_or_polarity": False,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_or_decision": False,
    }


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectBongardSemanticsError(f"{label} must be a sha256: address")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectBongardSemanticsError(f"{label} must be a raw SHA-256")
    return value


def _task_id(value: object) -> str:
    if not isinstance(value, str) or _TASK_ID.fullmatch(value) is None:
        raise ObjectBongardSemanticsError("task ID is outside the official grammar")
    return value


def object_bongard_semantics_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _catalog_lines() -> str:
    rows: list[str] = []
    for item in OBJECT_FEATURE_CATALOG:
        maximum = "unbounded" if item.maximum is None else str(item.maximum)
        rows.append(
            f"- {item.feature_id}; unit={item.unit}; range=0..{maximum}; "
            f"meaning={item.operational_description}"
        )
    return "\n".join(rows)


def object_bongard_semantics_prompt() -> str:
    return (
        "Inspect twelve drawings arranged as two neutral groups of six, named "
        "group_0_ref_00 through group_0_ref_05 and group_1_ref_00 through "
        "group_1_ref_05. For each group, write one concise sentence describing "
        "a recurring visible appearance and nominate one or more matching "
        "feature identifiers from the complete frozen measurement catalog. "
        "Ignore pose, scale, location, and incidental stroke variation. Return "
        "group_0 then group_1. Emit prose and feature identifiers only: do not "
        "choose an operator, threshold, number, polarity, weight, negation, "
        "disjunction, executable text, or experimental role. Python "
        "alone may later test a finite predeclared operationalization.\n\n"
        "Frozen measurement catalog:\n"
        + _catalog_lines()
    )


def object_bongard_semantics_protocol_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-object-task-semantics-protocol.v1",
            "protocol_id": SEMANTIC_PROTOCOL_ID,
            "source_digest": object_bongard_semantics_source_digest(),
            "prompt_sha256": hashlib.sha256(
                object_bongard_semantics_prompt().encode("utf-8")
            ).hexdigest(),
            "output_schema_digest": canonical_digest(
                prototype_object_description_output_schema()
            ),
            "feature_catalog_digest": OBJECT_FEATURE_CATALOG_DIGEST,
            "group_ids": list(GROUP_IDS),
            "images_per_group": GROUP_SIZE,
            "downstream_operationalization": (
                "explicit-finite-python-version-space-only"
            ),
            **_authority_data(),
        }
    )


def _panel_groups(
    group_0_panel_ids: Sequence[str], group_1_panel_ids: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    groups = (tuple(group_0_panel_ids), tuple(group_1_panel_ids))
    if any(
        len(group) != GROUP_SIZE
        or group != tuple(sorted(set(group)))
        or any(not isinstance(item, str) or not item for item in group)
        for group in groups
    ) or set(groups[0]) & set(groups[1]):
        raise ObjectBongardSemanticsError(
            "semantic groups must be two disjoint sorted six-panel tuples"
        )
    return groups


def _presentation(
    groups: tuple[tuple[str, ...], tuple[str, ...]],
    support_png_by_panel_id: Mapping[str, bytes],
) -> tuple[tuple[str, bytes], ...]:
    expected = set(groups[0]) | set(groups[1])
    if (
        not isinstance(support_png_by_panel_id, Mapping)
        or set(support_png_by_panel_id) != expected
        or any(not isinstance(key, str) for key in support_png_by_panel_id)
    ):
        raise ObjectBongardSemanticsError("support PNG key set differs")
    rows: list[tuple[str, bytes]] = []
    for group_index, group in enumerate(groups):
        for image_index, panel_id in enumerate(group):
            try:
                payload = _observer._legacy._validate_exact_png(
                    support_png_by_panel_id[panel_id], panel_id
                )
            except PrototypeSceneObserverError as exc:
                raise ObjectBongardSemanticsError("support image is not an exact PNG") from exc
            rows.append(
                (f"group_{group_index}_ref_{image_index:02d}.png", payload)
            )
    return tuple(rows)


def _artifact_preimage(value: "ObjectBongardSemanticArtifact") -> dict[str, object]:
    return {
        "schema": SEMANTIC_ARTIFACT_SCHEMA,
        "status": value.status.value,
        "task_id": value.task_id,
        "observation_context_digest": value.observation_context_digest,
        "group_panel_ids": [list(group) for group in value.group_panel_ids],
        "presentation": [item.to_data() for item in value.presentation],
        "prompt_digest": value.prompt_digest,
        "output_schema_digest": value.output_schema_digest,
        "protocol_digest": value.protocol_digest,
        "source_digest": value.source_digest,
        "transport_source_digest": value.transport_source_digest,
        "feature_catalog_digest": value.feature_catalog_digest,
        "model": value.model,
        "reasoning_effort": value.reasoning_effort,
        "model_digest": value.model_digest,
        "expected_launcher_digest": value.expected_launcher_digest,
        "cloud_policy_cache_binding": value.cloud_policy_cache_binding,
        "model_catalog_digest": value.model_catalog_digest,
        "no_tools_attestation_digest": value.no_tools_attestation_digest,
        "environment_digest": value.environment_digest,
        "model_payload": value.model_payload,
        "receipt": _observer._receipt_to_data(value.receipt),
        "receipt_identity": value.receipt_identity,
        "rubrics": list(value.rubrics),
        "feature_families": [list(group) for group in value.feature_families],
        "failure_code": value.failure_code,
        "failure_type": value.failure_type,
        "physical_call_count": 1,
        "vision_prose_is_audit_evidence": True,
        "feature_nominations_constrain_version_space": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardSemanticArtifact:
    status: PrototypeSceneObserverStatus
    task_id: str
    observation_context_digest: str
    group_panel_ids: tuple[tuple[str, ...], tuple[str, ...]]
    presentation: tuple[PrototypeImageIdentity, ...]
    prompt_digest: str
    output_schema_digest: str
    protocol_digest: str
    source_digest: str
    transport_source_digest: str
    feature_catalog_digest: str
    model: str
    reasoning_effort: str
    model_digest: str
    expected_launcher_digest: str | None
    cloud_policy_cache_binding: str
    model_catalog_digest: str
    no_tools_attestation_digest: str
    environment_digest: str
    model_payload: Mapping[str, Any] | None
    receipt: object | None
    receipt_identity: str | None
    rubrics: tuple[str, ...]
    feature_families: tuple[tuple[str, ...], ...]
    failure_code: str | None
    failure_type: str | None
    artifact_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, PrototypeSceneObserverStatus):
            raise TypeError("semantic status has the wrong type")
        _task_id(self.task_id)
        _address(self.observation_context_digest, "observation context digest")
        _panel_groups(*self.group_panel_ids)
        expected_names = tuple(
            f"group_{group}_ref_{index:02d}.png"
            for group in range(2)
            for index in range(GROUP_SIZE)
        )
        if (
            len(self.presentation) != 2 * GROUP_SIZE
            or tuple(item.name for item in self.presentation) != expected_names
        ):
            raise ObjectBongardSemanticsError("semantic presentation differs")
        for name in (
            "prompt_digest", "output_schema_digest", "protocol_digest",
            "source_digest", "transport_source_digest", "feature_catalog_digest",
            "model_digest", "model_catalog_digest", "no_tools_attestation_digest",
            "environment_digest", "artifact_digest",
        ):
            _digest(getattr(self, name), name)
        if (
            self.feature_catalog_digest != OBJECT_FEATURE_CATALOG_DIGEST
            or self.prompt_digest
            != hashlib.sha256(object_bongard_semantics_prompt().encode("utf-8")).hexdigest()
            or self.output_schema_digest
            != canonical_digest(prototype_object_description_output_schema())
            or self.protocol_digest != object_bongard_semantics_protocol_digest()
            or self.source_digest != object_bongard_semantics_source_digest()
        ):
            raise ObjectBongardSemanticsError("semantic protocol binding differs")
        if self.receipt is None:
            if self.receipt_identity is not None:
                raise ObjectBongardSemanticsError("receipt identity lacks receipt")
        elif getattr(self.receipt, "receipt_digest", None) != self.receipt_identity:
            raise ObjectBongardSemanticsError("semantic receipt identity differs")
        success = self.status is PrototypeSceneObserverStatus.SUCCESS
        parser_error = self.status is PrototypeSceneObserverStatus.PARSER_ERROR
        transport_error = self.status is PrototypeSceneObserverStatus.TRANSPORT_ERROR
        if success:
            if (
                self.model_payload is None
                or self.receipt is None
                or len(self.rubrics) != 2
                or len(self.feature_families) != 2
                or any(not family for family in self.feature_families)
                or self.failure_code is not None
                or self.failure_type is not None
            ):
                raise ObjectBongardSemanticsError("successful semantic artifact differs")
            for family in self.feature_families:
                if (
                    family != tuple(sorted(set(family), key=OBJECT_FEATURE_IDS.index))
                    or any(item not in OBJECT_FEATURE_IDS for item in family)
                ):
                    raise ObjectBongardSemanticsError("feature family is not canonical")
        elif parser_error:
            if (
                self.model_payload is None
                or self.receipt is None
                or self.rubrics
                or self.feature_families
                or self.failure_code != "semantic_payload_rejected"
                or self.failure_type is None
            ):
                raise ObjectBongardSemanticsError("parser-error semantic artifact differs")
        elif transport_error:
            if (
                self.model_payload is not None
                or self.receipt is not None
                or self.rubrics
                or self.feature_families
                or self.failure_code != "semantic_transport_failed"
                or self.failure_type is None
            ):
                raise ObjectBongardSemanticsError("transport-error semantic artifact differs")
        else:
            raise ObjectBongardSemanticsError("unsupported semantic terminal state")
        if canonical_digest(_artifact_preimage(self)) != self.artifact_digest:
            raise ObjectBongardSemanticsError("semantic artifact digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_artifact_preimage(self), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any], *, expected_artifact_digest: str | None = None
    ) -> "ObjectBongardSemanticArtifact":
        fields = {
            "schema", "status", "task_id", "observation_context_digest",
            "group_panel_ids", "presentation", "prompt_digest",
            "output_schema_digest", "protocol_digest", "source_digest",
            "transport_source_digest", "feature_catalog_digest", "model",
            "reasoning_effort", "model_digest", "expected_launcher_digest",
            "cloud_policy_cache_binding", "model_catalog_digest",
            "no_tools_attestation_digest", "environment_digest", "model_payload",
            "receipt", "receipt_identity", "rubrics", "feature_families",
            "failure_code", "failure_type", "physical_call_count",
            "vision_prose_is_audit_evidence",
            "feature_nominations_constrain_version_space", *_authority_data(),
            "artifact_digest",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ObjectBongardSemanticsError("semantic artifact fields differ")
        if (
            value["schema"] != SEMANTIC_ARTIFACT_SCHEMA
            or value["physical_call_count"] != 1
            or value["vision_prose_is_audit_evidence"] is not True
            or value["feature_nominations_constrain_version_space"] is not True
            or any(value[key] != item for key, item in _authority_data().items())
            or not isinstance(value["group_panel_ids"], list)
            or len(value["group_panel_ids"]) != 2
            or not isinstance(value["presentation"], list)
            or not isinstance(value["rubrics"], list)
            or not isinstance(value["feature_families"], list)
        ):
            raise ObjectBongardSemanticsError("semantic artifact policy differs")
        result = cls(
            status=PrototypeSceneObserverStatus(value["status"]),
            task_id=value["task_id"],
            observation_context_digest=value["observation_context_digest"],
            group_panel_ids=tuple(
                tuple(group) for group in value["group_panel_ids"]
            ),  # type: ignore[arg-type]
            presentation=tuple(
                PrototypeImageIdentity.from_data(item)
                for item in value["presentation"]
            ),
            prompt_digest=value["prompt_digest"],
            output_schema_digest=value["output_schema_digest"],
            protocol_digest=value["protocol_digest"],
            source_digest=value["source_digest"],
            transport_source_digest=value["transport_source_digest"],
            feature_catalog_digest=value["feature_catalog_digest"],
            model=value["model"],
            reasoning_effort=value["reasoning_effort"],
            model_digest=value["model_digest"],
            expected_launcher_digest=value["expected_launcher_digest"],
            cloud_policy_cache_binding=value["cloud_policy_cache_binding"],
            model_catalog_digest=value["model_catalog_digest"],
            no_tools_attestation_digest=value["no_tools_attestation_digest"],
            environment_digest=value["environment_digest"],
            model_payload=value["model_payload"],
            receipt=_observer._receipt_from_data(value["receipt"]),
            receipt_identity=value["receipt_identity"],
            rubrics=tuple(value["rubrics"]),
            feature_families=tuple(
                tuple(group) for group in value["feature_families"]
            ),
            failure_code=value["failure_code"],
            failure_type=value["failure_type"],
            artifact_digest=value["artifact_digest"],
        )
        if expected_artifact_digest is not None and result.artifact_digest != _digest(
            expected_artifact_digest, "expected semantic artifact digest"
        ):
            raise ObjectBongardSemanticsError("semantic artifact commitment differs")
        if result.to_data() != dict(value):
            raise ObjectBongardSemanticsError("semantic artifact is not canonical")
        return result


def _build_artifact(**values: object) -> ObjectBongardSemanticArtifact:
    provisional = object.__new__(ObjectBongardSemanticArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardSemanticArtifact(
        **values,  # type: ignore[arg-type]
        artifact_digest=canonical_digest(_artifact_preimage(provisional)),
    )


def describe_object_bongard_support(
    *,
    task_id: str,
    group_0_panel_ids: Sequence[str],
    group_1_panel_ids: Sequence[str],
    support_png_by_panel_id: Mapping[str, bytes],
    observation_context_digest: str,
    model: str,
    reasoning_effort: str = "medium",
    minutes: int = 15,
    verbose: bool = False,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    expected_launcher_digest: str | None = None,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    transport: object = run_codex_named_images_structured,
) -> ObjectBongardSemanticArtifact:
    """Produce prose plus feature IDs without exposing labels or numeric choices."""

    task = _task_id(task_id)
    context = _address(observation_context_digest, "observation context digest")
    groups = _panel_groups(group_0_panel_ids, group_1_panel_ids)
    presentation = _presentation(groups, support_png_by_panel_id)
    identities = _observer._legacy._image_identities(presentation)
    if not callable(transport):
        raise TypeError("transport must be callable")
    policy, model_catalog_digest, no_tools_digest = _observer._runtime_identities(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    prompt = object_bongard_semantics_prompt()
    schema = prototype_object_description_output_schema()
    _observer._legacy.validate_codex_strict_output_schema(schema)
    _observer._legacy._assert_model_visible_boundary(
        prompt,
        schema,
        tuple(name for name, _ in presentation),
        hidden_values=(task, context, *(item for group in groups for item in group)),
    )
    common: dict[str, object] = {
        "task_id": task,
        "observation_context_digest": context,
        "group_panel_ids": groups,
        "presentation": identities,
        "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "output_schema_digest": canonical_digest(schema),
        "protocol_digest": object_bongard_semantics_protocol_digest(),
        "source_digest": object_bongard_semantics_source_digest(),
        "transport_source_digest": _observer.prototype_scene_transport_source_digest(),
        "feature_catalog_digest": OBJECT_FEATURE_CATALOG_DIGEST,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "model_digest": _observer.prototype_scene_observer_model_digest(
            model, reasoning_effort
        ),
        "expected_launcher_digest": expected_launcher_digest,
        "cloud_policy_cache_binding": policy,
        "model_catalog_digest": model_catalog_digest,
        "no_tools_attestation_digest": no_tools_digest,
        "environment_digest": _observer.prototype_scene_observer_environment_digest(
            model=model,
            reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=policy,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_digest,
        ),
    }
    try:
        payload, receipt = _observer._legacy._stage_and_call(
            presentation,
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
            transport=transport,  # type: ignore[arg-type]
        )
    except Exception as exc:
        return _build_artifact(
            **common,
            status=PrototypeSceneObserverStatus.TRANSPORT_ERROR,
            model_payload=None,
            receipt=None,
            receipt_identity=None,
            rubrics=(),
            feature_families=(),
            failure_code="semantic_transport_failed",
            failure_type=type(exc).__name__,
        )
    try:
        parsed = parse_prototype_object_description_payload(payload)
    except (TypeError, ValueError):
        return _build_artifact(
            **common,
            status=PrototypeSceneObserverStatus.PARSER_ERROR,
            model_payload=payload,
            receipt=receipt,
            receipt_identity=receipt.receipt_digest,
            rubrics=(),
            feature_families=(),
            failure_code="semantic_payload_rejected",
            failure_type="PrototypeObjectProtocolError",
        )
    return _build_artifact(
        **common,
        status=PrototypeSceneObserverStatus.SUCCESS,
        model_payload=payload,
        receipt=receipt,
        receipt_identity=receipt.receipt_digest,
        rubrics=tuple(item.prose for item in parsed.audit_rubrics),
        feature_families=tuple(parsed.feature_families),
        failure_code=None,
        failure_type=None,
    )


def verify_object_bongard_semantic_artifact(
    artifact: ObjectBongardSemanticArtifact,
    *,
    support_png_by_panel_id: Mapping[str, bytes],
    expected_task_id: str,
    expected_observation_context_digest: str,
    expected_artifact_digest: str,
) -> ObjectBongardSemanticArtifact:
    """Cold-replay exact support bytes, receipt, parser, and nominations."""

    if not isinstance(artifact, ObjectBongardSemanticArtifact):
        raise TypeError("artifact must be ObjectBongardSemanticArtifact")
    if (
        artifact.task_id != _task_id(expected_task_id)
        or artifact.observation_context_digest
        != _address(expected_observation_context_digest, "expected context digest")
        or artifact.artifact_digest
        != _digest(expected_artifact_digest, "expected semantic artifact digest")
    ):
        raise ObjectBongardSemanticsError("semantic artifact parents differ")
    presentation = _presentation(artifact.group_panel_ids, support_png_by_panel_id)
    if artifact.presentation != _observer._legacy._image_identities(presentation):
        raise ObjectBongardSemanticsError("semantic presentation replay differs")
    if artifact.receipt is not None:
        assert artifact.model_payload is not None
        _observer._verify_receipt(
            artifact.receipt,
            presentation,
            object_bongard_semantics_prompt(),
            prototype_object_description_output_schema(),
            artifact.model_payload,
        )
    if artifact.status is PrototypeSceneObserverStatus.SUCCESS:
        parsed = parse_prototype_object_description_payload(artifact.model_payload)
        if (
            tuple(item.prose for item in parsed.audit_rubrics) != artifact.rubrics
            or tuple(parsed.feature_families) != artifact.feature_families
        ):
            raise ObjectBongardSemanticsError("semantic payload replay differs")
    restored = ObjectBongardSemanticArtifact.from_data(
        artifact.to_data(), expected_artifact_digest=expected_artifact_digest
    )
    if restored != artifact:
        raise ObjectBongardSemanticsError("semantic artifact round trip differs")
    return artifact


__all__ = (
    "GROUP_IDS",
    "GROUP_SIZE",
    "ObjectBongardSemanticArtifact",
    "ObjectBongardSemanticsError",
    "SEMANTIC_ARTIFACT_SCHEMA",
    "describe_object_bongard_support",
    "object_bongard_semantics_prompt",
    "object_bongard_semantics_protocol_digest",
    "object_bongard_semantics_source_digest",
    "verify_object_bongard_semantic_artifact",
)

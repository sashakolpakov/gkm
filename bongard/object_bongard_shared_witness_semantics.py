"""One-call proposer for two ranked structured shared-witness contrasts.

This versioned path coexists with the source-sealed historical semantic
artifact.  The model can emit only four prose components per rank: one shared
anchor, one visual axis, and one endpoint value for each neutral group.  Python
validates the components and renders both descriptions from the same anchor
and axis before any observer or support-selection step.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping, Sequence

from bongard import object_bongard_semantics as _legacy_presentation
from bongard import prototype_object_scene_observer as _observer
from bongard.canonical import canonical_digest
from bongard.object_bongard_shared_witness import (
    ObjectBongardSharedWitnessContrast,
    ObjectBongardSharedWitnessError,
    SHARED_WITNESS_IR_ID,
    SHARED_WITNESS_RENDERER_ID,
    object_bongard_shared_witness_source_digest,
)
from bongard.object_bongard_soft_cues import ObjectBongardSoftCuePair
from bongard.prototype_object_scene_observer import (
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexNoToolsAttestation,
    PrototypeImageIdentity,
    PrototypeSceneObserverError,
    PrototypeSceneObserverStatus,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.transport import run_codex_named_images_structured


SHARED_WITNESS_SEMANTIC_ARTIFACT_SCHEMA = (
    "gkm.bongard-shared-witness-semantics-artifact.v1"
)
SHARED_WITNESS_SEMANTIC_PROTOCOL_ID = (
    "bongard.shared-witness-semantics/two-ranked-single-entity-axis-contrasts-v1"
)
GROUP_IDS = ("group_0", "group_1")
GROUP_SIZE = 6
SHARED_WITNESS_CANDIDATE_COUNT = 2

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_TASK_ID = re.compile(r"(?:bd|ff|hd)_[A-Za-z0-9_.-]+\Z")


class ObjectBongardSharedWitnessSemanticsError(ValueError):
    """A structured semantic turn, artifact, or replay is malformed."""


# Same public-style error name for campaign code that swaps only this module.
ObjectBongardSemanticsError = ObjectBongardSharedWitnessSemanticsError


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "model_emits_shared_anchor_axis_endpoints_only": True,
        "python_renders_both_descriptions": True,
        "independent_free_form_group_cues_representable": False,
        "same_individual_required_within_panel": True,
        "same_physical_individual_across_panels_required": False,
        "same_entity_kind_across_panels_required": True,
        "single_visual_axis_required": True,
        "explicit_semantic_negation_allowed": False,
        "model_can_choose_operator_threshold_or_polarity": False,
        "feature_catalog_used": False,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_or_decision": False,
        "lean_required_for_replay": False,
    }


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectBongardSharedWitnessSemanticsError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectBongardSharedWitnessSemanticsError(
            f"{label} must be a sha256: address"
        )
    return value


def _task_id(value: object) -> str:
    if not isinstance(value, str) or _TASK_ID.fullmatch(value) is None:
        raise ObjectBongardSharedWitnessSemanticsError(
            "task ID is outside the official grammar"
        )
    return value


def _groups(
    group_0_panel_ids: Sequence[str], group_1_panel_ids: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    groups = (tuple(group_0_panel_ids), tuple(group_1_panel_ids))
    if any(
        len(group) != GROUP_SIZE
        or group != tuple(sorted(set(group)))
        or any(not isinstance(item, str) or not item for item in group)
        for group in groups
    ) or set(groups[0]) & set(groups[1]):
        raise ObjectBongardSharedWitnessSemanticsError(
            "semantic groups must be two disjoint sorted six-panel tuples"
        )
    return groups


def _presentation(
    groups: tuple[tuple[str, ...], tuple[str, ...]],
    support_png_by_panel_id: Mapping[str, bytes],
) -> tuple[tuple[str, bytes], ...]:
    try:
        return _legacy_presentation._presentation(groups, support_png_by_panel_id)
    except Exception as exc:
        raise ObjectBongardSharedWitnessSemanticsError(
            "support presentation differs"
        ) from exc


def object_bongard_shared_witness_semantics_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def object_bongard_shared_witness_semantics_prompt() -> str:
    return (
        "Inspect twelve drawings arranged as two neutral groups of six, named "
        "group_0_ref_00 through group_0_ref_05 and group_1_ref_00 through "
        "group_1_ref_05. Return exactly two ranked proposals named proposal_0 "
        "then proposal_1. For each proposal, identify one singular visible entity "
        "kind that recurs across the drawings. This means the same entity kind, not "
        "one physical individual shared across panels. Put "
        "that noun phrase in shared_anchor. Name exactly one visible attribute of "
        "each such individual in visual_axis. Within each panel, both endpoints must "
        "be scored on the same individual instance. Put one positive value of that exact axis "
        "in group_0_endpoint and a different positive value of that same axis in "
        "group_1_endpoint. The endpoints must be alternative values on one entity; "
        "they must not be two separately coexisting features, full descriptions, "
        "clauses, or different parts selected for each group. Do not express an "
        "endpoint as absence, lack, plainness, emptiness, or an un-prefixed state. "
        "Use lowercase atomic phrases with letters, spaces, apostrophes, or hyphens. "
        "Do not put group roles, comparison words, logic, digits, operators, pose, "
        "scale, or location in any field. Prefer a concrete topology, count, angle, "
        "curvature, junction, arrangement, symmetry, or marking axis. proposal_0 is "
        "the strongest contrast. proposal_1 must use a genuinely different shared "
        "anchor or visual axis, not paraphrased endpoints. Python alone renders "
        "Description A and Description B with the same anchor and axis. Python "
        "alone fixes orientation, uncertainty, thresholds, selection, and replay."
    )


def object_bongard_shared_witness_semantics_output_schema() -> dict[str, object]:
    proposal = {
        "type": "object",
        "properties": {
            "shared_anchor": {"type": "string"},
            "visual_axis": {"type": "string"},
            "group_0_endpoint": {"type": "string"},
            "group_1_endpoint": {"type": "string"},
        },
        "required": [
            "shared_anchor",
            "visual_axis",
            "group_0_endpoint",
            "group_1_endpoint",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"proposal_0": proposal, "proposal_1": proposal},
        "required": ["proposal_0", "proposal_1"],
        "additionalProperties": False,
    }


def object_bongard_shared_witness_semantics_protocol_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-shared-witness-semantics-protocol.v1",
            "protocol_id": SHARED_WITNESS_SEMANTIC_PROTOCOL_ID,
            "source_digest": object_bongard_shared_witness_semantics_source_digest(),
            "shared_witness_ir_source_digest": (
                object_bongard_shared_witness_source_digest()
            ),
            "legacy_presentation_helper_source_digest": (
                _legacy_presentation.object_bongard_semantics_source_digest()
            ),
            "prompt_sha256": hashlib.sha256(
                object_bongard_shared_witness_semantics_prompt().encode("utf-8")
            ).hexdigest(),
            "output_schema_digest": canonical_digest(
                object_bongard_shared_witness_semantics_output_schema()
            ),
            "ir_id": SHARED_WITNESS_IR_ID,
            "renderer_id": SHARED_WITNESS_RENDERER_ID,
            "group_ids": list(GROUP_IDS),
            "images_per_group": GROUP_SIZE,
            "ranked_candidate_count": SHARED_WITNESS_CANDIDATE_COUNT,
            "rank_one_requires_different_anchor_or_axis": True,
            "free_form_group_cue_fields_present": False,
            "rendered_cues_are_model_output": False,
            **_authority_data(),
        }
    )


def _parse_shared_witness_payload(
    payload: object,
) -> tuple[ObjectBongardSharedWitnessContrast, ObjectBongardSharedWitnessContrast]:
    if not isinstance(payload, Mapping) or set(payload) != {
        "proposal_0",
        "proposal_1",
    }:
        raise ObjectBongardSharedWitnessSemanticsError(
            "shared-witness payload fields differ"
        )
    contrasts: list[ObjectBongardSharedWitnessContrast] = []
    expected_fields = {
        "shared_anchor",
        "visual_axis",
        "group_0_endpoint",
        "group_1_endpoint",
    }
    for rank in range(SHARED_WITNESS_CANDIDATE_COUNT):
        row = payload[f"proposal_{rank}"]
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise ObjectBongardSharedWitnessSemanticsError(
                f"shared-witness proposal {rank} fields differ"
            )
        try:
            contrasts.append(
                ObjectBongardSharedWitnessContrast.create(
                    rank,
                    shared_anchor=row["shared_anchor"],
                    visual_axis=row["visual_axis"],
                    group_0_endpoint=row["group_0_endpoint"],
                    group_1_endpoint=row["group_1_endpoint"],
                )
            )
        except (TypeError, ValueError, ObjectBongardSharedWitnessError) as exc:
            raise ObjectBongardSharedWitnessSemanticsError(
                "shared-witness proposal violates the closed IR"
            ) from exc
    rank_axes = tuple(
        (item.shared_anchor.casefold(), item.visual_axis.casefold())
        for item in contrasts
    )
    if rank_axes[0] == rank_axes[1]:
        raise ObjectBongardSharedWitnessSemanticsError(
            "rank one must use a different shared anchor or visual axis"
        )
    return tuple(contrasts)  # type: ignore[return-value]


def _artifact_content(
    value: "ObjectBongardSharedWitnessSemanticArtifact",
) -> dict[str, object]:
    return {
        "schema": SHARED_WITNESS_SEMANTIC_ARTIFACT_SCHEMA,
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
        "contrast_candidates": [
            item.to_data() for item in value.contrast_candidates
        ],
        "failure_code": value.failure_code,
        "failure_type": value.failure_type,
        "physical_call_count": 1,
        "rendered_descriptions_persisted_inside_contrasts": True,
        "observer_must_persist_individual_witness_evidence": True,
        "direct_comparative_score_is_sufficient_evidence": False,
        "ranked_candidate_count": SHARED_WITNESS_CANDIDATE_COUNT,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardSharedWitnessSemanticArtifact:
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
    contrast_candidates: tuple[ObjectBongardSharedWitnessContrast, ...]
    failure_code: str | None
    failure_type: str | None
    artifact_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, PrototypeSceneObserverStatus):
            raise TypeError("shared-witness semantic status has the wrong type")
        _task_id(self.task_id)
        _address(self.observation_context_digest, "observation context digest")
        _groups(*self.group_panel_ids)
        expected_names = tuple(
            f"group_{group}_ref_{index:02d}.png"
            for group in range(2)
            for index in range(GROUP_SIZE)
        )
        if (
            len(self.presentation) != 2 * GROUP_SIZE
            or tuple(item.name for item in self.presentation) != expected_names
        ):
            raise ObjectBongardSharedWitnessSemanticsError(
                "shared-witness semantic presentation differs"
            )
        for name in (
            "prompt_digest",
            "output_schema_digest",
            "protocol_digest",
            "source_digest",
            "transport_source_digest",
            "model_digest",
            "model_catalog_digest",
            "no_tools_attestation_digest",
            "environment_digest",
            "artifact_digest",
        ):
            _digest(getattr(self, name), name)
        if (
            self.prompt_digest
            != hashlib.sha256(
                object_bongard_shared_witness_semantics_prompt().encode("utf-8")
            ).hexdigest()
            or self.output_schema_digest
            != canonical_digest(object_bongard_shared_witness_semantics_output_schema())
            or self.protocol_digest
            != object_bongard_shared_witness_semantics_protocol_digest()
            or self.source_digest
            != object_bongard_shared_witness_semantics_source_digest()
        ):
            raise ObjectBongardSharedWitnessSemanticsError(
                "shared-witness semantic protocol binding differs"
            )
        if self.receipt is None:
            if self.receipt_identity is not None:
                raise ObjectBongardSharedWitnessSemanticsError(
                    "receipt identity lacks a receipt"
                )
        elif getattr(self.receipt, "receipt_digest", None) != self.receipt_identity:
            raise ObjectBongardSharedWitnessSemanticsError(
                "shared-witness semantic receipt identity differs"
            )
        if self.status is PrototypeSceneObserverStatus.SUCCESS:
            if (
                self.model_payload is None
                or self.receipt is None
                or len(self.contrast_candidates)
                != SHARED_WITNESS_CANDIDATE_COUNT
                or tuple(item.candidate_rank for item in self.contrast_candidates)
                != (0, 1)
                or self.failure_code is not None
                or self.failure_type is not None
            ):
                raise ObjectBongardSharedWitnessSemanticsError(
                    "successful shared-witness semantic artifact differs"
                )
        elif self.status is PrototypeSceneObserverStatus.PARSER_ERROR:
            if (
                self.model_payload is None
                or self.receipt is None
                or self.contrast_candidates
                or self.failure_code != "shared_witness_payload_rejected"
                or self.failure_type is None
            ):
                raise ObjectBongardSharedWitnessSemanticsError(
                    "parser-error shared-witness artifact differs"
                )
        elif self.status is PrototypeSceneObserverStatus.TRANSPORT_ERROR:
            if (
                self.model_payload is not None
                or self.receipt is not None
                or self.contrast_candidates
                or self.failure_code != "shared_witness_transport_failed"
                or self.failure_type is None
            ):
                raise ObjectBongardSharedWitnessSemanticsError(
                    "transport-error shared-witness artifact differs"
                )
        else:
            raise ObjectBongardSharedWitnessSemanticsError(
                "unsupported shared-witness terminal state"
            )
        if self.artifact_digest != canonical_digest(_artifact_content(self)):
            raise ObjectBongardSharedWitnessSemanticsError(
                "shared-witness semantic artifact digest differs"
            )

    @property
    def soft_cue_candidates(self) -> tuple[ObjectBongardSoftCuePair, ...]:
        return tuple(item.soft_cue_pair for item in self.contrast_candidates)

    def to_data(self) -> dict[str, object]:
        return {**_artifact_content(self), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        expected_artifact_digest: str | None = None,
    ) -> "ObjectBongardSharedWitnessSemanticArtifact":
        expected_fields = {
            "schema",
            "status",
            "task_id",
            "observation_context_digest",
            "group_panel_ids",
            "presentation",
            "prompt_digest",
            "output_schema_digest",
            "protocol_digest",
            "source_digest",
            "transport_source_digest",
            "model",
            "reasoning_effort",
            "model_digest",
            "expected_launcher_digest",
            "cloud_policy_cache_binding",
            "model_catalog_digest",
            "no_tools_attestation_digest",
            "environment_digest",
            "model_payload",
            "receipt",
            "receipt_identity",
            "contrast_candidates",
            "failure_code",
            "failure_type",
            "physical_call_count",
            "rendered_descriptions_persisted_inside_contrasts",
            "observer_must_persist_individual_witness_evidence",
            "direct_comparative_score_is_sufficient_evidence",
            "ranked_candidate_count",
            *_authority_data(),
            "artifact_digest",
        }
        if not isinstance(value, Mapping) or set(value) != expected_fields:
            raise ObjectBongardSharedWitnessSemanticsError(
                "shared-witness artifact fields differ"
            )
        if (
            value["schema"] != SHARED_WITNESS_SEMANTIC_ARTIFACT_SCHEMA
            or value["physical_call_count"] != 1
            or value["rendered_descriptions_persisted_inside_contrasts"] is not True
            or value["observer_must_persist_individual_witness_evidence"] is not True
            or value["direct_comparative_score_is_sufficient_evidence"] is not False
            or value["ranked_candidate_count"] != SHARED_WITNESS_CANDIDATE_COUNT
            or any(value[key] != item for key, item in _authority_data().items())
            or not isinstance(value["group_panel_ids"], list)
            or not isinstance(value["presentation"], list)
            or not isinstance(value["contrast_candidates"], list)
        ):
            raise ObjectBongardSharedWitnessSemanticsError(
                "shared-witness artifact policy differs"
            )
        try:
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
                contrast_candidates=tuple(
                    ObjectBongardSharedWitnessContrast.from_data(item)
                    for item in value["contrast_candidates"]
                ),
                failure_code=value["failure_code"],
                failure_type=value["failure_type"],
                artifact_digest=value["artifact_digest"],
            )
        except (TypeError, ValueError) as exc:
            raise ObjectBongardSharedWitnessSemanticsError(
                "shared-witness artifact payload is malformed"
            ) from exc
        if expected_artifact_digest is not None and result.artifact_digest != _digest(
            expected_artifact_digest, "expected semantic artifact digest"
        ):
            raise ObjectBongardSharedWitnessSemanticsError(
                "shared-witness artifact commitment differs"
            )
        if result.to_data() != dict(value):
            raise ObjectBongardSharedWitnessSemanticsError(
                "shared-witness artifact is not canonical"
            )
        return result


# Same public-style artifact name for explicit module swaps.  It is not a
# subclass of the historical type and cannot pass an old strict type check.
ObjectBongardSemanticArtifact = ObjectBongardSharedWitnessSemanticArtifact


def _build_artifact(**values: object) -> ObjectBongardSharedWitnessSemanticArtifact:
    provisional = object.__new__(ObjectBongardSharedWitnessSemanticArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardSharedWitnessSemanticArtifact(
        **values,  # type: ignore[arg-type]
        artifact_digest=canonical_digest(_artifact_content(provisional)),
    )


def describe_object_bongard_shared_witness_support(
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
) -> ObjectBongardSharedWitnessSemanticArtifact:
    """Propose two ranked shared-witness contrasts in one vision call."""

    task = _task_id(task_id)
    context = _address(observation_context_digest, "observation context digest")
    groups = _groups(group_0_panel_ids, group_1_panel_ids)
    presentation = _presentation(groups, support_png_by_panel_id)
    identities = _observer._legacy._image_identities(presentation)
    if not callable(transport):
        raise TypeError("transport must be callable")
    policy, catalog_digest, no_tools_digest = _observer._runtime_identities(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    prompt = object_bongard_shared_witness_semantics_prompt()
    schema = object_bongard_shared_witness_semantics_output_schema()
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
        "protocol_digest": object_bongard_shared_witness_semantics_protocol_digest(),
        "source_digest": object_bongard_shared_witness_semantics_source_digest(),
        "transport_source_digest": _observer.prototype_scene_transport_source_digest(),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "model_digest": _observer.prototype_scene_observer_model_digest(
            model, reasoning_effort
        ),
        "expected_launcher_digest": expected_launcher_digest,
        "cloud_policy_cache_binding": policy,
        "model_catalog_digest": catalog_digest,
        "no_tools_attestation_digest": no_tools_digest,
        "environment_digest": _observer.prototype_scene_observer_environment_digest(
            model=model,
            reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=policy,
            model_catalog_digest=catalog_digest,
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
            contrast_candidates=(),
            failure_code="shared_witness_transport_failed",
            failure_type=type(exc).__name__,
        )
    try:
        contrasts = _parse_shared_witness_payload(payload)
    except (TypeError, ValueError):
        return _build_artifact(
            **common,
            status=PrototypeSceneObserverStatus.PARSER_ERROR,
            model_payload=payload,
            receipt=receipt,
            receipt_identity=receipt.receipt_digest,
            contrast_candidates=(),
            failure_code="shared_witness_payload_rejected",
            failure_type="ObjectBongardSharedWitnessError",
        )
    return _build_artifact(
        **common,
        status=PrototypeSceneObserverStatus.SUCCESS,
        model_payload=payload,
        receipt=receipt,
        receipt_identity=receipt.receipt_digest,
        contrast_candidates=contrasts,
        failure_code=None,
        failure_type=None,
    )


# Public-style name used by semantic callers after an explicit module switch.
describe_object_bongard_support = describe_object_bongard_shared_witness_support


def verify_object_bongard_shared_witness_semantic_artifact(
    artifact: ObjectBongardSharedWitnessSemanticArtifact,
    *,
    support_png_by_panel_id: Mapping[str, bytes],
    expected_task_id: str,
    expected_observation_context_digest: str,
    expected_artifact_digest: str,
) -> ObjectBongardSharedWitnessSemanticArtifact:
    """Cold-replay pixels, receipt, parser, renderer, and exact IR."""

    if not isinstance(artifact, ObjectBongardSharedWitnessSemanticArtifact):
        raise TypeError("artifact must be a shared-witness semantic artifact")
    if (
        artifact.task_id != _task_id(expected_task_id)
        or artifact.observation_context_digest
        != _address(expected_observation_context_digest, "expected context digest")
        or artifact.artifact_digest
        != _digest(expected_artifact_digest, "expected artifact digest")
    ):
        raise ObjectBongardSharedWitnessSemanticsError(
            "shared-witness artifact parents differ"
        )
    presentation = _presentation(artifact.group_panel_ids, support_png_by_panel_id)
    if artifact.presentation != _observer._legacy._image_identities(presentation):
        raise ObjectBongardSharedWitnessSemanticsError(
            "shared-witness presentation replay differs"
        )
    if artifact.receipt is not None:
        assert artifact.model_payload is not None
        _observer._verify_receipt(
            artifact.receipt,
            presentation,
            object_bongard_shared_witness_semantics_prompt(),
            object_bongard_shared_witness_semantics_output_schema(),
            artifact.model_payload,
        )
    if artifact.status is PrototypeSceneObserverStatus.SUCCESS:
        reparsed = _parse_shared_witness_payload(artifact.model_payload)
        if reparsed != artifact.contrast_candidates:
            raise ObjectBongardSharedWitnessSemanticsError(
                "shared-witness parser replay differs"
            )
    restored = ObjectBongardSharedWitnessSemanticArtifact.from_data(
        artifact.to_data(), expected_artifact_digest=expected_artifact_digest
    )
    if restored != artifact:
        raise ObjectBongardSharedWitnessSemanticsError(
            "shared-witness artifact round trip differs"
        )
    return artifact


verify_object_bongard_semantic_artifact = (
    verify_object_bongard_shared_witness_semantic_artifact
)
object_bongard_semantics_prompt = object_bongard_shared_witness_semantics_prompt
object_bongard_semantics_output_schema = (
    object_bongard_shared_witness_semantics_output_schema
)
object_bongard_semantics_protocol_digest = (
    object_bongard_shared_witness_semantics_protocol_digest
)
object_bongard_semantics_source_digest = (
    object_bongard_shared_witness_semantics_source_digest
)


__all__ = (
    "GROUP_IDS",
    "GROUP_SIZE",
    "ObjectBongardSemanticArtifact",
    "ObjectBongardSemanticsError",
    "ObjectBongardSharedWitnessSemanticArtifact",
    "ObjectBongardSharedWitnessSemanticsError",
    "SHARED_WITNESS_CANDIDATE_COUNT",
    "SHARED_WITNESS_SEMANTIC_ARTIFACT_SCHEMA",
    "SHARED_WITNESS_SEMANTIC_PROTOCOL_ID",
    "describe_object_bongard_shared_witness_support",
    "describe_object_bongard_support",
    "object_bongard_shared_witness_semantics_output_schema",
    "object_bongard_shared_witness_semantics_prompt",
    "object_bongard_shared_witness_semantics_protocol_digest",
    "object_bongard_shared_witness_semantics_source_digest",
    "object_bongard_semantics_output_schema",
    "object_bongard_semantics_prompt",
    "object_bongard_semantics_protocol_digest",
    "object_bongard_semantics_source_digest",
    "verify_object_bongard_shared_witness_semantic_artifact",
    "verify_object_bongard_semantic_artifact",
)

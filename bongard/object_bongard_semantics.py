"""Profile-free visual proposal of two frozen positive soft-cue pairs.

The model sees two neutral groups of six support images and emits exactly two
ranked *forward* pairs of bounded positive visual phrases.  It cannot choose an
operator, threshold, polarity, formula, or executable code.  Each phrase is
typed and content-addressed; Python alone supplies the fixed ordinal observer,
deadband, dispositions, finite candidate family, and later selection rule.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.object_bongard_soft_cues import (
    ObjectBongardSoftCueError,
    ObjectBongardSoftCuePair,
    object_bongard_soft_cue_grammar_digest,
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


SEMANTIC_ARTIFACT_SCHEMA = "gkm.bongard-object-task-semantics.v3"
SEMANTIC_PROTOCOL_ID = (
    "bongard.object-task-semantics/two-ranked-positive-soft-cue-pairs-v5"
)
GROUP_IDS = ("group_0", "group_1")
GROUP_SIZE = 6
SOFT_CUE_CANDIDATE_COUNT = 2

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_TASK_ID = re.compile(r"(?:bd|ff|hd)_[A-Za-z0-9_.-]+\Z")


class ObjectBongardSemanticsError(ValueError):
    """A semantic turn, artifact, or replay commitment is malformed."""


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "model_can_propose_positive_soft_cue_text_only": True,
        "model_can_choose_operator_threshold_or_polarity": False,
        "explicit_semantic_negation_allowed": False,
        "feature_catalog_used": False,
        "soft_cue_text_is_observed_not_executed": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_or_decision": False,
        "lean_required_for_replay": False,
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


def object_bongard_semantics_prompt() -> str:
    return (
        "Inspect twelve drawings arranged as two neutral groups of six, named "
        "group_0_ref_00 through group_0_ref_05 and group_1_ref_00 through "
        "group_1_ref_05. Consider both groups jointly. Return exactly two "
        "ranked forward visual proposals named proposal_0 then proposal_1. "
        "Each proposal contains one cue for group_0 and one cue for group_1. "
        "A cue must state a visible invariant that recurs across all six "
        "members of its group and is not similarly characteristic of the six "
        "members of the opposite group. proposal_0 is your strongest pair. "
        "proposal_1 is the strongest genuinely alternate pair; it may reuse "
        "one good group cue when the cue for the opposite group changes, but "
        "the complete ordered pair must change. Inspect every drawing before "
        "writing either proposal. Prefer concrete parts, spelled-out counts, "
        "topology, angles, and relations over a bare resemblance term. Ignore "
        "pose, scale, location, and incidental stroke variation. Each cue field "
        "must contain one short positive atomic visible phrase that another "
        "vision observer can apply to an isolated drawing. Do not put a "
        "comparison or experimental role inside a cue. Cue text must not use "
        "no, not, without, lacking, absent, missing, and, or, than, versus, "
        "different, distinct, unlike, other, except, digits, or operator "
        "symbols. Spelled-out visible counts are allowed. Do not choose a "
        "threshold, polarity, weight, executable text, or hidden role. Python "
        "alone supplies the fixed observer scale, uncertainty semantics, "
        "finite executable family, and later selection procedure. Group names "
        "are neutral and do not indicate class polarity."
    )


def object_bongard_semantics_output_schema() -> dict[str, object]:
    pair = {
        "type": "object",
        "properties": {
            "group_0_cue_text": {"type": "string"},
            "group_1_cue_text": {"type": "string"},
        },
        "required": ["group_0_cue_text", "group_1_cue_text"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"proposal_0": pair, "proposal_1": pair},
        "required": ["proposal_0", "proposal_1"],
        "additionalProperties": False,
    }


def object_bongard_semantics_protocol_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-object-task-semantics-protocol.v4",
            "protocol_id": SEMANTIC_PROTOCOL_ID,
            "source_digest": object_bongard_semantics_source_digest(),
            "prompt_sha256": hashlib.sha256(
                object_bongard_semantics_prompt().encode("utf-8")
            ).hexdigest(),
            "output_schema_digest": canonical_digest(
                object_bongard_semantics_output_schema()
            ),
            "soft_cue_grammar_digest": object_bongard_soft_cue_grammar_digest(),
            "group_ids": list(GROUP_IDS),
            "images_per_group": GROUP_SIZE,
            "ranked_forward_proposal_count": SOFT_CUE_CANDIDATE_COUNT,
            "cue_pairs_are_ordered_group_0_over_group_1": True,
            "ordered_pairs_must_be_distinct": True,
            "one_group_cue_may_repeat_across_ranks": True,
            "feature_catalog_used": False,
            "cross_group_comparison_required": True,
            "cue_must_recur_within_named_group": True,
            "cue_must_be_more_characteristic_than_in_other_group": True,
            "independently_typical_cue_allowed": False,
            "group_names_encode_class_polarity": False,
            "soft_cue_text_is_typed_and_content_addressed": True,
            "observer_rubric_derivation": "exact-ordered-soft-cue-text-wrapper",
            "downstream_operationalization": (
                "four-predeclared-python-candidates-two-ranks-times-two-scopes"
            ),
            "semantic_parser_constructs_profiles": False,
            **_authority_data(),
        }
    )


def _parse_semantic_payload(
    payload: object,
) -> tuple[ObjectBongardSoftCuePair, ObjectBongardSoftCuePair]:
    """Parse exactly two ranked cue pairs without constructing a predicate."""

    if not isinstance(payload, Mapping) or set(payload) != {
        "proposal_0", "proposal_1"
    }:
        raise ObjectBongardSemanticsError("semantic payload fields differ")
    values: list[ObjectBongardSoftCuePair] = []
    for rank in range(SOFT_CUE_CANDIDATE_COUNT):
        row = payload[f"proposal_{rank}"]
        if (
            not isinstance(row, Mapping)
            or set(row) != {"group_0_cue_text", "group_1_cue_text"}
        ):
            raise ObjectBongardSemanticsError(
                f"semantic proposal {rank} fields differ"
            )
        try:
            values.append(
                ObjectBongardSoftCuePair.create(
                    rank,
                    row["group_0_cue_text"],
                    row["group_1_cue_text"],
                )
            )
        except (TypeError, ValueError, ObjectBongardSoftCueError) as exc:
            raise ObjectBongardSemanticsError("semantic soft cue is invalid") from exc
    if (
        values[0].group_0_cue.cue_digest,
        values[0].group_1_cue.cue_digest,
    ) == (
        values[1].group_0_cue.cue_digest,
        values[1].group_1_cue.cue_digest,
    ):
        raise ObjectBongardSemanticsError(
            "semantic proposals must contain distinct ordered cue pairs"
        )
    return tuple(values)  # type: ignore[return-value]


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
        "soft_cue_candidates": [
            item.to_data() for item in value.soft_cue_candidates
        ],
        "failure_code": value.failure_code,
        "failure_type": value.failure_type,
        "physical_call_count": 1,
        "vision_prose_defines_soft_cue_identity": True,
        "feature_catalog_constrains_identity_or_decision": False,
        "ranked_forward_candidate_count": SOFT_CUE_CANDIDATE_COUNT,
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
    soft_cue_candidates: tuple[ObjectBongardSoftCuePair, ...]
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
            "source_digest", "transport_source_digest",
            "model_digest", "model_catalog_digest", "no_tools_attestation_digest",
            "environment_digest", "artifact_digest",
        ):
            _digest(getattr(self, name), name)
        if (
            self.prompt_digest
            != hashlib.sha256(object_bongard_semantics_prompt().encode("utf-8")).hexdigest()
            or self.output_schema_digest
            != canonical_digest(object_bongard_semantics_output_schema())
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
                or len(self.soft_cue_candidates) != SOFT_CUE_CANDIDATE_COUNT
                or tuple(
                    item.candidate_rank for item in self.soft_cue_candidates
                )
                != (0, 1)
                or len(
                    {item.pair_digest for item in self.soft_cue_candidates}
                )
                != SOFT_CUE_CANDIDATE_COUNT
                or self.failure_code is not None
                or self.failure_type is not None
            ):
                raise ObjectBongardSemanticsError("successful semantic artifact differs")
            if any(
                not isinstance(item, ObjectBongardSoftCuePair)
                for item in self.soft_cue_candidates
            ):
                raise TypeError("semantic soft cue candidates have the wrong type")
        elif parser_error:
            if (
                self.model_payload is None
                or self.receipt is None
                or self.soft_cue_candidates
                or self.failure_code != "semantic_payload_rejected"
                or self.failure_type is None
            ):
                raise ObjectBongardSemanticsError("parser-error semantic artifact differs")
        elif transport_error:
            if (
                self.model_payload is not None
                or self.receipt is not None
                or self.soft_cue_candidates
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
            "transport_source_digest", "model",
            "reasoning_effort", "model_digest", "expected_launcher_digest",
            "cloud_policy_cache_binding", "model_catalog_digest",
            "no_tools_attestation_digest", "environment_digest", "model_payload",
            "receipt", "receipt_identity", "soft_cue_candidates",
            "failure_code", "failure_type", "physical_call_count",
            "vision_prose_defines_soft_cue_identity",
            "feature_catalog_constrains_identity_or_decision",
            "ranked_forward_candidate_count", *_authority_data(),
            "artifact_digest",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ObjectBongardSemanticsError("semantic artifact fields differ")
        if (
            value["schema"] != SEMANTIC_ARTIFACT_SCHEMA
            or value["physical_call_count"] != 1
            or value["vision_prose_defines_soft_cue_identity"] is not True
            or value["feature_catalog_constrains_identity_or_decision"] is not False
            or value["ranked_forward_candidate_count"]
            != SOFT_CUE_CANDIDATE_COUNT
            or any(value[key] != item for key, item in _authority_data().items())
            or not isinstance(value["group_panel_ids"], list)
            or len(value["group_panel_ids"]) != 2
            or not isinstance(value["presentation"], list)
            or not isinstance(value["soft_cue_candidates"], list)
        ):
            raise ObjectBongardSemanticsError("semantic artifact policy differs")
        try:
            soft_cue_candidates = tuple(
                ObjectBongardSoftCuePair.from_data(item)
                for item in value["soft_cue_candidates"]
            )
        except (TypeError, ValueError) as exc:
            raise ObjectBongardSemanticsError(
                "semantic soft cue slate is invalid"
            ) from exc
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
            soft_cue_candidates=soft_cue_candidates,
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
    """Produce exactly two ranked positive soft-cue pairs in one vision call."""

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
    schema = object_bongard_semantics_output_schema()
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
            soft_cue_candidates=(),
            failure_code="semantic_transport_failed",
            failure_type=type(exc).__name__,
        )
    try:
        soft_cue_candidates = _parse_semantic_payload(payload)
    except (TypeError, ValueError):
        return _build_artifact(
            **common,
            status=PrototypeSceneObserverStatus.PARSER_ERROR,
            model_payload=payload,
            receipt=receipt,
            receipt_identity=receipt.receipt_digest,
            soft_cue_candidates=(),
            failure_code="semantic_payload_rejected",
            failure_type="ObjectBongardSoftCueError",
        )
    return _build_artifact(
        **common,
        status=PrototypeSceneObserverStatus.SUCCESS,
        model_payload=payload,
        receipt=receipt,
        receipt_identity=receipt.receipt_digest,
        soft_cue_candidates=soft_cue_candidates,
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
    """Cold-replay exact support bytes, receipt, and typed cue slate."""

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
            object_bongard_semantics_output_schema(),
            artifact.model_payload,
        )
    if artifact.status is PrototypeSceneObserverStatus.SUCCESS:
        soft_cue_candidates = _parse_semantic_payload(artifact.model_payload)
        if soft_cue_candidates != artifact.soft_cue_candidates:
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
    "SOFT_CUE_CANDIDATE_COUNT",
    "ObjectBongardSemanticArtifact",
    "ObjectBongardSemanticsError",
    "SEMANTIC_ARTIFACT_SCHEMA",
    "describe_object_bongard_support",
    "object_bongard_semantics_prompt",
    "object_bongard_semantics_output_schema",
    "object_bongard_semantics_protocol_digest",
    "object_bongard_semantics_source_digest",
    "verify_object_bongard_semantic_artifact",
)

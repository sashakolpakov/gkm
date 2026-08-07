"""Headless object-hypothesis observer with Python-authoritative replay.

The raw entry point observes any committed scene without references, prose, or
profiles.  It produces a receipt-bound full-catalog shard bundle.  The legacy
prototype-pair wrapper may subsequently evaluate fixed Python profiles against
that evidence, without exposing those profiles to the scene observer.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass, field
import hashlib
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.prototype_object_hypotheses import (
    ObjectHypothesisPacket,
    extract_object_hypotheses,
    object_hypothesis_extractor_artifact_digest,
    render_object_hypothesis_atlas,
    verify_object_hypothesis_packet,
)
from bongard.prototype_object_profiles import (
    OBJECT_FEATURE_CATALOG_DIGEST,
    ObjectFeatureCell,
    ObjectLocalObservationPacket,
    ObjectProfile,
    ObjectProfileEvaluation,
    evaluate_object_profile,
    verify_object_profile_evaluation,
)
from bongard import prototype_scene_observer as _legacy


PROTOTYPE_REFERENCE_CATALOG_SCHEMA = _legacy.PROTOTYPE_REFERENCE_CATALOG_SCHEMA
PROTOTYPE_RUBRIC_DESCRIPTION_ARTIFACT_SCHEMA = (
    "gkm.bongard-object-feature-family-description-artifact.v2"
)
PROTOTYPE_SCENE_OBSERVER_ARTIFACT_SCHEMA = (
    "gkm.bongard-object-scene-observer-artifact.v2"
)
PROTOTYPE_OBJECT_FEATURE_SHARD_ARTIFACT_SCHEMA = (
    "gkm.bongard-object-feature-shard-artifact.v1"
)
PROTOTYPE_OBJECT_FEATURE_OBSERVER_ARTIFACT_SCHEMA = (
    "gkm.bongard-raw-object-feature-observer-artifact.v1"
)
PROTOTYPE_OBJECT_FEATURE_OBSERVER_PROTOCOL_ID = (
    "bongard.raw-object-feature-observer/profile-blind-atlas-shards-v1"
)
PROTOTYPE_SCENE_OBSERVER_PROTOCOL_ID = (
    "bongard.prototype-object-scene-observer/profile-blind-atlas-shards-v2"
)
PROTOTYPE_GROUP_IDS = _legacy.PROTOTYPE_GROUP_IDS
PPM_SCALE = _legacy.PPM_SCALE

PrototypeSceneObserverError = _legacy.PrototypeSceneObserverError
PrototypeScenePayloadError = _legacy.PrototypeScenePayloadError
PrototypeSceneObserverStatus = _legacy.PrototypeSceneObserverStatus
PrototypeRubricState = _legacy.PrototypeRubricState
PrototypeSceneScoreState = _legacy.PrototypeSceneScoreState
PrototypeSceneDescriptionState = _legacy.PrototypeSceneDescriptionState
PrototypeImageIdentity = _legacy.PrototypeImageIdentity
PrototypeReferenceBinding = _legacy.PrototypeReferenceBinding
PrototypeReferenceCatalog = _legacy.PrototypeReferenceCatalog
PrototypeRubric = _legacy.PrototypeRubric
PrototypeSceneDescriptionObservation = _legacy.PrototypeSceneDescriptionObservation
PrototypeSceneScore = _legacy.PrototypeSceneScore
NamedImageTransport = _legacy.NamedImageTransport
CloudPolicyCacheSnapshot = _legacy.CloudPolicyCacheSnapshot
CodexModelCatalogSnapshot = _legacy.CodexModelCatalogSnapshot
CodexNoToolsAttestation = _legacy.CodexNoToolsAttestation
run_codex_named_images_structured = _legacy.run_codex_named_images_structured

build_prototype_reference_catalog = _legacy.build_prototype_reference_catalog
verify_prototype_reference_catalog = _legacy.verify_prototype_reference_catalog


def prototype_scene_observer_source_digest() -> str:
    """Return the immutable loaded source identity for this active observer."""

    return _LOADED_SOURCE_SHA256


def prototype_scene_transport_source_digest() -> str:
    return _legacy.prototype_scene_transport_source_digest()


def prototype_scene_observer_model_digest(
    model: str, reasoning_effort: str
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-object-observer-model-identity.v1",
            "model": model,
            "reasoning_effort": reasoning_effort,
            "legacy_transport_model_digest": (
                _legacy.prototype_scene_observer_model_digest(
                    model, reasoning_effort
                )
            ),
        }
    )


def _authority_data() -> dict[str, object]:
    return {
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_decision": False,
    }


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SCENE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}\Z")


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PrototypeSceneObserverError(f"{label} must be a lowercase sha256")
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise PrototypeSceneObserverError(f"{label} must be a sha256: address")
    return value


def _exact(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise PrototypeSceneObserverError(f"{label} fields differ from schema")
    return value


def _canonical_payload(value: object) -> dict[str, Any]:
    return _legacy._canonical_payload(value)


def prototype_scene_observer_environment_digest(
    *,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str | None,
    cloud_policy_cache_binding: str,
    model_catalog_digest: str,
    no_tools_attestation_digest: str,
) -> str:
    if expected_launcher_digest is not None:
        _digest(expected_launcher_digest, "launcher digest")
    if cloud_policy_cache_binding != "absent":
        _address(cloud_policy_cache_binding, "policy cache binding")
    return canonical_digest(
        {
            "schema": "gkm.bongard-object-observer-environment.v1",
            "model_digest": prototype_scene_observer_model_digest(
                model, reasoning_effort
            ),
            "expected_launcher_digest": expected_launcher_digest,
            "cloud_policy_cache_binding": cloud_policy_cache_binding,
            "model_catalog_digest": _digest(
                model_catalog_digest, "model catalog digest"
            ),
            "no_tools_attestation_digest": _digest(
                no_tools_attestation_digest, "no-tools attestation digest"
            ),
            "observer_source_digest": prototype_scene_observer_source_digest(),
            "transport_source_digest": prototype_scene_transport_source_digest(),
            "python_authority": True,
            "lean_required": False,
        }
    )


def _object_protocol() -> Any:
    # Imported lazily so this observer and the shared closed protocol can land
    # independently without introducing an import cycle.
    from bongard import prototype_object_observer_protocol

    return prototype_object_observer_protocol


def prototype_rubric_description_prompt() -> str:
    return _object_protocol().prototype_object_description_prompt()


def prototype_rubric_description_output_schema() -> dict[str, object]:
    return _object_protocol().prototype_object_description_output_schema()


def prototype_rubric_description_protocol_digest() -> str:
    return _object_protocol().prototype_object_description_protocol_digest()


def prototype_scene_scoring_protocol_digest() -> str:
    """Stable campaign precommit identity; per-packet identities remain nested."""

    protocol = _object_protocol()
    return canonical_digest(
        {
            "schema": "gkm.bongard-object-scene-observer-protocol.v1",
            "protocol_id": PROTOTYPE_SCENE_OBSERVER_PROTOCOL_ID,
            "observer_source_digest": prototype_scene_observer_source_digest(),
            "hypothesis_extractor_artifact_digest": (
                object_hypothesis_extractor_artifact_digest()
            ),
            "object_protocol_source_digest": (
                protocol.prototype_object_protocol_source_digest()
            ),
            "feature_protocol_family_digest": (
                protocol.prototype_object_feature_protocol_family_digest()
            ),
            "feature_output_schema_digest": canonical_digest(
                protocol.prototype_object_feature_output_schema()
            ),
            "feature_catalog_digest": OBJECT_FEATURE_CATALOG_DIGEST,
            "formula": "two frozen positive profiles; same-hypothesis conjunction",
            "scene_model_inputs": "opaque exhaustive atlas only",
            "profile_blind": True,
            "reference_blind": True,
            "python_authority": True,
            "lean_required": False,
        }
    )


def prototype_object_feature_observer_protocol_digest() -> str:
    """Profile-free raw observer identity for generic Bongard scenes."""

    protocol = _object_protocol()
    return canonical_digest(
        {
            "schema": "gkm.bongard-raw-object-feature-observer-protocol.v1",
            "protocol_id": PROTOTYPE_OBJECT_FEATURE_OBSERVER_PROTOCOL_ID,
            "observer_source_digest": prototype_scene_observer_source_digest(),
            "transport_source_digest": prototype_scene_transport_source_digest(),
            "hypothesis_extractor_artifact_digest": (
                object_hypothesis_extractor_artifact_digest()
            ),
            "object_protocol_source_digest": (
                protocol.prototype_object_protocol_source_digest()
            ),
            "feature_protocol_family_digest": (
                protocol.prototype_object_feature_protocol_family_digest()
            ),
            "feature_catalog_digest": OBJECT_FEATURE_CATALOG_DIGEST,
            "input": "one exact committed scene PNG",
            "output": "full-catalog typed local observation packets",
            "description_or_profile_required": False,
            "profile_blind": True,
            "reference_blind": True,
            "physical_calls": "one staged transport invocation per canonical shard",
            "shard_admission_scope": "internal_scene_bundle_only",
            **_authority_data(),
        }
    )


def _receipt_to_data(receipt: object | None) -> object:
    return None if receipt is None else receipt.to_dict()  # type: ignore[union-attr]


def _receipt_from_data(value: object) -> object | None:
    return None if value is None else _legacy._receipt_from_data(value)


def _failure_digest(
    phase: str,
    status: PrototypeSceneObserverStatus,
    failure_code: str | None,
    failure_type: str | None,
    payload: Mapping[str, Any] | None,
) -> str | None:
    return _legacy._failure_digest(
        phase, status, failure_code, failure_type, payload
    )


def _error_rubrics(reason: str, kind: str) -> tuple[PrototypeRubric, ...]:
    return _legacy._rubric_error_pair(reason, kind)


def _error_scores(reason: str, kind: str) -> tuple[PrototypeSceneScore, ...]:
    return _legacy._score_error_pair(reason, kind)


def _defined_rubrics(audit_texts: Sequence[object]) -> tuple[PrototypeRubric, ...]:
    from bongard.prototype_pair_cohort import OPAQUE_TAG_IDS

    if len(audit_texts) != 2:
        raise PrototypeScenePayloadError("description did not exhaust both groups")
    rows: list[PrototypeRubric] = []
    for tag_id, group_id, audit in zip(
        OPAQUE_TAG_IDS, PROTOTYPE_GROUP_IDS, audit_texts, strict=True
    ):
        prose = getattr(audit, "prose", None)
        state = getattr(audit, "state", None)
        state_value = getattr(state, "value", state)
        if state_value == "defined" and isinstance(prose, str):
            try:
                rows.append(PrototypeRubric.defined(tag_id, group_id, prose))
                continue
            except (TypeError, ValueError):
                pass
        rows.append(
            PrototypeRubric.error(
                tag_id,
                group_id,
                "audit_prose_unavailable",
                "PrototypeObjectAuditTextUnavailable",
            )
        )
    return tuple(rows)


def _common_data(artifact: object) -> dict[str, object]:
    return {
        "status": artifact.status.value,  # type: ignore[attr-defined]
        "plan_digest": artifact.plan_digest,  # type: ignore[attr-defined]
        "catalog_digest": artifact.catalog_digest,  # type: ignore[attr-defined]
        "presentation": [x.to_data() for x in artifact.presentation],  # type: ignore[attr-defined]
        "prompt_digest": artifact.prompt_digest,  # type: ignore[attr-defined]
        "output_schema_digest": artifact.output_schema_digest,  # type: ignore[attr-defined]
        "protocol_digest": artifact.protocol_digest,  # type: ignore[attr-defined]
        "source_digest": artifact.source_digest,  # type: ignore[attr-defined]
        "transport_source_digest": artifact.transport_source_digest,  # type: ignore[attr-defined]
        "model": artifact.model,  # type: ignore[attr-defined]
        "reasoning_effort": artifact.reasoning_effort,  # type: ignore[attr-defined]
        "model_digest": artifact.model_digest,  # type: ignore[attr-defined]
        "expected_launcher_digest": artifact.expected_launcher_digest,  # type: ignore[attr-defined]
        "cloud_policy_cache_binding": artifact.cloud_policy_cache_binding,  # type: ignore[attr-defined]
        "model_catalog_digest": artifact.model_catalog_digest,  # type: ignore[attr-defined]
        "no_tools_attestation_digest": artifact.no_tools_attestation_digest,  # type: ignore[attr-defined]
        "environment_digest": artifact.environment_digest,  # type: ignore[attr-defined]
        "model_payload": artifact.model_payload,  # type: ignore[attr-defined]
        "receipt": _receipt_to_data(artifact.receipt),  # type: ignore[attr-defined]
        "failure_code": artifact.failure_code,  # type: ignore[attr-defined]
        "failure_type": artifact.failure_type,  # type: ignore[attr-defined]
        "failure_digest": artifact.failure_digest,  # type: ignore[attr-defined]
    }


_COMMON_FIELDS = {
    "status", "plan_digest", "catalog_digest", "presentation",
    "prompt_digest", "output_schema_digest", "protocol_digest",
    "source_digest", "transport_source_digest", "model", "reasoning_effort",
    "model_digest", "expected_launcher_digest", "cloud_policy_cache_binding",
    "model_catalog_digest", "no_tools_attestation_digest", "environment_digest",
    "model_payload", "receipt", "failure_code", "failure_type", "failure_digest",
}


def _validate_common(
    artifact: object,
    *,
    expected_protocol_digest: str,
    phase: str,
    aggregate_model_output: bool = False,
) -> None:
    status = artifact.status  # type: ignore[attr-defined]
    if not isinstance(status, PrototypeSceneObserverStatus):
        raise TypeError("artifact status has the wrong type")
    _address(artifact.plan_digest, "plan digest")  # type: ignore[attr-defined]
    for name in (
        "catalog_digest", "prompt_digest", "output_schema_digest",
        "protocol_digest", "source_digest", "transport_source_digest",
        "model_digest", "model_catalog_digest", "no_tools_attestation_digest",
        "environment_digest", "artifact_digest",
    ):
        _digest(getattr(artifact, name), name)
    if artifact.protocol_digest != expected_protocol_digest:  # type: ignore[attr-defined]
        raise PrototypeSceneObserverError(f"{phase} protocol digest differs")
    if artifact.source_digest != prototype_scene_observer_source_digest():  # type: ignore[attr-defined]
        raise PrototypeSceneObserverError(f"{phase} source digest differs")
    if artifact.transport_source_digest != prototype_scene_transport_source_digest():  # type: ignore[attr-defined]
        raise PrototypeSceneObserverError(f"{phase} transport digest differs")
    if artifact.model_digest != prototype_scene_observer_model_digest(  # type: ignore[attr-defined]
        artifact.model, artifact.reasoning_effort  # type: ignore[attr-defined]
    ):
        raise PrototypeSceneObserverError(f"{phase} model digest differs")
    if artifact.environment_digest != prototype_scene_observer_environment_digest(  # type: ignore[attr-defined]
        model=artifact.model,  # type: ignore[attr-defined]
        reasoning_effort=artifact.reasoning_effort,  # type: ignore[attr-defined]
        expected_launcher_digest=artifact.expected_launcher_digest,  # type: ignore[attr-defined]
        cloud_policy_cache_binding=artifact.cloud_policy_cache_binding,  # type: ignore[attr-defined]
        model_catalog_digest=artifact.model_catalog_digest,  # type: ignore[attr-defined]
        no_tools_attestation_digest=artifact.no_tools_attestation_digest,  # type: ignore[attr-defined]
    ):
        raise PrototypeSceneObserverError(f"{phase} environment digest differs")
    if not isinstance(artifact.presentation, tuple) or any(  # type: ignore[attr-defined]
        not isinstance(item, PrototypeImageIdentity)
        for item in artifact.presentation  # type: ignore[attr-defined]
    ):
        raise TypeError("presentation must be a typed tuple")
    payload = artifact.model_payload  # type: ignore[attr-defined]
    if payload is not None:
        payload = _canonical_payload(payload)
        object.__setattr__(artifact, "model_payload", payload)
    expected_failure = _failure_digest(
        phase,
        status,
        artifact.failure_code,  # type: ignore[attr-defined]
        artifact.failure_type,  # type: ignore[attr-defined]
        payload,
    )
    if artifact.failure_digest != expected_failure:  # type: ignore[attr-defined]
        raise PrototypeSceneObserverError(f"{phase} failure digest differs")
    if aggregate_model_output:
        if artifact.receipt is not None or payload is not None:  # type: ignore[attr-defined]
            raise PrototypeSceneObserverError(
                f"aggregate {phase} must store output in child artifacts"
            )
    elif status is PrototypeSceneObserverStatus.SUCCESS:
        if payload is None or artifact.receipt is None:  # type: ignore[attr-defined]
            raise PrototypeSceneObserverError(f"successful {phase} lacks receipt/payload")
    elif status is PrototypeSceneObserverStatus.PARSER_ERROR:
        if payload is None or artifact.receipt is None:  # type: ignore[attr-defined]
            raise PrototypeSceneObserverError(f"parser-error {phase} lacks receipt/payload")
    elif artifact.receipt is not None or payload is not None:  # type: ignore[attr-defined]
        raise PrototypeSceneObserverError(f"failed {phase} claims model output")


def _description_preimage(
    artifact: "PrototypeRubricDescriptionArtifact",
) -> dict[str, object]:
    return {
        "schema": PROTOTYPE_RUBRIC_DESCRIPTION_ARTIFACT_SCHEMA,
        **_common_data(artifact),
        "feature_families": [list(item) for item in artifact.feature_families],
        "profiles": [item.to_data() for item in artifact.profiles],
        "rubrics": [item.to_data() for item in artifact.rubrics],
        "runtime_authority": _authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PrototypeRubricDescriptionArtifact:
    status: PrototypeSceneObserverStatus
    plan_digest: str
    catalog_digest: str
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
    failure_code: str | None
    failure_type: str | None
    failure_digest: str | None
    feature_families: tuple[tuple[str, ...], ...]
    profiles: tuple[ObjectProfile, ...]
    rubrics: tuple[PrototypeRubric, ...]
    artifact_digest: str
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_common(
            self,
            expected_protocol_digest=prototype_rubric_description_protocol_digest(),
            phase="object-reference-description",
        )
        expected_names = tuple(
            f"{group}_ref_{index}.png"
            for group in PROTOTYPE_GROUP_IDS
            for index in range(3)
        )
        if tuple(item.name for item in self.presentation) != expected_names:
            raise PrototypeSceneObserverError("description presentation differs")
        if self.prompt_digest != hashlib.sha256(
            prototype_rubric_description_prompt().encode("utf-8")
        ).hexdigest():
            raise PrototypeSceneObserverError("description prompt digest differs")
        if self.output_schema_digest != canonical_digest(
            prototype_rubric_description_output_schema()
        ):
            raise PrototypeSceneObserverError("description schema digest differs")
        if not isinstance(self.profiles, tuple) or any(
            not isinstance(item, ObjectProfile) for item in self.profiles
        ):
            raise TypeError("profiles must be a typed tuple")
        if not isinstance(self.feature_families, tuple) or any(
            not isinstance(item, tuple) or any(not isinstance(value, str) for value in item)
            for item in self.feature_families
        ):
            raise TypeError("feature families must be a tuple of string tuples")
        if not isinstance(self.rubrics, tuple) or len(self.rubrics) != 2:
            raise PrototypeSceneObserverError("audit rubrics must exhaust both groups")
        if self.status is PrototypeSceneObserverStatus.SUCCESS:
            if len(self.profiles) != 2 or tuple(
                item.profile_id for item in self.profiles
            ) != PROTOTYPE_GROUP_IDS:
                raise PrototypeSceneObserverError("description profiles differ from group order")
            parsed = _object_protocol().parse_prototype_object_description_payload(
                self.model_payload
            )
            if (
                tuple(parsed.feature_families) != self.feature_families
                or
                tuple(parsed.profiles) != self.profiles
                or _defined_rubrics(parsed.audit_rubrics) != self.rubrics
            ):
                raise PrototypeSceneObserverError("description payload replay differs")
        else:
            if self.feature_families or self.profiles:
                raise PrototypeSceneObserverError("failed description carries feature families")
            expected_errors = {
                PrototypeSceneObserverStatus.PARSER_ERROR: _error_rubrics(
                    "observer_payload_rejected", "PrototypeScenePayloadError"
                ),
                PrototypeSceneObserverStatus.TRANSPORT_ERROR: _error_rubrics(
                    "observer_transport_failed", "PrototypeSceneTransportFailure"
                ),
                PrototypeSceneObserverStatus.INTERNAL_ERROR: _error_rubrics(
                    "observer_internal_error", "PrototypeSceneInternalError"
                ),
            }
            if self.status not in expected_errors or self.rubrics != expected_errors[self.status]:
                raise PrototypeSceneObserverError("failed description rubrics differ")
            if self.status is PrototypeSceneObserverStatus.PARSER_ERROR:
                try:
                    _object_protocol().parse_prototype_object_description_payload(
                        self.model_payload
                    )
                except (TypeError, ValueError):
                    pass
                else:
                    raise PrototypeSceneObserverError("parser-error payload is admissible")
        computed = canonical_digest(_description_preimage(self))
        if self.artifact_digest != computed:
            raise PrototypeSceneObserverError("description artifact digest differs")
        object.__setattr__(self, "_sealed_digest", computed)

    def to_data(self) -> dict[str, object]:
        return {**_description_preimage(self), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        expected_artifact_digest: str | None = None,
    ) -> "PrototypeRubricDescriptionArtifact":
        raw = _exact(
            value,
            {
                "schema", *_COMMON_FIELDS, "feature_families", "profiles", "rubrics",
                "runtime_authority", "artifact_digest",
            },
            "object description artifact",
        )
        if raw["schema"] != PROTOTYPE_RUBRIC_DESCRIPTION_ARTIFACT_SCHEMA:
            raise PrototypeSceneObserverError("unsupported description artifact")
        if raw["runtime_authority"] != _authority_data():
            raise PrototypeSceneObserverError("description authority differs")
        if not isinstance(raw["presentation"], list) or not isinstance(raw["feature_families"], list) or not isinstance(raw["profiles"], list) or not isinstance(raw["rubrics"], list):
            raise PrototypeSceneObserverError("description child collections are invalid")
        if any(not isinstance(item, list) for item in raw["feature_families"]):
            raise PrototypeSceneObserverError("description feature families are invalid")
        result = cls(
            status=PrototypeSceneObserverStatus(raw["status"]),
            plan_digest=raw["plan_digest"],
            catalog_digest=raw["catalog_digest"],
            presentation=tuple(PrototypeImageIdentity.from_data(x) for x in raw["presentation"]),
            prompt_digest=raw["prompt_digest"],
            output_schema_digest=raw["output_schema_digest"],
            protocol_digest=raw["protocol_digest"],
            source_digest=raw["source_digest"],
            transport_source_digest=raw["transport_source_digest"],
            model=raw["model"],
            reasoning_effort=raw["reasoning_effort"],
            model_digest=raw["model_digest"],
            expected_launcher_digest=raw["expected_launcher_digest"],
            cloud_policy_cache_binding=raw["cloud_policy_cache_binding"],
            model_catalog_digest=raw["model_catalog_digest"],
            no_tools_attestation_digest=raw["no_tools_attestation_digest"],
            environment_digest=raw["environment_digest"],
            model_payload=None if raw["model_payload"] is None else _canonical_payload(raw["model_payload"]),
            receipt=_receipt_from_data(raw["receipt"]),
            failure_code=raw["failure_code"],
            failure_type=raw["failure_type"],
            failure_digest=raw["failure_digest"],
            feature_families=tuple(tuple(item) for item in raw["feature_families"]),
            profiles=tuple(ObjectProfile.from_data(x) for x in raw["profiles"]),
            rubrics=tuple(PrototypeRubric.from_data(x) for x in raw["rubrics"]),
            artifact_digest=raw["artifact_digest"],
        )
        if expected_artifact_digest is not None and result.artifact_digest != _digest(expected_artifact_digest, "expected description artifact digest"):
            raise PrototypeSceneObserverError("description artifact commitment differs")
        if result.to_data() != dict(raw):
            raise PrototypeSceneObserverError("description artifact is not canonical")
        return result

    def assert_untampered(self) -> None:
        computed = canonical_digest(_description_preimage(self))
        if computed != self.artifact_digest or computed != self._sealed_digest:
            raise PrototypeSceneObserverError("description artifact changed after sealing")


def _runtime_identities(
    *,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str | None,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
) -> tuple[str, str, str]:
    policy = _legacy._policy_cache_binding(cloud_policy_cache_snapshot)
    model_catalog_digest, no_tools_digest = _legacy._validate_no_tools_runtime(
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=policy,
    )
    return policy, model_catalog_digest, no_tools_digest


def _build_description_artifact(
    *,
    status: PrototypeSceneObserverStatus,
    catalog: PrototypeReferenceCatalog,
    identities: tuple[PrototypeImageIdentity, ...],
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str | None,
    cloud_policy_cache_binding: str,
    model_catalog_digest: str,
    no_tools_attestation_digest: str,
    payload: Mapping[str, Any] | None,
    receipt: object | None,
    failure_code: str | None,
    failure_type: str | None,
    feature_families: tuple[tuple[str, ...], ...],
    profiles: tuple[ObjectProfile, ...],
    rubrics: tuple[PrototypeRubric, ...],
) -> PrototypeRubricDescriptionArtifact:
    prompt = prototype_rubric_description_prompt()
    schema = prototype_rubric_description_output_schema()
    canonical_payload = None if payload is None else _canonical_payload(payload)
    values: dict[str, object] = {
        "status": status,
        "plan_digest": catalog.plan_digest,
        "catalog_digest": catalog.catalog_digest,
        "presentation": identities,
        "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "output_schema_digest": canonical_digest(schema),
        "protocol_digest": prototype_rubric_description_protocol_digest(),
        "source_digest": prototype_scene_observer_source_digest(),
        "transport_source_digest": prototype_scene_transport_source_digest(),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "model_digest": prototype_scene_observer_model_digest(model, reasoning_effort),
        "expected_launcher_digest": expected_launcher_digest,
        "cloud_policy_cache_binding": cloud_policy_cache_binding,
        "model_catalog_digest": model_catalog_digest,
        "no_tools_attestation_digest": no_tools_attestation_digest,
        "environment_digest": prototype_scene_observer_environment_digest(
            model=model,
            reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=cloud_policy_cache_binding,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_attestation_digest,
        ),
        "model_payload": canonical_payload,
        "receipt": receipt,
        "failure_code": failure_code,
        "failure_type": failure_type,
        "failure_digest": _failure_digest(
            "object-reference-description",
            status,
            failure_code,
            failure_type,
            canonical_payload,
        ),
        "feature_families": feature_families,
        "profiles": profiles,
        "rubrics": rubrics,
    }
    provisional = object.__new__(PrototypeRubricDescriptionArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return PrototypeRubricDescriptionArtifact(
        **values,  # type: ignore[arg-type]
        artifact_digest=canonical_digest(_description_preimage(provisional)),
    )


def describe_prototype_references(
    catalog: PrototypeReferenceCatalog,
    prototype_png_by_panel_id: Mapping[str, bytes],
    *,
    expected_catalog_digest: str,
    model: str,
    reasoning_effort: str = "medium",
    minutes: int = 15,
    verbose: bool = False,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    expected_launcher_digest: str | None = None,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    transport: NamedImageTransport = run_codex_named_images_structured,
) -> PrototypeRubricDescriptionArtifact:
    """Freeze two closed profiles from the exact six neutral references."""

    if not isinstance(catalog, PrototypeReferenceCatalog):
        raise TypeError("catalog must be PrototypeReferenceCatalog")
    if catalog.catalog_digest != _digest(expected_catalog_digest, "expected catalog digest"):
        raise PrototypeSceneObserverError("reference catalog differs from commitment")
    if not callable(transport):
        raise TypeError("transport must be callable")
    presentation = _legacy._reference_presentation(catalog, prototype_png_by_panel_id)
    identities = _legacy._image_identities(presentation)
    if identities != catalog.presentation:
        raise PrototypeSceneObserverError("reference presentation differs from catalog")
    policy, model_catalog_digest, no_tools_digest = _runtime_identities(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    prompt = prototype_rubric_description_prompt()
    schema = prototype_rubric_description_output_schema()
    _legacy.validate_codex_strict_output_schema(schema)
    from bongard.prototype_pair_cohort import OPAQUE_TAG_IDS

    _legacy._assert_model_visible_boundary(
        prompt,
        schema,
        tuple(name for name, _ in presentation),
        hidden_values=(
            catalog.plan_digest,
            *OPAQUE_TAG_IDS,
            *(item.source_panel_id for item in catalog.bindings),
        ),
    )
    try:
        payload, receipt = _legacy._stage_and_call(
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
            transport=transport,
        )
    except Exception as exc:
        return _build_description_artifact(
            status=PrototypeSceneObserverStatus.TRANSPORT_ERROR,
            catalog=catalog,
            identities=identities,
            model=model,
            reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=policy,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_digest,
            payload=None,
            receipt=None,
            failure_code="transport_failed",
            failure_type=_legacy._exception_type(exc),
            feature_families=(),
            profiles=(),
            rubrics=_error_rubrics("observer_transport_failed", "PrototypeSceneTransportFailure"),
        )
    try:
        parsed = _object_protocol().parse_prototype_object_description_payload(payload)
        profiles = tuple(parsed.profiles)
        feature_families = tuple(parsed.feature_families)
        rubrics = _defined_rubrics(parsed.audit_rubrics)
        return _build_description_artifact(
            status=PrototypeSceneObserverStatus.SUCCESS,
            catalog=catalog,
            identities=identities,
            model=model,
            reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=policy,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_digest,
            payload=payload,
            receipt=receipt,
            failure_code=None,
            failure_type=None,
            feature_families=feature_families,
            profiles=profiles,
            rubrics=rubrics,
        )
    except (TypeError, ValueError):
        return _build_description_artifact(
            status=PrototypeSceneObserverStatus.PARSER_ERROR,
            catalog=catalog,
            identities=identities,
            model=model,
            reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=policy,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_digest,
            payload=payload,
            receipt=receipt,
            failure_code="payload_rejected",
            failure_type="PrototypeScenePayloadError",
            feature_families=(),
            profiles=(),
            rubrics=_error_rubrics("observer_payload_rejected", "PrototypeScenePayloadError"),
        )


def seal_prototype_rubric_description_internal_error(
    catalog: PrototypeReferenceCatalog,
    prototype_png_by_panel_id: Mapping[str, bytes],
    *,
    expected_catalog_digest: str,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str | None,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    exception: Exception,
) -> PrototypeRubricDescriptionArtifact:
    if not isinstance(exception, Exception):
        raise TypeError("exception must be Exception")
    if catalog.catalog_digest != _digest(expected_catalog_digest, "expected catalog digest"):
        raise PrototypeSceneObserverError("reference catalog differs from commitment")
    presentation = _legacy._reference_presentation(catalog, prototype_png_by_panel_id)
    identities = _legacy._image_identities(presentation)
    policy, model_catalog_digest, no_tools_digest = _runtime_identities(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    return _build_description_artifact(
        status=PrototypeSceneObserverStatus.INTERNAL_ERROR,
        catalog=catalog,
        identities=identities,
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=policy,
        model_catalog_digest=model_catalog_digest,
        no_tools_attestation_digest=no_tools_digest,
        payload=None,
        receipt=None,
        failure_code="observer_internal_error",
        failure_type=_legacy._exception_type(exception),
        feature_families=(),
        profiles=(),
        rubrics=_error_rubrics("observer_internal_error", "PrototypeSceneInternalError"),
    )


def _verify_receipt(
    receipt: object,
    presentation: Sequence[tuple[str, bytes]],
    prompt: str,
    schema: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    with tempfile.TemporaryDirectory(prefix="bongard-object-observer-replay-") as raw:
        paths: list[str] = []
        for name, data in presentation:
            target = Path(raw) / name
            target.write_bytes(data)
            paths.append(str(target.resolve()))
        _legacy.validate_codex_named_image_receipt(
            receipt,
            prompt,
            tuple(paths),
            tuple(name for name, _ in presentation),
            schema,
            payload,
        )


def verify_prototype_rubric_description_artifact(
    artifact: PrototypeRubricDescriptionArtifact,
    catalog: PrototypeReferenceCatalog,
    prototype_png_by_panel_id: Mapping[str, bytes],
    *,
    expected_catalog_digest: str,
    expected_artifact_digest: str,
) -> PrototypeRubricDescriptionArtifact:
    if not isinstance(artifact, PrototypeRubricDescriptionArtifact):
        raise TypeError("artifact must be PrototypeRubricDescriptionArtifact")
    artifact.assert_untampered()
    if catalog.catalog_digest != _digest(expected_catalog_digest, "expected catalog digest"):
        raise PrototypeSceneObserverError("catalog differs from commitment")
    presentation = _legacy._reference_presentation(catalog, prototype_png_by_panel_id)
    if (
        artifact.plan_digest != catalog.plan_digest
        or artifact.catalog_digest != catalog.catalog_digest
        or artifact.presentation != _legacy._image_identities(presentation)
        or artifact.artifact_digest != _digest(expected_artifact_digest, "expected artifact digest")
    ):
        raise PrototypeSceneObserverError("description differs from cold reconstruction")
    if artifact.receipt is not None:
        assert artifact.model_payload is not None
        _verify_receipt(
            artifact.receipt,
            presentation,
            prototype_rubric_description_prompt(),
            prototype_rubric_description_output_schema(),
            artifact.model_payload,
        )
    if artifact.status is PrototypeSceneObserverStatus.SUCCESS:
        parsed = _object_protocol().parse_prototype_object_description_payload(artifact.model_payload)
        if (
            tuple(parsed.feature_families) != artifact.feature_families
            or tuple(parsed.profiles) != artifact.profiles
            or _defined_rubrics(parsed.audit_rubrics) != artifact.rubrics
        ):
            raise PrototypeSceneObserverError("description payload replay differs")
    decoded = PrototypeRubricDescriptionArtifact.from_data(
        artifact.to_data(), expected_artifact_digest=expected_artifact_digest
    )
    if decoded != artifact:
        raise PrototypeSceneObserverError("description cold round trip differs")
    return artifact


def _score_from_disposition(
    tag_id: str, group_id: str, disposition: Disposition
) -> PrototypeSceneScore:
    if disposition is Disposition.PRESENT:
        return PrototypeSceneScore.scored(tag_id, group_id, PPM_SCALE, PPM_SCALE)
    if disposition is Disposition.CERTIFIED_ABSENT:
        return PrototypeSceneScore.scored(tag_id, group_id, 0, 0)
    if disposition is Disposition.INDETERMINATE:
        return PrototypeSceneScore.indeterminate(
            tag_id, group_id, "object_profile_indeterminate"
        )
    return PrototypeSceneScore.error(
        tag_id,
        group_id,
        "object_profile_error",
        "ObjectProfileEvaluationError",
    )


def _scores_from_evaluations(
    evaluations: Sequence[ObjectProfileEvaluation],
) -> tuple[PrototypeSceneScore, ...]:
    from bongard.prototype_pair_cohort import OPAQUE_TAG_IDS

    if len(evaluations) != 2:
        raise PrototypeSceneObserverError("profile evaluations must exhaust both groups")
    return tuple(
        _score_from_disposition(tag_id, group_id, evaluation.disposition)
        for tag_id, group_id, evaluation in zip(
            OPAQUE_TAG_IDS, PROTOTYPE_GROUP_IDS, evaluations, strict=True
        )
    )


def _audit_description(value: object) -> PrototypeSceneDescriptionObservation:
    state = getattr(getattr(value, "state", None), "value", getattr(value, "state", None))
    prose = getattr(value, "prose", None)
    if state == "defined" and isinstance(prose, str):
        try:
            return PrototypeSceneDescriptionObservation.defined(prose)
        except (TypeError, ValueError):
            return PrototypeSceneDescriptionObservation.rejected()
    if state == "rejected":
        return PrototypeSceneDescriptionObservation.rejected()
    return PrototypeSceneDescriptionObservation.unavailable(
        "audit_description_unavailable", "PrototypeObjectAuditTextUnavailable"
    )


def _feature_shard_preimage(
    artifact: "PrototypeObjectFeatureShardArtifact",
) -> dict[str, object]:
    return {
        "schema": PROTOTYPE_OBJECT_FEATURE_SHARD_ARTIFACT_SCHEMA,
        "spec": artifact.spec.to_data(),  # type: ignore[union-attr]
        "status": artifact.status.value,  # type: ignore[union-attr]
        "presentation": artifact.presentation.to_data(),
        "prompt_digest": artifact.prompt_digest,
        "output_schema_digest": artifact.output_schema_digest,
        "protocol_digest": artifact.protocol_digest,
        "model_payload": artifact.model_payload,
        "receipt": _receipt_to_data(artifact.receipt),
        "receipt_identity": artifact.receipt_identity,
        "payload_digest": artifact.payload_digest,
        "failure_code": artifact.failure_code,
        "failure_type": artifact.failure_type,
        "cells": [item.to_data() for item in artifact.cells],
        "profile_blind": True,
        "reference_blind": True,
    }


@dataclass(frozen=True, slots=True)
class PrototypeObjectFeatureShardArtifact:
    """One issued transport invocation and its exact bounded cover."""

    spec: object
    status: object
    presentation: PrototypeImageIdentity
    prompt_digest: str
    output_schema_digest: str
    protocol_digest: str
    model_payload: Mapping[str, Any] | None
    receipt: object | None
    receipt_identity: str | None
    payload_digest: str | None
    failure_code: str | None
    failure_type: str | None
    cells: tuple[ObjectFeatureCell, ...]
    artifact_digest: str
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        protocol = _object_protocol()
        if not isinstance(self.spec, protocol.ObjectFeatureShardSpec):
            raise TypeError("feature shard spec has the wrong type")
        if not isinstance(self.status, protocol.ObjectFeatureShardStatus):
            raise TypeError("feature shard status has the wrong type")
        if not isinstance(self.presentation, PrototypeImageIdentity):
            raise TypeError("feature shard presentation has the wrong type")
        if self.presentation.name != self.spec.sheet_name:
            raise PrototypeSceneObserverError("feature shard presents another sheet")
        for name in (
            "prompt_digest",
            "output_schema_digest",
            "protocol_digest",
            "artifact_digest",
        ):
            _digest(getattr(self, name), f"feature shard {name}")
        payload = self.model_payload
        if payload is not None:
            payload = _canonical_payload(payload)
            object.__setattr__(self, "model_payload", payload)
        if not isinstance(self.cells, tuple) or any(
            not isinstance(item, ObjectFeatureCell) for item in self.cells
        ):
            raise TypeError("feature shard cells must be a typed tuple")
        if self.status is protocol.ObjectFeatureShardStatus.SUCCESS:
            if (
                payload is None
                or self.receipt is None
                or self.receipt_identity is None
                or self.payload_digest is None
                or self.failure_code is not None
                or self.failure_type is not None
            ):
                raise PrototypeSceneObserverError("successful feature shard is incomplete")
        elif self.status is protocol.ObjectFeatureShardStatus.PARSER_ERROR:
            if (
                payload is None
                or self.receipt is None
                or self.receipt_identity is None
                or self.payload_digest is None
                or self.failure_code is None
                or self.failure_type is None
                or self.cells
            ):
                raise PrototypeSceneObserverError("parser-error feature shard differs")
        else:
            if (
                payload is not None
                or self.receipt is not None
                or self.receipt_identity is not None
                or self.payload_digest is not None
                or self.failure_code is None
                or self.failure_type is None
                or self.cells
            ):
                raise PrototypeSceneObserverError("transport-error feature shard differs")
        if self.receipt is not None:
            actual_receipt = getattr(self.receipt, "receipt_digest", None)
            if actual_receipt != self.receipt_identity:
                raise PrototypeSceneObserverError("feature shard receipt identity differs")
        if self.receipt_identity is not None:
            _digest(self.receipt_identity, "feature shard receipt identity")
        if self.payload_digest is not None:
            _digest(self.payload_digest, "feature shard payload digest")
            if payload is None or self.payload_digest != canonical_digest(dict(payload)):
                raise PrototypeSceneObserverError("feature shard payload digest differs")
        for name in ("failure_code", "failure_type"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value or value != value.strip()
            ):
                raise PrototypeSceneObserverError(f"feature shard {name} is invalid")
        computed = canonical_digest(_feature_shard_preimage(self))
        if self.artifact_digest != computed:
            raise PrototypeSceneObserverError("feature shard artifact digest differs")
        object.__setattr__(self, "_sealed_digest", computed)

    def to_data(self) -> dict[str, object]:
        return {**_feature_shard_preimage(self), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(cls, value: object) -> "PrototypeObjectFeatureShardArtifact":
        raw = _exact(
            value,
            {
                "schema", "spec", "status", "presentation", "prompt_digest",
                "output_schema_digest", "protocol_digest", "model_payload",
                "receipt", "receipt_identity", "payload_digest", "failure_code",
                "failure_type", "cells", "profile_blind", "reference_blind",
                "artifact_digest",
            },
            "feature shard artifact",
        )
        if (
            raw["schema"] != PROTOTYPE_OBJECT_FEATURE_SHARD_ARTIFACT_SCHEMA
            or raw["profile_blind"] is not True
            or raw["reference_blind"] is not True
            or not isinstance(raw["presentation"], Mapping)
            or not isinstance(raw["cells"], list)
        ):
            raise PrototypeSceneObserverError("unsupported feature shard artifact")
        protocol = _object_protocol()
        result = cls(
            spec=protocol.ObjectFeatureShardSpec.from_data(raw["spec"]),
            status=protocol.ObjectFeatureShardStatus(raw["status"]),
            presentation=PrototypeImageIdentity.from_data(raw["presentation"]),
            prompt_digest=raw["prompt_digest"],
            output_schema_digest=raw["output_schema_digest"],
            protocol_digest=raw["protocol_digest"],
            model_payload=(
                None
                if raw["model_payload"] is None
                else _canonical_payload(raw["model_payload"])
            ),
            receipt=_receipt_from_data(raw["receipt"]),
            receipt_identity=raw["receipt_identity"],
            payload_digest=raw["payload_digest"],
            failure_code=raw["failure_code"],
            failure_type=raw["failure_type"],
            cells=tuple(ObjectFeatureCell.from_data(item) for item in raw["cells"]),
            artifact_digest=raw["artifact_digest"],
        )
        if result.to_data() != dict(raw):
            raise PrototypeSceneObserverError("feature shard artifact is not canonical")
        return result

    def assert_untampered(self) -> None:
        computed = canonical_digest(_feature_shard_preimage(self))
        if computed != self.artifact_digest or computed != self._sealed_digest:
            raise PrototypeSceneObserverError("feature shard artifact changed after sealing")


def _build_feature_shard_artifact(
    *,
    packet: ObjectHypothesisPacket,
    spec: object,
    presentation: PrototypeImageIdentity,
    status: object,
    payload: Mapping[str, Any] | None,
    receipt: object | None,
    failure_code: str | None,
    failure_type: str | None,
    cells: tuple[ObjectFeatureCell, ...],
) -> PrototypeObjectFeatureShardArtifact:
    protocol = _object_protocol()
    prompt = protocol.prototype_object_feature_shard_prompt(packet, spec)
    schema = protocol.prototype_object_feature_output_schema()
    canonical_payload = None if payload is None else _canonical_payload(payload)
    values: dict[str, object] = {
        "spec": spec,
        "status": status,
        "presentation": presentation,
        "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "output_schema_digest": canonical_digest(schema),
        "protocol_digest": protocol.prototype_object_feature_shard_protocol_digest(
            packet, spec
        ),
        "model_payload": canonical_payload,
        "receipt": receipt,
        "receipt_identity": (
            None if receipt is None else getattr(receipt, "receipt_digest")
        ),
        "payload_digest": (
            None if canonical_payload is None else canonical_digest(dict(canonical_payload))
        ),
        "failure_code": failure_code,
        "failure_type": failure_type,
        "cells": cells,
    }
    provisional = object.__new__(PrototypeObjectFeatureShardArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return PrototypeObjectFeatureShardArtifact(
        **values,  # type: ignore[arg-type]
        artifact_digest=canonical_digest(_feature_shard_preimage(provisional)),
    )


def _feature_shard_outcomes(
    packet: ObjectHypothesisPacket,
    artifacts: Sequence[PrototypeObjectFeatureShardArtifact],
) -> tuple[object, ...]:
    protocol = _object_protocol()
    outcomes: list[object] = []
    for artifact in artifacts:
        artifact.assert_untampered()
        if artifact.status is protocol.ObjectFeatureShardStatus.SUCCESS:
            parsed = protocol.parse_prototype_object_feature_shard_payload(
                packet, artifact.spec, artifact.model_payload
            )
            if parsed.cells != artifact.cells or parsed.payload_digest != artifact.payload_digest:
                raise PrototypeSceneObserverError("feature shard payload replay differs")
            audit = parsed.audit_description
        elif artifact.status is protocol.ObjectFeatureShardStatus.PARSER_ERROR:
            try:
                protocol.parse_prototype_object_feature_shard_payload(
                    packet, artifact.spec, artifact.model_payload
                )
            except (TypeError, ValueError):
                pass
            else:
                raise PrototypeSceneObserverError("parser-error feature shard is admissible")
            audit = protocol.ObjectAuditText.rejected()
        else:
            audit = protocol.ObjectAuditText.rejected()
        outcomes.append(
            protocol.ObjectFeatureShardOutcome(
                artifact.spec.spec_digest,
                artifact.status,
                artifact.cells,
                artifact.receipt_identity,
                artifact.payload_digest,
                artifact.failure_code,
                artifact.failure_type,
                audit,
            )
        )
    return tuple(outcomes)


def _raw_feature_preimage(
    artifact: "PrototypeObjectFeatureObserverArtifact",
) -> dict[str, object]:
    return {
        "schema": PROTOTYPE_OBJECT_FEATURE_OBSERVER_ARTIFACT_SCHEMA,
        "observation_context_digest": artifact.observation_context_digest,
        "scene_id": artifact.scene_id,
        "scene_digest": artifact.scene_digest,
        "presentation": [item.to_data() for item in artifact.presentation],
        "hypothesis_packet": artifact.hypothesis_packet.to_data(),
        "feature_shard_plan": artifact.feature_shard_plan.to_data(),  # type: ignore[union-attr]
        "feature_shards": [item.to_data() for item in artifact.feature_shards],
        "physical_call_count": artifact.physical_call_count,
        "ordered_receipt_identities": list(artifact.ordered_receipt_identities),
        "shard_admission_scope": artifact.shard_admission_scope,
        "local_packets": [item.to_data() for item in artifact.local_packets],
        "description_observation": artifact.description_observation.to_data(),
        "protocol_digest": artifact.protocol_digest,
        "source_digest": artifact.source_digest,
        "transport_source_digest": artifact.transport_source_digest,
        "model": artifact.model,
        "reasoning_effort": artifact.reasoning_effort,
        "model_digest": artifact.model_digest,
        "expected_launcher_digest": artifact.expected_launcher_digest,
        "cloud_policy_cache_binding": artifact.cloud_policy_cache_binding,
        "model_catalog_digest": artifact.model_catalog_digest,
        "no_tools_attestation_digest": artifact.no_tools_attestation_digest,
        "environment_digest": artifact.environment_digest,
        "runtime_authority": _authority_data(),
        "description_or_profile_required": False,
        "profile_blind": True,
        "reference_blind": True,
    }


@dataclass(frozen=True, slots=True)
class PrototypeObjectFeatureObserverArtifact:
    """Raw full-catalog vision evidence, independent of any Bongard rule."""

    observation_context_digest: str
    scene_id: str
    scene_digest: str
    presentation: tuple[PrototypeImageIdentity, ...]
    hypothesis_packet: ObjectHypothesisPacket
    feature_shard_plan: object
    feature_shards: tuple[PrototypeObjectFeatureShardArtifact, ...]
    physical_call_count: int
    ordered_receipt_identities: tuple[str | None, ...]
    shard_admission_scope: str
    local_packets: tuple[ObjectLocalObservationPacket, ...]
    description_observation: PrototypeSceneDescriptionObservation
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
    artifact_digest: str
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _address(self.observation_context_digest, "raw observation context digest")
        if not isinstance(self.scene_id, str) or _SCENE_ID.fullmatch(self.scene_id) is None:
            raise PrototypeSceneObserverError("raw scene identity is invalid")
        _digest(self.scene_digest, "raw scene digest")
        if not isinstance(self.hypothesis_packet, ObjectHypothesisPacket):
            raise TypeError("raw hypothesis packet has the wrong type")
        if self.hypothesis_packet.panel_digest != self.scene_digest:
            raise PrototypeSceneObserverError("raw hypothesis packet binds another scene")
        expected_presentation = tuple(
            PrototypeImageIdentity(sheet.name, sheet.png_byte_count, sheet.png_digest)
            for sheet in self.hypothesis_packet.atlas_sheets
        )
        if self.presentation != expected_presentation:
            raise PrototypeSceneObserverError("raw atlas presentation differs")
        protocol = _object_protocol()
        if not isinstance(self.feature_shard_plan, protocol.ObjectFeatureShardPlan):
            raise TypeError("raw feature shard plan has the wrong type")
        protocol.verify_prototype_object_feature_shard_plan(
            self.feature_shard_plan, self.hypothesis_packet
        )
        if tuple(item.spec for item in self.feature_shards) != self.feature_shard_plan.shards:
            raise PrototypeSceneObserverError("raw feature shards differ from plan")
        if (
            isinstance(self.physical_call_count, bool)
            or not isinstance(self.physical_call_count, int)
            or self.physical_call_count != len(self.feature_shards)
            or self.physical_call_count != len(self.feature_shard_plan.shards)
        ):
            raise PrototypeSceneObserverError("raw physical call count differs")
        if self.ordered_receipt_identities != tuple(
            item.receipt_identity for item in self.feature_shards
        ):
            raise PrototypeSceneObserverError("raw ordered receipt identities differ")
        if self.shard_admission_scope != "internal_scene_bundle_only":
            raise PrototypeSceneObserverError("raw shard admission scope differs")
        identity_by_name = {item.name: item for item in self.presentation}
        schema_digest = canonical_digest(protocol.prototype_object_feature_output_schema())
        for shard in self.feature_shards:
            prompt = protocol.prototype_object_feature_shard_prompt(
                self.hypothesis_packet, shard.spec
            )
            if (
                shard.presentation != identity_by_name.get(shard.spec.sheet_name)
                or shard.prompt_digest
                != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                or shard.output_schema_digest != schema_digest
                or shard.protocol_digest
                != protocol.prototype_object_feature_shard_protocol_digest(
                    self.hypothesis_packet, shard.spec
                )
            ):
                raise PrototypeSceneObserverError("raw feature shard binding differs")
        parsed = protocol.assemble_prototype_object_feature_shards(
            self.hypothesis_packet,
            self.feature_shard_plan,
            _feature_shard_outcomes(self.hypothesis_packet, self.feature_shards),
            feature_model_id=self.model,
        )
        if tuple(parsed.packets) != self.local_packets:
            raise PrototypeSceneObserverError("raw local packets differ from shard assembly")
        if _audit_description(parsed.audit_description) != self.description_observation:
            raise PrototypeSceneObserverError("raw audit description differs")
        if self.protocol_digest != prototype_object_feature_observer_protocol_digest():
            raise PrototypeSceneObserverError("raw observer protocol digest differs")
        if self.source_digest != prototype_scene_observer_source_digest():
            raise PrototypeSceneObserverError("raw observer source digest differs")
        if self.transport_source_digest != prototype_scene_transport_source_digest():
            raise PrototypeSceneObserverError("raw transport source digest differs")
        if self.model_digest != prototype_scene_observer_model_digest(
            self.model, self.reasoning_effort
        ):
            raise PrototypeSceneObserverError("raw model digest differs")
        if self.environment_digest != prototype_scene_observer_environment_digest(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            expected_launcher_digest=self.expected_launcher_digest,
            cloud_policy_cache_binding=self.cloud_policy_cache_binding,
            model_catalog_digest=self.model_catalog_digest,
            no_tools_attestation_digest=self.no_tools_attestation_digest,
        ):
            raise PrototypeSceneObserverError("raw environment digest differs")
        for name in (
            "protocol_digest", "source_digest", "transport_source_digest",
            "model_digest", "model_catalog_digest", "no_tools_attestation_digest",
            "environment_digest", "artifact_digest",
        ):
            _digest(getattr(self, name), f"raw {name}")
        computed = canonical_digest(_raw_feature_preimage(self))
        if self.artifact_digest != computed:
            raise PrototypeSceneObserverError("raw feature artifact digest differs")
        object.__setattr__(self, "_sealed_digest", computed)

    def to_data(self) -> dict[str, object]:
        return {**_raw_feature_preimage(self), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(
        cls,
        value: object,
        *,
        expected_artifact_digest: str | None = None,
    ) -> "PrototypeObjectFeatureObserverArtifact":
        raw = _exact(
            value,
            {
                "schema", "observation_context_digest", "scene_id", "scene_digest",
                "presentation", "hypothesis_packet", "feature_shard_plan",
                "feature_shards", "physical_call_count", "ordered_receipt_identities",
                "shard_admission_scope", "local_packets", "description_observation",
                "protocol_digest", "source_digest", "transport_source_digest",
                "model", "reasoning_effort", "model_digest",
                "expected_launcher_digest", "cloud_policy_cache_binding",
                "model_catalog_digest", "no_tools_attestation_digest",
                "environment_digest", "runtime_authority",
                "description_or_profile_required", "profile_blind", "reference_blind",
                "artifact_digest",
            },
            "raw feature observer artifact",
        )
        for name in (
            "presentation", "feature_shards", "ordered_receipt_identities", "local_packets"
        ):
            if not isinstance(raw[name], list):
                raise PrototypeSceneObserverError(f"raw {name} must be a JSON list")
        if (
            raw["schema"] != PROTOTYPE_OBJECT_FEATURE_OBSERVER_ARTIFACT_SCHEMA
            or raw["runtime_authority"] != _authority_data()
            or raw["description_or_profile_required"] is not False
            or raw["profile_blind"] is not True
            or raw["reference_blind"] is not True
            or not isinstance(raw["hypothesis_packet"], Mapping)
            or not isinstance(raw["feature_shard_plan"], Mapping)
            or not isinstance(raw["description_observation"], Mapping)
        ):
            raise PrototypeSceneObserverError("unsupported raw feature artifact")
        protocol = _object_protocol()
        result = cls(
            observation_context_digest=raw["observation_context_digest"],
            scene_id=raw["scene_id"],
            scene_digest=raw["scene_digest"],
            presentation=tuple(
                PrototypeImageIdentity.from_data(item) for item in raw["presentation"]
            ),
            hypothesis_packet=ObjectHypothesisPacket.from_data(raw["hypothesis_packet"]),
            feature_shard_plan=protocol.ObjectFeatureShardPlan.from_data(
                raw["feature_shard_plan"]
            ),
            feature_shards=tuple(
                PrototypeObjectFeatureShardArtifact.from_data(item)
                for item in raw["feature_shards"]
            ),
            physical_call_count=raw["physical_call_count"],
            ordered_receipt_identities=tuple(raw["ordered_receipt_identities"]),
            shard_admission_scope=raw["shard_admission_scope"],
            local_packets=tuple(
                ObjectLocalObservationPacket.from_data(item)
                for item in raw["local_packets"]
            ),
            description_observation=PrototypeSceneDescriptionObservation.from_data(
                raw["description_observation"]
            ),
            protocol_digest=raw["protocol_digest"],
            source_digest=raw["source_digest"],
            transport_source_digest=raw["transport_source_digest"],
            model=raw["model"],
            reasoning_effort=raw["reasoning_effort"],
            model_digest=raw["model_digest"],
            expected_launcher_digest=raw["expected_launcher_digest"],
            cloud_policy_cache_binding=raw["cloud_policy_cache_binding"],
            model_catalog_digest=raw["model_catalog_digest"],
            no_tools_attestation_digest=raw["no_tools_attestation_digest"],
            environment_digest=raw["environment_digest"],
            artifact_digest=raw["artifact_digest"],
        )
        if expected_artifact_digest is not None and result.artifact_digest != _digest(
            expected_artifact_digest, "expected raw feature artifact digest"
        ):
            raise PrototypeSceneObserverError("raw feature artifact commitment differs")
        if result.to_data() != dict(raw):
            raise PrototypeSceneObserverError("raw feature artifact is not canonical")
        return result

    def assert_untampered(self) -> None:
        computed = canonical_digest(_raw_feature_preimage(self))
        if computed != self.artifact_digest or computed != self._sealed_digest:
            raise PrototypeSceneObserverError("raw feature artifact changed after sealing")


def _build_raw_feature_artifact(
    *,
    observation_context_digest: str,
    scene_id: str,
    scene: bytes,
    packet: ObjectHypothesisPacket,
    identities: tuple[PrototypeImageIdentity, ...],
    plan: object,
    shards: tuple[PrototypeObjectFeatureShardArtifact, ...],
    local_packets: tuple[ObjectLocalObservationPacket, ...],
    description_observation: PrototypeSceneDescriptionObservation,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str | None,
    cloud_policy_cache_binding: str,
    model_catalog_digest: str,
    no_tools_attestation_digest: str,
) -> PrototypeObjectFeatureObserverArtifact:
    values: dict[str, object] = {
        "observation_context_digest": observation_context_digest,
        "scene_id": scene_id,
        "scene_digest": hashlib.sha256(scene).hexdigest(),
        "presentation": identities,
        "hypothesis_packet": packet,
        "feature_shard_plan": plan,
        "feature_shards": shards,
        "physical_call_count": len(shards),
        "ordered_receipt_identities": tuple(item.receipt_identity for item in shards),
        "shard_admission_scope": "internal_scene_bundle_only",
        "local_packets": local_packets,
        "description_observation": description_observation,
        "protocol_digest": prototype_object_feature_observer_protocol_digest(),
        "source_digest": prototype_scene_observer_source_digest(),
        "transport_source_digest": prototype_scene_transport_source_digest(),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "model_digest": prototype_scene_observer_model_digest(model, reasoning_effort),
        "expected_launcher_digest": expected_launcher_digest,
        "cloud_policy_cache_binding": cloud_policy_cache_binding,
        "model_catalog_digest": model_catalog_digest,
        "no_tools_attestation_digest": no_tools_attestation_digest,
        "environment_digest": prototype_scene_observer_environment_digest(
            model=model,
            reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=cloud_policy_cache_binding,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_attestation_digest,
        ),
    }
    provisional = object.__new__(PrototypeObjectFeatureObserverArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return PrototypeObjectFeatureObserverArtifact(
        **values,  # type: ignore[arg-type]
        artifact_digest=canonical_digest(_raw_feature_preimage(provisional)),
    )


def _run_feature_shards(
    packet: ObjectHypothesisPacket,
    atlas: Sequence[tuple[str, bytes]],
    *,
    model: str,
    reasoning_effort: str,
    minutes: int,
    verbose: bool,
    executable: str,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None,
    expected_launcher_digest: str | None,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    transport: NamedImageTransport,
    hidden_values: Sequence[str],
) -> tuple[object, tuple[PrototypeObjectFeatureShardArtifact, ...]]:
    """Issue the deterministic plan in order, continuing after typed failures."""

    protocol = _object_protocol()
    plan = protocol.plan_prototype_object_feature_shards(packet)
    schema = protocol.prototype_object_feature_output_schema()
    _legacy.validate_codex_strict_output_schema(schema)
    atlas_by_name = dict(atlas)
    identities = _legacy._image_identities(atlas)
    identity_by_name = {item.name: item for item in identities}
    artifacts: list[PrototypeObjectFeatureShardArtifact] = []
    for spec in plan.shards:
        prompt = protocol.prototype_object_feature_shard_prompt(packet, spec)
        presentation = ((spec.sheet_name, atlas_by_name[spec.sheet_name]),)
        _legacy._assert_model_visible_boundary(
            prompt,
            schema,
            (spec.sheet_name,),
            hidden_values=hidden_values,
        )
        try:
            payload, receipt = _legacy._stage_and_call(
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
                transport=transport,
            )
        except Exception as exc:
            artifacts.append(
                _build_feature_shard_artifact(
                    packet=packet,
                    spec=spec,
                    presentation=identity_by_name[spec.sheet_name],
                    status=protocol.ObjectFeatureShardStatus.TRANSPORT_ERROR,
                    payload=None,
                    receipt=None,
                    failure_code="shard_transport_failed",
                    failure_type=_legacy._exception_type(exc),
                    cells=(),
                )
            )
            continue
        try:
            parsed = protocol.parse_prototype_object_feature_shard_payload(
                packet, spec, payload
            )
        except (TypeError, ValueError):
            artifacts.append(
                _build_feature_shard_artifact(
                    packet=packet,
                    spec=spec,
                    presentation=identity_by_name[spec.sheet_name],
                    status=protocol.ObjectFeatureShardStatus.PARSER_ERROR,
                    payload=payload,
                    receipt=receipt,
                    failure_code="shard_payload_rejected",
                    failure_type="PrototypeScenePayloadError",
                    cells=(),
                )
            )
        else:
            artifacts.append(
                _build_feature_shard_artifact(
                    packet=packet,
                    spec=spec,
                    presentation=identity_by_name[spec.sheet_name],
                    status=protocol.ObjectFeatureShardStatus.SUCCESS,
                    payload=payload,
                    receipt=receipt,
                    failure_code=None,
                    failure_type=None,
                    cells=tuple(parsed.cells),
                )
            )
    return plan, tuple(artifacts)


def _scene_preimage(artifact: "PrototypeSceneObserverArtifact") -> dict[str, object]:
    return {
        "schema": PROTOTYPE_SCENE_OBSERVER_ARTIFACT_SCHEMA,
        **_common_data(artifact),
        "observation_context_digest": artifact.observation_context_digest,
        "rubric_description_digest": artifact.rubric_description_digest,
        "scene_task_id": artifact.scene_task_id,
        "scene_panel_id": artifact.scene_panel_id,
        "scene_digest": artifact.scene_digest,
        "raw_feature_artifact_digest": artifact.raw_feature_artifact_digest,
        "hypothesis_packet": (
            None
            if artifact.hypothesis_packet is None
            else artifact.hypothesis_packet.to_data()
        ),
        "feature_shard_plan": (
            None
            if artifact.feature_shard_plan is None
            else artifact.feature_shard_plan.to_data()
        ),
        "feature_shards": [item.to_data() for item in artifact.feature_shards],
        "physical_call_count": artifact.physical_call_count,
        "ordered_receipt_identities": list(artifact.ordered_receipt_identities),
        "shard_admission_scope": artifact.shard_admission_scope,
        "local_packets": [item.to_data() for item in artifact.local_packets],
        "evaluations": [item.to_data() for item in artifact.evaluations],
        "description_observation": artifact.description_observation.to_data(),
        "scores": [item.to_data() for item in artifact.scores],
        "runtime_authority": _authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PrototypeSceneObserverArtifact:
    status: PrototypeSceneObserverStatus
    plan_digest: str
    catalog_digest: str
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
    failure_code: str | None
    failure_type: str | None
    failure_digest: str | None
    observation_context_digest: str
    rubric_description_digest: str
    scene_task_id: str
    scene_panel_id: str
    scene_digest: str
    raw_feature_artifact_digest: str | None
    hypothesis_packet: ObjectHypothesisPacket | None
    feature_shard_plan: object | None
    feature_shards: tuple[PrototypeObjectFeatureShardArtifact, ...]
    physical_call_count: int
    ordered_receipt_identities: tuple[str | None, ...]
    shard_admission_scope: str
    local_packets: tuple[ObjectLocalObservationPacket, ...]
    evaluations: tuple[ObjectProfileEvaluation, ...]
    description_observation: PrototypeSceneDescriptionObservation
    scores: tuple[PrototypeSceneScore, ...]
    artifact_digest: str
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_common(
            self,
            expected_protocol_digest=prototype_scene_scoring_protocol_digest(),
            phase="object-scene-features",
            aggregate_model_output=True,
        )
        _address(self.observation_context_digest, "observation context digest")
        _digest(self.rubric_description_digest, "rubric description digest")
        _digest(self.scene_digest, "scene digest")
        if self.raw_feature_artifact_digest is not None:
            _digest(self.raw_feature_artifact_digest, "raw feature artifact digest")
        if (
            not isinstance(self.scene_task_id, str)
            or not self.scene_task_id
            or not isinstance(self.scene_panel_id, str)
            or not self.scene_panel_id.startswith(f"bd/{self.scene_task_id}/")
            or not self.scene_panel_id.endswith(".png")
        ):
            raise PrototypeSceneObserverError("scheduled scene identity is invalid")
        if not isinstance(self.local_packets, tuple) or any(
            not isinstance(item, ObjectLocalObservationPacket)
            for item in self.local_packets
        ):
            raise TypeError("local packets must be a typed tuple")
        if not isinstance(self.evaluations, tuple) or any(
            not isinstance(item, ObjectProfileEvaluation)
            for item in self.evaluations
        ):
            raise TypeError("evaluations must be a typed tuple")
        if not isinstance(self.description_observation, PrototypeSceneDescriptionObservation):
            raise TypeError("description observation has wrong type")
        if not isinstance(self.scores, tuple) or len(self.scores) != 2:
            raise PrototypeSceneObserverError("scores must exhaust both groups")
        if (
            isinstance(self.physical_call_count, bool)
            or not isinstance(self.physical_call_count, int)
            or self.physical_call_count < 0
        ):
            raise PrototypeSceneObserverError("physical call count is invalid")
        if not isinstance(self.feature_shards, tuple) or any(
            not isinstance(item, PrototypeObjectFeatureShardArtifact)
            for item in self.feature_shards
        ):
            raise TypeError("feature shards must be a typed tuple")
        if not isinstance(self.ordered_receipt_identities, tuple) or any(
            item is not None and (not isinstance(item, str) or _SHA256.fullmatch(item) is None)
            for item in self.ordered_receipt_identities
        ):
            raise PrototypeSceneObserverError("ordered receipt identities are invalid")
        if self.shard_admission_scope != "internal_scene_bundle_only":
            raise PrototypeSceneObserverError("feature shard admission scope differs")
        status = self.status
        if status in {
            PrototypeSceneObserverStatus.PARSER_ERROR,
            PrototypeSceneObserverStatus.TRANSPORT_ERROR,
        }:
            raise PrototypeSceneObserverError(
                "v2 scene-level parser/transport failures must be typed per shard"
            )
        if status is PrototypeSceneObserverStatus.SUCCESS:
            if not isinstance(self.hypothesis_packet, ObjectHypothesisPacket):
                raise PrototypeSceneObserverError("attempted scene lacks hypothesis packet")
            if self.raw_feature_artifact_digest is None:
                raise PrototypeSceneObserverError("attempted scene lacks raw feature commitment")
            if self.hypothesis_packet.panel_digest != self.scene_digest:
                raise PrototypeSceneObserverError("hypothesis packet binds another scene")
            expected_presentation = tuple(
                PrototypeImageIdentity(sheet.name, sheet.png_byte_count, sheet.png_digest)
                for sheet in self.hypothesis_packet.atlas_sheets
            )
            if self.presentation != expected_presentation:
                raise PrototypeSceneObserverError("atlas presentation differs from packet")
            object_protocol = _object_protocol()
            if not isinstance(
                self.feature_shard_plan, object_protocol.ObjectFeatureShardPlan
            ):
                raise PrototypeSceneObserverError("attempted scene lacks feature shard plan")
            object_protocol.verify_prototype_object_feature_shard_plan(
                self.feature_shard_plan, self.hypothesis_packet
            )
            if tuple(item.spec for item in self.feature_shards) != self.feature_shard_plan.shards:
                raise PrototypeSceneObserverError("feature shard artifacts differ from plan")
            if self.physical_call_count != len(self.feature_shards) or self.physical_call_count != len(
                self.feature_shard_plan.shards
            ):
                raise PrototypeSceneObserverError("physical call count differs from issued shards")
            if self.ordered_receipt_identities != tuple(
                item.receipt_identity for item in self.feature_shards
            ):
                raise PrototypeSceneObserverError("ordered receipt manifest differs")
            identity_by_name = {item.name: item for item in self.presentation}
            schema_digest = canonical_digest(
                object_protocol.prototype_object_feature_output_schema()
            )
            for shard in self.feature_shards:
                expected_prompt = object_protocol.prototype_object_feature_shard_prompt(
                    self.hypothesis_packet, shard.spec
                )
                if (
                    shard.presentation != identity_by_name.get(shard.spec.sheet_name)
                    or shard.prompt_digest
                    != hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest()
                    or shard.output_schema_digest != schema_digest
                    or shard.protocol_digest
                    != object_protocol.prototype_object_feature_shard_protocol_digest(
                        self.hypothesis_packet, shard.spec
                    )
                ):
                    raise PrototypeSceneObserverError("feature shard protocol binding differs")
            outcomes = _feature_shard_outcomes(
                self.hypothesis_packet, self.feature_shards
            )
            parsed = object_protocol.assemble_prototype_object_feature_shards(
                self.hypothesis_packet,
                self.feature_shard_plan,
                outcomes,
                feature_model_id=self.model,
            )
            if tuple(parsed.packets) != self.local_packets:
                raise PrototypeSceneObserverError("local packets differ from shard assembly")
            if _audit_description(parsed.audit_description) != self.description_observation:
                raise PrototypeSceneObserverError("scene audit differs from shard assembly")
            if len(self.local_packets) != 3 or len(self.evaluations) != 2:
                raise PrototypeSceneObserverError("successful scene lacks exhaustive decisions")
            expected_scenarios = tuple(
                item.scenario_id for item in self.hypothesis_packet.scenarios
            )
            if tuple(item.scenario_id for item in self.local_packets) != expected_scenarios:
                raise PrototypeSceneObserverError("local packet scenarios differ")
            packet_digest = self.hypothesis_packet.digest()
            expected_feature_protocol = (
                object_protocol.prototype_object_feature_protocol_digest(
                    self.hypothesis_packet
                )
            )
            for packet in self.local_packets:
                if (
                    packet.panel_digest != self.scene_digest
                    or packet.visual_witness_packet_digest
                    != self.hypothesis_packet.visual_witness_packet_digest
                    or packet.hypothesis_catalog_digest != packet_digest
                    or packet.feature_protocol_digest != expected_feature_protocol
                    or packet.feature_model_id != self.model
                ):
                    raise PrototypeSceneObserverError("local packet provenance differs")
            local_digests = tuple(item.packet_digest for item in self.local_packets)
            if any(
                item.scenario_packet_digests != local_digests
                for item in self.evaluations
            ):
                raise PrototypeSceneObserverError("evaluation packet binding differs")
            if self.scores != _scores_from_evaluations(self.evaluations):
                raise PrototypeSceneObserverError("scores differ from Python evaluations")
        else:
            if (
                self.hypothesis_packet is not None
                or self.raw_feature_artifact_digest is not None
                or self.presentation
                or self.feature_shard_plan is not None
                or self.feature_shards
                or self.physical_call_count != 0
                or self.ordered_receipt_identities
            ):
                raise PrototypeSceneObserverError("unattempted scene carries shard evidence")
            if self.local_packets or self.evaluations:
                raise PrototypeSceneObserverError("failed scene carries decisional evaluation")
            expected_failure_rows = {
                PrototypeSceneObserverStatus.INTERNAL_ERROR: (
                    PrototypeSceneDescriptionObservation.unavailable(
                        "observer_internal_error", "PrototypeSceneInternalError"
                    ),
                    _error_scores("observer_internal_error", "PrototypeSceneInternalError"),
                ),
                PrototypeSceneObserverStatus.PREREQUISITE_ERROR: (
                    PrototypeSceneDescriptionObservation.unavailable(
                        "profile_prerequisite_failed", "PrototypeObjectProfilePrerequisiteFailure"
                    ),
                    _error_scores("profile_prerequisite_failed", "PrototypeObjectProfilePrerequisiteFailure"),
                ),
            }
            if (
                self.status not in expected_failure_rows
                or (self.description_observation, self.scores)
                != expected_failure_rows[self.status]
            ):
                raise PrototypeSceneObserverError("failed scene typed rows differ")
        expected_prompt, expected_schema = _scene_prompt_schema(self.hypothesis_packet)
        if (
            self.prompt_digest != hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest()
            or self.output_schema_digest != canonical_digest(expected_schema)
        ):
            raise PrototypeSceneObserverError("scene prompt/schema digest differs")
        computed = canonical_digest(_scene_preimage(self))
        if self.artifact_digest != computed:
            raise PrototypeSceneObserverError("scene artifact digest differs")
        object.__setattr__(self, "_sealed_digest", computed)

    def to_data(self) -> dict[str, object]:
        return {**_scene_preimage(self), "artifact_digest": self.artifact_digest}

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        expected_artifact_digest: str | None = None,
    ) -> "PrototypeSceneObserverArtifact":
        raw = _exact(
            value,
            {
                "schema", *_COMMON_FIELDS, "observation_context_digest",
                "rubric_description_digest", "scene_task_id", "scene_panel_id",
                "scene_digest", "raw_feature_artifact_digest", "hypothesis_packet",
                "feature_shard_plan",
                "feature_shards", "physical_call_count", "ordered_receipt_identities",
                "shard_admission_scope", "local_packets", "evaluations",
                "description_observation", "scores", "runtime_authority",
                "artifact_digest",
            },
            "object scene artifact",
        )
        if raw["schema"] != PROTOTYPE_SCENE_OBSERVER_ARTIFACT_SCHEMA or raw["runtime_authority"] != _authority_data():
            raise PrototypeSceneObserverError("unsupported scene artifact")
        for name in (
            "presentation", "feature_shards", "ordered_receipt_identities",
            "local_packets", "evaluations", "scores",
        ):
            if not isinstance(raw[name], list):
                raise PrototypeSceneObserverError(f"scene {name} must be a JSON list")
        packet = raw["hypothesis_packet"]
        if packet is not None and not isinstance(packet, Mapping):
            raise PrototypeSceneObserverError("hypothesis packet must be object or null")
        if raw["feature_shard_plan"] is not None and not isinstance(
            raw["feature_shard_plan"], Mapping
        ):
            raise PrototypeSceneObserverError("feature shard plan must be object or null")
        if not isinstance(raw["description_observation"], Mapping):
            raise PrototypeSceneObserverError("description observation must be object")
        result = cls(
            status=PrototypeSceneObserverStatus(raw["status"]),
            plan_digest=raw["plan_digest"],
            catalog_digest=raw["catalog_digest"],
            presentation=tuple(PrototypeImageIdentity.from_data(x) for x in raw["presentation"]),
            prompt_digest=raw["prompt_digest"],
            output_schema_digest=raw["output_schema_digest"],
            protocol_digest=raw["protocol_digest"],
            source_digest=raw["source_digest"],
            transport_source_digest=raw["transport_source_digest"],
            model=raw["model"],
            reasoning_effort=raw["reasoning_effort"],
            model_digest=raw["model_digest"],
            expected_launcher_digest=raw["expected_launcher_digest"],
            cloud_policy_cache_binding=raw["cloud_policy_cache_binding"],
            model_catalog_digest=raw["model_catalog_digest"],
            no_tools_attestation_digest=raw["no_tools_attestation_digest"],
            environment_digest=raw["environment_digest"],
            model_payload=None if raw["model_payload"] is None else _canonical_payload(raw["model_payload"]),
            receipt=_receipt_from_data(raw["receipt"]),
            failure_code=raw["failure_code"],
            failure_type=raw["failure_type"],
            failure_digest=raw["failure_digest"],
            observation_context_digest=raw["observation_context_digest"],
            rubric_description_digest=raw["rubric_description_digest"],
            scene_task_id=raw["scene_task_id"],
            scene_panel_id=raw["scene_panel_id"],
            scene_digest=raw["scene_digest"],
            raw_feature_artifact_digest=raw["raw_feature_artifact_digest"],
            hypothesis_packet=None if packet is None else ObjectHypothesisPacket.from_data(packet),
            feature_shard_plan=(
                None
                if raw["feature_shard_plan"] is None
                else _object_protocol().ObjectFeatureShardPlan.from_data(
                    raw["feature_shard_plan"]
                )
            ),
            feature_shards=tuple(
                PrototypeObjectFeatureShardArtifact.from_data(item)
                for item in raw["feature_shards"]
            ),
            physical_call_count=raw["physical_call_count"],
            ordered_receipt_identities=tuple(raw["ordered_receipt_identities"]),
            shard_admission_scope=raw["shard_admission_scope"],
            local_packets=tuple(ObjectLocalObservationPacket.from_data(x) for x in raw["local_packets"]),
            evaluations=tuple(ObjectProfileEvaluation.from_data(x) for x in raw["evaluations"]),
            description_observation=PrototypeSceneDescriptionObservation.from_data(raw["description_observation"]),
            scores=tuple(PrototypeSceneScore.from_data(x) for x in raw["scores"]),
            artifact_digest=raw["artifact_digest"],
        )
        if expected_artifact_digest is not None and result.artifact_digest != _digest(expected_artifact_digest, "expected scene artifact digest"):
            raise PrototypeSceneObserverError("scene artifact commitment differs")
        if result.to_data() != dict(raw):
            raise PrototypeSceneObserverError("scene artifact is not canonical")
        return result

    def assert_untampered(self) -> None:
        computed = canonical_digest(_scene_preimage(self))
        if computed != self.artifact_digest or computed != self._sealed_digest:
            raise PrototypeSceneObserverError("scene artifact changed after sealing")

    def to_calibration_observation_data(
        self, *, calibration_plan_digest: str
    ) -> Mapping[str, Any]:
        """Adapt frozen Python decisions to the existing calibration DTO."""

        self.assert_untampered()
        expected = _address(calibration_plan_digest, "calibration plan digest")
        if self.observation_context_digest != expected:
            raise PrototypeSceneObserverError(
                "scene did not precommit the requested calibration plan"
            )
        if self.status is PrototypeSceneObserverStatus.PREREQUISITE_ERROR:
            raise PrototypeSceneObserverError(
                "prerequisite failure made zero observer calls"
            )
        if self.expected_launcher_digest is None:
            raise PrototypeSceneObserverError(
                "calibration requires a launcher commitment"
            )
        from bongard.prototype_scene_calibration import (
            OBSERVER_ADAPTER_PROTOCOL_ID,
            PrototypeSceneCalibrationObservation,
            PrototypeSceneScoreStatus,
            PrototypeSceneTagScore,
        )

        adapted: list[PrototypeSceneTagScore] = []
        for score in self.scores:
            if self.status is PrototypeSceneObserverStatus.PARSER_ERROR:
                status = PrototypeSceneScoreStatus.PARSER_ERROR
            elif self.status is PrototypeSceneObserverStatus.TRANSPORT_ERROR:
                status = PrototypeSceneScoreStatus.TRANSPORT_ERROR
            elif score.state is PrototypeSceneScoreState.SCORED:
                status = PrototypeSceneScoreStatus.SCORE
            elif score.state is PrototypeSceneScoreState.INDETERMINATE:
                status = PrototypeSceneScoreStatus.INDETERMINATE
            else:
                status = PrototypeSceneScoreStatus.ERROR
            adapted.append(
                PrototypeSceneTagScore(
                    tag_id=score.tag_id,
                    status=status,
                    lower_ppm=(
                        score.lower_ppm
                        if status is PrototypeSceneScoreStatus.SCORE
                        else None
                    ),
                    upper_ppm=(
                        score.upper_ppm
                        if status is PrototypeSceneScoreStatus.SCORE
                        else None
                    ),
                    reason_code=(
                        "scored"
                        if status is PrototypeSceneScoreStatus.SCORE
                        else score.reason_code or "observer_indeterminate"
                    ),
                    error_type=(
                        None
                        if status is PrototypeSceneScoreStatus.SCORE
                        else score.error_type or "PrototypeSceneIndeterminate"
                    ),
                )
            )
        return PrototypeSceneCalibrationObservation(
            calibration_plan_digest=expected,
            cohort_plan_digest=self.plan_digest,
            task_id=self.scene_task_id,
            panel_id=self.scene_panel_id,
            observer_artifact_digest="sha256:" + self.artifact_digest,
            observer_artifact_schema=PROTOTYPE_SCENE_OBSERVER_ARTIFACT_SCHEMA,
            description_catalog_digest="sha256:" + self.rubric_description_digest,
            prototype_reference_digest="sha256:" + self.catalog_digest,
            observer_protocol_id=PROTOTYPE_SCENE_OBSERVER_PROTOCOL_ID,
            observer_protocol_digest="sha256:" + self.protocol_digest,
            model_id=self.model,
            model_identity_digest="sha256:" + self.model_digest,
            environment_digest="sha256:" + self.environment_digest,
            observer_call_count=self.physical_call_count,
            scores=tuple(adapted),  # type: ignore[arg-type]
            adapter_protocol_id=OBSERVER_ADAPTER_PROTOCOL_ID,
        ).to_data()


def _scene_prompt_schema(
    packet: ObjectHypothesisPacket | None,
) -> tuple[str, Mapping[str, Any]]:
    if packet is None:
        return (
            "Object feature observation was not authorized.",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
                "required": [],
            },
        )
    protocol = _object_protocol()
    plan = protocol.plan_prototype_object_feature_shards(packet)
    return (
        "Internal profile-blind feature-shard bundle; no aggregate model call. "
        f"plan_digest={plan.plan_digest}; physical_calls={len(plan.shards)}.",
        protocol.prototype_object_feature_output_schema(),
    )


def _build_scene_artifact(
    *,
    status: PrototypeSceneObserverStatus,
    catalog: PrototypeReferenceCatalog,
    rubric_artifact: PrototypeRubricDescriptionArtifact,
    observation_context_digest: str,
    scene_task_id: str,
    scene_panel_id: str,
    exact_scene_png_bytes: bytes,
    raw_feature_artifact_digest: str | None,
    identities: tuple[PrototypeImageIdentity, ...],
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str | None,
    cloud_policy_cache_binding: str,
    model_catalog_digest: str,
    no_tools_attestation_digest: str,
    payload: Mapping[str, Any] | None,
    receipt: object | None,
    failure_code: str | None,
    failure_type: str | None,
    hypothesis_packet: ObjectHypothesisPacket | None,
    feature_shard_plan: object | None,
    feature_shards: tuple[PrototypeObjectFeatureShardArtifact, ...],
    local_packets: tuple[ObjectLocalObservationPacket, ...],
    evaluations: tuple[ObjectProfileEvaluation, ...],
    description_observation: PrototypeSceneDescriptionObservation,
    scores: tuple[PrototypeSceneScore, ...],
) -> PrototypeSceneObserverArtifact:
    prompt, schema = _scene_prompt_schema(hypothesis_packet)
    canonical_payload = None if payload is None else _canonical_payload(payload)
    values: dict[str, object] = {
        "status": status,
        "plan_digest": catalog.plan_digest,
        "catalog_digest": catalog.catalog_digest,
        "presentation": identities,
        "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "output_schema_digest": canonical_digest(schema),
        "protocol_digest": prototype_scene_scoring_protocol_digest(),
        "source_digest": prototype_scene_observer_source_digest(),
        "transport_source_digest": prototype_scene_transport_source_digest(),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "model_digest": prototype_scene_observer_model_digest(model, reasoning_effort),
        "expected_launcher_digest": expected_launcher_digest,
        "cloud_policy_cache_binding": cloud_policy_cache_binding,
        "model_catalog_digest": model_catalog_digest,
        "no_tools_attestation_digest": no_tools_attestation_digest,
        "environment_digest": prototype_scene_observer_environment_digest(
            model=model,
            reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=cloud_policy_cache_binding,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_attestation_digest,
        ),
        "model_payload": canonical_payload,
        "receipt": receipt,
        "failure_code": failure_code,
        "failure_type": failure_type,
        "failure_digest": _failure_digest(
            "object-scene-features", status, failure_code, failure_type, canonical_payload
        ),
        "observation_context_digest": observation_context_digest,
        "rubric_description_digest": rubric_artifact.artifact_digest,
        "scene_task_id": scene_task_id,
        "scene_panel_id": scene_panel_id,
        "scene_digest": hashlib.sha256(exact_scene_png_bytes).hexdigest(),
        "raw_feature_artifact_digest": raw_feature_artifact_digest,
        "hypothesis_packet": hypothesis_packet,
        "feature_shard_plan": feature_shard_plan,
        "feature_shards": feature_shards,
        "physical_call_count": len(feature_shards),
        "ordered_receipt_identities": tuple(
            item.receipt_identity for item in feature_shards
        ),
        "shard_admission_scope": "internal_scene_bundle_only",
        "local_packets": local_packets,
        "evaluations": evaluations,
        "description_observation": description_observation,
        "scores": scores,
    }
    provisional = object.__new__(PrototypeSceneObserverArtifact)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return PrototypeSceneObserverArtifact(
        **values,  # type: ignore[arg-type]
        artifact_digest=canonical_digest(_scene_preimage(provisional)),
    )


def observe_prototype_object_features(
    exact_scene_png_bytes: bytes,
    *,
    scene_id: str,
    observation_context_digest: str,
    expected_scene_sha256: str,
    model: str,
    reasoning_effort: str = "medium",
    minutes: int = 15,
    verbose: bool = False,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    expected_launcher_digest: str | None = None,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    transport: NamedImageTransport = run_codex_named_images_structured,
) -> PrototypeObjectFeatureObserverArtifact:
    """Observe one committed scene without references, descriptions, or profiles."""

    if not callable(transport):
        raise TypeError("transport must be callable")
    scene = _legacy._validate_exact_png(exact_scene_png_bytes, "scene")
    if hashlib.sha256(scene).hexdigest() != _digest(
        expected_scene_sha256, "expected raw scene sha256"
    ):
        raise PrototypeSceneObserverError("raw scene differs from byte commitment")
    _address(observation_context_digest, "raw observation context digest")
    if not isinstance(scene_id, str) or _SCENE_ID.fullmatch(scene_id) is None:
        raise PrototypeSceneObserverError("raw scene identity is invalid")
    policy, model_catalog_digest, no_tools_digest = _runtime_identities(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    packet = extract_object_hypotheses(scene)
    atlas = render_object_hypothesis_atlas(packet, scene)
    identities = _legacy._image_identities(atlas)
    plan, shards = _run_feature_shards(
        packet,
        atlas,
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
        hidden_values=(observation_context_digest, scene_id),
    )
    protocol = _object_protocol()
    parsed = protocol.assemble_prototype_object_feature_shards(
        packet,
        plan,
        _feature_shard_outcomes(packet, shards),
        feature_model_id=model,
    )
    return _build_raw_feature_artifact(
        observation_context_digest=observation_context_digest,
        scene_id=scene_id,
        scene=scene,
        packet=packet,
        identities=identities,
        plan=plan,
        shards=shards,
        local_packets=tuple(parsed.packets),
        description_observation=_audit_description(parsed.audit_description),
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=policy,
        model_catalog_digest=model_catalog_digest,
        no_tools_attestation_digest=no_tools_digest,
    )


def verify_prototype_object_feature_observer_artifact(
    artifact: PrototypeObjectFeatureObserverArtifact,
    exact_scene_png_bytes: bytes,
    *,
    expected_scene_id: str,
    expected_observation_context_digest: str,
    expected_scene_sha256: str,
    expected_artifact_digest: str,
) -> PrototypeObjectFeatureObserverArtifact:
    """Cold-rebuild the raw packet, atlas, receipts, and full-catalog packets."""

    if not isinstance(artifact, PrototypeObjectFeatureObserverArtifact):
        raise TypeError("artifact must be PrototypeObjectFeatureObserverArtifact")
    artifact.assert_untampered()
    scene = _legacy._validate_exact_png(exact_scene_png_bytes, "scene")
    scene_digest = hashlib.sha256(scene).hexdigest()
    if (
        scene_digest != _digest(expected_scene_sha256, "expected raw scene sha256")
        or artifact.scene_digest != scene_digest
        or artifact.scene_id != expected_scene_id
        or artifact.observation_context_digest
        != _address(expected_observation_context_digest, "expected raw context digest")
        or artifact.artifact_digest
        != _digest(expected_artifact_digest, "expected raw feature artifact digest")
    ):
        raise PrototypeSceneObserverError("raw feature artifact differs from commitment")
    rebuilt = extract_object_hypotheses(scene)
    if rebuilt != artifact.hypothesis_packet:
        raise PrototypeSceneObserverError("cold raw hypothesis extraction differs")
    verify_object_hypothesis_packet(rebuilt, scene)
    atlas = render_object_hypothesis_atlas(rebuilt, scene)
    if artifact.presentation != _legacy._image_identities(atlas):
        raise PrototypeSceneObserverError("cold raw atlas differs")
    protocol = _object_protocol()
    plan = protocol.plan_prototype_object_feature_shards(rebuilt)
    if plan != artifact.feature_shard_plan:
        raise PrototypeSceneObserverError("cold raw shard plan differs")
    schema = protocol.prototype_object_feature_output_schema()
    atlas_by_name = dict(atlas)
    for shard in artifact.feature_shards:
        if shard.receipt is not None:
            assert shard.model_payload is not None
            _verify_receipt(
                shard.receipt,
                ((shard.spec.sheet_name, atlas_by_name[shard.spec.sheet_name]),),
                protocol.prototype_object_feature_shard_prompt(rebuilt, shard.spec),
                schema,
                shard.model_payload,
            )
    parsed = protocol.assemble_prototype_object_feature_shards(
        rebuilt,
        plan,
        _feature_shard_outcomes(rebuilt, artifact.feature_shards),
        feature_model_id=artifact.model,
    )
    if (
        tuple(parsed.packets) != artifact.local_packets
        or _audit_description(parsed.audit_description)
        != artifact.description_observation
    ):
        raise PrototypeSceneObserverError("cold raw feature assembly differs")
    decoded = PrototypeObjectFeatureObserverArtifact.from_data(
        artifact.to_data(), expected_artifact_digest=expected_artifact_digest
    )
    if decoded != artifact:
        raise PrototypeSceneObserverError("raw feature artifact cold round trip differs")
    return artifact


def _validate_scene_inputs(
    exact_scene_png_bytes: bytes,
    *,
    scene_task_id: str,
    scene_panel_id: str,
    observation_context_digest: str,
    expected_scene_sha256: str,
    catalog: PrototypeReferenceCatalog,
    expected_catalog_digest: str,
    rubric_artifact: PrototypeRubricDescriptionArtifact,
    expected_rubric_artifact_digest: str,
    prototype_png_by_panel_id: Mapping[str, bytes],
) -> bytes:
    scene = _legacy._validate_exact_png(exact_scene_png_bytes, "scene")
    if hashlib.sha256(scene).hexdigest() != _digest(expected_scene_sha256, "expected scene sha256"):
        raise PrototypeSceneObserverError("scene differs from external byte commitment")
    _address(observation_context_digest, "observation context digest")
    if (
        not isinstance(scene_task_id, str)
        or not scene_task_id
        or not isinstance(scene_panel_id, str)
        or not scene_panel_id.startswith(f"bd/{scene_task_id}/")
        or not scene_panel_id.endswith(".png")
    ):
        raise PrototypeSceneObserverError("scheduled scene identity is invalid")
    if catalog.catalog_digest != _digest(expected_catalog_digest, "expected catalog digest"):
        raise PrototypeSceneObserverError("catalog differs from commitment")
    verify_prototype_rubric_description_artifact(
        rubric_artifact,
        catalog,
        prototype_png_by_panel_id,
        expected_catalog_digest=expected_catalog_digest,
        expected_artifact_digest=expected_rubric_artifact_digest,
    )
    return scene


def observe_prototype_scene(
    exact_scene_png_bytes: bytes,
    *,
    scene_task_id: str,
    scene_panel_id: str,
    observation_context_digest: str,
    expected_scene_sha256: str,
    catalog: PrototypeReferenceCatalog,
    prototype_png_by_panel_id: Mapping[str, bytes],
    expected_catalog_digest: str,
    rubric_artifact: PrototypeRubricDescriptionArtifact,
    expected_rubric_artifact_digest: str,
    model: str,
    reasoning_effort: str = "medium",
    minutes: int = 15,
    verbose: bool = False,
    executable: str = "codex",
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    expected_launcher_digest: str | None = None,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    transport: NamedImageTransport = run_codex_named_images_structured,
) -> PrototypeSceneObserverArtifact:
    """Observe one scene once, profile-blind, then evaluate in pure Python."""

    if not callable(transport):
        raise TypeError("transport must be callable")
    scene = _validate_scene_inputs(
        exact_scene_png_bytes,
        scene_task_id=scene_task_id,
        scene_panel_id=scene_panel_id,
        observation_context_digest=observation_context_digest,
        expected_scene_sha256=expected_scene_sha256,
        catalog=catalog,
        expected_catalog_digest=expected_catalog_digest,
        rubric_artifact=rubric_artifact,
        expected_rubric_artifact_digest=expected_rubric_artifact_digest,
        prototype_png_by_panel_id=prototype_png_by_panel_id,
    )
    policy, model_catalog_digest, no_tools_digest = _runtime_identities(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    if rubric_artifact.status is not PrototypeSceneObserverStatus.SUCCESS:
        return _build_scene_artifact(
            status=PrototypeSceneObserverStatus.PREREQUISITE_ERROR,
            catalog=catalog,
            rubric_artifact=rubric_artifact,
            observation_context_digest=observation_context_digest,
            scene_task_id=scene_task_id,
            scene_panel_id=scene_panel_id,
            exact_scene_png_bytes=scene,
            raw_feature_artifact_digest=None,
            identities=(),
            model=model,
            reasoning_effort=reasoning_effort,
            expected_launcher_digest=expected_launcher_digest,
            cloud_policy_cache_binding=policy,
            model_catalog_digest=model_catalog_digest,
            no_tools_attestation_digest=no_tools_digest,
            payload=None,
            receipt=None,
            failure_code="profile_prerequisite_failed",
            failure_type="PrototypeObjectProfilePrerequisiteFailure",
            hypothesis_packet=None,
            feature_shard_plan=None,
            feature_shards=(),
            local_packets=(),
            evaluations=(),
            description_observation=PrototypeSceneDescriptionObservation.unavailable(
                "profile_prerequisite_failed", "PrototypeObjectProfilePrerequisiteFailure"
            ),
            scores=_error_scores("profile_prerequisite_failed", "PrototypeObjectProfilePrerequisiteFailure"),
        )
    raw_features = observe_prototype_object_features(
        scene,
        scene_id=scene_panel_id,
        observation_context_digest=observation_context_digest,
        expected_scene_sha256=expected_scene_sha256,
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
    local_packets = raw_features.local_packets
    evaluations = tuple(
        evaluate_object_profile(profile, local_packets)
        for profile in rubric_artifact.profiles
    )
    return _build_scene_artifact(
        status=PrototypeSceneObserverStatus.SUCCESS,
        catalog=catalog,
        rubric_artifact=rubric_artifact,
        observation_context_digest=observation_context_digest,
        scene_task_id=scene_task_id,
        scene_panel_id=scene_panel_id,
        exact_scene_png_bytes=scene,
        raw_feature_artifact_digest=raw_features.artifact_digest,
        identities=raw_features.presentation,
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=policy,
        model_catalog_digest=model_catalog_digest,
        no_tools_attestation_digest=no_tools_digest,
        payload=None,
        receipt=None,
        failure_code=None,
        failure_type=None,
        hypothesis_packet=raw_features.hypothesis_packet,
        feature_shard_plan=raw_features.feature_shard_plan,
        feature_shards=raw_features.feature_shards,
        local_packets=local_packets,
        evaluations=evaluations,
        description_observation=raw_features.description_observation,
        scores=_scores_from_evaluations(evaluations),
    )


def seal_prototype_scene_internal_error(
    exact_scene_png_bytes: bytes,
    *,
    scene_task_id: str,
    scene_panel_id: str,
    observation_context_digest: str,
    expected_scene_sha256: str,
    catalog: PrototypeReferenceCatalog,
    prototype_png_by_panel_id: Mapping[str, bytes],
    expected_catalog_digest: str,
    rubric_artifact: PrototypeRubricDescriptionArtifact,
    expected_rubric_artifact_digest: str,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str | None,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
    exception: Exception,
) -> PrototypeSceneObserverArtifact:
    """Seal unexpected local failure as exhaustive ERROR without absence."""

    if not isinstance(exception, Exception):
        raise TypeError("exception must be Exception")
    scene = _validate_scene_inputs(
        exact_scene_png_bytes,
        scene_task_id=scene_task_id,
        scene_panel_id=scene_panel_id,
        observation_context_digest=observation_context_digest,
        expected_scene_sha256=expected_scene_sha256,
        catalog=catalog,
        expected_catalog_digest=expected_catalog_digest,
        rubric_artifact=rubric_artifact,
        expected_rubric_artifact_digest=expected_rubric_artifact_digest,
        prototype_png_by_panel_id=prototype_png_by_panel_id,
    )
    policy, model_catalog_digest, no_tools_digest = _runtime_identities(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    return _build_scene_artifact(
        status=PrototypeSceneObserverStatus.INTERNAL_ERROR,
        catalog=catalog,
        rubric_artifact=rubric_artifact,
        observation_context_digest=observation_context_digest,
        scene_task_id=scene_task_id,
        scene_panel_id=scene_panel_id,
        exact_scene_png_bytes=scene,
        raw_feature_artifact_digest=None,
        identities=(),
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_binding=policy,
        model_catalog_digest=model_catalog_digest,
        no_tools_attestation_digest=no_tools_digest,
        payload=None,
        receipt=None,
        failure_code="observer_internal_error",
        failure_type=_legacy._exception_type(exception),
        hypothesis_packet=None,
        feature_shard_plan=None,
        feature_shards=(),
        local_packets=(),
        evaluations=(),
        description_observation=PrototypeSceneDescriptionObservation.unavailable(
            "observer_internal_error", "PrototypeSceneInternalError"
        ),
        scores=_error_scores("observer_internal_error", "PrototypeSceneInternalError"),
    )


def verify_prototype_scene_observer_artifact(
    artifact: PrototypeSceneObserverArtifact,
    exact_scene_png_bytes: bytes,
    *,
    expected_scene_task_id: str,
    expected_scene_panel_id: str,
    expected_observation_context_digest: str,
    expected_scene_sha256: str,
    catalog: PrototypeReferenceCatalog,
    prototype_png_by_panel_id: Mapping[str, bytes],
    expected_catalog_digest: str,
    rubric_artifact: PrototypeRubricDescriptionArtifact,
    expected_rubric_artifact_digest: str,
    expected_artifact_digest: str,
) -> PrototypeSceneObserverArtifact:
    """Cold-rebuild pixels, atlas, receipt, parse, and Python decisions."""

    if not isinstance(artifact, PrototypeSceneObserverArtifact):
        raise TypeError("artifact must be PrototypeSceneObserverArtifact")
    artifact.assert_untampered()
    scene = _validate_scene_inputs(
        exact_scene_png_bytes,
        scene_task_id=expected_scene_task_id,
        scene_panel_id=expected_scene_panel_id,
        observation_context_digest=expected_observation_context_digest,
        expected_scene_sha256=expected_scene_sha256,
        catalog=catalog,
        expected_catalog_digest=expected_catalog_digest,
        rubric_artifact=rubric_artifact,
        expected_rubric_artifact_digest=expected_rubric_artifact_digest,
        prototype_png_by_panel_id=prototype_png_by_panel_id,
    )
    if (
        artifact.artifact_digest != _digest(expected_artifact_digest, "expected scene artifact digest")
        or artifact.plan_digest != catalog.plan_digest
        or artifact.catalog_digest != catalog.catalog_digest
        or artifact.rubric_description_digest != rubric_artifact.artifact_digest
        or artifact.scene_task_id != expected_scene_task_id
        or artifact.scene_panel_id != expected_scene_panel_id
        or artifact.observation_context_digest != expected_observation_context_digest
        or artifact.scene_digest != hashlib.sha256(scene).hexdigest()
    ):
        raise PrototypeSceneObserverError("scene differs from cold parent reconstruction")
    if artifact.status is PrototypeSceneObserverStatus.SUCCESS:
        assert artifact.hypothesis_packet is not None
        rebuilt = extract_object_hypotheses(scene)
        if rebuilt != artifact.hypothesis_packet:
            raise PrototypeSceneObserverError("cold hypothesis extraction differs")
        verify_object_hypothesis_packet(rebuilt, scene)
        atlas = render_object_hypothesis_atlas(rebuilt, scene)
        if artifact.presentation != _legacy._image_identities(atlas):
            raise PrototypeSceneObserverError("cold atlas presentation differs")
        prompt, schema = _scene_prompt_schema(rebuilt)
        if (
            artifact.prompt_digest != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or artifact.output_schema_digest != canonical_digest(schema)
        ):
            raise PrototypeSceneObserverError("cold feature protocol differs")
        protocol = _object_protocol()
        rebuilt_plan = protocol.plan_prototype_object_feature_shards(rebuilt)
        if rebuilt_plan != artifact.feature_shard_plan:
            raise PrototypeSceneObserverError("cold feature shard plan differs")
        atlas_by_name = dict(atlas)
        for shard in artifact.feature_shards:
            if shard.receipt is not None:
                assert shard.model_payload is not None
                shard_prompt = protocol.prototype_object_feature_shard_prompt(
                    rebuilt, shard.spec
                )
                _verify_receipt(
                    shard.receipt,
                    ((shard.spec.sheet_name, atlas_by_name[shard.spec.sheet_name]),),
                    shard_prompt,
                    schema,
                    shard.model_payload,
                )
        outcomes = _feature_shard_outcomes(rebuilt, artifact.feature_shards)
        parsed = protocol.assemble_prototype_object_feature_shards(
            rebuilt,
            rebuilt_plan,
            outcomes,
            feature_model_id=artifact.model,
        )
        packets = tuple(parsed.packets)
        if packets != artifact.local_packets:
            raise PrototypeSceneObserverError("cold local feature packets differ")
        evaluations = tuple(
            evaluate_object_profile(profile, packets)
            for profile in rubric_artifact.profiles
        )
        if evaluations != artifact.evaluations:
            raise PrototypeSceneObserverError("cold profile evaluations differ")
        for profile, evaluation in zip(
            rubric_artifact.profiles, evaluations, strict=True
        ):
            verify_object_profile_evaluation(
                evaluation, profile=profile, packets=packets
            )
        if (
            _audit_description(parsed.audit_description)
            != artifact.description_observation
            or _scores_from_evaluations(evaluations) != artifact.scores
        ):
            raise PrototypeSceneObserverError("cold score mapping differs")
        rebuilt_raw = _build_raw_feature_artifact(
            observation_context_digest=artifact.observation_context_digest,
            scene_id=artifact.scene_panel_id,
            scene=scene,
            packet=rebuilt,
            identities=artifact.presentation,
            plan=rebuilt_plan,
            shards=artifact.feature_shards,
            local_packets=artifact.local_packets,
            description_observation=artifact.description_observation,
            model=artifact.model,
            reasoning_effort=artifact.reasoning_effort,
            expected_launcher_digest=artifact.expected_launcher_digest,
            cloud_policy_cache_binding=artifact.cloud_policy_cache_binding,
            model_catalog_digest=artifact.model_catalog_digest,
            no_tools_attestation_digest=artifact.no_tools_attestation_digest,
        )
        if rebuilt_raw.artifact_digest != artifact.raw_feature_artifact_digest:
            raise PrototypeSceneObserverError("cold raw feature commitment differs")
    else:
        prompt, schema = _scene_prompt_schema(None)
        if (
            artifact.prompt_digest != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or artifact.output_schema_digest != canonical_digest(schema)
        ):
            raise PrototypeSceneObserverError("unattempted protocol marker differs")
    decoded = PrototypeSceneObserverArtifact.from_data(
        artifact.to_data(), expected_artifact_digest=expected_artifact_digest
    )
    if decoded != artifact:
        raise PrototypeSceneObserverError("scene cold round trip differs")
    return artifact


def prototype_scene_observer_prompt(packet: ObjectHypothesisPacket) -> str:
    return _scene_prompt_schema(packet)[0]


def prototype_scene_observer_output_schema(
    packet: ObjectHypothesisPacket | None = None,
) -> dict[str, object]:
    # The strict schema is packet-independent; exhaustive length/order is
    # enforced by the packet-bound parser and protocol digest.
    return _object_protocol().prototype_object_feature_output_schema()


observe_prototype_whole_scene = observe_prototype_scene


__all__ = (
    "PPM_SCALE",
    "PROTOTYPE_GROUP_IDS",
    "PROTOTYPE_OBJECT_FEATURE_OBSERVER_ARTIFACT_SCHEMA",
    "PROTOTYPE_OBJECT_FEATURE_OBSERVER_PROTOCOL_ID",
    "PROTOTYPE_OBJECT_FEATURE_SHARD_ARTIFACT_SCHEMA",
    "PROTOTYPE_REFERENCE_CATALOG_SCHEMA",
    "PROTOTYPE_RUBRIC_DESCRIPTION_ARTIFACT_SCHEMA",
    "PROTOTYPE_SCENE_OBSERVER_ARTIFACT_SCHEMA",
    "PROTOTYPE_SCENE_OBSERVER_PROTOCOL_ID",
    "NamedImageTransport",
    "PrototypeImageIdentity",
    "PrototypeObjectFeatureObserverArtifact",
    "PrototypeObjectFeatureShardArtifact",
    "PrototypeReferenceBinding",
    "PrototypeReferenceCatalog",
    "PrototypeRubric",
    "PrototypeRubricDescriptionArtifact",
    "PrototypeRubricState",
    "PrototypeSceneDescriptionObservation",
    "PrototypeSceneDescriptionState",
    "PrototypeSceneObserverArtifact",
    "PrototypeSceneObserverError",
    "PrototypeSceneObserverStatus",
    "PrototypeScenePayloadError",
    "PrototypeSceneScore",
    "PrototypeSceneScoreState",
    "build_prototype_reference_catalog",
    "describe_prototype_references",
    "observe_prototype_object_features",
    "observe_prototype_scene",
    "observe_prototype_whole_scene",
    "prototype_rubric_description_output_schema",
    "prototype_rubric_description_prompt",
    "prototype_rubric_description_protocol_digest",
    "prototype_object_feature_observer_protocol_digest",
    "prototype_scene_observer_environment_digest",
    "prototype_scene_observer_model_digest",
    "prototype_scene_observer_output_schema",
    "prototype_scene_observer_prompt",
    "prototype_scene_observer_source_digest",
    "prototype_scene_scoring_protocol_digest",
    "prototype_scene_transport_source_digest",
    "seal_prototype_rubric_description_internal_error",
    "seal_prototype_scene_internal_error",
    "verify_prototype_reference_catalog",
    "verify_prototype_object_feature_observer_artifact",
    "verify_prototype_rubric_description_artifact",
    "verify_prototype_scene_observer_artifact",
)

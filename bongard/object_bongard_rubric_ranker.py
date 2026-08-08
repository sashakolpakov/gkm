"""Text-only Codex ranking over verified rubric-predicate survivors.

The closed Python rubric version space supplies the exact survivor set.  This
module exposes only an opaque version-space digest, the frozen ordered
target-versus-foil rubric derived from the neutral groups' catalog cue IDs,
and immutable candidate aliases/formulas to Codex.  The semantic model's free
prose is never shown to the ranker.  Codex may order the aliases; it cannot
create or edit a predicate.  Python checks the exact permutation, resolves
aliases to candidate digests, and cold verifies the complete causal receipt
without another model call.

No pixels, panel identities, support-side names, held-out material, or Lean
content enter the model-visible prompt.  Python remains the sole predicate,
identity, ranking-resolution, and replay authority.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

import bongard.transport as _transport_module
from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import (
    CodexNoToolsAttestation,
    validate_codex_no_tools_attestation,
)
from bongard.object_bongard_rubric_observer import (
    ObjectBongardRubricObserverArtifact,
    ObjectBongardRubricSpec,
    RUBRIC_ORDINAL_LEVEL_ANCHORS,
    object_bongard_catalog_contrast_rubric,
    object_bongard_catalog_cue_rubric,
    object_bongard_rubric_ordinal_scale_digest,
)
from bongard.object_bongard_semantics import ObjectBongardSemanticArtifact
from bongard.object_bongard_rubric_version_space import (
    ObjectBongardRubricCandidate,
    ObjectBongardRubricSupportVersionSpace,
    cold_verify_object_bongard_rubric_support_version_space,
    object_bongard_rubric_version_space_algorithm_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.prototype_object_profiles import OBJECT_FEATURE_CATALOG_DIGEST
from bongard.transport import (
    CODEX_ISOLATION_POLICY,
    CODEX_RECEIPT_SCHEMA,
    REASONING_EFFORTS,
    TEXT_STRUCTURED_INPUT_DIGEST_SCHEMA,
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexProposerFailure,
    CodexReceipt,
    CodexStructuredResult,
    run_codex_text_structured,
    validate_codex_strict_output_schema,
    validate_codex_text_receipt,
)


OBJECT_BONGARD_RUBRIC_RANKER_PROTOCOL_ID = (
    "bongard.object-rubric-version-space/text-only-contrastive-codex-ranker-v2"
)
OBJECT_BONGARD_RUBRIC_RANKER_PROTOCOL_SCHEMA = (
    "gkm.bongard-object-rubric-ranker-protocol.v2"
)
OBJECT_BONGARD_RUBRIC_RANK_INPUT_SCHEMA = (
    "gkm.bongard-object-rubric-rank-input.v2"
)
OBJECT_BONGARD_RUBRIC_RANK_OUTPUT_SCHEMA = (
    "gkm.bongard-object-rubric-rank-output.v1"
)
OBJECT_BONGARD_RUBRIC_RANK_RESPONSE_SCHEMA = (
    "gkm.bongard-object-rubric-rank-response.v1"
)
OBJECT_BONGARD_RUBRIC_RANK_RECEIPT_SCHEMA = (
    "gkm.bongard-object-rubric-rank-receipt.v1"
)

MAX_SURVIVOR_COUNT = 2
MAX_PROMPT_UTF8_BYTES = 64_000
MAX_RUBRIC_UTF8_BYTES = 2_048

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}\Z")
_FORBIDDEN_ROLE_REFERENCE = re.compile(
    r"(?:\bgroup[_ -]?[01](?:_ref)?\b|"
    r"\b(?:positive|negative)\s+(?:side|support|example)s?\b|"
    r"\bsupport\s+(?:side|label)s?\b|"
    r"\bquery\s+(?:panel|item|input|example)s?\b)",
    re.IGNORECASE,
)


class ObjectBongardRubricRankerError(RuntimeError):
    """A rubric-rank input, output, receipt, or replay pin is invalid."""


TextStructuredTransport = Callable[..., CodexStructuredResult]


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "codex_may_rank_verified_survivors_only": True,
        "codex_may_edit_candidate_formulas": False,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_ranking_or_replay": False,
    }


def object_bongard_rubric_ranker_authority_data() -> dict[str, object]:
    return _authority_data()


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectBongardRubricRankerError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ObjectBongardRubricRankerError(f"{label} must be a bounded identifier")
    return value


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectBongardRubricRankerError(f"{label} fields differ from schema")
    return value


def object_bongard_rubric_ranker_source_digest() -> str:
    return verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )


def object_bongard_rubric_ranker_transport_source_digest() -> str:
    source = getattr(_transport_module, "__file__", None)
    if not isinstance(source, str) or not source:
        raise ObjectBongardRubricRankerError("text transport source is unavailable")
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def _freeze_spec(spec: object) -> ObjectBongardRubricSpec:
    if not isinstance(spec, ObjectBongardRubricSpec):
        raise TypeError("rubric_spec must be ObjectBongardRubricSpec")
    try:
        restored = ObjectBongardRubricSpec.from_data(spec.to_data())
    except Exception as exc:
        raise ObjectBongardRubricRankerError(
            "rubric spec is not canonical"
        ) from exc
    if restored != spec:
        raise ObjectBongardRubricRankerError("rubric spec cold round trip differs")
    _digest(spec.spec_digest, "rubric spec digest")
    _digest(spec.semantic_artifact_digest, "semantic artifact digest")
    if (
        not isinstance(spec.feature_nominations, tuple)
        or not spec.feature_nominations
        or any(not isinstance(item, str) or not item for item in spec.feature_nominations)
        or len(set(spec.feature_nominations)) != len(spec.feature_nominations)
    ):
        raise ObjectBongardRubricRankerError(
            "rubric feature nominations are not canonical"
        )
    _neutral_rubric(spec.rubric, "contrastive observer rubric")
    return spec


def _neutral_rubric(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value
        or not value.isprintable()
        or len(value.encode("utf-8")) > MAX_RUBRIC_UTF8_BYTES
        or _FORBIDDEN_ROLE_REFERENCE.search(value) is not None
    ):
        raise ObjectBongardRubricRankerError(
            f"{label} violates the neutral text-only boundary"
        )
    return value


def _ordinal_scale() -> tuple[str, tuple[tuple[int, str], ...]]:
    try:
        anchors = tuple(
            (level, _neutral_rubric(text, f"rubric level {level} anchor"))
            for level, text in RUBRIC_ORDINAL_LEVEL_ANCHORS
        )
    except (TypeError, ValueError) as exc:
        raise ObjectBongardRubricRankerError(
            "rubric ordinal scale anchors are malformed"
        ) from exc
    if (
        tuple(level for level, _ in anchors) != tuple(range(5))
        or len({text for _, text in anchors}) != len(anchors)
    ):
        raise ObjectBongardRubricRankerError(
            "rubric ordinal scale must define unique ordered levels 0..4"
        )
    scale_digest = _digest(
        object_bongard_rubric_ordinal_scale_digest(),
        "rubric ordinal scale digest",
    )
    return scale_digest, anchors


def _freeze_semantic_artifact(
    value: object, spec: ObjectBongardRubricSpec
) -> ObjectBongardSemanticArtifact:
    if not isinstance(value, ObjectBongardSemanticArtifact):
        raise TypeError("semantic_artifact must be ObjectBongardSemanticArtifact")
    try:
        restored = ObjectBongardSemanticArtifact.from_data(
            value.to_data(), expected_artifact_digest=value.artifact_digest
        )
    except Exception as exc:
        raise ObjectBongardRubricRankerError(
            "semantic artifact is not canonical"
        ) from exc
    if restored != value:
        raise ObjectBongardRubricRankerError(
            "semantic artifact cold round trip differs"
        )
    try:
        expected_spec = ObjectBongardRubricSpec.from_semantic_artifact(
            value, expected_artifact_digest=value.artifact_digest
        )
    except Exception as exc:
        raise ObjectBongardRubricRankerError(
            "semantic artifact cannot derive a frozen catalog rubric"
        ) from exc
    if (
        value.artifact_digest != spec.semantic_artifact_digest
        or value.feature_catalog_digest != OBJECT_FEATURE_CATALOG_DIGEST
        or expected_spec != spec
    ):
        raise ObjectBongardRubricRankerError(
            "rubric spec is not exactly derived from the ordered semantic cues"
        )
    target_id, target_rubric, contrast_id, contrast_rubric = (
        _semantic_catalog_cues(value)
    )
    if (
        spec.feature_nominations != (target_id, contrast_id)
        or spec.rubric
        != object_bongard_catalog_contrast_rubric(target_id, contrast_id)
        or contrast_id == target_id
        or contrast_rubric == target_rubric
    ):
        raise ObjectBongardRubricRankerError(
            "target and foil cues do not derive the exact ordered catalog contrast"
        )
    return value


def _semantic_catalog_cues(
    semantic: ObjectBongardSemanticArtifact,
) -> tuple[str, str, str, str]:
    if (
        semantic.feature_catalog_digest != OBJECT_FEATURE_CATALOG_DIGEST
        or len(semantic.feature_families) != 2
        or any(len(group) != 1 for group in semantic.feature_families)
    ):
        raise ObjectBongardRubricRankerError(
            "semantic artifact must bind one frozen catalog cue per group"
        )
    target_id = semantic.feature_families[0][0]
    contrast_id = semantic.feature_families[1][0]
    try:
        target_rubric = _neutral_rubric(
            object_bongard_catalog_cue_rubric(target_id),
            "target catalog cue rubric",
        )
        contrast_rubric = _neutral_rubric(
            object_bongard_catalog_cue_rubric(contrast_id),
            "neutral contrast catalog cue rubric",
        )
    except (TypeError, ValueError) as exc:
        raise ObjectBongardRubricRankerError(
            "semantic cue does not resolve in the frozen feature catalog"
        ) from exc
    return target_id, target_rubric, contrast_id, contrast_rubric


def _freeze_version_space(
    value: object,
    spec: ObjectBongardRubricSpec,
    positive_support_artifacts: Sequence[ObjectBongardRubricObserverArtifact],
    negative_support_artifacts: Sequence[ObjectBongardRubricObserverArtifact],
) -> tuple[
    ObjectBongardRubricSupportVersionSpace,
    tuple[ObjectBongardRubricCandidate, ...],
    tuple[ObjectBongardRubricObserverArtifact, ...],
    tuple[ObjectBongardRubricObserverArtifact, ...],
]:
    if not isinstance(value, ObjectBongardRubricSupportVersionSpace):
        raise TypeError(
            "version_space must be ObjectBongardRubricSupportVersionSpace"
        )
    try:
        positives = tuple(positive_support_artifacts)
        negatives = tuple(negative_support_artifacts)
    except TypeError as exc:
        raise ObjectBongardRubricRankerError(
            "rubric support artifacts must be finite sequences"
        ) from exc
    frozen_sides: list[tuple[ObjectBongardRubricObserverArtifact, ...]] = []
    for label, artifacts in (("positive", positives), ("negative", negatives)):
        frozen: list[ObjectBongardRubricObserverArtifact] = []
        for artifact in artifacts:
            if not isinstance(artifact, ObjectBongardRubricObserverArtifact):
                raise TypeError(
                    f"{label}_support_artifacts must contain "
                    "ObjectBongardRubricObserverArtifact"
                )
            try:
                restored_artifact = ObjectBongardRubricObserverArtifact.from_data(
                    artifact.to_data()
                )
            except Exception as exc:
                raise ObjectBongardRubricRankerError(
                    f"{label} rubric support artifact is not canonical"
                ) from exc
            if restored_artifact != artifact:
                raise ObjectBongardRubricRankerError(
                    f"{label} rubric support artifact cold round trip differs"
                )
            frozen.append(restored_artifact)
        frozen_sides.append(tuple(frozen))
    positives, negatives = frozen_sides
    try:
        restored = ObjectBongardRubricSupportVersionSpace.from_data(value.to_data())
    except Exception as exc:
        raise ObjectBongardRubricRankerError(
            "rubric version space is not canonical"
        ) from exc
    if restored != value:
        raise ObjectBongardRubricRankerError(
            "rubric version-space cold round trip differs"
        )
    if value.rubric_spec_digest != spec.spec_digest:
        raise ObjectBongardRubricRankerError(
            "rubric version space belongs to a different spec"
        )
    try:
        verified = cold_verify_object_bongard_rubric_support_version_space(
            restored,
            spec,
            positives,
            negatives,
        )
    except Exception as exc:
        raise ObjectBongardRubricRankerError(
            "rubric version space does not cold-replay from the exact support "
            "artifacts"
        ) from exc
    if verified != restored:
        raise ObjectBongardRubricRankerError(
            "cold-verified rubric version space differs"
        )
    survivors = tuple(
        verified.survivor(item) for item in verified.survivor_candidate_digests
    )
    if not 1 <= len(survivors) <= MAX_SURVIVOR_COUNT:
        raise ObjectBongardRubricRankerError(
            "rubric ranker requires between one and two verified survivors"
        )
    if (
        tuple(item.candidate_digest for item in survivors)
        != verified.survivor_candidate_digests
        or len({item.candidate_digest for item in survivors}) != len(survivors)
        or len({item.candidate_id for item in survivors}) != len(survivors)
        or len({item.formula for item in survivors}) != len(survivors)
    ):
        raise ObjectBongardRubricRankerError(
            "verified survivor identity or formula inventory differs"
        )
    for candidate in survivors:
        try:
            if ObjectBongardRubricCandidate.from_data(
                candidate.to_data()
            ) != candidate:
                raise ObjectBongardRubricRankerError(
                    "rubric candidate is not canonical"
                )
        except Exception as exc:
            if isinstance(exc, ObjectBongardRubricRankerError):
                raise
            raise ObjectBongardRubricRankerError(
                "rubric candidate is not canonical"
            ) from exc
        if candidate.rubric_spec_digest != spec.spec_digest:
            raise ObjectBongardRubricRankerError(
                "rubric candidate belongs to a different spec"
            )
    return verified, survivors, positives, negatives


def object_bongard_rubric_rank_input_digest(
    *,
    version_space: ObjectBongardRubricSupportVersionSpace,
    rubric_spec: ObjectBongardRubricSpec,
    semantic_artifact: ObjectBongardSemanticArtifact,
    positive_support_artifacts: Sequence[ObjectBongardRubricObserverArtifact],
    negative_support_artifacts: Sequence[ObjectBongardRubricObserverArtifact],
) -> str:
    spec = _freeze_spec(rubric_spec)
    semantic = _freeze_semantic_artifact(semantic_artifact, spec)
    version, survivors, _, _ = _freeze_version_space(
        version_space,
        spec,
        positive_support_artifacts,
        negative_support_artifacts,
    )
    return _rank_input_digest_from_frozen(version, spec, semantic, survivors)


def _rank_input_digest_from_frozen(
    version: ObjectBongardRubricSupportVersionSpace,
    spec: ObjectBongardRubricSpec,
    semantic: ObjectBongardSemanticArtifact,
    survivors: tuple[ObjectBongardRubricCandidate, ...],
) -> str:
    scale_digest, anchors = _ordinal_scale()
    target_id, target_rubric, contrast_id, contrast_rubric = (
        _semantic_catalog_cues(semantic)
    )
    return canonical_digest(
        {
            "schema": OBJECT_BONGARD_RUBRIC_RANK_INPUT_SCHEMA,
            "rubric_spec_digest": spec.spec_digest,
            "semantic_artifact_digest": semantic.artifact_digest,
            "feature_catalog_digest": OBJECT_FEATURE_CATALOG_DIGEST,
            "contrastive_observer_rubric": spec.rubric,
            "target_rubric": target_rubric,
            "target_feature_nominations": [target_id],
            "neutral_contrast_rubric": contrast_rubric,
            "neutral_contrast_feature_nominations": [contrast_id],
            "catalog_rubric_derivation_verified": True,
            "semantic_audit_rubrics_model_visible": False,
            "rubric_ordinal_scale_digest": scale_digest,
            "rubric_ordinal_level_anchors": [
                {"level": level, "meaning": text} for level, text in anchors
            ],
            "version_space_digest": version.version_space_digest,
            "ordered_verified_survivors": [item.to_data() for item in survivors],
            "model_visible_image_material": False,
            "model_visible_panel_identities": False,
            "model_visible_support_sides": False,
            "model_visible_held_out_material": False,
            "model_visible_lean_material": False,
            **_authority_data(),
        }
    )


def _rank_inputs(
    *,
    version_space: ObjectBongardRubricSupportVersionSpace,
    rubric_spec: ObjectBongardRubricSpec,
    semantic_artifact: ObjectBongardSemanticArtifact,
    positive_support_artifacts: Sequence[ObjectBongardRubricObserverArtifact],
    negative_support_artifacts: Sequence[ObjectBongardRubricObserverArtifact],
    rank_input_digest: str,
) -> tuple[
    ObjectBongardRubricSupportVersionSpace,
    ObjectBongardRubricSpec,
    ObjectBongardSemanticArtifact,
    tuple[ObjectBongardRubricCandidate, ...],
    tuple[ObjectBongardRubricObserverArtifact, ...],
    tuple[ObjectBongardRubricObserverArtifact, ...],
]:
    spec = _freeze_spec(rubric_spec)
    semantic = _freeze_semantic_artifact(semantic_artifact, spec)
    version, survivors, positives, negatives = _freeze_version_space(
        version_space,
        spec,
        positive_support_artifacts,
        negative_support_artifacts,
    )
    supplied = _digest(rank_input_digest, "rank input digest")
    expected = _rank_input_digest_from_frozen(version, spec, semantic, survivors)
    if supplied != expected:
        raise ObjectBongardRubricRankerError(
            "rubric rank input digest differs from its canonical preimage"
        )
    return version, spec, semantic, survivors, positives, negatives


def _aliases(count: int) -> tuple[str, ...]:
    if not 1 <= count <= MAX_SURVIVOR_COUNT:
        raise ObjectBongardRubricRankerError("rubric alias count is outside bounds")
    return tuple(f"r{index:03d}" for index in range(count))


def object_bongard_rubric_ranker_prompt(
    *,
    version_space: ObjectBongardRubricSupportVersionSpace,
    rubric_spec: ObjectBongardRubricSpec,
    semantic_artifact: ObjectBongardSemanticArtifact,
    positive_support_artifacts: Sequence[ObjectBongardRubricObserverArtifact],
    negative_support_artifacts: Sequence[ObjectBongardRubricObserverArtifact],
    rank_input_digest: str,
) -> str:
    version, spec, semantic, survivors, positives, negatives = _rank_inputs(
        version_space=version_space,
        rubric_spec=rubric_spec,
        semantic_artifact=semantic_artifact,
        positive_support_artifacts=positive_support_artifacts,
        negative_support_artifacts=negative_support_artifacts,
        rank_input_digest=rank_input_digest,
    )
    return _ranker_prompt_from_frozen(
        version,
        spec,
        semantic,
        survivors,
        positives,
        negatives,
        rank_input_digest,
    )


def _ranker_prompt_from_frozen(
    version: ObjectBongardRubricSupportVersionSpace,
    spec: ObjectBongardRubricSpec,
    semantic: ObjectBongardSemanticArtifact,
    survivors: tuple[ObjectBongardRubricCandidate, ...],
    positives: tuple[ObjectBongardRubricObserverArtifact, ...],
    negatives: tuple[ObjectBongardRubricObserverArtifact, ...],
    rank_input_digest: str,
) -> str:
    rows = "\n".join(
        (
            f"- {alias}; candidate_id={candidate.candidate_id}; "
            f"digest={candidate.candidate_digest}; formula={candidate.formula}"
        )
        for alias, candidate in zip(_aliases(len(survivors)), survivors, strict=True)
    )
    scale_digest, anchors = _ordinal_scale()
    target_id, target_rubric, contrast_id, contrast_rubric = (
        _semantic_catalog_cues(semantic)
    )
    anchor_rows = "\n".join(
        f"- level={level}; meaning={text}" for level, text in anchors
    )
    prompt = (
        "Rank the already-admissible immutable candidates by how naturally "
        "each fixed scope and threshold operationalizes the target frozen "
        "catalog cue rubric against the neutral frozen catalog contrast cue "
        "rubric. Return every bounded alias "
        "exactly once, best first. Do not change a scope, operator, threshold, "
        "formula, identity, or polarity. Use only the material below and return "
        "no explanation.\n\n"
        f"rank_input_digest: {rank_input_digest}\n"
        f"version_space_digest: {version.version_space_digest}\n"
        f"rubric_spec_digest: {spec.spec_digest}\n"
        f"feature_catalog_digest: {OBJECT_FEATURE_CATALOG_DIGEST}\n"
        f"contrastive_observer_rubric: {spec.rubric}\n"
        f"target_cue_id: {target_id}\n"
        f"target_rubric: {target_rubric}\n"
        f"neutral_contrast_cue_id: {contrast_id}\n"
        f"neutral_contrast_rubric: {contrast_rubric}\n\n"
        f"rubric_ordinal_scale_digest: {scale_digest}\n"
        f"rubric_ordinal_level_anchors:\n{anchor_rows}\n\n"
        f"immutable_candidates:\n{rows}"
    )
    if len(prompt.encode("utf-8")) > MAX_PROMPT_UTF8_BYTES:
        raise ObjectBongardRubricRankerError(
            "rubric rank prompt exceeds its byte guard"
        )
    if _FORBIDDEN_ROLE_REFERENCE.search(prompt) is not None:
        raise ObjectBongardRubricRankerError(
            "rubric rank prompt crosses the sealed text-only boundary"
        )
    for artifact in positives + negatives:
        if artifact.artifact_digest in prompt or (
            len(artifact.panel_id) >= 8 and artifact.panel_id in prompt
        ):
            raise ObjectBongardRubricRankerError(
                "rubric rank prompt exposes support artifact identity"
            )
    return prompt


def object_bongard_rubric_ranker_output_schema() -> dict[str, object]:
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "ordered_aliases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Every supplied bounded alias exactly once, best first.",
            }
        },
        "required": ["ordered_aliases"],
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(schema)
    return schema


def object_bongard_rubric_ranker_protocol_digest() -> str:
    scale_digest, anchors = _ordinal_scale()
    return canonical_digest(
        {
            "schema": OBJECT_BONGARD_RUBRIC_RANKER_PROTOCOL_SCHEMA,
            "protocol_id": OBJECT_BONGARD_RUBRIC_RANKER_PROTOCOL_ID,
            "source_digest": object_bongard_rubric_ranker_source_digest(),
            "transport_source_digest": (
                object_bongard_rubric_ranker_transport_source_digest()
            ),
            "transport_entrypoint": "run_codex_text_structured",
            "receipt_schema": CODEX_RECEIPT_SCHEMA,
            "input_digest_schema": TEXT_STRUCTURED_INPUT_DIGEST_SCHEMA,
            "isolation_policy": CODEX_ISOLATION_POLICY,
            "output_schema": object_bongard_rubric_ranker_output_schema(),
            "output_rule": "exact-alias-permutation-resolved-to-candidate-digests",
            "maximum_survivor_count": MAX_SURVIVOR_COUNT,
            "maximum_prompt_utf8_bytes": MAX_PROMPT_UTF8_BYTES,
            "version_space_algorithm_digest": (
                object_bongard_rubric_version_space_algorithm_digest()
            ),
            "feature_catalog_digest": OBJECT_FEATURE_CATALOG_DIGEST,
            "semantic_cue_policy": (
                "ordered-group-zero-target-group-one-foil-one-distinct-"
                "feature-id-per-neutral-group"
            ),
            "rubric_grounding_policy": (
                "exact-frozen-catalog-ordered-target-versus-foil-contrast"
            ),
            "semantic_audit_prose_model_visible": False,
            "rubric_ordinal_scale_digest": scale_digest,
            "rubric_ordinal_level_anchors": [
                {"level": level, "meaning": text} for level, text in anchors
            ],
            "candidate_formula_language": (
                "immutable-object-or-scene-rubric-level-at-least-threshold"
            ),
            "model_visible_image_material": False,
            "model_visible_panel_identities": False,
            "model_visible_support_sides": False,
            "model_visible_held_out_material": False,
            "model_visible_lean_material": False,
            **_authority_data(),
        }
    )


def object_bongard_rubric_ranker_model_identity_digest(
    model: str, reasoning_effort: str
) -> str:
    if not isinstance(model, str) or _MODEL.fullmatch(model) is None:
        raise ObjectBongardRubricRankerError("rubric ranker model is invalid")
    if reasoning_effort not in REASONING_EFFORTS:
        raise ObjectBongardRubricRankerError(
            "rubric ranker reasoning effort is invalid"
        )
    return canonical_digest(
        {
            "schema": "gkm.bongard-object-rubric-ranker-model-request.v1",
            "model": model,
            "reasoning_effort": reasoning_effort,
            "identity_evidence_policy": (
                "receipt-reported-model-or-explicit-cli-model-flag"
            ),
        }
    )


def object_bongard_rubric_ranker_environment_digest(
    *,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str,
    expected_cloud_policy_cache_binding: str,
    expected_transport_source_digest: str,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
) -> str:
    model_digest = object_bongard_rubric_ranker_model_identity_digest(
        model, reasoning_effort
    )
    launcher = _digest(expected_launcher_digest, "expected launcher digest")
    transport_source = _digest(
        expected_transport_source_digest, "expected transport source digest"
    )
    if transport_source != object_bongard_rubric_ranker_transport_source_digest():
        raise ObjectBongardRubricRankerError(
            "text transport source differs from external commitment"
        )
    policy = expected_cloud_policy_cache_binding
    if policy != "absent" and (
        not isinstance(policy, str)
        or not policy.startswith("sha256:")
        or _DIGEST.fullmatch(policy[7:]) is None
    ):
        raise ObjectBongardRubricRankerError(
            "expected policy-cache binding is invalid"
        )
    if not isinstance(model_catalog_snapshot, CodexModelCatalogSnapshot):
        raise ObjectBongardRubricRankerError(
            "exact Codex model catalog snapshot is required"
        )
    try:
        attestation = validate_codex_no_tools_attestation(
            no_tools_attestation,
            expected_launcher_digest=launcher,
            expected_model_catalog_digest=model_catalog_snapshot.raw_digest,
            expected_cloud_policy_cache_binding=policy,
        )
    except (CodexProposerFailure, TypeError, ValueError) as exc:
        raise ObjectBongardRubricRankerError(
            "Codex no-tools runtime differs from its frozen attestation"
        ) from exc
    return canonical_digest(
        {
            "schema": "gkm.bongard-object-rubric-ranker-environment.v1",
            "model_identity_digest": model_digest,
            "expected_launcher_digest": launcher,
            "expected_cloud_policy_cache_binding": policy,
            "model_catalog_digest": model_catalog_snapshot.raw_digest,
            "no_tools_attestation_digest": attestation.attestation_digest,
            "ranker_source_digest": object_bongard_rubric_ranker_source_digest(),
            "transport_source_digest": transport_source,
            "receipt_schema": CODEX_RECEIPT_SCHEMA,
            "input_digest_schema": TEXT_STRUCTURED_INPUT_DIGEST_SCHEMA,
            "isolation_policy": CODEX_ISOLATION_POLICY,
            **_authority_data(),
        }
    )


def _canonical_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ObjectBongardRubricRankerError(
            "rubric rank payload must be a JSON object"
        )
    try:
        decoded = json.loads(canonical_json(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardRubricRankerError(
            "rubric rank payload is not finite canonical JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise ObjectBongardRubricRankerError(
            "rubric rank payload must be an object"
        )
    return decoded


def _parse_alias_payload(
    payload: Mapping[str, Any],
    survivors: tuple[ObjectBongardRubricCandidate, ...],
) -> tuple[str, ...]:
    if set(payload) != {"ordered_aliases"}:
        raise ObjectBongardRubricRankerError(
            "rubric rank payload fields differ from schema"
        )
    values = payload["ordered_aliases"]
    if not isinstance(values, list) or any(
        not isinstance(item, str) for item in values
    ):
        raise ObjectBongardRubricRankerError(
            "ordered rubric aliases must be a JSON list"
        )
    ordered = tuple(values)
    aliases = _aliases(len(survivors))
    if (
        len(ordered) != len(aliases)
        or len(set(ordered)) != len(ordered)
        or set(ordered) != set(aliases)
    ):
        raise ObjectBongardRubricRankerError(
            "rubric rank payload must be the exact survivor-alias permutation"
        )
    by_alias = {
        alias: candidate.candidate_digest
        for alias, candidate in zip(aliases, survivors, strict=True)
    }
    return tuple(by_alias[item] for item in ordered)


def _ordered_alias_payload(
    ordered_candidate_digests: Sequence[str],
    survivors: tuple[ObjectBongardRubricCandidate, ...],
) -> dict[str, object]:
    digest_to_alias = {
        candidate.candidate_digest: alias
        for alias, candidate in zip(
            _aliases(len(survivors)), survivors, strict=True
        )
    }
    try:
        ordered = [digest_to_alias[item] for item in ordered_candidate_digests]
    except KeyError as exc:
        raise ObjectBongardRubricRankerError(
            "rubric rank response contains a foreign candidate digest"
        ) from exc
    return {"ordered_aliases": ordered}


def _rank_output_digest(ordered_candidate_digests: Sequence[str]) -> str:
    return canonical_digest(
        {
            "schema": OBJECT_BONGARD_RUBRIC_RANK_OUTPUT_SCHEMA,
            "ordered_candidate_digests": list(ordered_candidate_digests),
        }
    )


def _response_content(value: "ObjectBongardRubricRankResponse") -> dict[str, object]:
    return {
        "schema": OBJECT_BONGARD_RUBRIC_RANK_RESPONSE_SCHEMA,
        "ordered_candidate_digests": list(value.ordered_candidate_digests),
        "selected_candidate_digest": value.selected_candidate_digest,
        "rubric_spec_digest": value.rubric_spec_digest,
        "semantic_artifact_digest": value.semantic_artifact_digest,
        "version_space_digest": value.version_space_digest,
        "ranker_protocol_id": value.ranker_protocol_id,
        "ranker_protocol_digest": value.ranker_protocol_digest,
        "model_id": value.model_id,
        "model_identity_digest": value.model_identity_digest,
        "environment_digest": value.environment_digest,
        "rank_input_digest": value.rank_input_digest,
        "output_digest": value.output_digest,
        "receipt": dict(value.receipt),
        "receipt_digest": value.receipt_digest,
        "complete_survivor_permutation": True,
        "candidate_formulas_are_immutable": True,
        "image_material_included": False,
        "panel_identities_included": False,
        "support_sides_included": False,
        "held_out_material_included": False,
        "lean_material_included": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricRankResponse:
    """Canonical exact candidate permutation with receipt-attested provenance."""

    ordered_candidate_digests: tuple[str, ...]
    selected_candidate_digest: str
    rubric_spec_digest: str
    semantic_artifact_digest: str
    version_space_digest: str
    ranker_protocol_id: str
    ranker_protocol_digest: str
    model_id: str
    model_identity_digest: str
    environment_digest: str
    rank_input_digest: str
    output_digest: str
    receipt: Mapping[str, Any]
    receipt_digest: str
    response_digest: str

    def __post_init__(self) -> None:
        if (
            not self.ordered_candidate_digests
            or len(self.ordered_candidate_digests) > MAX_SURVIVOR_COUNT
            or len(set(self.ordered_candidate_digests))
            != len(self.ordered_candidate_digests)
        ):
            raise ObjectBongardRubricRankerError(
                "rubric rank response is not a bounded permutation"
            )
        for index, item in enumerate(self.ordered_candidate_digests):
            _digest(item, f"ordered candidate digest {index}")
        if self.selected_candidate_digest != self.ordered_candidate_digests[0]:
            raise ObjectBongardRubricRankerError(
                "selected rubric candidate is not ranked first"
            )
        _identifier(self.ranker_protocol_id, "ranker protocol ID")
        _identifier(self.model_id, "model ID")
        for name in (
            "rubric_spec_digest",
            "semantic_artifact_digest",
            "version_space_digest",
            "ranker_protocol_digest",
            "model_identity_digest",
            "environment_digest",
            "rank_input_digest",
            "output_digest",
            "receipt_digest",
            "response_digest",
        ):
            _digest(getattr(self, name), name)
        if not isinstance(self.receipt, Mapping) or any(
            not isinstance(key, str) for key in self.receipt
        ):
            raise ObjectBongardRubricRankerError(
                "rubric rank receipt must be an object"
            )
        try:
            canonical_receipt = json.loads(
                canonical_json(dict(self.receipt)).decode("utf-8")
            )
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise ObjectBongardRubricRankerError(
                "rubric rank receipt is not canonical JSON"
            ) from exc
        object.__setattr__(self, "receipt", canonical_receipt)
        if set(canonical_receipt) != {"ranker_binding", "transport_receipt"}:
            raise ObjectBongardRubricRankerError(
                "rubric rank receipt envelope fields differ"
            )
        expected_binding = {
            "ranker_protocol_id": self.ranker_protocol_id,
            "ranker_protocol_digest": self.ranker_protocol_digest,
            "model_id": self.model_id,
            "model_identity_digest": self.model_identity_digest,
            "environment_digest": self.environment_digest,
            "rubric_spec_digest": self.rubric_spec_digest,
            "semantic_artifact_digest": self.semantic_artifact_digest,
            "version_space_digest": self.version_space_digest,
            "rank_input_digest": self.rank_input_digest,
            "output_digest": self.output_digest,
        }
        if (
            canonical_receipt["ranker_binding"] != expected_binding
            or not isinstance(canonical_receipt["transport_receipt"], Mapping)
            or not canonical_receipt["transport_receipt"]
            or self.output_digest
            != _rank_output_digest(self.ordered_candidate_digests)
            or self.receipt_digest
            != canonical_digest(
                {
                    "schema": OBJECT_BONGARD_RUBRIC_RANK_RECEIPT_SCHEMA,
                    "receipt": canonical_receipt,
                }
            )
            or self.response_digest != canonical_digest(_response_content(self))
        ):
            raise ObjectBongardRubricRankerError(
                "rubric rank response provenance differs"
            )

    @classmethod
    def seal(
        cls,
        *,
        ordered_candidate_digests: Sequence[str],
        rubric_spec_digest: str,
        semantic_artifact_digest: str,
        version_space_digest: str,
        ranker_protocol_id: str,
        ranker_protocol_digest: str,
        model_id: str,
        model_identity_digest: str,
        environment_digest: str,
        rank_input_digest: str,
        transport_receipt: Mapping[str, Any],
    ) -> "ObjectBongardRubricRankResponse":
        ordered = tuple(ordered_candidate_digests)
        if not ordered:
            raise ObjectBongardRubricRankerError(
                "rubric rank response cannot be empty"
            )
        output_digest = _rank_output_digest(ordered)
        receipt = {
            "ranker_binding": {
                "ranker_protocol_id": ranker_protocol_id,
                "ranker_protocol_digest": ranker_protocol_digest,
                "model_id": model_id,
                "model_identity_digest": model_identity_digest,
                "environment_digest": environment_digest,
                "rubric_spec_digest": rubric_spec_digest,
                "semantic_artifact_digest": semantic_artifact_digest,
                "version_space_digest": version_space_digest,
                "rank_input_digest": rank_input_digest,
                "output_digest": output_digest,
            },
            "transport_receipt": dict(transport_receipt),
        }
        receipt_digest = canonical_digest(
            {
                "schema": OBJECT_BONGARD_RUBRIC_RANK_RECEIPT_SCHEMA,
                "receipt": receipt,
            }
        )
        values: dict[str, object] = {
            "ordered_candidate_digests": ordered,
            "selected_candidate_digest": ordered[0],
            "rubric_spec_digest": rubric_spec_digest,
            "semantic_artifact_digest": semantic_artifact_digest,
            "version_space_digest": version_space_digest,
            "ranker_protocol_id": ranker_protocol_id,
            "ranker_protocol_digest": ranker_protocol_digest,
            "model_id": model_id,
            "model_identity_digest": model_identity_digest,
            "environment_digest": environment_digest,
            "rank_input_digest": rank_input_digest,
            "output_digest": output_digest,
            "receipt": receipt,
            "receipt_digest": receipt_digest,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            response_digest=canonical_digest(_response_content(provisional)),
        )

    def assert_matches(
        self,
        *,
        survivor_candidate_digests: Sequence[str],
        rubric_spec_digest: str,
        semantic_artifact_digest: str,
        version_space_digest: str,
        rank_input_digest: str,
    ) -> None:
        survivors = tuple(survivor_candidate_digests)
        if (
            self.rubric_spec_digest != rubric_spec_digest
            or self.semantic_artifact_digest != semantic_artifact_digest
            or self.version_space_digest != version_space_digest
            or self.rank_input_digest != rank_input_digest
            or len(self.ordered_candidate_digests) != len(survivors)
            or set(self.ordered_candidate_digests) != set(survivors)
        ):
            raise ObjectBongardRubricRankerError(
                "rubric rank response must be the exact verified survivor permutation"
            )

    def to_data(self) -> dict[str, object]:
        return {**_response_content(self), "response_digest": self.response_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardRubricRankResponse":
        raw = _fields(
            value,
            {
                "schema",
                "ordered_candidate_digests",
                "selected_candidate_digest",
                "rubric_spec_digest",
                "semantic_artifact_digest",
                "version_space_digest",
                "ranker_protocol_id",
                "ranker_protocol_digest",
                "model_id",
                "model_identity_digest",
                "environment_digest",
                "rank_input_digest",
                "output_digest",
                "receipt",
                "receipt_digest",
                "complete_survivor_permutation",
                "candidate_formulas_are_immutable",
                "image_material_included",
                "panel_identities_included",
                "support_sides_included",
                "held_out_material_included",
                "lean_material_included",
                *_authority_data(),
                "response_digest",
            },
            "rubric rank response",
        )
        if (
            raw["schema"] != OBJECT_BONGARD_RUBRIC_RANK_RESPONSE_SCHEMA
            or raw["complete_survivor_permutation"] is not True
            or raw["candidate_formulas_are_immutable"] is not True
            or raw["image_material_included"] is not False
            or raw["panel_identities_included"] is not False
            or raw["support_sides_included"] is not False
            or raw["held_out_material_included"] is not False
            or raw["lean_material_included"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["ordered_candidate_digests"], list)
            or not isinstance(raw["receipt"], Mapping)
        ):
            raise ObjectBongardRubricRankerError(
                "rubric rank response policy differs"
            )
        result = cls(
            ordered_candidate_digests=tuple(raw["ordered_candidate_digests"]),
            selected_candidate_digest=raw["selected_candidate_digest"],
            rubric_spec_digest=raw["rubric_spec_digest"],
            semantic_artifact_digest=raw["semantic_artifact_digest"],
            version_space_digest=raw["version_space_digest"],
            ranker_protocol_id=raw["ranker_protocol_id"],
            ranker_protocol_digest=raw["ranker_protocol_digest"],
            model_id=raw["model_id"],
            model_identity_digest=raw["model_identity_digest"],
            environment_digest=raw["environment_digest"],
            rank_input_digest=raw["rank_input_digest"],
            output_digest=raw["output_digest"],
            receipt=dict(raw["receipt"]),
            receipt_digest=raw["receipt_digest"],
            response_digest=raw["response_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricRankerError(
                "rubric rank response is not canonical"
            )
        return result


def _validate_transport_receipt(
    *,
    receipt: Mapping[str, Any],
    prompt: str,
    schema: Mapping[str, Any],
    payload: Mapping[str, Any],
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str,
    expected_cloud_policy_cache_binding: str,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
) -> None:
    try:
        validate_codex_text_receipt(receipt, prompt, schema)
    except (CodexProposerFailure, TypeError, ValueError) as exc:
        raise ObjectBongardRubricRankerError(
            "text rubric-rank receipt does not bind the frozen input"
        ) from exc
    if (
        receipt["requested_model"] != model
        or receipt["requested_reasoning_effort"] != reasoning_effort
        or receipt["codex_launcher_digest"] != expected_launcher_digest
        or receipt["cloud_config_bundle_cache_binding"]
        != expected_cloud_policy_cache_binding
        or receipt["model_catalog_digest"] != model_catalog_snapshot.raw_digest
        or receipt["tool_surface_attestation_digest"]
        != no_tools_attestation.attestation_digest
        or receipt["structured_output_digest"] != canonical_digest(dict(payload))
    ):
        raise ObjectBongardRubricRankerError(
            "text rubric-rank receipt model, environment, or payload differs"
        )


def verify_object_bongard_rubric_rank_response(
    response: ObjectBongardRubricRankResponse,
    *,
    version_space: ObjectBongardRubricSupportVersionSpace,
    rubric_spec: ObjectBongardRubricSpec,
    semantic_artifact: ObjectBongardSemanticArtifact,
    positive_support_artifacts: Sequence[ObjectBongardRubricObserverArtifact],
    negative_support_artifacts: Sequence[ObjectBongardRubricObserverArtifact],
    rank_input_digest: str,
    expected_response_digest: str,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str,
    expected_cloud_policy_cache_binding: str,
    expected_transport_source_digest: str,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
) -> ObjectBongardRubricRankResponse:
    """Cold-verify one persisted response without invoking the transport."""

    if not isinstance(response, ObjectBongardRubricRankResponse):
        raise TypeError("response must be ObjectBongardRubricRankResponse")
    version, spec, semantic, survivors, positives, negatives = _rank_inputs(
        version_space=version_space,
        rubric_spec=rubric_spec,
        semantic_artifact=semantic_artifact,
        positive_support_artifacts=positive_support_artifacts,
        negative_support_artifacts=negative_support_artifacts,
        rank_input_digest=rank_input_digest,
    )
    if response.response_digest != _digest(
        expected_response_digest, "expected response digest"
    ):
        raise ObjectBongardRubricRankerError(
            "rubric rank response differs from external commitment"
        )
    response.assert_matches(
        survivor_candidate_digests=tuple(
            item.candidate_digest for item in survivors
        ),
        rubric_spec_digest=spec.spec_digest,
        semantic_artifact_digest=semantic.artifact_digest,
        version_space_digest=version.version_space_digest,
        rank_input_digest=rank_input_digest,
    )
    expected_model_identity = object_bongard_rubric_ranker_model_identity_digest(
        model, reasoning_effort
    )
    expected_environment = object_bongard_rubric_ranker_environment_digest(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        expected_cloud_policy_cache_binding=expected_cloud_policy_cache_binding,
        expected_transport_source_digest=expected_transport_source_digest,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    if (
        response.ranker_protocol_id != OBJECT_BONGARD_RUBRIC_RANKER_PROTOCOL_ID
        or response.ranker_protocol_digest
        != object_bongard_rubric_ranker_protocol_digest()
        or response.model_id != model
        or response.model_identity_digest != expected_model_identity
        or response.environment_digest != expected_environment
    ):
        raise ObjectBongardRubricRankerError(
            "rubric rank response protocol, model, or environment differs"
        )
    prompt = _ranker_prompt_from_frozen(
        version,
        spec,
        semantic,
        survivors,
        positives,
        negatives,
        rank_input_digest,
    )
    schema = object_bongard_rubric_ranker_output_schema()
    payload = _ordered_alias_payload(response.ordered_candidate_digests, survivors)
    transport_receipt = response.receipt.get("transport_receipt")
    if not isinstance(transport_receipt, Mapping):
        raise ObjectBongardRubricRankerError(
            "rubric rank transport receipt is invalid"
        )
    _validate_transport_receipt(
        receipt=transport_receipt,
        prompt=prompt,
        schema=schema,
        payload=payload,
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        expected_cloud_policy_cache_binding=expected_cloud_policy_cache_binding,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    if ObjectBongardRubricRankResponse.from_data(response.to_data()) != response:
        raise ObjectBongardRubricRankerError(
            "rubric rank response cold round trip differs"
        )
    return response


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricRanker:
    """Configured receipt-attested text-only rubric-candidate ranker."""

    model: str
    expected_launcher_digest: str
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot
    expected_cloud_policy_cache_binding: str
    expected_transport_source_digest: str
    model_catalog_snapshot: CodexModelCatalogSnapshot
    no_tools_attestation: CodexNoToolsAttestation
    reasoning_effort: str = "medium"
    minutes: int = 15
    verbose: bool = False
    executable: str = "codex"
    transport: TextStructuredTransport = field(
        default=run_codex_text_structured, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object_bongard_rubric_ranker_model_identity_digest(
            self.model, self.reasoning_effort
        )
        _digest(self.expected_launcher_digest, "expected launcher digest")
        if not isinstance(self.cloud_policy_cache_snapshot, CloudPolicyCacheSnapshot):
            raise ObjectBongardRubricRankerError(
                "an exact cloud policy-cache snapshot is required"
            )
        if (
            self.expected_cloud_policy_cache_binding
            != self.cloud_policy_cache_snapshot.binding
        ):
            raise ObjectBongardRubricRankerError(
                "policy-cache snapshot differs from external commitment"
            )
        object_bongard_rubric_ranker_environment_digest(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            expected_launcher_digest=self.expected_launcher_digest,
            expected_cloud_policy_cache_binding=(
                self.expected_cloud_policy_cache_binding
            ),
            expected_transport_source_digest=self.expected_transport_source_digest,
            model_catalog_snapshot=self.model_catalog_snapshot,
            no_tools_attestation=self.no_tools_attestation,
        )
        if (
            isinstance(self.minutes, bool)
            or not isinstance(self.minutes, int)
            or not 1 <= self.minutes <= 120
        ):
            raise ObjectBongardRubricRankerError(
                "rubric ranker timeout minutes must lie in 1..120"
            )
        if not isinstance(self.verbose, bool):
            raise TypeError("verbose must be bool")
        if not isinstance(self.executable, str) or not self.executable:
            raise ObjectBongardRubricRankerError(
                "rubric ranker executable must be nonempty"
            )
        if not callable(self.transport):
            raise TypeError("rubric ranker transport must be callable")

    @property
    def protocol_digest(self) -> str:
        return object_bongard_rubric_ranker_protocol_digest()

    @property
    def model_identity_digest(self) -> str:
        return object_bongard_rubric_ranker_model_identity_digest(
            self.model, self.reasoning_effort
        )

    @property
    def environment_digest(self) -> str:
        return object_bongard_rubric_ranker_environment_digest(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            expected_launcher_digest=self.expected_launcher_digest,
            expected_cloud_policy_cache_binding=(
                self.expected_cloud_policy_cache_binding
            ),
            expected_transport_source_digest=self.expected_transport_source_digest,
            model_catalog_snapshot=self.model_catalog_snapshot,
            no_tools_attestation=self.no_tools_attestation,
        )

    def __call__(
        self,
        version_space: ObjectBongardRubricSupportVersionSpace,
        *,
        rubric_spec: ObjectBongardRubricSpec,
        semantic_artifact: ObjectBongardSemanticArtifact,
        positive_support_artifacts: Sequence[ObjectBongardRubricObserverArtifact],
        negative_support_artifacts: Sequence[ObjectBongardRubricObserverArtifact],
        rank_input_digest: str,
    ) -> ObjectBongardRubricRankResponse:
        version, spec, semantic, survivors, positives, negatives = _rank_inputs(
            version_space=version_space,
            rubric_spec=rubric_spec,
            semantic_artifact=semantic_artifact,
            positive_support_artifacts=positive_support_artifacts,
            negative_support_artifacts=negative_support_artifacts,
            rank_input_digest=rank_input_digest,
        )
        prompt = _ranker_prompt_from_frozen(
            version,
            spec,
            semantic,
            survivors,
            positives,
            negatives,
            rank_input_digest,
        )
        schema = object_bongard_rubric_ranker_output_schema()
        try:
            result = self.transport(
                prompt,
                schema,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                minutes=self.minutes,
                verbose=self.verbose,
                executable=self.executable,
                cloud_policy_cache_snapshot=self.cloud_policy_cache_snapshot,
                model_catalog_snapshot=self.model_catalog_snapshot,
                expected_launcher_digest=self.expected_launcher_digest,
                tool_surface_attestation=self.no_tools_attestation,
                expected_tool_surface_attestation_digest=(
                    self.no_tools_attestation.attestation_digest
                ),
            )
        except Exception as exc:
            raise ObjectBongardRubricRankerError(
                "text-only rubric rank transport failed"
            ) from exc
        if not isinstance(result, CodexStructuredResult):
            raise ObjectBongardRubricRankerError(
                "text rubric transport returned the wrong result type"
            )
        payload = _canonical_payload(result.payload)
        ordered_digests = _parse_alias_payload(payload, survivors)
        if not isinstance(result.receipt, CodexReceipt):
            raise ObjectBongardRubricRankerError(
                "text rubric transport returned no CodexReceipt"
            )
        receipt = result.receipt.to_dict()
        _validate_transport_receipt(
            receipt=receipt,
            prompt=prompt,
            schema=schema,
            payload=payload,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            expected_launcher_digest=self.expected_launcher_digest,
            expected_cloud_policy_cache_binding=(
                self.expected_cloud_policy_cache_binding
            ),
            model_catalog_snapshot=self.model_catalog_snapshot,
            no_tools_attestation=self.no_tools_attestation,
        )
        response = ObjectBongardRubricRankResponse.seal(
            ordered_candidate_digests=ordered_digests,
            rubric_spec_digest=spec.spec_digest,
            semantic_artifact_digest=semantic.artifact_digest,
            version_space_digest=version.version_space_digest,
            ranker_protocol_id=OBJECT_BONGARD_RUBRIC_RANKER_PROTOCOL_ID,
            ranker_protocol_digest=self.protocol_digest,
            model_id=self.model,
            model_identity_digest=self.model_identity_digest,
            environment_digest=self.environment_digest,
            rank_input_digest=rank_input_digest,
            transport_receipt=receipt,
        )
        response.assert_matches(
            survivor_candidate_digests=tuple(
                item.candidate_digest for item in survivors
            ),
            rubric_spec_digest=spec.spec_digest,
            semantic_artifact_digest=semantic.artifact_digest,
            version_space_digest=version.version_space_digest,
            rank_input_digest=rank_input_digest,
        )
        return response

    def verify_response(
        self,
        response: ObjectBongardRubricRankResponse,
        *,
        version_space: ObjectBongardRubricSupportVersionSpace,
        rubric_spec: ObjectBongardRubricSpec,
        semantic_artifact: ObjectBongardSemanticArtifact,
        positive_support_artifacts: Sequence[ObjectBongardRubricObserverArtifact],
        negative_support_artifacts: Sequence[ObjectBongardRubricObserverArtifact],
        rank_input_digest: str,
        expected_response_digest: str,
    ) -> ObjectBongardRubricRankResponse:
        return verify_object_bongard_rubric_rank_response(
            response,
            version_space=version_space,
            rubric_spec=rubric_spec,
            semantic_artifact=semantic_artifact,
            positive_support_artifacts=positive_support_artifacts,
            negative_support_artifacts=negative_support_artifacts,
            rank_input_digest=rank_input_digest,
            expected_response_digest=expected_response_digest,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            expected_launcher_digest=self.expected_launcher_digest,
            expected_cloud_policy_cache_binding=(
                self.expected_cloud_policy_cache_binding
            ),
            expected_transport_source_digest=self.expected_transport_source_digest,
            model_catalog_snapshot=self.model_catalog_snapshot,
            no_tools_attestation=self.no_tools_attestation,
        )


__all__ = (
    "MAX_PROMPT_UTF8_BYTES",
    "MAX_RUBRIC_UTF8_BYTES",
    "MAX_SURVIVOR_COUNT",
    "OBJECT_BONGARD_RUBRIC_RANKER_PROTOCOL_ID",
    "OBJECT_BONGARD_RUBRIC_RANKER_PROTOCOL_SCHEMA",
    "OBJECT_BONGARD_RUBRIC_RANK_INPUT_SCHEMA",
    "OBJECT_BONGARD_RUBRIC_RANK_OUTPUT_SCHEMA",
    "OBJECT_BONGARD_RUBRIC_RANK_RESPONSE_SCHEMA",
    "ObjectBongardRubricRankResponse",
    "ObjectBongardRubricRanker",
    "ObjectBongardRubricRankerError",
    "TextStructuredTransport",
    "object_bongard_rubric_rank_input_digest",
    "object_bongard_rubric_ranker_authority_data",
    "object_bongard_rubric_ranker_environment_digest",
    "object_bongard_rubric_ranker_model_identity_digest",
    "object_bongard_rubric_ranker_output_schema",
    "object_bongard_rubric_ranker_prompt",
    "object_bongard_rubric_ranker_protocol_digest",
    "object_bongard_rubric_ranker_source_digest",
    "object_bongard_rubric_ranker_transport_source_digest",
    "verify_object_bongard_rubric_rank_response",
)

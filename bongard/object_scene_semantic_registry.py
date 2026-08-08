"""Role-aware semantic concept proposal over frozen scene descriptions.

The proposer in this module is deliberately text-only.  It sees prose already
frozen by the blind visual discovery pass plus the committed support roles.  It
proposes affirmative concepts for both orientations in one response.  Python
then freezes the union as a scoped soft-tag registry; fresh role-blind visual
passes decide every resulting tag with the frontend's four dispositions.

This is an operational observation contract, not a theorem-proving layer.
Python owns preparation, parsing, identity, verification, and replay.  Lean is
absent and removable.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_scene_visual_frontend import (
    OBJECT_SCENE_MAX_REGISTERED_TAGS,
    ObjectSceneSoftTag,
    ObjectSceneSoftTagRegistry,
    ObjectSceneTranscriptArtifact,
    ObjectSceneTranscriptMode,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard import object_scene_visual_frontend as _frontend
from bongard.transport import validate_codex_strict_output_schema


ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE = (
    "role_aware_semantic_concept_proposal"
)
PREPARED_SCHEMA = "gkm.object-scene-semantic-registry-prepared.v1"
CONCEPT_SCHEMA = "gkm.object-scene-semantic-registry-concept.v1"
PROPOSAL_SCHEMA = "gkm.object-scene-semantic-registry-proposal.v1"
MAX_CONCEPTS_PER_ORIENTATION = 16
MIN_CITATIONS_PER_CONCEPT = 2
MAX_CITATIONS_PER_CONCEPT = 16
SUPPORT_PANEL_COUNT = 12
SUPPORT_PANEL_COUNT_PER_ROLE = 6

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ALIAS = re.compile(r"panel_[0-9]{3}\Z")
_SEMANTIC_POLICY_LEAK = re.compile(
    r"\b(?:side|support|bucket|orientation|more|less|fewer|most|least|"
    r"common|frequent|typically|usually|never|isn't|isnt|avoids?|avoiding|"
    r"free|fails?|cannot|can't|cant|plus|also|both|while|along|combined|"
    r"higher|lower|rarer|dominant|prevalent|exclusive|contrastive|"
    r"occurrences?|frequency|often|always|sometimes|only)\b|\bnon[- ]?[a-z]",
    re.IGNORECASE,
)
_GAP_CODES = frozenset(("payload_rejected", "insufficient_discovery_evidence"))


class ObjectSceneSemanticRegistryError(ValueError):
    """A semantic proposal or its provenance is malformed."""


class ObjectSceneSemanticRegistryPayloadError(ObjectSceneSemanticRegistryError):
    """A receipted proposer payload violates the frozen semantic protocol."""


def object_scene_semantic_registry_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "derivation_mode": ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE,
        "zero_image_proposer": True,
        "both_orientations_in_one_call": True,
        "registered_evaluator_receives_roles": False,
        "semantic_proposal_is_not_a_truth_assignment": True,
        "citation_count_is_not_visual_confidence": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_or_decision": False,
    }


def object_scene_semantic_registry_protocol_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.object-scene-semantic-registry-protocol.v1",
            "source_digest": object_scene_semantic_registry_source_digest(),
            "frontend_source_digest": _frontend.object_scene_visual_frontend_source_digest(),
            "prepared_schema": PREPARED_SCHEMA,
            "concept_schema": CONCEPT_SCHEMA,
            "proposal_schema": PROPOSAL_SCHEMA,
            "maximum_concepts_per_orientation": MAX_CONCEPTS_PER_ORIENTATION,
            "maximum_union_concepts": OBJECT_SCENE_MAX_REGISTERED_TAGS,
            "minimum_distinct_same_orientation_citations": MIN_CITATIONS_PER_CONCEPT,
            "support_panel_count": SUPPORT_PANEL_COUNT,
            "support_panel_count_per_role": SUPPORT_PANEL_COUNT_PER_ROLE,
            "alias_order": "sha256-artifact-digest-then-opaque-sequential-alias",
            "registry_order": (
                "descending-distinct-cited-panel-count-then-scope-then-phrase"
            ),
            **_authority_data(),
        }
    )


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectSceneSemanticRegistryError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectSceneSemanticRegistryError(f"{label} must be a raw SHA-256")
    return value


def _canonical_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ObjectSceneSemanticRegistryError(f"{label} must be an object")
    try:
        restored = json.loads(canonical_json(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectSceneSemanticRegistryError(
            f"{label} must be canonical finite JSON"
        ) from exc
    if not isinstance(restored, dict):
        raise ObjectSceneSemanticRegistryError(f"{label} must be an object")
    return restored


def _role_rows_digest(rows: Sequence[Mapping[str, object]]) -> str:
    return canonical_digest(
        {
            "schema": "gkm.object-scene-semantic-registry-role-rows.v1",
            "rows": sorted(
                (dict(item) for item in rows),
                key=lambda item: str(item.get("blind_panel_id", "")),
            ),
            "roles_revealed_after_blind_discovery": True,
        }
    )


def _semantic_cell_view(value: object) -> dict[str, object]:
    raw = dict(getattr(value, "to_data")())
    for key in tuple(raw):
        if key.endswith("_digest"):
            raw.pop(key)
    return raw


def _transcript_view(alias: str, transcript: object | None) -> dict[str, object]:
    if transcript is None:
        return {
            "panel_alias": alias,
            "observation_status": "unavailable",
            "panel_summary": None,
            "panel_open_tags": [],
            "entities": [],
        }
    return {
        "panel_alias": alias,
        "observation_status": "available",
        "panel_summary": getattr(transcript, "panel_summary"),
        "panel_open_tags": [
            _semantic_cell_view(item)
            for item in getattr(transcript, "panel_open_tags")
        ],
        "entities": [
            {
                "entity_alias": f"entity_{index:03d}",
                "summary": row.summary,
                "open_tags": [
                    _semantic_cell_view(item) for item in row.open_tags
                ],
                "qualitative_observations": [
                    _semantic_cell_view(item) for item in row.qualitative_cells
                ],
                "count_observations": [
                    _semantic_cell_view(item) for item in row.count_cells
                ],
            }
            for index, row in enumerate(getattr(transcript, "objects"))
        ],
    }


def _proposal_output_schema(
    side0_aliases: Sequence[str], side1_aliases: Sequence[str]
) -> dict[str, object]:
    def concept(aliases: Sequence[str]) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "scope": {"type": "string", "enum": ["panel", "entity"]},
                "phrase": {"type": "string"},
                "citations": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(aliases)},
                },
            },
            "required": ["scope", "phrase", "citations"],
            "additionalProperties": False,
        }

    result: dict[str, object] = {
        "type": "object",
        "properties": {
            "side0_positive": {
                "type": "array",
                "items": concept(side0_aliases),
            },
            "side1_positive": {
                "type": "array",
                "items": concept(side1_aliases),
            },
        },
        "required": ["side0_positive", "side1_positive"],
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(result)
    return result


def _prepared_content(value: "ObjectScenePreparedSemanticRegistryProposal") -> dict[str, object]:
    return {
        "schema": PREPARED_SCHEMA,
        "protocol_digest": value.protocol_digest,
        "source_digest": value.source_digest,
        "source_artifact_digests": list(value.source_artifact_digests),
        "source_transcript_digests": list(value.source_transcript_digests),
        "source_panel_digests": list(value.source_panel_digests),
        "role_rows_digest": value.role_rows_digest,
        "alias_bindings": [dict(item) for item in value.alias_bindings],
        "model_view": dict(value.model_view),
        "model_view_digest": value.model_view_digest,
        "prompt": value.prompt,
        "prompt_digest": value.prompt_digest,
        "output_schema": dict(value.output_schema),
        "output_schema_digest": value.output_schema_digest,
        "pixels_or_images_in_proposer_input": False,
        "task_lineage_ids_in_model_visible_input": False,
        "formula_candidates_in_model_visible_input": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectScenePreparedSemanticRegistryProposal:
    protocol_digest: str
    source_digest: str
    source_artifact_digests: tuple[str, ...]
    source_transcript_digests: tuple[str, ...]
    source_panel_digests: tuple[str, ...]
    role_rows_digest: str
    alias_bindings: tuple[Mapping[str, object], ...]
    model_view: Mapping[str, object]
    model_view_digest: str
    prompt: str
    prompt_digest: str
    output_schema: Mapping[str, object]
    output_schema_digest: str
    preparation_digest: str

    def __post_init__(self) -> None:
        for item, label in (
            (self.protocol_digest, "semantic registry protocol digest"),
            (self.source_digest, "semantic registry source digest"),
            (self.role_rows_digest, "role rows digest"),
            (self.model_view_digest, "model view digest"),
            (self.prompt_digest, "prompt digest"),
            (self.output_schema_digest, "output schema digest"),
            (self.preparation_digest, "preparation digest"),
        ):
            _digest(item, label)
        for item in (
            *self.source_artifact_digests,
            *self.source_transcript_digests,
            *self.source_panel_digests,
        ):
            _digest(item, "prepared source digest")
        if (
            self.protocol_digest != object_scene_semantic_registry_protocol_digest()
            or self.source_digest != object_scene_semantic_registry_source_digest()
            or self.source_artifact_digests
            != tuple(sorted(set(self.source_artifact_digests)))
            or self.source_transcript_digests
            != tuple(sorted(set(self.source_transcript_digests)))
            or self.source_panel_digests
            != tuple(sorted(set(self.source_panel_digests)))
            or len(self.alias_bindings) != len(self.source_artifact_digests)
            or tuple(item.get("alias") for item in self.alias_bindings)
            != tuple(f"panel_{index:03d}" for index in range(len(self.alias_bindings)))
            or any(
                not isinstance(item, Mapping)
                or set(item)
                != {
                    "alias", "scene_id", "artifact_digest", "panel_digest",
                    "transcript_digest", "historical_role", "usable",
                }
                for item in self.alias_bindings
            )
            or canonical_digest(dict(self.model_view)) != self.model_view_digest
            or canonical_digest(self.prompt) != self.prompt_digest
            or canonical_digest(dict(self.output_schema))
            != self.output_schema_digest
            or self.preparation_digest != canonical_digest(_prepared_content(self))
        ):
            raise ObjectSceneSemanticRegistryError(
                "prepared semantic registry proposal differs"
            )
        validate_codex_strict_output_schema(dict(self.output_schema))

    def to_data(self) -> dict[str, object]:
        return {**_prepared_content(self), "preparation_digest": self.preparation_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectScenePreparedSemanticRegistryProposal":
        expected = {
            "schema", "protocol_digest", "source_digest",
            "source_artifact_digests", "source_transcript_digests",
            "source_panel_digests", "role_rows_digest", "alias_bindings",
            "model_view", "model_view_digest", "prompt", "prompt_digest",
            "output_schema", "output_schema_digest",
            "pixels_or_images_in_proposer_input",
            "task_lineage_ids_in_model_visible_input",
            "formula_candidates_in_model_visible_input", *_authority_data(),
            "preparation_digest",
        }
        raw = _fields(value, expected, "prepared semantic registry proposal")
        if (
            raw["schema"] != PREPARED_SCHEMA
            or raw["pixels_or_images_in_proposer_input"] is not False
            or raw["task_lineage_ids_in_model_visible_input"] is not False
            or raw["formula_candidates_in_model_visible_input"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or any(
                not isinstance(raw[key], list)
                for key in (
                    "source_artifact_digests", "source_transcript_digests",
                    "source_panel_digests", "alias_bindings",
                )
            )
        ):
            raise ObjectSceneSemanticRegistryError(
                "prepared semantic registry policy differs"
            )
        result = cls(
            raw["protocol_digest"], raw["source_digest"],
            tuple(raw["source_artifact_digests"]),
            tuple(raw["source_transcript_digests"]),
            tuple(raw["source_panel_digests"]), raw["role_rows_digest"],
            tuple(dict(item) for item in raw["alias_bindings"]),
            _canonical_mapping(raw["model_view"], "prepared model view"),
            raw["model_view_digest"], raw["prompt"], raw["prompt_digest"],
            _canonical_mapping(raw["output_schema"], "prepared output schema"),
            raw["output_schema_digest"], raw["preparation_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneSemanticRegistryError(
                "prepared semantic registry proposal is not canonical"
            )
        return result


def prepare_object_scene_semantic_registry_proposal(
    discovery_artifacts: Sequence[ObjectSceneTranscriptArtifact],
    role_rows: Sequence[Mapping[str, object]],
) -> ObjectScenePreparedSemanticRegistryProposal:
    artifacts = tuple(discovery_artifacts)
    roles = tuple(role_rows)
    if len(artifacts) != SUPPORT_PANEL_COUNT or len(roles) != SUPPORT_PANEL_COUNT:
        raise ObjectSceneSemanticRegistryError(
            "semantic proposal requires exactly twelve support artifacts and roles"
        )
    expected_role_fields = {
        "ordinal", "neutral_panel_digest", "historical_role", "blind_panel_id"
    }
    by_scene: dict[str, Mapping[str, object]] = {}
    for row in roles:
        raw = _fields(row, expected_role_fields, "semantic proposal role row")
        if (
            type(raw["ordinal"]) is not int
            or raw["ordinal"] < 0
            or not isinstance(raw["blind_panel_id"], str)
            or not isinstance(raw["neutral_panel_digest"], str)
            or _DIGEST.fullmatch(raw["neutral_panel_digest"]) is None
            or raw["historical_role"] not in (0, 1)
            or raw["blind_panel_id"] in by_scene
        ):
            raise ObjectSceneSemanticRegistryError(
                "semantic proposal role inventory differs"
            )
        by_scene[raw["blind_panel_id"]] = raw
    if len({row["ordinal"] for row in roles}) != len(roles):
        raise ObjectSceneSemanticRegistryError("semantic proposal role ordinals repeat")
    if tuple(sorted(row["historical_role"] for row in roles)) != (
        (0,) * SUPPORT_PANEL_COUNT_PER_ROLE
        + (1,) * SUPPORT_PANEL_COUNT_PER_ROLE
    ):
        raise ObjectSceneSemanticRegistryError(
            "semantic proposal requires exactly six panels per support role"
        )

    ranked: list[tuple[str, ObjectSceneTranscriptArtifact, Mapping[str, object]]] = []
    for artifact in artifacts:
        if (
            not isinstance(artifact, ObjectSceneTranscriptArtifact)
            or artifact.mode is not ObjectSceneTranscriptMode.DISCOVERY
            or artifact.scene_id not in by_scene
        ):
            raise ObjectSceneSemanticRegistryError(
                "semantic proposal input is not a committed discovery artifact"
            )
        artifact.assert_untampered()
        key = hashlib.sha256(
            ("object-scene-semantic-alias-v1:" + artifact.artifact_digest).encode(
                "ascii"
            )
        ).hexdigest()
        ranked.append((key, artifact, by_scene[artifact.scene_id]))
    if (
        len({item[1].scene_id for item in ranked}) != len(ranked)
        or set(by_scene) != {item[1].scene_id for item in ranked}
    ):
        raise ObjectSceneSemanticRegistryError(
            "semantic proposal discovery identities differ"
        )
    ranked.sort(key=lambda item: (item[0], item[1].artifact_digest))

    bindings: list[dict[str, object]] = []
    model_rows: dict[int, list[dict[str, object]]] = {0: [], 1: []}
    for index, (_, artifact, role) in enumerate(ranked):
        alias = f"panel_{index:03d}"
        transcript = artifact.transcript
        usable = transcript is not None
        bindings.append(
            {
                "alias": alias,
                "scene_id": artifact.scene_id,
                "artifact_digest": artifact.artifact_digest,
                "panel_digest": artifact.panel_digest,
                "transcript_digest": (
                    None if transcript is None else transcript.transcript_digest
                ),
                "historical_role": role["historical_role"],
                "usable": usable,
            }
        )
        model_rows[role["historical_role"]].append(
            _transcript_view(alias, transcript)
        )
    role_aliases = {
        side: tuple(
            item["alias"]
            for item in bindings
            if item["historical_role"] == side
        )
        for side in (0, 1)
    }
    model_view: dict[str, object] = {
        "side0_support_descriptions": model_rows[0],
        "side1_support_descriptions": model_rows[1],
    }
    output_schema = _proposal_output_schema(
        role_aliases[0], role_aliases[1]
    )
    prompt = (
        "From the frozen visual descriptions below, propose candidate visual "
        "concepts for BOTH support orientations in one response. Each concept "
        "must be a single lowercase affirmative visual phrase, scoped either "
        "to the whole panel or to one visible entity. Cite at least two distinct "
        "panel aliases from that concept's own support bucket. A citation says "
        "where the prose suggests the concept; it is not a truth assignment. "
        "Do not compare the buckets, mention roles or labels, use negation, say "
        "that something is missing, or package multiple conditions into one "
        "phrase. Internal visible relations such as mismatched parts or unequal "
        "edge lengths are affirmative and allowed. Supply at most 16 concepts "
        "per bucket. Python will discard bucket membership, freeze one union, "
        "and two fresh role-blind visual passes will judge every phrase. Return "
        "only the required JSON object.\n\nFrozen descriptions:\n"
        + canonical_json(model_view).decode("utf-8")
    )
    hidden = {
        *(artifact.scene_id for artifact in artifacts),
        *(artifact.artifact_digest for artifact in artifacts),
        *(artifact.panel_digest for artifact in artifacts),
        *(artifact.inventory_digest for artifact in artifacts),
    }
    if any(item and item in prompt for item in hidden):
        raise ObjectSceneSemanticRegistryError(
            "semantic proposer prompt leaks hidden panel lineage"
        )
    if len(prompt.encode("utf-8")) > 256_000:
        raise ObjectSceneSemanticRegistryError(
            "semantic proposer prompt exceeds bounded text envelope"
        )
    values = {
        "protocol_digest": object_scene_semantic_registry_protocol_digest(),
        "source_digest": object_scene_semantic_registry_source_digest(),
        "source_artifact_digests": tuple(
            sorted(artifact.artifact_digest for artifact in artifacts)
        ),
        "source_transcript_digests": tuple(
            sorted(
                artifact.transcript.transcript_digest
                for artifact in artifacts
                if artifact.transcript is not None
            )
        ),
        "source_panel_digests": tuple(
            sorted(
                artifact.panel_digest
                for artifact in artifacts
                if artifact.transcript is not None
            )
        ),
        "role_rows_digest": _role_rows_digest(roles),
        "alias_bindings": tuple(bindings),
        "model_view": model_view,
        "model_view_digest": canonical_digest(model_view),
        "prompt": prompt,
        "prompt_digest": canonical_digest(prompt),
        "output_schema": output_schema,
        "output_schema_digest": canonical_digest(output_schema),
    }
    provisional = object.__new__(ObjectScenePreparedSemanticRegistryProposal)
    for key, item in values.items():
        object.__setattr__(provisional, key, item)
    return ObjectScenePreparedSemanticRegistryProposal(
        **values, preparation_digest=canonical_digest(_prepared_content(provisional))
    )


def _concept_content(value: "ObjectSceneSemanticRegistryConcept") -> dict[str, object]:
    return {
        "schema": CONCEPT_SCHEMA,
        "orientation": value.orientation,
        "scope": value.scope,
        "phrase": value.phrase,
        "citations": list(value.citations),
        "citation_count": len(value.citations),
        "affirmative_observation_hypothesis_not_truth": True,
    }


@dataclass(frozen=True, order=True, slots=True)
class ObjectSceneSemanticRegistryConcept:
    orientation: str
    scope: str
    phrase: str
    citations: tuple[str, ...]
    concept_digest: str

    def __post_init__(self) -> None:
        if self.orientation not in ("side0_positive", "side1_positive"):
            raise ObjectSceneSemanticRegistryError("concept orientation differs")
        try:
            scope = _frontend._soft_tag_scope(self.scope)
            phrase = _frontend._normalized_positive_tag(self.phrase)
        except Exception as exc:
            raise ObjectSceneSemanticRegistryError(
                "concept is not scoped atomic affirmative prose"
            ) from exc
        if (
            scope != self.scope
            or phrase != self.phrase
            or _SEMANTIC_POLICY_LEAK.search(self.phrase) is not None
            or not MIN_CITATIONS_PER_CONCEPT
            <= len(self.citations)
            <= MAX_CITATIONS_PER_CONCEPT
            or self.citations != tuple(sorted(set(self.citations)))
            or any(_ALIAS.fullmatch(item) is None for item in self.citations)
        ):
            raise ObjectSceneSemanticRegistryError(
                "semantic concept scope, phrase, or citations differ"
            )
        _digest(self.concept_digest, "semantic concept digest")
        if self.concept_digest != canonical_digest(_concept_content(self)):
            raise ObjectSceneSemanticRegistryError("semantic concept digest differs")

    @classmethod
    def create(
        cls,
        orientation: str,
        scope: object,
        phrase: object,
        citations: object,
    ) -> "ObjectSceneSemanticRegistryConcept":
        if (
            not isinstance(citations, list)
            or any(not isinstance(item, str) for item in citations)
            or len(set(citations)) != len(citations)
        ):
            raise ObjectSceneSemanticRegistryError(
                "semantic concept citations must be distinct"
            )
        try:
            normalized_scope = _frontend._soft_tag_scope(scope)
            normalized_phrase = _frontend._normalized_positive_tag(phrase)
            normalized_citations = tuple(sorted(citations))
        except Exception as exc:
            raise ObjectSceneSemanticRegistryError(
                "semantic concept payload differs"
            ) from exc
        provisional = object.__new__(cls)
        values = {
            "orientation": orientation,
            "scope": normalized_scope,
            "phrase": normalized_phrase,
            "citations": normalized_citations,
        }
        for key, item in values.items():
            object.__setattr__(provisional, key, item)
        return cls(**values, concept_digest=canonical_digest(_concept_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_concept_content(self), "concept_digest": self.concept_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneSemanticRegistryConcept":
        raw = _fields(
            value,
            {
                "schema", "orientation", "scope", "phrase", "citations",
                "citation_count", "affirmative_observation_hypothesis_not_truth",
                "concept_digest",
            },
            "semantic registry concept",
        )
        if (
            raw["schema"] != CONCEPT_SCHEMA
            or raw["affirmative_observation_hypothesis_not_truth"] is not True
            or raw["citation_count"]
            != (len(raw["citations"]) if isinstance(raw["citations"], list) else -1)
            or not isinstance(raw["citations"], list)
        ):
            raise ObjectSceneSemanticRegistryError("semantic concept policy differs")
        result = cls(
            raw["orientation"], raw["scope"], raw["phrase"],
            tuple(raw["citations"]), raw["concept_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneSemanticRegistryError("semantic concept is not canonical")
        return result


def _proposal_content(value: "ObjectSceneSemanticRegistryProposal") -> dict[str, object]:
    return {
        "schema": PROPOSAL_SCHEMA,
        "status": value.status,
        "derivation_mode": ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE,
        "protocol_digest": value.protocol_digest,
        "source_digest": value.source_digest,
        "preparation_digest": value.preparation_digest,
        "role_rows_digest": value.role_rows_digest,
        "source_artifact_digests": list(value.source_artifact_digests),
        "source_transcript_digests": list(value.source_transcript_digests),
        "source_panel_digests": list(value.source_panel_digests),
        "model_payload": value.model_payload,
        "model_payload_digest": value.model_payload_digest,
        "side0_positive": [item.to_data() for item in value.side0_positive],
        "side1_positive": [item.to_data() for item in value.side1_positive],
        "gap_code": value.gap_code,
        "registry_digest": value.registry_digest,
        "union_discards_orientation_membership_before_visual_evaluation": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneSemanticRegistryProposal:
    status: str
    protocol_digest: str
    source_digest: str
    preparation_digest: str
    role_rows_digest: str
    source_artifact_digests: tuple[str, ...]
    source_transcript_digests: tuple[str, ...]
    source_panel_digests: tuple[str, ...]
    model_payload: Mapping[str, Any] | None
    model_payload_digest: str | None
    side0_positive: tuple[ObjectSceneSemanticRegistryConcept, ...]
    side1_positive: tuple[ObjectSceneSemanticRegistryConcept, ...]
    gap_code: str | None
    registry_digest: str
    proposal_digest: str

    def __post_init__(self) -> None:
        for item, label in (
            (self.protocol_digest, "proposal protocol digest"),
            (self.source_digest, "proposal source digest"),
            (self.preparation_digest, "proposal preparation digest"),
            (self.role_rows_digest, "proposal role rows digest"),
            (self.registry_digest, "proposal registry digest"),
            (self.proposal_digest, "semantic proposal digest"),
        ):
            _digest(item, label)
        if (
            self.status not in ("proposed", "typed_proposal_gap")
            or self.protocol_digest != object_scene_semantic_registry_protocol_digest()
            or self.source_digest != object_scene_semantic_registry_source_digest()
            or self.source_artifact_digests
            != tuple(sorted(set(self.source_artifact_digests)))
            or self.source_transcript_digests
            != tuple(sorted(set(self.source_transcript_digests)))
            or self.source_panel_digests
            != tuple(sorted(set(self.source_panel_digests)))
        ):
            raise ObjectSceneSemanticRegistryError("semantic proposal provenance differs")
        concepts = (*self.side0_positive, *self.side1_positive)
        if (
            len(self.side0_positive) > MAX_CONCEPTS_PER_ORIENTATION
            or len(self.side1_positive) > MAX_CONCEPTS_PER_ORIENTATION
            or len(concepts) > OBJECT_SCENE_MAX_REGISTERED_TAGS
            or len({(item.scope, item.phrase) for item in concepts}) != len(concepts)
            or tuple(item.orientation for item in self.side0_positive)
            != ("side0_positive",) * len(self.side0_positive)
            or tuple(item.orientation for item in self.side1_positive)
            != ("side1_positive",) * len(self.side1_positive)
        ):
            raise ObjectSceneSemanticRegistryError("semantic proposal concept inventory differs")
        if self.status == "proposed":
            if (
                not self.side0_positive
                or not self.side1_positive
                or self.gap_code is not None
                or self.model_payload is None
                or self.model_payload_digest
                != canonical_digest(dict(self.model_payload))
            ):
                raise ObjectSceneSemanticRegistryError("successful semantic proposal differs")
        else:
            if concepts or self.gap_code not in _GAP_CODES:
                raise ObjectSceneSemanticRegistryError(
                    "typed semantic proposal gap differs"
                )
            if self.gap_code == "payload_rejected":
                if (
                    self.model_payload is None
                    or self.model_payload_digest
                    != canonical_digest(dict(self.model_payload))
                ):
                    raise ObjectSceneSemanticRegistryError(
                        "rejected semantic payload is not bound into its gap"
                    )
            elif self.model_payload is None:
                if self.model_payload_digest is not None:
                    raise ObjectSceneSemanticRegistryError(
                        "evidence gap payload digest differs"
                    )
            elif self.model_payload_digest != canonical_digest(dict(self.model_payload)):
                raise ObjectSceneSemanticRegistryError(
                    "evidence gap model payload binding differs"
                )
        if self.proposal_digest != canonical_digest(_proposal_content(self)):
            raise ObjectSceneSemanticRegistryError("semantic proposal digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_proposal_content(self), "proposal_digest": self.proposal_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneSemanticRegistryProposal":
        expected = {
            "schema", "status", "derivation_mode", "protocol_digest",
            "source_digest", "preparation_digest", "role_rows_digest",
            "source_artifact_digests", "source_transcript_digests",
            "source_panel_digests", "model_payload", "model_payload_digest",
            "side0_positive", "side1_positive", "gap_code", "registry_digest",
            "union_discards_orientation_membership_before_visual_evaluation",
            *_authority_data(), "proposal_digest",
        }
        raw = _fields(value, expected, "semantic registry proposal")
        if (
            raw["schema"] != PROPOSAL_SCHEMA
            or raw["derivation_mode"]
            != ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
            or raw["union_discards_orientation_membership_before_visual_evaluation"]
            is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or any(
                not isinstance(raw[key], list)
                for key in (
                    "source_artifact_digests", "source_transcript_digests",
                    "source_panel_digests", "side0_positive", "side1_positive",
                )
            )
        ):
            raise ObjectSceneSemanticRegistryError("semantic proposal policy differs")
        result = cls(
            raw["status"], raw["protocol_digest"], raw["source_digest"],
            raw["preparation_digest"], raw["role_rows_digest"],
            tuple(raw["source_artifact_digests"]),
            tuple(raw["source_transcript_digests"]),
            tuple(raw["source_panel_digests"]),
            None if raw["model_payload"] is None else _canonical_mapping(
                raw["model_payload"], "semantic proposal model payload"
            ),
            raw["model_payload_digest"],
            tuple(
                ObjectSceneSemanticRegistryConcept.from_data(item)
                for item in raw["side0_positive"]
            ),
            tuple(
                ObjectSceneSemanticRegistryConcept.from_data(item)
                for item in raw["side1_positive"]
            ),
            raw["gap_code"], raw["registry_digest"], raw["proposal_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneSemanticRegistryError("semantic proposal is not canonical")
        return result


def _semantic_registry(
    prepared: ObjectScenePreparedSemanticRegistryProposal,
    concepts: Sequence[ObjectSceneSemanticRegistryConcept],
) -> ObjectSceneSoftTagRegistry:
    ranked = sorted(
        concepts,
        key=lambda item: (-len(item.citations), item.scope, item.phrase),
    )
    tags = tuple(
        ObjectSceneSoftTag(
            f"tag_{index:04d}", item.scope, item.phrase,
            len(item.citations),
            _frontend._soft_tag_content_digest(item.scope, item.phrase),
        )
        for index, item in enumerate(ranked)
    )
    values = {
        "source_transcript_digests": prepared.source_transcript_digests,
        "source_panel_digests": prepared.source_panel_digests,
        "tags": tags,
        "dropped_tags": (),
    }
    provisional = object.__new__(ObjectSceneSoftTagRegistry)
    for key, item in values.items():
        object.__setattr__(provisional, key, item)
    return ObjectSceneSoftTagRegistry(
        **values,
        registry_digest=canonical_digest(_frontend._registry_content(provisional)),
    )


def _proposal(
    prepared: ObjectScenePreparedSemanticRegistryProposal,
    *,
    status: str,
    model_payload: Mapping[str, Any] | None,
    side0: Sequence[ObjectSceneSemanticRegistryConcept],
    side1: Sequence[ObjectSceneSemanticRegistryConcept],
    gap_code: str | None,
    registry: ObjectSceneSoftTagRegistry,
) -> ObjectSceneSemanticRegistryProposal:
    payload = None if model_payload is None else _canonical_mapping(
        model_payload, "semantic registry model payload"
    )
    values = {
        "status": status,
        "protocol_digest": prepared.protocol_digest,
        "source_digest": prepared.source_digest,
        "preparation_digest": prepared.preparation_digest,
        "role_rows_digest": prepared.role_rows_digest,
        "source_artifact_digests": prepared.source_artifact_digests,
        "source_transcript_digests": prepared.source_transcript_digests,
        "source_panel_digests": prepared.source_panel_digests,
        "model_payload": payload,
        "model_payload_digest": None if payload is None else canonical_digest(payload),
        "side0_positive": tuple(side0),
        "side1_positive": tuple(side1),
        "gap_code": gap_code,
        "registry_digest": registry.registry_digest,
    }
    provisional = object.__new__(ObjectSceneSemanticRegistryProposal)
    for key, item in values.items():
        object.__setattr__(provisional, key, item)
    return ObjectSceneSemanticRegistryProposal(
        **values, proposal_digest=canonical_digest(_proposal_content(provisional))
    )


def build_object_scene_semantic_registry_proposal(
    prepared: ObjectScenePreparedSemanticRegistryProposal,
    payload: Mapping[str, Any],
) -> tuple[ObjectSceneSemanticRegistryProposal, ObjectSceneSoftTagRegistry]:
    if not isinstance(prepared, ObjectScenePreparedSemanticRegistryProposal):
        raise TypeError("prepared must be ObjectScenePreparedSemanticRegistryProposal")
    ObjectScenePreparedSemanticRegistryProposal.from_data(prepared.to_data())
    allowed = {
        side: {
            item["alias"]
            for item in prepared.alias_bindings
            if item["historical_role"] == side and item["usable"] is True
        }
        for side in (0, 1)
    }
    try:
        raw = _fields(
            _canonical_mapping(payload, "semantic registry proposal payload"),
            {"side0_positive", "side1_positive"},
            "semantic registry proposal payload",
        )
        if any(not isinstance(raw[key], list) for key in raw):
            raise ObjectSceneSemanticRegistryError(
                "semantic proposal buckets must be arrays"
            )
        if (
            not 1 <= len(raw["side0_positive"]) <= MAX_CONCEPTS_PER_ORIENTATION
            or not 1 <= len(raw["side1_positive"]) <= MAX_CONCEPTS_PER_ORIENTATION
            or len(raw["side0_positive"]) + len(raw["side1_positive"])
            > OBJECT_SCENE_MAX_REGISTERED_TAGS
        ):
            raise ObjectSceneSemanticRegistryError(
                "semantic proposal bucket capacity differs"
            )
        buckets: dict[int, list[ObjectSceneSemanticRegistryConcept]] = {
            0: [], 1: []
        }
        for side, key in ((0, "side0_positive"), (1, "side1_positive")):
            for item in raw[key]:
                concept_raw = _fields(
                    item,
                    {"scope", "phrase", "citations"},
                    "semantic concept payload",
                )
                concept = ObjectSceneSemanticRegistryConcept.create(
                    key,
                    concept_raw["scope"],
                    concept_raw["phrase"],
                    concept_raw["citations"],
                )
                if not set(concept.citations).issubset(allowed[side]):
                    raise ObjectSceneSemanticRegistryError(
                        "semantic concept cites a foreign or cross-side panel"
                    )
                buckets[side].append(concept)
            buckets[side].sort(
                key=lambda item: (item.scope, item.phrase, item.citations)
            )
        concepts = (*buckets[0], *buckets[1])
        if len({(item.scope, item.phrase) for item in concepts}) != len(concepts):
            raise ObjectSceneSemanticRegistryError(
                "semantic proposal repeats a scoped phrase"
            )
    except ObjectSceneSemanticRegistryPayloadError:
        raise
    except ObjectSceneSemanticRegistryError as exc:
        raise ObjectSceneSemanticRegistryPayloadError(str(exc)) from exc
    registry = _semantic_registry(prepared, concepts)
    proposal = _proposal(
        prepared,
        status="proposed",
        model_payload=raw,
        side0=buckets[0],
        side1=buckets[1],
        gap_code=None,
        registry=registry,
    )
    return proposal, registry


def parse_object_scene_semantic_registry_proposal(
    prepared: ObjectScenePreparedSemanticRegistryProposal,
    payload: Mapping[str, Any],
) -> tuple[ObjectSceneSemanticRegistryProposal, ObjectSceneSoftTagRegistry]:
    """Compatibility name: parsing always returns the sealed proposal and union."""

    return build_object_scene_semantic_registry_proposal(prepared, payload)


def build_object_scene_semantic_registry_gap(
    prepared: ObjectScenePreparedSemanticRegistryProposal,
    gap_code: str,
    rejected_payload: Mapping[str, Any] | None = None,
) -> tuple[ObjectSceneSemanticRegistryProposal, ObjectSceneSoftTagRegistry]:
    if not isinstance(prepared, ObjectScenePreparedSemanticRegistryProposal):
        raise TypeError("prepared must be ObjectScenePreparedSemanticRegistryProposal")
    if gap_code not in _GAP_CODES:
        raise ObjectSceneSemanticRegistryError("semantic proposal gap code differs")
    if gap_code == "payload_rejected" and rejected_payload is None:
        raise ObjectSceneSemanticRegistryError(
            "payload-rejected gap must bind its rejected payload"
        )
    registry = _semantic_registry(prepared, ())
    return (
        _proposal(
            prepared,
            status="typed_proposal_gap",
            model_payload=rejected_payload,
            side0=(),
            side1=(),
            gap_code=gap_code,
            registry=registry,
        ),
        registry,
    )


def verify_object_scene_semantic_registry_proposal(
    proposal: ObjectSceneSemanticRegistryProposal | Mapping[str, object],
    registry: ObjectSceneSoftTagRegistry,
    discovery_artifacts: Sequence[ObjectSceneTranscriptArtifact],
    role_rows: Sequence[Mapping[str, object]],
) -> ObjectSceneSemanticRegistryProposal:
    restored = ObjectSceneSemanticRegistryProposal.from_data(
        proposal.to_data()
        if isinstance(proposal, ObjectSceneSemanticRegistryProposal)
        else proposal
    )
    if not isinstance(registry, ObjectSceneSoftTagRegistry):
        raise TypeError("registry must be ObjectSceneSoftTagRegistry")
    restored_registry = ObjectSceneSoftTagRegistry.from_data(registry.to_data())
    prepared = prepare_object_scene_semantic_registry_proposal(
        discovery_artifacts, role_rows
    )
    if restored.preparation_digest != prepared.preparation_digest:
        raise ObjectSceneSemanticRegistryError(
            "semantic proposal belongs to another prepared input"
        )
    if restored.status == "proposed":
        assert restored.model_payload is not None
        expected, expected_registry = build_object_scene_semantic_registry_proposal(
            prepared, restored.model_payload
        )
    else:
        assert restored.gap_code is not None
        expected, expected_registry = build_object_scene_semantic_registry_gap(
            prepared, restored.gap_code, restored.model_payload
        )
    if (
        expected != restored
        or expected_registry != restored_registry
        or restored.registry_digest != restored_registry.registry_digest
    ):
        raise ObjectSceneSemanticRegistryError(
            "semantic proposal or union registry differs on replay"
        )
    return restored


__all__ = (
    "MAX_CONCEPTS_PER_ORIENTATION",
    "ObjectScenePreparedSemanticRegistryProposal",
    "ObjectSceneSemanticRegistryConcept",
    "ObjectSceneSemanticRegistryError",
    "ObjectSceneSemanticRegistryPayloadError",
    "ObjectSceneSemanticRegistryProposal",
    "ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE",
    "SUPPORT_PANEL_COUNT",
    "SUPPORT_PANEL_COUNT_PER_ROLE",
    "build_object_scene_semantic_registry_gap",
    "build_object_scene_semantic_registry_proposal",
    "object_scene_semantic_registry_protocol_digest",
    "object_scene_semantic_registry_source_digest",
    "parse_object_scene_semantic_registry_proposal",
    "prepare_object_scene_semantic_registry_proposal",
    "verify_object_scene_semantic_registry_proposal",
)

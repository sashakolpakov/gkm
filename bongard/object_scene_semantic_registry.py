"""Role-aware multimodal concept proposal over frozen support evidence.

The proposer sees the prose frozen by the blind visual discovery pass plus the
same twelve already-exposed panels and proposal atlases under opaque aliases.
Only after the discovery freeze does it receive the committed support roles.
It proposes affirmative concepts for both orientations in one response.
Python then freezes the union as a scoped soft-tag registry; fresh role-blind
visual passes decide every resulting tag with the frontend's four dispositions.

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
    OBJECT_SCENE_MAX_ACCEPTED_VARIANTS,
    OBJECT_SCENE_MAX_NEAR_MISS_BOUNDARIES,
    OBJECT_SCENE_MAX_REGISTERED_TAGS,
    OBJECT_SCENE_MAX_REQUIRED_WITNESSES,
    OBJECT_SCENE_MAX_TAG_CHARACTERS,
    OBJECT_SCENE_MIN_REQUIRED_WITNESSES,
    OBJECT_SCENE_OPERATIONAL_WITNESS_KINDS,
    ObjectSceneOperationalWitness,
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
PREPARED_SCHEMA = "gkm.object-scene-semantic-registry-prepared.v6"
CONCEPT_SCHEMA = "gkm.object-scene-semantic-registry-concept.v6"
DROPPED_CONCEPT_SCHEMA = "gkm.object-scene-semantic-registry-dropped-concept.v2"
PROPOSAL_SCHEMA = "gkm.object-scene-semantic-registry-proposal.v6"
MAX_CONCEPTS_PER_ORIENTATION = 16
MAX_CONCEPT_PHRASE_CHARACTERS = OBJECT_SCENE_MAX_TAG_CHARACTERS
SUPPORT_PANEL_COUNT = 12
SUPPORT_PANEL_COUNT_PER_ROLE = 6
EXACT_SUPPORT_BINDINGS_PER_CONCEPT = SUPPORT_PANEL_COUNT_PER_ROLE

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ALIAS = re.compile(r"panel_[0-9]{3}\Z")
_ENTITY_ALIAS = re.compile(r"entity_[0-9]{3}\Z")
_SEMANTIC_POLICY_LEAK = re.compile(
    r"\b(?:side|support|bucket|orientation|more|less|fewer|most|least|"
    r"common|frequent|typically|usually|never|isn't|isnt|avoids?|avoiding|"
    r"fails?|cannot|can't|cant|or|plus|also|while|combined|"
    r"rarer|dominant|prevalent|exclusive|contrastive|"
    r"occurrences?|frequency|often|always|sometimes|only)\b|\bnon[- ]?[a-z]",
    re.IGNORECASE,
)
_OPERATIONAL_CRITERION_POLICY_LEAK = re.compile(
    r"\b(?:support(?:s|ed)?|bucket|orientation|historical[ -]?role|"
    r"side[01]|positive[ -]examples?|negative[ -]examples?|citations?|"
    r"frequency|frequent|typically|usually|never|always|sometimes|"
    r"prevalent|rarer|more[ -]common|less[ -]common|fewer[ -]panels?|"
    r"most[ -]panels?|least[ -]panels?|across[ -]panels?)\b|panel_[0-9]{3}\b",
    re.IGNORECASE,
)
_GAP_CODES = frozenset(("payload_rejected", "insufficient_discovery_evidence"))
_DROP_REASON_CODES = frozenset(
    (
        "malformed_concept",
        "binding_policy",
        "phrase_policy",
        "criteria_policy",
        "foreign_binding",
        "target_binding_policy",
        "duplicate_scoped_phrase",
    )
)


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
        "zero_image_proposer": False,
        "named_image_multimodal_proposer": True,
        "all_support_panels_and_atlases_attached": True,
        "every_card_binds_one_target_per_positive_support_panel": True,
        "both_orientations_in_one_call": True,
        "registered_evaluator_receives_roles": False,
        "semantic_proposal_is_not_a_truth_assignment": True,
        "soft_predicates_are_transparent_witness_macros": True,
        "registered_observer_authors_macro_disposition": False,
        "python_compiles_witness_dispositions": True,
        "support_binding_count_is_not_visual_confidence": True,
        "invalid_optional_concept_discards_valid_concepts": False,
        "all_quarantined_invalid_concepts_and_finite_reasons_persisted": True,
        "orientation_coverage_gap_suppresses_otherwise_valid_concepts_from_registry": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_or_decision": False,
    }


def object_scene_semantic_registry_protocol_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.object-scene-semantic-registry-protocol.v6",
            "source_digest": object_scene_semantic_registry_source_digest(),
            "frontend_source_digest": _frontend.object_scene_visual_frontend_source_digest(),
            "prepared_schema": PREPARED_SCHEMA,
            "concept_schema": CONCEPT_SCHEMA,
            "dropped_concept_schema": DROPPED_CONCEPT_SCHEMA,
            "proposal_schema": PROPOSAL_SCHEMA,
            "maximum_concepts_per_orientation": MAX_CONCEPTS_PER_ORIENTATION,
            "maximum_concept_phrase_characters": MAX_CONCEPT_PHRASE_CHARACTERS,
            "maximum_union_concepts": OBJECT_SCENE_MAX_REGISTERED_TAGS,
            "operational_witness_kinds": list(
                OBJECT_SCENE_OPERATIONAL_WITNESS_KINDS
            ),
            "minimum_required_witnesses": OBJECT_SCENE_MIN_REQUIRED_WITNESSES,
            "maximum_required_witnesses": OBJECT_SCENE_MAX_REQUIRED_WITNESSES,
            "maximum_accepted_variants": OBJECT_SCENE_MAX_ACCEPTED_VARIANTS,
            "maximum_near_miss_boundaries": (
                OBJECT_SCENE_MAX_NEAR_MISS_BOUNDARIES
            ),
            "exact_one_same_orientation_target_binding_per_panel": (
                EXACT_SUPPORT_BINDINGS_PER_CONCEPT
            ),
            "support_panel_count": SUPPORT_PANEL_COUNT,
            "support_panel_count_per_role": SUPPORT_PANEL_COUNT_PER_ROLE,
            "alias_order": "sha256-artifact-digest-then-opaque-sequential-alias",
            "registry_order": (
                "descending-distinct-bound-panel-count-then-scope-then-phrase"
            ),
            "optional_concept_failure_rule": (
                "quarantine-exact-input-row-with-finite-reason;"
                "accept-only-if-each-orientation-retains-one-concept"
            ),
            "duplicate_scoped_phrase_rule": "quarantine-every-member-of-group",
            "operational_card_rule": (
                "freeze-typed-affirmative-witnesses-inclusion-variants-and-"
                "near-miss-boundaries;observer-judges-witnesses-not-macro;"
                "python-error-dominant-strong-kleene-conjunction"
            ),
            "operational_card_identity_rule": (
                "criteria-digest-binds-cues-and-variants;"
                "tag-digest-binds-scope-phrase-and-criteria-digest"
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


def _semantic_image_name(alias: str, source_name: str) -> str:
    if _ALIAS.fullmatch(alias) is None:
        raise ObjectSceneSemanticRegistryError("semantic image alias differs")
    if source_name == "panel.png":
        return f"{alias}.png"
    if re.fullmatch(r"objects_[0-9]{3}\.png", source_name) is None:
        raise ObjectSceneSemanticRegistryError(
            "semantic source presentation name differs"
        )
    return f"{alias}_{source_name}"


def _transcript_view(
    alias: str,
    artifact: ObjectSceneTranscriptArtifact,
) -> dict[str, object]:
    transcript = artifact.transcript
    attached_images = [
        _semantic_image_name(alias, item.name) for item in artifact.presentation
    ]
    proposal_map = [
        {
            "entity_alias": f"entity_{index:03d}",
            "atlas_image": _semantic_image_name(alias, item.atlas_name),
            "atlas_row": item.row_index,
            "atlas_column": item.column_index,
        }
        for index, item in enumerate(artifact.inventory.objects)
    ]
    if transcript is None:
        return {
            "panel_alias": alias,
            "panel_image": _semantic_image_name(alias, "panel.png"),
            "attached_images": attached_images,
            "proposal_atlas_map": proposal_map,
            "observation_status": "unavailable",
            "panel_summary": None,
            "panel_open_tags": [],
            "entities": [],
        }
    return {
        "panel_alias": alias,
        "panel_image": _semantic_image_name(alias, "panel.png"),
        "attached_images": attached_images,
        "proposal_atlas_map": proposal_map,
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
    side0_aliases: Sequence[str],
    side1_aliases: Sequence[str],
    exposed_entity_aliases: Sequence[str],
) -> dict[str, object]:
    def concept(aliases: Sequence[str]) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "scope": {"type": "string", "enum": ["panel", "entity"]},
                "phrase": {
                    "type": "string",
                    "description": (
                        "One lowercase affirmative visual phrase of 2 to "
                        f"{MAX_CONCEPT_PHRASE_CHARACTERS} ASCII characters; "
                        "no negation, alternatives, labels, or bucket comparisons."
                    ),
                },
                "required_witnesses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": list(OBJECT_SCENE_OPERATIONAL_WITNESS_KINDS),
                            },
                            "statement": {"type": "string"},
                        },
                        "required": ["kind", "statement"],
                        "additionalProperties": False,
                    },
                    "description": (
                        f"Between {OBJECT_SCENE_MIN_REQUIRED_WITNESSES} and "
                        f"{OBJECT_SCENE_MAX_REQUIRED_WITNESSES} typed affirmative "
                        "visual witnesses. Each witness is one atomic check and "
                        "must not join alternatives with or or either. "
                        "Every witness must visibly hold "
                        "on the same scope-derived binding for PRESENT."
                    ),
                },
                "accepted_variants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Between 0 and "
                        f"{OBJECT_SCENE_MAX_ACCEPTED_VARIANTS} affirmative "
                        "inclusion/equivalence clauses saying which visible "
                        "variants count. Canonical comma-space lists and or are "
                        "allowed here; these clauses never vote."
                    ),
                },
                "near_miss_boundaries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Between 0 and "
                        f"{OBJECT_SCENE_MAX_NEAR_MISS_BOUNDARIES} explicit "
                        "exclusions for visually "
                        "confusable configurations. Describe the configuration "
                        "affirmatively, then use exactly one controlled exclusion "
                        "phrase; these clauses guide witness interpretation and "
                        "never vote."
                    ),
                },
                "support_bindings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "panel_alias": {
                                "type": "string",
                                "enum": list(aliases),
                            },
                            "target_alias": {
                                "type": "string",
                                "enum": [
                                    "whole_panel",
                                    *list(exposed_entity_aliases),
                                ],
                            },
                        },
                        "required": ["panel_alias", "target_alias"],
                        "additionalProperties": False,
                    },
                    "description": (
                        "Exactly six objects in ascending panel_alias order, "
                        "one for every panel in this orientation. For panel "
                        "scope target_alias is whole_panel. For entity scope "
                        "target_alias is one entity_alias actually exposed by "
                        "that panel's model-view row. All required witnesses "
                        "must hold on that one bound target; never pool cues "
                        "from multiple entities."
                    ),
                },
            },
            "required": [
                "scope",
                "phrase",
                "required_witnesses",
                "accepted_variants",
                "near_miss_boundaries",
                "support_bindings",
            ],
            "additionalProperties": False,
        }

    result: dict[str, object] = {
        "type": "object",
        "properties": {
            "side0_positive": {
                "type": "array",
                "items": concept(side0_aliases),
                "description": (
                    f"Between 1 and {MAX_CONCEPTS_PER_ORIENTATION} affirmative "
                    "concept objects for side0."
                ),
            },
            "side1_positive": {
                "type": "array",
                "items": concept(side1_aliases),
                "description": (
                    f"Between 1 and {MAX_CONCEPTS_PER_ORIENTATION} affirmative "
                    "concept objects for side1."
                ),
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
        "pixels_or_images_in_proposer_input": True,
        "all_support_presentations_in_proposer_input": True,
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
            "all_support_presentations_in_proposer_input",
            "task_lineage_ids_in_model_visible_input",
            "formula_candidates_in_model_visible_input", *_authority_data(),
            "preparation_digest",
        }
        raw = _fields(value, expected, "prepared semantic registry proposal")
        if (
            raw["schema"] != PREPARED_SCHEMA
            or raw["pixels_or_images_in_proposer_input"] is not True
            or raw["all_support_presentations_in_proposer_input"] is not True
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
            _transcript_view(alias, artifact)
        )
    all_role_aliases = {
        side: tuple(
            item["alias"]
            for item in bindings
            if item["historical_role"] == side
        )
        for side in (0, 1)
    }
    role_aliases = all_role_aliases
    model_view: dict[str, object] = {
        "side0_support_descriptions": model_rows[0],
        "side1_support_descriptions": model_rows[1],
        "required_positive_binding_panels": {
            "side0_positive": list(role_aliases[0]),
            "side1_positive": list(role_aliases[1]),
        },
    }
    exposed_entity_aliases = tuple(
        sorted(
            {
                str(entity["entity_alias"])
                for rows in model_rows.values()
                for row in rows
                for entity in row["proposal_atlas_map"]
            }
        )
    )
    output_schema = _proposal_output_schema(
        role_aliases[0], role_aliases[1], exposed_entity_aliases
    )
    prompt = (
        "Inspect every attached panel and proposal-atlas image together with "
        "the frozen blind descriptions below. Image filenames are opaque and "
        "are bound to panel_alias and entity atlas locations in the supplied "
        "view. The two support orientations were revealed only after blind "
        "discovery was durably frozen. Propose candidate visual concepts for "
        "BOTH support orientations in one response. Prefer the strongest "
        "concepts visibly shared by every panel in their claimed orientation "
        "and useful for distinguishing that orientation from the other one. "
        "Each concept "
        "must be a single lowercase affirmative visual phrase, scoped either "
        "to the whole panel or to one visible entity. For every phrase, freeze "
        "an operational card rather than merely restating its label: provide "
        "one to three required_witnesses. Each witness has a bounded kind and "
        "one affirmative visually local statement that can be judged separately "
        "from the visible rendering. A required witness must state one check: "
        "never join alternative cues with 'or' or 'either'. Put category "
        "alternatives in accepted_variants instead. Also provide zero to two "
        "accepted_variants that state affirmative visual inclusions or "
        "equivalences; these may enumerate alternatives with canonical commas "
        "and 'or'. Provide zero to "
        "two near_miss_boundaries that explicitly exclude visually confusable "
        "configurations. Variants and boundaries clarify witness interpretation; "
        "they never replace a witness and never vote. For entity scope, every "
        "witness binds to the same single frozen entity proposal. For panel "
        "scope, every witness binds to the same whole composition. A fresh "
        "role-blind observer will judge each witness separately. Python alone "
        "will compile the card: ERROR dominates; otherwise any clearly "
        "contradicted witness yields CERTIFIED_ABSENT; all PRESENT yields PRESENT; "
        "every other combination yields INDETERMINATE. Do not ask the observer "
        "for the card's final state. Resolve meaningful visual tolerances and "
        "category boundaries such as wedge/fan/sector in the card. Do not use "
        "a support binding, support frequency, or bucket identity "
        "as a visual cue. Every card must provide support_bindings in ascending "
        "panel_alias order with exactly one entry for each of the six panel "
        "aliases in its own orientation. Each entry has exactly panel_alias and "
        "target_alias. For panel scope, target_alias must be whole_panel. For "
        "entity scope, target_alias must be one entity_alias actually exposed in "
        "that panel's model-view row. Every required witness in the card must "
        "visibly hold on that same single bound target; never pool one cue from "
        "one entity with another cue from a different entity. A support binding "
        "is a proposal-level grounding claim, not a verified truth assignment. "
        "In concept phrases, required "
        "witnesses, and accepted variants, do not compare buckets, mention "
        "experimental roles or labels, use negation, or say something is "
        "missing. Near-miss boundaries are the sole exception: phrase each as "
        "a controlled exclusion using 'does not qualify', 'is excluded', or "
        "'falls outside'. The configuration before that exclusion must itself be "
        "affirmative: do not use no, none, neither, never, without, lacks, "
        "missing, or a second exclusion phrase. Ordinary internal visual "
        "relations and conjunctions such "
        "as mismatched upper and lower portions, circular and triangular marks, "
        "lower-left placement, unequal edge lengths, visible paths, and visible "
        "sides are affirmative and allowed. Every phrase must be 2 to 80 ASCII "
        "characters. Every witness statement, accepted variant, and near-miss "
        "boundary must be 8 to 160 ASCII characters. In witness statements and "
        "near-miss boundaries use only lowercase ASCII letters, spaces, "
        "apostrophes, and hyphens. Accepted variants may additionally use "
        "canonical comma-space separators. Supply at most 16 concepts "
        "per bucket. Python will discard bucket membership, freeze one union of "
        "transparent witness macros, and two fresh role-blind visual passes will "
        "judge their witnesses. Return only the required JSON object.\n\nFrozen descriptions:\n"
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
            sorted(artifact.panel_digest for artifact in artifacts)
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


def _support_bindings_data(
    value: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    return [
        {"panel_alias": panel_alias, "target_alias": target_alias}
        for panel_alias, target_alias in value
    ]


def _support_bindings_digest(value: Sequence[tuple[str, str]]) -> str:
    return canonical_digest(
        {
            "schema": "gkm.object-scene-semantic-support-bindings.v1",
            "support_bindings": _support_bindings_data(value),
            "one_target_per_positive_panel": True,
        }
    )


def _normalized_support_bindings(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise ObjectSceneSemanticRegistryError(
            "semantic concept support bindings must be an array"
        )
    rows: list[tuple[str, str]] = []
    for item in value:
        raw = _fields(
            item,
            {"panel_alias", "target_alias"},
            "semantic concept support binding",
        )
        panel_alias = raw["panel_alias"]
        target_alias = raw["target_alias"]
        if (
            not isinstance(panel_alias, str)
            or _ALIAS.fullmatch(panel_alias) is None
            or not isinstance(target_alias, str)
            or (
                target_alias != "whole_panel"
                and _ENTITY_ALIAS.fullmatch(target_alias) is None
            )
        ):
            raise ObjectSceneSemanticRegistryError(
                "semantic concept support binding aliases differ"
            )
        rows.append((panel_alias, target_alias))
    result = tuple(rows)
    if (
        len(result) != EXACT_SUPPORT_BINDINGS_PER_CONCEPT
        or len({panel_alias for panel_alias, _ in result}) != len(result)
        or result != tuple(sorted(result, key=lambda item: item[0]))
    ):
        raise ObjectSceneSemanticRegistryError(
            "semantic concept support bindings are not exact and canonical"
        )
    return result


def _concept_content(value: "ObjectSceneSemanticRegistryConcept") -> dict[str, object]:
    return {
        "schema": CONCEPT_SCHEMA,
        "orientation": value.orientation,
        "scope": value.scope,
        "phrase": value.phrase,
        "required_witnesses": [
            item.to_data() for item in value.required_witnesses
        ],
        "accepted_variants": list(value.accepted_variants),
        "near_miss_boundaries": list(value.near_miss_boundaries),
        "criteria_digest": value.criteria_digest,
        "support_bindings": _support_bindings_data(value.support_bindings),
        "support_binding_count": len(value.support_bindings),
        "support_bindings_digest": value.support_bindings_digest,
        "affirmative_observation_hypothesis_not_truth": True,
        "observer_judges_witnesses_not_macro": True,
        "python_compiles_macro_disposition": True,
    }


def _normalized_semantic_scope_phrase(
    scope: object,
    phrase: object,
) -> tuple[str, str]:
    try:
        normalized_scope = _frontend._soft_tag_scope(scope)
        normalized_phrase = _frontend._normalized_positive_tag(phrase)
    except Exception as exc:
        raise ObjectSceneSemanticRegistryError(
            "concept is not scoped affirmative visual prose"
        ) from exc
    if (
        len(normalized_phrase) > MAX_CONCEPT_PHRASE_CHARACTERS
        or _SEMANTIC_POLICY_LEAK.search(normalized_phrase) is not None
    ):
        raise ObjectSceneSemanticRegistryError(
            "concept phrase contains policy or experimental logic"
        )
    return normalized_scope, normalized_phrase


def _compiled_semantic_operational_card(
    scope: str,
    phrase: str,
    required_witnesses: object,
    accepted_variants: object,
    near_miss_boundaries: object,
) -> ObjectSceneSoftTag:
    try:
        card = ObjectSceneSoftTag.create(
            "tag_0000",
            scope,
            phrase,
            EXACT_SUPPORT_BINDINGS_PER_CONCEPT,
            required_witnesses,
            accepted_variants,
            near_miss_boundaries,
        )
    except Exception as exc:
        raise ObjectSceneSemanticRegistryError(
            "concept operational witness card differs"
        ) from exc
    clauses = (
        *(item.statement for item in card.required_witnesses),
        *card.accepted_variants,
        *card.near_miss_boundaries,
    )
    if any(_OPERATIONAL_CRITERION_POLICY_LEAK.search(item) for item in clauses):
        raise ObjectSceneSemanticRegistryError(
            "concept operational witness card leaks experimental policy"
        )
    return card


@dataclass(frozen=True, order=True, slots=True)
class ObjectSceneSemanticRegistryConcept:
    orientation: str
    scope: str
    phrase: str
    required_witnesses: tuple[ObjectSceneOperationalWitness, ...]
    accepted_variants: tuple[str, ...]
    near_miss_boundaries: tuple[str, ...]
    criteria_digest: str
    support_bindings: tuple[tuple[str, str], ...]
    support_bindings_digest: str
    concept_digest: str

    def __post_init__(self) -> None:
        if self.orientation not in ("side0_positive", "side1_positive"):
            raise ObjectSceneSemanticRegistryError("concept orientation differs")
        scope, phrase = _normalized_semantic_scope_phrase(
            self.scope, self.phrase
        )
        card = _compiled_semantic_operational_card(
            scope,
            phrase,
            self.required_witnesses,
            self.accepted_variants,
            self.near_miss_boundaries,
        )
        if (
            scope != self.scope
            or phrase != self.phrase
            or card.required_witnesses != self.required_witnesses
            or card.accepted_variants != self.accepted_variants
            or card.near_miss_boundaries != self.near_miss_boundaries
            or card.criteria_digest != self.criteria_digest
            or len(self.support_bindings) != EXACT_SUPPORT_BINDINGS_PER_CONCEPT
            or self.support_bindings
            != tuple(sorted(self.support_bindings, key=lambda item: item[0]))
            or len({item[0] for item in self.support_bindings})
            != len(self.support_bindings)
            or (
                scope == "panel"
                and any(
                    target_alias != "whole_panel"
                    for _, target_alias in self.support_bindings
                )
            )
            or (
                scope == "entity"
                and any(
                    target_alias == "whole_panel"
                    for _, target_alias in self.support_bindings
                )
            )
            or any(
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not isinstance(item[1], str)
                or _ALIAS.fullmatch(item[0]) is None
                or (
                    item[1] != "whole_panel"
                    and _ENTITY_ALIAS.fullmatch(item[1]) is None
                )
                for item in self.support_bindings
            )
            or self.support_bindings_digest
            != _support_bindings_digest(self.support_bindings)
        ):
            raise ObjectSceneSemanticRegistryError(
                "semantic concept phrase, card, or support bindings differ"
            )
        _digest(self.criteria_digest, "semantic concept criteria digest")
        _digest(
            self.support_bindings_digest,
            "semantic concept support bindings digest",
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
        required_witnesses: object,
        accepted_variants: object,
        near_miss_boundaries: object,
        support_bindings: object,
    ) -> "ObjectSceneSemanticRegistryConcept":
        normalized_scope, normalized_phrase = _normalized_semantic_scope_phrase(
            scope, phrase
        )
        card = _compiled_semantic_operational_card(
            normalized_scope,
            normalized_phrase,
            required_witnesses,
            accepted_variants,
            near_miss_boundaries,
        )
        normalized_bindings = _normalized_support_bindings(support_bindings)
        provisional = object.__new__(cls)
        values = {
            "orientation": orientation,
            "scope": normalized_scope,
            "phrase": normalized_phrase,
            "required_witnesses": card.required_witnesses,
            "accepted_variants": card.accepted_variants,
            "near_miss_boundaries": card.near_miss_boundaries,
            "criteria_digest": card.criteria_digest,
            "support_bindings": normalized_bindings,
            "support_bindings_digest": _support_bindings_digest(
                normalized_bindings
            ),
        }
        for key, item in values.items():
            object.__setattr__(provisional, key, item)
        return cls(**values, concept_digest=canonical_digest(_concept_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_concept_content(self), "concept_digest": self.concept_digest}

    @property
    def citations(self) -> tuple[str, ...]:
        """Derived bound-panel inventory for legacy in-process consumers."""

        return tuple(panel_alias for panel_alias, _ in self.support_bindings)

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneSemanticRegistryConcept":
        raw = _fields(
            value,
            {
                "schema", "orientation", "scope", "phrase",
                "required_witnesses", "accepted_variants",
                "near_miss_boundaries", "criteria_digest", "support_bindings",
                "support_binding_count", "support_bindings_digest",
                "affirmative_observation_hypothesis_not_truth",
                "observer_judges_witnesses_not_macro",
                "python_compiles_macro_disposition", "concept_digest",
            },
            "semantic registry concept",
        )
        if (
            raw["schema"] != CONCEPT_SCHEMA
            or raw["affirmative_observation_hypothesis_not_truth"] is not True
            or raw["observer_judges_witnesses_not_macro"] is not True
            or raw["python_compiles_macro_disposition"] is not True
            or raw["support_binding_count"]
            != (
                len(raw["support_bindings"])
                if isinstance(raw["support_bindings"], list)
                else -1
            )
            or not isinstance(raw["support_bindings"], list)
            or not isinstance(raw["required_witnesses"], list)
            or not isinstance(raw["accepted_variants"], list)
            or not isinstance(raw["near_miss_boundaries"], list)
        ):
            raise ObjectSceneSemanticRegistryError("semantic concept policy differs")
        try:
            result = cls(
                raw["orientation"], raw["scope"], raw["phrase"],
                tuple(
                    ObjectSceneOperationalWitness.from_data(item)
                    for item in raw["required_witnesses"]
                ),
                tuple(raw["accepted_variants"]),
                tuple(raw["near_miss_boundaries"]),
                raw["criteria_digest"],
                _normalized_support_bindings(raw["support_bindings"]),
                raw["support_bindings_digest"],
                raw["concept_digest"],
            )
        except ObjectSceneSemanticRegistryError:
            raise
        except Exception as exc:
            raise ObjectSceneSemanticRegistryError(
                "semantic concept operational card differs"
            ) from exc
        if result.to_data() != dict(raw):
            raise ObjectSceneSemanticRegistryError("semantic concept is not canonical")
        return result


def _dropped_concept_content(
    value: "ObjectSceneDroppedSemanticRegistryConcept",
) -> dict[str, object]:
    return {
        "schema": DROPPED_CONCEPT_SCHEMA,
        "orientation": value.orientation,
        "input_index": value.input_index,
        "payload_digest": value.payload_digest,
        "reason_code": value.reason_code,
        "optional_bad_concept_does_not_discard_valid_concepts": True,
    }


@dataclass(frozen=True, order=True, slots=True)
class ObjectSceneDroppedSemanticRegistryConcept:
    orientation: str
    input_index: int
    payload_digest: str
    reason_code: str
    drop_digest: str

    def __post_init__(self) -> None:
        if (
            self.orientation not in ("side0_positive", "side1_positive")
            or type(self.input_index) is not int
            or self.input_index < 0
            or self.reason_code not in _DROP_REASON_CODES
        ):
            raise ObjectSceneSemanticRegistryError(
                "dropped semantic concept disposition differs"
            )
        _digest(self.payload_digest, "dropped concept payload digest")
        _digest(self.drop_digest, "dropped concept digest")
        if self.drop_digest != canonical_digest(_dropped_concept_content(self)):
            raise ObjectSceneSemanticRegistryError(
                "dropped semantic concept digest differs"
            )

    @classmethod
    def create(
        cls,
        orientation: str,
        input_index: int,
        payload: object,
        reason_code: str,
    ) -> "ObjectSceneDroppedSemanticRegistryConcept":
        values = {
            "orientation": orientation,
            "input_index": input_index,
            "payload_digest": canonical_digest(payload),
            "reason_code": reason_code,
        }
        provisional = object.__new__(cls)
        for key, item in values.items():
            object.__setattr__(provisional, key, item)
        return cls(
            **values,
            drop_digest=canonical_digest(_dropped_concept_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_dropped_concept_content(self), "drop_digest": self.drop_digest}

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectSceneDroppedSemanticRegistryConcept":
        raw = _fields(
            value,
            {
                "schema",
                "orientation",
                "input_index",
                "payload_digest",
                "reason_code",
                "optional_bad_concept_does_not_discard_valid_concepts",
                "drop_digest",
            },
            "dropped semantic concept",
        )
        if (
            raw["schema"] != DROPPED_CONCEPT_SCHEMA
            or raw["optional_bad_concept_does_not_discard_valid_concepts"]
            is not True
        ):
            raise ObjectSceneSemanticRegistryError(
                "dropped semantic concept policy differs"
            )
        result = cls(
            raw["orientation"],
            raw["input_index"],
            raw["payload_digest"],
            raw["reason_code"],
            raw["drop_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneSemanticRegistryError(
                "dropped semantic concept is not canonical"
            )
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
        "dropped_concepts": [item.to_data() for item in value.dropped_concepts],
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
    dropped_concepts: tuple[ObjectSceneDroppedSemanticRegistryConcept, ...]
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
            or self.dropped_concepts
            != tuple(
                sorted(
                    self.dropped_concepts,
                    key=lambda item: (
                        item.orientation,
                        item.input_index,
                        item.payload_digest,
                    ),
                )
            )
            or len(
                {
                    (item.orientation, item.input_index)
                    for item in self.dropped_concepts
                }
            )
            != len(self.dropped_concepts)
        ):
            raise ObjectSceneSemanticRegistryError("semantic proposal concept inventory differs")
        if self.model_payload is not None:
            for dropped in self.dropped_concepts:
                bucket = self.model_payload.get(dropped.orientation)
                if (
                    not isinstance(bucket, list)
                    or dropped.input_index >= len(bucket)
                    or canonical_digest(bucket[dropped.input_index])
                    != dropped.payload_digest
                ):
                    raise ObjectSceneSemanticRegistryError(
                        "dropped semantic concept payload binding differs"
                    )
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
            "side0_positive", "side1_positive", "dropped_concepts", "gap_code", "registry_digest",
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
                    "dropped_concepts",
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
            tuple(
                ObjectSceneDroppedSemanticRegistryConcept.from_data(item)
                for item in raw["dropped_concepts"]
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
        ObjectSceneSoftTag.create(
            f"tag_{index:04d}",
            item.scope,
            item.phrase,
            len(item.citations),
            item.required_witnesses,
            item.accepted_variants,
            item.near_miss_boundaries,
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
    dropped: Sequence[ObjectSceneDroppedSemanticRegistryConcept],
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
        "dropped_concepts": tuple(dropped),
        "gap_code": gap_code,
        "registry_digest": registry.registry_digest,
    }
    provisional = object.__new__(ObjectSceneSemanticRegistryProposal)
    for key, item in values.items():
        object.__setattr__(provisional, key, item)
    return ObjectSceneSemanticRegistryProposal(
        **values, proposal_digest=canonical_digest(_proposal_content(provisional))
    )


def _exposed_entity_targets_by_panel(
    prepared: ObjectScenePreparedSemanticRegistryProposal,
) -> dict[str, frozenset[str]]:
    result: dict[str, frozenset[str]] = {}
    for key in ("side0_support_descriptions", "side1_support_descriptions"):
        rows = prepared.model_view.get(key)
        if not isinstance(rows, list):
            raise ObjectSceneSemanticRegistryError(
                "prepared semantic model-view support rows differ"
            )
        for row in rows:
            if not isinstance(row, Mapping):
                raise ObjectSceneSemanticRegistryError(
                    "prepared semantic model-view support row differs"
                )
            panel_alias = row.get("panel_alias")
            proposal_atlas_map = row.get("proposal_atlas_map")
            if (
                not isinstance(panel_alias, str)
                or _ALIAS.fullmatch(panel_alias) is None
                or panel_alias in result
                or not isinstance(proposal_atlas_map, list)
            ):
                raise ObjectSceneSemanticRegistryError(
                    "prepared semantic model-view target inventory differs"
                )
            aliases: set[str] = set()
            for entity in proposal_atlas_map:
                target_alias = (
                    entity.get("entity_alias")
                    if isinstance(entity, Mapping)
                    else None
                )
                if (
                    not isinstance(target_alias, str)
                    or _ENTITY_ALIAS.fullmatch(target_alias) is None
                    or target_alias in aliases
                ):
                    raise ObjectSceneSemanticRegistryError(
                        "prepared semantic model-view entity target differs"
                    )
                aliases.add(target_alias)
            result[panel_alias] = frozenset(aliases)
    if set(result) != {
        str(item["alias"]) for item in prepared.alias_bindings
    }:
        raise ObjectSceneSemanticRegistryError(
            "prepared semantic model-view panel targets differ"
        )
    return result


def _project_semantic_payload(
    prepared: ObjectScenePreparedSemanticRegistryProposal,
    payload: Mapping[str, Any],
    *,
    require_usable_buckets: bool,
) -> tuple[
    Mapping[str, Any],
    dict[int, list[ObjectSceneSemanticRegistryConcept]],
    tuple[ObjectSceneDroppedSemanticRegistryConcept, ...],
]:
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
    except ObjectSceneSemanticRegistryPayloadError:
        raise
    except ObjectSceneSemanticRegistryError as exc:
        raise ObjectSceneSemanticRegistryPayloadError(str(exc)) from exc

    allowed = {
        side: {
            item["alias"]
            for item in prepared.alias_bindings
            if item["historical_role"] == side
        }
        for side in (0, 1)
    }
    exposed_entity_targets = _exposed_entity_targets_by_panel(prepared)
    indexed: dict[
        int, list[tuple[int, ObjectSceneSemanticRegistryConcept]]
    ] = {0: [], 1: []}
    dropped: list[ObjectSceneDroppedSemanticRegistryConcept] = []
    for side, key in ((0, "side0_positive"), (1, "side1_positive")):
        for input_index, item in enumerate(raw[key]):
            try:
                concept_raw = _fields(
                    item,
                    {
                        "scope",
                        "phrase",
                        "required_witnesses",
                        "accepted_variants",
                        "near_miss_boundaries",
                        "support_bindings",
                    },
                    "semantic concept payload",
                )
            except ObjectSceneSemanticRegistryError:
                dropped.append(
                    ObjectSceneDroppedSemanticRegistryConcept.create(
                        key, input_index, item, "malformed_concept"
                    )
                )
                continue
            try:
                support_bindings = _normalized_support_bindings(
                    concept_raw["support_bindings"]
                )
            except ObjectSceneSemanticRegistryError:
                dropped.append(
                    ObjectSceneDroppedSemanticRegistryConcept.create(
                        key, input_index, item, "binding_policy"
                    )
                )
                continue
            bound_panels = {panel_alias for panel_alias, _ in support_bindings}
            if not bound_panels.issubset(allowed[side]):
                dropped.append(
                    ObjectSceneDroppedSemanticRegistryConcept.create(
                        key, input_index, item, "foreign_binding"
                    )
                )
                continue
            if bound_panels != allowed[side]:
                dropped.append(
                    ObjectSceneDroppedSemanticRegistryConcept.create(
                        key, input_index, item, "binding_policy"
                    )
                )
                continue
            try:
                normalized_scope, _ = _normalized_semantic_scope_phrase(
                    concept_raw["scope"], concept_raw["phrase"]
                )
            except ObjectSceneSemanticRegistryError:
                dropped.append(
                    ObjectSceneDroppedSemanticRegistryConcept.create(
                        key, input_index, item, "phrase_policy"
                    )
                )
                continue
            if (
                normalized_scope == "panel"
                and any(
                    target_alias != "whole_panel"
                    for _, target_alias in support_bindings
                )
            ) or (
                normalized_scope == "entity"
                and any(
                    target_alias not in exposed_entity_targets[panel_alias]
                    for panel_alias, target_alias in support_bindings
                )
            ):
                dropped.append(
                    ObjectSceneDroppedSemanticRegistryConcept.create(
                        key, input_index, item, "target_binding_policy"
                    )
                )
                continue
            if any(
                not isinstance(concept_raw[field], list)
                for field in (
                    "required_witnesses",
                    "accepted_variants",
                    "near_miss_boundaries",
                )
            ):
                dropped.append(
                    ObjectSceneDroppedSemanticRegistryConcept.create(
                        key, input_index, item, "criteria_policy"
                    )
                )
                continue
            try:
                concept = ObjectSceneSemanticRegistryConcept.create(
                    key,
                    concept_raw["scope"],
                    concept_raw["phrase"],
                    concept_raw["required_witnesses"],
                    concept_raw["accepted_variants"],
                    concept_raw["near_miss_boundaries"],
                    concept_raw["support_bindings"],
                )
            except ObjectSceneSemanticRegistryError:
                dropped.append(
                    ObjectSceneDroppedSemanticRegistryConcept.create(
                        key, input_index, item, "criteria_policy"
                    )
                )
                continue
            indexed[side].append((input_index, concept))

    key_counts: dict[tuple[str, str], int] = {}
    for rows in indexed.values():
        for _, concept in rows:
            identity = (concept.scope, concept.phrase)
            key_counts[identity] = key_counts.get(identity, 0) + 1
    repeated = {identity for identity, count in key_counts.items() if count > 1}
    buckets: dict[int, list[ObjectSceneSemanticRegistryConcept]] = {0: [], 1: []}
    for side, key in ((0, "side0_positive"), (1, "side1_positive")):
        for input_index, concept in indexed[side]:
            if (concept.scope, concept.phrase) in repeated:
                dropped.append(
                    ObjectSceneDroppedSemanticRegistryConcept.create(
                        key,
                        input_index,
                        raw[key][input_index],
                        "duplicate_scoped_phrase",
                    )
                )
            else:
                buckets[side].append(concept)
        buckets[side].sort(
            key=lambda item: (item.scope, item.phrase, item.support_bindings)
        )
    dropped.sort(
        key=lambda item: (item.orientation, item.input_index, item.payload_digest)
    )
    if require_usable_buckets and any(not buckets[side] for side in (0, 1)):
        raise ObjectSceneSemanticRegistryPayloadError(
            "semantic proposal has no usable concept in one or both buckets"
        )
    return raw, buckets, tuple(dropped)


def build_object_scene_semantic_registry_proposal(
    prepared: ObjectScenePreparedSemanticRegistryProposal,
    payload: Mapping[str, Any],
) -> tuple[ObjectSceneSemanticRegistryProposal, ObjectSceneSoftTagRegistry]:
    if not isinstance(prepared, ObjectScenePreparedSemanticRegistryProposal):
        raise TypeError("prepared must be ObjectScenePreparedSemanticRegistryProposal")
    ObjectScenePreparedSemanticRegistryProposal.from_data(prepared.to_data())
    raw, buckets, dropped = _project_semantic_payload(
        prepared, payload, require_usable_buckets=True
    )
    concepts = (*buckets[0], *buckets[1])
    registry = _semantic_registry(prepared, concepts)
    proposal = _proposal(
        prepared,
        status="proposed",
        model_payload=raw,
        side0=buckets[0],
        side1=buckets[1],
        dropped=dropped,
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
    dropped: tuple[ObjectSceneDroppedSemanticRegistryConcept, ...] = ()
    if rejected_payload is not None:
        try:
            _, projected, dropped = _project_semantic_payload(
                prepared,
                rejected_payload,
                require_usable_buckets=False,
            )
        except ObjectSceneSemanticRegistryPayloadError:
            pass
        else:
            if gap_code == "payload_rejected" and all(
                projected[side] for side in (0, 1)
            ):
                raise ObjectSceneSemanticRegistryError(
                    "payload-rejected gap binds a usable semantic proposal"
                )
    registry = _semantic_registry(prepared, ())
    return (
        _proposal(
            prepared,
            status="typed_proposal_gap",
            model_payload=rejected_payload,
            side0=(),
            side1=(),
            dropped=dropped,
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
    "MAX_CONCEPT_PHRASE_CHARACTERS",
    "ObjectSceneDroppedSemanticRegistryConcept",
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

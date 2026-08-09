"""Text-only ranking of an exact nonempty anchor-predicate survivor set.

The deterministic Python version space decides admissibility.  Codex receives
only bounded aliases, one closed anchor kind per alias, and the affirmative
visible statements already frozen into that survivor.  It may return one exact
permutation; it cannot add, remove, rewrite, or synthesize a predicate.
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

import bongard.transport as _transport_runtime
from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import (
    CodexNoToolsAttestation,
    validate_codex_no_tools_attestation,
)
from bongard.object_scene_anchor_version_space import (
    ObjectSceneAnchorOrientation,
    ObjectSceneAnchorPredicateCandidate,
    ObjectSceneAnchorSupportVersionSpace,
    object_scene_anchor_version_space_algorithm_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
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


OBJECT_SCENE_ANCHOR_CANDIDATE_RANKER_PROTOCOL_ID = (
    "bongard.object-scene-anchor-candidate-ranker/text-only-exact-union-permutation-v2"
)
OBJECT_SCENE_ANCHOR_CANDIDATE_RANKER_PROTOCOL_SCHEMA = (
    "gkm.object-scene-anchor-candidate-ranker-protocol.v2"
)
OBJECT_SCENE_ANCHOR_RANK_CANDIDATE_SCHEMA = (
    "gkm.object-scene-anchor-rank-candidate.v1"
)
OBJECT_SCENE_ANCHOR_RANK_INPUT_SCHEMA = "gkm.object-scene-anchor-rank-input.v2"
OBJECT_SCENE_ANCHOR_RANK_OUTPUT_SCHEMA = "gkm.object-scene-anchor-rank-output.v1"
OBJECT_SCENE_ANCHOR_RANK_RESPONSE_SCHEMA = "gkm.object-scene-anchor-rank-response.v2"
OBJECT_SCENE_ANCHOR_RANK_RECEIPT_SCHEMA = "gkm.object-scene-anchor-rank-receipt.v1"

MAX_SURVIVOR_COUNT = 64
MAX_PROMPT_UTF8_BYTES = 64_000

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_ALIAS = re.compile(r"choice_[0-9]{3}\Z")
_FORBIDDEN_VISIBLE = re.compile(
    r"\b(?:panel|query|target|foil|support|contrast|side[01]|orientation|"
    r"formula|predicate|digest|pixel|image)\b",
    re.IGNORECASE,
)


class ObjectSceneAnchorCandidateRankerError(RuntimeError):
    """A rank input, output, receipt, runtime pin, or replay is invalid."""


class ObjectSceneAnchorRankCapacityGap(ObjectSceneAnchorCandidateRankerError):
    """The exact survivor union cannot fit the fixed structured-output schema."""

    def __init__(
        self,
        survivor_count: int,
        maximum_survivor_count: int,
        child_version_space_digests: Sequence[str],
    ) -> None:
        self.survivor_count = survivor_count
        self.maximum_survivor_count = maximum_survivor_count
        self.child_version_space_digests = tuple(child_version_space_digests)
        super().__init__(
            "verified survivor union exceeds the ranker capacity; "
            "no candidates were pruned"
        )


TextStructuredTransport = Callable[..., CodexStructuredResult]


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "codex_may_rank_verified_survivors_only": True,
        "codex_may_edit_candidates": False,
        "complete_survivor_permutation_required": True,
        "model_visible_visual_material": False,
        "model_visible_panel_identities": False,
        "model_visible_query_material": False,
    }


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectSceneAnchorCandidateRankerError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectSceneAnchorCandidateRankerError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectSceneAnchorCandidateRankerError(f"{label} must be a sha256: address")
    return value


def _canonical_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ObjectSceneAnchorCandidateRankerError("rank payload must be a JSON object")
    try:
        decoded = json.loads(canonical_json(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectSceneAnchorCandidateRankerError(
            "rank payload is not finite canonical JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise ObjectSceneAnchorCandidateRankerError("rank payload must be an object")
    return decoded


def object_scene_anchor_candidate_ranker_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def object_scene_anchor_candidate_ranker_transport_source_digest() -> str:
    source = getattr(_transport_runtime, "__file__", None)
    if not isinstance(source, str) or not source:
        raise ObjectSceneAnchorCandidateRankerError("text transport source is unavailable")
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def _rank_candidate_content(value: "ObjectSceneAnchorRankCandidate") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_RANK_CANDIDATE_SCHEMA,
        "alias": value.alias,
        "candidate_digest": value.candidate_digest,
        "anchor_kind": value.anchor_kind,
        "witness_digests": list(value.witness_digests),
        "affirmative_statements": list(value.affirmative_statements),
        "model_visible_fields": ["alias", "anchor_kind", "affirmative_statements"],
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorRankCandidate:
    """Exact survivor identity plus its bounded model-visible projection."""

    alias: str
    candidate_digest: str
    anchor_kind: str
    witness_digests: tuple[str, ...]
    affirmative_statements: tuple[str, ...]
    presentation_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.alias, str) or _ALIAS.fullmatch(self.alias) is None:
            raise ObjectSceneAnchorCandidateRankerError("rank alias differs")
        _digest(self.candidate_digest, "rank candidate digest")
        if self.anchor_kind not in ("entity", "part", "frame"):
            raise ObjectSceneAnchorCandidateRankerError("rank anchor kind differs")
        if (
            type(self.witness_digests) is not tuple
            or not 1 <= len(self.witness_digests) <= 3
            or self.witness_digests != tuple(sorted(set(self.witness_digests)))
            or type(self.affirmative_statements) is not tuple
            or len(self.affirmative_statements) != len(self.witness_digests)
        ):
            raise ObjectSceneAnchorCandidateRankerError("rank witness inventory differs")
        for item in self.witness_digests:
            _digest(item, "rank witness digest")
        for statement in self.affirmative_statements:
            if (
                not isinstance(statement, str)
                or statement != statement.strip()
                or not 3 <= len(statement) <= 240
                or not statement.isprintable()
                or _FORBIDDEN_VISIBLE.search(statement) is not None
            ):
                raise ObjectSceneAnchorCandidateRankerError(
                    "rank statement crosses the text-only boundary"
                )
        _digest(self.presentation_digest, "rank presentation digest")
        if self.presentation_digest != canonical_digest(_rank_candidate_content(self)):
            raise ObjectSceneAnchorCandidateRankerError("rank presentation digest differs")

    @classmethod
    def create(
        cls,
        *,
        alias: str,
        candidate_digest: str,
        anchor_kind: str,
        witness_digests: Sequence[str],
        affirmative_statements: Sequence[str],
    ) -> "ObjectSceneAnchorRankCandidate":
        values = {
            "alias": alias,
            "candidate_digest": candidate_digest,
            "anchor_kind": anchor_kind,
            "witness_digests": tuple(witness_digests),
            "affirmative_statements": tuple(affirmative_statements),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            presentation_digest=canonical_digest(_rank_candidate_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_rank_candidate_content(self), "presentation_digest": self.presentation_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorRankCandidate":
        raw = _fields(
            value,
            {
                "schema", "alias", "candidate_digest", "anchor_kind",
                "witness_digests", "affirmative_statements", "model_visible_fields",
                *_authority_data(), "presentation_digest",
            },
            "rank candidate",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_RANK_CANDIDATE_SCHEMA
            or raw["model_visible_fields"] != ["alias", "anchor_kind", "affirmative_statements"]
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["witness_digests"], list)
            or not isinstance(raw["affirmative_statements"], list)
        ):
            raise ObjectSceneAnchorCandidateRankerError("rank candidate policy differs")
        result = cls(
            raw["alias"], raw["candidate_digest"], raw["anchor_kind"],
            tuple(raw["witness_digests"]), tuple(raw["affirmative_statements"]),
            raw["presentation_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorCandidateRankerError("rank candidate is not canonical")
        return result


def _rank_scope_digest(
    child_version_space_digests: Sequence[str],
    child_orientations: Sequence[str],
    *,
    version_space_algorithm_digest: str,
    language_digest: str,
) -> str:
    if len(child_version_space_digests) == 1:
        return child_version_space_digests[0]
    return canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-rank-scope.v1",
            "child_version_spaces": [
                {
                    "version_space_digest": digest,
                    "orientation": orientation,
                }
                for digest, orientation in zip(
                    child_version_space_digests, child_orientations, strict=True
                )
            ],
            "version_space_algorithm_digest": version_space_algorithm_digest,
            "language_digest": language_digest,
            "all_nonempty_children_required": True,
            "exact_survivor_union_required": True,
            "no_orientation_preference": True,
        }
    )


def _rank_input_content(value: "ObjectSceneAnchorRankInput") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_RANK_INPUT_SCHEMA,
        "version_space_digest": value.version_space_digest,
        "version_space_algorithm_digest": value.version_space_algorithm_digest,
        "language_digest": value.language_digest,
        "child_version_space_digests": list(value.child_version_space_digests),
        "child_orientations": list(value.child_orientations),
        "survivor_candidate_digests": list(value.survivor_candidate_digests),
        "candidate_origin_version_space_digests": list(
            value.candidate_origin_version_space_digests
        ),
        "candidate_origin_orientations": list(value.candidate_origin_orientations),
        "candidates": [item.to_data() for item in value.candidates],
        "survivor_count": value.survivor_count,
        "maximum_survivor_count": MAX_SURVIVOR_COUNT,
        "no_silent_pruning": True,
        "child_order": "version-space-digest-ascending",
        "within_child_order": "verified-survivor-order",
        "all_nonempty_children_required": True,
        "exact_survivor_union_required": True,
        "no_orientation_preference": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorRankInput:
    version_space_digest: str
    version_space_algorithm_digest: str
    language_digest: str
    child_version_space_digests: tuple[str, ...]
    child_orientations: tuple[str, ...]
    survivor_candidate_digests: tuple[str, ...]
    candidate_origin_version_space_digests: tuple[str, ...]
    candidate_origin_orientations: tuple[str, ...]
    candidates: tuple[ObjectSceneAnchorRankCandidate, ...]
    survivor_count: int
    rank_input_digest: str

    def __post_init__(self) -> None:
        for label, item in (
            ("rank version-space digest", self.version_space_digest),
            ("rank algorithm digest", self.version_space_algorithm_digest),
            ("rank language digest", self.language_digest),
            ("rank input digest", self.rank_input_digest),
        ):
            _digest(item, label)
        if self.version_space_algorithm_digest != object_scene_anchor_version_space_algorithm_digest():
            raise ObjectSceneAnchorCandidateRankerError("rank algorithm binding differs")
        allowed_orientations = {item.value for item in ObjectSceneAnchorOrientation}
        if (
            type(self.child_version_space_digests) is not tuple
            or not 1 <= len(self.child_version_space_digests) <= 2
            or self.child_version_space_digests
            != tuple(sorted(set(self.child_version_space_digests)))
            or type(self.child_orientations) is not tuple
            or len(self.child_orientations) != len(self.child_version_space_digests)
            or any(item not in allowed_orientations for item in self.child_orientations)
            or len(set(self.child_orientations)) != len(self.child_orientations)
            or self.version_space_digest
            != _rank_scope_digest(
                self.child_version_space_digests,
                self.child_orientations,
                version_space_algorithm_digest=self.version_space_algorithm_digest,
                language_digest=self.language_digest,
            )
            or type(self.survivor_count) is not int
            or not 1 <= self.survivor_count <= MAX_SURVIVOR_COUNT
            or type(self.survivor_candidate_digests) is not tuple
            or len(self.survivor_candidate_digests) != self.survivor_count
            or len(set(self.survivor_candidate_digests)) != self.survivor_count
            or type(self.candidate_origin_version_space_digests) is not tuple
            or len(self.candidate_origin_version_space_digests) != self.survivor_count
            or type(self.candidate_origin_orientations) is not tuple
            or len(self.candidate_origin_orientations) != self.survivor_count
            or type(self.candidates) is not tuple
            or len(self.candidates) != self.survivor_count
            or any(type(item) is not ObjectSceneAnchorRankCandidate for item in self.candidates)
            or tuple(item.alias for item in self.candidates)
            != tuple(f"choice_{index:03d}" for index in range(self.survivor_count))
            or tuple(item.candidate_digest for item in self.candidates)
            != self.survivor_candidate_digests
        ):
            raise ObjectSceneAnchorCandidateRankerError(
                "rank input is not the complete bounded survivor inventory"
            )
        child_pairs = set(zip(self.child_version_space_digests, self.child_orientations))
        origin_pairs = tuple(
            zip(
                self.candidate_origin_version_space_digests,
                self.candidate_origin_orientations,
                strict=True,
            )
        )
        if set(origin_pairs) != child_pairs:
            raise ObjectSceneAnchorCandidateRankerError(
                "rank candidate origins differ from the nonempty child spaces"
            )
        for item in (
            *self.child_version_space_digests,
            *self.survivor_candidate_digests,
            *self.candidate_origin_version_space_digests,
        ):
            _digest(item, "rank survivor digest")
        if self.rank_input_digest != canonical_digest(_rank_input_content(self)):
            raise ObjectSceneAnchorCandidateRankerError("rank input digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_rank_input_content(self), "rank_input_digest": self.rank_input_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorRankInput":
        raw = _fields(
            value,
            {
                "schema", "version_space_digest", "version_space_algorithm_digest",
                "language_digest", "child_version_space_digests", "child_orientations",
                "survivor_candidate_digests", "candidate_origin_version_space_digests",
                "candidate_origin_orientations", "candidates", "survivor_count",
                "maximum_survivor_count", "no_silent_pruning", "child_order",
                "within_child_order", "all_nonempty_children_required",
                "exact_survivor_union_required", "no_orientation_preference",
                *_authority_data(), "rank_input_digest",
            },
            "rank input",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_RANK_INPUT_SCHEMA
            or raw["maximum_survivor_count"] != MAX_SURVIVOR_COUNT
            or raw["no_silent_pruning"] is not True
            or raw["child_order"] != "version-space-digest-ascending"
            or raw["within_child_order"] != "verified-survivor-order"
            or raw["all_nonempty_children_required"] is not True
            or raw["exact_survivor_union_required"] is not True
            or raw["no_orientation_preference"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["child_version_space_digests"], list)
            or not isinstance(raw["child_orientations"], list)
            or not isinstance(raw["survivor_candidate_digests"], list)
            or not isinstance(raw["candidate_origin_version_space_digests"], list)
            or not isinstance(raw["candidate_origin_orientations"], list)
            or not isinstance(raw["candidates"], list)
        ):
            raise ObjectSceneAnchorCandidateRankerError("rank input policy differs")
        result = cls(
            raw["version_space_digest"], raw["version_space_algorithm_digest"],
            raw["language_digest"], tuple(raw["child_version_space_digests"]),
            tuple(raw["child_orientations"]), tuple(raw["survivor_candidate_digests"]),
            tuple(raw["candidate_origin_version_space_digests"]),
            tuple(raw["candidate_origin_orientations"]),
            tuple(ObjectSceneAnchorRankCandidate.from_data(item) for item in raw["candidates"]),
            raw["survivor_count"], raw["rank_input_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorCandidateRankerError("rank input is not canonical")
        return result


def freeze_object_scene_anchor_rank_input(
    version_space: ObjectSceneAnchorSupportVersionSpace,
    additional_version_space: ObjectSceneAnchorSupportVersionSpace | None = None,
) -> ObjectSceneAnchorRankInput:
    """Freeze one survivor set or the exact union of two explicit orientations."""

    if type(version_space) is not ObjectSceneAnchorSupportVersionSpace:
        raise TypeError("version_space must be exact ObjectSceneAnchorSupportVersionSpace")
    if additional_version_space is not None and type(
        additional_version_space
    ) is not ObjectSceneAnchorSupportVersionSpace:
        raise TypeError(
            "additional_version_space must be exact ObjectSceneAnchorSupportVersionSpace"
        )
    versions = tuple(
        sorted(
            (
                ObjectSceneAnchorSupportVersionSpace.from_data(item.to_data())
                for item in (version_space, additional_version_space)
                if item is not None
            ),
            key=lambda item: item.version_space_digest,
        )
    )
    child_digests = tuple(item.version_space_digest for item in versions)
    child_orientations = tuple(item.orientation.value for item in versions)
    if len(set(child_digests)) != len(child_digests):
        raise ObjectSceneAnchorCandidateRankerError(
            "rank child version spaces must be distinct"
        )
    if len(versions) == 2 and len(set(child_orientations)) != 2:
        raise ObjectSceneAnchorCandidateRankerError(
            "two rank child spaces must have distinct explicit orientations"
        )
    if len({item.language.language_digest for item in versions}) != 1 or any(
        item.language != versions[0].language for item in versions[1:]
    ):
        raise ObjectSceneAnchorCandidateRankerError(
            "rank child spaces must share one exact predicate language"
        )
    if any(not item.survivor_candidate_digests for item in versions):
        raise ObjectSceneAnchorCandidateRankerError(
            "every rank child requires a nonempty verified survivor set"
        )
    survivor_digests = tuple(
        digest
        for version in versions
        for digest in version.survivor_candidate_digests
    )
    if len(set(survivor_digests)) != len(survivor_digests):
        raise ObjectSceneAnchorCandidateRankerError(
            "candidate digest occurs in more than one rank child space"
        )
    if len(survivor_digests) > MAX_SURVIVOR_COUNT:
        raise ObjectSceneAnchorRankCapacityGap(
            len(survivor_digests), MAX_SURVIVOR_COUNT, child_digests
        )
    presentations: list[ObjectSceneAnchorRankCandidate] = []
    origin_digests: list[str] = []
    origin_orientations: list[str] = []
    for version in versions:
        candidate_by_digest = {
            item.candidate_digest: item for item in version.candidates
        }
        atom_by_digest = {item.atom_digest: item for item in version.language.atoms}
        witness_by_digest = {
            item.witness_digest: item for item in version.language.vocabulary.entries
        }
        for digest in version.survivor_candidate_digests:
            try:
                candidate: ObjectSceneAnchorPredicateCandidate = candidate_by_digest[digest]
                atoms = tuple(atom_by_digest[item] for item in candidate.atom_digests)
                witnesses = tuple(
                    witness_by_digest[item] for item in candidate.witness_digests
                )
            except KeyError as exc:  # Defensive; the version-space class checks this.
                raise ObjectSceneAnchorCandidateRankerError(
                    "survivor projection is outside the frozen language"
                ) from exc
            anchor_kinds = {item.binding_spec.anchor_kind for item in atoms}
            if len(anchor_kinds) != 1 or any(
                item.binding_spec.spec_digest != candidate.binding_spec_digest
                for item in atoms
            ):
                raise ObjectSceneAnchorCandidateRankerError(
                    "survivor anchor projection differs from its candidate"
                )
            presentations.append(
                ObjectSceneAnchorRankCandidate.create(
                    alias=f"choice_{len(presentations):03d}",
                    candidate_digest=candidate.candidate_digest,
                    anchor_kind=next(iter(anchor_kinds)),
                    witness_digests=candidate.witness_digests,
                    affirmative_statements=tuple(
                        item.statement for item in witnesses
                    ),
                )
            )
            origin_digests.append(version.version_space_digest)
            origin_orientations.append(version.orientation.value)
    algorithm_digest = versions[0].algorithm_digest
    language_digest = versions[0].language.language_digest
    values = {
        "version_space_digest": _rank_scope_digest(
            child_digests,
            child_orientations,
            version_space_algorithm_digest=algorithm_digest,
            language_digest=language_digest,
        ),
        "version_space_algorithm_digest": algorithm_digest,
        "language_digest": language_digest,
        "child_version_space_digests": child_digests,
        "child_orientations": child_orientations,
        "survivor_candidate_digests": survivor_digests,
        "candidate_origin_version_space_digests": tuple(origin_digests),
        "candidate_origin_orientations": tuple(origin_orientations),
        "candidates": tuple(presentations),
        "survivor_count": len(presentations),
    }
    provisional = object.__new__(ObjectSceneAnchorRankInput)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorRankInput(
        **values, rank_input_digest=canonical_digest(_rank_input_content(provisional))
    )


def object_scene_anchor_candidate_ranker_prompt(
    rank_input: ObjectSceneAnchorRankInput,
) -> str:
    if type(rank_input) is not ObjectSceneAnchorRankInput:
        raise TypeError("rank_input must be exact ObjectSceneAnchorRankInput")
    frozen = ObjectSceneAnchorRankInput.from_data(rank_input.to_data())
    rows = "\n".join(
        f"- {item.alias}: anchor_kind={item.anchor_kind}; "
        f"affirmative_statements={json.dumps(list(item.affirmative_statements), ensure_ascii=False)}"
        for item in frozen.candidates
    )
    prompt = (
        "Rank every immutable verified choice by conceptual coherence, visual "
        "naturalness, parsimony, and resistance to accidental detail. Each "
        "choice is an indivisible bundle that already passed deterministic "
        "checks. Return every alias exactly once, best first. Do not add, "
        "remove, combine, split, or rewrite a bundle. Return no explanation.\n\n"
        f"immutable_choices:\n{rows}"
    )
    if len(prompt.encode("utf-8")) > MAX_PROMPT_UTF8_BYTES:
        raise ObjectSceneAnchorCandidateRankerError("rank prompt exceeds its byte guard")
    # The fixed instructions necessarily say "choice" but expose no formal or
    # experimental identifiers.  The candidate rows are restricted further by
    # ObjectSceneAnchorRankCandidate.
    if _FORBIDDEN_VISIBLE.search(prompt) is not None:
        raise ObjectSceneAnchorCandidateRankerError(
            "rank prompt crosses the sealed text-only boundary"
        )
    for item in frozen.candidates:
        if (
            item.candidate_digest in prompt
            or any(digest in prompt for digest in item.witness_digests)
        ):
            raise ObjectSceneAnchorCandidateRankerError(
                "rank prompt exposes a formal candidate identity"
            )
    return prompt


def object_scene_anchor_candidate_ranker_output_schema(
    rank_input: ObjectSceneAnchorRankInput,
) -> dict[str, object]:
    if type(rank_input) is not ObjectSceneAnchorRankInput:
        raise TypeError("rank_input must be exact ObjectSceneAnchorRankInput")
    aliases = [item.alias for item in rank_input.candidates]
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "ordered_aliases": {
                "type": "array",
                "items": {"type": "string", "enum": aliases},
                "description": "Every supplied immutable alias exactly once, best first.",
            }
        },
        "required": ["ordered_aliases"],
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(schema)
    return schema


def object_scene_anchor_candidate_ranker_protocol_digest() -> str:
    return canonical_digest(
        {
            "schema": OBJECT_SCENE_ANCHOR_CANDIDATE_RANKER_PROTOCOL_SCHEMA,
            "protocol_id": OBJECT_SCENE_ANCHOR_CANDIDATE_RANKER_PROTOCOL_ID,
            "source_digest": object_scene_anchor_candidate_ranker_source_digest(),
            "transport_source_digest": object_scene_anchor_candidate_ranker_transport_source_digest(),
            "transport_entrypoint": "run_codex_text_structured",
            "receipt_schema": CODEX_RECEIPT_SCHEMA,
            "input_digest_schema": TEXT_STRUCTURED_INPUT_DIGEST_SCHEMA,
            "isolation_policy": CODEX_ISOLATION_POLICY,
            "version_space_algorithm_digest": object_scene_anchor_version_space_algorithm_digest(),
            "rank_scope_rule": (
                "exact-union-of-one-or-two-nonempty-distinct-orientation-version-spaces"
            ),
            "child_order": "version-space-digest-ascending",
            "within_child_order": "verified-survivor-order",
            "duplicate_candidate_digests_across_children_rejected": True,
            "orientation_not_model_visible": True,
            "output_rule": "exact-permutation-of-every-verified-survivor-alias",
            "maximum_survivor_count": MAX_SURVIVOR_COUNT,
            "maximum_prompt_utf8_bytes": MAX_PROMPT_UTF8_BYTES,
            "model_visible_candidate_fields": [
                "alias", "anchor_kind", "affirmative_statements"
            ],
            **_authority_data(),
        }
    )


def object_scene_anchor_candidate_ranker_model_digest(
    model: str, reasoning_effort: str
) -> str:
    if not isinstance(model, str) or _MODEL.fullmatch(model) is None:
        raise ObjectSceneAnchorCandidateRankerError("ranker model differs")
    if reasoning_effort not in REASONING_EFFORTS:
        raise ObjectSceneAnchorCandidateRankerError("ranker reasoning effort differs")
    return canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-candidate-ranker-model.v1",
            "model": model,
            "reasoning_effort": reasoning_effort,
        }
    )


def _runtime_digest_from_pins(
    *,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str,
    cloud_policy_cache_binding: str,
    model_catalog_digest: str,
    no_tools_attestation_digest: str,
) -> str:
    model_digest = object_scene_anchor_candidate_ranker_model_digest(
        model, reasoning_effort
    )
    launcher = _digest(expected_launcher_digest, "ranker launcher digest")
    catalog = _digest(model_catalog_digest, "ranker model catalog digest")
    attestation = _digest(no_tools_attestation_digest, "ranker no-tools digest")
    if cloud_policy_cache_binding != "absent":
        _address(cloud_policy_cache_binding, "ranker policy-cache binding")
    return canonical_digest(
        {
            "schema": "gkm.object-scene-anchor-candidate-ranker-runtime.v1",
            "model_digest": model_digest,
            "expected_launcher_digest": launcher,
            "cloud_policy_cache_binding": cloud_policy_cache_binding,
            "model_catalog_digest": catalog,
            "no_tools_attestation_digest": attestation,
            "protocol_digest": object_scene_anchor_candidate_ranker_protocol_digest(),
            "source_digest": object_scene_anchor_candidate_ranker_source_digest(),
            "transport_source_digest": object_scene_anchor_candidate_ranker_transport_source_digest(),
            **_authority_data(),
        }
    )


def object_scene_anchor_candidate_ranker_runtime_digest(
    *,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
) -> str:
    if not isinstance(cloud_policy_cache_snapshot, CloudPolicyCacheSnapshot):
        raise ObjectSceneAnchorCandidateRankerError(
            "an exact cloud policy-cache snapshot is required"
        )
    if not isinstance(model_catalog_snapshot, CodexModelCatalogSnapshot):
        raise ObjectSceneAnchorCandidateRankerError(
            "an exact model catalog snapshot is required"
        )
    launcher = _digest(expected_launcher_digest, "ranker launcher digest")
    try:
        attestation = validate_codex_no_tools_attestation(
            no_tools_attestation,
            expected_launcher_digest=launcher,
            expected_model_catalog_digest=model_catalog_snapshot.raw_digest,
            expected_cloud_policy_cache_binding=cloud_policy_cache_snapshot.binding,
        )
    except (CodexProposerFailure, TypeError, ValueError) as exc:
        raise ObjectSceneAnchorCandidateRankerError(
            "ranker no-tools runtime differs from its attestation"
        ) from exc
    return _runtime_digest_from_pins(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=launcher,
        cloud_policy_cache_binding=cloud_policy_cache_snapshot.binding,
        model_catalog_digest=model_catalog_snapshot.raw_digest,
        no_tools_attestation_digest=attestation.attestation_digest,
    )


def _parse_alias_payload(
    payload: Mapping[str, Any], rank_input: ObjectSceneAnchorRankInput
) -> tuple[str, ...]:
    raw = _fields(payload, {"ordered_aliases"}, "rank payload")
    values = raw["ordered_aliases"]
    if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
        raise ObjectSceneAnchorCandidateRankerError("ordered aliases must be a JSON list")
    ordered = tuple(values)
    aliases = tuple(item.alias for item in rank_input.candidates)
    if (
        len(ordered) != len(aliases)
        or len(set(ordered)) != len(ordered)
        or set(ordered) != set(aliases)
    ):
        raise ObjectSceneAnchorCandidateRankerError(
            "rank payload must be the exact survivor-alias permutation"
        )
    by_alias = {
        item.alias: item.candidate_digest for item in rank_input.candidates
    }
    return tuple(by_alias[item] for item in ordered)


def _alias_payload(
    ordered_candidate_digests: Sequence[str], rank_input: ObjectSceneAnchorRankInput
) -> dict[str, object]:
    by_digest = {
        item.candidate_digest: item.alias for item in rank_input.candidates
    }
    try:
        aliases = [by_digest[item] for item in ordered_candidate_digests]
    except KeyError as exc:
        raise ObjectSceneAnchorCandidateRankerError(
            "rank output contains a foreign candidate"
        ) from exc
    return {"ordered_aliases": aliases}


def _output_digest(ordered_candidate_digests: Sequence[str]) -> str:
    return canonical_digest(
        {
            "schema": OBJECT_SCENE_ANCHOR_RANK_OUTPUT_SCHEMA,
            "ordered_candidate_digests": list(ordered_candidate_digests),
        }
    )


def _candidate_origin(
    rank_input: ObjectSceneAnchorRankInput, candidate_digest: str
) -> tuple[str, str]:
    try:
        index = rank_input.survivor_candidate_digests.index(candidate_digest)
    except ValueError as exc:
        raise ObjectSceneAnchorCandidateRankerError(
            "selected candidate is outside the exact rank scope"
        ) from exc
    return (
        rank_input.candidate_origin_version_space_digests[index],
        rank_input.candidate_origin_orientations[index],
    )


def _response_content(value: "ObjectSceneAnchorRankResponse") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_ANCHOR_RANK_RESPONSE_SCHEMA,
        "rank_input": value.rank_input.to_data(),
        "rank_input_digest": value.rank_input_digest,
        "version_space_digest": value.version_space_digest,
        "child_version_space_digests": list(value.child_version_space_digests),
        "child_orientations": list(value.child_orientations),
        "ordered_candidate_digests": list(value.ordered_candidate_digests),
        "selected_candidate_digest": value.selected_candidate_digest,
        "selected_origin_version_space_digest": (
            value.selected_origin_version_space_digest
        ),
        "selected_origin_orientation": value.selected_origin_orientation,
        "output_digest": value.output_digest,
        "model_payload": dict(value.model_payload),
        "prompt_digest": value.prompt_digest,
        "output_schema_digest": value.output_schema_digest,
        "protocol_id": value.protocol_id,
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
        "runtime_digest": value.runtime_digest,
        "physical_call_count": value.physical_call_count,
        "receipt": dict(value.receipt),
        "receipt_digest": value.receipt_digest,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorRankResponse:
    """Receipt-attested exact survivor permutation and first-ranked selection."""

    rank_input: ObjectSceneAnchorRankInput
    rank_input_digest: str
    version_space_digest: str
    child_version_space_digests: tuple[str, ...]
    child_orientations: tuple[str, ...]
    ordered_candidate_digests: tuple[str, ...]
    selected_candidate_digest: str
    selected_origin_version_space_digest: str
    selected_origin_orientation: str
    output_digest: str
    model_payload: Mapping[str, Any]
    prompt_digest: str
    output_schema_digest: str
    protocol_id: str
    protocol_digest: str
    source_digest: str
    transport_source_digest: str
    model: str
    reasoning_effort: str
    model_digest: str
    expected_launcher_digest: str
    cloud_policy_cache_binding: str
    model_catalog_digest: str
    no_tools_attestation_digest: str
    runtime_digest: str
    physical_call_count: int
    receipt: Mapping[str, Any]
    receipt_digest: str
    response_digest: str

    def __post_init__(self) -> None:
        if type(self.rank_input) is not ObjectSceneAnchorRankInput:
            raise TypeError("rank response input has the wrong type")
        for label, item in (
            ("rank input digest", self.rank_input_digest),
            ("version-space digest", self.version_space_digest),
            ("selected origin version-space digest", self.selected_origin_version_space_digest),
            ("rank output digest", self.output_digest),
            ("prompt digest", self.prompt_digest),
            ("output schema digest", self.output_schema_digest),
            ("ranker protocol digest", self.protocol_digest),
            ("ranker source digest", self.source_digest),
            ("ranker transport source digest", self.transport_source_digest),
            ("ranker model digest", self.model_digest),
            ("ranker launcher digest", self.expected_launcher_digest),
            ("ranker model catalog digest", self.model_catalog_digest),
            ("ranker no-tools digest", self.no_tools_attestation_digest),
            ("ranker runtime digest", self.runtime_digest),
            ("ranker receipt digest", self.receipt_digest),
            ("rank response digest", self.response_digest),
        ):
            _digest(item, label)
        if self.cloud_policy_cache_binding != "absent":
            _address(self.cloud_policy_cache_binding, "ranker policy-cache binding")
        survivors = self.rank_input.survivor_candidate_digests
        selected_origin = _candidate_origin(
            self.rank_input, self.selected_candidate_digest
        )
        if (
            self.rank_input.rank_input_digest != self.rank_input_digest
            or self.rank_input.version_space_digest != self.version_space_digest
            or self.child_version_space_digests
            != self.rank_input.child_version_space_digests
            or self.child_orientations != self.rank_input.child_orientations
            or type(self.ordered_candidate_digests) is not tuple
            or len(self.ordered_candidate_digests) != len(survivors)
            or len(set(self.ordered_candidate_digests)) != len(survivors)
            or set(self.ordered_candidate_digests) != set(survivors)
            or self.selected_candidate_digest != self.ordered_candidate_digests[0]
            or (
                self.selected_origin_version_space_digest,
                self.selected_origin_orientation,
            )
            != selected_origin
            or self.output_digest != _output_digest(self.ordered_candidate_digests)
            or self.physical_call_count != 1
        ):
            raise ObjectSceneAnchorCandidateRankerError(
                "rank response is not the exact complete survivor permutation"
            )
        for item in self.ordered_candidate_digests:
            _digest(item, "ordered candidate digest")
        payload = _canonical_payload(self.model_payload)
        object.__setattr__(self, "model_payload", payload)
        if _parse_alias_payload(payload, self.rank_input) != self.ordered_candidate_digests:
            raise ObjectSceneAnchorCandidateRankerError(
                "rank payload differs from resolved output"
            )
        prompt = object_scene_anchor_candidate_ranker_prompt(self.rank_input)
        schema = object_scene_anchor_candidate_ranker_output_schema(self.rank_input)
        if (
            self.prompt_digest != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or self.output_schema_digest != canonical_digest(schema)
            or self.protocol_id != OBJECT_SCENE_ANCHOR_CANDIDATE_RANKER_PROTOCOL_ID
            or self.protocol_digest != object_scene_anchor_candidate_ranker_protocol_digest()
            or self.source_digest != object_scene_anchor_candidate_ranker_source_digest()
            or self.transport_source_digest
            != object_scene_anchor_candidate_ranker_transport_source_digest()
            or self.model_digest
            != object_scene_anchor_candidate_ranker_model_digest(
                self.model, self.reasoning_effort
            )
            or self.runtime_digest
            != _runtime_digest_from_pins(
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                expected_launcher_digest=self.expected_launcher_digest,
                cloud_policy_cache_binding=self.cloud_policy_cache_binding,
                model_catalog_digest=self.model_catalog_digest,
                no_tools_attestation_digest=self.no_tools_attestation_digest,
            )
        ):
            raise ObjectSceneAnchorCandidateRankerError(
                "rank response protocol, prompt, model, or runtime differs"
            )
        receipt = _canonical_payload(self.receipt)
        object.__setattr__(self, "receipt", receipt)
        try:
            validate_codex_text_receipt(receipt, prompt, schema)
        except (CodexProposerFailure, TypeError, ValueError) as exc:
            raise ObjectSceneAnchorCandidateRankerError(
                "rank receipt does not bind the frozen text input"
            ) from exc
        if (
            receipt.get("requested_model") != self.model
            or receipt.get("requested_reasoning_effort") != self.reasoning_effort
            or receipt.get("codex_launcher_digest") != self.expected_launcher_digest
            or receipt.get("cloud_config_bundle_cache_binding")
            != self.cloud_policy_cache_binding
            or receipt.get("model_catalog_digest") != self.model_catalog_digest
            or receipt.get("tool_surface_attestation_digest")
            != self.no_tools_attestation_digest
            or receipt.get("structured_output_digest") != canonical_digest(payload)
            or self.receipt_digest
            != canonical_digest(
                {
                    "schema": OBJECT_SCENE_ANCHOR_RANK_RECEIPT_SCHEMA,
                    "receipt": receipt,
                }
            )
        ):
            raise ObjectSceneAnchorCandidateRankerError(
                "rank receipt model, runtime, payload, or digest differs"
            )
        if self.response_digest != canonical_digest(_response_content(self)):
            raise ObjectSceneAnchorCandidateRankerError("rank response digest differs")

    @classmethod
    def seal(
        cls,
        *,
        rank_input: ObjectSceneAnchorRankInput,
        ordered_candidate_digests: Sequence[str],
        model_payload: Mapping[str, Any],
        model: str,
        reasoning_effort: str,
        expected_launcher_digest: str,
        cloud_policy_cache_binding: str,
        model_catalog_digest: str,
        no_tools_attestation_digest: str,
        receipt: Mapping[str, Any],
    ) -> "ObjectSceneAnchorRankResponse":
        ordered = tuple(ordered_candidate_digests)
        if not ordered:
            raise ObjectSceneAnchorCandidateRankerError("rank response cannot be empty")
        prompt = object_scene_anchor_candidate_ranker_prompt(rank_input)
        schema = object_scene_anchor_candidate_ranker_output_schema(rank_input)
        canonical_receipt = _canonical_payload(receipt)
        values = {
            "rank_input": rank_input,
            "rank_input_digest": rank_input.rank_input_digest,
            "version_space_digest": rank_input.version_space_digest,
            "child_version_space_digests": rank_input.child_version_space_digests,
            "child_orientations": rank_input.child_orientations,
            "ordered_candidate_digests": ordered,
            "selected_candidate_digest": ordered[0],
            "selected_origin_version_space_digest": _candidate_origin(
                rank_input, ordered[0]
            )[0],
            "selected_origin_orientation": _candidate_origin(rank_input, ordered[0])[1],
            "output_digest": _output_digest(ordered),
            "model_payload": _canonical_payload(model_payload),
            "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "output_schema_digest": canonical_digest(schema),
            "protocol_id": OBJECT_SCENE_ANCHOR_CANDIDATE_RANKER_PROTOCOL_ID,
            "protocol_digest": object_scene_anchor_candidate_ranker_protocol_digest(),
            "source_digest": object_scene_anchor_candidate_ranker_source_digest(),
            "transport_source_digest": object_scene_anchor_candidate_ranker_transport_source_digest(),
            "model": model,
            "reasoning_effort": reasoning_effort,
            "model_digest": object_scene_anchor_candidate_ranker_model_digest(
                model, reasoning_effort
            ),
            "expected_launcher_digest": expected_launcher_digest,
            "cloud_policy_cache_binding": cloud_policy_cache_binding,
            "model_catalog_digest": model_catalog_digest,
            "no_tools_attestation_digest": no_tools_attestation_digest,
            "runtime_digest": _runtime_digest_from_pins(
                model=model,
                reasoning_effort=reasoning_effort,
                expected_launcher_digest=expected_launcher_digest,
                cloud_policy_cache_binding=cloud_policy_cache_binding,
                model_catalog_digest=model_catalog_digest,
                no_tools_attestation_digest=no_tools_attestation_digest,
            ),
            "physical_call_count": 1,
            "receipt": canonical_receipt,
            "receipt_digest": canonical_digest(
                {
                    "schema": OBJECT_SCENE_ANCHOR_RANK_RECEIPT_SCHEMA,
                    "receipt": canonical_receipt,
                }
            ),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            response_digest=canonical_digest(_response_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_response_content(self), "response_digest": self.response_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorRankResponse":
        raw = _fields(
            value,
            {
                "schema", "rank_input", "rank_input_digest", "version_space_digest",
                "child_version_space_digests", "child_orientations",
                "ordered_candidate_digests", "selected_candidate_digest",
                "selected_origin_version_space_digest", "selected_origin_orientation",
                "output_digest",
                "model_payload", "prompt_digest", "output_schema_digest", "protocol_id",
                "protocol_digest", "source_digest", "transport_source_digest", "model",
                "reasoning_effort", "model_digest", "expected_launcher_digest",
                "cloud_policy_cache_binding", "model_catalog_digest",
                "no_tools_attestation_digest", "runtime_digest", "physical_call_count",
                "receipt", "receipt_digest", *_authority_data(), "response_digest",
            },
            "rank response",
        )
        if (
            raw["schema"] != OBJECT_SCENE_ANCHOR_RANK_RESPONSE_SCHEMA
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["rank_input"], Mapping)
            or not isinstance(raw["child_version_space_digests"], list)
            or not isinstance(raw["child_orientations"], list)
            or not isinstance(raw["ordered_candidate_digests"], list)
            or not isinstance(raw["model_payload"], Mapping)
            or not isinstance(raw["receipt"], Mapping)
        ):
            raise ObjectSceneAnchorCandidateRankerError("rank response policy differs")
        result = cls(
            rank_input=ObjectSceneAnchorRankInput.from_data(raw["rank_input"]),
            rank_input_digest=raw["rank_input_digest"],
            version_space_digest=raw["version_space_digest"],
            child_version_space_digests=tuple(raw["child_version_space_digests"]),
            child_orientations=tuple(raw["child_orientations"]),
            ordered_candidate_digests=tuple(raw["ordered_candidate_digests"]),
            selected_candidate_digest=raw["selected_candidate_digest"],
            selected_origin_version_space_digest=raw[
                "selected_origin_version_space_digest"
            ],
            selected_origin_orientation=raw["selected_origin_orientation"],
            output_digest=raw["output_digest"],
            model_payload=dict(raw["model_payload"]), prompt_digest=raw["prompt_digest"],
            output_schema_digest=raw["output_schema_digest"], protocol_id=raw["protocol_id"],
            protocol_digest=raw["protocol_digest"], source_digest=raw["source_digest"],
            transport_source_digest=raw["transport_source_digest"], model=raw["model"],
            reasoning_effort=raw["reasoning_effort"], model_digest=raw["model_digest"],
            expected_launcher_digest=raw["expected_launcher_digest"],
            cloud_policy_cache_binding=raw["cloud_policy_cache_binding"],
            model_catalog_digest=raw["model_catalog_digest"],
            no_tools_attestation_digest=raw["no_tools_attestation_digest"],
            runtime_digest=raw["runtime_digest"], physical_call_count=raw["physical_call_count"],
            receipt=dict(raw["receipt"]), receipt_digest=raw["receipt_digest"],
            response_digest=raw["response_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorCandidateRankerError("rank response is not canonical")
        return result


def verify_object_scene_anchor_rank_response(
    response: ObjectSceneAnchorRankResponse,
    *,
    version_space: ObjectSceneAnchorSupportVersionSpace,
    additional_version_space: ObjectSceneAnchorSupportVersionSpace | None = None,
    expected_response_digest: str,
    expected_rank_input_digest: str,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
) -> ObjectSceneAnchorRankResponse:
    """Cold replay all bindings and the transport receipt without a model call."""

    if type(response) is not ObjectSceneAnchorRankResponse:
        raise TypeError("response must be exact ObjectSceneAnchorRankResponse")
    restored = ObjectSceneAnchorRankResponse.from_data(response.to_data())
    if restored.response_digest != _digest(
        expected_response_digest, "expected rank response digest"
    ):
        raise ObjectSceneAnchorCandidateRankerError(
            "rank response differs from external commitment"
        )
    expected_input = freeze_object_scene_anchor_rank_input(
        version_space, additional_version_space
    )
    if (
        expected_input.rank_input_digest
        != _digest(expected_rank_input_digest, "expected rank input digest")
        or restored.rank_input != expected_input
    ):
        raise ObjectSceneAnchorCandidateRankerError(
            "rank input differs from the exact current survivor set"
        )
    expected_runtime = object_scene_anchor_candidate_ranker_runtime_digest(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    if (
        restored.model != model
        or restored.reasoning_effort != reasoning_effort
        or restored.expected_launcher_digest != expected_launcher_digest
        or restored.cloud_policy_cache_binding != cloud_policy_cache_snapshot.binding
        or restored.model_catalog_digest != model_catalog_snapshot.raw_digest
        or restored.no_tools_attestation_digest
        != no_tools_attestation.attestation_digest
        or restored.runtime_digest != expected_runtime
    ):
        raise ObjectSceneAnchorCandidateRankerError(
            "rank response runtime differs from the frozen environment"
        )
    return restored


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorCandidateRanker:
    """Configured one-call, receipt-attested, text-only survivor ranker."""

    model: str
    expected_launcher_digest: str
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot
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
        object_scene_anchor_candidate_ranker_model_digest(
            self.model, self.reasoning_effort
        )
        _digest(self.expected_launcher_digest, "ranker launcher digest")
        object_scene_anchor_candidate_ranker_runtime_digest(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            expected_launcher_digest=self.expected_launcher_digest,
            cloud_policy_cache_snapshot=self.cloud_policy_cache_snapshot,
            model_catalog_snapshot=self.model_catalog_snapshot,
            no_tools_attestation=self.no_tools_attestation,
        )
        if type(self.minutes) is not int or not 1 <= self.minutes <= 120:
            raise ObjectSceneAnchorCandidateRankerError(
                "ranker timeout minutes must lie in 1..120"
            )
        if type(self.verbose) is not bool:
            raise TypeError("verbose must be bool")
        if not isinstance(self.executable, str) or not self.executable:
            raise ObjectSceneAnchorCandidateRankerError(
                "ranker executable must be nonempty"
            )
        if not callable(self.transport):
            raise TypeError("ranker transport must be callable")

    @property
    def runtime_digest(self) -> str:
        return object_scene_anchor_candidate_ranker_runtime_digest(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            expected_launcher_digest=self.expected_launcher_digest,
            cloud_policy_cache_snapshot=self.cloud_policy_cache_snapshot,
            model_catalog_snapshot=self.model_catalog_snapshot,
            no_tools_attestation=self.no_tools_attestation,
        )

    def __call__(
        self,
        version_space: ObjectSceneAnchorSupportVersionSpace,
        additional_version_space: ObjectSceneAnchorSupportVersionSpace | None = None,
        *,
        expected_rank_input_digest: str,
    ) -> ObjectSceneAnchorRankResponse:
        rank_input = freeze_object_scene_anchor_rank_input(
            version_space, additional_version_space
        )
        if rank_input.rank_input_digest != _digest(
            expected_rank_input_digest, "expected rank input digest"
        ):
            raise ObjectSceneAnchorCandidateRankerError(
                "rank input differs from external commitment"
            )
        prompt = object_scene_anchor_candidate_ranker_prompt(rank_input)
        schema = object_scene_anchor_candidate_ranker_output_schema(rank_input)
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
            raise ObjectSceneAnchorCandidateRankerError(
                "text-only candidate rank transport failed; no candidate was selected"
            ) from exc
        if not isinstance(result, CodexStructuredResult):
            raise ObjectSceneAnchorCandidateRankerError(
                "candidate rank transport returned the wrong result; no candidate was selected"
            )
        try:
            payload = _canonical_payload(result.payload)
            ordered = _parse_alias_payload(payload, rank_input)
        except Exception as exc:
            raise ObjectSceneAnchorCandidateRankerError(
                "candidate rank payload was rejected; no candidate was selected"
            ) from exc
        if not isinstance(result.receipt, CodexReceipt):
            raise ObjectSceneAnchorCandidateRankerError(
                "candidate rank transport returned no receipt; no candidate was selected"
            )
        response = ObjectSceneAnchorRankResponse.seal(
            rank_input=rank_input,
            ordered_candidate_digests=ordered,
            model_payload=payload,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            expected_launcher_digest=self.expected_launcher_digest,
            cloud_policy_cache_binding=self.cloud_policy_cache_snapshot.binding,
            model_catalog_digest=self.model_catalog_snapshot.raw_digest,
            no_tools_attestation_digest=self.no_tools_attestation.attestation_digest,
            receipt=result.receipt.to_dict(),
        )
        if response.runtime_digest != self.runtime_digest:
            raise ObjectSceneAnchorCandidateRankerError(
                "candidate rank response runtime differs; no candidate was selected"
            )
        return response

    def verify_response(
        self,
        response: ObjectSceneAnchorRankResponse,
        *,
        version_space: ObjectSceneAnchorSupportVersionSpace,
        additional_version_space: ObjectSceneAnchorSupportVersionSpace | None = None,
        expected_response_digest: str,
        expected_rank_input_digest: str,
    ) -> ObjectSceneAnchorRankResponse:
        return verify_object_scene_anchor_rank_response(
            response,
            version_space=version_space,
            additional_version_space=additional_version_space,
            expected_response_digest=expected_response_digest,
            expected_rank_input_digest=expected_rank_input_digest,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            expected_launcher_digest=self.expected_launcher_digest,
            cloud_policy_cache_snapshot=self.cloud_policy_cache_snapshot,
            model_catalog_snapshot=self.model_catalog_snapshot,
            no_tools_attestation=self.no_tools_attestation,
        )


__all__ = (
    "MAX_PROMPT_UTF8_BYTES",
    "MAX_SURVIVOR_COUNT",
    "OBJECT_SCENE_ANCHOR_CANDIDATE_RANKER_PROTOCOL_ID",
    "OBJECT_SCENE_ANCHOR_CANDIDATE_RANKER_PROTOCOL_SCHEMA",
    "OBJECT_SCENE_ANCHOR_RANK_CANDIDATE_SCHEMA",
    "OBJECT_SCENE_ANCHOR_RANK_INPUT_SCHEMA",
    "OBJECT_SCENE_ANCHOR_RANK_OUTPUT_SCHEMA",
    "OBJECT_SCENE_ANCHOR_RANK_RESPONSE_SCHEMA",
    "ObjectSceneAnchorCandidateRanker",
    "ObjectSceneAnchorCandidateRankerError",
    "ObjectSceneAnchorRankCapacityGap",
    "ObjectSceneAnchorRankCandidate",
    "ObjectSceneAnchorRankInput",
    "ObjectSceneAnchorRankResponse",
    "TextStructuredTransport",
    "freeze_object_scene_anchor_rank_input",
    "object_scene_anchor_candidate_ranker_model_digest",
    "object_scene_anchor_candidate_ranker_output_schema",
    "object_scene_anchor_candidate_ranker_prompt",
    "object_scene_anchor_candidate_ranker_protocol_digest",
    "object_scene_anchor_candidate_ranker_runtime_digest",
    "object_scene_anchor_candidate_ranker_source_digest",
    "object_scene_anchor_candidate_ranker_transport_source_digest",
    "verify_object_scene_anchor_rank_response",
)

"""Text-only Codex ranking over a frozen object-profile version space.

The ranker receives only already-verified positive ``AT_LEAST`` profiles and
neutral semantic summaries.  Codex returns a preference order over bounded
aliases; Python checks that it is the exact survivor permutation and resolves
the aliases back to immutable profile digests.  Formulas, thresholds, units,
and polarity are never model-editable.  No image bytes or held-out material
enter this protocol.
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
from bongard.prototype_object_profiles import (
    OBJECT_FEATURE_CATALOG,
    OBJECT_FEATURE_CATALOG_DIGEST,
    OBJECT_FEATURE_IDS,
    ObjectProfile,
    ObjectProfileOperator,
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


OBJECT_BONGARD_CODEX_RANKER_PROTOCOL_ID = (
    "bongard.object-version-space/text-only-codex-ranker-v1"
)
OBJECT_BONGARD_CODEX_RANKER_PROTOCOL_SCHEMA = (
    "gkm.bongard-object-codex-ranker-protocol.v1"
)
OBJECT_BONGARD_RANK_INPUT_SCHEMA = "gkm.bongard-object-rank-input.v1"
OBJECT_BONGARD_RANK_OUTPUT_SCHEMA = "gkm.bongard-object-rank-output.v1"
OBJECT_BONGARD_RANK_RESPONSE_SCHEMA = "gkm.bongard-object-rank-response.v1"
MAX_SURVIVOR_COUNT = 1_740
MAX_PROMPT_UTF8_BYTES = 2_000_000
MAX_RUBRIC_UTF8_BYTES = 768

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}\Z")
_PROSE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ,.'-]{0,767}\Z")
_FORBIDDEN_VISIBLE_WORD = re.compile(
    r"\b(?:pixel|label|query|positive|negative)s?\b", re.IGNORECASE
)
_FEATURE_BY_ID = {item.feature_id: item for item in OBJECT_FEATURE_CATALOG}


class ObjectBongardCodexRankerError(RuntimeError):
    """A rank input, transport, payload, receipt, or replay pin is invalid."""


TextStructuredTransport = Callable[..., CodexStructuredResult]


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "codex_may_rank_verified_survivors_only": True,
        "codex_may_edit_formulas": False,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_ranking_or_replay": False,
    }


def object_bongard_codex_ranker_authority_data() -> dict[str, object]:
    return _authority_data()


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectBongardCodexRankerError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ObjectBongardCodexRankerError(f"{label} must be a bounded identifier")
    return value


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectBongardCodexRankerError(f"{label} fields differ from schema")
    return value


def object_bongard_codex_ranker_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def object_bongard_codex_ranker_transport_source_digest() -> str:
    source = getattr(_transport_module, "__file__", None)
    if not isinstance(source, str) or not source:
        raise ObjectBongardCodexRankerError("text transport source is unavailable")
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def _freeze_survivors(values: Sequence[ObjectProfile]) -> tuple[ObjectProfile, ...]:
    if isinstance(values, (str, bytes)):
        raise ObjectBongardCodexRankerError("survivors must be a finite sequence")
    survivors = tuple(values)
    if not 1 <= len(survivors) <= MAX_SURVIVOR_COUNT:
        raise ObjectBongardCodexRankerError(
            f"survivor count must lie in 1..{MAX_SURVIVOR_COUNT}"
        )
    for profile in survivors:
        if not isinstance(profile, ObjectProfile):
            raise TypeError("every survivor must be ObjectProfile")
        try:
            if ObjectProfile.from_data(profile.to_data()) != profile:
                raise ObjectBongardCodexRankerError("survivor is not canonical")
        except (TypeError, ValueError) as exc:
            raise ObjectBongardCodexRankerError("survivor is not canonical") from exc
        if (
            not 1 <= len(profile.atoms) <= 2
            or any(atom.operator is not ObjectProfileOperator.AT_LEAST for atom in profile.atoms)
        ):
            raise ObjectBongardCodexRankerError(
                "ranker admits only one- or two-atom positive AT_LEAST profiles"
            )
    if (
        len({item.profile_digest for item in survivors}) != len(survivors)
        or len({item.profile_id for item in survivors}) != len(survivors)
        or len({item.atoms for item in survivors}) != len(survivors)
    ):
        raise ObjectBongardCodexRankerError(
            "survivor identities and formulas must be unique"
        )
    return survivors


def _rubric(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value
        or len(value.encode("utf-8")) > MAX_RUBRIC_UTF8_BYTES
        or _PROSE.fullmatch(value) is None
        or _FORBIDDEN_VISIBLE_WORD.search(value) is not None
    ):
        raise ObjectBongardCodexRankerError(
            f"{label} violates the neutral text-only policy"
        )
    return value


def _freeze_semantics(
    neutral_rubrics: Sequence[str],
    feature_nominations: Sequence[Sequence[str]],
) -> tuple[tuple[str, str], tuple[tuple[str, ...], tuple[str, ...]]]:
    if isinstance(neutral_rubrics, (str, bytes)):
        raise ObjectBongardCodexRankerError("neutral rubrics must be a sequence")
    rubrics = tuple(
        _rubric(value, f"neutral rubric {index}")
        for index, value in enumerate(neutral_rubrics)
    )
    if len(rubrics) != 2:
        raise ObjectBongardCodexRankerError("exactly two neutral rubrics are required")
    if isinstance(feature_nominations, (str, bytes)):
        raise ObjectBongardCodexRankerError("feature nominations must be a sequence")
    families = tuple(tuple(group) for group in feature_nominations)
    if len(families) != 2:
        raise ObjectBongardCodexRankerError(
            "exactly two neutral feature-nomination groups are required"
        )
    for group in families:
        if (
            not group
            or any(not isinstance(item, str) or item not in OBJECT_FEATURE_IDS for item in group)
            or group != tuple(sorted(set(group), key=OBJECT_FEATURE_IDS.index))
        ):
            raise ObjectBongardCodexRankerError(
                "feature nominations must be unique and in catalog order"
            )
    return rubrics, families  # type: ignore[return-value]


def object_bongard_rank_input_digest(
    *,
    survivors: Sequence[ObjectProfile],
    neutral_rubrics: Sequence[str],
    feature_nominations: Sequence[Sequence[str]],
    semantic_artifact_digest: str,
    version_space_digest: str,
) -> str:
    frozen = _freeze_survivors(survivors)
    rubrics, families = _freeze_semantics(neutral_rubrics, feature_nominations)
    return canonical_digest(
        {
            "schema": OBJECT_BONGARD_RANK_INPUT_SCHEMA,
            "ordered_survivors": [item.to_data() for item in frozen],
            "neutral_rubrics": list(rubrics),
            "feature_nominations": [list(group) for group in families],
            "semantic_artifact_digest": _digest(
                semantic_artifact_digest, "semantic artifact digest"
            ),
            "version_space_digest": _digest(
                version_space_digest, "version-space digest"
            ),
            "image_material_included": False,
            "side_names_included": False,
            "held_out_material_included": False,
            **_authority_data(),
        }
    )


def _rank_inputs(
    *,
    survivors: Sequence[ObjectProfile],
    neutral_rubrics: Sequence[str],
    feature_nominations: Sequence[Sequence[str]],
    semantic_artifact_digest: str,
    version_space_digest: str,
    rank_input_digest: str,
) -> tuple[
    tuple[ObjectProfile, ...],
    tuple[str, str],
    tuple[tuple[str, ...], tuple[str, ...]],
]:
    frozen = _freeze_survivors(survivors)
    rubrics, families = _freeze_semantics(neutral_rubrics, feature_nominations)
    semantic = _digest(semantic_artifact_digest, "semantic artifact digest")
    version = _digest(version_space_digest, "version-space digest")
    supplied = _digest(rank_input_digest, "rank input digest")
    expected = object_bongard_rank_input_digest(
        survivors=frozen,
        neutral_rubrics=rubrics,
        feature_nominations=families,
        semantic_artifact_digest=semantic,
        version_space_digest=version,
    )
    if supplied != expected:
        raise ObjectBongardCodexRankerError(
            "rank input digest differs from its canonical preimage"
        )
    return frozen, rubrics, families


def _aliases(count: int) -> tuple[str, ...]:
    return tuple(f"c{index:04d}" for index in range(count))


def _atom_text(profile: ObjectProfile) -> str:
    parts: list[str] = []
    for atom in profile.atoms:
        spec = _FEATURE_BY_ID[atom.feature_id]
        parts.append(
            f"{atom.feature_id} AT_LEAST {atom.target} {spec.unit} "
            f"({spec.operational_description})"
        )
    return " AND, on the same object lineage, ".join(parts)


def object_bongard_codex_ranker_prompt(
    *,
    survivors: Sequence[ObjectProfile],
    neutral_rubrics: Sequence[str],
    feature_nominations: Sequence[Sequence[str]],
    semantic_artifact_digest: str,
    version_space_digest: str,
    rank_input_digest: str,
) -> str:
    frozen, rubrics, families = _rank_inputs(
        survivors=survivors,
        neutral_rubrics=neutral_rubrics,
        feature_nominations=feature_nominations,
        semantic_artifact_digest=semantic_artifact_digest,
        version_space_digest=version_space_digest,
        rank_input_digest=rank_input_digest,
    )
    rows = "\n".join(
        f"- {alias}; digest={profile.profile_digest}; formula={_atom_text(profile)}"
        for alias, profile in zip(_aliases(len(frozen)), frozen, strict=True)
    )
    prompt = (
        "Rank the already-admissible formulas by semantic fit to the recurring "
        "visual contrast in the two neutral summaries. Return every bounded "
        "alias exactly once, best first. The formulas are immutable: do not "
        "change an operator, threshold, unit, feature, conjunction, or polarity. "
        "Use only the material below and return no explanation.\n\n"
        f"rank_input_digest: {rank_input_digest}\n"
        f"version_space_digest: {version_space_digest}\n"
        f"semantic_artifact_digest: {semantic_artifact_digest}\n"
        f"neutral_group_0_summary: {rubrics[0]}\n"
        f"neutral_group_0_nominations: {', '.join(families[0])}\n"
        f"neutral_group_1_summary: {rubrics[1]}\n"
        f"neutral_group_1_nominations: {', '.join(families[1])}\n\n"
        f"immutable_survivors:\n{rows}"
    )
    if len(prompt.encode("utf-8")) > MAX_PROMPT_UTF8_BYTES:
        raise ObjectBongardCodexRankerError("rank prompt exceeds its byte guard")
    if _FORBIDDEN_VISIBLE_WORD.search(prompt) is not None:
        raise ObjectBongardCodexRankerError(
            "rank prompt crosses the sealed text-only boundary"
        )
    return prompt


def object_bongard_codex_ranker_output_schema() -> dict[str, object]:
    # The strict transport admits at most 1,000 enum members, while this
    # complete language has at most 1,740 survivors.  Alias membership and the
    # exact permutation are therefore checked by the closed Python parser.
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


def object_bongard_codex_ranker_protocol_digest() -> str:
    return canonical_digest(
        {
            "schema": OBJECT_BONGARD_CODEX_RANKER_PROTOCOL_SCHEMA,
            "protocol_id": OBJECT_BONGARD_CODEX_RANKER_PROTOCOL_ID,
            "source_digest": object_bongard_codex_ranker_source_digest(),
            "transport_source_digest": (
                object_bongard_codex_ranker_transport_source_digest()
            ),
            "transport_entrypoint": "run_codex_text_structured",
            "receipt_schema": CODEX_RECEIPT_SCHEMA,
            "input_digest_schema": TEXT_STRUCTURED_INPUT_DIGEST_SCHEMA,
            "isolation_policy": CODEX_ISOLATION_POLICY,
            "output_schema": object_bongard_codex_ranker_output_schema(),
            "output_rule": "exact-alias-permutation-resolved-to-profile-digests",
            "maximum_survivor_count": MAX_SURVIVOR_COUNT,
            "maximum_prompt_utf8_bytes": MAX_PROMPT_UTF8_BYTES,
            "profile_language": "one-or-two-positive-at-least-atoms-same-lineage",
            "feature_catalog_digest": OBJECT_FEATURE_CATALOG_DIGEST,
            "model_visible_image_material": False,
            "model_visible_side_names": False,
            "model_visible_held_out_material": False,
            **_authority_data(),
        }
    )


def object_bongard_codex_ranker_model_identity_digest(
    model: str, reasoning_effort: str
) -> str:
    if not isinstance(model, str) or _MODEL.fullmatch(model) is None:
        raise ObjectBongardCodexRankerError("ranker model is invalid")
    if reasoning_effort not in REASONING_EFFORTS:
        raise ObjectBongardCodexRankerError("ranker reasoning effort is invalid")
    return canonical_digest(
        {
            "schema": "gkm.bongard-object-codex-ranker-model-request.v1",
            "model": model,
            "reasoning_effort": reasoning_effort,
            "identity_evidence_policy": (
                "receipt-reported-model-or-explicit-cli-model-flag"
            ),
        }
    )


def object_bongard_codex_ranker_environment_digest(
    *,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str,
    expected_cloud_policy_cache_binding: str,
    expected_transport_source_digest: str,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
) -> str:
    model_digest = object_bongard_codex_ranker_model_identity_digest(
        model, reasoning_effort
    )
    launcher = _digest(expected_launcher_digest, "expected launcher digest")
    transport_source = _digest(
        expected_transport_source_digest, "expected transport source digest"
    )
    if transport_source != object_bongard_codex_ranker_transport_source_digest():
        raise ObjectBongardCodexRankerError(
            "text transport source differs from external commitment"
        )
    policy = expected_cloud_policy_cache_binding
    if policy != "absent":
        if (
            not isinstance(policy, str)
            or not policy.startswith("sha256:")
            or _DIGEST.fullmatch(policy[7:]) is None
        ):
            raise ObjectBongardCodexRankerError(
                "expected policy-cache binding is invalid"
            )
    if not isinstance(model_catalog_snapshot, CodexModelCatalogSnapshot):
        raise ObjectBongardCodexRankerError(
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
        raise ObjectBongardCodexRankerError(
            "Codex no-tools runtime differs from its frozen attestation"
        ) from exc
    return canonical_digest(
        {
            "schema": "gkm.bongard-object-codex-ranker-environment.v1",
            "model_identity_digest": model_digest,
            "expected_launcher_digest": launcher,
            "expected_cloud_policy_cache_binding": policy,
            "model_catalog_digest": model_catalog_snapshot.raw_digest,
            "no_tools_attestation_digest": attestation.attestation_digest,
            "ranker_source_digest": object_bongard_codex_ranker_source_digest(),
            "transport_source_digest": transport_source,
            "receipt_schema": CODEX_RECEIPT_SCHEMA,
            "input_digest_schema": TEXT_STRUCTURED_INPUT_DIGEST_SCHEMA,
            "isolation_policy": CODEX_ISOLATION_POLICY,
            **_authority_data(),
        }
    )


def _canonical_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ObjectBongardCodexRankerError("rank payload must be a JSON object")
    try:
        decoded = json.loads(canonical_json(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardCodexRankerError(
            "rank payload is not finite canonical JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise ObjectBongardCodexRankerError("rank payload must be an object")
    return decoded


def _parse_alias_payload(
    payload: Mapping[str, Any], survivors: tuple[ObjectProfile, ...]
) -> tuple[str, ...]:
    if set(payload) != {"ordered_aliases"}:
        raise ObjectBongardCodexRankerError("rank payload fields differ from schema")
    values = payload["ordered_aliases"]
    if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
        raise ObjectBongardCodexRankerError("ordered aliases must be a JSON list")
    ordered = tuple(values)
    aliases = _aliases(len(survivors))
    if (
        len(ordered) != len(aliases)
        or len(set(ordered)) != len(ordered)
        or set(ordered) != set(aliases)
    ):
        raise ObjectBongardCodexRankerError(
            "rank payload must be the exact survivor-alias permutation"
        )
    by_alias = {
        alias: profile.profile_digest
        for alias, profile in zip(aliases, survivors, strict=True)
    }
    return tuple(by_alias[item] for item in ordered)


def _ordered_alias_payload(
    ordered_profile_digests: Sequence[str], survivors: tuple[ObjectProfile, ...]
) -> dict[str, object]:
    digest_to_alias = {
        profile.profile_digest: alias
        for alias, profile in zip(_aliases(len(survivors)), survivors, strict=True)
    }
    try:
        ordered = [digest_to_alias[item] for item in ordered_profile_digests]
    except KeyError as exc:
        raise ObjectBongardCodexRankerError(
            "rank response contains a foreign profile digest"
        ) from exc
    return {"ordered_aliases": ordered}


def _rank_output_digest(ordered_profile_digests: Sequence[str]) -> str:
    return canonical_digest(
        {
            "schema": OBJECT_BONGARD_RANK_OUTPUT_SCHEMA,
            "ordered_profile_digests": list(ordered_profile_digests),
        }
    )


def _response_content(value: "ObjectBongardRankResponse") -> dict[str, object]:
    return {
        "schema": OBJECT_BONGARD_RANK_RESPONSE_SCHEMA,
        "ordered_profile_digests": list(value.ordered_profile_digests),
        "selected_profile_digest": value.selected_profile_digest,
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
        "formulas_are_immutable": True,
        "image_material_included": False,
        "side_names_included": False,
        "held_out_material_included": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRankResponse:
    """Canonical exact survivor permutation and receipt-attested provenance."""

    ordered_profile_digests: tuple[str, ...]
    selected_profile_digest: str
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
            not self.ordered_profile_digests
            or len(set(self.ordered_profile_digests)) != len(self.ordered_profile_digests)
        ):
            raise ObjectBongardCodexRankerError("rank response is not a permutation")
        for index, item in enumerate(self.ordered_profile_digests):
            _digest(item, f"ordered profile digest {index}")
        if self.selected_profile_digest != self.ordered_profile_digests[0]:
            raise ObjectBongardCodexRankerError("selected profile is not ranked first")
        _identifier(self.ranker_protocol_id, "ranker protocol ID")
        _identifier(self.model_id, "model ID")
        for name in (
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
            raise ObjectBongardCodexRankerError("rank receipt must be an object")
        try:
            canonical_receipt = json.loads(
                canonical_json(dict(self.receipt)).decode("utf-8")
            )
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise ObjectBongardCodexRankerError("rank receipt is not canonical JSON") from exc
        object.__setattr__(self, "receipt", canonical_receipt)
        if set(canonical_receipt) != {"ranker_binding", "transport_receipt"}:
            raise ObjectBongardCodexRankerError("rank receipt envelope fields differ")
        expected_binding = {
            "ranker_protocol_id": self.ranker_protocol_id,
            "ranker_protocol_digest": self.ranker_protocol_digest,
            "model_id": self.model_id,
            "model_identity_digest": self.model_identity_digest,
            "environment_digest": self.environment_digest,
            "rank_input_digest": self.rank_input_digest,
            "output_digest": self.output_digest,
        }
        if (
            canonical_receipt["ranker_binding"] != expected_binding
            or not isinstance(canonical_receipt["transport_receipt"], Mapping)
            or not canonical_receipt["transport_receipt"]
            or self.output_digest
            != _rank_output_digest(self.ordered_profile_digests)
            or self.receipt_digest
            != canonical_digest(
                {
                    "schema": "gkm.bongard-object-rank-receipt.v1",
                    "receipt": canonical_receipt,
                }
            )
            or self.response_digest != canonical_digest(_response_content(self))
        ):
            raise ObjectBongardCodexRankerError("rank response provenance differs")

    @classmethod
    def seal(
        cls,
        *,
        ordered_profile_digests: Sequence[str],
        ranker_protocol_id: str,
        ranker_protocol_digest: str,
        model_id: str,
        model_identity_digest: str,
        environment_digest: str,
        rank_input_digest: str,
        transport_receipt: Mapping[str, Any],
    ) -> "ObjectBongardRankResponse":
        ordered = tuple(ordered_profile_digests)
        if not ordered:
            raise ObjectBongardCodexRankerError("rank response cannot be empty")
        output_digest = _rank_output_digest(ordered)
        receipt = {
            "ranker_binding": {
                "ranker_protocol_id": ranker_protocol_id,
                "ranker_protocol_digest": ranker_protocol_digest,
                "model_id": model_id,
                "model_identity_digest": model_identity_digest,
                "environment_digest": environment_digest,
                "rank_input_digest": rank_input_digest,
                "output_digest": output_digest,
            },
            "transport_receipt": dict(transport_receipt),
        }
        receipt_digest = canonical_digest(
            {"schema": "gkm.bongard-object-rank-receipt.v1", "receipt": receipt}
        )
        values: dict[str, object] = {
            "ordered_profile_digests": ordered,
            "selected_profile_digest": ordered[0],
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
        survivor_profile_digests: Sequence[str],
        rank_input_digest: str,
    ) -> None:
        survivors = tuple(survivor_profile_digests)
        if (
            self.rank_input_digest != rank_input_digest
            or len(self.ordered_profile_digests) != len(survivors)
            or set(self.ordered_profile_digests) != set(survivors)
        ):
            raise ObjectBongardCodexRankerError(
                "rank response must be the exact frozen survivor permutation"
            )

    def to_data(self) -> dict[str, object]:
        return {**_response_content(self), "response_digest": self.response_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardRankResponse":
        raw = _fields(
            value,
            {
                "schema",
                "ordered_profile_digests",
                "selected_profile_digest",
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
                "formulas_are_immutable",
                "image_material_included",
                "side_names_included",
                "held_out_material_included",
                *_authority_data(),
                "response_digest",
            },
            "object rank response",
        )
        if (
            raw["schema"] != OBJECT_BONGARD_RANK_RESPONSE_SCHEMA
            or raw["complete_survivor_permutation"] is not True
            or raw["formulas_are_immutable"] is not True
            or raw["image_material_included"] is not False
            or raw["side_names_included"] is not False
            or raw["held_out_material_included"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or not isinstance(raw["ordered_profile_digests"], list)
            or not isinstance(raw["receipt"], Mapping)
        ):
            raise ObjectBongardCodexRankerError("object rank response policy differs")
        result = cls(
            ordered_profile_digests=tuple(raw["ordered_profile_digests"]),
            selected_profile_digest=raw["selected_profile_digest"],
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
            raise ObjectBongardCodexRankerError("object rank response is not canonical")
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
        raise ObjectBongardCodexRankerError(
            "text rank receipt does not bind the frozen input"
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
        raise ObjectBongardCodexRankerError(
            "text rank receipt model, environment, or payload differs"
        )


def verify_object_bongard_rank_response(
    response: ObjectBongardRankResponse,
    *,
    survivors: Sequence[ObjectProfile],
    neutral_rubrics: Sequence[str],
    feature_nominations: Sequence[Sequence[str]],
    semantic_artifact_digest: str,
    version_space_digest: str,
    rank_input_digest: str,
    expected_response_digest: str,
    model: str,
    reasoning_effort: str,
    expected_launcher_digest: str,
    expected_cloud_policy_cache_binding: str,
    expected_transport_source_digest: str,
    model_catalog_snapshot: CodexModelCatalogSnapshot,
    no_tools_attestation: CodexNoToolsAttestation,
) -> ObjectBongardRankResponse:
    """Cold-verify a persisted response without invoking the transport."""

    if not isinstance(response, ObjectBongardRankResponse):
        raise TypeError("response must be ObjectBongardRankResponse")
    frozen, _, _ = _rank_inputs(
        survivors=survivors,
        neutral_rubrics=neutral_rubrics,
        feature_nominations=feature_nominations,
        semantic_artifact_digest=semantic_artifact_digest,
        version_space_digest=version_space_digest,
        rank_input_digest=rank_input_digest,
    )
    if response.response_digest != _digest(
        expected_response_digest, "expected response digest"
    ):
        raise ObjectBongardCodexRankerError(
            "rank response differs from external commitment"
        )
    response.assert_matches(
        survivor_profile_digests=tuple(item.profile_digest for item in frozen),
        rank_input_digest=rank_input_digest,
    )
    expected_model_identity = object_bongard_codex_ranker_model_identity_digest(
        model, reasoning_effort
    )
    expected_environment = object_bongard_codex_ranker_environment_digest(
        model=model,
        reasoning_effort=reasoning_effort,
        expected_launcher_digest=expected_launcher_digest,
        expected_cloud_policy_cache_binding=expected_cloud_policy_cache_binding,
        expected_transport_source_digest=expected_transport_source_digest,
        model_catalog_snapshot=model_catalog_snapshot,
        no_tools_attestation=no_tools_attestation,
    )
    if (
        response.ranker_protocol_id != OBJECT_BONGARD_CODEX_RANKER_PROTOCOL_ID
        or response.ranker_protocol_digest
        != object_bongard_codex_ranker_protocol_digest()
        or response.model_id != model
        or response.model_identity_digest != expected_model_identity
        or response.environment_digest != expected_environment
    ):
        raise ObjectBongardCodexRankerError(
            "rank response protocol, model, or environment differs"
        )
    prompt = object_bongard_codex_ranker_prompt(
        survivors=frozen,
        neutral_rubrics=neutral_rubrics,
        feature_nominations=feature_nominations,
        semantic_artifact_digest=semantic_artifact_digest,
        version_space_digest=version_space_digest,
        rank_input_digest=rank_input_digest,
    )
    schema = object_bongard_codex_ranker_output_schema()
    payload = _ordered_alias_payload(response.ordered_profile_digests, frozen)
    transport_receipt = response.receipt.get("transport_receipt")
    if not isinstance(transport_receipt, Mapping):
        raise ObjectBongardCodexRankerError("rank transport receipt is invalid")
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
    if ObjectBongardRankResponse.from_data(response.to_data()) != response:
        raise ObjectBongardCodexRankerError("rank response cold round trip differs")
    return response


@dataclass(frozen=True, slots=True)
class ObjectBongardCodexRanker:
    """Configured receipt-attested text-only ranker."""

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
        object_bongard_codex_ranker_model_identity_digest(
            self.model, self.reasoning_effort
        )
        _digest(self.expected_launcher_digest, "expected launcher digest")
        if not isinstance(self.cloud_policy_cache_snapshot, CloudPolicyCacheSnapshot):
            raise ObjectBongardCodexRankerError(
                "an exact cloud policy-cache snapshot is required"
            )
        if (
            self.expected_cloud_policy_cache_binding
            != self.cloud_policy_cache_snapshot.binding
        ):
            raise ObjectBongardCodexRankerError(
                "policy-cache snapshot differs from external commitment"
            )
        object_bongard_codex_ranker_environment_digest(
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
            raise ObjectBongardCodexRankerError(
                "ranker timeout minutes must lie in 1..120"
            )
        if not isinstance(self.verbose, bool):
            raise TypeError("verbose must be bool")
        if not isinstance(self.executable, str) or not self.executable:
            raise ObjectBongardCodexRankerError("ranker executable must be nonempty")
        if not callable(self.transport):
            raise TypeError("ranker transport must be callable")

    @property
    def protocol_digest(self) -> str:
        return object_bongard_codex_ranker_protocol_digest()

    @property
    def model_identity_digest(self) -> str:
        return object_bongard_codex_ranker_model_identity_digest(
            self.model, self.reasoning_effort
        )

    @property
    def environment_digest(self) -> str:
        return object_bongard_codex_ranker_environment_digest(
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
        survivors: Sequence[ObjectProfile],
        *,
        neutral_rubrics: Sequence[str],
        feature_nominations: Sequence[Sequence[str]],
        semantic_artifact_digest: str,
        version_space_digest: str,
        rank_input_digest: str,
    ) -> ObjectBongardRankResponse:
        frozen, _, _ = _rank_inputs(
            survivors=survivors,
            neutral_rubrics=neutral_rubrics,
            feature_nominations=feature_nominations,
            semantic_artifact_digest=semantic_artifact_digest,
            version_space_digest=version_space_digest,
            rank_input_digest=rank_input_digest,
        )
        prompt = object_bongard_codex_ranker_prompt(
            survivors=frozen,
            neutral_rubrics=neutral_rubrics,
            feature_nominations=feature_nominations,
            semantic_artifact_digest=semantic_artifact_digest,
            version_space_digest=version_space_digest,
            rank_input_digest=rank_input_digest,
        )
        schema = object_bongard_codex_ranker_output_schema()
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
            raise ObjectBongardCodexRankerError("text-only rank transport failed") from exc
        if not isinstance(result, CodexStructuredResult):
            raise ObjectBongardCodexRankerError(
                "text transport returned the wrong result type"
            )
        payload = _canonical_payload(result.payload)
        ordered_digests = _parse_alias_payload(payload, frozen)
        if not isinstance(result.receipt, CodexReceipt):
            raise ObjectBongardCodexRankerError(
                "text transport returned no CodexReceipt"
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
        response = ObjectBongardRankResponse.seal(
            ordered_profile_digests=ordered_digests,
            ranker_protocol_id=OBJECT_BONGARD_CODEX_RANKER_PROTOCOL_ID,
            ranker_protocol_digest=self.protocol_digest,
            model_id=self.model,
            model_identity_digest=self.model_identity_digest,
            environment_digest=self.environment_digest,
            rank_input_digest=rank_input_digest,
            transport_receipt=receipt,
        )
        response.assert_matches(
            survivor_profile_digests=tuple(item.profile_digest for item in frozen),
            rank_input_digest=rank_input_digest,
        )
        return response

    def verify_response(
        self,
        response: ObjectBongardRankResponse,
        *,
        survivors: Sequence[ObjectProfile],
        neutral_rubrics: Sequence[str],
        feature_nominations: Sequence[Sequence[str]],
        semantic_artifact_digest: str,
        version_space_digest: str,
        rank_input_digest: str,
        expected_response_digest: str,
    ) -> ObjectBongardRankResponse:
        return verify_object_bongard_rank_response(
            response,
            survivors=survivors,
            neutral_rubrics=neutral_rubrics,
            feature_nominations=feature_nominations,
            semantic_artifact_digest=semantic_artifact_digest,
            version_space_digest=version_space_digest,
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


__all__ = [
    "MAX_PROMPT_UTF8_BYTES",
    "MAX_SURVIVOR_COUNT",
    "OBJECT_BONGARD_CODEX_RANKER_PROTOCOL_ID",
    "OBJECT_BONGARD_CODEX_RANKER_PROTOCOL_SCHEMA",
    "OBJECT_BONGARD_RANK_INPUT_SCHEMA",
    "OBJECT_BONGARD_RANK_OUTPUT_SCHEMA",
    "OBJECT_BONGARD_RANK_RESPONSE_SCHEMA",
    "ObjectBongardCodexRanker",
    "ObjectBongardCodexRankerError",
    "ObjectBongardRankResponse",
    "TextStructuredTransport",
    "object_bongard_codex_ranker_authority_data",
    "object_bongard_codex_ranker_environment_digest",
    "object_bongard_codex_ranker_model_identity_digest",
    "object_bongard_codex_ranker_output_schema",
    "object_bongard_codex_ranker_prompt",
    "object_bongard_codex_ranker_protocol_digest",
    "object_bongard_codex_ranker_source_digest",
    "object_bongard_codex_ranker_transport_source_digest",
    "object_bongard_rank_input_digest",
    "verify_object_bongard_rank_response",
]

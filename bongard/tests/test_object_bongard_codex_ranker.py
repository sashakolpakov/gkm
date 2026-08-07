"""Offline tests for the generic text-only object-profile ranker."""

from __future__ import annotations

import ast
from copy import deepcopy
import itertools
from pathlib import Path
import re
from typing import Any, Mapping

import pytest

import bongard.transport as transport_module
from bongard.canonical import canonical_digest
from bongard.object_bongard_codex_ranker import (
    MAX_PROMPT_UTF8_BYTES,
    MAX_SURVIVOR_COUNT,
    OBJECT_BONGARD_CODEX_RANKER_PROTOCOL_ID,
    ObjectBongardCodexRanker,
    ObjectBongardCodexRankerError,
    ObjectBongardRankResponse,
    object_bongard_codex_ranker_authority_data,
    object_bongard_codex_ranker_output_schema,
    object_bongard_codex_ranker_prompt,
    object_bongard_codex_ranker_protocol_digest,
    object_bongard_codex_ranker_transport_source_digest,
    object_bongard_rank_input_digest,
)
from bongard.prototype_object_profiles import (
    OBJECT_FEATURE_CATALOG,
    ObjectProfile,
    ObjectProfileAtom,
    ObjectProfileOperator,
)
from bongard.tests.no_tools_fixture import canonical_no_tools_runtime
from bongard.transport import (
    CODEX_APPLY_PATCH_TOOL_TYPE,
    CODEX_EFFECTIVE_TOOL_MODE,
    CODEX_ISOLATION_POLICY,
    CODEX_RECEIPT_SCHEMA,
    CODEX_TOOL_SURFACE_DIGEST,
    CODEX_TRANSPORT_POLICY_DIGEST,
    PINNED_CODEX_CLI_VERSION,
    CloudPolicyCacheSnapshot,
    CodexReceipt,
    CodexStructuredResult,
)


MODEL = "gpt-5.6-sol"
EFFORT = "medium"
LAUNCHER_DIGEST = "b" * 64
MODEL_CATALOG, NO_TOOLS_ATTESTATION = canonical_no_tools_runtime(LAUNCHER_DIGEST)
SEMANTIC_DIGEST = "a" * 64
VERSION_DIGEST = "c" * 64
RUBRICS = (
    "A pointed bird like contour with several slanted spans.",
    "A rounded contour arrangement with a visible opening.",
)
NOMINATIONS = (
    (
        "pointed_terminal_appendage_count",
        "oblique_span_support_ppm",
        "bird_like_support_ppm",
    ),
    ("open_outline_support_ppm", "rounded_leaf_support_ppm"),
)


def _profile(profile_id: str, *atoms: tuple[str, int]) -> ObjectProfile:
    return ObjectProfile.create(
        profile_id,
        tuple(
            ObjectProfileAtom(feature_id, ObjectProfileOperator.AT_LEAST, target)
            for feature_id, target in atoms
        ),
    )


SURVIVORS = (
    _profile("object-rank:bird", ("bird_like_support_ppm", 750_000)),
    _profile("object-rank:oblique", ("oblique_span_support_ppm", 500_000)),
    _profile(
        "object-rank:pointed-and-bird",
        ("pointed_terminal_appendage_count", 2),
        ("bird_like_support_ppm", 500_000),
    ),
)
RANK_INPUT_DIGEST = object_bongard_rank_input_digest(
    survivors=SURVIVORS,
    neutral_rubrics=RUBRICS,
    feature_nominations=NOMINATIONS,
    semantic_artifact_digest=SEMANTIC_DIGEST,
    version_space_digest=VERSION_DIGEST,
)


def _receipt(
    prompt: str,
    schema: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    launcher_digest: str = LAUNCHER_DIGEST,
) -> CodexReceipt:
    schema_digest = canonical_digest(dict(schema))
    text_capture = NO_TOOLS_ATTESTATION.to_dict()["captures"][0]
    binding = {
        "model_catalog_digest": MODEL_CATALOG.raw_digest,
        "transport_policy_digest": CODEX_TRANSPORT_POLICY_DIGEST,
        "command_digest": text_capture["normalized_command_digest"],
        "effective_tool_mode": CODEX_EFFECTIVE_TOOL_MODE,
        "apply_patch_tool_type": CODEX_APPLY_PATCH_TOOL_TYPE,
        "tool_surface_digest": CODEX_TOOL_SURFACE_DIGEST,
        "tool_surface_attestation_digest": NO_TOOLS_ATTESTATION.attestation_digest,
    }
    causal = transport_module._causal_text_input_metadata(
        prompt, schema_digest, binding
    )
    body: dict[str, Any] = {
        "schema": CODEX_RECEIPT_SCHEMA,
        "source": "codex-cli",
        "requested_model": MODEL,
        "reported_model": "",
        "model_identity_evidence": "explicit-cli-model-flag;jsonl-omits-model",
        "requested_reasoning_effort": EFFORT,
        "input_tokens": 100,
        "cached_input_tokens": 0,
        "output_tokens": 20,
        "reasoning_output_tokens": 5,
        "thread_id": "00000000-0000-4000-8000-000000000071",
        "codex_cli_version": PINNED_CODEX_CLI_VERSION,
        "codex_launcher_digest": launcher_digest,
        "cloud_config_bundle_cache_binding": "absent",
        **causal,
        "output_schema_digest": schema_digest,
        "structured_output_digest": canonical_digest(dict(payload)),
        "proposed_source_digest": "",
        "proposed_log_digest": "",
        "event_stream_digest": "d" * 64,
        "event_types": [
            "thread.started",
            "turn.started",
            "item.completed",
            "turn.completed",
        ],
        "item_types": ["agent_message"],
        "isolation_policy": CODEX_ISOLATION_POLICY,
        "outcome": "success",
    }
    body["receipt_digest"] = canonical_digest(body)
    return CodexReceipt(
        **{
            **body,
            "event_types": tuple(body["event_types"]),
            "item_types": tuple(body["item_types"]),
        }
    )


def _ranker(transport) -> ObjectBongardCodexRanker:
    return ObjectBongardCodexRanker(
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
        expected_cloud_policy_cache_binding="absent",
        expected_transport_source_digest=(
            object_bongard_codex_ranker_transport_source_digest()
        ),
        model_catalog_snapshot=MODEL_CATALOG,
        no_tools_attestation=NO_TOOLS_ATTESTATION,
        transport=transport,
    )


def _call(ranker: ObjectBongardCodexRanker) -> ObjectBongardRankResponse:
    return ranker(
        SURVIVORS,
        neutral_rubrics=RUBRICS,
        feature_nominations=NOMINATIONS,
        semantic_artifact_digest=SEMANTIC_DIGEST,
        version_space_digest=VERSION_DIGEST,
        rank_input_digest=RANK_INPUT_DIGEST,
    )


def test_exact_text_only_rank_is_receipted_and_cold_verified() -> None:
    calls = 0

    def transport(prompt, schema, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["model"] == MODEL
        assert kwargs["reasoning_effort"] == EFFORT
        assert kwargs["expected_launcher_digest"] == LAUNCHER_DIGEST
        assert kwargs["cloud_policy_cache_snapshot"].binding == "absent"
        assert kwargs["model_catalog_snapshot"] is MODEL_CATALOG
        assert kwargs["tool_surface_attestation"] is NO_TOOLS_ATTESTATION
        assert kwargs["expected_tool_surface_attestation_digest"] == (
            NO_TOOLS_ATTESTATION.attestation_digest
        )
        assert "c0000" in prompt and "c0002" in prompt
        assert "bird_like_support_ppm AT_LEAST 750000 ppm" in prompt
        assert "pointed_terminal_appendage_count AT_LEAST 2 count" in prompt
        assert "on the same object lineage" in prompt
        assert RUBRICS[0] in prompt and RUBRICS[1] in prompt
        assert not re.search(
            r"\b(?:pixel|label|query|positive|negative)s?\b", prompt, re.I
        )
        assert schema == object_bongard_codex_ranker_output_schema()
        assert "enum" not in schema["properties"]["ordered_aliases"]["items"]
        payload = {"ordered_aliases": ["c0002", "c0000", "c0001"]}
        return CodexStructuredResult(payload, _receipt(prompt, schema, payload))

    ranker = _ranker(transport)
    response = _call(ranker)
    assert calls == 1
    assert response.ordered_profile_digests == (
        SURVIVORS[2].profile_digest,
        SURVIVORS[0].profile_digest,
        SURVIVORS[1].profile_digest,
    )
    assert response.selected_profile_digest == SURVIVORS[2].profile_digest
    assert response.ranker_protocol_id == OBJECT_BONGARD_CODEX_RANKER_PROTOCOL_ID
    assert response.ranker_protocol_digest == (
        object_bongard_codex_ranker_protocol_digest()
    )
    assert ObjectBongardRankResponse.from_data(response.to_data()) == response
    assert ranker.verify_response(
        response,
        survivors=SURVIVORS,
        neutral_rubrics=RUBRICS,
        feature_nominations=NOMINATIONS,
        semantic_artifact_digest=SEMANTIC_DIGEST,
        version_space_digest=VERSION_DIGEST,
        rank_input_digest=RANK_INPUT_DIGEST,
        expected_response_digest=response.response_digest,
    ) is response
    authority = object_bongard_codex_ranker_authority_data()
    assert authority["python_is_canonical_authority"] is True
    assert authority["codex_may_edit_formulas"] is False
    assert authority["lean_required"] is False
    assert authority["lean_removable"] is True


@pytest.mark.parametrize(
    "payload",
    (
        {"ordered_aliases": ["c0000", "c0001"]},
        {"ordered_aliases": ["c0000", "c0000", "c0002"]},
        {"ordered_aliases": ["c0000", "c0001", "c9999"]},
        {
            "ordered_aliases": ["c0000", "c0001", "c0002"],
            "explanation": "not admitted",
        },
    ),
)
def test_incomplete_duplicate_foreign_or_extra_payload_is_rejected(
    payload: Mapping[str, Any],
) -> None:
    def transport(prompt, schema, **kwargs):
        return CodexStructuredResult(dict(payload), _receipt(prompt, schema, payload))

    with pytest.raises(ObjectBongardCodexRankerError):
        _call(_ranker(transport))


def test_receipt_and_response_tampering_fail_closed() -> None:
    payload = {"ordered_aliases": ["c0000", "c0001", "c0002"]}

    def wrong_prompt(prompt, schema, **kwargs):
        return CodexStructuredResult(
            payload, _receipt(prompt + " altered", schema, payload)
        )

    with pytest.raises(ObjectBongardCodexRankerError, match="frozen input"):
        _call(_ranker(wrong_prompt))

    def valid(prompt, schema, **kwargs):
        return CodexStructuredResult(payload, _receipt(prompt, schema, payload))

    ranker = _ranker(valid)
    response = _call(ranker)
    tampered = deepcopy(response.to_data())
    tampered["selected_profile_digest"] = SURVIVORS[1].profile_digest
    with pytest.raises(ObjectBongardCodexRankerError):
        ObjectBongardRankResponse.from_data(tampered)
    with pytest.raises(ObjectBongardCodexRankerError, match="external commitment"):
        ranker.verify_response(
            response,
            survivors=SURVIVORS,
            neutral_rubrics=RUBRICS,
            feature_nominations=NOMINATIONS,
            semantic_artifact_digest=SEMANTIC_DIGEST,
            version_space_digest=VERSION_DIGEST,
            rank_input_digest=RANK_INPUT_DIGEST,
            expected_response_digest="0" * 64,
        )


def test_rank_input_digest_and_neutral_boundary_are_mandatory() -> None:
    calls = 0

    def forbidden(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("transport must not run")

    ranker = _ranker(forbidden)
    with pytest.raises(ObjectBongardCodexRankerError, match="canonical preimage"):
        ranker(
            SURVIVORS,
            neutral_rubrics=RUBRICS,
            feature_nominations=NOMINATIONS,
            semantic_artifact_digest=SEMANTIC_DIGEST,
            version_space_digest=VERSION_DIGEST,
            rank_input_digest="0" * 64,
        )
    bad_rubrics = ("A query shaped contour.", RUBRICS[1])
    with pytest.raises(ObjectBongardCodexRankerError, match="neutral"):
        object_bongard_rank_input_digest(
            survivors=SURVIVORS,
            neutral_rubrics=bad_rubrics,
            feature_nominations=NOMINATIONS,
            semantic_artifact_digest=SEMANTIC_DIGEST,
            version_space_digest=VERSION_DIGEST,
        )
    assert calls == 0


def _complete_grid_profiles() -> tuple[ObjectProfile, ...]:
    thresholds_by_feature = tuple(
        tuple(
            ObjectProfileAtom(
                spec.feature_id, ObjectProfileOperator.AT_LEAST, threshold
            )
            for threshold in (
                (250_000, 500_000, 750_000, 1_000_000)
                if spec.unit == "ppm"
                else (1, 2, 3, 4)
            )
        )
        for spec in OBJECT_FEATURE_CATALOG
    )
    atom_groups = [
        (atom,) for group in thresholds_by_feature for atom in group
    ]
    atom_groups.extend(
        (left, right)
        for left_group, right_group in itertools.combinations(
            thresholds_by_feature, 2
        )
        for left in left_group
        for right in right_group
    )
    return tuple(
        ObjectProfile.create(f"object-rank-grid:{index:04d}", atoms)
        for index, atoms in enumerate(atom_groups)
    )


def test_full_1740_profile_grid_fits_prompt_without_schema_enum() -> None:
    profiles = _complete_grid_profiles()
    assert len(profiles) == MAX_SURVIVOR_COUNT == 1_740
    digest = object_bongard_rank_input_digest(
        survivors=profiles,
        neutral_rubrics=RUBRICS,
        feature_nominations=NOMINATIONS,
        semantic_artifact_digest=SEMANTIC_DIGEST,
        version_space_digest=VERSION_DIGEST,
    )
    prompt = object_bongard_codex_ranker_prompt(
        survivors=profiles,
        neutral_rubrics=RUBRICS,
        feature_nominations=NOMINATIONS,
        semantic_artifact_digest=SEMANTIC_DIGEST,
        version_space_digest=VERSION_DIGEST,
        rank_input_digest=digest,
    )
    assert "c1739" in prompt
    assert len(prompt.encode("utf-8")) < MAX_PROMPT_UTF8_BYTES
    schema = object_bongard_codex_ranker_output_schema()
    assert "enum" not in schema["properties"]["ordered_aliases"]["items"]


def test_ranker_has_no_lean_import() -> None:
    source_path = Path(__file__).parents[1] / "object_bongard_codex_ranker.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    assert not any("lean" in name.lower() for name in imported)

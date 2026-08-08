"""Offline tests for the text-only verified-rubric survivor ranker."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping

import pytest

from bongard.canonical import canonical_digest
from bongard.object_bongard_rubric_observer import (
    ObjectBongardRubricObserverArtifact,
    ObjectBongardRubricSpec,
    RUBRIC_ORDINAL_LEVEL_ANCHORS,
    object_bongard_rubric_ordinal_scale_digest,
)
from bongard.object_bongard_rubric_ranker import (
    OBJECT_BONGARD_RUBRIC_RANKER_PROTOCOL_ID,
    ObjectBongardRubricRankResponse,
    ObjectBongardRubricRanker,
    ObjectBongardRubricRankerError,
    object_bongard_rubric_rank_input_digest,
    object_bongard_rubric_ranker_authority_data,
    object_bongard_rubric_ranker_output_schema,
    object_bongard_rubric_ranker_prompt,
    object_bongard_rubric_ranker_protocol_digest,
    object_bongard_rubric_ranker_transport_source_digest,
)
from bongard.object_bongard_rubric_version_space import (
    ObjectBongardRubricSupportVersionSpace,
    build_object_bongard_rubric_support_version_space,
)
from bongard.object_bongard_turn_journal import (
    ObjectBongardTextTurnJournalTransport,
    ObjectBongardTurnRuntime,
)
from bongard.tests.test_object_bongard_semantics import _describe as _describe_semantic
from bongard.tests.test_object_bongard_rubric_version_space import (
    _observed_artifact,
)
from bongard.tests.no_tools_fixture import canonical_no_tools_runtime
import bongard.transport as transport_module
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
MODEL_CATALOG, NO_TOOLS_ATTESTATION = canonical_no_tools_runtime(
    LAUNCHER_DIGEST
)
TARGET_RUBRIC = "A winged angular form with several slanted spans."
CONTRAST_RUBRIC = "A rounded compact form with a curved boundary."
FEATURES = ("oblique_span_support_ppm", "bird_like_support_ppm")


@lru_cache(maxsize=1)
def _semantic_artifact():
    artifact, calls = _describe_semantic()
    assert calls == 1
    return artifact


def _spec(semantic_artifact=None) -> ObjectBongardRubricSpec:
    semantic = _semantic_artifact() if semantic_artifact is None else semantic_artifact
    return ObjectBongardRubricSpec.from_semantic_artifact(
        semantic, expected_artifact_digest=semantic.artifact_digest
    )


@lru_cache(maxsize=None)
def _version_space(
    spec: ObjectBongardRubricSpec, *, survivor_count: int = 3
) -> tuple[
    ObjectBongardRubricSupportVersionSpace,
    tuple[ObjectBongardRubricObserverArtifact, ...],
    tuple[ObjectBongardRubricObserverArtifact, ...],
]:
    if survivor_count not in (0, 3):
        raise ValueError("rank fixture supports only zero or three survivors")
    if survivor_count == 0:
        _, witnessed, absent = _version_space(spec, survivor_count=3)
        version = build_object_bongard_rubric_support_version_space(
            spec, absent, witnessed
        )
        assert not version.survivor_candidate_digests
        return version, absent, witnessed
    positives = tuple(
        _observed_artifact(
            f"bd/rank_fixture/1/{index}.png",
            image_index=index,
            object_interval=(3, 3),
            scene_interval=(0, 0),
            rubric_spec=spec,
        )
        for index in range(6)
    )
    negatives = tuple(
        _observed_artifact(
            f"bd/rank_fixture/0/{index}.png",
            image_index=index + 6,
            object_interval=(0, 0),
            scene_interval=(0, 0),
            rubric_spec=spec,
        )
        for index in range(6)
    )
    version = build_object_bongard_rubric_support_version_space(
        spec, positives, negatives
    )
    assert len(version.survivor_candidate_digests) == survivor_count
    return version, positives, negatives


def _text_receipt(
    prompt: str,
    schema: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    launcher_digest: str = LAUNCHER_DIGEST,
) -> CodexReceipt:
    schema_digest = canonical_digest(dict(schema))
    capture = next(
        row
        for row in NO_TOOLS_ATTESTATION.to_dict()["captures"]
        if row["modality"] == "text"
    )
    binding = {
        "model_catalog_digest": MODEL_CATALOG.raw_digest,
        "transport_policy_digest": CODEX_TRANSPORT_POLICY_DIGEST,
        "command_digest": capture["normalized_command_digest"],
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
        "thread_id": "00000000-0000-4000-8000-000000000091",
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


class _Transport:
    def __init__(self, aliases: tuple[str, ...] = ("r002", "r000", "r001")):
        self.aliases = aliases
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, prompt, schema, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        assert kwargs["model"] == MODEL
        assert kwargs["reasoning_effort"] == EFFORT
        assert kwargs["expected_launcher_digest"] == LAUNCHER_DIGEST
        assert kwargs["cloud_policy_cache_snapshot"].binding == "absent"
        assert kwargs["model_catalog_snapshot"] is MODEL_CATALOG
        assert kwargs["tool_surface_attestation"] is NO_TOOLS_ATTESTATION
        assert kwargs["expected_tool_surface_attestation_digest"] == (
            NO_TOOLS_ATTESTATION.attestation_digest
        )
        assert schema == object_bongard_rubric_ranker_output_schema()
        payload = {"ordered_aliases": list(self.aliases)}
        return CodexStructuredResult(
            payload, _text_receipt(prompt, schema, payload)
        )


class _ForbiddenTransport:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("model transport called during cold replay")


def _ranker(transport) -> ObjectBongardRubricRanker:
    return ObjectBongardRubricRanker(
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
        expected_cloud_policy_cache_binding="absent",
        expected_transport_source_digest=(
            object_bongard_rubric_ranker_transport_source_digest()
        ),
        model_catalog_snapshot=MODEL_CATALOG,
        no_tools_attestation=NO_TOOLS_ATTESTATION,
        minutes=3,
        executable="/private/synthetic-codex",
        transport=transport,
    )


def _inputs():
    semantic = _semantic_artifact()
    spec = _spec(semantic)
    version, positives, negatives = _version_space(spec)
    rank_input = object_bongard_rubric_rank_input_digest(
        version_space=version,
        rubric_spec=spec,
        semantic_artifact=semantic,
        positive_support_artifacts=positives,
        negative_support_artifacts=negatives,
    )
    return semantic, spec, version, positives, negatives, rank_input


def _call(ranker: ObjectBongardRubricRanker):
    semantic, spec, version, positives, negatives, rank_input = _inputs()
    return ranker(
        version,
        rubric_spec=spec,
        semantic_artifact=semantic,
        positive_support_artifacts=positives,
        negative_support_artifacts=negatives,
        rank_input_digest=rank_input,
    )


def test_exact_verified_survivors_are_ranked_and_cold_verified() -> None:
    semantic, spec, version, positives, negatives, rank_input = _inputs()
    transport = _Transport()
    ranker = _ranker(transport)

    response = ranker(
        version,
        rubric_spec=spec,
        semantic_artifact=semantic,
        positive_support_artifacts=positives,
        negative_support_artifacts=negatives,
        rank_input_digest=rank_input,
    )

    expected = version.survivor_candidate_digests
    assert transport.calls == 1
    assert response.ordered_candidate_digests == (
        expected[2],
        expected[0],
        expected[1],
    )
    assert response.selected_candidate_digest == expected[2]
    assert response.ranker_protocol_id == OBJECT_BONGARD_RUBRIC_RANKER_PROTOCOL_ID
    assert response.ranker_protocol_digest == (
        object_bongard_rubric_ranker_protocol_digest()
    )
    assert ObjectBongardRubricRankResponse.from_data(response.to_data()) == response
    assert ranker.verify_response(
        response,
        version_space=version,
        rubric_spec=spec,
        semantic_artifact=semantic,
        positive_support_artifacts=positives,
        negative_support_artifacts=negatives,
        rank_input_digest=rank_input,
        expected_response_digest=response.response_digest,
    ) is response
    assert transport.calls == 1


def test_prompt_contains_only_rubrics_and_immutable_candidate_inventory() -> None:
    semantic, spec, version, positives, negatives, rank_input = _inputs()
    prompt = object_bongard_rubric_ranker_prompt(
        version_space=version,
        rubric_spec=spec,
        semantic_artifact=semantic,
        positive_support_artifacts=positives,
        negative_support_artifacts=negatives,
        rank_input_digest=rank_input,
    )

    assert TARGET_RUBRIC in prompt and CONTRAST_RUBRIC in prompt
    for digest in version.survivor_candidate_digests:
        candidate = version.survivor(digest)
        assert candidate.candidate_id in prompt
        assert candidate.candidate_digest in prompt
        assert candidate.formula in prompt
    assert object_bongard_rubric_ordinal_scale_digest() in prompt
    for level, meaning in RUBRIC_ORDINAL_LEVEL_ANCHORS:
        assert f"level={level}; meaning={meaning}" in prompt
    for artifact in positives + negatives:
        assert artifact.panel_id not in prompt
        assert artifact.artifact_digest not in prompt
    assert not re.search(
        r"(?:\bgroup[_ -]?[01](?:_ref)?\b|"
        r"\b(?:positive|negative)\s+(?:side|support|example)s?\b|"
        r"\bsupport\s+(?:side|label)s?\b|"
        r"\bquery\s+(?:panel|item|input|example)s?\b)",
        prompt,
        re.I,
    )
    assert object_bongard_rubric_ranker_output_schema() == {
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


@pytest.mark.parametrize(
    "aliases",
    [
        ("r000", "r001"),
        ("r000", "r000", "r002"),
        ("r000", "r001", "r999"),
        ("r000", "r001", "r002", "r003"),
    ],
)
def test_incomplete_duplicate_foreign_or_extra_aliases_are_rejected(
    aliases: tuple[str, ...],
) -> None:
    with pytest.raises(ObjectBongardRubricRankerError):
        _call(_ranker(_Transport(aliases)))


def test_extra_payload_field_is_rejected() -> None:
    def transport(prompt, schema, **kwargs):
        payload = {
            "ordered_aliases": ["r000", "r001", "r002"],
            "formula": "model-authored formula is forbidden",
        }
        return CodexStructuredResult(payload, _text_receipt(prompt, schema, payload))

    with pytest.raises(ObjectBongardRubricRankerError):
        _call(_ranker(transport))


def test_stale_receipt_and_response_tamper_fail_closed() -> None:
    def stale(prompt, schema, **kwargs):
        payload = {"ordered_aliases": ["r000", "r001", "r002"]}
        return CodexStructuredResult(
            payload, _text_receipt(prompt + " altered", schema, payload)
        )

    with pytest.raises(ObjectBongardRubricRankerError, match="frozen input"):
        _call(_ranker(stale))

    ranker = _ranker(_Transport())
    response = _call(ranker)
    tampered = deepcopy(response.to_data())
    tampered["selected_candidate_digest"] = response.ordered_candidate_digests[1]
    with pytest.raises(ObjectBongardRubricRankerError):
        ObjectBongardRubricRankResponse.from_data(tampered)


def test_spec_artifact_contrast_and_rank_input_drift_fail_before_transport() -> None:
    semantic, spec, version, positives, negatives, rank_input = _inputs()
    transport = _Transport()
    ranker = _ranker(transport)
    with pytest.raises(ObjectBongardRubricRankerError):
        ranker(
            version,
            rubric_spec=spec,
            semantic_artifact=semantic,
            positive_support_artifacts=(negatives[0], *positives[1:]),
            negative_support_artifacts=negatives,
            rank_input_digest=rank_input,
        )
    forged_target = ObjectBongardRubricSpec.create(
        semantic.artifact_digest,
        "A compact pointed contour with multiple angled segments.",
        FEATURES,
    )
    with pytest.raises(ObjectBongardRubricRankerError):
        ranker(
            version,
            rubric_spec=forged_target,
            semantic_artifact=semantic,
            positive_support_artifacts=positives,
            negative_support_artifacts=negatives,
            rank_input_digest=rank_input,
        )
    with pytest.raises(ObjectBongardRubricRankerError):
        ranker(
            version,
            rubric_spec=spec,
            semantic_artifact=semantic,
            positive_support_artifacts=positives,
            negative_support_artifacts=negatives,
            rank_input_digest="f" * 64,
        )
    assert transport.calls == 0


def test_no_survivor_gap_cannot_reach_ranker() -> None:
    semantic = _semantic_artifact()
    spec = _spec(semantic)
    version, positives, negatives = _version_space(spec, survivor_count=0)
    with pytest.raises(ObjectBongardRubricRankerError, match="verified survivors"):
        object_bongard_rubric_rank_input_digest(
            version_space=version,
            rubric_spec=spec,
            semantic_artifact=semantic,
            positive_support_artifacts=positives,
            negative_support_artifacts=negatives,
        )


def test_text_turn_journal_is_drop_in_exactly_once_transport(tmp_path: Path) -> None:
    semantic, spec, version, positives, negatives, rank_input = _inputs()
    prompt = object_bongard_rubric_ranker_prompt(
        version_space=version,
        rubric_spec=spec,
        semantic_artifact=semantic,
        positive_support_artifacts=positives,
        negative_support_artifacts=negatives,
        rank_input_digest=rank_input,
    )
    schema = object_bongard_rubric_ranker_output_schema()
    policy = CloudPolicyCacheSnapshot(None)
    runtime = ObjectBongardTurnRuntime(
        model=MODEL,
        reasoning_effort=EFFORT,
        minutes=3,
        verbose=False,
        executable="/private/synthetic-codex",
        cloud_policy_cache_snapshot=policy,
        model_catalog_snapshot=MODEL_CATALOG,
        expected_launcher_digest=LAUNCHER_DIGEST,
        no_tools_attestation=NO_TOOLS_ATTESTATION,
        transport_source_digest=(
            object_bongard_rubric_ranker_transport_source_digest()
        ),
    )
    physical = _Transport()
    journal = ObjectBongardTextTurnJournalTransport(
        tmp_path / "rubric-rank-journal",
        authorization_digest="sha256:" + "1" * 64,
        execution_precommit_digest="sha256:" + "2" * 64,
        task_id="bd_rubric_rank_fixture",
        turn_kind="rubric_rank",
        expected_prompt=prompt,
        expected_output_schema=schema,
        runtime=runtime,
        underlying_transport=physical,
    )
    ranker = ObjectBongardRubricRanker(
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        cloud_policy_cache_snapshot=policy,
        expected_cloud_policy_cache_binding="absent",
        expected_transport_source_digest=(
            object_bongard_rubric_ranker_transport_source_digest()
        ),
        model_catalog_snapshot=MODEL_CATALOG,
        no_tools_attestation=NO_TOOLS_ATTESTATION,
        minutes=3,
        executable="/private/synthetic-codex",
        transport=journal,
    )

    first = ranker(
        version,
        rubric_spec=spec,
        semantic_artifact=semantic,
        positive_support_artifacts=positives,
        negative_support_artifacts=negatives,
        rank_input_digest=rank_input,
    )
    assert physical.calls == 1
    assert journal.verify().terminal_status == "success"

    forbidden = _ForbiddenTransport()
    restarted_journal = ObjectBongardTextTurnJournalTransport(
        tmp_path / "rubric-rank-journal",
        authorization_digest="sha256:" + "1" * 64,
        execution_precommit_digest="sha256:" + "2" * 64,
        task_id="bd_rubric_rank_fixture",
        turn_kind="rubric_rank",
        expected_prompt=prompt,
        expected_output_schema=schema,
        runtime=runtime,
        underlying_transport=forbidden,
    )
    restarted_ranker = ObjectBongardRubricRanker(
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        cloud_policy_cache_snapshot=policy,
        expected_cloud_policy_cache_binding="absent",
        expected_transport_source_digest=(
            object_bongard_rubric_ranker_transport_source_digest()
        ),
        model_catalog_snapshot=MODEL_CATALOG,
        no_tools_attestation=NO_TOOLS_ATTESTATION,
        minutes=3,
        executable="/private/synthetic-codex",
        transport=restarted_journal,
    )
    replayed = restarted_ranker(
        version,
        rubric_spec=spec,
        semantic_artifact=semantic,
        positive_support_artifacts=positives,
        negative_support_artifacts=negatives,
        rank_input_digest=rank_input,
    )

    assert replayed == first
    assert forbidden.calls == 0
    assert restarted_journal.reused_call_count == 1


def test_python_authority_and_clean_unlean_boundary() -> None:
    authority = object_bongard_rubric_ranker_authority_data()
    assert authority["python_is_canonical_authority"] is True
    assert authority["codex_may_rank_verified_survivors_only"] is True
    assert authority["codex_may_edit_candidate_formulas"] is False
    assert authority["lean_required"] is False
    assert authority["lean_removable"] is True
    assert authority["lean_affects_identity_ranking_or_replay"] is False

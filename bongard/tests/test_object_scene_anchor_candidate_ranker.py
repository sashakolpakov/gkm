"""Offline tests for the exact text-only anchor survivor ranker."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import re
from typing import Any, Mapping

import pytest

import bongard.object_scene_anchor_candidate_ranker as ranker_module
import bongard.transport as transport_runtime
from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_scene_anchor_candidate_ranker import (
    ObjectSceneAnchorCandidateRanker,
    ObjectSceneAnchorCandidateRankerError,
    ObjectSceneAnchorRankInput,
    ObjectSceneAnchorRankResponse,
    freeze_object_scene_anchor_rank_input,
    object_scene_anchor_candidate_ranker_output_schema,
    object_scene_anchor_candidate_ranker_prompt,
    object_scene_anchor_candidate_ranker_transport_source_digest,
)
from bongard.object_scene_anchor_version_space import (
    ObjectSceneAnchorOrientation,
    build_object_scene_anchor_support_version_space,
)
from bongard.tests.no_tools_fixture import canonical_no_tools_runtime
from bongard.tests.test_object_scene_anchor_version_space import (
    _decision,
    _language,
    _panel_evaluation,
    _panel_manifest,
)
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


@lru_cache(maxsize=1)
def _version():
    decisions = (_decision("object_0000"), _decision("object_0001"))
    target_ids = tuple(f"rank_target_{index:02d}" for index in range(6))
    contrast_ids = tuple(f"rank_contrast_{index:02d}" for index in range(6))
    manifests = {
        panel_id: _panel_manifest(index + 100, decisions)
        for index, panel_id in enumerate((*target_ids, *contrast_ids))
    }
    language = _language({panel_id: manifests[panel_id] for panel_id in target_ids})

    def target_state(_panel_id, _object_id, _witness_digest):
        return Disposition.PRESENT

    def contrast_state(_panel_id, _object_id, _witness_digest):
        return Disposition.CERTIFIED_ABSENT

    targets = tuple(
        _panel_evaluation(panel_id, manifests[panel_id], language, target_state)
        for panel_id in target_ids
    )
    contrasts = tuple(
        _panel_evaluation(panel_id, manifests[panel_id], language, contrast_state)
        for panel_id in contrast_ids
    )
    version = build_object_scene_anchor_support_version_space(
        language=language,
        orientation=ObjectSceneAnchorOrientation.SIDE0_POSITIVE,
        targets=targets,
        contrasts=contrasts,
    )
    assert len(version.survivor_candidate_digests) == 7
    return version


def _text_receipt(
    prompt: str,
    schema: Mapping[str, Any],
    payload: Mapping[str, Any],
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
    causal = transport_runtime._causal_text_input_metadata(prompt, schema_digest, binding)
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
        "thread_id": "00000000-0000-4000-8000-000000000081",
        "codex_cli_version": PINNED_CODEX_CLI_VERSION,
        "codex_launcher_digest": LAUNCHER_DIGEST,
        "cloud_config_bundle_cache_binding": "absent",
        **causal,
        "output_schema_digest": schema_digest,
        "structured_output_digest": canonical_digest(dict(payload)),
        "proposed_source_digest": "",
        "proposed_log_digest": "",
        "event_stream_digest": "e" * 64,
        "event_types": [
            "thread.started", "turn.started", "item.completed", "turn.completed"
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
    def __init__(self, aliases: tuple[str, ...] | None = None):
        self.aliases = aliases
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, prompt, schema, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        assert kwargs["model"] == MODEL
        assert kwargs["reasoning_effort"] == EFFORT
        assert kwargs["expected_launcher_digest"] == LAUNCHER_DIGEST
        aliases = tuple(
            schema["properties"]["ordered_aliases"]["items"]["enum"]
        )
        ordered = self.aliases if self.aliases is not None else tuple(reversed(aliases))
        payload = {"ordered_aliases": list(ordered)}
        return CodexStructuredResult(payload, _text_receipt(prompt, schema, payload))


def _ranker(transport) -> ObjectSceneAnchorCandidateRanker:
    return ObjectSceneAnchorCandidateRanker(
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
        model_catalog_snapshot=MODEL_CATALOG,
        no_tools_attestation=NO_TOOLS_ATTESTATION,
        minutes=3,
        executable="/private/synthetic-codex",
        transport=transport,
    )


def test_one_call_ranks_exact_survivor_permutation_and_cold_replays() -> None:
    version = _version()
    rank_input = freeze_object_scene_anchor_rank_input(version)
    transport = _Transport()
    ranker = _ranker(transport)
    response = ranker(
        version, expected_rank_input_digest=rank_input.rank_input_digest
    )

    assert transport.calls == 1
    assert response.ordered_candidate_digests == tuple(
        reversed(version.survivor_candidate_digests)
    )
    assert response.selected_candidate_digest == version.survivor_candidate_digests[-1]
    assert ObjectSceneAnchorRankResponse.from_data(response.to_data()) == response
    assert ranker.verify_response(
        response,
        version_space=version,
        expected_response_digest=response.response_digest,
        expected_rank_input_digest=rank_input.rank_input_digest,
    ) == response
    assert transport.calls == 1


def test_prompt_exposes_only_alias_anchor_kind_and_affirmative_statements() -> None:
    version = _version()
    rank_input = freeze_object_scene_anchor_rank_input(version)
    prompt = object_scene_anchor_candidate_ranker_prompt(rank_input)

    for item in rank_input.candidates:
        assert item.alias in prompt
        assert f"anchor_kind={item.anchor_kind}" in prompt
        for statement in item.affirmative_statements:
            assert statement in prompt
        assert item.candidate_digest not in prompt
        assert item.presentation_digest not in prompt
        assert not any(digest in prompt for digest in item.witness_digests)
    for panel_id in version.support_panel_ids:
        assert panel_id not in prompt
    assert version.version_space_digest not in prompt
    assert version.language.language_digest not in prompt
    assert not re.search(
        r"\b(?:panel|query|target|foil|support|contrast|side[01]|orientation|"
        r"formula|predicate|digest|pixel|image)\b",
        prompt,
        re.I,
    )
    schema = object_scene_anchor_candidate_ranker_output_schema(rank_input)
    assert schema["properties"]["ordered_aliases"]["items"]["enum"] == [
        item.alias for item in rank_input.candidates
    ]


@pytest.mark.parametrize(
    "aliases",
    (
        ("choice_000",),
        tuple("choice_000" for _ in range(7)),
        tuple(f"choice_{index:03d}" for index in range(6)) + ("choice_999",),
        tuple(f"choice_{index:03d}" for index in range(7)) + ("choice_007",),
    ),
)
def test_omitted_duplicate_foreign_or_extra_aliases_select_nothing(aliases) -> None:
    version = _version()
    rank_input = freeze_object_scene_anchor_rank_input(version)
    with pytest.raises(ObjectSceneAnchorCandidateRankerError, match="no candidate was selected"):
        _ranker(_Transport(tuple(aliases)))(
            version, expected_rank_input_digest=rank_input.rank_input_digest
        )


def test_transport_failure_and_capacity_guard_select_nothing(monkeypatch) -> None:
    version = _version()
    rank_input = freeze_object_scene_anchor_rank_input(version)

    class Failed:
        calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("offline")

    failed = Failed()
    with pytest.raises(ObjectSceneAnchorCandidateRankerError, match="no candidate was selected"):
        _ranker(failed)(version, expected_rank_input_digest=rank_input.rank_input_digest)
    assert failed.calls == 1

    monkeypatch.setattr(ranker_module, "MAX_SURVIVOR_COUNT", 6)
    with pytest.raises(ObjectSceneAnchorCandidateRankerError, match="no candidates were pruned"):
        freeze_object_scene_anchor_rank_input(version)


def test_extra_output_and_resealed_response_tamper_fail_closed() -> None:
    version = _version()
    rank_input = freeze_object_scene_anchor_rank_input(version)

    def edited(prompt, schema, **kwargs):
        aliases = schema["properties"]["ordered_aliases"]["items"]["enum"]
        payload = {
            "ordered_aliases": aliases,
            "new_rule": "model-authored content is forbidden",
        }
        return CodexStructuredResult(payload, _text_receipt(prompt, schema, payload))

    with pytest.raises(ObjectSceneAnchorCandidateRankerError, match="no candidate was selected"):
        _ranker(edited)(version, expected_rank_input_digest=rank_input.rank_input_digest)

    response = _ranker(_Transport())(
        version, expected_rank_input_digest=rank_input.rank_input_digest
    )
    tampered = deepcopy(response.to_data())
    tampered["selected_candidate_digest"] = response.ordered_candidate_digests[1]
    tampered["response_digest"] = canonical_digest(
        {key: value for key, value in tampered.items() if key != "response_digest"}
    )
    with pytest.raises(ObjectSceneAnchorCandidateRankerError, match="exact complete"):
        ObjectSceneAnchorRankResponse.from_data(tampered)


def test_rank_input_roundtrip_and_transport_source_pin() -> None:
    rank_input = freeze_object_scene_anchor_rank_input(_version())
    assert ObjectSceneAnchorRankInput.from_data(rank_input.to_data()) == rank_input
    assert object_scene_anchor_candidate_ranker_transport_source_digest() == (
        ranker_module.object_scene_anchor_candidate_ranker_transport_source_digest()
    )

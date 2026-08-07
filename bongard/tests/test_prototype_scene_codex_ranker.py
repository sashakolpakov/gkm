"""Offline tests for the bounded zero-image prototype-scene ranker."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any, Mapping

import pytest

import bongard.transport as transport_module
from bongard.canonical import canonical_digest
from bongard.prototype_scene_codex_ranker import (
    MAX_SURVIVOR_COUNT,
    PROTOTYPE_SCENE_CODEX_RANKER_PROTOCOL_ID,
    PrototypeSceneCodexRanker,
    PrototypeSceneCodexRankerError,
    prototype_scene_codex_ranker_authority_data,
    prototype_scene_codex_ranker_output_schema,
    prototype_scene_codex_ranker_prompt,
    prototype_scene_codex_ranker_protocol_digest,
    prototype_scene_codex_ranker_source_digest,
    prototype_scene_codex_ranker_transport_source_digest,
    verify_prototype_scene_codex_rank_response,
)
from bongard.prototype_scene_headless_runner import (
    PrototypeSceneHeadlessError,
    PrototypeSceneRankResponse,
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
    TEXT_STRUCTURED_INPUT_DIGEST_SCHEMA,
    CloudPolicyCacheSnapshot,
    CodexReceipt,
    CodexStructuredResult,
)


MODEL = "gpt-5.6-sol"
EFFORT = "medium"
LAUNCHER_DIGEST = "b" * 64
MODEL_CATALOG, NO_TOOLS_ATTESTATION = canonical_no_tools_runtime(LAUNCHER_DIGEST)
RANK_INPUT_DIGEST = "sha256:" + "a" * 64
SURVIVORS = (
    "prototype-scene:atom:opaque_visual_tag_0",
    "prototype-scene:atom:opaque_visual_tag_1",
    "prototype-scene:positive-and:opaque_visual_tag_0+opaque_visual_tag_1",
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
        "tool_surface_attestation_digest": (
            NO_TOOLS_ATTESTATION.attestation_digest
        ),
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
        "input_tokens": 20,
        "cached_input_tokens": 0,
        "output_tokens": 10,
        "reasoning_output_tokens": 2,
        "thread_id": "00000000-0000-4000-8000-000000000031",
        "codex_cli_version": PINNED_CODEX_CLI_VERSION,
        "codex_launcher_digest": launcher_digest,
        "cloud_config_bundle_cache_binding": "absent",
        **causal,
        "output_schema_digest": schema_digest,
        "structured_output_digest": canonical_digest(dict(payload)),
        "proposed_source_digest": "",
        "proposed_log_digest": "",
        "event_stream_digest": "c" * 64,
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


def _ranker(transport) -> PrototypeSceneCodexRanker:
    return PrototypeSceneCodexRanker(
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
        expected_cloud_policy_cache_binding="absent",
        expected_transport_source_digest=(
            prototype_scene_codex_ranker_transport_source_digest()
        ),
        model_catalog_snapshot=MODEL_CATALOG,
        no_tools_attestation=NO_TOOLS_ATTESTATION,
        transport=transport,
    )


def test_exact_text_only_permutation_is_receipted_and_cold_verified() -> None:
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
        assert RANK_INPUT_DIGEST in prompt
        assert all(item in prompt for item in SURVIVORS)
        lowered = prompt.lower()
        assert "pixel" not in lowered
        assert "label" not in lowered
        assert "query" not in lowered
        assert schema == prototype_scene_codex_ranker_output_schema(SURVIVORS)
        assert schema["properties"]["ordered_candidate_ids"]["items"][
            "enum"
        ] == list(SURVIVORS)
        payload = {"ordered_candidate_ids": list(reversed(SURVIVORS))}
        return CodexStructuredResult(payload, _receipt(prompt, schema, payload))

    ranker = _ranker(transport)
    response = ranker(SURVIVORS, RANK_INPUT_DIGEST)
    assert calls == 1
    assert response.ordered_candidate_ids == tuple(reversed(SURVIVORS))
    assert response.ranker_protocol_id == PROTOTYPE_SCENE_CODEX_RANKER_PROTOCOL_ID
    assert response.ranker_protocol_digest == (
        prototype_scene_codex_ranker_protocol_digest()
    )
    assert response.input_digest == RANK_INPUT_DIGEST
    assert response.receipt["transport_receipt"]["input_digest_schema"] == (
        TEXT_STRUCTURED_INPUT_DIGEST_SCHEMA
    )
    response.assert_matches(
        expected_input_digest=RANK_INPUT_DIGEST,
        survivor_candidate_ids=SURVIVORS,
    )
    assert PrototypeSceneRankResponse.from_data(response.to_data()) == response
    assert ranker.verify_response(
        response,
        survivor_candidate_ids=SURVIVORS,
        rank_input_digest=RANK_INPUT_DIGEST,
        expected_response_digest=response.record_digest,
    ) is response
    authority = prototype_scene_codex_ranker_authority_data()
    assert authority["python_is_canonical_authority"] is True
    assert authority["lean_present"] is False
    assert authority["lean_removable"] is True
    assert len(prototype_scene_codex_ranker_source_digest()) == 64


@pytest.mark.parametrize(
    "payload",
    (
        {"ordered_candidate_ids": list(SURVIVORS[:-1])},
        {"ordered_candidate_ids": [SURVIVORS[0], SURVIVORS[0], SURVIVORS[2]]},
        {
            "ordered_candidate_ids": [
                SURVIVORS[0],
                SURVIVORS[1],
                "prototype-scene:foreign",
            ]
        },
        {
            "ordered_candidate_ids": list(SURVIVORS),
            "explanation": "not admitted",
        },
    ),
)
def test_incomplete_duplicate_foreign_or_extra_payload_is_rejected(
    payload: Mapping[str, Any],
) -> None:
    def transport(prompt, schema, **kwargs):
        return CodexStructuredResult(
            dict(payload), _receipt(prompt, schema, payload)
        )

    with pytest.raises(PrototypeSceneCodexRankerError):
        _ranker(transport)(SURVIVORS, RANK_INPUT_DIGEST)


def test_foreign_prompt_launcher_or_payload_receipt_is_rejected() -> None:
    payload = {"ordered_candidate_ids": list(SURVIVORS)}

    def foreign_prompt(prompt, schema, **kwargs):
        return CodexStructuredResult(
            payload, _receipt(prompt + " altered", schema, payload)
        )

    with pytest.raises(PrototypeSceneCodexRankerError, match="frozen input"):
        _ranker(foreign_prompt)(SURVIVORS, RANK_INPUT_DIGEST)

    def foreign_launcher(prompt, schema, **kwargs):
        return CodexStructuredResult(
            payload,
            _receipt(prompt, schema, payload, launcher_digest="d" * 64),
        )

    with pytest.raises(PrototypeSceneCodexRankerError, match="environment"):
        _ranker(foreign_launcher)(SURVIVORS, RANK_INPUT_DIGEST)

    def foreign_payload_digest(prompt, schema, **kwargs):
        receipt = _receipt(
            prompt,
            schema,
            {"ordered_candidate_ids": list(reversed(SURVIVORS))},
        )
        return CodexStructuredResult(payload, receipt)

    with pytest.raises(PrototypeSceneCodexRankerError, match="payload"):
        _ranker(foreign_payload_digest)(SURVIVORS, RANK_INPUT_DIGEST)


def test_foreign_attestation_receipt_is_rejected_live_and_cold() -> None:
    payload = {"ordered_candidate_ids": list(SURVIVORS)}
    prompt = prototype_scene_codex_ranker_prompt(SURVIVORS, RANK_INPUT_DIGEST)
    schema = prototype_scene_codex_ranker_output_schema(SURVIVORS)
    receipt = _receipt(prompt, schema, payload).to_dict()
    foreign_attestation_digest = "e" * 64
    binding = {
        "model_catalog_digest": receipt["model_catalog_digest"],
        "transport_policy_digest": receipt["transport_policy_digest"],
        "command_digest": receipt["command_digest"],
        "effective_tool_mode": receipt["effective_tool_mode"],
        "apply_patch_tool_type": receipt["apply_patch_tool_type"],
        "tool_surface_digest": receipt["tool_surface_digest"],
        "tool_surface_attestation_digest": foreign_attestation_digest,
    }
    receipt.update(
        transport_module._causal_text_input_metadata(
            prompt, canonical_digest(schema), binding
        )
    )
    receipt["receipt_digest"] = canonical_digest(
        {key: value for key, value in receipt.items() if key != "receipt_digest"}
    )
    forged_receipt = CodexReceipt(
        **{
            **receipt,
            "event_types": tuple(receipt["event_types"]),
            "item_types": tuple(receipt["item_types"]),
        }
    )

    def transport(_prompt, _schema, **_kwargs):
        return CodexStructuredResult(payload, forged_receipt)

    with pytest.raises(PrototypeSceneCodexRankerError, match="environment"):
        _ranker(transport)(SURVIVORS, RANK_INPUT_DIGEST)

    ranker = _ranker(lambda *_args, **_kwargs: None)
    response = PrototypeSceneRankResponse.seal(
        ordered_candidate_ids=SURVIVORS,
        ranker_protocol_id=PROTOTYPE_SCENE_CODEX_RANKER_PROTOCOL_ID,
        ranker_protocol_digest=ranker.protocol_digest,
        model_id=MODEL,
        model_identity_digest=ranker.model_identity_digest,
        environment_digest=ranker.environment_digest,
        input_digest=RANK_INPUT_DIGEST,
        receipt=receipt,
    )
    with pytest.raises(PrototypeSceneCodexRankerError, match="environment"):
        verify_prototype_scene_codex_rank_response(
            response,
            survivor_candidate_ids=SURVIVORS,
            rank_input_digest=RANK_INPUT_DIGEST,
            expected_response_digest=response.record_digest,
            model=MODEL,
            reasoning_effort=EFFORT,
            expected_launcher_digest=LAUNCHER_DIGEST,
            expected_cloud_policy_cache_binding="absent",
            expected_transport_source_digest=(
                prototype_scene_codex_ranker_transport_source_digest()
            ),
            model_catalog_snapshot=MODEL_CATALOG,
            no_tools_attestation=NO_TOOLS_ATTESTATION,
        )


def test_external_source_policy_and_launcher_pins_fail_before_transport() -> None:
    calls = 0

    def forbidden(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("transport must not run")

    common = {
        "model": MODEL,
        "cloud_policy_cache_snapshot": CloudPolicyCacheSnapshot(None),
        "expected_cloud_policy_cache_binding": "absent",
        "expected_transport_source_digest": (
            prototype_scene_codex_ranker_transport_source_digest()
        ),
        "model_catalog_snapshot": MODEL_CATALOG,
        "no_tools_attestation": NO_TOOLS_ATTESTATION,
        "transport": forbidden,
    }
    with pytest.raises(PrototypeSceneCodexRankerError, match="launcher"):
        PrototypeSceneCodexRanker(
            expected_launcher_digest="bad", **common
        )
    with pytest.raises(PrototypeSceneCodexRankerError, match="transport source"):
        PrototypeSceneCodexRanker(
            expected_launcher_digest=LAUNCHER_DIGEST,
            **{**common, "expected_transport_source_digest": "0" * 64},
        )
    with pytest.raises(PrototypeSceneCodexRankerError, match="policy-cache"):
        PrototypeSceneCodexRanker(
            expected_launcher_digest=LAUNCHER_DIGEST,
            **{
                **common,
                "expected_cloud_policy_cache_binding": "sha256:" + "e" * 64,
            },
        )
    assert calls == 0


def test_bounded_inputs_and_external_response_commitment_fail_closed() -> None:
    calls = 0

    def transport(prompt, schema, **kwargs):
        nonlocal calls
        calls += 1
        payload = {"ordered_candidate_ids": list(SURVIVORS)}
        return CodexStructuredResult(payload, _receipt(prompt, schema, payload))

    ranker = _ranker(transport)
    for survivors in (
        (),
        (SURVIVORS[0], SURVIVORS[0]),
        ("contains whitespace",),
        tuple(f"id-{index}" for index in range(MAX_SURVIVOR_COUNT + 1)),
    ):
        with pytest.raises(PrototypeSceneCodexRankerError):
            ranker(survivors, RANK_INPUT_DIGEST)  # type: ignore[arg-type]
    with pytest.raises(PrototypeSceneCodexRankerError, match="rank input"):
        ranker(SURVIVORS, "not-a-digest")
    assert calls == 0

    response = ranker(SURVIVORS, RANK_INPUT_DIGEST)
    assert calls == 1
    with pytest.raises(PrototypeSceneCodexRankerError, match="external commitment"):
        ranker.verify_response(
            response,
            survivor_candidate_ids=SURVIVORS,
            rank_input_digest=RANK_INPUT_DIGEST,
            expected_response_digest="sha256:" + "0" * 64,
        )

    tampered = deepcopy(response.to_data())
    tampered["receipt"]["transport_receipt"]["prompt_digest"] = "0" * 64
    with pytest.raises(PrototypeSceneHeadlessError):
        PrototypeSceneRankResponse.from_data(tampered)

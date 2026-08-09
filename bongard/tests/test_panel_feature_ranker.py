"""Offline tests for the support-only typed panel-feature Codex ranker."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any, Mapping

import pytest

import bongard.transport as transport_runtime
from bongard.canonical import canonical_digest
from bongard.panel_feature_predicate import (
    EngineeringDisposition,
    EngineeringFeatureVersionSpace,
    EngineeringSupportTable,
    FeatureVocabulary,
)
from bongard.panel_feature_proposer import parse_panel_feature_proposer_payload
from bongard.panel_feature_ranker import (
    PANEL_FEATURE_MAX_RANK_CANDIDATES,
    PanelFeatureRankArtifact,
    PanelFeatureRankInput,
    PanelFeatureRankerError,
    cold_replay_panel_feature_rank_artifact,
    panel_feature_rank_transport_provenance,
    panel_feature_ranker_output_schema,
    panel_feature_ranker_prompt,
    rank_panel_feature_version_spaces,
)
from bongard.panel_soft_ontology import (
    ClosedCount,
    ExactSegmentCountParameters,
    FeatureFamily,
    NativeOrientation,
    PanelFeatureSpec,
    ReferenceFrame,
    SubjectScope,
)
from bongard.tests.no_tools_fixture import canonical_no_tools_runtime
from bongard.tests.test_panel_feature_task_runner import (
    _RECEIPT,
    _count_spec,
    _payload,
    _png,
    _presentation_digest,
    _task,
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
POLICY = CloudPolicyCacheSnapshot(None)
MODEL_CATALOG, NO_TOOLS_ATTESTATION = canonical_no_tools_runtime(LAUNCHER_DIGEST)


def _proposer_result(*, multiple: bool):
    task = _task()
    panels = tuple(_png(index) for index in range(12))
    payload = deepcopy(_payload(multiple=multiple))
    for ordinal, key in enumerate(sorted(payload)):
        row = payload[key]
        assert isinstance(row, dict)
        row["archival_summary"] = f"A visible closed feature number {ordinal}"
        row["archival_indicator_a"] = f"Complete drawing cue number {ordinal}"
        row["archival_indicator_b"] = f"Local visual cue number {ordinal}"
    return parse_panel_feature_proposer_payload(
        payload,
        proposer_receipt_digest=_RECEIPT,
        support_set_digest=_presentation_digest(panels),
        task_context_digest=task.record_digest.split(":", 1)[1],
    )


def _spaces(*, empty_side1: bool = False):
    proposer = _proposer_result(multiple=True)
    side0_specs = tuple(
        item.spec
        for item in proposer.nominations
        if item.native_orientation is NativeOrientation.SIDE0_POSITIVE
    )
    side1_specs = tuple(
        item.spec
        for item in proposer.nominations
        if item.native_orientation is NativeOrientation.SIDE1_POSITIVE
    )
    vocabulary = FeatureVocabulary.create(
        side0_specs=side0_specs,
        side1_specs=side1_specs,
    )
    panels = tuple(
        hashlib.sha256(f"rank-panel-{index:03d}".encode("ascii")).hexdigest()
        for index in range(12)
    )
    side0_native = set(vocabulary.side0_native_spec_digests)
    side1_native = set(vocabulary.side1_native_spec_digests)
    values: dict[tuple[str, str], EngineeringDisposition] = {}
    for index, panel in enumerate(panels):
        for spec in vocabulary.specs:
            if index < 6:
                disposition = (
                    EngineeringDisposition.MATCH
                    if spec.spec_digest in side0_native
                    else EngineeringDisposition.NONMATCH
                )
            else:
                disposition = (
                    EngineeringDisposition.MATCH
                    if spec.spec_digest in side1_native
                    else EngineeringDisposition.NONMATCH
                )
                if empty_side1 and spec.spec_digest in side1_native and index == 6:
                    disposition = EngineeringDisposition.INDETERMINATE
            values[(panel, spec.spec_digest)] = disposition
    table = EngineeringSupportTable.create(vocabulary, panels, values)
    side0 = EngineeringFeatureVersionSpace.create(
        table,
        NativeOrientation.SIDE0_POSITIVE,
        panels[:6],
        panels[6:],
    )
    side1 = EngineeringFeatureVersionSpace.create(
        table,
        NativeOrientation.SIDE1_POSITIVE,
        panels[:6],
        panels[6:],
    )
    return proposer, side0, side1


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
    causal = transport_runtime._causal_text_input_metadata(
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
        "thread_id": "00000000-0000-4000-8000-000000000196",
        "codex_cli_version": PINNED_CODEX_CLI_VERSION,
        "codex_launcher_digest": LAUNCHER_DIGEST,
        "cloud_config_bundle_cache_binding": POLICY.binding,
        **causal,
        "output_schema_digest": schema_digest,
        "structured_output_digest": canonical_digest(dict(payload)),
        "proposed_source_digest": "",
        "proposed_log_digest": "",
        "event_stream_digest": "6" * 64,
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
    def __init__(self, aliases: tuple[str, ...], *, extra_field: bool = False):
        self.aliases = aliases
        self.extra_field = extra_field
        self.calls = 0

    def __call__(self, prompt, schema, **kwargs):
        self.calls += 1
        assert kwargs["model"] == MODEL
        assert kwargs["reasoning_effort"] == EFFORT
        assert kwargs["expected_launcher_digest"] == LAUNCHER_DIGEST
        assert kwargs["cloud_policy_cache_snapshot"] == POLICY
        payload: dict[str, object] = {"ordered_aliases": list(self.aliases)}
        if self.extra_field:
            payload["invented"] = "forbidden"
        return CodexStructuredResult(payload, _text_receipt(prompt, schema, payload))


def _run(proposer, side0, side1, aliases, *, extra_field: bool = False):
    transport = _Transport(aliases, extra_field=extra_field)
    artifact = rank_panel_feature_version_spaces(
        side0,
        side1,
        proposer,
        model=MODEL,
        reasoning_effort=EFFORT,
        minutes=15,
        verbose=False,
        executable="codex",
        expected_launcher_digest=LAUNCHER_DIGEST,
        cloud_policy_cache_snapshot=POLICY,
        model_catalog_snapshot=MODEL_CATALOG,
        no_tools_attestation=NO_TOOLS_ATTESTATION,
        transport=transport,
        allow_unverified_transport=True,
    )
    return artifact, transport


def test_one_receipted_union_rank_is_private_and_selects_each_hidden_orientation():
    proposer, side0, side1 = _spaces()
    rank_input = PanelFeatureRankInput.freeze(side0, side1, proposer)
    formulas = rank_input.formula_by_alias
    desired = tuple(
        next(
            alias
            for alias in reversed(rank_input.candidate_aliases)
            if formulas[alias].native_orientation is orientation
        )
        for orientation in (
            NativeOrientation.SIDE0_POSITIVE,
            NativeOrientation.SIDE1_POSITIVE,
        )
    )
    order = desired + tuple(
        alias for alias in rank_input.candidate_aliases if alias not in desired
    )
    artifact, transport = _run(proposer, side0, side1, order)

    assert transport.calls == 1
    assert artifact.selected_formula_digests == tuple(
        formulas[alias].formula_digest for alias in desired
    )
    assert artifact.selected_side0_formula.native_orientation is (
        NativeOrientation.SIDE0_POSITIVE
    )
    assert artifact.selected_side1_formula.native_orientation is (
        NativeOrientation.SIDE1_POSITIVE
    )
    assert artifact.to_data()["logical_rank_attempts"] == 1
    assert artifact.to_data()["transport_invocations"] == 1
    assert artifact.to_data()["successful_receipt_envelopes"] == 1
    assert artifact.to_data()["cold_replay_model_calls"] == 0
    assert PanelFeatureRankArtifact.from_data(artifact.to_data()) == artifact
    assert (
        cold_replay_panel_feature_rank_artifact(
            artifact,
            side0_version_space=side0,
            side1_version_space=side1,
            proposer_result=proposer,
            expected_artifact_digest=artifact.artifact_digest,
            model=MODEL,
            reasoning_effort=EFFORT,
            expected_launcher_digest=LAUNCHER_DIGEST,
            cloud_policy_cache_snapshot=POLICY,
            model_catalog_snapshot=MODEL_CATALOG,
            no_tools_attestation=NO_TOOLS_ATTESTATION,
        )
        == artifact
    )
    assert transport.calls == 1

    prompt = panel_feature_ranker_prompt(rank_input)
    assert "closed_typed_specs" in prompt
    assert "archival_summary" in prompt
    assert "narration_executable" in prompt
    assert "side0_positive" not in prompt and "side1_positive" not in prompt
    assert "block_a" not in prompt and "block_b" not in prompt
    assert all(panel not in prompt for panel in side0.support_table.panel_digests)
    assert all(
        formula.formula_digest not in prompt for formula in formulas.values()
    )
    assert all(
        spec.spec_digest not in prompt
        for spec in side0.support_table.vocabulary.specs
    )
    assert panel_feature_ranker_output_schema(rank_input)["required"] == [
        "ordered_aliases"
    ]


def test_empty_orientation_oversize_union_and_unknown_narration_fail_before_call():
    proposer, side0, side1 = _spaces(empty_side1=True)
    with pytest.raises(PanelFeatureRankerError, match="each hidden orientation"):
        PanelFeatureRankInput.freeze(side0, side1, proposer)

    proposer, side0, side1 = _spaces()
    unknown = _proposer_result(multiple=False)
    with pytest.raises(PanelFeatureRankerError, match="unknown archival narration"):
        PanelFeatureRankInput.freeze(side0, side1, unknown)

    large_side0, large_side1 = _large_spaces()
    assert (
        len(large_side0.survivor_formulas) + len(large_side1.survivor_formulas)
        > PANEL_FEATURE_MAX_RANK_CANDIDATES
    )
    with pytest.raises(PanelFeatureRankerError, match="exceeds thirty"):
        PanelFeatureRankInput.freeze(large_side0, large_side1, proposer)


def _large_spaces():
    counts = tuple(ClosedCount)[:6]
    side0_specs = tuple(_count_spec(item) for item in counts)
    side1_specs = tuple(
        PanelFeatureSpec(
            FeatureFamily.EXACT_SEGMENT_COUNT,
            SubjectScope.WHOLE_PANEL,
            ReferenceFrame.NONE,
            ExactSegmentCountParameters(item),
        )
        for item in counts
    )
    vocabulary = FeatureVocabulary.create(
        side0_specs=side0_specs,
        side1_specs=side1_specs,
    )
    panels = tuple(
        hashlib.sha256(f"large-panel-{index:03d}".encode("ascii")).hexdigest()
        for index in range(12)
    )
    side0_native = set(vocabulary.side0_native_spec_digests)
    values = {
        (panel, spec.spec_digest): (
            EngineeringDisposition.MATCH
            if (index < 6) == (spec.spec_digest in side0_native)
            else EngineeringDisposition.NONMATCH
        )
        for index, panel in enumerate(panels)
        for spec in vocabulary.specs
    }
    table = EngineeringSupportTable.create(vocabulary, panels, values)
    return (
        EngineeringFeatureVersionSpace.create(
            table,
            NativeOrientation.SIDE0_POSITIVE,
            panels[:6],
            panels[6:],
        ),
        EngineeringFeatureVersionSpace.create(
            table,
            NativeOrientation.SIDE1_POSITIVE,
            panels[:6],
            panels[6:],
        ),
    )


def test_payload_omission_extra_duplicate_and_selected_digest_tampering_fail_closed():
    proposer, side0, side1 = _spaces()
    rank_input = PanelFeatureRankInput.freeze(side0, side1, proposer)
    aliases = rank_input.candidate_aliases

    with pytest.raises(PanelFeatureRankerError, match="exact full alias permutation"):
        _run(proposer, side0, side1, aliases[:-1])
    with pytest.raises(PanelFeatureRankerError, match="exact full alias permutation"):
        _run(proposer, side0, side1, (aliases[0],) * len(aliases))
    with pytest.raises(PanelFeatureRankerError, match="fields differ"):
        _run(proposer, side0, side1, aliases, extra_field=True)

    artifact, _transport = _run(proposer, side0, side1, tuple(reversed(aliases)))
    tampered = deepcopy(artifact.to_data())
    tampered["selected_side0_formula_digest"] = "f" * 64
    with pytest.raises(PanelFeatureRankerError):
        PanelFeatureRankArtifact.from_data(tampered)

    tampered = deepcopy(artifact.to_data())
    tampered["model_payload"]["ordered_aliases"] = list(aliases[:-1])
    with pytest.raises(PanelFeatureRankerError):
        PanelFeatureRankArtifact.from_data(tampered)

    tampered = deepcopy(artifact.to_data())
    tampered["receipt_digest"] = "e" * 64
    with pytest.raises(PanelFeatureRankerError):
        PanelFeatureRankArtifact.from_data(tampered)


def test_unverified_transport_needs_opt_in_and_production_shape_is_explicit():
    proposer, side0, side1 = _spaces()
    rank_input = PanelFeatureRankInput.freeze(side0, side1, proposer)
    injected = _Transport(rank_input.candidate_aliases)
    with pytest.raises(PanelFeatureRankerError, match="explicit engineering/test opt-in"):
        rank_panel_feature_version_spaces(
            side0,
            side1,
            proposer,
            model=MODEL,
            reasoning_effort=EFFORT,
            minutes=15,
            verbose=False,
            executable="codex",
            expected_launcher_digest=LAUNCHER_DIGEST,
            cloud_policy_cache_snapshot=POLICY,
            model_catalog_snapshot=MODEL_CATALOG,
            no_tools_attestation=NO_TOOLS_ATTESTATION,
            transport=injected,
        )
    assert injected.calls == 0
    production = panel_feature_rank_transport_provenance(
        transport_runtime.run_codex_text_structured
    )
    assert production.kind == "production_direct"
    assert production.production_transport_chain_verified is True
    assert production.benchmark_sealable is False

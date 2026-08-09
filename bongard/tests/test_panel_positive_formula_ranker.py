"""Offline tests for the isolated one-positive-formula headless ranker."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any, Mapping

import pytest

import bongard.transport as transport_runtime
from bongard.canonical import canonical_digest
from bongard.panel_feature_closed_catalog_inventory import (
    ClosedCatalogFormulaVersionSpace,
    complete_whole_panel_feature_vocabulary,
)
from bongard.panel_feature_predicate import (
    AllOf,
    EngineeringDisposition,
    EngineeringFeatureVersionSpace,
    EngineeringSupportTable,
    FeatureVocabulary,
)
from bongard.panel_positive_formula_ranker import (
    PositiveFormulaRankArtifact,
    PositiveFormulaRankInput,
    PositiveFormulaRankTransportProvenance,
    PositiveFormulaRankerError,
    cold_replay_positive_formula_rank_artifact,
    positive_formula_rank_transport_provenance,
    positive_formula_ranker_output_schema,
    positive_formula_ranker_prompt,
    rank_positive_formula_version_space,
)
from bongard.panel_soft_ontology import (
    ClosedCount,
    ComponentCountParameters,
    FeatureFamily,
    NativeOrientation,
    PanelFeatureSpec,
    ReferenceFrame,
    SubjectScope,
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
SOURCE_ADDRESS = "sha256:" + "9" * 64
POLICY = CloudPolicyCacheSnapshot(None)
MODEL_CATALOG, NO_TOOLS_ATTESTATION = canonical_no_tools_runtime(LAUNCHER_DIGEST)


def _count_spec(count: ClosedCount) -> PanelFeatureSpec:
    return PanelFeatureSpec(
        FeatureFamily.COMPONENT_COUNT,
        SubjectScope.WHOLE_PANEL,
        ReferenceFrame.NONE,
        ComponentCountParameters(count),
    )


def _space(
    orientation: NativeOrientation = NativeOrientation.SIDE0_POSITIVE,
) -> EngineeringFeatureVersionSpace:
    primary_specs = tuple(
        _count_spec(count)
        for count in (ClosedCount.TWO, ClosedCount.THREE, ClosedCount.FOUR)
    )
    other_specs = (_count_spec(ClosedCount.FIVE),)
    side0_specs, side1_specs = (
        (primary_specs, other_specs)
        if orientation is NativeOrientation.SIDE0_POSITIVE
        else (other_specs, primary_specs)
    )
    vocabulary = FeatureVocabulary.create(
        side0_specs=side0_specs,
        side1_specs=side1_specs,
    )
    panels = tuple(
        hashlib.sha256(f"positive-rank-panel-{index:03d}".encode("ascii")).hexdigest()
        for index in range(12)
    )
    primary_digests = {item.spec_digest for item in primary_specs}
    primary_order = tuple(item.spec_digest for item in primary_specs)
    native_indices = (
        set(range(6))
        if orientation is NativeOrientation.SIDE0_POSITIVE
        else set(range(6, 12))
    )
    counter_indices = tuple(index for index in range(12) if index not in native_indices)
    counter_position = {
        panel_index: position
        for position, panel_index in enumerate(counter_indices)
    }
    values: dict[tuple[str, str], EngineeringDisposition] = {}
    for panel_index, panel in enumerate(panels):
        for spec in vocabulary.specs:
            digest = spec.spec_digest
            if digest not in primary_digests:
                disposition = EngineeringDisposition.NONMATCH
            elif panel_index in native_indices:
                disposition = EngineeringDisposition.MATCH
            else:
                position = counter_position[panel_index]
                if digest == primary_order[0]:
                    disposition = (
                        EngineeringDisposition.MATCH
                        if position < 3
                        else EngineeringDisposition.NONMATCH
                    )
                elif digest == primary_order[1]:
                    disposition = (
                        EngineeringDisposition.NONMATCH
                        if position < 3
                        else EngineeringDisposition.MATCH
                    )
                else:
                    disposition = EngineeringDisposition.NONMATCH
            values[(panel, digest)] = disposition
    table = EngineeringSupportTable.create(vocabulary, panels, values)
    return EngineeringFeatureVersionSpace.create(
        table,
        orientation,
        panels[:6],
        panels[6:],
    )


def _closed_space(
    orientation: NativeOrientation,
) -> ClosedCatalogFormulaVersionSpace:
    vocabulary = complete_whole_panel_feature_vocabulary()
    panels = tuple(
        hashlib.sha256(f"closed-rank-panel-{index:03d}".encode("ascii")).hexdigest()
        for index in range(12)
    )
    selected = tuple(item.spec_digest for item in vocabulary.specs[:3])
    native_indices = (
        set(range(6))
        if orientation is NativeOrientation.SIDE0_POSITIVE
        else set(range(6, 12))
    )
    counter_indices = tuple(index for index in range(12) if index not in native_indices)
    counter_position = {
        panel_index: position
        for position, panel_index in enumerate(counter_indices)
    }
    values: dict[tuple[str, str], EngineeringDisposition] = {}
    for panel_index, panel in enumerate(panels):
        for spec in vocabulary.specs:
            if spec.spec_digest not in selected:
                disposition = EngineeringDisposition.INDETERMINATE
            elif panel_index in native_indices:
                disposition = EngineeringDisposition.MATCH
            else:
                position = counter_position[panel_index]
                if spec.spec_digest == selected[0]:
                    disposition = (
                        EngineeringDisposition.MATCH
                        if position < 3
                        else EngineeringDisposition.NONMATCH
                    )
                elif spec.spec_digest == selected[1]:
                    disposition = (
                        EngineeringDisposition.NONMATCH
                        if position < 3
                        else EngineeringDisposition.MATCH
                    )
                else:
                    disposition = EngineeringDisposition.NONMATCH
            values[(panel, spec.spec_digest)] = disposition
    table = EngineeringSupportTable.create(vocabulary, panels, values)
    return ClosedCatalogFormulaVersionSpace.create(
        table,
        orientation,
        panels[:6],
        panels[6:],
    )


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
        "thread_id": "00000000-0000-4000-8000-000000000271",
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
        self.prompt = ""
        self.schema: Mapping[str, Any] = {}

    def __call__(self, prompt, schema, **kwargs):
        self.calls += 1
        self.prompt = prompt
        self.schema = schema
        assert kwargs["model"] == MODEL
        assert kwargs["reasoning_effort"] == EFFORT
        assert kwargs["expected_launcher_digest"] == LAUNCHER_DIGEST
        assert kwargs["cloud_policy_cache_snapshot"] == POLICY
        payload: dict[str, object] = {"ordered_aliases": list(self.aliases)}
        if self.extra_field:
            payload["invented"] = "forbidden"
        return CodexStructuredResult(payload, _text_receipt(prompt, schema, payload))


def _run(
    space: EngineeringFeatureVersionSpace | ClosedCatalogFormulaVersionSpace,
    aliases: tuple[str, ...],
    *,
    extra_field: bool = False,
):
    transport = _Transport(aliases, extra_field=extra_field)
    artifact = rank_positive_formula_version_space(
        space,
        source_survivor_inventory_address=SOURCE_ADDRESS,
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


@pytest.mark.parametrize("orientation", tuple(NativeOrientation))
def test_exact_closed_catalog_primary_space_is_accepted_without_orientation_assumption(
    orientation: NativeOrientation,
):
    space = _closed_space(orientation)
    rank_input = PositiveFormulaRankInput.freeze(
        space,
        source_survivor_inventory_address=SOURCE_ADDRESS,
    )
    artifact, transport = _run(space, tuple(reversed(rank_input.candidate_aliases)))

    assert transport.calls == 1
    assert artifact.rank_input.source_positive_version_space_digest == (
        space.version_space_digest
    )
    selected = artifact.resolve_selected_all_of(
        space,
        source_survivor_inventory_address=SOURCE_ADDRESS,
    )
    assert selected.native_orientation is orientation
    assert selected.formula_digest == artifact.selected_formula_digest


@pytest.mark.parametrize("orientation", tuple(NativeOrientation))
def test_one_positive_rank_is_private_selects_first_and_cold_replays_zero_call(
    orientation: NativeOrientation,
):
    space = _space(orientation)
    rank_input = PositiveFormulaRankInput.freeze(
        space,
        source_survivor_inventory_address=SOURCE_ADDRESS,
    )
    assert len(space.survivor_formulas) == 4
    assert rank_input.survivor_formula_digests == tuple(
        sorted(space.survivor_formula_digests)
    )
    order = tuple(reversed(rank_input.candidate_aliases))
    artifact, transport = _run(space, order)

    assert transport.calls == 1
    assert artifact.selected_formula_digest == (
        rank_input.candidate_by_alias[order[0]].formula_digest
    )
    assert artifact.selected_formula == rank_input.candidate_by_alias[order[0]]
    assert isinstance(
        artifact.resolve_selected_all_of(
            space, source_survivor_inventory_address=SOURCE_ADDRESS
        ),
        AllOf,
    )
    assert artifact.source_positive_version_space_digest == space.version_space_digest
    assert artifact.artifact_address == "sha256:" + artifact.artifact_digest
    assert artifact.benchmark_sealable is False
    assert artifact.logical_rank_attempts == 1
    assert artifact.transport_invocations == 1
    assert artifact.successful_receipt_envelopes == 1
    data = artifact.to_data()
    assert data["logical_rank_attempts"] == 1
    assert data["transport_invocations"] == 1
    assert data["successful_receipt_envelopes"] == 1
    assert data["python_selections"] == 1
    assert data["cold_replay_model_calls"] == 0
    assert data["negative_formula_present"] is False
    assert data["formula_negation_allowed"] is False
    assert data["polarity_flip_allowed"] is False
    assert data["lean_present"] is False
    assert PositiveFormulaRankArtifact.from_data(data) == artifact

    replayed = cold_replay_positive_formula_rank_artifact(
        artifact,
        positive_version_space=space,
        source_survivor_inventory_address=SOURCE_ADDRESS,
        expected_artifact_address=artifact.artifact_address,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        cloud_policy_cache_snapshot=POLICY,
        model_catalog_snapshot=MODEL_CATALOG,
        no_tools_attestation=NO_TOOLS_ATTESTATION,
    )
    assert replayed == artifact
    assert transport.calls == 1

    prompt = positive_formula_ranker_prompt(rank_input)
    assert prompt == transport.prompt
    assert "typed_formula_wire" in prompt
    assert "support_profile" in prompt
    assert "concept_examples" in prompt and "counterexamples" in prompt
    assert "side0_positive" not in prompt and "side1_positive" not in prompt
    assert "task_id" not in prompt and "panel_id" not in prompt
    assert SOURCE_ADDRESS not in prompt
    assert all(panel not in prompt for panel in space.support_table.panel_digests)
    assert all(
        formula.formula_digest not in prompt for formula in space.formulas
    )
    assert all(
        spec.spec_digest not in prompt for spec in space.support_table.vocabulary.specs
    )
    assert positive_formula_ranker_output_schema(rank_input)["required"] == [
        "ordered_aliases"
    ]


def test_hidden_source_custody_cannot_change_model_visible_rank_prompt():
    space = _space(NativeOrientation.SIDE0_POSITIVE)
    first = PositiveFormulaRankInput.freeze(
        space, source_survivor_inventory_address="sha256:" + "7" * 64
    )
    second = PositiveFormulaRankInput.freeze(
        space, source_survivor_inventory_address="sha256:" + "8" * 64
    )

    assert first.rank_input_digest != second.rank_input_digest
    assert first.rank_input_address != second.rank_input_address
    assert first.candidate_view_digest == second.candidate_view_digest
    assert first.candidate_view_address == second.candidate_view_address
    assert positive_formula_ranker_prompt(first) == positive_formula_ranker_prompt(
        second
    )
    assert first.source_survivor_inventory_address not in positive_formula_ranker_prompt(
        first
    )
    assert second.source_survivor_inventory_address not in positive_formula_ranker_prompt(
        second
    )


def test_full_permutation_is_mandatory_and_bad_payload_gets_no_retry():
    space = _space()
    rank_input = PositiveFormulaRankInput.freeze(
        space, source_survivor_inventory_address=SOURCE_ADDRESS
    )
    aliases = rank_input.candidate_aliases

    short = _Transport(aliases[:-1])
    with pytest.raises(PositiveFormulaRankerError, match="exact full alias permutation"):
        rank_positive_formula_version_space(
            space,
            source_survivor_inventory_address=SOURCE_ADDRESS,
            model=MODEL,
            reasoning_effort=EFFORT,
            minutes=15,
            verbose=False,
            executable="codex",
            expected_launcher_digest=LAUNCHER_DIGEST,
            cloud_policy_cache_snapshot=POLICY,
            model_catalog_snapshot=MODEL_CATALOG,
            no_tools_attestation=NO_TOOLS_ATTESTATION,
            transport=short,
            allow_unverified_transport=True,
        )
    assert short.calls == 1

    duplicate = _Transport((aliases[0],) * len(aliases))
    with pytest.raises(PositiveFormulaRankerError, match="exact full alias permutation"):
        rank_positive_formula_version_space(
            space,
            source_survivor_inventory_address=SOURCE_ADDRESS,
            model=MODEL,
            reasoning_effort=EFFORT,
            minutes=15,
            verbose=False,
            executable="codex",
            expected_launcher_digest=LAUNCHER_DIGEST,
            cloud_policy_cache_snapshot=POLICY,
            model_catalog_snapshot=MODEL_CATALOG,
            no_tools_attestation=NO_TOOLS_ATTESTATION,
            transport=duplicate,
            allow_unverified_transport=True,
        )
    assert duplicate.calls == 1

    extra = _Transport(aliases, extra_field=True)
    with pytest.raises(PositiveFormulaRankerError, match="fields differ"):
        rank_positive_formula_version_space(
            space,
            source_survivor_inventory_address=SOURCE_ADDRESS,
            model=MODEL,
            reasoning_effort=EFFORT,
            minutes=15,
            verbose=False,
            executable="codex",
            expected_launcher_digest=LAUNCHER_DIGEST,
            cloud_policy_cache_snapshot=POLICY,
            model_catalog_snapshot=MODEL_CATALOG,
            no_tools_attestation=NO_TOOLS_ATTESTATION,
            transport=extra,
            allow_unverified_transport=True,
        )
    assert extra.calls == 1


def test_injected_transport_needs_opt_in_and_sealable_kind_is_distinct():
    space = _space()
    rank_input = PositiveFormulaRankInput.freeze(
        space, source_survivor_inventory_address=SOURCE_ADDRESS
    )
    injected = _Transport(rank_input.candidate_aliases)
    with pytest.raises(PositiveFormulaRankerError, match="explicit engineering/test opt-in"):
        rank_positive_formula_version_space(
            space,
            source_survivor_inventory_address=SOURCE_ADDRESS,
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
    direct = positive_formula_rank_transport_provenance(
        transport_runtime.run_codex_text_structured
    )
    assert direct.kind == "production_direct"
    assert direct.production_transport_chain_verified is True
    assert direct.benchmark_sealable is False
    journal = PositiveFormulaRankTransportProvenance.create(
        "production_exactly_once_journal"
    )
    assert journal.production_transport_chain_verified is True
    assert journal.benchmark_sealable is True


def test_artifact_profile_selection_receipt_and_source_tampering_fail_closed():
    space = _space()
    rank_input = PositiveFormulaRankInput.freeze(
        space, source_survivor_inventory_address=SOURCE_ADDRESS
    )
    artifact, _transport = _run(space, tuple(reversed(rank_input.candidate_aliases)))

    tampered = deepcopy(artifact.to_data())
    tampered["selected_formula_digest"] = "f" * 64
    with pytest.raises(PositiveFormulaRankerError):
        PositiveFormulaRankArtifact.from_data(tampered)

    tampered = deepcopy(artifact.to_data())
    tampered["selected_formula"]["support_profile"]["concept_examples"][0] = "nonmatch"
    with pytest.raises(PositiveFormulaRankerError):
        PositiveFormulaRankArtifact.from_data(tampered)

    tampered = deepcopy(artifact.to_data())
    tampered["receipt_digest"] = "e" * 64
    with pytest.raises(PositiveFormulaRankerError):
        PositiveFormulaRankArtifact.from_data(tampered)

    with pytest.raises(PositiveFormulaRankerError, match="externally verified"):
        cold_replay_positive_formula_rank_artifact(
            artifact,
            positive_version_space=space,
            source_survivor_inventory_address="sha256:" + "8" * 64,
            expected_artifact_address=artifact.artifact_address,
            model=MODEL,
            reasoning_effort=EFFORT,
            expected_launcher_digest=LAUNCHER_DIGEST,
            cloud_policy_cache_snapshot=POLICY,
            model_catalog_snapshot=MODEL_CATALOG,
            no_tools_attestation=NO_TOOLS_ATTESTATION,
        )

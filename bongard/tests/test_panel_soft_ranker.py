"""Offline tests for the support-only panel-soft Codex ranker."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import pytest

import bongard.transport as transport_runtime
from bongard.canonical import canonical_digest
from bongard.object_bongard_turn_journal import (
    ObjectBongardTextTurnJournalTransport,
    ObjectBongardTurnRuntime,
)
from bongard.panel_soft_observer import aggregate_panel_soft_observer_artifacts
from bongard.panel_soft_predicate import (
    PanelSoftAtom,
    PanelSoftEngineeringVersionSpace,
    PanelSoftObservationTable,
    PanelSoftVocabulary,
)
from bongard.panel_soft_ranker import (
    PanelSoftRankArtifact,
    PanelSoftRankInput,
    PanelSoftRankerError,
    panel_soft_rank_transport_provenance,
    panel_soft_ranker_output_schema,
    panel_soft_ranker_prompt,
    rank_panel_soft_version_space,
    verify_panel_soft_rank_artifact,
)
from bongard.tests.no_tools_fixture import canonical_no_tools_runtime
from bongard.prototype_scene_observer import prototype_scene_transport_source_digest
from bongard.tests.test_panel_soft_engineering_task_runner import (
    _fixture,
)
from bongard.tests.test_panel_soft_predicate import (
    _contract,
    _separating_rows,
    _support_panels,
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


def _space():
    task, proposer, _support_map, support_artifacts, _observe = _fixture()
    assert proposer.vocabulary is not None
    table = aggregate_panel_soft_observer_artifacts(
        support_artifacts,
        ordered_panel_commitments=tuple(
            (item.panel_id, item.panel_png_digest) for item in support_artifacts
        ),
        expected_vocabulary=proposer.vocabulary,
        expected_contract=support_artifacts[0].contract,
    )
    return task, PanelSoftEngineeringVersionSpace.create(
        table, task.side_0_support_panel_ids, task.side_1_support_panel_ids
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
        "thread_id": "00000000-0000-4000-8000-000000000097",
        "codex_cli_version": PINNED_CODEX_CLI_VERSION,
        "codex_launcher_digest": LAUNCHER_DIGEST,
        "cloud_config_bundle_cache_binding": POLICY.binding,
        **causal,
        "output_schema_digest": schema_digest,
        "structured_output_digest": canonical_digest(dict(payload)),
        "proposed_source_digest": "",
        "proposed_log_digest": "",
        "event_stream_digest": "9" * 64,
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
    def __init__(self, aliases: tuple[str, ...]):
        self.aliases = aliases
        self.calls = 0

    def __call__(self, prompt, schema, **kwargs):
        self.calls += 1
        assert kwargs["model"] == MODEL
        assert kwargs["reasoning_effort"] == EFFORT
        assert kwargs["expected_launcher_digest"] == LAUNCHER_DIGEST
        assert kwargs["cloud_policy_cache_snapshot"] == POLICY
        payload = {"ordered_aliases": list(self.aliases)}
        return CodexStructuredResult(payload, _text_receipt(prompt, schema, payload))


def _run(space, aliases):
    transport = _Transport(aliases)
    artifact = rank_panel_soft_version_space(
        space,
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


def test_one_receipted_union_rank_selects_first_survivor_per_hidden_orientation() -> None:
    task, space = _space()
    rank_input = PanelSoftRankInput.freeze(space)
    formulas = rank_input.formula_by_alias
    desired = []
    for orientation in ("side0_positive", "side1_positive"):
        desired.append(
            max(
                (alias for alias, formula in formulas.items() if formula.orientation == orientation),
                key=lambda alias: len(formulas[alias].atom_digests),
            )
        )
    order = tuple(desired) + tuple(
        alias for alias in rank_input.candidate_aliases if alias not in desired
    )
    artifact, transport = _run(space, order)

    assert transport.calls == 1
    assert artifact.transport_provenance.kind == "injected_unverified"
    assert artifact.transport_provenance.benchmark_sealable is False
    assert artifact.to_data()["successful_receipt_envelopes"] == 1
    assert artifact.to_data()["cold_replay_physical_model_call_authenticated"] is False
    assert artifact.selected_formula_digests == tuple(
        formulas[alias].formula_digest for alias in desired
    )
    assert all(
        len(formulas[alias].atom_digests) == 4 for alias in desired
    )
    assert PanelSoftRankArtifact.from_data(artifact.to_data()) == artifact
    assert verify_panel_soft_rank_artifact(
        artifact,
        version_space=space,
        expected_artifact_digest=artifact.artifact_digest,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        cloud_policy_cache_snapshot=POLICY,
        model_catalog_snapshot=MODEL_CATALOG,
        no_tools_attestation=NO_TOOLS_ATTESTATION,
    ) == artifact
    with pytest.raises(PanelSoftRankerError):
        verify_panel_soft_rank_artifact(
            artifact,
            version_space=space,
            expected_artifact_digest=artifact.artifact_digest,
            model=MODEL,
            reasoning_effort=EFFORT,
            expected_launcher_digest=LAUNCHER_DIGEST,
            cloud_policy_cache_snapshot=POLICY,
            model_catalog_snapshot=MODEL_CATALOG,
            no_tools_attestation=NO_TOOLS_ATTESTATION,
            require_benchmark_sealable=True,
        )
    assert transport.calls == 1

    prompt = panel_soft_ranker_prompt(rank_input)
    assert rank_input.to_data()["model_visible_candidate_fields"] == [
        "opaque_alias",
        "lexically_filtered_atom_text",
        "lexically_filtered_witness_texts",
    ]
    assert "lexically_filtered_atom_text" in prompt
    assert "affirmative_description" not in prompt
    assert task.task_id not in prompt
    assert "side0_positive" not in prompt and "side1_positive" not in prompt
    assert all(panel_id not in prompt for panel_id in space.support_table.panel_ids)
    assert panel_soft_ranker_output_schema(rank_input)["required"] == [
        "ordered_aliases"
    ]


def test_unverified_transport_requires_explicit_opt_in() -> None:
    _task, space = _space()
    rank_input = PanelSoftRankInput.freeze(space)
    injected = _Transport(rank_input.candidate_aliases)
    with pytest.raises(PanelSoftRankerError, match="explicit engineering/test opt-in"):
        rank_panel_soft_version_space(
            space,
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
    production = panel_soft_rank_transport_provenance(
        transport_runtime.run_codex_text_structured
    )
    assert production.kind == "production_direct"
    assert production.production_transport_chain_verified is True
    assert production.benchmark_sealable is False


def test_exactly_once_journal_over_exact_transport_is_benchmark_sealable(
    tmp_path,
) -> None:
    runtime = ObjectBongardTurnRuntime(
        model=MODEL,
        reasoning_effort=EFFORT,
        minutes=15,
        verbose=False,
        executable="codex",
        cloud_policy_cache_snapshot=POLICY,
        model_catalog_snapshot=MODEL_CATALOG,
        expected_launcher_digest=LAUNCHER_DIGEST,
        no_tools_attestation=NO_TOOLS_ATTESTATION,
        transport_source_digest=prototype_scene_transport_source_digest(),
    )
    journal = ObjectBongardTextTurnJournalTransport(
        tmp_path / "rank-turn",
        authorization_digest="sha256:" + "1" * 64,
        execution_precommit_digest="sha256:" + "2" * 64,
        task_id="bd_rank_transport_0000",
        turn_kind="support_rank",
        expected_prompt="rank prompt",
        expected_output_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        runtime=runtime,
        underlying_transport=transport_runtime.run_codex_text_structured,
    )
    provenance = panel_soft_rank_transport_provenance(journal)
    assert provenance.kind == "production_exactly_once_journal"
    assert provenance.production_transport_chain_verified is True
    assert provenance.benchmark_sealable is True
    assert provenance.to_data()[
        "transport_history_authenticated_by_rank_artifact_alone"
    ] is False
    assert provenance.to_data()[
        "benchmark_requires_external_typed_journal_terminal"
    ] is True


def test_ranker_uses_the_exact_upstream_soft_prose_grammar() -> None:
    proposer_digest = "a" * 64
    rows = (
        (
            "side0_positive",
            "The panel carries oblique strokes",
            "Several strokes lean across the figure",
            "Sharp corners meet along diagonal directions",
        ),
        (
            "side0_positive",
            "A loop sits on one side",
            "A rounded enclosure occupies the left region",
            "One curved outline closes around a central area",
        ),
        (
            "side1_positive",
            "Diagonal orientation shapes the drawing",
            "Slanted marks define the overall figure",
            "Angled corners organize the complete form",
        ),
        (
            "side1_positive",
            "Smooth bends define the complete figure",
            "Curved turns shape the outer contour",
            "A flowing stroke changes direction gradually",
        ),
    )
    vocabulary = PanelSoftVocabulary.create(
        tuple(
            PanelSoftAtom.create(
                atom_id=f"atom_{index:04d}",
                orientation=orientation,
                phrase=phrase,
                witnesses=(witness_a, witness_b),
                proposer_artifact_digest=proposer_digest,
            )
            for index, (orientation, phrase, witness_a, witness_b) in enumerate(rows)
        )
    )
    table = PanelSoftObservationTable.create(
        vocabulary=vocabulary,
        contract=_contract(vocabulary),
        panels=_support_panels(),
        raw_verdict_rows=_separating_rows(),
    )
    space = PanelSoftEngineeringVersionSpace.create(
        table, table.panel_ids[:6], table.panel_ids[6:]
    )
    prompt = panel_soft_ranker_prompt(PanelSoftRankInput.freeze(space))
    for _orientation, phrase, _witness_a, _witness_b in rows:
        assert phrase in prompt


def test_duplicate_alias_and_artifact_tampering_fail_closed() -> None:
    _task, space = _space()
    rank_input = PanelSoftRankInput.freeze(space)
    bad_order = (rank_input.candidate_aliases[0],) * len(rank_input.candidate_aliases)
    with pytest.raises(PanelSoftRankerError):
        _run(space, bad_order)

    artifact, _transport = _run(space, tuple(reversed(rank_input.candidate_aliases)))
    tampered = deepcopy(artifact.to_data())
    tampered["selected_side0_formula_digest"] = tampered[
        "selected_side1_formula_digest"
    ]
    with pytest.raises(PanelSoftRankerError):
        PanelSoftRankArtifact.from_data(tampered)

    boolean_counter = deepcopy(artifact.to_data())
    boolean_counter["logical_rank_attempts"] = True
    with pytest.raises(PanelSoftRankerError):
        PanelSoftRankArtifact.from_data(boolean_counter)

    boolean_authority = deepcopy(artifact.to_data())
    boolean_authority["engineering_only"] = 1
    with pytest.raises(PanelSoftRankerError):
        PanelSoftRankArtifact.from_data(boolean_authority)

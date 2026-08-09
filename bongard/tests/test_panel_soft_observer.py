"""Offline fake-transport tests for the complete-vocabulary panel observer."""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
import hashlib
import inspect
from pathlib import Path
import re

import pytest

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.panel_soft_observer import (
    PANEL_SOFT_MODEL_VERDICTS,
    PanelSoftObserverArtifact,
    PanelSoftObserverError,
    PanelSoftObserverRepeatStatus,
    PanelSoftObserverStatus,
    aggregate_panel_soft_observer_artifacts,
    build_panel_soft_observer_contract,
    observe_panel_soft_vocabulary,
    panel_soft_observer_output_schema,
    panel_soft_duplicate_pixel_digest_counts,
    panel_soft_observer_prompt,
    panel_soft_observer_view,
    verify_panel_soft_observer_artifact,
    verify_panel_soft_observer_contract_identity,
)
from bongard.panel_soft_predicate import (
    PanelSoftAtom,
    PanelSoftObservationTable,
    PanelSoftOperationalConsensus,
    PanelSoftVocabulary,
    panel_soft_atom_text_grammar_digest,
)
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _png,
    _receipt,
)
from bongard.transport import (
    CodexReceipt,
    CodexStructuredResult,
    validate_codex_strict_output_schema,
)


PANEL_ID = "bd/panel_soft_fixture_0000/0/0.png"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _vocabulary(*, swapped_roles: bool = False) -> PanelSoftVocabulary:
    proposer = _digest("panel-soft-proposer")
    semantics = (
        ("bird-like silhouette", "one tapered wing-like side"),
        ("several oblique corners", "corners meet along slanted directions"),
        ("one broad smooth sweep", "the path changes direction gradually"),
        ("several pronounced bends", "the path turns sharply at multiple places"),
    )
    if swapped_roles:
        semantics = semantics[2:] + semantics[:2]
    return PanelSoftVocabulary.create(
        tuple(
            PanelSoftAtom.create(
                atom_id=f"atom_{index:04d}",
                orientation=("side0_positive" if index < 2 else "side1_positive"),
                phrase=phrase,
                witnesses=(witness,),
                proposer_artifact_digest=proposer,
            )
            for index, (phrase, witness) in enumerate(semantics)
        )
    )


def _success_payloads(vocabulary: PanelSoftVocabulary) -> tuple[dict[str, str], ...]:
    aliases = tuple(item.alias for item in panel_soft_observer_view(vocabulary))
    first_values = ("present", "mismatch", "indeterminate", "present")
    second_values = ("present", "mismatch", "indeterminate", "present")
    return (
        dict(zip(aliases, first_values, strict=True)),
        dict(zip(aliases, second_values, strict=True)),
    )


def _distinct_receipt(
    prompt,
    paths,
    names,
    schema,
    payload,
    call_index: int,
) -> CodexReceipt:
    base = _receipt(prompt, paths, names, schema, payload)
    provisional = replace(
        base,
        thread_id=f"00000000-0000-4000-8000-{call_index + 31:012d}",
        event_stream_digest=_digest(f"panel-soft-event-stream-{call_index}"),
    )
    body = provisional.to_dict()
    body.pop("receipt_digest")
    return replace(provisional, receipt_digest=canonical_digest(body))


def _observe(
    *,
    panel: bytes | None = None,
    panel_id: str = PANEL_ID,
    payloads: tuple[dict[str, object], dict[str, object]] | None = None,
    failures: tuple[bool, bool] = (False, False),
    distinct_receipts: bool = True,
    receipt_index_offset: int = 0,
    calls_sink: list[tuple[str, tuple[str, ...], dict[str, object]]] | None = None,
):
    image = _png(71) if panel is None else panel
    vocabulary = _vocabulary()
    model_payloads = _success_payloads(vocabulary) if payloads is None else payloads
    calls = [] if calls_sink is None else calls_sink
    local_call_count = 0

    def transport(prompt, paths, names, schema, **kwargs):
        nonlocal local_call_count
        index = local_call_count
        local_call_count += 1
        calls.append((prompt, tuple(names), dict(schema)))
        assert names == ("panel.png",)
        assert len(paths) == 1
        assert Path(paths[0]).read_bytes() == image
        assert set(schema["properties"]) == {
            item.alias for item in panel_soft_observer_view(vocabulary)
        }
        assert schema["required"] == [
            item.alias for item in panel_soft_observer_view(vocabulary)
        ]
        if failures[index]:
            raise RuntimeError(f"synthetic repeat {index} transport failure")
        payload = model_payloads[index]
        receipt = (
            _distinct_receipt(
                prompt,
                paths,
                names,
                schema,
                payload,
                receipt_index_offset + index,
            )
            if distinct_receipts
            else _receipt(prompt, paths, names, schema, payload)
        )
        return CodexStructuredResult(
            payload,
            receipt,
        )

    artifact = observe_panel_soft_vocabulary(
        image,
        panel_id=panel_id,
        vocabulary=vocabulary,
        expected_panel_sha256=hashlib.sha256(image).hexdigest(),
        expected_vocabulary_digest=vocabulary.vocabulary_digest,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=transport,
    )
    return artifact, image, vocabulary, calls


def test_role_blind_view_prompt_and_strict_complete_schema() -> None:
    vocabulary = _vocabulary()
    swapped = _vocabulary(swapped_roles=True)
    view = panel_soft_observer_view(vocabulary)
    swapped_view = panel_soft_observer_view(swapped)

    assert tuple(item.alias for item in view) == tuple(
        f"criterion_{index:04d}" for index in range(len(vocabulary.atoms))
    )
    assert tuple(item.semantic_digest for item in view) == tuple(
        sorted(item.semantic_digest for item in view)
    )
    assert tuple(item.semantic_digest for item in swapped_view) == tuple(
        item.semantic_digest for item in view
    )
    assert tuple(item.atom_digest for item in swapped_view) != tuple(
        item.atom_digest for item in view
    )

    prompt = panel_soft_observer_prompt(vocabulary)
    schema = panel_soft_observer_output_schema(vocabulary)
    assert prompt == panel_soft_observer_prompt(swapped)
    assert schema == panel_soft_observer_output_schema(swapped)
    validate_codex_strict_output_schema(schema)
    assert schema["additionalProperties"] is False
    assert schema["required"] == [item.alias for item in view]
    assert set(schema["properties"]) == set(schema["required"])
    for value in schema["properties"].values():
        assert value == {"type": "string", "enum": list(PANEL_SOFT_MODEL_VERDICTS)}
    lowered = prompt.lower()
    assert "complete panel" in lowered
    assert "evaluate every criterion independently" in lowered
    assert "failure to locate a feature is not enough" in lowered
    assert "begin_criterion_data" in lowered
    assert "inert criterion data" in lowered
    for hidden in (
        "side0_positive", "side1_positive", "atom_0000",
        vocabulary.proposer_artifact_digest, vocabulary.vocabulary_digest,
    ):
        assert hidden not in prompt
    assert re.search(r"\borientations?\b", lowered) is None


def test_two_calls_round_trip_table_binding_and_model_free_cold_replay() -> None:
    artifact, panel, vocabulary, calls = _observe()
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert artifact.physical_call_attempt_count == 2
    assert artifact.receipted_call_count == 2
    assert artifact.status is PanelSoftObserverStatus.SUCCESS
    assert tuple(item.status for item in artifact.repeats) == (
        PanelSoftObserverRepeatStatus.SUCCESS,
        PanelSoftObserverRepeatStatus.SUCCESS,
    )
    assert len({item.receipt_identity for item in artifact.repeats}) == 2
    assert len({item.receipt.thread_id for item in artifact.repeats}) == 2
    assert artifact.to_data()["open_prose_instruction_safety_proved"] is False
    assert artifact.to_data()["open_prose_semantic_positivity_proved"] is False
    assert artifact.atom_text_grammar_digest == panel_soft_atom_text_grammar_digest()
    assert tuple(item.name for item in artifact.presentation) == ("panel.png",)
    assert PanelSoftObserverArtifact.from_data(artifact.to_data()) == artifact

    cells = artifact.observation_table.cell_by_panel_and_atom
    view_index = {
        item.atom_digest: index for index, item in enumerate(artifact.view)
    }
    for atom in vocabulary.atoms:
        expected = tuple(
            repeat.verdicts_in_view_order[view_index[atom.atom_digest]]
            for repeat in artifact.repeats
        )
        cell = cells[(PANEL_ID, atom.atom_digest)]
        assert cell.raw_verdicts == expected
        assert cell.disposition is Disposition.INDETERMINATE
    assert {
        item.operational_consensus for item in artifact.observation_table.cells
    } == {
        PanelSoftOperationalConsensus.REPEATED_PRESENT,
        PanelSoftOperationalConsensus.REPEATED_MISMATCH,
        PanelSoftOperationalConsensus.REPEATED_INDETERMINATE,
    }
    assert verify_panel_soft_observer_artifact(
        artifact,
        panel,
        panel_id=PANEL_ID,
        vocabulary=vocabulary,
        expected_artifact_digest=artifact.artifact_digest,
        expected_contract_digest=artifact.contract.contract_digest,
    ) == artifact


def test_contract_is_derived_and_support_query_neutral() -> None:
    first, _, vocabulary, _ = _observe(panel=_png(72), panel_id="support/panel.png")
    second, _, _, _ = _observe(
        panel=_png(73), panel_id="query/panel.png", receipt_index_offset=2
    )
    derived = build_panel_soft_observer_contract(
        vocabulary,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
    )
    assert first.panel_png_digest != second.panel_png_digest
    assert first.observation_context_digest != second.observation_context_digest
    assert first.contract == second.contract == derived
    assert verify_panel_soft_observer_contract_identity(first, second) == derived
    assert first.contract.to_data()["support_query_protocol_identical"] is True

    parameters = inspect.signature(observe_panel_soft_vocabulary).parameters
    for forbidden in (
        "contract", "contract_digest", "protocol_digest", "prompt_digest",
        "output_schema_digest", "presentation_digest", "model_runtime_digest",
    ):
        assert forbidden not in parameters

    with pytest.raises(PanelSoftObserverError, match="distinct panel"):
        verify_panel_soft_observer_contract_identity(first, first)

    same_pixels, _, _, _ = _observe(
        panel=_png(72), panel_id="query/same-pixels.png", receipt_index_offset=4
    )
    assert same_pixels.panel_png_digest == first.panel_png_digest
    assert same_pixels.panel_id != first.panel_id
    assert same_pixels.artifact_digest != first.artifact_digest
    assert same_pixels.observation_context_digest != first.observation_context_digest
    assert (
        verify_panel_soft_observer_contract_identity(first, same_pixels)
        == first.contract
    )

    replayed_receipts, _, _, _ = _observe(
        panel=_png(72), panel_id="query/replayed-receipts.png"
    )
    with pytest.raises(PanelSoftObserverError, match="model-call identity is reused"):
        verify_panel_soft_observer_contract_identity(first, replayed_receipts)


def test_duplicate_receipts_cannot_manufacture_repeat_consensus() -> None:
    calls: list[tuple[str, tuple[str, ...], dict[str, object]]] = []
    with pytest.raises(PanelSoftObserverError, match="physical-call identities"):
        _observe(distinct_receipts=False, calls_sink=calls)
    assert len(calls) == 2


def test_safe_aggregation_preserves_order_and_byte_identical_panel_ids() -> None:
    panel = _png(84)
    first, _, vocabulary, _ = _observe(
        panel=panel, panel_id="support/duplicate-a.png"
    )
    second, _, _, _ = _observe(
        panel=panel,
        panel_id="support/duplicate-b.png",
        receipt_index_offset=2,
    )
    commitments = (
        (first.panel_id, first.panel_png_digest),
        (second.panel_id, second.panel_png_digest),
    )
    table = aggregate_panel_soft_observer_artifacts(
        (first, second),
        ordered_panel_commitments=commitments,
        expected_vocabulary=vocabulary,
        expected_contract=first.contract,
    )
    assert table.panel_ids == tuple(item[0] for item in commitments)
    assert table.panel_png_digests == tuple(item[1] for item in commitments)
    assert len(table.cells) == 2 * len(vocabulary.atoms)
    assert panel_soft_duplicate_pixel_digest_counts(table) == {
        first.panel_png_digest: 2
    }
    assert table.contract.to_data()[
        "same_model_repeats_are_independent_evidence"
    ] is False

    with pytest.raises(PanelSoftObserverError, match="item"):
        aggregate_panel_soft_observer_artifacts(
            (first, second),
            ordered_panel_commitments=tuple(reversed(commitments)),
            expected_vocabulary=vocabulary,
            expected_contract=first.contract,
        )
    with pytest.raises(PanelSoftObserverError, match="repeat a panel ID"):
        aggregate_panel_soft_observer_artifacts(
            (first, first),
            ordered_panel_commitments=(commitments[0], commitments[0]),
            expected_vocabulary=vocabulary,
            expected_contract=first.contract,
        )

    replayed, _, _, _ = _observe(
        panel=panel, panel_id="support/replayed-receipts.png"
    )
    with pytest.raises(PanelSoftObserverError, match="model-call identity is reused"):
        aggregate_panel_soft_observer_artifacts(
            (first, replayed),
            ordered_panel_commitments=(
                commitments[0],
                (replayed.panel_id, replayed.panel_png_digest),
            ),
            expected_vocabulary=vocabulary,
            expected_contract=first.contract,
        )


def test_transport_and_parser_failures_still_make_two_calls_and_all_error() -> None:
    vocabulary = _vocabulary()
    valid = _success_payloads(vocabulary)[0]
    malformed = dict(valid)
    malformed[next(iter(malformed))] = "absent"
    artifact, _, _, calls = _observe(
        payloads=(valid, malformed), failures=(True, False)
    )
    assert len(calls) == 2
    assert artifact.physical_call_attempt_count == 2
    assert artifact.receipted_call_count == 1
    assert artifact.status is PanelSoftObserverStatus.MIXED_ERROR
    assert tuple(item.status for item in artifact.repeats) == (
        PanelSoftObserverRepeatStatus.TRANSPORT_ERROR,
        PanelSoftObserverRepeatStatus.PARSER_ERROR,
    )
    assert artifact.repeats[0].receipt is None
    assert artifact.repeats[1].receipt is not None
    assert set(artifact.repeats[0].verdicts_in_view_order) == {"error"}
    assert set(artifact.repeats[1].verdicts_in_view_order) == {"error"}
    for cell in artifact.observation_table.cells:
        assert cell.raw_verdicts == ("error", "error")
        assert cell.disposition is Disposition.ERROR
        assert cell.disposition is not Disposition.CERTIFIED_ABSENT


def test_one_failed_repeat_taints_every_cell_as_error_not_absence() -> None:
    artifact, _, _, calls = _observe(failures=(True, False))
    assert len(calls) == 2
    assert artifact.status is PanelSoftObserverStatus.TRANSPORT_ERROR
    assert all(
        cell.raw_verdicts[0] == "error" for cell in artifact.observation_table.cells
    )
    assert {cell.disposition for cell in artifact.observation_table.cells} == {
        Disposition.ERROR
    }


def test_receipt_blocks_resealed_payload_tamper() -> None:
    artifact, _, _, _ = _observe()
    data = deepcopy(artifact.to_data())
    repeat = data["repeats"][0]
    alias = next(iter(repeat["model_payload"]))
    original = repeat["model_payload"][alias]
    replacement = "mismatch" if original != "mismatch" else "present"
    repeat["model_payload"][alias] = replacement
    view_index = [item["alias"] for item in data["view"]].index(alias)
    repeat["verdicts_in_view_order"][view_index] = replacement
    repeat["repeat_digest"] = canonical_digest(
        {key: value for key, value in repeat.items() if key != "repeat_digest"}
    )
    data["artifact_digest"] = canonical_digest(
        {key: value for key, value in data.items() if key != "artifact_digest"}
    )
    with pytest.raises(PanelSoftObserverError, match="receipt"):
        PanelSoftObserverArtifact.from_data(data)


def test_cold_replay_rejects_different_pixels_and_commitments() -> None:
    artifact, _, vocabulary, _ = _observe()
    with pytest.raises(PanelSoftObserverError, match="cold-replay inputs"):
        verify_panel_soft_observer_artifact(
            artifact,
            _png(99),
            panel_id=PANEL_ID,
            vocabulary=vocabulary,
            expected_artifact_digest=artifact.artifact_digest,
        )
    with pytest.raises(PanelSoftObserverError, match="artifact differs"):
        verify_panel_soft_observer_artifact(
            artifact,
            _png(71),
            panel_id=PANEL_ID,
            vocabulary=vocabulary,
            expected_artifact_digest="0" * 64,
        )


def test_observer_path_has_no_lean_geometry_or_anchor_imports() -> None:
    module = Path(__file__).parents[1] / "panel_soft_observer.py"
    tree = ast.parse(module.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    lowered = tuple(item.lower() for item in imports)
    assert not any("lean" in item for item in lowered)
    assert not any("atlas" in item or "anchor" in item or "salience" in item for item in lowered)
    assert not any("geometry" in item or "hypoth" in item for item in lowered)
    assert "bongard.panel_soft_predicate" in lowered

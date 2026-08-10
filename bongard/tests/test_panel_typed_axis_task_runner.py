"""Synthetic, no-dataset-pixel tests for the typed-axis task runner."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from bongard.object_bongard_release_gate import ObjectBongardReleaseStore
from bongard.panel_typed_axis_headless_proposer import (
    HeadlessTypedAxisAttemptErrorArtifact,
    HeadlessTypedAxisProposerRequest,
    run_headless_typed_axis_proposer,
)
from bongard.panel_typed_axis_slate_v2 import (
    AXES,
    Axis,
    AxisNomination,
    SupportSide,
    TypedAxisCell,
    TypedAxisInventory,
    TypedNominationSlate,
    TypedSupportMatrix,
    TypedSupportRow,
)
from bongard.panel_typed_axis_task_runner import (
    TypedAxisFormulaFreeze,
    TypedAxisQueryEvidence,
    TypedAxisQueryOutcome,
    TypedAxisRankInput,
    TypedAxisTaskGap,
    TypedAxisTaskRunnerError,
    cold_replay_typed_axis_query_decision,
    cold_replay_typed_axis_task_result,
    evaluate_typed_axis_query,
    persist_typed_axis_formula_commit,
    run_typed_axis_formula_task,
    typed_axis_rank_prompt,
)
from bongard.tests.test_panel_positive_formula_ranker import _text_receipt
from bongard.tests.test_panel_typed_axis_headless_proposer import (
    _images,
    _payload,
    _runtime,
    _transport,
)
from bongard.transport import CodexStructuredResult


PROTOCOL = "sha256:" + "7" * 64
PRIMARY_VALUES = {
    Axis.TOPOLOGY: "closed",
    Axis.COMPONENT_COUNT: 1,
    Axis.STRAIGHT_ACTION_COUNT: 4,
    Axis.PRIMITIVE_MIX_OR_ARC_COUNT: "straight_only",
    Axis.CATALOG_CONVEXITY: "catalog_convex",
    Axis.SYMMETRY: "none",
    Axis.ASPECT_ORIENTATION: "compact",
    Axis.TEXTURE: "plain",
}


def _row(index: int, side: SupportSide, values: dict[Axis, int | str]) -> TypedSupportRow:
    return TypedSupportRow(
        f"secret_{side.value}_row_{index:02d}",
        side,
        tuple(TypedAxisCell.python_exact(axis, values[axis], PROTOCOL) for axis in AXES),
    )


def _matrix(kind: str) -> TypedSupportMatrix:
    rows = [_row(index, SupportSide.PRIMARY, dict(PRIMARY_VALUES)) for index in range(6)]
    for index in range(6):
        values = dict(PRIMARY_VALUES)
        if kind == "multi":
            values[Axis.TOPOLOGY] = "open"
        elif kind == "unique":
            if index < 3:
                values[Axis.TOPOLOGY] = "open"
            else:
                values[Axis.STRAIGHT_ACTION_COUNT] = 3
        elif kind != "zero":
            raise AssertionError(kind)
        rows.append(_row(index, SupportSide.CONTRAST, values))
    return TypedSupportMatrix.freeze(rows)


def _attempt(matrix: TypedSupportMatrix, *, error: bool = False):
    primary, contrast = _images()
    runtime = _runtime()
    request = HeadlessTypedAxisProposerRequest.build(
        primary, contrast, matrix=matrix, runtime=runtime
    )
    payload = _payload()
    if error:
        payload["straight_action_count"]["value"] = 4
    calls: list[object] = []
    result = run_headless_typed_axis_proposer(
        primary,
        contrast,
        matrix=matrix,
        request=request,
        runtime=runtime,
        transport=_transport(payload, (*primary, *contrast), calls),
    )
    assert len(calls) == 1
    return result


class _RankTransport:
    def __init__(self, aliases: tuple[str, ...], *, short: bool = False):
        self.aliases = aliases
        self.short = short
        self.calls = 0
        self.prompt = ""

    def __call__(self, prompt, schema, **_kwargs):
        self.calls += 1
        self.prompt = prompt
        aliases = self.aliases[:-1] if self.short else self.aliases
        payload = {"ordered_aliases": list(aliases)}
        return CodexStructuredResult(payload, _text_receipt(prompt, schema, payload))


def test_zero_survivors_is_typed_gap_and_mandatory_attempt_is_bound() -> None:
    matrix = _matrix("zero")
    inventory = TypedAxisInventory.derive(matrix)
    attempt = _attempt(matrix)
    transport = _RankTransport(())
    result = run_typed_axis_formula_task(
        inventory,
        attempt,
        expected_headless_attempt_digest=attempt.attempt_digest,
        rank_runtime=_runtime(),
        rank_transport=transport,
    )
    assert type(result) is TypedAxisTaskGap
    assert transport.calls == 0
    assert result.to_data()["headless_attempt_mandatory"] is True
    assert result.to_data()["headless_attempt_omission_or_reroll_allowed"] is False
    assert result.to_data()["query_release_authorized"] is False
    assert cold_replay_typed_axis_task_result(
        result,
        inventory=inventory,
        headless_attempt=attempt,
        expected_artifact_address=result.record_digest,
    ) == result

    with pytest.raises(TypedAxisTaskRunnerError, match="attempt binding"):
        run_typed_axis_formula_task(
            inventory,
            attempt,
            expected_headless_attempt_digest="0" * 64,
        )
    with pytest.raises(TypeError, match="headless"):
        run_typed_axis_formula_task(  # type: ignore[arg-type]
            inventory, None, expected_headless_attempt_digest=attempt.attempt_digest
        )


def test_unique_survivor_uses_zero_rank_calls_and_ignores_error_attempt_narration() -> None:
    matrix = _matrix("unique")
    all_gap_hints = TypedNominationSlate.freeze(
        matrix,
        tuple(AxisNomination.gap(axis, "ignored_hint") for axis in AXES),
    )
    inventory = TypedAxisInventory.derive(matrix, all_gap_hints)
    assert inventory.admitted_formula_ids
    assert len(inventory.admitted_formula_ids) == 1
    attempt = _attempt(matrix, error=True)
    assert type(attempt) is HeadlessTypedAxisAttemptErrorArtifact
    transport = _RankTransport(("should_not_be_called",))
    freeze = run_typed_axis_formula_task(
        inventory,
        attempt,
        expected_headless_attempt_digest=attempt.attempt_digest,
        rank_runtime=_runtime(),
        rank_transport=transport,
    )
    assert type(freeze) is TypedAxisFormulaFreeze
    assert transport.calls == 0
    assert freeze.selection_mode == "unique_survivor_zero_rank_call"
    assert freeze.selected_formula_wire == {
        "operator": "all_of",
        "atoms": [
            {"axis": "topology", "equals": "closed"},
            {"axis": "straight_action_count", "equals": 4},
        ],
    }
    assert freeze.to_data()["headless_nominations_or_prose_enter_selection"] is False
    assert cold_replay_typed_axis_task_result(
        freeze,
        inventory=inventory,
        headless_attempt=attempt,
        expected_artifact_address=freeze.record_digest,
    ) == freeze


def test_multi_survivor_makes_exactly_one_receipted_wire_only_rank_call() -> None:
    matrix = _matrix("multi")
    inventory = TypedAxisInventory.derive(matrix)
    attempt = _attempt(matrix)
    rank_input = TypedAxisRankInput.from_inventory(inventory)
    transport = _RankTransport(tuple(reversed(rank_input.aliases)))
    freeze = run_typed_axis_formula_task(
        inventory,
        attempt,
        expected_headless_attempt_digest=attempt.attempt_digest,
        rank_runtime=_runtime(),
        rank_transport=transport,
    )
    assert type(freeze) is TypedAxisFormulaFreeze
    assert transport.calls == 1
    assert freeze.rank_artifact is not None
    assert freeze.rank_artifact.to_data()["benchmark_sealable"] is False
    assert freeze.rank_artifact.to_data()["rank_transport_invocations"] == 1
    assert freeze.selected_formula_id == inventory.admitted_formula_ids[-1]
    prompt = transport.prompt
    assert prompt == typed_axis_rank_prompt(rank_input)
    assert attempt.outcome.positive_description not in prompt
    assert all(row.row_key not in prompt for row in matrix.rows)
    for forbidden in ("primary_", "contrast_", "panel", "query", "role", "side", "narration"):
        assert forbidden not in prompt.lower()
    assert set(rank_input.visible_data()[0]) == {"opaque_alias", "formula_wire"}
    assert cold_replay_typed_axis_task_result(
        freeze,
        inventory=inventory,
        headless_attempt=attempt,
        expected_artifact_address=freeze.record_digest,
    ) == freeze
    assert transport.calls == 1

    short = _RankTransport(rank_input.aliases, short=True)
    with pytest.raises(TypedAxisTaskRunnerError, match="exact full permutation"):
        run_typed_axis_formula_task(
            inventory,
            attempt,
            expected_headless_attempt_digest=attempt.attempt_digest,
            rank_runtime=_runtime(),
            rank_transport=short,
        )
    assert short.calls == 1


def _query_cells(kind: str) -> tuple[TypedAxisCell, ...]:
    values = dict(PRIMARY_VALUES)
    if kind == "nonmatch":
        values[Axis.STRAIGHT_ACTION_COUNT] = 3
    cells = [TypedAxisCell.python_exact(axis, values[axis], PROTOCOL) for axis in AXES]
    index = AXES.index(Axis.STRAIGHT_ACTION_COUNT)
    if kind == "indeterminate":
        cells[index] = TypedAxisCell.gap(
            Axis.STRAIGHT_ACTION_COUNT, PROTOCOL, "query_observer_gap"
        )
    elif kind == "error":
        cells[index] = TypedAxisCell.error(
            Axis.STRAIGHT_ACTION_COUNT, PROTOCOL, "query_observer_error"
        )
    elif kind not in {"match", "nonmatch"}:
        raise AssertionError(kind)
    return tuple(cells)


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("match", TypedAxisQueryOutcome.POSITIVE),
        ("nonmatch", TypedAxisQueryOutcome.NEGATIVE),
        ("indeterminate", TypedAxisQueryOutcome.ABSTAIN),
        ("error", TypedAxisQueryOutcome.ERROR),
    ],
)
def test_durable_commit_query_mapping_zero_call_replay_and_tamper(
    tmp_path: Path, kind: str, expected: TypedAxisQueryOutcome
) -> None:
    matrix = _matrix("unique")
    inventory = TypedAxisInventory.derive(matrix)
    attempt = _attempt(matrix)
    freeze = run_typed_axis_formula_task(
        inventory,
        attempt,
        expected_headless_attempt_digest=attempt.attempt_digest,
    )
    assert type(freeze) is TypedAxisFormulaFreeze
    store = ObjectBongardReleaseStore((tmp_path / "store").resolve())
    commit, commit_receipt = persist_typed_axis_formula_commit(store, freeze)
    evidence = TypedAxisQueryEvidence.create(
        query_id=f"synthetic_query_{kind}",
        query_panel_sha256=hashlib.sha256(kind.encode("ascii")).hexdigest(),
        query_release_custody_address="sha256:" + "a" * 64,
        observer_artifact_address="sha256:" + "b" * 64,
        formula_commit=commit,
        cells=_query_cells(kind),
    )
    decision = evaluate_typed_axis_query(
        commit,
        commit_receipt,
        store=store,
        inventory=inventory,
        headless_attempt=attempt,
        query_evidence=evidence,
    )
    assert decision.outcome is expected
    assert decision.to_data()["negative_formula_evaluated"] is False
    assert decision.to_data()["model_calls_during_query_evaluation"] == 0
    assert cold_replay_typed_axis_query_decision(
        decision,
        commit=commit,
        commit_receipt=commit_receipt,
        store=store,
        inventory=inventory,
        headless_attempt=attempt,
        expected_artifact_address=decision.record_digest,
    ) == decision

    tampered = deepcopy(decision.to_data())
    tampered["outcome"] = "negative" if expected is not TypedAxisQueryOutcome.NEGATIVE else "positive"
    with pytest.raises(TypedAxisTaskRunnerError):
        type(decision).from_data(tampered)

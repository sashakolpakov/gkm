"""Focused offline tests for the Codex semantic-cone transport."""
from __future__ import annotations

import copy
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cofibered_proposer as C
import run_semantic_cone as runner


def _hypothesis(index: int) -> dict:
    return {
        "hypothesis_id": f"h{index}",
        "description": "The positive panels have a higher object count.",
        "polarity": "positive_satisfies",
        "diagram": {"edges": [
            {
                "target": "scene",
                "call": {"leg_name": "parse_scene", "args": ["panel"]},
            },
            {
                "target": "score",
                "call": {"leg_name": "object_count", "args": ["scene"]},
            },
        ]},
        "score_node": "score",
        "order": "high_positive",
        "semantic_requirements": ["higher object count"],
        "witness_requirements": [],
        "relations": [],
        "cofibrations": [],
        "preservation_morphisms": [{
            "name": "translate",
            "scope": "panel",
            "expected_effect": "preserve",
        }],
    }


def _payload(offset: int = 0) -> dict:
    return {"hypotheses": [_hypothesis(offset + i) for i in range(3)]}


class _Receipt:
    def __init__(self, value: dict) -> None:
        self.value = value

    def to_dict(self) -> dict:
        return self.value


def test_codex_semantic_proposal_reuses_direct_structured_runner_and_receipt(
        monkeypatch):
    payload = _payload()
    receipt = {
        "schema": "bongard.codex-cli-proposer-receipt/v2",
        "input_digest": "input-bound-to-prompt-and-images",
        "panel_view_digest": "exact-private-image-view",
        "structured_output_digest": "exact-hypothesis-output",
        "event_stream_digest": "exact-jsonl-stream",
        "isolation_policy": (
            "ephemeral-image-only-view-read-only-no-tools-no-config-rules/v1"),
    }
    calls = []

    def fake_run(task, panel_paths, output_schema, **kwargs):
        calls.append((task, panel_paths, output_schema, kwargs))
        return SimpleNamespace(
            payload=copy.deepcopy(payload), receipt=_Receipt(receipt))

    monkeypatch.setattr(C.codex_headless, "run_codex_structured", fake_run)
    panel_paths = [
        f"/private/panels/{side}_{index}.png"
        for side in ("pos", "neg") for index in range(6)
    ]
    proposer = C.CodexCofiberedProposer(
        "gpt-test", minutes=7, reasoning_effort="high",
        verbose=False, executable="codex-test")

    bundle = proposer.propose("problem_00", panel_paths)

    assert len(calls) == 1
    task, observed_paths, schema, kwargs = calls[0]
    assert observed_paths == tuple(panel_paths)
    assert schema is C.HYPOTHESES_SCHEMA
    assert C.CODEX_SUBMISSION_INSTRUCTION in task
    assert "Submit 3 to 8 hypotheses through the tool." not in task
    assert all(path not in task for path in panel_paths)
    assert kwargs == {
        "model": "gpt-test",
        "reasoning_effort": "high",
        "minutes": 7,
        "verbose": False,
        "executable": "codex-test",
    }
    assert [item.hypothesis_id for item in bundle.hypotheses] == [
        "h0", "h1", "h2"]
    assert bundle.proposer_kind == "codex"
    assert bundle.parse_error == ""
    assert json.loads(bundle.raw_text) == payload
    assert bundle.model_receipts == (receipt,)


def test_codex_semantic_refinement_resends_images_and_binds_prior_round(
        monkeypatch):
    responses = [(_payload(), {"thread_id": "turn-0"}),
                 (_payload(10), {"thread_id": "turn-1"})]
    calls = []

    def fake_run(task, panel_paths, output_schema, **kwargs):
        calls.append((task, tuple(panel_paths), output_schema, kwargs))
        payload, receipt = responses[len(calls) - 1]
        return SimpleNamespace(
            payload=copy.deepcopy(payload), receipt=_Receipt(receipt))

    monkeypatch.setattr(C.codex_headless, "run_codex_structured", fake_run)
    panels = [
        f"/private/panels/{side}_{index}.png"
        for side in ("pos", "neg") for index in range(6)
    ]
    proposer = C.CodexCofiberedProposer("gpt-test", verbose=False)
    first = proposer.propose("problem_00", panels)
    feedback = "MISSING_LEG: add an executable ContourWitness path"

    second = proposer.refine("problem_00", feedback)

    assert len(calls) == 2
    assert calls[0][1] == calls[1][1] == tuple(panels)
    assert calls[0][2] is calls[1][2] is C.HYPOTHESES_SCHEMA
    assert feedback in calls[1][0]
    assert first.raw_text in calls[1][0]
    assert "complete replacement hypothesis bundle" in calls[1][0]
    assert [item.hypothesis_id for item in second.hypotheses] == [
        "h10", "h11", "h12"]
    assert second.model_receipts == ({"thread_id": "turn-1"},)


def test_codex_semantic_refine_requires_a_successful_initial_turn():
    proposer = C.CodexCofiberedProposer(verbose=False)
    with pytest.raises(RuntimeError, match=r"refine\(\) before propose"):
        proposer.refine("problem_00", "feedback")


def test_cofibered_factory_and_cli_expose_codex(monkeypatch):
    codex = C.make_cofibered_proposer(
        "codex", model=None, codex_minutes=9,
        codex_reasoning_effort="low", verbose=False,
        codex_executable="codex-test")
    assert isinstance(codex, C.CodexCofiberedProposer)
    assert codex.model == C.CODEX_DEFAULT_MODEL
    assert codex.minutes == 9
    assert codex.reasoning_effort == "low"
    assert codex.executable == "codex-test"

    anthropic = C.make_cofibered_proposer(
        "anthropic", model="sonnet", max_tokens=1234)
    assert isinstance(anthropic, C.AnthropicCofiberedProposer)
    assert anthropic.model == C.MODEL_MAP["sonnet"]
    assert anthropic.max_tokens == 1234

    monkeypatch.setattr(
        sys, "argv", ["run_semantic_cone.py", "--proposer", "codex"])
    args = runner.parse_args()
    assert args.proposer == "codex"
    assert args.model == C.CODEX_DEFAULT_MODEL


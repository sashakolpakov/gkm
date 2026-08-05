"""Focused contract, information-boundary, and replay tests."""
from __future__ import annotations

import copy
import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import contrastive_oracle as C
import grounded_predicate_ir as G


def _png(path: Path, token: int) -> str:
    path.write_bytes(C.PNG_SIGNATURE + bytes([token]) * (9 + token))
    return str(path)


@pytest.fixture
def images(tmp_path):
    positives = [_png(tmp_path / f"prototype-{i}.png", i + 1)
                 for i in range(6)]
    foils = [_png(tmp_path / f"foil-{i}.png", i + 21)
             for i in range(6)]
    target = _png(tmp_path / "secret_neg_problem_identifier.png", 51)
    return positives, foils, target


def _contract(images):
    positives, foils, _target = images
    return C.ContrastiveOracleContract.create(
        "contains a bird-like articulated silhouette", positives, foils,
        model="fixture-model", reasoning_effort="medium")


def _payload(presentation, *, affirmative=5, turn_abstain=False):
    comparisons = []
    for index, pair in enumerate(presentation.pairs):
        choose_affirmative = index < affirmative
        closer = pair.affirmative_side if choose_affirmative else (
            "right" if pair.affirmative_side == "left" else "left")
        comparisons.append({
            "pair_id": pair.pair_id,
            "closer_to": closer,
            "abstain": False,
            "abstention_reason": "",
            "visible_evidence": f"visible shape comparison {index}",
        })
    return {
        "abstain": turn_abstain,
        "abstention_reason": "target is occluded" if turn_abstain else "",
        "comparisons": comparisons,
    }


def _install_transport(monkeypatch, contract, target, payload, captured):
    presentation = C.derive_presentation(
        contract, C.ImageIdentity.from_path(target))

    def fake_run(task, paths, names, schema, **kwargs):
        captured.update(
            task=task, paths=tuple(paths), names=tuple(names),
            schema=schema, kwargs=kwargs)
        binding = C._named_binding(task, schema, presentation)
        receipt = {
            **binding,
            "task_digest": binding["prompt_digest"],
            "structured_output_digest": C._raw_digest(payload),
            "requested_model": contract.model,
            "requested_reasoning_effort": contract.reasoning_effort,
            "input_digest_schema": C.codex_proposer.NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
        }
        return SimpleNamespace(
            payload=payload,
            receipt=SimpleNamespace(to_dict=lambda: dict(receipt)),
        )

    monkeypatch.setattr(
        C.codex_proposer, "run_codex_named_images_structured", fake_run)
    # Transport receipts have a much larger structural schema; binding fields
    # are independently checked by contrastive_oracle in these unit mocks.
    monkeypatch.setattr(C.codex_proposer, "validate_codex_receipt",
                        lambda _receipt: None)
    return presentation


def test_contract_is_content_addressed_and_path_order_independent(images):
    positives, foils, _target = images
    first = _contract(images)
    second = C.ContrastiveOracleContract.create(
        first.affirmative_claim, list(reversed(positives)),
        [foils[i] for i in (2, 5, 1, 4, 0, 3)],
        model="fixture-model", reasoning_effort="medium")
    assert first.digest() == second.digest()
    assert C.ContrastiveOracleContract.from_dict(first.to_dict()) == first
    assert first.output_schema_digest == C._digest(C.contrastive_output_schema())
    assert len(first.positive_prototypes) == len(first.hard_negative_foils) == 6


def test_constructor_rejects_anchor_bytes_that_differ_from_contract(images):
    positives, foils, _target = images
    contract = _contract(images)
    Path(positives[0]).write_bytes(C.PNG_SIGNATURE + b"mutated")
    with pytest.raises(ValueError, match="differ from the frozen contract"):
        C.CodexContrastiveOracle(contract, positives, foils)


def test_one_neutral_fresh_turn_has_six_balanced_content_bound_pairs(
        images, monkeypatch):
    positives, foils, target = images
    contract = _contract(images)
    presentation = C.derive_presentation(
        contract, C.ImageIdentity.from_path(target))
    payload = _payload(presentation)
    captured = {}
    _install_transport(monkeypatch, contract, target, payload, captured)
    result = C.CodexContrastiveOracle(
        contract, positives, foils, executable="fixture-codex").evaluate(target)

    assert isinstance(result.observation, G.Present)
    assert result.observation.value is True
    assert len(captured["paths"]) == 13
    assert captured["paths"].count(target) == 1
    assert captured["names"] == presentation.ordered_names()
    assert all("pos" not in name and "neg" not in name
               for name in captured["names"])
    assert "secret_neg_problem_identifier" not in captured["task"]
    assert sum(pair.affirmative_side == "left"
               for pair in presentation.pairs) == 3
    assert sum(pair.affirmative_side == "right"
               for pair in presentation.pairs) == 3
    assert captured["kwargs"]["model"] == "fixture-model"


@pytest.mark.parametrize(
    ("affirmative_votes", "kind", "value"),
    [(5, G.Present, True), (1, G.Present, False),
     (4, G.Indeterminate, None), (2, G.Indeterminate, None)],
)
def test_harness_owns_fixed_five_of_six_rule(
        images, monkeypatch, affirmative_votes, kind, value):
    positives, foils, target = images
    contract = _contract(images)
    presentation = C.derive_presentation(
        contract, C.ImageIdentity.from_path(target))
    payload = _payload(presentation, affirmative=affirmative_votes)
    _install_transport(monkeypatch, contract, target, payload, {})
    result = C.CodexContrastiveOracle(contract, positives, foils).evaluate(target)
    assert isinstance(result.observation, kind)
    if isinstance(result.observation, G.Present):
        assert result.observation.value is value
        assert result.observation.unit is G.Unit.BOOLEAN
    else:
        assert result.observation.mode == "oracle-no-supermajority"


def test_explicit_turn_abstention_is_indeterminate(images, monkeypatch):
    positives, foils, target = images
    contract = _contract(images)
    presentation = C.derive_presentation(
        contract, C.ImageIdentity.from_path(target))
    payload = _payload(presentation, affirmative=6, turn_abstain=True)
    _install_transport(monkeypatch, contract, target, payload, {})
    result = C.CodexContrastiveOracle(contract, positives, foils).evaluate(target)
    assert isinstance(result.observation, G.Indeterminate)
    assert result.observation.mode == "oracle-abstained"


def test_malformed_pair_vote_fails_closed_as_typed_error(images, monkeypatch):
    positives, foils, target = images
    contract = _contract(images)
    presentation = C.derive_presentation(
        contract, C.ImageIdentity.from_path(target))
    payload = _payload(presentation)
    payload["comparisons"][0]["pair_id"] = "pair_01"
    _install_transport(monkeypatch, contract, target, payload, {})
    result = C.CodexContrastiveOracle(contract, positives, foils).evaluate(target)
    assert isinstance(result.observation, G.Error)
    assert result.evidence is None


def test_evidence_cold_replay_and_receipt_binding_tamper(images, monkeypatch):
    positives, foils, target = images
    contract = _contract(images)
    presentation = C.derive_presentation(
        contract, C.ImageIdentity.from_path(target))
    payload = _payload(presentation)
    _install_transport(monkeypatch, contract, target, payload, {})
    live = C.CodexContrastiveOracle(contract, positives, foils).evaluate(target)
    stored = live.to_dict()
    replayed = C.replay_evaluation(
        contract, stored, target_png_path=target)
    assert replayed.observation.to_dict() == live.observation.to_dict()
    assert replayed.evidence.digest() == live.evidence.digest()

    tampered = copy.deepcopy(stored)
    tampered["evidence"]["receipt"]["panel_view_digest"] = "0" * 64
    unsigned_evidence = {
        key: item for key, item in tampered["evidence"].items()
        if key != "evidence_digest"}
    tampered["evidence"]["evidence_digest"] = C._digest(unsigned_evidence)
    unsigned_evaluation = {
        key: item for key, item in tampered.items()
        if key != "evaluation_digest"}
    tampered["evaluation_digest"] = C._digest(unsigned_evaluation)
    with pytest.raises(ValueError, match="panel_view_digest"):
        C.replay_evaluation(contract, tampered, target_png_path=target)


def test_oracle_observable_is_boolean_and_forces_hybrid_taint(images):
    positives, foils, _target = images
    contract = _contract(images)
    oracle = C.CodexContrastiveOracle(contract, positives, foils)
    observable = oracle.observable_contract()
    assert observable.source is G.ObservableSource.ORACLE
    assert observable.value_type is G.ValueType.BOOLEAN
    registry = G.ObservableRegistry()
    registry.register(observable)
    compiled = G.compile_predicate(
        G.Compare(C.OBSERVABLE_ID, G.ComparisonOperator.EQ,
                  G.Literal(True, G.Unit.BOOLEAN)), registry)
    assert compiled.taint is G.Taint.HYBRID


def test_contract_digest_binds_claim_model_and_reasoning(images):
    contract = _contract(images)
    changed = dict(contract.to_dict())
    changed["affirmative_claim"] = "contains an oblique angular object"
    changed_claim = C.ContrastiveOracleContract.from_dict(changed)
    assert changed_claim.digest() != contract.digest()
    changed = dict(contract.to_dict())
    changed["reasoning_effort"] = "high"
    changed_reasoning = C.ContrastiveOracleContract.from_dict(changed)
    assert changed_reasoning.digest() != contract.digest()


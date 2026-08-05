"""Focused tests for the two-presentation HYBRID contrastive protocol."""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grounded_contrastive_oracle as C  # noqa: E402
import grounded_predicate_ir as G  # noqa: E402


def _png(path: Path, token: int) -> str:
    path.write_bytes(C.PNG_SIGNATURE + bytes([token]) * (11 + token))
    return str(path)


@pytest.fixture
def images(tmp_path: Path):
    exemplars = [_png(tmp_path / f"secret-positive-{i}.png", i + 1)
                 for i in range(6)]
    foils = [_png(tmp_path / f"secret-negative-{i}.png", i + 21)
             for i in range(6)]
    target = _png(tmp_path / "secret-target-label.png", 51)
    return exemplars, foils, target


def _contract(images):
    exemplars, foils, _ = images
    return C.ContrastiveOracleContract.create(
        "contains a bird-like outlined object", exemplars, foils,
        "fixture-model", "medium")


def _install_transport(monkeypatch, contract, target, roles_by_trial,
                       captured, *, malformed_trial=None,
                       tamper_receipt_trial=None, production_threads=None):
    calls = 0

    def fake_run(prompt, paths, names, schema, **kwargs):
        nonlocal calls
        trial = calls
        calls += 1
        presentation = C._presentation(contract, trial)
        comparisons = []
        desired = roles_by_trial[trial]
        for pair in presentation:
            role = desired[pair.pair_key]
            if role == "unknown":
                choice = "tie"
            elif role == pair.left_role:
                choice = "left"
            else:
                choice = "right"
            comparisons.append({
                "pair_id": pair.slot_id,
                "choice": choice,
                "evidence": f"visible comparison for {pair.slot_id}",
            })
        payload = {"comparisons": comparisons}
        if malformed_trial == trial:
            payload["comparisons"][0]["pair_id"] = "comparison_01"
        target_id = C.ImageBinding.from_path(target)
        binding = C._named_binding(prompt, schema, target_id, presentation)
        receipt = {
            "source": (
                "codex-cli" if production_threads is not None
                else "offline-fixture"),
            "requested_model": contract.model,
            "requested_reasoning_effort": contract.reasoning_effort,
            "input_digest_schema": C.codex_proposer.NAMED_IMAGE_INPUT_DIGEST_SCHEMA,
            "task_digest": binding["prompt_digest"],
            "structured_output_digest": C._raw_digest(payload),
            **binding,
        }
        if production_threads is not None:
            receipt["thread_id"] = production_threads[trial]
        if tamper_receipt_trial == trial:
            receipt["panel_view_digest"] = "0" * 64
        captured.append({
            "prompt": prompt, "paths": tuple(paths), "names": tuple(names),
            "schema": schema, "kwargs": kwargs,
        })
        return SimpleNamespace(
            payload=payload,
            receipt=SimpleNamespace(to_dict=lambda: dict(receipt)))

    monkeypatch.setattr(
        C.codex_proposer, "run_codex_named_images_structured", fake_run)
    if production_threads is not None:
        monkeypatch.setattr(
            C.codex_proposer, "validate_codex_receipt", lambda _receipt: None)


def test_create_content_selects_exactly_three_pairs_independent_of_input_order(
        images):
    exemplars, foils, _ = images
    first = _contract(images)
    second = C.ContrastiveOracleContract.create(
        first.claim, list(reversed(exemplars)),
        [foils[i] for i in (2, 5, 1, 4, 0, 3)],
        first.model, first.reasoning_effort)
    assert first.digest() == second.digest()
    assert len(first.pairs) == 3
    assert len({p.exemplar.content_digest for p in first.pairs}) == 3
    assert len({p.foil.content_digest for p in first.pairs}) == 3
    assert first.protocol_status == "HYBRID-EXPLORATORY"
    assert first.calibrator is None
    restored = C.ContrastiveOracleContract.from_dict(first.to_dict())
    assert restored.digest() == first.digest() == first.bundle_digest


def test_two_fresh_neutral_presentations_swap_sides_and_pair_order(
        images, monkeypatch):
    _, _, target = images
    contract = _contract(images)
    roles = tuple({p.pair_key: "anchor" for p in contract.pairs}
                  for _ in range(2))
    captured = []
    _install_transport(monkeypatch, contract, target, roles, captured)
    result = C.CodexContrastiveOracle(contract).evaluate(target)

    assert isinstance(result.observation, G.Present)
    assert result.observation.value is True
    assert len(captured) == 2
    assert all(len(call["paths"]) == 7 for call in captured)
    expected_names = ("target.png",) + tuple(
        name for pair_id in C.PAIR_IDS
        for name in (f"{pair_id}_left.png", f"{pair_id}_right.png"))
    assert all(call["names"] == expected_names for call in captured)
    assert "secret-target-label" not in captured[0]["prompt"]
    assert all(word not in captured[0]["prompt"].lower()
               for word in ("exemplar", "foil", "positive", "negative"))
    first = C._presentation(contract, 0)
    second = C._presentation(contract, 1)
    assert [p.pair_key for p in second] == list(
        reversed([p.pair_key for p in first]))
    first_by_key = {p.pair_key: p for p in first}
    for pair in second:
        old = first_by_key[pair.pair_key]
        assert pair.left == old.right and pair.right == old.left
    assert captured[0]["kwargs"]["model"] == "fixture-model"


@pytest.mark.parametrize(
    ("roles", "expected_type", "expected_value"), [
        (("anchor", "anchor", "unknown"), G.Present, True),
        (("foil", "foil", "unknown"), G.Present, False),
        (("anchor", "foil", "unknown"), G.Indeterminate, None),
        (("anchor", "unknown", "unknown"), G.Indeterminate, None),
    ])
def test_frozen_two_of_three_zero_opposition_decoder(
        images, monkeypatch, roles, expected_type, expected_value):
    _, _, target = images
    contract = _contract(images)
    mapping = {pair.pair_key: role for pair, role in zip(contract.pairs, roles)}
    captured = []
    _install_transport(monkeypatch, contract, target,
                       (mapping, dict(mapping)), captured)
    result = C.CodexContrastiveOracle(contract).evaluate(target)
    assert isinstance(result.observation, expected_type)
    if isinstance(result.observation, G.Present):
        assert result.observation.value is expected_value
        assert result.observation.unit is G.Unit.BOOLEAN
    else:
        assert result.observation.mode == "contrastive-oracle-inconclusive"


def test_pair_disagreement_across_trials_normalizes_to_unknown(
        images, monkeypatch):
    _, _, target = images
    contract = _contract(images)
    first = {pair.pair_key: "anchor" for pair in contract.pairs}
    second = {pair.pair_key: "foil" for pair in contract.pairs}
    captured = []
    _install_transport(monkeypatch, contract, target,
                       (first, second), captured)
    result = C.CodexContrastiveOracle(contract).evaluate(target)
    assert isinstance(result.observation, G.Indeterminate)
    assert result.evidence is not None
    assert result.evidence.normalized_votes == (
        ("pair-0", "unknown"), ("pair-1", "unknown"),
        ("pair-2", "unknown"))


@pytest.mark.parametrize("failure", ["schema", "receipt"])
def test_schema_or_binding_failure_is_typed_error(
        images, monkeypatch, failure):
    _, _, target = images
    contract = _contract(images)
    roles = tuple({p.pair_key: "anchor" for p in contract.pairs}
                  for _ in range(2))
    captured = []
    _install_transport(
        monkeypatch, contract, target, roles, captured,
        malformed_trial=1 if failure == "schema" else None,
        tamper_receipt_trial=1 if failure == "receipt" else None)
    result = C.CodexContrastiveOracle(contract).evaluate(target)
    assert isinstance(result.observation, G.Error)
    assert result.evidence is None


def test_reference_byte_drift_is_typed_binding_error(images, monkeypatch):
    _, _, target = images
    contract = _contract(images)
    selected_path = contract.pairs[0].exemplar.source_path
    assert selected_path is not None
    Path(selected_path).write_bytes(C.PNG_SIGNATURE + b"changed")
    result = C.CodexContrastiveOracle(contract).evaluate(target)
    assert isinstance(result.observation, G.Error)
    assert "frozen binding" in result.observation.detail
    stored = result.to_dict()
    assert stored["target"] == C.ImageBinding.from_path(target).to_dict()
    replayed = C.replay_evaluation(
        contract, stored, target_png_path=target)
    assert replayed.to_dict() == stored


def test_observable_is_boolean_oracle_and_forces_hybrid_taint(images):
    contract = _contract(images)
    observable = C.CodexContrastiveOracle(contract).observable_contract()
    assert observable.source is G.ObservableSource.ORACLE
    assert observable.taint is G.Taint.HYBRID
    registry = G.ObservableRegistry()
    registry.register(observable)
    compiled = G.compile_predicate(
        G.Compare(C.OBSERVABLE_ID, G.ComparisonOperator.EQ,
                  G.Literal(True, G.Unit.BOOLEAN)), registry)
    assert compiled.taint is G.Taint.HYBRID


def test_cold_replay_validates_both_trials_without_model_call(
        images, monkeypatch):
    _, _, target = images
    contract = _contract(images)
    roles = tuple({p.pair_key: "anchor" for p in contract.pairs}
                  for _ in range(2))
    captured = []
    _install_transport(monkeypatch, contract, target, roles, captured)
    live = C.CodexContrastiveOracle(contract).evaluate(target)
    stored = live.to_dict()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("cold replay invoked a model")

    monkeypatch.setattr(
        C.codex_proposer, "run_codex_named_images_structured", forbidden)
    restored = C.ContrastiveOracleContract.from_dict(contract.to_dict())
    replayed = C.replay_evaluation(
        restored, stored, target_png_path=target)
    assert replayed.observation.to_dict() == live.observation.to_dict()
    assert replayed.evidence is not None
    assert replayed.evidence.to_dict() == live.evidence.to_dict()
    assert replayed.to_dict() == stored


def _resign(stored):
    evidence = stored.get("evidence")
    if isinstance(evidence, dict):
        evidence_unsigned = {
            key: value for key, value in evidence.items()
            if key != "evidence_digest"}
        evidence["evidence_digest"] = C._digest(evidence_unsigned)
    evaluation_unsigned = {
        key: value for key, value in stored.items()
        if key != "evaluation_digest"}
    stored["evaluation_digest"] = C._digest(evaluation_unsigned)
    return stored


@pytest.mark.parametrize("tamper", [
    "presentation", "payload", "receipt", "normalized", "observation",
])
def test_cold_replay_rejects_resigned_semantic_tampering(
        images, monkeypatch, tamper):
    _, _, target = images
    contract = _contract(images)
    roles = tuple({p.pair_key: "anchor" for p in contract.pairs}
                  for _ in range(2))
    _install_transport(monkeypatch, contract, target, roles, [])
    stored = C.CodexContrastiveOracle(contract).evaluate(target).to_dict()
    changed = copy.deepcopy(stored)
    if tamper == "presentation":
        changed["evidence"]["trials"][1]["presentation"][0]["pair_key"] = \
            "pair-0"
    elif tamper == "payload":
        changed["evidence"]["trials"][1]["payload"]["comparisons"][0][
            "choice"] = "foil"
    elif tamper == "receipt":
        changed["evidence"]["trials"][1]["receipt"][
            "panel_view_digest"] = "0" * 64
    elif tamper == "normalized":
        changed["evidence"]["normalized_votes"][0][1] = "foil"
    else:
        changed["observation"]["value"] = False
    _resign(changed)
    with pytest.raises(ValueError):
        C.replay_evaluation(contract, changed, target_png_path=target)


def test_cold_replay_rejects_changed_target_bytes(images, monkeypatch):
    _, _, target = images
    contract = _contract(images)
    roles = tuple({p.pair_key: "anchor" for p in contract.pairs}
                  for _ in range(2))
    _install_transport(monkeypatch, contract, target, roles, [])
    stored = C.CodexContrastiveOracle(contract).evaluate(target).to_dict()
    Path(target).write_bytes(C.PNG_SIGNATURE + b"different target")
    with pytest.raises(ValueError, match="target bytes"):
        C.replay_evaluation(contract, stored, target_png_path=target)


def test_rebind_checks_full_frozen_reference_pools(images, tmp_path):
    exemplars, foils, target = images
    contract = _contract(images)
    changed = list(exemplars)
    changed[0] = _png(tmp_path / "replacement.png", 77)
    oracle = C.CodexContrastiveOracle(contract, changed, foils)
    result = oracle.evaluate(target)
    assert isinstance(result.observation, G.Error)
    assert "reference pools differ" in result.observation.detail
    # The target is bound before the pre-transport reference error is exposed,
    # so runner replay classifies the record INVALID instead of aborting.
    replayed = C.replay_evaluation(
        contract, result.to_dict(), target_png_path=target)
    assert isinstance(replayed.observation, G.Error)
    assert replayed.to_dict() == result.to_dict()


def test_live_production_trials_require_distinct_fresh_thread_ids(
        images, monkeypatch):
    _, _, target = images
    contract = _contract(images)
    roles = tuple({p.pair_key: "anchor" for p in contract.pairs}
                  for _ in range(2))
    _install_transport(
        monkeypatch, contract, target, roles, [],
        production_threads=("same-thread", "same-thread"))
    result = C.CodexContrastiveOracle(contract).evaluate(target)
    assert isinstance(result.observation, G.Error)
    assert "not fresh threads" in result.observation.detail


def test_cold_replay_rejects_resigned_duplicate_production_thread_id(
        images, monkeypatch):
    _, _, target = images
    contract = _contract(images)
    roles = tuple({p.pair_key: "anchor" for p in contract.pairs}
                  for _ in range(2))
    _install_transport(
        monkeypatch, contract, target, roles, [],
        production_threads=("fresh-thread-a", "fresh-thread-b"))
    stored = C.CodexContrastiveOracle(contract).evaluate(target).to_dict()
    assert stored["observation"]["status"] == "present"
    changed = copy.deepcopy(stored)
    changed["evidence"]["trials"][1]["receipt"]["thread_id"] = \
        "fresh-thread-a"
    _resign(changed)
    with pytest.raises(ValueError, match="not fresh threads"):
        C.replay_evaluation(contract, changed, target_png_path=target)


def test_observable_referent_is_operational_not_direct_prose_truth(images):
    observable = C.CodexContrastiveOracle(
        _contract(images)).observable_contract()
    assert observable.referent == (
        "panel.operational-resemblance-to-frozen-claim-reference-bundle")

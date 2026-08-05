"""Closed-catalog and information-boundary tests for the Codex proposer."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grounded_observables as O
import grounded_proposer as P


class _Receipt:
    def to_dict(self):
        return {"receipt_digest": "fixture-receipt"}


def test_proposer_emits_only_catalog_intents_and_harness_assigns_ids(
        monkeypatch) -> None:
    _registry, descriptors = O.default_grounded_observables()
    captured = {}
    payload = {
        "analysis": "The positive panels share an asymmetric point contact.",
        "intents": [
            {
                "observable_id": O.SMALL_GAP_ID,
                "shape": "low",
                "rationale": "The narrow exterior opening is small.",
            },
            {
                "observable_id": O.SMALL_GAP_ID,
                "shape": "low",
                "rationale": "A duplicate must not buy another atom.",
            },
            {
                "observable_id": O.GAP_RATIO_ID,
                "shape": "high",
                "rationale": "The two exterior gaps are strongly asymmetric.",
            },
        ],
    }

    def fake_run(task, paths, schema, **kwargs):
        captured.update(task=task, paths=paths, schema=schema, kwargs=kwargs)
        return SimpleNamespace(payload=payload, receipt=_Receipt())

    monkeypatch.setattr(P.codex_proposer, "run_codex_structured", fake_run)
    paths = [f"/opaque/support_{index}.png" for index in range(12)]
    bundle = P.CodexGroundedIntentProposer(
        descriptors, model="fixture-model").propose("opaque-00", paths)

    assert [(item.intent_id, item.observable_id, item.shape)
            for item in bundle.intents] == [
        ("intent-00", O.SMALL_GAP_ID, "low"),
        ("intent-01", O.GAP_RATIO_ID, "high"),
    ]
    assert captured["paths"] == paths
    assert "CLOSED_OBSERVABLE_CATALOG_DIGEST" in captured["task"]
    assert "fit numeric bounds" in captured["task"]
    assert set(captured["schema"]["properties"]["intents"]["items"]
               ["properties"]["observable_id"]["enum"]) == {
        descriptor.contract.observable_id for descriptor in descriptors}
    assert bundle.catalog_digest == P.grounded_catalog_digest(descriptors)
    assert bundle.receipt == {"receipt_digest": "fixture-receipt"}


def test_proposer_rejects_transport_payload_outside_closed_catalog(
        monkeypatch) -> None:
    _registry, descriptors = O.default_grounded_observables()

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(payload={
            "analysis": "Invent an unregistered bird detector.",
            "intents": [{
                "observable_id": "oracle/birdness",
                "shape": "high",
                "rationale": "It looks bird-like.",
            }],
        }, receipt=_Receipt())

    monkeypatch.setattr(P.codex_proposer, "run_codex_structured", fake_run)
    with pytest.raises(ValueError, match="outside the closed catalog"):
        P.CodexGroundedIntentProposer(descriptors).propose(
            "opaque-00", [f"panel-{index}.png" for index in range(12)])


def test_proposer_never_receives_query_panels_or_problem_identity(
        monkeypatch) -> None:
    _registry, descriptors = O.default_grounded_observables()
    captured = {}

    def fake_run(task, paths, _schema, **_kwargs):
        captured.update(task=task, paths=tuple(paths))
        return SimpleNamespace(payload={
            "analysis": "Use the registered gap asymmetry.",
            "intents": [{
                "observable_id": O.GAP_RATIO_ID,
                "shape": "high",
                "rationale": "Positive panels have a much larger ratio.",
            }],
        }, receipt=_Receipt())

    monkeypatch.setattr(P.codex_proposer, "run_codex_structured", fake_run)
    support = [f"/support/posneg-{index}.png" for index in range(12)]
    P.CodexGroundedIntentProposer(descriptors).propose(
        "secret-problem-name", support)
    assert captured["paths"] == tuple(support)
    assert "secret-problem-name" not in captured["task"]
    assert "/query/" not in captured["task"]

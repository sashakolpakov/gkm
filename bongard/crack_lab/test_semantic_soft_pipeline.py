"""Offline tests for prose-grounded blind soft predicates."""
from __future__ import annotations

import os
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import semantic_soft_pipeline as S
from soft_semantics import SoftAbsent, SoftEvidence, content_digest


PRODUCER = content_digest({"fixture": "blind-soft-scorer/v1"})


def bird_spec(*, order: str = "high_positive",
              comparison: str = "absolute") -> S.SoftPredicateSpec:
    return S.SoftPredicateSpec(
        hypothesis_id="bird-silhouette",
        claim="A bird-like articulated silhouette is present.",
        operational_definition=(
            "A central body has two lateral appendages attached on opposed "
            "sides, with an optional small head-like projection."),
        order=order,
        comparison=comparison,
        aggregation="all",
        required_cues=(
            S.SoftCueSpec("central-body", "One visually central body-like mass."),
            S.SoftCueSpec(
                "paired-appendages",
                "Two lateral appendages attach to the central body."),
            S.SoftCueSpec(
                "opposed-placement",
                "The appendages occupy approximately opposed lateral sides."),
        ),
        disqualifiers=(
            S.SoftCueSpec(
                "radial-petal-layout",
                "Several interchangeable petals form a radial arrangement."),
        ),
        preservation_morphisms=(
            "translation", "rotation", "uniform_scale", "stroke_width"),
    )


def evidence_for(spec: S.SoftPredicateSpec, membership: float,
                 *, uncertainty: float = 0.1) -> S.PanelSoftScore:
    disqualifier_ids = {cue.cue_id for cue in spec.disqualifiers}
    scores = tuple(
        (cue_id, 1.0 - membership if cue_id in disqualifier_ids else membership)
        for cue_id in spec.cue_ids)
    composed = S._compose_membership(spec, dict(scores))
    return S.PanelSoftScore(
        spec_digest=spec.digest(),
        result=SoftEvidence(
            spec.hypothesis_id, composed, PRODUCER,
            components=scores, provenance=(spec.digest(),)),
        cue_scores=scores,
        cue_evidence=tuple((cue_id, "fixture evidence")
                           for cue_id in spec.cue_ids),
        uncertainty=uncertainty,
        receipt={"receipt_digest": content_digest({"fixture": composed})},
    )


class MappingScorer:
    def __init__(self, values):
        self.values = values
        self.call_order = []

    def score(self, spec, path):
        self.call_order.append(os.path.basename(path))
        with open(path, "rb") as handle:
            value = self.values[handle.read()]
        if value is None:
            return S.PanelSoftScore(
                spec.digest(),
                SoftAbsent(spec.hypothesis_id, "fixture-abstention"),
                (), (), 1.0, {})
        return evidence_for(spec, value)


@pytest.fixture
def panels(tmp_path):
    paths = []
    for side in ("pos", "neg"):
        for index in range(6):
            path = tmp_path / f"{side}_{index}.png"
            presentation = np.full((16, 16), 255, dtype=np.uint8)
            presentation[index, 0 if side == "pos" else 1] = 0
            Image.fromarray(presentation, mode="L").save(path, format="PNG")
            paths.append(str(path))
    return paths


def test_soft_spec_is_side_free_and_content_addressed():
    spec = bird_spec()
    assert spec.digest().startswith("sha256:")
    assert "order" not in spec.scoring_rubric()
    assert "comparison" not in spec.scoring_rubric()
    assert "hypothesis_id" not in spec.scoring_rubric()
    prompt = S.build_blind_score_prompt(spec)
    assert "high_positive" not in prompt
    assert "pos_0" not in prompt
    proposal_prompt = S.build_soft_proposal_prompt("problem_00")
    assert "nearest-foil" in proposal_prompt
    assert "attachment turns" in proposal_prompt

    with pytest.raises(ValueError, match="side-free"):
        S.SoftPredicateSpec(
            **{**spec.__dict__, "claim": "The positives look bird-like."})
    with pytest.raises(ValueError, match="side"):
        S.SoftCueSpec("positive-cue", "A visible central body exists.")


def test_blind_payload_is_mechanically_composed_and_not_self_scored():
    spec = bird_spec()
    payload = {
        "atomic_scores": [
            {"cue_id": "central-body", "score": .9, "evidence": "body"},
            {"cue_id": "paired-appendages", "score": .8, "evidence": "pair"},
            {"cue_id": "opposed-placement", "score": .7, "evidence": "opposed"},
            {"cue_id": "radial-petal-layout", "score": .2, "evidence": "weak"},
        ],
        "uncertainty": .1,
        "abstain": False,
        "abstention_reason": "not needed",
    }
    panel = S.panel_soft_score_from_payload(
        spec, payload, {"receipt": "fixture"}, producer_digest=PRODUCER)
    assert isinstance(panel.result, SoftEvidence)
    # min(required)=.7 and 1-max(disqualifier)=.8
    assert panel.result.membership == pytest.approx(.7)

    payload["uncertainty"] = .8
    panel = S.panel_soft_score_from_payload(
        spec, payload, {}, producer_digest=PRODUCER)
    assert isinstance(panel.result, SoftAbsent)
    assert panel.result.reason_code == "high-uncertainty"


def test_absolute_soft_rule_solves_and_replays_without_requery(panels):
    spec = bird_spec()
    values = {
        open(path, "rb").read(): (.85 if index < 6 else .15)
        for index, path in enumerate(panels)
    }
    scorer = MappingScorer(values)
    labels = [True] * 6 + [False] * 6
    verified = S.verify_soft_predicate(spec, panels, labels, scorer)
    assert verified.accepted
    assert verified.support_errors == 0
    assert verified.rotated_loo_errors == 0
    assert verified.rotated_loo_checks == 72
    assert verified.threshold == .5
    assert scorer.call_order == ["panel.png"] * 12
    replayed = S.replay_soft_verification(spec, verified.evidence, labels)
    assert replayed.to_dict() == verified.to_dict()


def test_selector_cannot_negate_an_affirmative_soft_predicate():
    with pytest.raises(ValueError, match="high_positive|negating"):
        bird_spec(order="low_positive")


def test_typed_abstention_never_becomes_a_numeric_sentinel(panels):
    spec = bird_spec()
    values = {
        open(path, "rb").read(): (.9 if index < 6 else .1)
        for index, path in enumerate(panels)
    }
    values[open(panels[0], "rb").read()] = None
    verified = S.verify_soft_predicate(
        spec, panels, [True] * 6 + [False] * 6, MappingScorer(values))
    assert verified.scores[0] is None
    assert verified.invalid_measurements == 1
    assert not verified.accepted


def test_exactly_ambiguous_membership_never_satisfies_absolute_rule(panels):
    spec = S.SoftPredicateSpec(
        hypothesis_id="single-cue",
        claim="A single visible cue is present.",
        operational_definition="Score the cue directly.",
        order="high_positive",
        comparison="absolute",
        aggregation="all",
        required_cues=(S.SoftCueSpec("cue", "The cue is visible."),),
        disqualifiers=(),
        preservation_morphisms=("translation",),
    )
    values = {
        open(path, "rb").read(): (0.5 if index < 6 else 0.0)
        for index, path in enumerate(panels)
    }
    verified = S.verify_soft_predicate(
        spec, panels, [True] * 6 + [False] * 6, MappingScorer(values))
    assert verified.support_predictions == (False,) * 12
    assert verified.support_errors == 6
    assert verified.rule.endswith(">0.5")
    assert not verified.accepted


def test_codex_blind_scorer_stages_only_one_neutral_image(
        panels, monkeypatch):
    spec = bird_spec()
    captured = {}
    payload = {
        "atomic_scores": [
            {"cue_id": cue_id, "score": .8, "evidence": "visible cue"}
            for cue_id in spec.cue_ids
        ],
        "uncertainty": .1,
        "abstain": False,
        "abstention_reason": "not needed",
    }

    class Receipt:
        def to_dict(self):
            return {"receipt_digest": "fixture"}

    def fake_run(task, image_paths, image_names, schema, **kwargs):
        captured.update({
            "task": task, "paths": image_paths, "names": image_names,
            "schema": schema, "kwargs": kwargs,
        })
        return SimpleNamespace(payload=payload, receipt=Receipt())

    monkeypatch.setattr(
        S.codex_proposer, "run_codex_named_images_structured", fake_run)
    score = S.CodexBlindSoftScorer(model="fixture-model").score(
        spec, panels[0])
    assert isinstance(score.result, SoftEvidence)
    assert captured["paths"] == [panels[0]]
    assert captured["names"] == ["panel.png"]
    assert "high_positive" not in captured["task"]
    assert "pos_0" not in captured["task"]


def test_codex_proposer_freezes_valid_side_free_rubrics(
        panels, monkeypatch):
    spec = bird_spec()
    raw_hypothesis = {
        key: value for key, value in spec.to_dict().items()
        if key not in {"version", "hypothesis_id"}
    }
    raw_hypothesis["required_cues"] = [
        {"description": cue["description"]}
        for cue in raw_hypothesis["required_cues"]]
    raw_hypothesis["disqualifiers"] = [
        {"description": cue["description"]}
        for cue in raw_hypothesis["disqualifiers"]]
    payload = {
        "analysis": "A bird-like silhouette is the strongest interpretation.",
        "hypotheses": [
            json.loads(json.dumps(raw_hypothesis))
            for _ in range(3)
        ],
    }

    class Receipt:
        def to_dict(self):
            return {"receipt_digest": "fixture"}

    captured = {}

    def fake_run(task, image_paths, schema, **kwargs):
        captured.update({"paths": image_paths, "schema": schema})
        return SimpleNamespace(payload=payload, receipt=Receipt())

    monkeypatch.setattr(S.codex_proposer, "run_codex_structured", fake_run)
    bundle = S.CodexSoftHypothesisProposer(model="fixture-model").propose(
        "problem_00", panels)
    assert len(bundle.hypotheses) == 3
    assert [item.hypothesis_id for item in bundle.hypotheses] == [
        "hypothesis-00", "hypothesis-01", "hypothesis-02"]
    assert bundle.hypotheses[0].cue_ids == (
        "required-00", "required-01", "required-02", "veto-00")
    assert captured["paths"] == panels
    assert captured["schema"] == S.SOFT_HYPOTHESES_OUTPUT_SCHEMA


def test_blind_batch_scores_all_rubrics_in_one_neutral_panel_turn(
        panels, monkeypatch):
    first = bird_spec()
    second = S.SoftPredicateSpec(
        **{**first.__dict__, "hypothesis_id": "winged-silhouette"})
    captured = {}

    def evaluation(alias, spec, score):
        return {
            "rubric_id": alias,
            "atomic_scores": [
                {"cue_id": cue_id, "score": score,
                 "evidence": "visible fixture cue"}
                for cue_id in spec.cue_ids
            ],
            "uncertainty": .1,
            "abstain": False,
            "abstention_reason": "not needed",
        }

    class Receipt:
        def to_dict(self):
            return {"receipt_digest": "fixture"}

    def fake_run(task, paths, names, schema, **kwargs):
        captured.update({"task": task, "paths": paths, "names": names})
        return SimpleNamespace(
            payload={"evaluations": [
                evaluation("rubric_00", first, .8),
                evaluation("rubric_01", second, .7),
            ]},
            receipt=Receipt(),
        )

    monkeypatch.setattr(
        S.codex_proposer, "run_codex_named_images_structured", fake_run)
    scores = S.CodexBlindSoftBatchScorer(model="fixture-model").score_many(
        (first, second), panels[0])
    assert len(scores) == 2
    assert captured["paths"] == [panels[0]]
    assert captured["names"] == ["panel.png"]
    assert "high_positive" not in captured["task"]
    assert "pos_0" not in captured["task"]

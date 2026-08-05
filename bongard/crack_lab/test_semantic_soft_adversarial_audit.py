"""Adversarial regressions for the SEMANTIC-SOFT information boundary."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import semantic_soft_pipeline as soft
from soft_semantics import SoftEvidence, content_digest


PRODUCER_DIGEST = content_digest({"fixture": "soft-adversarial-audit/v1"})
LABELS = (True,) * 6 + (False,) * 6


def _bird_spec(*, order: str = "high_positive") -> soft.SoftPredicateSpec:
    return soft.SoftPredicateSpec(
        hypothesis_id="bird-shape",
        claim="A bird-like silhouette is present.",
        operational_definition=(
            "A central body has two opposed lateral appendages."),
        order=order,
        comparison="absolute",
        aggregation="all",
        required_cues=(
            soft.SoftCueSpec(
                "bird-form", "A central body and two appendages are visible."),
        ),
        disqualifiers=(),
        preservation_morphisms=("translation",),
    )


def _panel_score(
        spec: soft.SoftPredicateSpec, membership: float, *,
        concept_id: str | None = None,
        cue_membership: float | None = None,
        ) -> soft.PanelSoftScore:
    cue_value = membership if cue_membership is None else cue_membership
    cue_scores = tuple((cue_id, cue_value) for cue_id in spec.cue_ids)
    return soft.PanelSoftScore(
        spec_digest=spec.digest(),
        result=SoftEvidence(
            concept_id or spec.hypothesis_id,
            membership,
            PRODUCER_DIGEST,
            components=cue_scores,
            provenance=(spec.digest(),),
        ),
        cue_scores=cue_scores,
        cue_evidence=tuple(
            (cue_id, "adversarial fixture evidence")
            for cue_id in spec.cue_ids),
        uncertainty=0.0,
        receipt={
            "receipt_digest": content_digest(
                {"fixture": "soft-adversarial-audit/v1"}),
        },
    )


def test_affirmative_absolute_membership_cannot_be_declared_low_positive(
        tmp_path):
    del tmp_path
    with pytest.raises(ValueError, match="high_positive|negating"):
        _bird_spec(order="low_positive")


def test_blind_prompt_is_noninterfering_under_selector_only_changes():
    high = _bird_spec()
    low = soft.SoftPredicateSpec(
        **{**high.__dict__, "comparison": "relative"})
    assert high.scoring_rubric() == low.scoring_rubric()
    assert soft.build_blind_score_prompt(high) == \
        soft.build_blind_score_prompt(low)


def test_replay_rejects_membership_unbound_to_frozen_cues_and_concept():
    spec = _bird_spec()
    evidence = tuple(
        _panel_score(
            spec,
            0.9 if index < 6 else 0.1,
            concept_id="unrelated-concept",
            cue_membership=0.1 if index < 6 else 0.9,
        )
        for index in range(12)
    )
    with pytest.raises(ValueError, match="evidence|concept|membership|cue"):
        soft.replay_soft_verification(spec, evidence, LABELS)


def test_verifier_cannot_accept_a_filename_side_channel(tmp_path):
    spec = _bird_spec()
    paths = []
    for side in ("pos", "neg"):
        for index in range(6):
            path = tmp_path / f"{side}_{index}.png"
            presentation = np.full((16, 16), 255, dtype=np.uint8)
            presentation[5, 5] = 0
            Image.fromarray(presentation, mode="L").save(path, format="PNG")
            paths.append(str(path))

    class FilenameScorer:
        def score(self, candidate, path):
            membership = 0.9 if os.path.basename(path).startswith("pos_") \
                else 0.1
            return _panel_score(candidate, membership)

    verification = soft.verify_soft_predicate(
        spec, paths, LABELS, FilenameScorer())
    assert not verification.accepted


def test_rubric_rejects_ordinal_support_group_references():
    with pytest.raises(ValueError, match="side-free|panel identities"):
        soft.SoftPredicateSpec(
            hypothesis_id="ordinal-leak",
            claim="The figure matches one of the first six examples.",
            operational_definition="It resembles a shape presented first.",
            order="high_positive",
            comparison="absolute",
            aggregation="all",
            required_cues=(
                soft.SoftCueSpec(
                    "first-group", "It matches a member of the first group."),
            ),
            disqualifiers=(),
            preservation_morphisms=("translation",),
        )

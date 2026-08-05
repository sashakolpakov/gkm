"""Regressions for runtime witness typing independent of class spelling."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cofibered_proposer import _witness_type_lines
from semantic_compiler import compile_hypothesis
from semantic_ir import (
    DiagramEdge,
    DiagramSpec,
    LegCall,
    MorphSpec,
    SemanticHypothesis,
)
from semantic_legs import (
    default_registry,
    is_pair_witness_codomain,
    is_witness_codomain,
    result_type_for_codomain,
)
from semantic_requirements import (
    score_depends_on_witness,
    witness_type_suggestions,
)
from semantic_selection import complexity_for_cone
from semantic_verifier import _semantic_decision_plan
from visual_witnesses import PairWitness, PointContactSignature, Witness


def _point_contact_hypothesis() -> SemanticHypothesis:
    return SemanticHypothesis(
        version="0.1",
        hypothesis_id="point-contact-presence",
        description="touching loops",
        polarity="positive_satisfies",
        diagram=DiagramSpec(edges=(
            DiagramEdge(
                "signature",
                LegCall("extract_point_contact_signature", ("panel",)),
            ),
            DiagramEdge(
                "score",
                LegCall(
                    "point_contact_small_exterior_gap_degrees",
                    ("signature",),
                ),
            ),
        )),
        score_node="score",
        order="low_positive",
        preservation_morphisms=(MorphSpec("translation", "panel"),),
        semantic_requirements=("touching loops",),
        witness_requirements=("PointContactSignature",),
    )


def test_point_contact_signature_is_a_runtime_typed_witness() -> None:
    runtime_type = result_type_for_codomain("PointContactSignature")

    assert runtime_type is PointContactSignature
    assert issubclass(runtime_type, Witness)
    assert not issubclass(runtime_type, PairWitness)
    assert is_witness_codomain("PointContactSignature")
    assert not is_pair_witness_codomain("PointContactSignature")
    assert is_pair_witness_codomain("CirclePairWitness")
    assert not is_witness_codomain("Measurement")
    assert not is_witness_codomain("MadeUpWitness")


def test_non_suffix_witness_is_visible_to_requirements_and_proposer() -> None:
    registry = default_registry()

    assert score_depends_on_witness(
        {"signature": "PointContactSignature"},
        frozenset({"signature"}),
    )
    assert "PointContactSignature" in witness_type_suggestions(
        "signature", registry)
    assert "PointContactSignature" in _witness_type_lines().split(", ")


def test_compiler_and_verifier_treat_signature_domain_as_witness_presence() -> None:
    registry = default_registry()
    cone = compile_hypothesis(_point_contact_hypothesis(), registry)
    plan = _semantic_decision_plan(cone, registry)

    assert cone.node_types["signature"] == "PointContactSignature"
    assert plan.issue == ""
    assert plan.witness_presence is True


def test_complexity_prices_non_suffix_witness_type() -> None:
    cone = SimpleNamespace(
        hypothesis=SimpleNamespace(
            diagram=SimpleNamespace(edges=()),
            cofibrations=(),
        ),
        node_types={
            "panel": "Panel",
            "signature": "PointContactSignature",
        },
    )

    breakdown = complexity_for_cone(cone)

    assert breakdown.witness_type_cost == 1

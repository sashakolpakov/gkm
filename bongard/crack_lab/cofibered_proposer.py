"""Cofibered semantic-cone proposer interface.

The real experiment uses an LLM proposer that returns typed cone IR only. It
does not return final classifier code. Static proposals are allowed only in
unit tests and are labeled as fixtures.
"""
from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from typing import Protocol

from semantic_ir import SemanticHypothesis, parse_hypotheses_json

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KEY_FILE = os.path.join(REPO_ROOT, "ANTHROPIC_API_KEY.env.local")
MODEL_MAP = {"sonnet": "claude-sonnet-5", "opus": "claude-opus-4-8"}


@dataclass(frozen=True)
class ProposalBundle:
    problem_id: str
    hypotheses: tuple[SemanticHypothesis, ...]
    raw_text: str
    proposer_kind: str


class CofiberedProposer(Protocol):
    def propose(self, problem_id: str, panel_paths: list[str]) -> ProposalBundle:
        ...


def _load_api_key() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    text = open(KEY_FILE, encoding="utf-8").read().strip()
    if text.startswith("ANTHROPIC") and "=" in text:
        return text.split("=", 1)[1].strip()
    return text


def _extract_json(text: str) -> str:
    blocks = re.findall(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return blocks[-1].strip() if blocks else text.strip()


class AnthropicCofiberedProposer:
    def __init__(self, model: str = "sonnet", max_tokens: int = 5000) -> None:
        self.model = MODEL_MAP.get(model, model)
        self.max_tokens = max_tokens

    def propose(self, problem_id: str, panel_paths: list[str]) -> ProposalBundle:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("anthropic package is required for LLM proposals") from exc

        content = [{"type": "text", "text": PROMPT.format(problem_id=problem_id)}]
        for path in panel_paths:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("ascii")
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": data,
                },
            })
        client = anthropic.Anthropic(api_key=_load_api_key())
        msg = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        raw = "\n".join(
            block.text for block in msg.content if getattr(block, "type", "") == "text")
        hypotheses = parse_hypotheses_json(_extract_json(raw))
        return ProposalBundle(problem_id, hypotheses, raw, "anthropic")


class StaticFixtureProposer:
    """Test-only proposer; do not use for reported experiments."""

    def __init__(self, hypotheses: tuple[SemanticHypothesis, ...]) -> None:
        self.hypotheses = hypotheses

    def propose(self, problem_id: str, panel_paths: list[str]) -> ProposalBundle:
        return ProposalBundle(problem_id, self.hypotheses,
                              json.dumps({"hypotheses": [h.to_dict() for h in self.hypotheses]}),
                              "static_fixture")


PROMPT = """\
You are proposing semantic cones for a Bongard problem, not writing a classifier.
The goal is to recover human-like semantic descriptions, not merely reusable
numeric differentiators. A proposal such as "more ink" or "larger bounding box"
is only admissible if it is explicitly an object-level semantic claim such as
"there are two principal objects" or "the principal object is elongated".

Problem id: {problem_id}

You see 12 images: first six positive, next six negative. Return JSON only.
Do not return Python code. Do not mention dataset concepts or filenames.

Return:
{{
  "hypotheses": [
    {{
      "version": "0.1",
      "hypothesis_id": "short_unique_id",
      "description": "human-like semantic invariant/relation, not a raw pixel statistic",
      "polarity": "positive_satisfies",
      "diagram": {{
        "edges": [
          {{"target": "scene", "call": {{"leg_name": "parse_scene", "args": ["panel"]}}}},
          {{"target": "main", "call": {{"leg_name": "select_largest", "args": ["scene"]}}}},
          {{"target": "score", "call": {{"leg_name": "closure_ratio", "args": ["main"]}}}}
        ]
      }},
      "score_node": "score",
      "order": "low_positive",
      "semantic_requirements": ["open curve"],
      "witness_requirements": ["ContourWitness"],
      "relations": [],
      "preservation_morphisms": [
        {{"name": "translate", "scope": "panel", "expected_effect": "preserve"}},
        {{"name": "uniform_scale", "scope": "panel", "expected_effect": "preserve"}}
      ],
      "contrast_interventions": []
    }}
  ]
}}

Available legs:
- parse_scene: Panel -> Scene
- extract_contours: Object -> ContourWitness
- fit_polygon: ContourWitness -> PolygonWitness
- classify_triangle: PolygonWitness -> TriangleWitness
- classify_quadrilateral: PolygonWitness -> QuadrilateralWitness
- fit_circle: ContourWitness -> CircleWitness
- fit_multiple_circles: Scene -> CirclePairWitness
- circle_pair_intersection: CirclePairWitness -> CircleIntersectionWitness
- build_part_graph: Scene -> PartGraphWitness
- detect_attachment: PartGraphWitness -> ContactWitness
- detect_intersection: PartGraphWitness -> IntersectionWitness
- detect_radial_arrangement: PartGraphWitness -> RadialArrangementWitness
- reflection_symmetry: Object -> SymmetryWitness
- rotational_symmetry_order: Object -> SymmetryWitness
- witness_confidence: TriangleWitness -> Measurement
- quadrilateral_confidence: QuadrilateralWitness -> Measurement
- circle_residual: CircleWitness -> Measurement
- circle_intersection_confidence: CircleIntersectionWitness -> Measurement
- radial_part_count: RadialArrangementWitness -> Measurement
- symmetry_order_score: SymmetryWitness -> Measurement
- object_count: Scene -> Measurement
- select_largest: Scene -> Object
- largest_area: Scene -> Measurement
- bbox_aspect: Object -> Measurement
- bbox_fill: Object -> Measurement
- closure_ratio: Object -> Measurement
- symmetry_residual: Object -> Measurement

Use 3 to 8 hypotheses. If a needed leg is missing, still return the closest
typed diagram using available primitive legs and make the missing relation
clear in semantic_requirements/witness_requirements. Do not replace rich
concepts such as triangle, circle, bird-like, pinwheel, attachment, or
intersection by bbox/fill/aspect/closure proxies. The harness will return
MISSING_LEG rather than accepting that weakening.
"""

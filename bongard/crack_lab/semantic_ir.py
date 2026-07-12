"""Typed semantic-cone intermediate representation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from cofibrations import CofibrationSpec


Literal = str | int | float | bool | None


@dataclass(frozen=True)
class LegCall:
    leg_name: str
    args: tuple[str, ...]
    parameters: tuple[tuple[str, Literal], ...] = ()

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "LegCall":
        return LegCall(
            leg_name=str(data["leg_name"]),
            args=tuple(str(x) for x in data.get("args", ())),
            parameters=tuple((str(k), v) for k, v in data.get("parameters", ())),
        )


@dataclass(frozen=True)
class DiagramEdge:
    target: str
    call: LegCall

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "DiagramEdge":
        return DiagramEdge(str(data["target"]), LegCall.from_dict(data["call"]))


@dataclass(frozen=True)
class DiagramSpec:
    edges: tuple[DiagramEdge, ...]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "DiagramSpec":
        return DiagramSpec(tuple(DiagramEdge.from_dict(e) for e in data.get("edges", ())))


@dataclass(frozen=True)
class MorphSpec:
    name: str
    scope: str
    expected_effect: str = "preserve"
    parameters: dict[str, Literal] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "MorphSpec":
        name = data.get("name") or data.get("description") or data.get("semantic_term")
        if not name:
            name = "unnamed_morphism"
        return MorphSpec(
            name=str(name),
            scope=str(data.get("scope", "panel")),
            expected_effect=str(data.get("expected_effect", "preserve")),
            parameters=dict(data.get("parameters", {})),
        )


@dataclass(frozen=True)
class SemanticHypothesis:
    version: str
    hypothesis_id: str
    description: str
    polarity: str
    diagram: DiagramSpec
    score_node: str
    order: str
    preservation_morphisms: tuple[MorphSpec, ...] = ()
    contrast_interventions: tuple[MorphSpec, ...] = ()
    semantic_requirements: tuple[str, ...] = ()
    witness_requirements: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()
    # Gluings are proposer-generated, never library constants: each spec
    # binds diagram nodes and is verified mechanically per panel.
    cofibrations: tuple[CofibrationSpec, ...] = ()
    complexity_hint: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "SemanticHypothesis":
        return SemanticHypothesis(
            version=str(data.get("version", "0.1")),
            hypothesis_id=str(data["hypothesis_id"]),
            description=str(data.get("description", "")),
            polarity=str(data.get("polarity", "positive_satisfies")),
            diagram=DiagramSpec.from_dict(data.get("diagram", {})),
            score_node=str(data["score_node"]),
            order=str(data.get("order", "low_positive")),
            preservation_morphisms=tuple(
                MorphSpec.from_dict(x) for x in data.get("preservation_morphisms", ())),
            contrast_interventions=tuple(
                MorphSpec.from_dict(x) for x in data.get("contrast_interventions", ())),
            semantic_requirements=tuple(str(x) for x in data.get("semantic_requirements", ())),
            witness_requirements=tuple(str(x) for x in data.get("witness_requirements", ())),
            relations=tuple(str(x) for x in data.get("relations", ())),
            cofibrations=tuple(
                CofibrationSpec.from_dict(x) for x in data.get("cofibrations", ())),
            complexity_hint=int(data.get("complexity_hint", 0)),
        )


def parse_hypotheses_json(text: str) -> tuple[SemanticHypothesis, ...]:
    data = json.loads(text)
    if isinstance(data, dict) and "hypotheses" in data:
        data = data["hypotheses"]
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("semantic proposal must be a hypothesis or hypothesis list")
    return tuple(SemanticHypothesis.from_dict(x) for x in data)

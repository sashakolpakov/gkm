"""Risk and conditional complexity records for Bongard semantic candidates."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class Track(str, Enum):
    UNRESTRICTED = "UNRESTRICTED"
    SEMANTIC_PURE = "SEMANTIC-PURE"
    HYBRID = "HYBRID"


@dataclass(frozen=True)
class RiskVector:
    R_support: float = 1.0
    R_rotated_LOO: float = 1.0
    R_naturality: float = 0.0
    R_contrast: float = 0.0
    R_counterfactual: float = 0.0
    R_parser_stability: float = 0.0
    R_archive_regression: float = 0.0

    def scalar(self, weights: dict[str, float] | None = None) -> float:
        w = {
            "R_support": 1.0,
            "R_rotated_LOO": 1.0,
            "R_naturality": 1.0,
            "R_contrast": 1.0,
            "R_counterfactual": 1.0,
            "R_parser_stability": 1.0,
            "R_archive_regression": 1.0,
        }
        if weights:
            w.update(weights)
        return sum(float(getattr(self, k)) * v for k, v in w.items())


@dataclass(frozen=True)
class ComplexityBreakdown:
    new_leg_cost: int = 0
    witness_type_cost: int = 0
    diagram_node_cost: int = 0
    diagram_edge_cost: int = 0
    leg_call_cost: int = 0
    binding_cost: int = 0
    parameter_cost: int = 0
    cofibration_attachment_cost: int = 0
    residual_code_cost: int = 0
    exception_cost: int = 0
    literal_lookup_cost: int = 0

    @property
    def total(self) -> int:
        return sum(int(v) for v in asdict(self).values())


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    track: Track
    semantic_admissible: bool
    risk: RiskVector
    complexity: ComplexityBreakdown
    lambda_value: float = 0.02
    diagnostics: tuple[str, ...] = ()
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    @property
    def free_energy(self) -> float:
        return self.risk.scalar() + self.lambda_value * self.complexity.total

    def to_dict(self) -> dict:
        data = asdict(self)
        data["track"] = self.track.value
        data["free_energy"] = self.free_energy
        return data


def complexity_for_cone(cone, promoted_legs: set[str] | None = None,
                        residual_code_cost: int = 0,
                        exception_cost: int = 0) -> ComplexityBreakdown:
    promoted = promoted_legs or set()
    new_legs = [leg for leg in cone.used_legs if leg not in promoted]
    node_count = max(0, len(cone.node_types) - 1)
    edge_count = len(cone.used_legs)
    witness_types = {
        typ for typ in cone.node_types.values()
        if typ.endswith("Witness") and typ not in promoted
    }
    return ComplexityBreakdown(
        new_leg_cost=sum(1 for _ in new_legs),
        witness_type_cost=len(witness_types),
        diagram_node_cost=node_count,
        diagram_edge_cost=edge_count,
        leg_call_cost=edge_count,
        binding_cost=edge_count,
        parameter_cost=sum(len(edge.call.parameters) for edge in cone.hypothesis.diagram.edges),
        residual_code_cost=residual_code_cost,
        exception_cost=exception_cost,
    )


def pareto_frontier(candidates: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
    out: list[CandidateEvaluation] = []
    for cand in candidates:
        dominated = False
        for other in candidates:
            if other is cand:
                continue
            no_worse = (
                other.risk.R_support <= cand.risk.R_support
                and other.risk.R_rotated_LOO <= cand.risk.R_rotated_LOO
                and other.risk.R_naturality <= cand.risk.R_naturality
                and other.risk.R_counterfactual <= cand.risk.R_counterfactual
                and other.complexity.total <= cand.complexity.total
            )
            strictly_better = (
                other.risk.R_support < cand.risk.R_support
                or other.risk.R_rotated_LOO < cand.risk.R_rotated_LOO
                or other.risk.R_naturality < cand.risk.R_naturality
                or other.risk.R_counterfactual < cand.risk.R_counterfactual
                or other.complexity.total < cand.complexity.total
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            out.append(cand)
    return out

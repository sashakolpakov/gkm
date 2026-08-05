"""Risk, admissibility, and conditional-complexity selection primitives.

The semantic runner has several independent empirical risks.  A risk that was
not measured is represented by ``None``; it is never silently treated as zero.
Callers may explicitly exclude or penalize an unmeasured dimension, but the
default free-energy path is strict.

This module deliberately contains no verifier policy.  A runner chooses the
risk dimensions required by its protocol and passes them to the helpers below.
That makes partial experiments possible without letting an omitted check look
like a successful check.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Literal

from semantic_legs import is_witness_codomain


RISK_FIELDS = (
    "R_support",
    "R_rotated_LOO",
    "R_naturality",
    "R_contrast",
    "R_counterfactual",
    "R_parser_stability",
    "R_archive_regression",
)

UnmeasuredPolicy = Literal["error", "exclude", "penalize"]


class UnmeasuredRiskError(ValueError):
    """A requested risk scalar contains dimensions that were not measured."""

    def __init__(self, risk_fields: Iterable[str]) -> None:
        self.risk_fields = tuple(risk_fields)
        super().__init__(
            "unmeasured risk dimensions: " + ", ".join(self.risk_fields))


class Track(str, Enum):
    UNRESTRICTED = "UNRESTRICTED"
    SEMANTIC_PURE = "SEMANTIC-PURE"
    HYBRID = "HYBRID"


def _finite_nonnegative(value: float | int, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return number


def _risk_field_tuple(risk_fields: Iterable[str] | None) -> tuple[str, ...]:
    requested = RISK_FIELDS if risk_fields is None else tuple(dict.fromkeys(risk_fields))
    unknown = tuple(name for name in requested if name not in RISK_FIELDS)
    if unknown:
        raise KeyError("unknown risk dimensions: " + ", ".join(unknown))
    return requested


def _risk_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    out = {name: 1.0 for name in RISK_FIELDS}
    if weights:
        unknown = tuple(name for name in weights if name not in RISK_FIELDS)
        if unknown:
            raise KeyError("unknown risk weights: " + ", ".join(unknown))
        for name, value in weights.items():
            out[name] = _finite_nonnegative(value, f"weight {name}")
    return out


@dataclass(frozen=True)
class RiskVector:
    R_support: float | None = None
    R_rotated_LOO: float | None = None
    R_naturality: float | None = None
    R_contrast: float | None = None
    R_counterfactual: float | None = None
    R_parser_stability: float | None = None
    R_archive_regression: float | None = None

    def __post_init__(self) -> None:
        for name in RISK_FIELDS:
            value = getattr(self, name)
            if value is not None:
                _finite_nonnegative(value, name)

    @property
    def measured_fields(self) -> tuple[str, ...]:
        return tuple(name for name in RISK_FIELDS if getattr(self, name) is not None)

    @property
    def unmeasured_fields(self) -> tuple[str, ...]:
        return tuple(name for name in RISK_FIELDS if getattr(self, name) is None)

    @property
    def fully_measured(self) -> bool:
        return not self.unmeasured_fields

    def scalar(self, weights: Mapping[str, float] | None = None, *,
               risk_fields: Iterable[str] | None = None,
               unmeasured: UnmeasuredPolicy = "error",
               unmeasured_penalty: float | None = None) -> float:
        """Return a weighted scalar without conflating unknown risk with zero.

        ``risk_fields`` selects the dimensions in the protocol.  By default all
        dimensions are selected and every nonzero-weight dimension must be
        measured.  ``exclude`` and ``penalize`` are available only as explicit
        caller choices.
        """
        selected = _risk_field_tuple(risk_fields)
        w = _risk_weights(weights)
        if unmeasured not in {"error", "exclude", "penalize"}:
            raise ValueError(f"unknown unmeasured-risk policy {unmeasured!r}")
        if unmeasured == "penalize":
            if unmeasured_penalty is None:
                raise ValueError("unmeasured_penalty is required for penalize policy")
            penalty = _finite_nonnegative(unmeasured_penalty, "unmeasured_penalty")
        else:
            penalty = 0.0

        missing = tuple(
            name for name in selected
            if w[name] != 0.0 and getattr(self, name) is None)
        if missing and unmeasured == "error":
            raise UnmeasuredRiskError(missing)

        total = 0.0
        for name in selected:
            weight = w[name]
            if weight == 0.0:
                continue
            value = getattr(self, name)
            if value is None:
                if unmeasured == "penalize":
                    total += weight * penalty
                continue
            total += weight * float(value)
        return total


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

    def __post_init__(self) -> None:
        for component in fields(self):
            value = getattr(self, component.name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{component.name} must be an integer")
            if value < 0:
                raise ValueError(f"{component.name} must be nonnegative")

    @property
    def total(self) -> int:
        return sum(getattr(self, component.name) for component in fields(self))

    def to_dict(self) -> dict[str, int]:
        data = asdict(self)
        data["total"] = self.total
        return data


def conditional_free_energy(
        risk: RiskVector,
        complexity: ComplexityBreakdown,
        lambda_value: float = 0.02,
        *,
        risk_weights: Mapping[str, float] | None = None,
        risk_fields: Iterable[str] | None = None,
        unmeasured: UnmeasuredPolicy = "error",
        unmeasured_penalty: float | None = None) -> float:
    """Compute ``R + lambda * C(M | L)`` under an explicit risk protocol."""
    lam = _finite_nonnegative(lambda_value, "lambda_value")
    return risk.scalar(
        risk_weights,
        risk_fields=risk_fields,
        unmeasured=unmeasured,
        unmeasured_penalty=unmeasured_penalty,
    ) + lam * complexity.total


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

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be nonempty")
        if not isinstance(self.track, Track):
            raise TypeError("track must be a Track value")
        _finite_nonnegative(self.lambda_value, "lambda_value")

    @property
    def free_energy(self) -> float:
        """Strict full-vector free energy, retained as the compatibility API."""
        return conditional_free_energy(
            self.risk, self.complexity, self.lambda_value)

    def score(self, *, risk_weights: Mapping[str, float] | None = None,
              risk_fields: Iterable[str] | None = None,
              unmeasured: UnmeasuredPolicy = "error",
              unmeasured_penalty: float | None = None) -> float:
        return conditional_free_energy(
            self.risk,
            self.complexity,
            self.lambda_value,
            risk_weights=risk_weights,
            risk_fields=risk_fields,
            unmeasured=unmeasured,
            unmeasured_penalty=unmeasured_penalty,
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["track"] = self.track.value
        data["unmeasured_risks"] = list(self.risk.unmeasured_fields)
        # Serialization must remain diagnostic-safe for partial experiments.
        # ``None`` is explicit and cannot be mistaken for a successful zero.
        data["free_energy"] = (
            self.free_energy if self.risk.fully_measured else None)
        return data


def _validated_cost_map(costs: Mapping[str, int] | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, value in (costs or {}).items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"leg definition cost for {name} must be an integer")
        if value < 0:
            raise ValueError(f"leg definition cost for {name} must be nonnegative")
        out[str(name)] = value
    return out


def complexity_for_cone(cone, promoted_legs: set[str] | None = None,
                        residual_code_cost: int = 0,
                        exception_cost: int = 0,
                        *,
                        promoted_witness_types: set[str] | None = None,
                        leg_definition_costs: Mapping[str, int] | None = None,
                        literal_lookup_cost: int = 0) -> ComplexityBreakdown:
    """Price a cone conditionally on a promoted semantic library.

    A leg definition is charged once even when the cone calls it repeatedly;
    calls and bindings remain per edge.  Promoted leg names and promoted
    witness-type names are separate namespaces, fixing the earlier accidental
    comparison of witness types against ``promoted_legs``.
    """
    promoted = set(promoted_legs or ())
    promoted_types = set(promoted_witness_types or ())
    definition_costs = _validated_cost_map(leg_definition_costs)
    edges = tuple(cone.hypothesis.diagram.edges)
    used_leg_names = tuple(edge.call.leg_name for edge in edges)
    new_leg_names = sorted(set(used_leg_names) - promoted)
    node_count = max(0, len(cone.node_types) - int("panel" in cone.node_types))
    edge_count = len(edges)
    witness_types = {
        typ for typ in cone.node_types.values()
        if is_witness_codomain(typ) and typ not in promoted_types
    }
    cofibrations = tuple(getattr(cone.hypothesis, "cofibrations", ()))
    return ComplexityBreakdown(
        new_leg_cost=sum(definition_costs.get(name, 1) for name in new_leg_names),
        witness_type_cost=len(witness_types),
        diagram_node_cost=node_count,
        diagram_edge_cost=edge_count,
        leg_call_cost=edge_count,
        binding_cost=edge_count,
        parameter_cost=sum(len(edge.call.parameters) for edge in edges),
        cofibration_attachment_cost=sum(
            1 for spec in cofibrations if getattr(spec, "attachment_leg", "")),
        residual_code_cost=residual_code_cost,
        exception_cost=exception_cost,
        literal_lookup_cost=literal_lookup_cost,
    )


def admissibility_issues(
        candidate: CandidateEvaluation,
        *,
        required_risks: Iterable[str] = (),
        risk_limits: Mapping[str, float] | None = None,
        require_fully_measured: bool = False) -> tuple[str, ...]:
    """Return machine-readable reasons a candidate is not selectable."""
    required = set(_risk_field_tuple(required_risks))
    if require_fully_measured:
        required.update(RISK_FIELDS)
    limits = risk_limits or {}
    unknown_limits = tuple(name for name in limits if name not in RISK_FIELDS)
    if unknown_limits:
        raise KeyError("unknown risk limits: " + ", ".join(unknown_limits))

    issues: list[str] = []
    if not candidate.semantic_admissible:
        issues.append("semantic_inadmissible")
    for name in RISK_FIELDS:
        value = getattr(candidate.risk, name)
        if name in required and value is None:
            issues.append(f"unmeasured:{name}")
        if name in limits:
            limit = _finite_nonnegative(limits[name], f"limit {name}")
            if value is None:
                marker = f"unmeasured:{name}"
                if marker not in issues:
                    issues.append(marker)
            elif float(value) > limit:
                issues.append(f"risk_limit:{name}")
    return tuple(issues)


def is_admissible(candidate: CandidateEvaluation, *,
                  required_risks: Iterable[str] = (),
                  risk_limits: Mapping[str, float] | None = None,
                  require_fully_measured: bool = False) -> bool:
    return not admissibility_issues(
        candidate,
        required_risks=required_risks,
        risk_limits=risk_limits,
        require_fully_measured=require_fully_measured,
    )


def _dominates(other: CandidateEvaluation, candidate: CandidateEvaluation,
               comparison_fields: tuple[str, ...] | None) -> bool:
    if comparison_fields is None:
        # Partial vectors with different measurement signatures are
        # incomparable.  This keeps "not measured" distinct from zero.
        if other.risk.measured_fields != candidate.risk.measured_fields:
            return False
        names = other.risk.measured_fields
    else:
        names = comparison_fields
    if not names:
        return False
    other_values = tuple(float(getattr(other.risk, name)) for name in names)
    candidate_values = tuple(float(getattr(candidate.risk, name)) for name in names)
    no_worse = all(a <= b for a, b in zip(other_values, candidate_values)) \
        and other.complexity.total <= candidate.complexity.total
    strictly_better = any(a < b for a, b in zip(other_values, candidate_values)) \
        or other.complexity.total < candidate.complexity.total
    return no_worse and strictly_better


def _require_single_track(candidates: Sequence[CandidateEvaluation]) -> None:
    tracks = {candidate.track for candidate in candidates}
    if len(tracks) > 1:
        labels = ", ".join(sorted(track.value for track in tracks))
        raise ValueError(
            f"candidate selection cannot mix experiment tracks: {labels}")


def pareto_frontier(
        candidates: Sequence[CandidateEvaluation],
        *,
        risk_fields: Iterable[str] | None = None,
        required_risks: Iterable[str] = (),
        risk_limits: Mapping[str, float] | None = None,
        require_fully_measured: bool = False) -> list[CandidateEvaluation]:
    """Return the admissible risk/complexity frontier in input order.

    When ``risk_fields`` is supplied, those dimensions are both required and
    compared.  Without it, candidates are compared only to candidates with the
    same explicit measurement signature; missing dimensions never become zero.
    """
    _require_single_track(candidates)
    comparison_fields = (
        _risk_field_tuple(risk_fields) if risk_fields is not None else None)
    required = set(_risk_field_tuple(required_risks))
    if comparison_fields is not None:
        required.update(comparison_fields)
    selectable = [
        candidate for candidate in candidates
        if is_admissible(
            candidate,
            required_risks=required,
            risk_limits=risk_limits,
            require_fully_measured=require_fully_measured,
        )
    ]
    return [
        candidate for candidate in selectable
        if not any(
            other is not candidate and _dominates(other, candidate, comparison_fields)
            for other in selectable)
    ]


def rank_candidates(
        candidates: Sequence[CandidateEvaluation],
        *,
        risk_fields: Iterable[str] = RISK_FIELDS,
        risk_weights: Mapping[str, float] | None = None,
        risk_limits: Mapping[str, float] | None = None,
        pareto_only: bool = True) -> list[CandidateEvaluation]:
    """Rank fully specified admissible candidates for runner-side selection."""
    _require_single_track(candidates)
    selected_fields = _risk_field_tuple(risk_fields)
    pool = [
        candidate for candidate in candidates
        if is_admissible(
            candidate,
            required_risks=selected_fields,
            risk_limits=risk_limits,
        )
    ]
    if pareto_only:
        pool = pareto_frontier(
            pool,
            risk_fields=selected_fields,
            risk_limits=risk_limits,
        )
    return sorted(
        pool,
        key=lambda candidate: (
            candidate.score(
                risk_fields=selected_fields,
                risk_weights=risk_weights,
            ),
            candidate.complexity.total,
            candidate.candidate_id,
        ),
    )


def select_candidate(
        candidates: Sequence[CandidateEvaluation],
        *,
        risk_fields: Iterable[str] = RISK_FIELDS,
        risk_weights: Mapping[str, float] | None = None,
        risk_limits: Mapping[str, float] | None = None,
        pareto_only: bool = True) -> CandidateEvaluation | None:
    """Return the best selectable candidate, or ``None`` if none qualify."""
    ranked = rank_candidates(
        candidates,
        risk_fields=risk_fields,
        risk_weights=risk_weights,
        risk_limits=risk_limits,
        pareto_only=pareto_only,
    )
    return ranked[0] if ranked else None

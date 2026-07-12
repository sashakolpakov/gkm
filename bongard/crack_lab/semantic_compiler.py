"""Mechanical compiler from semantic IR to executable cone measurements."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from semantic_ir import SemanticHypothesis
from semantic_legs import LegRegistry
from semantic_requirements import MissingLeg, requirements_for_hypothesis


class CompileError(ValueError):
    pass


class MissingLegError(CompileError):
    def __init__(self, missing: MissingLeg) -> None:
        self.missing = missing
        super().__init__(missing.describe())


@dataclass
class ExecutionTrace:
    node_values: dict[str, Any]
    node_types: dict[str, str]
    leg_status: dict[str, str]
    errors: tuple[str, ...] = ()


@dataclass
class CompiledCone:
    hypothesis: SemanticHypothesis
    used_legs: tuple[str, ...]
    node_types: dict[str, str]
    node_dependencies: dict[str, frozenset[str]]
    complexity: int

    def trace(self, panel: np.ndarray, registry: LegRegistry) -> ExecutionTrace:
        values: dict[str, Any] = {"panel": panel}
        types: dict[str, str] = {"panel": "Panel"}
        statuses: dict[str, str] = {}
        errors: list[str] = []
        for edge in self.hypothesis.diagram.edges:
            call = edge.call
            try:
                leg = registry.get(call.leg_name)
                args = [values[name] for name in call.args]
                values[edge.target] = leg.implementation(*args)
                types[edge.target] = leg.codomain
                statuses[edge.target] = "ok"
            except Exception as exc:
                values[edge.target] = None
                statuses[edge.target] = f"error:{type(exc).__name__}"
                errors.append(f"{edge.target}:{exc}")
        return ExecutionTrace(values, types, statuses, tuple(errors))

    def score(self, panel: np.ndarray, registry: LegRegistry) -> tuple[float, ExecutionTrace]:
        tr = self.trace(panel, registry)
        if tr.errors:
            return 0.0, tr
        if self.hypothesis.score_node not in tr.node_values:
            return 0.0, ExecutionTrace(
                tr.node_values, tr.node_types, tr.leg_status,
                (f"missing score node {self.hypothesis.score_node}",))
        value = tr.node_values[self.hypothesis.score_node]
        try:
            return float(value), tr
        except Exception:
            return 0.0, ExecutionTrace(
                tr.node_values, tr.node_types, tr.leg_status,
                (f"non-numeric score node {self.hypothesis.score_node}",))


def compile_hypothesis(hypothesis: SemanticHypothesis,
                       registry: LegRegistry) -> CompiledCone:
    env_types: dict[str, str] = {"panel": "Panel"}
    dependencies: dict[str, frozenset[str]] = {"panel": frozenset()}
    used: list[str] = []
    complexity = hypothesis.complexity_hint
    for edge in hypothesis.diagram.edges:
        call = edge.call
        try:
            leg = registry.get(call.leg_name)
        except KeyError as exc:
            raise CompileError(str(exc)) from exc
        if len(call.args) != len(leg.domain):
            raise CompileError(f"{call.leg_name}: arity mismatch")
        for arg, expected in zip(call.args, leg.domain):
            actual = env_types.get(arg)
            if actual is None:
                raise CompileError(f"{call.leg_name}: unresolved argument {arg}")
            if actual != expected:
                raise CompileError(
                    f"{call.leg_name}: {arg} has type {actual}, expected {expected}")
        if edge.target in env_types:
            raise CompileError(f"node {edge.target} is already bound")
        env_types[edge.target] = leg.codomain
        deps = {edge.target, call.leg_name}
        for arg in call.args:
            deps.update(dependencies.get(arg, frozenset()))
            deps.add(arg)
        dependencies[edge.target] = frozenset(deps)
        used.append(call.leg_name)
        complexity += leg.complexity + 1 + len(call.parameters)
    if hypothesis.score_node not in env_types:
        raise CompileError(f"score node {hypothesis.score_node} is not produced")
    if env_types[hypothesis.score_node] != "Measurement":
        raise CompileError("score node must have type Measurement")
    if hypothesis.order not in {"low_positive", "high_positive"}:
        raise CompileError("order must be low_positive or high_positive")
    cone = CompiledCone(hypothesis, tuple(used), env_types, dependencies, complexity)
    _validate_semantic_requirements(cone, registry)
    return cone


def _validate_semantic_requirements(cone: CompiledCone, registry: LegRegistry) -> None:
    """Reject semantic weakening before verifier/MDL selection.

    A description that names a rich concept must be witnessed by the compiled
    graph.  Bbox/fill/aspect/closure scalars remain legal when explicitly
    named as the concept, but they cannot stand in for triangle/circle/bird,
    attachment, intersection, pinwheel, etc.
    """
    node_types = set(cone.node_types.values())
    score_deps = cone.node_dependencies.get(cone.hypothesis.score_node, frozenset())
    used = set(cone.used_legs)
    available = tuple(sorted(node_types))

    explicit_witnesses = tuple(getattr(cone.hypothesis, "witness_requirements", ()))
    for witness_type in explicit_witnesses:
        if witness_type and witness_type not in node_types:
            raise MissingLegError(MissingLeg(
                semantic_term=witness_type,
                required_witness_types=(witness_type,),
                available_terminal_types=available,
                attempted_paths=cone.used_legs,
            ))

    for req in requirements_for_hypothesis(cone.hypothesis):
        missing_types = tuple(t for t in req.primitive_required_types if t not in node_types)
        accepted = any(t in node_types for t in req.accepted_types)
        if missing_types or not accepted:
            required = tuple(dict.fromkeys(req.primitive_required_types + req.accepted_types))
            raise MissingLegError(MissingLeg(
                semantic_term=req.term,
                required_witness_types=required,
                available_terminal_types=available,
                unresolved_relation=req.unresolved_relation,
                attempted_paths=cone.used_legs,
                missing_legs=req.missing_legs,
            ))

        required_nodes = tuple(
            name for name, typ in cone.node_types.items()
            if typ in req.primitive_required_types or typ in req.accepted_types
        )
        if required_nodes and not any(node in score_deps for node in required_nodes):
            raise MissingLegError(MissingLeg(
                semantic_term=req.term,
                required_witness_types=tuple(dict.fromkeys(req.primitive_required_types + req.accepted_types)),
                available_terminal_types=available,
                unresolved_relation="required witness is decorative; final score does not depend on it",
                attempted_paths=cone.used_legs,
            ))

        proxy_only = used <= {
            "parse_scene", "select_largest", "select_largest_object",
            "select_principal_objects", "select_all_objects",
            "bbox_fill", "bbox_aspect", "total_ink", "largest_area",
            "closure_ratio", "symmetry_residual", "object_count",
        }
        if proxy_only:
            raise MissingLegError(MissingLeg(
                semantic_term=req.term,
                required_witness_types=tuple(dict.fromkeys(req.primitive_required_types + req.accepted_types)),
                available_terminal_types=available,
                unresolved_relation="rich concept discharged only by scalar proxy legs",
                attempted_paths=cone.used_legs,
                missing_legs=req.missing_legs,
            ))

"""Mechanical compiler from semantic IR to executable cone measurements."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from semantic_ir import SemanticHypothesis
from semantic_legs import LegRegistry
from semantic_requirements import (
    MissingLeg,
    audit_term_coverage,
    leg_suggestions,
    term_tokens,
)


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
    complexity += sum(spec.complexity_cost for spec in hypothesis.cofibrations)
    cone = CompiledCone(hypothesis, tuple(used), env_types, dependencies, complexity)
    _validate_gluings(cone, registry)
    _validate_semantic_requirements(cone, registry)
    return cone


def _validate_gluings(cone: CompiledCone, registry: LegRegistry) -> None:
    """Statically check proposer-generated gluing requests.

    Nothing here is concept-specific: a gluing is admitted or rejected purely
    on whether its declared nodes exist with the declared types and whether
    its attachment leg is implemented.  A missing attachment leg is reported
    as MISSING_LEG so representation poverty stays a visible outcome.
    """
    node_types = cone.node_types
    score_deps = cone.node_dependencies.get(cone.hypothesis.score_node, frozenset())
    available = tuple(sorted(set(node_types.values())))
    for spec in cone.hypothesis.cofibrations:
        if spec.attachment_leg:
            try:
                registry.get(spec.attachment_leg)
            except KeyError:
                raise MissingLegError(MissingLeg(
                    semantic_term=spec.name,
                    required_witness_types=tuple(
                        t for t in (spec.source_type, spec.target_type) if t),
                    available_terminal_types=available,
                    unresolved_relation="gluing attachment leg is not implemented",
                    attempted_paths=cone.used_legs,
                    missing_legs=(spec.attachment_leg,),
                ))
        for node_attr, type_attr in (("source_node", "source_type"),
                                     ("target_node", "target_type")):
            node = getattr(spec, node_attr)
            declared = getattr(spec, type_attr)
            if not node:
                continue
            if node not in node_types:
                raise CompileError(
                    f"gluing {spec.name}: {node_attr} {node} is not a diagram node")
            if declared and node_types[node] != declared:
                raise CompileError(
                    f"gluing {spec.name}: {node} has type {node_types[node]}, "
                    f"declared {declared}")
        if spec.target_node and spec.target_node not in score_deps:
            raise MissingLegError(MissingLeg(
                semantic_term=spec.name,
                required_witness_types=tuple(
                    t for t in (spec.source_type, spec.target_type) if t),
                available_terminal_types=available,
                unresolved_relation="gluing is decorative; final score does not depend on it",
                attempted_paths=cone.used_legs,
            ))


def _validate_semantic_requirements(cone: CompiledCone, registry: LegRegistry) -> None:
    """Reject semantic weakening before verifier/MDL selection.

    Every declared term must be witnessed by structure the score actually
    depends on, be explicitly proxy-covered by a used leg's own contract, or
    be carried by a declared gluing.  The audit is registry-driven; there is
    no concept table to weaken.
    """
    node_types = cone.node_types
    score_deps = cone.node_dependencies.get(cone.hypothesis.score_node, frozenset())
    available = tuple(sorted(set(node_types.values())))
    known_witness_types = tuple(
        t for t in registry.terminal_types() if t.endswith("Witness"))

    # witness_requirements entries are free-form strings from the proposer:
    # any exact witness type name found inside is enforced (present and
    # load-bearing); phrases naming no known type fall through to the general
    # term audit instead of being rejected on phrasing.
    audit_extras: list[str] = []
    for requirement in getattr(cone.hypothesis, "witness_requirements", ()):
        if not requirement:
            continue
        named_types = [t for t in known_witness_types if t in requirement]
        if not named_types:
            audit_extras.append(requirement)
            continue
        for witness_type in named_types:
            if witness_type not in node_types.values():
                suggestions: tuple[str, ...] = ()
                for token in term_tokens(witness_type):
                    suggestions += leg_suggestions(token, registry)
                raise MissingLegError(MissingLeg(
                    semantic_term=witness_type,
                    required_witness_types=(witness_type,),
                    available_terminal_types=available,
                    attempted_paths=cone.used_legs,
                    missing_legs=tuple(dict.fromkeys(suggestions)),
                ))
            nodes_of_type = {n for n, t in node_types.items() if t == witness_type}
            if not nodes_of_type & score_deps:
                raise MissingLegError(MissingLeg(
                    semantic_term=witness_type,
                    required_witness_types=(witness_type,),
                    available_terminal_types=available,
                    unresolved_relation="required witness is decorative; final score does not depend on it",
                    attempted_paths=cone.used_legs,
                ))

    failures = audit_term_coverage(
        cone.hypothesis, node_types, score_deps, cone.used_legs, registry,
        extra_terms=tuple(audit_extras))
    if failures:
        raise MissingLegError(failures[0])

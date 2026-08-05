"""Mechanical compiler from semantic IR to four-disposition cone traces.

Successful values, semantic absence, indeterminacy, and implementation errors
remain distinct throughout execution; in particular, uncertainty is never
lowered to a false Boolean score.
"""
from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import numpy as np

from semantic_ir import SemanticHypothesis
from semantic_legs import (
    LegRegistry,
    WitnessAbsent,
    WitnessIndeterminate,
    is_pair_witness_codomain,
    is_witness_codomain,
    result_contract_issue,
    result_type_for_codomain,
)
from semantic_requirements import (
    MissingLeg,
    audit_term_coverage,
    calibrated_claim_signature,
    calibrated_claims,
    explicit_terms,
    is_quantity_word,
    leg_suggestions,
    metric_identities,
    plural_content_tokens,
    proxy_score_direction,
    term_matches_contract_claim,
    term_matches_phrase,
    term_matches_produced_claim,
    term_metric_compatible,
    term_tokens,
)


class CompileError(ValueError):
    pass


class MissingLegError(CompileError):
    def __init__(self, missing: MissingLeg) -> None:
        self.missing = missing
        super().__init__(missing.describe())


@dataclass(frozen=True)
class AbsentValue:
    """A typed value that is honestly absent on this panel."""

    expected_type: str
    reason: str


@dataclass(frozen=True)
class IndeterminateValue:
    """A typed value whose presence or absence could not be established."""

    expected_type: str
    reason: str


@dataclass(frozen=True)
class FailedValue:
    """A value unavailable because an upstream implementation failed."""

    expected_type: str
    reason: str


def is_absent_value(value: Any) -> bool:
    return isinstance(value, AbsentValue)


def is_indeterminate_value(value: Any) -> bool:
    return isinstance(value, IndeterminateValue)


@dataclass
class ExecutionTrace:
    node_values: dict[str, Any]
    node_types: dict[str, str]
    leg_status: dict[str, str]
    errors: tuple[str, ...] = ()
    witness_absences: dict[str, tuple[str, str]] = field(default_factory=dict)
    witness_indeterminacies: dict[str, tuple[str, str]] = field(
        default_factory=dict)


@dataclass
class CompiledCone:
    hypothesis: SemanticHypothesis
    used_legs: tuple[str, ...]
    node_types: dict[str, str]
    node_dependencies: dict[str, frozenset[str]]
    leg_dependencies: dict[str, frozenset[str]]
    complexity: int

    def trace(self, panel: np.ndarray, registry: LegRegistry) -> ExecutionTrace:
        values: dict[str, Any] = {"panel": panel}
        types: dict[str, str] = {"panel": "Panel"}
        statuses: dict[str, str] = {}
        errors: list[str] = []
        witness_absences: dict[str, tuple[str, str]] = {}
        witness_indeterminacies: dict[str, tuple[str, str]] = {}
        for edge in self.hypothesis.diagram.edges:
            call = edge.call
            leg = registry.get(call.leg_name)
            types[edge.target] = leg.codomain
            args = [values[name] for name in call.args]
            failed = next((v for v in args if isinstance(v, FailedValue)), None)
            if failed is not None:
                values[edge.target] = FailedValue(leg.codomain, failed.reason)
                statuses[edge.target] = "blocked_by_error"
                continue
            indeterminate = next(
                (v for v in args if isinstance(v, IndeterminateValue)), None)
            if indeterminate is not None:
                values[edge.target] = IndeterminateValue(
                    leg.codomain, indeterminate.reason)
                statuses[edge.target] = "blocked_by_indeterminate"
                continue
            absent = next((v for v in args if isinstance(v, AbsentValue)), None)
            if absent is not None:
                values[edge.target] = AbsentValue(leg.codomain, absent.reason)
                statuses[edge.target] = "blocked_by_absence"
                continue
            try:
                value = leg.implementation(*args, **dict(call.parameters))
                contract_issue = result_contract_issue(leg, value)
                if contract_issue:
                    raise TypeError(
                        f"codomain contract violation: {contract_issue}")
                values[edge.target] = value
                statuses[edge.target] = "ok"
            except WitnessIndeterminate as exc:
                if exc.failure_mode not in leg.indeterminate_modes:
                    reason = (
                        f"{edge.target}:{leg.name} raised undeclared witness "
                        f"indeterminacy {exc.failure_mode!r}: {exc}")
                    values[edge.target] = FailedValue(leg.codomain, reason)
                    statuses[edge.target] = (
                        "error:UndeclaredWitnessIndeterminacy")
                    errors.append(reason)
                else:
                    values[edge.target] = IndeterminateValue(
                        leg.codomain, str(exc))
                    statuses[edge.target] = "indeterminate"
                    witness_indeterminacies[edge.target] = (
                        leg.name, exc.failure_mode)
            except WitnessAbsent as exc:
                if exc.failure_mode not in leg.failure_modes:
                    reason = (
                        f"{edge.target}:{leg.name} raised undeclared witness "
                        f"absence {exc.failure_mode!r}: {exc}")
                    values[edge.target] = FailedValue(leg.codomain, reason)
                    statuses[edge.target] = "error:UndeclaredWitnessAbsence"
                    errors.append(reason)
                else:
                    values[edge.target] = AbsentValue(leg.codomain, str(exc))
                    statuses[edge.target] = "absent"
                    witness_absences[edge.target] = (
                        leg.name, exc.failure_mode)
            except Exception as exc:
                reason = f"{edge.target}:{exc}"
                values[edge.target] = FailedValue(leg.codomain, reason)
                statuses[edge.target] = f"error:{type(exc).__name__}"
                errors.append(reason)
        return ExecutionTrace(
            values, types, statuses, tuple(errors), witness_absences,
            witness_indeterminacies)

    def score(self, panel: np.ndarray,
              registry: LegRegistry) -> tuple[float | None, ExecutionTrace]:
        tr = self.trace(panel, registry)
        if tr.errors:
            return None, tr
        if self.hypothesis.score_node not in tr.node_values:
            return None, ExecutionTrace(
                node_values=tr.node_values,
                node_types=tr.node_types,
                leg_status=tr.leg_status,
                errors=(f"missing score node {self.hypothesis.score_node}",),
                witness_absences=tr.witness_absences,
                witness_indeterminacies=tr.witness_indeterminacies)
        value = tr.node_values[self.hypothesis.score_node]
        if isinstance(value, AbsentValue):
            return None, tr
        if isinstance(value, IndeterminateValue):
            return None, tr
        if isinstance(value, FailedValue):
            return None, ExecutionTrace(
                node_values=tr.node_values,
                node_types=tr.node_types,
                leg_status=tr.leg_status,
                errors=(value.reason,),
                witness_absences=tr.witness_absences,
                witness_indeterminacies=tr.witness_indeterminacies)
        try:
            score = float(value)
            if not math.isfinite(score):
                raise ValueError("non-finite")
            return score, tr
        except Exception:
            return None, ExecutionTrace(
                node_values=tr.node_values,
                node_types=tr.node_types,
                leg_status=tr.leg_status,
                errors=(f"non-numeric score node {self.hypothesis.score_node}",),
                witness_absences=tr.witness_absences,
                witness_indeterminacies=tr.witness_indeterminacies)


def compile_hypothesis(hypothesis: SemanticHypothesis,
                       registry: LegRegistry) -> CompiledCone:
    _validate_hypothesis_header(hypothesis, registry)
    env_types: dict[str, str] = {"panel": "Panel"}
    node_dependencies: dict[str, frozenset[str]] = {
        "panel": frozenset({"panel"})}
    leg_dependencies: dict[str, frozenset[str]] = {"panel": frozenset()}
    used: list[str] = []
    # Complexity is derived exclusively by the harness.  A proposal cannot
    # buy a shorter description by supplying a hint or a negative cost.
    complexity = 0
    for edge in hypothesis.diagram.edges:
        call = edge.call
        try:
            leg = registry.get(call.leg_name)
        except KeyError as exc:
            raise CompileError(str(exc)) from exc
        if len(call.args) != len(leg.domain):
            raise CompileError(f"{call.leg_name}: arity mismatch")
        parameter_names = [name for name, _ in call.parameters]
        if len(parameter_names) != len(set(parameter_names)):
            raise CompileError(f"{call.leg_name}: duplicate call parameter")
        try:
            inspect.signature(leg.implementation).bind(
                *([None] * len(call.args)), **dict(call.parameters))
        except TypeError as exc:
            raise CompileError(
                f"{call.leg_name}: invalid call parameters: {exc}") from exc
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
        node_deps = {edge.target}
        leg_deps = {call.leg_name}
        for arg in call.args:
            node_deps.update(node_dependencies.get(arg, frozenset()))
            node_deps.add(arg)
            leg_deps.update(leg_dependencies.get(arg, frozenset()))
        node_dependencies[edge.target] = frozenset(node_deps)
        leg_dependencies[edge.target] = frozenset(leg_deps)
        used.append(call.leg_name)
        complexity += leg.complexity + 1 + len(call.parameters)
    if hypothesis.score_node not in env_types:
        raise CompileError(f"score node {hypothesis.score_node} is not produced")
    if env_types[hypothesis.score_node] != "Measurement":
        raise CompileError("score node must have type Measurement")
    if hypothesis.order not in {"low_positive", "high_positive"}:
        raise CompileError("order must be low_positive or high_positive")
    complexity += sum(
        1 + len(spec.interface_fields) + len(spec.added_fields)
        + len(spec.preserved_invariants) + int(bool(spec.projection_leg))
        for spec in hypothesis.cofibrations
    )
    cone = CompiledCone(
        hypothesis, tuple(used), env_types, node_dependencies,
        leg_dependencies, complexity)
    _validate_gluings(cone, registry)
    _validate_semantic_requirements(cone, registry)
    return cone


def _validate_hypothesis_header(hypothesis: SemanticHypothesis,
                                registry: LegRegistry) -> None:
    """Validate proposal-controlled metadata before it can affect the gate."""
    if hypothesis.version != "0.1":
        raise CompileError(f"unsupported semantic IR version {hypothesis.version!r}")
    if not hypothesis.hypothesis_id.strip():
        raise CompileError("hypothesis_id must be non-empty")
    if not hypothesis.description.strip():
        raise CompileError("description must be non-empty")
    if hypothesis.polarity != "positive_satisfies":
        raise CompileError("only positive_satisfies polarity is supported")
    if hypothesis.complexity_hint != 0:
        raise CompileError("proposal-controlled complexity_hint is not accepted")
    if not hypothesis.semantic_requirements or not any(
            str(term).strip() for term in hypothesis.semantic_requirements):
        raise CompileError("semantic_requirements must declare the candidate's semantics")
    if not hypothesis.preservation_morphisms:
        raise CompileError("at least one preservation morphism must be declared")
    for morph in hypothesis.preservation_morphisms:
        if not morph.name.strip():
            raise CompileError("preservation morphism name must be non-empty")
        if morph.scope != "panel":
            raise CompileError(
                f"preservation morphism {morph.name}: only panel scope is supported")
        if morph.expected_effect != "preserve":
            raise CompileError(
                f"preservation morphism {morph.name}: expected_effect must be preserve")
        if morph.parameters:
            raise CompileError(
                f"preservation morphism {morph.name}: parameters are not "
                "executable yet")
    if hypothesis.contrast_interventions:
        raise CompileError(
            "contrast_interventions are not executable yet; refusing to record "
            "an unmeasured zero-risk declaration")

    # This IR has one polarity only: structured requirements are predicates
    # satisfied by the positive side.  Opposite-side prose cannot silently be
    # normalized into the same unscoped claim.  We permit `than negatives`
    # (and its explicit-panel variants) solely as the complement of a
    # positive-side scalar comparison; other negative-side narration needs a
    # future side-scoped IR rather than an unsafe bag-of-words interpretation.
    side_checked_description = re.sub(
        r"\bthan\s+(?:in\s+)?(?:the\s+)?(?:negatives|negative\s+"
        r"(?:panels?|figures?|images?|scenes?|objects?|examples?|class))\b",
        "than comparison class",
        hypothesis.description,
        flags=re.IGNORECASE,
    )
    if re.search(r"\bnegatives?\b", side_checked_description,
                 re.IGNORECASE):
        raise CompileError(
            "description assigns semantics to the negative side, but the "
            "current IR supports only positive_satisfies requirements")
    for term in explicit_terms(hypothesis):
        if re.search(r"\b(?:positives?|negatives?)\b", term,
                     re.IGNORECASE):
            raise CompileError(
                "semantic_requirements/relations must be side-free predicates; "
                f"class scope is fixed by positive_satisfies: {term!r}")

    score_contract = None
    score_edges = [
        edge for edge in hypothesis.diagram.edges
        if edge.target == hypothesis.score_node
    ]
    if len(score_edges) == 1:
        try:
            score_contract = registry.get(score_edges[0].call.leg_name)
        except KeyError:
            pass

    # The structured declarations are authoritative.  Every substantive term
    # in the prose description must also be named explicitly, preventing both
    # known and novel prose (for example `triangle` or `bird-like`) from
    # laundering an object-count-only structured claim.
    declared_terms = explicit_terms(hypothesis)
    declared = {
        token for term in declared_terms for token in term_tokens(term)
    }

    def declared_match(token: str) -> bool:
        if token in declared or any(
                term_matches_phrase(token, other) for other in declared):
            return True
        return bool(score_contract) \
            and term_matches_contract_claim(token, score_contract) \
            and any(term_matches_contract_claim(term, score_contract)
                    for term in declared_terms)

    framing_relation_pattern = re.compile(
        r"\b(?:figure|panel|image|scene)s?\s+"
        r"(?:contains?|consists?)\b",
        re.IGNORECASE,
    )
    framing_relations = (
        {"contain", "consist"}
        if framing_relation_pattern.search(hypothesis.description) else set()
    )
    if re.search(r"\battach(?:ed|ment)?\s+to\b",
                 hypothesis.description, re.IGNORECASE):
        framing_relations.add("to")

    undeclared = []
    for token in term_tokens(hypothesis.description):
        if is_quantity_word(token):
            continue
        if token in framing_relations:
            continue
        if not declared_match(token):
            undeclared.append(token)
    if undeclared:
        raise CompileError(
            "description names terms absent from semantic_requirements/"
            f"relations: {', '.join(dict.fromkeys(undeclared))}")

    # A bag of global markers cannot distinguish "two parts, three contacts"
    # from the same numbers attached to the opposite nouns, nor at-most from
    # exact.  Compare canonical clause-bound operators and metric identity.
    def canonical_calibration(text: str) -> tuple[tuple, ...]:
        if score_contract is None:
            return calibrated_claim_signature(text)
        claims = calibrated_claims(text)
        proxy_direction, proxy_issue = proxy_score_direction(
            text, score_contract)
        signatures: list[tuple] = []
        for claim in claims:
            operator = claim.operator
            anchor = claim.anchor
            if anchor and term_matches_contract_claim(
                    " ".join(anchor), score_contract):
                anchor = (f"score:{score_contract.name}",)
            mode = operator.mode
            target = operator.target
            direction = operator.direction
            negated = operator.negated
            if score_contract.measurement_kind == "count" \
                    and mode == "absence":
                mode, target, direction, negated = "exact", 0.0, None, False
            elif score_contract.measurement_kind == "binary" \
                    and proxy_direction is not None and not proxy_issue \
                    and target is None \
                    and mode != "unsupported":
                mode = "exact"
                target = 1.0 if proxy_direction == "high" else 0.0
                direction = None
                negated = False
            if target is not None and float(target).is_integer():
                target = int(target)
            signatures.append((anchor, mode, target, direction, negated))
        if not claims and score_contract.measurement_kind == "binary" \
                and proxy_direction is not None and not proxy_issue:
            signatures.append((
                (f"score:{score_contract.name}",), "exact",
                1 if proxy_direction == "high" else 0, None, False,
            ))
        return tuple(sorted(signatures, key=repr))

    description_calibration = canonical_calibration(hypothesis.description)
    declared_calibration = tuple(sorted((
        signature
        for term in explicit_terms(hypothesis)
        for signature in canonical_calibration(term)
    ), key=repr))
    description_metrics = set(metric_identities(
        hypothesis.description, include_generic=False))
    declared_metrics = {
        metric
        for term in explicit_terms(hypothesis)
        for metric in metric_identities(term, include_generic=False)
    }
    if description_calibration != declared_calibration \
            or description_metrics != declared_metrics:
        raise CompileError(
            "description and semantic_requirements/relations disagree on "
            "score calibration (operators must remain bound to the same "
            "claim and metric)")

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
        required = {
            "name": spec.name,
            "source_node": spec.source_node,
            "target_node": spec.target_node,
            "source_type": spec.source_type,
            "target_type": spec.target_type,
            "attachment_leg": spec.attachment_leg,
        }
        missing_fields = [name for name, value in required.items() if not value.strip()]
        if missing_fields:
            raise CompileError(
                f"gluing {spec.name or '<unnamed>'}: missing required fields "
                + ", ".join(missing_fields))
        if spec.source_node == spec.target_node:
            raise CompileError(
                f"gluing {spec.name}: source_node and target_node must be distinct")
        if not spec.interface_fields or not spec.added_fields:
            raise CompileError(
                f"gluing {spec.name}: interface_fields and added_fields must be non-empty")
        overlap = sorted(set(spec.interface_fields) & set(spec.added_fields))
        if overlap:
            raise CompileError(
                f"gluing {spec.name}: interface_fields and added_fields must "
                f"be disjoint (overlap: {', '.join(overlap)})")
        if not math.isfinite(spec.tolerance) or not (0.0 < spec.tolerance <= 2.0):
            raise CompileError(
                f"gluing {spec.name}: tolerance must be finite and in (0, 2]")
        if spec.complexity_cost != 1:
            raise CompileError(
                f"gluing {spec.name}: proposal-controlled complexity_cost "
                "is not accepted")

        target_runtime_type = result_type_for_codomain(spec.target_type)
        if target_runtime_type is not None and is_dataclass(target_runtime_type):
            target_fields = {item.name for item in fields(target_runtime_type)}
            declared_fields = set(spec.interface_fields) | set(spec.added_fields)
            unknown_fields = sorted(declared_fields - target_fields)
            if unknown_fields:
                raise CompileError(
                    f"gluing {spec.name}: fields are absent from "
                    f"{spec.target_type}: {', '.join(unknown_fields)}")
            bookkeeping = {
                "confidence", "residual", "provenance", "part_id", "role",
                "source_id", "source_component_id", "object_id", "id",
            }
            nonstructural = sorted(declared_fields & bookkeeping)
            if nonstructural:
                raise CompileError(
                    f"gluing {spec.name}: interface/patch fields must be "
                    "structural, not witness bookkeeping: "
                    + ", ".join(nonstructural))

        try:
            attachment = registry.get(spec.attachment_leg)
        except KeyError:
            raise MissingLegError(MissingLeg(
                semantic_term=spec.name,
                required_witness_types=(spec.source_type, spec.target_type),
                available_terminal_types=available,
                unresolved_relation="gluing attachment leg is not implemented",
                attempted_paths=cone.used_legs,
                missing_legs=(spec.attachment_leg,),
            ))
        if spec.target_type not in attachment.domain:
            raise CompileError(
                f"gluing {spec.name}: attachment leg {spec.attachment_leg} "
                f"must accept declared target type {spec.target_type}")
        if not is_witness_codomain(attachment.codomain):
            raise CompileError(
                f"gluing {spec.name}: attachment leg {spec.attachment_leg} "
                "must produce a typed relation witness")
        attachment_type = result_type_for_codomain(attachment.codomain)
        if attachment_type is None or not is_dataclass(attachment_type) \
                or not {"source_a", "source_b"} <= {
                    item.name for item in fields(attachment_type)}:
            raise CompileError(
                f"gluing {spec.name}: attachment leg {spec.attachment_leg} "
                "must produce a relation witness with source_a/source_b")

        for node_attr, type_attr in (("source_node", "source_type"),
                                     ("target_node", "target_type")):
            node = getattr(spec, node_attr)
            declared = getattr(spec, type_attr)
            if node not in node_types:
                raise CompileError(
                    f"gluing {spec.name}: {node_attr} {node} is not a diagram node")
            if declared and node_types[node] != declared:
                raise CompileError(
                    f"gluing {spec.name}: {node} has type {node_types[node]}, "
                    f"declared {declared}")
        if spec.projection_leg:
            try:
                projection = registry.get(spec.projection_leg)
            except KeyError as exc:
                raise CompileError(
                    f"gluing {spec.name}: missing projection leg "
                    f"{spec.projection_leg}") from exc
            if projection.domain != (spec.target_type,) \
                    or projection.codomain != spec.source_type:
                raise CompileError(
                    f"gluing {spec.name}: projection {spec.projection_leg} must be "
                    f"{spec.target_type} -> {spec.source_type}")
        attachment_edges = [
            edge for edge in cone.hypothesis.diagram.edges
            if edge.call.leg_name == spec.attachment_leg
            and spec.target_node in edge.call.args
            and edge.target in score_deps
        ]
        if not attachment_edges:
            raise MissingLegError(MissingLeg(
                semantic_term=spec.name,
                required_witness_types=(spec.source_type, spec.target_type),
                available_terminal_types=available,
                unresolved_relation=(
                    "declared attachment leg is not executed from target_node "
                    "on the final score path"),
                attempted_paths=cone.used_legs,
                missing_legs=(spec.attachment_leg,),
            ))
        source_extractors = [
            edge for edge in cone.hypothesis.diagram.edges
            if edge.target == spec.source_node
            and spec.target_node in edge.call.args
            and registry.get(edge.call.leg_name).codomain == spec.source_type
        ]
        if spec.source_node not in score_deps and not source_extractors:
            raise MissingLegError(MissingLeg(
                semantic_term=spec.name,
                required_witness_types=(spec.source_type, spec.target_type),
                available_terminal_types=available,
                unresolved_relation=(
                    "gluing source is neither load-bearing nor extracted from "
                    "the declared target"),
                attempted_paths=cone.used_legs,
            ))
        if spec.projection_leg and not any(
                edge.call.leg_name == spec.projection_leg
                for edge in source_extractors):
            raise CompileError(
                f"gluing {spec.name}: declared projection leg "
                f"{spec.projection_leg} is not the executed target-to-source edge")
        if spec.target_node not in score_deps:
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
    score_leg_deps = cone.leg_dependencies.get(
        cone.hypothesis.score_node, frozenset())
    available = tuple(sorted(set(node_types.values())))
    known_witness_types = tuple(
        t for t in registry.terminal_types() if is_witness_codomain(t))

    score_edge = next(
        edge for edge in cone.hypothesis.diagram.edges
        if edge.target == cone.hypothesis.score_node
    )
    score_contract = registry.get(score_edge.call.leg_name)
    direct_witness_domain = any(
        is_witness_codomain(domain) for domain in score_contract.domain)
    producer_contracts = tuple(
        registry.get(edge.call.leg_name)
        for edge in cone.hypothesis.diagram.edges
        if edge.target in score_edge.call.args
    )
    pair_types = {
        node_types[node]
        for node in score_deps
        if node in node_types
        and is_pair_witness_codomain(node_types[node])
    }

    # Preserve representation-poverty reporting before score-calibration
    # diagnostics.  If the load-bearing cone cannot express "circle" at all,
    # MISSING_LEG is more truthful than saying its current unrelated scalar is
    # the wrong calibration target.
    early_failures = audit_term_coverage(
        cone.hypothesis, node_types, score_deps, score_leg_deps,
        cone.used_legs, registry)
    representation_failure = next((
        failure for failure in early_failures
        if not failure.required_witness_types
        or not (set(failure.required_witness_types) & set(node_types.values()))
    ), None)
    if representation_failure is not None:
        raise MissingLegError(representation_failure)

    def anchor_matches_pair(anchor: tuple[str, ...]) -> bool:
        return bool(anchor) and any(
            all(token.lower() in pair_type.lower() for token in anchor)
            for pair_type in pair_types
        )

    # Every calibrated phrase must name the actual final measurement.  Typed
    # upstream structure alone cannot discharge "three parts" while the score
    # executes contact_count.  The one structural exception is an exact pair
    # claim backed by a load-bearing PairWitness construction.
    for term in explicit_terms(cone.hypothesis):
        named_metrics = metric_identities(term)
        if named_metrics and not term_metric_compatible(term, score_contract):
            raise CompileError(
                f"semantic score metric is not executed by "
                f"{score_contract.name}: {term!r}")
        directional_proxy = any(
            term_matches_phrase(term, proxy)
            for proxy, _direction in score_contract.proxy_directions
        )
        if score_contract.measurement_kind in {"binary", "continuous"} \
                and directional_proxy and plural_content_tokens(term):
            raise CompileError(
                f"a plural claim on scalar alias {score_contract.name} "
                f"requires a matching count Measurement: {term!r}")
        for claim in calibrated_claims(term):
            operator = claim.operator
            if not claim.anchor:
                raise CompileError(
                    f"semantic score operator has no claim anchor: {term!r}")
            anchor_text = " ".join(claim.anchor)
            matches_score = term_matches_contract_claim(
                anchor_text, score_contract)
            matches_producer = (
                score_contract.measurement_kind == "continuous"
                and direct_witness_domain and any(
                    term_matches_produced_claim(anchor_text, producer)
                    for producer in producer_contracts
                )
            )
            pair_backed = (
                operator.mode == "exact" and operator.target == 2.0
                and anchor_matches_pair(claim.anchor)
            )
            if not (matches_score or matches_producer or pair_backed):
                raise CompileError(
                    f"semantic score operator is not bound to the final "
                    f"measurement {score_contract.name}: {term!r}")

            # Unsupported syntax is diagnosed by the verifier without ever
            # being evaluated.  Association is still checked above so it
            # cannot hide an unrelated claim.
            if operator.mode == "unsupported":
                continue
            if score_contract.measurement_kind == "binary" \
                    and operator.target is not None \
                    and operator.target not in {0.0, 1.0}:
                raise CompileError(
                    f"binary score {score_contract.name} cannot execute "
                    f"target {operator.target:g}: {term!r}")
            if score_contract.measurement_kind == "binary" \
                    and operator.target is not None and directional_proxy:
                raise CompileError(
                    f"a cardinal on binary alias {score_contract.name} "
                    f"requires a matching count Measurement: {term!r}")
            if score_contract.measurement_kind != "continuous":
                continue
            if operator.target is not None and not named_metrics:
                if pair_backed:
                    continue
                raise CompileError(
                    f"structural cardinality requires a matching count "
                    f"Measurement: {term!r}")
            if operator.mode == "relative" and not named_metrics \
                    and not directional_proxy:
                raise CompileError(
                    f"relative structural quantity requires a matching "
                    f"Measurement: {term!r}")
            if operator.mode == "absence" and not (
                    direct_witness_domain or directional_proxy):
                raise CompileError(
                    f"structural absence requires an executable witness or "
                    f"count Measurement: {term!r}")

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
        cone.hypothesis, node_types, score_deps, score_leg_deps,
        cone.used_legs, registry,
        extra_terms=tuple(audit_extras))
    if failures:
        raise MissingLegError(failures[0])

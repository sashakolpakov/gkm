"""Verification for mechanically compiled semantic cones.

Risk here is empirical and reported un-collapsed:

- ``support_errors``   — misclassifications of the fitted 1-D threshold rule
  on the 12 support panels;
- ``loo_errors``       — post-selection threshold leave-one-out errors
  (threshold refit on 11 panels, one panel predicted);
- ``rotated_loo_errors`` — positive/negative pair-rotated holdout errors
  (72 predictions for a 6+6 problem, matching the established crack protocol);
- ``naturality_errors``— violations of the cone's own declared preservation
  morphisms, executed as actual panel transforms (cone invariance);
- ``cofibration_errors``— per-panel failures of proposer-declared gluings.

All checks are general: the harness executes declared structure; it knows no
problem-specific concepts.  Morphisms without an exact pixel action (e.g.
uniform_scale on 1-px strokes) are reported in ``unchecked_morphisms``
instead of being silently passed or faked.

An indeterminate witness is neither present nor absent.  It yields no Boolean
prediction, fails support and holdout checks regardless of side, and is an
unconditional semantic-admission failure.  This prevents extractor abstention
on a negative panel from masquerading as a correct ``False`` decision.

The proposer sees all labeled panels before choosing the cone. Consequently
the LOO coordinates diagnose threshold stability only; they are not an
untouched representation-level estimate of semantic generalization.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field

import numpy as np

from cofibrations import verify_cofibration
from dataset import Problem
from semantic_compiler import (
    CompiledCone,
    CompileError,
    MissingLegError,
    compile_hypothesis,
    is_absent_value,
    is_indeterminate_value,
)
from semantic_ir import MorphSpec, SemanticHypothesis
from semantic_legs import (
    LegRegistry,
    is_pair_witness_codomain,
    is_witness_codomain,
)
from semantic_requirements import (
    CalibratedClaim,
    ScoreOperator,
    calibrated_claims,
    explicit_terms,
    metric_identities,
    plural_content_tokens,
    proxy_score_direction,
    parse_score_operator,
    raw_word_tokens,
    term_matches_contract_claim,
    term_matches_phrase,
    term_matches_produced_claim,
)
from semantic_selection import (
    ComplexityBreakdown,
    RiskVector,
    complexity_for_cone,
)


@dataclass(frozen=True)
class ThresholdRule:
    node: str
    order: str
    threshold: float

    def predict(self, score: float | None) -> bool:
        if score is None or not math.isfinite(float(score)):
            return False
        return score <= self.threshold if self.order == "low_positive" else score >= self.threshold

    def describe(self) -> str:
        op = "<=" if self.order == "low_positive" else ">="
        return f"{self.node}{op}{self.threshold:.5g}"


@dataclass(frozen=True)
class ScoreConstraint:
    mode: str
    target: float | None = None

    def holds(self, value: float) -> bool:
        target = self.target
        if self.mode == "presence":
            return value > 0.0
        if self.mode == "exact":
            return math.isclose(value, float(target), rel_tol=1e-9,
                                abs_tol=1e-9)
        if self.mode == "not_exact":
            return not math.isclose(value, float(target), rel_tol=1e-9,
                                    abs_tol=1e-9)
        if self.mode == "at_least":
            return value >= float(target)
        if self.mode == "at_most":
            return value <= float(target)
        if self.mode == "greater_than":
            return value > float(target)
        if self.mode == "less_than":
            return value < float(target)
        raise AssertionError(self.mode)

    def describe(self) -> str:
        operators = {
            "presence": ">0",
            "exact": "==",
            "not_exact": "!=",
            "at_least": ">=",
            "at_most": "<=",
            "greater_than": ">",
            "less_than": "<",
        }
        if self.mode == "presence":
            return operators[self.mode]
        target = float(self.target)
        rendered = str(int(target)) if target.is_integer() else f"{target:.5g}"
        return operators[self.mode] + rendered


@dataclass(frozen=True)
class FixedRule:
    node: str
    constraints: tuple[ScoreConstraint, ...] = ()
    witness_presence: bool | None = None
    threshold: float = 0.0

    def predict(self, score: float | None) -> bool:
        finite = score is not None and math.isfinite(float(score))
        if self.witness_presence is not None:
            return finite is self.witness_presence
        if not finite:
            return False
        return all(constraint.holds(float(score))
                   for constraint in self.constraints)

    def describe(self) -> str:
        if self.witness_presence is not None:
            state = "present" if self.witness_presence else "absent"
            return f"{self.node}:{state}"
        return " & ".join(
            f"{self.node}{constraint.describe()}"
            for constraint in self.constraints
        ) or f"{self.node}:invalid"


@dataclass(frozen=True)
class SemanticDecisionPlan:
    fixed_constraints: tuple[ScoreConstraint, ...] = ()
    witness_presence: bool | None = None
    score_direction: str | None = None
    issue: str = ""

    @property
    def fixed(self) -> bool:
        return bool(self.fixed_constraints) or self.witness_presence is not None


@dataclass
class ConeVerification:
    hypothesis_id: str
    accepted: bool
    support_accuracy: float
    loo_accuracy: float
    support_errors: int
    loo_errors: int
    n_examples: int
    rule: str
    threshold: float
    fold_threshold_min: float
    fold_threshold_max: float
    predicate_errors: int
    complexity: int
    rotated_loo_accuracy: float = 0.0
    rotated_loo_errors: int = 0
    rotated_loo_checks: int = 0
    naturality_errors: int = 0
    cofibration_errors: int = 0
    unchecked_morphisms: tuple[str, ...] = ()
    declared_morphism_checks: int = 0
    worst_transform: str = ""
    stress_errors: int = 0
    stress_checks: int = 0
    worst_stress_transform: str = ""
    structural_absences: int = 0
    witness_absences: dict[str, int] = field(default_factory=dict)
    indeterminate_evaluations: int = 0
    witness_indeterminacies: dict[str, int] = field(default_factory=dict)
    semantic_admissible: bool = False
    risk: RiskVector = field(default_factory=RiskVector)
    complexity_breakdown: ComplexityBreakdown = field(
        default_factory=ComplexityBreakdown)
    compile_error: str = ""
    semantic_issue: str = ""
    missing_leg: dict | None = None
    scores: tuple[float | None, ...] = field(default_factory=tuple)
    score_dispositions: tuple[str, ...] = field(default_factory=tuple)
    support_predictions: tuple[bool | None, ...] = field(default_factory=tuple)
    support_labels: tuple[bool, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return asdict(self)


def _fit_threshold(scores: np.ndarray, labels: np.ndarray, order: str) -> ThresholdRule:
    finite = np.isfinite(scores)
    uniq = np.unique(scores[finite])
    if len(uniq) == 0:
        return ThresholdRule("score", order, 0.0)
    if len(uniq) == 1:
        thresholds = uniq
    else:
        thresholds = np.concatenate((
            [uniq[0] - 1e-9],
            (uniq[:-1] + uniq[1:]) / 2.0,
            [uniq[-1] + 1e-9],
        ))
    best = None
    for t in thresholds:
        pred = np.zeros(len(scores), dtype=bool)
        pred[finite] = (scores[finite] <= t if order == "low_positive"
                        else scores[finite] >= t)
        errors = int(np.sum(pred != labels))
        margin = float(np.min(np.abs(scores[finite] - t)))
        key = (errors, -margin, abs(float(t)))
        if best is None or key < best[0]:
            best = (key, ThresholdRule("score", order, float(t)))
    return best[1]


def _score_contract(cone: CompiledCone, registry: LegRegistry):
    edge = next(
        edge for edge in cone.hypothesis.diagram.edges
        if edge.target == cone.hypothesis.score_node
    )
    return edge, registry.get(edge.call.leg_name)


def _proxy_direction(term: str, operator: ScoreOperator, contract
                     ) -> tuple[str | None, str]:
    return proxy_score_direction(term, contract)


def _score_context(cone: CompiledCone, registry: LegRegistry):
    edge, contract = _score_contract(cone, registry)
    producers = tuple(
        registry.get(candidate.call.leg_name)
        for candidate in cone.hypothesis.diagram.edges
        if candidate.target in edge.call.args
    )
    score_deps = cone.node_dependencies.get(
        cone.hypothesis.score_node, frozenset())
    pair_types = {
        cone.node_types[node]
        for node in score_deps
        if node in cone.node_types
        and is_pair_witness_codomain(cone.node_types[node])
    }
    return edge, contract, producers, pair_types


def _claim_matches_pair(claim: CalibratedClaim,
                        pair_types: set[str]) -> bool:
    return (
        claim.operator.mode == "exact"
        and claim.operator.target == 2.0
        and bool(claim.anchor)
        and any(all(token.lower() in pair_type.lower()
                    for token in claim.anchor)
                for pair_type in pair_types)
    )


def _unsupported_operator_issue(term: str, measurement_kind: str) -> str:
    words = set(raw_word_tokens(term))
    if measurement_kind == "continuous" and words & {
            "high", "low", "large", "small", "long", "short"}:
        return "semantic_relative_measurement_requires_direction"
    prefix = "semantic_count" if measurement_kind in {"count", "binary"} \
        else "semantic_score"
    return f"{prefix}_operator_unsupported"


def _semantic_decision_plan(
        cone: CompiledCone, registry: LegRegistry) -> SemanticDecisionPlan:
    _edge, contract, producers, pair_types = _score_context(cone, registry)
    terms = explicit_terms(cone.hypothesis)
    direct_witness_domain = any(
        is_witness_codomain(domain) for domain in contract.domain)
    constraints: list[ScoreConstraint] = []
    directions: set[str] = set()
    categorical: list[bool] = []
    associated = False
    has_relative_or_numeric_metric = False
    has_learned_relative = False

    for term in terms:
        matches_score = term_matches_contract_claim(term, contract)
        matches_producer = (
            contract.measurement_kind == "continuous"
            and direct_witness_domain and any(
                term_matches_produced_claim(term, producer)
                for producer in producers
            )
        )
        term_claims = calibrated_claims(term)
        pair_claims = tuple(
            claim for claim in term_claims
            if _claim_matches_pair(claim, pair_types)
        )
        if pair_claims and len(pair_claims) == len(term_claims):
            # The exact-two structure has already been executed by the typed
            # pair constructor; it is not a predicate on this scalar score.
            associated = True
            continue
        if not (matches_score or matches_producer):
            continue
        associated = True

        if contract.measurement_kind in {"binary", "continuous"} \
                and contract.proxy_directions \
                and plural_content_tokens(term) \
                and any(term_matches_phrase(term, proxy)
                        for proxy, _direction in contract.proxy_directions):
            return SemanticDecisionPlan(
                issue="semantic_scalar_plural_requires_count")

        if not term_claims:
            operator = ScoreOperator(None)
            claims = (CalibratedClaim((), operator),)
        else:
            claims = tuple(claim for claim in term_claims
                           if claim not in pair_claims)
        for claim in claims:
            operator = claim.operator
            if operator.mode == "unsupported":
                return SemanticDecisionPlan(issue=_unsupported_operator_issue(
                    term, contract.measurement_kind))

            if contract.measurement_kind == "count":
                if operator.mode is None:
                    constraints.append(ScoreConstraint("presence"))
                elif operator.mode == "absence":
                    constraints.append(ScoreConstraint("exact", 0.0))
                elif operator.mode == "relative":
                    if operator.direction:
                        directions.add(operator.direction)
                        has_learned_relative = True
                elif operator.mode in {
                        "exact", "not_exact", "at_least", "at_most",
                        "greater_than", "less_than"}:
                    constraints.append(ScoreConstraint(
                        operator.mode, operator.target))
                    if operator.direction:
                        directions.add(operator.direction)
                else:
                    return SemanticDecisionPlan(
                        issue="semantic_count_operator_unsupported")
                continue

            if contract.measurement_kind == "binary":
                proxy_direction, issue = _proxy_direction(
                    term, operator, contract)
                if issue:
                    return SemanticDecisionPlan(issue=issue)
                if operator.target is not None:
                    if proxy_direction is not None:
                        return SemanticDecisionPlan(
                            issue="semantic_binary_cardinal_requires_count")
                    if operator.target not in {0.0, 1.0}:
                        return SemanticDecisionPlan(
                            issue="semantic_binary_target_out_of_range")
                    constraints.append(ScoreConstraint(
                        operator.mode, operator.target))
                    if operator.direction:
                        directions.add(operator.direction)
                elif proxy_direction:
                    target = 1.0 if proxy_direction == "high" else 0.0
                    constraints.append(ScoreConstraint("exact", target))
                    directions.add(proxy_direction)
                elif operator.mode == "absence":
                    constraints.append(ScoreConstraint("exact", 0.0))
                    directions.add("low")
                elif operator.mode == "relative":
                    if operator.direction:
                        directions.add(operator.direction)
                elif operator.mode is None:
                    constraints.append(ScoreConstraint("presence"))
                continue

            # Continuous measurements have three honest modes: a fixed
            # numeric predicate, an explicitly directional scalar comparison,
            # or presence/absence of the final typed witness.
            named_metrics = metric_identities(term)
            proxy_direction, issue = _proxy_direction(
                term, operator, contract)
            if issue:
                return SemanticDecisionPlan(issue=issue)
            if operator.target is not None:
                constraints.append(ScoreConstraint(
                    operator.mode, operator.target))
                has_relative_or_numeric_metric = True
                if operator.direction:
                    directions.add(operator.direction)
            elif proxy_direction:
                directions.add(proxy_direction)
                has_relative_or_numeric_metric = True
                has_learned_relative = True
            elif operator.mode == "relative":
                if not operator.direction:
                    return SemanticDecisionPlan(
                        issue="semantic_relative_measurement_requires_direction")
                directions.add(operator.direction)
                has_relative_or_numeric_metric = True
                has_learned_relative = True
            elif named_metrics:
                return SemanticDecisionPlan(
                    issue="semantic_relative_measurement_requires_direction")
            elif direct_witness_domain and (matches_score or matches_producer):
                categorical.append(operator.mode != "absence")

    if len(directions) > 1 and (has_learned_relative or not constraints):
        return SemanticDecisionPlan(issue="semantic_score_direction_conflict")
    # A two-sided fixed interval naturally contains both a low and a high
    # bound.  It is non-monotone as a whole, so `order` is irrelevant; the
    # conjunction itself is executed in every protocol.
    direction = next(iter(directions)) if len(directions) == 1 else None
    if direction and cone.hypothesis.order != f"{direction}_positive":
        return SemanticDecisionPlan(issue=(
            f"semantic_score_direction_mismatch:{direction}:"
            f"{cone.hypothesis.order}"))
    if not associated:
        return SemanticDecisionPlan(
            issue="semantic_measurement_has_no_calibrated_claim")
    if constraints and has_learned_relative:
        return SemanticDecisionPlan(
            issue="semantic_mixed_absolute_relative_constraints")
    if categorical and not has_relative_or_numeric_metric and not constraints:
        if len(set(categorical)) > 1:
            return SemanticDecisionPlan(
                issue="semantic_witness_presence_conflict")
        return SemanticDecisionPlan(
            witness_presence=categorical[0], score_direction=direction)
    if constraints:
        return SemanticDecisionPlan(
            fixed_constraints=tuple(constraints), score_direction=direction)
    if direction:
        return SemanticDecisionPlan(score_direction=direction)
    return SemanticDecisionPlan(
        issue="semantic_measurement_has_no_calibrated_claim")


def _semantic_score_issue(plan: SemanticDecisionPlan, rule,
                          scores: np.ndarray, n_pos: int,
                          measurement_kind: str) -> str:
    if plan.issue:
        return plan.issue
    positive = scores[:n_pos]
    negative = scores[n_pos:]
    if plan.witness_presence is not None:
        finite_positive = np.isfinite(positive)
        finite_negative = np.isfinite(negative)
        if plan.witness_presence:
            if not finite_positive.all():
                return "semantic_witness_claim_missing_on_positive"
            if finite_negative.any():
                return "semantic_witness_claim_present_on_negative"
        else:
            if finite_positive.any():
                return "semantic_witness_absence_violated_on_positive"
            if not finite_negative.all():
                return "semantic_witness_absence_also_holds_on_negative"
        return ""
    if plan.fixed_constraints:
        if not np.isfinite(scores).all():
            return "semantic_fixed_calibration_requires_finite_scores"
        positive_ok = np.array([rule.predict(float(value))
                                for value in positive], dtype=bool)
        negative_ok = np.array([rule.predict(float(value))
                                for value in negative], dtype=bool)
        prefix = "semantic_count" if measurement_kind in {"count", "binary"} \
            else "semantic_measurement"
        if not positive_ok.all():
            failed_mode = next(
                (constraint.mode for constraint in plan.fixed_constraints
                 if not all(constraint.holds(float(value))
                            for value in positive)),
                "constraint",
            )
            return f"{prefix}_positive_violates_{failed_mode}"
        # Constraints form a conjunction: a negative is a counterexample only
        # when it satisfies all conjuncts, not merely one of them.
        if negative_ok.any():
            suffix = (plan.fixed_constraints[0].mode
                      if len(plan.fixed_constraints) == 1 else "conjunction")
            return f"{prefix}_negative_satisfies_{suffix}"
    return ""


# Exact pixel actions for declared preservation morphisms.  Only transforms
# that provably preserve the 1-px stroke encoding are executed; anything
# else is reported as unchecked rather than being approximated dishonestly.

def _translate_panel(panel: np.ndarray) -> np.ndarray | None:
    dy, dx = 6, 4
    ink = int((np.asarray(panel) > 0).sum())
    out = np.zeros_like(panel)
    out[dy:, dx:] = panel[:panel.shape[0] - dy, :panel.shape[1] - dx]
    if int((out > 0).sum()) != ink:
        return None  # ink would be clipped; this panel is unverifiable here
    return out


def _offgrid_rotation(panel: np.ndarray, degrees: float) -> np.ndarray | None:
    """Connectivity-preserving off-grid rotation.

    Grid-exact 90-degree rotation hides anisotropy: a measurement can be
    stable under rot90 yet swing wildly at 45 degrees (axis-aligned bounding
    boxes, raster skeleton degree counts).  Bilinear interpolation keeps the
    stroke connected (unlike nearest-neighbour, which shatters it), so this is
    a fair test of true rotational invariance.
    """
    try:
        import scipy.ndimage as ndi
    except Exception:
        return None
    rot = ndi.rotate(np.asarray(panel, dtype=float), degrees, order=1,
                     reshape=False)
    out = (rot > 0.25).astype(np.asarray(panel).dtype)
    if int((out > 0).sum()) == 0:
        return None
    return out


def _dataset_stress_transforms(panel: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Broad robustness diagnostics kept separate from declared naturality.

    Bongard-LOGO renders every shape at a random orientation, so orientation
    is normally a nuisance. Reflection is not valid for every possible concept
    (chirality is the obvious counterexample), so this battery is diagnostic;
    only declared preservation morphisms participate in semantic admission.
    """
    battery: list[tuple[str, np.ndarray]] = []
    arr = np.asarray(panel)
    for k in (1, 2, 3):
        battery.append((f"rot{90 * k}", np.rot90(arr, k).copy()))
    battery.append(("reflect", np.fliplr(arr).copy()))
    for deg in (30.0, 55.0):
        rotated = _offgrid_rotation(arr, deg)
        if rotated is not None:
            battery.append((f"rot{int(deg)}", rotated))
    translated = _translate_panel(arr)
    if translated is not None:
        battery.append(("translate", translated))
    return battery


def _normal_morphism_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _declared_transforms(morph: MorphSpec, panel: np.ndarray
                         ) -> list[tuple[str, np.ndarray]] | None:
    """Resolve a declared panel morphism to honest executable actions.

    ``None`` means unsupported and must be reported. An empty list means the
    action is supported but cannot be applied to this panel without clipping.
    """
    name = _normal_morphism_name(morph.name)
    arr = np.asarray(panel)
    if name in {"translate", "translation"}:
        moved = _translate_panel(arr)
        return [] if moved is None else [(name, moved)]

    exact_rotation = re.fullmatch(r"(?:rotate|rotation)_?(90|180|270)", name)
    if exact_rotation:
        degrees = int(exact_rotation.group(1))
        return [(name, np.rot90(arr, degrees // 90).copy())]

    if name in {"rotate", "rotation"}:
        transforms = [
            ("rot90", np.rot90(arr, 1).copy()),
            ("rot180", np.rot90(arr, 2).copy()),
            ("rot270", np.rot90(arr, 3).copy()),
        ]
        for degrees in (30.0, 55.0):
            rotated = _offgrid_rotation(arr, degrees)
            if rotated is not None:
                transforms.append((f"rot{int(degrees)}", rotated))
        return transforms

    if name in {"reflect", "reflection", "mirror"}:
        return [
            ("reflect_horizontal", np.flipud(arr).copy()),
            ("reflect_vertical", np.fliplr(arr).copy()),
        ]
    if name in {"reflect_horizontal", "reflection_horizontal", "mirror_horizontal"}:
        return [(name, np.flipud(arr).copy())]
    if name in {"reflect_vertical", "reflection_vertical", "mirror_vertical"}:
        return [(name, np.fliplr(arr).copy())]

    # Uniform scaling has no exact action on a one-pixel raster stroke in the
    # current substrate. Never silently approximate it.
    return None


def _resolve_projection_fn(spec, registry: LegRegistry):
    if not spec.projection_leg:
        return None
    try:
        leg = registry.get(spec.projection_leg)
    except KeyError:
        return None
    if leg.domain == (spec.target_type,):
        return leg.implementation
    return None


def verify_compiled_cone(cone: CompiledCone, registry: LegRegistry,
                         problem: Problem, max_support_errors: int = 0,
                         max_loo_errors: int = 0,
                         max_rotated_loo_errors: int = 0) -> ConeVerification:
    panels = [p for p, _ in problem.panels()]
    labels = np.array([lab for _, lab in problem.panels()], dtype=bool)
    scores = np.full(len(panels), np.nan, dtype=float)
    predicate_errors = 0
    structural_absences = 0
    indeterminate_evaluations = 0
    cofibration_errors = 0
    traces = []
    witness_absences: dict[str, int] = {}
    witness_indeterminacies: dict[str, int] = {}
    score_dispositions: list[str] = []
    for i, panel in enumerate(panels):
        score, trace = cone.score(panel, registry)
        indeterminate = bool(trace.witness_indeterminacies) or any(
            is_indeterminate_value(value)
            for value in trace.node_values.values())
        if score is not None:
            scores[i] = score
        if trace.errors:
            score_dispositions.append("error")
        elif indeterminate:
            score_dispositions.append("indeterminate")
            indeterminate_evaluations += 1
        elif score is None:
            score_dispositions.append("semantic_absent")
            structural_absences += 1
        else:
            score_dispositions.append("present")
        traces.append(trace)
        predicate_errors += int(bool(trace.errors))
        side = "pos" if i < len(problem.pos) else "neg"
        for node, (leg_name, failure_mode) in trace.witness_absences.items():
            key = f"{side}:{node}:{leg_name}:{failure_mode}"
            witness_absences[key] = witness_absences.get(key, 0) + 1
        for node, (leg_name, failure_mode) \
                in trace.witness_indeterminacies.items():
            key = f"{side}:{node}:{leg_name}:{failure_mode}"
            witness_indeterminacies[key] = (
                witness_indeterminacies.get(key, 0) + 1)

    for spec in cone.hypothesis.cofibrations:
        if not (spec.source_node and spec.target_node):
            continue
        projection_fn = _resolve_projection_fn(spec, registry)
        attachment_nodes = tuple(
            edge.target for edge in cone.hypothesis.diagram.edges
            if edge.call.leg_name == spec.attachment_leg
            and spec.target_node in edge.call.args
        )
        for trace, expected_positive in zip(traces, labels):
            if trace.errors or trace.witness_indeterminacies:
                if expected_positive:
                    cofibration_errors += 1
                continue
            # Positive-satisfies polarity claims this gluing for the positive
            # side.  A negative panel is allowed to contain the carrier
            # objects while lacking the actual attachment relation.
            if not expected_positive:
                continue
            attachment_values = tuple(
                trace.node_values[node]
                for node in attachment_nodes
                if node in trace.node_values
                and not is_absent_value(trace.node_values[node])
                and not is_indeterminate_value(trace.node_values[node])
            )
            if not attachment_values:
                # Verifying only that the carrier graph contains the source is
                # insufficient: the declared attachment relation itself must
                # have produced a witness on every positive panel.
                cofibration_errors += 1
                continue
            source = trace.node_values.get(spec.source_node)
            target = trace.node_values.get(spec.target_node)
            if source is None or target is None \
                    or is_absent_value(source) or is_absent_value(target) \
                    or is_indeterminate_value(source) \
                    or is_indeterminate_value(target):
                # With positive_satisfies polarity the declared gluing must
                # actually exist on every positive panel.  Negative panels
                # may honestly lack it.
                cofibration_errors += 1
                continue
            try:
                checks = tuple(
                    verify_cofibration(
                        source, target, spec, projection_fn=projection_fn,
                        attachment_value=attachment_value)
                    for attachment_value in attachment_values
                )
            except Exception:
                # Proposer-declared field names or a projection implementation
                # must fail the gluing check, never escape the verifier.
                cofibration_errors += 1
                continue
            if not any(check.ok for check in checks):
                cofibration_errors += 1

    plan = _semantic_decision_plan(cone, registry)
    _score_edge, score_contract = _score_contract(cone, registry)
    if plan.fixed:
        first_target = next((
            constraint.target for constraint in plan.fixed_constraints
            if constraint.target is not None
        ), 0.0)
        full_rule = FixedRule(
            cone.hypothesis.score_node,
            plan.fixed_constraints,
            plan.witness_presence,
            float(first_target),
        )
    else:
        fitted = _fit_threshold(scores, labels, cone.hypothesis.order)
        full_rule = ThresholdRule(
            cone.hypothesis.score_node, cone.hypothesis.order,
            fitted.threshold)
    support_pred: list[bool | None] = []
    for score, disposition in zip(scores, score_dispositions):
        if disposition in {"indeterminate", "error"}:
            support_pred.append(None)
        else:
            support_pred.append(full_rule.predict(
                float(score) if math.isfinite(float(score)) else None))
    # Unknown is its own disposition.  It always fails an empirical check;
    # it is never silently cast to False (which would reward uncertainty on
    # the negative side).
    support_errors = sum(
        prediction is None or prediction != bool(label)
        for prediction, label in zip(support_pred, labels))

    # Execute the proposal's declared preservation morphisms. Unsupported
    # declarations remain explicit instead of receiving a silent zero risk.
    naturality_errors = 0
    drift_by_transform: dict[str, int] = {}
    unchecked: list[str] = []
    declared_morphism_checks = 0
    for morph in cone.hypothesis.preservation_morphisms:
        applied = 0
        skipped = 0
        unsupported = False
        for i, panel in enumerate(panels):
            transforms = _declared_transforms(morph, panel)
            if transforms is None:
                unsupported = True
                break
            if not transforms:
                skipped += 1
            for name, transformed in transforms:
                applied += 1
                declared_morphism_checks += 1
                new_score, trace = cone.score(transformed, registry)
                transformed_indeterminate = bool(
                    trace.witness_indeterminacies) or any(
                        is_indeterminate_value(value)
                        for value in trace.node_values.values())
                original_prediction = support_pred[i]
                broke = bool(trace.errors) or transformed_indeterminate \
                    or original_prediction is None or (
                        full_rule.predict(new_score) != original_prediction)
                if broke:
                    naturality_errors += 1
                    drift_by_transform[name] = drift_by_transform.get(name, 0) + 1
        if unsupported:
            unchecked.append(morph.name)
        elif skipped:
            unchecked.append(
                f"{morph.name} (not applicable to {skipped}/"
                f"{len(panels)} panels)")
        elif applied == 0:
            unchecked.append(f"{morph.name} (no applicable panels)")
    worst_transform = max(drift_by_transform, key=drift_by_transform.get) \
        if drift_by_transform else ""

    # Dataset-wide stress is valuable primitive feedback, but it is not the
    # same claim as proposal-declared naturality (reflection can change a
    # chiral concept). Keep it separately visible and out of admission.
    stress_errors = 0
    stress_checks = 0
    stress_by_transform: dict[str, int] = {}
    for i, panel in enumerate(panels):
        for name, transformed in _dataset_stress_transforms(panel):
            stress_checks += 1
            new_score, trace = cone.score(transformed, registry)
            transformed_indeterminate = bool(
                trace.witness_indeterminacies) or any(
                    is_indeterminate_value(value)
                    for value in trace.node_values.values())
            original_prediction = support_pred[i]
            broke = bool(trace.errors) or transformed_indeterminate \
                or original_prediction is None or (
                    full_rule.predict(new_score) != original_prediction)
            if broke:
                stress_errors += 1
                stress_by_transform[name] = stress_by_transform.get(name, 0) + 1
    worst_stress_transform = max(
        stress_by_transform, key=stress_by_transform.get) \
        if stress_by_transform else ""

    correct = 0
    total = 0
    thresholds = []
    for held_idx in range(len(labels)):
        mask = np.array([k != held_idx for k in range(len(labels))])
        fold = (full_rule if plan.fixed else
                _fit_threshold(scores[mask], labels[mask],
                               cone.hypothesis.order))
        thresholds.append(fold.threshold)
        held_score = (float(scores[held_idx])
                      if math.isfinite(float(scores[held_idx])) else None)
        if score_dispositions[held_idx] not in {"indeterminate", "error"}:
            correct += int(fold.predict(held_score) == labels[held_idx])
        total += 1
    loo_errors = total - correct

    # The established crack protocol uses pair-rotated threshold refitting: for
    # every positive/negative pair, fit on the other ten panels and predict
    # both held-out panels.  This yields 72 predictions for a 6+6 problem and
    # prevents a selector from benefiting from which side was held out.
    rotated_correct = 0
    rotated_total = 0
    n_pos = len(problem.pos)
    for pos_idx in range(n_pos):
        for neg_offset in range(len(problem.neg)):
            neg_idx = n_pos + neg_offset
            mask = np.array([
                k not in (pos_idx, neg_idx) for k in range(len(labels))
            ])
            fold = (full_rule if plan.fixed else
                    _fit_threshold(scores[mask], labels[mask],
                                   cone.hypothesis.order))
            for held_idx in (pos_idx, neg_idx):
                held_score = (
                    float(scores[held_idx])
                    if math.isfinite(float(scores[held_idx])) else None
                )
                if score_dispositions[held_idx] not in {
                        "indeterminate", "error"}:
                    rotated_correct += int(
                        fold.predict(held_score) == labels[held_idx])
                rotated_total += 1
    rotated_loo_errors = rotated_total - rotated_correct
    semantic_issue = (
        semantic_quality_issue(cone)
        or _semantic_score_issue(
            plan, full_rule, scores, n_pos,
            score_contract.measurement_kind)
    )
    semantic_admissible = (
        predicate_errors == 0
        and indeterminate_evaluations == 0
        and naturality_errors == 0
        and cofibration_errors == 0
        and not unchecked
        and not semantic_issue
    )
    accepted = (
        semantic_admissible
        and support_errors <= max_support_errors
        and loo_errors <= max_loo_errors
        and rotated_loo_errors <= max_rotated_loo_errors
    )
    risk = RiskVector(
        R_support=support_errors / len(labels),
        R_rotated_LOO=(
            rotated_loo_errors / rotated_total if rotated_total else None),
        R_naturality=(
            naturality_errors / declared_morphism_checks
            if declared_morphism_checks and not unchecked else None),
        # Contrast/counterfactual/archive checks are not implemented in this
        # runner.  They remain explicit nulls, never successful zeros.
        R_contrast=None,
        R_counterfactual=None,
        # This coordinate is the whole-cone decision drift under the broad
        # dataset stress battery, not proposal-declared naturality.
        R_parser_stability=(
            stress_errors / stress_checks if stress_checks else None),
        R_archive_regression=None,
    )
    complexity_breakdown = complexity_for_cone(
        cone,
        # The checked-in registry is the conditioning library L; its
        # definitions and witness types are not re-charged per candidate.
        promoted_legs=set(registry.names()),
        promoted_witness_types=set(registry.terminal_types()),
    )
    return ConeVerification(
        hypothesis_id=cone.hypothesis.hypothesis_id,
        accepted=accepted,
        support_accuracy=1.0 - support_errors / len(labels),
        loo_accuracy=correct / total if total else 0.0,
        support_errors=support_errors,
        loo_errors=loo_errors,
        n_examples=len(labels),
        rule=full_rule.describe(),
        threshold=full_rule.threshold,
        fold_threshold_min=float(min(thresholds)) if thresholds else full_rule.threshold,
        fold_threshold_max=float(max(thresholds)) if thresholds else full_rule.threshold,
        predicate_errors=predicate_errors,
        complexity=complexity_breakdown.total,
        rotated_loo_accuracy=(
            rotated_correct / rotated_total if rotated_total else 0.0),
        rotated_loo_errors=rotated_loo_errors,
        rotated_loo_checks=rotated_total,
        naturality_errors=naturality_errors,
        cofibration_errors=cofibration_errors,
        unchecked_morphisms=tuple(unchecked),
        declared_morphism_checks=declared_morphism_checks,
        worst_transform=worst_transform,
        stress_errors=stress_errors,
        stress_checks=stress_checks,
        worst_stress_transform=worst_stress_transform,
        structural_absences=structural_absences,
        witness_absences=witness_absences,
        indeterminate_evaluations=indeterminate_evaluations,
        witness_indeterminacies=witness_indeterminacies,
        semantic_admissible=semantic_admissible,
        risk=risk,
        complexity_breakdown=complexity_breakdown,
        semantic_issue=semantic_issue,
        scores=tuple(
            float(s) if math.isfinite(float(s)) else None for s in scores),
        score_dispositions=tuple(score_dispositions),
        support_predictions=tuple(support_pred),
        support_labels=tuple(bool(value) for value in labels),
    )


def _failed_verification(hypothesis: SemanticHypothesis, problem: Problem,
                         rule: str,
                         compile_error: str, semantic_issue: str = "",
                         missing_leg: dict | None = None) -> ConeVerification:
    n_examples = len(problem.pos) + len(problem.neg)
    rotated_checks = 2 * len(problem.pos) * len(problem.neg)
    return ConeVerification(
        hypothesis_id=hypothesis.hypothesis_id,
        accepted=False,
        support_accuracy=0.0,
        loo_accuracy=0.0,
        support_errors=n_examples,
        loo_errors=n_examples,
        n_examples=n_examples,
        rule=rule,
        threshold=0.0,
        fold_threshold_min=0.0,
        fold_threshold_max=0.0,
        predicate_errors=0,
        complexity=0,
        rotated_loo_errors=rotated_checks,
        rotated_loo_checks=rotated_checks,
        compile_error=compile_error,
        semantic_issue=semantic_issue,
        missing_leg=missing_leg,
    )


def verify_hypothesis(hypothesis: SemanticHypothesis, registry: LegRegistry,
                      problem: Problem, max_support_errors: int = 0,
                      max_loo_errors: int = 0,
                      max_rotated_loo_errors: int = 0) -> ConeVerification:
    try:
        cone = compile_hypothesis(hypothesis, registry)
    except MissingLegError as exc:
        return _failed_verification(hypothesis, problem, "MISSING_LEG", str(exc),
                                    "MISSING_LEG", exc.missing.to_dict())
    except CompileError as exc:
        return _failed_verification(
            hypothesis, problem, "COMPILE_ERROR", str(exc))
    return verify_compiled_cone(
        cone, registry, problem,
        max_support_errors=max_support_errors,
        max_loo_errors=max_loo_errors,
        max_rotated_loo_errors=max_rotated_loo_errors,
    )


def semantic_quality_issue(cone: CompiledCone) -> str:
    """Classify cones that separate panels but are not human-like semantics.

    This is intentionally conservative. It does not try to judge the English
    solution; it only rejects the worst known failure: a direct panel-level
    measurement pretending to be a semantic cone.
    """
    hyp = cone.hypothesis
    score_nodes = cone.node_dependencies.get(hyp.score_node, frozenset())
    leg_names = set(cone.leg_dependencies.get(
        hyp.score_node, frozenset()))
    has_scene = "parse_scene" in leg_names
    has_object_or_relation = any(
        cone.node_types.get(node) in {"Scene", "Object", "Relation"}
        for node in score_nodes
    )
    if not hyp.description.strip():
        return "missing_human_description"
    if not hyp.preservation_morphisms:
        return "missing_declared_morphisms"
    if hyp.score_node == "score" and leg_names <= {"total_ink"}:
        return "measurement_only_direct_panel_statistic"
    if not has_scene and not has_object_or_relation:
        return "no_object_or_relation_factorization"
    lowered = hyp.description.lower()
    raw_terms = {"pixel hash", "file order", "panel index"}
    if any(term in lowered for term in raw_terms):
        return "raw_artifact_description"
    return ""

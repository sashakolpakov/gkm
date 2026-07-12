"""Verification for mechanically compiled semantic cones.

Risk here is empirical and reported un-collapsed:

- ``support_errors``   — misclassifications of the fitted 1-D threshold rule
  on the 12 support panels;
- ``loo_errors``       — leave-one-out errors (threshold refit on 11 panels,
  held-out panel predicted);
- ``naturality_errors``— violations of the cone's own declared preservation
  morphisms, executed as actual panel transforms (cone invariance);
- ``cofibration_errors``— per-panel failures of proposer-declared gluings.

All checks are general: the harness executes declared structure; it knows no
problem-specific concepts.  Morphisms without an exact pixel action (e.g.
uniform_scale on 1-px strokes) are reported in ``unchecked_morphisms``
instead of being silently passed or faked.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from cofibrations import verify_cofibration
from dataset import Problem
from semantic_compiler import CompiledCone, CompileError, MissingLegError, compile_hypothesis
from semantic_ir import SemanticHypothesis
from semantic_legs import LegRegistry


@dataclass(frozen=True)
class ThresholdRule:
    node: str
    order: str
    threshold: float

    def predict(self, score: float) -> bool:
        return score <= self.threshold if self.order == "low_positive" else score >= self.threshold

    def describe(self) -> str:
        op = "<=" if self.order == "low_positive" else ">="
        return f"{self.node}{op}{self.threshold:.5g}"


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
    naturality_errors: int = 0
    cofibration_errors: int = 0
    unchecked_morphisms: tuple[str, ...] = ()
    compile_error: str = ""
    semantic_issue: str = ""
    missing_leg: dict | None = None
    scores: tuple[float, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return asdict(self)


def _fit_threshold(scores: np.ndarray, labels: np.ndarray, order: str) -> ThresholdRule:
    uniq = np.unique(scores)
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
        pred = scores <= t if order == "low_positive" else scores >= t
        errors = int(np.sum(pred != labels))
        margin = float(np.min(np.abs(scores - t))) if len(scores) else 0.0
        key = (errors, -margin, abs(float(t)))
        if best is None or key < best[0]:
            best = (key, ThresholdRule("score", order, float(t)))
    return best[1]


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


def _rotate_panel(panel: np.ndarray) -> np.ndarray | None:
    return np.rot90(np.asarray(panel)).copy()


def _reflect_panel(panel: np.ndarray) -> np.ndarray | None:
    return np.fliplr(np.asarray(panel)).copy()


_MORPHISM_ACTIONS = {
    "translate": _translate_panel,
    "translation": _translate_panel,
    "rotate": _rotate_panel,
    "rotation": _rotate_panel,
    "reflect": _reflect_panel,
    "reflection": _reflect_panel,
    "mirror": _reflect_panel,
}


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
                         max_loo_errors: int = 0) -> ConeVerification:
    panels = [p for p, _ in problem.panels()]
    labels = np.array([lab for _, lab in problem.panels()], dtype=bool)
    scores = np.zeros(len(panels), dtype=float)
    predicate_errors = 0
    cofibration_errors = 0
    traces = []
    for i, panel in enumerate(panels):
        score, trace = cone.score(panel, registry)
        scores[i] = score
        traces.append(trace)
        predicate_errors += int(bool(trace.errors))

    for spec in cone.hypothesis.cofibrations:
        if not (spec.source_node and spec.target_node):
            continue
        projection_fn = _resolve_projection_fn(spec, registry)
        for trace in traces:
            if trace.errors:
                continue
            source = trace.node_values.get(spec.source_node)
            target = trace.node_values.get(spec.target_node)
            if source is None or target is None:
                continue
            check = verify_cofibration(source, target, spec,
                                       projection_fn=projection_fn)
            if not check.ok:
                cofibration_errors += 1

    full_rule = _fit_threshold(scores, labels, cone.hypothesis.order)
    full_rule = ThresholdRule(cone.hypothesis.score_node, cone.hypothesis.order,
                              full_rule.threshold)
    support_pred = np.array([full_rule.predict(float(s)) for s in scores])
    support_errors = int(np.sum(support_pred != labels))

    naturality_errors = 0
    unchecked: list[str] = []
    for morph in cone.hypothesis.preservation_morphisms:
        if morph.expected_effect != "preserve":
            continue
        action = _MORPHISM_ACTIONS.get(morph.name.strip().lower())
        if action is None:
            unchecked.append(morph.name)
            continue
        for i, panel in enumerate(panels):
            transformed = action(panel)
            if transformed is None:
                continue
            new_score, trace = cone.score(transformed, registry)
            if trace.errors:
                naturality_errors += 1
                continue
            if full_rule.predict(float(new_score)) != bool(support_pred[i]):
                naturality_errors += 1

    correct = 0
    total = 0
    thresholds = []
    for held_idx in range(len(labels)):
        mask = np.array([k != held_idx for k in range(len(labels))])
        fold = _fit_threshold(scores[mask], labels[mask], cone.hypothesis.order)
        thresholds.append(fold.threshold)
        correct += int(fold.predict(float(scores[held_idx])) == labels[held_idx])
        total += 1
    loo_errors = total - correct
    semantic_issue = semantic_quality_issue(cone)
    accepted = (
        predicate_errors == 0
        and support_errors <= max_support_errors
        and loo_errors <= max_loo_errors
        and naturality_errors == 0
        and cofibration_errors == 0
        and not semantic_issue
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
        complexity=cone.complexity,
        naturality_errors=naturality_errors,
        cofibration_errors=cofibration_errors,
        unchecked_morphisms=tuple(unchecked),
        semantic_issue=semantic_issue,
        scores=tuple(float(s) for s in scores),
    )


def _failed_verification(hypothesis: SemanticHypothesis, rule: str,
                         compile_error: str, semantic_issue: str = "",
                         missing_leg: dict | None = None) -> ConeVerification:
    return ConeVerification(
        hypothesis_id=hypothesis.hypothesis_id,
        accepted=False,
        support_accuracy=0.0,
        loo_accuracy=0.0,
        support_errors=12,
        loo_errors=12,
        n_examples=12,
        rule=rule,
        threshold=0.0,
        fold_threshold_min=0.0,
        fold_threshold_max=0.0,
        predicate_errors=0,
        complexity=0,
        compile_error=compile_error,
        semantic_issue=semantic_issue,
        missing_leg=missing_leg,
    )


def verify_hypothesis(hypothesis: SemanticHypothesis, registry: LegRegistry,
                      problem: Problem, max_support_errors: int = 0,
                      max_loo_errors: int = 0) -> ConeVerification:
    try:
        cone = compile_hypothesis(hypothesis, registry)
    except MissingLegError as exc:
        return _failed_verification(hypothesis, "MISSING_LEG", str(exc),
                                    "MISSING_LEG", exc.missing.to_dict())
    except CompileError as exc:
        return _failed_verification(hypothesis, "COMPILE_ERROR", str(exc))
    return verify_compiled_cone(
        cone, registry, problem,
        max_support_errors=max_support_errors,
        max_loo_errors=max_loo_errors,
    )


def semantic_quality_issue(cone: CompiledCone) -> str:
    """Classify cones that separate panels but are not human-like semantics.

    This is intentionally conservative. It does not try to judge the English
    solution; it only rejects the worst known failure: a direct panel-level
    measurement pretending to be a semantic cone.
    """
    hyp = cone.hypothesis
    edge_calls = [edge.call for edge in hyp.diagram.edges]
    leg_names = {call.leg_name for call in edge_calls}
    has_scene = any(call.leg_name == "parse_scene" for call in edge_calls)
    has_object_or_relation = any(
        cone.node_types.get(edge.target) in {"Scene", "Object", "Relation"}
        for edge in hyp.diagram.edges
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

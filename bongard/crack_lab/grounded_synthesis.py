"""Support-only synthesis for closed, grounded Bongard predicates.

The proposer chooses measurement *intents*.  This module alone chooses numeric
bounds and Boolean structure.  Every candidate atom must be robustly true on
all positive support intervals, determinate on every support panel, and use an
affirmative comparison.  Synthesis searches positive conjunctions only: it
cannot rescue a bad proposal by reversing its polarity.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Sequence

import grounded_predicate_ir as G


GROUNDED_SYNTHESIS_SCHEMA = "bongard.grounded-synthesis/v1"
INTENT_SHAPES = frozenset({"exact", "low", "high", "band"})


class NoGroundedSeparator(ValueError):
    """The proposed registered measurements have no exact positive formula."""


@dataclass(frozen=True)
class MeasurementIntent:
    intent_id: str
    observable_id: str
    shape: str

    def __post_init__(self) -> None:
        if not isinstance(self.intent_id, str) or not self.intent_id:
            raise ValueError("intent_id must be nonempty text")
        if not isinstance(self.observable_id, str) or not self.observable_id:
            raise ValueError("observable_id must be nonempty text")
        if self.shape not in INTENT_SHAPES:
            raise ValueError("unknown measurement-intent shape")

    def to_dict(self) -> dict[str, str]:
        return {
            "intent_id": self.intent_id,
            "observable_id": self.observable_id,
            "shape": self.shape,
        }


@dataclass(frozen=True)
class SupportCase:
    case_id: str
    context: Any
    label: bool

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("case_id must be nonempty text")
        if not isinstance(self.label, bool):
            raise ValueError("case label must be boolean")


@dataclass(frozen=True)
class CaseEvaluation:
    case_id: str
    label: bool
    prediction: bool | None
    disposition: str
    trace: G.EvaluationTrace

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "label": self.label,
            "prediction": self.prediction,
            "disposition": self.disposition,
            "trace": self.trace.to_dict(),
        }


@dataclass(frozen=True)
class DatasetEvaluation:
    cases: tuple[CaseEvaluation, ...]
    predictions: tuple[bool | None, ...]
    error_count: int
    indeterminate_count: int
    exact: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": [case.to_dict() for case in self.cases],
            "predictions": list(self.predictions),
            "error_count": self.error_count,
            "indeterminate_count": self.indeterminate_count,
            "exact": self.exact,
        }


@dataclass(frozen=True)
class CandidateAtom:
    intent_id: str
    predicate: G.Compare
    support_predictions: tuple[bool, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "predicate": self.predicate.to_dict(),
            "support_predictions": list(self.support_predictions),
        }


@dataclass(frozen=True)
class FrozenSynthesisResult:
    intents: tuple[MeasurementIntent, ...]
    predicate: G.PredicateNode
    compiled: G.CompiledPredicate
    selected_intent_ids: tuple[str, ...]
    candidate_atoms: tuple[CandidateAtom, ...]
    support_evaluation: DatasetEvaluation
    registry_digest: str
    synthesis_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GROUNDED_SYNTHESIS_SCHEMA,
            "intents": [intent.to_dict() for intent in self.intents],
            "predicate": self.predicate.to_dict(),
            "compiled": self.compiled.canonical_dict(),
            "predicate_digest": self.compiled.digest,
            "selected_intent_ids": list(self.selected_intent_ids),
            "candidate_atoms": [atom.to_dict() for atom in self.candidate_atoms],
            "support_evaluation": self.support_evaluation.to_dict(),
            "registry_digest": self.registry_digest,
            "synthesis_digest": self.synthesis_digest,
        }


@dataclass(frozen=True)
class SensitivityPoint:
    relative_delta: float
    predicate: G.PredicateNode
    evaluation: DatasetEvaluation

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_delta": self.relative_delta,
            "predicate": self.predicate.to_dict(),
            "evaluation": self.evaluation.to_dict(),
        }


def _case_disposition(trace: G.EvaluationTrace) -> tuple[str, bool | None]:
    # A Boolean result cannot hide a broken/unknown leaf.  This is stricter
    # than the Kleene connective itself and is the admission policy used by
    # grounded synthesis and hidden-query evaluation.
    observations = tuple(value for _key, value in trace.observations)
    if any(isinstance(value, G.Error) for value in observations) \
            or isinstance(trace.result, G.Error):
        return "error", None
    if any(isinstance(value, G.Indeterminate) for value in observations) \
            or isinstance(trace.result, G.Indeterminate):
        return "indeterminate", None
    if isinstance(trace.result, G.Present) \
            and trace.result.unit is G.Unit.BOOLEAN \
            and isinstance(trace.result.value, bool):
        return "present", trace.result.value
    return "error", None


def evaluate_cases(
        compiled: G.CompiledPredicate,
        cases: Sequence[SupportCase],
        ) -> DatasetEvaluation:
    frozen_cases = tuple(cases)
    if not frozen_cases:
        raise ValueError("dataset evaluation requires at least one case")
    if len({case.case_id for case in frozen_cases}) != len(frozen_cases):
        raise ValueError("case IDs must be unique")
    rows: list[CaseEvaluation] = []
    predictions: list[bool | None] = []
    error_count = indeterminate_count = 0
    for case in frozen_cases:
        trace = compiled.evaluate_with_trace(case.context)
        disposition, prediction = _case_disposition(trace)
        if disposition == "error":
            error_count += 1
        elif disposition == "indeterminate":
            indeterminate_count += 1
        predictions.append(prediction)
        rows.append(CaseEvaluation(
            case.case_id, case.label, prediction, disposition, trace))
    exact = error_count == 0 and indeterminate_count == 0 and all(
        prediction is case.label
        for prediction, case in zip(predictions, frozen_cases))
    return DatasetEvaluation(
        tuple(rows), tuple(predictions), error_count,
        indeterminate_count, exact)


def _real(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("numeric intent requires real-valued observations")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("numeric observation must be finite")
    return result


def _positive_observations(
        intent: MeasurementIntent,
        cases: tuple[SupportCase, ...],
        registry: G.ObservableRegistry,
        ) -> tuple[G.Present, ...]:
    values: list[G.Present] = []
    for case in cases:
        if not case.label:
            continue
        observation = registry.evaluate(intent.observable_id, case.context)
        if not isinstance(observation, G.Present):
            return ()
        values.append(observation)
    return tuple(values)


def _present_negative_observations(
        intent: MeasurementIntent,
        cases: tuple[SupportCase, ...],
        registry: G.ObservableRegistry,
        ) -> tuple[G.Present, ...]:
    # Certified absence is already false for every positive comparison and
    # therefore needs no numeric surrogate.  Unknown/error values remain in
    # the later full atom evaluation, which rejects the atom.
    values: list[G.Present] = []
    for case in cases:
        if case.label:
            continue
        observation = registry.evaluate(intent.observable_id, case.context)
        if isinstance(observation, G.Present):
            values.append(observation)
    return tuple(values)


def _strict_above(value: int | float, integer: bool) -> int | float:
    if integer:
        return int(value) + 1
    result = math.nextafter(float(value), math.inf)
    if not math.isfinite(result):
        raise ValueError("cannot form finite upper threshold")
    return result


def _strict_below(value: int | float, integer: bool) -> int | float:
    if integer:
        return int(value) - 1
    result = math.nextafter(float(value), -math.inf)
    if not math.isfinite(result):
        raise ValueError("cannot form finite lower threshold")
    return result


def _atom_for_intent(
        intent: MeasurementIntent,
        cases: tuple[SupportCase, ...],
        registry: G.ObservableRegistry,
        ) -> G.Compare | None:
    if intent.observable_id not in registry:
        raise ValueError(
            f"intent {intent.intent_id!r} names an unknown observable")
    contract = registry.get(intent.observable_id)
    positives = _positive_observations(intent, cases, registry)
    negatives = _present_negative_observations(intent, cases, registry)
    if not positives:
        return None
    if any(value.unit is not contract.unit for value in positives):
        return None

    if intent.shape == "exact":
        first = positives[0]
        if any(value.value != first.value
               or value.lower != value.upper
               or value.value != value.lower for value in positives):
            return None
        return G.Compare(
            intent.observable_id, G.ComparisonOperator.EQ,
            G.Literal(first.value, contract.unit))

    if contract.value_type not in {G.ValueType.INTEGER, G.ValueType.REAL}:
        return None
    integer = contract.value_type is G.ValueType.INTEGER
    lowers = [_real(value.lower) for value in positives]
    uppers = [_real(value.upper) for value in positives]
    if integer:
        if any(not isinstance(value.lower, Integral)
               or not isinstance(value.upper, Integral) for value in positives):
            return None
    if intent.shape == "low":
        positive_edge = max(uppers)
        negative_lowers = [_real(value.lower) for value in negatives]
        if negative_lowers and positive_edge < min(negative_lowers):
            negative_edge = min(negative_lowers)
            threshold = positive_edge + 1 if integer else \
                (positive_edge + negative_edge) / 2.0
        else:
            threshold = _strict_above(positive_edge, integer)
        return G.Compare(
            intent.observable_id, G.ComparisonOperator.LT,
            G.Literal(threshold, contract.unit))
    if intent.shape == "high":
        positive_edge = min(lowers)
        negative_uppers = [_real(value.upper) for value in negatives]
        if negative_uppers and max(negative_uppers) < positive_edge:
            negative_edge = max(negative_uppers)
            threshold = int(positive_edge) - 1 if integer else \
                (negative_edge + positive_edge) / 2.0
        else:
            threshold = _strict_below(positive_edge, integer)
        return G.Compare(
            intent.observable_id, G.ComparisonOperator.GT,
            G.Literal(threshold, contract.unit))
    assert intent.shape == "band"
    positive_lower = min(lowers)
    positive_upper = max(uppers)
    below = [_real(value.upper) for value in negatives
             if _real(value.upper) < positive_lower]
    above = [_real(value.lower) for value in negatives
             if _real(value.lower) > positive_upper]
    lower: int | float = ((max(below) + positive_lower) / 2.0
                          if below else positive_lower)
    upper: int | float = ((positive_upper + min(above)) / 2.0
                          if above else positive_upper)
    if integer:
        lower, upper = int(math.ceil(lower)), int(math.floor(upper))
    return G.Compare(
        intent.observable_id, G.ComparisonOperator.BETWEEN,
        G.Literal(lower, contract.unit),
        G.Literal(upper, contract.unit))


def _candidate_atoms(
        intents: tuple[MeasurementIntent, ...],
        cases: tuple[SupportCase, ...],
        registry: G.ObservableRegistry,
        ) -> tuple[CandidateAtom, ...]:
    by_extension: dict[tuple[bool, ...], CandidateAtom] = {}
    labels = tuple(case.label for case in cases)
    for intent in intents:
        atom = _atom_for_intent(intent, cases, registry)
        if atom is None:
            continue
        compiled = G.compile_predicate(atom, registry)
        evaluation = evaluate_cases(compiled, cases)
        if evaluation.error_count or evaluation.indeterminate_count \
                or any(prediction is None for prediction in evaluation.predictions):
            continue
        predictions = tuple(bool(value) for value in evaluation.predictions)
        if any(not prediction for prediction, label in zip(predictions, labels)
               if label):
            continue
        if all(prediction for prediction, label in zip(predictions, labels)
               if not label):
            continue
        candidate = CandidateAtom(intent.intent_id, atom, predictions)
        previous = by_extension.get(predictions)
        if previous is None or G.canonical_json(candidate.predicate.to_dict()) \
                < G.canonical_json(previous.predicate.to_dict()):
            by_extension[predictions] = candidate
    return tuple(sorted(
        by_extension.values(),
        key=lambda item: G.canonical_json(item.predicate.to_dict())))


def _formula(atoms: tuple[CandidateAtom, ...]) -> G.PredicateNode:
    if len(atoms) == 1:
        return atoms[0].predicate
    return G.All(tuple(atom.predicate for atom in atoms))


def synthesize_grounded_predicate(
        intents: Sequence[MeasurementIntent],
        support_cases: Sequence[SupportCase],
        registry: G.ObservableRegistry,
        ) -> FrozenSynthesisResult:
    """Fit bounds and freeze the smallest exact positive conjunction.

    Numeric bounds are the tightest interval-safe envelopes containing every
    positive support observation.  Negative labels select among already
    grounded atoms and conjunctions; they never choose polarity or redefine an
    observable.  Query data is not an argument to this function.
    """
    frozen_intents = tuple(intents)
    frozen_cases = tuple(support_cases)
    if not frozen_intents:
        raise NoGroundedSeparator("no measurement intents were proposed")
    if len({intent.intent_id for intent in frozen_intents}) != len(frozen_intents):
        raise ValueError("intent IDs must be unique")
    if not frozen_cases or not any(case.label for case in frozen_cases) \
            or not any(not case.label for case in frozen_cases):
        raise ValueError("support must contain positive and negative cases")
    if len({case.case_id for case in frozen_cases}) != len(frozen_cases):
        raise ValueError("support case IDs must be unique")

    atoms = _candidate_atoms(frozen_intents, frozen_cases, registry)
    if not atoms:
        raise NoGroundedSeparator(
            "no proposed intent produced a determinate positive atom")
    labels = tuple(case.label for case in frozen_cases)
    selected: tuple[CandidateAtom, ...] | None = None
    best_errors = len(frozen_cases) + 1
    for size in range(1, len(atoms) + 1):
        exact: list[tuple[CandidateAtom, ...]] = []
        for subset in itertools.combinations(atoms, size):
            predictions = tuple(all(
                atom.support_predictions[index] for atom in subset)
                for index in range(len(frozen_cases)))
            errors = sum(prediction is not label
                         for prediction, label in zip(predictions, labels))
            best_errors = min(best_errors, errors)
            if errors == 0:
                exact.append(subset)
        if exact:
            selected = min(
                exact,
                key=lambda subset: G.canonical_json(_formula(subset).to_dict()))
            break
    if selected is None:
        raise NoGroundedSeparator(
            f"no exact positive conjunction; best support error count={best_errors}")

    predicate = _formula(selected)
    compiled = G.compile_predicate(predicate, registry)
    support_evaluation = evaluate_cases(compiled, frozen_cases)
    if not support_evaluation.exact:
        raise RuntimeError("selected formula did not reproduce exact support")
    body = {
        "schema": GROUNDED_SYNTHESIS_SCHEMA,
        "intents": [intent.to_dict() for intent in frozen_intents],
        "predicate": predicate.to_dict(),
        "predicate_digest": compiled.digest,
        "selected_intent_ids": [atom.intent_id for atom in selected],
        "registry_digest": registry.version_digest(),
        "support": support_evaluation.to_dict(),
    }
    return FrozenSynthesisResult(
        intents=frozen_intents,
        predicate=predicate,
        compiled=compiled,
        selected_intent_ids=tuple(atom.intent_id for atom in selected),
        candidate_atoms=atoms,
        support_evaluation=support_evaluation,
        registry_digest=registry.version_digest(),
        synthesis_digest=G.canonical_digest(body),
    )


def evaluate_hidden_queries(
        frozen: FrozenSynthesisResult,
        query_cases: Sequence[SupportCase],
        ) -> DatasetEvaluation:
    """Evaluate a frozen predicate; query labels cannot alter the formula."""
    if not isinstance(frozen, FrozenSynthesisResult):
        raise TypeError("hidden evaluation requires a frozen synthesis result")
    return evaluate_cases(frozen.compiled, query_cases)


def _shift_literal(value: int | float, delta: float) -> int | float:
    amount = max(abs(float(value)), 1.0) * abs(delta)
    shifted = float(value) + math.copysign(amount, delta)
    return int(round(shifted)) if isinstance(value, int) else shifted


def _perturb(node: G.PredicateNode, delta: float) -> G.PredicateNode:
    if isinstance(node, G.Compare):
        value = node.threshold.value
        if isinstance(value, bool) or not isinstance(value, Real) \
                or node.operator is G.ComparisonOperator.EQ:
            return node
        if node.operator is G.ComparisonOperator.BETWEEN:
            assert node.upper is not None
            width_sign = -1.0 if delta < 0.0 else 1.0
            lower = _shift_literal(value, -width_sign * abs(delta))
            upper = _shift_literal(node.upper.value, width_sign * abs(delta))
            if lower > upper:
                return node
            return G.Compare(
                node.observable_id, node.operator,
                G.Literal(lower, node.threshold.unit),
                G.Literal(upper, node.upper.unit))
        return G.Compare(
            node.observable_id, node.operator,
            G.Literal(_shift_literal(value, delta), node.threshold.unit))
    if isinstance(node, G.All):
        return G.All(tuple(_perturb(child, delta) for child in node.children))
    if isinstance(node, G.Any):
        return G.Any(tuple(_perturb(child, delta) for child in node.children))
    if isinstance(node, G.Not):
        return G.Not(_perturb(node.child, delta))
    raise TypeError("unknown predicate node")


def threshold_sensitivity(
        frozen: FrozenSynthesisResult,
        cases: Sequence[SupportCase],
        registry: G.ObservableRegistry,
        *, relative_delta: float = 0.05,
        ) -> tuple[SensitivityPoint, ...]:
    """Report fixed-formula threshold perturbations; this is not CV/LOO."""
    if not math.isfinite(relative_delta) or not 0.0 < relative_delta < 1.0:
        raise ValueError("relative_delta must be finite and in (0, 1)")
    points: list[SensitivityPoint] = []
    for delta in (-relative_delta, 0.0, relative_delta):
        predicate = frozen.predicate if delta == 0.0 \
            else _perturb(frozen.predicate, delta)
        evaluation = evaluate_cases(
            G.compile_predicate(predicate, registry), cases)
        points.append(SensitivityPoint(delta, predicate, evaluation))
    return tuple(points)


__all__ = [
    "CandidateAtom",
    "CaseEvaluation",
    "DatasetEvaluation",
    "FrozenSynthesisResult",
    "GROUNDED_SYNTHESIS_SCHEMA",
    "MeasurementIntent",
    "NoGroundedSeparator",
    "SensitivityPoint",
    "SupportCase",
    "evaluate_cases",
    "evaluate_hidden_queries",
    "synthesize_grounded_predicate",
    "threshold_sensitivity",
]

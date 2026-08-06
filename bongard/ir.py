"""Closed, positive-only predicate IR with interval-safe evaluation.

The primary track has exactly three syntax nodes: :class:`Atom`,
:class:`AllOf`, and :class:`AnyOf`.  There is no negation node, polarity flag,
user expression, dynamic import, or dynamic leg lookup.  An atom calls an
exact version-and-digest-pinned leg from the verifier registry.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, TypeAlias

from bongard.evidence import (
    Disposition,
    Evidence,
    Provenance,
    SoftSemanticObservation,
    Uncertainty,
)
from bongard.legs.contracts import (
    AffirmativeRelation,
    ContractViolation,
    LegReference,
    LegRegistry,
    Literal,
    RegistrySnapshot,
    TypedValue,
    Unit,
    ValueType,
)


class IRValidationError(ValueError):
    """A formula lies outside the closed or typed primary-track grammar."""


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: Unit

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not math.isfinite(self.value):
            raise ValueError("quantity value must be a finite scalar")

    def to_data(self) -> dict[str, object]:
        return {"value": self.value, "unit": self.unit.value}


@dataclass(frozen=True)
class Interval:
    """A closed scalar interval; comparisons never inspect its midpoint."""

    lower: float
    upper: float
    unit: Unit

    def __post_init__(self) -> None:
        if isinstance(self.lower, bool) or isinstance(self.upper, bool):
            raise TypeError("interval bounds cannot be boolean")
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("interval bounds must be finite")
        if self.lower > self.upper:
            raise ValueError("interval lower bound exceeds upper bound")

    @classmethod
    def point(cls, value: float, unit: Unit) -> "Interval":
        return cls(float(value), float(value), unit)

    def to_data(self) -> dict[str, object]:
        return {"lower": self.lower, "upper": self.upper, "unit": self.unit.value}


class Relation(str, Enum):
    """Positive assertions supported by an atom.

    ``NOT_EQUAL`` and analogous complement operations are intentionally not in
    the vocabulary.  A candidate cannot win by flipping the support rule.
    """

    PRESENT = "present"
    AT_LEAST = "at_least"
    AT_MOST = "at_most"
    BETWEEN = "between"


def _is_literal(value: object) -> bool:
    return value is None or (
        isinstance(value, (str, int, float, bool))
        and not (isinstance(value, float) and not math.isfinite(value))
    )


@dataclass(frozen=True)
class StaticLegCall:
    """One syntactically static call to a pinned registered leg."""

    leg: LegReference
    arguments: tuple[str, ...]
    parameters: tuple[tuple[str, Literal], ...] = ()

    def __post_init__(self) -> None:
        if not self.arguments or any(not name.strip() for name in self.arguments):
            raise ValueError("static leg call requires named boundary arguments")
        names = [name for name, _ in self.parameters]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("static leg parameters must be unique and sorted")
        if any(not _is_literal(value) for _, value in self.parameters):
            raise ValueError("static leg parameters must be finite JSON literals")

    def to_data(self) -> dict[str, object]:
        return {
            "leg": self.leg.to_data(),
            "arguments": list(self.arguments),
            "parameters": [list(item) for item in self.parameters],
        }


@dataclass(frozen=True)
class Atom:
    call: StaticLegCall
    relation: Relation
    claim: str
    lower: Quantity | None = None
    upper: Quantity | None = None

    def __post_init__(self) -> None:
        if not self.claim.strip():
            raise ValueError("atom claim must be non-empty")
        if self.relation is Relation.PRESENT:
            if self.lower is not None or self.upper is not None:
                raise ValueError("present atom cannot have thresholds")
        elif self.relation in (Relation.AT_LEAST, Relation.AT_MOST):
            if self.lower is None or self.upper is not None:
                raise ValueError(f"{self.relation.value} requires one threshold")
        elif self.relation is Relation.BETWEEN:
            if self.lower is None or self.upper is None:
                raise ValueError("between requires lower and upper thresholds")
            if self.lower.unit != self.upper.unit:
                raise ValueError("between threshold units differ")
            if self.lower.value > self.upper.value:
                raise ValueError("between lower threshold exceeds upper")

    def to_data(self) -> dict[str, object]:
        data: dict[str, object] = {
            "type": "atom",
            "call": self.call.to_data(),
            "relation": self.relation.value,
            "claim": self.claim,
        }
        if self.lower is not None:
            data["lower"] = self.lower.to_data()
        if self.upper is not None:
            data["upper"] = self.upper.to_data()
        return data


@dataclass(frozen=True)
class AllOf:
    terms: tuple["Formula", ...]
    justification: str

    def __post_init__(self) -> None:
        if len(self.terms) < 2:
            raise ValueError("and requires at least two terms")
        if not self.justification.strip():
            raise ValueError("and requires an explicit semantic justification")

    def to_data(self) -> dict[str, object]:
        return {
            "type": "and",
            "terms": [term.to_data() for term in self.terms],
            "justification": self.justification,
        }


@dataclass(frozen=True)
class AnyOf:
    terms: tuple["Formula", ...]
    justification: str

    def __post_init__(self) -> None:
        if len(self.terms) < 2:
            raise ValueError("or requires at least two terms")
        if not self.justification.strip():
            raise ValueError("or requires an explicit semantic justification")

    def to_data(self) -> dict[str, object]:
        return {
            "type": "or",
            "terms": [term.to_data() for term in self.terms],
            "justification": self.justification,
        }


Formula: TypeAlias = Atom | AllOf | AnyOf


def formula_digest(formula: Formula) -> str:
    payload = json.dumps(
        formula.to_data(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atoms(formula: Formula) -> tuple[Atom, ...]:
    if isinstance(formula, Atom):
        return (formula,)
    return tuple(atom for term in formula.terms for atom in atoms(term))


def validate_formula(
    formula: Formula,
    registry: LegRegistry | RegistrySnapshot,
    boundary_types: Mapping[str, ValueType],
) -> None:
    """Statically typecheck all calls against an exact registry snapshot."""

    for atom in atoms(formula):
        try:
            contract = registry.resolve(atom.call.leg)
        except ContractViolation as exc:
            raise IRValidationError(str(exc)) from exc
        if len(atom.call.arguments) != len(contract.domain):
            raise IRValidationError(
                f"{contract.name} expects {len(contract.domain)} arguments, "
                f"got {len(atom.call.arguments)}"
            )
        for index, (name, expected) in enumerate(
            zip(atom.call.arguments, contract.domain, strict=True)
        ):
            actual = boundary_types.get(name)
            if actual is None:
                raise IRValidationError(f"unknown boundary input {name!r}")
            if actual != expected:
                raise IRValidationError(
                    f"{contract.name} argument {index} ({name}) has {actual}, "
                    f"expected {expected}"
                )
        parameter_names = [name for name, _ in atom.call.parameters]
        undeclared = set(parameter_names) - contract.parameter_names
        if undeclared:
            raise IRValidationError(
                "undeclared parameters for "
                f"{contract.name}: {', '.join(sorted(undeclared))}"
            )
        if atom.relation is Relation.PRESENT:
            if contract.codomain.unit is not Unit.NONE:
                raise IRValidationError(
                    f"scalar leg {contract.name} requires an interval comparison"
                )
        else:
            if contract.codomain.unit is Unit.NONE:
                raise IRValidationError(
                    f"non-scalar leg {contract.name} only supports present"
                )
            assert atom.lower is not None
            if atom.lower.unit != contract.codomain.unit:
                raise IRValidationError(
                    f"{contract.name} yields {contract.codomain.unit.value}, "
                    f"threshold uses {atom.lower.unit.value}"
                )
            if atom.upper is not None and atom.upper.unit != contract.codomain.unit:
                raise IRValidationError(
                    f"{contract.name} upper threshold has the wrong unit"
                )
        try:
            affirmative = AffirmativeRelation(atom.relation.value)
        except ValueError as exc:  # pragma: no cover - enums evolve together.
            raise IRValidationError(
                f"relation {atom.relation.value} has no leg-contract meaning"
            ) from exc
        if affirmative not in contract.affirmative_relations:
            allowed = ", ".join(
                sorted(item.value for item in contract.affirmative_relations)
            )
            raise IRValidationError(
                f"{atom.relation.value} is not an affirmative orientation for "
                f"{contract.name}; allowed: {allowed}"
            )


def _derived_provenance(
    operation: str, parents: tuple[Evidence[object], ...]
) -> Provenance:
    return Provenance.composed(
        producer="bongard.closed_ir",
        version="1",
        method=operation,
        parents=tuple(parent.provenance for parent in parents),
    )


def _propagate_nonpresent(
    evidence: Evidence[object], operation: str
) -> Evidence[bool]:
    provenance = _derived_provenance(operation, (evidence,))
    if evidence.disposition is Disposition.CERTIFIED_ABSENT:
        return Evidence.certified_absent(
            provenance,
            evidence.certificate or "registered leg certified absence",
            evidence.uncertainty,
        )
    if evidence.disposition is Disposition.INDETERMINATE:
        return Evidence.indeterminate(
            provenance,
            evidence.reason or "registered leg was indeterminate",
            evidence.uncertainty,
        )
    if evidence.disposition is Disposition.ERROR:
        return Evidence.error(
            provenance,
            evidence.error_type or "LegError",
            evidence.reason or "registered leg failed",
        )
    raise AssertionError("propagation requested for present evidence")


def _measurement_interval(evidence: Evidence[TypedValue]) -> Interval:
    typed = evidence.unwrap()
    value = typed.value
    unit = typed.type.unit
    if unit is Unit.NONE:
        raise IRValidationError("non-scalar typed value has no comparison interval")
    if isinstance(value, SoftSemanticObservation):
        if unit is not Unit.PROBABILITY:
            raise IRValidationError("soft semantic observation must use probability")
        return Interval(value.support.lower, value.support.upper, unit)
    if isinstance(value, Interval):
        if value.unit != unit:
            raise IRValidationError("runtime interval unit differs from leg codomain")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IRValidationError("scalar leg returned a non-scalar runtime value")
    if evidence.uncertainty is not None:
        return Interval(
            evidence.uncertainty.lower, evidence.uncertainty.upper, unit
        )
    return Interval.point(float(value), unit)


def _compare(atom: Atom, interval: Interval) -> bool | None:
    """Return True, False, or None when an interval straddles the boundary."""

    assert atom.lower is not None
    threshold = atom.lower.value
    if atom.relation is Relation.AT_LEAST:
        if interval.lower >= threshold:
            return True
        if interval.upper < threshold:
            return False
        return None
    if atom.relation is Relation.AT_MOST:
        if interval.upper <= threshold:
            return True
        if interval.lower > threshold:
            return False
        return None
    if atom.relation is Relation.BETWEEN:
        assert atom.upper is not None
        if interval.lower >= threshold and interval.upper <= atom.upper.value:
            return True
        if interval.upper < threshold or interval.lower > atom.upper.value:
            return False
        return None
    raise AssertionError(f"not a comparison relation: {atom.relation}")


def _evaluate_atom(
    atom: Atom,
    registry: LegRegistry,
    bindings: Mapping[str, TypedValue],
) -> Evidence[bool]:
    try:
        arguments = tuple(bindings[name] for name in atom.call.arguments)
    except KeyError as exc:
        raise IRValidationError(f"missing runtime boundary input {exc.args[0]!r}") from exc
    result = registry.invoke(atom.call.leg, arguments, atom.call.parameters)
    if not result.is_present:
        return _propagate_nonpresent(result, f"atom:{atom.relation.value}")
    provenance = _derived_provenance(f"atom:{atom.relation.value}", (result,))
    if atom.relation is Relation.PRESENT:
        return Evidence.present(True, provenance)
    try:
        interval = _measurement_interval(result)
        decision = _compare(atom, interval)
    except (IRValidationError, TypeError, ValueError) as exc:
        return Evidence.error(provenance, type(exc).__name__, str(exc))
    if decision is True:
        return Evidence.present(True, provenance)
    if decision is False:
        return Evidence.certified_absent(
            provenance,
            f"closed interval [{interval.lower}, {interval.upper}] "
            f"does not satisfy {atom.relation.value}",
        )
    return Evidence.indeterminate(
        provenance,
        "measurement interval straddles the atom threshold",
        Uncertainty(interval.lower, interval.upper, causes=("threshold_overlap",)),
    )


def _combine(
    formula: AllOf | AnyOf,
    results: tuple[Evidence[bool], ...],
) -> Evidence[bool]:
    operation = "and" if isinstance(formula, AllOf) else "or"
    provenance = _derived_provenance(operation, results)
    errors = [item for item in results if item.disposition is Disposition.ERROR]
    if errors:
        first = errors[0]
        return Evidence.error(
            provenance,
            first.error_type or "ChildError",
            first.reason or f"{operation} child failed",
        )
    if isinstance(formula, AllOf):
        absences = [
            item
            for item in results
            if item.disposition is Disposition.CERTIFIED_ABSENT
        ]
        if absences:
            return Evidence.certified_absent(
                provenance,
                "conjunct certified absent: "
                + (absences[0].certificate or "unspecified certificate"),
            )
        indeterminate = [
            item for item in results if item.disposition is Disposition.INDETERMINATE
        ]
        if indeterminate:
            return Evidence.indeterminate(
                provenance, "one or more conjuncts are indeterminate"
            )
        return Evidence.present(True, provenance)

    present = [item for item in results if item.disposition is Disposition.PRESENT]
    if present:
        return Evidence.present(True, provenance)
    indeterminate = [
        item for item in results if item.disposition is Disposition.INDETERMINATE
    ]
    if indeterminate:
        return Evidence.indeterminate(
            provenance, "no disjunct is present and at least one is indeterminate"
        )
    return Evidence.certified_absent(
        provenance, "every disjunct is certified absent"
    )


def evaluate_formula(
    formula: Formula,
    registry: LegRegistry,
    bindings: Mapping[str, TypedValue],
) -> Evidence[bool]:
    """Evaluate after a mandatory static typecheck of the full formula."""

    validate_formula(
        formula, registry, {name: value.type for name, value in bindings.items()}
    )
    if isinstance(formula, Atom):
        return _evaluate_atom(formula, registry, bindings)
    results = tuple(evaluate_formula(term, registry, bindings) for term in formula.terms)
    return _combine(formula, results)


def _expect_keys(data: Mapping[str, Any], required: set[str], optional: set[str]) -> None:
    keys = set(data)
    missing = required - keys
    extra = keys - required - optional
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing " + ", ".join(sorted(missing)))
        if extra:
            parts.append("unknown " + ", ".join(sorted(extra)))
        raise IRValidationError("; ".join(parts))


def _quantity_from_data(data: Mapping[str, Any]) -> Quantity:
    _expect_keys(data, {"value", "unit"}, set())
    try:
        return Quantity(float(data["value"]), Unit(str(data["unit"])))
    except (TypeError, ValueError) as exc:
        raise IRValidationError(f"invalid quantity: {exc}") from exc


def _call_from_data(data: Mapping[str, Any]) -> StaticLegCall:
    _expect_keys(data, {"leg", "arguments", "parameters"}, set())
    leg_data = data["leg"]
    if not isinstance(leg_data, Mapping):
        raise IRValidationError("call leg must be an object")
    _expect_keys(leg_data, {"name", "version", "contract_digest"}, set())
    try:
        reference = LegReference(
            name=str(leg_data["name"]),
            version=str(leg_data["version"]),
            contract_digest=str(leg_data["contract_digest"]),
        )
        arguments = tuple(str(item) for item in data["arguments"])
        raw_parameters = data["parameters"]
        parameters = tuple((str(item[0]), item[1]) for item in raw_parameters)
        return StaticLegCall(reference, arguments, parameters)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise IRValidationError(f"invalid static leg call: {exc}") from exc


def formula_from_data(data: Mapping[str, Any]) -> Formula:
    """Parse only the closed grammar; unknown nodes and fields fail closed."""

    node_type = data.get("type")
    if node_type == "atom":
        _expect_keys(
            data,
            {"type", "call", "relation", "claim"},
            {"lower", "upper"},
        )
        call_data = data["call"]
        if not isinstance(call_data, Mapping):
            raise IRValidationError("atom call must be an object")
        try:
            relation = Relation(str(data["relation"]))
        except ValueError as exc:
            raise IRValidationError(f"unsupported atom relation {data['relation']!r}") from exc
        lower_data = data.get("lower")
        upper_data = data.get("upper")
        return Atom(
            call=_call_from_data(call_data),
            relation=relation,
            claim=str(data["claim"]),
            lower=(
                _quantity_from_data(lower_data)
                if isinstance(lower_data, Mapping)
                else None
            ),
            upper=(
                _quantity_from_data(upper_data)
                if isinstance(upper_data, Mapping)
                else None
            ),
        )
    if node_type in {"and", "or"}:
        _expect_keys(data, {"type", "terms", "justification"}, set())
        raw_terms = data["terms"]
        if not isinstance(raw_terms, list) or any(
            not isinstance(term, Mapping) for term in raw_terms
        ):
            raise IRValidationError("composite terms must be a list of objects")
        terms = tuple(formula_from_data(term) for term in raw_terms)
        constructor = AllOf if node_type == "and" else AnyOf
        return constructor(terms, str(data["justification"]))
    raise IRValidationError(
        f"unsupported IR node {node_type!r}; primary track permits atom/and/or only"
    )

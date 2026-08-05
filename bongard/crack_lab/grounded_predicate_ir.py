"""A small, closed-world IR for grounded Bongard predicates.

The important boundary in this module is the observable registry.  A
predicate may combine registered measurements, but it cannot manufacture a
new visual measurement in its syntax.  Open-world/oracle measurements are
allowed only when their contract says so, and make the compiled predicate
``HYBRID`` rather than ``PURE``.

There are two different kinds of non-values.  ``SemanticAbsent`` is a
negative certificate: the referent required by an observable is known not to
exist (for example, a gap ratio at a panel with no point contact), so a
comparison over it is false.  ``Indeterminate`` means that the referent may
exist but the registered procedure cannot decide, and is Kleene-unknown.
Neither is a numeric sentinel.  ``Error`` is an invalid evaluation and is
never masked.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Any as PythonAny, Callable, Mapping, TypeAlias


PREDICATE_IR_SCHEMA = "bongard.grounded-predicate-ir/v0.2"
OBSERVABLE_CONTRACT_SCHEMA = "bongard.observable-contract/v0.2"

_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_.-]*(?:/[a-z0-9_.-]+)*\Z")
_MODE_RE = re.compile(r"[a-z][a-z0-9_.-]*\Z")
_VERSION_RE = re.compile(r"v[1-9][0-9]*\Z")


def canonical_json(value: PythonAny) -> str:
    """Return deterministic, finite JSON suitable for content addressing."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_digest(value: PythonAny) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _identifier(value: str, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a canonical identifier")
    return value


def _mode(value: str, field: str) -> str:
    if not isinstance(value, str) or _MODE_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a canonical mode")
    return value


def _provenance(value: tuple[str, ...]) -> tuple[str, ...]:
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError("provenance entries must be nonempty strings")
    return result


class ValueType(str, Enum):
    INTEGER = "integer"
    REAL = "real"
    BOOLEAN = "boolean"
    TEXT = "text"


class Unit(str, Enum):
    """Nominal units understood by v0.2; no implicit conversions exist."""

    COUNT = "count"
    RATIO = "ratio"
    DEGREES = "degrees"
    RADIANS = "radians"
    PIXELS = "pixels"
    UNITLESS = "unitless"
    BOOLEAN = "boolean"
    TEXT = "text"


class Reducer(str, Enum):
    IDENTITY = "identity"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    RATIO = "ratio"


class ObservableSource(str, Enum):
    DETERMINISTIC = "deterministic"
    ORACLE = "oracle"


class Taint(str, Enum):
    PURE = "PURE"
    HYBRID = "HYBRID"


class ComparisonOperator(str, Enum):
    EQ = "eq"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    BETWEEN = "between"


class Invariance(str, Enum):
    TRANSLATION = "translation"
    ROTATION = "rotation"
    REFLECTION = "reflection"
    UNIFORM_SCALE = "uniform-scale"
    OBJECT_PERMUTATION = "object-permutation"
    STROKE_WIDTH = "stroke-width"


@dataclass(frozen=True)
class Present:
    value: int | float | bool | str
    unit: Unit | str
    provenance: tuple[str, ...] = ()
    lower: int | float | None = None
    upper: int | float | None = None

    def __post_init__(self) -> None:
        try:
            unit = Unit(self.unit)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown unit {self.unit!r}") from exc
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "provenance", _provenance(self.provenance))
        numeric = isinstance(self.value, Real) and not isinstance(self.value, bool)
        if numeric:
            if not math.isfinite(float(self.value)):
                raise ValueError("present numeric value must be finite")
            lower = self.value if self.lower is None else self.lower
            upper = self.value if self.upper is None else self.upper
            for bound, name in ((lower, "lower"), (upper, "upper")):
                if isinstance(bound, bool) or not isinstance(bound, Real) \
                        or not math.isfinite(float(bound)):
                    raise ValueError(f"present {name} bound must be a finite number")
            if lower > self.value or self.value > upper:
                raise ValueError("present bounds must satisfy lower <= value <= upper")
            object.__setattr__(self, "lower", lower)
            object.__setattr__(self, "upper", upper)
        elif self.lower is not None or self.upper is not None:
            raise ValueError("interval bounds apply only to numeric present values")

    def to_dict(self) -> dict[str, PythonAny]:
        result = {
            "status": "present",
            "value": self.value,
            "unit": self.unit.value,
            "provenance": list(self.provenance),
        }
        if self.lower is not None:
            result["lower"] = self.lower
            result["upper"] = self.upper
        return result


@dataclass(frozen=True)
class SemanticAbsent:
    mode: str
    detail: str = ""
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _mode(self.mode, "semantic absence mode")
        if not isinstance(self.detail, str):
            raise ValueError("semantic absence detail must be text")
        object.__setattr__(self, "provenance", _provenance(self.provenance))

    def to_dict(self) -> dict[str, PythonAny]:
        return {
            "status": "semantic-absent",
            "mode": self.mode,
            "detail": self.detail,
            "provenance": list(self.provenance),
        }


@dataclass(frozen=True)
class Indeterminate:
    mode: str
    detail: str = ""
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _mode(self.mode, "indeterminate mode")
        if not isinstance(self.detail, str):
            raise ValueError("indeterminate detail must be text")
        object.__setattr__(self, "provenance", _provenance(self.provenance))

    def to_dict(self) -> dict[str, PythonAny]:
        return {
            "status": "indeterminate",
            "mode": self.mode,
            "detail": self.detail,
            "provenance": list(self.provenance),
        }


@dataclass(frozen=True)
class Error:
    code: str
    detail: str = ""
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _mode(self.code, "error code")
        if not isinstance(self.detail, str):
            raise ValueError("error detail must be text")
        object.__setattr__(self, "provenance", _provenance(self.provenance))

    def to_dict(self) -> dict[str, PythonAny]:
        return {
            "status": "error",
            "code": self.code,
            "detail": self.detail,
            "provenance": list(self.provenance),
        }


Observation: TypeAlias = Present | SemanticAbsent | Indeterminate | Error


def _normalise_modes(values: tuple[str, ...], field: str) -> tuple[str, ...]:
    result = tuple(sorted(set(values)))
    for value in result:
        _mode(value, field)
    if len(result) != len(tuple(values)):
        raise ValueError(f"{field} must not contain duplicates")
    return result


def _normalise_invariances(
    values: tuple[Invariance | str, ...],
) -> tuple[Invariance, ...]:
    try:
        result = tuple(sorted({Invariance(value) for value in values}, key=lambda x: x.value))
    except (TypeError, ValueError) as exc:
        raise ValueError("unknown invariance") from exc
    if len(result) != len(tuple(values)):
        raise ValueError("invariances must not contain duplicates")
    return result


def _value_matches_type(value: PythonAny, value_type: ValueType) -> bool:
    if value_type is ValueType.INTEGER:
        return isinstance(value, Integral) and not isinstance(value, bool)
    if value_type is ValueType.REAL:
        return isinstance(value, Real) and not isinstance(value, bool) \
            and math.isfinite(float(value))
    if value_type is ValueType.BOOLEAN:
        return isinstance(value, bool)
    return isinstance(value, str)


def _unit_matches_type(unit: Unit, value_type: ValueType) -> bool:
    if value_type is ValueType.BOOLEAN:
        return unit is Unit.BOOLEAN
    if value_type is ValueType.TEXT:
        return unit is Unit.TEXT
    return unit not in {Unit.BOOLEAN, Unit.TEXT}


@dataclass(frozen=True)
class ObservableContract:
    """Declarative contract plus the only evaluator admitted for one leaf."""

    observable_id: str
    value_type: ValueType | str
    unit: Unit | str
    referent: str
    reducer: Reducer | str
    evaluator: Callable[[PythonAny], Observation]
    semantic_absence_modes: tuple[str, ...] = ()
    indeterminate_modes: tuple[str, ...] = ()
    invariances: tuple[Invariance | str, ...] = ()
    version: str = "v1"
    source: ObservableSource | str = ObservableSource.DETERMINISTIC

    def __post_init__(self) -> None:
        _identifier(self.observable_id, "observable_id")
        _identifier(self.referent, "referent")
        if not isinstance(self.version, str) or _VERSION_RE.fullmatch(self.version) is None:
            raise ValueError("version must have form vN")
        if not callable(self.evaluator):
            raise ValueError("evaluator must be callable")
        try:
            value_type = ValueType(self.value_type)
            unit = Unit(self.unit)
            reducer = Reducer(self.reducer)
            source = ObservableSource(self.source)
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown observable contract enum value") from exc
        if not _unit_matches_type(unit, value_type):
            raise ValueError(f"unit {unit.value} is incompatible with {value_type.value}")
        absence = _normalise_modes(
            self.semantic_absence_modes, "semantic absence mode")
        indeterminate = _normalise_modes(
            self.indeterminate_modes, "indeterminate mode")
        overlap = set(absence) & set(indeterminate)
        if overlap:
            raise ValueError("absence and indeterminate modes must be disjoint")
        invariances = _normalise_invariances(self.invariances)
        object.__setattr__(self, "value_type", value_type)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "reducer", reducer)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "semantic_absence_modes", absence)
        object.__setattr__(self, "indeterminate_modes", indeterminate)
        object.__setattr__(self, "invariances", invariances)

    @property
    def stable_id(self) -> str:
        return self.observable_id

    @property
    def taint(self) -> Taint:
        return Taint.HYBRID if self.source is ObservableSource.ORACLE else Taint.PURE

    def contract_dict(self) -> dict[str, PythonAny]:
        """Serializable semantics.  The callable is identified by version."""
        return {
            "schema": OBSERVABLE_CONTRACT_SCHEMA,
            "observable_id": self.observable_id,
            "version": self.version,
            "value_type": self.value_type.value,
            "unit": self.unit.value,
            "referent": self.referent,
            "reducer": self.reducer.value,
            "semantic_absence_modes": list(self.semantic_absence_modes),
            "indeterminate_modes": list(self.indeterminate_modes),
            "invariances": [item.value for item in self.invariances],
            "source": self.source.value,
        }

    def version_digest(self) -> str:
        return canonical_digest(self.contract_dict())

    def evaluate(self, context: PythonAny) -> Observation:
        try:
            result = self.evaluator(context)
        except Exception as exc:  # the trusted boundary fails closed
            return Error(
                "evaluator-exception",
                f"{type(exc).__name__}: {exc}",
                (self.observable_id, self.version_digest()),
            )
        violation = self.result_contract_violation(result)
        if violation:
            return Error(
                "observable-contract-violation",
                violation,
                (self.observable_id, self.version_digest()),
            )
        return result

    def result_contract_violation(self, result: PythonAny) -> str | None:
        if isinstance(result, Present):
            if result.unit is not self.unit:
                return (
                    f"observable returned unit {result.unit.value}, "
                    f"expected {self.unit.value}"
                )
            if not _value_matches_type(result.value, self.value_type):
                return f"observable returned the wrong {self.value_type.value} value type"
            if self.value_type is ValueType.INTEGER and (
                not isinstance(result.lower, Integral)
                or isinstance(result.lower, bool)
                or not isinstance(result.upper, Integral)
                or isinstance(result.upper, bool)
            ):
                return "integer observable returned non-integer interval bounds"
            return None
        if isinstance(result, SemanticAbsent):
            if result.mode not in self.semantic_absence_modes:
                return f"undeclared semantic absence mode {result.mode!r}"
            return None
        if isinstance(result, Indeterminate):
            if result.mode not in self.indeterminate_modes:
                return f"undeclared indeterminate mode {result.mode!r}"
            return None
        if isinstance(result, Error):
            return None
        return "evaluator must return Present, SemanticAbsent, Indeterminate, or Error"


class ObservableRegistry:
    """Append-only-by-ID observable registry used by the compiler."""

    def __init__(self) -> None:
        self._contracts: dict[str, ObservableContract] = {}

    def register(self, contract: ObservableContract) -> ObservableContract:
        if contract.observable_id in self._contracts:
            raise ValueError(f"observable {contract.observable_id!r} is already registered")
        self._contracts[contract.observable_id] = contract
        return contract

    def get(self, observable_id: str) -> ObservableContract:
        try:
            return self._contracts[observable_id]
        except KeyError as exc:
            raise KeyError(f"unknown observable {observable_id!r}") from exc

    def evaluate(self, observable_id: str, context: PythonAny) -> Observation:
        return self.get(observable_id).evaluate(context)

    def contracts(self) -> tuple[ObservableContract, ...]:
        return tuple(self._contracts[key] for key in sorted(self._contracts))

    def version_digest(self) -> str:
        return canonical_digest({
            "schema": OBSERVABLE_CONTRACT_SCHEMA,
            "contracts": [
                {
                    "observable_id": contract.observable_id,
                    "version_digest": contract.version_digest(),
                }
                for contract in self.contracts()
            ],
        })

    def __contains__(self, observable_id: object) -> bool:
        return observable_id in self._contracts

    def __len__(self) -> int:
        return len(self._contracts)


@dataclass(frozen=True)
class Literal:
    value: int | float | bool | str
    unit: Unit | str

    def __post_init__(self) -> None:
        try:
            unit = Unit(self.unit)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown unit {self.unit!r}") from exc
        object.__setattr__(self, "unit", unit)
        if isinstance(self.value, Real) and not isinstance(self.value, bool):
            if not math.isfinite(float(self.value)):
                raise ValueError("literal numeric value must be finite")
        elif not isinstance(self.value, (bool, str)):
            raise ValueError("literal must be a finite number, boolean, or string")

    def to_dict(self) -> dict[str, PythonAny]:
        return {"value": self.value, "unit": self.unit.value}

    @staticmethod
    def from_dict(data: Mapping[str, PythonAny]) -> "Literal":
        if set(data) != {"value", "unit"}:
            raise ValueError("literal must contain exactly value and unit")
        return Literal(data["value"], data["unit"])


@dataclass(frozen=True)
class Compare:
    observable_id: str
    operator: ComparisonOperator | str
    threshold: Literal
    upper: Literal | None = None

    def __post_init__(self) -> None:
        _identifier(self.observable_id, "comparison observable_id")
        try:
            operator = ComparisonOperator(self.operator)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown comparison operator {self.operator!r}") from exc
        object.__setattr__(self, "operator", operator)
        if not isinstance(self.threshold, Literal):
            raise ValueError("comparison threshold must be a Literal")
        if operator is ComparisonOperator.BETWEEN:
            if not isinstance(self.upper, Literal):
                raise ValueError("between comparison requires an upper Literal")
        elif self.upper is not None:
            raise ValueError("only between comparison accepts an upper literal")

    def to_dict(self) -> dict[str, PythonAny]:
        result: dict[str, PythonAny] = {
            "node": "compare",
            "observable_id": self.observable_id,
            "operator": self.operator.value,
            "threshold": self.threshold.to_dict(),
        }
        if self.upper is not None:
            result["upper"] = self.upper.to_dict()
        return result


@dataclass(frozen=True)
class All:
    children: tuple["PredicateNode", ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "children", tuple(self.children))
        if not self.children:
            raise ValueError("all node requires at least one child")

    def to_dict(self) -> dict[str, PythonAny]:
        children = sorted(
            (child.to_dict() for child in self.children), key=canonical_json)
        return {"node": "all", "children": children}


@dataclass(frozen=True)
class Any:
    children: tuple["PredicateNode", ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "children", tuple(self.children))
        if not self.children:
            raise ValueError("any node requires at least one child")

    def to_dict(self) -> dict[str, PythonAny]:
        children = sorted(
            (child.to_dict() for child in self.children), key=canonical_json)
        return {"node": "any", "children": children}


@dataclass(frozen=True)
class Not:
    child: "PredicateNode"

    def to_dict(self) -> dict[str, PythonAny]:
        return {"node": "not", "child": self.child.to_dict()}


PredicateNode: TypeAlias = Compare | All | Any | Not
AllOf = All
AnyOf = Any


def predicate_from_dict(data: Mapping[str, PythonAny]) -> PredicateNode:
    if not isinstance(data, Mapping):
        raise ValueError("predicate node must be an object")
    node = data.get("node")
    if node == "compare":
        allowed = {"node", "observable_id", "operator", "threshold", "upper"}
        if set(data) - allowed:
            raise ValueError("compare node has unknown fields")
        threshold = data.get("threshold")
        if not isinstance(threshold, Mapping):
            raise ValueError("compare threshold must be an object")
        upper_data = data.get("upper")
        if upper_data is not None and not isinstance(upper_data, Mapping):
            raise ValueError("compare upper must be an object")
        return Compare(
            observable_id=str(data.get("observable_id", "")),
            operator=str(data.get("operator", "")),
            threshold=Literal.from_dict(threshold),
            upper=Literal.from_dict(upper_data) if upper_data is not None else None,
        )
    if node in {"all", "any"}:
        if set(data) != {"node", "children"}:
            raise ValueError(f"{node} node must contain exactly node and children")
        raw_children = data["children"]
        if not isinstance(raw_children, (list, tuple)):
            raise ValueError(f"{node} children must be a list")
        children = tuple(predicate_from_dict(item) for item in raw_children)
        return All(children) if node == "all" else Any(children)
    if node == "not":
        if set(data) != {"node", "child"}:
            raise ValueError("not node must contain exactly node and child")
        return Not(predicate_from_dict(data["child"]))
    raise ValueError(f"unknown predicate node {node!r}")


class PredicateCompileError(ValueError):
    pass


class UnknownObservableError(PredicateCompileError):
    pass


class UnitMismatchError(PredicateCompileError):
    pass


class PredicateTypeError(PredicateCompileError):
    pass


@dataclass(frozen=True)
class EvaluationTrace:
    result: Observation
    observations: tuple[tuple[str, Observation], ...]
    predicate_digest: str
    taint: Taint

    def to_dict(self) -> dict[str, PythonAny]:
        return {
            "predicate_digest": self.predicate_digest,
            "taint": self.taint.value,
            "result": self.result.to_dict(),
            "observations": [
                {"observable_id": key, "result": result.to_dict()}
                for key, result in self.observations
            ],
        }


@dataclass(frozen=True)
class CompiledPredicate:
    predicate: PredicateNode
    contracts: tuple[ObservableContract, ...]
    taint: Taint
    digest: str

    def contract_bindings(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (contract.observable_id, contract.version_digest())
            for contract in self.contracts
        )

    def canonical_dict(self) -> dict[str, PythonAny]:
        return {
            "schema": PREDICATE_IR_SCHEMA,
            "predicate": self.predicate.to_dict(),
            "contracts": [
                {"observable_id": key, "version_digest": digest}
                for key, digest in self.contract_bindings()
            ],
            "taint": self.taint.value,
        }

    def evaluate(self, context: PythonAny) -> Observation:
        return self.evaluate_with_trace(context).result

    def evaluate_with_trace(self, context: PythonAny) -> EvaluationTrace:
        contract_map = {
            contract.observable_id: contract for contract in self.contracts}
        observations: dict[str, Observation] = {}

        def observe(observable_id: str) -> Observation:
            if observable_id not in observations:
                observations[observable_id] = contract_map[observable_id].evaluate(context)
            return observations[observable_id]

        result = _evaluate_node(self.predicate, observe)
        return EvaluationTrace(
            result=result,
            observations=tuple(sorted(observations.items())),
            predicate_digest=self.digest,
            taint=self.taint,
        )


def _literal_type_issue(literal: Literal, contract: ObservableContract) -> str | None:
    if literal.unit is not contract.unit:
        return (
            f"literal unit {literal.unit.value} does not match observable unit "
            f"{contract.unit.value}"
        )
    if not _value_matches_type(literal.value, contract.value_type):
        return f"literal has the wrong {contract.value_type.value} value type"
    return None


def _compile_node(
    node: PredicateNode,
    registry: ObservableRegistry,
    used: dict[str, ObservableContract],
) -> None:
    if isinstance(node, Compare):
        try:
            contract = registry.get(node.observable_id)
        except KeyError as exc:
            raise UnknownObservableError(str(exc)) from exc
        issue = _literal_type_issue(node.threshold, contract)
        if issue:
            if node.threshold.unit is not contract.unit:
                raise UnitMismatchError(f"{node.observable_id}: {issue}")
            raise PredicateTypeError(f"{node.observable_id}: {issue}")
        if node.operator is not ComparisonOperator.EQ \
                and contract.value_type not in {ValueType.INTEGER, ValueType.REAL}:
            raise PredicateTypeError(
                f"{node.observable_id}: {node.operator.value} requires a numeric observable")
        if node.operator is ComparisonOperator.BETWEEN:
            assert node.upper is not None
            issue = _literal_type_issue(node.upper, contract)
            if issue:
                if node.upper.unit is not contract.unit:
                    raise UnitMismatchError(f"{node.observable_id}: {issue}")
                raise PredicateTypeError(f"{node.observable_id}: {issue}")
            if node.threshold.value > node.upper.value:
                raise PredicateCompileError(
                    f"{node.observable_id}: between lower bound exceeds upper bound")
        used[contract.observable_id] = contract
        return
    if isinstance(node, (All, Any)):
        for child in node.children:
            if not isinstance(child, (Compare, All, Any, Not)):
                raise PredicateCompileError("Boolean node has an invalid child")
            _compile_node(child, registry, used)
        return
    if isinstance(node, Not):
        if not isinstance(node.child, (Compare, All, Any, Not)):
            raise PredicateCompileError("not node has an invalid child")
        _compile_node(node.child, registry, used)
        return
    raise PredicateCompileError("invalid predicate node")


def compile_predicate(
    predicate: PredicateNode | Mapping[str, PythonAny],
    registry: ObservableRegistry,
) -> CompiledPredicate:
    """Typecheck and bind every leaf to an already-registered observable."""
    if isinstance(predicate, Mapping):
        try:
            predicate = predicate_from_dict(predicate)
        except (TypeError, ValueError) as exc:
            raise PredicateCompileError(str(exc)) from exc
    if not isinstance(predicate, (Compare, All, Any, Not)):
        raise PredicateCompileError("predicate must be a Boolean AST node")
    used: dict[str, ObservableContract] = {}
    _compile_node(predicate, registry, used)
    contracts = tuple(used[key] for key in sorted(used))
    taint = Taint.HYBRID if any(
        contract.source is ObservableSource.ORACLE for contract in contracts
    ) else Taint.PURE
    body = {
        "schema": PREDICATE_IR_SCHEMA,
        "predicate": predicate.to_dict(),
        "contracts": [
            {
                "observable_id": contract.observable_id,
                "version_digest": contract.version_digest(),
            }
            for contract in contracts
        ],
        "taint": taint.value,
    }
    return CompiledPredicate(
        predicate=predicate,
        contracts=contracts,
        taint=taint,
        digest=canonical_digest(body),
    )


def _unknown_choice(results: tuple[Observation, ...]) -> Indeterminate | None:
    for result in results:
        if isinstance(result, Indeterminate):
            return result
    return None


def _truth_result(value: bool) -> Present:
    return Present(value, Unit.BOOLEAN)


def _evaluate_node(
    node: PredicateNode,
    observe: Callable[[str], Observation],
) -> Observation:
    if isinstance(node, Compare):
        observation = observe(node.observable_id)
        # Absence is an observed negative fact about this atom, not a failed
        # measurement.  The raw SemanticAbsent certificate remains available
        # in EvaluationTrace.observations.
        if isinstance(observation, SemanticAbsent):
            return _truth_result(False)
        if not isinstance(observation, Present):
            return observation
        try:
            numeric = isinstance(observation.value, Real) \
                and not isinstance(observation.value, bool)
            if numeric:
                assert observation.lower is not None and observation.upper is not None
                decision = _compare_numeric_interval(
                    node.operator,
                    observation.lower,
                    observation.upper,
                    node.threshold.value,
                    node.upper.value if node.upper is not None else None,
                )
                if decision is None:
                    return Indeterminate(
                        "comparison-boundary-overlap",
                        (
                            f"{node.observable_id} interval "
                            f"[{observation.lower!r}, {observation.upper!r}] "
                            f"overlaps the {node.operator.value} boundary"
                        ),
                        observation.provenance,
                    )
                result = decision
            elif node.operator is ComparisonOperator.EQ:
                result = observation.value == node.threshold.value
            elif node.operator is ComparisonOperator.LT:
                result = observation.value < node.threshold.value
            elif node.operator is ComparisonOperator.LE:
                result = observation.value <= node.threshold.value
            elif node.operator is ComparisonOperator.GT:
                result = observation.value > node.threshold.value
            elif node.operator is ComparisonOperator.GE:
                result = observation.value >= node.threshold.value
            else:
                assert node.upper is not None
                result = node.threshold.value <= observation.value <= node.upper.value
        except Exception as exc:
            return Error("comparison-error", f"{type(exc).__name__}: {exc}")
        return _truth_result(bool(result))

    if isinstance(node, Not):
        child = _evaluate_node(node.child, observe)
        if isinstance(child, Present):
            return _truth_result(not bool(child.value))
        return child

    if isinstance(node, (All, Any)):
        # Evaluate every child: Error is an invalid computation and therefore
        # cannot be hidden behind the short-circuit value of another child.
        ordered = tuple(sorted(node.children, key=lambda child: canonical_json(child.to_dict())))
        results = tuple(_evaluate_node(child, observe) for child in ordered)
        error = next((result for result in results if isinstance(result, Error)), None)
        if error is not None:
            return error
        present_truths = tuple(
            bool(result.value) for result in results if isinstance(result, Present))
        unknown = _unknown_choice(results)
        if isinstance(node, All):
            if any(not value for value in present_truths):
                return _truth_result(False)
            if unknown is not None:
                return unknown
            return _truth_result(True)
        if any(present_truths):
            return _truth_result(True)
        if unknown is not None:
            return unknown
        return _truth_result(False)

    return Error("invalid-compiled-node", type(node).__name__)


def _compare_numeric_interval(
    operator: ComparisonOperator,
    lower: int | float,
    upper: int | float,
    threshold: int | float | bool | str,
    second: int | float | bool | str | None,
) -> bool | None:
    """Return a robust interval decision, or None when the boundary overlaps."""
    assert isinstance(threshold, Real) and not isinstance(threshold, bool)
    if operator is ComparisonOperator.EQ:
        if lower == upper == threshold:
            return True
        if threshold < lower or threshold > upper:
            return False
        return None
    if operator is ComparisonOperator.LT:
        if upper < threshold:
            return True
        if lower >= threshold:
            return False
        return None
    if operator is ComparisonOperator.LE:
        if upper <= threshold:
            return True
        if lower > threshold:
            return False
        return None
    if operator is ComparisonOperator.GT:
        if lower > threshold:
            return True
        if upper <= threshold:
            return False
        return None
    if operator is ComparisonOperator.GE:
        if lower >= threshold:
            return True
        if upper < threshold:
            return False
        return None
    assert operator is ComparisonOperator.BETWEEN
    assert isinstance(second, Real) and not isinstance(second, bool)
    if lower >= threshold and upper <= second:
        return True
    if upper < threshold or lower > second:
        return False
    return None


__all__ = [
    "All",
    "AllOf",
    "Any",
    "AnyOf",
    "Compare",
    "ComparisonOperator",
    "CompiledPredicate",
    "Error",
    "EvaluationTrace",
    "Indeterminate",
    "Invariance",
    "Literal",
    "Not",
    "OBSERVABLE_CONTRACT_SCHEMA",
    "ObservableContract",
    "ObservableRegistry",
    "ObservableSource",
    "Observation",
    "PREDICATE_IR_SCHEMA",
    "PredicateCompileError",
    "PredicateNode",
    "PredicateTypeError",
    "Present",
    "Reducer",
    "SemanticAbsent",
    "Taint",
    "Unit",
    "UnitMismatchError",
    "UnknownObservableError",
    "ValueType",
    "canonical_digest",
    "canonical_json",
    "compile_predicate",
    "predicate_from_dict",
]

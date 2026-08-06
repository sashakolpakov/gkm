"""Backend boundary for typed Bongard predicates.

The scientific contract is the serialized closed IR plus its typed registry
snapshot and four-disposition evidence. It is not a Lean term, a Python source
string, or the identity of a particular checker. Backends consume that
contract; they do not define it.

The canonical reference backend is pure Python. It can both invoke registered
Python legs during an online episode and replay already committed atom evidence
without invoking a leg, model, subprocess, or proof assistant. A future Lean
backend may independently check the latter operation, but benchmark execution
and cold replay must not depend on it.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from bongard.artifacts import (
    QueryReplayInput,
    TruthEvidenceRecord,
    replay_cold_payload,
    replay_query,
)
from bongard.evidence import Evidence
from bongard.ir import Formula, evaluate_formula, validate_formula
from bongard.legs.contracts import (
    LegRegistry,
    RegistrySnapshot,
    TypedValue,
    ValueType,
)


@runtime_checkable
class PredicateBackend(Protocol):
    """Operational contract shared by predicate implementations.

    ``backend_id`` is diagnostic metadata only. It is deliberately absent
    from formula and evidence digests: changing or adding an independent
    checker must not change the scientific statement being checked.
    """

    backend_id: str

    def validate(
        self,
        formula: Formula,
        registry: LegRegistry | RegistrySnapshot,
        boundary_types: Mapping[str, ValueType],
    ) -> None:
        """Check the closed grammar and every static typed attachment."""

    def evaluate(
        self,
        formula: Formula,
        registry: LegRegistry,
        bindings: Mapping[str, TypedValue],
    ) -> Evidence[bool]:
        """Invoke registered legs and evaluate one formula in an episode."""

    def replay_query(
        self,
        formula: Formula,
        query: QueryReplayInput,
    ) -> Evidence[bool]:
        """Compose committed atom evidence without invoking any leg."""

    def replay_payload(
        self,
        formula_data: Mapping[str, Any],
        cold_inputs_data: Mapping[str, Any],
    ) -> tuple[tuple[str, TruthEvidenceRecord], ...]:
        """Replay decoded canonical JSON with digest and path checks."""


class PythonPredicateBackend:
    """Pure-Python reference semantics for the backend-neutral contract."""

    backend_id = "python-closed-ir/v1"

    def validate(
        self,
        formula: Formula,
        registry: LegRegistry | RegistrySnapshot,
        boundary_types: Mapping[str, ValueType],
    ) -> None:
        validate_formula(formula, registry, boundary_types)

    def evaluate(
        self,
        formula: Formula,
        registry: LegRegistry,
        bindings: Mapping[str, TypedValue],
    ) -> Evidence[bool]:
        return evaluate_formula(formula, registry, bindings)

    def replay_query(
        self,
        formula: Formula,
        query: QueryReplayInput,
    ) -> Evidence[bool]:
        return replay_query(formula, query)

    def replay_payload(
        self,
        formula_data: Mapping[str, Any],
        cold_inputs_data: Mapping[str, Any],
    ) -> tuple[tuple[str, TruthEvidenceRecord], ...]:
        return replay_cold_payload(formula_data, cold_inputs_data)


PYTHON_PREDICATE_BACKEND: PredicateBackend = PythonPredicateBackend()


__all__ = [
    "PYTHON_PREDICATE_BACKEND",
    "PredicateBackend",
    "PythonPredicateBackend",
]

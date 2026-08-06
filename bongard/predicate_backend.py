"""Internal pure-Python execution facade for typed Bongard predicates.

Python is the sole authoritative semantics.  The scientific contract is the
Python-defined serialized closed IR, typed registry snapshot, and
four-disposition evidence; no interchangeable execution backend may redefine
their meaning.  The legacy ``PredicateBackend`` name below describes only an
internal Python protocol used to keep validation, online evaluation, and cold
replay behind one narrow facade.

An external checker must consume an already-frozen Python artifact through
``bongard.semantic_checker`` and emit a detached, non-authoritative sidecar.
Installing, changing, failing, disagreeing, or deleting such a checker cannot
change a predicate, evidence value, formula, result, decision, replay, or ID.
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
    """Legacy-named internal protocol for the authoritative Python executor.

    ``backend_id`` is diagnostic metadata only. It is deliberately absent
    from formula and evidence digests. This protocol is not an extension
    point for proof assistants or alternative scientific semantics.
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
    """The sole authoritative execution semantics for the closed Python IR."""

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

"""Gluing (cofibration) contracts for witness-preserving extensions.

A practical cofibration here is the canonical morphism ``A -> A ⊔_I P``: the
source witness ``A`` is glued to a patch ``P`` along an interface ``I``.

Earlier revisions verified this with strict inclusions: every declared source
field had to appear verbatim in the target, and the projection had to return
an object equal to the source on the nose.  That is the wrong contract.  A
gluing morphism identifies structure along the interface, so part IDs may be
renumbered, coordinates re-expressed, and derived fields recomputed.  Only
the declared invariants must survive, and only up to the glue map.

The verified contract is:

1. patch locality        — the declared interface and added fields are
   populated on the target (the patch attaches along ``I``; it does not
   scribble over undeclared structure);
2. source glued in       — the target contains a substructure that is
   glue-equivalent to the source: same witness type and same declared
   invariants, modulo a single consistent renaming of witness IDs and a
   numeric tolerance;
3. projection            — if a projection is available, applying it to the
   target recovers the source up to glue-equivalence (never ``==``);
4. trace transport       — recorded execution traces are compared with the
   same glue-equivalence, not verbatim.

Check (2) falls back to check (3) when the target does not carry a literal
copy of the source structure (e.g. a QuadrilateralWitness glued into a
PartGraphWitness survives only as a part contour; the declared
``projection_leg`` is then the arrow that re-extracts it).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any, Callable

import numpy as np

# Field names whose string values are identifiers or producer-side labels:
# gluing may rename them, as long as the renaming is one consistent
# bijection per check.
ID_FIELD_NAMES = frozenset({
    "part_id",
    "source_id",
    "source_component_id",
    "object_id",
    "source_a",
    "source_b",
    "id",
    "role",
})

# Bookkeeping fields carry provenance, not structure.
IGNORED_FIELDS = frozenset({"provenance"})

DEFAULT_TOLERANCE = 2.0


@dataclass(frozen=True)
class CofibrationSpec:
    """Declaration of a gluing ``source -> source ⊔_interface patch``.

    Specs are NOT hard-coded per concept in this library.  The proposer
    generates them inside its typed cone IR (binding ``source_node`` and
    ``target_node`` to diagram nodes); the harness only verifies them
    mechanically.  Naming an ``attachment_leg`` that does not exist yet is a
    legitimate outcome — the compiler reports it as MISSING_LEG instead of
    accepting a weakened cone.  Hard-coded specs are allowed only as unit
    test fixtures.
    """

    name: str
    source_type: str
    target_type: str
    interface_fields: tuple[str, ...]
    added_fields: tuple[str, ...]
    attachment_leg: str
    preserved_invariants: tuple[str, ...] = ()
    projection_leg: str | None = None
    source_node: str = ""
    target_node: str = ""
    tolerance: float = DEFAULT_TOLERANCE
    complexity_cost: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "CofibrationSpec":
        return CofibrationSpec(
            name=str(data.get("name", "unnamed_gluing")),
            source_type=str(data.get("source_type", "")),
            target_type=str(data.get("target_type", "")),
            interface_fields=tuple(str(x) for x in data.get("interface_fields", ())),
            added_fields=tuple(str(x) for x in data.get("added_fields", ())),
            attachment_leg=str(data.get("attachment_leg", "")),
            preserved_invariants=tuple(str(x) for x in data.get("preserved_invariants", ())),
            projection_leg=(str(data["projection_leg"])
                            if data.get("projection_leg") else None),
            source_node=str(data.get("source_node", "")),
            target_node=str(data.get("target_node", "")),
            tolerance=float(data.get("tolerance", DEFAULT_TOLERANCE)),
            complexity_cost=int(data.get("complexity_cost", 1)),
        )


@dataclass(frozen=True)
class CofibrationCheck:
    ok: bool
    first_failed: str = ""
    details: str = ""
    glue_map: tuple[tuple[str, str], ...] = ()


class _GlueMap:
    """One consistent bijection between source and target identifiers."""

    def __init__(self) -> None:
        self.forward: dict[str, str] = {}
        self.backward: dict[str, str] = {}

    def bind(self, src: str, dst: str) -> bool:
        if src in self.forward:
            return self.forward[src] == dst
        if dst in self.backward:
            return self.backward[dst] == src
        self.forward[src] = dst
        self.backward[dst] = src
        return True

    def snapshot(self) -> tuple[dict[str, str], dict[str, str]]:
        return dict(self.forward), dict(self.backward)

    def restore(self, snap: tuple[dict[str, str], dict[str, str]]) -> None:
        self.forward, self.backward = snap

    def items(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self.forward.items()))


def _field_items(value: Any) -> list[tuple[str, Any]] | None:
    if is_dataclass(value) and not isinstance(value, type):
        return [(f.name, getattr(value, f.name)) for f in fields(value)]
    if isinstance(value, dict):
        return [(str(k), value[k]) for k in sorted(value, key=str)]
    return None


def glue_equivalent(a: Any, b: Any, glue: _GlueMap,
                    tolerance: float = DEFAULT_TOLERANCE,
                    field_name: str = "") -> bool:
    """Structural equality up to ID renaming and numeric tolerance."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) \
            and np.array_equal(a, b)
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            return a == b
        return abs(float(a) - float(b)) <= tolerance
    if isinstance(a, str) and isinstance(b, str):
        if field_name in ID_FIELD_NAMES:
            return glue.bind(a, b)
        return a == b
    items_a, items_b = _field_items(a), _field_items(b)
    if items_a is not None and items_b is not None:
        if type(a).__name__ != type(b).__name__ and not (
                isinstance(a, dict) and isinstance(b, dict)):
            return False
        items_a = [(k, v) for k, v in items_a if k not in IGNORED_FIELDS]
        items_b = [(k, v) for k, v in items_b if k not in IGNORED_FIELDS]
        names_a = {k for k, _ in items_a}
        names_b = {k for k, _ in items_b}
        if names_a != names_b:
            return False
        return all(
            glue_equivalent(va, dict(items_b)[k], glue, tolerance, k)
            for k, va in items_a
        )
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            return False
        structured = any(_field_items(x) is not None for x in a)
        if not structured:
            return all(glue_equivalent(x, y, glue, tolerance, field_name)
                       for x, y in zip(a, b))
        return _match_multiset(list(a), list(b), glue, tolerance, field_name)
    return a == b


def _match_multiset(src: list, dst: list, glue: _GlueMap,
                    tolerance: float, field_name: str) -> bool:
    """Order-insensitive matching for collections of structured witnesses.

    Gluing may reorder parts; a backtracking bijection search keeps the glue
    map consistent across the whole collection.  Collections here are tiny
    (parts of one panel), so the search is cheap.
    """
    if not src:
        return not dst
    head, rest = src[0], src[1:]
    for i, cand in enumerate(dst):
        snap = glue.snapshot()
        if glue_equivalent(head, cand, glue, tolerance, field_name) and \
                _match_multiset(rest, dst[:i] + dst[i + 1:], glue, tolerance, field_name):
            return True
        glue.restore(snap)
    return False


def _subvalues(value: Any):
    yield value
    items = _field_items(value)
    if items is not None:
        for _, v in items:
            yield from _subvalues(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from _subvalues(v)


def _restricted(source: Any, invariants: tuple[str, ...]) -> Any:
    if not invariants:
        return source
    items = _field_items(source)
    if items is None:
        return source
    keep = {k: v for k, v in items if k in invariants}
    return keep or source


def verify_patch_locality(target: Any, spec: CofibrationSpec) -> CofibrationCheck:
    """The patch attaches along the declared interface and added fields."""
    def _get(name: str) -> Any:
        if isinstance(target, dict):
            return target.get(name)
        return getattr(target, name, None)

    for name in spec.interface_fields:
        value = _get(name)
        if value is None or value == () or value == "":
            return CofibrationCheck(False, "interface_missing", name)
    for name in spec.added_fields:
        value = _get(name)
        if value is None or value == () or value == "":
            return CofibrationCheck(False, "patch_missing", name)
    return CofibrationCheck(True)


def verify_source_glued_in(source: Any, target: Any,
                           spec: CofibrationSpec) -> CofibrationCheck:
    """Some substructure of the target is glue-equivalent to the source."""
    wanted = _restricted(source, spec.preserved_invariants)
    source_type = type(source).__name__
    for sub in _subvalues(target):
        if spec.preserved_invariants:
            if type(sub).__name__ != source_type and _field_items(sub) is None:
                continue
            candidate = _restricted(sub, spec.preserved_invariants) \
                if type(sub).__name__ == source_type else sub
        else:
            candidate = sub
        glue = _GlueMap()
        if glue_equivalent(wanted, candidate, glue, spec.tolerance):
            return CofibrationCheck(True, glue_map=glue.items())
    return CofibrationCheck(False, "source_not_glued",
                            f"no substructure glue-equivalent to {source_type}")


def verify_projection_recovers_source(source: Any, target: Any,
                                      spec: CofibrationSpec,
                                      projection_fn: Callable | None = None
                                      ) -> CofibrationCheck:
    """Projection recovers the source up to the glue map, never ``==``."""
    if projection_fn is None:
        if spec.projection_leg is None:
            return CofibrationCheck(True)
        projected = target.get(spec.projection_leg) if isinstance(target, dict) \
            else getattr(target, spec.projection_leg, None)
    else:
        projected = projection_fn(target)
    if projected is None:
        return CofibrationCheck(False, "projection_missing", spec.projection_leg or "")
    glue = _GlueMap()
    if glue_equivalent(source, projected, glue, spec.tolerance):
        return CofibrationCheck(True, glue_map=glue.items())
    return CofibrationCheck(False, "projection_recovers_source",
                            spec.projection_leg or "projection_fn")


def verify_trace_transport(source_trace: dict[str, Any],
                           target_trace: dict[str, Any],
                           spec: CofibrationSpec) -> CofibrationCheck:
    keys = spec.preserved_invariants or tuple(source_trace)
    glue = _GlueMap()
    for key in keys:
        if key not in source_trace:
            continue
        if not glue_equivalent(source_trace.get(key), target_trace.get(key),
                               glue, spec.tolerance, key):
            return CofibrationCheck(False, "trace_transport", key)
    return CofibrationCheck(True, glue_map=glue.items())


def verify_cofibration(source: Any, target: Any, spec: CofibrationSpec,
                       source_trace: dict[str, Any] | None = None,
                       target_trace: dict[str, Any] | None = None,
                       projection_fn: Callable | None = None) -> CofibrationCheck:
    locality = verify_patch_locality(target, spec)
    if not locality.ok:
        return locality
    glued = verify_source_glued_in(source, target, spec)
    if not glued.ok:
        # The target need not carry a literal copy of the source structure;
        # a declared projection that re-extracts it is an equally valid
        # witness that the gluing preserved the source.
        if spec.projection_leg is not None or projection_fn is not None:
            projected = verify_projection_recovers_source(
                source, target, spec, projection_fn)
            if projected.ok:
                glued = projected
            else:
                return glued
        else:
            return glued
    projection = verify_projection_recovers_source(source, target, spec, projection_fn)
    if not projection.ok:
        return projection
    if source_trace is not None and target_trace is not None:
        transport = verify_trace_transport(source_trace, target_trace, spec)
        if not transport.ok:
            return transport
    return CofibrationCheck(True, glue_map=glued.glue_map or projection.glue_map)

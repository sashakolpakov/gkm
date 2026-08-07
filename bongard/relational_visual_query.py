"""Closed same-binding Python queries over candidate-independent loop scenes.

The query language is intentionally finite and positive.  Version three binds
two distinct loop roles and conjoins registered micro-predicates.  Every
clause in a binding is evaluated against the same role assignment; a triangle
fact from one object can therefore never be combined with the area of another.
There is no negation node, polarity flag, arbitrary expression, callback, or
pixel access in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Sequence, TypeAlias

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.loop_geometry import LoopGeometryWitness
from bongard.loop_scene_witnesses import LoopScenePacket
from bongard.point_contact import PairContactObservation


RELATIONAL_QUERY_SCHEMA = "gkm.bongard-relational-visual-query.v3"
RELATIONAL_RESULT_SCHEMA = "gkm.bongard-relational-visual-query-result.v3"
RELATIONAL_QUERY_ALGORITHM_ID = "bongard.relational-visual-query/python-v3"

_ROLE_ID = re.compile(r"role-[0-9]{2}\Z")
_CLAUSE_ID = re.compile(r"clause-[0-9]{2}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

ALLOWED_SIDE_COUNTS = tuple(range(3, 9))
ALLOWED_AREA_RATIOS = (
    (1, 16),
    (1, 12),
    (1, 8),
    (1, 6),
    (1, 4),
    (1, 3),
    (1, 2),
)
ALLOWED_OBLIQUENESS_THRESHOLDS_MILLIDEGREES = (
    5_000,
    10_000,
    15_000,
    20_000,
)


def _exact_fields(
    data: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if not isinstance(data, Mapping) or set(data) != expected:
        raise ValueError(f"{label} fields differ from the static schema")


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase sha256")
    return value


def relational_query_source_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def relational_query_algorithm_digest() -> str:
    return canonical_digest(
        {
            "algorithm_id": RELATIONAL_QUERY_ALGORITHM_ID,
            "source_digest": relational_query_source_digest(),
            "roles": (
                "two distinct scenario-local loops meeting the frozen exact "
                "geometry-resolution floor"
            ),
            "formula": "closed conjunction only",
            "predicates": [
                "loop.area_ratio_at_most",
                "loop.edge_obliqueness_at_least",
                "loop.side_count_equal",
                "pair.point_contact",
            ],
            "allowed_side_counts": list(ALLOWED_SIDE_COUNTS),
            "allowed_area_ratios": [list(item) for item in ALLOWED_AREA_RATIOS],
            "allowed_obliqueness_thresholds_millidegrees": list(
                ALLOWED_OBLIQUENESS_THRESHOLDS_MILLIDEGREES
            ),
            "aggregation": {
                "conjunction": "absence-blocker, then error, then indeterminate",
                "existential": (
                    "present-witness; otherwise propagate role-domain/binding "
                    "error or uncertainty; absence only over a resolved exhaustive domain"
                ),
                "scenarios": "unanimous present/absence; else error or indeterminate",
            },
        }
    )


@dataclass(frozen=True, order=True, slots=True)
class Rational:
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        _integer(self.numerator, "ratio numerator", minimum=1)
        _integer(self.denominator, "ratio denominator", minimum=1)
        if self.numerator >= self.denominator:
            raise ValueError("area ratio must be a proper positive fraction")
        from math import gcd

        if gcd(self.numerator, self.denominator) != 1:
            raise ValueError("area ratio must be reduced")
        if (self.numerator, self.denominator) not in ALLOWED_AREA_RATIOS:
            raise ValueError("area ratio is outside the frozen finite grid")

    def to_data(self) -> dict[str, int]:
        return {"numerator": self.numerator, "denominator": self.denominator}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "Rational":
        _exact_fields(data, frozenset({"numerator", "denominator"}), "ratio")
        return cls(data["numerator"], data["denominator"])


@dataclass(frozen=True, order=True, slots=True)
class RoleSpec:
    role_id: str
    domain: str = "substantive_closed_loop"

    def __post_init__(self) -> None:
        if not isinstance(self.role_id, str) or _ROLE_ID.fullmatch(self.role_id) is None:
            raise ValueError("role_id is not canonical")
        if self.domain != "substantive_closed_loop":
            raise ValueError(
                "v3 roles range only over explicitly certified substantive loops"
            )

    def to_data(self) -> dict[str, str]:
        return {"role_id": self.role_id, "domain": self.domain}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "RoleSpec":
        _exact_fields(data, frozenset({"role_id", "domain"}), "role spec")
        return cls(data["role_id"], data["domain"])


class PredicateID(str, Enum):
    SIDE_COUNT_EQUAL = "loop.side_count_equal"
    AREA_RATIO_AT_MOST = "loop.area_ratio_at_most"
    EDGE_OBLIQUENESS_AT_LEAST = "loop.edge_obliqueness_at_least"
    POINT_CONTACT = "pair.point_contact"


@dataclass(frozen=True, slots=True)
class SideCountClause:
    clause_id: str
    role_id: str
    count: int
    predicate: PredicateID = PredicateID.SIDE_COUNT_EQUAL

    def __post_init__(self) -> None:
        _validate_clause_id(self.clause_id)
        _validate_role_id(self.role_id)
        if self.count not in ALLOWED_SIDE_COUNTS:
            raise ValueError("side count is outside the frozen finite grid")
        if self.predicate is not PredicateID.SIDE_COUNT_EQUAL:
            raise ValueError("side-count clause predicate is fixed")

    def to_data(self) -> dict[str, object]:
        return {
            "clause_id": self.clause_id,
            "predicate": self.predicate.value,
            "role_id": self.role_id,
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class AreaRatioClause:
    clause_id: str
    numerator_role_id: str
    denominator_role_id: str
    ratio: Rational
    predicate: PredicateID = PredicateID.AREA_RATIO_AT_MOST

    def __post_init__(self) -> None:
        _validate_clause_id(self.clause_id)
        _validate_role_id(self.numerator_role_id)
        _validate_role_id(self.denominator_role_id)
        if self.numerator_role_id == self.denominator_role_id:
            raise ValueError("area ratio roles must differ")
        if not isinstance(self.ratio, Rational):
            raise TypeError("area ratio clause requires a Rational")
        if self.predicate is not PredicateID.AREA_RATIO_AT_MOST:
            raise ValueError("area-ratio clause predicate is fixed")

    def to_data(self) -> dict[str, object]:
        return {
            "clause_id": self.clause_id,
            "predicate": self.predicate.value,
            "numerator_role_id": self.numerator_role_id,
            "denominator_role_id": self.denominator_role_id,
            "ratio": self.ratio.to_data(),
        }


@dataclass(frozen=True, slots=True)
class EdgeObliquenessClause:
    clause_id: str
    role_id: str
    threshold_millidegrees: int
    predicate: PredicateID = PredicateID.EDGE_OBLIQUENESS_AT_LEAST

    def __post_init__(self) -> None:
        _validate_clause_id(self.clause_id)
        _validate_role_id(self.role_id)
        if self.threshold_millidegrees not in (
            ALLOWED_OBLIQUENESS_THRESHOLDS_MILLIDEGREES
        ):
            raise ValueError("obliqueness threshold is outside the frozen finite grid")
        if self.predicate is not PredicateID.EDGE_OBLIQUENESS_AT_LEAST:
            raise ValueError("obliqueness clause predicate is fixed")

    def to_data(self) -> dict[str, object]:
        return {
            "clause_id": self.clause_id,
            "predicate": self.predicate.value,
            "role_id": self.role_id,
            "threshold_millidegrees": self.threshold_millidegrees,
        }


@dataclass(frozen=True, slots=True)
class PointContactClause:
    clause_id: str
    first_role_id: str
    second_role_id: str
    predicate: PredicateID = PredicateID.POINT_CONTACT

    def __post_init__(self) -> None:
        _validate_clause_id(self.clause_id)
        _validate_role_id(self.first_role_id)
        _validate_role_id(self.second_role_id)
        if self.first_role_id == self.second_role_id:
            raise ValueError("point-contact roles must differ")
        if self.predicate is not PredicateID.POINT_CONTACT:
            raise ValueError("point-contact clause predicate is fixed")

    def to_data(self) -> dict[str, object]:
        return {
            "clause_id": self.clause_id,
            "predicate": self.predicate.value,
            "first_role_id": self.first_role_id,
            "second_role_id": self.second_role_id,
        }


Clause: TypeAlias = (
    SideCountClause | AreaRatioClause | EdgeObliquenessClause | PointContactClause
)


def _validate_clause_id(value: object) -> str:
    if not isinstance(value, str) or _CLAUSE_ID.fullmatch(value) is None:
        raise ValueError("clause_id is not canonical")
    return value


def _validate_role_id(value: object) -> str:
    if not isinstance(value, str) or _ROLE_ID.fullmatch(value) is None:
        raise ValueError("role_id is not canonical")
    return value


def clause_from_data(data: Mapping[str, Any]) -> Clause:
    if not isinstance(data, Mapping) or not isinstance(data.get("predicate"), str):
        raise ValueError("query clause must name a predicate")
    predicate = PredicateID(data["predicate"])
    if predicate is PredicateID.SIDE_COUNT_EQUAL:
        _exact_fields(
            data,
            frozenset({"clause_id", "predicate", "role_id", "count"}),
            "side-count clause",
        )
        return SideCountClause(data["clause_id"], data["role_id"], data["count"])
    if predicate is PredicateID.AREA_RATIO_AT_MOST:
        _exact_fields(
            data,
            frozenset(
                {
                    "clause_id",
                    "predicate",
                    "numerator_role_id",
                    "denominator_role_id",
                    "ratio",
                }
            ),
            "area-ratio clause",
        )
        ratio = data["ratio"]
        if not isinstance(ratio, Mapping):
            raise TypeError("area ratio must be an object")
        return AreaRatioClause(
            data["clause_id"],
            data["numerator_role_id"],
            data["denominator_role_id"],
            Rational.from_data(ratio),
        )
    if predicate is PredicateID.POINT_CONTACT:
        _exact_fields(
            data,
            frozenset(
                {"clause_id", "predicate", "first_role_id", "second_role_id"}
            ),
            "point-contact clause",
        )
        return PointContactClause(
            data["clause_id"], data["first_role_id"], data["second_role_id"]
        )
    _exact_fields(
        data,
        frozenset(
            {"clause_id", "predicate", "role_id", "threshold_millidegrees"}
        ),
        "obliqueness clause",
    )
    return EdgeObliquenessClause(
        data["clause_id"], data["role_id"], data["threshold_millidegrees"]
    )


@dataclass(frozen=True, slots=True)
class RelationalVisualQuery:
    roles: tuple[RoleSpec, RoleSpec]
    distinct: tuple[tuple[str, str], ...]
    clauses: tuple[Clause, ...]
    all_clause_ids: tuple[str, ...]
    algorithm_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.roles, tuple) or len(self.roles) != 2 or any(
            not isinstance(item, RoleSpec) for item in self.roles
        ):
            raise TypeError("v3 query requires exactly two typed roles")
        if tuple(item.role_id for item in self.roles) != ("role-00", "role-01"):
            raise ValueError("v3 query roles must be role-00 and role-01")
        if self.distinct != (("role-00", "role-01"),):
            raise ValueError("v3 query must require its two roles to be distinct")
        if not isinstance(self.clauses, tuple) or not self.clauses:
            raise ValueError("query clauses must be a nonempty tuple")
        if any(
            not isinstance(
                item,
                (
                    SideCountClause,
                    AreaRatioClause,
                    EdgeObliquenessClause,
                    PointContactClause,
                )
            )
            for item in self.clauses
        ):
            raise TypeError("query contains an unknown clause type")
        clause_ids = tuple(item.clause_id for item in self.clauses)
        if clause_ids != tuple(f"clause-{index:02d}" for index in range(len(clause_ids))):
            raise ValueError("query clause IDs must be consecutive and ordered")
        if self.all_clause_ids != clause_ids:
            raise ValueError("closed conjunction must reference every clause in order")
        roles = {item.role_id for item in self.roles}
        for clause in self.clauses:
            if isinstance(clause, (SideCountClause, EdgeObliquenessClause)):
                referenced = {clause.role_id}
            elif isinstance(clause, AreaRatioClause):
                referenced = {
                    clause.numerator_role_id,
                    clause.denominator_role_id,
                }
            else:
                referenced = {clause.first_role_id, clause.second_role_id}
            if not referenced <= roles:
                raise ValueError("query clause references an undeclared role")
        _digest(self.algorithm_digest, "relational query algorithm_digest")
        if self.algorithm_digest != relational_query_algorithm_digest():
            raise ValueError("relational query algorithm digest drifted")

    @classmethod
    def factorized_shape_ratio(
        cls,
        *,
        numerator_side_count: int,
        denominator_side_count: int,
        ratio: Rational,
        denominator_obliqueness_millidegrees: int | None = None,
        require_point_contact: bool = False,
    ) -> "RelationalVisualQuery":
        clauses: list[Clause] = [
            SideCountClause("clause-00", "role-00", numerator_side_count),
            SideCountClause("clause-01", "role-01", denominator_side_count),
            AreaRatioClause("clause-02", "role-00", "role-01", ratio),
        ]
        if denominator_obliqueness_millidegrees is not None:
            clauses.append(
                EdgeObliquenessClause(
                    f"clause-{len(clauses):02d}",
                    "role-01",
                    denominator_obliqueness_millidegrees,
                )
            )
        if type(require_point_contact) is not bool:
            raise TypeError("require_point_contact must be a literal Boolean")
        if require_point_contact:
            clauses.append(
                PointContactClause(
                    f"clause-{len(clauses):02d}", "role-00", "role-01"
                )
            )
        return cls(
            roles=(RoleSpec("role-00"), RoleSpec("role-01")),
            distinct=(("role-00", "role-01"),),
            clauses=tuple(clauses),
            all_clause_ids=tuple(item.clause_id for item in clauses),
            algorithm_digest=relational_query_algorithm_digest(),
        )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": RELATIONAL_QUERY_SCHEMA,
            "algorithm_id": RELATIONAL_QUERY_ALGORITHM_ID,
            "algorithm_digest": self.algorithm_digest,
            "roles": [item.to_data() for item in self.roles],
            "distinct": [list(item) for item in self.distinct],
            "clauses": [item.to_data() for item in self.clauses],
            "formula": {"op": "all", "clause_ids": list(self.all_clause_ids)},
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "RelationalVisualQuery":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "algorithm_id",
                    "algorithm_digest",
                    "roles",
                    "distinct",
                    "clauses",
                    "formula",
                }
            ),
            "relational visual query",
        )
        if (
            data["schema"] != RELATIONAL_QUERY_SCHEMA
            or data["algorithm_id"] != RELATIONAL_QUERY_ALGORITHM_ID
        ):
            raise ValueError("unsupported relational visual query")
        roles = data["roles"]
        distinct = data["distinct"]
        clauses = data["clauses"]
        formula = data["formula"]
        if not isinstance(roles, list) or any(not isinstance(item, Mapping) for item in roles):
            raise TypeError("query roles must be an object list")
        if not isinstance(distinct, list) or any(
            not isinstance(item, list)
            or len(item) != 2
            or any(not isinstance(role, str) for role in item)
            for item in distinct
        ):
            raise TypeError("query distinct constraints must be role pairs")
        if not isinstance(clauses, list) or any(
            not isinstance(item, Mapping) for item in clauses
        ):
            raise TypeError("query clauses must be an object list")
        if not isinstance(formula, Mapping):
            raise TypeError("query formula must be an object")
        _exact_fields(formula, frozenset({"op", "clause_ids"}), "query formula")
        if formula["op"] != "all" or not isinstance(formula["clause_ids"], list):
            raise ValueError("query formula must be a closed conjunction")
        return cls(
            roles=tuple(RoleSpec.from_data(item) for item in roles),  # type: ignore[arg-type]
            distinct=tuple((item[0], item[1]) for item in distinct),
            clauses=tuple(clause_from_data(item) for item in clauses),
            all_clause_ids=tuple(formula["clause_ids"]),
            algorithm_digest=data["algorithm_digest"],
        )

    def digest(self) -> str:
        return canonical_digest(self.to_data())


@dataclass(frozen=True, slots=True)
class ClauseEvaluation:
    clause_id: str
    disposition: Disposition
    detail: str

    def __post_init__(self) -> None:
        _validate_clause_id(self.clause_id)
        if not isinstance(self.disposition, Disposition):
            raise TypeError("clause evaluation disposition must be a Disposition")
        if not isinstance(self.detail, str) or not self.detail.strip():
            raise ValueError("clause evaluation detail must be nonempty")

    def to_data(self) -> dict[str, str]:
        return {
            "clause_id": self.clause_id,
            "disposition": self.disposition.value,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class BindingEvaluation:
    bindings: tuple[tuple[str, str], ...]
    clauses: tuple[ClauseEvaluation, ...]
    disposition: Disposition

    def __post_init__(self) -> None:
        if self.bindings != tuple(sorted(self.bindings)):
            raise ValueError("role bindings must be role-sorted")
        if tuple(role for role, _ in self.bindings) != ("role-00", "role-01"):
            raise ValueError("binding must assign both v3 roles")
        if self.bindings[0][1] == self.bindings[1][1]:
            raise ValueError("binding violates the distinct-role constraint")
        if not isinstance(self.clauses, tuple) or not self.clauses:
            raise ValueError("binding evaluation requires clause results")
        expected = _conjunction_disposition(self.clauses)
        if self.disposition is not expected:
            raise ValueError("binding disposition disagrees with its clauses")

    def to_data(self) -> dict[str, object]:
        return {
            "bindings": [list(item) for item in self.bindings],
            "clauses": [item.to_data() for item in self.clauses],
            "disposition": self.disposition.value,
        }


@dataclass(frozen=True, slots=True)
class ScenarioQueryResult:
    scenario_id: str
    role_domain: tuple[tuple[str, Disposition], ...]
    bindings: tuple[BindingEvaluation, ...]
    disposition: Disposition
    reason_code: str

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id:
            raise ValueError("scenario query result requires a scenario_id")
        if not isinstance(self.role_domain, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or not isinstance(item[1], Disposition)
            for item in self.role_domain
        ):
            raise TypeError("scenario role_domain must be loop/disposition pairs")
        if self.role_domain != tuple(sorted(self.role_domain, key=lambda item: item[0])):
            raise ValueError("scenario role_domain must be loop-ID sorted")
        if len({item[0] for item in self.role_domain}) != len(self.role_domain):
            raise ValueError("scenario role_domain loop IDs must be unique")
        if not isinstance(self.bindings, tuple):
            raise TypeError("scenario bindings must be a tuple")
        present_ids = tuple(
            loop_id
            for loop_id, disposition in self.role_domain
            if disposition is Disposition.PRESENT
        )
        expected_bindings = tuple(
            (("role-00", first), ("role-01", second))
            for first in present_ids
            for second in present_ids
            if first != second
        )
        if tuple(item.bindings for item in self.bindings) != expected_bindings:
            raise ValueError(
                "scenario bindings do not exhaust the resolved role domain"
            )
        expected = _existential_disposition(
            self.bindings,
            tuple(item[1] for item in self.role_domain),
        )
        if self.disposition is not expected:
            raise ValueError("scenario disposition disagrees with exhaustive bindings")
        expected_reason = {
            Disposition.PRESENT: "binding_witness",
            Disposition.CERTIFIED_ABSENT: "all_bindings_blocked",
            Disposition.INDETERMINATE: "unresolved_binding",
            Disposition.ERROR: "binding_error",
        }[self.disposition]
        if self.reason_code != expected_reason:
            raise ValueError("scenario reason_code disagrees with disposition")

    def to_data(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "role_domain": [
                {"loop_id": loop_id, "disposition": disposition.value}
                for loop_id, disposition in self.role_domain
            ],
            "bindings": [item.to_data() for item in self.bindings],
            "disposition": self.disposition.value,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True, slots=True)
class RelationalQueryResult:
    panel_digest: str
    packet_digest: str
    query_digest: str
    evaluator_algorithm_digest: str
    scenarios: tuple[ScenarioQueryResult, ...]
    disposition: Disposition

    def __post_init__(self) -> None:
        _digest(self.panel_digest, "result panel_digest")
        _digest(self.packet_digest, "result packet_digest")
        _digest(self.query_digest, "result query_digest")
        _digest(self.evaluator_algorithm_digest, "result evaluator_algorithm_digest")
        if self.evaluator_algorithm_digest != relational_query_algorithm_digest():
            raise ValueError("result evaluator algorithm digest drifted")
        if not isinstance(self.scenarios, tuple) or not self.scenarios:
            raise ValueError("result scenarios must be a nonempty tuple")
        expected = _scenario_consensus(tuple(item.disposition for item in self.scenarios))
        if self.disposition is not expected:
            raise ValueError("result disposition disagrees with scenario consensus")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": RELATIONAL_RESULT_SCHEMA,
            "panel_digest": self.panel_digest,
            "packet_digest": self.packet_digest,
            "query_digest": self.query_digest,
            "evaluator_algorithm_digest": self.evaluator_algorithm_digest,
            "scenarios": [item.to_data() for item in self.scenarios],
            "disposition": self.disposition.value,
        }

    def digest(self) -> str:
        return canonical_digest(self.to_data())


def _conjunction_disposition(
    clauses: Sequence[ClauseEvaluation],
) -> Disposition:
    dispositions = tuple(item.disposition for item in clauses)
    if Disposition.CERTIFIED_ABSENT in dispositions:
        return Disposition.CERTIFIED_ABSENT
    if Disposition.ERROR in dispositions:
        return Disposition.ERROR
    if Disposition.INDETERMINATE in dispositions:
        return Disposition.INDETERMINATE
    return Disposition.PRESENT


def _existential_disposition(
    bindings: Sequence[BindingEvaluation],
    role_domain_dispositions: Sequence[Disposition] = (),
) -> Disposition:
    if any(item.disposition is Disposition.PRESENT for item in bindings):
        return Disposition.PRESENT
    if any(item.disposition is Disposition.ERROR for item in bindings) or any(
        item is Disposition.ERROR for item in role_domain_dispositions
    ):
        return Disposition.ERROR
    if any(item.disposition is Disposition.INDETERMINATE for item in bindings) or any(
        item is Disposition.INDETERMINATE for item in role_domain_dispositions
    ):
        return Disposition.INDETERMINATE
    return Disposition.CERTIFIED_ABSENT


def _scenario_consensus(dispositions: tuple[Disposition, ...]) -> Disposition:
    if all(item is Disposition.PRESENT for item in dispositions):
        return Disposition.PRESENT
    if all(item is Disposition.CERTIFIED_ABSENT for item in dispositions):
        return Disposition.CERTIFIED_ABSENT
    if Disposition.ERROR in dispositions:
        return Disposition.ERROR
    return Disposition.INDETERMINATE


def _side_count(
    clause: SideCountClause, loop: LoopGeometryWitness
) -> ClauseEvaluation:
    interval = loop.polygon.side_count
    if loop.polygon.disposition is Disposition.INDETERMINATE:
        interval_detail = (
            "unavailable"
            if interval is None
            else f"[{interval.lower},{interval.upper}]"
        )
        return ClauseEvaluation(
            clause.clause_id,
            Disposition.INDETERMINATE,
            (
                "side-count observer is indeterminate: "
                f"{loop.polygon.reason_code}; interval={interval_detail}"
            ),
        )
    if interval is None:
        return ClauseEvaluation(
            clause.clause_id,
            Disposition.ERROR,
            "present side-count observer omitted its required interval",
        )
    if clause.count < interval.lower or clause.count > interval.upper:
        return ClauseEvaluation(
            clause.clause_id,
            Disposition.CERTIFIED_ABSENT,
            f"target {clause.count} outside frozen interval [{interval.lower},{interval.upper}]",
        )
    if interval.exact and loop.polygon.disposition is Disposition.PRESENT:
        return ClauseEvaluation(
            clause.clause_id,
            Disposition.PRESENT,
            f"exact frozen side count {clause.count}",
        )
    return ClauseEvaluation(
        clause.clause_id,
        Disposition.INDETERMINATE,
        f"target {clause.count} lies in non-point side interval",
    )


def _area_ratio(
    clause: AreaRatioClause,
    numerator: LoopGeometryWitness,
    denominator: LoopGeometryWitness,
) -> ClauseEvaluation:
    left = numerator.area_pixels * clause.ratio.denominator
    right = denominator.area_pixels * clause.ratio.numerator
    if left <= right:
        disposition = Disposition.PRESENT
        relation = "<="
    else:
        disposition = Disposition.CERTIFIED_ABSENT
        relation = ">"
    return ClauseEvaluation(
        clause.clause_id,
        disposition,
        f"exact integer cross-product {left} {relation} {right}",
    )


def _edge_obliqueness(
    clause: EdgeObliquenessClause, loop: LoopGeometryWitness
) -> ClauseEvaluation:
    interval = loop.edge_obliqueness.minimum_millidegrees
    if (
        loop.polygon.disposition is Disposition.INDETERMINATE
        or loop.edge_obliqueness.disposition is Disposition.INDETERMINATE
    ):
        interval_detail = (
            "unavailable"
            if interval is None
            else f"[{interval.lower},{interval.upper}]"
        )
        return ClauseEvaluation(
            clause.clause_id,
            Disposition.INDETERMINATE,
            (
                "edge-obliqueness observer is indeterminate: "
                f"{loop.edge_obliqueness.reason_code}; interval={interval_detail}"
            ),
        )
    if interval is None:
        return ClauseEvaluation(
            clause.clause_id,
            Disposition.ERROR,
            "present edge-obliqueness observer omitted its required interval",
        )
    if interval.lower >= clause.threshold_millidegrees and (
        loop.edge_obliqueness.disposition is Disposition.PRESENT
    ):
        return ClauseEvaluation(
            clause.clause_id,
            Disposition.PRESENT,
            f"lower bound {interval.lower} >= {clause.threshold_millidegrees}",
        )
    if interval.upper < clause.threshold_millidegrees:
        return ClauseEvaluation(
            clause.clause_id,
            Disposition.CERTIFIED_ABSENT,
            f"upper bound {interval.upper} < {clause.threshold_millidegrees}",
        )
    return ClauseEvaluation(
        clause.clause_id,
        Disposition.INDETERMINATE,
        "obliqueness interval overlaps the threshold",
    )


def _point_contact(
    clause: PointContactClause,
    first: LoopGeometryWitness,
    second: LoopGeometryWitness,
    contacts: Mapping[tuple[str, str], PairContactObservation],
) -> ClauseEvaluation:
    loop_ids = tuple(sorted((first.loop_id, second.loop_id)))
    observation = contacts.get(loop_ids)
    if observation is None:
        return ClauseEvaluation(
            clause.clause_id,
            Disposition.ERROR,
            "exhaustive pair-contact observation is missing",
        )
    detail = (
        f"{observation.contact_kind.value}: {observation.reason_code}; "
        f"gap_ppm_upper={observation.normalized_gap_ppm_upper}; "
        f"spread_ppm_upper={observation.interface_spread_ppm_upper}"
    )
    return ClauseEvaluation(clause.clause_id, observation.disposition, detail)


def _evaluate_binding(
    query: RelationalVisualQuery,
    first: LoopGeometryWitness,
    second: LoopGeometryWitness,
    contacts: Mapping[tuple[str, str], PairContactObservation],
) -> BindingEvaluation:
    values = {"role-00": first, "role-01": second}
    evaluations: list[ClauseEvaluation] = []
    for clause in query.clauses:
        if isinstance(clause, SideCountClause):
            evaluations.append(_side_count(clause, values[clause.role_id]))
        elif isinstance(clause, AreaRatioClause):
            evaluations.append(
                _area_ratio(
                    clause,
                    values[clause.numerator_role_id],
                    values[clause.denominator_role_id],
                )
            )
        elif isinstance(clause, EdgeObliquenessClause):
            evaluations.append(_edge_obliqueness(clause, values[clause.role_id]))
        else:
            evaluations.append(
                _point_contact(
                    clause,
                    values[clause.first_role_id],
                    values[clause.second_role_id],
                    contacts,
                )
            )
    clause_tuple = tuple(evaluations)
    return BindingEvaluation(
        bindings=(("role-00", first.loop_id), ("role-01", second.loop_id)),
        clauses=clause_tuple,
        disposition=_conjunction_disposition(clause_tuple),
    )


def evaluate_relational_query(
    query: RelationalVisualQuery, packet: LoopScenePacket
) -> RelationalQueryResult:
    if not isinstance(query, RelationalVisualQuery):
        raise TypeError("query must be a RelationalVisualQuery")
    if not isinstance(packet, LoopScenePacket):
        raise TypeError("packet must be a LoopScenePacket")
    scenarios: list[ScenarioQueryResult] = []
    for scenario in packet.scenarios:
        contacts = {item.loop_ids: item for item in scenario.contacts}
        role_domain = tuple(
            (loop.loop_id, loop.substantiveness.disposition)
            for loop in scenario.loops
        )
        eligible = tuple(
            loop
            for loop in scenario.loops
            if loop.substantiveness.disposition is Disposition.PRESENT
        )
        bindings = tuple(
            _evaluate_binding(query, first, second, contacts)
            for first in eligible
            for second in eligible
            if first.loop_id != second.loop_id
        )
        disposition = _existential_disposition(
            bindings, tuple(item[1] for item in role_domain)
        )
        reason = {
            Disposition.PRESENT: "binding_witness",
            Disposition.CERTIFIED_ABSENT: "all_bindings_blocked",
            Disposition.INDETERMINATE: "unresolved_binding",
            Disposition.ERROR: "binding_error",
        }[disposition]
        scenarios.append(
            ScenarioQueryResult(
                scenario_id=scenario.scenario_id,
                role_domain=role_domain,
                bindings=bindings,
                disposition=disposition,
                reason_code=reason,
            )
        )
    scenario_tuple = tuple(scenarios)
    return RelationalQueryResult(
        panel_digest=packet.panel_digest,
        packet_digest=packet.digest(),
        query_digest=query.digest(),
        evaluator_algorithm_digest=relational_query_algorithm_digest(),
        scenarios=scenario_tuple,
        disposition=_scenario_consensus(
            tuple(item.disposition for item in scenario_tuple)
        ),
    )


def verify_relational_query_result(
    result: RelationalQueryResult,
    query: RelationalVisualQuery,
    packet: LoopScenePacket,
) -> RelationalQueryResult:
    if not isinstance(result, RelationalQueryResult):
        raise TypeError("result must be a RelationalQueryResult")
    replay = evaluate_relational_query(query, packet)
    if replay != result:
        raise ValueError("relational query result differs from model-free replay")
    return result


def enumerate_factorized_shape_ratio_queries() -> tuple[RelationalVisualQuery, ...]:
    """Return the complete finite v3 positive-conjunction search space."""

    return tuple(
        RelationalVisualQuery.factorized_shape_ratio(
            numerator_side_count=numerator,
            denominator_side_count=denominator,
            ratio=Rational(*ratio),
            denominator_obliqueness_millidegrees=obliqueness,
            require_point_contact=require_contact,
        )
        for numerator in ALLOWED_SIDE_COUNTS
        for denominator in ALLOWED_SIDE_COUNTS
        for ratio in ALLOWED_AREA_RATIOS
        for obliqueness in (
            None,
            *ALLOWED_OBLIQUENESS_THRESHOLDS_MILLIDEGREES,
        )
        for require_contact in (False, True)
    )


def exact_support_separators(
    positive_packets: Sequence[LoopScenePacket],
    negative_packets: Sequence[LoopScenePacket],
) -> tuple[RelationalVisualQuery, ...]:
    """Support-only finite search with no polarity flip or negation rescue."""

    if not positive_packets or not negative_packets:
        raise ValueError("separator search requires both support classes")
    candidates: list[RelationalVisualQuery] = []
    for query in enumerate_factorized_shape_ratio_queries():
        if all(
            evaluate_relational_query(query, packet).disposition
            is Disposition.PRESENT
            for packet in positive_packets
        ) and all(
            evaluate_relational_query(query, packet).disposition
            is Disposition.CERTIFIED_ABSENT
            for packet in negative_packets
        ):
            candidates.append(query)
    return tuple(candidates)


__all__ = [
    "ALLOWED_AREA_RATIOS",
    "ALLOWED_OBLIQUENESS_THRESHOLDS_MILLIDEGREES",
    "ALLOWED_SIDE_COUNTS",
    "AreaRatioClause",
    "BindingEvaluation",
    "ClauseEvaluation",
    "EdgeObliquenessClause",
    "PointContactClause",
    "PredicateID",
    "Rational",
    "RelationalQueryResult",
    "RelationalVisualQuery",
    "RoleSpec",
    "ScenarioQueryResult",
    "SideCountClause",
    "enumerate_factorized_shape_ratio_queries",
    "evaluate_relational_query",
    "exact_support_separators",
    "relational_query_algorithm_digest",
    "relational_query_source_digest",
    "verify_relational_query_result",
]

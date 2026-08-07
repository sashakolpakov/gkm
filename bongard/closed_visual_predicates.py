"""Pure-Python closed predicate union and support expressibility oracle.

This module is deliberately backend-neutral: canonical Python data and integer
comparisons are authoritative, and no theorem prover participates in artifact
identity or decisions.  The positive language is a tagged sum of:

* the existing same-binding :class:`RelationalVisualQuery`;
* a conjunction of one to three registered positive direct count atoms; and
* one registered bilateral coverage or reflection-residual threshold atom.

There is no ``Not``, polarity field, arbitrary callback, or proposer-supplied
code.  Reflection mismatch is the directly measured positive residual
``1_000_000 - reflected-ink coverage``.

Deliberate v1 boundary: this is a tagged sum, not a conjunction across the
three predicate families.  Direct-count and bilateral atoms are panel-global,
not bound to loop/object roles.  Cross-family and object-bound conjunctions
require a future closed schema rather than an implicit callback here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import hashlib
from itertools import combinations, product
from pathlib import Path
import re
from typing import Any, Mapping, Sequence, TypeAlias

import bongard.relational_visual_query as relational_visual_query
from bongard.canonical import canonical_digest, canonical_json
from bongard.composite_visual_packet import (
    BilateralSymmetryScenarioWitness,
    ExactPanelWitnessPacket,
    bilateral_symmetry_witness_extractor_digest,
    exact_panel_witness_extractor_digest,
)
from bongard.evidence import Disposition, Evidence, Provenance
from bongard.relational_visual_query import (
    PointContactClause,
    RelationalVisualQuery,
    enumerate_factorized_shape_ratio_queries,
    evaluate_relational_query,
    relational_query_algorithm_digest,
)
from bongard.scenario_semantics import (
    JOINT_SCENARIO_SEMANTICS_VERSION,
    evaluate_joint_scenario_conjunction,
)
from bongard.typed_visual_proposal import (
    MAX_DETERMINISTIC_ATOMS,
    TypedDeterministicAtom,
)
from bongard.visual_predicate_catalog import (
    DIRECT_VISUAL_ATOM_CATALOG,
    direct_visual_catalog_digest,
    evaluate_direct_atom_by_scenario,
)
from bongard.visual_witnesses import VISUAL_WITNESS_SCENARIO_IDS


CLOSED_PREDICATE_SCHEMA = "gkm.bongard-closed-panel-predicate.v1"
DIRECT_COUNT_PREDICATE_SCHEMA = "gkm.bongard-direct-count-predicate.v1"
SYMMETRY_PREDICATE_SCHEMA = "gkm.bongard-symmetry-threshold-predicate.v1"
CLOSED_RESULT_SCHEMA = "gkm.bongard-closed-panel-predicate-result.v1"
FROZEN_LIBRARY_SCHEMA = "gkm.bongard-frozen-closed-predicate-library.v1"
EXPRESSIBILITY_RESULT_SCHEMA = "gkm.bongard-support-expressibility-result.v1"
CLOSED_LANGUAGE_ID = "bongard.closed-positive-panel-language/v1"
EVALUATOR_ALGORITHM_ID = "bongard.closed-positive-panel-evaluator/v1"
LIBRARY_ALGORITHM_ID = "bongard.proposer-reachable-closed-positive-library/v2"
ORACLE_ALGORITHM_ID = "bongard.support-only-expressibility-oracle/v1"
SYMMETRY_THRESHOLDS_PPM = (
    250_000,
    500_000,
    600_000,
    700_000,
    750_000,
    800_000,
    850_000,
    900_000,
    950_000,
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IMPORT_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_DIRECT_CATALOG_DIGEST = direct_visual_catalog_digest()
_DIRECT_ALLOWED_SELECTIONS = {
    spec.catalog_key: frozenset(
        canonical_json(
            {
                "catalog_key": spec.catalog_key,
                "comparison": option.comparison,
                "arguments": dict(option.arguments),
            }
        )
        for option in spec.allowed_options
    )
    for spec in DIRECT_VISUAL_ATOM_CATALOG.atoms
}


def _exact_fields(
    data: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if not isinstance(data, Mapping) or set(data) != expected:
        raise ValueError(f"{label} fields differ from the static schema")


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase sha256")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty stripped text")
    return value


def _source_digest() -> str:
    # This binds the exact source loaded by this Python process.  A fresh
    # process necessarily recomputes it; callers cannot supply the value.
    return _IMPORT_SOURCE_DIGEST


def closed_visual_predicate_source_digest() -> str:
    return _source_digest()


@lru_cache(maxsize=1)
def closed_visual_predicate_evaluator_digest() -> str:
    return canonical_digest(
        {
            "algorithm_id": EVALUATOR_ALGORITHM_ID,
            "source_digest": _source_digest(),
            "language_id": CLOSED_LANGUAGE_ID,
            "relational_query_algorithm_digest": relational_query_algorithm_digest(),
            "direct_visual_catalog_digest": _DIRECT_CATALOG_DIGEST,
            "bilateral_extractor_digest": (
                bilateral_symmetry_witness_extractor_digest()
            ),
            "exact_panel_extractor_digest": exact_panel_witness_extractor_digest(),
            "joint_scenario_semantics": JOINT_SCENARIO_SEMANTICS_VERSION,
            "numeric_domain": "integers with closed intervals",
            "python_is_authority": True,
        }
    )


class ClosedPredicateKind(str, Enum):
    RELATIONAL = "relational"
    DIRECT_COUNTS = "direct_counts"
    SYMMETRY = "symmetry"


class SymmetryMetric(str, Enum):
    COVERAGE_AT_LEAST = "symmetry.coverage_at_least"
    REFLECTION_MISMATCH_AT_LEAST = "symmetry.reflection_mismatch_at_least"


@dataclass(frozen=True, slots=True)
class DirectCountPredicate:
    """A complete conjunction of one to three registered positive count atoms."""

    atoms: tuple[TypedDeterministicAtom, ...]
    catalog_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.atoms, tuple) or not 1 <= len(self.atoms) <= (
            MAX_DETERMINISTIC_ATOMS
        ):
            raise ValueError(
                f"direct predicate requires 1..{MAX_DETERMINISTIC_ATOMS} atoms"
            )
        _digest(self.catalog_digest, "direct predicate catalog_digest")
        if self.catalog_digest != _DIRECT_CATALOG_DIGEST:
            raise ValueError("direct predicate belongs to a different catalog")
        expected_ids = tuple(f"atom-{index:02d}" for index in range(len(self.atoms)))
        if tuple(item.atom_id for item in self.atoms) != expected_ids:
            raise ValueError("direct predicate atom IDs must be consecutive")
        selections: list[bytes] = []
        keys: list[str] = []
        for atom in self.atoms:
            if not isinstance(atom, TypedDeterministicAtom):
                raise TypeError("direct predicate atoms must be typed selections")
            selected = canonical_json(
                {
                    "catalog_key": atom.catalog_key,
                    "comparison": atom.comparison,
                    "arguments": dict(atom.arguments),
                }
            )
            if selected not in _DIRECT_ALLOWED_SELECTIONS.get(
                atom.catalog_key, frozenset()
            ):
                raise ValueError("direct predicate atom lies outside the catalog")
            keys.append(atom.catalog_key)
            selections.append(
                canonical_json(
                    {
                        "catalog_key": atom.catalog_key,
                        "comparison": atom.comparison,
                        "arguments": dict(atom.arguments),
                    }
                )
            )
        if len(selections) != len(set(selections)):
            raise ValueError("direct predicate repeats a catalog option")
        # Two exact targets for one count cannot both be present.  Excluding
        # that contradiction loses no possible forward separator and keeps the
        # exhaustive finite library materially smaller.
        if len(keys) != len(set(keys)):
            raise ValueError("direct predicate may use each count capability once")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": DIRECT_COUNT_PREDICATE_SCHEMA,
            "catalog_digest": self.catalog_digest,
            "atoms": [item.to_data() for item in self.atoms],
            "formula": {
                "kind": "all",
                "atom_ids": [item.atom_id for item in self.atoms],
            },
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "DirectCountPredicate":
        _exact_fields(
            data,
            frozenset({"schema", "catalog_digest", "atoms", "formula"}),
            "direct count predicate",
        )
        if data["schema"] != DIRECT_COUNT_PREDICATE_SCHEMA:
            raise ValueError("unsupported direct count predicate")
        raw_atoms = data["atoms"]
        if not isinstance(raw_atoms, list) or not 1 <= len(raw_atoms) <= (
            MAX_DETERMINISTIC_ATOMS
        ):
            raise TypeError("direct atoms must be a list of length 1..3")
        atoms: list[TypedDeterministicAtom] = []
        for index, raw in enumerate(raw_atoms):
            if not isinstance(raw, Mapping):
                raise TypeError("direct atom must be a JSON object")
            _exact_fields(
                raw,
                frozenset({"atom_id", "catalog_key", "comparison", "arguments"}),
                f"direct atom {index}",
            )
            if raw["atom_id"] != f"atom-{index:02d}":
                raise ValueError("direct atom ID differs from its position")
            arguments = raw["arguments"]
            if not isinstance(arguments, Mapping):
                raise TypeError("direct atom arguments must be an object")
            spec = DIRECT_VISUAL_ATOM_CATALOG.get(raw["catalog_key"])
            comparison, canonical_arguments = spec.canonical_selection(
                raw["comparison"], arguments, raw["atom_id"]
            )
            atoms.append(
                TypedDeterministicAtom(
                    atom_id=raw["atom_id"],
                    catalog_key=raw["catalog_key"],
                    comparison=comparison,
                    arguments=canonical_arguments,
                )
            )
        formula = data["formula"]
        if not isinstance(formula, Mapping):
            raise TypeError("direct formula must be an object")
        _exact_fields(formula, frozenset({"kind", "atom_ids"}), "direct formula")
        expected_ids = [item.atom_id for item in atoms]
        if formula["kind"] != "all" or formula["atom_ids"] != expected_ids:
            raise ValueError("direct formula must reference every atom exactly once")
        result = cls(tuple(atoms), data["catalog_digest"])
        if result.to_data() != dict(data):
            raise ValueError("direct predicate is not canonically represented")
        return result


@dataclass(frozen=True, order=True, slots=True)
class SymmetryThresholdPredicate:
    """One registered positive integer threshold over coverage or residual."""

    metric: SymmetryMetric
    threshold_ppm: int

    def __post_init__(self) -> None:
        if not isinstance(self.metric, SymmetryMetric):
            raise TypeError("symmetry metric must be a registered enum value")
        if isinstance(self.threshold_ppm, bool) or not isinstance(
            self.threshold_ppm, int
        ):
            raise TypeError("symmetry threshold must be an integer")
        if self.threshold_ppm not in SYMMETRY_THRESHOLDS_PPM:
            raise ValueError("symmetry threshold lies outside the frozen grid")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": SYMMETRY_PREDICATE_SCHEMA,
            "metric": self.metric.value,
            "threshold_ppm": self.threshold_ppm,
            "comparison": "at_least",
            "unit": "parts_per_million",
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "SymmetryThresholdPredicate":
        _exact_fields(
            data,
            frozenset({"schema", "metric", "threshold_ppm", "comparison", "unit"}),
            "symmetry predicate",
        )
        if data["schema"] != SYMMETRY_PREDICATE_SCHEMA:
            raise ValueError("unsupported symmetry predicate")
        if data["comparison"] != "at_least" or data["unit"] != (
            "parts_per_million"
        ):
            raise ValueError("symmetry predicate orientation or unit drifted")
        result = cls(SymmetryMetric(data["metric"]), data["threshold_ppm"])
        if result.to_data() != dict(data):
            raise ValueError("symmetry predicate is not canonically represented")
        return result


PredicatePayload: TypeAlias = (
    RelationalVisualQuery | DirectCountPredicate | SymmetryThresholdPredicate
)


@lru_cache(maxsize=100_000)
def _closed_panel_predicate_digest(predicate: object) -> str:
    # Closed predicate values are immutable and hashable. The complete oracle
    # repeatedly orders and verifies the same 65,678 values, so caching avoids
    # rebuilding identical canonical JSON several times.
    return canonical_digest(predicate.to_data())  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class ClosedPanelPredicate:
    """Tagged, serializable sum with no negative-orientation constructor."""

    kind: ClosedPredicateKind
    payload: PredicatePayload

    def __post_init__(self) -> None:
        expected = {
            ClosedPredicateKind.RELATIONAL: RelationalVisualQuery,
            ClosedPredicateKind.DIRECT_COUNTS: DirectCountPredicate,
            ClosedPredicateKind.SYMMETRY: SymmetryThresholdPredicate,
        }
        if not isinstance(self.kind, ClosedPredicateKind):
            raise TypeError("closed predicate kind must be an enum value")
        if not isinstance(self.payload, expected[self.kind]):
            raise TypeError("closed predicate payload differs from its tag")

    @classmethod
    def relational(cls, query: RelationalVisualQuery) -> "ClosedPanelPredicate":
        return cls(ClosedPredicateKind.RELATIONAL, query)

    @classmethod
    def direct(cls, predicate: DirectCountPredicate) -> "ClosedPanelPredicate":
        return cls(ClosedPredicateKind.DIRECT_COUNTS, predicate)

    @classmethod
    def symmetry(
        cls, predicate: SymmetryThresholdPredicate
    ) -> "ClosedPanelPredicate":
        return cls(ClosedPredicateKind.SYMMETRY, predicate)

    def to_data(self) -> dict[str, object]:
        return {
            "schema": CLOSED_PREDICATE_SCHEMA,
            "language_id": CLOSED_LANGUAGE_ID,
            "kind": self.kind.value,
            "payload": self.payload.to_data(),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ClosedPanelPredicate":
        _exact_fields(
            data,
            frozenset({"schema", "language_id", "kind", "payload"}),
            "closed panel predicate",
        )
        if (
            data["schema"] != CLOSED_PREDICATE_SCHEMA
            or data["language_id"] != CLOSED_LANGUAGE_ID
        ):
            raise ValueError("unsupported closed panel predicate")
        payload = data["payload"]
        if not isinstance(payload, Mapping):
            raise TypeError("closed predicate payload must be an object")
        kind = ClosedPredicateKind(data["kind"])
        if kind is ClosedPredicateKind.RELATIONAL:
            parsed: PredicatePayload = RelationalVisualQuery.from_data(payload)
        elif kind is ClosedPredicateKind.DIRECT_COUNTS:
            parsed = DirectCountPredicate.from_data(payload)
        else:
            parsed = SymmetryThresholdPredicate.from_data(payload)
        result = cls(kind, parsed)
        if result.to_data() != dict(data):
            raise ValueError("closed predicate is not canonically represented")
        return result

    @property
    def digest(self) -> str:
        return _closed_panel_predicate_digest(self)


@dataclass(frozen=True, order=True, slots=True)
class ClosedScenarioResult:
    scenario_id: str
    disposition: Disposition
    detail: str

    def __post_init__(self) -> None:
        if self.scenario_id not in VISUAL_WITNESS_SCENARIO_IDS:
            raise ValueError("closed result contains an unknown scenario")
        if not isinstance(self.disposition, Disposition):
            raise TypeError("closed scenario disposition must be typed")
        _text(self.detail, "closed scenario detail")

    def to_data(self) -> dict[str, str]:
        return {
            "scenario_id": self.scenario_id,
            "disposition": self.disposition.value,
            "detail": self.detail,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ClosedScenarioResult":
        _exact_fields(
            data,
            frozenset({"scenario_id", "disposition", "detail"}),
            "closed scenario result",
        )
        return cls(
            data["scenario_id"], Disposition(data["disposition"]), data["detail"]
        )


def _scenario_consensus(dispositions: Sequence[Disposition]) -> Disposition:
    values = tuple(dispositions)
    if values and all(item is Disposition.PRESENT for item in values):
        return Disposition.PRESENT
    if values and all(item is Disposition.CERTIFIED_ABSENT for item in values):
        return Disposition.CERTIFIED_ABSENT
    if Disposition.ERROR in values:
        return Disposition.ERROR
    return Disposition.INDETERMINATE


@dataclass(frozen=True, slots=True)
class ClosedPredicateResult:
    panel_digest: str
    packet_digest: str
    predicate_digest: str
    evaluator_digest: str
    scenarios: tuple[ClosedScenarioResult, ...]
    disposition: Disposition

    def __post_init__(self) -> None:
        _digest(self.panel_digest, "closed result panel_digest")
        _digest(self.packet_digest, "closed result packet_digest")
        _digest(self.predicate_digest, "closed result predicate_digest")
        _digest(self.evaluator_digest, "closed result evaluator_digest")
        if self.evaluator_digest != closed_visual_predicate_evaluator_digest():
            raise ValueError("closed result evaluator identity drifted")
        if not isinstance(self.scenarios, tuple) or tuple(
            item.scenario_id for item in self.scenarios
        ) != VISUAL_WITNESS_SCENARIO_IDS:
            raise ValueError("closed result must retain every canonical scenario")
        if any(not isinstance(item, ClosedScenarioResult) for item in self.scenarios):
            raise TypeError("closed result scenarios must be typed")
        expected = _scenario_consensus(
            tuple(item.disposition for item in self.scenarios)
        )
        if self.disposition is not expected:
            raise ValueError("closed result disagrees with scenario consensus")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": CLOSED_RESULT_SCHEMA,
            "panel_digest": self.panel_digest,
            "packet_digest": self.packet_digest,
            "predicate_digest": self.predicate_digest,
            "evaluator_digest": self.evaluator_digest,
            "scenarios": [item.to_data() for item in self.scenarios],
            "disposition": self.disposition.value,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ClosedPredicateResult":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "panel_digest",
                    "packet_digest",
                    "predicate_digest",
                    "evaluator_digest",
                    "scenarios",
                    "disposition",
                }
            ),
            "closed predicate result",
        )
        if data["schema"] != CLOSED_RESULT_SCHEMA:
            raise ValueError("unsupported closed predicate result")
        scenarios = data["scenarios"]
        if not isinstance(scenarios, list) or any(
            not isinstance(item, Mapping) for item in scenarios
        ):
            raise TypeError("closed result scenarios must be an object list")
        result = cls(
            panel_digest=data["panel_digest"],
            packet_digest=data["packet_digest"],
            predicate_digest=data["predicate_digest"],
            evaluator_digest=data["evaluator_digest"],
            scenarios=tuple(ClosedScenarioResult.from_data(item) for item in scenarios),
            disposition=Disposition(data["disposition"]),
        )
        if result.to_data() != dict(data):
            raise ValueError("closed predicate result is not canonical")
        return result

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_data())


def _symmetry_evidence(
    predicate: SymmetryThresholdPredicate,
    witness: BilateralSymmetryScenarioWitness,
) -> Evidence[bool]:
    provenance = Provenance(
        producer="bongard.closed_visual_predicates",
        version="1",
        method=predicate.metric.value,
        input_digests=(witness.provenance_digest,),
        artifact_digest=closed_visual_predicate_evaluator_digest(),
        details=tuple(
            sorted(
                (
                    ("scenario_id", witness.scenario_id),
                    ("threshold_ppm", str(predicate.threshold_ppm)),
                )
            )
        ),
    )
    if witness.disposition is Disposition.ERROR:
        return Evidence.error(
            provenance,
            witness.error_type or "BilateralWitnessError",
            witness.reason or "bilateral witness failed",
        )
    if witness.disposition is Disposition.INDETERMINATE:
        return Evidence.indeterminate(
            provenance, witness.reason or "bilateral witness is indeterminate"
        )
    if witness.disposition is Disposition.CERTIFIED_ABSENT:
        return Evidence.certified_absent(
            provenance,
            witness.certificate
            or "bilateral measurement has no foreground carrier",
        )
    interval = (
        witness.coverage_ppm
        if predicate.metric is SymmetryMetric.COVERAGE_AT_LEAST
        else witness.mismatch_ppm
    )
    if interval is None:
        return Evidence.error(
            provenance,
            "MissingSymmetryInterval",
            "present bilateral witness omitted its required interval",
        )
    if interval.lower >= predicate.threshold_ppm:
        return Evidence.present(True, provenance)
    if interval.upper < predicate.threshold_ppm:
        return Evidence.certified_absent(
            provenance,
            f"interval upper {interval.upper} is below {predicate.threshold_ppm}",
        )
    return Evidence.indeterminate(
        provenance,
        f"interval [{interval.lower},{interval.upper}] straddles "
        f"{predicate.threshold_ppm}",
    )


DirectAtomCache: TypeAlias = dict[
    tuple[str, str], Mapping[str, Evidence[bool]]
]
DirectAtomDigestCache: TypeAlias = dict[TypedDeterministicAtom, str]


def _direct_atom_digest(
    atom: TypedDeterministicAtom,
    cache: DirectAtomDigestCache | None,
) -> str:
    if cache is None:
        return canonical_digest(atom.to_data())
    digest = cache.get(atom)
    if digest is None:
        digest = canonical_digest(atom.to_data())
        cache[atom] = digest
    return digest


def _evaluate_direct(
    predicate: DirectCountPredicate,
    packet: ExactPanelWitnessPacket,
    cache: DirectAtomCache | None,
    packet_digest: str,
    atom_digest_cache: DirectAtomDigestCache | None = None,
) -> tuple[ClosedScenarioResult, ...]:
    evidence_by_scenario: dict[str, dict[str, Evidence[bool]]] = {
        scenario_id: {} for scenario_id in VISUAL_WITNESS_SCENARIO_IDS
    }
    for atom in predicate.atoms:
        cache_key = (
            packet_digest,
            _direct_atom_digest(atom, atom_digest_cache),
        )
        evidence = None if cache is None else cache.get(cache_key)
        if evidence is None:
            evidence = evaluate_direct_atom_by_scenario(packet.visual_bundle, atom)
            if cache is not None:
                cache[cache_key] = evidence
        for scenario_id in VISUAL_WITNESS_SCENARIO_IDS:
            evidence_by_scenario[scenario_id][atom.atom_id] = evidence[scenario_id]
    joint = evaluate_joint_scenario_conjunction(evidence_by_scenario)
    return tuple(
        ClosedScenarioResult(
            scenario_id,
            disposition,
            "complete direct conjunction evaluated inside this scenario",
        )
        for scenario_id, disposition in joint.scenario_dispositions
    )


ClosedDispositionOutcome: TypeAlias = tuple[
    tuple[Disposition, ...], Disposition
]


def _direct_scenario_conjunction_disposition(
    dispositions: Sequence[Disposition],
) -> Disposition:
    """Disposition-only equivalent of the archived direct conjunction."""

    values = tuple(dispositions)
    if Disposition.ERROR in values:
        return Disposition.ERROR
    if Disposition.CERTIFIED_ABSENT in values:
        return Disposition.CERTIFIED_ABSENT
    if Disposition.INDETERMINATE in values:
        return Disposition.INDETERMINATE
    return Disposition.PRESENT


def _evaluate_direct_dispositions(
    predicate: DirectCountPredicate,
    packet: ExactPanelWitnessPacket,
    cache: DirectAtomCache,
    packet_digest: str,
    atom_digest_cache: DirectAtomDigestCache,
) -> tuple[Disposition, ...]:
    """Evaluate direct atoms without allocating result/provenance wrappers."""

    dispositions_by_scenario: dict[str, list[Disposition]] = {
        scenario_id: [] for scenario_id in VISUAL_WITNESS_SCENARIO_IDS
    }
    for atom in predicate.atoms:
        cache_key = (
            packet_digest,
            _direct_atom_digest(atom, atom_digest_cache),
        )
        evidence = cache.get(cache_key)
        if evidence is None:
            evidence = evaluate_direct_atom_by_scenario(packet.visual_bundle, atom)
            cache[cache_key] = evidence
        for scenario_id in VISUAL_WITNESS_SCENARIO_IDS:
            dispositions_by_scenario[scenario_id].append(
                evidence[scenario_id].disposition
            )
    return tuple(
        _direct_scenario_conjunction_disposition(
            dispositions_by_scenario[scenario_id]
        )
        for scenario_id in VISUAL_WITNESS_SCENARIO_IDS
    )


def _symmetry_disposition(
    predicate: SymmetryThresholdPredicate,
    witness: BilateralSymmetryScenarioWitness,
) -> Disposition:
    """Disposition-only equivalent of :func:`_symmetry_evidence`."""

    if witness.disposition is Disposition.ERROR:
        return Disposition.ERROR
    if witness.disposition is Disposition.INDETERMINATE:
        return Disposition.INDETERMINATE
    if witness.disposition is Disposition.CERTIFIED_ABSENT:
        return Disposition.CERTIFIED_ABSENT
    interval = (
        witness.coverage_ppm
        if predicate.metric is SymmetryMetric.COVERAGE_AT_LEAST
        else witness.mismatch_ppm
    )
    if interval is None:
        return Disposition.ERROR
    if interval.lower >= predicate.threshold_ppm:
        return Disposition.PRESENT
    if interval.upper < predicate.threshold_ppm:
        return Disposition.CERTIFIED_ABSENT
    return Disposition.INDETERMINATE


def _evaluate_relational_dispositions(
    query: RelationalVisualQuery,
    packet: ExactPanelWitnessPacket,
) -> tuple[Disposition, ...]:
    """Run relational semantics without content-addressed result objects."""

    dispositions: list[Disposition] = []
    for scenario in packet.loop_scene.scenarios:
        contacts = {item.loop_ids: item for item in scenario.contacts}
        role_domain_dispositions = tuple(
            loop.substantiveness.disposition for loop in scenario.loops
        )
        eligible = tuple(
            loop
            for loop in scenario.loops
            if loop.substantiveness.disposition is Disposition.PRESENT
        )
        bindings = tuple(
            relational_visual_query._evaluate_binding(
                query, first, second, contacts
            )
            for first in eligible
            for second in eligible
            if first.loop_id != second.loop_id
        )
        dispositions.append(
            relational_visual_query._existential_disposition(
                bindings, role_domain_dispositions
            )
        )
    return tuple(dispositions)


def _evaluate_closed_dispositions(
    predicate: ClosedPanelPredicate,
    packet: ExactPanelWitnessPacket,
    *,
    direct_atom_cache: DirectAtomCache,
    direct_atom_digest_cache: DirectAtomDigestCache,
    precomputed_packet_digest: str,
) -> ClosedDispositionOutcome:
    """Oracle-only evaluator for already validated, content-addressed inputs.

    Only scenario and panel dispositions are returned.  In particular, this
    path never constructs ``ClosedScenarioResult`` or ``ClosedPredicateResult``
    and never asks for the source-bound evaluator digest.
    """

    if predicate.kind is ClosedPredicateKind.RELATIONAL:
        assert isinstance(predicate.payload, RelationalVisualQuery)
        scenarios = _evaluate_relational_dispositions(predicate.payload, packet)
    elif predicate.kind is ClosedPredicateKind.DIRECT_COUNTS:
        assert isinstance(predicate.payload, DirectCountPredicate)
        scenarios = _evaluate_direct_dispositions(
            predicate.payload,
            packet,
            direct_atom_cache,
            precomputed_packet_digest,
            direct_atom_digest_cache,
        )
    else:
        assert isinstance(predicate.payload, SymmetryThresholdPredicate)
        scenarios = tuple(
            _symmetry_disposition(predicate.payload, witness)
            for witness in packet.bilateral_symmetry.scenarios
        )
    return scenarios, _scenario_consensus(scenarios)


def _evaluate_closed_predicate(
    predicate: ClosedPanelPredicate,
    packet: ExactPanelWitnessPacket,
    *,
    direct_atom_cache: DirectAtomCache | None,
    direct_atom_digest_cache: DirectAtomDigestCache | None = None,
    packet_is_prevalidated: bool = False,
    precomputed_packet_digest: str | None = None,
) -> ClosedPredicateResult:
    if not isinstance(predicate, ClosedPanelPredicate):
        raise TypeError("predicate must be a ClosedPanelPredicate")
    if not isinstance(packet, ExactPanelWitnessPacket):
        raise TypeError("packet must be an ExactPanelWitnessPacket")
    # A strict in-memory round trip detects forged nested values without opening
    # pixels.  Exact-byte verification remains available in the packet module.
    if not packet_is_prevalidated and (
        ExactPanelWitnessPacket.from_data(packet.to_data()) != packet
    ):
        raise ValueError("exact panel packet is not canonically represented")
    packet_digest = (
        packet.digest()
        if precomputed_packet_digest is None
        else _digest(precomputed_packet_digest, "precomputed packet digest")
    )
    if predicate.kind is ClosedPredicateKind.RELATIONAL:
        assert isinstance(predicate.payload, RelationalVisualQuery)
        relational = evaluate_relational_query(predicate.payload, packet.loop_scene)
        scenarios = tuple(
            ClosedScenarioResult(
                item.scenario_id,
                item.disposition,
                "relational same-binding existential: " + item.reason_code,
            )
            for item in relational.scenarios
        )
    elif predicate.kind is ClosedPredicateKind.DIRECT_COUNTS:
        assert isinstance(predicate.payload, DirectCountPredicate)
        scenarios = _evaluate_direct(
            predicate.payload,
            packet,
            direct_atom_cache,
            packet_digest,
            direct_atom_digest_cache,
        )
    else:
        assert isinstance(predicate.payload, SymmetryThresholdPredicate)
        evidence_by_scenario = {
            item.scenario_id: {
                "atom-00": _symmetry_evidence(predicate.payload, item)
            }
            for item in packet.bilateral_symmetry.scenarios
        }
        joint = evaluate_joint_scenario_conjunction(evidence_by_scenario)
        scenarios = tuple(
            ClosedScenarioResult(
                scenario_id,
                disposition,
                "registered bilateral interval threshold evaluated in this scenario",
            )
            for scenario_id, disposition in joint.scenario_dispositions
        )
    return ClosedPredicateResult(
        panel_digest=packet.panel_digest,
        packet_digest=packet_digest,
        predicate_digest=predicate.digest,
        evaluator_digest=closed_visual_predicate_evaluator_digest(),
        scenarios=scenarios,
        disposition=_scenario_consensus(
            tuple(item.disposition for item in scenarios)
        ),
    )


def evaluate_closed_predicate(
    predicate: ClosedPanelPredicate,
    packet: ExactPanelWitnessPacket,
) -> ClosedPredicateResult:
    """Evaluate one closed positive predicate with correlated-scenario semantics."""

    return _evaluate_closed_predicate(
        predicate,
        packet,
        direct_atom_cache=None,
        packet_is_prevalidated=False,
        precomputed_packet_digest=None,
    )


def verify_closed_predicate_result(
    result: ClosedPredicateResult,
    predicate: ClosedPanelPredicate,
    packet: ExactPanelWitnessPacket,
) -> ClosedPredicateResult:
    if not isinstance(result, ClosedPredicateResult):
        raise TypeError("result must be a ClosedPredicateResult")
    replay = evaluate_closed_predicate(predicate, packet)
    if replay != result:
        raise ValueError("closed predicate result differs from model-free replay")
    return result


def _direct_option_grid() -> tuple[tuple[str, str, tuple[tuple[str, object], ...]], ...]:
    options: list[tuple[str, str, tuple[tuple[str, object], ...]]] = []
    for spec in sorted(
        DIRECT_VISUAL_ATOM_CATALOG.atoms, key=lambda item: item.catalog_key
    ):
        for option in sorted(
            spec.allowed_options, key=lambda item: canonical_json(item.to_data())
        ):
            options.append((spec.catalog_key, option.comparison, option.arguments))
    return tuple(options)


def enumerate_complete_closed_predicates() -> tuple[ClosedPanelPredicate, ...]:
    """Return the complete proposer-reachable union, independent of packets."""

    members: list[ClosedPanelPredicate] = [
        ClosedPanelPredicate.relational(item)
        for item in enumerate_factorized_shape_ratio_queries()
        if not any(isinstance(clause, PointContactClause) for clause in item.clauses)
    ]
    by_key: dict[str, tuple[tuple[str, str, tuple[tuple[str, object], ...]], ...]] = {}
    for option in _direct_option_grid():
        by_key.setdefault(option[0], tuple())
        by_key[option[0]] = (*by_key[option[0]], option)
    keys = tuple(sorted(by_key))
    # Atom values are immutable. Reuse the 240 possible (position, option)
    # objects rather than retaining roughly 190k duplicate Python objects in
    # the materialized oracle library.
    atom_cache: dict[
        tuple[int, tuple[str, str, tuple[tuple[str, object], ...]]],
        TypedDeterministicAtom,
    ] = {}

    def intern_atom(
        index: int,
        selection: tuple[str, str, tuple[tuple[str, object], ...]],
    ) -> TypedDeterministicAtom:
        key = (index, selection)
        atom = atom_cache.get(key)
        if atom is None:
            atom = TypedDeterministicAtom(
                atom_id=f"atom-{index:02d}",
                catalog_key=selection[0],
                comparison=selection[1],
                arguments=selection[2],
            )
            atom_cache[key] = atom
        return atom

    for arity in range(1, MAX_DETERMINISTIC_ATOMS + 1):
        for chosen_keys in combinations(keys, arity):
            for selections in product(*(by_key[key] for key in chosen_keys)):
                atoms = tuple(
                    intern_atom(index, selection)
                    for index, selection in enumerate(selections)
                )
                members.append(
                    ClosedPanelPredicate.direct(
                        DirectCountPredicate(atoms, _DIRECT_CATALOG_DIGEST)
                    )
                )
    members.extend(
        ClosedPanelPredicate.symmetry(
            SymmetryThresholdPredicate(metric, threshold)
        )
        for metric in SymmetryMetric
        for threshold in SYMMETRY_THRESHOLDS_PPM
    )
    return tuple(sorted(members, key=lambda item: item.digest))


@dataclass(frozen=True, slots=True)
class CompleteClosedPredicateLibraryIdentity:
    """Compact extensional identity shared by oracle and headless runner."""

    construction_id: str
    source_digest: str
    evaluator_digest: str
    construction_grid_digest: str
    complete_member_digest: str
    member_count: int

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": "gkm.bongard-proposer-reachable-closed-library-identity.v2",
            "algorithm_id": LIBRARY_ALGORITHM_ID,
            "construction_id": self.construction_id,
            "source_digest": self.source_digest,
            "evaluator_digest": self.evaluator_digest,
            "construction_grid_digest": self.construction_grid_digest,
            "complete_member_digest": self.complete_member_digest,
            "member_count": self.member_count,
            "membership": "proposer-reachable-closed-constructor/v2",
            "materialized_members_in_identity": False,
        }


@lru_cache(maxsize=1)
def complete_closed_predicate_library_identity(
) -> CompleteClosedPredicateLibraryIdentity:
    """Freeze the complete union specification without allocating its AST tuple."""

    relational_count = sum(
        not any(isinstance(clause, PointContactClause) for clause in item.clauses)
        for item in enumerate_factorized_shape_ratio_queries()
    )
    options_by_key = {
        spec.catalog_key: tuple(
            sorted(
                (option.to_data() for option in spec.allowed_options),
                key=canonical_json,
            )
        )
        for spec in sorted(
            DIRECT_VISUAL_ATOM_CATALOG.atoms, key=lambda item: item.catalog_key
        )
    }
    keys = tuple(sorted(options_by_key))
    direct_count = 0
    for arity in range(1, MAX_DETERMINISTIC_ATOMS + 1):
        for chosen_keys in combinations(keys, arity):
            multiplicity = 1
            for key in chosen_keys:
                multiplicity *= len(options_by_key[key])
            direct_count += multiplicity
    symmetry_count = len(tuple(SymmetryMetric)) * len(SYMMETRY_THRESHOLDS_PPM)
    grid = {
        "language_id": CLOSED_LANGUAGE_ID,
        "relational": {
            "side_counts": list(relational_visual_query.ALLOWED_SIDE_COUNTS),
            "area_ratios": [
                list(item) for item in relational_visual_query.ALLOWED_AREA_RATIOS
            ],
            "obliqueness_millidegrees": [
                None,
                *relational_visual_query.ALLOWED_OBLIQUENESS_THRESHOLDS_MILLIDEGREES,
            ],
            "point_contact_enabled": False,
            "member_count": relational_count,
        },
        "direct_counts": {
            "catalog_digest": _DIRECT_CATALOG_DIGEST,
            "options_by_key": options_by_key,
            "atom_arity": list(range(1, MAX_DETERMINISTIC_ATOMS + 1)),
            "unique_catalog_keys": True,
            "member_count": direct_count,
        },
        "symmetry": {
            "metrics": [item.value for item in SymmetryMetric],
            "thresholds_ppm": list(SYMMETRY_THRESHOLDS_PPM),
            "member_count": symmetry_count,
        },
    }
    member_count = relational_count + direct_count + symmetry_count
    construction_grid_digest = canonical_digest(grid)
    return CompleteClosedPredicateLibraryIdentity(
        construction_id="complete-proposer-reachable-closed-union/v2",
        source_digest=_source_digest(),
        evaluator_digest=closed_visual_predicate_evaluator_digest(),
        construction_grid_digest=construction_grid_digest,
        complete_member_digest=canonical_digest(
            {
                "schema": (
                    "gkm.bongard-complete-proposer-reachable-member-specification.v2"
                ),
                "algorithm_id": LIBRARY_ALGORITHM_ID,
                "construction_grid_digest": construction_grid_digest,
                "member_count": member_count,
                "membership": "proposer-reachable-closed-constructor/v2",
            }
        ),
        member_count=member_count,
    )


@dataclass(frozen=True, slots=True)
class FrozenClosedPredicateLibrary:
    """Immutable content-addressed library constructed before packet access."""

    construction_id: str
    source_digest: str
    evaluator_digest: str
    members: tuple[ClosedPanelPredicate, ...]

    def __post_init__(self) -> None:
        if self.construction_id not in {
            "explicit-closed-members/v1",
            "complete-proposer-reachable-closed-union/v2",
        }:
            raise ValueError("unknown frozen library construction ID")
        _digest(self.source_digest, "library source_digest")
        if self.source_digest != _source_digest():
            raise ValueError("frozen library source identity is not current")
        _digest(self.evaluator_digest, "library evaluator_digest")
        if self.evaluator_digest != closed_visual_predicate_evaluator_digest():
            raise ValueError("frozen library evaluator identity drifted")
        if not isinstance(self.members, tuple) or not self.members:
            raise ValueError("frozen library requires at least one member")
        if any(not isinstance(item, ClosedPanelPredicate) for item in self.members):
            raise TypeError("frozen library members must be closed predicates")
        digests = tuple(item.digest for item in self.members)
        if digests != tuple(sorted(digests)) or len(digests) != len(set(digests)):
            raise ValueError("frozen library members must be unique and digest-sorted")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": FROZEN_LIBRARY_SCHEMA,
            "algorithm_id": LIBRARY_ALGORITHM_ID,
            "construction_id": self.construction_id,
            "source_digest": self.source_digest,
            "evaluator_digest": self.evaluator_digest,
            "members": [item.to_data() for item in self.members],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "FrozenClosedPredicateLibrary":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "algorithm_id",
                    "construction_id",
                    "source_digest",
                    "evaluator_digest",
                    "members",
                }
            ),
            "frozen closed predicate library",
        )
        if (
            data["schema"] != FROZEN_LIBRARY_SCHEMA
            or data["algorithm_id"] != LIBRARY_ALGORITHM_ID
        ):
            raise ValueError("unsupported frozen closed predicate library")
        members = data["members"]
        if not isinstance(members, list) or any(
            not isinstance(item, Mapping) for item in members
        ):
            raise TypeError("frozen library members must be an object list")
        result = cls(
            construction_id=data["construction_id"],
            source_digest=data["source_digest"],
            evaluator_digest=data["evaluator_digest"],
            members=tuple(ClosedPanelPredicate.from_data(item) for item in members),
        )
        if result.to_data() != dict(data):
            raise ValueError("frozen library is not canonically represented")
        return result

    @property
    def digest(self) -> str:
        # Commit to every canonical member without first materializing the
        # enormous nested JSON representation of the complete union.
        return canonical_digest(
            {
                "schema": FROZEN_LIBRARY_SCHEMA,
                "algorithm_id": LIBRARY_ALGORITHM_ID,
                "digest_representation": "ordered-member-digests/v1",
                "construction_id": self.construction_id,
                "source_digest": self.source_digest,
                "evaluator_digest": self.evaluator_digest,
                "member_digests": [item.digest for item in self.members],
            }
        )


def freeze_closed_predicate_library(
    predicates: Sequence[ClosedPanelPredicate],
) -> FrozenClosedPredicateLibrary:
    """Freeze explicit members; the API deliberately accepts no panel packets."""

    if not isinstance(predicates, Sequence) or isinstance(
        predicates, (str, bytes, bytearray)
    ):
        raise TypeError("predicates must be a sequence of closed predicates")
    members = tuple(sorted(tuple(predicates), key=lambda item: item.digest))
    return FrozenClosedPredicateLibrary(
        construction_id="explicit-closed-members/v1",
        source_digest=_source_digest(),
        evaluator_digest=closed_visual_predicate_evaluator_digest(),
        members=members,
    )


def freeze_complete_closed_predicate_library() -> FrozenClosedPredicateLibrary:
    """Freeze the complete union before any support/query packet is accepted."""

    identity = complete_closed_predicate_library_identity()
    members = enumerate_complete_closed_predicates()
    if len(members) != identity.member_count:
        raise ValueError(
            "materialized complete library differs from its compact identity"
        )
    return FrozenClosedPredicateLibrary(
        construction_id=identity.construction_id,
        source_digest=identity.source_digest,
        evaluator_digest=identity.evaluator_digest,
        members=members,
    )


class OracleDiagnosis(str, Enum):
    NO_LANGUAGE_SEPARATOR = "no_language_separator"
    LANGUAGE_SEPARATOR_EXISTS_NO_MODEL_PROPOSAL = (
        "language_separator_exists_no_model_proposal"
    )
    MODEL_MISSED_SEPARATOR = "model_missed_separator"
    MODEL_FOUND_SEPARATOR = "model_found_separator"


@dataclass(frozen=True, slots=True)
class SupportExpressibilityResult:
    """Exhaustive support-only report; it contains no query inputs or labels."""

    library_digest: str
    evaluator_digest: str
    positive_packet_digests: tuple[str, ...]
    negative_packet_digests: tuple[str, ...]
    separator_digests: tuple[str, ...]
    evaluation_matrix_digest: str
    model_predicate_digest: str | None
    model_is_exact_separator: bool | None
    diagnosis: OracleDiagnosis

    def __post_init__(self) -> None:
        _digest(self.library_digest, "oracle library_digest")
        _digest(self.evaluator_digest, "oracle evaluator_digest")
        _digest(self.evaluation_matrix_digest, "oracle evaluation_matrix_digest")
        if self.evaluator_digest != closed_visual_predicate_evaluator_digest():
            raise ValueError("oracle evaluator identity drifted")
        for label, values in (
            ("positive", self.positive_packet_digests),
            ("negative", self.negative_packet_digests),
        ):
            if not isinstance(values, tuple) or not values:
                raise ValueError(f"oracle requires nonempty {label} supports")
            for value in values:
                _digest(value, f"oracle {label} packet digest")
            if len(values) != len(set(values)):
                raise ValueError(f"oracle {label} packet digests repeat")
        if set(self.positive_packet_digests) & set(self.negative_packet_digests):
            raise ValueError("one packet cannot occur in both support classes")
        if not isinstance(self.separator_digests, tuple):
            raise TypeError("oracle separator digests must be a tuple")
        for value in self.separator_digests:
            _digest(value, "oracle separator digest")
        if self.separator_digests != tuple(sorted(self.separator_digests)) or len(
            self.separator_digests
        ) != len(set(self.separator_digests)):
            raise ValueError("oracle separator digests must be unique and sorted")
        if self.model_predicate_digest is not None:
            _digest(self.model_predicate_digest, "oracle model predicate digest")
        if self.model_is_exact_separator is not None and type(
            self.model_is_exact_separator
        ) is not bool:
            raise TypeError("model_is_exact_separator must be Boolean or null")
        if not isinstance(self.diagnosis, OracleDiagnosis):
            raise TypeError("oracle diagnosis must be a registered enum")
        has_separator = bool(self.separator_digests)
        expected = (
            OracleDiagnosis.NO_LANGUAGE_SEPARATOR
            if not has_separator
            else OracleDiagnosis.LANGUAGE_SEPARATOR_EXISTS_NO_MODEL_PROPOSAL
            if self.model_predicate_digest is None
            else OracleDiagnosis.MODEL_FOUND_SEPARATOR
            if self.model_is_exact_separator
            else OracleDiagnosis.MODEL_MISSED_SEPARATOR
        )
        if self.diagnosis is not expected:
            raise ValueError("oracle diagnosis disagrees with exact separators")
        if self.model_predicate_digest is None and self.model_is_exact_separator is not None:
            raise ValueError("oracle cannot score an absent model predicate")
        if self.model_predicate_digest is not None and self.model_is_exact_separator is None:
            raise ValueError("oracle model predicate requires an exact-separator result")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": EXPRESSIBILITY_RESULT_SCHEMA,
            "algorithm_id": ORACLE_ALGORITHM_ID,
            "library_digest": self.library_digest,
            "evaluator_digest": self.evaluator_digest,
            "positive_packet_digests": list(self.positive_packet_digests),
            "negative_packet_digests": list(self.negative_packet_digests),
            "separator_digests": list(self.separator_digests),
            "evaluation_matrix_digest": self.evaluation_matrix_digest,
            "model_predicate_digest": self.model_predicate_digest,
            "model_is_exact_separator": self.model_is_exact_separator,
            "diagnosis": self.diagnosis.value,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "SupportExpressibilityResult":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "algorithm_id",
                    "library_digest",
                    "evaluator_digest",
                    "positive_packet_digests",
                    "negative_packet_digests",
                    "separator_digests",
                    "evaluation_matrix_digest",
                    "model_predicate_digest",
                    "model_is_exact_separator",
                    "diagnosis",
                }
            ),
            "support expressibility result",
        )
        if (
            data["schema"] != EXPRESSIBILITY_RESULT_SCHEMA
            or data["algorithm_id"] != ORACLE_ALGORITHM_ID
        ):
            raise ValueError("unsupported support expressibility result")
        for key in (
            "positive_packet_digests",
            "negative_packet_digests",
            "separator_digests",
        ):
            if not isinstance(data[key], list) or any(
                not isinstance(item, str) for item in data[key]
            ):
                raise TypeError(f"oracle {key} must be a string list")
        result = cls(
            library_digest=data["library_digest"],
            evaluator_digest=data["evaluator_digest"],
            positive_packet_digests=tuple(data["positive_packet_digests"]),
            negative_packet_digests=tuple(data["negative_packet_digests"]),
            separator_digests=tuple(data["separator_digests"]),
            evaluation_matrix_digest=data["evaluation_matrix_digest"],
            model_predicate_digest=data["model_predicate_digest"],
            model_is_exact_separator=data["model_is_exact_separator"],
            diagnosis=OracleDiagnosis(data["diagnosis"]),
        )
        if result.to_data() != dict(data):
            raise ValueError("support expressibility result is not canonical")
        return result

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_data())


def support_only_expressibility_oracle(
    library: FrozenClosedPredicateLibrary,
    *,
    positive_support_packets: Sequence[ExactPanelWitnessPacket],
    negative_support_packets: Sequence[ExactPanelWitnessPacket],
    model_predicate: ClosedPanelPredicate | None = None,
) -> SupportExpressibilityResult:
    """Exhaustively evaluate a pre-frozen library on labelled supports only."""

    if not isinstance(library, FrozenClosedPredicateLibrary):
        raise TypeError("oracle requires a pre-frozen closed predicate library")
    if FrozenClosedPredicateLibrary.from_data(library.to_data()) != library:
        raise ValueError("frozen library fails strict replay before evaluation")
    positives = tuple(positive_support_packets)
    negatives = tuple(negative_support_packets)
    if not positives or not negatives:
        raise ValueError("oracle requires both positive and negative supports")
    packets = (*positives, *negatives)
    if any(not isinstance(item, ExactPanelWitnessPacket) for item in packets):
        raise TypeError("oracle support inputs must be exact panel packets")
    for packet in packets:
        if ExactPanelWitnessPacket.from_data(packet.to_data()) != packet:
            raise ValueError("oracle received a noncanonical support packet")
    packet_digests = tuple(item.digest() for item in packets)
    if len(packet_digests) != len(set(packet_digests)):
        raise ValueError("oracle support packets must be unique")
    library_digest = library.digest
    member_digests = tuple(item.digest for item in library.members)
    library_by_digest = dict(zip(member_digests, library.members, strict=True))
    model_predicate_digest = (
        None if model_predicate is None else model_predicate.digest
    )
    if (
        model_predicate_digest is not None
        and model_predicate_digest not in library_by_digest
    ):
        raise ValueError("model predicate is not a member of the frozen library")

    matrix: list[dict[str, object]] = []
    separator_digests: list[str] = []
    direct_cache: DirectAtomCache = {}
    direct_atom_digest_cache: DirectAtomDigestCache = {}
    digest_by_identity = {
        id(packet): digest for packet, digest in zip(packets, packet_digests, strict=True)
    }
    model_exact: bool | None = None
    for predicate, predicate_digest in zip(
        library.members, member_digests, strict=True
    ):
        positive_results = tuple(
            _evaluate_closed_dispositions(
                predicate,
                packet,
                direct_atom_cache=direct_cache,
                direct_atom_digest_cache=direct_atom_digest_cache,
                precomputed_packet_digest=digest_by_identity[id(packet)],
            )
            for packet in positives
        )
        negative_results = tuple(
            _evaluate_closed_dispositions(
                predicate,
                packet,
                direct_atom_cache=direct_cache,
                direct_atom_digest_cache=direct_atom_digest_cache,
                precomputed_packet_digest=digest_by_identity[id(packet)],
            )
            for packet in negatives
        )
        exact = all(
            item[1] is Disposition.PRESENT for item in positive_results
        ) and all(
            item[1] is Disposition.CERTIFIED_ABSENT
            for item in negative_results
        )
        matrix.append(
            {
                "predicate_digest": predicate_digest,
                "positive": [item[1].value for item in positive_results],
                "negative": [item[1].value for item in negative_results],
                "exact_forward_separator": exact,
            }
        )
        if exact:
            separator_digests.append(predicate_digest)
        if (
            model_predicate_digest is not None
            and predicate_digest == model_predicate_digest
        ):
            model_exact = exact
    separator_tuple = tuple(sorted(separator_digests))
    diagnosis = (
        OracleDiagnosis.NO_LANGUAGE_SEPARATOR
        if not separator_tuple
        else OracleDiagnosis.LANGUAGE_SEPARATOR_EXISTS_NO_MODEL_PROPOSAL
        if model_predicate is None
        else OracleDiagnosis.MODEL_FOUND_SEPARATOR
        if model_exact
        else OracleDiagnosis.MODEL_MISSED_SEPARATOR
    )
    return SupportExpressibilityResult(
        library_digest=library_digest,
        evaluator_digest=closed_visual_predicate_evaluator_digest(),
        positive_packet_digests=packet_digests[: len(positives)],
        negative_packet_digests=packet_digests[len(positives) :],
        separator_digests=separator_tuple,
        evaluation_matrix_digest=canonical_digest(
            {
                "schema": "gkm.bongard-support-evaluation-matrix.v1",
                "library_digest": library_digest,
                "rows": matrix,
            }
        ),
        model_predicate_digest=model_predicate_digest,
        model_is_exact_separator=model_exact,
        diagnosis=diagnosis,
    )


__all__ = [
    "CLOSED_LANGUAGE_ID",
    "CLOSED_PREDICATE_SCHEMA",
    "CLOSED_RESULT_SCHEMA",
    "DIRECT_COUNT_PREDICATE_SCHEMA",
    "EXPRESSIBILITY_RESULT_SCHEMA",
    "FROZEN_LIBRARY_SCHEMA",
    "SYMMETRY_PREDICATE_SCHEMA",
    "SYMMETRY_THRESHOLDS_PPM",
    "ClosedPanelPredicate",
    "ClosedPredicateKind",
    "ClosedPredicateResult",
    "ClosedScenarioResult",
    "CompleteClosedPredicateLibraryIdentity",
    "DirectCountPredicate",
    "FrozenClosedPredicateLibrary",
    "OracleDiagnosis",
    "SupportExpressibilityResult",
    "SymmetryMetric",
    "SymmetryThresholdPredicate",
    "closed_visual_predicate_evaluator_digest",
    "closed_visual_predicate_source_digest",
    "complete_closed_predicate_library_identity",
    "enumerate_complete_closed_predicates",
    "evaluate_closed_predicate",
    "freeze_closed_predicate_library",
    "freeze_complete_closed_predicate_library",
    "support_only_expressibility_oracle",
    "verify_closed_predicate_result",
]

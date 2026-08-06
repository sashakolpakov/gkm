"""Joint-scenario semantics for deterministic visual conjunctions.

Each preprocessing scenario retains a correlated witness packet.  The full
direct conjunction is evaluated inside each packet before the scenario
results are compared.  This is intentionally different from taking an
independent min/max envelope for every feature or reaching consensus on each
atom separately: different near-miss atoms may reject different scenarios
while the complete conjunction is still constructively false in all of them.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

from bongard.evidence import Disposition, Evidence, Provenance


JOINT_SCENARIO_SEMANTICS_VERSION = "joint-direct-conjunction/v1"


class ScenarioSemanticsError(ValueError):
    """Scenario or atom evidence does not satisfy the closed joint contract."""


def _digest(data: object) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_evidence(value: object, label: str) -> Evidence[bool]:
    if not isinstance(value, Evidence):
        raise ScenarioSemanticsError(f"{label} must be Evidence[bool]")
    if value.disposition is Disposition.PRESENT and value.value is not True:
        raise ScenarioSemanticsError(f"{label} present evidence must contain true")
    return value


@dataclass(frozen=True, slots=True)
class ScenarioConjunctionResult:
    """One archived complete-conjunction outcome per correlated scenario."""

    scenario_dispositions: tuple[tuple[str, Disposition], ...]
    atom_ids: tuple[str, ...]
    evidence: Evidence[bool]

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_dispositions, tuple) or len(
            self.scenario_dispositions
        ) < 2:
            raise ScenarioSemanticsError("at least two scenario outcomes are required")
        scenario_ids = tuple(name for name, _ in self.scenario_dispositions)
        if scenario_ids != tuple(sorted(scenario_ids)):
            raise ScenarioSemanticsError("scenario outcomes must be sorted")
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ScenarioSemanticsError("scenario outcome IDs must be unique")
        if any(
            not isinstance(disposition, Disposition)
            for _, disposition in self.scenario_dispositions
        ):
            raise ScenarioSemanticsError("scenario outcome disposition is invalid")
        if not isinstance(self.atom_ids, tuple) or not self.atom_ids:
            raise ScenarioSemanticsError("joint conjunction requires at least one atom")
        if self.atom_ids != tuple(sorted(self.atom_ids)):
            raise ScenarioSemanticsError("joint atom IDs must be sorted")
        if len(self.atom_ids) != len(set(self.atom_ids)):
            raise ScenarioSemanticsError("joint atom IDs must be unique")
        _validate_evidence(self.evidence, "joint scenario result")

    def to_data(self) -> dict[str, object]:
        return {
            "version": JOINT_SCENARIO_SEMANTICS_VERSION,
            "atom_ids": list(self.atom_ids),
            "scenario_dispositions": [
                {"scenario_id": scenario_id, "disposition": disposition.value}
                for scenario_id, disposition in self.scenario_dispositions
            ],
            "result_disposition": self.evidence.disposition.value,
            "result_provenance_digest": self.evidence.provenance.digest(),
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_data())


def _scenario_conjunction(
    scenario_id: str,
    atom_evidence: tuple[tuple[str, Evidence[bool]], ...],
) -> Evidence[bool]:
    parents = tuple(value.provenance for _, value in atom_evidence)
    provenance = Provenance.composed(
        producer="bongard.joint_scenario_semantics",
        version="1",
        method="complete_direct_conjunction_inside_scenario",
        parents=parents,
        details=(
            ("atom_ids_digest", _digest([name for name, _ in atom_evidence])),
            ("scenario_id", scenario_id),
        ),
    )
    errors = tuple(
        value for _, value in atom_evidence if value.disposition is Disposition.ERROR
    )
    if errors:
        return Evidence.error(
            provenance,
            errors[0].error_type or "DirectAtomError",
            errors[0].reason or "direct atom failed inside scenario",
        )
    absences = tuple(
        value
        for _, value in atom_evidence
        if value.disposition is Disposition.CERTIFIED_ABSENT
    )
    if absences:
        return Evidence.certified_absent(
            provenance,
            "direct conjunct certified nonmatch inside scenario: "
            + (absences[0].certificate or "unspecified direct certificate"),
        )
    if any(
        value.disposition is Disposition.INDETERMINATE
        for _, value in atom_evidence
    ):
        return Evidence.indeterminate(
            provenance, "one or more direct conjuncts are indeterminate in scenario"
        )
    return Evidence.present(True, provenance)


def evaluate_joint_scenario_conjunction(
    evidence_by_scenario: Mapping[str, Mapping[str, Evidence[bool]]],
) -> ScenarioConjunctionResult:
    """Evaluate all direct atoms per scenario, then require scenario consensus.

    ``CERTIFIED_ABSENT`` here is an operational constructive nonmatch: in every
    retained preprocessing scenario at least one registered direct atom has a
    negative witness.  A failed extraction or malformed atom is always
    ``ERROR`` and can never contribute to that certificate.
    """

    if not isinstance(evidence_by_scenario, Mapping) or len(evidence_by_scenario) < 2:
        raise ScenarioSemanticsError("at least two scenario evidence maps are required")
    scenario_ids = tuple(sorted(evidence_by_scenario))
    if any(not isinstance(name, str) or not name for name in scenario_ids):
        raise ScenarioSemanticsError("scenario IDs must be non-empty strings")
    first = evidence_by_scenario[scenario_ids[0]]
    if not isinstance(first, Mapping) or not first:
        raise ScenarioSemanticsError("each scenario requires at least one direct atom")
    atom_ids = tuple(sorted(first))
    if any(not isinstance(name, str) or not name for name in atom_ids):
        raise ScenarioSemanticsError("atom IDs must be non-empty strings")

    per_scenario: list[tuple[str, Evidence[bool]]] = []
    for scenario_id in scenario_ids:
        raw = evidence_by_scenario[scenario_id]
        if not isinstance(raw, Mapping) or tuple(sorted(raw)) != atom_ids:
            raise ScenarioSemanticsError(
                "every scenario must provide the same complete direct atom set"
            )
        ordered = tuple(
            (atom_id, _validate_evidence(raw[atom_id], f"{scenario_id}.{atom_id}"))
            for atom_id in atom_ids
        )
        per_scenario.append(
            (scenario_id, _scenario_conjunction(scenario_id, ordered))
        )

    parents = tuple(value.provenance for _, value in per_scenario)
    provenance = Provenance.composed(
        producer="bongard.joint_scenario_semantics",
        version="1",
        method="scenario_consensus_after_complete_direct_conjunction",
        parents=parents,
        details=(
            ("atom_ids_digest", _digest(list(atom_ids))),
            ("scenario_ids_digest", _digest(list(scenario_ids))),
        ),
    )
    dispositions = tuple(value.disposition for _, value in per_scenario)
    errors = tuple(value for _, value in per_scenario if value.disposition is Disposition.ERROR)
    if errors:
        evidence = Evidence.error(
            provenance,
            errors[0].error_type or "ScenarioEvaluationError",
            errors[0].reason or "one retained scenario failed",
        )
    elif all(item is Disposition.PRESENT for item in dispositions):
        evidence = Evidence.present(True, provenance)
    elif all(item is Disposition.CERTIFIED_ABSENT for item in dispositions):
        evidence = Evidence.certified_absent(
            provenance,
            "the complete direct conjunction has a constructive nonmatch "
            "in every retained preprocessing scenario",
        )
    else:
        evidence = Evidence.indeterminate(
            provenance,
            "retained preprocessing scenarios disagree or contain an "
            "indeterminate complete-conjunction outcome",
        )
    return ScenarioConjunctionResult(
        scenario_dispositions=tuple(
            (scenario_id, value.disposition) for scenario_id, value in per_scenario
        ),
        atom_ids=atom_ids,
        evidence=evidence,
    )


__all__ = [
    "JOINT_SCENARIO_SEMANTICS_VERSION",
    "ScenarioConjunctionResult",
    "ScenarioSemanticsError",
    "evaluate_joint_scenario_conjunction",
]

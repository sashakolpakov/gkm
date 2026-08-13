"""Closed count-rule language over complete panel-program hypotheses.

Every candidate formula is evaluated on each concrete retained hypothesis and
then supervaluated as a whole.  This preserves correlations that would be lost
by projecting ambiguous hypotheses into independent marginal count sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.panel_program_observation import (
    PanelProgramObservation,
    ProgramHypothesisObservation,
)
from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")

LANGUAGE_SCHEMA = "gkm.panel-program-count-rule-language.v1"
VERSION_SPACE_SCHEMA = "gkm.panel-program-count-version-space.v1"
RULE_SCHEMA = "gkm.panel-program-frozen-count-rule.v1"
DECISION_SCHEMA = "gkm.panel-program-rule-decision.v1"
SUPPORT_GAP_SCHEMA = "gkm.panel-program-count-support-gap.v1"


class PanelProgramPredicateError(ValueError):
    """A formula, support proof, frozen rule, or decision differs."""


class ProgramSupportGapError(PanelProgramPredicateError):
    """Raised with a canonical zero-survivor support certificate."""

    def __init__(self, gap: "ProgramSupportGap") -> None:
        if type(gap) is not ProgramSupportGap:
            raise TypeError("gap must be exact ProgramSupportGap")
        self.gap = ProgramSupportGap.from_data(gap.to_data())
        super().__init__(
            f"semantic support gap ({self.gap.gap_kind}): "
            f"0/{self.gap.candidate_count} survivors; "
            f"matrix={self.gap.version_space_digest}; gap={self.gap.gap_digest}"
        )


class ProgramAxis(str, Enum):
    STRAIGHT = "straight_count"
    ARC = "arc_count"
    TOTAL = "total_count"
    MIX = "primitive_mix"


class ProgramDisposition(str, Enum):
    PRESENT = "present"
    CERTIFIED_ABSENT = "certified_absent"
    INDETERMINATE = "indeterminate"
    ERROR = "error"


_DOMAINS: dict[ProgramAxis, tuple[int | str, ...]] = {
    ProgramAxis.STRAIGHT: tuple(range(10)),
    ProgramAxis.ARC: tuple(range(10)),
    ProgramAxis.TOTAL: tuple(range(1, 10)),
    ProgramAxis.MIX: ("arc_only", "mixed", "straight_only"),
}


def source_sha256() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def program_predicate_algorithm_digest() -> str:
    return _address(
        {
            "schema": "gkm.panel-program-predicate-algorithm.v1",
            "source_digest": "sha256:" + source_sha256(),
            "language_schema": LANGUAGE_SCHEMA,
            "support_policy": (
                "all-six-positive-present-and-all-six-contrast-certified-absent"
            ),
            "selection_policy": "minimum-atom-count-then-formula-digest",
            "whole_formula_supervaluation": True,
        }
    )


def _address(data: object) -> str:
    return "sha256:" + canonical_digest(data)


def _require_address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise PanelProgramPredicateError(f"{label} must be a sha256: address")
    return value


def _atom_content(value: "ProgramAtom") -> dict[str, object]:
    return {"axis": value.axis.value, "expected": value.expected}


@dataclass(frozen=True, slots=True)
class ProgramAtom:
    axis: ProgramAxis
    expected: int | str
    atom_digest: str

    def __post_init__(self) -> None:
        if type(self.axis) is not ProgramAxis:
            raise TypeError("axis must be exact ProgramAxis")
        if self.expected not in _DOMAINS[self.axis] or (
            self.axis is not ProgramAxis.MIX and type(self.expected) is not int
        ) or (self.axis is ProgramAxis.MIX and type(self.expected) is not str):
            raise PanelProgramPredicateError("atom expected value differs")
        _require_address(self.atom_digest, "atom digest")
        if self.atom_digest != _address(_atom_content(self)):
            raise PanelProgramPredicateError("atom digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_atom_content(self), "atom_digest": self.atom_digest}

    @classmethod
    def create(cls, axis: ProgramAxis, expected: int | str) -> "ProgramAtom":
        provisional = object.__new__(cls)
        object.__setattr__(provisional, "axis", axis)
        object.__setattr__(provisional, "expected", expected)
        return cls(axis, expected, _address(_atom_content(provisional)))

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ProgramAtom":
        if type(data) is not dict or set(data) != {"axis", "expected", "atom_digest"}:
            raise PanelProgramPredicateError("atom fields differ")
        result = cls(ProgramAxis(data["axis"]), data["expected"], data["atom_digest"])
        if result.to_data() != data:
            raise PanelProgramPredicateError("atom is not canonical")
        return result


def _formula_content(value: "ProgramFormula") -> dict[str, object]:
    return {
        "schema": LANGUAGE_SCHEMA,
        "operator": "all_of",
        "atoms": [item.to_data() for item in value.atoms],
        "positive_only": True,
    }


@dataclass(frozen=True, slots=True)
class ProgramFormula:
    atoms: tuple[ProgramAtom, ...]
    formula_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.atoms) is not tuple
            or not 1 <= len(self.atoms) <= 2
            or any(type(item) is not ProgramAtom for item in self.atoms)
            or tuple(item.atom_digest for item in self.atoms)
            != tuple(sorted(item.atom_digest for item in self.atoms))
            or len({item.axis for item in self.atoms}) != len(self.atoms)
        ):
            raise PanelProgramPredicateError("formula inventory differs")
        for item in self.atoms:
            ProgramAtom.__post_init__(item)
        _require_address(self.formula_digest, "formula digest")
        if self.formula_digest != _address(_formula_content(self)):
            raise PanelProgramPredicateError("formula digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_formula_content(self), "formula_digest": self.formula_digest}

    @classmethod
    def create(cls, atoms: Sequence[ProgramAtom]) -> "ProgramFormula":
        frozen = tuple(sorted(tuple(atoms), key=lambda item: item.atom_digest))
        provisional = object.__new__(cls)
        object.__setattr__(provisional, "atoms", frozen)
        return cls(frozen, _address(_formula_content(provisional)))

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ProgramFormula":
        expected = {"schema", "operator", "atoms", "positive_only", "formula_digest"}
        if (
            type(data) is not dict or set(data) != expected
            or data["schema"] != LANGUAGE_SCHEMA or data["operator"] != "all_of"
            or data["positive_only"] is not True or type(data["atoms"]) is not list
        ):
            raise PanelProgramPredicateError("formula fields differ")
        result = cls(
            tuple(ProgramAtom.from_data(item) for item in data["atoms"]),
            data["formula_digest"],
        )
        if result.to_data() != data:
            raise PanelProgramPredicateError("formula is not canonical")
        return result


def enumerate_program_formulas() -> tuple[ProgramFormula, ...]:
    """Return the complete frozen singleton and cross-axis pair inventory."""

    atoms = tuple(
        ProgramAtom.create(axis, expected)
        for axis in ProgramAxis
        for expected in _DOMAINS[axis]
    )
    formulas = [ProgramFormula.create((atom,)) for atom in atoms]
    formulas.extend(
        ProgramFormula.create((left, right))
        for index, left in enumerate(atoms)
        for right in atoms[index + 1 :]
        if left.axis is not right.axis
    )
    return tuple(sorted(formulas, key=lambda item: item.formula_digest))


def _atom_truth(atom: ProgramAtom, hypothesis: ProgramHypothesisObservation) -> bool:
    values: dict[ProgramAxis, int | str] = {
        ProgramAxis.STRAIGHT: hypothesis.straight_count,
        ProgramAxis.ARC: hypothesis.arc_count,
        ProgramAxis.TOTAL: hypothesis.total_count,
        ProgramAxis.MIX: hypothesis.mix,
    }
    return values[atom.axis] == atom.expected


def _evaluation_content(value: "ProgramFormulaEvaluation") -> dict[str, object]:
    return {
        "formula_digest": value.formula_digest,
        "observation_digest": value.observation_digest,
        "hypothesis_truths": [[key, truth] for key, truth in value.hypothesis_truths],
        "disposition": value.disposition.value,
        "reason": value.reason,
        "certificate": value.certificate,
    }


@dataclass(frozen=True, slots=True)
class ProgramFormulaEvaluation:
    formula_digest: str
    observation_digest: str
    hypothesis_truths: tuple[tuple[str, bool], ...]
    disposition: ProgramDisposition
    reason: str | None
    certificate: str | None
    evaluation_digest: str

    def __post_init__(self) -> None:
        _require_address(self.formula_digest, "formula digest")
        _require_address(self.observation_digest, "observation digest")
        _require_address(self.evaluation_digest, "evaluation digest")
        if (
            type(self.hypothesis_truths) is not tuple
            or any(type(row) is not tuple or len(row) != 2 or type(row[1]) is not bool for row in self.hypothesis_truths)
            or tuple(key for key, _ in self.hypothesis_truths)
            != tuple(sorted({key for key, _ in self.hypothesis_truths}))
            or type(self.disposition) is not ProgramDisposition
        ):
            raise PanelProgramPredicateError("formula evaluation rows differ")
        for key, _ in self.hypothesis_truths:
            _require_address(key, "hypothesis digest")
        if self.disposition is ProgramDisposition.PRESENT:
            valid = bool(self.hypothesis_truths) and all(v for _, v in self.hypothesis_truths) and self.reason is None and self.certificate is None
        elif self.disposition is ProgramDisposition.CERTIFIED_ABSENT:
            valid = bool(self.hypothesis_truths) and not any(v for _, v in self.hypothesis_truths) and type(self.certificate) is str and bool(self.certificate) and self.reason is None
        else:
            valid = type(self.reason) is str and bool(self.reason) and self.certificate is None
        if not valid or self.evaluation_digest != _address(_evaluation_content(self)):
            raise PanelProgramPredicateError("formula evaluation disposition differs")

    def to_data(self) -> dict[str, object]:
        return {**_evaluation_content(self), "evaluation_digest": self.evaluation_digest}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ProgramFormulaEvaluation":
        expected = {"formula_digest", "observation_digest", "hypothesis_truths", "disposition", "reason", "certificate", "evaluation_digest"}
        if type(data) is not dict or set(data) != expected or type(data["hypothesis_truths"]) is not list:
            raise PanelProgramPredicateError("evaluation fields differ")
        result = cls(
            data["formula_digest"], data["observation_digest"],
            tuple((row[0], row[1]) for row in data["hypothesis_truths"]),
            ProgramDisposition(data["disposition"]), data["reason"],
            data["certificate"], data["evaluation_digest"],
        )
        if result.to_data() != data:
            raise PanelProgramPredicateError("evaluation is not canonical")
        return result


def evaluate_program_formula(
    formula: ProgramFormula,
    observation: PanelProgramObservation,
) -> ProgramFormulaEvaluation:
    if type(formula) is not ProgramFormula or type(observation) is not PanelProgramObservation:
        raise TypeError("formula/observation types differ")
    frozen_formula = ProgramFormula.from_data(formula.to_data())
    frozen_observation = PanelProgramObservation.from_data(observation.to_data())
    return _evaluate_validated(frozen_formula, frozen_observation)


def _evaluate_validated(
    frozen_formula: ProgramFormula,
    frozen_observation: PanelProgramObservation,
) -> ProgramFormulaEvaluation:
    """Evaluate already canonical parents inside a complete matrix replay."""

    truths = tuple(
        (hypothesis.hypothesis_digest, all(_atom_truth(atom, hypothesis) for atom in frozen_formula.atoms))
        for hypothesis in frozen_observation.hypotheses
    )
    if frozen_observation.state == "error":
        disposition = ProgramDisposition.ERROR
        reason = frozen_observation.reason
        certificate = None
    elif frozen_observation.state == "gap" or not truths:
        disposition = ProgramDisposition.INDETERMINATE
        reason = frozen_observation.reason or "no complete program hypotheses"
        certificate = None
    elif all(value for _, value in truths):
        disposition = ProgramDisposition.PRESENT
        reason = certificate = None
    elif not any(value for _, value in truths):
        disposition = ProgramDisposition.CERTIFIED_ABSENT
        reason = None
        certificate = "formula false for every complete frozen minimum hypothesis"
    else:
        disposition = ProgramDisposition.INDETERMINATE
        reason = "formula truth differs across complete frozen minimum hypotheses"
        certificate = None
    values = {
        "formula_digest": frozen_formula.formula_digest,
        "observation_digest": frozen_observation.observation_digest,
        "hypothesis_truths": truths,
        "disposition": disposition,
        "reason": reason,
        "certificate": certificate,
    }
    provisional = object.__new__(ProgramFormulaEvaluation)
    for key, item in values.items():
        object.__setattr__(provisional, key, item)
    return ProgramFormulaEvaluation(
        **values, evaluation_digest=_address(_evaluation_content(provisional))
    )


def _space_content(value: "ProgramVersionSpace") -> dict[str, object]:
    return {
        "schema": VERSION_SPACE_SCHEMA,
        "predicate_algorithm_digest": value.predicate_algorithm_digest,
        "predicate_source_digest": value.predicate_source_digest,
        "language_digest": value.language_digest,
        "positive_observations": [item.to_data() for item in value.positive_observations],
        "contrast_observations": [item.to_data() for item in value.contrast_observations],
        "positive_observation_digests": list(value.positive_observation_digests),
        "contrast_observation_digests": list(value.contrast_observation_digests),
        "formula_digests": list(value.formula_digests),
        "matrix": [[item.to_data() for item in row] for row in value.matrix],
        "survivor_formula_digests": list(value.survivor_formula_digests),
        "candidate_count": value.candidate_count,
        "survivor_count": value.survivor_count,
        "complete_matrix": True,
        "strict_support_policy": "all-six-positive-present-and-all-six-contrast-certified-absent",
    }


@dataclass(frozen=True, slots=True)
class ProgramVersionSpace:
    predicate_algorithm_digest: str
    predicate_source_digest: str
    language_digest: str
    positive_observations: tuple[PanelProgramObservation, ...]
    contrast_observations: tuple[PanelProgramObservation, ...]
    positive_observation_digests: tuple[str, ...]
    contrast_observation_digests: tuple[str, ...]
    formula_digests: tuple[str, ...]
    matrix: tuple[tuple[ProgramFormulaEvaluation, ...], ...]
    survivor_formula_digests: tuple[str, ...]
    candidate_count: int
    survivor_count: int
    version_space_digest: str

    def __post_init__(self) -> None:
        for label, digest in (
            ("predicate algorithm", self.predicate_algorithm_digest),
            ("predicate source", self.predicate_source_digest),
            ("language", self.language_digest),
            ("version space", self.version_space_digest),
        ):
            _require_address(digest, label + " digest")
        formulas = enumerate_program_formulas()
        expected_formula_digests = tuple(item.formula_digest for item in formulas)
        if (
            type(self.positive_observations) is not tuple
            or len(self.positive_observations) != 6
            or type(self.contrast_observations) is not tuple
            or len(self.contrast_observations) != 6
            or any(type(item) is not PanelProgramObservation for item in (*self.positive_observations, *self.contrast_observations))
            or type(self.positive_observation_digests) is not tuple
            or len(self.positive_observation_digests) != 6
            or type(self.contrast_observation_digests) is not tuple
            or len(self.contrast_observation_digests) != 6
            or self.formula_digests != expected_formula_digests
            or type(self.matrix) is not tuple or len(self.matrix) != len(formulas)
            or any(type(row) is not tuple or len(row) != 12 for row in self.matrix)
            or type(self.candidate_count) is not int or self.candidate_count != len(formulas)
            or type(self.survivor_count) is not int or self.survivor_count != len(self.survivor_formula_digests)
            or self.survivor_formula_digests != tuple(sorted(set(self.survivor_formula_digests)))
        ):
            raise PanelProgramPredicateError("version-space inventory differs")
        observations = self.positive_observations + self.contrast_observations
        for item in observations:
            PanelProgramObservation.from_data(item.to_data())
        bindings = {
            (
                item.observer_source_digest,
                item.observer_algorithm_digest,
                item.search_space_digest,
                item.hypothesis_policy_digest,
            )
            for item in observations
        }
        if len(bindings) != 1:
            raise PanelProgramPredicateError(
                "version-space observer policies differ"
            )
        if (
            self.positive_observation_digests
            != tuple(item.observation_digest for item in self.positive_observations)
            or self.contrast_observation_digests
            != tuple(item.observation_digest for item in self.contrast_observations)
        ):
            raise PanelProgramPredicateError("version-space observation bindings differ")
        all_observations = self.positive_observation_digests + self.contrast_observation_digests
        for index, (formula, row) in enumerate(zip(formulas, self.matrix, strict=True)):
            if tuple(item.formula_digest for item in row) != (formula.formula_digest,) * 12 or tuple(item.observation_digest for item in row) != all_observations:
                raise PanelProgramPredicateError("version-space matrix binding differs")
            survives = all(item.disposition is ProgramDisposition.PRESENT for item in row[:6]) and all(item.disposition is ProgramDisposition.CERTIFIED_ABSENT for item in row[6:])
            if survives != (formula.formula_digest in self.survivor_formula_digests):
                raise PanelProgramPredicateError(f"version-space survivor row {index} differs")
            expected_row = tuple(_evaluate_validated(formula, observation) for observation in observations)
            if row != expected_row:
                raise PanelProgramPredicateError(
                    f"version-space semantic row {index} differs"
                )
        if (
            self.predicate_algorithm_digest != program_predicate_algorithm_digest()
            or self.predicate_source_digest != "sha256:" + source_sha256()
            or self.language_digest != _address([item.to_data() for item in formulas])
            or self.version_space_digest != _address(_space_content(self))
        ):
            raise PanelProgramPredicateError("version-space digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_space_content(self), "version_space_digest": self.version_space_digest}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ProgramVersionSpace":
        expected = {"schema", "predicate_algorithm_digest", "predicate_source_digest", "language_digest", "positive_observations", "contrast_observations", "positive_observation_digests", "contrast_observation_digests", "formula_digests", "matrix", "survivor_formula_digests", "candidate_count", "survivor_count", "complete_matrix", "strict_support_policy", "version_space_digest"}
        if type(data) is not dict or set(data) != expected or data["schema"] != VERSION_SPACE_SCHEMA or data["complete_matrix"] is not True or type(data["matrix"]) is not list:
            raise PanelProgramPredicateError("version-space fields differ")
        result = cls(
            data["predicate_algorithm_digest"], data["predicate_source_digest"],
            data["language_digest"],
            tuple(PanelProgramObservation.from_data(item) for item in data["positive_observations"]),
            tuple(PanelProgramObservation.from_data(item) for item in data["contrast_observations"]),
            tuple(data["positive_observation_digests"]),
            tuple(data["contrast_observation_digests"]), tuple(data["formula_digests"]),
            tuple(tuple(ProgramFormulaEvaluation.from_data(item) for item in row) for row in data["matrix"]),
            tuple(data["survivor_formula_digests"]), data["candidate_count"],
            data["survivor_count"], data["version_space_digest"],
        )
        if result.to_data() != data:
            raise PanelProgramPredicateError("version space is not canonical")
        return result


def build_program_version_space(
    positive_observations: Sequence[PanelProgramObservation],
    contrast_observations: Sequence[PanelProgramObservation],
) -> ProgramVersionSpace:
    if type(positive_observations) not in (tuple, list) or type(contrast_observations) not in (tuple, list):
        raise TypeError("support observations must be exact sequences")
    positive = tuple(PanelProgramObservation.from_data(item.to_data()) for item in positive_observations)
    contrast = tuple(PanelProgramObservation.from_data(item.to_data()) for item in contrast_observations)
    if len(positive) != 6 or len(contrast) != 6:
        raise PanelProgramPredicateError("strict support requires exactly six plus six")
    bindings = {
        (
            item.observer_source_digest,
            item.observer_algorithm_digest,
            item.search_space_digest,
            item.hypothesis_policy_digest,
        )
        for item in (*positive, *contrast)
    }
    if len(bindings) != 1:
        raise PanelProgramPredicateError("support observer policies differ")
    formulas = enumerate_program_formulas()
    observations = positive + contrast
    matrix = tuple(tuple(_evaluate_validated(formula, item) for item in observations) for formula in formulas)
    survivors = tuple(sorted(
        formula.formula_digest for formula, row in zip(formulas, matrix, strict=True)
        if all(item.disposition is ProgramDisposition.PRESENT for item in row[:6])
        and all(item.disposition is ProgramDisposition.CERTIFIED_ABSENT for item in row[6:])
    ))
    values = {
        "predicate_algorithm_digest": program_predicate_algorithm_digest(),
        "predicate_source_digest": "sha256:" + source_sha256(),
        "language_digest": _address([item.to_data() for item in formulas]),
        "positive_observations": positive,
        "contrast_observations": contrast,
        "positive_observation_digests": tuple(item.observation_digest for item in positive),
        "contrast_observation_digests": tuple(item.observation_digest for item in contrast),
        "formula_digests": tuple(item.formula_digest for item in formulas),
        "matrix": matrix,
        "survivor_formula_digests": survivors,
        "candidate_count": len(formulas),
        "survivor_count": len(survivors),
    }
    provisional = object.__new__(ProgramVersionSpace)
    for key, item in values.items():
        object.__setattr__(provisional, key, item)
    return ProgramVersionSpace(
        **values, version_space_digest=_address(_space_content(provisional))
    )


def _rule_content(value: "FrozenProgramRule") -> dict[str, object]:
    return {
        "schema": RULE_SCHEMA,
        "version_space": value.version_space.to_data(),
        "version_space_digest": value.version_space_digest,
        "language_digest": value.language_digest,
        "formula": value.formula.to_data(),
        "formula_digest": value.formula_digest,
        "selection_policy": "minimum-atom-count-then-formula-digest",
        "whole_formula_supervaluation": True,
    }


def _gap_content(value: "ProgramSupportGap") -> dict[str, object]:
    return {
        "schema": SUPPORT_GAP_SCHEMA,
        "version_space_digest": value.version_space_digest,
        "language_digest": value.language_digest,
        "candidate_count": value.candidate_count,
        "survivor_count": 0,
        "gap_kind": value.gap_kind,
        "error_cell_count": value.error_cell_count,
        "indeterminate_cell_count": value.indeterminate_cell_count,
        "error_observation_digests": list(value.error_observation_digests),
        "indeterminate_observation_digests": list(
            value.indeterminate_observation_digests
        ),
        "reason": {
            "language_gap": "complete-language-has-no-strict-support-separator",
            "observation_error": "support-observer-error-prevents-rule-freeze",
            "witness_ambiguity": "support-indeterminacy-prevents-rule-freeze",
        }[value.gap_kind],
        "complete_matrix_evaluated": True,
    }


@dataclass(frozen=True, slots=True)
class ProgramSupportGap:
    version_space_digest: str
    language_digest: str
    candidate_count: int
    survivor_count: int
    gap_kind: str
    error_cell_count: int
    indeterminate_cell_count: int
    error_observation_digests: tuple[str, ...]
    indeterminate_observation_digests: tuple[str, ...]
    gap_digest: str

    def __post_init__(self) -> None:
        _require_address(self.version_space_digest, "gap version-space digest")
        _require_address(self.language_digest, "gap language digest")
        _require_address(self.gap_digest, "support gap digest")
        if (
            type(self.candidate_count) is not int
            or self.candidate_count != len(enumerate_program_formulas())
            or type(self.survivor_count) is not int
            or self.survivor_count != 0
            or type(self.gap_kind) is not str
            or self.gap_kind not in {
                "language_gap", "observation_error", "witness_ambiguity"
            }
            or type(self.error_cell_count) is not int
            or self.error_cell_count < 0
            or type(self.indeterminate_cell_count) is not int
            or self.indeterminate_cell_count < 0
            or type(self.error_observation_digests) is not tuple
            or type(self.indeterminate_observation_digests) is not tuple
            or self.error_observation_digests
            != tuple(sorted(set(self.error_observation_digests)))
            or self.indeterminate_observation_digests
            != tuple(sorted(set(self.indeterminate_observation_digests)))
            or any(
                _ADDRESS.fullmatch(item) is None
                for item in (
                    *self.error_observation_digests,
                    *self.indeterminate_observation_digests,
                )
            )
            or self.gap_kind
            != (
                "observation_error"
                if self.error_cell_count
                else "witness_ambiguity"
                if self.indeterminate_cell_count
                else "language_gap"
            )
            or self.gap_digest != _address(_gap_content(self))
        ):
            raise PanelProgramPredicateError("support gap differs")

    def to_data(self) -> dict[str, object]:
        return {**_gap_content(self), "gap_digest": self.gap_digest}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ProgramSupportGap":
        expected = {
            "schema", "version_space_digest", "language_digest",
            "candidate_count", "survivor_count", "gap_kind",
            "error_cell_count", "indeterminate_cell_count",
            "error_observation_digests", "indeterminate_observation_digests",
            "reason",
            "complete_matrix_evaluated", "gap_digest",
        }
        if (
            type(data) is not dict
            or set(data) != expected
            or data["schema"] != SUPPORT_GAP_SCHEMA
            or data["complete_matrix_evaluated"] is not True
            or type(data["error_observation_digests"]) is not list
            or type(data["indeterminate_observation_digests"]) is not list
        ):
            raise PanelProgramPredicateError("support gap fields differ")
        result = cls(
            data["version_space_digest"], data["language_digest"],
            data["candidate_count"], data["survivor_count"], data["gap_kind"],
            data["error_cell_count"], data["indeterminate_cell_count"],
            tuple(data["error_observation_digests"]),
            tuple(data["indeterminate_observation_digests"]),
            data["gap_digest"],
        )
        if result.to_data() != data:
            raise PanelProgramPredicateError("support gap is not canonical")
        return result


def program_support_gap(version_space: ProgramVersionSpace) -> ProgramSupportGap:
    if type(version_space) is not ProgramVersionSpace:
        raise TypeError("version_space must be exact ProgramVersionSpace")
    space = ProgramVersionSpace.from_data(version_space.to_data())
    if space.survivor_count != 0 or space.survivor_formula_digests:
        raise PanelProgramPredicateError("support space is not a zero-survivor gap")
    observations = space.positive_observations + space.contrast_observations
    error_indices = {
        index
        for row in space.matrix
        for index, evaluation in enumerate(row)
        if evaluation.disposition is ProgramDisposition.ERROR
    }
    indeterminate_indices = {
        index
        for row in space.matrix
        for index, evaluation in enumerate(row)
        if evaluation.disposition is ProgramDisposition.INDETERMINATE
    }
    error_cell_count = sum(
        evaluation.disposition is ProgramDisposition.ERROR
        for row in space.matrix
        for evaluation in row
    )
    indeterminate_cell_count = sum(
        evaluation.disposition is ProgramDisposition.INDETERMINATE
        for row in space.matrix
        for evaluation in row
    )
    values = {
        "version_space_digest": space.version_space_digest,
        "language_digest": space.language_digest,
        "candidate_count": space.candidate_count,
        "survivor_count": 0,
        "gap_kind": (
            "observation_error"
            if error_cell_count
            else "witness_ambiguity"
            if indeterminate_cell_count
            else "language_gap"
        ),
        "error_cell_count": error_cell_count,
        "indeterminate_cell_count": indeterminate_cell_count,
        "error_observation_digests": tuple(
            sorted(
                {
                    observations[index].observation_digest
                    for index in error_indices
                }
            )
        ),
        "indeterminate_observation_digests": tuple(
            sorted(
                {
                    observations[index].observation_digest
                    for index in indeterminate_indices
                }
            )
        ),
    }
    provisional = object.__new__(ProgramSupportGap)
    for key, item in values.items():
        object.__setattr__(provisional, key, item)
    return ProgramSupportGap(
        **values, gap_digest=_address(_gap_content(provisional))
    )


@dataclass(frozen=True, slots=True)
class FrozenProgramRule:
    version_space: ProgramVersionSpace
    version_space_digest: str
    language_digest: str
    formula: ProgramFormula
    formula_digest: str
    rule_digest: str

    def __post_init__(self) -> None:
        for label, value in (("version space", self.version_space_digest), ("language", self.language_digest), ("formula", self.formula_digest), ("rule", self.rule_digest)):
            _require_address(value, label + " digest")
        if type(self.version_space) is not ProgramVersionSpace:
            raise TypeError("version_space must be exact ProgramVersionSpace")
        space = ProgramVersionSpace.from_data(self.version_space.to_data())
        selected = min(
            (
                item
                for item in enumerate_program_formulas()
                if item.formula_digest in space.survivor_formula_digests
            ),
            key=lambda item: (len(item.atoms), item.formula_digest),
            default=None,
        )
        if type(self.formula) is not ProgramFormula or ProgramFormula.from_data(self.formula.to_data()) != self.formula or self.version_space != space or self.version_space_digest != space.version_space_digest or self.language_digest != space.language_digest or selected != self.formula or self.formula.formula_digest != self.formula_digest or self.rule_digest != _address(_rule_content(self)):
            raise PanelProgramPredicateError("frozen rule differs")

    def to_data(self) -> dict[str, object]:
        return {**_rule_content(self), "rule_digest": self.rule_digest}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "FrozenProgramRule":
        expected = {"schema", "version_space", "version_space_digest", "language_digest", "formula", "formula_digest", "selection_policy", "whole_formula_supervaluation", "rule_digest"}
        if type(data) is not dict or set(data) != expected or data["schema"] != RULE_SCHEMA or data["whole_formula_supervaluation"] is not True or type(data["version_space"]) is not dict or type(data["formula"]) is not dict:
            raise PanelProgramPredicateError("frozen rule fields differ")
        result = cls(ProgramVersionSpace.from_data(data["version_space"]), data["version_space_digest"], data["language_digest"], ProgramFormula.from_data(data["formula"]), data["formula_digest"], data["rule_digest"])
        if result.to_data() != data:
            raise PanelProgramPredicateError("frozen rule is not canonical")
        return result


def freeze_program_rule(version_space: ProgramVersionSpace) -> FrozenProgramRule:
    if type(version_space) is not ProgramVersionSpace:
        raise TypeError("version_space must be exact ProgramVersionSpace")
    space = ProgramVersionSpace.from_data(version_space.to_data())
    if not space.survivor_formula_digests:
        raise ProgramSupportGapError(program_support_gap(space))
    by_digest = {item.formula_digest: item for item in enumerate_program_formulas()}
    formula = min(
        (by_digest[item] for item in space.survivor_formula_digests),
        key=lambda item: (len(item.atoms), item.formula_digest),
    )
    values = {"version_space": space, "version_space_digest": space.version_space_digest, "language_digest": space.language_digest, "formula": formula, "formula_digest": formula.formula_digest}
    provisional = object.__new__(FrozenProgramRule)
    for key, item in values.items():
        object.__setattr__(provisional, key, item)
    return FrozenProgramRule(**values, rule_digest=_address(_rule_content(provisional)))


def _decision_content(value: "ProgramRuleDecision") -> dict[str, object]:
    return {
        "schema": DECISION_SCHEMA,
        "rule_digest": value.rule_digest,
        "observation_digest": value.observation_digest,
        "evaluation": value.evaluation.to_data(),
        "evaluation_digest": value.evaluation_digest,
        "prediction": value.prediction,
    }


@dataclass(frozen=True, slots=True)
class ProgramRuleDecision:
    rule_digest: str
    observation_digest: str
    evaluation: ProgramFormulaEvaluation
    evaluation_digest: str
    prediction: str
    decision_digest: str

    def __post_init__(self) -> None:
        for label, value in (("rule", self.rule_digest), ("observation", self.observation_digest), ("evaluation", self.evaluation_digest), ("decision", self.decision_digest)):
            _require_address(value, label + " digest")
        if type(self.evaluation) is not ProgramFormulaEvaluation or ProgramFormulaEvaluation.from_data(self.evaluation.to_data()) != self.evaluation or self.evaluation.evaluation_digest != self.evaluation_digest:
            raise PanelProgramPredicateError("decision evaluation differs")
        expected = {ProgramDisposition.PRESENT: "positive", ProgramDisposition.CERTIFIED_ABSENT: "contrast", ProgramDisposition.INDETERMINATE: "abstain", ProgramDisposition.ERROR: "error"}[self.evaluation.disposition]
        if (
            self.observation_digest != self.evaluation.observation_digest
            or self.prediction != expected
            or self.decision_digest != _address(_decision_content(self))
        ):
            raise PanelProgramPredicateError("decision prediction differs")

    def to_data(self) -> dict[str, object]:
        return {**_decision_content(self), "decision_digest": self.decision_digest}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ProgramRuleDecision":
        expected = {"schema", "rule_digest", "observation_digest", "evaluation", "evaluation_digest", "prediction", "decision_digest"}
        if type(data) is not dict or set(data) != expected or data["schema"] != DECISION_SCHEMA or type(data["evaluation"]) is not dict:
            raise PanelProgramPredicateError("decision fields differ")
        result = cls(data["rule_digest"], data["observation_digest"], ProgramFormulaEvaluation.from_data(data["evaluation"]), data["evaluation_digest"], data["prediction"], data["decision_digest"])
        if result.to_data() != data:
            raise PanelProgramPredicateError("decision is not canonical")
        return result


def evaluate_frozen_program_rule(rule: FrozenProgramRule, observation: PanelProgramObservation) -> ProgramRuleDecision:
    if type(rule) is not FrozenProgramRule or type(observation) is not PanelProgramObservation:
        raise TypeError("rule/observation types differ")
    frozen_rule = FrozenProgramRule.from_data(rule.to_data())
    frozen_observation = PanelProgramObservation.from_data(observation.to_data())
    support_bindings = {
        (
            item.observer_source_digest,
            item.observer_algorithm_digest,
            item.search_space_digest,
            item.hypothesis_policy_digest,
        )
        for item in (
            *frozen_rule.version_space.positive_observations,
            *frozen_rule.version_space.contrast_observations,
        )
    }
    query_binding = (
        frozen_observation.observer_source_digest,
        frozen_observation.observer_algorithm_digest,
        frozen_observation.search_space_digest,
        frozen_observation.hypothesis_policy_digest,
    )
    if support_bindings != {query_binding}:
        raise PanelProgramPredicateError(
            "query observation policy differs from frozen support"
        )
    evaluation = evaluate_program_formula(frozen_rule.formula, frozen_observation)
    prediction = {ProgramDisposition.PRESENT: "positive", ProgramDisposition.CERTIFIED_ABSENT: "contrast", ProgramDisposition.INDETERMINATE: "abstain", ProgramDisposition.ERROR: "error"}[evaluation.disposition]
    values = {"rule_digest": frozen_rule.rule_digest, "observation_digest": frozen_observation.observation_digest, "evaluation": evaluation, "evaluation_digest": evaluation.evaluation_digest, "prediction": prediction}
    provisional = object.__new__(ProgramRuleDecision)
    for key, item in values.items():
        object.__setattr__(provisional, key, item)
    return ProgramRuleDecision(**values, decision_digest=_address(_decision_content(provisional)))


def cold_replay_program_version_space(data: Mapping[str, Any], *, expected_digest: str) -> ProgramVersionSpace:
    space = ProgramVersionSpace.from_data(data)
    if space.version_space_digest != _require_address(expected_digest, "expected version-space digest"):
        raise PanelProgramPredicateError("version-space external commitment differs")
    return space


def cold_replay_frozen_program_rule(data: Mapping[str, Any], *, expected_digest: str) -> FrozenProgramRule:
    rule = FrozenProgramRule.from_data(data)
    if rule.rule_digest != _require_address(expected_digest, "expected rule digest"):
        raise PanelProgramPredicateError("rule external commitment differs")
    return rule


def cold_replay_program_support_gap(
    data: Mapping[str, Any],
    *,
    version_space: ProgramVersionSpace,
    expected_digest: str,
) -> ProgramSupportGap:
    gap = ProgramSupportGap.from_data(data)
    expected_space = ProgramVersionSpace.from_data(version_space.to_data())
    expected_gap = program_support_gap(expected_space)
    if (
        gap != expected_gap
        or gap.gap_digest
        != _require_address(expected_digest, "expected support-gap digest")
    ):
        raise PanelProgramPredicateError("support-gap external commitment differs")
    return gap


def cold_replay_program_rule_decision(
    data: Mapping[str, Any],
    *,
    rule: FrozenProgramRule,
    observation: PanelProgramObservation,
    expected_digest: str,
) -> ProgramRuleDecision:
    decision = ProgramRuleDecision.from_data(data)
    expected = evaluate_frozen_program_rule(rule, observation)
    if (
        decision != expected
        or decision.decision_digest
        != _require_address(expected_digest, "expected decision digest")
    ):
        raise PanelProgramPredicateError("decision cold replay differs")
    return decision


__all__ = (
    "DECISION_SCHEMA", "LANGUAGE_SCHEMA", "RULE_SCHEMA", "SUPPORT_GAP_SCHEMA",
    "VERSION_SPACE_SCHEMA",
    "FrozenProgramRule", "PanelProgramPredicateError", "ProgramAtom",
    "ProgramAxis", "ProgramDisposition", "ProgramFormula",
    "ProgramFormulaEvaluation", "ProgramRuleDecision", "ProgramSupportGap",
    "ProgramSupportGapError", "ProgramVersionSpace", "build_program_version_space",
    "cold_replay_frozen_program_rule", "cold_replay_program_rule_decision",
    "cold_replay_program_support_gap",
    "cold_replay_program_version_space", "enumerate_program_formulas",
    "evaluate_frozen_program_rule", "evaluate_program_formula",
    "freeze_program_rule", "program_support_gap", "source_sha256",
)

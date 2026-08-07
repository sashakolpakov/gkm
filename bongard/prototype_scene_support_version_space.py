"""Complete three-candidate support synthesis for prototype-scene tags."""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.prototype_pair_cohort import OPAQUE_TAG_IDS
from bongard.prototype_scene_calibration import (
    PrototypeSceneCalibrationFamily,
    PrototypeSceneDisposition,
)
from bongard.prototype_scene_predicates import (
    PrototypeScenePanelEvaluation,
    PrototypeScenePredicateLibrary,
)


CANDIDATE_SCHEMA = "gkm.bongard-prototype-scene-support-candidate.v1"
VERSION_SCHEMA = "gkm.bongard-prototype-scene-support-version-space.v1"
GAP_SCHEMA = "gkm.bongard-prototype-scene-support-gap.v1"
DIAGNOSTIC_SCHEMA = "gkm.bongard-prototype-scene-support-diagnostic.v1"
RANKING_SCHEMA = "gkm.bongard-prototype-scene-verified-ranking.v1"
CANDIDATE_RESULT_SCHEMA = "gkm.bongard-prototype-scene-candidate-result.v1"
ALGORITHM_ID = "bongard.prototype-scene-support/complete-positive-pair-v1"

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,255}\Z")


class PrototypeSceneSupportError(ValueError):
    """A support candidate, version space, ranking, or result is invalid."""


class PrototypeSceneCandidateKind(str, Enum):
    ATOM = "positive_atom"
    POSITIVE_CONJUNCTION = "positive_two_atom_conjunction"


class PrototypeSceneGapKind(str, Enum):
    LANGUAGE_GAP = "language_gap"
    WITNESS_GAP = "witness_gap"


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _require_address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise PrototypeSceneSupportError(f"{label} must be a sha256: address")
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise PrototypeSceneSupportError(f"{label} must be a bounded identifier")
    return value


def _fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PrototypeSceneSupportError(f"{label} fields differ from schema")


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_defines_identity_or_decision": False,
        "lean_required_for_replay": False,
        "lean_removal_changes_decision": False,
    }


def prototype_scene_support_algorithm_digest() -> str:
    return _address(
        {
            "schema": "gkm.bongard-prototype-scene-support-algorithm.v1",
            "algorithm_id": ALGORITHM_ID,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "candidates": "two-positive-atoms-plus-their-one-positive-conjunction",
            "support": "ordered-six-positive-plus-six-negative",
            "positive_accept": PrototypeSceneDisposition.CALIBRATED_PRESENT.value,
            "negative_accept": PrototypeSceneDisposition.CALIBRATED_ABSENT.value,
            "conjunction": "any-CA-then-CA;all-CP-then-CP;any-E-then-E;else-I",
            "negation": False,
            **_authority_data(),
        }
    )


def _atom_ids(library: PrototypeScenePredicateLibrary) -> tuple[str, str]:
    return tuple(item.predicate_id for item in library.predicates)  # type: ignore[return-value]


def _candidate_content(value: "PrototypeSceneSupportCandidate") -> dict[str, object]:
    return {
        "schema": CANDIDATE_SCHEMA,
        "candidate_id": value.candidate_id,
        "kind": value.kind.value,
        "atom_predicate_ids": list(value.atom_predicate_ids),
        "positive_only": True,
        "negation": False,
        **_authority_data(),
    }


@dataclass(frozen=True, order=True, slots=True)
class PrototypeSceneSupportCandidate:
    candidate_id: str
    kind: PrototypeSceneCandidateKind
    atom_predicate_ids: tuple[str, ...]
    record_digest: str

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, "candidate_id")
        expected = 1 if self.kind is PrototypeSceneCandidateKind.ATOM else 2
        if (
            not isinstance(self.kind, PrototypeSceneCandidateKind)
            or len(self.atom_predicate_ids) != expected
            or self.atom_predicate_ids != tuple(sorted(set(self.atom_predicate_ids)))
        ):
            raise PrototypeSceneSupportError("candidate atom inventory differs")
        expected_id = (
            self.atom_predicate_ids[0]
            if expected == 1
            else "prototype-scene:positive-and:"
            + "+".join(item.rsplit(":", 1)[-1] for item in self.atom_predicate_ids)
        )
        if (
            self.candidate_id != expected_id
            or self.record_digest != _address(_candidate_content(self))
        ):
            raise PrototypeSceneSupportError("candidate identity differs")

    @classmethod
    def seal(
        cls, kind: PrototypeSceneCandidateKind, atom_ids: Sequence[str]
    ) -> "PrototypeSceneSupportCandidate":
        atoms = tuple(sorted(atom_ids))
        candidate_id = (
            atoms[0]
            if kind is PrototypeSceneCandidateKind.ATOM and len(atoms) == 1
            else "prototype-scene:positive-and:"
            + "+".join(item.rsplit(":", 1)[-1] for item in atoms)
        )
        provisional = object.__new__(cls)
        object.__setattr__(provisional, "candidate_id", candidate_id)
        object.__setattr__(provisional, "kind", kind)
        object.__setattr__(provisional, "atom_predicate_ids", atoms)
        return cls(candidate_id, kind, atoms, _address(_candidate_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_candidate_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "PrototypeSceneSupportCandidate":
        expected = {
            "schema",
            "candidate_id",
            "kind",
            "atom_predicate_ids",
            "positive_only",
            "negation",
            *_authority_data(),
            "record_digest",
        }
        _fields(value, expected, "prototype-scene candidate")
        if not isinstance(value["atom_predicate_ids"], list):
            raise PrototypeSceneSupportError("candidate atoms are malformed")
        result = cls(
            value["candidate_id"],
            PrototypeSceneCandidateKind(value["kind"]),
            tuple(value["atom_predicate_ids"]),
            value["record_digest"],
        )
        if result.to_data() != dict(value):
            raise PrototypeSceneSupportError("candidate is not canonical")
        return result


def complete_prototype_scene_candidates(
    library: PrototypeScenePredicateLibrary,
) -> tuple[PrototypeSceneSupportCandidate, ...]:
    if not isinstance(library, PrototypeScenePredicateLibrary):
        raise TypeError("library must be PrototypeScenePredicateLibrary")
    atoms = _atom_ids(library)
    return tuple(
        sorted(
            (
                PrototypeSceneSupportCandidate.seal(
                    PrototypeSceneCandidateKind.ATOM, (atoms[0],)
                ),
                PrototypeSceneSupportCandidate.seal(
                    PrototypeSceneCandidateKind.ATOM, (atoms[1],)
                ),
                PrototypeSceneSupportCandidate.seal(
                    PrototypeSceneCandidateKind.POSITIVE_CONJUNCTION, atoms
                ),
            ),
            key=lambda item: item.candidate_id,
        )
    )


def combine_positive_dispositions(
    values: Sequence[PrototypeSceneDisposition],
) -> PrototypeSceneDisposition:
    dispositions = tuple(values)
    if not dispositions:
        raise PrototypeSceneSupportError("conjunction requires atoms")
    if PrototypeSceneDisposition.CALIBRATED_ABSENT in dispositions:
        return PrototypeSceneDisposition.CALIBRATED_ABSENT
    if all(item is PrototypeSceneDisposition.CALIBRATED_PRESENT for item in dispositions):
        return PrototypeSceneDisposition.CALIBRATED_PRESENT
    if PrototypeSceneDisposition.ERROR in dispositions:
        return PrototypeSceneDisposition.ERROR
    return PrototypeSceneDisposition.INDETERMINATE


def evaluate_prototype_scene_candidate_disposition(
    candidate: PrototypeSceneSupportCandidate,
    library: PrototypeScenePredicateLibrary,
    family: PrototypeSceneCalibrationFamily,
    panel: PrototypeScenePanelEvaluation,
) -> PrototypeSceneDisposition:
    library.assert_matches_family(family)
    panel.assert_matches(family)
    complete = complete_prototype_scene_candidates(library)
    if candidate not in complete:
        raise PrototypeSceneSupportError("candidate is outside complete inventory")
    by_predicate = {
        predicate.predicate_id: panel.result(predicate.tag_id).disposition
        for predicate in library.predicates
    }
    values = tuple(by_predicate[item] for item in candidate.atom_predicate_ids)
    return values[0] if len(values) == 1 else combine_positive_dispositions(values)


@dataclass(frozen=True, order=True, slots=True)
class PrototypeSceneSupportDiagnostic:
    candidate_id: str
    definite_counterexample_panel_ids: tuple[str, ...]
    indeterminate_panel_ids: tuple[str, ...]
    error_panel_ids: tuple[str, ...]

    def to_data(self) -> dict[str, object]:
        return {
            "schema": DIAGNOSTIC_SCHEMA,
            "candidate_id": self.candidate_id,
            "definite_counterexample_panel_ids": list(
                self.definite_counterexample_panel_ids
            ),
            "indeterminate_panel_ids": list(self.indeterminate_panel_ids),
            "error_panel_ids": list(self.error_panel_ids),
        }

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "PrototypeSceneSupportDiagnostic":
        expected = {
            "schema",
            "candidate_id",
            "definite_counterexample_panel_ids",
            "indeterminate_panel_ids",
            "error_panel_ids",
        }
        _fields(value, expected, "support diagnostic")
        result = cls(
            value["candidate_id"],
            tuple(value["definite_counterexample_panel_ids"]),
            tuple(value["indeterminate_panel_ids"]),
            tuple(value["error_panel_ids"]),
        )
        if result.to_data() != dict(value):
            raise PrototypeSceneSupportError("diagnostic is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class PrototypeSceneSupportGap:
    kind: PrototypeSceneGapKind
    diagnostics: tuple[PrototypeSceneSupportDiagnostic, ...]
    record_digest: str

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": GAP_SCHEMA,
            "kind": self.kind.value,
            "diagnostics": [item.to_data() for item in self.diagnostics],
            **_authority_data(),
        }

    def __post_init__(self) -> None:
        if (
            not isinstance(self.kind, PrototypeSceneGapKind)
            or not self.diagnostics
            or self.record_digest != _address(self.content_dict())
        ):
            raise PrototypeSceneSupportError("support gap differs")

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypeSceneSupportGap":
        expected = {"schema", "kind", "diagnostics", *_authority_data(), "record_digest"}
        _fields(value, expected, "support gap")
        if not isinstance(value["diagnostics"], list):
            raise PrototypeSceneSupportError("gap diagnostics are malformed")
        result = cls(
            PrototypeSceneGapKind(value["kind"]),
            tuple(
                PrototypeSceneSupportDiagnostic.from_data(item)
                for item in value["diagnostics"]
            ),
            value["record_digest"],
        )
        if result.to_data() != dict(value):
            raise PrototypeSceneSupportError("gap is not canonical")
        return result


def _version_content(value: "PrototypeSceneSupportVersionSpace") -> dict[str, object]:
    return {
        "schema": VERSION_SCHEMA,
        "algorithm_id": ALGORITHM_ID,
        "algorithm_digest": value.algorithm_digest,
        "library_digest": value.library_digest,
        "calibration_family_digest": value.calibration_family_digest,
        "candidates": [item.to_data() for item in value.candidates],
        "support_panel_ids": list(value.support_panel_ids),
        "support_panel_digests": list(value.support_panel_digests),
        "support_sides": ["positive"] * 6 + ["negative"] * 6,
        "rows": [list(item) for item in value.rows],
        "survivor_candidate_ids": list(value.survivor_candidate_ids),
        "gap": None if value.gap is None else value.gap.to_data(),
        "complete_candidate_inventory": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PrototypeSceneSupportVersionSpace:
    algorithm_digest: str
    library_digest: str
    calibration_family_digest: str
    candidates: tuple[PrototypeSceneSupportCandidate, ...]
    support_panel_ids: tuple[str, ...]
    support_panel_digests: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    survivor_candidate_ids: tuple[str, ...]
    gap: PrototypeSceneSupportGap | None
    record_digest: str

    def __post_init__(self) -> None:
        for name in (
            "algorithm_digest",
            "library_digest",
            "calibration_family_digest",
            "record_digest",
        ):
            _require_address(getattr(self, name), name)
        if (
            self.algorithm_digest != prototype_scene_support_algorithm_digest()
            or len(self.candidates) != 3
            or len(self.support_panel_ids) != 12
            or len(set(self.support_panel_ids)) != 12
            or len(self.support_panel_digests) != 12
            or len(self.rows) != 3
            or any(len(row) != 12 for row in self.rows)
            or self.survivor_candidate_ids
            != tuple(sorted(set(self.survivor_candidate_ids)))
            or bool(self.survivor_candidate_ids) == (self.gap is not None)
            or self.record_digest != _address(_version_content(self))
        ):
            raise PrototypeSceneSupportError("version-space identity differs")

    def row(self, candidate_id: str) -> tuple[str, ...]:
        matches = tuple(
            row
            for candidate, row in zip(self.candidates, self.rows, strict=True)
            if candidate.candidate_id == candidate_id
        )
        if len(matches) != 1:
            raise PrototypeSceneSupportError("candidate row is absent")
        return matches[0]

    def assert_matches(
        self,
        library: PrototypeScenePredicateLibrary,
        family: PrototypeSceneCalibrationFamily,
        positives: Sequence[PrototypeScenePanelEvaluation],
        negatives: Sequence[PrototypeScenePanelEvaluation],
    ) -> None:
        if build_prototype_scene_support_version_space(
            library, family, positives, negatives
        ) != self:
            raise PrototypeSceneSupportError("cold support replay differs")

    def to_data(self) -> dict[str, object]:
        return {**_version_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "PrototypeSceneSupportVersionSpace":
        expected = {
            "schema",
            "algorithm_id",
            "algorithm_digest",
            "library_digest",
            "calibration_family_digest",
            "candidates",
            "support_panel_ids",
            "support_panel_digests",
            "support_sides",
            "rows",
            "survivor_candidate_ids",
            "gap",
            "complete_candidate_inventory",
            *_authority_data(),
            "record_digest",
        }
        _fields(value, expected, "prototype-scene version space")
        raw_gap = value["gap"]
        if raw_gap is not None and not isinstance(raw_gap, Mapping):
            raise PrototypeSceneSupportError("version gap is malformed")
        result = cls(
            algorithm_digest=value["algorithm_digest"],
            library_digest=value["library_digest"],
            calibration_family_digest=value["calibration_family_digest"],
            candidates=tuple(
                PrototypeSceneSupportCandidate.from_data(item)
                for item in value["candidates"]
            ),
            support_panel_ids=tuple(value["support_panel_ids"]),
            support_panel_digests=tuple(value["support_panel_digests"]),
            rows=tuple(tuple(item) for item in value["rows"]),
            survivor_candidate_ids=tuple(value["survivor_candidate_ids"]),
            gap=(
                None if raw_gap is None else PrototypeSceneSupportGap.from_data(raw_gap)
            ),
            record_digest=value["record_digest"],
        )
        if result.to_data() != dict(value):
            raise PrototypeSceneSupportError("version space is not canonical")
        return result


def _is_separator(row: Sequence[str]) -> bool:
    return all(
        item == PrototypeSceneDisposition.CALIBRATED_PRESENT.value for item in row[:6]
    ) and all(
        item == PrototypeSceneDisposition.CALIBRATED_ABSENT.value for item in row[6:]
    )


def build_prototype_scene_support_version_space(
    library: PrototypeScenePredicateLibrary,
    family: PrototypeSceneCalibrationFamily,
    positives: Sequence[PrototypeScenePanelEvaluation],
    negatives: Sequence[PrototypeScenePanelEvaluation],
) -> PrototypeSceneSupportVersionSpace:
    positive_panels = tuple(positives)
    negative_panels = tuple(negatives)
    if len(positive_panels) != 6 or len(negative_panels) != 6:
        raise PrototypeSceneSupportError("support requires exactly six panels per side")
    panels = positive_panels + negative_panels
    library.assert_matches_family(family)
    if any(not isinstance(item, PrototypeScenePanelEvaluation) for item in panels):
        raise TypeError("support panels must be PrototypeScenePanelEvaluation")
    ids = tuple(item.panel_id for item in panels)
    if len(set(ids)) != 12:
        raise PrototypeSceneSupportError("support panel IDs must be unique")
    if any(item.calibration_family_digest != library.calibration_family_digest for item in panels):
        raise PrototypeSceneSupportError("support family differs from library")
    for panel in panels:
        panel.assert_matches(family)
    candidates = complete_prototype_scene_candidates(library)
    rows = tuple(
        tuple(
            evaluate_prototype_scene_candidate_disposition(
                candidate, library, family, panel
            ).value
            for panel in panels
        )
        for candidate in candidates
    )
    survivors = tuple(
        sorted(
            candidate.candidate_id
            for candidate, row in zip(candidates, rows, strict=True)
            if _is_separator(row)
        )
    )
    gap = None
    if not survivors:
        diagnostics = []
        for candidate, row in zip(candidates, rows, strict=True):
            definite = tuple(
                panel.panel_id
                for index, panel in enumerate(panels)
                if (
                    index < 6
                    and row[index] == PrototypeSceneDisposition.CALIBRATED_ABSENT.value
                )
                or (
                    index >= 6
                    and row[index] == PrototypeSceneDisposition.CALIBRATED_PRESENT.value
                )
            )
            diagnostics.append(
                PrototypeSceneSupportDiagnostic(
                    candidate.candidate_id,
                    definite,
                    tuple(
                        panel.panel_id
                        for panel, state in zip(panels, row, strict=True)
                        if state == PrototypeSceneDisposition.INDETERMINATE.value
                    ),
                    tuple(
                        panel.panel_id
                        for panel, state in zip(panels, row, strict=True)
                        if state == PrototypeSceneDisposition.ERROR.value
                    ),
                )
            )
        kind = (
            PrototypeSceneGapKind.LANGUAGE_GAP
            if all(item.definite_counterexample_panel_ids for item in diagnostics)
            else PrototypeSceneGapKind.WITNESS_GAP
        )
        provisional_gap = object.__new__(PrototypeSceneSupportGap)
        object.__setattr__(provisional_gap, "kind", kind)
        object.__setattr__(provisional_gap, "diagnostics", tuple(diagnostics))
        gap = PrototypeSceneSupportGap(
            kind, tuple(diagnostics), _address(provisional_gap.content_dict())
        )
    values: dict[str, object] = {
        "algorithm_digest": prototype_scene_support_algorithm_digest(),
        "library_digest": library.record_digest,
        "calibration_family_digest": library.calibration_family_digest,
        "candidates": candidates,
        "support_panel_ids": ids,
        "support_panel_digests": tuple(item.record_digest for item in panels),
        "rows": rows,
        "survivor_candidate_ids": survivors,
        "gap": gap,
    }
    provisional = object.__new__(PrototypeSceneSupportVersionSpace)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return PrototypeSceneSupportVersionSpace(
        **values,  # type: ignore[arg-type]
        record_digest=_address(_version_content(provisional)),
    )


@dataclass(frozen=True, slots=True)
class PrototypeSceneVerifiedRanking:
    version_space_digest: str
    library_digest: str
    ordered_candidate_ids: tuple[str, ...]
    record_digest: str

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": RANKING_SCHEMA,
            "version_space_digest": self.version_space_digest,
            "library_digest": self.library_digest,
            "ordered_candidate_ids": list(self.ordered_candidate_ids),
            "complete_survivor_permutation": True,
            **_authority_data(),
        }

    def __post_init__(self) -> None:
        _require_address(self.version_space_digest, "version_space_digest")
        _require_address(self.library_digest, "ranking library_digest")
        if (
            not self.ordered_candidate_ids
            or len(set(self.ordered_candidate_ids)) != len(self.ordered_candidate_ids)
            or self.record_digest != _address(self.content_dict())
        ):
            raise PrototypeSceneSupportError("ranking identity differs")

    def assert_matches(
        self,
        version: PrototypeSceneSupportVersionSpace,
        library: PrototypeScenePredicateLibrary,
    ) -> None:
        if rank_prototype_scene_survivors(
            version, library, self.ordered_candidate_ids
        ) != self:
            raise PrototypeSceneSupportError("cold ranking replay differs")

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypeSceneVerifiedRanking":
        expected = {
            "schema",
            "version_space_digest",
            "library_digest",
            "ordered_candidate_ids",
            "complete_survivor_permutation",
            *_authority_data(),
            "record_digest",
        }
        _fields(value, expected, "verified ranking")
        result = cls(
            value["version_space_digest"],
            value["library_digest"],
            tuple(value["ordered_candidate_ids"]),
            value["record_digest"],
        )
        if result.to_data() != dict(value):
            raise PrototypeSceneSupportError("ranking is not canonical")
        return result


def rank_prototype_scene_survivors(
    version: PrototypeSceneSupportVersionSpace,
    library: PrototypeScenePredicateLibrary,
    ordered_candidate_ids: Sequence[str],
) -> PrototypeSceneVerifiedRanking:
    ordered = tuple(ordered_candidate_ids)
    if (
        version.library_digest != library.record_digest
        or not version.survivor_candidate_ids
        or len(ordered) != len(set(ordered))
        or set(ordered) != set(version.survivor_candidate_ids)
    ):
        raise PrototypeSceneSupportError(
            "ranking must be an exact complete survivor permutation"
        )
    provisional = object.__new__(PrototypeSceneVerifiedRanking)
    object.__setattr__(provisional, "version_space_digest", version.record_digest)
    object.__setattr__(provisional, "library_digest", library.record_digest)
    object.__setattr__(provisional, "ordered_candidate_ids", ordered)
    return PrototypeSceneVerifiedRanking(
        version.record_digest,
        library.record_digest,
        ordered,
        _address(provisional.content_dict()),
    )


@dataclass(frozen=True, slots=True)
class PrototypeSceneCandidateResult:
    candidate_id: str
    candidate_digest: str
    library_digest: str
    panel_id: str
    panel_digest: str
    disposition: PrototypeSceneDisposition
    atom_result_digests: tuple[str, ...]
    record_digest: str

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": CANDIDATE_RESULT_SCHEMA,
            "candidate_id": self.candidate_id,
            "candidate_digest": self.candidate_digest,
            "library_digest": self.library_digest,
            "panel_id": self.panel_id,
            "panel_digest": self.panel_digest,
            "disposition": self.disposition.value,
            "atom_result_digests": list(self.atom_result_digests),
            **_authority_data(),
        }

    def __post_init__(self) -> None:
        for name in ("candidate_digest", "library_digest", "panel_digest", "record_digest"):
            _require_address(getattr(self, name), name)
        if (
            not isinstance(self.disposition, PrototypeSceneDisposition)
            or self.record_digest != _address(self.content_dict())
        ):
            raise PrototypeSceneSupportError("candidate result differs")

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "PrototypeSceneCandidateResult":
        expected = {
            "schema",
            "candidate_id",
            "candidate_digest",
            "library_digest",
            "panel_id",
            "panel_digest",
            "disposition",
            "atom_result_digests",
            *_authority_data(),
            "record_digest",
        }
        _fields(value, expected, "candidate result")
        result = cls(
            value["candidate_id"],
            value["candidate_digest"],
            value["library_digest"],
            value["panel_id"],
            value["panel_digest"],
            PrototypeSceneDisposition(value["disposition"]),
            tuple(value["atom_result_digests"]),
            value["record_digest"],
        )
        if result.to_data() != dict(value):
            raise PrototypeSceneSupportError("candidate result is not canonical")
        return result


def evaluate_prototype_scene_candidate(
    candidate: PrototypeSceneSupportCandidate,
    library: PrototypeScenePredicateLibrary,
    family: PrototypeSceneCalibrationFamily,
    panel: PrototypeScenePanelEvaluation,
) -> PrototypeSceneCandidateResult:
    disposition = evaluate_prototype_scene_candidate_disposition(
        candidate, library, family, panel
    )
    by_id = {item.predicate_id: item.tag_id for item in library.predicates}
    atom_digests = tuple(
        panel.result(by_id[atom_id]).record_digest
        for atom_id in candidate.atom_predicate_ids
    )
    values: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "candidate_digest": candidate.record_digest,
        "library_digest": library.record_digest,
        "panel_id": panel.panel_id,
        "panel_digest": panel.record_digest,
        "disposition": disposition,
        "atom_result_digests": atom_digests,
    }
    provisional = object.__new__(PrototypeSceneCandidateResult)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return PrototypeSceneCandidateResult(
        **values,  # type: ignore[arg-type]
        record_digest=_address(provisional.content_dict()),
    )


__all__ = [
    "ALGORITHM_ID",
    "PrototypeSceneCandidateKind",
    "PrototypeSceneCandidateResult",
    "PrototypeSceneGapKind",
    "PrototypeSceneSupportCandidate",
    "PrototypeSceneSupportError",
    "PrototypeSceneSupportGap",
    "PrototypeSceneSupportVersionSpace",
    "PrototypeSceneVerifiedRanking",
    "build_prototype_scene_support_version_space",
    "combine_positive_dispositions",
    "complete_prototype_scene_candidates",
    "evaluate_prototype_scene_candidate",
    "evaluate_prototype_scene_candidate_disposition",
    "prototype_scene_support_algorithm_digest",
    "rank_prototype_scene_survivors",
]

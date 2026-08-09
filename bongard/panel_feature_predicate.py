"""Finite Python predicates over the typed whole-panel feature ontology.

The scientific lane consumes only calibrated :class:`Disposition` values.
The engineering lane is deliberately separate: its enum and artifacts are
marked uncalibrated and cannot be passed to scientific constructors.  Both
lanes share only a closed positive ``AllOf`` language over typed feature
specifications; neither accepts executable text or a polarity-repair operator.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
import re
from typing import Any, Mapping, NoReturn, Sequence

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.panel_soft_ontology import (
    NativeOrientation,
    PanelFeatureSpec,
    feature_catalog_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


FEATURE_VOCABULARY_SCHEMA = "gkm.bongard-feature-vocabulary.v1"
FEATURE_SUPPORT_CELL_SCHEMA = "gkm.bongard-feature-support-cell.v1"
FEATURE_SUPPORT_TABLE_SCHEMA = "gkm.bongard-feature-support-table.v1"
ENGINEERING_SUPPORT_CELL_SCHEMA = "gkm.bongard-engineering-feature-support-cell.v1"
ENGINEERING_SUPPORT_TABLE_SCHEMA = "gkm.bongard-engineering-feature-support-table.v1"
ALL_OF_SCHEMA = "gkm.bongard-feature-all-of.v1"
FORMULA_GAP_DIAGNOSTIC_SCHEMA = "gkm.bongard-feature-formula-gap-diagnostic.v1"
FEATURE_SUPPORT_GAP_SCHEMA = "gkm.bongard-feature-support-gap.v1"
FEATURE_VERSION_SPACE_SCHEMA = "gkm.bongard-feature-version-space.v1"
ENGINEERING_VERSION_SPACE_SCHEMA = "gkm.bongard-engineering-feature-version-space.v1"
FROZEN_FEATURE_PREDICATE_SCHEMA = "gkm.bongard-frozen-feature-predicate.v1"
FROZEN_FEATURE_PAIR_SCHEMA = "gkm.bongard-frozen-feature-predicate-pair.v1"
FROZEN_ENGINEERING_PREDICATE_SCHEMA = (
    "gkm.bongard-frozen-engineering-feature-predicate.v1"
)
FROZEN_ENGINEERING_PAIR_SCHEMA = (
    "gkm.bongard-frozen-engineering-feature-predicate-pair.v1"
)
FEATURE_QUERY_DECISION_SCHEMA = "gkm.bongard-feature-query-decision.v1"
ENGINEERING_QUERY_DECISION_SCHEMA = (
    "gkm.bongard-engineering-feature-query-decision.v1"
)
PANEL_FEATURE_ALGORITHM_ID = (
    "bongard.panel-feature-predicate/positive-all-of-python-v1"
)
PANEL_FEATURE_MAX_CONJUNCTION = 2
PANEL_FEATURE_SUPPORTS_PER_SIDE = 6

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ORIENTATION_ORDER = (
    NativeOrientation.SIDE0_POSITIVE,
    NativeOrientation.SIDE1_POSITIVE,
)


class PanelFeaturePredicateError(ValueError):
    """A vocabulary, table, formula, freeze, or decision is non-canonical."""


class _NoBool:
    def __bool__(self) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is not a Boolean value")


class FeatureSupportGapKind(str, Enum):
    NATIVE_MISS = "native_miss"
    UNCERTIFIED_CONTRAST = "uncertified_contrast"
    ERROR = "error"
    NO_SEPARATOR = "no_separator"


_GAP_KIND_ORDER = tuple(FeatureSupportGapKind)


class FeatureQueryOutcome(_NoBool, str, Enum):
    SIDE0 = "side0"
    SIDE1 = "side1"
    ABSTAIN = "abstain"
    ERROR = "error"


class EngineeringDisposition(_NoBool, str, Enum):
    """Uncalibrated repeated-observer consensus, never scientific evidence."""

    MATCH = "match"
    NONMATCH = "nonmatch"
    INDETERMINATE = "indeterminate"
    ERROR = "error"

    @property
    def engineering_only(self) -> bool:
        return True

    @property
    def uncalibrated(self) -> bool:
        return True


class EngineeringQueryOutcome(_NoBool, str, Enum):
    SIDE0 = "side0"
    SIDE1 = "side1"
    ABSTAIN = "abstain"
    ERROR = "error"


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise PanelFeaturePredicateError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise PanelFeaturePredicateError(f"{label} must be a lowercase SHA-256")
    return value


def _orientation(value: object) -> NativeOrientation:
    if type(value) is not NativeOrientation:
        raise TypeError("native orientation must be NativeOrientation")
    return value


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "execution_semantics": "closed-positive-all-of-python-v1",
    }


def _language_data() -> dict[str, object]:
    return {
        "operator": "all_of",
        "minimum_atoms": 1,
        "maximum_atoms": PANEL_FEATURE_MAX_CONJUNCTION,
        "negation_allowed": False,
        "complement_allowed": False,
        "polarity_flip_allowed": False,
        "arbitrary_code_allowed": False,
    }


def _engineering_data() -> dict[str, object]:
    return {
        "engineering_only": True,
        "uncalibrated": True,
        "scientific_evidence": False,
        "benchmark_authoritative": False,
    }


def panel_feature_predicate_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def panel_feature_predicate_algorithm_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-feature-predicate-algorithm.v1",
            "algorithm_id": PANEL_FEATURE_ALGORITHM_ID,
            "implementation_source_sha256": panel_feature_predicate_source_digest(),
            "feature_catalog_digest": feature_catalog_digest(),
            "support_panels_per_side": PANEL_FEATURE_SUPPORTS_PER_SIDE,
            "scientific_positive": Disposition.PRESENT.value,
            "scientific_negative": Disposition.CERTIFIED_ABSENT.value,
            "engineering_positive": EngineeringDisposition.MATCH.value,
            "engineering_negative": EngineeringDisposition.NONMATCH.value,
            **_language_data(),
            **_authority_data(),
        }
    )


def _vocabulary_content(value: "FeatureVocabulary") -> dict[str, object]:
    return {
        "schema": FEATURE_VOCABULARY_SCHEMA,
        "feature_catalog_digest": feature_catalog_digest(),
        "specs": [item.to_data() for item in value.specs],
        "side0_native_spec_digests": list(value.side0_native_spec_digests),
        "side1_native_spec_digests": list(value.side1_native_spec_digests),
        "deduplication_key": "exact-feature-spec-digest",
        **_authority_data(),
    }


def _deduplicate_specs(specs: Sequence[PanelFeatureSpec]) -> tuple[PanelFeatureSpec, ...]:
    by_digest: dict[str, PanelFeatureSpec] = {}
    for spec in specs:
        if type(spec) is not PanelFeatureSpec:
            raise TypeError("feature vocabulary accepts exact PanelFeatureSpec values")
        previous = by_digest.setdefault(spec.spec_digest, spec)
        if previous.to_data() != spec.to_data():  # SHA-256 collision guard.
            raise PanelFeaturePredicateError("feature spec digest collision")
    return tuple(by_digest[key] for key in sorted(by_digest))


@dataclass(frozen=True, slots=True)
class FeatureVocabulary(_NoBool):
    """A canonical global spec inventory with native-orientation membership."""

    specs: tuple[PanelFeatureSpec, ...]
    side0_native_spec_digests: tuple[str, ...]
    side1_native_spec_digests: tuple[str, ...]
    vocabulary_digest: str

    def __post_init__(self) -> None:
        if type(self.specs) is not tuple or not self.specs or any(
            type(item) is not PanelFeatureSpec for item in self.specs
        ):
            raise TypeError("feature vocabulary needs an exact non-empty spec tuple")
        digests = tuple(item.spec_digest for item in self.specs)
        if digests != tuple(sorted(digests)) or len(digests) != len(set(digests)):
            raise PanelFeaturePredicateError("feature specs must be unique and digest-sorted")
        admitted = set(digests)
        for name in ("side0_native_spec_digests", "side1_native_spec_digests"):
            row = getattr(self, name)
            if (
                type(row) is not tuple
                or not row
                or row != tuple(sorted(row))
                or len(row) != len(set(row))
                or not set(row) <= admitted
            ):
                raise PanelFeaturePredicateError(
                    "native feature rows must be non-empty, unique, sorted vocabulary subsets"
                )
        _digest(self.vocabulary_digest, "feature vocabulary digest")
        if self.vocabulary_digest != canonical_digest(_vocabulary_content(self)):
            raise PanelFeaturePredicateError("feature vocabulary digest differs")

    @classmethod
    def create(
        cls,
        *,
        side0_specs: Sequence[PanelFeatureSpec],
        side1_specs: Sequence[PanelFeatureSpec],
    ) -> "FeatureVocabulary":
        side0 = _deduplicate_specs(tuple(side0_specs))
        side1 = _deduplicate_specs(tuple(side1_specs))
        specs = _deduplicate_specs(side0 + side1)
        values = {
            "specs": specs,
            "side0_native_spec_digests": tuple(item.spec_digest for item in side0),
            "side1_native_spec_digests": tuple(item.spec_digest for item in side1),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, vocabulary_digest=canonical_digest(_vocabulary_content(provisional)))

    def native_spec_digests(self, orientation: NativeOrientation) -> tuple[str, ...]:
        _orientation(orientation)
        return (
            self.side0_native_spec_digests
            if orientation is NativeOrientation.SIDE0_POSITIVE
            else self.side1_native_spec_digests
        )

    def to_data(self) -> dict[str, object]:
        return {**_vocabulary_content(self), "vocabulary_digest": self.vocabulary_digest}

    @classmethod
    def from_data(cls, value: object) -> "FeatureVocabulary":
        raw = _fields(
            value,
            {
                "schema",
                "feature_catalog_digest",
                "specs",
                "side0_native_spec_digests",
                "side1_native_spec_digests",
                "deduplication_key",
                *_authority_data(),
                "vocabulary_digest",
            },
            "feature vocabulary",
        )
        if (
            raw["schema"] != FEATURE_VOCABULARY_SCHEMA
            or raw["feature_catalog_digest"] != feature_catalog_digest()
            or raw["deduplication_key"] != "exact-feature-spec-digest"
            or any(raw[key] != item for key, item in _authority_data().items())
            or type(raw["specs"]) is not list
            or type(raw["side0_native_spec_digests"]) is not list
            or type(raw["side1_native_spec_digests"]) is not list
        ):
            raise PanelFeaturePredicateError("feature vocabulary policy differs")
        result = cls(
            tuple(PanelFeatureSpec.from_data(item) for item in raw["specs"]),
            tuple(raw["side0_native_spec_digests"]),
            tuple(raw["side1_native_spec_digests"]),
            raw["vocabulary_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelFeaturePredicateError("feature vocabulary is not canonical")
        return result


def _scientific_cell_content(value: "FeatureSupportCell") -> dict[str, object]:
    return {
        "schema": FEATURE_SUPPORT_CELL_SCHEMA,
        "panel_digest": value.panel_digest,
        "spec_digest": value.spec_digest,
        "disposition": value.disposition.value,
        "semantics": "scientific-disposition",
        **_authority_data(),
    }


@dataclass(frozen=True, order=True, slots=True)
class FeatureSupportCell(_NoBool):
    panel_digest: str
    spec_digest: str
    disposition: Disposition
    cell_digest: str

    def __post_init__(self) -> None:
        _digest(self.panel_digest, "support panel digest")
        _digest(self.spec_digest, "support spec digest")
        if type(self.disposition) is not Disposition:
            raise TypeError("scientific support requires exact Disposition values")
        _digest(self.cell_digest, "support cell digest")
        if self.cell_digest != canonical_digest(_scientific_cell_content(self)):
            raise PanelFeaturePredicateError("support cell digest differs")

    @classmethod
    def create(cls, panel_digest: str, spec_digest: str, disposition: Disposition) -> "FeatureSupportCell":
        values = {
            "panel_digest": _digest(panel_digest, "support panel digest"),
            "spec_digest": _digest(spec_digest, "support spec digest"),
            "disposition": disposition,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, cell_digest=canonical_digest(_scientific_cell_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_scientific_cell_content(self), "cell_digest": self.cell_digest}

    @classmethod
    def from_data(cls, value: object) -> "FeatureSupportCell":
        raw = _fields(
            value,
            {"schema", "panel_digest", "spec_digest", "disposition", "semantics", *_authority_data(), "cell_digest"},
            "feature support cell",
        )
        if raw["schema"] != FEATURE_SUPPORT_CELL_SCHEMA or raw["semantics"] != "scientific-disposition" or any(
            raw[key] != item for key, item in _authority_data().items()
        ):
            raise PanelFeaturePredicateError("support cell policy differs")
        try:
            state = Disposition(raw["disposition"])
        except (TypeError, ValueError) as exc:
            raise PanelFeaturePredicateError("support disposition is unknown") from exc
        result = cls(raw["panel_digest"], raw["spec_digest"], state, raw["cell_digest"])
        if result.to_data() != dict(raw):
            raise PanelFeaturePredicateError("support cell is not canonical")
        return result


def _engineering_cell_content(value: "EngineeringSupportCell") -> dict[str, object]:
    return {
        "schema": ENGINEERING_SUPPORT_CELL_SCHEMA,
        "panel_digest": value.panel_digest,
        "spec_digest": value.spec_digest,
        "disposition": value.disposition.value,
        **_engineering_data(),
        **_authority_data(),
    }


@dataclass(frozen=True, order=True, slots=True)
class EngineeringSupportCell(_NoBool):
    panel_digest: str
    spec_digest: str
    disposition: EngineeringDisposition
    cell_digest: str

    def __post_init__(self) -> None:
        _digest(self.panel_digest, "engineering panel digest")
        _digest(self.spec_digest, "engineering spec digest")
        if type(self.disposition) is not EngineeringDisposition:
            raise TypeError("engineering support requires EngineeringDisposition")
        _digest(self.cell_digest, "engineering cell digest")
        if self.cell_digest != canonical_digest(_engineering_cell_content(self)):
            raise PanelFeaturePredicateError("engineering cell digest differs")

    @classmethod
    def create(cls, panel_digest: str, spec_digest: str, disposition: EngineeringDisposition) -> "EngineeringSupportCell":
        values = {
            "panel_digest": _digest(panel_digest, "engineering panel digest"),
            "spec_digest": _digest(spec_digest, "engineering spec digest"),
            "disposition": disposition,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, cell_digest=canonical_digest(_engineering_cell_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_engineering_cell_content(self), "cell_digest": self.cell_digest}

    @classmethod
    def from_data(cls, value: object) -> "EngineeringSupportCell":
        raw = _fields(
            value,
            {"schema", "panel_digest", "spec_digest", "disposition", *_engineering_data(), *_authority_data(), "cell_digest"},
            "engineering support cell",
        )
        if raw["schema"] != ENGINEERING_SUPPORT_CELL_SCHEMA or any(
            raw[key] != item for key, item in {**_engineering_data(), **_authority_data()}.items()
        ):
            raise PanelFeaturePredicateError("engineering support cell policy differs")
        try:
            state = EngineeringDisposition(raw["disposition"])
        except (TypeError, ValueError) as exc:
            raise PanelFeaturePredicateError("engineering disposition is unknown") from exc
        result = cls(raw["panel_digest"], raw["spec_digest"], state, raw["cell_digest"])
        if result.to_data() != dict(raw):
            raise PanelFeaturePredicateError("engineering support cell is not canonical")
        return result


def _table_content(value: object, *, engineering: bool) -> dict[str, object]:
    schema = ENGINEERING_SUPPORT_TABLE_SCHEMA if engineering else FEATURE_SUPPORT_TABLE_SCHEMA
    result = {
        "schema": schema,
        "vocabulary": value.vocabulary.to_data(),
        "panel_digests": list(value.panel_digests),
        "cells": [item.to_data() for item in value.cells],
        "cell_order": "panel-digest-then-spec-digest",
        "complete_panel_spec_matrix": True,
        **_authority_data(),
    }
    if engineering:
        result.update(_engineering_data())
    else:
        result["semantics"] = "scientific-disposition"
    return result


def _validate_table_shape(value: object, cell_type: type) -> None:
    if type(value.vocabulary) is not FeatureVocabulary:
        raise TypeError("support table vocabulary must be FeatureVocabulary")
    if (
        type(value.panel_digests) is not tuple
        or not value.panel_digests
        or value.panel_digests != tuple(sorted(value.panel_digests))
        or len(value.panel_digests) != len(set(value.panel_digests))
    ):
        raise PanelFeaturePredicateError("panel digests must be non-empty, unique, and sorted")
    for item in value.panel_digests:
        _digest(item, "table panel digest")
    if type(value.cells) is not tuple or any(type(item) is not cell_type for item in value.cells):
        raise TypeError("support table cells have the wrong type")
    expected = tuple(
        (panel, spec.spec_digest)
        for panel in value.panel_digests
        for spec in value.vocabulary.specs
    )
    actual = tuple((item.panel_digest, item.spec_digest) for item in value.cells)
    if actual != expected:
        raise PanelFeaturePredicateError("support table is not the exact complete panel/spec matrix")


def _make_cells(vocabulary: FeatureVocabulary, panels: tuple[str, ...], values: Mapping[tuple[str, str], object], *, engineering: bool) -> tuple[object, ...]:
    if not isinstance(values, Mapping):
        raise TypeError("support values must be a mapping")
    expected = {(panel, spec.spec_digest) for panel in panels for spec in vocabulary.specs}
    if set(values) != expected or any(type(key) is not tuple or len(key) != 2 for key in values):
        raise PanelFeaturePredicateError("support values must exactly cover every panel/spec key")
    factory = EngineeringSupportCell.create if engineering else FeatureSupportCell.create
    return tuple(factory(panel, spec.spec_digest, values[(panel, spec.spec_digest)]) for panel in panels for spec in vocabulary.specs)


@dataclass(frozen=True, slots=True)
class FeatureSupportTable(_NoBool):
    vocabulary: FeatureVocabulary
    panel_digests: tuple[str, ...]
    cells: tuple[FeatureSupportCell, ...]
    table_digest: str

    def __post_init__(self) -> None:
        _validate_table_shape(self, FeatureSupportCell)
        _digest(self.table_digest, "support table digest")
        if self.table_digest != canonical_digest(_table_content(self, engineering=False)):
            raise PanelFeaturePredicateError("support table digest differs")

    @classmethod
    def create(cls, vocabulary: FeatureVocabulary, panel_digests: Sequence[str], values: Mapping[tuple[str, str], Disposition]) -> "FeatureSupportTable":
        if type(vocabulary) is not FeatureVocabulary:
            raise TypeError("support table vocabulary must be FeatureVocabulary")
        panels = tuple(sorted(panel_digests))
        cells = _make_cells(vocabulary, panels, values, engineering=False)
        values_ = {"vocabulary": vocabulary, "panel_digests": panels, "cells": cells}
        provisional = object.__new__(cls)
        for name, item in values_.items():
            object.__setattr__(provisional, name, item)
        return cls(**values_, table_digest=canonical_digest(_table_content(provisional, engineering=False)))

    def disposition(self, panel_digest: str, spec_digest: str) -> Disposition:
        key = (_digest(panel_digest, "panel digest"), _digest(spec_digest, "spec digest"))
        try:
            return next(item.disposition for item in self.cells if (item.panel_digest, item.spec_digest) == key)
        except StopIteration as exc:
            raise PanelFeaturePredicateError("panel/spec key is absent") from exc

    def to_data(self) -> dict[str, object]:
        return {**_table_content(self, engineering=False), "table_digest": self.table_digest}

    @classmethod
    def from_data(cls, value: object) -> "FeatureSupportTable":
        raw = _fields(value, {"schema", "vocabulary", "panel_digests", "cells", "cell_order", "complete_panel_spec_matrix", "semantics", *_authority_data(), "table_digest"}, "feature support table")
        if raw["schema"] != FEATURE_SUPPORT_TABLE_SCHEMA or raw["cell_order"] != "panel-digest-then-spec-digest" or raw["complete_panel_spec_matrix"] is not True or raw["semantics"] != "scientific-disposition" or any(raw[key] != item for key, item in _authority_data().items()) or type(raw["panel_digests"]) is not list or type(raw["cells"]) is not list:
            raise PanelFeaturePredicateError("feature support table policy differs")
        result = cls(FeatureVocabulary.from_data(raw["vocabulary"]), tuple(raw["panel_digests"]), tuple(FeatureSupportCell.from_data(item) for item in raw["cells"]), raw["table_digest"])
        if result.to_data() != dict(raw):
            raise PanelFeaturePredicateError("feature support table is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class EngineeringSupportTable(_NoBool):
    vocabulary: FeatureVocabulary
    panel_digests: tuple[str, ...]
    cells: tuple[EngineeringSupportCell, ...]
    table_digest: str

    def __post_init__(self) -> None:
        _validate_table_shape(self, EngineeringSupportCell)
        _digest(self.table_digest, "engineering table digest")
        if self.table_digest != canonical_digest(_table_content(self, engineering=True)):
            raise PanelFeaturePredicateError("engineering table digest differs")

    @classmethod
    def create(cls, vocabulary: FeatureVocabulary, panel_digests: Sequence[str], values: Mapping[tuple[str, str], EngineeringDisposition]) -> "EngineeringSupportTable":
        if type(vocabulary) is not FeatureVocabulary:
            raise TypeError("engineering table vocabulary must be FeatureVocabulary")
        panels = tuple(sorted(panel_digests))
        cells = _make_cells(vocabulary, panels, values, engineering=True)
        values_ = {"vocabulary": vocabulary, "panel_digests": panels, "cells": cells}
        provisional = object.__new__(cls)
        for name, item in values_.items():
            object.__setattr__(provisional, name, item)
        return cls(**values_, table_digest=canonical_digest(_table_content(provisional, engineering=True)))

    def disposition(self, panel_digest: str, spec_digest: str) -> EngineeringDisposition:
        key = (_digest(panel_digest, "panel digest"), _digest(spec_digest, "spec digest"))
        try:
            return next(item.disposition for item in self.cells if (item.panel_digest, item.spec_digest) == key)
        except StopIteration as exc:
            raise PanelFeaturePredicateError("engineering panel/spec key is absent") from exc

    def to_data(self) -> dict[str, object]:
        return {**_table_content(self, engineering=True), "table_digest": self.table_digest}

    @classmethod
    def from_data(cls, value: object) -> "EngineeringSupportTable":
        raw = _fields(value, {"schema", "vocabulary", "panel_digests", "cells", "cell_order", "complete_panel_spec_matrix", *_engineering_data(), *_authority_data(), "table_digest"}, "engineering support table")
        policy = {**_engineering_data(), **_authority_data()}
        if raw["schema"] != ENGINEERING_SUPPORT_TABLE_SCHEMA or raw["cell_order"] != "panel-digest-then-spec-digest" or raw["complete_panel_spec_matrix"] is not True or any(raw[key] != item for key, item in policy.items()) or type(raw["panel_digests"]) is not list or type(raw["cells"]) is not list:
            raise PanelFeaturePredicateError("engineering support table policy differs")
        result = cls(FeatureVocabulary.from_data(raw["vocabulary"]), tuple(raw["panel_digests"]), tuple(EngineeringSupportCell.from_data(item) for item in raw["cells"]), raw["table_digest"])
        if result.to_data() != dict(raw):
            raise PanelFeaturePredicateError("engineering support table is not canonical")
        return result


def _formula_content(value: "AllOf") -> dict[str, object]:
    return {
        "schema": ALL_OF_SCHEMA,
        "algorithm_digest": panel_feature_predicate_algorithm_digest(),
        "vocabulary_digest": value.vocabulary_digest,
        "native_orientation": value.native_orientation.value,
        "spec_digests": list(value.spec_digests),
        **_language_data(),
        **_authority_data(),
    }


@dataclass(frozen=True, order=True, slots=True)
class AllOf(_NoBool):
    vocabulary_digest: str
    native_orientation: NativeOrientation
    spec_digests: tuple[str, ...]
    formula_digest: str

    def __post_init__(self) -> None:
        _digest(self.vocabulary_digest, "formula vocabulary digest")
        _orientation(self.native_orientation)
        if type(self.spec_digests) is not tuple or not 1 <= len(self.spec_digests) <= PANEL_FEATURE_MAX_CONJUNCTION or self.spec_digests != tuple(sorted(self.spec_digests)) or len(self.spec_digests) != len(set(self.spec_digests)):
            raise PanelFeaturePredicateError("AllOf atoms must be one or two unique sorted specs")
        for item in self.spec_digests:
            _digest(item, "formula spec digest")
        _digest(self.formula_digest, "formula digest")
        if self.formula_digest != canonical_digest(_formula_content(self)):
            raise PanelFeaturePredicateError("formula digest differs")

    @classmethod
    def create(cls, vocabulary: FeatureVocabulary, native_orientation: NativeOrientation, spec_digests: Sequence[str]) -> "AllOf":
        if type(vocabulary) is not FeatureVocabulary:
            raise TypeError("formula vocabulary must be FeatureVocabulary")
        _orientation(native_orientation)
        row = tuple(spec_digests)
        if row != tuple(sorted(row)) or not set(row) <= set(vocabulary.native_spec_digests(native_orientation)):
            raise PanelFeaturePredicateError("formula atoms are not canonical native vocabulary members")
        values = {"vocabulary_digest": vocabulary.vocabulary_digest, "native_orientation": native_orientation, "spec_digests": row}
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, formula_digest=canonical_digest(_formula_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_formula_content(self), "formula_digest": self.formula_digest}

    @classmethod
    def from_data(cls, value: object) -> "AllOf":
        raw = _fields(value, {"schema", "algorithm_digest", "vocabulary_digest", "native_orientation", "spec_digests", *_language_data(), *_authority_data(), "formula_digest"}, "AllOf formula")
        if raw["schema"] != ALL_OF_SCHEMA or raw["algorithm_digest"] != panel_feature_predicate_algorithm_digest() or any(raw[key] != item for key, item in {**_language_data(), **_authority_data()}.items()) or type(raw["spec_digests"]) is not list:
            raise PanelFeaturePredicateError("formula policy differs")
        try:
            orientation = NativeOrientation(raw["native_orientation"])
        except (TypeError, ValueError) as exc:
            raise PanelFeaturePredicateError("formula orientation is unknown") from exc
        result = cls(raw["vocabulary_digest"], orientation, tuple(raw["spec_digests"]), raw["formula_digest"])
        if result.to_data() != dict(raw):
            raise PanelFeaturePredicateError("formula is not canonical")
        return result


def enumerate_all_of(vocabulary: FeatureVocabulary, native_orientation: NativeOrientation) -> tuple[AllOf, ...]:
    if type(vocabulary) is not FeatureVocabulary:
        raise TypeError("formula inventory vocabulary must be FeatureVocabulary")
    atoms = vocabulary.native_spec_digests(_orientation(native_orientation))
    return tuple(AllOf.create(vocabulary, native_orientation, row) for size in range(1, min(PANEL_FEATURE_MAX_CONJUNCTION, len(atoms)) + 1) for row in combinations(atoms, size))


def _and_scientific(values: Sequence[Disposition]) -> Disposition:
    row = tuple(values)
    if not row:
        raise PanelFeaturePredicateError("AllOf cannot be empty")
    if Disposition.ERROR in row:
        return Disposition.ERROR
    if Disposition.CERTIFIED_ABSENT in row:
        return Disposition.CERTIFIED_ABSENT
    if all(item is Disposition.PRESENT for item in row):
        return Disposition.PRESENT
    return Disposition.INDETERMINATE


def _and_engineering(values: Sequence[EngineeringDisposition]) -> EngineeringDisposition:
    row = tuple(values)
    if not row:
        raise PanelFeaturePredicateError("AllOf cannot be empty")
    if EngineeringDisposition.ERROR in row:
        return EngineeringDisposition.ERROR
    if EngineeringDisposition.NONMATCH in row:
        return EngineeringDisposition.NONMATCH
    if all(item is EngineeringDisposition.MATCH for item in row):
        return EngineeringDisposition.MATCH
    return EngineeringDisposition.INDETERMINATE


def evaluate_all_of(formula: AllOf, table: FeatureSupportTable, panel_digest: str) -> Disposition:
    if type(formula) is not AllOf or type(table) is not FeatureSupportTable:
        raise TypeError("scientific evaluation requires AllOf and FeatureSupportTable")
    if formula.vocabulary_digest != table.vocabulary.vocabulary_digest or not set(formula.spec_digests) <= set(table.vocabulary.native_spec_digests(formula.native_orientation)):
        raise PanelFeaturePredicateError("formula and scientific table vocabulary differ")
    panel = _digest(panel_digest, "evaluation panel digest")
    return _and_scientific(tuple(table.disposition(panel, spec) for spec in formula.spec_digests))


def evaluate_engineering_all_of(formula: AllOf, table: EngineeringSupportTable, panel_digest: str) -> EngineeringDisposition:
    if type(formula) is not AllOf or type(table) is not EngineeringSupportTable:
        raise TypeError("engineering evaluation requires AllOf and EngineeringSupportTable")
    if formula.vocabulary_digest != table.vocabulary.vocabulary_digest or not set(formula.spec_digests) <= set(table.vocabulary.native_spec_digests(formula.native_orientation)):
        raise PanelFeaturePredicateError("formula and engineering table vocabulary differ")
    panel = _digest(panel_digest, "engineering evaluation panel digest")
    return _and_engineering(tuple(table.disposition(panel, spec) for spec in formula.spec_digests))


@dataclass(frozen=True, slots=True)
class FormulaGapDiagnostic(_NoBool):
    formula_digest: str
    kinds: tuple[FeatureSupportGapKind, ...]
    native_miss_panel_digests: tuple[str, ...]
    uncertified_contrast_panel_digests: tuple[str, ...]
    error_panel_digests: tuple[str, ...]
    no_separator_panel_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        _digest(self.formula_digest, "gap formula digest")
        if type(self.kinds) is not tuple or not self.kinds or any(type(item) is not FeatureSupportGapKind for item in self.kinds) or self.kinds != tuple(item for item in _GAP_KIND_ORDER if item in set(self.kinds)):
            raise PanelFeaturePredicateError("gap kinds must be non-empty, unique, and canonical")
        inventories = []
        for name in ("native_miss_panel_digests", "uncertified_contrast_panel_digests", "error_panel_digests", "no_separator_panel_digests"):
            row = getattr(self, name)
            if type(row) is not tuple or row != tuple(sorted(row)) or len(row) != len(set(row)):
                raise PanelFeaturePredicateError("gap panel inventories must be unique and sorted")
            for item in row:
                _digest(item, "gap panel digest")
            inventories.append(set(row))
        if any(inventories[left] & inventories[right] for left in range(4) for right in range(left + 1, 4)):
            raise PanelFeaturePredicateError("gap panel categories overlap")
        expected = tuple(kind for kind, row in zip(_GAP_KIND_ORDER, inventories, strict=True) if row)
        if self.kinds != expected:
            raise PanelFeaturePredicateError("gap kinds differ from panel categories")

    def to_data(self) -> dict[str, object]:
        return {"schema": FORMULA_GAP_DIAGNOSTIC_SCHEMA, "formula_digest": self.formula_digest, "kinds": [item.value for item in self.kinds], "native_miss_panel_digests": list(self.native_miss_panel_digests), "uncertified_contrast_panel_digests": list(self.uncertified_contrast_panel_digests), "error_panel_digests": list(self.error_panel_digests), "no_separator_panel_digests": list(self.no_separator_panel_digests)}

    @classmethod
    def from_data(cls, value: object) -> "FormulaGapDiagnostic":
        raw = _fields(value, {"schema", "formula_digest", "kinds", "native_miss_panel_digests", "uncertified_contrast_panel_digests", "error_panel_digests", "no_separator_panel_digests"}, "formula gap diagnostic")
        if raw["schema"] != FORMULA_GAP_DIAGNOSTIC_SCHEMA or any(type(raw[name]) is not list for name in ("kinds", "native_miss_panel_digests", "uncertified_contrast_panel_digests", "error_panel_digests", "no_separator_panel_digests")):
            raise PanelFeaturePredicateError("gap diagnostic policy differs")
        try:
            kinds = tuple(FeatureSupportGapKind(item) for item in raw["kinds"])
        except (TypeError, ValueError) as exc:
            raise PanelFeaturePredicateError("gap kind is unknown") from exc
        result = cls(raw["formula_digest"], kinds, tuple(raw["native_miss_panel_digests"]), tuple(raw["uncertified_contrast_panel_digests"]), tuple(raw["error_panel_digests"]), tuple(raw["no_separator_panel_digests"]))
        if result.to_data() != dict(raw):
            raise PanelFeaturePredicateError("gap diagnostic is not canonical")
        return result


def _gap_content(value: "FeatureSupportGap") -> dict[str, object]:
    return {"schema": FEATURE_SUPPORT_GAP_SCHEMA, "kinds": [item.value for item in value.kinds], "diagnostics": [item.to_data() for item in value.diagnostics], **_authority_data()}


@dataclass(frozen=True, slots=True)
class FeatureSupportGap(_NoBool):
    kinds: tuple[FeatureSupportGapKind, ...]
    diagnostics: tuple[FormulaGapDiagnostic, ...]
    gap_digest: str

    def __post_init__(self) -> None:
        if type(self.diagnostics) is not tuple or not self.diagnostics or any(type(item) is not FormulaGapDiagnostic for item in self.diagnostics) or tuple(item.formula_digest for item in self.diagnostics) != tuple(sorted(item.formula_digest for item in self.diagnostics)):
            raise PanelFeaturePredicateError("gap diagnostics must be non-empty and formula-sorted")
        expected = tuple(item for item in _GAP_KIND_ORDER if any(item in row.kinds for row in self.diagnostics))
        if self.kinds != expected:
            raise PanelFeaturePredicateError("aggregate gap kinds differ")
        _digest(self.gap_digest, "support gap digest")
        if self.gap_digest != canonical_digest(_gap_content(self)):
            raise PanelFeaturePredicateError("support gap digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_gap_content(self), "gap_digest": self.gap_digest}

    @classmethod
    def from_data(cls, value: object) -> "FeatureSupportGap":
        raw = _fields(value, {"schema", "kinds", "diagnostics", *_authority_data(), "gap_digest"}, "feature support gap")
        if raw["schema"] != FEATURE_SUPPORT_GAP_SCHEMA or any(raw[key] != item for key, item in _authority_data().items()) or type(raw["kinds"]) is not list or type(raw["diagnostics"]) is not list:
            raise PanelFeaturePredicateError("support gap policy differs")
        try:
            kinds = tuple(FeatureSupportGapKind(item) for item in raw["kinds"])
        except (TypeError, ValueError) as exc:
            raise PanelFeaturePredicateError("support gap kind is unknown") from exc
        result = cls(kinds, tuple(FormulaGapDiagnostic.from_data(item) for item in raw["diagnostics"]), raw["gap_digest"])
        if result.to_data() != dict(raw):
            raise PanelFeaturePredicateError("support gap is not canonical")
        return result


def _panels_for_orientation(orientation: NativeOrientation, side0: tuple[str, ...], side1: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (side0, side1) if orientation is NativeOrientation.SIDE0_POSITIVE else (side1, side0)


def _validate_support_sides(table: object, side0: tuple[str, ...], side1: tuple[str, ...]) -> None:
    for row in (side0, side1):
        if type(row) is not tuple or len(row) != PANEL_FEATURE_SUPPORTS_PER_SIDE or row != tuple(sorted(row)) or len(row) != len(set(row)):
            raise PanelFeaturePredicateError("support needs exactly six unique sorted panels per side")
        for item in row:
            _digest(item, "support-side panel digest")
    if set(side0) & set(side1) or set(table.panel_digests) != set(side0 + side1):
        raise PanelFeaturePredicateError("support-side inventory differs from exact table panels")


def _scientific_diagnostic(formula: AllOf, row: tuple[Disposition, ...], side0: tuple[str, ...], side1: tuple[str, ...]) -> FormulaGapDiagnostic:
    native, contrast = _panels_for_orientation(formula.native_orientation, side0, side1)
    ordered_panels = side0 + side1
    by_panel = dict(zip(ordered_panels, row, strict=True))
    error = tuple(sorted(panel for panel in ordered_panels if by_panel[panel] is Disposition.ERROR))
    native_miss = tuple(sorted(panel for panel in native if by_panel[panel] in {Disposition.CERTIFIED_ABSENT, Disposition.INDETERMINATE}))
    uncertified = tuple(sorted(panel for panel in contrast if by_panel[panel] is Disposition.INDETERMINATE))
    no_separator = tuple(sorted(panel for panel in contrast if by_panel[panel] is Disposition.PRESENT))
    inventories = (native_miss, uncertified, error, no_separator)
    kinds = tuple(kind for kind, panels in zip(_GAP_KIND_ORDER, inventories, strict=True) if panels)
    return FormulaGapDiagnostic(formula.formula_digest, kinds, *inventories)


def _make_gap(formulas: tuple[AllOf, ...], rows: tuple[tuple[Disposition, ...], ...], side0: tuple[str, ...], side1: tuple[str, ...]) -> FeatureSupportGap:
    diagnostics = tuple(sorted((_scientific_diagnostic(formula, row, side0, side1) for formula, row in zip(formulas, rows, strict=True)), key=lambda item: item.formula_digest))
    kinds = tuple(item for item in _GAP_KIND_ORDER if any(item in row.kinds for row in diagnostics))
    provisional = object.__new__(FeatureSupportGap)
    object.__setattr__(provisional, "kinds", kinds)
    object.__setattr__(provisional, "diagnostics", diagnostics)
    return FeatureSupportGap(kinds, diagnostics, canonical_digest(_gap_content(provisional)))


def _version_content(value: object, *, engineering: bool) -> dict[str, object]:
    result = {
        "schema": ENGINEERING_VERSION_SPACE_SCHEMA if engineering else FEATURE_VERSION_SPACE_SCHEMA,
        "algorithm_id": PANEL_FEATURE_ALGORITHM_ID,
        "algorithm_digest": panel_feature_predicate_algorithm_digest(),
        "support_table": value.support_table.to_data(),
        "native_orientation": value.native_orientation.value,
        "side0_panel_digests": list(value.side0_panel_digests),
        "side1_panel_digests": list(value.side1_panel_digests),
        "formulas": [item.to_data() for item in value.formulas],
        "rows": [[item.value for item in row] for row in value.rows],
        "survivor_formula_digests": list(value.survivor_formula_digests),
        "support_rule": "positive-on-all-six-native-and-negative-on-all-six-contrast",
        **_language_data(),
        **_authority_data(),
    }
    if engineering:
        result.update(_engineering_data())
    else:
        result["gap"] = None if value.gap is None else value.gap.to_data()
        result["semantics"] = "scientific-disposition"
    return result


def _scientific_survives(formula: AllOf, row: tuple[Disposition, ...], side0: tuple[str, ...], side1: tuple[str, ...]) -> bool:
    native, contrast = _panels_for_orientation(formula.native_orientation, side0, side1)
    by_panel = dict(zip(side0 + side1, row, strict=True))
    return all(by_panel[item] is Disposition.PRESENT for item in native) and all(by_panel[item] is Disposition.CERTIFIED_ABSENT for item in contrast)


def _engineering_survives(formula: AllOf, row: tuple[EngineeringDisposition, ...], side0: tuple[str, ...], side1: tuple[str, ...]) -> bool:
    native, contrast = _panels_for_orientation(formula.native_orientation, side0, side1)
    by_panel = dict(zip(side0 + side1, row, strict=True))
    return all(by_panel[item] is EngineeringDisposition.MATCH for item in native) and all(by_panel[item] is EngineeringDisposition.NONMATCH for item in contrast)


@dataclass(frozen=True, slots=True)
class FeatureVersionSpace(_NoBool):
    support_table: FeatureSupportTable
    native_orientation: NativeOrientation
    side0_panel_digests: tuple[str, ...]
    side1_panel_digests: tuple[str, ...]
    formulas: tuple[AllOf, ...]
    rows: tuple[tuple[Disposition, ...], ...]
    survivor_formula_digests: tuple[str, ...]
    gap: FeatureSupportGap | None
    version_space_digest: str

    def __post_init__(self) -> None:
        if type(self.support_table) is not FeatureSupportTable:
            raise TypeError("scientific version space requires FeatureSupportTable")
        _orientation(self.native_orientation)
        _validate_support_sides(self.support_table, self.side0_panel_digests, self.side1_panel_digests)
        expected_formulas = enumerate_all_of(self.support_table.vocabulary, self.native_orientation)
        if self.formulas != expected_formulas:
            raise PanelFeaturePredicateError("scientific formula inventory differs")
        expected_rows = tuple(tuple(evaluate_all_of(formula, self.support_table, panel) for panel in self.side0_panel_digests + self.side1_panel_digests) for formula in expected_formulas)
        if self.rows != expected_rows:
            raise PanelFeaturePredicateError("scientific version-space rows differ")
        survivors = tuple(formula.formula_digest for formula, row in zip(expected_formulas, expected_rows, strict=True) if _scientific_survives(formula, row, self.side0_panel_digests, self.side1_panel_digests))
        if self.survivor_formula_digests != survivors:
            raise PanelFeaturePredicateError("scientific survivors differ")
        expected_gap = None if survivors else _make_gap(expected_formulas, expected_rows, self.side0_panel_digests, self.side1_panel_digests)
        if self.gap != expected_gap:
            raise PanelFeaturePredicateError("scientific typed gap differs")
        _digest(self.version_space_digest, "scientific version-space digest")
        if self.version_space_digest != canonical_digest(_version_content(self, engineering=False)):
            raise PanelFeaturePredicateError("scientific version-space digest differs")

    @classmethod
    def create(cls, support_table: FeatureSupportTable, native_orientation: NativeOrientation, side0_panel_digests: Sequence[str], side1_panel_digests: Sequence[str]) -> "FeatureVersionSpace":
        side0, side1 = tuple(sorted(side0_panel_digests)), tuple(sorted(side1_panel_digests))
        formulas = enumerate_all_of(support_table.vocabulary, _orientation(native_orientation))
        rows = tuple(tuple(evaluate_all_of(formula, support_table, panel) for panel in side0 + side1) for formula in formulas)
        survivors = tuple(formula.formula_digest for formula, row in zip(formulas, rows, strict=True) if _scientific_survives(formula, row, side0, side1))
        gap = None if survivors else _make_gap(formulas, rows, side0, side1)
        values = {"support_table": support_table, "native_orientation": native_orientation, "side0_panel_digests": side0, "side1_panel_digests": side1, "formulas": formulas, "rows": rows, "survivor_formula_digests": survivors, "gap": gap}
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, version_space_digest=canonical_digest(_version_content(provisional, engineering=False)))

    @property
    def survivor_formulas(self) -> tuple[AllOf, ...]:
        admitted = set(self.survivor_formula_digests)
        return tuple(item for item in self.formulas if item.formula_digest in admitted)

    def to_data(self) -> dict[str, object]:
        return {**_version_content(self, engineering=False), "version_space_digest": self.version_space_digest}

    @classmethod
    def from_data(cls, value: object) -> "FeatureVersionSpace":
        expected = {"schema", "algorithm_id", "algorithm_digest", "support_table", "native_orientation", "side0_panel_digests", "side1_panel_digests", "formulas", "rows", "survivor_formula_digests", "support_rule", "gap", "semantics", *_language_data(), *_authority_data(), "version_space_digest"}
        raw = _fields(value, expected, "feature version space")
        if raw["schema"] != FEATURE_VERSION_SPACE_SCHEMA or raw["algorithm_id"] != PANEL_FEATURE_ALGORITHM_ID or raw["algorithm_digest"] != panel_feature_predicate_algorithm_digest() or raw["support_rule"] != "positive-on-all-six-native-and-negative-on-all-six-contrast" or raw["semantics"] != "scientific-disposition" or any(raw[key] != item for key, item in {**_language_data(), **_authority_data()}.items()) or any(type(raw[name]) is not list for name in ("side0_panel_digests", "side1_panel_digests", "formulas", "rows", "survivor_formula_digests")):
            raise PanelFeaturePredicateError("scientific version-space policy differs")
        try:
            orientation = NativeOrientation(raw["native_orientation"])
            rows = tuple(tuple(Disposition(item) for item in row) for row in raw["rows"])
        except (TypeError, ValueError) as exc:
            raise PanelFeaturePredicateError("scientific version-space enum differs") from exc
        result = cls(FeatureSupportTable.from_data(raw["support_table"]), orientation, tuple(raw["side0_panel_digests"]), tuple(raw["side1_panel_digests"]), tuple(AllOf.from_data(item) for item in raw["formulas"]), rows, tuple(raw["survivor_formula_digests"]), None if raw["gap"] is None else FeatureSupportGap.from_data(raw["gap"]), raw["version_space_digest"])
        if result.to_data() != dict(raw):
            raise PanelFeaturePredicateError("scientific version space is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class EngineeringFeatureVersionSpace(_NoBool):
    support_table: EngineeringSupportTable
    native_orientation: NativeOrientation
    side0_panel_digests: tuple[str, ...]
    side1_panel_digests: tuple[str, ...]
    formulas: tuple[AllOf, ...]
    rows: tuple[tuple[EngineeringDisposition, ...], ...]
    survivor_formula_digests: tuple[str, ...]
    version_space_digest: str

    def __post_init__(self) -> None:
        if type(self.support_table) is not EngineeringSupportTable:
            raise TypeError("engineering version space requires EngineeringSupportTable")
        _orientation(self.native_orientation)
        _validate_support_sides(self.support_table, self.side0_panel_digests, self.side1_panel_digests)
        expected_formulas = enumerate_all_of(self.support_table.vocabulary, self.native_orientation)
        expected_rows = tuple(tuple(evaluate_engineering_all_of(formula, self.support_table, panel) for panel in self.side0_panel_digests + self.side1_panel_digests) for formula in expected_formulas)
        if self.formulas != expected_formulas or self.rows != expected_rows:
            raise PanelFeaturePredicateError("engineering inventory or rows differ")
        survivors = tuple(formula.formula_digest for formula, row in zip(expected_formulas, expected_rows, strict=True) if _engineering_survives(formula, row, self.side0_panel_digests, self.side1_panel_digests))
        if self.survivor_formula_digests != survivors:
            raise PanelFeaturePredicateError("engineering survivors differ")
        _digest(self.version_space_digest, "engineering version-space digest")
        if self.version_space_digest != canonical_digest(_version_content(self, engineering=True)):
            raise PanelFeaturePredicateError("engineering version-space digest differs")

    @classmethod
    def create(cls, support_table: EngineeringSupportTable, native_orientation: NativeOrientation, side0_panel_digests: Sequence[str], side1_panel_digests: Sequence[str]) -> "EngineeringFeatureVersionSpace":
        side0, side1 = tuple(sorted(side0_panel_digests)), tuple(sorted(side1_panel_digests))
        formulas = enumerate_all_of(support_table.vocabulary, _orientation(native_orientation))
        rows = tuple(tuple(evaluate_engineering_all_of(formula, support_table, panel) for panel in side0 + side1) for formula in formulas)
        survivors = tuple(formula.formula_digest for formula, row in zip(formulas, rows, strict=True) if _engineering_survives(formula, row, side0, side1))
        values = {"support_table": support_table, "native_orientation": native_orientation, "side0_panel_digests": side0, "side1_panel_digests": side1, "formulas": formulas, "rows": rows, "survivor_formula_digests": survivors}
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, version_space_digest=canonical_digest(_version_content(provisional, engineering=True)))

    @property
    def survivor_formulas(self) -> tuple[AllOf, ...]:
        admitted = set(self.survivor_formula_digests)
        return tuple(item for item in self.formulas if item.formula_digest in admitted)

    def to_data(self) -> dict[str, object]:
        return {**_version_content(self, engineering=True), "version_space_digest": self.version_space_digest}

    @classmethod
    def from_data(cls, value: object) -> "EngineeringFeatureVersionSpace":
        expected = {"schema", "algorithm_id", "algorithm_digest", "support_table", "native_orientation", "side0_panel_digests", "side1_panel_digests", "formulas", "rows", "survivor_formula_digests", "support_rule", *_language_data(), *_engineering_data(), *_authority_data(), "version_space_digest"}
        raw = _fields(value, expected, "engineering version space")
        policy = {**_language_data(), **_engineering_data(), **_authority_data()}
        if raw["schema"] != ENGINEERING_VERSION_SPACE_SCHEMA or raw["algorithm_id"] != PANEL_FEATURE_ALGORITHM_ID or raw["algorithm_digest"] != panel_feature_predicate_algorithm_digest() or raw["support_rule"] != "positive-on-all-six-native-and-negative-on-all-six-contrast" or any(raw[key] != item for key, item in policy.items()) or any(type(raw[name]) is not list for name in ("side0_panel_digests", "side1_panel_digests", "formulas", "rows", "survivor_formula_digests")):
            raise PanelFeaturePredicateError("engineering version-space policy differs")
        try:
            orientation = NativeOrientation(raw["native_orientation"])
            rows = tuple(tuple(EngineeringDisposition(item) for item in row) for row in raw["rows"])
        except (TypeError, ValueError) as exc:
            raise PanelFeaturePredicateError("engineering version-space enum differs") from exc
        result = cls(EngineeringSupportTable.from_data(raw["support_table"]), orientation, tuple(raw["side0_panel_digests"]), tuple(raw["side1_panel_digests"]), tuple(AllOf.from_data(item) for item in raw["formulas"]), rows, tuple(raw["survivor_formula_digests"]), raw["version_space_digest"])
        if result.to_data() != dict(raw):
            raise PanelFeaturePredicateError("engineering version space is not canonical")
        return result


def _selected_formula(space: object) -> AllOf:
    if not space.survivor_formulas:
        raise PanelFeaturePredicateError("cannot freeze an empty version space")
    return min(space.survivor_formulas, key=lambda item: (len(item.spec_digests), item.formula_digest))


def _predicate_content(value: object, *, engineering: bool) -> dict[str, object]:
    result = {"schema": FROZEN_ENGINEERING_PREDICATE_SCHEMA if engineering else FROZEN_FEATURE_PREDICATE_SCHEMA, "version_space": value.version_space.to_data(), "selected_formula_digest": value.selected_formula_digest, "selection_rule": "minimum-atom-count-then-formula-digest", **_language_data(), **_authority_data()}
    if engineering:
        result.update(_engineering_data())
    return result


@dataclass(frozen=True, slots=True)
class FrozenFeaturePredicate(_NoBool):
    version_space: FeatureVersionSpace
    selected_formula_digest: str
    predicate_digest: str

    def __post_init__(self) -> None:
        if type(self.version_space) is not FeatureVersionSpace:
            raise TypeError("scientific predicate requires FeatureVersionSpace")
        if self.selected_formula_digest != _selected_formula(self.version_space).formula_digest:
            raise PanelFeaturePredicateError("frozen scientific selection differs")
        _digest(self.predicate_digest, "scientific predicate digest")
        if self.predicate_digest != canonical_digest(_predicate_content(self, engineering=False)):
            raise PanelFeaturePredicateError("scientific predicate digest differs")

    @classmethod
    def create(cls, version_space: FeatureVersionSpace) -> "FrozenFeaturePredicate":
        selected = _selected_formula(version_space).formula_digest
        provisional = object.__new__(cls)
        object.__setattr__(provisional, "version_space", version_space)
        object.__setattr__(provisional, "selected_formula_digest", selected)
        return cls(version_space, selected, canonical_digest(_predicate_content(provisional, engineering=False)))

    @property
    def formula(self) -> AllOf:
        return next(item for item in self.version_space.formulas if item.formula_digest == self.selected_formula_digest)

    def to_data(self) -> dict[str, object]:
        return {**_predicate_content(self, engineering=False), "predicate_digest": self.predicate_digest}

    @classmethod
    def from_data(cls, value: object) -> "FrozenFeaturePredicate":
        raw = _fields(value, {"schema", "version_space", "selected_formula_digest", "selection_rule", *_language_data(), *_authority_data(), "predicate_digest"}, "frozen scientific predicate")
        if raw["schema"] != FROZEN_FEATURE_PREDICATE_SCHEMA or raw["selection_rule"] != "minimum-atom-count-then-formula-digest" or any(raw[key] != item for key, item in {**_language_data(), **_authority_data()}.items()):
            raise PanelFeaturePredicateError("frozen scientific predicate policy differs")
        result = cls(FeatureVersionSpace.from_data(raw["version_space"]), raw["selected_formula_digest"], raw["predicate_digest"])
        if result.to_data() != dict(raw):
            raise PanelFeaturePredicateError("frozen scientific predicate is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class FrozenEngineeringFeaturePredicate(_NoBool):
    version_space: EngineeringFeatureVersionSpace
    selected_formula_digest: str
    predicate_digest: str

    def __post_init__(self) -> None:
        if type(self.version_space) is not EngineeringFeatureVersionSpace:
            raise TypeError("engineering predicate requires EngineeringFeatureVersionSpace")
        if self.selected_formula_digest != _selected_formula(self.version_space).formula_digest:
            raise PanelFeaturePredicateError("frozen engineering selection differs")
        _digest(self.predicate_digest, "engineering predicate digest")
        if self.predicate_digest != canonical_digest(_predicate_content(self, engineering=True)):
            raise PanelFeaturePredicateError("engineering predicate digest differs")

    @classmethod
    def create(cls, version_space: EngineeringFeatureVersionSpace) -> "FrozenEngineeringFeaturePredicate":
        selected = _selected_formula(version_space).formula_digest
        provisional = object.__new__(cls)
        object.__setattr__(provisional, "version_space", version_space)
        object.__setattr__(provisional, "selected_formula_digest", selected)
        return cls(version_space, selected, canonical_digest(_predicate_content(provisional, engineering=True)))

    @property
    def formula(self) -> AllOf:
        return next(item for item in self.version_space.formulas if item.formula_digest == self.selected_formula_digest)

    def to_data(self) -> dict[str, object]:
        return {**_predicate_content(self, engineering=True), "predicate_digest": self.predicate_digest}

    @classmethod
    def from_data(cls, value: object) -> "FrozenEngineeringFeaturePredicate":
        raw = _fields(value, {"schema", "version_space", "selected_formula_digest", "selection_rule", *_language_data(), *_engineering_data(), *_authority_data(), "predicate_digest"}, "frozen engineering predicate")
        if raw["schema"] != FROZEN_ENGINEERING_PREDICATE_SCHEMA or raw["selection_rule"] != "minimum-atom-count-then-formula-digest" or any(raw[key] != item for key, item in {**_language_data(), **_engineering_data(), **_authority_data()}.items()):
            raise PanelFeaturePredicateError("frozen engineering predicate policy differs")
        result = cls(EngineeringFeatureVersionSpace.from_data(raw["version_space"]), raw["selected_formula_digest"], raw["predicate_digest"])
        if result.to_data() != dict(raw):
            raise PanelFeaturePredicateError("frozen engineering predicate is not canonical")
        return result


def _pair_content(value: object, *, engineering: bool) -> dict[str, object]:
    result = {"schema": FROZEN_ENGINEERING_PAIR_SCHEMA if engineering else FROZEN_FEATURE_PAIR_SCHEMA, "side0_predicate": value.side0_predicate.to_data(), "side1_predicate": value.side1_predicate.to_data(), "two_native_orientations_required": True, **_language_data(), **_authority_data()}
    if engineering:
        result.update(_engineering_data())
    return result


def _validate_pair(value: object, predicate_type: type) -> None:
    if type(value.side0_predicate) is not predicate_type or type(value.side1_predicate) is not predicate_type:
        raise TypeError("predicate pair members have the wrong type")
    left, right = value.side0_predicate.version_space, value.side1_predicate.version_space
    if left.native_orientation is not NativeOrientation.SIDE0_POSITIVE or right.native_orientation is not NativeOrientation.SIDE1_POSITIVE or left.support_table.vocabulary.vocabulary_digest != right.support_table.vocabulary.vocabulary_digest or left.support_table.table_digest != right.support_table.table_digest or left.side0_panel_digests != right.side0_panel_digests or left.side1_panel_digests != right.side1_panel_digests:
        raise PanelFeaturePredicateError("predicate pair is not one frozen predicate per native orientation")


@dataclass(frozen=True, slots=True)
class FrozenFeaturePredicatePair(_NoBool):
    side0_predicate: FrozenFeaturePredicate
    side1_predicate: FrozenFeaturePredicate
    pair_digest: str

    def __post_init__(self) -> None:
        _validate_pair(self, FrozenFeaturePredicate)
        _digest(self.pair_digest, "scientific predicate-pair digest")
        if self.pair_digest != canonical_digest(_pair_content(self, engineering=False)):
            raise PanelFeaturePredicateError("scientific predicate-pair digest differs")

    @classmethod
    def create(cls, side0_space: FeatureVersionSpace, side1_space: FeatureVersionSpace) -> "FrozenFeaturePredicatePair":
        values = {"side0_predicate": FrozenFeaturePredicate.create(side0_space), "side1_predicate": FrozenFeaturePredicate.create(side1_space)}
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, pair_digest=canonical_digest(_pair_content(provisional, engineering=False)))

    @property
    def vocabulary(self) -> FeatureVocabulary:
        return self.side0_predicate.version_space.support_table.vocabulary

    def to_data(self) -> dict[str, object]:
        return {**_pair_content(self, engineering=False), "pair_digest": self.pair_digest}

    @classmethod
    def from_data(cls, value: object) -> "FrozenFeaturePredicatePair":
        raw = _fields(value, {"schema", "side0_predicate", "side1_predicate", "two_native_orientations_required", *_language_data(), *_authority_data(), "pair_digest"}, "scientific predicate pair")
        if raw["schema"] != FROZEN_FEATURE_PAIR_SCHEMA or raw["two_native_orientations_required"] is not True or any(raw[key] != item for key, item in {**_language_data(), **_authority_data()}.items()):
            raise PanelFeaturePredicateError("scientific predicate pair policy differs")
        result = cls(FrozenFeaturePredicate.from_data(raw["side0_predicate"]), FrozenFeaturePredicate.from_data(raw["side1_predicate"]), raw["pair_digest"])
        if result.to_data() != dict(raw):
            raise PanelFeaturePredicateError("scientific predicate pair is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class FrozenEngineeringFeaturePredicatePair(_NoBool):
    side0_predicate: FrozenEngineeringFeaturePredicate
    side1_predicate: FrozenEngineeringFeaturePredicate
    pair_digest: str

    def __post_init__(self) -> None:
        _validate_pair(self, FrozenEngineeringFeaturePredicate)
        _digest(self.pair_digest, "engineering predicate-pair digest")
        if self.pair_digest != canonical_digest(_pair_content(self, engineering=True)):
            raise PanelFeaturePredicateError("engineering predicate-pair digest differs")

    @classmethod
    def create(cls, side0_space: EngineeringFeatureVersionSpace, side1_space: EngineeringFeatureVersionSpace) -> "FrozenEngineeringFeaturePredicatePair":
        values = {"side0_predicate": FrozenEngineeringFeaturePredicate.create(side0_space), "side1_predicate": FrozenEngineeringFeaturePredicate.create(side1_space)}
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, pair_digest=canonical_digest(_pair_content(provisional, engineering=True)))

    @property
    def vocabulary(self) -> FeatureVocabulary:
        return self.side0_predicate.version_space.support_table.vocabulary

    def to_data(self) -> dict[str, object]:
        return {**_pair_content(self, engineering=True), "pair_digest": self.pair_digest}

    @classmethod
    def from_data(cls, value: object) -> "FrozenEngineeringFeaturePredicatePair":
        raw = _fields(value, {"schema", "side0_predicate", "side1_predicate", "two_native_orientations_required", *_language_data(), *_engineering_data(), *_authority_data(), "pair_digest"}, "engineering predicate pair")
        if raw["schema"] != FROZEN_ENGINEERING_PAIR_SCHEMA or raw["two_native_orientations_required"] is not True or any(raw[key] != item for key, item in {**_language_data(), **_engineering_data(), **_authority_data()}.items()):
            raise PanelFeaturePredicateError("engineering predicate pair policy differs")
        result = cls(FrozenEngineeringFeaturePredicate.from_data(raw["side0_predicate"]), FrozenEngineeringFeaturePredicate.from_data(raw["side1_predicate"]), raw["pair_digest"])
        if result.to_data() != dict(raw):
            raise PanelFeaturePredicateError("engineering predicate pair is not canonical")
        return result


def _query_outcome(side0: Disposition, side1: Disposition) -> FeatureQueryOutcome:
    if Disposition.ERROR in (side0, side1):
        return FeatureQueryOutcome.ERROR
    if side0 is Disposition.PRESENT and side1 is Disposition.CERTIFIED_ABSENT:
        return FeatureQueryOutcome.SIDE0
    if side1 is Disposition.PRESENT and side0 is Disposition.CERTIFIED_ABSENT:
        return FeatureQueryOutcome.SIDE1
    return FeatureQueryOutcome.ABSTAIN


def _engineering_query_outcome(side0: EngineeringDisposition, side1: EngineeringDisposition) -> EngineeringQueryOutcome:
    if EngineeringDisposition.ERROR in (side0, side1):
        return EngineeringQueryOutcome.ERROR
    if side0 is EngineeringDisposition.MATCH and side1 is EngineeringDisposition.NONMATCH:
        return EngineeringQueryOutcome.SIDE0
    if side1 is EngineeringDisposition.MATCH and side0 is EngineeringDisposition.NONMATCH:
        return EngineeringQueryOutcome.SIDE1
    return EngineeringQueryOutcome.ABSTAIN


def _query_content(value: object, *, engineering: bool) -> dict[str, object]:
    result = {"schema": ENGINEERING_QUERY_DECISION_SCHEMA if engineering else FEATURE_QUERY_DECISION_SCHEMA, "predicate_pair": value.predicate_pair.to_data(), "query_table": value.query_table.to_data(), "panel_digest": value.panel_digest, "side0_disposition": value.side0_disposition.value, "side1_disposition": value.side1_disposition.value, "outcome": value.outcome.value, "decision_rule": "native-positive-and-other-native-negative", "nonmatch_alone_predicts_opposite": False, **_authority_data()}
    if engineering:
        result.update(_engineering_data())
    else:
        result["semantics"] = "scientific-disposition"
    return result


@dataclass(frozen=True, slots=True)
class FeatureQueryDecision(_NoBool):
    predicate_pair: FrozenFeaturePredicatePair
    query_table: FeatureSupportTable
    panel_digest: str
    side0_disposition: Disposition
    side1_disposition: Disposition
    outcome: FeatureQueryOutcome
    decision_digest: str

    def __post_init__(self) -> None:
        if type(self.predicate_pair) is not FrozenFeaturePredicatePair or type(self.query_table) is not FeatureSupportTable:
            raise TypeError("scientific query decision inputs have the wrong type")
        _digest(self.panel_digest, "query panel digest")
        if self.query_table.panel_digests != (self.panel_digest,) or self.query_table.vocabulary.vocabulary_digest != self.predicate_pair.vocabulary.vocabulary_digest:
            raise PanelFeaturePredicateError("scientific query table must be the exact one-panel frozen vocabulary")
        expected0 = evaluate_all_of(self.predicate_pair.side0_predicate.formula, self.query_table, self.panel_digest)
        expected1 = evaluate_all_of(self.predicate_pair.side1_predicate.formula, self.query_table, self.panel_digest)
        if (self.side0_disposition, self.side1_disposition, self.outcome) != (expected0, expected1, _query_outcome(expected0, expected1)):
            raise PanelFeaturePredicateError("scientific two-sided query replay differs")
        _digest(self.decision_digest, "scientific query decision digest")
        if self.decision_digest != canonical_digest(_query_content(self, engineering=False)):
            raise PanelFeaturePredicateError("scientific query decision digest differs")

    @classmethod
    def create(cls, predicate_pair: FrozenFeaturePredicatePair, query_table: FeatureSupportTable, panel_digest: str) -> "FeatureQueryDecision":
        panel = _digest(panel_digest, "query panel digest")
        side0 = evaluate_all_of(predicate_pair.side0_predicate.formula, query_table, panel)
        side1 = evaluate_all_of(predicate_pair.side1_predicate.formula, query_table, panel)
        values = {"predicate_pair": predicate_pair, "query_table": query_table, "panel_digest": panel, "side0_disposition": side0, "side1_disposition": side1, "outcome": _query_outcome(side0, side1)}
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, decision_digest=canonical_digest(_query_content(provisional, engineering=False)))

    def to_data(self) -> dict[str, object]:
        return {**_query_content(self, engineering=False), "decision_digest": self.decision_digest}

    @classmethod
    def from_data(cls, value: object) -> "FeatureQueryDecision":
        raw = _fields(value, {"schema", "predicate_pair", "query_table", "panel_digest", "side0_disposition", "side1_disposition", "outcome", "decision_rule", "nonmatch_alone_predicts_opposite", "semantics", *_authority_data(), "decision_digest"}, "scientific query decision")
        if raw["schema"] != FEATURE_QUERY_DECISION_SCHEMA or raw["decision_rule"] != "native-positive-and-other-native-negative" or raw["nonmatch_alone_predicts_opposite"] is not False or raw["semantics"] != "scientific-disposition" or any(raw[key] != item for key, item in _authority_data().items()):
            raise PanelFeaturePredicateError("scientific query policy differs")
        try:
            result = cls(FrozenFeaturePredicatePair.from_data(raw["predicate_pair"]), FeatureSupportTable.from_data(raw["query_table"]), raw["panel_digest"], Disposition(raw["side0_disposition"]), Disposition(raw["side1_disposition"]), FeatureQueryOutcome(raw["outcome"]), raw["decision_digest"])
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelFeaturePredicateError):
                raise
            raise PanelFeaturePredicateError("scientific query enum differs") from exc
        if result.to_data() != dict(raw):
            raise PanelFeaturePredicateError("scientific query decision is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class EngineeringQueryDecision(_NoBool):
    predicate_pair: FrozenEngineeringFeaturePredicatePair
    query_table: EngineeringSupportTable
    panel_digest: str
    side0_disposition: EngineeringDisposition
    side1_disposition: EngineeringDisposition
    outcome: EngineeringQueryOutcome
    decision_digest: str

    def __post_init__(self) -> None:
        if type(self.predicate_pair) is not FrozenEngineeringFeaturePredicatePair or type(self.query_table) is not EngineeringSupportTable:
            raise TypeError("engineering query decision inputs have the wrong type")
        _digest(self.panel_digest, "engineering query panel digest")
        if self.query_table.panel_digests != (self.panel_digest,) or self.query_table.vocabulary.vocabulary_digest != self.predicate_pair.vocabulary.vocabulary_digest:
            raise PanelFeaturePredicateError("engineering query table must be the exact one-panel frozen vocabulary")
        expected0 = evaluate_engineering_all_of(self.predicate_pair.side0_predicate.formula, self.query_table, self.panel_digest)
        expected1 = evaluate_engineering_all_of(self.predicate_pair.side1_predicate.formula, self.query_table, self.panel_digest)
        if (self.side0_disposition, self.side1_disposition, self.outcome) != (expected0, expected1, _engineering_query_outcome(expected0, expected1)):
            raise PanelFeaturePredicateError("engineering two-sided query replay differs")
        _digest(self.decision_digest, "engineering query decision digest")
        if self.decision_digest != canonical_digest(_query_content(self, engineering=True)):
            raise PanelFeaturePredicateError("engineering query decision digest differs")

    @classmethod
    def create(cls, predicate_pair: FrozenEngineeringFeaturePredicatePair, query_table: EngineeringSupportTable, panel_digest: str) -> "EngineeringQueryDecision":
        panel = _digest(panel_digest, "engineering query panel digest")
        side0 = evaluate_engineering_all_of(predicate_pair.side0_predicate.formula, query_table, panel)
        side1 = evaluate_engineering_all_of(predicate_pair.side1_predicate.formula, query_table, panel)
        values = {"predicate_pair": predicate_pair, "query_table": query_table, "panel_digest": panel, "side0_disposition": side0, "side1_disposition": side1, "outcome": _engineering_query_outcome(side0, side1)}
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, decision_digest=canonical_digest(_query_content(provisional, engineering=True)))

    def to_data(self) -> dict[str, object]:
        return {**_query_content(self, engineering=True), "decision_digest": self.decision_digest}

    @classmethod
    def from_data(cls, value: object) -> "EngineeringQueryDecision":
        raw = _fields(value, {"schema", "predicate_pair", "query_table", "panel_digest", "side0_disposition", "side1_disposition", "outcome", "decision_rule", "nonmatch_alone_predicts_opposite", *_engineering_data(), *_authority_data(), "decision_digest"}, "engineering query decision")
        policy = {**_engineering_data(), **_authority_data()}
        if raw["schema"] != ENGINEERING_QUERY_DECISION_SCHEMA or raw["decision_rule"] != "native-positive-and-other-native-negative" or raw["nonmatch_alone_predicts_opposite"] is not False or any(raw[key] != item for key, item in policy.items()):
            raise PanelFeaturePredicateError("engineering query policy differs")
        try:
            result = cls(FrozenEngineeringFeaturePredicatePair.from_data(raw["predicate_pair"]), EngineeringSupportTable.from_data(raw["query_table"]), raw["panel_digest"], EngineeringDisposition(raw["side0_disposition"]), EngineeringDisposition(raw["side1_disposition"]), EngineeringQueryOutcome(raw["outcome"]), raw["decision_digest"])
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelFeaturePredicateError):
                raise
            raise PanelFeaturePredicateError("engineering query enum differs") from exc
        if result.to_data() != dict(raw):
            raise PanelFeaturePredicateError("engineering query decision is not canonical")
        return result


__all__ = (
    "AllOf",
    "EngineeringDisposition",
    "EngineeringFeatureVersionSpace",
    "EngineeringQueryDecision",
    "EngineeringQueryOutcome",
    "EngineeringSupportCell",
    "EngineeringSupportTable",
    "FeatureQueryDecision",
    "FeatureQueryOutcome",
    "FeatureSupportCell",
    "FeatureSupportGap",
    "FeatureSupportGapKind",
    "FeatureSupportTable",
    "FeatureVersionSpace",
    "FeatureVocabulary",
    "FormulaGapDiagnostic",
    "FrozenEngineeringFeaturePredicate",
    "FrozenEngineeringFeaturePredicatePair",
    "FrozenFeaturePredicate",
    "FrozenFeaturePredicatePair",
    "PANEL_FEATURE_MAX_CONJUNCTION",
    "PANEL_FEATURE_SUPPORTS_PER_SIDE",
    "PanelFeaturePredicateError",
    "enumerate_all_of",
    "evaluate_all_of",
    "evaluate_engineering_all_of",
    "panel_feature_predicate_algorithm_digest",
    "panel_feature_predicate_source_digest",
)

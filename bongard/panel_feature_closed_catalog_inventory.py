"""Candidate-independent support inventory for whole-panel feature predicates.

This module closes a specific vocabulary-gating bug: a vision proposer is
useful narration, but it must not decide which registered predicates exist.
The inventory therefore enumerates every registered whole-panel feature atom,
forms every positive one- or two-atom ``AllOf`` formula, and only then applies
the preregistered support rule to the composite formula.

ShapeBongard HD support is treated as one positive concept versus a
heterogeneous negative class.  A caller supplies the dataset-declared positive
orientation.  The opposite orientation is retained as a diagnostic version
space, but no coherent "negative concept" formula is required.  This matters
for conjunction tasks: different negative examples may violate different
atoms, so neither atom is allowed to be discarded by a univariate contrast
filter before the conjunction is evaluated.

The artifact consumes typed archived support observations only.  It accepts no
PNG bytes, query values, callbacks, model boundary, executable prose, or Lean
input.  Python is the canonical executable authority and cold replay is a pure
deterministic reconstruction.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
from functools import cache
from itertools import combinations
import json
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.panel_batched_typed_codex_observer import (
    complete_whole_panel_feature_axes,
)
from bongard.panel_feature_observation import (
    EngineeringFeatureDisposition,
    PanelFeatureObservationSet,
)
from bongard.panel_feature_observer_protocol import all_axis_variants
from bongard.panel_feature_predicate import (
    ALL_OF_SCHEMA,
    ENGINEERING_MIN_DECISIVE_SUPPORTS_PER_SIDE,
    ENGINEERING_SUPPORT_RULE,
    PANEL_FEATURE_MAX_CONJUNCTION,
    PANEL_FEATURE_SUPPORTS_PER_SIDE,
    AllOf,
    EngineeringDisposition,
    EngineeringSupportTable,
    FeatureVocabulary,
    panel_feature_predicate_algorithm_digest,
)
from bongard.panel_feature_proposer import (
    PANEL_FEATURE_PROPOSER_PROTOCOL_ID,
    PANEL_FEATURE_PROPOSER_RESULT_SCHEMA,
    PanelFeatureProposerResult,
    panel_feature_proposer_contract_digest,
)
from bongard.panel_soft_ontology import (
    NativeOrientation,
    PanelFeatureSpec,
    feature_catalog_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


CLOSED_CATALOG_SUPPORT_INVENTORY_SCHEMA = (
    "gkm.bongard-panel-feature-closed-catalog-support-inventory.v1"
)
CLOSED_CATALOG_PROPOSER_SNAPSHOT_SCHEMA = (
    "gkm.bongard-panel-feature-proposer-narration-snapshot.v1"
)
CLOSED_CATALOG_SUPPORT_GAP_SCHEMA = (
    "gkm.bongard-panel-feature-closed-catalog-support-gap.v1"
)
CLOSED_CATALOG_FORMULA_VERSION_SPACE_SCHEMA = (
    "gkm.bongard-panel-feature-closed-catalog-formula-version-space.v1"
)
CLOSED_CATALOG_SUPPORT_INVENTORY_ID = (
    "bongard.panel-feature/closed-whole-panel-support-inventory-python-v1"
)
CLOSED_CATALOG_SUPPORT_PANEL_COUNT = 12
CLOSED_CATALOG_SUPPORTS_PER_SIDE = 6

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ORIENTATIONS = (
    NativeOrientation.SIDE0_POSITIVE,
    NativeOrientation.SIDE1_POSITIVE,
)


class ClosedCatalogSupportInventoryError(ValueError):
    """The closed catalog, observation archive, profile, or digest differs."""


class ClosedCatalogSupportInventoryStatus(str, Enum):
    PRIMARY_VERSION_SPACE_NONEMPTY = "primary_version_space_nonempty"
    PRIMARY_SUPPORT_GAP = "primary_support_gap"


class ClosedCatalogSupportGapKind(str, Enum):
    NO_PRIMARY_SUPPORT_CONSISTENT_FORMULA = (
        "no_primary_support_consistent_formula"
    )


def panel_feature_closed_catalog_inventory_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise ClosedCatalogSupportInventoryError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ClosedCatalogSupportInventoryError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _orientation(value: object) -> NativeOrientation:
    if type(value) is not NativeOrientation:
        raise TypeError("primary orientation must be exact NativeOrientation")
    return value


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_affects_identity_inventory_selection_or_replay": False,
    }


def _catalog_policy_data() -> dict[str, object]:
    return {
        "complete_registered_whole_panel_catalog": True,
        "candidate_catalog_selected_by_proposer": False,
        "proposer_nominations_filter_atoms": False,
        "atom_level_contrast_prefilter": False,
        "form_conjunctions_before_support_admission": True,
        "maximum_conjunction_atoms": PANEL_FEATURE_MAX_CONJUNCTION,
        "support_rule": ENGINEERING_SUPPORT_RULE,
        "negative_class_requires_coherent_positive_formula": False,
        "opposite_orientation_is_diagnostic_only": True,
    }


def _sealed_lane_data() -> dict[str, object]:
    return {
        "support_only": True,
        "support_png_bytes_included": False,
        "query_input_included": False,
        "query_pixels_included": False,
        "query_release_capability": False,
        "callbacks_accepted": False,
        "live_model_calls": 0,
        "cold_replay_model_calls": 0,
        "engineering_only": True,
        "scientific_calibration_supplied": False,
        "benchmark_authoritative": False,
    }


def _complete_axes_data() -> list[dict[str, object]]:
    return [item.to_data() for item in complete_whole_panel_feature_axes()]


def _complete_axes_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-panel-feature-closed-axis-catalog.v1",
            "axes": _complete_axes_data(),
            "whole_panel_only": True,
            "complete": True,
            "candidate_selected": False,
        }
    )


@cache
def _complete_specs() -> tuple[PanelFeatureSpec, ...]:
    by_digest: dict[str, PanelFeatureSpec] = {}
    for axis in complete_whole_panel_feature_axes():
        for spec in all_axis_variants(axis):
            previous = by_digest.setdefault(spec.spec_digest, spec)
            if previous != spec:  # SHA-256 collision guard.
                raise ClosedCatalogSupportInventoryError(
                    "whole-panel feature spec digest collision"
                )
    return tuple(by_digest[item] for item in sorted(by_digest))


@cache
def complete_whole_panel_feature_vocabulary() -> FeatureVocabulary:
    """Return all registered whole-panel specs for either possible orientation."""

    specs = _complete_specs()
    return FeatureVocabulary.create(side0_specs=specs, side1_specs=specs)


def panel_feature_closed_catalog_inventory_algorithm_digest() -> str:
    vocabulary = complete_whole_panel_feature_vocabulary()
    return canonical_digest(
        {
            "schema": "gkm.bongard-panel-feature-closed-catalog-algorithm.v1",
            "inventory_id": CLOSED_CATALOG_SUPPORT_INVENTORY_ID,
            "implementation_source_sha256": (
                panel_feature_closed_catalog_inventory_source_digest()
            ),
            "feature_catalog_digest": feature_catalog_digest(),
            "whole_panel_axis_catalog_digest": _complete_axes_digest(),
            "complete_vocabulary_digest": vocabulary.vocabulary_digest,
            "predicate_algorithm_digest": (
                panel_feature_predicate_algorithm_digest()
            ),
            **_catalog_policy_data(),
            **_authority_data(),
        }
    )


def _snapshot_rows(
    raw: Mapping[str, Any], name: str
) -> tuple[Mapping[str, Any], ...]:
    value = raw.get(name)
    if type(value) is not list or any(not isinstance(item, Mapping) for item in value):
        raise ClosedCatalogSupportInventoryError(
            f"proposer snapshot {name} differ"
        )
    return tuple(value)


def _proposer_snapshot_inventory(
    canonical_result_json: str,
) -> tuple[
    str,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    if type(canonical_result_json) is not str or not canonical_result_json:
        raise ClosedCatalogSupportInventoryError(
            "proposer snapshot needs canonical result JSON"
        )
    try:
        decoded = json.loads(canonical_result_json)
    except (TypeError, ValueError) as exc:
        raise ClosedCatalogSupportInventoryError(
            "proposer snapshot JSON is malformed"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise ClosedCatalogSupportInventoryError(
            "proposer snapshot result is not an object"
        )
    if canonical_json(decoded).decode("utf-8") != canonical_result_json:
        raise ClosedCatalogSupportInventoryError(
            "proposer snapshot JSON is not canonical"
        )
    required = {
        "schema",
        "protocol_id",
        "contract_digest",
        "payload_digest",
        "receipt_digest",
        "nominations",
        "language_gaps",
        "nomination_gaps",
        "observer_vocabulary",
        "typed_feature_specs_only",
        "narration_executable",
        "global_spec_deduplication",
    }
    if (
        set(decoded) != required
        or decoded["schema"] != PANEL_FEATURE_PROPOSER_RESULT_SCHEMA
        or decoded["protocol_id"] != PANEL_FEATURE_PROPOSER_PROTOCOL_ID
        or decoded["contract_digest"] != panel_feature_proposer_contract_digest()
        or decoded["typed_feature_specs_only"] is not True
        or decoded["narration_executable"] is not False
        or decoded["global_spec_deduplication"] is not True
    ):
        raise ClosedCatalogSupportInventoryError(
            "proposer snapshot policy differs"
        )
    _digest(decoded["payload_digest"], "proposer payload digest")
    _digest(decoded["receipt_digest"], "proposer receipt digest")
    nominations = _snapshot_rows(decoded, "nominations")
    language_gaps = _snapshot_rows(decoded, "language_gaps")
    nomination_gaps = _snapshot_rows(decoded, "nomination_gaps")
    nominated: list[str] = []
    for row in nominations:
        try:
            proposal = row["proposal"]
            if not isinstance(proposal, Mapping):
                raise TypeError
            spec = PanelFeatureSpec.from_data(proposal["spec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ClosedCatalogSupportInventoryError(
                "proposer snapshot nomination spec differs"
            ) from exc
        nominated.append(spec.spec_digest)
    nominated_row = tuple(sorted(nominated))
    if len(nominated_row) != len(set(nominated_row)):
        raise ClosedCatalogSupportInventoryError(
            "proposer snapshot nomination specs are duplicated"
        )
    return (
        canonical_digest(decoded),
        nominated_row,
        tuple(sorted(canonical_digest(item) for item in language_gaps)),
        tuple(sorted(canonical_digest(item) for item in nomination_gaps)),
    )


def _proposer_snapshot_content(
    value: "ProposerNarrationSnapshot",
) -> dict[str, object]:
    return {
        "schema": CLOSED_CATALOG_PROPOSER_SNAPSHOT_SCHEMA,
        "canonical_result_json": value.canonical_result_json,
        "proposer_result_digest": value.proposer_result_digest,
        "nominated_spec_digests": list(value.nominated_spec_digests),
        "language_gap_digests": list(value.language_gap_digests),
        "nomination_gap_digests": list(value.nomination_gap_digests),
        "full_proposer_output_retained": True,
        "narration_and_provenance_only": True,
        "vocabulary_authority": False,
        "candidate_selection_authority": False,
        "executable": False,
    }


@dataclass(frozen=True, slots=True)
class ProposerNarrationSnapshot:
    """Immutable full proposer output retained without granting it authority."""

    canonical_result_json: str
    proposer_result_digest: str
    nominated_spec_digests: tuple[str, ...]
    language_gap_digests: tuple[str, ...]
    nomination_gap_digests: tuple[str, ...]
    snapshot_digest: str

    def __post_init__(self) -> None:
        expected = _proposer_snapshot_inventory(self.canonical_result_json)
        if (
            self.proposer_result_digest != expected[0]
            or self.nominated_spec_digests != expected[1]
            or self.language_gap_digests != expected[2]
            or self.nomination_gap_digests != expected[3]
        ):
            raise ClosedCatalogSupportInventoryError(
                "proposer narration snapshot inventory differs"
            )
        for label, row in (
            ("nominated spec", self.nominated_spec_digests),
            ("language gap", self.language_gap_digests),
            ("nomination gap", self.nomination_gap_digests),
        ):
            if type(row) is not tuple or row != tuple(sorted(row)):
                raise ClosedCatalogSupportInventoryError(
                    f"proposer {label} digests are not canonical"
                )
            for item in row:
                _digest(item, f"proposer {label} digest")
        _digest(self.snapshot_digest, "proposer snapshot digest")
        if self.snapshot_digest != canonical_digest(_proposer_snapshot_content(self)):
            raise ClosedCatalogSupportInventoryError(
                "proposer narration snapshot digest differs"
            )

    @classmethod
    def create(
        cls, proposer_result: PanelFeatureProposerResult
    ) -> "ProposerNarrationSnapshot":
        if type(proposer_result) is not PanelFeatureProposerResult:
            raise TypeError("proposer snapshot needs exact PanelFeatureProposerResult")
        encoded = canonical_json(proposer_result.to_data()).decode("utf-8")
        result_digest, nominated, language, nomination = (
            _proposer_snapshot_inventory(encoded)
        )
        if proposer_result.result_digest != result_digest:
            raise ClosedCatalogSupportInventoryError(
                "proposer result digest differs from its exact output"
            )
        values = {
            "canonical_result_json": encoded,
            "proposer_result_digest": result_digest,
            "nominated_spec_digests": nominated,
            "language_gap_digests": language,
            "nomination_gap_digests": nomination,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            snapshot_digest=canonical_digest(
                _proposer_snapshot_content(provisional)
            ),
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_proposer_snapshot_content(self),
            "snapshot_digest": self.snapshot_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ProposerNarrationSnapshot":
        raw = _fields(
            value,
            {
                "schema",
                "canonical_result_json",
                "proposer_result_digest",
                "nominated_spec_digests",
                "language_gap_digests",
                "nomination_gap_digests",
                "full_proposer_output_retained",
                "narration_and_provenance_only",
                "vocabulary_authority",
                "candidate_selection_authority",
                "executable",
                "snapshot_digest",
            },
            "proposer narration snapshot",
        )
        if (
            raw["schema"] != CLOSED_CATALOG_PROPOSER_SNAPSHOT_SCHEMA
            or raw["full_proposer_output_retained"] is not True
            or raw["narration_and_provenance_only"] is not True
            or raw["vocabulary_authority"] is not False
            or raw["candidate_selection_authority"] is not False
            or raw["executable"] is not False
            or any(
                type(raw[name]) is not list
                for name in (
                    "nominated_spec_digests",
                    "language_gap_digests",
                    "nomination_gap_digests",
                )
            )
        ):
            raise ClosedCatalogSupportInventoryError(
                "proposer narration snapshot policy differs"
            )
        result = cls(
            raw["canonical_result_json"],
            raw["proposer_result_digest"],
            tuple(raw["nominated_spec_digests"]),
            tuple(raw["language_gap_digests"]),
            tuple(raw["nomination_gap_digests"]),
            raw["snapshot_digest"],
        )
        if result.to_data() != dict(raw):
            raise ClosedCatalogSupportInventoryError(
                "proposer narration snapshot is not canonical"
            )
        return result


def _observation_disposition(
    value: EngineeringFeatureDisposition,
) -> EngineeringDisposition:
    if type(value) is not EngineeringFeatureDisposition:
        raise TypeError(
            "closed catalog needs exact EngineeringFeatureDisposition values"
        )
    mapping = {
        EngineeringFeatureDisposition.MATCH: EngineeringDisposition.MATCH,
        EngineeringFeatureDisposition.NONMATCH: EngineeringDisposition.NONMATCH,
        EngineeringFeatureDisposition.INDETERMINATE: (
            EngineeringDisposition.INDETERMINATE
        ),
        EngineeringFeatureDisposition.ERROR: EngineeringDisposition.ERROR,
    }
    if set(mapping) != set(EngineeringFeatureDisposition):  # pragma: no cover
        raise RuntimeError("closed-catalog disposition mapping is incomplete")
    return mapping[value]


def _and_engineering(
    values: Sequence[EngineeringDisposition],
) -> EngineeringDisposition:
    row = tuple(values)
    if not row:
        raise ClosedCatalogSupportInventoryError("AllOf profile cannot be empty")
    if EngineeringDisposition.ERROR in row:
        return EngineeringDisposition.ERROR
    if EngineeringDisposition.NONMATCH in row:
        return EngineeringDisposition.NONMATCH
    if all(item is EngineeringDisposition.MATCH for item in row):
        return EngineeringDisposition.MATCH
    return EngineeringDisposition.INDETERMINATE


def _validate_support_partition(
    table: EngineeringSupportTable,
    side0: tuple[str, ...],
    side1: tuple[str, ...],
) -> None:
    if type(table) is not EngineeringSupportTable:
        raise TypeError("closed catalog version space needs EngineeringSupportTable")
    for row in (side0, side1):
        if (
            type(row) is not tuple
            or len(row) != PANEL_FEATURE_SUPPORTS_PER_SIDE
            or row != tuple(sorted(row))
            or len(row) != len(set(row))
        ):
            raise ClosedCatalogSupportInventoryError(
                "closed catalog needs six unique sorted panels per side"
            )
        for item in row:
            _digest(item, "closed catalog support panel digest")
    if set(side0) & set(side1) or set(table.panel_digests) != set(side0 + side1):
        raise ClosedCatalogSupportInventoryError(
            "closed catalog support partition differs from the table"
        )


def _all_of_content_fast(
    *,
    vocabulary_digest: str,
    native_orientation: NativeOrientation,
    spec_digests: tuple[str, ...],
    algorithm_digest: str,
) -> dict[str, object]:
    """Exact public ``AllOf`` serialization without repeated source hashing."""

    return {
        "schema": ALL_OF_SCHEMA,
        "algorithm_digest": algorithm_digest,
        "vocabulary_digest": vocabulary_digest,
        "native_orientation": native_orientation.value,
        "spec_digests": list(spec_digests),
        "operator": "all_of",
        "minimum_atoms": 1,
        "maximum_atoms": PANEL_FEATURE_MAX_CONJUNCTION,
        "negation_allowed": False,
        "complement_allowed": False,
        "polarity_flip_allowed": False,
        "arbitrary_code_allowed": False,
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "execution_semantics": "closed-positive-all-of-python-v1",
    }


def _all_of_data_fast(
    formula: AllOf, *, algorithm_digest: str
) -> dict[str, object]:
    return {
        **_all_of_content_fast(
            vocabulary_digest=formula.vocabulary_digest,
            native_orientation=formula.native_orientation,
            spec_digests=formula.spec_digests,
            algorithm_digest=algorithm_digest,
        ),
        "formula_digest": formula.formula_digest,
    }


def _all_of_static_data(value: Mapping[str, object]) -> dict[str, object]:
    """Return the public serialization fields that cannot vary by formula."""

    return {
        key: item
        for key, item in value.items()
        if key not in {"spec_digests", "formula_digest"}
    }


def _public_all_of_templates(
    *,
    vocabulary: FeatureVocabulary,
    native_orientation: NativeOrientation,
    atom_rows: tuple[tuple[str, ...], ...],
    algorithm_digest: str,
) -> dict[int, dict[str, object]]:
    """Seal the public constructor/serializer branch for each atom count.

    The predicate authority has one structural branch per permitted atom
    count.  Calling it once for every branch establishes the exact constant
    serialization fields.  Formula-specific fields are checked exhaustively
    by :func:`_validate_optimized_all_of` below.
    """

    templates: dict[int, dict[str, object]] = {}
    for size in range(1, PANEL_FEATURE_MAX_CONJUNCTION + 1):
        try:
            row = next(item for item in atom_rows if len(item) == size)
        except StopIteration as exc:  # pragma: no cover - closed catalog guard
            raise ClosedCatalogSupportInventoryError(
                "closed catalog lacks an AllOf authority branch witness"
            ) from exc
        witness = AllOf.create(vocabulary, native_orientation, row)
        public_data = witness.to_data()
        fast_data = _all_of_data_fast(
            witness, algorithm_digest=algorithm_digest
        )
        if public_data != fast_data:
            raise ClosedCatalogSupportInventoryError(
                "optimized AllOf encoding differs from the predicate authority"
            )
        templates[size] = _all_of_static_data(public_data)
    return templates


def _validate_optimized_all_of(
    formula: AllOf,
    *,
    vocabulary: FeatureVocabulary,
    native_orientation: NativeOrientation,
    native_spec_digests: frozenset[str],
    expected_spec_digests: tuple[str, ...],
    expected_content: Mapping[str, object],
    expected_formula_digest: str,
    algorithm_digest: str,
    public_static_data: Mapping[str, object],
) -> None:
    """Exhaustively validate one fast value against the public ``AllOf`` seal.

    This is the batched equivalent of the public dataclass invariant and
    serializer checks.  The public authority supplies every constant field;
    this function checks every formula-specific field, vocabulary membership,
    and content digest.  No formula is admitted on the strength of a sample.
    """

    if (
        type(formula) is not AllOf
        or formula.vocabulary_digest != vocabulary.vocabulary_digest
        or formula.native_orientation is not native_orientation
        or type(formula.spec_digests) is not tuple
        or formula.spec_digests != expected_spec_digests
        or not 1 <= len(formula.spec_digests) <= PANEL_FEATURE_MAX_CONJUNCTION
        or formula.spec_digests != tuple(sorted(formula.spec_digests))
        or len(formula.spec_digests) != len(set(formula.spec_digests))
        or any(
            type(item) is not str or _DIGEST.fullmatch(item) is None
            for item in formula.spec_digests
        )
        or not set(formula.spec_digests) <= native_spec_digests
        or type(formula.formula_digest) is not str
        or _DIGEST.fullmatch(formula.formula_digest) is None
    ):
        raise ClosedCatalogSupportInventoryError(
            "optimized AllOf value violates the predicate invariants"
        )
    data = _all_of_data_fast(formula, algorithm_digest=algorithm_digest)
    content = dict(data)
    content.pop("formula_digest")
    if (
        _all_of_static_data(data) != dict(public_static_data)
        or data["spec_digests"] != list(expected_spec_digests)
        or content != dict(expected_content)
        or formula.formula_digest != expected_formula_digest
    ):
        raise ClosedCatalogSupportInventoryError(
            "optimized AllOf value differs from public serialization"
        )


@cache
def _complete_formula_inventory(
    native_orientation: NativeOrientation,
) -> tuple[AllOf, ...]:
    """Build exact public values with constant source checks and a full seal."""

    orientation = _orientation(native_orientation)
    vocabulary = complete_whole_panel_feature_vocabulary()
    atoms = vocabulary.native_spec_digests(orientation)
    native_atoms = frozenset(atoms)
    algorithm = panel_feature_predicate_algorithm_digest()
    atom_rows = tuple(
        row
        for size in range(
            1, min(PANEL_FEATURE_MAX_CONJUNCTION, len(atoms)) + 1
        )
        for row in combinations(atoms, size)
    )
    public_templates = _public_all_of_templates(
        vocabulary=vocabulary,
        native_orientation=orientation,
        atom_rows=atom_rows,
        algorithm_digest=algorithm,
    )
    formulas: list[AllOf] = []
    for row in atom_rows:
        content = _all_of_content_fast(
            vocabulary_digest=vocabulary.vocabulary_digest,
            native_orientation=orientation,
            spec_digests=row,
            algorithm_digest=algorithm,
        )
        formula_digest = canonical_digest(content)
        formula = object.__new__(AllOf)
        object.__setattr__(formula, "vocabulary_digest", vocabulary.vocabulary_digest)
        object.__setattr__(formula, "native_orientation", orientation)
        object.__setattr__(formula, "spec_digests", row)
        object.__setattr__(formula, "formula_digest", formula_digest)
        _validate_optimized_all_of(
            formula,
            vocabulary=vocabulary,
            native_orientation=orientation,
            native_spec_digests=native_atoms,
            expected_spec_digests=row,
            expected_content=content,
            expected_formula_digest=formula_digest,
            algorithm_digest=algorithm,
            public_static_data=public_templates[len(row)],
        )
        formulas.append(formula)
    result = tuple(formulas)
    if not result:
        raise ClosedCatalogSupportInventoryError(
            "closed catalog formula inventory is empty"
        )
    return result


def _formula_rows(
    table: EngineeringSupportTable,
    formulas: tuple[AllOf, ...],
    side0: tuple[str, ...],
    side1: tuple[str, ...],
) -> tuple[tuple[EngineeringDisposition, ...], ...]:
    # The predicate module's public evaluator scans the full cell tuple for
    # every atom lookup.  This exact immutable index changes only complexity,
    # not semantics: ERROR dominates, then NONMATCH, then all-MATCH, otherwise
    # INDETERMINATE.
    by_key = {
        (item.panel_digest, item.spec_digest): item.disposition
        for item in table.cells
    }
    return tuple(
        tuple(
            _and_engineering(
                tuple(by_key[(panel, spec)] for spec in formula.spec_digests)
            )
            for panel in side0 + side1
        )
        for formula in formulas
    )


def _formula_survives(
    formula: AllOf,
    row: tuple[EngineeringDisposition, ...],
    side0: tuple[str, ...],
    side1: tuple[str, ...],
) -> bool:
    native, contrast = (
        (side0, side1)
        if formula.native_orientation is NativeOrientation.SIDE0_POSITIVE
        else (side1, side0)
    )
    by_panel = dict(zip(side0 + side1, row, strict=True))
    native_values = tuple(by_panel[item] for item in native)
    contrast_values = tuple(by_panel[item] for item in contrast)
    return (
        native_values.count(EngineeringDisposition.MATCH)
        >= ENGINEERING_MIN_DECISIVE_SUPPORTS_PER_SIDE
        and native_values.count(EngineeringDisposition.INDETERMINATE)
        <= PANEL_FEATURE_SUPPORTS_PER_SIDE
        - ENGINEERING_MIN_DECISIVE_SUPPORTS_PER_SIDE
        and all(
            item
            not in {EngineeringDisposition.NONMATCH, EngineeringDisposition.ERROR}
            for item in native_values
        )
        and contrast_values.count(EngineeringDisposition.NONMATCH)
        >= ENGINEERING_MIN_DECISIVE_SUPPORTS_PER_SIDE
        and contrast_values.count(EngineeringDisposition.INDETERMINATE)
        <= PANEL_FEATURE_SUPPORTS_PER_SIDE
        - ENGINEERING_MIN_DECISIVE_SUPPORTS_PER_SIDE
        and all(
            item
            not in {EngineeringDisposition.MATCH, EngineeringDisposition.ERROR}
            for item in contrast_values
        )
    )


def _formula_space_content(
    value: "ClosedCatalogFormulaVersionSpace",
) -> dict[str, object]:
    predicate_algorithm = panel_feature_predicate_algorithm_digest()
    return {
        "schema": CLOSED_CATALOG_FORMULA_VERSION_SPACE_SCHEMA,
        "algorithm_digest": panel_feature_closed_catalog_inventory_algorithm_digest(),
        "predicate_algorithm_digest": predicate_algorithm,
        "support_table_digest": value.support_table.table_digest,
        "vocabulary_digest": value.support_table.vocabulary.vocabulary_digest,
        "native_orientation": value.native_orientation.value,
        "side0_panel_digests": list(value.side0_panel_digests),
        "side1_panel_digests": list(value.side1_panel_digests),
        "formulas": [
            _all_of_data_fast(item, algorithm_digest=predicate_algorithm)
            for item in value.formulas
        ],
        "rows": [[item.value for item in row] for row in value.rows],
        "survivor_formula_digests": list(value.survivor_formula_digests),
        "formula_profile_order": "formula-inventory-then-side0-six-then-side1-six",
        "complete_formula_inventory": True,
        "support_rule": ENGINEERING_SUPPORT_RULE,
        "engineering_only": True,
        "scientific_calibration_supplied": False,
        **_catalog_policy_data(),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ClosedCatalogFormulaVersionSpace:
    """Complete formula/profile inventory using indexed, exact evaluation."""

    support_table: EngineeringSupportTable
    native_orientation: NativeOrientation
    side0_panel_digests: tuple[str, ...]
    side1_panel_digests: tuple[str, ...]
    formulas: tuple[AllOf, ...]
    rows: tuple[tuple[EngineeringDisposition, ...], ...]
    survivor_formula_digests: tuple[str, ...]
    version_space_digest: str

    def __post_init__(self) -> None:
        _orientation(self.native_orientation)
        _validate_support_partition(
            self.support_table,
            self.side0_panel_digests,
            self.side1_panel_digests,
        )
        if self.support_table.vocabulary != complete_whole_panel_feature_vocabulary():
            raise ClosedCatalogSupportInventoryError(
                "closed catalog formula space has a partial vocabulary"
            )
        expected_formulas = _complete_formula_inventory(self.native_orientation)
        expected_rows = _formula_rows(
            self.support_table,
            expected_formulas,
            self.side0_panel_digests,
            self.side1_panel_digests,
        )
        expected_survivors = tuple(
            formula.formula_digest
            for formula, row in zip(expected_formulas, expected_rows, strict=True)
            if _formula_survives(
                formula,
                row,
                self.side0_panel_digests,
                self.side1_panel_digests,
            )
        )
        if (
            self.formulas != expected_formulas
            or self.rows != expected_rows
            or self.survivor_formula_digests != expected_survivors
        ):
            raise ClosedCatalogSupportInventoryError(
                "closed catalog formula inventory or profiles differ"
            )
        _digest(self.version_space_digest, "closed catalog version-space digest")
        if self.version_space_digest != canonical_digest(
            _formula_space_content(self)
        ):
            raise ClosedCatalogSupportInventoryError(
                "closed catalog version-space digest differs"
            )

    @classmethod
    def create(
        cls,
        support_table: EngineeringSupportTable,
        native_orientation: NativeOrientation,
        side0_panel_digests: Sequence[str],
        side1_panel_digests: Sequence[str],
    ) -> "ClosedCatalogFormulaVersionSpace":
        if type(support_table) is not EngineeringSupportTable:
            raise TypeError("closed catalog version space needs EngineeringSupportTable")
        orientation = _orientation(native_orientation)
        side0 = tuple(sorted(side0_panel_digests))
        side1 = tuple(sorted(side1_panel_digests))
        _validate_support_partition(support_table, side0, side1)
        if support_table.vocabulary != complete_whole_panel_feature_vocabulary():
            raise ClosedCatalogSupportInventoryError(
                "closed catalog formula space has a partial vocabulary"
            )
        formulas = _complete_formula_inventory(orientation)
        rows = _formula_rows(support_table, formulas, side0, side1)
        survivors = tuple(
            formula.formula_digest
            for formula, row in zip(formulas, rows, strict=True)
            if _formula_survives(formula, row, side0, side1)
        )
        values = {
            "support_table": support_table,
            "native_orientation": orientation,
            "side0_panel_digests": side0,
            "side1_panel_digests": side1,
            "formulas": formulas,
            "rows": rows,
            "survivor_formula_digests": survivors,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            version_space_digest=canonical_digest(
                _formula_space_content(provisional)
            ),
        )

    @property
    def survivor_formulas(self) -> tuple[AllOf, ...]:
        admitted = set(self.survivor_formula_digests)
        return tuple(
            item for item in self.formulas if item.formula_digest in admitted
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_formula_space_content(self),
            "version_space_digest": self.version_space_digest,
        }

    @classmethod
    def from_data(
        cls,
        value: object,
        *,
        support_table: EngineeringSupportTable,
    ) -> "ClosedCatalogFormulaVersionSpace":
        raw = _fields(
            value,
            {
                "schema",
                "algorithm_digest",
                "predicate_algorithm_digest",
                "support_table_digest",
                "vocabulary_digest",
                "native_orientation",
                "side0_panel_digests",
                "side1_panel_digests",
                "formulas",
                "rows",
                "survivor_formula_digests",
                "formula_profile_order",
                "complete_formula_inventory",
                "support_rule",
                "engineering_only",
                "scientific_calibration_supplied",
                *_catalog_policy_data(),
                *_authority_data(),
                "version_space_digest",
            },
            "closed catalog formula version space",
        )
        policy = {**_catalog_policy_data(), **_authority_data()}
        if (
            type(support_table) is not EngineeringSupportTable
            or raw["schema"] != CLOSED_CATALOG_FORMULA_VERSION_SPACE_SCHEMA
            or raw["algorithm_digest"]
            != panel_feature_closed_catalog_inventory_algorithm_digest()
            or raw["predicate_algorithm_digest"]
            != panel_feature_predicate_algorithm_digest()
            or raw["support_table_digest"] != support_table.table_digest
            or raw["vocabulary_digest"]
            != support_table.vocabulary.vocabulary_digest
            or raw["formula_profile_order"]
            != "formula-inventory-then-side0-six-then-side1-six"
            or raw["complete_formula_inventory"] is not True
            or raw["support_rule"] != ENGINEERING_SUPPORT_RULE
            or raw["engineering_only"] is not True
            or raw["scientific_calibration_supplied"] is not False
            or any(raw[name] != item for name, item in policy.items())
            or any(
                type(raw[name]) is not list
                for name in (
                    "side0_panel_digests",
                    "side1_panel_digests",
                    "formulas",
                    "rows",
                    "survivor_formula_digests",
                )
            )
        ):
            raise ClosedCatalogSupportInventoryError(
                "closed catalog formula version-space policy differs"
            )
        try:
            orientation = NativeOrientation(raw["native_orientation"])
            expected_formulas = _complete_formula_inventory(orientation)
            predicate_algorithm = panel_feature_predicate_algorithm_digest()
            if raw["formulas"] != [
                _all_of_data_fast(item, algorithm_digest=predicate_algorithm)
                for item in expected_formulas
            ]:
                raise ClosedCatalogSupportInventoryError(
                    "closed catalog serialized formula inventory differs"
                )
            result = cls(
                support_table,
                orientation,
                tuple(raw["side0_panel_digests"]),
                tuple(raw["side1_panel_digests"]),
                expected_formulas,
                tuple(
                    tuple(EngineeringDisposition(item) for item in row)
                    for row in raw["rows"]
                ),
                tuple(raw["survivor_formula_digests"]),
                raw["version_space_digest"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ClosedCatalogSupportInventoryError):
                raise
            raise ClosedCatalogSupportInventoryError(
                "closed catalog formula version-space value differs"
            ) from exc
        if result.to_data() != dict(raw):
            raise ClosedCatalogSupportInventoryError(
                "closed catalog formula version space is not canonical"
            )
        return result


def _canonical_observations(
    observations: Sequence[PanelFeatureObservationSet],
) -> tuple[PanelFeatureObservationSet, ...]:
    if isinstance(observations, (str, bytes, bytearray, Mapping)):
        raise TypeError("support observations must be an ordered sequence")
    try:
        values = tuple(observations)
    except TypeError as exc:
        raise TypeError("support observations must be an ordered sequence") from exc
    if len(values) != CLOSED_CATALOG_SUPPORT_PANEL_COUNT or any(
        type(item) is not PanelFeatureObservationSet for item in values
    ):
        raise ClosedCatalogSupportInventoryError(
            "closed catalog needs exact side0-six then side1-six observations"
        )
    restored = tuple(
        PanelFeatureObservationSet.from_data(item.to_data()) for item in values
    )
    if restored != values:
        raise ClosedCatalogSupportInventoryError(
            "support observation canonical replay differs"
        )
    required_axes = tuple(
        item.axis_digest for item in complete_whole_panel_feature_axes()
    )
    for item in values:
        actual_axes = tuple(
            axis.axis.axis_digest for axis in item.axis_observations
        )
        if actual_axes != required_axes:
            raise ClosedCatalogSupportInventoryError(
                "support observation omits part of the complete whole-panel catalog"
            )
    if len({item.panel_digest for item in values}) != len(values):
        raise ClosedCatalogSupportInventoryError(
            "support observation panel digests must be unique"
        )
    if (
        len({item.observer_contract_digest for item in values}) != 1
        or len({item.measurement_protocol_digest for item in values}) != 1
    ):
        raise ClosedCatalogSupportInventoryError(
            "support observations do not share one observer contract and protocol"
        )
    return restored


def _derive_inventory(
    observations: Sequence[PanelFeatureObservationSet],
) -> tuple[
    tuple[PanelFeatureObservationSet, ...],
    FeatureVocabulary,
    EngineeringSupportTable,
    ClosedCatalogFormulaVersionSpace,
    ClosedCatalogFormulaVersionSpace,
]:
    values = _canonical_observations(observations)
    vocabulary = complete_whole_panel_feature_vocabulary()
    table_values = {
        (observation.panel_digest, spec.spec_digest): _observation_disposition(
            observation.evaluate(spec)
        )
        for observation in values
        for spec in vocabulary.specs
    }
    table = EngineeringSupportTable.create(
        vocabulary,
        tuple(item.panel_digest for item in values),
        table_values,
    )
    side0 = tuple(item.panel_digest for item in values[:6])
    side1 = tuple(item.panel_digest for item in values[6:])
    return (
        values,
        vocabulary,
        table,
        ClosedCatalogFormulaVersionSpace.create(
            table, NativeOrientation.SIDE0_POSITIVE, side0, side1
        ),
        ClosedCatalogFormulaVersionSpace.create(
            table, NativeOrientation.SIDE1_POSITIVE, side0, side1
        ),
    )


def _gap_content(value: "ClosedCatalogSupportGap") -> dict[str, object]:
    return {
        "schema": CLOSED_CATALOG_SUPPORT_GAP_SCHEMA,
        "kind": value.kind.value,
        "primary_orientation": value.primary_orientation.value,
        "primary_version_space_digest": value.primary_version_space_digest,
        "primary_formula_count": value.primary_formula_count,
        "primary_survivor_count": value.primary_survivor_count,
        "opposite_version_space_digest": value.opposite_version_space_digest,
        "opposite_survivor_count_diagnostic_only": (
            value.opposite_survivor_count_diagnostic_only
        ),
        "missing_negative_class_formula_is_not_a_gap": True,
        "query_release_authorized": False,
        **_catalog_policy_data(),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ClosedCatalogSupportGap:
    """Typed failure of the declared positive formula inventory."""

    kind: ClosedCatalogSupportGapKind
    primary_orientation: NativeOrientation
    primary_version_space_digest: str
    primary_formula_count: int
    primary_survivor_count: int
    opposite_version_space_digest: str
    opposite_survivor_count_diagnostic_only: int
    gap_digest: str

    def __post_init__(self) -> None:
        if self.kind is not ClosedCatalogSupportGapKind.NO_PRIMARY_SUPPORT_CONSISTENT_FORMULA:
            raise ClosedCatalogSupportInventoryError(
                "closed catalog support gap kind differs"
            )
        _orientation(self.primary_orientation)
        _digest(self.primary_version_space_digest, "primary version-space digest")
        _digest(self.opposite_version_space_digest, "opposite version-space digest")
        if (
            type(self.primary_formula_count) is not int
            or self.primary_formula_count < 1
            or self.primary_survivor_count != 0
            or type(self.opposite_survivor_count_diagnostic_only) is not int
            or self.opposite_survivor_count_diagnostic_only < 0
        ):
            raise ClosedCatalogSupportInventoryError(
                "closed catalog support gap counts differ"
            )
        _digest(self.gap_digest, "closed catalog support gap digest")
        if self.gap_digest != canonical_digest(_gap_content(self)):
            raise ClosedCatalogSupportInventoryError(
                "closed catalog support gap digest differs"
            )

    @classmethod
    def create(
        cls,
        primary: ClosedCatalogFormulaVersionSpace,
        opposite: ClosedCatalogFormulaVersionSpace,
    ) -> "ClosedCatalogSupportGap":
        if (
            type(primary) is not ClosedCatalogFormulaVersionSpace
            or type(opposite) is not ClosedCatalogFormulaVersionSpace
            or primary.native_orientation is opposite.native_orientation
            or primary.support_table != opposite.support_table
            or primary.survivor_formula_digests
        ):
            raise ClosedCatalogSupportInventoryError(
                "typed support gap requires one empty primary space and its opposite diagnostic"
            )
        values = {
            "kind": (
                ClosedCatalogSupportGapKind.NO_PRIMARY_SUPPORT_CONSISTENT_FORMULA
            ),
            "primary_orientation": primary.native_orientation,
            "primary_version_space_digest": primary.version_space_digest,
            "primary_formula_count": len(primary.formulas),
            "primary_survivor_count": 0,
            "opposite_version_space_digest": opposite.version_space_digest,
            "opposite_survivor_count_diagnostic_only": len(
                opposite.survivor_formula_digests
            ),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values, gap_digest=canonical_digest(_gap_content(provisional))
        )

    def to_data(self) -> dict[str, object]:
        return {**_gap_content(self), "gap_digest": self.gap_digest}

    @classmethod
    def from_data(cls, value: object) -> "ClosedCatalogSupportGap":
        raw = _fields(
            value,
            {
                "schema",
                "kind",
                "primary_orientation",
                "primary_version_space_digest",
                "primary_formula_count",
                "primary_survivor_count",
                "opposite_version_space_digest",
                "opposite_survivor_count_diagnostic_only",
                "missing_negative_class_formula_is_not_a_gap",
                "query_release_authorized",
                *_catalog_policy_data(),
                *_authority_data(),
                "gap_digest",
            },
            "closed catalog support gap",
        )
        policy = {**_catalog_policy_data(), **_authority_data()}
        if (
            raw["schema"] != CLOSED_CATALOG_SUPPORT_GAP_SCHEMA
            or raw["missing_negative_class_formula_is_not_a_gap"] is not True
            or raw["query_release_authorized"] is not False
            or any(raw[name] != item for name, item in policy.items())
        ):
            raise ClosedCatalogSupportInventoryError(
                "closed catalog support gap policy differs"
            )
        try:
            result = cls(
                ClosedCatalogSupportGapKind(raw["kind"]),
                NativeOrientation(raw["primary_orientation"]),
                raw["primary_version_space_digest"],
                raw["primary_formula_count"],
                raw["primary_survivor_count"],
                raw["opposite_version_space_digest"],
                raw["opposite_survivor_count_diagnostic_only"],
                raw["gap_digest"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ClosedCatalogSupportInventoryError):
                raise
            raise ClosedCatalogSupportInventoryError(
                "closed catalog support gap value differs"
            ) from exc
        if result.to_data() != dict(raw):
            raise ClosedCatalogSupportInventoryError(
                "closed catalog support gap is not canonical"
            )
        return result


def _space_for_orientation(
    side0: ClosedCatalogFormulaVersionSpace,
    side1: ClosedCatalogFormulaVersionSpace,
    orientation: NativeOrientation,
) -> tuple[ClosedCatalogFormulaVersionSpace, ClosedCatalogFormulaVersionSpace]:
    _orientation(orientation)
    return (side0, side1) if orientation is NativeOrientation.SIDE0_POSITIVE else (side1, side0)


def _inventory_phase(
    side0: ClosedCatalogFormulaVersionSpace,
    side1: ClosedCatalogFormulaVersionSpace,
    primary_orientation: NativeOrientation,
) -> tuple[
    ClosedCatalogSupportInventoryStatus,
    ClosedCatalogSupportGap | None,
]:
    primary, opposite = _space_for_orientation(
        side0, side1, primary_orientation
    )
    if primary.survivor_formula_digests:
        return (
            ClosedCatalogSupportInventoryStatus.PRIMARY_VERSION_SPACE_NONEMPTY,
            None,
        )
    return (
        ClosedCatalogSupportInventoryStatus.PRIMARY_SUPPORT_GAP,
        ClosedCatalogSupportGap.create(primary, opposite),
    )


def _inventory_content(
    value: "ClosedCatalogSupportInventory",
) -> dict[str, object]:
    return {
        "schema": CLOSED_CATALOG_SUPPORT_INVENTORY_SCHEMA,
        "inventory_id": CLOSED_CATALOG_SUPPORT_INVENTORY_ID,
        "algorithm_digest": panel_feature_closed_catalog_inventory_algorithm_digest(),
        "feature_catalog_digest": feature_catalog_digest(),
        "whole_panel_axes": _complete_axes_data(),
        "whole_panel_axis_catalog_digest": _complete_axes_digest(),
        "proposer_snapshot": value.proposer_snapshot.to_data(),
        "support_observations": [
            item.to_data() for item in value.support_observations
        ],
        "support_observation_order": "side0-six-then-side1-six",
        "primary_orientation": value.primary_orientation.value,
        "vocabulary": value.vocabulary.to_data(),
        "complete_spec_count": len(value.vocabulary.specs),
        "support_table": value.support_table.to_data(),
        "side0_version_space": value.side0_version_space.to_data(),
        "side1_version_space": value.side1_version_space.to_data(),
        "profiles_are_formula_rows_in_version_spaces": True,
        "primary_status": value.status.value,
        "support_gap": None if value.support_gap is None else value.support_gap.to_data(),
        **_catalog_policy_data(),
        **_sealed_lane_data(),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ClosedCatalogSupportInventory:
    """Full candidate-independent support formula inventory and profiles."""

    proposer_snapshot: ProposerNarrationSnapshot
    support_observations: tuple[PanelFeatureObservationSet, ...]
    primary_orientation: NativeOrientation
    vocabulary: FeatureVocabulary
    support_table: EngineeringSupportTable
    side0_version_space: ClosedCatalogFormulaVersionSpace
    side1_version_space: ClosedCatalogFormulaVersionSpace
    status: ClosedCatalogSupportInventoryStatus
    support_gap: ClosedCatalogSupportGap | None
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.proposer_snapshot) is not ProposerNarrationSnapshot:
            raise TypeError("closed catalog needs a proposer narration snapshot")
        _orientation(self.primary_orientation)
        if type(self.support_observations) is not tuple:
            raise TypeError("closed catalog support observations must be a tuple")
        (
            observations,
            vocabulary,
            table,
            side0,
            side1,
        ) = _derive_inventory(self.support_observations)
        status, gap = _inventory_phase(
            side0, side1, self.primary_orientation
        )
        if (
            observations != self.support_observations
            or vocabulary != self.vocabulary
            or table != self.support_table
            or side0 != self.side0_version_space
            or side1 != self.side1_version_space
            or status is not self.status
            or gap != self.support_gap
        ):
            raise ClosedCatalogSupportInventoryError(
                "closed catalog inventory differs from deterministic replay"
            )
        _digest(self.record_digest, "closed catalog inventory digest")
        if self.record_digest != canonical_digest(_inventory_content(self)):
            raise ClosedCatalogSupportInventoryError(
                "closed catalog inventory digest differs"
            )

    @property
    def artifact_address(self) -> str:
        return "sha256:" + self.record_digest

    @property
    def primary_version_space(self) -> ClosedCatalogFormulaVersionSpace:
        return _space_for_orientation(
            self.side0_version_space,
            self.side1_version_space,
            self.primary_orientation,
        )[0]

    @property
    def opposite_diagnostic_version_space(self) -> ClosedCatalogFormulaVersionSpace:
        return _space_for_orientation(
            self.side0_version_space,
            self.side1_version_space,
            self.primary_orientation,
        )[1]

    @classmethod
    def create(
        cls,
        proposer_result: PanelFeatureProposerResult,
        support_observations: Sequence[PanelFeatureObservationSet],
        *,
        primary_orientation: NativeOrientation,
    ) -> "ClosedCatalogSupportInventory":
        snapshot = ProposerNarrationSnapshot.create(proposer_result)
        observations, vocabulary, table, side0, side1 = _derive_inventory(
            support_observations
        )
        orientation = _orientation(primary_orientation)
        status, gap = _inventory_phase(side0, side1, orientation)
        values = {
            "proposer_snapshot": snapshot,
            "support_observations": observations,
            "primary_orientation": orientation,
            "vocabulary": vocabulary,
            "support_table": table,
            "side0_version_space": side0,
            "side1_version_space": side1,
            "status": status,
            "support_gap": gap,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            record_digest=canonical_digest(_inventory_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_inventory_content(self),
            "record_digest": self.record_digest,
            "artifact_address": self.artifact_address,
        }

    @classmethod
    def from_data(cls, value: object) -> "ClosedCatalogSupportInventory":
        raw = _fields(
            value,
            {
                "schema",
                "inventory_id",
                "algorithm_digest",
                "feature_catalog_digest",
                "whole_panel_axes",
                "whole_panel_axis_catalog_digest",
                "proposer_snapshot",
                "support_observations",
                "support_observation_order",
                "primary_orientation",
                "vocabulary",
                "complete_spec_count",
                "support_table",
                "side0_version_space",
                "side1_version_space",
                "profiles_are_formula_rows_in_version_spaces",
                "primary_status",
                "support_gap",
                *_catalog_policy_data(),
                *_sealed_lane_data(),
                *_authority_data(),
                "record_digest",
                "artifact_address",
            },
            "closed catalog support inventory",
        )
        policy = {
            **_catalog_policy_data(),
            **_sealed_lane_data(),
            **_authority_data(),
        }
        if (
            raw["schema"] != CLOSED_CATALOG_SUPPORT_INVENTORY_SCHEMA
            or raw["inventory_id"] != CLOSED_CATALOG_SUPPORT_INVENTORY_ID
            or raw["algorithm_digest"]
            != panel_feature_closed_catalog_inventory_algorithm_digest()
            or raw["feature_catalog_digest"] != feature_catalog_digest()
            or raw["whole_panel_axes"] != _complete_axes_data()
            or raw["whole_panel_axis_catalog_digest"] != _complete_axes_digest()
            or raw["support_observation_order"] != "side0-six-then-side1-six"
            or raw["profiles_are_formula_rows_in_version_spaces"] is not True
            or any(raw[name] != item for name, item in policy.items())
            or type(raw["support_observations"]) is not list
        ):
            raise ClosedCatalogSupportInventoryError(
                "closed catalog support inventory policy differs"
            )
        try:
            table = EngineeringSupportTable.from_data(raw["support_table"])
            result = cls(
                proposer_snapshot=ProposerNarrationSnapshot.from_data(
                    raw["proposer_snapshot"]
                ),
                support_observations=tuple(
                    PanelFeatureObservationSet.from_data(item)
                    for item in raw["support_observations"]
                ),
                primary_orientation=NativeOrientation(raw["primary_orientation"]),
                vocabulary=FeatureVocabulary.from_data(raw["vocabulary"]),
                support_table=table,
                side0_version_space=ClosedCatalogFormulaVersionSpace.from_data(
                    raw["side0_version_space"], support_table=table
                ),
                side1_version_space=ClosedCatalogFormulaVersionSpace.from_data(
                    raw["side1_version_space"], support_table=table
                ),
                status=ClosedCatalogSupportInventoryStatus(raw["primary_status"]),
                support_gap=(
                    None
                    if raw["support_gap"] is None
                    else ClosedCatalogSupportGap.from_data(raw["support_gap"])
                ),
                record_digest=raw["record_digest"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ClosedCatalogSupportInventoryError):
                raise
            raise ClosedCatalogSupportInventoryError(
                "closed catalog support inventory value differs"
            ) from exc
        if (
            raw["complete_spec_count"] != len(result.vocabulary.specs)
            or raw["artifact_address"] != result.artifact_address
            or result.to_data() != dict(raw)
        ):
            raise ClosedCatalogSupportInventoryError(
                "closed catalog support inventory is not canonical"
            )
        return result


def cold_replay_closed_catalog_support_inventory(
    archived: ClosedCatalogSupportInventory,
    *,
    expected_artifact_address: str | None = None,
) -> ClosedCatalogSupportInventory:
    """Rebuild the exact inventory from archived typed support, with zero calls."""

    if type(archived) is not ClosedCatalogSupportInventory:
        raise TypeError("cold replay needs exact ClosedCatalogSupportInventory")
    if expected_artifact_address is not None and (
        type(expected_artifact_address) is not str
        or _ADDRESS.fullmatch(expected_artifact_address) is None
        or archived.artifact_address != expected_artifact_address
    ):
        raise ClosedCatalogSupportInventoryError(
            "expected closed-catalog artifact address differs"
        )
    restored = ClosedCatalogSupportInventory.from_data(archived.to_data())
    status, gap = _inventory_phase(
        restored.side0_version_space,
        restored.side1_version_space,
        restored.primary_orientation,
    )
    values = {
        "proposer_snapshot": restored.proposer_snapshot,
        "support_observations": restored.support_observations,
        "primary_orientation": restored.primary_orientation,
        "vocabulary": restored.vocabulary,
        "support_table": restored.support_table,
        "side0_version_space": restored.side0_version_space,
        "side1_version_space": restored.side1_version_space,
        "status": status,
        "support_gap": gap,
    }
    provisional = object.__new__(ClosedCatalogSupportInventory)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    replayed = ClosedCatalogSupportInventory(
        **values,
        record_digest=canonical_digest(_inventory_content(provisional)),
    )
    if replayed != archived:
        raise ClosedCatalogSupportInventoryError(
            "closed catalog support inventory differs on cold replay"
        )
    return replayed


__all__ = (
    "CLOSED_CATALOG_FORMULA_VERSION_SPACE_SCHEMA",
    "CLOSED_CATALOG_SUPPORT_INVENTORY_ID",
    "CLOSED_CATALOG_SUPPORT_INVENTORY_SCHEMA",
    "ClosedCatalogFormulaVersionSpace",
    "ClosedCatalogSupportGap",
    "ClosedCatalogSupportGapKind",
    "ClosedCatalogSupportInventory",
    "ClosedCatalogSupportInventoryError",
    "ClosedCatalogSupportInventoryStatus",
    "ProposerNarrationSnapshot",
    "cold_replay_closed_catalog_support_inventory",
    "complete_whole_panel_feature_vocabulary",
    "panel_feature_closed_catalog_inventory_algorithm_digest",
    "panel_feature_closed_catalog_inventory_source_digest",
)

"""Candidate-neutral typed observations over a frozen owner inventory.

The vision instrument does not answer a candidate-relative yes/no question.
For one feature *axis* (family, scope, and reference frame), it reports the
complete set of closed variants observed on every eligible owner binding.
Python compares a :class:`PanelFeatureSpec` with those resolved sets later.

These records are intentionally labelled engineering-only.  A complete model
answer is not a scientific calibration result, and this module never converts
one into :class:`bongard.predicate_backend.Disposition`.  It supplies a safe
operational bridge for the EOD diagnostic while the separately calibrated
scientific projection remains fail-closed in ``panel_soft_ontology``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations, permutations
import re
from typing import Any, Mapping

from bongard.canonical import canonical_digest
from bongard.panel_soft_ontology import (
    FAMILY_CONTRACTS,
    ClosedCount,
    ComponentCountParameters,
    ExactSegmentCountParameters,
    OwnerInventory,
    PanelFeatureSpec,
    PanelSoftOntologyError,
    QuantizedPoint,
    ReferenceFrame,
    SubjectBinding,
    SubjectBindingKind,
    SubjectScope,
    FeatureFamily,
    coherent_top_level_component_owner_ids,
    descendant_segment_owner_ids,
    subject_search_region,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


FEATURE_AXIS_SCHEMA = "gkm.bongard-panel-feature-axis.v1"
BINDING_OBSERVATION_SCHEMA = "gkm.bongard-panel-binding-feature-observation.v1"
ELIGIBLE_DOMAIN_GAP_SCHEMA = "gkm.bongard-panel-eligible-domain-gap.v1"
PANEL_AXIS_OBSERVATION_SCHEMA = "gkm.bongard-panel-axis-observation.v2"
PANEL_FEATURE_OBSERVATION_SET_SCHEMA = "gkm.bongard-panel-feature-observation-set.v2"
ENGINEERING_FEATURE_CELL_SCHEMA = "gkm.bongard-engineering-feature-cell.v1"
FEATURE_OBSERVATION_PROTOCOL_ID = (
    "bongard.panel-feature-observation/complete-closed-variants-per-binding-v2"
)

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class PanelFeatureObservationError(ValueError):
    """A candidate-neutral feature observation is malformed or incomplete."""


class BindingResolution(str, Enum):
    COMPLETE = "complete"
    UNCLEAR = "unclear"
    ERROR = "error"


class ObservationIssue(str, Enum):
    UNVERIFIED_EMPTY_DOMAIN = "unverified_empty_domain"
    AMBIGUOUS_GEOMETRY = "ambiguous_geometry"
    AMBIGUOUS_OWNERSHIP = "ambiguous_ownership"
    OUTSIDE_CLOSED_CATALOG = "outside_closed_catalog"
    RESOLUTION_LIMIT = "resolution_limit"
    CAPACITY_LIMIT = "capacity_limit"
    PARSER_FAILURE = "parser_failure"
    TRANSPORT_FAILURE = "transport_failure"
    INTEGRITY_FAILURE = "integrity_failure"


_UNCLEAR_ISSUES = frozenset(
    {
        ObservationIssue.AMBIGUOUS_GEOMETRY,
        ObservationIssue.AMBIGUOUS_OWNERSHIP,
        ObservationIssue.OUTSIDE_CLOSED_CATALOG,
        ObservationIssue.RESOLUTION_LIMIT,
        ObservationIssue.CAPACITY_LIMIT,
    }
)
_ERROR_ISSUES = frozenset(
    {
        ObservationIssue.PARSER_FAILURE,
        ObservationIssue.TRANSPORT_FAILURE,
        ObservationIssue.INTEGRITY_FAILURE,
    }
)

_SINGLE_VALUED_FAMILIES = frozenset(
    {
        FeatureFamily.COMPONENT_COUNT,
        FeatureFamily.EXACT_SEGMENT_COUNT,
        FeatureFamily.TURN_PROFILE,
        FeatureFamily.OPEN_TRACE,
        FeatureFamily.CLOSED_LOOP,
        FeatureFamily.ASPECT_RATIO,
        FeatureFamily.TEXTURE_COMPOSITION,
    }
)
_COUNT_BY_INT = {index: item for index, item in enumerate(ClosedCount, start=1)}


class EngineeringFeatureDisposition(str, Enum):
    """Uncalibrated operational state; never a scientific disposition."""

    MATCH = "engineering_match"
    NONMATCH = "engineering_nonmatch"
    INDETERMINATE = "engineering_indeterminate"
    ERROR = "engineering_error"


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise PanelFeatureObservationError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise PanelFeatureObservationError(f"{label} must be a lowercase SHA-256")
    return value


def _canonical_roundtrip(value: object, raw: Mapping[str, Any], label: str) -> None:
    if value.to_data() != dict(raw):  # type: ignore[attr-defined]
        raise PanelFeatureObservationError(f"{label} is not canonical")


def _point_in_region(point: QuantizedPoint, minimum: QuantizedPoint, maximum: QuantizedPoint) -> bool:
    return (
        minimum.x <= point.x <= maximum.x
        and minimum.y <= point.y <= maximum.y
    )


@dataclass(frozen=True, order=True, slots=True)
class FeatureAxis:
    """The candidate-independent part of one closed feature specification."""

    family: FeatureFamily
    subject_scope: SubjectScope
    reference_frame: ReferenceFrame

    def __post_init__(self) -> None:
        if type(self.family) is not FeatureFamily:
            raise TypeError("feature-axis family must be FeatureFamily")
        if type(self.subject_scope) is not SubjectScope:
            raise TypeError("feature-axis scope must be SubjectScope")
        if type(self.reference_frame) is not ReferenceFrame:
            raise TypeError("feature-axis frame must be ReferenceFrame")
        if (
            self.subject_scope,
            self.reference_frame,
        ) not in FAMILY_CONTRACTS[self.family].allowed_scope_frames:
            raise PanelFeatureObservationError(
                "feature-axis scope/reference-frame pair is not registered"
            )

    @classmethod
    def for_spec(cls, spec: PanelFeatureSpec) -> "FeatureAxis":
        if type(spec) is not PanelFeatureSpec:
            raise TypeError("feature axis requires PanelFeatureSpec")
        return cls(spec.family, spec.subject_scope, spec.reference_frame)

    def contains(self, spec: PanelFeatureSpec) -> bool:
        return type(spec) is PanelFeatureSpec and self == FeatureAxis.for_spec(spec)

    @property
    def axis_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": FEATURE_AXIS_SCHEMA,
            "family": self.family.value,
            "subject_scope": self.subject_scope.value,
            "reference_frame": self.reference_frame.value,
        }

    @classmethod
    def from_data(cls, value: object) -> "FeatureAxis":
        raw = _fields(
            value,
            {"schema", "family", "subject_scope", "reference_frame"},
            "feature axis",
        )
        if raw["schema"] != FEATURE_AXIS_SCHEMA:
            raise PanelFeatureObservationError("feature-axis schema differs")
        try:
            result = cls(
                FeatureFamily(raw["family"]),
                SubjectScope(raw["subject_scope"]),
                ReferenceFrame(raw["reference_frame"]),
            )
        except (TypeError, ValueError, PanelSoftOntologyError) as exc:
            if isinstance(exc, PanelFeatureObservationError):
                raise
            raise PanelFeatureObservationError("feature-axis value differs") from exc
        _canonical_roundtrip(result, raw, "feature axis")
        return result


@dataclass(frozen=True, order=True, slots=True)
class EligibleDomainGap:
    """Typed proof obligation for an empty projected binding domain.

    The engineering observer has no independent owner-kind completeness
    certificate.  Consequently an empty projection is recorded, but it can
    never be interpreted as evidence that a feature is absent.
    """

    issue: ObservationIssue
    inventory_digest: str
    axis_digest: str
    eligible_binding_count: int

    def __post_init__(self) -> None:
        if self.issue is not ObservationIssue.UNVERIFIED_EMPTY_DOMAIN:
            raise PanelFeatureObservationError(
                "eligible-domain gap has the wrong issue"
            )
        _digest(self.inventory_digest, "eligible-domain inventory digest")
        _digest(self.axis_digest, "eligible-domain axis digest")
        if (
            type(self.eligible_binding_count) is not int
            or self.eligible_binding_count != 0
        ):
            raise PanelFeatureObservationError(
                "eligible-domain gap must certify exactly zero projected bindings"
            )

    @classmethod
    def unverified_empty(
        cls, inventory: OwnerInventory, axis: FeatureAxis
    ) -> "EligibleDomainGap":
        if type(inventory) is not OwnerInventory or type(axis) is not FeatureAxis:
            raise TypeError("eligible-domain gap needs typed inventory and axis")
        if eligible_axis_bindings(axis, inventory):
            raise PanelFeatureObservationError(
                "eligible-domain gap cannot cover a nonempty projection"
            )
        return cls(
            ObservationIssue.UNVERIFIED_EMPTY_DOMAIN,
            inventory.inventory_digest,
            axis.axis_digest,
            0,
        )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": ELIGIBLE_DOMAIN_GAP_SCHEMA,
            "issue": self.issue.value,
            "inventory_digest": self.inventory_digest,
            "axis_digest": self.axis_digest,
            "eligible_binding_count": self.eligible_binding_count,
            "independent_empty_domain_certificate_supplied": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "EligibleDomainGap":
        raw = _fields(
            value,
            {
                "schema",
                "issue",
                "inventory_digest",
                "axis_digest",
                "eligible_binding_count",
                "independent_empty_domain_certificate_supplied",
            },
            "eligible-domain gap",
        )
        if (
            raw["schema"] != ELIGIBLE_DOMAIN_GAP_SCHEMA
            or raw["independent_empty_domain_certificate_supplied"] is not False
        ):
            raise PanelFeatureObservationError("eligible-domain gap policy differs")
        try:
            result = cls(
                ObservationIssue(raw["issue"]),
                raw["inventory_digest"],
                raw["axis_digest"],
                raw["eligible_binding_count"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelFeatureObservationError):
                raise
            raise PanelFeatureObservationError(
                "eligible-domain gap value differs"
            ) from exc
        _canonical_roundtrip(result, raw, "eligible-domain gap")
        return result


def eligible_axis_bindings(
    axis: FeatureAxis, inventory: OwnerInventory
) -> tuple[SubjectBinding, ...]:
    """Project bindings without choosing a candidate parameter value."""

    if type(axis) is not FeatureAxis or type(inventory) is not OwnerInventory:
        raise TypeError("axis binding projection requires typed axis and inventory")
    contract = FAMILY_CONTRACTS[axis.family]
    binding_kind = contract.binding_by_scope[axis.subject_scope]
    eligible_kinds = set(contract.owner_kinds_by_scope[axis.subject_scope])
    owners = tuple(
        item.owner_id for item in inventory.owners if item.kind in eligible_kinds
    )
    if binding_kind is SubjectBindingKind.PANEL:
        bindings = (SubjectBinding(SubjectBindingKind.PANEL, ()),)
    elif binding_kind is SubjectBindingKind.UNARY:
        bindings = tuple(
            SubjectBinding(SubjectBindingKind.UNARY, (owner,)) for owner in owners
        )
    elif binding_kind is SubjectBindingKind.UNORDERED_PAIR:
        bindings = tuple(
            SubjectBinding(SubjectBindingKind.UNORDERED_PAIR, pair)
            for pair in combinations(owners, 2)
        )
    else:
        bindings = tuple(
            SubjectBinding(SubjectBindingKind.ORDERED_CONTAINER_CONTAINED, pair)
            for pair in permutations(owners, 2)
        )
    return bindings


@dataclass(frozen=True, slots=True)
class BindingFeatureObservation:
    """One raw, closed-axis report for one panel-local subject binding."""

    axis_digest: str
    binding: SubjectBinding
    resolution: BindingResolution
    observed_specs: tuple[PanelFeatureSpec, ...]
    evidence_points: tuple[QuantizedPoint, ...]
    issue: ObservationIssue | None
    observation_receipt_digest: str

    def __post_init__(self) -> None:
        _digest(self.axis_digest, "binding observation axis digest")
        if type(self.binding) is not SubjectBinding:
            raise TypeError("binding observation needs SubjectBinding")
        if type(self.resolution) is not BindingResolution:
            raise TypeError("binding resolution has the wrong type")
        if type(self.observed_specs) is not tuple or any(
            type(item) is not PanelFeatureSpec for item in self.observed_specs
        ):
            raise TypeError("observed specs must be a PanelFeatureSpec tuple")
        spec_digests = tuple(item.spec_digest for item in self.observed_specs)
        if spec_digests != tuple(sorted(spec_digests)) or len(spec_digests) != len(
            set(spec_digests)
        ):
            raise PanelFeatureObservationError(
                "observed specs must be unique and sorted by digest"
            )
        if any(
            FeatureAxis.for_spec(item).axis_digest != self.axis_digest
            for item in self.observed_specs
        ):
            raise PanelFeatureObservationError(
                "observed spec lies outside the binding feature axis"
            )
        if (
            self.observed_specs
            and self.observed_specs[0].family in _SINGLE_VALUED_FAMILIES
            and len(self.observed_specs) != 1
        ):
            raise PanelFeatureObservationError(
                "single-valued feature axis resolved to multiple variants"
            )
        if type(self.evidence_points) is not tuple or any(
            type(item) is not QuantizedPoint for item in self.evidence_points
        ):
            raise TypeError("binding evidence points must be a Grid16 tuple")
        if self.evidence_points != tuple(sorted(self.evidence_points)) or len(
            self.evidence_points
        ) != len(set(self.evidence_points)):
            raise PanelFeatureObservationError(
                "binding evidence points must be unique and sorted"
            )
        if len(self.evidence_points) > 16:
            raise PanelFeatureObservationError("too many binding evidence points")
        if self.resolution is BindingResolution.COMPLETE:
            if self.issue is not None:
                raise PanelFeatureObservationError(
                    "complete binding observation carries an issue"
                )
            if self.observed_specs and not self.evidence_points:
                raise PanelFeatureObservationError(
                    "an observed variant needs panel-local evidence points"
                )
            if not self.observed_specs and self.evidence_points:
                raise PanelFeatureObservationError(
                    "empty resolved variant set cannot carry witness points"
                )
        elif self.resolution is BindingResolution.UNCLEAR:
            if self.issue not in _UNCLEAR_ISSUES:
                raise PanelFeatureObservationError(
                    "unclear binding observation has the wrong issue"
                )
            if self.observed_specs or self.evidence_points:
                raise PanelFeatureObservationError(
                    "unclear binding observation cannot claim resolved evidence"
                )
        else:
            if self.issue not in _ERROR_ISSUES:
                raise PanelFeatureObservationError(
                    "errored binding observation has the wrong issue"
                )
            if self.observed_specs or self.evidence_points:
                raise PanelFeatureObservationError(
                    "errored binding observation cannot claim resolved evidence"
                )
        _digest(self.observation_receipt_digest, "binding observation receipt")

    @property
    def observation_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": BINDING_OBSERVATION_SCHEMA,
            "axis_digest": self.axis_digest,
            "binding": self.binding.to_data(),
            "resolution": self.resolution.value,
            "observed_specs": [item.to_data() for item in self.observed_specs],
            "evidence_points": [item.to_data() for item in self.evidence_points],
            "issue": None if self.issue is None else self.issue.value,
            "observation_receipt_digest": self.observation_receipt_digest,
            "engineering_only": True,
            "scientific_calibration_supplied": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "BindingFeatureObservation":
        raw = _fields(
            value,
            {
                "schema",
                "axis_digest",
                "binding",
                "resolution",
                "observed_specs",
                "evidence_points",
                "issue",
                "observation_receipt_digest",
                "engineering_only",
                "scientific_calibration_supplied",
            },
            "binding feature observation",
        )
        if (
            raw["schema"] != BINDING_OBSERVATION_SCHEMA
            or raw["engineering_only"] is not True
            or raw["scientific_calibration_supplied"] is not False
            or type(raw["observed_specs"]) is not list
            or type(raw["evidence_points"]) is not list
        ):
            raise PanelFeatureObservationError(
                "binding feature observation policy differs"
            )
        try:
            result = cls(
                raw["axis_digest"],
                SubjectBinding.from_data(raw["binding"]),
                BindingResolution(raw["resolution"]),
                tuple(PanelFeatureSpec.from_data(item) for item in raw["observed_specs"]),
                tuple(QuantizedPoint.from_data(item) for item in raw["evidence_points"]),
                None if raw["issue"] is None else ObservationIssue(raw["issue"]),
                raw["observation_receipt_digest"],
            )
        except (TypeError, ValueError, PanelSoftOntologyError) as exc:
            if isinstance(exc, PanelFeatureObservationError):
                raise
            raise PanelFeatureObservationError(
                "binding feature observation value differs"
            ) from exc
        _canonical_roundtrip(result, raw, "binding feature observation")
        return result


@dataclass(frozen=True, slots=True)
class PanelAxisObservation:
    """Exact observation coverage for one feature axis on one panel."""

    inventory: OwnerInventory
    axis: FeatureAxis
    observer_contract_digest: str
    measurement_protocol_digest: str
    binding_observations: tuple[BindingFeatureObservation, ...]
    domain_gap: EligibleDomainGap | None = None

    def __post_init__(self) -> None:
        if type(self.inventory) is not OwnerInventory or type(self.axis) is not FeatureAxis:
            raise TypeError("panel-axis observation needs typed inventory and axis")
        _digest(self.observer_contract_digest, "observer contract digest")
        _digest(self.measurement_protocol_digest, "measurement protocol digest")
        if type(self.binding_observations) is not tuple or any(
            type(item) is not BindingFeatureObservation
            for item in self.binding_observations
        ):
            raise TypeError("panel-axis binding observations have the wrong type")
        expected = eligible_axis_bindings(self.axis, self.inventory)
        actual = tuple(item.binding for item in self.binding_observations)
        if actual != expected:
            raise PanelFeatureObservationError(
                "panel-axis observation does not cover each eligible binding exactly once"
            )
        if expected:
            if self.domain_gap is not None:
                raise PanelFeatureObservationError(
                    "nonempty panel-axis domain cannot carry an empty-domain gap"
                )
        elif self.domain_gap != EligibleDomainGap.unverified_empty(
            self.inventory, self.axis
        ):
            raise PanelFeatureObservationError(
                "empty panel-axis domain needs the exact typed unresolved gap"
            )
        if any(
            item.axis_digest != self.axis.axis_digest
            for item in self.binding_observations
        ):
            raise PanelFeatureObservationError("binding observation has another axis")
        for item in self.binding_observations:
            region = subject_search_region(item.binding, self.inventory)
            if any(
                not _point_in_region(point, region.minimum, region.maximum)
                for point in item.evidence_points
            ):
                raise PanelFeatureObservationError(
                    "binding evidence point lies outside its derived search region"
                )

    @property
    def panel_digest(self) -> str:
        return self.inventory.panel_digest

    @property
    def observation_digest(self) -> str:
        return canonical_digest(self.to_data())

    def evaluate(self, spec: PanelFeatureSpec) -> EngineeringFeatureDisposition:
        """Evaluate after observation; never expose the target during measurement."""

        if type(spec) is not PanelFeatureSpec:
            raise TypeError("feature evaluation requires PanelFeatureSpec")
        if not self.axis.contains(spec):
            raise PanelFeatureObservationError("feature spec lies outside observed axis")
        if self.domain_gap is not None:
            return EngineeringFeatureDisposition.INDETERMINATE
        if any(
            row.resolution is BindingResolution.COMPLETE
            and spec in row.observed_specs
            for row in self.binding_observations
        ):
            return EngineeringFeatureDisposition.MATCH
        if any(
            row.resolution is BindingResolution.ERROR
            for row in self.binding_observations
        ):
            return EngineeringFeatureDisposition.ERROR
        if (
            self.binding_observations
            and self.inventory.enumeration_complete
            and all(
                row.resolution is BindingResolution.COMPLETE
                and bool(row.observed_specs)
                for row in self.binding_observations
            )
        ):
            return EngineeringFeatureDisposition.NONMATCH
        return EngineeringFeatureDisposition.INDETERMINATE

    def to_data(self) -> dict[str, object]:
        return {
            "schema": PANEL_AXIS_OBSERVATION_SCHEMA,
            "protocol_id": FEATURE_OBSERVATION_PROTOCOL_ID,
            "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
            "inventory": self.inventory.to_data(),
            "axis": self.axis.to_data(),
            "observer_contract_digest": self.observer_contract_digest,
            "measurement_protocol_digest": self.measurement_protocol_digest,
            "binding_observations": [
                item.to_data() for item in self.binding_observations
            ],
            "domain_gap": (
                None if self.domain_gap is None else self.domain_gap.to_data()
            ),
            "candidate_parameter_visible_during_measurement": False,
            "engineering_only": True,
            "scientific_calibration_supplied": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "PanelAxisObservation":
        raw = _fields(
            value,
            {
                "schema",
                "protocol_id",
                "predicate_authority_id",
                "inventory",
                "axis",
                "observer_contract_digest",
                "measurement_protocol_digest",
                "binding_observations",
                "domain_gap",
                "candidate_parameter_visible_during_measurement",
                "engineering_only",
                "scientific_calibration_supplied",
            },
            "panel-axis observation",
        )
        if (
            raw["schema"] != PANEL_AXIS_OBSERVATION_SCHEMA
            or raw["protocol_id"] != FEATURE_OBSERVATION_PROTOCOL_ID
            or raw["predicate_authority_id"] != PYTHON_PREDICATE_AUTHORITY_ID
            or raw["candidate_parameter_visible_during_measurement"] is not False
            or raw["engineering_only"] is not True
            or raw["scientific_calibration_supplied"] is not False
            or type(raw["binding_observations"]) is not list
        ):
            raise PanelFeatureObservationError("panel-axis observation policy differs")
        result = cls(
            OwnerInventory.from_data(raw["inventory"]),
            FeatureAxis.from_data(raw["axis"]),
            raw["observer_contract_digest"],
            raw["measurement_protocol_digest"],
            tuple(
                BindingFeatureObservation.from_data(item)
                for item in raw["binding_observations"]
            ),
            (
                None
                if raw["domain_gap"] is None
                else EligibleDomainGap.from_data(raw["domain_gap"])
            ),
        )
        _canonical_roundtrip(result, raw, "panel-axis observation")
        return result


@dataclass(frozen=True, slots=True)
class PanelFeatureObservationSet:
    """All measured axes for one panel under one exact observer contract."""

    inventory: OwnerInventory
    observer_contract_digest: str
    measurement_protocol_digest: str
    axis_observations: tuple[PanelAxisObservation, ...]

    def __post_init__(self) -> None:
        if type(self.inventory) is not OwnerInventory:
            raise TypeError("panel feature observation set needs OwnerInventory")
        _digest(self.observer_contract_digest, "observation-set contract digest")
        _digest(self.measurement_protocol_digest, "observation-set protocol digest")
        if type(self.axis_observations) is not tuple or any(
            type(item) is not PanelAxisObservation for item in self.axis_observations
        ):
            raise TypeError("axis observations have the wrong type")
        axes = tuple(item.axis.axis_digest for item in self.axis_observations)
        if axes != tuple(sorted(axes)) or len(axes) != len(set(axes)):
            raise PanelFeatureObservationError(
                "axis observations must be unique and sorted by digest"
            )
        if any(
            item.inventory != self.inventory
            or item.observer_contract_digest != self.observer_contract_digest
            or item.measurement_protocol_digest != self.measurement_protocol_digest
            for item in self.axis_observations
        ):
            raise PanelFeatureObservationError(
                "axis observation has different inventory or observer custody"
            )

    @property
    def panel_digest(self) -> str:
        return self.inventory.panel_digest

    @property
    def observation_set_digest(self) -> str:
        return canonical_digest(self.to_data())

    def evaluate(self, spec: PanelFeatureSpec) -> EngineeringFeatureDisposition:
        if type(spec) is not PanelFeatureSpec:
            raise TypeError("feature evaluation requires PanelFeatureSpec")
        axis = FeatureAxis.for_spec(spec)
        matches = tuple(
            item for item in self.axis_observations if item.axis == axis
        )
        if not matches:
            return EngineeringFeatureDisposition.INDETERMINATE
        return matches[0].evaluate(spec)

    def to_data(self) -> dict[str, object]:
        return {
            "schema": PANEL_FEATURE_OBSERVATION_SET_SCHEMA,
            "protocol_id": FEATURE_OBSERVATION_PROTOCOL_ID,
            "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
            "inventory": self.inventory.to_data(),
            "observer_contract_digest": self.observer_contract_digest,
            "measurement_protocol_digest": self.measurement_protocol_digest,
            "axis_observations": [item.to_data() for item in self.axis_observations],
            "engineering_only": True,
            "scientific_calibration_supplied": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "PanelFeatureObservationSet":
        raw = _fields(
            value,
            {
                "schema",
                "protocol_id",
                "predicate_authority_id",
                "inventory",
                "observer_contract_digest",
                "measurement_protocol_digest",
                "axis_observations",
                "engineering_only",
                "scientific_calibration_supplied",
            },
            "panel feature observation set",
        )
        if (
            raw["schema"] != PANEL_FEATURE_OBSERVATION_SET_SCHEMA
            or raw["protocol_id"] != FEATURE_OBSERVATION_PROTOCOL_ID
            or raw["predicate_authority_id"] != PYTHON_PREDICATE_AUTHORITY_ID
            or raw["engineering_only"] is not True
            or raw["scientific_calibration_supplied"] is not False
            or type(raw["axis_observations"]) is not list
        ):
            raise PanelFeatureObservationError(
                "panel feature observation-set policy differs"
            )
        result = cls(
            OwnerInventory.from_data(raw["inventory"]),
            raw["observer_contract_digest"],
            raw["measurement_protocol_digest"],
            tuple(
                PanelAxisObservation.from_data(item)
                for item in raw["axis_observations"]
            ),
        )
        _canonical_roundtrip(result, raw, "panel feature observation set")
        return result


@dataclass(frozen=True, slots=True)
class EngineeringFeatureCell:
    """Content-addressed operational evaluation of one frozen spec on a panel."""

    panel_digest: str
    spec_digest: str
    observation_set_digest: str
    disposition: EngineeringFeatureDisposition

    def __post_init__(self) -> None:
        _digest(self.panel_digest, "engineering cell panel digest")
        _digest(self.spec_digest, "engineering cell spec digest")
        _digest(self.observation_set_digest, "engineering observation-set digest")
        if type(self.disposition) is not EngineeringFeatureDisposition:
            raise TypeError("engineering cell disposition has the wrong type")

    @classmethod
    def evaluate(
        cls, observations: PanelFeatureObservationSet, spec: PanelFeatureSpec
    ) -> "EngineeringFeatureCell":
        if type(observations) is not PanelFeatureObservationSet:
            raise TypeError("engineering evaluation needs an observation set")
        if type(spec) is not PanelFeatureSpec:
            raise TypeError("engineering evaluation needs a feature spec")
        return cls(
            observations.panel_digest,
            spec.spec_digest,
            observations.observation_set_digest,
            observations.evaluate(spec),
        )

    @property
    def cell_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": ENGINEERING_FEATURE_CELL_SCHEMA,
            "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
            "panel_digest": self.panel_digest,
            "spec_digest": self.spec_digest,
            "observation_set_digest": self.observation_set_digest,
            "disposition": self.disposition.value,
            "engineering_only": True,
            "uncalibrated": True,
            "scientific_evidence": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "EngineeringFeatureCell":
        raw = _fields(
            value,
            {
                "schema",
                "predicate_authority_id",
                "panel_digest",
                "spec_digest",
                "observation_set_digest",
                "disposition",
                "engineering_only",
                "uncalibrated",
                "scientific_evidence",
            },
            "engineering feature cell",
        )
        if (
            raw["schema"] != ENGINEERING_FEATURE_CELL_SCHEMA
            or raw["predicate_authority_id"] != PYTHON_PREDICATE_AUTHORITY_ID
            or raw["engineering_only"] is not True
            or raw["uncalibrated"] is not True
            or raw["scientific_evidence"] is not False
        ):
            raise PanelFeatureObservationError("engineering feature-cell policy differs")
        try:
            result = cls(
                raw["panel_digest"],
                raw["spec_digest"],
                raw["observation_set_digest"],
                EngineeringFeatureDisposition(raw["disposition"]),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelFeatureObservationError):
                raise
            raise PanelFeatureObservationError(
                "engineering feature-cell value differs"
            ) from exc
        _canonical_roundtrip(result, raw, "engineering feature cell")
        return result


def derive_inventory_count_observation(
    inventory: OwnerInventory,
    axis: FeatureAxis,
    *,
    observer_contract_digest: str,
    measurement_protocol_digest: str,
) -> PanelAxisObservation:
    """Derive the two exact-count axes from the frozen owner graph.

    This is still engineering-only because inventory completeness itself is an
    empirical claim.  It avoids asking vision the redundant candidate-relative
    question "is the count N?" once a complete owner graph has been frozen.
    """

    if type(inventory) is not OwnerInventory or type(axis) is not FeatureAxis:
        raise TypeError("count derivation requires typed inventory and feature axis")
    if axis.family not in {
        FeatureFamily.COMPONENT_COUNT,
        FeatureFamily.EXACT_SEGMENT_COUNT,
    }:
        raise PanelFeatureObservationError("feature axis is not an exact-count axis")
    _digest(observer_contract_digest, "count observer contract digest")
    _digest(measurement_protocol_digest, "count measurement protocol digest")
    rows: list[BindingFeatureObservation] = []
    for binding in eligible_axis_bindings(axis, inventory):
        if not inventory.enumeration_complete:
            observed_specs: tuple[PanelFeatureSpec, ...] = ()
            points: tuple[QuantizedPoint, ...] = ()
            resolution = BindingResolution.UNCLEAR
            issue: ObservationIssue | None = ObservationIssue.RESOLUTION_LIMIT
        else:
            if axis.family is FeatureFamily.COMPONENT_COUNT:
                counted_ids = coherent_top_level_component_owner_ids(inventory)
            else:
                parent = binding.owner_ids[0]
                counted_ids = descendant_segment_owner_ids(
                    parent,
                    inventory,
                )
            owner_by_id = {item.owner_id: item for item in inventory.owners}
            counted = tuple(owner_by_id[item] for item in counted_ids)
            count = len(counted)
            closed_count = _COUNT_BY_INT.get(count)
            if closed_count is None:
                observed_specs = ()
                points = ()
                # The measurement is complete, but no positive registered
                # alternative grounds a closed-catalog exclusion.  Evaluation
                # therefore keeps every registered count indeterminate.
                resolution = BindingResolution.COMPLETE
                issue = None
            else:
                parameters = (
                    ComponentCountParameters(closed_count)
                    if axis.family is FeatureFamily.COMPONENT_COUNT
                    else ExactSegmentCountParameters(closed_count)
                )
                observed_specs = (
                    PanelFeatureSpec(
                        axis.family,
                        axis.subject_scope,
                        axis.reference_frame,
                        parameters,
                    ),
                )
                points = tuple(
                    sorted({item.region.minimum for item in counted})
                )
                resolution = BindingResolution.COMPLETE
                issue = None
        receipt = canonical_digest(
            {
                "schema": "gkm.bongard-inventory-count-derivation-receipt.v1",
                "protocol_id": FEATURE_OBSERVATION_PROTOCOL_ID,
                "inventory_digest": inventory.inventory_digest,
                "axis_digest": axis.axis_digest,
                "binding_digest": binding.binding_digest,
                "observer_contract_digest": observer_contract_digest,
                "measurement_protocol_digest": measurement_protocol_digest,
                "resolution": resolution.value,
                "observed_spec_digests": [
                    item.spec_digest for item in observed_specs
                ],
                "issue": None if issue is None else issue.value,
                "engineering_only": True,
            }
        )
        rows.append(
            BindingFeatureObservation(
                axis.axis_digest,
                binding,
                resolution,
                observed_specs,
                points,
                issue,
                receipt,
            )
        )
    return PanelAxisObservation(
        inventory,
        axis,
        observer_contract_digest,
        measurement_protocol_digest,
        tuple(rows),
        (
            None
            if rows
            else EligibleDomainGap.unverified_empty(inventory, axis)
        ),
    )

"""Strict, candidate-neutral vision protocol for typed panel observations.

One call observes one raw panel and one complete feature axis.  The target
candidate parameter is never shown: the model receives every registered
variant for the axis and every eligible binding from the already frozen owner
inventory.  Missing rows, unknown aliases, incomplete rows, and evidence
outside a binding's derived Grid16 search region are rejected by Python.

The protocol produces engineering-only observations.  Its receipt can bind a
real headless Codex call, but that does not replace scientific calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
import json
import re
from typing import Any, Mapping

from bongard.canonical import canonical_digest
from bongard.panel_feature_observation import (
    BindingFeatureObservation,
    BindingResolution,
    FeatureAxis,
    ObservationIssue,
    PanelAxisObservation,
    PanelFeatureObservationError,
    eligible_axis_bindings,
)
from bongard.panel_soft_ontology import (
    FAMILY_CONTRACTS,
    OwnerInventory,
    PanelFeatureSpec,
    PanelSoftOntologyError,
    QuantizedPoint,
    QuantizedRegion,
    SubjectBinding,
    feature_catalog_data,
    subject_search_region,
)


FEATURE_AXIS_VIEW_SCHEMA = "gkm.bongard-feature-axis-observer-view.v1"
FEATURE_AXIS_VIEW_PROTOCOL_ID = (
    "bongard.panel-feature-observer/one-panel-one-complete-axis-v1"
)
MAX_BINDINGS_PER_AXIS_CALL = 36

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_VARIANT_ALIAS = re.compile(r"variant_[0-9]{4}\Z")
_BINDING_ALIAS = re.compile(r"binding_[0-9]{4}\Z")


class PanelFeatureObserverProtocolError(ValueError):
    """The model view, strict payload, or operational failure differs."""


class AxisCallStatus(str, Enum):
    SUCCESS = "success"
    CAPACITY_GAP = "capacity_gap"
    PARSER_ERROR = "parser_error"
    TRANSPORT_ERROR = "transport_error"
    INTEGRITY_ERROR = "integrity_error"


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise PanelFeatureObserverProtocolError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise PanelFeatureObserverProtocolError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _canonical_roundtrip(value: object, raw: Mapping[str, Any], label: str) -> None:
    if value.to_data() != dict(raw):  # type: ignore[attr-defined]
        raise PanelFeatureObserverProtocolError(f"{label} is not canonical")


def _closed_value(value: object) -> str:
    if type(value) is str:
        return value
    if (
        isinstance(value, Mapping)
        and set(value) == {"code", "exact_count"}
        and type(value["code"]) is str
        and type(value["exact_count"]) is int
    ):
        return value["code"]
    raise PanelFeatureObserverProtocolError("closed catalog value differs")


def all_axis_variants(axis: FeatureAxis) -> tuple[PanelFeatureSpec, ...]:
    """Enumerate the entire parameter catalog for an axis, not a target subset."""

    if type(axis) is not FeatureAxis:
        raise TypeError("axis variant enumeration requires FeatureAxis")
    rows = [
        item
        for item in feature_catalog_data()["families"]  # type: ignore[index]
        if item["family"] == axis.family.value
    ]
    if len(rows) != 1:
        raise PanelFeatureObserverProtocolError("feature family catalog row differs")
    schema_fields = rows[0]["parameter_schema"]["fields"]
    if type(schema_fields) is not list or not schema_fields:
        raise PanelFeatureObserverProtocolError("feature parameter catalog differs")
    names: list[str] = []
    domains: list[tuple[str, ...]] = []
    for field in schema_fields:
        if (
            not isinstance(field, Mapping)
            or set(field) != {"name", "closed_values"}
            or type(field["name"]) is not str
            or type(field["closed_values"]) is not list
            or not field["closed_values"]
        ):
            raise PanelFeatureObserverProtocolError(
                "feature parameter field catalog differs"
            )
        names.append(field["name"])
        domains.append(tuple(_closed_value(item) for item in field["closed_values"]))
    parameter_type = FAMILY_CONTRACTS[axis.family].parameter_type
    variants: list[PanelFeatureSpec] = []
    for values in product(*domains):
        try:
            parameters = parameter_type.from_data(dict(zip(names, values, strict=True)))
            variants.append(
                PanelFeatureSpec(
                    axis.family,
                    axis.subject_scope,
                    axis.reference_frame,
                    parameters,
                )
            )
        except (TypeError, ValueError, PanelSoftOntologyError) as exc:
            raise PanelFeatureObserverProtocolError(
                "closed axis variant construction failed"
            ) from exc
    ordered = tuple(sorted(variants, key=lambda item: item.spec_digest))
    if len({item.spec_digest for item in ordered}) != len(ordered):
        raise PanelFeatureObserverProtocolError("axis variants are not unique")
    return ordered


@dataclass(frozen=True, slots=True)
class AxisVariantAlias:
    alias: str
    spec: PanelFeatureSpec

    def __post_init__(self) -> None:
        if type(self.alias) is not str or _VARIANT_ALIAS.fullmatch(self.alias) is None:
            raise PanelFeatureObserverProtocolError("variant alias differs")
        if type(self.spec) is not PanelFeatureSpec:
            raise TypeError("variant alias needs PanelFeatureSpec")

    def to_data(self) -> dict[str, object]:
        return {"alias": self.alias, "spec": self.spec.to_data()}

    @classmethod
    def from_data(cls, value: object) -> "AxisVariantAlias":
        raw = _fields(value, {"alias", "spec"}, "axis variant alias")
        result = cls(raw["alias"], PanelFeatureSpec.from_data(raw["spec"]))
        _canonical_roundtrip(result, raw, "axis variant alias")
        return result


@dataclass(frozen=True, slots=True)
class AxisBindingAlias:
    alias: str
    binding: SubjectBinding
    search_region: QuantizedRegion

    def __post_init__(self) -> None:
        if type(self.alias) is not str or _BINDING_ALIAS.fullmatch(self.alias) is None:
            raise PanelFeatureObserverProtocolError("binding alias differs")
        if type(self.binding) is not SubjectBinding:
            raise TypeError("binding alias needs SubjectBinding")
        if type(self.search_region) is not QuantizedRegion:
            raise TypeError("binding alias needs QuantizedRegion")

    def to_data(self) -> dict[str, object]:
        return {
            "alias": self.alias,
            "binding": self.binding.to_data(),
            "search_region": self.search_region.to_data(),
        }

    @classmethod
    def from_data(cls, value: object) -> "AxisBindingAlias":
        raw = _fields(
            value, {"alias", "binding", "search_region"}, "axis binding alias"
        )
        result = cls(
            raw["alias"],
            SubjectBinding.from_data(raw["binding"]),
            QuantizedRegion.from_data(raw["search_region"]),
        )
        _canonical_roundtrip(result, raw, "axis binding alias")
        return result


@dataclass(frozen=True, slots=True)
class FeatureAxisObservationView:
    """Internal alias map whose model projection hides candidate and role data."""

    inventory: OwnerInventory
    axis: FeatureAxis
    variants: tuple[AxisVariantAlias, ...]
    bindings: tuple[AxisBindingAlias, ...]

    def __post_init__(self) -> None:
        if type(self.inventory) is not OwnerInventory or type(self.axis) is not FeatureAxis:
            raise TypeError("axis observer view needs typed inventory and axis")
        expected_variants = all_axis_variants(self.axis)
        if self.variants != tuple(
            AxisVariantAlias(f"variant_{index:04d}", spec)
            for index, spec in enumerate(expected_variants)
        ):
            raise PanelFeatureObserverProtocolError(
                "axis observer view does not contain the full variant catalog"
            )
        expected_bindings = eligible_axis_bindings(self.axis, self.inventory)
        if self.bindings != tuple(
            AxisBindingAlias(
                f"binding_{index:04d}",
                binding,
                subject_search_region(binding, self.inventory),
            )
            for index, binding in enumerate(expected_bindings)
        ):
            raise PanelFeatureObserverProtocolError(
                "axis observer view does not contain the exact eligible bindings"
            )
        if len(self.bindings) > MAX_BINDINGS_PER_AXIS_CALL:
            raise PanelFeatureObserverProtocolError(
                "axis observer view exceeds the fixed per-call binding capacity"
            )

    @classmethod
    def build(
        cls, inventory: OwnerInventory, axis: FeatureAxis
    ) -> "FeatureAxisObservationView":
        if type(inventory) is not OwnerInventory or type(axis) is not FeatureAxis:
            raise TypeError("axis observer view needs typed inventory and axis")
        variants = tuple(
            AxisVariantAlias(f"variant_{index:04d}", spec)
            for index, spec in enumerate(all_axis_variants(axis))
        )
        bindings = tuple(
            AxisBindingAlias(
                f"binding_{index:04d}",
                binding,
                subject_search_region(binding, inventory),
            )
            for index, binding in enumerate(eligible_axis_bindings(axis, inventory))
        )
        return cls(inventory, axis, variants, bindings)

    @property
    def view_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": FEATURE_AXIS_VIEW_SCHEMA,
            "protocol_id": FEATURE_AXIS_VIEW_PROTOCOL_ID,
            "inventory": self.inventory.to_data(),
            "axis": self.axis.to_data(),
            "variants": [item.to_data() for item in self.variants],
            "bindings": [item.to_data() for item in self.bindings],
            "candidate_parameter_in_view": False,
            "support_role_in_view": False,
            "engineering_only": True,
            "scientific_calibration_supplied": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "FeatureAxisObservationView":
        raw = _fields(
            value,
            {
                "schema",
                "protocol_id",
                "inventory",
                "axis",
                "variants",
                "bindings",
                "candidate_parameter_in_view",
                "support_role_in_view",
                "engineering_only",
                "scientific_calibration_supplied",
            },
            "feature axis observer view",
        )
        if (
            raw["schema"] != FEATURE_AXIS_VIEW_SCHEMA
            or raw["protocol_id"] != FEATURE_AXIS_VIEW_PROTOCOL_ID
            or raw["candidate_parameter_in_view"] is not False
            or raw["support_role_in_view"] is not False
            or raw["engineering_only"] is not True
            or raw["scientific_calibration_supplied"] is not False
            or type(raw["variants"]) is not list
            or type(raw["bindings"]) is not list
        ):
            raise PanelFeatureObserverProtocolError("axis observer view policy differs")
        result = cls(
            OwnerInventory.from_data(raw["inventory"]),
            FeatureAxis.from_data(raw["axis"]),
            tuple(AxisVariantAlias.from_data(item) for item in raw["variants"]),
            tuple(AxisBindingAlias.from_data(item) for item in raw["bindings"]),
        )
        _canonical_roundtrip(result, raw, "feature axis observer view")
        return result

    def model_data(self) -> dict[str, object]:
        """Return exactly the inert data rendered into the vision prompt."""

        owner_by_id = {item.owner_id: item for item in self.inventory.owners}
        owner_alias = {
            item.owner_id: f"object_{index:04d}"
            for index, item in enumerate(self.inventory.owners)
        }
        return {
            "schema": "gkm.bongard-feature-axis-model-view.v1",
            "axis": {
                "family": self.axis.family.value,
                "subject_scope": self.axis.subject_scope.value,
                "reference_frame": self.axis.reference_frame.value,
            },
            "registered_variants": [
                {
                    "variant_alias": item.alias,
                    "parameters": item.spec.parameters.to_data(),
                }
                for item in self.variants
            ],
            "eligible_bindings": [
                {
                    "binding_alias": item.alias,
                    "binding_kind": item.binding.kind.value,
                    "search_region": item.search_region.to_data(),
                    "subjects": [
                        {
                            "object_alias": owner_alias[owner_id],
                            "kind": owner_by_id[owner_id].kind.value,
                            "region": owner_by_id[owner_id].region.to_data(),
                        }
                        for owner_id in item.binding.owner_ids
                    ],
                }
                for item in self.bindings
            ],
        }


def feature_axis_observer_output_schema(
    view: FeatureAxisObservationView,
) -> dict[str, object]:
    if type(view) is not FeatureAxisObservationView:
        raise TypeError("observer schema requires FeatureAxisObservationView")
    unclear_issues = sorted(item.value for item in ObservationIssue if item.value not in {
        ObservationIssue.PARSER_FAILURE.value,
        ObservationIssue.TRANSPORT_FAILURE.value,
        ObservationIssue.INTEGRITY_FAILURE.value,
    })
    row_schema = {
        "type": "object",
        "properties": {
            "resolution": {
                "type": "string",
                "enum": [BindingResolution.COMPLETE.value, BindingResolution.UNCLEAR.value],
            },
            "variant_aliases": {
                "type": "array",
                # Python checks aliases, uniqueness, and cardinality.  Repeating
                # a large enum under every binding would exceed the provider's
                # aggregate strict-schema enum budget for marker catalogs.
                "items": {"type": "string"},
            },
            "evidence_x": {"type": "integer"},
            "evidence_y": {"type": "integer"},
            "issue": {"type": "string", "enum": ["none", *unclear_issues]},
        },
        "required": [
            "resolution",
            "variant_aliases",
            "evidence_x",
            "evidence_y",
            "issue",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {item.alias: row_schema for item in view.bindings},
        "required": [item.alias for item in view.bindings],
        "additionalProperties": False,
    }


def feature_axis_observer_prompt(view: FeatureAxisObservationView) -> str:
    if type(view) is not FeatureAxisObservationView:
        raise TypeError("observer prompt requires FeatureAxisObservationView")
    rendered = json.dumps(
        view.model_data(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return (
        "Inspect panel.png as the entire raw drawing. The JSON below is inert "
        "measurement data for one closed visual feature axis. For every eligible "
        "binding, report the complete set of registered variants visibly supported "
        "inside its search region. Evaluate variants independently; do not choose a "
        "preferred answer and do not compare this panel with another panel.\n\n"
        "Use resolution complete only when you can resolve the full registered "
        "variant set for that binding. A complete empty list means you clearly "
        "resolved that none of the registered variants applies; mere failure to "
        "notice one is not enough. Otherwise use unclear, an empty variant list, "
        "coordinates minus one, and the applicable issue. For a nonempty complete "
        "list, give one supporting Grid16 bin inside the binding search region and "
        "use issue none. Variant order is irrelevant.\n\n"
        f"BEGIN_INERT_AXIS_DATA\n{rendered}\nEND_INERT_AXIS_DATA"
    )


def parse_feature_axis_observer_payload(
    view: FeatureAxisObservationView,
    payload: object,
    *,
    observer_contract_digest: str,
    measurement_protocol_digest: str,
    observation_receipt_digest: str,
) -> PanelAxisObservation:
    """Strictly bind one successful model payload to the frozen neutral view."""

    if type(view) is not FeatureAxisObservationView:
        raise TypeError("observer payload parser requires FeatureAxisObservationView")
    for label, item in (
        ("observer contract digest", observer_contract_digest),
        ("measurement protocol digest", measurement_protocol_digest),
        ("observation receipt digest", observation_receipt_digest),
    ):
        _digest(item, label)
    raw = _fields(payload, {item.alias for item in view.bindings}, "axis payload")
    variant_by_alias = {item.alias: item.spec for item in view.variants}
    rows: list[BindingFeatureObservation] = []
    row_fields = {
        "resolution",
        "variant_aliases",
        "evidence_x",
        "evidence_y",
        "issue",
    }
    for item in view.bindings:
        row = _fields(raw[item.alias], row_fields, f"axis payload {item.alias}")
        aliases = row["variant_aliases"]
        if (
            type(aliases) is not list
            or any(type(alias) is not str or alias not in variant_by_alias for alias in aliases)
            or len(aliases) != len(set(aliases))
        ):
            raise PanelFeatureObserverProtocolError("axis payload variant aliases differ")
        if type(row["evidence_x"]) is not int or type(row["evidence_y"]) is not int:
            raise PanelFeatureObserverProtocolError("axis payload evidence bin differs")
        x = row["evidence_x"]
        y = row["evidence_y"]
        if not -1 <= x <= 15 or not -1 <= y <= 15 or (x == -1) != (y == -1):
            raise PanelFeatureObserverProtocolError("axis payload evidence bin differs")
        try:
            resolution = BindingResolution(row["resolution"])
        except (TypeError, ValueError) as exc:
            raise PanelFeatureObserverProtocolError("axis payload resolution differs") from exc
        if resolution is BindingResolution.ERROR:
            raise PanelFeatureObserverProtocolError("model payload cannot self-assert error")
        if resolution is BindingResolution.COMPLETE:
            if row["issue"] != "none" or bool(aliases) != (x != -1):
                raise PanelFeatureObserverProtocolError(
                    "complete axis payload row has inconsistent evidence"
                )
            observed_specs = tuple(
                sorted(
                    (variant_by_alias[alias] for alias in aliases),
                    key=lambda spec: spec.spec_digest,
                )
            )
            points = () if x == -1 else (QuantizedPoint(x, y),)
            issue = None
        else:
            if aliases or x != -1 or row["issue"] == "none":
                raise PanelFeatureObserverProtocolError(
                    "unclear axis payload row claims resolved evidence"
                )
            try:
                issue = ObservationIssue(row["issue"])
            except (TypeError, ValueError) as exc:
                raise PanelFeatureObserverProtocolError(
                    "unclear axis payload issue differs"
                ) from exc
            observed_specs = ()
            points = ()
        rows.append(
            BindingFeatureObservation(
                view.axis.axis_digest,
                item.binding,
                resolution,
                observed_specs,
                points,
                issue,
                observation_receipt_digest,
            )
        )
    try:
        return PanelAxisObservation(
            view.inventory,
            view.axis,
            observer_contract_digest,
            measurement_protocol_digest,
            tuple(rows),
        )
    except PanelFeatureObservationError as exc:
        raise PanelFeatureObserverProtocolError(
            "axis payload failed typed observation validation"
        ) from exc


def unresolved_axis_observation(
    inventory: OwnerInventory,
    axis: FeatureAxis,
    *,
    observer_contract_digest: str,
    measurement_protocol_digest: str,
    observation_receipt_digest: str,
    issue: ObservationIssue,
) -> PanelAxisObservation:
    """Create a total fail-closed row vector for capacity or call failure."""

    if type(inventory) is not OwnerInventory or type(axis) is not FeatureAxis:
        raise TypeError("unresolved axis observation needs typed inventory and axis")
    for label, item in (
        ("observer contract digest", observer_contract_digest),
        ("measurement protocol digest", measurement_protocol_digest),
        ("observation receipt digest", observation_receipt_digest),
    ):
        _digest(item, label)
    if type(issue) is not ObservationIssue:
        raise TypeError("unresolved axis issue has the wrong type")
    resolution = (
        BindingResolution.ERROR
        if issue
        in {
            ObservationIssue.PARSER_FAILURE,
            ObservationIssue.TRANSPORT_FAILURE,
            ObservationIssue.INTEGRITY_FAILURE,
        }
        else BindingResolution.UNCLEAR
    )
    rows = tuple(
        BindingFeatureObservation(
            axis.axis_digest,
            binding,
            resolution,
            (),
            (),
            issue,
            observation_receipt_digest,
        )
        for binding in eligible_axis_bindings(axis, inventory)
    )
    return PanelAxisObservation(
        inventory,
        axis,
        observer_contract_digest,
        measurement_protocol_digest,
        rows,
    )

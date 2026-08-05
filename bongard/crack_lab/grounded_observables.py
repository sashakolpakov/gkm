"""Candidate-independent visual observables for grounded Bongard predicates.

The VLM may select these observable IDs, but it cannot redefine their image
semantics.  Each evaluator runs on a neutral panel context and returns a
typed value, a certified semantic absence, an indeterminate measurement, or
an implementation error.  The point-contact extraction is cached so several
formula leaves share exactly the same segmentation and rays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

import grounded_predicate_ir as G
import semantic_legs as L
from visual_witnesses import PointContactSignature


SMALL_GAP_ID = "junction.point-contact.small-exterior-gap/v1"
LARGE_GAP_ID = "junction.point-contact.large-exterior-gap/v1"
GAP_RATIO_ID = "junction.point-contact.exterior-gap-ratio/v1"


@dataclass
class GroundedPanelContext:
    panel: np.ndarray
    _cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        array = np.asarray(self.panel)
        if array.ndim != 2 or not np.issubdtype(array.dtype, np.number) \
                or not np.isfinite(array).all():
            raise ValueError("grounded panel context requires a finite 2-D array")
        self.panel = np.ascontiguousarray(array)

    def point_contact(self) -> PointContactSignature | G.SemanticAbsent \
            | G.Indeterminate | G.Error:
        key = "point-contact-signature/v1"
        if key in self._cache:
            return self._cache[key]
        try:
            result: PointContactSignature | G.SemanticAbsent \
                | G.Indeterminate | G.Error = \
                L.extract_point_contact_signature(self.panel)
        except L.WitnessIndeterminate as exc:
            result = G.Indeterminate(
                exc.failure_mode,
                str(exc),
                (key, "extract_point_contact_signature"),
            )
        except L.WitnessAbsent as exc:
            if exc.failure_mode == "no_point_contact_signature":
                result = G.SemanticAbsent(
                    exc.failure_mode,
                    str(exc),
                    (key, "extract_point_contact_signature"),
                )
            else:
                result = G.Error(
                    "undeclared-point-contact-failure",
                    f"{exc.failure_mode}: {exc}",
                    (key, "extract_point_contact_signature"),
                )
        except Exception as exc:
            result = G.Error(
                "point-contact-extractor-error",
                f"{type(exc).__name__}: {exc}",
                (key, "extract_point_contact_signature"),
            )
        self._cache[key] = result
        return result


@dataclass(frozen=True)
class ObservableDescriptor:
    contract: G.ObservableContract
    description: str
    admissible_shapes: tuple[str, ...]

    def prompt_dict(self) -> dict[str, Any]:
        body = self.contract.contract_dict()
        body["description"] = self.description
        body["admissible_shapes"] = list(self.admissible_shapes)
        return body


def _context(value: Any) -> GroundedPanelContext:
    if not isinstance(value, GroundedPanelContext):
        raise TypeError("observable evaluator requires GroundedPanelContext")
    return value


def _forward_nonvalue(value: Any) -> G.SemanticAbsent | G.Indeterminate \
        | G.Error | None:
    if isinstance(value, (G.SemanticAbsent, G.Indeterminate, G.Error)):
        return value
    return None


def _small_gap(value: Any) -> G.Observation:
    signature = _context(value).point_contact()
    forwarded = _forward_nonvalue(signature)
    if forwarded is not None:
        return forwarded
    assert isinstance(signature, PointContactSignature)
    gap = signature.exterior_gaps[0]
    degrees = float(gap.degrees)
    uncertainty = float(gap.uncertainty_degrees)
    return G.Present(
        degrees,
        G.Unit.DEGREES,
        gap.provenance + signature.provenance,
        max(0.0, degrees - uncertainty),
        min(360.0, degrees + uncertainty),
    )


def _large_gap(value: Any) -> G.Observation:
    signature = _context(value).point_contact()
    forwarded = _forward_nonvalue(signature)
    if forwarded is not None:
        return forwarded
    assert isinstance(signature, PointContactSignature)
    gap = signature.exterior_gaps[1]
    degrees = float(gap.degrees)
    uncertainty = float(gap.uncertainty_degrees)
    return G.Present(
        degrees,
        G.Unit.DEGREES,
        gap.provenance + signature.provenance,
        max(0.0, degrees - uncertainty),
        min(360.0, degrees + uncertainty),
    )


def _gap_ratio(value: Any) -> G.Observation:
    signature = _context(value).point_contact()
    forwarded = _forward_nonvalue(signature)
    if forwarded is not None:
        return forwarded
    assert isinstance(signature, PointContactSignature)
    small, large = signature.exterior_gaps
    small_low = float(small.degrees - small.uncertainty_degrees)
    small_high = float(small.degrees + small.uncertainty_degrees)
    large_low = max(0.0, float(
        large.degrees - large.uncertainty_degrees))
    large_high = float(large.degrees + large.uncertainty_degrees)
    if small_low <= 0.0 or small_high <= 0.0:
        return G.Indeterminate(
            "point-contact-fit-indeterminate",
            "gap-ratio denominator interval reaches zero",
            signature.provenance,
        )
    ratio = float(large.degrees / small.degrees)
    return G.Present(
        ratio,
        G.Unit.RATIO,
        signature.provenance,
        large_low / small_high,
        large_high / small_low,
    )


def default_grounded_observables(
        ) -> tuple[G.ObservableRegistry, tuple[ObservableDescriptor, ...]]:
    """Return the closed deterministic basis exposed to grounded proposers."""
    invariances = (
        G.Invariance.TRANSLATION,
        G.Invariance.ROTATION,
        G.Invariance.REFLECTION,
        G.Invariance.UNIFORM_SCALE,
    )
    absence = ("no_point_contact_signature",)
    indeterminate = ("point_contact_fit_indeterminate",)
    descriptors = (
        ObservableDescriptor(
            G.ObservableContract(
                SMALL_GAP_ID,
                G.ValueType.REAL,
                G.Unit.DEGREES,
                "point-contact.exterior-gap.small",
                G.Reducer.MIN,
                _small_gap,
                semantic_absence_modes=absence,
                indeterminate_modes=indeterminate,
                invariances=invariances,
            ),
            (
                "Smaller of the two cyclic cross-owner exterior gaps in the "
                "unique two-loop point-contact signature, with fit uncertainty."
            ),
            ("low", "high", "band"),
        ),
        ObservableDescriptor(
            G.ObservableContract(
                LARGE_GAP_ID,
                G.ValueType.REAL,
                G.Unit.DEGREES,
                "point-contact.exterior-gap.large",
                G.Reducer.MAX,
                _large_gap,
                semantic_absence_modes=absence,
                indeterminate_modes=indeterminate,
                invariances=invariances,
            ),
            (
                "Larger of the two cyclic cross-owner exterior gaps in the "
                "same unique two-loop point-contact signature."
            ),
            ("low", "high", "band"),
        ),
        ObservableDescriptor(
            G.ObservableContract(
                GAP_RATIO_ID,
                G.ValueType.REAL,
                G.Unit.RATIO,
                "point-contact.exterior-gap-asymmetry",
                G.Reducer.RATIO,
                _gap_ratio,
                semantic_absence_modes=absence,
                indeterminate_modes=indeterminate,
                invariances=invariances,
            ),
            (
                "Ratio of the large to small exterior gap, carrying the "
                "conservative interval induced by both angular fits."
            ),
            ("low", "high", "band"),
        ),
    )
    registry = G.ObservableRegistry()
    for descriptor in descriptors:
        registry.register(descriptor.contract)
    return registry, descriptors


__all__ = [
    "GAP_RATIO_ID",
    "GroundedPanelContext",
    "LARGE_GAP_ID",
    "ObservableDescriptor",
    "SMALL_GAP_ID",
    "default_grounded_observables",
]

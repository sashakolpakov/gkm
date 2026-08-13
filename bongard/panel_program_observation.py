"""Canonical hypothesis-preserving observations for panel programs.

This module is the neutral semantic boundary between a pixel observer and the
closed Bongard rule language.  It stores every complete minimum explanation;
it never stores task side, support/query role, generator program, or label.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import platform
import re
from typing import Any, Callable, Literal, Mapping

from bongard.canonical import canonical_digest
from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_STATE = Literal["identified", "ambiguous", "gap", "error"]

OBSERVATION_SCHEMA = "gkm.panel-program-observation.v2"
HYPOTHESIS_POLICY_ID = "complete-minimum-exact-program-hypotheses/v2"


class PanelProgramObservationError(ValueError):
    """An observer result or canonical record differs."""


def source_sha256() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _address(data: object) -> str:
    return "sha256:" + canonical_digest(data)


def _require_address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise PanelProgramObservationError(f"{label} must be a sha256: address")
    return value


def _exact_int_tuple(value: object, label: str, *, upper: int = 4096) -> tuple[int, ...]:
    if (
        type(value) is not tuple
        or any(type(item) is not int for item in value)
        or value != tuple(sorted(set(value)))
        or (value and (value[0] < 0 or value[-1] >= upper))
    ):
        raise PanelProgramObservationError(f"{label} differs")
    return value


def _exact_yx_tuple(value: object, label: str) -> tuple[tuple[int, int], ...]:
    if (
        type(value) is not tuple
        or any(
            type(point) is not tuple
            or len(point) != 2
            or any(type(item) is not int or not 0 <= item < 64 for item in point)
            for point in value
        )
        or value != tuple(sorted(set(value)))
    ):
        raise PanelProgramObservationError(f"{label} differs")
    return value


def _primitive_content(value: "ProgramPrimitiveObservation") -> dict[str, object]:
    return {
        "kind": value.kind,
        "ink_pixels": list(value.ink_pixels),
        "mask_digest": value.mask_digest,
        "boundary_pixels_yx": [list(point) for point in value.boundary_pixels_yx],
        "endpoints_yx": [list(point) for point in value.endpoints_yx],
        "path_ids": list(value.path_ids),
    }


@dataclass(frozen=True, slots=True)
class ProgramPrimitiveObservation:
    kind: Literal["line", "arc"]
    ink_pixels: tuple[int, ...]
    mask_digest: str
    boundary_pixels_yx: tuple[tuple[int, int], ...]
    endpoints_yx: tuple[tuple[int, int], ...]
    path_ids: tuple[int, ...]
    primitive_digest: str

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind not in ("line", "arc"):
            raise PanelProgramObservationError("primitive kind differs")
        ink = _exact_int_tuple(self.ink_pixels, "primitive ink")
        if not ink:
            raise PanelProgramObservationError("primitive ink is empty")
        _exact_yx_tuple(self.boundary_pixels_yx, "primitive boundary")
        _exact_yx_tuple(self.endpoints_yx, "primitive endpoints")
        _exact_int_tuple(self.path_ids, "primitive path IDs", upper=10_000)
        expected_mask = "sha256:" + hashlib.sha256(
            b"".join(item.to_bytes(2, "big") for item in ink)
        ).hexdigest()
        if self.mask_digest != expected_mask:
            raise PanelProgramObservationError("primitive mask digest differs")
        _require_address(self.primitive_digest, "primitive digest")
        if self.primitive_digest != _address(_primitive_content(self)):
            raise PanelProgramObservationError("primitive content digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_primitive_content(self), "primitive_digest": self.primitive_digest}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ProgramPrimitiveObservation":
        expected = {
            "kind", "ink_pixels", "mask_digest", "boundary_pixels_yx",
            "endpoints_yx", "path_ids", "primitive_digest",
        }
        if type(data) is not dict or set(data) != expected:
            raise PanelProgramObservationError("primitive fields differ")
        result = cls(
            data["kind"], tuple(data["ink_pixels"]), data["mask_digest"],
            tuple(tuple(point) for point in data["boundary_pixels_yx"]),
            tuple(tuple(point) for point in data["endpoints_yx"]),
            tuple(data["path_ids"]), data["primitive_digest"],
        )
        if result.to_data() != data:
            raise PanelProgramObservationError("primitive is not canonical")
        return result


def _hypothesis_content(value: "ProgramHypothesisObservation") -> dict[str, object]:
    return {
        "straight_count": value.straight_count,
        "arc_count": value.arc_count,
        "primitives": [item.to_data() for item in value.primitives],
        "reconstructed_ink_pixels": list(value.reconstructed_ink_pixels),
        "xor_pixel_count": value.xor_pixel_count,
        "intersection_over_union": value.intersection_over_union,
    }


@dataclass(frozen=True, slots=True)
class ProgramHypothesisObservation:
    straight_count: int
    arc_count: int
    primitives: tuple[ProgramPrimitiveObservation, ...]
    reconstructed_ink_pixels: tuple[int, ...]
    xor_pixel_count: int
    intersection_over_union: float
    hypothesis_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.straight_count) is not int
            or type(self.arc_count) is not int
            or not 1 <= self.straight_count + self.arc_count <= 9
            or type(self.primitives) is not tuple
            or len(self.primitives) != self.straight_count + self.arc_count
            or any(type(item) is not ProgramPrimitiveObservation for item in self.primitives)
        ):
            raise PanelProgramObservationError("hypothesis inventory differs")
        for item in self.primitives:
            ProgramPrimitiveObservation.__post_init__(item)
        if tuple(item.primitive_digest for item in self.primitives) != tuple(
            sorted(item.primitive_digest for item in self.primitives)
        ):
            raise PanelProgramObservationError("hypothesis primitives are not canonical")
        if (
            sum(item.kind == "line" for item in self.primitives) != self.straight_count
            or sum(item.kind == "arc" for item in self.primitives) != self.arc_count
        ):
            raise PanelProgramObservationError("hypothesis kind counts differ")
        reconstructed = _exact_int_tuple(
            self.reconstructed_ink_pixels, "hypothesis reconstructed ink"
        )
        expected = tuple(sorted({p for item in self.primitives for p in item.ink_pixels}))
        if reconstructed != expected:
            raise PanelProgramObservationError("hypothesis union differs")
        if type(self.xor_pixel_count) is not int or self.xor_pixel_count != 0:
            raise PanelProgramObservationError("hypothesis is not exact")
        if type(self.intersection_over_union) is not float or self.intersection_over_union != 1.0:
            raise PanelProgramObservationError("hypothesis IoU differs")
        _require_address(self.hypothesis_digest, "hypothesis digest")
        if self.hypothesis_digest != _address(_hypothesis_content(self)):
            raise PanelProgramObservationError("hypothesis content digest differs")

    @property
    def total_count(self) -> int:
        return self.straight_count + self.arc_count

    @property
    def mix(self) -> str:
        if self.arc_count == 0:
            return "straight_only"
        if self.straight_count == 0:
            return "arc_only"
        return "mixed"

    def to_data(self) -> dict[str, object]:
        return {**_hypothesis_content(self), "hypothesis_digest": self.hypothesis_digest}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ProgramHypothesisObservation":
        expected = {
            "straight_count", "arc_count", "primitives", "reconstructed_ink_pixels",
            "xor_pixel_count", "intersection_over_union", "hypothesis_digest",
        }
        if type(data) is not dict or set(data) != expected or type(data["primitives"]) is not list:
            raise PanelProgramObservationError("hypothesis fields differ")
        result = cls(
            data["straight_count"], data["arc_count"],
            tuple(ProgramPrimitiveObservation.from_data(item) for item in data["primitives"]),
            tuple(data["reconstructed_ink_pixels"]), data["xor_pixel_count"],
            data["intersection_over_union"], data["hypothesis_digest"],
        )
        if result.to_data() != data:
            raise PanelProgramObservationError("hypothesis is not canonical")
        return result


def _observation_content(value: "PanelProgramObservation") -> dict[str, object]:
    return {
        "schema": OBSERVATION_SCHEMA,
        "panel_png_digest": value.panel_png_digest,
        "observer_source_digest": value.observer_source_digest,
        "observer_algorithm_digest": value.observer_algorithm_digest,
        "search_space_digest": value.search_space_digest,
        "hypothesis_policy_digest": value.hypothesis_policy_digest,
        "state": value.state,
        "reason": value.reason,
        "error_type": value.error_type,
        "foreground_pixel_count": value.foreground_pixel_count,
        "skeleton_pixel_count": value.skeleton_pixel_count,
        "minimum_primitive_count": value.minimum_primitive_count,
        "hypotheses": [item.to_data() for item in value.hypotheses],
    }


@dataclass(frozen=True, slots=True)
class PanelProgramObservation:
    panel_png_digest: str
    observer_source_digest: str
    observer_algorithm_digest: str
    search_space_digest: str
    hypothesis_policy_digest: str
    state: _STATE
    reason: str | None
    error_type: str | None
    foreground_pixel_count: int | None
    skeleton_pixel_count: int | None
    minimum_primitive_count: int | None
    hypotheses: tuple[ProgramHypothesisObservation, ...]
    observation_digest: str

    def __post_init__(self) -> None:
        for label, value in (
            ("panel PNG digest", self.panel_png_digest),
            ("observer source digest", self.observer_source_digest),
            ("observer algorithm digest", self.observer_algorithm_digest),
            ("search-space digest", self.search_space_digest),
            ("hypothesis policy digest", self.hypothesis_policy_digest),
            ("observation digest", self.observation_digest),
        ):
            _require_address(value, label)
        if type(self.state) is not str or self.state not in ("identified", "ambiguous", "gap", "error"):
            raise PanelProgramObservationError("observation state differs")
        if type(self.hypotheses) is not tuple or any(
            type(item) is not ProgramHypothesisObservation for item in self.hypotheses
        ):
            raise PanelProgramObservationError("observation hypotheses differ")
        for item in self.hypotheses:
            ProgramHypothesisObservation.__post_init__(item)
        if tuple(item.hypothesis_digest for item in self.hypotheses) != tuple(
            sorted({item.hypothesis_digest for item in self.hypotheses})
        ):
            raise PanelProgramObservationError("observation hypothesis order differs")
        if self.state in ("identified", "ambiguous"):
            expected = 1 if self.state == "identified" else None
            reconstructed = {
                item.reconstructed_ink_pixels for item in self.hypotheses
            }
            if (
                self.reason is not None or self.error_type is not None
                or (expected is not None and len(self.hypotheses) != expected)
                or (self.state == "ambiguous" and len(self.hypotheses) < 2)
                or any(type(v) is not int or v <= 0 for v in (
                    self.foreground_pixel_count, self.skeleton_pixel_count,
                    self.minimum_primitive_count,
                ))
                or any(item.total_count != self.minimum_primitive_count for item in self.hypotheses)
                or len(reconstructed) != 1
                or len(next(iter(reconstructed), ())) != self.foreground_pixel_count
            ):
                raise PanelProgramObservationError("successful observation payload differs")
        else:
            if (
                type(self.reason) is not str or not self.reason
                or self.hypotheses or self.minimum_primitive_count is not None
                or (self.state == "error") != (type(self.error_type) is str and bool(self.error_type))
                or (self.state == "gap" and self.error_type is not None)
            ):
                raise PanelProgramObservationError("failed observation payload differs")
            for value in (self.foreground_pixel_count, self.skeleton_pixel_count):
                if value is not None and (type(value) is not int or value <= 0):
                    raise PanelProgramObservationError("failed observation count differs")
        if self.observation_digest != _address(_observation_content(self)):
            raise PanelProgramObservationError("observation content digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_observation_content(self), "observation_digest": self.observation_digest}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "PanelProgramObservation":
        expected = {
            "schema", "panel_png_digest", "observer_source_digest",
            "observer_algorithm_digest", "search_space_digest", "hypothesis_policy_digest",
            "state", "reason", "error_type", "foreground_pixel_count",
            "skeleton_pixel_count", "minimum_primitive_count", "hypotheses",
            "observation_digest",
        }
        if type(data) is not dict or set(data) != expected or data["schema"] != OBSERVATION_SCHEMA:
            raise PanelProgramObservationError("observation fields differ")
        result = cls(
            data["panel_png_digest"], data["observer_source_digest"],
            data["observer_algorithm_digest"], data["search_space_digest"],
            data["hypothesis_policy_digest"], data["state"], data["reason"],
            data["error_type"], data["foreground_pixel_count"],
            data["skeleton_pixel_count"], data["minimum_primitive_count"],
            tuple(ProgramHypothesisObservation.from_data(item) for item in data["hypotheses"]),
            data["observation_digest"],
        )
        if result.to_data() != data:
            raise PanelProgramObservationError("observation is not canonical")
        return result


def _make_primitive(value: object) -> ProgramPrimitiveObservation:
    values = {
        "kind": value.kind,
        "ink_pixels": tuple(value.ink_pixels),
        "mask_digest": value.mask_sha256,
        "boundary_pixels_yx": tuple(value.boundary_pixels_yx),
        "endpoints_yx": tuple(value.endpoints_yx),
        "path_ids": tuple(value.path_ids),
    }
    provisional = object.__new__(ProgramPrimitiveObservation)
    for key, item in values.items():
        object.__setattr__(provisional, key, item)
    return ProgramPrimitiveObservation(
        **values, primitive_digest=_address(_primitive_content(provisional))
    )


def _make_hypothesis(value: object) -> ProgramHypothesisObservation:
    primitives = tuple(sorted((_make_primitive(item) for item in value.primitives), key=lambda x: x.primitive_digest))
    values = {
        "straight_count": value.straight_count,
        "arc_count": value.arc_count,
        "primitives": primitives,
        "reconstructed_ink_pixels": tuple(value.reconstructed_ink_pixels),
        "xor_pixel_count": value.xor_pixel_count,
        "intersection_over_union": value.intersection_over_union,
    }
    provisional = object.__new__(ProgramHypothesisObservation)
    for key, item in values.items():
        object.__setattr__(provisional, key, item)
    return ProgramHypothesisObservation(
        **values, hypothesis_digest=_address(_hypothesis_content(provisional))
    )


def _connected_search_space_digest() -> str:
    from bongard import panel_action_count_connected_synthesizer as synthesizer

    return synthesizer.sealed_catalog_digest()


def _observer_dependency_bindings() -> dict[str, str]:
    """Bind every implementation/runtime dependency used by the raw fitter."""

    import numpy as np
    import PIL
    import scipy
    from bongard import panel_action_count_ordered_path_inversion as ordered
    from bongard import panel_action_count_synthetic_identifiability as identifiability

    return {
        "panel_program_ordered_path_source": "sha256:" + ordered.source_sha256(),
        "panel_program_ordered_fixture_source": (
            "sha256:" + identifiability.source_sha256()
        ),
        "panel_program_runtime_dependencies": _address(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pillow": PIL.__version__,
                "scipy": scipy.__version__,
            }
        ),
    }


def _algorithm_digest(
    observer_source_digest: str,
    search_space_digest: str,
    dependency_bindings: Mapping[str, str],
) -> str:
    return _address({
        "algorithm": "connected-complete-minimum-exact-cover-to-program-observation/v2",
        "observer_source_digest": observer_source_digest,
        "observation_source_digest": "sha256:" + source_sha256(),
        "search_space_digest": search_space_digest,
        "hypothesis_policy": HYPOTHESIS_POLICY_ID,
        "dependency_bindings": dict(sorted(dependency_bindings.items())),
    })


def connected_program_observer_bindings() -> dict[str, str]:
    """Return metadata-only bindings that must be frozen before pixel release."""

    from bongard import panel_action_count_connected_synthetic as connected
    from bongard import panel_action_count_connected_synthesizer as synthesizer

    observer_source = "sha256:" + synthesizer.source_sha256()
    search_space = _connected_search_space_digest()
    policy = _address({"id": HYPOTHESIS_POLICY_ID, "complete": True, "maximum": 9})
    dependencies = _observer_dependency_bindings()
    return {
        "panel_program_observation_source": "sha256:" + source_sha256(),
        "panel_program_observer_source": observer_source,
        "panel_program_observer_algorithm": _algorithm_digest(
            observer_source, search_space, dependencies
        ),
        "panel_program_search_space": search_space,
        "panel_program_connected_catalog": search_space,
        "panel_program_hypothesis_policy": policy,
        "panel_program_fixture_source": "sha256:" + connected.source_sha256(),
        **dependencies,
    }


def adapt_connected_fit_outcome(
    png_bytes: bytes,
    outcome: object,
) -> PanelProgramObservation:
    """Adapt a raw fit and bind it to the exact supplied bytes.

    This neutral adapter does not consult synthetic issuance, task roles, or a
    target oracle.  It does independently verify that successful hypotheses
    reconstruct the foreground of these exact PNG bytes.
    """

    if type(png_bytes) is not bytes:
        raise TypeError("PNG transport must be exact bytes")
    panel_png_digest = "sha256:" + hashlib.sha256(png_bytes).hexdigest()
    from bongard import panel_action_count_connected_synthesizer as synthesizer

    if type(outcome) is not synthesizer.ConnectedFitOutcome:
        raise TypeError("outcome must be exact ConnectedFitOutcome")
    synthesizer.ConnectedFitOutcome.__post_init__(outcome)
    canonical_outcome = synthesizer.fit_authenticated_png_hypotheses(png_bytes)
    if outcome != canonical_outcome:
        raise PanelProgramObservationError(
            "fit outcome is not the complete canonical replay for these PNG bytes"
        )
    if outcome.hypotheses:
        try:
            from PIL import Image

            with Image.open(BytesIO(png_bytes)) as image:
                if image.format != "PNG" or image.size != (64, 64):
                    raise PanelProgramObservationError("fit parent is not one 64x64 PNG")
                image.load()
                gray = image.convert("L")
                foreground = tuple(
                    index for index, value in enumerate(gray.tobytes()) if value < 128
                )
        except PanelProgramObservationError:
            raise
        except Exception as exc:
            raise PanelProgramObservationError("cannot decode fit parent PNG") from exc
        if any(
            tuple(item.reconstructed_ink_pixels) != foreground
            for item in outcome.hypotheses
        ):
            raise PanelProgramObservationError(
                "fit hypotheses do not reconstruct the supplied PNG"
            )
    observer_source = "sha256:" + synthesizer.source_sha256()
    search_space = _connected_search_space_digest()
    dependencies = _observer_dependency_bindings()
    policy = _address({"id": HYPOTHESIS_POLICY_ID, "complete": True, "maximum": 9})
    hypotheses = tuple(sorted((_make_hypothesis(item) for item in outcome.hypotheses), key=lambda item: item.hypothesis_digest))
    values = {
        "panel_png_digest": panel_png_digest,
        "observer_source_digest": observer_source,
        "observer_algorithm_digest": _algorithm_digest(
            observer_source, search_space, dependencies
        ),
        "search_space_digest": search_space,
        "hypothesis_policy_digest": policy,
        "state": outcome.disposition.lower(),
        "reason": outcome.reason,
        "error_type": None,
        "foreground_pixel_count": outcome.foreground_pixel_count,
        "skeleton_pixel_count": outcome.skeleton_pixel_count,
        "minimum_primitive_count": outcome.minimum_primitive_count,
        "hypotheses": hypotheses,
    }
    provisional = object.__new__(PanelProgramObservation)
    for key, item in values.items():
        object.__setattr__(provisional, key, item)
    return PanelProgramObservation(
        **values, observation_digest=_address(_observation_content(provisional))
    )


def _observe_program_png(
    png_bytes: bytes,
    *,
    fitter: Callable[[bytes], object],
) -> PanelProgramObservation:
    """Run one exact fitter and freeze its independently replayed hypotheses."""

    if type(png_bytes) is not bytes:
        raise TypeError("PNG transport must be exact bytes")
    digest = "sha256:" + hashlib.sha256(png_bytes).hexdigest()
    try:
        outcome = fitter(png_bytes)
        return adapt_connected_fit_outcome(png_bytes, outcome)
    except Exception as exc:
        # Transport/fitter failures remain typed errors rather than Boolean false.
        try:
            from bongard import panel_action_count_connected_synthesizer as synthesizer

            observer_source = "sha256:" + synthesizer.source_sha256()
            search_space = _connected_search_space_digest()
            dependencies = _observer_dependency_bindings()
        except Exception:
            observer_source = "sha256:" + source_sha256()
            search_space = _address({"unavailable": True})
            dependencies = _address({"unavailable": True})
        policy = _address({"id": HYPOTHESIS_POLICY_ID, "complete": True, "maximum": 9})
        values = {
            "panel_png_digest": digest,
            "observer_source_digest": observer_source,
            "observer_algorithm_digest": _algorithm_digest(
                observer_source,
                search_space,
                dependencies
                if isinstance(dependencies, dict)
                else {"panel_program_runtime_dependencies": dependencies},
            ),
            "search_space_digest": search_space,
            "hypothesis_policy_digest": policy,
            "state": "error",
            "reason": str(exc) or type(exc).__name__,
            "error_type": type(exc).__name__,
            "foreground_pixel_count": None,
            "skeleton_pixel_count": None,
            "minimum_primitive_count": None,
            "hypotheses": (),
        }
        provisional = object.__new__(PanelProgramObservation)
        for key, item in values.items():
            object.__setattr__(provisional, key, item)
        return PanelProgramObservation(
            **values, observation_digest=_address(_observation_content(provisional))
        )


def observe_authenticated_program_png(png_bytes: bytes) -> PanelProgramObservation:
    """Observe bytes already authenticated by a distinct release capability.

    This function accepts only bytes and performs no archive, path, role, label,
    or synthetic-issuer lookup.  Calling it is not release authority; the
    official custody adapter invokes it only after a typed panel release.
    """

    from bongard.panel_action_count_connected_synthesizer import (
        fit_authenticated_png_hypotheses,
    )

    return _observe_program_png(
        png_bytes, fitter=fit_authenticated_png_hypotheses
    )


def observe_connected_program_png(
    png_bytes: bytes,
) -> PanelProgramObservation:
    """Run the synthetic-issued observer and freeze its hypotheses."""

    from bongard.panel_action_count_connected_synthesizer import fit_png_hypotheses

    return _observe_program_png(png_bytes, fitter=fit_png_hypotheses)


__all__ = (
    "HYPOTHESIS_POLICY_ID", "OBSERVATION_SCHEMA", "PanelProgramObservation",
    "PanelProgramObservationError", "ProgramHypothesisObservation",
    "ProgramPrimitiveObservation", "adapt_connected_fit_outcome",
    "connected_program_observer_bindings", "observe_authenticated_program_png",
    "observe_connected_program_png",
    "source_sha256",
)

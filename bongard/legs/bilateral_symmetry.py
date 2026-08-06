"""Candidate-independent quantitative bilateral-symmetry evidence.

The visual proposal is deliberately absent from this module.  The operation
consumes only exact panel bytes (or an explicitly typed test array), applies a
fixed luminance/threshold procedure, and searches a fixed reflection-axis
grid.  It reports *how much* reflected ink is supported by nearby observed
ink.  It does not assert that an object is semantically "a matched pair of
lobes" and it does not choose a threshold after seeing Bongard labels.

The score is a fraction in ``[0, 1]``.  Its uncertainty interval is the
deterministic sensitivity envelope over three fixed foreground thresholds.
This is numerical/preprocessing uncertainty, not a calibrated population
confidence interval.  A closed-IR ``AT_LEAST`` atom must make the subsequent
semantic decision and becomes indeterminate when this envelope straddles its
frozen threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
from typing import NoReturn

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION

from bongard.evidence import Disposition, Evidence, Provenance, Uncertainty
from bongard.legs.contracts import (
    AffirmativeRelation,
    InvarianceContract,
    LegContract,
    LegReference,
    LegRegistry,
    LegSemantics,
    PANEL,
    Transform,
    Unit,
    ValueType,
)


ALGORITHM_ID = "centroid-reflection-ink-coverage-grid/v1"
LEG_NAME = "bilateral_symmetry_score"
LEG_VERSION = "1.0.0"
BILATERAL_SYMMETRY_SCORE = ValueType("measurement", Unit.FRACTION)

# These values are part of the operation digest.  They are not proposer
# arguments and therefore cannot be tuned per candidate or support set.
_FOREGROUND_THRESHOLDS = (96, 128, 160)
_REFERENCE_THRESHOLD = 128
_COARSE_AXIS_STEP_DEGREES = 2.0
_REFINE_AXIS_STEP_DEGREES = 0.25
_REFINE_RADIUS_DEGREES = 2.0
_MATCH_TOLERANCE_FRACTION = 0.0125
_MIN_MATCH_TOLERANCE_PIXELS = 1
_MAX_MATCH_TOLERANCE_PIXELS = 12
_MIN_FOREGROUND_PIXELS = 32
_MIN_BOUNDING_DIAGONAL_PIXELS = 8.0
_MAX_PANEL_PIXELS = 4096 * 4096


class BilateralSymmetryInputError(ValueError):
    """The panel cannot enter the fixed raster measurement."""

    def __init__(self, message: str, *, input_digest: str | None = None) -> None:
        super().__init__(message)
        self.input_digest = input_digest


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _operation_data() -> dict[str, object]:
    return {
        "algorithm": ALGORITHM_ID,
        "axis_grid": {
            "coarse_step_degrees": _COARSE_AXIS_STEP_DEGREES,
            "refine_radius_degrees": _REFINE_RADIUS_DEGREES,
            "refine_step_degrees": _REFINE_AXIS_STEP_DEGREES,
        },
        "decoder": {
            "numpy": np.__version__,
            "pillow": PILLOW_VERSION,
            "rgba_background": 255,
            "rgb_to_luminance_integer_weights": [299, 587, 114],
        },
        "foreground_thresholds": list(_FOREGROUND_THRESHOLDS),
        "reference_threshold": _REFERENCE_THRESHOLD,
        "match_tolerance": {
            "bounding_diagonal_fraction": _MATCH_TOLERANCE_FRACTION,
            "minimum_pixels": _MIN_MATCH_TOLERANCE_PIXELS,
            "maximum_pixels": _MAX_MATCH_TOLERANCE_PIXELS,
        },
        "measurement_guards": {
            "maximum_panel_pixels": _MAX_PANEL_PIXELS,
            "minimum_bounding_diagonal_pixels": _MIN_BOUNDING_DIAGONAL_PIXELS,
            "minimum_foreground_pixels": _MIN_FOREGROUND_PIXELS,
            "reject_border_clipping": True,
        },
    }


def _module_source_sha256() -> str:
    """Bind helper code as well as the small registered projection wrapper."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def operation_digest() -> str:
    """Return the environment-and-source-bound operational identity."""

    payload = {
        "operation": _operation_data(),
        "module_source_sha256": _module_source_sha256(),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _array_input_digest(array: np.ndarray, mode: str) -> str:
    header = _canonical_json_bytes(
        {
            "schema": "bongard.ndarray-panel/v1",
            "dtype": array.dtype.str,
            "mode": mode,
            "shape": list(array.shape),
        }
    )
    hasher = hashlib.sha256()
    hasher.update(len(header).to_bytes(8, "big"))
    hasher.update(header)
    hasher.update(np.ascontiguousarray(array).tobytes(order="C"))
    return hasher.hexdigest()


def _rgba_to_luminance(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.uint32, copy=False)
    rgb = values[..., :3]
    if values.shape[2] == 4:
        alpha = values[..., 3:4]
        rgb = (rgb * alpha + 255 * (255 - alpha) + 127) // 255
    luminance = (
        299 * rgb[..., 0] + 587 * rgb[..., 1] + 114 * rgb[..., 2] + 500
    ) // 1000
    return np.ascontiguousarray(luminance.astype(np.uint8))


def _validate_dimensions(array: np.ndarray) -> None:
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise BilateralSymmetryInputError("panel dimensions must both be at least two")
    if array.shape[0] * array.shape[1] > _MAX_PANEL_PIXELS:
        raise BilateralSymmetryInputError("panel exceeds the fixed pixel-count limit")


def _decode_array(panel: np.ndarray) -> tuple[np.ndarray, str, str]:
    raw_mode = (
        "L"
        if panel.ndim == 2
        else "RGB"
        if panel.ndim == 3 and panel.shape[2:] == (3,)
        else "RGBA"
        if panel.ndim == 3 and panel.shape[2:] == (4,)
        else f"raw-{panel.ndim}d"
    )
    input_digest = _array_input_digest(panel, raw_mode)
    if panel.dtype != np.uint8:
        raise BilateralSymmetryInputError(
            "array panel dtype must be exactly uint8", input_digest=input_digest
        )
    if panel.ndim == 2:
        mode = "L"
        luminance = np.ascontiguousarray(panel)
    elif panel.ndim == 3 and panel.shape[2] in (3, 4):
        mode = "RGB" if panel.shape[2] == 3 else "RGBA"
        luminance = _rgba_to_luminance(panel)
    else:
        raise BilateralSymmetryInputError(
            "array panel must have shape (height,width), (height,width,3), "
            "or (height,width,4)",
            input_digest=input_digest,
        )
    try:
        _validate_dimensions(luminance)
    except BilateralSymmetryInputError as exc:
        raise BilateralSymmetryInputError(
            str(exc), input_digest=input_digest
        ) from exc
    return luminance, input_digest, f"numpy-{mode}"


def _decode_png(raw: bytes) -> tuple[np.ndarray, str]:
    try:
        with Image.open(BytesIO(raw)) as encoded:
            if encoded.format != "PNG":
                raise BilateralSymmetryInputError("encoded panel must be a PNG")
            if getattr(encoded, "n_frames", 1) != 1:
                raise BilateralSymmetryInputError("encoded panel must have one frame")
            width, height = encoded.size
            if width < 2 or height < 2 or width * height > _MAX_PANEL_PIXELS:
                raise BilateralSymmetryInputError(
                    "encoded panel dimensions violate the fixed limits"
                )
            rgba = np.asarray(encoded.convert("RGBA"), dtype=np.uint8)
    except BilateralSymmetryInputError:
        raise
    except Exception as exc:  # noqa: BLE001 - decode disposition boundary.
        raise BilateralSymmetryInputError(
            f"PNG decoding failed: {type(exc).__name__}: {exc}"
        ) from exc
    luminance = _rgba_to_luminance(rgba)
    _validate_dimensions(luminance)
    return luminance, "pillow-png-rgba"


def _decode_panel(panel: object) -> tuple[np.ndarray, str, str]:
    if isinstance(panel, np.ndarray):
        return _decode_array(panel)
    if isinstance(panel, (str, Path)):
        try:
            raw = Path(panel).read_bytes()
        except OSError as exc:
            raise BilateralSymmetryInputError(
                f"panel bytes could not be read: {type(exc).__name__}: {exc}"
            ) from exc
    elif isinstance(panel, bytes):
        raw = panel
    else:
        raise BilateralSymmetryInputError(
            "panel must be a PNG path, PNG bytes, or uint8 luminance/RGB array"
        )
    digest = hashlib.sha256(raw).hexdigest()
    try:
        luminance, decoder = _decode_png(raw)
    except BilateralSymmetryInputError as exc:
        raise BilateralSymmetryInputError(
            str(exc), input_digest=digest
        ) from exc
    return luminance, digest, decoder


def _fallback_input_digest(panel: object) -> str:
    identity = {
        "schema": "bongard.unreadable-panel-identity/v1",
        "python_type": f"{type(panel).__module__}.{type(panel).__qualname__}",
    }
    return hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()


def _provenance(
    input_digest: str,
    method: str,
    *,
    details: tuple[tuple[str, str], ...] = (),
) -> Provenance:
    common = (
        ("algorithm_id", ALGORITHM_ID),
        ("operation_digest", operation_digest()),
    )
    return Provenance(
        producer="bongard.bilateral_symmetry",
        version=LEG_VERSION,
        method=method,
        input_digests=(input_digest,),
        details=tuple(sorted((*common, *details))),
    )


def _disk_dilation(mask: np.ndarray, radius: int) -> np.ndarray:
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask)
    for delta_y in range(-radius, radius + 1):
        for delta_x in range(-radius, radius + 1):
            if delta_x * delta_x + delta_y * delta_y > radius * radius:
                continue
            result |= padded[
                radius + delta_y : radius + delta_y + height,
                radius + delta_x : radius + delta_x + width,
            ]
    return result


def _reflection_match_fraction(
    xs: np.ndarray,
    ys: np.ndarray,
    centroid_x: float,
    centroid_y: float,
    angle_degrees: float,
    dilated: np.ndarray,
) -> float:
    radians = math.radians(angle_degrees)
    axis_x = math.cos(radians)
    axis_y = math.sin(radians)
    centered_x = xs - centroid_x
    centered_y = ys - centroid_y
    projection = centered_x * axis_x + centered_y * axis_y
    reflected_x = np.rint(
        centroid_x + 2.0 * projection * axis_x - centered_x
    ).astype(np.int64)
    reflected_y = np.rint(
        centroid_y + 2.0 * projection * axis_y - centered_y
    ).astype(np.int64)
    valid = (
        (reflected_x >= 0)
        & (reflected_x < dilated.shape[1])
        & (reflected_y >= 0)
        & (reflected_y < dilated.shape[0])
    )
    matches = np.zeros(xs.size, dtype=bool)
    matches[valid] = dilated[reflected_y[valid], reflected_x[valid]]
    return float(np.count_nonzero(matches) / xs.size)


def _better_axis(
    candidate: tuple[float, float], current: tuple[float, float] | None
) -> bool:
    if current is None:
        return True
    candidate_score, candidate_angle = candidate
    current_score, current_angle = current
    return candidate_score > current_score or (
        candidate_score == current_score and candidate_angle < current_angle
    )


@dataclass(frozen=True)
class _MaskScore:
    score: float
    axis_degrees: float
    foreground_pixels: int
    bounding_width_pixels: int
    bounding_height_pixels: int
    bounding_diagonal_pixels: float
    match_tolerance_pixels: int


def _score_mask(mask: np.ndarray) -> _MaskScore | str:
    ys, xs = np.nonzero(mask)
    foreground_pixels = int(xs.size)
    if foreground_pixels == 0:
        return "absent"
    if (
        bool(mask[0, :].any())
        or bool(mask[-1, :].any())
        or bool(mask[:, 0].any())
        or bool(mask[:, -1].any())
    ):
        return "border_clipped"
    if foreground_pixels < _MIN_FOREGROUND_PIXELS:
        return "insufficient_foreground"

    minimum_x = int(xs.min())
    maximum_x = int(xs.max())
    minimum_y = int(ys.min())
    maximum_y = int(ys.max())
    width = maximum_x - minimum_x + 1
    height = maximum_y - minimum_y + 1
    diagonal = math.hypot(width, height)
    if diagonal < _MIN_BOUNDING_DIAGONAL_PIXELS:
        return "insufficient_spatial_extent"

    tolerance = int(round(diagonal * _MATCH_TOLERANCE_FRACTION))
    tolerance = min(
        _MAX_MATCH_TOLERANCE_PIXELS,
        max(_MIN_MATCH_TOLERANCE_PIXELS, tolerance),
    )
    dilated = _disk_dilation(mask, tolerance)
    float_xs = xs.astype(np.float64)
    float_ys = ys.astype(np.float64)
    centroid_x = float(float_xs.mean())
    centroid_y = float(float_ys.mean())

    best: tuple[float, float] | None = None
    coarse_count = int(round(180.0 / _COARSE_AXIS_STEP_DEGREES))
    for index in range(coarse_count):
        angle = index * _COARSE_AXIS_STEP_DEGREES
        score = _reflection_match_fraction(
            float_xs,
            float_ys,
            centroid_x,
            centroid_y,
            angle,
            dilated,
        )
        candidate = (score, angle)
        if _better_axis(candidate, best):
            best = candidate
    assert best is not None

    best_coarse_angle = best[1]
    refine_count = int(
        round(2.0 * _REFINE_RADIUS_DEGREES / _REFINE_AXIS_STEP_DEGREES)
    )
    refined_angles = {
        round(
            (
                best_coarse_angle
                - _REFINE_RADIUS_DEGREES
                + index * _REFINE_AXIS_STEP_DEGREES
            )
            % 180.0,
            10,
        )
        for index in range(refine_count + 1)
    }
    for angle in sorted(refined_angles):
        score = _reflection_match_fraction(
            float_xs,
            float_ys,
            centroid_x,
            centroid_y,
            angle,
            dilated,
        )
        candidate = (score, angle)
        if _better_axis(candidate, best):
            best = candidate

    return _MaskScore(
        score=best[0],
        axis_degrees=best[1],
        foreground_pixels=foreground_pixels,
        bounding_width_pixels=width,
        bounding_height_pixels=height,
        bounding_diagonal_pixels=diagonal,
        match_tolerance_pixels=tolerance,
    )


@dataclass(frozen=True)
class BilateralSymmetryObservation:
    """A typed numerical observation, explicitly not semantic truth."""

    score: float
    support: Uncertainty
    best_axis_degrees: float
    foreground_pixels: int
    bounding_width_pixels: int
    bounding_height_pixels: int
    bounding_diagonal_pixels: float
    match_tolerance_pixels: int
    threshold_scores: tuple[tuple[int, float], ...]
    input_digest: str
    operation_digest: str
    provenance: Provenance

    def __post_init__(self) -> None:
        scalar_values = (
            self.score,
            self.best_axis_degrees,
            self.bounding_diagonal_pixels,
        )
        if any(not math.isfinite(value) for value in scalar_values):
            raise ValueError("bilateral-symmetry observation must be finite")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("bilateral-symmetry score must lie in [0, 1]")
        if not self.support.lower <= self.score <= self.support.upper:
            raise ValueError("score must lie inside its support interval")
        if not 0.0 <= self.best_axis_degrees < 180.0:
            raise ValueError("reflection axis must lie in [0, 180) degrees")
        for name, value in (
            ("foreground_pixels", self.foreground_pixels),
            ("bounding_width_pixels", self.bounding_width_pixels),
            ("bounding_height_pixels", self.bounding_height_pixels),
            ("match_tolerance_pixels", self.match_tolerance_pixels),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if tuple(threshold for threshold, _ in self.threshold_scores) != (
            _FOREGROUND_THRESHOLDS
        ):
            raise ValueError("threshold scores differ from the fixed operation")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for _, value in self.threshold_scores
        ):
            raise ValueError("threshold scores must be finite fractions")
        if len(self.input_digest) != 64 or len(self.operation_digest) != 64:
            raise ValueError("observation digests must be lowercase sha256 values")
        if any(
            character not in "0123456789abcdef"
            for digest in (self.input_digest, self.operation_digest)
            for character in digest
        ):
            raise ValueError("observation digests must be lowercase sha256 values")
        if self.provenance.input_digests != (self.input_digest,):
            raise ValueError("observation provenance does not bind its panel")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": "bongard.bilateral-symmetry-observation/v1",
            "score": self.score,
            "support": {
                "lower": self.support.lower,
                "upper": self.support.upper,
                "confidence_level": self.support.confidence_level,
                "causes": list(self.support.causes),
            },
            "best_axis_degrees": self.best_axis_degrees,
            "foreground_pixels": self.foreground_pixels,
            "bounding_width_pixels": self.bounding_width_pixels,
            "bounding_height_pixels": self.bounding_height_pixels,
            "bounding_diagonal_pixels": self.bounding_diagonal_pixels,
            "match_tolerance_pixels": self.match_tolerance_pixels,
            "threshold_scores": [list(item) for item in self.threshold_scores],
            "input_digest": self.input_digest,
            "operation_digest": self.operation_digest,
            "provenance_digest": self.provenance.digest(),
        }

    def digest(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.to_data())).hexdigest()

    def __bool__(self) -> NoReturn:
        raise TypeError(
            "bilateral-symmetry observation is a measurement, not truth; "
            "compare its full interval through the closed IR"
        )


def _nonpresent(
    disposition: Disposition,
    provenance: Provenance,
    *,
    reason: str | None = None,
    certificate: str | None = None,
    error_type: str | None = None,
    uncertainty: Uncertainty | None = None,
) -> Evidence[BilateralSymmetryObservation]:
    if disposition is Disposition.CERTIFIED_ABSENT:
        assert certificate is not None
        return Evidence.certified_absent(provenance, certificate, uncertainty)
    if disposition is Disposition.INDETERMINATE:
        assert reason is not None
        return Evidence.indeterminate(provenance, reason, uncertainty)
    if disposition is Disposition.ERROR:
        assert reason is not None and error_type is not None
        return Evidence.error(provenance, error_type, reason)
    raise AssertionError("non-present helper received PRESENT")


def measure_bilateral_symmetry(
    panel: object,
) -> Evidence[BilateralSymmetryObservation]:
    """Measure a fixed best-axis reflected-ink agreement from panel pixels."""

    try:
        luminance, input_digest, decoder = _decode_panel(panel)
    except Exception as exc:  # noqa: BLE001 - evidence disposition boundary.
        input_digest = getattr(exc, "input_digest", None) or _fallback_input_digest(
            panel
        )
        origin = _provenance(
            input_digest,
            "input_error",
            details=(("input_type", type(panel).__name__),),
        )
        return _nonpresent(
            Disposition.ERROR,
            origin,
            error_type=type(exc).__name__,
            reason=str(exc) or repr(exc),
        )

    threshold_results = tuple(
        (threshold, _score_mask(luminance <= threshold))
        for threshold in _FOREGROUND_THRESHOLDS
    )
    state_details = tuple(
        (
            f"threshold_{threshold}",
            "measured" if isinstance(result, _MaskScore) else result,
        )
        for threshold, result in threshold_results
    )
    base_details = (
        ("decoder", decoder),
        ("height_pixels", str(luminance.shape[0])),
        *state_details,
        ("width_pixels", str(luminance.shape[1])),
    )

    if all(result == "absent" for _, result in threshold_results):
        origin = _provenance(input_digest, "foreground_absence", details=base_details)
        return _nonpresent(
            Disposition.CERTIFIED_ABSENT,
            origin,
            certificate=(
                "all fixed luminance thresholds found exactly zero foreground "
                f"pixels; operation={operation_digest()}"
            ),
        )

    if any(not isinstance(result, _MaskScore) for _, result in threshold_results):
        origin = _provenance(input_digest, "measurement_guard", details=base_details)
        states = ", ".join(
            f"{threshold}:{'measured' if isinstance(result, _MaskScore) else result}"
            for threshold, result in threshold_results
        )
        return _nonpresent(
            Disposition.INDETERMINATE,
            origin,
            reason=f"fixed preprocessing ensemble is not fully measurable ({states})",
            uncertainty=Uncertainty(
                0.0,
                1.0,
                causes=("preprocessing_guard_or_visibility",),
            ),
        )

    measured = tuple(
        (threshold, result)
        for threshold, result in threshold_results
        if isinstance(result, _MaskScore)
    )
    reference = next(
        result for threshold, result in measured if threshold == _REFERENCE_THRESHOLD
    )
    score_pairs = tuple((threshold, result.score) for threshold, result in measured)
    lower = min(score for _, score in score_pairs)
    upper = max(score for _, score in score_pairs)
    support = Uncertainty(
        lower,
        upper,
        causes=("fixed_luminance_threshold_sensitivity",),
    )
    origin = _provenance(
        input_digest,
        "deterministic_reflection_grid",
        details=(
            ("axis_degrees", format(reference.axis_degrees, ".10g")),
            ("decoder", decoder),
            ("foreground_pixels", str(reference.foreground_pixels)),
            ("match_tolerance_pixels", str(reference.match_tolerance_pixels)),
            ("reference_threshold", str(_REFERENCE_THRESHOLD)),
            ("support_lower", format(lower, ".17g")),
            ("support_upper", format(upper, ".17g")),
        ),
    )
    observation = BilateralSymmetryObservation(
        score=reference.score,
        support=support,
        best_axis_degrees=reference.axis_degrees,
        foreground_pixels=reference.foreground_pixels,
        bounding_width_pixels=reference.bounding_width_pixels,
        bounding_height_pixels=reference.bounding_height_pixels,
        bounding_diagonal_pixels=reference.bounding_diagonal_pixels,
        match_tolerance_pixels=reference.match_tolerance_pixels,
        threshold_scores=score_pairs,
        input_digest=input_digest,
        operation_digest=operation_digest(),
        provenance=origin,
    )
    return Evidence.present(observation, origin, support)


def bilateral_symmetry_score(panel: object) -> Evidence[float]:
    """Registered scalar projection preserving all four dispositions."""

    return measure_bilateral_symmetry(panel).map(lambda observation: observation.score)


def bilateral_symmetry_contract() -> LegContract:
    """Construct the exact positive-orientation contract for this operation."""

    return LegContract(
        name=LEG_NAME,
        version=LEG_VERSION,
        domain=(PANEL,),
        codomain=BILATERAL_SYMMETRY_SCORE,
        implementation=bilateral_symmetry_score,
        affirmative_relations=frozenset({AffirmativeRelation.AT_LEAST}),
        invariance=InvarianceContract(
            invariant_under=frozenset(
                {Transform.TRANSLATION, Transform.REFLECTION}
            ),
            sensitive_to=frozenset(
                {
                    Transform.RASTER_RESOLUTION,
                    Transform.STROKE_WIDTH,
                    Transform.STYLE,
                }
            ),
        ),
        semantics=LegSemantics.DETERMINISTIC_MEASUREMENT,
        cost=8,
        operational_digest=operation_digest(),
    )


def register_bilateral_symmetry_leg(registry: LegRegistry) -> LegReference:
    """Register the exact leg; callers retain ownership of registry freezing."""

    if not isinstance(registry, LegRegistry):
        raise TypeError("bilateral symmetry leg requires a LegRegistry")
    return registry.register(bilateral_symmetry_contract())


__all__ = [
    "ALGORITHM_ID",
    "BILATERAL_SYMMETRY_SCORE",
    "BilateralSymmetryInputError",
    "BilateralSymmetryObservation",
    "LEG_NAME",
    "LEG_VERSION",
    "bilateral_symmetry_contract",
    "bilateral_symmetry_score",
    "measure_bilateral_symmetry",
    "operation_digest",
    "register_bilateral_symmetry_leg",
]

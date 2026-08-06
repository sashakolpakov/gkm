from __future__ import annotations

import hashlib
from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from bongard.evidence import Disposition
from bongard.ir import (
    Atom,
    IRValidationError,
    Quantity,
    Relation,
    StaticLegCall,
    evaluate_formula,
    validate_formula,
)
from bongard.legs import (
    BILATERAL_SYMMETRY_SCORE,
    PANEL,
    AffirmativeRelation,
    LegRegistry,
    LegSemantics,
    Transform,
    TypedValue,
    Unit,
    bilateral_symmetry_contract,
    bilateral_symmetry_operation_digest,
    measure_bilateral_symmetry,
    register_bilateral_symmetry_leg,
)


def _matched_lobes() -> np.ndarray:
    panel = np.full((96, 96), 255, dtype=np.uint8)
    yy, xx = np.indices(panel.shape)
    ink = (xx - 28) ** 2 + (yy - 48) ** 2 <= 15**2
    ink |= (xx - 68) ** 2 + (yy - 48) ** 2 <= 15**2
    ink |= (np.abs(yy - 48) <= 2) & (xx >= 28) & (xx <= 68)
    panel[ink] = 0
    return panel


def _asymmetric_near_miss() -> np.ndarray:
    panel = np.full((96, 96), 255, dtype=np.uint8)
    yy, xx = np.indices(panel.shape)
    ink = (xx - 25) ** 2 + (yy - 55) ** 2 <= 16**2
    ink |= (xx - 68) ** 2 + (yy - 31) ** 2 <= 7**2
    ink |= (yy >= 45) & (yy <= 50) & (xx >= 25) & (xx <= 70)
    ink |= (yy >= 52) & (yy <= 78) & (xx >= 48) & (xx <= 82)
    panel[ink] = 0
    return panel


def _threshold_sensitive_panel() -> np.ndarray:
    panel = np.full((96, 96), 255, dtype=np.uint8)
    yy, xx = np.indices(panel.shape)
    ink = (xx - 28) ** 2 + (yy - 48) ** 2 <= 12**2
    ink |= (xx - 68) ** 2 + (yy - 48) ** 2 <= 12**2
    ink |= (np.abs(yy - 48) <= 2) & (xx >= 28) & (xx <= 68)
    panel[ink] = 0
    # This patch is deliberately excluded by the 96/128 thresholds and
    # included by 160.  The resulting interval records preprocessing
    # sensitivity instead of silently choosing the favorable threshold.
    panel[63:81, 52:85] = 150
    return panel


def _shift_without_clipping(
    panel: np.ndarray, *, down: int, right: int
) -> np.ndarray:
    shifted = np.full_like(panel, 255)
    shifted[down:, right:] = panel[: panel.shape[0] - down, : panel.shape[1] - right]
    return shifted


def _formula(reference: object, threshold: float = 0.9) -> Atom:
    return Atom(
        StaticLegCall(reference, ("panel",)),  # type: ignore[arg-type]
        Relation.AT_LEAST,
        "foreground geometry has high bilateral reflection agreement",
        Quantity(threshold, Unit.FRACTION),
    )


def _registry() -> tuple[LegRegistry, object]:
    registry = LegRegistry()
    reference = register_bilateral_symmetry_leg(registry)
    return registry.freeze(), reference


def test_fixed_measurement_separates_symmetric_shape_from_near_miss() -> None:
    matched = measure_bilateral_symmetry(_matched_lobes())
    near_miss = measure_bilateral_symmetry(_asymmetric_near_miss())

    assert matched.disposition is Disposition.PRESENT
    assert near_miss.disposition is Disposition.PRESENT
    assert matched.unwrap().score == 1.0
    assert near_miss.unwrap().score < 0.85
    assert matched.unwrap().score > near_miss.unwrap().score


def test_measurement_is_translation_and_reflection_invariant_as_declared() -> None:
    original = measure_bilateral_symmetry(_matched_lobes()).unwrap()
    shifted = measure_bilateral_symmetry(
        _shift_without_clipping(_matched_lobes(), down=5, right=7)
    ).unwrap()
    reflected = measure_bilateral_symmetry(np.fliplr(_matched_lobes())).unwrap()

    assert shifted.score == original.score
    assert shifted.support == original.support
    assert reflected.score == original.score
    assert reflected.support == original.support


def test_threshold_sensitivity_is_an_interval_not_a_favorable_point() -> None:
    evidence = measure_bilateral_symmetry(_threshold_sensitive_panel())
    observation = evidence.unwrap()

    assert observation.score == 1.0
    assert observation.threshold_scores[0] == (96, 1.0)
    assert observation.threshold_scores[1] == (128, 1.0)
    assert observation.threshold_scores[2][1] < 0.85
    assert observation.support.lower == observation.threshold_scores[2][1]
    assert observation.support.upper == 1.0
    assert evidence.uncertainty == observation.support


def test_four_dispositions_remain_distinct_at_the_pixel_boundary() -> None:
    present = measure_bilateral_symmetry(_matched_lobes())

    blank = np.full((64, 64), 255, dtype=np.uint8)
    absent = measure_bilateral_symmetry(blank)

    tiny = blank.copy()
    tiny[30:34, 30:34] = 0
    indeterminate = measure_bilateral_symmetry(tiny)

    malformed = measure_bilateral_symmetry(np.zeros((8, 8), dtype=np.float64))

    assert present.disposition is Disposition.PRESENT
    assert absent.disposition is Disposition.CERTIFIED_ABSENT
    assert "zero foreground" in (absent.certificate or "")
    assert indeterminate.disposition is Disposition.INDETERMINATE
    assert "insufficient_foreground" in (indeterminate.reason or "")
    assert indeterminate.uncertainty is not None
    assert malformed.disposition is Disposition.ERROR
    assert malformed.error_type == "BilateralSymmetryInputError"
    assert malformed.disposition is not Disposition.CERTIFIED_ABSENT


def test_border_clipping_is_indeterminate_not_false_symmetry() -> None:
    panel = np.full((64, 64), 255, dtype=np.uint8)
    panel[0:12, 20:44] = 0
    evidence = measure_bilateral_symmetry(panel)
    assert evidence.disposition is Disposition.INDETERMINATE
    assert "border_clipped" in (evidence.reason or "")


def test_observation_is_typed_content_addressed_and_not_boolean() -> None:
    panel = _matched_lobes()
    first = measure_bilateral_symmetry(panel).unwrap()
    second = measure_bilateral_symmetry(panel).unwrap()
    changed = panel.copy()
    changed[1, 1] = 254
    changed_observation = measure_bilateral_symmetry(changed).unwrap()

    assert first == second
    assert first.digest() == second.digest()
    assert first.input_digest != changed_observation.input_digest
    assert first.provenance.input_digests == (first.input_digest,)
    assert first.operation_digest == bilateral_symmetry_operation_digest()
    assert dict(first.provenance.details)["algorithm_id"].endswith("/v1")
    with pytest.raises(TypeError, match="measurement, not truth"):
        bool(first)


def test_png_path_and_bytes_bind_exact_container_bytes(tmp_path) -> None:
    path = tmp_path / "query.png"
    Image.fromarray(_matched_lobes(), mode="L").save(path, format="PNG")
    raw = path.read_bytes()

    from_path = measure_bilateral_symmetry(path).unwrap()
    from_bytes = measure_bilateral_symmetry(raw).unwrap()
    assert from_path.score == from_bytes.score
    assert from_path.support == from_bytes.support
    assert from_path.input_digest == hashlib.sha256(raw).hexdigest()
    assert from_bytes.input_digest == from_path.input_digest

    encoded = BytesIO()
    Image.fromarray(_matched_lobes(), mode="L").save(encoded, format="BMP")
    rejected = measure_bilateral_symmetry(encoded.getvalue())
    assert rejected.disposition is Disposition.ERROR
    assert "must be a PNG" in (rejected.reason or "")
    assert rejected.provenance.input_digests == (
        hashlib.sha256(encoded.getvalue()).hexdigest(),
    )

    other_rejected = measure_bilateral_symmetry(b"different malformed bytes")
    assert other_rejected.disposition is Disposition.ERROR
    assert other_rejected.provenance.input_digests != (
        rejected.provenance.input_digests
    )


def test_contract_is_fixed_positive_scalar_and_binds_full_operation() -> None:
    contract = bilateral_symmetry_contract()
    assert contract.domain == (PANEL,)
    assert contract.codomain == BILATERAL_SYMMETRY_SCORE
    assert contract.semantics is LegSemantics.DETERMINISTIC_MEASUREMENT
    assert contract.affirmative_relations == frozenset(
        {AffirmativeRelation.AT_LEAST}
    )
    assert contract.parameter_names == frozenset()
    assert contract.operational_digest == bilateral_symmetry_operation_digest()
    assert contract.invariance.invariant_under == frozenset(
        {Transform.TRANSLATION, Transform.REFLECTION}
    )
    assert Transform.RASTER_RESOLUTION in contract.invariance.sensitive_to


def test_closed_ir_uses_full_interval_and_cannot_flip_polarity() -> None:
    registry, reference = _registry()
    formula = _formula(reference)

    matched = evaluate_formula(
        formula,
        registry,
        {"panel": TypedValue(PANEL, _matched_lobes())},
    )
    near_miss = evaluate_formula(
        formula,
        registry,
        {"panel": TypedValue(PANEL, _asymmetric_near_miss())},
    )
    overlap = evaluate_formula(
        formula,
        registry,
        {"panel": TypedValue(PANEL, _threshold_sensitive_panel())},
    )

    assert matched.disposition is Disposition.PRESENT
    assert near_miss.disposition is Disposition.CERTIFIED_ABSENT
    assert overlap.disposition is Disposition.INDETERMINATE
    assert "straddles" in (overlap.reason or "")

    reversed_formula = Atom(
        StaticLegCall(reference, ("panel",)),  # type: ignore[arg-type]
        Relation.AT_MOST,
        "low symmetry",
        Quantity(0.9, Unit.FRACTION),
    )
    with pytest.raises(IRValidationError, match="not an affirmative orientation"):
        validate_formula(reversed_formula, registry, {"panel": PANEL})


def test_runtime_failures_and_blank_panels_never_become_a_present_claim() -> None:
    registry, reference = _registry()
    formula = _formula(reference)

    blank = evaluate_formula(
        formula,
        registry,
        {"panel": TypedValue(PANEL, np.full((64, 64), 255, dtype=np.uint8))},
    )
    malformed = evaluate_formula(
        formula,
        registry,
        {"panel": TypedValue(PANEL, object())},
    )

    assert blank.disposition is Disposition.CERTIFIED_ABSENT
    assert malformed.disposition is Disposition.ERROR
    assert malformed.disposition is not Disposition.CERTIFIED_ABSENT

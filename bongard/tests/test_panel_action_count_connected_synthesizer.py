from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from bongard import panel_action_count_connected_synthetic as connected
from bongard import panel_action_count_connected_synthesizer as subject


EXPECTED_SOURCE_SHA256 = (
    "9257fdfe9b3044fb48b2c40248063dc4b20e7b9c100a2fa700e7eaca8fe50b4d"
)


@lru_cache(maxsize=1)
def _corpus() -> tuple[object, ...]:
    rows = connected.build_connected_corpus()
    assert type(rows) is tuple and rows
    return rows


def _layout(sample: object) -> str:
    value = getattr(sample, "layout_truth", None)
    if value is None:
        value = getattr(getattr(sample, "panel_program", None), "layout", None)
    assert type(value) is str
    return value


def _boundary_kinds(sample: object) -> tuple[str, ...]:
    raw = getattr(sample, "boundary_truth", None)
    if type(raw) is str:
        return (raw,)
    if type(raw) is tuple:
        values = tuple(
            value
            if type(value) is str
            else getattr(value, "kind", None)
            or getattr(value, "boundary_kind", None)
            for value in raw
        )
    else:
        values = (
            getattr(raw, "kind", None)
            or getattr(raw, "boundary_kind", None),
        )
    assert all(type(value) is str for value in values)
    return tuple(values)


def _sample(*, boundary: str | None = None, layout: str | None = None) -> object:
    for sample in _corpus():
        if boundary is not None and boundary not in _boundary_kinds(sample):
            continue
        if layout is not None and _layout(sample) != layout:
            continue
        return sample
    raise AssertionError(f"connected fixture absent: boundary={boundary}, layout={layout}")


def _target_pairs(sample: object) -> tuple[tuple[int, int], ...]:
    target = connected.exact_cover_target(sample.png_bytes)
    return tuple(
        sorted(
            pair.as_tuple() if callable(getattr(pair, "as_tuple", None)) else tuple(pair)
            for pair in target.count_pairs
        )
    )


@pytest.mark.parametrize("boundary", ("LL", "LA", "AL", "AA"))
def test_raw_exact_cover_handles_every_ordered_boundary_kind(boundary: str) -> None:
    sample = _sample(boundary=boundary, layout="single_shape")

    outcome = subject.fit_png_hypotheses(sample.png_bytes)

    assert outcome.disposition in {"IDENTIFIED", "AMBIGUOUS"}
    assert outcome.reason is None
    assert outcome.exact_reconstruction
    assert outcome.candidate_pairs == _target_pairs(sample)
    assert outcome.minimum_primitive_count == min(sum(pair) for pair in outcome.candidate_pairs)
    assert outcome.paths
    assert outcome.boundary_pixels_yx
    assert all(
        hypothesis.xor_pixel_count == 0
        and hypothesis.intersection_over_union == 1.0
        and hypothesis.reconstructed_ink_pixels
        and hypothesis.primitive_ids
        and hypothesis.primitive_kinds
        for hypothesis in outcome.hypotheses
    )


def test_two_shape_union_is_solved_by_the_same_global_search() -> None:
    sample = _sample(layout="two_shape")

    outcome = subject.fit_png_hypotheses(sample.png_bytes)

    assert outcome.exact_reconstruction
    assert outcome.candidate_pairs == _target_pairs(sample)
    assert len({path.component_id for path in outcome.paths}) >= 2


def test_every_hypothesis_rerenders_the_exact_issued_png() -> None:
    samples = tuple(_sample(boundary=kind, layout="single_shape") for kind in ("LL", "LA", "AL", "AA"))
    samples += (_sample(layout="two_shape"),)
    for sample in samples:
        outcome = subject.fit_png_hypotheses(sample.png_bytes)
        for hypothesis in outcome.hypotheses:
            assert connected.render_catalog_program(hypothesis.primitive_ids) == sample.png_bytes


def test_raw_predictions_do_not_consult_target_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = _sample(boundary="LA", layout="single_shape")
    before = subject.fit_png_hypotheses(sample.png_bytes)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("raw fitting consulted the target oracle")

    monkeypatch.setattr(connected, "exact_cover_target", forbidden)
    after = subject.fit_png_hypotheses(sample.png_bytes)
    assert after == before


def test_catalog_record_mutation_after_cache_fill_cannot_change_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = _sample(boundary="AL", layout="single_shape")
    subject._catalog_masks.cache_clear()
    before = subject.fit_png_hypotheses(sample.png_bytes)
    monkeypatch.setattr(
        connected,
        "primitive_catalog",
        lambda: (_ for _ in ()).throw(AssertionError("catalog reread after seal")),
    )

    assert subject.fit_png_hypotheses(sample.png_bytes) == before


def test_materially_distinct_minimum_geometries_and_pair_set_are_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Four adjacent target pixels admit exactly two two-mask partitions:
    # {01, 23} as two lines and {02, 13} as two arcs.  Cross-partition pairs
    # omit one target pixel, so no mixed cover is exact.  This isolates the raw
    # search's uncertainty semantics without consulting a target oracle.
    target_pixels = (32 * 64 + 30, 32 * 64 + 31, 32 * 64 + 32, 32 * 64 + 33)
    rows = (
        ("toy-line-a", "line", (target_pixels[0], target_pixels[1])),
        ("toy-line-b", "line", (target_pixels[2], target_pixels[3])),
        ("toy-arc-a", "arc", (target_pixels[0], target_pixels[2])),
        ("toy-arc-b", "arc", (target_pixels[1], target_pixels[3])),
    )
    catalog = tuple(
        SimpleNamespace(
            primitive_id=primitive_id,
            kind=kind,
            ink_pixels=ink,
            endpoints_yx=tuple((pixel // 64, pixel % 64) for pixel in ink),
            boundary_pixels=ink,
        )
        for primitive_id, kind, ink in rows
    )
    image = np.full((64, 64), 255, dtype=np.uint8)
    image[32, 30:34] = 0
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", optimize=False)
    monkeypatch.setattr(connected, "require_issued_connected_png", lambda _raw: None)
    monkeypatch.setattr(connected, "primitive_catalog", lambda: catalog)
    subject._catalog_masks.cache_clear()
    try:
        outcome = subject.fit_png_hypotheses(buffer.getvalue())
    finally:
        subject._catalog_masks.cache_clear()

    assert outcome.disposition == "AMBIGUOUS"
    assert outcome.candidate_pairs == ((0, 2), (2, 0))
    assert len(outcome.hypotheses) == 2
    assert outcome.exact_reconstruction
    assert len({hypothesis.geometry_key for hypothesis in outcome.hypotheses}) == 2


def test_multi_action_single_shape_is_one_graph_component_with_multiple_primitives() -> None:
    sample = _sample(boundary="LL", layout="single_shape")

    outcome = subject.fit_png_hypotheses(sample.png_bytes)

    assert outcome.minimum_primitive_count is not None
    assert outcome.minimum_primitive_count >= 2
    assert {path.component_id for path in outcome.paths} == {0}
    assert all(len(hypothesis.primitives) >= 2 for hypothesis in outcome.hypotheses)


def test_unissued_and_nonexact_byte_transports_fail_closed() -> None:
    image = np.full((64, 64), 255, dtype=np.uint8)
    image[0, 0] = image[17, 43] = image[63, 63] = 0
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", optimize=False)
    with pytest.raises(subject.ConnectedSynthesisError, match="not issued"):
        subject.fit_png_hypotheses(buffer.getvalue())

    class BytesSubclass(bytes):
        pass

    issued = _sample(layout="single_shape").png_bytes
    with pytest.raises(subject.ConnectedSynthesisError, match="exact bytes"):
        subject.fit_png_hypotheses(BytesSubclass(issued))
    for malformed in (bytearray(issued), memoryview(issued), "not-png", None):
        with pytest.raises(subject.ConnectedSynthesisError, match="exact bytes"):
            subject.fit_png_hypotheses(malformed)  # type: ignore[arg-type]


def test_fit_api_has_no_target_parameter_and_source_has_no_target_call() -> None:
    import ast
    import inspect

    assert tuple(inspect.signature(subject.fit_png_hypotheses).parameters) == (
        "png_bytes",
    )
    tree = ast.parse(inspect.getsource(subject.fit_png_hypotheses))
    called_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "exact_cover_target" not in called_names
    assert "canonical_visible_pair" not in called_names
    assert subject.source_sha256() == EXPECTED_SOURCE_SHA256

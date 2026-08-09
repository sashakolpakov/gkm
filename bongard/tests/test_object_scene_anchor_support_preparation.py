from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO

from PIL import Image, ImageDraw
import pytest

from bongard.canonical import canonical_digest
from bongard.object_bongard_panel_rubric_calibration import (
    load_object_bongard_panel_rubric_calibration_source,
)
from bongard.object_scene_anchor_support_preparation import (
    ObjectSceneAnchorSupportCorpusFreeze,
    ObjectSceneAnchorSupportPanelFreeze,
    ObjectSceneAnchorSupportPanelInput,
    ObjectSceneAnchorSupportPanelRuntimeBundle,
    ObjectSceneAnchorSupportPreparationError,
    build_object_scene_anchor_support_panel,
    freeze_object_scene_anchor_support_corpus,
    verify_object_scene_anchor_support_panel_runtime,
)
from bongard.transport import MAX_PANEL_PNG_BYTES


def _scene(width: int = 96) -> bytes:
    image = Image.new("RGB", (width, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.line((8, 30, 20, 8, 32, 30, 8, 30), fill="black", width=3)
    draw.line((58, 42, 70, 18, 84, 42, 58, 42), fill="black", width=3)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _input(index: int, *, width: int = 96) -> ObjectSceneAnchorSupportPanelInput:
    payload = _scene(width)
    return ObjectSceneAnchorSupportPanelInput(
        panel_alias=f"panel_{index:03d}",
        support_bucket_index=0 if index < 6 else 1,
        source_digest="1" * 64,
        source_panel_binding_digest=hashlib.sha256(
            f"binding-{index}".encode("ascii")
        ).hexdigest(),
        source_ordinal=index,
        task_id=f"task-{index}",
        panel_id=f"support/task-{index}/0",
        original_panel_png_digest=hashlib.sha256(payload).hexdigest(),
        exact_original_png_bytes=payload,
    )


@pytest.fixture(scope="module")
def prepared_panel():
    panel_input = _input(0)
    return panel_input, build_object_scene_anchor_support_panel(panel_input)


def _has_bytes(value: object) -> bool:
    if isinstance(value, bytes):
        return True
    if isinstance(value, dict):
        return any(_has_bytes(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_bytes(item) for item in value)
    return False


def _reseal(value: dict[str, object], digest_key: str) -> None:
    value[digest_key] = canonical_digest(
        {key: item for key, item in value.items() if key != digest_key}
    )


def test_panel_freeze_is_complete_byte_free_and_transport_safe(
    prepared_panel,
) -> None:
    panel_input, runtime = prepared_panel
    freeze = runtime.freeze
    data = freeze.to_data()

    assert ObjectSceneAnchorSupportPanelFreeze.from_data(data) == freeze
    assert not _has_bytes(data)
    assert freeze.panel_alias == "panel_000"
    assert freeze.support_bucket_index == 0
    assert freeze.original_panel_png_digest == hashlib.sha256(
        panel_input.exact_original_png_bytes
    ).hexdigest()
    assert freeze.support_sheet_png_digest == hashlib.sha256(
        runtime.exact_support_sheet_png_bytes
    ).hexdigest()
    assert freeze.support_sheet_png_byte_count == len(
        runtime.exact_support_sheet_png_bytes
    )
    assert 0 < freeze.support_sheet_png_byte_count <= MAX_PANEL_PNG_BYTES
    assert freeze.proposal_count == len(freeze.object_ids) == 2
    assert freeze.object_ids == tuple(
        item.object_id for item in freeze.inventory.objects
    )


def test_panel_runtime_cold_replays_every_artifact_from_source_pixels(
    prepared_panel,
) -> None:
    panel_input, runtime = prepared_panel

    assert (
        verify_object_scene_anchor_support_panel_runtime(
            runtime,
            panel_input,
            expected_freeze_digest=runtime.freeze.freeze_digest,
        )
        == runtime
    )


def test_panel_runtime_rejects_byte_and_source_metadata_substitution(
    prepared_panel,
) -> None:
    panel_input, runtime = prepared_panel
    damaged_sheet = runtime.exact_support_sheet_png_bytes[:-1] + b"x"
    with pytest.raises(
        ObjectSceneAnchorSupportPreparationError,
        match="runtime support bytes",
    ):
        ObjectSceneAnchorSupportPanelRuntimeBundle(
            freeze=runtime.freeze,
            exact_original_png_bytes=runtime.exact_original_png_bytes,
            exact_support_sheet_png_bytes=damaged_sheet,
        )

    wrong_input = ObjectSceneAnchorSupportPanelInput(
        panel_alias=panel_input.panel_alias,
        support_bucket_index=1,
        source_digest=panel_input.source_digest,
        source_panel_binding_digest=panel_input.source_panel_binding_digest,
        source_ordinal=panel_input.source_ordinal,
        task_id=panel_input.task_id,
        panel_id=panel_input.panel_id,
        original_panel_png_digest=panel_input.original_panel_png_digest,
        exact_original_png_bytes=panel_input.exact_original_png_bytes,
    )
    with pytest.raises(
        ObjectSceneAnchorSupportPreparationError,
        match="source input differs",
    ):
        verify_object_scene_anchor_support_panel_runtime(runtime, wrong_input)


def test_panel_freeze_rejects_nested_artifact_omission_even_if_outer_resealed(
    prepared_panel,
) -> None:
    _, runtime = prepared_panel
    data = deepcopy(runtime.freeze.to_data())
    del data["inventory"]["objects"][0]
    _reseal(data["inventory"], "inventory_digest")
    _reseal(data, "freeze_digest")

    with pytest.raises(ValueError):
        ObjectSceneAnchorSupportPanelFreeze.from_data(data)


def test_source_panel_adapter_keeps_alias_neutral_and_bucket_opaque() -> None:
    source = load_object_bongard_panel_rubric_calibration_source()
    item = ObjectSceneAnchorSupportPanelInput.from_source_panel(
        source, source.panels[7], panel_alias="panel_007"
    )

    assert item.panel_alias == "panel_007"
    assert item.support_bucket_index == 1
    assert item.source_digest == source.source_digest
    assert item.source_panel_binding_digest == source.panels[7].panel_binding_digest
    assert item.exact_original_png_bytes == source.panels[7].exact_png_bytes


@pytest.fixture(scope="module")
def prepared_corpus(prepared_panel):
    _, first = prepared_panel
    panels = [first.freeze]
    # Varying the canvas width yields twelve distinct exact panels while
    # preserving the same small, clean two-object geometry for a focused test.
    for index in range(1, 12):
        panels.append(
            build_object_scene_anchor_support_panel(
                _input(index, width=96 + index)
            ).freeze
        )
    return freeze_object_scene_anchor_support_corpus("1" * 64, tuple(panels))


def test_corpus_freeze_is_exact_unique_six_by_six_and_roundtrips(
    prepared_corpus,
) -> None:
    freeze = prepared_corpus
    data = freeze.to_data()

    assert ObjectSceneAnchorSupportCorpusFreeze.from_data(data) == freeze
    assert freeze.panel_aliases == tuple(f"panel_{index:03d}" for index in range(12))
    assert tuple(item.support_bucket_index for item in freeze.panels) == (
        (0,) * 6 + (1,) * 6
    )
    assert freeze.bucket_0_count == freeze.bucket_1_count == 6
    assert len(set(freeze.original_panel_png_digests)) == 12
    assert freeze.complete_object_count == sum(
        item.proposal_count for item in freeze.panels
    )
    assert not _has_bytes(data)


def test_corpus_freeze_rejects_reuse_reordering_and_bad_complete_count(
    prepared_corpus,
) -> None:
    panels = prepared_corpus.panels
    with pytest.raises(ObjectSceneAnchorSupportPreparationError, match="exact unique"):
        freeze_object_scene_anchor_support_corpus(
            prepared_corpus.source_digest,
            (panels[0], panels[0], *panels[2:]),
        )
    with pytest.raises(ObjectSceneAnchorSupportPreparationError, match="exact unique"):
        freeze_object_scene_anchor_support_corpus(
            prepared_corpus.source_digest,
            (panels[1], panels[0], *panels[2:]),
        )

    data = deepcopy(prepared_corpus.to_data())
    data["complete_object_count"] += 1
    _reseal(data, "freeze_digest")
    with pytest.raises(ObjectSceneAnchorSupportPreparationError, match="exact unique"):
        ObjectSceneAnchorSupportCorpusFreeze.from_data(data)

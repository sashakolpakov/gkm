"""Focused tests for the one-call contextual typed-count support probe."""

from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import inspect
from pathlib import Path

from PIL import Image, ImageDraw

from bongard import panel_positive_contextual_typed_count_probe_command as command
from bongard.evidence import Disposition
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.object_bongard_turn_journal import ObjectBongardTurnRuntime
from bongard.panel_positive_contextual_typed_count_probe_command import (
    DEFAULT_CONTEXT_POLICY,
    DEFAULT_INK_ZOOM_POLICY,
    DEFAULT_TYPED_COUNT_POLICY,
    ROLE_VIEW_NAMES,
    _batch_prompt,
    _batch_schema,
    _load_context_policy,
    _project_payload,
    _support_result,
    run_contextual_typed_count_support_probe,
)
from bongard.panel_positive_prose_exposed_probe_command import (
    _deterministic_ink_zoom,
    _load_ink_zoom_policy,
    _load_typed_count_policy,
)
from bongard.prototype_scene_observer import prototype_scene_transport_source_digest
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _receipt,
)
from bongard.transport import CodexStructuredResult


def _panel(seed: int) -> bytes:
    image = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(image)
    inset = 90 + seed
    draw.polygon(
        [(inset, 420), (256, 70 + seed), (430 - seed, 420)],
        outline="black",
        width=8,
    )
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _payload(*, contrast_present: bool = False) -> dict[str, int]:
    result: dict[str, int] = {}
    for ordinal, name in enumerate(ROLE_VIEW_NAMES):
        stem = name.removesuffix(".png")
        result[f"{stem}_convex_score_lower"] = 3
        result[f"{stem}_convex_score_upper"] = 4
        if ordinal < 6 or (contrast_present and ordinal == 6):
            straight = (4, 4)
        else:
            straight = (2, 3)
        result[f"{stem}_straight_action_count_lower"] = straight[0]
        result[f"{stem}_straight_action_count_upper"] = straight[1]
        result[f"{stem}_curved_action_count_lower"] = 0
        result[f"{stem}_curved_action_count_upper"] = 2
    return result


def test_preregistered_prompt_schema_and_api_are_exactly_support_only() -> None:
    context, _ = _load_context_policy(DEFAULT_CONTEXT_POLICY)
    typed, _ = _load_typed_count_policy(DEFAULT_TYPED_COUNT_POLICY)
    schema = _batch_schema()
    prompt = _batch_prompt(typed)
    assert context["physical_call_plan"] == {
        "support_context_batch_observer": 1,
        "query": 0,
    }
    assert len(schema["properties"]) == 72
    assert schema["required"] == list(schema["properties"])
    assert "primary_00.png" in prompt and "contrast_05.png" in prompt
    assert "Contrast drawings may be heterogeneous" in prompt
    assert "Do not test equality to four" in prompt
    assert all(
        "query" not in name
        for name in inspect.signature(
            run_contextual_typed_count_support_probe
        ).parameters
    )


def test_python_projection_writes_pass_or_typed_gap() -> None:
    panels = tuple(_panel(index) for index in range(12))
    zoom_policy, _ = _load_ink_zoom_policy(DEFAULT_INK_ZOOM_POLICY)
    transformed = tuple(_deterministic_ink_zoom(panel, zoom_policy) for panel in panels)
    views = tuple(item[0] for item in transformed)
    records = tuple(item[1] for item in transformed)
    rows = _project_payload(
        _payload(), source_panels=panels, observer_views=views, zoom_records=records
    )
    passed, primary, contrast, reasons = _support_result(rows)
    assert passed is True and reasons == ()
    assert primary[Disposition.PRESENT.value] == 6
    assert contrast[Disposition.CERTIFIED_ABSENT.value] == 6

    gap_rows = _project_payload(
        _payload(contrast_present=True),
        source_panels=panels,
        observer_views=views,
        zoom_records=records,
    )
    passed, _primary, contrast, reasons = _support_result(gap_rows)
    assert passed is False
    assert contrast[Disposition.PRESENT.value] == 1
    assert "contrast_present_contradiction" in reasons


def test_end_to_end_uses_one_exactly_once_twelve_image_journal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = ObjectBongardTaskPlan.create(
        "hd_convex-has_four_straight_lines_0001",
        seed_digest="sha256:" + "34" * 32,
    )
    panel_ids = task.side_0_support_panel_ids + task.side_1_support_panel_ids
    panels = tuple(_panel(index) for index in range(12))
    monkeypatch.setattr(
        command,
        "_read_source",
        lambda path: (task, panel_ids, panels, "a" * 64),
    )
    runtime = ObjectBongardTurnRuntime(
        model=MODEL,
        reasoning_effort=EFFORT,
        minutes=20,
        verbose=False,
        executable="codex",
        cloud_policy_cache_snapshot=None,
        model_catalog_snapshot=NO_TOOLS_KWARGS["model_catalog_snapshot"],
        expected_launcher_digest=LAUNCHER_DIGEST,
        no_tools_attestation=NO_TOOLS_KWARGS["no_tools_attestation"],
        transport_source_digest=prototype_scene_transport_source_digest(),
    )
    runtime_evidence = command._record(
        {"schema": "fixture-runtime.v1", "runtime_binding": runtime.binding}
    )
    monkeypatch.setattr(
        command,
        "_runtime",
        lambda **kwargs: (runtime, runtime_evidence),
    )
    calls = 0
    payload = _payload()

    def physical(prompt, paths, names, schema, **kwargs):
        nonlocal calls
        calls += 1
        assert tuple(names) == ROLE_VIEW_NAMES
        assert len(paths) == 12
        return CodexStructuredResult(
            deepcopy(payload), _receipt(prompt, paths, names, schema, payload)
        )

    monkeypatch.setattr(command, "run_codex_named_images_structured", physical)
    kwargs = {
        "source_archive": tmp_path / "fixture-source.json",
        "output_root": tmp_path / "output",
        "model": MODEL,
        "reasoning_effort": EFFORT,
        "minutes": 20,
        "launcher_sha256": LAUNCHER_DIGEST,
    }
    first = run_contextual_typed_count_support_probe(**kwargs)
    second = run_contextual_typed_count_support_probe(**kwargs)
    assert calls == 1
    assert first == second
    assert first["status"] == "support_pass"
    assert first["physical_model_calls"] == 1
    assert first["journal_terminal"]["terminal_status"] == "success"
    assert len(first["rows"]) == 12
    assert (tmp_path / "output" / "completion.json").is_file()

"""Focused live-wrapper tests for the exposed support atom slate."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
from pathlib import Path

from bongard import panel_positive_atom_slate as atom_module
from bongard import panel_positive_atom_slate_exposed_probe_command as command
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.object_bongard_turn_journal import ObjectBongardTurnRuntime
from bongard.panel_positive_atom_slate import ATOM_IDS, PROPOSER_IMAGE_NAMES
from bongard.panel_positive_atom_slate_exposed_probe_command import (
    _authorization_and_precommit,
    run_atom_slate_exposed_support_probe,
)
from bongard.prototype_scene_observer import prototype_scene_transport_source_digest
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _png,
    _receipt,
)
from bongard.transport import CodexStructuredResult


def _slate_payload() -> dict[str, str]:
    atoms = (
        "convex carrier",
        "four straight structural runs",
        "single closed contour",
        "bilateral symmetry",
        "oblique corners",
        "central point contact",
        "curved outer boundary",
        "nested figure",
    )
    return dict(zip(ATOM_IDS, atoms, strict=True))


def _row_payload(ordinal: int) -> dict[str, int]:
    result: dict[str, int] = {}
    for atom_index, atom_id in enumerate(ATOM_IDS):
        if atom_index >= 2:
            bounds = (1, 3)
        elif ordinal < 6:
            bounds = (3, 4)
        elif atom_index == 0:
            bounds = (3, 4) if ordinal < 9 else (0, 1)
        else:
            bounds = (0, 1) if ordinal < 9 else (3, 4)
        result[f"{atom_id}_lower"] = bounds[0]
        result[f"{atom_id}_upper"] = bounds[1]
    return result


def test_precommit_is_fixed_support_only_and_has_no_query_api() -> None:
    task = ObjectBongardTaskPlan.create(
        "hd_convex-has_four_straight_lines_0001",
        seed_digest="sha256:" + "34" * 32,
    )
    panel_ids = task.side_0_support_panel_ids + task.side_1_support_panel_ids
    panels = tuple(_png(410 + index) for index in range(12))
    authorization, precommit = _authorization_and_precommit(
        task=task,
        panel_ids=panel_ids,
        panels=panels,
        source_archive_sha256="a" * 64,
    )
    assert authorization["query_pixels_available_to_command"] is False
    assert precommit["physical_call_plan"] == {
        "support_atom_proposer": 1,
        "support_all_atom_panel_observers": 12,
        "query": 0,
    }
    assert precommit["formula_count"] == 36
    assert precommit["model_formula_threshold_or_polarity_selection_allowed"] is False
    assert all(
        "query" not in name
        for name in inspect.signature(run_atom_slate_exposed_support_probe).parameters
    )


def test_end_to_end_is_exactly_once_and_finds_heterogeneous_pair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = ObjectBongardTaskPlan.create(
        "hd_convex-has_four_straight_lines_0001",
        seed_digest="sha256:" + "34" * 32,
    )
    panel_ids = task.side_0_support_panel_ids + task.side_1_support_panel_ids
    panels = tuple(_png(430 + index) for index in range(12))
    ordinal_by_digest = {
        hashlib.sha256(panel).hexdigest(): ordinal
        for ordinal, panel in enumerate(panels)
    }
    monkeypatch.setattr(
        command,
        "_read_source",
        lambda path: (task, panel_ids, panels, "b" * 64),
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
    calls: list[tuple[str, ...]] = []
    slate_payload = _slate_payload()

    def physical(prompt, paths, names, schema, **kwargs):
        frozen_names = tuple(names)
        calls.append(frozen_names)
        if frozen_names == PROPOSER_IMAGE_NAMES:
            payload = slate_payload
        else:
            assert frozen_names == ("panel.png",)
            panel = Path(paths[0]).read_bytes()
            payload = _row_payload(
                ordinal_by_digest[hashlib.sha256(panel).hexdigest()]
            )
        return CodexStructuredResult(
            deepcopy(payload), _receipt(prompt, paths, names, schema, payload)
        )

    monkeypatch.setattr(
        atom_module, "run_codex_named_images_structured", physical
    )
    kwargs = {
        "source_archive": tmp_path / "source.json",
        "output_root": tmp_path / "output",
        "model": MODEL,
        "reasoning_effort": EFFORT,
        "minutes": 20,
        "launcher_sha256": LAUNCHER_DIGEST,
        "workers": 4,
    }
    first = run_atom_slate_exposed_support_probe(**kwargs)
    second = run_atom_slate_exposed_support_probe(**kwargs)
    assert first == second
    assert len(calls) == 13
    assert calls.count(PROPOSER_IMAGE_NAMES) == 1
    assert calls.count(("panel.png",)) == 12
    assert first["status"] == "support_pass"
    assert first["admitted_formula_count"] == 1
    admitted = first["support_inventory"]["admitted_formulas"]
    assert admitted[0]["atom_ids"] == ["atom_00", "atom_01"]
    assert first["all_artifacts_benchmark_sealable"] is True
    assert first["query_pixels_available_to_command"] is False
    assert len(first["panel_journal_terminals"]) == 12
    assert (tmp_path / "output" / "completion.json").is_file()

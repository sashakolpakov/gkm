"""Focused offline launch/verify test for the panel calibration command."""

from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Lock

from bongard.object_bongard_panel_rubric_calibration_command import (
    AUTHORIZATION_FILENAME,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    PRECOMMIT_FILENAME,
    _load_plan_inputs,
    _run_loaded_calibration,
    _verify_loaded_calibration,
)
from bongard.object_bongard_panel_rubric_observer import (
    object_bongard_panel_rubric_prompt,
)
from bongard.tests.no_tools_fixture import canonical_no_tools_runtime
from bongard.tests.test_object_bongard_panel_rubric_calibration import (
    NOMINATION_ROOT,
    SOURCE_ROOT,
)
from bongard.tests.test_prototype_scene_observer import _receipt
from bongard.transport import (
    PINNED_CODEX_CLI_VERSION,
    CloudPolicyCacheSnapshot,
    CodexStructuredResult,
)


LAUNCHER_DIGEST = "b" * 64


def test_fresh_24_call_launch_and_model_free_verify(tmp_path: Path) -> None:
    plan, nomination = _load_plan_inputs(
        nomination_root=NOMINATION_ROOT,
        source_directory=SOURCE_ROOT,
    )
    cache = CloudPolicyCacheSnapshot(None)
    catalog, attestation = canonical_no_tools_runtime(LAUNCHER_DIGEST)
    preflight: list[str] = []

    def cache_snapshotter():
        preflight.append("cache")
        return cache

    def catalog_snapshotter():
        preflight.append("catalog")
        return catalog

    def fingerprinter(executable, *, expected_launcher_digest):
        assert executable == "codex"
        assert expected_launcher_digest == LAUNCHER_DIGEST
        preflight.append("fingerprint")
        return {
            "version": PINNED_CODEX_CLI_VERSION,
            "launcher_digest": LAUNCHER_DIGEST,
        }

    def runtime_attester(**kwargs):
        assert kwargs["cloud_policy_cache_snapshot"] is cache
        assert kwargs["model_catalog_snapshot"] is catalog
        preflight.append("attestation")
        return attestation

    by_png = {item.png_sha256: item for item in plan.source.panels}
    calls: list[tuple[int, int]] = []
    lock = Lock()
    root = tmp_path / "calibration"

    def transport(prompt, paths, names, schema, **_kwargs):
        assert (root / AUTHORIZATION_FILENAME).is_file()
        assert (root / PRECOMMIT_FILENAME).is_file()
        assert names == ("panel.png",)
        panel_digest = hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest()
        panel = by_png[panel_digest]
        rank = next(
            spec.candidate_rank
            for spec in plan.rubric_specs
            if object_bongard_panel_rubric_prompt(spec) == prompt
        )
        with lock:
            calls.append((rank, panel.ordinal))
        level = 4 if panel.group_index == 0 else 0
        payload = {"lower": level, "upper": level}
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    launched = _run_loaded_calibration(
        root,
        plan,
        nomination,
        parallel_workers=4,
        expected_launcher_sha256=LAUNCHER_DIGEST,
        transport=transport,
        cloud_policy_cache_snapshotter=cache_snapshotter,
        model_catalog_snapshotter=catalog_snapshotter,
        launcher_fingerprinter=fingerprinter,
        runtime_attester=runtime_attester,
    )
    assert preflight == ["cache", "catalog", "fingerprint", "attestation"]
    assert len(calls) == 24
    assert set(calls) == {
        (rank, panel.ordinal)
        for rank in (0, 1)
        for panel in plan.source.panels
    }
    assert launched.accepted is True
    assert launched.selected_candidate_rank == 0
    assert launched.fresh_call_count == 24
    assert launched.reused_call_count == 0
    assert _verify_loaded_calibration(root, plan, nomination) == launched
    assert len(calls) == 24
    assert DEFAULT_MODEL == "gpt-5.6-sol"
    assert DEFAULT_REASONING_EFFORT == "medium"
    serialized = (root / "result.json").read_text("utf-8")
    assert '"lean_present":false' in serialized
    assert '"lean_removable":true' in serialized

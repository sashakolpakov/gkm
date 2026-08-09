"""Offline boundary tests for the exposed-support smoke command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import bongard.panel_feature_exposed_support_smoke_command as smoke
from bongard.panel_feature_exposed_support_smoke_command import (
    DEFAULT_SOURCE_ARCHIVE,
    PanelFeatureExposedSupportSmokeError,
    _authorization,
    _metadata_only,
    _read_source,
    _runtime,
)
from bongard.tests.no_tools_fixture import canonical_no_tools_runtime
from bongard.transport import CloudPolicyCacheSnapshot


_LAUNCHER_SHA256 = "1" * 64


def test_real_source_is_exactly_twelve_supports_and_zero_queries() -> None:
    result = _metadata_only(DEFAULT_SOURCE_ARCHIVE)
    assert result["task_id"] == "hd_convex-has_four_straight_lines_0001"
    assert result["support_panel_count"] == 12
    assert result["query_pixel_count"] == 0
    assert result["observer_axis_count"] == 9
    assert "straight_segment_count" in result["observer_axis_families"]
    assert "convexity" in result["observer_axis_families"]


def test_source_with_any_query_or_freeze_material_fails_closed(tmp_path) -> None:
    raw = json.loads(DEFAULT_SOURCE_ARCHIVE.read_text(encoding="utf-8"))
    for field, value in (
        ("query_png_base64_by_side", {"side_0": "AA=="}),
        ("query_source_calls_made", 1),
        ("freeze", {"forged": True}),
        ("rank_artifact", {"forged": True}),
    ):
        changed = dict(raw)
        changed[field] = value
        path = tmp_path / f"{field}.json"
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(PanelFeatureExposedSupportSmokeError):
            _read_source(path)


def test_precommit_authorizes_no_query_or_freeze() -> None:
    task, panel_ids, panels, source_digest = _read_source(DEFAULT_SOURCE_ARCHIVE)
    authorization, precommit = _authorization(
        task, panel_ids, panels, source_digest
    )
    assert authorization["query_release_or_observation_authorized"] is False
    assert precommit["physical_call_plan"]["query"] == 0
    assert precommit["query_pixels_available_to_command"] is False
    assert precommit["frozen_predicate_created"] is False


def test_runtime_replay_uses_exact_stored_preimages_without_resnapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog, attestation = canonical_no_tools_runtime(_LAUNCHER_SHA256)
    fingerprint = {
        "version": "codex-cli replay-fixture",
        "launcher_digest": _LAUNCHER_SHA256,
    }
    monkeypatch.setattr(
        smoke, "snapshot_cloud_policy_cache", lambda: CloudPolicyCacheSnapshot(None)
    )
    monkeypatch.setattr(smoke, "snapshot_pinned_model_catalog", lambda: catalog)
    monkeypatch.setattr(smoke, "attest_codex_no_tools", lambda **_kwargs: attestation)
    monkeypatch.setattr(
        smoke,
        "codex_cli_authenticated_fingerprint",
        lambda _executable, *, expected_launcher_digest: {
            **fingerprint,
            "launcher_digest": expected_launcher_digest,
        },
    )
    kwargs = {
        "output_root": tmp_path,
        "authorization": {"record_digest": "sha256:" + "2" * 64},
        "precommit": {"record_digest": "sha256:" + "3" * 64},
        "model": "gpt-5.6-sol",
        "reasoning_effort": "medium",
        "minutes": 15,
        "executable": "codex-fixture",
        "launcher_sha256": _LAUNCHER_SHA256,
        "verbose": False,
    }
    first_runtime, first_evidence = _runtime(**kwargs)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("runtime replay tried to regenerate a frozen preimage")

    monkeypatch.setattr(smoke, "snapshot_cloud_policy_cache", forbidden)
    monkeypatch.setattr(smoke, "snapshot_pinned_model_catalog", forbidden)
    monkeypatch.setattr(smoke, "attest_codex_no_tools", forbidden)
    replay_runtime, replay_evidence = _runtime(**kwargs)

    assert replay_runtime.binding == first_runtime.binding
    assert replay_evidence == first_evidence

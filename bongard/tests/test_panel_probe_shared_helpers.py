from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import bongard.panel_probe_custody as custody
import bongard.panel_probe_transport as probe_transport


def test_probe_record_is_bounded_write_once_and_cold_authenticated(
    tmp_path: Path,
) -> None:
    record = custody.make_probe_record({"schema": "fixture.v1", "value": [2, 1]})
    target = tmp_path / "nested" / "record.json"
    custody.write_once_or_verify_probe_record(target, record)
    custody.write_once_or_verify_probe_record(target, record)
    assert custody.read_probe_record(target) == record
    assert json.loads(target.read_bytes()) == record

    changed = {**record, "value": [1, 2]}
    with pytest.raises(custody.PanelProbeCustodyError, match="differs"):
        custody.write_once_or_verify_probe_record(target, changed)
    target.write_bytes(target.read_bytes().replace(b"fixture", b"tampered"))
    with pytest.raises(custody.PanelProbeCustodyError):
        custody.read_probe_record(target)


def test_probe_record_rejects_digest_field_and_symlink(tmp_path: Path) -> None:
    with pytest.raises(custody.PanelProbeCustodyError, match="already contains"):
        custody.make_probe_record({"record_digest": "forged"})
    source = tmp_path / "source.json"
    source.write_bytes(b"{}\n")
    link = tmp_path / "link.json"
    link.symlink_to(source)
    with pytest.raises(custody.PanelProbeCustodyError, match="symlink"):
        custody.read_probe_record(link)


def test_probe_transport_delegates_exact_runtime_and_canonicalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = object()
    captured: dict[str, object] = {}

    def stage(images, **kwargs):
        captured["images"] = images
        captured.update(kwargs)
        return {"z": 2, "a": 1}, receipt

    monkeypatch.setattr(probe_transport._scene_runtime, "_stage_and_call", stage)
    runtime = SimpleNamespace(
        model="model",
        reasoning_effort="medium",
        minutes=3,
        verbose=False,
        executable="codex",
        cloud_policy_cache_snapshot="cache",
        expected_launcher_digest="launcher",
        model_catalog_snapshot="catalog",
        no_tools_attestation="attestation",
    )
    images = (("panel.png", b"png"),)
    payload, returned_receipt = probe_transport.call_panel_probe(
        images,
        prompt="prompt",
        schema={"type": "object"},
        journal="journal",
        runtime=runtime,
    )
    assert payload == {"a": 1, "z": 2}
    assert returned_receipt is receipt
    assert captured == {
        "images": images,
        "prompt": "prompt",
        "schema": {"type": "object"},
        "model": "model",
        "reasoning_effort": "medium",
        "minutes": 3,
        "verbose": False,
        "executable": "codex",
        "cloud_policy_cache_snapshot": "cache",
        "expected_launcher_digest": "launcher",
        "model_catalog_snapshot": "catalog",
        "no_tools_attestation": "attestation",
        "transport": "journal",
    }


def test_shared_helper_sources_are_content_addressed() -> None:
    assert len(custody.panel_probe_custody_source_digest()) == 64
    assert len(probe_transport.panel_probe_transport_source_digest()) == 64


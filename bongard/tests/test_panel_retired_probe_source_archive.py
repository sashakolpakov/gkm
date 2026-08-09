from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from bongard.canonical import canonical_digest, canonical_json
from bongard.panel_retired_probe_source_archive import (
    DEFAULT_SOURCE_SNAPSHOT_RECORD_DIGEST,
    RetiredProbeSourceArchiveError,
    load_retired_probe_source_archive,
    verify_retired_source_binding,
    verify_retired_source_bound_record,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    PACKAGE_ROOT
    / "data/panel_positive_live_support_diagnostic_summary_20260809_v1.json"
)
EXPECTED_SOURCE_DIGESTS = {
    "bongard.panel_feature_exposed_support_smoke_command": {
        "0d9aff32c6407e0f33be85d05b49af1d2e0fae0eed21af2febaa2c3f2ed94d13"
    },
    "bongard.panel_positive_prose_exposed_probe_command": {
        "0cfba4885942389055e7d17a794dbe51008ba7f989de9a052debfa6c6f25919c",
        "14dcf8b75b777a5be298cdfdd2208930096954dcbbf5ace6dcc0cdd24a8faf2e",
        "28a6d6389a007868fcac82e5c84e776f83f30a2c1035c9ad096966b6db0835a7",
        "49abd1712cf98b7a7b8d211b3947932c7bb6f5c4e49d0e50471c366e2853feb9",
        "4c0486477596fc8ad7a1c5c645051b0c2c39135df0cb595ef85eecffb15523d0",
        "52ed69cc63834f7dbe27e55bfa2e296e098ec512f5ae0a459b2d64b13acd2268",
        "7f466c47309822d44d30c45f1b5d2eefa16f1eb4618da46230c35c8c2818ff27",
        "b58904092744431d12d4a16c3ac517cad613057f6390a116ff17e05f4e15d712",
    },
    "bongard.panel_positive_contextual_typed_count_probe_command": {
        "07da149634fb336747154725e4731b3b9953081bcaa7e1b6d117eec8f514dd94"
    },
    "bongard.panel_positive_atom_slate_exposed_probe_command": {
        "42ad8195809ebb226a18d9133c9dcb80a585594086652064e16eda287813f9c9"
    },
}


def test_source_snapshot_is_pinned_and_contains_every_bound_generation() -> None:
    archive = load_retired_probe_source_archive()
    assert archive.record_digest == DEFAULT_SOURCE_SNAPSHOT_RECORD_DIGEST
    actual: dict[str, set[str]] = {}
    for entry in archive.entries.values():
        actual.setdefault(entry["module"], set()).add(entry["source_sha256"])
        source = archive.sources[entry["snapshot_id"]]
        assert hashlib.sha256(source).hexdigest() == entry["source_sha256"]
    assert actual == EXPECTED_SOURCE_DIGESTS


def test_decoder_has_no_compile_or_execution_surface() -> None:
    source_path = PACKAGE_ROOT / "panel_retired_probe_source_archive.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert called_names.isdisjoint({"compile", "eval", "exec"})


def test_compact_diagnostic_source_bindings_resolve_to_exact_preimages() -> None:
    summary_raw = SUMMARY.read_bytes()
    summary = json.loads(summary_raw)
    assert summary_raw == canonical_json(summary) + b"\n"
    body = dict(summary)
    digest = body.pop("record_digest")
    assert digest == "sha256:" + canonical_digest(body)
    bindings = (
        (
            "bongard.panel_positive_contextual_typed_count_probe_command",
            summary["runs"]["contextual_typed_count"]["digests"]["command_source"],
        ),
        (
            "bongard.panel_positive_atom_slate_exposed_probe_command",
            summary["runs"]["positive_atom_slate"]["digests"]["command_source"],
        ),
    )
    archive = load_retired_probe_source_archive()
    for module, source_sha256 in bindings:
        source = verify_retired_source_binding(
            module, source_sha256, archive=archive
        )
        assert hashlib.sha256(source).hexdigest() == source_sha256
    assert all(
        run["calls"]["query_observer_calls"] == 0
        and run["calls"]["query_release_calls"] == 0
        for run in summary["runs"].values()
    )


def test_source_bound_record_verifier_rejects_tampering(tmp_path: Path) -> None:
    archive = load_retired_probe_source_archive()
    module = "bongard.panel_positive_atom_slate_exposed_probe_command"
    source_sha256 = next(iter(EXPECTED_SOURCE_DIGESTS[module]))
    body = {
        "schema": "test.retired-source-binding.v1",
        "command_source_digest": source_sha256,
    }
    record = {**body, "record_digest": "sha256:" + canonical_digest(body)}
    path = tmp_path / "authorization.json"
    path.write_bytes(canonical_json(record) + b"\n")
    verified = verify_retired_source_bound_record(path, module, archive=archive)
    assert verified["command_source_digest"] == source_sha256

    record["command_source_digest"] = "0" * 64
    path.write_bytes(canonical_json(record) + b"\n")
    with pytest.raises(RetiredProbeSourceArchiveError, match="record digest"):
        verify_retired_source_bound_record(path, module, archive=archive)

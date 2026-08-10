from __future__ import annotations

import ast
import base64
from dataclasses import FrozenInstanceError
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from bongard.canonical import canonical_digest, canonical_json
from bongard import panel_retired_pipeline_archive as subject
from bongard.panel_retired_pipeline_archive import (
    DEFAULT_SOURCE_SNAPSHOT,
    DEFAULT_SOURCE_SNAPSHOT_RECORD_DIGEST,
    RETIREMENT_RECEIPT_CLAIM,
    RetiredPipelineArchiveError,
    RetiredPipelineArtifactBinding,
    RetiredPipelineRetirementReceipt,
    load_retired_pipeline_source_archive,
    load_retirement_receipt,
    verify_retired_pipeline_source_binding,
)


ROOT = Path(__file__).resolve().parents[2]
SUBJECT = ROOT / "bongard/panel_retired_pipeline_archive.py"
CURRENT_COMMIT = "a35cf269e418241da8db4fef6fb72ede20e5780f"
HISTORICAL_PHASE_COMMIT = "d358be3a71beeccaa31c010e83e0a3229a4e80de"
HISTORICAL_PHASE_BLOB = "6720684a8cd1d04ef0a00993d54e30a6b722d760"
HISTORICAL_PHASE_SHA256 = (
    "673b4811886a611c21bace1e90366acc1b2b7abfd43d53811903e973e2c85f7b"
)
CURRENT_PHASE_SHA256 = (
    "cde224ae978ee8f22b952e43624a9083e4f5c16cbc703c27e7a32d49c35f0c19"
)
SOFT_MANIFEST_BINDING = (
    "downloads/ShapeBongard_V2_full/"
    "panel_soft_exact_unused_train_20260809_ranked_v1/objects/"
    "panel-soft-source-manifest/"
    "fd14be21b945788aa34cb8808823039cbb24e170d831fa48f6a626c6d9dffa11.json"
)


def _git_blob_oid(source: bytes) -> str:
    header = b"blob " + str(len(source)).encode("ascii") + b"\0"
    return hashlib.sha1(header + source).hexdigest()  # noqa: S324


def _git_source(commit: str, relative_path: str) -> bytes:
    return subprocess.check_output(
        ("git", "cat-file", "blob", f"{commit}:{relative_path}"), cwd=ROOT
    )


def _git_source_oid(commit: str, relative_path: str) -> str:
    return subprocess.check_output(
        ("git", "rev-parse", f"{commit}:{relative_path}"),
        cwd=ROOT,
        text=True,
    ).strip()


def _snapshot_value() -> dict[str, object]:
    return json.loads(DEFAULT_SOURCE_SNAPSHOT.read_bytes())


def _reseal_snapshot(value: dict[str, object]) -> str:
    body = dict(value)
    body.pop("record_digest", None)
    digest = "sha256:" + canonical_digest(body)
    value["record_digest"] = digest
    return digest


def _write_canonical(path: Path, value: object) -> None:
    path.write_bytes(canonical_json(value) + b"\n")


def _receipt() -> RetiredPipelineRetirementReceipt:
    archive = load_retired_pipeline_source_archive()
    return RetiredPipelineRetirementReceipt.create(
        active_successor_pipeline_id=(
            "typed-geometry-calibrated-soft-positive-version-space-python-v1"
        ),
        source_snapshot_record_digest=archive.record_digest,
        source_snapshot_file_sha256=archive.snapshot_file_sha256,
        source_snapshot_entry_count=len(archive.entries),
        legacy_modules=(
            "bongard.panel_action_count_phase_command",
            "bongard.panel_action_count_tiny_local_train_command",
        ),
        artifact_bindings=(
            RetiredPipelineArtifactBinding(
                relative_path="archive/legacy-result.json",
                raw_sha256="sha256:" + "1" * 64,
                byte_count=123,
                canonical_record_digest="sha256:" + "2" * 64,
                disposition="remove_after_successor_acceptance",
            ),
            RetiredPipelineArtifactBinding(
                relative_path="archive/exposure.json",
                raw_sha256="sha256:" + "3" * 64,
                byte_count=456,
                canonical_record_digest=None,
                disposition="retain_irreducible",
            ),
        ),
    )


def test_default_snapshot_authenticates_all_exact_inert_sources() -> None:
    archive = load_retired_pipeline_source_archive()
    assert archive.record_digest == DEFAULT_SOURCE_SNAPSHOT_RECORD_DIGEST
    assert archive.snapshot_file_sha256 == (
        "sha256:7e146f797b36cea633103ec40385233ac8df774197535511b4f1467cad17bef5"
    )
    assert len(archive.entries) == 48
    assert tuple(archive.entries) == tuple(sorted(archive.entries))
    assert tuple(archive.sources) == tuple(sorted(archive.sources))
    assert set(archive.entries) == set(archive.sources)

    for snapshot_id, entry in archive.entries.items():
        source = archive.sources[snapshot_id]
        assert len(source) == entry["source_byte_count"]
        assert hashlib.sha256(source).hexdigest() == entry["source_sha256"]
        assert _git_blob_oid(source) == entry["git_blob_oid"]
        assert entry["source_commit"] in {CURRENT_COMMIT, HISTORICAL_PHASE_COMMIT}
        assert source == _git_source(
            entry["source_commit"], entry["relative_path"]
        )
        assert entry["git_blob_oid"] == _git_source_oid(
            entry["source_commit"], entry["relative_path"]
        )


def test_snapshot_survives_all_live_legacy_source_paths_being_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolated_root = tmp_path / "post-retirement"
    isolated_root.mkdir()
    copied_snapshot = isolated_root / "source-snapshot.json"
    copied_snapshot.write_bytes(DEFAULT_SOURCE_SNAPSHOT.read_bytes())
    for entry in _snapshot_value()["entries"]:
        assert not (isolated_root / entry["relative_path"]).exists()

    monkeypatch.chdir(isolated_root)
    archive = load_retired_pipeline_source_archive(copied_snapshot)
    assert len(archive.entries) == 48
    assert set(archive.entries) == set(archive.sources)


def test_loaded_archive_is_deeply_immutable() -> None:
    archive = load_retired_pipeline_source_archive()
    snapshot_id = next(iter(archive.entries))
    entry = archive.entries[snapshot_id]
    bindings = entry["artifact_bindings"]
    source = archive.sources[snapshot_id]
    assert type(bindings) is tuple
    with pytest.raises(TypeError):
        archive.entries[snapshot_id] = entry  # type: ignore[index]
    with pytest.raises(TypeError):
        archive.sources[snapshot_id] = source  # type: ignore[index]
    with pytest.raises(TypeError):
        entry["module"] = "bongard.tampered"  # type: ignore[index]
    with pytest.raises(TypeError):
        bindings[0] = "tampered/in-memory-binding"  # type: ignore[index]
    assert archive.entries[snapshot_id]["artifact_bindings"] == bindings


def test_all_31_panel_soft_manifest_preimages_are_present() -> None:
    archive = load_retired_pipeline_source_archive()
    entries = tuple(
        entry
        for entry in archive.entries.values()
        if SOFT_MANIFEST_BINDING in entry["artifact_bindings"]
    )
    assert len(entries) == 31
    assert sum(
        entry["module"].startswith("bongard.panel_soft_") for entry in entries
    ) == 6
    for entry in entries:
        assert entry["provenance_kind"] == "working_tree_and_git_blob"
        assert entry["source_commit"] == CURRENT_COMMIT


def test_historical_phase_blob_round_trips_exact_git_preimage() -> None:
    archive = load_retired_pipeline_source_archive()
    source = verify_retired_pipeline_source_binding(
        "bongard.panel_action_count_phase_command",
        HISTORICAL_PHASE_SHA256,
        archive=archive,
    )
    expected = subprocess.check_output(
        ("git", "cat-file", "blob", HISTORICAL_PHASE_BLOB), cwd=ROOT
    )
    assert source == expected
    assert len(source) == 35_719
    entry = archive.entries[
        "bongard.panel_action_count_phase_command@sha256:"
        + HISTORICAL_PHASE_SHA256
    ]
    assert entry["provenance_kind"] == "git_blob"
    assert entry["source_commit"] == HISTORICAL_PHASE_COMMIT
    assert entry["git_blob_oid"] == HISTORICAL_PHASE_BLOB

    current = archive.source_for(
        "bongard.panel_action_count_phase_command", CURRENT_PHASE_SHA256
    )
    assert current == _git_source(
        CURRENT_COMMIT, "bongard/panel_action_count_phase_command.py"
    )
    assert current != source


def test_unknown_or_malformed_source_identity_fails_closed() -> None:
    archive = load_retired_pipeline_source_archive()
    with pytest.raises(RetiredPipelineArchiveError, match="absent"):
        archive.source_for(
            "bongard.panel_action_count_phase_command", "f" * 64
        )
    with pytest.raises(RetiredPipelineArchiveError, match="module"):
        archive.source_for("../phase", HISTORICAL_PHASE_SHA256)
    with pytest.raises(RetiredPipelineArchiveError, match="raw SHA-256"):
        archive.source_for(
            "bongard.panel_action_count_phase_command", "sha256:" + "f" * 64
        )


def test_outer_snapshot_digest_tampering_fails_closed(tmp_path: Path) -> None:
    value = _snapshot_value()
    value["entries"][0]["source_byte_count"] += 1
    path = tmp_path / "snapshot.json"
    _write_canonical(path, value)
    with pytest.raises(RetiredPipelineArchiveError, match="digest"):
        load_retired_pipeline_source_archive(path)


def test_resealed_entry_tampering_reaches_preimage_check(tmp_path: Path) -> None:
    value = _snapshot_value()
    value["entries"][0]["source_byte_count"] += 1
    digest = _reseal_snapshot(value)
    path = tmp_path / "snapshot.json"
    _write_canonical(path, value)
    with pytest.raises(RetiredPipelineArchiveError, match="preimage"):
        load_retired_pipeline_source_archive(path, expected_record_digest=digest)


def test_concatenated_gzip_member_is_rejected(tmp_path: Path) -> None:
    value = _snapshot_value()
    metadata = value["archive"]
    compressed = base64.b64decode(metadata["payload_base64"], validate=True)
    concatenated = compressed + gzip.compress(b"{}", mtime=0)
    metadata["payload_base64"] = base64.b64encode(concatenated).decode("ascii")
    metadata["compressed_byte_count"] = len(concatenated)
    metadata["compressed_sha256"] = hashlib.sha256(concatenated).hexdigest()
    digest = _reseal_snapshot(value)
    path = tmp_path / "snapshot.json"
    _write_canonical(path, value)
    with pytest.raises(RetiredPipelineArchiveError, match="trailing data"):
        load_retired_pipeline_source_archive(path, expected_record_digest=digest)


def test_high_expansion_gzip_fails_at_output_cap_without_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = gzip.compress(b"x" * 1_000_000, compresslevel=9, mtime=0)
    monkeypatch.setattr(subject, "MAX_UNCOMPRESSED_ARCHIVE_BYTES", 10)
    with pytest.raises(RetiredPipelineArchiveError, match="output bound"):
        subject._bounded_gzip_decompress(payload)

    tree = ast.parse(SUBJECT.read_text(encoding="utf-8"), filename=str(SUBJECT))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_bounded_gzip_decompress"
    )
    assert all(
        not (isinstance(node, ast.Attribute) and node.attr == "flush")
        for node in ast.walk(function)
    )


def test_truncated_gzip_stream_is_rejected() -> None:
    payload = gzip.compress(b"bounded historical source", mtime=0)
    with pytest.raises(RetiredPipelineArchiveError, match="gzip differs"):
        subject._bounded_gzip_decompress(payload[:-8])


def test_inner_source_tampering_is_rejected_after_full_reseal(tmp_path: Path) -> None:
    value = _snapshot_value()
    metadata = value["archive"]
    compressed = base64.b64decode(metadata["payload_base64"], validate=True)
    archive = json.loads(gzip.decompress(compressed))
    archive["sources"][0]["source_utf8"] += "\n"
    raw_archive = canonical_json(archive)
    rebuilt = gzip.compress(raw_archive, compresslevel=9, mtime=0)
    metadata.update(
        {
            "compressed_byte_count": len(rebuilt),
            "compressed_sha256": hashlib.sha256(rebuilt).hexdigest(),
            "payload_base64": base64.b64encode(rebuilt).decode("ascii"),
            "uncompressed_byte_count": len(raw_archive),
            "uncompressed_sha256": hashlib.sha256(raw_archive).hexdigest(),
        }
    )
    digest = _reseal_snapshot(value)
    path = tmp_path / "snapshot.json"
    _write_canonical(path, value)
    with pytest.raises(RetiredPipelineArchiveError, match="preimage"):
        load_retired_pipeline_source_archive(path, expected_record_digest=digest)


def test_snapshot_symlink_is_rejected(tmp_path: Path) -> None:
    link = tmp_path / "snapshot-link.json"
    link.symlink_to(DEFAULT_SOURCE_SNAPSHOT)
    with pytest.raises(RetiredPipelineArchiveError, match="regular file"):
        load_retired_pipeline_source_archive(link)


def test_artifact_binding_and_receipt_round_trip_canonically(tmp_path: Path) -> None:
    receipt = _receipt()
    assert receipt.execution_authorized is False
    assert receipt.deletion_authorized is False
    assert receipt.files_removed == 0
    assert receipt.to_data()["claim"] == RETIREMENT_RECEIPT_CLAIM
    assert RetiredPipelineRetirementReceipt.from_data(receipt.to_data()) == receipt
    assert tuple(item.relative_path for item in receipt.artifact_bindings) == (
        "archive/exposure.json",
        "archive/legacy-result.json",
    )

    path = tmp_path / "receipt.json"
    _write_canonical(path, receipt.to_data())
    assert load_retirement_receipt(
        path, expected_record_digest=receipt.record_digest
    ) == receipt
    with pytest.raises(RetiredPipelineArchiveError, match="address differs"):
        load_retirement_receipt(
            path, expected_record_digest="sha256:" + "f" * 64
        )


def test_receipt_has_no_mutable_nested_aliases() -> None:
    receipt = _receipt()
    original = receipt.to_data()
    exposed = receipt.to_data()
    exposed["legacy_modules"].append("bongard.tampered")
    exposed["artifact_bindings"][0]["relative_path"] = "tampered/path.json"
    assert receipt.to_data() == original
    assert type(receipt.legacy_modules) is tuple
    assert type(receipt.artifact_bindings) is tuple
    with pytest.raises(FrozenInstanceError):
        receipt.files_removed = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        receipt.artifact_bindings[0].disposition = (  # type: ignore[misc]
            "retain_irreducible"
        )


@pytest.mark.parametrize(
    "override",
    (
        {"execution_authorized": True},
        {"deletion_authorized": True},
        {"files_removed": 1},
        {"files_removed": False},
    ),
)
def test_receipt_creation_can_never_claim_authority(override: dict[str, object]) -> None:
    archive = load_retired_pipeline_source_archive()
    arguments = {
        "active_successor_pipeline_id": "successor-v1",
        "source_snapshot_record_digest": archive.record_digest,
        "source_snapshot_file_sha256": archive.snapshot_file_sha256,
        "source_snapshot_entry_count": len(archive.entries),
        "legacy_modules": ("bongard.panel_action_count_phase_command",),
        "artifact_bindings": (
            RetiredPipelineArtifactBinding(
                relative_path="archive/result.json",
                raw_sha256="sha256:" + "1" * 64,
                byte_count=1,
                canonical_record_digest=None,
                disposition="retain_irreducible",
            ),
        ),
        **override,
    }
    with pytest.raises(RetiredPipelineArchiveError, match="cannot grant authority"):
        RetiredPipelineRetirementReceipt.create(**arguments)


def test_resealed_receipt_authority_tampering_still_fails() -> None:
    value = _receipt().to_data()
    value["deletion_authorized"] = True
    body = dict(value)
    body.pop("record_digest")
    value["record_digest"] = "sha256:" + canonical_digest(body)
    with pytest.raises(RetiredPipelineArchiveError, match="cannot grant authority"):
        RetiredPipelineRetirementReceipt.from_data(value)


def test_receipt_rejects_duplicate_module_or_artifact_paths() -> None:
    receipt = _receipt()
    binding = receipt.artifact_bindings[0]
    with pytest.raises(RetiredPipelineArchiveError, match="legacy module"):
        RetiredPipelineRetirementReceipt.create(
            active_successor_pipeline_id=receipt.active_successor_pipeline_id,
            source_snapshot_record_digest=receipt.source_snapshot_record_digest,
            source_snapshot_file_sha256=receipt.source_snapshot_file_sha256,
            source_snapshot_entry_count=receipt.source_snapshot_entry_count,
            legacy_modules=(receipt.legacy_modules[0], receipt.legacy_modules[0]),
            artifact_bindings=(binding,),
        )
    with pytest.raises(RetiredPipelineArchiveError, match="artifact inventory"):
        RetiredPipelineRetirementReceipt.create(
            active_successor_pipeline_id=receipt.active_successor_pipeline_id,
            source_snapshot_record_digest=receipt.source_snapshot_record_digest,
            source_snapshot_file_sha256=receipt.source_snapshot_file_sha256,
            source_snapshot_entry_count=receipt.source_snapshot_entry_count,
            legacy_modules=(receipt.legacy_modules[0],),
            artifact_bindings=(binding, binding),
        )


def test_decoder_source_has_no_code_loading_or_cli_surface() -> None:
    source = SUBJECT.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SUBJECT))
    banned_calls: list[str] = []
    banned_imports: list[str] = []
    main_surfaces: list[str] = []
    retired_import_prefixes = (
        "bongard.panel_action_count_",
        "bongard.panel_action_decomposition_",
        "bongard.panel_soft_",
        "bongard.panel_typed_axis_custody_v2",
        "bongard.panel_typed_axis_task_runner_v2",
    )
    dynamic_attributes = {
        "exec_module",
        "find_spec",
        "import_module",
        "load_module",
        "module_from_spec",
        "run_module",
        "run_path",
        "spec_from_file_location",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {
                "__import__",
                "compile",
                "eval",
                "exec",
            }:
                banned_calls.append(node.func.id)
            if isinstance(node.func, ast.Attribute) and node.func.attr in dynamic_attributes:
                banned_calls.append(node.func.attr)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"importlib", "runpy"} or alias.name.startswith(
                    retired_import_prefixes
                ):
                    banned_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in {"importlib", "runpy"} or module.startswith(
                retired_import_prefixes
            ):
                banned_imports.append(module)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "main":
                main_surfaces.append(node.name)
        elif isinstance(node, ast.If):
            if "__name__" in ast.unparse(node.test):
                main_surfaces.append(ast.unparse(node.test))
    assert banned_calls == []
    assert banned_imports == []
    assert main_surfaces == []
    assert "importlib" not in source
    assert "__import__" not in source

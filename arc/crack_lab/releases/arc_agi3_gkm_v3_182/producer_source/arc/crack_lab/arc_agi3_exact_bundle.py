#!/usr/bin/env python3
"""Exact, atomic manifests for ARC-AGI-3 evidence bundles.

The manifest covers every regular file and directory below a bundle root,
except the manifest itself. Verification fails closed on changed, missing, or
extra entries and on symlinks, hard links, and special files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import uuid
from pathlib import Path
from typing import Any


SCHEMA = 1
KIND = "arc_agi3_exact_bundle_manifest"
MANIFEST_NAME = "BUNDLE_MANIFEST.json"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
ARCHIVE_MAX_FILES_PER_BUNDLE = 256
ARCHIVE_MAX_DIRECTORIES_PER_BUNDLE = 64
ARCHIVE_MAX_BYTES_PER_BUNDLE = 16 * 1024 * 1024
ARCHIVE_FORBIDDEN_COMPONENTS = frozenset(
    {
        ".campaign_locks",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "artifact_lineage",
        "interrupted_snapshot",
        "protected_transcript",
        "scratch_workspace",
        "wip_context",
        "workspace",
    }
)
ARCHIVE_FORBIDDEN_NAMES = frozenset(
    {
        ".DS_Store",
        ".orchestrate.lock",
        "latest.json",
        "proposer_last.log",
        "raw_codex_turn.jsonl",
    }
)


class ExactBundleError(RuntimeError):
    """An evidence bundle is unsafe, incomplete, stale, or malformed."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _regular_file_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ExactBundleError(f"cannot open regular file: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ExactBundleError(
                f"bundle file must be regular and unaliased: {path}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        for field in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_gid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        ):
            if getattr(before, field) != getattr(after, field):
                raise ExactBundleError(
                    f"bundle file changed while being read: {path}"
                )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(_regular_file_bytes(path)).hexdigest()


def _validate_root(root: Path) -> Path:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ExactBundleError("bundle root must be a regular directory")
    return root


def _snapshot(
    root: Path,
    *,
    manifest_name: str = MANIFEST_NAME,
) -> tuple[list[str], dict[str, str]]:
    root = _validate_root(root)
    directories: list[str] = []
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ExactBundleError(
                f"bundle contains a symlink: {relative}"
            )
        if path.is_dir():
            directories.append(relative)
            continue
        if not path.is_file():
            raise ExactBundleError(
                f"bundle contains a special entry: {relative}"
            )
        if relative == manifest_name:
            continue
        files[relative] = _sha256_file(path)
    return directories, files


def build_manifest(
    root: Path,
    *,
    bundle_id: str,
    manifest_name: str = MANIFEST_NAME,
) -> dict[str, Any]:
    if (
        not isinstance(bundle_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", bundle_id)
        is None
    ):
        raise ExactBundleError("bundle_id is malformed")
    directories, files = _snapshot(
        root,
        manifest_name=manifest_name,
    )
    return {
        "schema": SCHEMA,
        "kind": KIND,
        "bundle_id": bundle_id,
        "directories": directories,
        "files_sha256": files,
    }


def write_manifest_atomic(
    root: Path,
    *,
    bundle_id: str,
    manifest_name: str = MANIFEST_NAME,
    replace: bool = False,
    fault_at: str | None = None,
) -> dict[str, Any]:
    """Create one durable manifest without exposing partial bytes."""

    root = _validate_root(root)
    target = root / manifest_name
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ExactBundleError("manifest target is unsafe")
    if target.exists() and not replace:
        raise ExactBundleError("manifest already exists")
    manifest = build_manifest(
        root,
        bundle_id=bundle_id,
        manifest_name=manifest_name,
    )
    payload = _canonical_json(manifest) + b"\n"
    pending = root / f".{manifest_name}.pending-{uuid.uuid4().hex}"
    descriptor = -1
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(pending, flags, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ExactBundleError("short manifest write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        if fault_at == "after_file_sync":
            raise RuntimeError("injected pre-publication failure")
        os.replace(pending, target)
        _fsync_directory(root)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if pending.exists() or pending.is_symlink():
            pending.unlink()
    verified = verify_manifest(
        root,
        manifest_name=manifest_name,
        expected_bundle_id=bundle_id,
    )
    if verified["manifest"] != manifest:
        raise ExactBundleError("published manifest changed after creation")
    return verified


def _validate_declared_files(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ExactBundleError("files_sha256 must be an object")
    result: dict[str, str] = {}
    for relative, digest in value.items():
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
        ):
            raise ExactBundleError(
                f"invalid manifest file entry: {relative!r}"
            )
        result[relative] = digest
    return result


def verify_manifest(
    root: Path,
    *,
    manifest_name: str = MANIFEST_NAME,
    expected_bundle_id: str | None = None,
) -> dict[str, Any]:
    root = _validate_root(root)
    manifest_path = root / manifest_name
    raw = _regular_file_bytes(manifest_path)
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExactBundleError("manifest is not valid JSON") from exc
    required = {
        "schema",
        "kind",
        "bundle_id",
        "directories",
        "files_sha256",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != required
        or manifest["schema"] != SCHEMA
        or isinstance(manifest["schema"], bool)
        or manifest["kind"] != KIND
        or not isinstance(manifest["bundle_id"], str)
        or (
            expected_bundle_id is not None
            and manifest["bundle_id"] != expected_bundle_id
        )
        or not isinstance(manifest["directories"], list)
        or any(
            not isinstance(value, str)
            for value in manifest["directories"]
        )
        or manifest["directories"]
        != sorted(set(manifest["directories"]))
    ):
        raise ExactBundleError("manifest schema or identity mismatch")
    declared_files = _validate_declared_files(
        manifest["files_sha256"]
    )
    actual_directories, actual_files = _snapshot(
        root,
        manifest_name=manifest_name,
    )
    if actual_directories != manifest["directories"]:
        raise ExactBundleError(
            "bundle directory set differs from manifest"
        )
    if set(actual_files) != set(declared_files):
        raise ExactBundleError(
            "bundle file set differs from manifest"
        )
    mismatched = sorted(
        relative
        for relative, digest in actual_files.items()
        if declared_files[relative] != digest
    )
    if mismatched:
        raise ExactBundleError(
            "bundle file hash differs from manifest: "
            + ", ".join(mismatched)
        )
    if _sha256_file(manifest_path) != hashlib.sha256(raw).hexdigest():
        raise ExactBundleError("manifest changed during verification")
    return {
        "status": "PASS",
        "bundle_id": manifest["bundle_id"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "file_count": len(actual_files),
        "directory_count": len(actual_directories),
        "manifest": manifest,
    }


def _verify_archive_retention(root: Path, result: dict[str, Any]) -> None:
    """Reject mutable workspaces and oversized archaeology from quarantine."""

    manifest = result["manifest"]
    files = sorted(manifest["files_sha256"])
    directories = manifest["directories"]
    if len(files) > ARCHIVE_MAX_FILES_PER_BUNDLE:
        raise ExactBundleError(
            "archive bundle exceeds retained-file cap: "
            f"{root.name} has {len(files)}, cap "
            f"{ARCHIVE_MAX_FILES_PER_BUNDLE}"
        )
    if len(directories) > ARCHIVE_MAX_DIRECTORIES_PER_BUNDLE:
        raise ExactBundleError(
            "archive bundle exceeds retained-directory cap: "
            f"{root.name} has {len(directories)}, cap "
            f"{ARCHIVE_MAX_DIRECTORIES_PER_BUNDLE}"
        )

    forbidden: list[str] = []
    for relative in directories + files:
        parts = relative.split("/")
        name = parts[-1]
        if (
            ARCHIVE_FORBIDDEN_COMPONENTS.intersection(parts)
            or name in ARCHIVE_FORBIDDEN_NAMES
            or name.startswith("codex_turn_")
            and name.endswith(".jsonl")
            or name.endswith((".pyc", ".pyo"))
        ):
            forbidden.append(relative)
    if forbidden:
        raise ExactBundleError(
            "archive bundle contains stale operational paths: "
            + ", ".join(sorted(forbidden)[:8])
        )

    retained_bytes = sum((root / relative).stat().st_size for relative in files)
    if retained_bytes > ARCHIVE_MAX_BYTES_PER_BUNDLE:
        raise ExactBundleError(
            "archive bundle exceeds retained-byte cap: "
            f"{root.name} has {retained_bytes}, cap "
            f"{ARCHIVE_MAX_BYTES_PER_BUNDLE}"
        )


def verify_archive(archive_root: Path) -> dict[str, Any]:
    """Verify every immediate child as one exact, closed bundle."""

    archive_root = _validate_root(archive_root)
    children: list[Path] = []
    for entry in sorted(archive_root.iterdir()):
        if entry.is_symlink() or not entry.is_dir():
            raise ExactBundleError(
                "archive root contains a non-bundle entry: "
                f"{entry.name}"
            )
        children.append(entry)
    if not children:
        raise ExactBundleError("archive contains no bundles")
    results = []
    for child in children:
        result = verify_manifest(child)
        _verify_archive_retention(child, result)
        results.append(result)
    return {
        "status": "PASS",
        "archive_root": str(archive_root),
        "bundle_count": len(results),
        "file_count": sum(result["file_count"] for result in results),
        "directory_count": sum(
            result["directory_count"] for result in results
        ),
        "bundle_manifests_sha256": {
            result["bundle_id"]: result["manifest_sha256"]
            for result in results
        },
        "retention_policy": {
            "max_files_per_bundle": ARCHIVE_MAX_FILES_PER_BUNDLE,
            "max_directories_per_bundle": (
                ARCHIVE_MAX_DIRECTORIES_PER_BUNDLE
            ),
            "max_bytes_per_bundle": ARCHIVE_MAX_BYTES_PER_BUNDLE,
            "stale_operational_paths_forbidden": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--bundle-id", required=True)
    create.add_argument("--replace", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--bundle-id")
    verify_archive_parser = subparsers.add_parser("verify-archive")
    verify_archive_parser.add_argument(
        "--root", type=Path, required=True
    )
    args = parser.parse_args(argv)
    if args.command == "create":
        result = write_manifest_atomic(
            args.root,
            bundle_id=args.bundle_id,
            replace=args.replace,
        )
    elif args.command == "verify":
        result = verify_manifest(
            args.root,
            expected_bundle_id=args.bundle_id,
        )
    else:
        result = verify_archive(args.root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

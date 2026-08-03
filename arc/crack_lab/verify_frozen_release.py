#!/usr/bin/env python3
"""Verify a frozen ARC-AGI-3 receipt in its bound historical control context.

Release receipts deliberately bind verifier and control-file bytes.  A later
checkout must therefore not rebuild an old receipt with whatever controls are
currently on disk.  This wrapper reads the receipt's ``source_revision``,
extracts only its hash-bound verifier/control/environment files from local Git,
checks those bytes before execution, and runs the historical release gate
against the requested immutable artifact tree.  An already extracted verifier
checkout can be supplied for source archives without Git history.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


MAX_RECEIPT_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
SUBPROCESS_TIMEOUT_SECONDS = 180
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
GATE_PATH = "arc/crack_lab/arc_agi3_release_gate.py"


class FrozenReleaseError(RuntimeError):
    """The receipt or its historical verification context is invalid."""


def _sanitized_environment() -> dict[str, str]:
    """Return a secret-free environment for Git and historical verification."""
    return {
        "PATH": os.environ.get("PATH", os.defpath),
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONNOUSERSITE": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
    }


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FrozenReleaseError("receipt is not canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise FrozenReleaseError(f"{label} contains a non-path key")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise FrozenReleaseError(f"{label} contains an unsafe path: {value!r}")
    return value


def _hash_map(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise FrozenReleaseError(f"{label} must be a nonempty hash map")
    result: dict[str, str] = {}
    for raw_path, digest in value.items():
        path = _safe_relative_path(raw_path, label=label)
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise FrozenReleaseError(f"{label} has an invalid SHA-256 for {path}")
        result[path] = digest
    return result


def load_receipt(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    receipt = Path(path)
    try:
        metadata = receipt.lstat()
    except OSError as exc:
        raise FrozenReleaseError("cannot stat release receipt") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size > MAX_RECEIPT_BYTES
    ):
        raise FrozenReleaseError("release receipt is not a bounded single-link file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(receipt, flags)
    except OSError as exc:
        raise FrozenReleaseError("cannot securely open release receipt") from exc
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size > MAX_RECEIPT_BYTES
        ):
            raise FrozenReleaseError(
                "release receipt changed during bounded read"
            )
        raw = handle.read(MAX_RECEIPT_BYTES + 1)
    if len(raw) > MAX_RECEIPT_BYTES:
        raise FrozenReleaseError("release receipt is unexpectedly large")
    digest = _sha256_bytes(raw)
    if receipt.suffix != ".json" or receipt.stem != digest:
        raise FrozenReleaseError("release receipt filename is not its content hash")
    try:
        body = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FrozenReleaseError("release receipt is invalid JSON") from exc
    if not isinstance(body, dict) or raw != _canonical_json(body) + b"\n":
        raise FrozenReleaseError("release receipt bytes are not canonical JSON")

    identity = body.get("release_identity")
    revision = identity.get("source_revision") if isinstance(identity, dict) else None
    if not isinstance(revision, str) or not REVISION_RE.fullmatch(revision):
        raise FrozenReleaseError("release receipt has no valid source revision")

    expected: dict[str, str] = {}
    for label, value in (
        ("control contract", body.get("control_contract", {}).get("files_sha256")),
        ("verifier", body.get("verifier", {}).get("files_sha256")),
    ):
        for bound_path, bound_hash in _hash_map(value, label=label).items():
            previous = expected.get(bound_path)
            if previous is not None and previous != bound_hash:
                raise FrozenReleaseError(
                    f"receipt assigns conflicting hashes to {bound_path}"
                )
            expected[bound_path] = bound_hash
    for metadata_path, metadata_hash in _hash_map(
        body.get("inventory_metadata_sha256"), label="inventory metadata"
    ).items():
        bound_path = f"environment_files/{metadata_path}"
        previous = expected.get(bound_path)
        if previous is not None and previous != metadata_hash:
            raise FrozenReleaseError(
                f"receipt assigns conflicting hashes to {bound_path}"
            )
        expected[bound_path] = metadata_hash
    if GATE_PATH not in expected:
        raise FrozenReleaseError("receipt does not bind the release-gate source")
    return body, expected


def _validate_bound_root(root: Path, expected: Mapping[str, str]) -> Path:
    supplied = Path(root)
    try:
        supplied_metadata = supplied.lstat()
    except OSError as exc:
        raise FrozenReleaseError("cannot stat historical verifier root") from exc
    if stat.S_ISLNK(supplied_metadata.st_mode):
        raise FrozenReleaseError("historical verifier root must not be a symlink")
    resolved = supplied.resolve()
    try:
        metadata = resolved.lstat()
    except OSError as exc:
        raise FrozenReleaseError("cannot stat historical verifier root") from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise FrozenReleaseError("historical verifier root is not a real directory")
    for relative, expected_hash in sorted(expected.items()):
        _read_bound_file(resolved, relative, expected_hash)
    return resolved


def _read_bound_file(root: Path, relative: str, expected_hash: str) -> bytes:
    """Read one bound file while rejecting symlinked path components."""
    pure = PurePosixPath(relative)
    cursor = root
    for part in pure.parts[:-1]:
        cursor /= part
        try:
            item = cursor.lstat()
        except OSError as exc:
            raise FrozenReleaseError(
                f"historical verifier root is missing {relative}"
            ) from exc
        if (
            not stat.S_ISDIR(item.st_mode)
            or stat.S_ISLNK(item.st_mode)
        ):
            raise FrozenReleaseError(
                f"historical verifier path component is not a real directory: {relative}"
            )
    path = root.joinpath(*pure.parts)
    try:
        item = path.lstat()
    except OSError as exc:
        raise FrozenReleaseError(
            f"historical verifier root is missing {relative}"
        ) from exc
    if (
        not stat.S_ISREG(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_nlink != 1
        or item.st_size > MAX_RECEIPT_BYTES
    ):
        raise FrozenReleaseError(
            f"historical verifier byte mismatch at {relative}"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FrozenReleaseError(
            f"cannot securely open historical verifier file {relative}"
        ) from exc
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise FrozenReleaseError(
                f"historical verifier file changed during read: {relative}"
            )
        raw = handle.read(MAX_RECEIPT_BYTES + 1)
    if len(raw) > MAX_RECEIPT_BYTES or _sha256_bytes(raw) != expected_hash:
        raise FrozenReleaseError(
            f"historical verifier byte mismatch at {relative}"
        )
    return raw


def _copy_bound_root(
    supplied: Path, expected: Mapping[str, str], output: Path
) -> Path:
    """Copy only receipt-allowlisted bytes into a private execution tree."""
    source = _validate_bound_root(supplied, expected)
    for relative, expected_hash in sorted(expected.items()):
        raw = _read_bound_file(source, relative, expected_hash)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(raw)
    return _validate_bound_root(output, expected)


def _archive_bound_root(
    *, repo_root: Path, revision: str, expected: Mapping[str, str], output: Path
) -> Path:
    repo = Path(repo_root).resolve()
    if not (repo / ".git").exists():
        raise FrozenReleaseError(
            "local Git history is unavailable; supply --verifier-root"
        )
    clean_env = _sanitized_environment()
    for check in (
        ["git", "-C", str(repo), "cat-file", "-e", f"{revision}^{{commit}}"],
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", revision, "HEAD"],
    ):
        try:
            checked = subprocess.run(
                check,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env=clean_env,
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise FrozenReleaseError("historical Git revision check timed out") from exc
        if checked.returncode != 0:
            raise FrozenReleaseError(
                "receipt source revision is unavailable or is not an ancestor of HEAD"
            )
    command = [
        "git",
        "-C",
        str(repo),
        "archive",
        "--format=tar",
        revision,
        "--",
        *sorted(expected),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=clean_env,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise FrozenReleaseError("historical verifier extraction timed out") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise FrozenReleaseError(
            "cannot extract the receipt-bound source revision"
            + (f": {detail[:500]}" if detail else "")
        )
    if len(completed.stdout) > MAX_ARCHIVE_BYTES:
        raise FrozenReleaseError("historical verifier archive is unexpectedly large")

    expected_names = set(expected)
    expected_directories = {
        parent.as_posix()
        for name in expected_names
        for parent in PurePosixPath(name).parents
        if parent.as_posix() != "."
    }
    try:
        archive = tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:")
    except tarfile.TarError as exc:
        raise FrozenReleaseError("historical verifier archive is invalid") from exc
    with archive:
        seen: set[str] = set()
        seen_casefolded: set[str] = set()
        for member in archive:
            name = _safe_relative_path(member.name, label="historical archive")
            folded = name.casefold()
            if name in seen or folded in seen_casefolded:
                raise FrozenReleaseError(
                    f"historical archive contains duplicate entry {name}"
                )
            seen.add(name)
            seen_casefolded.add(folded)
            target = output / name
            if member.isdir():
                if name not in expected_directories:
                    raise FrozenReleaseError(
                        f"historical archive contains unexpected directory {name}"
                    )
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or name not in expected_names:
                raise FrozenReleaseError(
                    f"historical archive contains unexpected entry {name}"
                )
            if member.size < 0 or member.size > MAX_RECEIPT_BYTES:
                raise FrozenReleaseError(
                    f"historical archive file is unexpectedly large: {name}"
                )
            source = archive.extractfile(member)
            if source is None:
                raise FrozenReleaseError(f"cannot extract historical file {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(source.read())
    return _validate_bound_root(output, expected)


def verify_frozen_release(
    *,
    receipt_path: Path,
    canonical_root: Path,
    repo_root: Path,
    verifier_root: Path | None = None,
) -> dict[str, Any]:
    body, expected = load_receipt(receipt_path)
    revision = body["release_identity"]["source_revision"]
    receipt_sha256 = Path(receipt_path).stem

    with tempfile.TemporaryDirectory(prefix="gkm-release-verifier-") as tmp_name:
        private_root = Path(tmp_name) / "bound"
        if verifier_root is None:
            root = _archive_bound_root(
                repo_root=repo_root,
                revision=revision,
                expected=expected,
                output=private_root,
            )
        else:
            root = _copy_bound_root(verifier_root, expected, private_root)
        gate = root / GATE_PATH
        mode = (
            "verify-partial"
            if body.get("kind") == "partial_campaign_freeze"
            else "verify"
        )
        command = [
            sys.executable,
            str(gate),
            "--canonical-root",
            str(Path(canonical_root).resolve()),
            "--environments-root",
            str(root / "environment_files"),
            mode,
            "--receipt",
            str(Path(receipt_path).resolve()),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_sanitized_environment(),
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise FrozenReleaseError("historical release gate timed out") from exc
        if completed.returncode != 0:
            detail = (completed.stdout + "\n" + completed.stderr).strip()
            raise FrozenReleaseError(
                "historical release gate failed"
                + (f": {detail[-1000:]}" if detail else "")
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise FrozenReleaseError(
                "historical release gate returned invalid JSON"
            ) from exc
        if not isinstance(result, dict) or result.get("status") != "PASS":
            raise FrozenReleaseError("historical release gate did not pass")
        result["receipt_sha256"] = receipt_sha256
        result["verification_context_source_revision"] = revision
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Git repository containing the receipt-bound source revision",
    )
    parser.add_argument(
        "--verifier-root",
        type=Path,
        help="already extracted receipt-bound source tree (Git fallback)",
    )
    args = parser.parse_args()
    try:
        result = verify_frozen_release(
            receipt_path=args.receipt.resolve(),
            canonical_root=args.canonical_root.resolve(),
            repo_root=args.repo_root.resolve(),
            verifier_root=(
                args.verifier_root.resolve() if args.verifier_root else None
            ),
        )
    except (FrozenReleaseError, OSError) as exc:
        print(_canonical_json({"status": "FAIL", "error": str(exc)}).decode())
        return 1
    print(_canonical_json(result).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

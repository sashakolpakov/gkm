#!/usr/bin/env python3
"""Generate the controller image's canonical in-image supply-chain manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Sequence


DEFAULT_OUTPUT = Path(
    "/usr/local/share/arc-agi3/controller-supply-chain.json"
)
OBSERVED_PATHS = (
    (Path("/usr/local/bin/arc-agi3-contiguous-controller-guardian"), True),
    (Path("/usr/local/bin/codex"), True),
    (Path("/usr/local/lib/codex/bin/codex.js"), True),
    (Path("/usr/local/share/arc-agi3/codex-package.json"), False),
    (
        Path(
            "/usr/local/share/arc-agi3/"
            "app-server-protocol.schemas.json"
        ),
        False,
    ),
    (
        Path(
            "/usr/local/share/arc-agi3/"
            "app-server-protocol.v2.schemas.json"
        ),
        False,
    ),
)


class SupplyChainError(RuntimeError):
    """The immutable controller supply chain could not be recorded."""


MAX_OBSERVED_FILE_BYTES = 512 * 1024 * 1024


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _read_regular(path: Path, *, executable: bool) -> bytes:
    try:
        before = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise SupplyChainError(
            f"controller supply-chain input is unavailable: {path}"
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 0 < before.st_size <= MAX_OBSERVED_FILE_BYTES
        or bool(before.st_mode & 0o111) is not executable
    ):
        raise SupplyChainError(
            f"controller supply-chain input is unsafe: {path}"
        )
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        current = os.fstat(descriptor)
        if (
            current.st_dev,
            current.st_ino,
            current.st_mode,
            current.st_nlink,
            current.st_uid,
            current.st_gid,
            current.st_size,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_uid,
            before.st_gid,
            before.st_size,
        ):
            raise SupplyChainError(
                f"controller supply-chain input changed: {path}"
            )
        raw = bytearray()
        while len(raw) <= MAX_OBSERVED_FILE_BYTES:
            block = os.read(
                descriptor,
                min(
                    1024 * 1024,
                    MAX_OBSERVED_FILE_BYTES + 1 - len(raw),
                ),
            )
            if not block:
                break
            raw.extend(block)
        if len(raw) != before.st_size:
            raise SupplyChainError(
                f"controller supply-chain input changed: {path}"
            )
        return bytes(raw)
    except OSError as exc:
        raise SupplyChainError(
            f"controller supply-chain input cannot be read: {path}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def build_manifest(*, codex_cli_version: str) -> dict[str, object]:
    if (
        not isinstance(codex_cli_version, str)
        or not codex_cli_version.startswith("codex-cli ")
        or len(codex_cli_version) > 128
        or any(
            character in codex_cli_version
            for character in ("\x00", "\n", "\r")
        )
    ):
        raise SupplyChainError("Codex CLI version is malformed")
    files: list[dict[str, object]] = []
    for path, executable in OBSERVED_PATHS:
        raw = _read_regular(path, executable=executable)
        files.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "executable": executable,
            }
        )
    return {
        "schema": 1,
        "kind": "arc_agi3_controller_supply_chain",
        "codex_cli_version": codex_cli_version,
        "files": files,
    }


def write_new_manifest(
    path: Path,
    value: dict[str, object],
) -> str:
    target = Path(path)
    if not target.is_absolute() or target.parent.is_symlink():
        raise SupplyChainError(
            "controller supply-chain output must be an absolute unaliased path"
        )
    payload = _canonical_json(value) + b"\n"
    descriptor = os.open(
        target,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise SupplyChainError(
                    "short controller supply-chain manifest write"
                )
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--codex-cli-version", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    digest = write_new_manifest(
        args.output,
        build_manifest(codex_cli_version=args.codex_cli_version),
    )
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

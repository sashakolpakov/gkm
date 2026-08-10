"""Allowlist-based construction of a solver-visible source bundle.

This Phase-0 guard proves projection semantics.  A production solver campaign
must additionally run in a separate process with only the projected directory
mounted or copied into its workspace.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

PUBLIC_SOURCE_PATHS = ("README.md", "interface.py", "protocol.py")


class SourceBoundaryError(ValueError):
    """Raised when a source or destination escapes an explicit allowlist."""


def package_root() -> Path:
    return Path(__file__).resolve().parent


def list_public_sources() -> tuple[str, ...]:
    return PUBLIC_SOURCE_PATHS


def _public_source(relative_path: str) -> Path:
    candidate = PurePosixPath(relative_path)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or relative_path not in PUBLIC_SOURCE_PATHS
    ):
        raise SourceBoundaryError(f"source is not public: {relative_path!r}")

    root = package_root()
    source = root / relative_path
    if source.is_symlink():
        raise SourceBoundaryError(f"public source may not be a symlink: {relative_path}")
    resolved = source.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise SourceBoundaryError(f"public source escaped package: {relative_path}")
    if not resolved.is_file():
        raise SourceBoundaryError(f"public source is not a file: {relative_path}")
    return resolved


def read_public_source(relative_path: str) -> str:
    return _public_source(relative_path).read_text(encoding="utf-8")


def _resolved_destination(destination: Path, write_root: Path) -> tuple[Path, Path]:
    root = write_root.resolve(strict=True)
    if not root.is_dir():
        raise SourceBoundaryError("write_root must be a directory")
    if destination.is_symlink():
        raise SourceBoundaryError("destination may not be a symlink")
    prospective = destination.resolve(strict=False)
    if not prospective.is_relative_to(root):
        raise SourceBoundaryError("destination escaped the declared write root")
    destination.mkdir(parents=True, exist_ok=True)
    resolved = destination.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise SourceBoundaryError("destination escaped the declared write root")
    return resolved, root


def materialize_public_sources(destination: Path, *, write_root: Path) -> tuple[Path, ...]:
    """Copy only public files into ``destination`` below ``write_root``."""

    resolved_destination, resolved_root = _resolved_destination(destination, write_root)
    outputs: list[Path] = []
    for relative_path in PUBLIC_SOURCE_PATHS:
        source = _public_source(relative_path)
        output = resolved_destination / relative_path
        if output.is_symlink():
            raise SourceBoundaryError(f"output may not be a symlink: {relative_path}")
        if output.exists() and not output.is_file():
            raise SourceBoundaryError(f"output is not a regular file: {relative_path}")
        if not output.parent.resolve(strict=True).is_relative_to(resolved_root):
            raise SourceBoundaryError(f"output escaped write root: {relative_path}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(output, flags, 0o644)
        except OSError as error:
            raise SourceBoundaryError(
                f"could not safely create public output: {relative_path}"
            ) from error
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(source.read_bytes())
        outputs.append(output)
    return tuple(outputs)


__all__ = [
    "PUBLIC_SOURCE_PATHS",
    "SourceBoundaryError",
    "list_public_sources",
    "materialize_public_sources",
    "read_public_source",
]

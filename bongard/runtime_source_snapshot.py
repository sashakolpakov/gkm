"""Import-time source seals for executable campaign authorities.

Reading ``module.__file__`` later proves only what is on disk at that later
instant.  It does not prove that those bytes are the code Python imported.
Each participating module therefore calls :func:`capture_loaded_source` while
its module frame is executing.  The helper recompiles the exact bytes then on
disk and compares the complete marshalled module code tree (including every
nested function and comprehension) with the executing module code object.

The resulting source SHA-256 is immutable in this registry.  Campaign launch
and cold replay additionally require the current disk bytes to retain that
same digest, closing both the import-to-precommit and post-import drift gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import marshal
from pathlib import Path
import re
import sys
from types import CodeType


_RAW_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class RuntimeSourceSnapshotError(RuntimeError):
    """Loaded bytecode, import-time source, or current source disagrees."""


@dataclass(frozen=True, slots=True)
class LoadedSourceSnapshot:
    module_name: str
    source_path: str
    source_sha256: str
    module_code_sha256: str


_SNAPSHOTS: dict[str, LoadedSourceSnapshot] = {}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalized_constant(value: object) -> object:
    if isinstance(value, CodeType):
        return ("code", _normalized_code(value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_normalized_constant(item) for item in value))
    if isinstance(value, frozenset):
        encoded = sorted(marshal.dumps(_normalized_constant(item)) for item in value)
        return ("frozenset", tuple(encoded))
    return (type(value).__name__, value)


def _normalized_code(code: CodeType) -> tuple[object, ...]:
    """Return the complete stable identity of a code tree.

    CPython may attach interpreter-private state to a live code object.  The
    semantic fields below exclude that state while retaining nested bodies,
    bytecode, constants, names, locals, closures, positions, and exception
    tables.
    """

    return (
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_nlocals,
        code.co_stacksize,
        code.co_flags,
        code.co_code,
        tuple(_normalized_constant(item) for item in code.co_consts),
        code.co_names,
        code.co_varnames,
        code.co_filename,
        code.co_name,
        code.co_qualname,
        code.co_firstlineno,
        code.co_linetable,
        code.co_exceptiontable,
        code.co_freevars,
        code.co_cellvars,
    )


def _module_code_sha256(code: object) -> str:
    if not isinstance(code, CodeType):
        raise RuntimeSourceSnapshotError("module code identity is unavailable")
    return _sha256(repr(_normalized_code(code)).encode("utf-8", errors="strict"))


def capture_loaded_source(module_name: str, source_file: str) -> str:
    """Seal exact source after proving it compiled to the executing module."""

    if not isinstance(module_name, str) or not module_name:
        raise RuntimeSourceSnapshotError("module name must be nonempty text")
    if not isinstance(source_file, str) or not source_file:
        raise RuntimeSourceSnapshotError("module source path must be nonempty text")
    try:
        frame = sys._getframe(1)
    except (AttributeError, ValueError) as exc:  # pragma: no cover - CPython guard
        raise RuntimeSourceSnapshotError("executing module frame is unavailable") from exc
    if frame.f_globals.get("__name__") != module_name:
        raise RuntimeSourceSnapshotError("source seal must run in its module frame")

    source_path = Path(source_file).resolve(strict=True)
    source_bytes = source_path.read_bytes()
    try:
        compiled = compile(
            source_bytes,
            frame.f_code.co_filename,
            "exec",
            dont_inherit=True,
            optimize=sys.flags.optimize,
        )
    except (SyntaxError, UnicodeError, ValueError) as exc:
        raise RuntimeSourceSnapshotError(
            f"runtime source for {module_name!r} cannot be recompiled"
        ) from exc
    loaded_code = _normalized_code(frame.f_code)
    compiled_code = _normalized_code(compiled)
    if compiled_code != loaded_code:
        raise RuntimeSourceSnapshotError(
            f"runtime source for {module_name!r} differs from imported code"
        )
    loaded_code_digest = _module_code_sha256(frame.f_code)

    snapshot = LoadedSourceSnapshot(
        module_name=module_name,
        source_path=str(source_path),
        source_sha256=_sha256(source_bytes),
        module_code_sha256=loaded_code_digest,
    )
    previous = _SNAPSHOTS.setdefault(module_name, snapshot)
    if previous != snapshot:
        raise RuntimeSourceSnapshotError(
            f"runtime source seal for {module_name!r} was already fixed"
        )
    return snapshot.source_sha256


def verify_loaded_source(
    module_name: str, *, expected_source_sha256: str | None = None
) -> str:
    """Verify the immutable import seal against current bytes and an optional pin."""

    snapshot = _SNAPSHOTS.get(module_name)
    if snapshot is None:
        raise RuntimeSourceSnapshotError(
            f"runtime module {module_name!r} has no import-time source seal"
        )
    current = _sha256(Path(snapshot.source_path).read_bytes())
    if current != snapshot.source_sha256:
        raise RuntimeSourceSnapshotError(
            f"runtime source for {module_name!r} changed after import"
        )
    if expected_source_sha256 is not None:
        if (
            not isinstance(expected_source_sha256, str)
            or _RAW_SHA256.fullmatch(expected_source_sha256) is None
        ):
            raise RuntimeSourceSnapshotError(
                "expected runtime source digest must be lowercase SHA-256"
            )
        if snapshot.source_sha256 != expected_source_sha256:
            raise RuntimeSourceSnapshotError(
                f"runtime source for {module_name!r} differs from its precommit"
            )
    return snapshot.source_sha256


RUNTIME_SOURCE_SNAPSHOT_SHA256 = capture_loaded_source(__name__, __file__)


__all__ = (
    "LoadedSourceSnapshot",
    "RUNTIME_SOURCE_SNAPSHOT_SHA256",
    "RuntimeSourceSnapshotError",
    "capture_loaded_source",
    "verify_loaded_source",
)

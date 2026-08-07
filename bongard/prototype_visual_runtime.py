"""Exact dependency identity for the prototype visual observation stack.

The interpreter hash alone is not an adequate replay identity for raster
segmentation.  NumPy, SciPy, Pillow, zlib, and their native extension bytes
can change the pixels, masks, or component inventory while every project
source file remains unchanged.  This module produces a finite canonical
manifest over the installed distributions and native files used by that
stack.  It is Python-authoritative; Lean is neither imported nor consulted.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import hashlib
import importlib
import importlib.metadata
import importlib.util
from functools import lru_cache
from pathlib import Path
import platform
import re
import stat
import sys
from typing import Any
import zlib

from bongard.canonical import canonical_digest
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


MANIFEST_SCHEMA = "gkm.bongard-prototype-visual-runtime-manifest.v1"
MANIFEST_ALGORITHM_ID = "bongard.prototype-visual-runtime/exact-native-files-v1"
_DISTRIBUTIONS = ("numpy", "scipy", "Pillow")
_MODULES = (
    "numpy",
    "numpy._core._multiarray_umath",
    "scipy",
    "scipy.ndimage._nd_image",
    "PIL",
    "PIL._imaging",
    "zlib",
)
_NATIVE_SUFFIX = re.compile(r"(?:\.so(?:\.[0-9]+)*|\.dylib|\.pyd|\.dll)\Z")
_MAX_BOUND_FILE_BYTES = 512 * 1024 * 1024
_MAX_NATIVE_FILES = 4096


class PrototypeVisualRuntimeError(RuntimeError):
    """The installed visual dependency stack cannot be bound exactly."""


def _exact_regular_file(path: Path, *, label: str) -> dict[str, object]:
    try:
        resolved = path.resolve(strict=True)
        before = resolved.stat()
    except OSError as exc:
        raise PrototypeVisualRuntimeError(f"cannot resolve {label}") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_size < 0
        or before.st_size > _MAX_BOUND_FILE_BYTES
    ):
        raise PrototypeVisualRuntimeError(f"{label} is not a bounded regular file")
    try:
        payload = resolved.read_bytes()
        after = resolved.stat()
    except OSError as exc:
        raise PrototypeVisualRuntimeError(f"cannot read {label}") from exc
    if (
        len(payload) != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise PrototypeVisualRuntimeError(f"{label} changed while hashing")
    return {
        "resolved_path": str(resolved),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _module_record(module_name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise PrototypeVisualRuntimeError(
            f"cannot import visual runtime module {module_name!r}"
        ) from exc
    source = getattr(module, "__file__", None)
    if not isinstance(source, str) or not source:
        spec = importlib.util.find_spec(module_name)
        source = None if spec is None else spec.origin
    if not isinstance(source, str) or not source or source in {"built-in", "frozen"}:
        raise PrototypeVisualRuntimeError(
            f"visual runtime module {module_name!r} has no exact file"
        )
    return {
        "module": module_name,
        "file": _exact_regular_file(Path(source), label=f"module {module_name}"),
    }


def _distribution_record(distribution_name: str) -> dict[str, object]:
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise PrototypeVisualRuntimeError(
            f"visual distribution {distribution_name!r} is unavailable"
        ) from exc
    files = distribution.files
    if files is None:
        raise PrototypeVisualRuntimeError(
            f"visual distribution {distribution_name!r} has no file inventory"
        )
    native_paths: list[Path] = []
    record_paths: list[Path] = []
    for relative in files:
        text = str(relative)
        absolute = Path(distribution.locate_file(relative))
        if _NATIVE_SUFFIX.search(text) is not None:
            native_paths.append(absolute)
        if text.endswith(".dist-info/RECORD"):
            record_paths.append(absolute)
    if len(record_paths) != 1:
        raise PrototypeVisualRuntimeError(
            f"visual distribution {distribution_name!r} lacks one RECORD"
        )
    unique_native = tuple(sorted({path.resolve() for path in native_paths}, key=str))
    if len(unique_native) > _MAX_NATIVE_FILES:
        raise PrototypeVisualRuntimeError(
            f"visual distribution {distribution_name!r} has too many native files"
        )
    return {
        "distribution": distribution_name,
        "canonical_name": distribution.metadata.get("Name", distribution_name),
        "version": distribution.version,
        "record_file": _exact_regular_file(
            record_paths[0], label=f"{distribution_name} RECORD"
        ),
        "native_files": [
            _exact_regular_file(path, label=f"{distribution_name} native file")
            for path in unique_native
        ],
    }


def visual_runtime_dependency_manifest() -> dict[str, Any]:
    """Return a canonical-data manifest for every active raster dependency."""

    return {
        "schema": MANIFEST_SCHEMA,
        "algorithm_id": MANIFEST_ALGORITHM_ID,
        "source_sha256": _LOADED_SOURCE_SHA256,
        "platform": {
            "sys_platform": sys.platform,
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "byteorder": sys.byteorder,
        },
        "zlib": {
            "compile_version": zlib.ZLIB_VERSION,
            "runtime_version": zlib.ZLIB_RUNTIME_VERSION,
        },
        "modules": [_module_record(name) for name in _MODULES],
        "distributions": [
            _distribution_record(name) for name in _DISTRIBUTIONS
        ],
        "authority": {
            "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
            "python_is_canonical_authority": True,
            "lean_present": False,
            "lean_required": False,
            "lean_removable": True,
            "lean_affects_identity_or_decision": False,
        },
    }


@lru_cache(maxsize=1)
def visual_runtime_dependency_digest() -> str:
    """Return the content identity consumed by observation/replay protocols."""

    return canonical_digest(visual_runtime_dependency_manifest())


__all__ = [
    "MANIFEST_ALGORITHM_ID",
    "MANIFEST_SCHEMA",
    "PrototypeVisualRuntimeError",
    "visual_runtime_dependency_digest",
    "visual_runtime_dependency_manifest",
]

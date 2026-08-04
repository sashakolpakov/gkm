#!/usr/bin/env python3
"""Shared closed schema for contiguous-campaign winning solver source.

The same exact bytes are loaded by the trusted host publisher and baked into
the isolated proposer image.  Besides names and byte bounds, this module
proves a static import closure over the complete flat source set: source may
import another declared local ``.py`` stem, the Python standard library, or
the one version-pinned third-party dependency (NumPy).  Relative imports and
ambient package roots fail before WIP retention, isolated replay, or
promotion.  Callers remain responsible for descriptor-confined reads, alias
rejection, and durable publication.
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Mapping


SCHEMA = 2
REQUIRED_FILES = frozenset({"legs.py", "players.py", "solve.py"})
ALLOWED_SUFFIXES = frozenset({".py", ".json", ".txt"})
PINNED_NUMPY_VERSION = "2.4.4"
ALLOWED_THIRD_PARTY_ROOTS = frozenset({"numpy"})
STDLIB_ROOTS = frozenset(sys.stdlib_module_names)
FORBIDDEN_FILES = frozenset(
    {
        "candidate_path.json",
        "checkpoint.json",
        "current.json",
        "frontier_brief.json",
        "host_candidate_path.json",
        "host_promotion_receipt.json",
        "input_bundle_receipt.json",
        "promotion_manifest.json",
        "promotion_receipt.json",
        "worker_outcome.json",
        "wip_manifest.json",
    }
)
MAX_FILES = 16
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024
_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\.(?:py|json|txt)")


class SourceSchemaError(ValueError):
    """A proposed solver-source inventory is outside the closed schema."""


def _validate_import_root(
    *,
    filename: str,
    module: str,
    local_stems: frozenset[str],
) -> None:
    root = module.partition(".")[0]
    if root in local_stems:
        if module != root:
            raise SourceSchemaError(
                f"local source import is not flat in {filename}: {module}"
            )
        return
    if root in STDLIB_ROOTS or root in ALLOWED_THIRD_PARTY_ROOTS:
        return
    raise SourceSchemaError(
        f"source imports an undeclared ambient root in {filename}: {root}"
    )


def validate_source_import_closure(
    payloads: Mapping[str, bytes],
) -> None:
    """Prove that every static import resolves inside the closed runtime.

    Local modules are exactly the flat ``.py`` payload stems.  Dotted local
    packages would require undeclared paths, so they are rejected.  Local
    modules may not shadow a standard-library or pinned third-party root;
    otherwise the same import could resolve differently across host and
    isolated replay roles.
    """

    python_names = tuple(
        sorted(name for name in payloads if name.endswith(".py"))
    )
    local_stems = frozenset(name[:-3] for name in python_names)
    collisions = local_stems & (
        STDLIB_ROOTS | ALLOWED_THIRD_PARTY_ROOTS
    )
    if collisions:
        raise SourceSchemaError(
            "local source shadows a closed-runtime import root: "
            + ",".join(sorted(collisions))
        )
    for filename in python_names:
        try:
            source = payloads[filename].decode("utf-8")
            tree = ast.parse(
                source,
                filename=filename,
                mode="exec",
                feature_version=(3, 12),
            )
        except (SyntaxError, UnicodeError, ValueError) as exc:
            raise SourceSchemaError(
                f"source file is not valid pinned-runtime Python: {filename}"
            ) from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level != 0 or not node.module:
                    raise SourceSchemaError(
                        f"relative source import is forbidden in {filename}"
                    )
                _validate_import_root(
                    filename=filename,
                    module=node.module,
                    local_stems=local_stems,
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    _validate_import_root(
                        filename=filename,
                        module=alias.name,
                        local_stems=local_stems,
                    )


def validate_source_payloads(
    payloads: Mapping[str, bytes],
) -> tuple[str, ...]:
    """Validate and return the canonical flat source-file inventory."""

    if not isinstance(payloads, Mapping):
        raise SourceSchemaError("source payloads must be a mapping")
    names = tuple(sorted(payloads))
    if (
        not REQUIRED_FILES.issubset(names)
        or not len(names) <= MAX_FILES
        or len(names) != len(set(names))
    ):
        raise SourceSchemaError(
            "source inventory lacks core files or exceeds its bound"
        )
    total = 0
    for name in names:
        raw = payloads[name]
        if (
            not isinstance(name, str)
            or _NAME_RE.fullmatch(name) is None
            or name in FORBIDDEN_FILES
            or not isinstance(raw, bytes)
            or len(raw) > MAX_FILE_BYTES
        ):
            raise SourceSchemaError(
                f"source file is outside the closed schema: {name!r}"
            )
        try:
            raw.decode("utf-8")
        except UnicodeError as exc:
            raise SourceSchemaError(
                f"source file is not UTF-8: {name}"
            ) from exc
        total += len(raw)
        if total > MAX_TOTAL_BYTES:
            raise SourceSchemaError(
                "source payloads exceed their aggregate byte bound"
            )
    validate_source_import_closure(payloads)
    return names


__all__ = [
    "ALLOWED_SUFFIXES",
    "ALLOWED_THIRD_PARTY_ROOTS",
    "FORBIDDEN_FILES",
    "MAX_FILES",
    "MAX_FILE_BYTES",
    "MAX_TOTAL_BYTES",
    "REQUIRED_FILES",
    "SCHEMA",
    "PINNED_NUMPY_VERSION",
    "STDLIB_ROOTS",
    "SourceSchemaError",
    "validate_source_import_closure",
    "validate_source_payloads",
]

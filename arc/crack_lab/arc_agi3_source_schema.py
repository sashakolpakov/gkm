#!/usr/bin/env python3
"""Shared closed schema for contiguous-campaign winning solver source.

This module is intentionally dependency-free because the same exact bytes are
loaded by the trusted host publisher and baked into the isolated proposer
image.  It validates names and byte bounds only; callers remain responsible
for descriptor-confined reads, alias rejection, and durable publication.
"""

from __future__ import annotations

import re
from collections.abc import Mapping


SCHEMA = 1
REQUIRED_FILES = frozenset({"legs.py", "players.py", "solve.py"})
ALLOWED_SUFFIXES = frozenset({".py", ".json", ".txt"})
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
    return names


__all__ = [
    "ALLOWED_SUFFIXES",
    "FORBIDDEN_FILES",
    "MAX_FILES",
    "MAX_FILE_BYTES",
    "MAX_TOTAL_BYTES",
    "REQUIRED_FILES",
    "SCHEMA",
    "SourceSchemaError",
    "validate_source_payloads",
]

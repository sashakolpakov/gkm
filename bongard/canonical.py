"""Dependency-free canonical JSON and SHA-256 helpers.

The active visual pipeline uses content-addressed Python values at every
boundary.  Those primitives must not import an earlier benchmark, predicate
IR, theorem checker, or artifact schema merely to serialize a value.  Keep
this module deliberately small and one-way: higher protocol layers may import
it, while it imports only the Python standard library.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import hashlib
import json
import math
from typing import Mapping


def _validate_json_value(value: object, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite float")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path}: canonical JSON object keys must be strings")
        for key, item in value.items():
            _validate_json_value(item, f"{path}.{key}")
        return
    raise ValueError(
        f"{path}: unsupported canonical JSON value {type(value).__name__}"
    )


def canonical_json(data: object) -> bytes:
    """Return the sole canonical JSON encoding used by active protocols."""

    try:
        _validate_json_value(data)
        text = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonical-JSON encodable: {exc}") from exc
    return text.encode("utf-8")


def canonical_digest(data: object) -> str:
    """Return lowercase SHA-256 over :func:`canonical_json` bytes."""

    return hashlib.sha256(canonical_json(data)).hexdigest()


__all__ = ("canonical_digest", "canonical_json")

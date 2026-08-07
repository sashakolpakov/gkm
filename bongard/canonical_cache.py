"""Exact canonical-content memoization for frozen scientific artifacts.

The weak identity cache is deliberately outside every artifact instance and
serialized schema.  Callers provide a compact, immutable anchor made only from
scalar fields and child content digests.  When that anchor is unchanged,
repeated digest requests return the SHA-256 of the already validated canonical
bytes and repeated ``content_data`` requests decode those same bytes.  If any
anchor component changes, the exact content builder runs again, so ordinary
tamper checks retain their previous behaviour.

This changes neither canonical JSON bytes nor scientific identifiers.  It only
avoids recursively rebuilding the same immutable parent tree while a Stage-A
campaign is being finalized and cold-replayed.
"""

from __future__ import annotations

import hashlib
import json
from threading import RLock
from typing import Any, Callable, Mapping, TypeVar
import weakref

from bongard.canonical import canonical_json


_T = TypeVar("_T")
_CACHE_LOCK = RLock()
_CACHE_BY_ID: dict[
    int,
    tuple[weakref.ReferenceType[object], object, bytes, str],
] = {}


def _anchor_identity(value: object) -> tuple[object, ...]:
    """Freeze one compact cache anchor with exact Python scalar types.

    Ordinary Python equality deliberately identifies values such as ``1``,
    ``1.0``, and ``True``.  Canonical JSON does not.  Retaining the caller's
    anchor object directly would also let an in-place mutation alter both the
    old and new side of the comparison.  Cache anchors are intentionally
    limited to immutable scalar trees, so take a detached, type-tagged
    snapshot before comparing them.
    """

    if value is None:
        return ("none",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", value)
    if type(value) is float:
        return ("float", value.hex())
    if type(value) is str:
        return ("str", value)
    if type(value) is bytes:
        return ("bytes", value)
    if type(value) is tuple:
        return (
            "tuple",
            tuple(_anchor_identity(item) for item in value),
        )
    raise TypeError(
        "canonical cache anchors must contain only exact immutable "
        "None/str/bytes/bool/int/float/tuple values"
    )


def _drop_cache(instance_id: int, reference: weakref.ReferenceType[object]) -> None:
    with _CACHE_LOCK:
        cached = _CACHE_BY_ID.get(instance_id)
        if cached is not None and cached[0] is reference:
            _CACHE_BY_ID.pop(instance_id, None)


def _ensure_cache(
    instance: object,
    anchor: object,
    builder: Callable[[], Mapping[str, Any]],
) -> tuple[tuple[object, bytes, str], Mapping[str, Any] | None]:
    instance_id = id(instance)
    anchor_identity = _anchor_identity(anchor)
    with _CACHE_LOCK:
        retained = _CACHE_BY_ID.get(instance_id)
        if retained is not None:
            reference, cached_anchor, cached_payload, cached_digest = retained
            if (
                reference() is instance
                and cached_anchor == anchor_identity
                and hashlib.sha256(cached_payload).hexdigest() == cached_digest
            ):
                return (cached_anchor, cached_payload, cached_digest), None

    content = builder()
    payload = canonical_json(content)
    frozen = (
        anchor_identity,
        payload,
        hashlib.sha256(payload).hexdigest(),
    )
    try:
        reference = weakref.ref(
            instance,
            lambda item, key=instance_id: _drop_cache(key, item),
        )
    except TypeError:
        # Slotted objects without weak-reference support remain correct and
        # simply do not receive the optional optimization.
        return frozen, content
    with _CACHE_LOCK:
        _CACHE_BY_ID[instance_id] = (
            reference,
            frozen[0],
            frozen[1],
            frozen[2],
        )
    return frozen, content


def cached_content_data(
    instance: object,
    anchor: object,
    builder: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    """Return exact content, rebuilding only when the compact anchor changes."""

    cached, fresh = _ensure_cache(instance, anchor, builder)
    if fresh is not None:
        return dict(fresh)
    decoded = json.loads(cached[1])
    if not isinstance(decoded, dict):  # pragma: no cover - builders are mappings.
        raise TypeError("cached canonical artifact content must decode to an object")
    return decoded


def cached_content_digest(
    instance: object,
    anchor: object,
    builder: Callable[[], Mapping[str, Any]],
) -> str:
    """Return the digest of the exact canonical bytes for ``builder``."""

    cached, _ = _ensure_cache(instance, anchor, builder)
    return cached[2]


def cached_content_bytes(
    instance: object,
    anchor: object,
    builder: Callable[[], Mapping[str, Any]],
) -> bytes:
    """Return the exact canonical content bytes, primarily for regression tests."""

    cached, _ = _ensure_cache(instance, anchor, builder)
    return cached[1]


__all__ = [
    "cached_content_bytes",
    "cached_content_data",
    "cached_content_digest",
]

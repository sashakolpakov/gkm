"""Deterministic replay specifications for typed Bongard semantic runs.

The semantic runner deliberately keeps generated workspaces out of version
control.  That makes a small, self-contained replay document the useful unit
of durable evidence: it must carry the exact panel bytes and cone IR while
pinning the semantic-leg registry, verifier implementation, acceptance policy,
and runtime dependencies that give those bytes meaning.

This module is infrastructure only.  It does not import the semantic compiler
or verifier and it never calls an LLM.  Integration code can therefore build a
``SemanticRunSpec`` immediately before promotion, save it, load it in a cold
process, assert compatibility, materialize the panels/cones, and only then call
``verify_hypothesis``.

Design invariants
-----------------

* JSON is canonical (sorted keys, compact separators, finite numbers only).
* Binary panels are bit-packed; arbitrary numeric panels use canonical
  little-endian C-order bytes.  Both forms hash the same canonical array bytes.
* Cone hashes cover canonical JSON, not Python object identity or repr output.
* Registry hashes cover every public contract plus callable and source-module
  digests, so changing a helper in ``semantic_legs.py`` changes the fingerprint.
* A verifier policy says explicitly whether an accepted result is exact or
  tolerant.  Unexecuted checks must be named by the caller rather than silently
  represented by a zero risk.
* Saving is atomic and, by default, restricted to the ``bongard/`` tree.

The format is intentionally versioned and strict.  Missing files, NumPy,
package metadata, source code, or replay compatibility are distinct errors;
none are converted into a failed scientific result.
"""
from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import sys
import tempfile
import textwrap
from dataclasses import asdict, dataclass, is_dataclass, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


RUNSPEC_SCHEMA = "bongard.semantic-runspec/v1"
PANEL_SCHEMA = "bongard.panel-canonical/v1"
CONE_SCHEMA = "bongard.semantic-cone-canonical/v1"
REGISTRY_SCHEMA = "bongard.semantic-registry/v3"
VERIFIER_SCHEMA = "bongard.semantic-verifier/v1"
POLICY_SCHEMA = "bongard.semantic-verifier-policy/v2"

PACKED_BINARY_ENCODING = "numpy-packbits-little-base64/v1"
RAW_ENCODING = "numpy-c-order-little-base64/v1"
DIGEST_PREFIX = "sha256:"
MAX_PANEL_ELEMENTS = 16_777_216
MAX_RUNSPEC_BYTES = 64 * 1024 * 1024

BONGARD_ROOT = Path(__file__).resolve().parents[1]


class SemanticReplayError(RuntimeError):
    """Base error for RunSpec construction and replay."""


class ReplayValidationError(SemanticReplayError):
    """A document or nested payload violates the replay schema."""


class ReplayDataMissingError(SemanticReplayError):
    """Required replay bytes, source, or a RunSpec file are absent."""


class ReplayMissingDependencyError(SemanticReplayError):
    """A dependency required to construct or replay the evidence is missing."""


class ReplayProvenanceMismatchError(SemanticReplayError):
    """Current code/runtime does not match recorded replay provenance."""


class ReplayWriteBoundaryError(SemanticReplayError):
    """A save target escapes the explicitly allowed output tree."""


def _normalise_json(value: Any, path: str = "$") -> Any:
    """Return a JSON-only value with deterministic numeric normalization."""
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ReplayValidationError(f"{path}: non-finite JSON number")
        return 0.0 if number == 0.0 else number
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ReplayValidationError(
                    f"{path}: JSON object key {key!r} is not a string")
            out[key] = _normalise_json(item, f"{path}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [_normalise_json(item, f"{path}[{index}]")
                for index, item in enumerate(value)]
    raise ReplayValidationError(
        f"{path}: unsupported canonical JSON value {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode canonical UTF-8 JSON suitable for hashing and durable storage."""
    normalised = _normalise_json(value)
    try:
        text = json.dumps(
            normalised,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:  # defensive; normalization is strict
        raise ReplayValidationError(f"cannot encode canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def canonical_json_digest(value: Any) -> str:
    return DIGEST_PREFIX + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _validate_digest(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.startswith(DIGEST_PREFIX):
        raise ReplayValidationError(f"{path}: expected sha256 digest")
    suffix = value[len(DIGEST_PREFIX):]
    if len(suffix) != 64 or any(c not in "0123456789abcdef" for c in suffix):
        raise ReplayValidationError(f"{path}: malformed sha256 digest")
    return value


def _strict_int(value: Any, path: str, *, minimum: int = 0) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ReplayValidationError(f"{path}: expected integer")
    result = int(value)
    if result < minimum:
        raise ReplayValidationError(f"{path}: expected integer >= {minimum}")
    return result


def _require_v1_keys(
    value: Mapping[str, Any],
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
    path: str,
) -> None:
    """Reject extension fields until a newer schema defines their meaning."""
    required_keys = set(required)
    allowed_keys = required_keys | set(optional)
    observed_keys = set(value)
    unknown = sorted(observed_keys - allowed_keys, key=repr)
    missing = sorted(required_keys - observed_keys)
    if unknown:
        raise ReplayValidationError(
            f"{path}: unknown v1 keys: "
            + ", ".join(repr(key) for key in unknown))
    if missing:
        raise ReplayValidationError(
            f"{path}: missing required keys: {', '.join(missing)}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ReplayValidationError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _reject_nonfinite_json_constant(token: str) -> Any:
    raise ReplayValidationError(f"non-finite JSON constant {token!r}")


def _require_numpy():
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise ReplayMissingDependencyError(
            "NumPy is required to encode or materialize panel payloads") from exc
    return np


def _canonical_array(panel: Any):
    np = _require_numpy()
    array = np.asarray(panel)
    if array.ndim != 2:
        raise ReplayValidationError(
            f"panel must be a 2-D array, got shape {array.shape!r}")
    if not array.size or array.size > MAX_PANEL_ELEMENTS:
        raise ReplayValidationError(
            f"panel element count {array.size} is outside 1..{MAX_PANEL_ELEMENTS}")
    if array.dtype.kind not in "buif":
        raise ReplayValidationError(
            f"panel dtype {array.dtype} is not a plain bool/int/float dtype")
    if array.dtype.kind == "f" and not bool(np.isfinite(array).all()):
        raise ReplayValidationError("panel contains non-finite floating values")
    dtype = array.dtype
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(dtype, copy=False))
    # Packed binary payloads cannot preserve the sign bit of floating zero.
    # Canonicalize -0.0 without mutating a caller-owned contiguous array.
    if canonical.dtype.kind == "f":
        negative_zero = (canonical == 0) & np.signbit(canonical)
        if bool(negative_zero.any()):
            canonical = canonical.copy()
            canonical[negative_zero] = 0.0
    return canonical


def _panel_content_digest(array: Any) -> str:
    canonical = _canonical_array(array)
    header = {
        "schema": PANEL_SCHEMA,
        "shape": list(canonical.shape),
        "dtype": canonical.dtype.str,
    }
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(header))
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return DIGEST_PREFIX + digest.hexdigest()


@dataclass(frozen=True)
class PanelRecord:
    side: str
    index: int
    shape: tuple[int, int]
    dtype: str
    encoding: str
    data: str
    content_digest: str

    @classmethod
    def from_array(cls, panel: Any, side: str, index: int) -> "PanelRecord":
        np = _require_numpy()
        if side not in {"pos", "neg"}:
            raise ReplayValidationError(f"panel side must be pos or neg, got {side!r}")
        if not isinstance(index, Integral) or int(index) < 0:
            raise ReplayValidationError(f"panel index must be non-negative, got {index!r}")
        array = _canonical_array(panel)
        binary = bool(np.logical_or(array == 0, array == 1).all())
        if binary:
            packed = np.packbits(
                np.asarray(array != 0, dtype=np.uint8).reshape(-1),
                bitorder="little",
            ).tobytes()
            encoding = PACKED_BINARY_ENCODING
            payload = packed
        else:
            encoding = RAW_ENCODING
            payload = array.tobytes(order="C")
        return cls(
            side=side,
            index=int(index),
            shape=(int(array.shape[0]), int(array.shape[1])),
            dtype=array.dtype.str,
            encoding=encoding,
            data=base64.b64encode(payload).decode("ascii"),
            content_digest=_panel_content_digest(array),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PanelRecord":
        if not isinstance(value, Mapping):
            raise ReplayValidationError("panel record must be an object")
        _require_v1_keys(
            value,
            required=(
                "side", "index", "shape", "dtype", "encoding", "data",
                "content_digest",
            ),
            path="panel",
        )
        try:
            shape_raw = value["shape"]
            if (not isinstance(shape_raw, (list, tuple))
                    or len(shape_raw) != 2):
                raise ReplayValidationError("panel shape must have two dimensions")
            for key in ("side", "dtype", "encoding", "data", "content_digest"):
                if not isinstance(value[key], str):
                    raise ReplayValidationError(f"panel.{key}: expected string")
            record = cls(
                side=value["side"],
                index=_strict_int(value["index"], "panel.index"),
                shape=(
                    _strict_int(shape_raw[0], "panel.shape[0]", minimum=1),
                    _strict_int(shape_raw[1], "panel.shape[1]", minimum=1),
                ),
                dtype=value["dtype"],
                encoding=value["encoding"],
                data=value["data"],
                content_digest=value["content_digest"],
            )
        except KeyError as exc:
            raise ReplayValidationError(f"panel record missing {exc.args[0]!r}") from exc
        record.validate()
        return record

    def _payload_bytes(self) -> bytes:
        try:
            return base64.b64decode(self.data.encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
            raise ReplayValidationError(
                f"{self.side}_{self.index}: invalid base64 panel payload") from exc

    def decode(self):
        np = _require_numpy()
        if self.side not in {"pos", "neg"}:
            raise ReplayValidationError(f"invalid panel side {self.side!r}")
        if (not isinstance(self.index, int) or isinstance(self.index, bool)
                or self.index < 0 or len(self.shape) != 2
                or any(not isinstance(d, int) or isinstance(d, bool) or d <= 0
                       for d in self.shape)):
            raise ReplayValidationError(f"invalid panel shape/index for {self.side}_{self.index}")
        count = math.prod(self.shape)
        if count > MAX_PANEL_ELEMENTS:
            raise ReplayValidationError(f"panel {self.side}_{self.index} is too large")
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise ReplayValidationError(f"invalid panel dtype {self.dtype!r}") from exc
        if dtype.kind not in "buif" or dtype.hasobject:
            raise ReplayValidationError(f"unsupported panel dtype {self.dtype!r}")
        # NumPy reports an explicitly little-endian dtype as native ("=") on
        # little-endian hosts, so validate the serialized dtype spelling.
        if dtype.itemsize > 1 and not self.dtype.startswith("<"):
            raise ReplayValidationError(
                f"panel dtype {self.dtype!r} is not canonical little-endian")
        payload = self._payload_bytes()
        if self.encoding == PACKED_BINARY_ENCODING:
            expected = (count + 7) // 8
            if len(payload) != expected:
                raise ReplayValidationError(
                    f"{self.side}_{self.index}: packed payload has {len(payload)} "
                    f"bytes, expected {expected}")
            unpacked = np.unpackbits(
                np.frombuffer(payload, dtype=np.uint8), bitorder="little")
            if bool(unpacked[count:].any()):
                raise ReplayValidationError(
                    f"{self.side}_{self.index}: non-zero unused packed bits")
            array = unpacked[:count].astype(dtype, copy=False).reshape(self.shape)
        elif self.encoding == RAW_ENCODING:
            expected = count * dtype.itemsize
            if len(payload) != expected:
                raise ReplayValidationError(
                    f"{self.side}_{self.index}: raw payload has {len(payload)} "
                    f"bytes, expected {expected}")
            array = np.frombuffer(payload, dtype=dtype).copy().reshape(self.shape)
        else:
            raise ReplayValidationError(
                f"{self.side}_{self.index}: unsupported encoding {self.encoding!r}")
        return np.ascontiguousarray(array)

    def validate(self) -> None:
        _validate_digest(self.content_digest, "panel.content_digest")
        array = self.decode()
        observed = _panel_content_digest(array)
        if observed != self.content_digest:
            raise ReplayValidationError(
                f"{self.side}_{self.index}: panel digest mismatch "
                f"({observed} != {self.content_digest})")

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "index": self.index,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "encoding": self.encoding,
            "content_digest": self.content_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.manifest_entry()
        value["data"] = self.data
        return value


def panel_set_digest(records: Sequence[PanelRecord]) -> str:
    manifest = [record.manifest_entry() for record in records]
    return canonical_json_digest({"schema": PANEL_SCHEMA, "panels": manifest})


def panel_records_from_problem(problem: Any) -> tuple[PanelRecord, ...]:
    try:
        positives = tuple(problem.pos)
        negatives = tuple(problem.neg)
    except (AttributeError, TypeError) as exc:
        raise ReplayValidationError(
            "problem must expose iterable .pos and .neg panel collections") from exc
    if not positives or not negatives:
        raise ReplayValidationError("problem must contain positive and negative panels")
    records = [PanelRecord.from_array(panel, "pos", index)
               for index, panel in enumerate(positives)]
    records.extend(PanelRecord.from_array(panel, "neg", index)
                   for index, panel in enumerate(negatives))
    return tuple(records)


def _cone_payload(cone: Any) -> dict[str, Any]:
    if hasattr(cone, "to_dict") and callable(cone.to_dict):
        cone = cone.to_dict()
    if not isinstance(cone, Mapping):
        raise ReplayValidationError("semantic cone must be a mapping or expose to_dict()")
    value = _normalise_json(cone, "$.cone")
    if not isinstance(value, dict):  # for type checkers; Mapping normalizes to dict
        raise ReplayValidationError("semantic cone did not normalize to an object")
    return value


def semantic_cone_digest(cone: Any) -> str:
    return canonical_json_digest({"schema": CONE_SCHEMA, "cone": _cone_payload(cone)})


@dataclass(frozen=True)
class ConeRecord:
    cone_id: str
    cone: Mapping[str, Any]
    cone_digest: str
    expected_verification: Mapping[str, Any] | None = None

    @classmethod
    def from_cone(
        cls,
        cone: Any,
        expected_verification: Mapping[str, Any] | None = None,
        cone_id: str | None = None,
    ) -> "ConeRecord":
        payload = _cone_payload(cone)
        digest = semantic_cone_digest(payload)
        identifier = cone_id or payload.get("hypothesis_id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ReplayValidationError(
                "semantic cone must carry a non-empty hypothesis_id/cone_id")
        expected = None
        if expected_verification is not None:
            expected_value = _normalise_json(
                expected_verification, "$.expected_verification")
            if not isinstance(expected_value, dict):
                raise ReplayValidationError("expected verification must be an object")
            expected = expected_value
        record = cls(identifier, payload, digest, expected)
        record.validate()
        return record

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConeRecord":
        if not isinstance(value, Mapping):
            raise ReplayValidationError("cone record must be an object")
        _require_v1_keys(
            value,
            required=("cone_id", "cone", "cone_digest"),
            optional=("expected_verification",),
            path="cone",
        )
        try:
            cone = _cone_payload(value["cone"])
            expected_raw = value.get("expected_verification")
            expected = None if expected_raw is None else _normalise_json(
                expected_raw, "$.expected_verification")
            if expected is not None and not isinstance(expected, dict):
                raise ReplayValidationError("expected verification must be an object")
            if not isinstance(value["cone_id"], str):
                raise ReplayValidationError("cone_id must be a string")
            if not isinstance(value["cone_digest"], str):
                raise ReplayValidationError("cone_digest must be a string")
            record = cls(
                cone_id=value["cone_id"],
                cone=cone,
                cone_digest=value["cone_digest"],
                expected_verification=expected,
            )
        except KeyError as exc:
            raise ReplayValidationError(f"cone record missing {exc.args[0]!r}") from exc
        record.validate()
        return record

    def validate(self) -> None:
        if not isinstance(self.cone_id, str) or not self.cone_id.strip():
            raise ReplayValidationError("cone_id must be a non-empty string")
        hypothesis_id = self.cone.get("hypothesis_id")
        if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
            raise ReplayValidationError(
                f"cone {self.cone_id!r} has no non-empty hypothesis_id")
        if hypothesis_id != self.cone_id:
            raise ReplayValidationError(
                f"cone_id {self.cone_id!r} does not match cone hypothesis_id "
                f"{hypothesis_id!r}")
        _validate_digest(self.cone_digest, f"cone[{self.cone_id}].cone_digest")
        observed = semantic_cone_digest(self.cone)
        if observed != self.cone_digest:
            raise ReplayValidationError(
                f"cone {self.cone_id!r} digest mismatch "
                f"({observed} != {self.cone_digest})")
        if self.expected_verification is not None:
            _normalise_json(self.expected_verification, "$.expected_verification")
            expected_id = self.expected_verification.get("hypothesis_id")
            if expected_id is not None and expected_id != self.cone_id:
                raise ReplayValidationError(
                    f"expected verification hypothesis_id {expected_id!r} does not "
                    f"match cone_id {self.cone_id!r}")

    def manifest_entry(self) -> dict[str, Any]:
        return {"cone_id": self.cone_id, "cone_digest": self.cone_digest}

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "cone_id": self.cone_id,
            "cone": copy.deepcopy(dict(self.cone)),
            "cone_digest": self.cone_digest,
        }
        if self.expected_verification is not None:
            value["expected_verification"] = copy.deepcopy(
                dict(self.expected_verification))
        return value


def cone_set_digest(records: Sequence[ConeRecord]) -> str:
    manifest = [record.manifest_entry() for record in records]
    return canonical_json_digest({"schema": CONE_SCHEMA, "cones": manifest})


def _normalised_source(obj: Any) -> str | None:
    try:
        source = inspect.getsource(obj)
    except (OSError, TypeError):
        return None
    source = textwrap.dedent(source).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in source.split("\n")]
    return "\n".join(lines).strip() + "\n"


def callable_fingerprint(
    fn: Callable[..., Any],
    *,
    require_source: bool = True,
) -> dict[str, Any]:
    """Fingerprint a callable and its whole source module.

    The module digest is load-bearing: hashing only the registered wrapper
    would miss changes in private geometry helpers called by that wrapper.
    """
    if not callable(fn):
        raise ReplayValidationError(f"expected callable, got {type(fn).__name__}")
    module_name = getattr(fn, "__module__", type(fn).__module__)
    qualname = getattr(fn, "__qualname__", type(fn).__qualname__)
    source = _normalised_source(fn)
    module = inspect.getmodule(fn)
    module_source = _normalised_source(module) if module is not None else None
    if require_source and (source is None or module_source is None):
        missing = "callable" if source is None else "module"
        raise ReplayDataMissingError(
            f"cannot capture {missing} source for {module_name}.{qualname}")
    return {
        "module": str(module_name),
        "qualname": str(qualname),
        "source_digest": (
            DIGEST_PREFIX + hashlib.sha256(source.encode("utf-8")).hexdigest()
            if source is not None else None
        ),
        "module_source_digest": (
            DIGEST_PREFIX + hashlib.sha256(module_source.encode("utf-8")).hexdigest()
            if module_source is not None else None
        ),
        "source_complete": source is not None and module_source is not None,
    }


def source_object_fingerprint(
    obj: Any,
    *,
    require_source: bool = True,
) -> dict[str, Any]:
    """Fingerprint a caller-supplied implementation module or source object.

    Verifier behavior is spread across the compiler, IR validation,
    requirements, cofibration, selection, and witness modules.  A digest of
    ``verify_hypothesis``'s defining module alone cannot detect changes in
    those imported implementations.  Callers supply those load-bearing
    objects explicitly so the provenance boundary remains auditable rather
    than relying on fragile import-graph guessing.

    For an instance, the source of its class and defining module is pinned;
    mutable instance state is deliberately not treated as source provenance.
    """
    if inspect.ismodule(obj):
        target = obj
        module = obj
        kind = "module"
        module_name = getattr(obj, "__name__", "")
        qualname = module_name
    else:
        target = obj if (inspect.isclass(obj) or callable(obj)) else type(obj)
        module = inspect.getmodule(target)
        kind = (
            "class" if inspect.isclass(target)
            else "callable" if callable(target)
            else "object"
        )
        if target is not obj:
            kind = "instance-class"
        module_name = getattr(target, "__module__", type(target).__module__)
        qualname = getattr(target, "__qualname__", type(target).__qualname__)

    source = _normalised_source(target)
    module_source = _normalised_source(module) if module is not None else None
    if require_source and (source is None or module_source is None):
        missing = "object" if source is None else "module"
        raise ReplayDataMissingError(
            f"cannot capture {missing} source for {module_name}.{qualname}")
    return {
        "kind": kind,
        "module": str(module_name),
        "qualname": str(qualname),
        "source_digest": (
            DIGEST_PREFIX + hashlib.sha256(source.encode("utf-8")).hexdigest()
            if source is not None else None
        ),
        "module_source_digest": (
            DIGEST_PREFIX + hashlib.sha256(module_source.encode("utf-8")).hexdigest()
            if module_source is not None else None
        ),
        "source_complete": source is not None and module_source is not None,
    }


def capture_source_fingerprints(
    sources: Mapping[str, Any] | None,
    *,
    require_source: bool = True,
) -> dict[str, dict[str, Any]]:
    """Return a name-stable manifest for verifier-supporting source objects."""
    if sources is None:
        return {}
    if not isinstance(sources, Mapping):
        raise ReplayValidationError("verifier_sources must be a name/object mapping")
    names = list(sources)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ReplayValidationError(
            "verifier_sources keys must be non-empty strings")
    manifest: dict[str, dict[str, Any]] = {}
    for name in sorted(names):
        manifest[name] = source_object_fingerprint(
            sources[name], require_source=require_source)
    return manifest


def _resolved_verifier_sources(
    verifier_sources: Mapping[str, Any] | None,
    verifier_related_sources: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Resolve the bootstrap spelling while rejecting ambiguous manifests."""
    if verifier_sources is not None and verifier_related_sources is not None:
        raise ReplayValidationError(
            "pass only one of verifier_sources or verifier_related_sources")
    return (verifier_related_sources
            if verifier_related_sources is not None else verifier_sources)


def _validate_source_fingerprint(value: Any, path: str) -> None:
    if not isinstance(value, Mapping):
        raise ReplayValidationError(f"{path}: source fingerprint must be an object")
    for key in ("kind", "module", "qualname"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ReplayValidationError(f"{path}.{key}: expected non-empty string")
    if not isinstance(value.get("source_complete"), bool):
        raise ReplayValidationError(f"{path}.source_complete: expected boolean")
    for key in ("source_digest", "module_source_digest"):
        digest = value.get(key)
        if digest is not None:
            _validate_digest(digest, f"{path}.{key}")
        elif value["source_complete"]:
            raise ReplayValidationError(
                f"{path}.{key}: complete source fingerprint has no digest")


def _validate_source_manifest(value: Any, path: str) -> None:
    if not isinstance(value, Mapping):
        raise ReplayValidationError(f"{path}: expected source-fingerprint object")
    for name, fingerprint in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ReplayValidationError(f"{path}: source names must be non-empty strings")
        _validate_source_fingerprint(fingerprint, f"{path}.{name}")


def _contract_manifest(contract: Any, require_source: bool) -> dict[str, Any]:
    required = ("name", "domain", "codomain", "implementation", "complexity",
                "invariances", "equivariances", "failure_modes",
                "indeterminate_modes", "version", "proxy_for",
                "measurement_kind", "proxy_directions")
    missing = [name for name in required if not hasattr(contract, name)]
    if missing:
        raise ReplayValidationError(
            f"leg contract is missing fields: {', '.join(missing)}")
    complexity = _strict_int(contract.complexity, f"leg[{contract.name}].complexity")
    return {
        "name": str(contract.name),
        # Domain order is part of a typed multicategory arrow; do not sort it.
        "domain": list(str(value) for value in contract.domain),
        "codomain": str(contract.codomain),
        "complexity": complexity,
        "invariances": sorted(str(value) for value in contract.invariances),
        "equivariances": sorted(str(value) for value in contract.equivariances),
        "failure_modes": list(str(value) for value in contract.failure_modes),
        "indeterminate_modes": [
            str(value) for value in contract.indeterminate_modes],
        "version": str(contract.version),
        "proxy_for": list(str(value) for value in contract.proxy_for),
        "measurement_kind": contract.measurement_kind,
        "proxy_directions": [
            [str(term), str(direction)]
            for term, direction in contract.proxy_directions
        ],
        "implementation": callable_fingerprint(
            contract.implementation, require_source=require_source),
    }


def registry_fingerprint(
    registry: Any,
    *,
    require_implementation_source: bool = True,
    include_manifest: bool = False,
) -> dict[str, Any]:
    try:
        contracts = list(registry.contracts())
    except (AttributeError, TypeError) as exc:
        raise ReplayValidationError(
            "semantic registry must expose contracts()") from exc
    manifests = [_contract_manifest(contract, require_implementation_source)
                 for contract in contracts]
    manifests.sort(key=lambda value: value["name"])
    names = [value["name"] for value in manifests]
    if len(names) != len(set(names)):
        raise ReplayValidationError("semantic registry contains duplicate leg names")
    if not names:
        raise ReplayValidationError("semantic registry contains no leg contracts")
    payload = {"schema": REGISTRY_SCHEMA, "contracts": manifests}
    fingerprint: dict[str, Any] = {
        "schema": REGISTRY_SCHEMA,
        "digest": canonical_json_digest(payload),
        "contract_count": len(manifests),
        "contract_names": names,
        "source_complete": all(
            value["implementation"]["source_complete"] for value in manifests),
    }
    if include_manifest:
        fingerprint["contracts"] = manifests
    return fingerprint


def capture_dependency_versions(
    distributions: Iterable[str],
    *,
    strict: bool = True,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for name in sorted(set(str(item) for item in distributions)):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            if strict:
                raise ReplayMissingDependencyError(
                    f"required distribution {name!r} is not installed") from exc
            records.append({"distribution": name, "status": "missing"})
        else:
            records.append({
                "distribution": name,
                "status": "present",
                "version": str(version),
            })
    return records


def capture_verifier_provenance(
    verifier: Callable[..., Any],
    *,
    dependency_distributions: Iterable[str] = ("numpy", "scipy", "scikit-image"),
    verifier_sources: Mapping[str, Any] | None = None,
    verifier_related_sources: Mapping[str, Any] | None = None,
    require_source: bool = True,
    strict_dependencies: bool = True,
) -> dict[str, Any]:
    return {
        "schema": VERIFIER_SCHEMA,
        "implementation": callable_fingerprint(verifier, require_source=require_source),
        "related_sources": capture_source_fingerprints(
            _resolved_verifier_sources(
                verifier_sources, verifier_related_sources),
            require_source=require_source,
        ),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": getattr(sys.implementation, "cache_tag", ""),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
        },
        "dependencies": capture_dependency_versions(
            dependency_distributions, strict=strict_dependencies),
    }


@dataclass(frozen=True)
class VerifierPolicy:
    """Acceptance semantics that must accompany every replay verdict."""

    max_support_errors: int = 0
    max_loo_errors: int = 0
    max_rotated_loo_errors: int = 0
    require_zero_predicate_errors: bool = True
    require_zero_indeterminate_evaluations: bool = True
    require_zero_naturality_errors: bool = True
    require_zero_cofibration_errors: bool = True
    require_zero_unchecked_morphisms: bool = True
    require_semantic_quality: bool = True
    threshold_policy: str = (
        "fixed-semantic-predicate-reused;"
        "relative-threshold-fit-inside-each-loo-fold"
    )
    # The current verifier records fold thresholds but does not gate admission
    # on overlap.  Default provenance must describe that behavior truthfully.
    require_threshold_overlap: bool = False
    max_fold_threshold_span: float | None = None
    transform_policy: str = "recorded-verifier-battery"
    unexecuted_checks: tuple[str, ...] = ()

    @property
    def acceptance_mode(self) -> str:
        exact_gates = (
            self.require_zero_predicate_errors,
            self.require_zero_indeterminate_evaluations,
            self.require_zero_naturality_errors,
            self.require_zero_cofibration_errors,
            self.require_zero_unchecked_morphisms,
            self.require_semantic_quality,
        )
        return (
            "exact"
            if (self.max_support_errors == 0
                and self.max_loo_errors == 0
                and self.max_rotated_loo_errors == 0
                and all(exact_gates))
            else "tolerant"
        )

    def validate(self) -> None:
        for name in (
            "max_support_errors",
            "max_loo_errors",
            "max_rotated_loo_errors",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ReplayValidationError(f"policy.{name} must be a non-negative int")
        for name in (
            "require_zero_predicate_errors",
            "require_zero_indeterminate_evaluations",
            "require_zero_naturality_errors",
            "require_zero_cofibration_errors",
            "require_zero_unchecked_morphisms",
            "require_semantic_quality",
            "require_threshold_overlap",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ReplayValidationError(f"policy.{name} must be boolean")
        if not self.threshold_policy.strip() or not self.transform_policy.strip():
            raise ReplayValidationError("threshold and transform policies must be named")
        if not self.require_zero_indeterminate_evaluations:
            raise ReplayValidationError(
                "policy cannot admit indeterminate semantic evaluations")
        if self.max_fold_threshold_span is not None:
            if (not isinstance(self.max_fold_threshold_span, Real)
                    or isinstance(self.max_fold_threshold_span, bool)
                    or not math.isfinite(float(self.max_fold_threshold_span))
                    or float(self.max_fold_threshold_span) < 0.0):
                raise ReplayValidationError(
                    "policy.max_fold_threshold_span must be finite and non-negative")
            raise ReplayValidationError(
                "policy.max_fold_threshold_span is not an admission gate in the "
                "current verifier")
        if self.require_threshold_overlap:
            raise ReplayValidationError(
                "policy.require_threshold_overlap is not enforced by the current "
                "verifier")
        if any(not isinstance(item, str) or not item.strip()
               for item in self.unexecuted_checks):
            raise ReplayValidationError(
                "policy.unexecuted_checks must contain non-empty names")
        if len(self.unexecuted_checks) != len(set(self.unexecuted_checks)):
            raise ReplayValidationError("policy.unexecuted_checks contains duplicates")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": POLICY_SCHEMA,
            "acceptance_mode": self.acceptance_mode,
            "max_support_errors": self.max_support_errors,
            "max_loo_errors": self.max_loo_errors,
            "max_rotated_loo_errors": self.max_rotated_loo_errors,
            "require_zero_predicate_errors": self.require_zero_predicate_errors,
            "require_zero_indeterminate_evaluations": (
                self.require_zero_indeterminate_evaluations),
            "require_zero_naturality_errors": self.require_zero_naturality_errors,
            "require_zero_cofibration_errors": self.require_zero_cofibration_errors,
            "require_zero_unchecked_morphisms": (
                self.require_zero_unchecked_morphisms),
            "require_semantic_quality": self.require_semantic_quality,
            "threshold_policy": self.threshold_policy,
            "require_threshold_overlap": self.require_threshold_overlap,
            "max_fold_threshold_span": self.max_fold_threshold_span,
            "transform_policy": self.transform_policy,
            "unexecuted_checks": sorted(self.unexecuted_checks),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VerifierPolicy":
        if not isinstance(value, Mapping):
            raise ReplayValidationError("verifier policy must be an object")
        _require_v1_keys(
            value,
            required=(
                "schema",
                "acceptance_mode",
                "max_support_errors",
                "max_loo_errors",
                "require_zero_predicate_errors",
                "require_zero_indeterminate_evaluations",
                "require_zero_naturality_errors",
                "require_zero_cofibration_errors",
                "require_semantic_quality",
                "threshold_policy",
                "require_threshold_overlap",
                "max_fold_threshold_span",
                "transform_policy",
                "unexecuted_checks",
            ),
            # These fields were added during the v1 bootstrap.  Loading an
            # earlier v1 document assigns fail-closed defaults below.
            optional=(
                "max_rotated_loo_errors",
                "require_zero_unchecked_morphisms",
            ),
            path="verifier.policy",
        )
        if value.get("schema") != POLICY_SCHEMA:
            raise ReplayValidationError(
                f"unsupported verifier policy schema {value.get('schema')!r}")
        for key in ("threshold_policy", "transform_policy"):
            if not isinstance(value.get(key), str):
                raise ReplayValidationError(f"policy.{key} must be a string")
        unexecuted = value.get("unexecuted_checks", ())
        if (not isinstance(unexecuted, (list, tuple))
                or any(not isinstance(item, str) for item in unexecuted)):
            raise ReplayValidationError(
                "policy.unexecuted_checks must be a string list")
        try:
            policy = cls(
                max_support_errors=value["max_support_errors"],
                max_loo_errors=value["max_loo_errors"],
                max_rotated_loo_errors=value.get("max_rotated_loo_errors", 0),
                require_zero_predicate_errors=value["require_zero_predicate_errors"],
                require_zero_indeterminate_evaluations=value[
                    "require_zero_indeterminate_evaluations"],
                require_zero_naturality_errors=value["require_zero_naturality_errors"],
                require_zero_cofibration_errors=value["require_zero_cofibration_errors"],
                require_zero_unchecked_morphisms=value.get(
                    "require_zero_unchecked_morphisms", True),
                require_semantic_quality=value["require_semantic_quality"],
                threshold_policy=value["threshold_policy"],
                require_threshold_overlap=value["require_threshold_overlap"],
                max_fold_threshold_span=value.get("max_fold_threshold_span"),
                transform_policy=value["transform_policy"],
                unexecuted_checks=tuple(unexecuted),
            )
        except KeyError as exc:
            raise ReplayValidationError(
                f"verifier policy missing {exc.args[0]!r}") from exc
        policy.validate()
        if value.get("acceptance_mode") != policy.acceptance_mode:
            raise ReplayValidationError(
                "verifier acceptance_mode does not match its error thresholds")
        return policy


def _ordered_panel_records(records: Sequence[PanelRecord]) -> tuple[PanelRecord, ...]:
    order = {"pos": 0, "neg": 1}
    result = tuple(sorted(records, key=lambda item: (order.get(item.side, 99), item.index)))
    seen: set[tuple[str, int]] = set()
    for record in result:
        record.validate()
        key = (record.side, record.index)
        if key in seen:
            raise ReplayValidationError(f"duplicate panel slot {record.side}_{record.index}")
        seen.add(key)
    for side in ("pos", "neg"):
        indexes = [record.index for record in result if record.side == side]
        if not indexes:
            raise ReplayValidationError(f"RunSpec has no {side} panels")
        if indexes != list(range(len(indexes))):
            raise ReplayValidationError(
                f"{side} panel indexes must be contiguous from zero, got {indexes}")
    return result


def _ordered_cone_records(records: Sequence[ConeRecord]) -> tuple[ConeRecord, ...]:
    result = tuple(sorted(records, key=lambda item: (item.cone_id, item.cone_digest)))
    if not result:
        raise ReplayValidationError("RunSpec must contain at least one semantic cone")
    seen: set[str] = set()
    for record in result:
        record.validate()
        if record.cone_id in seen:
            raise ReplayValidationError(f"duplicate cone_id {record.cone_id!r}")
        seen.add(record.cone_id)
    return result


@dataclass(frozen=True)
class SemanticRunSpec:
    schema: str
    problem: Mapping[str, Any]
    panels: tuple[PanelRecord, ...]
    panel_set_digest: str
    cones: tuple[ConeRecord, ...]
    cone_set_digest: str
    registry: Mapping[str, Any]
    verifier: Mapping[str, Any]
    provenance: Mapping[str, Any]
    spec_digest: str

    def body_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "problem": copy.deepcopy(dict(self.problem)),
            "panels": [record.to_dict() for record in self.panels],
            "panel_set_digest": self.panel_set_digest,
            "cones": [record.to_dict() for record in self.cones],
            "cone_set_digest": self.cone_set_digest,
            "registry": copy.deepcopy(dict(self.registry)),
            "verifier": copy.deepcopy(dict(self.verifier)),
            "provenance": copy.deepcopy(dict(self.provenance)),
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.body_dict()
        value["spec_digest"] = self.spec_digest
        return value

    def sealed(self) -> "SemanticRunSpec":
        return replace(self, spec_digest=canonical_json_digest(self.body_dict()))

    def validate(self) -> None:
        if self.schema != RUNSPEC_SCHEMA:
            raise ReplayValidationError(f"unsupported RunSpec schema {self.schema!r}")
        problem = _normalise_json(self.problem, "$.problem")
        if not isinstance(problem, dict):
            raise ReplayValidationError("RunSpec problem must be an object")
        for key in ("opaque_id", "problem_id", "category"):
            if key not in problem or not isinstance(problem[key], str):
                raise ReplayValidationError(f"RunSpec problem.{key} must be a string")
        ordered_panels = _ordered_panel_records(self.panels)
        if ordered_panels != self.panels:
            raise ReplayValidationError("RunSpec panels are not in canonical order")
        observed_panels = panel_set_digest(self.panels)
        _validate_digest(self.panel_set_digest, "panel_set_digest")
        if observed_panels != self.panel_set_digest:
            raise ReplayValidationError("RunSpec panel_set_digest mismatch")
        ordered_cones = _ordered_cone_records(self.cones)
        if ordered_cones != self.cones:
            raise ReplayValidationError("RunSpec cones are not in canonical order")
        observed_cones = cone_set_digest(self.cones)
        _validate_digest(self.cone_set_digest, "cone_set_digest")
        if observed_cones != self.cone_set_digest:
            raise ReplayValidationError("RunSpec cone_set_digest mismatch")

        registry = _normalise_json(self.registry, "$.registry")
        if not isinstance(registry, dict) or registry.get("schema") != REGISTRY_SCHEMA:
            raise ReplayValidationError("RunSpec registry fingerprint is malformed")
        _validate_digest(registry.get("digest"), "registry.digest")
        if (not isinstance(registry.get("contract_count"), int)
                or isinstance(registry.get("contract_count"), bool)
                or registry["contract_count"] < 1):
            raise ReplayValidationError("registry.contract_count must be an int")
        names = registry.get("contract_names")
        if (not isinstance(names, list)
                or any(not isinstance(name, str) or not name for name in names)
                or names != sorted(names)
                or len(names) != registry["contract_count"]):
            raise ReplayValidationError(
                "registry.contract_names must be sorted and match contract_count")

        verifier = _normalise_json(self.verifier, "$.verifier")
        if not isinstance(verifier, dict):
            raise ReplayValidationError("RunSpec verifier must be an object")
        VerifierPolicy.from_dict(verifier.get("policy", {}))
        verifier_prov = verifier.get("provenance")
        if (not isinstance(verifier_prov, dict)
                or verifier_prov.get("schema") != VERIFIER_SCHEMA):
            raise ReplayValidationError("RunSpec verifier provenance is malformed")
        implementation = verifier_prov.get("implementation")
        if not isinstance(implementation, dict):
            raise ReplayValidationError("verifier implementation fingerprint is missing")
        for key in ("module", "qualname"):
            if not isinstance(implementation.get(key), str) or not implementation[key]:
                raise ReplayValidationError(
                    f"verifier.implementation.{key} must be a non-empty string")
        if not isinstance(implementation.get("source_complete"), bool):
            raise ReplayValidationError(
                "verifier.implementation.source_complete must be boolean")
        for key in ("source_digest", "module_source_digest"):
            if implementation.get(key) is not None:
                _validate_digest(implementation[key], f"verifier.implementation.{key}")
            elif implementation["source_complete"]:
                raise ReplayValidationError(
                    f"verifier.implementation.{key} is missing")
        _validate_source_manifest(
            verifier_prov.get("related_sources", {}),
            "verifier.related_sources",
        )
        python_record = verifier_prov.get("python")
        if not isinstance(python_record, dict):
            raise ReplayValidationError("verifier.python provenance is missing")
        for key in ("implementation", "version", "cache_tag"):
            if not isinstance(python_record.get(key), str):
                raise ReplayValidationError(f"verifier.python.{key} must be a string")
        platform_record = verifier_prov.get("platform")
        if not isinstance(platform_record, dict):
            raise ReplayValidationError("verifier.platform provenance is missing")
        for key in ("system", "machine", "byteorder"):
            if not isinstance(platform_record.get(key), str):
                raise ReplayValidationError(f"verifier.platform.{key} must be a string")
        if platform_record["byteorder"] not in {"little", "big"}:
            raise ReplayValidationError(
                "verifier.platform.byteorder must be little or big")
        dependencies = verifier_prov.get("dependencies")
        if not isinstance(dependencies, list):
            raise ReplayValidationError("verifier.dependencies must be a list")
        for index, dependency in enumerate(dependencies):
            path = f"verifier.dependencies[{index}]"
            if not isinstance(dependency, dict):
                raise ReplayValidationError(f"{path} must be an object")
            if (not isinstance(dependency.get("distribution"), str)
                    or not dependency["distribution"]):
                raise ReplayValidationError(
                    f"{path}.distribution must be a non-empty string")
            if dependency.get("status") not in {"present", "missing"}:
                raise ReplayValidationError(
                    f"{path}.status must be present or missing")
            if (dependency["status"] == "present"
                    and (not isinstance(dependency.get("version"), str)
                         or not dependency["version"])):
                raise ReplayValidationError(
                    f"{path}.version must be recorded for a present dependency")
        _normalise_json(self.provenance, "$.provenance")

        _validate_digest(self.spec_digest, "spec_digest")
        observed_spec = canonical_json_digest(self.body_dict())
        if observed_spec != self.spec_digest:
            raise ReplayValidationError(
                f"RunSpec digest mismatch ({observed_spec} != {self.spec_digest})")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, validate: bool = True) -> "SemanticRunSpec":
        if not isinstance(value, Mapping):
            raise ReplayValidationError("RunSpec document must be an object")
        _require_v1_keys(
            value,
            required=(
                "schema",
                "problem",
                "panels",
                "panel_set_digest",
                "cones",
                "cone_set_digest",
                "registry",
                "verifier",
                "provenance",
                "spec_digest",
            ),
            path="RunSpec",
        )
        try:
            problem = _normalise_json(value["problem"], "$.problem")
            registry = _normalise_json(value["registry"], "$.registry")
            verifier = _normalise_json(value["verifier"], "$.verifier")
            provenance = _normalise_json(value["provenance"], "$.provenance")
            spec = cls(
                schema=str(value["schema"]),
                problem=problem,
                panels=tuple(PanelRecord.from_dict(item) for item in value["panels"]),
                panel_set_digest=str(value["panel_set_digest"]),
                cones=tuple(ConeRecord.from_dict(item) for item in value["cones"]),
                cone_set_digest=str(value["cone_set_digest"]),
                registry=registry,
                verifier=verifier,
                provenance=provenance,
                spec_digest=str(value["spec_digest"]),
            )
        except KeyError as exc:
            raise ReplayValidationError(f"RunSpec missing {exc.args[0]!r}") from exc
        except TypeError as exc:
            raise ReplayValidationError(f"malformed RunSpec collection: {exc}") from exc
        if validate:
            spec.validate()
        return spec


def build_runspec(
    *,
    opaque_id: str,
    problem: Any,
    cones: Sequence[Any],
    registry: Any,
    verifier: Callable[..., Any],
    policy: VerifierPolicy | None = None,
    expected_verifications: Mapping[str, Mapping[str, Any]] | None = None,
    provenance: Mapping[str, Any] | None = None,
    dependency_distributions: Iterable[str] = ("numpy", "scipy", "scikit-image"),
    verifier_sources: Mapping[str, Any] | None = None,
    verifier_related_sources: Mapping[str, Any] | None = None,
    include_ground_truth: bool = False,
    require_source: bool = True,
    strict_dependencies: bool = True,
) -> SemanticRunSpec:
    """Build and fully validate a deterministic cold-replay document."""
    if not isinstance(opaque_id, str) or not opaque_id.strip():
        raise ReplayValidationError("opaque_id must be a non-empty string")
    problem_meta: dict[str, Any] = {
        "opaque_id": opaque_id,
        # Do not leak a dataset/source identifier into a proposer-visible run
        # artifact.  The opaque ID is the replay identity by default.
        "problem_id": opaque_id,
        "category": str(getattr(problem, "category", "")),
    }
    if include_ground_truth:
        problem_meta["source_problem_id"] = str(
            getattr(problem, "problem_id", ""))
        problem_meta["concept"] = str(getattr(problem, "concept", ""))

    panel_records = _ordered_panel_records(panel_records_from_problem(problem))
    expected = expected_verifications or {}
    cone_records: list[ConeRecord] = []
    for cone in cones:
        payload = _cone_payload(cone)
        identifier = payload.get("hypothesis_id")
        expected_value = expected.get(identifier) if isinstance(identifier, str) else None
        cone_records.append(ConeRecord.from_cone(
            payload,
            expected_verification=expected_value,
            cone_id=identifier if isinstance(identifier, str) else None,
        ))
    ordered_cones = _ordered_cone_records(cone_records)
    unknown_expected = sorted(set(expected) - {record.cone_id for record in ordered_cones})
    if unknown_expected:
        raise ReplayValidationError(
            "expected verifications reference unknown cones: "
            + ", ".join(unknown_expected))
    active_policy = policy or VerifierPolicy()
    active_policy.validate()
    registry_record = registry_fingerprint(
        registry, require_implementation_source=require_source)
    verifier_record = {
        "policy": active_policy.to_dict(),
        "provenance": capture_verifier_provenance(
            verifier,
            dependency_distributions=dependency_distributions,
            verifier_sources=verifier_sources,
            verifier_related_sources=verifier_related_sources,
            require_source=require_source,
            strict_dependencies=strict_dependencies,
        ),
    }
    provenance_record = _normalise_json(provenance or {}, "$.provenance")
    spec = SemanticRunSpec(
        schema=RUNSPEC_SCHEMA,
        problem=_normalise_json(problem_meta),
        panels=panel_records,
        panel_set_digest=panel_set_digest(panel_records),
        cones=ordered_cones,
        cone_set_digest=cone_set_digest(ordered_cones),
        registry=registry_record,
        verifier=verifier_record,
        provenance=provenance_record,
        spec_digest=DIGEST_PREFIX + "0" * 64,
    ).sealed()
    spec.validate()
    return spec


def _resolved_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def save_runspec(
    path: str | os.PathLike[str],
    spec: SemanticRunSpec,
    *,
    allowed_root: str | os.PathLike[str] | None = None,
    create_parents: bool = False,
) -> Path:
    """Atomically save canonical JSON, restricted to ``bongard/`` by default."""
    spec.validate()
    target = Path(path)
    root = Path(allowed_root) if allowed_root is not None else BONGARD_ROOT
    resolved_target = target.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    if (resolved_target == resolved_root
            or not _resolved_within(target, root)
            or not _resolved_within(target.parent, root)):
        raise ReplayWriteBoundaryError(
            f"RunSpec target {target} escapes or is not a file strictly inside "
            f"allowed root {root}")
    parent = target.parent
    if create_parents:
        parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir():
        raise ReplayDataMissingError(f"RunSpec parent directory does not exist: {parent}")
    payload = canonical_json_bytes(spec.to_dict()) + b"\n"
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=parent, prefix=f".{target.name}.", suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
    return target


def load_runspec(path: str | os.PathLike[str]) -> SemanticRunSpec:
    source = Path(path)
    if not source.is_file():
        raise ReplayDataMissingError(f"RunSpec file does not exist: {source}")
    try:
        size = source.stat().st_size
        if size > MAX_RUNSPEC_BYTES:
            raise ReplayValidationError(
                f"RunSpec is {size} bytes; maximum is {MAX_RUNSPEC_BYTES}")
        raw = source.read_bytes()
    except OSError as exc:
        raise ReplayDataMissingError(f"cannot read RunSpec {source}: {exc}") from exc
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayValidationError(f"RunSpec is not valid UTF-8 JSON: {exc}") from exc
    return SemanticRunSpec.from_dict(value, validate=True)


@dataclass(frozen=True)
class ColdReplayInputs:
    problem: Mapping[str, Any]
    positive_panels: tuple[Any, ...]
    negative_panels: tuple[Any, ...]
    cones: tuple[Mapping[str, Any], ...]
    expected_verifications: Mapping[str, Mapping[str, Any]]
    policy: VerifierPolicy
    registry: Mapping[str, Any]
    verifier_provenance: Mapping[str, Any]
    provenance: Mapping[str, Any]


def materialize_cold_inputs(spec: SemanticRunSpec) -> ColdReplayInputs:
    """Decode only data carried by the RunSpec; no dataset access is performed."""
    spec.validate()
    positives = tuple(record.decode() for record in spec.panels if record.side == "pos")
    negatives = tuple(record.decode() for record in spec.panels if record.side == "neg")
    cones = tuple(copy.deepcopy(dict(record.cone)) for record in spec.cones)
    expected = {
        record.cone_id: copy.deepcopy(dict(record.expected_verification))
        for record in spec.cones if record.expected_verification is not None
    }
    verifier = dict(spec.verifier)
    return ColdReplayInputs(
        problem=copy.deepcopy(dict(spec.problem)),
        positive_panels=positives,
        negative_panels=negatives,
        cones=cones,
        expected_verifications=expected,
        policy=VerifierPolicy.from_dict(verifier["policy"]),
        registry=copy.deepcopy(dict(spec.registry)),
        verifier_provenance=copy.deepcopy(dict(verifier["provenance"])),
        provenance=copy.deepcopy(dict(spec.provenance)),
    )


def validate_registry_compatibility(spec: SemanticRunSpec, registry: Any) -> None:
    expected = dict(spec.registry)
    current = registry_fingerprint(
        registry,
        require_implementation_source=bool(expected.get("source_complete", True)),
    )
    if current.get("digest") != expected.get("digest"):
        raise ReplayProvenanceMismatchError(
            "semantic registry fingerprint mismatch: "
            f"recorded={expected.get('digest')} current={current.get('digest')}")


def validate_verifier_compatibility(
    spec: SemanticRunSpec,
    verifier: Callable[..., Any],
    *,
    verifier_sources: Mapping[str, Any] | None = None,
    verifier_related_sources: Mapping[str, Any] | None = None,
    strict_dependency_versions: bool = True,
    strict_python_version: bool = True,
) -> None:
    recorded = dict(spec.verifier)["provenance"]
    recorded_impl = recorded.get("implementation", {})
    current_impl = callable_fingerprint(
        verifier, require_source=bool(recorded_impl.get("source_complete", True)))
    for key in ("module", "qualname", "source_digest", "module_source_digest"):
        if current_impl.get(key) != recorded_impl.get(key):
            raise ReplayProvenanceMismatchError(
                f"verifier {key} mismatch: recorded={recorded_impl.get(key)!r} "
                f"current={current_impl.get(key)!r}")

    recorded_sources = recorded.get("related_sources", {})
    current_source_objects = _resolved_verifier_sources(
        verifier_sources, verifier_related_sources)
    if recorded_sources and current_source_objects is None:
        raise ReplayDataMissingError(
            "replay requires the verifier_sources used to create this RunSpec: "
            + ", ".join(sorted(recorded_sources)))
    require_related_source = all(
        bool(fingerprint.get("source_complete", True))
        for fingerprint in recorded_sources.values()
    )
    current_sources = capture_source_fingerprints(
        current_source_objects, require_source=require_related_source)
    recorded_names = set(recorded_sources)
    current_names = set(current_sources)
    if current_names != recorded_names:
        raise ReplayProvenanceMismatchError(
            "verifier-related source set mismatch: "
            f"recorded={sorted(recorded_names)} current={sorted(current_names)}")
    for name in sorted(recorded_names):
        recorded_source = recorded_sources[name]
        current_source = current_sources[name]
        for key in (
            "kind",
            "module",
            "qualname",
            "source_digest",
            "module_source_digest",
            "source_complete",
        ):
            if current_source.get(key) != recorded_source.get(key):
                raise ReplayProvenanceMismatchError(
                    f"verifier-related source {name!r} {key} mismatch: "
                    f"recorded={recorded_source.get(key)!r} "
                    f"current={current_source.get(key)!r}")

    python_record = recorded.get("python", {})
    current_python = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "cache_tag": getattr(sys.implementation, "cache_tag", ""),
    }
    if strict_python_version:
        for key, current_value in current_python.items():
            if python_record.get(key) != current_value:
                raise ReplayProvenanceMismatchError(
                    f"Python {key} mismatch: recorded={python_record.get(key)!r} "
                    f"current={current_value!r}")
        platform_record = recorded.get("platform", {})
        current_platform = {
            "system": platform.system(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
        }
        for key, current_value in current_platform.items():
            if platform_record.get(key) != current_value:
                raise ReplayProvenanceMismatchError(
                    f"runtime platform {key} mismatch: "
                    f"recorded={platform_record.get(key)!r} "
                    f"current={current_value!r}")

    for dependency in recorded.get("dependencies", []):
        name = dependency.get("distribution")
        if not isinstance(name, str) or not name:
            raise ReplayValidationError("recorded dependency has no distribution name")
        if dependency.get("status") != "present":
            raise ReplayMissingDependencyError(
                f"recorded replay dependency {name!r} was missing")
        try:
            current_version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ReplayMissingDependencyError(
                f"replay dependency {name!r} is not installed") from exc
        if strict_dependency_versions and current_version != dependency.get("version"):
            raise ReplayProvenanceMismatchError(
                f"dependency {name!r} version mismatch: "
                f"recorded={dependency.get('version')} current={current_version}")


def assert_replay_compatible(
    spec: SemanticRunSpec,
    *,
    registry: Any,
    verifier: Callable[..., Any],
    verifier_sources: Mapping[str, Any] | None = None,
    verifier_related_sources: Mapping[str, Any] | None = None,
    strict_dependency_versions: bool = True,
    strict_python_version: bool = True,
) -> None:
    """Fail closed unless current registry, verifier, Python, and deps match."""
    spec.validate()
    validate_registry_compatibility(spec, registry)
    validate_verifier_compatibility(
        spec,
        verifier,
        verifier_sources=verifier_sources,
        verifier_related_sources=verifier_related_sources,
        strict_dependency_versions=strict_dependency_versions,
        strict_python_version=strict_python_version,
    )


__all__ = [
    "BONGARD_ROOT",
    "CONE_SCHEMA",
    "ColdReplayInputs",
    "ConeRecord",
    "PACKED_BINARY_ENCODING",
    "PANEL_SCHEMA",
    "PanelRecord",
    "RAW_ENCODING",
    "REGISTRY_SCHEMA",
    "RUNSPEC_SCHEMA",
    "ReplayDataMissingError",
    "ReplayMissingDependencyError",
    "ReplayProvenanceMismatchError",
    "ReplayValidationError",
    "ReplayWriteBoundaryError",
    "SemanticReplayError",
    "SemanticRunSpec",
    "VerifierPolicy",
    "assert_replay_compatible",
    "build_runspec",
    "callable_fingerprint",
    "canonical_json_bytes",
    "canonical_json_digest",
    "capture_dependency_versions",
    "capture_source_fingerprints",
    "capture_verifier_provenance",
    "cone_set_digest",
    "load_runspec",
    "materialize_cold_inputs",
    "panel_records_from_problem",
    "panel_set_digest",
    "registry_fingerprint",
    "save_runspec",
    "semantic_cone_digest",
    "source_object_fingerprint",
    "validate_registry_compatibility",
    "validate_verifier_compatibility",
]

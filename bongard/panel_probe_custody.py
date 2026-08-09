"""Shared bounded custody for active panel-observation commands.

This module owns canonical records, write-once persistence, and the pinned
headless-Codex runtime preimage.  It contains no task semantics, candidate
language, support/query policy, measurement thresholds, or formula logic.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import base64
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import (
    CodexNoToolsAttestation,
    attest_codex_no_tools,
)
from bongard.object_bongard_turn_journal import ObjectBongardTurnRuntime
from bongard.prototype_scene_observer import prototype_scene_transport_source_digest
from bongard.transport import (
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    codex_cli_authenticated_fingerprint,
    snapshot_cloud_policy_cache,
    snapshot_pinned_model_catalog,
)


DEFAULT_PROBE_LAUNCHER_SHA256 = (
    "19c4f144c5226a9f17c58e6f0fa854843b0f77a6eb420f40e2745a12f10f5d37"
)
DEFAULT_PROBE_MODEL = "gpt-5.6-sol"
DEFAULT_PROBE_REASONING_EFFORT = "medium"
PROBE_RUNTIME_SCHEMA = "gkm.bongard-panel-probe-runtime.v1"
MAX_PROBE_RECORD_BYTES = 64 * 1024 * 1024


class PanelProbeCustodyError(RuntimeError):
    """A canonical record, write-once store, or runtime preimage differs."""


def panel_probe_custody_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def make_probe_record(body: Mapping[str, Any]) -> dict[str, Any]:
    """Return one detached canonical object with a SHA-256 content address."""

    frozen = json.loads(canonical_json(dict(body)).decode("utf-8"))
    if "record_digest" in frozen:
        raise PanelProbeCustodyError("record body already contains record_digest")
    return {**frozen, "record_digest": "sha256:" + canonical_digest(frozen)}


def write_once_or_verify_probe_record(
    path: str | Path, value: Mapping[str, Any]
) -> None:
    """Persist canonical JSON once, or verify the exact existing bytes."""

    target = Path(path)
    payload = canonical_json(dict(value)) + b"\n"
    if len(payload) > MAX_PROBE_RECORD_BYTES:
        raise PanelProbeCustodyError("probe record exceeds the byte bound")
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise PanelProbeCustodyError("probe record parent is unsafe")
    try:
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError:
        if target.is_symlink() or target.read_bytes() != payload:
            raise PanelProbeCustodyError(f"existing artifact differs: {target}")
        return
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise PanelProbeCustodyError("probe record write stalled")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def read_probe_record(path: str | Path) -> dict[str, Any]:
    """Read and authenticate one bounded canonical record."""

    source = Path(path)
    if source.is_symlink():
        raise PanelProbeCustodyError(f"artifact is a symlink: {source}")
    try:
        info = source.stat()
    except OSError as exc:
        raise PanelProbeCustodyError(f"cannot stat artifact: {source}") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or not 0 < info.st_size <= MAX_PROBE_RECORD_BYTES
    ):
        raise PanelProbeCustodyError(f"artifact size or type differs: {source}")
    payload = source.read_bytes()
    if len(payload) != info.st_size or not payload.endswith(b"\n"):
        raise PanelProbeCustodyError(f"artifact encoding differs: {source}")
    try:
        raw = json.loads(payload[:-1].decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PanelProbeCustodyError(f"artifact is malformed: {source}") from exc
    if (
        not isinstance(raw, dict)
        or canonical_json(raw) + b"\n" != payload
        or type(raw.get("record_digest")) is not str
    ):
        raise PanelProbeCustodyError(f"artifact is not canonical: {source}")
    body = {key: value for key, value in raw.items() if key != "record_digest"}
    if raw["record_digest"] != "sha256:" + canonical_digest(body):
        raise PanelProbeCustodyError(f"artifact digest differs: {source}")
    return raw


def load_or_create_probe_runtime(
    *,
    output_root: str | Path,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    model: str,
    reasoning_effort: str,
    minutes: int,
    executable: str,
    launcher_sha256: str,
    verbose: bool,
) -> tuple[ObjectBongardTurnRuntime, dict[str, Any]]:
    """Load the exact stored runtime preimages or create them once."""

    root = Path(output_root)
    authorization_digest = authorization.get("record_digest")
    precommit_digest = precommit.get("record_digest")
    if type(authorization_digest) is not str or type(precommit_digest) is not str:
        raise PanelProbeCustodyError("runtime parents lack record digests")
    prior_path = root / "runtime.json"
    if prior_path.exists():
        prior = read_probe_record(prior_path)
        try:
            cache_encoded = prior["cloud_policy_cache_base64"]
            cache = CloudPolicyCacheSnapshot(
                None
                if cache_encoded is None
                else base64.b64decode(cache_encoded, validate=True)
            )
            catalog = CodexModelCatalogSnapshot(
                base64.b64decode(prior["model_catalog_base64"], validate=True)
            )
            attestation = CodexNoToolsAttestation.from_mapping(
                prior["no_tools_attestation"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PanelProbeCustodyError("stored runtime preimages differ") from exc
        fingerprint = codex_cli_authenticated_fingerprint(
            executable, expected_launcher_digest=launcher_sha256
        )
        runtime = ObjectBongardTurnRuntime(
            model=model,
            reasoning_effort=reasoning_effort,
            minutes=minutes,
            verbose=verbose,
            executable=executable,
            cloud_policy_cache_snapshot=cache,
            model_catalog_snapshot=catalog,
            expected_launcher_digest=launcher_sha256,
            no_tools_attestation=attestation,
            transport_source_digest=prototype_scene_transport_source_digest(),
        )
        if (
            prior.get("schema") != PROBE_RUNTIME_SCHEMA
            or prior.get("custody_source_digest")
            != panel_probe_custody_source_digest()
            or prior.get("authorization_digest") != authorization_digest
            or prior.get("execution_precommit_digest") != precommit_digest
            or prior.get("launcher_fingerprint") != dict(fingerprint)
            or prior.get("runtime_binding") != runtime.binding
        ):
            raise PanelProbeCustodyError(
                "stored runtime differs from the live pinned replay request"
            )
        return runtime, prior

    cache = snapshot_cloud_policy_cache()
    catalog = snapshot_pinned_model_catalog()
    fingerprint = codex_cli_authenticated_fingerprint(
        executable, expected_launcher_digest=launcher_sha256
    )
    if fingerprint.get("launcher_digest") != launcher_sha256:
        raise PanelProbeCustodyError("launcher fingerprint differs")
    attestation = attest_codex_no_tools(
        executable=executable,
        expected_launcher_digest=launcher_sha256,
        model_catalog_snapshot=catalog,
        cloud_policy_cache_snapshot=cache,
    )
    runtime = ObjectBongardTurnRuntime(
        model=model,
        reasoning_effort=reasoning_effort,
        minutes=minutes,
        verbose=verbose,
        executable=executable,
        cloud_policy_cache_snapshot=cache,
        model_catalog_snapshot=catalog,
        expected_launcher_digest=launcher_sha256,
        no_tools_attestation=attestation,
        transport_source_digest=prototype_scene_transport_source_digest(),
    )
    evidence = make_probe_record(
        {
            "schema": PROBE_RUNTIME_SCHEMA,
            "custody_source_digest": panel_probe_custody_source_digest(),
            "authorization_digest": authorization_digest,
            "execution_precommit_digest": precommit_digest,
            "runtime_binding": runtime.binding,
            "launcher_fingerprint": dict(fingerprint),
            "cloud_policy_cache_base64": (
                None
                if cache.data is None
                else base64.b64encode(cache.data).decode("ascii")
            ),
            "model_catalog_base64": base64.b64encode(catalog.data).decode("ascii"),
            "no_tools_attestation": attestation.to_dict(),
        }
    )
    write_once_or_verify_probe_record(prior_path, evidence)
    return runtime, evidence


__all__ = (
    "DEFAULT_PROBE_LAUNCHER_SHA256",
    "DEFAULT_PROBE_MODEL",
    "DEFAULT_PROBE_REASONING_EFFORT",
    "MAX_PROBE_RECORD_BYTES",
    "PROBE_RUNTIME_SCHEMA",
    "PanelProbeCustodyError",
    "load_or_create_probe_runtime",
    "make_probe_record",
    "panel_probe_custody_source_digest",
    "read_probe_record",
    "write_once_or_verify_probe_record",
)

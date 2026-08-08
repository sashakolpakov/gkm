"""Sealed one-turn soft-cue-pair nomination for rubric calibration.

This command is deliberately smaller than the calibration campaign.  It opens
only the twelve already-released development PNGs, freezes the exact two
neutral groups and Codex runtime before inference, performs one journaled
named-image turn, and then cold-verifies the artifact without model access.
The exact two ranked positive cue pairs are typed, content-addressed
predecessors; downstream calibration must consume that verified value rather
than accepting caller-supplied text or digest strings.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import argparse
import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Any, Callable, Mapping

from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import (
    CodexNoToolsAttestation,
    attest_codex_no_tools,
)
from bongard.object_bongard_semantics import (
    GROUP_SIZE,
    SOFT_CUE_CANDIDATE_COUNT,
    ObjectBongardSemanticArtifact,
    describe_object_bongard_support,
    object_bongard_semantics_output_schema,
    object_bongard_semantics_prompt,
    object_bongard_semantics_protocol_digest,
    object_bongard_semantics_source_digest,
    verify_object_bongard_semantic_artifact,
)
from bongard.object_bongard_soft_cues import (
    object_bongard_soft_cue_grammar_digest,
    object_bongard_soft_cue_source_digest,
)
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTurnRuntime,
    object_bongard_turn_journal_source_digest,
    verify_object_bongard_turn_journal,
)
from bongard.prototype_object_scene_observer import (
    PrototypeSceneObserverStatus,
    prototype_scene_transport_source_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.transport import (
    PINNED_CODEX_CLI_VERSION,
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexStructuredResult,
    codex_cli_authenticated_fingerprint,
    run_codex_named_images_structured,
    snapshot_cloud_policy_cache,
    snapshot_pinned_model_catalog,
)


AUTHORIZATION_SCHEMA = "gkm.bongard-object-rubric-nomination-authorization.v4"
PRECOMMIT_SCHEMA = "gkm.bongard-object-rubric-nomination-precommit.v4"
REPLAY_SCHEMA = "gkm.bongard-object-rubric-nomination-cold-replay.v4"
RESULT_SCHEMA = "gkm.bongard-object-rubric-nomination-result.v4"
COMMAND_ID = "bongard.object-rubric-nomination/seal-all-support-soft-cue-slate-v4"

AUTHORIZATION_FILENAME = "authorization.json"
PRECOMMIT_FILENAME = "execution_precommit.json"
ARTIFACT_FILENAME = "semantic_artifact.json"
REPLAY_FILENAME = "cold_replay.json"
RESULT_FILENAME = "result.json"
JOURNAL_DIRECTORY = "journals"

NOMINATION_MODEL = "gpt-5.6-sol"
NOMINATION_REASONING_EFFORT = "medium"
DEFAULT_MINUTES = 15
DEFAULT_EXECUTABLE = "codex"
DEFAULT_EXPECTED_LAUNCHER_SHA256 = (
    "19c4f144c5226a9f17c58e6f0fa854843b0f77a6eb420f40e2745a12f10f5d37"
)

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MAX_RECORD_BYTES = 64 * 1024 * 1024


class ObjectBongardRubricNominationCommandError(RuntimeError):
    """The nomination filesystem boundary or replay is invalid."""


def object_bongard_rubric_nomination_command_source_digest() -> str:
    return verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "model_can_propose_positive_soft_cue_text_only": True,
        "model_can_choose_operator_threshold_or_polarity": False,
        "soft_cue_text_defines_identity": True,
        "soft_cue_text_is_observed_not_executed": True,
        "feature_catalog_used": False,
        "soft_cue_proposer_sees_all_support_panels": True,
        "support_panels_per_group": GROUP_SIZE,
        "held_out_query_pixels_presented_to_proposer": False,
        "negation_allowed": False,
        "query_pixels_used": False,
        "fresh_broad_cohort_pixels_used": False,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_defines_identity_or_decision": False,
        "lean_required_for_replay": False,
    }


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _require_address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectBongardRubricNominationCommandError(
            f"{label} must be a sha256: address"
        )
    return value


def _record(body: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    frozen = json.loads(canonical_json(dict(body)).decode("utf-8"))
    return {**frozen, digest_field: _address(frozen)}


def _validate_record(
    value: object,
    *,
    schema: str,
    digest_field: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ObjectBongardRubricNominationCommandError(
            f"{label} must be an object"
        )
    raw = json.loads(canonical_json(dict(value)).decode("utf-8"))
    if raw.get("schema") != schema or digest_field not in raw:
        raise ObjectBongardRubricNominationCommandError(
            f"{label} schema or fields differ"
        )
    digest = _require_address(raw[digest_field], f"{label} digest")
    body = {key: item for key, item in raw.items() if key != digest_field}
    if digest != _address(body):
        raise ObjectBongardRubricNominationCommandError(
            f"{label} digest differs"
        )
    return raw


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_root(value: str | os.PathLike[str]) -> Path:
    requested = Path(value).expanduser().absolute()
    requested.mkdir(mode=0o700, parents=True, exist_ok=True)
    resolved = requested.resolve(strict=True)
    info = resolved.lstat()
    if (
        requested != resolved
        or not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
    ):
        raise ObjectBongardRubricNominationCommandError(
            "nomination output root must be one canonical real directory"
        )
    return resolved


def _read_record(path: Path, label: str) -> dict[str, Any]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise ObjectBongardRubricNominationCommandError(
            "platform lacks no-follow record access"
        )
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= _MAX_RECORD_BYTES
        ):
            raise ObjectBongardRubricNominationCommandError(
                f"{label} is not bounded singly-linked regular data"
            )
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as exc:
        raise ObjectBongardRubricNominationCommandError(
            f"cannot open {label}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        blocks: list[bytes] = []
        total = 0
        while True:
            block = os.read(
                descriptor, min(1024 * 1024, _MAX_RECORD_BYTES + 1 - total)
            )
            if not block:
                break
            blocks.append(block)
            total += len(block)
            if total > _MAX_RECORD_BYTES:
                raise ObjectBongardRubricNominationCommandError(
                    f"{label} exceeds its byte bound"
                )
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(opened) or identity(opened) != identity(after):
        raise ObjectBongardRubricNominationCommandError(
            f"{label} changed while being read"
        )
    payload = b"".join(blocks)
    try:
        decoded = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardRubricNominationCommandError(
            f"{label} is not canonical UTF-8 JSON"
        ) from exc
    if not isinstance(decoded, dict) or canonical_json(decoded) + b"\n" != payload:
        raise ObjectBongardRubricNominationCommandError(
            f"{label} bytes are not canonical"
        )
    return decoded


def _write_once(path: Path, value: Mapping[str, Any], label: str) -> None:
    payload = canonical_json(dict(value)) + b"\n"
    if path.exists():
        if _read_record(path, label) != dict(value):
            raise ObjectBongardRubricNominationCommandError(
                f"existing {label} differs"
            )
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ObjectBongardRubricNominationCommandError(
            f"cannot exclusively persist {label}"
        ) from exc
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ObjectBongardRubricNominationCommandError(
                    f"could not completely persist {label}"
                )
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    if _read_record(path, label) != dict(value):
        raise ObjectBongardRubricNominationCommandError(
            f"persisted {label} differs"
        )


def _encode_bytes(value: bytes | None) -> str | None:
    return None if value is None else base64.b64encode(value).decode("ascii")


def _decode_bytes(value: object, label: str) -> bytes | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ObjectBongardRubricNominationCommandError(
            f"{label} base64 is invalid"
        )
    try:
        decoded = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeError, ValueError) as exc:
        raise ObjectBongardRubricNominationCommandError(
            f"{label} base64 is invalid"
        ) from exc
    if _encode_bytes(decoded) != value:
        raise ObjectBongardRubricNominationCommandError(
            f"{label} base64 is not canonical"
        )
    return decoded


def _load_calibration_source(source_root: str | os.PathLike[str]):
    # Lazy import keeps the typed predecessor importable by calibration code.
    from bongard.object_bongard_rubric_calibration import (
        load_object_bongard_rubric_calibration_source,
    )

    return load_object_bongard_rubric_calibration_source(source_root)


def _source_digests() -> list[dict[str, str]]:
    from bongard.object_bongard_rubric_calibration import (
        object_bongard_rubric_calibration_source_digest,
    )

    rows = {
        "nomination_command_source_sha256": (
            object_bongard_rubric_nomination_command_source_digest()
        ),
        "calibration_source_sha256": (
            object_bongard_rubric_calibration_source_digest()
        ),
        "semantics_source_sha256": object_bongard_semantics_source_digest(),
        "soft_cue_source_sha256": object_bongard_soft_cue_source_digest(),
        "turn_journal_source_sha256": object_bongard_turn_journal_source_digest(),
        "transport_source_sha256": prototype_scene_transport_source_digest(),
    }
    return [{"role": role, "sha256": rows[role]} for role in sorted(rows)]


def _groups(source: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    group_0 = tuple(sorted(item.panel_id for item in source.group_a_panels))
    group_1 = tuple(sorted(item.panel_id for item in source.group_b_panels))
    if (
        len(group_0) != GROUP_SIZE
        or len(group_1) != GROUP_SIZE
        or set(group_0) & set(group_1)
    ):
        raise ObjectBongardRubricNominationCommandError(
            "calibration source does not provide exact disjoint 6+6 support groups"
        )
    return group_0, group_1


def _panel_commitments(source: object) -> list[dict[str, object]]:
    return [
        {
            "group_id": f"group_{index}",
            "panel_ids": list(group),
            "png_sha256": [
                next(item.png_sha256 for item in source.panels if item.panel_id == panel)
                for panel in group
            ],
            "panel_binding_digests": [
                next(
                    item.panel_binding_digest
                    for item in source.panels
                    if item.panel_id == panel
                )
                for panel in group
            ],
        }
        for index, group in enumerate(_groups(source))
    ]


def _authorization(
    source: object,
    *,
    minutes: int,
    executable: str,
    expected_launcher_sha256: str,
) -> dict[str, Any]:
    if source.nomination_artifact is not None:
        raise ObjectBongardRubricNominationCommandError(
            "nomination authorization requires the pre-nomination source"
        )
    if isinstance(minutes, bool) or not isinstance(minutes, int) or not 1 <= minutes <= 120:
        raise ObjectBongardRubricNominationCommandError("minutes must lie in 1..120")
    if not isinstance(executable, str) or not executable:
        raise ObjectBongardRubricNominationCommandError("executable is invalid")
    if not isinstance(expected_launcher_sha256, str) or _RAW_DIGEST.fullmatch(
        expected_launcher_sha256
    ) is None:
        raise ObjectBongardRubricNominationCommandError(
            "expected launcher digest is invalid"
        )
    body = {
        "schema": AUTHORIZATION_SCHEMA,
        "command_id": COMMAND_ID,
        "source_digest": source.source_digest,
        "source_digests": _source_digests(),
        "context_task_id": source.panels[0].task_id,
        "context_task_id_policy": (
            "lowest-selected-ordinal-task-id-is-transport-context-only"
        ),
        "groups": _panel_commitments(source),
        "semantic_protocol_digest": object_bongard_semantics_protocol_digest(),
        "semantic_output_schema_digest": canonical_digest(
            object_bongard_semantics_output_schema()
        ),
        "soft_cue_grammar_digest": object_bongard_soft_cue_grammar_digest(),
        "ranked_soft_cue_pair_count": SOFT_CUE_CANDIDATE_COUNT,
        "physical_model_call_count": 1,
        "fresh_calibration_panels_used": False,
        "historical_released_pixels_only": True,
        "authorization_and_precommit_required_before_inference": True,
        "runtime_policy": {
            "model": NOMINATION_MODEL,
            "reasoning_effort": NOMINATION_REASONING_EFFORT,
            "minutes": minutes,
            "verbose": False,
            "executable": executable,
            "expected_launcher_sha256": expected_launcher_sha256,
        },
        **_authority_data(),
    }
    return _record(body, "authorization_digest")


def _create_runtime(
    authorization: Mapping[str, Any],
    *,
    cache_snapshotter: Callable[[], CloudPolicyCacheSnapshot],
    catalog_snapshotter: Callable[[], CodexModelCatalogSnapshot],
    launcher_fingerprinter: Callable[..., Mapping[str, str]],
    runtime_attester: Callable[..., CodexNoToolsAttestation],
) -> tuple[ObjectBongardTurnRuntime, Mapping[str, str]]:
    policy = authorization["runtime_policy"]
    cache = cache_snapshotter()
    catalog = catalog_snapshotter()
    if not isinstance(cache, CloudPolicyCacheSnapshot) or not isinstance(
        catalog, CodexModelCatalogSnapshot
    ):
        raise ObjectBongardRubricNominationCommandError(
            "runtime snapshotter returned the wrong type"
        )
    fingerprint = launcher_fingerprinter(
        policy["executable"],
        expected_launcher_digest=policy["expected_launcher_sha256"],
    )
    attestation = runtime_attester(
        executable=policy["executable"],
        expected_launcher_digest=policy["expected_launcher_sha256"],
        model_catalog_snapshot=catalog,
        cloud_policy_cache_snapshot=cache,
    )
    runtime = ObjectBongardTurnRuntime(
        model=policy["model"],
        reasoning_effort=policy["reasoning_effort"],
        minutes=policy["minutes"],
        verbose=policy["verbose"],
        executable=policy["executable"],
        cloud_policy_cache_snapshot=cache,
        model_catalog_snapshot=catalog,
        expected_launcher_digest=policy["expected_launcher_sha256"],
        no_tools_attestation=attestation,
        transport_source_digest=prototype_scene_transport_source_digest(),
    )
    return runtime, fingerprint


def _precommit(
    authorization: Mapping[str, Any],
    runtime: ObjectBongardTurnRuntime,
    fingerprint: Mapping[str, str],
) -> dict[str, Any]:
    body = {
        "schema": PRECOMMIT_SCHEMA,
        "command_id": COMMAND_ID,
        "authorization_digest": authorization["authorization_digest"],
        "source_digest": authorization["source_digest"],
        "source_digests": authorization["source_digests"],
        "context_task_id": authorization["context_task_id"],
        "groups": authorization["groups"],
        "semantic_protocol_digest": authorization["semantic_protocol_digest"],
        "semantic_output_schema_digest": authorization[
            "semantic_output_schema_digest"
        ],
        "soft_cue_grammar_digest": authorization["soft_cue_grammar_digest"],
        "ranked_soft_cue_pair_count": authorization[
            "ranked_soft_cue_pair_count"
        ],
        "runtime_binding": runtime.binding,
        "cloud_policy_cache_snapshot_base64": _encode_bytes(
            runtime.cloud_policy_cache_snapshot.data
            if runtime.cloud_policy_cache_snapshot is not None
            else None
        ),
        "model_catalog_snapshot_base64": _encode_bytes(
            runtime.model_catalog_snapshot.data
        ),
        "no_tools_attestation": runtime.no_tools_attestation.to_dict(),
        "launcher_fingerprint": dict(fingerprint),
        "precommit_fsynced_before_inference": True,
        "physical_model_call_count": 1,
        **_authority_data(),
    }
    return _record(body, "precommit_digest")


def _runtime_from_precommit(
    precommit: Mapping[str, Any], authorization: Mapping[str, Any]
) -> ObjectBongardTurnRuntime:
    binding = precommit["runtime_binding"]
    cache = CloudPolicyCacheSnapshot(
        _decode_bytes(
            precommit["cloud_policy_cache_snapshot_base64"], "policy cache"
        )
    )
    catalog_bytes = _decode_bytes(
        precommit["model_catalog_snapshot_base64"], "model catalog"
    )
    if catalog_bytes is None:
        raise ObjectBongardRubricNominationCommandError(
            "model catalog snapshot cannot be absent"
        )
    catalog = CodexModelCatalogSnapshot(catalog_bytes)
    attestation = CodexNoToolsAttestation.from_mapping(
        precommit["no_tools_attestation"]
    )
    runtime = ObjectBongardTurnRuntime(
        model=binding["model"],
        reasoning_effort=binding["reasoning_effort"],
        minutes=binding["minutes"],
        verbose=binding["verbose"],
        executable=binding["executable"],
        cloud_policy_cache_snapshot=cache,
        model_catalog_snapshot=catalog,
        expected_launcher_digest=binding["expected_launcher_digest"],
        no_tools_attestation=attestation,
        transport_source_digest=binding["transport_source_digest"],
    )
    if (
        runtime.binding != binding
        or authorization["runtime_policy"]
        != {
            "model": runtime.model,
            "reasoning_effort": runtime.reasoning_effort,
            "minutes": runtime.minutes,
            "verbose": runtime.verbose,
            "executable": runtime.executable,
            "expected_launcher_sha256": runtime.expected_launcher_digest,
        }
    ):
        raise ObjectBongardRubricNominationCommandError(
            "precommitted runtime differs from authorization"
        )
    return runtime


def _verify_precommit_before_inference(
    precommit: object,
    authorization: Mapping[str, Any],
) -> ObjectBongardTurnRuntime:
    """Validate the complete frozen runtime envelope before any journal call."""

    raw = _validate_record(
        precommit,
        schema=PRECOMMIT_SCHEMA,
        digest_field="precommit_digest",
        label="nomination precommit",
    )
    expected_fields = {
        "schema",
        "command_id",
        "authorization_digest",
        "source_digest",
        "source_digests",
        "context_task_id",
        "groups",
        "semantic_protocol_digest",
        "semantic_output_schema_digest",
        "soft_cue_grammar_digest",
        "ranked_soft_cue_pair_count",
        "runtime_binding",
        "cloud_policy_cache_snapshot_base64",
        "model_catalog_snapshot_base64",
        "no_tools_attestation",
        "launcher_fingerprint",
        "precommit_fsynced_before_inference",
        "physical_model_call_count",
        *_authority_data(),
        "precommit_digest",
    }
    if set(raw) != expected_fields:
        raise ObjectBongardRubricNominationCommandError(
            "nomination precommit fields differ"
        )
    if (
        raw["command_id"] != COMMAND_ID
        or raw["authorization_digest"] != authorization["authorization_digest"]
        or raw["source_digest"] != authorization["source_digest"]
        or raw["source_digests"] != authorization["source_digests"]
        or raw["source_digests"] != _source_digests()
        or raw["context_task_id"] != authorization["context_task_id"]
        or raw["groups"] != authorization["groups"]
        or raw["semantic_protocol_digest"]
        != authorization["semantic_protocol_digest"]
        or raw["semantic_protocol_digest"]
        != object_bongard_semantics_protocol_digest()
        or raw["semantic_output_schema_digest"]
        != authorization["semantic_output_schema_digest"]
        or raw["semantic_output_schema_digest"]
        != canonical_digest(object_bongard_semantics_output_schema())
        or raw["soft_cue_grammar_digest"]
        != authorization["soft_cue_grammar_digest"]
        or raw["soft_cue_grammar_digest"]
        != object_bongard_soft_cue_grammar_digest()
        or raw["ranked_soft_cue_pair_count"] != SOFT_CUE_CANDIDATE_COUNT
        or raw["precommit_fsynced_before_inference"] is not True
        or raw["physical_model_call_count"] != 1
        or any(raw[key] != value for key, value in _authority_data().items())
    ):
        raise ObjectBongardRubricNominationCommandError(
            "nomination precommit policy or parents differ"
        )
    policy = authorization["runtime_policy"]
    expected_fingerprint = {
        "version": PINNED_CODEX_CLI_VERSION,
        "launcher_digest": policy["expected_launcher_sha256"],
    }
    if raw["launcher_fingerprint"] != expected_fingerprint:
        raise ObjectBongardRubricNominationCommandError(
            "nomination launcher fingerprint differs"
        )
    return _runtime_from_precommit(raw, authorization)


def _support(source: object) -> dict[str, bytes]:
    admitted = set().union(*_groups(source))
    support = {
        item.panel_id: item.exact_png_bytes
        for item in source.panels
        if item.panel_id in admitted
    }
    if set(support) != admitted:
        raise ObjectBongardRubricNominationCommandError(
            "all-support PNG inventory differs from the authorization"
        )
    return support


def _images(source: object) -> tuple[tuple[str, bytes], ...]:
    support = _support(source)
    return tuple(
        (f"group_{group_index}_ref_{index:02d}.png", support[panel_id])
        for group_index, group in enumerate(_groups(source))
        for index, panel_id in enumerate(group)
    )


def _forbidden_transport(**_: object) -> CodexStructuredResult:
    raise ObjectBongardRubricNominationCommandError(
        "cold replay attempted a model transport"
    )


def _journal(
    root: Path,
    source: object,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    runtime: ObjectBongardTurnRuntime,
    transport: Callable[..., CodexStructuredResult],
) -> ObjectBongardNamedImageTurnJournalTransport:
    return ObjectBongardNamedImageTurnJournalTransport(
        root / JOURNAL_DIRECTORY / "semantic_nomination",
        authorization_digest=authorization["authorization_digest"],
        execution_precommit_digest=precommit["precommit_digest"],
        task_id=authorization["context_task_id"],
        turn_kind="semantic_nomination",
        expected_prompt=object_bongard_semantics_prompt(),
        expected_images=_images(source),
        expected_output_schema=object_bongard_semantics_output_schema(),
        runtime=runtime,
        underlying_transport=transport,
    )


def _verify_prefix(
    root: Path,
    source: object,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    ObjectBongardTurnRuntime,
    ObjectBongardSemanticArtifact,
    dict[str, object],
]:
    authorization = _validate_record(
        _read_record(root / AUTHORIZATION_FILENAME, "nomination authorization"),
        schema=AUTHORIZATION_SCHEMA,
        digest_field="authorization_digest",
        label="nomination authorization",
    )
    policy = authorization["runtime_policy"]
    expected_authorization = _authorization(
        source,
        minutes=policy["minutes"],
        executable=policy["executable"],
        expected_launcher_sha256=policy["expected_launcher_sha256"],
    )
    if authorization != expected_authorization:
        raise ObjectBongardRubricNominationCommandError(
            "nomination authorization differs from exact replay"
        )
    precommit = _validate_record(
        _read_record(root / PRECOMMIT_FILENAME, "nomination precommit"),
        schema=PRECOMMIT_SCHEMA,
        digest_field="precommit_digest",
        label="nomination precommit",
    )
    if (
        precommit["authorization_digest"] != authorization["authorization_digest"]
        or precommit["source_digest"] != source.source_digest
        or precommit["source_digests"] != _source_digests()
        or precommit["context_task_id"] != authorization["context_task_id"]
        or precommit["groups"] != authorization["groups"]
    ):
        raise ObjectBongardRubricNominationCommandError(
            "nomination precommit parents differ"
        )
    runtime = _verify_precommit_before_inference(precommit, authorization)
    artifact_data = _read_record(
        root / ARTIFACT_FILENAME, "semantic nomination artifact"
    )
    artifact = ObjectBongardSemanticArtifact.from_data(artifact_data)
    verify_object_bongard_semantic_artifact(
        artifact,
        support_png_by_panel_id=_support(source),
        expected_task_id=authorization["context_task_id"],
        expected_observation_context_digest=precommit["precommit_digest"],
        expected_artifact_digest=artifact.artifact_digest,
    )
    journal = _journal(
        root, source, authorization, precommit, runtime, _forbidden_transport
    )
    summary = verify_object_bongard_turn_journal(journal).to_data()
    if summary["terminal_status"] not in {"success", "failure"}:
        raise ObjectBongardRubricNominationCommandError(
            "nomination journal is not terminal"
        )
    return authorization, precommit, runtime, artifact, summary


def _soft_cue_pair_commitments(
    artifact: ObjectBongardSemanticArtifact,
) -> list[dict[str, object]]:
    """Expose the exact ranked text identities without a catalog surrogate."""

    if artifact.status is not PrototypeSceneObserverStatus.SUCCESS:
        return []
    if len(artifact.soft_cue_candidates) != SOFT_CUE_CANDIDATE_COUNT:
        raise ObjectBongardRubricNominationCommandError(
            "successful nomination does not contain exactly two soft-cue pairs"
        )
    return [
        {
            "candidate_rank": pair.candidate_rank,
            "pair_digest": pair.pair_digest,
            "group_0_cue_digest": pair.group_0_cue.cue_digest,
            "group_0_cue_text": pair.group_0_cue.text,
            "group_1_cue_digest": pair.group_1_cue.cue_digest,
            "group_1_cue_text": pair.group_1_cue.text,
        }
        for pair in artifact.soft_cue_candidates
    ]


def _replay_record(
    source: object,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    artifact: ObjectBongardSemanticArtifact,
    journal_summary: Mapping[str, object],
) -> dict[str, Any]:
    body = {
        "schema": REPLAY_SCHEMA,
        "command_id": COMMAND_ID,
        "source_digest": source.source_digest,
        "authorization_digest": authorization["authorization_digest"],
        "execution_precommit_digest": precommit["precommit_digest"],
        "semantic_artifact_digest": artifact.artifact_digest,
        "semantic_status": artifact.status.value,
        "soft_cue_grammar_digest": object_bongard_soft_cue_grammar_digest(),
        "ranked_soft_cue_pairs": _soft_cue_pair_commitments(artifact),
        "journal_summary": dict(journal_summary),
        "verified_physical_turn_count": 1,
        "model_calls_during_replay": 0,
        "exact_png_bytes_replayed": True,
        "source_digests": _source_digests(),
        **_authority_data(),
    }
    return _record(body, "replay_digest")


def _result_record(
    source: object,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    artifact: ObjectBongardSemanticArtifact,
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    accepted = artifact.status is PrototypeSceneObserverStatus.SUCCESS
    body = {
        "schema": RESULT_SCHEMA,
        "command_id": COMMAND_ID,
        "accepted": accepted,
        "source_digest": source.source_digest,
        "authorization_digest": authorization["authorization_digest"],
        "execution_precommit_digest": precommit["precommit_digest"],
        "semantic_artifact_digest": artifact.artifact_digest,
        "cold_replay_digest": replay["replay_digest"],
        "semantic_status": artifact.status.value,
        "soft_cue_grammar_digest": object_bongard_soft_cue_grammar_digest(),
        "ranked_soft_cue_pairs": _soft_cue_pair_commitments(artifact),
        "physical_model_call_count": 1,
        "model_calls_during_replay": 0,
        **_authority_data(),
    }
    return _record(body, "result_digest")


@dataclass(frozen=True, slots=True)
class VerifiedObjectBongardRubricNomination:
    artifact: ObjectBongardSemanticArtifact
    authorization_digest: str
    execution_precommit_digest: str
    cold_replay_digest: str
    result_digest: str
    source_digest: str
    accepted: bool

    def __post_init__(self) -> None:
        for label, value in (
            ("authorization", self.authorization_digest),
            ("execution precommit", self.execution_precommit_digest),
            ("cold replay", self.cold_replay_digest),
            ("result", self.result_digest),
        ):
            _require_address(value, f"verified {label} digest")
        if not isinstance(self.source_digest, str) or _RAW_DIGEST.fullmatch(
            self.source_digest
        ) is None:
            raise ObjectBongardRubricNominationCommandError(
                "verified source digest is invalid"
            )
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be bool")
        if self.accepted != (
            self.artifact.status is PrototypeSceneObserverStatus.SUCCESS
        ):
            raise ObjectBongardRubricNominationCommandError(
                "verified result acceptance differs from semantic status"
            )


def cold_verify_object_bongard_rubric_nomination(
    output_root: str | os.PathLike[str],
    *,
    source_root: str | os.PathLike[str],
) -> VerifiedObjectBongardRubricNomination:
    root = _ensure_root(output_root)
    expected_names = {
        AUTHORIZATION_FILENAME,
        PRECOMMIT_FILENAME,
        ARTIFACT_FILENAME,
        REPLAY_FILENAME,
        RESULT_FILENAME,
        JOURNAL_DIRECTORY,
    }
    if {item.name for item in root.iterdir()} != expected_names:
        raise ObjectBongardRubricNominationCommandError(
            "nomination root inventory differs"
        )
    source = _load_calibration_source(source_root)
    authorization, precommit, _, artifact, summary = _verify_prefix(root, source)
    expected_replay = _replay_record(
        source, authorization, precommit, artifact, summary
    )
    replay = _validate_record(
        _read_record(root / REPLAY_FILENAME, "nomination cold replay"),
        schema=REPLAY_SCHEMA,
        digest_field="replay_digest",
        label="nomination cold replay",
    )
    if replay != expected_replay:
        raise ObjectBongardRubricNominationCommandError(
            "nomination replay record differs"
        )
    expected_result = _result_record(
        source, authorization, precommit, artifact, replay
    )
    result = _validate_record(
        _read_record(root / RESULT_FILENAME, "nomination result"),
        schema=RESULT_SCHEMA,
        digest_field="result_digest",
        label="nomination result",
    )
    if result != expected_result:
        raise ObjectBongardRubricNominationCommandError(
            "nomination result record differs"
        )
    return VerifiedObjectBongardRubricNomination(
        artifact=artifact,
        authorization_digest=authorization["authorization_digest"],
        execution_precommit_digest=precommit["precommit_digest"],
        cold_replay_digest=replay["replay_digest"],
        result_digest=result["result_digest"],
        source_digest=source.source_digest,
        accepted=result["accepted"],
    )


def run_object_bongard_rubric_nomination(
    output_root: str | os.PathLike[str],
    *,
    source_root: str | os.PathLike[str],
    minutes: int = DEFAULT_MINUTES,
    executable: str = DEFAULT_EXECUTABLE,
    expected_launcher_sha256: str = DEFAULT_EXPECTED_LAUNCHER_SHA256,
    cache_snapshotter: Callable[[], CloudPolicyCacheSnapshot] = (
        snapshot_cloud_policy_cache
    ),
    catalog_snapshotter: Callable[[], CodexModelCatalogSnapshot] = (
        snapshot_pinned_model_catalog
    ),
    launcher_fingerprinter: Callable[..., Mapping[str, str]] = (
        codex_cli_authenticated_fingerprint
    ),
    runtime_attester: Callable[..., CodexNoToolsAttestation] = (
        attest_codex_no_tools
    ),
    visual_transport: Callable[..., CodexStructuredResult] = (
        run_codex_named_images_structured
    ),
) -> VerifiedObjectBongardRubricNomination:
    root = _ensure_root(output_root)
    source = _load_calibration_source(source_root)
    authorization = _authorization(
        source,
        minutes=minutes,
        executable=executable,
        expected_launcher_sha256=expected_launcher_sha256,
    )
    _write_once(
        root / AUTHORIZATION_FILENAME, authorization, "nomination authorization"
    )
    precommit_path = root / PRECOMMIT_FILENAME
    if precommit_path.exists():
        precommit = _validate_record(
            _read_record(precommit_path, "nomination precommit"),
            schema=PRECOMMIT_SCHEMA,
            digest_field="precommit_digest",
            label="nomination precommit",
        )
    else:
        runtime, fingerprint = _create_runtime(
            authorization,
            cache_snapshotter=cache_snapshotter,
            catalog_snapshotter=catalog_snapshotter,
            launcher_fingerprinter=launcher_fingerprinter,
            runtime_attester=runtime_attester,
        )
        precommit = _precommit(authorization, runtime, fingerprint)
        _write_once(precommit_path, precommit, "nomination precommit")
    # This full replay is the last operation before constructing a transport-
    # capable journal.  A self-digested but altered resume record cannot cause
    # a physical call and fail only afterward.
    runtime = _verify_precommit_before_inference(precommit, authorization)
    journal = _journal(
        root, source, authorization, precommit, runtime, visual_transport
    )
    artifact = describe_object_bongard_support(
        task_id=authorization["context_task_id"],
        group_0_panel_ids=_groups(source)[0],
        group_1_panel_ids=_groups(source)[1],
        support_png_by_panel_id=_support(source),
        observation_context_digest=precommit["precommit_digest"],
        model=runtime.model,
        reasoning_effort=runtime.reasoning_effort,
        minutes=runtime.minutes,
        verbose=runtime.verbose,
        executable=runtime.executable,
        cloud_policy_cache_snapshot=runtime.cloud_policy_cache_snapshot,
        expected_launcher_digest=runtime.expected_launcher_digest,
        model_catalog_snapshot=runtime.model_catalog_snapshot,
        no_tools_attestation=runtime.no_tools_attestation,
        transport=journal,
    )
    verify_object_bongard_semantic_artifact(
        artifact,
        support_png_by_panel_id=_support(source),
        expected_task_id=authorization["context_task_id"],
        expected_observation_context_digest=precommit["precommit_digest"],
        expected_artifact_digest=artifact.artifact_digest,
    )
    _write_once(
        root / ARTIFACT_FILENAME,
        artifact.to_data(),
        "semantic nomination artifact",
    )
    authorization, precommit, _, artifact, summary = _verify_prefix(root, source)
    replay = _replay_record(source, authorization, precommit, artifact, summary)
    _write_once(root / REPLAY_FILENAME, replay, "nomination cold replay")
    result = _result_record(source, authorization, precommit, artifact, replay)
    _write_once(root / RESULT_FILENAME, result, "nomination result")
    return cold_verify_object_bongard_rubric_nomination(
        root, source_root=source_root
    )


def copy_verified_object_bongard_rubric_nomination(
    output_root: str | os.PathLike[str],
    destination_root: str | os.PathLike[str],
    *,
    source_root: str | os.PathLike[str],
) -> VerifiedObjectBongardRubricNomination:
    source_path = Path(output_root).expanduser().resolve(strict=True)
    verified = cold_verify_object_bongard_rubric_nomination(
        source_path, source_root=source_root
    )
    destination = Path(destination_root).expanduser().absolute()
    if destination.exists():
        raise ObjectBongardRubricNominationCommandError(
            "nomination copy destination already exists"
        )
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.copytree(source_path, destination, copy_function=shutil.copy2)
    _fsync_directory(destination.parent)
    copied = cold_verify_object_bongard_rubric_nomination(
        destination, source_root=source_root
    )
    if copied != verified:
        raise ObjectBongardRubricNominationCommandError(
            "copied nomination predecessor differs"
        )
    return copied


def _default_source_root() -> Path:
    from bongard.object_bongard_rubric_calibration import (
        DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE,
    )

    return DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m bongard.object_bongard_rubric_nomination_command",
        description=(
            "Seal, run, or cold-verify two ranked neutral Bongard soft-cue pairs"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    launch = sub.add_parser("launch")
    launch.add_argument("--output-root", type=Path, required=True)
    launch.add_argument("--source-root", type=Path, default=_default_source_root())
    launch.add_argument("--minutes", type=int, default=DEFAULT_MINUTES)
    launch.add_argument("--executable", default=DEFAULT_EXECUTABLE)
    launch.add_argument(
        "--expected-launcher-sha256",
        default=DEFAULT_EXPECTED_LAUNCHER_SHA256,
    )
    verify = sub.add_parser("verify")
    verify.add_argument("--output-root", type=Path, required=True)
    verify.add_argument("--source-root", type=Path, default=_default_source_root())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "launch":
        verified = run_object_bongard_rubric_nomination(
            args.output_root,
            source_root=args.source_root,
            minutes=args.minutes,
            executable=args.executable,
            expected_launcher_sha256=args.expected_launcher_sha256,
        )
    else:
        verified = cold_verify_object_bongard_rubric_nomination(
            args.output_root, source_root=args.source_root
        )
    print(
        json.dumps(
            {
                "accepted": verified.accepted,
                "authorization_digest": verified.authorization_digest,
                "execution_precommit_digest": verified.execution_precommit_digest,
                "semantic_artifact_digest": verified.artifact.artifact_digest,
                "cold_replay_digest": verified.cold_replay_digest,
                "result_digest": verified.result_digest,
                "ranked_soft_cue_pairs": _soft_cue_pair_commitments(
                    verified.artifact
                ),
                **_authority_data(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if verified.accepted else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = (
    "ObjectBongardRubricNominationCommandError",
    "VerifiedObjectBongardRubricNomination",
    "cold_verify_object_bongard_rubric_nomination",
    "copy_verified_object_bongard_rubric_nomination",
    "object_bongard_rubric_nomination_command_source_digest",
    "run_object_bongard_rubric_nomination",
)

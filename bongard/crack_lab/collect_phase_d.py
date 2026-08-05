"""Collect one artifact-certified preregistered Phase D campaign offline.

The scientific runners publish one ``phase-d-track-report/v7`` document per
arm.  This module is the deliberately small publication boundary above those
runners: it searches only the explicitly named directories (never the repo),
requires every report to live in its preregistered execution-tag artifact,
replays that artifact through the owning track's validator, requires exact
checkpoint/report record identity, delegates cross-arm scientific checks to
:mod:`phase_d_protocol`, and writes one canonical, write-once campaign.

No proposer, API request, dataset sampler, or network operation is invoked by
this module.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

try:  # package import in tests, script import from crack_lab
    from . import phase_d_protocol, semantic_replay
except ImportError:  # pragma: no cover - exercised by direct script usage
    import phase_d_protocol  # type: ignore
    import semantic_replay  # type: ignore


CAMPAIGN_SCHEMA = "bongard.phase-d-campaign/v6"
ARTIFACT_CERTIFICATION_SCHEMA = \
    "bongard.phase-d-artifact-certification/v1"
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_PREDICATE_SOURCE_BYTES = 1_000_000
MAX_PREDICATE_LOG_BYTES = 1_000_000

_CAMPAIGN_KEYS = {
    "schema",
    "preregistration_schema",
    "preregistration_digest",
    "corpus_digest",
    "corpus_problem_count",
    "corpus_panel_set_digests",
    "track_report_schema",
    "arm_count",
    "reports",
    "artifact_certifications",
    "aggregates",
    "campaign_digest",
}

_ARTIFACT_CERTIFICATION_KEYS = {
    "schema", "arm_id", "execution_tag", "track", "artifact_kind",
    "report_digest", "checkpoint_digest", "results_digest",
    "scientific_source_kind", "scientific_source_digest",
    "replay_receipts_digest", "certification_digest",
}


class CampaignCollectionError(RuntimeError):
    """Input discovery or write-once campaign publication failed closed."""


@dataclass(frozen=True)
class DiscoveredTrackReport:
    """A validated report coupled to the regular file it came from."""

    report: dict[str, Any]
    origin: str
    artifact_dir: str
    certification: dict[str, Any]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CampaignCollectionError(
                f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _reject_nonfinite(token: str) -> Any:
    raise CampaignCollectionError(f"non-finite JSON constant {token!r}")


def _load_json(path: str, description: str) -> Any:
    """Load strict bounded JSON through one stable, no-follow descriptor."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) \
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = -1
    try:
        fd = os.open(path, flags)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise CampaignCollectionError(
                f"{description} is not a singly-linked regular file: {path}")
        if before.st_size > MAX_JSON_BYTES:
            raise CampaignCollectionError(
                f"{description} exceeds {MAX_JSON_BYTES} bytes: {path}")
        blocks: list[bytes] = []
        remaining = MAX_JSON_BYTES + 1
        while remaining:
            block = os.read(fd, min(1024 * 1024, remaining))
            if not block:
                break
            blocks.append(block)
            remaining -= len(block)
        data = b"".join(blocks)
        after = os.fstat(fd)
        current = os.lstat(path)
        identity = lambda item: (
            item.st_dev, item.st_ino, item.st_size,
            item.st_mtime_ns, item.st_ctime_ns)
        if len(data) > MAX_JSON_BYTES:
            raise CampaignCollectionError(
                f"{description} exceeds {MAX_JSON_BYTES} bytes: {path}")
        if identity(before) != identity(after) \
                or identity(after) != identity(current) \
                or len(data) != after.st_size:
            raise CampaignCollectionError(
                f"{description} changed while being read: {path}")
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except CampaignCollectionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignCollectionError(
            f"cannot load {description} {path!r}: {exc}") from exc
    finally:
        if fd >= 0:
            os.close(fd)


def load_preregistration(path: str) -> dict[str, Any]:
    """Load and validate the exact Phase D v6 preregistration document."""
    value = _load_json(os.path.abspath(path), "Phase D preregistration")
    if not isinstance(value, dict):
        raise CampaignCollectionError("Phase D preregistration must be an object")
    try:
        phase_d_protocol.validate_preregistration(value)
    except phase_d_protocol.PhaseDProtocolError as exc:
        raise CampaignCollectionError(
            f"invalid Phase D preregistration: {exc}") from exc
    return value


def _normalise_report_dirs(report_dirs: Sequence[str]) -> tuple[str, ...]:
    if not report_dirs:
        raise CampaignCollectionError(
            "at least one explicit report directory is required")
    normalised: list[str] = []
    seen: set[str] = set()
    for supplied in report_dirs:
        if not isinstance(supplied, str) or not supplied:
            raise CampaignCollectionError("report directory paths must be nonempty")
        path = os.path.realpath(os.path.abspath(supplied))
        if path in seen:
            raise CampaignCollectionError(
                f"report directory was supplied more than once: {supplied!r}")
        if not os.path.isdir(path):
            raise CampaignCollectionError(
                f"report directory does not exist or is not a directory: {supplied!r}")
        seen.add(path)
        normalised.append(path)
    # Directory argument order cannot affect discovery, errors, or publication.
    return tuple(sorted(normalised))


def _matching_arm(
        report: Mapping[str, Any],
        preregistration: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [
        arm for arm in preregistration["arms"]
        if arm["arm_id"] == report.get("arm_id")
    ]
    if len(matches) != 1:
        raise CampaignCollectionError("track report arm was not preregistered")
    return matches[0]


def _expected_artifact_from_origin(
        origin: str, arm: Mapping[str, Any]) -> str:
    """Bind a report path to the execution-tag artifact naming contract."""
    path = os.path.realpath(os.path.abspath(origin))
    reports_dir = os.path.dirname(path)
    artifact = os.path.dirname(reports_dir)
    suffix = (
        "_predicates" if arm["track"] == "UNRESTRICTED" else "_semantic")
    expected_artifact_name = arm["execution_tag"] + suffix
    expected_filename = arm["arm_id"].replace(":", "__") + ".json"
    if os.path.basename(reports_dir) != "track_reports" \
            or os.path.basename(artifact) != expected_artifact_name \
            or os.path.basename(path) != expected_filename:
        raise CampaignCollectionError(
            f"track report origin is not its expected execution-tag artifact: "
            f"{origin!r}")
    return artifact


def _artifact_json(artifact: str, filename: str, description: str) -> dict[str, Any]:
    path = os.path.join(artifact, filename)
    if os.path.islink(path) or not os.path.isfile(path):
        raise CampaignCollectionError(
            f"execution artifact lacks regular {description}: {path!r}")
    value = _load_json(path, description)
    if not isinstance(value, dict):
        raise CampaignCollectionError(
            f"execution artifact {description} must be an object: {path!r}")
    return value


def _require_regular_artifact_file(
        artifact: str, filename: str, description: str, *,
        required: bool = True) -> None:
    path = os.path.join(artifact, filename)
    if not os.path.lexists(path):
        if required:
            raise CampaignCollectionError(
                f"execution artifact lacks {description}: {path!r}")
        return
    if os.path.islink(path) or not os.path.isfile(path):
        raise CampaignCollectionError(
            f"execution artifact {description} must be a regular file: {path!r}")


def _raw_file_digest(path: str) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise CampaignCollectionError(
            f"cannot digest artifact source {path!r}: {exc}") from exc
    return "sha256:" + digest.hexdigest()


def _copy_stable_regular_file(
        source: str, destination: str, *,
        maximum_bytes: int = MAX_JSON_BYTES) -> None:
    """Copy one no-symlink file and reject concurrent source mutation."""
    try:
        initial = os.lstat(source)
    except OSError as exc:
        raise CampaignCollectionError(
            f"cannot snapshot artifact file {source!r}: {exc}") from exc
    if not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1:
        raise CampaignCollectionError(
            f"artifact evidence must be a singly-linked regular file: {source!r}")
    if initial.st_size > maximum_bytes:
        raise CampaignCollectionError(
            f"artifact evidence exceeds {maximum_bytes} bytes: {source!r}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) \
        | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise CampaignCollectionError(
            f"cannot snapshot artifact file {source!r}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 \
                or before.st_size > maximum_bytes:
            raise CampaignCollectionError(
                f"artifact evidence is not a bounded regular file: {source!r}")
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with os.fdopen(os.dup(descriptor), "rb") as source_handle, \
                open(destination, "wb") as destination_handle:
            copied = 0
            while True:
                block = source_handle.read(min(
                    1024 * 1024, maximum_bytes + 1 - copied))
                if not block:
                    break
                copied += len(block)
                if copied > maximum_bytes:
                    raise CampaignCollectionError(
                        f"artifact evidence grew beyond {maximum_bytes} "
                        f"bytes: {source!r}")
                destination_handle.write(block)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = os.lstat(source)
    except OSError as exc:
        raise CampaignCollectionError(
            f"artifact evidence changed during snapshot: {source!r}") from exc
    identity = lambda item: (
        item.st_dev, item.st_ino, item.st_size,
        item.st_mtime_ns, item.st_ctime_ns)
    if identity(initial) != identity(before) \
            or identity(before) != identity(after) \
            or identity(after) != identity(current):
        raise CampaignCollectionError(
            f"artifact evidence changed during snapshot: {source!r}")


def _retained_file_identity(path: str) -> tuple[int, int, int, int, int]:
    try:
        item = os.lstat(path)
    except OSError as exc:
        raise CampaignCollectionError(
            f"cannot retain artifact evidence identity {path!r}: {exc}") from exc
    if not stat.S_ISREG(item.st_mode):
        raise CampaignCollectionError(
            f"artifact evidence must be a regular file: {path!r}")
    return (
        item.st_dev, item.st_ino, item.st_size,
        item.st_mtime_ns, item.st_ctime_ns,
    )


def _recheck_retained_files(
        retained: Sequence[tuple[str, tuple[int, int, int, int, int]]]) -> None:
    for path, expected in retained:
        if _retained_file_identity(path) != expected:
            raise CampaignCollectionError(
                f"artifact evidence changed after copying: {path!r}")


def _snapshot_json_directory(
        source_artifact: str, destination_artifact: str, dirname: str) \
        -> list[tuple[str, tuple[int, int, int, int, int]]]:
    source = os.path.join(source_artifact, dirname)
    if not os.path.lexists(source):
        return []
    try:
        metadata = os.lstat(source)
    except OSError as exc:
        raise CampaignCollectionError(
            f"cannot inspect artifact directory {source!r}: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise CampaignCollectionError(
            f"artifact evidence directory is not regular: {source!r}")
    before = sorted(os.listdir(source))
    destination = os.path.join(destination_artifact, dirname)
    os.makedirs(destination, exist_ok=True)
    retained = []
    for name in before:
        path = os.path.join(source, name)
        if not name.endswith(".json"):
            raise CampaignCollectionError(
                f"artifact replay directory has a non-JSON entry: {path!r}")
        retained.append((path, _retained_file_identity(path)))
        _copy_stable_regular_file(path, os.path.join(destination, name))
    if before != sorted(os.listdir(source)):
        raise CampaignCollectionError(
            f"artifact replay directory changed during snapshot: {source!r}")
    _recheck_retained_files(retained)
    return retained


def _snapshot_artifact(
        source_artifact: str, destination_parent: str, track: str) -> str:
    """Freeze all scientific evidence used by one certification pass."""
    try:
        metadata = os.lstat(source_artifact)
    except OSError as exc:
        raise CampaignCollectionError(
            f"cannot inspect execution artifact {source_artifact!r}: {exc}") \
            from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise CampaignCollectionError(
            "execution artifact must be a real directory")
    before = sorted(os.listdir(source_artifact))
    destination = os.path.join(
        destination_parent, os.path.basename(source_artifact))
    os.makedirs(destination, exist_ok=False)
    common = (
        "checkpoint.json", "results.json", "corpus_manifest.json",
        "corpus_panels.json", "control_manifest.json")
    track_files = (
        ("predicates.py", "predicates_log.md", "pricing_contract.json",
         "pending_checkpoint.json")
        if track == "UNRESTRICTED" else
        ("artifact_binding.json", "promoted_cones.json"))
    retained = []
    for name in (*common, *track_files):
        source = os.path.join(source_artifact, name)
        if os.path.lexists(source):
            retained.append((source, _retained_file_identity(source)))
            maximum_bytes = (
                MAX_PREDICATE_SOURCE_BYTES if name == "predicates.py" else
                MAX_PREDICATE_LOG_BYTES if name == "predicates_log.md" else
                MAX_JSON_BYTES)
            _copy_stable_regular_file(
                source, os.path.join(destination, name),
                maximum_bytes=maximum_bytes)
    if track == "SEMANTIC-PURE":
        for dirname in ("replay_specs", "replay_receipts"):
            retained.extend(_snapshot_json_directory(
                source_artifact, destination, dirname))
    if before != sorted(os.listdir(source_artifact)):
        raise CampaignCollectionError(
            "execution artifact changed during certification snapshot")
    _recheck_retained_files(retained)
    return destination


def _artifact_json_list(
        artifact: str, filename: str, description: str) -> list[Any]:
    path = os.path.join(artifact, filename)
    if os.path.islink(path) or not os.path.isfile(path):
        raise CampaignCollectionError(
            f"execution artifact lacks regular {description}: {path!r}")
    value = _load_json(path, description)
    if not isinstance(value, list):
        raise CampaignCollectionError(
            f"execution artifact {description} must be a list: {path!r}")
    return value


def _artifact_json_directory(
        artifact: str, dirname: str, expected_oids: Sequence[str],
        description: str) -> list[dict[str, Any]]:
    """Load exactly one regular JSON document per expected opaque ID."""
    directory = os.path.join(artifact, dirname)
    expected_names = {f"{oid}.json" for oid in expected_oids}
    if not os.path.lexists(directory):
        if expected_names:
            raise CampaignCollectionError(
                f"execution artifact lacks {description} directory")
        return []
    if os.path.islink(directory) or not os.path.isdir(directory):
        raise CampaignCollectionError(
            f"execution artifact {description} path must be a regular directory")
    try:
        names = set(os.listdir(directory))
    except OSError as exc:
        raise CampaignCollectionError(
            f"cannot inspect artifact {description} directory: {exc}") from exc
    if names != expected_names:
        raise CampaignCollectionError(
            f"execution artifact {description} files differ from solved IDs")
    values: list[dict[str, Any]] = []
    for oid in sorted(expected_oids):
        path = os.path.join(directory, f"{oid}.json")
        if os.path.islink(path) or not os.path.isfile(path):
            raise CampaignCollectionError(
                f"artifact {description} must contain regular JSON files")
        value = _load_json(path, f"semantic {description} for {oid}")
        if not isinstance(value, dict):
            raise CampaignCollectionError(
                f"semantic {description} for {oid} must be an object")
        values.append(value)
    return values


def _build_artifact_certification(
        report: Mapping[str, Any], arm: Mapping[str, Any], *,
        artifact_kind: str, checkpoint_digest: str, results_digest: str,
        scientific_source_kind: str, scientific_source_digest: str,
        replay_receipts_digest: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": ARTIFACT_CERTIFICATION_SCHEMA,
        "arm_id": arm["arm_id"],
        "execution_tag": arm["execution_tag"],
        "track": arm["track"],
        "artifact_kind": artifact_kind,
        "report_digest": semantic_replay.canonical_json_digest(report),
        "checkpoint_digest": checkpoint_digest,
        "results_digest": results_digest,
        "scientific_source_kind": scientific_source_kind,
        "scientific_source_digest": scientific_source_digest,
        "replay_receipts_digest": replay_receipts_digest,
    }
    body["certification_digest"] = \
        semantic_replay.canonical_json_digest(body)
    return body


def _load_protocol_evidence(
        artifact: str, arm: Mapping[str, Any],
        preregistration: Mapping[str, Any]) \
        -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    manifest = _artifact_json(
        artifact, "corpus_manifest.json", "corpus manifest")
    bundle = _artifact_json(
        artifact, "corpus_panels.json", "corpus panel bundle")
    try:
        phase_d_protocol.validate_corpus_manifest(manifest)
        phase_d_protocol.validate_corpus_bundle(bundle, manifest)
    except phase_d_protocol.PhaseDProtocolError as exc:
        raise CampaignCollectionError(
            f"execution artifact corpus evidence is invalid: {exc}") from exc
    panel_digests = [
        entry["panel_set_digest"] for entry in manifest["problems"]]
    if manifest["corpus_digest"] != preregistration["corpus_digest"] \
            or manifest["problem_count"] != \
            preregistration["corpus_problem_count"] \
            or panel_digests != preregistration["corpus_panel_set_digests"]:
        raise CampaignCollectionError(
            "execution artifact corpus differs from the preregistration")

    control_path = os.path.join(artifact, "control_manifest.json")
    control: dict[str, Any] | None = None
    if arm["condition"] == phase_d_protocol.SHUFFLED_SIDES:
        control = _artifact_json(
            artifact, "control_manifest.json", "control manifest")
        try:
            phase_d_protocol.validate_shuffled_control_manifest(
                control, manifest)
        except phase_d_protocol.PhaseDProtocolError as exc:
            raise CampaignCollectionError(
                f"execution artifact control evidence is invalid: {exc}") from exc
        controlled_panels = [
            entry["controlled_panel_set_digest"]
            for entry in control["problems"]]
        expected_control = next(
            item for item in preregistration["shuffled_sides"]["controls"]
            if item["replicate"] == arm["replicate"])
        if control["control_digest"] != arm["control_digest"] \
                or control["replicate"] != arm["replicate"] \
                or controlled_panels != expected_control["panel_set_digests"]:
            raise CampaignCollectionError(
                "execution artifact control differs from its preregistered arm")
    elif os.path.lexists(control_path):
        raise CampaignCollectionError(
            "non-shuffled execution artifact carries a control manifest")
    return manifest, bundle, control


def _exact_records_equal(
        report: Mapping[str, Any], checkpoint_records: Sequence[Mapping[str, Any]],
        *, scale: int) -> None:
    try:
        runner_records = phase_d_protocol.runner_records_from_track_report(report)
    except phase_d_protocol.PhaseDProtocolError as exc:
        raise CampaignCollectionError(
            f"cannot reverse report-only record stamps: {exc}") from exc
    prefix = list(checkpoint_records[:scale])
    if semantic_replay.canonical_json_bytes(runner_records) != \
            semantic_replay.canonical_json_bytes(prefix):
        raise CampaignCollectionError(
            "track report records differ from the originating runner checkpoint")


def _validate_grown_execution_binding(
        checkpoint_binding: object, checkpoint_history: object,
        arm: Mapping[str, Any], preregistration: Mapping[str, Any]) -> None:
    """Accept the completed same-family artifact backing a smaller report."""
    family_arms = sorted(
        (
            candidate for candidate in preregistration["arms"]
            if candidate["track"] == arm["track"]
            and candidate["condition"] == arm["condition"]
            and candidate["replicate"] == arm["replicate"]
            and candidate["control_digest"] == arm["control_digest"]
            and candidate["execution_tag"] == arm["execution_tag"]
        ),
        key=lambda candidate: candidate["scale"],
    )
    family_bindings = [
        phase_d_protocol.execution_binding(
            preregistration, candidate["arm_id"])
        for candidate in family_arms
    ]
    matching = [
        (index, binding)
        for index, binding in enumerate(family_bindings)
        if binding == checkpoint_binding and binding["scale"] >= arm["scale"]
    ]
    if len(matching) != 1:
        raise CampaignCollectionError(
            "execution checkpoint is not bound to this or a larger "
            "completed same-family scale")
    index, _ = matching[0]
    if checkpoint_history != family_bindings[:index + 1]:
        raise CampaignCollectionError(
            "execution checkpoint Phase binding history differs")


def _load_unrestricted_checkpoint(
        artifact: str, arm: Mapping[str, Any],
        preregistration: Mapping[str, Any]):
    """Lazy-load and fully replay one unrestricted execution artifact."""
    try:
        try:
            from . import bongard_legs
        except ImportError:  # pragma: no cover - direct script usage
            import bongard_legs  # type: ignore
        if os.path.lexists(os.path.join(
                artifact, bongard_legs.PENDING_CHECKPOINT_FILE)):
            raise CampaignCollectionError(
                "unrestricted artifact has an incomplete staged promotion")
        _require_regular_artifact_file(
            artifact, bongard_legs.LIBRARY_FILE, "predicate source")
        _require_regular_artifact_file(
            artifact, bongard_legs.LOG_FILE, "predicate source log",
            required=True)
        checkpoint_json = _artifact_json(
            artifact, "checkpoint.json", "unrestricted checkpoint")
        manifest, bundle, control = _load_protocol_evidence(
            artifact, arm, preregistration)
        results = _artifact_json(
            artifact, "results.json", "unrestricted results")
        checkpoint = bongard_legs._load_checkpoint(artifact)
        if checkpoint is None:
            raise CampaignCollectionError(
                "unrestricted execution artifact has no checkpoint")
        if checkpoint_json != checkpoint.to_json():
            raise CampaignCollectionError(
                "unrestricted checkpoint does not round-trip exactly")
        raw_condition = (
            phase_d_protocol.OBSERVED
            if arm["condition"] == "primary" else arm["condition"])
        if checkpoint.tag != arm["execution_tag"] \
                or checkpoint.track != "UNRESTRICTED" \
                or checkpoint.condition != raw_condition \
                or checkpoint.label_policy != arm["label_policy"] \
                or checkpoint.sharing_policy != arm["sharing_policy"] \
                or checkpoint.corpus_digest != preregistration["corpus_digest"] \
                or checkpoint.control_digest != arm["control_digest"]:
            raise CampaignCollectionError(
                "unrestricted checkpoint identity differs from its arm/tag")
        _validate_grown_execution_binding(
            checkpoint.phase_execution_binding,
            checkpoint.phase_execution_binding_history,
            arm, preregistration)
        if any(
                receipt.get("source") != "codex-cli"
                for record in checkpoint.records
                for receipt in record.proposer_receipts):
            raise CampaignCollectionError(
                "unrestricted Phase artifact contains a test-injected receipt")
        rebuilt_results = bongard_legs._reconcile_results(
            checkpoint, results, corpus_manifest=manifest)
        if semantic_replay.canonical_json_bytes(results) != \
                semantic_replay.canonical_json_bytes(rebuilt_results):
            raise CampaignCollectionError(
                "unrestricted results differ from checkpoint evidence")
        replay_receipt = {
            "validator": "bongard_legs._load_checkpoint/cold-replay",
            "status": "PASS",
            "source_trace_digest": "sha256:" + checkpoint.source_trace_digest,
            "record_count": len(checkpoint.records),
        }
        materials = {
            "checkpoint_digest":
                semantic_replay.canonical_json_digest(checkpoint_json),
            "results_digest":
                semantic_replay.canonical_json_digest(results),
            "scientific_source_kind": bongard_legs.LIBRARY_FILE,
            "scientific_source_digest": _raw_file_digest(
                os.path.join(artifact, bongard_legs.LIBRARY_FILE)),
            "replay_receipts_digest":
                semantic_replay.canonical_json_digest([replay_receipt]),
        }
        return (bongard_legs, checkpoint, manifest, bundle, control,
                materials)
    except CampaignCollectionError:
        raise
    except Exception as exc:
        raise CampaignCollectionError(
            f"unrestricted execution artifact failed replay certification: {exc}") \
            from exc


def _certify_unrestricted_report(
        report: Mapping[str, Any], artifact: str, arm: Mapping[str, Any],
        preregistration: Mapping[str, Any]) \
        -> tuple[dict[str, Any], list[dict[str, Any]]]:
    module, checkpoint, _, _, _, materials = _load_unrestricted_checkpoint(
        artifact, arm, preregistration)
    scale = arm["scale"]
    if len(checkpoint.records) < scale \
            or (arm["condition"] == phase_d_protocol.NO_SHARE
                and len(checkpoint.records) != scale):
        raise CampaignCollectionError(
            "unrestricted checkpoint does not contain the exact report prefix")
    _exact_records_equal(
        report, [asdict(record) for record in checkpoint.records], scale=scale)
    if arm["condition"] != phase_d_protocol.NO_SHARE:
        certification = _build_artifact_certification(
            report, arm, artifact_kind="unrestricted-predicate-library",
            **materials)
        return certification, [asdict(record) for record in checkpoint.records]

    primary_arm = next(
        item for item in preregistration["arms"]
        if item["track"] == arm["track"]
        and item["condition"] == "primary"
        and item["scale"] == scale)
    artifact_parent = os.path.dirname(artifact)
    source_artifact = os.path.join(
        artifact_parent, primary_arm["execution_tag"] + "_predicates")
    _, source_checkpoint, _, _, _, _ = _load_unrestricted_checkpoint(
        source_artifact, primary_arm, preregistration)
    if len(source_checkpoint.records) < scale:
        raise CampaignCollectionError(
            "no-share source checkpoint lacks its complete primary prefix")
    try:
        repriced = module.reprice_no_share(
            source_checkpoint, tag=arm["execution_tag"], max_problems=scale,
            phase_execution_binding=phase_d_protocol.execution_binding(
                preregistration, arm["arm_id"]))
    except Exception as exc:
        raise CampaignCollectionError(
            f"no-share artifact cannot be reproduced by exact repricing: {exc}") \
            from exc
    if semantic_replay.canonical_json_bytes(repriced.to_json()) != \
            semantic_replay.canonical_json_bytes(checkpoint.to_json()):
        raise CampaignCollectionError(
            "no-share checkpoint is not the exact primary-prefix reprice")
    certification = _build_artifact_certification(
        report, arm, artifact_kind="unrestricted-no-share-reprice",
        **materials)
    return certification, [asdict(record) for record in checkpoint.records]


def _fresh_replay_semantic_receipts(
        semantic_artifacts: Any, artifact: str, checkpoint: dict[str, Any],
        promoted_cones: list[Any], manifest: dict[str, Any],
        bundle: dict[str, Any], control: dict[str, Any] | None,
        solved_oids: set[str]) -> list[dict[str, Any]]:
    promoted_oids = {
        item.get("opaque_id") for item in promoted_cones
        if isinstance(item, Mapping)}
    if len(promoted_oids) != len(promoted_cones) \
            or promoted_oids != solved_oids:
        raise CampaignCollectionError(
            "semantic promoted cones differ from solved checkpoint records")
    ordered_oids = sorted(solved_oids)
    specs = _artifact_json_directory(
        artifact, "replay_specs", ordered_oids, "replay specs")
    stored_receipts = _artifact_json_directory(
        artifact, "replay_receipts", ordered_oids, "replay receipts")
    if not ordered_oids:
        return []

    with tempfile.TemporaryDirectory(prefix="bongard-phase-d-replay-") as temp:
        temp_specs = os.path.join(temp, "replay_specs")
        os.makedirs(temp_specs)
        for oid, _spec in zip(ordered_oids, specs):
            # The source was already established as a regular, strict JSON
            # file.  Replay a private copy so issuing fresh receipts cannot
            # mutate the collected artifact.
            source = os.path.join(
                artifact, "replay_specs", f"{oid}.json")
            destination = os.path.join(temp_specs, f"{oid}.json")
            shutil.copy2(source, destination)
        generated = semantic_artifacts._cold_replay_specs(
            temp, promoted_cones, checkpoint,
            corpus_manifest=manifest,
            corpus_bundle=bundle,
            control_manifest=control,
        )
    if semantic_replay.canonical_json_bytes(generated) != \
            semantic_replay.canonical_json_bytes(stored_receipts):
        raise CampaignCollectionError(
            "fresh semantic replay receipts differ from stored artifact receipts")
    return generated


def _certify_semantic_report(
        report: Mapping[str, Any], artifact: str, arm: Mapping[str, Any],
        preregistration: Mapping[str, Any]) \
        -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Lazy-load semantic validation without making proposer/API calls."""
    try:
        try:
            from . import semantic_artifacts
        except ImportError:  # pragma: no cover - direct script usage
            import semantic_artifacts  # type: ignore
        checkpoint = _artifact_json(
            artifact, "checkpoint.json", "semantic checkpoint")
        results = _artifact_json(
            artifact, "results.json", "semantic results")
        manifest, bundle, control = _load_protocol_evidence(
            artifact, arm, preregistration)
        raw_condition = (
            phase_d_protocol.OBSERVED
            if arm["condition"] == "primary" else arm["condition"])
        policy = preregistration["execution_policy"]["semantic_pure"]
        preregistered_runtime = preregistration["execution_policy"]["runtime"]
        expected_hash_runtime = {
            "python_hash_seed_env": preregistered_runtime[
                "python_hash_seed_env"],
            "python_hash_probes": preregistered_runtime[
                "python_hash_probes"],
        }
        verifier = checkpoint.get("verifier_policy")
        selection = checkpoint.get("selection")
        expected_verifier = {
            "max_support_errors": policy["max_support_errors"],
            "max_threshold_loo_errors": policy["max_loo_errors"],
            "max_pair_threshold_loo_errors": policy["max_rotated_loo_errors"],
            "exact_status_requires_all_three_zero": True,
            "representation_selection_scope": "all_12_labeled_panels",
        }
        if checkpoint.get("artifact_state") not in {"RUN_COMPLETE", "PROMOTED"} \
                or checkpoint.get("runner") != "semantic_cone" \
                or checkpoint.get("active_track") != "SEMANTIC-PURE" \
                or checkpoint.get("tag") != arm["execution_tag"] \
                or checkpoint.get("condition") != raw_condition \
                or checkpoint.get("sharing_policy") != arm["sharing_policy"] \
                or checkpoint.get("proposer") != policy["proposer"] \
                or checkpoint.get("model") != policy["model"] \
                or checkpoint.get("max_tokens") != policy["max_tokens"] \
                or checkpoint.get("rounds") != policy["rounds"] \
                or checkpoint.get("phase_python_hash_runtime") != \
                expected_hash_runtime \
                or verifier != expected_verifier \
                or not isinstance(selection, dict) \
                or selection.get("method") != policy["selection_method"] \
                or selection.get("lambda") != policy["lambda"] \
                or selection.get("risk_fields") != policy["selection_risk_fields"] \
                or selection.get("unmeasured_risks") != policy[
                    "selection_unmeasured_risks"]:
            raise CampaignCollectionError(
                "semantic checkpoint policy/arm identity differs from preregistration")
        _validate_grown_execution_binding(
            checkpoint.get("phase_execution_binding"),
            checkpoint.get("phase_execution_binding_history"),
            arm, preregistration)
        expected_binding = semantic_artifacts._artifact_binding(
            checkpoint, manifest, bundle, control)
        if not semantic_artifacts._check_artifact_binding(
                artifact, expected_binding):
            raise CampaignCollectionError(
                "semantic artifact lacks its committed arm binding")
        validated = semantic_artifacts._validate_run_inputs(
            checkpoint, results, manifest, bundle, control,
            require_complete=True)
        checkpoint_records = checkpoint.get("records")
        if not isinstance(checkpoint_records, list) \
                or len(checkpoint_records) < arm["scale"]:
            raise CampaignCollectionError(
                "semantic checkpoint does not contain the report prefix")
        _exact_records_equal(
            report, checkpoint_records, scale=arm["scale"])
        promoted_cones = _artifact_json_list(
            artifact, "promoted_cones.json", "promoted cone evidence")
        solved_oids = set(validated["solved_oids"])
        if solved_oids and checkpoint.get("artifact_state") != "PROMOTED":
            raise CampaignCollectionError(
                "solved semantic checkpoint lacks promoted replay evidence")
        receipts = _fresh_replay_semantic_receipts(
            semantic_artifacts, artifact, checkpoint, promoted_cones,
            manifest, bundle, control, solved_oids)
        certification = _build_artifact_certification(
            report, arm, artifact_kind="semantic-typed-cones",
            checkpoint_digest=
                semantic_replay.canonical_json_digest(checkpoint),
            results_digest=semantic_replay.canonical_json_digest(results),
            scientific_source_kind="promoted_cones.json",
            scientific_source_digest=
                semantic_replay.canonical_json_digest(promoted_cones),
            replay_receipts_digest=
                semantic_replay.canonical_json_digest(receipts),
        )
        return certification, copy.deepcopy(checkpoint_records)
    except CampaignCollectionError:
        raise
    except Exception as exc:
        raise CampaignCollectionError(
            f"semantic execution artifact failed replay certification: {exc}") \
            from exc


def _certify_report_origin(
        report: Mapping[str, Any], origin: str,
        preregistration: Mapping[str, Any]) \
        -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    arm = _matching_arm(report, preregistration)
    artifact = _expected_artifact_from_origin(origin, arm)
    with tempfile.TemporaryDirectory(
            prefix="bongard-phase-d-artifact-") as temporary:
        snapshot_parent = os.path.join(temporary, "agent_solutions")
        os.makedirs(snapshot_parent)
        snapshot = _snapshot_artifact(
            artifact, snapshot_parent, arm["track"])
        if arm["condition"] == phase_d_protocol.NO_SHARE:
            primary_arm = next(
                item for item in preregistration["arms"]
                if item["track"] == arm["track"]
                and item["condition"] == "primary"
                and item["scale"] == arm["scale"])
            source_artifact = os.path.join(
                os.path.dirname(artifact),
                primary_arm["execution_tag"] + "_predicates")
            _snapshot_artifact(
                source_artifact, snapshot_parent, "UNRESTRICTED")
        if arm["track"] == "UNRESTRICTED":
            certification, checkpoint_records = _certify_unrestricted_report(
                report, snapshot, arm, preregistration)
        else:
            certification, checkpoint_records = _certify_semantic_report(
                report, snapshot, arm, preregistration)
    return artifact, certification, checkpoint_records


def discover_track_reports(
        report_dirs: Sequence[str],
        preregistration: Mapping[str, Any],
        *,
        excluded_paths: Sequence[str] = ()) -> list[DiscoveredTrackReport]:
    """Load direct-child JSON reports from explicit directories only.

    Report filenames are not trusted as arm identities; the validated
    ``arm_id`` inside each document is authoritative and must reproduce the
    runner's canonical filename.  Every direct ``.json`` file (apart from
    explicitly excluded inputs/outputs) must be a v7 track report.
    Subdirectories are intentionally never traversed.
    """
    try:
        phase_d_protocol.validate_preregistration(preregistration)
    except phase_d_protocol.PhaseDProtocolError as exc:
        raise CampaignCollectionError(
            f"invalid Phase D preregistration: {exc}") from exc
    directories = _normalise_report_dirs(report_dirs)
    excluded = {
        os.path.realpath(os.path.abspath(path))
        for path in excluded_paths
    }
    reports: list[DiscoveredTrackReport] = []
    origins: dict[str, str] = {}
    certified_artifacts: dict[
        str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for directory in directories:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as exc:
            raise CampaignCollectionError(
                f"cannot inspect report directory {directory!r}: {exc}") from exc
        for entry in entries:
            if not entry.name.endswith(".json"):
                continue
            path = os.path.abspath(entry.path)
            if os.path.realpath(path) in excluded:
                continue
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise CampaignCollectionError(
                    f"report JSON must be a direct regular file: {path!r}")
            value = _load_json(path, "Phase D track report")
            if not isinstance(value, dict) or value.get("schema") != \
                    phase_d_protocol.TRACK_REPORT_SCHEMA:
                raise CampaignCollectionError(
                    f"JSON file is not an exact Phase D v7 track report: {path!r}")
            try:
                phase_d_protocol.validate_track_report(
                    value, preregistration, _preregistration_validated=True)
            except phase_d_protocol.PhaseDProtocolError as exc:
                raise CampaignCollectionError(
                    f"invalid Phase D track report {path!r}: {exc}") from exc
            arm_id = value["arm_id"]
            previous = origins.get(arm_id)
            if previous is not None:
                raise CampaignCollectionError(
                    f"duplicate track report for arm {arm_id!r}: "
                    f"{previous!r} and {path!r}")
            origins[arm_id] = path
            arm = _matching_arm(value, preregistration)
            artifact = _expected_artifact_from_origin(path, arm)
            cached = certified_artifacts.get(artifact)
            if cached is None:
                artifact, certification, checkpoint_records = \
                    _certify_report_origin(value, path, preregistration)
                certified_artifacts[artifact] = (
                    certification, checkpoint_records)
            else:
                template, checkpoint_records = cached
                _exact_records_equal(
                    value, checkpoint_records, scale=arm["scale"])
                certification = _build_artifact_certification(
                    value, arm,
                    artifact_kind=template["artifact_kind"],
                    checkpoint_digest=template["checkpoint_digest"],
                    results_digest=template["results_digest"],
                    scientific_source_kind=
                        template["scientific_source_kind"],
                    scientific_source_digest=
                        template["scientific_source_digest"],
                    replay_receipts_digest=
                        template["replay_receipts_digest"],
                )
            reports.append(DiscoveredTrackReport(
                report=value, origin=path, artifact_dir=artifact,
                certification=certification))
    return reports


def _ordered_complete_reports(
        reports: Sequence[Mapping[str, Any]],
        preregistration: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the campaign and return reports in preregistered arm order."""
    try:
        phase_d_protocol.validate_complete_report_collection(
            reports, preregistration)
    except phase_d_protocol.PhaseDProtocolError as exc:
        raise CampaignCollectionError(
            f"incomplete or inconsistent Phase D report collection: {exc}") from exc
    by_arm = {report["arm_id"]: report for report in reports}
    return [
        copy.deepcopy(dict(by_arm[arm["arm_id"]]))
        for arm in preregistration["arms"]
    ]


def _aggregate_counts(
        ordered_reports: Sequence[Mapping[str, Any]],
        preregistration: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Count outcomes within, and only within, track/condition/scale cells."""
    by_arm = {report["arm_id"]: report for report in ordered_reports}
    cells: dict[tuple[str, str, int], dict[str, Any]] = {}
    order: list[tuple[str, str, int]] = []
    for arm in preregistration["arms"]:
        key = (arm["track"], arm["condition"], arm["scale"])
        if key not in cells:
            cells[key] = {
                "track": arm["track"],
                "condition": arm["condition"],
                "scale": arm["scale"],
                "arm_count": 0,
                "attempted": 0,
                "solved": 0,
                "unsolved": 0,
                "ordinary_unsolved": 0,
                "verifier_failures": 0,
            }
            order.append(key)
        report = by_arm[arm["arm_id"]]
        cell = cells[key]
        cell["arm_count"] += 1
        cell["attempted"] += report["attempted"]
        cell["solved"] += report["solved"]
        cell["unsolved"] += report["attempted"] - report["solved"]
        verifier_failures = sum(
            record.get("status") == "VERIFIER_FAILURE_UNRESTRICTED"
            for record in report["records"])
        cell["verifier_failures"] += verifier_failures
        cell["ordinary_unsolved"] += (
            report["attempted"] - report["solved"] - verifier_failures)
    return [cells[key] for key in order]


def _validate_cross_arm_codex_turn_uniqueness(
        reports: Sequence[Mapping[str, Any]]) -> None:
    """Reject receipt transplants across independently adaptive Codex arms.

    Held-fixed no-share reports intentionally reuse their primary receipts and
    are excluded because they do not launch proposer turns.
    """
    thread_owners: dict[str, str] = {}
    event_stream_owners: dict[str, str] = {}
    for report in reports:
        if report.get("track") != "UNRESTRICTED" \
                or report.get("sharing_policy") != phase_d_protocol.SHARED:
            continue
        execution_tag = report.get("execution_tag")
        for record in report.get("records", []):
            for receipt in record.get("proposer_receipts", []):
                thread_id = receipt.get("thread_id")
                event_digest = receipt.get("event_stream_digest")
                thread_owner = thread_owners.setdefault(
                    thread_id, execution_tag)
                event_owner = event_stream_owners.setdefault(
                    event_digest, execution_tag)
                if thread_owner != execution_tag \
                        or event_owner != execution_tag:
                    raise CampaignCollectionError(
                        "independent unrestricted arms reuse Codex turn "
                        "identity evidence")


def _ordered_artifact_certifications(
        certifications: Sequence[Mapping[str, Any]],
        reports: Sequence[Mapping[str, Any]],
        preregistration: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(certifications, Sequence) \
            or isinstance(certifications, (str, bytes)):
        raise CampaignCollectionError(
            "artifact certifications must be a sequence")
    by_arm: dict[str, Mapping[str, Any]] = {}
    report_by_arm = {report["arm_id"]: report for report in reports}
    tag_evidence: dict[str, tuple[Any, ...]] = {}
    for index, certification in enumerate(certifications):
        if not isinstance(certification, Mapping):
            raise CampaignCollectionError(
                f"artifact_certifications[{index}] must be an object")
        keys = set(certification)
        if keys != _ARTIFACT_CERTIFICATION_KEYS:
            raise CampaignCollectionError(
                f"artifact certification keys differ at index {index}")
        if certification["schema"] != ARTIFACT_CERTIFICATION_SCHEMA:
            raise CampaignCollectionError(
                "unsupported artifact certification schema")
        arm_id = certification.get("arm_id")
        if not isinstance(arm_id, str) or arm_id in by_arm:
            raise CampaignCollectionError(
                "artifact certifications have duplicate/malformed arm IDs")
        by_arm[arm_id] = certification

    ordered: list[dict[str, Any]] = []
    for arm in preregistration["arms"]:
        certification = by_arm.get(arm["arm_id"])
        report = report_by_arm.get(arm["arm_id"])
        if certification is None or report is None:
            raise CampaignCollectionError(
                "artifact certifications do not cover the complete arm table")
        expected_kind = (
            "semantic-typed-cones"
            if arm["track"] == "SEMANTIC-PURE" else
            "unrestricted-no-share-reprice"
            if arm["condition"] == phase_d_protocol.NO_SHARE else
            "unrestricted-predicate-library")
        expected_source_kind = (
            "promoted_cones.json"
            if arm["track"] == "SEMANTIC-PURE" else "predicates.py")
        if certification["execution_tag"] != arm["execution_tag"] \
                or certification["track"] != arm["track"] \
                or certification["artifact_kind"] != expected_kind \
                or certification["scientific_source_kind"] != \
                expected_source_kind:
            raise CampaignCollectionError(
                "artifact certification identity differs from its arm")
        digest_fields = (
            "report_digest", "checkpoint_digest", "results_digest",
            "scientific_source_digest", "replay_receipts_digest",
            "certification_digest",
        )
        if any(not _is_digest(certification.get(name))
               for name in digest_fields):
            raise CampaignCollectionError(
                "artifact certification contains a malformed digest")
        if certification["report_digest"] != \
                semantic_replay.canonical_json_digest(report):
            raise CampaignCollectionError(
                "artifact certification report digest does not reproduce")
        body = {
            key: value for key, value in certification.items()
            if key != "certification_digest"}
        if semantic_replay.canonical_json_digest(body) != \
                certification["certification_digest"]:
            raise CampaignCollectionError(
                "artifact certification digest does not reproduce")
        stable_evidence = tuple(
            certification[name] for name in (
                "track", "artifact_kind", "checkpoint_digest",
                "results_digest", "scientific_source_kind",
                "scientific_source_digest", "replay_receipts_digest"))
        prior = tag_evidence.setdefault(
            certification["execution_tag"], stable_evidence)
        if prior != stable_evidence:
            raise CampaignCollectionError(
                "one execution tag has inconsistent artifact evidence")
        ordered.append(copy.deepcopy(dict(certification)))
    if set(by_arm) != {arm["arm_id"] for arm in preregistration["arms"]}:
        raise CampaignCollectionError(
            "artifact certifications contain unregistered arms")
    return ordered


def build_campaign_artifact(
        preregistration: Mapping[str, Any],
        reports: Sequence[Mapping[str, Any]],
        artifact_certifications: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a self-identifying, digest-bound complete campaign document."""
    try:
        phase_d_protocol.validate_preregistration(preregistration)
    except phase_d_protocol.PhaseDProtocolError as exc:
        raise CampaignCollectionError(
            f"invalid Phase D preregistration: {exc}") from exc
    ordered = _ordered_complete_reports(reports, preregistration)
    _validate_cross_arm_codex_turn_uniqueness(ordered)
    ordered_certifications = _ordered_artifact_certifications(
        artifact_certifications, ordered, preregistration)
    body: dict[str, Any] = {
        "schema": CAMPAIGN_SCHEMA,
        "preregistration_schema": phase_d_protocol.PREREGISTRATION_SCHEMA,
        "preregistration_digest": preregistration["preregistration_digest"],
        "corpus_digest": preregistration["corpus_digest"],
        "corpus_problem_count": preregistration["corpus_problem_count"],
        "corpus_panel_set_digests": copy.deepcopy(
            preregistration["corpus_panel_set_digests"]),
        "track_report_schema": phase_d_protocol.TRACK_REPORT_SCHEMA,
        "arm_count": len(ordered),
        "reports": ordered,
        "artifact_certifications": ordered_certifications,
        "aggregates": _aggregate_counts(ordered, preregistration),
    }
    body["campaign_digest"] = semantic_replay.canonical_json_digest(body)
    return body


def _is_digest(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    suffix = value[7:]
    return len(suffix) == 64 and all(char in "0123456789abcdef" for char in suffix)


def validate_campaign_artifact(
        value: Mapping[str, Any],
        preregistration: Mapping[str, Any]) -> None:
    """Reconstruct a campaign artifact exactly, including its aggregate cells."""
    if not isinstance(value, Mapping):
        raise CampaignCollectionError("campaign artifact must be an object")
    if set(value) != _CAMPAIGN_KEYS:
        missing = sorted(_CAMPAIGN_KEYS - set(value))
        extra = sorted(set(value) - _CAMPAIGN_KEYS)
        raise CampaignCollectionError(
            f"campaign artifact keys differ (missing={missing}, extra={extra})")
    if value.get("schema") != CAMPAIGN_SCHEMA:
        raise CampaignCollectionError("unsupported Phase D campaign schema")
    digest = value.get("campaign_digest")
    if not _is_digest(digest):
        raise CampaignCollectionError("campaign_digest is malformed")
    observed = semantic_replay.canonical_json_digest({
        key: item for key, item in value.items() if key != "campaign_digest"
    })
    if observed != digest:
        raise CampaignCollectionError("campaign digest does not reproduce")
    reports = value.get("reports")
    if not isinstance(reports, list):
        raise CampaignCollectionError("campaign reports must be a list")
    certifications = value.get("artifact_certifications")
    if not isinstance(certifications, list):
        raise CampaignCollectionError(
            "campaign artifact_certifications must be a list")
    expected = build_campaign_artifact(
        preregistration, reports, certifications)
    if semantic_replay.canonical_json_bytes(value) != \
            semantic_replay.canonical_json_bytes(expected):
        raise CampaignCollectionError(
            "campaign identity, report order, or aggregates differ from "
            "the preregistered collection")


def _load_existing_campaign(
        path: str, preregistration: Mapping[str, Any],
        expected: Mapping[str, Any]) -> dict[str, Any]:
    if os.path.islink(path):
        raise CampaignCollectionError(
            f"existing campaign output may not be a symbolic link: {path!r}")
    value = _load_json(path, "existing Phase D campaign")
    if not isinstance(value, dict):
        raise CampaignCollectionError("existing campaign output must be an object")
    validate_campaign_artifact(value, preregistration)
    if semantic_replay.canonical_json_bytes(value) != \
            semantic_replay.canonical_json_bytes(expected):
        raise CampaignCollectionError(
            "existing campaign output differs from the collected campaign")
    return value


def write_campaign_once(
        path: str,
        campaign: Mapping[str, Any],
        preregistration: Mapping[str, Any]) -> dict[str, Any]:
    """Publish complete canonical bytes atomically without overwriting a path."""
    validate_campaign_artifact(campaign, preregistration)
    target = os.path.abspath(path)
    parent = os.path.dirname(target)
    try:
        os.makedirs(parent, exist_ok=True)
    except OSError as exc:
        raise CampaignCollectionError(
            f"cannot create campaign output directory {parent!r}: {exc}") from exc
    if os.path.lexists(target):
        return _load_existing_campaign(target, preregistration, campaign)

    payload = semantic_replay.canonical_json_bytes(campaign) + b"\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb", dir=parent,
                prefix=f".{os.path.basename(target)}.", suffix=".tmp",
                delete=False) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link is an atomic create-if-absent publication of the already
        # fsynced bytes.  Unlike os.replace, it cannot overwrite a concurrent
        # writer's different or corrupt campaign.
        try:
            os.link(temporary, target)
        except FileExistsError:
            return _load_existing_campaign(target, preregistration, campaign)
        directory_fd = os.open(
            parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return dict(campaign)
    except CampaignCollectionError:
        raise
    except OSError as exc:
        raise CampaignCollectionError(
            f"cannot atomically publish Phase D campaign {target!r}: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def collect_campaign(
        preregistration_path: str,
        report_dirs: Sequence[str],
        out_path: str) -> dict[str, Any]:
    """Collect, validate, and atomically publish one complete campaign."""
    prereg_path = os.path.abspath(preregistration_path)
    output_path = os.path.abspath(out_path)
    preregistration = load_preregistration(prereg_path)
    discovered = discover_track_reports(
        report_dirs,
        preregistration,
        excluded_paths=(prereg_path, output_path),
    )
    reports = [item.report for item in discovered]
    certifications = [item.certification for item in discovered]
    campaign = build_campaign_artifact(
        preregistration, reports, certifications)
    return write_campaign_once(output_path, campaign, preregistration)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect a complete offline preregistered Phase D campaign")
    parser.add_argument(
        "--preregistration", required=True,
        help="path to the frozen bongard.phase-d-preregistration/v6 JSON")
    parser.add_argument(
        "--report-dir", action="append", nargs="+", required=True,
        metavar="PATH",
        help=("explicit execution-artifact track_reports directory containing "
              "direct-child v7 reports; "
              "repeat the option or provide multiple paths"),
    )
    parser.add_argument(
        "--out", required=True,
        help="write-once path for the canonical campaign JSON")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _argument_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    report_dirs = [path for group in args.report_dir for path in group]
    try:
        collect_campaign(args.preregistration, report_dirs, args.out)
    except CampaignCollectionError as exc:
        parser.error(str(exc))
    print(os.path.abspath(args.out))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_CERTIFICATION_SCHEMA",
    "CAMPAIGN_SCHEMA",
    "CampaignCollectionError",
    "DiscoveredTrackReport",
    "build_campaign_artifact",
    "collect_campaign",
    "discover_track_reports",
    "load_preregistration",
    "main",
    "parse_args",
    "validate_campaign_artifact",
    "write_campaign_once",
]

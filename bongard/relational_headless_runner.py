"""One-shot headless Codex benchmark for closed Python visual predicates.

This runner is intentionally independent of :mod:`bongard.benchmark`.  The
generic episode planner hashes every panel, and the generic runner verifies
both query sources before proposal.  Here an explicit train/validation task is
exposure-precommitted first, only its twelve support panels are then opened,
and both held-out paths remain unresolved until a support-verified Python
predicate and its Codex receipt have been durably frozen.

The model returns four finite predicate parameters plus one audit-only
rationale, never Python source.  Those parameters construct a
:class:`RelationalVisualQuery`; the canonical Python evaluator is the only
prediction authority.  Point-contact is deliberately disabled in this first
benchmark protocol.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import stat
import tempfile
import unicodedata
import zipfile
from typing import Any, Callable, Mapping, Sequence

from bongard.artifacts import canonical_digest, canonical_json
from bongard.cohorts import classify_task
from bongard.closed_visual_predicates import (
    ClosedPanelPredicate,
    ClosedPredicateKind,
    DirectCountPredicate,
    LIBRARY_ALGORITHM_ID,
    SymmetryMetric,
    SymmetryThresholdPredicate,
    complete_closed_predicate_library_identity,
    evaluate_closed_predicate,
    verify_closed_predicate_result,
)
from bongard.composite_visual_packet import (
    ExactPanelWitnessPacket,
    composite_visual_packet_source_digest,
    exact_panel_witness_extractor_digest,
    extract_exact_panel_witness_packet,
    verify_exact_panel_witness_packet,
)
from bongard.corpus import PNG_SIGNATURE, SplitIndex
from bongard.evidence import Disposition
from bongard.exposure import (
    ExposureLedger,
    semantic_resolver_policy_digest,
)
from bongard.historical_exposure import load_historical_exposure
from bongard.loop_scene_witnesses import (
    LoopScenePacket,
    extract_loop_scene_witnesses,
    loop_scene_catalog_digest,
    loop_scene_extractor_digest,
    verify_loop_scene_packet,
)
from bongard.relational_visual_query import (
    ALLOWED_AREA_RATIOS,
    ALLOWED_OBLIQUENESS_THRESHOLDS_MILLIDEGREES,
    ALLOWED_SIDE_COUNTS,
    PointContactClause,
    Rational,
    RelationalVisualQuery,
    evaluate_relational_query,
    relational_query_algorithm_digest,
    verify_relational_query_result,
)
from bongard.release import OfficialReleaseDescriptor
from bongard.transport import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_REASONING_EFFORT,
    STRUCTURED_INPUT_DIGEST_SCHEMA,
    CloudPolicyCacheSnapshot,
    CodexReceipt,
    CodexStructuredResult,
    ordered_panel_view_digest,
    run_codex_structured,
    semantic_panel_set_digest,
    snapshot_cloud_policy_cache,
    validate_codex_receipt,
    validate_codex_strict_output_schema,
)
from bongard.typed_visual_proposal import TypedDeterministicAtom
from bongard.visual_predicate_catalog import (
    DIRECT_VISUAL_ATOM_CATALOG,
    direct_visual_catalog_digest,
)


PLAN_SCHEMA = "gkm.bongard-relational-headless-plan.v4"
FREEZE_SCHEMA = "gkm.bongard-relational-headless-proposal-freeze.v4"
PREDICTION_SCHEMA = "gkm.bongard-relational-headless-predictions.v4"
RUN_SCHEMA = "gkm.bongard-relational-headless-run.v4"
FAILURE_SCHEMA = "gkm.bongard-relational-headless-terminal-failure.v4"
PROJECTION_SCHEMA = "gkm.bongard-neutral-loop-scene-projection.v1"
SUPPORT_SELECTION_SCHEMA = "gkm.bongard-relational-support-selection.v1"
RELEASE_PANEL_RECEIPT_SCHEMA = "gkm.bongard-release-panel-receipt.v1"
PROPOSAL_SCHEMA_ID = "gkm.bongard-relational-headless-proposal.v1"
ENGINEERING_PROPOSAL_SCHEMA_ID = (
    "gkm.bongard-closed-visual-headless-proposal.v1"
)
PROTOCOL_ID = "bongard.relational-headless/python-query-v4"
SELECTION_ALGORITHM_ID = "bongard.relational-headless/seed-heldout-v4"
STRICT_DEV_MODE = "strict-dev"
EXACT_UNUSED_TRAIN_ENGINEERING_MODE = (
    "exact-unused-train-semantics-reused-engineering"
)
STRICT_DEV_ADMISSION_POLICY_ID = (
    "bongard.relational-headless/historically-clean-semantic-dev-v1"
)
EXACT_UNUSED_TRAIN_ENGINEERING_ADMISSION_POLICY_ID = (
    "bongard.relational-headless/exact-unused-train-semantics-reused-allowlist-v1"
)
EXACT_UNUSED_TRAIN_ENGINEERING_TASK_IDS = (
    "bd_big_small_equil_triangles_0000",
    "bd_big_small_obtuse_triangles_0000",
    "bd_big_small_right_triangles_0000",
    "bd_two_mirror_unbala_triangles_0000",
    "bd_two_unbalanced_triangles_0000",
)
EXPLICITLY_SEALED_ENGINEERING_TASK_ID = (
    "bd_unbala_trapezoid_right_triangle_0000"
)
CAMPAIGN_AUTHORIZATION_PHASE = (
    "relational-headless-full-current-dev-campaign-v4"
)
CAMPAIGN_AUTHORIZATION_ACTOR = "headless-codex-relational-campaign"
CAMPAIGN_AUTHORIZATION_PURPOSE = (
    "atomic full-cohort disclosure before serial one-shot DEV execution"
)
ENGINEERING_CAMPAIGN_AUTHORIZATION_PHASE = (
    "closed-visual-exact-unused-train-engineering-campaign-v2"
)
ENGINEERING_CAMPAIGN_AUTHORIZATION_ACTOR = (
    "headless-codex-closed-visual-engineering-campaign"
)
ENGINEERING_CAMPAIGN_AUTHORIZATION_PURPOSE = (
    "atomic fixed-allowlist TRAIN disclosure for semantics-reused engineering"
)
ENGINEERING_EXPOSURE_PHASE = "closed-visual-exact-unused-train-engineering-v1"
ENGINEERING_EXPOSURE_ACTOR = "headless-codex-closed-visual-engineering-proposer"
ENGINEERING_EXPOSURE_PURPOSE = (
    "one-shot support proposal and two-query closed Python predicate replay"
)

_TASK_ID = re.compile(r"^(ff|bd|hd)_[A-Za-z0-9_.-]+\Z")
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_PNG_BYTES = 32 * 1024 * 1024
_REASONING_EFFORTS = frozenset(
    {"minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
)

_RATIO_BY_ID = {f"{numerator}/{denominator}": (numerator, denominator)
                for numerator, denominator in ALLOWED_AREA_RATIOS}
_RATIO_IDS = tuple(_RATIO_BY_ID)


class RelationalHeadlessRunError(RuntimeError):
    """The standalone runner crossed or could not establish a protocol edge."""


StructuredTransport = Callable[..., CodexStructuredResult]
PanelPacket = LoopScenePacket | ExactPanelWitnessPacket
PacketExtractor = Callable[[bytes], PanelPacket]
PacketVerifier = Callable[..., PanelPacket]
PngReader = Callable[[Path], bytes]


def _open_absolute_no_symlinks(path: Path, *, directory: bool) -> int:
    """Open one absolute path while rejecting a symlink in every component."""

    candidate = Path(os.path.abspath(os.path.expanduser(str(path))))
    if not candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts[1:]):
        raise RelationalHeadlessRunError("authenticated path is not canonical absolute")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise RelationalHeadlessRunError("platform lacks O_NOFOLLOW")
    descriptor = os.open("/", flags | directory_flag)
    try:
        for index, component in enumerate(candidate.parts[1:]):
            final = index == len(candidate.parts[1:]) - 1
            component_flags = flags | nofollow
            if not final or directory:
                component_flags |= directory_flag
            next_descriptor = os.open(
                component, component_flags, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _descriptor_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
    )


def _secure_file_digest(path: Path) -> tuple[str, int, tuple[int, ...]]:
    descriptor = _open_absolute_no_symlinks(path, directory=False)
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RelationalHeadlessRunError("release input is not a private regular file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _descriptor_identity(before) != _descriptor_identity(after):
            raise RelationalHeadlessRunError("release input changed while hashing")
        return digest.hexdigest(), before.st_size, _descriptor_identity(before)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class ReleaseArchiveAuthenticator:
    """Pinned official descriptor plus mechanically authenticated release ZIP."""

    release_descriptor_digest: str
    release_descriptor_source_digest: str
    corpus_manifest_digest: str
    split_source_digest: str
    archive_digest: str
    archive_size_bytes: int
    archive_path: Path = field(repr=False, compare=False)
    archive_identity: tuple[int, ...] = field(repr=False)
    central_directory_digest: str
    members: tuple[tuple[str, int, int], ...] = field(repr=False)

    @classmethod
    def load(
        cls,
        *,
        release_descriptor_path: str | Path,
        expected_release_descriptor_digest: str,
        archive_path: str | Path,
    ) -> "ReleaseArchiveAuthenticator":
        descriptor_path = Path(release_descriptor_path)
        descriptor_bytes = _stable_read(descriptor_path, maximum=1024 * 1024)
        try:
            raw = json.loads(descriptor_bytes)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RelationalHeadlessRunError("release descriptor is malformed") from exc
        if not isinstance(raw, Mapping):
            raise RelationalHeadlessRunError("release descriptor is not an object")
        descriptor = OfficialReleaseDescriptor.from_dict(raw)
        expected = _require_address(
            expected_release_descriptor_digest,
            "official release descriptor digest",
        )
        if descriptor.digest != expected:
            raise RelationalHeadlessRunError("official release descriptor differs")
        archive = Path(os.path.abspath(os.path.expanduser(str(archive_path))))
        digest, size, identity = _secure_file_digest(archive)
        if (
            "sha256:" + digest != descriptor.archive_sha256
            or size != descriptor.archive_size_bytes
        ):
            raise RelationalHeadlessRunError("official release archive identity differs")
        archive_descriptor = _open_absolute_no_symlinks(archive, directory=False)
        try:
            with os.fdopen(os.dup(archive_descriptor), "rb") as handle:
                with zipfile.ZipFile(handle) as bundle:
                    infos = bundle.infolist()
        except (OSError, zipfile.BadZipFile) as exc:
            raise RelationalHeadlessRunError("official release archive is invalid") from exc
        finally:
            os.close(archive_descriptor)
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise RelationalHeadlessRunError("official release archive repeats members")
        members = tuple(
            sorted((info.filename, info.file_size, info.CRC) for info in infos)
        )
        return cls(
            release_descriptor_digest=expected,
            release_descriptor_source_digest="sha256:"
            + hashlib.sha256(descriptor_bytes).hexdigest(),
            corpus_manifest_digest=descriptor.corpus_manifest_sha256,
            split_source_digest=descriptor.split_sha256,
            archive_digest=descriptor.archive_sha256,
            archive_size_bytes=descriptor.archive_size_bytes,
            archive_path=archive,
            archive_identity=identity,
            central_directory_digest=canonical_digest(
                [
                    {"member": name, "size_bytes": size, "crc32": crc}
                    for name, size, crc in members
                ]
            ),
            members=members,
        )

    def identity_data(self) -> dict[str, object]:
        return {
            "release_descriptor_digest": self.release_descriptor_digest,
            "release_descriptor_source_digest": self.release_descriptor_source_digest,
            "corpus_manifest_digest": self.corpus_manifest_digest,
            "split_source_digest": self.split_source_digest,
            "archive_digest": self.archive_digest,
            "archive_size_bytes": self.archive_size_bytes,
            "central_directory_digest": self.central_directory_digest,
            "panel_authentication": "authorized-phase-exact-zip-member-byte-match",
            "layout": "archive/images",
        }

    def authenticate(
        self, relative_path: str, payload: bytes
    ) -> dict[str, object]:
        if (
            not isinstance(relative_path, str)
            or relative_path.startswith("/")
            or "\\" in relative_path
            or any(part in {"", ".", ".."} for part in relative_path.split("/"))
        ):
            raise RelationalHeadlessRunError("release panel relative path is malformed")
        member = "ShapeBongard_V2/" + relative_path
        indexed = {name: (size, crc) for name, size, crc in self.members}
        if member not in indexed:
            raise RelationalHeadlessRunError("panel is absent from official release archive")
        descriptor = _open_absolute_no_symlinks(self.archive_path, directory=False)
        try:
            opened = os.fstat(descriptor)
            if _descriptor_identity(opened) != self.archive_identity:
                raise RelationalHeadlessRunError("release archive changed after plan")
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                with zipfile.ZipFile(handle) as bundle:
                    released = bundle.read(member)
            after = os.fstat(descriptor)
            if _descriptor_identity(after) != self.archive_identity:
                raise RelationalHeadlessRunError("release archive changed while reading")
        except (OSError, KeyError, zipfile.BadZipFile, RuntimeError) as exc:
            if isinstance(exc, RelationalHeadlessRunError):
                raise
            raise RelationalHeadlessRunError("cannot authenticate release panel") from exc
        finally:
            os.close(descriptor)
        expected_size, crc = indexed[member]
        if len(released) != expected_size or released != payload:
            raise RelationalHeadlessRunError(
                "extracted panel bytes differ from official release member"
            )
        digest = "sha256:" + hashlib.sha256(released).hexdigest()
        content = {
            "schema": RELEASE_PANEL_RECEIPT_SCHEMA,
            "relative_path": relative_path,
            "archive_member": member,
            "sha256": digest,
            "size_bytes": len(released),
            "zip_crc32": crc,
            "release_descriptor_digest": self.release_descriptor_digest,
            "archive_digest": self.archive_digest,
            "central_directory_digest": self.central_directory_digest,
        }
        return {**content, "digest": canonical_digest(content)}


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RelationalHeadlessRunError(f"{label} must be 64 lowercase hex digits")
    return value


def _require_address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise RelationalHeadlessRunError(
            f"{label} must be a sha256: content address"
        )
    return value


def _raw_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _text_digest(value: str) -> str:
    return _raw_digest(value.encode("utf-8", errors="strict"))


def _draw(task_id: str, seed: str, purpose: str, modulus: int) -> int:
    if not isinstance(seed, str) or not seed.strip() or "\x00" in seed:
        raise RelationalHeadlessRunError("seed must be non-empty NUL-free text")
    payload = "\0".join((SELECTION_ALGORITHM_ID, task_id, seed, purpose))
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest(), 16) % modulus


def _support_selection_data(
    task_id: str,
    positive_indices: Sequence[int],
    negative_indices: Sequence[int],
) -> dict[str, object]:
    return {
        "schema": SUPPORT_SELECTION_SCHEMA,
        "task_id": task_id,
        "positive_indices": sorted(positive_indices),
        "negative_indices": sorted(negative_indices),
    }


def _support_selection_hiding_commitment(
    selection: Mapping[str, object], key: str
) -> str:
    """Keyed commitment; public brute force cannot validate 49 schedules."""

    _require_sha256(key, "support selection commitment key")
    return hmac.new(
        bytes.fromhex(key),
        canonical_json(dict(selection)),
        hashlib.sha256,
    ).hexdigest()


def _seal(content: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(content)
    if "digest" in body:
        raise RelationalHeadlessRunError("artifact content already contains digest")
    return {**body, "digest": canonical_digest(body)}


def _verify_seal(value: Mapping[str, Any], schema: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != schema:
        raise RelationalHeadlessRunError(f"unsupported {schema} artifact")
    data = dict(value)
    digest = data.pop("digest", None)
    if digest != canonical_digest(data):
        raise RelationalHeadlessRunError(f"{schema} digest does not reproduce")
    return dict(value)


def _stable_read(path: Path, *, maximum: int | None = None) -> bytes:
    """Read one unchanged regular file through an O_NOFOLLOW descriptor."""

    if not hasattr(os, "O_NOFOLLOW"):
        raise RelationalHeadlessRunError("platform lacks O_NOFOLLOW")
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise RelationalHeadlessRunError(f"cannot inspect {path}") from exc
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise RelationalHeadlessRunError(f"input is not a private regular file: {path}")
    if maximum is not None and not 0 < before.st_size <= maximum:
        raise RelationalHeadlessRunError(f"input size is invalid: {path}")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RelationalHeadlessRunError(f"cannot open {path}") from exc
    chunks: list[bytes] = []
    try:
        opened = os.fstat(descriptor)
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if not stat.S_ISREG(opened.st_mode) or opened_identity != identity:
            raise RelationalHeadlessRunError(f"input changed while opening: {path}")
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1_048_576, remaining))
            if not chunk:
                raise RelationalHeadlessRunError(f"short read from {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RelationalHeadlessRunError(f"input grew while reading: {path}")
        after = os.fstat(descriptor)
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if after_identity != identity:
            raise RelationalHeadlessRunError(f"input changed while reading: {path}")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _read_png_no_follow(path: Path) -> bytes:
    payload = _stable_read(Path(path), maximum=_MAX_PNG_BYTES)
    if not payload.startswith(PNG_SIGNATURE):
        raise RelationalHeadlessRunError(f"panel lacks PNG signature: {path}")
    return payload


def _read_authenticated_release_panel(
    *,
    corpus_root: str | Path,
    path: Path,
    authenticator: ReleaseArchiveAuthenticator,
    observer_reader: PngReader,
) -> tuple[bytes, dict[str, object]]:
    """Read below a no-symlink root and byte-match the official ZIP member."""

    root = Path(os.path.abspath(os.path.expanduser(str(corpus_root))))
    candidate = Path(os.path.abspath(str(path)))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise RelationalHeadlessRunError("panel path escapes corpus root") from exc
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise RelationalHeadlessRunError("panel relative path is malformed")
    descriptor = _open_absolute_no_symlinks(candidate, directory=False)
    chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= _MAX_PNG_BYTES
        ):
            raise RelationalHeadlessRunError("panel is not a bounded private file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _descriptor_identity(before) != _descriptor_identity(after):
            raise RelationalHeadlessRunError("panel changed while securely reading")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if len(payload) != before.st_size or not payload.startswith(PNG_SIGNATURE):
        raise RelationalHeadlessRunError("panel is not an exact PNG")
    if observer_reader is not _read_png_no_follow:
        observed = observer_reader(candidate)
        if observed != payload:
            raise RelationalHeadlessRunError(
                "injected panel reader differs from secure authenticated read"
            )
    receipt = authenticator.authenticate(relative.as_posix(), payload)
    return payload, receipt


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once_durable(path: Path, payload: bytes) -> Path:
    """Exclusively create, fsync, and byte-for-byte reload one artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError:
        if _stable_read(destination) != payload:
            raise RelationalHeadlessRunError(
                f"refusing to overwrite different artifact at {destination}"
            )
    except OSError as exc:
        raise RelationalHeadlessRunError(f"cannot create {destination}") from exc
    else:
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise RelationalHeadlessRunError(f"short write to {destination}")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    _fsync_directory(destination.parent)
    if _stable_read(destination) != payload:
        raise RelationalHeadlessRunError(
            f"durable artifact reload differs at {destination}"
        )
    return destination


def _persist_artifact(
    directory: str | Path,
    artifact: Mapping[str, Any],
    *,
    suffix: str,
) -> tuple[Path, dict[str, Any]]:
    digest = _require_sha256(artifact.get("digest"), "artifact digest")
    payload = canonical_json(dict(artifact)) + b"\n"
    path = _write_once_durable(Path(directory) / f"{digest}.{suffix}.json", payload)
    try:
        decoded = json.loads(_stable_read(path))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RelationalHeadlessRunError(f"cannot decode durable artifact {path}") from exc
    if not isinstance(decoded, dict) or canonical_json(decoded) + b"\n" != payload:
        raise RelationalHeadlessRunError(f"durable artifact is not canonical: {path}")
    return path, decoded


def _persist_exposure(
    successor: ExposureLedger, directory: str | Path
) -> tuple[Path, ExposureLedger]:
    filename = successor.digest.removeprefix("sha256:") + ".exposure.json"
    payload = successor.to_json().encode("utf-8")
    path = _write_once_durable(Path(directory) / filename, payload)
    reloaded = ExposureLedger.load(path)
    if reloaded != successor or _stable_read(path) != payload:
        raise RelationalHeadlessRunError("durable exposure successor differs")
    return path, reloaded


def relational_proposal_schema() -> dict[str, object]:
    """Strict one-proposal schema; there is no backup or polarity field."""

    result: dict[str, object] = {
        "type": "object",
        "properties": {
            "numerator_side_count": {
                "type": "integer",
                "enum": list(ALLOWED_SIDE_COUNTS),
            },
            "denominator_side_count": {
                "type": "integer",
                "enum": list(ALLOWED_SIDE_COUNTS),
            },
            "area_ratio": {"type": "string", "enum": list(_RATIO_IDS)},
            "denominator_obliqueness_millidegrees": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "integer",
                        "enum": list(
                            ALLOWED_OBLIQUENESS_THRESHOLDS_MILLIDEGREES
                        ),
                    },
                ]
            },
            "rationale": {"type": "string"},
        },
        "required": [
            "numerator_side_count",
            "denominator_side_count",
            "area_ratio",
            "denominator_obliqueness_millidegrees",
            "rationale",
        ],
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(result)
    return result


def _direct_atom_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "catalog_key": {
                "type": "string",
                "enum": sorted(
                    item.catalog_key for item in DIRECT_VISUAL_ATOM_CATALOG.atoms
                ),
            },
            "comparison": {"type": "string", "enum": ["equal"]},
            "target_count": {"type": "integer", "enum": list(range(1, 9))},
        },
        "required": ["catalog_key", "comparison", "target_count"],
        "additionalProperties": False,
    }


def closed_visual_proposal_schema() -> dict[str, object]:
    """Strict tagged proposal with nullable inactive branches and no code slot."""

    relational = {
        "type": "object",
        "properties": {
            "numerator_side_count": {
                "type": "integer",
                "enum": list(ALLOWED_SIDE_COUNTS),
            },
            "denominator_side_count": {
                "type": "integer",
                "enum": list(ALLOWED_SIDE_COUNTS),
            },
            "area_ratio": {"type": "string", "enum": list(_RATIO_IDS)},
            "denominator_obliqueness_millidegrees": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "integer",
                        "enum": list(
                            ALLOWED_OBLIQUENESS_THRESHOLDS_MILLIDEGREES
                        ),
                    },
                ]
            },
        },
        "required": [
            "numerator_side_count",
            "denominator_side_count",
            "area_ratio",
            "denominator_obliqueness_millidegrees",
        ],
        "additionalProperties": False,
    }
    symmetry = {
        "type": "object",
        "properties": {
            "metric": {
                "type": "string",
                "enum": [item.value for item in SymmetryMetric],
            },
            "threshold_ppm": {
                "type": "integer",
                "enum": [
                    250_000,
                    500_000,
                    600_000,
                    700_000,
                    750_000,
                    800_000,
                    850_000,
                    900_000,
                    950_000,
                ],
            },
        },
        "required": ["metric", "threshold_ppm"],
        "additionalProperties": False,
    }
    nullable_atom = {"anyOf": [{"type": "null"}, _direct_atom_schema()]}
    result: dict[str, object] = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": [item.value for item in ClosedPredicateKind],
            },
            "relational": {"anyOf": [{"type": "null"}, relational]},
            "direct_atom_0": nullable_atom,
            "direct_atom_1": nullable_atom,
            "direct_atom_2": nullable_atom,
            "symmetry": {"anyOf": [{"type": "null"}, symmetry]},
            "rationale": {"type": "string"},
        },
        "required": [
            "kind",
            "relational",
            "direct_atom_0",
            "direct_atom_1",
            "direct_atom_2",
            "symmetry",
            "rationale",
        ],
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(result)
    return result


@dataclass(frozen=True, slots=True)
class FrozenCompleteClosedLibraryIndex:
    """Compact pre-pixel identity of the exhaustive constructive union.

    The complete materialized tuple is useful for offline oracle sweeps, but
    retaining 65,678 Python ASTs in the one-shot runner is unnecessary and can
    exhaust a headless worker.  This index freezes the exhaustive construction
    grids and source/evaluator identities; membership is then checked by the
    same closed constructors used by that construction, before any packet is
    evaluated.
    """

    construction_id: str
    member_count: int
    predicate_source_digest: str
    evaluator_digest: str
    construction_grid_digest: str
    complete_member_digest: str

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": (
                "gkm.bongard-compact-proposer-reachable-closed-library-index.v2"
            ),
            "algorithm_id": LIBRARY_ALGORITHM_ID,
            "construction_id": self.construction_id,
            "member_count": self.member_count,
            "predicate_source_digest": self.predicate_source_digest,
            "evaluator_digest": self.evaluator_digest,
            "construction_grid_digest": self.construction_grid_digest,
            "complete_member_digest": self.complete_member_digest,
            "membership": "proposer-reachable-closed-constructor/v2",
            "packet_inputs_accepted": False,
        }


def _complete_closed_library() -> FrozenCompleteClosedLibraryIndex:
    identity = complete_closed_predicate_library_identity()
    return FrozenCompleteClosedLibraryIndex(
        construction_id=identity.construction_id,
        member_count=identity.member_count,
        predicate_source_digest=identity.source_digest,
        evaluator_digest=identity.evaluator_digest,
        construction_grid_digest=identity.construction_grid_digest,
        complete_member_digest=identity.complete_member_digest,
    )


def _closed_library_binding(
    library: FrozenCompleteClosedLibraryIndex,
) -> dict[str, object]:
    return {
        "predicate_authority": "canonical-pure-python",
        "lean_required": False,
        "semantic_checker_imported": False,
        "construction_id": library.construction_id,
        "library_digest": library.digest,
        "member_count": library.member_count,
        "library_source_digest": library.predicate_source_digest,
        "predicate_source_digest": library.predicate_source_digest,
        "evaluator_digest": library.evaluator_digest,
        "construction_grid_digest": library.construction_grid_digest,
        "complete_member_digest": library.complete_member_digest,
        "composite_packet_source_digest": composite_visual_packet_source_digest(),
        "composite_extractor_digest": exact_panel_witness_extractor_digest(),
        "direct_catalog_digest": direct_visual_catalog_digest(),
    }


def _selection_protocol_digest(
    benchmark_mode: str = STRICT_DEV_MODE,
    closed_library: FrozenCompleteClosedLibraryIndex | None = None,
) -> str:
    if benchmark_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE:
        if closed_library is None:
            raise RelationalHeadlessRunError(
                "engineering protocol requires a pre-frozen closed library"
            )
        proposal_schema_id = ENGINEERING_PROPOSAL_SCHEMA_ID
        admission_policy_id = EXACT_UNUSED_TRAIN_ENGINEERING_ADMISSION_POLICY_ID
        proposal_schema = closed_visual_proposal_schema()
        evaluator_binding: Mapping[str, object] = _closed_library_binding(
            closed_library
        )
    elif benchmark_mode == STRICT_DEV_MODE:
        proposal_schema_id = PROPOSAL_SCHEMA_ID
        admission_policy_id = STRICT_DEV_ADMISSION_POLICY_ID
        proposal_schema = relational_proposal_schema()
        evaluator_binding = {
            "query_algorithm_digest": relational_query_algorithm_digest(),
            "loop_scene_extractor_digest": loop_scene_extractor_digest(),
            "loop_scene_catalog_digest": loop_scene_catalog_digest(),
        }
    else:
        raise RelationalHeadlessRunError("unknown headless benchmark mode")
    return canonical_digest(
        {
            "protocol_id": PROTOCOL_ID,
            "benchmark_mode": benchmark_mode,
            "runner_python_source_digest": (
                relational_headless_runner_source_digest()
            ),
            "proposal_schema_id": proposal_schema_id,
            "selection_algorithm_id": SELECTION_ALGORITHM_ID,
            "task_admission_policy_id": admission_policy_id,
            "proposal_schema": proposal_schema,
            "evaluator_binding": dict(evaluator_binding),
            "support_gate": {
                "positive": Disposition.PRESENT.value,
                "negative": Disposition.CERTIFIED_ABSENT.value,
                "direction": "forward-only",
                "negation": False,
                "polarity_flip": False,
            },
            "point_contact_enabled": False,
        }
    )


def relational_headless_runner_source_digest() -> str:
    """Bind plans to the exact standalone runner implementation bytes."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class RelationalHeadlessPlan:
    benchmark_mode: str
    task_admission_policy_id: str
    task_id: str
    family: str
    split: str
    corpus_digest: str
    split_source_digest: str
    release_descriptor_digest: str
    release_descriptor_source_digest: str
    release_archive_digest: str
    release_archive_size_bytes: int
    release_central_directory_digest: str
    exposure_predecessor_digest: str
    semantic_resolution_digest: str
    historical_seed_digest: str
    semantic_resolver_policy_digest: str
    strict_dev_concepts: tuple[str, ...]
    strict_dev_classification_digest: str
    seed_digest: str
    query_schedule_commitment: str
    support_selection_hiding_commitment: str
    protocol_digest: str
    proposal_schema_digest: str
    model: str
    reasoning_effort: str
    minutes: int
    expected_launcher_digest: str
    cloud_policy_cache_binding: str
    exposure_observed_at: str
    closed_predicate_binding: Mapping[str, object] | None
    _seed: str = field(repr=False, compare=False)
    _label_nonce: str = field(repr=False, compare=False)
    _positive_query_index: int = field(repr=False, compare=False)
    _negative_query_index: int = field(repr=False, compare=False)
    _query_order: tuple[str, str] = field(repr=False, compare=False)
    _support_selection_key: str = field(repr=False, compare=False)
    _release_authenticator: ReleaseArchiveAuthenticator = field(
        repr=False, compare=False
    )
    _closed_library: FrozenCompleteClosedLibraryIndex | None = field(
        repr=False, compare=False
    )

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        if self.benchmark_mode == STRICT_DEV_MODE:
            admission: dict[str, object] = {
                "mode": STRICT_DEV_MODE,
                "policy_id": STRICT_DEV_ADMISSION_POLICY_ID,
                "required_historically_clean": True,
                "required_semantic_cohort": "dev",
                "concepts": list(self.strict_dev_concepts),
                "classification_digest": self.strict_dev_classification_digest,
            }
            predicate_authority = "pure-python-relational-query"
        elif self.benchmark_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE:
            admission = {
                "mode": EXACT_UNUSED_TRAIN_ENGINEERING_MODE,
                "policy_id": (
                    EXACT_UNUSED_TRAIN_ENGINEERING_ADMISSION_POLICY_ID
                ),
                "split_required": "train",
                "regime_required": None,
                "active_ledger_exact_task_required": "unseen",
                "historical_exact_task_required": "not_recorded",
                "historical_panel_count_required": 0,
                "historical_semantic_exposure_required": (
                    "historically_exposed"
                ),
                "historically_clean_required": False,
                "semantic_cohort_required": None,
                "semantic_unseen_required": False,
                "fixed_task_allowlist": list(
                    EXACT_UNUSED_TRAIN_ENGINEERING_TASK_IDS
                ),
                "explicit_sealed_task_rejected": (
                    EXPLICITLY_SEALED_ENGINEERING_TASK_ID
                ),
                "concepts": list(self.strict_dev_concepts),
                "classification_digest": self.strict_dev_classification_digest,
            }
            predicate_authority = "pure-python-closed-visual-predicate-union"
        else:  # pragma: no cover - constructor is internal.
            raise RelationalHeadlessRunError("unknown plan benchmark mode")
        result: dict[str, object] = {
            "schema": PLAN_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "benchmark_mode": self.benchmark_mode,
            "task_id": self.task_id,
            "family": self.family,
            "split": self.split,
            "official_test_authorized": False,
            "action_program_json_authorized": False,
            "predicate_authority": predicate_authority,
            "corpus_digest": self.corpus_digest,
            "split_source_digest": self.split_source_digest,
            "release_authentication": {
                "release_descriptor_digest": self.release_descriptor_digest,
                "release_descriptor_source_digest": (
                    self.release_descriptor_source_digest
                ),
                "archive_digest": self.release_archive_digest,
                "archive_size_bytes": self.release_archive_size_bytes,
                "central_directory_digest": (
                    self.release_central_directory_digest
                ),
                "panel_authentication": (
                    "authorized-phase-exact-zip-member-byte-match"
                ),
                "layout": "archive/images",
            },
            "exposure_predecessor_digest": self.exposure_predecessor_digest,
            "semantic_resolution_digest": self.semantic_resolution_digest,
            "historical_seed_digest": self.historical_seed_digest,
            "semantic_resolver_policy_digest": self.semantic_resolver_policy_digest,
            "task_admission": admission,
            "seed_digest": self.seed_digest,
            "query_schedule_commitment": self.query_schedule_commitment,
            "support_selection_hiding_commitment": (
                self.support_selection_hiding_commitment
            ),
            "support_selection_opening_publicly_disclosed": False,
            "protocol_digest": self.protocol_digest,
            "proposal_schema_digest": self.proposal_schema_digest,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "minutes": self.minutes,
            "expected_launcher_digest": self.expected_launcher_digest,
            "cloud_policy_cache_binding": self.cloud_policy_cache_binding,
            "exposure_observed_at": self.exposure_observed_at,
        }
        if self.closed_predicate_binding is not None:
            result["closed_predicate_binding"] = dict(
                self.closed_predicate_binding
            )
        if self.benchmark_mode == STRICT_DEV_MODE:
            result["strict_dev_admission"] = {
                key: value for key, value in admission.items() if key != "mode"
            }
        else:
            result["engineering_train_admission"] = admission
        return result

    @property
    def positive_support_indices(self) -> tuple[int, ...]:
        return tuple(i for i in range(7) if i != self._positive_query_index)

    @property
    def negative_support_indices(self) -> tuple[int, ...]:
        return tuple(i for i in range(7) if i != self._negative_query_index)

    def reveal_schedule(self) -> dict[str, object]:
        return {
            "seed": self._seed,
            "label_nonce": self._label_nonce,
            "positive_query_index": self._positive_query_index,
            "negative_query_index": self._negative_query_index,
            "query_order": list(self._query_order),
        }


def prepare_relational_headless_plan(
    *,
    task_id: str,
    split_index: SplitIndex,
    predecessor: ExposureLedger,
    expected_exposure_predecessor_digest: str | None = None,
    expected_corpus_digest: str,
    expected_split_source_digest: str,
    seed: str,
    exposure_observed_at: str,
    expected_launcher_digest: str,
    release_authenticator: ReleaseArchiveAuthenticator,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    model: str = DEFAULT_CODEX_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    minutes: int = 15,
    label_nonce: str | None = None,
    support_selection_key: str | None = None,
    benchmark_mode: str = STRICT_DEV_MODE,
    closed_library: FrozenCompleteClosedLibraryIndex | None = None,
) -> RelationalHeadlessPlan:
    """Freeze metadata-only selection and policy without touching corpus paths."""

    if not isinstance(split_index, SplitIndex) or not split_index.groups:
        raise RelationalHeadlessRunError("runner requires a non-empty split index")
    match = _TASK_ID.fullmatch(task_id) if isinstance(task_id, str) else None
    if match is None:
        raise RelationalHeadlessRunError("task_id is not a canonical Bongard ID")
    family = match.group(1)
    corpus_digest = _require_address(expected_corpus_digest, "corpus digest")
    if not isinstance(release_authenticator, ReleaseArchiveAuthenticator):
        raise TypeError("release_authenticator must be ReleaseArchiveAuthenticator")
    if release_authenticator.corpus_manifest_digest != corpus_digest:
        raise RelationalHeadlessRunError(
            "release descriptor corpus manifest differs from plan"
        )
    predecessor.assert_corpus(corpus_digest)
    if expected_exposure_predecessor_digest is not None and (
        predecessor.digest
        != _require_address(
            expected_exposure_predecessor_digest,
            "exposure predecessor digest",
        )
    ):
        raise RelationalHeadlessRunError("exposure predecessor differs from pin")
    split_digest = _require_address(
        expected_split_source_digest, "split source digest"
    )
    if split_index.source_digest != split_digest:
        raise RelationalHeadlessRunError("split index differs from external digest")
    if release_authenticator.split_source_digest != split_digest:
        raise RelationalHeadlessRunError("release descriptor split differs from plan")
    if benchmark_mode not in {
        STRICT_DEV_MODE,
        EXACT_UNUSED_TRAIN_ENGINEERING_MODE,
    }:
        raise RelationalHeadlessRunError("unknown headless benchmark mode")
    assignment = split_index.assignment(task_id)
    if benchmark_mode == STRICT_DEV_MODE:
        if assignment.split not in {"train", "val"} or assignment.regime is not None:
            raise RelationalHeadlessRunError(
                "relational headless runner permits explicit train/val tasks only"
            )
    elif (
        assignment.split != "train"
        or assignment.regime is not None
        or task_id not in EXACT_UNUSED_TRAIN_ENGINEERING_TASK_IDS
        or task_id == EXPLICITLY_SEALED_ENGINEERING_TASK_ID
    ):
        if task_id == EXPLICITLY_SEALED_ENGINEERING_TASK_ID:
            raise RelationalHeadlessRunError(
                "sealed task is forbidden in TRAIN engineering mode"
            )
        raise RelationalHeadlessRunError(
            "engineering mode requires the fixed five-task TRAIN allowlist"
        )
    if not isinstance(model, str) or _MODEL.fullmatch(model) is None:
        raise RelationalHeadlessRunError("model identifier is malformed")
    if reasoning_effort not in _REASONING_EFFORTS:
        raise RelationalHeadlessRunError("reasoning effort is not allowlisted")
    predecessor.assert_unseen(task_ids=(task_id,))
    historical = load_historical_exposure()
    cohort = classify_task(
        task_id,
        historical,
        split=assignment.split,
        regime=assignment.regime,
    )
    resolver_digest = semantic_resolver_policy_digest(historical)
    if benchmark_mode == STRICT_DEV_MODE:
        if not cohort.historically_clean or cohort.semantic_cohort != "dev":
            raise RelationalHeadlessRunError(
                "relational headless runner requires a historically-clean strict "
                f"DEV task; got historically_clean={cohort.historically_clean}, "
                f"semantic_cohort={cohort.semantic_cohort!r}"
            )
        if cohort.family != family:
            raise RelationalHeadlessRunError("strict DEV classifier family differs")
        resolution = predecessor.assert_semantically_unseen(
            task_ids=(task_id,),
            historical_seed=historical,
            expected_historical_seed_digest=historical.seed_digest,
            expected_resolver_policy_digest=resolver_digest,
        )
        semantic_resolution_digest = canonical_digest(resolution.to_dict())
        admission_policy_id = STRICT_DEV_ADMISSION_POLICY_ID
        frozen_library = None
        closed_binding = None
        schema = relational_proposal_schema()
    else:
        if (
            cohort.family != family
            or cohort.split != "train"
            or cohort.regime is not None
            or cohort.exact_task_exposure != "not_recorded"
            or cohort.exact_panel_record_count != 0
            or cohort.semantic_exposure != "historically_exposed"
            or cohort.semantic_cohort is not None
            or cohort.historically_clean is not False
        ):
            raise RelationalHeadlessRunError(
                "engineering admission requires exact-unused TRAIN metadata "
                "with historical semantic exposure"
            )
        # This mode intentionally does not assert semantic unseen: historical
        # semantic reuse is its explicit label and admission condition.
        semantic_resolution_digest = canonical_digest(
            {
                "mode": EXACT_UNUSED_TRAIN_ENGINEERING_MODE,
                "semantic_unseen_asserted": False,
                "historical_classification": cohort.to_dict(),
                "active_ledger_predecessor_digest": predecessor.digest,
            }
        )
        admission_policy_id = (
            EXACT_UNUSED_TRAIN_ENGINEERING_ADMISSION_POLICY_ID
        )
        expected_library = _complete_closed_library()
        if closed_library is not None and (
            not isinstance(closed_library, FrozenCompleteClosedLibraryIndex)
            or closed_library.to_data() != expected_library.to_data()
        ):
            raise RelationalHeadlessRunError(
                "engineering mode requires the complete pre-frozen library"
            )
        frozen_library = expected_library
        closed_binding = _closed_library_binding(frozen_library)
        if (
            closed_binding["construction_id"]
            != "complete-proposer-reachable-closed-union/v2"
            or closed_binding["member_count"] != 65_678
        ):
            raise RelationalHeadlessRunError(
                "complete closed predicate library identity drifted"
            )
        schema = closed_visual_proposal_schema()
    positive_query_index = _draw(task_id, seed, "positive-query", 7)
    negative_query_index = _draw(task_id, seed, "negative-query", 7)
    order = (
        ("positive", "negative")
        if _draw(task_id, seed, "query-order", 2) == 0
        else ("negative", "positive")
    )
    nonce = label_nonce or secrets.token_hex(32)
    _require_sha256(nonce, "label nonce")
    schedule = {
        "positive_query_index": positive_query_index,
        "negative_query_index": negative_query_index,
        "query_order": list(order),
    }
    schedule_commitment = canonical_digest(
        {
            "schema": "gkm.bongard-relational-query-schedule-seal.v1",
            **schedule,
            "nonce": nonce,
        }
    )
    support_selection = _support_selection_data(
        task_id,
        [i for i in range(7) if i != positive_query_index],
        [i for i in range(7) if i != negative_query_index],
    )
    selection_key = support_selection_key or secrets.token_hex(32)
    _require_sha256(selection_key, "support selection commitment key")
    if isinstance(minutes, bool) or not isinstance(minutes, int) or not 1 <= minutes <= 120:
        raise RelationalHeadlessRunError("minutes must be an integer in [1, 120]")
    launcher = _require_sha256(expected_launcher_digest, "launcher digest")
    cache = cloud_policy_cache_snapshot or CloudPolicyCacheSnapshot(None)
    return RelationalHeadlessPlan(
        benchmark_mode=benchmark_mode,
        task_admission_policy_id=admission_policy_id,
        task_id=task_id,
        family=family,
        split=assignment.split,
        corpus_digest=corpus_digest,
        split_source_digest=split_digest,
        release_descriptor_digest=(
            release_authenticator.release_descriptor_digest
        ),
        release_descriptor_source_digest=(
            release_authenticator.release_descriptor_source_digest
        ),
        release_archive_digest=release_authenticator.archive_digest,
        release_archive_size_bytes=release_authenticator.archive_size_bytes,
        release_central_directory_digest=(
            release_authenticator.central_directory_digest
        ),
        exposure_predecessor_digest=predecessor.digest,
        semantic_resolution_digest=semantic_resolution_digest,
        historical_seed_digest=historical.seed_digest,
        semantic_resolver_policy_digest=resolver_digest,
        strict_dev_concepts=cohort.parsed.concepts,
        strict_dev_classification_digest=canonical_digest(cohort.to_dict()),
        seed_digest=_text_digest(seed),
        query_schedule_commitment=schedule_commitment,
        support_selection_hiding_commitment=(
            _support_selection_hiding_commitment(
                support_selection, selection_key
            )
        ),
        protocol_digest=_selection_protocol_digest(
            benchmark_mode, frozen_library
        ),
        proposal_schema_digest=canonical_digest(schema),
        model=model,
        reasoning_effort=reasoning_effort,
        minutes=minutes,
        expected_launcher_digest=launcher,
        cloud_policy_cache_binding=cache.binding,
        exposure_observed_at=exposure_observed_at,
        closed_predicate_binding=closed_binding,
        _seed=seed,
        _label_nonce=nonce,
        _positive_query_index=positive_query_index,
        _negative_query_index=negative_query_index,
        _query_order=order,
        _support_selection_key=selection_key,
        _release_authenticator=release_authenticator,
        _closed_library=frozen_library,
    )


@dataclass(frozen=True, slots=True)
class _SupportPanel:
    polarity: str
    source_index: int
    payload: bytes = field(repr=False)
    source_sha256: str
    packet: PanelPacket
    release_panel_receipt: Mapping[str, Any]
    presentation_name: str = ""

    def with_name(self, name: str) -> "_SupportPanel":
        return _SupportPanel(
            self.polarity,
            self.source_index,
            self.payload,
            self.source_sha256,
            self.packet,
            self.release_panel_receipt,
            name,
        )


def neutral_loop_scene_projection(packet: LoopScenePacket) -> dict[str, object]:
    """Whitelist the candidate-independent observables shown to the proposer.

    Pixel/provenance digests, task identity, paths, source indices, and Bongard
    side are deliberately absent.  The outer prompt pairs this neutral packet
    with one canonical ``pos_i`` or ``neg_i`` presentation name.
    """

    if not isinstance(packet, LoopScenePacket):
        raise TypeError("packet must be a LoopScenePacket")
    scenarios: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(packet.scenarios):
        loops: list[dict[str, object]] = []
        substantive = tuple(
            loop
            for loop in scenario.loops
            if loop.substantiveness.disposition is Disposition.PRESENT
        )
        for loop in substantive:
            side_count = loop.polygon.side_count
            obliqueness = loop.edge_obliqueness.minimum_millidegrees
            loops.append(
                {
                    "loop_id": loop.loop_id,
                    "area_pixels": loop.area_pixels,
                    "substantiveness": loop.substantiveness.disposition.value,
                    "polygon": {
                        "disposition": loop.polygon.disposition.value,
                        "side_count": (
                            None
                            if side_count is None
                            else {"lower": side_count.lower, "upper": side_count.upper}
                        ),
                    },
                    "edge_obliqueness": {
                        "disposition": loop.edge_obliqueness.disposition.value,
                        "minimum_millidegrees": (
                            None
                            if obliqueness is None
                            else {
                                "lower": obliqueness.lower,
                                "upper": obliqueness.upper,
                            }
                        ),
                    },
                }
            )
        scenarios.append(
            {
                "scenario_id": f"scenario-{scenario_index:02d}",
                "loops": loops,
                "excluded_below_floor_count": len(scenario.loops) - len(substantive),
            }
        )
    return {
        "schema": PROJECTION_SCHEMA,
        "semantics": (
            "candidate-independent-operational-observation-not-world-truth"
        ),
        "width_pixels": packet.width_pixels,
        "height_pixels": packet.height_pixels,
        "scenarios": scenarios,
    }


def neutral_closed_visual_projection(
    packet: ExactPanelWitnessPacket,
) -> dict[str, object]:
    """Small candidate-independent view; attached PNGs remain primary vision input."""

    if not isinstance(packet, ExactPanelWitnessPacket):
        raise TypeError("packet must be an ExactPanelWitnessPacket")
    return {
        "schema": "gkm.bongard-neutral-closed-visual-projection.v1",
        "semantics": "operational-observations-not-world-truth",
        "loop_scene": neutral_loop_scene_projection(packet.loop_scene),
        "bilateral_scenarios": [
            {
                "scenario_id": item.scenario_id,
                "disposition": item.disposition.value,
                "coverage_ppm": (
                    None
                    if item.coverage_ppm is None
                    else item.coverage_ppm.to_data()
                ),
                "reflection_mismatch_ppm": (
                    None
                    if item.mismatch_ppm is None
                    else item.mismatch_ppm.to_data()
                ),
            }
            for item in packet.bilateral_symmetry.scenarios
        ],
    }


def _proposal_prompt(
    panels: Sequence[_SupportPanel],
    *,
    benchmark_mode: str = STRICT_DEV_MODE,
) -> str:
    if tuple(item.presentation_name for item in panels) != tuple(
        [f"pos_{index}.png" for index in range(6)]
        + [f"neg_{index}.png" for index in range(6)]
    ):
        raise RelationalHeadlessRunError("support presentation order differs")
    if benchmark_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE:
        if any(
            not isinstance(item.packet, ExactPanelWitnessPacket)
            for item in panels
        ):
            raise RelationalHeadlessRunError(
                "engineering support requires composite packets"
            )
        packet_view = {
            "schema": "gkm.bongard-closed-visual-headless-support-view.v1",
            "panels": [
                {
                    "presentation_name": item.presentation_name,
                    "neutral_closed_visual": neutral_closed_visual_projection(
                        item.packet
                    ),
                }
                for item in panels
            ],
        }
        grid = {
            "predicate_kinds": [item.value for item in ClosedPredicateKind],
            "relational": {
                "side_counts": list(ALLOWED_SIDE_COUNTS),
                "area_ratios": list(_RATIO_IDS),
                "denominator_obliqueness_millidegrees": [
                    None,
                    *ALLOWED_OBLIQUENESS_THRESHOLDS_MILLIDEGREES,
                ],
                "point_contact_enabled": False,
            },
            "direct_counts": {
                "catalog_keys": sorted(
                    item.catalog_key for item in DIRECT_VISUAL_ATOM_CATALOG.atoms
                ),
                "target_counts": list(range(1, 9)),
                "atom_count": "1..3 unique catalog keys",
            },
            "symmetry": {
                "metrics": [item.value for item in SymmetryMetric],
                "threshold_unit": "parts_per_million",
            },
        }
        return (
            "Use the twelve attached labelled PNGs as the primary visual input. "
            "Propose exactly one tagged closed positive predicate: one relational "
            "parameter tuple, a conjunction of one to three direct positive count "
            "atoms, or one bilateral metric/threshold. Set every inactive tagged "
            "branch to null. The predicate must be PRESENT on every pos panel and "
            "CERTIFIED_ABSENT on every neg panel under the supplied operational "
            "observations. There is no Not, polarity flip, arbitrary source code, "
            "callback, alternative list, or held-out inference. Reflection mismatch "
            "is a directly measured residual, not logical negation. Rationale is "
            "audit text only.\n\nFROZEN_GRID="
            + canonical_json(grid).decode("utf-8")
            + "\nSUPPORT_VIEW="
            + canonical_json(packet_view).decode("utf-8")
        )
    if benchmark_mode != STRICT_DEV_MODE:
        raise RelationalHeadlessRunError("unknown prompt benchmark mode")
    if any(not isinstance(item.packet, LoopScenePacket) for item in panels):
        raise RelationalHeadlessRunError("strict DEV support requires loop packets")
    packet_view = {
        "schema": "gkm.bongard-relational-headless-support-view.v1",
        "panels": [
            {
                "presentation_name": item.presentation_name,
                "neutral_loop_scene": neutral_loop_scene_projection(item.packet),
            }
            for item in panels
        ],
    }
    grid = {
        "side_counts": list(ALLOWED_SIDE_COUNTS),
        "area_ratios": list(_RATIO_IDS),
        "denominator_obliqueness_millidegrees": [
            None,
            *ALLOWED_OBLIQUENESS_THRESHOLDS_MILLIDEGREES,
        ],
        "point_contact_enabled": False,
    }
    return (
        "You are proposing one closed, forward Bongard predicate from the twelve "
        "attached labelled support PNGs. The verifier has also supplied "
        "candidate-independent operational loop-scene observations below. Choose exactly one "
        "finite parameter tuple that means: there exist two distinct substantive "
        "closed loops; role 0 has the numerator side count; role 1 has the "
        "denominator side count; area(role0)/area(role1) is at most the chosen "
        "ratio; and, if selected, role 1 meets the obliqueness threshold. Every "
        "clause must hold on the same ordered pair. The predicate must be PRESENT "
        "on every pos panel and CERTIFIED_ABSENT on every neg panel. Do not swap "
        "the sides, negate a predicate, emit code, return alternatives, or infer "
        "anything about held-out panels. Point contact is disabled. Rationale is "
        "audit text only and is never executed.\n\n"
        "FROZEN_GRID="
        + canonical_json(grid).decode("utf-8")
        + "\nSUPPORT_VIEW="
        + canonical_json(packet_view).decode("utf-8")
    )


def parse_relational_proposal(payload: Mapping[str, Any]) -> RelationalVisualQuery:
    """Construct the only executable object from one strict model payload."""

    expected = {
        "numerator_side_count",
        "denominator_side_count",
        "area_ratio",
        "denominator_obliqueness_millidegrees",
        "rationale",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise RelationalHeadlessRunError("proposal fields differ from strict schema")
    numerator = payload["numerator_side_count"]
    denominator = payload["denominator_side_count"]
    ratio_id = payload["area_ratio"]
    obliqueness = payload["denominator_obliqueness_millidegrees"]
    rationale = payload["rationale"]
    if numerator not in ALLOWED_SIDE_COUNTS or denominator not in ALLOWED_SIDE_COUNTS:
        raise RelationalHeadlessRunError("proposal side count is outside frozen grid")
    if not isinstance(ratio_id, str) or ratio_id not in _RATIO_BY_ID:
        raise RelationalHeadlessRunError("proposal ratio is outside frozen grid")
    if obliqueness is not None and obliqueness not in (
        ALLOWED_OBLIQUENESS_THRESHOLDS_MILLIDEGREES
    ):
        raise RelationalHeadlessRunError(
            "proposal obliqueness is outside frozen grid"
        )
    if not isinstance(rationale, str):
        raise RelationalHeadlessRunError("proposal rationale must be text")
    try:
        rationale_bytes = rationale.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise RelationalHeadlessRunError(
            "proposal rationale is not clean UTF-8"
        ) from exc
    if (
        rationale != rationale.strip()
        or not 1 <= len(rationale_bytes) <= 1_024
        or any(unicodedata.category(character).startswith("C") for character in rationale)
    ):
        raise RelationalHeadlessRunError(
            "proposal rationale must be stripped, control-free UTF-8 in 1..1024 bytes"
        )
    query = RelationalVisualQuery.factorized_shape_ratio(
        numerator_side_count=numerator,
        denominator_side_count=denominator,
        ratio=Rational(*_RATIO_BY_ID[ratio_id]),
        denominator_obliqueness_millidegrees=obliqueness,
        require_point_contact=False,
    )
    if any(isinstance(clause, PointContactClause) for clause in query.clauses):
        raise RelationalHeadlessRunError("point-contact clause is disabled")
    replay = RelationalVisualQuery.from_data(query.to_data())
    if replay != query:
        raise RelationalHeadlessRunError("proposal query round trip differs")
    return query


def _validate_rationale(rationale: object) -> None:
    if not isinstance(rationale, str):
        raise RelationalHeadlessRunError("proposal rationale must be text")
    try:
        rationale_bytes = rationale.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise RelationalHeadlessRunError(
            "proposal rationale is not clean UTF-8"
        ) from exc
    if (
        rationale != rationale.strip()
        or not 1 <= len(rationale_bytes) <= 1_024
        or any(
            unicodedata.category(character).startswith("C")
            for character in rationale
        )
    ):
        raise RelationalHeadlessRunError(
            "proposal rationale must be stripped, control-free UTF-8 in 1..1024 bytes"
        )


def _require_complete_library_member(
    predicate: ClosedPanelPredicate,
    library: FrozenCompleteClosedLibraryIndex,
) -> ClosedPanelPredicate:
    """Require canonical construction membership in the pre-frozen full index."""

    if library != _complete_closed_library():
        raise RelationalHeadlessRunError(
            "closed predicate library index differs from the pre-frozen complete union"
        )
    try:
        replay = ClosedPanelPredicate.from_data(predicate.to_data())
    except (TypeError, ValueError) as exc:  # pragma: no cover - constructors close it.
        raise RelationalHeadlessRunError(
            "proposed predicate is not canonical closed-library data"
        ) from exc
    if replay != predicate:
        raise RelationalHeadlessRunError(
            "proposed predicate is not a member of the pre-frozen complete library"
        )
    if (
        predicate.kind is ClosedPredicateKind.RELATIONAL
        and isinstance(predicate.payload, RelationalVisualQuery)
        and any(
            isinstance(clause, PointContactClause)
            for clause in predicate.payload.clauses
        )
    ):
        raise RelationalHeadlessRunError(
            "point-contact predicate is not proposer-reachable in this library"
        )
    return predicate


def parse_closed_visual_proposal(
    payload: Mapping[str, Any],
    *,
    library: FrozenCompleteClosedLibraryIndex,
) -> ClosedPanelPredicate:
    """Parse one compact tagged payload into the closed executable union."""

    expected = {
        "kind",
        "relational",
        "direct_atom_0",
        "direct_atom_1",
        "direct_atom_2",
        "symmetry",
        "rationale",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise RelationalHeadlessRunError(
            "closed proposal fields differ from strict tagged schema"
        )
    _validate_rationale(payload["rationale"])
    try:
        kind = ClosedPredicateKind(payload["kind"])
    except (TypeError, ValueError) as exc:
        raise RelationalHeadlessRunError("closed proposal kind is invalid") from exc
    relational = payload["relational"]
    direct_slots = tuple(payload[f"direct_atom_{index}"] for index in range(3))
    symmetry = payload["symmetry"]
    if kind is ClosedPredicateKind.RELATIONAL:
        if (
            not isinstance(relational, Mapping)
            or any(item is not None for item in direct_slots)
            or symmetry is not None
        ):
            raise RelationalHeadlessRunError(
                "relational tag requires only its relational branch"
            )
        query = parse_relational_proposal(
            {**dict(relational), "rationale": payload["rationale"]}
        )
        predicate = ClosedPanelPredicate.relational(query)
    elif kind is ClosedPredicateKind.DIRECT_COUNTS:
        if relational is not None or symmetry is not None:
            raise RelationalHeadlessRunError(
                "direct-count tag requires inactive relational/symmetry branches"
            )
        nonnull = tuple(item for item in direct_slots if item is not None)
        if not 1 <= len(nonnull) <= 3 or direct_slots[: len(nonnull)] != nonnull:
            raise RelationalHeadlessRunError(
                "direct-count proposal requires one to three contiguous atoms"
            )
        selections: list[tuple[str, str, tuple[tuple[str, object], ...]]] = []
        for item in nonnull:
            if not isinstance(item, Mapping) or set(item) != {
                "catalog_key",
                "comparison",
                "target_count",
            }:
                raise RelationalHeadlessRunError("direct atom fields differ")
            try:
                spec = DIRECT_VISUAL_ATOM_CATALOG.get(item["catalog_key"])
                comparison, arguments = spec.canonical_selection(
                    item["comparison"],
                    {"target_count": item["target_count"]},
                    "proposal-atom",
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RelationalHeadlessRunError(
                    "direct atom lies outside the frozen catalog"
                ) from exc
            selections.append((spec.catalog_key, comparison, arguments))
        selections.sort(key=lambda item: item[0])
        if len({item[0] for item in selections}) != len(selections):
            raise RelationalHeadlessRunError(
                "direct count conjunction repeats a catalog capability"
            )
        atoms = tuple(
            TypedDeterministicAtom(
                atom_id=f"atom-{index:02d}",
                catalog_key=item[0],
                comparison=item[1],
                arguments=item[2],
            )
            for index, item in enumerate(selections)
        )
        predicate = ClosedPanelPredicate.direct(
            DirectCountPredicate(atoms, direct_visual_catalog_digest())
        )
    else:
        if (
            relational is not None
            or any(item is not None for item in direct_slots)
            or not isinstance(symmetry, Mapping)
            or set(symmetry) != {"metric", "threshold_ppm"}
        ):
            raise RelationalHeadlessRunError(
                "symmetry tag requires only its symmetry branch"
            )
        try:
            predicate = ClosedPanelPredicate.symmetry(
                SymmetryThresholdPredicate(
                    SymmetryMetric(symmetry["metric"]),
                    symmetry["threshold_ppm"],
                )
            )
        except (TypeError, ValueError) as exc:
            raise RelationalHeadlessRunError(
                "symmetry proposal lies outside the frozen grid"
            ) from exc
    return _require_complete_library_member(predicate, library)


def _task_root_after_exposure(corpus_root: str | Path, plan: RelationalHeadlessPlan) -> Path:
    """Construct the official lexical layout only after durable exposure.

    Deliberately do not call ``resolve`` or any path predicate here: those
    operations follow symlinks.  The component-by-component no-follow opener
    is the sole authority when an authorized panel is actually read.
    """

    root = Path(os.path.abspath(os.path.expanduser(str(corpus_root))))
    return root / plan.family / "images" / plan.task_id


def _support_paths(
    task_root: Path, plan: RelationalHeadlessPlan
) -> tuple[tuple[str, int, Path], ...]:
    result: list[tuple[str, int, Path]] = []
    for polarity, label, indices in (
        ("positive", "1", plan.positive_support_indices),
        ("negative", "0", plan.negative_support_indices),
    ):
        for index in indices:
            # The omitted held-out filename is never constructed here.
            result.append((polarity, index, task_root / label / f"{index}.png"))
    return tuple(result)


def _query_paths_after_freeze(
    task_root: Path, plan: RelationalHeadlessPlan
) -> tuple[tuple[str, int, Path], tuple[str, int, Path]]:
    by_side = {
        "positive": (
            plan._positive_query_index,
            task_root / "1" / f"{plan._positive_query_index}.png",
        ),
        "negative": (
            plan._negative_query_index,
            task_root / "0" / f"{plan._negative_query_index}.png",
        ),
    }
    result = tuple(
        (side, by_side[side][0], by_side[side][1])
        for side in plan._query_order
    )
    if len(result) != 2:
        raise RelationalHeadlessRunError("query schedule must contain two slots")
    return result[0], result[1]


def _extract_support(
    corpus_root: str | Path,
    task_root: Path,
    plan: RelationalHeadlessPlan,
    *,
    png_reader: PngReader,
    extractor: PacketExtractor,
    packet_verifier: PacketVerifier,
) -> tuple[_SupportPanel, ...]:
    raw: list[_SupportPanel] = []
    for polarity, index, path in _support_paths(task_root, plan):
        payload, release_receipt = _read_authenticated_release_panel(
            corpus_root=corpus_root,
            path=path,
            authenticator=plan._release_authenticator,
            observer_reader=png_reader,
        )
        packet = extractor(payload)
        packet_verifier(packet, expected_png_bytes=payload)
        digest = _raw_digest(payload)
        if packet.panel_digest != digest:
            raise RelationalHeadlessRunError("support packet names different pixels")
        raw.append(
            _SupportPanel(
                polarity,
                index,
                payload,
                digest,
                packet,
                release_receipt,
            )
        )
    named: list[_SupportPanel] = []
    for polarity, prefix in (("positive", "pos"), ("negative", "neg")):
        side = sorted(
            (item for item in raw if item.polarity == polarity),
            key=lambda item: (item.source_sha256, item.source_index),
        )
        if len(side) != 6:
            raise RelationalHeadlessRunError("support side does not contain six panels")
        named.extend(
            item.with_name(f"{prefix}_{slot}.png") for slot, item in enumerate(side)
        )
    return tuple(named)


def _stage_support_view(root: Path, panels: Sequence[_SupportPanel]) -> tuple[str, ...]:
    paths: list[str] = []
    for item in panels:
        path = root / item.presentation_name
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            view = memoryview(item.payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise RelationalHeadlessRunError("short support staging write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if _stable_read(path) != item.payload:
            raise RelationalHeadlessRunError("staged support bytes differ")
        paths.append(str(path))
    return tuple(paths)


def _receipt_dict(receipt: CodexReceipt | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(receipt, CodexReceipt):
        return receipt.to_dict()
    if not isinstance(receipt, Mapping):
        raise RelationalHeadlessRunError("transport receipt is not a mapping")
    return dict(receipt)


def _validate_transport_result(
    result: CodexStructuredResult,
    *,
    prompt: str,
    schema: Mapping[str, Any],
    support_paths: Sequence[str],
    plan: RelationalHeadlessPlan,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = getattr(result, "payload", None)
    receipt_value = getattr(result, "receipt", None)
    if not isinstance(payload, Mapping):
        raise RelationalHeadlessRunError("transport payload is not an object")
    payload_dict = dict(payload)
    receipt = _receipt_dict(receipt_value)
    validate_codex_receipt(receipt)
    expected = {
        "input_digest_schema": STRUCTURED_INPUT_DIGEST_SCHEMA,
        "requested_model": plan.model,
        "requested_reasoning_effort": plan.reasoning_effort,
        "codex_launcher_digest": plan.expected_launcher_digest,
        "cloud_config_bundle_cache_binding": plan.cloud_policy_cache_binding,
        "task_digest": _text_digest(prompt),
        "prompt_digest": _text_digest(prompt),
        "output_schema_digest": canonical_digest(schema),
        "panel_view_digest": ordered_panel_view_digest(support_paths),
        "panel_set_digest": semantic_panel_set_digest(support_paths),
        "structured_output_digest": canonical_digest(payload_dict),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RelationalHeadlessRunError(
                f"transport receipt {key} differs from frozen invocation"
            )
    return payload_dict, receipt


ExecutablePredicate = RelationalVisualQuery | ClosedPanelPredicate


def _proposal_schema_for_plan(plan: RelationalHeadlessPlan) -> dict[str, object]:
    return (
        closed_visual_proposal_schema()
        if plan.benchmark_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE
        else relational_proposal_schema()
    )


def _parse_proposal_for_plan(
    payload: Mapping[str, Any], plan: RelationalHeadlessPlan
) -> ExecutablePredicate:
    if plan.benchmark_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE:
        if plan._closed_library is None:
            raise RelationalHeadlessRunError("engineering plan lost its library")
        return parse_closed_visual_proposal(
            payload, library=plan._closed_library
        )
    if plan.benchmark_mode != STRICT_DEV_MODE:
        raise RelationalHeadlessRunError("unknown plan benchmark mode")
    return parse_relational_proposal(payload)


def _predicate_data(predicate: ExecutablePredicate) -> dict[str, object]:
    return predicate.to_data()


def _predicate_digest(predicate: ExecutablePredicate) -> str:
    return (
        predicate.digest
        if isinstance(predicate, ClosedPanelPredicate)
        else predicate.digest()
    )


def _packet_from_data(
    data: Mapping[str, Any], plan: RelationalHeadlessPlan
) -> PanelPacket:
    if plan.benchmark_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE:
        return ExactPanelWitnessPacket.from_data(data)
    return LoopScenePacket.from_data(data)


def _evaluate_predicate(
    predicate: ExecutablePredicate,
    packet: PanelPacket,
    plan: RelationalHeadlessPlan,
) -> Any:
    if plan.benchmark_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE:
        if not isinstance(predicate, ClosedPanelPredicate) or not isinstance(
            packet, ExactPanelWitnessPacket
        ):
            raise RelationalHeadlessRunError(
                "engineering predicate/packet types differ"
            )
        result = evaluate_closed_predicate(predicate, packet)
        verify_closed_predicate_result(result, predicate, packet)
        return result
    if not isinstance(predicate, RelationalVisualQuery) or not isinstance(
        packet, LoopScenePacket
    ):
        raise RelationalHeadlessRunError("strict DEV predicate/packet types differ")
    result = evaluate_relational_query(predicate, packet)
    verify_relational_query_result(result, predicate, packet)
    return result


def _result_digest(result: Any) -> str:
    digest = getattr(result, "digest")
    return digest() if callable(digest) else digest


def _neutral_projection(
    packet: PanelPacket, plan: RelationalHeadlessPlan
) -> dict[str, object]:
    if plan.benchmark_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE:
        if not isinstance(packet, ExactPanelWitnessPacket):
            raise RelationalHeadlessRunError("engineering packet is not composite")
        return neutral_closed_visual_projection(packet)
    if not isinstance(packet, LoopScenePacket):
        raise RelationalHeadlessRunError("strict DEV packet is not a loop scene")
    return neutral_loop_scene_projection(packet)


def _support_entry(
    panel: _SupportPanel,
    query: ExecutablePredicate,
    plan: RelationalHeadlessPlan,
) -> dict[str, object]:
    result = _evaluate_predicate(query, panel.packet, plan)
    projection = _neutral_projection(panel.packet, plan)
    return {
        "presentation_name": panel.presentation_name,
        "polarity": panel.polarity,
        "source_index": panel.source_index,
        "source_sha256": panel.source_sha256,
        "byte_count": len(panel.payload),
        "release_panel_receipt": dict(panel.release_panel_receipt),
        "packet": panel.packet.to_data(),
        "packet_digest": panel.packet.digest(),
        "neutral_projection": projection,
        "neutral_projection_digest": canonical_digest(projection),
        "query_result": result.to_data(),
        "query_result_digest": _result_digest(result),
    }


def _expected_release_relative_path(
    plan: RelationalHeadlessPlan,
    *,
    polarity: str,
    source_index: int,
) -> str:
    if polarity not in {"positive", "negative"}:
        raise RelationalHeadlessRunError("panel polarity is malformed")
    if (
        isinstance(source_index, bool)
        or not isinstance(source_index, int)
        or not 0 <= source_index <= 6
    ):
        raise RelationalHeadlessRunError("panel source index is malformed")
    label = "1" if polarity == "positive" else "0"
    return (
        f"{plan.family}/images/{plan.task_id}/{label}/{source_index}.png"
    )


def _verify_release_panel_receipt(
    receipt: object,
    *,
    plan: RelationalHeadlessPlan,
    polarity: str,
    source_index: int,
    source_sha256: str,
    byte_count: int,
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise RelationalHeadlessRunError("release panel receipt is not an object")
    data = dict(receipt)
    content = {key: value for key, value in data.items() if key != "digest"}
    relative_path = _expected_release_relative_path(
        plan, polarity=polarity, source_index=source_index
    )
    expected = {
        "schema": RELEASE_PANEL_RECEIPT_SCHEMA,
        "relative_path": relative_path,
        "archive_member": "ShapeBongard_V2/" + relative_path,
        "sha256": "sha256:" + _require_sha256(
            source_sha256, "panel source digest"
        ),
        "size_bytes": byte_count,
        "release_descriptor_digest": plan.release_descriptor_digest,
        "archive_digest": plan.release_archive_digest,
        "central_directory_digest": plan.release_central_directory_digest,
    }
    if (
        set(data)
        != {
            *expected,
            "zip_crc32",
            "digest",
        }
        or any(data.get(key) != value for key, value in expected.items())
        or isinstance(data.get("zip_crc32"), bool)
        or not isinstance(data.get("zip_crc32"), int)
        or not 0 <= data["zip_crc32"] <= 0xFFFFFFFF
        or data.get("digest") != canonical_digest(content)
    ):
        raise RelationalHeadlessRunError("release panel receipt differs")
    return data


def _support_identities(
    entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, object]]:
    identities: list[dict[str, object]] = []
    for entry in entries:
        name = entry.get("presentation_name")
        byte_count = entry.get("byte_count")
        source_sha256 = entry.get("source_sha256")
        if (
            not isinstance(name, str)
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
        ):
            raise RelationalHeadlessRunError("support byte identity is malformed")
        identities.append(
            {
                "name": name,
                "byte_count": byte_count,
                "content_digest": _require_sha256(
                    source_sha256, "support source digest"
                ),
            }
        )
    return identities


def _structured_input_digest(
    *,
    prompt: str,
    identities: Sequence[Mapping[str, object]],
    panel_view_digest: str,
    panel_set_digest: str,
    output_schema_digest: str,
) -> str:
    prompt_digest = _text_digest(prompt)
    return canonical_digest(
        {
            "schema": STRUCTURED_INPUT_DIGEST_SCHEMA,
            "task": prompt,
            "ordered_panel_identities": [dict(item) for item in identities],
            "panel_view_digest": panel_view_digest,
            "panel_set_digest": panel_set_digest,
            "prompt_digest": prompt_digest,
            "output_schema_digest": output_schema_digest,
        }
    )


def _gate_accepted(entries: Sequence[Mapping[str, Any]]) -> bool:
    for entry in entries:
        result = entry.get("query_result")
        if not isinstance(result, Mapping):
            return False
        expected = (
            Disposition.PRESENT.value
            if entry.get("polarity") == "positive"
            else Disposition.CERTIFIED_ABSENT.value
        )
        if result.get("disposition") != expected:
            return False
    return True


def _build_freeze(
    *,
    plan: RelationalHeadlessPlan,
    exposure_successor: ExposureLedger,
    prompt: str,
    schema: Mapping[str, Any],
    proposal_payload: Mapping[str, Any],
    receipt: Mapping[str, Any],
    query: ExecutablePredicate,
    panels: Sequence[_SupportPanel],
) -> dict[str, Any]:
    entries = tuple(_support_entry(panel, query, plan) for panel in panels)
    identities = _support_identities(entries)
    panel_view_digest = canonical_digest(identities)
    schema_digest = canonical_digest(schema)
    payload_digest = canonical_digest(proposal_payload)
    panel_set_digest = _require_address(
        receipt.get("panel_set_digest"), "receipt panel set digest"
    )
    input_digest = _structured_input_digest(
        prompt=prompt,
        identities=identities,
        panel_view_digest=panel_view_digest,
        panel_set_digest=panel_set_digest,
        output_schema_digest=schema_digest,
    )
    transport_request = {
        "input_digest_schema": STRUCTURED_INPUT_DIGEST_SCHEMA,
        "model": plan.model,
        "reasoning_effort": plan.reasoning_effort,
        "launcher_digest": plan.expected_launcher_digest,
        "cloud_policy_cache_binding": plan.cloud_policy_cache_binding,
        "prompt_digest": _text_digest(prompt),
        "output_schema_digest": schema_digest,
        "structured_output_digest": payload_digest,
        "panel_view_digest": panel_view_digest,
        "panel_set_digest": panel_set_digest,
        "input_digest": input_digest,
    }
    content = {
        "schema": FREEZE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "benchmark_mode": plan.benchmark_mode,
        "plan_digest": plan.digest,
        "exposure_successor_digest": exposure_successor.digest,
        "support_selection_hiding_commitment": (
            plan.support_selection_hiding_commitment
        ),
        "support_selection_opening": {
            "key": plan._support_selection_key,
            "selection": _support_selection_data(
                plan.task_id,
                plan.positive_support_indices,
                plan.negative_support_indices,
            ),
        },
        "support_prompt": prompt,
        "support_prompt_digest": _text_digest(prompt),
        "proposal_schema": dict(schema),
        "proposal_schema_digest": schema_digest,
        "proposal_payload": dict(proposal_payload),
        "proposal_payload_digest": payload_digest,
        "transport_request": transport_request,
        "transport_request_digest": canonical_digest(transport_request),
        "codex_receipt": dict(receipt),
        "codex_receipt_digest": receipt["receipt_digest"],
        "query": _predicate_data(query),
        "query_digest": _predicate_digest(query),
        "closed_predicate_binding": (
            None
            if plan.closed_predicate_binding is None
            else dict(plan.closed_predicate_binding)
        ),
        "point_contact_enabled": False,
        "support_gate_policy": {
            "positive_required": Disposition.PRESENT.value,
            "negative_required": Disposition.CERTIFIED_ABSENT.value,
            "forward_only": True,
            "negation_allowed": False,
            "polarity_flip_allowed": False,
        },
        "support_entries": list(entries),
        "support_gate_accepted": _gate_accepted(entries),
        "query_pixels_opened": False,
    }
    return _seal(content)


def verify_relational_proposal_freeze(
    value: Mapping[str, Any],
    *,
    plan: RelationalHeadlessPlan,
    exposure_successor: ExposureLedger,
) -> dict[str, Any]:
    """Cold-verify the typed predicate and all support packet/result records."""

    data = _verify_seal(value, FREEZE_SCHEMA)
    if not isinstance(plan, RelationalHeadlessPlan) or not isinstance(
        exposure_successor, ExposureLedger
    ):
        raise TypeError("freeze verification requires plan and exposure successor")
    if (
        data.get("protocol_id") != PROTOCOL_ID
        or data.get("benchmark_mode") != plan.benchmark_mode
        or data.get("plan_digest") != plan.digest
        or data.get("exposure_successor_digest") != exposure_successor.digest
        or data.get("support_selection_hiding_commitment")
        != plan.support_selection_hiding_commitment
        or data.get("point_contact_enabled") is not False
        or data.get("query_pixels_opened") is not False
        or data.get("closed_predicate_binding")
        != (
            None
            if plan.closed_predicate_binding is None
            else dict(plan.closed_predicate_binding)
        )
    ):
        raise RelationalHeadlessRunError("proposal freeze policy differs")
    schema = data.get("proposal_schema")
    payload = data.get("proposal_payload")
    query_data = data.get("query")
    receipt = data.get("codex_receipt")
    request = data.get("transport_request")
    if not all(
        isinstance(item, Mapping)
        for item in (schema, payload, query_data, receipt, request)
    ):
        raise RelationalHeadlessRunError("proposal freeze typed fields are malformed")
    validate_codex_strict_output_schema(schema)
    expected_schema = _proposal_schema_for_plan(plan)
    if dict(schema) != expected_schema or data.get(
        "proposal_schema_digest"
    ) != canonical_digest(expected_schema):
        raise RelationalHeadlessRunError("proposal schema digest differs")
    if data.get("proposal_payload_digest") != canonical_digest(payload):
        raise RelationalHeadlessRunError("proposal payload digest differs")
    if plan.benchmark_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE:
        query: ExecutablePredicate = ClosedPanelPredicate.from_data(query_data)
    else:
        query = RelationalVisualQuery.from_data(query_data)
    if (
        query != _parse_proposal_for_plan(payload, plan)
        or data.get("query_digest") != _predicate_digest(query)
    ):
        raise RelationalHeadlessRunError("frozen query differs from finite proposal")
    if isinstance(query, RelationalVisualQuery) and any(
        isinstance(clause, PointContactClause) for clause in query.clauses
    ):
        raise RelationalHeadlessRunError("frozen query contains disabled point contact")
    entries = data.get("support_entries")
    if not isinstance(entries, list) or len(entries) != 12:
        raise RelationalHeadlessRunError("proposal freeze must cover twelve supports")
    expected_names = [f"pos_{index}.png" for index in range(6)] + [
        f"neg_{index}.png" for index in range(6)
    ]
    if [entry.get("presentation_name") for entry in entries] != expected_names:
        raise RelationalHeadlessRunError("support presentation names differ")
    reconstructed_panels: list[_SupportPanel] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise RelationalHeadlessRunError("support entry is not an object")
        packet_data = entry.get("packet")
        result_data = entry.get("query_result")
        projection = entry.get("neutral_projection")
        if not all(
            isinstance(item, Mapping)
            for item in (packet_data, result_data, projection)
        ):
            raise RelationalHeadlessRunError("support entry typed data is malformed")
        packet = _packet_from_data(packet_data, plan)
        if entry.get("packet_digest") != packet.digest():
            raise RelationalHeadlessRunError("support packet digest differs")
        if entry.get("source_sha256") != packet.panel_digest:
            raise RelationalHeadlessRunError("support packet/pixel digest differs")
        expected_projection = _neutral_projection(packet, plan)
        if dict(projection) != expected_projection or entry.get(
            "neutral_projection_digest"
        ) != canonical_digest(expected_projection):
            raise RelationalHeadlessRunError("neutral support projection differs")
        replay = _evaluate_predicate(query, packet, plan)
        if dict(result_data) != replay.to_data() or entry.get(
            "query_result_digest"
        ) != _result_digest(replay):
            raise RelationalHeadlessRunError("support query result differs from replay")
        expected_polarity = "positive" if index < 6 else "negative"
        if entry.get("polarity") != expected_polarity:
            raise RelationalHeadlessRunError("support polarity/order differs")
        source_index = entry.get("source_index")
        byte_count = entry.get("byte_count")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or not 0 <= source_index <= 6
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
        ):
            raise RelationalHeadlessRunError("support source metadata is malformed")
        release_receipt = _verify_release_panel_receipt(
            entry.get("release_panel_receipt"),
            plan=plan,
            polarity=expected_polarity,
            source_index=source_index,
            source_sha256=packet.panel_digest,
            byte_count=byte_count,
        )
        reconstructed_panels.append(
            _SupportPanel(
                expected_polarity,
                source_index,
                b"",
                packet.panel_digest,
                packet,
                release_receipt,
                entry["presentation_name"],
            )
        )
    for polarity in ("positive", "negative"):
        indices = tuple(sorted(
            item.source_index
            for item in reconstructed_panels
            if item.polarity == polarity
        ))
        if len(indices) != len(set(indices)):
            raise RelationalHeadlessRunError("support side repeats a source index")
        expected_indices = (
            plan.positive_support_indices
            if polarity == "positive"
            else plan.negative_support_indices
        )
        if indices != expected_indices:
            raise RelationalHeadlessRunError(
                "support source indices differ from the frozen held-out schedule"
            )
    support_selection = _support_selection_data(
        plan.task_id,
        plan.positive_support_indices,
        plan.negative_support_indices,
    )
    opening = data.get("support_selection_opening")
    if (
        not isinstance(opening, Mapping)
        or set(opening) != {"key", "selection"}
        or opening.get("key") != plan._support_selection_key
        or opening.get("selection") != support_selection
        or _support_selection_hiding_commitment(
            support_selection, plan._support_selection_key
        )
        != plan.support_selection_hiding_commitment
    ):
        raise RelationalHeadlessRunError(
            "support selection hiding commitment does not open"
        )
    if data.get("support_gate_accepted") is not _gate_accepted(entries):
        raise RelationalHeadlessRunError("support gate decision differs")
    support_prompt = data.get("support_prompt")
    expected_prompt = _proposal_prompt(
        reconstructed_panels, benchmark_mode=plan.benchmark_mode
    )
    if (
        not isinstance(support_prompt, str)
        or support_prompt != expected_prompt
        or data.get("support_prompt_digest") != _text_digest(expected_prompt)
    ):
        raise RelationalHeadlessRunError("support prompt digest differs")
    request_keys = {
        "input_digest_schema",
        "model",
        "reasoning_effort",
        "launcher_digest",
        "cloud_policy_cache_binding",
        "prompt_digest",
        "output_schema_digest",
        "structured_output_digest",
        "panel_view_digest",
        "panel_set_digest",
        "input_digest",
    }
    if set(request) != request_keys:
        raise RelationalHeadlessRunError("transport request fields differ")
    model = request.get("model")
    reasoning_effort = request.get("reasoning_effort")
    launcher_digest = request.get("launcher_digest")
    cache_binding = request.get("cloud_policy_cache_binding")
    if not isinstance(model, str) or _MODEL.fullmatch(model) is None:
        raise RelationalHeadlessRunError("frozen transport model is malformed")
    if reasoning_effort not in _REASONING_EFFORTS:
        raise RelationalHeadlessRunError("frozen reasoning effort differs")
    _require_sha256(launcher_digest, "frozen launcher digest")
    if cache_binding != "absent":
        _require_address(cache_binding, "frozen cache binding")
    identities = _support_identities(entries)
    panel_view_digest = canonical_digest(identities)
    panel_set_digest = _require_address(
        request.get("panel_set_digest"), "frozen panel set digest"
    )
    schema_digest = canonical_digest(expected_schema)
    payload_digest = canonical_digest(payload)
    expected_request = {
        "input_digest_schema": STRUCTURED_INPUT_DIGEST_SCHEMA,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "launcher_digest": launcher_digest,
        "cloud_policy_cache_binding": cache_binding,
        "prompt_digest": _text_digest(expected_prompt),
        "output_schema_digest": schema_digest,
        "structured_output_digest": payload_digest,
        "panel_view_digest": panel_view_digest,
        "panel_set_digest": panel_set_digest,
        "input_digest": _structured_input_digest(
            prompt=expected_prompt,
            identities=identities,
            panel_view_digest=panel_view_digest,
            panel_set_digest=panel_set_digest,
            output_schema_digest=schema_digest,
        ),
    }
    if dict(request) != expected_request or data.get(
        "transport_request_digest"
    ) != canonical_digest(expected_request):
        raise RelationalHeadlessRunError("transport request commitment differs")
    validate_codex_receipt(receipt)
    receipt_expected = {
        "input_digest_schema": STRUCTURED_INPUT_DIGEST_SCHEMA,
        "requested_model": model,
        "requested_reasoning_effort": reasoning_effort,
        "codex_launcher_digest": launcher_digest,
        "cloud_config_bundle_cache_binding": cache_binding,
        "task_digest": expected_request["prompt_digest"],
        "prompt_digest": expected_request["prompt_digest"],
        "output_schema_digest": schema_digest,
        "structured_output_digest": payload_digest,
        "panel_view_digest": panel_view_digest,
        "panel_set_digest": panel_set_digest,
        "input_digest": expected_request["input_digest"],
    }
    if any(receipt.get(key) != expected for key, expected in receipt_expected.items()):
        raise RelationalHeadlessRunError("Codex receipt differs from frozen request")
    if data.get("codex_receipt_digest") != receipt.get("receipt_digest"):
        raise RelationalHeadlessRunError("proposal receipt digest differs")
    return data


def _query_entry(
    *,
    query_id: str,
    source_index: int,
    payload: bytes,
    packet: PanelPacket,
    query: ExecutablePredicate,
    plan: RelationalHeadlessPlan,
    release_panel_receipt: Mapping[str, Any],
) -> dict[str, object]:
    result = _evaluate_predicate(query, packet, plan)
    prediction: bool | None
    if result.disposition is Disposition.PRESENT:
        prediction = True
    elif result.disposition is Disposition.CERTIFIED_ABSENT:
        prediction = False
    else:
        prediction = None
    return {
        "query_id": query_id,
        "source_index": source_index,
        "source_sha256": _raw_digest(payload),
        "byte_count": len(payload),
        "release_panel_receipt": dict(release_panel_receipt),
        "packet": packet.to_data(),
        "packet_digest": packet.digest(),
        "query_result": result.to_data(),
        "query_result_digest": _result_digest(result),
        "predicted_positive": prediction,
    }


def _build_predictions(
    *,
    freeze: Mapping[str, Any],
    query: ExecutablePredicate,
    plan: RelationalHeadlessPlan,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return _seal(
        {
            "schema": PREDICTION_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "benchmark_mode": plan.benchmark_mode,
            "freeze_digest": freeze["digest"],
            "query_digest": _predicate_digest(query),
            "closed_predicate_binding": (
                None
                if plan.closed_predicate_binding is None
                else dict(plan.closed_predicate_binding)
            ),
            "joint": True,
            "labels_revealed": False,
            "entries": [dict(item) for item in entries],
        }
    )


def verify_relational_predictions(
    value: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
    plan: RelationalHeadlessPlan,
    exposure_successor: ExposureLedger,
) -> dict[str, Any]:
    data = _verify_seal(value, PREDICTION_SCHEMA)
    verified_freeze = verify_relational_proposal_freeze(
        freeze,
        plan=plan,
        exposure_successor=exposure_successor,
    )
    if (
        data.get("protocol_id") != PROTOCOL_ID
        or data.get("benchmark_mode") != plan.benchmark_mode
        or data.get("freeze_digest") != verified_freeze["digest"]
        or data.get("joint") is not True
        or data.get("labels_revealed") is not False
        or data.get("closed_predicate_binding")
        != (
            None
            if plan.closed_predicate_binding is None
            else dict(plan.closed_predicate_binding)
        )
    ):
        raise RelationalHeadlessRunError("prediction commitment policy differs")
    query_data = verified_freeze["query"]
    if not isinstance(query_data, Mapping):
        raise RelationalHeadlessRunError("frozen query is not an object")
    query: ExecutablePredicate = (
        ClosedPanelPredicate.from_data(query_data)
        if plan.benchmark_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE
        else RelationalVisualQuery.from_data(query_data)
    )
    if data.get("query_digest") != _predicate_digest(query):
        raise RelationalHeadlessRunError("prediction query digest differs")
    entries = data.get("entries")
    if not isinstance(entries, list) or len(entries) != 2:
        raise RelationalHeadlessRunError("joint prediction must contain two entries")
    if [item.get("query_id") for item in entries] != ["query-0", "query-1"]:
        raise RelationalHeadlessRunError("prediction query IDs/order differ")
    expected_source_indices = [
        (
            plan._positive_query_index
            if side == "positive"
            else plan._negative_query_index
        )
        for side in plan._query_order
    ]
    for slot, (entry, expected_source_index) in enumerate(
        zip(entries, expected_source_indices, strict=True)
    ):
        if not isinstance(entry, Mapping):
            raise RelationalHeadlessRunError("prediction entry is not an object")
        packet_data = entry.get("packet")
        result_data = entry.get("query_result")
        if not isinstance(packet_data, Mapping) or not isinstance(result_data, Mapping):
            raise RelationalHeadlessRunError("prediction typed data is malformed")
        packet = _packet_from_data(packet_data, plan)
        source_index = entry.get("source_index")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index != expected_source_index
        ):
            raise RelationalHeadlessRunError(
                "prediction source index differs from held-out schedule"
            )
        if (
            entry.get("packet_digest") != packet.digest()
            or entry.get("source_sha256") != packet.panel_digest
        ):
            raise RelationalHeadlessRunError("prediction packet binding differs")
        polarity = plan._query_order[slot]
        _verify_release_panel_receipt(
            entry.get("release_panel_receipt"),
            plan=plan,
            polarity=polarity,
            source_index=source_index,
            source_sha256=packet.panel_digest,
            byte_count=entry.get("byte_count"),
        )
        replay = _evaluate_predicate(query, packet, plan)
        if (
            dict(result_data) != replay.to_data()
            or entry.get("query_result_digest") != _result_digest(replay)
        ):
            raise RelationalHeadlessRunError("prediction result differs from replay")
        expected_prediction = (
            True
            if replay.disposition is Disposition.PRESENT
            else False
            if replay.disposition is Disposition.CERTIFIED_ABSENT
            else None
        )
        if entry.get("predicted_positive") is not expected_prediction:
            raise RelationalHeadlessRunError("Boolean prediction differs from disposition")
    return data


def _verify_schedule_reveal(
    plan: RelationalHeadlessPlan, reveal: Mapping[str, Any]
) -> None:
    expected = plan.reveal_schedule()
    if dict(reveal) != expected:
        raise RelationalHeadlessRunError("query schedule reveal differs from live plan")
    if _text_digest(reveal["seed"]) != plan.seed_digest:
        raise RelationalHeadlessRunError("revealed seed differs from commitment")
    if _draw(plan.task_id, reveal["seed"], "positive-query", 7) != reveal[
        "positive_query_index"
    ]:
        raise RelationalHeadlessRunError("positive held-out index differs from seed")
    if _draw(plan.task_id, reveal["seed"], "negative-query", 7) != reveal[
        "negative_query_index"
    ]:
        raise RelationalHeadlessRunError("negative held-out index differs from seed")
    expected_order = (
        ["positive", "negative"]
        if _draw(plan.task_id, reveal["seed"], "query-order", 2) == 0
        else ["negative", "positive"]
    )
    if reveal["query_order"] != expected_order:
        raise RelationalHeadlessRunError("query order differs from seed")
    sealed = canonical_digest(
        {
            "schema": "gkm.bongard-relational-query-schedule-seal.v1",
            "positive_query_index": reveal["positive_query_index"],
            "negative_query_index": reveal["negative_query_index"],
            "query_order": reveal["query_order"],
            "nonce": reveal["label_nonce"],
        }
    )
    if sealed != plan.query_schedule_commitment:
        raise RelationalHeadlessRunError("query schedule commitment does not open")


@dataclass(frozen=True, slots=True)
class RelationalHeadlessOutcome:
    status: str
    plan: RelationalHeadlessPlan
    exposure_successor: ExposureLedger
    plan_path: Path
    exposure_path: Path
    terminal_path: Path
    artifact: Mapping[str, Any]
    freeze_path: Path | None = None
    prediction_path: Path | None = None

    def to_data(self) -> dict[str, object]:
        return {
            "status": self.status,
            "plan_digest": self.plan.digest,
            "exposure_successor_digest": self.exposure_successor.digest,
            "plan_path": str(self.plan_path),
            "exposure_path": str(self.exposure_path),
            "freeze_path": None if self.freeze_path is None else str(self.freeze_path),
            "prediction_path": (
                None if self.prediction_path is None else str(self.prediction_path)
            ),
            "terminal_path": str(self.terminal_path),
            "terminal_digest": self.artifact["digest"],
        }


def _terminal_failure(
    *,
    plan: RelationalHeadlessPlan,
    successor: ExposureLedger,
    phase: str,
    error: BaseException,
    artifact_store: str | Path,
    plan_path: Path,
    exposure_path: Path,
    freeze: Mapping[str, Any] | None = None,
    freeze_path: Path | None = None,
    predictions: Mapping[str, Any] | None = None,
    prediction_path: Path | None = None,
    labels_revealed: bool = False,
) -> RelationalHeadlessOutcome:
    content = {
        "schema": FAILURE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "benchmark_mode": plan.benchmark_mode,
        "status": "terminal_failure",
        "phase": phase,
        "error_type": type(error).__module__ + "." + type(error).__qualname__,
        "error_message": str(error) or type(error).__name__,
        "plan_digest": plan.digest,
        "exposure_successor_digest": successor.digest,
        "freeze_digest": None if freeze is None else freeze.get("digest"),
        "prediction_digest": (
            None if predictions is None else predictions.get("digest")
        ),
        "query_labels_revealed": labels_revealed,
        "reroll_attempted": False,
    }
    artifact = _seal(content)
    terminal_path, reloaded = _persist_artifact(
        artifact_store, artifact, suffix="relational-headless-failure"
    )
    _verify_seal(reloaded, FAILURE_SCHEMA)
    return RelationalHeadlessOutcome(
        "terminal_failure",
        plan,
        successor,
        plan_path,
        exposure_path,
        terminal_path,
        reloaded,
        freeze_path=freeze_path,
        prediction_path=prediction_path,
    )


def run_relational_headless(
    *,
    corpus_root: str | Path,
    task_id: str,
    split_index: SplitIndex,
    predecessor: ExposureLedger,
    expected_corpus_digest: str,
    expected_split_source_digest: str,
    expected_exposure_predecessor_digest: str,
    seed: str,
    exposure_observed_at: str,
    exposure_store: str | Path,
    artifact_store: str | Path,
    expected_launcher_digest: str,
    release_authenticator: ReleaseArchiveAuthenticator,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot | None = None,
    model: str = DEFAULT_CODEX_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    minutes: int = 15,
    executable: str = "codex",
    verbose: bool = False,
    transport: StructuredTransport = run_codex_structured,
    png_reader: PngReader = _read_png_no_follow,
    extractor: PacketExtractor | None = None,
    packet_verifier: PacketVerifier | None = None,
    label_nonce: str | None = None,
    support_selection_key: str | None = None,
    precommitted_exposure_successor: ExposureLedger | None = None,
    precommitted_exposure_path: str | Path | None = None,
    precommitted_campaign_task_ids: Sequence[str] = (),
    precommitted_campaign_source: str | None = None,
    precommitted_campaign_task_plan_digest: str | None = None,
    benchmark_mode: str = STRICT_DEV_MODE,
    closed_library: FrozenCompleteClosedLibraryIndex | None = None,
) -> RelationalHeadlessOutcome:
    """Run exactly one explicit-task proposal and model-free query replay.

    There is one and only one transport call.  Every exception after the
    exposure edge becomes a write-once terminal artifact; no exception path
    retries the proposer or opens a held-out path before a verified freeze.
    """

    if extractor is None:
        extractor = (
            extract_exact_panel_witness_packet
            if benchmark_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE
            else extract_loop_scene_witnesses
        )
    if packet_verifier is None:
        packet_verifier = (
            verify_exact_panel_witness_packet
            if benchmark_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE
            else verify_loop_scene_packet
        )
    if not all(callable(item) for item in (transport, png_reader, extractor, packet_verifier)):
        raise TypeError("runner dependencies must be callable")
    cache = cloud_policy_cache_snapshot or CloudPolicyCacheSnapshot(None)
    plan = prepare_relational_headless_plan(
        task_id=task_id,
        split_index=split_index,
        predecessor=predecessor,
        expected_exposure_predecessor_digest=(
            expected_exposure_predecessor_digest
        ),
        expected_corpus_digest=expected_corpus_digest,
        expected_split_source_digest=expected_split_source_digest,
        seed=seed,
        exposure_observed_at=exposure_observed_at,
        expected_launcher_digest=expected_launcher_digest,
        release_authenticator=release_authenticator,
        cloud_policy_cache_snapshot=cache,
        model=model,
        reasoning_effort=reasoning_effort,
        minutes=minutes,
        label_nonce=label_nonce,
        support_selection_key=support_selection_key,
        benchmark_mode=benchmark_mode,
        closed_library=closed_library,
    )
    if precommitted_campaign_task_plan_digest is not None and (
        plan.digest
        != _require_sha256(
            precommitted_campaign_task_plan_digest,
            "precommitted campaign task plan digest",
        )
    ):
        raise RelationalHeadlessRunError(
            "live task plan differs from campaign precommit"
        )
    plan_artifact = _seal(plan.to_data())
    if plan_artifact["digest"] != plan.digest:
        raise RelationalHeadlessRunError("plan artifact digest differs")
    plan_path, reloaded_plan = _persist_artifact(
        artifact_store, plan_artifact, suffix="relational-headless-plan"
    )
    _verify_seal(reloaded_plan, PLAN_SCHEMA)

    # This is the last metadata-only step.  It must be durable and exactly
    # reloaded before corpus_root is resolved or a task/panel Path is formed.
    known_task_ids = set().union(
        split_index.canonical_groups["train"],
        split_index.canonical_groups["val"],
        split_index.canonical_groups["test"],
    )
    if precommitted_exposure_successor is None:
        if (
            precommitted_exposure_path is not None
            or precommitted_campaign_task_ids
            or precommitted_campaign_source is not None
            or precommitted_campaign_task_plan_digest is not None
        ):
            raise RelationalHeadlessRunError(
                "partial campaign exposure authorization is forbidden"
            )
        if plan.benchmark_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE:
            exposure_phase = ENGINEERING_EXPOSURE_PHASE
            exposure_actor = ENGINEERING_EXPOSURE_ACTOR
            exposure_purpose = ENGINEERING_EXPOSURE_PURPOSE
        else:
            exposure_phase = "relational-headless-benchmark"
            exposure_actor = "headless-codex-relational-proposer"
            exposure_purpose = "one-shot support proposal and two-query Python replay"
        successor = predecessor.record(
            phase=exposure_phase,
            actor=exposure_actor,
            purpose=exposure_purpose,
            task_ids=(task_id,),
            source=f"{PROTOCOL_ID}:plan:{plan.digest}",
            observed_at=exposure_observed_at,
            known_task_ids=known_task_ids,
            sealed_task_ids=split_index.canonical_groups["test"],
            require_unseen=True,
        )
        exposure_path, reloaded_successor = _persist_exposure(
            successor, exposure_store
        )
        if reloaded_successor != successor:
            raise RelationalHeadlessRunError("exposure reload differs before pixels")
    else:
        successor = precommitted_exposure_successor
        expected_campaign_authorization = (
            (
                ENGINEERING_CAMPAIGN_AUTHORIZATION_PHASE,
                ENGINEERING_CAMPAIGN_AUTHORIZATION_ACTOR,
                ENGINEERING_CAMPAIGN_AUTHORIZATION_PURPOSE,
            )
            if plan.benchmark_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE
            else (
                CAMPAIGN_AUTHORIZATION_PHASE,
                CAMPAIGN_AUTHORIZATION_ACTOR,
                CAMPAIGN_AUTHORIZATION_PURPOSE,
            )
        )
        if (
            not isinstance(successor, ExposureLedger)
            or precommitted_exposure_path is None
            or not precommitted_campaign_task_ids
            or not isinstance(precommitted_campaign_source, str)
            or not precommitted_campaign_source.strip()
            or precommitted_campaign_task_plan_digest is None
        ):
            raise RelationalHeadlessRunError(
                "campaign exposure authorization is incomplete"
            )
        expected_campaign_ids = tuple(sorted(set(precommitted_campaign_task_ids)))
        if (
            len(successor.events) != len(predecessor.events) + 1
            or successor.events[:-1] != predecessor.events
            or successor.corpus_digest != predecessor.corpus_digest
        ):
            raise RelationalHeadlessRunError(
                "campaign successor is not one atomic edge after predecessor"
            )
        campaign_event = successor.events[-1]
        if (
            campaign_event.task_ids != expected_campaign_ids
            or task_id not in expected_campaign_ids
            or campaign_event.panel_ids
            or campaign_event.phase != expected_campaign_authorization[0]
            or campaign_event.actor != expected_campaign_authorization[1]
            or campaign_event.purpose != expected_campaign_authorization[2]
            or campaign_event.source != precommitted_campaign_source
            or campaign_event.observed_at != exposure_observed_at
        ):
            raise RelationalHeadlessRunError(
                "campaign exposure edge differs from frozen authorization"
            )
        exposure_path = Path(precommitted_exposure_path)
        if ExposureLedger.load(exposure_path) != successor:
            raise RelationalHeadlessRunError(
                "campaign exposure successor differs on cold reload"
            )

    phase = "support-path-resolution"
    freeze: dict[str, Any] | None = None
    freeze_path: Path | None = None
    predictions: dict[str, Any] | None = None
    prediction_path: Path | None = None
    labels_revealed = False
    try:
        task_root = _task_root_after_exposure(corpus_root, plan)
        phase = "support-extraction"
        panels = _extract_support(
            corpus_root,
            task_root,
            plan,
            png_reader=png_reader,
            extractor=extractor,
            packet_verifier=packet_verifier,
        )
        prompt = _proposal_prompt(panels, benchmark_mode=plan.benchmark_mode)
        schema = _proposal_schema_for_plan(plan)
        phase = "single-codex-proposal"
        # No loop, retry wrapper, backup proposal, or candidate enumeration.
        with tempfile.TemporaryDirectory(
            prefix="bongard-relational-support-"
        ) as directory:
            support_paths = _stage_support_view(Path(directory), panels)
            result = transport(
                prompt,
                support_paths,
                schema,
                model=plan.model,
                reasoning_effort=plan.reasoning_effort,
                minutes=plan.minutes,
                verbose=verbose,
                executable=executable,
                cloud_policy_cache_snapshot=cache,
                expected_launcher_digest=plan.expected_launcher_digest,
            )
            proposal_payload, receipt = _validate_transport_result(
                result,
                prompt=prompt,
                schema=schema,
                support_paths=support_paths,
                plan=plan,
            )
            phase = "proposal-parse"
            query = _parse_proposal_for_plan(proposal_payload, plan)
            phase = "support-forward-replay"
            freeze = _build_freeze(
                plan=plan,
                exposure_successor=successor,
                prompt=prompt,
                schema=schema,
                proposal_payload=proposal_payload,
                receipt=receipt,
                query=query,
                panels=panels,
            )
            phase = "proposal-freeze-persistence"
            freeze_path, reloaded_freeze = _persist_artifact(
                artifact_store,
                freeze,
                suffix="relational-proposal-freeze",
            )
            freeze = verify_relational_proposal_freeze(
                reloaded_freeze,
                plan=plan,
                exposure_successor=successor,
            )

        if freeze["support_gate_accepted"] is not True:
            terminal = _seal(
                {
                    "schema": RUN_SCHEMA,
                    "protocol_id": PROTOCOL_ID,
                    "benchmark_mode": plan.benchmark_mode,
                    "status": "support_rejected",
                    "plan_digest": plan.digest,
                    "exposure_successor_digest": successor.digest,
                    "freeze_digest": freeze["digest"],
                    "prediction_digest": None,
                    "query_paths_resolved": False,
                    "query_pixels_opened": False,
                    "query_labels_revealed": False,
                    "reroll_attempted": False,
                }
            )
            terminal_path, reloaded = _persist_artifact(
                artifact_store, terminal, suffix="relational-headless-run"
            )
            _verify_seal(reloaded, RUN_SCHEMA)
            return RelationalHeadlessOutcome(
                "support_rejected",
                plan,
                successor,
                plan_path,
                exposure_path,
                terminal_path,
                reloaded,
                freeze_path=freeze_path,
            )

        # The held-out Path objects are first constructed here, strictly after
        # the freeze file has been fsynced, reloaded, and cold-verified.
        phase = "query-path-resolution"
        query_paths = _query_paths_after_freeze(task_root, plan)
        query_runtime: list[
            tuple[
                str,
                str,
                int,
                bytes,
                PanelPacket,
                Mapping[str, Any],
            ]
        ] = []
        phase = "query-extraction"
        for slot, (polarity, source_index, path) in enumerate(query_paths):
            payload, release_receipt = _read_authenticated_release_panel(
                corpus_root=corpus_root,
                path=path,
                authenticator=plan._release_authenticator,
                observer_reader=png_reader,
            )
            packet = extractor(payload)
            packet_verifier(packet, expected_png_bytes=payload)
            if packet.panel_digest != _raw_digest(payload):
                raise RelationalHeadlessRunError("query packet names different pixels")
            query_runtime.append(
                (
                    f"query-{slot}",
                    polarity,
                    source_index,
                    payload,
                    packet,
                    release_receipt,
                )
            )
        phase = "joint-prediction-persistence"
        prediction_entries = tuple(
            _query_entry(
                query_id=query_id,
                source_index=source_index,
                payload=payload,
                packet=packet,
                query=query,
                plan=plan,
                release_panel_receipt=release_receipt,
            )
            for (
                query_id,
                _polarity,
                source_index,
                payload,
                packet,
                release_receipt,
            ) in query_runtime
        )
        predictions = _build_predictions(
            freeze=freeze,
            query=query,
            plan=plan,
            entries=prediction_entries,
        )
        prediction_path, reloaded_predictions = _persist_artifact(
            artifact_store,
            predictions,
            suffix="relational-predictions",
        )
        predictions = verify_relational_predictions(
            reloaded_predictions,
            freeze=freeze,
            plan=plan,
            exposure_successor=successor,
        )

        # Only the exact durable joint commitment authorizes materializing the
        # polarity mapping and scoring it.
        phase = "label-reveal-and-score"
        schedule_reveal = plan.reveal_schedule()
        _verify_schedule_reveal(plan, schedule_reveal)
        labels = [
            {
                "query_id": query_id,
                "positive": polarity == "positive",
                "source_index": source_index,
            }
            for (
                query_id,
                polarity,
                source_index,
                _payload,
                _packet,
                _release_receipt,
            ) in query_runtime
        ]
        labels_revealed = True
        prediction_by_id = {
            item["query_id"]: item["predicted_positive"]
            for item in predictions["entries"]
        }
        correct = sum(
            prediction_by_id[item["query_id"]] is item["positive"]
            for item in labels
        )
        abstentions = sum(
            prediction_by_id[item["query_id"]] is None for item in labels
        )
        errors = sum(
            item["query_result"]["disposition"] == Disposition.ERROR.value
            for item in predictions["entries"]
        )
        final = _seal(
            {
                "schema": RUN_SCHEMA,
                "protocol_id": PROTOCOL_ID,
                "benchmark_mode": plan.benchmark_mode,
                "status": "complete",
                "plan_digest": plan.digest,
                "exposure_successor_digest": successor.digest,
                "freeze_digest": freeze["digest"],
                "prediction_digest": predictions["digest"],
                "schedule_reveal": schedule_reveal,
                "labels": labels,
                "selected_panel_manifest": {
                    "schema": "gkm.bongard-selected-release-panels.v1",
                    "release_authentication": plan.to_data()[
                        "release_authentication"
                    ],
                    "support": [
                        dict(item.release_panel_receipt) for item in panels
                    ],
                    "query": [
                        {
                            "query_id": query_id,
                            "release_panel_receipt": dict(release_receipt),
                        }
                        for (
                            query_id,
                            _polarity,
                            _source_index,
                            _payload,
                            _packet,
                            release_receipt,
                        ) in query_runtime
                    ],
                },
                "score": {
                    "correct": correct,
                    "total": 2,
                    "abstentions": abstentions,
                    "errors": errors,
                },
                "query_paths_resolved_after_freeze": True,
                "predictions_persisted_before_labels": True,
                "reroll_attempted": False,
            }
        )
        terminal_path, reloaded_final = _persist_artifact(
            artifact_store, final, suffix="relational-headless-run"
        )
        _verify_seal(reloaded_final, RUN_SCHEMA)
        return RelationalHeadlessOutcome(
            "complete",
            plan,
            successor,
            plan_path,
            exposure_path,
            terminal_path,
            reloaded_final,
            freeze_path=freeze_path,
            prediction_path=prediction_path,
        )
    except Exception as exc:  # noqa: BLE001 - terminalize every post-exposure fault.
        return _terminal_failure(
            plan=plan,
            successor=successor,
            phase=phase,
            error=exc,
            artifact_store=artifact_store,
            plan_path=plan_path,
            exposure_path=exposure_path,
            freeze=freeze,
            freeze_path=freeze_path,
            predictions=predictions,
            prediction_path=prediction_path,
            labels_revealed=labels_revealed,
        )


def load_relational_artifact(path: str | Path) -> dict[str, Any]:
    """Load one canonical write-once runner artifact without trusting its name."""

    payload = _stable_read(Path(path))
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RelationalHeadlessRunError("runner artifact is malformed JSON") from exc
    if not isinstance(value, dict) or canonical_json(value) + b"\n" != payload:
        raise RelationalHeadlessRunError("runner artifact is not canonical")
    digest = _require_sha256(value.get("digest"), "runner artifact digest")
    if digest not in Path(path).name:
        raise RelationalHeadlessRunError("runner artifact filename/address differs")
    return value


def cold_replay_relational_headless_run(
    *,
    plan: RelationalHeadlessPlan,
    exposure_successor: ExposureLedger,
    freeze: Mapping[str, Any],
    predictions: Mapping[str, Any],
    final_run: Mapping[str, Any],
    support_png_bytes: Mapping[str, bytes],
    query_png_bytes: Mapping[str, bytes],
    release_authenticator: ReleaseArchiveAuthenticator,
) -> Mapping[str, Any]:
    """Re-extract exact pixels and reproduce the complete score model-free."""

    exact_packet_cache: dict[str, ExactPanelWitnessPacket] = {}

    def verify_cold_packet(packet: PanelPacket, payload: bytes) -> None:
        if isinstance(packet, ExactPanelWitnessPacket):
            source_digest = _raw_digest(payload)
            previous = exact_packet_cache.get(source_digest)
            if previous is None:
                verify_exact_panel_witness_packet(
                    packet, expected_png_bytes=payload
                )
                exact_packet_cache[source_digest] = packet
            elif previous != packet:
                raise RelationalHeadlessRunError(
                    "same cold PNG digest produced different composite packets"
                )
        else:
            verify_loop_scene_packet(packet, expected_png_bytes=payload)

    final = _verify_seal(final_run, RUN_SCHEMA)
    if (
        not isinstance(release_authenticator, ReleaseArchiveAuthenticator)
        or release_authenticator.release_descriptor_digest
        != plan.release_descriptor_digest
        or release_authenticator.release_descriptor_source_digest
        != plan.release_descriptor_source_digest
        or release_authenticator.corpus_manifest_digest != plan.corpus_digest
        or release_authenticator.split_source_digest != plan.split_source_digest
        or release_authenticator.archive_digest != plan.release_archive_digest
        or release_authenticator.archive_size_bytes
        != plan.release_archive_size_bytes
        or release_authenticator.central_directory_digest
        != plan.release_central_directory_digest
    ):
        raise RelationalHeadlessRunError(
            "cold replay release authenticator differs from plan"
        )
    if final.get("status") != "complete":
        raise RelationalHeadlessRunError("cold replay requires a complete run")
    if (
        final.get("protocol_id") != PROTOCOL_ID
        or final.get("benchmark_mode") != plan.benchmark_mode
    ):
        raise RelationalHeadlessRunError("cold replay benchmark mode differs")
    verified_freeze = verify_relational_proposal_freeze(
        freeze,
        plan=plan,
        exposure_successor=exposure_successor,
    )
    verified_predictions = verify_relational_predictions(
        predictions,
        freeze=verified_freeze,
        plan=plan,
        exposure_successor=exposure_successor,
    )
    if (
        final.get("plan_digest") != plan.digest
        or final.get("exposure_successor_digest") != exposure_successor.digest
        or final.get("freeze_digest") != verified_freeze["digest"]
        or final.get("prediction_digest") != verified_predictions["digest"]
    ):
        raise RelationalHeadlessRunError("final artifact chain differs")
    request = verified_freeze.get("transport_request")
    receipt = verified_freeze.get("codex_receipt")
    if not isinstance(request, Mapping) or not isinstance(receipt, Mapping):
        raise RelationalHeadlessRunError("frozen transport envelope is malformed")
    if (
        request.get("model") != plan.model
        or request.get("reasoning_effort") != plan.reasoning_effort
        or request.get("launcher_digest") != plan.expected_launcher_digest
        or request.get("cloud_policy_cache_binding")
        != plan.cloud_policy_cache_binding
        or verified_freeze.get("proposal_schema_digest")
        != plan.proposal_schema_digest
        or plan.protocol_digest
        != _selection_protocol_digest(
            plan.benchmark_mode, plan._closed_library
        )
    ):
        raise RelationalHeadlessRunError("frozen transport differs from plan")
    query_data = verified_freeze["query"]
    if not isinstance(query_data, Mapping):
        raise RelationalHeadlessRunError("frozen query is not an object")
    query: ExecutablePredicate = (
        ClosedPanelPredicate.from_data(query_data)
        if plan.benchmark_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE
        else RelationalVisualQuery.from_data(query_data)
    )
    entries = verified_freeze["support_entries"]
    if set(support_png_bytes) != {
        item["presentation_name"] for item in entries
    }:
        raise RelationalHeadlessRunError("cold support byte inventory differs")
    staged_panels: list[_SupportPanel] = []
    for entry in entries:
        payload = support_png_bytes[entry["presentation_name"]]
        if (
            _raw_digest(payload) != entry["source_sha256"]
            or len(payload) != entry["byte_count"]
        ):
            raise RelationalHeadlessRunError("cold support bytes differ")
        packet = _packet_from_data(entry["packet"], plan)
        verify_cold_packet(packet, payload)
        replay = _evaluate_predicate(query, packet, plan)
        if replay.to_data() != entry["query_result"]:
            raise RelationalHeadlessRunError("cold support result differs")
        release_receipt = _verify_release_panel_receipt(
            entry.get("release_panel_receipt"),
            plan=plan,
            polarity=entry["polarity"],
            source_index=entry["source_index"],
            source_sha256=entry["source_sha256"],
            byte_count=entry["byte_count"],
        )
        if release_authenticator.authenticate(
            release_receipt["relative_path"], payload
        ) != release_receipt:
            raise RelationalHeadlessRunError(
                "cold support release receipt differs"
            )
        staged_panels.append(
            _SupportPanel(
                entry["polarity"],
                entry["source_index"],
                payload,
                entry["source_sha256"],
                packet,
                release_receipt,
                entry["presentation_name"],
            )
        )
    with tempfile.TemporaryDirectory(
        prefix="bongard-relational-cold-support-"
    ) as directory:
        staged_paths = _stage_support_view(Path(directory), staged_panels)
        panel_view_digest = ordered_panel_view_digest(staged_paths)
        panel_set_digest = semantic_panel_set_digest(staged_paths)
    if (
        panel_view_digest != request.get("panel_view_digest")
        or panel_set_digest != request.get("panel_set_digest")
        or receipt.get("panel_view_digest") != panel_view_digest
        or receipt.get("panel_set_digest") != panel_set_digest
    ):
        raise RelationalHeadlessRunError("cold support presentation digest differs")
    identities = _support_identities(entries)
    input_digest = _structured_input_digest(
        prompt=verified_freeze["support_prompt"],
        identities=identities,
        panel_view_digest=panel_view_digest,
        panel_set_digest=panel_set_digest,
        output_schema_digest=verified_freeze["proposal_schema_digest"],
    )
    if (
        input_digest != request.get("input_digest")
        or receipt.get("input_digest") != input_digest
    ):
        raise RelationalHeadlessRunError("cold structured input digest differs")
    prediction_entries = verified_predictions["entries"]
    if set(query_png_bytes) != {item["query_id"] for item in prediction_entries}:
        raise RelationalHeadlessRunError("cold query byte inventory differs")
    selected_manifest = final.get("selected_panel_manifest")
    if not isinstance(selected_manifest, Mapping) or set(selected_manifest) != {
        "schema",
        "release_authentication",
        "support",
        "query",
    }:
        raise RelationalHeadlessRunError("selected panel manifest is malformed")
    if (
        selected_manifest.get("schema")
        != "gkm.bongard-selected-release-panels.v1"
        or selected_manifest.get("release_authentication")
        != plan.to_data()["release_authentication"]
        or selected_manifest.get("support")
        != [entry["release_panel_receipt"] for entry in entries]
    ):
        raise RelationalHeadlessRunError("selected support manifest differs")
    query_manifest = selected_manifest.get("query")
    if not isinstance(query_manifest, list) or len(query_manifest) != 2:
        raise RelationalHeadlessRunError("selected query manifest differs")
    for slot, entry in enumerate(prediction_entries):
        payload = query_png_bytes[entry["query_id"]]
        if (
            _raw_digest(payload) != entry["source_sha256"]
            or len(payload) != entry["byte_count"]
        ):
            raise RelationalHeadlessRunError("cold query bytes differ")
        packet = _packet_from_data(entry["packet"], plan)
        verify_cold_packet(packet, payload)
        replay = _evaluate_predicate(query, packet, plan)
        if replay.to_data() != entry["query_result"]:
            raise RelationalHeadlessRunError("cold query result differs")
        manifest_entry = query_manifest[slot]
        polarity = plan._query_order[slot]
        if (
            not isinstance(manifest_entry, Mapping)
            or set(manifest_entry) != {"query_id", "release_panel_receipt"}
            or manifest_entry.get("query_id") != entry["query_id"]
        ):
            raise RelationalHeadlessRunError("selected query manifest differs")
        release_receipt = _verify_release_panel_receipt(
            entry.get("release_panel_receipt"),
            plan=plan,
            polarity=polarity,
            source_index=entry["source_index"],
            source_sha256=entry["source_sha256"],
            byte_count=entry["byte_count"],
        )
        if manifest_entry.get("release_panel_receipt") != release_receipt:
            raise RelationalHeadlessRunError(
                "selected query manifest/prediction receipt differs"
            )
        if release_authenticator.authenticate(
            release_receipt["relative_path"], payload
        ) != release_receipt:
            raise RelationalHeadlessRunError(
                "cold query release receipt differs"
            )
    reveal = final.get("schedule_reveal")
    labels = final.get("labels")
    if not isinstance(reveal, Mapping) or not isinstance(labels, list):
        raise RelationalHeadlessRunError("final label reveal is malformed")
    _verify_schedule_reveal(plan, reveal)
    expected_labels = [
        {
            "query_id": f"query-{slot}",
            "positive": side == "positive",
            "source_index": (
                reveal["positive_query_index"]
                if side == "positive"
                else reveal["negative_query_index"]
            ),
        }
        for slot, side in enumerate(reveal["query_order"])
    ]
    if labels != expected_labels:
        raise RelationalHeadlessRunError("final labels differ from schedule reveal")
    prediction_source_indices = {
        item["query_id"]: item["source_index"] for item in prediction_entries
    }
    if any(
        prediction_source_indices[item["query_id"]] != item["source_index"]
        for item in labels
    ):
        raise RelationalHeadlessRunError(
            "prediction source indices differ from revealed schedule labels"
        )
    predictions_by_id = {
        item["query_id"]: item["predicted_positive"]
        for item in prediction_entries
    }
    score = {
        "correct": sum(
            predictions_by_id[item["query_id"]] is item["positive"]
            for item in labels
        ),
        "total": 2,
        "abstentions": sum(
            predictions_by_id[item["query_id"]] is None for item in labels
        ),
        "errors": sum(
            item["query_result"]["disposition"] == Disposition.ERROR.value
            for item in prediction_entries
        ),
    }
    if final.get("score") != score:
        raise RelationalHeadlessRunError("final score differs from cold replay")
    return final_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bongard.relational_headless_runner",
        description=(
            "Run one exposure-precommitted train/val relational Bongard episode"
        ),
    )
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--ledger-in", required=True)
    parser.add_argument("--expected-ledger-digest", required=True)
    parser.add_argument("--expected-corpus-digest", required=True)
    parser.add_argument("--expected-split-digest", required=True)
    parser.add_argument("--expected-release-digest", required=True)
    parser.add_argument("--release-descriptor-file", required=True, type=Path)
    parser.add_argument("--release-archive", required=True, type=Path)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--exposure-observed-at", required=True)
    parser.add_argument("--exposure-store", required=True)
    parser.add_argument("--artifact-store", required=True)
    parser.add_argument("--expected-codex-launcher-sha256", required=True)
    parser.add_argument("--model", default=DEFAULT_CODEX_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--minutes", type=int, default=15)
    parser.add_argument("--executable", default="codex")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    split_index = SplitIndex.load(args.split_file)
    predecessor = ExposureLedger.load(args.ledger_in)
    cache = snapshot_cloud_policy_cache()
    release_authenticator = ReleaseArchiveAuthenticator.load(
        release_descriptor_path=args.release_descriptor_file,
        expected_release_descriptor_digest=args.expected_release_digest,
        archive_path=args.release_archive,
    )
    outcome = run_relational_headless(
        corpus_root=args.corpus_root,
        task_id=args.task_id,
        split_index=split_index,
        predecessor=predecessor,
        expected_corpus_digest=args.expected_corpus_digest,
        expected_split_source_digest=args.expected_split_digest,
        expected_exposure_predecessor_digest=args.expected_ledger_digest,
        seed=args.seed,
        exposure_observed_at=args.exposure_observed_at,
        exposure_store=args.exposure_store,
        artifact_store=args.artifact_store,
        expected_launcher_digest=args.expected_codex_launcher_sha256,
        release_authenticator=release_authenticator,
        cloud_policy_cache_snapshot=cache,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        minutes=args.minutes,
        executable=args.executable,
        verbose=args.verbose,
    )
    print(json.dumps(outcome.to_data(), sort_keys=True))
    return 0 if outcome.status == "complete" else 2


__all__ = [
    "EXACT_UNUSED_TRAIN_ENGINEERING_ADMISSION_POLICY_ID",
    "EXACT_UNUSED_TRAIN_ENGINEERING_MODE",
    "EXACT_UNUSED_TRAIN_ENGINEERING_TASK_IDS",
    "EXPLICITLY_SEALED_ENGINEERING_TASK_ID",
    "FAILURE_SCHEMA",
    "FREEZE_SCHEMA",
    "PLAN_SCHEMA",
    "PREDICTION_SCHEMA",
    "PROTOCOL_ID",
    "RUN_SCHEMA",
    "STRICT_DEV_MODE",
    "FrozenCompleteClosedLibraryIndex",
    "RelationalHeadlessOutcome",
    "RelationalHeadlessPlan",
    "RelationalHeadlessRunError",
    "ReleaseArchiveAuthenticator",
    "cold_replay_relational_headless_run",
    "load_relational_artifact",
    "closed_visual_proposal_schema",
    "neutral_closed_visual_projection",
    "neutral_loop_scene_projection",
    "parse_relational_proposal",
    "parse_closed_visual_proposal",
    "prepare_relational_headless_plan",
    "relational_proposal_schema",
    "relational_headless_runner_source_digest",
    "run_relational_headless",
    "verify_relational_predictions",
    "verify_relational_proposal_freeze",
]


if __name__ == "__main__":
    raise SystemExit(main())

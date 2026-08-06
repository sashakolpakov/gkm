"""No-reroll production command boundary for the atomic Bongard smoke.

The boundary authenticates the complete official release, the exact machine
record for consumed attempt two, and its exposure successor before starting
attempt three.  It freezes every authoritative Python source before release
authentication, requires seven fresh private output stores, and runs two
fixed generic structured transport preflight calls outside the 29-call
Bongard journal.  It then persists the preflight and command commitments and
exclusively creates a permanent claim beside the canonical predecessor before
creating any private seed or exposure.  The claim protects that local path;
copying the ledger to another path is outside this boundary.  This module owns
the exposure, journal, prediction, and terminal durability boundaries and
never prints task identities.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any, Callable, ContextManager, Mapping

from bongard.artifacts import canonical_digest, canonical_json
from bongard.atomic_smoke_precommit import (
    OFFICIAL_A3_LEDGER_DIGEST,
    OFFICIAL_B053_LEDGER_DIGEST,
    OFFICIAL_CORPUS_MANIFEST_DIGEST,
    OFFICIAL_RELEASE_DESCRIPTOR_DIGEST,
    OFFICIAL_SPLIT_SOURCE_DIGEST,
    OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST,
    AtomicSmokePrecommit,
    prepare_atomic_smoke_precommit,
)
from bongard.atomic_smoke_runner import (
    AtomicSmokeJournalReceipt,
    AtomicSmokeRun,
    atomic_smoke_proposal_schema,
    atomic_smoke_run_protocol_digest,
    atomic_smoke_scorer_schema,
    run_atomic_smoke,
    validate_atomic_smoke_proposal_payload,
    validate_atomic_smoke_scorer_payload,
)
from bongard.corpus import ShapeBongardCorpus
from bongard.exposure import ExposureLedger
from bongard.release import (
    DEFAULT_RELEASE_PATH,
    OfficialReleaseDescriptor,
    load_official_release,
)
from bongard.semantic_calibration_command import (
    StageASourceDependencyIdentity,
    StageATrustedCorpus,
    freeze_stage_a_source_dependencies,
    load_stage_a_cache_snapshot,
    persist_stage_a_cache_snapshot,
)
from bongard.transport import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_REASONING_EFFORT,
    CloudPolicyCacheSnapshot,
    CodexStructuredResult,
    StagedCodexLauncher,
    run_codex_named_images_structured,
    run_codex_text_structured,
    snapshot_cloud_policy_cache,
    stage_codex_launcher,
    validate_codex_text_receipt,
)


ATOMIC_SMOKE_COMMAND_AUTHENTICATED_SCHEMA = (
    "gkm.bongard-atomic-smoke-authenticated-inputs.v4"
)
ATOMIC_SMOKE_COMMAND_CONFIG_SCHEMA = "gkm.bongard-atomic-smoke-command-config.v4"
ATOMIC_SMOKE_COMMAND_TERMINAL_SCHEMA = "gkm.bongard-atomic-smoke-command-terminal.v4"
ATOMIC_SMOKE_COMMAND_RECEIPT_SCHEMA = "gkm.bongard-atomic-smoke-command-receipt.v1"
ATOMIC_SMOKE_PREFLIGHT_SCHEMA = "gkm.bongard-atomic-smoke-transport-preflight.v1"
ATOMIC_SMOKE_PREFLIGHT_PROTOCOL_SCHEMA = (
    "gkm.bongard-atomic-smoke-transport-preflight-protocol.v1"
)
ATOMIC_SMOKE_ATTEMPT_CLAIM_SCHEMA = "gkm.bongard-atomic-smoke-attempt-claim.v1"
ATOMIC_SMOKE_ATTEMPT_ORDINAL = 3
ATOMIC_SMOKE_COMMAND_SCOPE = (
    "one-exploratory-repeated-generator-train-successor-smoke/v4"
)
ATOMIC_SMOKE_PREFLIGHT_MINUTES = 5
ATOMIC_SMOKE_CLAIM_PROTECTION_SCOPE = (
    "local-canonical-predecessor-path-only; copied-ledger-paths-"
    "require-external-copy-control"
)
ATOMIC_SMOKE_NATIVE_LAUNCHER_DIGEST = (
    "ae1d3ffe6d48aec6a4dc3f50e7eb8e0d11962485a6a9406c5a7012139383da02"
)
ATOMIC_SMOKE_PRIOR_RECORD_SCHEMA = (
    "gkm.bongard-atomic-smoke-attempt2-proposal-contract-failure.v1"
)
ATOMIC_SMOKE_PRIOR_RECORD_FILE_SHA256 = (
    "242ebc5914020a683a6f34a0b50688bf3190f4c4cbd6d345d15ebb5e775eb6b3"
)
ATOMIC_SMOKE_PRIOR_PREDECESSOR_DIGEST = (
    "sha256:b0533c1a8e94a190f5f382be5031e4318acb6ded2b635ac32172ee238c97de0a"
)
ATOMIC_SMOKE_PRIOR_CONFIG_DIGEST = (
    "sha256:c99557c15548555c63119d3255b4c2421521a9f22f5759aea105c182d24415bc"
)
ATOMIC_SMOKE_PRIOR_CONFIG_FILE_SHA256 = (
    "sha256:7526fe5a3f64e39ae39f7bbc2709e6e86e9374fdf4668713dca8895e4aa3a92b"
)
ATOMIC_SMOKE_PRIOR_PRECOMMIT_DIGEST = (
    "sha256:efd6ab6b84836f0a55ec88f32169b6e91e5c213d194fe8126100a522c28b160a"
)
ATOMIC_SMOKE_PRIOR_RUN_DIGEST = (
    "sha256:1dfd18b2e7a7b4f721e34473282d9eafa28f64bbf7e4e9be945ad101e0123bd9"
)
ATOMIC_SMOKE_PRIOR_RUN_FILE_SHA256 = (
    "sha256:f1309a4573059ad1bf5c95f7bbc908e877bc6cab15efa2ee7fac1c1a1816e8ba"
)
ATOMIC_SMOKE_PRIOR_TERMINAL_DIGEST = (
    "sha256:c6de2ba54c51f25e2635a953b2bceb8a30b3a46f0d29c4a2b820ad5c3060ead8"
)
ATOMIC_SMOKE_PRIOR_TERMINAL_FILE_SHA256 = (
    "sha256:72c75254899785a8e539af9e0a23357c58d09226cb75cf7ba1d798533091d465"
)
ATOMIC_SMOKE_PRIOR_EVIDENCE_DIGEST = (
    "078840ed37dd32620203d9f161ba497b504f6fbfc4acd61c87fd6969a2ab137a"
)
ATOMIC_SMOKE_PRIOR_JOURNAL_HEADER_DIGEST = (
    "da72ce24630b2278c471b3841d1ca12eece5524805fb8af77a078a0cbe158c49"
)
ATOMIC_SMOKE_PRIOR_JOURNAL_RECEIPT_DIGEST = (
    "016de55609959aeb92300fe0dbe094c5888fbf51b86ed824c805899691200c80"
)
ATOMIC_SMOKE_PRIOR_REASON_DIGEST = (
    "34b41a10ae89287ed97c875c6833047ff5896a7081debd144f484833292fe42f"
)
ATOMIC_SMOKE_PRIOR_EXACT_ERROR = (
    "invalid positive_description: soft cue positive_description contains "
    "a forbidden prose character U+003F"
)
ATOMIC_SMOKE_PRIOR_SOURCE_COMMIT = "d0864525146a05795c030674fa0159feb43913c1"
ATOMIC_SMOKE_PRIOR_SOURCE_TAG = "bongard-atomic-successor-pre-smoke-20260806"
ATOMIC_SMOKE_PRIOR_SOURCE_DEPENDENCY_DIGEST = (
    "d0a4e25a8a3d75401bf452b73635372ff12b9a8b4971d9f00be9272eec2850cf"
)
ATOMIC_SMOKE_PRIOR_REMAINING_UNIVERSE_DIGEST = (
    "sha256:3b1a0ce4f9df6e1f9881fb932ec680a988e76afde860c687154401d005c52ee9"
)
DEFAULT_PRIOR_ATTEMPT_RECORD_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "atomic_smoke_attempt2_proposal_contract_failure_v1.json"
)

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")

NamedImageTransport = Callable[..., CodexStructuredResult]
TextTransport = Callable[..., CodexStructuredResult]
SecretFactory = Callable[[int], str]
LauncherStager = Callable[..., ContextManager[StagedCodexLauncher]]


class AtomicSmokeCommandError(RuntimeError):
    """The production command violated authentication or causal durability."""


class AtomicSmokeSourceMutationError(AtomicSmokeCommandError):
    """Authoritative sources changed or became unreadable after the freeze."""

    def __init__(
        self,
        phase: str,
        expected: StageASourceDependencyIdentity,
        observed: StageASourceDependencyIdentity | None,
        observation_error_digest: str | None = None,
    ) -> None:
        self.phase = _text(phase, "source mutation phase")
        self.expected = expected
        self.observed = observed
        self.observation_error_digest = observation_error_digest
        if observed is None:
            _hex(observation_error_digest, "source observation error digest")
            state = "became unreadable"
        else:
            if observed == expected or observation_error_digest is not None:
                raise AtomicSmokeCommandError("source mutation evidence is inconsistent")
            state = f"changed to {observed.digest}"
        super().__init__(f"authoritative sources {state} during {self.phase}")


def _hex(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise AtomicSmokeCommandError(f"{label} must be 64 lowercase hex")
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise AtomicSmokeCommandError(f"{label} must be a sha256: address")
    return value


def _text(value: object, label: str, *, maximum: int = 512) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > maximum
    ):
        raise AtomicSmokeCommandError(f"{label} must be bounded exact text")
    return value


def _exact_int(value: object, label: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise AtomicSmokeCommandError(
            f"{label} must be an exact integer in [{minimum}, {maximum}]"
        )
    return value


def _canonical_clone(value: object, label: str) -> Any:
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise AtomicSmokeCommandError(f"{label} is not canonical JSON") from exc


def _canonical_run_views(
    run: AtomicSmokeRun,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Detach the runner's immutable views through its canonical wire form."""

    if not isinstance(run, AtomicSmokeRun):
        raise TypeError("run must be AtomicSmokeRun")
    run_data = _canonical_clone(run.to_data(), "atomic smoke run")
    if not isinstance(run_data, dict):
        raise AtomicSmokeCommandError("atomic smoke run root is not an object")
    precommit_data = run_data.get("precommit_public_data")
    if not isinstance(precommit_data, dict):
        raise AtomicSmokeCommandError("atomic smoke run precommit is not an object")
    if AtomicSmokeRun.from_data(run_data).digest != run.digest:
        raise AtomicSmokeCommandError("atomic smoke run wire form differs")
    return run_data, precommit_data


def _stable_read(
    path: Path,
    *,
    maximum: int = 128 * 1024 * 1024,
    fsync_file: bool = False,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise AtomicSmokeCommandError("platform lacks no-follow file access")
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AtomicSmokeCommandError(f"cannot open exact file {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise AtomicSmokeCommandError("exact artifact is not singly-linked regular data")
        if before.st_size > maximum:
            raise AtomicSmokeCommandError("exact artifact exceeds its byte bound")
        blocks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not block:
                break
            blocks.append(block)
            total += len(block)
            if total > maximum:
                raise AtomicSmokeCommandError("exact artifact became oversized")
        if fsync_file:
            os.fsync(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or total != after.st_size:
        raise AtomicSmokeCommandError("exact artifact changed while being read")
    return b"".join(blocks)


def _open_store(directory: str | Path) -> tuple[Path, int]:
    requested = Path(directory).expanduser()
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise AtomicSmokeCommandError("artifact store must already exist") from exc
    if requested.absolute() != resolved or not resolved.is_dir():
        raise AtomicSmokeCommandError("artifact store must be one canonical directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise AtomicSmokeCommandError("cannot open artifact store") from exc
    info = os.fstat(descriptor)
    if not stat.S_ISDIR(info.st_mode):
        os.close(descriptor)
        raise AtomicSmokeCommandError("artifact store descriptor is not a directory")
    return resolved, descriptor


@dataclass(frozen=True, slots=True)
class _StoreBinding:
    label: str
    path: Path
    identity: tuple[int, int, int, int]

    @classmethod
    def freeze(cls, label: str, directory: str | Path) -> "_StoreBinding":
        path, descriptor = _open_store(directory)
        try:
            info = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if (
            getattr(info, "st_uid", -1) != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise AtomicSmokeCommandError(
                f"{_text(label, 'store label', maximum=64)} store must be owner-only 0700"
            )
        return cls(
            _text(label, "store label", maximum=64),
            path,
            (
                info.st_dev,
                info.st_ino,
                info.st_mode,
                getattr(info, "st_uid", -1),
            ),
        )

    def check(self, phase: str) -> None:
        path, descriptor = _open_store(self.path)
        try:
            info = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        observed = (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            getattr(info, "st_uid", -1),
        )
        if path != self.path or observed != self.identity:
            raise AtomicSmokeCommandError(
                f"{self.label} store changed during {_text(phase, 'store phase')}"
            )

    def assert_pristine(self, phase: str) -> None:
        """Require an empty store without weakening its frozen identity."""

        self.check(phase)
        _path, descriptor = _open_store(self.path)
        try:
            try:
                entries = os.listdir(descriptor)
            except OSError as exc:
                raise AtomicSmokeCommandError(
                    f"cannot enumerate {self.label} store during {phase}"
                ) from exc
        finally:
            os.close(descriptor)
        if entries:
            raise AtomicSmokeCommandError(
                f"{self.label} store must be fresh and pristine before attempt start"
            )

    def assert_exact_entries(
        self, expected_names: set[str], phase: str
    ) -> None:
        """Reject foreign files appearing after the initial pristine check."""

        self.check(phase)
        if any(
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            for name in expected_names
        ):
            raise AtomicSmokeCommandError("expected store manifest is invalid")
        _path, descriptor = _open_store(self.path)
        try:
            observed = set(os.listdir(descriptor))
        finally:
            os.close(descriptor)
        if observed != expected_names:
            raise AtomicSmokeCommandError(
                f"{self.label} store manifest changed during {phase}"
            )


def _freeze_pristine_stores(
    directories: Mapping[str, str | Path],
) -> dict[str, _StoreBinding]:
    stores = {
        label: _StoreBinding.freeze(label, directory)
        for label, directory in directories.items()
    }
    identities = {(item.identity[0], item.identity[1]) for item in stores.values()}
    if len(identities) != len(stores):
        raise AtomicSmokeCommandError(
            "attempt-owned output stores must be distinct canonical directories"
        )
    for store in stores.values():
        store.assert_pristine("initial-pristine-check")
    return stores


def _write_content_addressed(
    directory: str | Path,
    *,
    digest: str,
    suffix: str,
    payload: bytes,
) -> Path:
    address = _address(digest, "artifact content address")
    store, store_fd = _open_store(directory)
    filename = address.removeprefix("sha256:") + suffix
    destination = store / filename
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        os.close(store_fd)
        raise AtomicSmokeCommandError("platform lacks safe exclusive persistence")
    flags |= os.O_NOFOLLOW
    try:
        try:
            descriptor = os.open(filename, flags, 0o600, dir_fd=store_fd)
        except FileExistsError:
            if destination.is_symlink() or _stable_read(
                destination, fsync_file=True
            ) != payload:
                raise AtomicSmokeCommandError(
                    "content-addressed path contains different or linked bytes"
                )
        else:
            try:
                offset = 0
                while offset < len(payload):
                    written = os.write(descriptor, payload[offset:])
                    if written <= 0:
                        raise AtomicSmokeCommandError("short artifact write")
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.fsync(store_fd)
    finally:
        os.close(store_fd)
    if _stable_read(destination) != payload:
        raise AtomicSmokeCommandError("reloaded artifact differs after fsync")
    return destination


def _prior_attempt_binding_data() -> dict[str, object]:
    return {
        "schema": "gkm.bongard-atomic-smoke-prior-attempt-binding.v2",
        "record_schema": ATOMIC_SMOKE_PRIOR_RECORD_SCHEMA,
        "record_file_sha256": ATOMIC_SMOKE_PRIOR_RECORD_FILE_SHA256,
        "prior_attempt_ordinal": 2,
        "source_snapshot": {
            "commit": ATOMIC_SMOKE_PRIOR_SOURCE_COMMIT,
            "tag": ATOMIC_SMOKE_PRIOR_SOURCE_TAG,
            "source_dependency_digest": (
                ATOMIC_SMOKE_PRIOR_SOURCE_DEPENDENCY_DIGEST
            ),
        },
        "exposure_lineage": {
            "prior_predecessor_content_address": (
                ATOMIC_SMOKE_PRIOR_PREDECESSOR_DIGEST
            ),
            "active_successor_content_address": (
                OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST
            ),
        },
        "artifacts": {
            "command_config_content_address": ATOMIC_SMOKE_PRIOR_CONFIG_DIGEST,
            "command_config_file_sha256": (
                ATOMIC_SMOKE_PRIOR_CONFIG_FILE_SHA256
            ),
            "precommit_content_address": ATOMIC_SMOKE_PRIOR_PRECOMMIT_DIGEST,
            "run_content_address": ATOMIC_SMOKE_PRIOR_RUN_DIGEST,
            "run_file_sha256": ATOMIC_SMOKE_PRIOR_RUN_FILE_SHA256,
            "terminal_content_address": ATOMIC_SMOKE_PRIOR_TERMINAL_DIGEST,
            "terminal_file_sha256": ATOMIC_SMOKE_PRIOR_TERMINAL_FILE_SHA256,
            "evidence_digest": ATOMIC_SMOKE_PRIOR_EVIDENCE_DIGEST,
        },
        "journal": {
            "header_digest": ATOMIC_SMOKE_PRIOR_JOURNAL_HEADER_DIGEST,
            "receipt_digest": ATOMIC_SMOKE_PRIOR_JOURNAL_RECEIPT_DIGEST,
            "intent_count": 13,
            "result_count": 13,
            "state": "result-closed",
            "terminal_persisted": True,
        },
        "call_boundary": {
            "neutral_support_description_calls": 12,
            "atom_proposal_calls": 1,
            "support_scoring_calls": 0,
            "query_calls": 0,
            "formula_frozen": False,
            "selection_archive_persisted": False,
            "prediction_persisted": False,
            "query_labels_materialized": False,
            "query_labels_revealed": False,
            "score": None,
            "run_persisted": True,
            "proposal_receipt_valid": True,
            "proposal_schema_valid": True,
            "proposed_phrase_count": 10,
            "proposed_phrases_ending_in_question_mark": 10,
        },
        "failure": {
            "result_class": "implementation_contract_failure",
            "phase": "atom-proposal",
            "error_type": "AtomicSemanticSynthesisError",
            "exact_error": ATOMIC_SMOKE_PRIOR_EXACT_ERROR,
            "reason_digest": ATOMIC_SMOKE_PRIOR_REASON_DIGEST,
            "cold_replay_passed": True,
        },
        "launcher": {
            "digest": ATOMIC_SMOKE_NATIVE_LAUNCHER_DIGEST,
            "version": "codex-cli 0.146.0",
        },
        "consumption": {
            "selected_task_consumed": True,
            "selected_task_may_be_rerolled": False,
        },
        "remaining_universe": {
            "count": 8,
            "task_ids_digest": ATOMIC_SMOKE_PRIOR_REMAINING_UNIVERSE_DIGEST,
        },
        "claim_authority": {
            "dependence_design_authorized": False,
            "calibration_authorized": False,
            "semantic_claim_authorized": False,
            "benchmark_claim_authorized": False,
            "official_test_authorized": False,
            "score_claim_authorized": False,
            "vision_accuracy_claim_authorized": False,
            "negation_evidence_claim_authorized": False,
            "is_bongard_result": False,
        },
    }


@dataclass(frozen=True, slots=True)
class AtomicSmokePriorAttemptRecord:
    """Authenticated machine record and closed lineage of attempt two."""

    file_sha256: str
    record_schema: str
    source_commit: str
    source_tag: str
    source_dependency_digest: str
    predecessor_digest: str
    successor_digest: str
    command_config_digest: str
    precommit_digest: str
    run_digest: str
    terminal_digest: str
    evidence_digest: str
    journal_header_digest: str
    journal_receipt_digest: str
    failure_reason_digest: str
    remaining_universe_digest: str

    def __post_init__(self) -> None:
        if (
            self.file_sha256 != ATOMIC_SMOKE_PRIOR_RECORD_FILE_SHA256
            or self.record_schema != ATOMIC_SMOKE_PRIOR_RECORD_SCHEMA
            or self.source_commit != ATOMIC_SMOKE_PRIOR_SOURCE_COMMIT
            or self.source_tag != ATOMIC_SMOKE_PRIOR_SOURCE_TAG
            or self.source_dependency_digest
            != ATOMIC_SMOKE_PRIOR_SOURCE_DEPENDENCY_DIGEST
            or self.predecessor_digest != ATOMIC_SMOKE_PRIOR_PREDECESSOR_DIGEST
            or self.successor_digest
            != OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST
            or self.command_config_digest != ATOMIC_SMOKE_PRIOR_CONFIG_DIGEST
            or self.precommit_digest != ATOMIC_SMOKE_PRIOR_PRECOMMIT_DIGEST
            or self.run_digest != ATOMIC_SMOKE_PRIOR_RUN_DIGEST
            or self.terminal_digest != ATOMIC_SMOKE_PRIOR_TERMINAL_DIGEST
            or self.evidence_digest != ATOMIC_SMOKE_PRIOR_EVIDENCE_DIGEST
            or self.journal_header_digest
            != ATOMIC_SMOKE_PRIOR_JOURNAL_HEADER_DIGEST
            or self.journal_receipt_digest
            != ATOMIC_SMOKE_PRIOR_JOURNAL_RECEIPT_DIGEST
            or self.failure_reason_digest != ATOMIC_SMOKE_PRIOR_REASON_DIGEST
            or self.remaining_universe_digest
            != ATOMIC_SMOKE_PRIOR_REMAINING_UNIVERSE_DIGEST
        ):
            raise AtomicSmokeCommandError("prior attempt record lineage differs")

    def to_data(self) -> dict[str, object]:
        return _prior_attempt_binding_data()

    @property
    def digest(self) -> str:
        return "sha256:" + canonical_digest(self.to_data())

    @classmethod
    def _from_data(
        cls, raw: Mapping[str, Any], *, file_sha256: str
    ) -> "AtomicSmokePriorAttemptRecord":
        if set(raw) != {
            "schema", "source_snapshot", "artifacts", "call_boundary",
            "claim_policy", "consuming_attempt", "failure", "forensics",
            "launcher", "remaining_universe",
        } or raw.get("schema") != ATOMIC_SMOKE_PRIOR_RECORD_SCHEMA:
            raise AtomicSmokeCommandError("prior attempt record schema differs")
        names = (
            "source_snapshot", "artifacts", "call_boundary", "claim_policy",
            "consuming_attempt", "failure", "forensics", "launcher",
            "remaining_universe",
        )
        if any(not isinstance(raw.get(name), Mapping) for name in names):
            raise AtomicSmokeCommandError("prior attempt record facts are malformed")
        source = raw["source_snapshot"]
        artifacts = raw["artifacts"]
        calls = raw["call_boundary"]
        claims = raw["claim_policy"]
        attempt = raw["consuming_attempt"]
        failure = raw["failure"]
        forensics = raw["forensics"]
        launcher = raw["launcher"]
        remaining = raw["remaining_universe"]
        expected_artifacts = {
            "command_config_content_address": ATOMIC_SMOKE_PRIOR_CONFIG_DIGEST,
            "command_config_file_sha256": ATOMIC_SMOKE_PRIOR_CONFIG_FILE_SHA256,
            "evidence_digest": ATOMIC_SMOKE_PRIOR_EVIDENCE_DIGEST,
            "exposure_predecessor_content_address": (
                ATOMIC_SMOKE_PRIOR_PREDECESSOR_DIGEST
            ),
            "exposure_successor_content_address": (
                OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST
            ),
            "journal_header_digest": ATOMIC_SMOKE_PRIOR_JOURNAL_HEADER_DIGEST,
            "journal_receipt_digest": ATOMIC_SMOKE_PRIOR_JOURNAL_RECEIPT_DIGEST,
            "precommit_content_address": ATOMIC_SMOKE_PRIOR_PRECOMMIT_DIGEST,
            "run_content_address": ATOMIC_SMOKE_PRIOR_RUN_DIGEST,
            "run_file_sha256": ATOMIC_SMOKE_PRIOR_RUN_FILE_SHA256,
            "terminal_content_address": ATOMIC_SMOKE_PRIOR_TERMINAL_DIGEST,
            "terminal_file_sha256": ATOMIC_SMOKE_PRIOR_TERMINAL_FILE_SHA256,
        }
        expected_calls = {
            "atom_proposal_calls": 1,
            "durable_intent_count": 13,
            "durable_result_count": 13,
            "journal_terminal_persisted": True,
            "formula_frozen": False,
            "neutral_support_description_calls": 12,
            "prediction_persisted": False,
            "proposed_phrase_count": 10,
            "proposed_phrases_ending_in_question_mark": 10,
            "proposal_receipt_valid": True,
            "proposal_schema_valid": True,
            "query_calls": 0,
            "query_labels_materialized": False,
            "query_labels_revealed": False,
            "run_persisted": True,
            "score": None,
            "selection_archive_persisted": False,
            "support_scoring_calls": 0,
        }
        expected_claims = _prior_attempt_binding_data()["claim_authority"]
        expected_attempt = {
            "attempt_ordinal": 2,
            "result_class": "implementation_contract_failure",
            "selected_task_consumed": True,
            "selected_task_may_be_rerolled": False,
        }
        expected_failure = {
            "error_type": "AtomicSemanticSynthesisError",
            "exact_error": ATOMIC_SMOKE_PRIOR_EXACT_ERROR,
            "phase": "atom-proposal",
            "reason_digest": ATOMIC_SMOKE_PRIOR_REASON_DIGEST,
        }
        if (
            canonical_json(artifacts) != canonical_json(expected_artifacts)
            or canonical_json(calls) != canonical_json(expected_calls)
            or canonical_json(claims) != canonical_json(expected_claims)
            or canonical_json(attempt) != canonical_json(expected_attempt)
            or canonical_json(failure) != canonical_json(expected_failure)
            or canonical_json(source)
            != canonical_json(
                {
                    "commit": ATOMIC_SMOKE_PRIOR_SOURCE_COMMIT,
                    "source_dependency_digest": (
                        ATOMIC_SMOKE_PRIOR_SOURCE_DEPENDENCY_DIGEST
                    ),
                    "tag": ATOMIC_SMOKE_PRIOR_SOURCE_TAG,
                }
            )
            or canonical_json(launcher)
            != canonical_json(
                {
                    "digest": ATOMIC_SMOKE_NATIVE_LAUNCHER_DIGEST,
                    "version": "codex-cli 0.146.0",
                }
            )
            or canonical_json(remaining)
            != canonical_json(
                {
                    "count": 8,
                    "task_ids_digest": (
                        ATOMIC_SMOKE_PRIOR_REMAINING_UNIVERSE_DIGEST
                    ),
                }
            )
            or set(forensics)
            != {"cold_replay_passed", "contract_contradiction", "interpretation"}
            or forensics["cold_replay_passed"] is not True
            or not isinstance(forensics["contract_contradiction"], str)
            or not isinstance(forensics["interpretation"], str)
        ):
            raise AtomicSmokeCommandError("prior attempt record causal facts differ")
        return cls(
            file_sha256=file_sha256,
            record_schema=raw["schema"],
            source_commit=source["commit"],
            source_tag=source["tag"],
            source_dependency_digest=source["source_dependency_digest"],
            predecessor_digest=artifacts[
                "exposure_predecessor_content_address"
            ],
            successor_digest=artifacts["exposure_successor_content_address"],
            command_config_digest=artifacts["command_config_content_address"],
            precommit_digest=artifacts["precommit_content_address"],
            run_digest=artifacts["run_content_address"],
            terminal_digest=artifacts["terminal_content_address"],
            evidence_digest=artifacts["evidence_digest"],
            journal_header_digest=artifacts["journal_header_digest"],
            journal_receipt_digest=artifacts["journal_receipt_digest"],
            failure_reason_digest=failure["reason_digest"],
            remaining_universe_digest=remaining["task_ids_digest"],
        )

    @classmethod
    def load(cls, path: str | Path) -> "AtomicSmokePriorAttemptRecord":
        record_path = Path(path).expanduser().absolute()
        payload = _stable_read(record_path, maximum=1_048_576)
        file_sha256 = hashlib.sha256(payload).hexdigest()
        if file_sha256 != ATOMIC_SMOKE_PRIOR_RECORD_FILE_SHA256:
            raise AtomicSmokeCommandError(
                "prior attempt record file differs from exact pin"
            )
        try:
            raw = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise AtomicSmokeCommandError(
                "prior attempt record is not exact JSON"
            ) from exc
        if not isinstance(raw, Mapping):
            raise AtomicSmokeCommandError("prior attempt record root is not an object")
        return cls._from_data(raw, file_sha256=file_sha256)


@dataclass(frozen=True, slots=True)
class AtomicSmokeAuthenticatedInputs:
    """Official corpus, active predecessor, and consumed attempt-two record."""

    trusted: StageATrustedCorpus = field(repr=False)
    release: OfficialReleaseDescriptor
    predecessor: ExposureLedger
    prior_attempt: AtomicSmokePriorAttemptRecord
    predecessor_path: Path
    predecessor_file_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.trusted, StageATrustedCorpus):
            raise TypeError("trusted must be StageATrustedCorpus")
        if not isinstance(self.release, OfficialReleaseDescriptor):
            raise TypeError("release must be OfficialReleaseDescriptor")
        if not isinstance(self.predecessor, ExposureLedger):
            raise TypeError("predecessor must be ExposureLedger")
        if not isinstance(self.prior_attempt, AtomicSmokePriorAttemptRecord):
            raise TypeError(
                "prior_attempt must be AtomicSmokePriorAttemptRecord"
            )
        if (
            not isinstance(self.predecessor_path, Path)
            or not self.predecessor_path.is_absolute()
            or self.predecessor_path.resolve(strict=True) != self.predecessor_path
        ):
            raise AtomicSmokeCommandError(
                "active predecessor path must be an exact canonical regular file"
            )
        _hex(
            self.predecessor_file_sha256,
            "active predecessor raw file digest",
        )
        if hashlib.sha256(_stable_read(self.predecessor_path)).hexdigest() != (
            self.predecessor_file_sha256
        ):
            raise AtomicSmokeCommandError(
                "active predecessor raw file differs from authenticated bytes"
            )
        if self.trusted.authentication_mode != "official-release-archive-and-corpus/v1":
            raise AtomicSmokeCommandError(
                "production smoke requires official-release authentication"
            )
        if self.release.digest != OFFICIAL_RELEASE_DESCRIPTOR_DIGEST:
            raise AtomicSmokeCommandError("release descriptor differs from official pin")
        if (
            self.release.corpus_manifest_sha256 != OFFICIAL_CORPUS_MANIFEST_DIGEST
            or self.trusted.full_manifest.digest != OFFICIAL_CORPUS_MANIFEST_DIGEST
        ):
            raise AtomicSmokeCommandError("full corpus manifest differs from official pin")
        if (
            self.release.split_sha256 != OFFICIAL_SPLIT_SOURCE_DIGEST
            or self.trusted.corpus.split.source_digest != OFFICIAL_SPLIT_SOURCE_DIGEST
        ):
            raise AtomicSmokeCommandError("split source differs from official pin")
        if self.predecessor.digest != OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST:
            raise AtomicSmokeCommandError(
                "active predecessor differs from official successor pin"
            )
        if self.predecessor.corpus_digest != OFFICIAL_CORPUS_MANIFEST_DIGEST:
            raise AtomicSmokeCommandError(
                "active predecessor belongs to another corpus"
            )
        if len(self.predecessor.events) < 2:
            raise AtomicSmokeCommandError(
                "active predecessor lacks its two authenticated ancestors"
            )
        immediate_parent = ExposureLedger(
            corpus_digest=self.predecessor.corpus_digest,
            events=self.predecessor.events[:-1],
        )
        if immediate_parent.digest != OFFICIAL_B053_LEDGER_DIGEST:
            raise AtomicSmokeCommandError(
                "active predecessor does not descend exactly from immediate B053"
            )
        historical_ancestor = ExposureLedger(
            corpus_digest=self.predecessor.corpus_digest,
            events=self.predecessor.events[:-2],
        )
        if historical_ancestor.digest != OFFICIAL_A3_LEDGER_DIGEST:
            raise AtomicSmokeCommandError(
                "active predecessor does not descend exactly from historical A3"
            )
        if not self.predecessor.exposed_task_ids <= set(self.trusted.corpus.task_ids):
            raise AtomicSmokeCommandError(
                "active predecessor contains IDs outside the corpus"
            )
        if (
            self.prior_attempt.successor_digest != self.predecessor.digest
            or self.prior_attempt.predecessor_digest != immediate_parent.digest
        ):
            raise AtomicSmokeCommandError(
                "prior attempt record differs from active exposure lineage"
            )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": ATOMIC_SMOKE_COMMAND_AUTHENTICATED_SCHEMA,
            "scope": ATOMIC_SMOKE_COMMAND_SCOPE,
            "attempt_ordinal": ATOMIC_SMOKE_ATTEMPT_ORDINAL,
            "authentication": self.trusted.to_data(),
            "release_descriptor_digest": self.release.digest,
            "corpus_manifest_digest": self.trusted.full_manifest.digest,
            "split_source_digest": self.trusted.corpus.split.source_digest,
            "active_exposure_predecessor_digest": self.predecessor.digest,
            "active_exposure_predecessor_canonical_path": str(
                self.predecessor_path
            ),
            "active_exposure_predecessor_raw_file_sha256": (
                self.predecessor_file_sha256
            ),
            "immediate_b053_parent_ledger_digest": OFFICIAL_B053_LEDGER_DIGEST,
            "historical_a3_ancestor_ledger_digest": OFFICIAL_A3_LEDGER_DIGEST,
            "prior_attempt_record": self.prior_attempt.to_data(),
            "prior_attempt_record_binding_digest": self.prior_attempt.digest,
        }

    @property
    def digest(self) -> str:
        return "sha256:" + canonical_digest(self.to_data())


def authenticate_atomic_smoke_inputs(
    *,
    corpus_path: str | Path,
    archive_path: str | Path,
    predecessor_ledger_path: str | Path,
    prior_attempt_record_path: str | Path = DEFAULT_PRIOR_ATTEMPT_RECORD_PATH,
    release_descriptor_path: str | Path = DEFAULT_RELEASE_PATH,
) -> AtomicSmokeAuthenticatedInputs:
    """Authenticate the official release, predecessor, and attempt-two record."""

    prior_attempt = AtomicSmokePriorAttemptRecord.load(prior_attempt_record_path)
    release = load_official_release(release_descriptor_path)
    if release.digest != OFFICIAL_RELEASE_DESCRIPTOR_DIGEST:
        raise AtomicSmokeCommandError(
            "checked release descriptor differs from the command pin"
        )
    corpus = ShapeBongardCorpus.discover(
        corpus_path, require_complete=True, require_split=True
    )
    trusted = StageATrustedCorpus.from_official_release(
        corpus=corpus,
        release=release,
        archive_path=archive_path,
    )
    requested_ledger_path = Path(predecessor_ledger_path).expanduser().absolute()
    try:
        ledger_path = requested_ledger_path.resolve(strict=True)
    except OSError as exc:
        raise AtomicSmokeCommandError(
            "active predecessor ledger path is unavailable"
        ) from exc
    try:
        ledger_payload = _stable_read(ledger_path)
        ledger_raw = json.loads(ledger_payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AtomicSmokeCommandError(
            "active predecessor ledger is not exact JSON"
        ) from exc
    if not isinstance(ledger_raw, Mapping):
        raise AtomicSmokeCommandError(
            "active predecessor ledger root is not an object"
        )
    predecessor = ExposureLedger.from_dict(ledger_raw)
    return AtomicSmokeAuthenticatedInputs(
        trusted,
        release,
        predecessor,
        prior_attempt,
        ledger_path,
        hashlib.sha256(ledger_payload).hexdigest(),
    )


_PREFLIGHT_PROPOSAL_PROMPT = (
    "Return exactly this structured value: one atom whose phrase is "
    "Is one closed triangular outline visible?"
)
_PREFLIGHT_PROPOSAL_PAYLOAD: dict[str, object] = {
    "atoms": [{"phrase": "Is one closed triangular outline visible?"}],
}
_PREFLIGHT_ATOM_ID = "0" * 64
_PREFLIGHT_SCORER_PROMPT = (
    "Return exactly one structured result for identifier "
    + _PREFLIGHT_ATOM_ID
    + " with disposition present and explanation Fixed transport "
    "conformance response."
)
_PREFLIGHT_SCORER_PAYLOAD: dict[str, object] = {
    "results": [
        {
            "atom_id": _PREFLIGHT_ATOM_ID,
            "disposition": "present",
            "explanation": "Fixed transport conformance response.",
        }
    ],
}


def atomic_smoke_preflight_protocol_data() -> dict[str, object]:
    """Exact zero-image, zero-secret, non-task transport conformance plan."""

    proposal_schema = atomic_smoke_proposal_schema()
    scorer_schema = atomic_smoke_scorer_schema()
    validate_atomic_smoke_proposal_payload(_PREFLIGHT_PROPOSAL_PAYLOAD)
    validate_atomic_smoke_scorer_payload(
        _PREFLIGHT_SCORER_PAYLOAD,
        expected_atom_ids=(_PREFLIGHT_ATOM_ID,),
    )
    return {
        "schema": ATOMIC_SMOKE_PREFLIGHT_PROTOCOL_SCHEMA,
        "proposal": {
            "prompt": _PREFLIGHT_PROPOSAL_PROMPT,
            "output_schema": proposal_schema,
            "expected_payload": _canonical_clone(
                _PREFLIGHT_PROPOSAL_PAYLOAD, "preflight proposal payload"
            ),
            "production_parser": "validate_atomic_smoke_proposal_payload",
        },
        "scoring": {
            "prompt": _PREFLIGHT_SCORER_PROMPT,
            "output_schema": scorer_schema,
            "expected_payload": _canonical_clone(
                _PREFLIGHT_SCORER_PAYLOAD, "preflight scorer payload"
            ),
            "expected_atom_ids": [_PREFLIGHT_ATOM_ID],
            "production_parser": "validate_atomic_smoke_scorer_payload",
        },
        "transport_call_count": 2,
        "bongard_call_count": 0,
        "secret_count": 0,
        "contains_images": False,
        "contains_bongard_material": False,
        "contains_task_material": False,
        "runner_protocol_digest": atomic_smoke_run_protocol_digest(),
    }


def atomic_smoke_preflight_protocol_digest() -> str:
    return canonical_digest(atomic_smoke_preflight_protocol_data())


def _preflight_receipt_content_data(
    *,
    launcher_path: str,
    launcher_digest: str,
    launcher_version: str,
    model: str,
    reasoning_effort: str,
    proposal_transport_receipt_digest: str,
    scoring_transport_receipt_digest: str,
    protocol_digest: str,
) -> dict[str, object]:
    protocol = atomic_smoke_preflight_protocol_data()
    return {
        "schema": ATOMIC_SMOKE_PREFLIGHT_SCHEMA,
        "scope": ATOMIC_SMOKE_COMMAND_SCOPE,
        "attempt_ordinal": ATOMIC_SMOKE_ATTEMPT_ORDINAL,
        "protocol_digest": protocol_digest,
        "launcher": {
            "canonical_staged_path": launcher_path,
            "raw_sha256": launcher_digest,
            "version": launcher_version,
        },
        "model": model,
        "reasoning_effort": reasoning_effort,
        "proposal_transport_receipt_digest": (
            proposal_transport_receipt_digest
        ),
        "scoring_transport_receipt_digest": scoring_transport_receipt_digest,
        "proposal_payload_digest": canonical_digest(
            _PREFLIGHT_PROPOSAL_PAYLOAD
        ),
        "scoring_payload_digest": canonical_digest(_PREFLIGHT_SCORER_PAYLOAD),
        "transport_call_count": protocol["transport_call_count"],
        "bongard_call_count": protocol["bongard_call_count"],
        "secret_count": protocol["secret_count"],
        "contains_images": protocol["contains_images"],
        "contains_bongard_material": protocol["contains_bongard_material"],
        "contains_task_material": protocol["contains_task_material"],
        "production_parsers_validated": True,
    }


@dataclass(frozen=True, slots=True)
class AtomicSmokeTransportPreflightReceipt:
    """Safe receipt for two fixed calls outside the 29-call Bongard journal."""

    launcher_path: str
    launcher_digest: str
    launcher_version: str
    model: str
    reasoning_effort: str
    proposal_transport_receipt_digest: str
    scoring_transport_receipt_digest: str
    protocol_digest: str
    receipt_digest: str

    def __post_init__(self) -> None:
        launcher_path = Path(_text(
            self.launcher_path, "preflight launcher path", maximum=4096
        ))
        if not launcher_path.is_absolute():
            raise AtomicSmokeCommandError(
                "preflight launcher path must be absolute"
            )
        _hex(self.launcher_digest, "preflight launcher digest")
        if self.launcher_digest != ATOMIC_SMOKE_NATIVE_LAUNCHER_DIGEST:
            raise AtomicSmokeCommandError("preflight launcher differs from pin")
        _text(self.launcher_version, "preflight launcher version", maximum=128)
        _text(self.model, "preflight model", maximum=128)
        _text(self.reasoning_effort, "preflight reasoning effort", maximum=32)
        _hex(
            self.proposal_transport_receipt_digest,
            "proposal transport receipt digest",
        )
        _hex(
            self.scoring_transport_receipt_digest,
            "scoring transport receipt digest",
        )
        _hex(self.protocol_digest, "preflight protocol digest")
        if self.protocol_digest != atomic_smoke_preflight_protocol_digest():
            raise AtomicSmokeCommandError("preflight protocol differs")
        _address(self.receipt_digest, "preflight receipt digest")
        if self.receipt_digest != "sha256:" + canonical_digest(
            self.content_data()
        ):
            raise AtomicSmokeCommandError("preflight receipt digest differs")

    def content_data(self) -> dict[str, object]:
        return _preflight_receipt_content_data(
            launcher_path=self.launcher_path,
            launcher_digest=self.launcher_digest,
            launcher_version=self.launcher_version,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            proposal_transport_receipt_digest=(
                self.proposal_transport_receipt_digest
            ),
            scoring_transport_receipt_digest=(
                self.scoring_transport_receipt_digest
            ),
            protocol_digest=self.protocol_digest,
        )

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "receipt_digest": self.receipt_digest}

    @classmethod
    def create(
        cls,
        *,
        launcher_path: str,
        launcher_digest: str,
        launcher_version: str,
        model: str,
        reasoning_effort: str,
        proposal_transport_receipt_digest: str,
        scoring_transport_receipt_digest: str,
    ) -> "AtomicSmokeTransportPreflightReceipt":
        values = {
            "launcher_path": launcher_path,
            "launcher_digest": launcher_digest,
            "launcher_version": launcher_version,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "proposal_transport_receipt_digest": (
                proposal_transport_receipt_digest
            ),
            "scoring_transport_receipt_digest": (
                scoring_transport_receipt_digest
            ),
            "protocol_digest": atomic_smoke_preflight_protocol_digest(),
        }
        return cls(
            **values,
            receipt_digest="sha256:" + canonical_digest(
                _preflight_receipt_content_data(**values)
            ),
        )

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "AtomicSmokeTransportPreflightReceipt":
        expected = {
            "schema", "scope", "attempt_ordinal", "protocol_digest",
            "launcher", "model", "reasoning_effort",
            "proposal_transport_receipt_digest",
            "scoring_transport_receipt_digest", "proposal_payload_digest",
            "scoring_payload_digest", "transport_call_count",
            "bongard_call_count", "secret_count", "contains_images",
            "contains_bongard_material", "contains_task_material",
            "production_parsers_validated", "receipt_digest",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise AtomicSmokeCommandError("preflight receipt fields differ")
        launcher = value["launcher"]
        if not isinstance(launcher, Mapping) or set(launcher) != {
            "canonical_staged_path", "raw_sha256", "version"
        }:
            raise AtomicSmokeCommandError("preflight launcher fields differ")
        if (
            value["schema"] != ATOMIC_SMOKE_PREFLIGHT_SCHEMA
            or value["scope"] != ATOMIC_SMOKE_COMMAND_SCOPE
            or type(value["attempt_ordinal"]) is not int
            or value["attempt_ordinal"] != ATOMIC_SMOKE_ATTEMPT_ORDINAL
            or value["proposal_payload_digest"]
            != canonical_digest(_PREFLIGHT_PROPOSAL_PAYLOAD)
            or value["scoring_payload_digest"]
            != canonical_digest(_PREFLIGHT_SCORER_PAYLOAD)
            or type(value["transport_call_count"]) is not int
            or value["transport_call_count"] != 2
            or type(value["bongard_call_count"]) is not int
            or value["bongard_call_count"] != 0
            or type(value["secret_count"]) is not int
            or value["secret_count"] != 0
            or value["contains_images"] is not False
            or value["contains_bongard_material"] is not False
            or value["contains_task_material"] is not False
            or value["production_parsers_validated"] is not True
        ):
            raise AtomicSmokeCommandError("preflight receipt authority differs")
        result = cls(
            launcher_path=launcher["canonical_staged_path"],
            launcher_digest=launcher["raw_sha256"],
            launcher_version=launcher["version"],
            model=value["model"],
            reasoning_effort=value["reasoning_effort"],
            proposal_transport_receipt_digest=value[
                "proposal_transport_receipt_digest"
            ],
            scoring_transport_receipt_digest=value[
                "scoring_transport_receipt_digest"
            ],
            protocol_digest=value["protocol_digest"],
            receipt_digest=value["receipt_digest"],
        )
        if result.to_data() != _canonical_clone(value, "preflight receipt"):
            raise AtomicSmokeCommandError("preflight receipt facts differ")
        return result


PreflightRunner = Callable[..., AtomicSmokeTransportPreflightReceipt]


def _execute_atomic_smoke_transport_preflight(
    *,
    staged: StagedCodexLauncher,
    model: str,
    reasoning_effort: str,
    transport: TextTransport,
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot,
) -> AtomicSmokeTransportPreflightReceipt:
    """Execute two fixed zero-image calls and validate production parsers."""

    if not isinstance(staged, StagedCodexLauncher):
        raise TypeError("preflight staged launcher has the wrong type")
    if not callable(transport):
        raise TypeError("preflight transport must be callable")

    def invoke(
        prompt: str,
        schema: Mapping[str, Any],
        expected_payload: Mapping[str, Any],
    ) -> CodexStructuredResult:
        result = transport(
            prompt,
            schema,
            model=model,
            reasoning_effort=reasoning_effort,
            minutes=ATOMIC_SMOKE_PREFLIGHT_MINUTES,
            verbose=False,
            executable=staged.executable,
            cloud_policy_cache_snapshot=cloud_policy_cache_snapshot,
            expected_launcher_digest=staged.launcher_digest,
        )
        if not isinstance(result, CodexStructuredResult):
            raise TypeError("preflight transport returned the wrong type")
        receipt = result.receipt.to_dict()
        validate_codex_text_receipt(receipt, prompt, schema)
        if (
            canonical_json(result.payload) != canonical_json(expected_payload)
            or receipt["structured_output_digest"]
            != canonical_digest(result.payload)
            or receipt["requested_model"] != model
            or receipt["requested_reasoning_effort"] != reasoning_effort
            or receipt["codex_launcher_digest"] != staged.launcher_digest
            or receipt["codex_cli_version"] != staged.version
        ):
            raise AtomicSmokeCommandError(
                "preflight transport result differs from its fixed envelope"
            )
        return result

    proposal = invoke(
        _PREFLIGHT_PROPOSAL_PROMPT,
        atomic_smoke_proposal_schema(),
        _PREFLIGHT_PROPOSAL_PAYLOAD,
    )
    validate_atomic_smoke_proposal_payload(proposal.payload)
    scoring = invoke(
        _PREFLIGHT_SCORER_PROMPT,
        atomic_smoke_scorer_schema(),
        _PREFLIGHT_SCORER_PAYLOAD,
    )
    validate_atomic_smoke_scorer_payload(
        scoring.payload,
        expected_atom_ids=(_PREFLIGHT_ATOM_ID,),
    )
    return AtomicSmokeTransportPreflightReceipt.create(
        launcher_path=staged.executable,
        launcher_digest=staged.launcher_digest,
        launcher_version=staged.version,
        model=model,
        reasoning_effort=reasoning_effort,
        proposal_transport_receipt_digest=proposal.receipt.receipt_digest,
        scoring_transport_receipt_digest=scoring.receipt.receipt_digest,
    )


@dataclass(frozen=True, slots=True)
class AtomicSmokeCommandConfig:
    """Secret-free command commitment persisted before task selection."""

    input_authentication_digest: str
    source_dependencies: StageASourceDependencyIdentity
    cache_binding: str
    cache_file_sha256: str
    cache_byte_count: int
    expected_launcher_digest: str
    staged_launcher_path: str
    launcher_version: str
    preflight_receipt: AtomicSmokeTransportPreflightReceipt
    preflight_receipt_file_sha256: str
    preflight_receipt_filename: str
    preflight_receipt_byte_count: int
    run_protocol_digest: str
    model: str
    reasoning_effort: str
    minutes: int
    verifier_id: str
    verbose: bool = False

    def __post_init__(self) -> None:
        _address(self.input_authentication_digest, "input authentication digest")
        if not isinstance(self.source_dependencies, StageASourceDependencyIdentity):
            raise TypeError("source_dependencies has the wrong type")
        if self.cache_binding != "absent":
            _address(self.cache_binding, "cloud policy cache binding")
        _address(self.cache_file_sha256, "cache snapshot file digest")
        _exact_int(self.cache_byte_count, "cache byte count", minimum=0, maximum=4_194_304)
        _hex(self.expected_launcher_digest, "launcher digest")
        if self.expected_launcher_digest != ATOMIC_SMOKE_NATIVE_LAUNCHER_DIGEST:
            raise AtomicSmokeCommandError("launcher digest differs from production native pin")
        staged_path = Path(_text(
            self.staged_launcher_path, "staged launcher path", maximum=4096
        ))
        if not staged_path.is_absolute():
            raise AtomicSmokeCommandError("staged launcher path must be absolute")
        _text(self.launcher_version, "launcher version", maximum=128)
        if not isinstance(
            self.preflight_receipt, AtomicSmokeTransportPreflightReceipt
        ):
            raise TypeError("preflight_receipt has the wrong type")
        if (
            self.preflight_receipt.launcher_path != self.staged_launcher_path
            or self.preflight_receipt.launcher_digest
            != self.expected_launcher_digest
            or self.preflight_receipt.launcher_version != self.launcher_version
            or self.preflight_receipt.model != self.model
            or self.preflight_receipt.reasoning_effort != self.reasoning_effort
        ):
            raise AtomicSmokeCommandError(
                "preflight receipt differs from command execution identity"
            )
        _address(
            self.preflight_receipt_file_sha256,
            "preflight receipt file digest",
        )
        expected_preflight_filename = (
            self.preflight_receipt.receipt_digest.removeprefix("sha256:")
            + ".atomic-smoke-preflight.json"
        )
        if self.preflight_receipt_filename != expected_preflight_filename:
            raise AtomicSmokeCommandError("preflight receipt filename differs")
        _exact_int(
            self.preflight_receipt_byte_count,
            "preflight receipt byte count",
            minimum=1,
            maximum=1_048_576,
        )
        if self.run_protocol_digest != atomic_smoke_run_protocol_digest():
            raise AtomicSmokeCommandError("run protocol digest differs")
        _text(self.model, "model", maximum=128)
        _text(self.reasoning_effort, "reasoning effort", maximum=32)
        _exact_int(self.minutes, "minutes", minimum=1, maximum=120)
        _text(self.verifier_id, "verifier ID", maximum=256)
        if not isinstance(self.verbose, bool):
            raise AtomicSmokeCommandError("verbose must be Boolean")

    def content_data(self) -> dict[str, object]:
        return {
            "schema": ATOMIC_SMOKE_COMMAND_CONFIG_SCHEMA,
            "scope": ATOMIC_SMOKE_COMMAND_SCOPE,
            "attempt_ordinal": ATOMIC_SMOKE_ATTEMPT_ORDINAL,
            "reference_execution": "python-canonical/lean-optional/v1",
            "official_release_descriptor_digest": (
                OFFICIAL_RELEASE_DESCRIPTOR_DIGEST
            ),
            "official_corpus_manifest_digest": OFFICIAL_CORPUS_MANIFEST_DIGEST,
            "official_split_source_digest": OFFICIAL_SPLIT_SOURCE_DIGEST,
            "official_active_exposure_predecessor_digest": (
                OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST
            ),
            "official_immediate_b053_parent_ledger_digest": (
                OFFICIAL_B053_LEDGER_DIGEST
            ),
            "official_historical_a3_ancestor_ledger_digest": (
                OFFICIAL_A3_LEDGER_DIGEST
            ),
            "prior_attempt_record": _prior_attempt_binding_data(),
            "prior_attempt_record_binding_digest": (
                "sha256:" + canonical_digest(_prior_attempt_binding_data())
            ),
            "input_authentication_digest": self.input_authentication_digest,
            "source_dependencies": self.source_dependencies.to_data(),
            "source_dependency_digest": self.source_dependencies.digest,
            "cloud_policy_cache_binding": self.cache_binding,
            "cloud_policy_cache_snapshot_file_sha256": self.cache_file_sha256,
            "cloud_policy_cache_snapshot_byte_count": self.cache_byte_count,
            "expected_launcher_digest": self.expected_launcher_digest,
            "staged_launcher_path": self.staged_launcher_path,
            "launcher_version": self.launcher_version,
            "transport_preflight": self.preflight_receipt.to_data(),
            "transport_preflight_receipt_digest": (
                self.preflight_receipt.receipt_digest
            ),
            "transport_preflight_file_sha256": (
                self.preflight_receipt_file_sha256
            ),
            "transport_preflight_filename": self.preflight_receipt_filename,
            "transport_preflight_byte_count": (
                self.preflight_receipt_byte_count
            ),
            "run_protocol_digest": self.run_protocol_digest,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "minutes": self.minutes,
            "verifier_id": self.verifier_id,
            "verbose": self.verbose,
            "secrets_generated_after_persistence": True,
            "launcher_authentication_required_before_secret_generation": True,
            "call_journal_required": True,
            "runner_command_config_binding_required": True,
            "secret_values_embedded": False,
            "dependence_design_authorized": False,
            "calibration_authorized": False,
            "benchmark_claim_authorized": False,
            "official_test_authorized": False,
        }

    @property
    def digest(self) -> str:
        return "sha256:" + canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "config_digest": self.digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "AtomicSmokeCommandConfig":
        expected = {
            "schema", "scope", "attempt_ordinal", "reference_execution",
            "official_release_descriptor_digest",
            "official_corpus_manifest_digest", "official_split_source_digest",
            "official_active_exposure_predecessor_digest",
            "official_immediate_b053_parent_ledger_digest",
            "official_historical_a3_ancestor_ledger_digest",
            "prior_attempt_record", "prior_attempt_record_binding_digest",
            "input_authentication_digest",
            "source_dependencies", "source_dependency_digest",
            "cloud_policy_cache_binding",
            "cloud_policy_cache_snapshot_file_sha256",
            "cloud_policy_cache_snapshot_byte_count", "expected_launcher_digest",
            "staged_launcher_path", "launcher_version",
            "transport_preflight", "transport_preflight_receipt_digest",
            "transport_preflight_file_sha256", "transport_preflight_filename",
            "transport_preflight_byte_count",
            "run_protocol_digest", "model", "reasoning_effort", "minutes",
            "verifier_id", "verbose", "secrets_generated_after_persistence",
            "launcher_authentication_required_before_secret_generation",
            "call_journal_required", "runner_command_config_binding_required",
            "secret_values_embedded", "dependence_design_authorized",
            "calibration_authorized", "benchmark_claim_authorized",
            "official_test_authorized", "config_digest",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise AtomicSmokeCommandError("command config fields differ")
        prior_attempt_record = value["prior_attempt_record"]
        expected_prior_attempt_record = _prior_attempt_binding_data()
        if (
            value["schema"] != ATOMIC_SMOKE_COMMAND_CONFIG_SCHEMA
            or value["scope"] != ATOMIC_SMOKE_COMMAND_SCOPE
            or value["attempt_ordinal"] != ATOMIC_SMOKE_ATTEMPT_ORDINAL
            or value["reference_execution"]
            != "python-canonical/lean-optional/v1"
            or value["official_release_descriptor_digest"]
            != OFFICIAL_RELEASE_DESCRIPTOR_DIGEST
            or value["official_corpus_manifest_digest"]
            != OFFICIAL_CORPUS_MANIFEST_DIGEST
            or value["official_split_source_digest"] != OFFICIAL_SPLIT_SOURCE_DIGEST
            or value["official_active_exposure_predecessor_digest"]
            != OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST
            or value["official_immediate_b053_parent_ledger_digest"]
            != OFFICIAL_B053_LEDGER_DIGEST
            or value["official_historical_a3_ancestor_ledger_digest"]
            != OFFICIAL_A3_LEDGER_DIGEST
            or not isinstance(prior_attempt_record, Mapping)
            or canonical_json(prior_attempt_record)
            != canonical_json(expected_prior_attempt_record)
            or value["prior_attempt_record_binding_digest"]
            != "sha256:" + canonical_digest(expected_prior_attempt_record)
            or value["secrets_generated_after_persistence"] is not True
            or value[
                "launcher_authentication_required_before_secret_generation"
            ] is not True
            or value["call_journal_required"] is not True
            or value["runner_command_config_binding_required"] is not True
            or value["secret_values_embedded"] is not False
            or any(
                value[name] is not False
                for name in (
                    "dependence_design_authorized", "calibration_authorized",
                    "benchmark_claim_authorized", "official_test_authorized",
                )
            )
            or not isinstance(value["source_dependencies"], Mapping)
            or not isinstance(value["transport_preflight"], Mapping)
        ):
            raise AtomicSmokeCommandError("command config authority differs")
        sources = StageASourceDependencyIdentity.from_data(
            value["source_dependencies"]
        )
        if value["source_dependency_digest"] != sources.digest:
            raise AtomicSmokeCommandError("command source digest parent differs")
        preflight = AtomicSmokeTransportPreflightReceipt.from_data(
            value["transport_preflight"]
        )
        if value["transport_preflight_receipt_digest"] != preflight.receipt_digest:
            raise AtomicSmokeCommandError("command preflight digest differs")
        result = cls(
            input_authentication_digest=value["input_authentication_digest"],
            source_dependencies=sources,
            cache_binding=value["cloud_policy_cache_binding"],
            cache_file_sha256=value[
                "cloud_policy_cache_snapshot_file_sha256"
            ],
            cache_byte_count=value[
                "cloud_policy_cache_snapshot_byte_count"
            ],
            expected_launcher_digest=value["expected_launcher_digest"],
            staged_launcher_path=value["staged_launcher_path"],
            launcher_version=value["launcher_version"],
            preflight_receipt=preflight,
            preflight_receipt_file_sha256=value[
                "transport_preflight_file_sha256"
            ],
            preflight_receipt_filename=value[
                "transport_preflight_filename"
            ],
            preflight_receipt_byte_count=value[
                "transport_preflight_byte_count"
            ],
            run_protocol_digest=value["run_protocol_digest"],
            model=value["model"],
            reasoning_effort=value["reasoning_effort"],
            minutes=value["minutes"],
            verifier_id=value["verifier_id"],
            verbose=value["verbose"],
        )
        if value["config_digest"] != result.digest or result.to_data() != _canonical_clone(
            value, "command config"
        ):
            raise AtomicSmokeCommandError("command config digest differs")
        return result


@dataclass(frozen=True, slots=True)
class AtomicSmokeDurabilityReceipt:
    kind: str
    path: Path = field(repr=False, compare=False)
    content_address: str
    file_sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        _text(self.kind, "durability receipt kind", maximum=64)
        _address(self.content_address, "durability content address")
        _address(self.file_sha256, "durability file digest")
        _exact_int(self.byte_count, "durability byte count", minimum=1, maximum=512_000_000)
        if not self.path.is_absolute():
            raise AtomicSmokeCommandError("durability receipt path must be absolute")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": ATOMIC_SMOKE_COMMAND_RECEIPT_SCHEMA,
            "kind": self.kind,
            "filename": self.path.name,
            "content_address": self.content_address,
            "file_sha256": self.file_sha256,
            "byte_count": self.byte_count,
            "persistence_protocol": "no-follow-exclusive-or-identical-fsync-file-dir-reload/v1",
        }


def _persist_preflight(
    receipt: AtomicSmokeTransportPreflightReceipt,
    directory: str | Path,
) -> AtomicSmokeDurabilityReceipt:
    if not isinstance(receipt, AtomicSmokeTransportPreflightReceipt):
        raise TypeError("receipt must be AtomicSmokeTransportPreflightReceipt")
    payload = canonical_json(receipt.to_data())
    path = _write_content_addressed(
        directory,
        digest=receipt.receipt_digest,
        suffix=".atomic-smoke-preflight.json",
        payload=payload,
    )
    decoded = json.loads(_stable_read(path, maximum=1_048_576))
    if not isinstance(decoded, Mapping) or (
        AtomicSmokeTransportPreflightReceipt.from_data(decoded) != receipt
    ):
        raise AtomicSmokeCommandError("reloaded preflight receipt differs")
    return AtomicSmokeDurabilityReceipt(
        "transport-preflight",
        path.resolve(),
        receipt.receipt_digest,
        "sha256:" + hashlib.sha256(payload).hexdigest(),
        len(payload),
    )


def _persist_config(
    config: AtomicSmokeCommandConfig, directory: str | Path
) -> AtomicSmokeDurabilityReceipt:
    payload = canonical_json(config.to_data())
    path = _write_content_addressed(
        directory,
        digest=config.digest,
        suffix=".atomic-smoke-command.json",
        payload=payload,
    )
    decoded = json.loads(_stable_read(path))
    if not isinstance(decoded, Mapping) or AtomicSmokeCommandConfig.from_data(
        decoded
    ).to_data() != config.to_data():
        raise AtomicSmokeCommandError("reloaded command config differs")
    return AtomicSmokeDurabilityReceipt(
        "command-config",
        path.resolve(),
        config.digest,
        "sha256:" + hashlib.sha256(payload).hexdigest(),
        len(payload),
    )


@dataclass(frozen=True, slots=True)
class AtomicSmokeAttemptClaim:
    """Permanent, seed-independent claim for one canonical predecessor path.

    This is deliberately a local-path protection.  Copying the same offline
    ledger bytes to another canonical pathname creates another claim namespace
    and is outside this mechanism; deployment must separately forbid copies.
    """

    predecessor_path: Path
    predecessor_digest: str
    predecessor_file_sha256: str
    prior_attempt_digest: str
    prior_attempt_file_sha256: str
    config_digest: str
    source_dependencies: StageASourceDependencyIdentity
    launcher_path: str
    launcher_digest: str
    launcher_version: str
    model: str
    reasoning_effort: str
    preflight_receipt: AtomicSmokeTransportPreflightReceipt
    preflight_file_sha256: str
    claim_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.predecessor_path, Path)
            or not self.predecessor_path.is_absolute()
        ):
            raise AtomicSmokeCommandError("claim predecessor path is not absolute")
        _address(self.predecessor_digest, "claim predecessor digest")
        if self.predecessor_digest != OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST:
            raise AtomicSmokeCommandError("claim predecessor differs from active pin")
        _hex(self.predecessor_file_sha256, "claim predecessor raw file digest")
        _address(self.prior_attempt_digest, "claim prior attempt digest")
        _hex(self.prior_attempt_file_sha256, "claim prior attempt raw file digest")
        if (
            self.prior_attempt_digest
            != "sha256:" + canonical_digest(_prior_attempt_binding_data())
            or self.prior_attempt_file_sha256
            != ATOMIC_SMOKE_PRIOR_RECORD_FILE_SHA256
        ):
            raise AtomicSmokeCommandError("claim prior attempt record differs")
        _address(self.config_digest, "claim config digest")
        if not isinstance(
            self.source_dependencies, StageASourceDependencyIdentity
        ):
            raise TypeError("claim source_dependencies has the wrong type")
        launcher_path = Path(_text(
            self.launcher_path, "claim launcher path", maximum=4096
        ))
        if not launcher_path.is_absolute():
            raise AtomicSmokeCommandError("claim launcher path is not absolute")
        _hex(self.launcher_digest, "claim launcher digest")
        _text(self.launcher_version, "claim launcher version", maximum=128)
        _text(self.model, "claim model", maximum=128)
        _text(self.reasoning_effort, "claim reasoning effort", maximum=32)
        if not isinstance(
            self.preflight_receipt, AtomicSmokeTransportPreflightReceipt
        ):
            raise TypeError("claim preflight receipt has the wrong type")
        _address(self.preflight_file_sha256, "claim preflight file digest")
        if (
            self.preflight_receipt.launcher_path != self.launcher_path
            or self.preflight_receipt.launcher_digest != self.launcher_digest
            or self.preflight_receipt.launcher_version != self.launcher_version
            or self.preflight_receipt.model != self.model
            or self.preflight_receipt.reasoning_effort != self.reasoning_effort
        ):
            raise AtomicSmokeCommandError(
                "claim preflight differs from launcher execution identity"
            )
        _address(self.claim_digest, "attempt claim digest")
        if self.claim_digest != "sha256:" + canonical_digest(
            self.content_data()
        ):
            raise AtomicSmokeCommandError("attempt claim digest differs")

    def content_data(self) -> dict[str, object]:
        return {
            "schema": ATOMIC_SMOKE_ATTEMPT_CLAIM_SCHEMA,
            "scope": ATOMIC_SMOKE_COMMAND_SCOPE,
            "attempt_ordinal": ATOMIC_SMOKE_ATTEMPT_ORDINAL,
            "reference_execution": "python-canonical/lean-optional/v1",
            "canonical_predecessor_path": str(self.predecessor_path),
            "predecessor_content_address": self.predecessor_digest,
            "predecessor_raw_file_sha256": self.predecessor_file_sha256,
            "prior_attempt_record_content_address": self.prior_attempt_digest,
            "prior_attempt_record_raw_file_sha256": (
                self.prior_attempt_file_sha256
            ),
            "command_config_content_address": self.config_digest,
            "source_snapshot": self.source_dependencies.to_data(),
            "source_snapshot_digest": self.source_dependencies.digest,
            "launcher": {
                "canonical_staged_path": self.launcher_path,
                "raw_sha256": self.launcher_digest,
                "version": self.launcher_version,
            },
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "transport_preflight": self.preflight_receipt.to_data(),
            "transport_preflight_receipt_digest": (
                self.preflight_receipt.receipt_digest
            ),
            "transport_preflight_file_sha256": self.preflight_file_sha256,
            "secret_count_at_claim": 0,
            "exposure_created_at_claim": False,
            "persistence_protocol": (
                "canonical-predecessor-path-exclusive-create-fsync-reload/v1"
            ),
            "protection_scope": (
                ATOMIC_SMOKE_CLAIM_PROTECTION_SCOPE
            ),
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "claim_digest": self.claim_digest}

    @classmethod
    def create(
        cls,
        *,
        inputs: AtomicSmokeAuthenticatedInputs,
        config: AtomicSmokeCommandConfig,
        preflight_persistence: AtomicSmokeDurabilityReceipt,
    ) -> "AtomicSmokeAttemptClaim":
        if (
            config.input_authentication_digest != inputs.digest
            or inputs.predecessor.digest
            != OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST
        ):
            raise AtomicSmokeCommandError(
                "claim inputs differ from authenticated command config"
            )
        if (
            preflight_persistence.kind != "transport-preflight"
            or preflight_persistence.content_address
            != config.preflight_receipt.receipt_digest
            or preflight_persistence.file_sha256
            != config.preflight_receipt_file_sha256
            or preflight_persistence.path.name
            != config.preflight_receipt_filename
            or preflight_persistence.byte_count
            != config.preflight_receipt_byte_count
        ):
            raise AtomicSmokeCommandError(
                "claim preflight persistence chain differs"
            )
        values = {
            "predecessor_path": inputs.predecessor_path,
            "predecessor_digest": inputs.predecessor.digest,
            "predecessor_file_sha256": inputs.predecessor_file_sha256,
            "prior_attempt_digest": inputs.prior_attempt.digest,
            "prior_attempt_file_sha256": inputs.prior_attempt.file_sha256,
            "config_digest": config.digest,
            "source_dependencies": config.source_dependencies,
            "launcher_path": config.staged_launcher_path,
            "launcher_digest": config.expected_launcher_digest,
            "launcher_version": config.launcher_version,
            "model": config.model,
            "reasoning_effort": config.reasoning_effort,
            "preflight_receipt": config.preflight_receipt,
            "preflight_file_sha256": preflight_persistence.file_sha256,
        }
        content = {
            "schema": ATOMIC_SMOKE_ATTEMPT_CLAIM_SCHEMA,
            "scope": ATOMIC_SMOKE_COMMAND_SCOPE,
            "attempt_ordinal": ATOMIC_SMOKE_ATTEMPT_ORDINAL,
            "reference_execution": "python-canonical/lean-optional/v1",
            "canonical_predecessor_path": str(values["predecessor_path"]),
            "predecessor_content_address": values["predecessor_digest"],
            "predecessor_raw_file_sha256": values[
                "predecessor_file_sha256"
            ],
            "prior_attempt_record_content_address": values[
                "prior_attempt_digest"
            ],
            "prior_attempt_record_raw_file_sha256": values[
                "prior_attempt_file_sha256"
            ],
            "command_config_content_address": values["config_digest"],
            "source_snapshot": config.source_dependencies.to_data(),
            "source_snapshot_digest": config.source_dependencies.digest,
            "launcher": {
                "canonical_staged_path": values["launcher_path"],
                "raw_sha256": values["launcher_digest"],
                "version": values["launcher_version"],
            },
            "model": values["model"],
            "reasoning_effort": values["reasoning_effort"],
            "transport_preflight": config.preflight_receipt.to_data(),
            "transport_preflight_receipt_digest": (
                config.preflight_receipt.receipt_digest
            ),
            "transport_preflight_file_sha256": (
                preflight_persistence.file_sha256
            ),
            "secret_count_at_claim": 0,
            "exposure_created_at_claim": False,
            "persistence_protocol": (
                "canonical-predecessor-path-exclusive-create-fsync-reload/v1"
            ),
            "protection_scope": (
                ATOMIC_SMOKE_CLAIM_PROTECTION_SCOPE
            ),
        }
        return cls(
            **values,
            claim_digest="sha256:" + canonical_digest(content),
        )

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "AtomicSmokeAttemptClaim":
        expected = {
            "schema", "scope", "attempt_ordinal", "reference_execution",
            "canonical_predecessor_path", "predecessor_content_address",
            "predecessor_raw_file_sha256",
            "prior_attempt_record_content_address",
            "prior_attempt_record_raw_file_sha256",
            "command_config_content_address", "source_snapshot",
            "source_snapshot_digest", "launcher", "model",
            "reasoning_effort", "transport_preflight",
            "transport_preflight_receipt_digest",
            "transport_preflight_file_sha256", "secret_count_at_claim",
            "exposure_created_at_claim", "persistence_protocol",
            "protection_scope", "claim_digest",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise AtomicSmokeCommandError("attempt claim fields differ")
        launcher = value["launcher"]
        if not isinstance(launcher, Mapping) or set(launcher) != {
            "canonical_staged_path", "raw_sha256", "version"
        } or not isinstance(value["source_snapshot"], Mapping) or not isinstance(
            value["transport_preflight"], Mapping
        ):
            raise AtomicSmokeCommandError("attempt claim nested fields differ")
        sources = StageASourceDependencyIdentity.from_data(
            value["source_snapshot"]
        )
        preflight = AtomicSmokeTransportPreflightReceipt.from_data(
            value["transport_preflight"]
        )
        if (
            value["schema"] != ATOMIC_SMOKE_ATTEMPT_CLAIM_SCHEMA
            or value["scope"] != ATOMIC_SMOKE_COMMAND_SCOPE
            or value["attempt_ordinal"] != ATOMIC_SMOKE_ATTEMPT_ORDINAL
            or value["reference_execution"]
            != "python-canonical/lean-optional/v1"
            or value["source_snapshot_digest"] != sources.digest
            or value["transport_preflight_receipt_digest"]
            != preflight.receipt_digest
            or value["secret_count_at_claim"] != 0
            or type(value["secret_count_at_claim"]) is not int
            or value["exposure_created_at_claim"] is not False
            or value["persistence_protocol"]
            != "canonical-predecessor-path-exclusive-create-fsync-reload/v1"
            or value["protection_scope"]
            != ATOMIC_SMOKE_CLAIM_PROTECTION_SCOPE
        ):
            raise AtomicSmokeCommandError("attempt claim authority differs")
        result = cls(
            predecessor_path=Path(value["canonical_predecessor_path"]),
            predecessor_digest=value["predecessor_content_address"],
            predecessor_file_sha256=value["predecessor_raw_file_sha256"],
            prior_attempt_digest=value[
                "prior_attempt_record_content_address"
            ],
            prior_attempt_file_sha256=value[
                "prior_attempt_record_raw_file_sha256"
            ],
            config_digest=value["command_config_content_address"],
            source_dependencies=sources,
            launcher_path=launcher["canonical_staged_path"],
            launcher_digest=launcher["raw_sha256"],
            launcher_version=launcher["version"],
            model=value["model"],
            reasoning_effort=value["reasoning_effort"],
            preflight_receipt=preflight,
            preflight_file_sha256=value["transport_preflight_file_sha256"],
            claim_digest=value["claim_digest"],
        )
        if result.to_data() != _canonical_clone(value, "attempt claim"):
            raise AtomicSmokeCommandError("attempt claim is not canonical")
        return result


def _attempt_claim_path(predecessor_path: Path) -> Path:
    canonical = predecessor_path.resolve(strict=True)
    path_digest = hashlib.sha256(str(canonical).encode("utf-8")).hexdigest()
    return canonical.parent / (
        f".atomic-smoke-attempt-{ATOMIC_SMOKE_ATTEMPT_ORDINAL}-"
        f"{path_digest}.claim.json"
    )


def _persist_attempt_claim(claim: AtomicSmokeAttemptClaim) -> Path:
    """Exclusively consume this canonical predecessor path, never idempotently."""

    if not isinstance(claim, AtomicSmokeAttemptClaim):
        raise TypeError("claim must be AtomicSmokeAttemptClaim")
    if claim.predecessor_path.resolve(strict=True) != claim.predecessor_path:
        raise AtomicSmokeCommandError("claim predecessor path changed")
    if hashlib.sha256(_stable_read(claim.predecessor_path)).hexdigest() != (
        claim.predecessor_file_sha256
    ):
        raise AtomicSmokeCommandError("claim predecessor bytes changed")
    parent, parent_fd = _open_store(claim.predecessor_path.parent)
    created = False
    try:
        parent_info = os.fstat(parent_fd)
        if (
            getattr(parent_info, "st_uid", -1) != os.getuid()
            or stat.S_IMODE(parent_info.st_mode) != 0o700
        ):
            raise AtomicSmokeCommandError(
                "canonical predecessor directory must be current-UID owner-only 0700"
            )
        path = _attempt_claim_path(claim.predecessor_path)
        if path.parent != parent:
            raise AtomicSmokeCommandError("attempt claim parent changed")
        payload = canonical_json(claim.to_data())
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(
            os, "O_CLOEXEC", 0
        )
        if not hasattr(os, "O_NOFOLLOW"):
            raise AtomicSmokeCommandError(
                "platform lacks no-follow exclusive claim persistence"
            )
        flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path.name, flags, 0o600, dir_fd=parent_fd)
        except FileExistsError as exc:
            raise AtomicSmokeCommandError(
                "attempt claim already exists for canonical predecessor path"
            ) from exc
        created = True
        try:
            os.fchmod(descriptor, 0o600)
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise AtomicSmokeCommandError("short attempt claim write")
                offset += written
        finally:
            try:
                # A partial claim is still a permanent consumed-path marker.
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        try:
            if created:
                os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    if _stable_read(path) != payload:
        raise AtomicSmokeCommandError("reloaded attempt claim differs")
    info = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o600
        or getattr(info, "st_uid", -1) != os.getuid()
    ):
        raise AtomicSmokeCommandError("attempt claim metadata is unsafe")
    decoded = json.loads(_stable_read(path))
    if not isinstance(decoded, Mapping) or (
        AtomicSmokeAttemptClaim.from_data(decoded) != claim
    ):
        raise AtomicSmokeCommandError("attempt claim reload is not canonical")
    return path


class _SourceGuard:
    def __init__(
        self, root: Path, expected: StageASourceDependencyIdentity
    ) -> None:
        self.root = root
        self.expected = expected

    def check(self, phase: str) -> None:
        try:
            observed = freeze_stage_a_source_dependencies(self.root)
        except Exception as exc:
            reason = (str(exc) or repr(exc)).encode(
                "utf-8", errors="replace"
            )[:4096]
            raise AtomicSmokeSourceMutationError(
                phase,
                self.expected,
                None,
                hashlib.sha256(reason).hexdigest(),
            ) from exc
        if observed != self.expected:
            raise AtomicSmokeSourceMutationError(
                phase, self.expected, observed
            )

    def wrap(self, phase: str, transport: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(transport):
            raise TypeError("transport must be callable")

        def guarded(*args: Any, **kwargs: Any) -> Any:
            self.check("before-" + phase)
            try:
                return transport(*args, **kwargs)
            finally:
                self.check("after-" + phase)

        return guarded


@dataclass(frozen=True, slots=True)
class AtomicSmokeCommandTerminal:
    """Persistable terminal command record, including operational failures."""

    status: str
    phase: str
    config_digest: str
    source_dependency_digest: str
    source_dependency_state: str
    observed_source_dependency_digest: str | None
    source_observation_error_digest: str | None
    preflight_receipt_data: Mapping[str, Any]
    preflight_receipt_digest: str
    attempt_claim_data: Mapping[str, Any] | None
    attempt_claim_digest: str | None
    precommit_digest: str | None
    precommit_data: Mapping[str, Any] | None
    run_data: Mapping[str, Any] | None
    run_digest: str | None
    journal_receipt_data: Mapping[str, Any] | None
    journal_receipt_digest: str | None
    launcher_digest: str
    launcher_version: str | None
    failure_type: str | None
    failure_reason_digest: str | None
    terminal_digest: str

    def __post_init__(self) -> None:
        if self.status not in {"complete", "failed"}:
            raise AtomicSmokeCommandError("terminal status differs")
        _text(self.phase, "terminal phase", maximum=128)
        _address(self.config_digest, "terminal config digest")
        _hex(self.source_dependency_digest, "terminal source digest")
        if self.source_dependency_state == "unchanged":
            if (
                self.observed_source_dependency_digest
                != self.source_dependency_digest
                or self.source_observation_error_digest is not None
            ):
                raise AtomicSmokeCommandError("unchanged terminal source state differs")
        elif self.source_dependency_state == "mutated":
            _hex(self.observed_source_dependency_digest, "observed source digest")
            if (
                self.observed_source_dependency_digest
                == self.source_dependency_digest
                or self.source_observation_error_digest is not None
            ):
                raise AtomicSmokeCommandError("mutated terminal source state differs")
        elif self.source_dependency_state == "unreadable":
            if self.observed_source_dependency_digest is not None:
                raise AtomicSmokeCommandError("unreadable source state has an identity")
            _hex(self.source_observation_error_digest, "source observation error")
        else:
            raise AtomicSmokeCommandError("unknown terminal source state")
        if not isinstance(self.preflight_receipt_data, Mapping):
            raise AtomicSmokeCommandError("terminal preflight receipt is not an object")
        preflight = AtomicSmokeTransportPreflightReceipt.from_data(
            self.preflight_receipt_data
        )
        if self.preflight_receipt_digest != preflight.receipt_digest:
            raise AtomicSmokeCommandError("terminal preflight receipt differs")
        claim: AtomicSmokeAttemptClaim | None = None
        if self.attempt_claim_data is None:
            if self.attempt_claim_digest is not None:
                raise AtomicSmokeCommandError(
                    "terminal claim digest lacks claim data"
                )
        else:
            if not isinstance(self.attempt_claim_data, Mapping):
                raise AtomicSmokeCommandError("terminal claim is not an object")
            claim = AtomicSmokeAttemptClaim.from_data(self.attempt_claim_data)
            if (
                self.attempt_claim_digest != claim.claim_digest
                or claim.config_digest != self.config_digest
                or claim.source_dependencies.digest
                != self.source_dependency_digest
                or claim.preflight_receipt.receipt_digest
                != self.preflight_receipt_digest
            ):
                raise AtomicSmokeCommandError("terminal attempt claim differs")
        _hex(self.launcher_digest, "terminal launcher digest")
        if self.launcher_digest != ATOMIC_SMOKE_NATIVE_LAUNCHER_DIGEST:
            raise AtomicSmokeCommandError("terminal launcher differs from production pin")
        if self.launcher_version is not None:
            _text(self.launcher_version, "launcher version", maximum=128)
        if (
            preflight.launcher_digest != self.launcher_digest
            or preflight.launcher_version != self.launcher_version
        ):
            raise AtomicSmokeCommandError("terminal launcher differs from preflight")
        if self.precommit_digest is not None:
            _address(self.precommit_digest, "terminal precommit digest")
        if self.precommit_data is not None:
            public_precommit = AtomicSmokePrecommit.from_data(self.precommit_data)
            if self.precommit_digest != public_precommit.digest:
                raise AtomicSmokeCommandError(
                    "terminal precommit data differs from its digest"
                )
        if self.run_data is None:
            if (
                self.run_digest is not None
                or self.journal_receipt_data is not None
                or self.journal_receipt_digest is not None
            ):
                raise AtomicSmokeCommandError(
                    "terminal run or journal receipt lacks run data"
                )
        else:
            run = AtomicSmokeRun.from_data(self.run_data)
            if not isinstance(run.journal_receipt, AtomicSmokeJournalReceipt):
                raise AtomicSmokeCommandError("terminal run lacks typed journal receipt")
            if (
                self.run_digest != run.digest
                or run.command_config_digest != self.config_digest
                or self.precommit_digest != run.precommit_digest
                or (
                    self.precommit_data is not None
                    and canonical_json(self.precommit_data)
                    != canonical_json(
                        run.to_data()["precommit_public_data"]
                    )
                )
                or self.launcher_digest != run.expected_launcher_digest
                or self.source_dependency_digest != run.source_dependency_digest
                or self.journal_receipt_digest
                != run.journal_receipt.receipt_digest
                or self.journal_receipt_data is None
                or canonical_json(self.journal_receipt_data)
                != canonical_json(run.journal_receipt.to_data())
                or (
                    self.status == "complete" and run.status != "complete"
                )
                or claim is None
            ):
                raise AtomicSmokeCommandError("terminal wrapper differs from run")
        if self.status == "complete":
            if self.run_data is None or self.failure_type is not None or self.failure_reason_digest is not None:
                raise AtomicSmokeCommandError("complete terminal has failure state")
        elif self.run_data is None or self.failure_type is not None:
            _text(self.failure_type, "failure type", maximum=256)
            _hex(self.failure_reason_digest, "failure reason digest")
        elif self.failure_reason_digest is not None:
            raise AtomicSmokeCommandError("failed run has an untyped outer failure")
        elif AtomicSmokeRun.from_data(self.run_data).status != "failed":
            raise AtomicSmokeCommandError(
                "failed wrapper around a complete run lacks an outer failure"
            )
        if self.terminal_digest != "sha256:" + canonical_digest(self.content_data()):
            raise AtomicSmokeCommandError("terminal digest differs")

    def content_data(self) -> dict[str, object]:
        return {
            "schema": ATOMIC_SMOKE_COMMAND_TERMINAL_SCHEMA,
            "scope": ATOMIC_SMOKE_COMMAND_SCOPE,
            "attempt_ordinal": ATOMIC_SMOKE_ATTEMPT_ORDINAL,
            "status": self.status,
            "phase": self.phase,
            "config_digest": self.config_digest,
            "source_dependency_digest": self.source_dependency_digest,
            "source_dependency_state": self.source_dependency_state,
            "observed_source_dependency_digest": (
                self.observed_source_dependency_digest
            ),
            "source_observation_error_digest": self.source_observation_error_digest,
            "transport_preflight": dict(self.preflight_receipt_data),
            "transport_preflight_receipt_digest": (
                self.preflight_receipt_digest
            ),
            "attempt_claim": (
                None
                if self.attempt_claim_data is None
                else dict(self.attempt_claim_data)
            ),
            "attempt_claim_digest": self.attempt_claim_digest,
            "precommit_digest": self.precommit_digest,
            "precommit": (
                None if self.precommit_data is None else dict(self.precommit_data)
            ),
            "run": None if self.run_data is None else dict(self.run_data),
            "run_digest": self.run_digest,
            "journal_receipt": (
                None
                if self.journal_receipt_data is None
                else dict(self.journal_receipt_data)
            ),
            "journal_receipt_digest": self.journal_receipt_digest,
            "launcher_digest": self.launcher_digest,
            "launcher_version": self.launcher_version,
            "failure": (
                None
                if self.failure_type is None
                else {
                    "error_type": self.failure_type,
                    "reason_digest": self.failure_reason_digest,
                }
            ),
            "dependence_design_authorized": False,
            "calibration_authorized": False,
            "benchmark_claim_authorized": False,
            "official_test_authorized": False,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "terminal_digest": self.terminal_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "AtomicSmokeCommandTerminal":
        expected = {
            "schema", "scope", "attempt_ordinal", "status", "phase",
            "config_digest",
            "source_dependency_digest", "source_dependency_state",
            "observed_source_dependency_digest", "source_observation_error_digest",
            "transport_preflight", "transport_preflight_receipt_digest",
            "attempt_claim", "attempt_claim_digest",
            "precommit_digest", "precommit",
            "run", "run_digest", "journal_receipt", "journal_receipt_digest",
            "launcher_digest", "launcher_version", "failure",
            "dependence_design_authorized", "calibration_authorized",
            "benchmark_claim_authorized", "official_test_authorized",
            "terminal_digest",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise AtomicSmokeCommandError("terminal fields differ")
        if (
            value["schema"] != ATOMIC_SMOKE_COMMAND_TERMINAL_SCHEMA
            or value["scope"] != ATOMIC_SMOKE_COMMAND_SCOPE
            or value["attempt_ordinal"] != ATOMIC_SMOKE_ATTEMPT_ORDINAL
            or any(
            value[name] is not False
            for name in (
                "dependence_design_authorized", "calibration_authorized",
                "benchmark_claim_authorized", "official_test_authorized",
            )
            )
        ):
            raise AtomicSmokeCommandError("terminal authority differs")
        run = value["run"]
        journal_receipt = value["journal_receipt"]
        precommit = value["precommit"]
        preflight = value["transport_preflight"]
        claim = value["attempt_claim"]
        failure = value["failure"]
        if not isinstance(preflight, Mapping):
            raise AtomicSmokeCommandError("terminal preflight must be an object")
        if claim is not None and not isinstance(claim, Mapping):
            raise AtomicSmokeCommandError("terminal claim must be object or null")
        if precommit is not None and not isinstance(precommit, Mapping):
            raise AtomicSmokeCommandError("terminal precommit must be object or null")
        if run is not None and not isinstance(run, Mapping):
            raise AtomicSmokeCommandError("terminal run must be object or null")
        if journal_receipt is not None and not isinstance(
            journal_receipt, Mapping
        ):
            raise AtomicSmokeCommandError(
                "terminal journal receipt must be object or null"
            )
        if failure is not None and (
            not isinstance(failure, Mapping)
            or set(failure) != {"error_type", "reason_digest"}
        ):
            raise AtomicSmokeCommandError("terminal failure fields differ")
        result = cls(
            status=value["status"],
            phase=value["phase"],
            config_digest=value["config_digest"],
            source_dependency_digest=value["source_dependency_digest"],
            source_dependency_state=value["source_dependency_state"],
            observed_source_dependency_digest=value[
                "observed_source_dependency_digest"
            ],
            source_observation_error_digest=value[
                "source_observation_error_digest"
            ],
            preflight_receipt_data=_canonical_clone(
                preflight, "terminal preflight"
            ),
            preflight_receipt_digest=value[
                "transport_preflight_receipt_digest"
            ],
            attempt_claim_data=(
                None
                if claim is None
                else _canonical_clone(claim, "terminal attempt claim")
            ),
            attempt_claim_digest=value["attempt_claim_digest"],
            precommit_digest=value["precommit_digest"],
            precommit_data=(
                None
                if precommit is None
                else _canonical_clone(precommit, "terminal precommit")
            ),
            run_data=None if run is None else _canonical_clone(run, "terminal run"),
            run_digest=value["run_digest"],
            journal_receipt_data=(
                None
                if journal_receipt is None
                else _canonical_clone(journal_receipt, "terminal journal receipt")
            ),
            journal_receipt_digest=value["journal_receipt_digest"],
            launcher_digest=value["launcher_digest"],
            launcher_version=value["launcher_version"],
            failure_type=None if failure is None else failure["error_type"],
            failure_reason_digest=None if failure is None else failure["reason_digest"],
            terminal_digest=value["terminal_digest"],
        )
        if result.to_data() != _canonical_clone(value, "terminal"):
            raise AtomicSmokeCommandError("terminal is not canonical")
        return result

    @classmethod
    def from_run(
        cls,
        run: AtomicSmokeRun,
        *,
        config: AtomicSmokeCommandConfig,
        attempt_claim: AtomicSmokeAttemptClaim,
        launcher_version: str,
    ) -> "AtomicSmokeCommandTerminal":
        run_data, precommit_data = _canonical_run_views(run)
        if not isinstance(config, AtomicSmokeCommandConfig):
            raise TypeError("config must be AtomicSmokeCommandConfig")
        if not isinstance(attempt_claim, AtomicSmokeAttemptClaim):
            raise TypeError("attempt_claim must be AtomicSmokeAttemptClaim")
        if (
            run.command_config_digest != config.digest
            or attempt_claim.config_digest != config.digest
        ):
            raise AtomicSmokeCommandError("run differs from exact command config")
        journal_receipt_data = _canonical_clone(
            run.journal_receipt.to_data(), "run journal receipt"
        )
        values = {
            "status": "complete" if run.status == "complete" else "failed",
            "phase": run.terminal_phase,
            "config_digest": config.digest,
            "source_dependency_digest": run.source_dependency_digest,
            "source_dependency_state": "unchanged",
            "observed_source_dependency_digest": run.source_dependency_digest,
            "source_observation_error_digest": None,
            "preflight_receipt_data": config.preflight_receipt.to_data(),
            "preflight_receipt_digest": (
                config.preflight_receipt.receipt_digest
            ),
            "attempt_claim_data": attempt_claim.to_data(),
            "attempt_claim_digest": attempt_claim.claim_digest,
            "precommit_digest": run.precommit_digest,
            "precommit_data": precommit_data,
            "run_data": run_data,
            "run_digest": run.digest,
            "journal_receipt_data": journal_receipt_data,
            "journal_receipt_digest": run.journal_receipt.receipt_digest,
            "launcher_digest": run.expected_launcher_digest,
            "launcher_version": launcher_version,
            "failure_type": None,
            "failure_reason_digest": None,
        }
        content = {
            "schema": ATOMIC_SMOKE_COMMAND_TERMINAL_SCHEMA,
            "scope": ATOMIC_SMOKE_COMMAND_SCOPE,
            "attempt_ordinal": ATOMIC_SMOKE_ATTEMPT_ORDINAL,
            "status": values["status"],
            "phase": values["phase"],
            "config_digest": values["config_digest"],
            "source_dependency_digest": values["source_dependency_digest"],
            "source_dependency_state": values["source_dependency_state"],
            "observed_source_dependency_digest": values[
                "observed_source_dependency_digest"
            ],
            "source_observation_error_digest": values[
                "source_observation_error_digest"
            ],
            "transport_preflight": values["preflight_receipt_data"],
            "transport_preflight_receipt_digest": values[
                "preflight_receipt_digest"
            ],
            "attempt_claim": values["attempt_claim_data"],
            "attempt_claim_digest": values["attempt_claim_digest"],
            "precommit_digest": values["precommit_digest"],
            "precommit": values["precommit_data"],
            "run": values["run_data"],
            "run_digest": values["run_digest"],
            "journal_receipt": values["journal_receipt_data"],
            "journal_receipt_digest": values["journal_receipt_digest"],
            "launcher_digest": values["launcher_digest"],
            "launcher_version": values["launcher_version"],
            "failure": None,
            "dependence_design_authorized": False,
            "calibration_authorized": False,
            "benchmark_claim_authorized": False,
            "official_test_authorized": False,
        }
        digest = "sha256:" + canonical_digest(content)
        return cls(**values, terminal_digest=digest)

    @classmethod
    def failure(
        cls,
        error: Exception,
        *,
        phase: str,
        config: AtomicSmokeCommandConfig,
        precommit: AtomicSmokePrecommit | None,
        launcher_version: str | None,
        attempt_claim: AtomicSmokeAttemptClaim | None = None,
        run: AtomicSmokeRun | None = None,
    ) -> "AtomicSmokeCommandTerminal":
        reason = (str(error) or repr(error)).encode("utf-8", errors="replace")[:4096]
        mutation = error if isinstance(error, AtomicSmokeSourceMutationError) else None
        run_data: dict[str, Any] | None = None
        run_precommit_data: dict[str, Any] | None = None
        if run is not None:
            run_data, run_precommit_data = _canonical_run_views(run)
            if run.command_config_digest != config.digest:
                raise AtomicSmokeCommandError("failed run differs from command config")
        effective_precommit_digest = (
            run.precommit_digest
            if run is not None
            else None if precommit is None else precommit.digest
        )
        effective_precommit_data = (
            run_precommit_data
            if run is not None
            else precommit.to_data()
            if isinstance(precommit, AtomicSmokePrecommit)
            else None
        )
        values = {
            "status": "failed",
            "phase": phase,
            "config_digest": config.digest,
            "source_dependency_digest": config.source_dependencies.digest,
            "source_dependency_state": (
                "unchanged"
                if mutation is None
                else "mutated" if mutation.observed is not None else "unreadable"
            ),
            "observed_source_dependency_digest": (
                config.source_dependencies.digest
                if mutation is None
                else None if mutation.observed is None else mutation.observed.digest
            ),
            "source_observation_error_digest": (
                None if mutation is None else mutation.observation_error_digest
            ),
            "preflight_receipt_data": config.preflight_receipt.to_data(),
            "preflight_receipt_digest": (
                config.preflight_receipt.receipt_digest
            ),
            "attempt_claim_data": (
                None if attempt_claim is None else attempt_claim.to_data()
            ),
            "attempt_claim_digest": (
                None if attempt_claim is None else attempt_claim.claim_digest
            ),
            "precommit_digest": effective_precommit_digest,
            "precommit_data": effective_precommit_data,
            "run_data": run_data,
            "run_digest": None if run is None else run.digest,
            "journal_receipt_data": (
                None if run is None else run.journal_receipt.to_data()
            ),
            "journal_receipt_digest": (
                None if run is None else run.journal_receipt.receipt_digest
            ),
            "launcher_digest": config.expected_launcher_digest,
            "launcher_version": launcher_version,
            "failure_type": type(error).__name__,
            "failure_reason_digest": hashlib.sha256(reason).hexdigest(),
        }
        content = {
            "schema": ATOMIC_SMOKE_COMMAND_TERMINAL_SCHEMA,
            "scope": ATOMIC_SMOKE_COMMAND_SCOPE,
            "attempt_ordinal": ATOMIC_SMOKE_ATTEMPT_ORDINAL,
            "status": values["status"], "phase": values["phase"],
            "config_digest": values["config_digest"],
            "source_dependency_digest": values["source_dependency_digest"],
            "source_dependency_state": values["source_dependency_state"],
            "observed_source_dependency_digest": values[
                "observed_source_dependency_digest"
            ],
            "source_observation_error_digest": values[
                "source_observation_error_digest"
            ],
            "transport_preflight": values["preflight_receipt_data"],
            "transport_preflight_receipt_digest": values[
                "preflight_receipt_digest"
            ],
            "attempt_claim": values["attempt_claim_data"],
            "attempt_claim_digest": values["attempt_claim_digest"],
            "precommit_digest": values["precommit_digest"],
            "precommit": values["precommit_data"],
            "run": values["run_data"],
            "run_digest": values["run_digest"],
            "journal_receipt": values["journal_receipt_data"],
            "journal_receipt_digest": values["journal_receipt_digest"],
            "launcher_digest": values["launcher_digest"],
            "launcher_version": values["launcher_version"],
            "failure": {"error_type": values["failure_type"], "reason_digest": values["failure_reason_digest"]},
            "dependence_design_authorized": False, "calibration_authorized": False,
            "benchmark_claim_authorized": False, "official_test_authorized": False,
        }
        return cls(
            **values,
            terminal_digest="sha256:" + canonical_digest(content),
        )


def _persist_terminal(
    terminal: AtomicSmokeCommandTerminal,
    directory: str | Path,
    *,
    source_guard: _SourceGuard | None = None,
) -> AtomicSmokeDurabilityReceipt:
    if source_guard is not None:
        source_guard.check("before-terminal-serialization")
    payload = canonical_json(terminal.to_data())
    if source_guard is not None:
        source_guard.check("after-terminal-serialization")
    path = _write_content_addressed(
        directory,
        digest=terminal.terminal_digest,
        suffix=".atomic-smoke-terminal.json",
        payload=payload,
    )
    raw = json.loads(_stable_read(path, maximum=512_000_000))
    if not isinstance(raw, Mapping):
        raise AtomicSmokeCommandError("terminal root is not an object")
    if AtomicSmokeCommandTerminal.from_data(raw).to_data() != terminal.to_data():
        raise AtomicSmokeCommandError("reloaded terminal differs")
    if source_guard is not None:
        source_guard.check("after-terminal-persistence-reload")
    return AtomicSmokeDurabilityReceipt(
        "terminal-outcome",
        path.resolve(),
        terminal.terminal_digest,
        "sha256:" + hashlib.sha256(payload).hexdigest(),
        len(payload),
    )


def _persist_raw_run(
    run: AtomicSmokeRun,
    directory: str | Path,
) -> AtomicSmokeDurabilityReceipt:
    """Durably preserve the runner result before any wrapper work can fail."""

    run_data, _precommit_data = _canonical_run_views(run)
    payload = canonical_json(run_data)
    content_address = "sha256:" + _hex(run.digest, "raw run digest")
    path = _write_content_addressed(
        directory,
        digest=content_address,
        suffix=".atomic-smoke-run.json",
        payload=payload,
    )
    decoded = json.loads(_stable_read(path, maximum=512_000_000))
    if not isinstance(decoded, Mapping):
        raise AtomicSmokeCommandError("raw run root is not an object")
    reloaded = AtomicSmokeRun.from_data(decoded)
    if reloaded.digest != run.digest or reloaded.to_data() != run_data:
        raise AtomicSmokeCommandError("reloaded raw run differs")
    return AtomicSmokeDurabilityReceipt(
        "atomic-smoke-run",
        path.resolve(),
        content_address,
        "sha256:" + hashlib.sha256(payload).hexdigest(),
        len(payload),
    )


@dataclass(frozen=True, slots=True)
class AtomicSmokeCommandResult:
    config: AtomicSmokeCommandConfig
    config_receipt: AtomicSmokeDurabilityReceipt
    preflight: AtomicSmokeTransportPreflightReceipt
    preflight_receipt: AtomicSmokeDurabilityReceipt
    attempt_claim: AtomicSmokeAttemptClaim | None
    attempt_claim_path: Path | None
    precommit: AtomicSmokePrecommit | None = field(repr=False)
    run_receipt: AtomicSmokeDurabilityReceipt | None
    terminal: AtomicSmokeCommandTerminal
    terminal_receipt: AtomicSmokeDurabilityReceipt

    def __post_init__(self) -> None:
        if (
            self.config_receipt.kind != "command-config"
            or self.config_receipt.content_address != self.config.digest
            or self.preflight != self.config.preflight_receipt
            or self.preflight_receipt.kind != "transport-preflight"
            or self.preflight_receipt.content_address
            != self.preflight.receipt_digest
            or self.terminal.config_digest != self.config.digest
            or self.terminal_receipt.kind != "terminal-outcome"
            or self.terminal_receipt.content_address
            != self.terminal.terminal_digest
        ):
            raise AtomicSmokeCommandError("command result durability chain differs")
        if self.attempt_claim is None:
            if self.attempt_claim_path is not None:
                raise AtomicSmokeCommandError("claim path lacks typed claim")
        elif (
            self.attempt_claim_path is None
            or not self.attempt_claim_path.is_absolute()
            or self.terminal.attempt_claim_digest
            != self.attempt_claim.claim_digest
        ):
            raise AtomicSmokeCommandError("command result claim chain differs")
        if self.run_receipt is None:
            if self.terminal.run_digest is not None:
                raise AtomicSmokeCommandError("terminal run lacks its raw durability receipt")
        elif (
            self.terminal.run_digest is None
            or self.run_receipt.kind != "atomic-smoke-run"
            or self.run_receipt.content_address
            != "sha256:" + self.terminal.run_digest
        ):
            raise AtomicSmokeCommandError("raw run durability chain differs")


def _fresh_secrets(factory: SecretFactory) -> tuple[str, str, str]:
    if not callable(factory):
        raise TypeError("secret_factory must be callable")
    values = tuple(_hex(factory(32), "fresh private secret") for _ in range(3))
    if len(set(values)) != 3:
        raise AtomicSmokeCommandError("selection, episode, and label secrets must differ")
    return values  # type: ignore[return-value]


def _assert_non_test_precommit(precommit: AtomicSmokePrecommit) -> None:
    if precommit.episode_plan.split == "test":
        raise AtomicSmokeCommandError("official test is forbidden")


def run_atomic_smoke_command(
    *,
    corpus_path: str | Path,
    archive_path: str | Path,
    predecessor_ledger_path: str | Path,
    config_store_dir: str | Path,
    exposure_store_dir: str | Path,
    journal_store_dir: str | Path,
    prediction_store_dir: str | Path,
    terminal_store_dir: str | Path,
    cache_store_dir: str | Path,
    preflight_store_dir: str | Path,
    prior_attempt_record_path: str | Path = DEFAULT_PRIOR_ATTEMPT_RECORD_PATH,
    release_descriptor_path: str | Path = DEFAULT_RELEASE_PATH,
    expected_launcher_digest: str = ATOMIC_SMOKE_NATIVE_LAUNCHER_DIGEST,
    model: str = DEFAULT_CODEX_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    minutes: int = 15,
    verifier_id: str = "canonical-bongard-verifier",
    executable: str = "codex",
    verbose: bool = False,
    secret_factory: SecretFactory = secrets.token_hex,
    cache_snapshotter: Callable[[], CloudPolicyCacheSnapshot] = snapshot_cloud_policy_cache,
    launcher_stager: LauncherStager = stage_codex_launcher,
    preflight_runner: PreflightRunner = _execute_atomic_smoke_transport_preflight,
    named_image_transport: NamedImageTransport = run_codex_named_images_structured,
    text_transport: TextTransport = run_codex_text_structured,
) -> AtomicSmokeCommandResult:
    """Run attempt three with a permanent claim on one canonical ledger path.

    The claim is local-path protection, not a distributed ledger lock.  A copy
    of the same offline predecessor bytes at another pathname is deliberately
    outside this boundary and must be prevented by deployment copy controls.
    """

    # Freeze executable Python before complete-release authentication, which
    # can be long.  Loaded code and the recorded source snapshot now share the
    # same causal boundary.
    root = Path(__file__).resolve().parent
    frozen_sources = freeze_stage_a_source_dependencies(root)
    guard = _SourceGuard(root, frozen_sources)
    guard.check("before-input-authentication")
    inputs = authenticate_atomic_smoke_inputs(
        corpus_path=corpus_path,
        archive_path=archive_path,
        predecessor_ledger_path=predecessor_ledger_path,
        prior_attempt_record_path=prior_attempt_record_path,
        release_descriptor_path=release_descriptor_path,
    )
    guard.check("after-input-authentication")
    stores = _freeze_pristine_stores(
        {
            "config": config_store_dir,
            "exposure": exposure_store_dir,
            "journal": journal_store_dir,
            "prediction": prediction_store_dir,
            "terminal": terminal_store_dir,
            "cache": cache_store_dir,
            "preflight": preflight_store_dir,
        }
    )
    snapshot = cache_snapshotter()
    if not isinstance(snapshot, CloudPolicyCacheSnapshot):
        raise TypeError("cache_snapshotter must return CloudPolicyCacheSnapshot")
    stores["cache"].check("before-cache-persistence")
    cache_path, cache_file_sha256, cache_size = persist_stage_a_cache_snapshot(
        snapshot, cache_store_dir
    )
    if (
        not isinstance(cache_path, Path)
        or cache_path.parent.resolve(strict=True) != stores["cache"].path
        or not cache_path.is_file()
    ):
        raise AtomicSmokeCommandError(
            "cache persistence path differs from its pristine store"
        )
    reloaded_snapshot = load_stage_a_cache_snapshot(
        cache_path,
        expected_binding=snapshot.binding,
        expected_file_sha256=cache_file_sha256,
    )
    if reloaded_snapshot != snapshot:
        raise AtomicSmokeCommandError("reloaded cache snapshot differs")
    stores["cache"].check("after-cache-persistence-reload")
    config: AtomicSmokeCommandConfig | None = None
    config_receipt: AtomicSmokeDurabilityReceipt | None = None
    preflight: AtomicSmokeTransportPreflightReceipt | None = None
    preflight_receipt: AtomicSmokeDurabilityReceipt | None = None
    attempt_claim: AtomicSmokeAttemptClaim | None = None
    attempt_claim_path: Path | None = None
    precommit: AtomicSmokePrecommit | None = None
    launcher_version: str | None = None
    run: AtomicSmokeRun | None = None
    run_receipt: AtomicSmokeDurabilityReceipt | None = None
    phase = "launcher-staging"
    try:
        guard.check("before-launcher-staging")
        with launcher_stager(
            executable,
            expected_launcher_digest=expected_launcher_digest,
        ) as staged:
            if not isinstance(staged, StagedCodexLauncher):
                raise TypeError("launcher stager returned the wrong type")
            if staged.launcher_digest != expected_launcher_digest:
                raise AtomicSmokeCommandError("staged launcher differs from pin")
            launcher_version = staged.version
            guard.check("after-launcher-staging")

            phase = "transport-preflight"
            if not callable(preflight_runner):
                raise TypeError("preflight_runner must be callable")
            guard.check("before-transport-preflight")
            preflight = preflight_runner(
                staged=staged,
                model=model,
                reasoning_effort=reasoning_effort,
                transport=guard.wrap(
                    "preflight-text-model-call", text_transport
                ),
                cloud_policy_cache_snapshot=reloaded_snapshot,
            )
            if not isinstance(preflight, AtomicSmokeTransportPreflightReceipt):
                raise TypeError("preflight runner returned the wrong type")
            if (
                preflight.launcher_path != staged.executable
                or preflight.launcher_digest != staged.launcher_digest
                or preflight.launcher_version != staged.version
                or preflight.model != model
                or preflight.reasoning_effort != reasoning_effort
            ):
                raise AtomicSmokeCommandError(
                    "preflight receipt differs from staged execution"
                )
            validate_atomic_smoke_proposal_payload(
                _PREFLIGHT_PROPOSAL_PAYLOAD
            )
            validate_atomic_smoke_scorer_payload(
                _PREFLIGHT_SCORER_PAYLOAD,
                expected_atom_ids=(_PREFLIGHT_ATOM_ID,),
            )
            guard.check("after-transport-preflight")
            stores["preflight"].check("before-preflight-persistence")
            preflight_receipt = _persist_preflight(
                preflight, preflight_store_dir
            )
            stores["preflight"].check("after-preflight-persistence")

            config = AtomicSmokeCommandConfig(
                input_authentication_digest=inputs.digest,
                source_dependencies=frozen_sources,
                cache_binding=snapshot.binding,
                cache_file_sha256=cache_file_sha256,
                cache_byte_count=cache_size,
                expected_launcher_digest=expected_launcher_digest,
                staged_launcher_path=staged.executable,
                launcher_version=staged.version,
                preflight_receipt=preflight,
                preflight_receipt_file_sha256=preflight_receipt.file_sha256,
                preflight_receipt_filename=preflight_receipt.path.name,
                preflight_receipt_byte_count=preflight_receipt.byte_count,
                run_protocol_digest=atomic_smoke_run_protocol_digest(),
                model=model,
                reasoning_effort=reasoning_effort,
                minutes=minutes,
                verifier_id=verifier_id,
                verbose=verbose,
            )
            guard.check("before-command-config-persistence")
            stores["config"].check("before-command-config-persistence")
            config_receipt = _persist_config(config, config_store_dir)
            stores["config"].check("after-command-config-persistence")
            guard.check("after-command-config-persistence")

            phase = "attempt-claim"
            candidate_claim = AtomicSmokeAttemptClaim.create(
                inputs=inputs,
                config=config,
                preflight_persistence=preflight_receipt,
            )
            candidate_claim_path = _persist_attempt_claim(candidate_claim)
            attempt_claim = candidate_claim
            attempt_claim_path = candidate_claim_path
            guard.check("after-attempt-claim")

            # All output directories were empty at invocation start.  The four
            # stores that must still be untouched remain empty immediately
            # before the first random byte and before any exposure append.
            stores["config"].assert_exact_entries(
                {config_receipt.path.name}, "before-secret-generation"
            )
            stores["preflight"].assert_exact_entries(
                {preflight_receipt.path.name}, "before-secret-generation"
            )
            stores["cache"].assert_exact_entries(
                {cache_path.name}, "before-secret-generation"
            )
            for label in ("exposure", "journal", "prediction", "terminal"):
                stores[label].assert_exact_entries(
                    set(), "before-secret-generation"
                )
            phase = "secret-generation"
            selection_seed, episode_seed, label_nonce = _fresh_secrets(
                secret_factory
            )
            phase = "precommit"
            guard.check("before-precommit")
            stores["exposure"].check("before-exposure-precommit")
            precommit = prepare_atomic_smoke_precommit(
                inputs.trusted.corpus,
                seed=selection_seed,
                episode_seed=episode_seed,
                full_corpus_manifest=inputs.trusted.full_manifest,
                source_corpus_manifest_digest=OFFICIAL_CORPUS_MANIFEST_DIGEST,
                source_dependency_digest=frozen_sources.digest,
                exposure_ledger=inputs.predecessor,
                expected_exposure_ledger_digest=(
                    OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST
                ),
                label_seal_nonce=label_nonce,
                exposure_store_dir=exposure_store_dir,
                verifier_id=verifier_id,
            )
            stores["exposure"].check("after-exposure-precommit")
            guard.check("after-precommit")
            _assert_non_test_precommit(precommit)
            phase = "atomic-run"
            stores["journal"].check("before-atomic-run")
            stores["prediction"].check("before-atomic-run")
            run = run_atomic_smoke(
                precommit,
                source_dependency_digest=frozen_sources.digest,
                expected_protocol_digest=config.run_protocol_digest,
                expected_launcher_digest=config.expected_launcher_digest,
                command_config_digest=config.digest,
                journal_store_dir=journal_store_dir,
                prediction_store_dir=prediction_store_dir,
                model=config.model,
                reasoning_effort=config.reasoning_effort,
                minutes=config.minutes,
                executable=staged.executable,
                cloud_policy_cache_snapshot=reloaded_snapshot,
                named_image_transport=guard.wrap(
                    "named-image-model-call", named_image_transport
                ),
                text_transport=guard.wrap("text-model-call", text_transport),
                verbose=config.verbose,
            )
            if not isinstance(run, AtomicSmokeRun):
                raise TypeError("atomic runner returned the wrong terminal type")
            phase = "raw-run-persistence"
            stores["terminal"].check("before-raw-run-persistence")
            run_receipt = _persist_raw_run(run, terminal_store_dir)
            stores["terminal"].check("after-raw-run-persistence")
            phase = "atomic-run"
            stores["journal"].check("after-atomic-run")
            stores["prediction"].check("after-atomic-run")
            guard.check("after-atomic-run")
        guard.check("after-launcher-context")
        phase = "terminal-construction"
        guard.check("before-terminal-construction")
        terminal = AtomicSmokeCommandTerminal.from_run(
            run,
            config=config,
            attempt_claim=attempt_claim,
            launcher_version=launcher_version,
        )
        guard.check("after-terminal-construction")
    except Exception as exc:
        # Launcher/preflight/config failures have no durable config from which
        # to build a causally bound terminal.  They propagate before the claim,
        # all secrets, and all exposure.
        if (
            config is None
            or config_receipt is None
            or preflight is None
            or preflight_receipt is None
        ):
            raise
        terminal = AtomicSmokeCommandTerminal.failure(
            exc,
            phase=phase,
            config=config,
            precommit=precommit,
            launcher_version=launcher_version,
            attempt_claim=attempt_claim,
            run=run if isinstance(run, AtomicSmokeRun) else None,
        )
    try:
        stores["terminal"].check("before-terminal-persistence")
        terminal_receipt = _persist_terminal(
            terminal, terminal_store_dir, source_guard=guard
        )
        stores["terminal"].check("after-terminal-persistence")
    except AtomicSmokeSourceMutationError as exc:
        # A post-construction mutation may leave the earlier content-addressed
        # terminal on disk.  It is not returned as authoritative; a second,
        # explicit mutated-source terminal is persisted and returned instead.
        terminal = AtomicSmokeCommandTerminal.failure(
            exc,
            phase="terminal-persistence",
            config=config,
            precommit=precommit,
            launcher_version=launcher_version,
            attempt_claim=attempt_claim,
            run=run if isinstance(run, AtomicSmokeRun) else None,
        )
        stores["terminal"].check("before-mutated-terminal-persistence")
        terminal_receipt = _persist_terminal(terminal, terminal_store_dir)
        stores["terminal"].check("after-mutated-terminal-persistence")
    except Exception as exc:
        # If the primary outcome itself cannot be persisted, make one bounded
        # attempt to persist that operational failure at its own content
        # address.  Any unavailable/replaced store still escapes: without a
        # verified receipt there is no durable terminal outcome to return.
        terminal = AtomicSmokeCommandTerminal.failure(
            exc,
            phase="terminal-persistence",
            config=config,
            precommit=precommit,
            launcher_version=launcher_version,
            attempt_claim=attempt_claim,
            run=run if isinstance(run, AtomicSmokeRun) else None,
        )
        try:
            stores["terminal"].check("before-fallback-terminal-persistence")
            terminal_receipt = _persist_terminal(
                terminal, terminal_store_dir, source_guard=guard
            )
            stores["terminal"].check("after-fallback-terminal-persistence")
        except AtomicSmokeSourceMutationError as source_exc:
            terminal = AtomicSmokeCommandTerminal.failure(
                source_exc,
                phase="terminal-persistence",
                config=config,
                precommit=precommit,
                launcher_version=launcher_version,
                attempt_claim=attempt_claim,
                run=run if isinstance(run, AtomicSmokeRun) else None,
            )
            stores["terminal"].check(
                "before-fallback-mutated-terminal-persistence"
            )
            terminal_receipt = _persist_terminal(terminal, terminal_store_dir)
            stores["terminal"].check(
                "after-fallback-mutated-terminal-persistence"
            )
    return AtomicSmokeCommandResult(
        config=config,
        config_receipt=config_receipt,
        preflight=preflight,
        preflight_receipt=preflight_receipt,
        attempt_claim=attempt_claim,
        attempt_claim_path=attempt_claim_path,
        precommit=precommit,
        run_receipt=run_receipt,
        terminal=terminal,
        terminal_receipt=terminal_receipt,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the production boundary and emit one ID-redacted JSON status line."""

    parser = argparse.ArgumentParser(
        prog="python -m bongard.atomic_smoke_command",
        description="Run one authenticated exploratory atomic Bongard smoke.",
    )
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument(
        "--predecessor-ledger",
        "--exposure-ledger",
        dest="predecessor_ledger",
        required=True,
        help=(
            "canonical bfd ledger in a current-UID 0700 directory; the "
            "one-shot claim protects this pathname, not copied paths"
        ),
    )
    parser.add_argument(
        "--prior-attempt-record",
        "--prior-incident",
        dest="prior_attempt_record",
        default=str(DEFAULT_PRIOR_ATTEMPT_RECORD_PATH),
    )
    parser.add_argument("--config-store", required=True)
    parser.add_argument("--exposure-store", required=True)
    parser.add_argument("--journal-store", required=True)
    parser.add_argument("--prediction-store", required=True)
    parser.add_argument("--terminal-store", required=True)
    parser.add_argument("--cache-store", required=True)
    parser.add_argument(
        "--preflight-store",
        required=True,
        help="fresh current-UID 0700 store for the generic transport receipt",
    )
    parser.add_argument(
        "--release-descriptor", default=str(DEFAULT_RELEASE_PATH)
    )
    parser.add_argument(
        "--launcher-digest", default=ATOMIC_SMOKE_NATIVE_LAUNCHER_DIGEST
    )
    parser.add_argument("--model", default=DEFAULT_CODEX_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--minutes", type=int, default=15)
    parser.add_argument("--verifier-id", default="canonical-bongard-verifier")
    parser.add_argument("--executable", default="codex")
    args = parser.parse_args(argv)
    try:
        result = run_atomic_smoke_command(
            corpus_path=args.corpus,
            archive_path=args.archive,
            predecessor_ledger_path=args.predecessor_ledger,
            prior_attempt_record_path=args.prior_attempt_record,
            config_store_dir=args.config_store,
            exposure_store_dir=args.exposure_store,
            journal_store_dir=args.journal_store,
            prediction_store_dir=args.prediction_store,
            terminal_store_dir=args.terminal_store,
            cache_store_dir=args.cache_store,
            preflight_store_dir=args.preflight_store,
            release_descriptor_path=args.release_descriptor,
            expected_launcher_digest=args.launcher_digest,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            minutes=args.minutes,
            verifier_id=args.verifier_id,
            executable=args.executable,
            verbose=False,
        )
    except Exception as exc:
        reason = (str(exc) or repr(exc)).encode(
            "utf-8", errors="replace"
        )[:4096]
        payload = {
            "schema": "gkm.bongard-atomic-smoke-cli-result.v4",
            "scope": ATOMIC_SMOKE_COMMAND_SCOPE,
            "attempt_ordinal": ATOMIC_SMOKE_ATTEMPT_ORDINAL,
            "status": "operational-error-before-terminal",
            "error_type": type(exc).__name__,
            "reason_digest": hashlib.sha256(reason).hexdigest(),
            "transport_preflight_receipt_digest": None,
            "attempt_claim_digest": None,
            "attempt_claim_filename": None,
            "claim_protection_scope": (
                ATOMIC_SMOKE_CLAIM_PROTECTION_SCOPE
            ),
            "selected_task_id_included": False,
            "dependence_design_authorized": False,
            "calibration_authorized": False,
            "benchmark_claim_authorized": False,
            "official_test_authorized": False,
        }
        print(canonical_json(payload).decode("utf-8"), flush=True)
        return 2
    payload = {
        "schema": "gkm.bongard-atomic-smoke-cli-result.v4",
        "scope": ATOMIC_SMOKE_COMMAND_SCOPE,
        "attempt_ordinal": ATOMIC_SMOKE_ATTEMPT_ORDINAL,
        "status": result.terminal.status,
        "config_digest": result.config.digest,
        "terminal_digest": result.terminal.terminal_digest,
        "precommit_digest": result.terminal.precommit_digest,
        "run_digest": result.terminal.run_digest,
        "journal_receipt_digest": result.terminal.journal_receipt_digest,
        "journal_intent_count": (
            None
            if result.terminal.journal_receipt_data is None
            else result.terminal.journal_receipt_data["intent_count"]
        ),
        "journal_result_count": (
            None
            if result.terminal.journal_receipt_data is None
            else result.terminal.journal_receipt_data["result_count"]
        ),
        "transport_preflight_receipt_digest": result.preflight.receipt_digest,
        "attempt_claim_digest": (
            None
            if result.attempt_claim is None
            else result.attempt_claim.claim_digest
        ),
        "attempt_claim_filename": (
            None
            if result.attempt_claim_path is None
            else result.attempt_claim_path.name
        ),
        "claim_protection_scope": (
            ATOMIC_SMOKE_CLAIM_PROTECTION_SCOPE
        ),
        "run_persistence": (
            None if result.run_receipt is None else result.run_receipt.to_data()
        ),
        "config_persistence": result.config_receipt.to_data(),
        "terminal_persistence": result.terminal_receipt.to_data(),
        "selected_task_id_included": False,
        "dependence_design_authorized": False,
        "calibration_authorized": False,
        "benchmark_claim_authorized": False,
        "official_test_authorized": False,
    }
    print(canonical_json(payload).decode("utf-8"), flush=True)
    return 0 if result.terminal.status == "complete" else 2


__all__ = [
    "ATOMIC_SMOKE_ATTEMPT_ORDINAL",
    "ATOMIC_SMOKE_COMMAND_AUTHENTICATED_SCHEMA",
    "ATOMIC_SMOKE_COMMAND_CONFIG_SCHEMA",
    "ATOMIC_SMOKE_COMMAND_RECEIPT_SCHEMA",
    "ATOMIC_SMOKE_COMMAND_SCOPE",
    "ATOMIC_SMOKE_COMMAND_TERMINAL_SCHEMA",
    "ATOMIC_SMOKE_ATTEMPT_CLAIM_SCHEMA",
    "ATOMIC_SMOKE_CLAIM_PROTECTION_SCOPE",
    "ATOMIC_SMOKE_NATIVE_LAUNCHER_DIGEST",
    "ATOMIC_SMOKE_PREFLIGHT_PROTOCOL_SCHEMA",
    "ATOMIC_SMOKE_PREFLIGHT_SCHEMA",
    "ATOMIC_SMOKE_PRIOR_CONFIG_DIGEST",
    "ATOMIC_SMOKE_PRIOR_EVIDENCE_DIGEST",
    "ATOMIC_SMOKE_PRIOR_EXACT_ERROR",
    "ATOMIC_SMOKE_PRIOR_JOURNAL_HEADER_DIGEST",
    "ATOMIC_SMOKE_PRIOR_JOURNAL_RECEIPT_DIGEST",
    "ATOMIC_SMOKE_PRIOR_PRECOMMIT_DIGEST",
    "ATOMIC_SMOKE_PRIOR_PREDECESSOR_DIGEST",
    "ATOMIC_SMOKE_PRIOR_REASON_DIGEST",
    "ATOMIC_SMOKE_PRIOR_RECORD_FILE_SHA256",
    "ATOMIC_SMOKE_PRIOR_RECORD_SCHEMA",
    "ATOMIC_SMOKE_PRIOR_REMAINING_UNIVERSE_DIGEST",
    "ATOMIC_SMOKE_PRIOR_RUN_DIGEST",
    "ATOMIC_SMOKE_PRIOR_TERMINAL_DIGEST",
    "AtomicSmokeAuthenticatedInputs",
    "AtomicSmokeAttemptClaim",
    "AtomicSmokeCommandConfig",
    "AtomicSmokeCommandError",
    "AtomicSmokeCommandResult",
    "AtomicSmokeCommandTerminal",
    "AtomicSmokeDurabilityReceipt",
    "AtomicSmokePriorAttemptRecord",
    "AtomicSmokeTransportPreflightReceipt",
    "DEFAULT_PRIOR_ATTEMPT_RECORD_PATH",
    "authenticate_atomic_smoke_inputs",
    "atomic_smoke_preflight_protocol_data",
    "atomic_smoke_preflight_protocol_digest",
    "main",
    "run_atomic_smoke_command",
]


if __name__ == "__main__":
    raise SystemExit(main())

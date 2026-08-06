"""No-reroll production command boundary for the atomic Bongard smoke.

The boundary authenticates the complete official release, the consumed first
attempt, and its exact exposure successor before starting attempt two.  It
freezes every authoritative Python source, persists a secret-free command
commitment, and only then creates the three private seeds.  It owns the
exposure, journal, prediction, and terminal durability boundaries.  Nothing
in this module prints task identities.
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
    atomic_smoke_run_protocol_digest,
    run_atomic_smoke,
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
)


ATOMIC_SMOKE_COMMAND_AUTHENTICATED_SCHEMA = (
    "gkm.bongard-atomic-smoke-authenticated-inputs.v2"
)
ATOMIC_SMOKE_COMMAND_CONFIG_SCHEMA = "gkm.bongard-atomic-smoke-command-config.v2"
ATOMIC_SMOKE_COMMAND_TERMINAL_SCHEMA = "gkm.bongard-atomic-smoke-command-terminal.v2"
ATOMIC_SMOKE_COMMAND_RECEIPT_SCHEMA = "gkm.bongard-atomic-smoke-command-receipt.v1"
ATOMIC_SMOKE_ATTEMPT_ORDINAL = 2
ATOMIC_SMOKE_COMMAND_SCOPE = (
    "one-exploratory-repeated-generator-train-successor-smoke/v2"
)
ATOMIC_SMOKE_NATIVE_LAUNCHER_DIGEST = (
    "ae1d3ffe6d48aec6a4dc3f50e7eb8e0d11962485a6a9406c5a7012139383da02"
)
ATOMIC_SMOKE_PRIOR_INCIDENT_FILE_SHA256 = (
    "2cf35e733c9a392999ec904660b2b0bf17814c253e3936476023f3e815fc14ad"
)
ATOMIC_SMOKE_PRIOR_CONFIG_DIGEST = (
    "sha256:9dad0a5f468d1e8f3c65f7b83ac1ce7d2072e6541078bfbe9b4289ae3abdd451"
)
ATOMIC_SMOKE_PRIOR_OUTER_REASON_DIGEST = (
    "2825061e41346b498f7ceb0e338b0382fa807b2c968d534703927d6ce5f8376d"
)
ATOMIC_SMOKE_PRIOR_CALL_COUNT_LOWER = 0
ATOMIC_SMOKE_PRIOR_CALL_COUNT_UPPER = 29
DEFAULT_PRIOR_INCIDENT_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "atomic_smoke_n1_operational_failure_v1.json"
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


@dataclass(frozen=True, slots=True)
class AtomicSmokePriorIncident:
    """Exact machine-readable lineage of the consumed first live attempt."""

    file_sha256: str
    command_config_digest: str
    exposure_successor_digest: str
    outer_reason_digest: str
    prediction_persisted: bool
    terminal_persisted: bool
    successful_call_count_known: bool
    successful_call_count_lower: int
    successful_call_count_upper: int
    selected_task_consumed: bool
    selected_task_may_be_rerolled: bool
    runner_returned_typed_run: bool
    typed_run_output_recoverable: bool

    def __post_init__(self) -> None:
        if (
            self.file_sha256 != ATOMIC_SMOKE_PRIOR_INCIDENT_FILE_SHA256
            or self.command_config_digest != ATOMIC_SMOKE_PRIOR_CONFIG_DIGEST
            or self.exposure_successor_digest
            != OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST
            or self.outer_reason_digest != ATOMIC_SMOKE_PRIOR_OUTER_REASON_DIGEST
            or self.prediction_persisted is not False
            or self.terminal_persisted is not False
            or self.successful_call_count_known is not False
            or type(self.successful_call_count_lower) is not int
            or type(self.successful_call_count_upper) is not int
            or self.successful_call_count_lower
            != ATOMIC_SMOKE_PRIOR_CALL_COUNT_LOWER
            or self.successful_call_count_upper
            != ATOMIC_SMOKE_PRIOR_CALL_COUNT_UPPER
            or self.selected_task_consumed is not True
            or self.selected_task_may_be_rerolled is not False
            or self.runner_returned_typed_run is not True
            or self.typed_run_output_recoverable is not False
        ):
            raise AtomicSmokeCommandError("prior incident lineage differs")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": "gkm.bongard-atomic-smoke-prior-incident-binding.v1",
            "incident_record_schema": (
                "gkm.bongard-atomic-smoke-n1-operational-failure.v1"
            ),
            "incident_file_sha256": self.file_sha256,
            "prior_command_config_digest": self.command_config_digest,
            "active_exposure_predecessor_digest": (
                self.exposure_successor_digest
            ),
            "prior_outer_reason_digest": self.outer_reason_digest,
            "prediction_persisted": self.prediction_persisted,
            "terminal_persisted": self.terminal_persisted,
            "successful_model_call_count": {
                "known": self.successful_call_count_known,
                "lower_bound_inclusive": self.successful_call_count_lower,
                "upper_bound_inclusive": self.successful_call_count_upper,
            },
            "selected_task_consumed": self.selected_task_consumed,
            "selected_task_may_be_rerolled": self.selected_task_may_be_rerolled,
            "runner_returned_typed_atomic_smoke_run": (
                self.runner_returned_typed_run
            ),
            "typed_run_output_recoverable": self.typed_run_output_recoverable,
        }

    @property
    def digest(self) -> str:
        return "sha256:" + canonical_digest(self.to_data())

    @classmethod
    def load(cls, path: str | Path) -> "AtomicSmokePriorIncident":
        incident_path = Path(path).expanduser().absolute()
        payload = _stable_read(incident_path, maximum=1_048_576)
        file_sha256 = hashlib.sha256(payload).hexdigest()
        if file_sha256 != ATOMIC_SMOKE_PRIOR_INCIDENT_FILE_SHA256:
            raise AtomicSmokeCommandError("prior incident file differs from exact pin")
        try:
            raw = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise AtomicSmokeCommandError("prior incident is not exact JSON") from exc
        if not isinstance(raw, Mapping) or raw.get("schema") != (
            "gkm.bongard-atomic-smoke-n1-operational-failure.v1"
        ):
            raise AtomicSmokeCommandError("prior incident schema differs")
        try:
            artifacts = raw["artifacts"]
            attempt = raw["consuming_attempt"]
            forensics = raw["forensics"]
            count = forensics["successful_model_call_count"]
            claims = raw["claim_policy"]
            values = cls(
                file_sha256=file_sha256,
                command_config_digest=artifacts[
                    "command_config_content_address"
                ],
                exposure_successor_digest=artifacts[
                    "exposure_successor_content_address"
                ],
                outer_reason_digest=attempt["reason_digest"],
                prediction_persisted=artifacts["prediction_persisted"],
                terminal_persisted=artifacts["terminal_persisted"],
                successful_call_count_known=count["known"],
                successful_call_count_lower=count["lower_bound_inclusive"],
                successful_call_count_upper=count["upper_bound_inclusive"],
                selected_task_consumed=attempt["selected_task_consumed"],
                selected_task_may_be_rerolled=attempt[
                    "selected_task_may_be_rerolled"
                ],
                runner_returned_typed_run=forensics[
                    "runner_returned_typed_atomic_smoke_run"
                ],
                typed_run_output_recoverable=forensics[
                    "typed_run_output_recoverable"
                ],
            )
        except (KeyError, TypeError) as exc:
            raise AtomicSmokeCommandError("prior incident facts are malformed") from exc
        if (
            attempt.get("cli_error_type") != "AtomicSmokeCommandError"
            or attempt.get("exact_error")
            != "failed run precommit is not canonical JSON"
            or attempt.get("labels_materialized") is not False
            or attempt.get("labels_revealed") is not False
            or attempt.get("result_class") != "operational_failure"
            or forensics.get("runner_entered") is not True
            or forensics.get("run_phase_known") is not False
            or forensics.get("run_status_known") is not False
            or not isinstance(claims, Mapping)
            or any(value is not False for value in claims.values())
        ):
            raise AtomicSmokeCommandError("prior incident causal facts differ")
        return values


@dataclass(frozen=True, slots=True)
class AtomicSmokeAuthenticatedInputs:
    """Official corpus, active predecessor, historical parent, and incident."""

    trusted: StageATrustedCorpus = field(repr=False)
    release: OfficialReleaseDescriptor
    predecessor: ExposureLedger
    prior_incident: AtomicSmokePriorIncident

    def __post_init__(self) -> None:
        if not isinstance(self.trusted, StageATrustedCorpus):
            raise TypeError("trusted must be StageATrustedCorpus")
        if not isinstance(self.release, OfficialReleaseDescriptor):
            raise TypeError("release must be OfficialReleaseDescriptor")
        if not isinstance(self.predecessor, ExposureLedger):
            raise TypeError("predecessor must be ExposureLedger")
        if not isinstance(self.prior_incident, AtomicSmokePriorIncident):
            raise TypeError("prior_incident must be AtomicSmokePriorIncident")
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
        if not self.predecessor.events:
            raise AtomicSmokeCommandError("active predecessor has no prior append")
        historical_parent = ExposureLedger(
            corpus_digest=self.predecessor.corpus_digest,
            events=self.predecessor.events[:-1],
        )
        if historical_parent.digest != OFFICIAL_A3_LEDGER_DIGEST:
            raise AtomicSmokeCommandError(
                "active predecessor does not descend exactly from historical A3"
            )
        if not self.predecessor.exposed_task_ids <= set(self.trusted.corpus.task_ids):
            raise AtomicSmokeCommandError(
                "active predecessor contains IDs outside the corpus"
            )
        if (
            self.prior_incident.exposure_successor_digest
            != self.predecessor.digest
        ):
            raise AtomicSmokeCommandError(
                "prior incident differs from active predecessor"
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
            "historical_a3_parent_ledger_digest": OFFICIAL_A3_LEDGER_DIGEST,
            "prior_incident": self.prior_incident.to_data(),
            "prior_incident_binding_digest": self.prior_incident.digest,
        }

    @property
    def digest(self) -> str:
        return "sha256:" + canonical_digest(self.to_data())


def authenticate_atomic_smoke_inputs(
    *,
    corpus_path: str | Path,
    archive_path: str | Path,
    predecessor_ledger_path: str | Path,
    prior_incident_path: str | Path = DEFAULT_PRIOR_INCIDENT_PATH,
    release_descriptor_path: str | Path = DEFAULT_RELEASE_PATH,
) -> AtomicSmokeAuthenticatedInputs:
    """Authenticate the official release, successor predecessor, and incident."""

    prior_incident = AtomicSmokePriorIncident.load(prior_incident_path)
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
    ledger_path = Path(predecessor_ledger_path).expanduser().absolute()
    try:
        ledger_raw = json.loads(_stable_read(ledger_path))
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
        trusted, release, predecessor, prior_incident
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
            "official_release_descriptor_digest": (
                OFFICIAL_RELEASE_DESCRIPTOR_DIGEST
            ),
            "official_corpus_manifest_digest": OFFICIAL_CORPUS_MANIFEST_DIGEST,
            "official_split_source_digest": OFFICIAL_SPLIT_SOURCE_DIGEST,
            "official_active_exposure_predecessor_digest": (
                OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST
            ),
            "official_historical_a3_parent_ledger_digest": (
                OFFICIAL_A3_LEDGER_DIGEST
            ),
            "prior_incident_file_sha256": (
                ATOMIC_SMOKE_PRIOR_INCIDENT_FILE_SHA256
            ),
            "prior_attempt_command_config_digest": (
                ATOMIC_SMOKE_PRIOR_CONFIG_DIGEST
            ),
            "prior_attempt_outer_reason_digest": (
                ATOMIC_SMOKE_PRIOR_OUTER_REASON_DIGEST
            ),
            "prior_attempt_prediction_persisted": False,
            "prior_attempt_terminal_persisted": False,
            "prior_attempt_successful_model_call_count": {
                "known": False,
                "lower_bound_inclusive": ATOMIC_SMOKE_PRIOR_CALL_COUNT_LOWER,
                "upper_bound_inclusive": ATOMIC_SMOKE_PRIOR_CALL_COUNT_UPPER,
            },
            "prior_attempt_selected_task_consumed": True,
            "prior_attempt_selected_task_may_be_rerolled": False,
            "input_authentication_digest": self.input_authentication_digest,
            "source_dependencies": self.source_dependencies.to_data(),
            "source_dependency_digest": self.source_dependencies.digest,
            "cloud_policy_cache_binding": self.cache_binding,
            "cloud_policy_cache_snapshot_file_sha256": self.cache_file_sha256,
            "cloud_policy_cache_snapshot_byte_count": self.cache_byte_count,
            "expected_launcher_digest": self.expected_launcher_digest,
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
            "schema", "scope", "attempt_ordinal",
            "official_release_descriptor_digest",
            "official_corpus_manifest_digest", "official_split_source_digest",
            "official_active_exposure_predecessor_digest",
            "official_historical_a3_parent_ledger_digest",
            "prior_incident_file_sha256", "prior_attempt_command_config_digest",
            "prior_attempt_outer_reason_digest",
            "prior_attempt_prediction_persisted",
            "prior_attempt_terminal_persisted",
            "prior_attempt_successful_model_call_count",
            "prior_attempt_selected_task_consumed",
            "prior_attempt_selected_task_may_be_rerolled",
            "input_authentication_digest",
            "source_dependencies", "source_dependency_digest",
            "cloud_policy_cache_binding",
            "cloud_policy_cache_snapshot_file_sha256",
            "cloud_policy_cache_snapshot_byte_count", "expected_launcher_digest",
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
        prior_count = value["prior_attempt_successful_model_call_count"]
        if (
            not isinstance(prior_count, Mapping)
            or set(prior_count)
            != {"known", "lower_bound_inclusive", "upper_bound_inclusive"}
            or prior_count["known"] is not False
            or type(prior_count["lower_bound_inclusive"]) is not int
            or type(prior_count["upper_bound_inclusive"]) is not int
        ):
            raise AtomicSmokeCommandError("prior attempt call-count lineage differs")
        if (
            value["schema"] != ATOMIC_SMOKE_COMMAND_CONFIG_SCHEMA
            or value["scope"] != ATOMIC_SMOKE_COMMAND_SCOPE
            or value["attempt_ordinal"] != ATOMIC_SMOKE_ATTEMPT_ORDINAL
            or value["official_release_descriptor_digest"]
            != OFFICIAL_RELEASE_DESCRIPTOR_DIGEST
            or value["official_corpus_manifest_digest"]
            != OFFICIAL_CORPUS_MANIFEST_DIGEST
            or value["official_split_source_digest"] != OFFICIAL_SPLIT_SOURCE_DIGEST
            or value["official_active_exposure_predecessor_digest"]
            != OFFICIAL_SUCCESSOR_PREDECESSOR_LEDGER_DIGEST
            or value["official_historical_a3_parent_ledger_digest"]
            != OFFICIAL_A3_LEDGER_DIGEST
            or value["prior_incident_file_sha256"]
            != ATOMIC_SMOKE_PRIOR_INCIDENT_FILE_SHA256
            or value["prior_attempt_command_config_digest"]
            != ATOMIC_SMOKE_PRIOR_CONFIG_DIGEST
            or value["prior_attempt_outer_reason_digest"]
            != ATOMIC_SMOKE_PRIOR_OUTER_REASON_DIGEST
            or value["prior_attempt_prediction_persisted"] is not False
            or value["prior_attempt_terminal_persisted"] is not False
            or value["prior_attempt_successful_model_call_count"]
            != {
                "known": False,
                "lower_bound_inclusive": ATOMIC_SMOKE_PRIOR_CALL_COUNT_LOWER,
                "upper_bound_inclusive": ATOMIC_SMOKE_PRIOR_CALL_COUNT_UPPER,
            }
            or value["prior_attempt_selected_task_consumed"] is not True
            or value["prior_attempt_selected_task_may_be_rerolled"] is not False
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
        ):
            raise AtomicSmokeCommandError("command config authority differs")
        sources = StageASourceDependencyIdentity.from_data(
            value["source_dependencies"]
        )
        if value["source_dependency_digest"] != sources.digest:
            raise AtomicSmokeCommandError("command source digest parent differs")
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
        _hex(self.launcher_digest, "terminal launcher digest")
        if self.launcher_digest != ATOMIC_SMOKE_NATIVE_LAUNCHER_DIGEST:
            raise AtomicSmokeCommandError("terminal launcher differs from production pin")
        if self.launcher_version is not None:
            _text(self.launcher_version, "launcher version", maximum=128)
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
        failure = value["failure"]
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
        config_digest: str,
        launcher_version: str,
    ) -> "AtomicSmokeCommandTerminal":
        run_data, precommit_data = _canonical_run_views(run)
        if run.command_config_digest != config_digest:
            raise AtomicSmokeCommandError("run differs from exact command config")
        journal_receipt_data = _canonical_clone(
            run.journal_receipt.to_data(), "run journal receipt"
        )
        values = {
            "status": "complete" if run.status == "complete" else "failed",
            "phase": run.terminal_phase,
            "config_digest": config_digest,
            "source_dependency_digest": run.source_dependency_digest,
            "source_dependency_state": "unchanged",
            "observed_source_dependency_digest": run.source_dependency_digest,
            "source_observation_error_digest": None,
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
    precommit: AtomicSmokePrecommit | None = field(repr=False)
    run_receipt: AtomicSmokeDurabilityReceipt | None
    terminal: AtomicSmokeCommandTerminal
    terminal_receipt: AtomicSmokeDurabilityReceipt

    def __post_init__(self) -> None:
        if (
            self.config_receipt.kind != "command-config"
            or self.config_receipt.content_address != self.config.digest
            or self.terminal.config_digest != self.config.digest
            or self.terminal_receipt.kind != "terminal-outcome"
            or self.terminal_receipt.content_address
            != self.terminal.terminal_digest
        ):
            raise AtomicSmokeCommandError("command result durability chain differs")
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
    prior_incident_path: str | Path = DEFAULT_PRIOR_INCIDENT_PATH,
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
    named_image_transport: NamedImageTransport = run_codex_named_images_structured,
    text_transport: TextTransport = run_codex_text_structured,
) -> AtomicSmokeCommandResult:
    """Run successor attempt two after authenticating the consumed first attempt."""

    inputs = authenticate_atomic_smoke_inputs(
        corpus_path=corpus_path,
        archive_path=archive_path,
        predecessor_ledger_path=predecessor_ledger_path,
        prior_incident_path=prior_incident_path,
        release_descriptor_path=release_descriptor_path,
    )
    stores = {
        "config": _StoreBinding.freeze("config", config_store_dir),
        "exposure": _StoreBinding.freeze("exposure", exposure_store_dir),
        "journal": _StoreBinding.freeze("journal", journal_store_dir),
        "prediction": _StoreBinding.freeze("prediction", prediction_store_dir),
        "terminal": _StoreBinding.freeze("terminal", terminal_store_dir),
        "cache": _StoreBinding.freeze("cache", cache_store_dir),
    }
    root = Path(__file__).resolve().parent
    frozen_sources = freeze_stage_a_source_dependencies(root)
    snapshot = cache_snapshotter()
    if not isinstance(snapshot, CloudPolicyCacheSnapshot):
        raise TypeError("cache_snapshotter must return CloudPolicyCacheSnapshot")
    stores["cache"].check("before-cache-persistence")
    cache_path, cache_file_sha256, cache_size = persist_stage_a_cache_snapshot(
        snapshot, cache_store_dir
    )
    reloaded_snapshot = load_stage_a_cache_snapshot(
        cache_path,
        expected_binding=snapshot.binding,
        expected_file_sha256=cache_file_sha256,
    )
    if reloaded_snapshot != snapshot:
        raise AtomicSmokeCommandError("reloaded cache snapshot differs")
    stores["cache"].check("after-cache-persistence-reload")
    config = AtomicSmokeCommandConfig(
        input_authentication_digest=inputs.digest,
        source_dependencies=frozen_sources,
        cache_binding=snapshot.binding,
        cache_file_sha256=cache_file_sha256,
        cache_byte_count=cache_size,
        expected_launcher_digest=expected_launcher_digest,
        run_protocol_digest=atomic_smoke_run_protocol_digest(),
        model=model,
        reasoning_effort=reasoning_effort,
        minutes=minutes,
        verifier_id=verifier_id,
        verbose=verbose,
    )
    guard = _SourceGuard(root, frozen_sources)
    # This verified return is the causal boundary before any random secret.
    guard.check("before-command-config-persistence")
    stores["config"].check("before-command-config-persistence")
    config_receipt = _persist_config(config, config_store_dir)
    stores["config"].check("after-command-config-persistence")
    guard.check("after-command-config-persistence")
    precommit: AtomicSmokePrecommit | None = None
    launcher_version: str | None = None
    run: AtomicSmokeRun | None = None
    run_receipt: AtomicSmokeDurabilityReceipt | None = None
    phase = "launcher-staging"
    try:
        guard.check("before-launcher-staging")
        with launcher_stager(
            executable,
            expected_launcher_digest=config.expected_launcher_digest,
        ) as staged:
            if not isinstance(staged, StagedCodexLauncher):
                raise TypeError("launcher stager returned the wrong type")
            if staged.launcher_digest != config.expected_launcher_digest:
                raise AtomicSmokeCommandError("staged launcher differs from config")
            launcher_version = staged.version
            guard.check("after-launcher-staging")
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
            config_digest=config.digest,
            launcher_version=launcher_version,
        )
        guard.check("after-terminal-construction")
    except Exception as exc:
        terminal = AtomicSmokeCommandTerminal.failure(
            exc,
            phase=phase,
            config=config,
            precommit=precommit,
            launcher_version=launcher_version,
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
    )
    parser.add_argument(
        "--prior-incident", default=str(DEFAULT_PRIOR_INCIDENT_PATH)
    )
    parser.add_argument("--config-store", required=True)
    parser.add_argument("--exposure-store", required=True)
    parser.add_argument("--journal-store", required=True)
    parser.add_argument("--prediction-store", required=True)
    parser.add_argument("--terminal-store", required=True)
    parser.add_argument("--cache-store", required=True)
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
            prior_incident_path=args.prior_incident,
            config_store_dir=args.config_store,
            exposure_store_dir=args.exposure_store,
            journal_store_dir=args.journal_store,
            prediction_store_dir=args.prediction_store,
            terminal_store_dir=args.terminal_store,
            cache_store_dir=args.cache_store,
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
            "schema": "gkm.bongard-atomic-smoke-cli-result.v2",
            "scope": ATOMIC_SMOKE_COMMAND_SCOPE,
            "attempt_ordinal": ATOMIC_SMOKE_ATTEMPT_ORDINAL,
            "status": "operational-error-before-terminal",
            "error_type": type(exc).__name__,
            "reason_digest": hashlib.sha256(reason).hexdigest(),
            "selected_task_id_included": False,
            "dependence_design_authorized": False,
            "calibration_authorized": False,
            "benchmark_claim_authorized": False,
            "official_test_authorized": False,
        }
        print(canonical_json(payload).decode("utf-8"), flush=True)
        return 2
    payload = {
        "schema": "gkm.bongard-atomic-smoke-cli-result.v2",
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
    "ATOMIC_SMOKE_COMMAND_SCOPE",
    "ATOMIC_SMOKE_COMMAND_TERMINAL_SCHEMA",
    "ATOMIC_SMOKE_NATIVE_LAUNCHER_DIGEST",
    "ATOMIC_SMOKE_PRIOR_CONFIG_DIGEST",
    "ATOMIC_SMOKE_PRIOR_INCIDENT_FILE_SHA256",
    "ATOMIC_SMOKE_PRIOR_OUTER_REASON_DIGEST",
    "AtomicSmokeAuthenticatedInputs",
    "AtomicSmokeCommandConfig",
    "AtomicSmokeCommandError",
    "AtomicSmokeCommandResult",
    "AtomicSmokeCommandTerminal",
    "AtomicSmokeDurabilityReceipt",
    "AtomicSmokePriorIncident",
    "DEFAULT_PRIOR_INCIDENT_PATH",
    "authenticate_atomic_smoke_inputs",
    "main",
    "run_atomic_smoke_command",
]


if __name__ == "__main__":
    raise SystemExit(main())

"""Operational, write-once command boundary for Stage-A calibration.

This module deliberately does not own argument parsing.  It provides three
separate operations instead:

``authenticate inputs -> execute campaign -> persist terminal outcome``.

The split makes the expensive/model-bearing middle operation testable without
filesystem effects, while the final convenience function gives a CLI one
callable that cannot discard a canonical terminal failure.  Python is the
authoritative implementation; no proof-checker participates in execution,
verification, identifiers, or persistence.

Stage A estimates performance conditional on a proposer emitting a soft
claim.  A successful artifact is *not* evidence that mixed generator units
are independent and is not authorization for a support-gated or sealed run.
Those questions belong to the separately frozen Stage-B design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from threading import RLock
from typing import Any, Callable, Literal, Mapping

from bongard.artifacts import canonical_digest, canonical_json
from bongard.corpus import CorpusManifest, ShapeBongardCorpus
from bongard.exposure import ExposureLedger
from bongard.release import OfficialReleaseDescriptor
from bongard.semantic_calibration_campaign import (
    CAMPAIGN_SELECTION_ALGORITHM,
    SemanticCalibrationCampaignArtifact,
    SemanticCalibrationCampaignFitFailed,
    SemanticCalibrationCampaignNoSoftClaims,
    SemanticCalibrationCampaignProposalPhaseFailed,
    SemanticCalibrationCampaignScoringFailed,
    run_semantic_calibration_campaign,
    verify_semantic_campaign_against_corpus,
)
from bongard.semantic_protocol import build_prospective_soft_scorer_protocol
from bongard.soft_predicates import SoftScorerProtocol
from bongard.transport import (
    CloudPolicyCacheSnapshot,
    codex_cli_authenticated_fingerprint,
    run_codex_named_images_structured,
    run_codex_structured,
    snapshot_cloud_policy_cache,
)


STAGE_A_COMMAND_CONFIG_SCHEMA = "gkm.bongard-stage-a-command-config.v1"
STAGE_A_COMMAND_RECEIPT_SCHEMA_V1 = "gkm.bongard-stage-a-command-receipt.v1"
STAGE_A_COMMAND_RECEIPT_SCHEMA = "gkm.bongard-stage-a-command-receipt.v2"
STAGE_A_INPUT_AUTHENTICATION_SCHEMA = (
    "gkm.bongard-stage-a-input-authentication.v1"
)
STAGE_A_SOURCE_DEPENDENCY_SCHEMA = (
    "gkm.bongard-stage-a-source-dependency-identity.v1"
)
STAGE_A_SOURCE_DEPENDENCY_SCOPE = (
    "all-authoritative-non-test-non-crack-lab-bongard-python-sources;"
    "semantic-checker-sidecar-excluded/v2"
)
STAGE_A_OPERATIONAL_FAILURE_SCHEMA = (
    "gkm.bongard-stage-a-operational-failure.v1"
)
STAGE_A_SCOPE = (
    "descriptive-exploratory;conditional-on-soft-claim-emission;does-not-"
    "certify-unit-independence-population-inference-or-support-gated-"
    "deployment/v2"
)
DESCRIPTIVE_STAGE_A_DESIGN = "descriptive-exploratory-only/v1"

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TERMINAL_EXCEPTIONS = (
    SemanticCalibrationCampaignProposalPhaseFailed,
    SemanticCalibrationCampaignNoSoftClaims,
    SemanticCalibrationCampaignScoringFailed,
    SemanticCalibrationCampaignFitFailed,
)

CampaignRunner = Callable[..., SemanticCalibrationCampaignArtifact]
CampaignVerifier = Callable[..., tuple[SemanticCalibrationCampaignArtifact, Mapping[str, Any]]]
LauncherFingerprinter = Callable[..., Mapping[str, str]]
CacheSnapshotter = Callable[[], CloudPolicyCacheSnapshot]
ExposurePrecommit = Callable[
    [ExposureLedger, CloudPolicyCacheSnapshot],
    None,
]


class StageACalibrationCommandError(RuntimeError):
    """The operational boundary or its persistence contract was violated."""


@dataclass(frozen=True, slots=True)
class StageASourceDependencyIdentity:
    """Exact content identity of every executable Python source in Stage A.

    Paths are package-relative and therefore do not disclose a checkout path.
    The scope deliberately includes all non-test, non-crack-lab Python modules
    under ``bongard`` except the exact non-authoritative
    ``semantic_checker.py`` sidecar boundary.  This conservative superset
    closes lazy-import holes: adding, deleting, or changing any potentially
    authoritative module changes the identity even when that module had not
    yet been imported at startup.  Installing, changing, or deleting the
    detached checker sidecar cannot change an authoritative receipt identity.
    """

    entries: tuple[tuple[str, int, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or not self.entries:
            raise StageACalibrationCommandError(
                "Stage-A source dependency identity must contain source files"
            )
        paths: list[str] = []
        for entry in self.entries:
            if not isinstance(entry, tuple) or len(entry) != 3:
                raise StageACalibrationCommandError(
                    "Stage-A source dependency entry is not canonical"
                )
            relative_path, byte_count, source_digest = entry
            if (
                not isinstance(relative_path, str)
                or not relative_path
                or relative_path.startswith("/")
                or ".." in Path(relative_path).parts
                or "\\" in relative_path
                or not relative_path.endswith(".py")
            ):
                raise StageACalibrationCommandError(
                    "Stage-A source dependency path is invalid"
                )
            if (
                isinstance(byte_count, bool)
                or not isinstance(byte_count, int)
                or byte_count < 0
            ):
                raise StageACalibrationCommandError(
                    "Stage-A source dependency byte count is invalid"
                )
            _hex(source_digest, "Stage-A source dependency digest")
            paths.append(relative_path)
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise StageACalibrationCommandError(
                "Stage-A source dependency paths are not unique and sorted"
            )

    def content_data(self) -> dict[str, object]:
        return {
            "schema": STAGE_A_SOURCE_DEPENDENCY_SCHEMA,
            "scope": STAGE_A_SOURCE_DEPENDENCY_SCOPE,
            "files": [
                {
                    "relative_path": relative_path,
                    "byte_count": byte_count,
                    "sha256": source_digest,
                }
                for relative_path, byte_count, source_digest in self.entries
            ],
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "source_dependency_digest": self.digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "StageASourceDependencyIdentity":
        if not isinstance(value, Mapping) or set(value) != {
            "schema",
            "scope",
            "files",
            "source_dependency_digest",
        }:
            raise StageACalibrationCommandError(
                "Stage-A source dependency identity fields differ"
            )
        if (
            value["schema"] != STAGE_A_SOURCE_DEPENDENCY_SCHEMA
            or value["scope"] != STAGE_A_SOURCE_DEPENDENCY_SCOPE
            or not isinstance(value["files"], list)
        ):
            raise StageACalibrationCommandError(
                "unsupported Stage-A source dependency identity"
            )
        entries: list[tuple[str, int, str]] = []
        for raw in value["files"]:
            if not isinstance(raw, Mapping) or set(raw) != {
                "relative_path",
                "byte_count",
                "sha256",
            }:
                raise StageACalibrationCommandError(
                    "Stage-A source dependency file identity fields differ"
                )
            entries.append(
                (raw["relative_path"], raw["byte_count"], raw["sha256"])
            )
        result = cls(tuple(entries))
        archived = _hex(
            value["source_dependency_digest"],
            "Stage-A source dependency identity digest",
        )
        if result.digest != archived or result.to_data() != dict(value):
            raise StageACalibrationCommandError(
                "Stage-A source dependency identity digest differs"
            )
        return result


class StageASourceDependencyMutationError(StageACalibrationCommandError):
    """A source tree no longer matches the identity frozen at command start."""

    def __init__(
        self,
        *,
        phase: str,
        expected: StageASourceDependencyIdentity,
        observed: StageASourceDependencyIdentity | None,
        observation_error_digest: str | None = None,
    ) -> None:
        self.phase = _bounded_text(phase, "source dependency check phase")
        self.expected = expected
        self.observed = observed
        self.observation_error_digest = observation_error_digest
        if observed is None:
            _hex(observation_error_digest, "source observation error digest")
            detail = "could not be read"
        else:
            if observation_error_digest is not None:
                raise StageACalibrationCommandError(
                    "readable source mutation cannot have an observation error"
                )
            detail = f"changed to {observed.digest}"
        super().__init__(
            f"Stage-A source dependencies {detail} during {self.phase}; "
            f"expected {expected.digest}"
        )


_SOURCE_EXCLUDED_TOP_LEVEL = frozenset({"tests", "crack_lab", "manuscript"})
_SOURCE_EXCLUDED_RELATIVE_FILES = frozenset({"semantic_checker.py"})


def _read_stable_source(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise StageACalibrationCommandError(
                    f"Stage-A source dependency is not a regular file: {path}"
                )
            blocks: list[bytes] = []
            while block := os.read(descriptor, 1024 * 1024):
                blocks.append(block)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise StageACalibrationCommandError(
            f"cannot read Stage-A source dependency {path}"
        ) from exc
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise StageACalibrationCommandError(
            f"Stage-A source dependency changed while being read: {path}"
        )
    payload = b"".join(blocks)
    if len(payload) != before.st_size:
        raise StageACalibrationCommandError(
            f"Stage-A source dependency size changed while being read: {path}"
        )
    return payload


def freeze_stage_a_source_dependencies(
    root: str | Path | None = None,
) -> StageASourceDependencyIdentity:
    """Snapshot the complete conservative Stage-A Python source boundary."""

    source_root = (
        Path(__file__).resolve().parent
        if root is None
        else Path(root).expanduser().resolve()
    )
    if not source_root.is_dir():
        raise StageACalibrationCommandError(
            "Stage-A source dependency root must be a directory"
        )
    entries: list[tuple[str, int, str]] = []
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root)
        if (
            not relative.parts
            or relative.parts[0] in _SOURCE_EXCLUDED_TOP_LEVEL
            or relative.as_posix() in _SOURCE_EXCLUDED_RELATIVE_FILES
            or "__pycache__" in relative.parts
        ):
            continue
        payload = _read_stable_source(path)
        entries.append(
            (
                relative.as_posix(),
                len(payload),
                hashlib.sha256(payload).hexdigest(),
            )
        )
    return StageASourceDependencyIdentity(tuple(entries))


class _StageASourceDependencyGuard:
    def __init__(self, root: Path, expected: StageASourceDependencyIdentity) -> None:
        self.root = root
        self.expected = expected
        self._lock = RLock()
        self.first_failure: StageASourceDependencyMutationError | None = None

    def check(self, phase: str) -> None:
        with self._lock:
            if self.first_failure is not None:
                raise self.first_failure
            try:
                observed = freeze_stage_a_source_dependencies(self.root)
            except BaseException as exc:  # retain a bounded unreadable-state proof.
                reason = (str(exc) or repr(exc)).encode("utf-8", errors="replace")[
                    :4_000
                ]
                failure = StageASourceDependencyMutationError(
                    phase=phase,
                    expected=self.expected,
                    observed=None,
                    observation_error_digest=hashlib.sha256(reason).hexdigest(),
                )
                self.first_failure = failure
                raise failure from exc
            if observed != self.expected:
                failure = StageASourceDependencyMutationError(
                    phase=phase,
                    expected=self.expected,
                    observed=observed,
                )
                self.first_failure = failure
                raise failure

    def wrap_transport(self, phase: Literal["proposer", "scorer"], transport: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(transport):
            raise TypeError("guarded transport must be callable")

        def guarded(*args: Any, **kwargs: Any) -> Any:
            self.check(f"before-{phase}")
            try:
                return transport(*args, **kwargs)
            finally:
                self.check(f"after-{phase}")

        return guarded


def _hex(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise StageACalibrationCommandError(
            f"{label} must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise StageACalibrationCommandError(
            f"{label} must be a sha256: content address"
        )
    return value


def _bounded_text(value: object, label: str, *, maximum: int = 1024) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > maximum
    ):
        raise StageACalibrationCommandError(
            f"{label} must be a bounded non-empty string"
        )
    return value


def _assert_manifest_matches_corpus_metadata(
    corpus: ShapeBongardCorpus,
    manifest: CorpusManifest,
) -> None:
    """Reject an internally inconsistent caller-attested object pair.

    This check is metadata-only.  Only :meth:`StageATrustedCorpus.from_official_release`
    authenticates all panel bytes; ``from_trusted_objects`` explicitly relies
    on an upstream authority having already done that work.
    """

    if manifest.layout != corpus.layout:
        raise StageACalibrationCommandError(
            "trusted manifest layout differs from trusted corpus"
        )
    if dict(manifest.family_counts) != dict(corpus.family_counts):
        raise StageACalibrationCommandError(
            "trusted manifest family counts differ from trusted corpus"
        )
    if tuple(item.task_id for item in manifest.tasks) != corpus.task_ids:
        raise StageACalibrationCommandError(
            "trusted manifest task inventory differs from trusted corpus"
        )
    if manifest.split.to_manifest_dict() != corpus.split.to_manifest_dict():
        raise StageACalibrationCommandError(
            "trusted manifest split differs from trusted corpus"
        )


@dataclass(frozen=True, slots=True)
class StageATrustedCorpus:
    """A corpus/manifest pair plus an explicit authentication receipt.

    Construct this with :meth:`from_official_release` for the normal command
    path.  :meth:`from_trusted_objects` is an intentionally conspicuous escape
    hatch for an embedding that has already authenticated the objects at an
    outer boundary (and for small synthetic tests).
    """

    corpus: ShapeBongardCorpus
    full_manifest: CorpusManifest
    authentication_mode: Literal[
        "official-release-archive-and-corpus/v1",
        "caller-attested-trusted-objects/v1",
    ]
    trust_authority: str
    release_descriptor_digest: str | None = None
    archive_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.corpus, ShapeBongardCorpus):
            raise TypeError("corpus must be ShapeBongardCorpus")
        if not isinstance(self.full_manifest, CorpusManifest):
            raise TypeError("full_manifest must be CorpusManifest")
        _address(self.full_manifest.digest, "trusted full manifest digest")
        _bounded_text(self.trust_authority, "trust authority")
        _assert_manifest_matches_corpus_metadata(self.corpus, self.full_manifest)
        if self.authentication_mode == "official-release-archive-and-corpus/v1":
            _address(
                self.release_descriptor_digest,
                "official release descriptor digest",
            )
            _address(self.archive_sha256, "official archive digest")
        elif self.authentication_mode == "caller-attested-trusted-objects/v1":
            if self.release_descriptor_digest is not None or self.archive_sha256 is not None:
                raise StageACalibrationCommandError(
                    "caller-attested inputs cannot imply official archive verification"
                )
        else:  # pragma: no cover - Literal is not a runtime boundary.
            raise StageACalibrationCommandError(
                "unsupported Stage-A input authentication mode"
            )

    @classmethod
    def from_official_release(
        cls,
        *,
        corpus: ShapeBongardCorpus,
        release: OfficialReleaseDescriptor,
        archive_path: str | Path,
        supplied_manifest: CorpusManifest | None = None,
    ) -> "StageATrustedCorpus":
        """Hash-check the archive and freshly rebuild the complete corpus manifest."""

        if not isinstance(release, OfficialReleaseDescriptor):
            raise TypeError("release must be OfficialReleaseDescriptor")
        release.verify_archive(archive_path)
        verified_manifest = release.verify_corpus(
            corpus,
            manifest=supplied_manifest,
        )
        return cls(
            corpus=corpus,
            full_manifest=verified_manifest,
            authentication_mode="official-release-archive-and-corpus/v1",
            trust_authority=release.release_id,
            release_descriptor_digest=release.digest,
            archive_sha256=release.archive_sha256,
        )

    @classmethod
    def from_trusted_objects(
        cls,
        *,
        corpus: ShapeBongardCorpus,
        full_manifest: CorpusManifest,
        trust_authority: str,
    ) -> "StageATrustedCorpus":
        """Accept objects authenticated by a named outer boundary.

        This method does not hash all corpus pixels.  The method name, mode,
        and required authority ensure that this cannot be confused with the
        repository's official-release verifier.
        """

        return cls(
            corpus=corpus,
            full_manifest=full_manifest,
            authentication_mode="caller-attested-trusted-objects/v1",
            trust_authority=trust_authority,
        )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": STAGE_A_INPUT_AUTHENTICATION_SCHEMA,
            "authentication_mode": self.authentication_mode,
            "trust_authority": self.trust_authority,
            "full_manifest_digest": self.full_manifest.digest,
            "release_descriptor_digest": self.release_descriptor_digest,
            "archive_sha256": self.archive_sha256,
        }

    @property
    def digest(self) -> str:
        return "sha256:" + canonical_digest(self.to_data())

@dataclass(frozen=True, slots=True)
class StageACalibrationCommandConfig:
    """Frozen Stage-A protocol, selection, and execution parameters."""

    expected_codex_launcher_digest: str
    expected_exposure_ledger_digest: str
    design_mode: str
    selection_seed: str
    selection_seed_provenance: str
    candidate_count: int = 48
    semantic_cohort: str = "drill"
    families: tuple[str, ...] = ("bd", "hd")
    score_bin_edges: tuple[float, ...] = (0.0, 0.75, 1.0)
    affirmative_boundary: float = 0.5
    confidence_level: float = 0.90
    minimum_clusters_per_bin: int = 12
    proposer_model_id: str = "gpt-5.6-sol"
    proposer_reasoning_effort: str = "medium"
    scorer_model_id: str = "gpt-5.6-sol"
    scorer_reasoning_effort: str = "medium"
    proposer_minutes: int = 15
    scorer_minutes: int = 10
    proposer_max_workers: int = 4
    scorer_max_workers: int = 4
    verifier_id: str = "canonical-bongard-verifier"
    executable: str = "codex"
    label_nonce_root: str | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        _hex(
            self.expected_codex_launcher_digest,
            "externally supplied Codex launcher digest",
        )
        _address(
            self.expected_exposure_ledger_digest,
            "expected exposure ledger digest",
        )
        if self.design_mode != DESCRIPTIVE_STAGE_A_DESIGN:
            raise StageACalibrationCommandError(
                "only the explicit descriptive/exploratory Stage-A design is "
                "currently implemented; no inferential design is authorized"
            )
        _hex(
            self.selection_seed,
            "externally supplied 256-bit selection seed",
        )
        _bounded_text(
            self.selection_seed_provenance,
            "selection seed provenance",
        )
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or self.candidate_count < 1
        ):
            raise StageACalibrationCommandError(
                "candidate_count must be a positive integer"
            )
        if self.semantic_cohort not in {"drill", "dev"}:
            raise StageACalibrationCommandError(
                "semantic_cohort must be drill or dev"
            )
        if (
            not isinstance(self.families, tuple)
            or not self.families
            or len(set(self.families)) != len(self.families)
            or any(item not in {"bd", "hd"} for item in self.families)
        ):
            raise StageACalibrationCommandError(
                "families must be a non-empty duplicate-free tuple drawn from bd/hd"
            )
        if (
            not isinstance(self.score_bin_edges, tuple)
            or len(self.score_bin_edges) < 2
            or any(isinstance(value, bool) for value in self.score_bin_edges)
            or any(
                not isinstance(value, (int, float))
                for value in self.score_bin_edges
            )
            or tuple(sorted(self.score_bin_edges)) != self.score_bin_edges
            or len(set(self.score_bin_edges)) != len(self.score_bin_edges)
            or self.score_bin_edges[0] != 0.0
            or self.score_bin_edges[-1] != 1.0
        ):
            raise StageACalibrationCommandError(
                "score_bin_edges must increase strictly from 0.0 to 1.0"
            )
        if (
            isinstance(self.affirmative_boundary, bool)
            or not isinstance(self.affirmative_boundary, (int, float))
            or not 0.0 <= self.affirmative_boundary <= 1.0
        ):
            raise StageACalibrationCommandError(
                "affirmative_boundary must lie in [0, 1]"
            )
        if (
            isinstance(self.confidence_level, bool)
            or not isinstance(self.confidence_level, (int, float))
            or not 0.0 < self.confidence_level < 1.0
        ):
            raise StageACalibrationCommandError(
                "confidence_level must lie strictly between zero and one"
            )
        if (
            isinstance(self.minimum_clusters_per_bin, bool)
            or not isinstance(self.minimum_clusters_per_bin, int)
            or self.minimum_clusters_per_bin < 2
        ):
            raise StageACalibrationCommandError(
                "minimum_clusters_per_bin must be at least two"
            )
        for label, value in (
            ("proposer model id", self.proposer_model_id),
            ("proposer reasoning effort", self.proposer_reasoning_effort),
            ("scorer model id", self.scorer_model_id),
            ("scorer reasoning effort", self.scorer_reasoning_effort),
            ("verifier id", self.verifier_id),
            ("executable", self.executable),
        ):
            _bounded_text(value, label)
        for label, value, maximum in (
            ("proposer_minutes", self.proposer_minutes, 120),
            ("scorer_minutes", self.scorer_minutes, 120),
            ("proposer_max_workers", self.proposer_max_workers, 256),
            ("scorer_max_workers", self.scorer_max_workers, 256),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= maximum
            ):
                raise StageACalibrationCommandError(
                    f"{label} must lie in [1, {maximum}]"
                )
        if self.label_nonce_root is not None:
            _hex(self.label_nonce_root, "label nonce root")
        if not isinstance(self.verbose, bool):
            raise StageACalibrationCommandError("verbose must be Boolean")

    def build_protocol(self) -> SoftScorerProtocol:
        return build_prospective_soft_scorer_protocol(
            proposer_model_id=self.proposer_model_id,
            proposer_reasoning_effort=self.proposer_reasoning_effort,
            scorer_model_id=self.scorer_model_id,
            scorer_reasoning_effort=self.scorer_reasoning_effort,
            score_bin_edges=self.score_bin_edges,
            affirmative_boundary=self.affirmative_boundary,
            confidence_level=self.confidence_level,
            minimum_clusters_per_bin=self.minimum_clusters_per_bin,
        )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": STAGE_A_COMMAND_CONFIG_SCHEMA,
            "reference_execution": "python-only/v1",
            "stage_a_scope": STAGE_A_SCOPE,
            "design_mode": self.design_mode,
            "selection_seed": self.selection_seed,
            "selection_seed_provenance": self.selection_seed_provenance,
            "expected_codex_launcher_digest": self.expected_codex_launcher_digest,
            "expected_exposure_ledger_digest": self.expected_exposure_ledger_digest,
            "candidate_count": self.candidate_count,
            "semantic_cohort": self.semantic_cohort,
            "families": list(self.families),
            "score_bin_edges": list(self.score_bin_edges),
            "affirmative_boundary": self.affirmative_boundary,
            "confidence_level": self.confidence_level,
            "minimum_clusters_per_bin": self.minimum_clusters_per_bin,
            "proposer_model_id": self.proposer_model_id,
            "proposer_reasoning_effort": self.proposer_reasoning_effort,
            "scorer_model_id": self.scorer_model_id,
            "scorer_reasoning_effort": self.scorer_reasoning_effort,
            "proposer_minutes": self.proposer_minutes,
            "scorer_minutes": self.scorer_minutes,
            "proposer_max_workers": self.proposer_max_workers,
            "scorer_max_workers": self.scorer_max_workers,
            "verifier_id": self.verifier_id,
            "executable": self.executable,
            "label_nonce_root": self.label_nonce_root,
            "verbose": self.verbose,
        }

    @property
    def digest(self) -> str:
        return "sha256:" + canonical_digest(self.to_data())

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
    ) -> "StageACalibrationCommandConfig":
        expected = {
            "schema",
            "reference_execution",
            "stage_a_scope",
            "design_mode",
            "selection_seed",
            "selection_seed_provenance",
            "expected_codex_launcher_digest",
            "expected_exposure_ledger_digest",
            "candidate_count",
            "semantic_cohort",
            "families",
            "score_bin_edges",
            "affirmative_boundary",
            "confidence_level",
            "minimum_clusters_per_bin",
            "proposer_model_id",
            "proposer_reasoning_effort",
            "scorer_model_id",
            "scorer_reasoning_effort",
            "proposer_minutes",
            "scorer_minutes",
            "proposer_max_workers",
            "scorer_max_workers",
            "verifier_id",
            "executable",
            "label_nonce_root",
            "verbose",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise StageACalibrationCommandError(
                "Stage-A command config fields differ"
            )
        if (
            value["schema"] != STAGE_A_COMMAND_CONFIG_SCHEMA
            or value["reference_execution"] != "python-only/v1"
            or value["stage_a_scope"] != STAGE_A_SCOPE
            or not isinstance(value["families"], list)
            or not isinstance(value["score_bin_edges"], list)
        ):
            raise StageACalibrationCommandError(
                "Stage-A command config authority or collection types changed"
            )
        result = cls(
            expected_codex_launcher_digest=value[
                "expected_codex_launcher_digest"
            ],
            expected_exposure_ledger_digest=value[
                "expected_exposure_ledger_digest"
            ],
            design_mode=value["design_mode"],
            selection_seed=value["selection_seed"],
            selection_seed_provenance=value["selection_seed_provenance"],
            candidate_count=value["candidate_count"],
            semantic_cohort=value["semantic_cohort"],
            families=tuple(value["families"]),
            score_bin_edges=tuple(value["score_bin_edges"]),
            affirmative_boundary=value["affirmative_boundary"],
            confidence_level=value["confidence_level"],
            minimum_clusters_per_bin=value["minimum_clusters_per_bin"],
            proposer_model_id=value["proposer_model_id"],
            proposer_reasoning_effort=value["proposer_reasoning_effort"],
            scorer_model_id=value["scorer_model_id"],
            scorer_reasoning_effort=value["scorer_reasoning_effort"],
            proposer_minutes=value["proposer_minutes"],
            scorer_minutes=value["scorer_minutes"],
            proposer_max_workers=value["proposer_max_workers"],
            scorer_max_workers=value["scorer_max_workers"],
            verifier_id=value["verifier_id"],
            executable=value["executable"],
            label_nonce_root=value["label_nonce_root"],
            verbose=value["verbose"],
        )
        if result.to_data() != dict(value):
            raise StageACalibrationCommandError(
                "Stage-A command config is not canonical"
            )
        return result


@dataclass(frozen=True, slots=True)
class StageAOperationalFailureArtifact:
    """Typed terminal record for every non-campaign post-precommit failure."""

    failure_phase: str
    error_type: str
    reason_digest: str
    exposure_successor: ExposureLedger
    protocol_digest: str
    command_config_digest: str
    input_authentication_digest: str
    launcher_digest: str
    cloud_policy_cache_binding: str
    source_dependencies: StageASourceDependencyIdentity
    source_dependency_state: Literal["unchanged", "mutated", "unreadable"]
    observed_source_dependencies: StageASourceDependencyIdentity | None = None
    source_observation_error_digest: str | None = None

    def __post_init__(self) -> None:
        _bounded_text(self.failure_phase, "operational failure phase")
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,255}", self.error_type) is None:
            raise StageACalibrationCommandError(
                "operational failure type is invalid"
            )
        _hex(self.reason_digest, "operational failure reason digest")
        if not isinstance(self.exposure_successor, ExposureLedger):
            raise TypeError("exposure_successor must be ExposureLedger")
        _hex(self.protocol_digest, "operational failure protocol digest")
        _address(self.command_config_digest, "operational failure config digest")
        _address(
            self.input_authentication_digest,
            "operational failure input authentication digest",
        )
        _hex(self.launcher_digest, "operational failure launcher digest")
        if self.cloud_policy_cache_binding != "absent":
            _address(
                self.cloud_policy_cache_binding,
                "operational failure cache binding",
            )
        if not isinstance(
            self.source_dependencies,
            StageASourceDependencyIdentity,
        ):
            raise TypeError(
                "source_dependencies must be StageASourceDependencyIdentity"
            )
        if self.source_dependency_state == "unchanged":
            if (
                self.observed_source_dependencies != self.source_dependencies
                or self.source_observation_error_digest is not None
            ):
                raise StageACalibrationCommandError(
                    "unchanged source failure must retain the frozen identity"
                )
        elif self.source_dependency_state == "mutated":
            if (
                not isinstance(
                    self.observed_source_dependencies,
                    StageASourceDependencyIdentity,
                )
                or self.observed_source_dependencies == self.source_dependencies
                or self.source_observation_error_digest is not None
            ):
                raise StageACalibrationCommandError(
                    "mutated source failure lacks a distinct observed identity"
                )
        elif self.source_dependency_state == "unreadable":
            if self.observed_source_dependencies is not None:
                raise StageACalibrationCommandError(
                    "unreadable source failure cannot retain an observed identity"
                )
            _hex(
                self.source_observation_error_digest,
                "source observation error digest",
            )
        else:
            raise StageACalibrationCommandError(
                "unknown operational source dependency state"
            )

    def content_data(self) -> dict[str, object]:
        return {
            "schema": STAGE_A_OPERATIONAL_FAILURE_SCHEMA,
            "terminal_state": "post-precommit-operational-failure/v1",
            "failure_phase": self.failure_phase,
            "failure": {
                "error_type": self.error_type,
                "reason_digest": self.reason_digest,
            },
            "exposure_successor": self.exposure_successor.to_dict(),
            "exposure_successor_digest": self.exposure_successor.digest,
            "protocol_digest": self.protocol_digest,
            "command_config_digest": self.command_config_digest,
            "input_authentication_digest": self.input_authentication_digest,
            "launcher_digest": self.launcher_digest,
            "cloud_policy_cache_binding": self.cloud_policy_cache_binding,
            "source_dependencies": self.source_dependencies.to_data(),
            "source_dependency_digest": self.source_dependencies.digest,
            "source_dependency_state": self.source_dependency_state,
            "observed_source_dependencies": (
                None
                if self.observed_source_dependencies is None
                else self.observed_source_dependencies.to_data()
            ),
            "observed_source_dependency_digest": (
                None
                if self.observed_source_dependencies is None
                else self.observed_source_dependencies.digest
            ),
            "source_observation_error_digest": (
                self.source_observation_error_digest
            ),
            "label_state": "withheld",
            "campaign_result_state": "absent",
            "fit_authorized": False,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.content_data())

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "failure_digest": self.digest}


@dataclass(frozen=True, slots=True)
class StageAExecutionOutcome:
    """Immutable, canonical terminal bytes plus their exposure successor."""

    status: Literal["succeeded", "failed"]
    terminal_schema: str
    internal_digest: str
    terminal_payload: bytes
    exposure_successor: ExposureLedger
    protocol_digest: str
    command_config_payload: bytes
    command_config_digest: str
    input_authentication_digest: str
    launcher_version: str
    launcher_digest: str
    cloud_policy_cache_snapshot: CloudPolicyCacheSnapshot
    cloud_policy_cache_binding: str
    source_dependencies: StageASourceDependencyIdentity
    source_dependency_root: Path = field(repr=False, compare=False)
    cold_verified: bool
    stage_a_scope: str = STAGE_A_SCOPE

    def __post_init__(self) -> None:
        if self.status not in {"succeeded", "failed"}:
            raise StageACalibrationCommandError("invalid Stage-A terminal status")
        _bounded_text(self.terminal_schema, "terminal schema")
        _hex(self.internal_digest, "terminal internal digest")
        if not isinstance(self.terminal_payload, bytes):
            raise TypeError("terminal_payload must be exact bytes")
        try:
            decoded = json.loads(self.terminal_payload)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise StageACalibrationCommandError(
                "terminal payload is not UTF-8 JSON"
            ) from exc
        if (
            not isinstance(decoded, Mapping)
            or canonical_json(decoded) + b"\n" != self.terminal_payload
            or decoded.get("schema") != self.terminal_schema
        ):
            raise StageACalibrationCommandError(
                "terminal payload is not exact canonical JSON plus newline"
            )
        digest_field = "campaign_digest" if self.status == "succeeded" else "failure_digest"
        if decoded.get(digest_field) != self.internal_digest:
            raise StageACalibrationCommandError(
                "terminal payload internal digest field differs"
            )
        content = {key: value for key, value in decoded.items() if key != digest_field}
        if canonical_digest(content) != self.internal_digest:
            raise StageACalibrationCommandError(
                "terminal payload internal digest does not reproduce"
            )
        if not isinstance(self.exposure_successor, ExposureLedger):
            raise TypeError("exposure_successor must be ExposureLedger")
        _hex(self.protocol_digest, "soft protocol digest")
        if not isinstance(self.command_config_payload, bytes):
            raise TypeError("command_config_payload must be exact bytes")
        try:
            command_config = json.loads(self.command_config_payload)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise StageACalibrationCommandError(
                "frozen command config payload is not UTF-8 JSON"
            ) from exc
        if (
            not isinstance(command_config, Mapping)
            or canonical_json(command_config) != self.command_config_payload
            or command_config.get("schema") != STAGE_A_COMMAND_CONFIG_SCHEMA
            or "sha256:" + canonical_digest(command_config)
            != self.command_config_digest
        ):
            raise StageACalibrationCommandError(
                "frozen command config payload or digest does not reproduce"
            )
        _address(self.command_config_digest, "command config digest")
        _address(self.input_authentication_digest, "input authentication digest")
        _bounded_text(self.launcher_version, "Codex launcher version")
        _hex(self.launcher_digest, "Codex launcher digest")
        if not isinstance(
            self.cloud_policy_cache_snapshot,
            CloudPolicyCacheSnapshot,
        ):
            raise TypeError(
                "cloud_policy_cache_snapshot must retain the exact typed preimage"
            )
        if (
            self.cloud_policy_cache_binding
            != self.cloud_policy_cache_snapshot.binding
        ):
            raise StageACalibrationCommandError(
                "cloud policy cache binding differs from retained preimage"
            )
        if self.cloud_policy_cache_binding != "absent":
            _address(
                self.cloud_policy_cache_binding,
                "cloud policy cache binding",
            )
        if not isinstance(
            self.source_dependencies,
            StageASourceDependencyIdentity,
        ):
            raise TypeError(
                "source_dependencies must be StageASourceDependencyIdentity"
            )
        object.__setattr__(
            self,
            "source_dependency_root",
            Path(self.source_dependency_root).expanduser().resolve(),
        )
        if self.cold_verified is not (self.status == "succeeded"):
            raise StageACalibrationCommandError(
                "only a successful campaign can be marked cold-verified"
            )
        if self.stage_a_scope != STAGE_A_SCOPE:
            raise StageACalibrationCommandError("Stage-A scope changed")

    def terminal_data(self) -> Mapping[str, Any]:
        decoded = json.loads(self.terminal_payload)
        assert isinstance(decoded, Mapping)
        return decoded


@dataclass(frozen=True, slots=True)
class StageAPersistenceConfig:
    artifact_directory: Path
    exposure_directory: Path
    cache_snapshot_directory: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_directory", Path(self.artifact_directory))
        object.__setattr__(self, "exposure_directory", Path(self.exposure_directory))
        object.__setattr__(
            self,
            "cache_snapshot_directory",
            Path(self.cache_snapshot_directory),
        )
        if (
            self.cache_snapshot_directory.expanduser().resolve()
            == self.artifact_directory.expanduser().resolve()
        ):
            raise StageACalibrationCommandError(
                "private cache preimages require a directory separate from "
                "scientific artifacts"
            )


@dataclass(frozen=True, slots=True)
class StageACommandResult:
    status: Literal["succeeded", "failed"]
    terminal_schema: str
    internal_digest: str
    artifact_path: Path
    artifact_file_sha256: str
    exposure_ledger_path: Path
    exposure_ledger_digest: str
    exposure_ledger_file_sha256: str
    protocol_digest: str
    command_config_payload: bytes
    command_config_digest: str
    input_authentication_digest: str
    launcher_version: str
    launcher_digest: str
    cloud_policy_cache_snapshot_path: Path
    cloud_policy_cache_snapshot_file_sha256: str
    cloud_policy_cache_snapshot_byte_count: int
    cloud_policy_cache_binding: str
    source_dependencies: StageASourceDependencyIdentity
    command_receipt_path: Path
    command_receipt_digest: str
    command_receipt_file_sha256: str
    cold_verified: bool
    stage_a_scope: str = STAGE_A_SCOPE

    @property
    def command_config(self) -> Mapping[str, Any]:
        decoded = json.loads(self.command_config_payload)
        assert isinstance(decoded, Mapping)
        return decoded

    def to_data(self) -> dict[str, object]:
        return {
            "status": self.status,
            "terminal_schema": self.terminal_schema,
            "internal_digest": self.internal_digest,
            "artifact_path": str(self.artifact_path),
            "artifact_file_sha256": self.artifact_file_sha256,
            "exposure_ledger_path": str(self.exposure_ledger_path),
            "exposure_ledger_digest": self.exposure_ledger_digest,
            "exposure_ledger_file_sha256": self.exposure_ledger_file_sha256,
            "protocol_digest": self.protocol_digest,
            "command_config": dict(self.command_config),
            "command_config_digest": self.command_config_digest,
            "input_authentication_digest": self.input_authentication_digest,
            "launcher_version": self.launcher_version,
            "launcher_digest": self.launcher_digest,
            "cloud_policy_cache_snapshot_path": str(
                self.cloud_policy_cache_snapshot_path
            ),
            "cloud_policy_cache_snapshot_file_sha256": (
                self.cloud_policy_cache_snapshot_file_sha256
            ),
            "cloud_policy_cache_snapshot_byte_count": (
                self.cloud_policy_cache_snapshot_byte_count
            ),
            "cloud_policy_cache_binding": self.cloud_policy_cache_binding,
            "source_dependencies": self.source_dependencies.to_data(),
            "source_dependency_digest": self.source_dependencies.digest,
            "command_receipt_path": str(self.command_receipt_path),
            "command_receipt_digest": self.command_receipt_digest,
            "command_receipt_file_sha256": self.command_receipt_file_sha256,
            "cold_verified": self.cold_verified,
            "stage_a_scope": self.stage_a_scope,
            "python_predicate_authoritative": True,
            "optional_checker_may_affect_result": False,
        }


_COMMAND_RECEIPT_FIELDS_V1 = {
    "schema",
    "status",
    "stage_a_scope",
    "terminal_schema",
    "terminal_internal_digest",
    "terminal_artifact_path",
    "terminal_artifact_file_sha256",
    "exposure_ledger_path",
    "exposure_ledger_digest",
    "exposure_ledger_file_sha256",
    "protocol_digest",
    "command_config",
    "command_config_digest",
    "input_authentication_digest",
    "launcher_version",
    "launcher_digest",
    "cloud_policy_cache_binding",
    "cloud_policy_cache_snapshot_path",
    "cloud_policy_cache_snapshot_file_sha256",
    "cloud_policy_cache_snapshot_byte_count",
    "cloud_policy_cache_snapshot_bytes_embedded",
    "cold_verified",
    "python_predicate_authoritative",
    "optional_checker_may_affect_result",
    "command_receipt_digest",
}
_COMMAND_RECEIPT_FIELDS = _COMMAND_RECEIPT_FIELDS_V1 | {
    "source_dependencies",
    "source_dependency_digest",
}


def _absolute_path(value: object, label: str) -> Path:
    text = _bounded_text(value, label, maximum=16_384)
    result = Path(text)
    if not result.is_absolute():
        raise StageACalibrationCommandError(f"{label} must be absolute")
    return result


@dataclass(frozen=True, slots=True)
class StageACommandReceipt:
    """Strictly decoded command receipt used by a separate Stage-B process."""

    receipt_payload: bytes
    receipt_digest: str
    status: Literal["succeeded", "failed"]
    command_config: StageACalibrationCommandConfig
    terminal_internal_digest: str
    terminal_artifact_path: Path
    terminal_artifact_file_sha256: str
    exposure_ledger_path: Path
    exposure_ledger_digest: str
    exposure_ledger_file_sha256: str
    protocol_digest: str
    input_authentication_digest: str
    launcher_version: str
    launcher_digest: str
    cloud_policy_cache_binding: str
    cloud_policy_cache_snapshot_path: Path
    cloud_policy_cache_snapshot_file_sha256: str
    cloud_policy_cache_snapshot_byte_count: int
    source_dependencies: StageASourceDependencyIdentity | None
    cold_verified: bool

    @property
    def command_config_digest(self) -> str:
        return self.command_config.digest

    def to_data(self) -> Mapping[str, Any]:
        decoded = json.loads(self.receipt_payload)
        assert isinstance(decoded, Mapping)
        return decoded

    def load_cache_snapshot(self) -> CloudPolicyCacheSnapshot:
        snapshot = load_stage_a_cache_snapshot(
            self.cloud_policy_cache_snapshot_path,
            expected_binding=self.cloud_policy_cache_binding,
            expected_file_sha256=(
                self.cloud_policy_cache_snapshot_file_sha256
            ),
        )
        actual_size = 0 if snapshot.data is None else len(snapshot.data)
        if actual_size != self.cloud_policy_cache_snapshot_byte_count:
            raise StageACalibrationCommandError(
                "private cache snapshot byte count differs from command receipt"
            )
        return snapshot

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        *,
        expected_receipt_digest: str,
    ) -> "StageACommandReceipt":
        if not isinstance(payload, bytes):
            raise TypeError("command receipt payload must be bytes")
        try:
            decoded = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise StageACalibrationCommandError(
                "Stage-A command receipt is not UTF-8 JSON"
            ) from exc
        if not isinstance(decoded, Mapping) or canonical_json(decoded) + b"\n" != payload:
            raise StageACalibrationCommandError(
                "Stage-A command receipt bytes or fields are not canonical"
            )
        receipt_schema = decoded.get("schema")
        expected_fields = (
            _COMMAND_RECEIPT_FIELDS
            if receipt_schema == STAGE_A_COMMAND_RECEIPT_SCHEMA
            else (
                _COMMAND_RECEIPT_FIELDS_V1
                if receipt_schema == STAGE_A_COMMAND_RECEIPT_SCHEMA_V1
                else None
            )
        )
        if expected_fields is None:
            raise StageACalibrationCommandError(
                "unsupported Stage-A command receipt schema"
            )
        if set(decoded) != expected_fields:
            raise StageACalibrationCommandError(
                "Stage-A command receipt bytes or fields are not canonical"
            )
        archived_digest = _hex(
            decoded["command_receipt_digest"],
            "command receipt digest",
        )
        expected_digest = _hex(
            expected_receipt_digest,
            "externally expected command receipt digest",
        )
        content = {
            key: value
            for key, value in decoded.items()
            if key != "command_receipt_digest"
        }
        if (
            canonical_digest(content) != archived_digest
            or archived_digest != expected_digest
        ):
            raise StageACalibrationCommandError(
                "Stage-A command receipt digest differs"
            )
        status = decoded["status"]
        if status not in {"succeeded", "failed"}:
            raise StageACalibrationCommandError(
                "invalid Stage-A command receipt status"
            )
        cold_verified = decoded["cold_verified"]
        if (
            not isinstance(cold_verified, bool)
            or cold_verified is not (status == "succeeded")
            or decoded["stage_a_scope"] != STAGE_A_SCOPE
            or decoded["cloud_policy_cache_snapshot_bytes_embedded"] is not False
            or decoded["python_predicate_authoritative"] is not True
            or decoded["optional_checker_may_affect_result"] is not False
        ):
            raise StageACalibrationCommandError(
                "Stage-A command receipt authority or scope changed"
            )
        _bounded_text(decoded["terminal_schema"], "terminal schema")
        raw_config = decoded["command_config"]
        if not isinstance(raw_config, Mapping):
            raise StageACalibrationCommandError(
                "Stage-A command receipt config must be an object"
            )
        command_config = StageACalibrationCommandConfig.from_data(raw_config)
        config_digest = _address(
            decoded["command_config_digest"],
            "command config digest",
        )
        if command_config.digest != config_digest:
            raise StageACalibrationCommandError(
                "Stage-A command receipt config digest differs"
            )
        launcher_digest = _hex(decoded["launcher_digest"], "launcher digest")
        if launcher_digest != command_config.expected_codex_launcher_digest:
            raise StageACalibrationCommandError(
                "Stage-A command receipt launcher differs from frozen config"
            )
        cache_binding = decoded["cloud_policy_cache_binding"]
        if cache_binding != "absent":
            _address(cache_binding, "cloud policy cache binding")
        cache_file = _address(
            decoded["cloud_policy_cache_snapshot_file_sha256"],
            "cloud policy cache snapshot file digest",
        )
        cache_count = decoded["cloud_policy_cache_snapshot_byte_count"]
        if (
            isinstance(cache_count, bool)
            or not isinstance(cache_count, int)
            or cache_count < 0
        ):
            raise StageACalibrationCommandError(
                "cloud policy cache snapshot byte count is invalid"
            )
        empty_digest = "sha256:" + hashlib.sha256(b"").hexdigest()
        if cache_binding == "absent":
            if cache_count != 0 or cache_file != empty_digest:
                raise StageACalibrationCommandError(
                    "absent cache receipt does not bind the explicit empty handoff"
                )
        elif cache_binding != cache_file or cache_count == 0:
            raise StageACalibrationCommandError(
                "present cache receipt file identity differs from its binding"
            )
        source_dependencies = None
        if receipt_schema == STAGE_A_COMMAND_RECEIPT_SCHEMA:
            raw_sources = decoded["source_dependencies"]
            if not isinstance(raw_sources, Mapping):
                raise StageACalibrationCommandError(
                    "command receipt source dependencies must be an object"
                )
            source_dependencies = StageASourceDependencyIdentity.from_data(
                raw_sources
            )
            if source_dependencies.digest != _hex(
                decoded["source_dependency_digest"],
                "command receipt source dependency digest",
            ):
                raise StageACalibrationCommandError(
                    "command receipt source dependency digest differs"
                )
        return cls(
            receipt_payload=payload,
            receipt_digest=archived_digest,
            status=status,
            command_config=command_config,
            terminal_internal_digest=_hex(
                decoded["terminal_internal_digest"],
                "terminal internal digest",
            ),
            terminal_artifact_path=_absolute_path(
                decoded["terminal_artifact_path"],
                "terminal artifact path",
            ),
            terminal_artifact_file_sha256=_address(
                decoded["terminal_artifact_file_sha256"],
                "terminal artifact file digest",
            ),
            exposure_ledger_path=_absolute_path(
                decoded["exposure_ledger_path"],
                "exposure ledger path",
            ),
            exposure_ledger_digest=_address(
                decoded["exposure_ledger_digest"],
                "exposure ledger digest",
            ),
            exposure_ledger_file_sha256=_address(
                decoded["exposure_ledger_file_sha256"],
                "exposure ledger file digest",
            ),
            protocol_digest=_hex(decoded["protocol_digest"], "protocol digest"),
            input_authentication_digest=_address(
                decoded["input_authentication_digest"],
                "input authentication digest",
            ),
            launcher_version=_bounded_text(
                decoded["launcher_version"],
                "launcher version",
            ),
            launcher_digest=launcher_digest,
            cloud_policy_cache_binding=cache_binding,
            cloud_policy_cache_snapshot_path=_absolute_path(
                decoded["cloud_policy_cache_snapshot_path"],
                "cloud policy cache snapshot path",
            ),
            cloud_policy_cache_snapshot_file_sha256=cache_file,
            cloud_policy_cache_snapshot_byte_count=cache_count,
            source_dependencies=source_dependencies,
            cold_verified=cold_verified,
        )


def load_stage_a_command_receipt(
    path: str | Path,
    expected_receipt_digest: str,
) -> StageACommandReceipt:
    """Canonical-byte load with an externally supplied receipt commitment."""

    source = Path(path).expanduser().absolute()
    try:
        before = source.stat()
        payload = source.read_bytes()
        after = source.stat()
    except OSError as exc:
        raise StageACalibrationCommandError(
            f"cannot read Stage-A command receipt {source}"
        ) from exc
    if (
        not source.is_file()
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise StageACalibrationCommandError(
            "Stage-A command receipt changed while being read"
        )
    receipt = StageACommandReceipt.from_bytes(
        payload,
        expected_receipt_digest=expected_receipt_digest,
    )
    # Cache metadata is only meaningful relative to the exact private file.
    # Validate that causal parent as part of the strict public loader so a
    # separate Stage-B process never receives an unchecked handoff path.
    receipt.load_cache_snapshot()
    return receipt


def _proposal_archive_from_success(campaign: Any) -> Any:
    return campaign.score_batch.commitment_batch.proposal_archive


def _proposal_archive_from_terminal(error: BaseException) -> Any:
    if isinstance(
        error,
        (
            SemanticCalibrationCampaignProposalPhaseFailed,
            SemanticCalibrationCampaignNoSoftClaims,
        ),
    ):
        return error.proposal_archive
    if isinstance(
        error,
        (
            SemanticCalibrationCampaignScoringFailed,
            SemanticCalibrationCampaignFitFailed,
        ),
    ):
        return error.score_batch.commitment_batch.proposal_archive
    raise TypeError("error is not a canonical Stage-A terminal exception")


def _assert_archive_matches_command(
    archive: Any,
    *,
    trusted: StageATrustedCorpus,
    ledger: ExposureLedger,
    config: StageACalibrationCommandConfig,
    protocol: SoftScorerProtocol,
    snapshot: CloudPolicyCacheSnapshot,
) -> ExposureLedger:
    expected = {
        "selection_algorithm": CAMPAIGN_SELECTION_ALGORITHM,
        "protocol": protocol,
        "selection_seed": config.selection_seed,
        "candidate_count": config.candidate_count,
        "families": config.families,
        "semantic_cohort": config.semantic_cohort,
        "source_corpus_manifest_digest": trusted.full_manifest.digest,
        "exposure_predecessor": ledger,
    }
    for name, wanted in expected.items():
        if getattr(archive, name, object()) != wanted:
            raise StageACalibrationCommandError(
                f"campaign archive {name} differs from command commitment"
            )
    execution = getattr(archive, "execution_config", None)
    execution_expected = {
        "proposer_minutes": config.proposer_minutes,
        "scorer_minutes": config.scorer_minutes,
        "proposer_max_workers": config.proposer_max_workers,
        "scorer_max_workers": config.scorer_max_workers,
        "executable": config.executable,
        "expected_codex_launcher_digest": config.expected_codex_launcher_digest,
        "cloud_policy_cache_binding": snapshot.binding,
    }
    for name, wanted in execution_expected.items():
        if getattr(execution, name, object()) != wanted:
            raise StageACalibrationCommandError(
                f"campaign execution {name} differs from command commitment"
            )
    successor = getattr(archive, "exposure_successor", None)
    _assert_exact_exposure_successor(
        successor,
        trusted=trusted,
        ledger=ledger,
        config=config,
    )
    return successor


def _assert_exact_exposure_successor(
    successor: object,
    *,
    trusted: StageATrustedCorpus,
    ledger: ExposureLedger,
    config: StageACalibrationCommandConfig,
) -> None:
    if not isinstance(successor, ExposureLedger):
        raise StageACalibrationCommandError(
            "campaign has no typed exposure successor"
        )
    if (
        successor.corpus_digest != trusted.full_manifest.digest
        or successor.events[: len(ledger.events)] != ledger.events
        or len(successor.events) - len(ledger.events) != config.candidate_count
    ):
        raise StageACalibrationCommandError(
            "campaign exposure successor is not the exact candidate disclosure transition"
        )


def _canonical_terminal(
    data: Mapping[str, Any],
    *,
    status: Literal["succeeded", "failed"],
    expected_internal_digest: str,
) -> tuple[str, bytes]:
    exact = json.loads(canonical_json(dict(data)))
    if not isinstance(exact, dict):  # pragma: no cover - Mapping guarantees it.
        raise StageACalibrationCommandError("terminal artifact must be an object")
    schema = _bounded_text(exact.get("schema"), "terminal artifact schema")
    digest_field = "campaign_digest" if status == "succeeded" else "failure_digest"
    archived = _hex(exact.get(digest_field), f"terminal {digest_field}")
    expected = _hex(expected_internal_digest, "terminal object digest")
    if archived != expected:
        raise StageACalibrationCommandError(
            "terminal artifact digest differs from typed terminal object"
        )
    content = {key: value for key, value in exact.items() if key != digest_field}
    if canonical_digest(content) != archived:
        raise StageACalibrationCommandError(
            "terminal artifact internal digest does not reproduce"
        )
    return schema, canonical_json(exact) + b"\n"


def _bounded_operational_failure(error: BaseException) -> tuple[str, str]:
    error_type = type(error).__name__
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,255}", error_type) is None:
        error_type = "StageAOperationalError"
    reason = (str(error) or repr(error)).replace("\x00", "�").strip()
    encoded = reason.encode("utf-8", errors="replace")[:4_000]
    return error_type, hashlib.sha256(encoded).hexdigest()


def _post_precommit_failure_outcome(
    error: BaseException,
    *,
    phase: str,
    successor: ExposureLedger,
    protocol: SoftScorerProtocol,
    config: StageACalibrationCommandConfig,
    trusted: StageATrustedCorpus,
    launcher_version: str,
    launcher_digest: str,
    snapshot: CloudPolicyCacheSnapshot,
    source_guard: _StageASourceDependencyGuard,
) -> StageAExecutionOutcome:
    """Normalize any escaped post-precommit failure into canonical bytes."""

    try:
        source_guard.check("post-precommit-failure-normalization")
    except StageASourceDependencyMutationError as mutation:
        # Source drift takes precedence because it invalidates interpretation
        # of any coincident lower-level exception.
        error = mutation
        phase = mutation.phase
    mutation = source_guard.first_failure
    if mutation is None:
        source_state: Literal["unchanged", "mutated", "unreadable"] = (
            "unchanged"
        )
        observed = source_guard.expected
        observation_error_digest = None
    elif mutation.observed is None:
        source_state = "unreadable"
        observed = None
        observation_error_digest = mutation.observation_error_digest
    else:
        source_state = "mutated"
        observed = mutation.observed
        observation_error_digest = None
    error_type, reason_digest = _bounded_operational_failure(error)
    failure = StageAOperationalFailureArtifact(
        failure_phase=phase,
        error_type=error_type,
        reason_digest=reason_digest,
        exposure_successor=successor,
        protocol_digest=protocol.digest(),
        command_config_digest=config.digest,
        input_authentication_digest=trusted.digest,
        launcher_digest=launcher_digest,
        cloud_policy_cache_binding=snapshot.binding,
        source_dependencies=source_guard.expected,
        source_dependency_state=source_state,
        observed_source_dependencies=observed,
        source_observation_error_digest=observation_error_digest,
    )
    schema, payload = _canonical_terminal(
        failure.to_data(),
        status="failed",
        expected_internal_digest=failure.digest,
    )
    return StageAExecutionOutcome(
        status="failed",
        terminal_schema=schema,
        internal_digest=failure.digest,
        terminal_payload=payload,
        exposure_successor=successor,
        protocol_digest=protocol.digest(),
        command_config_payload=canonical_json(config.to_data()),
        command_config_digest=config.digest,
        input_authentication_digest=trusted.digest,
        launcher_version=launcher_version,
        launcher_digest=launcher_digest,
        cloud_policy_cache_snapshot=snapshot,
        cloud_policy_cache_binding=snapshot.binding,
        source_dependencies=source_guard.expected,
        source_dependency_root=source_guard.root,
        cold_verified=False,
    )


def execute_stage_a_calibration(
    trusted: StageATrustedCorpus,
    exposure_ledger: ExposureLedger,
    config: StageACalibrationCommandConfig,
    *,
    on_exposure_precommit: ExposurePrecommit,
    campaign_runner: CampaignRunner = run_semantic_calibration_campaign,
    campaign_verifier: CampaignVerifier = verify_semantic_campaign_against_corpus,
    launcher_fingerprinter: LauncherFingerprinter = (
        codex_cli_authenticated_fingerprint
    ),
    cache_snapshotter: CacheSnapshotter = snapshot_cloud_policy_cache,
    proposer_transport: Callable[..., Any] = run_codex_structured,
    scorer_transport: Callable[..., Any] = run_codex_named_images_structured,
    source_dependency_root: str | Path | None = None,
) -> StageAExecutionOutcome:
    """Preflight, execute, and cold-verify one Stage-A campaign.

    No campaign invocation (and therefore no selected PNG or model access)
    occurs before the actual launcher bytes match the externally supplied
    digest and one exact cloud-policy cache snapshot has been frozen.
    """

    if not isinstance(trusted, StageATrustedCorpus):
        raise TypeError("trusted must be StageATrustedCorpus")
    if not isinstance(exposure_ledger, ExposureLedger):
        raise TypeError("exposure_ledger must be ExposureLedger")
    if not isinstance(config, StageACalibrationCommandConfig):
        raise TypeError("config must be StageACalibrationCommandConfig")
    if exposure_ledger.digest != config.expected_exposure_ledger_digest:
        raise StageACalibrationCommandError(
            "exposure ledger differs from externally expected digest"
        )
    if exposure_ledger.corpus_digest != trusted.full_manifest.digest:
        raise StageACalibrationCommandError(
            "exposure ledger belongs to another corpus manifest"
        )
    if not callable(campaign_runner) or not callable(campaign_verifier):
        raise TypeError("campaign runner and verifier must be callable")
    if not callable(launcher_fingerprinter) or not callable(cache_snapshotter):
        raise TypeError("environment preflight functions must be callable")
    if not callable(on_exposure_precommit):
        raise TypeError("on_exposure_precommit must be callable")

    source_root = (
        Path(__file__).resolve().parent
        if source_dependency_root is None
        else Path(source_dependency_root).expanduser().resolve()
    )
    frozen_sources = freeze_stage_a_source_dependencies(source_root)
    source_guard = _StageASourceDependencyGuard(source_root, frozen_sources)
    protocol = config.build_protocol()
    source_guard.check("after-protocol-freeze")

    # Security boundary: this digest comes from config supplied outside the
    # command.  Never populate it from this fingerprint result.
    try:
        fingerprint = launcher_fingerprinter(
            config.executable,
            expected_launcher_digest=(
                config.expected_codex_launcher_digest
            ),
        )
    except Exception as exc:  # noqa: BLE001 - normalize the command boundary.
        raise StageACalibrationCommandError(
            "cannot fingerprint the configured Codex launcher"
        ) from exc
    if not isinstance(fingerprint, Mapping):
        raise StageACalibrationCommandError(
            "Codex launcher fingerprint is not an object"
        )
    actual_launcher = _hex(
        fingerprint.get("launcher_digest"),
        "actual Codex launcher digest",
    )
    launcher_version = _bounded_text(
        fingerprint.get("version"),
        "Codex launcher version",
    )
    if actual_launcher != config.expected_codex_launcher_digest:
        raise StageACalibrationCommandError(
            "actual Codex launcher differs from externally supplied digest"
        )
    try:
        snapshot = cache_snapshotter()
    except Exception as exc:  # noqa: BLE001 - normalize the command boundary.
        raise StageACalibrationCommandError(
            "cannot freeze the Codex cloud-policy cache"
        ) from exc
    if not isinstance(snapshot, CloudPolicyCacheSnapshot):
        raise StageACalibrationCommandError(
            "cache snapshotter returned an invalid value"
        )

    precommitted: list[ExposureLedger] = []

    def campaign_exposure_precommit(successor: ExposureLedger) -> None:
        if precommitted:
            raise StageACalibrationCommandError(
                "campaign invoked exposure precommit more than once"
            )
        _assert_exact_exposure_successor(
            successor,
            trusted=trusted,
            ledger=exposure_ledger,
            config=config,
        )
        source_guard.check("before-exposure-precommit")
        # Crossing into the caller-owned persistence hook is irrevocable for
        # accounting purposes.  Record the exact successor first so an
        # exception after the hook has created the ledger -- but before it can
        # return -- is still normalized as a post-precommit failure.  The
        # operational runner retries the same write-once precommit before
        # persisting that failure, so an earlier callback failure cannot cause
        # selected pixels or models to run without a durable successor.
        precommitted.append(successor)
        on_exposure_precommit(successor, snapshot)
        source_guard.check("after-exposure-precommit")

    def require_precommit(successor: ExposureLedger) -> None:
        if len(precommitted) != 1 or precommitted[0] != successor:
            raise StageACalibrationCommandError(
                "campaign did not durably precommit its exact exposure successor"
            )

    bound_proposer_transport = proposer_transport
    if proposer_transport is run_codex_structured:
        bound_proposer_transport = partial(
            proposer_transport,
            expected_launcher_digest=(
                config.expected_codex_launcher_digest
            ),
        )
    bound_scorer_transport = scorer_transport
    if scorer_transport is run_codex_named_images_structured:
        bound_scorer_transport = partial(
            scorer_transport,
            expected_launcher_digest=(
                config.expected_codex_launcher_digest
            ),
        )
    bound_proposer_transport = source_guard.wrap_transport(
        "proposer",
        bound_proposer_transport,
    )
    bound_scorer_transport = source_guard.wrap_transport(
        "scorer",
        bound_scorer_transport,
    )

    campaign_kwargs = {
        "candidate_count": config.candidate_count,
        "seed": config.selection_seed,
        "source_corpus_manifest_digest": trusted.full_manifest.digest,
        "expected_codex_launcher_digest": config.expected_codex_launcher_digest,
        "exposure_ledger": exposure_ledger,
        "expected_exposure_ledger_digest": config.expected_exposure_ledger_digest,
        "semantic_cohort": config.semantic_cohort,
        "families": config.families,
        "verifier_id": config.verifier_id,
        "label_nonce_root": config.label_nonce_root,
        "proposer_minutes": config.proposer_minutes,
        "scorer_minutes": config.scorer_minutes,
        "proposer_max_workers": config.proposer_max_workers,
        "scorer_max_workers": config.scorer_max_workers,
        "verbose": config.verbose,
        "executable": config.executable,
        "cloud_policy_cache_snapshot": snapshot,
        "proposer_transport": bound_proposer_transport,
        "scorer_transport": bound_scorer_transport,
        "on_exposure_precommit": campaign_exposure_precommit,
    }
    active_phase = "campaign-execution"
    try:
        try:
            campaign = campaign_runner(trusted.corpus, protocol, **campaign_kwargs)
            source_guard.check("after-campaign")
        except _TERMINAL_EXCEPTIONS as error:
            source_guard.check("after-campaign-terminal-failure")
            active_phase = "campaign-failure-validation"
            archive = _proposal_archive_from_terminal(error)
            successor = _assert_archive_matches_command(
                archive,
                trusted=trusted,
                ledger=exposure_ledger,
                config=config,
                protocol=protocol,
                snapshot=snapshot,
            )
            require_precommit(successor)
            active_phase = "failure-terminal-serialization"
            source_guard.check("before-terminal-serialization")
            schema, payload = _canonical_terminal(
                error.to_data(),
                status="failed",
                expected_internal_digest=error.digest,
            )
            source_guard.check("after-terminal-serialization")
            return StageAExecutionOutcome(
                status="failed",
                terminal_schema=schema,
                internal_digest=error.digest,
                terminal_payload=payload,
                exposure_successor=successor,
                protocol_digest=protocol.digest(),
                command_config_payload=canonical_json(config.to_data()),
                command_config_digest=config.digest,
                input_authentication_digest=trusted.digest,
                launcher_version=launcher_version,
                launcher_digest=actual_launcher,
                cloud_policy_cache_snapshot=snapshot,
                cloud_policy_cache_binding=snapshot.binding,
                source_dependencies=frozen_sources,
                source_dependency_root=source_root,
                cold_verified=False,
            )

        # Pass serialized data, not the warm object, to force the complete cold
        # decoder and pixel replay before any success bytes become persistable.
        active_phase = "cold-verification"
        source_guard.check("before-cold-verification")
        verified, _panels = campaign_verifier(
            campaign.to_data(),
            corpus=trusted.corpus,
            corpus_manifest=trusted.full_manifest,
        )
        source_guard.check("after-cold-verification")
        if (
            verified.digest != campaign.digest
            or verified.to_data() != campaign.to_data()
        ):
            raise StageACalibrationCommandError(
                "cold-verified campaign differs from warm campaign"
            )
        successor = _assert_archive_matches_command(
            _proposal_archive_from_success(verified),
            trusted=trusted,
            ledger=exposure_ledger,
            config=config,
            protocol=protocol,
            snapshot=snapshot,
        )
        require_precommit(successor)
        active_phase = "success-terminal-serialization"
        source_guard.check("before-terminal-serialization")
        schema, payload = _canonical_terminal(
            verified.to_data(),
            status="succeeded",
            expected_internal_digest=verified.digest,
        )
        source_guard.check("after-terminal-serialization")
        return StageAExecutionOutcome(
            status="succeeded",
            terminal_schema=schema,
            internal_digest=verified.digest,
            terminal_payload=payload,
            exposure_successor=successor,
            protocol_digest=protocol.digest(),
            command_config_payload=canonical_json(config.to_data()),
            command_config_digest=config.digest,
            input_authentication_digest=trusted.digest,
            launcher_version=launcher_version,
            launcher_digest=actual_launcher,
            cloud_policy_cache_snapshot=snapshot,
            cloud_policy_cache_binding=snapshot.binding,
            source_dependencies=frozen_sources,
            source_dependency_root=source_root,
            cold_verified=True,
        )
    except BaseException as error:
        # Nothing is archived before the durable disclosure boundary.  Once
        # that boundary has returned, every Python exception (including an
        # interrupt) becomes a typed, label-withheld failed outcome.
        if len(precommitted) != 1:
            raise
        return _post_precommit_failure_outcome(
            error,
            phase=active_phase,
            successor=precommitted[0],
            protocol=protocol,
            config=config,
            trusted=trusted,
            launcher_version=launcher_version,
            launcher_digest=actual_launcher,
            snapshot=snapshot,
            source_guard=source_guard,
        )


def _write_once_or_identical(path: Path, payload: bytes) -> Path:
    """Create one exact file, accepting only byte-identical idempotence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise StageACalibrationCommandError(
                f"cannot inspect existing Stage-A artifact {path}"
            ) from exc
        if existing != payload:
            raise StageACalibrationCommandError(
                f"refusing to overwrite different Stage-A artifact at {path}"
            )
    try:
        persisted = path.read_bytes()
    except OSError as exc:
        raise StageACalibrationCommandError(
            f"cannot verify persisted Stage-A artifact {path}"
        ) from exc
    if persisted != payload:
        raise StageACalibrationCommandError(
            f"persisted Stage-A artifact differs from intended bytes at {path}"
        )
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise StageACalibrationCommandError(
            f"cannot hash persisted file {path}"
        ) from exc
    return "sha256:" + digest.hexdigest()


_SECRET_KEY = re.compile(
    r"(?:^|[_-])(?:access[_-]?token|refresh[_-]?token|api[_-]?key|password|"
    r"credential|authorization|cookie|session[_-]?secret|private[_-]?key)"
    r"(?:$|[_-])",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"(?:\bBearer\s+[A-Za-z0-9._~+/-]+=*|\bsk-[A-Za-z0-9_-]{16,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----)",
    re.IGNORECASE,
)


def _assert_cache_snapshot_safe_for_private_handoff(
    snapshot: CloudPolicyCacheSnapshot,
) -> None:
    """Reject credential-like material before writing a local handoff.

    The signed cache currently contains account/user identifiers, so even a
    safe envelope is kept out of scientific artifacts and Git and written
    mode 0600.  This check is deliberately conservative about actual secrets;
    Stage B also reconstructs :class:`CloudPolicyCacheSnapshot`, re-running
    the transport layer's signed-envelope validator.
    """

    if snapshot.data is None:
        return
    try:
        decoded = json.loads(snapshot.data)
    except (UnicodeError, json.JSONDecodeError) as exc:  # pragma: no cover.
        raise StageACalibrationCommandError(
            "validated cache snapshot is no longer JSON"
        ) from exc

    def inspect(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise StageACalibrationCommandError(
                        f"cloud policy cache has a non-string key at {path}"
                    )
                if key != "signature" and _SECRET_KEY.search(key):
                    raise StageACalibrationCommandError(
                        "refusing to persist credential-like cloud policy "
                        f"cache field at {path}.{key}"
                    )
                inspect(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                inspect(child, f"{path}[{index}]")
        elif isinstance(value, str) and path != "$.signature":
            if _SECRET_VALUE.search(value):
                raise StageACalibrationCommandError(
                    "refusing to persist credential-like cloud policy cache value"
                )

    inspect(decoded, "$")


def _prepare_private_cache_directory(directory: Path) -> Path:
    """Require an explicit private directory outside Git or ignored by Git."""

    requested = directory.expanduser().absolute()
    try:
        if requested.is_symlink():
            raise StageACalibrationCommandError(
                "private cloud policy cache directory cannot be a symlink"
            )
    except OSError as exc:
        raise StageACalibrationCommandError(
            "cannot inspect the private cloud policy cache directory"
        ) from exc
    repository = Path(__file__).resolve().parents[1]
    resolved = requested.resolve()
    if resolved == repository or repository in resolved.parents:
        try:
            ignored = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "check-ignore",
                    "--no-index",
                    "--quiet",
                    "--",
                    str(resolved),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise StageACalibrationCommandError(
                "cannot prove that the private cache directory is Git-ignored"
            ) from exc
        if ignored.returncode != 0:
            raise StageACalibrationCommandError(
                "cloud policy cache snapshots inside the repository must use "
                "an explicitly Git-ignored directory"
            )
    try:
        resolved.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = resolved.lstat()
    except OSError as exc:
        raise StageACalibrationCommandError(
            "cannot create the private cloud policy cache directory"
        ) from exc
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or stat.S_IMODE(info.st_mode) & 0o077
        or (hasattr(os, "getuid") and info.st_uid != os.getuid())
    ):
        raise StageACalibrationCommandError(
            "cloud policy cache directory must be owner-only, non-symlinked, "
            "and owned by the current user"
        )
    return resolved


def _read_private_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise StageACalibrationCommandError(
            "platform cannot safely open a private cache snapshot"
        )
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise StageACalibrationCommandError(
            f"cannot open private cache snapshot {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or (hasattr(os, "getuid") and before.st_uid != os.getuid())
        ):
            raise StageACalibrationCommandError(
                "private cache snapshot must be a singly-linked owner-only file"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, 65_536)
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > 4 * 1024 * 1024:
                raise StageACalibrationCommandError(
                    "private cache snapshot is unexpectedly large"
                )
        after = os.fstat(descriptor)
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
            raise StageACalibrationCommandError(
                "private cache snapshot changed while being read"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_private_once_or_identical(path: Path, payload: bytes) -> Path:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise StageACalibrationCommandError(
            "platform cannot safely create a private cache snapshot"
        )
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        if _read_private_file(path) != payload:
            raise StageACalibrationCommandError(
                f"refusing to overwrite different private cache snapshot at {path}"
            )
        return path
    except OSError as exc:
        raise StageACalibrationCommandError(
            f"cannot create private cache snapshot {path}"
        ) from exc
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:  # pragma: no cover - os.write contract.
                raise StageACalibrationCommandError(
                    "short write while persisting private cache snapshot"
                )
            offset += written
        os.fsync(descriptor)
    except Exception:
        # Do not unlink here: even a partial disclosure must not be silently
        # discarded.  Its content-addressed validation will fail loudly.
        raise
    finally:
        os.close(descriptor)
    if _read_private_file(path) != payload:
        raise StageACalibrationCommandError(
            "persisted private cache snapshot differs from intended bytes"
        )
    return path


def persist_stage_a_cache_snapshot(
    snapshot: CloudPolicyCacheSnapshot,
    directory: str | Path,
) -> tuple[Path, str, int]:
    """Persist an exact, private, content-addressed Stage-A/Stage-B handoff."""

    if not isinstance(snapshot, CloudPolicyCacheSnapshot):
        raise TypeError("snapshot must be CloudPolicyCacheSnapshot")
    _assert_cache_snapshot_safe_for_private_handoff(snapshot)
    payload = b"" if snapshot.data is None else snapshot.data
    file_sha256 = "sha256:" + hashlib.sha256(payload).hexdigest()
    state = "absent" if snapshot.data is None else "present"
    suffix = "empty" if snapshot.data is None else "json"
    target_directory = _prepare_private_cache_directory(Path(directory))
    path = target_directory / (
        f"{file_sha256.removeprefix('sha256:')}.cloud-policy-cache-"
        f"{state}.{suffix}"
    )
    _write_private_once_or_identical(path, payload)
    return path.resolve(), file_sha256, len(payload)


def load_stage_a_cache_snapshot(
    path: str | Path,
    *,
    expected_binding: str,
    expected_file_sha256: str,
) -> CloudPolicyCacheSnapshot:
    """Load the exact Stage-A snapshot for Stage B; never inspect ambient cache."""

    if expected_binding != "absent":
        _address(expected_binding, "expected cloud policy cache binding")
    expected_file = _address(
        expected_file_sha256,
        "expected cloud policy cache snapshot file digest",
    )
    # Keep the caller's final path component unresolved so O_NOFOLLOW below
    # can reject a symlink rather than silently accepting its target.
    source = Path(path).expanduser().absolute()
    payload = _read_private_file(source)
    actual_file = "sha256:" + hashlib.sha256(payload).hexdigest()
    if actual_file != expected_file:
        raise StageACalibrationCommandError(
            "private cloud policy cache file differs from expected digest"
        )
    if expected_binding == "absent":
        if payload:
            raise StageACalibrationCommandError(
                "absent cloud policy cache handoff is not empty"
            )
        snapshot = CloudPolicyCacheSnapshot(None)
    else:
        snapshot = CloudPolicyCacheSnapshot(payload)
        _assert_cache_snapshot_safe_for_private_handoff(snapshot)
    if snapshot.binding != expected_binding:
        raise StageACalibrationCommandError(
            "private cloud policy cache preimage differs from expected binding"
        )
    return snapshot


@dataclass(frozen=True, slots=True)
class StageAExposurePrecommitReceipt:
    """Durable parents that precede selected-panel semantic materialization."""

    exposure_ledger_path: Path
    exposure_ledger_digest: str
    exposure_ledger_file_sha256: str
    cloud_policy_cache_snapshot_path: Path
    cloud_policy_cache_binding: str
    cloud_policy_cache_snapshot_file_sha256: str
    cloud_policy_cache_snapshot_byte_count: int


def _fsync_file_and_parent(path: Path) -> None:
    if not hasattr(os, "O_NOFOLLOW"):
        raise StageACalibrationCommandError(
            "platform cannot durably verify a write-once precommit"
        )
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, file_flags)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise StageACalibrationCommandError(
                    "precommit target is not a regular file"
                )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        directory_flags |= getattr(os, "O_DIRECTORY", 0)
        parent_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except OSError as exc:
        raise StageACalibrationCommandError(
            f"cannot fsync Stage-A precommit {path}"
        ) from exc


def persist_stage_a_exposure_precommit(
    successor: ExposureLedger,
    snapshot: CloudPolicyCacheSnapshot,
    persistence: StageAPersistenceConfig,
) -> StageAExposurePrecommitReceipt:
    """Persist and reload cache+ledger before selected panels enter semantics.

    The private cache is committed first.  Therefore the existence of the
    successor ledger implies that both causal parents reached durable storage.
    """

    if not isinstance(successor, ExposureLedger):
        raise TypeError("successor must be ExposureLedger")
    if not isinstance(snapshot, CloudPolicyCacheSnapshot):
        raise TypeError("snapshot must be CloudPolicyCacheSnapshot")
    if not isinstance(persistence, StageAPersistenceConfig):
        raise TypeError("persistence must be StageAPersistenceConfig")
    cache_path, cache_file_sha256, cache_byte_count = (
        persist_stage_a_cache_snapshot(
            snapshot,
            persistence.cache_snapshot_directory,
        )
    )
    _fsync_file_and_parent(cache_path)
    reloaded_cache = load_stage_a_cache_snapshot(
        cache_path,
        expected_binding=snapshot.binding,
        expected_file_sha256=cache_file_sha256,
    )
    if reloaded_cache != snapshot:
        raise StageACalibrationCommandError(
            "durable cache precommit differs from the frozen snapshot"
        )
    try:
        ledger_path = successor.write_content_addressed(
            persistence.exposure_directory
        ).resolve()
    except Exception as exc:  # noqa: BLE001 - normalize ledger persistence.
        raise StageACalibrationCommandError(
            "cannot persist Stage-A exposure successor before semantic access"
        ) from exc
    _fsync_file_and_parent(ledger_path)
    try:
        reloaded_ledger = ExposureLedger.load(ledger_path)
    except Exception as exc:  # noqa: BLE001 - normalize ledger verification.
        raise StageACalibrationCommandError(
            "cannot reload durable Stage-A exposure successor"
        ) from exc
    if reloaded_ledger != successor:
        raise StageACalibrationCommandError(
            "durable exposure successor differs from the in-memory precommit"
        )
    return StageAExposurePrecommitReceipt(
        exposure_ledger_path=ledger_path,
        exposure_ledger_digest=successor.digest,
        exposure_ledger_file_sha256=_file_sha256(ledger_path),
        cloud_policy_cache_snapshot_path=cache_path,
        cloud_policy_cache_binding=snapshot.binding,
        cloud_policy_cache_snapshot_file_sha256=cache_file_sha256,
        cloud_policy_cache_snapshot_byte_count=cache_byte_count,
    )


def _persist_stage_a_outcome_once(
    outcome: StageAExecutionOutcome,
    persistence: StageAPersistenceConfig,
    *,
    source_guard: _StageASourceDependencyGuard | None,
) -> StageACommandResult:
    """Persist one outcome, optionally checking sources before publication."""

    if not isinstance(outcome, StageAExecutionOutcome):
        raise TypeError("outcome must be StageAExecutionOutcome")
    if not isinstance(persistence, StageAPersistenceConfig):
        raise TypeError("persistence must be StageAPersistenceConfig")

    def check_source(phase: str) -> None:
        if source_guard is not None:
            source_guard.check(phase)

    check_source("before-terminal-artifact-persistence")
    suffix = "campaign" if outcome.status == "succeeded" else "failure"
    artifact_path = (
        persistence.artifact_directory
        / f"{outcome.internal_digest}.semantic-calibration-{suffix}.json"
    )
    # Failure details go first: even if the ledger directory is unavailable,
    # the exact terminal disclosure archive is not silently lost.
    _write_once_or_identical(artifact_path, outcome.terminal_payload)
    _fsync_file_and_parent(artifact_path)
    check_source("after-terminal-artifact-persistence")
    try:
        ledger_path = outcome.exposure_successor.write_content_addressed(
            persistence.exposure_directory
        )
    except Exception as exc:  # noqa: BLE001 - normalize exposure I/O boundary.
        raise StageACalibrationCommandError(
            "terminal artifact was preserved, but its exposure successor "
            f"could not be persisted: {artifact_path}"
        ) from exc
    check_source("after-exposure-ledger-persistence")
    try:
        cache_path, cache_file_sha256, cache_byte_count = (
            persist_stage_a_cache_snapshot(
                outcome.cloud_policy_cache_snapshot,
                persistence.cache_snapshot_directory,
            )
        )
    except Exception as exc:  # noqa: BLE001 - report prior durable parents.
        raise StageACalibrationCommandError(
            "terminal artifact and exposure successor were preserved, but the "
            "private Stage-A cache handoff could not be persisted: "
            f"artifact={artifact_path}, ledger={ledger_path}"
        ) from exc
    check_source("after-cache-snapshot-persistence")
    artifact_path = artifact_path.resolve()
    ledger_path = ledger_path.resolve()
    artifact_file_sha256 = _file_sha256(artifact_path)
    ledger_file_sha256 = _file_sha256(ledger_path)
    command_config_data = json.loads(outcome.command_config_payload)
    assert isinstance(command_config_data, Mapping)
    check_source("before-command-receipt-serialization")
    receipt_content = {
        "schema": STAGE_A_COMMAND_RECEIPT_SCHEMA,
        "status": outcome.status,
        "stage_a_scope": outcome.stage_a_scope,
        "terminal_schema": outcome.terminal_schema,
        "terminal_internal_digest": outcome.internal_digest,
        "terminal_artifact_path": str(artifact_path),
        "terminal_artifact_file_sha256": artifact_file_sha256,
        "exposure_ledger_path": str(ledger_path),
        "exposure_ledger_digest": outcome.exposure_successor.digest,
        "exposure_ledger_file_sha256": ledger_file_sha256,
        "protocol_digest": outcome.protocol_digest,
        "command_config": dict(command_config_data),
        "command_config_digest": outcome.command_config_digest,
        "input_authentication_digest": outcome.input_authentication_digest,
        "launcher_version": outcome.launcher_version,
        "launcher_digest": outcome.launcher_digest,
        "cloud_policy_cache_binding": outcome.cloud_policy_cache_binding,
        "cloud_policy_cache_snapshot_path": str(cache_path),
        "cloud_policy_cache_snapshot_file_sha256": cache_file_sha256,
        "cloud_policy_cache_snapshot_byte_count": cache_byte_count,
        "cloud_policy_cache_snapshot_bytes_embedded": False,
        "source_dependencies": outcome.source_dependencies.to_data(),
        "source_dependency_digest": outcome.source_dependencies.digest,
        "cold_verified": outcome.cold_verified,
        "python_predicate_authoritative": True,
        "optional_checker_may_affect_result": False,
    }
    receipt_digest = canonical_digest(receipt_content)
    receipt_data = {**receipt_content, "command_receipt_digest": receipt_digest}
    receipt_payload = canonical_json(receipt_data) + b"\n"
    receipt_path = persistence.artifact_directory / (
        f"{receipt_digest}.stage-a-command-receipt.json"
    )
    # Cold-decode every receipt field and reload the private cache before the
    # authoritative receipt filename exists.  A source change observed during
    # this final replay therefore cannot leave a successful receipt behind.
    verified_receipt = StageACommandReceipt.from_bytes(
        receipt_payload,
        expected_receipt_digest=receipt_digest,
    )
    verified_receipt.load_cache_snapshot()
    if verified_receipt.command_config.digest != outcome.command_config_digest:
        raise StageACalibrationCommandError(
            "persisted command receipt differs from frozen command config"
        )
    check_source("after-command-receipt-cold-reload")
    # Publishing the already replayed bytes is the last operation.  No Python
    # source-dependent interpretation occurs after this write-once boundary.
    _write_once_or_identical(receipt_path, receipt_payload)
    _fsync_file_and_parent(receipt_path)
    receipt_path = receipt_path.resolve()
    return StageACommandResult(
        status=outcome.status,
        terminal_schema=outcome.terminal_schema,
        internal_digest=outcome.internal_digest,
        artifact_path=artifact_path,
        artifact_file_sha256=artifact_file_sha256,
        exposure_ledger_path=ledger_path,
        exposure_ledger_digest=outcome.exposure_successor.digest,
        exposure_ledger_file_sha256=ledger_file_sha256,
        protocol_digest=outcome.protocol_digest,
        command_config_payload=outcome.command_config_payload,
        command_config_digest=outcome.command_config_digest,
        input_authentication_digest=outcome.input_authentication_digest,
        launcher_version=outcome.launcher_version,
        launcher_digest=outcome.launcher_digest,
        cloud_policy_cache_snapshot_path=cache_path,
        cloud_policy_cache_snapshot_file_sha256=cache_file_sha256,
        cloud_policy_cache_snapshot_byte_count=cache_byte_count,
        cloud_policy_cache_binding=outcome.cloud_policy_cache_binding,
        source_dependencies=outcome.source_dependencies,
        command_receipt_path=receipt_path,
        command_receipt_digest=receipt_digest,
        command_receipt_file_sha256=(
            "sha256:" + hashlib.sha256(receipt_payload).hexdigest()
        ),
        cold_verified=outcome.cold_verified,
    )


def _outcome_records_source_drift(outcome: StageAExecutionOutcome) -> bool:
    """Return whether an operational failure already records unreadable/drifted sources."""

    if (
        outcome.status != "failed"
        or outcome.terminal_schema != STAGE_A_OPERATIONAL_FAILURE_SCHEMA
    ):
        return False
    state = outcome.terminal_data().get("source_dependency_state")
    return state in {"mutated", "unreadable"}


def _persistence_failure_outcome(
    error: BaseException,
    *,
    outcome: StageAExecutionOutcome,
    source_guard: _StageASourceDependencyGuard,
) -> StageAExecutionOutcome:
    """Replace an unpublishable terminal result with a typed failure result."""

    phase = "terminal-persistence"
    try:
        source_guard.check("post-precommit-persistence-failure-normalization")
    except StageASourceDependencyMutationError as mutation:
        error = mutation
        phase = mutation.phase
    mutation = source_guard.first_failure
    if mutation is None:
        source_state: Literal["unchanged", "mutated", "unreadable"] = (
            "unchanged"
        )
        observed = source_guard.expected
        observation_error_digest = None
    elif mutation.observed is None:
        source_state = "unreadable"
        observed = None
        observation_error_digest = mutation.observation_error_digest
    else:
        source_state = "mutated"
        observed = mutation.observed
        observation_error_digest = None
    error_type, reason_digest = _bounded_operational_failure(error)
    failure = StageAOperationalFailureArtifact(
        failure_phase=phase,
        error_type=error_type,
        reason_digest=reason_digest,
        exposure_successor=outcome.exposure_successor,
        protocol_digest=outcome.protocol_digest,
        command_config_digest=outcome.command_config_digest,
        input_authentication_digest=outcome.input_authentication_digest,
        launcher_digest=outcome.launcher_digest,
        cloud_policy_cache_binding=outcome.cloud_policy_cache_binding,
        source_dependencies=source_guard.expected,
        source_dependency_state=source_state,
        observed_source_dependencies=observed,
        source_observation_error_digest=observation_error_digest,
    )
    schema, payload = _canonical_terminal(
        failure.to_data(),
        status="failed",
        expected_internal_digest=failure.digest,
    )
    return StageAExecutionOutcome(
        status="failed",
        terminal_schema=schema,
        internal_digest=failure.digest,
        terminal_payload=payload,
        exposure_successor=outcome.exposure_successor,
        protocol_digest=outcome.protocol_digest,
        command_config_payload=outcome.command_config_payload,
        command_config_digest=outcome.command_config_digest,
        input_authentication_digest=outcome.input_authentication_digest,
        launcher_version=outcome.launcher_version,
        launcher_digest=outcome.launcher_digest,
        cloud_policy_cache_snapshot=outcome.cloud_policy_cache_snapshot,
        cloud_policy_cache_binding=outcome.cloud_policy_cache_binding,
        source_dependencies=source_guard.expected,
        source_dependency_root=source_guard.root,
        cold_verified=False,
    )


def persist_stage_a_outcome(
    outcome: StageAExecutionOutcome,
    persistence: StageAPersistenceConfig,
) -> StageACommandResult:
    """Persist one terminal result or a typed post-precommit failure.

    Successful and canonical campaign-failure outcomes remain source-guarded
    until their receipt bytes have cold-replayed.  Any Python exception during
    that process is converted to a separate operational-failure artifact and
    failed receipt.  A source-drift outcome created earlier is already the
    fail-closed record and must remain persistable under the drift it records.
    """

    if not isinstance(outcome, StageAExecutionOutcome):
        raise TypeError("outcome must be StageAExecutionOutcome")
    if not isinstance(persistence, StageAPersistenceConfig):
        raise TypeError("persistence must be StageAPersistenceConfig")
    source_guard = _StageASourceDependencyGuard(
        outcome.source_dependency_root,
        outcome.source_dependencies,
    )
    if _outcome_records_source_drift(outcome):
        return _persist_stage_a_outcome_once(
            outcome,
            persistence,
            source_guard=None,
        )
    try:
        return _persist_stage_a_outcome_once(
            outcome,
            persistence,
            source_guard=source_guard,
        )
    except BaseException as error:
        failure_outcome = _persistence_failure_outcome(
            error,
            outcome=outcome,
            source_guard=source_guard,
        )
        return _persist_stage_a_outcome_once(
            failure_outcome,
            persistence,
            source_guard=None,
        )


def run_stage_a_calibration_command(
    trusted: StageATrustedCorpus,
    exposure_ledger: ExposureLedger,
    config: StageACalibrationCommandConfig,
    persistence: StageAPersistenceConfig,
    **execution_dependencies: Any,
) -> StageACommandResult:
    """Crash-safe execute and persist one terminal Stage-A result.

    The cache preimage and full-batch exposure successor become durable inside
    the campaign's pre-semantic-access hook.  A kill after that point can lose
    a terminal model artifact, but cannot make the disclosed task batch appear
    unseen.
    """

    if "on_exposure_precommit" in execution_dependencies:
        raise StageACalibrationCommandError(
            "run_stage_a_calibration_command owns the durable precommit hook"
        )
    precommit_attempts: list[
        tuple[ExposureLedger, CloudPolicyCacheSnapshot]
    ] = []
    precommits: list[StageAExposurePrecommitReceipt] = []

    def durable_precommit(
        successor: ExposureLedger,
        snapshot: CloudPolicyCacheSnapshot,
    ) -> None:
        if precommit_attempts:
            raise StageACalibrationCommandError(
                "durable Stage-A precommit was requested more than once"
            )
        # Retain the exact intended parents before entering filesystem code.
        # If that code writes the ledger and then raises (including an
        # interrupt), execute_stage_a_calibration can terminalize the error and
        # this runner can retry only these same write-once bytes.
        precommit_attempts.append((successor, snapshot))
        precommits.append(
            persist_stage_a_exposure_precommit(
                successor,
                snapshot,
                persistence,
            )
        )

    outcome = execute_stage_a_calibration(
        trusted,
        exposure_ledger,
        config,
        on_exposure_precommit=durable_precommit,
        **execution_dependencies,
    )
    if not precommit_attempts:
        # An interrupt can arrive after execute_stage_a_calibration records its
        # exact successor but before this callback executes its first Python
        # instruction.  Returning an outcome proves that boundary was crossed;
        # recover the same immutable parents from the outcome and complete the
        # conservative write-once precommit before publishing the failure.
        precommit_attempts.append(
            (
                outcome.exposure_successor,
                outcome.cloud_policy_cache_snapshot,
            )
        )
    if len(precommit_attempts) != 1:
        raise StageACalibrationCommandError(
            "campaign completed without one exact exposure precommit attempt"
        )
    if not precommits:
        successor, snapshot = precommit_attempts[0]
        precommits.append(
            persist_stage_a_exposure_precommit(
                successor,
                snapshot,
                persistence,
            )
        )
    if len(precommits) != 1:
        raise StageACalibrationCommandError(
            "campaign completed without one durable exposure precommit"
        )
    result = persist_stage_a_outcome(outcome, persistence)
    precommit = precommits[0]
    if (
        result.exposure_ledger_path != precommit.exposure_ledger_path
        or result.exposure_ledger_digest != precommit.exposure_ledger_digest
        or result.exposure_ledger_file_sha256
        != precommit.exposure_ledger_file_sha256
        or result.cloud_policy_cache_snapshot_path
        != precommit.cloud_policy_cache_snapshot_path
        or result.cloud_policy_cache_binding
        != precommit.cloud_policy_cache_binding
        or result.cloud_policy_cache_snapshot_file_sha256
        != precommit.cloud_policy_cache_snapshot_file_sha256
        or result.cloud_policy_cache_snapshot_byte_count
        != precommit.cloud_policy_cache_snapshot_byte_count
    ):
        raise StageACalibrationCommandError(
            "terminal persistence differs from the durable pre-access parents"
        )
    return result


__all__ = [
    "DESCRIPTIVE_STAGE_A_DESIGN",
    "STAGE_A_COMMAND_RECEIPT_SCHEMA_V1",
    "STAGE_A_OPERATIONAL_FAILURE_SCHEMA",
    "STAGE_A_SCOPE",
    "STAGE_A_SOURCE_DEPENDENCY_SCHEMA",
    "STAGE_A_SOURCE_DEPENDENCY_SCOPE",
    "StageACalibrationCommandConfig",
    "StageACalibrationCommandError",
    "StageACommandReceipt",
    "StageACommandResult",
    "StageAExecutionOutcome",
    "StageAExposurePrecommitReceipt",
    "StageAOperationalFailureArtifact",
    "StageAPersistenceConfig",
    "StageATrustedCorpus",
    "StageASourceDependencyIdentity",
    "StageASourceDependencyMutationError",
    "execute_stage_a_calibration",
    "freeze_stage_a_source_dependencies",
    "load_stage_a_cache_snapshot",
    "load_stage_a_command_receipt",
    "persist_stage_a_cache_snapshot",
    "persist_stage_a_exposure_precommit",
    "persist_stage_a_outcome",
    "run_stage_a_calibration_command",
]

"""Command-line entry points for the canonical Bongard pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence

from bongard.artifacts import (
    BlobRef,
    SupportCommitment,
    VerifiedRunArchive,
    canonical_digest,
    canonical_json,
    verify_archive_data,
    verify_support_commitment_data,
)
from bongard.benchmark import (
    SUPPORT_PROTOTYPE_PREDICATE_MODE,
    VISUAL_SEMANTIC_PREDICATE_MODE,
    SealedTestGuard,
    SupportGatePolicy,
    prepare_episode,
    run_episode,
)
from bongard.cohorts import build_cohort_report, classify_task
from bongard.corpus import PanelManifest, ShapeBongardCorpus, TaskManifest
from bongard.exposure import (
    ExposureLedger,
    ExposureViolation,
    LEDGER_SCHEMA,
    semantic_resolver_policy_digest,
    semantic_policy_blocked_keys,
)
from bongard.historical_exposure import load_historical_exposure
from bongard.proposer import HeadlessCodexEpisode
from bongard.pipeline_registry import (
    pipeline_registry_data,
    require_new_pipeline_execution,
)
from bongard.prototype_calibration import (
    PrototypeCalibrationIntegrityError,
    PrototypeCalibrationRecord,
)
from bongard.prototype_episode import HeadlessPrototypeEpisode
from bongard.semantic_episode import VisualSemanticEpisode
from bongard.semantic_protocol import build_visual_semantic_policy
from bongard.release import (
    DEFAULT_RELEASE_PATH,
    OfficialReleaseDescriptor,
    load_official_release,
)
from bongard.run_verification import (
    LEGACY_OUTER_RUN_SCHEMA,
    OUTER_RUN_SCHEMA,
    RunVerificationError,
    verify_completed_run_data,
    verify_rejected_run_data,
    verify_support_rejected_run_data,
)
from bongard.transport import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_REASONING_EFFORT,
    CodexProposerFailure,
    codex_cli_fingerprint,
    snapshot_cloud_policy_cache,
)


RUN_SCHEMA = OUTER_RUN_SCHEMA
EXPOSURE_PRECOMMIT_SCHEMA = "gkm.bongard-support-release-precommit.v2"
EXPOSURE_SOURCE = "bongard.cli.run"
CODEX_EXECUTABLE = "codex"
_MAX_EXPECTED_ACTION_PROGRAM_REPORT_BYTES = 1024 * 1024


class CliError(RuntimeError):
    pass


def _prefixed_digest(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _task_id_membership_digest(values: Sequence[str]) -> str:
    ordered = tuple(sorted(values))
    if len(ordered) != len(set(ordered)):
        raise CliError("task membership contains duplicate identifiers")
    payload = "".join(f"{value}\n" for value in ordered).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _is_prefixed_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _exposure_purpose(task_id: str, model: str, plan_digest: str) -> str:
    return (
        "support_release_precommit "
        f"task={task_id} model={model} plan_digest={plan_digest}"
    )


def _validate_run_exposure_args(
    *, exposure_dir: str | Path | None, ledger_in: str | Path | None,
    require_unseen: bool, sealed_test: bool,
) -> None:
    if exposure_dir is None or not str(exposure_dir).strip():
        raise CliError("every run requires --exposure-dir")
    if ledger_in is not None and not str(ledger_in).strip():
        raise CliError("--ledger-in must name a non-empty path")
    if require_unseen and ledger_in is None:
        raise CliError("--require-unseen requires --ledger-in")
    if sealed_test and ledger_in is None:
        raise CliError("--sealed-test requires --ledger-in")


def _validate_release_args(
    *, official_release: bool, archive: str | Path | None, sealed_test: bool = False
) -> None:
    if archive is not None and not official_release:
        raise CliError("--archive is meaningful only with --official-release")
    if official_release and (archive is None or not str(archive).strip()):
        raise CliError("--official-release requires --archive")
    if sealed_test and not official_release:
        raise CliError("--sealed-test requires exact --official-release verification")


def _validate_run_cohort_args(
    *,
    official_release: bool,
    require_unseen: bool,
    sealed_test: bool,
    expected_cohort: str | None,
) -> None:
    if expected_cohort not in {None, "drill", "dev", "sealed"}:
        raise CliError("run cohort must be drill, dev, or sealed")
    if official_release and expected_cohort is None:
        raise CliError("--official-release requires --cohort")
    if official_release and not require_unseen:
        raise CliError("--official-release requires --require-unseen")
    if expected_cohort is not None and not official_release:
        raise CliError("--cohort is meaningful only with --official-release")
    if official_release and sealed_test and expected_cohort != "sealed":
        raise CliError("--sealed-test requires --cohort sealed")
    if official_release and not sealed_test and expected_cohort == "sealed":
        raise CliError("the sealed cohort may be opened only with --sealed-test")


def _validate_codex_launcher(
    *, expected_sha256: str | None, official_release: bool
) -> Mapping[str, str] | None:
    """Pin official runs to externally supplied launcher bytes before support."""

    if expected_sha256 is None:
        if official_release:
            raise CliError(
                "--official-release requires --expected-codex-launcher-sha256"
            )
        return None
    if not _is_digest(expected_sha256):
        raise CliError(
            "expected Codex launcher SHA-256 must be exactly 64 lowercase hex digits"
        )
    try:
        fingerprint = codex_cli_fingerprint(CODEX_EXECUTABLE)
    except CodexProposerFailure as exc:
        raise CliError(f"cannot preflight the fixed Codex launcher: {exc}") from exc
    if fingerprint["launcher_digest"] != expected_sha256:
        raise CliError(
            "fixed Codex launcher SHA-256 is "
            f"{fingerprint['launcher_digest']}, expected {expected_sha256}"
        )
    return fingerprint


def _freeze_semantic_execution_environment(
    *,
    externally_expected_launcher_digest: str | None,
    stage_a_execution_config: object,
    official_release: bool,
) -> tuple[Mapping[str, str], Any]:
    """Preflight and freeze the exact transport environment calibrated in Stage A."""

    expected_launcher_digest = getattr(
        stage_a_execution_config, "expected_codex_launcher_digest", None
    )
    expected_cache_binding = getattr(
        stage_a_execution_config, "cloud_policy_cache_binding", None
    )
    if not _is_digest(expected_launcher_digest):
        raise CliError(
            "cold-verified Stage-A campaign has no valid launcher identity"
        )
    if expected_cache_binding != "absent" and not _is_prefixed_digest(
        expected_cache_binding
    ):
        raise CliError(
            "cold-verified Stage-A campaign has no valid cloud-policy cache "
            "identity"
        )
    if externally_expected_launcher_digest is None:
        raise CliError(
            "--predicate-mode visual-semantic requires "
            "--expected-codex-launcher-sha256"
        )
    if not _is_digest(externally_expected_launcher_digest):
        raise CliError(
            "expected Codex launcher SHA-256 must be exactly 64 lowercase "
            "hex digits"
        )
    if externally_expected_launcher_digest != expected_launcher_digest:
        raise CliError(
            "externally expected Codex launcher differs from the "
            "cold-verified Stage-A campaign"
        )
    fingerprint = _validate_codex_launcher(
        expected_sha256=expected_launcher_digest,
        official_release=official_release,
    )
    assert fingerprint is not None
    try:
        cache_snapshot = snapshot_cloud_policy_cache()
    except (CodexProposerFailure, OSError, TypeError, ValueError) as exc:
        raise CliError(
            f"cannot snapshot the Stage-A cloud-policy environment: {exc}"
        ) from exc
    if cache_snapshot.binding != expected_cache_binding:
        raise CliError(
            "current cloud-policy cache differs from the cold-verified "
            "Stage-A campaign"
        )
    return fingerprint, cache_snapshot


def _load_corpus(args: argparse.Namespace) -> ShapeBongardCorpus:
    return ShapeBongardCorpus.discover(
        args.corpus,
        split_file=args.split_file,
        require_complete=(
            getattr(args, "require_complete", False)
            or getattr(args, "official_release", False)
        ),
    )


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise CliError(f"refusing to overwrite existing run artifact: {path}") from exc


def _retired_pipeline_command(args: argparse.Namespace) -> int:
    """Fail before corpus, archive, pixel, transport, or output access."""

    require_new_pipeline_execution(args.retired_pipeline_id)
    raise AssertionError("retired pipeline unexpectedly authorized")


def _pipeline_status(_args: argparse.Namespace) -> int:
    sys.stdout.buffer.write(canonical_json(pipeline_registry_data()) + b"\n")
    return 0


def _strict_json_bytes(
    path: Path, *, expected_sha256: str | None = None
) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CliError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CliError("run artifact root must be a JSON object")
    if canonical_json(value) != raw:
        raise CliError("run artifact is not canonical JSON")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        if not _is_digest(expected_sha256):
            raise CliError("expected run SHA-256 must be exactly 64 lowercase hex digits")
        if actual_sha256 != expected_sha256:
            raise CliError(
                f"run file SHA-256 is {actual_sha256}, expected {expected_sha256}"
            )
    return value


def _load_prototype_calibration(path: str | Path) -> PrototypeCalibrationRecord:
    source = Path(path)
    try:
        payload = source.read_bytes()
        decoded = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CliError(f"cannot read prototype calibration {source}: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise CliError("prototype calibration root must be a JSON object")
    try:
        record = PrototypeCalibrationRecord.from_data(decoded)
    except (PrototypeCalibrationIntegrityError, TypeError, ValueError) as exc:
        raise CliError(f"prototype calibration is invalid: {exc}") from exc
    if record.canonical_json() + b"\n" != payload:
        raise CliError(
            "prototype calibration must be exact canonical JSON plus one newline"
        )
    # This rechecks the current extractor catalog and every selected space.
    # A stale calibration fails before corpus support pixels are released.
    record.to_freeze_policy()
    return record


def _load_semantic_calibration_campaign(
    path: str | Path,
    *,
    expected_campaign_digest: str,
    corpus: ShapeBongardCorpus,
    corpus_manifest: object,
) -> Any:
    """Load and cold-replay one externally pinned full Stage-A campaign.

    ``expected_campaign_digest`` is ``campaign.digest``: the SHA-256 of the
    campaign's canonical ``content_data()``.  It is deliberately not the file
    SHA-256, which would also cover the self-address field and trailing newline.
    """

    from bongard.corpus import CorpusManifest
    from bongard.semantic_calibration_campaign import (
        SEMANTIC_CALIBRATION_CAMPAIGN_SCHEMA,
        SemanticCalibrationCampaignError,
        verify_semantic_campaign_against_corpus,
    )

    if not _is_digest(expected_campaign_digest):
        raise CliError(
            "expected semantic calibration campaign digest must be exactly "
            "64 lowercase hex digits"
        )
    if not isinstance(corpus_manifest, CorpusManifest):
        raise CliError(
            "semantic calibration campaign requires a trusted full corpus manifest"
        )
    source = Path(path)
    try:
        payload = source.read_bytes()
        decoded = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CliError(
            f"cannot read semantic calibration campaign {source}: {exc}"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise CliError("semantic calibration campaign root must be a JSON object")
    if decoded.get("schema") != SEMANTIC_CALIBRATION_CAMPAIGN_SCHEMA:
        raise CliError(
            "visual-semantic runtime requires a full canonical Stage-A campaign; "
            "a bare semantic calibration artifact is not accepted"
        )
    if decoded.get("campaign_digest") != expected_campaign_digest:
        raise CliError(
            "semantic calibration campaign differs from the expected content digest"
        )
    if canonical_json(decoded) + b"\n" != payload:
        raise CliError(
            "semantic calibration campaign must be exact canonical JSON plus one newline"
        )
    try:
        campaign, _panel_map = verify_semantic_campaign_against_corpus(
            decoded,
            corpus=corpus,
            corpus_manifest=corpus_manifest,
        )
        build_visual_semantic_policy(
            campaign.calibration.family,
            prospective_protocol=campaign.calibration.protocol,
        )
    except (SemanticCalibrationCampaignError, TypeError, ValueError) as exc:
        raise CliError(
            f"semantic calibration campaign is invalid: {exc}"
        ) from exc
    if campaign.digest != expected_campaign_digest:
        raise CliError(
            "cold-replayed semantic calibration campaign differs from the "
            "expected content digest"
        )
    return campaign


def _inventory(args: argparse.Namespace) -> int:
    _validate_release_args(
        official_release=args.official_release,
        archive=args.archive,
    )
    corpus = _load_corpus(args)
    if args.require_complete:
        corpus.validate_complete(require_split=True)
    manifest = None
    official_release = None
    if args.official_release:
        if not args.archive:
            raise CliError("--official-release requires --archive")
        descriptor = load_official_release(args.release_descriptor)
        descriptor.verify_archive(args.archive)
        manifest = descriptor.verify_corpus(corpus)
        official_release = {
            "descriptor_digest": descriptor.digest,
            "release_id": descriptor.release_id,
            "archive_sha256": descriptor.archive_sha256,
            "archive_size_bytes": descriptor.archive_size_bytes,
            "task_ids_sha256": descriptor.task_ids_sha256,
            "corpus_manifest_sha256": descriptor.corpus_manifest_sha256,
        }
    if manifest is None:
        manifest = corpus.build_manifest()
    output = {
        "schema": "gkm.bongard-inventory-report.v1",
        "root": str(corpus.root),
        "layout": corpus.layout,
        "task_count": len(corpus),
        "family_counts": dict(corpus.family_counts),
        "split_counts": {
            name: len(values)
            for name, values in corpus.split.canonical_groups.items()
        },
        "manifest_digest": manifest.digest,
        "split_source_digest": corpus.split.source_digest,
        "official_release": official_release,
    }
    encoded = canonical_json(output)
    if args.out:
        _write_once(Path(args.out), encoded)
    sys.stdout.write(encoded.decode("utf-8") + "\n")
    return 0


def _resolve_explicit_action_program_source(
    value: str | Path, *, corpus: ShapeBongardCorpus, family: str
) -> Path:
    """Bind one explicit CLI path to the exact source the auditor will read."""

    if not str(value).strip():
        raise CliError(f"--{family}-action-programs must name a non-empty path")
    candidate = Path(value).expanduser()
    expected = corpus.root / family / f"{family}_action_programs.json"
    try:
        if candidate.is_symlink():
            raise CliError(
                f"explicit {family} action-program source must not be a symlink: "
                f"{candidate}"
            )
        if not candidate.is_file():
            raise CliError(
                f"explicit {family} action-program source is not a regular file: "
                f"{candidate}"
            )
        resolved_candidate = candidate.resolve(strict=True)
        resolved_expected = expected.resolve(strict=True)
    except CliError:
        raise
    except OSError as exc:
        raise CliError(
            f"cannot resolve explicit {family} action-program source: {exc}"
        ) from exc
    if resolved_candidate != resolved_expected:
        raise CliError(
            f"explicit {family} action-program source resolves to "
            f"{resolved_candidate}, but the audited corpus source is "
            f"{resolved_expected}"
        )
    return resolved_candidate


def _stable_expected_action_program_report(path: str | Path) -> bytes:
    source = Path(path).expanduser()
    try:
        if source.is_symlink():
            raise CliError(
                f"expected action-program report must not be a symlink: {source}"
            )
        before = source.stat()
        if not source.is_file():
            raise CliError(
                f"expected action-program report is not a regular file: {source}"
            )
        if (
            before.st_size <= 0
            or before.st_size > _MAX_EXPECTED_ACTION_PROGRAM_REPORT_BYTES
        ):
            raise CliError(
                "expected action-program report size is outside the "
                f"1..{_MAX_EXPECTED_ACTION_PROGRAM_REPORT_BYTES} byte bound"
            )
        payload = source.read_bytes()
        after = source.stat()
    except CliError:
        raise
    except OSError as exc:
        raise CliError(
            f"cannot read expected action-program report {source}: {exc}"
        ) from exc
    before_fingerprint = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_fingerprint = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_fingerprint != after_fingerprint or len(payload) != after.st_size:
        raise CliError(f"expected action-program report changed while reading: {source}")
    return payload


def _audit_action_programs(args: argparse.Namespace) -> int:
    """Regenerate only the privileged, aggregate post-hoc metadata audit."""

    # Keep this import local to the dedicated command.  Nothing in the run,
    # proposal, observer, or benchmark paths receives action-program metadata.
    from bongard.action_program_audit import audit_action_program_metadata

    expected_digest = args.expected_report_digest
    if expected_digest is not None and not _is_prefixed_digest(expected_digest):
        raise CliError(
            "--expected-report-digest must be sha256: followed by exactly "
            "64 lowercase hexadecimal digits"
        )

    corpus = ShapeBongardCorpus.discover(
        args.corpus,
        split_file=args.split_file,
        require_complete=True,
    )
    for family in ("ff", "bd", "hd"):
        _resolve_explicit_action_program_source(
            getattr(args, f"{family}_action_programs"),
            corpus=corpus,
            family=family,
        )

    release = load_official_release(args.release_descriptor)
    report = audit_action_program_metadata(corpus, official_release=release)
    if expected_digest is not None and report.digest != expected_digest:
        raise CliError(
            f"action-program report digest is {report.digest}, expected "
            f"{expected_digest}"
        )

    encoded = canonical_json(report.to_dict()) + b"\n"
    if args.expected_report is not None:
        expected = _stable_expected_action_program_report(args.expected_report)
        if expected != encoded:
            raise CliError(
                "regenerated action-program report differs from --expected-report: "
                f"observed sha256:{hashlib.sha256(encoded).hexdigest()}, "
                f"expected sha256:{hashlib.sha256(expected).hexdigest()}"
            )
    if args.out:
        _write_once(Path(args.out), encoded)
    sys.stdout.write(encoded.decode("utf-8"))
    return 0


def _cohorts(args: argparse.Namespace) -> int:
    corpus = _load_corpus(args)
    if args.require_complete:
        corpus.validate_complete(require_split=True)
    if isinstance(args.limit, bool) or args.limit < 0:
        raise CliError("--limit must be a non-negative integer")
    historical = load_historical_exposure()
    report = build_cohort_report(
        corpus,
        historical,
        split=args.split,
        family=args.family,
        cohort=args.cohort,
    )
    live_eligibility: dict[str, object] | None = None
    selected_records = report.records
    ledger_in = getattr(args, "ledger_in", None)
    if ledger_in is not None:
        if not str(ledger_in).strip():
            raise CliError("--ledger-in must name a non-empty path")

        # The ledger is not accepted against a cached or caller-supplied
        # digest. Rebuild the complete panel manifest from the corpus now and
        # bind all recorded task/panel IDs to that exact inventory.
        manifest = corpus.build_manifest()
        ledger = _load_bound_ledger(
            ledger_in=ledger_in,
            manifest=manifest,
            corpus=corpus,
        )
        resolver_policy_digest = semantic_resolver_policy_digest(historical)
        exposed = ledger.derive_exposed_semantic_keys(
            historical_seed=historical,
            expected_historical_seed_digest=historical.seed_digest,
            expected_resolver_policy_digest=resolver_policy_digest,
        )
        exposed_tasks = set(exposed.task_ids)
        exposed_semantics = set(exposed.semantic_keys)
        policy_blocked_semantics = set(semantic_policy_blocked_keys(historical))
        effective_exposed_semantics = exposed_semantics | policy_blocked_semantics

        # Ask the same bound exposure resolver for each distinct semantic
        # group. An empty ledger yields the candidate keys without collision;
        # comparing those typed keys avoids a second CLI-local interpretation
        # of Basic, Abstract, or Freeform identifiers.
        empty = ExposureLedger.create(manifest.digest)
        group_collisions: dict[tuple[str, tuple[str, ...]], bool] = {}
        exact_collision_ids: list[str] = []
        semantic_collision_ids: list[str] = []
        eligible_ids: list[str] = []
        for record in report.records:
            group = (record.family, record.parsed.concepts)
            if group not in group_collisions:
                try:
                    candidate = empty.assert_semantically_unseen(
                        task_ids=(record.task_id,),
                        historical_seed=historical,
                        expected_historical_seed_digest=historical.seed_digest,
                        expected_resolver_policy_digest=resolver_policy_digest,
                    )
                except ExposureViolation:
                    # A v2 morphology cluster may be policy-blocked even with
                    # an empty live ledger because it crosses historical or
                    # drill/dev/sealed information boundaries.
                    group_collisions[group] = True
                else:
                    group_collisions[group] = bool(
                        set(candidate.semantic_keys) & effective_exposed_semantics
                    )
            exact_collision = record.task_id in exposed_tasks
            semantic_collision = group_collisions[group]
            if exact_collision:
                exact_collision_ids.append(record.task_id)
            if semantic_collision:
                semantic_collision_ids.append(record.task_id)
            if not exact_collision and not semantic_collision:
                eligible_ids.append(record.task_id)

        exact_ids = tuple(sorted(exact_collision_ids))
        semantic_ids = tuple(sorted(semantic_collision_ids))
        exact_set = set(exact_ids)
        semantic_set = set(semantic_ids)
        both_ids = tuple(sorted(exact_set & semantic_set))
        excluded_ids = tuple(sorted(exact_set | semantic_set))
        live_ids = tuple(sorted(eligible_ids))
        live_set = set(live_ids)
        selected_records = tuple(
            record for record in report.records if record.task_id in live_set
        )
        live_eligibility_content = {
            "qualification": (
                "live exposure is a filtering overlay only; it does not alter "
                "the frozen historical cohort report or certify unseen panel bytes"
            ),
            "corpus_manifest_digest": manifest.digest,
            "ledger_digest": ledger.digest,
            "ledger_event_count": len(ledger.events),
            "historical_seed_digest": historical.seed_digest,
            "semantic_resolver_policy_digest": resolver_policy_digest,
            "recorded_semantic_keys_digest": _prefixed_digest(
                [key.to_dict() for key in exposed.semantic_keys]
            ),
            "policy_blocked_semantic_keys_digest": _prefixed_digest(
                [key.to_dict() for key in sorted(policy_blocked_semantics)]
            ),
            "effective_exposed_semantic_keys_digest": _prefixed_digest(
                [key.to_dict() for key in sorted(effective_exposed_semantics)]
            ),
            "counts": {
                "historical_scope": len(report.records),
                "ledger_recorded_tasks_total": len(exposed.task_ids),
                "ledger_recorded_semantic_keys_total": len(exposed.semantic_keys),
                "policy_blocked_semantic_keys_total": len(policy_blocked_semantics),
                "effective_exposed_semantic_keys_total": len(
                    effective_exposed_semantics
                ),
                "exact_task_collision": len(exact_ids),
                "semantic_key_collision": len(semantic_ids),
                "exact_and_semantic_collision": len(both_ids),
                "live_excluded_union": len(excluded_ids),
                "live_eligible": len(live_ids),
            },
            "membership_digests": {
                "exact_task_collision": _task_id_membership_digest(exact_ids),
                "semantic_key_collision": _task_id_membership_digest(semantic_ids),
                "exact_and_semantic_collision": _task_id_membership_digest(both_ids),
                "live_excluded_union": _task_id_membership_digest(excluded_ids),
                "live_eligible": _task_id_membership_digest(live_ids),
            },
        }
        live_eligibility = {
            **live_eligibility_content,
            "digest": _prefixed_digest(live_eligibility_content),
        }

    selected = [record.task_id for record in selected_records[: args.limit]]
    output = {
        "schema": "gkm.bongard-cohort-summary.v1",
        "qualification": report.to_dict()["qualification"],
        "report_digest": report.digest,
        "seed_digest": report.seed_digest,
        "split_index_digest": report.split_index_digest,
        "scope": dict(report.scope),
        "inventory_digest": report.inventory_digest,
        "counts": dict(report.counts),
        "membership_digests": dict(report.membership_digests),
        "selected_task_ids": selected,
        "selection_limit": args.limit,
    }
    if live_eligibility is not None:
        output["live_eligibility"] = live_eligibility
    encoded = canonical_json(output)
    if args.out:
        _write_once(Path(args.out), encoded)
    sys.stdout.write(encoded.decode("utf-8") + "\n")
    return 0


def _calibrate_semantic_stage_a(args: argparse.Namespace) -> int:
    """Run the official, exploratory-only visual-semantic calibration."""

    from bongard.semantic_calibration_command import (
        DESCRIPTIVE_STAGE_A_DESIGN,
        StageACalibrationCommandConfig,
        StageAPersistenceConfig,
        StageATrustedCorpus,
        run_stage_a_calibration_command,
    )

    corpus = ShapeBongardCorpus.discover(
        args.corpus,
        split_file=args.split_file,
        require_complete=True,
    )
    release = load_official_release(args.release_descriptor)
    trusted = StageATrustedCorpus.from_official_release(
        corpus=corpus,
        release=release,
        archive_path=args.archive,
    )
    ledger = _load_bound_ledger(
        ledger_in=args.ledger_in,
        manifest=trusted.full_manifest,
        corpus=corpus,
    )
    config = StageACalibrationCommandConfig(
        expected_codex_launcher_digest=args.expected_codex_launcher_sha256,
        expected_exposure_ledger_digest=args.expected_ledger_digest,
        design_mode=DESCRIPTIVE_STAGE_A_DESIGN,
        selection_seed=args.selection_seed,
        selection_seed_provenance=args.selection_seed_provenance,
        candidate_count=args.candidate_count,
        semantic_cohort="drill",
        families=("bd", "hd"),
        proposer_model_id=args.model,
        proposer_reasoning_effort=args.reasoning_effort,
        scorer_model_id=args.model,
        scorer_reasoning_effort=args.reasoning_effort,
        minimum_clusters_per_bin=args.minimum_clusters_per_bin,
        proposer_minutes=args.proposer_minutes,
        scorer_minutes=args.scorer_minutes,
        proposer_max_workers=args.workers,
        scorer_max_workers=args.workers,
        executable=CODEX_EXECUTABLE,
        verbose=args.verbose,
    )
    persistence = StageAPersistenceConfig(
        artifact_directory=Path(args.artifact_dir),
        exposure_directory=Path(args.exposure_dir),
        cache_snapshot_directory=Path(args.private_cache_dir),
    )
    result = run_stage_a_calibration_command(
        trusted,
        ledger,
        config,
        persistence,
    )
    sys.stdout.write(canonical_json(result.to_data()).decode("utf-8") + "\n")
    return 0 if result.status == "succeeded" else 2


def _validate_semantic_stage_b(args: argparse.Namespace) -> int:
    """Run the strict-disjoint, descriptive-only visual-semantic DEV batch."""

    from bongard.semantic_calibration_command import (
        load_stage_a_command_receipt,
    )
    from bongard.semantic_gated_dev_validation import (
        GatedDevAcceptancePolicy,
        plan_gated_dev_validation,
        run_gated_dev_validation,
    )

    corpus = ShapeBongardCorpus.discover(
        args.corpus,
        split_file=args.split_file,
        require_complete=True,
    )
    release = load_official_release(args.release_descriptor)
    release.verify_archive(args.archive)
    manifest = release.verify_corpus(corpus)
    receipt = load_stage_a_command_receipt(
        args.stage_a_command_receipt,
        args.expected_stage_a_command_receipt_digest,
    )
    campaign = _load_semantic_calibration_campaign(
        receipt.terminal_artifact_path,
        expected_campaign_digest=receipt.terminal_internal_digest,
        corpus=corpus,
        corpus_manifest=manifest,
    )
    cache_snapshot = receipt.load_cache_snapshot()
    predecessor = _load_bound_ledger(
        ledger_in=receipt.exposure_ledger_path,
        manifest=manifest,
        corpus=corpus,
    )
    if predecessor.digest != receipt.exposure_ledger_digest:
        raise CliError(
            "Stage-A command receipt exposure successor differs from its file"
        )
    policy = build_visual_semantic_policy(
        campaign.calibration.family,
        prospective_protocol=campaign.calibration.protocol,
    )
    split_digest = corpus.split.source_digest
    if split_digest is None:
        raise CliError("Stage-B requires the authenticated official split digest")
    plan = plan_gated_dev_validation(
        corpus,
        source_corpus_manifest=manifest,
        expected_source_corpus_manifest_digest=release.corpus_manifest_sha256,
        expected_split_source_digest=release.split_sha256,
        stage_a_campaign=campaign,
        stage_a_command_receipt=receipt,
        visual_semantic_policy=policy,
        exposure_predecessor=predecessor,
        expected_exposure_predecessor_digest=receipt.exposure_ledger_digest,
        public_seed=args.selection_seed,
        selection_seed_provenance=args.selection_seed_provenance,
        requested_task_count=args.task_count,
        exposure_observed_at=args.exposure_observed_at,
        cloud_policy_cache_snapshot=cache_snapshot,
        acceptance_policy=GatedDevAcceptancePolicy(),
        families=("bd", "hd"),
        task_max_workers=args.workers,
    )
    artifact = run_gated_dev_validation(
        corpus,
        plan,
        source_corpus_manifest=manifest,
        stage_a_campaign=campaign,
        stage_a_command_receipt=receipt,
        visual_semantic_policy=policy,
        exposure_predecessor=predecessor,
        exposure_output_directory=Path(args.exposure_dir),
        artifact_output_directory=Path(args.artifact_dir),
        cloud_policy_cache_snapshot=cache_snapshot,
        verbose=args.verbose,
    )
    artifact_path = (
        Path(args.artifact_dir)
        / f"{artifact.digest}.gated-dev-validation.json"
    ).resolve()
    exposure_path = (
        Path(args.exposure_dir)
        / (
            artifact.exposure_successor.digest.removeprefix("sha256:")
            + ".exposure.json"
        )
    ).resolve()
    output = {
        "schema": "gkm.bongard-stage-b-command-result.v1",
        "scientific_status": "descriptive-pilot",
        "authorizes_sealed_benchmark": False,
        "plan_digest": plan.digest,
        "validation_artifact_digest": artifact.digest,
        "validation_artifact_path": str(artifact_path),
        "exposure_successor_digest": artifact.exposure_successor.digest,
        "exposure_successor_path": str(exposure_path),
        "stage_a_command_receipt_digest": receipt.receipt_digest,
        "selected_by_family": {
            family: sum(item.family == family for item in plan.selections)
            for family in plan.families
        },
        "summary": artifact.summary.to_data(),
    }
    sys.stdout.write(canonical_json(output).decode("utf-8") + "\n")
    return 0


def _load_bound_ledger(
    *,
    ledger_in: str | Path | None,
    manifest: Any,
    corpus: ShapeBongardCorpus,
) -> ExposureLedger:
    ledger = ExposureLedger.load(ledger_in) if ledger_in is not None else ExposureLedger.create(
        manifest.digest
    )
    ledger.assert_corpus(manifest.digest)

    known_tasks = set(corpus.task_ids)
    unknown_tasks = set(ledger.exposed_task_ids) - known_tasks
    if unknown_tasks:
        raise CliError(
            f"input exposure ledger names tasks outside the corpus manifest: {sorted(unknown_tasks)}"
        )
    known_panels = {
        panel.panel_id
        for task_manifest in manifest.tasks
        for panel in task_manifest.panels
    }
    unknown_panels = set(ledger.explicitly_exposed_panel_ids) - known_panels
    if unknown_panels:
        raise CliError(
            "input exposure ledger names panels outside the corpus manifest: "
            f"{sorted(unknown_panels)}"
        )
    return ledger


def _precommit_exposure(
    *,
    corpus: ShapeBongardCorpus,
    manifest: Any,
    plan: Any,
    model: str,
    exposure_dir: str | Path,
    ledger_in: str | Path | None,
    require_unseen: bool,
    expected_cohort: str | None = None,
    require_semantic_unseen: bool = False,
) -> tuple[dict[str, Any], Path]:
    if not isinstance(model, str) or not model.strip():
        raise CliError("exposure precommit requires a non-empty model identity")
    ledger = _load_bound_ledger(
        ledger_in=ledger_in,
        manifest=manifest,
        corpus=corpus,
    )
    historical = None
    resolver_policy_digest = None
    assignment = None
    semantic_receipt = None
    if require_semantic_unseen or expected_cohort is not None:
        historical = load_historical_exposure()
        resolver_policy_digest = semantic_resolver_policy_digest(historical)
        assignment = classify_task(
            plan.task_id,
            historical,
            split=plan.split,
            regime=plan.regime,
        )
    if expected_cohort is not None and assignment is not None and (
        not assignment.historically_clean
        or assignment.semantic_cohort != expected_cohort
    ):
        raise CliError(
            f"task {plan.task_id!r} belongs to semantic cohort "
            f"{assignment.semantic_cohort!r}, expected {expected_cohort!r}"
        )
    if require_semantic_unseen:
        assert historical is not None and resolver_policy_digest is not None
        semantic_receipt = ledger.assert_semantically_unseen(
            task_ids=(plan.task_id,),
            historical_seed=historical,
            expected_historical_seed_digest=historical.seed_digest,
            expected_resolver_policy_digest=resolver_policy_digest,
        )
    known_panels = tuple(
        panel.panel_id
        for task_manifest in manifest.tasks
        for panel in task_manifest.panels
    )
    successor = ledger.record(
        phase="support_release_precommit",
        actor=model,
        purpose=_exposure_purpose(plan.task_id, model, plan.digest),
        task_ids=(plan.task_id,),
        source=EXPOSURE_SOURCE,
        known_task_ids=corpus.task_ids,
        known_panel_ids=known_panels,
        require_unseen=require_unseen,
    )
    event = successor.events[-1]
    try:
        successor_path = successor.write_content_addressed(Path(exposure_dir))
    except OSError as exc:
        raise CliError(f"cannot persist exposure precommit: {exc}") from exc
    exposure = {
        "schema": EXPOSURE_PRECOMMIT_SCHEMA,
        "corpus_manifest_digest": manifest.digest,
        "task_id": plan.task_id,
        "model": model,
        "plan_digest": plan.digest,
        "ledger_before_digest": ledger.digest,
        "ledger_after_digest": successor.digest,
        "event_digest": event.digest,
        "event": event.to_dict(),
        "ledger_before_event_count": len(ledger.events),
        "ledger_after_event_count": len(successor.events),
        "ledger_input_supplied": ledger_in is not None,
        "unseen_required": require_unseen,
        "semantic_unseen_required": require_semantic_unseen,
        "historical_seed_digest": (
            historical.seed_digest if historical is not None else None
        ),
        "semantic_resolver_policy_digest": resolver_policy_digest,
        "expected_semantic_cohort": expected_cohort,
        "classified_semantic_cohort": (
            assignment.semantic_cohort if assignment is not None else None
        ),
        "semantic_unseen_receipt": (
            semantic_receipt.to_dict() if semantic_receipt is not None else None
        ),
        "successor_filename": successor_path.name,
        # The local content address records what was written and when the
        # runner wrote it. It is not an externally anchored authenticity claim.
        "external_anchor": None,
    }
    return exposure, successor_path


def _verify_exposure_object(
    exposure: object,
    *,
    corpus_manifest_digest: object,
    episode: object,
) -> Mapping[str, Any]:
    if not isinstance(exposure, Mapping):
        raise CliError("run exposure must be an object")
    expected = {
        "schema",
        "corpus_manifest_digest",
        "task_id",
        "model",
        "plan_digest",
        "ledger_before_digest",
        "ledger_after_digest",
        "event_digest",
        "event",
        "ledger_before_event_count",
        "ledger_after_event_count",
        "ledger_input_supplied",
        "unseen_required",
        "semantic_unseen_required",
        "historical_seed_digest",
        "semantic_resolver_policy_digest",
        "expected_semantic_cohort",
        "classified_semantic_cohort",
        "semantic_unseen_receipt",
        "successor_filename",
        "external_anchor",
    }
    if set(exposure) != expected or exposure.get("schema") != EXPOSURE_PRECOMMIT_SCHEMA:
        raise CliError("run exposure fields or schema differ")
    if exposure.get("corpus_manifest_digest") != corpus_manifest_digest:
        raise CliError("run exposure belongs to a different corpus manifest")
    if not isinstance(episode, Mapping):
        raise CliError("run episode must be an object")
    if exposure.get("task_id") != episode.get("task_id"):
        raise CliError("run exposure task differs from episode task")
    if exposure.get("plan_digest") != episode.get("plan_digest"):
        raise CliError("run exposure plan differs from episode plan")
    if not _is_digest(exposure.get("plan_digest")):
        raise CliError("run exposure plan_digest is not a lowercase sha256")
    model = exposure.get("model")
    if not isinstance(model, str) or not model.strip():
        raise CliError("run exposure model must be non-empty")
    for field in ("ledger_before_digest", "ledger_after_digest", "event_digest"):
        if not _is_prefixed_digest(exposure.get(field)):
            raise CliError(f"run exposure {field} is not a sha256 content address")

    before_count = exposure.get("ledger_before_event_count")
    after_count = exposure.get("ledger_after_event_count")
    if (
        isinstance(before_count, bool)
        or not isinstance(before_count, int)
        or before_count < 0
        or isinstance(after_count, bool)
        or not isinstance(after_count, int)
        or after_count != before_count + 1
    ):
        raise CliError("run exposure event counts are not a one-event successor")
    ledger_input_supplied = exposure.get("ledger_input_supplied")
    unseen_required = exposure.get("unseen_required")
    semantic_unseen_required = exposure.get("semantic_unseen_required")
    if (
        not isinstance(ledger_input_supplied, bool)
        or not isinstance(unseen_required, bool)
        or not isinstance(semantic_unseen_required, bool)
    ):
        raise CliError("run exposure ledger-input/unseen flags must be Boolean")
    if semantic_unseen_required and not unseen_required:
        raise CliError("semantic unseen policy requires exact unseen policy")
    if unseen_required and not ledger_input_supplied:
        raise CliError("run exposure claims unseen without an input ledger")
    if episode.get("split") == "test" and not (
        unseen_required and ledger_input_supplied
    ):
        raise CliError("test episode exposure lacks an unseen input-ledger check")

    expected_cohort = exposure.get("expected_semantic_cohort")
    if expected_cohort not in {None, "drill", "dev", "sealed"}:
        raise CliError("run exposure expected semantic cohort is invalid")
    semantic_receipt = exposure.get("semantic_unseen_receipt")
    if semantic_unseen_required:
        historical = load_historical_exposure()
        policy_digest = semantic_resolver_policy_digest(historical)
        if exposure.get("historical_seed_digest") != historical.seed_digest:
            raise CliError("run exposure historical seed differs from the checked seed")
        if exposure.get("semantic_resolver_policy_digest") != policy_digest:
            raise CliError("run exposure semantic resolver policy differs")
        classified = classify_task(
            str(exposure.get("task_id")),
            historical,
            split=episode.get("split"),
            regime=episode.get("regime"),
        )
        if exposure.get("classified_semantic_cohort") != classified.semantic_cohort:
            raise CliError("run exposure semantic cohort classification differs")
        if expected_cohort is not None and (
            not classified.historically_clean
            or classified.semantic_cohort != expected_cohort
        ):
            raise CliError("run exposure task is outside its declared semantic cohort")
        if not isinstance(semantic_receipt, Mapping):
            raise CliError("run exposure lacks its semantic-unseen receipt")
        expected_receipt = ExposureLedger.create(
            str(corpus_manifest_digest)
        ).assert_semantically_unseen(
            task_ids=(str(exposure.get("task_id")),),
            historical_seed=historical,
            expected_historical_seed_digest=historical.seed_digest,
            expected_resolver_policy_digest=policy_digest,
        )
        expected_data = expected_receipt.to_dict()
        for field in (
            "task_ids",
            "semantic_keys",
            "historical_seed_digest",
            "resolver_policy_digest",
        ):
            if semantic_receipt.get(field) != expected_data[field]:
                raise CliError(
                    f"run exposure semantic-unseen receipt {field} differs"
                )
        if semantic_receipt.get("ledger_digest") != exposure.get(
            "ledger_before_digest"
        ):
            raise CliError(
                "run exposure semantic-unseen receipt does not bind its predecessor"
            )
        if set(semantic_receipt) != set(expected_data):
            raise CliError("run exposure semantic-unseen receipt fields differ")
    elif (
        semantic_receipt is not None
        or expected_cohort is not None
        or exposure.get("classified_semantic_cohort") is not None
        or exposure.get("historical_seed_digest") is not None
        or exposure.get("semantic_resolver_policy_digest") is not None
    ):
        raise CliError("run exposure has unsolicited semantic-policy fields")

    event = exposure.get("event")
    if not isinstance(event, Mapping):
        raise CliError("run exposure event must be an object")
    event_fields = {
        "sequence",
        "observed_at",
        "phase",
        "actor",
        "purpose",
        "task_ids",
        "panel_ids",
        "source",
        "previous_digest",
        "digest",
    }
    if set(event) != event_fields:
        raise CliError("run exposure event fields differ")
    if event.get("sequence") != before_count:
        raise CliError("run exposure event sequence differs from predecessor count")
    if event.get("phase") != "support_release_precommit":
        raise CliError("run exposure event has the wrong phase")
    if event.get("actor") != model:
        raise CliError("run exposure event actor differs from model")
    task_id = exposure.get("task_id")
    plan_digest = exposure.get("plan_digest")
    if event.get("purpose") != _exposure_purpose(task_id, model, plan_digest):
        raise CliError("run exposure event purpose does not bind task/model/plan")
    if event.get("task_ids") != [task_id] or event.get("panel_ids") != []:
        raise CliError("run exposure event does not name exactly the released task")
    if event.get("source") != EXPOSURE_SOURCE:
        raise CliError("run exposure event source differs")
    if not isinstance(event.get("observed_at"), str) or not event["observed_at"]:
        raise CliError("run exposure event timestamp is missing")
    previous = event.get("previous_digest")
    if before_count == 0:
        if previous is not None:
            raise CliError("initial exposure event has a predecessor digest")
    elif not _is_prefixed_digest(previous):
        raise CliError("successor exposure event lacks a predecessor event digest")
    event_content = {key: value for key, value in event.items() if key != "digest"}
    event_digest = _prefixed_digest(event_content)
    if event.get("digest") != event_digest or exposure.get("event_digest") != event_digest:
        raise CliError("run exposure event digest mismatch")

    before_digest = exposure.get("ledger_before_digest")
    after_digest = exposure.get("ledger_after_digest")
    if before_digest == after_digest:
        raise CliError("run exposure predecessor and successor digests are identical")
    expected_filename = after_digest.removeprefix("sha256:") + ".exposure.json"
    if exposure.get("successor_filename") != expected_filename:
        raise CliError("run exposure successor filename differs from its digest")
    if exposure.get("external_anchor") is not None:
        raise CliError("run record cannot assert an unverified external exposure anchor")

    # For a newly created ledger the entire predecessor and successor are
    # available in the run record, so both content addresses are reproducible.
    # For a non-empty predecessor we intentionally make no hash-chain
    # authenticity claim without loading that ledger or an external anchor.
    if before_count == 0:
        empty_ledger = {
            "schema": LEDGER_SCHEMA,
            "corpus_digest": corpus_manifest_digest,
            "events": [],
        }
        one_event_ledger = {
            "schema": LEDGER_SCHEMA,
            "corpus_digest": corpus_manifest_digest,
            "events": [dict(event)],
        }
        if before_digest != _prefixed_digest(empty_ledger):
            raise CliError("initial exposure predecessor digest mismatch")
        if after_digest != _prefixed_digest(one_event_ledger):
            raise CliError("initial exposure successor digest mismatch")
    return exposure


def _run_record(
    *,
    corpus: ShapeBongardCorpus,
    task_id: str,
    seed: str,
    session: Any | None,
    sealed_test: bool,
    exposure_dir: str | Path,
    ledger_in: str | Path | None,
    require_unseen: bool,
    model: str,
    expected_cohort: str | None = None,
    official_release: bool = False,
    archive_path: str | Path | None = None,
    release_descriptor: str | Path = DEFAULT_RELEASE_PATH,
    session_factory: Callable[[Any], Any] | None = None,
    support_gate_policy: SupportGatePolicy | None = None,
    predicate_mode: str | None = None,
    predicate_policy_digest: str | None = None,
    record_schema: str = RUN_SCHEMA,
    artifact_field: str = "vision",
    record_additions: Mapping[str, Any] | None = None,
    record_verifier: Callable[..., Any] | None = None,
    expected_corpus_manifest_digest: str | None = None,
    expected_split_source_digest: str | None = None,
) -> tuple[dict[str, Any], Any]:
    if (session is None) == (session_factory is None):
        raise CliError("run requires exactly one session or post-plan session factory")
    if not isinstance(record_schema, str) or not record_schema.strip():
        raise CliError("run record schema must be a non-empty string")
    if artifact_field not in {"vision", "prototype", "visual_semantic"}:
        raise CliError(
            "run artifact field must be vision, prototype, or visual_semantic"
        )
    additions = dict(record_additions or {})
    reserved = {
        "schema",
        "corpus_manifest_digest",
        "split_source_digest",
        "official_release",
        "plan",
        "support_commitment",
        "episode",
        "vision",
        "prototype",
        "visual_semantic",
        "run_archive",
        "exposure",
        "record_digest",
    }
    if set(additions) & reserved:
        raise CliError("run record additions collide with canonical outer fields")
    _validate_run_exposure_args(
        exposure_dir=exposure_dir,
        ledger_in=ledger_in,
        require_unseen=require_unseen,
        sealed_test=sealed_test,
    )
    _validate_release_args(
        official_release=official_release,
        archive=archive_path,
    )
    _validate_run_cohort_args(
        official_release=official_release,
        require_unseen=require_unseen,
        sealed_test=sealed_test,
        expected_cohort=expected_cohort,
    )
    release: OfficialReleaseDescriptor | None = None
    if official_release:
        assert archive_path is not None
        release = load_official_release(release_descriptor)
        release.verify_archive(archive_path)
        manifest = release.verify_corpus(corpus)
    else:
        manifest = corpus.build_manifest()
    if (
        expected_corpus_manifest_digest is not None
        and manifest.digest != expected_corpus_manifest_digest
    ):
        raise CliError(
            "run corpus manifest differs from the cold-verified calibration campaign"
        )
    if (
        expected_split_source_digest is not None
        and corpus.split.source_digest != expected_split_source_digest
    ):
        raise CliError(
            "run split source differs from the cold-verified calibration campaign"
        )
    plan = prepare_episode(
        corpus,
        task_id,
        seed=seed,
        corpus_manifest=manifest,
        predicate_mode=predicate_mode,
        predicate_policy_digest=predicate_policy_digest,
    )
    if session_factory is not None:
        session = session_factory(plan)
    if session is None or not callable(getattr(session, "artifact_data", None)):
        raise CliError("run session does not expose canonical artifact data")
    if plan.split == "test" and not sealed_test:
        raise CliError("test episodes require --sealed-test and a complete corpus")
    if plan.split != "test" and sealed_test:
        raise CliError("--sealed-test cannot be used for a non-test task")
    guard = None
    if sealed_test:
        guard = SealedTestGuard.capture(
            corpus, corpus_manifest=manifest, require_complete=True
        )
        guard.verify_all()
    exposure, successor_path = _precommit_exposure(
        corpus=corpus,
        manifest=manifest,
        plan=plan,
        model=model,
        exposure_dir=exposure_dir,
        ledger_in=ledger_in,
        require_unseen=require_unseen or sealed_test,
        expected_cohort=expected_cohort,
        require_semantic_unseen=official_release,
    )
    result = run_episode(
        plan,
        session,
        session,
        support_gate_policy=(support_gate_policy or SupportGatePolicy.empirical()),
        sealed_guard=guard,
    )
    if guard is not None:
        guard.verify_all()

    session_artifact = session.artifact_data()
    if not isinstance(session_artifact, Mapping):
        raise CliError("run session artifact must be a JSON object")
    session_artifact = dict(session_artifact)
    session_artifact["support_gate"] = (
        result.support_gate.to_data() if result.support_gate is not None else None
    )
    session_artifact["proposal_freeze"] = (
        result.proposal_freeze.to_data()
        if result.proposal_freeze is not None
        else None
    )
    archive = result.bundle.to_archive_data() if result.bundle else None
    if archive is not None:
        verified = verify_archive_data(archive)
        if artifact_field != "visual_semantic":
            proposal = session_artifact.get("proposal")
            if not isinstance(proposal, Mapping):
                raise CliError("completed run is missing its visual proposal artifact")
            proposal_digest = canonical_digest(proposal)
            if proposal_digest != verified.bundle.freeze.proposer_digest:
                raise CliError(
                    "visual proposal bytes differ from frozen proposer digest"
                )
    content: dict[str, Any] = {
        "schema": record_schema,
        "corpus_manifest_digest": manifest.digest,
        "split_source_digest": corpus.split.source_digest,
        "official_release": release.to_dict() if release is not None else None,
        "plan": plan.to_data(),
        "support_commitment": plan.support.to_data(),
        "episode": result.to_data(),
        artifact_field: session_artifact,
        "run_archive": archive,
        "exposure": exposure,
        **additions,
    }
    record = {**content, "record_digest": canonical_digest(content)}
    if record_verifier is not None:
        blob_sources = (
            (*plan._support_sources, *plan._query_sources)
            if result.bundle is not None
            else plan._support_sources
        )
        blob_bytes = {
            source.panel.blob_id: source.read_verified()
            for source in blob_sources
        }
        try:
            record_verifier(record, blob_bytes_by_id=blob_bytes)
        except (TypeError, ValueError, RuntimeError, OSError) as exc:
            raise CliError(
                f"run failed immediate profile-specific cold verification: {exc}"
            ) from exc
        return record, result
    if archive is not None:
        blob_bytes = {
            source.panel.blob_id: source.path.read_bytes()
            for source in (*plan._support_sources, *plan._query_sources)
        }
        try:
            verify_completed_run_data(record, blob_bytes_by_id=blob_bytes)
        except (RunVerificationError, OSError) as exc:
            raise CliError(
                f"completed run failed immediate strict cold verification: {exc}"
            ) from exc
    elif session_artifact.get("rejected_proposal_attempt") is not None:
        positive_sources = tuple(
            source for source in plan._support_sources if source.positive
        )
        negative_sources = tuple(
            source for source in plan._support_sources if not source.positive
        )
        support_bytes_by_name = {
            **{
                f"pos_{index}.png": source.read_verified()
                for index, source in enumerate(positive_sources)
            },
            **{
                f"neg_{index}.png": source.read_verified()
                for index, source in enumerate(negative_sources)
            },
        }
        try:
            verify_rejected_run_data(
                record, support_bytes_by_name=support_bytes_by_name
            )
        except (RunVerificationError, OSError) as exc:
            raise CliError(
                f"rejected proposal failed immediate strict cold verification: {exc}"
            ) from exc
    elif result.status.value == "support_rejected" \
            and result.support_gate is not None \
            and result.proposal_freeze is not None:
        support_blob_bytes = {
            source.panel.blob_id: source.read_verified()
            for source in plan._support_sources
        }
        try:
            verify_support_rejected_run_data(
                record,
                blob_bytes_by_id=support_blob_bytes,
            )
        except (RunVerificationError, OSError) as exc:
            raise CliError(
                f"rejected support gate failed immediate strict cold verification: {exc}"
            ) from exc
    return record, result


def _run(args: argparse.Namespace) -> int:
    _validate_run_exposure_args(
        exposure_dir=args.exposure_dir,
        ledger_in=args.ledger_in,
        require_unseen=args.require_unseen,
        sealed_test=args.sealed_test,
    )
    _validate_release_args(
        official_release=args.official_release,
        archive=args.archive,
        sealed_test=args.sealed_test,
    )
    expected_cohort = getattr(args, "cohort", None)
    _validate_run_cohort_args(
        official_release=args.official_release,
        require_unseen=args.require_unseen,
        sealed_test=args.sealed_test,
        expected_cohort=expected_cohort,
    )
    predicate_mode = getattr(args, "predicate_mode", "hybrid")
    if args.sealed_test and predicate_mode == "visual-semantic":
        raise CliError(
            "visual-semantic SEALED execution is disabled: the current "
            "Stage-A calibration is exploratory and the descriptive DEV "
            "benchmark cannot authorize a sealed run"
        )
    # A visual-semantic run gets its environment authority from the fully
    # cold-authenticated Stage-A campaign below.  Other modes retain their
    # existing external launcher preflight here.
    launcher_fingerprint = (
        None
        if predicate_mode == "visual-semantic"
        else _validate_codex_launcher(
            expected_sha256=getattr(
                args, "expected_codex_launcher_sha256", None
            ),
            official_release=args.official_release,
        )
    )
    corpus = _load_corpus(args)
    prototype_calibration_path = getattr(args, "prototype_calibration", None)
    semantic_campaign_path = getattr(
        args, "semantic_calibration_campaign", None
    )
    expected_semantic_campaign_digest = getattr(
        args, "expected_semantic_calibration_campaign_digest", None
    )
    run_options: dict[str, Any]
    if predicate_mode == "support-prototype":
        if prototype_calibration_path is None:
            raise CliError(
                "--predicate-mode support-prototype requires "
                "--prototype-calibration"
            )
        if (
            semantic_campaign_path is not None
            or expected_semantic_campaign_digest is not None
        ):
            raise CliError(
                "semantic calibration campaign arguments are valid only with "
                "--predicate-mode visual-semantic"
            )
        calibration = _load_prototype_calibration(prototype_calibration_path)
        policy = calibration.to_freeze_policy()
        from bongard.prototype_run_verification import (
            PROTOTYPE_OUTER_RUN_SCHEMA,
            verify_prototype_run_data,
        )

        def prototype_session(plan: Any) -> HeadlessPrototypeEpisode:
            return HeadlessPrototypeEpisode(
                support_commitment=plan.support,
                policy=policy,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                proposer_minutes=args.proposer_minutes,
                verbose=args.verbose,
                executable=CODEX_EXECUTABLE,
            )

        session = None
        run_options = {
            "session_factory": prototype_session,
            "support_gate_policy": SupportGatePolicy.prototype(),
            "predicate_mode": SUPPORT_PROTOTYPE_PREDICATE_MODE,
            "predicate_policy_digest": policy.digest(),
            "record_schema": PROTOTYPE_OUTER_RUN_SCHEMA,
            "artifact_field": "prototype",
            "record_additions": {"calibration": calibration.to_data()},
            "record_verifier": verify_prototype_run_data,
        }
    elif predicate_mode == "visual-semantic":
        if semantic_campaign_path is None:
            raise CliError(
                "--predicate-mode visual-semantic requires "
                "--semantic-calibration-campaign"
            )
        if expected_semantic_campaign_digest is None:
            raise CliError(
                "--predicate-mode visual-semantic requires "
                "--expected-semantic-calibration-campaign-digest"
            )
        if prototype_calibration_path is not None:
            raise CliError(
                "--prototype-calibration is valid only with "
                "--predicate-mode support-prototype"
            )
        if args.official_release:
            assert args.archive is not None
            semantic_release = load_official_release(args.release_descriptor)
            semantic_release.verify_archive(args.archive)
            semantic_manifest = semantic_release.verify_corpus(corpus)
        else:
            semantic_manifest = corpus.build_manifest()
        semantic_campaign = _load_semantic_calibration_campaign(
            semantic_campaign_path,
            expected_campaign_digest=expected_semantic_campaign_digest,
            corpus=corpus,
            corpus_manifest=semantic_manifest,
        )
        execution_config = (
            semantic_campaign.score_batch.commitment_batch.proposal_archive
            .execution_config
        )
        launcher_fingerprint, semantic_cache_snapshot = (
            _freeze_semantic_execution_environment(
                externally_expected_launcher_digest=getattr(
                    args, "expected_codex_launcher_sha256", None
                ),
                stage_a_execution_config=execution_config,
                official_release=args.official_release,
            )
        )
        semantic_calibration = semantic_campaign.calibration
        semantic_policy = build_visual_semantic_policy(
            semantic_calibration.family,
            prospective_protocol=semantic_calibration.protocol,
        )
        protocol = semantic_calibration.protocol
        if args.model != protocol.proposer_model_id:
            raise CliError(
                "--model differs from the proposer model frozen in the "
                "semantic calibration"
            )
        if args.reasoning_effort != protocol.proposer_reasoning_effort:
            raise CliError(
                "--reasoning-effort differs from the proposer effort frozen "
                "in the semantic calibration"
            )
        from bongard.semantic_run_verification import (
            VISUAL_SEMANTIC_OUTER_RUN_SCHEMA,
            verify_visual_semantic_run_data,
        )

        def verify_semantic_record(
            record: Mapping[str, Any], *, blob_bytes_by_id: Mapping[str, bytes]
        ) -> Any:
            return verify_visual_semantic_run_data(
                record,
                blob_bytes_by_id=blob_bytes_by_id,
                calibration_campaign=semantic_campaign,
            )

        def semantic_session(plan: Any) -> VisualSemanticEpisode:
            return VisualSemanticEpisode(
                task_id=plan.task_id,
                support_commitment=plan.support,
                policy=semantic_policy,
                family=semantic_calibration.family,
                protocol=semantic_calibration.protocol,
                proposer_minutes=args.proposer_minutes,
                scorer_minutes=args.observer_minutes,
                verbose=args.verbose,
                executable=CODEX_EXECUTABLE,
                cloud_policy_cache_snapshot=semantic_cache_snapshot,
                expected_codex_launcher_digest=(
                    execution_config.expected_codex_launcher_digest
                ),
                expected_cloud_policy_cache_binding=(
                    execution_config.cloud_policy_cache_binding
                ),
            )

        session = None
        run_options = {
            "session_factory": semantic_session,
            "support_gate_policy": SupportGatePolicy.visual_semantic(),
            "predicate_mode": VISUAL_SEMANTIC_PREDICATE_MODE,
            "predicate_policy_digest": semantic_policy.digest(),
            "record_schema": VISUAL_SEMANTIC_OUTER_RUN_SCHEMA,
            "artifact_field": "visual_semantic",
            "record_additions": {
                "calibration_campaign_digest": semantic_campaign.digest,
                "calibration": semantic_calibration.to_data(),
            },
            "record_verifier": verify_semantic_record,
            "expected_corpus_manifest_digest": (
                semantic_calibration.plan.corpus_manifest_digest
            ),
            "expected_split_source_digest": (
                semantic_calibration.plan.split_source_digest
            ),
        }
    elif predicate_mode == "hybrid":
        if prototype_calibration_path is not None:
            raise CliError(
                "--prototype-calibration is valid only with "
                "--predicate-mode support-prototype"
            )
        if (
            semantic_campaign_path is not None
            or expected_semantic_campaign_digest is not None
        ):
            raise CliError(
                "semantic calibration campaign arguments are valid only with "
                "--predicate-mode visual-semantic"
            )
        session = HeadlessCodexEpisode(
            observable_catalog={},
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            proposer_minutes=args.proposer_minutes,
            observer_minutes=args.observer_minutes,
            verbose=args.verbose,
            executable=CODEX_EXECUTABLE,
        )
        run_options = {}
    else:  # pragma: no cover - argparse closes this for public calls.
        raise CliError(f"unsupported predicate mode {predicate_mode!r}")
    record, result = _run_record(
        corpus=corpus,
        task_id=args.task_id,
        seed=args.seed,
        session=session,
        sealed_test=args.sealed_test,
        exposure_dir=args.exposure_dir,
        ledger_in=args.ledger_in,
        require_unseen=args.require_unseen,
        model=args.model,
        expected_cohort=expected_cohort,
        official_release=args.official_release,
        archive_path=args.archive,
        release_descriptor=args.release_descriptor,
        **run_options,
    )
    successor_path = Path(args.exposure_dir) / record["exposure"]["successor_filename"]
    encoded = canonical_json(record)
    destination = Path(args.out)
    _write_once(destination, encoded)
    summary = {
        "out": str(destination.resolve()),
        "record_sha256": hashlib.sha256(encoded).hexdigest(),
        "status": result.status.value,
        "score": result.score.to_data(),
        "calibration_campaign_digest": record.get(
            "calibration_campaign_digest"
        ),
        "exposure_ledger_before_digest": record["exposure"][
            "ledger_before_digest"
        ],
        "exposure_ledger_after_digest": record["exposure"][
            "ledger_after_digest"
        ],
        "exposure_event_digest": record["exposure"]["event_digest"],
        "exposure_ledger_out": str(successor_path.resolve()),
        "official_release_digest": (
            OfficialReleaseDescriptor.from_dict(record["official_release"]).digest
            if record["official_release"] is not None
            else None
        ),
        "codex_launcher_sha256": (
            launcher_fingerprint["launcher_digest"]
            if launcher_fingerprint is not None
            else None
        ),
        "codex_cli_version": (
            launcher_fingerprint["version"]
            if launcher_fingerprint is not None
            else None
        ),
    }
    sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
    return 0 if result.bundle is not None else 2


def _official_task_manifest(
    manifest: object, task_id: str
) -> TaskManifest:
    tasks = getattr(manifest, "tasks", ())
    matches = tuple(item for item in tasks if item.task_id == task_id)
    if len(matches) != 1:
        raise CliError(
            f"official corpus manifest contains {len(matches)} entries for "
            f"episode task {task_id!r}"
        )
    task_manifest = matches[0]
    if not isinstance(task_manifest, TaskManifest):
        # Test doubles may be structurally equivalent, but the exact verify
        # command accepts only manifests built by ShapeBongardCorpus.
        raise CliError("official task manifest has an unexpected representation")
    return task_manifest


def _archive_panel_refs(archive: VerifiedRunArchive) -> tuple[BlobRef, ...]:
    refs = tuple(item.panel for item in archive.bundle.support.support) + tuple(
        item.panel for item in archive.bundle.release.queries
    )
    if len(refs) != 14 or len({item.blob_id for item in refs}) != 14:
        raise CliError("verified archive does not contain fourteen unique panel BlobRefs")
    return refs


def _map_official_task_blob_bytes(
    task_manifest: TaskManifest,
    archive: VerifiedRunArchive,
) -> dict[str, bytes]:
    """Bind the archive's panel roles to the exact fourteen official PNGs.

    Source paths and artifact IDs intentionally have unrelated names.  Their
    sole admissible join key is the committed ``(SHA-256, byte_count)`` pair.
    The join must be bijective; duplicate identities are rejected instead of
    choosing an arbitrary source file.  Once joined, official polarity and
    panel index are checked against every support slot and revealed query
    label, so resealing an internally valid archive cannot relabel or permute
    the official task.

    The public run schema intentionally commits only ``seed_digest``, not a
    seed preimage.  Consequently this verifier binds the submitted release to
    one exact official role/index assignment, but does not claim to reproduce
    which query indices or query order the hidden seed would have selected.
    Adding that stronger claim requires a versioned run-schema migration.
    """

    panels = tuple(task_manifest.panels)
    refs = _archive_panel_refs(archive)
    if len(panels) != 14:
        raise CliError(
            f"official episode task has {len(panels)} panels, expected fourteen"
        )

    panels_by_side: dict[str, list[PanelManifest]] = {
        "positive": [],
        "negative": [],
    }
    panel_index: dict[tuple[str, int], list[PanelManifest]] = {}
    for panel in panels:
        if not isinstance(panel, PanelManifest):
            raise CliError("official task panel has an unexpected representation")
        if panel.task_id != task_manifest.task_id or panel.family != task_manifest.family:
            raise CliError("official panel task/family identity differs from its task")
        if panel.polarity not in panels_by_side:
            raise CliError(f"official panel has invalid polarity: {panel.panel_id}")
        if isinstance(panel.index, bool) or not isinstance(panel.index, int):
            raise CliError(f"official panel has invalid index: {panel.panel_id}")
        label = "1" if panel.polarity == "positive" else "0"
        expected_filename = f"{panel.index}.png"
        expected_panel_id = (
            f"{task_manifest.family}/{task_manifest.task_id}/{label}/"
            f"{expected_filename}"
        )
        if panel.filename != expected_filename or panel.panel_id != expected_panel_id:
            raise CliError(
                "official panel index/identity differs from its canonical task record"
            )
        panels_by_side[panel.polarity].append(panel)
        identity = (panel.sha256.removeprefix("sha256:"), panel.size_bytes)
        panel_index.setdefault(identity, []).append(panel)
    for polarity, side_panels in panels_by_side.items():
        if len(side_panels) != 7 or {panel.index for panel in side_panels} != set(range(7)):
            raise CliError(
                f"official {polarity} side does not contain canonical indices 0..6"
            )

    ref_index: dict[tuple[str, int], list[BlobRef]] = {}
    for ref in refs:
        if ref.media_type != "image/png":
            raise CliError(f"archive panel {ref.blob_id!r} is not an official PNG")
        identity = (ref.sha256, ref.byte_count)
        ref_index.setdefault(identity, []).append(ref)

    ambiguous_panels = {
        identity: values for identity, values in panel_index.items() if len(values) != 1
    }
    ambiguous_refs = {
        identity: values for identity, values in ref_index.items() if len(values) != 1
    }
    if ambiguous_panels or ambiguous_refs:
        raise CliError(
            "ambiguous digest+size mapping between official task PNGs and BlobRefs"
        )

    missing = sorted(set(ref_index) - set(panel_index))
    extras = sorted(set(panel_index) - set(ref_index))
    if missing or extras:
        raise CliError(
            "official task PNG identities do not exactly cover archive BlobRefs: "
            f"missing={missing}, extras={extras}"
        )

    bound_panels = {
        refs_for_identity[0].blob_id: panel_index[identity][0]
        for identity, refs_for_identity in ref_index.items()
    }

    support = tuple(archive.bundle.support.support)
    support_by_id = {item.panel.blob_id: item for item in support}
    expected_support_ids = {
        f"support-{polarity}-{slot}"
        for polarity in ("positive", "negative")
        for slot in range(6)
    }
    if set(support_by_id) != expected_support_ids:
        raise CliError("archive support does not use canonical official support slots")
    for blob_id, item in support_by_id.items():
        panel = bound_panels[blob_id]
        official_positive = panel.polarity == "positive"
        if item.positive is not official_positive:
            raise CliError(
                f"support label for {blob_id!r} differs from official polarity"
            )

    queries = tuple(archive.bundle.release.queries)
    expected_query_roles = {
        "query-0": "query-panel-0",
        "query-1": "query-panel-1",
    }
    if {
        query.query_id: query.panel.blob_id for query in queries
    } != expected_query_roles:
        raise CliError("archive query release does not use canonical public query roles")
    labels_by_id = {
        label.query_id: label for label in archive.bundle.labels.labels
    }
    if set(labels_by_id) != set(expected_query_roles):
        raise CliError("archive revealed labels do not cover the official query roles")
    query_panel_by_side: dict[str, PanelManifest] = {}
    for query in queries:
        panel = bound_panels[query.panel.blob_id]
        label = labels_by_id[query.query_id]
        official_positive = panel.polarity == "positive"
        if label.positive is not official_positive:
            raise CliError(
                f"revealed label for {query.query_id!r} differs from official polarity"
            )
        if panel.polarity in query_panel_by_side:
            raise CliError("official query release does not contain one panel per polarity")
        query_panel_by_side[panel.polarity] = panel
    if set(query_panel_by_side) != {"positive", "negative"}:
        raise CliError("official query release does not contain one panel per polarity")

    for polarity in ("positive", "negative"):
        query_panel = query_panel_by_side[polarity]
        expected_support = sorted(
            (
                panel
                for panel in panels_by_side[polarity]
                if panel.panel_id != query_panel.panel_id
            ),
            key=lambda panel: panel.index,
        )
        for slot, official_panel in enumerate(expected_support):
            blob_id = f"support-{polarity}-{slot}"
            if bound_panels[blob_id].to_dict() != official_panel.to_dict():
                raise CliError(
                    f"{blob_id!r} differs from its official panel index/identity"
                )

    result: dict[str, bytes] = {}
    for identity, ref_values in ref_index.items():
        ref = ref_values[0]
        panel = panel_index[identity][0]
        try:
            before = panel.path.stat()
            payload = panel.path.read_bytes()
            after = panel.path.stat()
        except OSError as exc:
            raise CliError(f"cannot read official task PNG {panel.path}: {exc}") from exc
        if (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise CliError(f"official task PNG changed while reading: {panel.path}")
        actual = (hashlib.sha256(payload).hexdigest(), len(payload))
        if actual != identity:
            raise CliError(
                f"official task PNG changed after manifest construction: {panel.path}"
            )
        result[ref.blob_id] = payload
    if set(result) != {item.blob_id for item in refs}:
        raise CliError("official task byte mapping is not a BlobRef bijection")
    return result


def _map_official_rejected_support_bytes(
    task_manifest: TaskManifest,
    attempt_value: object,
) -> dict[str, bytes]:
    """Resolve a rejected turn's named byte identities inside one official task."""

    if not isinstance(attempt_value, Mapping):
        raise CliError("proposal-failure record lacks a rejected proposal attempt")
    presentation = attempt_value.get("support_presentation")
    if not isinstance(presentation, list) or len(presentation) != 12:
        raise CliError("rejected proposal support presentation is malformed")
    buckets: dict[tuple[str, str, int], list[PanelManifest]] = {}
    for panel in task_manifest.panels:
        buckets.setdefault(
            (
                panel.polarity,
                panel.sha256.removeprefix("sha256:"),
                panel.size_bytes,
            ),
            [],
        ).append(panel)
    for values in buckets.values():
        values.sort(key=lambda item: (item.index, item.panel_id))

    result: dict[str, bytes] = {}
    for item in presentation:
        if not isinstance(item, Mapping):
            raise CliError("rejected proposal support identity is malformed")
        name = item.get("name")
        digest = item.get("content_digest")
        byte_count = item.get("byte_count")
        if not isinstance(name, str) or not isinstance(digest, str) \
                or isinstance(byte_count, bool) or not isinstance(byte_count, int):
            raise CliError("rejected proposal support identity is malformed")
        if re.fullmatch(r"pos_[0-5]\.png", name):
            polarity = "positive"
        elif re.fullmatch(r"neg_[0-5]\.png", name):
            polarity = "negative"
        else:
            raise CliError(f"rejected proposal support name is noncanonical: {name!r}")
        candidates = buckets.get((polarity, digest, byte_count), [])
        if not candidates:
            raise CliError(
                f"{name}: rejected support identity is absent from the official task"
            )
        panel = candidates.pop(0)
        try:
            payload = panel.path.read_bytes()
        except OSError as exc:
            raise CliError(f"cannot read official support PNG {panel.path}: {exc}") from exc
        if len(payload) != byte_count or hashlib.sha256(payload).hexdigest() != digest:
            raise CliError(f"{name}: official support PNG changed after manifest build")
        result[name] = payload
    if len(result) != 12:
        raise CliError("rejected proposal support presentation contains duplicate names")
    return result


def _map_official_support_blob_bytes(
    task_manifest: TaskManifest,
    support: SupportCommitment,
) -> dict[str, bytes]:
    """Resolve a v5 support preimage to exact bytes in one official task."""

    expected_ids = {
        *(f"support-negative-{slot}" for slot in range(6)),
        *(f"support-positive-{slot}" for slot in range(6)),
    }
    by_id = {item.panel.blob_id: item for item in support.support}
    if set(by_id) != expected_ids:
        raise CliError("support commitment does not use the canonical twelve slots")

    buckets: dict[tuple[str, str, int], list[PanelManifest]] = {}
    for panel in task_manifest.panels:
        buckets.setdefault(
            (
                panel.polarity,
                panel.sha256.removeprefix("sha256:"),
                panel.size_bytes,
            ),
            [],
        ).append(panel)
    for values in buckets.values():
        values.sort(key=lambda item: (item.index, item.panel_id))

    result: dict[str, bytes] = {}
    for blob_id in sorted(expected_ids):
        item = by_id[blob_id]
        polarity = "positive" if item.positive else "negative"
        if not blob_id.startswith(f"support-{polarity}-"):
            raise CliError(f"{blob_id!r} has the wrong support polarity")
        identity = (polarity, item.panel.sha256, item.panel.byte_count)
        candidates = buckets.get(identity, [])
        if not candidates:
            raise CliError(
                f"{blob_id}: support identity is absent from the official task"
            )
        panel = candidates.pop(0)
        try:
            payload = panel.path.read_bytes()
        except OSError as exc:
            raise CliError(f"cannot read official support PNG {panel.path}: {exc}") from exc
        if len(payload) != item.panel.byte_count \
                or hashlib.sha256(payload).hexdigest() != item.panel.sha256:
            raise CliError(
                f"{blob_id}: official support PNG changed after manifest build"
            )
        result[blob_id] = payload
    return result


def _bind_record_to_official_corpus(
    record: Mapping[str, Any],
    corpus: ShapeBongardCorpus,
    manifest: object,
) -> TaskManifest:
    if record.get("corpus_manifest_digest") != getattr(manifest, "digest", None):
        raise CliError("run corpus manifest differs from the supplied official corpus")
    if record.get("split_source_digest") != corpus.split.source_digest:
        raise CliError("run split source differs from the supplied official corpus")
    episode = record.get("episode")
    plan = record.get("plan")
    if not isinstance(episode, Mapping) or not isinstance(plan, Mapping):
        raise CliError("run episode and public plan must be objects")
    task_id = episode.get("task_id")
    if not isinstance(task_id, str) or plan.get("task_id") != task_id:
        raise CliError("episode task identity differs from its public plan")
    task_manifest = _official_task_manifest(manifest, task_id)
    if task_manifest.family != episode.get("family") \
            or task_manifest.family != plan.get("family"):
        raise CliError("episode family differs from its official task manifest")
    if plan.get("task_manifest_digest") != task_manifest.digest.removeprefix(
        "sha256:"
    ):
        raise CliError("public plan task manifest digest differs from official corpus")
    assignment = corpus.assignment(task_id)
    for label, actual in (("split", assignment.split), ("regime", assignment.regime)):
        if episode.get(label) != actual or plan.get(label) != actual:
            raise CliError(
                f"episode {label} assignment differs from the official split index"
            )
    return task_manifest


def _verify(args: argparse.Namespace) -> int:
    from bongard.prototype_run_verification import (
        PROTOTYPE_OUTER_RUN_SCHEMA,
        PrototypeRunVerificationError,
        verify_prototype_run_data,
    )
    from bongard.semantic_run_verification import (
        VISUAL_SEMANTIC_OUTER_RUN_SCHEMA,
        VisualSemanticRunVerificationError,
        verify_visual_semantic_run_data,
    )

    run_path = Path(args.run)
    expected_sha256 = args.expected_sha256
    record = _strict_json_bytes(
        run_path,
        expected_sha256=expected_sha256,
    )
    legacy_expected = {
        "schema",
        "corpus_manifest_digest",
        "split_source_digest",
        "official_release",
        "plan",
        "episode",
        "vision",
        "run_archive",
        "exposure",
        "record_digest",
    }
    schema = record.get("schema")
    if schema == PROTOTYPE_OUTER_RUN_SCHEMA:
        expected = {
            "schema",
            "corpus_manifest_digest",
            "split_source_digest",
            "official_release",
            "calibration",
            "plan",
            "support_commitment",
            "episode",
            "prototype",
            "run_archive",
            "exposure",
            "record_digest",
        }
    elif schema == VISUAL_SEMANTIC_OUTER_RUN_SCHEMA:
        expected = {
            "schema",
            "corpus_manifest_digest",
            "split_source_digest",
            "official_release",
            "calibration_campaign_digest",
            "calibration",
            "plan",
            "support_commitment",
            "episode",
            "visual_semantic",
            "run_archive",
            "exposure",
            "record_digest",
        }
    elif schema == RUN_SCHEMA:
        expected = legacy_expected | {"support_commitment"}
    elif schema == LEGACY_OUTER_RUN_SCHEMA:
        expected = legacy_expected
    else:
        expected = set()
    if set(record) != expected:
        raise CliError("run record fields or schema differ")
    content = {key: value for key, value in record.items() if key != "record_digest"}
    if canonical_digest(content) != record["record_digest"]:
        raise CliError("run record digest mismatch")
    release_data = record["official_release"]
    if release_data is not None and not isinstance(release_data, Mapping):
        raise CliError("official_release must be an object or null")
    embedded_release = (
        OfficialReleaseDescriptor.from_dict(release_data)
        if isinstance(release_data, Mapping)
        else None
    )
    if embedded_release is not None:
        if record["corpus_manifest_digest"] != embedded_release.corpus_manifest_sha256:
            raise CliError("run corpus manifest differs from official release")
        if record["split_source_digest"] != embedded_release.split_sha256:
            raise CliError("run split source differs from official release")
    episode = record["episode"]
    if isinstance(episode, Mapping) and episode.get("split") == "test" \
            and embedded_release is None:
        raise CliError("test run lacks an exact official release commitment")
    exposure = _verify_exposure_object(
        record["exposure"],
        corpus_manifest_digest=record["corpus_manifest_digest"],
        episode=record["episode"],
    )
    archive = record["run_archive"]
    if archive is not None and not isinstance(archive, Mapping):
        raise CliError("run_archive must be an object or null")
    if embedded_release is None:
        raise CliError("canonical verify requires an embedded official_release")

    descriptor_path = getattr(args, "release_descriptor", DEFAULT_RELEASE_PATH)
    trusted_release = load_official_release(descriptor_path)
    if embedded_release.to_dict() != trusted_release.to_dict():
        raise CliError("embedded official release differs from the trusted descriptor")
    archive_path = getattr(args, "archive", None)
    corpus_path = getattr(args, "corpus", None)
    if archive_path is None or not str(archive_path).strip():
        raise CliError("canonical verify requires --archive")
    if corpus_path is None or not str(corpus_path).strip():
        raise CliError("canonical verify requires --corpus")
    trusted_release.verify_archive(archive_path)
    corpus = ShapeBongardCorpus.discover(
        corpus_path,
        split_file=getattr(args, "split_file", None),
        require_complete=True,
    )
    manifest = trusted_release.verify_corpus(corpus)
    task_manifest = _bind_record_to_official_corpus(record, corpus, manifest)

    if schema == PROTOTYPE_OUTER_RUN_SCHEMA:
        try:
            support = verify_support_commitment_data(record["support_commitment"])
        except (TypeError, ValueError) as exc:
            raise CliError(f"prototype support commitment is invalid: {exc}") from exc
        if archive is None:
            blob_bytes = _map_official_support_blob_bytes(task_manifest, support)
        else:
            try:
                prototype_archive = verify_archive_data(archive)
            except (TypeError, ValueError) as exc:
                raise CliError(f"prototype run archive is invalid: {exc}") from exc
            blob_bytes = _map_official_task_blob_bytes(
                task_manifest, prototype_archive
            )
        try:
            verified_prototype = verify_prototype_run_data(
                record,
                blob_bytes_by_id=blob_bytes,
            )
        except PrototypeRunVerificationError as exc:
            raise CliError(
                f"cross-layer prototype verification failed: {exc}"
            ) from exc
        output = {
            "verified": True,
            "verification_scope": (
                "exact-official-python-prototype-pixel-reextraction-and-cold-replay"
            ),
            "record_sha256": expected_sha256,
            "external_anchor_verified": True,
            **verified_prototype.to_data(),
            "exposure_ledger_before_digest": exposure["ledger_before_digest"],
            "exposure_ledger_after_digest": exposure["ledger_after_digest"],
            "exposure_event_digest": exposure["event_digest"],
            "exposure_external_anchor": None,
            "official_release_digest": trusted_release.digest,
        }
        sys.stdout.write(json.dumps(output, sort_keys=True) + "\n")
        return 0

    if schema == VISUAL_SEMANTIC_OUTER_RUN_SCHEMA:
        semantic_campaign_path = getattr(
            args, "semantic_calibration_campaign", None
        )
        expected_semantic_campaign_digest = getattr(
            args, "expected_semantic_calibration_campaign_digest", None
        )
        if semantic_campaign_path is None:
            raise CliError(
                "visual-semantic verify requires "
                "--semantic-calibration-campaign"
            )
        if expected_semantic_campaign_digest is None:
            raise CliError(
                "visual-semantic verify requires "
                "--expected-semantic-calibration-campaign-digest"
            )
        semantic_campaign = _load_semantic_calibration_campaign(
            semantic_campaign_path,
            expected_campaign_digest=expected_semantic_campaign_digest,
            corpus=corpus,
            corpus_manifest=manifest,
        )
        try:
            support = verify_support_commitment_data(record["support_commitment"])
        except (TypeError, ValueError) as exc:
            raise CliError(
                f"visual-semantic support commitment is invalid: {exc}"
            ) from exc
        if archive is None:
            blob_bytes = _map_official_support_blob_bytes(task_manifest, support)
        else:
            try:
                semantic_archive = verify_archive_data(archive)
            except (TypeError, ValueError) as exc:
                raise CliError(
                    f"visual-semantic run archive is invalid: {exc}"
                ) from exc
            blob_bytes = _map_official_task_blob_bytes(
                task_manifest, semantic_archive
            )
        try:
            verified_semantic = verify_visual_semantic_run_data(
                record,
                blob_bytes_by_id=blob_bytes,
                calibration_campaign=semantic_campaign,
                expected_task_manifest_digest=(
                    task_manifest.digest.removeprefix("sha256:")
                ),
            )
        except VisualSemanticRunVerificationError as exc:
            raise CliError(
                f"cross-layer visual-semantic verification failed: {exc}"
            ) from exc
        output = {
            "verified": True,
            "verification_scope": (
                "exact-official-visual-semantic-pixel-reextraction-"
                "full-stage-a-campaign-python-atom-and-cold-replay"
            ),
            "record_sha256": expected_sha256,
            "external_anchor_verified": True,
            **verified_semantic.to_data(),
            "exposure_ledger_before_digest": exposure[
                "ledger_before_digest"
            ],
            "exposure_ledger_after_digest": exposure[
                "ledger_after_digest"
            ],
            "exposure_event_digest": exposure["event_digest"],
            "exposure_external_anchor": None,
            "official_release_digest": trusted_release.digest,
        }
        sys.stdout.write(json.dumps(output, sort_keys=True) + "\n")
        return 0

    if archive is None:
        vision = record.get("vision")
        status = episode.get("status") if isinstance(episode, Mapping) else None
        if status == "support_rejected":
            if schema == LEGACY_OUTER_RUN_SCHEMA:
                raise CliError(
                    "legacy v4 support-rejected run lacks the nonce-bearing "
                    "support commitment preimage required for cold verification"
                )
            try:
                support = verify_support_commitment_data(
                    record["support_commitment"]
                )
            except (TypeError, ValueError) as exc:
                raise CliError(f"support commitment is invalid: {exc}") from exc
            support_blob_bytes = _map_official_support_blob_bytes(
                task_manifest,
                support,
            )
            try:
                rejected_gate = verify_support_rejected_run_data(
                    record,
                    blob_bytes_by_id=support_blob_bytes,
                )
            except RunVerificationError as exc:
                raise CliError(
                    f"cross-layer support-rejection verification failed: {exc}"
                ) from exc
            output = {
                "verified": True,
                "verification_scope": (
                    "exact-official-support-rejection-byte-preimage-gate-replay"
                ),
                "record_sha256": expected_sha256,
                "external_anchor_verified": True,
                "run_id": rejected_gate.run_id,
                "support_commitment_digest": (
                    rejected_gate.support_commitment_digest
                ),
                "proposal_digest": rejected_gate.proposal_digest,
                "proposal_freeze_digest": rejected_gate.proposal_freeze_digest,
                "support_gate_digest": rejected_gate.support_gate_digest,
                "support_gate_result": rejected_gate.support_gate_result,
                "exposure_ledger_before_digest": exposure["ledger_before_digest"],
                "exposure_ledger_after_digest": exposure["ledger_after_digest"],
                "exposure_event_digest": exposure["event_digest"],
                "exposure_external_anchor": None,
                "verified_support_preimages": len(
                    rejected_gate.verified_blob_ids
                ),
                "official_release_digest": trusted_release.digest,
            }
            sys.stdout.write(json.dumps(output, sort_keys=True) + "\n")
            return 0
        attempt = (
            vision.get("rejected_proposal_attempt")
            if isinstance(vision, Mapping)
            else None
        )
        if attempt is None:
            raise CliError(
                "proposal failure occurred before a validated structured result; "
                "there is no replayable rejected proposal attempt"
            )
        support_bytes = _map_official_rejected_support_bytes(task_manifest, attempt)
        try:
            rejected = verify_rejected_run_data(
                record, support_bytes_by_name=support_bytes
            )
        except RunVerificationError as exc:
            raise CliError(
                f"cross-layer rejected-proposal verification failed: {exc}"
            ) from exc
        output = {
            "verified": True,
            "verification_scope": (
                "exact-official-rejected-proposal-byte-preimage-parser-replay"
            ),
            "record_sha256": expected_sha256,
            "external_anchor_verified": True,
            "run_id": rejected.run_id,
            "attempt_digest": rejected.attempt_digest,
            "receipt_digest": rejected.receipt_digest,
            "exposure_ledger_before_digest": exposure["ledger_before_digest"],
            "exposure_ledger_after_digest": exposure["ledger_after_digest"],
            "exposure_event_digest": exposure["event_digest"],
            "exposure_external_anchor": None,
            "verified_support_preimages": len(
                rejected.verified_support_preimages
            ),
            "unbound_outer_fields": list(rejected.unbound_outer_fields),
            "official_release_digest": trusted_release.digest,
        }
        sys.stdout.write(json.dumps(output, sort_keys=True) + "\n")
        return 0

    verified_archive = verify_archive_data(archive)
    blob_bytes = _map_official_task_blob_bytes(task_manifest, verified_archive)
    try:
        verified = verify_completed_run_data(
            record, blob_bytes_by_id=blob_bytes
        )
    except RunVerificationError as exc:
        raise CliError(f"cross-layer run verification failed: {exc}") from exc
    output = {
        "verified": True,
        "verification_scope": "exact-official-cross-layer-byte-preimage-replay",
        "record_sha256": expected_sha256,
        "external_anchor_verified": True,
        "run_id": verified.run_id,
        "archive_digest": verified.archive_digest,
        "chain_digest": verified.archive.replay_receipt.chain_digest,
        "determinate_correct": verified.archive.replay_receipt.determinate_correct,
        "determinate_total": verified.archive.replay_receipt.determinate_total,
        "abstentions": verified.archive.replay_receipt.abstentions,
        "exposure_ledger_before_digest": exposure["ledger_before_digest"],
        "exposure_ledger_after_digest": exposure["ledger_after_digest"],
        "exposure_event_digest": exposure["event_digest"],
        "exposure_external_anchor": None,
        "verified_blob_preimages": len(verified.verified_blob_ids),
        "official_release_digest": trusted_release.digest,
    }
    sys.stdout.write(json.dumps(output, sort_keys=True) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bongard")
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser("inventory", help="validate and hash a corpus")
    inventory.add_argument("--corpus", required=True)
    inventory.add_argument("--split-file")
    inventory.add_argument("--require-complete", action="store_true")
    inventory.add_argument("--official-release", action="store_true")
    inventory.add_argument("--archive")
    inventory.add_argument(
        "--release-descriptor",
        default=str(DEFAULT_RELEASE_PATH),
    )
    inventory.add_argument("--out")
    inventory.set_defaults(handler=_inventory)

    action_programs = commands.add_parser(
        "audit-action-programs",
        help=(
            "read-only audit of privileged post-hoc/oracle action-program metadata"
        ),
    )
    action_programs.add_argument("--corpus", required=True)
    action_programs.add_argument("--split-file")
    action_programs.add_argument("--ff-action-programs", required=True)
    action_programs.add_argument("--bd-action-programs", required=True)
    action_programs.add_argument("--hd-action-programs", required=True)
    action_programs.add_argument(
        "--release-descriptor",
        default=str(DEFAULT_RELEASE_PATH),
    )
    action_programs.add_argument(
        "--expected-report-digest",
        help="expected internal sha256: content address of the regenerated report",
    )
    action_programs.add_argument(
        "--expected-report",
        help="existing canonical report file that must exactly match regeneration",
    )
    action_programs.add_argument(
        "--out",
        help="write the canonical report once; existing files are never overwritten",
    )
    action_programs.set_defaults(handler=_audit_action_programs)

    cohorts = commands.add_parser(
        "cohorts", help="report historically unused official task cohorts"
    )
    cohorts.add_argument("--corpus", required=True)
    cohorts.add_argument("--split-file")
    cohorts.add_argument("--require-complete", action="store_true")
    cohorts.add_argument(
        "--split", choices=("train", "val", "test", "FF", "BA", "CM", "NV")
    )
    cohorts.add_argument("--family", choices=("ff", "bd", "hd"))
    cohorts.add_argument(
        "--cohort",
        choices=("clean", "drill", "dev", "sealed"),
    )
    cohorts.add_argument("--limit", type=int, default=20)
    cohorts.add_argument(
        "--ledger-in",
        help=(
            "optional live exposure ledger; exact-task and semantic-key "
            "collisions are excluded from selected_task_ids"
        ),
    )
    cohorts.add_argument("--out")
    cohorts.set_defaults(handler=_cohorts)

    pipelines = commands.add_parser(
        "pipeline-status",
        help="show active, retained, audit-only, and retired Bongard paths",
    )
    pipelines.set_defaults(handler=_pipeline_status)

    stage_a = commands.add_parser(
        "calibrate-semantic-stage-a",
        help=(
            "run the official descriptive-only visual-semantic calibration "
            "with a durable pre-model semantic-exposure precommit"
        ),
    )
    stage_a.add_argument("--corpus", required=True)
    stage_a.add_argument("--split-file")
    stage_a.add_argument("--archive", required=True)
    stage_a.add_argument(
        "--release-descriptor",
        default=str(DEFAULT_RELEASE_PATH),
    )
    stage_a.add_argument("--ledger-in", required=True)
    stage_a.add_argument("--expected-ledger-digest", required=True)
    stage_a.add_argument("--selection-seed", required=True)
    stage_a.add_argument("--selection-seed-provenance", required=True)
    stage_a.add_argument("--artifact-dir", required=True)
    stage_a.add_argument("--exposure-dir", required=True)
    stage_a.add_argument("--private-cache-dir", required=True)
    stage_a.add_argument(
        "--expected-codex-launcher-sha256",
        required=True,
        help="externally recorded SHA-256 of the fixed `codex` launcher",
    )
    stage_a.add_argument("--candidate-count", type=int, default=48)
    stage_a.add_argument(
        "--minimum-clusters-per-bin",
        type=int,
        default=12,
        help=(
            "frozen minimum distinct dependence clusters required in every "
            "calibration score bin"
        ),
    )
    stage_a.add_argument("--model", default="gpt-5.6-sol")
    stage_a.add_argument("--reasoning-effort", default="medium")
    stage_a.add_argument("--proposer-minutes", type=int, default=15)
    stage_a.add_argument("--scorer-minutes", type=int, default=10)
    stage_a.add_argument("--workers", type=int, default=4)
    stage_a.add_argument("--verbose", action="store_true")
    stage_a.set_defaults(
        handler=_retired_pipeline_command,
        retired_pipeline_id="legacy-visual-semantic-calibration-cli-v1",
    )

    stage_b = commands.add_parser(
        "validate-semantic-stage-b",
        help=(
            "run the strict-disjoint descriptive DEV validation from one "
            "successful Stage-A command receipt"
        ),
    )
    stage_b.add_argument("--corpus", required=True)
    stage_b.add_argument("--split-file")
    stage_b.add_argument("--archive", required=True)
    stage_b.add_argument(
        "--release-descriptor",
        default=str(DEFAULT_RELEASE_PATH),
    )
    stage_b.add_argument("--stage-a-command-receipt", required=True)
    stage_b.add_argument(
        "--expected-stage-a-command-receipt-digest",
        required=True,
    )
    stage_b.add_argument("--selection-seed", required=True)
    stage_b.add_argument("--selection-seed-provenance", required=True)
    stage_b.add_argument("--exposure-observed-at", required=True)
    stage_b.add_argument("--artifact-dir", required=True)
    stage_b.add_argument("--exposure-dir", required=True)
    stage_b.add_argument("--task-count", type=int, default=24)
    stage_b.add_argument("--workers", type=int, default=4)
    stage_b.add_argument("--verbose", action="store_true")
    stage_b.set_defaults(
        handler=_retired_pipeline_command,
        retired_pipeline_id="legacy-visual-semantic-calibration-cli-v1",
    )

    run = commands.add_parser(
        "run",
        help="retired legacy two-query launcher; use verify for old records",
    )
    run.add_argument("--corpus", required=True)
    run.add_argument("--split-file")
    run.add_argument("--task-id", required=True)
    run.add_argument("--seed", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--exposure-dir", required=True)
    run.add_argument("--ledger-in")
    run.add_argument("--require-unseen", action="store_true")
    run.add_argument("--cohort", choices=("drill", "dev", "sealed"))
    run.add_argument("--sealed-test", action="store_true")
    run.add_argument("--official-release", action="store_true")
    run.add_argument("--archive")
    run.add_argument(
        "--release-descriptor",
        default=str(DEFAULT_RELEASE_PATH),
    )
    run.add_argument("--model", default=DEFAULT_CODEX_MODEL)
    run.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    run.add_argument(
        "--predicate-mode",
        choices=("hybrid", "support-prototype", "visual-semantic"),
        default="hybrid",
        help=(
            "HYBRID uses model support/query judgments; support-prototype uses "
            "one Codex catalog selection and deterministic Python evaluation; "
            "visual-semantic uses typed direct predicates plus an optional "
            "calibrated blind soft scorer with Python-authoritative replay"
        ),
    )
    run.add_argument(
        "--prototype-calibration",
        help=(
            "canonical development calibration record; required only for "
            "--predicate-mode support-prototype"
        ),
    )
    run.add_argument(
        "--semantic-calibration-campaign",
        help=(
            "canonical full Stage-A campaign; required only for "
            "--predicate-mode visual-semantic"
        ),
    )
    run.add_argument(
        "--expected-semantic-calibration-campaign-digest",
        help=(
            "externally recorded canonical content digest of the full Stage-A "
            "campaign; required for --predicate-mode visual-semantic"
        ),
    )
    run.add_argument("--proposer-minutes", type=int, default=15)
    run.add_argument("--observer-minutes", type=int, default=10)
    run.add_argument(
        "--expected-codex-launcher-sha256",
        help=(
            "externally recorded SHA-256 of the fixed `codex` launcher; "
            "required for official-release and visual-semantic runs"
        ),
    )
    run.add_argument("--verbose", action="store_true")
    run.set_defaults(
        handler=_retired_pipeline_command,
        retired_pipeline_id="legacy-two-query-episode-cli-v1",
    )

    verify = commands.add_parser("verify", help="cold-verify one saved run")
    verify.add_argument("--run", required=True)
    verify.add_argument("--corpus", required=True)
    verify.add_argument("--split-file")
    verify.add_argument("--archive", required=True)
    verify.add_argument(
        "--release-descriptor",
        default=str(DEFAULT_RELEASE_PATH),
    )
    verify.add_argument(
        "--expected-sha256",
        required=True,
        help="externally recorded SHA-256 of the canonical run file",
    )
    verify.add_argument(
        "--semantic-calibration-campaign",
        help="canonical full Stage-A campaign used by a visual-semantic run",
    )
    verify.add_argument(
        "--expected-semantic-calibration-campaign-digest",
        help=(
            "externally recorded canonical content digest of that full "
            "Stage-A campaign"
        ),
    )
    verify.set_defaults(handler=_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (CliError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

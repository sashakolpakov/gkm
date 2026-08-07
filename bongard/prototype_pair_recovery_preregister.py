"""Metadata-only, same-seed recovery preregistration for a failed campaign.

The recovery policy is deliberately narrow.  It accepts one already verified
prototype-pair preregistration/plan, an exposure successor that appends exactly
that plan's 31 task IDs, and externally committed failure facts.  It reuses the
old seed and namespace without an override, so the successor ledger is the only
input that can change selection.  No panel, archive, model, or campaign store is
accepted or opened.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.exposure import ExposureLedger, semantic_resolver_policy_digest
from bongard.historical_exposure import load_historical_exposure
from bongard.prototype_pair_campaign_cli import (
    PREREGISTRATION_SCHEMA,
    PREREGISTRATION_SCOPE,
    _stable_read_regular,
    _strict_json_object,
    _task_inventory_from_split,
    verify_prototype_pair_campaign_metadata,
)
from bongard.prototype_pair_cohort import (
    OFFICIAL_BASIC_GENERATOR_SHA256,
    OFFICIAL_BASIC_SAMPLER_SHA256,
    OPAQUE_TAG_IDS,
    PrototypePairCohortPlan,
    plan_prototype_pair_cohort,
    prototype_pair_seed_commitment,
)
from bongard.release import OfficialReleaseDescriptor


RECOVERY_POLICY_ID = (
    "bongard.prototype-pair-recovery/same-seed-next-exact-unused-v1"
)
EXPECTED_FAILED_CAMPAIGN_STATUS = "description_gap"
EXPECTED_FAILED_OBSERVER_STATUS = "transport_error"
EXPECTED_RELEASE_PHASE = "prototype_pair_selected_task_release"
RECOVERY_GENERATOR_SOURCE_SHA256 = hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class PrototypePairRecoveryError(RuntimeError):
    """A recovery input, chain, selection, or output invariant failed."""


@dataclass(frozen=True, slots=True)
class PrototypePairRecoveryArtifacts:
    predecessor_path: Path
    cohort_plan_path: Path
    preregistration_path: Path
    predecessor_digest: str
    cohort_plan_digest: str
    preregistration_digest: str
    drill_task_id: str
    prototype_task_ids: tuple[str, str]
    selected_task_ids: tuple[str, ...]


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise PrototypePairRecoveryError(f"{label} must be a sha256: address")
    return value


def _raw_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_SHA256.fullmatch(value) is None:
        raise PrototypePairRecoveryError(f"{label} must be lowercase SHA-256")
    return value


def _utc_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PrototypePairRecoveryError(f"{label} must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise PrototypePairRecoveryError(f"{label} is not an ISO timestamp") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise PrototypePairRecoveryError(f"{label} is not UTC")
    return parsed


def _load_mapping(path: str | Path, *, label: str) -> Mapping[str, Any]:
    _source, payload = _stable_read_regular(path, label=label)
    return _strict_json_object(payload, label=label)


def _exclusive_write(path_value: str | Path, payload: bytes, *, label: str) -> Path:
    if not isinstance(payload, bytes) or not payload:
        raise PrototypePairRecoveryError(f"{label} payload must be nonempty bytes")
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(path_value))))
    parent = path.parent
    try:
        parent_info = parent.lstat()
    except OSError as exc:
        raise PrototypePairRecoveryError(
            f"{label} parent directory does not exist: {parent}"
        ) from exc
    if not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(parent_info.st_mode):
        raise PrototypePairRecoveryError(
            f"{label} parent must be a real directory: {parent}"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise PrototypePairRecoveryError(f"{label} already exists: {path}") from exc
    except OSError as exc:
        raise PrototypePairRecoveryError(f"cannot create {label}: {path}") from exc
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise PrototypePairRecoveryError(f"short write for {label}")
            written += count
        os.fsync(descriptor)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size != len(payload):
            raise PrototypePairRecoveryError(f"{label} durable size differs")
    finally:
        os.close(descriptor)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    directory = os.open(parent, directory_flags)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    _reloaded_path, reloaded = _stable_read_regular(path, label=label)
    if reloaded != payload:
        raise PrototypePairRecoveryError(f"{label} durable reload differs")
    return path


def _recovery_provenance(
    *,
    prior_preregistration_digest: str,
    prior_plan_digest: str,
    failed_campaign_digest: str,
    successor_exposure_digest: str,
) -> str:
    return ";".join(
        (
            RECOVERY_POLICY_ID,
            f"prior_preregistration_digest={prior_preregistration_digest}",
            f"prior_plan_digest={prior_plan_digest}",
            f"failed_campaign_digest={failed_campaign_digest}",
            f"failed_campaign_status={EXPECTED_FAILED_CAMPAIGN_STATUS}",
            f"failed_observer_status={EXPECTED_FAILED_OBSERVER_STATUS}",
            f"successor_exposure_digest={successor_exposure_digest}",
            "seed_and_namespace_reused=true",
            "prior_attempt_retained_in_denominator=true",
            f"generator_source_sha256={RECOVERY_GENERATOR_SOURCE_SHA256}",
        )
    )


def _preregistration(
    *,
    created_at: str,
    old_preregistration_digest: str,
    old_plan: PrototypePairCohortPlan,
    plan: PrototypePairCohortPlan,
    selection_seed: str,
    failed_campaign_digest: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": PREREGISTRATION_SCHEMA,
        "created_at": created_at,
        "scope": PREREGISTRATION_SCOPE,
        "seed": {
            "value": selection_seed,
            "provenance": _recovery_provenance(
                prior_preregistration_digest=old_preregistration_digest,
                prior_plan_digest=old_plan.record_digest,
                failed_campaign_digest=failed_campaign_digest,
                successor_exposure_digest=plan.exposure_predecessor_digest,
            ),
            "namespace": plan.namespace,
            "commitment": plan.selection_seed_commitment,
        },
        "source": {
            "release_descriptor_digest": plan.release_descriptor_digest,
            "corpus_manifest_digest": plan.corpus_manifest_digest,
            "split_source_digest": plan.split_source_digest,
            "task_inventory_digest": plan.task_inventory_digest,
            "historical_seed_digest": plan.historical_seed_digest,
            "exposure_predecessor_digest": plan.exposure_predecessor_digest,
        },
        "planner": {
            "algorithm_id": plan.algorithm_id,
            "source_sha256": plan.planner_source_sha256,
            "algorithm_digest": plan.planner_algorithm_digest,
        },
        "selection": {
            "candidate_count": len(plan.candidates),
            "selected_task_count": len(plan.selected_task_ids),
            "drill_task_id": plan.drill.task_id,
            "drill_shape_families": list(plan.drill.ordered_shapes),
            "plan_digest": plan.record_digest,
        },
        "statistics": {
            "opaque_tag_count": len(OPAQUE_TAG_IDS),
            "calibration_task_clusters_per_tag": plan.clusters_per_hypothesis,
            "hypothesis_count": plan.hypothesis_count,
            "confidence_level_ppm": plan.confidence_level_ppm,
            "zero_error_family_upper_ppm": plan.zero_error_family_upper_ppm,
            "targeted_engineering_tolerance_ppm": (
                plan.targeted_engineering_tolerance_ppm
            ),
            "zero_errors_required": plan.zero_errors_required_for_tolerance,
            "stronger_250k_claim_authorized": (
                plan.stronger_250k_claim_authorized
            ),
        },
        "execution": {
            "metadata_only_selection": True,
            "panel_bytes_opened_before_preregistration": False,
            "action_program_json_authorized": False,
            "thresholds_must_be_frozen_before_calibration": True,
            "formula_must_be_frozen_before_query_pixels": True,
            "cold_replay_must_be_model_free": True,
            "official_test_authorized": False,
        },
        "authority": {
            "predicate_authority_id": plan.predicate_authority_id,
            "python_is_canonical_authority": True,
            "lean_required": False,
            "lean_defines_artifact_identity": False,
            "lean_affects_selection_or_decision": False,
            "optional_secondary_checker_detachable": True,
        },
        "claims": {
            "targeted_engineering_only": True,
            "semantics_reused": True,
            "benchmark_claim_authorized": False,
            "unseen_claim_authorized": False,
        },
    }
    digest = canonical_digest(body)
    return {**body, "record_digest": digest}


def generate_prototype_pair_recovery_preregistration(
    *,
    old_preregistration_path: str | Path,
    expected_old_preregistration_digest: str,
    old_cohort_plan_path: str | Path,
    expected_old_cohort_plan_digest: str,
    old_exposure_predecessor_path: str | Path,
    successor_exposure_object_path: str | Path,
    expected_successor_exposure_digest: str,
    failed_campaign_digest: str,
    failed_campaign_status: str,
    failed_observer_status: str,
    release_descriptor_path: str | Path,
    split_path: str | Path,
    historical_seed_path: str | Path,
    output_exposure_predecessor_path: str | Path,
    output_cohort_plan_path: str | Path,
    output_preregistration_path: str | Path,
    created_at: str,
) -> PrototypePairRecoveryArtifacts:
    """Create and cold-verify one no-reroll successor preregistration."""

    prior_preregistration_digest = _raw_sha256(
        expected_old_preregistration_digest,
        "old preregistration digest",
    )
    prior_plan_digest = _address(
        expected_old_cohort_plan_digest,
        "old cohort plan digest",
    )
    successor_pin = _address(
        expected_successor_exposure_digest,
        "successor exposure digest",
    )
    failed_digest = _address(failed_campaign_digest, "failed campaign digest")
    if failed_campaign_status != EXPECTED_FAILED_CAMPAIGN_STATUS:
        raise PrototypePairRecoveryError("failed campaign status is not description_gap")
    if failed_observer_status != EXPECTED_FAILED_OBSERVER_STATUS:
        raise PrototypePairRecoveryError("failed observer status is not transport_error")
    recovery_time = _utc_timestamp(created_at, "recovery created_at")

    old_metadata = verify_prototype_pair_campaign_metadata(
        preregistration_path=old_preregistration_path,
        expected_preregistration_digest=prior_preregistration_digest,
        cohort_plan_path=old_cohort_plan_path,
        release_descriptor_path=release_descriptor_path,
        split_path=split_path,
        historical_seed_path=historical_seed_path,
        exposure_predecessor_path=old_exposure_predecessor_path,
    )
    old_plan = old_metadata.cohort_plan
    if (
        old_plan.record_digest != prior_plan_digest
        or len(old_plan.selected_task_ids) != 31
    ):
        raise PrototypePairRecoveryError("old plan differs from its external pin")

    successor_raw = _load_mapping(
        successor_exposure_object_path,
        label="successor exposure object",
    )
    successor = ExposureLedger.from_dict(successor_raw)
    if successor.digest != successor_pin:
        raise PrototypePairRecoveryError("successor exposure differs from its pin")
    predecessor = old_metadata.exposure_predecessor
    if (
        len(successor.events) != len(predecessor.events) + 1
        or successor.events[:-1] != predecessor.events
    ):
        raise PrototypePairRecoveryError(
            "successor exposure is not exactly one event after the old predecessor"
        )
    release_event = successor.events[-1]
    if (
        release_event.phase != EXPECTED_RELEASE_PHASE
        or release_event.panel_ids
        or release_event.task_ids != old_plan.selected_task_ids
    ):
        raise PrototypePairRecoveryError(
            "successor release event does not expose exactly the old selected tasks"
        )
    if recovery_time <= _utc_timestamp(
        release_event.observed_at,
        "successor release observed_at",
    ):
        raise PrototypePairRecoveryError(
            "recovery preregistration must postdate the successor release event"
        )

    release_raw = _load_mapping(release_descriptor_path, label="release descriptor")
    release = OfficialReleaseDescriptor.from_dict(release_raw)
    _split_source, split_bytes = _stable_read_regular(
        split_path,
        label="official split",
        byte_limit=16 * 1024 * 1024,
    )
    task_ids = _task_inventory_from_split(
        split_bytes,
        expected_task_inventory_digest=old_metadata.pins.task_inventory_digest,
        release_task_inventory_digest=release.task_ids_sha256,
    )
    historical_source, historical_payload = _stable_read_regular(
        historical_seed_path,
        label="historical seed",
    )
    historical = load_historical_exposure(historical_source, verify_evidence=False)
    _historical_after, historical_payload_after = _stable_read_regular(
        historical_source,
        label="historical seed",
    )
    if historical_payload_after != historical_payload:
        raise PrototypePairRecoveryError("historical seed changed during recovery")

    seed = old_metadata.pins.selection_seed
    namespace = old_metadata.pins.namespace
    if prototype_pair_seed_commitment(seed, namespace=namespace) != (
        old_metadata.pins.seed_commitment
    ):
        raise PrototypePairRecoveryError("old seed commitment does not replay")
    plan = plan_prototype_pair_cohort(
        release_descriptor=release,
        split_bytes=split_bytes,
        task_ids=task_ids,
        exposure_predecessor=successor,
        historical_seed=historical,
        selection_seed=seed,
        expected_seed_commitment=old_metadata.pins.seed_commitment,
        expected_release_descriptor_digest=old_metadata.pins.release_descriptor_digest,
        expected_corpus_manifest_digest=old_metadata.pins.corpus_manifest_digest,
        expected_split_source_digest=old_metadata.pins.split_source_digest,
        expected_task_inventory_digest=old_metadata.pins.task_inventory_digest,
        expected_exposure_predecessor_digest=successor_pin,
        expected_historical_seed_digest=old_metadata.pins.historical_seed_digest,
        expected_resolver_policy_digest=semantic_resolver_policy_digest(historical),
        expected_basic_sampler_sha256=OFFICIAL_BASIC_SAMPLER_SHA256,
        expected_basic_generator_sha256=OFFICIAL_BASIC_GENERATOR_SHA256,
        namespace=namespace,
    )
    old_selected = set(old_plan.selected_task_ids)
    if (
        plan.namespace != old_plan.namespace
        or plan.selection_seed_digest != old_plan.selection_seed_digest
        or plan.selection_seed_commitment != old_plan.selection_seed_commitment
        or plan.exposure_predecessor_digest != successor_pin
        or len(plan.selected_task_ids) != 31
        or old_selected & set(plan.selected_task_ids)
        or not old_selected.issubset(
            set(plan.excluded_exact_used_train_basic_task_ids)
        )
    ):
        raise PrototypePairRecoveryError(
            "recovery plan changed the seed/namespace or reused exposed tasks"
        )

    preregistration = _preregistration(
        created_at=created_at,
        old_preregistration_digest=prior_preregistration_digest,
        old_plan=old_plan,
        plan=plan,
        selection_seed=seed,
        failed_campaign_digest=failed_digest,
    )
    predecessor_payload = successor.to_json().encode("utf-8")
    plan_payload = canonical_json(plan.to_data()) + b"\n"
    preregistration_payload = canonical_json(preregistration) + b"\n"

    predecessor_output = _exclusive_write(
        output_exposure_predecessor_path,
        predecessor_payload,
        label="recovery exposure predecessor",
    )
    plan_output = _exclusive_write(
        output_cohort_plan_path,
        plan_payload,
        label="recovery cohort plan",
    )
    preregistration_output = _exclusive_write(
        output_preregistration_path,
        preregistration_payload,
        label="recovery preregistration",
    )

    verified = verify_prototype_pair_campaign_metadata(
        preregistration_path=preregistration_output,
        expected_preregistration_digest=preregistration["record_digest"],
        cohort_plan_path=plan_output,
        release_descriptor_path=release_descriptor_path,
        split_path=split_path,
        historical_seed_path=historical_seed_path,
        exposure_predecessor_path=predecessor_output,
    )
    if verified.cohort_plan != plan or verified.exposure_predecessor != successor:
        raise PrototypePairRecoveryError("recovery metadata cold verification differs")

    prototype_ids = tuple(item.task_id for item in plan.prototypes)
    if len(prototype_ids) != 2:
        raise PrototypePairRecoveryError("recovery plan does not have two prototypes")
    return PrototypePairRecoveryArtifacts(
        predecessor_path=predecessor_output,
        cohort_plan_path=plan_output,
        preregistration_path=preregistration_output,
        predecessor_digest=successor.digest,
        cohort_plan_digest=plan.record_digest,
        preregistration_digest=preregistration["record_digest"],
        drill_task_id=plan.drill.task_id,
        prototype_task_ids=(prototype_ids[0], prototype_ids[1]),
        selected_task_ids=plan.selected_task_ids,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create one same-seed metadata-only prototype-pair recovery"
    )
    for flag in (
        "old-preregistration",
        "old-cohort-plan",
        "old-exposure-predecessor",
        "successor-exposure-object",
        "release-descriptor",
        "split",
        "historical-seed",
        "output-exposure-predecessor",
        "output-cohort-plan",
        "output-preregistration",
    ):
        parser.add_argument(f"--{flag}", required=True)
    parser.add_argument("--expected-old-preregistration-digest", required=True)
    parser.add_argument("--expected-old-cohort-plan-digest", required=True)
    parser.add_argument("--expected-successor-exposure-digest", required=True)
    parser.add_argument("--failed-campaign-digest", required=True)
    parser.add_argument(
        "--failed-campaign-status",
        required=True,
        choices=(EXPECTED_FAILED_CAMPAIGN_STATUS,),
    )
    parser.add_argument(
        "--failed-observer-status",
        required=True,
        choices=(EXPECTED_FAILED_OBSERVER_STATUS,),
    )
    parser.add_argument("--created-at", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = generate_prototype_pair_recovery_preregistration(
        old_preregistration_path=args.old_preregistration,
        expected_old_preregistration_digest=(
            args.expected_old_preregistration_digest
        ),
        old_cohort_plan_path=args.old_cohort_plan,
        expected_old_cohort_plan_digest=args.expected_old_cohort_plan_digest,
        old_exposure_predecessor_path=args.old_exposure_predecessor,
        successor_exposure_object_path=args.successor_exposure_object,
        expected_successor_exposure_digest=(
            args.expected_successor_exposure_digest
        ),
        failed_campaign_digest=args.failed_campaign_digest,
        failed_campaign_status=args.failed_campaign_status,
        failed_observer_status=args.failed_observer_status,
        release_descriptor_path=args.release_descriptor,
        split_path=args.split,
        historical_seed_path=args.historical_seed,
        output_exposure_predecessor_path=args.output_exposure_predecessor,
        output_cohort_plan_path=args.output_cohort_plan,
        output_preregistration_path=args.output_preregistration,
        created_at=args.created_at,
    )
    summary = {
        "cohort_plan_digest": result.cohort_plan_digest,
        "drill_task_id": result.drill_task_id,
        "exposure_predecessor_digest": result.predecessor_digest,
        "preregistration_digest": result.preregistration_digest,
        "prototype_task_ids": list(result.prototype_task_ids),
        "selected_task_count": len(result.selected_task_ids),
    }
    os.write(1, canonical_json(summary) + b"\n")
    return 0


__all__ = (
    "EXPECTED_FAILED_CAMPAIGN_STATUS",
    "EXPECTED_FAILED_OBSERVER_STATUS",
    "PrototypePairRecoveryArtifacts",
    "PrototypePairRecoveryError",
    "RECOVERY_GENERATOR_SOURCE_SHA256",
    "RECOVERY_POLICY_ID",
    "generate_prototype_pair_recovery_preregistration",
    "main",
)


if __name__ == "__main__":  # pragma: no cover - exercised through CLI tests
    raise SystemExit(main())

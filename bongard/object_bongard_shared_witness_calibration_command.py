"""Sealed two-pass calibration for structured shared-witness predicates.

The command consumes one cold-verified accepted structured nomination and the
exact twelve already-exposed historical panels.  It executes two independent
passes over both ranked predicates and all panels (2 x 2 x 12 = 48 journaled
calls), durably freezes and reloads every artifact before support sides are
introduced, then admits only the lowest rank accepted in both passes with no
confident cross-pass polarity flip.  Query, broad-cohort, and official-test
pixels are outside this command.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import (
    CodexNoToolsAttestation,
    attest_codex_no_tools,
)
from bongard.evidence import Disposition
from bongard import object_bongard_rubric_nomination_command as _durable
from bongard.object_bongard_panel_rubric_calibration import (
    DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE,
    ObjectBongardPanelRubricCalibrationPanel,
    ObjectBongardPanelRubricCalibrationSource,
    load_object_bongard_panel_rubric_calibration_source,
    object_bongard_panel_rubric_calibration_source_digest,
)
from bongard.object_bongard_shared_witness import (
    ObjectBongardSharedWitnessRubricSpec,
    build_shared_witness_rubric_specs,
    object_bongard_shared_witness_source_digest,
)
from bongard.object_bongard_shared_witness_nomination_command import (
    VerifiedObjectBongardSharedWitnessNomination,
    object_bongard_shared_witness_nomination_command_source_digest,
    verify_object_bongard_shared_witness_nomination,
)
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTurnRuntime,
    object_bongard_turn_journal_source_digest,
    verify_object_bongard_turn_journal,
)
from bongard.prototype_object_scene_observer import (
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


AUTHORIZATION_SCHEMA = "gkm.bongard-shared-witness-calibration-authorization.v1"
PRECOMMIT_SCHEMA = "gkm.bongard-shared-witness-calibration-precommit.v1"
BATCH_SCHEMA = "gkm.bongard-shared-witness-calibration-blind-batch.v1"
FREEZE_SCHEMA = "gkm.bongard-shared-witness-calibration-freeze.v1"
ASSESSMENT_SCHEMA = "gkm.bongard-shared-witness-calibration-assessment.v1"
REPLAY_SCHEMA = "gkm.bongard-shared-witness-calibration-cold-replay.v1"
RESULT_SCHEMA = "gkm.bongard-shared-witness-calibration-result.v1"
COMMAND_ID = "bongard.shared-witness-calibration/two-pass-48-turn-v1"

AUTHORIZATION_FILENAME = "authorization.json"
PRECOMMIT_FILENAME = "execution_precommit.json"
BATCH_FILENAME = "blind_observation_batch.json"
FREEZE_FILENAME = "durable_freeze.json"
ASSESSMENT_FILENAME = "assessment.json"
REPLAY_FILENAME = "cold_replay.json"
RESULT_FILENAME = "result.json"
JOURNAL_DIRECTORY = "journals"

CALIBRATION_MODEL = "gpt-5.6-sol"
CALIBRATION_REASONING_EFFORT = "medium"
DEFAULT_MINUTES = 15
DEFAULT_EXECUTABLE = "codex"
DEFAULT_EXPECTED_LAUNCHER_SHA256 = (
    "19c4f144c5226a9f17c58e6f0fa854843b0f77a6eb420f40e2745a12f10f5d37"
)
DEFAULT_STRUCTURED_NOMINATION_ROOT = Path(
    "downloads/ShapeBongard_V2_full/"
    "object_shared_witness_nomination_20260808_v2"
)
DEFAULT_PARALLEL_WORKERS = 4
MAX_PARALLEL_WORKERS = 4
CALIBRATION_PASS_COUNT = 2
CALIBRATION_RANK_COUNT = 2
CALIBRATION_PANEL_COUNT = 12
CALIBRATION_JOB_COUNT = 48

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")

Transport = Callable[..., CodexStructuredResult]


class ObjectBongardSharedWitnessCalibrationCommandError(RuntimeError):
    """A calibration parent, runtime, batch, or replay differs."""


@dataclass(frozen=True, slots=True)
class _CalibrationInputs:
    source: ObjectBongardPanelRubricCalibrationSource
    nomination: VerifiedObjectBongardSharedWitnessNomination
    specs: tuple[
        ObjectBongardSharedWitnessRubricSpec,
        ObjectBongardSharedWitnessRubricSpec,
    ]


@dataclass(frozen=True, slots=True)
class _CalibrationJob:
    pass_index: int
    panel: ObjectBongardPanelRubricCalibrationPanel
    spec: ObjectBongardSharedWitnessRubricSpec


@dataclass(frozen=True, slots=True)
class VerifiedObjectBongardSharedWitnessCalibration:
    output_root: Path
    nomination_authorization_digest: str
    nomination_execution_precommit_digest: str
    nomination_result_digest: str
    nomination_replay_digest: str
    nomination_artifact_digest: str
    source_digest: str
    authorization_digest: str
    execution_precommit_digest: str
    batch_digest: str
    freeze_digest: str
    assessment_digest: str
    replay_digest: str
    result_digest: str
    accepted: bool
    selected_candidate_rank: int | None
    selected_spec_digest: str | None
    fresh_call_count: int
    reused_call_count: int

    def summary_data(self) -> dict[str, object]:
        return {
            "schema": "gkm.bongard-shared-witness-calibration-summary.v1",
            "output_root": str(self.output_root),
            "nomination_authorization_digest": (
                self.nomination_authorization_digest
            ),
            "nomination_execution_precommit_digest": (
                self.nomination_execution_precommit_digest
            ),
            "nomination_result_digest": self.nomination_result_digest,
            "nomination_replay_digest": self.nomination_replay_digest,
            "nomination_artifact_digest": self.nomination_artifact_digest,
            "source_digest": self.source_digest,
            "authorization_digest": self.authorization_digest,
            "execution_precommit_digest": self.execution_precommit_digest,
            "batch_digest": self.batch_digest,
            "freeze_digest": self.freeze_digest,
            "assessment_digest": self.assessment_digest,
            "cold_replay_digest": self.replay_digest,
            "result_digest": self.result_digest,
            "accepted": self.accepted,
            "selected_candidate_rank": self.selected_candidate_rank,
            "selected_spec_digest": self.selected_spec_digest,
            "fresh_call_count": self.fresh_call_count,
            "reused_call_count": self.reused_call_count,
            **_authority_data(),
        }


def object_bongard_shared_witness_calibration_command_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "structured_shared_witness_predicates_only": True,
        "independent_pass_count": CALIBRATION_PASS_COUNT,
        "rank_count": CALIBRATION_RANK_COUNT,
        "historical_exposed_panel_count": CALIBRATION_PANEL_COUNT,
        "physical_model_call_count": CALIBRATION_JOB_COUNT,
        "all_artifacts_frozen_before_support_labels": True,
        "minimum_expected_definite_per_side_per_pass": 5,
        "maximum_indeterminate_per_side_per_pass": 1,
        "confident_contradiction_or_error_allowed_per_pass": False,
        "cross_pass_present_absent_flip_allowed": False,
        "selection_rule": "lowest-rank-accepted-in-both-passes",
        "model_can_choose_operator_threshold_or_polarity": False,
        "negation_allowed": False,
        "polarity_flip_allowed": False,
        "query_pixels_used": False,
        "fresh_broad_cohort_pixels_used": False,
        "official_test_pixels_used": False,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_defines_identity_or_decision": False,
        "lean_required_for_replay": False,
        "lean_removal_changes_decision": False,
    }


def _fresh_root(value: str | os.PathLike[str]) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    parent = candidate.parent.resolve(strict=True)
    root = parent / candidate.name
    if not candidate.name or os.path.lexists(root):
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "calibration output root must be fresh"
        )
    root.mkdir(mode=0o700)
    _durable._fsync_directory(parent)
    return root


def _existing_root(value: str | os.PathLike[str]) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "calibration root cannot be a symlink"
        )
    root = candidate.resolve(strict=True)
    if not root.is_dir():
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "calibration root is not a directory"
        )
    return root


def _load_inputs(
    *,
    nomination_root: str | os.PathLike[str],
    source_root: str | os.PathLike[str],
) -> _CalibrationInputs:
    source = load_object_bongard_panel_rubric_calibration_source(source_root)
    nomination = verify_object_bongard_shared_witness_nomination(
        nomination_root, source_root=source_root
    )
    if not nomination.accepted:
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "structured nomination must be cold-verified and accepted"
        )
    specs = build_shared_witness_rubric_specs(
        nomination.artifact,
        expected_artifact_digest=nomination.artifact.artifact_digest,
    )
    if tuple(item.candidate_rank for item in specs) != (0, 1):
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "structured nomination does not provide exact ranks zero and one"
        )
    return _CalibrationInputs(source, nomination, specs)


def _source_identities() -> list[dict[str, str]]:
    # Observer/support sources are appended by their stable API at runtime.
    from bongard.object_bongard_shared_witness_observer import (
        object_bongard_shared_witness_panel_observer_source_digest,
    )
    from bongard.object_bongard_shared_witness_support import (
        object_bongard_shared_witness_support_source_digest,
    )

    rows = {
        "calibration_command_source_sha256": (
            object_bongard_shared_witness_calibration_command_source_digest()
        ),
        "historical_panel_source_sha256": (
            object_bongard_panel_rubric_calibration_source_digest()
        ),
        "nomination_command_source_sha256": (
            object_bongard_shared_witness_nomination_command_source_digest()
        ),
        "shared_witness_ir_source_sha256": (
            object_bongard_shared_witness_source_digest()
        ),
        "shared_witness_observer_source_sha256": (
            object_bongard_shared_witness_panel_observer_source_digest()
        ),
        "shared_witness_support_source_sha256": (
            object_bongard_shared_witness_support_source_digest()
        ),
        "turn_journal_source_sha256": object_bongard_turn_journal_source_digest(),
        "transport_source_sha256": prototype_scene_transport_source_digest(),
        "durable_record_helper_source_sha256": (
            _durable.object_bongard_rubric_nomination_command_source_digest()
        ),
    }
    return [{"role": key, "sha256": rows[key]} for key in sorted(rows)]


def _nomination_parent(
    nomination: VerifiedObjectBongardSharedWitnessNomination,
) -> dict[str, object]:
    return {
        "nomination_authorization_digest": nomination.authorization_digest,
        "nomination_execution_precommit_digest": (
            nomination.execution_precommit_digest
        ),
        "nomination_cold_replay_digest": nomination.cold_replay_digest,
        "nomination_result_digest": nomination.result_digest,
        "nomination_artifact_digest": nomination.artifact.artifact_digest,
        "nomination_source_digest": nomination.source_digest,
        "nomination_accepted": nomination.accepted,
    }


def _jobs(inputs: _CalibrationInputs) -> tuple[_CalibrationJob, ...]:
    jobs = tuple(
        _CalibrationJob(pass_index, panel, spec)
        for pass_index in range(CALIBRATION_PASS_COUNT)
        for spec in inputs.specs
        for panel in inputs.source.panels
    )
    if (
        len(jobs) != CALIBRATION_JOB_COUNT
        or len(
            {
                (item.pass_index, item.spec.spec_digest, item.panel.panel_binding_digest)
                for item in jobs
            }
        )
        != CALIBRATION_JOB_COUNT
    ):
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "calibration job inventory differs"
        )
    return jobs


def _authorization(
    inputs: _CalibrationInputs,
    *,
    parallel_workers: int,
    minutes: int,
    executable: str,
    expected_launcher_sha256: str,
) -> dict[str, Any]:
    if (
        isinstance(parallel_workers, bool)
        or not isinstance(parallel_workers, int)
        or not 1 <= parallel_workers <= MAX_PARALLEL_WORKERS
        or isinstance(minutes, bool)
        or not isinstance(minutes, int)
        or not 1 <= minutes <= 120
        or not isinstance(executable, str)
        or not executable
        or not isinstance(expected_launcher_sha256, str)
        or _RAW_DIGEST.fullmatch(expected_launcher_sha256) is None
    ):
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "calibration runtime selectors are invalid"
        )
    from bongard.object_bongard_shared_witness_observer import (
        object_bongard_shared_witness_panel_output_schema,
        object_bongard_shared_witness_panel_protocol_digest,
        object_bongard_shared_witness_panel_prompt,
    )
    from bongard.object_bongard_shared_witness_support import (
        object_bongard_shared_witness_support_algorithm_digest,
        object_bongard_shared_witness_support_policy_digest,
    )

    return _durable._record(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "command_id": COMMAND_ID,
            "nomination_parent": _nomination_parent(inputs.nomination),
            "historical_source_digest": inputs.source.source_digest,
            "historical_plan_file_sha256": (
                inputs.source.historical_plan_file_sha256
            ),
            "historical_plan_record_digest": (
                inputs.source.historical_plan_record_digest
            ),
            "rubric_specs": [item.to_data() for item in inputs.specs],
            "blind_panel_inventory": [
                {
                    "ordinal": panel.ordinal,
                    "panel_id": panel.panel_id,
                    "panel_binding_digest": panel.panel_binding_digest,
                    "png_sha256": panel.png_sha256,
                    "released_record_digest": panel.released_record_digest,
                }
                for panel in inputs.source.panels
            ],
            "job_order": "pass-then-rank-then-source-ordinal",
            "job_count": CALIBRATION_JOB_COUNT,
            "parallel_workers": parallel_workers,
            "observer_bindings": [
                {
                    "candidate_rank": spec.candidate_rank,
                    "rubric_spec_digest": spec.spec_digest,
                    "protocol_digest": (
                        object_bongard_shared_witness_panel_protocol_digest()
                    ),
                    "prompt_sha256": hashlib.sha256(
                        object_bongard_shared_witness_panel_prompt(spec).encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                    "output_schema_digest": canonical_digest(
                        object_bongard_shared_witness_panel_output_schema(spec)
                    ),
                }
                for spec in inputs.specs
            ],
            "support_algorithm_digest": (
                object_bongard_shared_witness_support_algorithm_digest()
            ),
            "support_policy_digest": (
                object_bongard_shared_witness_support_policy_digest()
            ),
            "source_identities": _source_identities(),
            "runtime_policy": {
                "model": CALIBRATION_MODEL,
                "reasoning_effort": CALIBRATION_REASONING_EFFORT,
                "minutes": minutes,
                "verbose": False,
                "executable": executable,
                "expected_launcher_sha256": expected_launcher_sha256,
            },
            "support_sides_present_in_authorization_but_model_hidden": True,
            "support_sides_present_in_blind_batch": False,
            **_authority_data(),
        },
        "authorization_digest",
    )


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
        raise ObjectBongardSharedWitnessCalibrationCommandError(
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
    inputs: _CalibrationInputs,
    authorization: Mapping[str, Any],
    runtime: ObjectBongardTurnRuntime,
    fingerprint: Mapping[str, str],
) -> dict[str, Any]:
    return _durable._record(
        {
            "schema": PRECOMMIT_SCHEMA,
            "command_id": COMMAND_ID,
            "authorization_digest": authorization["authorization_digest"],
            "nomination_parent": _nomination_parent(inputs.nomination),
            "historical_source_digest": inputs.source.source_digest,
            "rubric_spec_digests": [item.spec_digest for item in inputs.specs],
            "observer_bindings": authorization["observer_bindings"],
            "support_algorithm_digest": authorization[
                "support_algorithm_digest"
            ],
            "support_policy_digest": authorization["support_policy_digest"],
            "source_identities": authorization["source_identities"],
            "runtime_binding": runtime.binding,
            "cloud_policy_cache_snapshot_base64": _durable._encode_bytes(
                runtime.cloud_policy_cache_snapshot.data
                if runtime.cloud_policy_cache_snapshot is not None
                else None
            ),
            "model_catalog_snapshot_base64": _durable._encode_bytes(
                runtime.model_catalog_snapshot.data
            ),
            "no_tools_attestation": runtime.no_tools_attestation.to_dict(),
            "launcher_fingerprint": dict(fingerprint),
            "precommit_fsynced_before_any_of_48_calls": True,
            **_authority_data(),
        },
        "precommit_digest",
    )


def _runtime_from_precommit(
    precommit: Mapping[str, Any],
    authorization: Mapping[str, Any],
    inputs: _CalibrationInputs,
) -> ObjectBongardTurnRuntime:
    raw = _durable._validate_record(
        precommit,
        schema=PRECOMMIT_SCHEMA,
        digest_field="precommit_digest",
        label="shared-witness calibration precommit",
    )
    expected_fields = {
        "schema",
        "command_id",
        "authorization_digest",
        "nomination_parent",
        "historical_source_digest",
        "rubric_spec_digests",
        "observer_bindings",
        "support_algorithm_digest",
        "support_policy_digest",
        "source_identities",
        "runtime_binding",
        "cloud_policy_cache_snapshot_base64",
        "model_catalog_snapshot_base64",
        "no_tools_attestation",
        "launcher_fingerprint",
        "precommit_fsynced_before_any_of_48_calls",
        *_authority_data(),
        "precommit_digest",
    }
    if (
        set(raw) != expected_fields
        or raw["command_id"] != COMMAND_ID
        or raw["authorization_digest"] != authorization["authorization_digest"]
        or raw["nomination_parent"] != _nomination_parent(inputs.nomination)
        or raw["historical_source_digest"] != inputs.source.source_digest
        or raw["rubric_spec_digests"]
        != [item.spec_digest for item in inputs.specs]
        or raw["observer_bindings"] != authorization["observer_bindings"]
        or raw["support_algorithm_digest"]
        != authorization["support_algorithm_digest"]
        or raw["support_policy_digest"]
        != authorization["support_policy_digest"]
        or raw["source_identities"] != _source_identities()
        or raw["precommit_fsynced_before_any_of_48_calls"] is not True
        or any(raw[key] != item for key, item in _authority_data().items())
    ):
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "calibration precommit differs"
        )
    policy = authorization["runtime_policy"]
    if raw["launcher_fingerprint"] != {
        "version": PINNED_CODEX_CLI_VERSION,
        "launcher_digest": policy["expected_launcher_sha256"],
    }:
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "calibration launcher fingerprint differs"
        )
    binding = raw["runtime_binding"]
    catalog_bytes = _durable._decode_bytes(
        raw["model_catalog_snapshot_base64"], "model catalog"
    )
    if catalog_bytes is None:
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "model catalog snapshot is absent"
        )
    runtime = ObjectBongardTurnRuntime(
        model=binding["model"],
        reasoning_effort=binding["reasoning_effort"],
        minutes=binding["minutes"],
        verbose=binding["verbose"],
        executable=binding["executable"],
        cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(
            _durable._decode_bytes(
                raw["cloud_policy_cache_snapshot_base64"], "policy cache"
            )
        ),
        model_catalog_snapshot=CodexModelCatalogSnapshot(catalog_bytes),
        expected_launcher_digest=binding["expected_launcher_digest"],
        no_tools_attestation=CodexNoToolsAttestation.from_mapping(
            raw["no_tools_attestation"]
        ),
        transport_source_digest=binding["transport_source_digest"],
    )
    if (
        runtime.binding != binding
        or policy
        != {
            "model": runtime.model,
            "reasoning_effort": runtime.reasoning_effort,
            "minutes": runtime.minutes,
            "verbose": runtime.verbose,
            "executable": runtime.executable,
            "expected_launcher_sha256": runtime.expected_launcher_digest,
        }
    ):
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "calibration runtime differs from authorization"
        )
    return runtime


def _forbidden_transport(*_args: Any, **_kwargs: Any) -> CodexStructuredResult:
    raise AssertionError("cold shared-witness calibration attempted model transport")


def _journal_directory(root: Path, job: _CalibrationJob) -> Path:
    return (
        root
        / JOURNAL_DIRECTORY
        / f"pass_{job.pass_index}"
        / f"rank_{job.spec.candidate_rank}"
        / f"ordinal_{job.panel.ordinal:03d}"
        / "turn"
    )


def _observation_context(
    job: _CalibrationJob,
    *,
    authorization_digest: str,
    execution_precommit_digest: str,
) -> str:
    return "sha256:" + canonical_digest(
        {
            "schema": "gkm.bongard-shared-witness-calibration-observation-context.v1",
            "command_id": COMMAND_ID,
            "authorization_digest": authorization_digest,
            "execution_precommit_digest": execution_precommit_digest,
            "pass_index": job.pass_index,
            "candidate_rank": job.spec.candidate_rank,
            "rubric_spec_digest": job.spec.spec_digest,
            "panel_ordinal": job.panel.ordinal,
            "panel_id": job.panel.panel_id,
            "panel_binding_digest": job.panel.panel_binding_digest,
            "png_sha256": job.panel.png_sha256,
            "support_side_hidden": True,
        }
    )


def _observe_job(
    root: Path,
    job: _CalibrationJob,
    *,
    runtime: ObjectBongardTurnRuntime,
    authorization_digest: str,
    execution_precommit_digest: str,
    transport: Transport,
) -> tuple[object, object, int, int]:
    from bongard.object_bongard_shared_witness_observer import (
        ObjectBongardSharedWitnessPanelArtifact,
        object_bongard_shared_witness_panel_output_schema,
        object_bongard_shared_witness_panel_prompt,
        observe_object_bongard_shared_witness_panel,
        verify_object_bongard_shared_witness_panel_artifact,
    )

    prompt = object_bongard_shared_witness_panel_prompt(job.spec)
    schema = object_bongard_shared_witness_panel_output_schema(job.spec)
    context = _observation_context(
        job,
        authorization_digest=authorization_digest,
        execution_precommit_digest=execution_precommit_digest,
    )
    journal = ObjectBongardNamedImageTurnJournalTransport(
        _journal_directory(root, job),
        authorization_digest=authorization_digest,
        execution_precommit_digest=execution_precommit_digest,
        task_id=job.panel.task_id,
        turn_kind=(
            f"shared_witness_p{job.pass_index}_r{job.spec.candidate_rank}_"
            f"o{job.panel.ordinal:03d}"
        ),
        expected_prompt=prompt,
        expected_images=(("panel.png", job.panel.exact_png_bytes),),
        expected_output_schema=schema,
        runtime=runtime,
        underlying_transport=transport,
    )
    artifact = observe_object_bongard_shared_witness_panel(
        job.panel.exact_png_bytes,
        panel_id=job.panel.panel_id,
        rubric_spec=job.spec,
        expected_panel_sha256=job.panel.png_sha256,
        expected_rubric_spec_digest=job.spec.spec_digest,
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
        observation_context_digest=context,
    )
    restored = ObjectBongardSharedWitnessPanelArtifact.from_data(
        artifact.to_data()
    )
    verified = verify_object_bongard_shared_witness_panel_artifact(
        restored,
        job.panel.exact_png_bytes,
        panel_id=job.panel.panel_id,
        rubric_spec=job.spec,
        expected_artifact_digest=restored.artifact_digest,
        expected_runtime_identity_digest=restored.runtime_identity_digest,
    )
    if restored != artifact or verified != artifact:
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "shared-witness observer artifact round trip differs"
        )
    summary = verify_object_bongard_turn_journal(journal)
    return artifact, summary, journal.fresh_call_count, journal.reused_call_count


def _run_record(
    job: _CalibrationJob,
    artifact: object,
    summary: object,
    *,
    fresh_call_count: int,
    reused_call_count: int,
) -> dict[str, Any]:
    return _durable._record(
        {
            "schema": "gkm.bongard-shared-witness-calibration-run.v1",
            "pass_index": job.pass_index,
            "candidate_rank": job.spec.candidate_rank,
            "rubric_spec_digest": job.spec.spec_digest,
            "panel_ordinal": job.panel.ordinal,
            "panel_id": job.panel.panel_id,
            "panel_binding_digest": job.panel.panel_binding_digest,
            "png_sha256": job.panel.png_sha256,
            "observation_context_digest": getattr(
                artifact, "observation_context_digest"
            ),
            "observer_artifact": getattr(artifact, "to_data")(),
            "journal_summary": getattr(summary, "to_data")(),
            "fresh_call_count": fresh_call_count,
            "reused_call_count": reused_call_count,
            "support_side_present": False,
            "query_pixels_used": False,
            "fresh_broad_cohort_pixels_used": False,
            "official_test_pixels_used": False,
        },
        "run_digest",
    )


def _execute_batch(
    root: Path,
    inputs: _CalibrationInputs,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    runtime: ObjectBongardTurnRuntime,
    *,
    parallel_workers: int,
    transport: Transport,
) -> dict[str, Any]:
    from bongard.object_bongard_shared_witness_support import (
        object_bongard_shared_witness_support_algorithm_digest,
        object_bongard_shared_witness_support_policy_digest,
    )

    jobs = _jobs(inputs)
    if (
        object_bongard_shared_witness_support_algorithm_digest()
        != authorization["support_algorithm_digest"]
        or object_bongard_shared_witness_support_policy_digest()
        != authorization["support_policy_digest"]
    ):
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "support algorithm differs from pre-label authorization"
        )

    def execute(job: _CalibrationJob) -> dict[str, Any]:
        artifact, summary, fresh, reused = _observe_job(
            root,
            job,
            runtime=runtime,
            authorization_digest=authorization["authorization_digest"],
            execution_precommit_digest=precommit["precommit_digest"],
            transport=transport,
        )
        return _run_record(
            job,
            artifact,
            summary,
            fresh_call_count=fresh,
            reused_call_count=reused,
        )

    with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
        runs = tuple(pool.map(execute, jobs))
    fresh = sum(item["fresh_call_count"] for item in runs)
    reused = sum(item["reused_call_count"] for item in runs)
    if fresh != CALIBRATION_JOB_COUNT or reused != 0:
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "fresh calibration did not execute exactly 48 new calls"
        )
    return _durable._record(
        {
            "schema": BATCH_SCHEMA,
            "command_id": COMMAND_ID,
            "authorization_digest": authorization["authorization_digest"],
            "execution_precommit_digest": precommit["precommit_digest"],
            "nomination_parent": _nomination_parent(inputs.nomination),
            "historical_source_digest": inputs.source.source_digest,
            "run_order": "pass-then-rank-then-source-ordinal",
            "runs": list(runs),
            "run_count": len(runs),
            "fresh_call_count": fresh,
            "reused_call_count": reused,
            "support_labels_present": False,
            "artifacts_frozen_before_support_labels": True,
            "query_pixels_used": False,
            "fresh_broad_cohort_pixels_used": False,
            "official_test_pixels_used": False,
            **_authority_data(),
        },
        "batch_digest",
    )


def _batch_file_bytes(batch: Mapping[str, Any]) -> bytes:
    return canonical_json(dict(batch)) + b"\n"


def _freeze_record(
    inputs: _CalibrationInputs,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _batch_file_bytes(batch)
    return _durable._record(
        {
            "schema": FREEZE_SCHEMA,
            "command_id": COMMAND_ID,
            "authorization_digest": authorization["authorization_digest"],
            "execution_precommit_digest": precommit["precommit_digest"],
            "nomination_result_digest": inputs.nomination.result_digest,
            "historical_source_digest": inputs.source.source_digest,
            "batch_digest": batch["batch_digest"],
            "batch_file_sha256": hashlib.sha256(payload).hexdigest(),
            "batch_file_byte_count": len(payload),
            "frozen_artifact_count": CALIBRATION_JOB_COUNT,
            "batch_fsynced_and_reloaded": True,
            "support_labels_introduced": False,
            "query_pixels_used": False,
            "fresh_broad_cohort_pixels_used": False,
            "official_test_pixels_used": False,
            **_authority_data(),
        },
        "freeze_digest",
    )


def _validated_batch_artifacts(
    batch_value: object,
    inputs: _CalibrationInputs,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
) -> tuple[object, ...]:
    from bongard.object_bongard_shared_witness_observer import (
        ObjectBongardSharedWitnessPanelArtifact,
    )

    batch = _durable._validate_record(
        batch_value,
        schema=BATCH_SCHEMA,
        digest_field="batch_digest",
        label="shared-witness calibration blind batch",
    )
    expected_fields = {
        "schema",
        "command_id",
        "authorization_digest",
        "execution_precommit_digest",
        "nomination_parent",
        "historical_source_digest",
        "run_order",
        "runs",
        "run_count",
        "fresh_call_count",
        "reused_call_count",
        "support_labels_present",
        "artifacts_frozen_before_support_labels",
        "query_pixels_used",
        "fresh_broad_cohort_pixels_used",
        "official_test_pixels_used",
        *_authority_data(),
        "batch_digest",
    }
    if (
        set(batch) != expected_fields
        or batch["command_id"] != COMMAND_ID
        or batch["authorization_digest"] != authorization["authorization_digest"]
        or batch["execution_precommit_digest"] != precommit["precommit_digest"]
        or batch["nomination_parent"] != _nomination_parent(inputs.nomination)
        or batch["historical_source_digest"] != inputs.source.source_digest
        or batch["run_order"] != "pass-then-rank-then-source-ordinal"
        or not isinstance(batch["runs"], list)
        or batch["run_count"] != CALIBRATION_JOB_COUNT
        or len(batch["runs"]) != CALIBRATION_JOB_COUNT
        or batch["fresh_call_count"] != CALIBRATION_JOB_COUNT
        or batch["reused_call_count"] != 0
        or batch["support_labels_present"] is not False
        or batch["artifacts_frozen_before_support_labels"] is not True
        or any(batch[key] is not False for key in (
            "query_pixels_used",
            "fresh_broad_cohort_pixels_used",
            "official_test_pixels_used",
        ))
        or any(batch[key] != item for key, item in _authority_data().items())
    ):
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "blind calibration batch binding differs"
        )

    run_fields = {
        "schema",
        "pass_index",
        "candidate_rank",
        "rubric_spec_digest",
        "panel_ordinal",
        "panel_id",
        "panel_binding_digest",
        "png_sha256",
        "observation_context_digest",
        "observer_artifact",
        "journal_summary",
        "fresh_call_count",
        "reused_call_count",
        "support_side_present",
        "query_pixels_used",
        "fresh_broad_cohort_pixels_used",
        "official_test_pixels_used",
        "run_digest",
    }
    artifacts: list[object] = []
    for raw_run, job in zip(batch["runs"], _jobs(inputs), strict=True):
        run = _durable._validate_record(
            raw_run,
            schema="gkm.bongard-shared-witness-calibration-run.v1",
            digest_field="run_digest",
            label="shared-witness calibration run",
        )
        artifact = ObjectBongardSharedWitnessPanelArtifact.from_data(
            run.get("observer_artifact")
        )
        summary = _durable._validate_record(
            run.get("journal_summary"),
            schema="gkm.bongard-codex-turn-journal-summary.v1",
            digest_field="record_digest",
            label="shared-witness calibration journal summary",
        )
        if (
            set(run) != run_fields
            or run["pass_index"] != job.pass_index
            or run["candidate_rank"] != job.spec.candidate_rank
            or run["rubric_spec_digest"] != job.spec.spec_digest
            or run["panel_ordinal"] != job.panel.ordinal
            or run["panel_id"] != job.panel.panel_id
            or run["panel_binding_digest"] != job.panel.panel_binding_digest
            or run["png_sha256"] != job.panel.png_sha256
            or run["observation_context_digest"]
            != _observation_context(
                job,
                authorization_digest=authorization["authorization_digest"],
                execution_precommit_digest=precommit["precommit_digest"],
            )
            or artifact.panel_id != job.panel.panel_id
            or artifact.panel_digest != job.panel.png_sha256
            or artifact.observation_context_digest
            != run["observation_context_digest"]
            or artifact.rubric_spec_digest != job.spec.spec_digest
            or artifact.rubric_spec != job.spec
            or getattr(artifact, "physical_call_count") != 1
            or summary.get("terminal_status") not in {"success", "failure"}
            or run["fresh_call_count"] != 1
            or run["reused_call_count"] != 0
            or run["support_side_present"] is not False
            or any(run[key] is not False for key in (
                "query_pixels_used",
                "fresh_broad_cohort_pixels_used",
                "official_test_pixels_used",
            ))
        ):
            raise ObjectBongardSharedWitnessCalibrationCommandError(
                "blind calibration run differs"
            )
        artifacts.append(artifact)
    return tuple(artifacts)


def _validate_freeze(
    freeze_value: object,
    inputs: _CalibrationInputs,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    freeze = _durable._validate_record(
        freeze_value,
        schema=FREEZE_SCHEMA,
        digest_field="freeze_digest",
        label="shared-witness calibration durable freeze",
    )
    expected = _freeze_record(inputs, authorization, precommit, batch)
    if freeze != expected:
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "durable freeze differs from reloaded blind batch"
        )
    return freeze


def _assessment_record(
    inputs: _CalibrationInputs,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    batch: Mapping[str, Any],
    freeze: Mapping[str, Any],
    artifacts: Sequence[object],
) -> dict[str, Any]:
    from bongard.object_bongard_shared_witness_support import (
        SharedWitnessSupportAcceptanceTier,
        build_object_bongard_shared_witness_support_version_space,
        cold_verify_object_bongard_shared_witness_support_version_space,
    )

    jobs = _jobs(inputs)
    rows = {
        (
            job.pass_index,
            job.spec.candidate_rank,
            job.panel.panel_binding_digest,
        ): artifact
        for job, artifact in zip(jobs, artifacts, strict=True)
    }
    if len(rows) != CALIBRATION_JOB_COUNT:
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "assessment artifact inventory differs"
        )

    rank_assessments: list[dict[str, Any]] = []
    for spec in inputs.specs:
        spaces = []
        for pass_index in range(CALIBRATION_PASS_COUNT):
            targets = tuple(
                rows[(pass_index, spec.candidate_rank, panel.panel_binding_digest)]
                for panel in inputs.source.panels
                if panel.group_index == 0
            )
            foils = tuple(
                rows[(pass_index, spec.candidate_rank, panel.panel_binding_digest)]
                for panel in inputs.source.panels
                if panel.group_index == 1
            )
            space = build_object_bongard_shared_witness_support_version_space(
                spec, targets, foils
            )
            if cold_verify_object_bongard_shared_witness_support_version_space(
                space, spec, targets, foils
            ) != space:
                raise ObjectBongardSharedWitnessCalibrationCommandError(
                    "support version-space cold replay differs"
                )
            spaces.append(space)

        flips = []
        for panel in inputs.source.panels:
            first = rows[
                (0, spec.candidate_rank, panel.panel_binding_digest)
            ].observation.disposition
            second = rows[
                (1, spec.candidate_rank, panel.panel_binding_digest)
            ].observation.disposition
            if {first, second} == {
                Disposition.PRESENT,
                Disposition.CERTIFIED_ABSENT,
            }:
                flips.append(
                    {
                        "panel_id": panel.panel_id,
                        "panel_binding_digest": panel.panel_binding_digest,
                        "pass_0_disposition": first.value,
                        "pass_1_disposition": second.value,
                    }
                )
        pass_accepted = [
            space.support_acceptance_tier
            is not SharedWitnessSupportAcceptanceTier.REJECTED
            for space in spaces
        ]
        candidate_digests = {
            space.candidate.candidate_digest for space in spaces
        }
        if len(candidate_digests) != 1:
            raise ObjectBongardSharedWitnessCalibrationCommandError(
                "support passes disagree on candidate identity"
            )
        rank_accepted = all(pass_accepted) and not flips
        rank_assessments.append(
            {
                "candidate_rank": spec.candidate_rank,
                "rubric_spec_digest": spec.spec_digest,
                "candidate_digest": next(iter(candidate_digests)),
                "pass_support_version_spaces": [
                    space.to_data() for space in spaces
                ],
                "pass_acceptance_tiers": [
                    space.support_acceptance_tier.value for space in spaces
                ],
                "pass_accepted": pass_accepted,
                "accepted_in_both_passes": all(pass_accepted),
                "cross_pass_present_absent_flips": flips,
                "cross_pass_flip_count": len(flips),
                "rank_accepted": rank_accepted,
            }
        )
    selected = next(
        (item for item in rank_assessments if item["rank_accepted"]), None
    )
    return _durable._record(
        {
            "schema": ASSESSMENT_SCHEMA,
            "command_id": COMMAND_ID,
            "authorization_digest": authorization["authorization_digest"],
            "execution_precommit_digest": precommit["precommit_digest"],
            "nomination_parent": _nomination_parent(inputs.nomination),
            "historical_source_digest": inputs.source.source_digest,
            "batch_digest": batch["batch_digest"],
            "freeze_digest": freeze["freeze_digest"],
            "support_algorithm_digest": authorization[
                "support_algorithm_digest"
            ],
            "support_policy_digest": authorization["support_policy_digest"],
            "rank_assessments": rank_assessments,
            "accepted": selected is not None,
            "selected_candidate_rank": (
                None if selected is None else selected["candidate_rank"]
            ),
            "selected_spec_digest": (
                None if selected is None else selected["rubric_spec_digest"]
            ),
            "selected_candidate_digest": (
                None if selected is None else selected["candidate_digest"]
            ),
            "selection_rule_applied": "lowest-rank-accepted-in-both-passes",
            "support_labels_first_introduced_after_durable_freeze": True,
            "all_48_artifacts_reloaded_before_assessment": True,
            "model_calls_during_assessment": 0,
            "query_pixels_used": False,
            "fresh_broad_cohort_pixels_used": False,
            "official_test_pixels_used": False,
            **_authority_data(),
        },
        "assessment_digest",
    )


def _verify_journal_inventory(root: Path, inputs: _CalibrationInputs) -> None:
    journal_root = root / JOURNAL_DIRECTORY
    if not journal_root.is_dir() or journal_root.is_symlink():
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "calibration journal root is not a real directory"
        )
    expected_passes = {
        f"pass_{index}" for index in range(CALIBRATION_PASS_COUNT)
    }
    if {item.name for item in journal_root.iterdir()} != expected_passes:
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "calibration journal pass inventory differs"
        )
    expected_ranks = {f"rank_{spec.candidate_rank}" for spec in inputs.specs}
    expected_ordinals = {
        f"ordinal_{panel.ordinal:03d}" for panel in inputs.source.panels
    }
    for pass_index in range(CALIBRATION_PASS_COUNT):
        pass_root = journal_root / f"pass_{pass_index}"
        if (
            not pass_root.is_dir()
            or pass_root.is_symlink()
            or {item.name for item in pass_root.iterdir()} != expected_ranks
        ):
            raise ObjectBongardSharedWitnessCalibrationCommandError(
                "calibration journal rank inventory differs"
            )
        for spec in inputs.specs:
            rank_root = pass_root / f"rank_{spec.candidate_rank}"
            if (
                not rank_root.is_dir()
                or rank_root.is_symlink()
                or {item.name for item in rank_root.iterdir()}
                != expected_ordinals
            ):
                raise ObjectBongardSharedWitnessCalibrationCommandError(
                    "calibration journal ordinal inventory differs"
                )
            for panel in inputs.source.panels:
                ordinal_root = rank_root / f"ordinal_{panel.ordinal:03d}"
                turn_root = ordinal_root / "turn"
                if (
                    not ordinal_root.is_dir()
                    or ordinal_root.is_symlink()
                    or {item.name for item in ordinal_root.iterdir()} != {"turn"}
                    or not turn_root.is_dir()
                    or turn_root.is_symlink()
                    or {item.name for item in turn_root.iterdir()}
                    != {
                        "manifest.json",
                        "claim.json",
                        "result.json",
                        "outcome.json",
                    }
                ):
                    raise ObjectBongardSharedWitnessCalibrationCommandError(
                        "terminal calibration journal inventory differs"
                    )


def _replay_journals(
    root: Path,
    inputs: _CalibrationInputs,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    runtime: ObjectBongardTurnRuntime,
    batch: Mapping[str, Any],
) -> tuple[str, ...]:
    _verify_journal_inventory(root, inputs)
    journal_digests: list[str] = []
    for stored, job in zip(batch["runs"], _jobs(inputs), strict=True):
        artifact, summary, fresh, reused = _observe_job(
            root,
            job,
            runtime=runtime,
            authorization_digest=authorization["authorization_digest"],
            execution_precommit_digest=precommit["precommit_digest"],
            transport=_forbidden_transport,
        )
        if (
            artifact.to_data() != stored["observer_artifact"]
            or summary.to_data() != stored["journal_summary"]
            or fresh != 0
            or reused != 1
            or summary.record_digest is None
        ):
            raise ObjectBongardSharedWitnessCalibrationCommandError(
                "cold journal or observer replay differs from frozen batch"
            )
        journal_digests.append(summary.record_digest)
    if len(journal_digests) != CALIBRATION_JOB_COUNT:
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "cold journal replay count differs"
        )
    return tuple(journal_digests)


def _replay_record(
    inputs: _CalibrationInputs,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    batch: Mapping[str, Any],
    freeze: Mapping[str, Any],
    assessment: Mapping[str, Any],
    *,
    journal_summary_digests: Sequence[str],
) -> dict[str, Any]:
    if len(journal_summary_digests) != CALIBRATION_JOB_COUNT:
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "cold replay journal digest inventory differs"
        )
    return _durable._record(
        {
            "schema": REPLAY_SCHEMA,
            "command_id": COMMAND_ID,
            "authorization_digest": authorization["authorization_digest"],
            "execution_precommit_digest": precommit["precommit_digest"],
            "nomination_parent": _nomination_parent(inputs.nomination),
            "historical_source_digest": inputs.source.source_digest,
            "batch_digest": batch["batch_digest"],
            "freeze_digest": freeze["freeze_digest"],
            "assessment_digest": assessment["assessment_digest"],
            "journal_summary_digests": list(journal_summary_digests),
            "journal_replay_count": CALIBRATION_JOB_COUNT,
            "observer_artifact_replay_count": CALIBRATION_JOB_COUNT,
            "support_version_space_replay_count": (
                CALIBRATION_PASS_COUNT * CALIBRATION_RANK_COUNT
            ),
            "fresh_call_count_during_replay": 0,
            "reused_call_count_during_replay": CALIBRATION_JOB_COUNT,
            "model_calls_during_replay": 0,
            "transport_forbidden_during_replay": True,
            "support_projection_recomputed": True,
            "cross_pass_flips_recomputed": True,
            "lowest_rank_selection_recomputed": True,
            "query_pixels_used": False,
            "fresh_broad_cohort_pixels_used": False,
            "official_test_pixels_used": False,
            **_authority_data(),
        },
        "replay_digest",
    )


def _result_record(
    inputs: _CalibrationInputs,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    batch: Mapping[str, Any],
    freeze: Mapping[str, Any],
    assessment: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    parent = _nomination_parent(inputs.nomination)
    return _durable._record(
        {
            "schema": RESULT_SCHEMA,
            "command_id": COMMAND_ID,
            "nomination_parent": parent,
            "nomination_authorization_digest": parent[
                "nomination_authorization_digest"
            ],
            "nomination_execution_precommit_digest": parent[
                "nomination_execution_precommit_digest"
            ],
            "nomination_cold_replay_digest": parent[
                "nomination_cold_replay_digest"
            ],
            "nomination_result_digest": parent["nomination_result_digest"],
            "nomination_artifact_digest": parent[
                "nomination_artifact_digest"
            ],
            "nomination_source_digest": parent["nomination_source_digest"],
            "historical_source_digest": inputs.source.source_digest,
            "historical_plan_file_sha256": (
                inputs.source.historical_plan_file_sha256
            ),
            "historical_plan_record_digest": (
                inputs.source.historical_plan_record_digest
            ),
            "source_identities": authorization["source_identities"],
            "authorization_digest": authorization["authorization_digest"],
            "execution_precommit_digest": precommit["precommit_digest"],
            "batch_digest": batch["batch_digest"],
            "freeze_digest": freeze["freeze_digest"],
            "assessment_digest": assessment["assessment_digest"],
            "cold_replay_digest": replay["replay_digest"],
            "accepted": assessment["accepted"],
            "selected_candidate_rank": assessment["selected_candidate_rank"],
            "selected_spec_digest": assessment["selected_spec_digest"],
            "selected_candidate_digest": assessment[
                "selected_candidate_digest"
            ],
            "fresh_call_count": batch["fresh_call_count"],
            "reused_call_count": batch["reused_call_count"],
            "physical_call_denominator": CALIBRATION_JOB_COUNT,
            "campaign_gate_lineage_complete": True,
            "all_48_artifacts_frozen_and_reloaded_before_assessment": True,
            "model_calls_during_assessment_or_replay": 0,
            "query_pixels_used": False,
            "fresh_broad_cohort_pixels_used": False,
            "official_test_pixels_used": False,
            **_authority_data(),
        },
        "result_digest",
    )


def _verification(
    root: Path,
    inputs: _CalibrationInputs,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    batch: Mapping[str, Any],
    freeze: Mapping[str, Any],
    assessment: Mapping[str, Any],
    replay: Mapping[str, Any],
    result: Mapping[str, Any],
) -> VerifiedObjectBongardSharedWitnessCalibration:
    if (
        result["accepted"] is not assessment["accepted"]
        or result["selected_candidate_rank"]
        != assessment["selected_candidate_rank"]
        or result["selected_spec_digest"] != assessment["selected_spec_digest"]
        or result["fresh_call_count"] != CALIBRATION_JOB_COUNT
        or result["reused_call_count"] != 0
    ):
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "calibration result summary differs"
        )
    parent = _nomination_parent(inputs.nomination)
    return VerifiedObjectBongardSharedWitnessCalibration(
        root,
        parent["nomination_authorization_digest"],
        parent["nomination_execution_precommit_digest"],
        parent["nomination_result_digest"],
        parent["nomination_cold_replay_digest"],
        parent["nomination_artifact_digest"],
        inputs.source.source_digest,
        authorization["authorization_digest"],
        precommit["precommit_digest"],
        batch["batch_digest"],
        freeze["freeze_digest"],
        assessment["assessment_digest"],
        replay["replay_digest"],
        result["result_digest"],
        assessment["accepted"],
        assessment["selected_candidate_rank"],
        assessment["selected_spec_digest"],
        batch["fresh_call_count"],
        batch["reused_call_count"],
    )


def _validated_authorization(
    value: object, inputs: _CalibrationInputs
) -> dict[str, Any]:
    authorization = _durable._validate_record(
        value,
        schema=AUTHORIZATION_SCHEMA,
        digest_field="authorization_digest",
        label="shared-witness calibration authorization",
    )
    policy = authorization.get("runtime_policy")
    if not isinstance(policy, Mapping):
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "calibration authorization runtime policy is malformed"
        )
    expected = _authorization(
        inputs,
        parallel_workers=authorization.get("parallel_workers"),
        minutes=policy.get("minutes"),
        executable=policy.get("executable"),
        expected_launcher_sha256=policy.get("expected_launcher_sha256"),
    )
    if authorization != expected:
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "calibration authorization differs on replay"
        )
    return authorization


def _write_and_reload(
    path: Path,
    value: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    _durable._write_once(path, value, label)
    restored = _durable._read_record(path, label)
    if restored != dict(value):
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            f"persisted {label} differs"
        )
    return restored


def run_object_bongard_shared_witness_calibration(
    output_root: str | os.PathLike[str],
    *,
    nomination_root: str | os.PathLike[str] = DEFAULT_STRUCTURED_NOMINATION_ROOT,
    source_root: str | os.PathLike[str] = (
        DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE
    ),
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    minutes: int = DEFAULT_MINUTES,
    executable: str = DEFAULT_EXECUTABLE,
    expected_launcher_sha256: str = DEFAULT_EXPECTED_LAUNCHER_SHA256,
    transport: Transport = run_codex_named_images_structured,
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
) -> VerifiedObjectBongardSharedWitnessCalibration:
    """Launch exactly 48 fresh turns, freeze them, assess, and cold replay."""

    inputs = _load_inputs(
        nomination_root=nomination_root,
        source_root=source_root,
    )
    authorization = _authorization(
        inputs,
        parallel_workers=parallel_workers,
        minutes=minutes,
        executable=executable,
        expected_launcher_sha256=expected_launcher_sha256,
    )
    root = _fresh_root(output_root)
    authorization = _write_and_reload(
        root / AUTHORIZATION_FILENAME,
        authorization,
        label="shared-witness calibration authorization",
    )
    runtime, fingerprint = _create_runtime(
        authorization,
        cache_snapshotter=cache_snapshotter,
        catalog_snapshotter=catalog_snapshotter,
        launcher_fingerprinter=launcher_fingerprinter,
        runtime_attester=runtime_attester,
    )
    precommit = _precommit(
        inputs, authorization, runtime, fingerprint
    )
    precommit = _write_and_reload(
        root / PRECOMMIT_FILENAME,
        precommit,
        label="shared-witness calibration execution precommit",
    )
    runtime = _runtime_from_precommit(precommit, authorization, inputs)

    batch = _execute_batch(
        root,
        inputs,
        authorization,
        precommit,
        runtime,
        parallel_workers=parallel_workers,
        transport=transport,
    )
    batch = _write_and_reload(
        root / BATCH_FILENAME,
        batch,
        label="shared-witness calibration blind observation batch",
    )
    artifacts = _validated_batch_artifacts(
        batch, inputs, authorization, precommit
    )
    freeze = _freeze_record(inputs, authorization, precommit, batch)
    freeze = _write_and_reload(
        root / FREEZE_FILENAME,
        freeze,
        label="shared-witness calibration durable freeze",
    )
    freeze = _validate_freeze(
        freeze, inputs, authorization, precommit, batch
    )
    assessment = _assessment_record(
        inputs,
        authorization,
        precommit,
        batch,
        freeze,
        artifacts,
    )
    assessment = _write_and_reload(
        root / ASSESSMENT_FILENAME,
        assessment,
        label="shared-witness calibration assessment",
    )
    journal_digests = _replay_journals(
        root, inputs, authorization, precommit, runtime, batch
    )
    replay = _replay_record(
        inputs,
        authorization,
        precommit,
        batch,
        freeze,
        assessment,
        journal_summary_digests=journal_digests,
    )
    replay = _write_and_reload(
        root / REPLAY_FILENAME,
        replay,
        label="shared-witness calibration cold replay",
    )
    result = _result_record(
        inputs,
        authorization,
        precommit,
        batch,
        freeze,
        assessment,
        replay,
    )
    result = _write_and_reload(
        root / RESULT_FILENAME,
        result,
        label="shared-witness calibration result",
    )
    return _verification(
        root,
        inputs,
        authorization,
        precommit,
        batch,
        freeze,
        assessment,
        replay,
        result,
    )


def _verify_loaded_calibration(
    output_root: str | os.PathLike[str],
    inputs: _CalibrationInputs,
) -> VerifiedObjectBongardSharedWitnessCalibration:
    root = _existing_root(output_root)
    expected_inventory = {
        AUTHORIZATION_FILENAME,
        PRECOMMIT_FILENAME,
        BATCH_FILENAME,
        FREEZE_FILENAME,
        ASSESSMENT_FILENAME,
        REPLAY_FILENAME,
        RESULT_FILENAME,
        JOURNAL_DIRECTORY,
    }
    if {item.name for item in root.iterdir()} != expected_inventory:
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "calibration root inventory differs"
        )
    authorization = _validated_authorization(
        _durable._read_record(
            root / AUTHORIZATION_FILENAME, "calibration authorization"
        ),
        inputs,
    )
    precommit = _durable._read_record(
        root / PRECOMMIT_FILENAME, "calibration execution precommit"
    )
    runtime = _runtime_from_precommit(precommit, authorization, inputs)
    batch = _durable._read_record(
        root / BATCH_FILENAME, "calibration blind observation batch"
    )
    artifacts = _validated_batch_artifacts(
        batch, inputs, authorization, precommit
    )
    freeze = _validate_freeze(
        _durable._read_record(
            root / FREEZE_FILENAME, "calibration durable freeze"
        ),
        inputs,
        authorization,
        precommit,
        batch,
    )
    expected_assessment = _assessment_record(
        inputs,
        authorization,
        precommit,
        batch,
        freeze,
        artifacts,
    )
    assessment = _durable._validate_record(
        _durable._read_record(
            root / ASSESSMENT_FILENAME, "calibration assessment"
        ),
        schema=ASSESSMENT_SCHEMA,
        digest_field="assessment_digest",
        label="shared-witness calibration assessment",
    )
    if assessment != expected_assessment:
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "calibration assessment differs on replay"
        )
    journal_digests = _replay_journals(
        root, inputs, authorization, precommit, runtime, batch
    )
    expected_replay = _replay_record(
        inputs,
        authorization,
        precommit,
        batch,
        freeze,
        assessment,
        journal_summary_digests=journal_digests,
    )
    replay = _durable._validate_record(
        _durable._read_record(
            root / REPLAY_FILENAME, "calibration cold replay"
        ),
        schema=REPLAY_SCHEMA,
        digest_field="replay_digest",
        label="shared-witness calibration cold replay",
    )
    if replay != expected_replay:
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "calibration cold replay differs"
        )
    expected_result = _result_record(
        inputs,
        authorization,
        precommit,
        batch,
        freeze,
        assessment,
        replay,
    )
    result = _durable._validate_record(
        _durable._read_record(root / RESULT_FILENAME, "calibration result"),
        schema=RESULT_SCHEMA,
        digest_field="result_digest",
        label="shared-witness calibration result",
    )
    if result != expected_result:
        raise ObjectBongardSharedWitnessCalibrationCommandError(
            "calibration result differs"
        )
    return _verification(
        root,
        inputs,
        authorization,
        precommit,
        batch,
        freeze,
        assessment,
        replay,
        result,
    )


def verify_object_bongard_shared_witness_calibration(
    output_root: str | os.PathLike[str],
    *,
    nomination_root: str | os.PathLike[str] = DEFAULT_STRUCTURED_NOMINATION_ROOT,
    source_root: str | os.PathLike[str] = (
        DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE
    ),
) -> VerifiedObjectBongardSharedWitnessCalibration:
    """Cold-verify all 48 journals and all Python decisions with no model."""

    inputs = _load_inputs(
        nomination_root=nomination_root,
        source_root=source_root,
    )
    return _verify_loaded_calibration(output_root, inputs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python3 -m "
            "bongard.object_bongard_shared_witness_calibration_command"
        ),
        description=(
            "Launch or cold-verify the sealed 48-turn two-pass "
            "shared-witness calibration"
        ),
    )
    commands = parser.add_subparsers(dest="operation", required=True)
    for name in ("launch", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--output-root", required=True, type=Path)
        command.add_argument(
            "--nomination-root",
            type=Path,
            default=DEFAULT_STRUCTURED_NOMINATION_ROOT,
        )
        command.add_argument(
            "--source-root",
            type=Path,
            default=DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE,
        )
    launch = commands.choices["launch"]
    launch.add_argument(
        "--parallel-workers", type=int, default=DEFAULT_PARALLEL_WORKERS
    )
    launch.add_argument("--minutes", type=int, default=DEFAULT_MINUTES)
    launch.add_argument("--executable", default=DEFAULT_EXECUTABLE)
    launch.add_argument(
        "--expected-launcher-sha256",
        default=DEFAULT_EXPECTED_LAUNCHER_SHA256,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    try:
        common = {
            "nomination_root": args.nomination_root,
            "source_root": args.source_root,
        }
        if args.operation == "launch":
            verified = run_object_bongard_shared_witness_calibration(
                args.output_root,
                parallel_workers=args.parallel_workers,
                minutes=args.minutes,
                executable=args.executable,
                expected_launcher_sha256=args.expected_launcher_sha256,
                **common,
            )
        else:
            verified = verify_object_bongard_shared_witness_calibration(
                args.output_root, **common
            )
    except Exception as exc:
        try:
            prefix = str(exc).encode("utf-8", errors="replace")[:4096]
        except Exception:
            prefix = b""
        print(
            canonical_json(
                {
                    "schema": (
                        "gkm.bongard-shared-witness-calibration-command-error.v1"
                    ),
                    "error_type": (
                        f"{type(exc).__module__}.{type(exc).__qualname__}"
                    ),
                    "message_prefix_sha256": (
                        None
                        if not prefix
                        else hashlib.sha256(prefix).hexdigest()
                    ),
                    "raw_message_persisted": False,
                }
            ).decode("utf-8"),
            file=sys.stderr,
        )
        return 1
    print(canonical_json(verified.summary_data()).decode("utf-8"))
    return 0 if verified.accepted else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = (
    "CALIBRATION_JOB_COUNT",
    "DEFAULT_EXPECTED_LAUNCHER_SHA256",
    "DEFAULT_PARALLEL_WORKERS",
    "DEFAULT_STRUCTURED_NOMINATION_ROOT",
    "ObjectBongardSharedWitnessCalibrationCommandError",
    "VerifiedObjectBongardSharedWitnessCalibration",
    "main",
    "object_bongard_shared_witness_calibration_command_source_digest",
    "run_object_bongard_shared_witness_calibration",
    "verify_object_bongard_shared_witness_calibration",
)

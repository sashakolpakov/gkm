"""Diagnostic-only whole-panel probe for the rejected v10 rubric calibration.

This command is deliberately *not* a calibration authorization.  It reuses the
authenticated runtime objects frozen by the rejected v10 calibration, but that
old authorization did not authorize these twelve new whole-panel calls.  The
command therefore writes ``diagnostic_unsealed`` into every aggregate record
and cannot open query, broad-cohort, or official-test pixels.

The model-visible boundary contains exactly ``panel.png`` and the frozen rank-0
rubric.  Group membership is consumed only after all twelve artifacts have
been persisted and cold-replayed without model access.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_panel_rubric_observer import (
    ObjectBongardPanelRubricArtifact,
    PanelRubricDisposition,
    object_bongard_panel_rubric_observer_source_digest,
    object_bongard_panel_rubric_output_schema,
    object_bongard_panel_rubric_prompt,
    object_bongard_panel_rubric_protocol_digest,
    observe_object_bongard_panel_rubric,
    verify_object_bongard_panel_rubric_artifact,
)
from bongard.object_bongard_rubric_calibration import (
    DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE,
    ObjectBongardRubricCalibrationGroup,
    ObjectBongardRubricCalibrationPanel,
    ObjectBongardRubricCalibrationSource,
    load_object_bongard_rubric_calibration_source,
    object_bongard_rubric_calibration_source_digest,
)
from bongard.object_bongard_rubric_calibration_command import (
    ObjectBongardRubricCalibrationExecutionPrecommit,
    load_object_bongard_rubric_calibration_execution_precommit,
    object_bongard_rubric_calibration_command_source_digest,
)
from bongard.object_bongard_rubric_nomination_command import (
    VerifiedObjectBongardRubricNomination,
    cold_verify_object_bongard_rubric_nomination,
    object_bongard_rubric_nomination_command_source_digest,
)
from bongard.object_bongard_rubric_observer import (
    ObjectBongardRubricSpec,
    object_bongard_rubric_observer_source_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.transport import run_codex_named_images_structured


PROBE_MANIFEST_SCHEMA = "gkm.bongard-panel-rubric-probe-manifest.v1"
PROBE_REPLAY_SCHEMA = "gkm.bongard-panel-rubric-probe-artifact-replay.v1"
PROBE_RESULT_SCHEMA = "gkm.bongard-panel-rubric-probe-result.v1"
PROBE_STATUS = "diagnostic_unsealed"
PROBE_PANEL_COUNT = 12
PROBE_MAX_WORKERS = 4
MANIFEST_FILENAME = "manifest.json"
RESULT_FILENAME = "result.json"
ARTIFACT_DIRECTORY = "artifacts"
REPLAY_DIRECTORY = "replays"

DEFAULT_V10_NOMINATION_ROOT = Path(
    "downloads/ShapeBongard_V2_full/"
    "object_rubric_nomination_20260808_all_support_v10"
)
DEFAULT_REJECTED_V10_CALIBRATION_ROOT = Path(
    "downloads/ShapeBongard_V2_full/"
    "object_rubric_calibration_20260808_all_support_v10"
)

V10_NOMINATION_ARTIFACT_DIGEST = (
    "c765cdfaba7315ce04265e2151490a86f25d042347eac5cba8a7fc1282dc7c29"
)
V10_NOMINATION_AUTHORIZATION_DIGEST = (
    "sha256:65d2c58cb09bd3e7aeecde0093a50047ccb1676af105559758b589e5cdd368fe"
)
V10_NOMINATION_PRECOMMIT_DIGEST = (
    "sha256:caaa7aea85d3c35838c0abfbc052743f7fe05a7e52ff817c2a3a1c2e2ba992bd"
)
V10_NOMINATION_REPLAY_DIGEST = (
    "sha256:b1c20a920e12f4d2e85f42a3cee06d7565e308f52378e5edfb6bc4ee7c9ed6c4"
)
V10_NOMINATION_RESULT_DIGEST = (
    "sha256:2e0bcd7e0792641265806ccde66bac1af7f791746cf02051454f57ebf7fac4cf"
)
REJECTED_V10_CALIBRATION_AUTHORIZATION_DIGEST = (
    "sha256:096c890ec7b2a2fa7b943e2698527c4d30aa6246246085c433e17fd6f5be5cb5"
)
REJECTED_V10_CALIBRATION_PRECOMMIT_DIGEST = (
    "sha256:97c84bcc1542bf438f4c1ec0720047540432267c8446995076b9c78cfd318bc2"
)

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")


class ObjectBongardPanelRubricProbeError(RuntimeError):
    """The diagnostic boundary, persisted evidence, or replay differs."""


@dataclass(frozen=True, slots=True)
class _ProbeInputs:
    nomination: VerifiedObjectBongardRubricNomination
    source: ObjectBongardRubricCalibrationSource
    precommit: ObjectBongardRubricCalibrationExecutionPrecommit
    rubric_spec: ObjectBongardRubricSpec


@dataclass(frozen=True, slots=True)
class _BlindPanelJob:
    probe_index: int
    panel_id: str
    panel_sha256: str
    exact_png_bytes: bytes
    observation_context_digest: str


@dataclass(frozen=True, slots=True)
class VerifiedObjectBongardPanelRubricProbe:
    output_root: Path
    manifest_digest: str
    result_digest: str
    exact_survivor: bool
    group_counts: Mapping[str, Mapping[str, int]]

    def summary_data(self) -> dict[str, object]:
        return {
            "schema": "gkm.bongard-panel-rubric-probe-summary.v1",
            "status": PROBE_STATUS,
            "output_root": str(self.output_root),
            "manifest_digest": self.manifest_digest,
            "result_digest": self.result_digest,
            "exact_survivor": self.exact_survivor,
            "group_counts": {
                group: dict(counts) for group, counts in self.group_counts.items()
            },
            "old_calibration_authorization_authorizes_probe_jobs": False,
        }


Transport = Callable[..., Any]


def object_bongard_panel_rubric_probe_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_defines_identity_or_decision": False,
        "lean_required_for_replay": False,
        "negation_allowed": False,
        "polarity_flip_allowed": False,
        "threshold_selection_allowed": False,
        "model_selection_allowed": False,
    }


def _record(body: Mapping[str, Any]) -> dict[str, Any]:
    frozen = json.loads(canonical_json(dict(body)).decode("utf-8"))
    if not isinstance(frozen, dict):
        raise ObjectBongardPanelRubricProbeError("record body is not an object")
    return {**frozen, "record_digest": "sha256:" + canonical_digest(frozen)}


def _verify_record(value: object, *, schema: str, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ObjectBongardPanelRubricProbeError(f"{label} is not an object")
    raw = dict(value)
    digest = raw.pop("record_digest", None)
    if raw.get("schema") != schema or digest != "sha256:" + canonical_digest(raw):
        raise ObjectBongardPanelRubricProbeError(f"{label} digest or schema differs")
    return {**raw, "record_digest": digest}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once(path: Path, value: Mapping[str, Any], label: str) -> str:
    payload = canonical_json(dict(value)) + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ObjectBongardPanelRubricProbeError(f"{label} already exists") from exc
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    if path.read_bytes() != payload:
        raise ObjectBongardPanelRubricProbeError(f"persisted {label} changed")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardPanelRubricProbeError(f"cannot read {label}") from exc
    if not isinstance(value, dict) or payload != canonical_json(value) + b"\n":
        raise ObjectBongardPanelRubricProbeError(f"{label} is not canonical JSON")
    return value


def _fresh_output_root(value: str | os.PathLike[str]) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    parent = candidate.parent.resolve(strict=True)
    root = parent / candidate.name
    if not candidate.name or root.exists() or root.is_symlink():
        raise ObjectBongardPanelRubricProbeError("output root must be fresh")
    root.mkdir(mode=0o700)
    (root / ARTIFACT_DIRECTORY).mkdir(mode=0o700)
    (root / REPLAY_DIRECTORY).mkdir(mode=0o700)
    _fsync_directory(root)
    _fsync_directory(parent)
    return root


def _existing_output_root(value: str | os.PathLike[str]) -> Path:
    root = Path(value).expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ObjectBongardPanelRubricProbeError("probe root is not a directory")
    return root


def _load_probe_inputs(
    *,
    nomination_root: str | os.PathLike[str],
    rejected_calibration_root: str | os.PathLike[str],
    source_directory: str | os.PathLike[str],
) -> _ProbeInputs:
    nomination = cold_verify_object_bongard_rubric_nomination(
        nomination_root, source_root=source_directory
    )
    expected_nomination = (
        V10_NOMINATION_ARTIFACT_DIGEST,
        V10_NOMINATION_AUTHORIZATION_DIGEST,
        V10_NOMINATION_PRECOMMIT_DIGEST,
        V10_NOMINATION_REPLAY_DIGEST,
        V10_NOMINATION_RESULT_DIGEST,
        True,
    )
    observed_nomination = (
        nomination.artifact.artifact_digest,
        nomination.authorization_digest,
        nomination.execution_precommit_digest,
        nomination.cold_replay_digest,
        nomination.result_digest,
        nomination.accepted,
    )
    if observed_nomination != expected_nomination:
        raise ObjectBongardPanelRubricProbeError("nomination is not accepted v10")

    source = load_object_bongard_rubric_calibration_source(source_directory)
    if len(source.panels) != PROBE_PANEL_COUNT or source.nomination_artifact is not None:
        raise ObjectBongardPanelRubricProbeError(
            "probe source is not the exact pre-nomination exposed 12-panel source"
        )
    expected_groups = (
        tuple(sorted(item.panel_id for item in source.group_a_panels)),
        tuple(sorted(item.panel_id for item in source.group_b_panels)),
    )
    if nomination.artifact.group_panel_ids != expected_groups:
        raise ObjectBongardPanelRubricProbeError(
            "v10 nomination differs from exact exposed panel groups"
        )

    precommit = load_object_bongard_rubric_calibration_execution_precommit(
        rejected_calibration_root
    )
    if (
        precommit.precommit_digest != REJECTED_V10_CALIBRATION_PRECOMMIT_DIGEST
        or precommit.authorization_digest
        != REJECTED_V10_CALIBRATION_AUTHORIZATION_DIGEST
        or precommit.nomination_binding.artifact_digest
        != nomination.artifact.artifact_digest
        or precommit.nomination_binding.authorization_digest
        != nomination.authorization_digest
        or precommit.nomination_binding.execution_precommit_digest
        != nomination.execution_precommit_digest
        or precommit.nomination_binding.cold_replay_digest
        != nomination.cold_replay_digest
        or precommit.nomination_binding.command_result_digest
        != nomination.result_digest
    ):
        raise ObjectBongardPanelRubricProbeError(
            "runtime precommit is not the rejected v10 calibration precommit"
        )
    rubric_spec = ObjectBongardRubricSpec.from_semantic_artifact(
        nomination.artifact,
        expected_artifact_digest=nomination.artifact.artifact_digest,
        candidate_rank=0,
    )
    return _ProbeInputs(nomination, source, precommit, rubric_spec)


def _runtime_binding(inputs: _ProbeInputs) -> dict[str, object]:
    return json.loads(canonical_json(inputs.precommit.runtime.binding).decode("utf-8"))


def _blind_jobs(inputs: _ProbeInputs) -> tuple[_BlindPanelJob, ...]:
    runtime_digest = canonical_digest(
        {"schema": "gkm.bongard-panel-rubric-probe-runtime-binding.v1", **_runtime_binding(inputs)}
    )
    jobs: list[_BlindPanelJob] = []
    for index, panel in enumerate(inputs.source.panels):
        context = "sha256:" + canonical_digest(
            {
                "schema": "gkm.bongard-panel-rubric-probe-context.v1",
                "status": PROBE_STATUS,
                "probe_index": index,
                "panel_id": panel.panel_id,
                "panel_sha256": panel.png_sha256,
                "rubric_spec_digest": inputs.rubric_spec.spec_digest,
                "runtime_binding_digest": runtime_digest,
                "nomination_artifact_digest": inputs.nomination.artifact.artifact_digest,
                "rejected_calibration_precommit_digest": inputs.precommit.precommit_digest,
            }
        )
        jobs.append(
            _BlindPanelJob(
                index,
                panel.panel_id,
                panel.png_sha256,
                panel.exact_png_bytes,
                context,
            )
        )
    return tuple(jobs)


def _manifest(inputs: _ProbeInputs, *, parallel_workers: int) -> dict[str, Any]:
    if isinstance(parallel_workers, bool) or not 1 <= parallel_workers <= PROBE_MAX_WORKERS:
        raise ObjectBongardPanelRubricProbeError("parallel workers must lie in 1..4")
    jobs = _blind_jobs(inputs)
    prompt = object_bongard_panel_rubric_prompt(inputs.rubric_spec)
    schema = object_bongard_panel_rubric_output_schema()
    runtime = _runtime_binding(inputs)
    body = {
        "schema": PROBE_MANIFEST_SCHEMA,
        "status": PROBE_STATUS,
        "purpose": "rank-0-whole-panel-observer-diagnostic-on-already-exposed-calibration-only",
        "authorization": {
            "rejected_calibration_authorization_digest": inputs.precommit.authorization_digest,
            "rejected_calibration_precommit_digest": inputs.precommit.precommit_digest,
            "old_calibration_authorization_authorizes_probe_jobs": False,
            "new_probe_authorization_present": False,
            "benchmark_or_calibration_claim_authorized": False,
        },
        "pixel_scope": {
            "exact_already_exposed_calibration_panel_count": len(jobs),
            "query_pixels_opened": False,
            "broad_cohort_pixels_opened": False,
            "official_test_pixels_opened": False,
            "other_pixels_authorized": False,
        },
        "call_policy": {
            "rank": 0,
            "one_complete_panel_per_call": True,
            "calls_per_panel": 1,
            "physical_call_count": len(jobs),
            "parallel_workers": parallel_workers,
            "maximum_parallel_workers": PROBE_MAX_WORKERS,
            "model_visible_image_names": ["panel.png"],
            "labels_or_roles_visible_to_observer": False,
            "labels_used_for_disposition_aggregation_only_after_all_artifacts_persisted_and_replayed": True,
        },
        "nomination": {
            "artifact_digest": inputs.nomination.artifact.artifact_digest,
            "authorization_digest": inputs.nomination.authorization_digest,
            "execution_precommit_digest": inputs.nomination.execution_precommit_digest,
            "cold_replay_digest": inputs.nomination.cold_replay_digest,
            "result_digest": inputs.nomination.result_digest,
        },
        "calibration_source_digest": inputs.source.source_digest,
        "rubric_spec": inputs.rubric_spec.to_data(),
        "rubric_spec_digest": inputs.rubric_spec.spec_digest,
        "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "output_schema_digest": canonical_digest(schema),
        "runtime_binding": runtime,
        "runtime_binding_digest": canonical_digest(
            {"schema": "gkm.bongard-panel-rubric-probe-runtime-binding.v1", **runtime}
        ),
        "runtime_objects_reused_from_rejected_v10_precommit": True,
        "source_identities": {
            "probe_source_sha256": object_bongard_panel_rubric_probe_source_digest(),
            "panel_observer_source_sha256": object_bongard_panel_rubric_observer_source_digest(),
            "panel_observer_protocol_sha256": object_bongard_panel_rubric_protocol_digest(),
            "rubric_spec_authority_source_sha256": object_bongard_rubric_observer_source_digest(),
            "calibration_source_sha256": object_bongard_rubric_calibration_source_digest(),
            "calibration_command_source_sha256": object_bongard_rubric_calibration_command_source_digest(),
            "nomination_command_source_sha256": object_bongard_rubric_nomination_command_source_digest(),
            "runtime_transport_source_sha256": inputs.precommit.runtime.transport_source_digest,
        },
        "panels": [
            {
                "probe_index": job.probe_index,
                "panel_id": job.panel_id,
                "panel_sha256": job.panel_sha256,
                "observation_context_digest": job.observation_context_digest,
            }
            for job in jobs
        ],
        **_authority_data(),
    }
    return _record(body)


def _artifact_filename(index: int) -> str:
    return f"{index:03d}.json"


def _artifact_replay_record(
    *,
    manifest_digest: str,
    job: _BlindPanelJob,
    artifact: ObjectBongardPanelRubricArtifact,
    artifact_file_sha256: str,
) -> dict[str, Any]:
    return _record(
        {
            "schema": PROBE_REPLAY_SCHEMA,
            "status": PROBE_STATUS,
            "manifest_digest": manifest_digest,
            "probe_index": job.probe_index,
            "panel_id": job.panel_id,
            "panel_sha256": job.panel_sha256,
            "artifact_file_sha256": artifact_file_sha256,
            "artifact_digest": artifact.artifact_digest,
            "runtime_identity_digest": artifact.runtime_identity_digest,
            "observation_digest": artifact.observation.observation_digest,
            "disposition": artifact.observation.disposition.value,
            "physical_call_count": artifact.physical_call_count,
            "cold_replay_model_calls": 0,
            "cold_replay_verified": True,
            "old_calibration_authorization_authorizes_probe_job": False,
        }
    )


def _run_blind_job(
    *,
    root: Path,
    manifest_digest: str,
    inputs: _ProbeInputs,
    job: _BlindPanelJob,
    transport: Transport,
) -> ObjectBongardPanelRubricArtifact:
    runtime = inputs.precommit.runtime
    artifact = observe_object_bongard_panel_rubric(
        job.exact_png_bytes,
        panel_id=job.panel_id,
        rubric_spec=inputs.rubric_spec,
        expected_panel_sha256=job.panel_sha256,
        expected_rubric_spec_digest=inputs.rubric_spec.spec_digest,
        model=runtime.model,
        reasoning_effort=runtime.reasoning_effort,
        minutes=runtime.minutes,
        verbose=runtime.verbose,
        executable=runtime.executable,
        cloud_policy_cache_snapshot=runtime.cloud_policy_cache_snapshot,
        expected_launcher_digest=runtime.expected_launcher_digest,
        model_catalog_snapshot=runtime.model_catalog_snapshot,
        no_tools_attestation=runtime.no_tools_attestation,
        transport=transport,
        observation_context_digest=job.observation_context_digest,
    )
    artifact_path = root / ARTIFACT_DIRECTORY / _artifact_filename(job.probe_index)
    file_sha256 = _write_once(
        artifact_path, artifact.to_data(), f"artifact {job.probe_index}"
    )
    restored = ObjectBongardPanelRubricArtifact.from_data(
        _read_json(artifact_path, f"artifact {job.probe_index}")
    )
    replayed = verify_object_bongard_panel_rubric_artifact(
        restored,
        job.exact_png_bytes,
        panel_id=job.panel_id,
        rubric_spec=inputs.rubric_spec,
        expected_artifact_digest=artifact.artifact_digest,
        expected_runtime_identity_digest=artifact.runtime_identity_digest,
    )
    replay = _artifact_replay_record(
        manifest_digest=manifest_digest,
        job=job,
        artifact=replayed,
        artifact_file_sha256=file_sha256,
    )
    _write_once(
        root / REPLAY_DIRECTORY / _artifact_filename(job.probe_index),
        replay,
        f"artifact replay {job.probe_index}",
    )
    return replayed


_DISPOSITIONS = tuple(item.value for item in PanelRubricDisposition)


def _group_counts(
    source: ObjectBongardRubricCalibrationSource,
    artifacts_by_panel: Mapping[str, ObjectBongardPanelRubricArtifact],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for name, group in (
        ("group_a", ObjectBongardRubricCalibrationGroup.GROUP_A),
        ("group_b", ObjectBongardRubricCalibrationGroup.GROUP_B),
    ):
        panels = tuple(item for item in source.panels if item.group is group)
        result[name] = {
            disposition: sum(
                artifacts_by_panel[item.panel_id].observation.disposition.value
                == disposition
                for item in panels
            )
            for disposition in _DISPOSITIONS
        }
    return result


def _result_record(
    *,
    manifest_digest: str,
    inputs: _ProbeInputs,
    jobs: tuple[_BlindPanelJob, ...],
    artifacts: tuple[ObjectBongardPanelRubricArtifact, ...],
) -> dict[str, Any]:
    by_panel = {item.panel_id: item for item in artifacts}
    if len(by_panel) != PROBE_PANEL_COUNT or set(by_panel) != {
        item.panel_id for item in inputs.source.panels
    }:
        raise ObjectBongardPanelRubricProbeError("completed artifact inventory differs")
    counts = _group_counts(inputs.source, by_panel)
    exact_survivor = (
        counts["group_a"][PanelRubricDisposition.PRESENT.value] == 6
        and counts["group_a"][PanelRubricDisposition.CERTIFIED_ABSENCE.value] == 0
        and counts["group_a"][PanelRubricDisposition.INDETERMINATE.value] == 0
        and counts["group_a"][PanelRubricDisposition.ERROR.value] == 0
        and counts["group_b"][PanelRubricDisposition.CERTIFIED_ABSENCE.value] == 6
        and counts["group_b"][PanelRubricDisposition.PRESENT.value] == 0
        and counts["group_b"][PanelRubricDisposition.INDETERMINATE.value] == 0
        and counts["group_b"][PanelRubricDisposition.ERROR.value] == 0
    )
    return _record(
        {
            "schema": PROBE_RESULT_SCHEMA,
            "status": PROBE_STATUS,
            "manifest_digest": manifest_digest,
            "rank": 0,
            "rubric_spec_digest": inputs.rubric_spec.spec_digest,
            "physical_call_count": sum(item.physical_call_count for item in artifacts),
            "persisted_artifact_count": len(artifacts),
            "model_free_cold_replay_count": len(artifacts),
            "cold_replay_model_calls": 0,
            "group_counts": counts,
            "exact_survivor_rule": "group-a-six-present-and-group-b-six-certified-absence",
            "exact_survivor": exact_survivor,
            "labels_used_for_group_counts_after_all_calls_completed": True,
            "observer_received_group_labels_or_roles": False,
            "old_calibration_authorization_authorizes_probe_jobs": False,
            "benchmark_or_calibration_claim_authorized": False,
            "artifacts": [
                {
                    "probe_index": job.probe_index,
                    "panel_id": job.panel_id,
                    "artifact_digest": artifact.artifact_digest,
                    "runtime_identity_digest": artifact.runtime_identity_digest,
                    "observation_digest": artifact.observation.observation_digest,
                    "disposition": artifact.observation.disposition.value,
                }
                for job, artifact in zip(jobs, artifacts, strict=True)
            ],
            **_authority_data(),
        }
    )


def _verification(
    root: Path,
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
) -> VerifiedObjectBongardPanelRubricProbe:
    counts = result.get("group_counts")
    if not isinstance(counts, Mapping):
        raise ObjectBongardPanelRubricProbeError("result group counts are malformed")
    return VerifiedObjectBongardPanelRubricProbe(
        root,
        manifest["record_digest"],
        result["record_digest"],
        result["exact_survivor"],
        {
            group: dict(value)
            for group, value in counts.items()
            if isinstance(group, str) and isinstance(value, Mapping)
        },
    )


def _run_loaded_probe(
    output_root: str | os.PathLike[str],
    inputs: _ProbeInputs,
    *,
    parallel_workers: int,
    transport: Transport,
) -> VerifiedObjectBongardPanelRubricProbe:
    root = _fresh_output_root(output_root)
    manifest = _manifest(inputs, parallel_workers=parallel_workers)
    _write_once(root / MANIFEST_FILENAME, manifest, "pre-call manifest")
    manifest_digest = manifest["record_digest"]
    jobs = _blind_jobs(inputs)

    completed: dict[int, ObjectBongardPanelRubricArtifact] = {}
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {
            executor.submit(
                _run_blind_job,
                root=root,
                manifest_digest=manifest_digest,
                inputs=inputs,
                job=job,
                transport=transport,
            ): job.probe_index
            for job in jobs
        }
        for future in as_completed(futures):
            completed[futures[future]] = future.result()
    artifacts = tuple(completed[index] for index in range(PROBE_PANEL_COUNT))

    # This is the first point at which group membership affects a computation.
    result = _result_record(
        manifest_digest=manifest_digest,
        inputs=inputs,
        jobs=jobs,
        artifacts=artifacts,
    )
    _write_once(root / RESULT_FILENAME, result, "probe result")
    return _verification(root, manifest, result)


def run_object_bongard_panel_rubric_probe(
    output_root: str | os.PathLike[str],
    *,
    nomination_root: str | os.PathLike[str] = DEFAULT_V10_NOMINATION_ROOT,
    rejected_calibration_root: str | os.PathLike[str] = (
        DEFAULT_REJECTED_V10_CALIBRATION_ROOT
    ),
    source_directory: str | os.PathLike[str] = DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE,
    parallel_workers: int = PROBE_MAX_WORKERS,
    transport: Transport = run_codex_named_images_structured,
) -> VerifiedObjectBongardPanelRubricProbe:
    """Run twelve new diagnostic calls after cold-verifying every v10 parent."""

    inputs = _load_probe_inputs(
        nomination_root=nomination_root,
        rejected_calibration_root=rejected_calibration_root,
        source_directory=source_directory,
    )
    return _run_loaded_probe(
        output_root,
        inputs,
        parallel_workers=parallel_workers,
        transport=transport,
    )


def _verify_loaded_probe(
    output_root: str | os.PathLike[str], inputs: _ProbeInputs
) -> VerifiedObjectBongardPanelRubricProbe:
    root = _existing_output_root(output_root)
    if {item.name for item in root.iterdir()} != {
        MANIFEST_FILENAME,
        RESULT_FILENAME,
        ARTIFACT_DIRECTORY,
        REPLAY_DIRECTORY,
    }:
        raise ObjectBongardPanelRubricProbeError("probe root inventory differs")
    stored_manifest = _verify_record(
        _read_json(root / MANIFEST_FILENAME, "probe manifest"),
        schema=PROBE_MANIFEST_SCHEMA,
        label="probe manifest",
    )
    call_policy = stored_manifest.get("call_policy")
    if not isinstance(call_policy, Mapping):
        raise ObjectBongardPanelRubricProbeError("manifest call policy is malformed")
    workers = call_policy.get("parallel_workers")
    expected_manifest = _manifest(inputs, parallel_workers=workers)
    if stored_manifest != expected_manifest:
        raise ObjectBongardPanelRubricProbeError("pre-call manifest differs on replay")
    jobs = _blind_jobs(inputs)
    expected_files = {_artifact_filename(index) for index in range(PROBE_PANEL_COUNT)}
    artifact_root = root / ARTIFACT_DIRECTORY
    replay_root = root / REPLAY_DIRECTORY
    if (
        not artifact_root.is_dir()
        or not replay_root.is_dir()
        or {item.name for item in artifact_root.iterdir()} != expected_files
        or {item.name for item in replay_root.iterdir()} != expected_files
    ):
        raise ObjectBongardPanelRubricProbeError("artifact/replay inventory differs")
    artifacts: list[ObjectBongardPanelRubricArtifact] = []
    for job in jobs:
        artifact_path = artifact_root / _artifact_filename(job.probe_index)
        artifact_bytes = artifact_path.read_bytes()
        artifact = ObjectBongardPanelRubricArtifact.from_data(
            _read_json(artifact_path, f"artifact {job.probe_index}")
        )
        replayed = verify_object_bongard_panel_rubric_artifact(
            artifact,
            job.exact_png_bytes,
            panel_id=job.panel_id,
            rubric_spec=inputs.rubric_spec,
            expected_artifact_digest=artifact.artifact_digest,
            expected_runtime_identity_digest=artifact.runtime_identity_digest,
        )
        if replayed.observation_context_digest != job.observation_context_digest:
            raise ObjectBongardPanelRubricProbeError("artifact context differs")
        expected_replay = _artifact_replay_record(
            manifest_digest=stored_manifest["record_digest"],
            job=job,
            artifact=replayed,
            artifact_file_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
        )
        stored_replay = _verify_record(
            _read_json(
                replay_root / _artifact_filename(job.probe_index),
                f"artifact replay {job.probe_index}",
            ),
            schema=PROBE_REPLAY_SCHEMA,
            label=f"artifact replay {job.probe_index}",
        )
        if stored_replay != expected_replay:
            raise ObjectBongardPanelRubricProbeError("artifact replay differs")
        artifacts.append(replayed)
    expected_result = _result_record(
        manifest_digest=stored_manifest["record_digest"],
        inputs=inputs,
        jobs=jobs,
        artifacts=tuple(artifacts),
    )
    stored_result = _verify_record(
        _read_json(root / RESULT_FILENAME, "probe result"),
        schema=PROBE_RESULT_SCHEMA,
        label="probe result",
    )
    if stored_result != expected_result:
        raise ObjectBongardPanelRubricProbeError("probe result differs on replay")
    return _verification(root, stored_manifest, stored_result)


def verify_object_bongard_panel_rubric_probe(
    output_root: str | os.PathLike[str],
    *,
    nomination_root: str | os.PathLike[str] = DEFAULT_V10_NOMINATION_ROOT,
    rejected_calibration_root: str | os.PathLike[str] = (
        DEFAULT_REJECTED_V10_CALIBRATION_ROOT
    ),
    source_directory: str | os.PathLike[str] = DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE,
) -> VerifiedObjectBongardPanelRubricProbe:
    """Cold-replay a completed diagnostic directory without model access."""

    inputs = _load_probe_inputs(
        nomination_root=nomination_root,
        rejected_calibration_root=rejected_calibration_root,
        source_directory=source_directory,
    )
    return _verify_loaded_probe(output_root, inputs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or cold-verify the unsealed v10 rank-0 whole-panel diagnostic"
    )
    commands = parser.add_subparsers(dest="operation", required=True)
    for name in ("launch", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--output-root", required=True, type=Path)
        command.add_argument(
            "--nomination-root", type=Path, default=DEFAULT_V10_NOMINATION_ROOT
        )
        command.add_argument(
            "--rejected-calibration-root",
            type=Path,
            default=DEFAULT_REJECTED_V10_CALIBRATION_ROOT,
        )
        command.add_argument(
            "--source-directory",
            type=Path,
            default=DEFAULT_OBJECT_RUBRIC_CALIBRATION_SOURCE,
        )
    commands.choices["launch"].add_argument(
        "--parallel-workers", type=int, default=PROBE_MAX_WORKERS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    try:
        common = {
            "nomination_root": args.nomination_root,
            "rejected_calibration_root": args.rejected_calibration_root,
            "source_directory": args.source_directory,
        }
        if args.operation == "launch":
            verified = run_object_bongard_panel_rubric_probe(
                args.output_root,
                parallel_workers=args.parallel_workers,
                **common,
            )
        else:
            verified = verify_object_bongard_panel_rubric_probe(
                args.output_root, **common
            )
    except Exception as exc:
        print(
            canonical_json(
                {
                    "schema": "gkm.bongard-panel-rubric-probe-error.v1",
                    "status": PROBE_STATUS,
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:2000],
                }
            ).decode("utf-8"),
            file=sys.stderr,
        )
        return 1
    print(canonical_json(verified.summary_data()).decode("utf-8"))
    return 0 if verified.exact_survivor else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = (
    "DEFAULT_REJECTED_V10_CALIBRATION_ROOT",
    "DEFAULT_V10_NOMINATION_ROOT",
    "ObjectBongardPanelRubricProbeError",
    "PROBE_MAX_WORKERS",
    "PROBE_PANEL_COUNT",
    "PROBE_STATUS",
    "VerifiedObjectBongardPanelRubricProbe",
    "main",
    "object_bongard_panel_rubric_probe_source_digest",
    "run_object_bongard_panel_rubric_probe",
    "verify_object_bongard_panel_rubric_probe",
)

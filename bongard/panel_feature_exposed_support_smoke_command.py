"""Exactly-once, query-free headless panel-feature diagnostic.

This command intentionally reuses support pixels from one already-exposed
ShapeBongard TRAIN task.  The source archive contains no query pixels and the
loader refuses it if any query/freeze/rank action was recorded.  One
receipted proposer turn and one complete-catalog batched vision turn per
support panel are journaled before deterministic Python synthesis.  A
support-only rank turn is made only when both orientations are nonempty and a
multi-survivor version space needs salience ordering.  Nothing in this module
can release, observe, freeze for, or score a query panel.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import (
    CodexNoToolsAttestation,
    attest_codex_no_tools,
)
from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTextTurnJournalTransport,
    ObjectBongardTurnRuntime,
)
from bongard.panel_batched_typed_codex_observer import (
    BatchedFeatureAxisRequest,
    TypedBatchedAxisCodexArtifact,
    batched_feature_axis_output_schema,
    batched_feature_axis_prompt,
    complete_whole_panel_feature_axes,
    observe_typed_panel_axes_batched,
)
from bongard.panel_feature_evidence_bundle import (
    PanelFeatureEvidenceBundle,
    PanelFeatureEvidencePanel,
    PanelFeatureEvidencePhase,
    cold_replay_panel_feature_evidence_bundle,
)
from bongard.panel_feature_proposer import (
    PANEL_FEATURE_PRESENTATION_NAMES,
    panel_feature_proposer_output_schema,
    panel_feature_proposer_prompt,
)
from bongard.panel_feature_ranker import (
    PANEL_FEATURE_MAX_RANK_CANDIDATES,
    PanelFeatureRankArtifact,
    PanelFeatureRankInput,
    panel_feature_ranker_output_schema,
    panel_feature_ranker_prompt,
    rank_panel_feature_version_spaces,
)
from bongard.panel_feature_task_runner import (
    PanelFeatureSupportDerivation,
    PanelFeatureSupportDerivationStatus,
    derive_panel_feature_support,
)
from bongard.panel_owner_inventory import PANEL_OWNER_NEUTRAL_IMAGE_NAME
from bongard.panel_soft_ontology import NativeOrientation
from bongard.panel_typed_codex_observer import (
    HeadlessCodexPanelFeatureReceiptedCall,
    TypedProposerCodexCallArtifact,
    build_panel_only_observation_context,
    invoke_receipted_panel_feature_proposer,
)
from bongard.prototype_scene_observer import (
    prototype_scene_transport_source_digest,
)
from bongard.transport import (
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    codex_cli_authenticated_fingerprint,
    run_codex_named_images_structured,
    run_codex_text_structured,
    snapshot_cloud_policy_cache,
    snapshot_pinned_model_catalog,
)


SMOKE_SCHEMA = "gkm.bongard-panel-feature-exposed-support-smoke.v1"
DEFAULT_SOURCE_ARCHIVE = Path(
    "downloads/ShapeBongard_V2_full/"
    "panel_soft_exact_unused_train_20260809_ranked_v1/objects/"
    "panel-soft-task-archive/"
    "235940f43c076a7308cccd89121c0938007e364ea90f5a1a30cfe5e082442c5d.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "downloads/ShapeBongard_V2_full/"
    "panel_feature_exposed_support_smoke_20260809_v1"
)
DEFAULT_LAUNCHER_SHA256 = (
    "19c4f144c5226a9f17c58e6f0fa854843b0f77a6eb420f40e2745a12f10f5d37"
)
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "medium"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_SOURCE_BYTES = 64 * 1024 * 1024


class PanelFeatureExposedSupportSmokeError(RuntimeError):
    """The exposed-support source, runtime, journal, or replay failed closed."""


def _record(body: Mapping[str, Any]) -> dict[str, Any]:
    frozen = json.loads(canonical_json(dict(body)).decode("utf-8"))
    return {**frozen, "record_digest": "sha256:" + canonical_digest(frozen)}


def _write_once_or_verify(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json(dict(value)) + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != payload:
            raise PanelFeatureExposedSupportSmokeError(
                f"existing artifact differs: {path}"
            )
        return
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise PanelFeatureExposedSupportSmokeError("artifact write stalled")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _read_record(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise PanelFeatureExposedSupportSmokeError(f"artifact is a symlink: {path}")
    payload = path.read_bytes()
    if not payload.endswith(b"\n"):
        raise PanelFeatureExposedSupportSmokeError(f"artifact encoding differs: {path}")
    try:
        raw = json.loads(payload[:-1].decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PanelFeatureExposedSupportSmokeError(
            f"artifact is malformed: {path}"
        ) from exc
    if (
        not isinstance(raw, dict)
        or canonical_json(raw) + b"\n" != payload
        or type(raw.get("record_digest")) is not str
    ):
        raise PanelFeatureExposedSupportSmokeError(
            f"artifact is not canonical: {path}"
        )
    body = {key: value for key, value in raw.items() if key != "record_digest"}
    if raw["record_digest"] != "sha256:" + canonical_digest(body):
        raise PanelFeatureExposedSupportSmokeError(
            f"artifact digest differs: {path}"
        )
    return raw


def _read_source(path: Path) -> tuple[
    ObjectBongardTaskPlan, tuple[str, ...], tuple[bytes, ...], str
]:
    if path.is_symlink():
        raise PanelFeatureExposedSupportSmokeError("source archive is a symlink")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= _MAX_SOURCE_BYTES:
        raise PanelFeatureExposedSupportSmokeError("source archive is not bounded")
    payload = path.read_bytes()
    if len(payload) != info.st_size:
        raise PanelFeatureExposedSupportSmokeError("source archive changed while read")
    try:
        raw = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PanelFeatureExposedSupportSmokeError("source archive is malformed") from exc
    if not isinstance(raw, Mapping):
        raise PanelFeatureExposedSupportSmokeError("source archive is not an object")
    forbidden_nonempty = (
        "freeze",
        "freeze_commit",
        "predicate_pair",
        "rank_artifact",
    )
    if any(raw.get(name) is not None for name in forbidden_nonempty):
        raise PanelFeatureExposedSupportSmokeError(
            "source archive already contains a freeze or rank artifact"
        )
    if any(
        raw.get(name) not in ({}, [], 0, None)
        for name in (
            "query_png_base64_by_side",
            "query_artifacts",
            "query_decisions",
            "query_source_calls_made",
            "query_observer_invocations",
            "freeze_commit_calls_made",
            "freeze_reload_calls_made",
            "ranker_callback_invocations",
        )
    ):
        raise PanelFeatureExposedSupportSmokeError(
            "source archive is not an untouched support-gap archive"
        )
    if raw.get("status") != "support_gap":
        raise PanelFeatureExposedSupportSmokeError("source status is not support_gap")
    try:
        task = ObjectBongardTaskPlan.from_data(raw["task_plan"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PanelFeatureExposedSupportSmokeError("task plan is invalid") from exc
    panel_ids = task.side_0_support_panel_ids + task.side_1_support_panel_ids
    encoded = raw.get("support_png_base64_by_panel_id")
    if not isinstance(encoded, Mapping) or set(encoded) != set(panel_ids):
        raise PanelFeatureExposedSupportSmokeError("exact support panel set differs")
    panels: list[bytes] = []
    for panel_id in panel_ids:
        value = encoded[panel_id]
        if type(value) is not str:
            raise PanelFeatureExposedSupportSmokeError("support encoding differs")
        try:
            panel = base64.b64decode(value, validate=True)
        except (ValueError, TypeError) as exc:
            raise PanelFeatureExposedSupportSmokeError("support base64 differs") from exc
        if not panel.startswith(_PNG_SIGNATURE):
            raise PanelFeatureExposedSupportSmokeError("support panel is not PNG")
        panels.append(panel)
    if len({hashlib.sha256(item).digest() for item in panels}) != 12:
        raise PanelFeatureExposedSupportSmokeError("support panel content is duplicated")
    return task, panel_ids, tuple(panels), hashlib.sha256(payload).hexdigest()


def _authorization(
    task: ObjectBongardTaskPlan,
    panel_ids: Sequence[str],
    panels: Sequence[bytes],
    source_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    axes = complete_whole_panel_feature_axes()
    authorization = _record(
        {
            "schema": "gkm.bongard-panel-feature-exposed-support-authorization.v1",
            "source_archive_sha256": source_digest,
            "task_plan": task.to_data(),
            "support_panel_ids": list(panel_ids),
            "support_png_sha256": [hashlib.sha256(item).hexdigest() for item in panels],
            "observer_axis_digests": [item.axis_digest for item in axes],
            "source_archive_query_png_count": 0,
            "query_release_or_observation_authorized": False,
            "engineering_only": True,
            "scientific_benchmark": False,
            "python_is_canonical_authority": True,
            "lean_required": False,
        }
    )
    precommit = _record(
        {
            "schema": "gkm.bongard-panel-feature-exposed-support-precommit.v1",
            "authorization_digest": authorization["record_digest"],
            "physical_call_plan": {
                "proposer": 1,
                "support_batched_observers": 12,
                "support_ranker_maximum": 1,
                "query": 0,
            },
            "proposer_then_observers": True,
            "observer_catalog_fixed_before_model_calls": True,
            "ranker_support_only": True,
            "exactly_once_journals_required": True,
            "query_pixels_available_to_command": False,
            "frozen_predicate_created": False,
            "python_is_canonical_authority": True,
            "lean_required": False,
        }
    )
    return authorization, precommit


def _runtime(
    *,
    output_root: Path,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    model: str,
    reasoning_effort: str,
    minutes: int,
    executable: str,
    launcher_sha256: str,
    verbose: bool,
) -> tuple[ObjectBongardTurnRuntime, dict[str, Any]]:
    prior_path = output_root / "runtime.json"
    if prior_path.exists():
        prior = _read_record(prior_path)
        try:
            cache_encoded = prior["cloud_policy_cache_base64"]
            cache = CloudPolicyCacheSnapshot(
                None
                if cache_encoded is None
                else base64.b64decode(cache_encoded, validate=True)
            )
            catalog = CodexModelCatalogSnapshot(
                base64.b64decode(prior["model_catalog_base64"], validate=True)
            )
            attestation = CodexNoToolsAttestation.from_mapping(
                prior["no_tools_attestation"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PanelFeatureExposedSupportSmokeError(
                "stored runtime preimages differ"
            ) from exc
        fingerprint = codex_cli_authenticated_fingerprint(
            executable, expected_launcher_digest=launcher_sha256
        )
        runtime = ObjectBongardTurnRuntime(
            model=model,
            reasoning_effort=reasoning_effort,
            minutes=minutes,
            verbose=verbose,
            executable=executable,
            cloud_policy_cache_snapshot=cache,
            model_catalog_snapshot=catalog,
            expected_launcher_digest=launcher_sha256,
            no_tools_attestation=attestation,
            transport_source_digest=prototype_scene_transport_source_digest(),
        )
        if (
            prior.get("schema")
            != "gkm.bongard-panel-feature-exposed-support-runtime.v1"
            or prior.get("authorization_digest") != authorization["record_digest"]
            or prior.get("execution_precommit_digest") != precommit["record_digest"]
            or prior.get("launcher_fingerprint") != dict(fingerprint)
            or prior.get("runtime_binding") != runtime.binding
        ):
            raise PanelFeatureExposedSupportSmokeError(
                "stored runtime differs from the live pinned replay request"
            )
        return runtime, prior

    cache = snapshot_cloud_policy_cache()
    catalog = snapshot_pinned_model_catalog()
    fingerprint = codex_cli_authenticated_fingerprint(
        executable, expected_launcher_digest=launcher_sha256
    )
    if fingerprint.get("launcher_digest") != launcher_sha256:
        raise PanelFeatureExposedSupportSmokeError("launcher fingerprint differs")
    attestation = attest_codex_no_tools(
        executable=executable,
        expected_launcher_digest=launcher_sha256,
        model_catalog_snapshot=catalog,
        cloud_policy_cache_snapshot=cache,
    )
    runtime = ObjectBongardTurnRuntime(
        model=model,
        reasoning_effort=reasoning_effort,
        minutes=minutes,
        verbose=verbose,
        executable=executable,
        cloud_policy_cache_snapshot=cache,
        model_catalog_snapshot=catalog,
        expected_launcher_digest=launcher_sha256,
        no_tools_attestation=attestation,
        transport_source_digest=prototype_scene_transport_source_digest(),
    )
    evidence = _record(
        {
            "schema": "gkm.bongard-panel-feature-exposed-support-runtime.v1",
            "authorization_digest": authorization["record_digest"],
            "execution_precommit_digest": precommit["record_digest"],
            "runtime_binding": runtime.binding,
            "launcher_fingerprint": dict(fingerprint),
            "cloud_policy_cache_base64": (
                None if cache.data is None else base64.b64encode(cache.data).decode("ascii")
            ),
            "model_catalog_base64": base64.b64encode(catalog.data).decode("ascii"),
            "no_tools_attestation": attestation.to_dict(),
        }
    )
    _write_once_or_verify(output_root / "runtime.json", evidence)
    return runtime, evidence


def _observe_one(
    *,
    ordinal: int,
    task: ObjectBongardTaskPlan,
    panel: bytes,
    axes: tuple,
    output_root: Path,
    authorization_digest: str,
    precommit_digest: str,
    runtime: ObjectBongardTurnRuntime,
) -> tuple[int, TypedBatchedAxisCodexArtifact, dict[str, Any]]:
    context = build_panel_only_observation_context(
        panel,
        model=runtime.model,
        reasoning_effort=runtime.reasoning_effort,
        expected_launcher_digest=runtime.expected_launcher_digest,
        cloud_policy_cache_snapshot=runtime.cloud_policy_cache_snapshot,
        model_catalog_snapshot=runtime.model_catalog_snapshot,
        no_tools_attestation=runtime.no_tools_attestation,
    )
    request = BatchedFeatureAxisRequest.build(context, axes)
    prompt = batched_feature_axis_prompt(request)
    schema = batched_feature_axis_output_schema(request)
    journal = ObjectBongardNamedImageTurnJournalTransport(
        output_root / "journals" / f"support_{ordinal:02d}",
        authorization_digest=authorization_digest,
        execution_precommit_digest=precommit_digest,
        task_id=task.task_id,
        turn_kind=f"support_axis_{ordinal:02d}",
        expected_prompt=prompt,
        expected_images=((PANEL_OWNER_NEUTRAL_IMAGE_NAME, panel),),
        expected_output_schema=schema,
        runtime=runtime,
        underlying_transport=run_codex_named_images_structured,
    )
    artifact = observe_typed_panel_axes_batched(
        panel,
        axes=axes,
        panel_only_context=context,
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
    summary = journal.verify().to_data()
    _write_once_or_verify(
        output_root / "support_axis_artifacts" / f"{ordinal:02d}.json",
        artifact.to_data(),
    )
    return ordinal, artifact, summary


def run_exposed_support_smoke(
    *,
    source_archive: str | Path = DEFAULT_SOURCE_ARCHIVE,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    minutes: int = 20,
    executable: str = "codex",
    launcher_sha256: str = DEFAULT_LAUNCHER_SHA256,
    workers: int = 4,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run and cold-replay one already-exposed support-only diagnostic."""

    if type(workers) is not int or not 1 <= workers <= 12:
        raise PanelFeatureExposedSupportSmokeError("workers must lie in 1..12")
    source = Path(os.path.abspath(os.fspath(source_archive)))
    root = Path(os.path.abspath(os.fspath(output_root)))
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise PanelFeatureExposedSupportSmokeError("output root is unsafe")
    task, panel_ids, panels, source_digest = _read_source(source)
    axes = complete_whole_panel_feature_axes()
    authorization, precommit = _authorization(task, panel_ids, panels, source_digest)
    _write_once_or_verify(root / "authorization.json", authorization)
    _write_once_or_verify(root / "execution_precommit.json", precommit)
    runtime, runtime_evidence = _runtime(
        output_root=root,
        authorization=authorization,
        precommit=precommit,
        model=model,
        reasoning_effort=reasoning_effort,
        minutes=minutes,
        executable=executable,
        launcher_sha256=launcher_sha256,
        verbose=verbose,
    )

    proposer_prompt = panel_feature_proposer_prompt()
    proposer_schema = panel_feature_proposer_output_schema()
    proposer_images = tuple(zip(PANEL_FEATURE_PRESENTATION_NAMES, panels, strict=True))
    proposer_journal = ObjectBongardNamedImageTurnJournalTransport(
        root / "journals" / "proposer",
        authorization_digest=authorization["record_digest"],
        execution_precommit_digest=precommit["record_digest"],
        task_id=task.task_id,
        turn_kind="feature_proposer",
        expected_prompt=proposer_prompt,
        expected_images=proposer_images,
        expected_output_schema=proposer_schema,
        runtime=runtime,
        underlying_transport=run_codex_named_images_structured,
    )
    call = HeadlessCodexPanelFeatureReceiptedCall(
        task_context_digest=task.record_digest.removeprefix("sha256:"),
        block_orientations=(
            NativeOrientation.SIDE0_POSITIVE,
            NativeOrientation.SIDE1_POSITIVE,
        ),
        model=runtime.model,
        reasoning_effort=runtime.reasoning_effort,
        minutes=runtime.minutes,
        verbose=runtime.verbose,
        executable=runtime.executable,
        cloud_policy_cache_snapshot=runtime.cloud_policy_cache_snapshot,
        expected_launcher_digest=runtime.expected_launcher_digest,
        model_catalog_snapshot=runtime.model_catalog_snapshot,
        no_tools_attestation=runtime.no_tools_attestation,
        transport=proposer_journal,
    )
    proposer_result = invoke_receipted_panel_feature_proposer(panels, call=call)
    proposer_artifact: TypedProposerCodexCallArtifact = call.artifact
    proposer_summary = proposer_journal.verify().to_data()
    _write_once_or_verify(root / "proposer_artifact.json", proposer_artifact.to_data())
    _write_once_or_verify(root / "proposer_result.json", proposer_result.to_data())

    observed: list[TypedBatchedAxisCodexArtifact | None] = [None] * 12
    observer_summaries: list[dict[str, Any] | None] = [None] * 12
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _observe_one,
                ordinal=index,
                task=task,
                panel=panel,
                axes=axes,
                output_root=root,
                authorization_digest=authorization["record_digest"],
                precommit_digest=precommit["record_digest"],
                runtime=runtime,
            )
            for index, panel in enumerate(panels)
        ]
        for future in as_completed(futures):
            index, artifact, summary = future.result()
            observed[index] = artifact
            observer_summaries[index] = summary
    if any(item is None for item in observed + observer_summaries):
        raise PanelFeatureExposedSupportSmokeError("support observation set is incomplete")
    artifacts = tuple(item for item in observed if item is not None)
    summaries = tuple(item for item in observer_summaries if item is not None)
    evidence_panels = tuple(
        PanelFeatureEvidencePanel.derive_from_batched_artifact(
            phase=PanelFeatureEvidencePhase.SUPPORT,
            phase_index=index,
            panel_id=panel_id,
            panel_png=panel,
            batched_axis_artifact=artifact,
        )
        for index, (panel_id, panel, artifact) in enumerate(
            zip(panel_ids, panels, artifacts, strict=True)
        )
    )
    bundle = PanelFeatureEvidenceBundle.create(
        proposer_artifact=proposer_artifact,
        proposer_result=proposer_result,
        observer_axes=axes,
        panels=evidence_panels,
    )
    _write_once_or_verify(root / "evidence_bundle.json", bundle.to_data())

    derivation: PanelFeatureSupportDerivation | None
    if proposer_result.observer_vocabulary is None:
        derivation = None
        proposer_gap = _record(
            {
                "schema": "gkm.bongard-panel-feature-proposer-language-gap.v1",
                "proposer_result_digest": proposer_result.result_digest,
                "language_gaps": [item.to_data() for item in proposer_result.language_gaps],
                "nomination_gaps": [
                    item.to_data() for item in proposer_result.nomination_gaps
                ],
                "observer_vocabulary_present": False,
                "support_version_space_constructed": False,
                "failed_or_uncertain_evidence_counted_as_negative": False,
                "rank_or_freeze_permitted": False,
                "query_permitted": False,
            }
        )
        _write_once_or_verify(root / "proposer_language_gap.json", proposer_gap)
    else:
        derivation = derive_panel_feature_support(
            task,
            panels,
            proposer_result,
            tuple(item.observation_set for item in evidence_panels),
        )
        _write_once_or_verify(root / "support_derivation.json", derivation.to_data())

    rank_artifact: PanelFeatureRankArtifact | None = None
    rank_summary: dict[str, Any] | None = None
    counts = (
        (0, 0)
        if derivation is None
        else (
            len(derivation.side0_version_space.survivor_formula_digests),
            len(derivation.side1_version_space.survivor_formula_digests),
        )
    )
    if (
        derivation is not None
        and 0 not in counts
        and counts != (1, 1)
        and sum(counts) <= PANEL_FEATURE_MAX_RANK_CANDIDATES
    ):
        rank_input = PanelFeatureRankInput.freeze(
            derivation.side0_version_space,
            derivation.side1_version_space,
            proposer_result,
        )
        rank_journal = ObjectBongardTextTurnJournalTransport(
            root / "journals" / "ranker",
            authorization_digest=authorization["record_digest"],
            execution_precommit_digest=precommit["record_digest"],
            task_id=task.task_id,
            turn_kind="feature_ranker",
            expected_prompt=panel_feature_ranker_prompt(rank_input),
            expected_output_schema=panel_feature_ranker_output_schema(rank_input),
            runtime=runtime,
            underlying_transport=run_codex_text_structured,
        )
        rank_artifact = rank_panel_feature_version_spaces(
            derivation.side0_version_space,
            derivation.side1_version_space,
            proposer_result,
            model=runtime.model,
            reasoning_effort=runtime.reasoning_effort,
            minutes=runtime.minutes,
            verbose=runtime.verbose,
            executable=runtime.executable,
            expected_launcher_digest=runtime.expected_launcher_digest,
            cloud_policy_cache_snapshot=runtime.cloud_policy_cache_snapshot,  # type: ignore[arg-type]
            model_catalog_snapshot=runtime.model_catalog_snapshot,
            no_tools_attestation=runtime.no_tools_attestation,
            transport=rank_journal,
        )
        rank_summary = rank_journal.verify().to_data()
        _write_once_or_verify(root / "rank_artifact.json", rank_artifact.to_data())

    cold_replay_panel_feature_evidence_bundle(
        bundle, expected_bundle_address=bundle.bundle_address
    )
    if derivation is not None:
        PanelFeatureSupportDerivation.from_data(derivation.to_data())
    completion = _record(
        {
            "schema": SMOKE_SCHEMA,
            "authorization_digest": authorization["record_digest"],
            "execution_precommit_digest": precommit["record_digest"],
            "runtime_evidence_digest": runtime_evidence["record_digest"],
            "task_id": task.task_id,
            "source_archive_sha256": source_digest,
            "proposer_artifact_digest": proposer_artifact.artifact_digest,
            "proposer_result_digest": proposer_result.result_digest,
            "evidence_bundle_address": bundle.bundle_address,
            "support_derivation_address": (
                None if derivation is None else derivation.artifact_address
            ),
            "support_derivation_status": (
                "proposer_language_gap"
                if derivation is None
                else derivation.status.value
            ),
            "survivor_counts": list(counts),
            "rank_artifact_digest": (
                None if rank_artifact is None else rank_artifact.artifact_digest
            ),
            "proposer_journal": proposer_summary,
            "observer_journals": list(summaries),
            "rank_journal": rank_summary,
            "physical_model_call_count": 13 + (rank_artifact is not None),
            "query_pixel_count": 0,
            "query_release_calls": 0,
            "query_observer_calls": 0,
            "freeze_created": False,
            "cold_replay_model_calls": 0,
            "engineering_only": True,
            "scientific_benchmark": False,
            "python_is_canonical_authority": True,
            "lean_required": False,
        }
    )
    _write_once_or_verify(root / "completion.json", completion)
    return completion


def _metadata_only(source_archive: Path) -> dict[str, Any]:
    task, panel_ids, panels, source_digest = _read_source(source_archive.resolve())
    return {
        "task_id": task.task_id,
        "source_archive_sha256": source_digest,
        "support_panel_count": len(panels),
        "support_panel_ids": list(panel_ids),
        "query_pixel_count": 0,
        "observer_axis_count": len(complete_whole_panel_feature_axes()),
        "observer_axis_families": [
            item.family.value for item in complete_whole_panel_feature_axes()
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--minutes", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--executable", default="codex")
    parser.add_argument("--launcher-sha256", default=DEFAULT_LAUNCHER_SHA256)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args(argv)
    if args.metadata_only:
        result = _metadata_only(args.source_archive)
    else:
        result = run_exposed_support_smoke(
            source_archive=args.source_archive,
            output_root=args.output_root,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            minutes=args.minutes,
            executable=args.executable,
            launcher_sha256=args.launcher_sha256,
            workers=args.workers,
            verbose=args.verbose,
        )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

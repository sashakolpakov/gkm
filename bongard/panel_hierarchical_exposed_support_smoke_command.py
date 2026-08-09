"""Exactly-once, query-free smoke test for hierarchical panel observations.

This command can read only the twelve already-exposed support PNGs retained in
one historical support-gap archive.  It makes one neutral proposer call and one
candidate-blind hierarchical observation call per panel, constructs the full
closed catalog before applying contrast consistency to composite formulas, and
optionally ranks the surviving positive formulas.  It has no query release,
freeze, decision, or scoring API.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from bongard.object_bongard_batch import ObjectBongardTaskPlan
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTextTurnJournalTransport,
    ObjectBongardTurnRuntime,
)
from bongard.panel_batched_typed_codex_observer import (
    complete_whole_panel_feature_axes,
)
from bongard.panel_feature_closed_catalog_inventory import (
    ClosedCatalogSupportInventory,
    cold_replay_closed_catalog_support_inventory,
)
from bongard.panel_feature_exposed_support_smoke_command import (
    DEFAULT_LAUNCHER_SHA256,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SOURCE_ARCHIVE,
    PanelFeatureExposedSupportSmokeError,
    _read_source,
    _record,
    _runtime,
    _write_once_or_verify,
)
from bongard.panel_feature_proposer import (
    PANEL_FEATURE_PRESENTATION_NAMES,
    panel_feature_proposer_output_schema,
    panel_feature_proposer_prompt,
)
from bongard.panel_hierarchical_action_geometry import (
    panel_hierarchical_action_geometry_algorithm_digest,
    panel_hierarchical_action_geometry_source_digest,
)
from bongard.panel_hierarchical_visual_adapter import (
    HierarchicalPanelCodexArtifact,
    HierarchicalPanelObservationRequest,
    hierarchical_panel_output_schema,
    hierarchical_panel_prompt,
    observe_hierarchical_panel,
    panel_hierarchical_visual_adapter_source_digest,
    verify_hierarchical_panel_artifact,
)
from bongard.panel_owner_inventory import PANEL_OWNER_NEUTRAL_IMAGE_NAME
from bongard.panel_positive_formula_ranker import (
    POSITIVE_FORMULA_MAX_RANK_CANDIDATES,
    PositiveFormulaRankArtifact,
    PositiveFormulaRankInput,
    cold_replay_closed_catalog_primary_formula_rank_artifact,
    positive_formula_ranker_output_schema,
    positive_formula_ranker_prompt,
    rank_closed_catalog_primary_formula,
)
from bongard.panel_soft_ontology import NativeOrientation
from bongard.panel_typed_codex_observer import (
    HeadlessCodexPanelFeatureReceiptedCall,
    TypedProposerCodexCallArtifact,
    build_panel_only_observation_context,
    invoke_receipted_panel_feature_proposer,
    verify_typed_proposer_codex_artifact,
)
from bongard.transport import (
    run_codex_named_images_structured,
    run_codex_text_structured,
)


HIERARCHICAL_SMOKE_SCHEMA = (
    "gkm.bongard-panel-hierarchical-exposed-support-smoke.v1"
)
HIERARCHICAL_AUTHORIZATION_SCHEMA = (
    "gkm.bongard-panel-hierarchical-exposed-support-authorization.v1"
)
HIERARCHICAL_PRECOMMIT_SCHEMA = (
    "gkm.bongard-panel-hierarchical-exposed-support-precommit.v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "downloads/ShapeBongard_V2_full/"
    "panel_hierarchical_exposed_support_smoke_20260809_v1"
)


class PanelHierarchicalExposedSupportSmokeError(
    PanelFeatureExposedSupportSmokeError
):
    """The query-free hierarchical diagnostic failed closed."""


def _authorization(
    task: ObjectBongardTaskPlan,
    panel_ids: Sequence[str],
    panels: Sequence[bytes],
    source_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    axes = complete_whole_panel_feature_axes()
    authorization = _record(
        {
            "schema": HIERARCHICAL_AUTHORIZATION_SCHEMA,
            "source_archive_sha256": source_digest,
            "task_plan": task.to_data(),
            "support_panel_ids": list(panel_ids),
            "support_png_sha256": [
                hashlib.sha256(item).hexdigest() for item in panels
            ],
            "observer_axis_digests": [item.axis_digest for item in axes],
            "primary_orientation": NativeOrientation.SIDE0_POSITIVE.value,
            "observer_adapter_source_digest": (
                panel_hierarchical_visual_adapter_source_digest()
            ),
            "geometry_source_digest": (
                panel_hierarchical_action_geometry_source_digest()
            ),
            "geometry_algorithm_digest": (
                panel_hierarchical_action_geometry_algorithm_digest()
            ),
            "candidate_independent_observation": True,
            "macro_carrier_and_micro_texture_are_disjoint": True,
            "composites_enumerated_before_contrast_consistency": True,
            "opposite_orientation_is_diagnostic_only": True,
            "source_archive_query_png_count": 0,
            "query_release_or_observation_authorized": False,
            "engineering_only": True,
            "scientific_benchmark": False,
            "python_is_canonical_authority": True,
            "lean_present": False,
            "lean_required": False,
            "lean_removable": True,
        }
    )
    precommit = _record(
        {
            "schema": HIERARCHICAL_PRECOMMIT_SCHEMA,
            "authorization_digest": authorization["record_digest"],
            "physical_call_plan": {
                "proposer": 1,
                "support_hierarchical_observers": 12,
                "support_positive_formula_ranker_maximum": 1,
                "query": 0,
            },
            "proposer_then_observers": True,
            "observer_catalog_fixed_before_model_calls": True,
            "ranker_support_only": True,
            "exactly_once_journals_required": True,
            "query_pixels_available_to_command": False,
            "frozen_query_predicate_created": False,
            "negative_formula_required": False,
            "negation_or_polarity_flip_allowed": False,
            "python_is_canonical_authority": True,
            "lean_required": False,
        }
    )
    return authorization, precommit


def _observe_one(
    *,
    ordinal: int,
    task: ObjectBongardTaskPlan,
    panel: bytes,
    output_root: Path,
    authorization_digest: str,
    precommit_digest: str,
    runtime: ObjectBongardTurnRuntime,
) -> tuple[int, HierarchicalPanelCodexArtifact, dict[str, Any]]:
    context = build_panel_only_observation_context(
        panel,
        model=runtime.model,
        reasoning_effort=runtime.reasoning_effort,
        expected_launcher_digest=runtime.expected_launcher_digest,
        cloud_policy_cache_snapshot=runtime.cloud_policy_cache_snapshot,
        model_catalog_snapshot=runtime.model_catalog_snapshot,
        no_tools_attestation=runtime.no_tools_attestation,
    )
    request = HierarchicalPanelObservationRequest.build(context)
    journal = ObjectBongardNamedImageTurnJournalTransport(
        output_root / "journals" / f"support_{ordinal:02d}",
        authorization_digest=authorization_digest,
        execution_precommit_digest=precommit_digest,
        task_id=task.task_id,
        turn_kind=f"hierarchical_support_{ordinal:02d}",
        expected_prompt=hierarchical_panel_prompt(request),
        expected_images=((PANEL_OWNER_NEUTRAL_IMAGE_NAME, panel),),
        expected_output_schema=hierarchical_panel_output_schema(request),
        runtime=runtime,
        underlying_transport=run_codex_named_images_structured,
    )
    artifact = observe_hierarchical_panel(
        panel,
        request=request,
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
    if artifact.benchmark_sealable is not True:
        raise PanelHierarchicalExposedSupportSmokeError(
            "journaled hierarchical artifact is not benchmark-sealable"
        )
    verify_hierarchical_panel_artifact(
        artifact, panel, expected_artifact_digest=artifact.artifact_digest
    )
    summary = journal.verify().to_data()
    _write_once_or_verify(
        output_root / "support_hierarchical_artifacts" / f"{ordinal:02d}.json",
        artifact.to_data(),
    )
    return ordinal, artifact, summary


def run_hierarchical_exposed_support_smoke(
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
    """Run and cold-replay the already-exposed support-only diagnostic."""

    if type(workers) is not int or not 1 <= workers <= 12:
        raise PanelHierarchicalExposedSupportSmokeError(
            "workers must lie in 1..12"
        )
    source = Path(os.path.abspath(os.fspath(source_archive)))
    root = Path(os.path.abspath(os.fspath(output_root)))
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise PanelHierarchicalExposedSupportSmokeError("output root is unsafe")

    task, panel_ids, panels, source_digest = _read_source(source)
    if not all("/1/" in item for item in panel_ids[:6]) or not all(
        "/0/" in item for item in panel_ids[6:]
    ):
        raise PanelHierarchicalExposedSupportSmokeError(
            "exposed HD positive orientation is not side 0"
        )
    authorization, precommit = _authorization(
        task, panel_ids, panels, source_digest
    )
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

    proposer_journal = ObjectBongardNamedImageTurnJournalTransport(
        root / "journals" / "proposer",
        authorization_digest=authorization["record_digest"],
        execution_precommit_digest=precommit["record_digest"],
        task_id=task.task_id,
        turn_kind="feature_proposer",
        expected_prompt=panel_feature_proposer_prompt(),
        expected_images=tuple(
            zip(PANEL_FEATURE_PRESENTATION_NAMES, panels, strict=True)
        ),
        expected_output_schema=panel_feature_proposer_output_schema(),
        runtime=runtime,
        underlying_transport=run_codex_named_images_structured,
    )
    proposer_call = HeadlessCodexPanelFeatureReceiptedCall(
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
    proposer_result = invoke_receipted_panel_feature_proposer(
        panels, call=proposer_call
    )
    proposer_artifact: TypedProposerCodexCallArtifact = proposer_call.artifact
    proposer_summary = proposer_journal.verify().to_data()
    _write_once_or_verify(root / "proposer_artifact.json", proposer_artifact.to_data())
    _write_once_or_verify(root / "proposer_result.json", proposer_result.to_data())

    observed: list[HierarchicalPanelCodexArtifact | None] = [None] * 12
    observer_summaries: list[dict[str, Any] | None] = [None] * 12
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _observe_one,
                ordinal=index,
                task=task,
                panel=panel,
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
        raise PanelHierarchicalExposedSupportSmokeError(
            "hierarchical support observation set is incomplete"
        )
    artifacts = tuple(item for item in observed if item is not None)
    summaries = tuple(item for item in observer_summaries if item is not None)

    inventory = ClosedCatalogSupportInventory.create(
        proposer_result,
        tuple(item.observation_set for item in artifacts),
        primary_orientation=NativeOrientation.SIDE0_POSITIVE,
    )
    _write_once_or_verify(root / "closed_catalog_inventory.json", inventory.to_data())
    survivor_count = len(
        inventory.primary_version_space.survivor_formula_digests
    )
    opposite_diagnostic_count = len(
        inventory.opposite_diagnostic_version_space.survivor_formula_digests
    )

    rank_artifact: PositiveFormulaRankArtifact | None = None
    rank_summary: dict[str, Any] | None = None
    if 1 < survivor_count <= POSITIVE_FORMULA_MAX_RANK_CANDIDATES:
        rank_input = PositiveFormulaRankInput.freeze_closed_catalog_inventory(
            inventory
        )
        rank_journal = ObjectBongardTextTurnJournalTransport(
            root / "journals" / "positive_formula_ranker",
            authorization_digest=authorization["record_digest"],
            execution_precommit_digest=precommit["record_digest"],
            task_id=task.task_id,
            turn_kind="positive_formula_ranker",
            expected_prompt=positive_formula_ranker_prompt(rank_input),
            expected_output_schema=positive_formula_ranker_output_schema(rank_input),
            runtime=runtime,
            underlying_transport=run_codex_text_structured,
        )
        rank_artifact = rank_closed_catalog_primary_formula(
            inventory,
            model=runtime.model,
            reasoning_effort=runtime.reasoning_effort,
            minutes=runtime.minutes,
            verbose=runtime.verbose,
            executable=runtime.executable,
            expected_launcher_digest=runtime.expected_launcher_digest,
            cloud_policy_cache_snapshot=runtime.cloud_policy_cache_snapshot,
            model_catalog_snapshot=runtime.model_catalog_snapshot,
            no_tools_attestation=runtime.no_tools_attestation,
            transport=rank_journal,
        )
        if rank_artifact.benchmark_sealable is not True:
            raise PanelHierarchicalExposedSupportSmokeError(
                "journaled positive rank artifact is not benchmark-sealable"
            )
        rank_summary = rank_journal.verify().to_data()
        _write_once_or_verify(root / "rank_artifact.json", rank_artifact.to_data())

    if survivor_count == 0:
        status = "primary_support_gap"
        selected_formula = None
    elif survivor_count == 1:
        status = "primary_unique"
        selected_formula = inventory.primary_version_space.survivor_formulas[0]
    elif survivor_count <= POSITIVE_FORMULA_MAX_RANK_CANDIDATES:
        if rank_artifact is None:
            raise PanelHierarchicalExposedSupportSmokeError(
                "multi-survivor inventory was not ranked"
            )
        status = "primary_ranked"
        selected_formula = rank_artifact.resolve_selected_all_of(
            inventory.primary_version_space,
            source_survivor_inventory_address=inventory.artifact_address,
        )
    else:
        status = "primary_capacity_gap"
        selected_formula = None

    selection = _record(
        {
            "schema": "gkm.bongard-panel-hierarchical-support-selection.v1",
            "inventory_address": inventory.artifact_address,
            "status": status,
            "primary_survivor_count": survivor_count,
            "opposite_survivor_count_diagnostic_only": opposite_diagnostic_count,
            "selected_positive_formula": (
                None if selected_formula is None else selected_formula.to_data()
            ),
            "negative_formula_required": False,
            "query_release_authorized": False,
        }
    )
    _write_once_or_verify(root / "selection.json", selection)

    verify_typed_proposer_codex_artifact(
        proposer_artifact,
        panels,
        expected_artifact_digest=proposer_artifact.artifact_digest,
    )
    for artifact, panel in zip(artifacts, panels, strict=True):
        verify_hierarchical_panel_artifact(
            artifact, panel, expected_artifact_digest=artifact.artifact_digest
        )
    cold_replay_closed_catalog_support_inventory(
        inventory, expected_artifact_address=inventory.artifact_address
    )
    if rank_artifact is not None:
        cold_replay_closed_catalog_primary_formula_rank_artifact(
            rank_artifact,
            inventory=inventory,
            expected_artifact_address=rank_artifact.artifact_address,
            model=runtime.model,
            reasoning_effort=runtime.reasoning_effort,
            expected_launcher_digest=runtime.expected_launcher_digest,
            cloud_policy_cache_snapshot=runtime.cloud_policy_cache_snapshot,
            model_catalog_snapshot=runtime.model_catalog_snapshot,
            no_tools_attestation=runtime.no_tools_attestation,
            require_benchmark_sealable=True,
            expected_transport_provenance=rank_artifact.transport_provenance,
        )

    completion = _record(
        {
            "schema": HIERARCHICAL_SMOKE_SCHEMA,
            "authorization_digest": authorization["record_digest"],
            "execution_precommit_digest": precommit["record_digest"],
            "runtime_evidence_digest": runtime_evidence["record_digest"],
            "task_id": task.task_id,
            "source_archive_sha256": source_digest,
            "proposer_artifact_digest": proposer_artifact.artifact_digest,
            "hierarchical_artifact_digests": [
                item.artifact_digest for item in artifacts
            ],
            "closed_catalog_inventory_address": inventory.artifact_address,
            "selection_digest": selection["record_digest"],
            "status": status,
            "primary_survivor_count": survivor_count,
            "opposite_survivor_count_diagnostic_only": opposite_diagnostic_count,
            "selected_formula_digest": (
                None if selected_formula is None else selected_formula.formula_digest
            ),
            "rank_artifact_address": (
                None if rank_artifact is None else rank_artifact.artifact_address
            ),
            "physical_model_calls": 13 + int(rank_artifact is not None),
            "proposer_journal": proposer_summary,
            "observer_journals": list(summaries),
            "rank_journal": rank_summary,
            "cold_replay_model_calls": 0,
            "query_release_calls": 0,
            "query_observer_calls": 0,
            "query_pixels_available_to_command": False,
            "negative_formula_required": False,
            "negation_or_polarity_flip_allowed": False,
            "engineering_only": True,
            "scientific_benchmark": False,
            "python_is_canonical_authority": True,
            "lean_present": False,
            "lean_required": False,
            "lean_removable": True,
        }
    )
    _write_once_or_verify(root / "completion.json", completion)
    return completion


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", default=str(DEFAULT_SOURCE_ARCHIVE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--minutes", type=int, default=20)
    parser.add_argument("--executable", default="codex")
    parser.add_argument("--launcher-sha256", default=DEFAULT_LAUNCHER_SHA256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    result = run_hierarchical_exposed_support_smoke(
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
    print(result["record_digest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

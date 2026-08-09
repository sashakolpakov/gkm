"""Canonical lifecycle registry for Bongard execution paths.

The repository contains source that must remain importable to cold-verify old
artifacts.  Importability is not permission to spend more pixels or model
calls.  This module makes that distinction explicit and gives retired command
line entry points one shared fail-closed guard.

Python is the executable authority for the active successor.  Lean is neither
an execution dependency nor part of any registered predicate identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class PipelineLifecycle(str, Enum):
    """Operational status, independent of whether source is retained."""

    ACTIVE_DEVELOPMENT = "active_development"
    SHARED_REQUIRED = "shared_required"
    AUDIT_ONLY = "audit_only"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class PipelineRegistration:
    pipeline_id: str
    lifecycle: PipelineLifecycle
    new_execution_authorized: bool
    authorized_scope: str
    entrypoints: tuple[str, ...]
    source_modules: tuple[str, ...]
    removed_source_modules: tuple[str, ...]
    retained_for: tuple[str, ...]
    removal_blockers: tuple[str, ...]
    successor_pipeline_id: str | None = None

    def __post_init__(self) -> None:
        if not self.pipeline_id or not self.authorized_scope:
            raise ValueError("pipeline registration text must be nonempty")
        if len(set(self.entrypoints)) != len(self.entrypoints):
            raise ValueError("pipeline entrypoints must be unique")
        if len(set(self.source_modules)) != len(self.source_modules):
            raise ValueError("pipeline source modules must be unique")
        if len(set(self.removed_source_modules)) != len(
            self.removed_source_modules
        ):
            raise ValueError("removed source modules must be unique")
        if set(self.source_modules) & set(self.removed_source_modules):
            raise ValueError("retained and removed source modules overlap")
        if self.lifecycle in {
            PipelineLifecycle.AUDIT_ONLY,
            PipelineLifecycle.RETIRED,
        } and self.new_execution_authorized:
            raise ValueError("audit-only and retired pipelines cannot execute")
        if (
            self.lifecycle is PipelineLifecycle.RETIRED
            and self.successor_pipeline_id is None
        ):
            raise ValueError("a retired pipeline must name its successor")


class RetiredPipelineExecutionError(RuntimeError):
    """A retained historical implementation was invoked for a new run."""


ACTIVE_SUCCESSOR_PIPELINE_ID = (
    "typed-geometry-calibrated-soft-positive-version-space-python-v1"
)
SHARED_CUSTODY_PIPELINE_ID = "shared-custody-freeze-replay-python-v1"


def _registration(
    pipeline_id: str,
    lifecycle: PipelineLifecycle,
    *,
    new_execution_authorized: bool,
    authorized_scope: str,
    entrypoints: tuple[str, ...],
    source_modules: tuple[str, ...],
    removed_source_modules: tuple[str, ...] = (),
    retained_for: tuple[str, ...],
    removal_blockers: tuple[str, ...] = (),
    successor_pipeline_id: str | None = None,
) -> PipelineRegistration:
    return PipelineRegistration(
        pipeline_id=pipeline_id,
        lifecycle=lifecycle,
        new_execution_authorized=new_execution_authorized,
        authorized_scope=authorized_scope,
        entrypoints=entrypoints,
        source_modules=source_modules,
        removed_source_modules=removed_source_modules,
        retained_for=retained_for,
        removal_blockers=removal_blockers,
        successor_pipeline_id=successor_pipeline_id,
    )


_REGISTRATIONS = (
    _registration(
        ACTIVE_SUCCESSOR_PIPELINE_ID,
        PipelineLifecycle.ACTIVE_DEVELOPMENT,
        new_execution_authorized=True,
        authorized_scope=(
            "typed-observer FIT/calibration and support-only development; "
            "query release remains gated by calibrated support consistency"
        ),
        entrypoints=("bongard.panel_action_count_phase_command",),
        source_modules=(
            "bongard.panel_feature_observation",
            "bongard.panel_feature_empirical_calibration",
            "bongard.panel_feature_closed_catalog_inventory",
            "bongard.panel_positive_atom_slate",
            "bongard.panel_componentwise_positive_prose_observer",
            "bongard.panel_hierarchical_action_geometry",
            "bongard.panel_hierarchical_visual_adapter",
            "bongard.panel_positive_formula_ranker",
            "bongard.panel_feature_extracted_release_gate",
            "bongard.python_predicate_authority",
        ),
        retained_for=("successor implementation",),
    ),
    _registration(
        SHARED_CUSTODY_PIPELINE_ID,
        PipelineLifecycle.SHARED_REQUIRED,
        new_execution_authorized=False,
        authorized_scope="library use by the active successor and cold replay only",
        entrypoints=(),
        source_modules=(
            "bongard.object_bongard_turn_journal",
            "bongard.object_bongard_release_gate",
            "bongard.official_panel_archive",
            "bongard.official_extracted_panel_archive",
            "bongard.panel_feature_evidence_bundle",
            "bongard.panel_positive_prose_evidence_bundle",
        ),
        retained_for=(
            "exactly-once custody",
            "freeze-before-query enforcement",
            "tamper-detecting model-free replay",
        ),
    ),
    _registration(
        "legacy-two-query-episode-cli-v1",
        PipelineLifecycle.RETIRED,
        new_execution_authorized=False,
        authorized_scope="cold verification of already-created run records only",
        entrypoints=("python -m bongard run",),
        source_modules=(
            "bongard.benchmark",
            "bongard.proposer",
            "bongard.prototype_episode",
            "bongard.semantic_episode",
        ),
        retained_for=(
            "legacy run-schema decoding",
            "cold verification of immutable historical records",
        ),
        removal_blockers=(
            "prototype and semantic cold verifiers still import episode types",
            "verification-only decoders have not been split from live executors",
        ),
        successor_pipeline_id=ACTIVE_SUCCESSOR_PIPELINE_ID,
    ),
    _registration(
        "legacy-visual-semantic-calibration-cli-v1",
        PipelineLifecycle.RETIRED,
        new_execution_authorized=False,
        authorized_scope="cold verification of existing Stage-A/Stage-B records only",
        entrypoints=(
            "python -m bongard calibrate-semantic-stage-a",
            "python -m bongard validate-semantic-stage-b",
        ),
        source_modules=(
            "bongard.semantic_calibration_command",
            "bongard.semantic_gated_dev_validation",
            "bongard.semantic_run_verification",
        ),
        retained_for=(
            "historical calibration receipt verification",
            "legacy run replay",
        ),
        removal_blockers=(
            "legacy run verification still consumes Stage-A campaign types",
        ),
        successor_pipeline_id=ACTIVE_SUCCESSOR_PIPELINE_ID,
    ),
    _registration(
        "panel-soft-exact-unused-campaign-v1",
        PipelineLifecycle.RETIRED,
        new_execution_authorized=False,
        authorized_scope="cold replay of the completed zero-coverage campaign only",
        entrypoints=(
            "python -m bongard.panel_soft_engineering_campaign_command",
        ),
        source_modules=(
            "bongard.panel_soft_engineering_campaign_command",
            "bongard.panel_soft_engineering_task_runner",
            "bongard.panel_soft_observer",
            "bongard.panel_soft_proposer",
            "bongard.panel_soft_ranker",
        ),
        retained_for=(
            "75-call zero-coverage audit",
            "historical exposure and replay evidence",
        ),
        removal_blockers=(
            "checked campaign records bind the exact campaign module bytes",
            "cold replay compares the loaded source digest with those records",
        ),
        successor_pipeline_id=ACTIVE_SUCCESSOR_PIPELINE_ID,
    ),
    _registration(
        "panel-feature-exposed-support-smoke-v1",
        PipelineLifecycle.RETIRED,
        new_execution_authorized=False,
        authorized_scope="library helpers and historical support diagnostic replay only",
        entrypoints=(
            "python -m bongard.panel_feature_exposed_support_smoke_command",
        ),
        source_modules=("bongard.panel_feature_exposed_support_smoke_command",),
        retained_for=(
            "shared bounded source loader/runtime helper",
            "historical support diagnostic evidence",
        ),
        removal_blockers=(
            "active atom-slate, contextual, prose, and action-count probes import its bounded source/runtime helpers",
        ),
        successor_pipeline_id=ACTIVE_SUCCESSOR_PIPELINE_ID,
    ),
    _registration(
        "panel-positive-prose-exposed-probe-v1",
        PipelineLifecycle.RETIRED,
        new_execution_authorized=False,
        authorized_scope="historical support diagnostic replay only",
        entrypoints=(
            "python -m bongard.panel_positive_prose_exposed_probe_command",
        ),
        source_modules=(
            "bongard.panel_positive_prose_exposed_probe_command",
        ),
        retained_for=(
            "source-bound prose and typed-count diagnostic receipts",
            "temporary transport and deterministic-transform helpers",
        ),
        removal_blockers=(
            "panel_action_count_phase_command imports _call",
            "panel_positive_contextual_typed_count_probe_command imports _call, _component_conjunction_disposition, _count_four_interval, _count_interval, _deterministic_ink_zoom, _interval, _load_ink_zoom_policy, _load_typed_count_policy, and positive_prose_exposed_probe_source_digest",
            "checked diagnostic completions bind the exact module bytes",
        ),
        successor_pipeline_id=ACTIVE_SUCCESSOR_PIPELINE_ID,
    ),
    _registration(
        "panel-positive-contextual-typed-count-probe-v1",
        PipelineLifecycle.RETIRED,
        new_execution_authorized=False,
        authorized_scope="historical contextual support diagnostic replay only",
        entrypoints=(
            "python -m bongard.panel_positive_contextual_typed_count_probe_command",
        ),
        source_modules=(
            "bongard.panel_positive_contextual_typed_count_probe_command",
        ),
        retained_for=("source-bound contextual support-gap evidence",),
        removal_blockers=(
            "compact audit summary has not yet replaced source-bound replay",
        ),
        successor_pipeline_id=ACTIVE_SUCCESSOR_PIPELINE_ID,
    ),
    _registration(
        "panel-positive-atom-slate-exposed-probe-v1",
        PipelineLifecycle.RETIRED,
        new_execution_authorized=False,
        authorized_scope="historical atom-slate support diagnostic replay only",
        entrypoints=(
            "python -m bongard.panel_positive_atom_slate_exposed_probe_command",
        ),
        source_modules=(
            "bongard.panel_positive_atom_slate_exposed_probe_command",
        ),
        retained_for=("source-bound atom-slate zero-survivor evidence",),
        removal_blockers=(
            "compact audit summary has not yet replaced source-bound replay",
        ),
        successor_pipeline_id=ACTIVE_SUCCESSOR_PIPELINE_ID,
    ),
    _registration(
        "panel-hierarchical-exposed-support-smoke-v1",
        PipelineLifecycle.RETIRED,
        new_execution_authorized=False,
        authorized_scope="historical failure replay only",
        entrypoints=(
            "python -m bongard.panel_hierarchical_exposed_support_smoke_command",
        ),
        source_modules=(),
        removed_source_modules=(
            "bongard.panel_hierarchical_exposed_support_smoke_command",
        ),
        retained_for=(
            "immutable schema/parser failure JSON",
            "retained hierarchical visual adapter and geometry witnesses",
        ),
        successor_pipeline_id=ACTIVE_SUCCESSOR_PIPELINE_ID,
    ),
    _registration(
        "completed-support-diagnostic-artifacts-v1",
        PipelineLifecycle.AUDIT_ONLY,
        new_execution_authorized=False,
        authorized_scope="immutable data inspection and model-free replay only",
        entrypoints=(),
        source_modules=(),
        retained_for=(
            "exposure accounting",
            "failure provenance",
            "reproducible comparison with the successor",
        ),
    ),
)


CANONICAL_PIPELINE_REGISTRY: Mapping[str, PipelineRegistration] = MappingProxyType(
    {item.pipeline_id: item for item in _REGISTRATIONS}
)

if len(CANONICAL_PIPELINE_REGISTRY) != len(_REGISTRATIONS):
    raise RuntimeError("duplicate canonical Bongard pipeline identifier")


def pipeline_registration(pipeline_id: str) -> PipelineRegistration:
    try:
        return CANONICAL_PIPELINE_REGISTRY[pipeline_id]
    except KeyError as exc:
        raise KeyError(f"unknown Bongard pipeline {pipeline_id!r}") from exc


def require_new_pipeline_execution(pipeline_id: str) -> PipelineRegistration:
    """Return an authorized registration or fail before pixels/model calls."""

    registration = pipeline_registration(pipeline_id)
    if not registration.new_execution_authorized:
        successor = (
            ""
            if registration.successor_pipeline_id is None
            else f"; successor={registration.successor_pipeline_id}"
        )
        raise RetiredPipelineExecutionError(
            f"pipeline {pipeline_id!r} is {registration.lifecycle.value} and "
            f"cannot start a new execution ({registration.authorized_scope})"
            f"{successor}"
        )
    return registration


def pipeline_registry_data() -> dict[str, object]:
    """Return a deterministic, plain-data status report."""

    return {
        "schema": "gkm.bongard-pipeline-lifecycle-registry.v1",
        "active_successor_pipeline_id": ACTIVE_SUCCESSOR_PIPELINE_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "physical_retirement_plan": {
            "phase_1_removed_source": [
                "bongard/panel_hierarchical_exposed_support_smoke_command.py",
                "bongard/tests/test_panel_hierarchical_exposed_support_smoke_command.py",
            ],
            "phase_2_extract_shared_helpers_then_remove": {
                "helper_imports_from_panel_feature_exposed_support_smoke_command": {
                    "bongard.panel_action_count_phase_command": [
                        "DEFAULT_LAUNCHER_SHA256",
                        "DEFAULT_MODEL",
                        "DEFAULT_REASONING_EFFORT",
                        "_read_record",
                        "_record",
                        "_runtime",
                        "_write_once_or_verify",
                    ],
                    "bongard.panel_positive_atom_slate_exposed_probe_command": [
                        "DEFAULT_LAUNCHER_SHA256",
                        "DEFAULT_MODEL",
                        "DEFAULT_REASONING_EFFORT",
                        "DEFAULT_SOURCE_ARCHIVE",
                        "_read_source",
                        "_record",
                        "_runtime",
                        "_write_once_or_verify",
                    ],
                    "bongard.panel_positive_contextual_typed_count_probe_command": [
                        "DEFAULT_LAUNCHER_SHA256",
                        "DEFAULT_MODEL",
                        "DEFAULT_REASONING_EFFORT",
                        "DEFAULT_SOURCE_ARCHIVE",
                        "_read_source",
                        "_record",
                        "_runtime",
                        "_write_once_or_verify",
                    ],
                    "bongard.panel_positive_prose_exposed_probe_command": [
                        "DEFAULT_LAUNCHER_SHA256",
                        "DEFAULT_MODEL",
                        "DEFAULT_REASONING_EFFORT",
                        "DEFAULT_SOURCE_ARCHIVE",
                        "PanelFeatureExposedSupportSmokeError",
                        "_read_source",
                        "_record",
                        "_runtime",
                        "_write_once_or_verify",
                    ],
                },
                "helper_imports_from_panel_positive_prose_exposed_probe_command": {
                    "bongard.panel_action_count_phase_command": ["_call"],
                    "bongard.panel_positive_contextual_typed_count_probe_command": [
                        "_call",
                        "_component_conjunction_disposition",
                        "_count_four_interval",
                        "_count_interval",
                        "_deterministic_ink_zoom",
                        "_interval",
                        "_load_ink_zoom_policy",
                        "_load_typed_count_policy",
                        "positive_prose_exposed_probe_source_digest",
                    ],
                },
                "remove_source_and_exclusive_tests": [
                    "bongard/panel_feature_exposed_support_smoke_command.py",
                    "bongard/tests/test_panel_feature_exposed_support_smoke_command.py",
                    "bongard/panel_positive_prose_exposed_probe_command.py",
                    "bongard/tests/test_panel_positive_prose_exposed_probe_command.py",
                    "bongard/panel_positive_contextual_typed_count_probe_command.py",
                    "bongard/tests/test_panel_positive_contextual_typed_count_probe_command.py",
                    "bongard/panel_positive_atom_slate_exposed_probe_command.py",
                    "bongard/tests/test_panel_positive_atom_slate_exposed_probe_command.py",
                ],
            },
            "audit_artifact_policy": {
                "immutable_compact_records_to_retain": [
                    "bongard/data/historical_exposure_v1.json",
                    "bongard/data/panel_hierarchical_exposed_support_smoke_20260809_v1.failure.json",
                    "bongard/data/panel_positive_live_support_diagnostic_summary_20260809_v1.json",
                    "bongard/data/panel_action_count_measurement_fit_outcome_20260809_v1.json",
                    "bongard/data/panel_convex_four_lines_same_family_train_drill_20260809_v1.json",
                ],
                "raw_evidence_trees_pending_compaction": [
                    "downloads/ShapeBongard_V2_full/panel_soft_exact_unused_train_20260809_ranked_v1",
                    "downloads/ShapeBongard_V2_full/panel_feature_exposed_support_smoke_20260809_v1",
                    "downloads/ShapeBongard_V2_full/panel_hierarchical_exposed_support_smoke_20260809_v1",
                    "downloads/ShapeBongard_V2_full/panel_positive_atom_slate_exposed_probe_20260809_v1",
                    "downloads/ShapeBongard_V2_full/panel_positive_contextual_typed_count_probe_20260809_v1",
                    "downloads/ShapeBongard_V2_full/panel_positive_prose_exposed_probe_20260809_v1",
                    "downloads/ShapeBongard_V2_full/panel_positive_prose_exposed_probe_20260809_v2",
                    "downloads/ShapeBongard_V2_full/panel_positive_prose_known_semantic_probe_20260809_v1",
                    "downloads/ShapeBongard_V2_full/panel_positive_prose_componentwise_known_semantic_probe_20260809_v1",
                    "downloads/ShapeBongard_V2_full/panel_positive_prose_componentwise_known_semantic_probe_20260809_v2",
                    "downloads/ShapeBongard_V2_full/panel_positive_prose_componentwise_zoomed_known_semantic_probe_20260809_v1",
                    "downloads/ShapeBongard_V2_full/panel_positive_prose_corrected_action_zoomed_probe_20260809_v1",
                    "downloads/ShapeBongard_V2_full/panel_positive_typed_count_zoomed_probe_20260809_v1",
                ],
                "raw_evidence_tree_policy": (
                    "retain until compact summaries, exposure accounting, and "
                    "zero-call replay are independently verified; then remove "
                    "redundant derived views but retain irreducible receipts and preimages"
                ),
            },
        },
        "pipelines": [
            {
                "pipeline_id": item.pipeline_id,
                "lifecycle": item.lifecycle.value,
                "new_execution_authorized": item.new_execution_authorized,
                "authorized_scope": item.authorized_scope,
                "entrypoints": list(item.entrypoints),
                "source_modules": list(item.source_modules),
                "removed_source_modules": list(item.removed_source_modules),
                "retained_for": list(item.retained_for),
                "removal_blockers": list(item.removal_blockers),
                "successor_pipeline_id": item.successor_pipeline_id,
            }
            for item in CANONICAL_PIPELINE_REGISTRY.values()
        ],
    }


__all__ = (
    "ACTIVE_SUCCESSOR_PIPELINE_ID",
    "CANONICAL_PIPELINE_REGISTRY",
    "PipelineLifecycle",
    "PipelineRegistration",
    "RetiredPipelineExecutionError",
    "SHARED_CUSTODY_PIPELINE_ID",
    "pipeline_registration",
    "pipeline_registry_data",
    "require_new_pipeline_execution",
)

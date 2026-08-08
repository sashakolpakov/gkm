"""Sealed historical calibration for neutral scene-predicate discovery.

The command uses only twelve already-exposed historical panels.  It freezes a
predicate-independent object inventory and makes one group-blind discovery
call per panel.  After that blind batch is durable, it reveals the committed
support roles to one zero-image semantic proposer.  The proposer must supply
affirmative scoped concepts for both support orientations in one turn.  Their
union is frozen before two independent, role-blind registered-evaluation
passes.

Python constructs and verifies both support orientations.  An empty survivor
set is a typed gap and makes no ranker call.  Otherwise one mandatory
zero-image Codex turn may name exactly one frozen survivor digest; it cannot
invent or edit a formula.  The selected formula is durably frozen and every
visual and text journal is cold-replayed with physical transport forbidden.
Lean is absent, removable, and decision-inert.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import argparse
import base64
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
from bongard import object_bongard_rubric_nomination_command as _durable
from bongard.object_bongard_panel_rubric_calibration import (
    DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE,
    ObjectBongardPanelRubricCalibrationPanel,
    ObjectBongardPanelRubricCalibrationSource,
    load_object_bongard_panel_rubric_calibration_source,
    object_bongard_panel_rubric_calibration_source_digest,
)
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTextTurnJournalTransport,
    ObjectBongardTurnRuntime,
    object_bongard_turn_journal_source_digest,
    verify_object_bongard_turn_journal,
)
from bongard.object_bongard_scene_predicate_ir import (
    SCENE_CALIBRATION_BUNDLE_SCHEMA,
)
from bongard.prototype_scene_observer import prototype_scene_transport_source_digest
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.transport import (
    PINNED_CODEX_CLI_VERSION,
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexStructuredResult,
    codex_cli_authenticated_fingerprint,
    run_codex_named_images_structured,
    run_codex_text_structured,
    snapshot_cloud_policy_cache,
    snapshot_pinned_model_catalog,
    validate_codex_strict_output_schema,
)


COMMAND_ID = "bongard.scene-predicate-calibration/describe-propose-register-rank-v5"
AUTHORIZATION_SCHEMA = "gkm.bongard-scene-predicate-calibration-authorization.v5"
PRECOMMIT_SCHEMA = "gkm.bongard-scene-predicate-calibration-precommit.v5"
DISCOVERY_BATCH_SCHEMA = "gkm.bongard-scene-predicate-discovery-batch.v5"
DISCOVERY_FREEZE_SCHEMA = "gkm.bongard-scene-predicate-discovery-freeze.v5"
REGISTRY_FREEZE_SCHEMA = "gkm.bongard-scene-predicate-registry-freeze.v5"
EVALUATION_BATCH_SCHEMA = "gkm.bongard-scene-predicate-evaluation-batch.v5"
EVALUATION_FREEZE_SCHEMA = "gkm.bongard-scene-predicate-evaluation-freeze.v5"
ROLE_REVEAL_SCHEMA = "gkm.bongard-scene-predicate-role-reveal.v5"
SEMANTIC_PROPOSAL_INPUT_SCHEMA = (
    "gkm.bongard-scene-semantic-registry-proposal-input.v4"
)
SEMANTIC_PROPOSAL_RESULT_SCHEMA = (
    "gkm.bongard-scene-semantic-registry-proposal-result.v4"
)
ASSESSMENT_SCHEMA = "gkm.bongard-scene-predicate-calibration-assessment.v5"
RANK_INPUT_FREEZE_SCHEMA = "gkm.bongard-scene-predicate-rank-input-freeze.v5"
RANK_RESULT_SCHEMA = "gkm.bongard-scene-predicate-rank-result.v5"
FORMULA_FREEZE_SCHEMA = "gkm.bongard-scene-predicate-formula-freeze.v5"
REPLAY_SCHEMA = "gkm.bongard-scene-predicate-calibration-cold-replay.v5"
RESULT_SCHEMA = "gkm.bongard-scene-predicate-calibration-result.v5"
IR_BUNDLE_SCHEMA = SCENE_CALIBRATION_BUNDLE_SCHEMA

AUTHORIZATION_FILENAME = "authorization.json"
PRECOMMIT_FILENAME = "execution_precommit.json"
DISCOVERY_BATCH_FILENAME = "discovery_batch.json"
DISCOVERY_FREEZE_FILENAME = "discovery_freeze.json"
REGISTRY_FREEZE_FILENAME = "soft_tag_registry.json"
EVALUATION_A_BATCH_FILENAME = "registered_evaluation_a_batch.json"
EVALUATION_B_BATCH_FILENAME = "registered_evaluation_b_batch.json"
EVALUATION_FREEZE_FILENAME = "registered_evaluation_freeze.json"
ROLE_REVEAL_FILENAME = "role_reveal.json"
SEMANTIC_PROPOSAL_INPUT_FILENAME = "semantic_registry_proposal_input.json"
SEMANTIC_PROPOSAL_RESULT_FILENAME = "semantic_registry_proposal_result.json"
ASSESSMENT_FILENAME = "assessment.json"
RANK_INPUT_FREEZE_FILENAME = "rank_input_freeze.json"
RANK_RESULT_FILENAME = "rank_result.json"
FORMULA_FREEZE_FILENAME = "formula_freeze.json"
REPLAY_FILENAME = "cold_replay.json"
RESULT_FILENAME = "result.json"
JOURNAL_DIRECTORY = "journals"

MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "medium"
DEFAULT_MINUTES = 15
DEFAULT_EXECUTABLE = "codex"
DEFAULT_EXPECTED_LAUNCHER_SHA256 = (
    "19c4f144c5226a9f17c58e6f0fa854843b0f77a6eb420f40e2745a12f10f5d37"
)
DEFAULT_PARALLEL_WORKERS = 4
MAX_PARALLEL_WORKERS = 4
PANEL_COUNT = 12
DISCOVERY_VISUAL_CALL_COUNT = 12
REGISTERED_EVALUATION_VISUAL_CALL_COUNT_PER_PASS = 12
REGISTERED_EVALUATION_PASS_COUNT = 2
REGISTERED_EVALUATION_VISUAL_CALL_COUNT = 24
VISUAL_CALL_COUNT = 36
SEMANTIC_REGISTRY_PROPOSER_CALL_COUNT = 1
ACCEPTED_RANKER_CALL_COUNT = 1
ACCEPTED_PHYSICAL_CALL_COUNT = 38
MAX_REGISTERED_SOFT_TAGS = 32
TYPED_SEMANTIC_PROPOSAL_GAP = "typed_semantic_proposal_gap"
TYPED_LANGUAGE_GAP = "typed_language_gap"
TYPED_SELECTIVITY_GAP = "typed_selectivity_gap"
TYPED_GROUNDING_REPEATABILITY_GAP = "typed_grounding_repeatability_gap"
TYPED_CALIBRATION_GAP_STATUSES = (
    TYPED_SEMANTIC_PROPOSAL_GAP,
    TYPED_LANGUAGE_GAP,
    TYPED_SELECTIVITY_GAP,
    TYPED_GROUNDING_REPEATABILITY_GAP,
)

_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")

NamedImageTransport = Callable[..., CodexStructuredResult]
TextTransport = Callable[..., CodexStructuredResult]


class ObjectBongardScenePredicateCalibrationCommandError(RuntimeError):
    """A blind phase, role reveal, version space, rank, or replay differs."""


def object_bongard_scene_predicate_calibration_command_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "discovery_then_two_independent_registered_evaluations": True,
        "historical_already_exposed_panel_count": PANEL_COUNT,
        "visual_observation_call_count": VISUAL_CALL_COUNT,
        "semantic_registry_proposer_call_count": (
            SEMANTIC_REGISTRY_PROPOSER_CALL_COUNT
        ),
        "accepted_ranker_call_count": ACCEPTED_RANKER_CALL_COUNT,
        "discovery_omission_means_absence": False,
        "registered_soft_tag_requires_explicit_cells_in_passes_a_and_b": True,
        "registered_soft_tag_cells_are_ordered_witness_cells": True,
        "registered_macro_state_returned_by_model": False,
        "registered_macro_compiled_by_python": True,
        "registered_soft_tag_requires_two_repeat_cells": True,
        "registered_cell_merge_rule": (
            "merge-corresponding-witnesses-first;then-error-dominant;"
            "else-any-absent=absent;else-all-present=present;"
            "else-indeterminate"
        ),
        "soft_tag_minimum_distinct_panel_frequency": 2,
        "soft_tag_order": (
            "descending-distinct-cited-panel-frequency-then-scope-then-phrase"
        ),
        "maximum_registered_soft_tags": MAX_REGISTERED_SOFT_TAGS,
        "all_dropped_tags_and_reasons_persisted": True,
        "fixed_typed_observables_always_registered": True,
        "blind_discovery_frozen_before_support_role_reveal": True,
        "support_roles_revealed_before_semantic_synthesis": True,
        "benchmark_acceptance_requires_role_aware_semantic_registry": True,
        "exact_frequency_registry_acceptance_authorized": False,
        "affirmative_concepts_for_both_orientations_proposed_in_one_call": True,
        "semantic_invalid_optional_rows_are_quarantined_with_provenance": True,
        "semantic_structural_or_zero_orientation_payload_is_typed_gap_not_absence": True,
        "one_scoped_union_registry_frozen_before_registered_evaluation": True,
        "registered_evaluator_receives_support_roles": False,
        "concept_or_formula_added_after_registered_evaluation": False,
        "both_support_orientations_built_by_python": True,
        "empty_survivor_set_is_typed_gap": True,
        "ranker_called_on_empty_survivor_set": False,
        "codex_may_name_one_frozen_survivor_only": True,
        "codex_may_invent_or_edit_formula": False,
        "coverage_selectivity_repeatability_gates_separate": True,
        "query_pixels_used": False,
        "fresh_unused_or_broad_cohort_pixels_used": False,
        "official_test_pixels_used": False,
        "historical_exposure_ledger_append_required": False,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_artifact_identity_selection_decision_or_replay": False,
        "lean_required_for_replay": False,
    }


def _raw_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_DIGEST.fullmatch(value) is None:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            f"{label} must be a sha256: address"
        )
    return value


def _fresh_root(value: str | os.PathLike[str]) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    parent = candidate.parent.resolve(strict=True)
    root = parent / candidate.name
    if not candidate.name or os.path.lexists(root):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "calibration output root must be fresh"
        )
    root.mkdir(mode=0o700)
    _durable._fsync_directory(parent)
    return root


def _existing_root(value: str | os.PathLike[str]) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "calibration root cannot be a symlink"
        )
    root = candidate.resolve(strict=True)
    if not root.is_dir():
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "calibration root is not a directory"
        )
    return root


def _write_and_reload(
    path: Path, value: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    _durable._write_once(path, value, label)
    restored = _durable._read_record(path, label)
    if restored != dict(value):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            f"persisted {label} differs"
        )
    return restored


def _neutral_panel_content(value: "_NeutralPanel") -> dict[str, object]:
    return {
        "schema": "gkm.bongard-scene-predicate-neutral-historical-panel.v1",
        "ordinal": value.ordinal,
        "blind_panel_id": value.blind_panel_id,
        "journal_task_id": value.journal_task_id,
        "lineage_task_id": value.task_id,
        "lineage_panel_id": value.panel_id,
        "released_record_digest": value.released_record_digest,
        "png_sha256": value.png_sha256,
        "historical_role_or_group_serialized": False,
        "already_exposed": True,
    }


@dataclass(frozen=True, slots=True)
class _NeutralPanel:
    ordinal: int
    blind_panel_id: str
    journal_task_id: str
    task_id: str
    panel_id: str
    released_record_digest: str
    png_sha256: str
    exact_png_bytes: bytes
    neutral_panel_digest: str

    def __post_init__(self) -> None:
        match = re.fullmatch(r"calibration_panel_([0-9]{2})", self.blind_panel_id)
        if (
            type(self.ordinal) is not int
            or self.ordinal < 0
            or match is None
            or self.journal_task_id
            != f"bd_scene_calibration_{match.group(1)}"
            or not isinstance(self.task_id, str)
            or not self.task_id
            or not isinstance(self.panel_id, str)
            or not self.panel_id
            or not isinstance(self.exact_png_bytes, bytes)
            or not self.exact_png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
            or hashlib.sha256(self.exact_png_bytes).hexdigest() != self.png_sha256
            or canonical_digest(_neutral_panel_content(self))
            != self.neutral_panel_digest
        ):
            raise ObjectBongardScenePredicateCalibrationCommandError(
                "neutral historical panel binding differs"
            )
        _address(self.released_record_digest, "released record digest")
        _raw_digest(self.png_sha256, "historical PNG digest")
        _raw_digest(self.neutral_panel_digest, "neutral panel digest")

    @classmethod
    def from_historical(
        cls, panel: ObjectBongardPanelRubricCalibrationPanel, blind_index: int
    ) -> "_NeutralPanel":
        if type(blind_index) is not int or not 0 <= blind_index < PANEL_COUNT:
            raise ObjectBongardScenePredicateCalibrationCommandError(
                "blind panel index differs"
            )
        blind_panel_id = f"calibration_panel_{blind_index:02d}"
        journal_task_id = f"bd_scene_calibration_{blind_index:02d}"
        provisional = object.__new__(cls)
        values = (
            panel.ordinal,
            blind_panel_id,
            journal_task_id,
            panel.task_id,
            panel.panel_id,
            panel.released_record_digest,
            panel.png_sha256,
            panel.exact_png_bytes,
        )
        for name, item in zip(
            (
                "ordinal", "blind_panel_id", "journal_task_id", "task_id",
                "panel_id", "released_record_digest", "png_sha256",
                "exact_png_bytes",
            ),
            values,
            strict=True,
        ):
            object.__setattr__(provisional, name, item)
        return cls(*values, canonical_digest(_neutral_panel_content(provisional)))

    def commitment_data(self) -> dict[str, object]:
        return {
            **_neutral_panel_content(self),
            "neutral_panel_digest": self.neutral_panel_digest,
        }


@dataclass(frozen=True, slots=True)
class _CalibrationInputs:
    source: ObjectBongardPanelRubricCalibrationSource
    panels: tuple[_NeutralPanel, ...]
    inventories: tuple[Any, ...]
    atlas_png_by_panel_digest: Mapping[str, tuple[tuple[str, bytes], ...]]
    role_reveal_rows: tuple[Mapping[str, object], ...]
    role_commitment_digest: str


@dataclass(frozen=True, slots=True)
class VerifiedObjectBongardScenePredicateCalibration:
    output_root: Path
    source_digest: str
    authorization_digest: str
    execution_precommit_digest: str
    discovery_batch_digest: str
    discovery_freeze_digest: str
    registry_digest: str
    evaluation_a_batch_digest: str
    evaluation_b_batch_digest: str
    evaluation_freeze_digest: str
    role_reveal_digest: str
    semantic_proposal_digest: str
    assessment_digest: str
    rank_input_freeze_digest: str
    rank_result_digest: str
    formula_freeze_digest: str
    replay_digest: str
    result_digest: str
    status: str
    selected_survivor_digest: str | None
    visual_fresh_call_count: int
    semantic_proposer_fresh_call_count: int
    ranker_fresh_call_count: int

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"


def _role_reveal_rows(
    source: ObjectBongardPanelRubricCalibrationSource,
    panels: Sequence[_NeutralPanel],
) -> tuple[Mapping[str, object], ...]:
    by_ordinal = {item.ordinal: item for item in source.panels}
    rows = tuple(
        {
            "ordinal": panel.ordinal,
            "blind_panel_id": panel.blind_panel_id,
            "neutral_panel_digest": panel.neutral_panel_digest,
            "historical_role": by_ordinal[panel.ordinal].group_index,
        }
        for panel in panels
    )
    if tuple(item["historical_role"] for item in rows) != (0,) * 6 + (1,) * 6:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "historical role inventory differs"
        )
    return rows


def _role_commitment(rows: Sequence[Mapping[str, object]]) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-scene-predicate-role-commitment.v1",
            "rows": [dict(item) for item in rows],
            "reveal_after_blind_discovery_freeze_before_semantic_synthesis": True,
        }
    )


def _assert_role_blind(value: object, label: str) -> None:
    def keys(item: object) -> tuple[str, ...]:
        if isinstance(item, Mapping):
            return tuple(str(key) for key in item) + tuple(
                nested
                for child in item.values()
                for nested in keys(child)
            )
        if isinstance(item, (list, tuple)):
            return tuple(nested for child in item for nested in keys(child))
        return ()

    forbidden = {"group_index", "historical_role", "support_role", "side"}
    if any(key in forbidden for key in keys(value)):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            f"{label} leaks historical roles"
        )


def _load_inputs(
    source_root: str | os.PathLike[str],
) -> _CalibrationInputs:
    from bongard.object_scene_visual_frontend import (
        extract_object_scene_proposal_inventory,
        render_object_scene_proposal_atlas,
    )

    source = load_object_bongard_panel_rubric_calibration_source(source_root)
    panels = tuple(
        _NeutralPanel.from_historical(panel, index)
        for index, panel in enumerate(source.panels)
    )
    if (
        len(panels) != PANEL_COUNT
        or tuple(item.blind_panel_id for item in panels)
        != tuple(f"calibration_panel_{index:02d}" for index in range(PANEL_COUNT))
        or len({item.png_sha256 for item in panels}) != PANEL_COUNT
    ):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "neutral historical panel inventory differs"
        )
    inventories: list[Any] = []
    atlas_by_panel: dict[str, tuple[tuple[str, bytes], ...]] = {}
    for panel in panels:
        inventory = extract_object_scene_proposal_inventory(panel.exact_png_bytes)
        if inventory.panel_digest != panel.png_sha256:
            raise ObjectBongardScenePredicateCalibrationCommandError(
                "proposal inventory differs from historical PNG"
            )
        atlas = render_object_scene_proposal_atlas(
            inventory, panel.exact_png_bytes
        )
        inventories.append(inventory)
        atlas_by_panel[panel.neutral_panel_digest] = atlas
    rows = _role_reveal_rows(source, panels)
    return _CalibrationInputs(
        source,
        panels,
        tuple(inventories),
        atlas_by_panel,
        rows,
        _role_commitment(rows),
    )


def _source_identities() -> list[dict[str, str]]:
    from bongard.object_scene_visual_frontend import (
        object_scene_visual_frontend_source_digest,
    )
    from bongard.object_bongard_scene_predicate_ir import (
        object_bongard_scene_predicate_ir_source_digest,
    )
    from bongard.object_scene_semantic_registry import (
        object_scene_semantic_registry_source_digest,
    )

    rows = {
        "calibration_command_source_sha256": (
            object_bongard_scene_predicate_calibration_command_source_digest()
        ),
        "historical_panel_source_sha256": (
            object_bongard_panel_rubric_calibration_source_digest()
        ),
        "neutral_visual_frontend_source_sha256": (
            object_scene_visual_frontend_source_digest()
        ),
        "scene_predicate_ir_source_sha256": (
            object_bongard_scene_predicate_ir_source_digest()
        ),
        "scene_semantic_registry_source_sha256": (
            object_scene_semantic_registry_source_digest()
        ),
        "turn_journal_source_sha256": object_bongard_turn_journal_source_digest(),
        "transport_source_sha256": prototype_scene_transport_source_digest(),
        "durable_record_helper_source_sha256": (
            _durable.object_bongard_rubric_nomination_command_source_digest()
        ),
    }
    return [{"role": key, "sha256": rows[key]} for key in sorted(rows)]


def _validate_runtime_selectors(
    *,
    parallel_workers: int,
    minutes: int,
    executable: str,
    expected_launcher_sha256: str,
) -> None:
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
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "calibration runtime selectors are invalid"
        )


def _authorization(
    inputs: _CalibrationInputs,
    *,
    parallel_workers: int,
    minutes: int,
    executable: str,
    expected_launcher_sha256: str,
) -> dict[str, Any]:
    from bongard.object_scene_visual_frontend import (
        object_scene_inventory_protocol_digest,
    )
    from bongard.object_scene_semantic_registry import (
        object_scene_semantic_registry_protocol_digest,
    )

    _validate_runtime_selectors(
        parallel_workers=parallel_workers,
        minutes=minutes,
        executable=executable,
        expected_launcher_sha256=expected_launcher_sha256,
    )
    neutral_rows = [
        {
            "panel": panel.commitment_data(),
            "proposal_inventory": inventory.to_data(),
            "atlas_png_commitments": [
                {
                    "name": name,
                    "byte_count": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                for name, payload in inputs.atlas_png_by_panel_digest[
                    panel.neutral_panel_digest
                ]
            ],
        }
        for panel, inventory in zip(inputs.panels, inputs.inventories, strict=True)
    ]
    _assert_role_blind(neutral_rows, "authorization blind inventory")
    return _durable._record(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "command_id": COMMAND_ID,
            "historical_source_digest": inputs.source.source_digest,
            "historical_plan_file_sha256": inputs.source.historical_plan_file_sha256,
            "historical_plan_record_digest": inputs.source.historical_plan_record_digest,
            "neutral_panel_inventory": neutral_rows,
            "inventory_protocol_digest": object_scene_inventory_protocol_digest(),
            "semantic_registry_protocol_digest": (
                object_scene_semantic_registry_protocol_digest()
            ),
            "role_commitment_digest": inputs.role_commitment_digest,
            "role_reveal_serialized": False,
            "phase_order": [
                "discovery_12",
                "discovery_freeze",
                "support_role_reveal",
                "semantic_registry_proposer_1",
                "scoped_union_registry_freeze",
                "registered_evaluation_a_12",
                "registered_evaluation_b_12",
                "joint_registered_evaluation_freeze",
                "python_version_spaces",
                "conditional_survivor_rank",
                "formula_freeze",
                "cold_replay",
            ],
            "discovery_visual_calls": DISCOVERY_VISUAL_CALL_COUNT,
            "registered_evaluation_visual_calls": (
                REGISTERED_EVALUATION_VISUAL_CALL_COUNT
            ),
            "registered_evaluation_pass_count": (
                REGISTERED_EVALUATION_PASS_COUNT
            ),
            "exact_visual_call_count": VISUAL_CALL_COUNT,
            "exact_semantic_registry_proposer_call_count": (
                SEMANTIC_REGISTRY_PROPOSER_CALL_COUNT
            ),
            "accepted_total_physical_call_count": ACCEPTED_PHYSICAL_CALL_COUNT,
            "parallel_workers": parallel_workers,
            "source_identities": _source_identities(),
            "runtime_policy": {
                "model": MODEL,
                "reasoning_effort": REASONING_EFFORT,
                "minutes": minutes,
                "verbose": False,
                "executable": executable,
                "expected_launcher_sha256": expected_launcher_sha256,
            },
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
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "runtime snapshotter returned the wrong type"
        )
    fingerprint = launcher_fingerprinter(
        policy["executable"],
        expected_launcher_digest=policy["expected_launcher_sha256"],
    )
    if dict(fingerprint) != {
        "version": PINNED_CODEX_CLI_VERSION,
        "launcher_digest": policy["expected_launcher_sha256"],
    }:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "authenticated launcher fingerprint differs"
        )
    attestation = runtime_attester(
        executable=policy["executable"],
        expected_launcher_digest=policy["expected_launcher_sha256"],
        model_catalog_snapshot=catalog,
        cloud_policy_cache_snapshot=cache,
    )
    if not isinstance(attestation, CodexNoToolsAttestation):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "no-tools attester returned the wrong type"
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
            "historical_source_digest": inputs.source.source_digest,
            "neutral_panel_digests": [
                item.neutral_panel_digest for item in inputs.panels
            ],
            "proposal_inventory_digests": [
                item.inventory_digest for item in inputs.inventories
            ],
            "role_commitment_digest": inputs.role_commitment_digest,
            "semantic_registry_protocol_digest": authorization[
                "semantic_registry_protocol_digest"
            ],
            "role_reveal_serialized": False,
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
            "precommit_fsynced_before_any_visual_or_text_call": True,
            **_authority_data(),
        },
        "precommit_digest",
    )


def _runtime_from_precommit(
    precommit: Mapping[str, Any], authorization: Mapping[str, Any]
) -> ObjectBongardTurnRuntime:
    raw = _durable._validate_record(
        precommit,
        schema=PRECOMMIT_SCHEMA,
        digest_field="precommit_digest",
        label="scene-predicate calibration precommit",
    )
    binding = raw.get("runtime_binding")
    if (
        not isinstance(binding, Mapping)
        or raw.get("authorization_digest") != authorization["authorization_digest"]
        or raw.get("source_identities") != _source_identities()
        or raw.get("role_reveal_serialized") is not False
        or raw.get("semantic_registry_protocol_digest")
        != authorization["semantic_registry_protocol_digest"]
        or raw.get("precommit_fsynced_before_any_visual_or_text_call") is not True
        or raw.get("launcher_fingerprint")
        != {
            "version": PINNED_CODEX_CLI_VERSION,
            "launcher_digest": authorization["runtime_policy"][
                "expected_launcher_sha256"
            ],
        }
        or any(raw.get(key) != item for key, item in _authority_data().items())
    ):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "execution precommit differs"
        )
    catalog_bytes = _durable._decode_bytes(
        raw["model_catalog_snapshot_base64"], "model catalog"
    )
    if catalog_bytes is None:
        raise ObjectBongardScenePredicateCalibrationCommandError(
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
    if runtime.binding != dict(binding):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "runtime binding differs on reload"
        )
    return runtime


def _frontend_runtime_kwargs(runtime: ObjectBongardTurnRuntime) -> dict[str, object]:
    return {
        "model": runtime.model,
        "reasoning_effort": runtime.reasoning_effort,
        "minutes": runtime.minutes,
        "verbose": runtime.verbose,
        "executable": runtime.executable,
        "cloud_policy_cache_snapshot": runtime.cloud_policy_cache_snapshot,
        "model_catalog_snapshot": runtime.model_catalog_snapshot,
        "expected_launcher_digest": runtime.expected_launcher_digest,
        "no_tools_attestation": runtime.no_tools_attestation,
    }


def _journal_runtime_kwargs(runtime: ObjectBongardTurnRuntime) -> dict[str, object]:
    return {
        "model": runtime.model,
        "reasoning_effort": runtime.reasoning_effort,
        "minutes": runtime.minutes,
        "verbose": runtime.verbose,
        "executable": runtime.executable,
        "cloud_policy_cache_snapshot": runtime.cloud_policy_cache_snapshot,
        "model_catalog_snapshot": runtime.model_catalog_snapshot,
        "tool_surface_attestation": runtime.no_tools_attestation,
        "expected_launcher_digest": runtime.expected_launcher_digest,
        "expected_tool_surface_attestation_digest": (
            runtime.no_tools_attestation.attestation_digest
        ),
    }


def _observation_context_digest(
    *,
    authorization_digest: str,
    precommit_digest: str,
    neutral_panel_digest: str,
    stage: str,
) -> str:
    if stage not in ("discovery", "registered_a", "registered_b"):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "visual observation stage differs"
        )
    return "sha256:" + canonical_digest(
        {
            "schema": "gkm.bongard-scene-predicate-observation-context.v1",
            "authorization_digest": authorization_digest,
            "execution_precommit_digest": precommit_digest,
            "neutral_panel_digest": neutral_panel_digest,
            "stage": stage,
            "historical_role_visible": False,
            "candidate_or_formula_visible": False,
        }
    )


def _artifact_digest(value: object) -> str:
    digest = getattr(value, "artifact_digest", None)
    return _raw_digest(digest, "transcript artifact digest")


def _transcript_from_artifact(value: object) -> object:
    transcript = getattr(value, "transcript", None)
    if transcript is None or not hasattr(transcript, "to_data"):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "transcript artifact lacks a typed transcript"
        )
    return transcript


def _visual_batch_record(
    *,
    stage: str,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    inputs: _CalibrationInputs,
    artifacts: Sequence[object],
    journal_directories: Sequence[str],
) -> dict[str, Any]:
    values = tuple(artifacts)
    directories = tuple(journal_directories)
    if (
        stage not in ("discovery", "registered_a", "registered_b")
        or len(values) != PANEL_COUNT
        or len(directories) != PANEL_COUNT
        or len({_artifact_digest(item) for item in values}) != PANEL_COUNT
    ):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "blind visual batch inventory differs"
        )
    rows = [
        {
            "blind_panel_id": panel.blind_panel_id,
            "neutral_panel_digest": panel.neutral_panel_digest,
            "proposal_inventory_digest": inventory.inventory_digest,
            "observation_context_digest": _observation_context_digest(
                authorization_digest=authorization["authorization_digest"],
                precommit_digest=precommit["precommit_digest"],
                neutral_panel_digest=panel.neutral_panel_digest,
                stage=stage,
            ),
            "artifact": artifact.to_data(),
            "artifact_digest": _artifact_digest(artifact),
            "journal_directory": directory,
        }
        for panel, inventory, artifact, directory in zip(
            inputs.panels,
            inputs.inventories,
            values,
            directories,
            strict=True,
        )
    ]
    _assert_role_blind(rows, f"{stage} batch")
    return _durable._record(
        {
            "schema": (
                DISCOVERY_BATCH_SCHEMA
                if stage == "discovery"
                else EVALUATION_BATCH_SCHEMA
            ),
            "command_id": COMMAND_ID,
            "stage": stage,
            "authorization_digest": authorization["authorization_digest"],
            "execution_precommit_digest": precommit["precommit_digest"],
            "historical_source_digest": inputs.source.source_digest,
            "role_commitment_digest": inputs.role_commitment_digest,
            "role_reveal_serialized": False,
            "rows": rows,
            "fresh_visual_call_count": PANEL_COUNT,
            "reused_visual_call_count": 0,
            **_authority_data(),
        },
        "batch_digest",
    )


def _freeze_record(
    *,
    schema: str,
    phase: str,
    batch_digests: Sequence[str],
    artifact_digests: Sequence[str],
    parent_digest: str,
    digest_field: str,
    historical_roles_revealed: bool = False,
) -> dict[str, Any]:
    for digest in (*batch_digests, *artifact_digests):
        if not isinstance(digest, str) or (
            _RAW_DIGEST.fullmatch(digest) is None
            and _ADDRESS.fullmatch(digest) is None
        ):
            raise ObjectBongardScenePredicateCalibrationCommandError(
                "freeze dependency digest differs"
            )
    return _durable._record(
        {
            "schema": schema,
            "command_id": COMMAND_ID,
            "phase": phase,
            "parent_digest": parent_digest,
            "batch_digests": list(batch_digests),
            "artifact_digests": list(artifact_digests),
            "exact_canonical_bytes_fsynced_and_reloaded": True,
            "historical_roles_revealed": historical_roles_revealed,
            "selection_or_formula_built": False,
            **_authority_data(),
        },
        digest_field,
    )


def _role_reveal_record(
    inputs: _CalibrationInputs,
    discovery_freeze: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [dict(item) for item in inputs.role_reveal_rows]
    if _role_commitment(rows) != inputs.role_commitment_digest:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "historical role reveal breaks its precommitment"
        )
    return _durable._record(
        {
            "schema": ROLE_REVEAL_SCHEMA,
            "command_id": COMMAND_ID,
            "discovery_freeze_digest": discovery_freeze[
                "freeze_digest"
            ],
            "role_commitment_digest": inputs.role_commitment_digest,
            "rows": rows,
            "revealed_after_blind_discovery_freeze_before_semantic_synthesis": True,
            "semantic_registry_proposer_calls_after_reveal": 1,
            "registered_visual_calls_after_reveal": 24,
            **_authority_data(),
        },
        "role_reveal_digest",
    )


def _semantic_proposal_input_record(
    *,
    prepared: object,
    discovery_freeze: Mapping[str, Any],
    role_reveal: Mapping[str, Any],
) -> dict[str, Any]:
    from bongard.object_scene_semantic_registry import (
        object_scene_semantic_registry_protocol_digest,
    )

    return _durable._record(
        {
            "schema": SEMANTIC_PROPOSAL_INPUT_SCHEMA,
            "command_id": COMMAND_ID,
            "discovery_freeze_digest": discovery_freeze["freeze_digest"],
            "role_reveal_digest": role_reveal["role_reveal_digest"],
            "role_commitment_digest": role_reveal["role_commitment_digest"],
            "semantic_registry_protocol_digest": (
                object_scene_semantic_registry_protocol_digest()
            ),
            "prepared_input": prepared.to_data(),
            "preparation_digest": prepared.preparation_digest,
            "prepared_input_fsynced_before_semantic_proposer_call": True,
            **_authority_data(),
        },
        "semantic_proposal_input_digest",
    )


def _restore_semantic_proposal_input(
    record: Mapping[str, Any],
    *,
    discovery_freeze: Mapping[str, Any],
    role_reveal: Mapping[str, Any],
    discovery_artifacts: Sequence[object],
    role_rows: Sequence[Mapping[str, object]],
) -> object:
    from bongard.object_scene_semantic_registry import (
        ObjectScenePreparedSemanticRegistryProposal,
        prepare_object_scene_semantic_registry_proposal,
    )

    raw = _durable._validate_record(
        record,
        schema=SEMANTIC_PROPOSAL_INPUT_SCHEMA,
        digest_field="semantic_proposal_input_digest",
        label="scene semantic registry proposal input",
    )
    prepared = ObjectScenePreparedSemanticRegistryProposal.from_data(
        raw.get("prepared_input")
    )
    expected_prepared = prepare_object_scene_semantic_registry_proposal(
        discovery_artifacts, role_rows
    )
    expected = _semantic_proposal_input_record(
        prepared=expected_prepared,
        discovery_freeze=discovery_freeze,
        role_reveal=role_reveal,
    )
    if prepared != expected_prepared or raw != expected:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "semantic proposal input differs on replay"
        )
    return prepared


def _semantic_proposal_result_record(
    *,
    semantic_proposal_input: Mapping[str, Any],
    proposal: object,
    registry: object,
    payload: Mapping[str, Any],
    receipt: object,
    journal_directory: str,
    journal_summary_digest: str,
) -> dict[str, Any]:
    from bongard.object_scene_semantic_registry import (
        ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE,
    )

    status = getattr(proposal, "status", None)
    proposal_data = proposal.to_data() if hasattr(proposal, "to_data") else {}
    derivation_mode = proposal_data.get("derivation_mode")
    if (
        status not in ("proposed", "typed_proposal_gap")
        or derivation_mode != ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
        or getattr(proposal, "preparation_digest", None)
        != semantic_proposal_input["preparation_digest"]
        or getattr(proposal, "registry_digest", None)
        != getattr(registry, "registry_digest", None)
    ):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "semantic proposal status differs"
        )
    return _durable._record(
        {
            "schema": SEMANTIC_PROPOSAL_RESULT_SCHEMA,
            "command_id": COMMAND_ID,
            "semantic_proposal_input_digest": semantic_proposal_input[
                "semantic_proposal_input_digest"
            ],
            "semantic_proposal": proposal.to_data(),
            "semantic_proposal_digest": proposal.proposal_digest,
            "semantic_proposal_status": status,
            "semantic_proposal_valid": status == "proposed",
            "registry_derivation_mode": derivation_mode,
            "semantic_registry": registry.to_data(),
            "semantic_registry_digest": registry.registry_digest,
            "proposer_payload": _canonical_mapping(payload, "semantic proposer payload"),
            "proposer_receipt": receipt.to_dict(),
            "proposer_receipt_digest": receipt.receipt_digest,
            "proposer_journal_directory": journal_directory,
            "proposer_journal_summary_digest": journal_summary_digest,
            "proposer_fresh_call_count": 1,
            "proposer_reused_call_count": 0,
            **_authority_data(),
        },
        "semantic_proposal_result_digest",
    )


def _execute_semantic_proposal(
    root: Path,
    *,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    runtime: ObjectBongardTurnRuntime,
    semantic_proposal_input: Mapping[str, Any],
    prepared: object,
    text_transport: TextTransport,
) -> tuple[object, object, dict[str, Any]]:
    from bongard.object_scene_semantic_registry import (
        ObjectSceneSemanticRegistryPayloadError,
        build_object_scene_semantic_registry_gap,
        build_object_scene_semantic_registry_proposal,
    )

    relative = Path(JOURNAL_DIRECTORY) / "semantic_registry_proposer"
    journal = ObjectBongardTextTurnJournalTransport(
        root / relative,
        authorization_digest=authorization["authorization_digest"],
        execution_precommit_digest=precommit["precommit_digest"],
        task_id="bd_scene_calibration_semantic_registry_proposer",
        turn_kind="semantic_registry_proposal",
        expected_prompt=prepared.prompt,
        expected_output_schema=prepared.output_schema,
        runtime=runtime,
        underlying_transport=text_transport,
    )
    result = journal(
        prepared.prompt,
        prepared.output_schema,
        **_journal_runtime_kwargs(runtime),
    )
    payload = _canonical_mapping(result.payload, "semantic proposer payload")
    try:
        proposal, registry = build_object_scene_semantic_registry_proposal(
            prepared, payload
        )
    except ObjectSceneSemanticRegistryPayloadError:
        usable_by_role = {
            role: sum(
                item["usable"] is True and item["historical_role"] == role
                for item in prepared.alias_bindings
            )
            for role in (0, 1)
        }
        gap_code = (
            "insufficient_discovery_evidence"
            if any(count < 2 for count in usable_by_role.values())
            else "payload_rejected"
        )
        proposal, registry = build_object_scene_semantic_registry_gap(
            prepared, gap_code, payload
        )
    if journal.fresh_call_count != 1 or journal.reused_call_count != 0:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "semantic proposer journal did not make exactly one fresh call"
        )
    summary = verify_object_bongard_turn_journal(journal)
    record = _semantic_proposal_result_record(
        semantic_proposal_input=semantic_proposal_input,
        proposal=proposal,
        registry=registry,
        payload=payload,
        receipt=result.receipt,
        journal_directory=str(relative),
        journal_summary_digest=summary.record_digest,
    )
    return proposal, registry, record


def _restore_semantic_proposal_result(
    record: Mapping[str, Any],
    *,
    semantic_proposal_input: Mapping[str, Any],
    prepared: object,
    discovery_artifacts: Sequence[object],
    role_rows: Sequence[Mapping[str, object]],
) -> tuple[object, object]:
    from bongard.object_bongard_turn_journal import _receipt_from_data
    from bongard.object_scene_semantic_registry import (
        ObjectSceneSemanticRegistryPayloadError,
        ObjectSceneSemanticRegistryProposal,
        build_object_scene_semantic_registry_gap,
        build_object_scene_semantic_registry_proposal,
        verify_object_scene_semantic_registry_proposal,
    )
    from bongard.object_scene_visual_frontend import ObjectSceneSoftTagRegistry

    raw = _durable._validate_record(
        record,
        schema=SEMANTIC_PROPOSAL_RESULT_SCHEMA,
        digest_field="semantic_proposal_result_digest",
        label="scene semantic registry proposal result",
    )
    persisted_proposal = ObjectSceneSemanticRegistryProposal.from_data(
        raw.get("semantic_proposal")
    )
    persisted_registry = ObjectSceneSoftTagRegistry.from_data(
        raw.get("semantic_registry")
    )
    payload = _canonical_mapping(raw.get("proposer_payload"), "semantic proposer payload")
    if raw.get("semantic_proposal_status") == "proposed":
        proposal, registry = build_object_scene_semantic_registry_proposal(
            prepared, payload
        )
    elif raw.get("semantic_proposal_status") == "typed_proposal_gap":
        try:
            build_object_scene_semantic_registry_proposal(prepared, payload)
        except ObjectSceneSemanticRegistryPayloadError:
            pass
        else:
            raise ObjectBongardScenePredicateCalibrationCommandError(
                "semantic proposal gap payload is valid"
            )
        proposal, registry = build_object_scene_semantic_registry_gap(
            prepared, persisted_proposal.gap_code, payload
        )
    else:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "semantic proposal result status differs"
        )
    receipt = _receipt_from_data(raw.get("proposer_receipt"))
    expected = _semantic_proposal_result_record(
        semantic_proposal_input=semantic_proposal_input,
        proposal=proposal,
        registry=registry,
        payload=payload,
        receipt=receipt,
        journal_directory=raw.get("proposer_journal_directory"),
        journal_summary_digest=raw.get("proposer_journal_summary_digest"),
    )
    if (
        proposal != persisted_proposal
        or registry != persisted_registry
        or raw != expected
    ):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "semantic proposal result differs on reconstruction"
        )
    verify_object_scene_semantic_registry_proposal(
        proposal, registry, discovery_artifacts, role_rows
    )
    return proposal, registry


def _cold_replay_semantic_proposal(
    root: Path,
    *,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    runtime: ObjectBongardTurnRuntime,
    semantic_proposal_input: Mapping[str, Any],
    semantic_proposal_result: Mapping[str, Any],
    prepared: object,
    discovery_artifacts: Sequence[object],
    role_rows: Sequence[Mapping[str, object]],
) -> tuple[object, object, str]:
    relative = Path(JOURNAL_DIRECTORY) / "semantic_registry_proposer"
    journal = ObjectBongardTextTurnJournalTransport(
        root / relative,
        authorization_digest=authorization["authorization_digest"],
        execution_precommit_digest=precommit["precommit_digest"],
        task_id="bd_scene_calibration_semantic_registry_proposer",
        turn_kind="semantic_registry_proposal",
        expected_prompt=prepared.prompt,
        expected_output_schema=prepared.output_schema,
        runtime=runtime,
        underlying_transport=_forbidden_text_transport,
    )
    replayed = journal(
        prepared.prompt,
        prepared.output_schema,
        **_journal_runtime_kwargs(runtime),
    )
    summary = verify_object_bongard_turn_journal(journal)
    proposal, registry = _restore_semantic_proposal_result(
        semantic_proposal_result,
        semantic_proposal_input=semantic_proposal_input,
        prepared=prepared,
        discovery_artifacts=discovery_artifacts,
        role_rows=role_rows,
    )
    if (
        _canonical_mapping(replayed.payload, "replayed semantic proposer payload")
        != semantic_proposal_result["proposer_payload"]
        or replayed.receipt.to_dict()
        != semantic_proposal_result["proposer_receipt"]
        or summary.record_digest
        != semantic_proposal_result["proposer_journal_summary_digest"]
        or journal.fresh_call_count != 0
        or journal.reused_call_count != 1
    ):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "semantic proposer cold replay differs"
        )
    return proposal, registry, summary.record_digest


def _ranker_output_schema(survivor_digests: Sequence[str]) -> dict[str, object]:
    digests = tuple(survivor_digests)
    if (
        not 1 <= len(digests) <= 64
        or len(set(digests)) != len(digests)
        or any(_RAW_DIGEST.fullmatch(item) is None for item in digests)
    ):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "ranker survivor slate differs"
        )
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "selected_survivor_digest": {
                "type": "string",
                "enum": list(digests),
            }
        },
        "required": ["selected_survivor_digest"],
        "additionalProperties": False,
    }
    validate_codex_strict_output_schema(schema)
    return schema


def _forbidden_named_transport(*_args: object, **_kwargs: object) -> CodexStructuredResult:
    raise AssertionError("cold replay attempted a visual model call")


def _forbidden_text_transport(*_args: object, **_kwargs: object) -> CodexStructuredResult:
    raise AssertionError("cold replay attempted a text model call")


def _execute_visual_batch(
    root: Path,
    *,
    stage: str,
    inputs: _CalibrationInputs,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    runtime: ObjectBongardTurnRuntime,
    registry: object | None,
    parallel_workers: int,
    transport: NamedImageTransport,
) -> tuple[tuple[object, ...], dict[str, Any]]:
    from bongard.object_scene_visual_frontend import (
        ObjectSceneTranscriptMode,
        observe_object_scene_transcript,
        prepare_object_scene_transcript_inputs,
    )

    if stage == "discovery":
        mode = ObjectSceneTranscriptMode.DISCOVERY
        if registry is not None:
            raise ObjectBongardScenePredicateCalibrationCommandError(
                "discovery cannot receive a registry"
            )
    elif stage in ("registered_a", "registered_b"):
        mode = ObjectSceneTranscriptMode.REGISTERED_EVALUATION
        if registry is None:
            raise ObjectBongardScenePredicateCalibrationCommandError(
                "registered evaluation requires the frozen registry"
            )
    else:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "visual batch stage differs"
        )

    def run_one(index: int) -> tuple[object, str]:
        panel = inputs.panels[index]
        inventory = inputs.inventories[index]
        prepared = prepare_object_scene_transcript_inputs(
            panel.exact_png_bytes, inventory, mode, registry
        )
        relative = Path(JOURNAL_DIRECTORY) / stage / f"panel_{index:02d}"
        journal = ObjectBongardNamedImageTurnJournalTransport(
            root / relative,
            authorization_digest=authorization["authorization_digest"],
            execution_precommit_digest=precommit["precommit_digest"],
            task_id=panel.journal_task_id,
            turn_kind=stage,
            expected_prompt=prepared.prompt,
            expected_images=prepared.presentation,
            expected_output_schema=prepared.output_schema,
            runtime=runtime,
            underlying_transport=transport,
        )
        artifact = observe_object_scene_transcript(
            panel.exact_png_bytes,
            scene_id=panel.blind_panel_id,
            observation_context_digest=_observation_context_digest(
                authorization_digest=authorization["authorization_digest"],
                precommit_digest=precommit["precommit_digest"],
                neutral_panel_digest=panel.neutral_panel_digest,
                stage=stage,
            ),
            mode=mode,
            registry=registry,
            inventory=inventory,
            expected_panel_sha256=panel.png_sha256,
            **_frontend_runtime_kwargs(runtime),
            transport=journal,
        )
        if journal.fresh_call_count != 1 or journal.reused_call_count != 0:
            raise ObjectBongardScenePredicateCalibrationCommandError(
                "visual journal did not make exactly one fresh call"
            )
        return artifact, str(relative)

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        outcomes = tuple(executor.map(run_one, range(PANEL_COUNT)))
    artifacts = tuple(item[0] for item in outcomes)
    directories = tuple(item[1] for item in outcomes)
    batch = _visual_batch_record(
        stage=stage,
        authorization=authorization,
        precommit=precommit,
        inputs=inputs,
        artifacts=artifacts,
        journal_directories=directories,
    )
    return artifacts, batch


def _restore_visual_batch(
    batch: Mapping[str, Any],
    *,
    stage: str,
    inputs: _CalibrationInputs,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
) -> tuple[object, ...]:
    from bongard.object_scene_visual_frontend import (
        ObjectSceneTranscriptArtifact,
        ObjectSceneTranscriptMode,
    )

    schema = DISCOVERY_BATCH_SCHEMA if stage == "discovery" else EVALUATION_BATCH_SCHEMA
    raw = _durable._validate_record(
        batch,
        schema=schema,
        digest_field="batch_digest",
        label=f"scene-predicate {stage} batch",
    )
    rows = raw.get("rows")
    if (
        raw.get("stage") != stage
        or raw.get("authorization_digest") != authorization["authorization_digest"]
        or raw.get("execution_precommit_digest") != precommit["precommit_digest"]
        or raw.get("historical_source_digest") != inputs.source.source_digest
        or raw.get("role_commitment_digest") != inputs.role_commitment_digest
        or raw.get("role_reveal_serialized") is not False
        or raw.get("fresh_visual_call_count") != PANEL_COUNT
        or raw.get("reused_visual_call_count") != 0
        or not isinstance(rows, list)
        or len(rows) != PANEL_COUNT
        or any(raw.get(key) != item for key, item in _authority_data().items())
    ):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            f"{stage} batch policy differs"
        )
    artifacts: list[object] = []
    expected_mode = (
        ObjectSceneTranscriptMode.DISCOVERY
        if stage == "discovery"
        else ObjectSceneTranscriptMode.REGISTERED_EVALUATION
    )
    for panel, inventory, row in zip(
        inputs.panels, inputs.inventories, rows, strict=True
    ):
        if not isinstance(row, Mapping):
            raise ObjectBongardScenePredicateCalibrationCommandError(
                "visual batch row differs"
            )
        artifact = ObjectSceneTranscriptArtifact.from_data(
            row.get("artifact"), expected_artifact_digest=row.get("artifact_digest")
        )
        context = _observation_context_digest(
            authorization_digest=authorization["authorization_digest"],
            precommit_digest=precommit["precommit_digest"],
            neutral_panel_digest=panel.neutral_panel_digest,
            stage=stage,
        )
        if (
            row.get("blind_panel_id") != panel.blind_panel_id
            or row.get("neutral_panel_digest") != panel.neutral_panel_digest
            or row.get("proposal_inventory_digest") != inventory.inventory_digest
            or row.get("observation_context_digest") != context
            or artifact.scene_id != panel.blind_panel_id
            or artifact.observation_context_digest != context
            or artifact.inventory != inventory
            or artifact.mode is not expected_mode
            or row.get("artifact_digest") != artifact.artifact_digest
            or row.get("journal_directory")
            != str(
                Path(JOURNAL_DIRECTORY)
                / stage
                / f"panel_{len(artifacts):02d}"
            )
        ):
            raise ObjectBongardScenePredicateCalibrationCommandError(
                "visual batch artifact binding differs"
            )
        artifacts.append(artifact)
    expected = _visual_batch_record(
        stage=stage,
        authorization=authorization,
        precommit=precommit,
        inputs=inputs,
        artifacts=artifacts,
        journal_directories=[row["journal_directory"] for row in rows],
    )
    if raw != expected:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            f"{stage} batch differs on reconstruction"
        )
    return tuple(artifacts)


def _registry_freeze_record(
    *,
    discovery_freeze: Mapping[str, Any],
    role_reveal: Mapping[str, Any],
    semantic_proposal_result: Mapping[str, Any],
    semantic_registry_proposal: object,
    registry: object,
) -> tuple[object, dict[str, Any]]:
    from bongard.object_scene_semantic_registry import (
        ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE,
        object_scene_semantic_registry_protocol_digest,
        object_scene_semantic_registry_source_digest,
    )
    if (
        semantic_proposal_result.get("registry_derivation_mode")
        != ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
        or semantic_registry_proposal.to_data().get("derivation_mode")
        != ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
    ):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "registry freeze is not role-aware semantic synthesis"
        )
    record = _durable._record(
        {
            "schema": REGISTRY_FREEZE_SCHEMA,
            "command_id": COMMAND_ID,
            "discovery_freeze_digest": discovery_freeze["freeze_digest"],
            "role_reveal_digest": role_reveal["role_reveal_digest"],
            "semantic_proposal_result_digest": semantic_proposal_result[
                "semantic_proposal_result_digest"
            ],
            "semantic_proposal_digest": semantic_registry_proposal.proposal_digest,
            "semantic_proposal_status": semantic_registry_proposal.status,
            "registry_derivation_mode": (
                ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
            ),
            "semantic_registry_source_digest": (
                object_scene_semantic_registry_source_digest()
            ),
            "semantic_registry_protocol_digest": (
                object_scene_semantic_registry_protocol_digest()
            ),
            "registry": registry.to_data(),
            "registry_digest": registry.registry_digest,
            "registry_built_from_revealed_roles_and_frozen_discovery": True,
            "benchmark_acceptance_authorized_registry": (
                semantic_registry_proposal.status == "proposed"
            ),
            "exact_frequency_fallback_acceptance_authorized": False,
            "exact_normalized_affirmative_scoped_concepts_only": True,
            "typed_proposal_gap_has_zero_registered_tags": (
                semantic_registry_proposal.status != "typed_proposal_gap"
                or not registry.tags
            ),
            "registry_record_fsynced_and_reloaded_before_pass_a": True,
            **_authority_data(),
        },
        "registry_freeze_digest",
    )
    return registry, record


def _restore_registry_freeze(
    record: Mapping[str, Any],
    *,
    discovery_freeze: Mapping[str, Any],
    role_reveal: Mapping[str, Any],
    semantic_proposal_result: Mapping[str, Any],
    semantic_registry_proposal: object,
    semantic_registry: object,
) -> object:
    from bongard.object_scene_visual_frontend import (
        ObjectSceneSoftTagRegistry,
    )

    raw = _durable._validate_record(
        record,
        schema=REGISTRY_FREEZE_SCHEMA,
        digest_field="registry_freeze_digest",
        label="scene-predicate soft-tag registry freeze",
    )
    registry = ObjectSceneSoftTagRegistry.from_data(raw.get("registry"))
    expected_registry, expected = _registry_freeze_record(
        discovery_freeze=discovery_freeze,
        role_reveal=role_reveal,
        semantic_proposal_result=semantic_proposal_result,
        semantic_registry_proposal=semantic_registry_proposal,
        registry=semantic_registry,
    )
    if registry != expected_registry or raw != expected:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "soft-tag registry freeze differs on replay"
        )
    return registry


def _cold_replay_visual_batch(
    root: Path,
    *,
    stage: str,
    inputs: _CalibrationInputs,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    runtime: ObjectBongardTurnRuntime,
    registry: object | None,
    batch: Mapping[str, Any],
) -> tuple[tuple[object, ...], tuple[str, ...]]:
    from bongard.object_scene_visual_frontend import (
        ObjectSceneTranscriptMode,
        observe_object_scene_transcript,
        prepare_object_scene_transcript_inputs,
        verify_object_scene_transcript_artifact,
    )

    artifacts = _restore_visual_batch(
        batch,
        stage=stage,
        inputs=inputs,
        authorization=authorization,
        precommit=precommit,
    )
    mode = (
        ObjectSceneTranscriptMode.DISCOVERY
        if stage == "discovery"
        else ObjectSceneTranscriptMode.REGISTERED_EVALUATION
    )
    summaries: list[str] = []
    for index, (panel, inventory, artifact) in enumerate(
        zip(inputs.panels, inputs.inventories, artifacts, strict=True)
    ):
        context = _observation_context_digest(
            authorization_digest=authorization["authorization_digest"],
            precommit_digest=precommit["precommit_digest"],
            neutral_panel_digest=panel.neutral_panel_digest,
            stage=stage,
        )
        verify_object_scene_transcript_artifact(
            artifact,
            panel.exact_png_bytes,
            expected_scene_id=panel.blind_panel_id,
            expected_observation_context_digest=context,
            expected_panel_sha256=panel.png_sha256,
            expected_artifact_digest=artifact.artifact_digest,
        )
        prepared = prepare_object_scene_transcript_inputs(
            panel.exact_png_bytes, inventory, mode, registry
        )
        relative = Path(JOURNAL_DIRECTORY) / stage / f"panel_{index:02d}"
        journal = ObjectBongardNamedImageTurnJournalTransport(
            root / relative,
            authorization_digest=authorization["authorization_digest"],
            execution_precommit_digest=precommit["precommit_digest"],
            task_id=panel.journal_task_id,
            turn_kind=stage,
            expected_prompt=prepared.prompt,
            expected_images=prepared.presentation,
            expected_output_schema=prepared.output_schema,
            runtime=runtime,
            underlying_transport=_forbidden_named_transport,
        )
        replayed = observe_object_scene_transcript(
            panel.exact_png_bytes,
            scene_id=panel.blind_panel_id,
            observation_context_digest=context,
            mode=mode,
            registry=registry,
            inventory=inventory,
            expected_panel_sha256=panel.png_sha256,
            **_frontend_runtime_kwargs(runtime),
            transport=journal,
        )
        if (
            replayed != artifact
            or journal.fresh_call_count != 0
            or journal.reused_call_count != 1
        ):
            raise ObjectBongardScenePredicateCalibrationCommandError(
                f"{stage} journal cold replay differs"
            )
        summary = verify_object_bongard_turn_journal(journal)
        if summary.terminal_status not in ("success", "failure"):
            raise ObjectBongardScenePredicateCalibrationCommandError(
                "visual journal is not terminal"
            )
        summaries.append(summary.record_digest)
    return artifacts, tuple(summaries)


def _canonical_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            f"{label} must be a JSON object"
        )
    try:
        raw = json.loads(canonical_json(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            f"{label} is not canonical finite JSON"
        ) from exc
    if not isinstance(raw, dict):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            f"{label} must be a JSON object"
        )
    return raw


def _digest_free_ranker_value(value: object) -> object:
    """Project bound IR metadata into a readable, lineage-free ranker view."""

    if isinstance(value, Mapping):
        return {
            key: _digest_free_ranker_value(item)
            for key, item in value.items()
            if key == "candidate_digest" or not key.endswith("_digest")
        }
    if isinstance(value, (list, tuple)):
        return [_digest_free_ranker_value(item) for item in value]
    return value


def _digest_free_ranker_row(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _canonical_mapping(value, "ranker slate row")
    projected = _digest_free_ranker_value(row)
    if not isinstance(projected, dict):  # pragma: no cover - structural guard
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "ranker slate projection differs"
        )
    candidate_digest = projected.get("candidate_digest")
    if (
        not isinstance(candidate_digest, str)
        or _RAW_DIGEST.fullmatch(candidate_digest) is None
    ):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "ranker slate projection lost candidate identity"
        )
    return projected


def _ranker_prompt(slate_rows: Sequence[Mapping[str, Any]]) -> str:
    rows = tuple(_canonical_mapping(item, "ranker slate row") for item in slate_rows)
    digests = tuple(item.get("candidate_digest") for item in rows)
    _ranker_output_schema(digests)  # type: ignore[arg-type]
    return (
        "Choose exactly one already-frozen survivor. You may compare only the "
        "listed canonical formulas, orientations, complexities, and merged "
        "four-disposition support summaries. Every listed row has already "
        "passed the frozen coverage, selectivity, and A/B repeatability gates. "
        "Prefer the "
        "simplest visibly coherent formula that retains the strongest complete "
        "repeat-tested separation. Do not create, rewrite, negate, combine, or "
        "repair a formula. Return only the selected_survivor_digest field.\n\n"
        "Frozen survivor slate:\n"
        + canonical_json(list(rows)).decode("utf-8")
    )


def _assert_ranker_privacy(
    prompt: str,
    *,
    inputs: _CalibrationInputs,
    hidden_digests: Sequence[str],
) -> None:
    hidden = {
        inputs.source.source_digest,
        *hidden_digests,
        *(item.task_id for item in inputs.panels),
        *(item.panel_id for item in inputs.panels),
        *(item.png_sha256 for item in inputs.panels),
        *(item.inventory_digest for item in inputs.inventories),
    }
    if any(item and item in prompt for item in hidden):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "ranker prompt leaks historical identity or transcript lineage"
        )
    if len(prompt.encode("utf-8")) > 256_000:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "ranker prompt exceeds bounded slate envelope"
        )


def _rank_input_freeze_record(
    *,
    assessment: Mapping[str, Any],
    semantic_proposal_result: Mapping[str, Any],
    complete_survivor_digests: Sequence[str],
    slate_rows: Sequence[Mapping[str, Any]],
    omitted_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from bongard.object_scene_semantic_registry import (
        ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE,
    )

    complete = tuple(complete_survivor_digests)
    slate = tuple(_digest_free_ranker_row(item) for item in slate_rows)
    omitted = tuple(
        _canonical_mapping(item, "ranker omitted survivor row")
        for item in omitted_rows
    )
    semantic_eligible = (
        semantic_proposal_result.get("semantic_proposal_valid") is True
        and assessment.get("registry_derivation_mode")
        == ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
    )
    gap_status = assessment.get("typed_gap_status")
    if gap_status is not None and gap_status not in TYPED_CALIBRATION_GAP_STATUSES:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "rank input typed gap status differs"
        )
    if not semantic_eligible:
        slate = ()
        omitted = tuple(
            {
                "candidate_digest": digest,
                "reason": "mandatory_semantic_proposal_gap",
            }
            for digest in complete
        )
    slate_digests = tuple(item.get("candidate_digest") for item in slate)
    omitted_digests = tuple(item.get("candidate_digest") for item in omitted)
    if (
        len(set(complete)) != len(complete)
        or any(not isinstance(item, str) or _RAW_DIGEST.fullmatch(item) is None for item in complete)
        or any(not isinstance(item, str) or _RAW_DIGEST.fullmatch(item) is None for item in slate_digests)
        or any(not isinstance(item, str) or _RAW_DIGEST.fullmatch(item) is None for item in omitted_digests)
        or set(slate_digests) & set(omitted_digests)
        or set(slate_digests) | set(omitted_digests) != set(complete)
        or len(slate) > 64
        or any(item.get("reason") is None for item in omitted)
        or (
            (bool(complete) and semantic_eligible)
            == (gap_status is not None)
        )
        or (
            not semantic_eligible
            and gap_status != TYPED_SEMANTIC_PROPOSAL_GAP
        )
    ):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "ranker slate does not account for the complete survivor space"
        )
    if complete and not slate and semantic_eligible:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "nonempty survivor space produced an empty ranker slate"
        )
    hidden_lineage_digests = sorted(
        set(assessment_lineage_digests(assessment).values()) - set(complete)
    )
    return _durable._record(
        {
            "schema": RANK_INPUT_FREEZE_SCHEMA,
            "command_id": COMMAND_ID,
            "assessment_digest": assessment["assessment_digest"],
            "semantic_proposal_result_digest": semantic_proposal_result[
                "semantic_proposal_result_digest"
            ],
            "semantic_proposal_valid": semantic_eligible,
            "registry_derivation_mode": (
                ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
            ),
            "benchmark_acceptance_authorized_registry": semantic_eligible,
            "typed_gap_status": gap_status,
            "complete_survivor_digests": list(complete),
            "complete_survivor_count": len(complete),
            "ranker_slate": list(slate),
            "ranker_slate_digests": list(slate_digests),
            "ranker_slate_count": len(slate),
            "omitted_survivors": list(omitted),
            "hidden_lineage_digests": hidden_lineage_digests,
            "slate_equals_complete_space": len(slate) == len(complete),
            "canonical_bounded_semantic_stratified_order": True,
            "ranker_input_fsynced_and_reloaded_before_ranker_call": True,
            **_authority_data(),
        },
        "rank_input_freeze_digest",
    )


def _rank_survivor_slate(
    root: Path,
    *,
    inputs: _CalibrationInputs,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    runtime: ObjectBongardTurnRuntime,
    rank_input: Mapping[str, Any],
    text_transport: TextTransport,
) -> dict[str, Any]:
    slate_rows = rank_input["ranker_slate"]
    slate_digests = tuple(rank_input["ranker_slate_digests"])
    if not slate_digests:
        gap_status = rank_input.get("typed_gap_status")
        if gap_status not in TYPED_CALIBRATION_GAP_STATUSES:
            raise ObjectBongardScenePredicateCalibrationCommandError(
                "empty ranker slate lacks an evidence-based typed gap"
            )
        return _durable._record(
            {
                "schema": RANK_RESULT_SCHEMA,
                "command_id": COMMAND_ID,
                "rank_input_freeze_digest": rank_input[
                    "rank_input_freeze_digest"
                ],
                "status": gap_status,
                "typed_gap_status": gap_status,
                "ranker_called": False,
                "ranker_fresh_call_count": 0,
                "ranker_reused_call_count": 0,
                "selected_survivor_digest": None,
                "ranker_payload": None,
                "ranker_journal_directory": None,
                **_authority_data(),
            },
            "rank_result_digest",
        )
    prompt = _ranker_prompt(slate_rows)
    hidden = tuple(
        item
        for path, item in assessment_lineage_digests(rank_input).items()
        if not path.endswith("candidate_digest")
        and item not in set(rank_input["complete_survivor_digests"])
    )
    _assert_ranker_privacy(prompt, inputs=inputs, hidden_digests=hidden)
    schema = _ranker_output_schema(slate_digests)
    relative = Path(JOURNAL_DIRECTORY) / "ranker"
    journal = ObjectBongardTextTurnJournalTransport(
        root / relative,
        authorization_digest=authorization["authorization_digest"],
        execution_precommit_digest=precommit["precommit_digest"],
        task_id="bd_scene_calibration_ranker",
        turn_kind="survivor_rank",
        expected_prompt=prompt,
        expected_output_schema=schema,
        runtime=runtime,
        underlying_transport=text_transport,
    )
    result = journal(prompt, schema, **_journal_runtime_kwargs(runtime))
    payload = _canonical_mapping(result.payload, "ranker payload")
    selected = payload.get("selected_survivor_digest")
    if set(payload) != {"selected_survivor_digest"} or selected not in slate_digests:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "ranker named something outside the frozen survivor slate"
        )
    if journal.fresh_call_count != 1 or journal.reused_call_count != 0:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "ranker journal did not make exactly one fresh call"
        )
    summary = verify_object_bongard_turn_journal(journal)
    return _durable._record(
        {
            "schema": RANK_RESULT_SCHEMA,
            "command_id": COMMAND_ID,
            "rank_input_freeze_digest": rank_input["rank_input_freeze_digest"],
            "status": "selected_frozen_survivor",
            "typed_gap_status": None,
            "ranker_called": True,
            "ranker_fresh_call_count": 1,
            "ranker_reused_call_count": 0,
            "selected_survivor_digest": selected,
            "ranker_payload": payload,
            "ranker_journal_directory": str(relative),
            "ranker_journal_summary_digest": summary.record_digest,
            **_authority_data(),
        },
        "rank_result_digest",
    )


def assessment_lineage_digests(value: object) -> dict[str, str]:
    """Collect hidden non-candidate digests for ranker prompt leak checks."""

    result: dict[str, str] = {}

    def visit(item: object, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        elif isinstance(item, str) and (
            _RAW_DIGEST.fullmatch(item) is not None
            or _ADDRESS.fullmatch(item) is not None
        ):
            result[path] = item

    visit(value, "")
    return result


def _cold_replay_ranker(
    root: Path,
    *,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    runtime: ObjectBongardTurnRuntime,
    rank_input: Mapping[str, Any],
    rank_result: Mapping[str, Any],
) -> str | None:
    raw_input = _durable._validate_record(
        rank_input,
        schema=RANK_INPUT_FREEZE_SCHEMA,
        digest_field="rank_input_freeze_digest",
        label="scene-predicate rank input freeze",
    )
    raw_result = _durable._validate_record(
        rank_result,
        schema=RANK_RESULT_SCHEMA,
        digest_field="rank_result_digest",
        label="scene-predicate rank result",
    )
    slate_digests = tuple(raw_input["ranker_slate_digests"])
    if not slate_digests:
        gap_status = raw_input.get("typed_gap_status")
        if (
            gap_status not in TYPED_CALIBRATION_GAP_STATUSES
            or raw_result.get("status") != gap_status
            or raw_result.get("typed_gap_status") != gap_status
            or raw_result.get("ranker_called") is not False
            or raw_result.get("ranker_fresh_call_count") != 0
            or raw_result.get("ranker_reused_call_count") != 0
            or raw_result.get("selected_survivor_digest") is not None
            or (root / JOURNAL_DIRECTORY / "ranker").exists()
        ):
            raise ObjectBongardScenePredicateCalibrationCommandError(
                "empty survivor gap attempted or forged a ranker call"
            )
        return None
    prompt = _ranker_prompt(raw_input["ranker_slate"])
    schema = _ranker_output_schema(slate_digests)
    relative = Path(JOURNAL_DIRECTORY) / "ranker"
    journal = ObjectBongardTextTurnJournalTransport(
        root / relative,
        authorization_digest=authorization["authorization_digest"],
        execution_precommit_digest=precommit["precommit_digest"],
        task_id="bd_scene_calibration_ranker",
        turn_kind="survivor_rank",
        expected_prompt=prompt,
        expected_output_schema=schema,
        runtime=runtime,
        underlying_transport=_forbidden_text_transport,
    )
    result = journal(prompt, schema, **_journal_runtime_kwargs(runtime))
    payload = _canonical_mapping(result.payload, "cold-replayed ranker payload")
    selected = payload.get("selected_survivor_digest")
    summary = verify_object_bongard_turn_journal(journal)
    if (
        selected not in slate_digests
        or raw_result.get("status") != "selected_frozen_survivor"
        or raw_result.get("typed_gap_status") is not None
        or raw_result.get("ranker_called") is not True
        or raw_result.get("ranker_fresh_call_count") != 1
        or raw_result.get("ranker_reused_call_count") != 0
        or raw_result.get("selected_survivor_digest") != selected
        or raw_result.get("ranker_payload") != payload
        or raw_result.get("ranker_journal_directory") != str(relative)
        or raw_result.get("ranker_journal_summary_digest") != summary.record_digest
        or journal.fresh_call_count != 0
        or journal.reused_call_count != 1
    ):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "ranker cold replay differs"
        )
    return selected


def _formula_freeze_record(
    *,
    assessment: Mapping[str, Any],
    semantic_proposal_result: Mapping[str, Any],
    rank_input: Mapping[str, Any],
    rank_result: Mapping[str, Any],
    candidate_by_digest: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    from bongard.object_scene_semantic_registry import (
        ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE,
    )

    selected = rank_result.get("selected_survivor_digest")
    role_aware = (
        assessment.get("registry_derivation_mode")
        == ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
        and semantic_proposal_result.get("registry_derivation_mode")
        == ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
    )
    semantic_eligible = (
        semantic_proposal_result.get("semantic_proposal_valid") is True
        and role_aware
    )
    gap_status = rank_input.get("typed_gap_status")
    if rank_result.get("typed_gap_status") != gap_status:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "rank result and formula freeze typed gaps differ"
        )
    if selected is None:
        if rank_result.get("status") != gap_status:
            raise ObjectBongardScenePredicateCalibrationCommandError(
                "rank result and formula freeze statuses differ"
            )
        if semantic_eligible and (
            candidate_by_digest or rank_input.get("complete_survivor_digests")
        ):
            raise ObjectBongardScenePredicateCalibrationCommandError(
                "typed gap disagrees with the complete survivor space"
            )
        candidate = None
        if gap_status not in TYPED_CALIBRATION_GAP_STATUSES:
            raise ObjectBongardScenePredicateCalibrationCommandError(
                "formula freeze lacks an evidence-based typed gap"
            )
        status = gap_status
    else:
        if rank_result.get("status") != "selected_frozen_survivor":
            raise ObjectBongardScenePredicateCalibrationCommandError(
                "rank result did not select a frozen survivor"
            )
        if not semantic_eligible or gap_status is not None:
            raise ObjectBongardScenePredicateCalibrationCommandError(
                "unauthorized registry or typed gap cannot select a survivor"
            )
        if selected not in candidate_by_digest:
            raise ObjectBongardScenePredicateCalibrationCommandError(
                "ranker selection is absent from the complete survivor space"
            )
        candidate = _canonical_mapping(
            candidate_by_digest[selected], "selected survivor"
        )
        if candidate.get("candidate_digest") != selected:
            raise ObjectBongardScenePredicateCalibrationCommandError(
                "selected survivor identity differs"
            )
        status = "accepted"
    return _durable._record(
        {
            "schema": FORMULA_FREEZE_SCHEMA,
            "command_id": COMMAND_ID,
            "assessment_digest": assessment["assessment_digest"],
            "semantic_proposal_result_digest": semantic_proposal_result[
                "semantic_proposal_result_digest"
            ],
            "rank_input_freeze_digest": rank_input[
                "rank_input_freeze_digest"
            ],
            "rank_result_digest": rank_result["rank_result_digest"],
            "status": status,
            "typed_gap_status": gap_status,
            "registry_derivation_mode": (
                ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
            ),
            "benchmark_acceptance_authorized_registry": semantic_eligible,
            "exact_frequency_fallback_acceptance_authorized": False,
            "selected_survivor_digest": selected,
            "selected_candidate": candidate,
            "complete_python_formula_and_evidence_frozen": selected is not None,
            "codex_output_cannot_modify_selected_candidate": True,
            "formula_identity_owned_by_python": True,
            **_authority_data(),
        },
        "formula_freeze_digest",
    )


def _validate_ir_bundle(value: object) -> dict[str, Any]:
    from bongard.object_scene_semantic_registry import (
        ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE,
    )

    expected = {
        "schema",
        "ir_source_digest",
        "algorithm_digest",
        "registry_digest",
        "registry_derivation_mode",
        "registry_derivation_digest",
        "coverage_gate",
        "selectivity_gate",
        "repeatability_gate",
        "version_space",
        "candidates",
        "complete_survivor_digests",
        "ranker_slate",
        "omitted_survivors",
        "bundle_digest",
    }
    raw = _canonical_mapping(value, "scene predicate IR bundle")
    if set(raw) != expected or raw.get("schema") != IR_BUNDLE_SCHEMA:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "scene predicate IR bundle fields differ"
        )
    _raw_digest(raw["ir_source_digest"], "scene predicate IR source digest")
    _raw_digest(raw["algorithm_digest"], "scene predicate IR algorithm digest")
    _raw_digest(raw["registry_digest"], "IR registry digest")
    _raw_digest(
        raw["registry_derivation_digest"], "IR registry derivation digest"
    )
    if raw["registry_derivation_mode"] != ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "IR registry derivation mode differs"
        )
    gates: list[dict[str, Any]] = []
    for name in ("coverage_gate", "selectivity_gate", "repeatability_gate"):
        gate = _canonical_mapping(raw[name], name)
        if type(gate.get("passed")) is not bool:
            raise ObjectBongardScenePredicateCalibrationCommandError(
                f"{name} lacks an explicit Boolean decision"
            )
        gates.append(gate)
    if (
        not isinstance(raw["candidates"], list)
        or not isinstance(raw["complete_survivor_digests"], list)
        or not isinstance(raw["ranker_slate"], list)
        or not isinstance(raw["omitted_survivors"], list)
        or not isinstance(raw["version_space"], Mapping)
    ):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "scene predicate IR arrays differ"
        )
    candidates = tuple(
        _canonical_mapping(item, "scene predicate candidate")
        for item in raw["candidates"]
    )
    candidate_digests = tuple(item.get("candidate_digest") for item in candidates)
    survivors = tuple(raw["complete_survivor_digests"])
    slate = tuple(
        _canonical_mapping(item, "scene predicate ranker view")
        for item in raw["ranker_slate"]
    )
    omitted = tuple(
        _canonical_mapping(item, "omitted scene predicate survivor")
        for item in raw["omitted_survivors"]
    )
    slate_digests = tuple(item.get("candidate_digest") for item in slate)
    omitted_digests = tuple(item.get("candidate_digest") for item in omitted)
    if (
        len(set(candidate_digests)) != len(candidate_digests)
        or any(not isinstance(item, str) or _RAW_DIGEST.fullmatch(item) is None for item in candidate_digests)
        or len(set(survivors)) != len(survivors)
        or any(not isinstance(item, str) or _RAW_DIGEST.fullmatch(item) is None for item in survivors)
        or not set(survivors).issubset(candidate_digests)
        or len(slate) > 64
        or set(slate_digests) & set(omitted_digests)
        or set(slate_digests) | set(omitted_digests) != set(survivors)
        or any(item.get("reason") is None for item in omitted)
        or bool(survivors) is not all(gate["passed"] for gate in gates)
    ):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "scene predicate survivor/gate accounting differs"
        )
    body = {key: item for key, item in raw.items() if key != "bundle_digest"}
    if raw["bundle_digest"] != canonical_digest(body):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "scene predicate IR bundle digest differs"
        )
    return raw


def _typed_calibration_gap_status(
    *,
    semantic_proposal_valid: bool,
    ir_bundle: Mapping[str, Any],
) -> str | None:
    """Name the first failed evidence gate without collapsing distinct holes."""

    if type(semantic_proposal_valid) is not bool:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "semantic proposal decision is not Boolean"
        )
    if not semantic_proposal_valid:
        return TYPED_SEMANTIC_PROPOSAL_GAP
    bundle = _validate_ir_bundle(ir_bundle)
    if bundle["complete_survivor_digests"]:
        return None
    if bundle["coverage_gate"]["passed"] is not True:
        return TYPED_LANGUAGE_GAP
    if bundle["selectivity_gate"]["passed"] is not True:
        return TYPED_SELECTIVITY_GAP
    if bundle["repeatability_gate"]["passed"] is not True:
        return TYPED_GROUNDING_REPEATABILITY_GAP
    raise ObjectBongardScenePredicateCalibrationCommandError(
        "empty survivor space has no failed evidence gate"
    )


def _derive_ir_bundle(
    *,
    registry: object,
    semantic_registry_proposal: object,
    discovery_artifacts: Sequence[object],
    registered_a_artifacts: Sequence[object],
    registered_b_artifacts: Sequence[object],
    role_rows: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    from bongard.object_bongard_scene_predicate_ir import (
        build_object_bongard_scene_predicate_calibration_bundle,
    )

    result = build_object_bongard_scene_predicate_calibration_bundle(
        registry,
        tuple(discovery_artifacts),
        tuple(registered_a_artifacts),
        tuple(registered_b_artifacts),
        tuple(dict(item) for item in role_rows),
        semantic_registry_proposal=semantic_registry_proposal,
    )
    value = result.to_data() if hasattr(result, "to_data") else result
    return _validate_ir_bundle(value)


def _assessment_record(
    *,
    inputs: _CalibrationInputs,
    evaluation_freeze: Mapping[str, Any],
    role_reveal: Mapping[str, Any],
    semantic_proposal_result: Mapping[str, Any],
    registry: object,
    ir_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    from bongard.object_scene_semantic_registry import (
        ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE,
    )

    bundle = _validate_ir_bundle(ir_bundle)
    semantic_valid = semantic_proposal_result["semantic_proposal_valid"]
    gap_status = _typed_calibration_gap_status(
        semantic_proposal_valid=semantic_valid,
        ir_bundle=bundle,
    )
    if (
        bundle["registry_digest"] != registry.registry_digest
        or bundle["registry_derivation_digest"]
        != semantic_proposal_result["semantic_proposal_digest"]
        or bundle["registry_derivation_mode"]
        != semantic_proposal_result["registry_derivation_mode"]
        or bundle["registry_derivation_mode"]
        != ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
        or role_reveal["role_commitment_digest"] != inputs.role_commitment_digest
    ):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "IR assessment parent binding differs"
        )
    return _durable._record(
        {
            "schema": ASSESSMENT_SCHEMA,
            "command_id": COMMAND_ID,
            "registered_evaluation_freeze_digest": evaluation_freeze[
                "freeze_digest"
            ],
            "role_reveal_digest": role_reveal["role_reveal_digest"],
            "semantic_proposal_result_digest": semantic_proposal_result[
                "semantic_proposal_result_digest"
            ],
            "semantic_proposal_digest": semantic_proposal_result[
                "semantic_proposal_digest"
            ],
            "semantic_proposal_valid": semantic_proposal_result[
                "semantic_proposal_valid"
            ],
            "registry_derivation_mode": bundle["registry_derivation_mode"],
            "role_aware_semantic_registry": True,
            "exact_frequency_fallback_acceptance_authorized": False,
            "historical_source_digest": inputs.source.source_digest,
            "registry_digest": registry.registry_digest,
            "ir_bundle": bundle,
            "ir_bundle_digest": bundle["bundle_digest"],
            "coverage_gate": bundle["coverage_gate"],
            "selectivity_gate": bundle["selectivity_gate"],
            "repeatability_gate": bundle["repeatability_gate"],
            "complete_survivor_digests": bundle[
                "complete_survivor_digests"
            ],
            "typed_gap": gap_status is not None,
            "typed_gap_status": gap_status,
            "model_calls_during_python_assessment": 0,
            **_authority_data(),
        },
        "assessment_digest",
    )


def _registered_envelopes_match(
    artifacts_a: Sequence[object], artifacts_b: Sequence[object]
) -> bool:
    if len(artifacts_a) != PANEL_COUNT or len(artifacts_b) != PANEL_COUNT:
        return False
    fields = (
        "panel_digest",
        "inventory_digest",
        "registry_digest",
        "preparation_digest",
        "prompt_digest",
        "output_schema_digest",
        "presentation",
    )
    return all(
        all(getattr(left, name) == getattr(right, name) for name in fields)
        and left.observation_context_digest != right.observation_context_digest
        for left, right in zip(artifacts_a, artifacts_b, strict=True)
    )


def _replay_record(
    *,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    discovery_batch: Mapping[str, Any],
    semantic_proposal_input: Mapping[str, Any],
    semantic_proposal_result: Mapping[str, Any],
    registry_record: Mapping[str, Any],
    evaluation_a_batch: Mapping[str, Any],
    evaluation_b_batch: Mapping[str, Any],
    assessment: Mapping[str, Any],
    rank_input: Mapping[str, Any],
    rank_result: Mapping[str, Any],
    formula_freeze: Mapping[str, Any],
    visual_journal_summary_digests: Sequence[str],
    semantic_proposer_journal_summary_digest: str,
    ranker_replay_selected_digest: str | None,
) -> dict[str, Any]:
    from bongard.object_scene_semantic_registry import (
        ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE,
    )

    summaries = tuple(visual_journal_summary_digests)
    role_aware = (
        registry_record.get("registry_derivation_mode")
        == ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
        and assessment.get("registry_derivation_mode")
        == ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
        and formula_freeze.get("registry_derivation_mode")
        == ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
    )
    if (
        len(summaries) != VISUAL_CALL_COUNT
        or not role_aware
        or rank_result.get("typed_gap_status")
        != rank_input.get("typed_gap_status")
        or formula_freeze.get("typed_gap_status")
        != rank_input.get("typed_gap_status")
        or (
            formula_freeze.get("status") == "accepted"
            and formula_freeze.get(
                "benchmark_acceptance_authorized_registry"
            )
            is not True
        )
    ):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "cold replay inventory or registry authorization differs"
        )
    return _durable._record(
        {
            "schema": REPLAY_SCHEMA,
            "command_id": COMMAND_ID,
            "authorization_digest": authorization["authorization_digest"],
            "execution_precommit_digest": precommit["precommit_digest"],
            "discovery_batch_digest": discovery_batch["batch_digest"],
            "semantic_proposal_input_digest": semantic_proposal_input[
                "semantic_proposal_input_digest"
            ],
            "semantic_proposal_result_digest": semantic_proposal_result[
                "semantic_proposal_result_digest"
            ],
            "registry_freeze_digest": registry_record[
                "registry_freeze_digest"
            ],
            "evaluation_a_batch_digest": evaluation_a_batch["batch_digest"],
            "evaluation_b_batch_digest": evaluation_b_batch["batch_digest"],
            "assessment_digest": assessment["assessment_digest"],
            "rank_input_freeze_digest": rank_input[
                "rank_input_freeze_digest"
            ],
            "rank_result_digest": rank_result["rank_result_digest"],
            "formula_freeze_digest": formula_freeze["formula_freeze_digest"],
            "status": formula_freeze["status"],
            "typed_gap_status": rank_input["typed_gap_status"],
            "registry_derivation_mode": (
                ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
            ),
            "benchmark_acceptance_authorized_registry": formula_freeze[
                "benchmark_acceptance_authorized_registry"
            ],
            "exact_frequency_fallback_acceptance_authorized": False,
            "visual_journal_summary_digests": list(summaries),
            "semantic_proposer_journal_summary_digest": (
                semantic_proposer_journal_summary_digest
            ),
            "ranker_replay_selected_digest": ranker_replay_selected_digest,
            "model_calls_during_cold_replay": 0,
            "pixels_opened_outside_exact_historical_source_during_replay": 0,
            "all_36_visual_journals_cold_replayed": True,
            "semantic_proposer_journal_cold_replayed": True,
            "semantic_registry_companion_recomputed": True,
            "ranker_journal_cold_replayed_if_called": (
                rank_result["ranker_called"] is True
            ),
            "python_version_space_recomputed": True,
            "formula_recomputed_and_exact_matched": True,
            **_authority_data(),
        },
        "replay_digest",
    )


def _result_record(
    *,
    inputs: _CalibrationInputs,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    discovery_batch: Mapping[str, Any],
    discovery_freeze: Mapping[str, Any],
    semantic_proposal_input: Mapping[str, Any],
    semantic_proposal_result: Mapping[str, Any],
    registry_record: Mapping[str, Any],
    evaluation_a_batch: Mapping[str, Any],
    evaluation_b_batch: Mapping[str, Any],
    evaluation_freeze: Mapping[str, Any],
    role_reveal: Mapping[str, Any],
    assessment: Mapping[str, Any],
    rank_input: Mapping[str, Any],
    rank_result: Mapping[str, Any],
    formula_freeze: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    from bongard.object_scene_semantic_registry import (
        ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE,
    )

    accepted = formula_freeze["status"] == "accepted"
    gap_status = rank_input.get("typed_gap_status")
    role_aware_authorized = (
        semantic_proposal_result.get("semantic_proposal_valid") is True
        and semantic_proposal_result.get("registry_derivation_mode")
        == ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
        and registry_record.get("registry_derivation_mode")
        == ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
        and registry_record.get("benchmark_acceptance_authorized_registry")
        is True
        and assessment.get("registry_derivation_mode")
        == ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
        and formula_freeze.get("benchmark_acceptance_authorized_registry")
        is True
    )
    if accepted and not role_aware_authorized:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "benchmark acceptance lacks a role-aware semantic registry"
        )
    if (
        gap_status != rank_result.get("typed_gap_status")
        or gap_status != formula_freeze.get("typed_gap_status")
        or gap_status != replay.get("typed_gap_status")
        or accepted == (gap_status is not None)
    ):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "benchmark result typed gap trace differs"
        )
    return _durable._record(
        {
            "schema": RESULT_SCHEMA,
            "command_id": COMMAND_ID,
            "historical_source_digest": inputs.source.source_digest,
            "authorization_digest": authorization["authorization_digest"],
            "execution_precommit_digest": precommit["precommit_digest"],
            "discovery_batch_digest": discovery_batch["batch_digest"],
            "discovery_freeze_digest": discovery_freeze["freeze_digest"],
            "semantic_proposal_input_digest": semantic_proposal_input[
                "semantic_proposal_input_digest"
            ],
            "semantic_proposal_result_digest": semantic_proposal_result[
                "semantic_proposal_result_digest"
            ],
            "semantic_proposal_digest": semantic_proposal_result[
                "semantic_proposal_digest"
            ],
            "registry_freeze_digest": registry_record[
                "registry_freeze_digest"
            ],
            "registry_digest": registry_record["registry_digest"],
            "evaluation_a_batch_digest": evaluation_a_batch["batch_digest"],
            "evaluation_b_batch_digest": evaluation_b_batch["batch_digest"],
            "evaluation_freeze_digest": evaluation_freeze["freeze_digest"],
            "role_reveal_digest": role_reveal["role_reveal_digest"],
            "assessment_digest": assessment["assessment_digest"],
            "rank_input_freeze_digest": rank_input[
                "rank_input_freeze_digest"
            ],
            "rank_result_digest": rank_result["rank_result_digest"],
            "formula_freeze_digest": formula_freeze["formula_freeze_digest"],
            "cold_replay_digest": replay["replay_digest"],
            "status": formula_freeze["status"],
            "accepted": accepted,
            "typed_gap_status": gap_status,
            "registry_derivation_mode": (
                ROLE_AWARE_SEMANTIC_REGISTRY_DERIVATION_MODE
            ),
            "benchmark_acceptance_authorized_registry": role_aware_authorized,
            "exact_frequency_fallback_acceptance_authorized": False,
            "selected_survivor_digest": formula_freeze[
                "selected_survivor_digest"
            ],
            "visual_fresh_call_count": VISUAL_CALL_COUNT,
            "visual_reused_call_count": 0,
            "semantic_proposer_fresh_call_count": 1,
            "semantic_proposer_reused_call_count": 0,
            "ranker_fresh_call_count": 1 if accepted else 0,
            "ranker_reused_call_count": 0,
            "physical_model_call_count": (
                VISUAL_CALL_COUNT
                + SEMANTIC_REGISTRY_PROPOSER_CALL_COUNT
                + (1 if accepted else 0)
            ),
            "physical_model_call_denominator_if_accepted": (
                ACCEPTED_PHYSICAL_CALL_COUNT
            ),
            "roles_hidden_through_blind_discovery_freeze": True,
            "roles_revealed_only_to_zero_image_semantic_proposer": True,
            "registered_visual_evaluators_received_roles": False,
            "full_survivor_version_space_persisted": True,
            "formula_frozen_before_any_future_query": True,
            "query_pixels_used": False,
            "unused_train_or_test_pixels_used": False,
            **_authority_data(),
        },
        "result_digest",
    )


def _verified(
    root: Path,
    inputs: _CalibrationInputs,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    discovery_batch: Mapping[str, Any],
    discovery_freeze: Mapping[str, Any],
    semantic_proposal_result: Mapping[str, Any],
    registry_record: Mapping[str, Any],
    evaluation_a_batch: Mapping[str, Any],
    evaluation_b_batch: Mapping[str, Any],
    evaluation_freeze: Mapping[str, Any],
    role_reveal: Mapping[str, Any],
    assessment: Mapping[str, Any],
    rank_input: Mapping[str, Any],
    rank_result: Mapping[str, Any],
    formula_freeze: Mapping[str, Any],
    replay: Mapping[str, Any],
    result: Mapping[str, Any],
) -> VerifiedObjectBongardScenePredicateCalibration:
    return VerifiedObjectBongardScenePredicateCalibration(
        root,
        inputs.source.source_digest,
        authorization["authorization_digest"],
        precommit["precommit_digest"],
        discovery_batch["batch_digest"],
        discovery_freeze["freeze_digest"],
        registry_record["registry_digest"],
        evaluation_a_batch["batch_digest"],
        evaluation_b_batch["batch_digest"],
        evaluation_freeze["freeze_digest"],
        role_reveal["role_reveal_digest"],
        semantic_proposal_result["semantic_proposal_digest"],
        assessment["assessment_digest"],
        rank_input["rank_input_freeze_digest"],
        rank_result["rank_result_digest"],
        formula_freeze["formula_freeze_digest"],
        replay["replay_digest"],
        result["result_digest"],
        result["status"],
        result["selected_survivor_digest"],
        result["visual_fresh_call_count"],
        result["semantic_proposer_fresh_call_count"],
        result["ranker_fresh_call_count"],
    )


def run_object_bongard_scene_predicate_calibration(
    output_root: str | os.PathLike[str],
    *,
    source_root: str | os.PathLike[str] = (
        DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE
    ),
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    minutes: int = DEFAULT_MINUTES,
    executable: str = DEFAULT_EXECUTABLE,
    expected_launcher_sha256: str = DEFAULT_EXPECTED_LAUNCHER_SHA256,
    named_image_transport: NamedImageTransport = run_codex_named_images_structured,
    text_transport: TextTransport = run_codex_text_structured,
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
) -> VerifiedObjectBongardScenePredicateCalibration:
    """Run 36 visual turns, one proposer, and conditional survivor ranking."""

    from bongard.object_scene_semantic_registry import (
        prepare_object_scene_semantic_registry_proposal,
    )

    inputs = _load_inputs(source_root)
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
        label="scene-predicate calibration authorization",
    )
    runtime, fingerprint = _create_runtime(
        authorization,
        cache_snapshotter=cache_snapshotter,
        catalog_snapshotter=catalog_snapshotter,
        launcher_fingerprinter=launcher_fingerprinter,
        runtime_attester=runtime_attester,
    )
    precommit = _write_and_reload(
        root / PRECOMMIT_FILENAME,
        _precommit(inputs, authorization, runtime, fingerprint),
        label="scene-predicate calibration execution precommit",
    )
    runtime = _runtime_from_precommit(precommit, authorization)

    discovery_artifacts, discovery_batch = _execute_visual_batch(
        root,
        stage="discovery",
        inputs=inputs,
        authorization=authorization,
        precommit=precommit,
        runtime=runtime,
        registry=None,
        parallel_workers=parallel_workers,
        transport=named_image_transport,
    )
    discovery_batch = _write_and_reload(
        root / DISCOVERY_BATCH_FILENAME,
        discovery_batch,
        label="scene-predicate discovery batch",
    )
    discovery_artifacts = _restore_visual_batch(
        discovery_batch,
        stage="discovery",
        inputs=inputs,
        authorization=authorization,
        precommit=precommit,
    )
    discovery_freeze = _write_and_reload(
        root / DISCOVERY_FREEZE_FILENAME,
        _freeze_record(
            schema=DISCOVERY_FREEZE_SCHEMA,
            phase="discovery",
            batch_digests=(discovery_batch["batch_digest"],),
            artifact_digests=tuple(
                _artifact_digest(item) for item in discovery_artifacts
            ),
            parent_digest=precommit["precommit_digest"],
            digest_field="freeze_digest",
        ),
        label="scene-predicate discovery freeze",
    )
    role_reveal = _write_and_reload(
        root / ROLE_REVEAL_FILENAME,
        _role_reveal_record(inputs, discovery_freeze),
        label="scene-predicate historical role reveal",
    )
    prepared_semantic_proposal = prepare_object_scene_semantic_registry_proposal(
        discovery_artifacts, inputs.role_reveal_rows
    )
    semantic_proposal_input = _write_and_reload(
        root / SEMANTIC_PROPOSAL_INPUT_FILENAME,
        _semantic_proposal_input_record(
            prepared=prepared_semantic_proposal,
            discovery_freeze=discovery_freeze,
            role_reveal=role_reveal,
        ),
        label="scene semantic registry proposal input",
    )
    prepared_semantic_proposal = _restore_semantic_proposal_input(
        semantic_proposal_input,
        discovery_freeze=discovery_freeze,
        role_reveal=role_reveal,
        discovery_artifacts=discovery_artifacts,
        role_rows=inputs.role_reveal_rows,
    )
    semantic_registry_proposal, semantic_registry, semantic_proposal_result = (
        _execute_semantic_proposal(
            root,
            authorization=authorization,
            precommit=precommit,
            runtime=runtime,
            semantic_proposal_input=semantic_proposal_input,
            prepared=prepared_semantic_proposal,
            text_transport=text_transport,
        )
    )
    semantic_proposal_result = _write_and_reload(
        root / SEMANTIC_PROPOSAL_RESULT_FILENAME,
        semantic_proposal_result,
        label="scene semantic registry proposal result",
    )
    semantic_registry_proposal, semantic_registry = (
        _restore_semantic_proposal_result(
            semantic_proposal_result,
            semantic_proposal_input=semantic_proposal_input,
            prepared=prepared_semantic_proposal,
            discovery_artifacts=discovery_artifacts,
            role_rows=inputs.role_reveal_rows,
        )
    )
    registry, registry_record = _registry_freeze_record(
        discovery_freeze=discovery_freeze,
        role_reveal=role_reveal,
        semantic_proposal_result=semantic_proposal_result,
        semantic_registry_proposal=semantic_registry_proposal,
        registry=semantic_registry,
    )
    registry_record = _write_and_reload(
        root / REGISTRY_FREEZE_FILENAME,
        registry_record,
        label="scene-predicate soft-tag registry freeze",
    )
    registry = _restore_registry_freeze(
        registry_record,
        discovery_freeze=discovery_freeze,
        role_reveal=role_reveal,
        semantic_proposal_result=semantic_proposal_result,
        semantic_registry_proposal=semantic_registry_proposal,
        semantic_registry=semantic_registry,
    )

    evaluation_a_artifacts, evaluation_a_batch = _execute_visual_batch(
        root,
        stage="registered_a",
        inputs=inputs,
        authorization=authorization,
        precommit=precommit,
        runtime=runtime,
        registry=registry,
        parallel_workers=parallel_workers,
        transport=named_image_transport,
    )
    evaluation_a_batch = _write_and_reload(
        root / EVALUATION_A_BATCH_FILENAME,
        evaluation_a_batch,
        label="scene-predicate registered evaluation A batch",
    )
    evaluation_a_artifacts = _restore_visual_batch(
        evaluation_a_batch,
        stage="registered_a",
        inputs=inputs,
        authorization=authorization,
        precommit=precommit,
    )
    evaluation_b_artifacts, evaluation_b_batch = _execute_visual_batch(
        root,
        stage="registered_b",
        inputs=inputs,
        authorization=authorization,
        precommit=precommit,
        runtime=runtime,
        registry=registry,
        parallel_workers=parallel_workers,
        transport=named_image_transport,
    )
    evaluation_b_batch = _write_and_reload(
        root / EVALUATION_B_BATCH_FILENAME,
        evaluation_b_batch,
        label="scene-predicate registered evaluation B batch",
    )
    evaluation_b_artifacts = _restore_visual_batch(
        evaluation_b_batch,
        stage="registered_b",
        inputs=inputs,
        authorization=authorization,
        precommit=precommit,
    )
    if not _registered_envelopes_match(
        evaluation_a_artifacts, evaluation_b_artifacts
    ):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "registered pass A/B model-visible envelopes differ"
        )
    evaluation_freeze = _write_and_reload(
        root / EVALUATION_FREEZE_FILENAME,
        _freeze_record(
            schema=EVALUATION_FREEZE_SCHEMA,
            phase="registered_a_and_b",
            batch_digests=(
                evaluation_a_batch["batch_digest"],
                evaluation_b_batch["batch_digest"],
            ),
            artifact_digests=tuple(
                _artifact_digest(item)
                for item in (*evaluation_a_artifacts, *evaluation_b_artifacts)
            ),
            parent_digest=registry_record["registry_freeze_digest"],
            digest_field="freeze_digest",
            historical_roles_revealed=True,
        ),
        label="scene-predicate joint registered evaluation freeze",
    )
    ir_bundle = _derive_ir_bundle(
        registry=registry,
        semantic_registry_proposal=semantic_registry_proposal,
        discovery_artifacts=discovery_artifacts,
        registered_a_artifacts=evaluation_a_artifacts,
        registered_b_artifacts=evaluation_b_artifacts,
        role_rows=inputs.role_reveal_rows,
    )
    assessment = _write_and_reload(
        root / ASSESSMENT_FILENAME,
        _assessment_record(
            inputs=inputs,
            evaluation_freeze=evaluation_freeze,
            role_reveal=role_reveal,
            semantic_proposal_result=semantic_proposal_result,
            registry=registry,
            ir_bundle=ir_bundle,
        ),
        label="scene-predicate calibration assessment",
    )
    rank_input = _write_and_reload(
        root / RANK_INPUT_FREEZE_FILENAME,
        _rank_input_freeze_record(
            assessment=assessment,
            semantic_proposal_result=semantic_proposal_result,
            complete_survivor_digests=ir_bundle[
                "complete_survivor_digests"
            ],
            slate_rows=ir_bundle["ranker_slate"],
            omitted_rows=ir_bundle["omitted_survivors"],
        ),
        label="scene-predicate rank input freeze",
    )
    rank_result = _write_and_reload(
        root / RANK_RESULT_FILENAME,
        _rank_survivor_slate(
            root,
            inputs=inputs,
            authorization=authorization,
            precommit=precommit,
            runtime=runtime,
            rank_input=rank_input,
            text_transport=text_transport,
        ),
        label="scene-predicate rank result",
    )
    survivors = set(ir_bundle["complete_survivor_digests"])
    candidate_by_digest = {
        item["candidate_digest"]: item
        for item in ir_bundle["candidates"]
        if item["candidate_digest"] in survivors
        and semantic_proposal_result["semantic_proposal_valid"]
    }
    formula_freeze = _write_and_reload(
        root / FORMULA_FREEZE_FILENAME,
        _formula_freeze_record(
            assessment=assessment,
            semantic_proposal_result=semantic_proposal_result,
            rank_input=rank_input,
            rank_result=rank_result,
            candidate_by_digest=candidate_by_digest,
        ),
        label="scene-predicate formula freeze",
    )

    replay_discovery, discovery_summaries = _cold_replay_visual_batch(
        root,
        stage="discovery",
        inputs=inputs,
        authorization=authorization,
        precommit=precommit,
        runtime=runtime,
        registry=None,
        batch=discovery_batch,
    )
    replay_prepared_semantic_proposal = _restore_semantic_proposal_input(
        semantic_proposal_input,
        discovery_freeze=discovery_freeze,
        role_reveal=role_reveal,
        discovery_artifacts=replay_discovery,
        role_rows=inputs.role_reveal_rows,
    )
    replay_semantic_proposal, replay_semantic_registry, semantic_summary = (
        _cold_replay_semantic_proposal(
            root,
            authorization=authorization,
            precommit=precommit,
            runtime=runtime,
            semantic_proposal_input=semantic_proposal_input,
            semantic_proposal_result=semantic_proposal_result,
            prepared=replay_prepared_semantic_proposal,
            discovery_artifacts=replay_discovery,
            role_rows=inputs.role_reveal_rows,
        )
    )
    replay_registry = _restore_registry_freeze(
        registry_record,
        discovery_freeze=discovery_freeze,
        role_reveal=role_reveal,
        semantic_proposal_result=semantic_proposal_result,
        semantic_registry_proposal=replay_semantic_proposal,
        semantic_registry=replay_semantic_registry,
    )
    replay_a, a_summaries = _cold_replay_visual_batch(
        root,
        stage="registered_a",
        inputs=inputs,
        authorization=authorization,
        precommit=precommit,
        runtime=runtime,
        registry=replay_registry,
        batch=evaluation_a_batch,
    )
    replay_b, b_summaries = _cold_replay_visual_batch(
        root,
        stage="registered_b",
        inputs=inputs,
        authorization=authorization,
        precommit=precommit,
        runtime=runtime,
        registry=replay_registry,
        batch=evaluation_b_batch,
    )
    replay_bundle = _derive_ir_bundle(
        registry=replay_registry,
        semantic_registry_proposal=replay_semantic_proposal,
        discovery_artifacts=replay_discovery,
        registered_a_artifacts=replay_a,
        registered_b_artifacts=replay_b,
        role_rows=inputs.role_reveal_rows,
    )
    if replay_bundle != ir_bundle:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "Python scene predicate version space differs on cold replay"
        )
    replay_selected = _cold_replay_ranker(
        root,
        authorization=authorization,
        precommit=precommit,
        runtime=runtime,
        rank_input=rank_input,
        rank_result=rank_result,
    )
    if replay_selected != formula_freeze["selected_survivor_digest"]:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "formula freeze differs from cold-replayed ranker selection"
        )
    replay = _write_and_reload(
        root / REPLAY_FILENAME,
        _replay_record(
            authorization=authorization,
            precommit=precommit,
            discovery_batch=discovery_batch,
            semantic_proposal_input=semantic_proposal_input,
            semantic_proposal_result=semantic_proposal_result,
            registry_record=registry_record,
            evaluation_a_batch=evaluation_a_batch,
            evaluation_b_batch=evaluation_b_batch,
            assessment=assessment,
            rank_input=rank_input,
            rank_result=rank_result,
            formula_freeze=formula_freeze,
            visual_journal_summary_digests=(
                *discovery_summaries,
                *a_summaries,
                *b_summaries,
            ),
            semantic_proposer_journal_summary_digest=semantic_summary,
            ranker_replay_selected_digest=replay_selected,
        ),
        label="scene-predicate calibration cold replay",
    )
    result = _write_and_reload(
        root / RESULT_FILENAME,
        _result_record(
            inputs=inputs,
            authorization=authorization,
            precommit=precommit,
            discovery_batch=discovery_batch,
            discovery_freeze=discovery_freeze,
            semantic_proposal_input=semantic_proposal_input,
            semantic_proposal_result=semantic_proposal_result,
            registry_record=registry_record,
            evaluation_a_batch=evaluation_a_batch,
            evaluation_b_batch=evaluation_b_batch,
            evaluation_freeze=evaluation_freeze,
            role_reveal=role_reveal,
            assessment=assessment,
            rank_input=rank_input,
            rank_result=rank_result,
            formula_freeze=formula_freeze,
            replay=replay,
        ),
        label="scene-predicate calibration result",
    )
    return _verified(
        root,
        inputs,
        authorization,
        precommit,
        discovery_batch,
        discovery_freeze,
        semantic_proposal_result,
        registry_record,
        evaluation_a_batch,
        evaluation_b_batch,
        evaluation_freeze,
        role_reveal,
        assessment,
        rank_input,
        rank_result,
        formula_freeze,
        replay,
        result,
    )


def verify_object_bongard_scene_predicate_calibration(
    output_root: str | os.PathLike[str],
    *,
    source_root: str | os.PathLike[str] = (
        DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE
    ),
) -> VerifiedObjectBongardScenePredicateCalibration:
    """Cold-verify the complete calibration directory with transports forbidden."""

    inputs = _load_inputs(source_root)
    root = _existing_root(output_root)
    expected_inventory = {
        AUTHORIZATION_FILENAME,
        PRECOMMIT_FILENAME,
        DISCOVERY_BATCH_FILENAME,
        DISCOVERY_FREEZE_FILENAME,
        SEMANTIC_PROPOSAL_INPUT_FILENAME,
        SEMANTIC_PROPOSAL_RESULT_FILENAME,
        REGISTRY_FREEZE_FILENAME,
        EVALUATION_A_BATCH_FILENAME,
        EVALUATION_B_BATCH_FILENAME,
        EVALUATION_FREEZE_FILENAME,
        ROLE_REVEAL_FILENAME,
        ASSESSMENT_FILENAME,
        RANK_INPUT_FREEZE_FILENAME,
        RANK_RESULT_FILENAME,
        FORMULA_FREEZE_FILENAME,
        REPLAY_FILENAME,
        RESULT_FILENAME,
        JOURNAL_DIRECTORY,
    }
    if {item.name for item in root.iterdir()} != expected_inventory:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "calibration directory inventory differs"
        )
    authorization = _durable._validate_record(
        _durable._read_record(root / AUTHORIZATION_FILENAME, "authorization"),
        schema=AUTHORIZATION_SCHEMA,
        digest_field="authorization_digest",
        label="scene-predicate calibration authorization",
    )
    expected_authorization = _authorization(
        inputs,
        parallel_workers=authorization["parallel_workers"],
        minutes=authorization["runtime_policy"]["minutes"],
        executable=authorization["runtime_policy"]["executable"],
        expected_launcher_sha256=authorization["runtime_policy"][
            "expected_launcher_sha256"
        ],
    )
    if authorization != expected_authorization:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "authorization differs on cold replay"
        )
    precommit = _durable._read_record(root / PRECOMMIT_FILENAME, "precommit")
    runtime = _runtime_from_precommit(precommit, authorization)
    expected_precommit = _precommit(
        inputs,
        authorization,
        runtime,
        precommit["launcher_fingerprint"],
    )
    if precommit != expected_precommit:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "precommit differs on cold replay"
        )

    discovery_batch = _durable._read_record(
        root / DISCOVERY_BATCH_FILENAME, "discovery batch"
    )
    discovery_artifacts = _restore_visual_batch(
        discovery_batch,
        stage="discovery",
        inputs=inputs,
        authorization=authorization,
        precommit=precommit,
    )
    discovery_freeze = _durable._read_record(
        root / DISCOVERY_FREEZE_FILENAME, "discovery freeze"
    )
    expected_discovery_freeze = _freeze_record(
        schema=DISCOVERY_FREEZE_SCHEMA,
        phase="discovery",
        batch_digests=(discovery_batch["batch_digest"],),
        artifact_digests=tuple(
            _artifact_digest(item) for item in discovery_artifacts
        ),
        parent_digest=precommit["precommit_digest"],
        digest_field="freeze_digest",
    )
    if discovery_freeze != expected_discovery_freeze:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "discovery freeze differs on cold replay"
        )
    role_reveal = _durable._read_record(
        root / ROLE_REVEAL_FILENAME, "historical role reveal"
    )
    if role_reveal != _role_reveal_record(inputs, discovery_freeze):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "historical role reveal differs on replay"
        )
    semantic_proposal_input = _durable._read_record(
        root / SEMANTIC_PROPOSAL_INPUT_FILENAME,
        "semantic registry proposal input",
    )
    prepared_semantic_proposal = _restore_semantic_proposal_input(
        semantic_proposal_input,
        discovery_freeze=discovery_freeze,
        role_reveal=role_reveal,
        discovery_artifacts=discovery_artifacts,
        role_rows=inputs.role_reveal_rows,
    )
    semantic_proposal_result = _durable._read_record(
        root / SEMANTIC_PROPOSAL_RESULT_FILENAME,
        "semantic registry proposal result",
    )
    semantic_registry_proposal, semantic_registry = (
        _restore_semantic_proposal_result(
            semantic_proposal_result,
            semantic_proposal_input=semantic_proposal_input,
            prepared=prepared_semantic_proposal,
            discovery_artifacts=discovery_artifacts,
            role_rows=inputs.role_reveal_rows,
        )
    )
    registry_record = _durable._read_record(
        root / REGISTRY_FREEZE_FILENAME, "soft-tag registry freeze"
    )
    registry = _restore_registry_freeze(
        registry_record,
        discovery_freeze=discovery_freeze,
        role_reveal=role_reveal,
        semantic_proposal_result=semantic_proposal_result,
        semantic_registry_proposal=semantic_registry_proposal,
        semantic_registry=semantic_registry,
    )
    evaluation_a_batch = _durable._read_record(
        root / EVALUATION_A_BATCH_FILENAME, "registered evaluation A batch"
    )
    evaluation_a_artifacts = _restore_visual_batch(
        evaluation_a_batch,
        stage="registered_a",
        inputs=inputs,
        authorization=authorization,
        precommit=precommit,
    )
    evaluation_b_batch = _durable._read_record(
        root / EVALUATION_B_BATCH_FILENAME, "registered evaluation B batch"
    )
    evaluation_b_artifacts = _restore_visual_batch(
        evaluation_b_batch,
        stage="registered_b",
        inputs=inputs,
        authorization=authorization,
        precommit=precommit,
    )
    if not _registered_envelopes_match(
        evaluation_a_artifacts, evaluation_b_artifacts
    ):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "registered A/B visible envelopes differ on replay"
        )
    evaluation_freeze = _durable._read_record(
        root / EVALUATION_FREEZE_FILENAME, "registered evaluation freeze"
    )
    expected_evaluation_freeze = _freeze_record(
        schema=EVALUATION_FREEZE_SCHEMA,
        phase="registered_a_and_b",
        batch_digests=(
            evaluation_a_batch["batch_digest"],
            evaluation_b_batch["batch_digest"],
        ),
        artifact_digests=tuple(
            _artifact_digest(item)
            for item in (*evaluation_a_artifacts, *evaluation_b_artifacts)
        ),
        parent_digest=registry_record["registry_freeze_digest"],
        digest_field="freeze_digest",
        historical_roles_revealed=True,
    )
    if evaluation_freeze != expected_evaluation_freeze:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "registered evaluation freeze differs on replay"
        )
    replay_discovery, discovery_summaries = _cold_replay_visual_batch(
        root,
        stage="discovery",
        inputs=inputs,
        authorization=authorization,
        precommit=precommit,
        runtime=runtime,
        registry=None,
        batch=discovery_batch,
    )
    replay_prepared_semantic_proposal = _restore_semantic_proposal_input(
        semantic_proposal_input,
        discovery_freeze=discovery_freeze,
        role_reveal=role_reveal,
        discovery_artifacts=replay_discovery,
        role_rows=inputs.role_reveal_rows,
    )
    replay_semantic_proposal, replay_semantic_registry, semantic_summary = (
        _cold_replay_semantic_proposal(
            root,
            authorization=authorization,
            precommit=precommit,
            runtime=runtime,
            semantic_proposal_input=semantic_proposal_input,
            semantic_proposal_result=semantic_proposal_result,
            prepared=replay_prepared_semantic_proposal,
            discovery_artifacts=replay_discovery,
            role_rows=inputs.role_reveal_rows,
        )
    )
    replay_registry = _restore_registry_freeze(
        registry_record,
        discovery_freeze=discovery_freeze,
        role_reveal=role_reveal,
        semantic_proposal_result=semantic_proposal_result,
        semantic_registry_proposal=replay_semantic_proposal,
        semantic_registry=replay_semantic_registry,
    )
    replay_a, a_summaries = _cold_replay_visual_batch(
        root,
        stage="registered_a",
        inputs=inputs,
        authorization=authorization,
        precommit=precommit,
        runtime=runtime,
        registry=replay_registry,
        batch=evaluation_a_batch,
    )
    replay_b, b_summaries = _cold_replay_visual_batch(
        root,
        stage="registered_b",
        inputs=inputs,
        authorization=authorization,
        precommit=precommit,
        runtime=runtime,
        registry=replay_registry,
        batch=evaluation_b_batch,
    )
    ir_bundle = _derive_ir_bundle(
        registry=replay_registry,
        semantic_registry_proposal=replay_semantic_proposal,
        discovery_artifacts=replay_discovery,
        registered_a_artifacts=replay_a,
        registered_b_artifacts=replay_b,
        role_rows=inputs.role_reveal_rows,
    )
    assessment = _durable._read_record(
        root / ASSESSMENT_FILENAME, "calibration assessment"
    )
    expected_assessment = _assessment_record(
        inputs=inputs,
        evaluation_freeze=evaluation_freeze,
        role_reveal=role_reveal,
        semantic_proposal_result=semantic_proposal_result,
        registry=replay_registry,
        ir_bundle=ir_bundle,
    )
    if assessment != expected_assessment:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "calibration assessment differs on replay"
        )
    rank_input = _durable._read_record(
        root / RANK_INPUT_FREEZE_FILENAME, "rank input freeze"
    )
    expected_rank_input = _rank_input_freeze_record(
        assessment=assessment,
        semantic_proposal_result=semantic_proposal_result,
        complete_survivor_digests=ir_bundle["complete_survivor_digests"],
        slate_rows=ir_bundle["ranker_slate"],
        omitted_rows=ir_bundle["omitted_survivors"],
    )
    if rank_input != expected_rank_input:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "rank input freeze differs on replay"
        )
    rank_result = _durable._read_record(
        root / RANK_RESULT_FILENAME, "rank result"
    )
    replay_selected = _cold_replay_ranker(
        root,
        authorization=authorization,
        precommit=precommit,
        runtime=runtime,
        rank_input=rank_input,
        rank_result=rank_result,
    )
    survivors = set(ir_bundle["complete_survivor_digests"])
    candidate_by_digest = {
        item["candidate_digest"]: item
        for item in ir_bundle["candidates"]
        if item["candidate_digest"] in survivors
        and semantic_proposal_result["semantic_proposal_valid"]
    }
    formula_freeze = _durable._read_record(
        root / FORMULA_FREEZE_FILENAME, "formula freeze"
    )
    expected_formula_freeze = _formula_freeze_record(
        assessment=assessment,
        semantic_proposal_result=semantic_proposal_result,
        rank_input=rank_input,
        rank_result=rank_result,
        candidate_by_digest=candidate_by_digest,
    )
    if (
        formula_freeze != expected_formula_freeze
        or formula_freeze["selected_survivor_digest"] != replay_selected
    ):
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "formula freeze differs on replay"
        )
    replay = _durable._read_record(root / REPLAY_FILENAME, "cold replay")
    expected_replay = _replay_record(
        authorization=authorization,
        precommit=precommit,
        discovery_batch=discovery_batch,
        semantic_proposal_input=semantic_proposal_input,
        semantic_proposal_result=semantic_proposal_result,
        registry_record=registry_record,
        evaluation_a_batch=evaluation_a_batch,
        evaluation_b_batch=evaluation_b_batch,
        assessment=assessment,
        rank_input=rank_input,
        rank_result=rank_result,
        formula_freeze=formula_freeze,
        visual_journal_summary_digests=(
            *discovery_summaries,
            *a_summaries,
            *b_summaries,
        ),
        semantic_proposer_journal_summary_digest=semantic_summary,
        ranker_replay_selected_digest=replay_selected,
    )
    if replay != expected_replay:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "cold replay record differs"
        )
    result = _durable._read_record(root / RESULT_FILENAME, "calibration result")
    expected_result = _result_record(
        inputs=inputs,
        authorization=authorization,
        precommit=precommit,
        discovery_batch=discovery_batch,
        discovery_freeze=discovery_freeze,
        semantic_proposal_input=semantic_proposal_input,
        semantic_proposal_result=semantic_proposal_result,
        registry_record=registry_record,
        evaluation_a_batch=evaluation_a_batch,
        evaluation_b_batch=evaluation_b_batch,
        evaluation_freeze=evaluation_freeze,
        role_reveal=role_reveal,
        assessment=assessment,
        rank_input=rank_input,
        rank_result=rank_result,
        formula_freeze=formula_freeze,
        replay=replay,
    )
    if result != expected_result:
        raise ObjectBongardScenePredicateCalibrationCommandError(
            "calibration result differs"
        )
    return _verified(
        root,
        inputs,
        authorization,
        precommit,
        discovery_batch,
        discovery_freeze,
        semantic_proposal_result,
        registry_record,
        evaluation_a_batch,
        evaluation_b_batch,
        evaluation_freeze,
        role_reveal,
        assessment,
        rank_input,
        rank_result,
        formula_freeze,
        replay,
        result,
    )


def audit_object_bongard_scene_predicate_calibration_source(
    source_root: str | os.PathLike[str] = (
        DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE
    ),
) -> dict[str, Any]:
    """Audit the exact already-exposed source and deterministic inventories only."""

    inputs = _load_inputs(source_root)
    return {
        "schema": "gkm.bongard-scene-predicate-calibration-source-audit.v1",
        "command_id": COMMAND_ID,
        "historical_source_digest": inputs.source.source_digest,
        "historical_plan_file_sha256": inputs.source.historical_plan_file_sha256,
        "historical_plan_record_digest": inputs.source.historical_plan_record_digest,
        "panel_count": len(inputs.panels),
        "blind_panel_ids": [item.blind_panel_id for item in inputs.panels],
        "proposal_inventory_digests": [
            item.inventory_digest for item in inputs.inventories
        ],
        "proposal_counts": [len(item.objects) for item in inputs.inventories],
        "inventory_statuses": [item.inventory_status for item in inputs.inventories],
        "role_commitment_digest": inputs.role_commitment_digest,
        "role_reveal_serialized": False,
        "physical_model_call_count": 0,
        "query_pixels_used": False,
        "unused_train_or_test_pixels_used": False,
        **_authority_data(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    audit = commands.add_parser("audit-source")
    audit.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE,
    )
    launch = commands.add_parser("launch")
    verify = commands.add_parser("verify")
    for command in (launch, verify):
        command.add_argument("--output-root", required=True, type=Path)
        command.add_argument(
            "--source-root",
            type=Path,
            default=DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE,
        )
    launch.add_argument("--parallel-workers", type=int, default=DEFAULT_PARALLEL_WORKERS)
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
        if args.operation == "audit-source":
            print(
                canonical_json(
                    audit_object_bongard_scene_predicate_calibration_source(
                        args.source_root
                    )
                ).decode("utf-8")
            )
            return 0
        if args.operation == "launch":
            verified = run_object_bongard_scene_predicate_calibration(
                args.output_root,
                source_root=args.source_root,
                parallel_workers=args.parallel_workers,
                minutes=args.minutes,
                executable=args.executable,
                expected_launcher_sha256=args.expected_launcher_sha256,
            )
        else:
            verified = verify_object_bongard_scene_predicate_calibration(
                args.output_root,
                source_root=args.source_root,
            )
    except Exception as exc:
        print(
            canonical_json(
                {
                    "schema": "gkm.bongard-scene-predicate-calibration-command-error.v1",
                    "error_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "raw_message_persisted": False,
                }
            ).decode("utf-8"),
            file=sys.stderr,
        )
        return 1
    print(
        canonical_json(
            {
                "schema": "gkm.bongard-scene-predicate-calibration-summary.v1",
                "output_root": str(verified.output_root),
                "result_digest": verified.result_digest,
                "cold_replay_digest": verified.replay_digest,
                "status": verified.status,
                "selected_survivor_digest": verified.selected_survivor_digest,
                "visual_fresh_call_count": verified.visual_fresh_call_count,
                "semantic_proposer_fresh_call_count": (
                    verified.semantic_proposer_fresh_call_count
                ),
                "ranker_fresh_call_count": verified.ranker_fresh_call_count,
                **_authority_data(),
            }
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = (
    "ACCEPTED_PHYSICAL_CALL_COUNT",
    "AUTHORIZATION_FILENAME",
    "DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE",
    "FORMULA_FREEZE_FILENAME",
    "ObjectBongardScenePredicateCalibrationCommandError",
    "RESULT_FILENAME",
    "SEMANTIC_REGISTRY_PROPOSER_CALL_COUNT",
    "VISUAL_CALL_COUNT",
    "VerifiedObjectBongardScenePredicateCalibration",
    "audit_object_bongard_scene_predicate_calibration_source",
    "main",
    "object_bongard_scene_predicate_calibration_command_source_digest",
    "run_object_bongard_scene_predicate_calibration",
    "verify_object_bongard_scene_predicate_calibration",
)

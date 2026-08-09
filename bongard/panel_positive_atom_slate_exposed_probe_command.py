"""Run the bounded affirmative atom slate on exposed support pixels only.

The command reads the fixed ``_0001`` support-gap archive, makes one journaled
eight-atom proposer call over its six primary and six contrast supports, then
makes exactly twelve journaled one-panel calls that score all eight frozen atoms
at once.  After every artifact and journal terminal cold-replays, deterministic
Python enumerates the eight singletons and 28 affirmative pairs.  No query input,
release operation, query observer, negation, polarity choice, or Lean API exists.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import inspect
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from bongard import panel_positive_atom_slate as _atom
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTurnJournalSummary,
)
from bongard.panel_feature_exposed_support_smoke_command import (
    DEFAULT_LAUNCHER_SHA256,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SOURCE_ARCHIVE,
    _read_source,
    _record,
    _runtime,
    _write_once_or_verify,
)
from bongard.panel_positive_atom_slate import (
    ATOM_COUNT,
    FORMULA_COUNT,
    MINIMUM_DECISIVE_PER_SIDE,
    AtomPanelScoreArtifact,
    AtomPanelScoreRequest,
    AtomSlateProposerArtifact,
    AtomSlateProposerRequest,
    AtomSupportInventory,
    atom_panel_score_output_schema,
    atom_panel_score_prompt,
    atom_slate_proposer_output_schema,
    atom_slate_proposer_prompt,
    observe_affirmative_atom_panel,
    panel_positive_atom_slate_source_digest,
    propose_affirmative_atom_slate,
    verify_atom_panel_score_artifact,
    verify_atom_slate_proposer_artifact,
)
from bongard.panel_typed_codex_observer import build_panel_only_observation_context


AUTHORIZATION_SCHEMA = "gkm.bongard-positive-atom-slate-exposed-authorization.v1"
PRECOMMIT_SCHEMA = "gkm.bongard-positive-atom-slate-exposed-precommit.v1"
COMPLETION_SCHEMA = "gkm.bongard-positive-atom-slate-exposed-completion.v1"
TARGET_TASK_ID = "hd_convex-has_four_straight_lines_0001"
DEFAULT_OUTPUT_ROOT = Path(
    "downloads/ShapeBongard_V2_full/"
    "panel_positive_atom_slate_exposed_probe_20260809_v1"
)


class AtomSlateExposedProbeError(RuntimeError):
    """The exposed archive, exact journal, artifact, or inventory differs."""


def atom_slate_exposed_probe_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _authorization_and_precommit(
    *,
    task: object,
    panel_ids: Sequence[str],
    panels: Sequence[bytes],
    source_archive_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if getattr(task, "task_id", None) != TARGET_TASK_ID:
        raise AtomSlateExposedProbeError("atom slate probe is bound to another task")
    expected_ids = task.side_0_support_panel_ids + task.side_1_support_panel_ids
    if tuple(panel_ids) != expected_ids or len(panels) != 12:
        raise AtomSlateExposedProbeError("exact exposed support order differs")
    authorization = _record(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "command_source_digest": atom_slate_exposed_probe_source_digest(),
            "atom_core_source_digest": panel_positive_atom_slate_source_digest(),
            "source_archive_sha256": source_archive_sha256,
            "task_plan": task.to_data(),
            "support_panel_ids": list(panel_ids),
            "support_png_sha256": [hashlib.sha256(item).hexdigest() for item in panels],
            "primary_orientation": "side_0",
            "support_panel_count": 12,
            "query_pixels_available_to_command": False,
            "query_release_or_observation_authorized": False,
            "support_only_headless_atom_proposer": True,
            "positive_vs_contrast_role_proposer_visible": True,
            "panel_role_observer_visible": False,
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
            "schema": PRECOMMIT_SCHEMA,
            "authorization_digest": authorization["record_digest"],
            "physical_call_plan": {
                "support_atom_proposer": 1,
                "support_all_atom_panel_observers": 12,
                "query": 0,
            },
            "atom_slots": ATOM_COUNT,
            "formula_count": FORMULA_COUNT,
            "formula_space": "eight_singletons_and_twenty_eight_affirmative_pairs",
            "formula_enumeration_after_all_twelve_rows": True,
            "minimum_decisive_per_side": MINIMUM_DECISIVE_PER_SIDE,
            "maximum_indeterminate_per_side": 1,
            "contradictions_allowed_per_side": 0,
            "errors_allowed": 0,
            "exactly_once_journal_required_for_every_model_call": True,
            "model_formula_threshold_or_polarity_selection_allowed": False,
            "negative_formula_or_coherent_contrast_concept_allowed": False,
            "query_pixels_available_to_command": False,
            "python_is_canonical_authority": True,
            "lean_required": False,
        }
    )
    return authorization, precommit


def _typed_runtime(runtime: object, first_panel: bytes):
    return build_panel_only_observation_context(
        first_panel,
        model=runtime.model,
        reasoning_effort=runtime.reasoning_effort,
        expected_launcher_digest=runtime.expected_launcher_digest,
        cloud_policy_cache_snapshot=runtime.cloud_policy_cache_snapshot,
        model_catalog_snapshot=runtime.model_catalog_snapshot,
        no_tools_attestation=runtime.no_tools_attestation,
    ).runtime


def _observer_turn(
    *,
    ordinal: int,
    panel: bytes,
    task: object,
    proposer: AtomSlateProposerArtifact,
    proposer_terminal: ObjectBongardTurnJournalSummary,
    runtime: object,
    root: Path,
    authorization_digest: str,
    precommit_digest: str,
) -> tuple[int, AtomPanelScoreArtifact, ObjectBongardTurnJournalSummary]:
    request = AtomPanelScoreRequest.build_from_proposer(
        panel,
        ordinal,
        proposer,
        expected_proposer_artifact_digest=proposer.artifact_digest,
    )
    prompt = atom_panel_score_prompt(request)
    schema = atom_panel_score_output_schema(request)
    journal = ObjectBongardNamedImageTurnJournalTransport(
        root / "journals" / f"support_{ordinal:02d}",
        authorization_digest=authorization_digest,
        execution_precommit_digest=precommit_digest,
        task_id=task.task_id,
        turn_kind=f"positive_atom_support_{ordinal:02d}",
        expected_prompt=prompt,
        expected_images=(("panel.png", panel),),
        expected_output_schema=schema,
        runtime=runtime,
        underlying_transport=_atom.run_codex_named_images_structured,
    )
    artifact = observe_affirmative_atom_panel(
        panel,
        request=request,
        source_proposer_artifact=proposer,
        expected_source_proposer_artifact_digest=proposer.artifact_digest,
        proposer_journal_terminal=proposer_terminal,
        minutes=runtime.minutes,
        verbose=runtime.verbose,
        executable=runtime.executable,
        cloud_policy_cache_snapshot=runtime.cloud_policy_cache_snapshot,
        expected_launcher_digest=runtime.expected_launcher_digest,
        model_catalog_snapshot=runtime.model_catalog_snapshot,
        no_tools_attestation=runtime.no_tools_attestation,
        transport=journal,
    )
    terminal = journal.verify()
    restored = verify_atom_panel_score_artifact(
        artifact,
        panel,
        expected_artifact_digest=artifact.artifact_digest,
        source_proposer_artifact=proposer,
        expected_source_proposer_artifact_digest=proposer.artifact_digest,
        proposer_journal_terminal=proposer_terminal,
        panel_journal_terminal=terminal,
        expected_request_digest=request.request_digest,
    )
    _write_once_or_verify(
        root / "panel_artifacts" / f"support_{ordinal:02d}.json",
        restored.to_data(),
    )
    return ordinal, restored, terminal


def run_atom_slate_exposed_support_probe(
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
    """Run the exact 13-call support-only atom slate and persist its inventory."""

    if type(workers) is not int or not 1 <= workers <= 12:
        raise AtomSlateExposedProbeError("workers must lie in 1..12")
    source = Path(os.path.abspath(os.fspath(source_archive)))
    root = Path(os.path.abspath(os.fspath(output_root)))
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise AtomSlateExposedProbeError("output root is unsafe")
    task, panel_ids, panels, source_digest = _read_source(source)
    authorization, precommit = _authorization_and_precommit(
        task=task,
        panel_ids=panel_ids,
        panels=panels,
        source_archive_sha256=source_digest,
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
    typed_runtime = _typed_runtime(runtime, panels[0])
    proposer_request = AtomSlateProposerRequest.build(
        panels[:6], panels[6:], runtime=typed_runtime
    )
    proposer_journal = ObjectBongardNamedImageTurnJournalTransport(
        root / "journals" / "atom_proposer",
        authorization_digest=authorization["record_digest"],
        execution_precommit_digest=precommit["record_digest"],
        task_id=task.task_id,
        turn_kind="positive_atom_slate_proposer",
        expected_prompt=atom_slate_proposer_prompt(proposer_request),
        expected_images=tuple(
            zip(_atom.PROPOSER_IMAGE_NAMES, panels, strict=True)
        ),
        expected_output_schema=atom_slate_proposer_output_schema(proposer_request),
        runtime=runtime,
        underlying_transport=_atom.run_codex_named_images_structured,
    )
    proposer = propose_affirmative_atom_slate(
        panels[:6],
        panels[6:],
        request=proposer_request,
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
    proposer_terminal = proposer_journal.verify()
    proposer = verify_atom_slate_proposer_artifact(
        proposer,
        panels[:6],
        panels[6:],
        expected_artifact_digest=proposer.artifact_digest,
        proposer_journal_terminal=proposer_terminal,
    )
    _write_once_or_verify(root / "proposer_artifact.json", proposer.to_data())
    artifacts: list[AtomPanelScoreArtifact | None] = [None] * 12
    terminals: list[ObjectBongardTurnJournalSummary | None] = [None] * 12
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _observer_turn,
                ordinal=ordinal,
                panel=panel,
                task=task,
                proposer=proposer,
                proposer_terminal=proposer_terminal,
                runtime=runtime,
                root=root,
                authorization_digest=authorization["record_digest"],
                precommit_digest=precommit["record_digest"],
            )
            for ordinal, panel in enumerate(panels)
        ]
        for future in as_completed(futures):
            ordinal, artifact, terminal = future.result()
            artifacts[ordinal] = artifact
            terminals[ordinal] = terminal
    if any(item is None for item in artifacts + terminals):
        raise AtomSlateExposedProbeError("atom support observations are incomplete")
    frozen_artifacts = tuple(item for item in artifacts if item is not None)
    frozen_terminals = tuple(item for item in terminals if item is not None)
    inventory = AtomSupportInventory.create(
        proposer.slate, tuple(item.row for item in frozen_artifacts)
    )
    _write_once_or_verify(root / "support_inventory.json", inventory.to_data())
    completion = _record(
        {
            "schema": COMPLETION_SCHEMA,
            "command_source_digest": atom_slate_exposed_probe_source_digest(),
            "atom_core_source_digest": panel_positive_atom_slate_source_digest(),
            "authorization_digest": authorization["record_digest"],
            "execution_precommit_digest": precommit["record_digest"],
            "runtime_evidence_digest": runtime_evidence["record_digest"],
            "task_id": task.task_id,
            "proposer_artifact_digest": proposer.artifact_digest,
            "proposer_journal_terminal": proposer_terminal.to_data(),
            "panel_artifact_digests": [item.artifact_digest for item in frozen_artifacts],
            "panel_journal_terminals": [item.to_data() for item in frozen_terminals],
            "support_inventory": inventory.to_data(),
            "support_inventory_digest": inventory.inventory_digest,
            "admitted_formula_count": len(inventory.admitted_formulas),
            "support_formula_admitted": bool(inventory.admitted_formulas),
            "status": "support_pass" if inventory.admitted_formulas else "support_gap",
            "physical_model_calls": 13,
            "proposer_model_calls": 1,
            "support_observer_model_calls": 12,
            "query_observer_calls": 0,
            "query_release_calls": 0,
            "query_pixels_available_to_command": False,
            "query_release_authorized": False,
            "all_artifacts_benchmark_sealable": (
                proposer.benchmark_sealable
                and all(item.benchmark_sealable for item in frozen_artifacts)
            ),
            "model_formula_threshold_or_polarity_selection": False,
            "negative_formula_present": False,
            "python_is_canonical_authority": True,
            "engineering_only": True,
            "scientific_benchmark": False,
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
    result = run_atom_slate_exposed_support_probe(
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


assert "query" not in inspect.signature(run_atom_slate_exposed_support_probe).parameters


if __name__ == "__main__":
    raise SystemExit(main())

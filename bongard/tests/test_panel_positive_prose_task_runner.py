from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from bongard import panel_positive_prose_observer as observer_module
from bongard import panel_support_positive_proposer as proposer_module
from bongard.object_bongard_release_gate import (
    persist_object_bongard_task_commit,
    persist_object_bongard_task_freeze,
)
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTurnRuntime,
)
from bongard.panel_feature_extracted_release_gate import (
    release_panel_feature_extracted_query_panel,
    release_panel_feature_extracted_support_panel,
)
from bongard.panel_positive_prose_evidence_bundle import (
    PROPOSER_TURN_KIND,
    PositiveProseEvidenceBundle,
    PositiveProseEvidenceError,
    PositiveProseEvidencePhase,
    PositiveProseEvidenceRow,
    PositiveProseJournalTerminal,
    PositiveProsePanelRole,
    cold_replay_positive_prose_evidence_bundle,
)
from bongard.panel_positive_prose_observer import (
    PositiveProsePanelContext,
    PositiveProsePanelRequest,
    observe_positive_prose_panel,
    positive_prose_panel_output_schema,
    positive_prose_panel_prompt,
)
from bongard.panel_positive_prose_task_runner import (
    PositiveProseQueryDecision,
    PositiveProseQueryOutcome,
    PositiveProseSupportAdmission,
    PositiveProseSupportStatus,
    PositiveProseTaskFreeze,
    PositiveProseTaskFreezeCommit,
    PositiveProseTaskRunnerError,
    cold_replay_positive_prose_query_decision,
    verify_positive_prose_task_commit,
    verify_positive_prose_task_freeze,
)
from bongard.panel_support_positive_proposer import (
    SUPPORT_POSITIVE_PRESENTATION_NAMES,
    SupportPositiveProposerRequest,
    propose_support_positive_rubric,
    support_positive_proposer_output_schema,
    support_positive_proposer_prompt,
)
from bongard.prototype_scene_observer import prototype_scene_transport_source_digest
from bongard.tests.test_panel_feature_extracted_release_gate import _fixture, _prepare
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _receipt,
)
from bongard.transport import CodexStructuredResult


def _turn_runtime() -> ObjectBongardTurnRuntime:
    return ObjectBongardTurnRuntime(
        model=MODEL,
        reasoning_effort=EFFORT,
        minutes=15,
        verbose=False,
        executable="codex",
        cloud_policy_cache_snapshot=None,
        model_catalog_snapshot=NO_TOOLS_KWARGS["model_catalog_snapshot"],
        expected_launcher_digest=LAUNCHER_DIGEST,
        no_tools_attestation=NO_TOOLS_KWARGS["no_tools_attestation"],
        transport_source_digest=prototype_scene_transport_source_digest(),
    )


def _proposer_payload() -> dict[str, object]:
    result: dict[str, object] = {
        "cue_text": "convex carrier and four straight structural runs",
        "component_1": "convex carrier",
        "component_2": "four straight structural runs",
    }
    for index, name in enumerate(SUPPORT_POSITIVE_PRESENTATION_NAMES):
        result[name.removesuffix(".png") + "_estimate"] = (
            "supports" if index < 6 else "does_not_support"
        )
    return result


def _interval(kind: str) -> dict[str, int]:
    return {
        "present": {"lower": 3, "upper": 4},
        "certified_absent": {"lower": 0, "upper": 1},
        "indeterminate": {"lower": 1, "upper": 3},
    }[kind]


def _journaled_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    prepared,
    support_pngs: tuple[bytes, ...],
    support_kinds: tuple[str, ...],
    query_pngs: tuple[bytes, ...] = (),
    query_kinds: tuple[str, ...] = (),
    source_pair=None,
):
    task = prepared.plan.tasks[0]
    runtime = PositiveProsePanelContext.build(
        support_pngs[0],
        panel_id=task.side_0_support_panel_ids[0],
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        cloud_policy_cache_snapshot=None,
        **NO_TOOLS_KWARGS,
    ).runtime
    if source_pair is None:
        request = SupportPositiveProposerRequest.build(
            support_pngs[:6], support_pngs[6:], runtime=runtime
        )
        proposer_payload = _proposer_payload()

        def proposer_transport(prompt, paths, names, schema, **kwargs):
            return CodexStructuredResult(
                deepcopy(proposer_payload),
                _receipt(prompt, paths, names, schema, proposer_payload),
            )

        monkeypatch.setattr(
            proposer_module, "run_codex_named_images_structured", proposer_transport
        )
        proposer_journal = ObjectBongardNamedImageTurnJournalTransport(
            tmp_path / "journal-proposer",
            authorization_digest=prepared.authorization.record_digest,
            execution_precommit_digest=prepared.precommit.record_digest,
            task_id=task.task_id,
            turn_kind=PROPOSER_TURN_KIND,
            expected_prompt=support_positive_proposer_prompt(request),
            expected_images=tuple(
                zip(SUPPORT_POSITIVE_PRESENTATION_NAMES, support_pngs, strict=True)
            ),
            expected_output_schema=support_positive_proposer_output_schema(request),
            runtime=_turn_runtime(),
            underlying_transport=proposer_transport,
        )
        proposer = propose_support_positive_rubric(
            support_pngs[:6],
            support_pngs[6:],
            request=request,
            model=MODEL,
            reasoning_effort=EFFORT,
            expected_launcher_digest=LAUNCHER_DIGEST,
            cloud_policy_cache_snapshot=None,
            **NO_TOOLS_KWARGS,
            transport=proposer_journal,
        )
        proposer_terminal = PositiveProseJournalTerminal.verify_and_embed(
            proposer_journal,
            artifact_kind="proposer",
            artifact_digest=proposer.artifact_digest,
        )
    else:
        proposer, proposer_terminal = source_pair
        runtime = proposer.runtime

    support_ids = task.side_0_support_panel_ids + task.side_1_support_panel_ids
    query_ids = (task.side_0_query_panel_id, task.side_1_query_panel_id)
    all_specs = [
        (
            PositiveProseEvidencePhase.SUPPORT,
            index,
            PositiveProsePanelRole.PRIMARY_SUPPORT
            if index < 6
            else PositiveProsePanelRole.CONTRAST_SUPPORT,
            panel_id,
            panel,
            support_kinds[index],
        )
        for index, (panel_id, panel) in enumerate(
            zip(support_ids, support_pngs, strict=True)
        )
    ]
    if query_pngs:
        if len(query_pngs) != 2 or len(query_kinds) != 2:
            raise AssertionError("query fixture must be absent or complete")
        all_specs.extend(
            (
                PositiveProseEvidencePhase.QUERY,
                index,
                PositiveProsePanelRole.PRIMARY_QUERY
                if index == 0
                else PositiveProsePanelRole.CONTRAST_QUERY,
                panel_id,
                panel,
                query_kinds[index],
            )
            for index, (panel_id, panel) in enumerate(
                zip(query_ids, query_pngs, strict=True)
            )
        )
    rows = []
    for phase, index, role, panel_id, panel, kind in all_specs:
        context = PositiveProsePanelContext.build(
            panel,
            panel_id=panel_id,
            model=MODEL,
            reasoning_effort=EFFORT,
            expected_launcher_digest=LAUNCHER_DIGEST,
            cloud_policy_cache_snapshot=None,
            **NO_TOOLS_KWARGS,
        )
        panel_request = PositiveProsePanelRequest.build_from_proposer(
            context, proposer, expected_artifact_digest=proposer.artifact_digest
        )
        payload = _interval(kind)

        def panel_transport(prompt, paths, names, schema, _payload=payload, **kwargs):
            return CodexStructuredResult(
                deepcopy(_payload),
                _receipt(prompt, paths, names, schema, _payload),
            )

        monkeypatch.setattr(
            observer_module, "run_codex_named_images_structured", panel_transport
        )
        journal = ObjectBongardNamedImageTurnJournalTransport(
            tmp_path / f"journal-{phase.value}-{index:02d}",
            authorization_digest=prepared.authorization.record_digest,
            execution_precommit_digest=prepared.precommit.record_digest,
            task_id=task.task_id,
            turn_kind=f"positive_prose_{phase.value}_{index:02d}",
            expected_prompt=positive_prose_panel_prompt(panel_request),
            expected_images=(("panel.png", panel),),
            expected_output_schema=positive_prose_panel_output_schema(panel_request),
            runtime=_turn_runtime(),
            underlying_transport=panel_transport,
        )
        artifact = observe_positive_prose_panel(
            panel,
            request=panel_request,
            source_proposer_artifact=proposer,
            expected_source_proposer_artifact_digest=proposer.artifact_digest,
            expected_launcher_digest=LAUNCHER_DIGEST,
            cloud_policy_cache_snapshot=None,
            **NO_TOOLS_KWARGS,
            transport=journal,
        )
        terminal = PositiveProseJournalTerminal.verify_and_embed(
            journal,
            artifact_kind="panel_observer",
            artifact_digest=artifact.artifact_digest,
        )
        rows.append(
            PositiveProseEvidenceRow.create(
                phase=phase,
                phase_index=index,
                role=role,
                panel_id=panel_id,
                panel_png=panel,
                observer_artifact=artifact,
                journal_terminal=terminal,
            )
        )
    return proposer, proposer_terminal, tuple(rows)


def _support_bundle(tmp_path, monkeypatch, prepared, fixture, *, gap=False):
    task = prepared.plan.tasks[0]
    released = {}
    for panel_id in task.side_0_support_panel_ids + task.side_1_support_panel_ids:
        panel, _ = release_panel_feature_extracted_support_panel(
            prepared=prepared, archive=fixture.archive, panel_id=panel_id
        )
        released[panel_id] = panel
    support_ids = task.side_0_support_panel_ids + task.side_1_support_panel_ids
    pngs = tuple(released[item].exact_png_bytes for item in support_ids)
    kinds = (
        ("present",) * 5
        + (("certified_absent",) if gap else ("indeterminate",))
        + ("certified_absent",) * 5
        + ("indeterminate",)
    )
    proposer, terminal, rows = _journaled_artifacts(
        tmp_path,
        monkeypatch,
        prepared=prepared,
        support_pngs=pngs,
        support_kinds=kinds,
    )
    bundle = PositiveProseEvidenceBundle.create(
        task_plan=task,
        authorization_digest=prepared.authorization.record_digest,
        execution_precommit_digest=prepared.precommit.record_digest,
        proposer_artifact=proposer,
        proposer_journal_terminal=terminal,
        rows=rows,
    )
    return bundle, pngs


def test_exact_bundle_cold_replay_and_role_swap_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    prepared = _prepare(fixture)
    bundle, _ = _support_bundle(tmp_path, monkeypatch, prepared, fixture)
    assert cold_replay_positive_prose_evidence_bundle(
        bundle, expected_artifact_address=bundle.artifact_address
    ) == bundle
    assert len(bundle.support_rows) == 12
    assert not bundle.query_rows
    assert bundle.benchmark_sealable is True

    swapped = list(bundle.rows)
    swapped[0], swapped[6] = swapped[6], swapped[0]
    with pytest.raises(PositiveProseEvidenceError, match="IDs, order, or roles"):
        PositiveProseEvidenceBundle.create(
            task_plan=bundle.task_plan,
            authorization_digest=bundle.authorization_digest,
            execution_precommit_digest=bundle.execution_precommit_digest,
            proposer_artifact=bundle.proposer_artifact,
            proposer_journal_terminal=bundle.proposer_journal_terminal,
            rows=swapped,
        )


def test_tolerant_support_admission_is_explicit_and_contradiction_is_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    prepared = _prepare(fixture)
    bundle, _ = _support_bundle(tmp_path / "ok", monkeypatch, prepared, fixture)
    admission = PositiveProseSupportAdmission.derive(
        bundle, expected_bundle_address=bundle.artifact_address
    )
    assert admission.status is PositiveProseSupportStatus.SUPPORT_ADMISSIBLE
    assert admission.primary_counts == {
        "present": 5, "certified_absent": 0, "indeterminate": 1, "error": 0
    }
    assert admission.contrast_counts == {
        "present": 0, "certified_absent": 5, "indeterminate": 1, "error": 0
    }
    assert admission.to_data()["proposer_self_estimates_used_for_admission"] is False

    other_fixture = _fixture(tmp_path / "gap")
    other_prepared = _prepare(other_fixture)
    gap_bundle, _ = _support_bundle(
        tmp_path / "gap", monkeypatch, other_prepared, other_fixture, gap=True
    )
    gap = PositiveProseSupportAdmission.derive(
        gap_bundle, expected_bundle_address=gap_bundle.artifact_address
    )
    assert gap.status is PositiveProseSupportStatus.SUPPORT_GAP
    assert "primary_certified_absent_contradiction" in gap.gap_reasons
    with pytest.raises(PositiveProseTaskRunnerError, match="support gap"):
        PositiveProseTaskFreeze.seal(
            gap, execution_precommit=other_prepared.precommit
        )


def test_exact_durable_freeze_gates_extracted_queries_and_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    prepared = _prepare(fixture)
    support_bundle, support_pngs = _support_bundle(
        tmp_path, monkeypatch, prepared, fixture
    )
    admission = PositiveProseSupportAdmission.derive(
        support_bundle, expected_bundle_address=support_bundle.artifact_address
    )
    freeze = PositiveProseTaskFreeze.seal(
        admission, execution_precommit=prepared.precommit
    )
    assert verify_positive_prose_task_freeze(
        freeze, expected_record_digest=freeze.record_digest
    ) == freeze
    freeze_receipt = persist_object_bongard_task_freeze(
        store=prepared.store, freeze=freeze
    )
    commit = PositiveProseTaskFreezeCommit.seal(freeze, freeze_receipt)
    commit_receipt = persist_object_bongard_task_commit(
        store=prepared.store, commit=commit
    )
    assert verify_positive_prose_task_commit(
        commit,
        expected_record_digest=commit.record_digest,
        task_commit_store_receipt=commit_receipt,
    ) == commit

    task = prepared.plan.tasks[0]
    released_queries = []
    query_receipts = []
    for panel_id in (task.side_0_query_panel_id, task.side_1_query_panel_id):
        released, receipt = release_panel_feature_extracted_query_panel(
            prepared=prepared,
            archive=fixture.archive,
            panel_id=panel_id,
            task_freeze=freeze,
            task_commit=commit,
            task_freeze_receipt=freeze_receipt,
            task_commit_receipt=commit_receipt,
        )
        released_queries.append(released)
        query_receipts.append(receipt)

    _proposer, _proposer_terminal, rows = _journaled_artifacts(
        tmp_path / "full",
        monkeypatch,
        prepared=prepared,
        support_pngs=support_pngs,
        support_kinds=("present",) * 5 + ("indeterminate",)
        + ("certified_absent",) * 5 + ("indeterminate",),
        query_pngs=tuple(item.exact_png_bytes for item in released_queries),
        query_kinds=("present", "certified_absent"),
        source_pair=(
            support_bundle.proposer_artifact,
            support_bundle.proposer_journal_terminal,
        ),
    )
    # The independently reconstructed support calls are byte-identical but have
    # different journal identities. A query extension must retain the exact
    # frozen support artifacts, so replace them with the frozen rows.
    full_bundle = PositiveProseEvidenceBundle.create(
        task_plan=task,
        authorization_digest=prepared.authorization.record_digest,
        execution_precommit_digest=prepared.precommit.record_digest,
        proposer_artifact=support_bundle.proposer_artifact,
        proposer_journal_terminal=support_bundle.proposer_journal_terminal,
        rows=(*support_bundle.support_rows, *rows[12:]),
    )
    outcomes = []
    for released, receipt in zip(released_queries, query_receipts, strict=True):
        decision = PositiveProseQueryDecision.create(
            freeze,
            query_evidence_bundle=full_bundle,
            released_query_panel=released,
            query_release_store_receipt=receipt,
        )
        assert cold_replay_positive_prose_query_decision(
            decision, freeze=freeze, expected_artifact_address=decision.artifact_address
        ) == decision
        outcomes.append(decision.outcome)
    assert outcomes == [
        PositiveProseQueryOutcome.POSITIVE,
        PositiveProseQueryOutcome.NEGATIVE,
    ]
    assert freeze.selected_predicate.to_data()["decision_mapping"] == {
        "present": "positive",
        "certified_absent": "negative",
        "indeterminate": "abstain",
        "error": "error",
    }

"""Focused extracted-custody and no-narrator v2 runner tests."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import pytest

from bongard.object_bongard_release_gate import (
    ObjectBongardReleaseStore,
    persist_object_bongard_task_commit,
    persist_object_bongard_task_freeze,
)
from bongard.panel_action_count_cnn_typed_axis_adapter import (
    PopulationScope,
    build_cnn_typed_support_matrix,
)
from bongard.panel_feature_extracted_release_gate import (
    release_panel_feature_extracted_query_panel,
    release_panel_feature_extracted_support_panel,
)
from bongard.panel_typed_axis_custody_v2 import (
    TaskBoundTypedAxisSupportArtifact,
    TypedAxisCustodyV2Error,
    TypedAxisQueryObservationArtifact,
    cold_replay_task_bound_typed_axis_support,
)
from bongard.panel_typed_axis_slate_v2 import SupportSide, TypedAxisInventory
from bongard.panel_typed_axis_task_runner_v2 import (
    TypedAxisFormulaFreezeV2,
    TypedAxisTaskGapV2,
    TypedAxisTaskRunnerV2Error,
    build_typed_axis_rank_journal_v2,
    cold_replay_typed_axis_task_result_v2,
    run_typed_axis_formula_task_v2,
)
from bongard.tests import test_panel_feature_extracted_release_gate as gate_fixture
from bongard.tests.test_panel_action_count_cnn_typed_axis_adapter import (
    _batch,
    _catalog,
    _class_set,
    _grant,
    _peaked,
    _protocol,
)
from bongard.tests.test_panel_positive_formula_ranker import _text_receipt
from bongard.tests.test_panel_typed_axis_headless_proposer import _runtime
from bongard.transport import CodexStructuredResult


V2_SEED = "typed-axis-v2-70"


def _prepared(tmp_path: Path):
    old = gate_fixture.SEED
    gate_fixture.SEED = V2_SEED
    try:
        fixture = gate_fixture._fixture(tmp_path)
        prepared = gate_fixture._prepare(fixture)
    finally:
        gate_fixture.SEED = old
    task = prepared.plan.tasks[0]
    assert task.side_0_query_panel_id.endswith("/1/4.png")
    assert task.side_1_query_panel_id.endswith("/0/4.png")
    return fixture, prepared


def _probability_rows(rows, kind: str):
    result = []
    for row in rows:
        if row.side is SupportSide.PRIMARY:
            result.append(row)
            continue
        if kind == "unique":
            result.append(row)
            continue
        if kind == "zero":
            straight, catalog = 4, "convex"
        elif kind == "multi":
            straight, catalog = 3, "nonconvex"
        else:
            raise AssertionError(kind)
        straight_p = _peaked(10, straight)
        catalog_p = _catalog(catalog)
        result.append(
            replace(
                row,
                straight_logits=tuple(math.log(value) for value in straight_p),
                straight_probabilities=straight_p,
                straight_class_set=_class_set(straight_p, 0.6),
                catalog_logits=tuple(math.log(value) for value in catalog_p),
                catalog_probabilities=catalog_p,
                catalog_class_set=_class_set(catalog_p, 0.6),
            )
        )
    return tuple(result)


def _custody(tmp_path: Path, kind: str):
    fixture, prepared = _prepared(tmp_path)
    task = prepared.plan.tasks[0]
    expected_ids = tuple(
        f"{task.family}/{task.task_id}/1/{item}.png" for item in (0, 1, 2, 3, 5, 6)
    ) + tuple(
        f"{task.family}/{task.task_id}/0/{item}.png" for item in (0, 1, 2, 3, 5, 6)
    )
    released = []
    receipts = []
    for panel_id in expected_ids:
        panel, receipt = release_panel_feature_extracted_support_panel(
            prepared=prepared, archive=fixture.archive, panel_id=panel_id
        )
        released.append(panel)
        receipts.append(receipt)

    protocol = _protocol()
    grant = _grant(
        protocol,
        scope=PopulationScope.GENERIC_FRESH_V3,
        authorized_task_ids=(task.task_id,),
    )
    batch = _batch(protocol, grant, task_id=task.task_id)
    rows = _probability_rows(batch.rows, kind)
    rows = tuple(
        replace(
            row,
            png_sha256=panel.exact_png_digest,
            png_size_bytes=len(panel.exact_png_bytes),
        )
        for row, panel in zip(rows, released, strict=True)
    )
    batch = replace(
        batch,
        target_authorization_record_digest=prepared.authorization.record_digest,
        rows=rows,
    )
    matrix = build_cnn_typed_support_matrix(
        protocol=protocol, population_grant=grant, prediction_batch=batch
    )
    custody = TaskBoundTypedAxisSupportArtifact.create(
        task_plan=task,
        execution_precommit=prepared.precommit,
        release_authorization=prepared.authorization,
        released_support_panels=released,
        released_support_store_receipts=receipts,
        observer_matrix_artifact=matrix,
    )
    return fixture, prepared, custody


def test_support_custody_pins_folder_polarity_bytes_receipts_and_is_unsealable(
    tmp_path: Path,
) -> None:
    fixture, prepared, custody = _custody(tmp_path, "unique")
    expected_primary = tuple(
        f"{custody.task_plan.family}/{custody.task_id}/1/{item}.png"
        for item in (0, 1, 2, 3, 5, 6)
    )
    expected_contrast = tuple(
        f"{custody.task_plan.family}/{custody.task_id}/0/{item}.png"
        for item in (0, 1, 2, 3, 5, 6)
    )
    data = custody.to_data()
    assert data["observer_inference_externally_authenticated"] is False
    assert data["benchmark_sealable"] is False
    assert data["query_release_authorized"] is False
    assert custody.sealed_query_panel_ids == (
        custody.task_plan.side_0_query_panel_id,
        custody.task_plan.side_1_query_panel_id,
    )
    assert custody.released_support_panels[0].panel_id.endswith("/1/0.png")
    assert custody.released_support_panels[6].panel_id.endswith("/0/0.png")
    assert custody.task_plan.side_0_support_panel_ids == expected_primary
    assert custody.task_plan.side_1_support_panel_ids == expected_contrast
    assert tuple(item.panel_id for item in custody.released_support_panels) == (
        expected_primary + expected_contrast
    )
    assert cold_replay_task_bound_typed_axis_support(
        custody,
        store=prepared.store,
        archive=fixture.archive,
        expected_artifact_address=custody.record_digest,
    ) == custody

    swapped = list(custody.released_support_panels)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(TypedAxisCustodyV2Error):
        replace(custody, released_support_panels=tuple(swapped))

    # Preserve the same twelve-ID union while reversing its typed polarity.
    # The custody join must reject this even if an upstream object were forged
    # around ObjectBongardTaskPlan's own constructor checks.
    swapped_task = object.__new__(type(custody.task_plan))
    for field in custody.task_plan.__dataclass_fields__:
        value = getattr(custody.task_plan, field)
        if field == "side_0_support_panel_ids":
            value = custody.task_plan.side_1_support_panel_ids
        elif field == "side_1_support_panel_ids":
            value = custody.task_plan.side_0_support_panel_ids
        object.__setattr__(swapped_task, field, value)
    assert set(
        swapped_task.side_0_support_panel_ids
        + swapped_task.side_1_support_panel_ids
    ) == set(expected_primary + expected_contrast)
    with pytest.raises(TypedAxisCustodyV2Error, match="folder roles"):
        replace(custody, task_plan=swapped_task)


def test_both_query_folders_map_to_the_pinned_typed_sides(tmp_path: Path) -> None:
    fixture, prepared, custody = _custody(tmp_path, "unique")
    legacy_freeze = gate_fixture._freeze(prepared, fixture)
    freeze_receipt = persist_object_bongard_task_freeze(
        store=prepared.store, freeze=legacy_freeze
    )
    legacy_commit = gate_fixture._commit(legacy_freeze, freeze_receipt)
    commit_receipt = persist_object_bongard_task_commit(
        store=prepared.store, commit=legacy_commit
    )
    cases = (
        (custody.task_plan.side_0_query_panel_id, SupportSide.PRIMARY, custody.observer_matrix_artifact.prediction_batch.rows[0]),
        (custody.task_plan.side_1_query_panel_id, SupportSide.CONTRAST, custody.observer_matrix_artifact.prediction_batch.rows[6]),
    )
    for panel_id, side, template in cases:
        released, receipt = release_panel_feature_extracted_query_panel(
            prepared=prepared,
            archive=fixture.archive,
            panel_id=panel_id,
            task_freeze=legacy_freeze,
            task_commit=legacy_commit,
            task_freeze_receipt=freeze_receipt,
            task_commit_receipt=commit_receipt,
        )
        prediction = replace(
            template,
            panel_id=panel_id,
            side=side,
            ordinal=4,
            png_sha256=released.exact_png_digest,
            png_size_bytes=len(released.exact_png_bytes),
        )
        artifact = TypedAxisQueryObservationArtifact.create(
            support_custody=custody,
            formula_commit_address=legacy_commit.record_digest,
            released_query_panel=released,
            released_query_store_receipt=receipt,
            prediction=prediction,
        )
        assert artifact.prediction.side is side


def test_zero_and_unique_derive_before_rank_and_make_zero_model_calls(
    tmp_path: Path,
) -> None:
    _fixture0, _prepared0, zero = _custody(tmp_path / "zero", "zero")
    gap = run_typed_axis_formula_task_v2(zero)
    assert type(gap) is TypedAxisTaskGapV2
    assert gap.to_data()["rank_model_calls"] == 0
    assert gap.to_data()["narrator_model_calls"] == 0
    assert cold_replay_typed_axis_task_result_v2(
        gap, expected_artifact_address=gap.record_digest
    ) == gap

    _fixture1, _prepared1, unique = _custody(tmp_path / "unique", "unique")
    freeze = run_typed_axis_formula_task_v2(unique)
    assert type(freeze) is TypedAxisFormulaFreezeV2
    assert freeze.rank_artifact is None
    assert freeze.selection_mode == "unique_survivor_zero_model_calls"
    assert freeze.to_data()["narrator_model_calls"] == 0
    assert freeze.to_data()["observer_inference_externally_authenticated"] is False
    assert cold_replay_typed_axis_task_result_v2(
        freeze, expected_artifact_address=freeze.record_digest
    ) == freeze


def test_multi_uses_one_exactly_once_text_journal_and_cold_replay_is_zero_call(
    tmp_path: Path,
) -> None:
    _fixture, _prepared_release, support = _custody(tmp_path / "multi", "multi")
    inventory = TypedAxisInventory.derive(support.matrix)
    assert len(inventory.admitted_formula_ids) > 1
    runtime = _runtime()
    physical_calls = 0

    def transport(prompt, schema, **_kwargs):
        nonlocal physical_calls
        physical_calls += 1
        aliases = tuple(reversed(schema["properties"]["ordered_aliases"]["items"]["enum"]))
        payload = {"ordered_aliases": list(aliases)}
        return CodexStructuredResult(payload, _text_receipt(prompt, schema, payload))

    journal = build_typed_axis_rank_journal_v2(
        tmp_path / "rank-journal",
        support=support,
        inventory=inventory,
        runtime=runtime,
        underlying_transport=transport,
    )
    freeze = run_typed_axis_formula_task_v2(
        support, rank_runtime=runtime, rank_journal=journal
    )
    assert type(freeze) is TypedAxisFormulaFreezeV2
    assert freeze.rank_artifact is not None
    assert physical_calls == 1
    assert journal.fresh_call_count == 1
    assert freeze.to_data()["rank_model_calls"] == 1
    assert freeze.to_data()["narrator_model_calls"] == 0
    assert cold_replay_typed_axis_task_result_v2(
        freeze,
        expected_artifact_address=freeze.record_digest,
        rank_journal=journal,
    ) == freeze
    assert physical_calls == 1
    assert journal.reused_call_count == 0

    with pytest.raises(TypedAxisTaskRunnerV2Error, match="unique survivor"):
        _fixture2, _prepared2, unique = _custody(tmp_path / "misuse", "unique")
        run_typed_axis_formula_task_v2(
            unique, rank_runtime=runtime, rank_journal=journal
        )

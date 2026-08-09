from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO

from PIL import Image, ImageDraw
import pytest

from bongard import transport as transport_runtime
from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_scene_anchor_batch_observer import (
    OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS,
    ObjectSceneAnchorBatchObserverArtifact,
    _expected_records,
    object_scene_anchor_batch_observer_prompt,
    observe_object_scene_anchor_batches_twice,
    verify_object_scene_anchor_batch_observer_artifact,
)
from bongard.object_scene_anchor_bindings import (
    ObjectSceneAnchorBindingSpec,
    build_object_scene_anchor_binding_catalog,
)
from bongard.object_scene_anchor_cards import (
    ObjectSceneAnchorCardWitness,
    _binding_catalogs_digest,
)
from bongard.object_scene_anchor_observer import (
    freeze_object_scene_anchor_observer_vocabulary,
)
from bongard.object_scene_anchor_support_observation_join import (
    ObjectSceneAnchorSupportObservationJoinError,
    ObjectSceneAnchorSupportObservationPlan,
    ObjectSceneAnchorSupportObservationResult,
    build_object_scene_anchor_support_observation_plan,
    cold_verify_object_scene_anchor_support_observation_result,
    finalize_object_scene_anchor_support_observations,
    verify_object_scene_anchor_support_observation_runtime,
)
from bongard.object_scene_anchor_support_preparation import (
    ObjectSceneAnchorSupportCorpusRuntimeBundle,
    ObjectSceneAnchorSupportPanelInput,
    build_object_scene_anchor_support_panel,
    freeze_object_scene_anchor_support_corpus,
)
from bongard.object_scene_anchor_version_space import (
    ObjectSceneAnchorAtomCitation,
    ObjectSceneAnchorOrientation,
    ObjectSceneAnchorPredicateAtom,
    ObjectSceneAnchorPredicateLanguage,
    enumerate_object_scene_anchor_candidates,
)
from bongard.tests.no_tools_fixture import (
    canonical_codex_receipt,
    canonical_no_tools_runtime,
)
from bongard.transport import CodexReceipt, CodexStructuredResult


LAUNCHER_DIGEST = "a" * 64
MODEL = transport_runtime.DEFAULT_CODEX_MODEL
EFFORT = "medium"


def _panel_png(index: int) -> bytes:
    image = Image.new("RGB", (64 + index, 64), "white")
    draw = ImageDraw.Draw(image)
    if index < 6:
        draw.line((16, 46, 32, 12, 48, 46, 16, 46), fill="black", width=3)
    else:
        draw.line((32, 12, 32, 50), fill="black", width=3)
        draw.line((13, 31, 51, 31), fill="black", width=3)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _support_corpus() -> ObjectSceneAnchorSupportCorpusRuntimeBundle:
    source_digest = "1" * 64
    panels = []
    for index in range(12):
        payload = _panel_png(index)
        panel_input = ObjectSceneAnchorSupportPanelInput(
            panel_alias=f"panel_{index:03d}",
            support_bucket_index=0 if index < 6 else 1,
            source_digest=source_digest,
            source_panel_binding_digest=hashlib.sha256(
                f"source-binding-{index}".encode("ascii")
            ).hexdigest(),
            source_ordinal=index,
            task_id=f"synthetic-task-{index}",
            panel_id=f"synthetic-task-{index}/support/0",
            original_panel_png_digest=hashlib.sha256(payload).hexdigest(),
            exact_original_png_bytes=payload,
        )
        panels.append(build_object_scene_anchor_support_panel(panel_input))
    frozen_panels = tuple(item.freeze for item in panels)
    return ObjectSceneAnchorSupportCorpusRuntimeBundle(
        freeze=freeze_object_scene_anchor_support_corpus(
            source_digest, frozen_panels
        ),
        panels=tuple(panels),
    )


def _citations(corpus, spec, panel_indices):
    result = []
    for index in panel_indices:
        panel = corpus.freeze.panels[index]
        manifest = panel.panel_manifest
        catalogs = tuple(
            build_object_scene_anchor_binding_catalog(
                decision, spec, expected_object_id=object_id
            )
            for object_id, decision in zip(
                manifest.object_ids,
                manifest.object_decisions,
                strict=True,
            )
        )
        assert len(catalogs) == 1
        assert catalogs[0].hard_disposition is Disposition.PRESENT
        assert len(catalogs[0].bindings) == 1
        result.append(
            ObjectSceneAnchorAtomCitation.create(
                panel.panel_alias,
                manifest.manifest_digest,
                _binding_catalogs_digest(manifest, spec, catalogs),
                catalogs[0].bindings[0],
            )
        )
    return tuple(result)


def _language(
    corpus: ObjectSceneAnchorSupportCorpusRuntimeBundle,
) -> ObjectSceneAnchorPredicateLanguage:
    witnesses = (
        ObjectSceneAnchorCardWitness.create(
            "witness_00",
            "shape_appearance",
            "the highlighted loop has three oblique boundary segments",
        ),
        ObjectSceneAnchorCardWitness.create(
            "witness_01",
            "shape_appearance",
            "the bound figure is visibly cross-like",
        ),
    )
    vocabulary = freeze_object_scene_anchor_observer_vocabulary(witnesses)
    by_statement = {item.statement: item for item in vocabulary.entries}
    frame_spec = ObjectSceneAnchorBindingSpec.frame(3, 3)
    entity_spec = ObjectSceneAnchorBindingSpec.entity()
    frame_atom = ObjectSceneAnchorPredicateAtom.create(
        source_card_digest=hashlib.sha256(b"frame-card").hexdigest(),
        orientation=ObjectSceneAnchorOrientation.SIDE0_POSITIVE,
        binding_spec=frame_spec,
        witness_digests=(
            by_statement[
                "the highlighted loop has three oblique boundary segments"
            ].witness_digest,
        ),
        positive_support_citations=_citations(corpus, frame_spec, range(6)),
    )
    entity_atom = ObjectSceneAnchorPredicateAtom.create(
        source_card_digest=hashlib.sha256(b"entity-card").hexdigest(),
        orientation=ObjectSceneAnchorOrientation.SIDE1_POSITIVE,
        binding_spec=entity_spec,
        witness_digests=(
            by_statement[
                "the bound figure is visibly cross-like"
            ].witness_digest,
        ),
        positive_support_citations=_citations(
            corpus, entity_spec, range(6, 12)
        ),
    )
    return ObjectSceneAnchorPredicateLanguage.create(
        source_proposal_digest=hashlib.sha256(b"proposal").hexdigest(),
        vocabulary=vocabulary,
        atoms=(frame_atom, entity_atom),
    )


def _payload(batch, vocabulary) -> dict[str, object]:
    return {
        "cells": [
            {
                "subject_id": subject.subject_alias,
                "catalog_id": catalog.catalog_alias,
                "binding_id": binding.binding_id,
                "witness_id": witness.witness_id,
                "state": "P",
                "reason_code": "visible_match",
            }
            for subject, catalog, binding, _locator, witness in _expected_records(
                batch, vocabulary
            )
        ]
    }


def _unique_receipt(receipt: CodexReceipt, index: int) -> CodexReceipt:
    data = receipt.to_dict()
    data["thread_id"] = f"00000000-0000-4000-8000-{index + 1:012d}"
    data["event_stream_digest"] = f"{index + 1:x}" * 64
    unsigned = {key: item for key, item in data.items() if key != "receipt_digest"}
    data["receipt_digest"] = transport_runtime._digest(unsigned)
    transport_runtime.validate_codex_receipt(data)
    return CodexReceipt(
        **{
            **data,
            "event_types": tuple(data["event_types"]),
            "item_types": tuple(data["item_types"]),
        }
    )


def _artifact(runtime) -> tuple[ObjectSceneAnchorBatchObserverArtifact, int]:
    calls = 0

    def transport(prompt, paths, names, schema, **_kwargs):
        nonlocal calls
        batch = runtime.plan.batch_plan.batches[calls // 2]
        payload = _payload(batch, runtime.plan.language.vocabulary)
        receipt = canonical_codex_receipt(
            prompt,
            paths,
            schema,
            payload,
            launcher_digest=LAUNCHER_DIGEST,
            reasoning_effort=EFFORT,
            model=MODEL,
            names=names,
            command_fixture=f"support join call {calls}",
        )
        result = CodexStructuredResult(payload, _unique_receipt(receipt, calls))
        calls += 1
        return result

    model_catalog, attestation = canonical_no_tools_runtime(LAUNCHER_DIGEST)
    artifact = observe_object_scene_anchor_batches_twice(
        runtime.batch_inputs,
        plan=runtime.plan.batch_plan,
        expected_plan_digest=runtime.plan.batch_plan_digest,
        observation_plan_digest=runtime.plan.observation_context_digest,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        model_catalog_snapshot=model_catalog,
        no_tools_attestation=attestation,
        transport=transport,
    )
    return artifact, calls


def _contains_bytes(value: object) -> bool:
    if isinstance(value, bytes):
        return True
    if isinstance(value, dict):
        return any(_contains_bytes(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_bytes(item) for item in value)
    return False


def _historical_checker_values(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            if "l" + "ean" in key.casefold():
                yield key, item
            yield from _historical_checker_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _historical_checker_values(item)


@pytest.fixture(scope="module")
def planned():
    corpus = _support_corpus()
    language = _language(corpus)
    runtime = build_object_scene_anchor_support_observation_plan(
        corpus, language
    )
    return corpus, language, runtime


@pytest.fixture(scope="module")
def observed(planned):
    _corpus, _language_value, runtime = planned
    artifact, calls = _artifact(runtime)
    result = finalize_object_scene_anchor_support_observations(
        runtime.plan, artifact
    )
    return artifact, calls, result


def test_plan_freezes_every_catalog_present_only_batches_and_exact_cells(
    planned,
) -> None:
    corpus, language, runtime = planned
    plan = runtime.plan

    assert ObjectSceneAnchorSupportObservationPlan.from_data(plan.to_data()) == plan
    assert (
        verify_object_scene_anchor_support_observation_runtime(
            runtime,
            corpus_runtime=corpus,
            language=language,
            expected_plan_digest=plan.plan_digest,
        )
        == runtime
    )
    assert plan.panel_count == 12
    assert plan.binding_spec_count == 2
    assert plan.catalog_count == 24
    assert plan.present_catalog_count == 18
    assert plan.nonpresent_catalog_count == 6
    assert plan.rendered_view_count == 12
    assert len(runtime.batch_inputs) == 18
    assert plan.batch_cell_counts == (36,)
    assert plan.total_present_cell_count == 36
    assert plan.maximum_batch_cell_count == 36
    assert plan.maximum_cells_per_batch == OBJECT_SCENE_ANCHOR_BATCH_MAX_CELLS
    assert plan.maximum_batch_cell_count <= plan.maximum_cells_per_batch
    assert plan.total_present_cell_count == plan.batch_plan.cell_count

    assert tuple(item.panel_index for item in plan.catalog_records) == tuple(
        index for index in range(12) for _spec in range(2)
    )
    assert tuple(item.spec_index for item in plan.catalog_records) == (0, 1) * 12
    assert all(item.object_index == 0 for item in plan.catalog_records)
    assert all(
        (item.preparation_digest is not None)
        == (item.catalog.hard_disposition is Disposition.PRESENT)
        for item in plan.catalog_records
    )
    assert all(
        preparation.catalog.hard_disposition is Disposition.PRESENT
        for preparation in plan.batch_plan.preparations
    )
    assert sorted(
        len(subject.catalogs)
        for batch in plan.batch_plan.batches
        for subject in batch.subjects
    ) == [1] * 6 + [2] * 6

    persistent = plan.to_data()
    assert not _contains_bytes(persistent)
    historical = tuple(_historical_checker_values(persistent))
    assert historical
    assert all(
        item is (True if "removable" in key.casefold() else False)
        for key, item in historical
    )
    model_boundary = str(plan.batch_plan.to_data()).casefold()
    assert "support_bucket_index" not in model_boundary
    assert "synthetic-task" not in model_boundary
    prompt = object_scene_anchor_batch_observer_prompt(
        plan.batch_plan.batches[0], plan.language.vocabulary
    ).casefold()
    for forbidden in ("bucket", "side0", "side1", "positive", "negative"):
        assert forbidden not in prompt


def test_finalize_joins_observed_rows_and_hard_zero_rows_into_both_spaces(
    planned, observed
) -> None:
    _corpus, language, runtime = planned
    artifact, calls, result = observed

    assert calls == artifact.physical_call_count == 2
    assert (
        verify_object_scene_anchor_batch_observer_artifact(
            artifact,
            runtime.batch_inputs,
            expected_artifact_digest=artifact.artifact_digest,
            expected_plan_digest=runtime.plan.batch_plan_digest,
            expected_observation_plan_digest=(
                runtime.plan.observation_context_digest
            ),
        )
        == artifact
    )
    assert ObjectSceneAnchorSupportObservationResult.from_data(
        result.to_data()
    ) == result
    assert tuple(item.panel_id for item in result.panel_evaluations) == tuple(
        f"panel_{index:03d}" for index in range(12)
    )

    for evaluation in result.panel_evaluations[6:]:
        frame = next(
            item
            for item in evaluation.spec_matrices
            if item.binding_spec.anchor_kind == "frame"
        )
        assert frame.objects[0].catalog.hard_disposition is (
            Disposition.CERTIFIED_ABSENT
        )
        assert frame.objects[0].rows == ()

    forward = result.bucket0_positive_version_space
    inverse = result.bucket1_positive_version_space
    assert forward.orientation is ObjectSceneAnchorOrientation.SIDE0_POSITIVE
    assert inverse.orientation is ObjectSceneAnchorOrientation.SIDE1_POSITIVE
    assert forward.support_panel_ids == tuple(
        f"panel_{index:03d}" for index in range(12)
    )
    assert inverse.support_panel_ids == tuple(
        f"panel_{index:03d}" for index in (*range(6, 12), *range(6))
    )
    forward_candidates = enumerate_object_scene_anchor_candidates(
        language, ObjectSceneAnchorOrientation.SIDE0_POSITIVE
    )
    assert forward.survivor_candidate_digests == (
        forward_candidates[0].candidate_digest,
    )
    assert inverse.survivor_candidate_digests == ()
    assert (
        cold_verify_object_scene_anchor_support_observation_result(
            result, plan=runtime.plan, artifact=artifact
        )
        == result
    )


def test_plan_count_tamper_and_artifact_context_substitution_fail(
    planned, observed
) -> None:
    _corpus, _language_value, runtime = planned
    artifact, _calls, _result = observed
    damaged = deepcopy(runtime.plan.to_data())
    damaged["batch_cell_counts"][0] += 1
    damaged["plan_digest"] = canonical_digest(
        {key: item for key, item in damaged.items() if key != "plan_digest"}
    )
    with pytest.raises(
        ObjectSceneAnchorSupportObservationJoinError,
        match="counts or commitments",
    ):
        ObjectSceneAnchorSupportObservationPlan.from_data(damaged)

    other_plan = deepcopy(runtime.plan.to_data())
    other_plan["observation_context_digest"] = "sha256:" + "f" * 64
    other_plan["plan_digest"] = canonical_digest(
        {key: item for key, item in other_plan.items() if key != "plan_digest"}
    )
    with pytest.raises(ObjectSceneAnchorSupportObservationJoinError):
        ObjectSceneAnchorSupportObservationPlan.from_data(other_plan)
    assert artifact.observation_plan_digest == (
        runtime.plan.observation_context_digest
    )

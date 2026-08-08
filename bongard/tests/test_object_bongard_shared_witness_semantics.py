"""Offline tests for structured shared-witness proposal and rubric identity."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from bongard.object_bongard_shared_witness import (
    ObjectBongardSharedWitnessContrast,
    ObjectBongardSharedWitnessError,
    ObjectBongardSharedWitnessRubricSpec,
    build_shared_witness_rubric_specs,
)
from bongard.object_bongard_shared_witness_semantics import (
    GROUP_SIZE,
    ObjectBongardSharedWitnessSemanticArtifact,
    ObjectBongardSharedWitnessSemanticsError,
    SHARED_WITNESS_SEMANTIC_PROTOCOL_ID,
    describe_object_bongard_shared_witness_support,
    object_bongard_shared_witness_semantics_output_schema,
    object_bongard_shared_witness_semantics_prompt,
    verify_object_bongard_shared_witness_semantic_artifact,
)
from bongard.prototype_scene_observer import PrototypeSceneObserverStatus
from bongard.tests.test_prototype_scene_observer import (
    CONTEXT_DIGEST,
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _png,
    _receipt,
)
from bongard.transport import CodexStructuredResult


TASK_ID = "ff_nact2_5_0042"
GROUP_0 = tuple(f"ff/{TASK_ID}/1/{index}.png" for index in range(GROUP_SIZE))
GROUP_1 = tuple(f"ff/{TASK_ID}/0/{index}.png" for index in range(GROUP_SIZE))


def _images() -> dict[str, bytes]:
    return {
        panel_id: _png(index)
        for index, panel_id in enumerate((*GROUP_0, *GROUP_1))
    }


def _payload() -> dict[str, object]:
    return {
        "proposal_0": {
            "shared_anchor": "decorated figure",
            "visual_axis": "closed loop topology",
            "group_0_endpoint": "two loops touching at one vertex",
            "group_1_endpoint": "single loop with dangling branch",
        },
        "proposal_1": {
            "shared_anchor": "central outlined figure",
            "visual_axis": "junction angle profile",
            "group_0_endpoint": "four oblique rays meeting centrally",
            "group_1_endpoint": "three acute rays meeting centrally",
        },
    }


def _describe(payload: dict[str, object] | None = None):
    chosen = _payload() if payload is None else payload
    calls = 0

    def transport(prompt, paths, names, schema, **_kwargs):
        nonlocal calls
        calls += 1
        assert len(paths) == len(names) == 12
        assert names[0] == "group_0_ref_00.png"
        assert names[-1] == "group_1_ref_05.png"
        assert TASK_ID not in prompt
        assert schema == object_bongard_shared_witness_semantics_output_schema()
        return CodexStructuredResult(
            chosen, _receipt(prompt, paths, names, schema, chosen)
        )

    artifact = describe_object_bongard_shared_witness_support(
        task_id=TASK_ID,
        group_0_panel_ids=GROUP_0,
        group_1_panel_ids=GROUP_1,
        support_png_by_panel_id=_images(),
        observation_context_digest=CONTEXT_DIGEST,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=transport,
    )
    return artifact, calls


def test_proposer_persists_ir_and_python_rendered_same_witness_cues() -> None:
    artifact, calls = _describe()
    assert calls == 1
    assert artifact.status is PrototypeSceneObserverStatus.SUCCESS
    first, second = artifact.contrast_candidates
    assert (first.candidate_rank, second.candidate_rank) == (0, 1)
    assert first.shared_anchor == "decorated figure"
    assert first.visual_axis == "closed loop topology"
    common = (
        "The inventoried individual decorated figure is this witness; "
        "its closed loop topology appears "
    )
    assert first.rendered_group_0_cue.text == (
        common + "two loops touching at one vertex."
    )
    assert first.rendered_group_1_cue.text == (
        common + "single loop with dangling branch."
    )
    assert artifact.model_payload == _payload()
    assert artifact.soft_cue_candidates == (
        first.soft_cue_pair,
        second.soft_cue_pair,
    )
    data = artifact.to_data()
    assert data["independent_free_form_group_cues_representable"] is False
    assert data["observer_must_persist_individual_witness_evidence"] is True
    assert data["direct_comparative_score_is_sufficient_evidence"] is False
    assert ObjectBongardSharedWitnessSemanticArtifact.from_data(
        data, expected_artifact_digest=artifact.artifact_digest
    ) == artifact
    assert verify_object_bongard_shared_witness_semantic_artifact(
        artifact,
        support_png_by_panel_id=_images(),
        expected_task_id=TASK_ID,
        expected_observation_context_digest=CONTEXT_DIGEST,
        expected_artifact_digest=artifact.artifact_digest,
    ) is artifact


def test_v2_rubric_specs_retain_full_ir_without_downcasting() -> None:
    artifact, _ = _describe()
    specs = build_shared_witness_rubric_specs(
        artifact, expected_artifact_digest=artifact.artifact_digest
    )
    assert all(
        isinstance(item, ObjectBongardSharedWitnessRubricSpec) for item in specs
    )
    first = specs[0]
    assert first.contrast == artifact.contrast_candidates[0]
    assert first.target_cue == first.contrast.rendered_group_0_cue
    assert first.foil_cue == first.contrast.rendered_group_1_cue
    assert "same individual's closed loop topology" in first.rubric
    assert first.to_data()["observer_must_persist_witness_locator"] is True
    assert (
        first.to_data()["observer_must_inventory_all_top_level_anchor_instances"]
        is True
    )
    assert first.to_data()["observer_may_select_one_favorable_witness"] is False
    assert first.to_data()["observer_must_score_endpoints_separately"] is True
    assert (
        first.to_data()["direct_comparative_judgment_is_canonical_evidence"]
        is False
    )
    assert ObjectBongardSharedWitnessRubricSpec.from_data(first.to_data()) == first


def test_schema_makes_independent_legacy_cue_pairs_unrepresentable() -> None:
    schema = object_bongard_shared_witness_semantics_output_schema()
    proposal = schema["properties"]["proposal_0"]  # type: ignore[index]
    assert set(proposal["properties"]) == {  # type: ignore[index]
        "shared_anchor",
        "visual_axis",
        "group_0_endpoint",
        "group_1_endpoint",
    }
    assert "group_0_cue_text" not in proposal["properties"]  # type: ignore[index]
    old_pair = {
        "proposal_0": {
            "group_0_cue_text": "One undecorated figure appears",
            "group_1_cue_text": "Outlined triangular beads decorate a figure",
        },
        "proposal_1": _payload()["proposal_1"],
    }
    artifact, calls = _describe(old_pair)
    assert calls == 1
    assert artifact.status is PrototypeSceneObserverStatus.PARSER_ERROR
    assert artifact.contrast_candidates == ()


def test_old_rank_one_absence_endpoint_and_coexisting_full_cue_fail_closed() -> None:
    absence = _payload()
    absence["proposal_1"] = {
        "shared_anchor": "outlined figure",
        "visual_axis": "surface decoration",
        "group_0_endpoint": "undecorated figure appears",
        "group_1_endpoint": "outlined triangular beads",
    }
    failed, _ = _describe(absence)
    assert failed.status is PrototypeSceneObserverStatus.PARSER_ERROR

    bundled = _payload()
    bundled["proposal_1"] = {
        "shared_anchor": "outlined figure",
        "visual_axis": "surface decoration",
        "group_0_endpoint": "rounded contour and smooth surface",
        "group_1_endpoint": "triangular beads and acute junction",
    }
    failed, _ = _describe(bundled)
    assert failed.status is PrototypeSceneObserverStatus.PARSER_ERROR


def test_rank_one_requires_a_genuinely_different_anchor_or_axis() -> None:
    repeated_axis = _payload()
    repeated_axis["proposal_1"] = {
        "shared_anchor": "decorated figure",
        "visual_axis": "closed loop topology",
        "group_0_endpoint": "paired loops sharing one junction",
        "group_1_endpoint": "single loop bearing one branch",
    }
    failed, _ = _describe(repeated_axis)
    assert failed.status is PrototypeSceneObserverStatus.PARSER_ERROR


def test_contrast_rejects_equal_endpoints_and_tampered_rendering() -> None:
    with pytest.raises(ObjectBongardSharedWitnessError, match="distinct"):
        ObjectBongardSharedWitnessContrast.create(
            0,
            shared_anchor="decorated figure",
            visual_axis="closed loop topology",
            group_0_endpoint="two touching loops",
            group_1_endpoint="two touching loops",
        )
    artifact, _ = _describe()
    tampered = deepcopy(artifact.to_data())
    tampered["contrast_candidates"][0]["rendered_group_0_cue"]["text"] = (  # type: ignore[index]
        "One unrelated shape appears curved."
    )
    with pytest.raises(ObjectBongardSharedWitnessSemanticsError):
        ObjectBongardSharedWitnessSemanticArtifact.from_data(tampered)


def test_protocol_states_observer_evidence_gap_explicitly() -> None:
    prompt = object_bongard_shared_witness_semantics_prompt()
    assert SHARED_WITNESS_SEMANTIC_PROTOCOL_ID.endswith(
        "single-entity-axis-contrasts-v1"
    )
    assert "same entity kind, not one physical individual shared across panels" in prompt
    assert "Within each panel, both endpoints" in prompt
    assert "must not be two separately coexisting features" in prompt
    assert "Python alone renders Description A and Description B" in prompt
    source = (
        Path(__file__).parents[1]
        / "object_bongard_shared_witness_semantics.py"
    ).read_text("utf-8")
    assert "import lean" not in source.lower()

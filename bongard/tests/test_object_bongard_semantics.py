from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import pytest

from bongard.object_bongard_semantics import (
    GROUP_SIZE,
    SEMANTIC_PROTOCOL_ID,
    SOFT_CUE_CANDIDATE_COUNT,
    ObjectBongardSemanticArtifact,
    ObjectBongardSemanticsError,
    describe_object_bongard_support,
    object_bongard_semantics_output_schema,
    object_bongard_semantics_prompt,
    verify_object_bongard_semantic_artifact,
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
            "group_0_cue_text": (
                "Unequal sector-like subshapes joined at a common apex."
            ),
            "group_1_cue_text": (
                "Rounded contour tapering toward a pointed junction."
            ),
        },
        "proposal_1": {
            "group_0_cue_text": (
                "Unequal sector-like lobes sharing a central junction."
            ),
            "group_1_cue_text": (
                "Three line-like spans forming a triangular arrangement."
            ),
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
        assert all(panel_id not in prompt for panel_id in (*GROUP_0, *GROUP_1))
        assert schema == object_bongard_semantics_output_schema()
        return CodexStructuredResult(
            chosen, _receipt(prompt, paths, names, schema, chosen)
        )

    artifact = describe_object_bongard_support(
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


def test_semantic_turn_emits_two_ranked_positive_soft_cue_pairs() -> None:
    artifact, calls = _describe()
    assert calls == artifact.to_data()["physical_call_count"] == 1
    assert artifact.status is PrototypeSceneObserverStatus.SUCCESS
    assert len(artifact.soft_cue_candidates) == SOFT_CUE_CANDIDATE_COUNT == 2
    first, second = artifact.soft_cue_candidates
    assert (first.candidate_rank, second.candidate_rank) == (0, 1)
    assert first.group_0_cue.text.startswith("Unequal sector-like")
    assert second.group_1_cue.text.startswith("Three line-like spans")
    assert first.pair_digest != second.pair_digest
    data = artifact.to_data()
    assert data["feature_catalog_used"] is False
    assert data["vision_prose_defines_soft_cue_identity"] is True
    assert data["model_can_choose_operator_threshold_or_polarity"] is False
    assert data["python_is_canonical_authority"] is True
    assert data["lean_required"] is False
    assert data["lean_required_for_replay"] is False
    assert ObjectBongardSemanticArtifact.from_data(
        data, expected_artifact_digest=artifact.artifact_digest
    ) == artifact
    assert verify_object_bongard_semantic_artifact(
        artifact,
        support_png_by_panel_id=_images(),
        expected_task_id=TASK_ID,
        expected_observation_context_digest=CONTEXT_DIGEST,
        expected_artifact_digest=artifact.artifact_digest,
    ) is artifact


def test_prompt_and_schema_fix_exact_two_forward_proposals_without_catalog() -> None:
    prompt = object_bongard_semantics_prompt()
    schema = object_bongard_semantics_output_schema()
    assert SEMANTIC_PROTOCOL_ID == (
        "bongard.object-task-semantics/two-ranked-positive-soft-cue-pairs-v4"
    )
    assert "exactly two ranked forward visual proposals" in prompt
    assert "proposal_0 is your strongest pair" in prompt
    assert "proposal_1 is the strongest genuinely alternate pair" in prompt
    assert "it may reuse one good group cue" in prompt
    assert "Python alone supplies the fixed observer scale" in prompt
    assert "catalog" not in prompt.lower()
    assert set(schema["properties"]) == {"proposal_0", "proposal_1"}
    assert schema["required"] == ["proposal_0", "proposal_1"]


def test_parser_and_transport_failures_are_typed_empty_slates() -> None:
    malformed = _payload()
    del malformed["proposal_1"]
    parser, calls = _describe(malformed)
    assert calls == 1
    assert parser.status is PrototypeSceneObserverStatus.PARSER_ERROR
    assert parser.soft_cue_candidates == ()
    assert parser.failure_code == "semantic_payload_rejected"
    verify_object_bongard_semantic_artifact(
        parser,
        support_png_by_panel_id=_images(),
        expected_task_id=TASK_ID,
        expected_observation_context_digest=CONTEXT_DIGEST,
        expected_artifact_digest=parser.artifact_digest,
    )

    calls = 0

    def broken_transport(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("offline")

    failed = describe_object_bongard_support(
        task_id=TASK_ID,
        group_0_panel_ids=GROUP_0,
        group_1_panel_ids=GROUP_1,
        support_png_by_panel_id=_images(),
        observation_context_digest=CONTEXT_DIGEST,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=broken_transport,
    )
    assert calls == 1
    assert failed.status is PrototypeSceneObserverStatus.TRANSPORT_ERROR
    assert failed.soft_cue_candidates == ()
    assert failed.failure_code == "semantic_transport_failed"


@pytest.mark.parametrize(
    "bad_text",
    (
        "No curved spans.",
        "A circle and a triangle.",
        "More oblique than the other group.",
        "Target score >= 3.",
    ),
)
def test_non_atomic_negated_or_executable_cue_text_fails_closed(
    bad_text: str,
) -> None:
    payload = _payload()
    payload["proposal_0"]["group_0_cue_text"] = bad_text  # type: ignore[index]
    artifact, calls = _describe(payload)
    assert calls == 1
    assert artifact.status is PrototypeSceneObserverStatus.PARSER_ERROR
    assert artifact.soft_cue_candidates == ()


def test_identical_ordered_pairs_across_ranks_fail_closed_but_one_cue_may_repeat() -> None:
    allowed, _ = _describe()
    assert (
        allowed.soft_cue_candidates[0].group_0_cue.cue_digest
        != allowed.soft_cue_candidates[1].group_0_cue.cue_digest
    )
    payload = _payload()
    payload["proposal_1"] = deepcopy(payload["proposal_0"])
    failed, calls = _describe(payload)
    assert calls == 1
    assert failed.status is PrototypeSceneObserverStatus.PARSER_ERROR

    repeated = _payload()
    repeated["proposal_1"]["group_0_cue_text"] = (  # type: ignore[index]
        repeated["proposal_0"]["group_0_cue_text"]  # type: ignore[index]
    )
    accepted, _ = _describe(repeated)
    assert accepted.status is PrototypeSceneObserverStatus.SUCCESS
    assert (
        accepted.soft_cue_candidates[0].group_0_cue
        == accepted.soft_cue_candidates[1].group_0_cue
    )


def test_semantic_replay_rejects_pixel_and_artifact_tamper() -> None:
    artifact, _calls = _describe()
    changed_images = _images()
    changed_images[GROUP_0[0]] = _png(99)
    with pytest.raises(ObjectBongardSemanticsError, match="presentation replay"):
        verify_object_bongard_semantic_artifact(
            artifact,
            support_png_by_panel_id=changed_images,
            expected_task_id=TASK_ID,
            expected_observation_context_digest=CONTEXT_DIGEST,
            expected_artifact_digest=artifact.artifact_digest,
        )
    changed = deepcopy(artifact.to_data())
    changed["soft_cue_candidates"][0]["group_0_cue"]["text"] = (  # type: ignore[index]
        "A tampered visible arrangement."
    )
    with pytest.raises(ObjectBongardSemanticsError):
        ObjectBongardSemanticArtifact.from_data(changed)


def test_group_partition_is_exact_and_disjoint() -> None:
    with pytest.raises(ObjectBongardSemanticsError, match="disjoint sorted"):
        describe_object_bongard_support(
            task_id=TASK_ID,
            group_0_panel_ids=GROUP_0,
            group_1_panel_ids=(*GROUP_1[:-1], GROUP_0[0]),
            support_png_by_panel_id=_images(),
            observation_context_digest=CONTEXT_DIGEST,
            model=MODEL,
            reasoning_effort=EFFORT,
            expected_launcher_digest=LAUNCHER_DIGEST,
            **NO_TOOLS_KWARGS,
            transport=lambda *_args, **_kwargs: None,
        )


def test_semantic_module_constructs_no_profile_and_imports_no_lean() -> None:
    source = (
        Path(__file__).parents[1] / "object_bongard_semantics.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    assert not any("lean" in name.lower() for name in imported)
    assert "ObjectProfile" not in source
    assert "OBJECT_FEATURE_CATALOG" not in source
    assert "feature_ids" not in source

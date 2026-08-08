from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import pytest

from bongard.object_bongard_semantics import (
    GROUP_SIZE,
    SEMANTIC_PROTOCOL_ID,
    ObjectBongardSemanticArtifact,
    ObjectBongardSemanticsError,
    describe_object_bongard_support,
    object_bongard_semantics_prompt,
    verify_object_bongard_semantic_artifact,
)
from bongard.object_bongard_rubric_observer import (
    ObjectBongardRubricSpec,
    object_bongard_catalog_contrast_rubric,
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
        "profiles": [
            {
                "group_id": "group_0",
                "rubric": "A winged angular form with several slanted spans.",
                "feature_ids": ["bird_like_support_ppm"],
            },
            {
                "group_id": "group_1",
                "rubric": "A rounded compact form with a curved boundary.",
                "feature_ids": ["rounded_leaf_support_ppm"],
            },
        ]
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


def test_semantic_turn_emits_audit_prose_and_one_catalog_cue_per_group() -> None:
    artifact, calls = _describe()
    assert calls == artifact.to_data()["physical_call_count"] == 1
    assert artifact.status is PrototypeSceneObserverStatus.SUCCESS
    assert artifact.rubrics == (
        "A winged angular form with several slanted spans.",
        "A rounded compact form with a curved boundary.",
    )
    assert artifact.feature_families == (
        ("bird_like_support_ppm",),
        ("rounded_leaf_support_ppm",),
    )
    prompt = object_bongard_semantics_prompt()
    assert SEMANTIC_PROTOCOL_ID == (
        "bongard.object-task-semantics/joint-contrastive-two-neutral-groups-v2"
    )
    assert "two neutral groups of six" in prompt
    assert "Consider both groups jointly" in prompt
    assert "exactly one matching feature identifier" in prompt
    assert "must both recur within its group" in prompt
    assert "visibly more characteristic" in prompt
    assert "merely typical" in prompt
    assert "Group names are neutral" in prompt
    assert "retained only as audit text" in prompt
    assert "do not choose an operator, threshold" in prompt
    assert "500000" not in prompt and "500_000" not in prompt
    assert artifact.to_data()["model_can_choose_operator_threshold_or_polarity"] is False
    assert ObjectBongardSemanticArtifact.from_data(
        artifact.to_data(), expected_artifact_digest=artifact.artifact_digest
    ) == artifact
    assert verify_object_bongard_semantic_artifact(
        artifact,
        support_png_by_panel_id=_images(),
        expected_task_id=TASK_ID,
        expected_observation_context_digest=CONTEXT_DIGEST,
        expected_artifact_digest=artifact.artifact_digest,
    ) is artifact


def test_parser_and_transport_failures_are_typed_not_empty_nominations() -> None:
    malformed = _payload()
    malformed["profiles"][0]["feature_ids"] = []  # type: ignore[index]
    parser, calls = _describe(malformed)
    assert calls == 1
    assert parser.status is PrototypeSceneObserverStatus.PARSER_ERROR
    assert parser.feature_families == ()
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
    assert failed.feature_families == ()
    assert failed.failure_code == "semantic_transport_failed"


@pytest.mark.parametrize(
    "feature_ids",
    (
        [],
        ["bird_like_support_ppm", "oblique_span_support_ppm"],
    ),
)
def test_zero_or_multiple_cues_fail_closed(feature_ids: list[str]) -> None:
    payload = _payload()
    payload["profiles"][0]["feature_ids"] = feature_ids  # type: ignore[index]
    artifact, calls = _describe(payload)
    assert calls == 1
    assert artifact.status is PrototypeSceneObserverStatus.PARSER_ERROR
    assert artifact.feature_families == ()


def test_same_cue_cannot_masquerade_as_a_contrast() -> None:
    payload = _payload()
    payload["profiles"][1]["feature_ids"] = [  # type: ignore[index]
        "bird_like_support_ppm"
    ]
    artifact, calls = _describe(payload)
    assert calls == 1
    assert artifact.status is PrototypeSceneObserverStatus.PARSER_ERROR
    assert artifact.feature_families == ()


@pytest.mark.parametrize(
    "audit_prose",
    (
        "A bird-like contour and several rounded appendages recur.",
        "A figure distinct from compact rounded leaves recurs.",
    ),
)
def test_audit_prose_cannot_conjoin_or_implicitly_complement_the_predicate(
    audit_prose: str,
) -> None:
    payload = _payload()
    payload["profiles"][0]["rubric"] = audit_prose  # type: ignore[index]
    artifact, calls = _describe(payload)
    assert calls == 1
    assert artifact.status is PrototypeSceneObserverStatus.SUCCESS
    assert artifact.rubrics[0] == audit_prose

    spec = ObjectBongardRubricSpec.from_semantic_artifact(
        artifact, expected_artifact_digest=artifact.artifact_digest
    )
    assert spec.feature_nominations == (
        "bird_like_support_ppm",
        "rounded_leaf_support_ppm",
    )
    assert spec.rubric == object_bongard_catalog_contrast_rubric(
        "bird_like_support_ppm", "rounded_leaf_support_ppm"
    )
    assert audit_prose != spec.rubric


@pytest.mark.parametrize(
    "rubric",
    (
        "A figure without a curved boundary recurs.",
        "A form lacking an enclosed region recurs.",
        "No rounded appendage is visible.",
    ),
)
def test_explicit_semantic_negation_cannot_smuggle_a_not_predicate(
    rubric: str,
) -> None:
    payload = _payload()
    payload["profiles"][0]["rubric"] = rubric  # type: ignore[index]
    artifact, calls = _describe(payload)
    assert calls == 1
    assert artifact.status is PrototypeSceneObserverStatus.PARSER_ERROR
    assert artifact.rubrics == ()
    assert artifact.feature_families == ()
    assert artifact.failure_code == "semantic_payload_rejected"


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
    changed["feature_families"][0] = ["straight_span_count"]
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
    assert "DESCRIPTION_SUPPORT_TARGET" not in source
    assert "parse_prototype_object_description_payload" not in source

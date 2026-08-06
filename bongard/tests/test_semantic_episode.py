from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from bongard.benchmark import (
    EpisodeStatus,
    SupportGatePolicy,
    SupportGateResult,
    VISUAL_SEMANTIC_PREDICATE_MODE,
    prepare_episode,
    run_episode,
)
from bongard.corpus import ShapeBongardCorpus
from bongard.semantic_episode import VisualSemanticEpisode
from bongard.semantic_protocol import build_visual_semantic_policy
from bongard.tests.test_semantic_observation import _receipt as _scorer_receipt
from bongard.tests.test_semantic_synthesis import _family
from bongard.tests.test_typed_visual_transport import _receipt
from bongard.transport import CloudPolicyCacheSnapshot, CodexStructuredResult


def _draw_panel(path: Path, *, positive: bool, offset: int) -> None:
    image = Image.new("L", (96, 96), color=255)
    draw = ImageDraw.Draw(image)
    dx = offset
    dy = (offset * 2) % 7
    draw.ellipse((14 + dx, 30 + dy, 32 + dx, 48 + dy), fill=0)
    if positive:
        draw.rectangle((60 - dx, 28 + dy, 76 - dx, 46 + dy), fill=0)
    image.save(path, format="PNG")


def _corpus(tmp_path: Path) -> tuple[ShapeBongardCorpus, str]:
    task_id = "ff_nact2_5_0000"
    root = tmp_path / "ShapeBongard_V2"
    for label, positive in (("1", True), ("0", False)):
        directory = root / "ff" / "images" / task_id / label
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(7):
            _draw_panel(directory / f"{index}.png", positive=positive, offset=index)
    (root / "ShapeBongard_V2_split.json").write_text(
        json.dumps({"train": [task_id]}), encoding="utf-8"
    )
    return ShapeBongardCorpus.from_root(root), task_id


def _proposal_payload() -> dict[str, object]:
    return {
        "positive_description": "two separated solid ink components",
        "panel_descriptions": {
            **{
                f"pos_{index}": f"two separated compact marks, presentation {index}"
                for index in range(6)
            },
            **{
                f"neg_{index}": f"one compact mark, presentation {index}"
                for index in range(6)
            },
        },
        "view": "literal_ink",
        "deterministic_atoms": [
            {
                "catalog_key": "component.count",
                "comparison": "equal",
                "arguments": {"target_count": 2},
            }
        ],
        "soft_claim": None,
        "formula": {"kind": "all", "atom_indices": [0]},
    }


def _mixed_proposal_payload() -> dict[str, object]:
    return {
        **_proposal_payload(),
        "soft_claim": {
            "positive_description": "two separated compact ink masses",
            "cue_descriptions": [
                "two compact ink masses are visible",
                "a clear white gap separates the two masses",
            ],
        },
        "formula": {"kind": "all", "atom_indices": [0, 1]},
    }


def test_direct_only_semantic_episode_runs_sealed_python_path_end_to_end(
    tmp_path: Path,
) -> None:
    corpus, task_id = _corpus(tmp_path)
    family = _family()
    policy = build_visual_semantic_policy(
        family, prospective_protocol=family.protocol
    )
    plan = prepare_episode(
        corpus,
        task_id,
        seed="semantic-direct-only",
        predicate_mode=VISUAL_SEMANTIC_PREDICATE_MODE,
        predicate_policy_digest=policy.digest(),
    )
    payload = _proposal_payload()

    def proposer_transport(prompt, paths, schema, **kwargs):
        return CodexStructuredResult(
            payload=payload,
            receipt=_receipt(
                prompt,
                paths,
                schema,
                payload,
                model=kwargs["model"],
                effort=kwargs["reasoning_effort"],
            ),
        )

    def forbidden_scorer_transport(*args, **kwargs):
        raise AssertionError("direct-only episode attempted a model scorer call")

    episode = VisualSemanticEpisode(
        task_id=task_id,
        support_commitment=plan.support,
        policy=policy,
        family=family,
        protocol=family.protocol,
        cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
        expected_codex_launcher_digest="b" * 64,
        expected_cloud_policy_cache_binding="absent",
        proposer_transport=proposer_transport,
        scorer_transport=forbidden_scorer_transport,
    )
    result = run_episode(
        plan,
        episode,
        episode,
        support_gate_policy=SupportGatePolicy.visual_semantic(),
    )

    assert result.status is EpisodeStatus.COMPLETE
    assert result.score.image_correct == 2
    assert result.score.puzzle_correct
    assert result.support_gate is not None
    assert result.support_gate.result is SupportGateResult.ALIGNED
    assert result.support_gate.transport_attempt_count == 0
    assert len(episode.support_sessions) == 12
    assert all(item.artifact is not None for item in episode.support_sessions)
    assert len(episode.query_sessions) == 2
    assert len(episode.query_artifacts) == 2
    assert episode.pre_observation_commitment is not None
    assert episode.proposed_rule is not None
    assert episode.proposed_rule.proposer_digest == (
        episode.pre_observation_commitment.digest
    )
    assert result.proposal_freeze is not None
    assert result.proposal_freeze.proposer_digest == (
        episode.pre_observation_commitment.digest
    )
    assert result.bundle is not None
    assert result.bundle.verify().predictions_match

    archive = episode.artifact_data()
    assert archive["python_predicate_authoritative"] is True
    assert archive["optional_checker_may_affect_result"] is False
    encoded_precommit = json.dumps(
        archive["pre_observation_commitment"], sort_keys=True
    )
    for source in plan._query_sources:
        assert source.panel.sha256 not in encoded_precommit


def test_mixed_semantic_episode_scores_each_panel_once_then_cold_replays(
    tmp_path: Path,
) -> None:
    corpus, task_id = _corpus(tmp_path)
    family = _family()
    family.verify_calibration()
    policy = build_visual_semantic_policy(
        family, prospective_protocol=family.protocol
    )
    plan = prepare_episode(
        corpus,
        task_id,
        seed="semantic-mixed",
        predicate_mode=VISUAL_SEMANTIC_PREDICATE_MODE,
        predicate_policy_digest=policy.digest(),
    )
    payload = _mixed_proposal_payload()
    query_digests = {source.panel.sha256 for source in plan._query_sources}
    support_digests = {source.panel.sha256 for source in plan._support_sources}
    proposer_panel_digests: list[str] = []
    scorer_panel_digests: list[str] = []

    def proposer_transport(prompt, paths, schema, **kwargs):
        proposer_panel_digests.extend(
            hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in paths
        )
        assert query_digests.isdisjoint(proposer_panel_digests)
        return CodexStructuredResult(
            payload=payload,
            receipt=_receipt(
                prompt,
                paths,
                schema,
                payload,
                model=kwargs["model"],
                effort=kwargs["reasoning_effort"],
            ),
        )

    episode: VisualSemanticEpisode

    def scorer_transport(prompt, paths, names, schema, **kwargs):
        assert tuple(names) == ("query.png",)
        assert len(paths) == 1
        assert episode.proposed_rule is not None
        assert episode.pre_observation_commitment is not None
        assert episode.compiled is not None
        panel_bytes = Path(paths[0]).read_bytes()
        panel_digest = hashlib.sha256(panel_bytes).hexdigest()
        scorer_panel_digests.append(panel_digest)

        with Image.open(paths[0]) as source_image:
            image = source_image.convert("L")
        two_components = sum(image.histogram()[:128]) > 450
        soft_lowering = episode.compiled.lowering_archive.soft_lowering
        assert soft_lowering is not None
        payload = {
            "cue_judgments": [
                {
                    "cue_id": cue.cue_id,
                    "judgment": "supported" if two_components else "unsupported",
                    "witness_ids": ["panel:geometry"] if two_components else [],
                }
                for cue in soft_lowering.claim.cues
            ]
        }
        return CodexStructuredResult(
            payload=payload,
            receipt=_scorer_receipt(prompt, paths, names, schema, payload),
        )

    episode = VisualSemanticEpisode(
        task_id=task_id,
        support_commitment=plan.support,
        policy=policy,
        family=family,
        protocol=family.protocol,
        proposer_transport=proposer_transport,
        scorer_transport=scorer_transport,
    )
    result = run_episode(
        plan,
        episode,
        episode,
        support_gate_policy=SupportGatePolicy.visual_semantic(),
    )

    assert result.status is EpisodeStatus.COMPLETE
    assert result.score.image_correct == 2
    assert result.score.puzzle_correct
    assert result.support_gate is not None
    assert result.support_gate.result is SupportGateResult.ALIGNED
    assert result.support_gate.transport_attempt_count == 12
    assert len(proposer_panel_digests) == 12
    assert set(proposer_panel_digests) == support_digests
    assert len(scorer_panel_digests) == 14
    assert set(scorer_panel_digests[:12]) == support_digests
    assert query_digests.isdisjoint(scorer_panel_digests[:12])
    assert set(scorer_panel_digests[12:]) == query_digests

    observations = [
        session.artifact
        for session in (*episode.support_sessions, *episode.query_sessions)
    ]
    assert all(artifact is not None for artifact in observations)
    assert all(
        artifact.transport_attempted for artifact in observations if artifact
    )
    assert all(
        artifact.scorer_artifact is not None
        for artifact in observations
        if artifact
    )
    assert {
        artifact.scorer_artifact.record.score
        for artifact in observations
        if artifact is not None and artifact.scorer_artifact is not None
    } == {0.0, 1.0}

    assert result.proposal_freeze is not None
    assert result.phases.index("proposal_frozen") < result.phases.index(
        "query_released"
    )
    assert episode.pre_observation_commitment is not None
    encoded_precommit = json.dumps(
        episode.pre_observation_commitment.to_data(), sort_keys=True
    )
    assert all(digest not in encoded_precommit for digest in query_digests)

    assert result.bundle is not None
    predictions = {
        item.query_id: item.positive
        for item in result.bundle.predictions.predictions
    }
    labels = {
        item.query_id: item.positive for item in result.bundle.labels.labels
    }
    assert predictions == labels
    assert set(predictions.values()) == {False, True}
    calls_before_cold_replay = len(scorer_panel_digests)
    assert result.bundle.verify().predictions_match
    assert len(scorer_panel_digests) == calls_before_cold_replay


def test_semantic_episode_rejects_calibrated_execution_environment_drift(
    tmp_path: Path,
) -> None:
    corpus, task_id = _corpus(tmp_path)
    family = _family()
    policy = build_visual_semantic_policy(
        family, prospective_protocol=family.protocol
    )
    plan = prepare_episode(
        corpus,
        task_id,
        seed="semantic-environment-drift",
        predicate_mode=VISUAL_SEMANTIC_PREDICATE_MODE,
        predicate_policy_digest=policy.digest(),
    )
    payload = _proposal_payload()

    def proposer_transport(prompt, paths, schema, **kwargs):
        return CodexStructuredResult(
            payload=payload,
            receipt=_receipt(
                prompt,
                paths,
                schema,
                payload,
                model=kwargs["model"],
                effort=kwargs["reasoning_effort"],
            ),
        )

    episode = VisualSemanticEpisode(
        task_id=task_id,
        support_commitment=plan.support,
        policy=policy,
        family=family,
        protocol=family.protocol,
        cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
        expected_codex_launcher_digest="a" * 64,
        expected_cloud_policy_cache_binding="absent",
        proposer_transport=proposer_transport,
    )
    result = run_episode(
        plan,
        episode,
        episode,
        support_gate_policy=SupportGatePolicy.visual_semantic(),
    )
    assert result.status is EpisodeStatus.PROPOSAL_ERROR
    assert result.failure is not None
    assert "launcher differs from the Stage-A environment" in result.failure.reason
    assert not episode.support_sessions
    assert not episode.query_sessions

    nonempty_cache = CloudPolicyCacheSnapshot(
        b'{"signed_payload":{},"signature":"fixture"}'
    )
    with pytest.raises(ValueError, match="cloud-policy cache differs"):
        VisualSemanticEpisode(
            task_id=task_id,
            support_commitment=plan.support,
            policy=policy,
            family=family,
            protocol=family.protocol,
            cloud_policy_cache_snapshot=nonempty_cache,
            expected_codex_launcher_digest="b" * 64,
            expected_cloud_policy_cache_binding="absent",
        )

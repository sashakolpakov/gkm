from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

from bongard.benchmark import (
    EpisodeStatus,
    SUPPORT_PROTOTYPE_PREDICATE_MODE,
    SupportGatePolicy,
    prepare_episode,
    run_episode,
)
from bongard.corpus import ShapeBongardCorpus
from bongard.legs.neutral_features import (
    FEATURE_GROUP_IDS,
    extract_neutral_features,
    feature_group_catalog_digest,
    feature_space_for_group,
)
from bongard.prototype_artifacts import PrototypeFreezePolicy
from bongard.prototype_episode import HeadlessPrototypeEpisode


@dataclass(frozen=True)
class Receipt:
    receipt_digest: str = "fixture-receipt"
    input_digest: str = "fixture-input"
    thread_id: str = "fixture-thread"
    requested_model: str = "gpt-test"
    requested_reasoning_effort: str = "medium"

    def to_dict(self):
        return self.__dict__


def _draw_panel(path: Path, *, positive: bool, index: int) -> None:
    image = Image.new("RGB", (96, 96), "white")
    draw = ImageDraw.Draw(image)
    offset = index
    if positive:
        draw.rectangle((25 - offset, 26, 69 + offset, 68), fill="black")
    else:
        draw.rectangle((12, 27 - offset, 34, 66 + offset), fill="black")
        draw.rectangle((61, 27 + offset, 83, 66 - offset), fill="black")
    image.save(path, format="PNG", optimize=False)


def _corpus(tmp_path: Path) -> tuple[ShapeBongardCorpus, str]:
    root = tmp_path / "ShapeBongard_V2"
    task_id = "ff_nact2_5_0000"
    for positive, label in ((True, "1"), (False, "0")):
        directory = root / "ff" / "images" / task_id / label
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(7):
            _draw_panel(directory / f"{index}.png", positive=positive, index=index)
    (root / "ShapeBongard_V2_split.json").write_text(
        json.dumps({"train": [task_id]}), encoding="utf-8"
    )
    return ShapeBongardCorpus.from_root(root), task_id


def _proposal_payload() -> dict[str, object]:
    return {
        "positive_description": "one connected foreground component",
        "panel_descriptions": {
            **{f"pos_{index}": "one compact foreground block" for index in range(6)},
            **{f"neg_{index}": "two separated foreground blocks" for index in range(6)},
        },
        "view": "literal_ink",
        "observable_requests": [
            {
                "observable_id": "prototype.topology",
                "affirmative_interpretation": (
                    "one connected foreground component is present"
                ),
                "arguments": {},
            }
        ],
        "formula_template": {
            "kind": "all",
            "atoms": ["prototype.topology"],
        },
        "hybrid_claim": None,
        "confidence": "high",
    }


def _policy(margin: float = 1e-6) -> PrototypeFreezePolicy:
    return PrototypeFreezePolicy.create(
        feature_catalog_digest=feature_group_catalog_digest(),
        allowed_groups={
            group_id: (feature_space_for_group(group_id), margin)
            for group_id in FEATURE_GROUP_IDS
        },
    )


def test_headless_prototype_episode_freezes_features_before_codex_and_queries(
    tmp_path: Path,
) -> None:
    corpus, task_id = _corpus(tmp_path)
    policy = _policy()
    plan = prepare_episode(
        corpus,
        task_id,
        seed="prototype-integration",
        predicate_mode=SUPPORT_PROTOTYPE_PREDICATE_MODE,
        predicate_policy_digest=policy.digest(),
    )
    events: list[tuple[str, str]] = []

    def tracing_extractor(panel_bytes: bytes):
        events.append(("extract", hashlib.sha256(panel_bytes).hexdigest()))
        return extract_neutral_features(panel_bytes)

    def fake_transport(prompt, paths, schema, **kwargs):
        del kwargs
        events.append(("codex", hashlib.sha256(prompt.encode("utf-8")).hexdigest()))
        assert tuple(Path(path).name for path in paths) == tuple(
            [f"pos_{index}.png" for index in range(6)]
            + [f"neg_{index}.png" for index in range(6)]
        )
        assert schema["properties"]["hybrid_claim"] == {"type": "null"}
        return SimpleNamespace(payload=_proposal_payload(), receipt=Receipt())

    episode = HeadlessPrototypeEpisode(
        support_commitment=plan.support,
        policy=policy,
        proposer_transport=fake_transport,
        extractor=tracing_extractor,
    )
    result = run_episode(
        plan,
        episode,
        episode,
        support_gate_policy=SupportGatePolicy.prototype(),
    )

    assert result.status is EpisodeStatus.COMPLETE
    assert result.score.image_correct == 2
    assert result.score.puzzle_correct
    assert result.support_gate is not None and result.support_gate.accepted
    assert result.bundle is not None
    assert result.bundle.verify().predictions_match

    support_digests = {item.panel.sha256 for item in plan.support.support}
    query_digests = {item.panel.sha256 for item in plan.queries}
    assert [kind for kind, _ in events] == (
        ["extract"] * 12 + ["codex"] + ["extract"] * 14
    )
    assert {digest for _, digest in events[:12]} == support_digests
    assert events[12][0] == "codex"
    assert {digest for _, digest in events[13:25]} == support_digests
    assert {digest for _, digest in events[25:]} == query_digests
    assert not query_digests & {digest for _, digest in events[:25]}

    assert episode.prequery is not None
    episode.prequery.verify(plan.support)
    assert episode.prequery.selected_feature_group_id == "prototype.topology"
    assert episode.prequery.semantic_proposal_digest == (
        episode.proposal.digest.removeprefix("sha256:")  # type: ignore[union-attr]
    )
    assert len(episode.prequery.support_panel_digests) == 12
    assert set(episode.artifact_data()["observations"]) == {"query-0", "query-1"}
    for entry in result.support_gate.entries:
        assert "positive" not in entry.observer_artifact
        assert entry.observer_artifact["schema"] == (
            "bongard.support-prototype-support-replay/v1"
        )


def test_prototype_episode_plan_policy_must_match_before_pixels(
    tmp_path: Path,
) -> None:
    corpus, task_id = _corpus(tmp_path)
    committed = _policy(1e-6)
    different = _policy(0.02)
    plan = prepare_episode(
        corpus,
        task_id,
        seed="prototype-policy-mismatch",
        predicate_mode=SUPPORT_PROTOTYPE_PREDICATE_MODE,
        predicate_policy_digest=committed.digest(),
    )
    calls = 0

    def tracing_extractor(panel_bytes: bytes):
        nonlocal calls
        calls += 1
        return extract_neutral_features(panel_bytes)

    episode = HeadlessPrototypeEpisode(
        support_commitment=plan.support,
        policy=different,
        proposer_transport=lambda *args, **kwargs: None,  # pragma: no cover
        extractor=tracing_extractor,
    )
    try:
        run_episode(
            plan,
            episode,
            episode,
            support_gate_policy=SupportGatePolicy.prototype(),
        )
    except Exception as exc:
        assert "committed episode plan" in str(exc)
    else:  # pragma: no cover - fail loudly without depending on pytest import.
        raise AssertionError("policy mismatch was accepted")
    assert calls == 0

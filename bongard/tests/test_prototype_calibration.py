from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from bongard.benchmark import prepare_episode
from bongard.corpus import ShapeBongardCorpus
from bongard.legs.neutral_features import FEATURE_GROUP_IDS
from bongard.support_prototypes import FrozenFeatureSpace
from bongard.prototype_calibration import (
    LABEL_AUTHORIZATION,
    SELECTION_OBJECTIVE,
    PrototypeCalibrationError,
    PrototypeCalibrationIntegrityError,
    PrototypeCalibrationRecord,
    calibrate_prototype_margins,
)


def _draw_panel(path: Path, *, positive: bool, index: int) -> None:
    image = Image.new("RGB", (96, 96), "white")
    draw = ImageDraw.Draw(image)
    shift = index - 3
    if positive:
        draw.rectangle((24 + shift, 25, 70 + shift, 69), fill="black")
    else:
        draw.rectangle((13 + shift, 27, 35 + shift, 67), fill="black")
        draw.rectangle((60 - shift, 27, 82 - shift, 67), fill="black")
    image.save(path, format="PNG", optimize=False)


def _draw_clipped_panel(path: Path) -> None:
    image = Image.new("RGB", (96, 96), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 20, 32, 74), fill="black")
    image.save(path, format="PNG", optimize=False)


def _corpus(
    tmp_path: Path,
    *,
    task_ids: tuple[str, ...] = ("ff_nact2_5_0000", "ff_nact2_5_0001"),
    split: str = "train",
) -> ShapeBongardCorpus:
    root = tmp_path / "ShapeBongard_V2"
    for task_id in task_ids:
        for positive, label in ((True, "1"), (False, "0")):
            directory = root / "ff" / "images" / task_id / label
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(7):
                _draw_panel(
                    directory / f"{index}.png",
                    positive=positive,
                    index=index,
                )
    (root / "ShapeBongard_V2_split.json").write_text(
        json.dumps({split: list(task_ids)}), encoding="utf-8"
    )
    return ShapeBongardCorpus.from_root(root)


def _group(data: dict[str, object], group_id: str) -> dict[str, object]:
    return next(  # type: ignore[return-value]
        item for item in data["groups"] if item["group_id"] == group_id  # type: ignore[index,union-attr]
    )


def test_calibration_is_python_only_canonical_and_uses_exact_counts(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    task_ids = tuple(reversed(corpus.task_ids))
    record = calibrate_prototype_margins(
        corpus,
        task_ids,
        seed="development-seed",
        candidate_margins=(0.04, 0.000001, 0.01),
    )
    data = record.to_data()

    assert data["task_ids"] == sorted(task_ids)
    assert data["candidate_margin_grid"] == [0.000001, 0.01, 0.04]
    assert data["selection_objective"] == list(SELECTION_OBJECTIVE)
    assert data["label_boundary"] == {
        "authorization": LABEL_AUTHORIZATION,
        "official_test_tasks_rejected": True,
        "query_labels_used_for": "development-margin-calibration-only",
    }
    assert tuple(item["group_id"] for item in data["groups"]) == FEATURE_GROUP_IDS
    assert len(data["tasks"]) == 2
    assert len(data["task_plan_digest"]) == 64
    assert len(data["record_digest"]) == 64
    assert set(data["source_digests"]) == {
        "bongard.legs.neutral_features",
        "bongard.prototype_calibration",
        "bongard.support_prototypes",
    }

    for group_id in FEATURE_GROUP_IDS:
        group = _group(data, group_id)
        rows = group["candidate_counts"]
        assert len(rows) == 3
        assert group["selected_margin"] == rows[0]["margin"]
        assert record.selected_margin(group_id) == group["selected_margin"]
        for row in rows:
            support = row["support"]
            query = row["development_query"]
            assert row["task_count"] == 2
            assert support["panel_count"] == 24
            assert support["strict_pass_rate"] == [
                support["strict_pass_task_count"],
                2,
            ]
            assert sum(support["dispositions"].values()) == 24
            assert query["image_count"] == 4
            assert query["puzzle_count"] == 2
            assert query["image_accuracy"] == [query["correct_image_count"], 4]
            assert query["puzzle_accuracy"] == [
                query["correct_puzzle_count"],
                2,
            ]
            assert sum(query["dispositions"].values()) == 4

    assert b"Lean" not in record.canonical_json()
    assert b"Codex" not in record.canonical_json()
    restored = PrototypeCalibrationRecord.from_data(
        json.loads(record.canonical_json())
    )
    assert restored == record
    assert restored.digest() == record.digest()
    assert restored.canonical_json() == record.canonical_json()

    first_policy = record.to_freeze_policy()
    second_policy = restored.to_freeze_policy()
    assert first_policy == second_policy
    assert first_policy.digest() == second_policy.digest()
    assert {
        item.feature_group_id: item.decision_margin
        for item in first_policy.allowed_feature_groups
    } == {
        group_id: record.selected_margin(group_id)
        for group_id in FEATURE_GROUP_IDS
    }


def test_seeded_holdouts_match_the_official_episode_selection(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, task_ids=("ff_nact2_5_0000",))
    task_id = corpus.task_ids[0]
    seed = "same-official-seed"
    record = calibrate_prototype_margins(
        corpus,
        (task_id,),
        seed=seed,
        candidate_margins=(0.000001,),
    )
    binding = record.to_data()["tasks"][0]
    plan = prepare_episode(corpus, task_id, seed=seed)
    query_digests = {item.panel.sha256 for item in plan.queries}
    task_manifest = corpus.task(task_id).build_manifest()

    expected_positive = next(
        item
        for item in task_manifest.panels
        if item.polarity == "positive"
        and item.index == binding["positive_query_index"]
    )
    expected_negative = next(
        item
        for item in task_manifest.panels
        if item.polarity == "negative"
        and item.index == binding["negative_query_index"]
    )
    assert query_digests == {
        expected_positive.sha256.removeprefix("sha256:"),
        expected_negative.sha256.removeprefix("sha256:"),
    }
    assert all(len(item["groups"]) == 4 for item in record.to_data()["tasks"])


def test_official_test_task_is_rejected_before_any_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _corpus(
        tmp_path,
        task_ids=("ff_nact2_5_0000",),
        split="test",
    )
    calls = 0

    def forbidden_extractor(panel: bytes):
        nonlocal calls
        calls += 1
        raise AssertionError("test pixels must not be read by calibration")

    monkeypatch.setattr(
        "bongard.prototype_calibration.extract_neutral_features",
        forbidden_extractor,
    )
    with pytest.raises(PrototypeCalibrationError, match="official test task"):
        calibrate_prototype_margins(
            corpus,
            corpus.task_ids,
            seed="forbidden",
            candidate_margins=(0.01,),
        )
    assert calls == 0


def test_record_rejects_tampering_and_input_ambiguity(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, task_ids=("ff_nact2_5_0000",))
    record = calibrate_prototype_margins(
        corpus,
        corpus.task_ids,
        seed="integrity",
        candidate_margins=(0.000001, 0.01),
    )
    tampered = deepcopy(record.to_data())
    tampered["tasks"][0]["positive_query_index"] = (
        tampered["tasks"][0]["positive_query_index"] + 1
    ) % 7
    with pytest.raises(PrototypeCalibrationIntegrityError):
        PrototypeCalibrationRecord.from_data(tampered)

    with pytest.raises(PrototypeCalibrationError, match="duplicates"):
        calibrate_prototype_margins(
            corpus,
            (corpus.task_ids[0], corpus.task_ids[0]),
            seed="duplicate-task",
            candidate_margins=(0.01,),
        )
    with pytest.raises(PrototypeCalibrationError, match="duplicates"):
        calibrate_prototype_margins(
            corpus,
            corpus.task_ids,
            seed="duplicate-margin",
            candidate_margins=(0.01, 0.01),
        )
    with pytest.raises(PrototypeCalibrationIntegrityError, match="positive"):
        calibrate_prototype_margins(
            corpus,
            corpus.task_ids,
            seed="zero-margin",
            candidate_margins=(0.0,),
        )


def test_task_source_digest_binds_exact_panel_bytes(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, task_ids=("ff_nact2_5_0000",))
    record = calibrate_prototype_margins(
        corpus,
        corpus.task_ids,
        seed="source-binding",
        candidate_margins=(0.000001,),
    )
    binding = record.to_data()["tasks"][0]
    source_preimage = corpus.task(corpus.task_ids[0]).build_manifest()

    assert binding["task_source_digest"] == source_preimage.digest.removeprefix(
        "sha256:"
    )
    assert {
        item.sha256.removeprefix("sha256:") for item in source_preimage.panels
    } == {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in corpus.task(corpus.task_ids[0]).panels
    }


def test_unfittable_support_is_retained_in_every_denominator_and_round_trips(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path, task_ids=("ff_nact2_5_0000",))
    task_id = corpus.task_ids[0]
    seed = "unfittable-support"
    plan = prepare_episode(corpus, task_id, seed=seed)
    query_digests = {item.panel.sha256 for item in plan.queries}
    support_path = next(
        path
        for path in corpus.task(task_id).negative
        if hashlib.sha256(path.read_bytes()).hexdigest() not in query_digests
    )
    _draw_clipped_panel(support_path)

    record = calibrate_prototype_margins(
        corpus,
        (task_id,),
        seed=seed,
        candidate_margins=(0.000001, 0.01),
    )
    data = record.to_data()
    for binding in data["tasks"][0]["groups"]:
        assert binding["status"] == "unfittable_support"
        assert binding["non_evaluation_reason"] == (
            "not_evaluated_due_to_unfittable_support"
        )
        assert binding["fit_plan_digest"] is None
        assert binding["prototype_digest"] is None
        assert len(binding["support_extractions"]) == 12
        assert len(binding["query_extractions"]) == 2
        assert sum(
            item["disposition"] == "indeterminate"
            for item in binding["support_extractions"]
        ) == 1
        failed = next(
            item
            for item in binding["support_extractions"]
            if item["disposition"] == "indeterminate"
        )
        assert "border_clipped" in failed["reason"]
        assert len(failed["receipt_digest"]) == 64

    for group in data["groups"]:
        for row in group["candidate_counts"]:
            support = row["support"]
            query = row["development_query"]
            assert support["panel_count"] == 12
            assert support["strict_pass_task_count"] == 0
            assert support[
                "not_evaluated_due_to_unfittable_support_count"
            ] == 12
            assert support["dispositions"] == {
                "present": 0,
                "certified_absent": 0,
                "indeterminate": 0,
                "error": 12,
            }
            assert query["image_count"] == 2
            assert query["correct_image_count"] == 0
            assert query["correct_puzzle_count"] == 0
            assert query[
                "not_evaluated_due_to_unfittable_support_count"
            ] == 2
            assert query["dispositions"] == {
                "present": 0,
                "certified_absent": 0,
                "indeterminate": 0,
                "error": 2,
            }

    restored = PrototypeCalibrationRecord.from_data(
        json.loads(record.canonical_json())
    )
    assert restored == record
    tampered = deepcopy(record.to_data())
    tampered_binding = tampered["tasks"][0]["groups"][0]
    tampered_binding["status"] = "fitted"
    with pytest.raises(PrototypeCalibrationIntegrityError):
        PrototypeCalibrationRecord.from_data(tampered)


def test_nonpresent_heldout_query_preserves_its_disposition_as_wrong(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path, task_ids=("ff_nact2_5_0000",))
    task_id = corpus.task_ids[0]
    seed = "indeterminate-query"
    plan = prepare_episode(corpus, task_id, seed=seed)
    query_digests = {item.panel.sha256 for item in plan.queries}
    query_path = next(
        path
        for path in corpus.task(task_id).negative
        if hashlib.sha256(path.read_bytes()).hexdigest() in query_digests
    )
    _draw_clipped_panel(query_path)

    record = calibrate_prototype_margins(
        corpus,
        (task_id,),
        seed=seed,
        candidate_margins=(0.000001,),
    )
    data = record.to_data()
    for binding in data["tasks"][0]["groups"]:
        assert binding["status"] == "fitted"
        assert binding["fit_plan_digest"] is not None
        assert binding["prototype_digest"] is not None
        assert sum(
            item["disposition"] == "indeterminate"
            for item in binding["query_extractions"]
        ) == 1
    for group in data["groups"]:
        query = group["candidate_counts"][0]["development_query"]
        assert query["image_count"] == 2
        assert query["correct_image_count"] <= 1
        assert query["correct_puzzle_count"] == 0
        assert query[
            "not_evaluated_due_to_unfittable_support_count"
        ] == 0
        assert query["dispositions"]["indeterminate"] >= 1


def test_freeze_policy_rejects_an_archived_space_different_from_runtime(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path, task_ids=("ff_nact2_5_0000",))
    record = calibrate_prototype_margins(
        corpus,
        corpus.task_ids,
        seed="policy-current-space",
        candidate_margins=(0.000001,),
    )
    content = record.content_data()
    changed_group = content["groups"][0]
    changed_group["feature_space"]["extractor_version"] = "archived-drift"
    changed_space = FrozenFeatureSpace.from_data(changed_group["feature_space"])
    changed_group["feature_space_digest"] = changed_space.digest()
    drifted = PrototypeCalibrationRecord.create(content)

    with pytest.raises(PrototypeCalibrationIntegrityError, match="current feature"):
        drifted.to_freeze_policy()

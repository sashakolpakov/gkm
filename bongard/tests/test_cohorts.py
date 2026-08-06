from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from bongard.cohorts import (
    BYTE_EXPOSURE_QUALIFICATION,
    CohortError,
    build_cohort_report,
    classify_task,
    parse_official_task_id,
    select_tasks,
)
from bongard.corpus import BongardTask, ShapeBongardCorpus, SplitIndex
from bongard.historical_exposure import load_historical_exposure


def _task(task_id: str, family: str) -> BongardTask:
    # Empty path tuples are intentional: the cohort planner is an inventory
    # consumer and must never inspect a panel.
    return BongardTask(
        task_id=task_id,
        family=family,
        root=Path("/must-not-be-opened") / task_id,
        positive=(),
        negative=(),
    )


def _mini_corpus(*, reversed_order: bool = False) -> ShapeBongardCorpus:
    tasks = (
        _task("bd_trapez_parallelogram_0000", "bd"),
        _task("hd_balanced_two-exist_quadrangle_0000", "hd"),
        _task("ff_nact2_5_0000", "ff"),
    )
    if reversed_order:
        tasks = tuple(reversed(tasks))
    ids = tuple(task.task_id for task in tasks)
    return ShapeBongardCorpus(
        Path("/must-not-be-opened"),
        tasks,
        layout="archive",
        split=SplitIndex(groups=(("train", ids),)),
    )


def test_exact_parser_accepts_official_vocabularies_and_rejects_guesses():
    seed = load_historical_exposure()

    basic = parse_official_task_id("bd_trapez_parallelogram_0000", seed)
    assert basic.family == "bd"
    assert basic.concepts == ("trapez_parallelogram",)
    assert basic.instance == 0

    abstract = parse_official_task_id(
        "hd_balanced_two-exist_quadrangle_0019", seed
    )
    assert abstract.concepts == ("balanced_two", "exist_quadrangle")
    assert abstract.instance == 19

    freeform = parse_official_task_id("ff_nact2_5_0000", seed)
    assert freeform.concepts == ("nact2_5",)

    for invalid in (
        "bd_not_a_shape_0000",
        "hd_not_an_attribute_0000",
        "ff_nact99_0000",
        # This is a historical generator-shaped id, not an official Basic id.
        "bd_open_s5_0279",
        "hd_convex_0020",
        "ff_nact6_0300",
        "bd_trapez_parallelogram_000",
        "xx_trapez_parallelogram_0000",
    ):
        with pytest.raises(CohortError):
            parse_official_task_id(invalid, seed)


def test_parser_rejects_an_ambiguous_semantic_decomposition():
    seed = load_historical_exposure()
    synthetic = replace(
        seed,
        basic_shape_families=seed.basic_shape_families + ("left", "right", "left-right"),
    )
    with pytest.raises(CohortError, match="ambiguous"):
        parse_official_task_id("bd_left-right_0000", synthetic)


def test_three_mini_tasks_have_conservative_expected_dispositions():
    seed = load_historical_exposure()

    basic = classify_task("bd_trapez_parallelogram_0000", seed)
    assert basic.semantic_exposure == "unused_family_partition"
    assert basic.semantic_cohort == "drill"
    assert basic.exact_task_exposure == "not_recorded"
    assert basic.historically_clean

    abstract = classify_task(
        "hd_balanced_two-exist_quadrangle_0000", seed
    )
    assert abstract.semantic_exposure == "unused_abstract_pair"
    assert abstract.semantic_cohort == "sealed"
    assert abstract.historically_clean

    freeform = classify_task("ff_nact2_5_0000", seed)
    assert freeform.semantic_exposure == "indeterminate"
    assert freeform.semantic_cohort is None
    assert not freeform.historically_clean


def test_basic_pair_must_stay_inside_one_frozen_partition():
    seed = load_historical_exposure()
    same = classify_task(
        f"bd_{seed.partition.drill[0]}-{seed.partition.drill[1]}_0000", seed
    )
    assert same.semantic_cohort == "drill"
    assert same.historically_clean

    crossed = classify_task(
        f"bd_{seed.partition.drill[0]}-{seed.partition.dev[0]}_0000", seed
    )
    assert crossed.semantic_exposure == "mixed_unused_partitions"
    assert crossed.semantic_cohort is None
    assert not crossed.historically_clean


def test_abstract_pair_requires_the_exact_canonical_order():
    seed = load_historical_exposure()
    canonical = classify_task(
        "hd_balanced_two-exist_quadrangle_0000", seed
    )

    assert canonical.historically_clean
    with pytest.raises(CohortError, match="non-canonical"):
        classify_task("hd_exist_quadrangle-balanced_two_0000", seed)

    singleton = classify_task("hd_convex_0005", seed)
    assert singleton.semantic_exposure == "historically_exposed"
    assert not singleton.historically_clean


def test_all_twenty_abstract_instances_stay_with_their_pair_partition():
    seed = load_historical_exposure()
    task_counts = {"drill": 0, "dev": 0, "sealed": 0}
    seen_pairs: set[tuple[str, str]] = set()

    for cohort in ("drill", "dev", "sealed"):
        for pair in getattr(seed.abstract_partition, cohort):
            siblings = tuple(
                classify_task(f"hd_{pair[0]}-{pair[1]}_{index:04d}", seed)
                for index in range(20)
            )
            assert {record.semantic_cohort for record in siblings} == {cohort}
            assert all(record.historically_clean for record in siblings)
            assert {record.parsed.concepts for record in siblings} == {pair}
            task_counts[cohort] += len(siblings)
            seen_pairs.add(pair)

    assert len(seen_pairs) == 127
    assert task_counts == {"drill": 1700, "dev": 420, "sealed": 420}


def test_exact_task_and_semantic_exposure_are_independent_axes():
    seed = load_historical_exposure()
    exact = classify_task("hd_convex_0004", seed)
    same_semantics_different_task = classify_task("hd_convex_0005", seed)

    assert exact.exact_task_exposure == "recorded"
    assert same_semantics_different_task.exact_task_exposure == "not_recorded"
    assert exact.semantic_exposure == same_semantics_different_task.semantic_exposure
    assert exact.semantic_exposure == "historically_exposed"


def test_report_is_deterministic_and_contains_no_byte_unseen_claim():
    seed = load_historical_exposure()
    forward = build_cohort_report(_mini_corpus(), seed)
    reverse = build_cohort_report(_mini_corpus(reversed_order=True), seed)

    assert forward.digest == reverse.digest
    assert forward.inventory_digest == reverse.inventory_digest
    assert forward.to_dict()["qualification"] == BYTE_EXPOSURE_QUALIFICATION
    assert forward.count_map["tasks"] == 3
    assert forward.count_map["historically_clean"] == 2
    assert forward.count_map["drill"] == 1
    assert forward.count_map["sealed"] == 1
    assert forward.count_map["semantic:unused_abstract_pair"] == 1
    assert forward.count_map["semantic:indeterminate"] == 1
    assert tuple(record.task_id for record in forward.records) == (
        "bd_trapez_parallelogram_0000",
        "ff_nact2_5_0000",
        "hd_balanced_two-exist_quadrangle_0000",
    )


def test_selection_helpers_filter_by_split_family_and_clean_cohort():
    seed = load_historical_exposure()
    corpus = _mini_corpus()

    clean = select_tasks(corpus, seed, split="train")
    assert tuple(task.task_id for task in clean) == (
        "bd_trapez_parallelogram_0000",
        "hd_balanced_two-exist_quadrangle_0000",
    )
    basic_drill = select_tasks(
        corpus,
        seed,
        split="train",
        family="bd",
        cohort="drill",
    )
    assert tuple(task.task_id for task in basic_drill) == (
        "bd_trapez_parallelogram_0000",
    )
    abstract = build_cohort_report(
        corpus,
        seed,
        family="hd",
        cohort="sealed",
    )
    assert abstract.count_map["tasks"] == 1

    with pytest.raises(CohortError, match="family"):
        select_tasks(corpus, seed, family="basic")
    with pytest.raises(CohortError, match="cohort"):
        select_tasks(corpus, seed, cohort="maybe")
    with pytest.raises(CohortError, match="split"):
        select_tasks(corpus, seed, split="maybe")


def test_report_rejects_disagreement_between_task_id_and_inventory_family():
    seed = load_historical_exposure()
    corpus = ShapeBongardCorpus(
        Path("/must-not-be-opened"),
        (_task("bd_trapez_parallelogram_0000", "hd"),),
        layout="archive",
        split=SplitIndex.empty(),
    )
    with pytest.raises(CohortError, match="stored under"):
        build_cohort_report(corpus, seed)

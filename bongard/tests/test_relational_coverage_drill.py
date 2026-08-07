from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from bongard.artifacts import canonical_json
from bongard.corpus import PNG_SIGNATURE, SplitIndex
from bongard.exposure import ExposureLedger, ExposureViolation
from bongard.historical_exposure import load_historical_exposure
from bongard.loop_scene_witnesses import extract_loop_scene_witnesses
from bongard.relational_coverage_drill import (
    SCHEMA,
    SELECTION_SCHEMA,
    run_coverage_drill,
    select_exact_unused_pilot,
    task_generator_key,
)


CORPUS = "sha256:" + "1" * 64
SPLIT = "sha256:" + "2" * 64


def _split(
    *,
    train: tuple[str, ...],
    val: tuple[str, ...] = (),
    test: tuple[str, ...] = (),
) -> SplitIndex:
    return SplitIndex(
        groups=(("test", test), ("train", train), ("val", val)),
        source_digest=SPLIT,
    )


def _png() -> bytes:
    image = Image.new("L", (96, 96), color=255)
    draw = ImageDraw.Draw(image)
    draw.polygon(((12, 76), (29, 21), (46, 76)), outline=0, width=3)
    draw.polygon(((49, 72), (61, 20), (87, 30), (81, 78)), outline=0, width=3)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = buffer.getvalue()
    assert payload.startswith(PNG_SIGNATURE)
    return payload


def _write_task(root: Path, task_id: str, payload: bytes) -> None:
    family = task_id.split("_", 1)[0]
    for label in ("1", "0"):
        directory = root / family / "images" / task_id / label
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(7):
            (directory / f"{index}.png").write_bytes(payload)


def test_generator_strata_are_task_id_metadata_only() -> None:
    assert task_generator_key("ff_nact3_5_0123") == ("ff", "nact3_5")
    assert task_generator_key("bd_square-open_circle_0000") == (
        "bd",
        "square-open_circle",
    )
    assert task_generator_key("hd_symmetric-thin_shape_0019") == (
        "hd",
        "symmetric-thin_shape",
    )


def test_selection_is_hash_ranked_bounded_exact_unused_and_never_test() -> None:
    train = (
        "ff_nact2_5_0000",
        "ff_nact2_5_0001",
        "ff_nact3_3_0000",
        "ff_nact3_4_0000",
    )
    val = ("ff_nact4_0000", "ff_nact4_0001")
    test = ("ff_nact5_0299",)
    exposed = ExposureLedger.create(CORPUS).record(
        phase="fixture",
        actor="test",
        purpose="synthetic predecessor",
        task_ids=("ff_nact3_3_0000",),
        observed_at="2026-08-07T00:00:00Z",
    )
    first = select_exact_unused_pilot(
        _split(train=train, val=val, test=test),
        exposed,
        source_corpus_manifest_digest=CORPUS,
        namespace="fixture-rank",
        per_generator=1,
        per_split_family=2,
        minimum_strict_dev_reserve=0,
        require_official_split=False,
    )
    second = select_exact_unused_pilot(
        _split(train=tuple(reversed(train)), val=tuple(reversed(val)), test=test),
        exposed,
        source_corpus_manifest_digest=CORPUS,
        namespace="fixture-rank",
        per_generator=1,
        per_split_family=2,
        minimum_strict_dev_reserve=0,
        require_official_split=False,
    )
    assert first == second
    selected = {item.task_id for item in first.selected}
    assert "ff_nact3_3_0000" not in selected
    assert "ff_nact5_0299" not in selected
    assert selected <= set(train) | set(val)
    assert len([item for item in first.selected if item.split == "train"]) == 2
    assert len([item for item in first.selected if item.split == "val"]) == 1
    assert first.generator_shortlist_count == 3
    assert "semantic independence" in first.content_data()[
        "generator_stratification_qualification"
    ]


def test_default_policy_preserves_at_least_sixteen_strict_dev_slots() -> None:
    historical = load_historical_exposure()
    synthetic_dev_ids = tuple(
        f"bd_{concept}_0000" for concept in historical.partition.dev
    )
    selection = select_exact_unused_pilot(
        _split(train=("ff_nact2_5_0000", *synthetic_dev_ids)),
        ExposureLedger.create(CORPUS),
        source_corpus_manifest_digest=CORPUS,
        namespace="fixture-dev-reserve",
        per_generator=1,
        per_split_family=1,
        require_official_split=False,
    )
    assert len(selection.protected_strict_dev_task_ids) >= 16
    assert set(selection.protected_strict_dev_task_ids).isdisjoint(
        item.task_id for item in selection.selected
    )
    protection = selection.content_data()["strict_dev_protection"]
    assert protection["minimum_reserved"] == 16
    assert protection["reference_capacity_after_a3"] == 16
    assert protection["semantic_closure_task_count"] >= protection[
        "protected_task_count"
    ]
    assert protection["capacity_preservation"] == {
        "baseline_individually_viable_task_count": len(
            selection.protected_strict_dev_task_ids
        ),
        "postselection_individually_viable_task_count": len(
            selection.protected_strict_dev_task_ids
        ),
        "all_baseline_tasks_remain_viable": True,
        "selected_tokens_disjoint_from_protected_closure": True,
    }
    assert selection.content_data()["schema"] == SELECTION_SCHEMA


def test_semantic_closure_blocks_mixed_task_that_would_consume_dev_slot() -> None:
    historical = load_historical_exposure()
    dev_witness = "bd_asymmetric_goldfish_0000"
    selected_v1_blocker = (
        "bd_asymmetric_goldfish-unbala_three_intersect_circles2_0000"
    )
    already_exposed_other_concept = (
        "bd_unbala_three_intersect_circles2_0000"
    )
    safe = "ff_nact2_5_0000"
    predecessor = ExposureLedger.create(CORPUS).record(
        phase="fixture",
        actor="test",
        purpose="make only the blocker's other Basic concept non-unseen",
        task_ids=(already_exposed_other_concept,),
        observed_at="2026-08-07T00:00:00Z",
    )

    # The mixed task cannot yield a semantic-unseen receipt, but it still
    # discloses asymmetric_goldfish if selected.  This was the v1 hole.
    with pytest.raises(ExposureViolation):
        predecessor.assert_semantically_unseen(
            task_ids=(selected_v1_blocker,),
            historical_seed=historical,
            expected_historical_seed_digest=historical.seed_digest,
        )
    predecessor.assert_semantically_unseen(
        task_ids=(dev_witness,),
        historical_seed=historical,
        expected_historical_seed_digest=historical.seed_digest,
    )

    selection = select_exact_unused_pilot(
        _split(train=(dev_witness, selected_v1_blocker, safe)),
        predecessor,
        source_corpus_manifest_digest=CORPUS,
        namespace="fixture-full-semantic-closure",
        per_generator=1,
        per_split_family=1,
        minimum_strict_dev_reserve=1,
        require_official_split=False,
    )

    selected_ids = tuple(item.task_id for item in selection.selected)
    assert selected_ids == (safe,)
    assert selection.protected_strict_dev_task_ids == (dev_witness,)
    assert set(selection.protected_strict_dev_closure_task_ids) == {
        dev_witness,
        selected_v1_blocker,
    }
    assert set(selection.protected_strict_dev_disclosure_tokens) == {
        "basic_family:asymmetric_goldfish",
        "basic_morphology:asymmetric_goldfish",
    }
    protection = selection.content_data()["strict_dev_protection"]
    assert protection["capacity_preservation"][
        "all_baseline_tasks_remain_viable"
    ] is True

    successor = predecessor.record(
        phase="fixture-successor",
        actor="test",
        purpose="verify selected IDs preserve the baseline DEV witness",
        task_ids=selected_ids,
        observed_at="2026-08-07T00:01:00Z",
        require_unseen=True,
    )
    successor.assert_semantically_unseen(
        task_ids=(dev_witness,),
        historical_seed=historical,
        expected_historical_seed_digest=historical.seed_digest,
    )

    counterfactual = predecessor.record(
        phase="fixture-v1-counterfactual",
        actor="test",
        purpose="demonstrate the completed v1 blocker",
        task_ids=(selected_v1_blocker,),
        observed_at="2026-08-07T00:02:00Z",
        require_unseen=True,
    )
    with pytest.raises(ExposureViolation):
        counterfactual.assert_semantically_unseen(
            task_ids=(dev_witness,),
            historical_seed=historical,
            expected_historical_seed_digest=historical.seed_digest,
        )


def test_run_persists_successor_before_all_fourteen_png_accesses(tmp_path: Path) -> None:
    task_id = "ff_nact2_5_0000"
    sealed_id = "ff_nact2_5_0299"
    payload = _png()
    corpus_root = tmp_path / "ShapeBongard_V2"
    _write_task(corpus_root, task_id, payload)
    # An official-test directory exists but its pixels must never be resolved or read.
    _write_task(corpus_root, sealed_id, b"not a png")
    ledger_store = tmp_path / "ledgers"
    output_store = tmp_path / "reports"
    accesses: list[str] = []
    cached = None

    def reader(path: Path) -> bytes:
        persisted = tuple(ledger_store.glob("*.exposure.json"))
        assert len(persisted) == 1
        assert sealed_id not in str(path)
        accesses.append(str(path))
        return path.read_bytes()

    def extractor(exact_png: bytes):
        nonlocal cached
        assert tuple(ledger_store.glob("*.exposure.json"))
        if cached is None:
            cached = extract_loop_scene_witnesses(exact_png)
        return cached

    result = run_coverage_drill(
        corpus_root=corpus_root,
        split_index=_split(train=(task_id,), test=(sealed_id,)),
        source_corpus_manifest_digest=CORPUS,
        predecessor=ExposureLedger.create(CORPUS),
        exposure_store=ledger_store,
        output_store=output_store,
        namespace="fixture-run",
        per_generator=1,
        per_split_family=1,
        minimum_strict_dev_reserve=0,
        require_official_split=False,
        observed_at="2026-08-07T01:00:00Z",
        png_reader=reader,
        extractor=extractor,
    )

    assert len(accesses) == 14
    assert all(task_id in path for path in accesses)
    assert result.exposure_successor.exposed_task_ids == {task_id}
    assert result.exposure_successor_path.is_file()
    assert result.report_path.is_file()
    report = dict(result.report)
    assert report["schema"] == SCHEMA
    assert report["selection"]["schema"] == SELECTION_SCHEMA
    output_digest = report.pop("output_digest")
    from hashlib import sha256

    assert output_digest == "sha256:" + sha256(canonical_json(report)).hexdigest()
    assert json.loads(result.report_path.read_text()) == result.report
    assert report["restrictions"] == {
        "allowed_splits": ["train", "val"],
        "official_test_pixels_authorized": False,
        "action_program_json_authorized": False,
        "proposer_or_model_authorized": False,
        "candidate_dependent_extraction_authorized": False,
    }
    manifest = report["selected_task_manifest"]
    assert len(manifest["tasks"]) == 1
    assert len(manifest["tasks"][0]["panels"]) == 14
    assert {item["task_id"] for item in manifest["tasks"]} == {task_id}
    assert len(report["panel_receipts"]) == 14
    global_counts = report["aggregates"]["global"]
    assert global_counts["extractor"] == {
        "panels_attempted": 14,
        "panels_succeeded": 14,
        "panels_errored": 0,
        "error_types": {},
    }
    assert global_counts["scenarios"]["observed"] > 0
    assert "dispositions" in global_counts["polygon"]
    assert "dispositions" in global_counts["contact"]
    assert "dispositions" in global_counts["obliqueness"]


def test_extractor_failure_is_error_not_negative_evidence(tmp_path: Path) -> None:
    task_id = "ff_nact2_5_0000"
    corpus_root = tmp_path / "ShapeBongard_V2"
    _write_task(corpus_root, task_id, _png())

    def broken(_: bytes):
        raise ValueError("synthetic extractor failure")

    result = run_coverage_drill(
        corpus_root=corpus_root,
        split_index=_split(train=(task_id,)),
        source_corpus_manifest_digest=CORPUS,
        predecessor=ExposureLedger.create(CORPUS),
        exposure_store=tmp_path / "ledgers",
        output_store=tmp_path / "reports",
        namespace="fixture-errors",
        per_generator=1,
        per_split_family=1,
        minimum_strict_dev_reserve=0,
        require_official_split=False,
        observed_at="2026-08-07T02:00:00Z",
        extractor=broken,
    )
    counts = result.report["aggregates"]["global"]
    assert counts["extractor"]["panels_errored"] == 14
    assert counts["extractor"]["panels_succeeded"] == 0
    assert counts["loops"]["observed"] == 0
    assert counts["contact"]["dispositions"] == {}
    assert {item["status"] for item in result.report["panel_receipts"]} == {"error"}

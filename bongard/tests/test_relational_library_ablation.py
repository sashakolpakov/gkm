from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from bongard.artifacts import canonical_json
from bongard.exposure import ExposureLedger
from bongard.loop_scene_witnesses import extract_loop_scene_witnesses
import bongard.relational_library_ablation as A


CORPUS = "sha256:" + "1" * 64
SPLIT = "sha256:" + "2" * 64
TASK_ID = "ff_nact2_5_0000"


def _panel(*, triangle_radius: int, quadrilateral_radius: int) -> bytes:
    image = Image.new("RGB", (160, 160), "white")
    draw = ImageDraw.Draw(image)
    tc = (42, 82)
    triangle = [
        (tc[0], tc[1] - triangle_radius),
        (tc[0] - triangle_radius, tc[1] + triangle_radius),
        (tc[0] + triangle_radius, tc[1] + triangle_radius),
    ]
    qc = (112, 82)
    q = quadrilateral_radius
    quadrilateral = [
        (qc[0] - q, qc[1] - q),
        (qc[0] + q, qc[1] - q + 5),
        (qc[0] + q - 4, qc[1] + q),
        (qc[0] - q + 3, qc[1] + q - 5),
    ]
    draw.line(triangle + [triangle[0]], fill="black", width=4, joint="curve")
    draw.line(
        quadrilateral + [quadrilateral[0]],
        fill="black",
        width=4,
        joint="curve",
    )
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _write_task(root: Path, positive: bytes, negative: bytes) -> None:
    for label, payload in (("1", positive), ("0", negative)):
        directory = root / "ff" / "images" / TASK_ID / label
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(7):
            (directory / f"{index}.png").write_bytes(payload)


def _coverage_fixture(
    tmp_path: Path, *, coverage_errors: bool = False
) -> tuple[Path, Path, Path, dict[bytes, object]]:
    corpus_root = tmp_path / "ShapeBongard_V2"
    positive = _panel(triangle_radius=11, quadrilateral_radius=34)
    negative = _panel(triangle_radius=34, quadrilateral_radius=11)
    _write_task(corpus_root, positive, negative)
    packets = {
        positive: extract_loop_scene_witnesses(positive),
        negative: extract_loop_scene_witnesses(negative),
    }
    predecessor = ExposureLedger.create(CORPUS)
    selected_item = {
        "task_id": TASK_ID,
        "family": "ff",
        "split": "train",
        "generator": "nact2_5",
        "generator_rank": "3" * 64,
        "family_rank": "4" * 64,
    }
    selection_content = {
        "schema": A.COVERAGE_SELECTION_SCHEMA_V1,
        "algorithm_id": A.COVERAGE_ALGORITHM_V1,
        "namespace": "fixture-library-ablation",
        "allowed_splits": ["train", "val"],
        "per_generator": 1,
        "per_split_family": 1,
        "source_corpus_manifest_digest": CORPUS,
        "split_source_digest": SPLIT,
        "exposure_predecessor_digest": predecessor.digest,
        "exact_unused_count": 1,
        "strict_dev_protection": {
            "policy": "synthetic fixture",
            "reference_capacity_after_a3": 16,
            "minimum_reserved": 0,
            "protected_task_count": 0,
            "protected_task_ids_digest": A._address([]),
        },
        "generator_stratification_qualification": (
            "task-ID-derived engineering coverage strata; not evidence of "
            "semantic independence"
        ),
        "generator_shortlist_count": 1,
        "selected": [selected_item],
    }
    selection = {
        **selection_content,
        "digest": A._address(selection_content),
    }
    panels = []
    panel_receipts = []
    for polarity, label, payload in (
        ("positive", "1", positive),
        ("negative", "0", negative),
    ):
        packet = packets[payload]
        for index in range(7):
            panel_id = f"ff/{TASK_ID}/{label}/{index}.png"
            png_digest = A._bytes_address(payload)
            panels.append(
                {
                    "panel_id": panel_id,
                    "polarity": polarity,
                    "index": index,
                    "filename": f"{index}.png",
                    "sha256": png_digest,
                    "size_bytes": len(payload),
                }
            )
            panel_receipts.append(
                {
                    "panel_id": panel_id,
                    "png_sha256": png_digest,
                    "status": "error" if coverage_errors else "present",
                    "error_type": (
                        "builtins.ValueError" if coverage_errors else None
                    ),
                    "loop_scene_packet_digest": (
                        None if coverage_errors else packet.digest()
                    ),
                }
            )
    task_content = {
        "task_id": TASK_ID,
        "family": "ff",
        "split": "train",
        "generator": "nact2_5",
        "panels": panels,
    }
    task_manifest = {**task_content, "digest": A._address(task_content)}
    manifest_content = {
        "schema": A.COVERAGE_MANIFEST_SCHEMA_V1,
        "source_corpus_manifest_digest": CORPUS,
        "split_source_digest": SPLIT,
        "selection_digest": selection["digest"],
        "tasks": [task_manifest],
    }
    manifest = {**manifest_content, "digest": A._address(manifest_content)}
    algorithms = {
        "coverage_algorithm_id": A.COVERAGE_ALGORITHM_V1,
        "coverage_python_source_digest": "5" * 64,
        **A._current_extractor_identities(),
    }
    restrictions = dict(A._RESTRICTIONS)
    input_commitment = {
        "schema": "gkm.bongard-relational-coverage-input.v1",
        "source_corpus_manifest_digest": CORPUS,
        "split_source_digest": SPLIT,
        "exposure_predecessor_digest": predecessor.digest,
        "selection_digest": selection["digest"],
        "algorithm_identities": algorithms,
        "restrictions": restrictions,
    }
    input_digest = A._address(input_commitment)
    successor = predecessor.record(
        phase="relational-coverage-drill",
        actor="fixture",
        purpose="synthetic completed coverage report",
        task_ids=(TASK_ID,),
        source=f"relational-coverage-input:{input_digest}",
        observed_at="2026-08-07T00:00:00Z",
    )
    ledger_path = tmp_path / (
        successor.digest.removeprefix("sha256:") + ".exposure.json"
    )
    successor.write_once(ledger_path)
    report_content = {
        "schema": A.COVERAGE_SCHEMA_V1,
        "algorithm_id": A.COVERAGE_ALGORITHM_V1,
        "input_digest": input_digest,
        "source": {
            "corpus_manifest_digest": CORPUS,
            "split_source_digest": SPLIT,
        },
        "exposure": {
            "predecessor_digest": predecessor.digest,
            "successor_digest": successor.digest,
            "successor_event_count": len(successor.events),
            "successor_filename": ledger_path.name,
            "precommit_before_selected_png_access": True,
        },
        "restrictions": restrictions,
        "algorithm_identities": algorithms,
        "selection": selection,
        "selected_task_manifest": manifest,
        "panel_receipts": sorted(panel_receipts, key=lambda item: item["panel_id"]),
        "aggregates": {"fixture": True},
    }
    report = {**report_content, "output_digest": A._address(report_content)}
    report_path = tmp_path / (
        report["output_digest"].removeprefix("sha256:") + ".coverage.json"
    )
    report_path.write_bytes(canonical_json(report) + b"\n")
    return report_path, ledger_path, corpus_root, packets


def _reseal_report(tmp_path: Path, report: dict[str, object]) -> Path:
    content = dict(report)
    content.pop("output_digest", None)
    resealed = {**content, "output_digest": A._address(content)}
    path = tmp_path / (
        resealed["output_digest"].removeprefix("sha256:") + ".coverage.json"
    )
    path.write_bytes(canonical_json(resealed) + b"\n")
    return path


def test_complete_library_ablation_is_selected_only_forward_and_durable(
    tmp_path: Path,
) -> None:
    coverage_path, ledger_path, corpus_root, packet_cache = _coverage_fixture(tmp_path)
    accesses: list[Path] = []
    extraction_calls = 0

    def reader(path: Path) -> bytes:
        accesses.append(path)
        return path.read_bytes()

    def extractor(payload: bytes):
        nonlocal extraction_calls
        extraction_calls += 1
        return packet_cache[payload]

    result = A.run_relational_library_ablation(
        coverage_report_path=coverage_path,
        exposure_successor_path=ledger_path,
        corpus_root=corpus_root,
        output_store=tmp_path / "ablation",
        png_reader=reader,
        extractor=extractor,
    )

    assert len(accesses) == 14
    assert extraction_calls == 14
    assert all(TASK_ID in str(path) and path.suffix == ".png" for path in accesses)
    assert result.report["query_library"]["count"] == 2520
    assert result.report["qualification"]["evaluation_kind"] == (
        "resubstitution/library-coverage"
    )
    assert result.report["qualification"]["benchmark_or_generalization_result"] is False
    assert result.report["restrictions"]["new_exposure_event_created"] is False
    task = result.report["tasks"][0]
    assert task["full_7_plus_7_resubstitution"]["exact_forward_separator_count"] > 0
    folds = task["paired_leave_one_index_out"]["folds"]
    assert [fold["omitted_index_per_side"] for fold in folds] == list(range(7))
    assert all(fold["fit_panel_count"] == 12 for fold in folds)
    assert all(fold["fit_exact_forward_separator_count"] > 0 for fold in folds)
    assert all(fold["any_fit_separator_is_heldout_forward_correct"] for fold in folds)
    assert task["best_honest_forward_profile"]["profile"] == {
        "positive": {
            "present": 7,
            "certified_absent": 0,
            "indeterminate": 0,
            "error": 0,
        },
        "negative": {
            "present": 0,
            "certified_absent": 7,
            "indeterminate": 0,
            "error": 0,
        },
    }
    cold = json.loads(result.report_path.read_bytes())
    assert cold == result.report
    output_digest = cold.pop("output_digest")
    assert A._address(cold) == output_digest


def test_tampered_coverage_report_is_rejected_before_png_access(tmp_path: Path) -> None:
    coverage_path, ledger_path, corpus_root, _ = _coverage_fixture(tmp_path)
    report = json.loads(coverage_path.read_bytes())
    report["aggregates"] = {"tampered": True}
    coverage_path.write_bytes(canonical_json(report) + b"\n")
    accesses: list[Path] = []

    with pytest.raises(A.RelationalLibraryAblationError, match="output digest mismatch"):
        A.run_relational_library_ablation(
            coverage_report_path=coverage_path,
            exposure_successor_path=ledger_path,
            corpus_root=corpus_root,
            output_store=tmp_path / "ablation",
            png_reader=lambda path: accesses.append(path) or path.read_bytes(),
        )
    assert accesses == []


def test_resealed_official_test_authorization_is_still_rejected(tmp_path: Path) -> None:
    coverage_path, ledger_path, corpus_root, _ = _coverage_fixture(tmp_path)
    report = json.loads(coverage_path.read_bytes())
    report["restrictions"]["official_test_pixels_authorized"] = True
    malicious = _reseal_report(tmp_path, report)
    accesses: list[Path] = []

    with pytest.raises(A.RelationalLibraryAblationError, match="official-test/action/model"):
        A.run_relational_library_ablation(
            coverage_report_path=malicious,
            exposure_successor_path=ledger_path,
            corpus_root=corpus_root,
            output_store=tmp_path / "ablation",
            png_reader=lambda path: accesses.append(path) or path.read_bytes(),
        )
    assert accesses == []


def test_successor_must_end_in_exact_selected_coverage_event(tmp_path: Path) -> None:
    coverage_path, ledger_path, corpus_root, _ = _coverage_fixture(tmp_path)
    successor = ExposureLedger.load(ledger_path)
    wrong = successor.record(
        phase="unrelated",
        actor="fixture",
        purpose="must not be accepted as the named successor",
        task_ids=("ff_nact2_5_0001",),
        observed_at="2026-08-07T01:00:00Z",
    )
    wrong_path = tmp_path / (wrong.digest.removeprefix("sha256:") + ".exposure.json")
    wrong.write_once(wrong_path)
    accesses: list[Path] = []

    with pytest.raises(A.RelationalLibraryAblationError, match="digest/corpus binding"):
        A.run_relational_library_ablation(
            coverage_report_path=coverage_path,
            exposure_successor_path=wrong_path,
            corpus_root=corpus_root,
            output_store=tmp_path / "ablation",
            png_reader=lambda path: accesses.append(path) or path.read_bytes(),
        )
    assert accesses == []


def test_extractor_failure_is_error_and_never_negative_evidence(tmp_path: Path) -> None:
    coverage_path, ledger_path, corpus_root, _ = _coverage_fixture(
        tmp_path, coverage_errors=True
    )

    def broken(_: bytes):
        raise ValueError("synthetic replay failure")

    result = A.run_relational_library_ablation(
        coverage_report_path=coverage_path,
        exposure_successor_path=ledger_path,
        corpus_root=corpus_root,
        output_store=tmp_path / "ablation",
        extractor=broken,
    )
    task = result.report["tasks"][0]
    assert task["extraction_replay"]["failure_count"] == 14
    assert task["full_7_plus_7_resubstitution"]["exact_forward_separator_count"] == 0
    profile = task["best_honest_forward_profile"]["profile"]
    assert profile["positive"]["error"] == 7
    assert profile["negative"]["error"] == 7
    assert profile["negative"]["certified_absent"] == 0


def test_historical_extraction_error_stays_error_after_current_success(
    tmp_path: Path,
) -> None:
    coverage_path, ledger_path, corpus_root, packet_cache = _coverage_fixture(
        tmp_path, coverage_errors=True
    )
    result = A.run_relational_library_ablation(
        coverage_report_path=coverage_path,
        exposure_successor_path=ledger_path,
        corpus_root=corpus_root,
        output_store=tmp_path / "ablation",
        extractor=lambda payload: packet_cache[payload],
    )
    task = result.report["tasks"][0]
    assert task["extraction_replay"]["failure_count"] == 14
    assert {
        item["current_reextract_status"]
        for item in task["extraction_replay"]["receipts"]
    } == {"present_but_noncomparable"}
    assert task["full_7_plus_7_resubstitution"]["exact_forward_separator_count"] == 0


def test_current_packet_digest_drift_fails_closed(tmp_path: Path) -> None:
    coverage_path, ledger_path, corpus_root, packet_cache = _coverage_fixture(tmp_path)

    def drifted(payload: bytes):
        return replace(packet_cache[payload], parent_bundle_digest="0" * 64)

    with pytest.raises(A.RelationalLibraryAblationError, match="packet digest differs"):
        A.run_relational_library_ablation(
            coverage_report_path=coverage_path,
            exposure_successor_path=ledger_path,
            corpus_root=corpus_root,
            output_store=tmp_path / "ablation",
            extractor=drifted,
        )


def test_current_extractor_identity_drift_rejects_before_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coverage_path, ledger_path, corpus_root, _ = _coverage_fixture(tmp_path)
    original = A._current_extractor_identities

    def drifted() -> dict[str, str]:
        result = original()
        result["loop_scene_extractor_digest"] = "0" * 64
        return result

    monkeypatch.setattr(A, "_current_extractor_identities", drifted)
    accesses: list[Path] = []
    with pytest.raises(A.RelationalLibraryAblationError, match="extractor identities differ"):
        A.run_relational_library_ablation(
            coverage_report_path=coverage_path,
            exposure_successor_path=ledger_path,
            corpus_root=corpus_root,
            output_store=tmp_path / "ablation",
            png_reader=lambda path: accesses.append(path) or path.read_bytes(),
        )
    assert accesses == []


def test_precomputed_evaluator_matches_all_2520_canonical_query_results() -> None:
    payloads = (
        _panel(triangle_radius=11, quadrilateral_radius=34),
        _panel(triangle_radius=34, quadrilateral_radius=11),
    )
    queries = A.enumerate_factorized_shape_ratio_queries()
    plans = tuple(A._query_plan(query) for query in queries)
    assert len(queries) == 2520
    for payload in payloads:
        packet = extract_loop_scene_witnesses(payload)
        prepared = A._prepare_packet(packet)
        vectorized = A._evaluate_prepared_library(prepared, plans)
        for index, (query, plan) in enumerate(zip(queries, plans, strict=True)):
            canonical = A.evaluate_relational_query(query, packet)
            expected = A._PanelOutcome(
                canonical.disposition,
                A._failure_reason_codes(canonical, query),
            )
            assert A._evaluate_prepared_query(prepared, plan) == expected
            assert vectorized[index] is canonical.disposition

from __future__ import annotations

import hashlib
import copy
from io import BytesIO
from itertools import combinations, product
import json
from pathlib import Path
from typing import Any, Mapping
import zipfile

from PIL import Image, ImageDraw
import pytest

import bongard.closed_visual_predicates as closed_module
import bongard.relational_headless_runner as runner_module
from bongard.artifacts import canonical_digest
from bongard.corpus import SplitIndex
from bongard.exposure import ExposureLedger
from bongard.loop_scene_witnesses import extract_loop_scene_witnesses
from bongard.composite_visual_packet import extract_exact_panel_witness_packet
from bongard.closed_visual_predicates import (
    complete_closed_predicate_library_identity,
    enumerate_complete_closed_predicates,
)
from bongard.relational_headless_runner import (
    EXACT_UNUSED_TRAIN_ENGINEERING_MODE,
    EXACT_UNUSED_TRAIN_ENGINEERING_TASK_IDS,
    EXPLICITLY_SEALED_ENGINEERING_TASK_ID,
    ReleaseArchiveAuthenticator,
    RelationalHeadlessRunError,
    _write_once_durable,
    cold_replay_relational_headless_run,
    load_relational_artifact,
    parse_closed_visual_proposal,
    parse_relational_proposal,
    prepare_relational_headless_plan,
    run_relational_headless,
    verify_relational_predictions,
    verify_relational_proposal_freeze,
)
from bongard.release import OfficialReleaseDescriptor
from bongard.transport import (
    CODEX_ISOLATION_POLICY,
    CODEX_RECEIPT_SCHEMA,
    STRUCTURED_INPUT_DIGEST_SCHEMA,
    CodexReceipt,
    CodexStructuredResult,
    ordered_panel_view_digest,
    semantic_panel_set_digest,
)


CORPUS = "sha256:" + "1" * 64
SPLIT = "sha256:" + "2" * 64
LAUNCHER = "3" * 64
TASK = "bd_asymmetric_goldfish_0000"
SEALED = "ff_nact5_0299"
LABEL_NONCE = "4" * 64


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


def _write_task(root: Path, task: str = TASK) -> tuple[bytes, bytes]:
    positive = _panel(triangle_radius=11, quadrilateral_radius=34)
    negative = _panel(triangle_radius=34, quadrilateral_radius=11)
    for label, payload in (("1", positive), ("0", negative)):
        directory = root / "bd" / "images" / task / label
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(7):
            (directory / f"{index}.png").write_bytes(payload)
    return positive, negative


def _release_authenticator(
    tmp_path: Path, root: Path
) -> tuple[ReleaseArchiveAuthenticator, Path, Path]:
    archive_path = tmp_path / "ShapeBongard_V2.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as bundle:
        for panel in sorted(root.rglob("*.png")):
            bundle.write(
                panel,
                "ShapeBongard_V2/" + panel.relative_to(root).as_posix(),
            )
    archive_bytes = archive_path.read_bytes()
    descriptor = OfficialReleaseDescriptor(
        release_id="synthetic-relational-runner-test",
        archive_filename=archive_path.name,
        archive_sha256="sha256:" + hashlib.sha256(archive_bytes).hexdigest(),
        archive_size_bytes=len(archive_bytes),
        split_filename="synthetic-split.json",
        split_sha256=SPLIT,
        split_size_bytes=1,
        upstream_repository="synthetic",
        upstream_commit="a" * 40,
        family_counts=(("bd", 1),),
        primary_split_counts=(("test", 1), ("train", 1), ("val", 0)),
        regime_counts=(),
        task_ids_sha256="sha256:" + "9" * 64,
        corpus_manifest_sha256=CORPUS,
    )
    descriptor_path = tmp_path / "release.json"
    descriptor_path.write_text(
        json.dumps(
            descriptor.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return (
        ReleaseArchiveAuthenticator.load(
            release_descriptor_path=descriptor_path,
            expected_release_descriptor_digest=descriptor.digest,
            archive_path=archive_path,
        ),
        descriptor_path,
        archive_path,
    )


def _split(task: str = TASK) -> SplitIndex:
    return SplitIndex(
        groups=(("test", (SEALED,)), ("train", (task,)), ("val", ())),
        source_digest=SPLIT,
    )


def _receipt(
    *,
    prompt: str,
    paths: tuple[str, ...],
    schema: Mapping[str, Any],
    payload: Mapping[str, Any],
    model: str,
    reasoning_effort: str,
) -> CodexReceipt:
    prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    schema_digest = canonical_digest(schema)
    identities = []
    for path in paths:
        data = Path(path).read_bytes()
        identities.append(
            {
                "name": Path(path).name,
                "byte_count": len(data),
                "content_digest": hashlib.sha256(data).hexdigest(),
            }
        )
    panel_view = ordered_panel_view_digest(paths)
    panel_set = semantic_panel_set_digest(paths)
    input_digest = canonical_digest(
        {
            "schema": STRUCTURED_INPUT_DIGEST_SCHEMA,
            "task": prompt,
            "ordered_panel_identities": identities,
            "panel_view_digest": panel_view,
            "panel_set_digest": panel_set,
            "prompt_digest": prompt_digest,
            "output_schema_digest": schema_digest,
        }
    )
    body: dict[str, Any] = {
        "schema": CODEX_RECEIPT_SCHEMA,
        "source": "codex-cli",
        "requested_model": model,
        "reported_model": model,
        "model_identity_evidence": "jsonl-reported-model",
        "requested_reasoning_effort": reasoning_effort,
        "input_tokens": 1,
        "cached_input_tokens": 0,
        "output_tokens": 1,
        "reasoning_output_tokens": 0,
        "thread_id": "00000000-0000-4000-8000-000000000001",
        "codex_cli_version": "fixture",
        "codex_launcher_digest": LAUNCHER,
        "cloud_config_bundle_cache_binding": "absent",
        "task_digest": prompt_digest,
        "current_source_digest": "",
        "current_log_digest": "",
        "prompt_digest": prompt_digest,
        "input_digest_schema": STRUCTURED_INPUT_DIGEST_SCHEMA,
        "input_digest": input_digest,
        "output_schema_digest": schema_digest,
        "panel_view_digest": panel_view,
        "panel_set_digest": panel_set,
        "structured_output_digest": canonical_digest(payload),
        "proposed_source_digest": "",
        "proposed_log_digest": "",
        "event_stream_digest": "5" * 64,
        "event_types": [
            "thread.started",
            "turn.started",
            "item.completed",
            "turn.completed",
        ],
        "item_types": ["agent_message"],
        "isolation_policy": CODEX_ISOLATION_POLICY,
        "outcome": "success",
    }
    body["receipt_digest"] = canonical_digest(body)
    return CodexReceipt(
        **{
            **body,
            "event_types": tuple(body["event_types"]),
            "item_types": tuple(body["item_types"]),
        }
    )


def _cached_extractor(*, composite: bool = False):
    cache = {}

    def extract(payload: bytes):
        digest = hashlib.sha256(payload).hexdigest()
        if digest not in cache:
            cache[digest] = (
                extract_exact_panel_witness_packet(payload)
                if composite
                else extract_loop_scene_witnesses(payload)
            )
        return cache[digest]

    return extract


def _run(
    tmp_path: Path,
    *,
    proposal: Mapping[str, Any],
    transport_hook=None,
    before_run=None,
    task: str = TASK,
    benchmark_mode: str = runner_module.STRICT_DEV_MODE,
):
    root = tmp_path / "ShapeBongard_V2"
    positive, negative = _write_task(root, task)
    release_authenticator, _descriptor_path, _archive_path = (
        _release_authenticator(tmp_path, root)
    )
    predecessor = ExposureLedger.create(CORPUS)
    exposure_store = tmp_path / "exposure"
    artifact_store = tmp_path / "artifacts"
    plan = prepare_relational_headless_plan(
        task_id=task,
        split_index=_split(task),
        predecessor=predecessor,
        expected_exposure_predecessor_digest=predecessor.digest,
        expected_corpus_digest=CORPUS,
        expected_split_source_digest=SPLIT,
        seed="fixture-seed",
        exposure_observed_at="2026-08-07T12:00:00Z",
        expected_launcher_digest=LAUNCHER,
        release_authenticator=release_authenticator,
        label_nonce=LABEL_NONCE,
        benchmark_mode=benchmark_mode,
    )
    heldout = {
        root / "bd" / "images" / task / "1" / f"{plan._positive_query_index}.png",
        root / "bd" / "images" / task / "0" / f"{plan._negative_query_index}.png",
    }
    if before_run is not None:
        before_run(root, plan)
    accesses: list[Path] = []
    calls: list[tuple[str, tuple[str, ...]]] = []

    def reader(path: Path) -> bytes:
        assert len(tuple(exposure_store.glob("*.exposure.json"))) == 1
        if path in heldout:
            assert len(tuple(artifact_store.glob("*.relational-proposal-freeze.json"))) == 1
        accesses.append(path)
        return path.read_bytes()

    def transport(prompt, paths, schema, **kwargs):
        calls.append((prompt, tuple(paths)))
        assert not any(path in heldout for path in accesses)
        assert task not in prompt
        assert [Path(path).name for path in paths] == [
            *(f"pos_{index}.png" for index in range(6)),
            *(f"neg_{index}.png" for index in range(6)),
        ]
        if transport_hook is not None:
            return transport_hook(prompt, tuple(paths), schema, kwargs)
        receipt = _receipt(
            prompt=prompt,
            paths=tuple(paths),
            schema=schema,
            payload=proposal,
            model=kwargs["model"],
            reasoning_effort=kwargs["reasoning_effort"],
        )
        return CodexStructuredResult(dict(proposal), receipt)

    outcome = run_relational_headless(
        corpus_root=root,
        task_id=task,
        split_index=_split(task),
        predecessor=predecessor,
        expected_corpus_digest=CORPUS,
        expected_split_source_digest=SPLIT,
        expected_exposure_predecessor_digest=predecessor.digest,
        seed="fixture-seed",
        exposure_observed_at="2026-08-07T12:00:00Z",
        exposure_store=exposure_store,
        artifact_store=artifact_store,
        expected_launcher_digest=LAUNCHER,
        release_authenticator=release_authenticator,
        label_nonce=LABEL_NONCE,
        transport=transport,
        png_reader=reader,
        extractor=_cached_extractor(
            composite=benchmark_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE
        ),
        packet_verifier=lambda packet, **_kwargs: packet,
        benchmark_mode=benchmark_mode,
    )
    return outcome, accesses, heldout, calls, positive, negative


GOOD = {
    "numerator_side_count": 3,
    "denominator_side_count": 4,
    "area_ratio": "1/8",
    "denominator_obliqueness_millidegrees": None,
    "rationale": "small triangle and much larger quadrilateral",
}

ENGINEERING_GOOD = {
    "kind": "relational",
    "relational": {
        "numerator_side_count": 3,
        "denominator_side_count": 4,
        "area_ratio": "1/8",
        "denominator_obliqueness_millidegrees": None,
    },
    "direct_atom_0": None,
    "direct_atom_1": None,
    "direct_atom_2": None,
    "symmetry": None,
    "rationale": "small triangle and much larger quadrilateral",
}


def test_complete_run_freezes_before_queries_and_commits_predictions_before_labels(
    tmp_path: Path,
) -> None:
    outcome, accesses, heldout, calls, _positive, _negative = _run(
        tmp_path, proposal=GOOD
    )

    assert outcome.status == "complete"
    assert len(calls) == 1
    assert len(accesses) == 14
    assert set(accesses[-2:]) == heldout
    assert outcome.freeze_path is not None and outcome.freeze_path.is_file()
    assert outcome.prediction_path is not None and outcome.prediction_path.is_file()
    freeze = load_relational_artifact(outcome.freeze_path)
    predictions = load_relational_artifact(outcome.prediction_path)
    assert verify_relational_proposal_freeze(
        freeze,
        plan=outcome.plan,
        exposure_successor=outcome.exposure_successor,
    )["support_gate_accepted"] is True
    assert verify_relational_predictions(
        predictions,
        freeze=freeze,
        plan=outcome.plan,
        exposure_successor=outcome.exposure_successor,
    )[
        "labels_revealed"
    ] is False
    assert outcome.artifact["predictions_persisted_before_labels"] is True
    assert outcome.artifact["score"] == {
        "correct": 2,
        "total": 2,
        "abstentions": 0,
        "errors": 0,
    }
    assert outcome.exposure_successor.events[-1].source == (
        f"{runner_module.PROTOCOL_ID}:plan:{outcome.plan.digest}"
    )
    assert SEALED not in outcome.exposure_successor.exposed_task_ids
    assert all(
        "contacts" not in scenario
        for entry in freeze["support_entries"]
        for scenario in entry["neutral_projection"]["scenarios"]
    )
    support_bytes = {
        entry["presentation_name"]: (
            _positive if entry["polarity"] == "positive" else _negative
        )
        for entry in freeze["support_entries"]
    }
    query_bytes = {
        item["query_id"]: (
            _positive if label["positive"] else _negative
        )
        for item, label in zip(
            predictions["entries"], outcome.artifact["labels"], strict=True
        )
    }
    assert cold_replay_relational_headless_run(
        plan=outcome.plan,
        exposure_successor=outcome.exposure_successor,
        freeze=freeze,
        predictions=predictions,
        final_run=outcome.artifact,
        support_png_bytes=support_bytes,
        query_png_bytes=query_bytes,
        release_authenticator=outcome.plan._release_authenticator,
    ) == outcome.artifact


def test_engineering_train_mode_uses_composite_closed_union_and_cold_replays_without_lean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = EXACT_UNUSED_TRAIN_ENGINEERING_TASK_IDS[0]
    outcome, accesses, heldout, calls, positive, negative = _run(
        tmp_path,
        proposal=ENGINEERING_GOOD,
        task=task,
        benchmark_mode=EXACT_UNUSED_TRAIN_ENGINEERING_MODE,
    )

    assert outcome.status == "complete"
    assert len(calls) == 1 and len(accesses) == 14
    assert set(accesses[-2:]) == heldout
    plan_data = outcome.plan.to_data()
    admission = plan_data["engineering_train_admission"]
    assert admission["mode"] == EXACT_UNUSED_TRAIN_ENGINEERING_MODE
    assert admission["historical_semantic_exposure_required"] == (
        "historically_exposed"
    )
    assert admission["semantic_unseen_required"] is False
    binding = plan_data["closed_predicate_binding"]
    oracle_identity = complete_closed_predicate_library_identity()
    assert binding["predicate_authority"] == "canonical-pure-python"
    assert binding["lean_required"] is False
    assert binding["semantic_checker_imported"] is False
    assert binding["member_count"] == 65_678
    assert binding["construction_id"] == oracle_identity.construction_id
    assert binding["library_source_digest"] == oracle_identity.source_digest
    assert binding["evaluator_digest"] == oracle_identity.evaluator_digest
    assert binding["construction_grid_digest"] == (
        oracle_identity.construction_grid_digest
    )
    assert binding["complete_member_digest"] == (
        oracle_identity.complete_member_digest
    )
    assert len(binding["complete_member_digest"]) == 64
    assert len(binding["library_source_digest"]) == 64
    assert len(binding["evaluator_digest"]) == 64
    assert outcome.freeze_path is not None and outcome.prediction_path is not None
    freeze = load_relational_artifact(outcome.freeze_path)
    predictions = load_relational_artifact(outcome.prediction_path)
    assert freeze["closed_predicate_binding"] == binding
    assert predictions["closed_predicate_binding"] == binding
    assert freeze["query"]["kind"] == "relational"
    support_bytes = {
        entry["presentation_name"]: (
            positive if entry["polarity"] == "positive" else negative
        )
        for entry in freeze["support_entries"]
    }
    query_bytes = {
        item["query_id"]: (positive if label["positive"] else negative)
        for item, label in zip(
            predictions["entries"], outcome.artifact["labels"], strict=True
        )
    }
    original_import = __import__

    def no_lean_import(name, *args, **kwargs):
        lowered = name.lower()
        if "semantic_checker" in lowered or lowered == "lean" or lowered.startswith(
            "lean."
        ):
            raise AssertionError(f"forbidden Lean/semantic checker import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", no_lean_import)
    assert cold_replay_relational_headless_run(
        plan=outcome.plan,
        exposure_successor=outcome.exposure_successor,
        freeze=freeze,
        predictions=predictions,
        final_run=outcome.artifact,
        support_png_bytes=support_bytes,
        query_png_bytes=query_bytes,
        release_authenticator=outcome.plan._release_authenticator,
    ) == outcome.artifact


def test_engineering_tag_parser_closes_direct_symmetry_and_rejects_code(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ShapeBongard_V2"
    task = EXACT_UNUSED_TRAIN_ENGINEERING_TASK_IDS[0]
    _write_task(root, task)
    release_authenticator, _descriptor, _archive = _release_authenticator(
        tmp_path, root
    )
    predecessor = ExposureLedger.create(CORPUS)
    plan = prepare_relational_headless_plan(
        task_id=task,
        split_index=_split(task),
        predecessor=predecessor,
        expected_exposure_predecessor_digest=predecessor.digest,
        expected_corpus_digest=CORPUS,
        expected_split_source_digest=SPLIT,
        seed="fixture-seed",
        exposure_observed_at="2026-08-07T12:00:00Z",
        expected_launcher_digest=LAUNCHER,
        release_authenticator=release_authenticator,
        label_nonce=LABEL_NONCE,
        benchmark_mode=EXACT_UNUSED_TRAIN_ENGINEERING_MODE,
    )
    assert plan._closed_library is not None
    direct = {
        **ENGINEERING_GOOD,
        "kind": "direct_counts",
        "relational": None,
        "direct_atom_0": {
            "catalog_key": "component.count",
            "comparison": "equal",
            "target_count": 2,
        },
    }
    symmetry = {
        **ENGINEERING_GOOD,
        "kind": "symmetry",
        "relational": None,
        "symmetry": {
            "metric": "symmetry.coverage_at_least",
            "threshold_ppm": 900_000,
        },
    }
    assert parse_closed_visual_proposal(
        direct, library=plan._closed_library
    ).kind.value == "direct_counts"
    assert parse_closed_visual_proposal(
        symmetry, library=plan._closed_library
    ).kind.value == "symmetry"
    with pytest.raises(RelationalHeadlessRunError, match="fields"):
        parse_closed_visual_proposal(
            {**direct, "python_source": "return True"},
            library=plan._closed_library,
        )


def test_closed_schema_parser_reachability_equals_complete_library() -> None:
    """Every proposer-reachable value is exactly one complete-library member."""

    schema = runner_module.closed_visual_proposal_schema()
    properties = schema["properties"]
    kind_values = properties["kind"]["enum"]
    assert kind_values == [item.value for item in closed_module.ClosedPredicateKind]

    relational_schema = properties["relational"]["anyOf"][1]["properties"]
    side_counts = tuple(relational_schema["numerator_side_count"]["enum"])
    assert side_counts == tuple(runner_module.ALLOWED_SIDE_COUNTS)
    assert tuple(relational_schema["denominator_side_count"]["enum"]) == (
        side_counts
    )
    ratio_ids = tuple(relational_schema["area_ratio"]["enum"])
    assert ratio_ids == runner_module._RATIO_IDS
    obliqueness = (
        None,
        *relational_schema["denominator_obliqueness_millidegrees"]["anyOf"][1][
            "enum"
        ],
    )
    assert obliqueness == (
        None,
        *runner_module.ALLOWED_OBLIQUENESS_THRESHOLDS_MILLIDEGREES,
    )

    atom_schema = properties["direct_atom_0"]["anyOf"][1]
    assert properties["direct_atom_1"] == properties["direct_atom_0"]
    assert properties["direct_atom_2"] == properties["direct_atom_0"]
    atom_properties = atom_schema["properties"]
    catalog_keys = tuple(atom_properties["catalog_key"]["enum"])
    assert catalog_keys == tuple(
        sorted(
            item.catalog_key
            for item in runner_module.DIRECT_VISUAL_ATOM_CATALOG.atoms
        )
    )
    comparisons = tuple(atom_properties["comparison"]["enum"])
    target_counts = tuple(atom_properties["target_count"]["enum"])
    assert comparisons == ("equal",)
    assert target_counts == tuple(range(1, 9))
    for spec in runner_module.DIRECT_VISUAL_ATOM_CATALOG.atoms:
        assert {
            (option.comparison, dict(option.arguments)["target_count"])
            for option in spec.allowed_options
        } == set(product(comparisons, target_counts))

    symmetry_schema = properties["symmetry"]["anyOf"][1]["properties"]
    symmetry_metrics = tuple(symmetry_schema["metric"]["enum"])
    symmetry_thresholds = tuple(symmetry_schema["threshold_ppm"]["enum"])
    assert symmetry_metrics == tuple(
        item.value for item in closed_module.SymmetryMetric
    )
    assert symmetry_thresholds == closed_module.SYMMETRY_THRESHOLDS_PPM

    frozen_index = runner_module._complete_closed_library()
    reachable: set[str] = set()
    counts = {"relational": 0, "direct_counts": 0, "symmetry": 0}

    def accept(
        kind: str,
        *,
        relational: Mapping[str, object] | None = None,
        atoms: tuple[Mapping[str, object], ...] = (),
        symmetry: Mapping[str, object] | None = None,
    ) -> None:
        slots: list[Mapping[str, object] | None] = [None, None, None]
        slots[: len(atoms)] = atoms
        predicate = parse_closed_visual_proposal(
            {
                "kind": kind,
                "relational": relational,
                "direct_atom_0": slots[0],
                "direct_atom_1": slots[1],
                "direct_atom_2": slots[2],
                "symmetry": symmetry,
                "rationale": "canonical exhaustive reachability witness",
            },
            library=frozen_index,
        )
        assert predicate.digest not in reachable
        reachable.add(predicate.digest)
        counts[kind] += 1

    for numerator, denominator, ratio, threshold in product(
        side_counts, side_counts, ratio_ids, obliqueness
    ):
        accept(
            "relational",
            relational={
                "numerator_side_count": numerator,
                "denominator_side_count": denominator,
                "area_ratio": ratio,
                "denominator_obliqueness_millidegrees": threshold,
            },
        )

    for arity in range(1, 4):
        for keys in combinations(catalog_keys, arity):
            for targets in product(target_counts, repeat=arity):
                accept(
                    "direct_counts",
                    atoms=tuple(
                        {
                            "catalog_key": key,
                            "comparison": "equal",
                            "target_count": target,
                        }
                        for key, target in zip(keys, targets, strict=True)
                    ),
                )

    for metric, threshold in product(symmetry_metrics, symmetry_thresholds):
        accept(
            "symmetry",
            symmetry={"metric": metric, "threshold_ppm": threshold},
        )

    assert counts == {
        "relational": 1_260,
        "direct_counts": 64_400,
        "symmetry": 18,
    }
    materialized = enumerate_complete_closed_predicates()
    materialized_digests = {item.digest for item in materialized}
    assert len(materialized) == len(materialized_digests) == 65_678
    assert reachable == materialized_digests

    first, second = catalog_keys[:2]
    ordered = tuple(
        {
            "catalog_key": key,
            "comparison": "equal",
            "target_count": 1,
        }
        for key in (first, second)
    )
    reversed_predicate = parse_closed_visual_proposal(
        {
            "kind": "direct_counts",
            "relational": None,
            "direct_atom_0": ordered[1],
            "direct_atom_1": ordered[0],
            "direct_atom_2": None,
            "symmetry": None,
            "rationale": "ordering canonicalization witness",
        },
        library=frozen_index,
    )
    ordered_predicate = parse_closed_visual_proposal(
        {
            "kind": "direct_counts",
            "relational": None,
            "direct_atom_0": ordered[0],
            "direct_atom_1": ordered[1],
            "direct_atom_2": None,
            "symmetry": None,
            "rationale": "ordering canonicalization witness",
        },
        library=frozen_index,
    )
    assert reversed_predicate == ordered_predicate
    with pytest.raises(RelationalHeadlessRunError, match="repeats"):
        parse_closed_visual_proposal(
            {
                "kind": "direct_counts",
                "relational": None,
                "direct_atom_0": ordered[0],
                "direct_atom_1": ordered[0],
                "direct_atom_2": None,
                "symmetry": None,
                "rationale": "duplicate capability rejection witness",
            },
            library=frozen_index,
        )


def test_engineering_mode_explicitly_rejects_sealed_task_before_pixels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ShapeBongard_V2"
    _write_task(root)
    release_authenticator, _descriptor, _archive = _release_authenticator(
        tmp_path, root
    )
    predecessor = ExposureLedger.create(CORPUS)
    with pytest.raises(RelationalHeadlessRunError, match="sealed task"):
        prepare_relational_headless_plan(
            task_id=EXPLICITLY_SEALED_ENGINEERING_TASK_ID,
            split_index=_split(EXPLICITLY_SEALED_ENGINEERING_TASK_ID),
            predecessor=predecessor,
            expected_exposure_predecessor_digest=predecessor.digest,
            expected_corpus_digest=CORPUS,
            expected_split_source_digest=SPLIT,
            seed="fixture-seed",
            exposure_observed_at="2026-08-07T12:00:00Z",
            expected_launcher_digest=LAUNCHER,
            release_authenticator=release_authenticator,
            label_nonce=LABEL_NONCE,
            benchmark_mode=EXACT_UNUSED_TRAIN_ENGINEERING_MODE,
        )


def test_forward_support_rejection_never_resolves_or_reads_queries_or_rerolls(
    tmp_path: Path,
) -> None:
    reversed_proposal = {
        **GOOD,
        "numerator_side_count": 4,
        "denominator_side_count": 3,
    }
    outcome, accesses, heldout, calls, _positive, _negative = _run(
        tmp_path, proposal=reversed_proposal
    )

    assert outcome.status == "support_rejected"
    assert len(calls) == 1
    assert len(accesses) == 12
    assert set(accesses).isdisjoint(heldout)
    assert outcome.prediction_path is None
    assert outcome.artifact["query_paths_resolved"] is False
    assert outcome.artifact["reroll_attempted"] is False


def test_malformed_model_payload_is_one_terminal_attempt_with_no_query_access(
    tmp_path: Path,
) -> None:
    malformed = {**GOOD, "polarity_flip": True}
    outcome, accesses, heldout, calls, _positive, _negative = _run(
        tmp_path, proposal=malformed
    )

    assert outcome.status == "terminal_failure"
    assert outcome.artifact["phase"] == "proposal-parse"
    assert len(calls) == 1
    assert len(accesses) == 12
    assert set(accesses).isdisjoint(heldout)
    assert outcome.artifact["query_labels_revealed"] is False
    assert outcome.artifact["reroll_attempted"] is False


def test_write_once_persistence_refuses_different_bytes(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    _write_once_durable(path, b"one")
    _write_once_durable(path, b"one")
    with pytest.raises(RelationalHeadlessRunError, match="refusing to overwrite"):
        _write_once_durable(path, b"two")


def test_plan_rejects_task_outside_historically_clean_strict_dev(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ShapeBongard_V2"
    _write_task(root)
    release_authenticator, _descriptor, _archive = _release_authenticator(
        tmp_path, root
    )
    predecessor = ExposureLedger.create(CORPUS)
    with pytest.raises(
        RelationalHeadlessRunError,
        match="historically-clean strict DEV",
    ):
        prepare_relational_headless_plan(
            task_id="ff_nact2_5_0000",
            split_index=SplitIndex(
                groups=(
                    ("test", (SEALED,)),
                    ("train", ("ff_nact2_5_0000",)),
                    ("val", ()),
                ),
                source_digest=SPLIT,
            ),
            predecessor=predecessor,
            expected_exposure_predecessor_digest=predecessor.digest,
            expected_corpus_digest=CORPUS,
            expected_split_source_digest=SPLIT,
            seed="fixture-seed",
            exposure_observed_at="2026-08-07T12:00:00Z",
            expected_launcher_digest=LAUNCHER,
            release_authenticator=release_authenticator,
            label_nonce=LABEL_NONCE,
        )


@pytest.mark.parametrize(
    "rationale",
    ("", " leading", "trailing ", "line\nbreak", "x" * 1_025),
)
def test_rationale_is_bounded_stripped_control_free_utf8(rationale: str) -> None:
    with pytest.raises(RelationalHeadlessRunError, match="rationale"):
        parse_relational_proposal({**GOOD, "rationale": rationale})


def test_resealed_transport_request_substitution_is_rejected(tmp_path: Path) -> None:
    outcome, _accesses, _heldout, _calls, _positive, _negative = _run(
        tmp_path, proposal=GOOD
    )
    assert outcome.freeze_path is not None
    freeze = load_relational_artifact(outcome.freeze_path)
    request = dict(freeze["transport_request"])
    request["model"] = "substituted-model"
    tampered = {
        **freeze,
        "transport_request": request,
        "transport_request_digest": canonical_digest(request),
    }
    tampered.pop("digest")
    tampered["digest"] = canonical_digest(tampered)
    with pytest.raises(RelationalHeadlessRunError, match="receipt|transport"):
        verify_relational_proposal_freeze(
            tampered,
            plan=outcome.plan,
            exposure_successor=outcome.exposure_successor,
        )


def test_replay_binds_plan_exposure_support_and_query_source_indices(
    tmp_path: Path,
) -> None:
    outcome, _accesses, _heldout, _calls, _positive, _negative = _run(
        tmp_path, proposal=GOOD
    )
    assert outcome.freeze_path is not None
    assert outcome.prediction_path is not None
    freeze = load_relational_artifact(outcome.freeze_path)
    predictions = load_relational_artifact(outcome.prediction_path)

    def reseal(value):
        body = copy.deepcopy(dict(value))
        body.pop("digest")
        return {**body, "digest": canonical_digest(body)}

    wrong_plan = dict(freeze)
    wrong_plan["plan_digest"] = "0" * 64
    with pytest.raises(RelationalHeadlessRunError, match="policy"):
        verify_relational_proposal_freeze(
            reseal(wrong_plan),
            plan=outcome.plan,
            exposure_successor=outcome.exposure_successor,
        )

    wrong_exposure = dict(freeze)
    wrong_exposure["exposure_successor_digest"] = "sha256:" + "0" * 64
    with pytest.raises(RelationalHeadlessRunError, match="policy"):
        verify_relational_proposal_freeze(
            reseal(wrong_exposure),
            plan=outcome.plan,
            exposure_successor=outcome.exposure_successor,
        )

    wrong_support = copy.deepcopy(freeze)
    wrong_support["support_entries"][0]["source_index"] = (
        outcome.plan._positive_query_index
    )
    with pytest.raises(
        RelationalHeadlessRunError,
        match="source indices|release panel receipt",
    ):
        verify_relational_proposal_freeze(
            reseal(wrong_support),
            plan=outcome.plan,
            exposure_successor=outcome.exposure_successor,
        )

    wrong_query = copy.deepcopy(predictions)
    wrong_query["entries"][0]["source_index"] = (
        wrong_query["entries"][0]["source_index"] + 1
    ) % 7
    with pytest.raises(RelationalHeadlessRunError, match="source index"):
        verify_relational_predictions(
            reseal(wrong_query),
            freeze=freeze,
            plan=outcome.plan,
            exposure_successor=outcome.exposure_successor,
        )


def test_transport_failure_is_exactly_one_terminal_attempt(tmp_path: Path) -> None:
    def fail(_prompt, _paths, _schema, _kwargs):
        raise RuntimeError("fixture transport failure")

    outcome, accesses, heldout, calls, _positive, _negative = _run(
        tmp_path, proposal=GOOD, transport_hook=fail
    )
    assert outcome.status == "terminal_failure"
    assert outcome.artifact["phase"] == "single-codex-proposal"
    assert len(calls) == 1
    assert len(accesses) == 12
    assert set(accesses).isdisjoint(heldout)
    assert outcome.artifact["reroll_attempted"] is False


def test_extracted_panel_must_match_pinned_release_before_transport(
    tmp_path: Path,
) -> None:
    def alter_support(root: Path, plan) -> None:
        index = plan.positive_support_indices[0]
        (root / "bd" / "images" / TASK / "1" / f"{index}.png").write_bytes(
            _panel(triangle_radius=18, quadrilateral_radius=22)
        )

    outcome, _accesses, _heldout, calls, _positive, _negative = _run(
        tmp_path,
        proposal=GOOD,
        before_run=alter_support,
    )
    assert calls == []
    assert outcome.status == "terminal_failure"
    assert outcome.artifact["phase"] == "support-extraction"
    assert "official release member" in outcome.artifact["error_message"]


def test_symlinked_panel_parent_is_rejected_before_observer_or_transport(
    tmp_path: Path,
) -> None:
    def symlink_parent(root: Path, _plan) -> None:
        label = root / "bd" / "images" / TASK / "1"
        real = label.with_name("1-real")
        label.rename(real)
        label.symlink_to(real, target_is_directory=True)

    outcome, accesses, _heldout, calls, _positive, _negative = _run(
        tmp_path,
        proposal=GOOD,
        before_run=symlink_parent,
    )
    assert calls == []
    assert accesses == []
    assert outcome.status == "terminal_failure"
    assert outcome.artifact["phase"] == "support-extraction"


def test_cold_replay_reauthenticates_zip_receipts(tmp_path: Path) -> None:
    outcome, _accesses, _heldout, _calls, positive, negative = _run(
        tmp_path, proposal=GOOD
    )
    assert outcome.freeze_path is not None
    assert outcome.prediction_path is not None
    freeze = load_relational_artifact(outcome.freeze_path)
    predictions = load_relational_artifact(outcome.prediction_path)
    assert all("release_panel_receipt" in item for item in freeze["support_entries"])
    assert all("release_panel_receipt" in item for item in predictions["entries"])
    assert [item["release_panel_receipt"] for item in predictions["entries"]] == [
        item["release_panel_receipt"]
        for item in outcome.artifact["selected_panel_manifest"]["query"]
    ]
    support_bytes = {
        item["presentation_name"]: (
            positive if item["polarity"] == "positive" else negative
        )
        for item in freeze["support_entries"]
    }
    labels = {
        item["query_id"]: item["positive"] for item in outcome.artifact["labels"]
    }
    query_bytes = {
        item["query_id"]: positive if labels[item["query_id"]] else negative
        for item in predictions["entries"]
    }
    tampered = copy.deepcopy(dict(outcome.artifact))
    tampered["selected_panel_manifest"]["query"][0][
        "release_panel_receipt"
    ]["relative_path"] = "bd/images/bd_asymmetric_goldfish_0000/1/6.png"
    tampered.pop("digest")
    tampered["digest"] = canonical_digest(tampered)
    with pytest.raises(RelationalHeadlessRunError, match="manifest|receipt"):
        cold_replay_relational_headless_run(
            plan=outcome.plan,
            exposure_successor=outcome.exposure_successor,
            freeze=freeze,
            predictions=predictions,
            final_run=tampered,
            support_png_bytes=support_bytes,
            query_png_bytes=query_bytes,
            release_authenticator=outcome.plan._release_authenticator,
        )


def test_cli_returns_nonzero_for_terminal_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ShapeBongard_V2"
    _write_task(root)
    release_authenticator, descriptor_path, archive_path = (
        _release_authenticator(tmp_path, root)
    )
    split_file = tmp_path / "split.json"
    split_file.write_text(
        '{"train":["bd_asymmetric_goldfish_0000"],"val":[],"test":["ff_nact5_0299"]}',
        encoding="utf-8",
    )
    split = SplitIndex.load(split_file)
    ledger = ExposureLedger.create(CORPUS)
    ledger_path = ledger.write_once(tmp_path / "ledger.json")

    class _Outcome:
        status = "terminal_failure"

        @staticmethod
        def to_data():
            return {"status": "terminal_failure"}

    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return _Outcome()

    monkeypatch.setattr(runner_module, "run_relational_headless", fake_run)
    monkeypatch.setattr(
        runner_module,
        "snapshot_cloud_policy_cache",
        lambda: runner_module.CloudPolicyCacheSnapshot(None),
    )
    code = runner_module.main(
        [
            "--corpus-root",
            str(tmp_path / "unused-corpus"),
            "--split-file",
            str(split_file),
            "--task-id",
            TASK,
            "--ledger-in",
            str(ledger_path),
            "--expected-ledger-digest",
            ledger.digest,
            "--expected-corpus-digest",
            CORPUS,
            "--expected-split-digest",
            split.source_digest,
            "--expected-release-digest",
            release_authenticator.release_descriptor_digest,
            "--release-descriptor-file",
            str(descriptor_path),
            "--release-archive",
            str(archive_path),
            "--seed",
            "fixture",
            "--exposure-observed-at",
            "2026-08-07T12:00:00Z",
            "--exposure-store",
            str(tmp_path / "exposure"),
            "--artifact-store",
            str(tmp_path / "artifacts"),
            "--expected-codex-launcher-sha256",
            LAUNCHER,
        ]
    )
    assert code == 2
    assert len(calls) == 1

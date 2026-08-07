from __future__ import annotations

import hashlib
import fcntl
from dataclasses import replace
from io import BytesIO
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping
import zipfile

from PIL import Image, ImageDraw
import pytest

from bongard.artifacts import canonical_digest
from bongard.corpus import SplitIndex
from bongard.exposure import ExposureLedger
from bongard.loop_scene_witnesses import extract_loop_scene_witnesses
from bongard.relational_headless_campaign import (
    RelationalCampaignTaskReplayInput,
    RelationalHeadlessCampaignError,
    cold_replay_durable_relational_headless_campaign,
    cold_replay_relational_headless_campaign,
    prepare_full_current_dev_campaign,
    prepare_exact_unused_train_engineering_campaign,
    run_relational_headless_campaign,
    verify_relational_headless_campaign_plan_artifact,
    write_relational_headless_campaign_plan,
)
from bongard.relational_headless_runner import (
    EXACT_UNUSED_TRAIN_ENGINEERING_MODE,
    EXACT_UNUSED_TRAIN_ENGINEERING_TASK_IDS,
    EXPLICITLY_SEALED_ENGINEERING_TASK_ID,
)
import bongard.relational_headless_campaign as campaign_module
import bongard.relational_headless_runner as runner_module
from bongard.relational_headless_runner import (
    ReleaseArchiveAuthenticator,
    load_relational_artifact,
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
RELEASE = "sha256:" + "6" * 64
SCHEDULE_SECRET = "7" * 64
TASKS = (
    "bd_asymmetric_goldfish_0000",
    "bd_asymmetric_clamp_0000",
)
SEALED = "ff_nact5_0299"
GOOD = {
    "numerator_side_count": 3,
    "denominator_side_count": 4,
    "area_ratio": "1/8",
    "denominator_obliqueness_millidegrees": None,
    "rationale": "small triangle and much larger quadrilateral",
}


def _split() -> SplitIndex:
    return SplitIndex(
        groups=(("test", (SEALED,)), ("train", TASKS), ("val", ())),
        source_digest=SPLIT,
    )


def _panel(*, triangle_radius: int, quadrilateral_radius: int) -> bytes:
    image = Image.new("RGB", (160, 160), "white")
    draw = ImageDraw.Draw(image)
    triangle_center = (42, 82)
    triangle = [
        (triangle_center[0], triangle_center[1] - triangle_radius),
        (
            triangle_center[0] - triangle_radius,
            triangle_center[1] + triangle_radius,
        ),
        (
            triangle_center[0] + triangle_radius,
            triangle_center[1] + triangle_radius,
        ),
    ]
    quad_center = (112, 82)
    radius = quadrilateral_radius
    quadrilateral = [
        (quad_center[0] - radius, quad_center[1] - radius),
        (quad_center[0] + radius, quad_center[1] - radius + 5),
        (quad_center[0] + radius - 4, quad_center[1] + radius),
        (quad_center[0] - radius + 3, quad_center[1] + radius - 5),
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


def _write_corpus(root: Path) -> tuple[bytes, bytes]:
    positive = _panel(triangle_radius=11, quadrilateral_radius=34)
    negative = _panel(triangle_radius=34, quadrilateral_radius=11)
    for task_id in TASKS:
        for label, payload in (("1", positive), ("0", negative)):
            directory = root / "bd" / "images" / task_id / label
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(7):
                (directory / f"{index}.png").write_bytes(payload)
    return positive, negative


def _release_authenticator(
    tmp_path: Path, root: Path
) -> ReleaseArchiveAuthenticator:
    archive_path = tmp_path / "ShapeBongard_V2.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as bundle:
        for panel in sorted(root.rglob("*.png")):
            bundle.write(
                panel,
                "ShapeBongard_V2/" + panel.relative_to(root).as_posix(),
            )
    archive_bytes = archive_path.read_bytes()
    descriptor = OfficialReleaseDescriptor(
        release_id="synthetic-relational-campaign-test",
        archive_filename=archive_path.name,
        archive_sha256="sha256:" + hashlib.sha256(archive_bytes).hexdigest(),
        archive_size_bytes=len(archive_bytes),
        split_filename="synthetic-split.json",
        split_sha256=SPLIT,
        split_size_bytes=1,
        upstream_repository="synthetic",
        upstream_commit="a" * 40,
        family_counts=(("bd", 2),),
        primary_split_counts=(("test", 1), ("train", 2), ("val", 0)),
        regime_counts=(),
        task_ids_sha256="sha256:" + "9" * 64,
        corpus_manifest_sha256=CORPUS,
    )
    descriptor_path = tmp_path / "release.json"
    descriptor_path.write_text(
        json.dumps(
            descriptor.to_dict(), sort_keys=True, separators=(",", ":")
        )
        + "\n",
        encoding="utf-8",
    )
    return ReleaseArchiveAuthenticator.load(
        release_descriptor_path=descriptor_path,
        expected_release_descriptor_digest=descriptor.digest,
        archive_path=archive_path,
    )


def _receipt(
    *,
    prompt: str,
    paths: tuple[str, ...],
    schema: Mapping[str, Any],
    model: str,
    reasoning_effort: str,
    proposal: Mapping[str, Any] = GOOD,
) -> CodexReceipt:
    prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    schema_digest = canonical_digest(schema)
    identities = []
    for path in paths:
        payload = Path(path).read_bytes()
        identities.append(
            {
                "name": Path(path).name,
                "byte_count": len(payload),
                "content_digest": hashlib.sha256(payload).hexdigest(),
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
        "structured_output_digest": canonical_digest(proposal),
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


def _cached_extractor():
    cache = {}

    def extract(payload: bytes):
        digest = hashlib.sha256(payload).hexdigest()
        if digest not in cache:
            cache[digest] = extract_loop_scene_witnesses(payload)
        return cache[digest]

    return extract


def _campaign_fixture(tmp_path: Path, *, tag: str):
    root = tmp_path / "ShapeBongard_V2"
    _write_corpus(root)
    release_authenticator = _release_authenticator(tmp_path, root)
    predecessor = ExposureLedger.create(CORPUS)
    plan = prepare_full_current_dev_campaign(
        artifact_store=tmp_path / "artifacts",
        split_index=_split(),
        predecessor=predecessor,
        expected_release_descriptor_digest=(
            release_authenticator.release_descriptor_digest
        ),
        release_authenticator=release_authenticator,
        expected_corpus_digest=CORPUS,
        expected_split_source_digest=SPLIT,
        expected_exposure_predecessor_digest=predecessor.digest,
        campaign_seed=f"fixture-{tag}",
        selection_seed_provenance=f"synthetic {tag} seed",
        schedule_secret=SCHEDULE_SECRET,
        exposure_observed_at="2026-08-07T16:00:00Z",
        expected_task_count=2,
        expected_launcher_digest=LAUNCHER,
    )
    return root, predecessor, plan


def _attempt_replay_artifacts(
    artifact_store: Path,
    *,
    plan,
    ordinal: int,
    task_plan,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    claim_path, terminal_path = campaign_module._attempt_paths(
        artifact_store,
        plan=plan,
        ordinal=ordinal,
        task_plan=task_plan,
    )
    return (
        json.loads(claim_path.read_text(encoding="utf-8")),
        json.loads(terminal_path.read_text(encoding="utf-8")),
    )


def test_metadata_only_plan_is_full_strict_dev_and_fails_closed_on_count(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ShapeBongard_V2"
    _write_corpus(root)
    release_authenticator = _release_authenticator(tmp_path, root)
    predecessor = ExposureLedger.create(CORPUS)
    plan = prepare_full_current_dev_campaign(
        artifact_store=tmp_path / "artifacts",
        split_index=_split(),
        predecessor=predecessor,
        expected_release_descriptor_digest=(
            release_authenticator.release_descriptor_digest
        ),
        release_authenticator=release_authenticator,
        expected_corpus_digest=CORPUS,
        expected_split_source_digest=SPLIT,
        expected_exposure_predecessor_digest=predecessor.digest,
        campaign_seed="fixture-campaign",
        selection_seed_provenance="synthetic unit-test seed",
        schedule_secret=SCHEDULE_SECRET,
        exposure_observed_at="2026-08-07T15:00:00Z",
        expected_task_count=2,
        expected_launcher_digest=LAUNCHER,
    )
    assert set(plan.task_ids) == set(TASKS)
    assert len(plan.task_plans) == 2
    assert all(
        item.to_data()["strict_dev_admission"]["required_semantic_cohort"]
        == "dev"
        for item in plan.task_plans
    )
    assert plan.to_data()["execution_policy"][
        "cohort_exposure"
    ] == "one-atomic-ledger-edge-before-task-1"
    assert campaign_module.CAMPAIGN_EXPOSURE_PHASE == (
        runner_module.CAMPAIGN_AUTHORIZATION_PHASE
    )
    assert campaign_module.CAMPAIGN_EXPOSURE_ACTOR == (
        runner_module.CAMPAIGN_AUTHORIZATION_ACTOR
    )
    assert campaign_module.CAMPAIGN_EXPOSURE_PURPOSE == (
        runner_module.CAMPAIGN_AUTHORIZATION_PURPOSE
    )
    public_plan = plan.to_data()
    store_binding = public_plan["artifact_attempt_store_binding"]
    assert store_binding["normalized_absolute_path"] == os.path.realpath(
        tmp_path / "artifacts"
    )
    assert store_binding["normalization"] == (
        campaign_module.ARTIFACT_STORE_NORMALIZATION
    )
    assert store_binding["path_substitution_before_execution"] == "rejected"
    assert "deletion or copying" in store_binding[
        "residual_operator_filesystem_trust"
    ]
    assert SCHEDULE_SECRET not in str(public_plan)
    assert public_plan["schedule_secret_publicly_disclosed"] is False
    for task in plan.task_plans:
        public_seed_guess = campaign_module._derived_secret(
            plan._seed, "task-seed", task.task_id
        )
        assert hashlib.sha256(public_seed_guess.encode()).hexdigest() != (
            task.seed_digest
        )
        public_task = task.to_data()
        assert "support_selection_commitment" not in public_task
        assert "support_selection_opening" not in public_task
        hiding = public_task["support_selection_hiding_commitment"]
        # The former 49-value unkeyed search can neither validate nor recover
        # a schedule from the public plan.  A fixed wrong 256-bit key also
        # validates none of the candidates.
        for positive_query_index in range(7):
            for negative_query_index in range(7):
                candidate = runner_module._support_selection_data(
                    task.task_id,
                    tuple(
                        index
                        for index in range(7)
                        if index != positive_query_index
                    ),
                    tuple(
                        index
                        for index in range(7)
                        if index != negative_query_index
                    ),
                )
                assert canonical_digest(candidate) != hiding
                assert (
                    runner_module._support_selection_hiding_commitment(
                        candidate, "0" * 64
                    )
                    != hiding
                )

    wrong_secret_plan = prepare_full_current_dev_campaign(
        artifact_store=tmp_path / "artifacts",
        split_index=_split(),
        predecessor=predecessor,
        expected_release_descriptor_digest=(
            release_authenticator.release_descriptor_digest
        ),
        release_authenticator=release_authenticator,
        expected_corpus_digest=CORPUS,
        expected_split_source_digest=SPLIT,
        expected_exposure_predecessor_digest=predecessor.digest,
        campaign_seed="fixture-campaign",
        selection_seed_provenance="synthetic unit-test seed",
        schedule_secret="8" * 64,
        exposure_observed_at="2026-08-07T15:00:00Z",
        expected_task_count=2,
        expected_launcher_digest=LAUNCHER,
    )
    artifact = {**public_plan, "digest": canonical_digest(public_plan)}
    public_path = write_relational_headless_campaign_plan(
        tmp_path / "campaign.plan.json", plan=plan
    )
    assert json.loads(public_path.read_text()) == artifact
    assert SCHEDULE_SECRET not in public_path.read_text()
    with pytest.raises(RelationalHeadlessCampaignError, match="differs"):
        verify_relational_headless_campaign_plan_artifact(
            artifact, plan=wrong_secret_plan
        )

    with pytest.raises(RelationalHeadlessCampaignError, match="capacity"):
        prepare_full_current_dev_campaign(
            artifact_store=tmp_path / "artifacts",
            split_index=_split(),
            predecessor=predecessor,
            expected_release_descriptor_digest=(
                release_authenticator.release_descriptor_digest
            ),
            release_authenticator=release_authenticator,
            expected_corpus_digest=CORPUS,
            expected_split_source_digest=SPLIT,
            expected_exposure_predecessor_digest=predecessor.digest,
            campaign_seed="fixture-campaign",
            selection_seed_provenance="synthetic unit-test seed",
            schedule_secret=SCHEDULE_SECRET,
            exposure_observed_at="2026-08-07T15:00:00Z",
            expected_task_count=1,
            expected_launcher_digest=LAUNCHER,
        )


def test_metadata_only_engineering_campaign_is_exact_fixed_train_allowlist(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ShapeBongard_V2"
    _write_corpus(root)
    release_authenticator = _release_authenticator(tmp_path, root)
    predecessor = ExposureLedger.create(CORPUS)
    split = SplitIndex(
        groups=(
            ("test", (EXPLICITLY_SEALED_ENGINEERING_TASK_ID,)),
            ("train", EXACT_UNUSED_TRAIN_ENGINEERING_TASK_IDS),
            ("val", ()),
        ),
        source_digest=SPLIT,
    )
    plan = prepare_exact_unused_train_engineering_campaign(
        artifact_store=tmp_path / "engineering-artifacts",
        split_index=split,
        predecessor=predecessor,
        expected_release_descriptor_digest=(
            release_authenticator.release_descriptor_digest
        ),
        release_authenticator=release_authenticator,
        expected_corpus_digest=CORPUS,
        expected_split_source_digest=SPLIT,
        expected_exposure_predecessor_digest=predecessor.digest,
        campaign_seed="fixture-engineering-campaign",
        selection_seed_provenance="synthetic unit-test engineering seed",
        schedule_secret=SCHEDULE_SECRET,
        exposure_observed_at="2026-08-07T15:00:00Z",
        expected_task_count=5,
        expected_launcher_digest=LAUNCHER,
    )

    assert plan.campaign_mode == EXACT_UNUSED_TRAIN_ENGINEERING_MODE
    assert plan.task_ids == EXACT_UNUSED_TRAIN_ENGINEERING_TASK_IDS
    assert EXPLICITLY_SEALED_ENGINEERING_TASK_ID not in plan.task_ids
    assert plan.to_data()["selection"]["semantic_unseen_asserted"] is False
    assert all(item.split == "train" for item in plan.task_plans)
    assert all(
        item.to_data()["engineering_train_admission"][
            "historical_semantic_exposure_required"
        ]
        == "historically_exposed"
        for item in plan.task_plans
    )
    bindings = [item.closed_predicate_binding for item in plan.task_plans]
    assert all(item == bindings[0] for item in bindings)
    assert bindings[0]["member_count"] == 65_678
    assert bindings[0]["lean_required"] is False


def test_engineering_campaign_execution_routes_closed_mode_without_pixels_or_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ShapeBongard_V2"
    _write_corpus(root)
    release_authenticator = _release_authenticator(tmp_path, root)
    predecessor = ExposureLedger.create(CORPUS)
    split = SplitIndex(
        groups=(
            ("test", (EXPLICITLY_SEALED_ENGINEERING_TASK_ID,)),
            ("train", EXACT_UNUSED_TRAIN_ENGINEERING_TASK_IDS),
            ("val", ()),
        ),
        source_digest=SPLIT,
    )
    plan = prepare_exact_unused_train_engineering_campaign(
        artifact_store=tmp_path / "engineering-artifacts",
        split_index=split,
        predecessor=predecessor,
        expected_release_descriptor_digest=(
            release_authenticator.release_descriptor_digest
        ),
        release_authenticator=release_authenticator,
        expected_corpus_digest=CORPUS,
        expected_split_source_digest=SPLIT,
        expected_exposure_predecessor_digest=predecessor.digest,
        campaign_seed="fixture-engineering-execute",
        selection_seed_provenance="synthetic engineering dispatch seed",
        schedule_secret=SCHEDULE_SECRET,
        exposure_observed_at="2026-08-07T15:30:00Z",
        expected_task_count=5,
        expected_launcher_digest=LAUNCHER,
    )
    by_id = {item.task_id: item for item in plan.task_plans}
    calls: list[str] = []

    def fake_task_runner(**kwargs):
        task_plan = by_id[kwargs["task_id"]]
        assert kwargs["benchmark_mode"] == EXACT_UNUSED_TRAIN_ENGINEERING_MODE
        assert kwargs["closed_library"] is task_plan._closed_library
        assert kwargs["extractor"] is campaign_module.extract_exact_panel_witness_packet
        assert kwargs["packet_verifier"] is (
            campaign_module.verify_exact_panel_witness_packet
        )
        calls.append(task_plan.task_id)
        successor = kwargs["precommitted_exposure_successor"]
        artifact = campaign_module._seal(
            {
                "schema": campaign_module.FAILURE_SCHEMA,
                "protocol_id": runner_module.PROTOCOL_ID,
                "benchmark_mode": EXACT_UNUSED_TRAIN_ENGINEERING_MODE,
                "status": "terminal_failure",
                "phase": "synthetic-no-pixel-dispatch",
                "error_type": "SyntheticDispatch",
                "error_message": "intentional synthetic terminal",
                "plan_digest": task_plan.digest,
                "exposure_successor_digest": successor.digest,
                "freeze_digest": None,
                "prediction_digest": None,
                "query_labels_revealed": False,
                "reroll_attempted": False,
            }
        )
        return runner_module.RelationalHeadlessOutcome(
            "terminal_failure",
            task_plan,
            successor,
            tmp_path / "synthetic-plan.json",
            Path(kwargs["precommitted_exposure_path"]),
            tmp_path / f"{task_plan.task_id}.terminal.json",
            artifact,
        )

    monkeypatch.setattr(campaign_module, "run_relational_headless", fake_task_runner)
    outcome = run_relational_headless_campaign(
        corpus_root=root,
        plan=plan,
        split_index=split,
        predecessor=predecessor,
        exposure_store=tmp_path / "engineering-exposure",
        artifact_store=tmp_path / "engineering-artifacts",
        transport=lambda *_args, **_kwargs: pytest.fail("model call is forbidden"),
        png_reader=lambda *_args, **_kwargs: pytest.fail("pixel read is forbidden"),
    )
    assert tuple(calls) == EXACT_UNUSED_TRAIN_ENGINEERING_TASK_IDS
    assert outcome.report["campaign_mode"] == EXACT_UNUSED_TRAIN_ENGINEERING_MODE
    assert outcome.report["status_counts"]["terminal_failure"] == 5


def test_campaign_atomically_exposes_runs_aggregates_and_cold_replays(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ShapeBongard_V2"
    positive, negative = _write_corpus(root)
    release_authenticator = _release_authenticator(tmp_path, root)
    predecessor = ExposureLedger.create(CORPUS)
    plan = prepare_full_current_dev_campaign(
        artifact_store=tmp_path / "artifacts",
        split_index=_split(),
        predecessor=predecessor,
        expected_release_descriptor_digest=(
            release_authenticator.release_descriptor_digest
        ),
        release_authenticator=release_authenticator,
        expected_corpus_digest=CORPUS,
        expected_split_source_digest=SPLIT,
        expected_exposure_predecessor_digest=predecessor.digest,
        campaign_seed="fixture-campaign-run",
        selection_seed_provenance="synthetic integration-test seed",
        schedule_secret=SCHEDULE_SECRET,
        exposure_observed_at="2026-08-07T15:01:00Z",
        expected_task_count=2,
        expected_launcher_digest=LAUNCHER,
    )
    exposure_store = tmp_path / "exposure"
    artifact_store = tmp_path / "artifacts"
    opened: list[Path] = []
    calls = 0

    def reader(path: Path) -> bytes:
        assert len(tuple(exposure_store.glob("*.exposure.json"))) == 1
        assert len(
            tuple(artifact_store.glob("*.relational-headless-campaign-plan.json"))
        ) == 1
        assert len(
            tuple(artifact_store.glob("*.relational-headless-plan.json"))
        ) == 2
        opened.append(path)
        return path.read_bytes()

    def transport(prompt, paths, schema, **kwargs):
        nonlocal calls
        calls += 1
        receipt = _receipt(
            prompt=prompt,
            paths=tuple(paths),
            schema=schema,
            model=kwargs["model"],
            reasoning_effort=kwargs["reasoning_effort"],
        )
        return CodexStructuredResult(dict(GOOD), receipt)

    outcome = run_relational_headless_campaign(
        corpus_root=root,
        plan=plan,
        split_index=_split(),
        predecessor=predecessor,
        exposure_store=exposure_store,
        artifact_store=artifact_store,
        transport=transport,
        png_reader=reader,
        extractor=_cached_extractor(),
        packet_verifier=lambda packet, **_kwargs: packet,
    )

    assert calls == 2
    assert len(opened) == 28
    assert len(outcome.exposure_successor.events) == 1
    assert set(outcome.exposure_successor.events[0].task_ids) == set(TASKS)
    assert tuple(item.plan.task_id for item in outcome.task_outcomes) == plan.task_ids
    assert {item.status for item in outcome.task_outcomes} == {"complete"}
    assert outcome.report["task_count"] == 2
    assert outcome.report["joint_task_accuracy"] == {
        "correct": 2,
        "denominator": 2,
    }
    assert outcome.report["fixed_denominator_query_score"] == {
        "correct": 4,
        "denominator": 4,
        "unreleased_or_incorrect": 0,
    }
    assert outcome.report["all_terminal_outcomes_in_denominator"] is True

    replay_inputs = {}
    for ordinal, task_outcome in enumerate(outcome.task_outcomes):
        assert task_outcome.freeze_path is not None
        assert task_outcome.prediction_path is not None
        freeze = load_relational_artifact(task_outcome.freeze_path)
        predictions = load_relational_artifact(task_outcome.prediction_path)
        support_bytes = {
            entry["presentation_name"]: (
                positive if entry["polarity"] == "positive" else negative
            )
            for entry in freeze["support_entries"]
        }
        labels = {
            item["query_id"]: item["positive"]
            for item in task_outcome.artifact["labels"]
        }
        query_bytes = {
            entry["query_id"]: (
                positive if labels[entry["query_id"]] else negative
            )
            for entry in predictions["entries"]
        }
        attempt_claim, attempt_terminal = _attempt_replay_artifacts(
            artifact_store,
            plan=plan,
            ordinal=ordinal,
            task_plan=task_outcome.plan,
        )
        replay_inputs[task_outcome.plan.task_id] = (
            RelationalCampaignTaskReplayInput(
                freeze=freeze,
                predictions=predictions,
                support_png_bytes=support_bytes,
                query_png_bytes=query_bytes,
                attempt_claim=attempt_claim,
                attempt_terminal=attempt_terminal,
            )
        )
    calls_before_durable_replay = calls
    durable = cold_replay_durable_relational_headless_campaign(
        corpus_root=root,
        artifact_store=artifact_store,
        campaign_outcome=outcome,
        split_index=_split(),
        predecessor=predecessor,
        png_reader=reader,
    )
    assert calls == calls_before_durable_replay
    assert durable.receipt_path.is_file()
    receipt = durable.receipt
    assert receipt["complete_runs_replayed"] == 2
    assert receipt["all_tasks_accounted_for"] is True
    assert receipt["proposer_or_model_called_during_replay"] is False
    assert durable.receipt["complete_runs_replayed"] == 2
    assert durable.receipt["proposer_or_model_called_during_replay"] is False
    assert durable.receipt["source_identities"] == {
        "campaign_python_source_digest": plan.campaign_python_source_digest,
        "runner_python_source_digest": plan.runner_python_source_digest,
        "task_protocol_digest_set": sorted(
            {item.protocol_digest for item in plan.task_plans}
        ),
        "python_only": True,
        "lean_required": False,
        "semantic_checker_imported": False,
    }
    first_task_id = plan.task_ids[0]
    terminal_body = dict(replay_inputs[first_task_id].attempt_terminal)
    terminal_body.pop("digest")
    terminal_body["status"] = "support_rejected"
    tampered_terminal = {
        **terminal_body,
        "digest": canonical_digest(terminal_body),
    }
    tampered_replay_inputs = {
        **replay_inputs,
        first_task_id: replace(
            replay_inputs[first_task_id],
            attempt_terminal=tampered_terminal,
        ),
    }
    with pytest.raises(
        RelationalHeadlessCampaignError,
        match="attempt terminal replay",
    ):
        cold_replay_relational_headless_campaign(
            plan=plan,
            split_index=_split(),
            predecessor=predecessor,
            exposure_successor=outcome.exposure_successor,
            task_outcomes=outcome.task_outcomes,
            campaign_run=outcome.report,
            replay_inputs=tampered_replay_inputs,
        )
    edge_values = {
        "phase": campaign_module.CAMPAIGN_EXPOSURE_PHASE,
        "actor": campaign_module.CAMPAIGN_EXPOSURE_ACTOR,
        "purpose": campaign_module.CAMPAIGN_EXPOSURE_PURPOSE,
    }
    groups = _split().canonical_groups
    for field in ("phase", "actor", "purpose"):
        mutated = {**edge_values, field: "adversarial-substitution"}
        bad_successor = predecessor.record(
            phase=mutated["phase"],
            actor=mutated["actor"],
            purpose=mutated["purpose"],
            task_ids=plan.task_ids,
            source=(
                f"{campaign_module.CAMPAIGN_PROTOCOL_ID}:plan:{plan.digest}"
            ),
            observed_at=plan.exposure_observed_at,
            known_task_ids=(
                set(groups["train"]) | set(groups["val"]) | set(groups["test"])
            ),
            sealed_task_ids=groups["test"],
            require_unseen=True,
        )
        with pytest.raises(RelationalHeadlessCampaignError, match="atomic edge"):
            cold_replay_relational_headless_campaign(
                plan=plan,
                split_index=_split(),
                predecessor=predecessor,
                exposure_successor=bad_successor,
                task_outcomes=outcome.task_outcomes,
                campaign_run=outcome.report,
                replay_inputs=replay_inputs,
            )


def test_campaign_continues_after_failures_and_keeps_full_denominator(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ShapeBongard_V2"
    _write_corpus(root)
    release_authenticator = _release_authenticator(tmp_path, root)
    predecessor = ExposureLedger.create(CORPUS)
    plan = prepare_full_current_dev_campaign(
        artifact_store=tmp_path / "artifacts",
        split_index=_split(),
        predecessor=predecessor,
        expected_release_descriptor_digest=(
            release_authenticator.release_descriptor_digest
        ),
        release_authenticator=release_authenticator,
        expected_corpus_digest=CORPUS,
        expected_split_source_digest=SPLIT,
        expected_exposure_predecessor_digest=predecessor.digest,
        campaign_seed="fixture-campaign-failures",
        selection_seed_provenance="synthetic failure-path seed",
        schedule_secret=SCHEDULE_SECRET,
        exposure_observed_at="2026-08-07T15:02:00Z",
        expected_task_count=2,
        expected_launcher_digest=LAUNCHER,
    )
    calls = 0

    def failing_transport(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic proposer failure")

    artifact_store = tmp_path / "artifacts"
    outcome = run_relational_headless_campaign(
        corpus_root=root,
        plan=plan,
        split_index=_split(),
        predecessor=predecessor,
        exposure_store=tmp_path / "exposure",
        artifact_store=artifact_store,
        transport=failing_transport,
        png_reader=lambda path: path.read_bytes(),
        extractor=_cached_extractor(),
        packet_verifier=lambda packet, **_kwargs: packet,
    )

    assert calls == 2
    assert [item.status for item in outcome.task_outcomes] == [
        "terminal_failure",
        "terminal_failure",
    ]
    assert outcome.report["status_counts"] == {
        "complete": 0,
        "support_rejected": 0,
        "terminal_failure": 2,
    }
    assert outcome.report["joint_task_accuracy"] == {
        "correct": 0,
        "denominator": 2,
    }
    assert outcome.report["fixed_denominator_query_score"] == {
        "correct": 0,
        "denominator": 4,
        "unreleased_or_incorrect": 4,
    }
    assert all(
        task["reroll_attempted"] is False for task in outcome.report["tasks"]
    )
    replay_inputs = {}
    for ordinal, task_outcome in enumerate(outcome.task_outcomes):
        attempt_claim, attempt_terminal = _attempt_replay_artifacts(
            artifact_store,
            plan=plan,
            ordinal=ordinal,
            task_plan=task_outcome.plan,
        )
        replay_inputs[task_outcome.plan.task_id] = (
            RelationalCampaignTaskReplayInput(
                attempt_claim=attempt_claim,
                attempt_terminal=attempt_terminal,
            )
        )
    replay = cold_replay_relational_headless_campaign(
        plan=plan,
        split_index=_split(),
        predecessor=predecessor,
        exposure_successor=outcome.exposure_successor,
        task_outcomes=outcome.task_outcomes,
        campaign_run=outcome.report,
        replay_inputs=replay_inputs,
    )
    assert replay["terminal_failures_integrity_verified"] == 2
    calls_before_durable_replay = calls
    durable = cold_replay_durable_relational_headless_campaign(
        corpus_root=root,
        artifact_store=artifact_store,
        campaign_outcome=outcome,
        split_index=_split(),
        predecessor=predecessor,
        png_reader=lambda path: path.read_bytes(),
    )
    assert calls == calls_before_durable_replay
    assert durable.receipt["terminal_failures_integrity_verified"] == 2

    alternate_store = tmp_path / "rerun-artifacts"
    with pytest.raises(
        RelationalHeadlessCampaignError,
        match="artifact/attempt store differs.*plan binding",
    ):
        run_relational_headless_campaign(
            corpus_root=root,
            plan=plan,
            split_index=_split(),
            predecessor=predecessor,
            exposure_store=tmp_path / "rerun-exposure",
            artifact_store=alternate_store,
            transport=failing_transport,
            png_reader=lambda path: path.read_bytes(),
        )
    assert calls == calls_before_durable_replay
    assert not alternate_store.exists()

    first = outcome.task_outcomes[0]
    bad_body = dict(first.artifact)
    bad_body.pop("digest")
    bad_body["query_labels_revealed"] = True
    bad_artifact = {**bad_body, "digest": canonical_digest(bad_body)}
    bad_outcomes = (
        replace(first, artifact=bad_artifact),
        *outcome.task_outcomes[1:],
    )
    with pytest.raises(
        RelationalHeadlessCampaignError,
        match="label reveal|attempt terminal replay",
    ):
        cold_replay_relational_headless_campaign(
            plan=plan,
            split_index=_split(),
            predecessor=predecessor,
            exposure_successor=outcome.exposure_successor,
            task_outcomes=bad_outcomes,
            campaign_run=outcome.report,
            replay_inputs=replay_inputs,
        )


def test_durable_cold_replay_handles_support_rejected_inventory_without_model(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ShapeBongard_V2"
    _write_corpus(root)
    release_authenticator = _release_authenticator(tmp_path, root)
    predecessor = ExposureLedger.create(CORPUS)
    plan = prepare_full_current_dev_campaign(
        artifact_store=tmp_path / "artifacts",
        split_index=_split(),
        predecessor=predecessor,
        expected_release_descriptor_digest=(
            release_authenticator.release_descriptor_digest
        ),
        release_authenticator=release_authenticator,
        expected_corpus_digest=CORPUS,
        expected_split_source_digest=SPLIT,
        expected_exposure_predecessor_digest=predecessor.digest,
        campaign_seed="fixture-campaign-support-rejected",
        selection_seed_provenance="synthetic support rejection seed",
        schedule_secret=SCHEDULE_SECRET,
        exposure_observed_at="2026-08-07T15:02:30Z",
        expected_task_count=2,
        expected_launcher_digest=LAUNCHER,
    )
    reversed_proposal = {
        **GOOD,
        "numerator_side_count": 4,
        "denominator_side_count": 3,
    }
    model_calls = 0

    def transport(prompt, paths, schema, **kwargs):
        nonlocal model_calls
        model_calls += 1
        receipt = _receipt(
            prompt=prompt,
            paths=tuple(paths),
            schema=schema,
            model=kwargs["model"],
            reasoning_effort=kwargs["reasoning_effort"],
            proposal=reversed_proposal,
        )
        return CodexStructuredResult(dict(reversed_proposal), receipt)

    artifact_store = tmp_path / "artifacts"
    outcome = run_relational_headless_campaign(
        corpus_root=root,
        plan=plan,
        split_index=_split(),
        predecessor=predecessor,
        exposure_store=tmp_path / "exposure",
        artifact_store=artifact_store,
        transport=transport,
        png_reader=lambda path: path.read_bytes(),
        extractor=_cached_extractor(),
        packet_verifier=lambda packet, **_kwargs: packet,
    )
    assert model_calls == 2
    assert {item.status for item in outcome.task_outcomes} == {
        "support_rejected"
    }
    durable = cold_replay_durable_relational_headless_campaign(
        corpus_root=root,
        artifact_store=artifact_store,
        campaign_outcome=outcome,
        split_index=_split(),
        predecessor=predecessor,
        png_reader=lambda path: path.read_bytes(),
    )
    assert model_calls == 2
    assert durable.receipt["support_rejections_replayed"] == 2
    assert durable.receipt["complete_runs_replayed"] == 0
    assert durable.receipt["proposer_or_model_called_during_replay"] is False


def test_crash_after_claim_is_never_reproposed_on_resume(tmp_path: Path) -> None:
    root, predecessor, plan = _campaign_fixture(tmp_path, tag="claim-crash")
    exposure_store = tmp_path / "exposure"
    artifact_store = tmp_path / "artifacts"
    crash_calls = 0

    def crash_transport(*_args, **_kwargs):
        nonlocal crash_calls
        crash_calls += 1
        claims = tuple(artifact_store.rglob("*.claimed.json"))
        assert len(claims) == 1
        claim = json.loads(claims[0].read_text(encoding="utf-8"))
        assert claim["state"] == "CLAIMED"
        assert claim["task_id"] == plan.task_ids[0]
        journal_mode = os.lstat(claims[0].parent).st_mode
        assert stat.S_ISDIR(journal_mode)
        assert journal_mode & 0o077 == 0
        with pytest.raises(
            RelationalHeadlessCampaignError, match="already running"
        ):
            campaign_module._acquire_campaign_lock(artifact_store, plan)
        raise KeyboardInterrupt("synthetic crash after durable claim")

    with pytest.raises(KeyboardInterrupt, match="after durable claim"):
        run_relational_headless_campaign(
            corpus_root=root,
            plan=plan,
            split_index=_split(),
            predecessor=predecessor,
            exposure_store=exposure_store,
            artifact_store=artifact_store,
            transport=crash_transport,
            png_reader=lambda path: path.read_bytes(),
            extractor=_cached_extractor(),
            packet_verifier=lambda packet, **_kwargs: packet,
        )
    assert crash_calls == 1
    assert len(tuple(artifact_store.rglob("*.claimed.json"))) == 1
    assert tuple(artifact_store.rglob("*.terminal.json")) == ()

    resume_calls = 0

    def one_remaining_failure(*_args, **_kwargs):
        nonlocal resume_calls
        resume_calls += 1
        raise RuntimeError("second task proposer failure")

    resumed = run_relational_headless_campaign(
        corpus_root=root,
        plan=plan,
        split_index=_split(),
        predecessor=predecessor,
        exposure_store=exposure_store,
        artifact_store=artifact_store,
        transport=one_remaining_failure,
        png_reader=lambda path: path.read_bytes(),
        extractor=_cached_extractor(),
        packet_verifier=lambda packet, **_kwargs: packet,
    )
    assert resume_calls == 1
    assert resumed.task_outcomes[0].artifact["phase"] == (
        "campaign-resume-incomplete-claimed-attempt"
    )
    assert resumed.report["task_count"] == 2
    assert resumed.report["joint_task_accuracy"]["denominator"] == 2
    assert resumed.report["tasks"][0]["attempt_resumed"] is True
    assert resumed.report["tasks"][1]["attempt_resumed"] is False
    assert all(
        item["attempt_terminal_digest"] is not None
        for item in resumed.report["tasks"]
    )

    def forbidden_transport(*_args, **_kwargs):
        raise AssertionError("a terminal claimed task was reproposed")

    replayed_resume = run_relational_headless_campaign(
        corpus_root=root,
        plan=plan,
        split_index=_split(),
        predecessor=predecessor,
        exposure_store=exposure_store,
        artifact_store=artifact_store,
        transport=forbidden_transport,
        png_reader=lambda path: path.read_bytes(),
        extractor=_cached_extractor(),
        packet_verifier=lambda packet, **_kwargs: packet,
    )
    assert all(
        item["attempt_resumed"] is True
        for item in replayed_resume.report["tasks"]
    )
    durable = cold_replay_durable_relational_headless_campaign(
        corpus_root=root,
        artifact_store=artifact_store,
        campaign_outcome=replayed_resume,
        split_index=_split(),
        predecessor=predecessor,
        png_reader=lambda path: path.read_bytes(),
    )
    assert durable.receipt["terminal_failures_integrity_verified"] == 2
    assert durable.receipt["proposer_or_model_called_during_replay"] is False


def test_terminal_journal_failure_is_terminalized_and_loop_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, predecessor, plan = _campaign_fixture(tmp_path, tag="terminal-write")
    original = campaign_module._persist_attempt_terminal
    attempts = 0

    def fail_once(path, terminal):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("synthetic terminal journal failure")
        return original(path, terminal)

    monkeypatch.setattr(campaign_module, "_persist_attempt_terminal", fail_once)
    calls = 0

    def transport(prompt, paths, schema, **kwargs):
        nonlocal calls
        calls += 1
        return CodexStructuredResult(
            dict(GOOD),
            _receipt(
                prompt=prompt,
                paths=tuple(paths),
                schema=schema,
                model=kwargs["model"],
                reasoning_effort=kwargs["reasoning_effort"],
            ),
        )

    outcome = run_relational_headless_campaign(
        corpus_root=root,
        plan=plan,
        split_index=_split(),
        predecessor=predecessor,
        exposure_store=tmp_path / "exposure",
        artifact_store=tmp_path / "artifacts",
        transport=transport,
        png_reader=lambda path: path.read_bytes(),
        extractor=_cached_extractor(),
        packet_verifier=lambda packet, **_kwargs: packet,
    )
    assert calls == 2
    assert [item.status for item in outcome.task_outcomes] == [
        "terminal_failure",
        "complete",
    ]
    assert outcome.task_outcomes[0].artifact["phase"] == (
        "campaign-attempt-terminal-persistence"
    )
    assert outcome.task_outcomes[0].artifact["query_labels_revealed"] is True
    assert outcome.report["joint_task_accuracy"]["denominator"] == 2
    assert all(
        item["attempt_terminal_digest"] is not None
        for item in outcome.report["tasks"]
    )


def test_source_digest_change_fails_before_exposure_or_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, predecessor, plan = _campaign_fixture(tmp_path, tag="source-change")
    frozen_data = plan.to_data()
    frozen_digest = plan.digest
    monkeypatch.setattr(
        campaign_module,
        "relational_headless_runner_source_digest",
        lambda: "0" * 64,
    )
    monkeypatch.setattr(
        campaign_module,
        "relational_headless_campaign_source_digest",
        lambda: "f" * 64,
    )
    assert plan.to_data() == frozen_data
    assert plan.digest == frozen_digest
    calls = 0

    def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("pixels/model must not be exposed")

    exposure_store = tmp_path / "exposure"
    with pytest.raises(RelationalHeadlessCampaignError, match="reproduce|changed"):
        run_relational_headless_campaign(
            corpus_root=root,
            plan=plan,
            split_index=_split(),
            predecessor=predecessor,
            exposure_store=exposure_store,
            artifact_store=tmp_path / "artifacts",
            transport=forbidden,
            png_reader=forbidden,
        )
    assert calls == 0
    assert not exposure_store.exists()


def test_artifact_store_substitution_fails_before_writes_exposure_or_model(
    tmp_path: Path,
) -> None:
    root, predecessor, plan = _campaign_fixture(tmp_path, tag="store-binding")
    alternate_store = tmp_path / "alternate-artifacts"
    exposure_store = tmp_path / "alternate-exposure"
    calls = 0

    def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("store substitution reached pixels or model")

    with pytest.raises(
        RelationalHeadlessCampaignError,
        match="artifact/attempt store differs.*plan binding",
    ):
        run_relational_headless_campaign(
            corpus_root=root,
            plan=plan,
            split_index=_split(),
            predecessor=predecessor,
            exposure_store=exposure_store,
            artifact_store=alternate_store,
            transport=forbidden,
            png_reader=forbidden,
        )
    assert calls == 0
    assert not alternate_store.exists()
    assert not exposure_store.exists()


def test_symlinked_attempt_journal_is_rejected_before_exposure(
    tmp_path: Path,
) -> None:
    root, predecessor, plan = _campaign_fixture(tmp_path, tag="journal-symlink")
    artifact_store = tmp_path / "artifacts"
    artifact_store.mkdir()
    target = tmp_path / "journal-target"
    target.mkdir(mode=0o700)
    (artifact_store / "relational-headless-attempt-journal").symlink_to(
        target, target_is_directory=True
    )
    calls = 0

    def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("no model or pixel call is authorized")

    exposure_store = tmp_path / "exposure"
    with pytest.raises(RelationalHeadlessCampaignError, match="private and non-symlink"):
        run_relational_headless_campaign(
            corpus_root=root,
            plan=plan,
            split_index=_split(),
            predecessor=predecessor,
            exposure_store=exposure_store,
            artifact_store=artifact_store,
            transport=forbidden,
            png_reader=forbidden,
        )
    assert calls == 0
    assert not exposure_store.exists()


def test_campaign_cli_has_no_raw_schedule_secret_argument() -> None:
    actions = {action.dest: action for action in campaign_module._parser()._actions}
    assert "schedule_secret" not in actions
    assert actions["schedule_secret_file"].required is True
    assert actions["artifact_store"].required is True
    assert actions["campaign_mode"].default == runner_module.STRICT_DEV_MODE
    assert set(actions["campaign_mode"].choices) == {
        runner_module.STRICT_DEV_MODE,
        EXACT_UNUSED_TRAIN_ENGINEERING_MODE,
    }


def test_engineering_cli_write_plan_is_explicit_and_metadata_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    root = tmp_path / "ShapeBongard_V2"
    _write_corpus(root)
    release_authenticator = _release_authenticator(tmp_path, root)
    predecessor = ExposureLedger.create(CORPUS)
    split = SplitIndex(
        groups=(
            ("test", (EXPLICITLY_SEALED_ENGINEERING_TASK_ID,)),
            ("train", EXACT_UNUSED_TRAIN_ENGINEERING_TASK_IDS),
            ("val", ()),
        ),
        source_digest=SPLIT,
    )
    monkeypatch.setattr(
        campaign_module.SplitIndex,
        "load",
        classmethod(lambda _cls, _path: split),
    )
    monkeypatch.setattr(
        campaign_module.ExposureLedger,
        "load",
        classmethod(lambda _cls, _path: predecessor),
    )
    monkeypatch.setattr(
        campaign_module, "_read_private_schedule_secret", lambda _path: SCHEDULE_SECRET
    )
    monkeypatch.setattr(
        campaign_module.ReleaseArchiveAuthenticator,
        "load",
        classmethod(lambda _cls, **_kwargs: release_authenticator),
    )
    destination = tmp_path / "engineering.plan.json"
    result = campaign_module.main(
        [
            "--split-file",
            str(tmp_path / "synthetic-split.json"),
            "--ledger-in",
            str(tmp_path / "synthetic-ledger.json"),
            "--write-plan",
            str(destination),
            "--campaign-mode",
            EXACT_UNUSED_TRAIN_ENGINEERING_MODE,
            "--expected-release-digest",
            release_authenticator.release_descriptor_digest,
            "--release-descriptor-file",
            str(tmp_path / "synthetic-release.json"),
            "--release-archive",
            str(tmp_path / "synthetic-release.zip"),
            "--expected-corpus-digest",
            CORPUS,
            "--expected-split-digest",
            SPLIT,
            "--expected-ledger-digest",
            predecessor.digest,
            "--campaign-seed",
            "fixture-engineering-cli",
            "--selection-seed-provenance",
            "synthetic CLI metadata-only seed",
            "--schedule-secret-file",
            str(tmp_path / "synthetic-secret"),
            "--artifact-store",
            str(tmp_path / "engineering-artifacts"),
            "--exposure-observed-at",
            "2026-08-07T17:00:00Z",
            "--expected-task-count",
            "5",
            "--expected-codex-launcher-sha256",
            LAUNCHER,
        ]
    )
    assert result == 0
    public = json.loads(destination.read_text(encoding="utf-8"))
    assert public["campaign_mode"] == EXACT_UNUSED_TRAIN_ENGINEERING_MODE
    assert [item["task_id"] for item in public["tasks"]] == list(
        EXACT_UNUSED_TRAIN_ENGINEERING_TASK_IDS
    )
    assert json.loads(capsys.readouterr().out)["pixels_opened"] is False


def test_execute_cli_requires_durable_cold_replay_before_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    root, predecessor, plan = _campaign_fixture(tmp_path, tag="cli-execute")
    plan_file = tmp_path / "campaign.plan.json"
    write_relational_headless_campaign_plan(plan_file, plan=plan)
    monkeypatch.setattr(
        campaign_module.SplitIndex,
        "load",
        classmethod(lambda _cls, _path: _split()),
    )
    monkeypatch.setattr(
        campaign_module.ExposureLedger,
        "load",
        classmethod(lambda _cls, _path: predecessor),
    )
    monkeypatch.setattr(
        campaign_module,
        "_read_private_schedule_secret",
        lambda _path: SCHEDULE_SECRET,
    )
    monkeypatch.setattr(
        campaign_module.ReleaseArchiveAuthenticator,
        "load",
        classmethod(lambda _cls, **_kwargs: plan.task_plans[0]._release_authenticator),
    )
    calls: list[str] = []

    class FakeOutcome:
        def to_data(self):
            return {"status": "synthetic-complete"}

    class FakeReplay:
        def to_data(self):
            return {
                "cold_replay_digest": "sha256:" + "a" * 64,
                "cold_replay_path": str(tmp_path / "sealed-replay.json"),
                "proposer_or_model_called_during_replay": False,
            }

    outcome = FakeOutcome()

    def fake_run(**kwargs):
        assert kwargs["plan"].digest == plan.digest
        calls.append("run")
        return outcome

    def fake_replay(**kwargs):
        assert kwargs["campaign_outcome"] is outcome
        assert calls == ["run"]
        calls.append("replay")
        return FakeReplay()

    monkeypatch.setattr(campaign_module, "run_relational_headless_campaign", fake_run)
    monkeypatch.setattr(
        campaign_module,
        "cold_replay_durable_relational_headless_campaign",
        fake_replay,
    )
    args = [
        "--split-file",
        str(tmp_path / "synthetic-split.json"),
        "--ledger-in",
        str(tmp_path / "synthetic-ledger.json"),
        "--plan-file",
        str(plan_file),
        "--expected-plan-digest",
        plan.digest,
        "--expected-release-digest",
        plan.official_release_descriptor_digest,
        "--release-descriptor-file",
        str(tmp_path / "synthetic-release.json"),
        "--release-archive",
        str(tmp_path / "synthetic-release.zip"),
        "--expected-corpus-digest",
        CORPUS,
        "--expected-split-digest",
        SPLIT,
        "--expected-ledger-digest",
        predecessor.digest,
        "--campaign-seed",
        "fixture-cli-execute",
        "--selection-seed-provenance",
        "synthetic cli-execute seed",
        "--schedule-secret-file",
        str(tmp_path / "synthetic-secret"),
        "--exposure-observed-at",
        "2026-08-07T16:00:00Z",
        "--expected-task-count",
        "2",
        "--expected-codex-launcher-sha256",
        LAUNCHER,
        "--execute",
        "--corpus-root",
        str(root),
        "--exposure-store",
        str(tmp_path / "exposure"),
        "--artifact-store",
        str(tmp_path / "artifacts"),
    ]
    with pytest.raises(RelationalHeadlessCampaignError, match="differs"):
        campaign_module.main([*args[:-1], str(tmp_path / "substituted-artifacts")])
    assert calls == []
    assert not (tmp_path / "substituted-artifacts").exists()

    result = campaign_module.main(args)
    assert result == 0
    assert calls == ["run", "replay"]
    output = json.loads(capsys.readouterr().out)
    assert output["cold_replay"]["proposer_or_model_called_during_replay"] is False


def test_task_runner_rejects_each_campaign_edge_field_before_pixels(
    tmp_path: Path,
) -> None:
    root, predecessor, plan = _campaign_fixture(tmp_path, tag="edge-fields")
    task_plan = plan.task_plans[0]
    source = f"{campaign_module.CAMPAIGN_PROTOCOL_ID}:plan:{plan.digest}"
    groups = _split().canonical_groups
    exact = {
        "phase": campaign_module.CAMPAIGN_EXPOSURE_PHASE,
        "actor": campaign_module.CAMPAIGN_EXPOSURE_ACTOR,
        "purpose": campaign_module.CAMPAIGN_EXPOSURE_PURPOSE,
    }
    calls = 0

    def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("campaign edge must fail before pixels/model")

    for ordinal, field in enumerate(("phase", "actor", "purpose")):
        values = {**exact, field: "adversarial-substitution"}
        successor = predecessor.record(
            phase=values["phase"],
            actor=values["actor"],
            purpose=values["purpose"],
            task_ids=plan.task_ids,
            source=source,
            observed_at=plan.exposure_observed_at,
            known_task_ids=(
                set(groups["train"]) | set(groups["val"]) | set(groups["test"])
            ),
            sealed_task_ids=groups["test"],
            require_unseen=True,
        )
        exposure_path, _cold = campaign_module._persist_exposure(
            successor, tmp_path / f"bad-exposure-{ordinal}"
        )
        with pytest.raises(
            runner_module.RelationalHeadlessRunError,
            match="campaign exposure edge",
        ):
            runner_module.run_relational_headless(
                corpus_root=root,
                task_id=task_plan.task_id,
                split_index=_split(),
                predecessor=predecessor,
                expected_corpus_digest=CORPUS,
                expected_split_source_digest=SPLIT,
                expected_exposure_predecessor_digest=predecessor.digest,
                seed=task_plan._seed,
                exposure_observed_at=plan.exposure_observed_at,
                exposure_store=tmp_path / f"unused-exposure-{ordinal}",
                artifact_store=tmp_path / f"bad-artifacts-{ordinal}",
                expected_launcher_digest=task_plan.expected_launcher_digest,
                release_authenticator=task_plan._release_authenticator,
                cloud_policy_cache_snapshot=plan._cloud_policy_cache_snapshot,
                model=task_plan.model,
                reasoning_effort=task_plan.reasoning_effort,
                minutes=task_plan.minutes,
                transport=forbidden,
                png_reader=forbidden,
                label_nonce=task_plan._label_nonce,
                support_selection_key=task_plan._support_selection_key,
                precommitted_exposure_successor=successor,
                precommitted_exposure_path=exposure_path,
                precommitted_campaign_task_ids=plan.task_ids,
                precommitted_campaign_source=source,
                precommitted_campaign_task_plan_digest=task_plan.digest,
            )
    assert calls == 0

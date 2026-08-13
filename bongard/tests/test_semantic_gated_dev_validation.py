from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace
import threading
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw
import pytest

from bongard import semantic_gated_dev_validation as gated_dev_validation
from bongard.artifacts import canonical_digest, canonical_json
from bongard.cohorts import parse_official_task_id
from bongard.exposure import ExposureLedger, semantic_resolver_policy_digest
from bongard.historical_exposure import load_historical_exposure
from bongard.semantic_calibration_campaign import (
    CAMPAIGN_SELECTION_ALGORITHM_V1,
    SemanticCalibrationProposalArchive,
    run_semantic_calibration_campaign,
    verify_semantic_campaign_against_corpus,
)
from bongard.semantic_calibration_command import (
    DESCRIPTIVE_STAGE_A_DESIGN,
    STAGE_A_COMMAND_RECEIPT_SCHEMA,
    STAGE_A_COMMAND_RECEIPT_SCHEMA_V1,
    STAGE_A_SCOPE,
    StageACalibrationCommandConfig,
    StageACommandReceipt,
    freeze_stage_a_source_dependencies,
    persist_stage_a_cache_snapshot,
)
from bongard.semantic_gated_dev_validation import (
    GatedDevAcceptancePolicy,
    GatedDevTaskRun,
    GatedDevTransportIdentityError,
    GatedDevValidationArtifact,
    GatedDevValidationError,
    capture_gated_dev_replay_bytes,
    plan_gated_dev_validation,
    run_gated_dev_validation,
    _strictly_verify_gated_dev_task_run,
    _maximum_disjoint_candidates,
)
from bongard.semantic_protocol import (
    build_prospective_soft_scorer_protocol,
    build_visual_semantic_policy,
)
from bongard.tests.test_semantic_calibration_campaign import (
    _CAMPAIGN_DRILL_IDS,
    _LAUNCHER_DIGEST,
    _as_legacy_v1_proposal_archive,
    _proposal_payload as _stage_a_proposal_payload,
    _unique_receipt,
)
from bongard.tests.no_tools_fixture import canonical_codex_receipt
from bongard.transport import (
    DEFAULT_CODEX_MODEL,
    PINNED_CODEX_CLI_VERSION,
    CloudPolicyCacheSnapshot,
    CodexReceipt,
    CodexStructuredResult,
)
from bongard.typed_visual_proposal import PANEL_DESCRIPTION_KEYS
from bongard.corpus import ShapeBongardCorpus


_DEV_IDS = (
    "bd_inverse_trap_arc180_0000",
    "bd_symm_unbala_goldfish_0000",
)
_TEST_ID = "hd_balanced_two-exist_quadrangle_0000"
_STAGE_A_SEED = hashlib.sha256(b"external stage-a fixture seed").hexdigest()
_LAUNCHER_VERSION = PINNED_CODEX_CLI_VERSION


def _proposer_receipt(
    prompt: str,
    paths: Sequence[str],
    schema: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    model: str,
    effort: str,
) -> CodexReceipt:
    return canonical_codex_receipt(
        prompt,
        paths,
        schema,
        payload,
        launcher_digest=_LAUNCHER_DIGEST,
        model=model,
        reasoning_effort=effort,
        command_fixture="semantic proposer turn",
    )


def _scorer_receipt(
    prompt: str,
    paths: Sequence[str],
    names: Sequence[str],
    schema: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> CodexReceipt:
    return canonical_codex_receipt(
        prompt,
        paths,
        schema,
        payload,
        launcher_digest=_LAUNCHER_DIGEST,
        model=DEFAULT_CODEX_MODEL,
        reasoning_effort="medium",
        names=names,
        command_fixture="semantic scorer turn",
    )


def _draw_panel(
    path: Path,
    *,
    task_index: int,
    positive: bool,
    panel_index: int,
) -> None:
    image = Image.new("L", (96, 96), color=255)
    draw = ImageDraw.Draw(image)
    dx = panel_index
    dy = (2 * panel_index + task_index) % 8
    draw.ellipse((12 + dx, 30 + dy, 30 + dx, 48 + dy), fill=0)
    if positive:
        draw.rectangle((61 - dx, 28 + dy, 77 - dx, 46 + dy), fill=0)
    # A tiny task/panel-specific mark remains connected to the left component,
    # keeping all fourteen byte preimages distinct without changing count.
    draw.point((13 + dx, 31 + dy + (task_index % 2)), fill=0)
    image.save(path, format="PNG")


def _corpus(tmp_path: Path):
    root = tmp_path / "ShapeBongard_V2"
    task_ids = _CAMPAIGN_DRILL_IDS + _DEV_IDS + (_TEST_ID,)
    for task_index, task_id in enumerate(task_ids):
        family = task_id[:2]
        for label, positive in (("1", True), ("0", False)):
            directory = root / family / "images" / task_id / label
            directory.mkdir(parents=True, exist_ok=True)
            for panel_index in range(7):
                _draw_panel(
                    directory / f"{panel_index}.png",
                    task_index=task_index,
                    positive=positive,
                    panel_index=panel_index,
                )
    (root / "ShapeBongard_V2_split.json").write_text(
        json.dumps(
            {
                "train": list(_CAMPAIGN_DRILL_IDS + _DEV_IDS),
                "test_hd_comb": [_TEST_ID],
            }
        ),
        encoding="utf-8",
    )
    corpus = ShapeBongardCorpus.from_root(root)
    return corpus, corpus.build_manifest()


def _protocol():
    return build_prospective_soft_scorer_protocol(
        proposer_model_id=DEFAULT_CODEX_MODEL,
        proposer_reasoning_effort="medium",
        scorer_model_id=DEFAULT_CODEX_MODEL,
        scorer_reasoning_effort="medium",
        score_bin_edges=(0.0, 0.5, 1.0),
        affirmative_boundary=0.5,
        confidence_level=0.1,
        minimum_clusters_per_bin=2,
    )


def _stage_a(corpus, manifest):
    protocol = _protocol()
    proposer_count = 0
    scorer_count = 0

    def proposer_transport(prompt, paths, schema, **kwargs):
        nonlocal proposer_count
        index = proposer_count
        proposer_count += 1
        kind = "soft" if index < 4 else "direct" if index == 4 else "rejected"
        payload = _stage_a_proposal_payload(kind, index)
        receipt = _proposer_receipt(
            prompt,
            paths,
            schema,
            payload,
            model=kwargs["model"],
            effort=kwargs["reasoning_effort"],
        )
        return CodexStructuredResult(payload, _unique_receipt(receipt, index))

    def scorer_transport(prompt, paths, names, schema, **kwargs):
        del kwargs
        nonlocal scorer_count
        index = scorer_count
        scorer_count += 1
        cue_ids = tuple(
            schema["properties"]["cue_judgments"]["items"]["properties"]
            ["cue_id"]["enum"]
        )
        witness_ids = tuple(
            schema["properties"]["cue_judgments"]["items"]["properties"]
            ["witness_ids"]["items"]["enum"]
        )
        supported = index < 2
        payload = {
            "cue_judgments": [
                {
                    "cue_id": cue_id,
                    "judgment": "supported" if supported else "unsupported",
                    "witness_ids": [witness_ids[0]] if supported else [],
                }
                for cue_id in cue_ids
            ]
        }
        receipt = _scorer_receipt(prompt, paths, names, schema, payload)
        return CodexStructuredResult(
            payload, _unique_receipt(receipt, 100 + index)
        )

    predecessor = ExposureLedger.create(manifest.digest)
    campaign = run_semantic_calibration_campaign(
        corpus,
        protocol,
        candidate_count=6,
        seed=_STAGE_A_SEED,
        source_corpus_manifest_digest=manifest.digest,
        expected_codex_launcher_digest=_LAUNCHER_DIGEST,
        exposure_ledger=predecessor,
        expected_exposure_ledger_digest=predecessor.digest,
        label_nonce_root=hashlib.sha256(b"stage-a-label-root").hexdigest(),
        cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
        proposer_transport=proposer_transport,
        scorer_transport=scorer_transport,
    )
    assert proposer_count == 6
    assert scorer_count == 4
    return campaign


def _stage_a_command_receipt(
    tmp_path: Path,
    campaign,
) -> StageACommandReceipt:
    archive = campaign.score_batch.commitment_batch.proposal_archive
    protocol = archive.protocol
    execution = archive.execution_config
    config = StageACalibrationCommandConfig(
        expected_codex_launcher_digest=_LAUNCHER_DIGEST,
        expected_exposure_ledger_digest=archive.exposure_predecessor.digest,
        design_mode=DESCRIPTIVE_STAGE_A_DESIGN,
        selection_seed=_STAGE_A_SEED,
        selection_seed_provenance=(
            "fixture external seed fixed before Stage-A task selection"
        ),
        candidate_count=archive.candidate_count,
        semantic_cohort=archive.semantic_cohort,
        families=archive.families,
        score_bin_edges=protocol.score_bin_edges,
        affirmative_boundary=protocol.affirmative_boundary,
        confidence_level=protocol.confidence_level,
        minimum_clusters_per_bin=protocol.minimum_clusters_per_bin,
        proposer_model_id=protocol.proposer_model_id,
        proposer_reasoning_effort=protocol.proposer_reasoning_effort,
        scorer_model_id=protocol.scorer_model_id,
        scorer_reasoning_effort=protocol.scorer_reasoning_effort,
        proposer_minutes=execution.proposer_minutes,
        scorer_minutes=execution.scorer_minutes,
        proposer_max_workers=execution.proposer_max_workers,
        scorer_max_workers=execution.scorer_max_workers,
        verifier_id="canonical-bongard-verifier",
        executable=execution.executable,
    )
    terminal_path = (tmp_path / "stage-a-campaign.json").resolve()
    terminal_payload = canonical_json(campaign.to_data()) + b"\n"
    terminal_path.write_bytes(terminal_payload)
    ledger_path = archive.exposure_successor.write_content_addressed(
        tmp_path / "stage-a-ledger"
    ).resolve()
    ledger_payload = ledger_path.read_bytes()
    cache_path, cache_file_sha256, cache_count = persist_stage_a_cache_snapshot(
        CloudPolicyCacheSnapshot(None), tmp_path / "stage-a-cache"
    )
    source_dependencies = freeze_stage_a_source_dependencies()
    content = {
        "schema": STAGE_A_COMMAND_RECEIPT_SCHEMA,
        "status": "succeeded",
        "stage_a_scope": STAGE_A_SCOPE,
        "terminal_schema": campaign.to_data()["schema"],
        "terminal_internal_digest": campaign.digest,
        "terminal_artifact_path": str(terminal_path),
        "terminal_artifact_file_sha256": (
            "sha256:" + hashlib.sha256(terminal_payload).hexdigest()
        ),
        "exposure_ledger_path": str(ledger_path),
        "exposure_ledger_digest": archive.exposure_successor.digest,
        "exposure_ledger_file_sha256": (
            "sha256:" + hashlib.sha256(ledger_payload).hexdigest()
        ),
        "protocol_digest": protocol.digest(),
        "command_config": config.to_data(),
        "command_config_digest": config.digest,
        "input_authentication_digest": "sha256:" + "a" * 64,
        "launcher_version": _LAUNCHER_VERSION,
        "launcher_digest": _LAUNCHER_DIGEST,
        "cloud_policy_cache_binding": "absent",
        "cloud_policy_cache_snapshot_path": str(cache_path),
        "cloud_policy_cache_snapshot_file_sha256": cache_file_sha256,
        "cloud_policy_cache_snapshot_byte_count": cache_count,
        "cloud_policy_cache_snapshot_bytes_embedded": False,
        "source_dependencies": source_dependencies.to_data(),
        "source_dependency_digest": source_dependencies.digest,
        "cold_verified": True,
        "python_predicate_authoritative": True,
        "optional_checker_may_affect_result": False,
    }
    digest = canonical_digest(content)
    payload = canonical_json({**content, "command_receipt_digest": digest}) + b"\n"
    return StageACommandReceipt.from_bytes(
        payload,
        expected_receipt_digest=digest,
    )


def _stage_b_payload(*, rejected: bool = False) -> dict[str, object]:
    return {
        "positive_description": "two separated solid ink components",
        "panel_descriptions": {
            name: "literal compact ink components"
            for name in PANEL_DESCRIPTION_KEYS
        },
        "view": "literal_ink",
        "deterministic_atoms": [] if rejected else [
            {
                "catalog_key": "component.count",
                "comparison": "equal",
                "arguments": {"target_count": 2},
            }
        ],
        "soft_claim": None,
        "formula": {
            "kind": "all",
            "atom_indices": [] if rejected else [0],
        },
    }


@pytest.fixture(scope="module")
def frozen_system(tmp_path_factory: pytest.TempPathFactory):
    tmp_path = tmp_path_factory.mktemp("gated-dev-system")
    corpus, manifest = _corpus(tmp_path)
    campaign = _stage_a(corpus, manifest)
    command_receipt = _stage_a_command_receipt(tmp_path, campaign)
    policy = build_visual_semantic_policy(
        campaign.calibration.family,
        prospective_protocol=campaign.calibration.protocol,
    )
    predecessor = (
        campaign.score_batch.commitment_batch.proposal_archive.exposure_successor
    )
    base = (corpus, manifest, campaign, command_receipt, policy, predecessor)
    original = Path.read_bytes

    def guarded(path: Path):
        if any(task_id in str(path) for task_id in _DEV_IDS):
            raise AssertionError("Stage-B planning opened a fresh DEV PNG")
        return original(path)

    Path.read_bytes = guarded
    try:
        plan = _plan(base)
    finally:
        Path.read_bytes = original
    return (*base, plan)


def _plan(frozen_system, *, workers: int = 2):
    corpus, manifest, campaign, command_receipt, policy, predecessor = (
        frozen_system[:6]
    )
    acceptance = GatedDevAcceptancePolicy(
        confidence_level=0.001,
        minimum_selected_clusters=2,
        minimum_gate_passed_clusters=1,
        minimum_gate_coverage_lower=0.0,
        minimum_both_query_correct_lower=0.0,
        minimum_fully_determinate_lower=0.0,
        maximum_any_abstention_upper=1.0,
        maximum_any_error_upper=1.0,
    )
    return plan_gated_dev_validation(
        corpus,
        source_corpus_manifest=manifest,
        expected_source_corpus_manifest_digest=manifest.digest,
        expected_split_source_digest=manifest.split.source_digest,
        stage_a_campaign=campaign,
        stage_a_command_receipt=command_receipt,
        visual_semantic_policy=policy,
        exposure_predecessor=predecessor,
        expected_exposure_predecessor_digest=predecessor.digest,
        public_seed=hashlib.sha256(b"externally committed stage-b seed").hexdigest(),
        selection_seed_provenance=(
            "fixture external beacon fixed before DEV task identities were inspected"
        ),
        requested_task_count=2,
        exposure_observed_at="2026-08-06T12:00:00Z",
        cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
        acceptance_policy=acceptance,
        families=("bd",),
        task_max_workers=workers,
    )


@pytest.fixture(autouse=True)
def measured_fixture_launcher(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "bongard.semantic_gated_dev_validation.codex_cli_authenticated_fingerprint",
        lambda executable, *, expected_launcher_digest: {
            "version": _LAUNCHER_VERSION,
            "launcher_digest": expected_launcher_digest,
        },
    )


def _stage_b_transport(*, reject_one: bool = False, wrong_launcher: bool = False):
    count = 0
    lock = threading.Lock()

    def transport(prompt, paths, schema, **kwargs):
        nonlocal count
        assert kwargs["expected_launcher_digest"] == _LAUNCHER_DIGEST
        with lock:
            index = count
            count += 1
        payload = _stage_b_payload(rejected=reject_one and index == 1)
        receipt = _unique_receipt(
            _proposer_receipt(
                prompt,
                paths,
                schema,
                payload,
                model=kwargs["model"],
                effort=kwargs["reasoning_effort"],
            ),
            200 + index,
        )
        if wrong_launcher:
            raw = receipt.to_dict()
            raw["codex_launcher_digest"] = "c" * 64
            raw.pop("receipt_digest")
            raw["receipt_digest"] = canonical_digest(raw)
            receipt = CodexReceipt(
                **{
                    **raw,
                    "event_types": tuple(raw["event_types"]),
                    "item_types": tuple(raw["item_types"]),
                }
            )
        return CodexStructuredResult(payload, receipt)

    return transport, lambda: count


def test_hd_selector_uses_exact_maximum_matching_not_seed_retry() -> None:
    candidates = tuple(
        SimpleNamespace(
            family="hd",
            concepts=concepts,
            task_id=f"hd_{left}-{right}_0000",
        )
        for left, right, concepts in (
            ("a", "b", ("a", "b")),
            ("b", "c", ("b", "c")),
            ("c", "d", ("c", "d")),
        )
    )
    first = _maximum_disjoint_candidates(candidates, seed="1" * 64, family="hd")
    second = _maximum_disjoint_candidates(candidates, seed="2" * 64, family="hd")
    assert len(first) == len(second) == 2
    for chosen in (first, second):
        attributes = [concept for item in chosen for concept in item.concepts]
        assert len(attributes) == len(set(attributes))


def test_hd_selector_excludes_every_predecessor_ledger_constituent() -> None:
    historical = load_historical_exposure()
    resolver = semantic_resolver_policy_digest(historical)
    predecessor = ExposureLedger.create("sha256:" + "6" * 64).record(
        phase="semantic-calibration",
        actor="fixture",
        purpose="adversarial prior HD disclosure",
        task_ids=("hd_has_seven_straight_lines-exist_triangle_0000",),
        observed_at="2026-08-06T11:00:00Z",
    )
    candidates = tuple(
        SimpleNamespace(
            parsed=parse_official_task_id(task_id),
            split="train",
            historically_clean=True,
            semantic_cohort="dev",
        )
        for task_id in (
            # Each of these is a new exact pair, but it reuses one constituent.
            "hd_has_seven_straight_lines-exist_quadrangle_0000",
            "hd_exist_regular-exist_triangle_0000",
            # Only this pair is constituent-disjoint from the predecessor.
            "hd_exist_regular-exist_quadrangle_0000",
        )
    )
    report = SimpleNamespace(records=candidates)

    selected, availability = gated_dev_validation._select_strict_dev_tasks(
        None,
        report=report,
        families=("hd",),
        candidate_count=1,
        seed="7" * 64,
        exposure_predecessor=predecessor,
        historical=historical,
        resolver_digest=resolver,
        blocked_clusters=frozenset(),
    )

    assert gated_dev_validation._ledger_exposed_hd_constituent_attributes(
        predecessor
    ) == ("exist_triangle", "has_seven_straight_lines")
    assert tuple(item.task_id for item in selected) == (
        "hd_exist_regular-exist_quadrangle_0000",
    )
    assert availability == (("hd", 1),)
    with pytest.raises(GatedDevValidationError, match=r"permit 1 \(HD=1\)"):
        gated_dev_validation._select_strict_dev_tasks(
            None,
            report=report,
            families=("hd",),
            candidate_count=2,
            seed="7" * 64,
            exposure_predecessor=predecessor,
            historical=historical,
            resolver_digest=resolver,
            blocked_clusters=frozenset(),
        )


def test_stage_b_exposure_precommit_fsyncs_file_then_parent_and_reloads_exactly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    predecessor = ExposureLedger.create("sha256:" + "7" * 64)
    successor = predecessor.record(
        phase="dev-validation",
        actor="fixture",
        purpose="durability-order-test",
        task_ids=("bd_fixture_0000",),
        observed_at="2026-08-06T12:00:00Z",
    )
    calls: list[str] = []
    real_fsync = gated_dev_validation.os.fsync

    def tracked_fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        calls.append("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(descriptor)

    monkeypatch.setattr(gated_dev_validation.os, "fsync", tracked_fsync)
    path = gated_dev_validation._persist_gated_dev_exposure_precommit(
        successor,
        tmp_path / "exposure",
    )

    assert calls == ["file", "directory"]
    assert path.is_absolute()
    assert path.read_bytes() == successor.to_json().encode("utf-8")
    assert ExposureLedger.load(path) == successor


def test_stage_b_exposure_precommit_fsync_failure_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    successor = ExposureLedger.create("sha256:" + "8" * 64).record(
        phase="dev-validation",
        actor="fixture",
        purpose="durability-failure-test",
        task_ids=("bd_fixture_0000",),
        observed_at="2026-08-06T12:00:00Z",
    )

    def failed_fsync(_descriptor: int) -> None:
        raise OSError("fixture durability failure")

    monkeypatch.setattr(gated_dev_validation.os, "fsync", failed_fsync)
    with pytest.raises(GatedDevValidationError, match="cannot fsync Stage-B"):
        gated_dev_validation._persist_gated_dev_exposure_precommit(
            successor,
            tmp_path / "exposure",
        )

    # Preserve the attempted disclosure even when durability cannot be
    # certified; a later retry can only accept the same content-addressed bytes.
    paths = tuple((tmp_path / "exposure").glob("*.exposure.json"))
    assert len(paths) == 1
    assert paths[0].read_bytes() == successor.to_json().encode("utf-8")


def test_plan_is_dev_pixel_free_and_rejects_forged_stage_a_label(
    frozen_system, tmp_path: Path
) -> None:
    (
        corpus,
        manifest,
        campaign,
        command_receipt,
        policy,
        predecessor,
        plan,
    ) = frozen_system
    assert set(item.task_id for item in plan.selections) == set(_DEV_IDS)
    assert plan.available_clusters_by_family == (("bd", 2),)
    assert plan.maximum_admissible_task_count == 2
    assert all(
        token.startswith(("basic_family:", "basic_morphology:"))
        for item in plan.selections
        for token in item.disclosure_tokens
    )

    forged = deepcopy(campaign.to_data())
    forged["label_reveals"][0]["labels"][0]["positive"] = not (
        forged["label_reveals"][0]["labels"][0]["positive"]
    )
    with pytest.raises(Exception):
        verify_semantic_campaign_against_corpus(
            forged, corpus=corpus, corpus_manifest=manifest
        )
    with pytest.raises(GatedDevValidationError, match="cache snapshot"):
        run_gated_dev_validation(
            corpus,
            plan,
            source_corpus_manifest=manifest,
            stage_a_campaign=campaign,
            stage_a_command_receipt=command_receipt,
            visual_semantic_policy=policy,
            exposure_predecessor=predecessor,
            exposure_output_directory=tmp_path / "wrong-cache-exposure",
            artifact_output_directory=tmp_path / "wrong-cache-artifact",
            cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(
                b'{"signed_payload":{},"signature":"fixture"}'
            ),
        )


def test_selection_authority_rejects_reordered_tasks_and_forged_capacity(
    frozen_system,
) -> None:
    corpus, _manifest, _campaign, _receipt, _policy, predecessor, plan = (
        frozen_system
    )
    reordered = replace(plan, selections=tuple(reversed(plan.selections)))
    with pytest.raises(GatedDevValidationError, match="selection/order"):
        gated_dev_validation._audit_gated_dev_selection_against_corpus(
            corpus,
            plan=reordered,
            predecessor=predecessor,
        )

    inflated_availability = tuple(
        (family, count + (1 if index == 0 else 0))
        for index, (family, count) in enumerate(
            plan.available_clusters_by_family
        )
    )
    inflated = replace(
        plan,
        available_clusters_by_family=inflated_availability,
        maximum_admissible_task_count=sum(
            count for _, count in inflated_availability
        ),
    )
    with pytest.raises(GatedDevValidationError, match="availability/maximum"):
        gated_dev_validation._audit_gated_dev_selection_against_corpus(
            corpus,
            plan=inflated,
            predecessor=predecessor,
        )


def test_stage_b_requires_successful_source_frozen_v2_receipt(
    frozen_system,
) -> None:
    campaign = frozen_system[2]
    command_receipt = frozen_system[3]

    def historical_v1(*, status: str, cold_verified: bool) -> StageACommandReceipt:
        data = deepcopy(dict(command_receipt.to_data()))
        data["schema"] = STAGE_A_COMMAND_RECEIPT_SCHEMA_V1
        data["status"] = status
        data["cold_verified"] = cold_verified
        data.pop("source_dependencies")
        data.pop("source_dependency_digest")
        data.pop("command_receipt_digest")
        digest = canonical_digest(data)
        payload = canonical_json(
            {**data, "command_receipt_digest": digest}
        ) + b"\n"
        return StageACommandReceipt.from_bytes(
            payload,
            expected_receipt_digest=digest,
        )

    with pytest.raises(GatedDevValidationError, match="v2.*audit-only"):
        gated_dev_validation._authenticate_stage_a_command_receipt(
            historical_v1(status="succeeded", cold_verified=True),
            campaign=campaign,
        )
    # Historical failed v1 remains strictly decodable, then fails on terminal
    # status before the v2 source-freeze authorization check.
    with pytest.raises(GatedDevValidationError, match="successful cold-verified"):
        gated_dev_validation._authenticate_stage_a_command_receipt(
            historical_v1(status="failed", cold_verified=False),
            campaign=campaign,
        )


def test_stage_b_rejects_decodable_legacy_stage_a_selection_authority(
    frozen_system,
) -> None:
    campaign = frozen_system[2]
    current = campaign.score_batch.commitment_batch.proposal_archive
    legacy_data = _as_legacy_v1_proposal_archive(current)
    legacy = SemanticCalibrationProposalArchive.from_data(
        legacy_data,
        expected_digest=legacy_data["proposal_archive_digest"],
    )
    assert legacy.selection_algorithm == CAMPAIGN_SELECTION_ALGORITHM_V1

    with pytest.raises(
        GatedDevValidationError,
        match="constituent-disjoint.*legacy v1.*audit-only",
    ):
        gated_dev_validation._require_current_stage_a_selection_algorithm(legacy)


def test_concurrent_stage_b_persists_exposure_then_cold_replays_without_models(
    frozen_system, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (
        corpus,
        manifest,
        campaign,
        command_receipt,
        policy,
        predecessor,
        plan,
    ) = frozen_system
    exposure_dir = tmp_path / "exposure"
    artifact_dir = tmp_path / "artifacts"
    transport, call_count = _stage_b_transport()
    precommit_durable = False
    campaign_serializations = 0
    persist_precommit = gated_dev_validation._persist_gated_dev_exposure_precommit
    campaign_type = type(campaign)
    serialize_campaign = campaign_type.to_data

    def tracked_precommit(successor, directory):
        nonlocal precommit_durable
        path = persist_precommit(successor, directory)
        precommit_durable = True
        return path

    monkeypatch.setattr(
        gated_dev_validation,
        "_persist_gated_dev_exposure_precommit",
        tracked_precommit,
    )

    def tracked_campaign_to_data(self):
        nonlocal campaign_serializations
        campaign_serializations += 1
        return serialize_campaign(self)

    monkeypatch.setattr(campaign_type, "to_data", tracked_campaign_to_data)

    def exposure_checked_transport(*args, **kwargs):
        assert precommit_durable
        assert tuple(exposure_dir.glob("*.exposure.json"))
        return transport(*args, **kwargs)

    artifact = run_gated_dev_validation(
        corpus,
        plan,
        source_corpus_manifest=manifest,
        stage_a_campaign=campaign,
        stage_a_command_receipt=command_receipt,
        visual_semantic_policy=policy,
        exposure_predecessor=predecessor,
        exposure_output_directory=exposure_dir,
        artifact_output_directory=artifact_dir,
        cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
        label_nonce_root="stage-b-secret-label-root",
        proposer_transport=exposure_checked_transport,
        scorer_transport=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("direct-only Stage-B run called scorer")
        ),
    )
    assert precommit_durable
    assert all(
        run.outer_record["schema"]
        == gated_dev_validation.VISUAL_SEMANTIC_COMPACT_OUTER_RUN_SCHEMA
        and "calibration" not in run.outer_record
        and run.outer_record["calibration_campaign_digest"] == campaign.digest
        and run.outer_record["calibration_digest"] == campaign.calibration.digest
        for run in artifact.task_runs
    )
    assert call_count() == 2
    assert artifact.summary.selected_clusters == 2
    assert artifact.summary.gate_passed_clusters >= 1
    assert sum(dict(artifact.summary.status_counts).values()) == 2
    assert artifact.summary.conditional_image_correct == (
        2 * artifact.summary.gate_passed_clusters
    )
    assert artifact.summary.validation_status == "pilot"
    assert "dependence_design_authorized" in artifact.summary.failure_reasons
    assert len(tuple(artifact_dir.glob("*.gated-dev-validation.json"))) == 1

    replay = capture_gated_dev_replay_bytes(
        corpus,
        artifact,
        source_corpus_manifest=manifest,
    )
    artifact_data = artifact.to_data()
    assert artifact_data["stage_a_campaign"] == campaign.to_data()
    assert artifact_data["stage_a_campaign_digest"] == campaign.digest
    assert artifact_data["stage_a_command_receipt_digest"] == (
        command_receipt.receipt_digest
    )
    # The Stage-B plan digest determines every episode seed.  A producer must
    # not be able to rehash preregistration metadata after seeing the results,
    # update only the wrapper seeds/receipt pointers, and retain an old valid
    # outer episode whose actual query draw used another seed.
    rewrapped = deepcopy(artifact_data)
    rewrapped_plan_data = rewrapped["plan"]
    rewrapped_plan_data["selection_seed_provenance"] += " (rewrapped)"
    rewrapped_plan_data.pop("plan_digest")
    rewrapped_plan_data["plan_digest"] = canonical_digest(
        rewrapped_plan_data
    )
    rewrapped["plan_digest"] = rewrapped_plan_data["plan_digest"]
    rewrapped_plan = gated_dev_validation.GatedDevValidationPlan.from_data(
        rewrapped_plan_data
    )
    rewrapped_run_digests = []
    for run_data, original_run in zip(
        rewrapped["task_runs"], artifact.task_runs, strict=True
    ):
        run_data["episode_seed"] = gated_dev_validation._episode_seed(
            rewrapped_plan, original_run.selection
        )
        run_data.pop("task_run_digest")
        run_data["task_run_digest"] = canonical_digest(run_data)
        rewrapped_run_digests.append(run_data["task_run_digest"])
    rewrapped["task_run_digests"] = rewrapped_run_digests
    rewrapped_receipt_digests = []
    for receipt_data, run_digest in zip(
        rewrapped["task_replay_receipts"],
        rewrapped_run_digests,
        strict=True,
    ):
        receipt_data["task_run_digest"] = run_digest
        receipt_data.pop("replay_receipt_digest")
        receipt_data["replay_receipt_digest"] = canonical_digest(receipt_data)
        rewrapped_receipt_digests.append(
            receipt_data["replay_receipt_digest"]
        )
    rewrapped["task_replay_receipt_digests"] = rewrapped_receipt_digests
    rewrapped.pop("validation_artifact_digest")
    rewrapped["validation_artifact_digest"] = canonical_digest(rewrapped)
    with pytest.raises(
        GatedDevValidationError,
        match="wrapper episode seed differs from outer public plan",
    ):
        GatedDevValidationArtifact.from_data(
            rewrapped,
            stage_a_campaign=campaign,
            stage_a_command_receipt=command_receipt,
            corpus=corpus,
            source_corpus_manifest=manifest,
            replay_bytes_by_task=replay,
        )
    forged_constructor_plan = replace(
        artifact.plan,
        stage_a_family_digest="0" * 64,
    )
    with pytest.raises(
        GatedDevValidationError,
        match="campaign/calibration/family/protocol differs from plan",
    ):
        GatedDevValidationArtifact(
            forged_constructor_plan,
            artifact.stage_a_campaign,
            artifact.visual_semantic_policy,
            artifact.exposure_predecessor,
            artifact.exposure_successor,
            artifact.task_runs,
            artifact.task_replay_receipts,
            artifact.summary,
        )
    forged_cold_metadata = deepcopy(artifact_data)
    forged_cold_plan = forged_cold_metadata["plan"]
    forged_cold_plan["stage_a"]["protocol_digest"] = "0" * 64
    forged_cold_plan.pop("plan_digest")
    forged_cold_plan["plan_digest"] = canonical_digest(forged_cold_plan)
    forged_cold_metadata["plan_digest"] = forged_cold_plan["plan_digest"]
    forged_cold_metadata.pop("validation_artifact_digest")
    forged_cold_metadata["validation_artifact_digest"] = canonical_digest(
        forged_cold_metadata
    )
    with pytest.raises(
        GatedDevValidationError,
        match="central Stage-A family/protocol differs from Stage-B plan",
    ):
        GatedDevValidationArtifact.from_data(
            forged_cold_metadata,
            stage_a_campaign=campaign,
            stage_a_command_receipt=command_receipt,
            corpus=corpus,
            source_corpus_manifest=manifest,
            replay_bytes_by_task=replay,
        )
    tampered_central_campaign = deepcopy(artifact_data)
    tampered_central_campaign["stage_a_campaign"]["score_batch"] = {}
    tampered_central_campaign.pop("validation_artifact_digest")
    tampered_central_campaign["validation_artifact_digest"] = canonical_digest(
        tampered_central_campaign
    )
    with pytest.raises(Exception):
        GatedDevValidationArtifact.from_data(
            tampered_central_campaign,
            stage_a_campaign=campaign,
            stage_a_command_receipt=command_receipt,
            corpus=corpus,
            source_corpus_manifest=manifest,
            replay_bytes_by_task=replay,
        )
    substituted_receipt = deepcopy(artifact_data)
    substituted_receipt["stage_a_command_receipt_digest"] = "0" * 64
    substituted_receipt.pop("validation_artifact_digest")
    substituted_receipt["validation_artifact_digest"] = canonical_digest(
        substituted_receipt
    )
    with pytest.raises(Exception, match="campaign/receipt references"):
        GatedDevValidationArtifact.from_data(
            substituted_receipt,
            stage_a_campaign=campaign,
            stage_a_command_receipt=command_receipt,
            corpus=corpus,
            source_corpus_manifest=manifest,
            replay_bytes_by_task=replay,
        )
    substituted_record = deepcopy(artifact.task_runs[0].outer_record)
    substituted_record["calibration_digest"] = "0" * 64
    substituted_record.pop("record_digest")
    substituted_record["record_digest"] = canonical_digest(substituted_record)
    substituted_run = GatedDevTaskRun(
        artifact.task_runs[0].selection,
        artifact.task_runs[0].episode_seed,
        substituted_record,
    )
    with pytest.raises(Exception, match="calibration reference"):
        _strictly_verify_gated_dev_task_run(
            substituted_run,
            blob_bytes_by_id=replay[substituted_run.selection.task_id],
            stage_a_campaign=(
                gated_dev_validation._campaign_anchor_from_verified_stage_a(
                    campaign
                )
            ),
        )
    selection_audits = 0
    selection_audit = (
        gated_dev_validation._audit_gated_dev_selection_against_corpus
    )

    def tracked_selection_audit(*args, **kwargs):
        nonlocal selection_audits
        selection_audits += 1
        return selection_audit(*args, **kwargs)

    monkeypatch.setattr(
        gated_dev_validation,
        "_audit_gated_dev_selection_against_corpus",
        tracked_selection_audit,
    )
    before = call_count()
    decoded = GatedDevValidationArtifact.from_data(
        artifact_data,
        stage_a_campaign=campaign,
        stage_a_command_receipt=command_receipt,
        corpus=corpus,
        source_corpus_manifest=manifest,
        replay_bytes_by_task=replay,
    )
    assert decoded.digest == artifact.digest
    assert call_count() == before
    assert selection_audits == 1

    corrupted = {task: dict(values) for task, values in replay.items()}
    completed = next(run for run in artifact.task_runs if run.status == "complete")
    first_task = completed.selection.task_id
    first_blob = next(iter(corrupted[first_task]))
    corrupted[first_task][first_blob] += b"tamper"
    with pytest.raises(Exception):
        GatedDevValidationArtifact.from_data(
            artifact.to_data(),
            stage_a_campaign=campaign,
            stage_a_command_receipt=command_receipt,
            corpus=corpus,
            source_corpus_manifest=manifest,
            replay_bytes_by_task=corrupted,
        )
    # Full campaign serialization is a fixed batch-boundary cost, never a
    # per-task operation.  The two-task fixture exercises build, immediate
    # replay, successful public replay, the rewrapping attack, the two Stage-A
    # metadata-join attacks, and both other tamper paths.
    assert campaign_serializations <= 15


def test_attrition_stays_in_selected_denominator_and_terminal_pilot_is_written(
    frozen_system, tmp_path: Path
) -> None:
    (
        corpus,
        manifest,
        campaign,
        command_receipt,
        policy,
        predecessor,
        plan,
    ) = frozen_system
    transport, call_count = _stage_b_transport(reject_one=True)
    artifact = run_gated_dev_validation(
        corpus,
        plan,
        source_corpus_manifest=manifest,
        stage_a_campaign=campaign,
        stage_a_command_receipt=command_receipt,
        visual_semantic_policy=policy,
        exposure_predecessor=predecessor,
        exposure_output_directory=tmp_path / "exposure-attrition",
        artifact_output_directory=tmp_path / "artifact-attrition",
        cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
        label_nonce_root="attrition-label-root",
        proposer_transport=transport,
    )
    assert call_count() == 2
    assert artifact.summary.selected_clusters == 2
    assert 0 <= artifact.summary.gate_passed_clusters <= 1
    assert dict(artifact.summary.status_counts)["proposal_error"] == 1
    assert sum(dict(artifact.summary.status_counts).values()) == 2
    assert artifact.summary.bounds[0].cluster_count == 2
    assert artifact.summary.bounds[0].successes == (
        artifact.summary.gate_passed_clusters
    )
    assert artifact.summary.bounds[1].cluster_count == (
        artifact.summary.gate_passed_clusters
    )
    assert artifact.summary.validation_status == "pilot"
    assert len(tuple((tmp_path / "artifact-attrition").glob("*.json"))) == 1

    # Even a completely self-rehashed outer/task record cannot rewrite a
    # retained typed proposal rejection's fixed score and enter a denominator.
    rejected = next(run for run in artifact.task_runs if run.status == "proposal_error")
    forged_outer = deepcopy(rejected.outer_record)
    forged_outer["episode"]["score"]["errors"] = 0
    forged_outer.pop("record_digest")
    forged_outer["record_digest"] = canonical_digest(forged_outer)
    forged_run = GatedDevTaskRun(
        rejected.selection,
        rejected.episode_seed,
        forged_outer,
    )
    replay = capture_gated_dev_replay_bytes(
        corpus,
        artifact,
        source_corpus_manifest=manifest,
    )
    with pytest.raises(GatedDevValidationError, match="not a permitted replayable"):
        _strictly_verify_gated_dev_task_run(
            forged_run,
            blob_bytes_by_id=replay[rejected.selection.task_id],
            stage_a_campaign=campaign,
        )


def test_generic_proposer_transport_failure_is_batch_fatal_after_exposure(
    frozen_system, tmp_path: Path
) -> None:
    (
        corpus,
        manifest,
        campaign,
        command_receipt,
        policy,
        predecessor,
        plan,
    ) = frozen_system

    def infrastructure_failure(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("fixture proposer transport unavailable")

    exposure_dir = tmp_path / "exposure-infrastructure-failure"
    artifact_dir = tmp_path / "artifact-infrastructure-failure"
    with pytest.raises(GatedDevValidationError, match="batch-fatal"):
        run_gated_dev_validation(
            corpus,
            plan,
            source_corpus_manifest=manifest,
            stage_a_campaign=campaign,
            stage_a_command_receipt=command_receipt,
            visual_semantic_policy=policy,
            exposure_predecessor=predecessor,
            exposure_output_directory=exposure_dir,
            artifact_output_directory=artifact_dir,
            cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
            proposer_transport=infrastructure_failure,
        )
    assert len(tuple(exposure_dir.glob("*.exposure.json"))) == 1
    assert not artifact_dir.exists()


def test_successful_receipt_environment_substitution_aborts_batch(
    frozen_system, tmp_path: Path
) -> None:
    (
        corpus,
        manifest,
        campaign,
        command_receipt,
        policy,
        predecessor,
        plan,
    ) = frozen_system
    transport, call_count = _stage_b_transport(wrong_launcher=True)
    with pytest.raises(GatedDevTransportIdentityError, match="launcher/cache"):
        run_gated_dev_validation(
            corpus,
            plan,
            source_corpus_manifest=manifest,
            stage_a_campaign=campaign,
            stage_a_command_receipt=command_receipt,
            visual_semantic_policy=policy,
            exposure_predecessor=predecessor,
            exposure_output_directory=tmp_path / "exposure-mismatch",
            artifact_output_directory=tmp_path / "artifact-mismatch",
            cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
            proposer_transport=transport,
        )
    # At most the already-scheduled bounded workers can be in flight when the
    # first bad receipt trips the shared fatal monitor.
    assert 1 <= call_count() <= plan.task_max_workers
    assert len(tuple((tmp_path / "exposure-mismatch").glob("*.json"))) == 1
    assert not (tmp_path / "artifact-mismatch").exists()

"""Official-archive vertical slice for the panel-soft campaign command."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import threading
from typing import Any, Mapping
import zipfile

import pytest

import bongard.panel_soft_engineering_campaign_command as campaign_module
import bongard.panel_soft_ranker as ranker_module
from bongard.canonical import canonical_digest, canonical_json
from bongard.exposure import ExposureLedger
from bongard.historical_exposure import load_historical_exposure
from bongard.object_bongard_batch import (
    FAMILIES,
    object_bongard_task_inventory_digest,
)
from bongard.object_bongard_turn_journal import (
    ObjectBongardTurnJournalSummary,
    ObjectBongardTurnRuntime,
)
from bongard.object_bongard_release_gate import ObjectBongardReleaseGateError
from bongard.official_panel_archive import OfficialPanelArchive
from bongard.panel_soft_engineering_campaign_command import (
    PanelSoftEngineeringCampaignError,
    PanelSoftEngineeringCampaignRecord,
    PanelSoftEngineeringCampaignReplayReceipt,
    PanelSoftEngineeringRankTerminal,
    PanelSoftEngineeringCampaignTaskRecord,
    PanelSoftRankFailureEvidence,
    PanelSoftRankJournalEvidence,
    cold_replay_panel_soft_engineering_campaign,
    cold_replay_panel_soft_engineering_campaign_task,
    execute_panel_soft_engineering_campaign,
    execute_panel_soft_engineering_campaign_task,
    prepare_panel_soft_engineering_campaign,
)
from bongard.panel_soft_engineering_task_runner import (
    PanelSoftEngineeringTaskRunArchive,
    PanelSoftEngineeringTaskRunStatus,
)
from bongard.prototype_scene_observer import prototype_scene_transport_source_digest
from bongard.release import OfficialReleaseDescriptor
from bongard.tests.test_panel_soft_proposer import _payload as proposer_payload
from bongard.tests.test_panel_soft_ranker import _text_receipt
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    MODEL_CATALOG,
    NO_TOOLS_ATTESTATION,
    _png,
    _receipt,
)
from bongard.transport import (
    CloudPolicyCacheSnapshot,
    CodexStructuredResult,
    PINNED_CODEX_CLI_VERSION,
)


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json(dict(value)) + b"\n")


def _unique_receipt(prompt, paths, names, schema, payload, serial):
    base = _receipt(prompt, paths, names, schema, payload)
    provisional = replace(
        base,
        thread_id=f"00000000-0000-4000-8000-{serial:012d}",
        event_stream_digest=hashlib.sha256(
            f"panel-soft-campaign-event-{serial}".encode()
        ).hexdigest(),
    )
    body = provisional.to_dict()
    body.pop("receipt_digest")
    return replace(provisional, receipt_digest=canonical_digest(body))


def _unique_text_receipt(prompt, schema, payload, serial):
    base = _text_receipt(prompt, schema, payload)
    provisional = replace(
        base,
        thread_id=f"10000000-0000-4000-8000-{serial:012d}",
        event_stream_digest=hashlib.sha256(
            f"panel-soft-rank-event-{serial}".encode()
        ).hexdigest(),
    )
    body = provisional.to_dict()
    body.pop("receipt_digest")
    return replace(provisional, receipt_digest=canonical_digest(body))


def _synthetic_runtime() -> ObjectBongardTurnRuntime:
    return ObjectBongardTurnRuntime(
        model=MODEL,
        reasoning_effort=EFFORT,
        minutes=15,
        verbose=False,
        executable="codex",
        cloud_policy_cache_snapshot=CloudPolicyCacheSnapshot(None),
        model_catalog_snapshot=MODEL_CATALOG,
        expected_launcher_digest=LAUNCHER_DIGEST,
        no_tools_attestation=NO_TOOLS_ATTESTATION,
        transport_source_digest=prototype_scene_transport_source_digest(),
    )


def _official_fixture(
    tmp_path: Path,
    *,
    selection_mode: str = "deterministic_baseline",
    workers: int = 3,
):
    historical = load_historical_exposure()
    inventory = tuple(
        sorted(
            {
                *historical.exact_official_task_ids,
                *(
                    f"{family}_panel_soft_task{index:02d}"
                    for family in FAMILIES
                    for index in range(3)
                ),
            }
        )
    )
    inventory_digest = object_bongard_task_inventory_digest(inventory)
    corpus_digest = _address({"synthetic": "panel-soft-campaign-corpus"})
    predecessor = ExposureLedger.create(corpus_digest)
    archive_path = tmp_path / "ShapeBongard_V2.zip"
    pixels: dict[str, bytes] = {}
    native_side: dict[str, int] = {}
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as bundle:
        seed = 400
        for task_id in inventory:
            family = task_id.split("_", 1)[0]
            for physical_side in ("0", "1"):
                for index in range(7):
                    panel_id = f"{family}/{task_id}/{physical_side}/{index}.png"
                    png = _png(seed)
                    seed += 1
                    pixels[panel_id] = png
                    native_side[panel_id] = 0 if physical_side == "1" else 1
                    bundle.writestr(
                        f"ShapeBongard_V2/{family}/images/"
                        f"{task_id}/{physical_side}/{index}.png",
                        png,
                    )
    archive_bytes = archive_path.read_bytes()
    split_path = tmp_path / "ShapeBongard_V2_split.json"
    _write_json(split_path, {"train": list(inventory), "val": [], "test": []})
    split_bytes = split_path.read_bytes()
    descriptor = OfficialReleaseDescriptor(
        release_id="ShapeBongard_V2-panel-soft-campaign-test",
        archive_filename=archive_path.name,
        archive_sha256="sha256:" + hashlib.sha256(archive_bytes).hexdigest(),
        archive_size_bytes=len(archive_bytes),
        split_filename=split_path.name,
        split_sha256="sha256:" + hashlib.sha256(split_bytes).hexdigest(),
        split_size_bytes=len(split_bytes),
        upstream_repository="https://github.com/NVlabs/Bongard-LOGO",
        upstream_commit="2" * 40,
        family_counts=tuple(
            (family, sum(item.startswith(family + "_") for item in inventory))
            for family in FAMILIES
        ),
        primary_split_counts=(
            ("test", 0),
            ("train", len(inventory)),
            ("val", 0),
        ),
        regime_counts=(("BA", 0), ("CM", 0), ("FF", 0), ("NV", 0)),
        task_ids_sha256=inventory_digest,
        corpus_manifest_sha256=corpus_digest,
    )
    runtime = _synthetic_runtime()
    descriptor_path = tmp_path / "release.json"
    predecessor_path = tmp_path / "predecessor.exposure.json"
    _write_json(descriptor_path, descriptor.to_dict())
    _write_json(predecessor_path, predecessor.to_dict())

    def fresh_runtime_factory(**_kwargs):
        return runtime, {
            "version": PINNED_CODEX_CLI_VERSION,
            "launcher_digest": LAUNCHER_DIGEST,
        }

    campaign = prepare_panel_soft_engineering_campaign(
        output_root=tmp_path / "store",
        executable="codex",
        expected_launcher_sha256=LAUNCHER_DIGEST,
        fresh_runtime_factory=fresh_runtime_factory,
        descriptor_path=descriptor_path,
        archive_path=archive_path,
        split_path=split_path,
        predecessor_path=predecessor_path,
        selection_seed="panel-soft-official-archive-test",
        expected_selected_task_ids=None,
        expected_plan_digest=None,
        expected_predecessor_digest=predecessor.digest,
        expected_predecessor_file_sha256=None,
        require_official_split_counts=False,
        exposure_observed_at="2026-08-09T12:00:00Z",
        selection_mode=selection_mode,
        workers=workers,
    )
    return campaign, pixels, native_side


def _successful_named_transport(
    pixels: Mapping[str, bytes], native_side: Mapping[str, int]
):
    digest_to_id = {
        hashlib.sha256(panel).hexdigest(): panel_id
        for panel_id, panel in pixels.items()
    }
    side0_phrases = {
        value
        for key, value in proposer_payload().items()
        if key.startswith("side0_") and key.endswith("_phrase")
    }
    state = {"serial": 0}
    serial_lock = threading.Lock()

    def transport(prompt, paths, names, schema, **_kwargs):
        with serial_lock:
            state["serial"] += 1
            serial = state["serial"]
        if len(paths) == 12:
            payload = proposer_payload()
        else:
            panel_digest = hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest()
            panel_id = digest_to_id[panel_digest]
            criterion_text = prompt.split("BEGIN_CRITERION_DATA\n", 1)[1].split(
                "\nEND_CRITERION_DATA", 1
            )[0]
            criteria = json.loads(criterion_text)
            payload = {
                item["criterion_alias"]: (
                    "present"
                    if (item["affirmative_description"] in side0_phrases)
                    == (native_side[panel_id] == 0)
                    else "mismatch"
                )
                for item in criteria
            }
        return CodexStructuredResult(
            payload,
            _unique_receipt(
                prompt,
                paths,
                names,
                schema,
                payload,
                serial,
            ),
        )

    return transport, state


@pytest.mark.parametrize(
    "variant",
    (
        "model",
        "reasoning_effort",
        "minutes",
        "verbose",
        "executable",
        "launcher_request",
        "fingerprint_version",
        "fingerprint_extra_field",
        "transport_source",
    ),
)
def test_runtime_factory_must_match_exact_request_before_descriptor_or_pixels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    runtime = _synthetic_runtime()
    fingerprint = {
        "version": PINNED_CODEX_CLI_VERSION,
        "launcher_digest": LAUNCHER_DIGEST,
    }
    requested = {
        "model": MODEL,
        "reasoning_effort": EFFORT,
        "minutes": 15,
        "verbose": False,
        "executable": "codex",
        "expected_launcher_sha256": LAUNCHER_DIGEST,
    }
    if variant == "model":
        runtime = replace(runtime, model="wrong-model")
    elif variant == "reasoning_effort":
        runtime = replace(runtime, reasoning_effort="high")
    elif variant == "minutes":
        runtime = replace(runtime, minutes=16)
    elif variant == "verbose":
        runtime = replace(runtime, verbose=True)
    elif variant == "executable":
        runtime = replace(runtime, executable="wrong-codex")
    elif variant == "launcher_request":
        requested["expected_launcher_sha256"] = "0" * 64
    elif variant == "fingerprint_version":
        fingerprint["version"] = "codex-cli 0.0.0"
    elif variant == "fingerprint_extra_field":
        fingerprint["unexpected"] = "not-allowed"
    elif variant == "transport_source":
        runtime = replace(runtime, transport_source_digest="0" * 64)
    else:  # pragma: no cover - parameter list is closed above.
        raise AssertionError(variant)

    descriptor_touched = False

    def forbidden_descriptor_load(cls, _path):
        nonlocal descriptor_touched
        descriptor_touched = True
        raise AssertionError("descriptor reached before runtime rejection")

    monkeypatch.setattr(
        OfficialReleaseDescriptor,
        "load",
        classmethod(forbidden_descriptor_load),
    )

    def factory(**_kwargs):
        return runtime, fingerprint

    with pytest.raises(PanelSoftEngineeringCampaignError):
        prepare_panel_soft_engineering_campaign(
            output_root=tmp_path / "must-not-exist",
            fresh_runtime_factory=factory,
            **requested,
        )
    assert descriptor_touched is False
    assert not (tmp_path / "must-not-exist").exists()


def test_preparation_never_reads_panel_and_mirror_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_panel_read(*_args, **_kwargs):
        raise AssertionError("panel pixels were read during preparation")

    monkeypatch.setattr(OfficialPanelArchive, "read_panel", forbidden_panel_read)
    campaign, _pixels, _native_side = _official_fixture(tmp_path)
    assert campaign.research_exposure_successor_path.is_file()
    assert campaign.source_manifest_receipt.object_kind == (
        "panel-soft-source-manifest"
    )
    campaign.research_exposure_successor_path.write_bytes(b"{}\n")
    with pytest.raises(PanelSoftEngineeringCampaignError):
        campaign.__post_init__()


def test_fixed_denominator_and_external_campaign_replay_with_failed_turns(
    tmp_path: Path,
) -> None:
    campaign, _pixels, _native_side = _official_fixture(tmp_path)
    calls = 0
    call_lock = threading.Lock()
    turn_barrier = threading.Barrier(3)
    worker_names: set[str] = set()

    def failing_transport(*_args, **_kwargs):
        nonlocal calls
        with call_lock:
            calls += 1
            worker_names.add(threading.current_thread().name)
        turn_barrier.wait(timeout=20)
        raise RuntimeError("synthetic physical turn failure")

    record = execute_panel_soft_engineering_campaign(
        campaign, underlying_transport=failing_transport
    )
    assert isinstance(record, PanelSoftEngineeringCampaignRecord)
    assert calls == 3
    assert len(worker_names) == 3
    assert record.workers == 3
    assert tuple(item.task_plan for item in record.task_records) == campaign.plan.tasks
    assert record.to_data()["query_denominator"] == 6
    assert record.to_data()["error_count"] == 6
    assert record.to_data()["successful_model_call_count"] == 0
    assert record.to_data()["successful_model_call_counts_by_task"] == [
        {"task_id": item.task_plan.task_id, "successful_model_call_count": 0}
        for item in record.task_records
    ]
    assert record.to_data()["terminal_turn_count"] == 3
    assert all(
        item.turn_journal_summaries[0].terminal_status == "failure"
        for item in record.task_records
    )
    with pytest.raises(PanelSoftEngineeringCampaignError):
        cold_replay_panel_soft_engineering_campaign(
            campaign,
            record,
            expected_record_digest="0" * 64,
        )
    replay = cold_replay_panel_soft_engineering_campaign(
        campaign,
        record,
        expected_record_digest=record.record_digest,
    )
    assert isinstance(replay, PanelSoftEngineeringCampaignReplayReceipt)
    assert replay.campaign_record_digest == record.record_digest
    assert replay.expected_campaign_digest == record.record_digest
    assert replay.to_data()["model_calls_made"] == 0

    calls_before_replay = calls
    interactive_output = io.StringIO()
    interactive_replay = campaign_module._await_external_campaign_digest_and_replay(
        campaign,
        record,
        input_stream=io.StringIO(record.record_digest + "\n"),
        output_stream=interactive_output,
    )
    assert interactive_replay.campaign_record_digest == record.record_digest
    assert calls == calls_before_replay
    output_lines = interactive_output.getvalue().splitlines()
    assert len(output_lines) == 2
    completion_summary, replay_summary = map(json.loads, output_lines)
    assert completion_summary["campaign_record_digest"] == record.record_digest
    assert replay_summary["model_calls_made"] == 0
    assert "support_png_base64_by_panel_id" not in interactive_output.getvalue()
    assert max(map(len, output_lines)) < 2_000

    for supplied in (
        "0" * 64 + "\n",
        "",
        record.record_digest,
    ):
        with pytest.raises(PanelSoftEngineeringCampaignError):
            campaign_module._await_external_campaign_digest_and_replay(
                campaign,
                record,
                input_stream=io.StringIO(supplied),
                output_stream=io.StringIO(),
            )
        assert calls == calls_before_replay


def test_ranked_task_persists_verified_rank_custody_before_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, pixels, native_side = _official_fixture(
        tmp_path,
        selection_mode="support_only_codex_ranker",
        workers=1,
    )
    named_transport, named_state = _successful_named_transport(
        pixels, native_side
    )
    text_calls = 0

    def text_transport(prompt, schema, **_kwargs):
        nonlocal text_calls
        text_calls += 1
        aliases = tuple(
            schema["properties"]["ordered_aliases"]["items"]["enum"]
        )
        payload = {"ordered_aliases": list(aliases)}
        return CodexStructuredResult(
            payload,
            _unique_text_receipt(
                prompt, schema, payload, 900_000 + text_calls
            ),
        )

    monkeypatch.setattr(
        campaign_module, "run_codex_text_structured", text_transport
    )
    monkeypatch.setattr(ranker_module, "run_codex_text_structured", text_transport)
    archive = campaign.archive
    task = campaign.plan.tasks[0]
    query_ids = {task.side_0_query_panel_id, task.side_1_query_panel_id}
    query_reads: list[str] = []
    original_read_panel = OfficialPanelArchive.read_panel

    def tracked_read_panel(self: OfficialPanelArchive, panel_id: str):
        if self is archive and panel_id in query_ids:
            root = campaign.release.store.root / "objects"
            assert tuple((root / "panel-soft-rank-artifact").glob("*.json"))
            assert tuple(
                (root / "panel-soft-rank-journal-evidence").glob("*.json")
            )
            query_reads.append(panel_id)
        return original_read_panel(self, panel_id)

    monkeypatch.setattr(OfficialPanelArchive, "read_panel", tracked_read_panel)
    record = execute_panel_soft_engineering_campaign_task(
        task,
        campaign=campaign,
        underlying_transport=named_transport,
    )

    assert isinstance(record.runner_record, PanelSoftEngineeringTaskRunArchive)
    assert record.runner_record.status is PanelSoftEngineeringTaskRunStatus.COMPLETE
    assert record.runner_record.rank_artifact is not None
    assert record.runner_record.rank_artifact.transport_provenance.kind == (
        "production_exactly_once_journal"
    )
    assert record.runner_record.rank_artifact.transport_provenance.benchmark_sealable
    assert record.runner_record.allow_unverified_rank_artifact is False
    assert record.rank_artifact_store_receipt is not None
    assert isinstance(record.rank_journal_evidence, PanelSoftRankJournalEvidence)
    assert record.rank_journal_evidence_store_receipt is not None
    assert record.to_data()["selection_model_attempt_count"] == 1
    assert record.to_data()["successful_selection_model_call_count"] == 1
    assert text_calls == 1
    assert named_state["serial"] == 29
    assert query_reads == [task.side_0_query_panel_id, task.side_1_query_panel_id]
    assert len(record.released_panels) == 14
    assert cold_replay_panel_soft_engineering_campaign_task(
        campaign, record, expected_record_digest=record.record_digest
    ) == record

    evidence_data = record.rank_journal_evidence.to_data()
    evidence_data["terminal_attempt_count"] = True
    with pytest.raises(PanelSoftEngineeringCampaignError):
        PanelSoftRankJournalEvidence.from_data(evidence_data)
    task_data = record.to_data()
    task_data["selection_model_attempt_count"] = True
    with pytest.raises(PanelSoftEngineeringCampaignError):
        PanelSoftEngineeringCampaignTaskRecord.from_data(task_data)


@pytest.mark.parametrize("rank_failure_kind", ("invalid_result", "transport"))
def test_rank_failure_is_typed_terminal_without_query_or_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rank_failure_kind: str,
) -> None:
    campaign, pixels, native_side = _official_fixture(
        tmp_path,
        selection_mode="support_only_codex_ranker",
        workers=1,
    )
    named_transport, named_state = _successful_named_transport(
        pixels, native_side
    )
    text_calls = 0

    def text_transport(prompt, schema, **_kwargs):
        nonlocal text_calls
        text_calls += 1
        if rank_failure_kind == "transport":
            raise RuntimeError("synthetic rank transport failure")
        aliases = tuple(
            schema["properties"]["ordered_aliases"]["items"]["enum"]
        )
        payload = {"ordered_aliases": [aliases[0]] * len(aliases)}
        return CodexStructuredResult(
            payload,
            _unique_text_receipt(
                prompt, schema, payload, 950_000 + text_calls
            ),
        )

    monkeypatch.setattr(
        campaign_module, "run_codex_text_structured", text_transport
    )
    monkeypatch.setattr(ranker_module, "run_codex_text_structured", text_transport)
    task = campaign.plan.tasks[0]
    record = execute_panel_soft_engineering_campaign_task(
        task,
        campaign=campaign,
        underlying_transport=named_transport,
    )

    assert isinstance(record.runner_record, PanelSoftEngineeringRankTerminal)
    failure = record.runner_record.rank_failure_evidence
    assert isinstance(failure, PanelSoftRankFailureEvidence)
    assert failure.failure_disposition == (
        "invalid_rank_result"
        if rank_failure_kind == "invalid_result"
        else "transport_failure"
    )
    assert (failure.successful_call_identity is not None) == (
        rank_failure_kind == "invalid_result"
    )
    assert record.runner_record.to_data()["no_baseline_fallback"] is True
    assert record.to_data()["error_count"] == 2
    assert record.to_data()["query_release_count"] == 0
    assert record.to_data()["selection_model_attempt_count"] == 1
    assert record.to_data()["successful_selection_model_call_count"] == (
        1 if rank_failure_kind == "invalid_result" else 0
    )
    assert len(record.released_panels) == 12
    assert named_state["serial"] == 25
    assert text_calls == 1
    assert record.turn_journal_summaries[-1].terminal_status == (
        "success" if rank_failure_kind == "invalid_result" else "failure"
    )
    assert cold_replay_panel_soft_engineering_campaign_task(
        campaign, record, expected_record_digest=record.record_digest
    ) == record

    failure_data = failure.to_data()
    failure_data["terminal_attempt_count"] = True
    with pytest.raises(PanelSoftEngineeringCampaignError):
        PanelSoftRankFailureEvidence.from_data(failure_data)


def test_three_rank_failures_run_concurrently_and_aggregate_in_plan_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, pixels, native_side = _official_fixture(
        tmp_path,
        selection_mode="support_only_codex_ranker",
        workers=3,
    )
    named_transport, _named_state = _successful_named_transport(
        pixels, native_side
    )
    rank_barrier = threading.Barrier(3)
    rank_lock = threading.Lock()
    rank_serial = 0
    rank_worker_names: set[str] = set()

    def invalid_text_transport(prompt, schema, **_kwargs):
        nonlocal rank_serial
        with rank_lock:
            rank_serial += 1
            serial = rank_serial
            rank_worker_names.add(threading.current_thread().name)
        rank_barrier.wait(timeout=20)
        aliases = tuple(
            schema["properties"]["ordered_aliases"]["items"]["enum"]
        )
        payload = {"ordered_aliases": [aliases[0]] * len(aliases)}
        return CodexStructuredResult(
            payload,
            _unique_text_receipt(prompt, schema, payload, 980_000 + serial),
        )

    monkeypatch.setattr(
        campaign_module,
        "run_codex_text_structured",
        invalid_text_transport,
    )
    monkeypatch.setattr(
        ranker_module,
        "run_codex_text_structured",
        invalid_text_transport,
    )
    record = execute_panel_soft_engineering_campaign(
        campaign,
        underlying_transport=named_transport,
    )

    assert tuple(item.task_plan for item in record.task_records) == campaign.plan.tasks
    assert len(rank_worker_names) == 3
    assert rank_serial == 3
    assert all(
        isinstance(item.runner_record, PanelSoftEngineeringRankTerminal)
        and item.runner_record.rank_failure_evidence.failure_disposition
        == "invalid_rank_result"
        for item in record.task_records
    )
    data = record.to_data()
    assert data["error_count"] == 6
    assert data["query_release_count"] == 0
    assert data["selection_model_attempt_count"] == 3
    assert data["successful_selection_model_call_count"] == 3
    assert cold_replay_panel_soft_engineering_campaign(
        campaign,
        record,
        expected_record_digest=record.record_digest,
    ).to_data()["model_calls_made"] == 0


def test_one_task_persists_exposure_before_support_and_query_after_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, pixels, native_side = _official_fixture(tmp_path)
    prepared = campaign.release
    archive = campaign.archive
    task = prepared.plan.tasks[0]
    support_ids = (*task.side_0_support_panel_ids, *task.side_1_support_panel_ids)
    query_ids = (task.side_0_query_panel_id, task.side_1_query_panel_id)
    digest_to_id = {
        hashlib.sha256(panel).hexdigest(): panel_id
        for panel_id, panel in pixels.items()
    }
    events: list[tuple[str, str]] = []
    original_read_panel = OfficialPanelArchive.read_panel

    def tracked_read_panel(self: OfficialPanelArchive, panel_id: str):
        if self is archive:
            assert (prepared.store.root / prepared.exposure_receipt.relative_path).is_file()
            assert (
                prepared.store.root / campaign.runtime_evidence_receipt.relative_path
            ).is_file()
            assert campaign.research_exposure_successor_path.is_file()
            assert tuple(
                (prepared.store.root / "objects" / "panel-soft-release-authority").glob(
                    "*.json"
                )
            )
            phase = "query" if panel_id in query_ids else "support"
            if phase == "query":
                assert tuple(
                    (prepared.store.root / "objects" / "panel-soft-task-freeze").glob(
                        "*.json"
                    )
                )
                assert tuple(
                    (
                        prepared.store.root
                        / "objects"
                        / "panel-soft-task-freeze-commit"
                    ).glob("*.json")
                )
            events.append((phase, panel_id))
        return original_read_panel(self, panel_id)

    monkeypatch.setattr(OfficialPanelArchive, "read_panel", tracked_read_panel)
    serial = 0
    side0_phrases = {
        value
        for key, value in proposer_payload().items()
        if key.startswith("side0_") and key.endswith("_phrase")
    }

    def transport(prompt, paths, names, schema, **_kwargs):
        nonlocal serial
        serial += 1
        if len(paths) == 12:
            payload = proposer_payload()
        else:
            panel_digest = hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest()
            panel_id = digest_to_id[panel_digest]
            criterion_text = prompt.split("BEGIN_CRITERION_DATA\n", 1)[1].split(
                "\nEND_CRITERION_DATA", 1
            )[0]
            criteria = json.loads(criterion_text)
            payload = {
                item["criterion_alias"]: (
                    "present"
                    if (item["affirmative_description"] in side0_phrases)
                    == (native_side[panel_id] == 0)
                    else "mismatch"
                )
                for item in criteria
            }
        return CodexStructuredResult(
            payload, _unique_receipt(prompt, paths, names, schema, payload, serial)
        )

    record = execute_panel_soft_engineering_campaign_task(
        task,
        campaign=campaign,
        underlying_transport=transport,
    )
    assert isinstance(record, PanelSoftEngineeringCampaignTaskRecord)
    assert isinstance(record.runner_record, PanelSoftEngineeringTaskRunArchive)
    assert record.runner_record.status is PanelSoftEngineeringTaskRunStatus.COMPLETE
    assert (record.runner_record.correct_count, record.runner_record.determinate_count) == (
        2,
        2,
    )
    assert record.execution_precommit_digest == prepared.precommit.record_digest
    assert record.exposure_successor_digest == prepared.successor.digest
    assert len(record.released_panels) == 14
    assert len(record.release_store_receipts) == 14
    assert all(
        isinstance(item, ObjectBongardTurnJournalSummary)
        and item.terminal_status == "success"
        for item in record.turn_journal_summaries
    )
    expected_calls = (
        (1 if record.runner_record.proposer_artifact.receipt is not None else 0)
        + sum(
            repeat.receipt is not None
            for artifact in record.runner_record.support_artifacts
            for repeat in artifact.repeats
        )
        + sum(
            repeat.receipt is not None
            for artifact in record.runner_record.query_artifacts
            for repeat in artifact.repeats
        )
        + (1 if record.selector_call_identity is not None else 0)
    )
    assert len(record.turn_journal_summaries) == expected_calls
    assert len(record.successful_call_identities) == expected_calls
    assert tuple(item.panel_id for item in record.released_panels[:6]) == (
        task.side_0_support_panel_ids
    )
    assert all("/1/" in item for item in task.side_0_support_panel_ids)
    assert "/1/" in task.side_0_query_panel_id
    assert len(campaign.plan.tasks) == 3
    assert events[:12] == [("support", panel_id) for panel_id in support_ids]
    assert events[12:] == [("query", panel_id) for panel_id in query_ids]
    assert len(
        tuple(
            (prepared.store.root / "objects" / "released-support-panel").glob(
                "*.json"
            )
        )
    ) == 12
    assert len(
        tuple(
            (
                prepared.store.root
                / "objects"
                / "panel-soft-released-query-panel"
            ).glob("*.json")
        )
    ) == 2
    assert PanelSoftEngineeringCampaignTaskRecord.from_data(record.to_data()) == record
    assert cold_replay_panel_soft_engineering_campaign_task(
        campaign, record, expected_record_digest=record.record_digest
    ) == record
    with pytest.raises(PanelSoftEngineeringCampaignError):
        replace(
            record,
            release_store_receipts=(
                record.release_store_receipts[1],
                record.release_store_receipts[0],
                *record.release_store_receipts[2:],
            ),
        )
    with pytest.raises(PanelSoftEngineeringCampaignError):
        replace(
            record,
            turn_journal_summaries=(
                replace(record.turn_journal_summaries[0], terminal_status="failure"),
                *record.turn_journal_summaries[1:],
            ),
        )
    tampered_path = (
        prepared.store.root / record.release_store_receipts[0].relative_path
    )
    tampered_path.write_bytes(b"{}\n")
    with pytest.raises((PanelSoftEngineeringCampaignError, ObjectBongardReleaseGateError)):
        cold_replay_panel_soft_engineering_campaign_task(
            campaign, record, expected_record_digest=record.record_digest
        )

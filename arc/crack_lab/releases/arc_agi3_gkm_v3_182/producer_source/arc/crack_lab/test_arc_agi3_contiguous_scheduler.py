from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

import arc_agi3_contiguous_scheduler as S
import arc_agi3_contiguous_supervisor as Supervisor
import arc_agi3_codex_app_server_transport as Transport
import arc_agi3_arena_rpc as ArenaRpc


def _inventory() -> dict[str, int]:
    return Supervisor.authoritative_inventory()


def _write_source(root: Path, *, extra: str = "") -> str:
    root.mkdir(parents=True)
    (root / "legs.py").write_text(
        "def retained(env):\n    return 1\n" + extra
    )
    (root / "players.py").write_text(
        "from legs import retained\n"
        "def play_level_1(env):\n"
        "    return retained(env)\n"
    )
    (root / "solve.py").write_text(
        "from players import play_level_1\n"
        "def solve(env):\n"
        "    return play_level_1(env)\n"
    )
    digest = hashlib.sha256()
    for path in sorted(root.iterdir()):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(
            hashlib.sha256(path.read_bytes()).hexdigest().encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _snapshot(
    tmp_path: Path,
    *,
    limit_units: int | None = None,
    active_games: set[str] = set(),
    special_wip_game: str | None = None,
) -> S.CampaignSnapshot:
    parent = tmp_path / "parent"
    parent_sha = _write_source(parent)
    wip_root = tmp_path / "wip"
    candidate = wip_root / "source"
    candidate_sha = (
        _write_source(candidate)
        if special_wip_game is not None
        else None
    )
    inventory = _inventory()
    checkpoint_sha = "1" * 64
    frontiers = []
    for game, target in inventory.items():
        frontier_sha = S._frontier_digest(game, 0, checkpoint_sha)
        wip = None
        if game == special_wip_game:
            wip = S.WipBinding(
                snapshot_id=f"wip:{game}",
                wip_root_path=str(wip_root),
                wip_tree_sha256="a" * 64,
                solver_source_path=str(candidate),
                solver_source_tree_sha256=candidate_sha,
                game=game,
                target_level=1,
                parent_checkpoint_sha256=checkpoint_sha,
                frontier_sha256=frontier_sha,
                codex_thread_id="12345678-1234-4234-8234-123456789abc",
                final_thread_binding_path=str(
                    tmp_path / "final_thread_binding.json"
                ),
                final_thread_binding_sha256="2" * 64,
                wip_export_receipt_path=str(
                    tmp_path / "wip_export.json"
                ),
                wip_export_receipt_sha256="9" * 64,
                final_transcript_chain_receipt_path=str(
                    tmp_path / "final_transcript_chain.json"
                ),
                final_transcript_chain_receipt_sha256="3" * 64,
                transcript_chain_sha256="3" * 64,
                controller_state_scan_receipt_path=str(
                    tmp_path / "controller_state_scan.json"
                ),
                controller_state_scan_receipt_sha256="4" * 64,
                retained_canary_scan_receipt_path=str(
                    tmp_path / "retained_canary_scan.json"
                ),
                retained_canary_scan_receipt_sha256="5" * 64,
                taint_scan_receipt_path=str(
                    tmp_path / "taint_scan.json"
                ),
                taint_scan_receipt_sha256="6" * 64,
                token_usage_receipt_path=str(
                    tmp_path / "token_usage.json"
                ),
                token_usage_receipt_sha256="7" * 64,
                provider_usage_receipt_path=str(
                    tmp_path / "provider_usage.json"
                ),
                provider_usage_receipt_sha256="b" * 64,
                app_server_state_dir=str(tmp_path / "state"),
                app_server_state_tree_sha256="4" * 64,
                wip_publication_receipt_path=str(
                    tmp_path / "wip_publication.json"
                ),
                wip_publication_receipt_sha256="8" * 64,
                supervisory_handoff_sha256=None,
                supervisory_native_reproduction_receipt_path=None,
                supervisory_native_reproduction_receipt_sha256=None,
            )
        evidence = S.selection_evidence(
            parent_source_path=str(parent),
            parent_source_tree_sha256=parent_sha,
            candidate_source_path=(str(candidate) if wip else None),
            candidate_source_tree_sha256=(candidate_sha if wip else None),
        )
        frontiers.append(
            S.Frontier(
                game=game,
                target=target,
                reached=0,
                no_progress=1 if wip else 0,
                last_dispatch_sequence=0,
                parent_checkpoint_sha256=checkpoint_sha,
                parent_source_path=str(parent),
                parent_source_tree_sha256=parent_sha,
                frontier_sha256=frontier_sha,
                active_attempt_id=(
                    f"active:{game}" if game in active_games else None
                ),
                draining=False,
                blocked_reason=None,
                wip=wip,
                evidence=evidence,
                public_observation_receipt_sha256s=(),
                observation_ledger_sha256=(
                    S.public_observation_ledger_sha256(
                        game=game,
                        frontier_sha256=frontier_sha,
                        parent_checkpoint_sha256=checkpoint_sha,
                        receipt_sha256s=(),
                    )
                ),
            )
        )
    return S.CampaignSnapshot(
        campaign_id="campaign:test",
        journal_head_sequence=1,
        journal_head_digest="5" * 64,
        inventory=tuple(inventory.items()),
        max_lanes=6,
        frontiers=tuple(frontiers),
        budget=S.BudgetState(
            cost_window_id="window:test",
            limit_units=limit_units,
            settled_units=0,
        ),
    )


def _enabled_auxiliary_configuration() -> S.AuxiliaryLaunchConfiguration:
    return S.AuxiliaryLaunchConfiguration(
        schema=1,
        automatic_dispatch_enabled=True,
        backend_attested=True,
        input_bundle_attested=True,
        admission_gate_attested=True,
        model="gpt-5.6-sol",
        reasoning_effort="max",
        backend_contract_sha256="a" * 64,
        input_bundle_contract_sha256="b" * 64,
        admission_contract_sha256="c" * 64,
        supervisory_proposer=S.SupervisoryProposerLaunchConfiguration(
            schema=1,
            role=S.SUPERVISORY_PROPOSER_ROLE,
            automatic_dispatch_enabled=False,
            model="gpt-5.6-sol",
            reasoning_effort="max",
            context_limit_tokens=100_000,
            max_concurrency=1,
        ),
    )


def _auxiliary_snapshot(
    tmp_path: Path, *, no_progress: int
) -> tuple[S.CampaignSnapshot, S.Frontier]:
    base = _snapshot(tmp_path)
    selected_game = base.frontiers[0].game
    frontiers = []
    selected = None
    for frontier in base.frontiers:
        if frontier.game == selected_game:
            public_receipts = ("d" * 64,)
            selected = dataclasses.replace(
                frontier,
                no_progress=no_progress,
                last_dispatch_sequence=100,
                active_attempt_id="active:max-proposer",
                public_observation_receipt_sha256s=public_receipts,
                observation_ledger_sha256=(
                    S.public_observation_ledger_sha256(
                        game=frontier.game,
                        frontier_sha256=frontier.frontier_sha256,
                        parent_checkpoint_sha256=(
                            frontier.parent_checkpoint_sha256
                        ),
                        receipt_sha256s=public_receipts,
                    )
                ),
            )
            frontiers.append(selected)
        else:
            frontiers.append(
                dataclasses.replace(
                    frontier, blocked_reason="test-only-blocker"
                )
            )
    assert selected is not None
    settlements = []
    for index in range(no_progress):
        policy = S.retry_policy(index)
        settlements.append(S.CleanProposerSettlement(
            schema=1,
            game=selected.game,
            frontier_sha256=selected.frontier_sha256,
            parent_checkpoint_sha256=(
                selected.parent_checkpoint_sha256
            ),
            attempt_id=f"attempt:{index}",
            scheduler_decision_id=f"decision:{index}",
            no_progress_before=index,
            effort=policy.effort,
            soft_allocation_seconds=policy.soft_allocation_seconds,
            requested_wip_mode=policy.requested_wip_mode,
            supervisory_handoff_sha256=None,
            result_sequence=index + 2,
            result_digest=S.sha256_json({
                "kind": "clean_no_progress",
                "index": index,
            }),
        ))
    requests = []
    for origin in settlements:
        draft = S.NativeSidecarRequestDraft(
            schema=1,
            kind="NATIVE_SIDECAR_REQUEST_DRAFT",
            request_id=f"request:{origin.attempt_id}",
            game=selected.game,
            frontier_sha256=selected.frontier_sha256,
            parent_checkpoint_sha256=(
                selected.parent_checkpoint_sha256
            ),
            native_attempt_id=origin.attempt_id,
            semantic_brief=(
                "Falsify the unresolved exact-frontier mechanism using the "
                f"cited public observation at retry {origin.no_progress_before}."
            ),
            cited_public_observation_receipt_sha256s=("d" * 64,),
            scheduler_authored=False,
            live_lineage_mutation_authority=False,
            promotion_authority=False,
            draft_sha256="",
        )
        draft = dataclasses.replace(
            draft,
            draft_sha256=S.sha256_json(
                S._native_sidecar_request_draft_body(draft)
            ),
        )
        requests.append(
            S.native_sidecar_request_from_draft(
                draft, settlement=origin
            )
        )
    return (
        dataclasses.replace(
            base,
            max_lanes=2,
            frontiers=tuple(frontiers),
            clean_proposer_settlements=tuple(settlements),
            sidecar_requests=tuple(requests),
        ),
        selected,
    )


def _append_event(
    campaign: Path,
    events: list[dict],
    *,
    event_id: str,
    kind: str,
    payload: dict,
) -> dict:
    sequence = len(events) + 1
    body = {
        "schema": 1,
        "sequence": sequence,
        "event_id": event_id,
        "kind": kind,
        "recorded_at": float(sequence),
        "previous_digest": events[-1]["digest"] if events else None,
        "payload": payload,
    }
    event = {**body, "digest": S._event_digest(body)}
    journal = campaign / "attempt_journal"
    journal.mkdir(parents=True, exist_ok=True)
    (journal / f"{sequence:020d}-{event_id}.json").write_bytes(
        S.canonical_json(event) + b"\n"
    )
    events.append(event)
    return event


def _genesis(campaign: Path, source: Path, source_sha: str) -> list[dict]:
    inventory = _inventory()
    checkpoint_root = campaign / "zero_checkpoints"
    checkpoint_root.mkdir(parents=True)
    zero_checkpoints = {}
    for game in inventory:
        path = checkpoint_root / f"{game}.json"
        path.write_text(
            json.dumps(
                {
                    "game": game,
                    "reached": 0,
                    "total_marginal_C": 0,
                    "records": [],
                    "final_path": [],
                    "validated": False,
                },
                sort_keys=True,
            )
        )
        zero_checkpoints[game] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    zero_sources = {
        game: {"path": str(source), "sha256": source_sha}
        for game in inventory
    }
    events: list[dict] = []
    _append_event(
        campaign,
        events,
        event_id="campaign:genesis",
        kind="GENESIS",
        payload={
            "schema": 1,
            "campaign_id": "campaign:test",
            "inventory": inventory,
            "inventory_sha256": S.inventory_sha256(inventory),
            "max_lanes": 6,
            "limit": None,
            "limit_units": None,
            "cost_window_id": "window:test",
            "zero_checkpoints": zero_checkpoints,
            "zero_sources": zero_sources,
        },
    )
    return events


def _one_reserved_campaign(tmp_path: Path, *, reserve: bool = True):
    campaign = tmp_path / "campaign"
    source = tmp_path / "source"
    source_sha = _write_source(source)
    events = _genesis(campaign, source, source_sha)
    inventory = _inventory()
    zero_checkpoints = events[0]["payload"]["zero_checkpoints"]
    evidence = S.selection_evidence(
        parent_source_path=str(source),
        parent_source_tree_sha256=source_sha,
    )
    snapshot = S.CampaignSnapshot(
        campaign_id="campaign:test",
        journal_head_sequence=1,
        journal_head_digest=events[0]["digest"],
        inventory=tuple(inventory.items()),
        max_lanes=6,
        frontiers=tuple(
            S.Frontier(
                game=game,
                target=target,
                reached=0,
                no_progress=0,
                last_dispatch_sequence=0,
                parent_checkpoint_sha256=zero_checkpoints[game]["sha256"],
                parent_source_path=str(source),
                parent_source_tree_sha256=source_sha,
                frontier_sha256=S._frontier_digest(
                    game, 0, zero_checkpoints[game]["sha256"]
                ),
                active_attempt_id=None,
                draining=False,
                blocked_reason=None,
                wip=None,
                evidence=evidence,
                public_observation_receipt_sha256s=(),
                observation_ledger_sha256=(
                    S.public_observation_ledger_sha256(
                        game=game,
                        frontier_sha256=S._frontier_digest(
                            game,
                            0,
                            zero_checkpoints[game]["sha256"],
                        ),
                        parent_checkpoint_sha256=(
                            zero_checkpoints[game]["sha256"]
                        ),
                        receipt_sha256s=(),
                    )
                ),
            )
            for game, target in inventory.items()
        ),
        budget=S.BudgetState("window:test", None, 0),
    )
    decision = S.build_decision(
        snapshot,
        decision_id="decision:1",
        attempt_id="attempt:1",
        generation_id="generation:1",
        reservation_id="reservation:1",
    )
    assert decision is not None
    checkpoint_sha = zero_checkpoints[decision.choice.game]["sha256"]
    _append_event(
        campaign,
        events,
        event_id="decision:1",
        kind="SCHEDULER_DECISION",
        payload={"decision": S.decision_to_dict(decision)},
    )
    if reserve:
        generation_dir = (
            campaign / "generations" / "generation:1"
        )
        _append_event(
            campaign,
            events,
            event_id="attempt:1:reserved",
            kind="ATTEMPT_RESERVED",
            payload={
                "attempt_id": "attempt:1",
                "reservation": {
                    **S.reservation_binding(decision),
                    "campaign_id": "campaign:test",
                    "generation_id": "generation:1",
                    "attempt_id": "attempt:1",
                    "game": decision.choice.game,
                    "target_level": decision.choice.target_level,
                    "authoritative_target":
                        decision.choice.authoritative_target,
                    "parent_checkpoint_path": str(
                        campaign
                        / "zero_checkpoints"
                        / f"{decision.choice.game}.json"
                    ),
                    "parent_checkpoint_sha256": checkpoint_sha,
                    "frontier_sha256": S._frontier_digest(
                        decision.choice.game, 0, checkpoint_sha
                    ),
                    "generation_dir": str(generation_dir),
                    "host_transcript_path": str(
                        generation_dir / "host" / "backend.jsonl"
                    ),
                    "parent_source_path": str(source),
                    "parent_source_tree_sha256": source_sha,
                    "effort": decision.choice.effort,
                    "soft_allocation_seconds":
                        decision.choice.soft_allocation_seconds,
                    "wip_mode": decision.choice.effective_wip_mode,
                    "thread_mode": decision.choice.thread_mode,
                    "resume_thread_id": None,
                    "resume_thread_binding_sha256": None,
                    "wip": None,
                    "cost_limit_remaining": None,
                },
            },
        )
    return campaign, source, source_sha, events, decision


def _append_minimal_completed_lifecycle(
    campaign: Path,
    events: list[dict],
    *,
    attempt_id: str = "attempt:1",
    result_kind: str = "clean_no_progress",
) -> None:
    reservation_event = next(
        event
        for event in events
        if event["kind"] == "ATTEMPT_RESERVED"
        and event["payload"]["attempt_id"] == attempt_id
    )
    reservation = reservation_event["payload"]["reservation"]
    host_transcript_path = reservation["host_transcript_path"]
    receipt_root = (
        Path(host_transcript_path).parent / "public_observations"
    )
    receipt_root.mkdir(parents=True)
    basis = {
        "schema": 1,
        "kind": ArenaRpc.PUBLIC_ACTION_BASIS_KIND,
        "operation_index": 0,
        "previous_public_action_basis_sha256":
            ArenaRpc.PUBLIC_ACTION_BASIS_GENESIS_SHA256,
        "operation": {"op": "open"},
    }
    signature = {
        "schema": 1,
        "kind": ArenaRpc.PUBLIC_RESPONSE_SIGNATURE_KIND,
        "operation_index": 0,
        "result": {
            "binding_sha256": "c" * 64,
            "snapshot": {
                "frame": [[0]],
                "actions": [1],
                "levels_completed": reservation["target_level"] - 1,
                "terminal": False,
            },
        },
    }
    receipt = {
        "schema": 1,
        "kind": ArenaRpc.PUBLIC_OBSERVATION_RECEIPT_KIND,
        "game": reservation["game"],
        "frontier_sha256": reservation["frontier_sha256"],
        "parent_checkpoint_sha256":
            reservation["parent_checkpoint_sha256"],
        "public_action_basis": basis,
        "public_action_basis_sha256": S.sha256_json(basis),
        "public_response_signature": signature,
        "public_response_signature_sha256": S.sha256_json(signature),
    }
    raw = ArenaRpc.public_observation_receipt_bytes(receipt)
    digest = hashlib.sha256(raw).hexdigest()
    (receipt_root / f"{digest}.json").write_bytes(raw)
    transition = S.public_observation_transition(
        attempt_id=attempt_id,
        generation_id=reservation["generation_id"],
        game=reservation["game"],
        frontier_sha256=reservation["frontier_sha256"],
        parent_checkpoint_sha256=(
            reservation["parent_checkpoint_sha256"]
        ),
        host_transcript_path=host_transcript_path,
        result_kind=result_kind,
        receipt_sha256s=(digest,),
    )
    simple = (
        "ATTEMPT_PREPARED",
        "BACKEND_PREPARED",
        "ATTEMPT_LAUNCHED",
    )
    for index, kind in enumerate(simple, 1):
        _append_event(
            campaign,
            events,
            event_id=f"{attempt_id}:lifecycle:{index}",
            kind=kind,
            payload={"attempt_id": attempt_id},
        )
    _append_event(
        campaign,
        events,
        event_id=f"{attempt_id}:lifecycle:4",
        kind="ATTEMPT_EXITED",
        payload={
            "attempt_id": attempt_id,
            "terminal": {
                "status": "exited",
                "observation_sha256": "d" * 64,
                "exit_code": 0,
            },
        },
    )
    _append_event(
        campaign,
        events,
        event_id=f"{attempt_id}:observations:staging",
        kind="ATTEMPT_PUBLIC_OBSERVATIONS_STAGING",
        payload={
            "attempt_id": attempt_id,
            "transition": transition,
        },
    )
    _append_event(
        campaign,
        events,
        event_id=f"{attempt_id}:lifecycle:5",
        kind="ATTEMPT_COLLECTED",
        payload={
            "attempt_id": attempt_id,
            "collection": {
                "result": {"kind": result_kind},
                "native_public_observation_receipt_sha256s": [
                    digest
                ],
                "host_transcript_path": host_transcript_path,
                "structured_provider_outcome": "completed",
            },
            "public_observation_transition_sha256":
                S.sha256_json(transition),
        },
    )
    _append_event(
        campaign,
        events,
        event_id=f"{attempt_id}:lifecycle:6",
        kind="ATTEMPT_TORN_DOWN",
        payload={
            "attempt_id": attempt_id,
            "teardown": {
                "proof_sha256": "e" * 64,
                "container_inspect_absent": True,
                "container_top_absent": True,
                "identity_query_empty": True,
                "no_descendants": True,
                "app_server_process_absent": True,
                "app_server_process_group_absent": True,
            },
        },
    )


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _write_receipt(path: Path, value: dict) -> str:
    path.write_bytes(S.canonical_json(value) + b"\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _public_observation_transition_fixture(
    tmp_path: Path,
    *,
    result_kind: str = "clean_no_progress",
):
    attempt_id = "attempt:observations"
    generation_id = "generation:observations"
    game = "ar25"
    frontier_sha256 = "a" * 64
    parent_checkpoint_sha256 = "b" * 64
    host_transcript_path = (
        tmp_path / generation_id / "host" / "backend.jsonl"
    )
    receipt_root = host_transcript_path.parent / "public_observations"
    receipt_root.mkdir(parents=True)
    basis = {
        "schema": 1,
        "kind": ArenaRpc.PUBLIC_ACTION_BASIS_KIND,
        "operation_index": 0,
        "previous_public_action_basis_sha256":
            ArenaRpc.PUBLIC_ACTION_BASIS_GENESIS_SHA256,
        "operation": {"op": "open"},
    }
    signature = {
        "schema": 1,
        "kind": ArenaRpc.PUBLIC_RESPONSE_SIGNATURE_KIND,
        "operation_index": 0,
        "result": {
            "binding_sha256": "c" * 64,
            "snapshot": {
                "frame": [[0]],
                "actions": [1],
                "levels_completed": 0,
                "terminal": False,
            },
        },
    }
    receipt = {
        "schema": 1,
        "kind": ArenaRpc.PUBLIC_OBSERVATION_RECEIPT_KIND,
        "game": game,
        "frontier_sha256": frontier_sha256,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "public_action_basis": basis,
        "public_action_basis_sha256": S.sha256_json(basis),
        "public_response_signature": signature,
        "public_response_signature_sha256": S.sha256_json(signature),
    }
    raw = ArenaRpc.public_observation_receipt_bytes(receipt)
    digest = hashlib.sha256(raw).hexdigest()
    (receipt_root / f"{digest}.json").write_bytes(raw)
    transition = S.public_observation_transition(
        attempt_id=attempt_id,
        generation_id=generation_id,
        game=game,
        frontier_sha256=frontier_sha256,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        host_transcript_path=str(host_transcript_path),
        result_kind=result_kind,
        receipt_sha256s=(digest,),
    )
    expected = {
        "attempt_id": attempt_id,
        "generation_id": generation_id,
        "game": game,
        "frontier_sha256": frontier_sha256,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "host_transcript_path": str(host_transcript_path),
        "result_kind": result_kind,
        "receipt_sha256s": (digest,),
        "reopen_receipts": True,
    }
    return transition, expected, receipt_root / f"{digest}.json"


@pytest.mark.parametrize(
    ("result_kind", "authoritative"),
    (
        ("clean_no_progress", True),
        ("candidate", True),
        ("tainted", False),
        ("protocol_invalid", False),
        ("infrastructure", False),
        ("blocker", False),
    ),
)
def test_public_observation_transition_grants_only_clean_authority(
    tmp_path, result_kind, authoritative
):
    transition, expected, receipt_path = (
        _public_observation_transition_fixture(
            tmp_path, result_kind=result_kind
        )
    )
    admitted = S.validate_public_observation_transition(
        transition, **expected
    )
    assert admitted == (
        (receipt_path.stem,) if authoritative else ()
    )
    assert transition["authority"] == (
        "same_frontier_lineage"
        if authoritative
        else "forensic_only_no_lineage_authority"
    )


def test_public_observation_transition_rejects_forged_malformed_and_cross_attempt(
    tmp_path,
):
    transition, expected, receipt_path = (
        _public_observation_transition_fixture(tmp_path / "valid")
    )
    original = receipt_path.read_bytes()
    receipt_path.write_bytes(original + b" ")
    with pytest.raises(
        S.SchedulerError, match="bytes differ"
    ):
        S.validate_public_observation_transition(
            transition, **expected
        )
    receipt_path.write_bytes(original)

    malformed_root = tmp_path / "malformed"
    malformed_transition, malformed_expected, malformed_path = (
        _public_observation_transition_fixture(malformed_root)
    )
    malformed_raw = S.canonical_json({"schema": 1})
    malformed_digest = hashlib.sha256(malformed_raw).hexdigest()
    malformed_path.unlink()
    replacement = malformed_path.parent / f"{malformed_digest}.json"
    replacement.write_bytes(malformed_raw)
    malformed_transition["receipt_sha256s"] = [malformed_digest]
    malformed_expected["receipt_sha256s"] = (malformed_digest,)
    with pytest.raises(
        S.SchedulerError, match="malformed"
    ):
        S.validate_public_observation_transition(
            malformed_transition, **malformed_expected
        )

    other_host = (
        tmp_path
        / "other-attempt"
        / "generation:other"
        / "host"
        / "backend.jsonl"
    )
    cross_expected = {
        **expected,
        "attempt_id": "attempt:other",
        "generation_id": "generation:other",
        "host_transcript_path": str(other_host),
    }
    with pytest.raises(
        S.SchedulerError, match="crosses collection identity"
    ):
        S.validate_public_observation_transition(
            transition, **cross_expected
        )


def _terminal_wip(
    tmp_path: Path,
    decision: S.SchedulerDecision,
    *,
    missing_root: bool = False,
    forged_provider_settlement: bool = False,
) -> S.WipBinding:
    root = tmp_path / "retained_wip"
    source = root / "source"
    source_sha = _write_source(source)
    wip_tree_sha = _tree_sha256(root)
    state = tmp_path / "retained_app_server_state"
    state.mkdir()
    (state / "state.json").write_text('{"cursor":1}\n')
    state_sha = _tree_sha256(state)
    receipts = tmp_path / "retained_receipts"
    receipts.mkdir()
    campaign_id = "campaign:test"
    attempt_id = "attempt:1"
    thread_id = "12345678-1234-4234-8234-123456789abc"
    common = {
        "schema": 1,
        "campaign_id": campaign_id,
        "generation_id": "generation:1",
        "attempt_id": attempt_id,
        "attempt_spec_sha256": "d" * 64,
    }
    transcript_chain_sha = "a" * 64
    transcript_path = receipts / "transcript.json"
    transcript_sha = _write_receipt(
        transcript_path,
        {
            **common,
            "kind": "contiguous_final_transcript_chain",
            "thread_id": thread_id,
            "chain_head_sha256": transcript_chain_sha,
        },
    )
    controller_path = receipts / "controller.json"
    controller_sha = _write_receipt(
        controller_path,
        {
            **common,
            "kind": "contiguous_controller_state_scan",
            "controller_state_scan": {
                "status": "CLEAN",
                "hits": [],
                "canary_occurrences": 0,
            },
        },
    )
    retained_path = receipts / "retained.json"
    retained_sha = _write_receipt(
        retained_path,
        {
            **common,
            "kind": "contiguous_retained_canary_scan",
            "retained_canary_scan": {
                "status": "CLEAN",
                "hits": [],
                "canary_occurrences": 0,
            },
        },
    )
    taint_path = receipts / "taint.json"
    taint_sha = _write_receipt(
        taint_path,
        {
            **common,
            "kind": "contiguous_taint_scan",
            "status": "CLEAN",
            "hits": [],
            "canary_occurrences": 0,
        },
    )
    token_path = receipts / "token.json"
    token_sha = _write_receipt(
        token_path,
        {
            **common,
            "kind": "contiguous_token_usage",
            "thread_id": thread_id,
            "final_event_observed": True,
            "observations": [{"input_tokens": 1, "output_tokens": 1}],
        },
    )
    usage_observations = [{
        "total": {
            "inputTokens": 1,
            "cachedInputTokens": 0,
            "outputTokens": 1,
            "reasoningOutputTokens": 0,
            "totalTokens": 2,
        },
    }]
    pre_response = {
        "rateLimitsByLimitId": {
            "codex": {
                "planType": "team",
                "credits": {
                    "hasCredits": True,
                    "unlimited": True,
                    "balance": None,
                },
                "spendControlReached": False,
            },
        },
    }
    post_response = {
        "rateLimitsByLimitId": {
            "codex": {
                "planType": "team",
                "primary": {
                    "usedPercent": 0,
                    "resetsAt": 2_000_000,
                    "windowDurationMins": 7 * 24 * 60,
                },
            },
        },
    }
    pre_window = Transport.normalize_provider_usage_window(
        pre_response,
        phase="preflight",
        observation_sequence=1,
        authenticated_response_sha256=hashlib.sha256(
            Transport.canonical_json({"id": 1, "result": pre_response})
        ).hexdigest(),
        transcript_chain_sha256="b" * 64,
    )
    post_window = Transport.normalize_provider_usage_window(
        post_response,
        phase="postflight",
        observation_sequence=2,
        authenticated_response_sha256=hashlib.sha256(
            Transport.canonical_json({"id": 2, "result": post_response})
        ).hexdigest(),
        transcript_chain_sha256="c" * 64,
    )
    settlement = Transport.settle_provider_usage(
        pre_window,
        post_window,
        token_usage_observations=usage_observations,
    )
    settlement_payload = dataclasses.asdict(settlement)
    if forged_provider_settlement:
        settlement_payload["charge"] = 1.0
    provider_path = receipts / "provider.json"
    provider_sha = _write_receipt(
        provider_path,
        {
            **common,
            "kind": "contiguous_provider_usage",
            "generation_id": "generation:1",
            "attempt_spec_sha256": "d" * 64,
            "thread_id": thread_id,
            "turn_id": "turn:1",
            "token_usage_observations": usage_observations,
            "pre_provider_usage_window":
                Transport.provider_usage_window_to_dict(pre_window),
            "post_provider_usage_window":
                Transport.provider_usage_window_to_dict(post_window),
            "provider_usage_settlement": settlement_payload,
        },
    )
    selected = next(
        frontier
        for frontier in decision.eligible_frontiers
        if frontier["game"] == decision.choice.game
    )
    export_path = receipts / "wip_export.json"
    export_sha = _write_receipt(
        export_path,
        {
            **common,
            "kind": "contiguous_wip_export",
            "game": decision.choice.game,
            "target_level": decision.choice.target_level,
            "parent_checkpoint_sha256":
                selected["parent_checkpoint_sha256"],
            "frontier_sha256": selected["frontier_sha256"],
            "wip_tree_sha256": wip_tree_sha,
            "solver_source_tree_sha256": source_sha,
        },
    )
    final_path = receipts / "final_binding.json"
    final_sha = _write_receipt(
        final_path,
        {
            **common,
            "kind": "contiguous_final_thread_binding",
            "thread_id": thread_id,
            "transcript_chain_sha256": transcript_chain_sha,
            "token_usage_receipt_sha256": token_sha,
            "provider_usage_receipt_sha256": provider_sha,
            "app_server_state_tree_sha256": state_sha,
            "controller_state_scan_receipt_sha256": controller_sha,
            "retained_canary_scan_receipt_sha256": retained_sha,
            "taint_scan_receipt_sha256": taint_sha,
            "wip_export_receipt_sha256": export_sha,
        },
    )
    publication_path = receipts / "publication.json"
    declared_root = (
        tmp_path / "missing_retained_wip" if missing_root else root
    )
    declared_source = declared_root / "source"
    wip = S.WipBinding(
        snapshot_id="wip:attempt:1",
        wip_root_path=str(declared_root),
        wip_tree_sha256=wip_tree_sha,
        solver_source_path=str(declared_source),
        solver_source_tree_sha256=source_sha,
        game=decision.choice.game,
        target_level=decision.choice.target_level,
        parent_checkpoint_sha256=selected["parent_checkpoint_sha256"],
        frontier_sha256=selected["frontier_sha256"],
        codex_thread_id=thread_id,
        final_thread_binding_path=str(final_path),
        final_thread_binding_sha256=final_sha,
        wip_export_receipt_path=str(export_path),
        wip_export_receipt_sha256=export_sha,
        final_transcript_chain_receipt_path=str(transcript_path),
        final_transcript_chain_receipt_sha256=transcript_sha,
        transcript_chain_sha256=transcript_chain_sha,
        controller_state_scan_receipt_path=str(controller_path),
        controller_state_scan_receipt_sha256=controller_sha,
        retained_canary_scan_receipt_path=str(retained_path),
        retained_canary_scan_receipt_sha256=retained_sha,
        taint_scan_receipt_path=str(taint_path),
        taint_scan_receipt_sha256=taint_sha,
        token_usage_receipt_path=str(token_path),
        token_usage_receipt_sha256=token_sha,
        provider_usage_receipt_path=str(provider_path),
        provider_usage_receipt_sha256=provider_sha,
        app_server_state_dir=str(state),
        app_server_state_tree_sha256=state_sha,
        wip_publication_receipt_path=str(publication_path),
        wip_publication_receipt_sha256="0" * 64,
        supervisory_handoff_sha256=None,
        supervisory_native_reproduction_receipt_path=None,
        supervisory_native_reproduction_receipt_sha256=None,
    )
    publication = S.wip_binding_to_dict(wip)
    publication.pop("wip_publication_receipt_path")
    publication.pop("wip_publication_receipt_sha256")
    publication_sha = _write_receipt(
        publication_path,
        {
            **common,
            "kind": "contiguous_wip_publication",
            **publication,
        },
    )
    return dataclasses.replace(
        wip, wip_publication_receipt_sha256=publication_sha
    )


def _append_clean_no_progress_result(
    campaign: Path,
    events: list[dict],
    *,
    wip: S.WipBinding | None,
) -> None:
    _append_minimal_completed_lifecycle(campaign, events)
    _append_event(
        campaign,
        events,
        event_id="attempt:1:result",
        kind="ATTEMPT_RESULT",
        payload={
            "attempt_id": "attempt:1",
            "kind": "clean_no_progress",
            "cost_used": 0.0,
            "authenticated_cost_units": 0,
            "budget_reservation_id": "reservation:1",
            "scheduler_decision_id": "decision:1",
            "reason": "",
            "candidate": None,
            "wip": (
                S.wip_binding_to_dict(wip) if wip is not None else None
            ),
        },
    )


def test_retry_ladder_is_monotone_and_enters_long_coherence():
    rows = [S.retry_policy(index) for index in range(20)]
    ranks = [
        {"medium": 0, "high": 1, "xhigh": 2, "max": 3}[row.effort]
        for row in rows
    ]
    assert ranks == sorted(ranks)
    assert [row.soft_allocation_seconds // 60 for row in rows[:10]] == [
        15, 20, 25, 40, 60, 90, 120, 180, 180, 300
    ]
    assert rows[9].phase == "long_coherence"
    assert rows[9].requested_wip_mode == "exclude"
    assert rows[10].requested_wip_mode == "restore_clean_same_frontier"


def test_stubborn_frontier_adds_independent_sidecars_by_complexity():
    frontier_sha = "a" * 64
    disabled = S.auxiliary_analysis_policy(
        4, frontier_sha256=frontier_sha
    )
    assert disabled.phase == "disabled"
    assert disabled.model_effort_source is None
    assert disabled.specializations == ()

    diagnose = S.auxiliary_analysis_policy(
        5, frontier_sha256=frontier_sha
    )
    assert diagnose.phase == "diagnose"
    assert diagnose.role == "independent_side_expert"
    assert diagnose.model_effort_source == "campaign_launch_manifest"
    assert diagnose.max_parallel == 1
    assert diagnose.specializations == ("complexity_diagnosis",)
    assert diagnose.workspace_mode == "immutable_private_copy"
    assert diagnose.output_mode == "quarantine_only"
    assert diagnose.must_differ_from_active_lanes is True
    assert diagnose.minimum_socratic_passes == 1
    assert diagnose.mutates_live_lineage is False

    profile = S.ComplexityProfile(
        schema=1,
        profile_id="profile:1",
        round_index=0,
        frontier_sha256=frontier_sha,
        observation_receipt_sha256="b" * 64,
        taint_scan_receipt_sha256="c" * 64,
        priorities=(
            "prefix_compression",
            "exact_planning",
            "state_representation",
        ),
    )
    one = S.auxiliary_analysis_policy(
        5,
        frontier_sha256=frontier_sha,
        profile=profile,
    )
    assert one.specializations == ("prefix_compression",)
    assert one.max_parallel == 1

    two = S.auxiliary_analysis_policy(
        7,
        frontier_sha256=frontier_sha,
        profile=profile,
        completed_specializations=("prefix_compression",),
    )
    assert two.max_parallel == 2
    assert two.specializations == (
        "exact_planning",
        "state_representation",
    )


def test_one_retry_complexity_coordinate_drives_both_escalation_axes():
    frontier_sha = "4" * 64
    rows = [
        S.frontier_complexity_schedule(
            index, frontier_sha256=frontier_sha
        )
        for index in range(7)
    ]
    assert all(
        row.coordinate == "exact_frontier_clean_no_progress_retries"
        and row.no_progress == index
        for index, row in enumerate(rows)
    )
    assert [row.primary.effort for row in rows] == [
        "medium",
        "high",
        "xhigh",
        "xhigh",
        "max",
        "max",
        "max",
    ]
    assert [row.auxiliary.phase for row in rows] == [
        "disabled",
        "disabled",
        "disabled",
        "disabled",
        "disabled",
        "diagnose",
        "diagnose",
    ]
    assert (
        S.policy_projection()["operational_complexity_coordinate"]
        == "exact_frontier_clean_no_progress_retries"
    )


def test_supervision_loop_is_a_game_agnostic_receipt_reducer():
    expected = {
        "clean_no_progress": (
            "READY",
            1,
            "clear_before_optional_replacement",
        ),
        "tainted": (
            "READY",
            0,
            "revoke_same_thread_frontier_context",
        ),
        "protocol_invalid": (
            "READY",
            0,
            "revoke_same_thread_frontier_context",
        ),
        "infrastructure": (
            "READY",
            0,
            "retain_authenticated_if_no_exposure",
        ),
        "candidate": (
            "PROMOTING",
            0,
            "retain_authenticated_pending_promotion",
        ),
        "blocker": (
            "BLOCKED",
            0,
            "clear_terminal_frontier",
        ),
    }
    for kind, (phase, delta, prior_disposition) in expected.items():
        first = S.terminal_policy_transition(kind)
        second = S.terminal_policy_transition(kind)
        assert first == second
        assert first.schema == 2
        assert first.next_lane_phase == phase
        assert first.retry_coordinate_delta == delta
        assert first.prior_wip_disposition == prior_disposition
        assert S.advance_retry_coordinate(11, kind) == 11 + delta
    for kind in S.NONCOUNTING_RUNTIME_OUTCOMES:
        assert S.advance_retry_coordinate(11, kind) == 11
    with pytest.raises(S.SchedulerError, match="outside policy"):
        S.terminal_policy_transition("operator_override")
    with pytest.raises(S.SchedulerError, match="outside policy"):
        S.advance_retry_coordinate(11, "looks_promising")

    promotion_failure = S.promotion_failure_policy_transition()
    assert promotion_failure.event_kind == "PROMOTION_FAILED"
    assert promotion_failure.next_lane_phase == "READY"
    assert promotion_failure.retry_coordinate_delta == 0
    assert promotion_failure.blocker_authority is False
    assert promotion_failure.candidate_disposition == (
        "discard_rejected_candidate"
    )


@pytest.mark.parametrize(
    (
        "result_kind",
        "current_wip",
        "exposure_detected",
        "expected",
    ),
    (
        ("clean_no_progress", None, False, None),
        ("clean_no_progress", "replacement", False, "replacement"),
        # A rejected/missing current snapshot is represented by None after
        # the independent WIP admission gate and clears the coherence epoch.
        ("clean_no_progress", None, False, None),
        ("tainted", None, False, None),
        ("protocol_invalid", None, False, None),
        ("infrastructure", None, False, "prior"),
        ("infrastructure", None, True, None),
        ("candidate", None, False, "prior"),
        ("blocker", None, False, None),
    ),
)
def test_two_dimensional_terminal_wip_reduction_matrix(
    result_kind, current_wip, exposure_detected, expected
):
    transition = S.terminal_policy_transition(result_kind)
    assert S.reduce_terminal_wip(
        transition=transition,
        prior_wip="prior",
        current_attempt_wip=current_wip,
        exposure_detected=exposure_detected,
    ) == expected


def test_nonclean_current_attempt_wip_is_never_retained():
    with pytest.raises(
        S.SchedulerError, match="must discard"
    ):
        S.reduce_terminal_wip(
            transition=S.terminal_policy_transition(
                "infrastructure"
            ),
            prior_wip="prior",
            current_attempt_wip="untrusted-current",
            exposure_detected=False,
        )
    assert S.PROMOTION_FAILURE_CODES == (
        "promotion_gate_rejected",
        "promotion_commit_invalid",
    )
    promotion_failure = S.promotion_failure_policy_transition()

    projection = S.policy_projection()["supervision_loop"]
    assert projection["authority"] == "receipt_reducer_only"
    assert projection["cycle_stages"] == list(
        S.SUPERVISION_CYCLE_STAGES
    )
    assert "operator_hint" in projection["forbidden_inputs"]
    assert "game_semantics" in projection["forbidden_inputs"]
    assert (
        "interactive_operator_or_user_channel"
        in projection["forbidden_inputs"]
    )
    assert (
        "interactive_operator_or_user_channel"
        in S.SUPERVISORY_FORBIDDEN_INPUT_CLASSES
    )
    assert projection["silent_live_operator_steering"] is False
    assert projection["promotion_failure"] == {
        "codes": list(S.PROMOTION_FAILURE_CODES),
        "transition": dataclasses.asdict(promotion_failure),
    }
    assert (
        projection["blocker_authority"]["codes"]
        == list(S.HOST_BLOCKER_CODES)
    )
    serialized = S.canonical_json(projection)
    assert b"bp35" not in serialized
    assert b"lf52" not in serialized
    assert S.SCHEDULER_POLICY_SHA256 == S.sha256_json(
        S.policy_projection()
    )
    weakened = S.policy_projection()
    weakened["supervision_loop"]["forbidden_inputs"].remove(
        "interactive_operator_or_user_channel"
    )
    assert (
        S.sha256_json(weakened)
        != S.SCHEDULER_POLICY_SHA256
    )


def test_auxiliary_specialist_policy_is_game_agnostic_and_fail_closed():
    projection = S.policy_projection()["auxiliary_analysis"]
    assert projection["game_specific_rules"] is False
    assert projection["preserve_public_response_splits"] is True
    assert projection["hidden_state_completeness_claim"] is False
    serialized = S.canonical_json(projection)
    assert b"bp35" not in serialized
    assert b"lf52" not in serialized

    frontier_sha = "d" * 64
    with pytest.raises(S.SchedulerError, match="before"):
        S.auxiliary_analysis_policy(
            4,
            frontier_sha256=frontier_sha,
            active_specializations=("complexity_diagnosis",),
        )
    with pytest.raises(S.SchedulerError, match="stale"):
        S.auxiliary_analysis_policy(
            7,
            frontier_sha256=frontier_sha,
            profile=S.ComplexityProfile(
                schema=1,
                profile_id="profile:1",
                round_index=0,
                frontier_sha256="e" * 64,
                observation_receipt_sha256="f" * 64,
                taint_scan_receipt_sha256="1" * 64,
                priorities=("exact_planning",),
            ),
        )
    with pytest.raises(S.SchedulerError, match="ambiguous"):
        S.auxiliary_analysis_policy(
            7,
            frontier_sha256=frontier_sha,
            profile=S.ComplexityProfile(
                schema=1,
                profile_id="profile:1",
                round_index=0,
                frontier_sha256=frontier_sha,
                observation_receipt_sha256="f" * 64,
                taint_scan_receipt_sha256="1" * 64,
                priorities=("exact_planning", "exact_planning"),
            ),
        )
    profile = S.ComplexityProfile(
        schema=1,
        profile_id="profile:2",
        round_index=1,
        frontier_sha256=frontier_sha,
        observation_receipt_sha256="2" * 64,
        taint_scan_receipt_sha256="3" * 64,
        priorities=(
            "mechanism_induction",
            "exact_planning",
            "prefix_compression",
        ),
    )
    with pytest.raises(S.SchedulerError, match="exceed"):
        S.auxiliary_analysis_policy(
            7,
            frontier_sha256=frontier_sha,
            profile=profile,
            active_specializations=(
                "mechanism_induction",
                "exact_planning",
                "prefix_compression",
            ),
        )


def test_auxiliary_decision_uses_only_exact_clean_retry_history(tmp_path):
    snapshot, frontier = _auxiliary_snapshot(
        tmp_path, no_progress=5
    )
    configuration = _enabled_auxiliary_configuration()
    decision = S.build_auxiliary_decision(
        snapshot,
        decision_id="aux-decision:1",
        assignment_id="aux-assignment:1",
        reservation_id="aux-reservation:1",
        expert_id="aux-expert:1",
        thread_id="12345678-1234-4234-8234-123456789abc",
        observation_ledger_sha256="d" * 64,
        launch_configuration=configuration,
    )
    assert decision is not None
    assert decision.game == frontier.game
    assert decision.no_progress == 5
    assert decision.specialization == "complexity_diagnosis"
    assert decision.reasoning_effort == "max"
    assert decision.input_manifest.game == frontier.game
    assert decision.input_manifest.live_lineage_mounted is False
    assert decision.input_manifest_sha256 == S.sha256_json(
        dataclasses.asdict(decision.input_manifest)
    )
    S.verify_auxiliary_decision(
        snapshot, decision, launch_configuration=configuration
    )

    # Legacy status/recommendation data has no input channel into either axis.
    forged_external_status = {
        "no_progress": 99,
        "recommended_effort": "medium",
        "manually_trigger_auxiliary": True,
    }
    assert forged_external_status["recommended_effort"] == "medium"
    assert S.frontier_complexity_schedule(
        snapshot.frontiers[0].no_progress,
        frontier_sha256=frontier.frontier_sha256,
    ).primary.effort == "max"
    assert decision.reasoning_effort == configuration.reasoning_effort

    missing_clean_row = dataclasses.replace(
        snapshot,
        clean_proposer_settlements=(
            snapshot.clean_proposer_settlements[:-1]
        ),
    )
    with pytest.raises(S.SchedulerError, match="exact clean"):
        S.build_auxiliary_decision(
            missing_clean_row,
            decision_id="aux-decision:2",
            assignment_id="aux-assignment:2",
            reservation_id="aux-reservation:2",
            expert_id="aux-expert:2",
            thread_id="22345678-1234-4234-8234-123456789abc",
            observation_ledger_sha256="d" * 64,
            launch_configuration=configuration,
        )

    # Any still-eligible primary owns the free reservation first.
    other = snapshot.frontiers[1]
    primary_available = dataclasses.replace(
        snapshot,
        frontiers=(
            snapshot.frontiers[0],
            dataclasses.replace(other, blocked_reason=None),
            *snapshot.frontiers[2:],
        ),
    )
    assert S.build_auxiliary_decision(
        primary_available,
        decision_id="aux-decision:3",
        assignment_id="aux-assignment:3",
        reservation_id="aux-reservation:3",
        expert_id="aux-expert:3",
        thread_id="32345678-1234-4234-8234-123456789abc",
        observation_ledger_sha256="d" * 64,
        launch_configuration=configuration,
    ) is None


def test_next_round_diagnosis_cannot_be_dispatched_twice(tmp_path):
    current, frontier = _auxiliary_snapshot(
        tmp_path, no_progress=7
    )
    current = dataclasses.replace(current, max_lanes=3)
    configuration = _enabled_auxiliary_configuration()
    frontier_at_diagnosis = dataclasses.replace(
        frontier, no_progress=5
    )
    diagnosis_snapshot = dataclasses.replace(
        current,
        frontiers=(
            frontier_at_diagnosis,
            *current.frontiers[1:],
        ),
            clean_proposer_settlements=(
                current.clean_proposer_settlements[:5]
            ),
            sidecar_requests=current.sidecar_requests[:5],
        )
    diagnosis = S.build_auxiliary_decision(
        diagnosis_snapshot,
        decision_id="aux-decision:diagnosis0",
        assignment_id="aux-assignment:diagnosis0",
        reservation_id="aux-reservation:diagnosis0",
        expert_id="aux-expert:diagnosis0",
        thread_id="12345678-1234-4234-8234-123456789abc",
        observation_ledger_sha256="d" * 64,
        launch_configuration=configuration,
    )
    assert diagnosis is not None

    def admitted(decision):
        challenge = S.SocraticChallengeEvidence(
            schema=1,
            hypothesis="The selected obligation explains the observations.",
            counter_hypothesis="A different obligation is causal.",
            falsification_attempt="Compared the distinguishing public trace.",
            observation_receipt_sha256s=("d" * 64,),
            rejected_conclusions=("One account was rejected.",),
            surviving_conclusions=("One account remains.",),
        )
        output = S.AuxiliaryOutputEvidence(
            schema=1,
            assignment_id=decision.assignment_id,
            expert_id=decision.expert_id,
            thread_id=decision.thread_id,
            specialization=decision.specialization,
            frontier_sha256=decision.frontier_sha256,
            parent_checkpoint_sha256=(
                decision.parent_checkpoint_sha256
            ),
            input_manifest_sha256=decision.input_manifest_sha256,
            output_manifest_sha256="e" * 64,
            public_observation_receipt_sha256s=("d" * 64,),
            challenge=challenge,
            quarantined_artifact_sha256s=("f" * 64,),
            result_authority="quarantine_only",
            mutates_live_lineage=False,
        )
        return S.AuxiliaryAssignmentState(
            schema=1,
            assignment_id=decision.assignment_id,
            decision_id=decision.decision_id,
            reservation_id=decision.reservation_id,
            game=decision.game,
            frontier_sha256=decision.frontier_sha256,
            parent_checkpoint_sha256=(
                decision.parent_checkpoint_sha256
            ),
            trigger_no_progress=decision.no_progress,
            trigger_history_sha256=(
                decision.trigger_history_sha256
            ),
            profile_id=decision.profile_id,
            round_index=decision.round_index,
            specialization=decision.specialization,
            expert_id=decision.expert_id,
            thread_id=decision.thread_id,
            active_proposer_attempt_id=(
                decision.active_proposer_attempt_id
            ),
            input_manifest=decision.input_manifest,
            input_manifest_sha256=decision.input_manifest_sha256,
            observation_ledger_sha256=(
                decision.observation_ledger_sha256
            ),
            model=decision.model,
            reasoning_effort=decision.reasoning_effort,
            role=decision.role,
            context_limit_tokens=decision.context_limit_tokens,
            role_max_concurrency=decision.role_max_concurrency,
            supervisory_launch_configuration_sha256=(
                decision
                .supervisory_launch_configuration_sha256
            ),
            sidecar_request=decision.sidecar_request,
            sidecar_request_sha256=(
                decision.sidecar_request_sha256
            ),
            phase="ADMITTED",
            output=output,
            admission_receipt_path=str(
                tmp_path
                / f"{decision.assignment_id}-admission.json"
            ),
            admission_receipt_sha256="1" * 64,
            admitted_sequence=100 + decision.round_index,
            admitted_event_digest="2" * 64,
        )

    diagnosis_assignment = admitted(diagnosis)
    profile = S.ComplexityProfile(
        schema=1,
        profile_id="profile:round0",
        round_index=0,
        frontier_sha256=frontier.frontier_sha256,
        observation_receipt_sha256="d" * 64,
        taint_scan_receipt_sha256="a" * 64,
        priorities=("exact_planning",),
    )
    round_zero = S.ComplexityRoundState(
        schema=1,
        game=frontier.game,
        frontier_sha256=frontier.frontier_sha256,
        parent_checkpoint_sha256=frontier.parent_checkpoint_sha256,
        parent_source_tree_sha256=frontier.parent_source_tree_sha256,
        round_index=0,
        profile=profile,
        diagnosis_assignment_id=diagnosis.assignment_id,
        trigger_no_progress=diagnosis.no_progress,
        trigger_history_sha256=diagnosis.trigger_history_sha256,
        input_manifest_sha256=diagnosis.input_manifest_sha256,
        observation_ledger_sha256=(
            diagnosis.observation_ledger_sha256
        ),
        admission_receipt_path=str(
            (tmp_path / "round0-admission.json").resolve()
        ),
        admission_receipt_sha256="b" * 64,
        admitted_sequence=10,
        admitted_event_digest="c" * 64,
    )
    with_profile = dataclasses.replace(
        current,
        journal_head_sequence=10,
        journal_head_digest="1" * 64,
        complexity_rounds=(round_zero,),
        auxiliary_assignments=(diagnosis_assignment,),
    )
    specialist = S.build_auxiliary_decision(
        with_profile,
        decision_id="aux-decision:specialist0",
        assignment_id="aux-assignment:specialist0",
        reservation_id="aux-reservation:specialist0",
        expert_id="aux-expert:specialist0",
        thread_id="22345678-1234-4234-8234-123456789abc",
        observation_ledger_sha256="d" * 64,
        launch_configuration=configuration,
    )
    assert specialist is not None
    assert specialist.specialization == "exact_planning"
    completed = dataclasses.replace(
        with_profile,
        journal_head_sequence=11,
        journal_head_digest="2" * 64,
        auxiliary_assignments=(
            diagnosis_assignment,
            admitted(specialist),
        ),
    )
    next_diagnosis = S.build_auxiliary_decision(
        completed,
        decision_id="aux-decision:diagnosis1",
        assignment_id="aux-assignment:diagnosis1",
        reservation_id="aux-reservation:diagnosis1",
        expert_id="aux-expert:diagnosis1",
        thread_id="32345678-1234-4234-8234-123456789abc",
        observation_ledger_sha256="d" * 64,
        launch_configuration=configuration,
    )
    assert next_diagnosis is not None
    assert (
        next_diagnosis.specialization,
        next_diagnosis.round_index,
    ) == ("complexity_diagnosis", 1)

    def rehash_auxiliary(value):
        body = dataclasses.asdict(value)
        body.pop("decision_sha256")
        return dataclasses.replace(
            value, decision_sha256=S.sha256_json(body)
        )

    forged_auxiliary_decisions = (
        dataclasses.replace(
            next_diagnosis, game=completed.frontiers[1].game
        ),
        dataclasses.replace(
            next_diagnosis,
            no_progress=next_diagnosis.no_progress + 1,
        ),
        dataclasses.replace(
            next_diagnosis,
            round_index=next_diagnosis.round_index + 1,
        ),
        dataclasses.replace(
            next_diagnosis, specialization="exact_planning"
        ),
        dataclasses.replace(
            next_diagnosis, reasoning_effort="medium"
        ),
    )
    for forged in forged_auxiliary_decisions:
        with pytest.raises(
            S.SchedulerError, match="manually triggered.*forged"
        ):
            S.verify_auxiliary_decision(
                completed,
                rehash_auxiliary(forged),
                launch_configuration=configuration,
            )

        active_next_diagnosis = dataclasses.replace(
            admitted(next_diagnosis),
            phase="RESERVED",
            output=None,
            admission_receipt_path=None,
            admission_receipt_sha256=None,
            admitted_sequence=None,
            admitted_event_digest=None,
        )
    after_first_reservation = dataclasses.replace(
        completed,
        journal_head_sequence=12,
        journal_head_digest="3" * 64,
        auxiliary_assignments=(
            *completed.auxiliary_assignments,
            active_next_diagnosis,
        ),
    )
    assert S.build_auxiliary_decision(
        after_first_reservation,
        decision_id="aux-decision:diagnosis1-duplicate",
        assignment_id="aux-assignment:diagnosis1-duplicate",
        reservation_id="aux-reservation:diagnosis1-duplicate",
        expert_id="aux-expert:diagnosis1-duplicate",
        thread_id="42345678-1234-4234-8234-123456789abc",
        observation_ledger_sha256="d" * 64,
        launch_configuration=configuration,
    ) is None

    duplicate_assignment = dataclasses.replace(
        active_next_diagnosis,
        assignment_id="aux-assignment:diagnosis1-forged",
        decision_id="aux-decision:diagnosis1-forged",
        reservation_id="aux-reservation:diagnosis1-forged",
        expert_id="aux-expert:diagnosis1-forged",
        thread_id="52345678-1234-4234-8234-123456789abc",
    )
    with pytest.raises(S.SchedulerError, match="repeats"):
        S.validate_snapshot(dataclasses.replace(
            after_first_reservation,
            auxiliary_assignments=(
                *after_first_reservation.auxiliary_assignments,
                duplicate_assignment,
            ),
        ))


def test_auxiliary_output_has_no_candidate_or_live_wip_authority(tmp_path):
    snapshot, _ = _auxiliary_snapshot(tmp_path, no_progress=5)
    decision = S.build_auxiliary_decision(
        snapshot,
        decision_id="aux-decision:output",
        assignment_id="aux-assignment:output",
        reservation_id="aux-reservation:output",
        expert_id="aux-expert:output",
        thread_id="42345678-1234-4234-8234-123456789abc",
        observation_ledger_sha256="d" * 64,
        launch_configuration=_enabled_auxiliary_configuration(),
    )
    assert decision is not None
    challenge = {
        "schema": 1,
        "hypothesis": "A compact mechanism explains the observations.",
        "counter_hypothesis": "The apparent mechanism is incidental.",
        "falsification_attempt": "Replayed the distinguishing prefix.",
        "observation_receipt_sha256s": ["d" * 64],
        "rejected_conclusions": ["The incidental account failed."],
        "surviving_conclusions": ["The mechanism remains viable."],
    }
    output = {
        "schema": 1,
        "assignment_id": decision.assignment_id,
        "expert_id": decision.expert_id,
        "thread_id": decision.thread_id,
        "specialization": decision.specialization,
        "frontier_sha256": decision.frontier_sha256,
        "parent_checkpoint_sha256":
            decision.parent_checkpoint_sha256,
        "input_manifest_sha256": decision.input_manifest_sha256,
        "output_manifest_sha256": "e" * 64,
        "public_observation_receipt_sha256s": ["d" * 64],
        "challenge": challenge,
        "quarantined_artifact_sha256s": ["f" * 64],
        "result_authority": "quarantine_only",
        "mutates_live_lineage": False,
        "supervisory_handoff": None,
    }
    parsed = S.auxiliary_output_from_dict(output)
    assert parsed.result_authority == "quarantine_only"
    solver_source_sha256 = "b" * 64
    sealed_manifest = dataclasses.replace(
        decision.input_manifest,
        wip_snapshot_id="wip:typed-evidence",
        wip_tree_sha256="a" * 64,
        wip_solver_source_tree_sha256=solver_source_sha256,
        native_solver_source_tree_sha256s=(
            solver_source_sha256,
        ),
        authenticated_evidence_set_sha256=S.sha256_json(
            {
                "public_observation_receipts": ["d" * 64],
                "native_solver_source_trees": [
                    solver_source_sha256
                ],
                "side_expert": [],
            }
        ),
    )
    S.validate_auxiliary_input_manifest(sealed_manifest)
    with pytest.raises(S.SchedulerError, match="manifest"):
        S.validate_auxiliary_input_manifest(dataclasses.replace(
            sealed_manifest,
            forbidden_input_classes=tuple(
                item
                for item in sealed_manifest.forbidden_input_classes
                if item
                != "interactive_operator_or_user_channel"
            ),
        ))
    sealed_manifest_sha256 = S.sha256_json(
        dataclasses.asdict(sealed_manifest)
    )
    assignment = S.AuxiliaryAssignmentState(
        schema=1,
        assignment_id=decision.assignment_id,
        decision_id=decision.decision_id,
        reservation_id=decision.reservation_id,
        game=decision.game,
        frontier_sha256=decision.frontier_sha256,
        parent_checkpoint_sha256=decision.parent_checkpoint_sha256,
        trigger_no_progress=decision.no_progress,
        trigger_history_sha256=decision.trigger_history_sha256,
        profile_id=decision.profile_id,
        round_index=decision.round_index,
        specialization=decision.specialization,
        expert_id=decision.expert_id,
        thread_id=decision.thread_id,
        active_proposer_attempt_id=decision.active_proposer_attempt_id,
        input_manifest=sealed_manifest,
        input_manifest_sha256=sealed_manifest_sha256,
        observation_ledger_sha256=decision.observation_ledger_sha256,
        model=decision.model,
        reasoning_effort=decision.reasoning_effort,
        role=decision.role,
        context_limit_tokens=decision.context_limit_tokens,
        role_max_concurrency=decision.role_max_concurrency,
        supervisory_launch_configuration_sha256=(
            decision.supervisory_launch_configuration_sha256
        ),
        sidecar_request=decision.sidecar_request,
        sidecar_request_sha256=decision.sidecar_request_sha256,
        phase="RUNNING",
    )
    forged_solver_citation = dataclasses.replace(
        parsed,
        input_manifest_sha256=sealed_manifest_sha256,
        public_observation_receipt_sha256s=(
            solver_source_sha256,
        ),
        challenge=dataclasses.replace(
            parsed.challenge,
            observation_receipt_sha256s=(
                solver_source_sha256,
            ),
        ),
    )
    with pytest.raises(
        S.SchedulerError, match="assignment boundary"
    ):
        S.validate_auxiliary_output(
            forged_solver_citation, assignment=assignment
        )
    with pytest.raises(S.SchedulerError, match="schema"):
        S.auxiliary_output_from_dict({
            **output,
            "candidate": {"to_level": 1},
        })
    with pytest.raises(S.SchedulerError, match="schema"):
        S.auxiliary_output_from_dict({
            key: value
            for key, value in output.items()
            if key != "challenge"
        })


def test_runner_policy_verifier_rejects_deescalation():
    assert (
        S.verify_runner_policy(
            S.retry_policy,
            declared_policy_sha256=S.SCHEDULER_POLICY_SHA256,
        )
        == S.SCHEDULER_POLICY_SHA256
    )

    def broken(index: int):
        row = S.retry_policy(index)
        return (
            dataclasses.replace(row, effort="high")
            if index == 3
            else row
        )

    with pytest.raises(S.SchedulerError, match="differs"):
        S.verify_runner_policy(
            broken,
            declared_policy_sha256=S.SCHEDULER_POLICY_SHA256,
        )


def test_unlimited_mode_has_no_cost_cutoff_but_accounts_usage():
    state = S.BudgetState("window:test", None, 0)
    assert S.reservation_allowance(state, slots_to_fill=6) is None
    assert (
        S.reserve_budget(
            state,
            reservation_id="reservation:1",
            attempt_id="attempt:1",
            units=None,
        )
        == state
    )
    settled = S.settle_budget(
        state,
        reservation_id="reservation:1",
        attempt_id="attempt:1",
        charged_units=123,
    )
    assert settled.limit_units is None
    assert settled.settled_units == 123
    assert settled.live_reservations == ()


def test_finite_budget_reservations_cannot_overbook_or_double_settle():
    state = S.BudgetState("window:test", 10, 0)
    for index, slots in enumerate((3, 2, 1), 1):
        allowance = S.reservation_allowance(state, slots_to_fill=slots)
        state = S.reserve_budget(
            state,
            reservation_id=f"reservation:{index}",
            attempt_id=f"attempt:{index}",
            units=allowance,
        )
    assert sum(item.units for item in state.live_reservations) == 10
    first = next(
        item for item in state.live_reservations
        if item.reservation_id == "reservation:1"
    )
    with pytest.raises(S.SchedulerError, match="exceeds"):
        S.settle_budget(
            state,
            reservation_id=first.reservation_id,
            attempt_id=first.attempt_id,
            charged_units=first.units + 1,
        )
    settled = S.settle_budget(
        state,
        reservation_id=first.reservation_id,
        attempt_id=first.attempt_id,
        charged_units=first.units,
    )
    with pytest.raises(S.SchedulerError, match="missing or duplicated"):
        S.settle_budget(
            settled,
            reservation_id=first.reservation_id,
            attempt_id=first.attempt_id,
            charged_units=0,
        )


def test_selection_evidence_measures_conditional_novelty_and_reuse(tmp_path):
    parent = tmp_path / "parent"
    parent_sha = _write_source(parent)
    same = tmp_path / "same"
    same_sha = _write_source(same)
    evidence = S.selection_evidence(
        parent_source_path=str(parent),
        parent_source_tree_sha256=parent_sha,
        candidate_source_path=str(same),
        candidate_source_tree_sha256=same_sha,
    )
    assert evidence.conditional_novelty == 0
    assert evidence.metric == "positive_unmatched_normalized_ast_zlib_v1"
    assert evidence.retained_normalized_units > 0
    assert "retained" in evidence.reused_definition_calls
    S.verify_selection_evidence(evidence)

    legs = same / "legs.py"
    before = legs.stat(follow_symlinks=False)
    raw = legs.read_bytes()
    legs.write_bytes(raw.replace(b"return 1", b"return 2"))
    os.utime(
        legs,
        ns=(before.st_atime_ns, before.st_mtime_ns),
        follow_symlinks=False,
    )
    after = legs.stat(follow_symlinks=False)
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_ctime_ns != before.st_ctime_ns
    with pytest.raises(S.SchedulerError, match="hash changed"):
        S.verify_selection_evidence(evidence)


def test_dead_unreachable_call_does_not_witness_reuse(tmp_path):
    parent = tmp_path / "parent"
    parent_sha = _write_source(parent)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "legs.py").write_text(
        "def retained(env):\n    return 1\n"
        "def dead(env):\n    return retained(env)\n"
    )
    (candidate / "players.py").write_text(
        "def play_level_1(env):\n    return 0\n"
    )
    (candidate / "solve.py").write_text(
        "from players import play_level_1\n"
        "def solve(env):\n    return play_level_1(env)\n"
    )
    candidate_sha = hashlib.sha256()
    for path in sorted(candidate.iterdir()):
        candidate_sha.update(path.name.encode())
        candidate_sha.update(b"\0")
        candidate_sha.update(
            hashlib.sha256(path.read_bytes()).hexdigest().encode()
        )
        candidate_sha.update(b"\n")
    evidence = S.selection_evidence(
        parent_source_path=str(parent),
        parent_source_tree_sha256=parent_sha,
        candidate_source_path=str(candidate),
        candidate_source_tree_sha256=candidate_sha.hexdigest(),
    )
    assert "retained" not in evidence.reused_definition_calls


def test_unreferenced_level_and_literal_dead_branch_do_not_witness_reuse(
    tmp_path,
):
    parent = tmp_path / "parent"
    parent_sha = _write_source(parent)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "legs.py").write_text(
        "def retained(env):\n    return 1\n"
    )
    (candidate / "players.py").write_text(
        "from legs import retained\n"
        "def play_level_1(env):\n"
        "    if False:\n"
        "        return retained(env)\n"
        "    return 0\n"
        "def play_level_999(env):\n"
        "    return retained(env)\n"
    )
    (candidate / "solve.py").write_text(
        "from players import play_level_1\n"
        "def solve(env):\n    return play_level_1(env)\n"
    )
    candidate_sha = hashlib.sha256()
    for path in sorted(candidate.iterdir()):
        candidate_sha.update(path.name.encode())
        candidate_sha.update(b"\0")
        candidate_sha.update(
            hashlib.sha256(path.read_bytes()).hexdigest().encode()
        )
        candidate_sha.update(b"\n")
    evidence = S.selection_evidence(
        parent_source_path=str(parent),
        parent_source_tree_sha256=parent_sha,
        candidate_source_path=str(candidate),
        candidate_source_tree_sha256=candidate_sha.hexdigest(),
    )
    assert "retained" not in evidence.reused_definition_calls


def test_excluded_wip_cannot_bias_dispatch_evidence(tmp_path):
    snapshot = _snapshot(tmp_path, special_wip_game="bp35")
    bp35 = next(
        frontier for frontier in snapshot.frontiers
        if frontier.game == "bp35"
    )
    snapshot = dataclasses.replace(
        snapshot,
        frontiers=tuple(
            dataclasses.replace(frontier, no_progress=5)
            if frontier.game == "bp35"
            else dataclasses.replace(frontier, last_dispatch_sequence=1)
            for frontier in snapshot.frontiers
        ),
    )
    choice = S.choose_frontier(snapshot)
    assert choice is not None
    assert choice.game == "bp35"
    assert choice.requested_wip_mode == "exclude"
    assert choice.selected_wip is None
    assert choice.conditional_novelty == S.UNKNOWN_CONDITIONAL_NOVELTY
    assert choice.reused_definition_calls == ()
    assert choice.estimated_free_energy_micro == (
        -choice.success_prior_micro
        + S.FREE_ENERGY_COMPLEXITY_WEIGHT
        * S.UNKNOWN_CONDITIONAL_NOVELTY
    )


def test_selection_is_fair_then_free_energy_and_capacity_is_ceiling(tmp_path):
    snapshot = _snapshot(tmp_path, special_wip_game="bp35")
    # All last-dispatch sequences are equal.  The identical clean WIP has zero
    # conditional novelty and beats the fixed ignorance prior.
    choice = S.choose_frontier(snapshot)
    assert choice is not None
    assert choice.game == "bp35"
    assert choice.thread_mode == "resume"

    full = _snapshot(
        tmp_path / "full",
        active_games=set(_inventory()) - {"wa30"},
    )
    # The snapshot itself fails closed because active lanes exceed max_lanes;
    # capacity is never treated as a request to overlap more work.
    with pytest.raises(S.SchedulerError, match="capacity"):
        S.choose_frontier(full)


def test_decision_binds_snapshot_and_reservation(tmp_path):
    snapshot = _snapshot(tmp_path)
    decision = S.build_decision(
        snapshot,
        decision_id="decision:1",
        attempt_id="attempt:1",
        generation_id="generation:1",
        reservation_id="reservation:1",
    )
    assert decision is not None
    S.verify_decision(snapshot, decision)
    durable = json.loads(
        S.canonical_json(S.decision_to_dict(decision))
    )
    assert S.decision_from_dict(durable) == decision
    binding = S.reservation_binding(decision)
    assert binding["scheduler_decision_sha256"] == decision.decision_sha256

    def rehash_choice(choice):
        body = S._decision_body(
            snapshot=snapshot,
            choice=choice,
            decision_id=decision.decision_id,
            attempt_id=decision.attempt_id,
            generation_id=decision.generation_id,
            reservation_id=decision.reservation_id,
        )
        return dataclasses.replace(
            decision,
            choice=choice,
            decision_sha256=S.sha256_json(body),
        )

    other_game = next(
        frontier.game
        for frontier in snapshot.frontiers
        if frontier.game != decision.choice.game
    )
    forged_choices = (
        dataclasses.replace(decision.choice, game=other_game),
        dataclasses.replace(
            decision.choice,
            target_level=decision.choice.target_level + 1,
        ),
        dataclasses.replace(
            decision.choice,
            no_progress=decision.choice.no_progress + 1,
        ),
        dataclasses.replace(decision.choice, effort="high"),
        dataclasses.replace(
            decision.choice,
            soft_allocation_seconds=20 * 60,
        ),
        dataclasses.replace(
            decision.choice,
            requested_wip_mode="restore_clean_same_frontier",
            effective_wip_mode="restore_clean_same_frontier",
            thread_mode="resume",
        ),
    )
    for forged_choice in forged_choices:
        with pytest.raises(S.SchedulerError, match="stale.*forged"):
            S.verify_decision(
                snapshot, rehash_choice(forged_choice)
            )

    with pytest.raises(S.SchedulerError, match="stale"):
        S.verify_decision(
            dataclasses.replace(
                snapshot, journal_head_digest="6" * 64
            ),
            decision,
        )


def test_genesis_only_phase_fixture_cannot_claim_public_pass(tmp_path):
    campaign = tmp_path / "campaign"
    source = tmp_path / "source"
    source_sha = _write_source(source)
    _genesis(campaign, source, source_sha)
    receipt = S.audit_campaign(campaign)
    assert receipt["verdict"] == "FAIL"
    assert receipt["runner_lifecycle"] == {}
    assert "full runner lifecycle audit failed" in receipt["findings"][0]
    phase_summary = S.validate_journal_event_sequence(
        S.read_journal(campaign)
    )
    assert phase_summary["decisions"] == 0
    prefix = S.journal_prefix_status(campaign)
    assert prefix["headroom_bytes"] > 0
    event_file_bytes = sum(
        path.stat().st_size
        for path in (campaign / "attempt_journal").glob("*.json")
    )
    assert (
        prefix["used_bytes"]
        == event_file_bytes + 1
    )
    assert S.require_dispatch_headroom(campaign)["headroom_bytes"] >= (
        S.MIN_DISPATCH_HEADROOM_BYTES
    )
    assert S.MAX_JOURNALED_OBSERVATIONS_PER_ATTEMPT == 72
    assert S.MIN_JOURNALED_OBSERVATION_INTERVAL_SECONDS == 300
    retained = tmp_path / "receipt.json"
    retained.write_bytes(S.canonical_json(receipt) + b"\n")
    with pytest.raises(S.SchedulerError):
        S.verify_audit_receipt(campaign, retained)


def test_phase_reducer_accepts_exact_decision_reservation_pair(tmp_path):
    campaign = tmp_path / "campaign"
    source = tmp_path / "source"
    source_sha = _write_source(source)
    events = _genesis(campaign, source, source_sha)
    inventory = _inventory()
    zero_checkpoints = events[0]["payload"]["zero_checkpoints"]
    evidence = S.selection_evidence(
        parent_source_path=str(source),
        parent_source_tree_sha256=source_sha,
    )
    frontiers = tuple(
        S.Frontier(
            game=game,
            target=target,
            reached=0,
            no_progress=0,
            last_dispatch_sequence=0,
            parent_checkpoint_sha256=zero_checkpoints[game]["sha256"],
            parent_source_path=str(source),
            parent_source_tree_sha256=source_sha,
            frontier_sha256=S._frontier_digest(
                game, 0, zero_checkpoints[game]["sha256"]
            ),
            active_attempt_id=None,
            draining=False,
            blocked_reason=None,
            wip=None,
            evidence=evidence,
            public_observation_receipt_sha256s=(),
            observation_ledger_sha256=(
                S.public_observation_ledger_sha256(
                    game=game,
                    frontier_sha256=S._frontier_digest(
                        game,
                        0,
                        zero_checkpoints[game]["sha256"],
                    ),
                    parent_checkpoint_sha256=(
                        zero_checkpoints[game]["sha256"]
                    ),
                    receipt_sha256s=(),
                )
            ),
        )
        for game, target in inventory.items()
    )
    snapshot = S.CampaignSnapshot(
        campaign_id="campaign:test",
        journal_head_sequence=1,
        journal_head_digest=events[0]["digest"],
        inventory=tuple(inventory.items()),
        max_lanes=6,
        frontiers=frontiers,
        budget=S.BudgetState("window:test", None, 0),
    )
    decision = S.build_decision(
        snapshot,
        decision_id="decision:1",
        attempt_id="attempt:1",
        generation_id="generation:1",
        reservation_id="reservation:1",
    )
    assert decision is not None
    checkpoint_sha = zero_checkpoints[decision.choice.game]["sha256"]
    _append_event(
        campaign,
        events,
        event_id="decision:1",
        kind="SCHEDULER_DECISION",
        payload={"decision": S.decision_to_dict(decision)},
    )
    reservation = {
        **S.reservation_binding(decision),
        "campaign_id": "campaign:test",
        "generation_id": "generation:1",
        "attempt_id": "attempt:1",
        "game": decision.choice.game,
        "target_level": decision.choice.target_level,
        "authoritative_target": decision.choice.authoritative_target,
        "parent_checkpoint_path": str(
            campaign
            / "zero_checkpoints"
            / f"{decision.choice.game}.json"
        ),
        "parent_checkpoint_sha256": checkpoint_sha,
        "frontier_sha256": S._frontier_digest(
            decision.choice.game, 0, checkpoint_sha
        ),
        "parent_source_path": str(source),
        "parent_source_tree_sha256": source_sha,
        "effort": decision.choice.effort,
        "soft_allocation_seconds":
            decision.choice.soft_allocation_seconds,
        "wip_mode": decision.choice.effective_wip_mode,
        "thread_mode": decision.choice.thread_mode,
        "resume_thread_id": None,
        "resume_thread_binding_sha256": None,
        "wip": None,
        "cost_limit_remaining": None,
    }
    _append_event(
        campaign,
        events,
        event_id="attempt:1:reserved",
        kind="ATTEMPT_RESERVED",
        payload={
            "attempt_id": "attempt:1",
            "reservation": reservation,
        },
    )
    phase_summary = S.validate_journal_event_sequence(
        S.read_journal(campaign)
    )
    assert phase_summary["decisions"] == 1
    assert phase_summary["reservations"] == 1
    receipt = S.audit_campaign(campaign)
    assert receipt["verdict"] == "FAIL"
    assert "full runner lifecycle audit failed" in receipt["findings"][0]


def test_audit_rejects_reservation_without_decision(tmp_path):
    campaign = tmp_path / "campaign"
    source = tmp_path / "source"
    source_sha = _write_source(source)
    events = _genesis(campaign, source, source_sha)
    _append_event(
        campaign,
        events,
        event_id="attempt:1:reserved",
        kind="ATTEMPT_RESERVED",
        payload={"attempt_id": "attempt:1", "reservation": {}},
    )
    receipt = S.audit_campaign(campaign)
    assert receipt["verdict"] == "FAIL"
    assert "preceding scheduler decision" in receipt["findings"][0]


def test_audit_rejects_terminal_pending_decision(tmp_path):
    campaign, _, _, _, _ = _one_reserved_campaign(
        tmp_path, reserve=False
    )
    receipt = S.audit_campaign(campaign)
    assert receipt["verdict"] == "FAIL"
    assert "unconsumed scheduler decision" in receipt["findings"][0]


def test_audit_rejects_reused_settled_identities(tmp_path):
    campaign, _, _, events, decision = _one_reserved_campaign(tmp_path)
    _append_minimal_completed_lifecycle(campaign, events)
    _append_event(
        campaign,
        events,
        event_id="attempt:1:result",
        kind="ATTEMPT_RESULT",
        payload={
            "attempt_id": "attempt:1",
            "kind": "clean_no_progress",
            "cost_used": 0.0,
            "authenticated_cost_units": 0,
            "budget_reservation_id": "reservation:1",
            "scheduler_decision_id": "decision:1",
            "reason": "",
            "candidate": None,
            "wip": None,
        },
    )
    _append_event(
        campaign,
        events,
        event_id="decision:replay",
        kind="SCHEDULER_DECISION",
        payload={"decision": S.decision_to_dict(decision)},
    )
    receipt = S.audit_campaign(campaign)
    assert receipt["verdict"] == "FAIL"
    assert "reused" in receipt["findings"][0]


def _append_candidate_result(
    campaign: Path,
    events: list[dict],
    decision: S.SchedulerDecision,
    manifest: Path,
) -> str:
    parent_sha = next(
        frontier["parent_checkpoint_sha256"]
        for frontier in decision.eligible_frontiers
        if frontier["game"] == decision.choice.game
    )
    manifest.write_text('{"candidate":"exact"}\n')
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    _append_minimal_completed_lifecycle(
        campaign, events, result_kind="candidate"
    )
    _append_event(
        campaign,
        events,
        event_id="attempt:1:result",
        kind="ATTEMPT_RESULT",
        payload={
            "attempt_id": "attempt:1",
            "kind": "candidate",
            "cost_used": 0.0,
            "authenticated_cost_units": 0,
            "budget_reservation_id": "reservation:1",
            "scheduler_decision_id": "decision:1",
            "reason": "",
            "wip": None,
            "candidate": {
                "game": decision.choice.game,
                "from_level": 0,
                "to_level": 1,
                "parent_checkpoint_sha256": parent_sha,
                "candidate_manifest_path": str(manifest),
                "candidate_manifest_sha256": manifest_sha,
            },
        },
    )
    return manifest_sha


def _write_level_one_checkpoint(path: Path, game: str) -> str:
    path.write_text(
        json.dumps(
            {
                "game": game,
                "reached": 1,
                "total_marginal_C": 1,
                "records": [
                    {"level": 1, "marginal_C": 1, "reached": True}
                ],
                "final_path": [1],
                "validated": True,
            },
            sort_keys=True,
        )
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_audit_reopens_promoted_artifacts_and_rejects_missing_path(tmp_path):
    campaign, source, source_sha, events, decision = (
        _one_reserved_campaign(tmp_path)
    )
    manifest_sha = _append_candidate_result(
        campaign, events, decision, tmp_path / "candidate.json"
    )
    parent_sha = next(
        frontier["parent_checkpoint_sha256"]
        for frontier in decision.eligible_frontiers
        if frontier["game"] == decision.choice.game
    )
    _append_event(
        campaign,
        events,
        event_id="attempt:1:promotion",
        kind="PROMOTION_COMMITTED",
        payload={
            "attempt_id": "attempt:1",
            "from_level": 0,
            "to_level": 1,
            "parent_checkpoint_sha256": parent_sha,
            "candidate_manifest_sha256": manifest_sha,
            "checkpoint_path": str(tmp_path / "does-not-exist.json"),
            "checkpoint_sha256": "6" * 64,
            "source_path": str(source),
            "source_tree_sha256": source_sha,
        },
    )
    receipt = S.audit_campaign(campaign)
    assert receipt["verdict"] == "FAIL"
    assert "promoted checkpoint" in receipt["findings"][0]


def test_one_candidate_cannot_be_promoted_twice(tmp_path):
    campaign, source, source_sha, events, decision = (
        _one_reserved_campaign(tmp_path)
    )
    manifest_sha = _append_candidate_result(
        campaign, events, decision, tmp_path / "candidate.json"
    )
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint_sha = _write_level_one_checkpoint(
        checkpoint, decision.choice.game
    )
    parent_sha = next(
        frontier["parent_checkpoint_sha256"]
        for frontier in decision.eligible_frontiers
        if frontier["game"] == decision.choice.game
    )
    promotion = {
        "attempt_id": "attempt:1",
        "from_level": 0,
        "to_level": 1,
        "parent_checkpoint_sha256": parent_sha,
        "candidate_manifest_sha256": manifest_sha,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "source_path": str(source),
        "source_tree_sha256": source_sha,
    }
    _append_event(
        campaign,
        events,
        event_id="attempt:1:promotion",
        kind="PROMOTION_COMMITTED",
        payload=promotion,
    )
    _append_event(
        campaign,
        events,
        event_id="attempt:1:promotion:replay",
        kind="PROMOTION_COMMITTED",
        payload=promotion,
    )
    receipt = S.audit_campaign(campaign)
    assert receipt["verdict"] == "FAIL"
    assert "no exact candidate" in receipt["findings"][0]


def test_phase_reducer_reopens_complete_terminal_wip_evidence(tmp_path):
    campaign, _, _, events, decision = _one_reserved_campaign(tmp_path)
    wip = _terminal_wip(tmp_path, decision)
    _append_clean_no_progress_result(campaign, events, wip=wip)
    phase_summary = S.validate_journal_event_sequence(
        S.read_journal(campaign)
    )
    assert phase_summary["settlements"] == 1
    receipt = S.audit_campaign(campaign)
    assert receipt["verdict"] == "FAIL"
    assert "full runner lifecycle audit failed" in receipt["findings"][0]


def test_complete_terminal_wip_audit_reopens_every_bound_artifact(
    tmp_path,
):
    campaign, _, _, events, decision = _one_reserved_campaign(tmp_path)
    wip = _terminal_wip(tmp_path, decision)
    _append_clean_no_progress_result(campaign, events, wip=wip)
    baseline = S.validate_journal_event_sequence(
        S.read_journal(campaign)
    )
    assert baseline["settlements"] == 1

    mutations = {
        "source": Path(wip.solver_source_path) / "legs.py",
        "state": Path(wip.app_server_state_dir) / "state.json",
        "thread": Path(wip.final_thread_binding_path),
        "taint": Path(wip.taint_scan_receipt_path),
        "token": Path(wip.token_usage_receipt_path),
        "provider": Path(wip.provider_usage_receipt_path),
    }
    for label, path in mutations.items():
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with pytest.raises(S.SchedulerError) as rejected:
            S.validate_journal_event_sequence(
                S.read_journal(campaign)
            )
        assert str(rejected.value), label
        path.write_bytes(original)
    assert S.validate_journal_event_sequence(
        S.read_journal(campaign)
    ) == baseline


def test_phase_only_fixture_cannot_mint_pre_retention_pass(
    tmp_path,
):
    campaign, _, _, events, decision = _one_reserved_campaign(tmp_path)
    wip = _terminal_wip(tmp_path, decision)
    _append_clean_no_progress_result(campaign, events, wip=wip)
    receipt = S.audit_campaign(campaign)
    assert receipt["verdict"] == "FAIL"
    receipt_path = tmp_path / "scheduler-pre-retention.json"
    receipt_path.write_bytes(S.canonical_json(receipt) + b"\n")
    with pytest.raises(S.SchedulerError):
        S.verify_audit_receipt(campaign, receipt_path)
    with pytest.raises(S.SchedulerError):
        S.verify_pre_retention_audit_receipt(
            campaign,
            receipt_path,
            expected_receipt_sha256=receipt["receipt_sha256"],
        )
    with pytest.raises(
        S.SchedulerError, match="binding hash is malformed"
    ):
        S.verify_pre_retention_audit_receipt(
            campaign,
            receipt_path,
            expected_receipt_sha256="wrong",
        )


def test_audit_rejects_nonexistent_terminal_wip_tree(tmp_path):
    campaign, _, _, events, decision = _one_reserved_campaign(tmp_path)
    wip = _terminal_wip(tmp_path, decision, missing_root=True)
    _append_clean_no_progress_result(campaign, events, wip=wip)
    receipt = S.audit_campaign(campaign)
    assert receipt["verdict"] == "FAIL"
    assert "terminal WIP root" in receipt["findings"][0]


def test_audit_rejects_forged_terminal_provider_settlement(tmp_path):
    campaign, _, _, events, decision = _one_reserved_campaign(tmp_path)
    wip = _terminal_wip(
        tmp_path, decision, forged_provider_settlement=True
    )
    _append_clean_no_progress_result(campaign, events, wip=wip)
    receipt = S.audit_campaign(campaign)
    assert receipt["verdict"] == "FAIL"
    assert "provider usage" in receipt["findings"][0]


def test_audit_rejects_result_that_skips_attempt_lifecycle(tmp_path):
    campaign, _, _, events, _ = _one_reserved_campaign(tmp_path)
    _append_event(
        campaign,
        events,
        event_id="attempt:1:result",
        kind="ATTEMPT_RESULT",
        payload={
            "attempt_id": "attempt:1",
            "kind": "clean_no_progress",
            "cost_used": 0.0,
            "authenticated_cost_units": 0,
            "budget_reservation_id": "reservation:1",
            "scheduler_decision_id": "decision:1",
            "reason": "",
            "candidate": None,
            "wip": None,
        },
    )
    receipt = S.audit_campaign(campaign)
    assert receipt["verdict"] == "FAIL"
    assert "missing or duplicated" in receipt["findings"][0]


def test_audit_rejects_forged_authenticated_cost_units(tmp_path):
    campaign, _, _, events, _ = _one_reserved_campaign(tmp_path)
    _append_minimal_completed_lifecycle(campaign, events)
    _append_event(
        campaign,
        events,
        event_id="attempt:1:result",
        kind="ATTEMPT_RESULT",
        payload={
            "attempt_id": "attempt:1",
            "kind": "clean_no_progress",
            "cost_used": 1.0,
            "authenticated_cost_units": 0,
            "budget_reservation_id": "reservation:1",
            "scheduler_decision_id": "decision:1",
            "reason": "",
            "candidate": None,
            "wip": None,
        },
    )
    receipt = S.audit_campaign(campaign)
    assert receipt["verdict"] == "FAIL"
    assert "authenticated scheduler settlement" in receipt["findings"][0]


def test_receipt_verify_fails_after_journal_changes(tmp_path):
    campaign = tmp_path / "campaign"
    source = tmp_path / "source"
    source_sha = _write_source(source)
    events = _genesis(campaign, source, source_sha)
    receipt = S.audit_campaign(campaign)
    retained = tmp_path / "receipt.json"
    retained.write_bytes(S.canonical_json(receipt) + b"\n")
    _append_event(
        campaign,
        events,
        event_id="attempt:orphan:reserved",
        kind="ATTEMPT_RESERVED",
        payload={"attempt_id": "attempt:orphan", "reservation": {}},
    )
    with pytest.raises(S.SchedulerError, match="no longer matches"):
        S.verify_audit_receipt(campaign, retained)

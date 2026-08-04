from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import selectors
import shutil
import stat
import threading
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import pytest

import arc_agi3_contiguous_conformance as Conformance
import arc_agi3_arena_rpc as ArenaRpc
import arc_agi3_contiguous_runner as R
import arc_agi3_contiguous_supervisor as Contract

Transport = R.Transport
Taint = R.Taint


HASH = "a" * 64
IMAGE_DIGEST = "sha256:" + "1" * 64
COST_WINDOW_ID = "test-credit-window-v1"
WORKER_COMMAND = (
    "-I",
    "-m",
    "arc_agi3_proposer_worker",
    "--bridge-socket=/run/arc-agi3/proposer.sock",
    "--bridge-token-file=/run/arc-agi3/proposer-token",
    "--bridge-policy=/arc/input/bridge_policy.json",
    "--arena-socket=/arena/arena.sock",
    "--arena-token-file=/run/arc-agi3/token",
    "--workspace=/arc/workspace",
    "--export=/arc/export",
)
TEST_CONTROLLER_CANARIES = tuple(
    Taint.LiveCanary(
        category=category,
        location_name=f"runner-test:{category}",
        value=hashlib.sha256(
            f"runner-test-canary:{category}".encode()
        ).hexdigest(),
    )
    for category in Taint.CONTROLLER_CANARY_CATEGORIES
)
TEST_PROBE_ISOLATION_EVIDENCE = {
    "schema": Contract.PROBE_ISOLATION_SCHEMA,
    "kind": Contract.PROBE_ISOLATION_KIND,
    "authority": Contract.PROBE_ISOLATION_AUTHORITY,
    "algorithm": Contract.PROBE_ISOLATION_CANARY,
    "mode": Contract.VERIFIED_ISOLATED_CLONE_MODE,
    "seed_snapshot_sha256": "1" * 64,
    "seed_path_sha256": "2" * 64,
    "canary_status": "PASS",
    "failure_stage": "NONE",
    "canary_action": 1,
    "canary_action_sha256": hashlib.sha256(b"1").hexdigest(),
    "mutable_graph_status": "PASS",
    "shared_mutable_identity_count": 0,
    "mutable_graph_observation_sha256": "5" * 64,
    "seed_before_sha256": "3" * 64,
    "left_before_sha256": "3" * 64,
    "right_before_sha256": "3" * 64,
    "left_after_sha256": "4" * 64,
    "right_after_left_sha256": "3" * 64,
    "seed_after_left_sha256": "3" * 64,
    "right_after_sha256": "4" * 64,
    "seed_after_right_sha256": "3" * 64,
    "mutation_observed": True,
    "sibling_unchanged": True,
    "matching_trajectory": True,
    "fallback_process_ready": False,
    "fallback_process_identity_sha256": None,
}
(
    TEST_PROBE_ISOLATION_MODE,
    TEST_PROBE_ISOLATION_SHA256,
) = Contract.validate_probe_isolation_evidence(
    TEST_PROBE_ISOLATION_EVIDENCE,
    expected_seed_snapshot_sha256="1" * 64,
    expected_seed_path_sha256="2" * 64,
)


def test_app_server_pipe_drains_coalesced_lines_before_select():
    read_fd, write_fd = os.pipe()
    selector = selectors.DefaultSelector()
    read_stream = os.fdopen(read_fd, "rb", closefd=False)
    controller = Transport.CodexAppServerController.__new__(
        Transport.CodexAppServerController
    )
    try:
        os.set_blocking(read_fd, False)
        selector.register(
            read_stream, selectors.EVENT_READ, "stdout"
        )
        controller._selector = selector
        controller._stdout_buffer = bytearray()
        controller._stderr_buffer = bytearray()
        controller._stderr_complete = bytearray()
        controller._stdout_bytes_observed = 0
        controller._stderr_bytes_observed = 0
        controller._stdout_eof = False
        controller._stderr_eof = False
        controller._allow_protocol_eof = False
        controller.process = None
        os.write(write_fd, b'{"id":1}\n{"id":2}\n')
        assert controller._read_ready_line(0.5) == (
            "stdout",
            b'{"id":1}',
        )
        # No further write/readiness edge is needed for the retained line.
        assert controller._read_ready_line(0.01) == (
            "stdout",
            b'{"id":2}',
        )
    finally:
        selector.close()
        read_stream.close()
        os.close(read_fd)
        os.close(write_fd)


def _provider_window(
    response,
    *,
    phase,
    sequence,
):
    return Transport.normalize_provider_usage_window(
        response,
        phase=phase,
        observation_sequence=sequence,
        authenticated_response_sha256=(
            hashlib.sha256(
                Transport.canonical_json(
                    {"id": sequence, "result": response}
                )
            ).hexdigest()
        ),
        transcript_chain_sha256=hashlib.sha256(
            f"transcript:{sequence}".encode("ascii")
        ).hexdigest(),
    )


def _provider_tokens():
    return [{
        "total": {
            "inputTokens": 101,
            "cachedInputTokens": 17,
            "outputTokens": 23,
            "reasoningOutputTokens": 11,
            "totalTokens": 124,
        },
    }]


def _explicit_credit_snapshot(*, unlimited, balance, spend_reached):
    return {
        "rateLimitsByLimitId": {
            "codex": {
                "planType": "team",
                "credits": {
                    "hasCredits": True,
                    "unlimited": unlimited,
                    "balance": balance,
                },
                "spendControlReached": spend_reached,
            },
        },
    }


def _percentage_snapshot(*, used, resets_at=2_000_000):
    return {
        "rateLimitsByLimitId": {
            "codex": {
                "planType": "team",
                "primary": {
                    "usedPercent": used,
                    "resetsAt": resets_at,
                    "windowDurationMins": 7 * 24 * 60,
                },
            },
        },
    }


def test_provider_usage_requires_explicit_current_unlimited_proof():
    explicit = _provider_window(
        _explicit_credit_snapshot(
            unlimited=True,
            balance=None,
            spend_reached=False,
        ),
        phase="preflight",
        sequence=1,
    )
    assert explicit.authority == "explicit_unlimited"
    assert explicit.denomination == "credits"
    assert explicit.limit is None
    with pytest.raises(Transport.AppServerTransportError):
        _provider_window(
            _explicit_credit_snapshot(
                unlimited=True,
                balance=None,
                spend_reached=True,
            ),
            phase="preflight",
            sequence=1,
        )
    with pytest.raises(Transport.AppServerTransportError):
        _provider_window(
            {
                "rateLimits": _percentage_snapshot(
                    used=0
                )["rateLimitsByLimitId"]["codex"],
            },
            phase="preflight",
            sequence=1,
        )


def test_provider_usage_cached_unlimited_survives_only_legacy_100_percent():
    pre = _provider_window(
        _explicit_credit_snapshot(
            unlimited=True,
            balance=None,
            spend_reached=False,
        ),
        phase="preflight",
        sequence=1,
    )
    post = _provider_window(
        _percentage_snapshot(used=0),
        phase="postflight",
        sequence=2,
    )
    settlement = Transport.settle_provider_usage(
        pre,
        post,
        token_usage_observations=_provider_tokens(),
    )
    assert settlement.transition == (
        "cached_unlimited_legacy_postflight"
    )
    assert settlement.cost_window_id == pre.cost_window_id
    assert settlement.limit is None
    assert settlement.charge == 0
    assert settlement.requires_readmission is False


def test_provider_usage_new_explicit_finite_window_reenables_controls():
    pre = _provider_window(
        _explicit_credit_snapshot(
            unlimited=True,
            balance=None,
            spend_reached=False,
        ),
        phase="preflight",
        sequence=1,
    )
    post = _provider_window(
        _explicit_credit_snapshot(
            unlimited=False,
            balance=7,
            spend_reached=False,
        ),
        phase="postflight",
        sequence=2,
    )
    settlement = Transport.settle_provider_usage(
        pre,
        post,
        token_usage_observations=_provider_tokens(),
    )
    assert settlement.transition == "finite_window_reenabled"
    assert settlement.requires_readmission is True
    assert settlement.next_cost_window_id == post.cost_window_id
    assert settlement.next_cost_window_id != pre.cost_window_id


def test_provider_usage_rejects_stale_rotation_and_mixed_units():
    pre = _provider_window(
        _percentage_snapshot(used=10),
        phase="preflight",
        sequence=2,
    )
    stale = _provider_window(
        _percentage_snapshot(used=11),
        phase="postflight",
        sequence=2,
    )
    rotated = _provider_window(
        _percentage_snapshot(used=11, resets_at=3_000_000),
        phase="postflight",
        sequence=3,
    )
    mixed = _provider_window(
        _explicit_credit_snapshot(
            unlimited=False,
            balance=4,
            spend_reached=False,
        ),
        phase="postflight",
        sequence=3,
    )
    for post in (stale, rotated, mixed):
        with pytest.raises(Transport.AppServerTransportError):
            Transport.settle_provider_usage(
                pre,
                post,
                token_usage_observations=_provider_tokens(),
            )


def test_provider_usage_round_trip_rejects_untyped_mutation():
    window = _provider_window(
        _percentage_snapshot(used=10),
        phase="preflight",
        sequence=1,
    )
    encoded = Transport.provider_usage_window_to_dict(window)
    encoded["denomination"] = "tokens"
    with pytest.raises(Transport.AppServerTransportError):
        Transport.provider_usage_window_from_dict(encoded)


class Clock:
    def __init__(self, value: float = 1_000_000.0):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeInputBuilder:
    def __init__(self):
        self.layouts: dict[str, R.AttemptLayout] = {}

    @staticmethod
    def _blank_source_values() -> dict[str, str]:
        return {
            "legs.py": "def noop(env):\n    return None\n",
            "players.py": "from legs import noop\n",
            "solve.py": "def solve(env):\n    return None\n",
        }

    def initialize_lane_source(
        self, game: str, destination: Path
    ) -> tuple[str, str]:
        destination.mkdir(parents=True, exist_ok=True)
        for name, source in self._blank_source_values().items():
            target = destination / name
            if not target.exists():
                target.write_text(source, encoding="utf-8")
        digest = Contract._tree_hash(destination)
        R._seal_regular_tree(destination)
        return str(destination), digest

    def prepare(self, layout: R.AttemptLayout) -> R.InputBundleReceipt:
        prior = self.layouts.get(layout.attempt_id)
        if prior is not None:
            assert prior == layout
        self.layouts[layout.attempt_id] = layout
        input_root = Path(layout.input_dir)
        workspace_root = Path(layout.workspace_dir)
        checkpoint = Path(layout.parent_checkpoint_path)
        target = input_root / "checkpoint.json"
        if not target.exists():
            target.write_bytes(checkpoint.read_bytes())
            parent_source = input_root / "parent_source"
            parent_source.mkdir()
            for source_path in Path(
                layout.parent_source_path
            ).iterdir():
                assert source_path.is_file()
                (parent_source / source_path.name).write_bytes(
                    source_path.read_bytes()
                )
                (workspace_root / source_path.name).write_bytes(
                    source_path.read_bytes()
                )
            (workspace_root / "checkpoint.json").write_bytes(
                checkpoint.read_bytes()
            )
            if layout.wip is not None:
                shutil.copytree(layout.wip.path, input_root / "wip")
                for entry in Path(layout.wip.path).rglob("*"):
                    if entry.is_file():
                        relative = entry.relative_to(layout.wip.path)
                        target_entry = workspace_root / relative
                        target_entry.parent.mkdir(
                            parents=True, exist_ok=True
                        )
                        target_entry.write_bytes(entry.read_bytes())
        checkpoint_value = json.loads(checkpoint.read_text(encoding="utf-8"))
        parent_action_count = len(checkpoint_value["final_path"])
        remaining_action_budget = 600 - parent_action_count
        fresh_prefix_required = remaining_action_budget == 0
        brief_path = input_root / "frontier_brief.json"
        policy_path = input_root / "bridge_policy.json"
        brief = {
            "schema": 1,
            "kind": "arc_agi3_contiguous_frontier_brief",
            "campaign_id": layout.campaign_id,
            "generation_id": layout.generation_id,
            "attempt_id": layout.attempt_id,
            "game": layout.game,
            "target_level": layout.target_level,
            "authoritative_target": layout.authoritative_target,
            "parent_checkpoint_sha256": layout.parent_checkpoint_sha256,
            "frontier_sha256": layout.frontier_sha256,
            "parent_action_count": parent_action_count,
            "remaining_action_budget": remaining_action_budget,
            "fresh_prefix_required": fresh_prefix_required,
            "effort": layout.effort,
            "soft_allocation_seconds": layout.soft_allocation_seconds,
            "wip_mode": layout.wip_mode,
            "thread_mode": layout.thread_mode,
            "supervisory_handoff": (
                None
                if layout.supervisory_handoff is None
                else R.Scheduler.supervisory_prompt_projection(
                    layout.supervisory_handoff
                )
            ),
        }
        policy = {
            "schema": 1,
            "kind": "arc_agi3_contiguous_bridge_policy",
            "campaign_id": layout.campaign_id,
            "generation_id": layout.generation_id,
            "attempt_id": layout.attempt_id,
            "game": layout.game,
            "target_level": layout.target_level,
            "frontier_sha256": layout.frontier_sha256,
            "parent_checkpoint_sha256": layout.parent_checkpoint_sha256,
            "protocol_version":
                layout.proposer_transport.bridge_protocol_version,
            "operation_allowlist": list(
                layout.proposer_transport.bridge_operation_allowlist
            ),
            "exec_allowlist": list(
                layout.proposer_transport.bridge_exec_allowlist
            ),
            "workspace_root": "/arc/workspace",
            "export_root": "/arc/export",
            "bounds": {
                "max_request_bytes":
                    layout.proposer_transport.bridge_max_request_bytes,
                "max_response_bytes":
                    layout.proposer_transport.bridge_max_response_bytes,
                "max_file_bytes":
                    layout.proposer_transport.bridge_max_file_bytes,
                "max_total_export_bytes":
                    layout.proposer_transport.bridge_max_total_export_bytes,
                "max_processes":
                    layout.proposer_transport.bridge_max_processes,
                "max_exec_seconds":
                    layout.proposer_transport.bridge_max_exec_seconds,
            },
        }
        if not brief_path.exists():
            brief_path.write_text(
                json.dumps(brief, sort_keys=True), encoding="utf-8"
            )
            (workspace_root / "frontier_brief.json").write_text(
                json.dumps(brief, sort_keys=True), encoding="utf-8"
            )
        if not policy_path.exists():
            policy_path.write_text(
                json.dumps(policy, sort_keys=True), encoding="utf-8"
            )
        tree_hash = Contract._tree_hash(input_root)
        parent_source_hash = Contract._tree_hash(
            input_root / "parent_source"
        )
        initial_workspace_hash = Contract._tree_hash(
            workspace_root
        )
        receipt_path = Path(layout.generation_dir) / (
            "input_bundle_receipt.json"
        )
        body = {
            "schema": R.RUNNER_SCHEMA,
            "campaign_id": layout.campaign_id,
            "generation_id": layout.generation_id,
            "attempt_id": layout.attempt_id,
            "game": layout.game,
            "target_level": layout.target_level,
            "frontier_sha256": layout.frontier_sha256,
            "input_tree_sha256": tree_hash,
            "parent_source_tree_sha256": parent_source_hash,
            "initial_workspace_tree_sha256":
                initial_workspace_hash,
            "parent_checkpoint_sha256":
                layout.parent_checkpoint_sha256,
            "wip_tree_sha256": (
                layout.wip.tree_sha256 if layout.wip else None
            ),
            "wip_solver_source_tree_sha256": (
                layout.wip.solver_source_tree_sha256
                if layout.wip else None
            ),
            "frontier_brief_sha256":
                hashlib.sha256(brief_path.read_bytes()).hexdigest(),
            "bridge_policy_sha256":
                hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            "parent_action_count": parent_action_count,
            "remaining_action_budget": remaining_action_budget,
            "fresh_prefix_required": fresh_prefix_required,
            "supervisory_handoff_sha256": None,
            "supervisory_handoff_binding_receipt_sha256": None,
        }
        if not receipt_path.exists():
            receipt_path.write_text(
                json.dumps(body, sort_keys=True), encoding="utf-8"
            )
        receipt_hash = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        return R.InputBundleReceipt(
            receipt_path=str(receipt_path),
            receipt_sha256=receipt_hash,
            input_tree_sha256=tree_hash,
            parent_source_tree_sha256=parent_source_hash,
            initial_workspace_tree_sha256=
                initial_workspace_hash,
            parent_checkpoint_sha256=layout.parent_checkpoint_sha256,
            wip_tree_sha256=(
                layout.wip.tree_sha256 if layout.wip else None
            ),
            wip_solver_source_tree_sha256=(
                layout.wip.solver_source_tree_sha256
                if layout.wip else None
            ),
            frontier_brief_path=str(brief_path),
            frontier_brief_sha256=body["frontier_brief_sha256"],
            bridge_policy_path=str(policy_path),
            bridge_policy_sha256=body["bridge_policy_sha256"],
            parent_action_count=parent_action_count,
            remaining_action_budget=remaining_action_budget,
            fresh_prefix_required=fresh_prefix_required,
            supervisory_handoff_path=None,
            supervisory_handoff_sha256=None,
            supervisory_handoff_binding_receipt_path=None,
            supervisory_handoff_binding_receipt_sha256=None,
        )


class CrashAfterInputBuilder(FakeInputBuilder):
    def __init__(self):
        super().__init__()
        self.did_crash = False

    def prepare(self, layout: R.AttemptLayout) -> R.InputBundleReceipt:
        receipt = super().prepare(layout)
        if not self.did_crash:
            self.did_crash = True
            raise R.SimulatedCrash()
        return receipt


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_receipt(
    spec: R.AttemptSpec,
    receipt_path: Path,
    kind: str,
    **extra,
) -> tuple[str, str]:
    value = {
        "schema": 1,
        "kind": kind,
        "campaign_id": spec.campaign_id,
        "generation_id": spec.generation_id,
        "attempt_id": spec.attempt_id,
        "attempt_spec_sha256": R.proposer_attempt_binding_sha256(spec),
        **extra,
    }
    return str(receipt_path), _write_json(receipt_path, value)


def _signed_test_blocker_result(
    spec: R.AttemptSpec,
    *,
    canaries: tuple[Taint.LiveCanary, ...] = TEST_CONTROLLER_CANARIES,
) -> R.AttemptResult:
    """Build a host-authenticated blocker fixture for reducer tests."""

    host = Path(spec.host_transcript_path).parent
    parent = Contract.load_trusted_checkpoint(
        Path(spec.parent_checkpoint_path),
        expected_game=spec.game,
        authoritative_target=spec.authoritative_target,
    )
    parent_path = list(parent.final_path)
    parent_path_sha = hashlib.sha256(
        Transport.canonical_json(parent_path)
    ).hexdigest()
    parent_snapshot_sha = hashlib.sha256(
        Transport.canonical_json({
            "game": spec.game,
            "level": parent.reached,
            "path": parent_path,
            "terminal": True,
        })
    ).hexdigest()
    binding_body = {
        "schema": 1,
        "kind": "arena_session_binding",
        "campaign_id": spec.campaign_id,
        "generation_id": spec.generation_id,
        "attempt_id": spec.attempt_id,
        "game": spec.game,
        "parent_level": parent.reached,
        "target_level": spec.target_level,
        "parent_checkpoint_sha256":
            spec.parent_checkpoint_sha256,
        "frontier_sha256": spec.frontier_sha256,
        "exploration_mode": (
            "fresh_prefix"
            if spec.fresh_prefix_required
            else "continue_parent"
        ),
    }
    binding_sha = hashlib.sha256(
        Transport.canonical_json(binding_body)
    ).hexdigest()
    binding_event = {
        **binding_body,
        "binding_sha256": binding_sha,
        "session_id": hashlib.sha256(
            (spec.attempt_id + ":arena").encode()
        ).hexdigest(),
        "parent_path_sha256": parent_path_sha,
        "parent_replay_steps": len(parent_path),
        "seed_snapshot_sha256": parent_snapshot_sha,
        "exploration_seed_path_sha256": "2" * 64,
        "exploration_seed_snapshot_sha256": "1" * 64,
        "probe_isolation_mode": TEST_PROBE_ISOLATION_MODE,
        "probe_isolation_evidence":
            TEST_PROBE_ISOLATION_EVIDENCE,
        "probe_isolation_evidence_sha256":
            TEST_PROBE_ISOLATION_SHA256,
        "real_step_cap": 600,
        "total_step_cap": 1200,
        "reset_cap": 32,
    }
    arena_path, arena_sha = _bound_receipt(
        spec,
        host / "arena_session_binding_receipt.json",
        "contiguous_arena_session_binding",
        binding_event=binding_event,
    )
    arena_result = {
        "binding_sha256": binding_sha,
        "game": spec.game,
        "exploration_mode": binding_body["exploration_mode"],
        "parent_level": parent.reached,
        "levels_completed": parent.reached,
        "parent_path": parent_path,
        "path": parent_path,
        "parent_replay_steps": len(parent_path),
        "exploration_steps": 0,
        "resets": 0,
        "total_steps": len(parent_path),
        "parent_terminal": True,
        "parent_snapshot_sha256": parent_snapshot_sha,
    }
    code = R.HOST_BLOCKER_CODES[0]
    fields = {
        "authority": R.HOST_BLOCKER_AUTHORITY,
        "code": code,
        "game": spec.game,
        "frontier_sha256": spec.frontier_sha256,
        "parent_checkpoint_sha256": spec.parent_checkpoint_sha256,
        "parent_level": parent.reached,
        "target_level": spec.target_level,
        "arena_session_binding_receipt_path": arena_path,
        "arena_session_binding_receipt_sha256": arena_sha,
        "arena_binding_sha256": binding_sha,
        "parent_path_sha256": parent_path_sha,
        "parent_snapshot_sha256": parent_snapshot_sha,
        "parent_terminal": True,
        "arena_host_result": arena_result,
        "arena_host_result_sha256": hashlib.sha256(
            Transport.canonical_json(arena_result)
        ).hexdigest(),
    }
    unsigned = {
        "schema": 1,
        "kind": R.HOST_BLOCKER_RECEIPT_KIND,
        "campaign_id": spec.campaign_id,
        "generation_id": spec.generation_id,
        "attempt_id": spec.attempt_id,
        "attempt_spec_sha256":
            R.proposer_attempt_binding_sha256(spec),
        **fields,
    }
    authentication = R.host_blocker_authentication_sha256(
        unsigned, canaries
    )
    receipt_path, receipt_sha = _bound_receipt(
        spec,
        host / R.HOST_BLOCKER_RECEIPT_NAME,
        R.HOST_BLOCKER_RECEIPT_KIND,
        **fields,
        host_authentication_sha256=authentication,
    )
    evidence = R.HostBlockerEvidence(
        code=code,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha,
    )
    return R.AttemptResult(
        kind="blocker",
        reason=R.HOST_BLOCKER_REASON_PREFIX + code,
        blocker=evidence,
    )


def _rewrite_test_blocker_receipt(
    result: R.AttemptResult,
    *,
    mutate,
    resign: bool,
    canaries: tuple[Taint.LiveCanary, ...] = TEST_CONTROLLER_CANARIES,
) -> R.AttemptResult:
    assert result.blocker is not None
    path = Path(result.blocker.receipt_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    if resign:
        value.pop("host_authentication_sha256", None)
        value["host_authentication_sha256"] = (
            R.host_blocker_authentication_sha256(value, canaries)
        )
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    evidence = replace(
        result.blocker,
        receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    return replace(result, blocker=evidence)


def _app_scan_policy(
    spec: R.AttemptSpec,
) -> tuple[object, str]:
    brief = json.loads(
        Path(spec.frontier_brief_path).read_text(encoding="utf-8")
    )
    prompt = (
        "Solve exactly this receipt-bound ARC-AGI-3 frontier using only "
        "the contiguous_lane namespace. Immutable frontier:\n"
        + Transport.canonical_json(brief).decode("ascii")
    )
    return (
        Taint.AppServerScanPolicy(
            state_root=R.EXPECTED_CONTROLLER_STATE_ROOT,
            neutral_cwd=R.EXPECTED_CONTROLLER_NEUTRAL_CWD,
            model=spec.proposer_transport.model,
            model_provider=spec.proposer_transport.model_provider,
            reasoning_effort=spec.effort,
            thread_mode=spec.thread_mode,
            resume_thread_id=spec.resume_thread_id,
            hard_safety_seconds=spec.hard_safety_seconds,
            max_auth_refreshes=spec.max_auth_refreshes,
            prompt_sha256=hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest(),
        ),
        prompt,
    )


def _write_app_transcript(
    path: Path,
    spec: R.AttemptSpec,
    *,
    thread_id: str,
    turn_id: str,
) -> tuple[str, int]:
    spec = replace(
        spec,
        app_server_state_dir=R.EXPECTED_CONTROLLER_STATE_ROOT,
        neutral_host_cwd_path=R.EXPECTED_CONTROLLER_NEUTRAL_CWD,
    )
    policy, prompt = _app_scan_policy(spec)
    events: list[tuple[str, object]] = []

    critical_config = {
        "model": spec.proposer_transport.model,
        "model_provider": spec.proposer_transport.model_provider,
        "model_reasoning_effort": spec.effort,
        "approval_policy": "never",
        "sandbox_mode": "read-only",
        "web_search": "disabled",
    }
    disabled_features = {
        name: False
        for name in Taint.AUTHORITY_FEATURE_DENYLIST
    }
    origins = {
        **{
            name: {"source": "lane-config"}
            for name in critical_config
        },
        **{
            f"features.{name}": {"source": "lane-config"}
            for name in disabled_features
        },
    }
    preflight_results = {
        "initialize": {
            "codexHome": spec.app_server_state_dir,
            "platformFamily": "unix",
            "platformOs": "macos",
            "userAgent":
                "gkm-arc-agi3-contiguous/0.145.0 (runner-test)",
        },
        "account/login/start": {"type": "chatgptAuthTokens"},
        "account/read": {
            "account": {
                "email": "runner@example.invalid",
                "planType": "unknown",
                "type": "chatgpt",
            },
            "requiresOpenaiAuth": True,
        },
        "account/rateLimits/read": {
            "rateLimitsByLimitId": {
                "codex": {
                    "planType": "team",
                    "credits": {
                        "hasCredits": True,
                        "unlimited": True,
                        "balance": None,
                    },
                    "spendControlReached": False,
                }
            }
        },
        "model/list": {
            "data": [
                {
                    "id": spec.proposer_transport.model,
                    "model": spec.proposer_transport.model,
                    "supportedReasoningEfforts": [
                        {
                            "description": effort,
                            "reasoningEffort": effort,
                        }
                        for effort in (
                            "low",
                            "medium",
                            "high",
                            "xhigh",
                            "max",
                        )
                    ],
                }
            ],
            "nextCursor": None,
        },
        "modelProvider/capabilities/read": {
            "imageGeneration": True,
            "namespaceTools": True,
            "webSearch": True,
        },
        "config/read": {
            "config": {
                **critical_config,
                "apps": None,
                "features": disabled_features,
                "marketplaces": {},
                "mcp_servers": {},
                "plugins": {},
                "tools": None,
            },
            "layers": [
                {
                    "config": {
                        **critical_config,
                        "features": disabled_features,
                    },
                    "name": {
                        "file":
                            f"{spec.app_server_state_dir}/config.toml",
                        "type": "user",
                    },
                },
                {
                    "config": {},
                    "name": {
                        "file": "/etc/codex/config.toml",
                        "type": "system",
                    },
                },
            ],
            "origins": origins,
        },
        "skills/list": {
            "data": [
                {
                    "cwd": spec.neutral_host_cwd_path,
                    "errors": [],
                    "skills": [
                        {
                            "enabled": False,
                            "name": name,
                            "path": (
                                f"{spec.app_server_state_dir}/skills/"
                                f".system/{name}/SKILL.md"
                            ),
                            "scope": "system",
                        }
                        for name in Transport.DISABLED_SYSTEM_SKILLS
                    ],
                }
            ]
        },
        "hooks/list": {
            "data": [
                {
                    "cwd": spec.neutral_host_cwd_path,
                    "errors": [],
                    "hooks": [],
                    "warnings": [],
                }
            ]
        },
        "plugin/list": {
            "featuredPluginIds": [],
            "marketplaceLoadErrors": [],
            "marketplaces": [],
        },
        "app/list": {"data": [], "nextCursor": None},
        "experimentalFeature/list": {
            "data": [
                {"enabled": False, "name": name}
                for name in sorted(Taint.AUTHORITY_FEATURE_DENYLIST)
            ],
            "nextCursor": None,
        },
        "mcpServerStatus/list": {
            "data": [],
            "nextCursor": None,
        },
    }
    for index, method in enumerate(
        Transport.PREFLIGHT_REQUEST_SEQUENCE, 1
    ):
        request_id = f"preflight-{index}"
        params = (
            {
                "type": "chatgptAuthTokens",
                "accessToken": "REDACTED",
                "chatgptAccountId": "REDACTED",
            }
            if method == "account/login/start"
            else dict(
                Taint._expected_preflight_params(method, policy)
                or {}
            )
        )
        events.extend(
            (
                (
                    "client_request",
                    {
                        "id": request_id,
                        "method": method,
                        "params": params,
                    },
                ),
                (
                    "server_response",
                    {
                        "id": request_id,
                        "result": preflight_results[method],
                    },
                ),
            )
        )
        if method == "initialize":
            events.append(
                (
                    "client_notification",
                    {"method": "initialized", "params": {}},
                )
            )
        elif method == "account/login/start":
            events.extend(
                (
                    (
                        "server_notification",
                        {
                            "method": "account/login/completed",
                            "params": {
                                "error": None,
                                "loginId": None,
                                "success": True,
                            },
                            "emittedAtMs": 1,
                        },
                    ),
                    (
                        "server_notification",
                        {
                            "method": "account/updated",
                            "params": {
                                "authMode": "chatgptAuthTokens",
                                "planType": "unknown",
                            },
                            "emittedAtMs": 1,
                        },
                    ),
                )
            )

    common_thread = {
        "approvalPolicy": "never",
        "baseInstructions": Transport.BASE_INSTRUCTIONS,
        "cwd": spec.neutral_host_cwd_path,
        "developerInstructions": Transport.DEVELOPER_INSTRUCTIONS,
        "model": spec.proposer_transport.model,
        "modelProvider": spec.proposer_transport.model_provider,
        "runtimeWorkspaceRoots": [spec.neutral_host_cwd_path],
        "sandbox": "read-only",
    }
    if spec.thread_mode == "resume":
        thread_method = "thread/resume"
        thread_params = {
            **common_thread,
            "threadId": spec.resume_thread_id,
            "excludeTurns": False,
        }
    else:
        thread_method = "thread/start"
        thread_params = {
            **common_thread,
            "allowProviderModelFallback": False,
            "dynamicTools": list(Transport.DYNAMIC_TOOL_SPECS),
            "environments": [],
            "ephemeral": False,
            "experimentalRawEvents": False,
            "historyMode": "paginated",
            "selectedCapabilityRoots": [],
        }
    turn_params = {
        "threadId": thread_id,
        "input": [
            {
                "type": "text",
                "text": prompt,
                "text_elements": [],
            }
        ],
        "approvalPolicy": "never",
        "cwd": spec.neutral_host_cwd_path,
        "effort": spec.effort,
        "environments": [],
        "model": spec.proposer_transport.model,
        "runtimeWorkspaceRoots": [spec.neutral_host_cwd_path],
        "sandboxPolicy": {
            "type": "readOnly",
            "networkAccess": False,
        },
    }
    events.extend(
        (
            (
                "client_request",
                {
                    "id": "thread-request",
                    "method": thread_method,
                    "params": thread_params,
                },
            ),
            (
                "server_response",
                {
                    "id": "thread-request",
                    "result": {"thread": {"id": thread_id}},
                },
            ),
            (
                "server_notification",
                {
                    "method": "thread/started",
                    "params": {"thread": {"id": thread_id}},
                },
            ),
            (
                "client_request",
                {
                    "id": "turn-request",
                    "method": "turn/start",
                    "params": turn_params,
                },
            ),
            (
                "server_response",
                {
                    "id": "turn-request",
                    "result": {"turn": {"id": turn_id}},
                },
            ),
            (
                "server_notification",
                {
                    "method": "turn/started",
                    "params": {
                        "threadId": thread_id,
                        "turn": {"id": turn_id},
                    },
                },
            ),
            (
                "server_notification",
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": thread_id,
                        "turn": {"id": turn_id},
                    },
                },
            ),
            (
                "server_notification",
                {
                    "method": "thread/tokenUsage/updated",
                    "params": {
                        "threadId": thread_id,
                        "turnId": turn_id,
                    },
                },
            ),
        )
    )

    previous: str | None = None
    rows: list[bytes] = []
    for sequence, (direction, payload) in enumerate(events, 1):
        body = {
            "schema": 1,
            "sequence": sequence,
            "previous_digest": previous,
            "direction": direction,
            "payload": payload,
        }
        digest = hashlib.sha256(
            Transport.canonical_json(body)
        ).hexdigest()
        rows.append(
            Transport.canonical_json({**body, "digest": digest})
        )
        previous = digest
    path.write_bytes(b"\n".join(rows) + b"\n")
    assert previous is not None
    return previous, len(rows)


def _fake_target_boundary(
    spec: R.AttemptSpec,
) -> tuple[str, str, str, str]:
    """Freeze candidate workspace evidence with the production receipt shape."""

    workspace = Path(spec.workspace_dir)
    host = Path(spec.host_transcript_path).parent
    snapshot = host / "target_boundary_workspace"
    shutil.copytree(workspace, snapshot)
    inventory = Transport.inventory_controller_state(snapshot)
    request = {
        "request_id": str(
            uuid.uuid5(uuid.NAMESPACE_URL, spec.attempt_id + ":boundary")
        ),
        "sequence": 2,
        "mutation_id": spec.attempt_id + ":00000001",
        "arguments": {"action": 1},
    }
    boundary = {
        "schema": 1,
        "kind": "arc_agi3_contiguous_target_boundary",
        "attempt_id": spec.attempt_id,
        "game": spec.game,
        "target_level": spec.target_level,
        "levels_before": spec.target_level - 1,
        "levels_completed": spec.target_level,
        "arena_binding_sha256": "e" * 64,
        "bridge_request_id": request["request_id"],
        "bridge_sequence": request["sequence"],
        "bridge_mutation_id": request["mutation_id"],
        "crossing_action_sha256": hashlib.sha256(
            Transport.canonical_json(request["arguments"]["action"])
        ).hexdigest(),
        "exploration_suffix_sha256": hashlib.sha256(
            Transport.canonical_json([request["arguments"]["action"]])
        ).hexdigest(),
        "exploration_suffix_length": 1,
        "workspace_tree_sha256": inventory.tree_sha256,
        "workspace_inventory_sha256": inventory.inventory_sha256,
        "workspace_file_count": inventory.file_count,
        "workspace_total_bytes": inventory.total_bytes,
    }
    result = {
        "target_reached": True,
        "boundary": boundary,
        "boundary_sha256": hashlib.sha256(
            Transport.canonical_json(boundary)
        ).hexdigest(),
    }
    response = {"result": result}
    receipt_path, receipt_sha256 = _bound_receipt(
        spec,
        host / "target_boundary_receipt.json",
        "contiguous_target_boundary",
        boundary=boundary,
        boundary_sha256=result["boundary_sha256"],
        bridge_request=request,
        bridge_response=response,
        bridge_request_sha256=hashlib.sha256(
            Transport.canonical_json(request)
        ).hexdigest(),
        bridge_response_sha256=hashlib.sha256(
            Transport.canonical_json(response)
        ).hexdigest(),
        snapshot_root=str(snapshot),
        workspace_inventory=inventory.as_receipt(),
        pre_response_delivery=True,
        next_level_observation_withheld=True,
        workspace_frozen=True,
    )
    R._seal_regular_tree(snapshot)
    return (
        receipt_path,
        receipt_sha256,
        result["boundary_sha256"],
        inventory.tree_sha256,
    )


def _fake_canary_anchor(
    spec: R.AttemptSpec,
    canaries: tuple[Taint.LiveCanary, ...],
) -> dict[str, str]:
    normalized = Taint.validate_live_canaries(
        canaries, require_complete=True
    )
    commitments = [item.commitment() for item in normalized]
    placements = [
        {
            "category": item.category,
            "location_name": item.location_name,
            "provenance": item.provenance,
        }
        for item in normalized
    ]
    commitments_json = Transport.canonical_json(
        commitments
    ).decode("ascii")
    placements_json = Transport.canonical_json(
        placements
    ).decode("ascii")
    campaign = Path(spec.generation_dir).parent.parent
    escrow = (
        campaign
        / "containment_canary_escrow"
        / f"{spec.generation_id}.json"
    )
    escrow.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    _write_json(
        escrow,
        {
            "schema": 1,
            "kind": "contiguous_controller_canary_escrow",
            "campaign_id": spec.campaign_id,
            "generation_id": spec.generation_id,
            "attempt_id": spec.attempt_id,
            "attempt_spec_sha256":
                R.proposer_attempt_binding_sha256(spec),
            "storage_policy": "host_only_never_mounted",
            "canary_commitments": commitments,
            "host_only_canary_escrow":
                Taint.build_live_canary_reveal(normalized),
        },
    )
    os.chmod(escrow, 0o400, follow_symlinks=False)
    raw = escrow.read_bytes()
    metadata = os.stat(escrow, follow_symlinks=False)
    escrow_sha = hashlib.sha256(raw).hexdigest()
    identity_sha = hashlib.sha256(
        Transport.canonical_json({
            "path": str(escrow),
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": stat.S_IMODE(metadata.st_mode),
            "owner_uid": metadata.st_uid,
            "owner_gid": metadata.st_gid,
            "size": metadata.st_size,
            "sha256": escrow_sha,
        })
    ).hexdigest()
    return {
        "escrow_path": str(escrow),
        "escrow_sha256": escrow_sha,
        "escrow_identity_sha256": identity_sha,
        "commitments_json": commitments_json,
        "commitments_sha256": hashlib.sha256(
            commitments_json.encode("ascii")
        ).hexdigest(),
        "placement_descriptors_json": placements_json,
        "placement_descriptors_sha256": hashlib.sha256(
            placements_json.encode("ascii")
        ).hexdigest(),
    }


class FakeBackend:
    requires_controller_state_canaries = True

    def __init__(self, strategy=None):
        self.strategy = strategy
        self.controller_state_canaries = TEST_CONTROLLER_CANARIES
        self.specs: dict[str, R.AttemptSpec] = {}
        self.results: dict[str, R.AttemptResult] = {}
        self.preparations: dict[str, R.BackendPreparation] = {}
        self.launches: dict[str, R.BackendLaunch] = {}
        self.collections: dict[str, R.BackendCollection] = {}
        self.prepare_calls: list[str] = []
        self.launch_calls: list[str] = []
        self.poll_timeouts: list[float] = []
        self.teardown_calls: list[str] = []
        self.emergency_containment_calls: list[
            tuple[str, str]
        ] = []
        self.crash_after_first_launch = False
        self.crash_after_first_collect = False
        self._did_launch_crash = False
        self._did_collect_crash = False
        self.bad_teardown = False
        self.fail_prepare_after: int | None = None
        self.public_action_protocol_invalid = False
        self.protocol_invalid_scans: dict[
            str, tuple[str, str]
        ] = {}

    def prepare(self, spec: R.AttemptSpec) -> R.BackendPreparation:
        self.prepare_calls.append(spec.attempt_id)
        if (
            self.fail_prepare_after is not None
            and len(self.preparations) >= self.fail_prepare_after
            and spec.attempt_id not in self.preparations
        ):
            raise RuntimeError("synthetic preparation failure")
        prior = self.specs.get(spec.attempt_id)
        if prior is not None:
            assert prior == spec
        else:
            self.specs[spec.attempt_id] = spec
            if self.strategy is not None:
                result = self.strategy(spec)
                if result is not None:
                    self.results[spec.attempt_id] = result
        host = Path(spec.host_transcript_path).parent
        Path(spec.app_server_state_dir).mkdir(parents=True, exist_ok=True)
        neutral = Path(spec.neutral_host_cwd_path)
        neutral.mkdir(parents=True, exist_ok=True)
        canary_anchor = _fake_canary_anchor(
            spec, self.controller_state_canaries
        )
        attestation = host / "launch_attestation.json"
        if not attestation.exists():
            compatibility_snapshot = (
                R.CompatibilityClosure.canonical_closure_snapshot()
            )
            compatibility_client_sha256 = compatibility_snapshot[
                "client"
            ]["source_sha256"]
            compatibility_controls = compatibility_snapshot[
                "components"
            ]
            fake_socket_endpoint = {
                "path": str(host / "arena-mirror.sock"),
                "kind": "socket",
                "device": 1,
                "inode": 2,
                "mode": 0o600,
                "owner_uid": os.getuid(),
                "owner_gid": os.getgid(),
                "size": 0,
                "content_sha256": None,
            }
            fake_token_endpoint = {
                "path": spec.arena_token_file_path,
                "kind": "token",
                "device": 1,
                "inode": 3,
                "mode": 0o400,
                "owner_uid": os.getuid(),
                "owner_gid": os.getgid(),
                "size": 64,
                "content_sha256": "d" * 64,
            }
            attestation.write_text(
                json.dumps(
                    {
                        "attempt_id": spec.attempt_id,
                        "generation_id": spec.generation_id,
                        "image_digest": spec.image_digest,
                        "container_id": "e" * 64,
                        "container_inspect_sha256": "3" * 64,
                        "security_projection_sha256": "4" * 64,
                        "create_argv_sha256": "5" * 64,
                        "image": {
                            "requested_reference": spec.image_reference,
                            "manifest_digest": spec.image_digest,
                            "image_id": "sha256:" + "f" * 64,
                            "image_inspect_sha256": "2" * 64,
                            "worker_control_sha256": {
                                "org.gkm.arc-agi3.arena-rpc-client-sha256":
                                    compatibility_client_sha256,
                                "org.gkm.arc-agi3.container-recipe-sha256":
                                    compatibility_controls[
                                        "container_recipe"
                                    ]["sha256"],
                                "org.gkm.arc-agi3.solver-requirements-sha256":
                                    compatibility_controls[
                                        "solver_requirements"
                                    ]["sha256"],
                            },
                        },
                        "arena_rpc": {
                            "socket": fake_socket_endpoint,
                            "token_file": fake_token_endpoint,
                        },
                        "containment_canary_anchor": canary_anchor,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        arena_receipt_file = (
            host / "arena_session_binding_receipt.json"
        )
        if arena_receipt_file.exists():
            retained_arena = json.loads(
                arena_receipt_file.read_text(encoding="utf-8")
            )
            retained_event = retained_arena["binding_event"]
            arena_binding_path = str(arena_receipt_file)
            arena_binding_sha = hashlib.sha256(
                arena_receipt_file.read_bytes()
            ).hexdigest()
            probe_mode = retained_event["probe_isolation_mode"]
            probe_sha256 = retained_event[
                "probe_isolation_evidence_sha256"
            ]
        else:
            retained_event = {
                "campaign_id": spec.campaign_id,
                "generation_id": spec.generation_id,
                "attempt_id": spec.attempt_id,
                "game": spec.game,
                "parent_level": spec.target_level - 1,
                "target_level": spec.target_level,
                "parent_checkpoint_sha256":
                    spec.parent_checkpoint_sha256,
                "frontier_sha256": spec.frontier_sha256,
                "exploration_seed_snapshot_sha256": "1" * 64,
                "exploration_seed_path_sha256": "2" * 64,
                "probe_isolation_mode":
                    TEST_PROBE_ISOLATION_MODE,
                "probe_isolation_evidence":
                    TEST_PROBE_ISOLATION_EVIDENCE,
                "probe_isolation_evidence_sha256":
                    TEST_PROBE_ISOLATION_SHA256,
                "binding_sha256": "d" * 64,
                "session_id": "e" * 64,
            }
            arena_binding_path, arena_binding_sha = _bound_receipt(
                spec,
                arena_receipt_file,
                "contiguous_arena_session_binding",
                binding_event=retained_event,
            )
            probe_mode = TEST_PROBE_ISOLATION_MODE
            probe_sha256 = TEST_PROBE_ISOLATION_SHA256
        bridge_policy_path, bridge_policy_sha = _bound_receipt(
            spec,
            Path(spec.bridge_policy_receipt_path),
            "contiguous_bridge_policy",
            arena_session_binding_receipt_path=arena_binding_path,
            arena_session_binding_receipt_sha256=arena_binding_sha,
            probe_isolation_mode=probe_mode,
            probe_isolation_evidence_sha256=probe_sha256,
        )
        neutral_metadata = neutral.stat(follow_symlinks=False)
        neutral_path, neutral_sha = _bound_receipt(
            spec,
            host / "neutral_cwd_attestation.json",
            "contiguous_neutral_cwd_attestation",
            path=spec.neutral_host_cwd_path,
            owner_uid=neutral_metadata.st_uid,
            owner_gid=neutral_metadata.st_gid,
            mode=stat.S_IMODE(neutral_metadata.st_mode),
            tree_sha256=Contract._tree_hash(neutral),
            write_probe_status="DENIED",
        )
        transport = spec.proposer_transport
        config_path, config_sha = _bound_receipt(
            spec,
            host / "app_server_config_receipt.json",
            "contiguous_app_server_config",
            model=transport.model,
            model_provider=transport.model_provider,
            allow_provider_model_fallback=False,
            reasoning_effort=spec.effort,
            environments=[],
            selected_capability_roots=[],
            runtime_workspace_roots=["/controller-neutral"],
            native_proposer_workspace={
                "root": "/controller-neutral",
                "storage": "private-tmpfs",
                "git_root_equals_workspace": True,
                "git_ceiling_directories": "/controller-neutral",
                "git_discovery_across_filesystem": False,
                "parent_repo_mounts": 0,
                "campaign_plan_mounts": 0,
                "sidecar_or_quarantine_mounts": 0,
                "manuscript_comparator_benchmark_mounts": 0,
                "symlinks_allowed": False,
                "hardlinks_allowed": False,
            },
            dynamic_tool_namespace=transport.dynamic_tool_namespace,
            dynamic_tool_names=list(transport.dynamic_tool_names),
            controller_method_policy={
                "preflight_requests": list(
                    transport.controller_preflight_request_allowlist
                ),
                "preflight_notifications": list(
                    transport.controller_preflight_notification_allowlist
                ),
                "turn_requests": list(
                    transport.controller_turn_request_allowlist
                ),
            },
            builtin_tool_names=[],
            approval_policy="never",
            sandbox_policy={"type": "readOnly", "networkAccess": False},
            state_root="/controller-state",
            state_host_staging_root=spec.app_server_state_dir,
            state_mode=(
                "resume_staged_copy"
                if spec.thread_mode == "resume"
                else "new_reset"
            ),
            prior_state_root=(
                spec.wip.app_server_state_dir
                if spec.wip is not None
                else None
            ),
            prior_state_tree_sha256=(
                spec.wip.app_server_state_tree_sha256
                if spec.wip is not None
                else None
            ),
            staged_state_root=spec.app_server_state_dir,
            staged_initial_state_tree_sha256=(
                spec.initial_app_server_state_tree_sha256
            ),
            ambient_state_access_status="DENIED",
            state_root_write_probe_status=(
                "PENDING_REAL_CONTROLLER_PREFLIGHT"
            ),
            ambient_environment_names_stripped=[
                "CODEX_HOME",
                "HOME",
                "XDG_CONFIG_HOME",
                "XDG_DATA_HOME",
                "XDG_STATE_HOME",
            ],
        )
        binary_path, binary_sha = _bound_receipt(
            spec,
            host / "codex_binary_receipt.json",
            "contiguous_codex_binary",
            launcher_path=transport.codex_launcher_path,
            launcher_sha256=transport.codex_launcher_sha256,
            package_manifest_path=transport.codex_package_manifest_path,
            package_manifest_sha256=
                transport.codex_package_manifest_sha256,
            native_binary_path=transport.codex_binary_path,
            native_binary_sha256=transport.codex_binary_sha256,
            native_binary_bytes=transport.codex_binary_bytes,
            version=transport.codex_cli_version,
            observation_stage="pending_controller_guardian",
            controller_image_digest=transport.controller_image_digest,
            host_file_observation=False,
        )
        protocol_path, protocol_sha = _bound_receipt(
            spec,
            host / "app_server_protocol_schema_receipt.json",
            "contiguous_app_server_protocol_schema",
            path=transport.app_server_protocol_schema_path,
            sha256=transport.app_server_protocol_schema_sha256,
            bundle_path=transport.app_server_protocol_schema_bundle_path,
            bundle_sha256=
                transport.app_server_protocol_schema_bundle_sha256,
            observation_stage="pending_controller_guardian",
            controller_image_digest=transport.controller_image_digest,
            host_file_observation=False,
        )
        arena_volume_name = R._arena_volume_name(spec)
        arena_volume_observation_sha256 = "7" * 64
        arena_relay_container_id = hashlib.sha256(
            (spec.attempt_id + ":arena-relay").encode("ascii")
        ).hexdigest()
        arena_relay_image_observation_sha256 = "8" * 64
        arena_relay_container_observation_sha256 = "9" * 64
        arena_relay_attach_argv_sha256 = "a" * 64
        arena_relay_socket_identity_sha256 = "b" * 64
        readiness_nonce = "c" * 64
        arena_relay_readiness = (
            host / "arena_volume_readiness.json"
        )
        arena_relay_readiness.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "kind":
                        "arc_agi3_arena_volume_relay_readiness",
                    "status": "READY",
                    "campaign_id": spec.campaign_id,
                    "generation_id": spec.generation_id,
                    "attempt_id": spec.attempt_id,
                    "readiness_nonce": readiness_nonce,
                    "relay_pid": 4242,
                    "socket_path": "/arena/arena.sock",
                    "socket_mode": 0o666,
                    "network_mode_required": "none",
                    "transport": R.ARENA_VOLUME_TRANSPORT,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        os.chmod(arena_relay_readiness, 0o600)
        arena_relay_readiness_sha256 = hashlib.sha256(
            arena_relay_readiness.read_bytes()
        ).hexdigest()
        arena_relay_preparation = (
            host / "arena_volume_preparation.json"
        )
        arena_relay_preparation.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "kind": "arc_agi3_arena_volume_preparation",
                    "campaign_id": spec.campaign_id,
                    "generation_id": spec.generation_id,
                    "attempt_id": spec.attempt_id,
                    "game": spec.game,
                    "target_level": spec.target_level,
                    "transport": R.ARENA_VOLUME_TRANSPORT,
                    "volume_name": arena_volume_name,
                    "volume_observation_sha256":
                        arena_volume_observation_sha256,
                    "relay_container_id":
                        arena_relay_container_id,
                    "relay_image_reference":
                        transport.arena_relay_image_reference,
                    "relay_image_digest":
                        transport.arena_relay_image_digest,
                    "relay_image_observation_sha256":
                        arena_relay_image_observation_sha256,
                    "relay_container_observation_sha256":
                        arena_relay_container_observation_sha256,
                    "readiness_nonce": readiness_nonce,
                    "readiness_receipt_path":
                        str(arena_relay_readiness),
                    "readiness_receipt_sha256":
                        arena_relay_readiness_sha256,
                    "attach_argv_sha256":
                        arena_relay_attach_argv_sha256,
                    "arena_socket_identity_sha256":
                        arena_relay_socket_identity_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        os.chmod(arena_relay_preparation, 0o600)
        arena_relay_preparation_sha256 = hashlib.sha256(
            arena_relay_preparation.read_bytes()
        ).hexdigest()
        closure_root = host / "compatibility_arena_closure"
        if closure_root.exists():
            closure_receipt_sha256 = hashlib.sha256(
                (closure_root / "closure_receipt.json").read_bytes()
            ).hexdigest()
            closure_observation = (
                R.CompatibilityClosure.validate_closure(
                    closure_root, closure_receipt_sha256
                )
            )
        else:
            closure_observation = (
                R.CompatibilityClosure.prepare_closure(closure_root)
            )
            closure_receipt_sha256 = closure_observation[
                "receipt_sha256"
            ]
        attestation_value = json.loads(
            attestation.read_text(encoding="utf-8")
        )
        closure_controls = (
            R.CompatibilityClosure.canonical_closure_snapshot()[
                "components"
            ]
        )
        compatibility_turn_path, compatibility_turn_sha256 = (
            _bound_receipt(
                spec,
                host / "compatibility_turn_receipt.json",
                "arc_agi3_contiguous_compatibility_turn_binding",
                game=spec.game,
                target_level=spec.target_level,
                frontier_sha256=spec.frontier_sha256,
                parent_checkpoint_sha256=(
                    spec.parent_checkpoint_sha256
                ),
                closure={
                    "root": str(closure_root),
                    "receipt_sha256": closure_receipt_sha256,
                    "content_manifest_sha256": closure_observation[
                        "content_manifest_sha256"
                    ],
                    "client_sha256": closure_observation[
                        "client_sha256"
                    ],
                },
                host_rpc={
                    "session_binding_receipt_path": arena_binding_path,
                    "session_binding_receipt_sha256": arena_binding_sha,
                    "binding_sha256": retained_event[
                        "binding_sha256"
                    ],
                    "session_id": retained_event["session_id"],
                    "host_socket_path": spec.arena_socket_path,
                    "host_socket_identity_sha256": (
                        arena_relay_socket_identity_sha256
                    ),
                    "container_socket": attestation_value[
                        "arena_rpc"
                    ]["socket"],
                    "token_file": attestation_value["arena_rpc"][
                        "token_file"
                    ],
                    "token_bytes_retained": False,
                },
                transport={
                    "kind": R.ARENA_VOLUME_TRANSPORT,
                    "volume_name": arena_volume_name,
                    "volume_observation_sha256": (
                        arena_volume_observation_sha256
                    ),
                    "relay_container_id": arena_relay_container_id,
                    "relay_image_digest": (
                        transport.arena_relay_image_digest
                    ),
                    "relay_image_observation_sha256": (
                        arena_relay_image_observation_sha256
                    ),
                    "relay_container_observation_sha256": (
                        arena_relay_container_observation_sha256
                    ),
                    "readiness_receipt_sha256": (
                        arena_relay_readiness_sha256
                    ),
                    "preparation_receipt_sha256": (
                        arena_relay_preparation_sha256
                    ),
                    "attach_argv_sha256": (
                        arena_relay_attach_argv_sha256
                    ),
                },
                container={
                    "container_id": attestation_value["container_id"],
                    "requested_image_reference": attestation_value[
                        "image"
                    ]["requested_reference"],
                    "image_manifest_digest": attestation_value["image"][
                        "manifest_digest"
                    ],
                    "image_id": attestation_value["image"]["image_id"],
                    "image_observation_sha256": attestation_value["image"][
                        "image_inspect_sha256"
                    ],
                    "worker_control_sha256": attestation_value["image"][
                        "worker_control_sha256"
                    ],
                    "container_observation_sha256": attestation_value[
                        "container_inspect_sha256"
                    ],
                    "security_projection_sha256": attestation_value[
                        "security_projection_sha256"
                    ],
                    "create_argv_sha256": attestation_value[
                        "create_argv_sha256"
                    ],
                    "launch_attestation_sha256": hashlib.sha256(
                        attestation.read_bytes()
                    ).hexdigest(),
                    "container_recipe_sha256": closure_controls[
                        "container_recipe"
                    ]["sha256"],
                    "solver_requirements_sha256": closure_controls[
                        "solver_requirements"
                    ]["sha256"],
                },
                authority={
                    "scheduler_authority": False,
                    "mutation_authority": False,
                    "promotion_authority": False,
                    "launch_authority": False,
                    "runner_reopen_required_before_launch": True,
                },
            )
        )
        prepared = R.BackendPreparation(
            preparation_id="prepare:" + spec.attempt_id,
            launch_attestation_path=str(attestation),
            launch_attestation_sha256=hashlib.sha256(
                attestation.read_bytes()
            ).hexdigest(),
            observed_image_digest=spec.image_digest,
            image_observation_sha256="2" * 64,
            container_observation_sha256="3" * 64,
            bridge_policy_receipt_path=bridge_policy_path,
            bridge_policy_receipt_sha256=bridge_policy_sha,
            arena_session_binding_receipt_path=arena_binding_path,
            arena_session_binding_receipt_sha256=arena_binding_sha,
            compatibility_closure_path=str(closure_root),
            compatibility_closure_receipt_sha256=(
                closure_receipt_sha256
            ),
            compatibility_turn_receipt_path=compatibility_turn_path,
            compatibility_turn_receipt_sha256=(
                compatibility_turn_sha256
            ),
            arena_transport=R.ARENA_VOLUME_TRANSPORT,
            arena_volume_name=arena_volume_name,
            arena_volume_observation_sha256=(
                arena_volume_observation_sha256
            ),
            arena_relay_container_id=arena_relay_container_id,
            arena_relay_image_digest=(
                transport.arena_relay_image_digest
            ),
            arena_relay_image_observation_sha256=(
                arena_relay_image_observation_sha256
            ),
            arena_relay_container_observation_sha256=(
                arena_relay_container_observation_sha256
            ),
            arena_relay_readiness_receipt_path=str(
                arena_relay_readiness
            ),
            arena_relay_readiness_receipt_sha256=(
                arena_relay_readiness_sha256
            ),
            arena_relay_attach_argv_sha256=(
                arena_relay_attach_argv_sha256
            ),
            arena_relay_socket_identity_sha256=(
                arena_relay_socket_identity_sha256
            ),
            arena_relay_preparation_receipt_path=str(
                arena_relay_preparation
            ),
            arena_relay_preparation_receipt_sha256=(
                arena_relay_preparation_sha256
            ),
            probe_isolation_mode=probe_mode,
            probe_isolation_evidence_sha256=probe_sha256,
            neutral_cwd_attestation_path=neutral_path,
            neutral_cwd_attestation_sha256=neutral_sha,
            app_server_config_receipt_path=config_path,
            app_server_config_receipt_sha256=config_sha,
            codex_binary_receipt_path=binary_path,
            codex_binary_receipt_sha256=binary_sha,
            protocol_schema_receipt_path=protocol_path,
            protocol_schema_receipt_sha256=protocol_sha,
            controller_image_digest=transport.controller_image_digest,
            controller_egress_proxy_image_digest=(
                transport.controller_egress_proxy_image_digest
            ),
            controller_egress_policy_sha256=(
                transport.controller_egress_policy_sha256
            ),
            controller_canary_escrow_path=(
                canary_anchor["escrow_path"]
            ),
            controller_canary_escrow_sha256=(
                canary_anchor["escrow_sha256"]
            ),
            controller_canary_escrow_identity_sha256=(
                canary_anchor["escrow_identity_sha256"]
            ),
            controller_canary_commitments_json=(
                canary_anchor["commitments_json"]
            ),
            controller_canary_commitments_sha256=(
                canary_anchor["commitments_sha256"]
            ),
            controller_canary_placement_descriptors_json=(
                canary_anchor["placement_descriptors_json"]
            ),
            controller_canary_placement_descriptors_sha256=(
                canary_anchor[
                    "placement_descriptors_sha256"
                ]
            ),
            controller_supply_chain_unobserved_until_launch=True,
        )
        prior_prepared = self.preparations.setdefault(
            spec.attempt_id, prepared
        )
        assert prior_prepared == prepared
        return prepared

    def launch(
        self, spec: R.AttemptSpec, prepared: R.BackendPreparation
    ) -> R.BackendLaunch:
        assert prepared == self.preparations[spec.attempt_id]
        self.launch_calls.append(spec.attempt_id)
        host = Path(spec.host_transcript_path).parent
        container_id = hashlib.sha256(
            spec.attempt_id.encode()
        ).hexdigest()
        thread_id = (
            spec.resume_thread_id
            if spec.thread_mode == "resume"
            else str(uuid.uuid5(uuid.NAMESPACE_URL, spec.attempt_id + ":thread"))
        )
        assert thread_id is not None
        turn_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, spec.attempt_id + ":turn")
        )
        transcript_head, _ = _write_app_transcript(
            Path(spec.app_server_transcript_path),
            spec,
            thread_id=thread_id,
            turn_id=turn_id,
        )
        bridge_path, bridge_sha = _bound_receipt(
            spec,
            host / "bridge_runtime_attestation.json",
            "contiguous_bridge_runtime",
            container_id=container_id,
            socket_path=spec.bridge_socket_path,
            token_file_path=spec.bridge_token_file_path,
            socket_inode=101,
            token_inode=102,
            token_sha256="7" * 64,
            handshake_nonce_sha256="8" * 64,
            policy_receipt_sha256=
                prepared.bridge_policy_receipt_sha256,
        )
        controller_container_id = hashlib.sha256(
            (spec.attempt_id + ":controller").encode()
        ).hexdigest()
        egress_proxy_container_id = hashlib.sha256(
            (spec.attempt_id + ":egress-proxy").encode()
        ).hexdigest()
        launch_intent_sha256 = hashlib.sha256(
            (spec.attempt_id + ":controller-launch").encode()
        ).hexdigest()
        controller_launch_path, controller_launch_sha = _bound_receipt(
            spec,
            host / "controller_launch_receipt.json",
            "arc_agi3_controller_launch",
            controller_container_id=controller_container_id,
            controller_image_digest=(
                spec.proposer_transport.controller_image_digest
            ),
            egress_proxy_container_id=egress_proxy_container_id,
            egress_proxy_image_digest=(
                spec.proposer_transport
                .controller_egress_proxy_image_digest
            ),
            egress_policy_sha256=(
                spec.proposer_transport.controller_egress_policy_sha256
            ),
            launch_intent_sha256=launch_intent_sha256,
            credentials_in_argv_or_env=False,
            bridge_or_arena_mounts=0,
            authoritative_identity="controller_container_cgroup",
            containment_canary_anchor={
                "escrow_path":
                    prepared.controller_canary_escrow_path,
                "escrow_sha256":
                    prepared.controller_canary_escrow_sha256,
                "escrow_identity_sha256":
                    prepared.controller_canary_escrow_identity_sha256,
                "commitments_json":
                    prepared.controller_canary_commitments_json,
                "commitments_sha256":
                    prepared.controller_canary_commitments_sha256,
                "placement_descriptors_json":
                    prepared
                    .controller_canary_placement_descriptors_json,
                "placement_descriptors_sha256":
                    prepared
                    .controller_canary_placement_descriptors_sha256,
            },
        )
        supply_chain_manifest_sha256 = hashlib.sha256(
            (spec.attempt_id + ":controller-supply-chain").encode()
        ).hexdigest()
        guardian_path = host / "controller_guardian_start.json"
        _write_json(
            guardian_path,
            {
                "schema": 1,
                "kind": "arc_agi3_controller_guardian_start",
                "supply_chain_manifest_sha256":
                    supply_chain_manifest_sha256,
                "hard_safety_seconds": spec.hard_safety_seconds,
            },
        )
        guardian_sha = hashlib.sha256(
            guardian_path.read_bytes()
        ).hexdigest()
        substrate_identity_sha256 = hashlib.sha256(
            (spec.attempt_id + ":substrate").encode()
        ).hexdigest()
        substrate_path, substrate_sha = _bound_receipt(
            spec,
            host / "substrate_preflight_receipt.json",
            "contiguous_substrate_preflight",
            substrate_identity_sha256=substrate_identity_sha256,
            state_root=spec.app_server_state_dir,
            state_root_write_probe_status="PASS",
            state_database_initialized=True,
            path_alias_setup_status="PASS",
            preflight_stderr_bytes=0,
            preflight_stderr_sha256=hashlib.sha256(b"").hexdigest(),
            proposer_container_started=False,
            bridge_connected=False,
            thread_started=False,
            turn_started=False,
            controller_inspect_absent=True,
            controller_identity_query_empty=True,
            controller_no_descendants=True,
            egress_proxy_inspect_absent=True,
            egress_proxy_identity_query_empty=True,
            egress_proxy_no_descendants=True,
            status="PASS",
        )
        process_start = "start:" + spec.attempt_id
        runtime_path, runtime_sha = _bound_receipt(
            spec,
            host / "app_server_runtime_receipt.json",
            "contiguous_app_server_runtime",
            pid=4242,
            process_start=process_start,
            process_group_id=4242,
            state_root=spec.app_server_state_dir,
            neutral_cwd="/controller-neutral",
            neutral_host_staging_cwd=spec.neutral_host_cwd_path,
            thread_id=thread_id,
            turn_id=turn_id,
            thread_mode=spec.thread_mode,
            model=spec.proposer_transport.model,
            model_provider=spec.proposer_transport.model_provider,
            reasoning_effort=spec.effort,
            allow_provider_model_fallback=False,
            builtin_tool_names=[],
            dynamic_tool_namespace=
                spec.proposer_transport.dynamic_tool_namespace,
            dynamic_tool_names=list(
                spec.proposer_transport.dynamic_tool_names
            ),
            controller_method_policy={
                "preflight_requests": list(
                    spec.proposer_transport
                    .controller_preflight_request_allowlist
                ),
                "preflight_notifications": list(
                    spec.proposer_transport
                    .controller_preflight_notification_allowlist
                ),
                "turn_requests": list(
                    spec.proposer_transport
                    .controller_turn_request_allowlist
                ),
            },
            startup_probe_status="PASS",
            auth_probe_status="PASS",
            model_probe_status="PASS",
            bridge_probe_status="PASS",
            ambient_state_loaded=False,
            substrate_identity_sha256=substrate_identity_sha256,
            substrate_preflight_receipt_path=substrate_path,
            substrate_preflight_receipt_sha256=substrate_sha,
            state_root_write_probe_status="PASS",
            state_database_initialized=True,
            path_alias_setup_status="PASS",
        )
        chain_path, chain_sha = _bound_receipt(
            spec,
            host / "turn_start_transcript_chain_receipt.json",
            "contiguous_turn_start_transcript_chain",
            thread_id=thread_id,
            turn_id=turn_id,
            chain_head_sha256=transcript_head,
        )
        binding_path, binding_sha = _bound_receipt(
            spec,
            host / "turn_start_binding.json",
            "contiguous_turn_start_binding",
            thread_id=thread_id,
            turn_id=turn_id,
            thread_mode=spec.thread_mode,
            bridge_runtime_attestation_sha256=bridge_sha,
            app_server_runtime_receipt_sha256=runtime_sha,
            reasoning_effort=spec.effort,
            model=spec.proposer_transport.model,
            transcript_chain_sha256=transcript_head,
        )
        rebind_path = None
        rebind_sha = None
        if spec.thread_mode == "resume":
            assert spec.wip is not None
            rebind_path, rebind_sha = _bound_receipt(
                spec,
                host / "thread_rebinding_receipt.json",
                "contiguous_thread_rebinding",
                thread_id=spec.resume_thread_id,
                prior_thread_binding_sha256=
                    spec.resume_thread_binding_sha256,
                prior_transcript_chain_sha256=
                    spec.wip.transcript_chain_sha256,
                prior_app_server_state_tree_sha256=
                    spec.wip.app_server_state_tree_sha256,
                prior_app_server_state_dir=
                    spec.wip.app_server_state_dir,
                staged_app_server_state_dir=
                    spec.app_server_state_dir,
                staged_initial_state_tree_sha256=
                    spec.wip.app_server_state_tree_sha256,
                new_container_id=container_id,
                new_bridge_runtime_attestation_sha256=bridge_sha,
                old_bridge_revoked=True,
                no_binding_overlap=True,
            )
        launched = self.launches.setdefault(
            spec.attempt_id,
            R.BackendLaunch(
                backend_id="backend:" + spec.attempt_id,
                container_id=container_id,
                running_observation_sha256="4" * 64,
                substrate_identity_sha256=
                    substrate_identity_sha256,
                substrate_preflight_receipt_path=substrate_path,
                substrate_preflight_receipt_sha256=substrate_sha,
                bridge_runtime_attestation_path=bridge_path,
                bridge_runtime_attestation_sha256=bridge_sha,
                app_server_runtime_receipt_path=runtime_path,
                app_server_runtime_receipt_sha256=runtime_sha,
                app_server_pid=4242,
                app_server_process_start=process_start,
                app_server_process_group_id=4242,
                app_server_pid_is_diagnostic=True,
                process_identity_authority=(
                    "controller_container_cgroup"
                ),
                controller_container_id=controller_container_id,
                controller_image_digest=(
                    spec.proposer_transport.controller_image_digest
                ),
                egress_proxy_container_id=egress_proxy_container_id,
                egress_proxy_image_digest=(
                    spec.proposer_transport
                    .controller_egress_proxy_image_digest
                ),
                egress_policy_sha256=(
                    spec.proposer_transport
                    .controller_egress_policy_sha256
                ),
                controller_launch_intent_sha256=launch_intent_sha256,
                controller_launch_receipt_path=controller_launch_path,
                controller_launch_receipt_sha256=controller_launch_sha,
                controller_guardian_start_receipt_path=str(
                    guardian_path
                ),
                controller_guardian_start_receipt_sha256=guardian_sha,
                controller_supply_chain_manifest_sha256=(
                    supply_chain_manifest_sha256
                ),
                codex_thread_id=thread_id,
                codex_turn_id=turn_id,
                thread_binding_path=binding_path,
                thread_binding_sha256=binding_sha,
                transcript_chain_receipt_path=chain_path,
                transcript_chain_receipt_sha256=chain_sha,
                transcript_chain_sha256=transcript_head,
                thread_rebinding_receipt_path=rebind_path,
                thread_rebinding_receipt_sha256=rebind_sha,
            ),
        )
        if self.crash_after_first_launch and not self._did_launch_crash:
            self._did_launch_crash = True
            raise R.SimulatedCrash()
        return launched

    def poll(
        self,
        *,
        spec: R.AttemptSpec,
        prepared: R.BackendPreparation,
        launched: R.BackendLaunch,
        timeout_seconds: float,
    ) -> R.BackendPoll:
        assert prepared == self.preparations[spec.attempt_id]
        assert launched == self.launches[spec.attempt_id]
        self.poll_timeouts.append(timeout_seconds)
        if self.public_action_protocol_invalid:
            return R.BackendPoll(
                status="containment_fault",
                observation_sha256="5" * 64,
                exit_code=1,
            )
        if spec.attempt_id in self.results:
            return R.BackendPoll(
                status="exited",
                observation_sha256="5" * 64,
                exit_code=0,
            )
        return R.BackendPoll(
            status="running", observation_sha256="6" * 64
        )

    def collect(
        self,
        *,
        spec: R.AttemptSpec,
        prepared: R.BackendPreparation,
        launched: R.BackendLaunch,
        terminal: R.BackendPoll,
    ) -> R.BackendCollection:
        assert terminal.status != "running"
        host = Path(spec.host_transcript_path).parent
        transcript = Path(spec.host_transcript_path)
        if self.public_action_protocol_invalid:
            inventory = Transport.inventory_controller_state(
                Path(spec.app_server_state_dir)
            )
            state_scan = Taint.scan_controller_state(
                Path(spec.app_server_state_dir),
                inventory=inventory,
                canaries=self.controller_state_canaries,
            )
            _state_path, state_sha = _bound_receipt(
                spec,
                host / "controller_state_scan_receipt.json",
                "contiguous_controller_state_scan",
                scanner_source_sha256=Taint.source_sha256(),
                controller_state_scan=state_scan.as_receipt(),
            )
            partial_taint_path, partial_taint_sha = _bound_receipt(
                spec,
                host
                / "protocol_invalid_partial_taint_scan_receipt.json",
                "contiguous_protocol_invalid_partial_taint_scan",
                scanner_source_sha256=Taint.source_sha256(),
                records=[],
                hits=[],
                status="CLEAN",
                classification_authority=
                    "source_environment_taint_only",
            )
            partial_usage_path, partial_usage_sha = _bound_receipt(
                spec,
                host / "protocol_invalid_partial_usage_receipt.json",
                "contiguous_protocol_invalid_partial_usage",
                thread_id=launched.codex_thread_id,
                turn_id=launched.codex_turn_id,
                pre_provider_usage_window={"status": "authenticated"},
                token_usage_observations=[],
                observed_total_tokens=None,
                post_provider_usage_window=None,
                provider_usage_settlement=None,
                accounting_complete=False,
                unknown_token_usage=True,
                cost_settlement_authority=False,
            )
            retained_scan = Taint.scan_retained_canary_roots(
                {
                    "host_evidence": host,
                    "proposer_output": Path(spec.output_dir),
                },
                canaries=self.controller_state_canaries,
            )
            _retained_path, retained_sha = _bound_receipt(
                spec,
                Path(spec.generation_dir)
                / "retained_canary_scan_receipt.json",
                "contiguous_retained_canary_scan",
                scanner_source_sha256=Taint.source_sha256(),
                retained_canary_scan=retained_scan.as_receipt(),
                controller_state_scan_receipt_sha256=state_sha,
            )
            _absence_path, absence_sha = _bound_receipt(
                spec,
                host / "controller_absence_receipt.json",
                "contiguous_controller_absence",
                controller_container_id=launched.controller_container_id,
                egress_proxy_container_id=
                    launched.egress_proxy_container_id,
                all_exact_roles_absent=True,
            )
            receipt_path, receipt_sha = _bound_receipt(
                spec,
                host
                / "arena_public_action_protocol_invalid_receipt.json",
                "contiguous_arena_public_action_protocol_invalid",
                protocol_violation={
                    "schema": "arc-agi3-arena-rpc/v1",
                    "kind": "rpc",
                    "phase": "rejected",
                    "seq": 1,
                    "op": "step",
                    "ok": False,
                    "error": "coordinate action is outside public grammar",
                },
                protocol_violation_sha256=R.Scheduler.sha256_json({
                    "schema": "arc-agi3-arena-rpc/v1",
                    "kind": "rpc",
                    "phase": "rejected",
                    "seq": 1,
                    "op": "step",
                    "ok": False,
                    "error": "coordinate action is outside public grammar",
                }),
                proposer_containment_sha256="4" * 64,
                controller_absence_receipt_sha256=absence_sha,
                controller_state_scan_receipt_path=_state_path,
                controller_state_scan_receipt_sha256=state_sha,
                retained_canary_scan_receipt_path=_retained_path,
                retained_canary_scan_receipt_sha256=retained_sha,
                partial_taint_scan_receipt_path=partial_taint_path,
                partial_taint_scan_receipt_sha256=partial_taint_sha,
                partial_taint_status="CLEAN",
                partial_usage_receipt_path=partial_usage_path,
                partial_usage_receipt_sha256=partial_usage_sha,
                usage_accounting_complete=False,
                cost_used=0.0,
                cost_authority="explicit_unlimited_no_local_charge",
                candidate_admissible=False,
                wip_admissible=False,
                public_observation_admissible=False,
                sidecar_request_admissible=False,
                supervisory_handoff_admissible=False,
                promotion_admissible=False,
                restart_restoration_admissible=False,
                status="PROTOCOL_INVALID",
            )
            self.protocol_invalid_scans[spec.attempt_id] = (
                state_sha, retained_sha
            )
            raise R.BackendPublicActionProtocolInvalidError(
                receipt_path=receipt_path,
                receipt_sha256=receipt_sha,
                controller_state_scan_receipt_path=_state_path,
                controller_state_scan_receipt_sha256=state_sha,
                retained_canary_scan_receipt_path=_retained_path,
                retained_canary_scan_receipt_sha256=retained_sha,
                partial_taint_scan_receipt_path=partial_taint_path,
                partial_taint_scan_receipt_sha256=partial_taint_sha,
                partial_usage_receipt_path=partial_usage_path,
                partial_usage_receipt_sha256=partial_usage_sha,
                cost_used=0.0,
            )
        native_public_observation_receipt_sha256s: tuple[
            str, ...
        ] = ()
        public_root = host / "public_observations"
        public_root.mkdir(mode=0o700, exist_ok=True)
        if self.results[spec.attempt_id].kind in {
            "clean_no_progress",
            "candidate",
        }:
            public_action_basis = {
                "schema": 1,
                "kind": ArenaRpc.PUBLIC_ACTION_BASIS_KIND,
                "operation_index": 0,
                "previous_public_action_basis_sha256":
                    ArenaRpc.PUBLIC_ACTION_BASIS_GENESIS_SHA256,
                "operation": {"op": "open"},
            }
            public_response_signature = {
                "schema": 1,
                "kind": ArenaRpc.PUBLIC_RESPONSE_SIGNATURE_KIND,
                "operation_index": 0,
                "result": {
                    "binding_sha256": "d" * 64,
                    "snapshot": {
                        "frame": [[0]],
                        "actions": [1],
                        "levels_completed": spec.target_level - 1,
                        "terminal": False,
                    },
                },
            }
            public_receipt = {
                "schema": 1,
                "kind": ArenaRpc.PUBLIC_OBSERVATION_RECEIPT_KIND,
                "game": spec.game,
                "frontier_sha256": spec.frontier_sha256,
                "parent_checkpoint_sha256":
                    spec.parent_checkpoint_sha256,
                "public_action_basis": public_action_basis,
                "public_action_basis_sha256": hashlib.sha256(
                    R._canonical_json(public_action_basis)
                ).hexdigest(),
                "public_response_signature":
                    public_response_signature,
                "public_response_signature_sha256": hashlib.sha256(
                    R._canonical_json(public_response_signature)
                ).hexdigest(),
            }
            public_receipt_raw = (
                ArenaRpc.public_observation_receipt_bytes(
                    public_receipt
                )
            )
            public_receipt_sha256 = hashlib.sha256(
                public_receipt_raw
            ).hexdigest()
            public_path = (
                public_root / f"{public_receipt_sha256}.json"
            )
            if public_path.exists():
                assert public_path.read_bytes() == public_receipt_raw
            else:
                public_path.write_bytes(public_receipt_raw)
            native_public_observation_receipt_sha256s = (
                public_receipt_sha256,
            )
        if not transcript.exists():
            _write_json(
                transcript,
                {
                    "schema": ArenaRpc.RPC_SCHEMA,
                    "kind": "rpc",
                    "phase": "applied",
                    "seq": 0,
                    "op": "open",
                    "request_sha256": "e" * 64,
                    "ok": True,
                    "elapsed_ns": 1,
                    **(
                        {
                            "public_observation_receipt_sha256":
                                public_receipt_sha256,
                            "public_action_basis_sha256":
                                public_receipt[
                                    "public_action_basis_sha256"
                                ],
                            "public_response_signature_sha256":
                                public_receipt[
                                    "public_response_signature_sha256"
                                ],
                        }
                        if native_public_observation_receipt_sha256s
                        else {}
                    ),
                },
            )
        worker_outcome = Path(spec.output_dir) / R.WORKER_OUTCOME_NAME
        if not worker_outcome.exists():
            worker_outcome.write_text(
                json.dumps(
                    {
                        "attempt_id": spec.attempt_id,
                        "kind": self.results[spec.attempt_id].kind,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        stdout_path = host / "container.stdout"
        stderr_path = host / "container.stderr"
        _write_json(
            stdout_path,
            {
                "schema": 1,
                "kind": "arc_agi3_contiguous_proposer_worker_terminal",
                "exit_code": 0,
                "child_stdio_captured": True,
            },
        )
        _write_json(
            stderr_path,
            {"schema": 1, "kind": "container_stderr", "status": "clean"},
        )
        app_transcript = Path(spec.app_server_transcript_path)
        app_transcript_sha = hashlib.sha256(
            app_transcript.read_bytes()
        ).hexdigest()
        result = self.results[spec.attempt_id]
        provider_outcome = (
            "provider_failure"
            if result.kind == "infrastructure"
            else "completed"
        )
        turn_status = (
            "failed"
            if result.kind == "infrastructure"
            else "completed"
        )
        usage_observations = [
            {
                "threadId": launched.codex_thread_id,
                "turnId": launched.codex_turn_id,
                "total": {
                    "inputTokens": 1,
                    "cachedInputTokens": 0,
                    "outputTokens": 1,
                    "reasoningOutputTokens": 0,
                    "totalTokens": 2,
                },
            }
        ]
        usage_path, usage_sha = _bound_receipt(
            spec,
            host / "token_usage_receipt.json",
            "contiguous_token_usage",
            thread_id=launched.codex_thread_id,
            turn_id=launched.codex_turn_id,
            final_event_observed=True,
            wrong_identity_events=0,
            duplicate_events=0,
            hard_safety_seconds=spec.hard_safety_seconds,
            max_auth_refreshes=spec.max_auth_refreshes,
            auth_refresh_count=0,
            redacted_auth_refresh_response_sha256=[],
            credential_sentinel_scan_passed=True,
            post_turn_event_count=0,
            stdout_bytes=0,
            stderr_bytes=0,
            pipes_drained_to_eof=True,
            observations=usage_observations,
        )
        credits = (
            {
                "hasCredits": True,
                "unlimited": True,
                "balance": None,
            }
            if spec.cost_limit_remaining is None
            else {
                "hasCredits": True,
                "unlimited": False,
                "balance": float(spec.cost_limit_remaining),
            }
        )
        provider_snapshot = {
            "rateLimitsByLimitId": {
                "codex": {
                    "planType": "team",
                    "credits": credits,
                    "spendControlReached": False,
                }
            }
        }
        pre_usage_window = Transport.normalize_provider_usage_window(
            provider_snapshot,
            phase="preflight",
            observation_sequence=1,
            authenticated_response_sha256="a" * 64,
            transcript_chain_sha256=launched.transcript_chain_sha256,
        )
        post_usage_window = Transport.normalize_provider_usage_window(
            provider_snapshot,
            phase="postflight",
            observation_sequence=2,
            authenticated_response_sha256="b" * 64,
            transcript_chain_sha256=launched.transcript_chain_sha256,
        )
        provider_settlement = Transport.settle_provider_usage(
            pre_usage_window,
            post_usage_window,
            token_usage_observations=usage_observations,
        )
        provider_path, provider_sha = _bound_receipt(
            spec,
            host / "provider_usage_receipt.json",
            "contiguous_provider_usage",
            thread_id=launched.codex_thread_id,
            turn_id=launched.codex_turn_id,
            token_usage_observations=usage_observations,
            pre_provider_usage_window=asdict(pre_usage_window),
            post_provider_usage_window=asdict(post_usage_window),
            provider_usage_settlement=asdict(provider_settlement),
        )
        result = replace(
            result,
            cost_used=provider_settlement.charge,
        )
        self.results[spec.attempt_id] = result
        final_chain_path, final_chain_sha = _bound_receipt(
            spec,
            host / "final_transcript_chain_receipt.json",
            "contiguous_final_transcript_chain",
            thread_id=launched.codex_thread_id,
            turn_id=launched.codex_turn_id,
            chain_head_sha256=launched.transcript_chain_sha256,
            raw_transcript_sha256=app_transcript_sha,
            event_count=len(
                app_transcript.read_text(
                    encoding="utf-8"
                ).splitlines()
            ),
        )
        output_tree_sha = Contract._tree_hash(Path(spec.output_dir))
        candidate_sha = (
            result.candidate.candidate_manifest_sha256
            if result.candidate is not None
            else None
        )
        target_boundary = (
            _fake_target_boundary(spec)
            if result.kind == "candidate"
            else (None, None, None, None)
        )
        export_path, export_sha = _bound_receipt(
            spec,
            host / "bridge_export_receipt.json",
            "contiguous_bridge_export",
            container_id=launched.container_id,
            bridge_runtime_attestation_sha256=
                launched.bridge_runtime_attestation_sha256,
            output_tree_sha256=output_tree_sha,
            model_final_text_eligible=False,
            outcome=(
                "candidate" if result.kind == "candidate" else result.kind
            ),
            host_blocker_code=(
                result.blocker.code
                if result.blocker is not None
                else None
            ),
            host_blocker_receipt_sha256=(
                result.blocker.receipt_sha256
                if result.blocker is not None
                else None
            ),
            candidate_manifest_sha256=candidate_sha,
            target_boundary_receipt_sha256=target_boundary[1],
            target_boundary_sha256=target_boundary[2],
            target_boundary_workspace_tree_sha256=target_boundary[3],
        )
        state_tree_sha = Contract._tree_hash(
            Path(spec.app_server_state_dir)
        )
        model_final_sha = hashlib.sha256(b"").hexdigest()
        controller_inventory = Transport.inventory_controller_state(
            Path(spec.app_server_state_dir)
        )
        secret_path, secret_sha = _bound_receipt(
            spec,
            host / "secret_scan_receipt.json",
            "contiguous_secret_scan",
            scanned_sha256={
                "app_server_transcript": app_transcript_sha,
                "backend_transcript": hashlib.sha256(
                    transcript.read_bytes()
                ).hexdigest(),
                "container_stderr": hashlib.sha256(
                    stderr_path.read_bytes()
                ).hexdigest(),
                "container_stdout": hashlib.sha256(
                    stdout_path.read_bytes()
                ).hexdigest(),
                "output_tree": output_tree_sha,
                "app_server_state_tree": state_tree_sha,
            },
            controller_state_inventory=controller_inventory.as_receipt(),
            secret_occurrences=0,
            credential_generations_scanned=1,
            controller_terminal_scan_passed=True,
            status="PASS",
        )
        controller_state_scan = Taint.scan_controller_state(
            Path(spec.app_server_state_dir),
            inventory=controller_inventory,
            canaries=self.controller_state_canaries,
        )
        state_scan_path, state_scan_sha = _bound_receipt(
            spec,
            host / "controller_state_scan_receipt.json",
            "contiguous_controller_state_scan",
            scanner_source_sha256=Taint.source_sha256(),
            controller_state_scan=controller_state_scan.as_receipt(),
        )

        scan_policy, _prompt = _app_scan_policy(spec)
        scan_records = [
            Taint.scan_evidence(
                app_transcript,
                evidence_kind="app_server_jsonl",
                app_server_policy=scan_policy,
            ),
            Taint.scan_evidence(
                transcript, evidence_kind="backend_jsonl"
            ),
            Taint.scan_evidence(
                stdout_path, evidence_kind="container_stdout"
            ),
            Taint.scan_evidence(
                stderr_path, evidence_kind="container_stderr"
            ),
            *[
                Taint.scan_evidence(
                    entry, evidence_kind="candidate_output"
                )
                for entry in sorted(
                    entry
                    for entry in Path(spec.output_dir).rglob("*")
                    if entry.is_file()
                )
            ],
            *controller_state_scan.records,
        ]
        hits = sorted(
            {
                hit
                for record in scan_records
                for hit in record.hits
            }
        )
        taint_path, taint_sha = _bound_receipt(
            spec,
            host / "taint_scan_receipt.json",
            "contiguous_taint_scan",
            scanner_source_sha256=Taint.source_sha256(),
            records=[asdict(record) for record in scan_records],
            hits=hits,
            status="TAINT" if hits else "CLEAN",
        )
        # The production controller is already absent when collection
        # succeeds.  Seed the terminal absence/reconciliation receipts before
        # the retained-evidence inventory so replay observes the same closed
        # byte set after teardown.
        absence_path, absence_sha256 = _bound_receipt(
            spec,
            host / "controller_absence_receipt.json",
            "contiguous_controller_absence",
            controller_container_id=launched.controller_container_id,
            egress_proxy_container_id=
                launched.egress_proxy_container_id,
            all_exact_roles_absent=True,
        )
        _bound_receipt(
            spec,
            host / "probe_reconciliation_teardown.json",
            "contiguous_probe_reconciliation_teardown",
            controller_absence_receipt_path=absence_path,
            controller_absence_receipt_sha256=absence_sha256,
            all_exact_roles_absent=True,
        )
        retained_canary_scan = Taint.scan_retained_canary_roots(
            {
                "host_evidence": host,
                "proposer_output": Path(spec.output_dir),
            },
            canaries=self.controller_state_canaries,
        )
        retained_path, retained_sha = _bound_receipt(
            spec,
            Path(spec.generation_dir)
            / "retained_canary_scan_receipt.json",
            "contiguous_retained_canary_scan",
            scanner_source_sha256=Taint.source_sha256(),
            retained_canary_scan=retained_canary_scan.as_receipt(),
            controller_state_scan_receipt_sha256=state_scan_sha,
        )
        final_binding_path, final_binding_sha = _bound_receipt(
            spec,
            Path(spec.generation_dir) / "final_thread_binding.json",
            "contiguous_final_thread_binding",
            thread_id=launched.codex_thread_id,
            turn_id=launched.codex_turn_id,
            thread_mode=spec.thread_mode,
            turn_status=turn_status,
            provider_outcome=provider_outcome,
            transcript_chain_sha256=
                launched.transcript_chain_sha256,
            final_transcript_chain_receipt_sha256=final_chain_sha,
            token_usage_receipt_sha256=usage_sha,
            provider_usage_receipt_sha256=provider_sha,
            bridge_export_receipt_sha256=export_sha,
            host_blocker_code=(
                result.blocker.code
                if result.blocker is not None
                else None
            ),
            host_blocker_receipt_sha256=(
                result.blocker.receipt_sha256
                if result.blocker is not None
                else None
            ),
            secret_scan_receipt_sha256=secret_sha,
            app_server_state_tree_sha256=state_tree_sha,
            controller_state_inventory_sha256=(
                controller_inventory.inventory_sha256
            ),
            controller_state_scan_receipt_sha256=state_scan_sha,
            retained_canary_scan_receipt_sha256=retained_sha,
            taint_scan_receipt_sha256=taint_sha,
            wip_export_receipt_sha256=None,
            target_boundary_receipt_sha256=target_boundary[1],
            target_boundary_sha256=target_boundary[2],
            target_boundary_workspace_tree_sha256=target_boundary[3],
            model_final_text_sha256=model_final_sha,
            model_final_text_eligible=False,
        )
        collection = R.BackendCollection(
            result=result,
            worker_outcome_sha256=hashlib.sha256(
                worker_outcome.read_bytes()
            ).hexdigest(),
            output_tree_sha256=output_tree_sha,
            host_transcript_path=str(transcript),
            host_transcript_sha256=hashlib.sha256(
                transcript.read_bytes()
            ).hexdigest(),
            native_public_observation_receipt_sha256s=(
                native_public_observation_receipt_sha256s
            ),
            container_stdout_path=str(stdout_path),
            container_stdout_sha256=hashlib.sha256(
                stdout_path.read_bytes()
            ).hexdigest(),
            container_stderr_path=str(stderr_path),
            container_stderr_sha256=hashlib.sha256(
                stderr_path.read_bytes()
            ).hexdigest(),
            app_server_transcript_path=str(app_transcript),
            app_server_transcript_sha256=app_transcript_sha,
            codex_thread_id=launched.codex_thread_id,
            codex_turn_id=launched.codex_turn_id,
            structured_turn_status=turn_status,
            structured_provider_outcome=provider_outcome,
            token_usage_receipt_path=usage_path,
            token_usage_receipt_sha256=usage_sha,
            provider_usage_receipt_path=provider_path,
            provider_usage_receipt_sha256=provider_sha,
            final_transcript_chain_receipt_path=final_chain_path,
            final_transcript_chain_receipt_sha256=final_chain_sha,
            final_transcript_chain_sha256=
                launched.transcript_chain_sha256,
            final_thread_binding_path=final_binding_path,
            final_thread_binding_sha256=final_binding_sha,
            bridge_export_receipt_path=export_path,
            bridge_export_receipt_sha256=export_sha,
            secret_scan_receipt_path=secret_path,
            secret_scan_receipt_sha256=secret_sha,
            controller_state_scan_receipt_path=state_scan_path,
            controller_state_scan_receipt_sha256=state_scan_sha,
            controller_state_inventory_sha256=(
                controller_inventory.inventory_sha256
            ),
            retained_canary_scan_receipt_path=retained_path,
            retained_canary_scan_receipt_sha256=retained_sha,
            supervisory_native_reproduction_receipt_path=None,
            supervisory_native_reproduction_receipt_sha256=None,
            target_boundary_receipt_path=target_boundary[0],
            target_boundary_receipt_sha256=target_boundary[1],
            target_boundary_sha256=target_boundary[2],
            target_boundary_workspace_tree_sha256=target_boundary[3],
            taint_scan_receipt_path=taint_path,
            taint_scan_receipt_sha256=taint_sha,
            app_server_state_tree_sha256=state_tree_sha,
            model_final_text_sha256=model_final_sha,
        )
        if self.crash_after_first_collect and not self._did_collect_crash:
            self._did_collect_crash = True
            raise R.SimulatedCrash()
        self.collections[spec.attempt_id] = collection
        return collection

    def teardown(
        self,
        *,
        spec: R.AttemptSpec,
        prepared: R.BackendPreparation,
        launched: R.BackendLaunch,
        cause: str,
    ) -> R.BackendTeardownProof:
        self.teardown_calls.append(spec.attempt_id)
        for path in (
            Path(spec.arena_socket_path),
            Path(spec.arena_token_file_path),
            Path(spec.bridge_socket_path),
            Path(spec.bridge_token_file_path),
        ):
            if path.exists() or path.is_symlink():
                path.unlink()
        collection = self.collections.get(spec.attempt_id)
        if collection is None:
            state_scan_sha, retained_scan_sha = (
                self.protocol_invalid_scans[spec.attempt_id]
            )
        else:
            state_scan_sha = (
                collection.controller_state_scan_receipt_sha256
            )
            retained_scan_sha = (
                collection.retained_canary_scan_receipt_sha256
            )
        commitments = [
            item.commitment()
            for item in sorted(
                self.controller_state_canaries,
                key=lambda item: item.category,
            )
        ]
        reveal_root = (
            Path(spec.generation_dir).parent.parent
            / "containment_canary_reveals"
        )
        reveal_root.mkdir(mode=0o700, exist_ok=True)
        reveal_path = reveal_root / f"{spec.generation_id}.json"
        absence_path, absence_sha256 = _bound_receipt(
            spec,
            Path(spec.host_transcript_path).parent
            / "controller_absence_receipt.json",
            "contiguous_controller_absence",
            controller_container_id=launched.controller_container_id,
            egress_proxy_container_id=
                launched.egress_proxy_container_id,
            all_exact_roles_absent=True,
        )
        reconciliation_path, reconciliation_sha256 = _bound_receipt(
            spec,
            Path(spec.host_transcript_path).parent
            / "probe_reconciliation_teardown.json",
            "contiguous_probe_reconciliation_teardown",
            controller_absence_receipt_path=absence_path,
            controller_absence_receipt_sha256=absence_sha256,
            all_exact_roles_absent=True,
        )
        teardown_observation_sha256 = hashlib.sha256(
            R._canonical_json(
                {
                    "container_proof_sha256": "8" * 64,
                    "controller_absence_receipt_sha256":
                        absence_sha256,
                    "probe_reconciliation_receipt_sha256":
                        reconciliation_sha256,
                    "controller_container_id":
                        launched.controller_container_id,
                    "egress_proxy_container_id":
                        launched.egress_proxy_container_id,
                    "all_exact_roles_absent": True,
                }
            )
        ).hexdigest()
        reveal_value = {
            "schema": 1,
            "kind": "contiguous_controller_canary_reveal",
            "campaign_id": spec.campaign_id,
            "generation_id": spec.generation_id,
            "attempt_id": spec.attempt_id,
            "attempt_spec_sha256":
                R.proposer_attempt_binding_sha256(spec),
            "canary_escrow_sha256":
                prepared.controller_canary_escrow_sha256,
            "canary_escrow_identity_sha256":
                prepared.controller_canary_escrow_identity_sha256,
            "canary_commitments_sha256":
                prepared.controller_canary_commitments_sha256,
            "canary_placement_descriptors_sha256":
                prepared
                .controller_canary_placement_descriptors_sha256,
            "controller_container_id":
                launched.controller_container_id,
            "egress_proxy_container_id":
                launched.egress_proxy_container_id,
            "controller_absence_receipt_sha256":
                absence_sha256,
            "controller_state_scan_receipt_sha256":
                state_scan_sha,
            "retained_canary_scan_receipt_sha256":
                retained_scan_sha,
            "canary_commitments": commitments,
            "reveal": Taint.build_live_canary_reveal(
                self.controller_state_canaries
            ),
            "teardown_observation_sha256":
                teardown_observation_sha256,
        }
        if reveal_path.exists():
            assert json.loads(
                reveal_path.read_text(encoding="utf-8")
            ) == reveal_value
        else:
            _write_json(reveal_path, reveal_value)
            os.chmod(reveal_path, 0o400)
        reveal_sha = hashlib.sha256(
            reveal_path.read_bytes()
        ).hexdigest()
        arena_attachment_status = (
            "CLEAN_EOF"
            if cause == "normal_exit"
            else "ABORTED_CONTAINMENT"
        )
        arena_attachment = (
            {
                "schema": 1,
                "kind": "arc_agi3_arena_volume_attachment",
                "status": arena_attachment_status,
            }
            if cause == "normal_exit"
            else None
        )
        arena_attachment_sha256 = (
            hashlib.sha256(
                R._canonical_json(arena_attachment)
            ).hexdigest()
            if arena_attachment is not None
            else None
        )
        arena_teardown_path = (
            Path(spec.host_transcript_path).parent
            / "arena_volume_teardown.json"
        )
        arena_teardown_value = {
            "schema": 1,
            "kind": "arc_agi3_arena_volume_teardown",
            "campaign_id": spec.campaign_id,
            "generation_id": spec.generation_id,
            "attempt_id": spec.attempt_id,
            "transport": R.ARENA_VOLUME_TRANSPORT,
            "preparation_receipt_sha256":
                prepared.arena_relay_preparation_receipt_sha256,
            "relay_container_id":
                prepared.arena_relay_container_id,
            "volume_name": prepared.arena_volume_name,
            "attachment_status": arena_attachment_status,
            "attachment_receipt": arena_attachment,
            "attachment_receipt_sha256":
                arena_attachment_sha256,
            "relay_inspect_absent": True,
            "relay_top_absent": True,
            "relay_identity_query_empty": True,
            "volume_inspect_absent": True,
            "volume_identity_query_empty": True,
        }
        if arena_teardown_path.exists():
            assert json.loads(
                arena_teardown_path.read_text(encoding="utf-8")
            ) == arena_teardown_value
        else:
            _write_json(arena_teardown_path, arena_teardown_value)
            os.chmod(arena_teardown_path, 0o600)
        arena_teardown_sha256 = hashlib.sha256(
            arena_teardown_path.read_bytes()
        ).hexdigest()
        return R.BackendTeardownProof(
            container_id=launched.container_id,
            cause=cause,
            proof_sha256="8" * 64,
            container_inspect_absent=not self.bad_teardown,
            container_top_absent=True,
            identity_query_empty=True,
            no_descendants=True,
            app_server_process_absent=True,
            app_server_process_group_absent=True,
            bridge_socket_absent=True,
            bridge_token_absent=True,
            app_server_control_absent=True,
            arena_relay_container_id=(
                prepared.arena_relay_container_id
            ),
            arena_volume_name=prepared.arena_volume_name,
            arena_relay_inspect_absent=True,
            arena_relay_top_absent=True,
            arena_relay_identity_query_empty=True,
            arena_volume_inspect_absent=True,
            arena_volume_identity_query_empty=True,
            arena_relay_attachment_status=arena_attachment_status,
            arena_relay_teardown_receipt_path=str(
                arena_teardown_path
            ),
            arena_relay_teardown_receipt_sha256=(
                arena_teardown_sha256
            ),
            process_identity_authority="controller_container_cgroup",
            controller_container_id=launched.controller_container_id,
            egress_proxy_container_id=launched.egress_proxy_container_id,
            controller_inspect_absent=True,
            controller_identity_query_empty=True,
            controller_top_absent=True,
            controller_no_descendants=True,
            egress_proxy_inspect_absent=True,
            egress_proxy_identity_query_empty=True,
            egress_proxy_top_absent=True,
            egress_proxy_no_descendants=True,
            controller_absence_receipt_sha256=absence_sha256,
            canary_reveal_path=str(reveal_path),
            canary_reveal_sha256=reveal_sha,
        )

    def emergency_contain(
        self,
        *,
        spec: R.AttemptSpec,
        prepared: R.BackendPreparation | None,
        launched: R.BackendLaunch | None,
        prior_phase: str,
        reason: str,
    ) -> R.BackendEmergencyContainment:
        del prepared
        self.emergency_containment_calls.append(
            (spec.attempt_id, prior_phase)
        )
        for endpoint in (
            Path(spec.arena_socket_path),
            Path(spec.arena_token_file_path),
            Path(spec.bridge_socket_path),
            Path(spec.bridge_token_file_path),
        ):
            if endpoint.exists() or endpoint.is_symlink():
                endpoint.unlink()
        launched_container_id = (
            launched.container_id if launched is not None else None
        )
        path, digest = _bound_receipt(
            spec,
            Path(spec.host_transcript_path).parent
            / "storage_emergency_containment.json",
            "contiguous_storage_emergency_containment",
            prior_phase=prior_phase,
            reason=reason,
            launched_container_id=launched_container_id,
            attempt_container_absent=True,
            controller_roles_absent=True,
            arena_resources_absent=True,
            rpc_endpoints_absent=True,
            workspace_probe_containers_absent=True,
            host_process_groups_absent=True,
            containment_canaries_absent=True,
            no_descendants=True,
            solver_authority=False,
            wip_authority=False,
            cost_authority=False,
            promotion_authority=False,
            status="QUIESCED",
        )
        return R.BackendEmergencyContainment(
            containment_receipt_path=path,
            containment_receipt_sha256=digest,
            launched_container_id=launched_container_id,
            attempt_container_absent=True,
            controller_roles_absent=True,
            arena_resources_absent=True,
            rpc_endpoints_absent=True,
            workspace_probe_containers_absent=True,
            host_process_groups_absent=True,
            containment_canaries_absent=True,
            no_descendants=True,
        )

    def finish(self, attempt_id: str, result: R.AttemptResult) -> None:
        self.results[attempt_id] = result


def _typed_substrate_failure(
    spec: R.AttemptSpec,
    *,
    failure_class: str = "DETERMINISTIC_CONFIGURATION",
    failure_code: str = "controller_state_root_permission",
) -> R.BackendSubstratePreflightError:
    host = Path(spec.host_transcript_path).parent
    preflight_root = host / "substrate_preflight"
    assert not preflight_root.exists()
    state_inventory = R.Transport.inventory_controller_state(
        Path(spec.app_server_state_dir),
        sentinels=tuple(
            item.value for item in TEST_CONTROLLER_CANARIES
        ),
    )
    substrate_identity = hashlib.sha256(
        (spec.attempt_id + ":failed-substrate").encode()
    ).hexdigest()
    intent_path, intent_sha = _bound_receipt(
        spec,
        host / "substrate_preflight_intent.json",
        "contiguous_substrate_preflight_intent",
        substrate_identity_sha256=substrate_identity,
        preflight_root=str(preflight_root),
        state_root=spec.app_server_state_dir,
        initial_state_tree_sha256=state_inventory.tree_sha256,
        initial_state_inventory_sha256=
            state_inventory.inventory_sha256,
        prior_clean_wip_tree_sha256=None,
        proposer_container_started=False,
        bridge_connected=False,
        thread_started=False,
        turn_started=False,
        status="PENDING",
    )
    partial_path, partial_sha = _bound_receipt(
        spec,
        host / "substrate_preflight_partial_scan_receipt.json",
        "contiguous_substrate_preflight_partial_scan",
        substrate_preflight_intent_sha256=intent_sha,
        substrate_identity_sha256=substrate_identity,
        failure_stage="controller-start-and-initialize",
        error_type="PermissionError",
        failure_class=failure_class,
        failure_code=failure_code,
        scan_completed_before_purge=True,
        status="COMPLETE",
    )
    purge_path, purge_sha = _bound_receipt(
        spec,
        host / "substrate_preflight_purge_receipt.json",
        "contiguous_substrate_preflight_purge",
        substrate_preflight_intent_sha256=intent_sha,
        substrate_identity_sha256=substrate_identity,
        partial_scan_receipt_path=partial_path,
        partial_scan_receipt_sha256=partial_sha,
        post_purge_state_tree_sha256=state_inventory.tree_sha256,
        state_root_empty=True,
        preflight_root_absent=True,
        prior_clean_wip_tree_sha256=None,
        post_purge_clean_wip_tree_sha256=None,
        candidate_authority=False,
        wip_authority=False,
        promotion_authority=False,
        status="PASS",
    )
    tombstone = host / "backend_launch_failure.json"
    _write_json(
        tombstone,
        {
            "schema": 1,
            "kind": "contiguous_backend_launch_failure",
            "attempt_id": spec.attempt_id,
        },
    )
    tombstone_sha = hashlib.sha256(
        tombstone.read_bytes()
    ).hexdigest()
    failure_path, failure_sha = _bound_receipt(
        spec,
        host / "substrate_preflight_failure_receipt.json",
        "contiguous_substrate_preflight_failure",
        substrate_identity_sha256=substrate_identity,
        substrate_preflight_intent_path=intent_path,
        substrate_preflight_intent_sha256=intent_sha,
        preflight_root=str(preflight_root),
        state_root=spec.app_server_state_dir,
        failure_stage="controller-start-and-initialize",
        error_type="PermissionError",
        failure_class=failure_class,
        failure_code=failure_code,
        partial_scan_receipt_path=partial_path,
        partial_scan_receipt_sha256=partial_sha,
        purge_receipt_path=purge_path,
        purge_receipt_sha256=purge_sha,
        post_failure_state_tree_sha256=state_inventory.tree_sha256,
        state_root_empty=True,
        preflight_root_absent=True,
        prior_clean_wip_tree_sha256=None,
        post_purge_clean_wip_tree_sha256=None,
        backend_launch_failure_tombstone_path=str(tombstone),
        backend_launch_failure_tombstone_sha256=tombstone_sha,
        proposer_container_started=False,
        bridge_connected=False,
        thread_started=False,
        turn_started=False,
        candidate_authority=False,
        wip_authority=False,
        promotion_authority=False,
        cost_used=0.0,
        status="INFRASTRUCTURE",
    )
    return R.BackendSubstratePreflightError(
        substrate_identity_sha256=substrate_identity,
        failure_receipt_path=failure_path,
        failure_receipt_sha256=failure_sha,
    )


def _typed_preparation_quarantine(
    spec: R.AttemptSpec,
) -> R.BackendPreparationQuarantinedError:
    host = Path(spec.host_transcript_path).parent
    closure_root = host / "compatibility_arena_closure"
    staging = host / R.CompatibilityClosure._staging_name(
        closure_root
    )
    staging.mkdir(mode=0o700)
    partial = staging / R.CompatibilityClosure.CLIENT_NAME
    partial.write_bytes(b"retained partial closure bytes\n")
    partial.chmod(0o400)
    staging_observation = (
        R.CompatibilityClosure.observe_quarantined_staging(
            closure_root
        )
    )
    receipt_path, receipt_sha256 = _bound_receipt(
        spec,
        host / "compatibility_preparation_quarantine.json",
        "contiguous_compatibility_preparation_quarantine",
        failure_stage="compatibility_closure_prepare",
        failure_type="CompatibilityStagingAmbiguityError",
        closure_root=str(closure_root),
        closure_root_present=False,
        staging_observation=staging_observation,
        staging_observation_sha256=(
            staging_observation["observation_sha256"]
        ),
        container_identity_query_empty=True,
        arena_relay_absent=True,
        rpc_endpoints_absent=True,
        proposer_container_started=False,
        proposer_turn_started=False,
        candidate_authority=False,
        wip_authority=False,
        cost_authority=False,
        promotion_authority=False,
        old_evidence_reuse_authority=False,
        fresh_attempt_generation_required=True,
        status="QUARANTINED",
    )
    return R.BackendPreparationQuarantinedError(
        quarantine_receipt_path=receipt_path,
        quarantine_receipt_sha256=receipt_sha256,
    )


class QuarantinedPreparationBackend(FakeBackend):
    def __init__(self, quarantine_count: int):
        super().__init__(candidate_result)
        self.quarantine_count = quarantine_count
        self.quarantined_specs: list[R.AttemptSpec] = []

    def prepare(self, spec: R.AttemptSpec) -> R.BackendPreparation:
        if (
            len(self.quarantined_specs) < self.quarantine_count
            and spec.attempt_id not in self.specs
        ):
            self.prepare_calls.append(spec.attempt_id)
            self.specs[spec.attempt_id] = spec
            self.quarantined_specs.append(spec)
            raise _typed_preparation_quarantine(spec)
        return super().prepare(spec)


class SubstrateFailingBackend(FakeBackend):
    def launch(
        self,
        spec: R.AttemptSpec,
        prepared: R.BackendPreparation,
    ) -> R.BackendLaunch:
        del prepared
        self.launch_calls.append(spec.attempt_id)
        raise _typed_substrate_failure(spec)


def _test_substrate_health_probe(
    *,
    spec: R.AttemptSpec,
    authorization_id: str,
    authorization_receipt_sha256: str,
    probe_index: int,
    failed_substrate_identity_sha256: str,
    incident_failure_receipt_sha256: str,
    status: str,
) -> R.BackendSubstrateHealthProbe:
    root = (
        Path(spec.host_transcript_path).parent
        / "substrate_health_reprobes"
        / authorization_id
    )
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    health_state_root = root / "state" / "codex_home"
    health_runtime_root = root / "substrate_preflight"
    scan_path, scan_sha = _bound_receipt(
        spec,
        root / "scan.json",
        "contiguous_substrate_health_reprobe_scan",
        authorization_id=authorization_id,
        probe_index=probe_index,
        source_scan_receipt_path=None,
        source_scan_receipt_sha256=None,
        state_inventory_before_purge={
            "tree_sha256": "1" * 64,
            "inventory_sha256": "2" * 64,
            "file_count": 0,
            "total_bytes": 0,
            "files": [],
        },
        status="COMPLETE",
    )
    purge_path, purge_sha = _bound_receipt(
        spec,
        root / "purge.json",
        "contiguous_substrate_health_reprobe_purge",
        authorization_id=authorization_id,
        probe_index=probe_index,
        scan_receipt_sha256=scan_sha,
        health_state_root_absent=True,
        health_runtime_root_absent=True,
        prior_clean_wip_tree_sha256=None,
        post_clean_wip_tree_sha256=None,
        status="PASS",
    )
    preflight_path: str | None = None
    preflight_sha: str | None = None
    guardian_status: str | None = None
    if status == "PASS":
        preflight = root / "substrate_preflight_receipt.json"
        preflight_sha = _write_json(
            preflight,
            {
                "schema": 1,
                "kind": "contiguous_substrate_preflight",
                "status": "PASS",
            },
        )
        preflight_path = str(preflight)
        guardian_status = "PASS"
    remediation_body = {
        "schema": 1,
        "kind":
            "contiguous_substrate_health_rematerialization_evidence",
        "campaign_id": spec.campaign_id,
        "generation_id": spec.generation_id,
        "attempt_id": spec.attempt_id,
        "attempt_spec_sha256":
            R.proposer_attempt_binding_sha256(spec),
        "authorization_id": authorization_id,
        "probe_index": probe_index,
        "authorization_receipt_sha256":
            authorization_receipt_sha256,
        "failed_substrate_identity_sha256":
            failed_substrate_identity_sha256,
        "incident_failure_receipt_sha256":
            incident_failure_receipt_sha256,
        "fresh_state_root_created": True,
        "health_state_root": str(health_state_root),
        "initial_state_tree_sha256": "1" * 64,
        "initial_state_inventory_sha256": "2" * 64,
        "preflight_receipt_sha256": preflight_sha,
        "guardian_state_root_write_probe_status":
            guardian_status,
        "scan_receipt_sha256": scan_sha,
        "purge_receipt_sha256": purge_sha,
        "health_state_root_absent": True,
        "health_runtime_root_absent": True,
        "status": status,
    }
    remediation_epoch = hashlib.sha256(
        R._canonical_json(remediation_body)
    ).hexdigest()
    remediation_path = root / "rematerialization.json"
    remediation_sha = _write_json(
        remediation_path,
        {
            **remediation_body,
            "remediation_epoch_sha256": remediation_epoch,
        },
    )
    healthy_identity = (
        hashlib.sha256(R._canonical_json({
            "schema": 1,
            "kind": "healthy_controller_substrate_identity",
            "failed_substrate_identity_sha256":
                failed_substrate_identity_sha256,
            "remediation_epoch_sha256": remediation_epoch,
            "preflight_receipt_sha256": preflight_sha,
            "guardian_state_root_write_probe_status":
                guardian_status,
            "status": "PASS",
        })).hexdigest()
        if status == "PASS"
        else None
    )
    failure_class = (
        None
        if status == "PASS"
        else "DETERMINISTIC_CONFIGURATION"
    )
    failure_code = (
        None
        if status == "PASS"
        else "controller_state_root_permission"
    )
    receipt_path, receipt_sha = _bound_receipt(
        spec,
        root / "receipt.json",
        "contiguous_substrate_health_reprobe",
        authorization_id=authorization_id,
        authorization_receipt_sha256=
            authorization_receipt_sha256,
        probe_index=probe_index,
        failed_substrate_identity_sha256=
            failed_substrate_identity_sha256,
        healthy_substrate_identity_sha256=healthy_identity,
        incident_failure_receipt_sha256=
            incident_failure_receipt_sha256,
        remediation_epoch_sha256=remediation_epoch,
        rematerialization_evidence_path=str(remediation_path),
        rematerialization_evidence_sha256=remediation_sha,
        fresh_state_root_created=True,
        health_state_root=str(health_state_root),
        health_runtime_root=str(health_runtime_root),
        preflight_receipt_path=preflight_path,
        preflight_receipt_sha256=preflight_sha,
        guardian_state_root_write_probe_status=guardian_status,
        scan_receipt_path=scan_path,
        scan_receipt_sha256=scan_sha,
        purge_receipt_path=purge_path,
        purge_receipt_sha256=purge_sha,
        failure_class=failure_class,
        failure_code=failure_code,
        health_state_root_absent=True,
        health_runtime_root_absent=True,
        proposer_container_started=False,
        bridge_connected=False,
        thread_started=False,
        turn_started=False,
        candidate_authority=False,
        wip_authority=False,
        promotion_authority=False,
        cost_used=0.0,
        status=status,
    )
    return R.BackendSubstrateHealthProbe(
        authorization_id=authorization_id,
        probe_index=probe_index,
        remediation_epoch_sha256=remediation_epoch,
        failed_substrate_identity_sha256=(
            failed_substrate_identity_sha256
        ),
        healthy_substrate_identity_sha256=healthy_identity,
        incident_failure_receipt_sha256=(
            incident_failure_receipt_sha256
        ),
        failure_class=failure_class,
        failure_code=failure_code,
        status=status,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha,
    )


class RecoverableSubstrateBackend(SubstrateFailingBackend):
    def __init__(self):
        super().__init__()
        self.health_probe_calls: list[str] = []

    def probe_substrate_health(self, **kwargs):
        self.health_probe_calls.append(kwargs["authorization_id"])
        return _test_substrate_health_probe(
            spec=kwargs["spec"],
            authorization_id=kwargs["authorization_id"],
            authorization_receipt_sha256=(
                kwargs["authorization_receipt_sha256"]
            ),
            probe_index=kwargs["probe_index"],
            failed_substrate_identity_sha256=(
                kwargs["failed_substrate_identity_sha256"]
            ),
            incident_failure_receipt_sha256=(
                kwargs["incident_failure_receipt_sha256"]
            ),
            status=(
                "FAILED"
                if len(self.health_probe_calls) == 1
                else "PASS"
            ),
        )


class FakeAuxiliaryBackend:
    backend_contract_sha256 = "a" * 64
    input_bundle_contract_sha256 = "b" * 64
    admission_contract_sha256 = "c" * 64
    production_isolation_attested = True
    immutable_private_input_attested = True
    host_admission_attested = True
    descriptor_confined_receipts_attested = True

    def __init__(self, root: Path):
        self.root = root
        self.prepare_calls: list[str] = []
        self.launch_calls: list[str] = []
        self.poll_calls: list[str] = []
        self.collect_calls: list[str] = []
        self.teardown_calls: list[str] = []
        self.admit_calls: list[str] = []
        self.abort_calls: list[tuple[str, str, str]] = []
        self.keep_running: set[str] = set()
        self.fail_prepare_once = False
        self._prepare_failed = False

    @staticmethod
    def configuration() -> R.Scheduler.AuxiliaryLaunchConfiguration:
        return R.Scheduler.AuxiliaryLaunchConfiguration(
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
            supervisory_proposer=(
                R.Scheduler.SupervisoryProposerLaunchConfiguration(
                    schema=1,
                    role=R.Scheduler.SUPERVISORY_PROPOSER_ROLE,
                    automatic_dispatch_enabled=False,
                    model="gpt-5.6-sol",
                    reasoning_effort="max",
                    context_limit_tokens=200_000,
                    max_concurrency=1,
                )
            ),
        )

    @staticmethod
    def _write(path: Path, value: dict) -> tuple[str, str]:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = R._canonical_json(value) + b"\n"
        if path.exists():
            assert path.read_bytes() == payload
        else:
            path.write_bytes(payload)
        return str(path), hashlib.sha256(payload).hexdigest()

    def read_confined_receipt(
        self,
        decision: R.Scheduler.AuxiliaryDecision,
        path_value: str,
        *,
        maximum: int,
    ) -> bytes:
        selected = Path(path_value)
        root = self.root / decision.assignment_id
        relative = selected.parts[len(root.parts):]
        if (
            not selected.is_absolute()
            or selected.parts[:len(root.parts)] != root.parts
            or not relative
            or any(part in {"", ".", ".."} for part in relative)
        ):
            raise R.AuxiliaryBackendFatalError(
                "fake_auxiliary_path_escape"
            )
        return R._bounded_regular_bytes(selected, maximum=maximum)

    def prepare(
        self, decision: R.Scheduler.AuxiliaryDecision
    ) -> R.AuxiliaryPreparedInput:
        self.prepare_calls.append(decision.assignment_id)
        base = self.root / decision.assignment_id
        manifest = base / "input" / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest_payload = R._canonical_json(
            asdict(decision.input_manifest)
        )
        if manifest.exists():
            assert manifest.read_bytes() == manifest_payload
        else:
            manifest.write_bytes(manifest_payload)
        manifest_path = str(manifest)
        manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
        bundle = {
            "schema": 1,
            "kind": "auxiliary_private_input_bundle",
            "assignment_id": decision.assignment_id,
            "frontier_sha256": decision.frontier_sha256,
            "parent_checkpoint_sha256":
                decision.parent_checkpoint_sha256,
            "input_manifest_sha256": decision.input_manifest_sha256,
            "observation_ledger_sha256":
                decision.observation_ledger_sha256,
            "input_bundle_contract_sha256":
                decision.input_bundle_contract_sha256,
            "immutable_inputs": True,
            "live_lineage_mounted": False,
            "public_observations_only": True,
        }
        bundle_path, bundle_sha = self._write(
            base / "host" / "input_bundle_receipt.json",
            bundle,
        )
        if self.fail_prepare_once and not self._prepare_failed:
            self._prepare_failed = True
            raise RuntimeError("synthetic post-materialization crash")
        return R.AuxiliaryPreparedInput(
            input_manifest_path=manifest_path,
            input_manifest_sha256=manifest_sha,
            input_bundle_receipt_path=bundle_path,
            input_bundle_receipt_sha256=bundle_sha,
        )

    def launch(
        self,
        decision: R.Scheduler.AuxiliaryDecision,
        prepared: R.AuxiliaryPreparedInput,
    ) -> R.AuxiliaryLaunch:
        del prepared
        self.launch_calls.append(decision.assignment_id)
        body = {
            "schema": 1,
            "kind": "auxiliary_backend_launch",
            "assignment_id": decision.assignment_id,
            "backend_contract_sha256":
                decision.backend_contract_sha256,
            "expert_id": decision.expert_id,
            "thread_id": decision.thread_id,
            "model": decision.model,
            "reasoning_effort": decision.reasoning_effort,
            "fresh_context": True,
            "live_lineage_write_authority": False,
        }
        path, digest = self._write(
            self.root
            / decision.assignment_id
            / "host"
            / "launch_receipt.json",
            body,
        )
        return R.AuxiliaryLaunch(path, digest)

    def poll(
        self,
        decision: R.Scheduler.AuxiliaryDecision,
        prepared: R.AuxiliaryPreparedInput,
        launched: R.AuxiliaryLaunch,
        *,
        timeout_seconds: float,
    ) -> R.AuxiliaryPoll:
        del prepared, launched
        assert timeout_seconds == R.POLL_TIMEOUT_SECONDS
        self.poll_calls.append(decision.assignment_id)
        return R.AuxiliaryPoll(
            status=(
                "running"
                if decision.assignment_id in self.keep_running
                else "exited"
            ),
            observation_sha256="d" * 64,
        )

    def collect(
        self,
        decision: R.Scheduler.AuxiliaryDecision,
        prepared: R.AuxiliaryPreparedInput,
        launched: R.AuxiliaryLaunch,
        terminal: R.AuxiliaryPoll,
    ) -> R.AuxiliaryCollection:
        del prepared, launched
        assert terminal.status == "exited"
        self.collect_calls.append(decision.assignment_id)
        observation = (
            decision.input_manifest
            .authenticated_public_observation_receipt_sha256s[0]
        )
        output = R.Scheduler.AuxiliaryOutputEvidence(
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
            output_manifest_sha256=hashlib.sha256(
                f"output:{decision.assignment_id}".encode()
            ).hexdigest(),
            public_observation_receipt_sha256s=(observation,),
            challenge=R.Scheduler.SocraticChallengeEvidence(
                schema=1,
                hypothesis="The current mechanism explains the ledger.",
                counter_hypothesis="The mechanism is incidental.",
                falsification_attempt=(
                    "Replayed the distinguishing public prefix."
                ),
                observation_receipt_sha256s=(observation,),
                rejected_conclusions=(
                    "The incidental account does not survive.",
                ),
                surviving_conclusions=(
                    "The mechanism remains consistent.",
                ),
            ),
            quarantined_artifact_sha256s=(
                hashlib.sha256(
                    f"artifact:{decision.assignment_id}".encode()
                ).hexdigest(),
            ),
            result_authority="quarantine_only",
            mutates_live_lineage=False,
        )
        return R.AuxiliaryCollection(output=output, cost_used=0.25)

    def teardown(
        self,
        decision: R.Scheduler.AuxiliaryDecision,
        prepared: R.AuxiliaryPreparedInput,
        launched: R.AuxiliaryLaunch,
        collection: R.AuxiliaryCollection,
    ) -> R.AuxiliaryTeardown:
        del prepared, launched
        assert collection.output is not None
        self.teardown_calls.append(decision.assignment_id)
        body = {
            "schema": 1,
            "kind": "auxiliary_backend_teardown",
            "assignment_id": decision.assignment_id,
            "backend_contract_sha256":
                decision.backend_contract_sha256,
            "output_manifest_sha256":
                collection.output.output_manifest_sha256,
            "descendants_absent": True,
            "live_lineage_mutated": False,
        }
        path, digest = self._write(
            self.root
            / decision.assignment_id
            / "host"
            / "teardown_receipt.json",
            body,
        )
        return R.AuxiliaryTeardown(path, digest)

    def admit(
        self,
        decision: R.Scheduler.AuxiliaryDecision,
        output: R.Scheduler.AuxiliaryOutputEvidence,
    ) -> R.AuxiliaryAdmission:
        self.admit_calls.append(decision.assignment_id)
        base = self.root / decision.assignment_id / "host"
        common = {
            "schema": 1,
            "assignment_id": decision.assignment_id,
            "frontier_sha256": decision.frontier_sha256,
            "parent_checkpoint_sha256":
                decision.parent_checkpoint_sha256,
            "output_manifest_sha256": output.output_manifest_sha256,
        }
        replay_path, replay_sha = self._write(
            base / "fresh_replay_receipt.json",
            {
                **common,
                "kind": "auxiliary_fresh_public_replay",
                "status": "PASS",
            },
        )
        taint_path, taint_sha = self._write(
            base / "taint_receipt.json",
            {
                **common,
                "kind": "auxiliary_taint_scan",
                "status": "CLEAN",
            },
        )
        provenance_path, provenance_sha = self._write(
            base / "provenance_receipt.json",
            {
                **common,
                "kind": "auxiliary_provenance_scan",
                "status": "PASS",
            },
        )
        profile = (
            R.Scheduler.ComplexityProfile(
                schema=1,
                profile_id="profile:" + decision.assignment_id,
                round_index=decision.round_index,
                frontier_sha256=decision.frontier_sha256,
                observation_receipt_sha256=(
                    output.public_observation_receipt_sha256s[0]
                ),
                taint_scan_receipt_sha256=taint_sha,
                priorities=(
                    "mechanism_induction",
                    "exact_planning",
                ),
            )
            if decision.specialization == "complexity_diagnosis"
            else None
        )
        admitted_sha = R.Scheduler.sha256_json(
            asdict(profile if profile is not None else output)
        )
        admission_body = {
            **common,
            "kind": (
                "auxiliary_profile_admission"
                if profile is not None
                else "auxiliary_output_admission"
            ),
            "authority": "host_only",
            "admission_contract_sha256":
                decision.admission_contract_sha256,
            "fresh_replay_receipt_sha256": replay_sha,
            "taint_receipt_sha256": taint_sha,
            "provenance_receipt_sha256": provenance_sha,
            "admitted_evidence_sha256": admitted_sha,
            "verdict": "ADMITTED",
        }
        admission_path, admission_sha = self._write(
            base / "admission_receipt.json", admission_body
        )
        return R.AuxiliaryAdmission(
            verdict="ADMITTED",
            profile=profile,
            reason=None,
            fresh_replay_receipt_path=replay_path,
            fresh_replay_receipt_sha256=replay_sha,
            taint_receipt_path=taint_path,
            taint_receipt_sha256=taint_sha,
            provenance_receipt_path=provenance_path,
            provenance_receipt_sha256=provenance_sha,
            admission_receipt_path=admission_path,
            admission_receipt_sha256=admission_sha,
        )

    def abort(
        self,
        decision: R.Scheduler.AuxiliaryDecision,
        prepared: R.AuxiliaryPreparedInput | None,
        launched: R.AuxiliaryLaunch | None,
        *,
        prior_phase: str,
        reason: str,
    ) -> R.AuxiliaryAbort:
        del prepared, launched
        self.abort_calls.append(
            (decision.assignment_id, prior_phase, reason)
        )
        teardown = None
        if prior_phase == "RUNNING":
            body = {
                "schema": 1,
                "kind": "auxiliary_backend_abort_teardown",
                "assignment_id": decision.assignment_id,
                "backend_contract_sha256":
                    decision.backend_contract_sha256,
                "prior_phase": prior_phase,
                "descendants_absent": True,
                "live_lineage_mutated": False,
            }
            path, digest = self._write(
                self.root
                / decision.assignment_id
                / "host"
                / "abort_teardown_receipt.json",
                body,
            )
            teardown = R.AuxiliaryTeardown(path, digest)
        return R.AuxiliaryAbort(cost_used=0.0, teardown=teardown)


class _MemoryAuxiliaryJournal:
    def __init__(self, owner):
        self.owner = owner
        body = {
            "schema": 1,
            "sequence": 1,
            "event_id": "synthetic:genesis",
            "kind": "GENESIS",
            "recorded_at": 1.0,
            "previous_digest": None,
            "payload": {},
        }
        self.events = [{
            **body,
            "digest": hashlib.sha256(
                R._canonical_json(body)
            ).hexdigest(),
        }]

    def read(self):
        return list(self.events)

    def append(self, *, event_id, kind, payload, recorded_at):
        for event in self.events:
            if event["event_id"] == event_id:
                assert event["kind"] == kind
                assert event["payload"] == payload
                return event
        body = {
            "schema": 1,
            "sequence": len(self.events) + 1,
            "event_id": event_id,
            "kind": kind,
            "recorded_at": float(recorded_at),
            "previous_digest": self.events[-1]["digest"],
            "payload": json.loads(R._canonical_json(payload)),
        }
        event = {
            **body,
            "digest": hashlib.sha256(
                R._canonical_json(body)
            ).hexdigest(),
        }
        self.events.append(event)
        self.owner._reduce_auxiliary_event(event)
        return event


class _SyntheticAuxiliaryRunner(R.ContiguousCampaignRunner):
    """Fast exact-n harness around the production auxiliary cycle."""

    def __init__(self, root: Path, backend: FakeAuxiliaryBackend):
        self.root = root
        self.root.mkdir(parents=True)
        (self.root / "attempt_journal").mkdir()
        self.auxiliary = self.root / "auxiliary"
        self.auxiliary.mkdir()
        self.auxiliary_backend = backend
        self.auxiliary_launch_configuration = backend.configuration()
        self._trusted_auxiliary_event_digests = set()
        self.clock = Clock()
        self.id_factory = ids()
        inventory = Contract.authoritative_inventory()
        target = sorted(inventory)[0]
        source = self.root / "source"
        _, source_sha = FakeInputBuilder().initialize_lane_source(
            target, source
        )
        checkpoint_sha = "1" * 64
        lanes = {}
        for game, target_level in inventory.items():
            lanes[game] = {
                "target": target_level,
                "reached": 0,
                "no_progress": 5 if game == target else 0,
                "last_dispatch_sequence": 100 if game == target else 0,
                "checkpoint_sha256": checkpoint_sha,
                "source_path": str(source),
                "source_tree_sha256": source_sha,
                "active": (
                    "active:max-proposer" if game == target else None
                ),
                "blocked": (
                    None if game == target else "test-only-blocker"
                ),
                "wip": None,
                "clean_proposer_settlements": [],
                "public_observation_receipt_sha256s": [],
            }
        selected = lanes[target]
        selected_frontier = R.frontier_sha256(
            target, 0, checkpoint_sha
        )
        for index in range(5):
            policy = R.Scheduler.retry_policy(index)
            selected["clean_proposer_settlements"].append(
                R.Scheduler.CleanProposerSettlement(
                    schema=1,
                    game=target,
                    frontier_sha256=selected_frontier,
                    parent_checkpoint_sha256=checkpoint_sha,
                    attempt_id=f"attempt:{index}",
                    scheduler_decision_id=f"decision:{index}",
                    no_progress_before=index,
                    effort=policy.effort,
                    soft_allocation_seconds=(
                        policy.soft_allocation_seconds
                    ),
                    requested_wip_mode=policy.requested_wip_mode,
                    supervisory_handoff_sha256=None,
                    result_sequence=index + 2,
                    result_digest=hashlib.sha256(
                        f"clean:{index}".encode()
                    ).hexdigest(),
                )
            )
        observation_receipt_sha256 = "d" * 64
        selected[
            "public_observation_receipt_sha256s"
        ] = [observation_receipt_sha256]
        sidecar_requests = {}
        for settlement in selected["clean_proposer_settlements"]:
            draft = R.Scheduler.NativeSidecarRequestDraft(
                schema=1,
                kind="NATIVE_SIDECAR_REQUEST_DRAFT",
                request_id=f"request:{settlement.attempt_id}",
                game=target,
                frontier_sha256=selected_frontier,
                parent_checkpoint_sha256=checkpoint_sha,
                native_attempt_id=settlement.attempt_id,
                semantic_brief=(
                    "Falsify the unresolved mechanism at clean retry "
                    f"{settlement.no_progress_before}."
                ),
                cited_public_observation_receipt_sha256s=(
                    observation_receipt_sha256,
                ),
                scheduler_authored=False,
                live_lineage_mutation_authority=False,
                promotion_authority=False,
                draft_sha256="",
            )
            draft = replace(
                draft,
                draft_sha256=R.Scheduler.sha256_json(
                    R.Scheduler
                    ._native_sidecar_request_draft_body(draft)
                ),
            )
            request = (
                R.Scheduler.native_sidecar_request_from_draft(
                    draft, settlement=settlement
                )
            )
            sidecar_requests[request.request_sha256] = {
                "request": request,
                "origin_kind":
                    "NATIVE_SIDECAR_REQUEST_ADMITTED",
                "origin_id": settlement.attempt_id,
                "admitted_sequence": settlement.result_sequence + 1,
                "admitted_event_digest": hashlib.sha256(
                    f"request:{settlement.attempt_id}".encode()
                ).hexdigest(),
                "invalidated": False,
            }
        self.target_game = target
        self.old_frontier_sha256 = selected_frontier
        self._budget = R.Scheduler.BudgetState(
            cost_window_id="synthetic-window",
            limit_units=None,
            settled_units=0,
        )
        self._state = {
            "campaign_id": "campaign:synthetic-auxiliary",
            "inventory": inventory,
            "max_lanes": 2,
            "limit": None,
            "limit_units": None,
            "settled_cost_units": 0,
            "live_budget_reservations": [],
            "cost_window_id": "synthetic-window",
            "lanes": lanes,
            "attempts": {
                "active:max-proposer": {
                    "phase": "SYNTHETIC_ACTIVE"
                }
            },
            "auxiliary_assignments": {},
            "sidecar_requests": sidecar_requests,
            "complexity_rounds": [],
            "pending_scheduler_decision": None,
            "pending_auxiliary_decision": None,
            "used_scheduler_identifiers": [],
            "used_auxiliary_thread_ids": [],
            "failure_operation_circuits": {},
            "failure_domain_circuits": {},
            "substrate_incident": None,
            "operator_incident": None,
            "storage_incident": None,
            "storage_quiescence": None,
            "complete": False,
            "solved_levels": 0,
            "total_levels": sum(inventory.values()),
            "draining": False,
        }
        self.journal = _MemoryAuxiliaryJournal(self)

    def state(self):
        self._state["settled_cost_units"] = self._budget.settled_units
        self._state["live_budget_reservations"] = [
            asdict(item) for item in self._budget.live_reservations
        ]
        return self._state

    def _reserve_attempt(self, state):
        del state
        return None

    def _scheduler_snapshot_from_state(self, state):
        frontiers = []
        for game in sorted(state["lanes"]):
            lane = state["lanes"][game]
            frontier = R.frontier_sha256(
                game, lane["reached"], lane["checkpoint_sha256"]
            )
            frontiers.append(R.Scheduler.Frontier(
                game=game,
                target=lane["target"],
                reached=lane["reached"],
                no_progress=lane["no_progress"],
                last_dispatch_sequence=lane["last_dispatch_sequence"],
                parent_checkpoint_sha256=lane["checkpoint_sha256"],
                parent_source_path=lane["source_path"],
                parent_source_tree_sha256=lane["source_tree_sha256"],
                frontier_sha256=frontier,
                active_attempt_id=lane["active"],
                draining=False,
                blocked_reason=lane["blocked"],
                wip=None,
                evidence=R.Scheduler.selection_evidence(
                    parent_source_path=lane["source_path"],
                    parent_source_tree_sha256=(
                        lane["source_tree_sha256"]
                    ),
                ),
                public_observation_receipt_sha256s=tuple(
                    lane.get(
                        "public_observation_receipt_sha256s", ()
                    )
                ),
                observation_ledger_sha256=(
                    R.Scheduler.public_observation_ledger_sha256(
                        game=game,
                        frontier_sha256=frontier,
                        parent_checkpoint_sha256=(
                            lane["checkpoint_sha256"]
                        ),
                        receipt_sha256s=lane.get(
                            "public_observation_receipt_sha256s",
                            (),
                        ),
                    )
                ),
            ))
        head = self.journal.read()[-1]
        return R.Scheduler.validate_snapshot(
            R.Scheduler.CampaignSnapshot(
                campaign_id=state["campaign_id"],
                journal_head_sequence=head["sequence"],
                journal_head_digest=head["digest"],
                inventory=tuple(state["inventory"].items()),
                max_lanes=state["max_lanes"],
                frontiers=tuple(frontiers),
                budget=self._budget,
                clean_proposer_settlements=tuple(
                    settlement
                    for lane in state["lanes"].values()
                    for settlement in lane[
                        "clean_proposer_settlements"
                    ]
                ),
                complexity_rounds=tuple(
                    state["complexity_rounds"]
                ),
                auxiliary_assignments=tuple(
                    item["state"]
                    for item in
                    state["auxiliary_assignments"].values()
                ),
                sidecar_requests=tuple(
                    item["request"]
                    for item in state["sidecar_requests"].values()
                    if not item["invalidated"]
                ),
            )
        )

    def _auxiliary_observation_ledger_sha256(self, state, game):
        lane = state["lanes"][game]
        return R.Scheduler.public_observation_ledger_sha256(
            game=game,
            frontier_sha256=R.frontier_sha256(
                game,
                lane["reached"],
                lane["checkpoint_sha256"],
            ),
            parent_checkpoint_sha256=lane["checkpoint_sha256"],
            receipt_sha256s=tuple(
                lane["public_observation_receipt_sha256s"]
            ),
        )

    def _reduce_auxiliary_event(self, event):
        kind = event["kind"]
        payload = event["payload"]
        if kind == "FAILURE_CIRCUIT_FAILURE":
            operation_key = (
                f"{payload['operation']}:{payload['fault_domain']}"
            )
            self._state["failure_operation_circuits"][
                operation_key
            ] = {
                "consecutive": payload["operation_consecutive"],
                "failure_index": payload["operation_failure_index"],
                "retry_not_before": payload["retry_not_before"],
            }
            self._state["failure_domain_circuits"][
                payload["fault_domain"]
            ] = {
                "consecutive": payload["domain_consecutive"],
                "failure_index": payload["domain_failure_index"],
                "retry_not_before": payload["retry_not_before"],
                "last_operation": payload["operation"],
            }
            return
        if kind == "FAILURE_CIRCUIT_RESET":
            operation_key = (
                f"{payload['operation']}:{payload['fault_domain']}"
            )
            operation_state = self._state[
                "failure_operation_circuits"
            ][operation_key]
            domain_state = self._state[
                "failure_domain_circuits"
            ][payload["fault_domain"]]
            if payload["reset_operation"]:
                operation_state.update(
                    consecutive=0, retry_not_before=None
                )
            if payload["reset_domain"]:
                domain_state.update(
                    consecutive=0,
                    retry_not_before=None,
                    last_operation=None,
                )
            return
        if kind == "OPERATOR_INCIDENT":
            self._state["operator_incident"] = dict(payload)
            return
        if kind in {
            "NATIVE_SIDECAR_REQUEST_ADMITTED",
            "SUPERVISORY_SIDECAR_REQUEST_ADMITTED",
        }:
            request = R.Scheduler.sidecar_request_from_dict(
                payload["request"]
            )
            self._state["sidecar_requests"][
                request.request_sha256
            ] = {
                "request": request,
                "origin_kind": kind,
                "origin_id": (
                    payload["attempt_id"]
                    if kind
                    == "NATIVE_SIDECAR_REQUEST_ADMITTED"
                    else payload["assignment_id"]
                ),
                "admitted_sequence": event["sequence"],
                "admitted_event_digest": event["digest"],
                "invalidated": False,
            }
            return
        if kind == "AUXILIARY_DECISION":
            decision = R.Scheduler.auxiliary_decision_from_dict(
                payload["decision"]
            )
            self._state["pending_auxiliary_decision"] = (
                payload["decision"]
            )
            self._state["used_scheduler_identifiers"].extend([
                decision.decision_id,
                decision.assignment_id,
                decision.reservation_id,
                decision.expert_id,
            ])
            self._state["used_auxiliary_thread_ids"].append(
                decision.thread_id
            )
            return
        if kind == "AUXILIARY_RESERVED":
            decision = R.Scheduler.auxiliary_decision_from_dict(
                self._state["pending_auxiliary_decision"]
            )
            self._budget = R.Scheduler.reserve_budget(
                self._budget,
                reservation_id=decision.reservation_id,
                attempt_id=decision.assignment_id,
                units=decision.reservation_units,
            )
            assignment_state = R.Scheduler.AuxiliaryAssignmentState(
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
                input_manifest_sha256=(
                    decision.input_manifest_sha256
                ),
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
                phase="RESERVED",
            )
            self._state["auxiliary_assignments"][
                decision.assignment_id
            ] = {
                "state": assignment_state,
                "decision": decision,
                "prepared": None,
                "launched": None,
                "terminal": None,
                "collection": None,
                "teardown": None,
                "admission": None,
                "abort_reason": None,
            }
            self._state["pending_auxiliary_decision"] = None
            return
        assignment = self._state["auxiliary_assignments"][
            payload["assignment_id"]
        ]
        if kind == "AUXILIARY_INPUT_PREPARED":
            assignment["prepared"] = R.AuxiliaryPreparedInput(
                **{
                    key: payload[key]
                    for key in R.AuxiliaryPreparedInput
                    .__dataclass_fields__
                }
            )
            assignment["state"] = replace(
                assignment["state"], phase="INPUT_PREPARED"
            )
        elif kind == "AUXILIARY_LAUNCHED":
            assignment["launched"] = R.AuxiliaryLaunch(
                launch_receipt_path=payload["launch_receipt_path"],
                launch_receipt_sha256=(
                    payload["launch_receipt_sha256"]
                ),
            )
            assignment["state"] = replace(
                assignment["state"], phase="RUNNING"
            )
        elif kind == "AUXILIARY_RESULT_QUARANTINED":
            output = R.Scheduler.auxiliary_output_from_dict(
                payload["output"], assignment=assignment["state"]
            )
            self._budget = R.Scheduler.settle_budget(
                self._budget,
                reservation_id=assignment["state"].reservation_id,
                attempt_id=assignment["state"].assignment_id,
                charged_units=payload["authenticated_cost_units"],
            )
            assignment["collection"] = R.AuxiliaryCollection(
                output=output, cost_used=payload["cost_used"]
            )
            assignment["teardown"] = R.AuxiliaryTeardown(
                payload["teardown_receipt_path"],
                payload["teardown_receipt_sha256"],
            )
            assignment["state"] = replace(
                assignment["state"],
                phase="QUARANTINED",
                output=output,
            )
        elif kind == "AUXILIARY_PROFILE_ADMITTED":
            profile = R.Scheduler.complexity_profile_from_dict(
                payload["profile"]
            )
            lane = self._state["lanes"][assignment["state"].game]
            round_state = R.Scheduler.ComplexityRoundState(
                schema=1,
                game=assignment["state"].game,
                frontier_sha256=assignment["state"].frontier_sha256,
                parent_checkpoint_sha256=(
                    assignment["state"].parent_checkpoint_sha256
                ),
                parent_source_tree_sha256=lane["source_tree_sha256"],
                round_index=assignment["state"].round_index,
                profile=profile,
                diagnosis_assignment_id=(
                    assignment["state"].assignment_id
                ),
                trigger_no_progress=(
                    assignment["state"].trigger_no_progress
                ),
                trigger_history_sha256=(
                    assignment["state"].trigger_history_sha256
                ),
                input_manifest_sha256=(
                    assignment["state"].input_manifest_sha256
                ),
                observation_ledger_sha256=(
                    assignment["state"].observation_ledger_sha256
                ),
                admission_receipt_path=(
                    payload["admission_receipt_path"]
                ),
                admission_receipt_sha256=(
                    payload["admission_receipt_sha256"]
                ),
                admitted_sequence=event["sequence"],
                admitted_event_digest=event["digest"],
            )
            self._state["complexity_rounds"].append(round_state)
            assignment["state"] = replace(
                assignment["state"],
                phase="ADMITTED",
                admission_receipt_path=(
                    payload["admission_receipt_path"]
                ),
                admission_receipt_sha256=(
                    payload["admission_receipt_sha256"]
                ),
                admitted_sequence=event["sequence"],
                admitted_event_digest=event["digest"],
            )
        elif kind == "AUXILIARY_OUTPUT_REJECTED":
            assignment["state"] = replace(
                assignment["state"], phase="REJECTED"
            )
        elif kind == "AUXILIARY_OUTPUT_ADMITTED":
            assignment["state"] = replace(
                assignment["state"],
                phase="ADMITTED",
                admission_receipt_path=(
                    payload["admission_receipt_path"]
                ),
                admission_receipt_sha256=(
                    payload["admission_receipt_sha256"]
                ),
                admitted_sequence=event["sequence"],
                admitted_event_digest=event["digest"],
            )
        elif kind == "AUXILIARY_ABORTED":
            self._budget = R.Scheduler.settle_budget(
                self._budget,
                reservation_id=assignment["state"].reservation_id,
                attempt_id=assignment["state"].assignment_id,
                charged_units=payload["authenticated_cost_units"],
            )
            assignment["teardown"] = (
                R.AuxiliaryTeardown(
                    payload["teardown_receipt_path"],
                    payload["teardown_receipt_sha256"],
                )
                if payload["teardown_receipt_path"] is not None
                else None
            )
            assignment["abort_reason"] = payload["reason"]
            assignment["state"] = replace(
                assignment["state"], phase="ABORTED"
            )

    def promote(self):
        lane = self._state["lanes"][self.target_game]
        lane.update(
            reached=1,
            no_progress=0,
            checkpoint_sha256="2" * 64,
            active=None,
            clean_proposer_settlements=[],
            public_observation_receipt_sha256s=[],
        )
        self._state["solved_levels"] = 1
        for item in self._state["auxiliary_assignments"].values():
            if item["state"].frontier_sha256 == self.old_frontier_sha256:
                item["state"] = replace(
                    item["state"], invalidated=True
                )
        for item in self._state["sidecar_requests"].values():
            if (
                item["request"].frontier_sha256
                == self.old_frontier_sha256
            ):
                item["invalidated"] = True
        self._state["complexity_rounds"] = [
            replace(item, invalidated=True)
            if item.frontier_sha256 == self.old_frontier_sha256
            else item
            for item in self._state["complexity_rounds"]
        ]


class FakePromotionGate:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True)
        self.commits: dict[str, R.PromotionCommit] = {}
        self.calls: list[str] = []
        self.crash_after_first_commit = False
        self._did_commit_crash = False
        self.force_to_level_delta = 0

    def commit(
        self, *, spec: R.AttemptSpec, candidate: R.PromotionCandidate
    ) -> R.PromotionCommit:
        self.calls.append(spec.attempt_id)
        if spec.attempt_id not in self.commits:
            parent = json.loads(
                Path(spec.parent_checkpoint_path).read_text(
                    encoding="utf-8"
                )
            )
            to_level = spec.target_level + self.force_to_level_delta
            records = list(parent["records"])
            for level in range(parent["reached"] + 1, to_level + 1):
                records.append(
                    {"level": level, "marginal_C": 1, "reached": True}
                )
            exact_path = list(parent["final_path"]) + [1] * (
                to_level - parent["reached"]
            )
            checkpoint = {
                "game": spec.game,
                "reached": to_level,
                "total_marginal_C": sum(
                    record["marginal_C"] for record in records
                ),
                "records": records,
                "final_path": exact_path,
                "validated": True,
            }
            version_id = uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"fake-promotion:{spec.attempt_id}",
            ).hex
            version_root = self.root / "versions" / version_id
            subject_root = version_root / f"{spec.game}_legs"
            subject_root.mkdir(parents=True)
            path = subject_root / Contract.CHECKPOINT_NAME
            path.write_text(
                json.dumps(checkpoint, sort_keys=True), encoding="utf-8"
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            winning_source = (
                subject_root / Contract.WINNING_SOURCE_NAME
            )
            winning_source.mkdir()
            source_payloads = {
                entry.name: entry.read_bytes()
                for entry in Path(spec.workspace_dir).iterdir()
                if (
                    entry.is_file()
                    and entry.name
                    not in R.SourceSchema.FORBIDDEN_FILES
                    and entry.suffix
                    in R.SourceSchema.ALLOWED_SUFFIXES
                )
            }
            R.SourceSchema.validate_source_payloads(source_payloads)
            for name, payload in source_payloads.items():
                (winning_source / name).write_bytes(payload)
            source_digest = Contract._tree_hash(winning_source)
            R._seal_regular_tree(winning_source)
            receipt = (
                subject_root / Contract.HOST_RECEIPT_NAME
            )
            receipt.write_text(
                json.dumps(
                    {
                        "attempt_id": spec.attempt_id,
                        "source_tree_sha256": source_digest,
                        "supervisory_handoff_sha256": (
                            candidate.supervisory_handoff_sha256
                        ),
                        "supervisory_native_reproduction_receipt_sha256":
                            candidate
                            .supervisory_native_reproduction_receipt_sha256,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            receipt_digest = hashlib.sha256(
                receipt.read_bytes()
            ).hexdigest()
            self.commits[spec.attempt_id] = R.PromotionCommit(
                game=spec.game,
                from_level=spec.target_level - 1,
                to_level=to_level,
                parent_checkpoint_sha256=spec.parent_checkpoint_sha256,
                checkpoint_path=str(path),
                checkpoint_sha256=digest,
                exact_path=tuple(exact_path),
                promotion_receipt_sha256=receipt_digest,
                source_version_id=version_id,
                source_tree_sha256=source_digest,
                supervisory_handoff_sha256=(
                    candidate.supervisory_handoff_sha256
                ),
                supervisory_native_reproduction_receipt_sha256=(
                    candidate
                    .supervisory_native_reproduction_receipt_sha256
                ),
            )
        commit = self.commits[spec.attempt_id]
        if self.crash_after_first_commit and not self._did_commit_crash:
            self._did_commit_crash = True
            raise R.SimulatedCrash()
        return commit

    def recover(
        self, *, spec: R.AttemptSpec, candidate: R.PromotionCandidate
    ) -> R.PromotionCommit | None:
        return self.commits.get(spec.attempt_id)


def ids():
    index = 0

    def next_id():
        nonlocal index
        index += 1
        return str(uuid.UUID(int=index, version=4))

    return next_id


def backend_configuration() -> R.BackendConfiguration:
    transport = R.ProposerTransportConfiguration(
        model="gpt-5.6-sol",
        model_provider="openai",
        allow_provider_model_fallback=False,
        reasoning_effort_allowlist=R.EXPECTED_REASONING_EFFORT_ALLOWLIST,
        controller_image_reference=(
            "gkm/arc-controller@sha256:" + "9" * 64
        ),
        controller_image_digest="sha256:" + "9" * 64,
        controller_entrypoint=R.EXPECTED_CONTROLLER_ENTRYPOINT,
        controller_guardian_path=R.EXPECTED_CONTROLLER_ENTRYPOINT[0],
        controller_guardian_sha256="a" * 64,
        controller_user=R.EXPECTED_CONTROLLER_USER,
        controller_egress_policy=R.EXPECTED_CONTROLLER_EGRESS_POLICY,
        controller_egress_proxy_image_reference=(
            "gkm/arc-egress-proxy@sha256:" + "b" * 64
        ),
        controller_egress_proxy_image_digest="sha256:" + "b" * 64,
        controller_egress_policy_sha256="c" * 64,
        controller_cpus=2.0,
        controller_memory_bytes=4 * 1024**3,
        controller_pids=256,
        controller_tmpfs_bytes=512 * 1024**2,
        arena_transport="docker-attach-stdio+named-volume-unix",
        arena_relay_image_reference=(
            "gkm/arc-arena-relay@sha256:" + "d" * 64
        ),
        arena_relay_image_digest="sha256:" + "d" * 64,
        arena_relay_source_sha256=hashlib.sha256(
            (
                Path(R.__file__).with_name(
                    "arc_agi3_arena_volume_relay.py"
                )
            ).read_bytes()
        ).hexdigest(),
        codex_launcher_path="/opt/codex/bin/codex",
        codex_launcher_sha256="2" * 64,
        codex_package_manifest_path="/opt/codex/package.json",
        codex_package_manifest_sha256="3" * 64,
        codex_binary_path="/opt/codex/native/codex",
        codex_binary_sha256="4" * 64,
        codex_binary_bytes=271_134_288,
        codex_cli_version="codex-cli 0.145.0",
        app_server_protocol_schema_path="/opt/codex/schema.json",
        app_server_protocol_schema_sha256="5" * 64,
        app_server_protocol_schema_bundle_path="/opt/codex/schema-bundle.json",
        app_server_protocol_schema_bundle_sha256="6" * 64,
        controller_preflight_request_allowlist=
            R.EXPECTED_CONTROLLER_PREFLIGHT_REQUEST_ALLOWLIST,
        controller_preflight_notification_allowlist=
            R.EXPECTED_CONTROLLER_PREFLIGHT_NOTIFICATION_ALLOWLIST,
        controller_turn_request_allowlist=
            R.EXPECTED_CONTROLLER_TURN_REQUEST_ALLOWLIST,
        dynamic_tool_namespace=R.EXPECTED_DYNAMIC_TOOL_NAMESPACE,
        dynamic_tool_names=R.EXPECTED_DYNAMIC_TOOL_NAMES,
        bridge_protocol_version=1,
        bridge_operation_allowlist=
            R.EXPECTED_BRIDGE_OPERATION_ALLOWLIST,
        bridge_exec_allowlist=R.EXPECTED_BRIDGE_EXEC_ALLOWLIST,
        bridge_max_request_bytes=4 * 1024 * 1024,
        bridge_max_response_bytes=4 * 1024 * 1024,
        bridge_max_file_bytes=1024 * 1024,
        bridge_max_total_export_bytes=8 * 1024 * 1024,
        bridge_max_processes=32,
        bridge_max_exec_seconds=600,
    )
    return R.BackendConfiguration(
        image_reference="gkm/arc-runner@" + IMAGE_DIGEST,
        image_digest=IMAGE_DIGEST,
        worker_command=WORKER_COMMAND,
        resource_limits=R.ResourceLimitsProjection(
            cpus=2.0,
            memory_bytes=4 * 1024**3,
            pids=256,
            tmpfs_bytes=512 * 1024**2,
        ),
        proposer_transport=transport,
    )


def candidate_for(spec: R.AttemptSpec, *, to_delta: int = 0):
    manifest = Path(spec.output_dir) / Contract.CANDIDATE_NAME
    manifest.write_text(
        json.dumps(
            {
                "attempt_id": spec.attempt_id,
                "game": spec.game,
                "target_level": spec.target_level + to_delta,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return R.PromotionCandidate(
        game=spec.game,
        from_level=spec.target_level - 1,
        to_level=spec.target_level + to_delta,
        parent_checkpoint_sha256=spec.parent_checkpoint_sha256,
        candidate_manifest_path=str(manifest),
        candidate_manifest_sha256=hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest(),
        probe_isolation_mode=TEST_PROBE_ISOLATION_MODE,
        probe_isolation_evidence_sha256=(
            TEST_PROBE_ISOLATION_SHA256
        ),
        supervisory_handoff_sha256=None,
        supervisory_native_reproduction_receipt_sha256=None,
    )


def candidate_result(
    spec: R.AttemptSpec, *, to_delta: int = 0
) -> R.AttemptResult:
    return R.AttemptResult(
        kind="candidate",
        candidate=candidate_for(spec, to_delta=to_delta),
    )


def make_runner(
    tmp_path,
    *,
    backend=None,
    gate=None,
    builder=None,
    max_lanes=6,
    limit=None,
    clock=None,
    id_factory=None,
    auxiliary_backend=None,
    auxiliary_launch_configuration=None,
    operator_configuration_sha256=None,
):
    backend = backend or FakeBackend()
    gate = gate or FakePromotionGate(tmp_path / "promotions")
    builder = builder or FakeInputBuilder()
    clock = clock or Clock()
    runner = R.ContiguousCampaignRunner(
        tmp_path / "campaign",
        backend=backend,
        promotion_gate=gate,
        input_builder=builder,
        backend_configuration=backend_configuration(),
        cost_window_id=COST_WINDOW_ID,
        max_lanes=max_lanes,
        limit=limit,
        operator_configuration_sha256=(
            operator_configuration_sha256
        ),
        controller_state_canaries=(
            backend.controller_state_canaries
        ),
        auxiliary_backend=auxiliary_backend,
        auxiliary_launch_configuration=(
            auxiliary_launch_configuration
        ),
        clock=clock,
        id_factory=id_factory or ids(),
    )
    return runner, backend, gate, builder, clock


def _identity_fixture_wip(
    spec: R.AttemptSpec,
    *,
    attempt_spec_sha256: str | None = None,
) -> R.WipSnapshot:
    generation = Path(spec.generation_dir)
    host = generation / "host"
    wip_root = Path(spec.output_dir) / "wip"
    provisional = R.WipSnapshot(
        snapshot_id="wip:" + spec.attempt_id,
        wip_root_path=str(wip_root),
        wip_tree_sha256="a" * 64,
        solver_source_path=str(wip_root / "solver_source"),
        solver_source_tree_sha256="b" * 64,
        game=spec.game,
        target_level=spec.target_level,
        parent_checkpoint_sha256=spec.parent_checkpoint_sha256,
        frontier_sha256=spec.frontier_sha256,
        codex_thread_id=str(uuid.uuid4()),
        final_thread_binding_path=str(
            generation / "final_thread_binding.json"
        ),
        final_thread_binding_sha256="c" * 64,
        wip_export_receipt_path=str(
            host / "wip_export_receipt.json"
        ),
        wip_export_receipt_sha256="d" * 64,
        final_transcript_chain_receipt_path=str(
            host / "final_transcript_chain_receipt.json"
        ),
        final_transcript_chain_receipt_sha256="e" * 64,
        transcript_chain_sha256="f" * 64,
        controller_state_scan_receipt_path=str(
            host / "controller_state_scan_receipt.json"
        ),
        controller_state_scan_receipt_sha256="1" * 64,
        retained_canary_scan_receipt_path=str(
            generation / "retained_canary_scan_receipt.json"
        ),
        retained_canary_scan_receipt_sha256="2" * 64,
        taint_scan_receipt_path=str(
            host / "taint_scan_receipt.json"
        ),
        taint_scan_receipt_sha256="3" * 64,
        token_usage_receipt_path=str(
            host / "token_usage_receipt.json"
        ),
        token_usage_receipt_sha256="4" * 64,
        provider_usage_receipt_path=str(
            host / "provider_usage_receipt.json"
        ),
        provider_usage_receipt_sha256="5" * 64,
        app_server_state_dir=spec.app_server_state_dir,
        app_server_state_tree_sha256="6" * 64,
        wip_publication_receipt_path=str(
            generation / "wip_publication_receipt.json"
        ),
        wip_publication_receipt_sha256="0" * 64,
        supervisory_handoff_sha256=None,
        supervisory_native_reproduction_receipt_path=None,
        supervisory_native_reproduction_receipt_sha256=None,
    )
    publication = {
        "schema": 1,
        "kind": "contiguous_wip_publication",
        "campaign_id": spec.campaign_id,
        "generation_id": spec.generation_id,
        "attempt_id": spec.attempt_id,
        "attempt_spec_sha256": (
            R.proposer_attempt_binding_sha256(spec)
            if attempt_spec_sha256 is None
            else attempt_spec_sha256
        ),
        **R._wip_publication_fields(provisional),
    }
    publication_path = Path(
        provisional.wip_publication_receipt_path
    )
    publication_path.write_bytes(
        R._canonical_json(publication) + b"\n"
    )
    return replace(
        provisional,
        wip_publication_receipt_sha256=hashlib.sha256(
            publication_path.read_bytes()
        ).hexdigest(),
    )


def test_shared_terminal_wip_schema_round_trips_runner_and_scheduler(
    tmp_path,
):
    runner, backend, _, _, _ = make_runner(tmp_path, max_lanes=1)
    runner.cycle()
    spec = next(iter(backend.specs.values()))
    wip = _identity_fixture_wip(spec)
    encoded = R._wip_to_dict(wip)
    assert R._wip_from_dict(encoded) == wip
    assert R.Scheduler.wip_binding_from_dict(encoded) == wip


def test_runner_rejects_terminal_wip_provider_path_substitution(tmp_path):
    runner, backend, _, _, _ = make_runner(tmp_path, max_lanes=1)
    runner.cycle()
    spec = next(iter(backend.specs.values()))
    wip = _identity_fixture_wip(spec)
    substituted = replace(
        wip,
        provider_usage_receipt_path=str(
            Path(spec.generation_dir)
            / "substituted"
            / "provider_usage_receipt.json"
        ),
    )
    with pytest.raises(
        R.ContiguousRunnerError,
        match="one canonical generation",
    ):
        runner._validate_wip_for_spec(substituted, spec)


def test_runner_rejects_terminal_wip_attempt_spec_substitution(tmp_path):
    runner, backend, _, _, _ = make_runner(tmp_path, max_lanes=1)
    runner.cycle()
    spec = next(iter(backend.specs.values()))
    substituted = _identity_fixture_wip(
        spec, attempt_spec_sha256="9" * 64
    )
    with pytest.raises(
        R.ContiguousRunnerError,
        match="publication receipt is incomplete or substituted",
    ):
        runner._validate_wip_for_spec(substituted, spec)


def test_authoritative_inventory_six_disjoint_bound_lanes(tmp_path):
    runner, backend, _, builder, _ = make_runner(tmp_path)
    report = runner.cycle()
    state = runner.state()
    assert report["active_lanes"] == 6
    assert len(backend.specs) == 6
    assert len({spec.game for spec in backend.specs.values()}) == 6
    assert len(state["inventory"]) == 25
    assert sum(state["inventory"].values()) == 183
    assert all(spec.target_level == 1 for spec in backend.specs.values())
    for spec in backend.specs.values():
        assert spec.image_reference.endswith("@" + spec.image_digest)
        assert spec.worker_command == WORKER_COMMAND
        assert spec.resource_limits.pids == 256
        assert spec.attempt_id != spec.generation_id
        assert spec.attempt_id in builder.layouts
        assert Path(spec.arena_socket_path).name == "arena.sock"
        assert Path(spec.arena_token_file_path).name == "token"
        assert Path(spec.host_transcript_path).name == "backend.jsonl"
        assert stat_mode(Path(spec.input_dir)) == 0o700


def test_substrate_permission_failure_latches_all_unstarted_work_and_polling_is_quiet(
    tmp_path,
):
    backend = SubstrateFailingBackend()
    runner, _, _, _, clock = make_runner(
        tmp_path, backend=backend, max_lanes=6
    )

    first = runner.cycle()
    state = runner.state()
    incident = state["substrate_incident"]
    assert incident is not None
    assert incident["failure_class"] == "DETERMINISTIC_CONFIGURATION"
    assert incident["failure_code"] == "controller_state_root_permission"
    assert len(backend.launch_calls) == 1
    assert first["active_lanes"] == 0
    assert all(
        lane["no_progress"] == 0
        for lane in state["lanes"].values()
    )
    assert state["settled_cost_units"] == 0

    # One restart pass records the circuit deadline.  Polling before that
    # deadline is observationally quiet: no launcher call, journal event, or
    # artifact growth.
    runner.cycle(now=clock.value)
    baseline_events = len(runner.journal._read_authenticated())
    baseline_files = tuple(sorted(
        str(path.relative_to(runner.root))
        for path in runner.root.rglob("*")
        if path.is_file()
    ))
    baseline_launches = tuple(backend.launch_calls)
    for _ in range(100):
        runner.cycle(now=clock.value)
    assert tuple(backend.launch_calls) == baseline_launches
    assert len(runner.journal._read_authenticated()) == baseline_events
    assert tuple(sorted(
        str(path.relative_to(runner.root))
        for path in runner.root.rglob("*")
        if path.is_file()
    )) == baseline_files
    assert runner.state()["substrate_incident"] is not None


def test_allowlisted_meta_recovery_requires_real_pass_then_clears_both_latches(
    tmp_path,
):
    backend = RecoverableSubstrateBackend()
    runner, _, _, _, clock = make_runner(
        tmp_path,
        backend=backend,
        max_lanes=2,
        operator_configuration_sha256="9" * 64,
    )
    runner.cycle()
    runner.cycle(now=clock.value)
    clock.advance(
        R.SUBSTRATE_HEALTH_REPROBE_BACKOFF_SECONDS[0]
    )
    runner.cycle(now=clock.value)
    assert len(backend.health_probe_calls) == 1
    assert (
        runner.state()["substrate_incident"]["last_health_probe"][
            "status"
        ]
        == "FAILED"
    )
    runner.cycle(now=clock.value)
    latched = runner.state()
    assert latched["operator_incident"] is not None
    assert latched["substrate_incident"] is not None
    settled_before = latched["settled_cost_units"]
    no_progress_before = {
        game: lane["no_progress"]
        for game, lane in latched["lanes"].items()
    }

    restored = runner.apply_meta_substrate_recovery(
        meta_request_sha256="a" * 64,
        meta_response_sha256="b" * 64,
        meta_terminal_sha256="c" * 64,
        recommendation=R.META_SUBSTRATE_RECOVERY_RECOMMENDATION,
    )

    assert len(backend.health_probe_calls) == 2
    assert restored["operator_incident"] is None
    assert restored["substrate_incident"] is None
    assert restored["settled_cost_units"] == settled_before
    assert {
        game: lane["no_progress"]
        for game, lane in restored["lanes"].items()
    } == no_progress_before
    assert restored["failure_operation_circuits"][
        "substrate_health_reprobe:controller_substrate"
    ]["consecutive"] == 0
    assert restored["failure_domain_circuits"][
        "controller_substrate"
    ]["consecutive"] == 0
    kinds = [
        event["kind"]
        for event in runner.journal._read_authenticated()
    ]
    assert kinds.count("META_SUBSTRATE_RECOVERY_AUTHORIZED") == 1
    assert kinds.count("META_SUBSTRATE_HEALTH_RESTORED") == 1
    assert kinds.count("META_SUBSTRATE_RESUME_AUTHORIZED") == 1
    R.Scheduler.validate_journal_event_sequence(
        runner.journal.read()
    )


@pytest.mark.parametrize(
    ("byte_admitted", "inode_admitted", "error_code"),
    (
        (False, True, "insufficient_bytes"),
        (True, False, "insufficient_inodes"),
    ),
)
def test_filesystem_admission_commits_reachable_storage_incident_without_launch(
    tmp_path,
    monkeypatch,
    byte_admitted,
    inode_admitted,
    error_code,
):
    runner, backend, _, _, _ = make_runner(
        tmp_path, max_lanes=2
    )
    baseline = runner.journal.filesystem_admission_snapshot(
        required_event_bytes=1
    )
    blocked = {
        **baseline,
        "byte_admitted": byte_admitted,
        "inode_admitted": inode_admitted,
    }
    monkeypatch.setattr(
        runner.journal,
        "filesystem_admission_snapshot",
        lambda **_kwargs: dict(blocked),
    )

    report = runner.cycle()
    state = runner.state()

    assert backend.launch_calls == []
    assert state["storage_incident"] is not None
    assert state["storage_incident"]["error_code"] == error_code
    assert report["storage_incident"] == state["storage_incident"]
    assert not (
        runner.journal.emergency_reserve_path.exists()
        or runner.journal.emergency_reserve_path.is_symlink()
    )
    R.Scheduler.validate_journal_event_sequence(
        runner.journal.read()
    )


def test_enospc_during_event_commit_consumes_reserve_and_latches_campaign(
    tmp_path,
    monkeypatch,
):
    runner, backend, _, _, _ = make_runner(
        tmp_path, max_lanes=2
    )
    original = R._write_new_file
    injected = {"done": False}

    def fail_one_journal_commit(path, value):
        selected = Path(path)
        if (
            not injected["done"]
            and selected.parent == runner.journal.root
            and selected.name.startswith(".pending-")
        ):
            injected["done"] = True
            raise OSError(errno.ENOSPC, "fault-injected ENOSPC")
        return original(selected, value)

    monkeypatch.setattr(R, "_write_new_file", fail_one_journal_commit)
    report = runner.cycle()
    state = runner.state()

    assert injected["done"] is True
    assert backend.launch_calls == []
    assert state["storage_incident"]["error_code"] == "ENOSPC"
    assert state["storage_incident"]["failure_stage"] == "event_commit"
    assert report["storage_incident"] == state["storage_incident"]
    assert not list(runner.journal.root.glob(".pending-*"))
    R.Scheduler.validate_journal_event_sequence(
        runner.journal.read()
    )


def test_storage_incident_quiesces_live_primary_once_without_collection_or_authority(
    tmp_path,
    monkeypatch,
):
    runner, backend, gate, _, _ = make_runner(
        tmp_path, max_lanes=1
    )
    runner.cycle()
    live = runner.state()
    attempt_id, attempt = next(iter(live["attempts"].items()))
    assert attempt["phase"] == "RUNNING"
    baseline = runner.journal.filesystem_admission_snapshot(
        required_event_bytes=1
    )
    monkeypatch.setattr(
        runner.journal,
        "filesystem_admission_snapshot",
        lambda **_kwargs: {
            **baseline,
            "byte_admitted": False,
            "inode_admitted": True,
        },
    )

    report = runner.cycle()
    state = runner.state()
    assert backend.emergency_containment_calls == [
        (attempt_id, "RUNNING")
    ]
    assert backend.collections == {}
    assert backend.teardown_calls == []
    assert gate.calls == []
    assert report["effective_live_primary_children"] == 0
    assert report["effective_live_auxiliary_children"] == 0
    assert state["storage_incident"] is not None
    assert state["storage_quiescence"] is not None
    assert all(
        state["storage_quiescence"][name] is False
        for name in (
            "solver_authority",
            "wip_authority",
            "cost_authority",
            "promotion_authority",
        )
    )
    assert not (
        runner.journal.quiescence_reserve_path.exists()
        or runner.journal.quiescence_reserve_path.is_symlink()
    )

    repeated = runner.cycle()
    assert repeated["storage_emergency_stage_trace"] == [
        "verify_storage_emergency_quiescence"
    ]
    assert backend.emergency_containment_calls == [
        (attempt_id, "RUNNING")
    ]
    R.Scheduler.validate_journal_event_sequence(
        runner.journal.read()
    )


def test_journal_rolls_to_authenticated_checkpoint_segment_and_recovers_open_cut(
    tmp_path,
):
    campaign = tmp_path / "segmented-campaign"
    journal = R.DurableAttemptJournal(
        campaign / "attempt_journal"
    )
    for index in range(R.JOURNAL_SEGMENT_EVENT_LIMIT):
        journal.append(
            event_id=f"fixture:{index:04d}",
            kind="FIXTURE",
            payload={"index": index},
            recorded_at=float(index),
        )
    events = journal._read_authenticated()
    assert len(events) == R.JOURNAL_SEGMENT_EVENT_LIMIT
    # Crash cut after prior closure/new-segment genesis but before the first
    # event commit. Restart must reopen both immutable receipts and append the
    # same next hash-chain coordinate.
    segment = journal._ensure_segment_for_append(
        sequence=R.JOURNAL_SEGMENT_EVENT_LIMIT + 1,
        events=events,
    )
    assert segment.name == "segment-00000002"
    restarted = R.DurableAttemptJournal(
        campaign / "attempt_journal"
    )
    restarted.append(
        event_id="fixture:rollover",
        kind="FIXTURE",
        payload={"index": R.JOURNAL_SEGMENT_EVENT_LIMIT},
        recorded_at=float(R.JOURNAL_SEGMENT_EVENT_LIMIT),
    )

    reopened = restarted._read_authenticated()
    assert len(reopened) == R.JOURNAL_SEGMENT_EVENT_LIMIT + 1
    assert reopened[-1]["previous_digest"] == reopened[-2]["digest"]
    closure = R._read_json_file(
        campaign
        / "attempt_journal"
        / ".segment-00000001-closure.json"
    )
    checkpoint = R._read_json_file(
        segment / ".checkpoint.json"
    )
    assert closure["last_event_digest"] == reopened[-2]["digest"]
    assert (
        checkpoint["previous_segment_closure_sha256"]
        == R._sha256_file(
            campaign
            / "attempt_journal"
            / ".segment-00000001-closure.json"
        )
    )
    assert R.Scheduler.read_journal(campaign) == reopened


def test_journal_segment_controls_reject_rehashed_tamper_and_repair_mkdir_cut(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(R, "JOURNAL_SEGMENT_EVENT_LIMIT", 4)
    campaign = tmp_path / "segment-control-campaign"
    journal = R.DurableAttemptJournal(
        campaign / "attempt_journal"
    )
    for index in range(4):
        journal.append(
            event_id=f"small:{index}",
            kind="FIXTURE",
            payload={"index": index},
            recorded_at=float(index),
        )
    events = journal._read_authenticated()
    closure_path = (
        journal.root / ".segment-00000001-closure.json"
    )
    journal._close_segment(segment_number=1, events=events)
    directory_only = journal.root / "segment-00000002"
    directory_only.mkdir(mode=0o700)
    R._fsync_directory(journal.root)

    # mkdir is a safe authority-free crash cut.  Restart authenticates the
    # prior closure and the next append installs the exact genesis receipt.
    restarted = R.DurableAttemptJournal(journal.root)
    assert restarted.read() == events
    restarted.append(
        event_id="small:4",
        kind="FIXTURE",
        payload={"index": 4},
        recorded_at=4.0,
    )
    checkpoint_path = directory_only / ".checkpoint.json"
    assert checkpoint_path.exists()

    # Rehashing both mutable-looking control files cannot rewrite history:
    # the closure contents are recomputed from authenticated event digests.
    closure = R._read_json_file(closure_path)
    closure["event_inventory_sha256"] = "f" * 64
    os.chmod(closure_path, 0o600)
    closure_path.write_bytes(R._canonical_json(closure) + b"\n")
    os.chmod(closure_path, 0o400)
    checkpoint = R._read_json_file(checkpoint_path)
    checkpoint["previous_segment_closure_sha256"] = R._sha256_file(
        closure_path
    )
    os.chmod(checkpoint_path, 0o600)
    checkpoint_path.write_bytes(
        R._canonical_json(checkpoint) + b"\n"
    )
    os.chmod(checkpoint_path, 0o400)

    with pytest.raises(
        R.ContiguousRunnerError,
        match="segment closure changed",
    ):
        restarted.read()
    with pytest.raises(
        R.ContiguousRunnerError,
        match="segment closure changed",
    ):
        R.DurableAttemptJournal(journal.root).read()


def test_production_l1_uses_only_control_hashed_blank_scaffold(tmp_path):
    builder = R.ProductionInputBundleBuilder()
    runner, backend, _, _, _ = make_runner(
        tmp_path,
        builder=builder,
        max_lanes=1,
    )
    runner.cycle()
    spec = next(iter(backend.specs.values()))
    assert (
        spec.parent_source_tree_sha256
        == R.CANONICAL_BLANK_SCAFFOLD_TREE_SHA256
    )
    assert (
        Contract._tree_hash(Path(spec.parent_source_path))
        == R.CANONICAL_BLANK_SCAFFOLD_TREE_SHA256
    )
    assert (
        Contract._tree_hash(Path(spec.input_dir) / "parent_source")
        == spec.parent_source_tree_sha256
    )
    source_bytes = b"\n".join(
        path.read_bytes()
        for path in sorted(
            (Path(spec.input_dir) / "parent_source").iterdir()
        )
    )
    assert all(
        game.encode("ascii") not in source_bytes
        for game in Contract.authoritative_inventory()
    )


def test_source_schema_closes_imports_over_exact_flat_source_set():
    payloads = {
        "legs.py": (
            b"from collections import deque\n"
            b"from helpers import normalize\n"
            b"import numpy.linalg as linear_algebra\n"
        ),
        "players.py": b"from legs import *\n",
        "solve.py": b"import players\n",
        "helpers.py": b"def normalize(value):\n    return value\n",
        "policy_data.json": b'{}\n',
    }
    assert R.SourceSchema.validate_source_payloads(payloads) == (
        "helpers.py",
        "legs.py",
        "players.py",
        "policy_data.json",
        "solve.py",
    )
    assert R.SourceSchema.PINNED_NUMPY_VERSION == "2.4.4"


@pytest.mark.parametrize(
    "forbidden_import",
    (
        "from .players import play\n",
        "from arc.crack_lab import gkm_arena\n",
        "import environment_files\n",
        "import unknown_ambient_solver_package\n",
    ),
)
def test_source_schema_rejects_nonclosed_import_roots(
    forbidden_import,
):
    payloads = {
        "legs.py": forbidden_import.encode("utf-8"),
        "players.py": b"from legs import *\n",
        "solve.py": b"import players\n",
    }
    with pytest.raises(
        R.SourceSchema.SourceSchemaError,
        match="relative source import|undeclared ambient root",
    ):
        R.SourceSchema.validate_source_payloads(payloads)


def test_promoted_winning_source_is_exact_next_level_parent_after_restart(
    tmp_path,
):
    builder = R.ProductionInputBundleBuilder()
    backend = FakeBackend(candidate_result)
    gate = FakePromotionGate(tmp_path / "promotions")
    runner, _, _, _, _ = make_runner(
        tmp_path,
        backend=backend,
        gate=gate,
        builder=builder,
        max_lanes=1,
    )
    runner.cycle()
    level_one = next(iter(backend.specs.values()))
    game = level_one.game
    backend.strategy = lambda spec: (
        candidate_result(spec)
        if spec.game == game
        else _signed_test_blocker_result(spec)
    )
    sentinel = b"\nPROMOTED_SOURCE_SENTINEL = True\n"
    solve_path = Path(level_one.workspace_dir) / "solve.py"
    solve_path.write_bytes(solve_path.read_bytes() + sentinel)
    runner.cycle()
    commit = gate.commits[level_one.attempt_id]
    state = runner.state()
    assert state["lanes"][game]["reached"] == 1
    promoted_source = (
        Path(commit.checkpoint_path).parent
        / Contract.WINNING_SOURCE_NAME
    )
    assert state["lanes"][game]["source_path"] == str(
        promoted_source
    )
    assert (
        state["lanes"][game]["source_tree_sha256"]
        == commit.source_tree_sha256
        == Contract._tree_hash(promoted_source)
    )

    recovered = R.ContiguousCampaignRunner(
        tmp_path / "campaign",
        backend=backend,
        promotion_gate=gate,
        input_builder=builder,
        backend_configuration=backend_configuration(),
        cost_window_id=COST_WINDOW_ID,
        max_lanes=1,
        controller_state_canaries=backend.controller_state_canaries,
        id_factory=ids(),
    )
    recovered_state = recovered.state()
    recovered_lane = recovered_state["lanes"][game]
    assert recovered_lane["reached"] == 1
    assert recovered_lane["source_path"] == str(promoted_source)
    snapshot = recovered._scheduler_snapshot_from_state(
        recovered_state
    )
    next_level_frontier = next(
        frontier
        for frontier in snapshot.frontiers
        if frontier.game == game
    )
    assert next_level_frontier.reached == 1
    assert next_level_frontier.parent_source_path == str(
        promoted_source
    )
    assert (
        next_level_frontier.parent_source_tree_sha256
        == commit.source_tree_sha256
    )
    assert sentinel in (
        Path(next_level_frontier.parent_source_path) / "solve.py"
    ).read_bytes()

    prior_events = recovered.journal.read()
    blocked_identities = (
        set(recovered_state["used_scheduler_identifiers"])
        | set(recovered_state["used_auxiliary_thread_ids"])
        | {recovered_state["campaign_id"]}
    )
    recovery_report = recovered.cycle()
    assert recovery_report["recoverable_errors"] == []
    new_decisions = [
        R.Scheduler.decision_from_dict(event["payload"]["decision"])
        for event in recovered.journal.read()[len(prior_events):]
        if event["kind"] == "SCHEDULER_DECISION"
    ]
    assert len(new_decisions) == 1
    new_decision = new_decisions[0]
    assert {
        new_decision.decision_id,
        new_decision.attempt_id,
        new_decision.generation_id,
        new_decision.reservation_id,
    }.isdisjoint(blocked_identities)
    assert (
        len({
            new_decision.decision_id,
            new_decision.attempt_id,
            new_decision.generation_id,
            new_decision.reservation_id,
        })
        == 4
    )


def stat_mode(path: Path) -> int:
    return os.stat(path, follow_symlinks=False).st_mode & 0o777


def test_limit_none_disables_uniform_cost_cutoff(tmp_path):
    unlimited, backend, _, _, _ = make_runner(
        tmp_path / "unlimited", limit=None
    )
    report = unlimited.cycle()
    assert report["cost_control_enabled"] is False
    assert all(
        spec.cost_limit_remaining is None
        for spec in backend.specs.values()
    )
    finite, finite_backend, _, _, _ = make_runner(
        tmp_path / "finite", limit=0
    )
    assert finite.cycle()["cost_control_enabled"] is True
    assert finite_backend.specs == {}


def test_auxiliary_dispatch_is_default_off_and_fails_closed(tmp_path):
    assert R.CONTIGUOUS_AUXILIARY_LAUNCH_READY is False
    proposer = FakeBackend()
    common = {
        "backend": proposer,
        "promotion_gate": FakePromotionGate(tmp_path / "promotions"),
        "input_builder": FakeInputBuilder(),
        "backend_configuration": backend_configuration(),
        "cost_window_id": COST_WINDOW_ID,
        "max_lanes": 1,
        "controller_state_canaries":
            proposer.controller_state_canaries,
        "id_factory": ids(),
    }
    with pytest.raises(
        R.ContiguousRunnerError, match="has no backend"
    ):
        R.ContiguousCampaignRunner(
            tmp_path / "missing-auxiliary-backend",
            auxiliary_launch_configuration=(
                FakeAuxiliaryBackend.configuration()
            ),
            **common,
        )
    auxiliary = FakeAuxiliaryBackend(tmp_path / "auxiliary")
    auxiliary.backend_contract_sha256 = "f" * 64
    with pytest.raises(
        R.ContiguousRunnerError, match="exact attested"
    ):
        R.ContiguousCampaignRunner(
            tmp_path / "wrong-auxiliary-contract",
            auxiliary_backend=auxiliary,
            auxiliary_launch_configuration=(
                FakeAuxiliaryBackend.configuration()
            ),
            **common,
        )


def test_soft_deadline_drains_lane_locally_and_refills_unrelated_lane(tmp_path):
    clock = Clock()
    runner, backend, _, _, _ = make_runner(
        tmp_path, max_lanes=2, clock=clock
    )
    initial = runner.cycle()
    assert tuple(initial["supervision_stage_trace"]) == (
        R.Scheduler.SUPERVISION_CYCLE_STAGES
    )
    attempt_ids = list(backend.specs)
    clock.advance(15 * 60 + 1)
    assert runner.cycle()["draining"] is True
    assert len(backend.specs) == 2
    backend.finish(
        attempt_ids[0], R.AttemptResult(kind="clean_no_progress")
    )
    runner.cycle()
    assert runner.state()["draining"] is True
    assert len(backend.specs) == 3
    assert (
        backend.specs[attempt_ids[1]].game
        != backend.specs[list(backend.specs)[-1]].game
    )
    backend.finish(
        attempt_ids[1], R.AttemptResult(kind="clean_no_progress")
    )
    runner.cycle()
    assert runner.state()["draining"] is False
    assert len(backend.specs) > 2
    assert all(
        timeout == R.POLL_TIMEOUT_SECONDS
        for timeout in backend.poll_timeouts
    )


def test_unchanged_primary_poll_is_coalesced_across_360_minutes_and_restart(
    tmp_path,
):
    clock = Clock()
    runner, backend, gate, builder, _ = make_runner(
        tmp_path, max_lanes=1, clock=clock
    )
    runner.cycle()
    attempt_id = next(iter(backend.specs))
    projected_polls = int((360 * 60) / 0.05)
    assert projected_polls == 432_000

    # The first authenticated RUNNING sample is retained.  Subsequent
    # identical samples, including samples after the soft deadline, do not
    # consume journal rows merely because wall time advances.
    runner.cycle()
    clock.advance(360 * 60)
    runner.cycle()
    recovered = R.ContiguousCampaignRunner(
        tmp_path / "campaign",
        backend=backend,
        promotion_gate=gate,
        input_builder=builder,
        backend_configuration=backend_configuration(),
        cost_window_id=COST_WINDOW_ID,
        max_lanes=1,
        controller_state_canaries=backend.controller_state_canaries,
        clock=clock,
        id_factory=ids(),
    )
    recovered.cycle()

    attempt = recovered.state()["attempts"][attempt_id]
    assert attempt["phase"] == "DRAINING"
    assert attempt["observation_count"] == 1
    assert attempt["last_observation_sha256"] == "6" * 64
    observed_events = [
        event
        for event in recovered.journal.read()
        if event["kind"] == "ATTEMPT_OBSERVED"
        and event["payload"]["attempt_id"] == attempt_id
    ]
    assert len(observed_events) == 1


def test_real_cycle_rejects_frozen_supervision_stage_reordering(
    tmp_path,
    monkeypatch,
):
    frozen = tuple(
        Conformance.launch_requirements_snapshot()["body"][
            "supervision_cycle_stages"
        ]
    )
    assert frozen == R.Scheduler.SUPERVISION_CYCLE_STAGES
    runner, backend, _, _, _ = make_runner(
        tmp_path, max_lanes=1
    )
    journal_before = runner.journal.read()
    reordered = (
        frozen[1],
        frozen[0],
        *frozen[2:],
    )
    monkeypatch.setattr(
        R.Scheduler,
        "SUPERVISION_CYCLE_STAGES",
        reordered,
    )
    with pytest.raises(
        R.ContiguousRunnerError,
        match="departed from policy order",
    ):
        runner.cycle()
    assert runner.journal.read() == journal_before
    assert backend.specs == {}


def test_restart_after_deadline_polls_then_launches_other_prepared_lane(
    tmp_path,
):
    clock = Clock()
    backend = FakeBackend()
    backend.fail_prepare_after = 1
    runner, backend, gate, builder, _ = make_runner(
        tmp_path, backend=backend, max_lanes=2, clock=clock
    )
    runner.cycle()
    running = next(iter(backend.specs))
    state = runner.state()
    assert sum(
        attempt["phase"] == "PREPARED"
        for attempt in state["attempts"].values()
    ) == 1
    calls_before = len(backend.launch_calls)
    backend.fail_prepare_after = None
    clock.advance(15 * 60 + 1)
    recovered = R.ContiguousCampaignRunner(
        tmp_path / "campaign",
        backend=backend,
        promotion_gate=gate,
        input_builder=builder,
        backend_configuration=backend_configuration(),
        cost_window_id=COST_WINDOW_ID,
        max_lanes=2,
        controller_state_canaries=backend.controller_state_canaries,
        clock=clock,
        id_factory=ids(),
    )
    assert recovered.cycle()["draining"] is True
    assert len(backend.launch_calls) == calls_before + 1
    assert recovered.state()["attempts"][running]["phase"] == "DRAINING"
    assert sum(
        attempt["phase"] == "RUNNING"
        for attempt in recovered.state()["attempts"].values()
    ) == 1


def test_retained_prepare_crash_uses_fresh_generation_and_bounds_repeats(
    tmp_path: Path,
):
    backend = QuarantinedPreparationBackend(quarantine_count=1)
    runner, _, _, _, clock = make_runner(
        tmp_path / "one", backend=backend, max_lanes=1
    )
    first_report = runner.cycle()
    first_state = runner.state()
    assert first_report["started_attempts"] == []
    assert len(backend.quarantined_specs) == 1
    old_spec = backend.quarantined_specs[0]
    old_attempt = first_state["attempts"][old_spec.attempt_id]
    assert old_attempt["phase"] == "CLOSED"
    assert old_attempt["settled_result"] == R.AttemptResult(
        kind="infrastructure",
        cost_used=0.0,
        reason="compatibility_preparation_quarantined",
    )
    assert first_state["lanes"][old_spec.game]["active"] is None
    old_receipt = Path(
        old_attempt["preparation_quarantine"]["path"]
    )
    old_receipt_sha256 = hashlib.sha256(
        old_receipt.read_bytes()
    ).hexdigest()
    old_staging_observation = (
        R.CompatibilityClosure.observe_quarantined_staging(
            Path(old_spec.host_transcript_path).parent
            / "compatibility_arena_closure"
        )
    )

    # The dispatch pass stops after one quarantine, so even a zero first
    # backoff cannot churn identities inside the same supervision cycle.
    assert len(first_state["attempts"]) == 1
    second_report = runner.cycle()
    second_state = runner.state()
    assert len(second_report["started_attempts"]) == 1
    fresh_id = second_report["started_attempts"][0]
    fresh = second_state["attempts"][fresh_id]
    assert fresh["phase"] == "RUNNING"
    assert fresh["spec"].generation_id != old_spec.generation_id
    assert fresh["spec"].attempt_id != old_spec.attempt_id
    assert (
        Path(fresh["spec"].host_transcript_path).parent
        != Path(old_spec.host_transcript_path).parent
    )
    assert hashlib.sha256(old_receipt.read_bytes()).hexdigest() == (
        old_receipt_sha256
    )
    assert (
        R.CompatibilityClosure.observe_quarantined_staging(
            Path(old_spec.host_transcript_path).parent
            / "compatibility_arena_closure"
        )
        == old_staging_observation
    )

    repeated_backend = QuarantinedPreparationBackend(
        quarantine_count=R.FAILURE_CIRCUIT_THRESHOLD
    )
    repeated, _, _, _, repeated_clock = make_runner(
        tmp_path / "repeated",
        backend=repeated_backend,
        max_lanes=1,
    )
    for _index in range(R.FAILURE_CIRCUIT_THRESHOLD):
        repeated.cycle()
        repeated_clock.advance(
            max(R.OPERATION_RETRY_BACKOFF_SECONDS) + 1
        )
    repeated_state = repeated.state()
    assert len(repeated_backend.quarantined_specs) == (
        R.FAILURE_CIRCUIT_THRESHOLD
    )
    assert repeated_state["operator_incident"] is not None
    assert repeated_state["operator_incident"]["operation"] == (
        "backend_prepare"
    )
    retained_count = len(repeated_state["attempts"])
    repeated.cycle()
    assert len(repeated.state()["attempts"]) == retained_count
    assert all(
        attempt["phase"] == "CLOSED"
        for attempt in repeated.state()["attempts"].values()
    )


def test_escalation_is_frontier_bound_and_missing_wip_falls_back_to_exclude(
    tmp_path, monkeypatch
):
    target_game = sorted(Contract.authoritative_inventory())[0]
    specs: list[R.AttemptSpec] = []
    verified_decisions = 0
    original_verify_decision = R.Scheduler.verify_decision

    def verify_decision_once(snapshot, decision):
        nonlocal verified_decisions
        verified_decisions += 1
        return original_verify_decision(snapshot, decision)

    monkeypatch.setattr(
        R.Scheduler, "verify_decision", verify_decision_once
    )

    def strategy(spec):
        if spec.game != target_game:
            return _signed_test_blocker_result(spec)
        specs.append(spec)
        return R.AttemptResult(kind="clean_no_progress")

    runner, _, _, _, _ = make_runner(
        tmp_path, backend=FakeBackend(strategy), max_lanes=6
    )
    for _ in range(100):
        runner.cycle()
        if len(specs) >= 7:
            break
    assert [
        (s.effort, s.soft_allocation_seconds // 60, s.wip_mode)
        for s in specs[:7]
    ] == [
        ("medium", 15, "exclude"),
        ("high", 20, "exclude"),
        ("xhigh", 25, "exclude"),
        ("xhigh", 40, "exclude"),
        ("max", 60, "exclude"),
        ("max", 90, "exclude"),
        ("max", 120, "exclude"),
    ]
    decisions = [
        R.Scheduler.decision_from_dict(event["payload"]["decision"])
        for event in runner.journal.read()
        if event["kind"] == "SCHEDULER_DECISION"
        and event["payload"]["decision"]["choice"]["game"] == target_game
    ]
    assert [
        item.choice.requested_wip_mode for item in decisions[:7]
    ] == [
        "exclude",
        "restore_clean_same_frontier",
        "restore_clean_same_frontier",
        "restore_clean_same_frontier",
        "restore_clean_same_frontier",
        "exclude",
        "restore_clean_same_frontier",
    ]
    assert len({s.generation_dir for s in specs}) == len(specs)
    all_decision_events = sum(
        event["kind"] == "SCHEDULER_DECISION"
        for event in runner.journal.read()
    )
    # Bound authenticated work directly. Wall-clock assertions are not
    # admissible launch evidence because unrelated host load can change them.
    # Every durable decision is verified exactly once, and cached state reads
    # perform no additional decision verification.
    assert verified_decisions == all_decision_events
    before_cached_reads = verified_decisions
    for _ in range(3):
        runner.state()
    assert verified_decisions == before_cached_reads


def test_mismatched_wip_is_dropped_but_clean_failure_counts(tmp_path):
    target_game = sorted(Contract.authoritative_inventory())[0]
    specs: list[R.AttemptSpec] = []

    def strategy(spec):
        if spec.game != target_game:
            return _signed_test_blocker_result(spec)
        specs.append(spec)
        wip = _identity_fixture_wip(spec)
        return R.AttemptResult(
            kind="clean_no_progress",
            wip=replace(
                wip,
                parent_checkpoint_sha256="f" * 64,
            ),
        )

    runner, _, _, _, _ = make_runner(
        tmp_path, backend=FakeBackend(strategy), max_lanes=6
    )
    for _ in range(50):
        runner.cycle()
        if len(specs) >= 2:
            break
    assert len(specs) >= 2
    assert specs[1].effort == "high"
    assert specs[1].wip is None
    assert specs[1].wip_mode == "exclude"
    assert runner.state()["lanes"][target_game]["wip"] is None


def test_infrastructure_outcomes_do_not_escalate_effort(tmp_path):
    target_game = sorted(Contract.authoritative_inventory())[0]
    specs: list[R.AttemptSpec] = []

    def strategy(spec):
        if spec.game != target_game:
            return _signed_test_blocker_result(spec)
        specs.append(spec)
        return R.AttemptResult(kind="infrastructure", reason="retry")

    runner, _, _, _, _ = make_runner(
        tmp_path, backend=FakeBackend(strategy), max_lanes=1
    )
    for _ in range(50):
        runner.cycle()
        if len(specs) >= 2:
            break
    assert [spec.effort for spec in specs[:2]] == [
        "medium", "medium"
    ]
    assert runner.state()["lanes"][target_game]["no_progress"] == 0


@pytest.mark.parametrize(
    "outcome",
    (
        "tainted",
        "infrastructure",
        "blocker",
        "capacity",
        "rate_limit",
        "provider_failure",
        "containment_fault",
    ),
)
def test_only_clean_settlement_advances_shared_complexity_coordinate(
    outcome,
):
    assert R.advance_exact_frontier_clean_no_progress(5, outcome) == 5
    assert (
        R.advance_exact_frontier_clean_no_progress(
            5, "clean_no_progress"
        )
        == 6
    )
    shared = R.Scheduler.frontier_complexity_schedule(
        5, frontier_sha256="a" * 64
    )
    assert shared.primary == R.Scheduler.retry_policy(5)
    assert shared.primary.effort == "max"
    assert shared.auxiliary.phase == "diagnose"
    assert shared.auxiliary.max_parallel == 1
    expanded = R.Scheduler.frontier_complexity_schedule(
        7, frontier_sha256="a" * 64
    )
    assert expanded.primary == R.Scheduler.retry_policy(7)
    assert expanded.primary.effort == "max"
    assert expanded.auxiliary.max_parallel == 1
    # Without an admitted profile, n=7 still permits only one diagnosis.
    assert expanded.auxiliary.specializations == (
        "complexity_diagnosis",
    )


def test_containment_fault_precedes_blocker_candidate_and_no_progress(
    tmp_path,
):
    class ContainmentBackend(FakeBackend):
        def poll(
            self,
            *,
            spec,
            prepared,
            launched,
            timeout_seconds,
        ):
            assert prepared == self.preparations[spec.attempt_id]
            assert launched == self.launches[spec.attempt_id]
            self.poll_timeouts.append(timeout_seconds)
            if spec.attempt_id in self.results:
                return R.BackendPoll(
                    status="containment_fault",
                    observation_sha256="5" * 64,
                    exit_code=137,
                )
            return R.BackendPoll(
                status="running", observation_sha256="6" * 64
            )

    preserved = R.apply_terminal_result_precedence(
        "containment_fault",
        R.AttemptResult(
            kind="blocker",
            cost_used=7.25,
            reason="untrusted blocker label",
        ),
    )
    assert preserved.kind == "infrastructure"
    assert preserved.cost_used == 7.25
    assert preserved.candidate is None
    assert preserved.wip is None
    tainted = R.AttemptResult(kind="tainted", reason="strict scan")
    assert R.apply_terminal_result_precedence(
        "containment_fault", tainted
    ) == tainted

    for result_kind in (
        "blocker",
        "candidate",
        "clean_no_progress",
    ):
        def strategy(spec, kind=result_kind):
            if kind == "candidate":
                return candidate_result(spec)
            return R.AttemptResult(
                kind=kind,
                reason=f"synthetic {kind}",
            )

        backend = ContainmentBackend(strategy)
        runner, backend, gate, _, _ = make_runner(
            tmp_path / result_kind,
            backend=backend,
            max_lanes=1,
        )
        runner.cycle()
        attempt_id = next(iter(backend.specs))
        runner.cycle()
        result_events = [
            event
            for event in runner.journal.read()
            if event["kind"] == "ATTEMPT_RESULT"
            and event["payload"]["attempt_id"] == attempt_id
        ]
        assert len(result_events) == 1
        assert result_events[0]["payload"]["kind"] == "infrastructure"
        assert result_events[0]["payload"]["candidate"] is None
        assert result_events[0]["payload"]["wip"] is None
        state = runner.state()
        lane = state["lanes"][backend.specs[attempt_id].game]
        assert lane["no_progress"] == 0
        assert lane["blocked"] is None
        assert lane["reached"] == 0
        assert gate.calls == []
        assert state["attempts"][attempt_id][
            "settled_result"
        ].kind == "infrastructure"


BLOCKER_NEGATIVE_CASES = (
    "missing_evidence",
    "unknown_code",
    "malformed_receipt",
    "unsigned_receipt",
    "replayed_receipt",
    "wrong_attempt",
    "wrong_binding",
    "wrong_frontier",
    "wrong_target",
    "wrong_host_result",
    "wrong_path_hash",
)


def _assert_blocker_claim_is_noncounting_infrastructure(
    tmp_path: Path,
    case: str,
) -> None:
    replayed: R.AttemptResult | None = None
    if case == "replayed_receipt":
        foreign_backend = FakeBackend(
            lambda spec: _signed_test_blocker_result(spec)
        )
        foreign, _, _, _, _ = make_runner(
            tmp_path / "foreign",
            backend=foreign_backend,
            max_lanes=1,
        )
        foreign.cycle()
        replayed = next(iter(foreign_backend.results.values()))

    def claim(spec):
        if replayed is not None:
            return replayed
        result = _signed_test_blocker_result(spec)
        assert result.blocker is not None
        if case == "missing_evidence":
            return replace(result, blocker=None)
        if case == "unknown_code":
            return replace(
                result,
                blocker=replace(
                    result.blocker, code="model_declared_impossible"
                ),
            )
        if case == "malformed_receipt":
            return _rewrite_test_blocker_receipt(
                result,
                mutate=lambda value: value.clear(),
                resign=False,
            )
        if case == "unsigned_receipt":
            return _rewrite_test_blocker_receipt(
                result,
                mutate=lambda value: value.pop(
                    "host_authentication_sha256", None
                ),
                resign=False,
            )
        if case == "wrong_attempt":
            return _rewrite_test_blocker_receipt(
                result,
                mutate=lambda value: value.__setitem__(
                    "attempt_id", str(uuid.uuid4())
                ),
                resign=True,
            )
        if case == "wrong_binding":
            return _rewrite_test_blocker_receipt(
                result,
                mutate=lambda value: value.__setitem__(
                    "arena_binding_sha256", "1" * 64
                ),
                resign=True,
            )
        if case == "wrong_frontier":
            return _rewrite_test_blocker_receipt(
                result,
                mutate=lambda value: value.__setitem__(
                    "frontier_sha256", "2" * 64
                ),
                resign=True,
            )
        if case == "wrong_target":
            return _rewrite_test_blocker_receipt(
                result,
                mutate=lambda value: value.__setitem__(
                    "target_level", spec.target_level + 1
                ),
                resign=True,
            )
        if case == "wrong_host_result":
            def mutate_host_result(value):
                value["arena_host_result"]["levels_completed"] = (
                    spec.target_level
                )
                value["arena_host_result_sha256"] = hashlib.sha256(
                    Transport.canonical_json(
                        value["arena_host_result"]
                    )
                ).hexdigest()

            return _rewrite_test_blocker_receipt(
                result,
                mutate=mutate_host_result,
                resign=True,
            )
        if case == "wrong_path_hash":
            return _rewrite_test_blocker_receipt(
                result,
                mutate=lambda value: value.__setitem__(
                    "parent_path_sha256", "3" * 64
                ),
                resign=True,
            )
        raise AssertionError(case)

    backend = FakeBackend(claim)
    runner, _, gate, _, _ = make_runner(
        tmp_path / "subject",
        backend=backend,
        max_lanes=1,
    )
    runner.cycle()
    attempt_id = next(iter(backend.specs))
    game = backend.specs[attempt_id].game
    runner.cycle()
    state = runner.state()
    settled = state["attempts"][attempt_id]["settled_result"]
    assert settled.kind == "infrastructure"
    assert settled.blocker is None
    assert state["lanes"][game]["blocked"] is None
    assert state["lanes"][game]["no_progress"] == 0
    assert state["lanes"][game]["reached"] == 0
    assert gate.calls == []
    result_events = [
        event for event in runner.journal.read()
        if event["kind"] == "ATTEMPT_RESULT"
        and event["payload"]["attempt_id"] == attempt_id
    ]
    assert len(result_events) == 1
    assert result_events[0]["payload"]["kind"] == "infrastructure"
    assert result_events[0]["payload"]["blocker"] is None


def test_blocker_claim_negative_cross_product_is_noncounting_infrastructure(
    tmp_path,
):
    for case in BLOCKER_NEGATIVE_CASES:
        _assert_blocker_claim_is_noncounting_infrastructure(
            tmp_path / case, case
        )


def test_authenticated_blocker_recovers_idempotently_and_is_revalidated_closed(
    tmp_path,
):
    backend = FakeBackend(
        lambda spec: _signed_test_blocker_result(spec)
    )
    runner, backend, gate, builder, _ = make_runner(
        tmp_path,
        backend=backend,
        max_lanes=1,
    )
    runner.cycle()
    first_attempt = next(iter(backend.specs))
    first_spec = backend.specs[first_attempt]
    runner.cycle()
    state = runner.state()
    assert state["lanes"][first_spec.game]["blocked"] == (
        "host_blocker:arena_parent_terminal_before_target"
    )
    assert state["lanes"][first_spec.game]["no_progress"] == 0
    assert state["attempts"][first_attempt][
        "settled_result"
    ].kind == "blocker"
    signed_blocker = state["attempts"][first_attempt][
        "settled_result"
    ]
    containment = R.apply_terminal_result_precedence(
        "containment_fault", signed_blocker
    )
    assert containment.kind == "infrastructure"
    assert containment.blocker is None
    tainted = runner._sanitize_result(
        first_spec,
        replace(
            signed_blocker,
            kind="tainted",
            reason="trusted scan taint",
        ),
    )
    assert tainted.kind == "tainted"
    assert tainted.blocker is None
    assert gate.calls == []
    audit = R.Scheduler.audit_campaign(tmp_path / "campaign")
    # The public scheduler entrypoint has no live canary secret and must
    # not mint blocker authority from phase-only evidence.
    assert audit["verdict"] == "FAIL"
    assert "full runner lifecycle audit failed" in audit["findings"][0]
    authorized_runner_audit = R.audit_runner_state_read_only(
        tmp_path / "campaign",
        controller_state_canaries=(
            backend.controller_state_canaries
        ),
    )
    assert authorized_runner_audit["status"] == "PASS"
    assert authorized_runner_audit["journal_head_digest"] == (
        runner.journal.read()[-1]["digest"]
    )

    recovered = R.ContiguousCampaignRunner(
        tmp_path / "campaign",
        backend=backend,
        promotion_gate=gate,
        input_builder=builder,
        backend_configuration=backend_configuration(),
        cost_window_id=COST_WINDOW_ID,
        max_lanes=1,
        controller_state_canaries=backend.controller_state_canaries,
        id_factory=ids(),
    )
    events_before = recovered.journal.read()
    assert recovered.state()["attempts"][first_attempt][
        "settled_result"
    ].kind == "blocker"
    assert recovered.state()["lanes"][first_spec.game]["blocked"]
    assert recovered.journal.read() == events_before
    assert sum(
        event["kind"] == "ATTEMPT_RESULT"
        and event["payload"].get("attempt_id") == first_attempt
        for event in events_before
    ) == 1

    evidence = backend.results[first_attempt].blocker
    assert evidence is not None
    receipt_path = Path(evidence.receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["host_authentication_sha256"] = "0" * 64
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        R.ContiguousRunnerError,
        match="closed blocker authority changed",
    ):
        recovered.state()


def test_automatic_auxiliary_lifecycle_uses_same_clean_frontier_count(
    tmp_path,
):
    auxiliary_backend = FakeAuxiliaryBackend(
        tmp_path / "auxiliary-backend"
    )
    auxiliary_backend.fail_prepare_once = True
    runner = _SyntheticAuxiliaryRunner(
        tmp_path / "campaign", auxiliary_backend
    )
    state = runner.state()
    lane = state["lanes"][runner.target_game]
    assert lane["no_progress"] == 5
    assert [
        item.no_progress_before
        for item in lane["clean_proposer_settlements"]
    ] == list(range(5))
    assert R.Scheduler.retry_policy(lane["no_progress"]).effort == "max"

    # A recoverable preparation failure leaves an auditable reservation.
    first = runner.cycle()
    assert first["recoverable_errors"]
    reserved = [
        item
        for item in runner.state()["auxiliary_assignments"].values()
        if item["state"].phase == "RESERVED"
    ]
    assert len(reserved) == 1
    assert reserved[0]["state"].specialization == "complexity_diagnosis"

    # Recovery reuses the exact reservation, manifest, and assignment rather
    # than issuing a second decision.
    second = runner.cycle()
    state = runner.state()
    diagnosis = next(iter(state["auxiliary_assignments"].values()))
    assert diagnosis["state"].phase == "RUNNING", (
        second["recoverable_errors"]
    )
    assert auxiliary_backend.prepare_calls[0] == (
        auxiliary_backend.prepare_calls[1]
    )
    decisions = [
        R.Scheduler.auxiliary_decision_from_dict(
            event["payload"]["decision"]
        )
        for event in runner.journal.read()
        if event["kind"] == "AUXILIARY_DECISION"
    ]
    assert len(decisions) == 1
    assert decisions[0].no_progress == 5
    assert decisions[0].reasoning_effort == "max"
    assert decisions[0].active_proposer_attempt_id == (
        "active:max-proposer"
    )

    # Poll, quarantine, host-admit the diagnosis, then start one orthogonal
    # specialist from the admitted profile.
    diagnosis_admitted = False
    active_specialist_id = None
    for _ in range(8):
        runner.cycle()
        state = runner.state()
        diagnosis_admitted = any(
            item["state"].specialization == "complexity_diagnosis"
            and item["state"].phase == "ADMITTED"
            for item in state["auxiliary_assignments"].values()
        )
        active_specialist_id = next(
            (
                assignment_id
                for assignment_id, item in
                state["auxiliary_assignments"].items()
                if item["state"].specialization
                != "complexity_diagnosis"
                and item["state"].phase == "RUNNING"
            ),
            None,
        )
        if diagnosis_admitted and active_specialist_id is not None:
            break
    assert diagnosis_admitted
    assert active_specialist_id is not None
    assert all(decision.no_progress == 5 for decision in decisions)
    assert all(
        decision.reasoning_effort == "max"
        and decision.active_proposer_attempt_id == "active:max-proposer"
        for decision in decisions
    )

    # Let the first specialist settle into quarantine.  The scheduler may use
    # the now-free sidecar slot for the next orthogonal obligation.
    runner.cycle()
    state = runner.state()
    quarantined = [
        item
        for item in state["auxiliary_assignments"].values()
        if (
            item["state"].specialization != "complexity_diagnosis"
            and item["state"].phase == "QUARANTINED"
        )
    ]
    running = [
        item
        for item in state["auxiliary_assignments"].values()
        if item["state"].phase == "RUNNING"
    ]
    assert quarantined
    assert running

    # Promotion wins without waiting: quarantined evidence is rejected as
    # stale, while the running old-frontier sidecar is contained, torn down,
    # and usage-settled before its capacity is reused.
    old_frontier = runner.old_frontier_sha256
    runner.promote()
    runner.cycle()
    state = runner.state()
    lane = state["lanes"][runner.target_game]
    assert lane["reached"] == 1
    assert lane["no_progress"] == 0
    assert lane["clean_proposer_settlements"] == []
    old_assignments = [
        item
        for item in state["auxiliary_assignments"].values()
        if item["state"].frontier_sha256 == old_frontier
    ]
    assert old_assignments
    assert all(item["state"].invalidated for item in old_assignments)
    assert any(
        item["state"].phase == "REJECTED"
        and item["state"].output is not None
        for item in old_assignments
    )
    assert any(
        item["state"].phase == "ABORTED"
        and item["teardown"] is not None
        for item in old_assignments
    )
    assert not any(
        item["state"].phase in R.Scheduler.AUXILIARY_ACTIVE_PHASES
        for item in old_assignments
    )
    assert any(
        reason == "frontier_promoted" and phase == "RUNNING"
        for _, phase, reason in auxiliary_backend.abort_calls
    )
    kinds = [
        event["kind"] for event in runner.journal.read()
    ]
    for kind in (
        "AUXILIARY_DECISION",
        "AUXILIARY_RESERVED",
        "AUXILIARY_INPUT_PREPARED",
        "AUXILIARY_LAUNCHED",
        "AUXILIARY_RESULT_QUARANTINED",
        "AUXILIARY_PROFILE_ADMITTED",
        "AUXILIARY_OUTPUT_REJECTED",
        "AUXILIARY_ABORTED",
    ):
        assert kind in kinds
    assert state["live_budget_reservations"] == []


def test_backend_cannot_skip_level_and_is_still_torn_down(tmp_path):
    runner, backend, gate, _, _ = make_runner(
        tmp_path,
        backend=FakeBackend(
            lambda spec: candidate_result(spec, to_delta=1)
        ),
        max_lanes=1,
    )
    runner.cycle()
    runner.cycle()
    game = next(iter(backend.specs.values())).game
    assert runner.state()["lanes"][game]["reached"] == 0
    assert gate.calls == []
    kinds = [event["kind"] for event in runner.journal.read()]
    assert "ATTEMPT_COLLECTION_REJECTED" in kinds
    assert "ATTEMPT_TORN_DOWN" in kinds


def test_backend_cannot_override_controller_probe_isolation_mode(
    tmp_path,
):
    class ForgedModeBackend(FakeBackend):
        def prepare(self, spec):
            prepared = super().prepare(spec)
            return replace(
                prepared,
                probe_isolation_mode=(
                    Contract.FRESH_PROCESS_PER_CANDIDATE_MODE
                ),
            )

    backend = ForgedModeBackend()
    runner, _, _, _, _ = make_runner(
        tmp_path, backend=backend, max_lanes=1
    )
    for _ in range(3):
        runner.cycle()
    assert backend.launch_calls == []
    events = runner.journal.read()
    assert "BACKEND_PREPARED" not in {
        event["kind"] for event in events
    }
    assert any(
        event["kind"] == "ATTEMPT_RETRY"
        and event["payload"]["operation"] == "backend_prepare"
        for event in events
    )


def test_primary_operation_retry_backoff_is_durable_and_restart_exact(
    tmp_path,
):
    backend = FakeBackend()
    backend.fail_prepare_after = 0
    runner, _, gate, builder, clock = make_runner(
        tmp_path, backend=backend, max_lanes=1
    )
    runner.cycle()
    assert len(backend.prepare_calls) == 1
    runner.cycle()
    assert len(backend.prepare_calls) == 2
    attempt_id = next(iter(runner.state()["attempts"]))
    retry_events = [
        event
        for event in runner.journal.read()
        if event["kind"] == "ATTEMPT_RETRY"
    ]
    assert [
        (
            event["payload"]["operation_retry_index"],
            event["payload"]["backoff_seconds"],
        )
        for event in retry_events
    ] == [(1, 0.0), (2, 1.0)]

    runner.cycle()
    assert len(backend.prepare_calls) == 2
    recovered = R.ContiguousCampaignRunner(
        tmp_path / "campaign",
        backend=backend,
        promotion_gate=gate,
        input_builder=builder,
        backend_configuration=backend_configuration(),
        cost_window_id=COST_WINDOW_ID,
        max_lanes=1,
        controller_state_canaries=backend.controller_state_canaries,
        clock=clock,
        id_factory=ids(),
    )
    recovered.cycle()
    assert len(backend.prepare_calls) == 2
    clock.advance(1.0)
    recovered.cycle()
    assert len(backend.prepare_calls) == 3
    attempt = recovered.state()["attempts"][attempt_id]
    assert attempt["operation_retry_counts"]["backend_prepare"] == 3
    assert attempt["operation_retry_not_before"][
        "backend_prepare"
    ] == clock.value + 2.0


def test_campaign_global_failure_circuit_exhausts_across_operations_and_restart(
    tmp_path,
):
    runner, backend, gate, builder, clock = make_runner(
        tmp_path, max_lanes=1
    )
    runner.cycle()
    attempt_id = next(iter(runner.state()["attempts"]))
    operations = (
        "backend_prepare",
        "backend_launch",
        "backend_poll",
        "backend_collect",
        "promotion_commit",
        "input_materialize",
    )
    exception_types = tuple(
        type(f"SyntheticFailure{index}", (RuntimeError,), {})
        for index in range(len(operations))
    )
    for operation, exception_type in zip(
        operations, exception_types, strict=True
    ):
        failure = exception_type()
        assert (
            runner._classify_fault_domain(operation, failure)
            == "operation_error"
        )
        runner._record_circuit_failure(
            attempt_id=attempt_id,
            operation=operation,
            fault_domain="operation_error",
        )
        clock.advance(100.0)
    state = runner.state()
    assert state["operator_incident"] == {
        "attempt_id": attempt_id,
        "operation": operations[-1],
        "fault_domain": "operation_error",
        "operation_consecutive": 1,
        "domain_consecutive": R.FAILURE_CIRCUIT_THRESHOLD,
        "threshold": R.FAILURE_CIRCUIT_THRESHOLD,
        "reason_code": "failure_circuit_exhausted",
    }
    assert state["failure_domain_circuits"][
        "operation_error"
    ]["consecutive"] == R.FAILURE_CIRCUIT_THRESHOLD
    assert runner._reserve_attempt(state) is None
    assert runner._reserve_auxiliary(state) is None

    recovered = R.ContiguousCampaignRunner(
        tmp_path / "campaign",
        backend=backend,
        promotion_gate=gate,
        input_builder=builder,
        backend_configuration=backend_configuration(),
        cost_window_id=COST_WINDOW_ID,
        max_lanes=1,
        controller_state_canaries=backend.controller_state_canaries,
        clock=clock,
        id_factory=ids(),
    )
    assert recovered.state()["failure_domain_circuits"] == (
        state["failure_domain_circuits"]
    )
    assert recovered.state()["failure_operation_circuits"] == (
        state["failure_operation_circuits"]
    )
    runner_audit = R.audit_runner_state_read_only(
        tmp_path / "campaign"
    )
    assert runner_audit["status"] == "PASS"
    scheduler_audit = R.Scheduler.audit_campaign(
        tmp_path / "campaign"
    )
    assert scheduler_audit["verdict"] == "PASS"
    assert scheduler_audit["runner_lifecycle"][
        "journal_head_digest"
    ] == runner_audit["journal_head_digest"]


def test_only_matching_authenticated_success_resets_failure_circuit(
    tmp_path,
):
    backend = FakeBackend()
    backend.fail_prepare_after = 0
    runner, _, _, _, clock = make_runner(
        tmp_path, backend=backend, max_lanes=1
    )
    runner.cycle()
    clock.advance(100.0)
    runner.cycle()
    before = runner.state()
    assert before["failure_domain_circuits"][
        "operation_error"
    ]["consecutive"] == 2

    # A success on another exact operation cannot erase the prepare outage.
    runner._record_circuit_failure(
        attempt_id=next(iter(before["attempts"])),
        operation="backend_poll",
        fault_domain="operation_error",
    )
    assert runner.state()["failure_domain_circuits"][
        "operation_error"
    ]["consecutive"] == 3
    backend.fail_prepare_after = None
    clock.advance(100.0)
    runner.cycle()
    after = runner.state()
    assert after["failure_operation_circuits"][
        "backend_prepare:operation_error"
    ]["consecutive"] == 0
    # backend_poll is the most recent different failed operation, so the
    # campaign-global domain remains active.
    assert after["failure_domain_circuits"][
        "operation_error"
    ]["consecutive"] == 3
    reset_events = [
        event
        for event in runner.journal.read()
        if event["kind"] == "FAILURE_CIRCUIT_RESET"
    ]
    assert reset_events[-1]["payload"]["reset_operation"] is True
    assert reset_events[-1]["payload"]["reset_domain"] is False
    forged = dict(reset_events[-1])
    forged["payload"] = {
        **forged["payload"],
        "evidence_kind": "attempt_launched",
    }
    with pytest.raises(R.ContiguousRunnerError):
        runner.journal.append(
            event_id="forged-reset",
            kind="FAILURE_CIRCUIT_RESET",
            payload=forged["payload"],
            recorded_at=clock(),
        )
        runner.state()


@pytest.mark.parametrize(
    ("limit", "result_kind"),
    (
        (None, "clean_no_progress"),
        (5.0, "clean_no_progress"),
        (None, "candidate"),
        (5.0, "candidate"),
    ),
)
def test_solver_outcomes_and_budget_mode_never_increment_failure_circuits(
    tmp_path,
    limit,
    result_kind,
):
    strategy = (
        (lambda spec: candidate_result(spec))
        if result_kind == "candidate"
        else (
            lambda _spec: R.AttemptResult(
                kind="clean_no_progress"
            )
        )
    )
    runner, _, _, _, _ = make_runner(
        tmp_path,
        backend=FakeBackend(strategy),
        max_lanes=1,
        limit=limit,
    )
    runner.cycle()
    runner.cycle()
    state = runner.state()
    assert state["failure_operation_circuits"] == {}
    assert state["failure_domain_circuits"] == {}
    assert state["operator_incident"] is None


def test_typed_terminal_provider_failure_counts_only_after_teardown(
    tmp_path,
):
    runner, backend, _, _, _ = make_runner(
        tmp_path,
        backend=FakeBackend(
            lambda _spec: R.AttemptResult(
                kind="infrastructure",
                reason="synthetic",
            )
        ),
        max_lanes=1,
    )
    runner.cycle()
    assert runner.state()["failure_domain_circuits"] == {}
    runner.cycle()
    state = runner.state()
    assert backend.teardown_calls
    assert state["failure_domain_circuits"][
        "provider_failure"
    ]["consecutive"] == 1
    events = runner.journal.read()
    failure_index = next(
        index
        for index, event in enumerate(events)
        if (
            event["kind"] == "FAILURE_CIRCUIT_FAILURE"
            and event["payload"]["operation"] == "backend_terminal"
        )
    )
    teardown_index = next(
        index
        for index, event in enumerate(events)
        if event["kind"] == "ATTEMPT_TORN_DOWN"
    )
    assert teardown_index < failure_index


def test_operator_incident_blocks_dispatch_but_preserves_live_cleanup(
    tmp_path,
):
    class OnePollFailure(FakeBackend):
        def __init__(self):
            super().__init__(
                lambda _spec: R.AttemptResult(
                    kind="clean_no_progress"
                )
            )
            self.failed = False

        def poll(self, **kwargs):
            if not self.failed:
                self.failed = True
                raise RuntimeError("synthetic poll outage")
            return super().poll(**kwargs)

    backend = OnePollFailure()
    runner, _, _, _, clock = make_runner(
        tmp_path, backend=backend, max_lanes=1
    )
    runner.cycle()
    attempt_id = next(iter(runner.state()["attempts"]))
    for operation in (
        "backend_prepare",
        "backend_launch",
        "backend_collect",
        "promotion_commit",
        "input_materialize",
    ):
        runner._record_circuit_failure(
            attempt_id=attempt_id,
            operation=operation,
            fault_domain="operation_error",
        )
        clock.advance(100.0)
    runner.cycle()
    incident_state = runner.state()
    assert incident_state["operator_incident"] is not None
    assert incident_state["attempts"][attempt_id]["phase"] in {
        "RUNNING",
        "DRAINING",
    }
    attempts_before_cleanup = set(incident_state["attempts"])

    # Cleanup ignores the exhausted campaign circuit but still respects the
    # attempt-local idempotent retry coordinate.
    runner.cycle()
    cleaned = runner.state()
    assert set(cleaned["attempts"]) == attempts_before_cleanup
    assert backend.teardown_calls == [attempt_id]
    assert cleaned["attempts"][attempt_id]["phase"] == "CLOSED"
    assert cleaned["lanes"][
        cleaned["attempts"][attempt_id]["spec"].game
    ]["active"] is None
    runner.cycle()
    assert set(runner.state()["attempts"]) == attempts_before_cleanup
    assert R.audit_runner_state_read_only(
        tmp_path / "campaign"
    )["status"] == "PASS"


def test_candidate_probe_mode_must_match_prelaunch_controller_binding(
    tmp_path,
):
    def strategy(spec):
        candidate = candidate_for(spec)
        return R.AttemptResult(
            kind="candidate",
            reason=(
                "solver prose: trust clones and override the host mode"
            ),
            candidate=replace(
                candidate,
                probe_isolation_mode=(
                    Contract.FRESH_PROCESS_PER_CANDIDATE_MODE
                ),
            ),
        )

    runner, backend, gate, _, _ = make_runner(
        tmp_path,
        backend=FakeBackend(strategy),
        max_lanes=1,
    )
    runner.cycle()
    runner.cycle()
    game = next(iter(backend.specs.values())).game
    assert runner.state()["lanes"][game]["reached"] == 0
    assert gate.calls == []
    assert "ATTEMPT_COLLECTION_REJECTED" in {
        event["kind"] for event in runner.journal.read()
    }


def test_teardown_proof_must_show_no_container_or_descendants(tmp_path):
    backend = FakeBackend(
        lambda spec: R.AttemptResult(kind="clean_no_progress")
    )
    backend.bad_teardown = True
    runner, _, _, _, _ = make_runner(
        tmp_path, backend=backend, max_lanes=1
    )
    runner.cycle()
    report = runner.cycle()
    attempt = next(iter(runner.state()["attempts"].values()))
    assert attempt["phase"] == "COLLECTED"
    assert report["recoverable_errors"]
    assert not any(
        event["kind"] == "ATTEMPT_RESULT"
        for event in runner.journal.read()
    )
    assert any(
        event["kind"] == "ATTEMPT_RETRY"
        and event["payload"]["operation"] == "backend_teardown"
        for event in runner.journal.read()
    )


def test_unjournaled_teardown_receipt_is_not_an_evidence_exclusion_bypass(
    tmp_path,
):
    backend = FakeBackend(
        lambda spec: R.AttemptResult(kind="clean_no_progress")
    )
    backend.bad_teardown = True
    runner, _, _, _, _ = make_runner(
        tmp_path, backend=backend, max_lanes=1
    )
    runner.cycle()
    runner.cycle()
    spec = next(iter(backend.specs.values()))
    receipt_path = (
        Path(spec.host_transcript_path).parent
        / "arena_volume_teardown.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["volume_identity_query_empty"] = False
    receipt_path.write_text(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        R.ContiguousRunnerError,
        match="terminal Arena teardown evidence failed retained canary replay",
    ):
        runner.state()


def test_public_action_protocol_invalid_revokes_all_lineage_authority_and_restart(
    tmp_path,
):
    backend = FakeBackend(
        lambda spec: R.AttemptResult(
            kind="candidate",
            candidate=candidate_result(spec).candidate,
        )
    )
    backend.public_action_protocol_invalid = True
    runner, _, gate, builder, factory = make_runner(
        tmp_path, backend=backend, max_lanes=1
    )
    runner.cycle()
    runner.cycle()
    protocol_events = [
        event
        for event in runner.journal.read()
        if event["kind"]
        == "ATTEMPT_PUBLIC_ACTION_PROTOCOL_INVALID"
    ]
    assert len(protocol_events) == 1
    attempt_id = protocol_events[0]["payload"]["attempt_id"]
    settled = runner.state()["attempts"][attempt_id]
    assert settled["phase"] == "CLOSED"
    assert settled["settled_result"] == R.AttemptResult(
        kind="protocol_invalid",
        cost_used=0.0,
        reason="public_action_protocol_invalid",
    )
    assert settled["candidate"] is None
    assert settled["collection"] is None
    assert settled["protocol_invalid"] is not None
    assert runner.state()["sidecar_requests"] == {}
    assert gate.calls == []
    assert not any(
        path.is_file()
        for path in runner.public_observation_registry.rglob("*")
    )

    recovered = R.ContiguousCampaignRunner(
        tmp_path / "campaign",
        backend=backend,
        promotion_gate=gate,
        input_builder=builder,
        backend_configuration=backend_configuration(),
        cost_window_id=COST_WINDOW_ID,
        max_lanes=1,
        controller_state_canaries=backend.controller_state_canaries,
        id_factory=factory,
    )
    recovered_attempt = recovered.state()["attempts"][attempt_id]
    assert recovered_attempt["settled_result"].kind == "protocol_invalid"
    assert recovered.state()["sidecar_requests"] == {}
    assert recovered.state()["lanes"][
        backend.specs[attempt_id].game
    ]["wip"] is None
    assert sum(
        event["kind"] == "ATTEMPT_RESULT"
        and event["payload"]["attempt_id"] == attempt_id
        for event in recovered.journal.read()
    ) == 1
    assert R.Scheduler.audit_campaign(
        tmp_path / "campaign"
    )["verdict"] == "PASS"

    evidence = settled["protocol_invalid"]["terminal_evidence"]
    taint_path = Path(
        evidence["partial_taint_scan_receipt_path"]
    )
    original_taint = taint_path.read_bytes()
    taint_value = json.loads(original_taint)
    taint_value["status"] = "TAINT"
    taint_path.write_text(
        json.dumps(
            taint_value,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(R.ContiguousRunnerError):
        recovered.state()
    assert R.Scheduler.audit_campaign(
        tmp_path / "campaign"
    )["verdict"] == "FAIL"
    taint_path.write_bytes(original_taint)

    usage_path = Path(evidence["partial_usage_receipt_path"])
    usage_bytes = usage_path.read_bytes()
    usage_path.unlink()
    with pytest.raises(R.ContiguousRunnerError):
        recovered.state()
    assert R.Scheduler.audit_campaign(
        tmp_path / "campaign"
    )["verdict"] == "FAIL"
    usage_path.write_bytes(usage_bytes)


def test_crash_after_external_launch_recovers_idempotently(tmp_path):
    backend = FakeBackend()
    backend.crash_after_first_launch = True
    gate = FakePromotionGate(tmp_path / "promotions")
    builder = FakeInputBuilder()
    factory = ids()
    runner = R.ContiguousCampaignRunner(
        tmp_path / "campaign",
        backend=backend,
        promotion_gate=gate,
        input_builder=builder,
        backend_configuration=backend_configuration(),
        cost_window_id=COST_WINDOW_ID,
        max_lanes=1,
        controller_state_canaries=backend.controller_state_canaries,
        id_factory=factory,
    )
    with pytest.raises(R.SimulatedCrash):
        runner.cycle()
    assert len(backend.launches) == 1
    assert not any(
        event["kind"] == "ATTEMPT_LAUNCHED"
        for event in runner.journal.read()
    )
    recovered = R.ContiguousCampaignRunner(
        tmp_path / "campaign",
        backend=backend,
        promotion_gate=gate,
        input_builder=builder,
        backend_configuration=backend_configuration(),
        cost_window_id=COST_WINDOW_ID,
        max_lanes=1,
        controller_state_canaries=backend.controller_state_canaries,
        id_factory=factory,
    )
    recovered.cycle()
    assert len(backend.launches) == 1
    assert len(set(backend.launch_calls)) == 1


def test_crash_after_input_materialization_recovers_reserved_identity(
    tmp_path,
):
    backend = FakeBackend()
    builder = CrashAfterInputBuilder()
    gate = FakePromotionGate(tmp_path / "promotions")
    factory = ids()
    runner = R.ContiguousCampaignRunner(
        tmp_path / "campaign",
        backend=backend,
        promotion_gate=gate,
        input_builder=builder,
        backend_configuration=backend_configuration(),
        cost_window_id=COST_WINDOW_ID,
        max_lanes=1,
        controller_state_canaries=backend.controller_state_canaries,
        id_factory=factory,
    )
    with pytest.raises(R.SimulatedCrash):
        runner.cycle()
    attempt_id, attempt = next(iter(runner.state()["attempts"].items()))
    assert attempt["phase"] == "RESERVED"
    generation_id = attempt["reservation"].generation_id
    assert {path.name for path in runner.generations.iterdir()} == {
        generation_id
    }
    recovered = R.ContiguousCampaignRunner(
        tmp_path / "campaign",
        backend=backend,
        promotion_gate=gate,
        input_builder=builder,
        backend_configuration=backend_configuration(),
        cost_window_id=COST_WINDOW_ID,
        max_lanes=1,
        controller_state_canaries=backend.controller_state_canaries,
        id_factory=factory,
    )
    recovered.cycle()
    assert recovered.state()["attempts"][attempt_id]["phase"] == "RUNNING"
    assert list(backend.specs) == [attempt_id]


def test_crash_after_collection_recovers_before_teardown(tmp_path):
    backend = FakeBackend(
        lambda spec: R.AttemptResult(kind="clean_no_progress")
    )
    backend.crash_after_first_collect = True
    runner, _, gate, builder, _ = make_runner(
        tmp_path, backend=backend, max_lanes=1
    )
    runner.cycle()
    with pytest.raises(R.SimulatedCrash):
        runner.cycle()
    attempt = next(iter(runner.state()["attempts"].values()))
    assert attempt["phase"] == "EXITED"
    recovered = R.ContiguousCampaignRunner(
        tmp_path / "campaign",
        backend=backend,
        promotion_gate=gate,
        input_builder=builder,
        backend_configuration=backend_configuration(),
        cost_window_id=COST_WINDOW_ID,
        max_lanes=1,
        controller_state_canaries=backend.controller_state_canaries,
        id_factory=ids(),
    )
    recovered.cycle()
    assert any(
        event["kind"] == "ATTEMPT_TORN_DOWN"
        for event in recovered.journal.read()
    )


@pytest.mark.parametrize(
    "fault_point",
    (
        "before_registry_install",
        "after_registry_install",
        "before_collected_append",
        "after_collected_append",
    ),
)
def test_public_observation_registry_write_ahead_recovers_every_boundary(
    tmp_path, monkeypatch, fault_point
):
    backend = FakeBackend(
        lambda spec: R.AttemptResult(kind="clean_no_progress")
    )
    runner, _, gate, builder, clock = make_runner(
        tmp_path, backend=backend, max_lanes=1
    )
    runner.cycle()
    fired = False
    original_install = R._install_regular_bytes
    original_append = runner.journal.append

    def install(path, payload, *, overwrite=False):
        nonlocal fired
        is_registry = (
            Path(path).parent == runner.public_observation_registry
        )
        if (
            is_registry
            and fault_point == "before_registry_install"
            and not fired
        ):
            fired = True
            raise R.SimulatedCrash()
        result = original_install(
            path, payload, overwrite=overwrite
        )
        if (
            is_registry
            and fault_point == "after_registry_install"
            and not fired
        ):
            fired = True
            raise R.SimulatedCrash()
        return result

    def append(*, event_id, kind, payload, recorded_at):
        nonlocal fired
        if (
            kind == "ATTEMPT_COLLECTED"
            and fault_point == "before_collected_append"
            and not fired
        ):
            fired = True
            raise R.SimulatedCrash()
        result = original_append(
            event_id=event_id,
            kind=kind,
            payload=payload,
            recorded_at=recorded_at,
        )
        if (
            kind == "ATTEMPT_COLLECTED"
            and fault_point == "after_collected_append"
            and not fired
        ):
            fired = True
            raise R.SimulatedCrash()
        return result

    monkeypatch.setattr(R, "_install_regular_bytes", install)
    monkeypatch.setattr(runner.journal, "append", append)
    with pytest.raises(R.SimulatedCrash):
        runner.cycle()
    assert fired
    staging = [
        event
        for event in runner.journal.read()
        if event["kind"]
        == "ATTEMPT_PUBLIC_OBSERVATIONS_STAGING"
    ]
    assert len(staging) == 1
    runner.state()
    recovered = R.ContiguousCampaignRunner(
        tmp_path / "campaign",
        backend=backend,
        promotion_gate=gate,
        input_builder=builder,
        backend_configuration=backend_configuration(),
        cost_window_id=COST_WINDOW_ID,
        max_lanes=1,
        controller_state_canaries=backend.controller_state_canaries,
        clock=clock,
        id_factory=ids(),
    )
    recovered.cycle()
    assert any(
        attempt["phase"] == "CLOSED"
        for attempt in recovered.state()["attempts"].values()
    )
    audit = R.Scheduler.audit_campaign(recovered.root)
    assert audit["verdict"] == "PASS", audit["findings"]


def test_crash_after_external_promotion_is_exactly_once(tmp_path):
    backend = FakeBackend(candidate_result)
    gate = FakePromotionGate(tmp_path / "promotions")
    gate.crash_after_first_commit = True
    runner, _, _, builder, _ = make_runner(
        tmp_path, backend=backend, gate=gate, max_lanes=1
    )
    runner.cycle()
    with pytest.raises(R.SimulatedCrash):
        runner.cycle()
    game = next(iter(backend.specs.values())).game
    assert runner.state()["lanes"][game]["reached"] == 0
    recovered = R.ContiguousCampaignRunner(
        tmp_path / "campaign",
        backend=backend,
        promotion_gate=gate,
        input_builder=builder,
        backend_configuration=backend_configuration(),
        cost_window_id=COST_WINDOW_ID,
        max_lanes=1,
        controller_state_canaries=backend.controller_state_canaries,
        id_factory=ids(),
    )
    recovered.cycle()
    assert recovered.state()["lanes"][game]["reached"] == 1
    assert len(gate.commits) == 1


def test_input_mutation_and_wrong_wip_hash_fail_closed(tmp_path):
    runner, backend, _, _, _ = make_runner(tmp_path, max_lanes=1)
    runner.cycle()
    spec = next(iter(backend.specs.values()))
    (Path(spec.input_dir) / "solve.py").write_text(
        "tampered=True\n", encoding="utf-8"
    )
    with pytest.raises(R.ContiguousRunnerError):
        runner.state()


def test_lane_source_cache_revalidates_replacement_and_mtime_spoof(
    tmp_path, monkeypatch
):
    runner, _, _, _, _ = make_runner(tmp_path, max_lanes=1)
    game = sorted(runner.state()["lanes"])[0]
    source_root = Path(
        runner.state()["lanes"][game]["source_path"]
    )
    source_file = source_root / "solve.py"
    raw = source_file.read_bytes()
    metadata = source_file.stat(follow_symlinks=False)

    calls = 0
    original_tree_hash = Contract._tree_hash

    def counted_tree_hash(root):
        nonlocal calls
        calls += 1
        return original_tree_hash(root)

    monkeypatch.setattr(Contract, "_tree_hash", counted_tree_hash)
    source_root_mode = stat.S_IMODE(
        source_root.stat(follow_symlinks=False).st_mode
    )
    os.chmod(source_root, 0o700, follow_symlinks=False)
    replacement = source_root / ".replacement"
    replacement.write_bytes(raw)
    os.chmod(
        replacement,
        stat.S_IMODE(metadata.st_mode),
        follow_symlinks=False,
    )
    os.utime(
        replacement,
        ns=(metadata.st_atime_ns, metadata.st_mtime_ns),
        follow_symlinks=False,
    )
    os.replace(replacement, source_file)
    os.chmod(
        source_root, source_root_mode, follow_symlinks=False
    )
    runner.state()
    assert calls > 0

    replaced_metadata = source_file.stat(follow_symlinks=False)
    changed = bytes([raw[0] ^ 1]) + raw[1:]
    os.chmod(source_file, 0o600, follow_symlinks=False)
    source_file.write_bytes(changed)
    os.chmod(
        source_file,
        stat.S_IMODE(replaced_metadata.st_mode),
        follow_symlinks=False,
    )
    os.utime(
        source_file,
        ns=(
            replaced_metadata.st_atime_ns,
            replaced_metadata.st_mtime_ns,
        ),
        follow_symlinks=False,
    )
    spoofed = source_file.stat(follow_symlinks=False)
    assert spoofed.st_size == replaced_metadata.st_size
    assert spoofed.st_mtime_ns == replaced_metadata.st_mtime_ns
    assert spoofed.st_ctime_ns != replaced_metadata.st_ctime_ns
    with pytest.raises(
        R.ContiguousRunnerError, match="lane source changed"
    ):
        runner.state()


def test_reducer_checkpoint_is_not_mutable_through_returned_state(
    tmp_path,
):
    runner, _, _, _, _ = make_runner(tmp_path, max_lanes=1)
    state = runner.state()
    game = sorted(state["lanes"])[0]
    state["lanes"][game]["reached"] = 999
    state["lanes"][game]["clean_proposer_settlements"].append(
        "forged"
    )
    reopened = runner.state()
    assert reopened["lanes"][game]["reached"] == 0
    assert (
        reopened["lanes"][game]["clean_proposer_settlements"]
        == []
    )
    runner.cycle()
    returned = runner.state()
    attempt_id = next(iter(returned["attempts"]))
    durable_game = returned["attempts"][attempt_id]["spec"].game
    object.__setattr__(
        returned["attempts"][attempt_id]["spec"],
        "game",
        "xxxx",
    )
    assert (
        runner.state()["attempts"][attempt_id]["spec"].game
        == durable_game
    )


def test_unjournaled_generation_is_rejected(tmp_path):
    runner, _, _, _, _ = make_runner(tmp_path, max_lanes=1)
    orphan = runner.generations / str(uuid.uuid4())
    orphan.mkdir()
    with pytest.raises(
        R.ContiguousRunnerError,
        match="unjournaled identity",
    ):
        runner.state()


def test_read_only_runner_audit_reuses_full_reducer_without_writes(
    tmp_path, monkeypatch
):
    runner, _, _, _, _ = make_runner(tmp_path, max_lanes=1)
    campaign = runner.root

    def inventory():
        rows = {}
        for path in sorted(
            (campaign, *campaign.rglob("*")),
            key=lambda item: str(item),
        ):
            metadata = path.stat(follow_symlinks=False)
            relative = str(path.relative_to(campaign)) or "."
            rows[relative] = (
                stat.S_IFMT(metadata.st_mode),
                stat.S_IMODE(metadata.st_mode),
                metadata.st_size,
                metadata.st_mtime_ns,
                (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    if path.is_file()
                    else None
                ),
            )
        return rows

    before = inventory()

    def forbidden_mutating_constructor(*_args, **_kwargs):
        raise AssertionError("mutating journal constructor was called")

    monkeypatch.setattr(
        R.DurableAttemptJournal,
        "__init__",
        forbidden_mutating_constructor,
    )
    receipt = R.audit_runner_state_read_only(campaign)
    assert receipt["status"] == "PASS"
    assert receipt["solved_levels"] == 0
    assert receipt["journal_event_count"] == 1
    assert receipt["scheduler_policy_sha256"] == (
        R.SCHEDULER_POLICY_SHA256
    )
    assert R.verify_runner_state_audit(
        receipt, campaign_root=campaign
    ) == receipt
    assert inventory() == before
    forged = dict(receipt)
    forged["state_sha256"] = "0" * 64
    with pytest.raises(
        R.ContiguousRunnerError,
        match="stale or forged",
    ):
        R.verify_runner_state_audit(
            forged, campaign_root=campaign
        )


def test_pre_retention_scheduler_pass_reopens_exact_terminal_binding(
    tmp_path,
):
    backend = FakeBackend(
        lambda _spec: R.AttemptResult(kind="clean_no_progress")
    )
    runner, _, _, _, _ = make_runner(
        tmp_path, backend=backend, max_lanes=1
    )
    for _ in range(12):
        runner.cycle()
        if any(
            attempt["phase"] == "CLOSED"
            for attempt in runner.state()["attempts"].values()
        ):
            break
    assert any(
        attempt["phase"] == "CLOSED"
        for attempt in runner.state()["attempts"].values()
    )
    runner_receipt = R.audit_runner_state_read_only(runner.root)
    receipt = R.Scheduler.audit_campaign(runner.root)
    assert receipt["verdict"] == "PASS"
    assert receipt["runner_lifecycle"]["receipt_sha256"] == (
        runner_receipt["receipt_sha256"]
    )
    receipt_path = tmp_path / "scheduler-pre-retention-pass.json"
    receipt_path.write_bytes(R._canonical_json(receipt) + b"\n")

    for generation in tuple(runner.generations.iterdir()):
        shutil.rmtree(generation)
    assert R.Scheduler.verify_pre_retention_audit_receipt(
        runner.root,
        receipt_path,
        expected_receipt_sha256=receipt["receipt_sha256"],
    ) == receipt

    with pytest.raises(
        R.Scheduler.SchedulerError,
        match="terminal journal/control binding",
    ):
        R.Scheduler.verify_pre_retention_audit_receipt(
            runner.root,
            receipt_path,
            expected_receipt_sha256="0" * 64,
        )

    first_event = sorted(
        path
        for path in (runner.root / "attempt_journal").iterdir()
        if not path.name.startswith(".")
    )[0]
    original_event = first_event.read_bytes()
    original_mode = stat.S_IMODE(
        first_event.stat(follow_symlinks=False).st_mode
    )
    first_event.chmod(0o600)
    try:
        first_event.write_bytes(original_event + b" ")
        with pytest.raises(
            R.Scheduler.SchedulerError,
            match="terminal journal/control binding",
        ):
            R.Scheduler.verify_pre_retention_audit_receipt(
                runner.root,
                receipt_path,
                expected_receipt_sha256=receipt["receipt_sha256"],
            )
    finally:
        first_event.write_bytes(original_event)
        first_event.chmod(original_mode)

    journal = R.Scheduler.read_journal(runner.root)
    sequence = len(journal) + 1
    body = {
        "schema": 1,
        "sequence": sequence,
        "event_id": "retention:head:mutation",
        "kind": "OPERATOR_INCIDENT",
        "recorded_at": float(sequence),
        "previous_digest": journal[-1]["digest"],
        "payload": {},
    }
    event = {
        **body,
        "digest": R.Scheduler._event_digest(body),
    }
    extra = (
        runner.root
        / "attempt_journal"
        / f"{sequence:020d}-retention:head:mutation.json"
    )
    extra.write_bytes(R.Scheduler.canonical_json(event) + b"\n")
    with pytest.raises(
        R.Scheduler.SchedulerError,
        match="terminal journal/control binding",
    ):
        R.Scheduler.verify_pre_retention_audit_receipt(
            runner.root,
            receipt_path,
            expected_receipt_sha256=receipt["receipt_sha256"],
        )


def test_terminal_retention_plan_covers_real_compact_attempt_evidence(
    tmp_path,
):
    runner, backend, _, _, _ = make_runner(tmp_path, max_lanes=1)
    runner.cycle()
    first_attempt_id = next(iter(backend.specs))
    backend.finish(
        first_attempt_id,
        R.AttemptResult(kind="clean_no_progress"),
    )
    runner.cycle()
    state = runner.state()
    closed_attempts = {
        attempt_id: attempt
        for attempt_id, attempt in state["attempts"].items()
        if attempt["phase"] == "CLOSED"
    }
    assert list(closed_attempts) == [first_attempt_id]
    closed = closed_attempts[first_attempt_id]
    generation_id = closed["reservation"].generation_id
    runner_receipt = R.audit_runner_state_read_only(runner.root)
    terminal_receipt_projection = {
        **runner_receipt,
        "complete": True,
        "solved_levels": 1,
        "total_levels": 1,
        "attempt_ids": [first_attempt_id],
        "generation_ids": [generation_id],
    }
    terminal_state_projection = {
        **state,
        "complete": True,
        "solved_levels": 1,
        "total_levels": 1,
        "attempts": closed_attempts,
        "pending_scheduler_decision": None,
        "pending_auxiliary_decision": None,
    }
    intent = R._terminal_retention_plan(
        runner.root,
        state=terminal_state_projection,
        runner_state_receipt=terminal_receipt_projection,
        pre_cleanup_audits={"scheduler": "9" * 64},
    )
    assert R._validate_terminal_retention_intent(
        intent,
        campaign_root=runner.root,
        runner_state_receipt=terminal_receipt_projection,
        pre_cleanup_audits={"scheduler": "9" * 64},
    ) == intent
    R._stage_terminal_retention_exports(runner.root, intent)
    compact_inventory = R._terminal_retention_archive_inventory(
        runner.root, intent
    )
    assert len(compact_inventory) == len(intent["compact_exports"])
    references = {
        reference
        for item in intent["compact_exports"]
        for reference in item["references"]
    }
    assert {
        "attempt_spec",
        "worker_outcome",
        "spec.input_bundle_receipt",
        "prepared.launch_attestation",
        "launched.controller_launch_receipt",
        "collection.token_usage_receipt",
        "collection.provider_usage_receipt",
        "collection.taint_scan_receipt",
        "collection.secret_scan_receipt",
        "collection.controller_state_scan_receipt",
        "collection.bridge_export_receipt",
        "collection.final_thread_binding",
        "collection.final_transcript_chain_receipt",
        "teardown.canary_reveal",
    } <= references
    assert all(
        Path(item["retained_relative_path"]).name
        == f"{item['evidence_sha256']}.json"
        for item in intent["compact_exports"]
    )
    assert not any(
        forbidden in reference
        for reference in references
        for forbidden in (
            "host_transcript",
            "app_server_transcript",
            "container_stdout",
            "container_stderr",
        )
    )


def test_terminal_retention_purges_invalid_turn_without_compact_raw_export(
    tmp_path,
):
    runner, backend, _, _, _ = make_runner(
        tmp_path,
        backend=FakeBackend(
            lambda _spec: R.AttemptResult(
                kind="infrastructure",
                reason="synthetic_invalid_turn",
            )
        ),
        max_lanes=1,
    )
    runner.cycle()
    first_attempt_id = next(iter(backend.specs))
    runner.cycle()
    state = runner.state()
    closed = state["attempts"][first_attempt_id]
    assert closed["phase"] == "CLOSED"
    assert closed["settled_result"].kind == "infrastructure"
    generation_id = closed["reservation"].generation_id
    runner_receipt = R.audit_runner_state_read_only(runner.root)
    terminal_receipt_projection = {
        **runner_receipt,
        "complete": True,
        "solved_levels": 1,
        "total_levels": 1,
        "attempt_ids": [first_attempt_id],
        "generation_ids": [generation_id],
    }
    terminal_state_projection = {
        **state,
        "complete": True,
        "solved_levels": 1,
        "total_levels": 1,
        "attempts": {first_attempt_id: closed},
        "pending_scheduler_decision": None,
        "pending_auxiliary_decision": None,
    }
    intent = R._terminal_retention_plan(
        runner.root,
        state=terminal_state_projection,
        runner_state_receipt=terminal_receipt_projection,
        pre_cleanup_audits={"scheduler": "9" * 64},
    )
    assert intent["compact_exports"] == []
    assert intent["compact_export_bytes"] == 0
    assert intent["retention_policy"][
        "invalid_attempt_raw_bytes_retained"
    ] is False
    assert R._validate_terminal_retention_intent(
        intent,
        campaign_root=runner.root,
        runner_state_receipt=terminal_receipt_projection,
        pre_cleanup_audits={"scheduler": "9" * 64},
    ) == intent
    R._stage_terminal_retention_exports(runner.root, intent)
    assert R._terminal_retention_archive_inventory(
        runner.root, intent
    ) == []


def test_cycle_lock_serializes_competing_supervisors(tmp_path):
    runner, backend, _, _, _ = make_runner(tmp_path, max_lanes=1)
    lock = open(runner.root / ".cycle.lock", "a+", encoding="utf-8")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    started = threading.Event()
    finished = threading.Event()
    errors = []

    def run_cycle():
        started.set()
        try:
            runner.cycle()
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=run_cycle)
    thread.start()
    assert started.wait(timeout=1)
    time.sleep(0.05)
    assert backend.specs == {}
    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    lock.close()
    assert finished.wait(timeout=60)
    thread.join()
    assert not thread.is_alive()
    assert errors == []
    assert len(backend.specs) == 1


def test_journal_binds_digests_but_never_token_contents(tmp_path):
    runner, backend, _, _, _ = make_runner(
        tmp_path,
        backend=FakeBackend(
            lambda spec: R.AttemptResult(kind="clean_no_progress")
        ),
        max_lanes=1,
    )
    runner.cycle()
    spec = next(iter(backend.specs.values()))
    token = "super-secret-arena-token"
    Path(spec.arena_token_file_path).write_text(token, encoding="utf-8")
    runner.cycle()
    journal_bytes = b"".join(
        path.read_bytes()
        for path in runner.journal.root.glob("*.json")
    )
    assert token.encode() not in journal_bytes
    kinds = {event["kind"] for event in runner.journal.read()}
    assert {
        "BACKEND_PREPARED",
        "ATTEMPT_LAUNCHED",
        "ATTEMPT_EXITED",
        "ATTEMPT_COLLECTED",
        "ATTEMPT_TORN_DOWN",
        "ATTEMPT_RESULT",
    } <= kinds
    prepared = next(
        event for event in runner.journal.read()
        if event["kind"] == "ATTEMPT_PREPARED"
    )
    durable_spec = prepared["payload"]["spec"]
    assert durable_spec["image_digest"] == IMAGE_DIGEST
    assert durable_spec["input_tree_sha256"] == spec.input_tree_sha256
    assert durable_spec["arena_token_file_path"] == (
        spec.arena_token_file_path
    )
    assert "token" not in durable_spec


def test_pending_file_and_conflicting_event_id_are_fail_closed(tmp_path):
    runner, _, _, _, _ = make_runner(tmp_path, max_lanes=1)
    before = runner.journal.read()
    (runner.journal.root / ".pending-power-loss").write_text(
        '{"torn":', encoding="utf-8"
    )
    assert runner.journal.read() == before
    event = before[0]
    with pytest.raises(R.ContiguousRunnerError):
        runner.journal.append(
            event_id=event["event_id"],
            kind=event["kind"],
            payload={"conflict": True},
            recorded_at=1.0,
        )


def test_journal_cache_rejects_mutation_replacement_and_truncation(
    tmp_path,
):
    def journal_at(name):
        journal = R.DurableAttemptJournal(tmp_path / name)
        appended = journal.append(
            event_id="event:one",
            kind="TEST",
            payload={"value": 1},
            recorded_at=1.0,
        )
        appended["payload"]["value"] = 999
        history = journal.read()
        assert history[0]["payload"] == {"value": 1}
        history[0]["payload"]["value"] = 998
        history.clear()
        assert journal.read()[0]["payload"] == {"value": 1}
        idempotent = journal.append(
            event_id="event:one",
            kind="TEST",
            payload={"value": 1},
            recorded_at=2.0,
        )
        idempotent["payload"]["value"] = 997
        assert journal.read()[0]["payload"] == {"value": 1}
        event_path = next(journal.root.glob("*.json"))
        return journal, event_path

    journal, event_path = journal_at("in-place")
    directory_before = journal._directory_signature()
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["payload"]["value"] = 2
    os.chmod(event_path, 0o600, follow_symlinks=False)
    event_path.write_text(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    # Editing an existing file does not change its parent directory pointer;
    # the per-file inode/ctime/size signature must still invalidate the cache.
    assert journal._directory_signature() == directory_before
    with pytest.raises(
        R.ContiguousRunnerError, match="immutable prefix"
    ):
        journal.read()

    journal, event_path = journal_at("replacement")
    replacement = event_path.parent / ".replacement"
    replacement.write_bytes(event_path.read_bytes())
    os.chmod(replacement, 0o400, follow_symlinks=False)
    os.replace(replacement, event_path)
    with pytest.raises(
        R.ContiguousRunnerError, match="immutable prefix"
    ):
        journal.read()

    journal, event_path = journal_at("truncation")
    event_path.unlink()
    with pytest.raises(
        R.ContiguousRunnerError, match="truncated or replaced"
    ):
        journal.read()


def test_journal_event_read_is_descriptor_anchored_against_valid_race(
    tmp_path,
    monkeypatch,
):
    journal = R.DurableAttemptJournal(tmp_path / "raced")
    journal.append(
        event_id="event:one",
        kind="TEST",
        payload={"value": 1},
        recorded_at=1.0,
    )
    event_path = next(journal.root.glob("*.json"))
    # Force a disk read.  The rewrite below lands after the descriptor read
    # but before its pathname signature is accepted for caching.
    journal._cache = None
    journal._cache_names = ()
    journal._cache_file_signatures = ()
    journal._cache_directory_signature = None
    original_signature = journal._file_signature
    raced = False

    def rewrite_then_sample(path):
        nonlocal raced
        if not raced:
            raced = True
            event = json.loads(path.read_text(encoding="utf-8"))
            event["payload"] = {"value": 2}
            event["digest"] = journal._event_digest({
                key: value
                for key, value in event.items()
                if key != "digest"
            })
            os.chmod(path, 0o600, follow_symlinks=False)
            path.write_bytes(R._canonical_json(event) + b"\n")
            os.chmod(path, 0o400, follow_symlinks=False)
        return original_signature(path)

    monkeypatch.setattr(
        journal, "_file_signature", rewrite_then_sample
    )
    with pytest.raises(
        R.ContiguousRunnerError, match="anchored read"
    ):
        journal.read()

    # The raced-in event is independently valid, which proves the rejection
    # is the A-bytes/B-signature mismatch rather than malformed JSON/hash data.
    reopened = R.ReadOnlyAttemptJournal(journal.root)
    assert reopened.read()[0]["payload"] == {"value": 2}

    # A genuine append verifies only the new suffix while public return values
    # remain unable to alias the authenticated cache.
    continued = R.DurableAttemptJournal(journal.root)
    appended = continued.append(
        event_id="event:two",
        kind="TEST",
        payload={"value": 3},
        recorded_at=2.0,
    )
    appended["payload"]["value"] = 999
    history = continued.read()
    assert [event["payload"]["value"] for event in history] == [2, 3]
    history[0]["payload"]["value"] = 998
    history.pop()
    assert [
        event["payload"]["value"] for event in continued.read()
    ] == [2, 3]


def test_journal_file_signature_rejects_regular_pointer_swap(
    tmp_path,
    monkeypatch,
):
    journal = R.DurableAttemptJournal(tmp_path / "pointer-raced")
    journal.append(
        event_id="event:one",
        kind="TEST",
        payload={"value": 1},
        recorded_at=1.0,
    )
    event_path = next(journal.root.glob("*.json"))
    replacement = event_path.parent / ".replacement"
    replacement.write_bytes(event_path.read_bytes())
    os.chmod(replacement, 0o400, follow_symlinks=False)
    expected = journal._cache_file_signatures[0]
    original_stat = Path.stat
    raced = False

    def stat_then_swap(path, *args, **kwargs):
        nonlocal raced
        metadata = original_stat(path, *args, **kwargs)
        if Path(path) == event_path and not raced:
            raced = True
            os.replace(replacement, event_path)
        return metadata

    monkeypatch.setattr(Path, "stat", stat_then_swap)
    with pytest.raises(
        R.ContiguousRunnerError, match="pointer changed"
    ):
        journal._file_signature(event_path)
    assert raced is True
    assert original_stat(
        event_path, follow_symlinks=False
    ).st_ino != expected[1]

    # The same swap cannot let cache-prefix revalidation return old authority.
    journal._cache_directory_signature = journal._directory_signature()
    with pytest.raises(
        R.ContiguousRunnerError, match="immutable prefix"
    ):
        journal.read()


def test_full_scale_journal_authentication_reads_only_appended_suffix_bytes(
    tmp_path,
    monkeypatch,
):
    campaign = tmp_path / "full-scale"
    journal = R.DurableAttemptJournal(
        campaign / "attempt_journal"
    )
    target_bytes = (
        R.Scheduler.MAX_JOURNAL_PREFIX_BYTES
        - 2 * R.Scheduler.MAX_JOURNAL_EVENT_BYTES
    )
    retained_bytes = 1
    previous_digest = None
    sequence = 1
    while target_bytes - retained_bytes > 2_048:
        remaining = target_bytes - retained_bytes
        padding_size = max(
            1,
            min(
                R.Scheduler.MAX_JOURNAL_EVENT_BYTES - 2_048,
                remaining - 1_024,
            ),
        )
        event_id = f"scale:{sequence}"
        body = {
            "schema": R.JOURNAL_SCHEMA,
            "sequence": sequence,
            "event_id": event_id,
            "kind": "SCALE",
            "recorded_at": float(sequence),
            "previous_digest": previous_digest,
            "payload": {"padding": "x" * padding_size},
        }
        event = {
            **body,
            "digest": journal._event_digest(body),
        }
        event_bytes = len(R._canonical_json(event)) + 1
        assert (
            event_bytes
            <= R.Scheduler.MAX_JOURNAL_EVENT_BYTES
        )
        assert retained_bytes + event_bytes <= target_bytes
        path = journal.root / (
            f"{sequence:020d}-{event_id}.json"
        )
        R._write_new_file(path, event)
        os.chmod(path, 0o400, follow_symlinks=False)
        retained_bytes += event_bytes
        previous_digest = event["digest"]
        sequence += 1
    R._fsync_directory(journal.root)
    status = R.Scheduler.journal_prefix_status(campaign)
    assert status["used_bytes"] == retained_bytes
    assert retained_bytes >= target_bytes - 2_048

    prefix = journal._read_authenticated()
    prefix_count = len(prefix)
    suffix_id = f"scale:{sequence}"
    suffix_body = {
        "schema": R.JOURNAL_SCHEMA,
        "sequence": sequence,
        "event_id": suffix_id,
        "kind": "SCALE",
        "recorded_at": float(sequence),
        "previous_digest": previous_digest,
        "payload": {"padding": "suffix"},
    }
    suffix_event = {
        **suffix_body,
        "digest": journal._event_digest(suffix_body),
    }
    suffix_path = journal.root / (
        f"{sequence:020d}-{suffix_id}.json"
    )
    R._write_new_file(suffix_path, suffix_event)
    os.chmod(suffix_path, 0o400, follow_symlinks=False)
    R._fsync_directory(journal.root)
    suffix_size = suffix_path.stat(follow_symlinks=False).st_size
    assert (
        R.Scheduler.journal_prefix_status(campaign)[
            "used_bytes"
        ]
        <= R.Scheduler.MAX_JOURNAL_PREFIX_BYTES
    )

    anchored_reads: list[tuple[Path, int]] = []
    original_read = journal._read_event_anchored

    def counted_read(path):
        anchored_reads.append((
            path,
            path.stat(follow_symlinks=False).st_size,
        ))
        return original_read(path)

    monkeypatch.setattr(
        journal, "_read_event_anchored", counted_read
    )
    extended = journal._read_authenticated()
    assert len(extended) == prefix_count + 1
    assert anchored_reads == [(suffix_path, suffix_size)]
    assert sum(size for _, size in anchored_reads) <= (
        R.Scheduler.MAX_JOURNAL_EVENT_BYTES
    )


def test_backend_configuration_is_durable_and_cannot_change(tmp_path):
    runner, backend, gate, builder, clock = make_runner(
        tmp_path, max_lanes=1
    )
    changed = backend_configuration()
    changed = R.BackendConfiguration(
        image_reference="gkm/arc-runner@sha256:" + "2" * 64,
        image_digest="sha256:" + "2" * 64,
        worker_command=changed.worker_command,
        resource_limits=changed.resource_limits,
        proposer_transport=changed.proposer_transport,
    )
    with pytest.raises(R.ContiguousRunnerError):
        R.ContiguousCampaignRunner(
            tmp_path / "campaign",
            backend=backend,
            promotion_gate=gate,
            input_builder=builder,
            backend_configuration=changed,
            cost_window_id=COST_WINDOW_ID,
            max_lanes=1,
            controller_state_canaries=backend.controller_state_canaries,
            clock=clock,
            id_factory=ids(),
        )


def test_promotion_gate_cannot_commit_k_plus_two(tmp_path):
    backend = FakeBackend(candidate_result)
    gate = FakePromotionGate(tmp_path / "promotions")
    gate.force_to_level_delta = 1
    runner, _, _, _, _ = make_runner(
        tmp_path, backend=backend, gate=gate, max_lanes=1
    )
    runner.cycle()
    runner.cycle()
    game = next(iter(backend.specs.values())).game
    lane = runner.state()["lanes"][game]
    assert lane["reached"] == 0
    assert lane["no_progress"] == 0
    assert lane["blocked"] is None
    assert any(
        event["kind"] == "PROMOTION_FAILED"
        and event["payload"]["code"] == "promotion_commit_invalid"
        for event in runner.journal.read()
    )
    assert (
        R.Scheduler.audit_campaign(runner.root)["verdict"]
        == "PASS"
    )
    invalid = next(iter(gate.commits.values()))
    assert Path(invalid.checkpoint_path).is_file()
    assert (
        lane["checkpoint_path"]
        != invalid.checkpoint_path
    )


def test_promotion_recovery_cannot_select_invalid_k_plus_two(tmp_path):
    class LostAckInvalidGate(FakePromotionGate):
        def __init__(self, root):
            super().__init__(root)
            self.lost = False

        def commit(self, *, spec, candidate):
            value = super().commit(spec=spec, candidate=candidate)
            if not self.lost:
                self.lost = True
                raise OSError("injected acknowledgement loss")
            return value

    backend = FakeBackend(candidate_result)
    gate = LostAckInvalidGate(tmp_path / "promotions")
    gate.force_to_level_delta = 1
    runner, _, _, _, _ = make_runner(
        tmp_path, backend=backend, gate=gate, max_lanes=1
    )
    runner.cycle()
    runner.cycle()
    game = next(iter(backend.specs.values())).game
    invalid = next(iter(gate.commits.values()))
    lane = runner.state()["lanes"][game]
    assert Path(invalid.checkpoint_path).is_file()
    assert lane["reached"] == 0
    assert lane["no_progress"] == 0
    assert lane["blocked"] is None
    assert any(
        event["kind"] == "PROMOTION_FAILED"
        and event["payload"]["code"] == "promotion_commit_invalid"
        for event in runner.journal.read()
    )
    assert lane["checkpoint_path"] != invalid.checkpoint_path


def test_exact_lifecycle_reaches_one_authoritative_game_boundary(tmp_path):
    inventory = Contract.authoritative_inventory()
    # The first scheduler lane makes this a deterministic, bounded K -> K+1
    # production edge. Full-campaign inventory is asserted independently below.
    target_game = next(iter(inventory))

    def strategy(spec):
        if spec.game == target_game:
            return candidate_result(spec)
        return _signed_test_blocker_result(spec)

    backend = FakeBackend(strategy)
    runner, _, _, _, _ = make_runner(
        tmp_path, backend=backend, max_lanes=1
    )
    for _ in range(2):
        runner.cycle()
    state = runner.state()
    lane = state["lanes"][target_game]
    assert lane["reached"] == 1
    assert lane["no_progress"] == 0
    assert lane["wip"] is None
    assert state["total_levels"] == 183
    assert len(inventory) == 25
    assert sum(inventory.values()) == 183

    first_spec = next(
        spec for spec in backend.specs.values()
        if spec.game == target_game
    )
    assert first_spec.target_level == 1
    promoted_frontier_sha256 = R.frontier_sha256(
        target_game, lane["reached"], lane["checkpoint_sha256"]
    )
    assert first_spec.frontier_sha256 != promoted_frontier_sha256

    # Isolate the just-promoted lane in a pure scheduling projection.  The
    # next decision has no caller channel for reusing the old spec: promotion
    # resets the coordinate and binds a fresh K+1 -> K+2 frontier.
    snapshot = runner._scheduler_snapshot_from_state(state)
    isolated = replace(
        snapshot,
        max_lanes=1,
        frontiers=tuple(
            replace(
                frontier,
                active_attempt_id=None,
                draining=False,
                blocked_reason=(
                    None
                    if frontier.game == target_game
                    else "test-only-policy-isolation"
                ),
            )
            for frontier in snapshot.frontiers
        ),
    )
    next_decision = R.Scheduler.build_decision(
        isolated,
        decision_id="decision:fresh-next-frontier",
        attempt_id="attempt:fresh-next-frontier",
        generation_id="generation:fresh-next-frontier",
        reservation_id="reservation:fresh-next-frontier",
    )
    assert next_decision is not None
    assert next_decision.choice.game == target_game
    assert next_decision.choice.target_level == 2
    assert next_decision.choice.no_progress == 0
    assert next_decision.choice.effort == "medium"
    assert next_decision.choice.soft_allocation_seconds == 15 * 60
    assert next_decision.choice.requested_wip_mode == "exclude"

    old_max_spec = replace(
        first_spec,
        effort="max",
        soft_allocation_seconds=300 * 60,
    )
    stale_choice = replace(
        next_decision.choice,
        target_level=old_max_spec.target_level,
        no_progress=9,
        effort=old_max_spec.effort,
        soft_allocation_seconds=(
            old_max_spec.soft_allocation_seconds
        ),
    )
    with pytest.raises(
        R.Scheduler.SchedulerError,
        match="stale.*forged",
    ):
        R.Scheduler.verify_decision(
            isolated,
            replace(next_decision, choice=stale_choice),
        )

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import codex_failure_revision_contract as C


_FROZEN_INNER_HEAD = "0eeb29d6cce57a71dc0a20bffc471d21849d03de"
_FROZEN_INNER_SHA256 = (
    "eefa34dcef63adb4d99deb07de9fa1920d8ef792080427557d5460201ff32f94"
)
_FROZEN_INNER_FILES = (
    ("gkm_legs.py", _FROZEN_INNER_SHA256),
    (
        "gkm_arena.py",
        "9174a6ec78abea5b6c7cdc1afd49b47725cc3cf49a9e9c3c390b66f9aefd6b43",
    ),
    (
        "arc_agi3_proposer_boundary.py",
        "7ab5447704c83f607c3f61d7fb69f9df4690b71cf5abf10e381d6e875d6be202",
    ),
    (
        "codex_campaign_status.py",
        "d105008b30c955f7b77f62c0f9443a838b1a691b68eba4337f63cb0bba00426f",
    ),
    (
        "codex_usage_guard.py",
        "607d88946414508785dbc8d1b5a5da91c4e55c5e8162d57b09e2f37b2aedf9f4",
    ),
    (
        "claude_usage_guard.py",
        "b63092521d7b21279e2db3e821bdc6862fb74fafc50df8be81d2849572d75bf2",
    ),
    (
        "gkm_solve_agent.py",
        "81b2099ea568729a03604916ee4fa69baaae5b1ee710eb51a78808bcd2796810",
    ),
)
_FROZEN_INNER_DRIVER = r'''\
import hashlib
import importlib.machinery
import json
import sys
import types
from pathlib import Path

lab = types.ModuleType("lab")
lab.make_env = lambda *args, **kwargs: None
binder = types.ModuleType("llm_binder")
binder.ollama_text = lambda *args, **kwargs: ""
engine = types.ModuleType("arcengine")
engine.ActionInput = object
engine.GameAction = object
sys.modules.update({"lab": lab, "llm_binder": binder, "arcengine": engine})

source = Path(sys.argv[1]).resolve(strict=True)
request = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
source_hashes = request.get("source_sha256s")
if not isinstance(source_hashes, dict) or not source_hashes:
    raise SystemExit("historical source hash map is malformed")
source_bytes = None
for name, expected in source_hashes.items():
    if not isinstance(name, str) or Path(name).name != name:
        raise SystemExit("historical source name is unsafe")
    if not isinstance(expected, str) or len(expected) != 64:
        raise SystemExit("historical source digest is malformed")
    candidate = (source.parent / name).resolve(strict=True)
    if candidate.parent != source.parent:
        raise SystemExit("historical source escaped the staged directory")
    payload = candidate.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise SystemExit(f"historical source hash mismatch: {name}")
    if candidate == source:
        source_bytes = payload
if source_bytes is None:
    raise SystemExit("historical gkm_legs source is not pinned")
sys.path.insert(0, str(source.parent))
module = types.ModuleType("gkm_legs")
module.__file__ = str(source)
module.__package__ = ""
module.__loader__ = None
module.__spec__ = importlib.machinery.ModuleSpec(
    "gkm_legs", loader=None, origin=str(source)
)
sys.modules["gkm_legs"] = module
exec(compile(source_bytes, str(source), "exec", dont_inherit=True), module.__dict__)

result = module._aggregate_codex_revision_record(
    request["records"],
    ws=request["ws"],
    aggregate_logs=tuple(request["aggregate_logs"]),
    aggregate_identities=tuple(
        tuple(item) for item in request["aggregate_identities"]
    ),
    ledger_path=request["ledger_path"],
    max_rounds=4,
    minutes_limit=300,
    settlement_reserve_seconds=1_800.0,
    terminal_error=None,
)
sys.stdout.write(json.dumps(result, separators=(",", ":"), sort_keys=True))
'''


def _stage_frozen_inner_runner(tmp_path: Path) -> Path:
    fixture = (
        Path(__file__).with_name("testdata")
        / f"frozen_inner_{_FROZEN_INNER_HEAD}"
    )
    expected_names = {name for name, _digest in _FROZEN_INNER_FILES}
    assert {path.name for path in fixture.iterdir()} == expected_names
    destination = tmp_path / "historical" / "arc" / "crack_lab"
    destination.mkdir(parents=True)
    for name, expected in _FROZEN_INNER_FILES:
        source = fixture / name
        assert source.is_file() and not source.is_symlink()
        payload = source.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected
        (destination / name).write_bytes(payload)
    return destination / "gkm_legs.py"


def _line(event):
    return json.dumps(event, separators=(",", ":")).encode() + b"\n"


def _aggregate(*, timeout: bool = False):
    transcript = bytearray()
    diagnostics = bytearray()
    rounds = []
    for index in range(1, 5):
        terminal_timeout = timeout and index == 4
        events = [{"type": "thread.started", "thread_id": f"thread-{index}"}]
        if not terminal_timeout:
            events.append({
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100 + index,
                    "cached_input_tokens": 10 + index,
                    "output_tokens": 20 + index,
                    "reasoning_output_tokens": 5 + index,
                },
            })
        transcript_slice = b"".join(_line(event) for event in events)
        diagnostics_slice = f"round {index}\n".encode()
        transcript_offset = len(transcript)
        diagnostics_offset = len(diagnostics)
        transcript.extend(transcript_slice)
        diagnostics.extend(diagnostics_slice)
        usage = {
            "input_tokens": 0 if terminal_timeout else 100 + index,
            "cached_input_tokens": 0 if terminal_timeout else 10 + index,
            "output_tokens": 0 if terminal_timeout else 20 + index,
            "reasoning_output_tokens": 0 if terminal_timeout else 5 + index,
        }
        usage["observed_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        verifier = (
            {field: None for field in C.VERIFIER_FIELDS}
            if terminal_timeout else {
                "verifier_classification": "no_progress",
                "verification_mode": "replay",
                "verifier_error_present": False,
                "reached_before": 8,
                "reached_after": 8,
                "solved_target": False,
            }
        )
        row = {
            "round_index": index,
            "turn_kind": "proposal" if index == 1 else "revision",
            "thread_id": f"thread-{index}",
            "target_level": 9,
            "termination_kind": (
                "terminal_failure" if terminal_timeout else "completed"
            ),
            "allocation_policy": "hard",
            "allocation_basis_seconds": 100.0 * (5 - index),
            "rounds_left_at_launch": 5 - index,
            "allocation_seconds": 100.0,
            "minutes_limit": 100.0 / 60.0,
            "duration_seconds": 120.0 if terminal_timeout else 50.0,
            "allocation_expired": terminal_timeout,
            "timed_out": terminal_timeout,
            "returncode": -15 if terminal_timeout else 0,
            "launch_error": None,
            "interrupted": False,
            "process_group_stop_attempted": terminal_timeout,
            "process_group_quiesced": True,
            "surviving_process_group": False,
            "protected_transcript_status": "sealed",
            "protected_diagnostics_status": "sealed",
            "protected_transcript_error": None,
            "round_transcript_offset": transcript_offset,
            "round_transcript_size": len(transcript_slice),
            "round_transcript_sha256": hashlib.sha256(transcript_slice).hexdigest(),
            "round_diagnostics_offset": diagnostics_offset,
            "round_diagnostics_size": len(diagnostics_slice),
            "round_diagnostics_sha256": hashlib.sha256(diagnostics_slice).hexdigest(),
            "task_feedback_sha256": hashlib.sha256(f"task-{index}".encode()).hexdigest(),
            "failure_revision_protocol_sha256": C.PROTOCOL_SHA256,
            **C.BOUNDARY_BINDING,
            "failure_class": "containment" if terminal_timeout else None,
            "failure_detail_class": "hard_wall_time" if terminal_timeout else None,
            "public_action_protocol_violation": False,
            "filesystem_boundary_violation": False,
            "filesystem_boundary_violation_reason": None,
            "taint_verdict": "clean",
            **verifier,
            "thread_started_events": 1,
            "turn_completed_events": 0 if terminal_timeout else 1,
            "usage_reported": not terminal_timeout,
            **usage,
        }
        assert set(row) == C.ROUND_KEYS
        rounds.append(row)

    terminal = rounds[-1]
    record = {
        "event": "codex_exec",
        "target_level": 9,
        "reached": 8,
        "transcript": "aggregate.jsonl",
        "diagnostics": "aggregate.stderr.log",
        "failure_revision_protocol_sha256": C.PROTOCOL_SHA256,
        "rounds_used": 4,
        "rounds_max": 4,
        "terminal_round_index": 4,
        "rounds_evaluated": 3 if timeout else 4,
        "completed_round_count": 3 if timeout else 4,
        "timeout_round_count": 1 if timeout else 0,
        "timeout_round_indices": [4] if timeout else [],
        "rounds": rounds,
        "rounds_left_at_launch": terminal["rounds_left_at_launch"],
        "window_allocation_seconds": 18_000.0,
        "slice_budget_seconds": 16_200.0,
        "settlement_reserve_seconds": 1_800.0,
        "allocation_policy": "hard",
        "duration_seconds": round(sum(row["duration_seconds"] for row in rounds), 3),
        "minutes_limit": 300,
        "termination_kind": "terminal_failure" if timeout else "completed",
        "aggregate_terminal_status": (
            "terminal_failure" if timeout else "clean"
        ),
        "returncode": None,
        "returncode_authority": "host_aggregate",
        "allocation_expired": timeout,
        "timed_out": timeout,
        "process_group_stop_attempted": timeout,
        "process_group_quiesced": True,
        "surviving_process_group": False,
        "public_action_protocol_violation": False,
        "filesystem_boundary_violation": False,
        "filesystem_boundary_violation_reason": None,
        "taint_verdict": "clean",
        "failure_class": "containment" if timeout else None,
        "failure_detail_class": "hard_wall_time" if timeout else None,
        "interrupted": False,
        "thread_id": "thread-4",
        "thread_id_authority": "terminal_provider_thread",
        "task_feedback_sha256": terminal["task_feedback_sha256"],
        "protected_transcript_status": "sealed",
        "protected_transcript_error": None,
        "protected_transcript_size": len(transcript),
        "protected_transcript_sha256": hashlib.sha256(transcript).hexdigest(),
        "protected_diagnostics_status": "sealed",
        "protected_diagnostics_size": len(diagnostics),
        "protected_diagnostics_sha256": hashlib.sha256(diagnostics).hexdigest(),
        "thread_started_events": 4,
        "turn_completed_events": 3 if timeout else 4,
        "usage_reported": not timeout,
        **{
            field: sum(row[field] for row in rounds)
            for field in C.USAGE_FIELDS
        },
        **{field: terminal[field] for field in C.VERIFIER_FIELDS},
        **C.BOUNDARY_BINDING,
    }
    assert C.REQUIRED_TOP_KEYS <= set(record)
    return record, bytes(transcript), bytes(diagnostics)


def _validate(record, transcript, diagnostics):
    return C.validate_exec(
        record,
        expected_rounds_max=4,
        target_level=9,
        reached_before=8,
        transcript_payload=transcript,
        diagnostics_payload=diagnostics,
        require_evidence=True,
    )


def _outcome(record, metadata, *, solved=False):
    reached_after = record["rounds"][len(metadata) - 1]["reached_after"]
    outcome = {
        "thread_id": record["thread_id"],
        "codex_exec_transcript": record["transcript"],
        "target_level": 9,
        "reached_before": 8,
        "reached_after": reached_after,
        "solved_target": solved,
        "winning_path_present": solved,
        "winning_marginal_C": 0 if solved else None,
        "taint_verdict": "clean",
        "failure_revision_rounds": metadata,
    }
    outcome.update({
        field: record[field]
        for field in C.OUTCOME_RECORD_BINDING_FIELDS
        if field in record
    })
    return outcome


def _completed_prefix_terminal():
    record, transcript, diagnostics = _aggregate()
    rounds = record["rounds"][:2]
    transcript = transcript[:sum(
        row["round_transcript_size"] for row in rounds
    )]
    diagnostics = diagnostics[:sum(
        row["round_diagnostics_size"] for row in rounds
    )]
    record.update({
        "rounds": rounds,
        "rounds_used": 2,
        "rounds_left_at_launch": rounds[-1]["rounds_left_at_launch"],
        "terminal_round_index": 2,
        "rounds_evaluated": 2,
        "completed_round_count": 2,
        "termination_kind": "terminal_failure",
        "aggregate_terminal_status": "terminal_failure",
        "failure_class": "infrastructure",
        "failure_detail_class": "revision_window_exhausted",
        "allocation_expired": True,
        "duration_seconds": round(sum(
            row["duration_seconds"] for row in rounds
        ), 3),
        "thread_started_events": 2,
        "turn_completed_events": 2,
        "thread_id": rounds[-1]["thread_id"],
        "task_feedback_sha256": rounds[-1]["task_feedback_sha256"],
        "protected_transcript_size": len(transcript),
        "protected_transcript_sha256": hashlib.sha256(transcript).hexdigest(),
        "protected_diagnostics_size": len(diagnostics),
        "protected_diagnostics_sha256": hashlib.sha256(diagnostics).hexdigest(),
        **{
            field: sum(row[field] for row in rounds)
            for field in C.USAGE_FIELDS
        },
        **{field: rounds[-1][field] for field in C.VERIFIER_FIELDS},
    })
    return record, transcript, diagnostics


def test_completed_aggregate_and_outcome_are_exactly_authenticated():
    record, transcript, diagnostics = _aggregate()
    aggregate = _validate(record, transcript, diagnostics)
    assert aggregate is not None
    metadata = [{
        "target_level": 9,
        **{field: row[field] for field in (
            "round_index", "turn_kind", "termination_kind", *C.VERIFIER_FIELDS,
        )},
    } for row in record["rounds"]]
    C.validate_outcome(
        record,
        _outcome(record, metadata),
        aggregate,
        target_level=9,
        reached_before=8,
    )


def test_frozen_inner_aggregate_output_matches_outer_contract(tmp_path):
    synthetic, transcript, diagnostics = _aggregate()
    transcript_name = "codex_turn_frozen_inner.jsonl"
    diagnostics_name = "codex_turn_frozen_inner.stderr.log"
    workspace = tmp_path / "workspace"
    protected = tmp_path / ".proposer_transcripts" / workspace.name
    workspace.mkdir()
    protected.mkdir(parents=True)
    transcript_path = protected / transcript_name
    diagnostics_path = protected / diagnostics_name
    transcript_path.write_bytes(transcript)
    diagnostics_path.write_bytes(diagnostics)
    records = []
    for row in synthetic["rounds"]:
        raw = dict(row)
        raw.update({
            "event": "codex_exec",
            "started_at": "2026-08-10T00:00:00+00:00",
            "run_label": "lf52:L9:propose",
            "game": "lf52",
            "reached": 8,
            "transcript": transcript_name,
            "diagnostics": diagnostics_name,
            "workspace": workspace.name,
            "model": "gpt-5.6-sol",
            "reasoning_effort": "max",
        })
        records.append(raw)
    source = _stage_frozen_inner_runner(tmp_path)
    request_path = tmp_path / "frozen_inner_request.json"
    request_path.write_text(json.dumps({
        "records": records,
        "ws": str(workspace),
        "aggregate_logs": [transcript_name, diagnostics_name],
        "aggregate_identities": [
            [path.stat().st_dev, path.stat().st_ino]
            for path in (transcript_path, diagnostics_path)
        ],
        "ledger_path": str(tmp_path / "ledger.jsonl"),
        "source_sha256s": dict(_FROZEN_INNER_FILES),
    }), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable, "-I", "-c", _FROZEN_INNER_DRIVER,
            str(source), str(request_path),
        ],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    aggregate = json.loads(proc.stdout)
    assert aggregate is not None
    assert aggregate["rounds_left_at_launch"] == 1
    assert C.validate_exec(
        aggregate,
        target_level=9,
        reached_before=8,
        transcript_payload=transcript,
        diagnostics_payload=diagnostics,
        require_evidence=True,
    ) is not None


def test_unsolved_outcome_may_retain_a_verified_parent_path():
    record, transcript, diagnostics = _aggregate()
    aggregate = _validate(record, transcript, diagnostics)
    metadata = [{
        "target_level": 9,
        **{field: row[field] for field in (
            "round_index", "turn_kind", "termination_kind",
            *C.VERIFIER_FIELDS,
        )},
    } for row in record["rounds"]]
    outcome = _outcome(record, metadata)
    outcome["winning_path_present"] = True
    C.validate_outcome(
        record, outcome, aggregate,
        target_level=9, reached_before=8,
    )


@pytest.mark.parametrize("field,value", (
    ("thread_id", "wrong-thread"),
    ("codex_exec_transcript", "wrong.jsonl"),
    ("game", "wrong"),
    ("target_level", 99),
    ("reached_before", 999),
    ("reached_after", 777),
    ("taint_verdict", "poison"),
    ("run_label", "wrong:L9:propose"),
    ("model", "wrong-model"),
    ("reasoning_effort", "low"),
))
def test_aggregate_outcome_top_binding_fails_closed(field, value):
    record, transcript, diagnostics = _aggregate()
    record.update({
        "game": "lf52",
        "run_label": "lf52:L9:propose",
        "model": "gpt-5.6-sol",
        "reasoning_effort": "max",
    })
    aggregate = _validate(record, transcript, diagnostics)
    metadata = [{
        "target_level": 9,
        **{key: row[key] for key in (
            "round_index", "turn_kind", "termination_kind",
            *C.VERIFIER_FIELDS,
        )},
    } for row in record["rounds"]]
    outcome = _outcome(record, metadata)
    outcome[field] = value
    with pytest.raises(C.ContractError):
        C.validate_outcome(
            record, outcome, aggregate,
            target_level=9, reached_before=8,
        )


def test_completed_prefix_terminal_can_bind_a_clean_false_outcome():
    record, transcript, diagnostics = _completed_prefix_terminal()
    aggregate = _validate(record, transcript, diagnostics)
    assert aggregate is not None and aggregate.terminal_failure
    metadata = [{
        "target_level": 9,
        **{field: row[field] for field in (
            "round_index", "turn_kind", "termination_kind",
            *C.VERIFIER_FIELDS,
        )},
    } for row in record["rounds"]]
    C.validate_outcome(
        record,
        _outcome(record, metadata),
        aggregate,
        target_level=9,
        reached_before=8,
    )

    record["failure_detail_class"] = "terminal_infrastructure"
    record["allocation_expired"] = False
    with pytest.raises(C.ContractError, match="not canonical"):
        _validate(record, transcript, diagnostics)


@pytest.mark.parametrize("failure_class,failure_detail", (
    ("credit_out", "campaign_guard"),
    ("infrastructure", "between_round_infrastructure"),
))
def test_other_completed_prefix_terminal_causes_bind_false_outcomes(
    failure_class, failure_detail
):
    record, transcript, diagnostics = _completed_prefix_terminal()
    record.update({
        "failure_class": failure_class,
        "failure_detail_class": failure_detail,
        "allocation_expired": False,
    })
    aggregate = _validate(record, transcript, diagnostics)
    metadata = [{
        "target_level": 9,
        **{field: row[field] for field in (
            "round_index", "turn_kind", "termination_kind",
            *C.VERIFIER_FIELDS,
        )},
    } for row in record["rounds"]]
    C.validate_outcome(
        record,
        _outcome(record, metadata),
        aggregate,
        target_level=9,
        reached_before=8,
    )


def test_terminal_timeout_is_charged_but_cannot_have_feedback_or_outcome():
    record, transcript, diagnostics = _aggregate(timeout=True)
    aggregate = _validate(record, transcript, diagnostics)
    assert aggregate is not None and aggregate.timed_out
    with pytest.raises(C.ContractError, match="cannot append"):
        C.validate_outcome(
            record,
            {"solved_target": False, "failure_revision_rounds": []},
            aggregate,
            target_level=9,
            reached_before=8,
        )


@pytest.mark.parametrize("mutation", (
    "partial_markers", "slice_hash", "slice_offset", "lifecycle",
    "thread_reuse", "usage", "protocol", "boundary", "timeout_continued",
))
def test_aggregate_tampering_fails_closed(mutation):
    record, transcript, diagnostics = _aggregate(
        timeout=mutation == "timeout_continued"
    )
    if mutation == "partial_markers":
        record.pop("rounds_evaluated")
    elif mutation == "slice_hash":
        record["rounds"][1]["round_transcript_sha256"] = "0" * 64
    elif mutation == "slice_offset":
        record["rounds"][1]["round_diagnostics_offset"] += 1
    elif mutation == "lifecycle":
        record["rounds"][1]["turn_completed_events"] = 0
    elif mutation == "thread_reuse":
        record["rounds"][1]["thread_id"] = "thread-1"
    elif mutation == "usage":
        record["rounds"][1]["input_tokens"] += 1
    elif mutation == "protocol":
        record["rounds"][1]["failure_revision_protocol_sha256"] = "0" * 64
    elif mutation == "boundary":
        record["rounds"][1]["filesystem_boundary_policy_sha256"] = "0" * 64
    else:
        record["rounds"].append(dict(record["rounds"][-1]))
        record["rounds_used"] = 5
        record["rounds_max"] = 5
        record["terminal_round_index"] = 5
    with pytest.raises(C.ContractError):
        _validate(record, transcript, diagnostics)


@pytest.mark.parametrize("field,value", (
    ("target_level", 10),
    ("rounds_left_at_launch", 1),
    ("allocation_basis_seconds", 399.9999),
    ("allocation_seconds", 99.999),
    ("minutes_limit", 1.0),
))
def test_aggregate_fair_share_tampering_fails_closed(field, value):
    record, transcript, diagnostics = _aggregate()
    record["rounds"][0][field] = value
    with pytest.raises(C.ContractError):
        _validate(record, transcript, diagnostics)


def test_terminal_timeout_requires_a_started_provider_thread():
    record, transcript, diagnostics = _aggregate(timeout=True)
    terminal = record["rounds"][-1]
    terminal["thread_started_events"] = 0
    terminal["thread_id"] = None
    record["thread_started_events"] = 3
    with pytest.raises(C.ContractError):
        _validate(record, transcript, diagnostics)


def _zero_start_timeout_aggregate():
    record, transcript, diagnostics = _aggregate(timeout=True)
    terminal = record["rounds"][-1]
    transcript = transcript[:terminal["round_transcript_offset"]]
    terminal.update({
        "thread_id": None,
        "round_transcript_size": 0,
        "round_transcript_sha256": hashlib.sha256(b"").hexdigest(),
        "thread_started_events": 0,
        "turn_completed_events": 0,
    })
    transcript_sha = hashlib.sha256(transcript).hexdigest()
    binding = "\0".join((
        record["transcript"], transcript_sha,
    )).encode()
    record.update({
        "thread_id": "failure-revision-" + hashlib.sha256(binding).hexdigest(),
        "thread_id_authority": "host_aggregate_fallback",
        "protected_transcript_size": len(transcript),
        "protected_transcript_sha256": transcript_sha,
        "thread_started_events": 3,
        "turn_completed_events": 3,
    })
    return record, transcript, diagnostics


def test_zero_start_terminal_timeout_uses_exact_host_fallback_authority():
    record, transcript, diagnostics = _zero_start_timeout_aggregate()
    aggregate = _validate(record, transcript, diagnostics)
    assert aggregate is not None and aggregate.timed_out

    record["thread_id"] = "failure-revision-" + "0" * 64
    with pytest.raises(C.ContractError, match="fallback-thread authority"):
        _validate(record, transcript, diagnostics)


def test_zero_start_terminal_timeout_cannot_claim_a_completion():
    record, transcript, diagnostics = _zero_start_timeout_aggregate()
    record["rounds"][-1]["turn_completed_events"] = 1
    record["turn_completed_events"] = 4
    with pytest.raises(C.ContractError, match="lifecycle"):
        _validate(record, transcript, diagnostics)


@pytest.mark.parametrize("field,value", (
    ("minutes_limit", 299),
    ("window_allocation_seconds", 17_999.0),
    ("slice_budget_seconds", 16_199.0),
    ("settlement_reserve_seconds", 1_799.0),
))
def test_aggregate_window_timing_tampering_fails_closed(field, value):
    record, transcript, diagnostics = _aggregate()
    record[field] = value
    with pytest.raises(C.ContractError):
        _validate(record, transcript, diagnostics)


def test_self_consistent_noncanonical_treatment_window_fails_closed():
    record, transcript, diagnostics = _aggregate()
    record.update({
        "minutes_limit": 299,
        "window_allocation_seconds": 17_940.0,
        "settlement_reserve_seconds": 1_794.0,
        "slice_budget_seconds": 16_146.0,
    })
    with pytest.raises(C.ContractError, match="campaign totals"):
        _validate(record, transcript, diagnostics)


def test_terminal_non_timeout_cannot_claim_allocation_expiry():
    record, transcript, diagnostics = _aggregate(timeout=True)
    record["rounds"][-1]["timed_out"] = False
    record.update({
        "timed_out": False,
        "timeout_round_count": 0,
        "timeout_round_indices": [],
    })
    with pytest.raises(C.ContractError, match="control metadata"):
        _validate(record, transcript, diagnostics)


def _terminal_infrastructure_aggregate():
    record, transcript, diagnostics = _aggregate(timeout=True)
    terminal = record["rounds"][-1]
    terminal.update({
        "allocation_expired": False,
        "timed_out": False,
        "returncode": 17,
        "failure_class": "infrastructure",
        "failure_detail_class": "known_transient",
        "process_group_stop_attempted": False,
    })
    record.update({
        "allocation_expired": False,
        "timed_out": False,
        "timeout_round_count": 0,
        "timeout_round_indices": [],
        "process_group_stop_attempted": False,
        "failure_class": "infrastructure",
        "failure_detail_class": "terminal_infrastructure",
    })
    return record, transcript, diagnostics


def test_terminal_failure_matrix_accepts_exact_infrastructure_projection():
    record, transcript, diagnostics = _terminal_infrastructure_aggregate()
    aggregate = _validate(record, transcript, diagnostics)
    assert aggregate is not None and aggregate.terminal_failure


def test_terminal_failure_matrix_rejects_invented_nested_class():
    record, transcript, diagnostics = _terminal_infrastructure_aggregate()
    record["rounds"][-1].update({
        "failure_class": "solver_success",
        "failure_detail_class": "invented_terminal",
    })
    record.update({
        "failure_class": "invented_top",
        "failure_detail_class": "invented_top_detail",
    })
    with pytest.raises(C.ContractError, match="not canonical"):
        _validate(record, transcript, diagnostics)


def test_terminal_failure_matrix_rejects_incoherent_taint_verdict():
    record, transcript, diagnostics = _terminal_infrastructure_aggregate()
    record["rounds"][-1]["taint_verdict"] = "tainted"
    record["taint_verdict"] = "tainted"
    with pytest.raises(C.ContractError, match="control metadata"):
        _validate(record, transcript, diagnostics)


def test_terminal_failure_matrix_rejects_invented_top_projection():
    record, transcript, diagnostics = _terminal_infrastructure_aggregate()
    record.update({
        "failure_class": "infrastructure",
        "failure_detail_class": "invented_top_detail",
    })
    with pytest.raises(C.ContractError, match="projection disagrees"):
        _validate(record, transcript, diagnostics)


def test_hard_wall_time_requires_an_authenticated_timeout():
    record, transcript, diagnostics = _terminal_infrastructure_aggregate()
    record["rounds"][-1].update({
        "failure_class": "containment",
        "failure_detail_class": "hard_wall_time",
    })
    record.update({
        "failure_class": "containment",
        "failure_detail_class": "hard_wall_time",
    })
    with pytest.raises(C.ContractError, match="controls disagree"):
        _validate(record, transcript, diagnostics)


def test_surviving_process_group_is_not_admissible_as_sealed_authority():
    record, transcript, diagnostics = _terminal_infrastructure_aggregate()
    record["rounds"][-1].update({
        "failure_class": "evidence",
        "failure_detail_class": "surviving_process_group",
    })
    record.update({
        "failure_class": "evidence",
        "failure_detail_class": "terminal_evidence",
    })
    with pytest.raises(C.ContractError, match="not canonical"):
        _validate(record, transcript, diagnostics)


@pytest.mark.parametrize("failure_class,failure_detail,top_detail", (
    ("credit_out", "provider_credit_out", "campaign_guard"),
    ("infrastructure", "known_transient", "terminal_infrastructure"),
    ("infrastructure", "unknown_cli", "terminal_infrastructure"),
))
def test_provider_terminal_causes_require_nonzero_exit(
    failure_class, failure_detail, top_detail,
):
    record, transcript, diagnostics = _terminal_infrastructure_aggregate()
    record["rounds"][-1].update({
        "failure_class": failure_class,
        "failure_detail_class": failure_detail,
        "returncode": 0,
    })
    record.update({
        "failure_class": failure_class,
        "failure_detail_class": top_detail,
    })
    with pytest.raises(C.ContractError, match="controls disagree"):
        _validate(record, transcript, diagnostics)


def test_invalid_lifecycle_cause_requires_derived_lifecycle_failure():
    record, transcript, diagnostics = _aggregate()
    terminal = record["rounds"][-1]
    terminal.update({
        "termination_kind": "terminal_failure",
        "failure_class": "evidence",
        "failure_detail_class": "invalid_or_reused_turn_lifecycle",
        **{field: None for field in C.VERIFIER_FIELDS},
    })
    record.update({
        "rounds_evaluated": 3,
        "completed_round_count": 3,
        "termination_kind": "terminal_failure",
        "aggregate_terminal_status": "terminal_failure",
        "failure_class": "evidence",
        "failure_detail_class": "terminal_evidence",
        **{field: None for field in C.VERIFIER_FIELDS},
    })
    with pytest.raises(C.ContractError, match="controls disagree"):
        _validate(record, transcript, diagnostics)


def test_invalid_lifecycle_cause_requires_a_clean_provider_exit():
    record, transcript, diagnostics = _terminal_infrastructure_aggregate()
    record["rounds"][-1].update({
        "failure_class": "evidence",
        "failure_detail_class": "invalid_or_reused_turn_lifecycle",
        "returncode": -15,
    })
    record.update({
        "failure_class": "evidence",
        "failure_detail_class": "terminal_evidence",
    })
    with pytest.raises(C.ContractError, match="controls disagree"):
        _validate(record, transcript, diagnostics)


@pytest.mark.parametrize("failure_class,failure_detail,top_detail", (
    ("credit_out", "provider_credit_out", "campaign_guard"),
    ("infrastructure", "known_transient", "terminal_infrastructure"),
    ("infrastructure", "unknown_cli", "terminal_infrastructure"),
))
def test_provider_terminal_causes_cannot_claim_a_process_group_stop(
    failure_class, failure_detail, top_detail,
):
    record, transcript, diagnostics = _terminal_infrastructure_aggregate()
    record["rounds"][-1].update({
        "failure_class": failure_class,
        "failure_detail_class": failure_detail,
        "process_group_stop_attempted": True,
    })
    record.update({
        "failure_class": failure_class,
        "failure_detail_class": top_detail,
        "process_group_stop_attempted": True,
    })
    with pytest.raises(C.ContractError, match="controls disagree"):
        _validate(record, transcript, diagnostics)


@pytest.mark.parametrize("field", (
    "failure_revision_shadow", "round_shadow", "rounds_shadow",
))
def test_unknown_aggregate_control_marker_fails_closed(field):
    record, transcript, diagnostics = _aggregate()
    record[field] = "unexpected"
    with pytest.raises(C.ContractError, match="unknown revision-control"):
        _validate(record, transcript, diagnostics)


@pytest.mark.parametrize("evidence_kind", ("transcript", "diagnostics"))
def test_hash_consistent_protocol_marker_injection_fails_closed(
    evidence_kind,
):
    record, transcript, diagnostics = _aggregate()
    terminal = record["rounds"][-1]
    if evidence_kind == "transcript":
        injected = _line({
            "type": "notice",
            "message": C.PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER.decode(),
        })
        transcript += injected
        start = terminal["round_transcript_offset"]
        terminal_slice = transcript[start:]
        terminal["round_transcript_size"] = len(terminal_slice)
        terminal["round_transcript_sha256"] = hashlib.sha256(
            terminal_slice
        ).hexdigest()
        record["protected_transcript_size"] = len(transcript)
        record["protected_transcript_sha256"] = hashlib.sha256(
            transcript
        ).hexdigest()
    else:
        diagnostics += C.PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER + b"\n"
        start = terminal["round_diagnostics_offset"]
        terminal_slice = diagnostics[start:]
        terminal["round_diagnostics_size"] = len(terminal_slice)
        terminal["round_diagnostics_sha256"] = hashlib.sha256(
            terminal_slice
        ).hexdigest()
        record["protected_diagnostics_size"] = len(diagnostics)
        record["protected_diagnostics_sha256"] = hashlib.sha256(
            diagnostics
        ).hexdigest()
    with pytest.raises(C.ContractError, match="protocol-violation projection"):
        _validate(record, transcript, diagnostics)


def test_validator_cannot_relax_the_treatment_maximum():
    record, transcript, diagnostics = _aggregate()
    with pytest.raises(C.ContractError, match="noncanonical maximum"):
        C.validate_exec(
            record,
            expected_rounds_max=2,
            target_level=9,
            reached_before=8,
            transcript_payload=transcript,
            diagnostics_payload=diagnostics,
            require_evidence=True,
        )

"""Offline tests for exactly-once semantic and ranker turn journals."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from bongard.canonical import canonical_digest, canonical_json
import bongard.object_bongard_turn_journal as journal_module
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTextTurnJournalTransport,
    ObjectBongardTurnCallFailed,
    ObjectBongardTurnJournalError,
    ObjectBongardTurnNonterminalClaim,
    ObjectBongardTurnRuntime,
    verify_object_bongard_turn_journal,
)
from bongard.tests.no_tools_fixture import (
    canonical_codex_receipt,
    canonical_no_tools_runtime,
)
import bongard.transport as transport_module
from bongard.transport import (
    CODEX_APPLY_PATCH_TOOL_TYPE,
    CODEX_EFFECTIVE_TOOL_MODE,
    CODEX_ISOLATION_POLICY,
    CODEX_RECEIPT_SCHEMA,
    CODEX_TOOL_SURFACE_DIGEST,
    CODEX_TRANSPORT_POLICY_DIGEST,
    DEFAULT_CODEX_MODEL,
    PINNED_CODEX_CLI_VERSION,
    CloudPolicyCacheSnapshot,
    CodexReceipt,
    CodexStructuredResult,
)


MODEL = DEFAULT_CODEX_MODEL
EFFORT = "medium"
LAUNCHER_DIGEST = "1" * 64
TRANSPORT_SOURCE_DIGEST = "2" * 64
AUTHORIZATION_DIGEST = "sha256:" + "a" * 64
PRECOMMIT_DIGEST = "sha256:" + "b" * 64
TASK_ID = "bd_turn_journal_fixture"
PROMPT = "Inspect the neutral drawings and return the requested JSON object."
SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": ["value"],
    "additionalProperties": False,
}
PAYLOAD = {"value": "pointed bird-like object with oblique spans"}
IMAGES = (
    ("group_0_ref_00.png", b"\x89PNG\r\n\x1a\nfirst-exact-image"),
    ("group_1_ref_00.png", b"\x89PNG\r\n\x1a\nsecond-exact-image"),
)

MODEL_CATALOG, NO_TOOLS_ATTESTATION = canonical_no_tools_runtime(
    LAUNCHER_DIGEST
)


def _runtime(*, text: bool, minutes: int = 3) -> ObjectBongardTurnRuntime:
    return ObjectBongardTurnRuntime(
        model=MODEL,
        reasoning_effort=EFFORT,
        minutes=minutes,
        verbose=False,
        executable="/private/synthetic-codex",
        cloud_policy_cache_snapshot=(CloudPolicyCacheSnapshot(None) if text else None),
        model_catalog_snapshot=MODEL_CATALOG,
        expected_launcher_digest=LAUNCHER_DIGEST,
        no_tools_attestation=NO_TOOLS_ATTESTATION,
        transport_source_digest=TRANSPORT_SOURCE_DIGEST,
    )


def _kwargs(runtime: ObjectBongardTurnRuntime) -> dict[str, object]:
    return {
        "model": runtime.model,
        "reasoning_effort": runtime.reasoning_effort,
        "minutes": runtime.minutes,
        "verbose": runtime.verbose,
        "executable": runtime.executable,
        "cloud_policy_cache_snapshot": runtime.cloud_policy_cache_snapshot,
        "model_catalog_snapshot": runtime.model_catalog_snapshot,
        "tool_surface_attestation": runtime.no_tools_attestation,
        "expected_launcher_digest": runtime.expected_launcher_digest,
        "expected_tool_surface_attestation_digest": (
            runtime.no_tools_attestation.attestation_digest
        ),
    }


def _text_receipt(
    prompt: str,
    schema: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> CodexReceipt:
    schema_digest = canonical_digest(dict(schema))
    capture = next(
        row
        for row in NO_TOOLS_ATTESTATION.to_dict()["captures"]
        if row["modality"] == "text"
    )
    binding = {
        "model_catalog_digest": MODEL_CATALOG.raw_digest,
        "transport_policy_digest": CODEX_TRANSPORT_POLICY_DIGEST,
        "command_digest": capture["normalized_command_digest"],
        "effective_tool_mode": CODEX_EFFECTIVE_TOOL_MODE,
        "apply_patch_tool_type": CODEX_APPLY_PATCH_TOOL_TYPE,
        "tool_surface_digest": CODEX_TOOL_SURFACE_DIGEST,
        "tool_surface_attestation_digest": NO_TOOLS_ATTESTATION.attestation_digest,
    }
    causal = transport_module._causal_text_input_metadata(
        prompt, schema_digest, binding
    )
    body: dict[str, Any] = {
        "schema": CODEX_RECEIPT_SCHEMA,
        "source": "codex-cli",
        "requested_model": MODEL,
        "reported_model": "",
        "model_identity_evidence": "explicit-cli-model-flag;jsonl-omits-model",
        "requested_reasoning_effort": EFFORT,
        "input_tokens": 10,
        "cached_input_tokens": 0,
        "output_tokens": 2,
        "reasoning_output_tokens": 1,
        "thread_id": "00000000-0000-4000-8000-000000000081",
        "codex_cli_version": PINNED_CODEX_CLI_VERSION,
        "codex_launcher_digest": LAUNCHER_DIGEST,
        "cloud_config_bundle_cache_binding": "absent",
        **causal,
        "output_schema_digest": schema_digest,
        "structured_output_digest": canonical_digest(dict(payload)),
        "proposed_source_digest": "",
        "proposed_log_digest": "",
        "event_stream_digest": "c" * 64,
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


class _NamedTransport:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail
        self.journal: ObjectBongardNamedImageTurnJournalTransport | None = None

    def __call__(
        self,
        prompt: str,
        paths: Sequence[str],
        names: Sequence[str],
        schema: Mapping[str, Any],
        **kwargs: object,
    ) -> CodexStructuredResult:
        self.calls += 1
        if self.fail:
            raise OSError("unstable diagnostic must not enter durable identity")
        assert self.journal is not None
        assert self.journal.claim_path.exists()
        assert not self.journal.result_path.exists()
        assert not self.journal.outcome_path.exists()
        receipt = canonical_codex_receipt(
            prompt,
            paths,
            schema,
            PAYLOAD,
            launcher_digest=LAUNCHER_DIGEST,
            reasoning_effort=EFFORT,
            model=MODEL,
            names=names,
        )
        return CodexStructuredResult(dict(PAYLOAD), receipt)


class _TextTransport:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail
        self.journal: ObjectBongardTextTurnJournalTransport | None = None

    def __call__(
        self,
        prompt: str,
        schema: Mapping[str, Any],
        **kwargs: object,
    ) -> CodexStructuredResult:
        self.calls += 1
        if self.fail:
            raise OSError("unstable diagnostic must not enter durable identity")
        assert self.journal is not None
        assert self.journal.claim_path.exists()
        assert not self.journal.result_path.exists()
        assert not self.journal.outcome_path.exists()
        return CodexStructuredResult(
            dict(PAYLOAD), _text_receipt(prompt, schema, PAYLOAD)
        )


class _CrashTransport:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> CodexStructuredResult:
        self.calls += 1
        raise KeyboardInterrupt("process vanished before terminalization")


class _ForbiddenTransport:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> CodexStructuredResult:
        self.calls += 1
        raise AssertionError("cold replay called the physical transport")


def _write_images(tmp_path: Path) -> tuple[str, ...]:
    paths: list[str] = []
    for name, data in IMAGES:
        target = tmp_path / name
        target.write_bytes(data)
        paths.append(str(target.resolve()))
    return tuple(paths)


def _named_journal(
    tmp_path: Path,
    transport: object,
    *,
    prompt: str = PROMPT,
    runtime: ObjectBongardTurnRuntime | None = None,
    authorization_digest: str = AUTHORIZATION_DIGEST,
) -> ObjectBongardNamedImageTurnJournalTransport:
    journal = ObjectBongardNamedImageTurnJournalTransport(
        tmp_path / "named-journal",
        authorization_digest=authorization_digest,
        execution_precommit_digest=PRECOMMIT_DIGEST,
        task_id=TASK_ID,
        turn_kind="semantic_support",
        expected_prompt=prompt,
        expected_images=IMAGES,
        expected_output_schema=SCHEMA,
        runtime=_runtime(text=False) if runtime is None else runtime,
        underlying_transport=transport,  # type: ignore[arg-type]
    )
    if isinstance(transport, _NamedTransport):
        transport.journal = journal
    return journal


def _text_journal(
    tmp_path: Path,
    transport: object,
    *,
    prompt: str = PROMPT,
    runtime: ObjectBongardTurnRuntime | None = None,
) -> ObjectBongardTextTurnJournalTransport:
    journal = ObjectBongardTextTurnJournalTransport(
        tmp_path / "text-journal",
        authorization_digest=AUTHORIZATION_DIGEST,
        execution_precommit_digest=PRECOMMIT_DIGEST,
        task_id=TASK_ID,
        turn_kind="survivor_rank",
        expected_prompt=prompt,
        expected_output_schema=SCHEMA,
        runtime=_runtime(text=True) if runtime is None else runtime,
        underlying_transport=transport,  # type: ignore[arg-type]
    )
    if isinstance(transport, _TextTransport):
        transport.journal = journal
    return journal


def _invoke_named(
    journal: ObjectBongardNamedImageTurnJournalTransport,
    paths: Sequence[str],
    *,
    prompt: str = PROMPT,
    schema: Mapping[str, Any] = SCHEMA,
    runtime: ObjectBongardTurnRuntime | None = None,
) -> CodexStructuredResult:
    selected = _runtime(text=False) if runtime is None else runtime
    return journal(
        prompt,
        paths,
        tuple(name for name, _ in IMAGES),
        schema,
        **_kwargs(selected),
    )


def _invoke_text(
    journal: ObjectBongardTextTurnJournalTransport,
    *,
    prompt: str = PROMPT,
    schema: Mapping[str, Any] = SCHEMA,
    runtime: ObjectBongardTurnRuntime | None = None,
) -> CodexStructuredResult:
    selected = _runtime(text=True) if runtime is None else runtime
    return journal(prompt, schema, **_kwargs(selected))


def test_named_image_turn_claims_before_call_persists_result_before_terminal_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_images(tmp_path)
    writes: list[str] = []
    original = journal_module._write_once

    def recording_write(path, value):
        original(path, value)
        writes.append(path.name)

    monkeypatch.setattr(journal_module, "_write_once", recording_write)
    physical = _NamedTransport()
    journal = _named_journal(tmp_path, physical)

    result = _invoke_named(journal, paths)

    assert physical.calls == 1
    assert result.payload == PAYLOAD
    assert writes == ["manifest.json", "claim.json", "result.json", "outcome.json"]
    durable = json.loads(journal.result_path.read_text(encoding="utf-8"))
    outcome = json.loads(journal.outcome_path.read_text(encoding="utf-8"))
    assert durable["codex_structured_result"] == {
        "payload": PAYLOAD,
        "receipt": result.receipt.to_dict(),
    }
    assert outcome["result_digest"] == durable["record_digest"]
    assert outcome["result_persisted_and_fsynced_before_terminal"] is True
    summary = verify_object_bongard_turn_journal(journal)
    assert summary.terminal_status == "success"

    forbidden = _ForbiddenTransport()
    restarted = _named_journal(tmp_path, forbidden)
    replayed = _invoke_named(restarted, paths)

    assert forbidden.calls == 0
    assert replayed == result
    assert restarted.fresh_call_count == 0
    assert restarted.reused_call_count == 1


def test_text_turn_uses_exact_ranker_shape_and_replays_without_transport(
    tmp_path: Path,
) -> None:
    physical = _TextTransport()
    journal = _text_journal(tmp_path, physical)

    result = _invoke_text(journal)

    assert physical.calls == 1
    assert result.payload == PAYLOAD
    assert journal.verify().terminal_status == "success"

    forbidden = _ForbiddenTransport()
    restarted = _text_journal(tmp_path, forbidden)
    replayed = _invoke_text(restarted)

    assert forbidden.calls == 0
    assert replayed == result
    assert restarted.reused_call_count == 1


@pytest.mark.parametrize("modality", ["named", "text"])
def test_stranded_claim_refuses_restart_with_zero_calls(
    tmp_path: Path, modality: str
) -> None:
    crashing = _CrashTransport()
    if modality == "named":
        paths = _write_images(tmp_path)
        first = _named_journal(tmp_path, crashing)
        invoke = lambda journal: _invoke_named(journal, paths)
        restart = lambda transport: _named_journal(tmp_path, transport)
    else:
        first = _text_journal(tmp_path, crashing)
        invoke = _invoke_text
        restart = lambda transport: _text_journal(tmp_path, transport)

    with pytest.raises(KeyboardInterrupt):
        invoke(first)
    assert crashing.calls == 1
    assert first.claim_path.exists()
    assert not first.result_path.exists()
    assert not first.outcome_path.exists()
    with pytest.raises(ObjectBongardTurnNonterminalClaim):
        first.verify()

    forbidden = _ForbiddenTransport()
    restarted = restart(forbidden)
    with pytest.raises(ObjectBongardTurnNonterminalClaim, match="rerun is forbidden"):
        invoke(restarted)
    assert forbidden.calls == 0
    assert restarted.refused_call_count == 1


def test_transport_failure_is_terminal_typed_and_zero_call_replayed(
    tmp_path: Path,
) -> None:
    failing = _TextTransport(fail=True)
    journal = _text_journal(tmp_path, failing)

    with pytest.raises(ObjectBongardTurnCallFailed) as first:
        _invoke_text(journal)

    assert failing.calls == 1
    assert journal.verify().terminal_status == "failure"
    forbidden = _ForbiddenTransport()
    restarted = _text_journal(tmp_path, forbidden)
    with pytest.raises(ObjectBongardTurnCallFailed) as replayed:
        _invoke_text(restarted)
    assert forbidden.calls == 0
    assert replayed.value.turn_key == first.value.turn_key
    assert replayed.value.failure_digest == first.value.failure_digest


@pytest.mark.parametrize(
    "drift",
    ["prompt", "schema", "runtime", "image_bytes", "image_order"],
)
def test_invocation_drift_is_rejected_before_claim_or_transport(
    tmp_path: Path, drift: str
) -> None:
    paths = list(_write_images(tmp_path))
    physical = _NamedTransport()
    journal = _named_journal(tmp_path, physical)
    prompt = PROMPT
    schema: Mapping[str, Any] = SCHEMA
    runtime = _runtime(text=False)
    names = tuple(name for name, _ in IMAGES)
    if drift == "prompt":
        prompt += " altered"
    elif drift == "schema":
        changed = deepcopy(SCHEMA)
        changed["properties"]["value"]["description"] = "altered"
        schema = changed
    elif drift == "runtime":
        runtime = _runtime(text=False, minutes=4)
    elif drift == "image_bytes":
        Path(paths[0]).write_bytes(b"different")
    else:
        names = tuple(reversed(names))

    with pytest.raises(ObjectBongardTurnJournalError):
        journal(
            prompt,
            paths,
            names,
            schema,
            **_kwargs(runtime),
        )

    assert physical.calls == 0
    assert not journal.claim_path.exists()
    assert journal.attempted_call_count == 0


@pytest.mark.parametrize("drift", ["authorization", "prompt", "runtime"])
def test_restart_manifest_drift_fails_closed(tmp_path: Path, drift: str) -> None:
    _text_journal(tmp_path, _ForbiddenTransport())
    kwargs: dict[str, object] = {}
    if drift == "authorization":
        with pytest.raises(ObjectBongardTurnJournalError, match="manifest"):
            ObjectBongardTextTurnJournalTransport(
                tmp_path / "text-journal",
                authorization_digest="sha256:" + "d" * 64,
                execution_precommit_digest=PRECOMMIT_DIGEST,
                task_id=TASK_ID,
                turn_kind="survivor_rank",
                expected_prompt=PROMPT,
                expected_output_schema=SCHEMA,
                runtime=_runtime(text=True),
                underlying_transport=_ForbiddenTransport(),
            )
        return
    if drift == "prompt":
        kwargs["prompt"] = PROMPT + " altered"
    else:
        kwargs["runtime"] = _runtime(text=True, minutes=4)
    with pytest.raises(ObjectBongardTurnJournalError, match="manifest"):
        _text_journal(tmp_path, _ForbiddenTransport(), **kwargs)


def test_recomputed_digest_receipt_tamper_is_rejected_by_cold_replay(
    tmp_path: Path,
) -> None:
    journal = _text_journal(tmp_path, _TextTransport())
    _invoke_text(journal)
    result = json.loads(journal.result_path.read_text(encoding="utf-8"))
    outcome = json.loads(journal.outcome_path.read_text(encoding="utf-8"))

    receipt = result["codex_structured_result"]["receipt"]
    receipt["requested_reasoning_effort"] = "high"
    receipt_body = {
        key: value for key, value in receipt.items() if key != "receipt_digest"
    }
    receipt["receipt_digest"] = canonical_digest(receipt_body)
    result["receipt_digest"] = receipt["receipt_digest"]
    result_body = {
        key: value for key, value in result.items() if key != "record_digest"
    }
    result["record_digest"] = "sha256:" + canonical_digest(result_body)
    outcome["result_digest"] = result["record_digest"]
    outcome_body = {
        key: value for key, value in outcome.items() if key != "record_digest"
    }
    outcome["record_digest"] = "sha256:" + canonical_digest(outcome_body)
    journal.result_path.write_bytes(canonical_json(result) + b"\n")
    journal.outcome_path.write_bytes(canonical_json(outcome) + b"\n")

    with pytest.raises(ObjectBongardTurnJournalError):
        journal.verify()
    forbidden = _ForbiddenTransport()
    restarted = _text_journal(tmp_path, forbidden)
    with pytest.raises(ObjectBongardTurnJournalError):
        _invoke_text(restarted)
    assert forbidden.calls == 0


def test_manifest_and_claim_bind_authorization_precommit_task_and_turn_kind(
    tmp_path: Path,
) -> None:
    journal = _text_journal(tmp_path, _TextTransport())
    _invoke_text(journal)
    manifest = json.loads(journal.manifest_path.read_text(encoding="utf-8"))
    claim = json.loads(journal.claim_path.read_text(encoding="utf-8"))

    for record in (manifest, claim):
        assert record["authorization_digest"] == AUTHORIZATION_DIGEST
        assert record["execution_precommit_digest"] == PRECOMMIT_DIGEST
        assert record["task_id"] == TASK_ID
        assert record["turn_kind"] == "survivor_rank"
    authority = journal.verify().to_data()
    assert authority["python_is_canonical_authority"] is True
    assert authority["lean_required"] is False
    assert authority["lean_removable"] is True
    assert authority["lean_affects_identity_or_replay"] is False


@pytest.mark.parametrize("tamper_name", ["result.json", "unexpected.json"])
def test_preexisting_orphan_or_extra_record_refuses_physical_call(
    tmp_path: Path, tamper_name: str
) -> None:
    physical = _TextTransport()
    journal = _text_journal(tmp_path, physical)
    (journal.directory / tamper_name).write_text("{}\n", encoding="utf-8")

    with pytest.raises(ObjectBongardTurnJournalError):
        _invoke_text(journal)

    assert physical.calls == 0
    assert not journal.claim_path.exists()

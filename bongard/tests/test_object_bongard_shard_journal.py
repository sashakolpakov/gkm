from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw
import pytest

from bongard.object_bongard_shard_journal import (
    ObjectBongardShardCallFailed,
    ObjectBongardShardJournalError,
    ObjectBongardShardJournalTransport,
    ObjectBongardShardNonterminalClaim,
    verify_object_bongard_shard_journal,
)
from bongard.prototype_object_hypotheses import (
    extract_object_hypothesis_packet,
    render_object_hypothesis_atlas,
)
from bongard.prototype_object_observer_protocol import (
    plan_prototype_object_feature_shards,
    prototype_object_feature_output_schema,
    prototype_object_feature_shard_prompt,
)
from bongard.tests.no_tools_fixture import (
    canonical_codex_receipt,
    canonical_no_tools_runtime,
)
from bongard.transport import DEFAULT_CODEX_MODEL, CodexStructuredResult


LAUNCHER_DIGEST = "1" * 64
EFFORT = "medium"
MODEL_CATALOG, NO_TOOLS_ATTESTATION = canonical_no_tools_runtime(
    LAUNCHER_DIGEST
)


def _many_object_panel() -> bytes:
    image = Image.new("RGB", (192, 128), "white")
    draw = ImageDraw.Draw(image)
    for index in range(5):
        x = 10 + (index % 4) * 42
        y = 10 + (index // 4) * 50
        draw.rectangle((x, y, x + 10, y + 10), outline="black", width=2)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


@pytest.fixture(scope="module")
def packet_and_atlas():
    panel = _many_object_panel()
    packet = extract_object_hypothesis_packet(panel)
    atlas = render_object_hypothesis_atlas(packet, panel)
    plan = plan_prototype_object_feature_shards(packet)
    assert len(plan.shards) == 2
    return packet, atlas, plan


def _runtime_kwargs() -> dict[str, object]:
    return {
        "model": DEFAULT_CODEX_MODEL,
        "reasoning_effort": EFFORT,
        "minutes": 3,
        "verbose": False,
        "executable": "/private/synthetic-codex",
        "cloud_policy_cache_snapshot": None,
        "model_catalog_snapshot": MODEL_CATALOG,
        "expected_launcher_digest": LAUNCHER_DIGEST,
        "tool_surface_attestation": NO_TOOLS_ATTESTATION,
        "expected_tool_surface_attestation_digest": (
            NO_TOOLS_ATTESTATION.attestation_digest
        ),
    }


def _payload(packet, spec) -> dict[str, object]:
    sheet = next(item for item in packet.atlas_sheets if item.name == spec.sheet_name)
    return {
        "description": "Several isolated angular outlines are visible.",
        "rows": [
            {
                "slot_id": slot.slot_id,
                "states": ["s" for _ in spec.feature_ids],
                "lowers": [spec.shard_index for _ in spec.feature_ids],
                "uppers": [spec.shard_index for _ in spec.feature_ids],
            }
            for slot in sheet.slots
        ],
    }


class _Transport:
    def __init__(self, packet, plan, *, fail_indices: Sequence[int] = ()) -> None:
        self.packet = packet
        self.plan = plan
        self.fail_indices = frozenset(fail_indices)
        self.calls = 0

    def __call__(
        self,
        prompt: str,
        paths: Sequence[str],
        names: Sequence[str],
        schema: Mapping[str, Any],
        **kwargs: object,
    ) -> CodexStructuredResult:
        index = self.calls
        self.calls += 1
        if index in self.fail_indices:
            raise OSError("unstable diagnostic text must not enter the journal")
        spec = self.plan.shards[index]
        payload = _payload(self.packet, spec)
        receipt = canonical_codex_receipt(
            prompt,
            paths,
            schema,
            payload,
            launcher_digest=LAUNCHER_DIGEST,
            reasoning_effort=EFFORT,
            model=DEFAULT_CODEX_MODEL,
            names=names,
        )
        return CodexStructuredResult(payload=payload, receipt=receipt)


class _CrashTransport:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> CodexStructuredResult:
        self.calls += 1
        raise KeyboardInterrupt("process disappeared before terminalization")


def _paths(tmp_path: Path, atlas) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, data in atlas:
        path = tmp_path / name
        path.write_bytes(data)
        result[name] = str(path.resolve())
    return result


def _journal(tmp_path, packet, atlas, transport):
    return ObjectBongardShardJournalTransport(
        tmp_path / "journal",
        authorization_digest="sha256:" + "a" * 64,
        precommit_digest="sha256:" + "b" * 64,
        context_digest="c" * 64,
        panel_id="bd/task/1/0.png",
        packet=packet,
        atlas=atlas,
        expected_transport_kwargs=_runtime_kwargs(),
        underlying_transport=transport,
    )


def _invoke(journal, packet, spec, paths):
    return journal(
        prototype_object_feature_shard_prompt(packet, spec),
        (paths[spec.sheet_name],),
        (spec.sheet_name,),
        prototype_object_feature_output_schema(),
        **_runtime_kwargs(),
    )


def test_two_physical_shards_are_separately_claimed_and_cold_reused(
    tmp_path: Path, packet_and_atlas
) -> None:
    packet, atlas, plan = packet_and_atlas
    paths = _paths(tmp_path, atlas)
    fresh_transport = _Transport(packet, plan)
    first = _journal(tmp_path, packet, atlas, fresh_transport)

    outputs = [_invoke(first, packet, spec, paths) for spec in plan.shards]

    assert fresh_transport.calls == 2
    assert first.attempted_call_count == first.fresh_call_count == 2
    assert first.reused_call_count == first.refused_call_count == 0
    assert len(first.ordered_call_keys) == 2
    assert len(set(first.ordered_call_keys)) == 2
    summary = verify_object_bongard_shard_journal(first)
    assert summary.success_count == 2
    assert summary.failure_count == 0
    assert summary.terminal_call_keys == first.ordered_call_keys

    forbidden_transport = _Transport(packet, plan, fail_indices=(0, 1))
    replay = _journal(tmp_path, packet, atlas, forbidden_transport)
    replayed = [_invoke(replay, packet, spec, paths) for spec in plan.shards]

    assert forbidden_transport.calls == 0
    assert [item.payload for item in replayed] == [item.payload for item in outputs]
    assert replay.fresh_call_count == 0
    assert replay.reused_call_count == replay.attempted_call_count == 2
    assert replay.ordered_call_keys == first.ordered_call_keys


def test_partial_success_then_failure_is_terminal_and_zero_call_replay(
    tmp_path: Path, packet_and_atlas
) -> None:
    packet, atlas, plan = packet_and_atlas
    paths = _paths(tmp_path, atlas)
    transport = _Transport(packet, plan, fail_indices=(1,))
    journal = _journal(tmp_path, packet, atlas, transport)

    _invoke(journal, packet, plan.shards[0], paths)
    with pytest.raises(ObjectBongardShardCallFailed) as first_failure:
        _invoke(journal, packet, plan.shards[1], paths)

    assert transport.calls == 2
    assert str(first_failure.value) == (
        "physical object-observer shard transport failed"
    )
    summary = journal.verify()
    assert summary.success_count == 1
    assert summary.failure_count == 1

    forbidden = _Transport(packet, plan, fail_indices=(0, 1))
    replay = _journal(tmp_path, packet, atlas, forbidden)
    _invoke(replay, packet, plan.shards[0], paths)
    with pytest.raises(ObjectBongardShardCallFailed) as replayed_failure:
        _invoke(replay, packet, plan.shards[1], paths)

    assert forbidden.calls == 0
    assert replayed_failure.value.failure_digest == first_failure.value.failure_digest
    assert replayed_failure.value.call_key == first_failure.value.call_key
    assert replay.reused_call_count == 2


def test_preexisting_nonterminal_claim_refuses_transport(
    tmp_path: Path, packet_and_atlas
) -> None:
    packet, atlas, plan = packet_and_atlas
    paths = _paths(tmp_path, atlas)
    crashing = _CrashTransport()
    journal = _journal(tmp_path, packet, atlas, crashing)

    with pytest.raises(KeyboardInterrupt):
        _invoke(journal, packet, plan.shards[0], paths)
    assert crashing.calls == 1
    summary = journal.verify()
    assert summary.nonterminal_call_keys == (
        journal.call_key_for_spec(plan.shards[0]),
    )

    forbidden = _Transport(packet, plan, fail_indices=(0, 1))
    restarted = _journal(tmp_path, packet, atlas, forbidden)
    with pytest.raises(ObjectBongardShardNonterminalClaim, match="rerun is forbidden"):
        _invoke(restarted, packet, plan.shards[0], paths)

    assert forbidden.calls == 0
    assert restarted.attempted_call_count == 1
    assert restarted.fresh_call_count == restarted.reused_call_count == 0
    assert restarted.refused_call_count == 1


@pytest.mark.parametrize("mutation", ["prompt", "schema", "image", "runtime"])
def test_exact_shard_envelope_is_checked_before_claim(
    tmp_path: Path, packet_and_atlas, mutation: str
) -> None:
    packet, atlas, plan = packet_and_atlas
    paths = _paths(tmp_path, atlas)
    transport = _Transport(packet, plan)
    journal = _journal(tmp_path, packet, atlas, transport)
    spec = plan.shards[0]
    prompt = prototype_object_feature_shard_prompt(packet, spec)
    schema = prototype_object_feature_output_schema()
    kwargs = _runtime_kwargs()
    if mutation == "prompt":
        prompt += " altered"
    elif mutation == "schema":
        schema = deepcopy(schema)
        schema["description"] = "altered"
    elif mutation == "image":
        Path(paths[spec.sheet_name]).write_bytes(b"not the committed atlas")
    else:
        kwargs["minutes"] = 4

    with pytest.raises(ObjectBongardShardJournalError):
        journal(
            prompt,
            (paths[spec.sheet_name],),
            (spec.sheet_name,),
            schema,
            **kwargs,
        )

    assert transport.calls == 0
    assert journal.attempted_call_count == 0
    assert journal.verify().terminal_call_keys == ()


def test_cold_verifier_rejects_canonical_digest_recomputed_tamper(
    tmp_path: Path, packet_and_atlas
) -> None:
    packet, atlas, plan = packet_and_atlas
    paths = _paths(tmp_path, atlas)
    journal = _journal(tmp_path, packet, atlas, _Transport(packet, plan))
    _invoke(journal, packet, plan.shards[0], paths)
    call_key = journal.call_key_for_spec(plan.shards[0])
    _claim_path, result_path, _outcome_path = journal.record_paths(call_key)
    record = json.loads(result_path.read_text(encoding="utf-8"))
    record["receipt"]["requested_reasoning_effort"] = "high"
    body = {key: value for key, value in record.items() if key != "record_digest"}
    from bongard.canonical import canonical_digest, canonical_json

    record["record_digest"] = "sha256:" + canonical_digest(body)
    result_path.write_bytes(canonical_json(record) + b"\n")

    with pytest.raises(ObjectBongardShardJournalError):
        journal.verify()
    with pytest.raises(ObjectBongardShardJournalError):
        _journal(tmp_path, packet, atlas, _Transport(packet, plan)).verify()


def test_manifest_tamper_is_rejected_during_cold_construction(
    tmp_path: Path, packet_and_atlas
) -> None:
    packet, atlas, plan = packet_and_atlas
    journal = _journal(tmp_path, packet, atlas, _Transport(packet, plan))
    manifest_path = journal.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["panel_id"] = "bd/another/1/0.png"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ObjectBongardShardJournalError, match="manifest"):
        _journal(tmp_path, packet, atlas, _Transport(packet, plan))

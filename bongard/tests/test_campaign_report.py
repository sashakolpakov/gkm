from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bongard.artifacts import canonical_digest, canonical_json
from bongard.campaign_report import (
    CAMPAIGN_REPORT_SCHEMA,
    CAMPAIGN_TYPE,
    EXACT_SUPPORT_SCOPE,
    INFRASTRUCTURE_SCOPE,
    UNVERIFIED,
    VERIFIED,
    CampaignReportError,
    CampaignRunInput,
    build_campaign_report,
)


REPOSITORY = Path(__file__).resolve().parents[2]
RUNS = REPOSITORY / "bongard" / "runs" / "official_complete_drill_20260805"
DATA = REPOSITORY / "bongard" / "data"
RUN_FILENAMES = (
    "bd_asymm_trap_bridge-trans_arc_cup_0000_v10.json",
    "bd_asymm_unbala_goldfish-asymmetric_crown_0000_v11.json",
    "hd_closed_shape-has_obtuse_angle_0000_v12.json",
)


def _records() -> list[dict[str, object]]:
    return [json.loads((RUNS / filename).read_bytes()) for filename in RUN_FILENAMES]


def _inputs(records: list[dict[str, object]] | None = None) -> list[CampaignRunInput]:
    values = _records() if records is None else records
    result = []
    for index, (filename, record) in enumerate(zip(RUN_FILENAMES, values, strict=True)):
        payload = (RUNS / filename).read_bytes()
        result.append(
            CampaignRunInput(
                record=record,
                file_sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
                verification_disposition=UNVERIFIED if index == 0 else VERIFIED,
                verification_scope=(
                    INFRASTRUCTURE_SCOPE if index == 0 else EXACT_SUPPORT_SCOPE
                ),
            )
        )
    return result


def _readdress(record: dict[str, object]) -> None:
    record["record_digest"] = canonical_digest(
        {key: value for key, value in record.items() if key != "record_digest"}
    )


def test_official_v10_v12_campaign_reproduces_checked_report() -> None:
    report = build_campaign_report(
        _inputs(), campaign_id="official_complete_drill_smoke_v10_v12"
    ).to_dict()
    checked_bytes = (
        DATA / "official_complete_drill_smoke_v10_v12.json"
    ).read_bytes()
    checked = json.loads(checked_bytes)

    assert checked_bytes == canonical_json(checked) + b"\n"
    assert report == checked
    assert report["schema"] == CAMPAIGN_REPORT_SCHEMA
    assert report["campaign_type"] == CAMPAIGN_TYPE
    assert report["digest"] == "sha256:" + canonical_digest(
        {key: value for key, value in report.items() if key != "digest"}
    )


def test_stage_support_and_unique_receipt_totals_are_exact() -> None:
    report = build_campaign_report(
        list(reversed(_inputs())),
        campaign_id="official_complete_drill_smoke_v10_v12",
    ).to_dict()

    assert report["stages"] == {
        "attempts": 3,
        "proposer_successes": 2,
        "support_gate_replays": 2,
        "support_gate_passes": 0,
        "query_releases": 0,
        "completions": 0,
    }
    assert report["support_replay"] == {
        "forward_matches": 15,
        "reverse_matches": 9,
        "present": 19,
        "nonmatch": 5,
        "indeterminate": 0,
        "error": 0,
        "transport_attempts": 24,
        "panels": 24,
        "verified_support_rejections": 2,
    }
    assert report["transport_usage"] == {
        "successful_receipts": 26,
        "input_tokens": 233921,
        "cached_input_tokens": 23552,
        "output_tokens": 15001,
        "reasoning_output_tokens": 10694,
    }
    assert report["exposure_chain"] == {
        "initial_ledger_digest": (
            "sha256:65c8dd508f6c21e64b0c777a83159a470fbab12cfb8fee6adf588c0a9c400c8b"
        ),
        "final_ledger_digest": (
            "sha256:da01c133c87c551e01b581578b55d40283be0c62cbb23dddc18c5dc873b1ec9a"
        ),
        "initial_event_count": 13,
        "final_event_count": 16,
        "event_digests": [
            "sha256:dee13f7dae4e949882f516b8e8ca54eec7af8db0aa1fc47ca8a90aadb50195d7",
            "sha256:395fbacc33c3bc206a581e2d85cf856b89e978ce6133a3a2574e193d6d7484ab",
            "sha256:ce8f67fc54e3775932951c622d9f87dac805a12ac082bc66f5bc258764492c2e",
        ],
    }


def test_no_query_release_emits_no_score_or_accuracy_metrics() -> None:
    report = build_campaign_report(
        _inputs(), campaign_id="official_complete_drill_smoke_v10_v12"
    ).to_dict()

    assert report["stages"]["query_releases"] == 0
    assert "score" not in report
    assert "accuracy" not in report
    assert "query accuracy" in report["interpretation"]["does_not_measure"]
    first, second, third = report["runs"]
    assert first["verification"] == {
        "disposition": "unverified",
        "scope": "infrastructure-failure-record-integrity-only",
        "outer_record_integrity_verified": True,
        "cold_replay_verified": False,
    }
    assert all(
        run["verification"]["scope"]
        == "exact-official-support-rejection-byte-preimage-gate-replay"
        and run["verification"]["cold_replay_verified"] is True
        for run in (second, third)
    )


def test_record_digest_uniqueness_and_exact_exposure_chain_fail_closed() -> None:
    records = _records()
    records[0]["record_digest"] = "0" * 64
    with pytest.raises(CampaignReportError, match="record_digest does not reproduce"):
        build_campaign_report(_inputs(records), campaign_id="tampered")

    inputs = _inputs()
    with pytest.raises(CampaignReportError, match="run_id values must be unique"):
        build_campaign_report([inputs[0], inputs[0]], campaign_id="duplicate")

    records = _records()
    damaged = copy.deepcopy(records[2])
    exposure = damaged["exposure"]
    assert isinstance(exposure, dict)
    replacement = "sha256:" + "1" * 64
    exposure["ledger_before_digest"] = replacement
    unseen = exposure["semantic_unseen_receipt"]
    assert isinstance(unseen, dict)
    unseen["ledger_digest"] = replacement
    _readdress(damaged)
    records[2] = damaged
    with pytest.raises(
        CampaignReportError, match="predecessor/successor chain is not exact"
    ):
        build_campaign_report(_inputs(records), campaign_id="broken-chain")


def test_verification_scope_and_identity_mismatches_fail_closed() -> None:
    inputs = _inputs()
    invalid_scope = CampaignRunInput(
        record=inputs[0].record,
        file_sha256=inputs[0].file_sha256,
        verification_disposition=VERIFIED,
        verification_scope=EXACT_SUPPORT_SCOPE,
    )
    with pytest.raises(CampaignReportError, match="incompatible with its run"):
        build_campaign_report(
            [invalid_scope, inputs[1], inputs[2]], campaign_id="bad-scope"
        )

    records = _records()
    damaged = copy.deepcopy(records[2])
    exposure = damaged["exposure"]
    assert isinstance(exposure, dict)
    exposure["model"] = "gpt-other"
    _readdress(damaged)
    records[2] = damaged
    with pytest.raises(CampaignReportError, match="actor differs from model"):
        build_campaign_report(_inputs(records), campaign_id="bad-model")

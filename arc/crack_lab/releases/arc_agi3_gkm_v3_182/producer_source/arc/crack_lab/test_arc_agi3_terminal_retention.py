from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
import uuid
from pathlib import Path

import pytest

import arc_agi3_contiguous_runner as R


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _runner_receipt(
    root: Path,
    *,
    complete: bool = True,
    generation_ids: list[str] | None = None,
) -> dict:
    identities = sorted(generation_ids or [])
    body = {
        "schema": 1,
        "kind": "arc_agi3_contiguous_runner_state_audit",
        "status": "PASS",
        "campaign_root": str(root),
        "campaign_id": "terminal-retention-test",
        "journal_head_sequence": 17,
        "journal_head_digest": "1" * 64,
        "solved_levels": 183 if complete else 0,
        "total_levels": 183,
        "complete": complete,
        "attempt_ids": identities,
        "generation_ids": identities,
        "lane_boundaries": [],
    }
    return {
        **body,
        "receipt_sha256": hashlib.sha256(_canonical(body)).hexdigest(),
    }


def _write_json(path: Path, value: object) -> tuple[str, int]:
    payload = _canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _intent(
    root: Path,
    receipt: dict,
    generations: list[tuple[str, str, int]],
    *,
    pre_cleanup_audits: dict[str, str] | None = None,
) -> dict:
    prerequisites = dict(pre_cleanup_audits or {})
    exports = []
    for generation_id, digest, byte_count in generations:
        exports.append(
            {
                "attempt_id": generation_id,
                "generation_id": generation_id,
                "evidence_sha256": digest,
                "byte_count": byte_count,
                "references": ["collection.taint_scan_receipt"],
                "source_relative_paths": [
                    "host/taint_scan_receipt.json"
                ],
                "source_campaign_relative_paths": [],
                "source_absolute_paths": [],
                "retained_relative_path":
                    f"{generation_id}/{digest}.json",
            }
        )
    exports.sort(
        key=lambda item: (
            item["attempt_id"], item["retained_relative_path"]
        )
    )
    policy = {
        "phase": "final_campaign_only",
        "copy_all_compact_exports_before_first_purge": True,
        "generation_scratch_retained": False,
        "workspace_retained": False,
        "cache_retained": False,
        "raw_transcripts_retained": False,
        "stdout_stderr_retained": False,
        "invalid_attempt_raw_bytes_retained": False,
        "promotion_and_replay_authority":
            "external_unified_promotion_audit",
        "wip_needed_midcampaign": True,
    }
    body = {
        "schema": R.TERMINAL_RETENTION_SCHEMA,
        "kind": "arc_agi3_terminal_attempt_retention_intent",
        "campaign_root": str(root),
        "campaign_id": receipt["campaign_id"],
        "runner_state_receipt": dict(receipt),
        "runner_state_receipt_sha256": receipt["receipt_sha256"],
        "journal_head_sequence": receipt["journal_head_sequence"],
        "journal_head_digest": receipt["journal_head_digest"],
        "solved_levels": receipt["solved_levels"],
        "total_levels": receipt["total_levels"],
        "complete": True,
        "generation_ids": sorted(item[0] for item in generations),
        "attempt_ids": sorted(item[0] for item in generations),
        "compact_evidence_root": R.TERMINAL_RETENTION_EVIDENCE_NAME,
        "compact_exports": exports,
        "compact_exports_sha256":
            hashlib.sha256(_canonical(exports)).hexdigest(),
        "compact_export_bytes": sum(item[2] for item in generations),
        "lane_authorities": [],
        "lane_authorities_sha256":
            hashlib.sha256(_canonical([])).hexdigest(),
        "pre_cleanup_audits": prerequisites,
        "pre_cleanup_audits_sha256":
            hashlib.sha256(_canonical(prerequisites)).hexdigest(),
        "retention_policy": policy,
    }
    return {
        **body,
        "intent_sha256": hashlib.sha256(_canonical(body)).hexdigest(),
    }


def _reseal_intent(value: dict) -> dict:
    body = {
        key: item
        for key, item in value.items()
        if key != "intent_sha256"
    }
    value["intent_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return value


def _campaign(tmp_path: Path, count: int = 2):
    root = tmp_path / "campaign"
    generations_root = root / "generations"
    generations_root.mkdir(parents=True)
    outside = tmp_path / "outside-must-survive.txt"
    outside.write_text("outside generation\n", encoding="utf-8")
    rows = []
    for index in range(count):
        generation_id = str(uuid.uuid4())
        generation = generations_root / generation_id
        digest, byte_count = _write_json(
            generation / "host" / "taint_scan_receipt.json",
            {"schema": 1, "status": "PASS", "index": index},
        )
        (generation / "scratch" / "cache").mkdir(parents=True)
        (generation / "scratch" / "cache" / "solver.tmp").write_text(
            "not scientific evidence", encoding="utf-8"
        )
        (generation / "host" / "app_server.jsonl").write_text(
            '{"raw":"transcript"}\n', encoding="utf-8"
        )
        os.symlink(
            outside,
            generation / "scratch" / "cache" / "outside-link",
        )
        rows.append((generation_id, digest, byte_count))
    return root, rows


def _patch_read_only_audit(monkeypatch, receipt: dict, intent: dict):
    monkeypatch.setattr(
        R,
        "verify_runner_state_audit",
        lambda value, **_kwargs: receipt
        if value == receipt
        else (_ for _ in ()).throw(
            R.ContiguousRunnerError("wrong runner receipt")
        ),
    )
    monkeypatch.setattr(
        R,
        "_terminal_retention_state",
        lambda *_args, **_kwargs: {"complete": True},
    )
    monkeypatch.setattr(
        R,
        "_terminal_retention_plan",
        lambda *_args, **_kwargs: intent,
    )
    monkeypatch.setattr(
        R,
        "_terminal_retention_recovery_runner_receipt",
        lambda _root, *, expected_receipt=None: receipt
        if expected_receipt in (None, receipt)
        else (_ for _ in ()).throw(
            R.ContiguousRunnerError("wrong recovery runner receipt")
        ),
    )


@pytest.mark.parametrize(
    "field",
    (
        "copy_all_compact_exports_before_first_purge",
        "generation_scratch_retained",
        "workspace_retained",
        "cache_retained",
        "raw_transcripts_retained",
        "stdout_stderr_retained",
        "invalid_attempt_raw_bytes_retained",
        "wip_needed_midcampaign",
    ),
)
@pytest.mark.parametrize("replacement", (0, 1))
def test_terminal_retention_policy_requires_literal_booleans(
    tmp_path, field, replacement,
):
    root, rows = _campaign(tmp_path, count=1)
    receipt = _runner_receipt(
        root, generation_ids=[item[0] for item in rows]
    )
    intent = _intent(root, receipt, rows)
    assert R._validate_terminal_retention_intent(
        intent,
        campaign_root=root,
        runner_state_receipt=receipt,
        pre_cleanup_audits={},
    ) == intent
    forged = copy.deepcopy(intent)
    forged["retention_policy"][field] = replacement
    _reseal_intent(forged)
    with pytest.raises(
        R.ContiguousRunnerError, match="retention policy"
    ):
        R._validate_terminal_retention_intent(
            forged,
            campaign_root=root,
            runner_state_receipt=receipt,
            pre_cleanup_audits={},
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("invalid_attempt_raw_bytes_retained", None),
        ("invalid_attempt_raw_bytes_retained", "false"),
        ("phase", False),
        ("phase", 0),
        ("promotion_and_replay_authority", False),
        ("promotion_and_replay_authority", 0),
    ),
)
def test_terminal_retention_policy_rejects_missing_or_wrong_types(
    tmp_path, field, replacement,
):
    root, rows = _campaign(tmp_path, count=1)
    receipt = _runner_receipt(
        root, generation_ids=[item[0] for item in rows]
    )
    intent = _intent(root, receipt, rows)
    forged = copy.deepcopy(intent)
    if replacement is None:
        forged["retention_policy"].pop(field)
    else:
        forged["retention_policy"][field] = replacement
    _reseal_intent(forged)
    with pytest.raises(
        R.ContiguousRunnerError, match="retention policy"
    ):
        R._validate_terminal_retention_intent(
            forged,
            campaign_root=root,
            runner_state_receipt=receipt,
            pre_cleanup_audits={},
        )


def test_terminal_retention_crash_recovery_is_copy_before_purge(
    tmp_path, monkeypatch
):
    root, rows = _campaign(tmp_path)
    receipt = _runner_receipt(
        root, generation_ids=[item[0] for item in rows]
    )
    prerequisites = {"scheduler": "9" * 64}
    intent = _intent(
        root,
        receipt,
        rows,
        pre_cleanup_audits=prerequisites,
    )
    _patch_read_only_audit(monkeypatch, receipt, intent)

    original_rmtree = shutil.rmtree
    calls = 0

    def interrupted_rmtree(path, *args, **kwargs):
        nonlocal calls
        calls += 1
        original_rmtree(path, *args, **kwargs)
        if calls == 1:
            raise RuntimeError("synthetic crash after first purge")

    interrupted_rmtree.avoids_symlink_attacks = True
    monkeypatch.setattr(R.shutil, "rmtree", interrupted_rmtree)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        R.finalize_terminal_attempt_retention(
            root,
            receipt,
            pre_cleanup_audits=prerequisites,
        )

    evidence = root / R.TERMINAL_RETENTION_EVIDENCE_NAME
    assert (root / R.TERMINAL_RETENTION_INTENT_NAME).is_file()
    assert len(list(evidence.rglob("*.json"))) == len(rows)
    assert len(list((root / "generations").iterdir())) == len(rows) - 1
    assert not (root / R.TERMINAL_RETENTION_RECEIPT_NAME).exists()

    monkeypatch.setattr(R.shutil, "rmtree", original_rmtree)
    monkeypatch.setattr(
        R,
        "verify_runner_state_audit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(
                "partial purge must recover from its sealed intent"
            )
        ),
    )
    final = R.finalize_terminal_attempt_retention(
        root,
        receipt,
        pre_cleanup_audits=prerequisites,
    )
    assert final["status"] == "PASS"
    assert not list((root / "generations").iterdir())
    assert (tmp_path / "outside-must-survive.txt").read_text(
        encoding="utf-8"
    ) == "outside generation\n"
    assert final["generation_scratch_survivors"] == 0
    assert final["raw_stream_survivors"] == 0
    assert len(final["compact_evidence_inventory"]) == len(rows)
    assert R.audit_terminal_attempt_retention(
        root,
        receipt,
        pre_cleanup_audits=prerequisites,
    ) == final
    for path in (
        root / R.TERMINAL_RETENTION_INTENT_NAME,
        root / R.TERMINAL_RETENTION_RECEIPT_NAME,
        evidence,
        *evidence.rglob("*"),
    ):
        assert not path.is_symlink()
        assert stat.S_IMODE(
            path.stat(follow_symlinks=False).st_mode
        ) & 0o222 == 0
    assert not any(
        token in path.name.lower()
        for path in evidence.rglob("*")
        for token in (
            "scratch", "workspace", "cache", "transcript",
            "stdout", "stderr", "jsonl",
        )
    )
    # A crash after exact receipt installation but before mode sealing is
    # recoverable without regenerating or widening any evidence.
    retained_receipt = root / R.TERMINAL_RETENTION_RECEIPT_NAME
    retained_intent = root / R.TERMINAL_RETENTION_INTENT_NAME
    os.chmod(retained_receipt, 0o600)
    os.chmod(retained_intent, 0o600)
    assert R.finalize_terminal_attempt_retention(
        root,
        receipt,
        pre_cleanup_audits=prerequisites,
    ) == final
    assert stat.S_IMODE(retained_receipt.stat().st_mode) == 0o400
    assert stat.S_IMODE(retained_intent.stat().st_mode) == 0o400
    with pytest.raises(
        R.ContiguousRunnerError,
        match="stale or malformed",
    ):
        R.audit_terminal_attempt_retention(
            root,
            receipt,
            pre_cleanup_audits={"scheduler": "8" * 64},
        )


def test_terminal_retention_missing_export_fails_before_any_purge(
    tmp_path, monkeypatch
):
    root, rows = _campaign(tmp_path, count=1)
    receipt = _runner_receipt(
        root, generation_ids=[item[0] for item in rows]
    )
    intent = _intent(root, receipt, rows)
    source = (
        root
        / "generations"
        / rows[0][0]
        / "host"
        / "taint_scan_receipt.json"
    )
    source.unlink()
    _patch_read_only_audit(monkeypatch, receipt, intent)

    with pytest.raises(
        R.ContiguousRunnerError,
        match="compact evidence is missing",
    ):
        R.finalize_terminal_attempt_retention(root, receipt)
    assert (root / "generations" / rows[0][0]).is_dir()
    assert not (root / R.TERMINAL_RETENTION_RECEIPT_NAME).exists()


def test_terminal_retention_never_mutates_incomplete_campaign(
    tmp_path, monkeypatch
):
    root, _ = _campaign(tmp_path, count=1)
    receipt = _runner_receipt(root, complete=False)
    monkeypatch.setattr(
        R,
        "verify_runner_state_audit",
        lambda *_args, **_kwargs: receipt,
    )
    before = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
    )
    with pytest.raises(
        R.ContiguousRunnerError,
        match="forbidden before complete coverage",
    ):
        R.finalize_terminal_attempt_retention(root, receipt)
    after = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
    )
    assert before == after
    assert not (root / R.TERMINAL_RETENTION_INTENT_NAME).exists()


def test_terminal_retention_rejects_generation_symlink_before_intent(
    tmp_path, monkeypatch
):
    root = tmp_path / "campaign"
    (root / "generations").mkdir(parents=True)
    outside = tmp_path / "outside"
    digest, byte_count = _write_json(
        outside / "host" / "taint_scan_receipt.json",
        {"schema": 1, "status": "PASS"},
    )
    generation_id = str(uuid.uuid4())
    os.symlink(outside, root / "generations" / generation_id)
    receipt = _runner_receipt(
        root, generation_ids=[generation_id]
    )
    intent = _intent(
        root, receipt, [(generation_id, digest, byte_count)]
    )
    _patch_read_only_audit(monkeypatch, receipt, intent)

    with pytest.raises(
        R.ContiguousRunnerError,
        match="source generation is unavailable",
    ):
        R.finalize_terminal_attempt_retention(root, receipt)
    assert (outside / "host" / "taint_scan_receipt.json").is_file()
    assert not (root / R.TERMINAL_RETENTION_RECEIPT_NAME).exists()


def test_terminal_retention_rejects_import_cache_in_compact_archive(
    tmp_path, monkeypatch
):
    root, rows = _campaign(tmp_path, count=1)
    receipt = _runner_receipt(
        root, generation_ids=[item[0] for item in rows]
    )
    prerequisites = {"scheduler": "9" * 64}
    intent = _intent(
        root,
        receipt,
        rows,
        pre_cleanup_audits=prerequisites,
    )
    _patch_read_only_audit(monkeypatch, receipt, intent)
    final = R.finalize_terminal_attempt_retention(
        root,
        receipt,
        pre_cleanup_audits=prerequisites,
    )
    assert final["status"] == "PASS"

    evidence = root / R.TERMINAL_RETENTION_EVIDENCE_NAME
    attempt_root = evidence / rows[0][0]
    os.chmod(evidence, 0o700)
    os.chmod(attempt_root, 0o700)
    cache = attempt_root / "__pycache__"
    cache.mkdir(mode=0o700)
    poisoned = cache / "solver.cpython-312.pyc"
    poisoned.write_bytes(b"not retained scientific evidence")
    poisoned.chmod(0o400)
    cache.chmod(0o500)
    os.chmod(attempt_root, 0o500)
    os.chmod(evidence, 0o500)

    with pytest.raises(
        R.ContiguousRunnerError,
        match="extra/missing",
    ):
        R.audit_terminal_attempt_retention(
            root,
            receipt,
            pre_cleanup_audits=prerequisites,
        )

import copy
import datetime as dt
import hashlib
import json

import pytest

import arc_agi3_leaderboard_v3_gate as G

REAL_VERIFY_PUBLIC_REVISION = G.verify_public_revision


@pytest.fixture(autouse=True)
def _public_revision_is_reachable(monkeypatch):
    monkeypatch.setattr(
        G,
        "verify_public_revision",
        lambda revision: {
            "sha": revision,
            "html_url": f"https://github.com/sashakolpakov/gkm/commit/{revision}",
        },
    )


INVENTORY = {
    "ar25": 8,
    "bp35": 9,
    "cd82": 6,
    "cn04": 6,
    "dc22": 6,
    "ft09": 6,
    "g50t": 7,
    "ka59": 7,
    "lf52": 10,
    "lp85": 8,
    "ls20": 7,
    "m0r0": 6,
    "r11l": 6,
    "re86": 8,
    "s5i5": 8,
    "sb26": 8,
    "sc25": 6,
    "sk48": 8,
    "sp80": 6,
    "su15": 9,
    "tn36": 7,
    "tr87": 6,
    "tu93": 9,
    "vc33": 7,
    "wa30": 9,
}
RELEASE_SHA = "e" * 64
TREE_SHA = "f" * 64
REVISION = "d" * 40
ONLINE_ID = "11111111-1111-4111-8111-111111111111"
COMPETITION_ID = "22222222-2222-4222-8222-222222222222"
ONLINE_OPERATION_ID = "1" * 64
COMPETITION_OPERATION_ID = "2" * 64
VERIFIER_SHA = "3" * 64
CONTROL_SHA = "4" * 64
STORED_ACTIONS = sum(INVENTORY.values())


def _authors():
    return [
        {
            "name": "Alexander Kolpakov",
            "github": "https://github.com/sashakolpakov",
            "affiliation": "Independent",
        },
        {"name": "OpenAI GPT-5.6", "url": "https://openai.com/"},
    ]


def _baseline():
    return {
        "name": "Gödel–Kolmogorov Machine (GKM)",
        "authors": _authors(),
        "description": "General-purpose program-growth architecture.",
        "code_url": "https://github.com/sashakolpakov/gkm/tree/" + "a" * 40,
        "paper_url": "https://arxiv.org/abs/2601.00001",
        "versions": [
            {
                "version": "1.0",
                "date": dt.date(2026, 7, 15),
                "changes": "Initial release",
                "models": [{"name": "Claude Code"}],
                "scores": [
                    {
                        "benchmark": "arc-agi-3",
                        "scorecard_url": G.SCORECARD_URL_PREFIX + ONLINE_ID,
                        "set": "public",
                    }
                ],
            },
            {
                "version": "2.0",
                "date": dt.date(2026, 7, 31),
                "changes": "Frozen 181/183 release",
                "models": [{"name": "OpenAI GPT-5.6-sol"}],
                "scores": [
                    {
                        "benchmark": "arc-agi-3",
                        "scorecard_url": G.SCORECARD_URL_PREFIX
                        + "33333333-3333-4333-8333-333333333333",
                        "set": "public",
                    }
                ],
            },
        ],
    }


def _score_environment(game, depth, *, public, with_initial=False):
    environment_id = f"{game}-deadbeef"
    run = {
        "id": environment_id,
        "guid": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "score": 100.0,
        "levels_completed": depth,
        "actions": depth,
        "resets": 0 if with_initial else 1,
        "state": "WIN",
        "completed": True,
        "level_scores": [100.0] * depth,
        "level_actions": [1] * depth,
        "level_baseline_actions": [1] * depth,
        "number_of_levels": depth,
        "number_of_environments": 1,
    }
    if not public:
        run["message"] = None
    runs = [run]
    if with_initial:
        initial = copy.deepcopy(run)
        initial.update(
            {
                "score": 0.0,
                "levels_completed": 0,
                "actions": 0,
                "resets": 0,
                "state": "NOT_FINISHED",
                "completed": False,
                "level_scores": [0.0] * depth,
                "level_actions": [0] * depth,
            }
        )
        runs.insert(0, initial)
    return {
        "id": environment_id,
        "runs": runs,
        "score": 100.0,
        "actions": depth,
        "levels_completed": depth,
        "completed": True,
        "level_count": depth,
        "resets": 0 if with_initial else 1,
    }


def _closed_aggregate(mode, card_id, score, source_url, tags, opaque):
    environments = [
        _score_environment(
            game, depth, public=False, with_initial=mode == "online"
        )
        for game, depth in INVENTORY.items()
    ]
    return {
        "source_url": source_url,
        "tags": tags,
        "opaque": opaque,
        "card_id": card_id,
        "score": score,
        "environments": environments,
        "tags_scores": _tag_scores(public=False),
        "competition_mode": mode == "competition",
        "total_environments_completed": 25,
        "total_environments": 25,
        "total_levels_completed": 183,
        "total_levels": 183,
        "total_actions": 183,
    }


def _tag_scores(*, public):
    values = []
    for index, score_id in enumerate(
        ("click", "keyboard", "keyboard_click"), start=1
    ):
        value = {
            "id": score_id,
            "guid": f"{index}" * 8
            + "-1111-4111-8111-"
            + f"{index}" * 12,
            "score": 100.0,
            "levels_completed": 1,
            "actions": 1,
            "resets": 0,
            "state": "NOT_FINISHED",
            "completed": False,
            "number_of_levels": 1,
            "number_of_environments": 1,
        }
        if not public:
            value.update(
                {
                    "level_scores": None,
                    "level_actions": None,
                    "level_baseline_actions": None,
                    "message": None,
                }
            )
        values.append(value)
    return values


def _timestamp_offset(value, *, seconds):
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (parsed + dt.timedelta(seconds=seconds)).isoformat().replace(
        "+00:00", "Z"
    )


def _run(mode, card_id, *, score, started, closed):
    source_url = f"https://github.com/sashakolpakov/gkm/tree/{REVISION}"
    binding = {
        "binding_scope": "endpoint_checkpoint_bytes_only_after_full_gate",
        "receipt_sha256": RELEASE_SHA,
        "canonical_tree_sha256": TREE_SHA,
        "release_identity_source_revision": REVISION,
        "claimed_inventory": INVENTORY,
        "claimed_level_count": 183,
        "authoritative_level_count": 183,
    }
    release = _release()
    operation_id = (
        ONLINE_OPERATION_ID if mode == "online" else COMPETITION_OPERATION_ID
    )
    tags = G.expected_scorecard_tags(
        mode=mode,
        receipt_sha256=RELEASE_SHA,
        revision=REVISION,
    )
    opaque = G.expected_scorecard_opaque(
        operation_id=operation_id,
        mode=mode,
        receipt_sha256=RELEASE_SHA,
        canonical_tree_sha256=TREE_SHA,
        revision=REVISION,
    )
    return {
        "schema": 2,
        "mode": mode,
        "status": "PASS",
        "scorecard_id": card_id,
        "scorecard_url": G.SCORECARD_URL_PREFIX + card_id,
        "scorecard_open": {"status": "confirmed", "error_type": None},
        "scorecard_close": {"status": "confirmed", "error_type": None},
        "scorecard_tags": tags,
        "scorecard_opaque": opaque,
        "source_url": source_url,
        "source_revision": REVISION,
        "arc_agi_toolkit_version": "0.9.9",
        "started_at_utc": started,
        "closed_at_utc": closed,
        "scorecard_close_started_at_utc": _timestamp_offset(
            closed, seconds=-20
        ),
        "scorecard_close_finished_at_utc": _timestamp_offset(
            closed, seconds=-10
        ),
        "artifact_root": "arc/crack_lab/releases/v3/artifacts",
        "release_receipt": f"releases/receipts/{RELEASE_SHA}.json",
        "release_verification": {
            "status": "PASS",
            "games": 25,
            "levels": 183,
            "inventory_sha256": release["inventory_sha256"],
            "canonical_tree_sha256": TREE_SHA,
            "evidence_sha256": release["evidence_sha256"],
            "verifier_sha256": VERIFIER_SHA,
            "control_contract_sha256": CONTROL_SHA,
            "receipt_sha256": RELEASE_SHA,
            "verification_context_source_revision": REVISION,
        },
        "release_binding": binding,
        "checkpoint_sha256": {game: "a" * 64 for game in INVENTORY},
        "claimed_levels": 183,
        "authoritative_levels": 183,
        "stored_actions": STORED_ACTIONS,
        "command": {
            "entrypoint": "arc/crack_lab/replay_scorecard.py",
            "mode": mode,
            "games": list(INVENTORY),
            "artifact_root": "arc/crack_lab/releases/v3/artifacts",
            "release_receipt": f"releases/receipts/{RELEASE_SHA}.json",
            "expected_claimed_levels": 183,
            "preflight_only": False,
            "source_url": source_url,
            "source_revision": REVISION,
            "tags": tags,
        },
        "results": {
            game: {"remote": depth, "claimed": depth}
            for game, depth in INVENTORY.items()
        },
        "aggregate": _closed_aggregate(
            mode, card_id, score, source_url, tags, opaque
        ),
    }


def _public(run, *, published):
    competition = run["mode"] == "competition"
    value = {
        "source_url": run["source_url"],
        "card_id": run["scorecard_id"],
        "score": run["aggregate"]["score"],
        "published_at": published,
        "tags": run["scorecard_tags"],
        "opaque": run["scorecard_opaque"],
        "ai_agent": True,
        "environments": [
            _score_environment(
                game, depth, public=True, with_initial=not competition
            )
            for game, depth in INVENTORY.items()
        ],
        "tags_scores": _tag_scores(public=True),
        "open_at": run["started_at_utc"],
        "last_update": _timestamp_offset(published, seconds=-1),
        "total_environments_completed": 25,
        "total_environments": 25,
        "total_levels_completed": 183,
        "total_levels": 183,
        "total_actions": 183,
    }
    if competition:
        value["competition_mode"] = True
    return value


def _candidate(competition_run):
    candidate = copy.deepcopy(_baseline())
    candidate["code_url"] = (
        f"https://github.com/sashakolpakov/gkm/tree/{REVISION}"
    )
    candidate["versions"].append(
        {
            "version": "3.0",
            "date": dt.date(2026, 8, 3),
            "changes": "Complete schema-v2 183/183 release",
            "models": [
                {"name": "OpenAI GPT-5.6-sol (expanded campaign)"},
                {"name": "Claude Code (preserved legacy proposer lineages)"},
            ],
            "scores": [
                {
                    "benchmark": "arc-agi-3",
                    "scorecard_url": competition_run["scorecard_url"],
                    "set": "public",
                }
            ],
        }
    )
    return candidate


def _readme(competition_run):
    return f"""# GKM v3

Complete games: 25
Raw coverage: 183/183 (100%)
Frozen stored actions: {STORED_ACTIONS}
Official Competition actions: 183
Official Competition resets: 25
ARC toolkit: 0.9.9
Close recovery: none
Official score {competition_run['aggregate']['score']}.
OpenAI GPT-5.6. Revision {REVISION}. Receipt {RELEASE_SHA}.
Competition: {competition_run['scorecard_url']}

| Component | Origin/authoring agent | Admitted inputs | Transcript or source boundary | Verifier receipt | Promoted artifact |
|---|---|---|---|---|---|
| Solver | Native proposer | Public frames | Exact boundary | Receipt | Frozen release |
"""


def _release():
    identity = {"source_revision": REVISION}
    evidence = {
        game: [
            {"checkpoint_sha256": "a" * 64, "action_count": index + 1}
            for index in range(depth)
        ]
        for game, depth in INVENTORY.items()
    }
    return {
        "schema": 1,
        "release_identity": identity,
        "release_identity_sha256": G._json_sha256(identity),
        "canonical_game_count": 25,
        "authoritative_level_count": 183,
        "canonical_tree_sha256": TREE_SHA,
        "inventory": INVENTORY,
        "inventory_sha256": G._json_sha256(INVENTORY),
        "evidence": evidence,
        "evidence_sha256": G._json_sha256(evidence),
        "verifier": {"sha256": VERIFIER_SHA},
        "control_contract": {"sha256": CONTROL_SHA},
    }


def _append_journal_event(events, *, kind, operation_id, timestamp, payload):
    previous = events[-1]["sha256"] if events else None
    event = {
        "schema": 1,
        "journal_id": G.expected_journal_id(RELEASE_SHA),
        "sequence": len(events) + 1,
        "previous_event_sha256": previous,
        "event": kind,
        "operation_id": operation_id,
        "timestamp_utc": timestamp,
        "payload": payload,
    }
    line = G._canonical_json(event)
    record = {"event": event, "sha256": hashlib.sha256(line).hexdigest()}
    events.append(record)
    return record


def _snapshot_bytes(events):
    return b"".join(G._canonical_json(row["event"]) + b"\n" for row in events)


def _attach_journals(case):
    events = []
    for mode in ("online", "competition"):
        run = case[f"{mode}_run"]
        public = case[f"{mode}_public"]
        run.pop("run_journal", None)
        operation_id = (
            ONLINE_OPERATION_ID
            if mode == "online"
            else COMPETITION_OPERATION_ID
        )
        tags = G.expected_scorecard_tags(
            mode=mode,
            receipt_sha256=RELEASE_SHA,
            revision=REVISION,
        )
        opaque = G.expected_scorecard_opaque(
            operation_id=operation_id,
            mode=mode,
            receipt_sha256=RELEASE_SHA,
            canonical_tree_sha256=TREE_SHA,
            revision=REVISION,
        )
        run["scorecard_tags"] = tags
        run["scorecard_opaque"] = opaque
        public["tags"] = tags
        public["opaque"] = opaque
        public["ai_agent"] = True
        intent = _append_journal_event(
            events,
            kind="INTENT",
            operation_id=operation_id,
            timestamp=run["started_at_utc"],
            payload={
                "mode": mode,
                "source_url": run["source_url"],
                "source_revision": run["source_revision"],
                "arc_agi_toolkit_version": run["arc_agi_toolkit_version"],
                "release_receipt_sha256": RELEASE_SHA,
                "canonical_tree_sha256": TREE_SHA,
                "checkpoint_sha256_digest": G._json_sha256(
                    run["checkpoint_sha256"]
                ),
                "command_sha256": G._json_sha256(run["command"]),
                "output_receipt": (
                    "arc/crack_lab/run_journals/receipts/"
                    f"{RELEASE_SHA}/{mode}.json"
                ),
                "tags": tags,
                "opaque": opaque,
            },
        )
        opened = _append_journal_event(
            events,
            kind="OPENED",
            operation_id=operation_id,
            timestamp=run["started_at_utc"],
            payload={
                "card_id": run["scorecard_id"],
                "scorecard_url": run["scorecard_url"],
            },
        )
        receipt_core_sha256 = G._json_sha256(run)
        terminal_outcome = (
            "CLOSED_CONFIRMED_PASS"
            if run["status"] == "PASS"
            else "CLOSE_OUTCOME_AMBIGUOUS"
        )
        terminal = _append_journal_event(
            events,
            kind="TERMINAL",
            operation_id=operation_id,
            timestamp=run["closed_at_utc"],
            payload={
                "outcome": terminal_outcome,
                "card_id": run["scorecard_id"],
                "receipt_core_sha256": receipt_core_sha256,
            },
        )
        raw = _snapshot_bytes(events)
        snapshot_sha256 = hashlib.sha256(raw).hexdigest()
        run["run_journal"] = {
            "schema": 1,
            "journal_id": G.expected_journal_id(RELEASE_SHA),
            "live_journal": (
                f"arc/crack_lab/run_journals/{RELEASE_SHA}.jsonl"
            ),
            "snapshot": (
                "arc/crack_lab/run_journals/snapshots/"
                f"{snapshot_sha256}.jsonl"
            ),
            "snapshot_sha256": snapshot_sha256,
            "operation_id": operation_id,
            "opaque_sha256": G._json_sha256(opaque),
            "intent_sequence": intent["event"]["sequence"],
            "intent_event_sha256": intent["sha256"],
            "opened_sequence": opened["event"]["sequence"],
            "opened_event_sha256": opened["sha256"],
            "terminal_sequence": terminal["event"]["sequence"],
            "terminal_event_sha256": terminal["sha256"],
            "terminal_outcome": terminal_outcome,
            "receipt_core_sha256": receipt_core_sha256,
        }
        case[f"{mode}_journal_snapshot"] = raw
    return case


def _replace_snapshot(case, mode, raw):
    digest = hashlib.sha256(raw).hexdigest()
    binding = case[f"{mode}_run"]["run_journal"]
    binding["snapshot_sha256"] = digest
    binding["snapshot"] = (
        f"arc/crack_lab/run_journals/snapshots/{digest}.jsonl"
    )
    case[f"{mode}_journal_snapshot"] = raw


def _case():
    online = _run(
        "online",
        ONLINE_ID,
        score=99.0,
        started="2026-08-03T00:00:00Z",
        closed="2026-08-03T01:00:00Z",
    )
    competition = _run(
        "competition",
        COMPETITION_ID,
        score=99.5,
        started="2026-08-03T02:00:00Z",
        closed="2026-08-03T03:00:00Z",
    )
    case = {
        "baseline": _baseline(),
        "candidate": _candidate(competition),
        "candidate_readme": _readme(competition),
        "release": _release(),
        "release_sha256": RELEASE_SHA,
        "inventory": INVENTORY,
        "online_run": online,
        "competition_run": competition,
        "online_public": _public(
            online, published="2026-08-03T01:01:00Z"
        ),
        "competition_public": _public(
            competition, published="2026-08-03T03:01:00Z"
        ),
    }
    return _attach_journals(case)


def test_complete_v3_payload_passes_with_actual_nested_scorecard_shape():
    summary = G.validate_v3_payload(**_case())
    assert summary["status"] == "PASS"
    assert summary["raw_levels"] == 183
    assert summary["remote_close_recoveries"] == []
    assert summary["complete_games"] == 25
    assert summary["stored_actions"] == STORED_ACTIONS
    assert summary["official_actions"] == 183
    assert summary["total_resets"] == 25
    assert summary["arc_agi_toolkit_version"] == "0.9.9"
    assert summary["expected_pr_title"] == (
        "Add GKM — 99.5000% / 100.0000% raw: general-purpose "
        "replay-gated self-improving program synthesis"
    )


@pytest.mark.parametrize("outcome", ("reachable", "missing", "mismatch"))
def test_gate_public_revision_reachability_is_exact(monkeypatch, outcome):
    revision = "a" * 40

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, limit):
            resolved = revision if outcome != "mismatch" else "b" * 40
            return json.dumps(
                {
                    "sha": resolved,
                    "html_url": (
                        "https://github.com/sashakolpakov/gkm/commit/" + resolved
                    ),
                }
            ).encode()

    def open_commit(request, timeout):
        if outcome == "missing":
            raise G.urllib.error.URLError("404")
        return Response()

    monkeypatch.setattr(G.urllib.request, "urlopen", open_commit)
    if outcome == "reachable":
        assert REAL_VERIFY_PUBLIC_REVISION(revision)["sha"] == revision
    else:
        with pytest.raises(G.LeaderboardV3Error, match="reachable|different"):
            REAL_VERIFY_PUBLIC_REVISION(revision)


def test_pr_title_rounds_like_the_frozen_v2_title():
    assert G.expected_pr_title(98.11664037825032).startswith(
        "Add GKM — 98.1166% / 100.0000% raw:"
    )


def test_v3_rejects_stale_v1_v2_mutation():
    case = _case()
    case["candidate"]["versions"][1]["changes"] = "silently rewritten"
    with pytest.raises(G.LeaderboardV3Error, match="v1/v2 entries were mutated"):
        G.validate_v3_payload(**case)


def test_v3_rejects_forbidden_numeric_score():
    case = _case()
    case["candidate"]["versions"][2]["scores"][0]["score"] = 99.5
    with pytest.raises(G.LeaderboardV3Error, match="forbids a numeric score"):
        G.validate_v3_payload(**case)


def test_v3_rejects_mismatched_competition_receipt_url():
    case = _case()
    case["candidate"]["versions"][2]["scores"][0]["scorecard_url"] = (
        G.SCORECARD_URL_PREFIX + ONLINE_ID
    )
    with pytest.raises(G.LeaderboardV3Error, match="Competition run receipt"):
        G.validate_v3_payload(**case)


def test_v3_rejects_release_subdirectory_as_code_url():
    case = _case()
    case["candidate"]["code_url"] += (
        "/arc/crack_lab/releases/arc_agi3_gkm_v3_183"
    )
    with pytest.raises(G.LeaderboardV3Error, match="repository-root URL"):
        G.validate_v3_payload(**case)


@pytest.mark.parametrize("suffix", ("/README.md", "/arbitrary/path"))
def test_v3_rejects_arbitrary_code_url_suffixes(suffix):
    case = _case()
    case["candidate"]["code_url"] += suffix
    with pytest.raises(G.LeaderboardV3Error, match="repository-root URL"):
        G.validate_v3_payload(**case)


@pytest.mark.parametrize("suffix", ("?view=1", "#release"))
def test_v3_rejects_code_url_query_or_fragment(suffix):
    case = _case()
    case["candidate"]["code_url"] += suffix
    with pytest.raises(G.LeaderboardV3Error, match="repository-root URL"):
        G.validate_v3_payload(**case)


def test_v3_rejects_code_url_revision_mismatch():
    case = _case()
    case["candidate"]["code_url"] = (
        "https://github.com/sashakolpakov/gkm/tree/" + "c" * 40
    )
    with pytest.raises(G.LeaderboardV3Error, match="repository-root URL"):
        G.validate_v3_payload(**case)


def _write_content_addressed_receipt(tmp_path, inventory):
    identity = {"source_revision": "a" * 40}
    evidence = {
        game: [{"action_count": index + 1} for index in range(depth)]
        for game, depth in inventory.items()
    }
    body = {
        "schema": 1,
        "release_identity": identity,
        "release_identity_sha256": G._json_sha256(identity),
        "control_contract": {
            "sha256": CONTROL_SHA,
            "files_sha256": {
                "arc/crack_lab/arc_agi3_release_gate.py": "b" * 64
            }
        },
        "verifier": {
            "sha256": VERIFIER_SHA,
            "files_sha256": {
                "arc/crack_lab/arc_agi3_release_gate.py": "b" * 64
            }
        },
        "inventory_metadata_sha256": {"game/meta.json": "c" * 64},
        "canonical_game_count": 25,
        "authoritative_level_count": sum(inventory.values()),
        "canonical_tree_sha256": TREE_SHA,
        "inventory": inventory,
        "inventory_sha256": G._json_sha256(inventory),
        "evidence": evidence,
        "evidence_sha256": G._json_sha256(evidence),
    }
    raw = (
        json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    path = tmp_path / f"{hashlib.sha256(raw).hexdigest()}.json"
    path.write_bytes(raw)
    return path


def _verified_summary(body, digest):
    return {
        "status": "PASS",
        "games": 25,
        "levels": 183,
        "inventory_sha256": body["inventory_sha256"],
        "canonical_tree_sha256": body["canonical_tree_sha256"],
        "evidence_sha256": body["evidence_sha256"],
        "verifier_sha256": body["verifier"]["sha256"],
        "control_contract_sha256": body["control_contract"]["sha256"],
        "receipt_sha256": digest,
        "verification_context_source_revision": body["release_identity"][
            "source_revision"
        ],
    }


def test_v3_accepts_content_addressed_complete_release_shape(
    monkeypatch, tmp_path
):
    receipt = _write_content_addressed_receipt(tmp_path, INVENTORY)
    expected_body = json.loads(receipt.read_bytes())
    monkeypatch.setattr(
        G,
        "verify_frozen_release",
        lambda **kwargs: _verified_summary(expected_body, receipt.stem),
    )
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    body, digest, inventory = G.validate_complete_release_receipt(
        receipt, canonical_root=canonical
    )
    assert body["authoritative_level_count"] == 183
    assert digest == receipt.stem
    assert inventory == INVENTORY


def test_v3_rejects_incomplete_183_frontier(tmp_path):
    incomplete = dict(INVENTORY)
    incomplete["lf52"] = 9
    receipt = _write_content_addressed_receipt(tmp_path, incomplete)
    with pytest.raises(G.LeaderboardV3Error, match="not a complete 183-level"):
        G.validate_complete_release_receipt(
            receipt, canonical_root=tmp_path
        )


def test_v3_rejects_self_consistent_forged_receipt(monkeypatch, tmp_path):
    receipt = _write_content_addressed_receipt(tmp_path, INVENTORY)

    def reject(**kwargs):
        raise G.FrozenReleaseError("historical release gate failed")

    monkeypatch.setattr(G, "verify_frozen_release", reject)
    with pytest.raises(G.LeaderboardV3Error, match="independent historical"):
        G.validate_complete_release_receipt(
            receipt, canonical_root=tmp_path
        )


def test_v3_rejects_mutated_canonical_tree(monkeypatch, tmp_path):
    receipt = _write_content_addressed_receipt(tmp_path, INVENTORY)
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    (canonical / "mutated").write_text("changed")

    def reject_mutation(**kwargs):
        assert kwargs["canonical_root"] == canonical.resolve()
        raise G.FrozenReleaseError("canonical tree differs")

    monkeypatch.setattr(G, "verify_frozen_release", reject_mutation)
    with pytest.raises(G.LeaderboardV3Error, match="canonical tree differs"):
        G.validate_complete_release_receipt(
            receipt, canonical_root=canonical
        )


def test_v3_rejects_online_competition_source_receipt_mismatch():
    case = _case()
    other_revision = "c" * 40
    case["competition_run"]["source_revision"] = other_revision
    case["competition_run"]["source_url"] = (
        f"https://github.com/sashakolpakov/gkm/tree/{other_revision}"
    )
    with pytest.raises(G.LeaderboardV3Error, match="source revision differs"):
        G.validate_v3_payload(**case)


def test_v3_rejects_wrong_model_author_metadata():
    case = _case()
    case["candidate"]["authors"][1]["name"] = "OpenAI"
    with pytest.raises(G.LeaderboardV3Error, match="OpenAI GPT-5.6 model"):
        G.validate_v3_payload(**case)


@pytest.mark.parametrize("extra_versions", ([], ["3.0", "3.1"]))
def test_v3_rejects_missing_or_duplicate_v3(extra_versions):
    case = _case()
    case["candidate"]["versions"] = copy.deepcopy(
        case["candidate"]["versions"][:2]
    )
    template = _candidate(case["competition_run"])["versions"][2]
    for version in extra_versions:
        row = copy.deepcopy(template)
        row["version"] = version
        case["candidate"]["versions"].append(row)
    with pytest.raises(G.LeaderboardV3Error, match="missing or duplicated"):
        G.validate_v3_payload(**case)


def test_v3_requires_online_to_close_before_competition():
    case = _case()
    case["competition_run"]["started_at_utc"] = "2026-08-03T00:30:00Z"
    _attach_journals(case)
    with pytest.raises(
        G.LeaderboardV3Error, match="before the ONLINE|not monotonic"
    ):
        G.validate_v3_payload(**case)


def test_v3_can_recover_only_an_ambiguous_close_from_public_evidence():
    case = _case()
    case["competition_run"]["status"] = "FAIL"
    case["competition_run"]["scorecard_close"] = {
        "status": "ambiguous",
        "error_type": "ReadTimeout",
    }
    case["competition_run"]["aggregate"] = None
    case["candidate_readme"] = case["candidate_readme"].replace(
        "Close recovery: none", "Close recovery: competition"
    )
    _attach_journals(case)
    summary = G.validate_v3_payload(**case)
    assert summary["remote_close_recoveries"] == ["competition"]

    case["competition_public"]["published_at"] = None
    with pytest.raises(G.LeaderboardV3Error, match="publication"):
        G.validate_v3_payload(**case)


def test_v3_rejects_unaudited_toolkit_version():
    case = _case()
    case["competition_run"]["arc_agi_toolkit_version"] = "0.9.10"
    with pytest.raises(G.LeaderboardV3Error, match="audited arc-agi 0.9.9"):
        G.validate_v3_payload(**case)


def test_v3_rejects_host_path_or_secret_fields_in_run_receipt():
    case = _case()
    case["online_run"]["release_verification"]["receipt"] = (
        "/Users/private/release.json"
    )
    with pytest.raises(G.LeaderboardV3Error, match="path-safe"):
        G.validate_v3_payload(**case)

    case = _case()
    case["online_run"]["api_key"] = "top-secret"
    with pytest.raises(G.LeaderboardV3Error, match="field schema"):
        G.validate_v3_payload(**case)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("tags", ["gkm-v3"]),
        ("opaque", {"gkm_operation_id": "wrong"}),
        ("ai_agent", False),
    ),
)
def test_v3_rejects_public_card_provenance_mismatch(field, value):
    case = _case()
    case["competition_public"][field] = value
    with pytest.raises(G.LeaderboardV3Error, match="provenance metadata"):
        G.validate_v3_payload(**case)


def test_v3_rejects_truncated_journal_snapshot():
    case = _case()
    truncated = case["competition_journal_snapshot"][:-1]
    _replace_snapshot(case, "competition", truncated)
    with pytest.raises(G.LeaderboardV3Error, match="truncated"):
        G.validate_v3_payload(**case)


def test_v3_rejects_reordered_journal_snapshot():
    case = _case()
    lines = case["competition_journal_snapshot"].splitlines(keepends=True)
    lines[3], lines[4] = lines[4], lines[3]
    reordered = b"".join(lines)
    _replace_snapshot(case, "competition", reordered)
    with pytest.raises(G.LeaderboardV3Error, match="sequence|hash chain"):
        G.validate_v3_payload(**case)


def test_v3_rejects_wrong_canonical_journal_path():
    case = _case()
    case["competition_run"]["run_journal"]["live_journal"] = (
        "elsewhere/clean.jsonl"
    )
    with pytest.raises(G.LeaderboardV3Error, match="snapshot hash mismatch"):
        G.validate_v3_payload(**case)


def test_v3_rejects_hidden_unresolved_intent():
    case = _case()
    records = G._parse_journal_snapshot(case["competition_journal_snapshot"])
    duplicate_opaque = G.expected_scorecard_opaque(
        operation_id="5" * 64,
        mode="competition",
        receipt_sha256=RELEASE_SHA,
        canonical_tree_sha256=TREE_SHA,
        revision=REVISION,
    )
    _append_journal_event(
        records,
        kind="INTENT",
        operation_id="5" * 64,
        timestamp="2026-08-03T04:00:00Z",
        payload={
            **records[3]["event"]["payload"],
            "opaque": duplicate_opaque,
        },
    )
    raw = _snapshot_bytes(records)
    _replace_snapshot(case, "competition", raw)
    with pytest.raises(G.LeaderboardV3Error, match="unresolved open intent"):
        G.validate_v3_payload(**case)


def test_v3_rejects_duplicate_competition_open_chain():
    case = _case()
    records = G._parse_journal_snapshot(case["competition_journal_snapshot"])
    operation_id = "5" * 64
    card_id = "55555555-5555-4555-8555-555555555555"
    opaque = G.expected_scorecard_opaque(
        operation_id=operation_id,
        mode="competition",
        receipt_sha256=RELEASE_SHA,
        canonical_tree_sha256=TREE_SHA,
        revision=REVISION,
    )
    _append_journal_event(
        records,
        kind="INTENT",
        operation_id=operation_id,
        timestamp="2026-08-03T04:00:00Z",
        payload={**records[3]["event"]["payload"], "opaque": opaque},
    )
    _append_journal_event(
        records,
        kind="OPENED",
        operation_id=operation_id,
        timestamp="2026-08-03T04:00:01Z",
        payload={
            "card_id": card_id,
            "scorecard_url": G.SCORECARD_URL_PREFIX + card_id,
        },
    )
    _append_journal_event(
        records,
        kind="TERMINAL",
        operation_id=operation_id,
        timestamp="2026-08-03T04:00:02Z",
        payload={
            "outcome": "CLOSED_CONFIRMED_FAIL",
            "card_id": card_id,
            "receipt_core_sha256": "6" * 64,
        },
    )
    raw = _snapshot_bytes(records)
    _replace_snapshot(case, "competition", raw)
    with pytest.raises(G.LeaderboardV3Error, match="duplicate open attempt"):
        G.validate_v3_payload(**case)


@pytest.mark.parametrize(
    "extra_url",
    (
        G.SCORECARD_URL_PREFIX + ONLINE_ID,
        G.SCORECARD_URL_PREFIX + "55555555-5555-4555-8555-555555555555",
    ),
)
def test_v3_readme_rejects_stale_or_arbitrary_scorecard_url(extra_url):
    case = _case()
    case["candidate_readme"] += f"\nOther: {extra_url}\n"
    with pytest.raises(G.LeaderboardV3Error, match="exactly the definitive"):
        G.validate_v3_payload(**case)


def test_v3_readme_rejects_conflicting_official_score():
    case = _case()
    case["candidate_readme"] += "\nOfficial score 12.0.\n"
    with pytest.raises(G.LeaderboardV3Error, match="conflicting official score"):
        G.validate_v3_payload(**case)


def test_main_hashes_the_exact_candidate_bytes_it_validated(
    monkeypatch, tmp_path, capsys
):
    case = _case()
    baseline_path = tmp_path / "baseline.yaml"
    candidate_path = tmp_path / "candidate.yaml"
    readme_path = tmp_path / "README.md"
    candidate_raw = G.yaml.safe_dump(
        case["candidate"], sort_keys=False, allow_unicode=True
    ).encode()
    swapped_raw = b"name: swapped-after-validation\n"
    readme_path.write_text(case["candidate_readme"])
    online_raw = case["online_journal_snapshot"]
    competition_raw = case["competition_journal_snapshot"]
    online_snapshot = tmp_path / f"{hashlib.sha256(online_raw).hexdigest()}.jsonl"
    competition_snapshot = (
        tmp_path / f"{hashlib.sha256(competition_raw).hexdigest()}.jsonl"
    )
    online_snapshot.write_bytes(online_raw)
    competition_snapshot.write_bytes(competition_raw)
    original_read = G._read_regular
    candidate_reads = 0

    def swapping_read(path, *, label):
        nonlocal candidate_reads
        if G.Path(path) == candidate_path:
            candidate_reads += 1
            return candidate_raw if candidate_reads == 1 else swapped_raw
        return original_read(path, label=label)

    original_load_yaml = G._load_yaml

    def load_yaml(path, **kwargs):
        if G.Path(path) == baseline_path:
            return _baseline(), b"pinned-baseline"
        return original_load_yaml(path, **kwargs)

    monkeypatch.setattr(G, "_read_regular", swapping_read)
    monkeypatch.setattr(G, "_load_yaml", load_yaml)
    monkeypatch.setattr(
        G,
        "validate_complete_release_receipt",
        lambda *args, **kwargs: (_release(), RELEASE_SHA, INVENTORY),
    )
    monkeypatch.setattr(
        G,
        "_load_json",
        lambda path, **kwargs: (
            case["online_run"]
            if "online" in G.Path(path).name
            else case["competition_run"]
        ),
    )
    monkeypatch.setattr(
        G,
        "fetch_public_scorecard",
        lambda card_id: (
            case["online_public"]
            if card_id == ONLINE_ID
            else case["competition_public"]
        ),
    )
    monkeypatch.setattr(
        G,
        "validate_v3_payload",
        lambda **kwargs: {"status": "PASS"},
    )
    result = G.main(
        [
            "--baseline-yaml",
            str(baseline_path),
            "--candidate-yaml",
            str(candidate_path),
            "--candidate-readme",
            str(readme_path),
            "--release-receipt",
            str(tmp_path / "release.json"),
            "--canonical-release-root",
            str(tmp_path),
            "--online-run-receipt",
            str(tmp_path / "online.json"),
            "--competition-run-receipt",
            str(tmp_path / "competition.json"),
            "--online-journal-snapshot",
            str(online_snapshot),
            "--competition-journal-snapshot",
            str(competition_snapshot),
        ]
    )
    assert result == 0
    assert candidate_reads == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["candidate_yaml_sha256"] == hashlib.sha256(
        candidate_raw
    ).hexdigest()


@pytest.mark.parametrize(
    "raw",
    (
        b'{"schema":2,"schema":1}\n',
        b'{"value":NaN}\n',
        b'{"value":Infinity}\n',
    ),
)
def test_gate_json_loader_rejects_duplicates_and_nonfinite(tmp_path, raw):
    path = tmp_path / "record.json"
    path.write_bytes(raw)
    with pytest.raises(G.LeaderboardV3Error, match="duplicate|non-finite"):
        G._load_json(path, label="test record")


@pytest.mark.parametrize(
    "raw",
    (
        b"name: first\nname: second\n",
        b"name: .nan\n",
        b"name: .inf\n",
    ),
)
def test_gate_yaml_loader_rejects_duplicates_and_nonfinite(tmp_path, raw):
    path = tmp_path / "record.yaml"
    path.write_bytes(raw)
    with pytest.raises(
        G.LeaderboardV3Error, match="invalid YAML|non-finite"
    ):
        G._load_yaml(path, label="test YAML")


def test_v3_rejects_checkpoint_map_not_bound_to_release_evidence():
    case = _case()
    case["online_run"]["checkpoint_sha256"]["wa30"] = "9" * 64
    with pytest.raises(G.LeaderboardV3Error, match="release evidence"):
        G.validate_v3_payload(**case)


def test_v3_rejects_self_consistent_private_release_binding_field():
    case = _case()
    case["online_run"]["release_binding"]["private_path"] = "/Users/secret"
    with pytest.raises(G.LeaderboardV3Error, match="binding field schema"):
        G.validate_v3_payload(**case)


def test_v3_rejects_self_consistent_secret_result_field():
    case = _case()
    case["online_run"]["results"]["wa30"]["api_key"] = "secret"
    with pytest.raises(G.LeaderboardV3Error, match="endpoint differs"):
        G.validate_v3_payload(**case)


def test_v3_rejects_invalid_stored_action_type():
    case = _case()
    case["online_run"]["stored_actions"] = "not-an-int"
    with pytest.raises(G.LeaderboardV3Error, match="stored action"):
        G.validate_v3_payload(**case)


def test_v3_rejects_self_consistent_secret_aggregate_field():
    case = _case()
    case["online_run"]["aggregate"]["api_key"] = "secret"
    with pytest.raises(G.LeaderboardV3Error, match="aggregate schema"):
        G.validate_v3_payload(**case)


def test_v3_rejects_additional_failed_public_run():
    case = _case()
    failed = copy.deepcopy(
        case["competition_public"]["environments"][0]["runs"][0]
    )
    failed["guid"] = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    failed["state"] = "NOT_FINISHED"
    failed["completed"] = False
    case["competition_public"]["environments"][0]["runs"].append(failed)
    with pytest.raises(G.LeaderboardV3Error, match="terminal target WIN"):
        G.validate_v3_payload(**case)


def test_v3_rejects_source_url_suffix_even_at_correct_revision():
    case = _case()
    case["competition_run"]["source_url"] += "/arbitrary"
    with pytest.raises(G.LeaderboardV3Error, match="source URL is not immutable"):
        G.validate_v3_payload(**case)


def test_v3_rejects_wrong_integer_stored_actions_from_release_evidence():
    case = _case()
    case["online_run"]["stored_actions"] = STORED_ACTIONS - 1
    _attach_journals(case)
    with pytest.raises(G.LeaderboardV3Error, match="release evidence"):
        G.validate_v3_payload(**case)


@pytest.mark.parametrize(
    ("close", "message"),
    (
        (
            {"status": "confirmed", "error_type": None, "private_path": "/tmp/x"},
            "close outcome",
        ),
        (
            {"status": "ambiguous", "error_type": "ArbitrarySecretError"},
            "did not pass",
        ),
    ),
)
def test_v3_rejects_noncanonical_close_outcomes(close, message):
    case = _case()
    case["competition_run"]["scorecard_close"] = close
    if close["status"] == "ambiguous":
        case["competition_run"]["status"] = "FAIL"
        case["competition_run"]["aggregate"] = None
    _attach_journals(case)
    with pytest.raises(G.LeaderboardV3Error, match=message):
        G.validate_v3_payload(**case)


@pytest.mark.parametrize(
    "mutation",
    ("top", "v3", "model", "score"),
)
def test_v3_rejects_extra_candidate_schema_fields(mutation):
    case = _case()
    if mutation == "top":
        case["candidate"]["api_key"] = "secret"
    elif mutation == "v3":
        case["candidate"]["versions"][2]["private_path"] = "/Users/secret"
    elif mutation == "model":
        case["candidate"]["versions"][2]["models"][0]["api_key"] = "secret"
    else:
        case["candidate"]["versions"][2]["scores"][0]["private_path"] = (
            "/Users/secret"
        )
    with pytest.raises(
        G.LeaderboardV3Error,
        match="schema|metadata|target|host path",
    ):
        G.validate_v3_payload(**case)


@pytest.mark.parametrize(
    "contradiction",
    (
        "Toolkit 9.9.9.",
        "Actions 0.",
        "Resets 0.",
    ),
)
def test_v3_readme_rejects_contradictory_accounting(contradiction):
    case = _case()
    case["candidate_readme"] += f"\n{contradiction}\n"
    with pytest.raises(G.LeaderboardV3Error, match="conflicting|ambiguous"):
        G.validate_v3_payload(**case)


@pytest.mark.parametrize("field", ("api_key", "private_path"))
def test_v3_rejects_extra_public_top_level_secret_fields(field):
    case = _case()
    case["competition_public"][field] = "/Users/secret"
    with pytest.raises(G.LeaderboardV3Error, match="top-level schema"):
        G.validate_v3_payload(**case)


@pytest.mark.parametrize("field", ("actions", "resets"))
def test_v3_cross_binds_closed_and_public_game_accounting(field):
    case = _case()
    environment = case["competition_run"]["aggregate"]["environments"][0]
    environment[field] += 100
    environment["runs"][0][field] += 100
    if field == "actions":
        environment["runs"][0]["level_actions"][0] += 100
        case["competition_run"]["aggregate"]["total_actions"] += 100
    _attach_journals(case)
    with pytest.raises(G.LeaderboardV3Error, match="public game accounting"):
        G.validate_v3_payload(**case)


def test_v3_rejects_publication_outside_bound_close_interval():
    case = _case()
    forged = "2026-08-03T02:29:00Z"
    case["competition_public"]["published_at"] = forged
    case["competition_public"]["last_update"] = _timestamp_offset(
        forged, seconds=-1
    )
    with pytest.raises(G.LeaderboardV3Error, match="bound close interval"):
        G.validate_v3_payload(**case)


def test_v3_rejects_windows_host_paths_in_logical_receipt_fields():
    case = _case()
    windows_path = r"C:\Users\secret\artifacts"
    case["online_run"]["artifact_root"] = windows_path
    case["online_run"]["command"]["artifact_root"] = windows_path
    _attach_journals(case)
    with pytest.raises(G.LeaderboardV3Error, match="host-specific"):
        G.validate_v3_payload(**case)


def test_v3_rejects_secret_message_in_closed_tag_scores():
    case = _case()
    case["online_run"]["aggregate"]["tags_scores"][0]["message"] = (
        "/Users/secret/ARC_API_KEY=top-secret"
    )
    _attach_journals(case)
    with pytest.raises(G.LeaderboardV3Error, match="tag score schema"):
        G.validate_v3_payload(**case)


def test_v3_rejects_ambiguous_online_close_even_with_public_evidence():
    case = _case()
    case["online_run"]["status"] = "FAIL"
    case["online_run"]["scorecard_close"] = {
        "status": "ambiguous",
        "error_type": "ReadTimeout",
    }
    case["online_run"]["aggregate"] = None
    _attach_journals(case)
    with pytest.raises(G.LeaderboardV3Error, match="ONLINE PASS"):
        G.validate_v3_payload(**case)


@pytest.mark.parametrize(
    ("target", "field"),
    (
        ("run", "schema"),
        ("journal", "schema"),
        ("journal", "intent_sequence"),
    ),
)
def test_v3_rejects_boolean_schema_and_sequence_values(target, field):
    case = _case()
    if target == "run":
        case["online_run"][field] = True
        _attach_journals(case)
    else:
        case["online_run"]["run_journal"][field] = True
    with pytest.raises(G.LeaderboardV3Error, match="schema|sequence"):
        G.validate_v3_payload(**case)


def test_journal_parser_rejects_boolean_opaque_schema():
    case = _case()
    records = G._parse_journal_snapshot(case["online_journal_snapshot"])
    events = [copy.deepcopy(record["event"]) for record in records]
    events[0]["payload"]["opaque"]["schema"] = True
    previous = None
    raw_lines = []
    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence
        event["previous_event_sha256"] = previous
        line = G._canonical_json(event)
        previous = hashlib.sha256(line).hexdigest()
        raw_lines.append(line + b"\n")
    with pytest.raises(G.LeaderboardV3Error, match="opaque schema"):
        G._parse_journal_snapshot(b"".join(raw_lines))


def test_competition_recovery_history_projects_all_runs_exactly():
    case = _case()
    closed = case["competition_run"]["aggregate"]["environments"][0]
    public = case["competition_public"]["environments"][0]
    for environment, is_public in ((closed, False), (public, True)):
        failed = copy.deepcopy(environment["runs"][0])
        failed.update(
            {
                "guid": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                "score": 0.0,
                "levels_completed": 0,
                "actions": 1,
                "resets": 1,
                "state": "GAME_OVER",
                "completed": False,
                "level_scores": [0.0] * environment["level_count"],
                "level_actions": [1]
                + [0] * (environment["level_count"] - 1),
            }
        )
        if not is_public:
            failed["message"] = None
        environment["runs"].insert(0, failed)
        environment["actions"] += 1
        environment["resets"] += 1
    case["competition_run"]["aggregate"]["total_actions"] += 1
    case["competition_public"]["total_actions"] += 1
    case["candidate_readme"] = case["candidate_readme"].replace(
        "Official Competition actions: 183",
        "Official Competition actions: 184",
    ).replace(
        "Official Competition resets: 25",
        "Official Competition resets: 26",
    )
    _attach_journals(case)
    summary = G.validate_v3_payload(**case)
    assert summary["official_actions"] == 184
    assert summary["total_resets"] == 26


def test_v3_rejects_provider_id_with_embedded_host_path():
    case = _case()
    for container in (
        case["competition_run"]["aggregate"],
        case["competition_public"],
    ):
        environment = container["environments"][0]
        environment["id"] = "ar25-/Users/private"
        for run in environment["runs"]:
            run["id"] = environment["id"]
    _attach_journals(case)
    with pytest.raises(G.LeaderboardV3Error, match="environment ID|ambiguous"):
        G.validate_v3_payload(**case)


@pytest.mark.parametrize(
    ("target", "payload"),
    (
        ("changes", "Complete 183/183 from /Users/private/API_KEY-secret"),
        ("readme", "\nPrivate source: /Users/alice/.env ARC_API_KEY=secret\n"),
        ("changes", "Complete 183/183 secret=top-secret"),
        ("readme", "\nNUL follows: \x00\n"),
    ),
)
def test_v3_rejects_leaks_in_free_form_public_artifacts(target, payload):
    case = _case()
    if target == "changes":
        case["candidate"]["versions"][2]["changes"] = payload
    else:
        case["candidate_readme"] += payload
    with pytest.raises(
        G.LeaderboardV3Error, match="host path|NUL|secret assignment"
    ):
        G.validate_v3_payload(**case)


def test_v3_date_is_exact_competition_publication_utc_date():
    case = _case()
    case["candidate"]["versions"][2]["date"] = dt.date(2099, 1, 1)
    with pytest.raises(G.LeaderboardV3Error, match="publication date"):
        G.validate_v3_payload(**case)


@pytest.mark.parametrize(
    "date_value",
    (
        dt.datetime(2026, 8, 3, 0, 0, tzinfo=dt.timezone.utc),
        "20260803",
    ),
)
def test_v3_date_requires_an_exact_yaml_date_scalar(date_value):
    case = _case()
    case["candidate"]["versions"][2]["date"] = date_value
    with pytest.raises(G.LeaderboardV3Error, match="YYYY-MM-DD"):
        G.validate_v3_payload(**case)

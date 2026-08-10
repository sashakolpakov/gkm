from __future__ import annotations

import json

import codex_campaign_policy as P


def _binding(game: str, reached: int):
    checkpoint = P.Status.ZERO_SHA256 if reached == 0 else "a" * 64
    source = P.Status.ZERO_SHA256 if reached == 0 else "b" * 64
    return P.Status.validate_frontier_binding({
        "frontier_binding_schema": P.Status.FRONTIER_BINDING_SCHEMA,
        "game": game,
        "reached": reached,
        "target_level": reached + 1,
        "parent_action_count": reached,
        "parent_checkpoint_sha256": checkpoint,
        "parent_source_tree_sha256": source,
        "frontier_sha256": P.Status._sha256_json({
            "game": game,
            "reached": reached,
            "parent_checkpoint_sha256": checkpoint,
        }),
    })


def test_required_headroom_uses_worst_observed_rate_plus_margin():
    turns = [{
        "reasoning_effort": "high",
        "displayed_weekly_points_used": 8,
        "duration_seconds": 720,
    }]
    assert P.required_headroom("high", 12, turns) == 9
    assert P.required_headroom("high", 6, turns) == 6
    assert P.required_headroom("medium", 6, []) == 4
    assert P.required_headroom("max", 60, []) == 1


def test_required_headroom_excludes_operator_interruption_rate_outlier():
    turns = [
        {
            "reasoning_effort": "high",
            "displayed_weekly_points_used": 2,
            "duration_seconds": 53,
            "interrupted": True,
        },
        {
            "reasoning_effort": "high",
            "displayed_weekly_points_used": 2,
            "duration_seconds": 480,
            "interrupted": False,
        },
    ]
    assert P.required_headroom("high", 8, turns) == 6


def test_policy_holds_tail_and_builds_one_adaptive_seed():
    report = {
        "allowance": {"remaining_percent": 32},
        "local_window": {"runs": 15},
        "turns": [],
        "frontiers": [
            {
                **_binding(f"g{i}", 1),
                "game": f"g{i}", "next_level": 2,
                "current_level": 1, "authoritative_level_count": 8,
                "incumbent_kind": "promoted",
                "retry_complexity_n": 0,
                "recommended_effort": "medium", "recommended_minutes": 15,
                "recommended_wip_mode": "exclude",
                "recommended_auxiliary_parallelism": 0,
                "dispatch_mode": "fresh_frontier",
                "warm_wip_available": False, "external_evidence": {},
            }
            for i in range(4)
        ],
    }
    result = P.policy_report(report)
    assert result["admit_next_turn"] is False
    assert result["phase"] == "hold_for_weekly_reset"
    assert len(result["initial_queue"]) == 1
    assert result["initial_queue"][0]["effort"] == "medium"
    assert "--debrief-policy=never" in result["initial_queue"][0]["command"]
    assert "--transient-retries=0" in result["initial_queue"][0]["command"]


def test_policy_admits_seed_in_fresh_window():
    report = {
        "allowance": {"remaining_percent": 100},
        "local_window": {"runs": 0},
        "turns": [],
        "effort_efficiency": {},
        "frontiers": [{
            **_binding("cold", 0),
            "game": "cold", "next_level": 1,
            "current_level": 0, "authoritative_level_count": 8,
            "incumbent_kind": "cold_start", "retry_complexity_n": 0,
            "recommended_effort": "medium", "recommended_minutes": 15,
            "recommended_wip_mode": "exclude",
            "recommended_auxiliary_parallelism": 0,
            "dispatch_mode": "fresh_frontier",
            "warm_wip_available": False, "external_evidence": {},
        }],
    }
    result = P.policy_report(report)
    assert result["admit_next_turn"] is True
    assert result["phase"] == "run_initial_item_then_adapt"
    item = result["initial_queue"][0]
    assert item["seed_mode"] == "zero_seed"
    assert item["wip_mode"] == "exclude"
    assert "--seed-mode=zero_seed" in item["argv"]
    assert "--wip-mode=exclude" in item["argv"]


def test_policy_carries_authoritative_progress_into_machine_queue():
    progress = {
        "solved_levels": 174,
        "total_levels": 183,
        "remaining_levels": 9,
        "percent": 95.082,
    }
    report = {
        "allowance": {"remaining_percent": 100, "window_name": "unlimited"},
        "local_window": {"runs": 0},
        "turns": [],
        "canonical_progress": progress,
        "frontiers": [],
    }
    assert P.policy_report(report)["canonical_progress"] == progress


def test_high_rescue_summary_counts_only_high_after_medium_failure():
    turns = [
        {
            "game": "cold", "target_level": 1, "reasoning_effort": "high",
            "solved_target": True, "displayed_weekly_points_used": 2,
        },
        {
            "game": "a", "target_level": 3, "reasoning_effort": "medium",
            "solved_target": False, "displayed_weekly_points_used": 2,
        },
        {
            "game": "a", "target_level": 3, "reasoning_effort": "high",
            "solved_target": True, "displayed_weekly_points_used": 3,
        },
        {
            "game": "b", "target_level": 2, "reasoning_effort": "medium",
            "solved_target": False, "displayed_weekly_points_used": 1,
        },
        {
            "game": "b", "target_level": 2, "reasoning_effort": "high",
            "solved_target": False, "displayed_weekly_points_used": 2,
        },
    ]
    result = P.high_rescue_summary(turns)
    assert result["qualifying_high_attempts"] == 2
    assert result["replay_validated_rescues"] == 1
    assert result["rescue_rate"] == 0.5
    assert result["displayed_weekly_points"] == 5
    assert result["displayed_points_per_rescue"] == 5.0


def test_choose_exploitation_effort_requires_two_attempts_per_arm():
    partial = {
        "medium": {"proposal_attempts": 2, "solved_levels": 1,
                   "displayed_weekly_points": 4},
        "high": {"proposal_attempts": 1, "solved_levels": 1,
                 "displayed_weekly_points": 2},
    }
    assert P.choose_exploitation_effort(partial) is None
    partial["high"]["proposal_attempts"] = 2
    partial["high"]["displayed_weekly_points"] = 7
    assert P.choose_exploitation_effort(partial) == "medium"


def test_choose_exploitation_effort_uses_ast_only_near_cost_tie():
    efficiency = {
        "medium": {"proposal_attempts": 2, "solved_levels": 2,
                   "displayed_weekly_points": 5.0},
        "high": {"proposal_attempts": 2, "solved_levels": 2,
                 "displayed_weekly_points": 5.4},
    }
    quality = {
        "medium": {"median_conditional_ast_zlib_bytes": 6000},
        "high": {"median_conditional_ast_zlib_bytes": 3000},
    }
    assert P.choose_exploitation_effort(efficiency, quality) == "high"
    efficiency["high"]["displayed_weekly_points"] = 7.0
    assert P.choose_exploitation_effort(efficiency, quality) == "medium"


def test_adaptive_campaign_item_uses_retry_policy_not_historical_cost_arm():
    report = {
        "turns": [],
        "effort_efficiency": {
            "medium": {"proposal_attempts": 2, "solved_levels": 2,
                       "displayed_weekly_points": 5},
            "high": {"proposal_attempts": 2, "solved_levels": 1,
                     "displayed_weekly_points": 4},
        },
        "frontiers": [{
            **_binding("cold", 0),
            "game": "cold", "next_level": 1,
            "current_level": 0, "authoritative_level_count": 8,
            "incumbent_kind": "cold_start", "retry_complexity_n": 0,
            "recommended_effort": "medium", "recommended_minutes": 15,
            "recommended_wip_mode": "exclude",
            "recommended_auxiliary_parallelism": 0,
            "dispatch_mode": "fresh_frontier",
            "warm_wip_available": False, "external_evidence": {},
        }],
    }
    item = P.adaptive_campaign_item(report, reserve=20)
    assert item["effort"] == "medium"
    assert "--codex-weekly-reserve=20" in item["argv"]
    assert item["experiment_role"] == "retry_n0_fresh_frontier"
    assert "failure_revision_rounds" not in item
    assert "failure_revision_protocol_sha256" not in item
    assert not any(
        arg.startswith("--failure-revision-rounds=")
        for arg in item["argv"]
    )


def test_failure_revision_treatment_overrides_only_generation_intensity():
    report = {
        "allowance": {"remaining_percent": 100},
        "local_window": {"runs": 0},
        "turns": [],
        "frontiers": [{
            **_binding("cold", 0),
            "game": "cold", "next_level": 1,
            "current_level": 0, "authoritative_level_count": 8,
            "incumbent_kind": "cold_start", "retry_complexity_n": 0,
            "recommended_effort": "medium", "recommended_minutes": 15,
            "recommended_wip_mode": "exclude",
            "recommended_auxiliary_parallelism": 0,
            "dispatch_mode": "fresh_frontier",
            "warm_wip_available": False, "external_evidence": {},
        }],
    }
    omitted = P.policy_report(report)
    explicit_default = P.policy_report(report, failure_revision_rounds=1)
    assert json.dumps(omitted, sort_keys=True) == json.dumps(
        explicit_default, sort_keys=True
    )

    plan = P.policy_report(report, failure_revision_rounds=4)
    item = plan["initial_queue"][0]

    assert plan["phase"] == "run_one_frozen_failure_revision_item"
    assert plan["failure_revision_rounds"] == 4
    assert item["retry_complexity_n"] == 0
    assert item["seed_mode"] == "zero_seed"
    assert item["wip_mode"] == "exclude"
    assert item["dispatch_mode"] == "fresh_frontier"
    assert item["effort"] == "max"
    assert item["minutes"] == 300
    assert item["failure_revision_rounds"] == 4
    assert item["failure_revision_protocol_sha256"] == (
        P.FAILURE_REVISION_PROTOCOL_SHA256
    )
    assert "--codex-effort=max" in item["argv"]
    assert "--minutes=300" in item["argv"]
    assert "--codex-allocation-policy=hard" in item["argv"]
    assert item["argv"].count("--failure-revision-rounds=4") == 1


def test_adaptive_campaign_item_ranks_by_retry_coordinate_not_paid_attempts():
    report = {
        "turns": [],
        "effort_efficiency": {},
        "frontiers": [
            {
                **_binding("bad", 1),
                "game": "bad", "next_level": 2,
                "current_level": 1, "authoritative_level_count": 8,
                "incumbent_kind": "promoted",
                "retry_complexity_n": 2,
                "paid_attempts_at_frontier": 0,
                "quarantined_after_escalation_failure": True,
                "recommended_effort": "xhigh", "recommended_minutes": 25,
                "recommended_wip_mode": "restore_clean_same_frontier",
                "recommended_auxiliary_parallelism": 0,
                "dispatch_mode": "continue_clean_wip",
                "warm_wip_available": True,
            },
            {
                **_binding("good", 1),
                "game": "good", "next_level": 2,
                "current_level": 1, "authoritative_level_count": 8,
                "incumbent_kind": "promoted",
                "retry_complexity_n": 0,
                "paid_attempts_at_frontier": 99,
                "quarantined_after_escalation_failure": False,
                "recommended_effort": "medium", "recommended_minutes": 15,
                "recommended_wip_mode": "exclude",
                "recommended_auxiliary_parallelism": 0,
                "dispatch_mode": "fresh_frontier",
                "warm_wip_available": False, "external_evidence": {},
            },
        ],
    }
    item = P.adaptive_campaign_item(report, reserve=20)
    assert item["game"] == "good"


def test_unlimited_campaign_retries_quarantined_frontier_at_xhigh():
    report = {
        "allowance": {"window_name": "unlimited"},
        "turns": [],
        "frontiers": [{
            **_binding("hard", 2),
            "game": "hard",
            "current_level": 2,
            "next_level": 3,
            "authoritative_level_count": 7,
            "incumbent_kind": "promoted",
            "paid_attempts_at_frontier": 99,
            "retry_complexity_n": 2,
            "quarantined_after_escalation_failure": True,
            "priority_score": 3.0,
            "recommended_effort": "xhigh",
            "recommended_minutes": 25,
            "recommended_wip_mode": "restore_clean_same_frontier",
            "recommended_auxiliary_parallelism": 0,
            "dispatch_mode": "continue_clean_wip",
            "warm_wip_available": True,
            "warm_wip_attempt": "not_reached_abc123",
            "external_evidence": {},
        }],
    }
    item = P.adaptive_campaign_item(report, reserve=5)
    assert item["game"] == "hard"
    assert item["effort"] == "xhigh"
    assert item["minutes"] == 25
    assert "--codex-effort=xhigh" in item["argv"]
    assert item["experiment_role"] == "retry_n2_continue_clean_wip"
    assert item["seed_mode"] == "verified_parent"
    assert item["wip_mode"] == "restore_clean_same_frontier"
    assert item["expected_wip_attempt"] == "not_reached_abc123"
    assert "--expected-wip-attempt=not_reached_abc123" in item["argv"]
    assert item["cost_control_enabled"] is False
    assert item["max_campaign_runs"] == -1
    assert item["max_campaign_tokens"] == -1
    assert "--codex-allocation-policy=drain" in item["argv"]


def test_long_turns_alternate_coherent_reset_and_wip_continuation():
    row = {
        "game": "hard",
        "current_level": 5,
        "next_level": 6,
        "incumbent_kind": "promoted",
        "warm_wip_available": True,
    }
    row["retry_complexity_n"] = 5
    assert P.lineage_input_modes(row, minutes=90) == (
        "verified_parent", "exclude"
    )
    row["retry_complexity_n"] = 6
    assert P.lineage_input_modes(row, minutes=120) == (
        "verified_parent", "restore_clean_same_frontier"
    )
    row["retry_complexity_n"] = 7
    assert P.lineage_input_modes(row, minutes=180) == (
        "verified_parent", "exclude"
    )
    row["retry_complexity_n"] = 8
    assert P.lineage_input_modes(row, minutes=180) == (
        "verified_parent", "restore_clean_same_frontier"
    )


def test_policy_rejected_legacy_wip_becomes_clean_reset_before_plan():
    report = {
        "allowance": {"window_name": "unlimited"},
        "turns": [],
        "frontiers": [{
            **_binding("hard", 2),
            "game": "hard",
            "current_level": 2,
            "next_level": 3,
            "authoritative_level_count": 7,
            "incumbent_kind": "promoted",
            "retry_complexity_n": 2,
            "priority_score": 3.0,
            "recommended_effort": "xhigh",
            "recommended_minutes": 25,
            "recommended_wip_mode": "restore_clean_same_frontier",
            "recommended_auxiliary_parallelism": 0,
            "dispatch_mode": "continue_clean_wip",
            "warm_wip_available": False,
            "warm_wip_attempt": None,
            "warm_wip_phase": None,
            "warm_wip_recovery_required": False,
            "warm_wip_validation": (
                "rejected:filesystem_boundary_policy_binding"
            ),
            "external_evidence": {},
        }],
    }

    item = P.adaptive_campaign_item(report, reserve=5)

    assert item["wip_mode"] == "exclude"
    assert item["dispatch_mode"] == "filesystem_boundary_clean_reset"
    assert item["expected_wip_attempt"] is None
    assert item["warm_wip_validation"] == (
        "rejected:filesystem_boundary_policy_binding"
    )
    assert item["experiment_role"] == (
        "retry_n2_filesystem_boundary_clean_reset"
    )
    assert "--wip-mode=exclude" in item["argv"]


def test_clean_infrastructure_wip_overrides_n7_reset_without_advancing_n():
    report = {
        "allowance": {"window_name": "unlimited"},
        "turns": [],
        "frontiers": [{
            **_binding("hard", 8),
            "game": "hard",
            "current_level": 8,
            "next_level": 9,
            "authoritative_level_count": 10,
            "incumbent_kind": "promoted",
            "retry_complexity_n": 7,
            "priority_score": 3.0,
            "recommended_effort": "max",
            "recommended_minutes": 180,
            "recommended_wip_mode": "exclude",
            "recommended_auxiliary_parallelism": 2,
            "dispatch_mode": "repeated_hard_frontier_reset",
            "warm_wip_available": True,
            "warm_wip_attempt": "infrastructure_failure_transport_abc123",
            "warm_wip_phase": "infrastructure_failure_transport",
            "warm_wip_recovery_required": True,
            "external_evidence": {},
        }],
    }
    item = P.adaptive_campaign_item(report, reserve=5)
    assert item["retry_complexity_n"] == 7
    assert item["effort"] == "max"
    assert item["minutes"] == 180
    assert item["policy_dispatch_mode"] == "repeated_hard_frontier_reset"
    assert item["dispatch_mode"] == "recover_clean_infrastructure_wip"
    assert item["wip_mode"] == "restore_clean_same_frontier"
    assert item["experiment_role"] == (
        "retry_n7_recover_clean_infrastructure_wip"
    )
    assert item["expected_wip_attempt"] == (
        "infrastructure_failure_transport_abc123"
    )
    assert "--wip-mode=restore_clean_same_frontier" in item["argv"]


def test_n7_solver_no_progress_still_excludes_wip():
    row = {
        "game": "hard",
        "current_level": 8,
        "next_level": 9,
        "incumbent_kind": "promoted",
        "retry_complexity_n": 7,
        "warm_wip_available": True,
        "warm_wip_attempt": "not_reached_abc123",
        "warm_wip_phase": "not_reached",
        "warm_wip_recovery_required": False,
    }
    assert P.lineage_input_modes(row, minutes=180) == (
        "verified_parent", "exclude"
    )


def test_infrastructure_recovery_without_exact_capsule_fails_closed():
    row = {
        "game": "hard",
        "current_level": 8,
        "next_level": 9,
        "incumbent_kind": "promoted",
        "retry_complexity_n": 7,
        "warm_wip_available": True,
        "warm_wip_recovery_required": True,
    }
    try:
        P.lineage_input_modes(row, minutes=180)
    except ValueError as exc:
        assert "exact-frontier WIP capsule" in str(exc)
    else:
        raise AssertionError("unbound infrastructure WIP was accepted")


def test_unlimited_escalation_lengthens_uninterrupted_hard_frontier_turns():
    assert P.unlimited_escalation(3, 40) == (
        "xhigh", 40, "retry_n3_warm_hard_frontier"
    )
    assert P.unlimited_escalation(4, 60) == (
        "max", 60, "retry_n4_first_max"
    )
    assert P.unlimited_escalation(5, 90) == (
        "max", 90, "retry_n5_max_coherence_reset"
    )
    assert P.unlimited_escalation(6, 120) == (
        "max", 120, "retry_n6_max_cumulative"
    )
    assert P.unlimited_escalation(7, 180) == (
        "max", 180, "retry_n7_repeated_hard_frontier_reset"
    )


def test_paid_attempt_count_cannot_substitute_for_retry_coordinate():
    report = {
        "allowance": {"window_name": "unlimited"},
        "turns": [],
        "frontiers": [{
            "game": "hard",
            "current_level": 1,
            "next_level": 2,
            "authoritative_level_count": 8,
            "incumbent_kind": "promoted",
            "paid_attempts_at_frontier": 12,
            "warm_wip_available": True,
        }],
    }
    try:
        P.adaptive_campaign_item(report, reserve=5)
    except ValueError as exc:
        assert "retry_complexity_n" in str(exc)
    else:
        raise AssertionError("paid attempts silently steered the policy")

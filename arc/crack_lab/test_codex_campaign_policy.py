from __future__ import annotations

import codex_campaign_policy as P


def test_required_headroom_uses_worst_observed_rate_plus_margin():
    turns = [{
        "reasoning_effort": "high",
        "displayed_weekly_points_used": 8,
        "duration_seconds": 720,
    }]
    assert P.required_headroom("high", 12, turns) == 9
    assert P.required_headroom("high", 6, turns) == 6
    assert P.required_headroom("medium", 6, []) == 4


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
                "game": f"g{i}", "next_level": 2,
                "incumbent_kind": "promoted", "paid_attempts_at_frontier": 0,
                "quarantined_after_escalation_failure": False,
                "recommended_effort": "medium", "recommended_minutes": 8,
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
            "game": "cold", "next_level": 1,
            "incumbent_kind": "cold_start", "paid_attempts_at_frontier": 0,
            "quarantined_after_escalation_failure": False,
            "recommended_effort": "medium", "recommended_minutes": 6,
            "warm_wip_available": False, "external_evidence": {},
        }],
    }
    result = P.policy_report(report)
    assert result["admit_next_turn"] is True
    assert result["phase"] == "run_initial_item_then_adapt"


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


def test_adaptive_campaign_item_uses_cheaper_arm_on_untouched_cold():
    report = {
        "turns": [],
        "effort_efficiency": {
            "medium": {"proposal_attempts": 2, "solved_levels": 2,
                       "displayed_weekly_points": 5},
            "high": {"proposal_attempts": 2, "solved_levels": 1,
                     "displayed_weekly_points": 4},
        },
        "frontiers": [{
            "game": "cold", "next_level": 1,
            "incumbent_kind": "cold_start", "paid_attempts_at_frontier": 0,
            "warm_wip_available": False, "external_evidence": {},
        }],
    }
    item = P.adaptive_campaign_item(report, reserve=20)
    assert item["effort"] == "medium"
    assert "--codex-weekly-reserve=20" in item["argv"]
    assert item["experiment_role"] == "cold_L1_exploitation_medium"


def test_adaptive_campaign_item_skips_quarantined_frontier():
    report = {
        "turns": [],
        "effort_efficiency": {},
        "frontiers": [
            {
                "game": "bad", "next_level": 2,
                "incumbent_kind": "promoted",
                "quarantined_after_escalation_failure": True,
            },
            {
                "game": "good", "next_level": 2,
                "incumbent_kind": "promoted",
                "quarantined_after_escalation_failure": False,
                "recommended_effort": "medium", "recommended_minutes": 8,
                "warm_wip_available": False, "external_evidence": {},
            },
        ],
    }
    item = P.adaptive_campaign_item(report, reserve=20)
    assert item["game"] == "good"

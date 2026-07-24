"""Offline regression tests for the adaptive solve-preserving wall-time policy.

No ledger, no Codex turn, no network: synthetic joined-turn dicts drive the rule so
the wall-time sizing and the escalation-cost headroom can be checked without spending
any allowance.  The load-bearing property is solve-preserving sizing: the recommended
wall time never truncates a historically replay-validated solve, and it falls back to
the conservative static cap when an arm has too few solves to size empirically.
"""
import math

import codex_campaign_status as S
import codex_campaign_policy as P


def _turn(effort, level, solved, minutes, *, label="game:propose", **extra):
    row = {
        "run_label": label,
        "reasoning_effort": effort,
        "target_level": level,
        "solved_target": solved,
        "duration_seconds": minutes * 60.0,
        "game": extra.pop("game", "gg"),
    }
    row.update(extra)
    return row


# --- recommend_minutes -----------------------------------------------------

def test_thin_data_falls_back_to_static():
    # Two solves is below MIN_SOLVES_TO_SIZE, so the arm keeps its static cap.
    turns = [_turn("medium", 2, True, 5.0), _turn("medium", 2, True, 6.0)]
    out = S.recommend_minutes("continuation_L2+", "medium", turns)
    assert out["minutes"] == S.STATIC_WALL_MINUTES[("continuation_L2+", "medium")]
    assert out["basis"] == "static_fallback"
    assert out["solve_samples"] == 2


def test_empty_turns_fall_back_to_static():
    out = S.recommend_minutes("continuation_L2+", "high", [])
    assert out["minutes"] == 8
    assert out["basis"] == "static_fallback"
    assert out["slowest_solve_minutes"] is None


def test_solve_preserving_raises_for_censored_high_continuation():
    # Reproduces the ledger finding: high continuation solves run to ~12 min while
    # the old static cap was 8.  The rule must size UP to cover the slow solves.
    solves = [4.2, 5.6, 6.1, 6.4, 9.0, 12.0, 12.0]
    turns = [_turn("high", 3, True, m) for m in solves]
    out = S.recommend_minutes("continuation_L2+", "high", turns)
    assert out["minutes"] >= math.ceil(max(solves))  # never truncate a real solve
    assert out["minutes"] > 8                          # strictly above the old cap
    assert out["basis"] == "empirical_solve_preserving"
    assert out["slowest_solve_minutes"] == 12.0


def test_safe_trim_for_loose_cold_medium():
    # Cold-medium solves finish well under the 6-min static cap, so a small trim is
    # safe -- but the result must still exceed the slowest solve.
    solves = [1.7, 2.9, 4.0]
    turns = [_turn("medium", 1, True, m) for m in solves]
    out = S.recommend_minutes("cold_L1", "medium", turns)
    assert out["minutes"] < S.STATIC_WALL_MINUTES[("cold_L1", "medium")]
    assert out["minutes"] >= math.ceil(max(solves))


def test_never_truncates_slowest_validated_solve():
    for effort, phase in (("high", "continuation_L2+"), ("medium", "cold_L1")):
        for slowest in (3.1, 7.9, 11.5):
            solves = [1.0, 2.0, slowest]
            turns = [_turn(effort, 2 if phase != "cold_L1" else 1, True, m)
                     for m in solves]
            out = S.recommend_minutes(phase, effort, turns)
            assert out["minutes"] >= math.ceil(slowest)


def test_ceiling_guarantee_still_preserves_solves():
    # An implausibly slow solve exceeds the ceiling; the solve-preserving guarantee
    # must win over the secondary ceiling cap.
    solves = [5.0, 10.0, 20.0]
    turns = [_turn("high", 4, True, m) for m in solves]
    out = S.recommend_minutes("continuation_L2+", "high", turns)
    assert out["minutes"] >= 20


def test_floor_applies_for_tiny_solves():
    solves = [0.5, 0.6, 0.7]
    turns = [_turn("high", 2, True, m) for m in solves]
    out = S.recommend_minutes("continuation_L2+", "high", turns)
    assert out["minutes"] == S.WALL_MINUTES_FLOOR["high"]


def test_failures_and_non_propose_turns_are_ignored():
    # Only replay-validated proposal solves size the arm.  A long failure or a long
    # debrief must not inflate the recommendation.
    turns = [
        _turn("high", 3, True, 6.0),
        _turn("high", 3, True, 6.2),
        _turn("high", 3, True, 6.1),
        _turn("high", 3, False, 12.0),                      # failure: ignored
        _turn("high", 3, True, 30.0, label="game:debrief"),  # not a proposal: ignored
    ]
    out = S.recommend_minutes("continuation_L2+", "high", turns)
    assert out["solve_samples"] == 3
    assert out["slowest_solve_minutes"] == 6.2
    assert out["minutes"] < 12


def test_phase_split_separates_cold_and_continuation():
    turns = [_turn("medium", 1, True, m) for m in (2.0, 2.5, 3.0)]  # cold only
    cold = S.recommend_minutes("cold_L1", "medium", turns)
    cont = S.recommend_minutes("continuation_L2+", "medium", turns)
    assert cold["basis"] == "empirical_solve_preserving"
    assert cont["basis"] == "static_fallback"  # no L2+ solves in the sample


def test_zero_and_missing_duration_solves_are_dropped():
    turns = [
        _turn("medium", 2, True, 5.0),
        {"run_label": "g:propose", "reasoning_effort": "medium",
         "target_level": 2, "solved_target": True, "duration_seconds": 0},
        {"run_label": "g:propose", "reasoning_effort": "medium",
         "target_level": 2, "solved_target": True, "duration_seconds": None},
    ]
    out = S.recommend_minutes("continuation_L2+", "medium", turns)
    assert out["solve_samples"] == 1


# --- ranked_frontiers wiring ----------------------------------------------

def _frontier(game, next_level, kind):
    return {
        "game": game,
        "next_level": next_level,
        "incumbent_kind": kind,
        "priority_score": 1.0,
    }


def test_ranked_frontier_carries_adaptive_minutes_and_provenance():
    turns = [_turn("high", 3, True, m, game="zz")
             for m in (6.0, 8.0, 10.0, 12.0)] + \
            [_turn("high", 3, False, 8.0, game="zz")]
    # A prior high failure on the frontier routes it to a bounded high escalation.
    frontier = _frontier("zz", 3, "promoted")
    ranked = S.ranked_frontiers([frontier], turns)
    row = ranked[0]
    assert row["recommended_effort"] == "high"
    assert row["recommended_minutes"] >= math.ceil(12.0)
    assert row["recommended_minutes_basis"] == "empirical_solve_preserving"
    assert row["recommended_minutes_solve_samples"] == 4
    assert row["slowest_validated_solve_minutes"] == 12.0


def test_quarantined_frontier_gets_zero_minutes():
    # Two high failures quarantine the frontier; no turn, hence zero minutes.
    turns = [_turn("high", 2, False, 8.0, game="qq"),
             _turn("high", 2, False, 8.0, game="qq")]
    ranked = S.ranked_frontiers([_frontier("qq", 2, "promoted")], turns)
    row = ranked[0]
    assert row["quarantined_after_escalation_failure"] is True
    assert row["recommended_minutes"] == 0
    assert row["recommended_minutes_basis"] == "quarantined"


# --- escalation-cost headroom (required_headroom) --------------------------

def test_headroom_floors_with_no_evidence():
    assert P.required_headroom("medium", 6, []) == 4
    assert P.required_headroom("high", 8, []) == 6


def test_headroom_ignores_interrupted_turns():
    # A 53-second operator interruption charging 2 rounded points must not
    # extrapolate to a huge full-turn requirement; it is excluded from the rate.
    interrupted = {"reasoning_effort": "high", "interrupted": True,
                   "displayed_weekly_points_used": 2, "duration_seconds": 53}
    assert P.required_headroom("high", 8, [interrupted]) == 6


def test_headroom_two_points_over_eight_minutes_is_floor():
    normal = {"reasoning_effort": "high", "displayed_weekly_points_used": 2,
              "duration_seconds": 8 * 60}
    # rate 0.25 pt/min * 8 min + 1 = 3, below the six-point high floor.
    assert P.required_headroom("high", 8, [normal]) == 6

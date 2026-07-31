"""Reward verification for the compact level-9 candidate."""

import gkm_try

from probe9_verify import (
    ReachedLevel9,
    StopAtLevel9,
    boxes,
    target_state,
)


def common_prefix():
    remote_pick = [2] + [4] * 6 + [1, 5]
    first_delivery = remote_pick + [3] * 6 + [1] * 3 + [3, 5]
    second_pick = first_delivery + [2] * 3 + [4] * 5 + [1] * 2 + [5]
    place_bottom_middle = [2] + [3] * 6 + [1] * 3 + [5]
    return second_pick + place_bottom_middle


def direct_second_prefix():
    remote_pick = [2] + [4] * 6 + [1, 5]
    first_delivery = remote_pick + [3] * 6 + [1] * 3 + [3, 5]
    second_pick = first_delivery + [2] * 2 + [4] * 5 + [1, 5]
    place_bottom_middle = [2] + [3] * 6 + [1] * 3 + [5]
    return second_pick + place_bottom_middle


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel9(env))
    except ReachedLevel9:
        pass

    prefix = common_prefix()
    candidates = {
        "upper_contact": (
            [3, 5, 5, 5]
            + [2] + [3] * 4 + [5, 4]
            + [1] * 5 + [4] * 5 + [2] * 2 + [5, 1]
        ),
        "lower_contact": (
            [2, 4, 3, 5]
            + [3] * 5 + [5, 4]
            + [1] * 5 + [4] * 5 + [2] * 2 + [5, 1]
        ),
        "direct_second": (
            [2, 2, 3, 5]
            + [3] * 4 + [5, 4]
            + [1] * 6 + [4] * 5 + [2] * 2 + [5, 1]
            + [5] * 5
        ),
        "direct_idle": [2, 2, 3, 5] + [5] * 26,
        "direct_top_timed": (
            [2, 2, 3, 5]
            + [3] * 4 + [5, 4]
            + [1] * 6 + [4] * 5
            + [1, 2, 1, 2, 4, 4]
            + [2, 5, 1]
        ),
    }
    searched_tail = (
        [2, 2, 3, 5]
        + [3] * 4 + [5, 4]
        + [1] * 6 + [4] * 5
        + [5, 1, 3, 2, 5, 2, 5, 1]
    )
    for final_action in (1, 2, 3, 4, 5):
        candidates[f"searched_final_{final_action}"] = (
            searched_tail + [final_action]
        )
    searched_short_tail = (
        [2, 2, 3, 5]
        + [3] * 3 + [5, 4]
        + [1] * 6 + [4] * 5
        + [5, 1, 3, 2, 5, 2, 5, 1]
    )
    for final_action in (1, 2, 3, 4, 5):
        candidates[f"short_final_{final_action}"] = (
            searched_short_tail + [final_action]
        )
    no_dismiss_tail = (
        [2] * 2 + [3] * 5 + [5, 4]
        + [1] * 6 + [4] * 5
        + [5, 1, 3, 2, 5, 2, 5, 1]
    )
    for final_action in (1, 2, 3, 4, 5):
        candidates[f"no_dismiss_final_{final_action}"] = (
            no_dismiss_tail + [final_action]
        )

    for name, tail in candidates.items():
        candidate_prefix = (
            direct_second_prefix()
            if name.startswith(
                ("direct_", "searched_", "short_", "no_dismiss_")
            )
            else prefix
        )
        clone = env.clone()
        base_level = clone.levels_completed
        prior = None
        used = 0
        for action in candidate_prefix + tail:
            if clone.terminal() or clone.levels_completed > base_level:
                break
            clone.step(action)
            used += 1
            target = target_state(clone.frame())
            condensed = (target["empty"], target["filled"])
            if used >= 40 and condensed != prior:
                print(
                    "CANDIDATE_TRACE",
                    name,
                    used,
                    condensed,
                    boxes(clone.frame(), 12),
                    boxes(clone.frame(), 14),
                    flush=True,
                )
            prior = condensed
        print(
            "CANDIDATE_RESULT",
            name,
            {
                "requested": len(candidate_prefix + tail),
                "used": used,
                "reward": clone.levels_completed - base_level,
                "terminal": clone.terminal(),
                "target": target_state(clone.frame()),
            },
            flush=True,
        )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)

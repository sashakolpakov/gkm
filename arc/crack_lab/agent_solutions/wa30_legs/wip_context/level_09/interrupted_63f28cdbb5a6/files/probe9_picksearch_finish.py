"""Verify the one-action-faster pristine level-9 candidate for reward."""

import gkm_try

from probe9_candidate import direct_second_prefix
from probe9_verify import ReachedLevel9, StopAtLevel9, boxes, target_state


COMBINED_DISMISS_PICK = [2, 2, 3, 5, 3, 3, 3]
DELIVER_LOCAL = (
    [4] + [1] * 6 + [4] * 5
    + [4, 5, 1, 3, 2, 5, 2, 5, 1]
    + [5]
)


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel9(env))
    except ReachedLevel9:
        pass

    clone = env.clone()
    base_level = clone.levels_completed
    route = direct_second_prefix() + COMBINED_DISMISS_PICK + DELIVER_LOCAL
    prior = target_state(clone.frame())
    for turn, action in enumerate(route, 1):
        clone.step(action)
        current = target_state(clone.frame())
        if current != prior or turn >= 64:
            print(
                "FAST_EVENT",
                {
                    "turn": turn,
                    "action": action,
                    "level": clone.levels_completed,
                    "terminal": clone.terminal(),
                    "target": current,
                    "avatar": boxes(clone.frame(), 14),
                },
                flush=True,
            )
        prior = current
        if clone.terminal() or clone.levels_completed > base_level:
            break
    print(
        "FAST_RESULT",
        {
            "requested": len(route),
            "level": clone.levels_completed,
            "reward": clone.levels_completed - base_level,
            "terminal": clone.terminal(),
            "target": target_state(clone.frame()),
        },
        flush=True,
    )
    while (
        not clone.terminal()
        and clone.levels_completed == base_level
        and len(route) < 70
    ):
        clone.step(5)
        route.append(5)
        print(
            "FAST_WAIT",
            len(route),
            clone.levels_completed - base_level,
            clone.terminal(),
            target_state(clone.frame()),
            flush=True,
        )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)

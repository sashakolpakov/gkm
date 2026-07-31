"""Verify carrying the final level-9 cargo from below for a direct drop."""

import gkm_try

from probe9_candidate import direct_second_prefix
from probe9_verify import ReachedLevel9, StopAtLevel9, boxes, target_state


DISMISS_AND_PICK_BELOW = (
    [2, 2, 3, 5]
    + [2] + [3] * 4 + [1, 5]
)
DIRECT_TOP_RIGHT = [4] * 7 + [1] * 5 + [3, 5, 4]


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel9(env))
    except ReachedLevel9:
        pass

    clone = env.clone()
    base_level = clone.levels_completed
    route = direct_second_prefix() + DISMISS_AND_PICK_BELOW + DIRECT_TOP_RIGHT
    prior = target_state(clone.frame())
    for turn, action in enumerate(route, 1):
        clone.step(action)
        current = target_state(clone.frame())
        if current != prior or action == 5 or turn >= 60:
            print(
                "BELOW_EVENT",
                {
                    "turn": turn,
                    "action": action,
                    "level": clone.levels_completed,
                    "terminal": clone.terminal(),
                    "avatar": boxes(clone.frame(), 14),
                    "thief": boxes(clone.frame(), 15),
                    "target": current,
                },
                flush=True,
            )
        prior = current
        if clone.terminal() or clone.levels_completed > base_level:
            break
    print(
        "BELOW_RESULT",
        {
            "requested": len(route),
            "level": clone.levels_completed,
            "reward": clone.levels_completed - base_level,
            "terminal": clone.terminal(),
            "target": target_state(clone.frame()),
        },
        flush=True,
    )
    turn = len(route)
    while (
        turn < 70
        and not clone.terminal()
        and clone.levels_completed == base_level
    ):
        clone.step(5)
        turn += 1
        print(
            "BELOW_WAIT",
            turn,
            clone.levels_completed - base_level,
            clone.terminal(),
            target_state(clone.frame()),
            flush=True,
        )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)

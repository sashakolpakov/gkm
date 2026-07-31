"""Observe autonomous deliveries after the fast level-9 thief dismissal."""

import gkm_try

from probe9_prefix_shortcuts import direct_short_prefix, reach_level_9
from probe9_verify import boxes, target_state


PREFIX = direct_short_prefix() + [3] * 5 + [5]


def compact(env, turn):
    return {
        "turn": turn,
        "level": env.levels_completed,
        "terminal": env.terminal(),
        "helpers": boxes(env.frame(), 12),
        "cargo": boxes(env.frame(), 4),
        "target": target_state(env.frame()),
    }


def inspect(env):
    reach_level_9(env)
    state = env.clone()
    for action in PREFIX:
        state.step(action)
    turn = len(PREFIX)
    prior = None
    while not state.terminal() and state.levels_completed == env.levels_completed:
        current = compact(state, turn)
        condensed = (
            current["helpers"],
            current["target"]["empty"],
            current["target"]["filled"],
        )
        if condensed != prior:
            print("SHORT_IDLE_EVENT", current, flush=True)
        prior = condensed
        state.step(5)
        turn += 1
    print("SHORT_IDLE_END", compact(state, turn), flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)

"""Compact level-8 route experiments for the joint action cap."""

import gkm_try

from perception import bounded_bfs, connected_components
from probe8_entry import ReachedLevel8, StopAtLevel8
from probe8_trace import target_state
from probe9_verify import boxes, tile_map


def replay(base, actions):
    clone = base.clone()
    for action in actions:
        if clone.terminal() or clone.levels_completed > base.levels_completed:
            break
        clone.step(action)
    return clone


def state(env):
    empty, filled = target_state(env.frame())
    return {
        "level": env.levels_completed,
        "terminal": env.terminal(),
        "avatar": boxes(env.frame(), 14),
        "cargo": boxes(env.frame(), 4),
        "helpers": boxes(env.frame(), 12),
        "competitors": boxes(env.frame(), 15),
        "empty": empty,
        "filled": filled,
    }


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel8(env))
    except ReachedLevel8:
        pass

    idle_initial = env.clone()
    best = None
    for turn in range(0, 81):
        current = state(idle_initial)
        score = len(current["filled"])
        if best is None or score > best[0]:
            best = (score, turn, current)
            print("L8_RAW_IDLE_BEST", best, flush=True)
        if idle_initial.terminal() or idle_initial.levels_completed > 7:
            break
        idle_initial.step(5)

    reverse_top = [4] * 3 + [1] * 3 + [5] * 3
    reverse_top_state = replay(env, reverse_top)
    print(
        "L8_REVERSE_TOP",
        len(reverse_top),
        state(reverse_top_state),
        flush=True,
    )
    reverse_bottom_approach = (
        reverse_top + [2] * 3 + [4] * 5 + [2] * 5
    )
    approach_state = replay(env, reverse_bottom_approach)
    reverse_contact = bounded_bfs(
        approach_state,
        lambda candidate, path: not boxes(candidate.frame(), 15),
        max_states=5000,
        max_depth=10,
    )
    print(
        "L8_REVERSE_CONTACT",
        len(reverse_bottom_approach),
        state(approach_state),
        reverse_contact,
        flush=True,
    )
    reverse_dismiss = reverse_bottom_approach + reverse_contact
    reverse_state = replay(env, reverse_dismiss)
    print(
        "L8_REVERSE_DISMISSED",
        len(reverse_dismiss),
        state(reverse_state),
        flush=True,
    )
    print(
        "L8_REVERSE_DISMISSED_MAP",
        *tile_map(reverse_state.frame()),
        sep="\n",
        flush=True,
    )
    for left_steps in range(0, 7):
        reverse_both = (
            reverse_top
            + [2] * 3 + [4] * 5 + [2] * 5
            + [3] * left_steps + [5] * 3
        )
        result = replay(env, reverse_both)
        print(
            "L8_REVERSE_BOTH",
            left_steps,
            len(reverse_both),
            state(result),
            flush=True,
        )

    dismiss_bottom = [4] * 8 + [2] * 5 + [3] * 3 + [5]
    dismiss_top = (
        [4] * 3 + [1] * 5 + [3] * 5 + [1] * 4
        + [3, 1] + [5] * 3
    )
    bottom_state = replay(env, dismiss_bottom)
    print("L8_BOTTOM_DISMISSED", 17, state(bottom_state), flush=True)
    print(
        "L8_BOTTOM_MAP",
        *tile_map(bottom_state.frame()),
        sep="\n",
        flush=True,
    )
    dismissed = replay(env, dismiss_bottom + dismiss_top)
    print("L8_DISMISSED", 39, state(dismissed), flush=True)
    print("L8_DISMISSED_MAP", *tile_map(dismissed.frame()), sep="\n",
          flush=True)
    idle = dismissed.clone()
    prior = None
    for turn in range(39, 131):
        current = state(idle)
        condensed = (
            current["empty"],
            current["filled"],
            current["helpers"],
            current["level"],
            current["terminal"],
        )
        if condensed != prior and (
            prior is None
            or current["empty"] != prior[0]
            or current["filled"] != prior[1]
        ):
            print("L8_IDLE", turn, current, flush=True)
        if idle.terminal() or idle.levels_completed > env.levels_completed:
            break
        prior = condensed
        idle.step(5)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)

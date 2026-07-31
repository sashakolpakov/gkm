"""Reposition friendly lower deliveries into otherwise-unused target slots."""

import gkm_try

from perception import bounded_bfs
from probe8_entry import ReachedLevel8, StopAtLevel8
from probe8_reverse_idle import ROUTE
from probe8_reverse_stage import compact
from probe8_trace import target_state
from probe9_verify import tile_map


SHIFT_TWO = (
    [4] * 4 + [5] + [4] * 2 + [5]
    + [3] * 2 + [1] * 2 + [5] + [4] * 2 + [5]
)
SHIFT_BOTTOM_QUEUE = (
    [4] * 4 + [5] + [4] * 2 + [5]
    + [5] * 19
    + [4, 3, 5, 4, 5]
)


def run_candidate(base, actions, label):
    clone = base.clone()
    base_level = clone.levels_completed
    prior = compact(clone, len(ROUTE))
    for offset, action in enumerate(actions, 1):
        clone.step(action)
        current = compact(clone, len(ROUTE) + offset)
        if (
            action == 5
            or current["empty"] != prior["empty"]
            or current["filled"] != prior["filled"]
        ):
            print(label, action, current, flush=True)
        prior = current
    print(label + "_MAP", *tile_map(clone.frame()), sep="\n", flush=True)
    turn = len(ROUTE) + len(actions)
    while clone.levels_completed == base_level and not clone.terminal():
        clone.step(5)
        turn += 1
        current = compact(clone, turn)
        if (
            current["empty"] != prior["empty"]
            or current["filled"] != prior["filled"]
            or current["level"] != base_level
        ):
            print(label + "_WAIT", current, flush=True)
        prior = current
    print(label + "_RESULT", compact(clone, turn), flush=True)


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel8(env))
    except ReachedLevel8:
        pass
    start = env.clone()
    for action in ROUTE:
        start.step(action)
    run_candidate(start, SHIFT_TWO, "SHIFT_TWO")
    run_candidate(start, SHIFT_TWO[:8], "SHIFT_ONE")
    run_candidate(start, SHIFT_BOTTOM_QUEUE, "SHIFT_QUEUE")

    local = start.clone()
    setup = (
        [4] * 4 + [5] + [4] * 2 + [5]
        + [3] * 2 + [5] * 24
    )
    for action in setup:
        local.step(action)
    print("SHIFT_SEARCH_START", compact(local, 62), flush=True)
    path = bounded_bfs(
        local,
        lambda candidate, _: {
            (14, 13), (14, 14)
        }.issubset(set(target_state(candidate.frame())[1])),
        actions=(3, 4, 5, 1, 2),
        max_states=3000,
        max_depth=6,
    )
    print("SHIFT_SEARCH_RESULT", path, flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)

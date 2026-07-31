"""Combine an early top delivery with a later lower-courier dismissal."""

import gkm_try

from perception import bounded_bfs
from probe8_entry import ReachedLevel8, StopAtLevel8
from probe8_lower_manual import REVERSE_TOP
from probe8_reverse_stage import compact
from probe9_verify import boxes, tile_map


UPPER_DELIVERY = [1, 1, 5, 2] + [4] * 7 + [1, 5, 2]


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel8(env))
    except ReachedLevel8:
        pass

    delivered = env.clone()
    for action in REVERSE_TOP + UPPER_DELIVERY:
        delivered.step(action)
    print("COMBO_DELIVERED", compact(delivered, 24), flush=True)
    print(*tile_map(delivered.frame()), sep="\n", flush=True)

    approaches = (
        [3] * 6 + [2] * 3 + [4] * 5 + [2] * 4,
        [3] * 7 + [2] * 3 + [4] * 5 + [2] * 4,
        [3] * 6 + [2] * 3 + [4] * 4 + [2] * 4,
    )
    for index, approach in enumerate(approaches):
        node = delivered.clone()
        for action in approach:
            node.step(action)
        path = bounded_bfs(
            node,
            lambda candidate, _: not boxes(candidate.frame(), 15),
            actions=(5, 1, 2, 3, 4),
            max_states=10000,
            max_depth=9,
        )
        print(
            "COMBO_APPROACH",
            index,
            len(approach),
            compact(node, 24 + len(approach)),
            path,
            flush=True,
        )
        print(*tile_map(node.frame()), sep="\n", flush=True)

    best = delivered.clone()
    best_suffix = approaches[1] + [5, 3, 5]
    for action in best_suffix:
        best.step(action)
    print(
        "COMBO_DISMISSED",
        len(REVERSE_TOP + UPPER_DELIVERY + best_suffix),
        compact(best, len(REVERSE_TOP + UPPER_DELIVERY + best_suffix)),
        flush=True,
    )
    print(*tile_map(best.frame()), sep="\n", flush=True)

    lower_two = (
        [1] + [3] * 4 + [2, 5, 1] + [4] * 10 + [2, 5]
        + [3] * 8 + [2] * 3 + [3, 5, 4, 1] + [4] * 8 + [5]
    )
    base_level = best.levels_completed
    prior = compact(best, 45)
    for offset, action in enumerate(lower_two, 1):
        best.step(action)
        current = compact(best, 45 + offset)
        if (
            action == 5
            or current["empty"] != prior["empty"]
            or current["filled"] != prior["filled"]
            or current["level"] != base_level
        ):
            print("COMBO_LOWER", action, current, flush=True)
        prior = current
        if best.levels_completed > base_level or best.terminal():
            break
    wait_turn = 45 + len(lower_two)
    while best.levels_completed == base_level and not best.terminal():
        best.step(5)
        wait_turn += 1
        current = compact(best, wait_turn)
        if (
            current["empty"] != prior["empty"]
            or current["filled"] != prior["filled"]
            or current["level"] != base_level
        ):
            print("COMBO_WAIT", current, flush=True)
        prior = current
    print("COMBO_RESULT", compact(best, wait_turn), flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)

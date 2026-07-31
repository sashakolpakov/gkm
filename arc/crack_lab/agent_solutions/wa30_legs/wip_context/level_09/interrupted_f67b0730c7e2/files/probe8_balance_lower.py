"""Probe a lower delivery immediately after the first level-8 dismissal."""

import gkm_try

from perception import bounded_bfs
from probe8_entry import ReachedLevel8, StopAtLevel8
from probe8_reverse_stage import compact
from probe9_verify import boxes, tile_map


DISMISS_LOWER = [4] * 8 + [2] * 5 + [3] * 3 + [5]
LOWER_STAGES = {
    "port": [3, 3, 5] + [1] * 2 + [4] * 4 + [5, 2],
    "target": [3, 3, 5] + [1] * 2 + [4] * 5 + [5],
}
TRANSFER_ONE = [4] * 7 + [5, 2]
TRANSFER_TWO = [3] * 9 + [1, 5] + [4] * 11 + [1, 5, 2]


def finish(base, route, label):
    clone = base.clone()
    base_level = clone.levels_completed
    for action in route:
        clone.step(action)
        if clone.levels_completed > base_level or clone.terminal():
            break
    directed = len(route)
    print(
        "BALANCE_COMPOSED",
        label,
        "directed",
        directed,
        "competitors",
        boxes(clone.frame(), 15),
        compact(clone, directed),
        flush=True,
    )
    turn = directed
    while clone.levels_completed == base_level and not clone.terminal():
        clone.step(5)
        turn += 1
    print(
        "BALANCE_RESULT",
        label,
        turn,
        clone.levels_completed - base_level,
        compact(clone, turn),
        flush=True,
    )


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel8(env))
    except ReachedLevel8:
        pass
    base = env.clone()
    for action in DISMISS_LOWER:
        base.step(action)
    print("BALANCE_LOWER", "start", compact(base, 17), flush=True)
    for label, route in LOWER_STAGES.items():
        clone = base.clone()
        for action in route:
            clone.step(action)
        print(
            "BALANCE_LOWER",
            label,
            compact(clone, 17 + len(route)),
            flush=True,
        )
        print(*tile_map(clone.frame()), sep="\n", flush=True)
    top_tails = {
        "short": [5, 5, 3, 5],
        "wait_before": [5, 5, 5, 3, 5],
        "wait_after": [5, 5, 3, 5, 5],
    }
    for stage_label, stage in LOWER_STAGES.items():
        up_steps = 4 if stage_label == "port" else 3
        approach = [4] * 4 + [1] * up_steps + [3] * 4 + [1] * 4
        approached = env.clone()
        for action in DISMISS_LOWER + stage + approach:
            approached.step(action)
        contact = bounded_bfs(
            approached,
            lambda node, _path: not boxes(node.frame(), 15),
            actions=(5, 3, 1, 2, 4),
            max_states=8000,
            max_depth=12,
        )
        print(
            "BALANCE_CONTACT",
            stage_label,
            boxes(approached.frame(), 14),
            boxes(approached.frame(), 15),
            contact,
            flush=True,
        )
        if contact is not None:
            finish(
                env,
                DISMISS_LOWER
                + stage
                + approach
                + contact
                + TRANSFER_ONE
                + TRANSFER_TWO,
                stage_label + "_searched",
            )
        for tail_label, tail in top_tails.items():
            finish(
                env,
                DISMISS_LOWER
                + stage
                + approach
                + tail
                + TRANSFER_ONE
                + TRANSFER_TWO,
                stage_label + "_" + tail_label,
            )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)

"""Test timing-preserving removals of blocked moves in the level-9 prefix."""

import gkm_try

from perception import arr
from probe9_verify import (
    ReachedLevel9,
    StopAtLevel9,
    boxes,
    summary,
    target_state,
)


def joint_finish(env, max_depth=6):
    frontier = [(env.clone(), [])]
    for depth in range(1, max_depth + 1):
        next_states = {}
        wins = []
        for node, path in frontier:
            for action in node.actions:
                child = node.clone()
                child.step(action)
                child_path = path + [action]
                target = target_state(child.frame())
                if (
                    (5, 6) in target["filled"]
                    and not boxes(child.frame(), 15)
                ):
                    wins.append((child, child_path))
                else:
                    next_states.setdefault(
                        arr(child.frame()).tobytes(),
                        (child, child_path),
                    )
        if wins:
            return depth, wins
        frontier = list(next_states.values())
    return None, []


def build_variant(env, settle_pick, second_downs, second_ups=2):
    remote_pick = [2] + [4] * 6 + [1, 5]
    if settle_pick:
        remote_pick += [2]
    first_delivery = remote_pick + [3] * 6 + [1] * 3 + [3, 5]
    second_pick = (
        first_delivery
        + [2] * second_downs + [4] * 5
        + [1] * second_ups + [5]
    )
    place_bottom_middle = [2] + [3] * 6 + [1] * 3 + [5]
    path = second_pick + place_bottom_middle
    state = env.clone()
    for action in path:
        state.step(action)
    return state, path


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel9(env))
    except ReachedLevel9:
        pass

    for name, settle_pick, second_downs, second_ups in (
        ("baseline", True, 4, 2),
        ("no_pick_settle", False, 4, 2),
        ("three_downs", True, 3, 2),
        ("both_short", False, 3, 2),
        ("direct_second", False, 2, 1),
    ):
        state, prefix = build_variant(
            env, settle_pick, second_downs, second_ups
        )
        depth, wins = joint_finish(state)
        print(
            "VARIANT",
            name,
            {
                "prefix_turn": len(prefix),
                "prefix": summary(state, len(prefix)),
                "joint_depth": depth,
                "wins": [
                    {
                        "path": path,
                        "turn": len(prefix) + len(path),
                        "avatar": boxes(child.frame(), 14),
                        "target": target_state(child.frame()),
                    }
                    for child, path in wins[:12]
                ],
            },
            flush=True,
        )

    alternate_order = (
        [4] * 4 + [1, 5]
        + [2] + [3] * 5 + [1] * 3 + [5]
        + [2] * 3 + [4] * 7 + [1, 5]
        + [3] * 8 + [1] * 3 + [5]
    )
    alternate = env.clone()
    for action in alternate_order:
        alternate.step(action)
    depth, wins = joint_finish(alternate)
    print(
        "VARIANT",
        "alternate_order",
        {
            "prefix_turn": len(alternate_order),
            "prefix": summary(alternate, len(alternate_order)),
            "joint_depth": depth,
            "wins": [
                {
                    "path": path,
                    "turn": len(alternate_order) + len(path),
                    "avatar": boxes(child.frame(), 14),
                    "target": target_state(child.frame()),
                }
                for child, path in wins[:12]
            ],
        },
        flush=True,
    )


gkm_try.A.run_program("wa30", inspect)

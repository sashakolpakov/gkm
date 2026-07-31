"""Combine the second right-slot delivery with an early thief dismissal."""

import gkm_try

from perception import bounded_replay_bfs
from probe9_actual_candidates import state
from probe9_prefix_shortcuts import reach_level_9
from probe9_verify import boxes, target_state, tile_map


MIDDLE_RIGHT = (
    [4] * 5 + [1, 3, 5]
    + [2] * 2 + [3] * 5 + [1] * 5
    + [5, 2]
)
SECOND_PICK = [2] * 4 + [4] * 6 + [1, 5]


def combined_goal(env, _path):
    target = target_state(env.frame())
    return (
        (5, 5) in target["filled"]
        and (5, 7) in target["filled"]
        and not boxes(env.frame(), 15)
    )


def reward_goal(base_level):
    return lambda env, _path: env.levels_completed > base_level


def inspect(env):
    reach_level_9(env)
    picked = env.clone()
    prefix = MIDDLE_RIGHT + SECOND_PICK
    for action in prefix:
        picked.step(action)
    print("REVERSE_COMBINED_START", len(prefix), state(picked), flush=True)

    direct_bridge = [3] * 6 + [1] * 3 + [3, 5, 3, 5]
    direct = picked.clone()
    for action in direct_bridge:
        direct.step(action)
    print(
        "REVERSE_COMBINED_DIRECT",
        len(prefix) + len(direct_bridge),
        state(direct),
        flush=True,
    )
    if combined_goal(direct, direct_bridge):
        bridge = direct_bridge
    else:
        bridge = bounded_replay_bfs(
            picked,
            combined_goal,
            lambda node: node.actions,
            max_states=5000,
            max_depth=16,
        )
    print("REVERSE_COMBINED_BRIDGE", bridge, flush=True)
    if bridge is None:
        return
    bridged = picked.clone()
    for action in bridge:
        bridged.step(action)
    turn = len(prefix) + len(bridge)
    print("REVERSE_COMBINED_STATE", turn, state(bridged), flush=True)
    print(*tile_map(bridged.frame()), sep="\n", flush=True)

    direct_finish = (
        [3] * 5 + [2, 5]
        + [1] * 5 + [4] * 6
        + [2, 5, 1]
    )
    direct_final = bridged.clone()
    base_level = direct_final.levels_completed
    for action in direct_finish:
        direct_final.step(action)
        if direct_final.levels_completed > base_level:
            break
    print(
        "REVERSE_COMBINED_DIRECT_FINISH",
        turn + len(direct_finish),
        state(direct_final),
        flush=True,
    )
    for extra in direct_final.actions:
        child = direct_final.clone()
        child.step(extra)
        print(
            "REVERSE_COMBINED_EXTRA",
            extra,
            child.levels_completed - base_level,
            state(child),
            flush=True,
        )
    final_tick = direct_final.clone()
    final_tick.step(5)
    final_tick.step(5)
    print(
        "REVERSE_COMBINED_FINAL_TICK",
        final_tick.levels_completed - base_level,
        state(final_tick),
        flush=True,
    )
    if direct_final.levels_completed > base_level:
        finish = direct_finish
    else:
        finish = None
        for extra in direct_final.actions:
            child = direct_final.clone()
            child.step(extra)
            if child.levels_completed > base_level:
                finish = direct_finish + [extra]
                break
        if final_tick.levels_completed > base_level:
            finish = direct_finish + [5, 5]
    print("REVERSE_COMBINED_FINISH", finish, flush=True)
    if finish is not None:
        print("REVERSE_COMBINED_WIN", prefix + bridge + finish, flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)

"""Test compact approaches to the first two level-9 remote blocks."""

import gkm_try

from probe9_verify import (
    ReachedLevel9,
    StopAtLevel9,
    boxes,
    target_state,
    tile_map,
)


def reach_level_9(env):
    try:
        gkm_try.resumed_solve(StopAtLevel9(env))
    except ReachedLevel9:
        pass


def compact(env):
    avatar = boxes(env.frame(), 14)
    cargo = boxes(env.frame(), 4)
    return {
        "avatar": avatar,
        "cargo": cargo,
        "courier": boxes(env.frame(), 12),
        "thief": boxes(env.frame(), 15),
        "target": target_state(env.frame())["filled"],
    }


def run(env, name, path):
    child = env.clone()
    base = child.levels_completed
    used = 0
    for action in path:
        if child.terminal() or child.levels_completed > base:
            break
        child.step(action)
        used += 1
    print(
        "PREFIX_SHORTCUT",
        name,
        {
            "requested": len(path),
            "used": used,
            "reward": child.levels_completed - base,
            "terminal": child.terminal(),
            "state": compact(child),
        },
        flush=True,
    )


def direct_short_prefix():
    short_first = (
        [4] * 5 + [5]
        + [3] * 5 + [1] * 3 + [3, 5]
    )
    second_pick = [2] * 2 + [4] * 5 + [1, 5]
    place_bottom_middle = [2] + [3] * 6 + [1] * 3 + [5]
    return short_first + second_pick + place_bottom_middle


def inspect(env):
    reach_level_9(env)
    candidates = {
        "right4_use": [4] * 4 + [5],
        "right5_use": [4] * 5 + [5],
        "right4_down_use": [4] * 4 + [2, 5],
        "down_right4_use": [2] + [4] * 4 + [5],
        "down_right5_use": [2] + [4] * 5 + [5],
        "down_right6_up_use": [2] + [4] * 6 + [1, 5],
        "right4_down_right_use": [4] * 4 + [2, 4, 5],
    }
    for name, path in candidates.items():
        run(env, name, path)

    short_first = [4] * 5 + [5] + [3] * 5 + [1] * 3 + [3, 5]
    second_pick = [2] * 2 + [4] * 5 + [1, 5]
    place_bottom_middle = [2] + [3] * 6 + [1] * 3 + [5]
    short_prefix = direct_short_prefix()
    combined_dismiss = [2, 2, 3, 5, 3, 3, 3]
    position = [5, 4] + [1] * 6 + [4] * 5
    known_suffix = [4, 5, 1, 3, 2, 5, 2, 5, 1]
    run(env, "short_first", short_first)
    run(env, "short_prefix", short_prefix)
    mapped = env.clone()
    for action in short_prefix:
        mapped.step(action)
    print("SHORT_PREFIX_MAP", *tile_map(mapped.frame()), sep="\n", flush=True)
    run(
        env,
        "short_complete_known",
        short_prefix + combined_dismiss + position + known_suffix + [1] * 5,
    )
    short_dismiss = [3] * 5 + [5]
    short_pick = [3, 3, 1, 5]
    short_position = [4] + [1] * 6 + [4] * 5
    run(
        env,
        "short_complete_local",
        (
            short_prefix
            + short_dismiss
            + short_pick
            + short_position
            + known_suffix
            + [1] * 5
        ),
    )
    collision_pick = [1, 3, 3]
    run(
        env,
        "short_complete_collision",
        (
            short_prefix
            + short_dismiss
            + collision_pick
            + short_position
            + known_suffix
            + [1] * 8
        ),
    )
    run(
        env,
        "short_complete_collision_use",
        (
            short_prefix
            + short_dismiss
            + collision_pick
            + [5]
            + short_position
            + known_suffix
            + [1] * 8
        ),
    )
    phase_preserving_prefix = direct_short_prefix() + [2] * 3
    run(env, "phase_preserving_prefix", phase_preserving_prefix)
    run(
        env,
        "phase_preserving_complete",
        (
            phase_preserving_prefix
            + combined_dismiss
            + [5]
            + short_position
            + known_suffix
            + [1] * 3
        ),
    )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)

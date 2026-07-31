"""Exercise the wall-separated right depots with the full 12-slot metric."""

import gkm_try

from perception import arr, bounded_bfs
from probe9_actual_candidates import state
from probe9_prefix_shortcuts import reach_level_9
from probe9_two_staged_trace import LOCAL_FINISH
from probe9_verify import TARGET, boxes, tile_map


FULL_TARGET = TARGET | {(2, 13), (2, 14), (6, 13), (6, 14)}


def full_target_state(frame):
    grid = arr(frame)
    empty = []
    filled = []
    occupied = []
    for row, col in sorted(FULL_TARGET):
        colors = set(int(value) for value in
                     grid[row * 4:row * 4 + 4,
                          col * 4:col * 4 + 4].flat)
        if 2 in colors and 9 in colors:
            empty.append((row, col))
        if 4 in colors and 9 in colors:
            filled.append((row, col))
        if colors - {2, 9}:
            occupied.append((row, col, tuple(sorted(colors))))
    return tuple(empty), tuple(filled), tuple(occupied)


def show(env, turn, label):
    print(
        "RIGHT_DEPOT",
        label,
        turn,
        {
            "level": env.levels_completed,
            "terminal": env.terminal(),
            "avatar": boxes(env.frame(), 14),
            "couriers": boxes(env.frame(), 12),
            "thief": boxes(env.frame(), 15),
            "cargo": boxes(env.frame(), 4),
            "target": full_target_state(env.frame()),
        },
        flush=True,
    )
    print(*tile_map(env.frame()), sep="\n", flush=True)


def inspect(env):
    reach_level_9(env)
    clone = env.clone()
    show(clone, 0, "start")

    first = (
        [2] + [4] * 6 + [1, 5]
        + [1] * 2 + [5, 2]
    )
    second = [3] * 2 + [1, 5] + [4, 1, 5, 2]
    turn = 0
    third = [3] * 2 + [1] * 3 + [5] + [1] * 2 + [5, 2]
    for label, route in (
        ("lower_right", first),
        ("lower_left", second),
        ("upper_port", third),
    ):
        for action in route:
            clone.step(action)
            turn += 1
        show(clone, turn, label)

    candidate = clone.clone()
    to_intercept = [2] * 3 + [3] * 5
    intercept = clone.clone()
    for action in to_intercept:
        intercept.step(action)
    for path in (
        [2, 3, 3, 5],
        [2, 3, 3, 3, 5],
        [3, 2, 3, 5],
        [2, 3, 1, 3, 5],
        [1, 3, 3, 3, 5],
    ):
        child = intercept.clone()
        for action in path:
            child.step(action)
        print(
            "RIGHT_DEPOT_DISMISS_VARIANT",
            path,
            boxes(child.frame(), 14),
            boxes(child.frame(), 15),
            flush=True,
        )
    searched_dismiss = bounded_bfs(
        intercept,
        lambda node, path: not boxes(node.frame(), 15),
        max_states=5000,
        max_depth=8,
    )
    print("RIGHT_DEPOT_DISMISS_SEARCH", searched_dismiss, flush=True)
    dismiss = to_intercept + (
        searched_dismiss if searched_dismiss is not None else [2, 3, 3, 5]
    )
    local_right_finish = (
        [3] * 4 + [1, 5]
        + [4] * 7 + [1] * 3 + [3, 5, 2]
        + [5] * 7
    )
    prior = full_target_state(candidate.frame())
    for index, action in enumerate(dismiss + local_right_finish):
        candidate.step(action)
        turn += 1
        current = full_target_state(candidate.frame())
        if index < len(dismiss):
            print(
                "RIGHT_DEPOT_DISMISS_TRACE",
                turn,
                action,
                boxes(candidate.frame(), 14),
                boxes(candidate.frame(), 15),
                flush=True,
            )
        if (
            current != prior
            or candidate.levels_completed > env.levels_completed
            or turn >= 64
        ):
            print(
                "RIGHT_DEPOT_CANDIDATE",
                turn,
                action,
                candidate.levels_completed - env.levels_completed,
                current,
                boxes(candidate.frame(), 12),
                boxes(candidate.frame(), 15),
                flush=True,
            )
        prior = current
        if candidate.terminal() or candidate.levels_completed > env.levels_completed:
            break
    print(
        "RIGHT_DEPOT_RESULT",
        turn,
        candidate.levels_completed - env.levels_completed,
        candidate.terminal(),
        full_target_state(candidate.frame()),
        flush=True,
    )

    turn = 31
    prior = full_target_state(clone.frame())
    for _ in range(40):
        clone.step(5)
        turn += 1
        current = full_target_state(clone.frame())
        if current != prior:
            print(
                "RIGHT_DEPOT_IDLE",
                turn,
                current,
                boxes(clone.frame(), 12),
                flush=True,
            )
        prior = current
        if clone.terminal():
            break


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)

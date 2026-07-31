"""Exhaust the lateral junctions for the two directed level-6 branch orders."""
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)


CLOCK = [0, 0.0]


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)
    CLOCK[0] += 1
    target = CLOCK[0] / 280.0
    elapsed = time.monotonic() - CLOCK[1]
    if target > elapsed:
        time.sleep(target - elapsed)


def axis_moves(start, target, negative, positive):
    return [negative if target < start else positive] * (
        abs(target - start) // 3
    )


def apply(env, actions):
    for action in actions:
        step(env, action)


def x_states(base, point, start_left, lefts):
    work = base.clone()
    step(work, (6, *point))
    current = start_left
    for action in axis_moves(current, lefts[0], 3, 4):
        step(work, action)
    current = lefts[0]
    for left in lefts:
        for action in axis_moves(current, left, 3, 4):
            step(work, action)
        current = left
        yield work.clone(), left


def direct_path(a_top, a_left, d_top, d_left, c_top, c_left):
    return (
        [(6, 25, 33)]
        + axis_moves(32, c_top, 1, 2)
        + axis_moves(23, c_left, 3, 4)
        + [(6, 30, 19)]
        + axis_moves(17, a_top, 1, 2)
        + axis_moves(29, a_left, 3, 4)
        + [(6, 33, 45)]
        + axis_moves(44, d_top, 1, 2)
        + axis_moves(29, d_left, 3, 4)
        + [(6, 45, 18)] + [2] * 9 + [3] * 5
        + [5]
    )


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)
    CLOCK[:] = [0, time.monotonic()]
    base_level = env.levels_completed
    a_lefts = tuple(range(5, 54, 3))
    d_lefts = tuple(range(5, 54, 3))
    c_lefts = tuple(range(5, 45, 3))
    tested = 0

    for a_top, d_top, c_top in (
        (20, 35, 32),
    ):
        arranged = env.clone()
        apply(arranged, [(6, 45, 18)] + [2] * 9 + [3] * 5)
        apply(arranged, [(6, 25, 33)] + axis_moves(32, c_top, 1, 2))
        apply(arranged, [(6, 30, 19)] + axis_moves(17, a_top, 1, 2))
        apply(arranged, [(6, 33, 45)] + axis_moves(44, d_top, 1, 2))

        for c_node, c_left in x_states(
            arranged, (25, c_top + 1), 23, c_lefts,
        ):
            for a_node, a_left in x_states(
                c_node, (30, a_top + 1), 29, a_lefts,
            ):
                for d_node, d_left in x_states(
                    a_node, (33, d_top + 1), 29, d_lefts,
                ):
                    test = d_node.clone()
                    step(test, 5)
                    tested += 1
                    if test.levels_completed > base_level:
                        path = direct_path(
                            a_top, a_left, d_top, d_left, c_top, c_left,
                        )
                        print(
                            "JUNCTION_WIN",
                            (a_top, a_left, d_top, d_left, c_top, c_left),
                            "TESTED", tested, "STEPS", CLOCK[0], "PATH", path,
                        )
                        return
    print("JUNCTION_NONE", tested, "STEPS", CLOCK[0])


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)

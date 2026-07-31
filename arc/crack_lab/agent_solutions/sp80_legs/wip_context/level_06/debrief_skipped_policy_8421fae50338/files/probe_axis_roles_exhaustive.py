"""Exhaust the two directed side-port role assignments on level 6."""
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as Arena

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


def moves(start, target, negative, positive):
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
    for action in moves(current, lefts[0], 3, 4):
        step(work, action)
    current = lefts[0]
    for left in lefts:
        for action in moves(current, left, 3, 4):
            step(work, action)
        current = left
        yield work.clone(), left


def direct_path(a_top, a_left, b_top, c_top, d_top, d_left):
    return (
        [(6, 25, 33)]
        + moves(32, c_top, 1, 2)
        + [(6, 30, 19)]
        + moves(17, a_top, 1, 2)
        + moves(29, a_left, 3, 4)
        + [(6, 33, 45)]
        + moves(44, d_top, 1, 2)
        + moves(29, d_left, 3, 4)
        + [(6, 45, 18)]
        + moves(14, b_top, 1, 2)
        + moves(44, 29, 3, 4)
        + [5]
    )


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)
    CLOCK[:] = [0, time.monotonic()]
    base_level = env.levels_completed
    lefts = tuple(range(5, 54, 3))
    tested = 0

    # The singleton cell of each corner occupies the middle gap of its socket:
    # A top=32 for the right socket; D top=23/38 for a left socket.
    for c_top, d_top in ((38, 23), (23, 38)):
        for b_top in range(14, 45, 3):
            arranged = env.clone()
            apply(
                arranged,
                [(6, 25, 33)] + moves(32, c_top, 1, 2)
                + [(6, 30, 19)] + moves(17, 32, 1, 2)
                + [(6, 33, 45)] + moves(44, d_top, 1, 2)
                + [(6, 45, 18)]
                + moves(14, b_top, 1, 2) + moves(44, 29, 3, 4),
            )
            for a_node, a_left in x_states(
                arranged, (33, 36), 29, lefts,
            ):
                for d_node, d_left in x_states(
                    a_node, (33, d_top + 1), 29, lefts,
                ):
                    test = d_node.clone()
                    step(test, 5)
                    tested += 1
                    if test.levels_completed > base_level:
                        path = direct_path(
                            32, a_left, b_top, c_top, d_top, d_left,
                        )
                        print(
                            "AXIS_ROLE_WIN", (29, a_left), (b_top, 29),
                            (c_top, 23), (d_top, d_left),
                            "TESTED", tested, "STEPS", CLOCK[0],
                            "PATH", path,
                        )
                        return
    print("AXIS_ROLE_NONE", tested, "STEPS", CLOCK[0])


if __name__ == "__main__":
    levels, path, err = Arena.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)

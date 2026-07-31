"""Categorical level-6 sweep over port rows and lateral orderings."""
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)
from probe_central_stack import moves, placements


def overlaps(top, left, height, width, other):
    other_top, other_left, other_height, other_width = other
    return not (
        top + height <= other_top
        or other_top + other_height <= top
        or left + width <= other_left
        or other_left + other_width <= left
    )


def direct_path(a, d, c, b):
    return (
        [(6, 30, 19)]
        + moves(17, a[0], 1, 2) + moves(29, a[1], 3, 4)
        + [(6, 33, 45)]
        + moves(44, d[0], 1, 2) + moves(29, d[1], 3, 4)
        + [(6, 25, 33)]
        + moves(32, c[0], 1, 2)
        + [(6, 45, 18)]
        + moves(14, b[0], 1, 2) + moves(44, b[1], 3, 4)
        + [5]
    )


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)
    base_level = env.levels_completed
    started = time.monotonic()
    clock = [0]
    tested = 0
    side_tops = (20, 29, 35)
    lateral = (14, 26, 38)
    for a_node, a in placements(
        env, (30, 19), 17, 29, side_tops, lateral, clock,
    ):
        for d_node, d in placements(
            a_node, (33, 45), 44, 29, side_tops, lateral, clock,
        ):
            for c_node, c in placements(
                d_node, (25, 33), 32, 23,
                (20, 23, 26, 29, 32, 35, 38), (23,), clock,
            ):
                occupied = (
                    (a[0], a[1], 6, 6),
                    (d[0], d[1], 6, 6),
                    (c[0], c[1], 3, 15),
                )
                for b_left in lateral:
                    b_tops = tuple(
                        top for top in
                        (14, 17, 20, 23, 26, 29, 32, 35, 38)
                        if not any(
                            overlaps(top, b_left, 12, 3, other)
                            for other in occupied
                        )
                    )
                    if not b_tops:
                        continue
                    for b_node, b in placements(
                        c_node, (45, 18), 14, 44,
                        b_tops, (b_left,), clock,
                    ):
                        test = b_node.clone()
                        try:
                            test.step(5)
                        except IndexError:
                            continue
                        clock[0] += 1
                        tested += 1
                        if test.levels_completed > base_level:
                            path = direct_path(a, d, c, b)
                            print(
                                "CATEGORY_WIN", a, d, c, b,
                                "TESTED", tested, "STEPS", clock[0],
                                "PATH", path,
                            )
                            return
                        target = clock[0] / 280.0
                        delay = target - (time.monotonic() - started)
                        if delay > 0:
                            time.sleep(delay)
    print("CATEGORY_NONE", tested, "STEPS", clock[0])


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)

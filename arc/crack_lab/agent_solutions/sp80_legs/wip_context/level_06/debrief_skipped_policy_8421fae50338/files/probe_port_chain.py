"""Search the level-6 port-aligned directed-chain family."""
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)
from probe_central_stack import moves, placements


def direct_path(a, c, d, b):
    return (
        [(6, 30, 19)]
        + moves(17, a[0], 1, 2) + moves(29, a[1], 3, 4)
        + [(6, 25, 33)]
        + moves(32, c[0], 1, 2) + moves(23, c[1], 3, 4)
        + [(6, 33, 45)]
        + moves(44, d[0], 1, 2) + moves(29, d[1], 3, 4)
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
    for a_node, a in placements(
        env, (30, 19), 17, 29,
        (29,), tuple(range(32, 54, 3)), clock,
    ):
        for d_node, d in placements(
            a_node, (33, 45), 44, 29,
            (20,), tuple(range(5, 27, 3)), clock,
        ):
            for c_node, c in placements(
                d_node, (25, 33), 32, 23,
                (26,), tuple(range(17, 30, 3)), clock,
            ):
                for b_top in (26,):
                    for b_left in range(5, 57, 3):
                        b_iter = placements(
                            c_node, (45, 18), 14, 44,
                            (b_top,), (b_left,), clock,
                        )
                        try:
                            b_node, b = next(b_iter)
                        except IndexError:
                            continue
                        test = b_node.clone()
                        try:
                            test.step(5)
                        except IndexError:
                            continue
                        clock[0] += 1
                        tested += 1
                        if test.levels_completed > base_level:
                            path = direct_path(a, c, d, b)
                            print(
                                "PORT_WIN", a, c, d, b,
                                "TESTED", tested, "STEPS", clock[0],
                                "PATH", path,
                            )
                            return
                        target = clock[0] / 280.0
                        delay = target - (time.monotonic() - started)
                        if delay > 0:
                            time.sleep(delay)
    print("PORT_NONE", tested, "STEPS", clock[0])


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)

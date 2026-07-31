"""Exhaust exact side-socket assignments for the level-6 relays."""
import itertools
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)
from probe_central_stack import moves, placements


SOCKET_TOPS = (20, 29, 35)
B_TOPS = {
    20: (14, 17, 20, 23),
    29: (23, 26, 29, 32),
    35: (29, 32, 35, 38),
}


def direct_path(a, d, c, b):
    return (
        [(6, 30, 19)]
        + moves(17, a[0], 1, 2) + moves(29, a[1], 3, 4)
        + [(6, 33, 45)]
        + moves(44, d[0], 1, 2) + moves(29, d[1], 3, 4)
        + [(6, 25, 33)]
        + moves(32, c[0], 1, 2) + moves(23, c[1], 3, 4)
        + [(6, 45, 18)]
        + moves(14, b[0], 1, 2) + moves(44, b[1], 3, 4)
        + [5]
    )


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)
    started = time.monotonic()
    base_level = env.levels_completed
    clock = [0]
    tested = 0
    for a_top, d_top in itertools.permutations(SOCKET_TOPS, 2):
        b_socket = next(t for t in SOCKET_TOPS if t not in (a_top, d_top))
        b_lefts = (
            tuple(range(32, 51, 3)) if b_socket == 29
            else tuple(range(5, 27, 3))
        )
        for a_node, a in placements(
            env, (30, 19), 17, 29, (a_top,), (29,), clock,
        ):
            for d_node, d in placements(
                a_node, (33, 45), 44, 29, (d_top,), (26,), clock,
            ):
                for c_node, c in placements(
                    d_node, (25, 33), 32, 23,
                    tuple(range(14, 51, 3)), (23,), clock,
                ):
                    for b_node, b in placements(
                        c_node, (45, 18), 14, 44,
                        B_TOPS[b_socket], b_lefts, clock,
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
                                "SOCKET_WIN", a, d, c, b,
                                "TESTED", tested, "STEPS", clock[0],
                                "PATH", path,
                            )
                            return
                        target = clock[0] / 280.0
                        delay = target - (time.monotonic() - started)
                        if delay > 0:
                            time.sleep(delay)
    print("SOCKET_NONE", tested, "STEPS", clock[0])


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)

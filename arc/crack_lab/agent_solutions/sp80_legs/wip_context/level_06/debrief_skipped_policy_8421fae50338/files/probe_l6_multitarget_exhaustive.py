"""Incremental exhaustive sweep for the bar-matches-both-left-sockets role."""
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as Arena

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)
from probe_central_stack import moves, placements
from probe_local_l6 import path_for, safe_order


def apply(env, actions):
    for action in actions:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)
    started = time.monotonic()
    clock = [0]
    tested = 0
    errors = 0

    staged = env.clone()
    # Park D clear of every original selection point, then socket-align A.
    prefix = (
        [(6, 31, 46)] + [1] * 11 + [4] * 8
        + [(6, 30, 19)] + [2] * 4 + [4] * 5
    )
    apply(staged, prefix)
    clock[0] += len(prefix)

    c_tops = tuple(range(11, 48, 3))
    b_lefts = tuple(range(14, 57, 3))
    d_tops = tuple(range(11, 45, 3))
    d_lefts = tuple(range(14, 54, 3))
    for c_node, c in placements(
        staged, (25, 33), 32, 23, c_tops, (23,), clock,
    ):
        for b_node, b in placements(
            c_node, (45, 18), 14, 44, (26,), b_lefts, clock,
        ):
            for d_node, d in placements(
                b_node, (57, 13), 11, 53, d_tops, d_lefts, clock,
            ):
                try:
                    test = d_node.clone()
                    test.step(5)
                except IndexError:
                    errors += 1
                    continue
                clock[0] += 1
                tested += 1
                if test.levels_completed > env.levels_completed:
                    targets = {
                        "A": (29, 44),
                        "B": b,
                        "C": c,
                        "D": d,
                    }
                    order = safe_order(targets)
                    path = path_for(targets, order) if order else prefix
                    print(
                        "MULTITARGET_WIN", targets, "ORDER", order,
                        "TESTED", tested, "STEPS", clock[0], "PATH", path,
                    )
                    return
                delay = clock[0] / 280.0 - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)
    print(
        "MULTITARGET_NONE", "TESTED", tested,
        "ERRORS", errors, "STEPS", clock[0],
    )


if __name__ == "__main__":
    levels, path, err = Arena.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)

"""Bounded level-6 search around the observed central stack."""
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)


def moves(start, target, negative, positive):
    return [negative if target < start else positive] * (
        abs(target - start) // 3
    )


def placements(base, point, start_top, start_left, tops, lefts, clock):
    work = base.clone()
    current_top, current_left = start_top, start_left
    def paced(action):
        try:
            work.step(*action) if isinstance(action, tuple) else work.step(action)
        except IndexError:
            print("PLACE_ERROR", point, action, current_top, current_left)
            raise
    paced((6, *point))
    clock[0] += 1
    first_top, first_left = tops[0], lefts[0]
    for action in moves(current_top, first_top, 1, 2):
        paced(action)
        clock[0] += 1
    for action in moves(current_left, first_left, 3, 4):
        paced(action)
        clock[0] += 1
    current_top, current_left = first_top, first_left
    for row_index, top in enumerate(tops):
        for action in moves(current_top, top, 1, 2):
            paced(action)
            clock[0] += 1
        current_top = top
        row_lefts = lefts if row_index % 2 == 0 else tuple(reversed(lefts))
        for left in row_lefts:
            for action in moves(current_left, left, 3, 4):
                paced(action)
                clock[0] += 1
            current_left = left
            yield work.clone(), (top, left)


def direct_path(a, b, c, d):
    return (
        [(6, 30, 19)]
        + moves(17, a[0], 1, 2) + moves(29, a[1], 3, 4)
        + [(6, 25, 33)]
        + moves(32, c[0], 1, 2) + moves(23, c[1], 3, 4)
        + [(6, 31, 46)]
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
        (20,), (26, 29, 32), clock,
    ):
        for c_node, c in placements(
            a_node, (25, 33), 32, 23,
            (32,), tuple(range(11, 42, 3)), clock,
        ):
            for d_node, d in placements(
                c_node, (31, 46), 44, 29,
                (35,), (26, 29, 32), clock,
            ):
                for b_node, b in placements(
                    d_node, (45, 18), 14, 44,
                    (23,), tuple(range(11, 51, 3)), clock,
                ):
                    try:
                        b_node.step(5)
                        completed = b_node.levels_completed
                    except IndexError:
                        continue
                    clock[0] += 1
                    tested += 1
                    if completed > base_level:
                        path = direct_path(a, b, c, d)
                        print(
                            "CENTRAL_WIN", a, b, c, d,
                            "TESTED", tested, "STEPS", clock[0], "PATH", path,
                        )
                        return
                    target = clock[0] / 280.0
                    elapsed = time.monotonic() - started
                    if target > elapsed:
                        time.sleep(target - elapsed)
    print("CENTRAL_NONE", tested, "STEPS", clock[0])


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)

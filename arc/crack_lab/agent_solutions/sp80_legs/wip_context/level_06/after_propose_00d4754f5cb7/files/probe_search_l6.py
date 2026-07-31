"""Bounded, socket-aligned clone search for the level-6 arrangement."""
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)


SEARCH_STEPS = 0
SEARCH_START = None


def step(env, action):
    global SEARCH_STEPS
    env.step(*action) if isinstance(action, tuple) else env.step(action)
    SEARCH_STEPS += 1
    if SEARCH_STEPS % 100 == 0:
        target = SEARCH_STEPS / 280.0
        elapsed = time.monotonic() - SEARCH_START
        if target > elapsed:
            time.sleep(target - elapsed)


def moves(start, target, negative, positive):
    return [negative if target < start else positive] * (
        abs(target - start) // 3
    )


def placed_states(base, point, start_top, start_left, tops, lefts, prefix):
    """Yield snake-walk clones and exact replay paths for one selected piece."""
    work = base.clone()
    path = list(prefix)
    select = (6, *point)
    step(work, select)
    path.append(select)
    current_top, current_left = start_top, start_left
    first_top, first_left = tops[0], lefts[0]
    for action in moves(current_top, first_top, 1, 2):
        step(work, action)
        path.append(action)
    for action in moves(current_left, first_left, 3, 4):
        step(work, action)
        path.append(action)
    current_top, current_left = first_top, first_left

    for row_index, top in enumerate(tops):
        if top != current_top:
            for action in moves(current_top, top, 1, 2):
                step(work, action)
                path.append(action)
            current_top = top
        row_lefts = lefts if row_index % 2 == 0 else tuple(reversed(lefts))
        for left in row_lefts:
            if left != current_left:
                for action in moves(current_left, left, 3, 4):
                    step(work, action)
                    path.append(action)
                current_left = left
            yield work.clone(), list(path), (current_top, current_left)


def direct_states(base, point, start_top, start_left, tops, lefts, prefix):
    """Yield independently replayed placements to keep clone histories short."""
    for top in tops:
        for left in lefts:
            work = base.clone()
            path = list(prefix)
            select = (6, *point)
            step(work, select)
            path.append(select)
            for action in (
                moves(start_top, top, 1, 2)
                + moves(start_left, left, 3, 4)
            ):
                step(work, action)
                path.append(action)
            yield work, path, (top, left)


def probe(env):
    global SEARCH_START
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)
    base_level = env.levels_completed
    SEARCH_START = time.monotonic()

    a_tops = (17, 20, 23, 26, 29, 32, 35, 38)
    b_tops = (23, 29)
    c_tops = (23, 38)
    d_tops = a_tops
    tested = 0

    for a, pa, posa in placed_states(
        env, (30, 19), 17, 29, a_tops, (17, 29, 38), [],
    ):
        for b, pb, posb in placed_states(
            a, (45, 18), 14, 44, b_tops, (20, 29, 38), pa,
        ):
            for c, pc, posc in placed_states(
                b, (25, 33), 32, 23, c_tops, (23,), pb,
            ):
                for d, pd, posd in direct_states(
                    c, (31, 46), 44, 29, d_tops, (17, 29, 38), pc,
                ):
                    try:
                        step(d, 5)
                    except Exception as exc:
                        print(
                            "COMMIT_EXCEPTION", type(exc).__name__, str(exc),
                            posa, posb, posc, posd, "PATH", pd + [5],
                        )
                        return
                    tested += 1
                    if d.levels_completed > base_level:
                        path = pd + [5]
                        print(
                            "WIN", posa, posb, posc, posd,
                            "TESTED", tested, "STEPS", SEARCH_STEPS,
                            "PATH", path,
                        )
                        return
    print("NONE", tested, "STEPS", SEARCH_STEPS)


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)

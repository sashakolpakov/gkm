"""Compact exhaustive acceptance map for level 2's three projections."""
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from players import play_level_1


def paced_step(env, action, clock):
    env.step(*action) if isinstance(action, tuple) else env.step(action)
    clock[0] += 1
    target = clock[0] / 280.0
    elapsed = time.monotonic() - clock[1]
    if target > elapsed:
        time.sleep(target - elapsed)


def go_left(env, count, clock):
    for _ in range(count):
        paced_step(env, 3, clock)


def probe(env):
    play_level_1(env)
    base_level = env.levels_completed
    clock = [0, time.monotonic()]
    a_positions = (12, 16, 20, 24, 28)
    b_positions = (28, 32, 36, 40, 44)
    c_positions = (20, 24, 28, 32, 36)
    wins = []
    for a_left in a_positions:
        a = env.clone()
        paced_step(a, (6, 14, 18), clock)
        go_left(a, 2, clock)
        for _ in range(a_left // 4):
            paced_step(a, 4, clock)

        b = a.clone()
        paced_step(b, (6, 34, 26), clock)
        go_left(b, 7, clock)
        for _ in range(b_positions[0] // 4):
            paced_step(b, 4, clock)
        b_left = b_positions[0]
        for bi, target_b in enumerate(b_positions):
            if bi:
                paced_step(b, 4, clock)
                b_left = target_b
            c = b.clone()
            paced_step(c, (6, 30, 38), clock)
            go_left(c, 5, clock)
            for _ in range(c_positions[0] // 4):
                paced_step(c, 4, clock)
            c_left = c_positions[0]
            for ci, target_c in enumerate(c_positions):
                if ci:
                    paced_step(c, 4, clock)
                    c_left = target_c
                test = c.clone()
                try:
                    paced_step(test, 5, clock)
                    completed = test.levels_completed
                except Exception as exc:
                    print("L2_ERROR", a_left, b_left, c_left, type(exc).__name__)
                    continue
                if completed > base_level:
                    wins.append((a_left, b_left, c_left))
    print(
        "L2_ACCEPTANCE",
        {
            "count": len(wins),
            "axes": tuple(
                tuple(sorted(set(x[i] for x in wins))) for i in range(3)
            ),
            "first": tuple(wins[:24]),
            "steps": clock[0],
        },
    )


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)

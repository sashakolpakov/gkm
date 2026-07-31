"""Test the smallest directed socket-chain family on level 6."""
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)


def replay(env, path):
    node = env.clone()
    for action in path:
        node.step(*action) if isinstance(action, tuple) else node.step(action)
    return node


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)

    tested = 0
    steps = 0
    started = time.monotonic()
    for c_top in (14, 17):
        for d_top in (20, 23):
            for d_left in (14, 17, 20):
                for a_top in (29, 32):
                    for a_left in (35, 38, 41):
                        for b_top in (23, 26, 29):
                            for b_left in (8, 11, 14, 17):
                                a_y = 17 if c_top != 17 else 20
                                path = (
                                    [(6, 25, 33)] + [1] * ((32 - c_top) // 3)
                                    + [(6, 33, 45)]
                                    + [1] * ((44 - d_top) // 3)
                                    + [3] * ((29 - d_left) // 3)
                                    + [(6, 30, a_y)]
                                    + [2] * ((a_top - 17) // 3)
                                    + [4] * ((a_left - 29) // 3)
                                    + [(6, 45, 18)]
                                    + [2] * ((b_top - 14) // 3)
                                    + [3] * ((44 - b_left) // 3)
                                    + [5]
                                )
                                result = replay(env, path)
                                tested += 1
                                steps += len(path)
                                if result.levels_completed > env.levels_completed:
                                    print(
                                        "BRIDGE_WIN",
                                        (c_top, d_top, d_left, a_top,
                                         a_left, b_top, b_left),
                                        "PATH", path, "TESTED", tested,
                                    )
                                    return
                                target = steps / 280.0
                                elapsed = time.monotonic() - started
                                if target > elapsed:
                                    time.sleep(target - elapsed)
    print("BRANCH_NONE", tested)


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)

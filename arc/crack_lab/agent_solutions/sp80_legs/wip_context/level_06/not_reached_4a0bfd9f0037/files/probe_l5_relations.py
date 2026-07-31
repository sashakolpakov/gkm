"""Joint row acceptance around the known level-5 turn connection."""
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4,
)


def moves(start, target, negative=1, positive=2):
    return ([negative] if target < start else [positive]) * (
        abs(target - start) // 3
    )


def replay(env, path):
    node = env.clone()
    for action in path:
        node.step(*action) if isinstance(action, tuple) else node.step(action)
    return node


def probe(env):
    for player in (play_level_1, play_level_2, play_level_3, play_level_4):
        player(env)
    arranged = replay(env, (
        [(6, 48, 33)] + [3] * 6
        + [(6, 21, 33)] + [2] * 3
        + [(6, 34, 21)] + [4] * 3 + [2] * 7
        + [(6, 36, 45)] + [3] * 6 + [1] * 2
    ))
    wins = []
    steps = 0
    started = time.monotonic()
    for l_left in range(5, 51, 3):
        for feeder_left in range(5, 51, 3):
            path = (
                [(6, 17, 37)] + moves(14, l_left, 3, 4)
                + [(6, 22, 42)] + moves(20, feeder_left, 3, 4)
                + [5]
            )
            result = replay(arranged, path)
            steps += len(path)
            if result.levels_completed > env.levels_completed:
                wins.append((l_left, feeder_left))
            target = steps / 280.0
            elapsed = time.monotonic() - started
            if target > elapsed:
                time.sleep(target - elapsed)
    print("L5_TURN_COLS", wins, "STEPS", steps)


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)

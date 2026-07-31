"""Joint level-5 column acceptance near the known winning arrangement."""
import itertools
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as Arena

from players import play_level_1, play_level_2, play_level_3, play_level_4


def moves(start, target):
    return [3 if target < start else 4] * (abs(target - start) // 3)


def probe(env):
    for player in (play_level_1, play_level_2, play_level_3, play_level_4):
        player(env)
    arranged = env.clone()
    winning = (
        [(6, 48, 33)] + [3] * 6
        + [(6, 21, 33)] + [2] * 3
        + [(6, 34, 21)] + [4] * 3 + [2] * 7
        + [(6, 36, 45)] + [3] * 6 + [1] * 2
    )
    for action in winning:
        arranged.step(*action) if isinstance(action, tuple) else arranged.step(action)

    wins = []
    steps = 0
    started = time.monotonic()
    for wide15, wide9, wide12, elbow in itertools.product(
        (17, 20, 23, 26, 29),
        (14, 17, 20, 23, 26),
        (32, 35, 38, 41, 44),
        (14, 17, 20),
    ):
        node = arranged.clone()
        path = (
            [(6, 30, 33)] + moves(23, wide15)
            + [(6, 22, 42)] + moves(20, wide9)
            + [(6, 42, 42)] + moves(38, wide12)
            + [(6, 17, 37)] + moves(14, elbow)
            + [5]
        )
        for action in path:
            node.step(*action) if isinstance(action, tuple) else node.step(action)
        steps += len(path)
        if node.levels_completed > env.levels_completed:
            wins.append((wide15, wide9, wide12, elbow))
        delay = steps / 280.0 - (time.monotonic() - started)
        if delay > 0:
            time.sleep(delay)
    print("L5_XMAP", "COUNT", len(wins), "WINS", wins, "STEPS", steps)


if __name__ == "__main__":
    levels, path, err = Arena.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)

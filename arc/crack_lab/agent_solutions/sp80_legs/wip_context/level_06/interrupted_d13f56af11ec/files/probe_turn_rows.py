"""Map the verified level-5 elbow/feeder row relation."""
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from players import play_level_1, play_level_2, play_level_3, play_level_4


def shift(start, target):
    return ([1] if target < start else [2]) * (abs(target - start) // 3)


def probe(env):
    for player in (play_level_1, play_level_2, play_level_3, play_level_4):
        player(env)
    arranged = env.clone()
    path = (
        [(6, 48, 33)] + [3] * 6
        + [(6, 21, 33)] + [2] * 3
        + [(6, 34, 21)] + [4] * 3 + [2] * 7
        + [(6, 36, 45)] + [3] * 6 + [1] * 2
    )
    for action in path:
        arranged.step(*action) if isinstance(action, tuple) else arranged.step(action)

    wins = []
    steps = 0
    started = time.monotonic()
    rows = range(14, 51, 3)
    for feeder_top in rows:
        for elbow_top in rows:
            node = arranged.clone()
            actions = (
                [(6, 22, 42)] + shift(41, feeder_top)
                + [(6, 15, 39)] + shift(35, elbow_top)
                + [5]
            )
            for action in actions:
                node.step(*action) if isinstance(action, tuple) else node.step(action)
            steps += len(actions)
            if node.levels_completed > env.levels_completed:
                wins.append((elbow_top, feeder_top))
            target = steps / 280.0
            delay = target - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
    print("L5_TURN_ROWS", wins, "STEPS", steps)


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)

"""Correct column relation for level 5's side elbow and its feeder bar."""
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as Arena

from players import play_level_1, play_level_2, play_level_3, play_level_4


def shift(start, target, negative, positive):
    return [negative if target < start else positive] * (
        abs(target - start) // 3
    )


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
    for elbow_left in range(5, 33, 3):
        for feeder_left in range(5, 45, 3):
            node = arranged.clone()
            actions = (
                [(6, 17, 37)] + shift(14, elbow_left, 3, 4)
                + [(6, 22, 42)] + shift(20, feeder_left, 3, 4)
                + [5]
            )
            for action in actions:
                node.step(*action) if isinstance(action, tuple) else node.step(action)
            steps += len(actions)
            if node.levels_completed > env.levels_completed:
                wins.append((elbow_left, feeder_left))
            delay = steps / 280.0 - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
    print("L5_TURN_COLUMNS", wins, "STEPS", steps)


if __name__ == "__main__":
    levels, path, err = Arena.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)

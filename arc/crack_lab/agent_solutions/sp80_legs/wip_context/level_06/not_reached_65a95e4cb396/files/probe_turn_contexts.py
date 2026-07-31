"""Map which energized level-5 bars can feed the side-facing turn."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from players import play_level_1, play_level_2, play_level_3, play_level_4
from probe_central_stack import moves


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

    rows = (14, 32, 38, 41, 44, 50)
    wins = []
    for left_top in rows:
        for right_top in rows:
            node = arranged.clone()
            actions = (
                [(6, 22, 42)] + moves(41, left_top, 1, 2)
                + [(6, 42, 42)] + moves(41, right_top, 1, 2)
                + [5]
            )
            for action in actions:
                node.step(*action) if isinstance(action, tuple) else node.step(action)
            if node.levels_completed > env.levels_completed:
                wins.append((left_top, right_top))
    print("L5_TURN_CONTEXTS", wins)


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)

"""Check whether USE acceptance depends on which piece is selected."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from players import play_level_1, play_level_2, play_level_3, play_level_4


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
    points = ((30, 33), (22, 42), (42, 42), (17, 37))
    results = []
    for point in points:
        node = arranged.clone()
        node.step(6, *point)
        node.step(5)
        results.append((point, node.levels_completed))
    print("L5_COMMIT_SELECTION", results)


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)

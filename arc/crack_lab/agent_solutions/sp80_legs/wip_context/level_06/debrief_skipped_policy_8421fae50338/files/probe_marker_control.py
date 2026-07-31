"""Control whether the marked bar may align a different level-4 port."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as Arena

from players import play_level_1, play_level_2, play_level_3


def replay(env, path):
    node = env.clone()
    for action in path:
        node.step(*action) if isinstance(action, tuple) else node.step(action)
    return node


def probe(env):
    for player in (play_level_1, play_level_2, play_level_3):
        player(env)
    base = (
        [(6, 24, 18)] + [4] * 3
        + [(6, 45, 18)] + [3] * 8
        + [(6, 18, 30)] + [4] * 8
        + [(6, 49, 33)] + [3] * 10
        + [(6, 43, 42)] + [3] * 7
    )
    cases = {
        "known": base + [5],
        # Move the marked 21-wide bar from left32 to left20 (marker 41->29)
        # and extend the 15-wide upper bar from left26 to left38.
        "marker_p2": (
            base
            + [(6, 36, 30)] + [3] * 4
            + [(6, 30, 18)] + [4] * 4
            + [5]
        ),
        "marker_off": base + [(6, 36, 30), 3, 5],
    }
    print("MARKER_CONTROL", {
        name: replay(env, path).levels_completed for name, path in cases.items()
    })


if __name__ == "__main__":
    levels, path, err = Arena.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)

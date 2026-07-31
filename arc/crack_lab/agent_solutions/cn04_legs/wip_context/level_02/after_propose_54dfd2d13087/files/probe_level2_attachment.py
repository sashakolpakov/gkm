"""Search for a level-2 state where a colored piece has truly translated."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
from players import play_level_1


COLORS = (9, 11, 14)


def full_bboxes(frame):
    out = {}
    for color in COLORS:
        blobs = perception.connected_components(frame, colors=(color,), min_area=4)
        if blobs:
            best = max(blobs, key=lambda b: b.area)
            out[color] = (best.bbox, best.area)
    return out


def probe(env):
    play_level_1(env)
    initial = full_bboxes(env.frame())

    def moved(node, _):
        now = full_bboxes(node.frame())
        return any(
            color in now
            and now[color][1] == initial[color][1]
            and now[color][0] != initial[color][0]
            for color in COLORS
        )

    path = perception.bounded_bfs(
        env,
        moved,
        actions=(1, 2, 3, 4, 5),
        key_fn=lambda node: perception.arr(node.frame())[1:].tobytes(),
        max_states=50000,
        max_depth=110,
    )
    print("attachment_path", path)
    if path is not None:
        node = perception.replay(env, path)
        print("initial", initial, "after", full_bboxes(node.frame()))


arena.run_program("cn04", probe)

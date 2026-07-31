import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


def state(env):
    return tuple(
        (b.color, b.bbox, b.area)
        for b in connected_components(
            arr(env.frame())[:63], colors=(0, 5, 11, 12, 13, 14), min_area=2
        )
    )


def apply(node, path):
    for item in path:
        node.step(*item) if isinstance(item, tuple) else node.step(item)


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    node = env.clone()
    segments = (
        ("v_above_agent", [3] * 3),
        ("agent_lifts_v", [(6, 35, 52), 4, 3, 4]),
        ("stage_h_above_agent", [3] + [1] * 2 + [4] * 3),
        ("clear_v", [(6, 46, 29)] + [1] * 2),
        ("h_left_of_gap", [(6, 44, 34)] + [1] + [3] * 8),
        ("v_at_gap", [(6, 46, 23)] + [2] * 2 + [3] * 7),
        ("v_push_left_1", [3]),
        ("v_push_left_2", [3]),
        ("v_push_left_3", [3]),
        ("v_push_left_4", [3]),
        ("v_push_left_5", [3]),
        ("v_push_left_6", [3]),
    )
    for name, path in segments:
        apply(node, path)
        print(name, state(node))


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)

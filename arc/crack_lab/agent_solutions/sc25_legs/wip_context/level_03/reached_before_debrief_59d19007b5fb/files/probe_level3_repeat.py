import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import perception as P
from players import play_level_1, play_level_2


POINTS = ((30, 50), (30, 55), (30, 60))


def pieces(frame):
    return tuple(
        (b.color, b.area, b.bbox)
        for b in P.connected_components(frame, colors=(4, 6, 13), min_area=4)
        if b.bbox[0] < 45
    )


def panel(frame):
    return tuple(int(frame[y][x]) for x, y in POINTS)


def probe(env):
    play_level_1(env)
    play_level_2(env)
    node = env.clone()
    print("start", panel(node.frame()), pieces(node.frame()))
    for round_no in range(1, 11):
        for x, y in POINTS:
            node.step(6, x, y)
        gate_test = P.replay(node, [2, 2, 3, 3, 2])
        print(
            "round", round_no,
            "panel", panel(node.frame()),
            "pieces", pieces(node.frame()),
            "gate_avatar_colors", P.color_counts(gate_test.frame()),
            "gate_level", gate_test.levels_completed,
            "terminal", node.terminal(),
        )
        if node.terminal():
            break


levels, path, err = A.run_program("sc25", probe)
print("run", levels, len(path), err)

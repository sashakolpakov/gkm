import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, connected_components


BEST = (
    [3] * 5 + [2] + [3] * 4
    + [(6, 36, 52)] + [1] * 4 + [3]
    + [4] * 3 + [1] * 3 + [3] * 5
)


def summary(env):
    frame = arr(env.frame())[:63]
    return tuple(
        (b.color, b.bbox, b.area)
        for b in connected_components(
            frame, colors=(0, 4, 5, 11, 13, 14), min_area=2
        )
    )


def grid(env):
    frame = arr(env.frame())[:63, :63]
    symbols = {0: "s", 1: ".", 2: "#", 4: "t", 5: "s",
               11: "G", 12: "a", 13: "A", 14: "o", 15: "|"}
    lines = []
    for row in range(0, 63, 3):
        line = []
        for col in range(0, 63, 3):
            values, counts = __import__("numpy").unique(
                frame[row:row + 3, col:col + 3], return_counts=True
            )
            color = int(values[counts.argmax()])
            line.append(symbols[color])
        lines.append("".join(line))
    return "\n".join(lines)


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    print("map\n" + grid(env))
    node = env.clone()
    for action in BEST:
        node.step(*action) if isinstance(action, tuple) else node.step(action)
    print("best", node.levels_completed, summary(node))
    wait = node.clone()
    waits = []
    for count in range(1, 13):
        wait.step(3)
        waits.append((count, wait.levels_completed, summary(wait)[6:8]))
    print("waits", waits)
    for delay in range(6):
        child = node.clone()
        for _ in range(delay):
            child.step(3)
        child.step(4)
        child.step(3)
        print("reseat_h", delay, child.levels_completed)
    for delay in range(6):
        child = node.clone()
        for _ in range(delay):
            child.step(3)
        child.step(6, 28, 47)
        child.step(4)
        child.step(3)
        print("reseat_v", delay, child.levels_completed)
    for name, point in (
        ("large_piece", (13, 13)),
        ("large_target", (13, 40)),
        ("agent_top", (47, 9)),
        ("agent_bottom", (45, 50)),
    ):
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(6, *point)
            child.step(action)
            print("context", name, action, child.levels_completed, summary(child))
    for action in (1, 2, 3, 4):
        child = node.clone()
        child.step(action)
        print("next", action, child.levels_completed, summary(child))
    other = node.clone()
    other.step(6, 28, 47)
    for action in (1, 2, 3, 4):
        child = other.clone()
        child.step(action)
        print("other", action, child.levels_completed, summary(child))


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)

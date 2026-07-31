import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import perception as P
import players


def objects(node, min_area=2):
    return [
        (o["color"], o["bbox"], o["area"])
        for o in P.object_candidates(node.frame(), min_area=min_area)
        if o["color"] != 5
    ]


def block_map(node):
    chars = {5: ".", 6: "L", 7: "R", 8: "#", 9: "s",
             10: "A", 11: "s", 12: "C", 14: "D"}
    for r in range(16):
        row = []
        for c in range(16):
            sig = P.block_signatures(node.frame())[(r, c)]
            colors = [color for color in sig if color != 5]
            row.append(chars.get(colors[-1], "?") if colors else ".")
        print("".join(row))


def run(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)

    print("STATE", env.levels_completed, env.actions, P.color_counts(env.frame()))
    print("OBJECTS", objects(env))
    print("MAP")
    block_map(env)
    frame = P.arr(env.frame())
    print("CENTER", [tuple(int(v) for v in row) for row in frame[40:48, 28:36]])

    base = P.arr(env.frame()).copy()
    for action in env.actions:
        node = env.clone()
        node.step(action)
        delta = P.frame_delta(base, node.frame())
        print("KEY", action, (delta["count"], delta["bbox"]), objects(node))

    for color, bbox, area in objects(env):
        if area not in (4, 16):
            continue
        r0, c0, r1, c1 = bbox
        x, y = (c0 + c1) // 2, (r0 + r1) // 2
        selected = env.clone()
        before = P.arr(selected.frame()).copy()
        selected.step(6, x, y)
        click = P.frame_delta(before, selected.frame())
        moves = []
        for action in (1, 2, 3, 4, 5):
            moved = selected.clone()
            pre = P.arr(moved.frame()).copy()
            moved.step(action)
            delta = P.frame_delta(pre, moved.frame())
            moves.append((action, delta["count"], delta["bbox"], objects(moved)))
        print("SELECT", (color, bbox, area), (click["count"], click["bbox"]), moves)


A.run_program("m0r0", run)

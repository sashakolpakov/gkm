import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, color_counts, connected_components, frame_delta
def mixed_objects(frame):
    grid = arr(frame)
    background = max(color_counts(frame), key=color_counts(frame).get)
    occupied = grid != background
    seen = set()
    out = []
    for r in range(64):
        for c in range(64):
            if not occupied[r, c] or (r, c) in seen:
                continue
            todo = [(r, c)]
            seen.add((r, c))
            cells = []
            while todo:
                y, x = todo.pop()
                cells.append((y, x))
                for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    if (0 <= yy < 64 and 0 <= xx < 64
                            and occupied[yy, xx] and (yy, xx) not in seen):
                        seen.add((yy, xx))
                        todo.append((yy, xx))
            out.append((
                (min(y for y, _ in cells), min(x for _, x in cells),
                 max(y for y, _ in cells), max(x for _, x in cells)),
                len(cells), tuple(sorted({int(grid[y, x]) for y, x in cells})),
            ))
    return sorted(out)


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    base = env.frame()
    print("level", env.levels_completed, "actions", env.actions)
    print("counts", color_counts(base))
    advanced = env.clone()
    advanced.step(6, 0, 0)
    print("after_noop", advanced.levels_completed,
          color_counts(advanced.frame()), frame_delta(base, advanced.frame()))
    print("objects", mixed_objects(base))
    blobs = connected_components(base, min_area=2)
    print("components", [
        (b.color, b.bbox, b.area, tuple(round(v, 1) for v in b.centroid))
        for b in blobs
    ])
    print("active_components")
    for b in blobs:
        x, y = int(round(b.centroid[1])), int(round(b.centroid[0]))
        node = env.clone()
        before = node.frame()
        node.step(6, x, y)
        delta = frame_delta(before, node.frame())
        if delta["count"] > 1:
            print((b.color, (x, y), b.bbox), delta["count"] - 1,
                  delta["bbox"], "level", node.levels_completed)
    controls = (
        ("A<", (4, 51)), ("A>", (10, 51)), ("B", (17, 51)),
        ("C<", (26, 51)), ("C>", (32, 51)), ("D", (39, 51)),
        ("E<", (54, 51)), ("E>", (60, 51)),
        ("F<", (4, 58)), ("F>", (10, 58)), ("G", (17, 58)),
        ("H<", (26, 58)), ("H>", (32, 58)), ("I", (60, 58)),
    )
    print("trajectories")
    target = {(15, 25), (16, 24), (16, 26), (17, 25)}
    for name, point in controls:
        node = env.clone()
        states = []
        for step in range(1, 9):
            before = node.frame()
            node.step(6, *point)
            grid = arr(node.frame())
            marker = tuple(
                (int(r), int(c)) for r, c in zip(*((grid == 13).nonzero()))
                if (int(r), int(c)) not in target
            )
            delta = frame_delta(before, node.frame())
            states.append((step, delta["count"], marker))
            if delta["count"] == 0 or node.levels_completed > 6:
                break
        print(name, states)
    lookup = dict(controls)
    sequences = (
        ("down_cross", ("C>",) * 5 + ("H>",) * 12),
        ("down_cross_f", ("C>",) * 5 + ("F<",) * 2 + ("H>",) * 12),
        ("deep_cross", ("C>",) * 7 + ("H>",) * 12),
        ("corner_route", ("C>",) * 5 + ("H>",) * 8
         + ("C<",) * 5 + ("H>",) * 2),
    )
    print("sequences")
    for label, sequence in sequences:
        node = env.clone()
        states = []
        for name in sequence:
            before = node.frame()
            node.step(6, *lookup[name])
            grid = arr(node.frame())
            moving = tuple(
                (int(r), int(c)) for r, c in zip(*((grid == 13).nonzero()))
                if (int(r), int(c)) not in target
            )
            states.append((name, frame_delta(before, node.frame())["count"],
                           moving))
        print(label, states, "level", node.levels_completed)
    corner = env.clone()
    for name in ("C>",) * 5 + ("H>",) * 8:
        corner.step(6, *lookup[name])
    print("corner_controls")
    for name, point in controls:
        node = corner.clone()
        states = []
        for _ in range(4):
            before = node.frame()
            node.step(6, *point)
            grid = arr(node.frame())
            moving = tuple(
                (int(r), int(c)) for r, c in zip(*((grid == 13).nonzero()))
                if (int(r), int(c)) not in target
            )
            bodies = tuple(
                (b.color, b.bbox) for b in connected_components(
                    node.frame(), colors={9, 11, 12, 14}, min_area=4
                ) if b.bbox[0] < 42
            )
            delta = frame_delta(before, node.frame())["count"]
            states.append((delta, moving, bodies))
            if delta == 0:
                break
        print(name, states)


arena.run_program("s5i5", probe)

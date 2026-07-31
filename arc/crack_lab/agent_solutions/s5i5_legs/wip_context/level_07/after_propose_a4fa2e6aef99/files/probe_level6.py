import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, color_counts, connected_components, frame_delta
from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)


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
                    if (0 <= yy < 64 and 0 <= xx < 64 and occupied[yy, xx]
                            and (yy, xx) not in seen):
                        seen.add((yy, xx))
                        todo.append((yy, xx))
            out.append(((min(y for y, _ in cells), min(x for _, x in cells),
                         max(y for y, _ in cells), max(x for _, x in cells)),
                        len(cells), tuple(sorted({int(grid[y, x]) for y, x in cells}))))
    return sorted(out)


def probe(env):
    for play in (play_level_1, play_level_2, play_level_3, play_level_4, play_level_5):
        play(env)
    print("level", env.levels_completed, "actions", env.actions)
    print("counts", color_counts(env.frame()))
    print("objects", mixed_objects(env.frame()))
    print("components", [
        (b.color, b.bbox, b.area, tuple(round(v, 1) for v in b.centroid))
        for b in connected_components(env.frame(), min_area=3)
    ])

    base = env.frame()
    print("candidate_clicks")
    for b in connected_components(base, min_area=3):
        x, y = int(round(b.centroid[1])), int(round(b.centroid[0]))
        node = env.clone()
        before = node.frame()
        node.step(6, x, y)
        delta = frame_delta(before, node.frame())
        if delta["count"] > 1:
            print((b.color, (x, y), b.bbox), delta["count"] - 1,
                  delta["bbox"], "level", node.levels_completed)

    controls = (
        ("A", (12, 48)), ("B", (31, 48)), ("C", (50, 48)),
        ("A<", (9, 57)), ("A>", (15, 57)),
        ("B<", (28, 57)), ("B>", (34, 57)),
        ("C<", (47, 57)), ("C>", (53, 57)),
    )
    print("trajectories")
    for name, point in controls:
        node = env.clone()
        states = []
        for step in range(1, 9):
            before = node.frame()
            node.step(6, *point)
            delta = frame_delta(before, node.frame())
            grid = arr(node.frame())
            attached = tuple(
                (int(r), int(c)) for r, c in zip(*((grid[:27] == 13).nonzero()))
            )
            bodies = tuple(
                (b.color, b.bbox) for b in connected_components(
                    node.frame(), colors={9, 11, 14, 15}, min_area=10
                ) if b.bbox[0] < 45
            )
            states.append((step, delta["count"] - 1, attached, bodies))
            if delta["count"] <= 1 or node.levels_completed > 5:
                break
        print(name, states)


arena.run_program("s5i5", probe)

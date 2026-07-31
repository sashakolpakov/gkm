import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, color_counts, connected_components, frame_delta
from players import play_level_1, play_level_2, play_level_3, play_level_4


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
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (0 <= yy < 64 and 0 <= xx < 64 and occupied[yy, xx]
                            and (yy, xx) not in seen):
                        seen.add((yy, xx))
                        todo.append((yy, xx))
            out.append((
                (min(y for y, _ in cells), min(x for _, x in cells),
                 max(y for y, _ in cells), max(x for _, x in cells)),
                len(cells), tuple(sorted({int(grid[y, x]) for y, x in cells})),
            ))
    return sorted(out)


def probe(env):
    play_level_1(env)
    play_level_2(env)
    play_level_3(env)
    play_level_4(env)
    print("level", env.levels_completed, "actions", env.actions)
    print("counts", color_counts(env.frame()))
    print("objects", mixed_objects(env.frame()))
    print("components", [
        (b.color, b.bbox, b.area, tuple(round(v, 1) for v in b.centroid))
        for b in connected_components(env.frame(), min_area=3)
    ])

    base = env.frame()
    candidates = connected_components(base, min_area=5)
    print("candidate_clicks")
    for b in candidates:
        x, y = int(round(b.centroid[1])), int(round(b.centroid[0]))
        node = env.clone()
        before = node.frame()
        node.step(6, x, y)
        delta = frame_delta(before, node.frame())
        if delta["count"]:
            print((b.color, (x, y), b.bbox), delta["count"], delta["bbox"],
                  "level", node.levels_completed)

    controls = [
        ("vertical", ((6, 57), (12, 57))),
        ("upper", ((21, 57), (27, 57))),
        ("joint", ((36, 57), (42, 57))),
        ("marker", ((51, 57), (57, 57))),
    ]
    print("trajectories")
    for name, pair in controls:
        for side, point in enumerate(pair):
            node = env.clone()
            states = []
            for step in range(1, 13):
                before = node.frame()
                node.step(6, *point)
                delta = frame_delta(before, node.frame())
                if delta["count"] == 1:
                    break
                grid = arr(node.frame())
                markers = tuple(
                    (int(r), int(c)) for r, c in zip(*((grid == 13).nonzero()))
                )
                blobs = tuple(
                    (b.color, b.bbox)
                    for b in connected_components(node.frame(), colors={1, 9, 10, 12, 14, 15},
                                                  min_area=5)
                )
                states.append((step, delta["count"] - 1, markers, blobs,
                               node.levels_completed))
                if node.levels_completed > 4:
                    break
            print(name, side, states)


arena.run_program("s5i5", probe)

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, color_counts, connected_components, frame_delta
from players import play_level_1, play_level_2, play_level_3


def summary(frame):
    blobs = connected_components(frame, min_area=3)
    return [
        (b.color, b.bbox, b.area, tuple(round(v, 1) for v in b.centroid))
        for b in blobs
    ]


def upper_objects(frame):
    grid = arr(frame)
    occupied = (grid[:45] != 5)
    seen = set()
    out = []
    for r in range(45):
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
                    if (0 <= yy < 45 and 0 <= xx < 64 and occupied[yy, xx]
                            and (yy, xx) not in seen):
                        seen.add((yy, xx))
                        todo.append((yy, xx))
            colors = tuple(sorted({int(grid[y, x]) for y, x in cells}))
            out.append((min(y for y, _ in cells), min(x for _, x in cells),
                        max(y for y, _ in cells), max(x for _, x in cells),
                        len(cells), colors))
    return sorted(out)


def goals(frame):
    grid = arr(frame)
    return [tuple(map(int, point)) for point in zip(*((grid[:45] == 13).nonzero()))]


def probe(env):
    play_level_1(env)
    play_level_2(env)
    play_level_3(env)
    print("level", env.levels_completed, "actions", env.actions)
    print("counts", color_counts(env.frame()))
    print("upper_objects", upper_objects(env.frame()))
    print("goal_cells", goals(env.frame()))
    print("blobs")
    for item in summary(env.frame()):
        print(item)

    # Probe the center of every sizable lower-panel component independently.
    base = env.frame()
    controls = [
        b for b in connected_components(base, min_area=5) if b.centroid[0] >= 40
    ]
    print("control_deltas")
    for b in controls:
        node = env.clone()
        x, y = int(round(b.centroid[1])), int(round(b.centroid[0]))
        before = node.frame()
        node.step(6, x, y)
        d = frame_delta(before, node.frame())
        print((b.color, (x, y), b.bbox), d["count"], d["bbox"],
              "level", node.levels_completed)

    paired = [
        ("orange", ((6, 48), (12, 48))),
        ("magenta", ((51, 48), (57, 48))),
        ("center", ((28, 54), (34, 54))),
        ("brown", ((6, 57), (12, 57))),
        ("blue", ((51, 57), (57, 57))),
    ]
    print("trajectories")
    for name, pair in paired:
        for side, point in enumerate(pair):
            node = env.clone()
            states = []
            for step in range(1, 13):
                before = node.frame()
                node.step(6, *point)
                d = frame_delta(before, node.frame())
                spatial = d["count"] - 1  # bottom-right move counter
                if spatial or step == 1:
                    states.append((step, spatial, upper_objects(node.frame()),
                                   goals(node.frame()), node.levels_completed))
                if node.levels_completed > 3:
                    break
            print(name, side, states)


arena.run_program("s5i5", probe)

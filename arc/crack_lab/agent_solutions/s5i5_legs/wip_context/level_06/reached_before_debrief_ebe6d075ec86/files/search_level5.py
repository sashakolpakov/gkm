import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr
from players import play_level_1, play_level_2, play_level_3, play_level_4


CONTROLS = (
    ("A<", (6, 6, 57)),
    ("A>", (6, 12, 57)),
    ("B^", (6, 21, 57)),
    ("Bv", (6, 27, 57)),
    ("C-", (6, 36, 57)),
    ("C+", (6, 42, 57)),
    ("D<", (6, 51, 57)),
    ("D>", (6, 57, 57)),
)


def key(env):
    # Ignore the control panel and replay counter.
    return arr(env.frame())[:54].tobytes()


def search(root, max_states=50000, max_depth=64):
    queue = deque([(root.clone(), ())])
    seen = {key(root)}
    while queue and len(seen) < max_states:
        node, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for index, (_, action) in enumerate(CONTROLS):
            child = node.clone()
            child.step(*action)
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            child_path = path + (index,)
            if child.levels_completed > 4:
                return child_path, len(seen)
            queue.append((child, child_path))
    return None, len(seen)


def marker_objects(frame):
    grid = arr(frame)
    occupied = grid != 5
    seen = set()
    out = []
    for r in range(54):
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
                    if (0 <= yy < 54 and 0 <= xx < 64 and occupied[yy, xx]
                            and (yy, xx) not in seen):
                        seen.add((yy, xx))
                        todo.append((yy, xx))
            if any(grid[y, x] == 13 for y, x in cells) and len(cells) > 1:
                out.append((
                    (min(y for y, _ in cells), min(x for _, x in cells),
                     max(y for y, _ in cells), max(x for _, x in cells)),
                    len(cells), tuple(sorted({int(grid[y, x]) for y, x in cells})),
                ))
    return sorted(out)


def run(env):
    play_level_1(env)
    play_level_2(env)
    play_level_3(env)
    play_level_4(env)
    path, states = search(env)
    print("states", states)
    print("path", None if path is None else [CONTROLS[i][0] for i in path])
    if path is not None:
        node = env.clone()
        previous = None
        for at, index in enumerate(path + (-1,)):
            if index != previous:
                print("stage", at, "after", None if previous is None else CONTROLS[previous][0],
                      marker_objects(node.frame()), "level", node.levels_completed)
            if index >= 0:
                node.step(*CONTROLS[index][1])
            previous = index


arena.run_program("s5i5", run)

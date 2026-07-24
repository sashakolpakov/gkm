import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import arr, object_candidates


BUTTONS = ((20, 17), (14, 26), (14, 35), (39, 17), (48, 26), (48, 35))


def token_positions(env):
    items = []
    for obj in object_candidates(env.frame(), min_area=4):
        if obj["size"] == (2, 2) and obj["area"] == 4 and obj["bbox"][0] >= 16:
            items.append((obj["bbox"][0], obj["bbox"][1]))
    return tuple(items)


def token_state(env, positions):
    frame = arr(env.frame())
    items = []
    for r, c in positions:
        items.append((r, c, int(frame[r, c])))
    return tuple(items)


def cluster_score(state):
    cells = {(r, c): color for r, c, color in state}
    return sum(cells[(r, c)] == cells.get((r + dr, c + dc))
               for r, c in cells for dr, dc in ((0, 3), (3, 0)))


def probe(env):
    for _ in range(5):
        env.step(6, 5, 32)
    base_level = env.levels_completed
    positions = token_positions(env)
    start = token_state(env, positions)
    queue = deque([(env.clone(), ())])
    seen = {start}
    best = (cluster_score(start), ())
    expanded = 0
    while queue and len(seen) < 50000:
        node, path = queue.popleft()
        if len(path) >= 7:
            continue
        for button_index, (x, y) in enumerate(BUTTONS):
            child = node.clone()
            child.step(6, x, y)
            new_path = path + (button_index,)
            expanded += 1
            if child.levels_completed > base_level:
                print("FOUND", new_path, "expanded", expanded, "seen", len(seen),
                      "score", cluster_score(token_state(child, positions)))
                return
            state = token_state(child, positions)
            if state in seen:
                continue
            seen.add(state)
            score = cluster_score(state)
            if score > best[0]:
                best = (score, new_path)
                print("BEST", best, "seen", len(seen))
            queue.append((child, new_path))
    print("STOP", "expanded", expanded, "seen", len(seen), "best", best)


A.run_program("lp85", probe)

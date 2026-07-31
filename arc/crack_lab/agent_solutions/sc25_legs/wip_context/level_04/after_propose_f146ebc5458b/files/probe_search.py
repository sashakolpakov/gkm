import heapq
import itertools
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import perception as P
import players


MACROS = {
    "U": (1,),
    "D": (2,),
    "L": (3,),
    "R": (4,),
    "X": ((6, 30, 50), (6, 25, 55), (6, 35, 55), (6, 30, 60)),
    "V": ((6, 30, 50), (6, 30, 55), (6, 30, 60)),
    "H": ((6, 25, 55), (6, 30, 55), (6, 35, 55)),
}


def world_key(node):
    moving = tuple(
        (b.color, b.bbox, b.area)
        for b in P.connected_components(
            node.frame(), colors=(9, 10), min_area=1
        )
        if b.bbox[1] < 47
    )
    mechanisms = tuple(
        (b.color, b.bbox, b.area)
        for b in P.connected_components(
            node.frame(), colors=(6, 13, 14), min_area=1
        )
        if (b.color in (6, 13) and b.bbox[0] > 20)
        or (b.color == 14 and b.bbox[1] < 60)
    )
    frame = node.frame()
    panel = tuple(
        int(frame[y][x])
        for y in (50, 55, 60)
        for x in (25, 30, 35)
    )
    return (node.levels_completed, moving, mechanisms, panel)


def apply(node, actions):
    for action in actions:
        if node.terminal():
            break
        node.step(*action) if isinstance(action, tuple) else node.step(action)


def search(root, max_states=12000, max_cost=45):
    serial = itertools.count()
    queue = [(0, next(serial), ())]
    best = {world_key(root): 0}
    expanded = 0

    def reconstruct(path):
        node = root.clone()
        for name in path:
            apply(node, MACROS[name])
        return node

    while queue and expanded < max_states:
        cost, _, path = heapq.heappop(queue)
        node = reconstruct(path)
        if cost != best.get(world_key(node)):
            continue
        if node.levels_completed > root.levels_completed:
            return cost, path, expanded
        expanded += 1
        for name, actions in MACROS.items():
            new_cost = cost + len(actions)
            if new_cost > max_cost:
                continue
            child_path = path + (name,)
            child = reconstruct(child_path)
            if child.terminal() and child.levels_completed == root.levels_completed:
                continue
            key = world_key(child)
            if new_cost >= best.get(key, 10**9):
                continue
            best[key] = new_cost
            heapq.heappush(
                queue, (new_cost, next(serial), child_path)
            )
    return None, None, expanded


def probe(env):
    for k in (1, 2, 3):
        getattr(players, f"play_level_{k}")(env)
    print("SEARCH", search(env.clone(), max_states=6000, max_cost=45))


levels, path, err = A.run_program("sc25", probe)
print("DONE", levels, len(path), err)

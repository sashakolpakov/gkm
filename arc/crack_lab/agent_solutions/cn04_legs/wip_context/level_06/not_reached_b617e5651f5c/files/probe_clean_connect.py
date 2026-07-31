import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import numpy as np
import gkm_arena as arena
from scipy import ndimage

import players


def occupied_mask(frame):
    array = np.asarray(frame)
    background = int(np.bincount(array.ravel()).argmax())
    return array != background


def component_count(frame):
    return int(ndimage.label(occupied_mask(frame))[1])


def selection_roots(work):
    roots = {np.asarray(work.frame()).tobytes(): (None, work.clone())}
    frame = np.asarray(work.frame())
    for row in range(1, 64, 3):
        for col in range(1, 64, 3):
            if int(frame[row, col]) in (0, 9):
                continue
            child = work.clone()
            child.step(6, col, row)
            key = np.asarray(child.frame()).tobytes()
            roots.setdefault(key, ((col, row), child))
    return list(roots.values())


def shortest_clean_merge(root, occupied, components, max_states=6000, max_depth=32):
    queue = deque([(root.clone(), [])])
    seen = {np.asarray(root.frame()).tobytes()}
    while queue and len(seen) <= max_states:
        node, path = queue.popleft()
        if node.levels_completed >= 6 or component_count(node.frame()) < components:
            return path, node
        if len(path) >= max_depth:
            continue
        for action in range(1, 6):
            child = node.clone()
            child.step(action)
            frame = child.frame()
            if int(occupied_mask(frame).sum()) != occupied:
                continue
            key = np.asarray(frame).tobytes()
            if key in seen:
                continue
            seen.add(key)
            queue.append((child, path + [action]))
    return None, None


def probe(env):
    while env.levels_completed < 5:
        getattr(players, f"play_level_{env.levels_completed + 1}")(env)
    work = env.clone()
    occupied = int(occupied_mask(work.frame()).sum())
    whole_path = []
    for stage in range(5):
        components = component_count(work.frame())
        print("STAGE", stage, "components", components,
              "occupied", int(occupied_mask(work.frame()).sum()), flush=True)
        if work.levels_completed >= 6:
            print("FOUND", whole_path)
            return
        progress = None
        for click, root in selection_roots(work):
            path, result = shortest_clean_merge(root, occupied, components)
            print(" TRY", click, "PATH", path, flush=True)
            if path is not None:
                prefix = [] if click is None else [(6, *click)]
                progress = prefix + path
                work = result
                break
        if progress is None:
            print("BLOCKED", whole_path)
            return
        whole_path += progress


arena.run_program("cn04", probe)

import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import numpy as np
import gkm_arena as arena
from scipy import ndimage

import players
from perception import bounded_bfs


def occupied_components(frame):
    array = np.asarray(frame)
    values, counts = np.unique(array, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    mask = (array != background) & (array != 0)
    components = []
    labels, count = ndimage.label(mask)
    for label in range(1, count + 1):
        components.append(list(zip(*np.where(labels == label))))
    return sorted(components, key=len, reverse=True)


def component_samples(frame):
    array = np.asarray(frame)
    out = []
    for points in occupied_components(frame):
        black = [(r, c) for r, c in points if int(array[r, c]) == 4]
        if black:
            r, c = black[len(black) // 2]
            out.append((c, r))
    return out


def probe(env):
    while env.levels_completed < 5:
        getattr(players, f"play_level_{env.levels_completed + 1}")(env)
    work = env.clone()
    whole_path = []
    for stage in range(8):
        base_level = work.levels_completed
        count = len(occupied_components(work.frame()))
        print("stage", stage, "components", count,
              "sizes", [len(c) for c in occupied_components(work.frame())],
              flush=True)
        if base_level >= 6:
            print("FOUND", whole_path)
            return

        choices = [None] if stage == 0 else component_samples(work.frame())
        progress = None
        for click in choices:
            selected = work.clone()
            prefix = []
            if click is not None:
                selected.step(6, *click)
                prefix = [[6, *click]]
            path = bounded_bfs(
                selected,
                lambda node, _: (
                    node.levels_completed > base_level
                    or len(occupied_components(node.frame())) < count
                ),
                max_states=5000,
                max_depth=32,
            )
            print(" choice", click, "path", path, flush=True)
            if path is not None:
                progress = prefix + path
                break
        if progress is None:
            print("BLOCKED", whole_path)
            return
        for action in progress:
            if isinstance(action, list):
                work.step(*action)
            else:
                work.step(action)
        whole_path += progress


arena.run_program("cn04", probe)

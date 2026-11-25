"""Reusable skills learned from public frame/action experiments."""
from heapq import heappop, heappush
from itertools import count
import time

import numpy as np
from perception import arr, safe_step


def navigate_switchable_platforms(env, switches, avatar_color=14,
                                  target_color=11, stride=2,
                                  max_states=1200, max_depth=48):
    """Plan on clones, then cross walkways whose buttons change connectivity.

    Direction keys move the avatar along colored platforms. Buttons may move
    bridges, so avatar position alone is insufficient to identify a state.
    Distance to the destination supplies dense progress between sparse rewards.
    The bottom frame row is the observed move meter, excluded from state keys.
    """
    base_level = env.levels_completed
    initial = arr(env.frame())
    target = np.argwhere(initial == target_color).min(axis=0)
    actions = tuple(a for a in env.actions if a != 6) + tuple(
        (6, int(x), int(y)) for x, y in switches)

    def key(node):
        return arr(node.frame())[:-1].tobytes()

    def distance(node):
        cells = np.argwhere(arr(node.frame()) == avatar_color)
        if not len(cells):
            return 0
        position = cells.min(axis=0)
        return int(np.abs(position - target).sum()) / stride

    serial = count()
    start = env.clone()
    queue = [(distance(start), 0, next(serial), start, ())]
    costs = {key(start): 0}
    plan = None
    expanded = 0
    while queue and expanded < max_states:
        _, depth, _, node, path = heappop(queue)
        if depth != costs.get(key(node)):
            continue
        if depth >= max_depth or node.terminal():
            continue
        expanded += 1
        for action in actions:
            child = node.clone()
            safe_step(child, action)
            time.sleep(0.004)  # Keep foreground clone probing below ~300 steps/s.
            candidate = path + (action,)
            if child.levels_completed > base_level:
                plan = candidate
                break
            child_key = key(child)
            cost = depth + 1
            if cost >= costs.get(child_key, float('inf')):
                continue
            costs[child_key] = cost
            heappush(queue, (cost + distance(child), cost, next(serial),
                             child, candidate))
        if plan is not None:
            break
    if plan is None:
        raise RuntimeError('No verified platform route within clone search bounds')
    for action in plan:
        if env.terminal() or env.levels_completed > base_level:
            break
        safe_step(env, action)

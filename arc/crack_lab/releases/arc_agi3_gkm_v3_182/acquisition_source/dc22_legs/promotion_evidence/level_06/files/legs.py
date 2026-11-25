"""Reusable skills learned from public frame/action experiments."""
from heapq import heappop, heappush
from itertools import count
import time

import numpy as np
from perception import arr, safe_step


def manipulate_platform_controls(env, controls, verify_each=True):
    """Grab and drag platforms, or set cyclic bridges, through visible buttons.

    Each (x, y, presses) describes one control operation. Verify the whole
    operation on a clone first. Normally each press must change the frame.
    Set verify_each=False for controls whose hidden selection is only revealed
    by a later action; then the complete operation must change the frame.
    Occupied platforms may block drags; stage the walker on a fixed island
    before moving them.
    """
    plan = []
    probe = env.clone()
    initial = arr(probe.frame())[:-1].copy()
    for x, y, presses in controls:
        for _ in range(presses):
            before = arr(probe.frame())[:-1].copy()
            action = (6, x, y)
            safe_step(probe, action)
            time.sleep(0.004)
            if verify_each and np.array_equal(before, arr(probe.frame())[:-1]):
                raise RuntimeError('Platform control blocked or unavailable')
            plan.append(action)
    if not verify_each and np.array_equal(initial, arr(probe.frame())[:-1]):
        raise RuntimeError('Platform operation has no visible outcome')
    for action in plan:
        safe_step(env, action)


def navigate_switchable_platforms(env, switches, avatar_color=14,
                                  target_color=11, stride=2,
                                  max_states=1200, max_depth=48, waypoint=None,
                                  board_width=64):
    """Plan on clones, then cross walkways whose buttons change connectivity.

    Direction keys move the avatar along colored platforms. Buttons may move
    bridges, so avatar position alone is insufficient to identify a state.
    Distance to the destination supplies dense progress between sparse rewards.
    An optional (row, column) waypoint stages required pickups before the goal.
    The bottom frame row is the observed move meter, excluded from state keys.
    board_width excludes sidebar glyphs that reuse the avatar color when
    measuring waypoint distance; state keys retain the complete visible frame.
    """
    base_level = env.levels_completed
    initial = arr(env.frame())[:, :board_width]
    target = (np.asarray(waypoint) if waypoint is not None else
              np.argwhere(initial == target_color).min(axis=0))
    actions = tuple(a for a in env.actions if a != 6) + tuple(
        (6, int(x), int(y)) for x, y in switches)

    def key(node):
        return arr(node.frame())[:-1].tobytes()

    def distance(node):
        cells = np.argwhere(arr(node.frame())[:, :board_width] == avatar_color)
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
            reached_waypoint = (waypoint is not None and distance(child) == 0)
            if child.levels_completed > base_level or reached_waypoint:
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

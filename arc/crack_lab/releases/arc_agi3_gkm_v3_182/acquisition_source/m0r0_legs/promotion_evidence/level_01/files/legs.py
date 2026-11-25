"""Reusable skills derived from public frame and action observations."""
from collections import deque

import numpy as np
from perception import arr, connected_components, safe_step


def reunite_mirrored_pair(env, floor_color=5, avatar_color=10, max_states=10000):
    """Route a pair with shared vertical and opposite horizontal controls to meet.

    Infer square tiles and walkable cells from the frame, then search the joint
    positions. Remaining distance on this joint graph measures progress toward
    reunion, including detours needed to synchronize the pair against walls.
    Verify the complete route on a clone before applying it to the real arena.
    """
    frame = arr(env.frame())
    avatars = sorted(connected_components(frame, [avatar_color]),
                     key=lambda b: b.bbox[1])
    if len(avatars) != 2 or avatars[0].size != avatars[1].size:
        raise ValueError('Expected two equal square avatars')
    size, width = avatars[0].size
    if size != width:
        raise ValueError('Expected square tiles')
    start = tuple(b.top_left for b in avatars)
    oy, ox = start[0][0] % size, start[0][1] % size
    walkable = set()
    for y in range(oy, frame.shape[0] - size + 1, size):
        for x in range(ox, frame.shape[1] - size + 1, size):
            tile = frame[y:y + size, x:x + size]
            if np.all((tile == floor_color) | (tile == avatar_color)):
                walkable.add((y, x))

    directions = {1: (-size, 0), 2: (size, 0),
                  3: (0, -size), 4: (0, size)}

    def advance(state, action):
        dy, dx = directions[action]
        result = []
        for index, (y, x) in enumerate(state):
            target = (y + dy, x + (dx if index == 0 else -dx))
            result.append(target if target in walkable else (y, x))
        return tuple(result)

    queue = deque([start])
    parents = {start: None}
    target = None
    while queue and len(parents) <= max_states:
        state = queue.popleft()
        if state[0] == state[1]:
            target = state
            break
        for action in directions:
            successor = advance(state, action)
            if successor not in parents:
                parents[successor] = (state, action)
                queue.append(successor)
    if target is None:
        raise RuntimeError('No bounded reunion route found')
    path = []
    while parents[target] is not None:
        target, action = parents[target]
        path.append(action)
    path.reverse()

    clone = env.clone()
    base_level = env.levels_completed
    predicted = start
    for index, action in enumerate(path):
        safe_step(clone, action)
        predicted = advance(predicted, action)
        if clone.levels_completed > base_level:
            path = path[:index + 1]
            break
        observed = tuple(sorted(b.top_left for b in
                                connected_components(clone.frame(), [avatar_color])))
        if observed != tuple(sorted(predicted)):
            raise RuntimeError('Mirrored movement model failed clone verification')
    if clone.levels_completed <= base_level:
        raise RuntimeError('Reunion did not complete the level on the clone')
    for action in path:
        safe_step(env, action)
        if env.levels_completed > base_level or env.terminal():
            break

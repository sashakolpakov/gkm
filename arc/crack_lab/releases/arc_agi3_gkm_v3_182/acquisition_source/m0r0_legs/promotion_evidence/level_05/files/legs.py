"""Reusable skills derived from public frame and action observations."""
from collections import deque

import numpy as np
from perception import arr, connected_components, safe_step


def reunite_mirrored_pair(env, floor_color=5, avatar_color=10, max_states=10000,
                         hazard_colors=(), pressure_colors=(), horizontal_sign=1):
    """Route a pair with shared vertical and opposite horizontal controls to meet.

    Infer square tiles and walkable cells from the frame, then search the joint
    positions. Remaining distance on this joint graph measures progress toward
    reunion, including detours needed to synchronize the pair against walls.
    Exclude actions that attempt to enter a supplied hazard color, since those
    reset the pair rather than blocking movement. Verify the complete route on
    a clone before applying it to the real arena.

    Optional pressure colors mark single square plates and larger matching
    gates. A gate is traversable while either avatar holds a matching plate;
    search joint positions so holding a plate is a cooperative subgoal.
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
    hazards = set()
    plates = {color: {b.top_left for b in connected_components(frame, [color])
                      if b.size == (size, size)} for color in pressure_colors}
    gates = {}
    for y in range(oy, frame.shape[0] - size + 1, size):
        for x in range(ox, frame.shape[1] - size + 1, size):
            tile = frame[y:y + size, x:x + size]
            if np.any(np.isin(tile, hazard_colors)):
                hazards.add((y, x))
            if np.all((tile == floor_color) | (tile == avatar_color)):
                walkable.add((y, x))
            for color in pressure_colors:
                if np.all(tile == color):
                    walkable.add((y, x))
                    if (y, x) not in plates[color]:
                        gates[(y, x)] = color

    directions = {1: (-size, 0), 2: (size, 0),
                  3: (0, -size * horizontal_sign),
                  4: (0, size * horizontal_sign)}

    def advance(state, action):
        dy, dx = directions[action]
        active = {color for color, cells in plates.items()
                  if any(position in cells for position in state)}
        result = []
        for index, (y, x) in enumerate(state):
            target = (y + dy, x + (dx if index == 0 else -dx))
            if target in hazards:
                return None
            passable = (target in walkable and
                        (target not in gates or gates[target] in active))
            result.append(target if passable else (y, x))
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
            if successor is not None and successor not in parents:
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
        current = arr(clone.frame())
        observed = tuple(sorted((y, x) for y, x in walkable
                                if np.all(current[y:y + size, x:x + size]
                                          == avatar_color)))
        if observed != tuple(sorted(predicted)):
            raise RuntimeError(f'Mirrored movement model failed clone verification at {index}: action={action}, predicted={predicted}, observed={observed}')
    if clone.levels_completed <= base_level:
        raise RuntimeError('Reunion did not complete the level on the clone')
    for action in path:
        safe_step(env, action)
        if env.levels_completed > base_level or env.terminal():
            break


def park_selectable_blockers(env, floor_color=5, avatar_color=10,
                             blocker_color=9, selected_color=11,
                             allow_chamber_parking=False):
    """Select small blockers and park each in a reachable corridor dead end.

    Treat the number of blockers outside dead ends as dense progress. Infer
    tile spacing from the avatars, avoid occupied cells, and verify the whole
    parking sequence on a clone before committing it. If explicitly enabled
    for open chambers, park far from the initial obstruction on a cell whose
    removal preserves connectivity instead of requiring a dead end.
    """
    frame = arr(env.frame())
    avatars = connected_components(frame, [avatar_color])
    if len(avatars) != 2 or avatars[0].size != avatars[1].size:
        raise ValueError('Expected two equal avatars')
    size, width = avatars[0].size
    if size != width:
        raise ValueError('Expected square avatars')
    oy, ox = (v % size for v in avatars[0].top_left)
    occupied = {b.top_left for b in avatars}
    walkable = set()
    blockers = set()
    for y in range(oy, frame.shape[0] - size + 1, size):
        for x in range(ox, frame.shape[1] - size + 1, size):
            tile = frame[y:y + size, x:x + size]
            if np.all(np.isin(tile, (floor_color, avatar_color, blocker_color))):
                walkable.add((y, x))
                if np.any(tile == blocker_color):
                    blockers.add((y, x))
    directions = {1: (-size, 0), 2: (size, 0),
                  3: (0, -size), 4: (0, size)}

    def neighbors(position):
        y, x = position
        return [(a, (y + dy, x + dx))
                for a, (dy, dx) in directions.items()
                if (y + dy, x + dx) in walkable]

    leaves = {p for p in walkable if len(neighbors(p)) == 1}
    parking = leaves
    if not leaves and allow_chamber_parking:
        parking = set()
        for removed in walkable:
            remaining = walkable - {removed}
            reached = {min(remaining)}
            queue = deque(reached)
            while queue:
                for _, target in neighbors(queue.popleft()):
                    if target != removed and target not in reached:
                        reached.add(target)
                        queue.append(target)
            if reached == remaining:
                parking.add(removed)
    clone = env.clone()
    path = []

    def apply(action):
        safe_step(clone, action)
        path.append(action)

    def observed_cells(color):
        return {((b.bbox[0] - oy) // size * size + oy,
                 (b.bbox[1] - ox) // size * size + ox)
                for b in connected_components(clone.frame(), [color])
                if b.size[0] <= size and b.size[1] <= size}

    ay, ax = min(occupied)
    for start in sorted(blockers):
        if start in leaves:
            continue
        queue = deque([start])
        routes = {start: []}
        unavailable = (blockers - {start}) | occupied
        while queue:
            position = queue.popleft()
            for action, target in neighbors(position):
                if target not in routes and target not in unavailable:
                    routes[target] = routes[position] + [action]
                    queue.append(target)
        options = (parking & routes.keys()) - unavailable - {start}
        if not options:
            raise RuntimeError('No free reachable parking cell for blocker')
        target = min(options, key=lambda p: (
            len(routes[p]) if leaves else -len(routes[p]), p))
        by, bx = start
        center = size // 2
        apply((6, bx + center, by + center))
        position = start
        if observed_cells(selected_color) != {start}:
            raise RuntimeError('Blocker selection failed clone verification')
        for action in routes[target]:
            apply(action)
            dy, dx = directions[action]
            position = (position[0] + dy, position[1] + dx)
            if observed_cells(selected_color) != {position}:
                raise RuntimeError('Blocker movement failed clone verification')
        apply((6, ax + center, ay + center))
        blockers.remove(start)
        blockers.add(target)
        if (observed_cells(blocker_color) != blockers
                or observed_cells(avatar_color) != occupied):
            raise RuntimeError('Parking failed clone verification')
    for action in path:
        safe_step(env, action)

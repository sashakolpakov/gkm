"""Small reusable skills built from public frame observations and actions."""
from functools import lru_cache

from perception import connected_components, safe_step


def solve_peg_jumps(env, *, extra_sockets=(), target_pegs=None):
    """Read square sockets, plan orthogonal peg captures, verify, then execute."""
    def observe(arena):
        blobs = connected_components(arena.frame(), colors=(1, 14))
        sockets = {
            (b.bbox[1] + 2, b.bbox[0] + 2)
            for b in blobs
            if b.size == (4, 4) and b.area in (12, 16)
        }
        pegs = {
            (b.bbox[1] + 2, b.bbox[0] + 2)
            for b in blobs if b.color == 14 and b.size == (4, 4)
        }
        return sockets, frozenset(pegs)

    board, initial = observe(env)
    board.update(extra_sockets)
    target = None if target_pegs is None else frozenset(target_pegs)
    pitch = min(
        abs(a[axis] - b[axis])
        for a in board for b in board for axis in (0, 1)
        if a[axis] != b[axis]
    )
    jumps = []
    for x, y in sorted(board):
        for dx, dy in ((pitch, 0), (-pitch, 0), (0, pitch), (0, -pitch)):
            middle = (x + dx, y + dy)
            destination = (x + 2 * dx, y + 2 * dy)
            if middle in board and destination in board:
                jumps.append(((x, y), middle, destination))

    @lru_cache(None)
    def plan(pegs):
        if (target is not None and pegs == target) or (target is None and len(pegs) == 1):
            return ()
        for source, middle, destination in jumps:
            if source in pegs and middle in pegs and destination not in pegs:
                remaining = (pegs - {source, middle}) | {destination}
                suffix = plan(frozenset(remaining))
                if suffix is not None:
                    return ((source, destination),) + suffix
        return None

    path = plan(initial)
    if path is None:
        raise RuntimeError('No peg reduction path found')
    base_level = env.levels_completed
    clone = env.clone()
    for index, (source, destination) in enumerate(path):
        safe_step(clone, (6, *source))
        safe_step(clone, (6, *destination))
        if clone.levels_completed > base_level:
            path = path[:index + 1]
            break
        if len(observe(clone)[1]) != len(initial) - index - 1:
            raise RuntimeError('Observed jump disagrees with the capture model')
    if clone.levels_completed <= base_level and (target is None or observe(clone)[1] != target):
        raise RuntimeError('Peg reduction did not complete the level')
    for source, destination in path:
        safe_step(env, (6, *source))
        safe_step(env, (6, *destination))
        if env.levels_completed > base_level:
            return


def solve_peg_ferry(env):
    """Stage captures onto a movable socket, ferry it, and capture off it."""
    from collections import deque
    from time import sleep

    def carrier(arena):
        outlines = [b for b in connected_components(arena.frame(), colors=(11,))
                    if b.size == (6, 6)]
        if len(outlines) != 1:
            raise RuntimeError('Expected one movable socket')
        b = outlines[0]
        return b.bbox[1] + 3, b.bbox[0] + 3

    def routes(arena):
        start = carrier(arena)
        found = {start: ()}
        queue = deque([(arena.clone(), ())])
        while queue and len(found) < 100:
            node, path = queue.popleft()
            for action in (1, 2, 3, 4):
                child = node.clone()
                safe_step(child, action)
                sleep(0.004)
                position = carrier(child)
                if position not in found:
                    found[position] = path + (action,)
                    queue.append((child, path + (action,)))
        if queue:
            raise RuntimeError('Carrier routing bound exceeded')
        return found

    blobs = connected_components(env.frame(), colors=(1, 14))
    sockets = {(b.bbox[1] + 2, b.bbox[0] + 2) for b in blobs
               if b.size == (4, 4) and b.area in (12, 16)}
    pegs = {(b.bbox[1] + 2, b.bbox[0] + 2) for b in blobs
            if b.color == 14 and b.size == (4, 4)}
    pitch = min(abs(a[i] - b[i]) for a in sockets for b in sockets
                for i in (0, 1) if a[i] != b[i])
    directions = ((pitch, 0), (-pitch, 0), (0, pitch), (0, -pitch))
    unseen = set(sockets)
    groups = []
    while unseen:
        group = {unseen.pop()}
        pending = list(group)
        while pending:
            x, y = pending.pop()
            for dx, dy in directions:
                p = x + dx, y + dy
                if p in unseen:
                    unseen.remove(p)
                    group.add(p)
                    pending.append(p)
        groups.append(group)
    source = max(groups, key=lambda group: len(group & pegs))
    outside = pegs - source
    if len(outside) != 1:
        raise RuntimeError('Ferry expects one isolated destination peg')

    paths = routes(env)
    pickup = next((p for p in paths if any(
        (p[0] + dx, p[1] + dy) in source and
        (p[0] + 2 * dx, p[1] + 2 * dy) in source
        for dx, dy in directions)), None)
    if pickup is None:
        raise RuntimeError('No loading reach from carrier tracks')
    for action in paths[pickup]:
        safe_step(env, action)
    solve_peg_jumps(env, extra_sockets=(pickup,), target_pegs=outside | {pickup})

    paths = routes(env)
    dropoff = next((p for p in paths if any(
        (p[0] + dx, p[1] + dy) in outside and
        (p[0] + 2 * dx, p[1] + 2 * dy) in sockets
        for dx, dy in directions)), None)
    if dropoff is None:
        raise RuntimeError('No unloading reach from carrier tracks')
    for action in paths[dropoff]:
        safe_step(env, action)
    solve_peg_jumps(env)


def move_socket_team(env, actions, *, expected_centers):
    """Follow a tested shared-key route, checking visible socket endpoints on a clone."""
    from time import sleep

    def centers(arena):
        return sorted((b.bbox[1] + 3, b.bbox[0] + 3)
                      for b in connected_components(arena.frame(), colors=(11,))
                      if b.size[0] == 6)

    clone = env.clone()
    for action in actions:
        safe_step(clone, action)
        sleep(0.004)
    if centers(clone) != sorted(expected_centers):
        raise RuntimeError('Shared socket route disagrees with observed endpoints')
    for action in actions:
        safe_step(env, action)

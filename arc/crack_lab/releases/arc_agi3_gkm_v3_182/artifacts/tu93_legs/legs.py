"""Small reusable skills inferred from public frame/action experiments."""
from collections import deque
from heapq import heappop, heappush
from itertools import count
from time import sleep

from perception import arr, connected_components, replay


def clear_linked_sentries(env, sentry_color=8, max_states=256, max_depth=24,
                          incremental=False):
    """Find a safe approach that clears directional sentries on linked tiles.

    A sentry can kill an avatar approaching its facing side, while a flank
    removes it. Search observed clone transitions; remaining sentry pixels
    measure progress. Ignore the bottom-row move meter when merging states.
    Incremental mode confirms one flank at a time, retaining sentry locations
    across temporary patrol occlusions to measure permanent clearance.
    """
    if incremental:
        # Patrols can obscure sentries, so retain their entry-frame locations.
        targets = {((b.bbox[0] + b.bbox[2]) // 2,
                    (b.bbox[1] + b.bbox[3]) // 2)
                   for b in connected_components(env.frame(), colors=[sentry_color])}
        while targets:
            queue = deque([(env.clone(), [])])
            seen = {arr(env.frame())[:-1].tobytes()}
            found = None
            while queue and len(seen) <= max_states:
                node, path = queue.popleft()
                frame = arr(node.frame())
                cleared = {p for p in targets if int(frame[p]) in (0, 9, 14)}
                if cleared:
                    found = path, cleared
                    break
                if len(path) >= max_depth:
                    continue
                for action in node.actions:
                    child = node.clone()
                    child.step(action)
                    sleep(0.004)
                    if child.terminal():
                        continue
                    key = arr(child.frame())[:-1].tobytes()
                    if key not in seen:
                        seen.add(key)
                        queue.append((child, path + [action]))
            if found is None:
                raise RuntimeError('No safe incremental sentry flank within bounds')
            path, cleared = found
            trial = replay(env, path)
            if not all(int(arr(trial.frame())[p]) in (0, 9, 14) for p in cleared):
                raise RuntimeError('Sentry flank failed independent clone replay')
            for action in path:
                env.step(action)
            targets -= cleared
        return
    before = env.levels_completed
    queue = deque([(env.clone(), [])])
    seen = {arr(env.frame())[:-1].tobytes()}
    while queue and len(seen) <= max_states:
        node, path = queue.popleft()
        if node.terminal():
            continue
        if node.levels_completed > before or not (arr(node.frame()) == sentry_color).any():
            # Every transition in this route has already been tested on clones.
            for action in path:
                env.step(action)
            return
        if len(path) >= max_depth:
            continue
        for action in node.actions:
            child = node.clone()
            child.step(action)
            sleep(0.004)
            if child.terminal():
                continue
            key = arr(child.frame())[:-1].tobytes()
            if key not in seen:
                seen.add(key)
                queue.append((child, path + [action]))
    raise RuntimeError('No safe sentry-clearing route within clone search bounds')


def traverse_linked_tiles(env, avatar_color=9, goal_color=14, link_color=2):
    """Follow a shortest tile route, verifying completion on a clone first.

    Tiles are separated by one tile-width coloured links. Actions 1..4
    move north, south, west, east. Remaining distance in the link graph
    supplies dense progress toward the goal.
    """
    frame = arr(env.frame())
    avatar = connected_components(frame, colors=[avatar_color])[0]
    goal = connected_components(frame, colors=[goal_color])[0]

    def center(blob):
        r0, c0, r1, c1 = blob.bbox
        return (r0 + r1) // 2, (c0 + c1) // 2

    start, target = center(avatar), center(goal)
    stride = 2 * avatar.size[0]
    directions = {1: (-stride, 0), 2: (stride, 0),
                  3: (0, -stride), 4: (0, stride)}
    tile_colors = (0, avatar_color, goal_color)
    queue = deque([target])
    distance = {target: 0}
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions.values():
            nxt = r + dr, c + dc
            if not (0 <= nxt[0] < frame.shape[0] and
                    0 <= nxt[1] < frame.shape[1]):
                continue
            if (nxt not in distance and int(frame[nxt]) in tile_colors and
                    int(frame[r + dr // 2, c + dc // 2]) == link_color):
                distance[nxt] = distance[(r, c)] + 1
                queue.append(nxt)
    if start not in distance:
        raise RuntimeError('Goal is unreachable through visible links')
    path, pos = [], start
    while pos != target:
        for action, (dr, dc) in directions.items():
            nxt = pos[0] + dr, pos[1] + dc
            if (distance.get(nxt) == distance[pos] - 1 and
                    int(frame[pos[0] + dr // 2, pos[1] + dc // 2]) == link_color):
                path.append(action)
                pos = nxt
                break
        else:
            raise RuntimeError('No descending maze route')
    before = env.levels_completed
    trial = replay(env, path)
    if trial.levels_completed <= before:
        raise RuntimeError('Clone did not confirm the maze goal')
    for action in path:
        if env.terminal() or env.levels_completed > before:
            break
        env.step(action)


def traverse_patrolled_links(env, avatar_color=9, goal_color=14,
                             link_color=2, patrol_color=12,
                             max_states=2048, max_depth=80):
    """Time a linked-tile crossing around moving hazards on public clones.

    Static graph distance supplies dense progress. Frame states retain patrol
    positions and facing marks, allowing safe detours and backtracking when
    the shortest geometric route is unsafe. Only the bottom move meter is
    omitted from state identity.
    """
    frame = arr(env.frame())

    def center(blob):
        r0, c0, r1, c1 = blob.bbox
        return (r0 + r1) // 2, (c0 + c1) // 2

    avatar = connected_components(frame, colors=[avatar_color])[0]
    target = center(connected_components(frame, colors=[goal_color])[0])
    stride = 2 * avatar.size[0]
    directions = ((-stride, 0), (stride, 0), (0, -stride), (0, stride))
    distance, queue = {target: 0}, deque([target])
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nxt = r + dr, c + dc
            if (0 <= nxt[0] < frame.shape[0] and
                    0 <= nxt[1] < frame.shape[1] and nxt not in distance and
                    int(frame[nxt]) in (0, avatar_color, goal_color, patrol_color) and
                    int(frame[r + dr // 2, c + dc // 2]) == link_color):
                distance[nxt] = distance[(r, c)] + 1
                queue.append(nxt)

    def remaining(f):
        # The avatar's 3x3 footprint may contain a facing mark.
        ys, xs = (f == avatar_color).nonzero()
        if not len(ys):
            return max_depth + 1
        pos = (int(ys.min() + ys.max()) // 2,
               int(xs.min() + xs.max()) // 2)
        return distance.get(pos, max_depth + 1)

    before = env.levels_completed
    serial = count()
    start_key = frame[:-1].tobytes()
    best = {start_key: 0}
    heap = [(remaining(frame), next(serial), env.clone(), [], start_key)]
    while heap and len(best) <= max_states:
        _, _, node, path, key = heappop(heap)
        if len(path) != best[key]:
            continue
        if len(path) >= max_depth:
            continue
        for action in node.actions:
            child = node.clone()
            child.step(action)
            sleep(0.004)
            route = path + [action]
            if child.levels_completed > before:
                trial = replay(env, route)
                if trial.levels_completed <= before:
                    raise RuntimeError('Patrol route failed independent clone replay')
                for move in route:
                    env.step(move)
                return
            if child.terminal():
                continue
            f = arr(child.frame())
            child_key = f[:-1].tobytes()
            if len(route) < best.get(child_key, max_depth + 1):
                best[child_key] = len(route)
                heappush(heap, (len(route) + remaining(f), next(serial),
                                child, route, child_key))
    raise RuntimeError('No safe patrol crossing within clone search bounds')

"""Small reusable skills inferred from public frame/action experiments."""
from collections import deque

from perception import arr, connected_components, replay


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

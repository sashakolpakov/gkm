"""Reusable frame-driven skills."""
import heapq
import itertools

from perception import arr, connected_components, safe_step


def ascend_breakable_shaft(env, max_moves=60, origin=(15, 3), cell=6, shape=(10, 7)):
    """Route an upward-floating avatar through a shaft, clearing yellow tiles.

    Replan after every action because rising scrolls the visible board. The
    dense objective is a cheap upward route through air and breakable tiles;
    once the green exit appears, route to it instead of the frame boundary.
    """
    base_level = env.levels_completed
    ox, oy = origin
    rows, cols = shape
    for _ in range(max_moves):
        if env.terminal() or env.levels_completed > base_level:
            return
        frame = arr(env.frame())
        avatars = connected_components(frame, colors=[9])
        if not avatars:
            return
        cy, cx = avatars[0].centroid
        start_y = round((cy - oy) / cell)
        start_x = round((cx - ox) / cell)
        tiles = [[int(frame[oy + y * cell, ox + x * cell])
                  for x in range(cols)] for y in range(rows)]
        exit_visible = any(7 in row for row in tiles)
        serial = itertools.count()
        queue = [(0, next(serial), start_y, start_x, [])]
        seen = set()
        next_action = None
        while queue:
            cost, _, y, x, path = heapq.heappop(queue)
            if (y, x) in seen:
                continue
            seen.add((y, x))
            if path and (tiles[y][x] == 7 if exit_visible else y == 0):
                next_action = path[0]
                break
            for dy, dx in [(-1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < rows and 0 <= nx < cols):
                    continue
                if tiles[ny][nx] == 5:
                    continue
                actions = []
                if tiles[ny][nx] == 14:
                    actions.append((6, ox + nx * cell, oy + ny * cell))
                if dx:
                    actions.append(3 if dx < 0 else 4)
                heapq.heappush(queue, (cost + len(actions) + 0.01,
                                      next(serial), ny, nx, path + actions))
        if next_action is None:
            return
        safe_step(env, next_action)

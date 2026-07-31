# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.
import numpy as np
from collections import deque

# Movement action -> (dr, dc) in node space. Discovered by experiment on tu93:
#   1=UP, 2=DOWN, 3=LEFT, 4=RIGHT.
_DIR = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def _mode_color(f):
    vals, cnts = np.unique(f, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _least_color(f, bg):
    """Rarest non-background color (used as the avatar marker)."""
    vals, cnts = np.unique(f, return_counts=True)
    best, bc = None, None
    for v, c in zip(vals, cnts):
        v = int(v)
        if v == bg:
            continue
        if bc is None or c < bc:
            best, bc = v, int(c)
    return best


def parse_block_maze(f, cell=3):
    """Parse a grid maze drawn with fixed `cell`x`cell` blocks.

    Layout convention (discovered on tu93): nodes live at EVEN block indices,
    edges between adjacent nodes at ODD block indices. A block is 'present' if
    its centre pixel differs from the background colour. An edge is passable if
    its block is present. Returns a dict with graph + start/goal node coords.
    """
    f = np.asarray(f)
    bg = _mode_color(f)
    # Ignore uniform full-row / full-col borders (e.g. a frame edge) so they
    # don't skew the grid origin.
    # A border/HUD bar is a colour confined to a single row (or column) that
    # spans a large fraction of the frame. Robust to a depleting counter bar.
    H, W = f.shape
    border = set()
    for v in np.unique(f):
        v = int(v)
        if v == bg:
            continue
        vy, vx = np.where(f == v)
        h = vy.max() - vy.min() + 1
        w = vx.max() - vx.min() + 1
        if (h == 1 and w >= 0.5 * W) or (w == 1 and h >= 0.5 * H):
            border.add(v)
    mask = (f != bg)
    for b in border:
        mask &= (f != b)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    R0, C0 = int(ys.min()), int(xs.min())
    marker = _least_color(f, bg)

    def center(br, bc):
        r, c = R0 + br * cell, C0 + bc * cell
        if r + cell > 64 or c + cell > 64:
            return bg
        return int(f[r + cell // 2, c + cell // 2])

    def block_cells(br, bc):
        r, c = R0 + br * cell, C0 + bc * cell
        return f[r:r + cell, c:c + cell]

    nbr = (64 - R0) // cell
    nbc = (64 - C0) // cell
    ni = nbr // 2 + 1
    nj = nbc // 2 + 1

    node_color = {}
    for i in range(ni):
        for j in range(nj):
            col = center(2 * i, 2 * j)
            if col != bg:
                node_color[(i, j)] = col

    if not node_color:
        return None

    # open path color = most common node color
    from collections import Counter
    cc = Counter(node_color.values())
    open_col = cc.most_common(1)[0][0]

    # start = node whose block contains the marker
    start = None
    for (i, j) in node_color:
        blk = block_cells(2 * i, 2 * j)
        if marker is not None and (blk == marker).any():
            start = (i, j)
            break

    start_col = node_color.get(start) if start else None

    # goal = a node whose color differs from open path and from the start block
    goal = None
    for (i, j), col in node_color.items():
        if (i, j) == start:
            continue
        if col != open_col and col != start_col:
            goal = (i, j)
            break

    def edge_present(br, bc):
        return center(br, bc) != bg

    def neigh(i, j):
        out = []
        # up
        if i > 0 and edge_present(2 * i - 1, 2 * j) and (i - 1, j) in node_color:
            out.append((i - 1, j, 1))
        # down
        if edge_present(2 * i + 1, 2 * j) and (i + 1, j) in node_color:
            out.append((i + 1, j, 2))
        # left
        if j > 0 and edge_present(2 * i, 2 * j - 1) and (i, j - 1) in node_color:
            out.append((i, j - 1, 3))
        # right
        if edge_present(2 * i, 2 * j + 1) and (i, j + 1) in node_color:
            out.append((i, j + 1, 4))
        return out

    return {
        "bg": bg, "marker": marker, "open": open_col,
        "nodes": node_color, "start": start, "goal": goal, "neigh": neigh,
        "origin": (R0, C0), "cell": cell,
    }


def _maze_path_between(g, start, goal, blocked=()):
    """BFS between two parsed maze nodes, optionally avoiding some nodes."""
    if start is None or goal is None or start in blocked or goal in blocked:
        return None
    neigh = g["neigh"]
    q = deque([(start, [])])
    seen = {start}
    while q:
        node, path = q.popleft()
        if node == goal:
            return path
        for ni, nj, a in neigh(*node):
            if (ni, nj) not in seen and (ni, nj) not in blocked:
                seen.add((ni, nj))
                q.append(((ni, nj), path + [a]))
    return None


def maze_path_actions(f, cell=3):
    """BFS in node space from avatar to goal; return a list of key actions."""
    g = parse_block_maze(f, cell)
    if not g or g["start"] is None or g["goal"] is None:
        return None
    return _maze_path_between(g, g["start"], g["goal"])


def maze_path_via_color(f, waypoint_color, entry_action, cell=3):
    """Plan through a colored maze node, entering it from one required side.

    Until the waypoint disappears, route to the staging node without crossing
    the waypoint, enter it using ``entry_action``, then continue to the goal.
    Once it has disappeared, this reduces to ordinary maze path planning.
    """
    f = np.asarray(f)
    g = parse_block_maze(f, cell)
    if not g or g["start"] is None or g["goal"] is None:
        return None

    r0, c0 = g["origin"]
    waypoint = None
    for i, j in g["nodes"]:
        block = f[r0 + 2 * i * cell:r0 + (2 * i + 1) * cell,
                  c0 + 2 * j * cell:c0 + (2 * j + 1) * cell]
        if (block == waypoint_color).any():
            waypoint = (i, j)
            break
    if waypoint is None:
        return _maze_path_between(g, g["start"], g["goal"])

    dr, dc = _DIR[entry_action]
    staging = (waypoint[0] - dr, waypoint[1] - dc)
    valid_entry = any((ni, nj) == waypoint and action == entry_action
                      for ni, nj, action in g["neigh"](*staging))
    if not valid_entry:
        return None
    prefix = _maze_path_between(
        g, g["start"], staging, blocked={waypoint})
    suffix = _maze_path_between(g, waypoint, g["goal"])
    if prefix is None or suffix is None:
        return None
    return prefix + [entry_action] + suffix


def maze_path_via_direction_markers(
        f, waypoint_color, marker_color, goal_color, cell=3):
    """Route through all edge-marked waypoints, then to the colored goal.

    A marker on a waypoint edge encodes its required entry direction:
    left->up, right->down, bottom->right, top->left. While markers remain,
    choose the shortest currently reachable correctly staged entry without
    crossing another marker or the goal.
    """
    f = np.asarray(f)
    g = parse_block_maze(f, cell)
    if not g or g["start"] is None:
        return None

    r0, c0 = g["origin"]
    waypoints = {}
    danger_cells = {}
    goal = None
    marker_to_action = {(1, 0): 1, (1, 2): 2, (2, 1): 4, (0, 1): 3}
    marker_to_offset = {
        (0, 1): (-1, 0), (2, 1): (1, 0),
        (1, 0): (0, -1), (1, 2): (0, 1),
    }
    for i, j in g["nodes"]:
        block = f[r0 + 2 * i * cell:r0 + (2 * i + 1) * cell,
                  c0 + 2 * j * cell:c0 + (2 * j + 1) * cell]
        if (block == goal_color).any():
            goal = (i, j)
        if not (block == waypoint_color).any():
            continue
        positions = np.argwhere(block == marker_color)
        if len(positions) == 1:
            pos = tuple(int(x) for x in positions[0])
            if pos in marker_to_action:
                waypoints[(i, j)] = marker_to_action[pos]
                dr, dc = marker_to_offset[pos]
                danger_cells[(i, j)] = (i + dr, j + dc)

    if not waypoints:
        return _maze_path_between(g, g["start"], goal)

    blocked = set(waypoints)
    blocked.update(danger_cells.values())
    if goal is not None:
        blocked.add(goal)
    candidates = []
    for waypoint, action in waypoints.items():
        dr, dc = _DIR[action]
        staging = (waypoint[0] - dr, waypoint[1] - dc)
        valid_entry = any(
            (ni, nj) == waypoint and a == action
            for ni, nj, a in g["neigh"](*staging)
        )
        if not valid_entry:
            continue
        prefix = _maze_path_between(
            g, g["start"], staging, blocked=blocked - {waypoint})
        if prefix is not None:
            candidates.append(prefix + [action])
    return min(candidates, key=len) if candidates else None


def drive_replan(env, plan_fn, max_steps=300):
    """Generic closed-loop driver: sense -> plan -> commit ONE action -> repeat.

    Higher-order leg. `plan_fn(frame)` returns a list of key-actions for the
    current live frame; we commit only its first action, then re-plan from the
    fresh frame. This is robust to avatar/marker dynamics and any drift, since
    every step re-derives the plan. Stops on level gain, terminal, or when the
    planner yields no path. Returns True iff a level was gained.
    """
    base_levels = env.levels_completed
    for _ in range(max_steps):
        if env.terminal() or env.levels_completed > base_levels:
            return env.levels_completed > base_levels
        path = plan_fn(env.frame())
        if not path:
            return env.levels_completed > base_levels
        env.step(path[0])
    return env.levels_completed > base_levels


def drive_block_maze(env, cell=3, max_steps=300):
    """Navigate a fixed-block grid maze avatar to its goal.

    Thin composition: the maze planner (maze_path_actions) driven by the
    generic closed-loop re-planning leg (drive_replan).
    """
    return drive_replan(env, lambda f: maze_path_actions(f, cell), max_steps)


def drive_block_maze_via_color(
        env, waypoint_color, entry_action, cell=3, max_steps=300):
    """Navigate a block maze via a colored node with directional entry."""
    return drive_replan(
        env,
        lambda f: maze_path_via_color(
            f, waypoint_color, entry_action, cell),
        max_steps,
    )


def drive_block_maze_via_direction_markers(
        env, waypoint_color, marker_color, goal_color, cell=3, max_steps=300):
    """Clear directional maze waypoints before navigating to the goal."""
    return drive_replan(
        env,
        lambda f: maze_path_via_direction_markers(
            f, waypoint_color, marker_color, goal_color, cell),
        max_steps,
    )


def _bounded_visible_path(env, goal_fn, max_states=5000, max_depth=80):
    """Find a short key-action path using only visible, non-HUD frame state."""
    root = env.clone()

    def replay(path):
        node = root.clone()
        for action in path:
            node.step(action)
        return node

    def key(node):
        # tu93's final row is a move-budget HUD, not physical world state.
        return np.asarray(node.frame())[:-1].tobytes()

    queue = deque([[]])
    seen = {key(root)}
    while queue and len(seen) <= max_states:
        path = queue.popleft()
        node = replay(path)
        if goal_fn(node):
            return path
        if len(path) >= max_depth or node.terminal():
            continue
        for action in node.actions:
            child_path = path + [int(action)]
            child = replay(child_path)
            state = key(child)
            if state in seen:
                continue
            seen.add(state)
            if goal_fn(child):
                return child_path
            queue.append(child_path)
    return None


def drive_dynamic_maze_via_color(
        env, waypoint_color, max_states=5000, max_depth=80):
    """Clear visible colored gates, then finish a maze with turn-driven agents.

    The number of waypoint-colored pixels is the dense progress measure.
    Search is keyed by the physical frame without the depleting HUD row.
    """
    base_levels = env.levels_completed
    while not env.terminal():
        frame = np.asarray(env.frame())
        remaining = int(np.count_nonzero(frame == waypoint_color))
        if remaining == 0:
            break
        path = _bounded_visible_path(
            env,
            lambda node: int(np.count_nonzero(
                np.asarray(node.frame()) == waypoint_color)) < remaining,
            max_states,
            max_depth,
        )
        if not path:
            return False
        for action in path:
            env.step(action)
            if env.terminal() or env.levels_completed > base_levels:
                return env.levels_completed > base_levels

    path = _bounded_visible_path(
        env,
        lambda node: node.levels_completed > base_levels,
        max_states,
        max_depth,
    )
    if not path:
        return False
    for action in path:
        env.step(action)
        if env.terminal() or env.levels_completed > base_levels:
            break
    return env.levels_completed > base_levels


def drive_dynamic_directional_waypoints(
        env, waypoint_color, marker_color, avatar_color,
        cell=3, max_states=5000, max_depth=80):
    """Clear directional waypoints in a maze with turn-driven agents.

    Waypoint locations are fixed, but moving objects can temporarily cover
    their base color.  Count the direction marker at those fixed locations as
    dense progress, and only accept a decrease after the avatar has left the
    waypoint.  This distinguishes a cleared waypoint from transient overlap.
    """
    base_levels = env.levels_completed
    initial = np.asarray(env.frame())
    g = parse_block_maze(initial, cell)
    if not g:
        return False
    r0, c0 = g["origin"]
    waypoints = []
    for i, j in g["nodes"]:
        bounds = (
            r0 + 2 * i * cell, c0 + 2 * j * cell,
            r0 + (2 * i + 1) * cell, c0 + (2 * j + 1) * cell,
        )
        y0, x0, y1, x1 = bounds
        block = initial[y0:y1, x0:x1]
        if (block == waypoint_color).any() and (block == marker_color).any():
            waypoints.append(bounds)

    def blocks(frame):
        f = np.asarray(frame)
        return [f[y0:y1, x0:x1] for y0, x0, y1, x1 in waypoints]

    def remaining(frame):
        return sum(int((block == marker_color).any())
                   for block in blocks(frame))

    def avatar_on_waypoint(frame):
        return any((block == avatar_color).any() for block in blocks(frame))

    while not env.terminal():
        current = remaining(env.frame())
        if current == 0:
            break
        path = _bounded_visible_path(
            env,
            lambda node: (
                node.levels_completed > base_levels
                or (
                    not node.terminal()
                    and remaining(node.frame()) < current
                    and not avatar_on_waypoint(node.frame())
                )
            ),
            max_states,
            max_depth,
        )
        if not path:
            return False
        for action in path:
            env.step(action)
            if env.terminal() or env.levels_completed > base_levels:
                return env.levels_completed > base_levels

    path = _bounded_visible_path(
        env,
        lambda node: node.levels_completed > base_levels,
        max_states,
        max_depth,
    )
    if not path:
        return False
    for action in path:
        env.step(action)
        if env.terminal() or env.levels_completed > base_levels:
            break
    return env.levels_completed > base_levels

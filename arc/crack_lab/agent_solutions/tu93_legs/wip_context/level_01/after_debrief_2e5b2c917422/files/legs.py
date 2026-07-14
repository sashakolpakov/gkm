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
    }


def maze_path_actions(f, cell=3):
    """BFS in node space from avatar to goal; return a list of key actions."""
    g = parse_block_maze(f, cell)
    if not g or g["start"] is None or g["goal"] is None:
        return None
    start, goal, neigh = g["start"], g["goal"], g["neigh"]
    q = deque([(start, [])])
    seen = {start}
    while q:
        node, path = q.popleft()
        if node == goal:
            return path
        for ni, nj, a in neigh(*node):
            if (ni, nj) not in seen:
                seen.add((ni, nj))
                q.append(((ni, nj), path + [a]))
    return None


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

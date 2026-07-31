# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.
import itertools
import numpy as np


def _changed_cells(base, frame, row_limit=62):
    """Cells that differ between two frames, ignoring a bottom status bar."""
    d = np.argwhere(np.asarray(base) != np.asarray(frame))
    return [(int(r), int(c)) for r, c in d if r < row_limit]


def discover_toggle_tiles(env, step=2, row_limit=62):
    """Probe coordinate clicks on a clone and return one representative (x, y)
    click per independent toggle region. A "tile" is identified by the bounding
    box of the cells it changes (excluding the bottom status bar). Generic to any
    coordinate-click board where clicking a cell mutates a bounded region.
    """
    base = np.asarray(env.frame()).copy()
    tiles = {}
    for y in range(0, base.shape[0], step):
        for x in range(0, base.shape[1], step):
            c = env.clone()
            c.step(6, x, y)
            cells = _changed_cells(base, c.frame(), row_limit)
            if not cells:
                continue
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            key = (min(rs), min(cs), max(rs), max(cs))
            # keep the click closest to the region center as representative
            cx, cy = (min(cs) + max(cs)) // 2, (min(rs) + max(rs)) // 2
            prev = tiles.get(key)
            if prev is None:
                tiles[key] = (x, y)
            else:
                px, py = prev
                if (x - cx) ** 2 + (y - cy) ** 2 < (px - cx) ** 2 + (py - cy) ** 2:
                    tiles[key] = (x, y)
    return [tiles[k] for k in sorted(tiles)]


def search_toggle_solution(env, tiles=None, max_tiles=16):
    """Find a subset of tile clicks (each tile toggled at most once) that raises
    levels_completed, searching on clones from fewest clicks upward. Returns the
    click list [(x, y), ...] or None. Assumes independent toggle tiles whose
    target configuration is reached by clicking the right subset once each.
    """
    if tiles is None:
        tiles = discover_toggle_tiles(env)
    if not tiles or len(tiles) > max_tiles:
        return None
    base_level = env.levels_completed
    n = len(tiles)
    for k in range(0, n + 1):
        for combo in itertools.combinations(range(n), k):
            c = env.clone()
            for i in combo:
                x, y = tiles[i]
                c.step(6, x, y)
            if c.levels_completed > base_level:
                return [tiles[i] for i in combo]
    return None


def solve_toggle_board(env):
    """Discover the toggle board, search for a completing click subset, and
    commit those clicks on the real env. Returns True on success."""
    clicks = search_toggle_solution(env)
    if not clicks:
        return False
    for x, y in clicks:
        env.step(6, x, y)
    return True

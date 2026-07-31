# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.
#
# Substrate observed on bp35 (raw frames only):
#   * The 64x64 frame is a 6px band lattice: cell (i, j) is sampled at
#     (3 + 6i, 15 + 6j) -> a 10 x 7 symbolic grid.
#   * Row 63 is a move counter (one non-zero pixel per action taken).
#   * The avatar always renders on band row 6; the world scrolls instead.
#     After every action the avatar RISES through open cells until blocked,
#     so height is gained by opening the cell above it.
#   * Action 3 = step left one column, 4 = step right one column.
#     Action 6 with (x, y) pixel coordinates clicks a cell; clicking a
#     removable block (colour 14) turns it into open floor. Any other click
#     is a no-op that still burns a move.
#   * A floor rises from below one band every two moves and kills on contact,
#     so climbing must outpace the move counter.
#   * Reaching a prize cell (a colour outside the terrain palette) on the
#     avatar's row completes the level.
#   * Off-palette colours are NOT always prizes: from L2 on they are also
#     hazards that hang below their own band, so resting under one kills while
#     touching one sideways is simply solid. Nothing here models which is
#     which -- a clone probe decides, and dead successors are dropped.
#   * The world scrolls whenever the avatar rises, so any plan longer than one
#     action goes stale. Walk and climb legs re-read the frame every step.

import heapq
import itertools

import numpy as np

BAND = 6
GRID_ROWS = 10
GRID_COLS = 7
ROW_ANCHORS = [3 + BAND * i for i in range(GRID_ROWS)]
COL_ANCHORS = [15 + BAND * j for j in range(GRID_COLS)]
AVATAR_ROW = 6

WALL_COLORS = (3, 5)
OPEN_COLORS = (10,)
BLOCK_COLORS = (14,)
AVATAR_COLORS = (9, 11)
TOGGLE_COLORS = (12,)

# A band cell is 6x6 raw pixels. A solid object fills it as a rounded 5x5 box
# (21 px); an object that has been emptied leaves only a 5 px marker. Anything
# at or below this area is decoration the avatar walks and rises straight
# through, so area -- not colour -- decides solid from hollow.
HOLLOW_AREA = 8

WALL, OPEN, BLOCK, AVATAR, PRIZE = '#', ' ', 'Y', 'A', '*'

LEFT, RIGHT, CLICK = 3, 4, 6


# --- observation legs -------------------------------------------------

def pixels(frame):
    """The frame as an integer array, for cheap bulk comparisons."""
    return np.asarray(frame, dtype=np.int16)


def cell_fill(frame, i, j):
    """Centre colour of band cell (i, j) and how much of the cell it fills."""
    f = pixels(frame)
    color = int(f[ROW_ANCHORS[i], COL_ANCHORS[j]])
    r0, c0 = BAND * i, COL_ANCHORS[j] - 2
    patch = f[r0:r0 + BAND, c0:c0 + BAND]
    return color, int(np.count_nonzero(patch == color))


def cell_symbol(color, area=None):
    """Classify one band cell into the terrain alphabet.

    ``area`` is the cell fill from :func:`cell_fill`. It only matters for the
    toggling colours, whose solid box and hollow marker share a colour: the
    hollow form is plain floor, the solid form is a block a click opens.
    """
    color = int(color)
    if color in WALL_COLORS:
        return WALL
    if color in OPEN_COLORS:
        return OPEN
    if color in BLOCK_COLORS:
        return BLOCK
    if color in AVATAR_COLORS:
        return AVATAR
    if color in TOGGLE_COLORS and area is not None:
        return OPEN if area <= HOLLOW_AREA else BLOCK
    return PRIZE


def band_grid(frame):
    """Quantise a frame onto the 6px band lattice as a symbolic grid."""
    f = pixels(frame)
    grid = []
    for i in range(GRID_ROWS):
        row = []
        for j in range(GRID_COLS):
            color = int(f[ROW_ANCHORS[i], COL_ANCHORS[j]])
            patch = f[BAND * i:BAND * i + BAND,
                      COL_ANCHORS[j] - 2:COL_ANCHORS[j] + BAND - 2]
            row.append(cell_symbol(color, int(np.count_nonzero(patch == color))))
        grid.append(row)
    return grid


def avatar_column(frame):
    """Column index of the avatar on its fixed band row, or None if gone."""
    f = pixels(frame)
    for j in range(GRID_COLS):
        if int(f[ROW_ANCHORS[AVATAR_ROW], COL_ANCHORS[j]]) in AVATAR_COLORS:
            return j
    return None


def moves_used(frame):
    """Actions consumed so far, read off the counter strip in row 63."""
    return int(np.count_nonzero(pixels(frame)[63]))


def cell_is_toggle(frame, i, j):
    """True if band cell (i, j) is a click-toggled block, hollow or solid.

    Its hollow form reads as floor, so the avatar rises through it; clicking it
    makes it solid again. That is the only way to stand still while walking
    sideways, since otherwise every action lifts the avatar through the ceiling.
    """
    color, _ = cell_fill(frame, i, j)
    return color in TOGGLE_COLORS


def find_symbol(grid, symbol, row=None):
    """All (i, j) cells carrying ``symbol``; optionally restricted to a row."""
    rows = range(GRID_ROWS) if row is None else (row,)
    return [(i, j) for i in rows for j in range(GRID_COLS) if grid[i][j] == symbol]


def band_shift(before, after):
    """Bands the camera scrolled between two frames, i.e. the height gained.

    The camera pins the avatar to ``AVATAR_ROW``, so a rise shows up only as
    the world sliding down. Scored by how many raw rows line up at each
    candidate offset, which tolerates the cells the avatar itself repainted.
    """
    a, b = pixels(before)[:63], pixels(after)[:63]
    best_hits, best = -1, 0
    for bands in range(GRID_ROWS):
        offset = BAND * bands
        rows = 63 - offset
        hits = int(np.count_nonzero((a[:rows] == b[offset:offset + rows]).all(axis=1)))
        if hits > best_hits:
            best_hits, best = hits, bands
    return best


# --- action legs ------------------------------------------------------

def click_cell(env, i, j):
    """Click band cell (i, j); removes a removable block, else burns a move."""
    env.step(CLICK, COL_ANCHORS[j], ROW_ANCHORS[i])


def act(env, action, log=None):
    """Apply one action tuple -- ``(key,)`` or ``(CLICK, x, y)`` -- and record it."""
    env.step(*action)
    if log is not None:
        log.append(action)


def run_actions(env, actions):
    """Replay a recorded action list, stopping early if the episode ends."""
    for action in actions:
        if env.terminal():
            return False
        env.step(*action)
    return True


def click_action(i, j):
    """The action tuple that clicks band cell (i, j)."""
    return (CLICK, COL_ANCHORS[j], ROW_ANCHORS[i])


def walk_to(env, target_col, log=None, max_steps=32, seal=False):
    """Walk along the avatar row to ``target_col``, clearing blocks in the way.

    Re-reads the frame every step because the world scrolls under the avatar.
    With ``seal``, a hollow toggle cell over the next column is clicked solid
    first, so the walk stays on one band instead of the avatar rising into
    whatever is above -- which is how a sideways detour survives a ceiling of
    hazards. Returns True only if the avatar is alive and on the target column.
    """
    for _ in range(max_steps):
        frame = env.frame()
        col = avatar_column(frame)
        if col is None:
            return False
        if col == target_col:
            return True
        direction = 1 if target_col > col else -1
        grid = band_grid(frame)
        ahead = grid[AVATAR_ROW][col + direction]
        if ahead in (WALL, PRIZE):
            return False
        if ahead == BLOCK:
            act(env, click_action(AVATAR_ROW, col + direction), log)
            continue
        if (seal and grid[AVATAR_ROW - 1][col + direction] == OPEN
                and cell_is_toggle(frame, AVATAR_ROW - 1, col + direction)):
            act(env, click_action(AVATAR_ROW - 1, col + direction), log)
            continue
        act(env, (RIGHT if direction > 0 else LEFT,), log)
    return False


def run_plan(env, plan, base_level=None):
    """Execute a plan of ('click', i, j) / ('step', action) items.

    Returns False as soon as the episode ends or a level above ``base_level``
    is completed. ``base_level`` defaults to the level in progress on entry.
    """
    if base_level is None:
        base_level = env.levels_completed
    for item in plan:
        if env.terminal():
            return False
        if item[0] == 'click':
            click_cell(env, item[1], item[2])
        else:
            env.step(item[1])
        if env.terminal() or env.levels_completed > base_level:
            return False
    return True


# --- planning legs ----------------------------------------------------

def _column_plan(grid, j, walk, out):
    """Append a climb plan for column ``j`` reached via prefix ``walk``."""
    cleared = set()
    i = AVATAR_ROW - 1
    while i >= 0 and grid[i][j] == BLOCK:
        cleared.add(i)
        i -= 1
    if not cleared and i >= 0 and grid[i][j] != OPEN:
        return  # solid immediately overhead: this column cannot be climbed
    gain = 0
    i = AVATAR_ROW - 1
    while i >= 0 and (grid[i][j] == OPEN or i in cleared):
        gain += 1
        i -= 1
    if gain <= 0:
        return
    plan = list(walk) + [('click', i, j) for i in sorted(cleared, reverse=True)]
    out.append((len(plan), gain, plan))


def climb_plans(grid, col):
    """Enumerate (cost, height_gain, plan) options from the avatar's column.

    Walking sideways is blocked by walls; a removable block in the way is
    cleared with a click first.
    """
    out = []
    for direction in (-1, 1):
        j = col
        walk = []
        while True:
            _column_plan(grid, j, walk, out)
            nj = j + direction
            if not 0 <= nj < GRID_COLS or grid[AVATAR_ROW][nj] == WALL:
                break
            if grid[AVATAR_ROW][nj] == PRIZE:
                break
            if grid[AVATAR_ROW][nj] == BLOCK:
                walk = walk + [('click', AVATAR_ROW, nj)]
            walk = walk + [('step', LEFT if direction < 0 else RIGHT)]
            j = nj
    return out


def plan_actions(plan):
    """Convert a ('click', i, j) / ('step', key) plan into action tuples."""
    return [click_action(item[1], item[2]) if item[0] == 'click' else (item[1],)
            for item in plan]


def walk_plan(grid, col, target_col):
    """Sideways plan from ``col`` to ``target_col``, or None if blocked."""
    direction = 1 if target_col > col else -1
    plan = []
    j = col
    while j != target_col:
        nj = j + direction
        if grid[AVATAR_ROW][nj] == WALL:
            return None
        if grid[AVATAR_ROW][nj] == BLOCK:
            plan.append(('click', AVATAR_ROW, nj))
        plan.append(('step', LEFT if direction < 0 else RIGHT))
        j = nj
    return plan


# --- composite legs ---------------------------------------------------

def climb_tower(env, move_budget=56, stop_on_prize=True):
    """Greedily gain height until the prize shows up or no column can rise.

    Each iteration picks the climb plan with the best height-per-move ratio,
    which is the dense progress measure that outpaces the rising floor.
    """
    base_level = env.levels_completed
    climbed = 0
    while not env.terminal() and env.levels_completed <= base_level:
        frame = env.frame()
        grid = band_grid(frame)
        col = avatar_column(frame)
        if col is None or moves_used(frame) >= move_budget:
            break
        if stop_on_prize and find_symbol(grid, PRIZE, row=AVATAR_ROW):
            break
        options = climb_plans(grid, col)
        if not options:
            break
        options.sort(key=lambda p: (-p[1] / p[0], p[0]))
        cost, gain, plan = options[0]
        if not run_plan(env, plan):
            break
        climbed += gain
    return climbed


def prize_actions(env):
    """Actions that walk the avatar onto the nearest prize on its own row.

    ``None`` when there is no such cell or the way there is blocked. The cell
    may equally be a hazard -- only the caller's clone can tell -- so this is a
    candidate, not a promise.
    """
    if env.terminal():
        return None
    frame = env.frame()
    grid = band_grid(frame)
    col = avatar_column(frame)
    if col is None:
        return None
    targets = find_symbol(grid, PRIZE, row=AVATAR_ROW)
    if not targets:
        return None
    targets.sort(key=lambda t: abs(t[1] - col))
    plan = walk_plan(grid, col, targets[0][1])
    return None if plan is None else plan_actions(plan)


def reach_prize(env):
    """Walk sideways onto a prize cell sharing the avatar's row."""
    base_level = env.levels_completed
    actions = prize_actions(env)
    if not actions:
        return False
    run_actions(env, actions)
    return env.levels_completed > base_level


def climb_to_prize(env, move_budget=56):
    """Full ascent: climb the tower, then step onto the prize at the top."""
    climb_tower(env, move_budget=move_budget)
    return reach_prize(env)


# --- search legs ------------------------------------------------------

def climb_macros(env):
    """Successor states of one climb decision, as ``(actions, clone)`` pairs.

    Per column and per walking style (plain, or sealing hollow toggle ceilings
    to stay level): walk there, and walk there then clear the cell overhead so
    the avatar rises. Successors that kill the avatar are dropped, so callers
    need no hazard model -- the clone reports the hazard.
    """
    base_level = env.levels_completed
    seen = set()
    onto = prize_actions(env)
    if onto:
        node = env.clone()
        run_actions(node, onto)
        seen.add(tuple(onto))
        yield list(onto), node
    for j in range(GRID_COLS):
        for seal in (False, True):
            walk = []
            node = env.clone()
            if not walk_to(node, j, walk, seal=seal):
                continue
            key = tuple(walk)
            if key in seen:
                continue
            seen.add(key)
            if walk:
                yield list(walk), node
            if band_grid(node.frame())[AVATAR_ROW - 1][j] != BLOCK:
                continue
            risen = node.clone()
            step = click_action(AVATAR_ROW - 1, j)
            risen.step(*step)
            if (avatar_column(risen.frame()) is None
                    and risen.levels_completed <= base_level):
                continue  # clone died here: the cell overhead is a trap
            yield walk + [step], risen


def _climb_key(env, height):
    frame = env.frame()
    return height, avatar_column(frame), tuple(''.join(r) for r in band_grid(frame))


def climb_search(env, max_expansions=800):
    """Best-first ascent maximising bands gained; backtracks out of trap columns.

    ``climb_tower`` is greedy on height-per-move and so walks into shafts whose
    only exit is fatal. This keeps a frontier ordered by accumulated height
    (the dense progress measure) with move count as the tie-break, so a dead
    end costs an expansion rather than the run. Returns the actions that
    complete the level, else the deepest-climbing actions found.
    """
    base_level = env.levels_completed
    tie = itertools.count()
    root = env.clone()
    queue = [(0, 0, next(tie), root, 0, [])]
    seen = {_climb_key(root, 0)}
    best = (0, [])
    for _ in range(max_expansions):
        if not queue:
            break
        _, _, _, node, height, actions = heapq.heappop(queue)
        for step, child in climb_macros(node):
            if child.levels_completed > base_level:
                return actions + step
            gained = height + band_shift(node.frame(), child.frame())
            key = _climb_key(child, gained)
            if key in seen:
                continue
            seen.add(key)
            if gained > best[0]:
                best = (gained, actions + step)
            heapq.heappush(queue, (-gained, moves_used(child.frame()),
                                   next(tie), child, gained, actions + step))
    return best[1]


def climb_by_search(env, max_expansions=800):
    """Plan an ascent on clones, then commit it on the real env."""
    base_level = env.levels_completed
    actions = climb_search(env, max_expansions=max_expansions)
    if not actions:
        return False
    run_actions(env, actions)
    return env.levels_completed > base_level

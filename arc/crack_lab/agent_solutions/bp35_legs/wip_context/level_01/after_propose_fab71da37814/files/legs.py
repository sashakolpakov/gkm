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

WALL, OPEN, BLOCK, AVATAR, PRIZE = '#', ' ', 'Y', 'A', '*'

LEFT, RIGHT, CLICK = 3, 4, 6


# --- observation legs -------------------------------------------------

def cell_symbol(color):
    """Classify one raw colour into the terrain alphabet."""
    color = int(color)
    if color in WALL_COLORS:
        return WALL
    if color in OPEN_COLORS:
        return OPEN
    if color in BLOCK_COLORS:
        return BLOCK
    if color in AVATAR_COLORS:
        return AVATAR
    return PRIZE


def band_grid(frame):
    """Quantise a frame onto the 6px band lattice as a symbolic grid."""
    return [[cell_symbol(frame[r][c]) for c in COL_ANCHORS] for r in ROW_ANCHORS]


def avatar_column(frame):
    """Column index of the avatar on its fixed band row, or None if gone."""
    row = band_grid(frame)[AVATAR_ROW]
    for j, sym in enumerate(row):
        if sym == AVATAR:
            return j
    return None


def moves_used(frame):
    """Actions consumed so far, read off the counter strip in row 63."""
    return sum(1 for v in frame[63] if int(v) != 0)


def find_symbol(grid, symbol, row=None):
    """All (i, j) cells carrying ``symbol``; optionally restricted to a row."""
    rows = range(GRID_ROWS) if row is None else (row,)
    return [(i, j) for i in rows for j in range(GRID_COLS) if grid[i][j] == symbol]


# --- action legs ------------------------------------------------------

def click_cell(env, i, j):
    """Click band cell (i, j); removes a removable block, else burns a move."""
    env.step(CLICK, COL_ANCHORS[j], ROW_ANCHORS[i])


def run_plan(env, plan):
    """Execute a plan of ('click', i, j) / ('step', action) items.

    Returns False as soon as the episode ends or the level is completed.
    """
    for item in plan:
        if env.terminal():
            return False
        if item[0] == 'click':
            click_cell(env, item[1], item[2])
        else:
            env.step(item[1])
        if env.terminal() or env.levels_completed:
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
    climbed = 0
    while not env.terminal() and not env.levels_completed:
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


def reach_prize(env):
    """Walk sideways onto a prize cell sharing the avatar's row."""
    if env.terminal() or env.levels_completed:
        return bool(env.levels_completed)
    frame = env.frame()
    grid = band_grid(frame)
    col = avatar_column(frame)
    if col is None:
        return False
    targets = find_symbol(grid, PRIZE, row=AVATAR_ROW)
    if not targets:
        return False
    targets.sort(key=lambda t: abs(t[1] - col))
    plan = walk_plan(grid, col, targets[0][1])
    if plan is None:
        return False
    run_plan(env, plan)
    return bool(env.levels_completed)


def climb_to_prize(env, move_budget=56):
    """Full ascent: climb the tower, then step onto the prize at the top."""
    climb_tower(env, move_budget=move_budget)
    return reach_prize(env)

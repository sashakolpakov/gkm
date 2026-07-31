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
from collections import deque

BAND = 6
GRID_ROWS = 10
GRID_COLS = 8
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


def band_shift(before, after):
    """Bands the camera scrolled between two frames, i.e. the height gained.

    The camera pins the avatar to ``AVATAR_ROW``, so a rise shows up only as
    the world sliding down. Scored by how many raw rows line up at each
    candidate offset, which tolerates the cells the avatar itself repainted.
    """
    rows_before = [tuple(int(v) for v in before[r]) for r in range(63)]
    rows_after = [tuple(int(v) for v in after[r]) for r in range(63)]
    best_hits, best = -1, 0
    for bands in range(GRID_ROWS):
        offset = BAND * bands
        hits = sum(1 for r in range(63 - offset)
                   if rows_before[r] == rows_after[r + offset])
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


def plan_and_commit(env, search_leg, **search_options):
    """Find an action route on clones, then commit it on the real environment."""
    base_level = env.levels_completed
    actions = search_leg(env, **search_options)
    if not actions:
        return False
    run_actions(env, actions)
    return env.levels_completed > base_level


def click_action(i, j):
    """The action tuple that clicks band cell (i, j)."""
    return (CLICK, COL_ANCHORS[j], ROW_ANCHORS[i])


def walk_to(env, target_col, log=None, max_steps=32):
    """Walk along the avatar row to ``target_col``, clearing blocks in the way.

    Re-reads the frame every step because the world scrolls under the avatar.
    Returns True only if the avatar is alive and standing on the target column.
    """
    for _ in range(max_steps):
        frame = env.frame()
        col = avatar_column(frame)
        if col is None:
            return False
        if col == target_col:
            return True
        direction = 1 if target_col > col else -1
        ahead = band_grid(frame)[AVATAR_ROW][col + direction]
        if ahead in (WALL, PRIZE):
            return False
        if ahead == BLOCK:
            act(env, click_action(AVATAR_ROW, col + direction), log)
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


def reach_prize(env):
    """Walk sideways onto a prize cell sharing the avatar's row."""
    base_level = env.levels_completed
    if env.terminal():
        return False
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
    run_plan(env, plan, base_level)
    return env.levels_completed > base_level


def climb_to_prize(env, move_budget=56):
    """Full ascent: climb the tower, then step onto the prize at the top."""
    climb_tower(env, move_budget=move_budget)
    return reach_prize(env)


# --- search legs ------------------------------------------------------

def climb_macros(env):
    """Successor states of one climb decision, as ``(actions, clone)`` pairs.

    Two macros per column: walk there, and walk there then clear the cell
    overhead so the avatar rises. Successors that kill the avatar are dropped,
    so callers need no hazard model -- the clone reports the hazard.
    """
    base_level = env.levels_completed
    for j in range(GRID_COLS):
        walk = []
        node = env.clone()
        if not walk_to(node, j, walk):
            continue
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
    root = env
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
    return plan_and_commit(
        env, climb_search, max_expansions=max_expansions
    )


def _cell_shape(frame, i, j):
    """Centre colour and its area in one band cell.

    Shape distinguishes a small hazard from its expanded safe form even when
    both use the same colour.
    """
    r0, c0 = BAND * i, 13 + BAND * j
    color = int(frame[ROW_ANCHORS[i]][COL_ANCHORS[j]])
    area = sum(1 for r in range(r0, r0 + BAND)
               for c in range(c0, c0 + BAND)
               if int(frame[r][c]) == color)
    return color, area


def local_hazard_climb_search(env, max_expansions=600):
    """Search a timed climb using movement and only nearby interactions.

    Some towers use shape-changing hazards: interacting with a small hazard
    makes it safe to pass, while distant clicks merely waste the tight clock.
    The observational key retains each cell's shape as well as climbed height
    and elapsed moves.  Replaying paths from one root avoids retaining a large
    tree of nested clones.
    """
    base_level = env.levels_completed
    tie = itertools.count()
    root = env.clone()

    def key(node, height):
        frame = node.frame()
        return (height, moves_used(frame), avatar_column(frame),
                tuple(_cell_shape(frame, i, j)
                      for i in range(GRID_ROWS) for j in range(GRID_COLS)))

    def local_actions(node):
        frame = node.frame()
        col = avatar_column(frame)
        out = [(LEFT,), (RIGHT,)]
        if col is None:
            return out
        for i in (AVATAR_ROW - 1, AVATAR_ROW):
            for j in range(max(0, col - 1), min(GRID_COLS, col + 2)):
                color = int(frame[ROW_ANCHORS[i]][COL_ANCHORS[j]])
                if color not in OPEN_COLORS + WALL_COLORS + AVATAR_COLORS:
                    out.append(click_action(i, j))
        return out

    queue = [(0, 0, next(tie), 0, ())]
    seen = {key(root, 0)}
    for _ in range(max_expansions):
        if not queue:
            break
        _, _, _, height, path = heapq.heappop(queue)
        node = root.clone()
        run_actions(node, path)
        for action in local_actions(node):
            child = node.clone()
            child.step(*action)
            if child.levels_completed > base_level:
                return list(path + (action,))
            if child.terminal() or avatar_column(child.frame()) is None:
                continue
            gained = height + band_shift(node.frame(), child.frame())
            state = key(child, gained)
            if state in seen:
                continue
            seen.add(state)
            new_path = path + (action,)
            heapq.heappush(queue, (-gained, len(new_path), next(tie),
                                   gained, new_path))
    return []


def climb_local_hazards(env, max_expansions=600):
    """Plan a shape-changing hazard climb on clones, then commit it."""
    return plan_and_commit(
        env, local_hazard_climb_search, max_expansions=max_expansions
    )


def gravity_room_search(
    env, max_states=900, max_depth=48, remote_supports=False
):
    """Find a route through rooms controlled by gravity toggles and supports.

    Movement is always considered.  Global gravity toggles (colour 8) can
    change the whole room, while removable supports (14) are useful only when
    aligned with the avatar and shape-changing hazards (15) only when nearby.
    Reconstructing each state from one shallow root avoids chains of costly
    nested clones.
    """
    base_level = env.levels_completed
    root = env.clone()

    def avatar_cell(frame):
        for i, r in enumerate(ROW_ANCHORS):
            for j, c in enumerate(COL_ANCHORS):
                if int(frame[r][c]) in AVATAR_COLORS:
                    return i, j
        return None

    def choices(node):
        frame = node.frame()
        avatar = avatar_cell(frame)
        out = [(LEFT,), (RIGHT,)]
        if avatar is None:
            return out
        ai, aj = avatar
        for i, y in enumerate(ROW_ANCHORS):
            for j, x in enumerate(COL_ANCHORS):
                color = int(frame[y][x])
                if color == 8:
                    out.append((CLICK, x, y))
                elif color == 14 and (
                    remote_supports or (j == aj and abs(i - ai) <= 1)
                ):
                    out.append((CLICK, x, y))
                elif color == 15 and abs(i - ai) <= 1 and abs(j - aj) <= 1:
                    out.append((CLICK, x, y))
        return out

    def state_key(node):
        frame = node.frame()
        visible = bytes(int(value) for row in frame[:63] for value in row)
        return visible, moves_used(frame) % 2

    def reconstruct(path):
        node = root.clone()
        run_actions(node, path)
        return node

    queue = deque([()])
    seen = {state_key(root)}
    while queue and len(seen) < max_states:
        path = queue.popleft()
        node = reconstruct(path)
        if len(path) >= max_depth or node.terminal():
            continue
        for action in choices(node):
            new_path = path + (action,)
            child = reconstruct(new_path)
            if child.levels_completed > base_level:
                return list(new_path)
            if child.terminal() or avatar_cell(child.frame()) is None:
                continue
            key = state_key(child)
            if key in seen:
                continue
            seen.add(key)
            queue.append(new_path)
    return []


def solve_gravity_rooms(env, max_states=900, max_depth=48):
    """Plan context-safe gravity/support transitions, then commit the route."""
    return plan_and_commit(
        env, gravity_room_search, max_states=max_states, max_depth=max_depth
    )


def probe_level_five(env):
    """Compact exploratory report for the wider gravity-room layout."""
    def lattice(frame):
        return tuple(
            tuple(int(frame[y][x]) for x in range(15, 64, 6))
            for y in ROW_ANCHORS
        )

    base = env.frame()
    print("L5 lattice")
    for row in lattice(base):
        print(row)
    for y in ROW_ANCHORS:
        for x in range(15, 64, 6):
            color = int(base[y][x])
            if color not in (7, 8, 14, 15):
                continue
            clone = env.clone()
            clone.step(CLICK, x, y)
            before_grid = lattice(base)
            after_grid = lattice(clone.frame())
            changed = [
                (i, j, before_grid[i][j], after_grid[i][j])
                for i in range(len(before_grid))
                for j in range(len(before_grid[i]))
                if before_grid[i][j] != after_grid[i][j]
            ]
            print(
                "click", (y, x, color), "alive", not clone.terminal(),
                "level", clone.levels_completed, "changed", changed,
            )
    def avatar(frame):
        for i, y in enumerate(ROW_ANCHORS):
            for j, x in enumerate(range(15, 64, 6)):
                if int(frame[y][x]) in AVATAR_COLORS:
                    return i, j
        return None

    toggle = (CLICK, 51, 39)
    for direction, key in (("L", LEFT), ("R", RIGHT)):
        for count in range(7):
            clone = env.clone()
            path = [(key,)] * count
            run_actions(clone, path)
            before_toggle = avatar(clone.frame())
            if not clone.terminal():
                act(clone, toggle)
            print(
                "toggle_after", direction, count, "from", before_toggle,
                "to", avatar(clone.frame()), "alive", not clone.terminal(),
                "level", clone.levels_completed,
            )
    phase = env.clone()
    run_actions(phase, [(RIGHT,), (RIGHT,), (RIGHT,), toggle])
    print("phase1", "avatar", avatar(phase.frame()))
    for row in lattice(phase.frame()):
        print(row)
    for action in ((LEFT,), (RIGHT,)):
        clone = phase.clone()
        act(clone, action)
        print(
            "phase1_step", action, "avatar", avatar(clone.frame()),
            "alive", not clone.terminal(),
        )
    phase_frame = phase.frame()
    for i, y in enumerate(ROW_ANCHORS):
        for j, x in enumerate(range(15, 64, 6)):
            if int(phase_frame[y][x]) not in (8, 14, 15):
                continue
            clone = phase.clone()
            clone.step(CLICK, x, y)
            print(
                "phase1_click", (i, j, int(phase_frame[y][x])),
                "avatar", avatar(clone.frame()), "alive", not clone.terminal(),
            )
    def special(frame):
        return [
            (i, j, int(frame[y][x]))
            for i, y in enumerate(ROW_ANCHORS)
            for j, x in enumerate(range(15, 64, 6))
            if int(frame[y][x]) in (7, 8, 14, 15)
        ]

    support4 = (CLICK, 39, 33)
    support5 = (CLICK, 45, 33)
    tails = [
        [support4],
        [support4, (LEFT,)],
        [support4, (RIGHT,)],
        [support4, (RIGHT,), (LEFT,)],
        [support5, (RIGHT,)],
        [support4, support5, (RIGHT,)],
    ]
    for tail in tails:
        clone = phase.clone()
        run_actions(clone, tail)
        print(
            "phase1_tail", tail, "avatar", avatar(clone.frame()),
            "alive", not clone.terminal(), "special", special(clone.frame()),
        )

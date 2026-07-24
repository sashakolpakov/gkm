# Generated solver source index

Use line ranges to inspect only definitions relevant to the current level.

## players.py
- L5--8 `def play_level_1(env): # Level 1 is a toggle-tile board (framed 3x3): click a subset of tiles to # reach the hidden target configuration that completes the level.`; calls: solve_toggle_board
- L11--16 `def play_level_3(env): # Level 3 is a 'pattern-key' toggle board: a lattice of solid blocks with a # few decorated 'pattern' blocks whose 3x3 mini-keys dictate which # neighbouring blocks to toggle. No subset search is feasible (23 tiles), `; calls: solve_pattern_key_board
- L19--26 `def play_level_4(env): # Level 4 is the pattern-key board family again, but the blocks are now # THREE-STATE cells that cycle 9->8->12 on each click (level 3's blocks were # two-state). The mini-keys still decode a target: each marked neigh`; calls: solve_multistate_key_board
- L29--33 `def play_level_2(env): # Level 2 is the SAME toggle-tile board family, only a wider 3-row grid # (~13 tiles). No new skill is needed: the shared solve_toggle_board leg # probes for the tiles and subset-searches the completing clicks on clon`; calls: solve_toggle_board

## legs.py
- L9--12 `def _changed_cells(base, frame, row_limit=62):` — Cells that differ between two frames, ignoring a bottom status bar.; calls: int
- L15--42 `def discover_toggle_tiles(env, step=2, row_limit=62):` — Probe coordinate clicks on a clone and return one representative (x, y); calls: _changed_cells, max, min, range, sorted
- L45--65 `def search_toggle_solution(env, tiles=None, max_tiles=16):` — Find a subset of tile clicks (each tile toggled at most once) that raises; calls: discover_toggle_tiles, len, range
- L68--73 `def commit_plan(env, plan, apply_fn):` — Replay a planned action sequence on the real env, one action at a time.; calls: apply_fn
- L76--89 `def plan_and_commit(env, planner, apply_fn):` — Higher-order leg: search for a plan on clones, then commit it for real.; calls: commit_plan, planner
- L92--95 `def _click_xy(env, xy):` — Apply a single coordinate click (x, y) via the ARC click action.
- L98--108 `def solve_click_board(env, planner):` — Higher-order leg for coordinate-click boards: plan a set of (x, y) clicks; calls: plan_and_commit
- L111--115 `def solve_toggle_board(env):` — Discover the toggle board, search for a completing click subset, and; calls: solve_click_board
- L118--170 `def _grid_of_blocks(frame, row_limit=62):` — Perceive a regular grid of solid square blocks over a background.; calls: Counter, enumerate, len, range, set, sorted
- L173--187 `def pattern_key_grid(env):` — Perceive a pattern-key board and enforce its shape in ONE place.; calls: _grid_of_blocks
- L190--197 `def read_mini_key(blk):` — Read a decorated block's 3x3 mini-key: sample the centre pixel of each of; calls: int, range
- L200--211 `def mark_color(patterns):` — The 'mark' colour of a set of mini-keys: the least-common non-centre cell; calls: Counter, range, read_mini_key
- L214--218 `def block_click_xy(slots, idx, s):` — The coordinate-click (x, y) that targets the centre of block `idx` on a
- L221--224 `def block_center_color(frame, slots, idx, s):` — The solid colour currently shown by block `idx` (its centre pixel).; calls: int
- L227--242 `def key_marked_neighbours(patterns, uniform, mark):` — Yield (neighbour_idx, key_centre_colour) for every marked neighbour of; calls: range, read_mini_key
- L245--279 `def discover_pattern_key_clicks(env):` — Planner for the ft09 'pattern-key' toggle board.; calls: Counter, block_center_color, block_click_xy, mark_color, pattern_key_grid, range, read_mini_key, set, sorted
- L282--286 `def solve_pattern_key_board(env):` — Decode the pattern-key board's target from the frame and commit the; calls: solve_click_board
- L289--348 `def discover_multistate_key_clicks(env):` — Planner for a MULTI-STATE pattern-key toggle board.; calls: block_center_color, block_click_xy, build, enumerate, iter, key_marked_neighbours, len, mark_color, next, pattern_key_grid, +2
- L351--355 `def solve_multistate_key_board(env):` — Decode a multi-state pattern-key board (cells cycle through k colours) and; calls: solve_click_board

## perception.py
- L23--36 `class Blob:`; calls: dataclass
- L39--40 `def arr(frame) -> np.ndarray:`
- L43--45 `def color_counts(frame) -> Dict[int, int]:`; calls: arr, int, zip
- L48--79 `def connected_components(frame, colors: Optional[Iterable[int]] = None, min_area: int = 1) -> List[Blob]:`; calls: Blob, arr, int, len, max, min, range, sorted, sum
- L82--89 `def block_signatures(frame, cell: int = 4) -> Dict[Tuple[int, int], Tuple[int, ...]]:` — Partition a frame into fixed cells and return each cell's color signature.; calls: arr, int, range, sorted, tuple
- L92--110 `def object_candidates(frame, cell: int = 4, min_area: int = 4) -> List[dict]:` — A compact, game-agnostic object list from color components and cell signatures.; calls: arr, block_signatures, connected_components
- L113--123 `def frame_delta(before, after) -> dict:`; calls: arr, int, len, zip
- L126--133 `def action_deltas(env, actions: Sequence[int] = ACTIONS) -> Dict[int, dict]:`; calls: arr, frame_delta, int
- L136--142 `def replay(env, actions: Sequence[int]):`; calls: int
- L145--153 `def path_result(env, actions: Sequence[int]) -> dict:`; calls: bool, color_counts, int, len, object_candidates, replay
- L156--161 `def changed_signature(env, actions: Sequence[int], cell: int = 4):`; calls: block_signatures, replay, set, sorted
- L164--186 `def bounded_bfs(env, goal_fn, actions: Sequence[int] = (UP, DOWN, LEFT, RIGHT, USE), key_fn=None, max_states: int = 20000, max_depth: int = 80):` — Generic clone BFS over observational keys. Use small max_states first.; calls: arr, deque, goal_fn, int, key_fn, len
- L189--190 `def level_goal(base_level: int):`

## solve.py
- L3--13 `def solve(env): # dispatch to the per-level player for the current level, in a loop`; calls: fn, getattr

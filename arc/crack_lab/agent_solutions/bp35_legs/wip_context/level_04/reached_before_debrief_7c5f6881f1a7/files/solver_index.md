# Generated solver source index

Use line ranges to inspect only definitions relevant to the current level.

## players.py
- L5--7 `def play_level_1(env): # Ascend the block tower, then step onto the prize revealed at its top.`; calls: climb_to_prize
- L10--13 `def play_level_2(env): # Taller tower with fatal ceiling hazards, so the greedy ascent walks into # dead-end shafts: plan the climb by search instead, then commit it.`; calls: climb_by_search
- L16--18 `def play_level_3(env): # Nearby hazards must be made safe before each timed crossing.`; calls: climb_local_hazards

## legs.py
- L48--59 `def cell_symbol(color):` — Classify one raw colour into the terrain alphabet.; calls: int
- L62--64 `def band_grid(frame):` — Quantise a frame onto the 6px band lattice as a symbolic grid.; calls: cell_symbol
- L67--73 `def avatar_column(frame):` — Column index of the avatar on its fixed band row, or None if gone.; calls: band_grid, enumerate
- L76--78 `def moves_used(frame):` — Actions consumed so far, read off the counter strip in row 63.; calls: int, sum
- L81--84 `def find_symbol(grid, symbol, row=None):` — All (i, j) cells carrying ``symbol``; optionally restricted to a row.; calls: range
- L87--103 `def band_shift(before, after):` — Bands the camera scrolled between two frames, i.e. the height gained.; calls: int, range, sum, tuple
- L108--110 `def click_cell(env, i, j):` — Click band cell (i, j); removes a removable block, else burns a move.
- L113--117 `def act(env, action, log=None):` — Apply one action tuple -- ``(key,)`` or ``(CLICK, x, y)`` -- and record it.
- L120--126 `def run_actions(env, actions):` — Replay a recorded action list, stopping early if the episode ends.
- L129--131 `def click_action(i, j):` — The action tuple that clicks band cell (i, j).
- L134--155 `def walk_to(env, target_col, log=None, max_steps=32):` — Walk along the avatar row to ``target_col``, clearing blocks in the way.; calls: act, avatar_column, band_grid, click_action, range
- L158--175 `def run_plan(env, plan, base_level=None):` — Execute a plan of ('click', i, j) / ('step', action) items.; calls: click_cell
- L180--197 `def _column_plan(grid, j, walk, out):` — Append a climb plan for column ``j`` reached via prefix ``walk``.; calls: len, list, set, sorted
- L200--221 `def climb_plans(grid, col):` — Enumerate (cost, height_gain, plan) options from the avatar's column.; calls: _column_plan
- L224--237 `def walk_plan(grid, col, target_col):` — Sideways plan from ``col`` to ``target_col``, or None if blocked.
- L242--266 `def climb_tower(env, move_budget=56, stop_on_prize=True):` — Greedily gain height until the prize shows up or no column can rise.; calls: avatar_column, band_grid, climb_plans, find_symbol, moves_used, run_plan
- L269--287 `def reach_prize(env):` — Walk sideways onto a prize cell sharing the avatar's row.; calls: abs, avatar_column, band_grid, find_symbol, run_plan, walk_plan
- L290--293 `def climb_to_prize(env, move_budget=56):` — Full ascent: climb the tower, then step onto the prize at the top.; calls: climb_tower, reach_prize
- L298--321 `def climb_macros(env):` — Successor states of one climb decision, as ``(actions, clone)`` pairs.; calls: avatar_column, band_grid, click_action, list, range, walk_to
- L324--326 `def _climb_key(env, height):`; calls: avatar_column, band_grid, tuple
- L329--360 `def climb_search(env, max_expansions=800):` — Best-first ascent maximising bands gained; backtracks out of trap columns.; calls: _climb_key, band_shift, climb_macros, moves_used, next, range
- L363--370 `def climb_by_search(env, max_expansions=800):` — Plan an ascent on clones, then commit it on the real env.; calls: climb_search, run_actions
- L373--384 `def _cell_shape(frame, i, j):` — Centre colour and its area in one band cell.; calls: int, range, sum
- L387--442 `def local_hazard_climb_search(env, max_expansions=600):` — Search a timed climb using movement and only nearby interactions.; calls: _cell_shape, avatar_column, band_shift, click_action, int, key, len, list, local_actions, max, +6
- L445--452 `def climb_local_hazards(env, max_expansions=600):` — Plan a shape-changing hazard climb on clones, then commit it.; calls: local_hazard_climb_search, run_actions

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
- L189--229 `def bounded_replay_bfs(env, goal_fn, action_fn, key_fn=None, max_states: int = 20000, max_depth: int = 80):` — Path-only BFS for games whose deep Arena clones become expensive.; calls: action_fn, arr, deque, goal_fn, int, isinstance, key_fn, len, reconstruct
- L232--233 `def level_goal(base_level: int):`

## solve.py
- L3--13 `def solve(env): # dispatch to the per-level player for the current level, in a loop`; calls: fn, getattr

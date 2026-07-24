# Generated solver source index

Use line ranges to inspect only definitions relevant to the current level.

## players.py
- L5--9 `def play_level_1(env): # g50t level 1: move the avatar to the goal region, using USE to open the # gate that unlocks the far side of the maze. Fully handled by the general # unlock-then-reach leg (bounded-depth, grow-pruned toggle search).`; calls: solve_unlock_reach
- L12--19 `def play_level_2(env): # g50t level 2: same "reach the goal chamber" objective, but the avatar's # region is only linked to the chamber through gates that open remotely when # the avatar walks over wall segments; a USE resets the avatar to `; calls: solve_unlock_macro
- L22--28 `def play_level_3(env): # g50t level 3: another staged-unlock maze of the same family as level 2 -- # the goal only becomes movement-reachable after chaining several # walk-somewhere-then-USE openings. No new skill is needed: the SAME # gene`; calls: solve_unlock_macro
- L31--37 `def play_level_4(env): # g50t level 4: yet another staged-unlock maze of the level-2/3 family -- # the goal chamber only becomes movement-reachable after chaining several # walk-somewhere-then-USE openings. No new skill: the SAME general US`; calls: solve_unlock_macro
- L40--44 `def play_level_5(env): # This configuration has intermediate commits whose hidden effect appears # only on the next special surface. The frontier leg supplies the needed # bounded memory while still steering by movement reachability.`; calls: solve_frontier_unlock

## legs.py
- L32--40 `def clone_after(env, actions):` — Return a fresh clone of `env` with `actions` (an action or iterable of; calls: isinstance
- L46--66 `def _components(mask):` — 4-connected components of a boolean mask -> list of (top_row, top_col, cells).; calls: min, range
- L69--78 `def _shape_comps(frame, bg):` — All non-background components as (color, shape_sig, top_left, cells).; calls: _components, frozenset, int
- L81--104 `def avatar_tl(env):` — Top-left (row, col) of the avatar in env's current frame. The avatar is; calls: _shape_comps, any, clone_after, int
- L118--142 `def _reach_bfs(start, locate):` — BFS over avatar POSITIONS in the current (fixed) gate configuration.; calls: clone_after, deque, int, locate
- L145--147 `def _move_explore(start_env):` — Reach-BFS using the translation-based avatar detector (robust, slow).; calls: _reach_bfs
- L154--179 `def plan_unlock_reach(env, max_toggles=2):` — Return a concrete action list that raises levels_completed, or None.; calls: _move_explore, _use_macros, len, rec
- L182--187 `def run_path(env, path):` — Commit a planned action list on the real env, stopping if it terminates.
- L196--204 `def plan_and_commit(env, planner, **kwargs):` — Run `planner` on a clone of `env`; if it yields a non-empty action list,; calls: clone_after, planner, run_path
- L207--210 `def solve_unlock_reach(env, max_toggles=2):` — Leg: plan an unlock-then-reach route and commit it. Returns True if a; calls: plan_and_commit
- L223--232 `def _avatar_pos(frame):` — Top-left (row, col) of the avatar = largest compact color-9 blob that is; calls: _components, len
- L235--237 `def fast_reach(start):` — Reach-BFS using the clone-cheap color-9-blob avatar reader (fast).; calls: _avatar_pos, _reach_bfs
- L248--253 `def _use_macro(node, pos, path, reach_fn):` — Apply one walk-to-position-then-USE transition and probe its result.; calls: clone_after, reach_fn
- L256--267 `def _use_macros(node, reach, reach_fn):` — Yield (macro, child_env, child_reward_path, child_reach) for every; calls: _use_macro, len, sorted
- L284--309 `def plan_unlock_macro(env, max_expand=400, time_limit=550):` — Return a concrete winning action list, or None. Plans on clones only.; calls: _use_macros, fast_reach, frozenset, len
- L312--316 `def solve_unlock_macro(env, max_expand=400, time_limit=550):` — Leg: plan a USE-macro unlock-then-reach route and commit it. Returns; calls: plan_and_commit
- L330--339 `def _special_frontier(reach, frame):` — Reach entries whose 5x5 avatar footprint overlaps a special surface.; calls: len, sorted
- L342--399 `def plan_frontier_unlock(env, max_stages=10, max_stalls=2):` — Plan staged USE commits by maximizing movement reachability.; calls: _special_frontier, _use_macro, fast_reach, frozenset, int, len, max, range, set, sorted
- L402--405 `def solve_frontier_unlock(env, max_stages=10, max_stalls=2):` — Leg: plan and commit a special-surface frontier unlock route.; calls: plan_and_commit

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

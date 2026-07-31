# Generated solver source index

Use line ranges to inspect only definitions relevant to the current level.

## players.py
- L5--6 `def play_level_1(env):`; calls: follow_diagonal_lattice_to_ring
- L9--10 `def play_level_2(env):`; calls: merge_equal_squares_and_deliver_to_ring
- L13--15 `def play_level_3(env):`; calls: deliver_remaining_square_via_diagonal_detour, merge_equal_squares_and_deliver_to_ring
- L18--19 `def play_level_4(env):`; calls: merge_equal_squares_around_moving_cutter

## legs.py
- L9--37 `def follow_diagonal_lattice_to_ring(env, step=6, max_moves=12):` — Move the playfield avatar diagonally toward the ringed target.; calls: connected_components, int, max, min, range, round
- L40--112 `def merge_equal_squares_and_deliver_to_ring(env, step=6, max_moves=32):` — Merge equal playfield squares, then carry the result into its ring.; calls: abs, anchor, click, connected_components, enumerate, int, len, max, min, playfield_squares, +1
- L115--186 `def deliver_remaining_square_via_diagonal_detour(env, step=6, max_moves=16):` — Route a remaining diagonal-moving square around occupied target rings.; calls: abs, click, connected_components, int, len, max, pieces, rings, round, sum
- L189--419 `def merge_equal_squares_around_moving_cutter( env, step=6, max_states=5000, max_depth=30 ):` — Plan collision-free equal-square merges, then deliver into the ring.; calls: abs, anchor, any, avatar_center, candidate_actions, connected_components, enumerate, int, len, list, +12

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

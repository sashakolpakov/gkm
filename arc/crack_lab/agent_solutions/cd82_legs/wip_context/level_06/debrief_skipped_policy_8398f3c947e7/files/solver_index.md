# Generated solver source index

Use line ranges to inspect only definitions relevant to the current level.

## players.py
- L5--6 `def play_level_1(env):`; calls: move_vessel_below_and_apply
- L9--10 `def play_level_2(env):`; calls: apply_current_then_select_and_apply_southeast
- L13--23 `def play_level_3(env):`; calls: apply_west_north_east_north_layers_then_payload
- L26--36 `def play_level_4(env):`; calls: apply_northwest_southeast_west_layers_then_west_payload
- L39--49 `def play_level_5(env):`; calls: apply_north_southwest_southeast_layers_then_north_payload

## legs.py
- L5--8 `def move_vessel_below_and_apply(env):` — Roll the active vessel below the work tile, then apply its contents.
- L11--16 `def apply_current_then_select_and_apply_southeast(env, selector_x, selector_y):` — Apply the current top stamp, then select and apply a southeast stamp.
- L19--44 `def apply_west_north_east_north_layers_then_payload( env, west_selector, north_selector, east_selector, payload_selector, selector_y, payload_x, payload_y):` — Paint W/NW/E/NW layers, return north, then apply the top payload.
- L47--66 `def apply_northwest_southeast_west_layers_then_west_payload( env, northwest_selector, southeast_selector, west_selector, payload_selector, selector_y, payload_x, payload_y):` — Paint NW/SE/W layers, then apply the carried payload from the west.
- L69--89 `def apply_north_southwest_southeast_layers_then_north_payload( env, north_selector, southwest_selector, southeast_selector, payload_selector, selector_y, payload_x, payload_y):` — Paint N/SW/SE layers, return north, then apply the carried payload.

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

# Generated solver source index

Use line ranges to inspect only definitions relevant to the current level.

## players.py
- L5--6 `def play_level_1(env):`; calls: align_lower_cyan_tip_with_upper_notch
- L9--10 `def play_level_2(env):`; calls: align_lower_cyan_tip_with_upper_notch
- L13--14 `def play_level_3(env):`; calls: relay_height_between_adjacent_reservoirs
- L17--18 `def play_level_4(env):`; calls: cross_pressure_gates_then_align_height
- L21--22 `def play_level_5(env):`; calls: cross_horizontal_gates_then_align_opposing_markers
- L25--27 `def play_level_6(env):`; calls: align_marker_pair_with_pressure_controls, cross_pressure_gates_then_align_height

## legs.py
- L9--82 `def align_lower_cyan_tip_with_upper_notch( env, max_presses=12, marker_color=11, max_states=300 ):` — Use visible controls to horizontally align a lower marker to an upper notch.; calls: abs, arr, bounded_replay_bfs, connected_components, len, marker_distance, range
- L85--164 `def relay_height_between_adjacent_reservoirs(env, max_presses=32):` — Relay height leftward until every small same-color marker pair is level.; calls: abs, all, arr, connected_components, gaps, int, len, min, sum
- L167--274 `def cross_pressure_gates_then_align_height( env, marker_color=11, closed_gate_color=1, active_gate_colors=(12, 13, 14, 15), max_stages=10, max_states=700, max_depth=16, ):` — Move a marked platform through pressure gates, then align its height.; calls: abs, actions, arr, connected_components, deque, improved, int, len, marker_pair, max, +4
- L277--403 `def cross_horizontal_gates_then_align_opposing_markers( env, marker_colors=(11, 14), closed_gate_color=1, active_gate_colors=(12, 13, 14, 15), max_stages=14, max_states=2500, max_depth=24, ):` — Relay opposing marked platforms across horizontal pressure gates.; calls: abs, actions, arr, connected_components, deque, enumerate, improved, int, len, marker_pair, +6
- L406--475 `def align_marker_pair_with_pressure_controls( env, marker_color=11, active_gate_colors=(12, 13, 14, 15), max_stages=16, max_states=500, max_depth=12, ):` — Align an unequal marker pair using visible pressure controls and gates.; calls: abs, actions, arr, connected_components, deque, int, len, marker_gap, max, min, +2

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

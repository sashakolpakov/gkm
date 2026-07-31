# Generated solver source index

Use line ranges to inspect only definitions relevant to the current level.

## players.py
- L5--13 `def play_level_1(env):`; calls: traverse_two_stage_bridge_chain
- L16--28 `def play_level_2(env):`; calls: traverse_revealed_three_stage_bridge_chain
- L31--46 `def play_level_3(env):`; calls: traverse_teleport_revealed_exit_chain
- L49--68 `def play_level_4(env):`; calls: traverse_synchronized_builder_teleport_chain
- L71--86 `def play_level_5(env):`; calls: clear_level_5_platform_transfer
- L89--90 `def play_level_5(env):`; calls: clear_level_5_platform_transfer

## legs.py
- L5--9 `def walk_segments(env, segments):` — Walk a sequence of ``(direction, distance)`` path segments.; calls: range
- L12--14 `def toggle_control(env, point):` — Click a coordinate control at an ``(x, y)`` point.
- L17--31 `def traverse_two_stage_bridge_chain( env, lower_control, upper_control, entry_segments, pivot_segments, exit_segments):` — Cross two reconfigurable bridges without stranding the avatar.; calls: toggle_control, walk_segments
- L34--53 `def traverse_revealed_three_stage_bridge_chain( env, first_control, second_control, revealed_control, first_pivot_segments, reveal_segments, retreat_segments, return_to_first_segments, final_pivot_segments, exit_segments):` — Reveal and cross a third bridge beyond two reconfigurable bridges.; calls: toggle_control, walk_segments
- L56--83 `def traverse_teleport_revealed_exit_chain( env, bridge_control, connector_control, teleport_control, revealed_exit_control, initial_approach_segments, first_crossing_segments, connector_entry_segments, teleport_approach_segments, endpoint_s` — Cross reconfigurable bridges, teleport, and use a revealed exit.; calls: toggle_control, walk_segments
- L86--119 `def traverse_synchronized_builder_teleport_chain( env, bridge_control, shuttle_control, teleport_control, initial_approach_segments, upper_build_direction, upper_build_phases, upper_crossing_segments, bridge_restore_phases, shuttle_completi` — Traverse paired incremental bridges joined by a remote teleporter.; calls: range, toggle_control, walk_segments
- L122--184 `def clear_level_5_platform_transfer( env, builder_control=(46, 22), crossing_control=(56, 22), bridge_control=(52, 42), teleport_control=(52, 46), platform_up_control=(50, 28), platform_right_control=(56, 28), platform_rotate_control=(51, 3` — Route the movable assembly through both docks and the lower rotator.; calls: range, toggle_control, walk_segments

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

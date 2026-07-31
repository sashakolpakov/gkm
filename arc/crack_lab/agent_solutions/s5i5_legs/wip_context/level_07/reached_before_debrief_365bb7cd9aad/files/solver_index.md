# Generated solver source index

Use line ranges to inspect only definitions relevant to the current level.

## players.py
- L5--6 `def play_level_1(env):`; calls: align_slider_tips_to_hollow_targets
- L9--10 `def play_level_2(env):`; calls: move_articulated_marker_around_barrier
- L13--14 `def play_level_3(env):`; calls: dock_crossed_sliders_through_coupled_barriers
- L17--18 `def play_level_4(env):`; calls: extend_shared_marker_through_staged_crossbars
- L21--22 `def play_level_5(env):`; calls: thread_coupled_marker_through_reconfigurable_frame
- L25--26 `def play_level_6(env):`; calls: dock_three_link_arm_through_partitioned_chamber

## legs.py
- L7--13 `def click_coordinate(env, x, y, times=1, action=6):` — Click one screen coordinate repeatedly, stopping after level progress.; calls: int, range
- L16--91 `def align_slider_tips_to_hollow_targets(env, marker_color=13, action=6):` — Align orthogonal colored slider tips with same-axis hollow targets.; calls: abs, all, arr, click_coordinate, connected_components, int, len, min, range, round, +2
- L94--219 `def move_articulated_marker_around_barrier(env, marker_color=13, action=6):` — Use paired limb controls to carry a marker around a blocking wall.; calls: advance, all, arr, can_move, click_coordinate, connected_components, int, len, marker_and_target, range, +4
- L222--291 `def dock_crossed_sliders_through_coupled_barriers( env, horizontal_slider_color=10, vertical_slider_color=7, upper_limb_color=8, upper_joint_color=9, lower_limb_color=12, lower_joint_color=14, action=6, ):` — Stage two coupled barriers so crossed marker sliders can dock.; calls: click_coordinate, connected_components, int, len, press, range, round
- L294--334 `def extend_shared_marker_through_staged_crossbars( env, panel_color=2, action=6 ):` — Advance a shared marker while staging long and short crossbars.; calls: abs, click_coordinate, connected_components, int, len, min, press, range, round
- L337--378 `def thread_coupled_marker_through_reconfigurable_frame( env, panel_color=2, action=6 ):` — Stage four coupled controls to thread a marker through a moving frame.; calls: click_coordinate, connected_components, int, len, press, round
- L381--427 `def dock_three_link_arm_through_partitioned_chamber( env, panel_color=2, action=6 ):` — Reorient and extend a three-link arm through a partitioned chamber.; calls: click_coordinate, connected_components, len, press_coarse, press_fine, sorted

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

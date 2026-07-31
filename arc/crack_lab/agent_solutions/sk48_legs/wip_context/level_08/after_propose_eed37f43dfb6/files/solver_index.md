# Generated solver source index

Use line ranges to inspect only definitions relevant to the current level.

## players.py
- L5--19 `def play_level_1(env): # Collect the requested 8 first.`; calls: extend_tether, move_vertical_lanes, retract_tether
- L22--28 `def play_level_2(env):`; calls: reverse_row_train
- L31--32 `def play_level_3(env):`; calls: weave_vertical_four_train
- L35--36 `def play_level_4(env):`; calls: unweave_horizontal_pairs_to_vertical_heads
- L39--41 `def play_level_5(env):`; calls: interleave_lower_lane_train, stage_split_rows_for_lower_interleave
- L44--46 `def play_level_6(env):`; calls: deal_vertical_triplet_to_top_collector, shuttle_row_tokens_under_vertical_train
- L49--50 `def play_level_7(env):`; calls: weave_shared_center_cross

## legs.py
- L10--15 `def move_vertical_lanes(env, direction, lanes):` — Move the avatar and any attached horizontal train by whole lanes.; calls: range
- L18--23 `def extend_tether(env, steps):` — Extend the tether, pushing its attached train toward a new token.; calls: range
- L26--31 `def retract_tether(env, steps):` — Retract the tether, pulling a contacted token into the train.; calls: range
- L34--52 `def reverse_row_train(env, approach_lanes, stages, final_extension):` — Reverse a token row by staging each unwanted prefix one lane lower.; calls: extend_tether, move_vertical_lanes, retract_tether
- L55--76 `def weave_vertical_four_train(env, approach_lanes=4, thread_steps=3):` — Weave a four-token vertical stack into its requested horizontal train.; calls: extend_tether, move_vertical_lanes, retract_tether
- L79--118 `def unweave_horizontal_pairs_to_vertical_heads( env, approach_lanes=5, alignment_steps=2, far_tail_reach=5 ):` — Deal a four-token row into two ordered vertical two-token trains.; calls: extend_tether, move_vertical_lanes, range, retract_tether
- L121--176 `def stage_split_rows_for_lower_interleave(env):` — Route one token from each split row into a workable lower-lane pair.; calls: extend_tether, move_vertical_lanes, retract_tether
- L179--211 `def interleave_lower_lane_train(env):` — Insert a staged middle token between two lower-lane endpoints.; calls: extend_tether, move_vertical_lanes, retract_tether
- L214--219 `def repeat_action(env, action, count):` — Apply one opaque action a bounded number of times.; calls: range
- L222--241 `def select_live_collector(env, color, edge):` — Select the live collector of ``color`` nearest the named room edge.; calls: RuntimeError, connected_components, min, round
- L244--275 `def deal_vertical_triplet_to_top_collector(env):` — Deal a vertical triplet onto the perpendicular top collector.; calls: extend_tether, move_vertical_lanes, range, repeat_action, retract_tether, select_live_collector
- L278--307 `def shuttle_row_tokens_under_vertical_train( env, stages=((1, 5, 6), (2, 6, 5), (3, 5, 3)), ):` — Grow a left-owned row using a finished vertical train as a lift.; calls: extend_tether, move_vertical_lanes, repeat_action, retract_tether, select_live_collector
- L310--371 `def weave_shared_center_cross(env):` — Weave a checker and one offset hub into two crossing collector trains.; calls: RuntimeError, connected_components, extend_tether, max, min, move_vertical_lanes, repeat_action, retract_tether, round, select_live_collector

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

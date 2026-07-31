# Generated solver source index

Use line ranges to inspect only definitions relevant to the current level.

## players.py
- L5--6 `def play_level_1(env):`; calls: follow_diagonal_lattice_to_ring
- L9--10 `def play_level_2(env):`; calls: merge_equal_squares_and_deliver_to_ring
- L13--15 `def play_level_3(env):`; calls: deliver_remaining_square_via_diagonal_detour, merge_equal_squares_and_deliver_to_ring
- L18--19 `def play_level_4(env):`; calls: merge_equal_squares_around_moving_cutter
- L22--25 `def play_level_5(env):`; calls: merge_equal_squares_around_moving_cutter
- L28--29 `def play_level_6(env):`; calls: stage_large_square_for_diagonal_partner
- L32--33 `def play_level_7(env):`; calls: merge_equal_squares_around_moving_cutter
- L36--51 `def play_level_8(env):`; calls: corner_ring_targets_for_coupled_layout, merge_moving_bodies_preserving_cutter, merge_small_squares_along_corner_lane, move_solid_square_to_target, reseat_square_while_cutting_staged_square, route_cutter_and_merged_body_to_corner_rings

## legs.py
- L9--22 `def _solid_playfield_squares( env, colors=None, excluded_colors=(), min_area=1 ):`; calls: connected_components
- L25--26 `def _click(env, row, col):`; calls: int, round
- L29--38 `def _move_square_one_step(env, square, target, step=None):`; calls: _click, max, min
- L41--64 `def _body_groups(env, color):`; calls: abs, int, len, max, range, sorted, tuple
- L67--71 `def _body_center(points):`; calls: len, round, sum
- L74--75 `def _level_active(env, start_level):`
- L78--90 `def _control_body(env, start_level, color, offset, selector=None):`; calls: _body_center, _body_groups, _click, _level_active, selector
- L93--121 `def follow_diagonal_lattice_to_ring(env, step=6, max_moves=12):` — Move the playfield avatar diagonally toward the ringed target.; calls: connected_components, int, max, min, range, round
- L124--183 `def merge_equal_squares_and_deliver_to_ring(env, step=6, max_moves=32):` — Merge equal playfield squares, then carry the result into its ring.; calls: _click, _solid_playfield_squares, abs, anchor, connected_components, enumerate, len, max, min, round
- L186--247 `def deliver_remaining_square_via_diagonal_detour(env, step=6, max_moves=16):` — Route a remaining diagonal-moving square around occupied target rings.; calls: _click, _solid_playfield_squares, abs, connected_components, len, max, pieces, rings, round, sum
- L250--512 `def merge_equal_squares_around_moving_cutter( env, step=6, max_states=5000, max_depth=30, minimum_stage_mass=1 ):` — Plan collision-free equal-square merges, then deliver into the ring.; calls: abs, anchor, any, avatar_centers, candidate_actions, connected_components, enumerate, int, len, list, +13
- L515--574 `def stage_large_square_for_diagonal_partner(env, max_moves=16):` — Stage the largest square in the far ring, then advance its partner.; calls: _click, _move_square_one_step, _solid_playfield_squares, abs, connected_components, len, max, solid_squares
- L577--601 `def corner_ring_targets_for_coupled_layout(env):` — Identify the two left targets for the coupled corner-ring layout.; calls: _body_groups, _solid_playfield_squares, connected_components, len, map, max, min, sorted, tuple
- L604--616 `def merge_small_squares_along_corner_lane(env, target):` — Merge the nearby small pair along the lane to a corner target.; calls: _click, _solid_playfield_squares, len, max, round
- L619--633 `def move_solid_square_to_target(env, color, target, max_moves):` — Move one solid square toward a target in size-derived strides.; calls: _level_active, _move_square_one_step, _solid_playfield_squares, abs, map, max, range
- L636--660 `def merge_moving_bodies_preserving_cutter(env):` — Merge two moving bodies while retaining the third as a cutter.; calls: _body_center, _body_groups, _control_body, len, max
- L663--681 `def reseat_square_while_cutting_staged_square(env, target):` — Reseat one square while the remaining body cuts the staged square.; calls: _body_groups, _click, _level_active, _solid_playfield_squares, len, map, max, min
- L684--705 `def route_cutter_and_merged_body_to_corner_rings(env):` — Redirect the coupled bodies while keeping both squares seated.; calls: _control_body

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

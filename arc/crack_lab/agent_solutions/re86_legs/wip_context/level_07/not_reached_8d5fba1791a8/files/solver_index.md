# Generated solver source index

Use line ranges to inspect only definitions relevant to the current level.

## players.py
- L5--6 `def play_level_1(env):`; calls: align_colored_crosses_to_ring_axes
- L9--10 `def play_level_2(env):`; calls: align_selected_outlines_to_ring_markers
- L13--14 `def play_level_3(env):`; calls: cover_ring_markers_with_selected_shapes
- L17--18 `def play_level_4(env):`; calls: repaint_selected_shapes_to_cover_colored_ring_markers
- L21--22 `def play_level_5(env):`; calls: paint_shapes_at_swatches_to_cover_ring_markers
- L25--26 `def play_level_6(env):`; calls: deform_selected_shapes_to_colored_ring_axes

## legs.py
- L9--10 `def _most_common(values):`; calls: Counter
- L13--22 `def _move_on_lattice(env, row_delta, col_delta, step):`; calls: ValueError, abs, range
- L25--75 `def align_colored_crosses_to_ring_axes(env, ring_color=4):` — Align movable coloured crosses with the axes marked by matching rings.; calls: ValueError, _most_common, _move_on_lattice, arr, connected_components, defaultdict, enumerate, int, len, set, +1
- L78--166 `def align_selected_outlines_to_ring_markers(env, ring_color=4):` — Translate selectable coloured outlines through all matching ring centers.; calls: ValueError, _most_common, _move_on_lattice, abs, all, arr, connected_components, enumerate, int, len, +7
- L169--327 `def cover_ring_markers_with_selected_shapes(env, ring_color=4):` — Translate selectable line/X/diamond shapes to cover matching ring centers.; calls: ValueError, _most_common, _move_on_lattice, abs, arr, connected_components, covers, enumerate, int, len, +10
- L330--544 `def repaint_selected_shapes_to_cover_colored_ring_markers( env, ring_color=4, station_border_color=2):` — Repaint selectable shapes at swatches, then cover same-colour markers.; calls: ValueError, _most_common, _move_on_lattice, abs, arr, connected_components, covers, defaultdict, enumerate, fits, +12
- L547--552 `def _selected_center(env):`; calls: ValueError, arr, int, len, list, zip
- L555--569 `def _selection_cycle(env, limit=12):` — Centers of every selectable object, in the order USE cycles them.; calls: ValueError, _selected_center, len
- L572--580 `def _park_all(env, direction, count, distance):` — Push every selectable object off the board and return the bare frame.; calls: arr, range
- L583--610 `def _survey_markers_and_swatches(env, count, distance, ring_color, border_color):` — Ring markers and paint swatches, seen with the movable shapes parked.; calls: _most_common, _park_all, connected_components, int, len, range
- L613--663 `def _shape_offsets(env, index, count, distance):` — Cell offsets of one selectable shape, with the others parked away.; calls: ValueError, _most_common, _selected_center, arr, int, list, range, set, zip
- L666--672 `def _swatch_hit(offsets, center, swatches):`; calls: len
- L675--697 `def _paint_routes(offsets, start, start_color, swatches, step, rows, cols):` — BFS over (center, paint colour) states on the movement lattice.; calls: _swatch_hit, deque, range
- L700--705 `def _route_centers(previous, state):`
- L708--780 `def paint_shapes_at_swatches_to_cover_ring_markers( env, ring_color=4, station_border_color=2):` — Cover every ring marker with a shape recoloured by a paint swatch.; calls: ValueError, _move_on_lattice, _paint_routes, _route_centers, _selection_cycle, _shape_offsets, _survey_markers_and_swatches, arr, connected_components, enumerate, +9
- L783--873 `def deform_selected_shapes_to_colored_ring_axes(env, ring_color=4):` — Use barriers to reshape an outline and cross onto their coloured rings.; calls: Counter, ValueError, _move_on_lattice, arr, connected_components, defaultdict, execute, int, len, max, +2

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

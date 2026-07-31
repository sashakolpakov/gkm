# Generated solver source index

Use line ranges to inspect only definitions relevant to the current level.

## players.py
- L5--7 `def play_level_1(env): # tu93 level 1 is a fixed-block grid maze: steer the avatar to the goal.`; calls: drive_block_maze
- L10--12 `def play_level_2(env): # Approach the directional waypoint from below, then continue to the goal.`; calls: drive_block_maze_via_color
- L15--18 `def play_level_3(env): # Recovered from verified proposer path artifact: checkpoint.json+proposer_last.log`
- L21--22 `def play_level_4(env):`; calls: drive_dynamic_maze_via_color
- L25--26 `def play_level_5(env):`; calls: drive_dynamic_maze_via_color
- L29--31 `def play_level_6(env):`; calls: drive_dynamic_directional_waypoints

## legs.py
- L11--13 `def _mode_color(f):`; calls: int
- L16--26 `def _least_color(f, bg):` — Rarest non-background color (used as the avatar marker).; calls: int, zip
- L29--135 `def parse_block_maze(f, cell=3):` — Parse a grid maze drawn with fixed `cell`x`cell` blocks.; calls: Counter, _least_color, _mode_color, block_cells, center, edge_present, int, len, range, set
- L138--153 `def _maze_path_between(g, start, goal, blocked=()):` — BFS between two parsed maze nodes, optionally avoiding some nodes.; calls: deque, neigh
- L156--161 `def maze_path_actions(f, cell=3):` — BFS in node space from avatar to goal; return a list of key actions.; calls: _maze_path_between, parse_block_maze
- L164--198 `def maze_path_via_color(f, waypoint_color, entry_action, cell=3):` — Plan through a colored maze node, entering it from one required side.; calls: _maze_path_between, any, parse_block_maze
- L201--260 `def maze_path_via_direction_markers( f, waypoint_color, marker_color, goal_color, cell=3):` — Route through all edge-marked waypoints, then to the colored goal.; calls: _maze_path_between, any, int, len, min, parse_block_maze, set, tuple
- L263--280 `def drive_replan(env, plan_fn, max_steps=300):` — Generic closed-loop driver: sense -> plan -> commit ONE action -> repeat.; calls: plan_fn, range
- L283--289 `def drive_block_maze(env, cell=3, max_steps=300):` — Navigate a fixed-block grid maze avatar to its goal.; calls: drive_replan, maze_path_actions
- L292--300 `def drive_block_maze_via_color( env, waypoint_color, entry_action, cell=3, max_steps=300):` — Navigate a block maze via a colored node with directional entry.; calls: drive_replan, maze_path_via_color
- L303--311 `def drive_block_maze_via_direction_markers( env, waypoint_color, marker_color, goal_color, cell=3, max_steps=300):` — Clear directional maze waypoints before navigating to the goal.; calls: drive_replan, maze_path_via_direction_markers
- L314--347 `def _bounded_visible_path(env, goal_fn, max_states=5000, max_depth=80):` — Find a short key-action path using only visible, non-HUD frame state.; calls: deque, goal_fn, int, key, len, replay
- L350--389 `def drive_dynamic_maze_via_color( env, waypoint_color, max_states=5000, max_depth=80):` — Clear visible colored gates, then finish a maze with turn-driven agents.; calls: _bounded_visible_path, int
- L392--466 `def drive_dynamic_directional_waypoints( env, waypoint_color, marker_color, avatar_color, cell=3, max_states=5000, max_depth=80):` — Clear directional waypoints in a maze with turn-driven agents.; calls: _bounded_visible_path, any, avatar_on_waypoint, blocks, int, parse_block_maze, remaining, sum

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

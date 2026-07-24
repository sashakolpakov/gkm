# Generated solver source index

Use line ranges to inspect only definitions relevant to the current level.

## players.py
- L5--7 `def play_level_1(env): # tu93 level 1 is a fixed-block grid maze: steer the avatar to the goal.`; calls: drive_block_maze

## legs.py
- L11--13 `def _mode_color(f):`; calls: int
- L16--26 `def _least_color(f, bg):` — Rarest non-background color (used as the avatar marker).; calls: int, zip
- L29--134 `def parse_block_maze(f, cell=3):` — Parse a grid maze drawn with fixed `cell`x`cell` blocks.; calls: Counter, _least_color, _mode_color, block_cells, center, edge_present, int, len, range, set
- L137--153 `def maze_path_actions(f, cell=3):` — BFS in node space from avatar to goal; return a list of key actions.; calls: deque, neigh, parse_block_maze
- L156--173 `def drive_replan(env, plan_fn, max_steps=300):` — Generic closed-loop driver: sense -> plan -> commit ONE action -> repeat.; calls: plan_fn, range
- L176--182 `def drive_block_maze(env, cell=3, max_steps=300):` — Navigate a fixed-block grid maze avatar to its goal.; calls: drive_replan, maze_path_actions

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

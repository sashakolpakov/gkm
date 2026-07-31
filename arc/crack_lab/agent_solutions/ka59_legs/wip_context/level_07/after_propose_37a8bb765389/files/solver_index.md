# Generated solver source index

Use line ranges to inspect only definitions relevant to the current level.

## players.py
- L5--17 `def play_level_1(env): # Push the initially selected ring through the central barrier.`; calls: move_steps, select_at
- L20--59 `def play_level_2(env): # Stage the vertical piece against the lower horizontal gap.`; calls: move_steps, select_at
- L62--79 `def play_level_3(env): # Drive the small ring around the right large ring and send it down a lane.`; calls: move_steps
- L82--110 `def play_level_4(env): # Send the lower small ring through the barrier.`; calls: move_steps, select_at
- L113--119 `def play_level_5(env): # Climb until the cycling corridor agents hand the ring through the barrier.`; calls: move_steps
- L122--144 `def play_level_6(env): # Use the three cycling corridor agents to hand both rings across the # sealed barriers, then center them on their differently sized targets.`; calls: move_steps

## legs.py
- L7--10 `def move_steps(env, direction, count):` — Move the currently selected object repeatedly in one direction.; calls: range
- L13--15 `def select_at(env, x, y):` — Select the object occupying screen coordinate (x, y).

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

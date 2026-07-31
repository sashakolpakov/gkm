# Generated solver source index

Use line ranges to inspect only definitions relevant to the current level.

## players.py
- L5--6 `def play_level_1(env):`; calls: reunite_mirrored_pair
- L9--10 `def play_level_2(env):`; calls: reunite_mirrored_pair
- L13--16 `def play_level_3(env):`; calls: assemble_smaller_agents, relocate_selectable_blockers, reunite_mirrored_pair
- L19--21 `def play_level_4(env):`; calls: relocate_selectable_blockers, reunite_mirrored_pair
- L24--25 `def play_level_5(env):`; calls: reunite_mirrored_pair

## legs.py
- L5--13 `def follow_action_sequence(env, actions):` — Replay a verified key/coordinate route, stopping if the game terminates.; calls: isinstance
- L16--18 `def reunite_mirrored_pair(env, route):` — Drive a mirrored pair along a verified route until they reunite.; calls: follow_action_sequence
- L43--45 `def relocate_selectable_blockers(env, route):` — Select and move blocker groups to verified out-of-corridor positions.; calls: follow_action_sequence
- L67--69 `def assemble_smaller_agents(env):` — Bring the selectable smaller agents to the reunited main pair.; calls: follow_action_sequence

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

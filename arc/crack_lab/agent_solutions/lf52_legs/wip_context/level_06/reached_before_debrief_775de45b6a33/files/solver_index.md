# Generated solver source index

Use line ranges to inspect only definitions relevant to the current level.

## players.py
- L5--6 `def play_level_1(env):`; calls: solve_peg_solitaire
- L9--10 `def play_level_2(env):`; calls: solve_peg_solitaire_with_carrier
- L13--14 `def play_level_3(env):`; calls: solve_peg_solitaire_with_carrier
- L17--18 `def play_level_4(env):`; calls: solve_bridge_carrier_peg_solitaire
- L21--22 `def play_level_5(env):`; calls: solve_bridge_carrier_peg_solitaire
- L25--26 `def play_level_6(env):`; calls: solve_wrapped_bridge_carrier_peg_solitaire

## legs.py
- L11--23 `def _peg_board(frame):` — Return the slot lattice and occupied slots visible in a peg board.; calls: connected_components, frozenset
- L26--40 `def _lattice_step(slots):`; calls: gcd, sorted, zip
- L43--68 `def _peg_solution(slots, start):` — Find captures that leave the confirmed winning state of one peg.; calls: _lattice_step, deque, frozenset, len, sorted
- L71--79 `def solve_peg_solitaire(env):` — Solve a visible orthogonal peg-solitaire board using coordinate clicks.; calls: _peg_board, _peg_solution
- L82--108 `def _carrier_capture_macros(frame):` — Return visible peg captures, including an empty bordered carrier.; calls: _lattice_step, _peg_board, connected_components, sorted, tuple
- L111--153 `def solve_peg_solitaire_with_carrier(env, max_states=2000, max_depth=40):` — Solve peg boards connected by a key-movable bordered carrier slot.; calls: _carrier_capture_macros, arr, deque, int, isinstance, key, len, tuple
- L156--205 `def _bridge_carrier_state(frame):` — Return the puzzle-relevant geometry of a bridge/carrier peg board.; calls: connected_components, frozenset
- L208--225 `def _bridge_carrier_moves_from_state(state):`; calls: _lattice_step, sorted, tuple
- L228--230 `def _bridge_carrier_moves(frame):` — Return legal (kind, source, destination) peg moves.; calls: _bridge_carrier_moves_from_state, _bridge_carrier_state
- L233--363 `def solve_bridge_carrier_peg_solitaire( env, max_align_states=120, max_macros=40, alignment_lookahead=24):` — Solve peg boards joined by persistent bridges and a movable carrier.; calls: _bridge_carrier_moves_from_state, _bridge_carrier_state, _lattice_step, abs, alignment_path, alignment_priority, bridge_score, choose, choose_state, enumerate, +8
- L366--389 `def _movable_bridge_board(frame):` — Return visible slots, carriers, movable bridges, and pegs.; calls: connected_components
- L392--432 `def _movable_bridge_solution(frame, max_states=20000):` — Solve a visible relay board containing one reusable movable bridge.; calls: _movable_bridge_board, deque, frozenset, iter, len, next, set, sorted, tuple
- L435--438 `def _play_lattice_moves(env, moves):`
- L441--492 `def solve_wrapped_bridge_carrier_peg_solitaire(env):` — Solve a long relay whose carrier rails reveal successive peg regions.; calls: _movable_bridge_board, _movable_bridge_solution, _play_lattice_moves, iter, len, next

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

# Generated solver source index

Use line ranges to inspect only definitions relevant to the current level.

## players.py
- L13--14 `def play_level_1(env):`; calls: solve_peg_solitaire
- L17--18 `def play_level_2(env):`; calls: solve_peg_solitaire_with_carrier
- L21--22 `def play_level_3(env):`; calls: solve_peg_solitaire_with_carrier
- L25--26 `def play_level_4(env):`; calls: solve_bridge_carrier_peg_solitaire
- L29--30 `def play_level_5(env):`; calls: solve_bridge_carrier_peg_solitaire
- L33--34 `def play_level_6(env):`; calls: solve_wrapped_bridge_carrier_peg_solitaire
- L37--38 `def play_level_7(env):`; calls: solve_parallel_wrapped_bridge_carrier_peg_solitaire
- L41--42 `def play_level_8(env):`; calls: solve_grid_wrapped_bridge_carrier_peg_solitaire
- L45--46 `def play_level_9(env):`; calls: solve_multi_bridge_wrapped_carrier_peg_solitaire

## legs.py
- L11--16 `def play_action(env, action):` — Execute one key action or coordinate-click action.; calls: isinstance
- L19--22 `def play_actions(env, actions):` — Execute key and coordinate-click actions in order.; calls: play_action
- L25--31 `def play_lattice_moves(env, moves):` — Execute source/destination moves on the visible slot lattice.; calls: play_actions
- L34--46 `def _peg_board(frame):` — Return the slot lattice and occupied slots visible in a peg board.; calls: connected_components, frozenset
- L49--63 `def _lattice_step(slots):`; calls: gcd, sorted, zip
- L66--91 `def _peg_solution(slots, start):` — Find captures that leave the confirmed winning state of one peg.; calls: _lattice_step, deque, frozenset, len, sorted
- L94--100 `def solve_peg_solitaire(env):` — Solve a visible orthogonal peg-solitaire board using coordinate clicks.; calls: _peg_board, _peg_solution, play_lattice_moves
- L103--129 `def _carrier_capture_macros(frame):` — Return visible peg captures, including an empty bordered carrier.; calls: _lattice_step, _peg_board, connected_components, sorted, tuple
- L132--167 `def solve_peg_solitaire_with_carrier(env, max_states=2000, max_depth=40):` — Solve peg boards connected by a key-movable bordered carrier slot.; calls: _carrier_capture_macros, arr, deque, int, key, len, play_action, play_actions, tuple
- L170--219 `def _bridge_carrier_state(frame):` — Return the puzzle-relevant geometry of a bridge/carrier peg board.; calls: connected_components, frozenset
- L222--239 `def _bridge_carrier_moves_from_state(state):`; calls: _lattice_step, sorted, tuple
- L242--244 `def _bridge_carrier_moves(frame):` — Return legal (kind, source, destination) peg moves.; calls: _bridge_carrier_moves_from_state, _bridge_carrier_state
- L247--371 `def solve_bridge_carrier_peg_solitaire( env, max_align_states=120, max_macros=40, alignment_lookahead=24):` — Solve peg boards joined by persistent bridges and a movable carrier.; calls: _bridge_carrier_moves_from_state, _bridge_carrier_state, _lattice_step, abs, alignment_path, alignment_priority, bridge_score, choose, choose_state, enumerate, +8
- L374--397 `def _movable_bridge_board(frame):` — Return visible slots, carriers, movable bridges, and pegs.; calls: connected_components
- L400--440 `def _movable_bridge_solution(frame, max_states=20000):` — Solve a visible relay board containing one reusable movable bridge.; calls: _movable_bridge_board, deque, frozenset, iter, len, next, set, sorted, tuple
- L443--489 `def solve_wrapped_bridge_carrier_peg_solitaire(env):` — Solve a long relay whose carrier rails reveal successive peg regions.; calls: _movable_bridge_board, _movable_bridge_solution, iter, len, next, play_actions, play_lattice_moves
- L492--545 `def solve_parallel_wrapped_bridge_carrier_peg_solitaire(env):` — Relay pegs and a movable bridge through synchronized carrier rails.; calls: play_actions, play_lattice_moves
- L548--590 `def solve_grid_wrapped_bridge_carrier_peg_solitaire(env):` — Relay a peg through a grid of synchronized wrapped carriers.; calls: play_actions, play_lattice_moves
- L593--666 `def _multi_bridge_carrier_solution(frame, max_states=500000):` — Route multiple reusable bridges, capture to one peg, and load it.; calls: connected_components, deque, frozenset, iter, len, next, set, sorted, tuple
- L669--721 `def solve_multi_bridge_wrapped_carrier_peg_solitaire(env):` — Relay movable bridges through a wrapped carrier and capture the pegs.; calls: _multi_bridge_carrier_solution, play_actions, play_lattice_moves

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

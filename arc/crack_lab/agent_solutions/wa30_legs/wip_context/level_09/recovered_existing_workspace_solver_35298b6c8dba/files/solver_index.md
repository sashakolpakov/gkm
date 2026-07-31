# Generated solver source index

Use line ranges to inspect only definitions relevant to the current level.

## players.py
- L5--6 `def play_level_1(env):`; calls: fill_three_slot_target
- L9--10 `def play_level_2(env):`; calls: fill_bottom_slots_alongside_courier
- L13--14 `def play_level_3(env):`; calls: feed_three_blocks_to_courier
- L17--18 `def play_level_4(env):`; calls: feed_two_blocks_to_each_courier
- L21--22 `def play_level_5(env):`; calls: expedite_single_courier_with_four_blocks
- L25--26 `def play_level_6(env):`; calls: recover_two_blocks_across_courier_port
- L29--30 `def play_level_7(env):`; calls: hand_deliver_then_recover_one_courier_delivery
- L33--34 `def play_level_8(env):`; calls: disable_competing_couriers_and_expedite_paired_depots
- L37--38 `def play_level_9(env):`; calls: stage_three_deliveries_dismiss_thief_and_finish

## legs.py
- L7--12 `def walk(env, direction, steps=1):` — Move in one cardinal direction for a fixed number of grid steps.; calls: range
- L15--18 `def interact(env):` — Use the context-sensitive interaction action once.
- L21--46 `def fill_three_slot_target(env):` — Carry the three surrounding blocks into a horizontal three-slot target.; calls: interact, walk
- L49--71 `def fill_bottom_slots_alongside_courier(env):` — Fill two bottom target slots while an autonomous courier fills the rest.; calls: interact, walk
- L74--114 `def feed_three_blocks_to_courier(env):` — Keep two courier ports supplied until all five target blocks are delivered.; calls: interact, walk
- L117--168 `def feed_two_blocks_to_each_courier(env):` — Stage two cargo blocks at each of three wall-separated courier ports.; calls: interact, walk
- L171--219 `def expedite_single_courier_with_four_blocks(env):` — Place one cargo and shorten a courier's routes to three distant cargos.; calls: interact, walk
- L222--253 `def recover_two_blocks_across_courier_port(env):` — Return a courier delivery, then cross its cleared port for remote cargo.; calls: interact, walk
- L256--285 `def stage_and_recover_two_courier_deliveries(env):` — Stage an off-lane cargo, dismiss its courier, and recover both deliveries.; calls: interact, range, walk
- L288--312 `def hand_deliver_then_recover_one_courier_delivery(env):` — Hand-deliver the safe cargo, dismiss its courier, and recover its haul.; calls: interact, walk
- L315--363 `def disable_competing_couriers_and_expedite_paired_depots(env):` — Dismiss two wrong-way couriers and hand-deliver three slow cargos.; calls: interact, walk
- L366--410 `def stage_three_deliveries_dismiss_thief_and_finish(env):` — Stage three courier deliveries, dismiss a thief, and place final cargo.; calls: interact, walk

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

# Generated solver source index

Use line ranges to inspect only definitions relevant to the current level.

## players.py
- L5--8 `def play_level_1(env):` — Level 1: move the 9-block right 3 times then USE.; calls: bfs_or_fallback, make_bbox_key
- L11--29 `def play_level_2(env):` — Level 2: three movable blocks selected by clicking (action 6 = grab; calls: drive_objects
- L32--53 `def play_level_3(env):` — Level 3: same select-then-drive family as level 2, now with FOUR movable; calls: drive_objects
- L56--81 `def play_level_4(env):` — Level 4: same select-then-drive-then-commit family as levels 2 and 3.; calls: drive_objects

## legs.py
- L8--13 `def play_fixed_sequence(env, sequence):` — Play a fixed list of actions on the real env.; calls: int
- L16--21 `def click_select(env, x, y):` — Coordinate interaction: issue action 6 at pixel (x=col, y=row).; calls: int
- L24--33 `def grab_and_move(env, x, y, moves):` — Select-then-drive: grab the object under pixel (x=col, y=row) with the; calls: click_select, play_fixed_sequence
- L36--52 `def drive_objects(env, plan, commit=(5,)):` — Multi-object drive: the recurring sp80 select-then-drive-then-commit; calls: grab_and_move, list, play_fixed_sequence
- L55--65 `def make_bbox_key(color):` — Return a compact key function that tracks the bounding box of pixels; calls: int, len
- L68--75 `def bfs_or_fallback(env, key_fn, fallback, actions=(1, 2, 3, 4, 5, 6), max_states=500, max_depth=20):` — Try BFS (compact key) to find a winning path; play it.; calls: bfs_win_compact, play_fixed_sequence
- L78--105 `def bfs_win(env, actions=(1, 2, 3, 4, 5, 6), max_states=5000, max_depth=40):` — BFS over clone states to find shortest action sequence that increases; calls: deque, int, key_fn, len
- L108--129 `def bfs_win_compact(env, key_fn, actions=(1, 2, 3, 4, 5, 6), max_states=5000, max_depth=40):` — BFS with a custom key function (cheaper than full frame hash).; calls: deque, int, key_fn, len

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

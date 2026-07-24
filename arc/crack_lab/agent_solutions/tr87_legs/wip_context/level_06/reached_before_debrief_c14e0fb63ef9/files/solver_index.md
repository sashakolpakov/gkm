# Generated solver source index

Use line ranges to inspect only definitions relevant to the current level.

## players.py
- L5--8 `def play_level_1(env): # Level 1 is a tile-cycle matching puzzle: cycle each tile's glyph to a # hidden target. Reuse the general tile-cycle solver leg.`; calls: solve_tile_cycle_puzzle
- L11--17 `def play_level_2(env): # Level 2 is a legend-decode variant of the tile puzzle: a coded 'word' # (wide reference box) is translated through a key->value legend into the # sequence of glyphs the editable tiles must show. Same perceive->decod`; calls: solve_glyph_cipher_via
- L20--26 `def play_level_3(env): # Level 3 is a richer legend cipher: keys AND values are bordered boxes in # key->value pairs, and a single key may span several glyphs, so the coded # 'word' (the one unpaired key-colored box) must be SEGMENTED (toke`; calls: solve_glyph_cipher_via
- L29--35 `def play_level_4(env): # Level 4 is a COMPOSED cipher: two legends chained by border colour. A # source word (border SRC) is translated legend-by-legend (SRC->..->VAL, # following each pair's key-border -> value-border) into the target glyp`; calls: solve_glyph_cipher_via
- L38--42 `def play_level_5(env): # The reference source/target words are fixed, but the intervening legend # entries are cursor-editable word boxes. Discover the box cycles, solve # the induced segmentation constraints, and replay a clone-verified pl`; calls: solve_glyph_legend_synthesis_puzzle

## legs.py
- L17--26 `def action_bbox(env, action):` — Bounding box of the frame region that `action` changes, from `env`.
- L29--34 `def bbox_size(bbox):` — (height, width) of an inclusive bbox, or (0, 0) if bbox is None.
- L37--41 `def _box_contains(outer, inner):` — True iff inclusive bbox `outer` fully contains inclusive bbox `inner`.
- L44--57 `def action_period(env, action, key_fn, max_k=16, default=None):` — How many repeats of `action` until `key_fn(state)` returns to its start.; calls: key_fn, range
- L60--71 `def replay_for_reward(env, path):` — Execute `path` on the real `env`; return True if levels_completed rose.
- L74--88 `def solve_by_clone_search(env, discover, search):` — Higher-order leg: perceive -> search-on-clone -> replay-on-real.; calls: discover, replay_for_reward, search
- L93--108 `def tile_cycle_path(edit, move, counts):` — Serialize a per-tile plan into an edit/move action path.; calls: enumerate, len
- L113--150 `def discover_tile_cycle_puzzle(env):` — Auto-discover the tile/cursor structure from action deltas on clones.; calls: action_bbox, action_period, bbox_size, list, min, small
- L153--190 `def search_tile_cycle_config(env, edit, move, n_tiles, cycle, budget_s=200):` — Nested-clone DFS over per-tile glyph states; early-exit on reward.; calls: list, range, rec, tile_cycle_path
- L195--198 `def _glyph(frame, r, c, gh, gw, fg):` — Binary (fg?1:0) gh x gw glyph read at top-left (r,c).; calls: range, tuple
- L201--217 `def _glyph_canon(g):` — Canonical form of a binary glyph under the 8 D4 symmetries.; calls: map, range, tuple
- L220--225 `def _bordered_boxes(frame, color, inner_h, pitch, min_area=15):` — Connected components of `color` that are `inner_h`+2 tall (a glyph row
- L228--232 `def _box_glyphs(frame, bbox, gh, gw, pitch, fg):` — Split a bordered box into its 1..k glyphs (pitch-spaced along columns).; calls: _glyph, range
- L235--245 `def _nearest_right(box, candidates):` — The candidate blob on `box`'s row, nearest to its right, or None.
- L248--303 `def discover_glyph_tiles(env):` — Discover the shared front-end common to every tr87 glyph-cipher puzzle.; calls: Counter, _box_contains, action_bbox, discover_tile_cycle_puzzle, int, range
- L306--363 `def discover_glyph_cipher_puzzle(env):` — Auto-discover a legend-decode tile puzzle from a single frame.; calls: Counter, _box_glyphs, _cipher_spec, _glyph_canon, _nearest_right, discover_glyph_tiles, enumerate, len
- L366--382 `def _cipher_spec(base, editreg_bbox, seq_canon):` — Assemble the spec dict consumed by plan_glyph_cipher.; calls: len
- L385--412 `def plan_glyph_cipher(env, spec):` — Compute the edit/move path that sets each tile to its decoded target.; calls: _glyph, _glyph_canon, range, tile_cycle_path
- L415--427 `def solve_glyph_cipher_via(env, discover):` — Higher-order leg: solve ANY glyph-cipher variant given its `discover`.; calls: solve_by_clone_search
- L442--444 `def _box_glyphs_canon(frame, bbox, gh, gw, pitch, fg):` — Split a bordered box into its pitch-spaced glyphs as D4-canonical forms.; calls: _box_glyphs, _glyph_canon
- L447--470 `def _segment_word(word, keys):` — Unique tokenisation of glyph `word` into `keys` (each a glyph tuple).; calls: enumerate, len, list, rec, tuple
- L473--522 `def discover_glyph_legend_puzzle(env):` — Auto-discover a legend-with-segmentation glyph cipher from one frame.; calls: Counter, _box_glyphs_canon, _cipher_spec, _nearest_right, _segment_word, discover_glyph_tiles, len
- L542--593 `def discover_glyph_compose_puzzle(env):`; calls: _box_contains, _box_glyphs, _cipher_spec, _glyph, _glyph_canon, defaultdict, discover_glyph_tiles, len, range, set, +1
- L596--607 `def solve_tile_cycle_puzzle(env, budget_s=200):` — Thin composition: discover the tile-cycle structure, search a clone for; calls: search_tile_cycle_config, solve_by_clone_search
- L612--732 `def discover_glyph_legend_synthesis_puzzle(env):` — Discover a puzzle where the legend itself is cursor-editable.; calls: Counter, _box_glyphs, _glyph_canon, bbox_size, box_word, defaultdict, int, len, list, max, +5
- L735--799 `def plan_glyph_legend_synthesis(env, spec):` — Solve and clone-verify an editable legend synthesis specification.; calls: bbox_size, enumerate, extend, len, tuple, zip
- L802--805 `def solve_glyph_legend_synthesis_puzzle(env):` — Discover, symbolically plan, clone-verify, and replay a mutable legend.; calls: solve_by_clone_search

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

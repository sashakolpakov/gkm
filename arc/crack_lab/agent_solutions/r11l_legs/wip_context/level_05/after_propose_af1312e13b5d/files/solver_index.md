# Generated solver source index

Use line ranges to inspect only definitions relevant to the current level.

## players.py
- L5--9 `def play_level_1(env): # r11l level 1: align the rope's midpoint box onto the ring target. # Thin composition: one leg does the whole skill; the mechanics # (detect -> move active -> drag_endpoint the other) live in legs.py.`; calls: place_box_on_ring
- L12--17 `def play_level_2(env): # r11l level 2: several colour-coded rope systems sharing one cursor. # Each colour's box (centroid of its endpoints) must land on its own-colour # ring. A far box is rope-walked; a near one is straddled exactly. # Al`; calls: place_boxes_on_rings
- L20--26 `def play_level_4(env): # r11l level 4: same rope/box/ring mechanic, but boxes AND rings are # MULTI-COLOURED and decoy rings litter the board. Match each box to the # ring with an identical colour palette (discovering rope->box identity by `; calls: place_multicolor_boxes
- L29--34 `def play_level_3(env): # r11l level 3: another multi-rope board (more colour systems / farther # boxes). Same skill as level 2 — the generic solver discovers every # colour system, walks each far box in, then straddle-snaps it onto its # ow`; calls: place_boxes_on_rings

## legs.py
- L17--18 `def _arr(env):`
- L23--30 `def _neigh4_colors(f, r, c):` — Set of colours in the 4-connected neighbourhood of (r,c) (in bounds).; calls: int, set
- L33--43 `def _neigh8_colors(f, r, c):` — Set of colours in the 8-connected neighbourhood of (r,c) (in bounds).; calls: int, set
- L46--49 `def _centroid(points):` — Rounded (row,col) centroid of a non-empty list of (row,col) points.; calls: int, round
- L52--64 `def _isolated_cells(f, color):` — Single-pixel (4-isolated) cells of `color` — endpoint/ring markers.; calls: int
- L67--87 `def detect_rope(env):` — Return (active_endpoint, other_endpoint, ring_center) as (row,col) tuples.; calls: _arr, _centroid, _neigh4_colors, int
- L90--98 `def click(env, row, col):` — PRIMITIVE skill (written once): click the grid cell (row, col).; calls: int
- L101--103 `def move_active_to(env, row, col):` — Skill: move the currently active endpoint to (row, col).; calls: click
- L106--108 `def select_endpoint(env, point):` — Skill: activate an endpoint by clicking its marker at (row, col).; calls: click
- L111--121 `def drag_endpoint(env, frm, dst):` — Skill (written ONCE): pick up the endpoint whose marker is at `frm` and; calls: move_active_to, select_endpoint
- L124--140 `def place_box_on_ring(env):` — Position the two rope endpoints symmetrically about the ring centre so the; calls: _arr, detect_rope, drag_endpoint, min, move_active_to
- L157--164 `def box_center(env, color):` — (row,col) of the box (the 6 pixel wrapped in `color`) for a colour.; calls: _arr, int, max
- L167--177 `def active_pos(env):` — (row,col) of the currently picked-up (ACTIVE, 0-diamond) endpoint.; calls: _arr, _isolated_cells, _neigh8_colors, int
- L180--188 `def endpoints_of(env, color):` — List of (row,col) endpoint markers of `color` (0- or 3-diamond centres).; calls: _arr, _isolated_cells, _neigh8_colors, sorted
- L191--196 `def ring_center(env, color):` — (row,col) centre of the hollow diamond RING drawn in `color`.; calls: _arr, _centroid, _isolated_cells, _neigh4_colors
- L199--220 `def ring_systems(env):` — Discover every colour owning both a box and a ring.; calls: _arr, box_center, endpoints_of, int, ring_center
- L223--242 `def probe_drag(env, frm, dst, base_levels=None):` — Skill (written ONCE): TRY one `drag_endpoint` on a CLONE and report the; calls: active_pos, drag_endpoint
- L245--248 `def _can_drop(env, frm, dst):` — True iff picking the endpoint at `frm` and dropping at `dst` succeeds.; calls: probe_drag
- L251--302 `def walk_box_to(env, color, target, eps, iters=120):` — Rope-walk a box's centroid onto `target` by greedily moving whichever; calls: abs, any, box_center, drag_endpoint, enumerate, int, probe_drag, range, round, set
- L305--330 `def _run_straddle_plan(target_env, eps, perm, pat, target, base_levels, check_bounds):` — Skill (written ONCE): apply ONE symmetric-straddle plan on `target_env`.; calls: drag_endpoint, enumerate, tuple
- L333--376 `def straddle_box_to(env, color, target, eps):` — Place a box exactly on `target` by scattering its endpoints symmetrically; calls: _run_straddle_plan, box_center, len, permutations, range, set
- L388--407 `def _flood_nonbg(f, r, c, bg=5):` — Flood the 4-connected non-`bg` region from (r,c); return (colors,size).; calls: deque, int, len, set
- L410--422 `def multicolor_boxes(env, min_size=8):` — Boxes on a multi-coloured board: each is a solid blob of >= `min_size`; calls: _arr, _flood_nonbg, frozenset, int
- L425--442 `def _ring_outline_cells(f):` — Isolated single-colour cells (4-neighbourhood is only background) that; calls: int
- L445--480 `def multicolor_rings(env, radius=5):` — Cluster ring-outline cells into hollow diamonds. Returns list of; calls: _arr, _ring_outline_cells, abs, any, frozenset, int, len, range, round
- L483--499 `def marker_endpoints(env):` — Precise rope endpoint markers: single coloured pixels whose four; calls: _arr, all, int, len
- L502--503 `def _box_centers_set(env, min_size=8):`; calls: multicolor_boxes
- L506--524 `def group_endpoints_by_box(env):` — PROBE-discover which box each endpoint marker drives (general, no colour; calls: _box_centers_set, active_pos, click, iter, len, marker_endpoints, min, next
- L527--560 `def multicolor_systems(env):` — Full level-4 system list: match each multi-coloured box to the ring whose; calls: Counter, abs, group_endpoints_by_box, int, len, min, multicolor_boxes, multicolor_rings, sorted
- L563--599 `def solve_systems_walk_straddle(env, systems):` — HIGHER-ORDER composition leg (written ONCE): solve a whole board of; calls: abs, box_center, list, sorted, straddle_box_to, walk_box_to
- L602--607 `def place_multicolor_boxes(env):` — Solve a level-4-style board: drop every multi-coloured box into its; calls: multicolor_systems, solve_systems_walk_straddle
- L610--614 `def place_boxes_on_rings(env):` — Solve a multi-rope level: centre every colour's box on its own ring.; calls: ring_systems, solve_systems_walk_straddle

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

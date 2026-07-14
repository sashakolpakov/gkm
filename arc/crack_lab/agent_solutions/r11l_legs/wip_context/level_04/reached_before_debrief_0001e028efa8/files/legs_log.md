# Leg-library debrief log

Recurring composition patterns and repeated novelty.

## r11l level 1
Sole action = coordinate click `step(6,x,y)` (x=col,y=row). Two draggable
endpoints; a BOX is always their midpoint. A click moves the ACTIVE endpoint;
clicking on the other endpoint's marker selects it. WIN = box lands on the
hollow-diamond RING centre.
New legs: `detect_rope` (finds active endpoint / 3-endpoint / ring),
`place_box_on_ring` (put both endpoints on ring row at Rc±d so midpoint=ring).
Composed by `play_level_1`. Solves in 3 clicks.

Refactor: the raw action `env.step(6,x,y)` is now written ONCE in the
primitive `click(env,row,col)`. Two semantic skills wrap it:
`move_active_to` (drop the active endpoint) and `select_endpoint` (activate a
marker). `place_box_on_ring` composes them; `play_level_1` stays a one-line
thin composition.

## Candidate higher-order leg (recurring pattern)
Across the r11l endpoint puzzles the same shape recurs:

    detect_state(env) -> handles + target
    move_active_to(A)              # place handle #1
    select_endpoint(other)         # hand off control
    move_active_to(B)              # place handle #2 so a DERIVED point == target

i.e. "drive two coupled handles so their derived feature (here the midpoint
BOX) lands on a target." A higher-order leg would capture this:

    place_derived_on_target(env, detect, solve_positions)
        active, other, target = detect(env)
        posA, posB = solve_positions(active, other, target)
        move_active_to(env, *posA)
        select_endpoint(env, other)
        move_active_to(env, *posB)

`place_box_on_ring` is the midpoint instance, with `solve_positions` =
"both on target row at columns Rc-d, Rc+d". Future levels (different derived
feature or constraint) reuse the skeleton by supplying a new
`solve_positions`.

## r11l level 2
Multiple colour-coded rope systems share ONE pick/drop cursor. A colour C owns
a system iff it has BOTH a filled box (a `6` wrapped in C) and a hollow-diamond
RING (isolated C pixels touching only background 5). The box always sits at the
ROUNDED CENTROID of that colour's endpoints (level-2 had a 3-endpoint system in
colour 12 and a 2-endpoint system in colour 15). WIN = every box centred on its
OWN-colour ring (verified: cross-colour placement gives NO reward; both boxes on
same-colour rings simultaneously fires the level reward).

Key mechanic discovered by probing: each drop is bounded by a ROPE LENGTH from
the current box, so a far box must be WALKED (move one endpoint toward the ring,
the box drifts, repeat) — a single long jump is rejected. A near box is placed
in one shot by a symmetric STRADDLE (endpoint offsets summing to zero => centroid
== ring). Two gotchas that cost time:
  * Selecting/placing must be validated by the ACTIVE (0-diamond) marker, but the
    winning drop COMPLETES the level and mutates the frame, so straddle verify must
    also accept "levels_completed increased" as success (else it rejects the very
    move that wins).
  * Placing one box on its ring changes the other rope's reachable set. Walking
    (which clusters endpoints tightly around the ring) keeps a system out of the
    others' way; scattering (large straddle offsets) can block a neighbour. Hence:
    walk-first (farthest gap first), then straddle-snap.

New legs: `box_center`, `active_pos`, `endpoints_of`, `ring_center`,
`ring_systems` (perception), `walk_box_to` (rope-walk a centroid onto a target),
`straddle_box_to` (exact symmetric snap, level-completion aware), and the
orchestrator `place_boxes_on_rings`. `play_level_2` is a one-line composition.
Reused nothing from level 1 directly (different derived feature: centroid of N
endpoints vs midpoint of 2), but the family matches the "drive coupled handles so
a derived point hits a target" pattern noted earlier.

## Debrief refactor (after clearing level 2)
Comparing `play_level_2` against the earlier `play_level_1`, the *players* were
already thin (one leg call each), but the two leg families had duplicated the
same low-level code. Written ONCE now and shared by both:
  * `_neigh4_colors` / `_neigh8_colors` — neighbourhood colour sets. Previously
    `detect_rope` (L1) and `ring_center` (L2) each hand-rolled the 4-neighbour
    scan; `active_pos`/`endpoints_of` (L2) hand-rolled the 8-neighbour scan.
  * `_centroid(points)` — the rounded mean-of-points used identically by
    `detect_rope` (ring centre, L1) and `ring_center` (L2).
  * `_isolated_cells(f, color)` — single-pixel marker cells (was the L2-only
    `_components_area1`), now the one place that defines "isolated marker".
  * `drag_endpoint(env, frm, dst)` — THE recurring gesture: select an endpoint
    then move the active one (two clicks). Level 1's `place_box_on_ring`, and
    level 2's `walk_box_to` / `straddle_box_to` / `_can_drop`, all previously
    re-issued raw `click` pairs; they now all call this one leg.
Behaviour unchanged: `python gkm_try.py` still reports
`RESULT levels=2 moves=13 replay_ok=True`.

## Candidate higher-order leg (recurring composition pattern)
Every r11l solver reduces to the SAME shape: repeatedly `drag_endpoint` to push
a DERIVED feature (midpoint on L1, N-endpoint centroid on L2) toward a target,
verifying each proposed drag on a `env.clone()` and stopping when the feature
meets the target (or the level completes). A higher-order leg would capture it:

    converge_feature_to_target(env, feature_fn, endpoints, target,
                               propose_dsts, done):
        while not done(env):
            best = argmin over (i, dst in propose_dsts(endpoints, target))
                   of distance(feature_fn(clone_after_drag(i,dst)), target)
                   # a drag that COMPLETES the level always wins
            if best is None: break
            drag_endpoint(env, endpoints[i], dst); endpoints[i] = dst

`place_box_on_ring` is the closed-form 1-step instance (solve positions
directly); `walk_box_to` is the greedy multi-step instance; `straddle_box_to`
is the same skeleton with a symmetric-offset proposal that lands the centroid
exactly. All three share `drag_endpoint` + clone-verify + "level-completion
counts as success". Future r11l levels reuse the skeleton by supplying a new
`feature_fn` / `propose_dsts`.

## r11l level 3
Level 3 is a bigger multi-rope board (more colour systems, boxes farther from
their rings) but introduces NO new mechanic. `play_level_3` needed nothing
level-specific: it is the SAME one-line composition as level 2,
`place_boxes_on_rings(env)`. The whole board is solved by the generic
solver discovering every colour system, walking each far box in, then
straddle-snapping it onto its own-colour ring. Behaviour unchanged:
`python gkm_try.py` still reports `RESULT levels=3 moves=27 replay_ok=True`.

## Debrief refactor (after clearing level 3)
Comparing `play_level_3` to the earlier players confirmed the players were
already thin (`play_level_2` and `play_level_3` are the identical one-liner
`place_boxes_on_rings(env)`). The remaining duplication lived INSIDE legs.py,
and it was the same "try a move, then commit it" gesture written more than once.
Written ONCE now and shared:
  * `probe_drag(env, frm, dst, base_levels)` — TRY one `drag_endpoint` on a
    CLONE and report `(clone, landed, completed)` without mutating `env`. This
    "look before you leap" probe was hand-rolled in both `_can_drop` (the
    reachability test) and `walk_box_to` (the greedy inner loop); both now call
    the one leg. `landed` folds the axis-swapped `active_pos == dst` check and
    `completed` folds the "this move wins the level" check into one place.
  * `_run_straddle_plan(target_env, eps, perm, pat, target, base, check_bounds)`
    — apply ONE symmetric-straddle plan (permutation -> offset pattern) with
    early stop on level completion. `straddle_box_to` previously inlined this
    EXACT loop twice: once bounded on a clone to TRIAL, then again on the real
    env to COMMIT. The single leg is now called twice, distinguished only by
    the `check_bounds` flag (trial bounds-checks; commit replays the validated
    plan), removing a verbatim copy-paste block.

## Candidate higher-order leg (recurring composition pattern)
Three levels in, the pattern that recurs across the WHOLE family is now sharp,
in two nested layers:

  1. per-move: PROPOSE a drag, PROBE it on a clone, and only COMMIT if it
     improves (or wins). Concretely `probe_drag` + "keep the best; a
     level-completing move always wins" — shared by walk and straddle.
  2. per-board: for each independent SYSTEM, drive its DERIVED feature
     (midpoint / centroid) onto its target by WALK-then-STRADDLE, ordering
     systems farthest-gap-first so a placed system stays out of the others'
     reach. `place_boxes_on_rings` is this orchestrator; `place_box_on_ring`
     is its degenerate single-system, closed-form instance.

A higher-order leg would capture layer (2) directly:

    solve_systems(env, discover, feature_fn, place_one):
        systems = discover(env)                      # ring_systems
        systems.sort(key=gap_to_target, reverse=True)  # farthest first
        for s in systems:
            if feature_fn(env, s) == s.target: continue
            place_one(env, s)                        # walk_box_to; straddle_box_to

with `place_one` itself built from layer (1)'s probe/commit loop
(`converge_feature_to_target`, sketched in the level-2 debrief). Every current
r11l player would then be `solve_systems(env, ...)` with level-appropriate
`discover` / `feature_fn` / `place_one`, and `play_level_1..3` stay one line.

## r11l level 4
Same rope/box/ring physics as L2/L3 (a click moves the ACTIVE endpoint; the box
is the endpoints' midpoint/centroid; a distant box is rope-WALKED, a near one
STRADDLED), but the board is now MULTI-COLOURED and salted with DECOYS:
  * A BOX is a solid diamond around a `6` whose fill uses SEVERAL colours
    (e.g. 12/14 top + 15 bottom); a matching RING is a hollow diamond showing
    the SAME colour palette.  WIN = every box dropped into its own-palette ring.
  * MANY extra hollow diamonds exist whose colour set matches NO box (decoys),
    plus a big solid colour-10 slab; these must be ignored.
  * The old colour==system identity breaks (colour 14 belongs to two boxes; a
    box's fill colour is not its endpoint-marker colour).  So rope->box identity
    is discovered by PROBING, not colour.

New legs (all general, no level constants):
  * `_flood_nonbg` — 4-flood a non-background blob, returning its palette+size.
  * `multicolor_boxes` — solid `6`-centred blobs (size>=8) with their fill
    palette; distinguishes real boxes from tiny chevron/ring `6`-cells.
  * `_ring_outline_cells` + `multicolor_rings` — cluster background-touching
    ring-outline pixels into hollow diamonds, each with a centre + colour set.
  * `marker_endpoints` — precise endpoint markers (a pixel whose 4 orthogonal
    neighbours are ALL 0 = active cursor, or ALL 3 = idle), filtering the
    left-border artefacts the old `_isolated_cells`-based finder tripped on.
  * `group_endpoints_by_box` — PROBE which box each marker drives (select+nudge
    on a clone, see which `6` moved); the general substitute for colour-keyed
    `endpoints_of` when colours are ambiguous.
  * `multicolor_systems` — match box<->ring by identical palette, attach probed
    endpoints, and pick a tracking colour UNIQUE to each box so the reused
    `box_center`/walk/straddle stay unambiguous; skips decoy rings.
  * `place_multicolor_boxes` — orchestrator (farthest-gap-first) that reuses the
    existing `walk_box_to` + `straddle_box_to` per system.
`play_level_4` is the one-liner `place_multicolor_boxes(env)`; the walk/straddle
inner machinery is unchanged and shared with L1-L3.  `python gkm_try.py` reports
`RESULT levels=4 moves=45 replay_ok=True`.

Reuse note: `place_multicolor_boxes` and `place_boxes_on_rings` are the same
orchestrator differing only in the DISCOVER step (colour-keyed `ring_systems`
vs palette-matched + probe-grouped `multicolor_systems`).  A future refactor
could unify them as `solve_systems(env, discover)` with the two discoverers as
arguments; kept separate for now since L2/L3 remain green and the multicolour
discoverer is heavier (probing).

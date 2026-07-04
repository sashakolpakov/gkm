# Draft: general level-progression priors (distilled from the sp80-L2 stall pattern)

Status: DRAFT — merge into `gkm_arena.PRECONCEPTIONS` (and the propose-task discipline
into `gkm_legs._propose_task`) only BETWEEN runs, never mid-flight. Prompt text below
is game-agnostic by construction: no game names, mechanics, action numbers, colours,
or geometry from any specific game.

## Evidence (why these rules; not part of any prompt)

Three independent L2 stalls share one signature — full per-level budget burned,
`play_level_2` never written:

1. Opus sp80 L2 (2026-07-02, 40 min): L1 solved in 4 moves; the agent even left a
   correct L2 diagnosis in legs_log.md ("layout appears vertically flipped ...
   gravity/pour direction likely inverted; legs may need an axis parameter") — then
   ran out of budget re-exploring instead of retrying its L1 method transformed.
2. Opus wa30 L2 under neutral priors (2026-07-02, 45 min): same signature.
3. Sonnet sp80 L2 (2026-07-04, 30 min): worst case — zero code AND zero notes; the
   whole round's discoveries evaporated at the budget kill.

Counterpoint: every L1 falls quickly (sp80 L1 in ~10 min: ~8.5 min probing, one build
pass, win) and all three L1 solutions independently converged on the same spine
(sense → align → trigger). Diagnosis: the failures are not capability but (a) no
prior that levels are TRANSFORMED REPRISES of each other, (b) no knowledge
persistence across budget kills, (c) no build-first discipline on the level AFTER a
success.

## Proposed PRECONCEPTIONS additions (verbatim prompt text)

LEVELS ARE VARIATIONS: after you clear a level, the next is usually the SAME mechanic
under a transformation — mirrored or flipped layout, inverted direction of motion or
gravity, swapped colours, shifted geometry, more objects, tighter timing. On a new
level, FIRST test on a clone whether your previous method still works after such a
transform; write your skills parameterized by axis/direction/colour rather than baked
to one orientation, so the transformed retry is one call. Open free-form re-discovery
only after transformed reuse demonstrably fails.

CONSUMABLE AND HAZARDOUS ACTIONS: an action may be a limited resource — usable only a
few times before penalty or game over, with or without a visible counter. Probe
unfamiliar actions on CLONES and check whether REPEATED use degrades or ends the game;
treat any such action as a scarce budget, spent only on clone-verified winning moves,
never on live experimentation.

TIMERS AND MOVE BUDGETS: a level may carry a hidden or displayed countdown (e.g. a bar
or row of cells that depletes every step). Measure early how many real steps you are
allowed; search on clones and commit only a SHORT verified sequence on the real env.

ALL-OR-NOTHING LEVELS: when no frame-derived measure correlates with progress, the
level may be a combination lock — nothing visibly improves until everything is right.
Do not hunt for a gradient that does not exist: enumerate candidate parameters
(positions, counts, orders) on clones where failure is free, and commit only the
verified combination.

DELAYED EFFECTS: an action's consequence may unfold over several following frames;
after acting on a clone, let the dynamics settle (a few known-safe steps) before
judging what the action did.

DECOY ACTIONS: one or more actions may do nothing at all; identify no-ops once and
stop re-probing them in every context.

## Proposed propose-task discipline additions (harness, not strategy)

- On a new level, run the PREVIOUS level's player on a clone first — its failure mode
  is cheap information about what changed.
- BUILD FIRST: commit a rough candidate player within the first third of the budget,
  then iterate observe → refine → verify in short cycles; never explore long without
  committing code.
- PERSIST KNOWLEDGE: append every discovered mechanic, hypothesis, and dead end to
  NOTES.md the moment you learn it; a budget kill must not lose knowledge. Read
  NOTES.md first on every round.

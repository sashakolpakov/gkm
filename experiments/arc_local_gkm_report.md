# ARC-AGI-3: working online + local adapters, and a GKM crack attempt

This documents three things, in order: (1) the **online** ARC-AGI-3 adapter,
fixed against the documented + live contract; (2) a **local** (offline) adapter
that runs the same real games with no network and no per-step key; (3) an honest
attempt to **crack a game locally with the GKM colimit-cone** seek leg.

Source of truth: <https://docs.arcprize.org> (REST overview, command/scorecard
references, `local-vs-online`) cross-checked against live calls to
`https://three.arcprize.org` and the installed `arc_agi` / `arcengine` toolkit.

## 1. Online adapter (`arc_agi3_adapter.ArcEnv`) — fixed and verified live

The previous adapter connected but carried three contract bugs that the live
API exposed (verified 2026-06-16/17 with a real key on `sb26` and `ls20`):

| Symptom | Cause | Fix |
| --- | --- | --- |
| Reward always `0.0` | Read `payload["score"]`; **the API has no `score` field** | `score := levels_completed` (out of `win_levels`) when no explicit score |
| State silently coerced | API state is `"NOT_FINISHED"`, not `"IN_PROGRESS"` | `GameState` aliases `NOT_FINISHED`/`NOT_PLAYED` onto the running/not-started values |
| `ACTION1` → `"game ls20 not found"` | RESET accepts the **short code** but the action loop needs the **full id** | `reset()` resolves `ls20` → `ls20-9607627b` via `/api/games` (`resolve_full_game_id`) |

Also added: `ACTION7` (undo) to the action set; action body trimmed to the
documented `{game_id, guid}` (+ `x,y` for `ACTION6`). Cookies remain
load-bearing (the RESET `AWSALB*` session-affinity cookie must ride every
action, or it routes to a backend without the game).

Verified live, end-to-end, through the public `ArcEnv("ls20")` path:

```text
GET  /api/games            200, 25 games ({game_id, title, tags, baseline_actions})
POST /api/scorecard/open   200 -> {card_id}
POST /api/cmd/RESET        200, body {game_id (full), card_id}
RESET/ACTION response      {guid, frame:[64x64 grids], state:"NOT_FINISHED",
                            levels_completed, win_levels, available_actions, ...}
POST /api/cmd/ACTION1..7   200, body {game_id, guid} (+ x,y for ACTION6)
POST /api/scorecard/close  200, body {card_id}
```

(This corrects the earlier `arc_live_report.md`, which claimed a `score` field
and that the bare short code drives actions — it does not.)

## 2. Local adapter (`arc_agi3_adapter.LocalArcEnv`) — same surface, no network

`docs.arcprize.org/local-vs-online`: the official `arc_agi` toolkit runs the
real games **locally** via `arcengine` at ~2000 FPS with no rate limits.
`LocalArcEnv` wraps it behind the **same** `reset()/step() -> Snapshot` surface
as `ArcEnv`, so every connector below it is unchanged.

- `operation_mode="normal"`: downloads the game source (`<game>.py` +
  `metadata.json`) into `environment_files/` once (needs a key), then runs it
  locally. Subsequent runs need no key.
- `operation_mode="offline"`: fully local, key-free, network-free; uses the
  already-downloaded files.

`FrameDataRaw` from `arcengine` carries exactly the API's fields
(`frame`, `state`, `levels_completed`, `win_levels`, `available_actions`), so
the mapping is faithful. Verified on `ls20` (directional, `win_levels=7`) and
`wa30`: the same directional actions move the world locally as online.
`arc_agi`/`arcengine` are imported lazily, so the test suite stays hermetic;
`tests/test_arc_agi3_adapter.py::LocalArcEnvTests` skips when they are absent.

## 3. Cracking a game with GKM — honest attempt

```bash
python3 experiments/run_arc_local_gkm.py --game wa30 --mode offline
python3 experiments/run_arc_local_gkm.py --game ls20 --steps 120
```

The pipeline: `LocalArcEnv` → scene functor → `cone_foraging.witness_seek_leg`
bound to a colour channel → priced by `levels_completed`. The script
auto-detects the avatar (the colour whose cells translate by `k·delta` under a
move), binds the seek leg to each candidate goal colour, and compares against a
random-action control.

**The loop works mechanically on real local frames.** On `wa30` it detects the
avatar (colour 14) and steers it toward the goal colour:

```text
t= 0 avatar@(34, 50) obs=  NW -> UP
t= 1 avatar@(34, 46) obs=  NW -> UP
t= 2 avatar@(34, 42) obs=  NW -> UP     <- moved 50->46->42, then blocked
t= 3 avatar@(34, 42) obs=  NW -> UP
...
```

**But no binding completes a level, and neither does random:**

```text
GKM seek cone (avatar=14), levels_completed per goal colour:
  goal 1 2 4 7 9  -> all 0/9
RANDOM control (200 steps): best 0/9 (reaches GAME_OVER)
```

### Why — and what it means for GKM

The 2026 keyboard games are **structured puzzles**, not reach-a-cell navigation:

- `wa30`: a Sokoban-like board. Avatar (14) translates cleanly, but a box sits
  directly on the path to the goal container; the avatar stalls against it. The
  win condition involves the boxes, not the avatar's position.
- `ls20`: a block-sliding puzzle. Pressing a direction slides a two-colour block
  (9/12) by a fixed step; corner legends show target patterns. Winning means
  matching a pattern, not reaching a coloured cell.

So the minimal `witness_seek_leg` — a one-state greedy "move toward the nearest
object of the bound colour" controller — is the wrong shape for these games. It
steers correctly and halts honestly, but cannot route around obstacles or push
blocks. This is a faithful **negative result with a known mechanism**, not a
failure of the wiring:

- The colimit-cone *correspondence* holds: ARC colour-slot perception ↔
  substrate channel; `CALL(seek, colour)` ↔ binding a leg to a colour. The cone
  *selection by reward* (try each binding, price by `levels_completed`) runs.
- What is missing is leg *expressiveness*. The next step is exactly the rest of
  the `cone_foraging` machinery: legs with more states (sidestep-when-blocked),
  a push/interact primitive, and goal induction over multi-colour targets —
  evolved/selected against `levels_completed`, with the random control as the
  honesty floor.

The connector already flagged this open problem: *"Discovering the right action
primitives per game is the open problem, and it lives in the connector, not in
the method."* This run makes it concrete and reproducible on real local frames.

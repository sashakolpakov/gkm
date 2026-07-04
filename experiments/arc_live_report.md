# Real ARC-AGI-3: Live Connection and Perception

> **Correction (2026-06-17):** two claims below were wrong and are fixed in
> `arc_local_gkm_report.md` + the `ArcEnv` docstring. (1) There is **no `score`
> field**; progress is `levels_completed`/`win_levels`. (2) The bare **short
> code does not drive actions** — RESET accepts it but the next ACTION returns
> "game not found"; the **full id** (`ls20-9607627b`) is required, so `reset()`
> now resolves it via `/api/games`. Local key-free play is also available now
> (`LocalArcEnv`).

This documents using the **real** ARC-AGI-3 API (not the synthetic stub). With
an `ARC_API_KEY`, the connector lists games, opens a scorecard, RESETs a real
game, and runs the scene functor on the real frame. This resolves the earlier
ARC-vs-synthetic confusion: the perception pipeline now runs on genuine ARC
frames.

## Reproduction

The key lives in a gitignored `.env` (`ARC_API_KEY=...`), auto-loaded by the
probe; no need to inline it.

```bash
python3 experiments/run_arc_live_probe.py --game ls20
python3 experiments/run_arc_live_probe.py --game wa30 --actions 5
python3 experiments/run_arc_live_probe.py --list
```

No key → the script prints instructions and exits cleanly (the test suite is
hermetic). No frames are committed (the repo does not vendor datasets).

## Verified live (2026-06-14, real key)

```text
auth header          X-API-Key   (Authorization: Bearer -> 401)
GET /api/games       200, 25 games (e.g. ls20-9607627b, wa30-ee6fef47, ...)
POST /api/scorecard/open    200 -> {card_id}
POST /api/cmd/RESET  200 after a few retries; body {game_id, card_id}
                     game_id is the lowercase SHORT code ("ls20"), NOT the
                     full listing id ("ls20-9607627b")
RESET response       {guid, frame (list of 64x64 grids), score, state,
                      available_actions}
POST /api/cmd/ACTION1-5     200, body {game_id, guid} (+ x,y for ACTION6)
                            returns the next snapshot {frame, score, state, ...}
POST /api/scorecard/close   200, body {card_id}
```

### The cookie fix (action round-trip)

The action round-trip initially failed (`400 game <id> not found`) even right
after a successful RESET. Reading the official `arc_agi` 0.9.9 toolkit's
`remote_wrapper.py` revealed why: it drives the API through a
`requests.Session` with a shared cookie jar, updating cookies before and after
every call. The RESET response sets a **session-affinity cookie** that must be
sent on subsequent ACTION requests; without it the action lands on a backend
that does not hold the game instance → "not found". The urllib client now uses
a cookie-aware opener (`HTTPCookieProcessor`). With cookies carried, ACTION1-5
return 200 with valid state/score. This was a transport bug, not provisioning.

Real frames retrieved and run through the scene functor
(`arc_agi3_adapter.extract_scene` / `arc_scene_atoms`):

```text
game ls20: 64x64, colours [1,3,4,5,8,9,11,12]
           objects/colour {1:2, 3:1, 4:2, 5:4, 8:3, 9:5, 11:1, 12:1}, available [1,2,3,4]
game wa30: 64x64, colours [1,2,4,7,9,14]
           objects/colour {1:1, 2:1, 4:3, 7:1, 9:4, 14:1}, available [1,2,3,4,5]
```

So the connected-component scene functor, written and previously tested only on
synthetic grids, runs unchanged on real ARC frames and extracts a colour/object
decomposition. This is the meaningful real-data result: real-ARC **perception**
works.

## Test-case results (two games)

Action round-trip verified on both; the honest observation is that arrow
actions move nothing on these initial screens:

```text
ls20: available_actions [1,2,3,4]; ACTION1-4 each -> 200, state IN_PROGRESS,
      score 0.0, changed_cells 0. No object translated under any arrow action.
wa30: available_actions [1,2,3,4,5]; ACTION1-5 each -> 200, state IN_PROGRESS,
      score 0.0, changed_cells 0. (ACTION5 has zero delta and cannot identify a
      translating avatar; the probe now skips zero-delta actions in the vote.)
```

## Actions never advance any game state (unresolved)

Extensive investigation (2026-06-16): across 7 games and 100+ actions, NO
action ever changed the frame, score, state, or levels_completed. Verified, so
this is not a quick payload bug:

- Payload matches the docs and the official `arc_agi` toolkit: actions POST to
  `/api/cmd/ACTION{n}` with `{game_id, card_id, guid}` (card_id IS required —
  the minimal `{game_id, guid}` returns 400); ACTION6 adds top-level `x,y`.
- Verified with BOTH a urllib cookie-jar client and `requests.Session` (the
  exact library and flow of the docs' bare-bones example). With requests,
  RESET succeeds first-try (clean cookies); actions still no-op.
- ls20 (the docs' own example game, which a random agent reportedly wins in
  ~100 steps): rendered as a maze with a colour-1 player at (20,32)-(21,33).
  ACTION1-4 each return 200 but the player does not move and the full 64x64
  frame is byte-identical; 25 random actions and 10 repeated ACTION1 also
  produce zero change. Single-frame responses (no hidden animation frames).
- `action_input.id` in responses is always 0 (the model's RESET default), so
  it is not a reliable echo of the applied action.

So RESET + perception are solid, but action application is a wall I could not
pass from this environment. Honest hypotheses, untested:
1. Session-pool pollution on the key: this session opened many scorecards and
   some closes 404'd; a per-key concurrent-session limit could leave new game
   instances frozen/no-op (consistent with the RESET "not found" retries too).
2. An account/key tier where actions are accepted (200) but not applied.
3. A protocol detail the official toolkit handles internally that is not in the
   public docs and that I could not replicate (installing arc_agi to A/B test
   was declined for supply-chain reasons).

Decisive next step: run the official toolkit (`arc_agi` / ARC-AGI-3-Agents)
with the same key. If it ALSO cannot move the games, the cause is the
key/account/session pool; if it can, diff its exact request bytes against this
client to find the missing detail. I stopped live probing here rather than
burn more quota on blind retries.

## Not yet verified / open

- **Meaningful gameplay.** Driving a game to a score change needs the right
  interaction modality per game (several are tagged `click`/`keyboard_click`,
  i.e. ACTION6 with coordinates, which this probe does not yet drive). The
  transport is confirmed; choosing useful actions is the next step.
- **Avatar discovery on real frames.** The translation heuristic found no
  avatar on ls20/wa30 (arrow actions inert). Real ARC needs richer avatar/
  dynamics discovery (e.g. clicking, or detecting non-rigid changes).
- **Goal induction / solving real games.** Real games have hidden, complex
  dynamics; the synthetic stub's clear@c/avoid@c templates are not expected to
  transfer. Solving is the frontier (humans ~100%, frontier systems <1%), not
  a near-term claim. The transport and perception are now in place to attempt
  it.

## Rate-limit note

The live service throttles bursts: RESET succeeds within ~15s (≈5 retries) when
calls are spaced, but a flurry of calls yields persistent `400 not found` until
a cooldown (~45s helped). The client (`arc_agi3_adapter.ArcEnv`) retries RESET
and actions with a constant ~3s poll; the right operational discipline is to
space sessions, not to hammer. These results were gathered, then probing was
stopped to respect the limit.

## What this means for the synthetic experiments

The synthetic `GoalGame` experiments (`run_arc_goal_induction.py`,
`run_bongard_*`-style demos, the scene-atom-discovery report) are now correctly
understood as **offline logic fixtures**: they validate the discovery +
free-energy goal-induction *machinery* deterministically, with no network and
no claim of being real ARC. The real-ARC demonstration is this live probe. The
two are complementary: the fixture proves the algorithm; the live probe proves
the perception runs on real frames. Solving real games waits on the action loop
(transport) and richer dynamics/perception (frontier).

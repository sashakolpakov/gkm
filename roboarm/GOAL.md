# Goal: Godel-Kolmogorov machine RoboArm (`rb01`)

**Status:** complete — RoArm/C920 v3 campaign promoted and replay verified
**Source of truth:** [GKM_ROBOARM_SIMULATOR_SPEC_V3.md](GKM_ROBOARM_SIMULATOR_SPEC_V3.md)
**Implementation boundary:** `/Users/sasha/gkm/roboarm`

## Objective

Build and demonstrate a standalone ARC-style—but not ARC-API-bound—RoboArm
puzzle solved by a real headless-Codex Godel-Kolmogorov machine loop.

The proposer receives:

- the complete documented six-action controller;
- an exact deterministic `128×72×3 uint8` RGB camera observation;
- a separately timestamped RoArm feedback packet paired with the nearest C920s
  frame;
- sparse `levels_completed`, terminal state, and sealed outcomes from earlier
  generations.

It receives no connector, live environment, socket, token, clone handle,
private mechanics source, object coordinates, attachment flag, target
predicate, canonical path, or authority to execute actions. It may author only
retained proposal code and bounded declarative scenarios.

The host alone:

1. validates the closed scenario schema;
2. runs isolated authoritative digital-twin preflight;
3. applies a deterministic safety FSA;
4. mints a single-use commit permit only for a complete safe candidate after
   genuine earlier failure evidence;
5. guards each committed step against the admitted preflight camera and
   telemetry hashes;
6. seals observations and feeds only public projections to later generations;
7. verifies retained source from a fresh workspace and independently replays
   the exact first-acquisition boundary before promotion.

The desired scientific record is a real sequence of hypothesis, attempted
action, camera/telemetry observation, failure, retained-source revision,
safety-authorized success, and replay-gated promotion. A canonical route,
mocked proposer, staged failure, browser script, or polished animation is a
test—not discovery.

## Current mandatory round

The first operational round uses the pinned RoArm-M2-S chain, deterministic
command-space IK, swept collision, one workpiece, one barrier, one target bin,
bilateral grasp, rigid carry, release, gravity, support settlement, and sparse
completion.

The authoritative sensor contract is `rb01-roarm-c920-v3`:

- processed RGB8 C920s observation, shape `(72, 128, 3)`, from a declared
  1080p/30 UVC source with 78-degree diagonal field;
- deterministic pinhole approximation with calibrated pose;
- stock-style T=1051 encoder/derived-XYZ/load/torque-switch/voltage feedback,
  host controller state, and explicit cross-device timestamps/skew;
- no HUD or telemetry encoded in pixels;
- no semantic palette observation in the operational round.

Campaign `rb01-roarm-c920-v3-zero-seed-20260731` is the current live
acquisition under `rb01-roarm-c920-v3`. Superseded v2 campaign and viewer
artifacts were deleted; legacy contracts still fail viewer admission.

## Browser role

The browser is replay visualization only. For each saved genuine failure or
success it must show:

- the exact stored 128×72 RGB bytes supplied to the machine;
- the exact nearest-frame/arm-feedback pair and recorded timing skew;
- the recorded public action and sparse reward boundary;
- a larger, clearly labeled state-synchronized 3D human replay view;
- host-only mechanics events and disposition for explanation.

The larger view never becomes proposer input. The browser cannot propose,
preflight, commit, repair, select a policy, author a verdict, or promote
anything.

At the end of this goal, run the new acquisition campaign and verify a browser
demonstration containing several genuine failed replays and at least one
committed/fresh-replay success. Desktop and narrow captures may be retained
locally, but the versioned evidence boundary is the replay-validated export and
its browser tests.

## Required evidence

- Exact RGB camera and telemetry contracts tested for shape, ownership,
  determinism, privacy, and synchronization.
- Connector traces bind independent frame and telemetry hashes.
- Preflight/commit and fresh replay reproduce both sensor products.
- Legacy 4096-byte indexed artifacts fail schema admission.
- The proposer sandbox exposes sealed RGB/telemetry but no private source,
  network, socket, connector, or outside write path.
- A zero-seed live Codex campaign records at least one concrete failed grasp,
  rejected motion, or collision outcome.
- A later generation actually changes retained source using sealed evidence.
- Adjacent clean-admitted generations are audited for marginal source growth
  and unchanged invoked-leg reuse without relabeling revisions as solved
  levels.
- A safe candidate receives the one-use permit and passes source verification
  plus independent exact replay.
- Viewer export replays and revalidates every stored action, frame, telemetry
  packet, sparse reward, terminal state, and host snapshot.
- Browser playback shows genuine failure and success; unit tests and the
  production build are the versioned gates. Local page/console inspection is
  supplementary unless its evidence is checked in.
- All source and persistent artifacts remain below `roboarm/`; `/arc` is
  read-only and unchanged.

## Constraints

- ARC-style is a research interaction pattern, not an ARC runtime dependency.
- Use headless Codex only; no local model backend.
- The safety FSA may enforce generic action, trajectory, load, collision,
  budget, provenance, and replay invariants, but may not encode the task
  solution.
- Camera and telemetry must be operationally plausible, deterministic, and
  generated from authoritative mechanics. Do not relabel a symbolic raster as
  a camera.
- Do not expose private world snapshots to the proposer.
- Do not fake contact, attachment, carry, collision, release, settlement, or
  success with sprites, keyframes, video, or decorative tweening.
- Do not preserve obsolete campaign evidence as a fallback.
- Do not claim photorealism, sim-to-real validity, or physical-robot safety from
  this deterministic simulator result.

## Completion evidence

The corrected zero-seed campaign completed on 2026-07-31:

- 4 live headless-Codex proposer generations authored 7 bounded scenarios;
- 321 authoritative preflight actions recorded an empty grasp, a successful
  45 mm pickup, a motion rejection, and collision-rejected descent attempts;
- generation 2 reached the sparse goal as an uncommitted cyan-bin release;
- generation 3 replayed that route as a candidate and the FSA denied it with
  `preflight_not_commit_safe` because it retained three rejected descents;
- generation 4 removed exactly those no-op descent requests and proposed a
  clean 59-action route;
- the host issued one permit, committed 59 actions, verified fresh retained
  source for 59 actions, and independently replayed the exact 59-action path;
- `promoted`, `genuine_failed_attempt`, `revised_after_failure`,
  `source_verified`, and `path_replayed` are all true;
- the replay viewer contains three distinct failure replays, the authorized
  commit, and the independent exact replay under the v3 sensor contract.
- the construction-lineage profile records historical net growth
  `191 → 323 → 168 → 172`, conditional-AST novelty
  `3013 → 4095 → 4198 → 2740`, and transitively invoked unchanged legs
  `0 → 1 → 5 → 8`; it explicitly records zero strict direct-call witnesses
  and makes no solved-level sawtooth claim.

The simulator is deliberately quasi-static. A valid grasp is rigidly attached
until `OPEN_GRIPPER`; swept held-object collisions can reject motion, but this
round does not simulate acceleration-driven jaw slip or an involuntary
attached-object drop.

## Definition of done

The new RGB-camera contract, full proposal/FSA boundary, zero-seed acquisition,
fresh verification, exact replay, and replay viewer export all passed from the
same campaign identity. Test-only canonical behavior remains clearly
segregated. No old indexed campaign or viewer export is used in the final
claim. Browser unit tests and a production build verify the checked-in viewer;
local interactive captures are not part of the versioned evidence.

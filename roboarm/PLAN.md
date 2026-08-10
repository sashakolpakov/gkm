# Godel-Kolmogorov machine RoboArm implementation plan

**Status:** complete — all release gates passed
**Goal:** [GOAL.md](GOAL.md)
**Normative specification:** [GKM_ROBOARM_SIMULATOR_SPEC_V3.md](GKM_ROBOARM_SIMULATOR_SPEC_V3.md)
**Revised:** 2026-07-31

## 1. Fixed interpretation

`rb01` is a standalone Python environment. “ARC-style” means a compact
observation/action/sparse-reward experiment with deterministic replay and
program growth. It does not mean an ARC API implementation.

All writes remain below `/Users/sasha/gkm/roboarm`. The parent `/arc` project is
read-only research context. The RoboArm runtime does not import an ARC harness,
loader, game class, metadata format, action enum, or sprite system.

The scientific split is:

```text
headless Codex proposes programs and bounded scenarios
        ↓ no actuation authority
host validates → preflights → safety-FSA gates → optionally commits
        ↓ sealed public RGB camera + telemetry
later generation revises retained source
        ↓
fresh source verification → exact replay → promotion
        ↓
browser illustrates saved evidence only
```

## 2. Sensor contract migration

Status: complete.

- Replace the operational `(72, 128)` indexed raster with exact
  `(72, 128, 3)` RGB8 camera bytes.
- Use a deterministic 16:9 C920s approximation with declared 1080p/30 source,
  78-degree diagonal field, pose, perspective, occlusion, fitted articulated
  geometry, material shading, and contact shadows.
- Keep controller telemetry outside the image.
- Expose host controller state separately from T=1051 encoder angles,
  firmware-derived XYZ, raw servo loads, torque-switch flags, voltage, and
  camera/arm timestamps.
- Do not invent aperture, contact-force, collision-reason, or synchronized-clock
  sensors.
- Exclude object coordinates, attachment flags, target predicates, private
  mechanics events, and safety verdicts.
- Version the contract as `rb01-roarm-c920-v3`.
- Reject legacy 4096-byte indexed artifacts rather than decode them.

Gate:

- camera and feedback are defensive, deterministic, separately hashed, and
  bound with explicit pairing skew at every boundary;
- the scored round exposes no HUD/palette fallback.

## 3. Connector and safety binding

Status: implemented and tested.

- Record camera calibration, RGB bytes/hash, telemetry packet/hash, sparse
  reward, terminal state, and action for reset and each transition.
- Bind a commit permit to both expected frame hashes and expected telemetry
  hashes.
- Before each committed action, replay on a clone and compare both products.
- Keep full `visual_state` host-only for safety inspection and browser
  reconstruction.
- Project only public camera, telemetry, action, reward, terminal, and host
  disposition into later proposer generations.
- Continue to classify concrete empty-grasp, motion-rejection, and
  collision-rejection evidence from trusted host state.

Gate:

- forged/reused permits, malformed telemetry, private-field leaks, or
  preflight/commit sensor divergence fail closed.

## 4. Proposal workspace

Status: complete, including the production containment probe and live proposer.

- Public README documents exact actions, camera calibration, and telemetry.
- Generic `perception.py` decodes RGB8, computes generic color summaries,
  quantized connected components, frame deltas, and telemetry histories.
- Helpers contain no semantic object labels, target coordinates, palette roles,
  mechanics source, or canonical solution.
- Headless Codex receives no live environment, connector, socket, token, clone
  handle, network access, or private repository view.
- Scenario output remains closed to identifier, kind, hypothesis, expected
  observation, and legal public actions.
- Model-authored frame, telemetry, reward, terminal, outcome, authorization,
  and safety fields fail schema validation.

Gate:

- the no-model production sandbox probe reads exactly 27,648 camera bytes and
  the public telemetry packet while all private/source/network/write channels
  remain denied and connector action counts remain zero.

## 5. Replay and browser payload

Status: complete.

- Exact replay records both sensor products and their hashes.
- Viewer export reconstructs every turn from seed and public actions, then
  rejects any mismatched RGB bytes, telemetry, reward, terminal state, or
  snapshot.
- Browser artifacts and manifest use schema version 3 and carry camera
  calibration.
- The exact inset displays stored RGB bytes, not a browser recomputation.
- The telemetry panel displays the stored public packet.
- The larger Three.js scene is labeled as a state-synchronized human replay
  view, not machine input.
- Remove the obsolete browser palette renderer.
- Keep the canonical mechanics route under `/mechanics-test` and label it as
  developer regression only.

Gate:

- browser unit tests and production build pass;
- live-run automation sees the exact RGB canvas and genuine failed/promoted
  artifacts without browser errors.

## 6. Evidence rewrite

Status: complete.

The obsolete indexed-frame contents at the following paths were deleted and
were not archived or used as a fallback:

```text
artifacts/gkm/campaigns/<superseded-v2-campaign>/
web/public/campaign/
artifacts/browser/gkm-live-run/
```

Only the default viewer path was repopulated, from the new v3 campaign and its
replay-validated export. Before the live rerun:

- remove old scientific claims from reports;
- regenerate the developer mechanics fixture under sensor schema v2;
- run the full Python and browser suites;
- run the production Codex sandbox probe.

Gate:

- no report, default route, manifest, or selectable artifact cites the deleted
  indexed campaign as evidence.

## 7. Zero-seed acquisition rerun

Status: complete.

Run the same canonical identity from an empty solver lineage:

```bash
.venv/bin/python -m roboarm_game.gkm_runner \
  --campaign-id rb01-roarm-c920-v3-zero-seed-20260731 \
  --provider codex \
  --model gpt-5.6-sol \
  --reasoning-effort high \
  --minutes 30 \
  --generations 12 \
  --scenarios-per-generation 8 \
  --actions-per-scenario 160 \
  --committed-budget 2000 \
  --clone-budget 24000
```

The live campaign:

1. began with empty retained source;
2. proposed genuine falsifiable experiments from camera and telemetry;
3. recorded concrete operational failures;
4. fed sealed failures into later generations;
5. retained actual source revisions;
6. proposed a complete candidate without known-rejected actions;
7. received the one-use safety permit;
8. matched preflight during committed execution;
9. reproduced the candidate from fresh retained source;
10. passed independent exact RGB/telemetry replay at the first sparse acquisition
    boundary.

Gate:

- `promoted=true`, `genuine_failed_attempt=true`,
  `revised_after_failure=true`, `source_verified=true`, and
  `path_replayed=true`;
- the canonical mechanics fixture is not referenced by proposer payload or
  counted as discovery.

Measured result:

- 4 proposer generations and 7 scenarios;
- 321 isolated preflight actions;
- 59 committed actions, 59 fresh-source verification actions, and 59
  independent exact-replay actions;
- empty-grasp, successful 45 mm pickup, motion-rejection, and collision-
  rejection evidence preceded the successful correction;
- generation 2 discovered the goal in an experiment whose path still contained
  three rejected motions;
- generation 3 proposed that unsafe replay and the FSA rejected it;
- generation 4 proposed the clean 59-action candidate;
- one candidate was rejected by the FSA and exactly one commit permit was used.

## 8. Live viewer demonstration

Status: complete.

After promotion:

```bash
.venv/bin/python tools/export_campaign_viewer.py \
  artifacts/gkm/campaigns/rb01-roarm-c920-v3-zero-seed-20260731 \
  --destination web/public/campaign
cd web
npm test
npm run build
npm run dev
npm run live-run
```

Select several representative genuine failures, the discovery commit, and the
independent exact replay. Desktop, mid-attempt, terminal, interactive-orbit,
and narrow-viewport captures are optional local presentation evidence; the
versioned gate is the replay export plus browser tests and production build.

Gate:

- every visible replay is tied to the new sensor schema and same campaign;
- exact RGB input and telemetry remain synchronized;
- failed attempts visibly show the relevant unsuccessful grasp or rejected
  motion;
- success visibly shows grasp, carry, release, gravity settlement, and sparse
  completion;
- browser tests and the production build pass against the exported evidence.

Measured result:

- 3 distinct failure replays, 1 discovery-commit replay, and 1 independent
  exact replay;
- the checked-in viewer has 20 passing browser tests and a successful production
  build;
- earlier local Chrome inspection reported successful WebGL 2 playback, but its
  captures are not part of the versioned evidence boundary.

## 9. Final report

Status: complete.

`SAFETY_FSA_REPORT.md` and `OPERATIONAL_SLICE_REPORT.md` report the new campaign
only:

- proposer generations and scenarios;
- concrete failed mechanisms;
- source revisions;
- preflight, committed, fresh-verification, and exact-replay actions;
- safety rejections and permit evidence;
- camera and telemetry schema/hash evidence;
- browser replay count and screenshots;
- full test/build/sandbox results;
- local-only deployment disposition if external hosting remains unavailable.

Do not mention the deleted indexed campaign as retained evidence, historical
fallback, or alternate result.

## 10. Release gates

- **Boundary:** no write outside `roboarm`; no change under `/arc`.
- **Observation:** exact RGB8 `(72, 128, 3)` plus separate telemetry.
- **Privacy:** no private world labels in proposer-visible evidence.
- **Safety:** proposal-only Codex; host-only deterministic permit path.
- **Discovery:** real failed hypothesis followed by retained-source revision.
- **Promotion:** fresh source and exact replay, not clone-only success.
- **Viewer:** replay-only, exact input visibly distinct from larger human view.
- **Language:** user-facing copy says “Godel-Kolmogorov machine,” never “GKM
  SYSTEM.”
- **Evidence:** obsolete indexed campaign absent, new campaign authoritative.

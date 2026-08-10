# Godel-Kolmogorov machine RoboArm

`roboarm` is the isolated implementation home for the standalone `rb01`
ARC-style tabletop manipulation experiment.

> **Known actuator, unknown world dynamics.**

“ARC-style” describes the experimental loop: compact visual input, six
discrete actions, sparse completion, exact replay, hypothesis revision, and
program growth. This project has no dependency on an ARC API, engine, loader,
metadata format, sprite system, or game class.

## What is implemented

The operational round uses:

- the pinned Waveshare RoArm-M2-S Xacro transform chain;
- deterministic cylindrical command-space IK;
- joint bounds and swept full-body, gripper, self, fixed-workcell, and
  held-object collision;
- one workpiece, barrier, and target bin;
- bilateral enclosure, attachment, carry, release, gravity, support, and sparse
  target completion;
- a deterministic 128×72 RGB8 observation representing a downsampled C920s
  1080p/30 UVC capture;
- separately timestamped stock-style RoArm T=1051 feedback and host command
  state, paired with the nearest webcam frame.

The operational image is camera-only. It contains perspective, occlusion,
articulated geometry, material shading, and contact shadows. It contains no
painted HUD or semantic palette. The I/O packet separates host controller state
from the arm's encoder angles, firmware-derived XYZ, raw servo loads,
torque-enable flags, and supply voltage. It exposes no fictitious contact
sensor, metric jaw aperture, or collision reason.

```python
from roboarm_game import make_env

env = make_env("rb01-v1", seed=0, scenario="round-1")
rgb = env.reset()          # uint8, shape (72, 128, 3)
telemetry = env.telemetry()
next_rgb = env.step(2)
branch = env.clone()
```

The unscored Phase-0 calibration shell remains a protocol regression fixture.
It is not the operational campaign observation.

## Scientific loop

The production proposer is headless Codex. It receives the complete documented
six-action apparatus, exact RGB camera frames, paired controller
telemetry, sparse reward, terminal state, and sealed outcomes from earlier
generations.

It receives no connector, live environment, socket, token, clone handle,
private mechanics, object coordinates, target predicate, canonical route,
browser implementation, or authority to run an action.

Each model turn emits only declarative scenarios. The trusted host then runs:

```text
proposal
  → closed schema
  → isolated authoritative preflight
  → deterministic safety FSA
  → optional one-use commit permit
  → stepwise camera + telemetry interlock
  → fresh source verification
  → independent exact replay
  → promotion
```

Experiments never commit. A candidate cannot commit until an earlier generation
has produced a genuine operational failure and a later retained-source revision
has proposed a complete safe goal-reaching sequence.

The canonical 63-action mechanics route is a test only. It is never discovery
evidence.

## Current v3 live result

Campaign `rb01-roarm-c920-v3-zero-seed-20260731` completed from an empty
retained-source lineage using headless Codex (`gpt-5.6-sol`, high reasoning):

- 4 proposer generations, 7 scenarios, and 321 isolated preflight actions;
- genuine empty-grasp, motion-rejection, and collision-rejection evidence;
- a 45 mm pickup frontier and an uncommitted cyan-bin sparse success;
- one 62-action candidate denied because it retained three rejected descent
  commands;
- a revised 59-action candidate admitted and committed by the safety FSA;
- 59 fresh-source verification actions and 59 independent exact-replay
  actions;
- replay-gated promotion with clean protocol and no proposer actuation
  authority.

The obsolete v2 campaign and viewer artifacts were deleted before this run and
are not retained as fallback evidence. The authoritative result is summarized
in the checked-in [operational slice report](OPERATIONAL_SLICE_REPORT.md) and
[safety-FSA report](SAFETY_FSA_REPORT.md). The replay-validated portable
campaign receipt and selected failure/success traces are under
[`web/public/campaign/`](web/public/campaign/); full local campaign workspaces
under `artifacts/` are intentionally ignored.

### Retained legs and marginal complexity

The admitted source lineage now has a machine-readable and human-readable
audit. Across generations 1–4, historical positive net-growth complexity is
`191 → 323 → 168 → 172`, while conditional normalized-AST novelty is
`3013 → 4095 → 4198 → 2740` zlib-9 bytes. Transitively invoked,
normalized-AST-stable
`legs.py` definitions grow from `0 → 1 → 5 → 8`; the winning generation reuses
eight unchanged lower-level legs through its new composition.

This is intentionally labeled a **campaign construction profile**, not a
solved-level sawtooth. `rb01` has one promoted round, the direct unchanged-leg
call count is zero, and the final conditional-AST drop is about 35% rather than
the predeclared half-or-more sharp threshold. See the
[`lineage profile`](web/public/campaign/lineage_profile.json)
or the corresponding panel in the default browser viewer.

## Browser replay viewer

The browser illustrates saved attempts after the machine has acted. It shows:

- the exact 128×72 RGB bytes supplied to the proposer;
- the exact paired RoArm feedback packet and camera timing;
- public actions, sparse reward, trace role, and disposition;
- a larger, explicitly labeled state-synchronized 3D human replay view;
- host-only mechanics events for explanation.

The browser cannot propose, preflight, actuate, repair, verify, or promote a
solver.

The `/mechanics-test` route contains only developer regression fixtures. The
default route accepts only a replay-validated live campaign export.

## Verify locally

From `/Users/sasha/gkm/roboarm`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
PYTHONPATH=src:. .venv/bin/pytest -q

cd web
npm ci
npm test
npm run build
```

The current checked-in suites contain 91 Python tests and 20 browser tests.

Regenerate the segregated mechanics fixtures:

```bash
PYTHONPATH=src .venv/bin/python tools/export_mechanics_fixture.py
```

Run the no-model production sandbox proof:

```bash
.venv/bin/python tools/check_codex_sandbox.py
```

It must read 27,648 RGB bytes and public telemetry while private imports,
private reads, loopback TCP, Unix sockets, and writes outside the campaign
workspace are blocked—and while connector action counts remain zero.

## Acquire a v3 zero-seed campaign

A fresh reference-device-contract campaign can be produced with:

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

This live command additionally requires an installed, authenticated Codex CLI.
It writes the full campaign workspace below `artifacts/`, which is intentionally
excluded from version control; promote only validated portable exports to
`web/public/campaign/`.

The prior indexed-frame campaign and browser export were deleted before this
run. They are not preserved as fallbacks or valid evidence.

Regenerate the report and replay viewer:

```bash
.venv/bin/python tools/report_gkm_campaign.py \
  artifacts/gkm/campaigns/rb01-roarm-c920-v3-zero-seed-20260731

.venv/bin/python tools/export_campaign_viewer.py \
  artifacts/gkm/campaigns/rb01-roarm-c920-v3-zero-seed-20260731 \
  --destination web/public/campaign \
  --replace

cd web
npm run dev
# Another shell:
npm run live-run
```

The mechanics fixture and default campaign viewer now both use v3. The default
viewer export is tied to the promoted source and independently replay-validated
campaign receipts.

## Documents

- [GKM_ROBOARM_SIMULATOR_SPEC_V3.md](GKM_ROBOARM_SIMULATOR_SPEC_V3.md) —
  design and acceptance specification; the commands in this README and the
  checked-in reports describe the implemented release where early examples
  were superseded.
- [GOAL.md](GOAL.md) — current RGB-camera campaign goal.
- [PLAN.md](PLAN.md) — execution and release gates.
- [PHASE0_REPORT.md](PHASE0_REPORT.md) — standalone protocol proof.
- [SAFETY_FSA_REPORT.md](SAFETY_FSA_REPORT.md) — proposal/actuation boundary.
- [OPERATIONAL_SLICE_REPORT.md](OPERATIONAL_SLICE_REPORT.md) — mechanics,
  campaign, and browser verification.
- [HARDWARE_IO_REFERENCE.md](HARDWARE_IO_REFERENCE.md) — manufacturer-backed
  sensor, webcam, timestamping, and connector contract.
- [`src/roboarm_game/README.md`](src/roboarm_game/README.md) — the concise
  proposer-visible apparatus contract packaged with `roboarm_game`.

## Repository boundary

All project code, tests, caches, workspaces, reports, replays, and generated
artifacts belong under `/Users/sasha/gkm/roboarm`.

`/arc` may be inspected read-only for research discipline. No runtime import,
generated artifact, cache, or source change is permitted there.

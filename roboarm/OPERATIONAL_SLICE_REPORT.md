# Operational slice verification report

**Verified:** 2026-07-31
**Sensor contract:** `rb01-roarm-c920-v3`
**Campaign status:** v3 promotion complete and replay verified

## Implemented mechanics

The standalone `pick-place` round uses:

- the pinned RoArm-M2-S Xacro transform chain;
- deterministic cylindrical IK refined against exact transforms;
- bounded joint motion and swept full-body, gripper, self, fixed-workcell, and
  held-object legality;
- a physical workpiece, barrier, and target bin;
- opposing-jaw bilateral enclosure;
- rigid attachment and carried-object collision;
- release, gravity, highest-support settlement, and sparse completion.

The canonical 63-action pick/carry/release route remains a developer mechanics
regression fixture only.

Scene `pick-place-v2` uses the same collision solids in authoritative Python,
the deterministic camera renderer, and the browser reconstruction. Collision
checks cover the base, cylindrical links and joints, palm and both moving jaws,
the carried workpiece, table, barrier body and caution cap, target floor and
four walls, rear safety wall, and safety posts. Gripper aperture changes are
swept, and the carried-object offset rotates in the gripper frame. A rejected
command atomically retains the last legal configuration rather than displaying
interpenetration.

This is deterministic quasi-static robotics, not a general rigid-body engine.
A valid grasp is a rigid TCP-relative attachment. Held-object boxes participate
in swept collision and can cause atomic motion rejection. Load is a
mass/gravity/reach moment estimate. Opening the gripper detaches the object and
settles it immediately on the highest valid support. V1 has no velocity,
acceleration, impulse, bounce, sliding, jaw-force/friction-cone, or involuntary
slip integration; an attached object does not drop unless the gripper opens.

## Authoritative observation

The scored round no longer uses the symbolic palette/HUD raster.

`frame()` returns exact RGB8 bytes with shape `(72, 128, 3)` from a deterministic
pinhole approximation with a 78-degree diagonal and 43.3-degree vertical field
of view. The CPU sensor includes perspective,
occlusion, articulated robot geometry, driven jaws, workcell surfaces,
material shading, contact shadows, and fixed vignetting.

`telemetry()` separates host command/interlock state from stock-style T=1051
feedback: encoder angles, firmware-derived XYZ, signed raw servo loads,
torque-enable flags, and supply voltage. It also records independent arm request,
arm response, and camera capture timestamps plus their skew. It does not expose a
metric jaw aperture, contact force, simulator collision category, object state, or
goal predicate. Camera pixels contain no telemetry, semantic labels, object
coordinates, or success banner.

The connector records and hashes both products independently.

## Browser role

The viewer uses the exact stored RGB bytes for the machine-input inset and the
exact stored public telemetry for the controller panel. Its larger Three.js
scene is explicitly labeled as a state-synchronized human replay view.

Host-only snapshots drive that explanation layer but are removed from
proposer-visible evidence. The browser has no planning, action, preflight,
repair, verification, or promotion capability.

The old browser palette renderer was removed. `/mechanics-test` remains a
segregated developer route.

The human-view orbit is constrained to the open front of the cell, panning is
disabled, and the maximum camera distance is bounded. The rear safety wall
therefore remains visible collision geometry without being able to obscure the
robot during interactive inspection.

## Verification

```text
Python: 91 tests passed
Browser: 20 tests passed
Production Next.js build: passed
```

The tests cover:

- RGB shape, dtype, determinism, defensive ownership, and visual diversity;
- telemetry schema, privacy, synchronization, and ownership;
- exact cloning and mechanics replay;
- bilateral grasp, carried collision, target-floor rejection, release, and
  settlement;
- camera and telemetry hashing from preflight through commit;
- closed proposer schema and no-actuation boundary;
- fresh retained-source and exact-path replay gates;
- browser schema-v3 parsing and exact RGB decoding;
- developer-fixture segregation;
- Python/browser mechanics transform parity.

## Zero-seed acquisition

The live headless-Codex Godel-Kolmogorov machine campaign used no preloaded
solver route:

```text
Campaign: rb01-roarm-c920-v3-zero-seed-20260731
Sensor: rb01-roarm-c920-v3
Generations: 4
Proposed scenarios: 7
Authoritative preflight actions: 321
Committed actions: 59
Fresh-source verification actions: 59
Independent exact-replay actions: 59
```

Generation 1 bracketed the grasp frontier: 60 mm closed empty, 45 mm carried
the workpiece, and the attempted 30 mm descent was rejected. Generation 2
reached sparse completion with an uncommitted cyan-bin release, but the route
contained three collision-rejected descent requests. Generation 3 replayed
that path as a candidate; the FSA reproduced the violations and denied it with
`preflight_not_commit_safe`. Generation 4 removed exactly those rejected
no-op requests and proposed the clean 59-action candidate that was committed.

The committed run, fresh retained-source verification, and independent exact
replay reproduced every action, RGB frame, telemetry packet, sparse reward, and
terminal boundary.

## Browser evidence

The replay-validated export contains:

- generation-1 unloaded close / empty grasp;
- generation-1 motion rejection;
- generation-3 candidate safety rejection;
- the generation-4 safety-authorized commit;
- the independent promoted exact replay.

The replay export revalidated every action, RGB frame, telemetry packet,
sparse reward, terminal state, and host snapshot before writing schema-v3
artifacts. Browser unit tests and the production build are the release gates;
historical screenshot captures were deleted with the superseded artifacts.

The viewer is designed for local playback at `http://127.0.0.1:3000`. The
checked-in release evidence is the schema-v3 export under
`web/public/campaign/`; full campaign workspaces and local browser captures are
not versioned.

## Evidence disposition

The earlier indexed-camera and incomplete-collision contents were deleted
before the rerun and are not fallbacks. The checked-in claim is supported by
the scene-v2 RGB-camera implementation, tests, reports, and replay-validated
schema-v3 viewer export. Any prior local Chrome captures are presentation
evidence only and are not part of the repository's reproducible record.

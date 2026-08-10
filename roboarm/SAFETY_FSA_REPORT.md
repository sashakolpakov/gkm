# Headless-Codex proposal boundary and safety-FSA report

**Date:** 2026-07-31
**Sensor contract:** `rb01-roarm-c920-v3`
**Status:** v3 campaign promoted and independently replay verified

## Authority boundary

Headless Codex is a proposal-only component in the Godel-Kolmogorov machine
loop. Its generation workspace contains only:

```text
README.md
ROUND.md
evidence.json
gkm_propose.py
interface.py
legs.py
perception.py
players.py
protocol.py
scenario_contract.py
solve.py
solver_index.md
```

It contains no connector, `Arena`, socket, token, live environment, clone
handle, action method, mechanics source, private world state, canonical route,
browser implementation, parent `/arc` content, repository history, or
credentials.

The production permission profile denies general network access, all Unix
sockets, private source reads, and writes outside the generation workspace.
The model turn ends before the host executes any proposed action.

## Closed proposal contract

The untrusted proposal may contain only:

- scenario identifier;
- `experiment` or `candidate`;
- falsifiable hypothesis;
- expected public observation;
- 1–160 legal action IDs from `(1, 2, 3, 4, 5, 6)`.

Model-authored frames, telemetry, rewards, terminal state, observations,
events, `passed`, authorization, safety, success, and verdict fields fail
schema validation.

## Deterministic safety path

```text
RECEIVED
  → CONTRACT_VALIDATED
  → PREFLIGHTING
  → PREFLIGHT_OBSERVED
  → {VERIFIED | REJECTED | COMMIT_DEFERRED | COMMIT_AUTHORIZED}
  → COMMITTING
  → OBSERVED
  → VERIFIED
  → SEALED
```

An `experiment` always stops after isolated observation and can never commit.
A `candidate` must:

- complete its entire authoritative preflight;
- remain within action, load, trajectory, collision, and budget policy;
- reach the sparse goal;
- follow a concrete failed observation from an earlier generation;
- receive a single-use in-memory permit.

The FSA contains no target coordinate, grasp depth, canonical path, or task
recipe. It enforces generic safety, provenance, state-transition, and replay
invariants only.

## RGB-camera and telemetry interlock

The connector now seals two independent public sensor products at reset and
after every action:

1. exact 128×72×3 RGB8 camera bytes and SHA-256;
2. exact structured controller telemetry and canonical-JSON SHA-256.

Telemetry separates the T=104 host command and generic interlock state from
stock-style T=1051 encoder angles, firmware-derived XYZ, signed raw servo loads,
torque-enable flags, and voltage. Independent arm request/response and camera
capture timestamps record pairing skew. It omits metric jaw aperture, contact
force, simulator collision category, object coordinates, attachment flags,
support identity, target predicates, private events, and safety verdicts.

A commit permit is bound to every expected camera hash and telemetry hash from
the admitted preflight. Before each committed transition, a clone must
reproduce both products. Fresh-source verification, exact replay, public
projection, and browser export also bind both.

Legacy 4096-byte indexed frames and traces without telemetry use an obsolete
schema and fail admission.

## Regression evidence

Observed local verification after the sensor rewrite:

```text
PYTHONPATH=src:. .venv/bin/pytest -q
91 passed

cd web
npm test
3 files / 20 tests passed

npm run build
Next.js production build completed
```

The injected campaign regression still proves:

- isolated experiments cannot commit;
- empty grasp is concrete failure evidence;
- clone-only success is deferred;
- a known collision path is rejected before commit;
- a safe revised candidate receives the permit;
- preflight and commit camera/telemetry sequences match;
- retained source and exact action replay pass before promotion.

This injected sequence is test infrastructure, not discovery.

## Production containment evidence

The production no-model headless-Codex sandbox proof passed after the campaign.
It exposed exactly 27,648 RGB bytes and the synchronized public telemetry
packet while blocking private imports and reads, loopback TCP, Unix sockets,
and writes outside the proposer workspace. Connector preflight and commit
action counts both remained zero during the proof. Its generated probe
workspace was removed after validation.

All four live proposer processes reported:

- `actuation_channel_present=false`;
- `sandbox_network_enabled=false`;
- `web_search_disabled=true`;
- no residual process group after completion.

## Live campaign evidence

The contents of campaign
`rb01-roarm-c920-v3-zero-seed-20260731` are solely the v3 RGB/telemetry rerun:

- 4 clean proposer generations and 7 admitted scenarios;
- 321 isolated FSA preflight actions;
- concrete `empty_grasp`, motion-rejection, and collision-rejection evidence;
- retained-source revisions after those outcomes;
- generation 1 established the 45 mm attached-carry frontier;
- generation 2 sparse success as an experiment, with no commit authority and
  three rejected descent motions;
- generation 3 replay of that 62-action route as a candidate, rejected by the
  FSA as not commit-safe;
- generation 4 proposal of a clean 59-action candidate.

The FSA then:

1. reproduced the candidate in a fresh authoritative preflight;
2. verified all trajectory, collision, load, provenance, budget, RGB-hash, and
   telemetry-hash invariants;
3. issued a single-use permit;
4. committed 59 actions with a stepwise clone interlock;
5. verified the promoted retained source for 59 actions;
6. independently replayed the exact 59-action acquisition boundary.

The result flags are all true: `promoted`, `genuine_failed_attempt`,
`revised_after_failure`, `source_verified`, and `path_replayed`.
`fsa_rejections=1` records the unsafe generation-3 candidate denial in addition
to the genuine controller collision rejections observed during experiments.

The promotion receipt is
`2a20a533246dff3ca8fc42145792001a05568c642c1d376c5dca3fa039129d9b`.
The retained proposer source tree is
`3c8c4f3f4d4527735f260bea39768a955e7bbcea5d8a71429bcb1f2b8e584188`.

The prior indexed-frame and incomplete-collision campaign contents were deleted
before this run. They are not retained, cited, or loadable as fallbacks.

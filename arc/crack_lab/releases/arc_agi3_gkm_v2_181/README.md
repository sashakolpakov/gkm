# GKM ARC-AGI-3 v2 release

This directory is the frozen, replay-certified ARC-AGI-3 release of the
Gödel–Kolmogorov Machine (GKM). It contains the original frozen acquisition
source for every public game, a normalized schema-v2 artifact tree with exact
per-level evidence for the 181 levels reached by this snapshot, and the
machine-readable release receipt used by the scorecard replayer.

The important unit of generality is the **producer**, not the specialized
program it learns for one game. GKM applies the same game-independent proposal,
retention, audit, and replay protocol to all 25 public ARC-AGI-3 games. Its
learned output is executable source: reusable interaction skills, perception
routines, searches, planners, bindings, and—when that is the compact sufficient
solution—finite action programs.

## System boundary

GKM separates components that propose behavior from components that are
allowed to certify it:

| Component | Role | Admission authority |
|---|---|---|
| Fixed research harness | Provides the Arena interface, empty solver scaffold, proposal workspace, complexity accounting, evidence schema, and campaign controls | None |
| Native coding-model proposer | Interacts with the game and writes or extends the game-playing `legs.py`, `players.py`, and related probes | None |
| Campaign supervisor | Schedules clean continuations and restarts, changes effort after repeated failure, and may request an independently investigated obligation | None |
| Independent side expert | Investigates one difficult frontier in a quarantined copy and returns a hypothesis or candidate | None |
| Trusted host verifier | Scans evidence, executes candidate source from reset, replays the resulting path from a second reset, seals hashes, and promotes only a passing boundary | Sole promotion authority |
| Competition replayer | Sends an already admitted checkpoint path to the official API | None; scoring only |
| Human operator | Chooses the research question and resource envelope, monitors infrastructure, and decides what to publish | None over candidate promotion |

The game-playing files under `acquisition_source/` are learned artifacts
written within recorded model-proposal lineages. They are not a library of
human-supplied ARC answer programs. The fixed host can reject or retain a
candidate but does not silently repair a game solver during promotion. A
supervisor or side expert can supply untrusted information, but that information
has no standing until it appears in an admitted source boundary and passes the
same fresh host checks.

This distinction applies architecture-wide. A generic graph search, a learned
detector, a parameterized movement routine, a game-specific coordinate
binding, and a literal action sequence are all possible model-proposed cells.
Their differing specificity describes the executable knowledge GKM acquired;
it does not change who produced them.

## Why per-game programs are a general-purpose result

The producer begins each unseen game behind the same interface:

```text
reset() -> frame
step(action) -> frame
frame() -> 64x64 grid
levels_completed
actions
terminal()
clone()
```

No interface field names a game's objects, mechanics, goal, or solution. The
same proposer contract, blank scaffold, retention rule, complexity coordinate,
taint policy, action policy, replay gate, and no-forgetting check are applied
to each game.

Producing specialized programs is the intended output of this general method,
just as a compiler emits a program for a particular source or a program
synthesizer emits a solution for a particular specification. GKM's generated
game directories are therefore learned executable state, not exceptions to its
generality. The breadth of those artifacts—25 games and 181 public levels in
this frozen snapshot—is evidence that one producer architecture repeatedly
acquires different mechanics.

The local acquisition interface exposes resettable `clone()` lookahead. This
is a fixed capability of the public GKM Arena, available uniformly rather than
added for a particular game. It is used only during discovery. Official
Competition Mode receives no clone calls and no model inference; it replays the
already certified action path through the official API.

## Acquisition and promotion

For the next unresolved level, the campaign:

1. seeds an isolated workspace from the highest admitted solver;
2. asks the coding-model proposer to reuse incumbent legs and create any
   missing executable cells or bindings;
3. treats all resulting source, commands, and transcripts as untrusted;
4. rejects private game/runtime introspection, invalid action encodings,
   missing evidence, stale hashes, and malformed structured records;
5. executes the candidate from reset and records the exact first-passage path;
6. replays that path independently from a fresh reset;
7. seals source, path, transcript, provenance, audit results, and hashes at the
   winning boundary; and
8. promotes only if the entire boundary is internally consistent.

A deeper winning path also certifies every shallower first-passage prefix. This
is the executable no-forgetting condition: a new program may find a shorter or
more compositional route, but it must still reproduce all previously retained
obligations.

Unsuccessful or interrupted work remains outside the release. A model's claim
that it solved a level is never a certificate.

### Retrospective normalization

The live acquisition archive spans more than one historical evidence schema.
The release certifier therefore builds a separate, uniform schema-v2 tree; it
never edits the frozen acquisition source.

For each level it first tries retained model-written source from the relevant
winning phase. If that source still executes exactly from reset, the schema-v2
boundary preserves it. If a historical source snapshot is missing, no longer
executes independently, or exceeds the release-certification time bound, the
certifier generates a minimal deterministic capsule from the already
host-validated first-passage path and executes that capsule from reset. The
provenance record labels such a boundary
`deterministic_exact_path_reconstruction`; it is not relabeled as a historical
model-written source boundary and is not used to invent a marginal-complexity
observation.

This separation gives the submission both things it needs:

- `acquisition_source/` preserves the actual model-generated solver programs;
  and
- `artifacts/` provides one strict, machine-checkable evidence schema for
  endpoint replay.

## Kolmogorov/free-energy program growth

Each admitted boundary records retained program description and behavioral
gain. GKM can therefore distinguish:

- novelty, where a new mechanic requires additional executable structure;
- reuse, where a later level invokes an unchanged earlier leg with little new
  binding code; and
- compression, where a replacement preserves replay behavior with a shorter
  retained description.

This is the operational Schmidhuber/PowerPlay connection: proposals grow a
monotone archive of verified capabilities, while conditional source complexity
exposes the expected acquisition-and-reuse sawtooth. A numerical drop alone is
not called reuse; a strict witness also requires adjacent exact boundaries, an
unchanged earlier leg, a new caller that invokes it, and fresh replay of the
composition.

## Release contents

`acquisition_source/<game>_legs/` contains the four frozen runtime files from
the acquisition campaign: `legs.py`, `players.py`, `solve.py`, and
`checkpoint.json`. No WIP trees, failed attempts, or superseded archives are
included.

Every `artifacts/<game>_legs/` directory contains the normalized admitted
runtime:

- `legs.py`: generated reusable executable cells;
- `players.py`: generated level bindings;
- `solve.py`: dispatch entry point;
- `checkpoint.json`: the admitted cumulative path and depth; and
- `promotion_evidence/level_XX/`: exact files, transcript/provenance record,
  replay checks, action and taint checks, hashes, and manifest for that
  first-passage boundary.

The release root also contains:

- the authoritative 25-game inventory;
- a release identity binding the artifact to a public source revision;
- a content-addressed release receipt;
- a complete audit summary; and
- the deterministic scorecard entry point.

The retained checkpoint paths contain 7,001 game actions and reach 181 of the
183 public levels. These are local artifact facts, not a self-reported
ARC-AGI-3 score. The Community Leaderboard entry links the single official
25-game Competition-Mode scorecard generated from this receipt.

| Game | Verified depth | Stored path actions |
|---|---:|---:|
| `ar25` | 8/8 | 269 |
| `bp35` | 9/9 | 393 |
| `cd82` | 6/6 | 91 |
| `cn04` | 6/6 | 210 |
| `dc22` | 6/6 | 540 |
| `ft09` | 6/6 | 80 |
| `g50t` | 7/7 | 361 |
| `ka59` | 7/7 | 342 |
| `lf52` | 8/10 | 544 |
| `lp85` | 8/8 | 93 |
| `ls20` | 7/7 | 365 |
| `m0r0` | 6/6 | 230 |
| `r11l` | 6/6 | 115 |
| `re86` | 8/8 | 600 |
| `s5i5` | 8/8 | 329 |
| `sb26` | 8/8 | 124 |
| `sc25` | 6/6 | 144 |
| `sk48` | 8/8 | 506 |
| `sp80` | 6/6 | 151 |
| `su15` | 9/9 | 170 |
| `tn36` | 7/7 | 131 |
| `tr87` | 6/6 | 208 |
| `tu93` | 9/9 | 195 |
| `vc33` | 7/7 | 213 |
| `wa30` | 9/9 | 597 |
| **Total** | **181/183** | **7,001** |

## Reproduction layers

The release supports three separate claims:

1. **Integrity:** recompute manifests and hashes, then verify the release
   receipt and its exact source revision.
2. **Behavior:** run every admitted source boundary and literal checkpoint path
   locally from fresh reset.
3. **Endpoint:** perform a full 25-game ONLINE shakedown, then one
   receipt-bound Competition-Mode replay with zero proposer tokens.

The producer itself is also public and can be run from blank per-game
workspaces. Proposal sampling is stochastic, so method replication does not
mean that an independent run must emit byte-identical source. The deterministic
claim is narrower and stronger where it matters: no candidate enters the
archive or scorecard unless its exact retained bytes and action path pass the
fixed admission protocol.

The definitive commands and the public source/receipt identifiers are recorded
alongside the completed release rather than inferred from a mutable working
directory.

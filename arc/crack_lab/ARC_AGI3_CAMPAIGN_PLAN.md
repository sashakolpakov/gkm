# ARC-AGI-3 GKM campaign master plan

This is the canonical planning document for the ARC-AGI-3 Gödel–Kolmogorov
Machine (GKM) campaign. It incorporates the campaign goal, the unlimited
budget policy, the clean-room and replay rules, the 183-level completion target,
the uniform-artifact repair, the final scorecard and leaderboard release, the
manuscript comparison/reproduction work, and the later automated contiguous
campaign.

Earlier separate execution and budget policy documents were merged into this
file and removed. Historical snapshots and experiments remain available through
Git history; there is only one live prose policy.

`ARC_AGI3_CAMPAIGN_QUEUE.json` is the sole generated machine queue. It is a
replaceable runtime snapshot, not a second policy document; the runner validates
every item against this plan, the live authoritative inventory, the current
checkpoint, and the eligible same-frontier WIP immediately before dispatch.
Legacy `NEXT_RESET_CAMPAIGN.json`, `RUN_UNLIMITED_CAMPAIGN.json`, and similarly
named queues are obsolete and must not be regenerated.

## 1. Objective and release order

The primary objective is:

> Solve and replay-promote all **183/183** levels in the authoritative ARC API
> inventory, with a clean and auditable acquisition lineage, then publish the
> result through a full online shakedown, one definitive Competition-Mode
> scorecard and the ARC-AGI Community Leaderboard submission, with the
> manuscript and downstream documentation maintained as a separately verified
> empirical/reproduction surface.

The previously paused goal—continue in bounded waves and report near 90%—is now
a milestone inside this larger objective, not a stopping condition.

**Execution directive reaffirmed 2026-07-27:** report the 156/183 and 165/183
milestones, but continue without pausing until canonical coverage is 183/183.
Then execute the replay/audit freeze, ONLINE shakedown, definitive Competition
scorecard, and benchmark PR update in the order below.

**Major benchmark-update directive, 2026-07-30:** freeze the strongest current
canonical set for a major release while the exact `lf52` remainder continues in
separate workers. No further public mutation of PR #37 is permitted until that
freeze has uniform evidence, passes the full 25-game ONLINE shakedown, and
produces one definitive Competition-Mode scorecard for the frozen set. The next
PR edit is one coherent release: title, persistent body, YAML, README,
percentages, links, and scorecard change together. Do not post piecemeal
explanatory comments or publish the obsolete July score while this update is
being assembled. A later 183/183 completion triggers a separately audited
refresh. Preparatory contiguous-runner hardening may proceed in parallel;
production pilots and full launch remain last.

**Reordered execution directive, 2026-07-31:** the author explicitly released
and moved the conservative manuscript/downstream-documentation phase ahead of
the final 183/183 remainder. That phase is complete: the 25-game marginal table,
empirical figures, three Socratic passes, Sphinx documentation, PDFs, tests, and
integrity manifest were regenerated and verified without changing the
theoretical core. The separately frozen v2.0 181/183 benchmark payload was also
published and remotely reopened. Its exact remote branch passes the leaderboard
repository's validator locally, including live URL checks. The corresponding
fork workflow is still maintainer-gated as `action_required` with no executed
jobs, so a remote green check is not yet proven. Neither checkpoint substitutes
for the final 183/183 release.

The frozen v2.0 evidence remains byte-exact at commit `9235ed26` and under
`arc/crack_lab/releases/arc_agi3_gkm_v2_181/`. On 2026-07-31 its release branch
was merged into `master` at `e8927fc6`, so the two origin histories are now
reconciled while the receipt-bound release tip remains immutable. Its partial
receipt verifies 181 claimed boundaries and names only `lf52` L9--L10 as
unclaimed.

The remaining required order is:

1. Preserve the verified manuscript/downstream bundle and frozen v2.0 181/183
   release as immutable checkpoints.
2. Continue the exact `lf52` remainder to 183/183 under the retry-complexity
   policy.
3. Freeze the complete canonical set, make every winning boundary and promotion
   artifact uniform, and pass the full release audit.
4. Run the complete 25-game ONLINE shakedown and one definitive Competition-Mode
   scorecard for the 183/183 frozen set.
5. Append the complete release as a new leaderboard version, update PR #37's
   title/body/README/YAML/links atomically, and reopen every remote surface.
6. Launch the fully automated supervisor for a separate clean contiguous
   campaign using the final successful policy. This contiguous lineage never
   weakens or silently replaces the canonical result.

**Goal-level scheduler invariant:** one journal-derived quantity—the number
`n` of settled clean no-progress retries at the unchanged exact frontier—must
choose both escalation axes. The primary transition
`medium -> high -> xhigh -> max` and the auxiliary transition
`max -> max + sidecar(s)` are deterministic projections of that same `n`;
neither may be selected independently, manually, or by game identity.
Promotion resets `n=0`, while taint, infrastructure, rate-limit, blocker,
capacity, and containment outcomes do not increment it.

## 2. Current state and sources of truth

Snapshot at plan creation, 2026-07-27:

- Canonical replay-promoted coverage: **137/183** levels across all 25 games
  (74.86% raw coverage).
- Latest promotion: `bp35` L5, clean at xhigh, `marginal_C=44`.
- Last completed uniform-boundary audit: 133 exact historical winning-source
  checkpoints; only the old `wa30` L1–L3 acquisition boundaries remained
  historically unavailable at that instant.
- A fresh, separately rooted `wa30` reacquisition is running from an empty
  solver lineage and has already produced exact clean boundaries from L1
  onward. It may replace canonical `wa30` only after a full audited 9/9 result.
- The authoritative denominator is 183. The earlier 180-level estimate and its
  153/162 milestones are superseded.

Milestone register:

| Threshold | Status | Required action |
|---|---|---|
| 156/183 (at least 85%) | Achieved; campaign continued | Report only; continue cracking |
| 165/183 (at least 90%) | Achieved; exact-remainder phase active | Report; end broad rotation and attack the exact remainder |
| 183/183 (100%) | Pending | Freeze solving artifacts and begin the release pipeline |

The live state is not this paragraph. Query it from:

```sh
python3 arc/crack_lab/codex_campaign_status.py
python3 arc/crack_lab/codex_campaign_policy.py
python3 arc/audit_gkm_solved_checkpoints.py \
  arc/crack_lab/agent_solutions \
  --csv /tmp/gkm-current.csv \
  --json /tmp/gkm-current.json
MPLCONFIGDIR=/tmp/gkm-boundary-audit \
  python3 arc/audit_action_boundaries.py \
  arc/crack_lab/agent_solutions \
  --json /tmp/gkm-action-boundaries.json
```

Canonical solver state lives only in
`arc/crack_lab/agent_solutions/<game>_legs/`. Candidate reacquisitions live
under separately locked
`arc/crack_lab/runs/reacquisition/<lineage>/artifacts/` roots until an explicit
audited swap. `candidate_solutions/` retains only compact installation
receipts after a successful swap; superseded parallel trees are deleted and
are never live promotion targets. Scratch workspaces and reacquisition roots
are live process state, not an archive: an inactive scratch tree is deleted
after its terminal evidence has either been admitted into an exact bundle or
discarded, and an installed reacquisition root is deleted after the audited
swap. The final campaign tree may therefore contain only currently running
workspaces, never a museum of old attempts. Cleanup is terminal-process-aware:
it must preserve the shared protected-transcript root and every open transcript
pathname until its owning proposer has exited and the host has sealed the
bytes. A missing, unlinked, or unreadable protected transcript makes that turn
quarantine-only; the harness may not infer cleanliness from an older copied
`proposer_last.log`.

The formal contiguous runner treats journal-bound generation trees as live
recovery state until the exact 183/183 terminal boundary; it never deletes a
same-frontier WIP context mid-campaign merely to make the archive look clean.
At completion it executes one journal-head-bound retention transaction:
first bind and seal the independently reopened external promotion/replay
inventory, then copy and seal the exact compact taint, hash, usage, and terminal
receipts, and only then delete every generation tree. The transaction is bound
to the exact pre-cleanup scheduler PASS and is idempotently recoverable after a
crash: a restarted operator rechecks that PASS against the unchanged journal
head, journal prefix, and control bytes without trying to resurrect deleted
transient WIP. The unified terminal audit fails if any expected compact receipt
is absent or if any generation, scratch, workspace, cache, raw transcript,
stdout, or stderr survives.

## 3. Non-negotiable evidence boundary

Every proposing turn may use only:

- the public Arena interaction surface;
- the current game’s clean promoted solver;
- clean WIP from the same admissible lineage;
- the generated solver index and frontier brief;
- its own prior observations and proposer sequence.

It may not use:

- game implementation source or environment files;
- private runtime attributes or hidden simulator state;
- another system’s solution archive;
- canonical or candidate artifacts from a different lineage;
- post-hoc game-code labels;
- web search or external network access;
- API keys or expanded sandbox permissions.

Prior observations and prior proposer turns are admissible experience. They are
not taint unless they contain privileged code, environment, game-description,
or runtime information that materially helps solve the level.

A tainted attempt retains only typed negative operational metadata after its
required terminal audit; raw turn bytes are discarded. It contributes no
solver knowledge, promotion, comparison statistic, or WIP to a later clean
lineage.

The campaign plan, interactive supervisor notes, post-hoc labels, and
campaign-local quarantine metadata/evidence are host-side state; none lives
under the checkout. Merely recording a native proposer's tactic in this
document does not make that tactic a prompt. A proposer may receive
game-specific guidance only through an explicit lineage input: same-lineage
proposer WIP, an admitted side-expert result, or the authenticated supervisory
handoff defined in section 4.3.2. Every such input is named in the attempt
receipt, hash-bound to its exact parent/frontier and source role, taint-scanned,
and independently reproduced through the public Arena surface. Unrecorded
interactive guidance is inadmissible.

## 4. Cracking campaign

### 4.1 Sequential and parallel structure

Levels within one game are sequential: L\(k+1\) cannot start until L\(k\) has
passed replay and promotion. Different games may run concurrently. A per-game
workspace lock prevents two writers from mutating the same lineage.

This subsection records the exploratory completion campaign. Its supervisor
may run several disjoint headless jobs at once and, on an exceptionally hard
frontier, at most two explicitly separated primary candidate roots. Durable
usage-ledger appends are serialized even when provider-confirmed unlimited
turns overlap. The later contiguous campaign is deliberately stricter: section
10 permits one writable primary lineage per game plus only quarantine-only
retry-count-eligible sidecars.

Concurrency is capacity, not a utilization target. Distinct unsolved games are
parallelized first. After a frontier has exhausted the ordinary ladder and two
clean 180-minute max turns, the scheduler enters `LONG_COHERENCE`: it retains
at most two independent lanes for that frontier and gives them longer
uninterrupted allocations instead of filling spare capacity with additional
short duplicate turns. One lane restores the strongest eligible clean
same-frontier WIP; the other is the declared coherence-reset lineage. When only
sequential descendants remain, unused capacity is expected.

Persistent clean failure eventually changes the *kind* of parallelism rather
than multiplying writable lineages. Once the escalating primary ladder has
reached `max` and its first max turn has also ended at the same exact boundary,
the scheduler may allocate otherwise idle capacity to an independent side
expert. Allocation is not semantic authorship: the game-specific brief must be
an exact-frontier, authenticated `NATIVE_SIDECAR_REQUEST` emitted by the native
proposer or a sidecar request inside an admitted `SUPERVISORY_HANDOFF`. With no
such request, the slot stays idle. This role works from an immutable private
copy, pursues the bound request, and experiments only through the public Arena
interface. It cannot mutate the live proposer workspace. Eligibility and
capacity depend on frontier complexity evidence, never on a game ID,
remembered solution, operator hint, comparator result, or post-hoc label; the
scheduler and host operator may not invent or rewrite the tactical brief.

### 4.2 Uniform budget semantics

Budget interpretation is identical for every proposer:

- `limit=None` or provider `unlimited=true`: disable reserve, token, dollar,
  turn, and cumulative wall-spend controls for that proposer.
- finite limit: enable the proposer’s corresponding cost controls.
- unknown or unreadable limit: do not treat it as unlimited; fail closed unless
  explicit local caps safely govern admission.

Unlimited does **not** disable correctness controls. Every turn retains a soft
scheduling allocation, an independent liveness/containment watchdog,
transcript, taint scan, replay check, provenance record, and workspace lock.
The scheduling allocation is not a process-kill deadline.

The provider window identity is durable state, not an inference from a displayed
percentage. Every proposer preflight/postflight records the explicit window name
and limit ID. An explicit cached `unlimited` snapshot remains authoritative
across legacy postflights that report only `100% remaining`; only a newer
explicit finite-window observation may re-enable cost controls.

### 4.3 Explicit lineage-input modes

The automatic scheduler records two independent fields for every attempt:

| Field | Values | Meaning |
|---|---|---|
| `seed_mode` | `zero_seed`, `verified_parent` | Start without solver source, or seed the latest replay-validated parent |
| `wip_mode` | `exclude`, `restore_clean_same_frontier` | Exclude prior failed-attempt context, or restore taint-clean WIP for this exact game and level |

Their composite `lineage_input_mode` is recorded in the planned item, generated
command, and outcome. Restarting a process, changing effort, and choosing a
lineage-input mode are separate decisions. `--fresh` is a legacy zero-seed/no-WIP
alias and must not be generated by the supervisor.

The scheduler applies these rules:

- The first attempt in a new candidate root is `zero_seed+exclude`.
- A newly promoted next-level frontier is `verified_parent+exclude`.
- Ordinary medium/high/xhigh retries below 90 minutes restore clean WIP after a
  clean failure or interruption. This prevents short turns from repeatedly
  paying the same discovery cost.
- The first 90-minute max turn after the short ladder deliberately uses
  `exclude`, retaining only the verified parent. This is a coherence reset: the
  live proposer can build one uninterrupted model without inheriting stale
  hypotheses.
- The following 120-minute/max turn restores the complete clean snapshot
  produced by that long no-WIP turn. Two 180-minute turns alternate `exclude`
  and `restore_clean_same_frontier`. If both fail cleanly, the frontier enters
  `LONG_COHERENCE`: no more than two independent lanes receive 300-minute soft
  allocations, again pairing one coherence reset with one cumulative WIP
  continuation. This preserves both stochastic diversity and live
  reasoning-state continuity without multiplying short restarts.
- A tainted attempt is discarded and can never make WIP restore eligible.
- A clean provider, containment, or other infrastructure interruption is not a
  solver retry. The scheduler keeps the same `n`, effort, and soft allocation,
  pins the host-sealed exact-frontier WIP attempt by ID, and mechanically
  restores that capsule even when the ordinary row at `n` calls for `exclude`.
  The status reducer must reopen the pointer, metadata, complete file inventory,
  and exact-parent binding; the proposer runner must apply the current taint
  policy again immediately before restore. If the pointer changed, the capsule
  is stale, or any current check fails, dispatch stops rather than falling back
  to a different attempt or silently converting the event into no-progress.
- WIP eligibility is re-evaluated with the current taint policy on every
  restore; a snapshot accepted by an older scanner is never grandfathered, and
  ordinary unpinned recovery falls back only to the newest currently clean
  exact-frontier snapshot. A scheduler-pinned infrastructure recovery never
  substitutes a different capsule.
- A deliberate independent reacquisition uses `exclude` explicitly and a new
  candidate root; it never silently ignores a newer verified artifact.

“Saving proposer state” means keeping one turn uninterrupted. A model's private
live reasoning state cannot be resumed after process exit; only its externalized
WIP can. Therefore duration and WIP selection are orthogonal scheduler choices.
Cross-scratch WIP continuation restores both the probe/context files and the
newest taint-clean same-parent solver source. The host reseeds the immutable
validated checkpoint first, overlays only source from a snapshot whose embedded
checkpoint exactly matches that parent, and replays the overlaid source before
any new proposer call. A probes-only restore that silently reverts
`legs.py`/`players.py` to the promoted parent is a scheduler failure.
The automatic supervisor may override the long-turn alternation only from a
machine-readable, replay-backed progress witness—not from another campaign's
solution or an operator's post-hoc game interpretation.

#### 4.3.1 Independent side-expert escalation

The formal name of this stubborn-frontier tactic is
`OBSERVATION_ONLY_AUXILIARY_ANALYSIS`. It is an acquisition-time expert role,
not a release audit and not post-hoc interpretation. It preserves the tactic
introduced on the final exploratory frontiers:

1. The ordinary and hard-frontier ladders must be exhausted.
2. Once the first 60-minute `max` attempt at `no_progress=4` also settles as
   clean no-progress, the unchanged frontier becomes side-expert eligible at
   `no_progress=5`. Taint, infrastructure, rate-limit, and containment outcomes
   do not satisfy this trigger.
3. The scheduler may then add one independent side expert without interrupting
   the live `max` proposer only when it can bind the assignment to either an
   authenticated same-frontier `NATIVE_SIDECAR_REQUEST` or a sidecar request
   inside an admitted same-frontier `SUPERVISORY_HANDOFF`. The scheduler chooses
   whether capacity is available and deterministically selects among compatible
   requests; it cannot author, paraphrase, or extend their game-specific
   content. With no valid request, no side expert is launched.
4. The expert receives an immutable private copy containing only the verified
   parent, admitted clean same-frontier WIP, public-Arena observations, and the
   sealed request. Its model and reasoning effort are pinned in the contiguous
   campaign launch manifest; the role is not a synonym for an `ultra` provider
   setting. The request must identify one bounded unresolved obligation in the
   game-independent classes mechanism induction, state representation, exact
   planning, or prefix compression and must include evidence references and
   falsifiers. The side expert pursues that request rather than silently
   substituting another tactic.
5. If the subsequent 90-minute reset and 120-minute WIP continuation both fail
   cleanly, `no_progress>=7` may use otherwise idle capacity for at most two
   independent experts bound to distinct authenticated requests and unresolved
   obligations. It may not run duplicate assignments for one profile or create
   another writable artifact lineage.
6. When the selected obligations are exhausted and another complete
   reset/continuation pair fails, a fresh receipt-bound diagnosis may begin a
   new round. The scheduler never routes by game name or historical solution.

For observation-derived planning, equality of public frame hashes is only a
deduplication heuristic, never proof of hidden-state equality. A side expert
may refine a public key with a bounded one-step response signature over a
fixed, receipt-declared public action basis. If identical public keys split
under that signature, every distinct representative is preserved. Reports must
say “bounded behavioral equivalence,” state the tested basis and horizon, and
must not claim a complete hidden-state quotient.

Likewise, a public action that appears to undo an ordinary probe is never
assumed to rewind a reward, terminal state, or level transition. Reversible
search may use it only after a same-state exact-frame restoration test in the
current context, and it must check reward/terminal/level counters before
attempting rollback. A rewarded transition is an absorbing acquisition
boundary: stop that branch, seal the exact pre-debrief winning source and
action stream, and require a fresh independent replay from the exact parent.
No undo-derived continuation, reconstruction, or promotion has authority.

The expert's diagnosis is untrusted search-routing advice, not evidence that a
mechanic is true. All expert files remain quarantined. The expert may neither
edit canonical/WIP state nor promote. A proposed path, patch, abstraction, or
hint can enter the acquisition lineage only after the host binds it to the
exact parent/frontier, runs the standard taint and provenance scans, and
reproduces every relied-on observation or winning candidate through a fresh
public-interface replay. Failed or inadmissible expert output changes neither
`reached` nor `no_progress`.

This creates a clean `max -> independent side expert` transition. The important
escalation is the independent context, assignment, and evidence boundary—not a
provider effort label. Long `max` turns preserve coherent proposer state, while
private side experts test orthogonal hypotheses. The contiguous version adds a
recorded self-challenge before handoff; that is an explicit improvement, not a
retrospective claim about the exploratory sidecars. Every diagnosis,
assignment, terminal result, admission, and rejection is journaled and
usage-accounted under the same finite/unlimited budget semantics as proposer
turns. The journal also binds the assignment to its native-request or
supervisory-handoff origin; a manual/root-authored, scheduler-authored, stale,
cross-frontier, or substituted brief is rejected before launch.

The exploratory implementation that motivated this rule must be described
literally. On 2026-07-29, already-running session subagents originally assigned
to conformance audits were manually reassigned to `bp35` L7 and `lf52` prefix
compression. They were not headless proposer turns and exposed no model/effort
field, so they are not labeled `ultra`. They used private roots under
`/private/tmp`, copied exact-parent/clean-WIP inputs, shared the host filesystem
and process namespace, interacted only through public `gkm_arena`, repeated the
workspace taint check at Arena launch, wrote only inside their private copies,
and handed findings back through session messages. Neither current solving
brief required a formal Socratic pass. The contiguous analogue keeps the
independent/private-copy/public-observation tactic but strengthens it with a
minimal container input bundle, manifest-pinned model/effort, journaled
assignment/usage, process isolation, and a recorded self-challenge.

#### 4.3.2 Explicit supervisory-proposer escalation

The contiguous campaign may use an LLM supervisory proposer analogous to the
interactive supervisor used during the exploratory campaign. This is a
first-class acquisition role, not an unrecorded human override and not the
deterministic scheduler itself. Its purpose is to synthesize a difficult
frontier's authenticated native-proposer and side-expert evidence, challenge
the current hypotheses, and produce a concise game-specific tactical handoff
for a subsequent native proposer.

The deterministic scheduler alone chooses the game, effort, allocation,
WIP/reset mode, auxiliary capacity, and promotion transition from the
exact-frontier retry coordinate. The supervisory proposer may reason about the
already selected frontier and recommend what to test there; it may not alter
those scheduler decisions, mutate live WIP/canonical state, run a promotion,
or grant evidence authority.

Supervisory escalation is complexity-triggered:

- below the first side-expert threshold it is disabled;
- at `n>=5`, it may run after at least one authenticated max/native or
  side-expert result exists for the unchanged frontier;
- a later supervisory round requires new authenticated evidence or a complete
  scheduler-defined reset/continuation round since the previous handoff, so
  repeated summaries cannot consume idle capacity without information gain;
- the model, effort, context limit, and maximum concurrent supervisory roles
  are pinned in the launch manifest and are uniform across games.

Its input is a host-built, immutable, exact-allowlist bundle containing only
the verified parent identity and budget, the sealed exact-parent solver-source
snapshot, any selected admitted same-frontier WIP solver-source snapshot and
summary, public-observation transcripts, side-expert reports and their
admission/rejection receipts, and the generic solver/evidence contract. These
are authenticated outputs of the same lineage, not privileged game
information. The bundle contains no game/environment implementation, other
game or lineage, broader canonical solution archive, raw WIP/controller state,
comparator/manuscript material, post-hoc label, credential, or informal
session note. The supervisor runs in its own isolated role container with no
live proposer mount and no direct Arena mutation channel.

Its output is a schema-validated `SUPERVISORY_HANDOFF`: the unresolved
obligation, observations relied upon with exact evidence references, hypotheses
and falsifiers, one or more bounded next tests, rejected alternatives,
confidence/caveats, a Socratic self-challenge, and optionally one or more typed
sidecar requests. Such a request may seed a side expert only at the same exact
frontier under section 4.3.1; it does not allocate capacity. The host rejects
free-form or unreferenced factual claims, binds the accepted bytes to the exact
frontier/input manifest/model/usage receipt, and exposes them to a later native
proposer as an explicitly labeled unverified hypothesis. The native proposer
must reproduce every relied-on observation through its admitted public
interface. A handoff never counts as a win, retry, complexity witness, WIP
patch, or promotion; only the ordinary fresh replay, taint, source, hash, and
manifest gates can admit a resulting solver.

The proposer-visible projection contains the typed handoff and cryptographic
commitments only. Host receipt paths, controller state, assignment threads,
raw auxiliary output, and the complete admission binding remain host-only; the
mounted prompt bundle is rejected if it contains an absolute host path or any
undeclared binding field.

Exposure is treated conservatively. If an attempt bundle contains a
supervisory handoff, no WIP or candidate from that turn is admissible until the
complete native reproduction receipt passes, even if proposer prose claims the
handoff was ignored. The evidence labels the boundary handoff-exposed. A
passing reproduction binding may travel only with hash-linked WIP at the same
exact parent/frontier/handoff and is invalidated by any change to those
identities.

The reproduction receipt is host-derived, never a proposer-authored `PASS`
claim. The verifier reopens the cited source observation receipts, checks their
public action-basis and response-signature commitments against the native
attempt's authenticated public transcript or a fresh host replay, and writes
the source-to-native mapping only after equality is established.

| Role | Authenticated input | Output authority |
|---|---|---|
| Deterministic scheduler | Journal, exact frontier, receipts, capacity, budget | Chooses target/allocation/role transitions; no game-semantic advice |
| Native proposer | Exact parent, admitted WIP, public observations, optionally one admitted handoff | May create candidate/WIP; never self-promotes |
| Independent side expert | Immutable exact-parent bundle, public observations, and an authenticated native/supervisory sidecar request | Quarantine-only evidence |
| Supervisory proposer | Immutable allowlist of admitted native/side-expert evidence | Quarantine-only tactical hypothesis |
| Host verifier | Candidate plus complete lineage/evidence receipts | Sole replay/taint/hash/manifest admission and promotion authority |

The unified conformance suite must exercise the complete positive path and
inverse cases: hidden or cross-lineage input, omitted evidence reference,
forged/stale frontier, unpinned model, duplicate supervisory round, prompt
projection containing a host path or host-only field, direct workspace
mutation, scheduler-field override, metadata-only reissue of identical native
observations, WIP source, or side-expert findings, missing self-challenge,
manual/root/scheduler-authored sidecar brief, stale or cross-frontier request,
request substitution, unreproduced factual claim, and handoff-to-promotion
bypass. The manuscript and
downstream method/reproduction documentation must distinguish the
deterministic scheduler, native proposer, independent side expert, supervisory
proposer, and host verifier; report when a winning acquisition attempt was
exposed to a supervisory handoff; and reproduce the handoff manifests and
admission chain.

The first exploratory `bp35` sidecar closed when the independent canonical
lane promoted L7; it was neither admitted nor used. It exposed the generic
risk that equal `(frame_sha256, moves_used)` keys can have different bounded
public responses, so the conformance suite retains that inverse property. The
superseded exploratory bundle itself has been discarded and is not a
publication artifact or a source of quantitative claims.

Before dispatch, the runner verifies that `seed_mode` agrees with the current
checkpoint, that WIP restore has an eligible clean same-frontier snapshot, and
that both explicit CLI fields match the plan item. A mismatch is a hard
scheduler error, not a fallback.

One artifact-lineage lock is held per game for the full orchestration. Different
tags cannot race to write the same canonical artifact; a deliberately separate
candidate root has its own independent lock. The zero-cost existing-leg
precheck is also bounded to six candidates with a ten-second verifier cap per
candidate, so it cannot silently consume the sustained proposer interval.
Post-turn orphan/path recovery is bounded independently: proposer-log and
checkpoint exports are considered first, current-frontier file exports are
relevance-ordered and deduplicated, at most 24 distinct paths are admitted, at
most six can act as prefix candidates, and at most 72 pairwise glue attempts
are replayed. Exhausting that bounded recovery search preserves clean WIP and
records no progress; it may not hold a lane indefinitely after its proposer
has already exited.

The live Codex JSONL stream is written outside the proposer-writable workspace
and copied into the workspace only after the process and its child tools exit.
A proposer therefore cannot edit, truncate, or relocate its audit record.
Inside proposer tool subprocesses, `KeyboardInterrupt` from cancellation is
converted at the public Arena boundary into a one-line sanitized error; it may
not unwind through private engine frames into command output. Operator
interrupts outside that explicitly marked proposer environment still propagate.
Any interrupted workspace that fails the taint scan is ineligible for resumable
WIP. Its raw transcript, scratch source, copied canonical lineage, and mutable
snapshot are discarded; quarantine retains only a sanitized exact incident
receipt. Clean bounded-search evidence is retained only when it has a compact,
relocatable reproducer. The WIP persistence function independently repeats the
workspace taint scan immediately before copying files or replacing
`latest.json`; this fail-closed boundary prevents cancellation or transcript
finalization races from making a tainted snapshot restorable.

### 4.4 Effort and wall-time ladder

The ordinary level-local ladder is the following single versioned table.
`n` is the number of prior clean no-progress outcomes at the exact current
frontier; infrastructure, taint, and blocker outcomes do not increment it.
This `n` is the scheduler's single operational complexity coordinate. It is not
a claim to measure Kolmogorov complexity itself. It measures how much
increasingly capable, replay-gated search has failed at one unchanged boundary.
Both the primary effort ladder and the independent-sidecar ladder are
deterministic functions of this same coordinate; neither may use a game name or
an operator's difficulty judgment.

The counter is reconstructed from settled journal events, not copied from a
status report. In particular, legacy aggregates such as
`failed_attempts_at_frontier`, paid-turn count, transcript count, timeout count,
or branch count are not `n`: they may mix superseded, interrupted,
infrastructure, tainted, or parallel work. Each increment must name the exact
frontier and parent hashes, the prior coordinate, the policy-selected effort,
and the clean result receipt. A scheduler decision whose claimed `n` cannot be
replayed from that chain is rejected.

| `n` | State at the current level | Model/effort | Soft allocation | WIP mode |
|---:|---|---|---:|---|
| 0 | Fresh or newly promoted frontier | `gpt-5.6-sol` medium | 15 min | exclude |
| 1 | Clean medium failure | `gpt-5.6-sol` high | 20 min | restore |
| 2 | Clean high failure | `gpt-5.6-sol` xhigh | 25 min | restore |
| 3 | Warm hard frontier after the ordinary ladder | xhigh | 40 min | restore |
| 4 | Persistent hard frontier with useful clean WIP | max | 60 min | restore |
| 5 | First max coherence-reset attempt | max | 90 min | exclude |
| 6 | First cumulative max attempt | max | 120 min | restore |
| 7 | First repeated hard-frontier reset | max | 180 min | exclude |
| 8 | First repeated hard-frontier continuation | max | 180 min | restore |
| `>=9`, odd `n` | `LONG_COHERENCE` reset lane | max | 300 min | exclude |
| `>=10`, even `n` | `LONG_COHERENCE` cumulative lane | max | 300 min | restore |

Effort is monotone within a frontier: medium → high → xhigh → max, with no
de-escalation until an exact promotion resets the frontier to medium. An
infrastructure failure
does not count as solver no-progress. A clean failure keeps WIP. A taint failure
discards that attempt’s WIP.

The primary-lane table remains unchanged; its purpose is coherent solver
accumulation. The orthogonal role escalation begins after every ordinary
reasoning effort and the first max turn have failed cleanly at one exact
boundary:

| Complexity trigger | Auxiliary role | Semantic-brief origin | Model/effort | Parallelism | Result authority |
|---|---|---|---|---:|---|
| `n<5` | disabled | — | — | 0 | none |
| `n=5..6` | independent side expert | authenticated native request or admitted supervisory handoff | launch-manifest pin | 1 | quarantine only |
| `n>=7` | distinct independent side experts on unresolved obligations | distinct authenticated native/supervisory requests | launch-manifest pin | at most 2 | quarantine only |

Supervisory-proposer eligibility is a separate projection of the same
coordinate:

| Complexity trigger | Supervisory role | Model/effort | Parallelism | Result authority |
|---|---|---|---:|---|
| `n<5` | disabled | — | 0 | none |
| `n>=5` plus new authenticated native/side-expert evidence | synthesize one schema-bound tactical handoff for the already selected frontier | launch-manifest pin | at most 1 active per frontier | quarantine-only hypothesis |

A supervisory round does not become eligible merely because `n` increases.
After the first round, the evidence epoch must change through newly admitted
same-frontier observations or a complete scheduler-defined reset/continuation
pair. It may use only otherwise-idle auxiliary capacity and cannot displace,
interrupt, or duplicate the primary proposer.

“New observation evidence” is content-derived: a canonical public
action-basis/response-signature receipt or a changed admitted solver-source
snapshot. A new attempt ID, timestamp, transcript envelope, result receipt, or
retry-count increment with the same public content does not change the
evidence epoch.

This schedule is driven only by exact-frontier clean-failure count and
taint-clean, receipt-bound complexity evidence. It has no game-specific
exceptions. Promotion resets `no_progress` and invalidates every auxiliary
profile, side-expert assignment, supervisory handoff, and completion record
for the old frontier.
Thus `medium → high → xhigh → max` and `max → max + sidecar(s)` are one
complexity schedule, not two manually coordinated heuristics.

The exploratory compatibility queue and runner enforce the same projection.
`codex_campaign_policy.py` accepts only `retry_complexity_n`, rejects
paid-turn count as a substitute, pins `--codex-allocation-policy=drain`, and
emits `max_campaign_runs=max_campaign_tokens=-1` with zero reserve when the
provider explicitly reports `unlimited`. `codex_campaign_runner.py` reopens
the live authoritative frontier immediately before launch and rejects a stale
coordinate, effort, allocation, WIP mode, auxiliary count, or cost mode. In
particular, an old max-effort attempt specification cannot cascade through a
promotion: the new frontier must be dispatched from a fresh `n=0` decision.
The sole generated `ARC_AGI3_CAMPAIGN_QUEUE.json` was refreshed at 181/183
after removing its obsolete 174/183 paid-attempt policy snapshot.
The same 2026-07-29 audit found that the compatibility reducer still grouped
legacy retries by `(game, target_level)` alone. That can pool attempts launched
from different promoted checkpoints or source parents and manufacture an
inflated `n`. The prospective path now binds every frontier, queue item, CLI
dispatch, protected `codex_exec`, and independently joined level outcome to
the exact parent checkpoint SHA-256, parent solver-source-tree SHA-256,
parent action count, and the contiguous-compatible frontier digest. The runner
reopens all of those values immediately before launch, and `gkm_legs.py`
compares the scheduler-supplied binding again after acquiring and seeding the
lineage but before starting Codex. Partial, forged, stale, or source-mismatched
bindings fail closed. Historical rows without that binding remain visible as
`unbound_legacy_turns_for_game_level` but have no authority over retry
complexity, effort, WIP, or sidecar selection. The regenerated queue carries
the complete binding and its dry run starts no model. An append-only
retrospective correction record is not itself binding evidence: it is accepted
only when a separately produced, canonical, content-addressed receipt can be
reopened through descriptor-safe regular-file checks and exactly seals the
thread, transcript, full parent binding, baseline hashes and commit, and audit
assertions. The reducer never manufactures that evidence. A trusted receipt
producer must still reconstruct and replay the launch parent and independently
audit Git, transcript, and taint evidence; until it exists, the four pre-schema
live turns remain unbound and non-counting. The focused
status/policy/runner/timeout/GKM suite passes 177/177, including
distinct-parent, unconsumed-binding, missing-receipt, symlink, hard-link,
directory-replacement, and read-race adversaries.

The live 2026-07-29 audit also exposed a taint-scanner ambiguity rather than
solver taint: the Python expression `bridges|pegs` inside a recorded heredoc
was misread as a shell pipe into the host command `ps`. The scanner now
tokenizes Python heredoc bodies and exempts only unquoted Python NAME tokens;
quoted AWK/data labels are distinguished from commands, while literal process
commands in shell pipelines, shell `-c` strings, quoted subprocess arguments,
and mixed allowed/forbidden monitoring commands remain taint. Its focused
adversarial suite passes 19/19, including no-space shell pipelines, quoted
Python subprocess commands, quoted shell commands, and AWK's `"top"` data
label. Both active `lf52` L9 transcripts and the previously ambiguous `sb26`
L5 promotion transcript rescan with zero hits. A full artifact audit then
finds zero hits in 150 canonical files and zero tainted promotion chains; its
remaining FAIL is due to incomplete historical manifest prefixes for `ft09`,
`g50t`, `r11l`, `sp80`, `tr87`, and `tu93`, plus forensic/noncanonical WIP,
not admitted solver taint. A scanner correction changes no attempt
classification or promotion on its own: the immutable transcript must be
rescanned under the released scanner and still pass every replay, provenance,
and manifest gate.
Because contiguous taint imports this general scanner, its source is now an
explicit ordered control-contract member rather than an unhashed transitive
dependency. A registered S11 owner proves both that binding and the
syntax-aware positive/negative cases; changing the scanner now necessarily
changes the control-contract identity and invalidates prior prospective launch
receipts. The same pass removed a package/top-level duplicate-module identity:
package-mode contiguous taint now imports the package-qualified transport
object, with a direct-script fallback only when the package is unavailable.
The registered owner binds that identity, so mutation adversaries cannot
monkeypatch a different module object from the one production scans use. The
complete taint-plus-conformance slice passes 65/65, including both same-size
post-inventory mutation cases.

The 2026-07-29 ACTION6 incident is a separate public-protocol escape, not
source/environment taint. The public click interface is exactly
`[6, x, y]`, where `x` and `y` are plain integer screen positions in `0..63`.
Scalar key tokens are exactly `1..5` or `7`; bare scalar `6` is invalid because
ACTION6 is never meaningful without both coordinates.
The former local Arena coerced arbitrary values and forwarded coordinates
outside the visible 64x64 observation. An `lf52` L9 proposer used that defect
to test an invisible world position. That entire model turn and every saving
written later in the same turn are non-authoritative; the clean L8 parent is
unchanged and L9 restarts without that WIP. A retrospective scan validates all
30,772 retained checkpoint actions and finds no illegal promoted path token,
but finds six affirmative `(320,320)` executions in the historical `tn36` L1
acquisition transcript. Because L2–L7 descend from that acquisition, `tn36`
must be reacquired from zero through 7/7 before replacement. The old lineage
remains archived and explicitly superseded.

`arc/audit_action_protocol.py` is the machine-readable audit for this fault
class. It scans current and promotion checkpoints plus unique promotion
transcripts and distinguishes deterministic token validity from the legacy
coverage limit: the former Arena did not retain a trusted log of every dynamic
exploration call. Therefore “no recorded incident” is not represented as
proof of a universal negative. The final contiguous 183/183 lineage supplies
that proof with the complete host-authenticated RPC call log.

The strict gate was exercised again during a separately rooted `lf52` L9
xhigh attempt on 2026-07-29: a probe called bare scalar ACTION6. The Arena
rejected it before engine or budget mutation, the shared marker invalidated the
turn even though proposer code caught the exception, WIP was suppressed, and
promotion failed closed. The append-only campaign ledger retains the typed
incident; the raw rejected turn, copied L1–L8 lineage, workspace, WIP, and
repo-local incident bundle were discarded. Its replacement restarted from the
unchanged exact L8 parent with WIP excluded. The reproducible negative evidence
is the unified conformance inverse, not the discarded turn; this is not a
solver failure or a canonical-lineage mutation.

An independent empty-root `lf52` reacquisition on 2026-07-30 exposed a
scaffold-level version of the same footgun at L1. The proposer passed the
advertised action set `(1,2,3,4,6,7)` to `perception.action_deltas`; the old
helper blindly forwarded every scalar and therefore emitted bare ACTION6.
The protected marker correctly discarded the complete zero-level turn with no
artifact, WIP, retry-complexity, or canonical mutation. The generated
perception scaffold now validates every action locally before touching Arena,
tests advertised key actions while omitting bare ACTION6 by default, and
accepts coordinate probes only as explicit in-frame `(6,x,y)` tuples. Focused
scaffold and public-protocol regressions pass 18/18. The contiguous image,
control-contract digest, and empty-root pilots must contain and exercise this
safe helper; host rejection remains the final authority rather than being
weakened by client-side validation.

These values govern admission and rotation, not cancellation of a healthy
active proposer. If a soft allocation expires while its proposer turn is still
running, the lane enters `DRAINING`: it launches no further turn, sends no
interrupt or termination signal, and lets the active proposer and its child
probes finish naturally. Only after the turn exits does the supervisor capture
the immutable transcript, run taint/replay/output audits, preserve eligible
clean WIP, and rotate or escalate the lane. A healthy turn may therefore drain
past 90, 120, 180, or 300 minutes.
This barrier is lane-local: no second turn may overlap the draining lane, but
unrelated lanes remain dispatchable and continue filling available capacity.

The exploratory compatibility runner exposes this distinction explicitly as
`--codex-allocation-policy=drain`: crossing `--minutes` records
`allocation_expired=true` and waits for the healthy live Codex process without
signalling it. Under the legacy `hard` mode, expiry is settled as
`containment_timeout`, preserves only current-policy-clean partial WIP, and
never increments the clean no-progress counter. On 2026-07-29 an `lf52` L9
turn launched before this fix was terminated at exactly 300 minutes and
incorrectly snapshotted as `not_reached` despite lacking a `turn.completed`
event. Its canonical checkpoint remained the exact replay-valid L8 parent and
all retained bytes scanned clean, but the outcome is journal-corrected to a
noncounting containment timeout by an append-only correction in the unified
`runs/codex_campaign_usage.jsonl` ledger; its replacement uses the drain
policy.

Forced termination belongs to a separate containment path, never to allocation
expiry. It requires a machine-observed containment failure such as a stale
heartbeat beyond its configured grace, a hard container resource violation, or
an explicit supervisor shutdown. The supervisor first requests container-local
graceful stop; after the independently configured containment grace expires it
destroys that attempt's container/cgroup and proves no descendants remain. It
must never use host name-based process signalling. Post-termination taint and
artifact audits still decide whether any WIP is eligible.

Longer turns are a deliberate reasoning-state escalation, not merely a larger
cost allowance. WIP preserves files and compact observations but cannot
preserve the model's live chain of reasoning. After repeated clean short-turn
failures, prefer one uninterrupted 90-, 120-, 180-, or 300-minute turn to
several equivalent restarts. Eligibility is determined only from the current
clean lineage's observable attempt history: the supervisor must not use
knowledge that another campaign or artifact solved the level. Do not interrupt
an already running clean turn merely to change its bound; apply the longer
bound and lane consolidation to subsequent attempts. The 300-minute value is
the top soft allocation in the ladder, not a hung-process boundary. Under
unlimited semantics it may be repeated after clean rotation until the frontier
is solved; the separately bound six-hour liveness/containment watchdog handles
actual hangs and leaves room for a healthy active turn to drain.

One legacy-launch incident makes the distinction concrete. On 2026-07-29 the
exploratory `lf52` reset turn
`019fab2e-ea1b-76b0-aeb8-b1d8ecbcd9e9` omitted the explicit drain flag and was
stopped after 18,001.78 seconds with no `turn.completed` event. Its workspace
and immutable transcript were taint-clean, but the outcome is append-only
corrected to non-counting `containment/hard_wall_time`; the partial context is
retained under a `containment_timeout_corrected` WIP phase rather than counted
as clean no-progress. The exploratory CLI, direct agent call, and public
`orchestrate()` entry point now share one `drain` default, with the hard-timeout
path available only when explicitly requested and regression tests covering
both behaviors. The contiguous scheduler must continue to bind the
policy-selected allocation mode in every attempt receipt rather than rely on a
launcher default.

The same generic correction was later required for an already-resident
pre-fix `bp35` L9 process, not as a game exception. Turn
`019fabad-d96a-79a2-9261-fa61bc12b002` was stopped after 18,002.943 seconds
without `turn.completed` or `turn.failed`; its protected transcript, workspace
copy, and final proposer log were byte-identical at SHA-256
`ae9af95ec01b49338382fcd99e610e3750d89ba8bc181ebc693f6d3c281e5c6e`
and all scanned clean. The exact L8 parent and its complete 8/8 promotion chain
remained valid. The append-only ledger correction therefore records
`containment/hard_wall_time`, `retry_increment=0`, and
`solved_target=null`; the retained context is named
`containment_timeout_corrected_449b17cbff86`, and its continuation explicitly
uses `drain`. The original false `not_reached` snapshot remains immutable
for audit but is no longer the resumable pointer.

Every proposer frontier brief records the verified parent’s exact
action-boundary length and remaining real-action budget. Reuse is admitted only
when observations support the same mechanism: forced reuse is not compression.
When a suffix is correct but uncommittable under the global action cap, the
proposer must falsify its inherited mechanic or optimize earlier composed legs
from pristine level entries instead of searching ever-longer suffixes.
On 2026-07-29 an observation-only `lf52` L9 sidecar made this case concrete:
the then-authorized exact 530-action L8 prefix plus an independently derived
100-action L9 suffix replayed successfully 3/3, but totalled 630 against the
fixed 600-action cap. Its conditional public-mechanics search closed that
suffix class at 100 actions; this is not a global lower bound and has
quarantine-only authority. Its uniquely valuable local report remains outside
the checkout pending terminal campaign review and is neither staged nor a
publication artifact. The cleaner 528-action L8 composition was later
host-admitted strictly as a quarantine-search seed; it did not replace the
authoritative 544-action canonical parent and has no promotion authority by
itself. The same sealed 100-action suffix replayed successfully 3/3 from that
exact candidate parent and reached the identical HUD-masked L10 entry frame,
reducing the best candidate full L9 path to 628 actions. If the 528-action
composition is incorporated into a new independently source-replayed and
manifest-gated lineage, the live acquisition obligation is at least 28 further
actions of exact prefix or L9 compression merely to promote L9, plus a
still-unmeasured positive L10 reserve to finish 183/183. A direct extension of
the 544-action canonical parent instead needs at least 44 actions of saving
before any L10 reserve. A genuinely new public mechanic may discharge either
obligation; the over-cap route itself can never be admitted.

The same live frontier also exposed why private probe state is never
authoritative. A nested-clone experiment appeared to reach a 46-action
one-visible-peg state, but a pristine foreground replay on 2026-07-29 retained
both pegs and showed that the apparent shortcut was a clone-reconstruction
artifact. The proposer rejected it before candidate handoff; it changed no
checkpoint, WIP admission, retry coordinate, or solved count. This is not an
`lf52` scheduler exception. Generically, any clone-derived or private-probe
observation is hypothesis-only until the host reproduces it through a fresh
public-Arena replay from the exact authenticated parent. The canonical S09
owner enforces this ingress rule directly: a divergent clone-derived candidate
path and a candidate-authored all-PASS receipt both leave the promotion pointer
empty, while path and winning-source replay from public zero remain mandatory.

On 2026-07-30 the long 528-action-parent `lf52` L9 native turn attempted an
out-of-frame ACTION6 probe. The public wrapper rejected it without gameplay
effect, but the attempt still made the complete turn protocol-invalid. This is
kept distinct from source, environment, or game-description taint. The legacy parent
waited for turn completion instead of terminating on the first protected-log
marker; when the operator interrupted it, the WIP path checked only writable
workspace files and briefly created a resumable pointer without reopening the
host-owned transcript. That pointer, snapshot, workspace, transcript, copied
sidecar, and repo-local failed-attempt bundle have all been discarded; the
typed ledger incident remains, none of the generation may be reused, and
canonical coverage remained 181/183. The compatibility harness now polls the
append-only protected transcript, terminates the full proposer process group at
the first protocol marker, classifies the complete generation as
protocol-invalid, writes no WIP or promotion, and reopens protected transcripts
at every snapshot gate. Focused regressions cover live termination,
orchestration suppression, and the formerly missing protected-transcript
snapshot check. The contiguous unified suite must prove the stronger container
analogue: no WIP, candidate, observation, sidecar request, handoff, registry
entry, resumable pointer, or promotion survives the first invalid action, and
restart cannot restore any byte from that generation.

A later isolated `lf52` replacement lane exposed one remaining compatibility
gap: its probe caught the public exception and printed only the error's Python
type, so the returned marker never reached the protected transcript even
though the agent explicitly self-invalidated the turn. The parent briefly
snapshotted that turn as WIP before operator containment. The isolated
scratch, transcript, and reacquisition roots were discarded; no candidate or
canonical artifact changed. The public Arena now writes the exact protocol
marker directly to the inherited protected transcript descriptor before
raising, so catching or abbreviating the exception cannot hide the violation.
A narrow structured-transcript fallback also recognizes an explicit
agent-authored admission that its own turn attempted an out-of-frame action.
Regression tests require both paths to suppress WIP and distinguish an actual
self-invalidation from hypothetical policy discussion.

The 2026-07-30 retrospective action-protocol audit scanned 30,778 actions from
current and promotion checkpoints: every token satisfied the public key/ACTION6
type-and-range contract, and admitted promotion transcripts contained zero
affirmative malformed or out-of-frame findings. The submission-taint audit,
which now independently treats any protected protocol marker as
release-blocking, likewise found zero such markers in admitted evidence. This
is an evidence PASS, not a claim of complete historical call logging: 18
coordinate-action games were acquired before every public call was recorded at
a trusted host boundary, so the strict legacy verdict remains FAIL by design.
The contiguous host RPC and its `--require-complete-call-log` release audit
close that evidentiary gap. The separate known `tr87` parent-Git metadata
containment incident remains a final-release blocker and is handled by the
uniform reacquisition phase; it is not game-source or environment taint.

Two subsequent fixed-class public-observation searches closed narrower prefix
obligations without changing that conclusion. The L8 search exhausted
1,469/1,469 public states through 38 actions; the L5 search exhausted
2,936/2,936 public states through 59 actions. Neither found a candidate,
neither hit its fixed state cap, and both have frontier zero. These are
class-conditional operational observations, not global lower bounds or
publication claims; their repo-local exploratory bundles have been discarded.

The next auxiliary obligation was not chosen by an operator or a game-specific
hint. Before search, a sealed selector ranked authenticated exact segment
streams from the admitted 530-action prefix by required-saving coverage,
segment contribution, and unproved public macro-step proxy, while excluding
the already closed L5, L8, and known-mechanics L9 classes. It selected the
144-action L7 segment with a fixed `<=114` target, 20,000-state cap, and no
widening. Its selector and pre-search generator were operational,
quarantine-only inputs with no live-mutation or promotion authority; the
repo-local copies were discarded after the active search was independently
rooted outside the checkout.

The selector's rank-2 L6 sidecar was conservatively terminated before it
produced a candidate or negative result. `Ctrl-C` caused Python to print
unsolicited `gkm_arena` implementation text in a traceback. Although the
traceback exposed no private state values, game labels, or solution semantics,
that proposer context is tainted by implementation text and can never be
continued or exported. The typed ledger incident binds no candidate, no
admission, and no retry increment; the raw traceback, superseded attempt files,
and repo-local receipt copy were discarded. Any replacement must start from a
fresh context through a harness path that sanitizes implementation tracebacks,
with that behavior covered by the unified conformance suite.

The repo-local exploratory quarantine tree was removed completely on
2026-07-30 and must not be recreated or committed. Raw invalid turns, duplicate
canonical lineages, copied Git repositories, workspaces, WIP snapshots, stale
pointers, clean no-candidate reallocations, caches, and preflight-only failures
are not publication artifacts. Invalid attempts settle to typed append-only
ledger metadata; raw bytes are deleted after the required terminal audit.
Uniquely valuable non-promoted evidence may remain only in a receipt-bound
campaign or private-temporary root until terminal review, never under the
checkout and never staged. It becomes publishable only after reduction to a
compact relocatable exact bundle that passes the complete manifest verifier;
otherwise it is wiped.

The 528-action composition was reduced to such a compact replay bundle and
replayed from its sealed bytes with canonical path SHA-256
`d62fda76086ba5bad55f1d58416b59adf8c0a0cf061e22e809fa83e058fcf0c7`.
`arc_agi3_exact_bundle.py` creates exact manifests atomically and verifies the
complete regular file/directory set. The unified suite's pre/post checkout
inventory fails if any repo-local quarantine tree, raw taint, mutable status
pointer, duplicate workspace, exception dump, cache, or stale manifest appears.
The same cleanup removed every known tainted canonical WIP snapshot and its
pointer, including the impossible `re86` L9 target, then reran the complete
submission-taint scanner: canonical sources, successful-candidate WIP,
discarded WIP, frontier scaffolds, and every present promotion chain had zero
taint hits. Its remaining release failure is exclusively the explicitly
missing legacy promotion directories scheduled for deterministic boundary
certification; it is not an archive-retention exception. Generated caches and
inactive run workspaces are never retained as evidence.

In an unlimited campaign, “quarantine” means cooling and rotation, not permanent
abandonment. A frontier returns after other disjoint work has advanced, after
new generic legs become available, or when the campaign enters the final
hard-frontier phase.

### 4.5 Candidate selection

Exploratory-campaign selection may balance:

- next-level depth and distance to 183/183;
- clean WIP quality;
- observed failures and the next effort arm;
- external evidence that the level is solvable;
- opportunities to reuse existing legs;
- expected new description length.

The harness’s historical free-energy score is

\[
F=-R+0.02C,
\]

where \(R\) is levels reached and \(C\) is cumulative positive retained
description growth. Lower is better. This is a selection/accounting proxy, not
machine-independent Kolmogorov complexity and not a license to prefer a smaller
solver that fails replay.

The colimit interpretation is structural: a new player should compose retained
legs whenever the dynamics repeat, and add a new leg only for genuinely new
mechanics. Literal replay remains charged through literal-container cost.

The later contiguous scheduler has the stricter, machine-recomputable
selection rule below and never receives external/comparator evidence. For each
eligible current-lineage frontier it computes

\[
\widehat F=-\widehat p+0.02\widehat C,
\]

where \(\widehat p=1/(n+2)\) is a fixed versioned scheduling prior—not an
empirical success estimate—after \(n\) clean same-frontier failures.
\(\widehat C\) is the zlib length of the positive unmatched normalized
top-level AST units for an eligible clean WIP source against its exact promoted
parent. It is a conditional-description upper-bound proxy, not conditional
compression or Kolmogorov complexity itself. If no eligible WIP exists, the
policy uses a fixed documented ignorance prior rather than a caller estimate.
Unchanged retained definitions contribute zero unmatched novelty; only direct
calls on the static named-call graph reachable from `solve` or a
`play_level_*` entry point are recorded as colimit-leg reuse witnesses. Dead
helpers and ambiguous duplicate names do not witness reuse. The scheduler
orders eligible games
lexicographically by durable least-recent dispatch sequence, \(\widehat F\),
reuse witness count, and game ID. The least-recent key prevents starvation;
the remaining keys make the Kolmogorov/free-energy preference deterministic
without allowing a small non-replaying program to count as progress. Every
metric, source-tree hash, and ranking key is included in
`SCHEDULER_DECISION` and reopened by the scheduler audit.

### 4.5 Promotion transaction

A level is solved only after this transaction completes:

1. The proposer turn ends and its immutable transcript is stored.
2. Agent-authored commands, searched paths, changed paths, and workspace files
   pass the taint scanner.
3. Every public action passes the exact key/ACTION6 type and range contract:
   scalar keys are `1..5` or `7`, while ACTION6 is `[6,x,y]` with plain-integer
   `x,y` in `0..63`. A rejected action invalidates the whole turn and all of
   its WIP.
4. The exact candidate source immediately before debrief is captured.
5. A fresh Arena replay reaches the claimed level from the parent checkpoint.
6. `checkpoint.json` is upserted once for that level.
7. `promotion_evidence/level_K/manifest.json` records parent hash, transcript
   hash, promoted-file hashes, validation, and taint verdict.
8. The canonical artifact is promoted atomically.
9. Exact-checkpoint, action-protocol, marginal/reuse, promotion-chain, and
   submission-taint audits are refreshed.

Speculative edits and same-level revisions remain WIP. They never enter the
verified count or complexity denominator.

### 4.6 Reporting cadence

During the long campaign, report only:

- a replay-validated promotion;
- crossing 156/183 or 165/183;
- reaching 183/183;
- a material taint, replay, integrity, or infrastructure failure;
- a blocker requiring user authority.

Under the superseding 2026-07-31 reorder, the conservative manuscript and
downstream-documentation phase is complete and its verified bundle is
preserved. The foreground task is now the exact `lf52` remainder, followed by
the complete release freeze, full 25-game ONLINE shakedown, one definitive
Competition-Mode scorecard, and PR #37 update as one verified transaction. No
public score, reviewer response, or PR surface is updated again before the
183/183 frozen set passes those gates.

### 4.7 Implementation and evidence map

| Purpose | Canonical location |
|---|---|
| Solver/promotion harness | `arc/crack_lab/gkm_legs.py` |
| Adaptive policy and runner | `arc/crack_lab/codex_campaign_policy.py`, `codex_campaign_runner.py` |
| Live status and effort telemetry | `arc/crack_lab/codex_campaign_status.py` |
| Codex allowance/ledger guard | `arc/crack_lab/codex_usage_guard.py` |
| Claude subscription and API guards | `arc/crack_lab/claude_usage_guard.py` |
| Codex usage ledger | `arc/crack_lab/runs/codex_campaign_usage.jsonl` |
| Codex subscription/unlimited ledger | `arc/crack_lab/runs/codex_campaign_usage.jsonl` |
| Claude API-console ledger | `arc/crack_lab/runs/claude_campaign_usage.jsonl` |
| Claude subscription ledger | `arc/crack_lab/runs/claude_subscription_usage.jsonl` |
| Claude sweep ledger | `arc/crack_lab/runs/claude_sweep_usage.jsonl` |
| Canonical promoted artifacts | `arc/crack_lab/agent_solutions/*_legs/` |
| Exact winning-boundary audit | `arc/audit_results/gkm-solved-checkpoints.{json,csv}` |
| Exact resumable action-boundary audit | `arc/audit_action_boundaries.py` |
| Uniform schema-v2 boundary certifier | `arc/crack_lab/arc_agi3_boundary_certifier.py` |
| Uniform marginal/reuse audit | `arc/audit_results/marginal-literal-reuse.json` |
| Taint and promotion-chain audit | `arc/audit_submission_taint.py` |
| Local/Competition replay | `arc/crack_lab/replay_scorecard.py` |

Codex and Claude proposers use independently locked ledgers. Claude records
carry their billing pool (`api_console` or `subscription`) per run rather than
being inferred from the filename. Codex records distinguish finite and
unlimited allowance windows, but allowance percentage is not a token or credit
unit and must remain separate. The Claude CLI exposes no readable provider
remainder, so a finite Claude subscription campaign uses local turn/wall
limits. An Anthropic API campaign may use a local dollar ceiling derived from
provider-reported token usage. For either proposer, an explicit
unlimited/`None` setting bypasses cost admission while retaining the evidence
gates above.

## 5. Uniform artifact completion

At 183/183, every level should have the same evidence shape:

- exact pre-debrief `legs.py`, `players.py`, and `solve.py`;
- exact winning action path;
- clean transcript and taint verdict;
- replay result from the correct parent;
- hash-linked parent and promotion manifest;
- one marginal-complexity record;
- explicit provenance for any deterministic auto-solve reconstruction.

Five old source gaps (`lf52` L3, `ls20` L2, `sb26` L3/L4, `su15` L7) have
byte-identical reconstructions for the post-hoc complexity audit. That does not
by itself manufacture a contemporaneous promotion lineage.

The historical `ls20` artifact was replay-valid at 7/7 and retained exact
source snapshots for all seven levels, but it had no promotion manifests. Its
zero-seed clean-room reacquisition reached 7/7 on 2026-07-28 and passed 7/7
exact winning-source, 8/8 exact-action-boundary, seven-manifest, complete
lineage, hash, and taint gates. It is now installed canonically. The duplicate
historical tree was purged after its checkpoint and tree hashes, reached count,
and replacement reason were folded into the schema-2 installation receipt.

The earlier “complete-lineage” taint gate only required at least one manifest
per game and therefore overstated uniformity. The strict gate now requires
exactly one sequential manifest and exact-action checkpoint for every level
`1..reached`. It currently exposes 21 missing historical boundaries across
`ft09` (L1–L4), `g50t` (L1–L4), `r11l` (L1–L4), `sp80` (L1–L4), `tr87`
(L1–L4), and `tu93` (L1). These games remain replay-valid, but the canonical
set is not uniform until those boundaries are reacquired or replaced by clean
complete lineages.

The strengthened unified transcript-containment audit also identifies one
separate historical incident: the `tr87` L5 proposer ran an unscoped
`git diff --stat` before scratch workspaces had their own Git roots, so Git
walked upward and printed parent-repository filenames. It exposed no game
source, solution code, environment state, or helpful game description, and a
fresh public replay still validates the exact 6/6, 208-action checkpoint; it is
therefore recorded as a containment/metadata defect, not source-assisted
taint. Nevertheless the release gate remains red. Uniform repair must replace
the affected evidence with a fresh isolated replay/certification chain, or
reacquire L5–L6 from the exact L4 boundary if certification cannot discharge
the acquisition contract. The old broad transcript is removed only after the
replacement is independently complete and its incident hash is bound in the
replacement receipt.

The `ft09` reacquisition also supersedes a legacy L2 debrief that imported the
public Arena harness from outside its scratch workspace and inspected only
public method names/signatures and frames. The audit classifies that as
`harness_introspection`, not game/environment-source or private-runtime taint;
nevertheless it violates the current stricter workspace boundary, so none of
that WIP is admissible to the replacement lineage.

Old `wa30` L1–L3 cannot honestly be converted into historical acquisition
boundaries from its later L3-complete source. The remedy is the fresh L1–L9
reacquisition:

1. run in a separate artifact root with no seed or WIP restore;
2. capture every boundary under the current schema;
3. require 9/9, full replay, taint PASS, and manifest-chain PASS;
4. record the old lineage's exact identity in the replacement receipt;
5. atomically install the new lineage;
6. purge the duplicate old tree after post-install verification;
7. regenerate all GKM audit rows and figures.

Status: the clean 9/9 lineage passed 9/9 exact-source, 10/10
exact-action-boundary, taint, and nine-manifest gates and was installed at the
canonical path on 2026-07-28. The installed uniform lineage uses an exact
597-action path. The former 596-action lineage's checkpoint/tree hashes and
supersession reason are retained in the schema-2 installation receipt; its
duplicate tree was purged. The post-install canonical source audit has no
missing winning-source levels.

Supersession retention is metadata-only once a replacement passes its complete
post-install gates. The repository must not retain duplicate historical trees,
cache files, stale checksum lists, exception notes, or informal recovery
bundles. A retained evidence bundle must have one current exact root manifest
and pass the archive-wide verifier; otherwise it is rejected or purged rather
than left for a reviewer to decipher.

No retrospective certification may be mixed into the historical marginal
sequence as if it were an acquisition boundary.

### 5.1 Frozen schema-v2 certification

After coverage reaches 183/183, migrate the complete frozen canonical tree
through `arc/crack_lab/arc_agi3_release_gate.py`. The release gate does not
grandfather a schema-1 manifest or a Boolean claim such as `validated=true`.
Every one of the 183 exact boundaries must have separate hash-bound records for
the checkpoint, complete winning source, immutable host transcript, taint
audit, path-from-zero replay, source-from-zero replay, hash audit, and chained
schema-v2 manifest.

Start with the deterministic diagnostic:

```sh
python3 arc/crack_lab/arc_agi3_release_gate.py diagnose
```

Use its per-game/per-level migration queue to generate missing evidence through
trusted replay and audit code. Mutable campaign state such as `.campaign_locks`
and `wip_context` is excluded from the frozen evidence root. A legacy boundary
may be certified only by genuinely rerunning the required replay/audit gates;
wrapping its old Boolean fields in new JSON is invalid. Missing historical
boundaries require reconstruction or clean reacquisition. This certification
is a release proof and does not retroactively rewrite the historical
acquisition marginal.

When all 183 schema-v2 boundaries pass, issue one content-addressed release
receipt outside the frozen canonical tree, verify it against the unchanged
tree, and bind that receipt into ONLINE scoring, Competition scoring,
manuscript reproduction, and contiguous-run launch preflight.

The final release must also ship (or bind by immutable source revision) the
receipt-aware replay entry point and record exact all-game preflight, ONLINE,
and Competition commands. A clean-extraction regression must execute the
preflight command successfully and confirm that every README claim about
release contents names a file or immutable link that actually exists.

The mutable acquisition tree is not a publication archive. After the
schema-v2 release tree has passed fresh replay, taint, hash, boundary, and
manifest gates—and every scorecard/manuscript consumer has been switched to
that content-addressed tree—delete its `wip_context`, caches, locks, inactive
scratch trees, and installed reacquisition copies. Preserve historical
pre-debrief source authority only through the minimal certified source and
transcript bytes copied into the release tree. Do not ask readers to excavate
thousands of superseded WIP files.

The migration is performed by
`arc/crack_lab/arc_agi3_boundary_certifier.py`, which never rewrites the live
acquisition archive. It selects an exact retained source per boundary, stages
all local source dependencies, runs source-from-zero and path-from-zero replay,
creates a host certification transcript, scans every admitted byte, records
whether the source is a contemporaneous boundary or a deterministic
reconstruction, and builds a separate minimal schema-v2 tree. A deterministic
reconstruction can certify replay and release reproducibility but cannot be
relabeled as a historical acquisition marginal. The certifier and its tests
are part of the contiguous conformance/control-contract digest.

The certifier applies the clean-room filesystem/import policy to the exact
winning-source bytes at selection, again immediately before any host replay,
and again before release staging. Certified `solve(env)` source receives its
Arena object from the host and therefore has no raw-Arena import exception:
an absolute or parent path, private/sibling harness import, dynamic loader, or
other host-filesystem capability fails before execution. The compatibility
runner's separately hash-pinned `gkm_arena.run_program` capability remains
limited to acquisition probes and does not transfer into certified solver
source.

Source selection is phase-first, not merely replay-first. For a given boundary,
an exact retained `reached_before_debrief` snapshot outranks every
post-debrief/promoted or schema-1 source. A schema-1 promoted source is labeled
`legacy_schema1_promoted_source`; it is never inferred to be a historical
pre-debrief boundary. Historical-marginal authority additionally requires a
strict, complete, single-turn JSONL acquisition transcript: every line parses,
only admitted event types occur, the turn lifecycle closes, and no started
item remains open. Transcript or provenance failure is hash-bound into the
certification record and host transcript and fails closed for historical
authority. A malformed transcript, post-debrief source, or deterministic
reconstruction may still prove release replay when its applicable gates pass,
but it must set both `historical_source_boundary=false` and
`posthoc_acquisition_marginal_admissible=false`.

The native Codex launcher closes proposer stdin with `DEVNULL`; the complete
task is already bound in argv. Inheriting the supervisor PTY is forbidden
because it admits an unintended operator-input channel. Codex identifies the
closed descriptor as a piped stream and can still emit the deterministic
`Reading additional input from stdin...` diagnostic. The corrected launcher
therefore seals stdout as the strict JSONL acquisition transcript and seals
stderr as a separate immutable diagnostic sideband. Both files are scanned for
taint and action-protocol markers, reopened after complete process-group
quiescence, hashed into the turn ledger, and copied into promotion evidence;
failure to reopen either half invalidates the complete generation. Focused
regressions assert the closed descriptor, strict JSON stdout, sealed sideband,
and fail-closed sideband taint handling. A winning turn produced by an older
launcher that merged this diagnostic into JSONL may retain release-replay
evidence, but it cannot supply historical marginal authority and must be
reacquired from its exact parent under the corrected launcher before the final
uniform lineage is frozen.

The `bp35` L7 exploratory promotion illustrates this distinction. Its exact
196-action L6 parent and 256-action L7 boundary replay cleanly, and both the
retained pre-debrief and promoted post-debrief sources reproduce the same
resumed L7 route. However, the promoted source is post-debrief and the raw
acquisition/debrief logs contain non-JSON lines, so L7 has no historical
marginal authority under schema v2. The certifier must genuinely run the
first-ranked retained pre-debrief source from zero. If that source does not
reproduce L7, the exact boundary remains eligible only for the deterministic
release capsule below. This does not reduce exploratory coverage: `bp35` L8
subsequently promoted with a clean exact chain. On 2026-07-29, L9 then
promoted from the exact 310-action L8 parent with a 393-action replay-valid
path. The contemporaneous pre-debrief boundary
`reached_before_debrief_2545bedffa0e` independently replays to 9/9, its
workspace and proposer transcripts scan clean, and the L9 promotion manifest
has a complete hash-valid L1–L9 parent chain with no taint hits. The debrief
refactored the winning source, so historical complexity accounting continues
to use the captured pre-debrief source rather than the later promoted bytes.
The stale independent L9 sibling was invalidated and stopped only after this
promotion. This brings the live verified campaign to **181/183**, leaving only
`lf52` L9–L10.

The subsequent archive cleanup exposed one fail-open edge before it could
produce a promotion: three already-running `lf52` turns retained open file
descriptors after their shared protected-transcript directory entries were
removed. Their source and probe generations were discarded, not resumed, and
the clean count remained 181/183. The resident runner now seals a Codex JSONL
only through a single-link, regular-file descriptor read whose pathname,
device/inode, link count, size, and modification time remain stable through the
complete read. A missing, replaced, aliased, or unlinked protected transcript
is ledgered as `evidence/protected_transcript_unavailable`; it permits no
same-generation retry, recovery, WIP snapshot, or promotion. If the loss occurs
in an optional debrief, the independently replayed pre-debrief source is
restored and no unrecorded debrief bytes enter WIP or promoted source. The
same gate rejects a terminal Codex leader when any spawned process remains in
its process group: the complete group is terminated before transcript sealing,
and that generation is non-authoritative. The focused evidence/process-tree
regressions and the complete `test_gkm_legs.py` suite pass (109 tests). The
quarantined 528-action exact-L8 `lf52` compression lane retained its own intact
protected transcript but never acquired canonical promotion authority.
Subsequent L9 turns bind the authoritative 544-action L8 parent and must earn
any shorter prefix through the ordinary fresh-replay promotion gate.

On 2026-08-03, one observation-only auxiliary analysis was run at that exact
unchanged frontier without exposing it to the live native proposer. A
quarantined supervisory handoff (`dd692811...`) requested the bounded
state-representation obligation `OBL-L9-STATE-01`; its outer 119-file input
bundle is exact-manifested as `d5197bd6...`. The sidecar stopped within its
fixed caps at 27/32 clean replays, 75/75 prefix snapshots, and 0/128 successor
branches. Its complete 13-case terminal factorial associated reward only with
the five-shift far-endpoint capture, but generic one-peg completion is perfectly
confounded with that endpoint in the matrix. The prefix audit failed safely on
an incorrect instrumentation boundary after exhausting its snapshot cap and
was not rerun. The resulting JSON (`a14944e5...`) independently passed frontier,
content-addressed sequence, replay-count, taint, action-protocol, and authority
checks and is sealed by output manifest `8bbf7903...`; its verdict remains
`narrowed_not_discharged`. It is not a candidate, promotion, retry, WIP update,
or publication artifact and cannot enter a native lineage without a later
authenticated handoff plus native public-observation reproduction.
A second quarantine-only supervisory synthesis over exact input manifest
`30e3f124...` produced handoff `e4086b30...`, sealed by manifest `0536d4a7...`.
It contains three bounded, unexecuted native-reproduction tests and remains
forbidden from the currently live native turn.

A transient ultra-effort follow-up then attempted only the handoff's T2
obligation. It repaired the earlier prefix-67 bookkeeping failure and completed
the bounded enumeration, but its sealed input did not retain the comparison
table for prefixes 0--66 or predicted successor deltas; an earlier interface-
discovery invocation also lacked a cumulative action ledger. T2 therefore
remains undischarged. The follow-up had no authority, was not exposed to the
live proposer, and is not retained as release evidence. Another auxiliary retry
is inadmissible unless a fresh authenticated native request supplies the
missing public-observation comparison evidence.

If every retained source for an otherwise exact replay-valid boundary fails
fresh source execution, the certifier may use a minimal deterministic source
capsule generated from that boundary's canonical exact action prefix. This is a
last-resort release-reproducibility mechanism: the capsule and its checkpoint
basis are hash-bound, independently replayed from zero, taint-scanned, and
labeled `canonical_exact_action_boundary`. It is never a recovered historical
winning-source snapshot, never contributes a historical marginal or sawtooth
witness, and never upgrades the original acquisition provenance. The later
contiguous campaign remains the source of a fully contemporaneous straight
lineage.

## 6. Post-hoc complexity and sawtooth analysis

Source or game-code interpretation happens only after promotion and never flows
back into a proposer workspace.

Dynamics labeling is necessarily semiautomated. Scripts provide the immutable
source pair, replay outcome, AST delta, and candidate reuse witnesses; a human
or LLM then reads the already-solved code and records the semantic label and
rationale. The exact inputs, output label, and reviewer identity/session are
retained so another reviewer can repeat or challenge the judgment.

For every adjacent exact winning boundary, record:

- conditional normalized-AST novelty;
- historical `marginal_C`;
- unchanged named definitions;
- static direct-call reuse witnesses;
- whether the marginal fell, rose, or stayed flat;
- a post-hoc dynamics label: `reuse`, `extension`, or `novelty`;
- a source-grounded rationale;
- manifest and parent hashes.

A “sharp drop” means the current conditional marginal is at most half the
preceding nonzero marginal. A “hard reuse witness” means the winning entry-point
AST contains a direct call site to an unchanged named definition; it is static
evidence, not runtime branch instrumentation. A “coupled witness” has both on
the same transition.

The sawtooth claim is accepted only where code, replay, and dynamics agree:

- repeated dynamics + unchanged called leg + complexity drop supports reuse;
- genuinely new mechanics + retained solver growth supports novelty;
- contrary or flat cases remain visible.

## 7. Final remote scorecards

Do not publish another interim scorecard. Diagnostic cards are not release
artifacts:

- ONLINE shakedown `d4bc8e26-8959-47c3-a281-f1f0f5c66320` reproduced the then
  current 136/183 paths and reported 64.0488938523768%.
- Competition diagnostic `0aa609c5-6fd0-4893-b348-53184c73a834` exposed an
  ambiguous-timeout recovery bug on `lp85`; the server ultimately retained the
  win, but the client reported failure. It must not be linked as the final card.

The recovery code now rebuilds from the level actually reported after reset,
and its focused test suite passes.

After canonical 183/183:

1. Freeze the game list, checkpoint hashes, paths, and action counts.
2. Run all 25 games in ONLINE mode.
3. Require every remote endpoint to equal its frozen local endpoint.
4. Exercise/reset-check the rollback recovery behavior.
5. Close and archive the ONLINE shakedown card and report.
6. Run the identical frozen plan once in Competition Mode.
7. Require 183/183, a closed public card, and consistent per-game accounting.
8. Record weighted score, raw coverage, complete games, actions, resets,
   toolkit version, source commit, and scorecard URL.

Only the definitive Competition card enters the submission.

## 8. Manuscript and comparator release

The final manuscript compares GKM uniformly with OPINE, Retrodict, and
baseline1. The comparison must retain the exact meanings of:

- exact measured winning checkpoint;
- exact adjacent transition;
- comparable marginal transition;
- decrease;
- sharp drop;
- hard static reuse witness;
- coupled witness.

OPINE’s measured objects must be called pre-solve engine checkpoints where the
complete transient analyzer policy is unavailable. Retrodict’s retained memory
trajectory is admissible experience, not executable winning-source evidence.
baseline1’s current public harness should be described separately from its
disclosed and excluded development failures. No system is called tainted
without evidence of privileged code/environment/runtime access that helped
solve a game.

### 8.1 Documentation hygiene

Mutable campaign status has exactly one prose source of truth: this file. Live
counts are derived from canonical checkpoints and audit outputs, not copied into
READMEs, the methods chapter, or reproduction instructions. Published scorecards
and manuscript results may retain old numbers only when explicitly labeled as
frozen historical snapshots.

Before the final release, sweep `README.md`, `REPRODUCE_ARC.md`, `arc/README.md`,
`arc/ARC.md`, `docs/self_improving_agent.rst`, and every manuscript source for
stale claims. Regenerate every generated table and figure, then require a
repository-wide search to find no obsolete count described as current. Git
history is the archive; do not retain superseded planning documents merely as
historical records.

Release reproduction:

```sh
python3 arc/audit_submission_taint.py \
  arc/crack_lab/agent_solutions \
  --require-complete-lineage \
  --json /tmp/gkm-release-taint.json
PYTHONPATH=arc python3 arc/audit_action_protocol.py \
  arc/crack_lab/agent_solutions \
  --json /tmp/gkm-release-action-protocol.json
MPLCONFIGDIR=/tmp/gkm-release-boundaries \
  python3 arc/audit_action_boundaries.py \
  arc/crack_lab/agent_solutions \
  --require-complete-chain \
  --summary-only \
  --json /tmp/gkm-release-action-boundaries.json

make -C arc/manuscript test
make -C arc/manuscript reproduce

# With checksum-pinned raw comparator archives:
make -C arc/manuscript reproduce-full \
  OPINE_ARTIFACTS=<path> \
  BASELINE_RELEASE=<path> \
  BASELINE_REPO=<path> \
  RETRODICT_RUNS=<path>
```

The release must regenerate:

- GKM and comparator audit JSON/CSV;
- the canonical transcript/workspace taint and promotion-chain audit;
- the canonical ACTION6 token/acquisition-evidence audit, including explicit
  legacy call-log coverage limits;
- for the contiguous lineage, the container-policy audit over image digest,
  mounts, environment-variable names, network policy, and declared input/output
  hashes;
- for the contiguous lineage, the supervisory-handoff audit over complexity
  trigger, exact frontier, admitted native/side-expert inputs, pinned
  model/usage, Socratic challenge, native reproduction, and downstream
  promotion ancestry;
- comparison Markdown and TeX;
- manuscript tables;
- sawtooth and campaign figures;
- deterministic PDF/PNG outputs under the pinned rendering environment;
- `reproduction_report.json` and `SHA256SUMS.txt`;
- the compiled manuscript without warnings that affect validity.

### 8.2 Conservative empirical manuscript update

The author explicitly advanced this phase on 2026-07-31. The conservative
update and downstream-documentation pass are complete and verified; later
changes remain subject to the same evidence and parsimony rules below.

The theoretical development is frozen for this release. Do not rewrite the
definitions, structure-function/free-energy derivations, inverse-colimit and
cofibration results, compute-completeness theorem, or theoretical discussion
unless a concrete correctness error is found and documented for human review.
The experimental prose is also changed sparingly. The default decision is to
leave existing prose intact and strengthen the paper through audited data,
reproduction code, figures, and compact tables.

Keep every manuscript edit local until the human author reviews it. Do not push
or publish manuscript changes as part of the benchmark-PR mutation.

The preferred additions are:

1. extend `arc/manuscript/scripts/` so every new empirical table and figure is
   generated and checked by the unified reproduction suite;
2. update the existing `wa30` and `ls20` figures with the fuller uniform
   level histories while preserving the concepts and graphical language already
   used in the paper;
3. add one compact, auditable table of marginal-complexity trajectories for all
   25 games, using explicit missing/excluded markers rather than invented
   values; and
4. change captions or nearby empirical text only where the regenerated evidence
   makes the existing statement stale, incomplete, or materially weaker.

Perform exactly three conservative Socratic passes:

1. **Evidence pass:** for each proposed empirical change, ask which replay,
   taint, exact-boundary, manifest, or comparator artifact supports it. Reject
   the edit if the answer is not exact and reproducible.
2. **Scientific-value pass:** ask whether the new script, figure, table, caption,
   or local sentence makes acquisition, reuse, novelty, or cross-system
   comparison materially clearer. Reject cosmetic expansion and avoid changes
   to the theoretical sections.
3. **Parsimony/reproduction pass:** remove duplicative additions, reconcile only
   the empirical numbers that actually changed, regenerate all affected outputs,
   and run the complete manuscript test/reproduction suite.

Record the questions, retained changes, rejected edits, and evidence from all
three passes in the Socratic revision record. “No edit” is an acceptable and
preferred pass result when the current manuscript is already correct.

### 8.3 Authorship boundary

The manuscript remains solely authored by **Alexander Kolpakov**. The ARC
Community Leaderboard submission metadata separately lists
**OpenAI GPT-5.6** as a submission author: the model, not OpenAI as a company.
Do not add OpenAI GPT-5.6 to the manuscript byline or add a more detailed
contribution allocation.

## 9. ARC-AGI Community Leaderboard submission

Target: `arcprize/ARC-AGI-Community-Leaderboard` PR
[#37](https://github.com/arcprize/ARC-AGI-Community-Leaderboard/pull/37).

Required final title:

> Add GKM — `<official Competition %>` / `<raw level %>` raw:
> general-purpose replay-gated self-improving program synthesis

Fill both percentages only from the definitive closed scorecard and its
receipt-bound raw frontier. The title should emphasize the method while making
the verified major score update immediately visible.
The editable PR body is the persistent top-of-page submission summary and must
be replaced in full; do not rely on a later comment as a pseudo-pinned
description. GitHub comments are not treated as pinnable substitutes for the
body. Post only short, question-specific maintainer replies below it.
The final body and `submissions/gkm/` files must:

- lead with GKM as a general-purpose, compute-bounded self-improving
  program-producing architecture;
- explain that per-game and per-level solver programs are the learned,
  model-authored state produced and promoted by the general meta-loop, not
  evidence of separate human-written architectures;
- state the final weighted Competition score and separately state the exact
  receipt-bound raw level coverage of this release.  The present major update
  may close at the strongest fully audited frontier (currently 181/183);
  reaching 183/183 later requires a new full audit, ONLINE shakedown,
  Competition card, version entry, and title/body refresh rather than silently
  changing the frozen release;
- link only the definitive Competition scorecard;
- distinguish local clone-capable discovery from official-interface sample
  efficiency;
- summarize clean-room, replay, exact-boundary, and manifest gates;
- link the code, manuscript, reproduction command, frozen audit bundle, and
  scorecard;
- report the model/proposer lineage and describe the accounting method, but do
  not publish internal token, allowance, credit, wall-time, or dollar totals in
  the submission, manuscript, or public-facing repository documentation unless
  the leaderboard schema makes them mandatory;
- describe the Schmidhuber/PowerPlay/compression-progress relationship without
  equating the proxy with true Kolmogorov complexity;
- avoid stale interim counts and diagnostic scorecards.

The **GitHub benchmark submission itself** must make code and decision
provenance legible without requiring a maintainer to infer it from the
manuscript. This expanded explanation is a benchmark-submission requirement;
it is not a requirement to reproduce the same operational detail throughout
the manuscript.

Replace binary labels such as “hand-coded” or “automatic” with a concise
component-by-component provenance map that covers:

1. **Fixed research infrastructure:** the Arena/public-observation adapter,
   clean workspace scaffold, proposer launcher, replay runner, taint and
   action-protocol scanners, hashing, manifests, and scorecard tooling. State
   which of these were supplied by the research harness rather than learned as
   game solutions.
2. **Native solver proposer:** identify the artifact-specific proposer lineage
   accurately: the scored historical card used a legacy Claude Code lineage,
   while the later controlled expansion uses GPT-5.6-sol. In either lineage the
   native proposer inspected only its admitted workspace and public interaction
   history, then authored and edited the game-playing programs, including
   perception functions, solver legs, searches, planners, literal paths, and
   dispatch logic. Do not call those artifacts human-written merely because a
   human or supervisory model launched or reviewed the turn.
3. **Session-level meta-proposer/supervisor:** this Codex session selected
   targets, effort, continuation mode, and independent side analyses; diagnosed
   harness and infrastructure failures; and could provide only authenticated,
   observation-derived handoffs. State plainly that this was active
   model-assisted supervision, not the later deterministic unattended
   scheduler and not an unreported source-code author inside native solver
   workspaces.
4. **Auxiliary side experts:** sidecars could diagnose or propose
   observation-derived tactics, but their output remained quarantined until a
   clean solver lineage adopted it and the host independently replayed it.
   Sidecar output by itself never counted as a solved level.
5. **Trusted host verifier:** only the verifier could admit a result, after
   action-protocol and taint scans, exact fresh replay, capture of the winning
   pre-debrief source boundary, complete hashes, and a provenance-bound
   manifest. Neither the native proposer nor the supervisor could simply
   declare a promotion.
6. **Human role:** state the actual human contribution at the right altitude:
   setting the research objective and constraints, authorizing resources,
   reviewing evidence and priorities, and deciding what to publish. Do not
   describe the campaign as fully autonomous, and do not imply that these
   research decisions amount to hand-authoring the solver routines.
7. **Artifact path:** show the auditable chain from proposer transcript and
   workspace source, through the exact winning boundary and replay receipt, to
   the hashed manifest and promoted submission artifact. Explain how rejected,
   tainted, post-boundary, or unverifiable work is excluded.
8. **Interface caveat:** separate clone-capable local discovery from official
   Competition-mode evaluation, and report the latter only from its definitive
   scorecard.

Include a compact provenance table for the material submitted components, with
columns for component, origin/authoring agent, admitted inputs, transcript or
source boundary, verifier receipt, and promoted artifact. This table need not
list every helper function, but it must cover the shared scaffold, learned
per-game programs, the meta-proposer, sidecars if any influenced submitted
artifacts, and the promotion/scorecard path. Link each row to retained evidence
where that evidence can be published. No individual solver leg is a privileged
explanatory object: the submission must establish the provenance rule for all
game-playing source.

The maintainer's existing question receives a concise factual correction:
the referenced game-playing code was authored inside a native coding-model
turn rather than hand-written by the human operator, and the retained turn was
Claude Sonnet 5 rather than the later GPT-5.6-sol campaign. That reply points to
the general provenance section and does not add a routine-specific section to
the submission.

Detailed resource accounting remains an internal campaign audit: aggregate by
provider, model, and billing pool; preserve token classes and missing-telemetry
counts; keep credits and allowance percentages distinct; and never invent a
dollar conversion for subscription or unlimited usage. Store the reproducible
aggregate with the campaign audit outputs rather than copying totals into
public-facing prose.

The final manuscript acknowledgments must include this one-line credit:

> We thank OpenAI for providing access to GPT-5.6-sol.

GitHub CLI authentication is already configured.  Do not reauthenticate or
replace credentials because a sandboxed `gh auth status` probe is misleading;
use the existing authenticated mutation path and verify every remote change.
The leaderboard branch's update contract must be followed exactly. The verified
v2.0 payload already preserves v1.0 and records the 181/183 release; the final
183/183 payload must preserve both and append a new version rather than rewrite
their historical scorecards:

- preserve the historical v1.0 and v2.0 entries and append one new complete
  release entry under `versions`;
- for ARC-AGI-3, provide `scorecard_url` and `set: public`; do not add a
  forbidden numeric `score`;
- omit optional public `cost` unless the schema later makes it mandatory;
- list **OpenAI GPT-5.6** (the model, not OpenAI as a company) in submission
  `authors`, while keeping Alexander Kolpakov as the manuscript's sole author;
- point `code_url` and all evidence links at the exact frozen public source
  revision containing both the producing system and the promoted outputs—the
  generated paths alone are not an open-system submission.

Then:

1. update `submissions/gkm/README.md` and `submission.yaml`;
2. run the offline v3 release gate and the exact current upstream
   `.github/scripts/validate_submission.py` in a clean leaderboard checkout,
   recording the upstream validator SHA-256 and its successful result;
3. commit and push the `gkm-submission` branch;
4. change PR #37 to the gate-rendered title and set its body byte-for-byte to
   the validated `submissions/gkm/README.md`;
5. run `arc_agi3_leaderboard_v3_gate.py --verify-post-push` against the clean
   exact pushed head, passing the leaderboard checkout, full head SHA, and
   recorded upstream-validator SHA-256.  This read-only mode must reopen PR
   #37, bind its base/head repositories and branches, compare the remote YAML
   and README bytes, require exactly those two changed files, resolve every
   public URL, rerun the pinned upstream validator, and recheck the PR after
   the network reads;
6. require the exact-head `Validate Submission` workflow and `validate` check
   to pass.  `MAINTAINER_ACTION_REQUIRED` or `WORKFLOW_NOT_COMPLETE` is an
   explicit nonzero, not-complete result, never a release PASS;
7. inspect rendered links and reviewer comments, then report only confirmed
   remote changes.

## 10. Automated contiguous-campaign orchestrator

After the canonical 183/183 result is frozen and released, build and start a
fully automated background contiguous orchestrator using the final successful
policy. Its fail-closed safety kernel remains the contiguous supervisor module,
while the runner and container backend provide the execution layer.

The executable fail-closed contract is
`arc/crack_lab/arc_agi3_contiguous_supervisor.py`. Focused component and
regression tests remain useful development evidence, but no collection of
independently invoked test files authorizes launch. The launch authority is one
canonical end-to-end contiguous-campaign conformance suite, implemented at
`arc/crack_lab/arc_agi3_contiguous_conformance.py`, with one versioned invariant
registry, one entry command, and one machine-readable result artifact. The
eventual scheduler must import these gates rather than reimplementing them. It
must call
`launch_preflight()` with the exact container image digest before creating any
proposer process; a prose checklist or hand inspection is not launch
authorization. The preflight rejects a tested-image/launch-image mismatch, a
stale attestation whose control-code/test-tree hash differs from the current
tree, any change to the exact tested 25-game per-game inventory map, or any
suite command other than the canonical conformance command.
Immediately before launch, `launch_preflight()` also executes that exact
conformance suite again from the current control tree without a shell and
requires both an observed zero exit and a fresh valid result artifact. An
attestation field that merely claims `suite_exit_code: 0` is therefore not
sufficient launch authorization.
The unified suite binds the proposer harness, allowance guard, adaptive policy,
runner, Docker backend, Arena RPC, container worker, status/escalation logic,
supervisor store, release receipt, replay scorecard, action boundaries, taint
audit, and exact-winning-checkpoint audit into one cross-component contract.
Changing any bound source, invariant registry, supporting test, container
recipe, or expected result schema invalidates the control-contract digest.

The invariant registry is exact: every required launch invariant has one stable
identifier and one owning scenario. Missing, duplicated, skipped, unknown, or
unexpectedly xfailed identifiers make the aggregate result fail. A green
wrapper around a subset of focused tests is not a conformance pass. The result
artifact binds every scenario receipt, the frozen 183/183 release receipt,
container image digest, control-contract digest, authoritative inventory hash,
suite source hash, start/end timestamps, and observed exit status.
Before importing or executing any bound implementation or test, the driver
captures an immutable start snapshot of the complete control manifest and runs
against those exact bytes. It rehashes the live manifest after the final
scenario and requires byte-for-byte equality with the start snapshot. A
mid-suite mutation, including one that would make the final hash describe code
different from the code actually executed, invalidates the entire result.
The ordered control-file manifest and its digest algorithm also have exactly
one implementation in the conformance module. Supervisor preflight, container
attestation, and the release gate import that implementation; no component may
maintain a separately ordered copy of the file list or reimplement its hash.

The current `arc_agi3_contiguous_supervisor.py` is deliberately an admission,
promotion, recovery, and launch-preflight library; its CLI performs preflight
only and is not itself the campaign scheduler. The separately tested
`arc/crack_lab/arc_agi3_contiguous_runner.py` and formal orchestrator now supply
the state-machine and execution layers described below. Their existence and
focused tests are preparatory evidence only: they do not create production
launch authority.
The formal operator entry point is
`arc/crack_lab/arc_agi3_contiguous_orchestrator.py`; the runner remains its
state-machine library. The orchestrator requires every campaign root,
authoritative inventory, image/model/protocol digest, limit, isolation policy,
and conformance receipt explicitly, rejects unknown or duplicate options
before any filesystem mutation, invokes only absolute digest-bound
executables under a minimal environment and neutral working directory, and
runs the durable cycle/recovery/monitor loop until the declared terminal
condition. It never finds `python3`, Codex, Docker, configuration, credentials,
or plugins through ambient `PATH`, `PYTHONPATH`, login-shell state, pytest
environment, or user project discovery.
The executable preflight is explicitly fail-closed today, even for
schema-valid attestation JSON. Full production launch remains blocked until
genuine S01--S12 production observers and receipts, a receipt-bound minimal
role-image build with an immutable repository digest, the exact runtime and
production-stack evidence, the real ordered-pilot executor and authenticated
pilot gate, and the canonical frozen 183/183 release receipt are all produced,
reopened, and bound into one receipt-derived launch authority. Unit or mocked
passes cannot substitute for any of those objects.

The 2026-07-29 blocker-authority, candidate-ingress, policy-steering, and
auxiliary-dispatch passes were development checkpoints only. Their test counts
and control digests became stale as soon as the control tree changed and are
therefore not copied into this source-of-truth plan. Every current count,
policy identity, registry identity, protocol identity, and aggregate
control-contract digest belongs only in a freshly generated external
conformance receipt that reopens the exact tested bytes.
`launch_authority=false` throughout. Those passes fixed descriptor/cache races,
duplicate same-round diagnoses,
containment-payload precedence, durable recovery's incomplete trusted
worker-hash set, proposer-controlled blocker authority, and read-only
winning-source staging that previously could mask a fail-closed rejection or
stale-stage recovery with a cleanup error. Publication and recovery now share
one descriptor-safe staging cleanup path. Registration and result verification
now consume the
independently authored, control-hashed
`arc_agi3_contiguous_launch_requirements.json`, which owns exact S01–S12
requirements and rejects deletion, insertion, renaming, scenario-owner
substitution, and test-owner substitution. Registered adversarial checks also
make a real cycle reject stage reordering before journal mutation and prove
that near the 24 MiB journal bound only appended suffix event bytes are parsed
after descriptor-safe immutable-prefix metadata revalidation.
The steering pass additionally removed two forms of hidden operator authority:
a rejected promotion now discards only the candidate, retains eligible
same-frontier WIP, keeps `n` unchanged, and returns to `READY` under one of the
finite generic codes `promotion_gate_rejected` or
`promotion_commit_invalid`; it cannot create a permanent free-form blocker.
Likewise the operator cannot select `complete_or_quiescent_blocked` as a
successful terminal condition: only authoritative completion succeeds, while
an authenticated all-blocked quiescent state fails closed. Plain and isolated
direct orchestrator invocation now resolve only the adjacent control directory
and repository root and work without ambient `PYTHONPATH`.
These policy edits invalidate all earlier conformance results and journals for
prospective dispatch. The formal operator now requires an enabled, independently
attested auxiliary configuration and supplies the scheduler's `n>=5` decision
to one digest-pinned fixed-argv driver; no driver CLI field can select the game,
effort, round, or specialization. Driver and probe stderr remain immutable
host-only audit bytes, while proposer-visible stderr is a fixed one-line
projection with a separate classification receipt. Driver-returned paths are
component-walked from a pinned assignment-root descriptor; canonical receipt
bytes supply digest and JSON verification in one stable read, and every
recovered phase reopens its journaled prerequisites before the next driver
call. Reopened stdout/stderr must equal the command runner's exact digest and
length, and a successful canonical response receives a durable raw-byte
recovery binding before any result is acted upon. Launch remains closed until
the immutable conformance run is regenerated under the new control hash and a
real driver/configuration/attestation has passed the genuine S06/S07 paths.
The registered unit owners exercise the executable contract but do not
fabricate those machine-observed scenario receipts.
The blocker slice is owned explicitly by
`blocker_claim_cross_product_is_noncounting`,
`authenticated_blocker_recovery_is_idempotent`, and
`host_terminal_parent_issues_authenticated_blocker`; prose or an unregistered
test cannot substitute for those exact owners. The S03 host-blocker owner
replays the exact parent into the host Arena session, crosses the actual
single-client Unix RPC `open -> close` boundary, requires the delivered close,
obtains the trusted `ArenaHostResult`, and lets the production backend derive
and HMAC-sign the blocker receipt. Its negative controls reject a hand-built
lookalike result, cross-attempt replay, and post-signature result mutation.
Launch remains disabled until the canonical suite consumes
genuine machine-observed S01–S12 receipt bodies rather than synthesizable PASS
metadata and proves the pinned live-model acquisition, real OCI/auth/protocol
attacks, real six-lane isolation, provider-window budget/drain/sidecars,
real-daemon SIGKILL matrix, thread/WIP rebinding, live candidate/replay,
crash-injected promotion, retained-evidence mutation matrix, contiguous
schema-v2 183-boundary release, and hermetic sealed-runtime execution. The
host-authenticated blocker contract is implemented and independently
registered, and its finite S03 machine-observation scenario now passes through
the real host Arena/RPC boundary. This closes the blocker-specific test
obligation, but does not substitute for the remaining production OCI,
live-model, six-lane, and frozen 183/183 release evidence.

#### 10.1.1 Unattended-autonomy release gate

The 2026-07-30 adversarial audit established that launch-closed scaffolding is
not yet an unattended operator. The following are hard release blockers, not
items an operator may resolve interactively after launch:

- Runner and scheduler must consume one canonical, typed lifecycle transition
  schema. Differential/model tests feed every legal and illegal event sequence
  to both reducers and require identical state projections and next decisions.
  Legacy phase-only attempt, retry, observation, collection, and teardown
  events are rejected in a contiguous-schema campaign.
- Public-observation registry writes, promotion, WIP replacement, canary
  reveal/cleanup, terminal audit publication, usage settlement, and every
  other multi-file authority transition require a write-ahead intent,
  idempotent restart reconciliation, and crash injection before and after each
  durable mutation. Terminal PASS or terminal incident evidence is fsynced
  before any live canary or escrow cleanup.
- One process-wide operator lease covers recovery, cycle execution, terminal
  audit, and cleanup. It binds owner PID plus process-start identity, has a
  heartbeat and bounded acquisition wait, rejects a live second owner, and
  supports authenticated stale-owner takeover without signalling a reused PID.
  A separate watchdog/service restarts a crashed operator from durable state.
  Its startup deadline is a one-way phase: once the exact operator readiness
  receipt is authenticated, that deadline is permanently retired for the
  process incarnation. A transient lease-read failure, `RELEASED` observation,
  heartbeat reconciliation, or delayed terminal stdout may not reset or
  reapply it. Production tests must keep a healthy operator alive beyond the
  expired startup interval while injecting each such transition, then prove
  complete terminal stdout/receipt collection; only the independent runtime
  liveness/containment deadline may subsequently terminate it.
- Primary and auxiliary infrastructure operations use consecutive,
  operation-specific retry counters. A successful operation durably resets its
  counter. Completed attempts classified as provider, authentication,
  containment, or other infrastructure failures enter the same typed
  consecutive-failure circuit instead of being immediately redispatched as a
  fresh solver attempt. Bounded exponential backoff, error classification,
  retry exhaustion, independent teardown/abort authority, and a sealed incident
  transition prevent an unavailable provider, stale credential source, broken
  driver, or broken backend from being hammered forever. These limits never
  cap clean solver no-progress search under `limit=None`.
- The operator has exact durable terminal projections for `COMPLETE`,
  `AUTHENTICATED_BLOCKED`, `FINITE_BUDGET_EXHAUSTED`,
  `JOURNAL_OR_STORAGE_EXHAUSTED`, and `OPERATOR_INCIDENT`. A state with no
  active work and no admissible dispatch may not sleep forever. Top-level
  exceptions publish a bounded incident receipt and leave restartable state
  before exit.
- The append-only journal supports authenticated rollover/checkpoint segments
  and always reserves terminal-evidence headroom. The operator enforces
  campaign, generation, transcript, controller-state, auxiliary-operation, and
  free-space quotas before dispatch. Near-limit and `ENOSPC` fault matrices
  must preserve reconstructible authority and end in a typed terminal state.
  Release remains closed: the implementation currently enforces only a
  24 MiB journal-prefix ceiling and 1 MiB dispatch reserve; it has no
  authenticated journal rollover/checkpoint segments, filesystem
  byte/inode/free-space admission, `ENOSPC` crash-recovery matrix, or reachable
  `JOURNAL_OR_STORAGE_EXHAUSTED` terminal transition.
- Polling is coalesced: unchanged primary and auxiliary status may not create
  an unbounded new operation directory, transcript, or journal event.
  Long-duration and full-183-level soak tests prove bounded journal, inode, and
  byte growth, including 300/360-minute turns and crash recovery near every
  cap.
- The production meta-proposer and side-expert path must be reachable from
  actual native output through a sealed semantic request and a host binding to
  authenticated public-observation receipt IDs. It has no interactive/user
  input path, no scheduler-field authority, and no promotion authority. An
  inability to produce a valid bounded handoff is ordinary non-authoritative
  no-progress, never an operator question. Production-path conformance must
  exercise this route from real native worker output; constructing the request
  dataclass directly in a reducer test is not reachability evidence.
- The unattended scheduler may also invoke the sealed meta-proposer as the
  final bounded diagnostic step for an unresolved typed infrastructure or
  scheduler circuit. Its input is limited to authenticated incident,
  preflight, public-observation, and policy receipts, and its output is only a
  quarantined recommendation. It cannot write scheduler fields, authorize a
  retry, alter runtime isolation, consume or replace WIP, allocate cost, or
  admit a candidate. A trusted deterministic validator may translate a valid
  recommendation only into a finite allowlisted remediation transition whose
  complete preconditions are independently observed. Ordinary supervision
  polls cannot manufacture another meta-proposer invocation. A successful
  remediation must pass a fresh real substrate probe and produce a one-shot
  durable controller authorization before the latch is cleared and the same
  authenticated frontier resumes; meta-proposer text alone never resumes it.
  If that bounded
  path emits no valid remediation or the mechanically applied remediation
  fails its fresh real probe, the operator publishes a durable
  `OPERATOR_INCIDENT` with `paused=true`, preserves all clean state, stops new
  launches, and surfaces a human-intervention request. This is the only
  conversational handoff: it occurs after the autonomous state machine has
  reached a typed paused terminal, never inside a solver or scheduler turn.
- One generated launch-authority receipt, not separately flippable booleans,
  binds every role image, exact runtime/interpreter and loaded-module origins,
  operator/service configuration, full-suite receipt, genuine S01--S12
  receipts, and the frozen release. No component-local readiness boolean
  confers authority; full launch authority is derived only by reopening those
  receipts. The auxiliary adapter remains separately disabled until a real
  production implementation and its admission evidence exist.

The canonical conformance entry must run both the exact one-owner invariant
registry and the complete explicit allowlist of contiguous component test
files. Every collected supporting test must pass; skip, xfail, collection
error, missing file, unexpected file, or unregistered test-file drift fails
the same aggregate. This condition exists because a curated owner aggregate
previously remained green while full runner and container-backend files had
stale, failing paths. Focused repair slices remain diagnostics only.

The unattended operator has no conversational fallback. Its complete
machine-owned response matrix is:

| Observation | Required autonomous response |
|---|---|
| another live owner holds the process-start-bound lease | refuse the second owner without mutating campaign state |
| the recorded owner is stale after a crash | reauthenticate the lease and resume the same durable identities |
| one infrastructure operation fails transiently | back off and retry under its consecutive typed circuit |
| an infrastructure/provider/authentication circuit exhausts | contain outstanding work and publish `OPERATOR_INCIDENT` |
| a clean proposer makes no progress | advance the complexity coordinate and continue the unlimited search policy |
| a meta-proposer/side expert emits no valid bounded output | quarantine it as non-authoritative no-progress and continue |
| a final circuit-recovery meta-proposer or its validated remediation fails | publish paused `OPERATOR_INCIDENT`, preserve clean state, and request human intervention |
| the finite allowance cannot reserve another attempt | publish `FINITE_BUDGET_EXHAUSTED` |
| journal, quota, inode, or free-space headroom is unsafe | preserve terminal headroom and publish `JOURNAL_OR_STORAGE_EXHAUSTED` |
| no work is active and no transition is admissible | prove `AUTHENTICATED_BLOCKED` or publish `OPERATOR_INCIDENT`; never sleep forever |
| the operator is killed or exits unexpectedly | the independent watchdog restarts it from durable state without a new identity |
| all 183 exact boundaries pass the terminal audits | publish `COMPLETE`, then perform transactional canary cleanup |

Likewise, the boolean maps in a launch-attestation document are schema
commitments, not proof merely because they say `true`. The runner must generate
the attestation from observed probes of the exact digest-pinned container and
retain their machine-readable outputs: effective mounts, PID namespace and
host-PID invisibility, effective user/capabilities/no-new-privileges,
environment-variable names, network routes, Docker-socket absence, read-only
root, per-attempt private tmpfs, writable-output allowlist, sibling-lane
process-isolation challenge, cross-lane temporary-file harvesting challenge,
and
timeout teardown challenge. The teardown challenge must include a proposer
helper that calls `setsid()` (or the platform equivalent) and attempts to
outlive the proposer process; destroying the attempt container must still
remove it without touching any sibling lane. `launch_preflight()` must verify
the hashes of those retained observations against the exact launch
specification. An
operator-authored or self-asserted all-true map is never launch authorization.

As of 2026-07-28, the executable contract also incorporates the defects found
during the live campaign: strict schema validation happens before checkpoint
field access; the complete per-game authoritative map is checked before every
frontier admission; coordinate replay tokens (`[6, x, y]`) and key actions use
the public Arena grammar, with scalar keys restricted to `1..5` or `7` and
ACTION6 `x,y` restricted to plain integers in `0..63`; candidate and checkpoint
roots must be regular
host-owned paths; boolean/integer JSON aliases, symlinks, non-regular exports,
stale parents, and undeclared files fail closed; promotion trees and the atomic
pointer are fsynced before acknowledgement. Provider terminal failures are
parsed from structured Codex error events only. Model capacity and rate
limiting are infrastructure outcomes that preserve WIP and are retried without
charging solver no-progress; they can never be inferred from words such as
“insufficient” in a solver brief or probe transcript and can never be reported
as quota exhaustion. Guard-lock contention, rate-limit-query transport errors,
CLI launch failures, and unknown nonzero CLI exits are likewise infrastructure.
The real runner rejects unrecognized command-line options and handles
`--help` before workspace creation, so a scheduler typo cannot silently fall
through to the default `wa30` target. Game, target level, proposer, model, and
soft turn allocation are mandatory scheduler fields; duplicate options are
rejected instead of using last-value-wins semantics. The contiguous runner must
not implement that allocation by forwarding it as a lower-layer hard
`--minutes` kill deadline: allocation expiry changes the lane to `DRAINING`,
while only the separate containment watchdog may stop the attempt container;
only explicit finite allowance/reserve and local run/token-cap messages are
cost stops. Fresh-prefix candidate discovery includes standard `final_path`
exports and descriptive candidate/replay filenames; recovery refuses to glue
such a path onto the stale checkpoint and instead requires independent
path-from-zero and source-from-zero replay. Same-frontier WIP novelty is always
measured from the immutable promoted parent. The launch attestation must
explicitly cover these regressions, and all 25 current campaign checkpoints
must satisfy the same checkpoint parser used by the contiguous supervisor.

ACTION6 validation is deliberately host-authoritative. The container-side Arena
view forwards malformed but JSON-serializable actions to the trusted host. The
host validates before budget or engine mutation, durably appends a rejected RPC
event, closes the transport, and withholds `ArenaHostResult` because there was
no authenticated delivered clean close. Candidate admission, ordinary
clean-no-progress classification, retry increments, and WIP restoration all
require that host result. Thus solver code cannot catch a local range exception
and continue into an apparently clean turn. Non-serializable calls also abort
the transport and cannot produce a clean-close receipt.
One registered differential conformance owner feeds the same valid and invalid
JSON tokens through acquisition, scorecard replay, supervision, proposer
publication, release, trusted RPC, and retained-evidence audit. Any acceptance
divergence—including accidental acceptance of bare `6`—blocks launch.

Frontier admission also binds the exact parent action count and remaining
600-action budget into the host receipt. A boundary at the action cap is marked
`fresh_prefix_required` before dispatch, so the scheduler cannot waste a turn
trying to extend a structurally exhausted replay. The underlying real harness
parses every checkpoint field and replay token before constructing accounting
state; malformed, path-only, partially written, boolean-aliased, or
extra-field proposer files remain untrusted recovery candidates rather than
raising later in scheduling. Durable recovery revalidates the complete
embedded receipt, exact path, clean promotion manifest, winning-source hash,
gate schema, and pre-receipt source-tree hash; updating a pointer hash cannot
launder a failed or incomplete receipt.
Every promotion receipt separately requires successful full path-from-zero and
full winning-source-from-zero replay. The winning-source snapshot hash must
also equal one of the container's declared, hash-checked exports, so stale host
source cannot be paired accidentally with a new candidate path. Promotion
copying preserves any raced-in symlink as a link and rejects the staged tree
before hashing or publication instead of following it into the host
filesystem. Critical JSON, receipt, checkpoint-hash, and durability reads use
no-follow descriptor opens and require a regular inode, so a final-component
link swap cannot redirect the supervisor after its preliminary path check.
Regular files must also have exactly one hard link: an aliased checkpoint,
candidate export, receipt, or promoted-source file is rejected so supposedly
immutable bytes cannot be changed through an untracked pathname.
An authenticated read is one descriptor transaction: capture `fstat`, enforce
the declared size bound, read exactly that many bytes from the same descriptor,
reject trailing bytes, capture `fstat` again, and reject any inode, mode, link,
size, mtime, or ctime change before parsing and hash admission. A cache entry
may bind only the signature of those exact parsed bytes; suffix reuse must
revalidate the immutable prefix and current descriptor identity. Thus even a
valid JSON A-to-B replacement during the read cannot pair A's parsed value with
B's metadata or survive as a stale authenticated cache entry.
The embedded host receipt is created with no-follow, exclusive-create
semantics after the staged source hash is checked; if any file or link appears
at that path during the validation/write window, publication fails and the
previously selected version remains untouched. Per-game and promotion locks
are also opened without following links and must be unaliased regular files, so an outside
hard-link alias cannot be truncated as part of lock acquisition.
The shared usage ledger and both of its finite-admission/append locks use the
same no-follow, regular-inode, single-link rule for reads and durable appends;
an aliased accounting path fails closed rather than corrupting usage records or
admitting a stale finite-budget turn.
The store independently re-parses the retained promotion manifests as the
exact contiguous chain `L1..current`, verifies each parent-manifest hash, and
binds the current manifest to the exact checkpoint; a receipt that merely says
`manifest_chain: true` cannot hide a dropped historical boundary.
Per-game and store locks use no-follow opens; unknown game names,
symlinked lock roots/files, lock-path traversal, and dangling selected-version
pointers fail before scheduling. A later-level promotion must use the exact
checkpoint file in the currently selected immutable version; a byte-identical
copy, proposer-controlled parent, or store-local synthetic zero seed is not an
admissible parent.
Clone probes are hypothesis-generation tools, not promotion evidence. Some
games can carry timing or selected-object phase that is not faithfully
preserved by a clone made at an arbitrary suffix boundary. Any result observed
only after a mid-route clone is unverified; a timing-sensitive branch must be
replayed continuously from a pristine level entry, and promotion still
requires the independent path-from-zero and source-from-zero host gates. The
contiguous-campaign launch attestation therefore includes
`clone_only_success_not_promotion_evidence`.
Before any proposer observation, the trusted Arena controller also audits the
reachable mutable object graphs and runs one deterministic same-action sibling
canary on a sacrificial replay of the exact exploration seed. Disjoint,
fully inspectable mutable graphs plus matching isolated trajectories select
`verified_isolated_clone`; a sibling leak or inconclusive canary selects
`fresh_process_per_candidate`, in which every branch between resets is created
in a distinct authenticated engine process and the prior process is proved
dead before replacement. The selected mode and canary/process evidence are
bound into the pre-launch preparation, candidate journal, and retained
promotion provenance. Solver prose, worker output, and candidate manifests
have no mode-selection field, and neither mode relaxes independent
path-from-zero and winning-source-from-zero promotion replay.
Unfiltered host process inspection is a forbidden execution surface. A process
query whose output is restricted to the turn's own named local probe worker is
recorded as informational operational monitoring, not gameplay taint; it never
authorizes source, environment, other-process, or host-path inspection.
This exception covers the equivalent filtered Linux/macOS `pgrep` spellings
(`-af` and `-fl`) only when the quoted pattern names one workspace-local
`*.py` worker. Broad process queries remain taint. Host `pkill`/process control
is not covered outside the isolated contiguous-campaign container because an
identically named worker in another concurrent lane could be affected.
The contiguous-campaign launch contract therefore requires a private PID
namespace, no visibility of host PIDs, and supervisor-owned timeout teardown.
Its adversarial suite must start identically named workers in sibling lanes and
prove that proposer-issued name-based process control and supervisor timeout
teardown cannot signal either sibling. Passing the transcript taint scan does
not substitute for this containment test; both gates are required.
The same suite must prove that crossing a 90-, 120-, or 180-minute soft
allocation while the proposer remains active emits no stop signal and leaves
the attempt container running in `DRAINING`. A separate injected containment
fault must exercise graceful container-local stop and, only after containment
grace expires, supervisor-owned teardown. Tests that use allocation expiry as
the timeout fault do not satisfy this requirement.
Container stdout/stderr and structured Codex events must stream to a
host-owned, no-follow, single-link file that is never mounted into the
container. Taint admission scans those immutable host-captured bytes after the
container and every descendant are gone; a proposer-exported, truncated, or
rewritten `proposer_last.log` can be retained for diagnosis but can never mask
the host transcript or become promotion evidence.
Promotion-chain audits validate manifest container types and restrict every
declared transcript or promoted-file path to its evidence directory. Absolute
paths, parent traversal, symlinked evidence, and wrong list/map types produce a
machine-readable integrity FAIL rather than an uncaught auditor exception.

This is a separate experimental lineage. Its purpose is to measure how far the
policy goes without interactive operator steering and to produce a cleaner
chronology—not to rewrite the evidence of the original admissible campaign.

### 10.1 Supervisor responsibilities

The supervisor must:

- discover the authoritative 25-game/183-level inventory;
- derive each game target from the downloaded public toolkit metadata, verify
  exactly 25 games and 183 total levels before dispatch, reject a checkpoint
  beyond its game's target, and never use comparator achievements as target
  counts;
- start each game from an empty candidate root;
- execute every proposer attempt inside a fresh, separately isolated container
  whose filesystem view contains only the public Arena interaction adapter, the
  current game's admissible clean lineage/WIP, and that attempt's scratch
  directory;
- schedule only the next unsolved level in a game;
- treat a zero-seed request as a distinct acquisition mode, never as a synonym
  for restarting a proposer process: before dispatch, compare the candidate
  artifact checkpoint with the scratch checkpoint and refuse any zero-seed run
  whose scratch state is absent, invalid, or behind a validated artifact;
- run disjoint games concurrently under per-game locks;
- implement medium → high → xhigh → optional max escalation;
- retain clean WIP and revisit cooled frontiers;
- treat `limit=None` uniformly as cost-control disabled;
- record heartbeats, PIDs, attempts, effort, wall bounds, and exit reasons;
- bind the parent action count and remaining 600-action budget to every
  admitted frontier; require fresh-prefix optimization rather than ordinary
  parent extension when no real-action budget remains;
- restart safely after host or process interruption;
- run taint and replay gates on every turn;
- run manifest, exact-boundary, and audit gates on every promotion;
- keep campaign checkpoints exclusively host/supervisor-owned; proposer
  containers receive only a read-only resume view and may export candidate
  paths under a separate schema, never replace `checkpoint.json`;
- on restart, preserve a clean unpromoted source attached to the exact validated
  parent before refreshing the host checkpoint, then replay that source before
  any new proposer call; never seed over a possible orphaned win;
- provide a documented fresh replay from level 1 for prefix optimization, so a
  proposer never needs to disable or rewrite the resume checkpoint; host
  recovery may replace a stale prefix only when both the exported full path
  and the current workspace source independently replay from level 1 through
  the requested frontier;
- validate checkpoint schemas before every read and treat a path-only or
  malformed file as an untrusted candidate artifact, never trusted accounting
  state and never an uncaught exception;
- compute each frontier's acquisition marginal against the exact validated
  promoted parent source, never against already accumulated same-frontier WIP;
- never import canonical solution source, old WIP, post-hoc labels, game code,
  or environment files into a proposer workspace;
- keep candidate artifacts isolated from the canonical release;
- stop only at 183/183 or a machine-readable unresolved blocker;
- never publish scorecards, push Git, or edit the PR without an explicit release
  step.

### 10.2 Production scheduler and audit path

The contiguous campaign must have an executable scheduler; the interactive
session is not its scheduler. The canonical policy implementation is
`arc/crack_lab/arc_agi3_contiguous_scheduler.py`, imported by the production
runner rather than reimplemented there. The same module exposes the only
read-only audit interface:

```bash
python3 arc/crack_lab/arc_agi3_contiguous_scheduler.py audit \
  --campaign-root <root> --output <new-receipt.json>
python3 arc/crack_lab/arc_agi3_contiguous_scheduler.py verify \
  --campaign-root <root> --receipt <existing-receipt.json>
```

`audit` replays the append-only campaign journal from genesis and derives every
expected scheduling decision from admitted state. `verify` reopens and rehashes
that same immutable receipt, journal prefix, scheduler/control files, inventory,
and evidence; it cannot accept caller-supplied decisions, outcomes, test names,
or PASS flags. The scheduler module, its focused adversarial tests, and the
resulting receipt schema are part of the unified conformance control digest,
S06/S07, the reproduction suite, and launch preflight.
The live runner must not rescan or rehash the complete immutable history on
every scheduler cycle. It may cache only a previously verified journal and
promotion prefix, keyed by its exact terminal sequence/hash and invalidated by
any prefix or selected-pointer mismatch; new suffixes are then verified before
use. The terminal read-only audit remains an exhaustive independent replay and
rehash of the complete history. Conformance includes a bounded-runtime
regression so correct scheduling cannot become operationally unusable through
superlinear full-tree revalidation.

The straight-line contiguous scheduler applies these exact principles:

- treat **scheduling supervision** as a receipt-reduction state machine,
  distinct from the explicitly scheduled supervisory-proposer role in section
  4.3.2. On every cycle the scheduler consumes any durable unacknowledged decision,
  polls live attempts, collects terminal evidence, proves teardown, classifies
  and settles each result, commits exact promotions, invalidates or admits
  auxiliary output, recovers reserved work, fills distinct primary frontiers,
  and only then fills eligible auxiliary capacity. This order is part of the
  scheduler policy hash;
- derive every terminal transition from the fixed host-authenticated result
  table: clean no-progress returns the lane to `READY`, increments `n`, and may
  retain only newly admitted same-frontier WIP; taint and infrastructure return
  it to `READY` without increment or WIP; a candidate enters `PROMOTING` but
  remains quarantined until all exact promotion gates pass; a genuine
  machine-readable blocker enters `BLOCKED`. Blocker authority is never
  free-form proposer or operator text: schema v1 has the finite generic code
  `arena_parent_terminal_before_target`, derived only when the trusted Arena
  host result proves that the exact K parent snapshot is terminal before K+1.
  Its receipt binds the campaign/generation/attempt/spec, game, frontier,
  parent checkpoint, K/K+1, Arena session receipt, parent path and snapshot
  hashes, complete host result, and a live-only host HMAC. Missing, unknown,
  malformed, unsigned, replayed, wrong-attempt, wrong-binding, wrong-frontier,
  wrong-target, wrong-host-result, or wrong-path claims settle as noncounting
  infrastructure and remain retryable. A closed blocker receipt is reopened
  and reauthenticated on every reducer pass, including cached crash recovery.
  Provider capacity, rate-limit,
  containment, and transport failures never increment `n`. An authenticated
  containment/teardown fault has terminal-class precedence over an embedded
  proposer payload: any candidate or `blocker` claim from that faulty terminal
  remains quarantined, and the lane returns through infrastructure recovery
  rather than entering `PROMOTING` or `BLOCKED`;
- schedule distinct eligible games first, with at most one active attempt and
  one writable lineage per game; unlike the exploratory campaign, it never
  creates duplicate same-game candidate branches;
- use a deterministic fair ordering derived from durable last-dispatch
  sequence and game ID, never process-list order, wall-clock races, comparator
  scores, post-hoc labels, or operator hints;
- enforce the versioned effort/allocation/WIP ladder from section 4.4. A
  promotion resets `no_progress` to zero; clean no-progress increments it;
  infrastructure, capacity/rate-limit, taint, and blocker outcomes do not.
  Treat this exact-frontier retry count as the single operational complexity
  coordinate for both primary effort and independent-sidecar eligibility;
- at `no_progress >= 9`, enter `LONG_COHERENCE`: use `gpt-5.6-sol/max` with a
  300-minute soft allocation and alternate coherence-reset versus eligible
  clean same-frontier WIP continuation across attempts. Do not create short
  duplicate work merely to fill capacity;
- after the first max turn has failed cleanly and `no_progress=5`, activate
  `OBSERVATION_ONLY_AUXILIARY_ANALYSIS`. Keep the max proposer uninterrupted
  and allocate an independent side expert in an immutable private copy only
  from an authenticated same-frontier native-proposer request or an admitted
  supervisory handoff. The deterministic scheduler selects capacity and a
  compatible request but authors no game-specific brief. The actual
  model/effort comes from the launch manifest, not from the role name. At
  `no_progress>=7`, at most two distinct request-bound expert assignments may
  use otherwise idle capacity. Contiguous experts include a recorded
  self-challenge; all outputs are quarantine-only until exact-parent
  provenance, taint, and fresh-replay admission;
- at the same complexity-gated hard-frontier stage, allow the scheduler to
  dispatch the manifest-pinned supervisory proposer only through section
  4.3.2's authenticated input/output contract. It may synthesize
  game-specific next tests from admitted native/side-expert evidence, but
  cannot choose scheduler fields, mutate a lineage, or promote; an admitted
  handoff remains an unverified proposer input until the native proposer
  reproduces it;
- treat lane capacity as a ceiling. When fewer distinct games are eligible, or
  only sequential descendants remain, leave excess capacity idle. A soft
  allocation expiry changes only that lane to `DRAINING`, starts no successor,
  and sends no signal to the live turn; the separate six-hour containment
  policy remains the hard safety boundary;
- bind `limit=None` to disabled cost cutoff uniformly, while still enforcing
  all correctness, liveness, transcript, replay, taint, and promotion gates.
  Under a finite limit, reserve each attempt's allowance atomically under the
  scheduler journal/store lock so `settled spend + all live reservations <=
  limit`; concurrent lanes never each receive the full remaining campaign
  balance. Settlement or release occurs exactly once on terminal recovery;
- revisit cooled/quarantined frontiers only through their declared clean
  eligibility transition; never restore tainted WIP or silently substitute a
  different parent, frontier, thread, state tree, or source lineage.

The reducer may use only the hash-chained journal, authoritative inventory,
exact parent/frontier hashes, authenticated terminal and usage receipts,
taint/replay/provenance/manifest receipts, reconstructed retry count, durable
fairness state, capacity, and budget. Game semantics, model final prose,
operator hints, remembered solutions, comparator results, post-hoc labels,
process-list order, and wall-clock races are not scheduler inputs. Here,
“rule” means a deterministic branch in the versioned scheduler transition
table over those authenticated inputs—for example: how to recover after a
crash between candidate publication and promotion; whether a provider timeout
increments the exact-frontier retry coordinate; when a clean terminal WIP
snapshot is admissible; or what happens when a promotion receipt fails replay
or taint validation. A human or LLM may diagnose that such a **generic
scheduler transition or integrity contract** is absent, ambiguous, or wrong,
but may not silently steer a live campaign. Such a correction requires a new
policy hash, adversarial regression coverage, and prospective dispatch under
that version. Separately, the explicit supervisory proposer may emit a
game-specific tactical handoff for the already scheduler-selected hard
frontier, but only through the journaled section 4.3.2 contract; it cannot
encode a manually selected game, effort, sidecar count, or promotion exception.

Before any external create/launch side effect, the scheduler appends a canonical
`SCHEDULER_DECISION` receipt that binds the journal head, inventory and policy
hashes, eligible-frontier snapshot, active/draining/blocked lanes, capacity,
cost-window identity, chosen game/target, retry count, effort, soft allocation,
seed/WIP/thread modes, selected WIP and parent hashes, deterministic ranking
key, and reservation identity. Reservation must consume that exact decision
once. A crash before reservation leaves a replayable unconsumed decision; a
crash after reservation recovers the same identity and may not redispatch it.
Auxiliary decisions analogously bind the frontier and parent hashes, triggering
clean-failure count, private-copy input manifest, active proposer assignments,
selected orthogonal complexity obligation, expert model/effort, usage
reservation, and the required Socratic/admission policy before any expert is
launched.

The scheduler audit fails on a missing, extra, duplicated, reordered, mutated,
or stale decision; capacity overflow; concurrent same-game attempts; wrong
inventory/target; forged retry count; promotion without reset; taint or
infrastructure counted as solver failure; cost cutoff under `limit=None`;
finite-limit under-accounting; stale or ineligible WIP; wrong effort/allocation;
finite-limit overbooking or duplicate settlement; short duplicate dispatch in
`LONG_COHERENCE`; premature, game-specific, duplicate, or unjournaled side
expert dispatch; an expert mutating live WIP; missing Socratic findings;
specialist output entering WIP/promotion without exact-parent replay, taint,
and provenance admission; an unjournaled or stale supervisory handoff,
supervisor access outside its allowlist, a supervisor-selected scheduler
field, a missing Socratic challenge, or a native solver relying on an
unreproduced handoff claim; a signal at soft expiry; or
non-idempotent crash recovery. Focused tests cover each inverse plus
multi-process journal races, but only the genuine S06/S07 scenarios and the
immutable audit receipt can admit launch.

### 10.3 Container and information-flow boundary

The compatibility campaign and the unattended contiguous campaign have two
different, explicitly named guarantees. The compatibility runner applies a
prospective behavioural proposer-boundary policy: it permits only the exact
documented raw-`gkm_arena` capability, scans every executable workspace source
and immutable command record, polls the live turn, and repeats the gate before
WIP retention, recovery, source execution, and promotion. Evidence records the
policy schema and source digest. A policy mismatch makes WIP forensic-only and
forces a clean reset. This is a fail-closed audit gate, not a proof of host
filesystem confinement; a turn that began before the policy was loaded can be
audited retrospectively but cannot be represented as prospectively confined.
No new compatibility turn may receive raw-Arena authority until its import
closure is purified and receipt-bound: offline import must not read the
repository `.env`, proposer/LLM modules must be lazy or absent, workspace module
shadows must be rejected, and the physical Arena module plus local runtime
closure must match the JIT evidence receipt. The already-live pre-policy LF52
child is allowed to drain under its unchanged runtime, but that host runtime is
not modified mid-turn and its parent remains unable to promote directly.

As of 2026-08-04, the preparatory half of that obligation is implemented by
`arc_agi3_arena_rpc_client.py` and
`arc_agi3_compatibility_arena_closure.py`. The extracted client preserves the
NumPy `uint8` frame surface, while descriptor-confined closure validation binds
both a deterministic content manifest and per-instance custody. The container
recipe installs the same client and a hash-pinned NumPy requirements lock; host
attestation binds the client, both workers, source schema, requirements lock,
and reviewed recipe. This remains deliberately non-authoritative. The ordinary
contiguous backend now consumes the closure and binds it to the exact
scheduler-selected host session, socket, token digest, relay, unstarted
container, image digest, and reviewed build controls. Both backend and runner
reopen that receipt before launch. The
closure records `launch_authorized=false`; the per-turn receipt records
`authority.launch_authority=false` and has no scheduling, mutation, WIP,
candidate, or promotion authority. The production
scenario-observer, global-release, and ordered-pilot gates remain separate and
fail closed, so this integration alone does not make the unattended campaign
launch-ready.
Any pinned worker image must be rebuilt from a host-created, receipt-bound
minimal context after hashing the complete contiguous Containerfile and
passing that digest through the no-default
`ARC_AGI3_CONTAINER_RECIPE_SHA256` build argument. The recipe installs it as
the `org.gkm.arc-agi3.container-recipe-sha256` image label; runtime image
inspection and the per-turn compatibility receipt independently require
equality with the current host recipe. Existing or newly built images without
the exact label are release-blocked before container creation. This
preparatory tranche has not built or pushed the production role image and has
not emitted the context-manifest, build-metadata, repository-digest provenance
receipt required for launch.

If a host process dies during closure publication and leaves the typed staging
path, that attempt root is never retried or reused. The backend preserves a
bounded descriptor-safe staging observation, the runner closes the old
PREPARED identity with zero solver/WIP/cost/promotion authority, and a fresh
attempt/generation becomes eligible only through the ordinary failure-circuit
deadline. Repeated quarantines trip the same bounded operator-incident gate;
ordinary closure errors with no retained staging remain normal retry/circuit
failures and cannot churn fresh identities.
The proposed bespoke legacy host-admission layer was rejected and removed;
the pre-policy LF52 output remains quarantine-only and any promoted result must
be reproduced through the ordinary fresh contiguous path.

The contiguous runner supplies the hard unattended boundary: a fresh
unprivileged, network-disabled container receives no repository, home, game
source, Docker socket, or sibling-lane mount and reaches the public Arena only
through the host RPC capability. The behavioural source/transcript gate stays
inside that design as defense in depth. Any boundary finding deterministically
returns a candidate-free and WIP-free `tainted` result, leaves the retry
coordinate unchanged, and revokes same-thread frontier context. Neither a
model report nor a scheduler heuristic may override this transition. Launch
authority remains closed until the production adapter-to-runner-to-scheduler
test, the easier-game pilots, and the complete conformance receipt all pass.

Use the strong part of baseline1's current operational design—a separate Docker
boundary—as defense in depth. Container isolation does not replace transcript
taint scanning, replay, exact-boundary capture, or promotion audits.

The host supervisor owns the canonical/candidate stores and performs admission
and promotion. For each proposer turn it constructs a minimal immutable input
bundle, starts a fresh container, and exports only the declared attempt outputs.
The export allowlist excludes supervisor checkpoint and accounting files.
Every role image is likewise built from a host-created immutable minimal build
context whose exact file manifest and hashes are receipt-bound. Building from
the repository root without an exact deny-all/allowlist context is forbidden:
canonical/candidate solvers, WIP, transcripts, game/environment source,
manuscript material, credentials, and unrelated project files must never be
sent to the image builder merely because they share the checkout.
Before parsing or hashing an export, the host enforces fixed file-count,
per-file byte, aggregate-byte, path-depth, and sparse-file limits so a malformed
candidate cannot turn verification into unbounded host work.
Candidate replay paths use a distinct filename/schema and are adopted only
after host replay. Prefix optimization uses the host-provided fresh-replay mode
against the exported solver source.
The container must have:

- a read-only root filesystem and dropped Linux capabilities;
- no host Docker socket, repository checkout, `.git`, home directory, arbitrary
  host bind mount, game/environment source, other game lineage, or prior
  superseded lineage;
- if proposer diff ergonomics require a local Git boundary, create fresh
  disposable metadata inside that attempt, force hooks off for every
  host-created baseline command, and never restore or promote `.git` bytes;
- no API-key files, shell history, cloud credentials, or inherited secrets;
- one read-only mount for the public interaction adapter and current clean
  lineage bundle, plus one game-scoped writable scratch/output mount;
- no general Internet access. Model transport, if the selected headless client
  technically requires it, must be host-mediated or restricted to the minimum
  provider endpoint and must not expose a general browsing channel;
- explicit CPU, memory, process, and wall-time limits, while preserving the
  campaign's effort/duration escalation semantics;
- an immutable record of image digest, entrypoint, mounts, environment-variable
  names, network policy, input hashes, output hashes, exit status, and resource
  usage.

The public Arena bridge should expose observations and legal actions rather than
mounting implementation files. Prefer a narrow host-mediated IPC endpoint; it
must reject private/runtime introspection and a second client. Promotion occurs
only after the host independently scans the transcript and exported filesystem,
replays from the correct parent, and verifies the manifest chain.
For a `K -> K+1` attempt, the trusted Arena host replays the admitted parent
path from zero and verifies the exact parent level before exposing any
observation. That seeded root is an immutable lineage baseline, not the
proposer's mutable probe. Proposer exploration uses bounded clones of the
seeded baseline; reset discards and recreates a clone from that baseline and
never calls reset on the real root. Parent replay actions count against the
declared 600-action campaign budget, while clone/probe accounting is recorded
separately. A zero-state Arena for a nonzero parent, an unseeded new thread, or
a root-reset error is an infrastructure failure before the model turn, never a
clean solver attempt.

The sole explicit exception is an exhausted parent whose exact path already
contains 600 actions. Its attempt binding uses
`exploration_mode=fresh_prefix`: the trusted host still replays and hash-binds
the complete parent as lineage evidence, but the separate immutable
exploration seed is the public zero state. Reset reclones that zero seed and
the candidate receives a fresh 600-action path budget. Receipts bind both seed
hashes, both action counts, and the mode; promotion still requires independent
path-from-zero and winning-source-from-zero replay through the exact next
level. Ordinary `continue_parent` dispatch is rejected for an exhausted parent,
and `fresh_prefix` is rejected for a non-exhausted or level-zero parent. Only
the original worker connection may deliver the authenticated Arena `close`;
host containment uses server shutdown and treats a disconnect or forced
shutdown as promotion-ineligible rather than synthesizing a close with the
host-held token.

The production orchestrator has four explicit container roles—model
controller, proposer, replay, and ephemeral probe—and one trusted host
supervisor/protocol mediator. The model-controller container runs the
digest/version-pinned Codex CLI and matching generated app-server JSON-RPC
schema. It has no repository, home, game, engine, solver, Docker-socket,
Arena, worker-bridge, or general host-filesystem mount. It runs nonroot with a
read-only root filesystem and receives only its private controller state and
stdio control channels. The trusted host mediator, not the model-bearing
container, owns authentication and worker-bridge authority and translates
validated app-server tool calls to that bridge. Provider traffic from the
controller passes through an enforcing egress proxy/firewall that denies
localhost, private/link-local/metadata ranges, arbitrary DNS/IP/domain access,
arbitrary or unallowlisted CONNECT destinations, and destination rebinding,
while exact provider traffic succeeds. A DNS-name allowlist in prose is not
evidence; the live probe must also test DNS rebinding and the post-resolution
IP policy. The proxy is built from its own maintained digest-bound container
recipe and guardian/policy source, both in the unified control manifest; an
external image label or shared network namespace alone is not enforcement or
reproduction evidence. Live conformance proves the required ChatGPT/Codex
traffic and rejects unallowlisted DNS names, resolved public IPs, literal IPs,
redirects, CONNECT targets, SNI mismatches, IPv6, loopback, private,
link-local, and metadata destinations. The controller cannot be created,
started, or reattached merely because the proxy container reports `Running`:
the proxy installs a default-deny policy before publishing a typed
host-captured readiness receipt, and the supervisor reopens that receipt and
passes the live policy probes first. Restart revalidates the same policy,
ruleset, image, and network-namespace bindings.
The three-hour scheduler allocation is a soft `DRAINING` boundary, not the
app-server read deadline: a separately bound six-hour containment ceiling
permits a live turn to finish naturally. Long-turn authentication refreshes
are sequential, same-account, redacted, and receipt-bound; the ceiling
deterministically permits seven refresh pairs (one initially stale token plus
hourly rotations), while any eighth request, overlap, lineage change, unchanged
access token, or credential byte in retained evidence fails closed.
The controller uses a neutral read-only working directory inside its isolated
container. A neutral directory, `sandbox=read-only`, and a self-reported empty
tool list are not isolation evidence: the observed OCI
mount/namespace/capability receipt and a genuine capability-denial probe must
prove that repository, home, game/engine source, host processes, arbitrary
filesystem paths, and non-provider network destinations are inaccessible even
if a built-in tool is unexpectedly exposed. Built-in shell, filesystem,
environment, process, configuration, provider-fallback, and every configured
MCP capability are also disabled at the app-server policy layer. The only
model-visible operations are the exact attempt-bound app-server dynamic tools.
The trusted host mediator allowlists only the thread/turn methods required by
the campaign; standalone unsandboxed app-server process, shell, filesystem,
and configuration methods fail closed if requested or observed.
Every thread also supplies versioned minimal `baseInstructions` and
`developerInstructions` rather than inheriting an evolving Codex default; the
exact prompt bytes and hashes are part of the thread/turn binding and unified
control digest. Model prose about which tools it can see is not evidence:
preflight binds `config/read` origins/effective values, `skills/list`,
an empty `mcpServerStatus/list`, the exact dynamic-tool declarations, and the
observed turn event/item types from the pinned app-server.
Every lane receives a supervisor-created minimal controller state root mounted
only into that lane's model-controller container and never into the proposer
container. It inherits no user
configuration, auth file, prior session, memory, plugin, skill, MCP server,
rule, hook, history, or project configuration. The supervisor supplies only
the pinned provider handle, generated protocol schema, exact model and effort,
and disabled built-in capability policy. No Arena or worker-bridge endpoint or
token enters this container.
For the subscription-backed contiguous campaign, the trusted supervisor
descriptor-reads the selected host credential store only as an authentication
source; it never copies or mounts that store into a lane. It injects the
external ChatGPT token tuple through the pinned app-server schema's
preflight-only login exchange, records only the method/direction/phase/count
and redacted hashes, and then makes every account/login method inadmissible for
the real turn. If that exact schema or exchange is unavailable, authentication
fails closed rather than falling back to ambient Codex state, an API key, or a
copied auth file. A descriptor-safe bounded inventory secret-, taint-, and
canary-scans the complete controller state tree after initial login, after
every rotation, and before any WIP/admission decision, together with all
streams, logs, outputs, transcripts, errors, receipts, and retained campaign
evidence. If the pinned app-server persists live credential bytes or any
repository/game/sibling canary into controller state, that state is promotion-
and WIP-inadmissible until it is destroyed or a protocol-compatible
non-persisting design is used; it is never waved through as an expected file.
The six independently generated canary values are stored crash-durably only in
a supervisor-owned host-only escrow outside every generation, attempt, mount,
input, output, state, and retained-evidence root. Its protection claim is
precisely trusted-host ownership, a mode-0700 parent, a mode-0400 regular file,
descriptor-safe reads, and demonstrated absence from every container mount and
proposer-visible path—not cryptographic sealing. A fresh supervisor process
may reopen it only to perform trusted scans. No terminal reveal or scan result
is admissible until exact controller/proposer identity-absence and teardown
proofs have been verified; arbitrary caller-supplied absence hashes cannot
satisfy that ordering. Before any model or proposer execution, the exact
ordered six commitment rows and host-only escrow hash/identity are bound into
the backend preparation, both launch attestations, and the durable `PREPARED`
journal event. Crash recovery and every terminal scan/reveal/audit must trace
to that prelaunch anchor; a substituted self-consistent escrow or commitment
set is rejected.
The formal operator itself generates the six independent 32-byte random
markers, observably plants one before launch in each declared sensitive class
(host repository, host home, host-only environment, authentication-source
boundary, controller control root, and sibling lane), and binds the exact
placement descriptors without disclosing values to either container. A
credential canary uses a dedicated colocated decoy and never mutates the live
credential bytes. Missing, duplicated, pre-cleaned, or substituted planting
blocks launch or terminal admission. Crash-safe cleanup happens only after the
anchored terminal scans and reveal audit finish.
Token bytes must also be absent from argv, environment values, bridge traffic,
and container input/output.
The state root is an explicit supervisor-owned path under the contiguous
campaign control root and is the controller's only writable mount. Container
environment construction removes ambient `HOME`, `CODEX_HOME`, XDG
state/config variables, and launcher state before setting only the declared
lane paths. There is no fallback to `~/.codex`, a login-shell profile, or
inherited desktop-session state.
An initially empty directory is not evidence of this property: Codex may create
state databases or materialize packaged system capabilities during startup.
The pre-turn probe therefore inventories the post-initialization state tree,
reads the effective feature/tool/MCP configuration, and proves that skills,
memories, goals, apps, plugins, hooks, browsing/computer/image tools,
multi-agent/tool-discovery surfaces, shell/unified execution, and every MCP
server are disabled or unavailable. Only the exact receipt-bound dynamic tools
may be model-visible. Any
unexpected materialized capability or model-visible tool blocks the turn.
Before admitting a lane, the supervisor starts the pinned controller image
under the same nonroot identity, mounts, namespaces, egress policy, environment
allowlist, and sandbox profile that the real turn will use, completes
initialization/auth/model/capability probes, and then destroys the probe
container. An unwritable state database, PATH-alias
failure, authentication-store error, or capability/config drift fails
preflight before a proposer turn and is classified as infrastructure without
charging no-progress or discarding clean WIP.
The app-server process identity, controller-container identity, controller
connection, thread, and active turn are attempt-bound and independently
contained. A minimal launch-time guardian is PID 1 and the direct parent of the
app-server; it retains exact-child ownership, contains that child and its
process group on supervisor-control EOF, and may emit diagnostic closeout
output. That child-shared output is never absence authority: the trusted host
must observe absence of the immutable full controller-container ID and its
cgroup/descendants through the container daemon. Recovery and teardown bind
that host-observed absence evidence, the pinned image digest, guardian
identity, labels, and attempt. Durable launch intent precedes container
creation. Recovery never uses an unauthenticated compare-then-`killpg`
sequence against an ownerless host PID; absent or ambiguous
guardian/container ownership fails closed without signalling an unrelated
process. Interrupting or tearing down one lane must leave every simultaneously
active sibling app-server/turn alive.

Each proposer thread receives exactly one attempt-scoped capability set: the
versioned app-server `dynamicTools` declaration bound into its thread/turn
receipt. The app-server emits a server request only as `item/tool/call`; the
trusted host mediator validates its request ID, tool schema, thread, turn,
attempt, phase, cardinality, and idempotency identity before translating it to
the private authenticated worker bridge. There are zero configured MCP
servers, and any MCP startup/tool event is a containment failure. The dynamic
tool/worker bridge exposes only:

- path-confined reads and writes inside that attempt's proposer-container
  workspace;
- hash-frozen, no-writeback command execution in the separately attested
  ephemeral probe container;
- the public Arena observation/action RPC for that attempt; and
- declared candidate/WIP export and progress operations.

The bridge uses descriptor-relative confinement (`openat2`/dirfd-equivalent),
rejects absolute paths, traversal, symlinks, hard-link aliases, device files,
unexpected methods, duplicate/non-idempotent tool-call identifiers, and any
second client. The model-controller process therefore never edits an attempt
workspace directly; the trusted host mediator alone translates validated
JSON-RPC calls, and all solver code and probes execute in the isolated,
network-disabled proposer container. No API key, ChatGPT token, Codex home,
provider credential, or general network route enters that container.
Model-authored probe code is itself adversarial and must execute behind an
additional child boundary: a working directory, `python -I`, and a sanitized
environment are not filesystem, process, or Unix-socket confinement. The
child runs under an identity and mount/descriptor policy that can access only
the declared workspace plus a mediated public Arena channel; it cannot read
immutable parent input, bridge/Arena token files, bridge sockets, `/proc`
details outside its own process group, container metadata, or export state.
Every child process and descendant is accounted for and gone before the bridge
returns. Direct token-file reads, socket connections, background forks,
script-path replacement, and export races are mandatory S03 attacks.
Disabling executable probes is the correct interim fail-closed state, but it
does not constitute the final successful campaign policy: the contiguous
solver needs a separately attested execution role for the probe/search work
used by the supervised campaign. Prefer a host-launched ephemeral probe
container over a hash-frozen declared workspace snapshot, with no proposer
input/export/bridge credential mounts, no writeback, bounded complete streams,
an optional dedicated fully logged public Arena session, and descendant-free
teardown. Its request, snapshot, runtime, result, and teardown hashes are
correlated to the originating dynamic tool call and covered by S03/S11.
No MCP or tool subprocess is spawned by the app-server. The dynamic-tool
handler is part of the digest-bound trusted controller and talks to the worker
only over the attempt's private socket using a nonce/sequence-bound MAC; the
raw capability never appears on the wire, argv, environment, or transcript.
Any app-server `process/spawn`, MCP-startup, or undeclared server-request event
is a pre-turn or active-turn containment failure. The live-model conformance
scenario must prove that host authentication succeeds while dynamic-tool
arguments/results, the worker bridge, probe role, and proposer container cannot
observe any provider credential or ambient Codex state.

Thread persistence is WIP state, not ambient authority. Every thread has an
immutable lane/frontier/parent/WIP lineage binding, and every turn has a current
attempt binding covering attempt ID, proposer image digest, immutable input
hash, workspace attestation, bridge identity/schema, Codex binary/schema
digest, model/effort, and declared outputs. A thread may resume only at the same
clean frontier after controlled rebinding: the supervisor quiesces the old
attempt, revokes its bridge, and issues a signed rebinding receipt containing
the prior binding hash, exact retained WIP and transcript-tail hashes, and the
new attempt/container/workspace/bridge/input hashes. The controller must
acknowledge that receipt before any model or tool operation. An unacknowledged,
ambiguous, concurrent, or still-live old binding fails closed; thread ID alone
is never sufficient. Any lineage mismatch starts a fresh thread or quarantines
the attempt rather than importing context. Host-captured app-server events and
the exact turn transcript are retained and taint-scanned with the container
outputs.
The contiguous taint policy is stricter than any public-submission reporting
policy: harness introspection (`inspect.getsource`, private adapter fields,
`dir()`/reflection against Arena or environment objects), forbidden
app-server methods/tools, game/environment-source paths, and unknown event
shapes are actionable failures even if another audit labels them
“informational.” The scanner parses each transcript schema explicitly, binds
complete byte coverage and hashes in its receipt, and fails closed on an
unreadable, malformed, truncated, skipped, or unrecognized record; it never
subtracts a shared `INFORMATIONAL_HITS` set.
The dispatch receipt records `thread_mode=new|resume`. Zero seed, WIP exclusion,
promotion/frontier change, taint, coherence reset, changed model capability, or
any binding mismatch requires `new`; only clean exact-same-frontier WIP may use
`resume`. Structured token, rate-limit, terminal-error, and turn-completion
events are correlated to the exact thread and turn, appended monotonically, and
reconciled before accounting or retry classification.

A proposer attempt receives only the immutable receipt-bound parent/eligible
same-frontier WIP bundle, a fresh writable workspace, the lane bridge, and the
host-mediated GPT-5.6-sol controller. It exports only declared candidate
source, WIP, transcript, and accounting artifacts. Its source manifest names a
small bounded flat reusable program/data set containing the required
`legs.py`, `players.py`, and `solve.py`; the host never promotes files merely
because their suffix resembles source. Every declared source byte is
taint-scanned, source-from-zero replayed, hash-bound, and included in
description-length accounting. Undeclared probes, notes, screenshots,
checkpoints, WIP, receipts, and worker artifacts cannot enter the next level's
source tree. A separate trusted replay
worker receives the declared source read-only and exercises it through a fresh
host-owned Arena RPC session. The replay worker cannot modify the proposer
output, and the proposer cannot write a host checkpoint or promotion receipt.
Before model initialization, the trusted input builder materializes the
declared reusable solver source and eligible same-frontier WIP files into the
fresh writable workspace through one receipt-bound seed operation. The worker
attests the initial workspace tree and byte identity against the parent/WIP
manifests before accepting a bridge client. The model never reads `/arc/input`
directly, but it must be able to inspect and extend those seeded files; an
empty workspace paired only with a resumed thread is not leg reuse or WIP
restoration. Crash recovery reruns this real seeding path idempotently and
proves the seed bytes unchanged.
A runner that merely executes a solver already supplied in its immutable input
is a replay/certification tool, not a contiguous cracking campaign, and cannot
satisfy launch conformance.
Assistant prose or code appearing only in final model text is never a candidate
artifact; candidate bytes must have been created inside the proposer container
and exported through the attested lane bridge.

Add adversarial isolation tests before launch: attempts to read repository/game
source, enumerate host paths or environment secrets, open a second Arena client,
use general network access, smuggle undeclared files through outputs, or mutate
the read-only lineage must fail and leave auditable evidence. The same suite
must attempt cross-thread bridge reuse between two live lanes, thread/turn
receipt substitution, built-in-tool re-enablement, a forbidden app-server
method, bridge path races, replay of a tool-call identifier, and model/provider
fallback. Pin the container image by digest and reproduce its build from
versioned Dockerfiles and lockfiles.
The clean-start scenario must use the genuine live model controller and prove
that it creates and exports a tiny winning solver through the lane bridge;
prerecorded text, a supplied solver, a mocked model, or replay-only execution
cannot satisfy that scenario.
The suite also makes the operator's real home and ambient Codex state
inaccessible and proves the explicit per-lane state root still starts cleanly.
Its inverse test makes that declared root unwritable and requires a
machine-readable pre-turn infrastructure failure with no model request,
workspace mutation, WIP loss, or fallback to ambient state.

Add fault-injection tests at every supervisor transition. At minimum they must
cover a malformed/path-only checkpoint, attempted proposer checkpoint
replacement, stale exact boundary, wrong authoritative target, process death
before and after replay, process death between manifest write and atomic
promotion, partial output copy, tainted WIP persistence, and host restart with a
live or stale lock. A child that has already produced a clean winning source
must be recoverable without a second proposer call: replay it from the verified
parent, capture the exact winning-source phase, rebuild host accounting, and
either complete promotion atomically or leave the prior artifact untouched.
The killed-child test must restart through the real parent-seeding path and
prove the candidate source is byte-identical before and after seed admission;
a test that bypasses seeding does not cover the historical failure.
Promotion acknowledgement is a separate crash boundary from pointer
replacement: if the atomic pointer became durable but the caller did not
receive success, restart must call `recover()`, revalidate the selected
version/receipt/checkpoint, and record the promotion exactly once. It must not
re-propose, republish the same level, or roll back a valid durable pointer.
Run the one canonical conformance command before every contiguous-campaign
launch. It may invoke focused unit tests internally, but its mandatory
cross-component scenarios run the actual supervisor, runner, Docker backend,
Arena RPC, worker, release gate, and atomic store together. At minimum those
scenarios cover clean start, soft-allocation drain, crash and restart through a
fresh backend adapter, orphan rehydration from Docker labels and host
attestation, ambiguous/tampered recovery failure, exact `K -> K+1` promotion,
promotion-acknowledgement loss, container teardown with no descendants,
transcript/taint/hash binding, and full-inventory release-receipt admission.
Mock-only or in-process-only coverage cannot satisfy these scenario IDs.
No operator-owned host driver, Docker CLI, or other infrastructure subprocess
may become an untracked detached session. Its controller-owned launch identity
must exist durably before execution, and the production guardian/service
manager must bind that identity to the exact PID, process start time, process
group, executable/argument digest, and owning attempt. After an operator crash,
restart must discover, authenticate, terminate, and reap each exact surviving
owned process before dispatch resumes. It may never kill by PID alone or adopt
an unbound process. The conformance matrix must cover crash before and after
child registration, reparenting, PID reuse, an already-exited child, concurrent
children, interrupted cleanup, and a real OS service-manager restart; no
descendant or stale durable child record may remain after PASS.
Prepared-attempt recovery validates the exact trusted-worker label/hash map
derived from the current control contract—including the Arena RPC, replay, and
proposer workers—rather than a separately hard-coded subset. A missing, extra,
renamed, duplicated, or hash-mismatched worker label makes the durable
attestation unrecoverable; the fresh-adapter regression must prove that the
unaltered complete map rehydrates exactly once after supervisor restart.
The unified control manifest includes this canonical plan, the pinned Codex
binary identity, generated app-server protocol schema, container build
recipe/image digests, supervisor, runner, backend, Arena bridge, app-server
transport, taint scanner, proposer/replay workers, workspace-probe recipe,
release gate, and the exact tests implementing S01–S12; changing any of them
invalidates the conformance receipt. The canonical driver has two modes:
`run` executes the genuine scenarios and derives an immutable receipt from
observed processes, files, and hashes, while `verify` reopens and rehashes that
same receipt and control/evidence set. Launch preflight calls `verify`; it
cannot manufacture a new PASS from caller-supplied test names or outcomes.
Both modes use a hash-pinned absolute interpreter and control root, a scrubbed
environment, disabled plugin autoload, and no inherited `PYTHONPATH` or working
directory authority. The interpreter binding also covers its exact venv/base
runtime identity, `pyvenv.cfg`, standard-library/native-extension manifest,
and pytest plus dependency manifest; hashing one launcher binary while loading
mutable unbound packages is insufficient. If the selected venv uses symlinks,
every link and resolved target is recorded and revalidated before and after
execution rather than merely relaxing the no-alias check.

The canonical end-to-end scenario registry is exactly:

- **S01 — live clean acquisition:** from a zero solver and a fresh attempt,
  the genuine pinned model in the digest-pinned controller image creates solver
  bytes inside the proposer container, exports them through the attested
  host-mediated bridge, and wins one easiest authoritative controlled frontier.
  Independent replay verifies the exact win. The receipt binds model, provider,
  effort, controller/proposer image digests, prompt bytes, source, path, and
  token usage. No supplied solver, prerecorded response, replay-only shortcut,
  mock model, or host-authored candidate can satisfy S01.
- **S02 — authentication and state isolation:** the host-mediated external
  ChatGPT-token exchange succeeds over the pinned login RPC under a fresh lane
  controller state root; no raw credential or bridge/Arena endpoint is mounted
  into the model-bearing container. Post-init config, skills, MCP, state-tree,
  OCI mount, egress, capability-denial, and credential-containment probes pass.
  Unique canaries in the host repository, home, environment, auth source,
  controller control root, and a sibling lane remain unreadable even when
  every available or accidentally exposed built-in tool is asked to read them.
  The prompt discloses only canary paths/names, never values; the trusted
  verifier scans for the independently generated secret values. A genuine
  crash-after-launch through fresh-adapter recovery proves that the host-only
  canary escrow survives, remains unmounted and unreadable to both containers,
  and cannot be consumed as terminal evidence before verified containment and
  teardown.
  All streams, outputs, and the descriptor-safely inventoried complete
  controller state tree remain secret-, taint-, and canary-free across initial
  authentication and at least one real token rotation; account/source mutation
  fails closed. The inverse
  unwritable-state-root case fails before any model request without ambient
  fallback, mutation, WIP loss, or secret leakage.
- **S03 — proposer containment:** the digest-pinned container proves its exact
  mounts, namespaces, cgroup/resource limits, network-none policy, nonroot
  identity, read-only lineage, private bridge, single-client Arena access,
  bounded complete terminal streams, declared exports, and descendant-free
  teardown. All specified source, host-path, environment, network, second
  client, output-smuggling, link/race, and tool-call-replay attacks fail.
- **S04 — app-server protocol firewall:** one observed
  `initialize -> initialized` lifecycle and the authentication probe precede
  the turn; every request/notification is admitted by exact
  direction/phase/method/cardinality rules. Unknown, malformed, duplicate,
  wrong-phase, built-in-tool, account-during-turn, process, filesystem,
  configuration, plugin, arbitrary-MCP, model/provider-fallback, truncated,
  and unconsumed events fail closed, while token bytes remain absent from all
  retained evidence. Controller mounts/environment contain no Arena or worker
  bridge socket/token, direct controller/sibling/second-client connections
  fail, lost-response mutation replay executes exactly once, and forged
  call/thread/turn/path/source bindings are rejected.
- **S05 — concurrent lane isolation:** the real six-lane capacity controller
  runs disjoint attempt, process, app-server/thread, bridge, container,
  workspace, accounting, and transcript identities. Cross-lane bridge/thread
  substitution and teardown of one lane cannot mutate, interrupt, or revoke
  any sibling.
- **S06 — budget, escalation, and draining:** `limit=None` disables cost
  cutoffs uniformly; finite-limit accounting is monotone, atomically reserves
  disjoint concurrent allowances without overbooking, and settles each
  reservation exactly once; adaptive medium-to-high-to-xhigh-to-max escalation
  follows the declared policy. The same journal-reconstructed exact-frontier
  clean-retry counter then opens one auxiliary sidecar at `n=5` and at most two
  at `n>=7`; forged counts, manual/game-specific triggers, non-clean increments,
  primary starvation, duplicate obligations, direct WIP mutation, and
  non-quarantined output all fail. Sidecar admission is unique per
  `(frontier identity, auxiliary round, specialization)`, including the
  profile-less diagnosis that opens the next round; two idle slots may never
  dispatch the same diagnosis concurrently. Promotion resets both projections and
  safely invalidates any concurrent old-frontier analysis without losing its
  containment, teardown, usage-settlement, or quarantine evidence. Soft
  allocation expiry enters `DRAINING`, starts no new turn, sends no interrupt
  to the live turn, and preserves a complete terminal result.
- **S07 — durable dispatch and backend recovery:** real-daemon supervisor
  `SIGKILL` probes run before create, after controller/proposer create but before
  ID acknowledgement, after app-server start, mid-turn, after a bridge mutation
  with its response lost, after candidate publication but before ACK, after
  replay, and immediately before/after the atomic canonical pointer update. A
  fresh supervisor recovers the same reserved identity through immutable full
  container IDs, observed Docker labels, guardian diagnostics, host-observed
  cgroup/descendant absence, and host attestation, and reaches either one exact
  once-only promotion or explicit
  quarantine with a byte-identical prior pointer. It leaves zero leaked
  containers, processes, or probes. Missing, ambiguous, stale, duplicated, or
  tampered recovery evidence fails closed without signalling an unrelated
  process.
- **S08 — transactional WIP and thread rebinding:** clean same-frontier WIP
  resumes only with an acknowledged binding receipt; zero seed, frontier or
  parent change, WIP exclusion, taint, capability drift, or coherence reset
  starts a new thread. Clean no-progress, capacity/rate-limit, and
  infrastructure outcomes preserve or atomically roll back the matching WIP
  and app-server state; tainted or mismatched context is quarantined and can
  never be restored.
- **S09 — exact boundary admission:** an isolated candidate created by S01's
  production path advances exactly `K -> K+1` under independent path and
  exact-winning-source replay from zero and the verified parent, with the
  pre-debrief source boundary captured. Path-only, malformed, exhausted,
  stale, wrong-target, multi-level, checkpoint-replacement, or
  host-generated candidates are rejected.
- **S10 — atomic promotion and acknowledgement recovery:** injected death
  before/after replay, evidence copy, manifest durability, pointer replacement,
  and caller acknowledgement either leaves the old version untouched or
  recovers and records the new version exactly once. Recovery never
  re-proposes, republishes, silently rolls back a durable pointer, or invents
  accounting fields.
- **S11 — complete evidence and taint binding:** immutable host-captured model,
  bridge, container, replay, and terminal-stream bytes have complete coverage
  receipts and bind their hashes, schemas, byte counts, candidate exports,
  usage, parent, image, protocol, and control contract. The same receipt binds
  a bounded descriptor-safe inventory and general taint/canary scan of the
  complete retained controller state, so a state-only source, environment,
  credential, or sibling leak cannot enter WIP. Every deletion, truncation,
  substitution, unknown event, forbidden method/path, state-only canary, or
  post-receipt mutation makes reverification fail.
- **S12 — authoritative freeze and release:** the production release gate
  admits exactly the authoritative 25 games and 183 sequential boundaries,
  rejects every missing/extra/duplicate/wrong-target boundary, freezes only
  schema-v2 exact evidence, and issues one independently reverifiable
  content-addressed release receipt and scorecard input.

Each scenario has one versioned owner and one machine-observed receipt. The
canonical conformance result is the ordered composition of S01–S12 and fails
unless every owner actually executed against the exact pinned control and
runtime digests. A caller-supplied status map, a synthetic PASS object, a
pytest collection/exit code, or focused-unit-test success cannot create launch
authority. Unit tests may prepare fault fixtures, but they are supporting
evidence and never substitutes for the twelve production-path receipts.

### 10.3.1 Independent fail-closed launch-clearance record

The 2026-07-28 independent review leaves launch authority disabled. It accepts
the atomic immutable-snapshot publisher as a tested inner primitive, not as an
end-to-end conformance result. Clearance requires all of the following in one
contiguous evidence lineage:

1. execute from one sealed control/runtime snapshot using an exact usable
   interpreter and dependency manifest, pinned images, a private temporary
   directory, and no ambient `PATH`, environment, or writable control-tree
   authority;
2. run the genuine S01–S12 production scenarios and retain one durable
   machine-observed receipt per scenario that reopens the actual process and
   container identities, attestations, transcripts, streams, files, and
   outputs; a digest computed only from expected invariant metadata and a
   `PASS` label is synthesizable status metadata, not a scenario receipt;
3. bind the six observed host canary placements and their receipt into every
   attempt before either model-bearing container starts, recover that same
   binding through a fresh adapter after a real crash, and reveal values only
   after independently re-observed containment and teardown;
4. use one canary authority and receipt lineage in production. A separate
   global planting whose values are passed to a per-attempt backend as an
   unbound static tuple is not sufficient;
5. make the egress proxy deny by default from process start, require an
   authenticated typed readiness receipt plus live allow/deny probes before
   controller start, and repeat those checks after recovery;
6. durably publish and fsync the terminal audit/receipt before idempotent,
   descriptor-safe marker and escrow cleanup; a crash in either direction must
   resume without losing terminal evidence or unlinking a substituted path;
7. complete the real-model clean win, real six-lane isolation, authenticated
   budget/escalation/draining checks, real-daemon `SIGKILL` recovery matrix,
   transactional WIP, exact admission and atomic-promotion fault matrix,
   mutation inverses, and independently reopened 25-game/183-level release;
8. perform a fresh immutable `run`, a separate reopening `verify`, and bind the
   verified scenario receipts, frozen release, and exact image before
   receipt-derived launch authority may be issued.

Every test, scenario, build, and runtime staging directory must be created
under its receipt-bound private temporary or campaign root, never under the
repository working tree or ambient current directory. The unified suite records
a pre/post workspace inventory and must prove cleanup after success, ordinary
failure, injected crash, and restart; any new untracked staging tree, retained
poll-invocation fan-out, stale socket, or orphaned child blocks launch.

Focused unit tests, mocked release verification, helper-only canary tests, and
hashes of asserted booleans remain diagnostics. They cannot satisfy an item
above or authorize the contiguous campaign.

### 10.4 Supervisor state machine

For each `(game, next_level)`:

```text
READY
  -> PROPOSING(effort, soft_allocation)
  -> DRAINING (if allocation expires while proposer is active; no signal)
  -> TERMINAL_RECEIPT + USAGE_SETTLEMENT
  -> MANDATORY_CONTAINER/CGROUP_TEARDOWN
  -> OUTPUT_SCHEMA + TAINT + PROVENANCE CHECK
  -> FRESH_PUBLIC_REPLAY
  -> EXACT_BOUNDARY + MANIFEST CHECK
  -> ATOMIC_PROMOTE + POST_AUDIT
  -> READY(next_level)

clean failure -> admit same-frontier WIP -> n := n + 1 -> READY
taint failure -> discard attempt lineage -> n unchanged -> READY
infrastructure failure -> discard attempt WIP -> n unchanged -> READY
candidate -> quarantine -> exact promotion gates -> PROMOTE, or discard
  candidate + retain eligible prior WIP + n unchanged -> READY
genuine host code + receipt + live HMAC -> BLOCKED
unknown/malformed/unsigned/replayed/wrong-bound blocker -> infrastructure
containment fault -> container-local stop -> grace -> container/cgroup teardown
```

There is no coverage-threshold or “looks promising” transition in the
contiguous policy. Broad parallelism contracts naturally as games finish:
distinct eligible games are filled first, one writable primary lineage per
game; when only sequential descendants remain, spare primary capacity stays
idle or is used only by retry-count-eligible quarantine-only sidecars.
Unlimited budget semantics remove cost cutoff, not receipt, taint, replay,
promotion, liveness, or blocker gates.

Before every scheduling cycle, the global inventory gate must pass:

```text
len(per_game_targets) == 25
sum(per_game_targets.values()) == 183
0 <= canonical_reached[game] <= per_game_targets[game]
next_level is schedulable iff canonical_reached[game] < per_game_targets[game]
```

Failure of any condition is an infrastructure stop, not a solver attempt.
The executable gate must compare the complete per-game map with the
authoritative metadata, not merely the game count and 183-level sum. Each
dispatch must pass `admit_next_frontier()`, which rejects completed games,
unknown games, malformed host checkpoints, and anything other than exactly
`reached + 1`. In particular, `re86` ends at L8 and an L9 request is a
supervisor-contract failure before a proposer process exists.
As defense in depth, the underlying real-run entry point in `gkm_legs.py`
independently reads the selected game's public metadata before creating a
workspace and rejects any `--max-level` outside `1..authoritative_target`.
Therefore a scheduler regression or malformed manual command cannot bypass the
supervisor admission layer to launch a nonexistent level.

### 10.5 Contiguous-campaign acceptance

The supervisor run gets its own:

- artifact root;
- usage ledger;
- append-only scheduler decisions and a replay-verified scheduler audit
  receipt;
- attempt and promotion manifests;
- exact-boundary audit;
- sawtooth analysis;
- ONLINE shakedown card;
- optional Competition card if it reaches 183/183.

Launch acceptance additionally requires one fresh canonical conformance result
with every registered invariant passing. This includes a deliberate
killed-child recovery that produces the same exact promotion as the
uninterrupted control and a deliberate killed-promotion recovery proving that
rollback leaves the previous artifact byte-identical. Individual component
test reports are supporting diagnostics only and cannot be substituted for the
unified result.

The full 25-game contiguous reacquisition is not the first production use of
the operator. Before it can receive full-campaign launch authority, the exact
frozen image, scheduler, meta-proposer path, journal reducer, audit suite, and
watchdog must independently reacquire the two predeclared pilot games
**`ft09` followed by `lp85`** from empty artifact, WIP, controller-state, and
scheduler roots. These games were frozen before pilot launch because their
current canonical paths are short (80 and 93 actions respectively), while
together they exercise both multi-level automatic continuation and substantial
cross-level leg reuse; they may not be replaced after observing a pilot
outcome. The pilot manifest
is fixed before either run from public inventory and canonical clean-retry
complexity evidence; pilot selection cannot become a hidden game-specific
scheduler branch. Each pilot must solve every authoritative level of its game,
exercise at least one clean continuation/restart boundary, terminate without
operator steering, and pass replay, action-protocol, taint, exact-boundary,
hash, manifest, usage, containment, terminal-retention, and journal-replay
audits. At least one pilot must exercise the sealed meta-proposer through its
real production entry and mechanically validated handoff. Pilot artifacts
remain a separate noncanonical lineage. Only one host-authenticated pilot PASS
receipt binding both complete runs may unlock the full contiguous campaign; a
pilot pause or failure leaves full launch closed and requests human review.

Pilot admission must not require the launch authority it exists to establish.
The first pilot is authorized only by a fresh real prelaunch control-suite PASS
with `launch_authority=false` plus independently constructed and verified
backend, guardian, controller-substrate preflight, image, policy, and ledger
receipts. Those receipts authorize only the frozen two-pilot scope. The
authenticated receipt binding both completed pilot PASS results is the sole
transition that may mint full-campaign `launch_authority=true`; any dependency
from pilot admission back to that terminal authority is a release-blocking
cycle.

Promotion storage must use `VersionedArtifactStore`: versions are immutable and
the selected `current.json` pointer is replaced atomically. A crash before the
pointer swap leaves the prior version selected; a crash after it leaves a
complete hash-verified new version selected. Direct multi-file copying into the
currently selected artifact is forbidden.

`VersionedArtifactStore.publish()` is itself an admission boundary, not a raw
directory-copy API. It requires the validated candidate-output root, the exact
`FrontierAdmission`, and a host-owned promotion receipt binding the parent
checkpoint, candidate manifest, rebuilt exact checkpoint/path, winning-source
snapshot, promotion manifest, taint/replay/check results, and complete source
tree by SHA-256. Publication also requires the regular host-owned parent
checkpoint path: under the store lock, the store reruns authoritative
next-frontier admission from those parent bytes and requires the resulting
admission to equal the scheduler-supplied object. A scheduler therefore cannot
fabricate an admission, skip a level, select a nonexistent target, or publish
against a stale parent. Store, candidate-output, and promotion-source roots
must be pairwise disjoint. The store independently revalidates all admitted
bytes, requires each new version to extend the currently selected checkpoint
by exactly one level, embeds both the receipt and the host-captured candidate
manifest in the immutable level-evidence directory, and
serializes publication and recovery under a store lock. Recovery re-parses the
selected checkpoint against the authoritative per-game target rather than
trusting pointer hashes alone. Omitting or failing any gate cannot publish.

Every container export must pass `validate_candidate_output()`. In particular,
`checkpoint.json`, accounting, and promotion manifests are forbidden outputs;
all other files must be declared by relative path and SHA-256, and symlinks,
undeclared files, wrong-frontier outputs, and stale-parent outputs are fatal.
Preserved evidence bundles additionally pass
`arc_agi3_exact_bundle.verify_manifest()`, which rejects changed, missing,
extra, or stale files and extra directories. Manifest publication is an
fsynced atomic replacement, so a crash before publication leaves no visible
partial manifest. The canonical conformance registry owns an adversarial test
covering all five cases; a focused test that is not registered in the unified
suite is not launch authority.
The winning source named by a promotion receipt must be byte-identical to a
declared candidate export, and both that source and the candidate's exact path
must have independent host replay-from-zero PASS results in the receipt.
Before WIP retention, candidate replay, or promotion, the shared source schema
also parses the complete flat source set and closes every import over declared
local `.py` stems, the standard library, or pinned NumPy 2.4.4. Relative,
dotted-local, `arc.crack_lab`, `environment_files`, and other undeclared
ambient imports fail closed; the backend, orchestrator, and publisher all
reapply the same validator.
Every host checkpoint must pass `load_trusted_checkpoint()` before scheduling.
The pinned-image conformance run and adversarial isolation probes must produce
one result artifact accepted by `validate_launch_attestation()` and
`launch_preflight()`; absent or failed scenarios block launch. Its invariant
map is an exact schema: missing, skipped, duplicated, failed, extra, or obsolete
identifiers all block launch, as does a symlinked result or any digest mismatch.

It replaces the canonical release only if it independently reaches 183/183,
passes every release gate, and is explicitly selected. Otherwise it remains a
valuable automated-policy benchmark beside the canonical campaign.

## 11. Definition of done

The overall program is complete when:

- [ ] canonical coverage is 183/183;
- [ ] every canonical level replays from its correct parent;
- [ ] every canonical level has uniform exact-boundary evidence;
- [ ] `wa30` has a clean complete reacquisition lineage;
- [ ] `tn36` has a clean complete 7/7 reacquisition replacing the lineage
      affected by the historical out-of-range L1 probes;
- [ ] the canonical ACTION6 audit has no token or recorded acquisition
      incident, with its legacy call-log limitation disclosed;
- [ ] canonical taint and promotion-chain audits pass;
- [ ] the reproduction suite independently regenerates the taint audit (and,
      for the automated lineage, verifies every container execution manifest);
- [ ] the complete sawtooth/reuse table is regenerated without inferred gaps;
- [ ] the 25-game ONLINE shakedown passes;
- [ ] the definitive Competition card is closed and verified;
- [x] the manuscript, figures, tables, checksums, and PDF reproduce;
- [x] the manuscript retains Alexander Kolpakov as sole author, while the
      Community Leaderboard metadata lists OpenAI GPT-5.6 (the model, not
      OpenAI as a company) as a submission author;
- [x] GKM/OPINE/Retrodict/baseline1 statistics use one documented schema;
- [ ] PR #37 title, body, YAML, README, links, and scorecard are updated;
- [ ] the final v3 PR head passes both the pinned local upstream validator and
      the remotely reopened exact-head `Validate Submission` workflow/check;
- [x] the author explicitly reordered and released the conservative
      manuscript/downstream phase before the final 183/183 remainder;
- [ ] the production contiguous scheduler, deterministic journal replay
      auditor, immutable audit receipt, and genuine S06/S07 adversary paths
      pass through the unified reproduction/conformance suite;
- [ ] the complexity-driven independent side-expert escalation has no
      game-specific scheduler branches; every semantic brief is bound to an
      authenticated native-proposer request or admitted supervisory handoff,
      and its private-copy, orthogonal-assignment, Socratic-pass, usage,
      quarantine, taint, provenance, request-substitution, and fresh-replay
      inverse tests pass in that same suite;
- [ ] the complexity-triggered supervisory proposer has isolated,
      manifest-pinned inputs; schema-validated, Socratically challenged,
      hash-bound handoffs; no scheduler/mutation/promotion authority; native
      public-observation reproduction; complete inverse tests; and explicit
      treatment in the manuscript and downstream reproduction/method docs;
- [ ] journal replay proves that one exact-frontier clean-retry count selects
      both `medium -> high -> xhigh -> max` and
      `max -> max + sidecar(s)`, with promotion reset and every non-solver
      outcome excluded from the count;
- [ ] the policy-hashed receipt reducer—not free-form interactive intervention—
      proves the complete poll/collect/teardown/classify/admit/promote/refill
      cycle, while any semantic supervision enters only through the explicit
      supervisory-proposer contract; adversarial tests reject every other
      operator steering path and noncanonical terminal transition;
- [ ] the automated contiguous orchestrator is implemented, tested, and
      has cleanly completed at least two empty-root easy-game pilot
      reacquisitions through the exact frozen production stack, including one
      real sealed meta-proposer handoff, before launch of its
      container-isolated full contiguous rerun;
- [ ] failed final meta-proposer recovery produces a durable paused
      `OPERATOR_INCIDENT` and human-intervention request without mutating clean
      WIP, solver complexity, cost, or promotion authority;
- [ ] the final report distinguishes exploratory canonical, diagnostic, and
      contiguous lineages.

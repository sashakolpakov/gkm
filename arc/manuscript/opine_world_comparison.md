# ARC-AGI-3 Comparator Audit: Performance, Provenance, and Solver Growth

Audit date: 2026-07-26.

Primary sources:

- [ARC-AGI-3 technical report](https://arxiv.org/abs/2603.24621),
  [verified-testing policy](https://arcprize.org/policy), and
  [community-leaderboard notice](https://arcprize.org/leaderboard/community)
- [Graph-Based Exploration paper](https://arxiv.org/abs/2512.24156) and
  [code](https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore)
- [baseline1 paper 1](https://arxiv.org/abs/2605.05138),
  [paper 2](https://arxiv.org/abs/2607.15439),
  [code](https://github.com/astroseger/arc-3-agents-baseline1), and
  [paper-2 run archive](https://doi.org/10.5281/zenodo.21412274)
- [OPINE-World paper](https://arxiv.org/abs/2607.01531) and
  [code and artifacts](https://github.com/david-courtis/opine-world)
- [Retrodict code](https://github.com/ryanbbrown/Retrodict),
  [v2 release](https://github.com/ryanbbrown/Retrodict/releases/tag/v2.0), and
  [reported scorecard](https://arcprize.org/scorecards/9c403765-db5b-40b1-beab-6fa3f40119b0)

Measurements below distinguish a reported score, action replay, epistemically
clean generation of the action policy, and evidence of reusable solver growth.
Those are four different claims.

## Bottom line

1. **No public-demo result establishes that ARC-AGI-3 has been solved in the
   benchmark-generalization sense.** ARC Prize places harness results on a
   self-reported community track and reserves semi-private/private evaluation
   for held-out claims. baseline1's latest paper correctly calls its own
   183/183 result “public-set saturation.”
2. **Retrodict v2 deliberately reuses experience.** Its selected `sk48` run
   reads an earlier same-game run's observed boards, level boundaries, and
   action sequences. Under this comparison's admissibility rule, those are the
   system's own observations and proposer trajectory—not privileged game code,
   hidden runtime state, or source—and therefore are admissible reuse, not
   taint. The implementation detail is relevant to its reuse mechanism, not an
   adverse practical-audit verdict.
3. **baseline1's current result has the strongest practical audit story of the
   near-ceiling public results inspected here.** The paper discloses and
   excludes earlier runs from vulnerable harnesses, documents the hardened
   Docker boundary, publishes a large checksummed run archive, and expressly
   retains the model-contamination and held-out-evaluation caveats. It is not a
   private-set result, and its latest artifacts have not yet been reduced to
   exact winning-entry-point sawtooth measurements.
4. **OPINE-World has broad cross-level data reuse but only two certified
   executable sawtooth witnesses.** Its frames-only Bayesian diagnostic
   recomputes Beta posteriors from the cumulative cross-level replay buffer and
   exposes Thompson-ranked uncertain dynamics to Claude as exploration
   guidance. The engine, shared world model, synthesis handoffs, and analyzer
   context also persist. That is substantial reuse, but it is not yet the
   narrow executable claim: of 14 sharp engine-code drops, 12 accompany
   transient analyzer wins whose executable policies were not retained. Only
   `lp85` L4 and `tu93` L3 couple a sharp drop to a synthesized winning planner
   that calls unchanged earlier engine definitions.
5. **The graph explorer is a valid historical private-preview comparator, not
   a current 25-public-game solve.** It officially solved 12/25 private preview
   levels; after correcting a disclosed reset bug, five reruns ranged from 14
   to 19 with median 17. Its 4,000-step point solves two `ls20` levels, and its
   paper reports `ls20` level 3+ as intractable for exhaustive exploration.

## Audit rules

### Practical admissibility

A run is practically admissible only when:

- the evaluated task set and score definition are explicit;
- the exact selected run and all restarts/discards are identified;
- actions and relevant policy-generation artifacts are retained;
- the agent cannot read game source, another system's prior solution, unscored
  simulator state, the web, credentials, or undeclared benchmark descriptions;
  its own admitted observations and proposer/action trajectory may persist;
- a third party can reproduce the reported boundary from immutable or
  checksummed artifacts.

An official scorecard is strong evidence that a recorded action sequence was
accepted and scored. It is not, by itself, an audit of what information produced
that sequence.

### Solver-growth evidence

For a solved level, the checkpoint is the retained state that actually cleared
that level. Interim synthesis edits and post-win cleanup are excluded. Keep
separate:

1. cumulative executable description at the solved checkpoint;
2. conditional novelty since the preceding solved checkpoint;
3. an operational witness that the later solve invoked earlier machinery; and
4. descriptive memory such as a playbook, transcript, or provider context.

The narrow Kolmogorov--Schmidhuber screen requires all four of the following at
an adjacent level boundary:

1. an exact executable state \(P_{k-1}\) at the preceding win and \(P_k\) at
   the current win, without post-win edits substituted for either;
2. a conditional description \(M_k\), obtained by normalizing every top-level
   AST statement in \(P_k\), removing literal statements already in
   \(P_{k-1}\), serializing canonically, and compressing with zlib-9;
3. a sharp reduction \(M_k\le M_{k-1}/2\); and
4. a winning entry point at level \(k\) that actually calls a named definition
   whose normalized AST is unchanged from the preceding winning checkpoint.

Condition 4 without condition 3 is a hard executable-reuse witness, but not a
coupled sawtooth witness. Persistent observations, posterior counts, transcripts,
notes, and proposer/action trajectories are broad reuse and can be valuable; they
do not, by themselves, establish this narrower executable path.

This is a computable conditional-description proxy, not machine-independent
Kolmogorov complexity. The absence of explicit Kolmogorov, Schmidhuber,
PowerPlay, or Gödel-machine terminology is **not** a validity failure. At most,
it is a clue to inspect whether the system measures compression progress,
retains acquired skills, and operationally reuses them. OPINE in fact cites
Schmidhuber on curiosity; baseline1 motivates simplification partly through
MDL. Terminology is not the test.

## Comparator matrix

| System and reported result | Practical audit verdict | Sawtooth and reuse verdict |
| --- | --- | --- |
| ARC-AGI-3 direct-model release baselines | Official held-out calibration, but historically weak and not artifact-equivalent to agent harnesses. The technical report explicitly separates verified model testing from self-reported public harness research. | No persistent external solver is present, so no artifact-level solver-growth curve is available. |
| Graph explorer, preview challenge: official 12/25 private levels; post-fix median 17/25 | Open code and disclosed reset bug. The post-fix number is an author rerun, not the official private score. It concerns the six-game preview generation, not the current 25-game public set. | Graph state is reused within a level, but the method is explicitly non-learning and supplies no cross-level executable skill-growth or conditional-description sawtooth. This is a scope distinction, not taint. |
| baseline1, original GPT-5.5 xHigh artifact set: 174 clears in the selected bundle | Exact client/guarded-executor outputs and retained workspaces permit solve-boundary reconstruction. The paper documents earlier leakage incidents and excludes the vulnerable results. | In the 18 exact adjacent winning transitions, every winning command is a fresh literal action program; none calls a retained world-model definition. Five of eight comparable marginals decrease, none sharply. This verdict applies to the audited GPT-5.5 lineage, not automatically to v1.6. |
| baseline1 v1.6, GPT-5.6-sol: verification agent 183/183 at xHigh and max, 98.97 and 98.77 RHAE; textual agent 183/183 at max, 95.97 | Strong disclosure and archive. The current harness hides the true game ID, disables general web access, prevents a second client, and uses Docker isolation. However, it is one run per game on the public set, has no variance estimate, and the post-release model may recognize game visuals. No semi-private/private result is reported. | The fixed executable model and replay verifier can support reuse, but the paper reports aggregate performance/ablations, not an exact solve-boundary conditional-description trajectory. A complete v1.6 sawtooth and direct-call audit remains open; “verification” is not itself proof of shrinking acquisition cost. |
| OPINE-World: headline 20/25 games, 160/183 levels, 78.4 RHAE | Reported configuration is frames-only under Docker isolation, without mounted game source. The paper reports an audit of transcripts for game-source/web/credential access. It is one run per game. The released main logs yield 153 positive rewards: `s5i5` lacks a summary, and the `sb26` and `tr87` summary/main-log boundaries differ by one in opposite directions. Thus the exact main-log artifact does not independently reconstruct 160. | The cumulative replay buffer, Beta/Thompson exploration diagnostic, engine, shared world model, handoffs, and analyzer context provide broad cross-level reuse. Narrow executable test: 49/115 comparable engine marginals decrease and 14 sharply; 12 sharp drops accompany unretained transient-analyzer policies. Two (`lp85` L4 and `tu93` L3) couple a sharp drop to a synthesized winning planner that invokes unchanged earlier definitions. |
| Retrodict v1 audited artifact set: 170 solved memory checkpoints | Rich transcripts and memory-edit traces, but the selected v1 artifacts expose memory checkpoints rather than executable winning entry points. | Replaying writes yields 76 between-level playbook contractions. Twenty-three of 25 runs have no substantive scratch Python at solve checkpoints; the two exceptions show three expansions and no contraction. This is real descriptive-memory reuse, not an executable solver sawtooth. |
| Retrodict v2: headline 25/25, 183/183, 99.86 RHAE, 7,703 actions | No practical taint is established by the inspected same-game transcript reuse. The selected `sk48` run reads an earlier run's observations and proposer/action trajectory; these are admissible experience under the stated rule, not privileged code or hidden environment state. | Retrodict deliberately carries observations and a playbook across context resets, so “no reuse” would be false. The release, however, does not expose a per-level executable winning entry point with which to couple a conditional-description drop to an actual retained call. |
| GKM active-campaign audit snapshot at 2026-07-26 21:02 CEST | Promoted source, fresh replay, resource ledger, and source lineage are retained. The snapshot has 126 exact winning checkpoints and must not be silently mixed with later campaign state. | 99 exact adjacent transitions; 50/97 comparable marginals decrease and 14 are sharp. Thirty-eight wins call unchanged legs; nine are sharp coupled witnesses. |

## Practical findings in detail

### Retrodict v2: admissible experience reuse, not demonstrated taint

The v2 release archive exposes how its selected `sk48` run reused experience.
The audited downloads were:

```text
39d956bb3550a43adb13562de479501c8c26a685ef4762f0d43758c05509265e  final-runs.tar.gz (v1)
efe30976e86affb2987b114f4e70ce94889cdd3a4829db31fbcb86927115ef61  release-runs.tar.gz (v2)
```

Retrodict maps selected game `sk48` to run `20260718-184901`. Fourteen Python
tool calls in its released traces mention the earlier run
`20260714-072903`. The v1 archive independently contains that exact path and a
612-step log ending with 8 levels completed and `WIN`.

Examples from the selected v2 trace:

- `a25965e5cb29414a967b9c9fe26bf660.jsonl` loads both the current log and
  `../../20260714-072903/workspace/log.txt`, then aligns current boards to old
  step 305.
- `b5157c5f2cfe475da5cc900133afa55f.jsonl` loads the old 612-step run, prints
  every level-transition index, and dumps the earlier level-5 action sequence.
- `2c2c826dfc884443a92cc59d017d6ac6.jsonl` aligns the current level-5 entry
  board to old step 330 and prints the earlier level-6 action sequence.

These accesses succeed in the tool results. They show reuse of the system's
own observed frames, boundaries, and proposer/action history. Under the
admissibility rule used here, that is legitimate accumulated experience. The
trace does not show successful access to game implementation code, hidden
simulator state, or another privileged environment channel that helped solve
the game. It therefore supplies evidence of reuse, not a taint finding.

### baseline1: disclosed old failures, materially stronger current controls

baseline1 should not be labeled tainted merely because its authors found taint
in development. The first paper documents that old harnesses exposed the true
game ID, enabled web search, or allowed a second unscored local client, and says
those results were excluded. The current study says the true ID is absent from
files, arguments, environment variables, and API responses; general web and
Codex search are disabled; only necessary service endpoints are proxied; a
second client is rejected; and the agent runs in Docker.

The residual caveat is model contamination, not a demonstrated run-time leak:
GPT-5.6-sol postdates the public games and could recognize them visually. The
paper says so. Its 183/183 result is therefore evidence of public harness
saturation, not held-out ARC-AGI-3 generalization. The paper-2 Zenodo record
publishes multi-gigabyte raw run bundles with recorded checksums, so a complete
latest solve-boundary audit is possible. It has not been ingested into the
current exact-boundary audit suite. The repository and aggregate result tables
therefore establish the v1.6 architecture and score, but not a conditional
description reduction at a particular winning command.

### OPINE-World: secure reported mode, incomplete headline reconstruction

The paper's reported mode uses raw frames, Docker confinement, and no mounted
game source. It reports a four-sweep transcript audit finding no successful
source, web, or credential access. The public code also contains less secure
optional modes, including object-centric interfaces and non-Docker isolation
choices; those options are not evidence against the reported frames-only
Docker run.

The artifact boundary is the real qualification. Released main logs yield 153
positive-reward events, while the paper reports 160 levels. The archive lacks
an `s5i5` summary; repaired `sb26` has one more reward in its summary than the
main log, and `tr87` has the converse mismatch. The correct statement is not
“the 160 result is fabricated”; it is that the exact headline is not
reconstructible from the main-log layer alone. The release remains much more
auditable than a score-only claim.

### OPINE-World: what actually crosses a level boundary

The implementation makes broad reuse the normal run-wide data flow:

- `run.sh:26-39` is the reported frames-only Docker configuration.
  `play.py:304-316` enables the planner but leaves autonomous planner execution
  off; the analyzer may call `tools/plan.py` on demand.
- `engine.py:643-659` appends every real transition to one replay buffer.
  `engine.py:828-866` advances the level and flushes queued actions, but does
  not clear that buffer, hypotheses, shared documents, or learned code.
- `engine.py:4196-4203` stages the cumulative replay at each synthesis.
  `engine.py:4425-4450` normally seeds the next synthesis with the preceding
  `game_engine.py`; an escalation branch can reset to a stub while retaining
  the previous attempt separately.
- `engine.py:1758-1789` gives the analyzer the cumulative replay and current
  synthesis. `agentic_consumer.py:616-676` stages the replay, current engine,
  persistent notes, and planner tool; `agentic_consumer.py:1128-1140` normally
  resumes the same model session, and `agentic_consumer.py:1230-1237` writes
  notes back for the next call.
- `epistemic.py:190-260` recomputes Beta posterior, UCB, and Thompson priorities
  from the supplied cumulative replay. In the reported frames-only mode,
  `spriteless_eta.py:355-373` first requires a callable synthesized
  `extract_objects`; failure is recorded and exploration continues. The
  resulting diagnostic is prompt evidence, not an action controller.

Consequently, Bayesian/data/context reuse is architectural and persistent,
whereas synthesized-planner use is conditional. The four direct-call witnesses
among 121 adjacent wins are therefore a deliberately narrow lower bound, not a
count of all operational influence from prior observations or hypotheses.
Causal executable dependence cannot be certified for the 142 analyzer/unknown
wins because their transient winning entry points were not retained.
The [generated sharp-boundary enumeration](generated/comparator_stats.md#opine-sharp-conditional-drops)
lists all 14 drops: 12 transient-analyzer winners and the two coupled
synthesized-planner winners.

### Public scores, official scorecards, and held-out testing

ARC Prize's community page says public-set scores are self-reported unless
noted and that the Foundation cannot authenticate them by default. Its
technical report separately cautions that domain-specific harness scores are
valuable automation research but not automatically evidence of AGI progress.
The current verified-testing policy uses a distinct process and hidden sets.
Accordingly:

- a public score can support an engineering-performance claim;
- a server scorecard can support an action-replay claim;
- released traces and containment can support a provenance claim; and
- only appropriately held-out evaluation supports a generalization claim.

## Uniform narrow solved-checkpoint result

The GKM row is frozen at 2026-07-26 21:02 CEST while the campaign was still
running. Comparator rows are pinned released-artifact audits. The
[generated comparator table](generated/comparator_stats.md), produced directly
from the tracked joint JSON, is the sole authoritative source for the uniform
counts and witness enumeration.

OPINE therefore refutes the categorical hypothesis that every competitor solves
every level wholly anew. It does **not** refute the narrower claim that GKM has
the strongest measured literal-leg reuse in this exact artifact set.

## Reproducibility and provenance

The comparison table and manuscript macros are generated from
`arc/audit_results/marginal-literal-reuse.json`, SHA-256
`a2b6248dff6e4ed31de0299c9882120788745986ff3e699c3888025d4fb84775`.
That file contains the summary and every GKM, OPINE, and baseline1 row used by
the uniform screen. Its pinned comparator inputs are:

- `baseline1_gpt55_xhigh_solved_checkpoints.json`, SHA-256
  `a8bfecfd8c602794dca54d779e71f23cf3df558490c5a129cce9a9ec71ee47ed`;
- `opine-solved-checkpoints.json`, SHA-256
  `6c953f9033262d1a9719e3f5fd9e425be0786e2f7ea8db25d5808b718e69966e`;
- `retrodict-solved-checkpoint-memory.json`, SHA-256
  `e04f389a1a990a6a867e35cf4ee93f21d27aeec90e871e745b569fce80776b53`.

The semiautomated top-level command recomputes GKM, reuses checksum-pinned
comparator rows, regenerates comparison tables and figures, runs tests, rebuilds
the paper, and writes `arc/manuscript/reproduction_report.json`:

```bash
make -C arc/manuscript reproduce
```

For a non-writing drift inspection after the active campaign has advanced:

```bash
python3 arc/manuscript/scripts/reproduce_manuscript.py \
  --allow-live-gkm-drift --report /tmp/gkm-reproduction-report.json
```

A reported difference is provenance drift, not permission to edit old
denominators silently.

With the three external artifact roots available, the raw audits are:

```bash
python3 arc/audit_opine_solved_checkpoints.py "$OPINE_RESULTS" \
  --csv /tmp/opine.csv --json /tmp/opine.json
python3 arc/audit_baseline1_artifacts.py "$BASELINE1_GPT55_RUN" \
  --baseline-repo "$BASELINE1_REPO" \
  --csv /tmp/baseline1.csv --json /tmp/baseline1.json
python3 arc/audit_retrodict_artifacts.py "$RETRODICT_FINAL_RUNS" \
  --out-prefix /tmp/retrodict
```

The source analyzers are:

- `arc/audit_gkm_solved_checkpoints.py`
- `arc/audit_baseline1_artifacts.py`
- `arc/audit_opine_solved_checkpoints.py`
- `arc/audit_retrodict_artifacts.py`
- `arc/audit_marginal_literal_reuse.py`

The manuscript figures are regenerated by
`make -C arc/manuscript figures`; their source is
`arc/manuscript/scripts/generate_figures.py`. The complete paper build is
`make -C arc/manuscript paper`. `arc/manuscript/SHA256SUMS.txt` pins the
manuscript sources and generated figures after a successful build. For full
raw comparator reproduction, set `OPINE_ARTIFACTS`, `BASELINE_RELEASE`,
`BASELINE_REPO`, and `RETRODICT_RUNS`, then run
`make -C arc/manuscript reproduce-full`.

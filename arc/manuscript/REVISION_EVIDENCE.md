# Verification evidence

The manuscript uses the selected GPT-5.6 campaign throughout. This document identifies its frozen data, accounting baselines, and exact-source audit. No solver repair or replacement is part of the manuscript correction.

## Dataset authorities

- Endpoint and acquisition release: `9235ed26627140460efa1f6ca5e4041470cddc14` (PR #37).
- Partial-release receipt (JSON schema 1): `140e37ca7014d5aa6a48a3808fd94e90209c56499dbcd7df9f0fe733a29a7681`.
- Receipt-bound verifier revision: `c1f8168f230732f2d745c234555b3e3dfcb8aefa`.
- Winning-source history: `4d0e42f34d7b1db8305f03d725528dfdefe22511`, tree `85543629cfcafc70eb7230493f394059c8e0ac45` at `arc/crack_lab/agent_solutions`.
- Separate theoretical paper: [Pushout Attachments and Conditional Complexity of Executable Models](https://arxiv.org/abs/2609.25024). Its manuscript under `arxiv/` is separate from the empirical paper.

The implemented admission contract, later frozen endpoint certification, and older acquisition controls are distinguished. Endpoint certification does not establish that every acquisition used later containment controls.

## Central marginal-complexity analysis

The manuscript's central coordinate is the additional description of the next
verified solver given the retained program: ideal K(P_k | P_{k-1}), distinguished
from changes in total size and from discovery effort. The fixed empirical proxy M
compresses normalized top-level AST statements not literally present in the preceding
exact winning program. It omits complete decoding metadata; it is not a calibrated
Kolmogorov estimate or a complete code length.

`scripts/generate_marginal_profiles.py` consumes the corrected exact-source audit and
the frozen-release level inventory. It independently checks the 171-source scope,
141 comparable marginals, 64 decreases, 21 half-or-more drops, 54 static call-site
cases and 13 sharp coupled cases. Its two outputs are
`generated/conditional_marginal_profiles.{json,md}`; the JSON binds both input hashes.
`figures/marginal_complexity_profiles.{png,pdf}` now plots M for the declared dc22,
wa30 and ls20 cases. `figures/marginal_complexity_all_games.{png,pdf}` publishes all
25 games without selecting for oscillation. M1 uses an empty AST dictionary; missing
source, missing predecessors and unclaimed levels remain gaps. Static call sites
are marked separately from scalar decreases.

The earlier 4d0e42f3 and 0210e785 manuscript versions already separated conditional
AST novelty from the net-growth ledger and proposed that retained components could
make later descriptions cheaper. This revision restores that central question while
retaining the corrections to source provenance and implemented candidate selection.
For dc22, M is 1058, 604, 716, 885, 828, 1379 bytes: the L2 and L5 dips exist despite
strictly increasing promoted-source growth. The new routine at each dip calls unchanged `walk_segments` and `toggle_control`;
the old direct-call test excludes the intervening new routine. This is indirect
reuse, rather than an absence of retained calls.

The source-growth comparisons below are supporting accounting. The recorded-cycle
counter does not reconstruct all failed edits or the full discovery process.

## Source measurements and baselines

For a charged file `f`, `d(f)` counts nonblank, noncomment lines plus AST list/tuple/set literal elements and dictionary entries. These are source units, not bits, tokens, or ideal Kolmogorov complexity.

- `D_k = d(legs_k) + d(players_k)` is retained source size after promotion.
- `G_k` sums positive per-file changes between two authenticated consecutive promoted states, starting at L2. An initial-scaffold template does not authenticate `G_1`.
- `C_k_recorded` is the stored charge relative to the working source saved by the producer at the start of the winning iteration. That workspace can already include earlier work; it need not equal the preceding promotion.
- `F_diag = -k + 0.02 * sum(C_recorded)` is recorded after candidate fixation and does not select or admit candidates.

The contemporaneous producer at `cf36d6455057ed19eb7254d46b8d0a5846a9b567:arc/crack_lab/gkm_legs.py` reads current `legs.py` and `players.py` at loop entry (lines 1500–1503), passes them to auto-solve (lines 1512–1513), and uses them as the marginal-complexity baseline (line 1533). The earlier manuscript incorrectly treated that baseline as the preceding promotion. A difference between `C_recorded` and `G` therefore does not by itself identify a counter defect or uncharged source under the producer's actual contract.

## dc22: distinct baselines, valid solver, corrected audit

| Level | d(legs.py) | d(players.py) | D | Recorded C | Primary adjacent promoted G |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 24 | 26 | 50 | 49 | excluded |
| 2 | 43 | 78 | 121 | 71 | 71 |
| 3 | 70 | 147 | 217 | 96 | 96 |
| 4 | 109 | 215 | 324 | 107 | 107 |
| 5 | 269 | 257 | 526 | 2 | 202 |
| 6 | 592 | 259 | 851 | 325 | 325 |
| L2–L6 sum | | | | 601 | 801 |

The full recorded C total of 650 covers L1–L6. The primary comparison uses only L2–L6, where both adjacent promoted states are bound to historical source hashes: recorded C sums to 601 and G to 801. Thus `D_6 = D_1 + sum(G_2..G_6) = 50 + 801 = 851`, without assuming initial workspace bytes.

Before the accepted L5 auto-solve stub added two units, the working workspace already contained 160 library units and 40 player units beyond the preceding promotion. Consequently `C_5_recorded = 2` and `G_5 = 202` use different baselines. The C curve has a trough at L5; G is strictly increasing on the paired range. This difference is not a counter defect or an invalid solver. The raw solver, counters, and replay remain unchanged.

The separate `conditional_standard_scaffold` arrays assume initial file costs of zero and one. Under that explicitly conditional convention, including an initial growth value would give total 850. The actual initial workspace bytes are not authenticated, so this extension contributes neither `G_1` nor a reversal to the primary analysis.

Relevant source paths at history revision `4d0e42f...` are:

- `arc/crack_lab/agent_solutions/dc22_legs/wip_context/level_04/reached_before_debrief_12f44b3c26d6/files/`
- `arc/crack_lab/agent_solutions/dc22_legs/wip_context/level_05/auto_solve_debrief_skipped_0ec2282c641a/files/`
- `arc/crack_lab/agent_solutions/dc22_legs/wip_context/level_06/reached_before_debrief_7671882b61c3/files/`

The earlier audit reconstructed L5 as the previous promotion plus a stub. That reconstruction omitted source already present in the actual working workspace; the executed solver did not lack that source. The audit now reads the complete no-debrief winning snapshot. Its conditional-AST novelty values are `M_L5=828` and `M_L6=1379` bytes. The L5 transition `885→828` is no longer a half-or-more contraction. The other four no-debrief auto-solve snapshots (`lf52` L3, `sb26` L3/L4, `su15` L7) match the prior reconstruction after AST normalization.

`M` is a separate zlib-9 compressed normalized-AST novelty proxy. It omits complete decoding metadata and has no calibrated relation to ideal conditional Kolmogorov complexity.

## Paired C/G profile comparison

`generated/source_charge_comparison.{json,md,tex}` and `figures/source_charge_comparison.{png,pdf}` report the recorded charge and promoted-state growth on identical contiguous known domains. The JSON contains all 25 games; missing pairs are not bridged and flat differences are ignored within each segment. The primary audit authenticates 158 historical manifest-bound states, yielding 134 paired values in 24 games. Twenty games have at least three consecutive paired values, and 18 of those have a direction reversal in both series. Aggregate reversal counts are C=45 and G=43. Only one of the 134 paired values differs: `dc22` L5.

| Game | Paired levels | C reversals | G reversals | Interpretation |
| --- | --- | ---: | ---: | --- |
| `dc22` | L2–L6 | 2 | 0 | Recorded trough absent from promoted growth |
| `wa30` | L2–L9 | 2 | 2 | The rise–fall–rise pattern persists in both |
| `ls20` | L2–L7 | 1 | 1 | A shared final downturn, not repeated alternation |

The full L1–L9 recorded-C profile for `wa30` has three reversals, but restricting both curves to L2–L9 removes the initial recorded-only boundary. Counts from different level domains are not interchanged.

Six `tr87` certification capsules explicitly disclaim historical retained-source provenance and are excluded. A separate extended analysis adds 15 uniquely identified retained-phase candidates without independent promotion authentication. It has 147 paired values and C/G reversal counts 52/50, but these are not pooled with the primary counts. Endpoints do not fill missing historical states. This promoted-state comparison is distinct from the exact pre-debrief winning-source audit used for conditional novelty M.

## Exact-source audit scope

Three older auto-solve boundaries retain only a snapshot after paid debrief:

- `g50t_legs/wip_context/level_03/after_auto_solve_debrief_2bcf04981404`
- `g50t_legs/wip_context/level_04/after_auto_solve_debrief_7375339f7b7e`
- `r11l_legs/wip_context/level_03/after_auto_solve_debrief_1e1b0798f8e5`

These paths are under the frozen `arc/crack_lab/agent_solutions/` tree. The producer takes their snapshot after debrief (lines 1520–1529 of `cf36d645...:arc/crack_lab/gkm_legs.py`), without an authenticated pre-debrief workspace snapshot. Parent-plus-stub reconstruction is therefore insufficient for exact acquisition-source analysis. They remain replay-verified endpoints. This is an evidence boundary, not a finding that their solvers were invalid.

Together with `ft09` L2 and `tr87` L1–L6, these exclusions leave 171 exact winning sources, 144 adjacent transitions, and 141 successive marginal comparisons. Missing source is not bridged. The audit reports 64 decreases, 21 sharp drops, 54 static unchanged-leg call-site cases, 13 coupled static cases, and eight sharp drops without that call-site condition. Actual call execution is not instrumented; static references and whole-program replay remain separate evidence.

## Reproduction

From the repository root:

```bash
python3 arc/manuscript/scripts/audit_revision_sources.py \
  --output-dir /tmp/gkm-revision-audit
```

This authenticates the source-history tree and materializes 1,401 allowlisted source/checkpoint/metadata files. It does not import or execute solver programs or game implementations. Comparator rows remain the previously pinned non-GKM audit. The generated report records exact source paths, hashes, per-file costs, adjacent-state growth, recorded charges, and their differences without treating distinct baselines as a harness error.

The paired comparison can be regenerated independently:

```bash
python3 arc/manuscript/scripts/compare_source_charges.py \
  --output-dir /tmp/gkm-charge-comparison
```

The per-game tables and scalar figures use the frozen acquisition files at publication commit `9235ed...`. Their displayed `C_recorded` values are preserved workspace-local charges; they are not relabeled as adjacent retained-state growth. `G` is computed only from authenticated adjacent retained states, separately from the exact pre-debrief winning states used for `M`. No source gap is bridged or filled using a normalized endpoint capsule.

Current test, document-build, and integrity results are in `BUILD_VERIFICATION.md`. The frozen endpoint/taint/action/receipt evidence is identified separately from source-only manuscript checks. Discovery cost, causal search savings, actual helper-call execution, and held-out generalization are not established by this audit.

## Concrete source structure at marginal dips

`scripts/audit_reuse_cases.py` checks five specified cases against the same immutable
winning-source history. Outputs `generated/reuse_case_audit.{json,md,tex}` bind the
source-audit hash, both programs' source hashes, unchanged-definition AST hashes and
the complete named call paths. Duplicate player definitions are resolved by their
actual last definition. The four dc22/wa30 dips have unchanged-helper paths through
a newly added routine. The cn04 L3 player is a 23-action literal loop calling only
`env.step`, with M decreasing from 337 to 245 bytes. That is a concrete compact-action
case alongside the direct and indirect retained-helper cases. The suite-wide
54 direct-call and 13 sharp/direct-call counts are unchanged.

The manuscript reports these specific findings in place of generic doubt about
reuse at the dc22/wa30 troughs. The case audit records source call structure; it does
not add runtime call counts or a matched discovery-efficiency experiment.

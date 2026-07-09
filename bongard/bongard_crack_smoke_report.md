# Bongard Crack: First Smoke Run (Sonnet proposer, rendered Bongard-LOGO)

First live run of the `bongard/crack_lab/` harness (plan:
[`bongard_crack_plan.md`](bongard_crack_plan.md)): 2 fresh-seed Basic
Bongard-LOGO problems rendered to raw 128x128 panels, solved by the
**headless Claude Code Sonnet proposer** writing perception predicates from
pixels, with the harness doing the MDL rule composition and rotated-LOO
verification. This is a smoke run (n=2), not a result; it establishes that
the loop closes end to end.

## Reproduction

```bash
cd bongard/crack_lab
../../.venv/bin/python bongard_legs.py --limit=2 --seed=20260709 \
    --source=basic --tag=logo_smoke --minutes=10 --ladder=sonnet,sonnet
```

Replay validation (cold start, promoted artifact only):

```bash
../../.venv/bin/python -c "
import bongard_arena as A
problems = A.sample_problems('../../downloads/Bongard-LOGO', limit=2,
                             seed=20260709, source='basic')
preds = A.load_predicates('agent_solutions/logo_smoke_predicates/predicates.py')
for k, p in enumerate(problems):
    print(f'problem_{k:02d} ({p.concept}):', A.verify(preds, p).result_line())
"
```

## Results

```text
problem      concept           attempt  model   solved  heldout  rule                              marginal_C
problem_00   open_s5           1        sonnet  True    1.000    p_is_regular_circular_arc>=0.5    126
problem_01   arc_three_lines1  1        sonnet  True    1.000    p_is_wide_hand_drawn_arc>=0.5     10

solved 2/2 | total_marginal_C = 136 | F = 0.720
marginal-C trace: 126 -> 10
```

Both replay-validated from a cold start (deterministic substrate: re-running
the verifier IS the replay). No Opus escalation was needed; zero predicate
crashes.

## What happened, qualitatively

- **The proposer invented real perception, unprompted.** For problem_00 it
  wrote an algebraic (Kasa) circle fit and a geodesic-BFS curve-length
  estimator, with an explicit note that raw pixel count would be confounded
  by stroke thickness. No predicate names or recipes were in the priors.
- **The reuse collapse is already visible at n=2.** Problem_01 reused the
  circle-fit and arc-span helpers and paid only `marginal_C = 10` (one new
  composite predicate) versus problem_00's `126` (the whole perception
  stack). This is the sawtooth's reuse floor, prediction 1 of the plan, at
  the smallest possible scale.
- **The information boundary held.** The workspace contained only opaque
  `problem_XX/` panels; concept names (`open_s5`, `arc_three_lines1`)
  appear only in the harness-side `results.json`.

## Honest observations

1. **Atom pricing can be gamed by composite predicates — the proposer said
   so itself.** It deliberately folded an AND of two raw measurements
   (circle-fit residual, arc span) into ONE composite `p_*` predicate,
   because the rule search prices every atom equally and leave-one-out
   rotations could tie-break toward a cheaper-but-wrong single-threshold
   rule when the raw measurements were exposed separately. The *total*
   accounting still holds — the composite's internal complexity is charged
   in `marginal_C` (the library grew by exactly that code) — but
   `rule_cost` alone understates rule complexity, and the incentive pushes
   conjunction logic out of the transparent rule layer into opaque
   predicates. Logged as reconciliation item R4 in the plan; candidate fix
   is pricing an atom partly by its predicate's definition share.
2. **Articulation match is loose on Basic problems, as expected.** The
   intended Basic rule is shape identity ("this is `open_s5`"); the agent
   articulated a geometric characterization ("a clean single circular arc
   spanning ~107-111 degrees") — a faithful description of the rendered
   family, but not a name match. The name-match protocol becomes
   meaningful on Abstract concepts (`convex`, `symmetric`, ...), which
   this smoke run did not include.
3. **Composite predicates blur the rule/predicate boundary in the
   articulated rule.** `p_is_wide_hand_drawn_arc>=0.5` is only readable
   together with its definition; the human-readable rule is the
   predicate's docstring plus the threshold. Report tooling should print
   the selected predicates' bodies alongside the rule.
4. **n=2, Basic only, single seed.** Nothing here is evidence about
   Abstract concepts, the sawtooth at scale, or the controls under live
   proposers. Next: longer run (20-30 problems, basic+abstract), the
   no-share and shuffled-sides controls on live-solved problems, and the
   escalation statistics.

## Cost

Both problems were solved in one Sonnet attempt each, inside the 10-minute
cap (whole run ~16 minutes wall-clock). No Opus tokens were spent. This is
consistent with the plan's bet that Bongard is the cheap substrate for the
weak-proposer question.

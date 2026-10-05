# Does an interface extension repay its description cost?

Scope: this is a library-retention assay, not a direct LLM-versus-mechanical
cofibration comparison. The separately registered direct comparison is
[COFIBRATION_AB.md](COFIBRATION_AB.md).

## Prospective protocol

This follow-up measures future use of all six frozen LLM lineages from
[FRONTIER_RETRIES.md](FRONTIER_RETRIES.md), including both interface extensions.
The two-task result established exact source specialization, not subsequent
compression. This experiment separates those claims. This protocol is fixed
before any measured new search or model call.

### Tasks and interventions

Continue each lineage with three tasks: swap leaf pairs, retain the first token
of each leaf pair, and retain the second token. The framed recursive input and
the swapped before/between/after payloads are unchanged. Copying and duplicating
leaves were the two original tasks, so these are the other three operations in
the existing elementary basis. No new primitive or desired helper is supplied.
Training inputs are identical between the three tasks at each seed; their target
outputs differ. Use fresh data seeds **21, 22, 23** and disjoint token pools for
training, validation, hidden evaluation and stress tests.

For each of the six lineages and each fresh seed, compare:

1. **Before:** library after task 1, without the task-2 attachment.
2. **After:** library after task 2, including the complete retained attachment.

Both arms can only search the same existing single-call adapter grammar: choose
a retained procedure and bind its Boolean/procedure arguments. Each task allows
10 seconds and 10,000 candidates. A winning adapter is retained for the next
task; no new helper is synthesized. Failed tasks do not erase the library or
prevent attempts at the other, independent operation variants. Report exhaustion
of this finite adapter subset separately from time/candidate limits. A failure
does **not** prove that arbitrary compositions or mechanical synthesis cannot
solve the task.

This yields **108 prospective retained-library outcomes**: 6 lineages × 3 fresh
seeds × 2 snapshots × 3 tasks. Trials share the same six learned programs; the
fresh seeds are robustness checks, not 18 independent abstraction discoveries.

Add **nine independent cold controls**, one for each fresh seed and task. Each
starts from the original elementary library and three supplied source programs,
with no learned helper or prior-task history. Use the unchanged `gpt-5.6-sol`
prompt, transport, structured language and first-valid retention rule. Allow
300 seconds and four replies per task, including the common 10-second reuse
probe. These are newly declared budgets, not the earlier 600-second/six-reply
allowance. No additional retry is added after looking at results. Worst case:
45 minutes across the nine cold tasks, at most 36 model calls. Completed receipt
usage is reported; managed access does not establish a dollar cost.

The cold controls ask how much description an actual independent solution uses,
not the minimum possible description. They are shared comparators across the
six lineages, not six independent cold trials per task. Do not claim a new
equal-compute LLM-versus-mechanical synthesis comparison from this experiment.

### Measurement and checks

- Preserve the existing expression-node count. Plot/report original task-1 and
  task-2 additions, followed by the three prospective additions. Also record new
  argument declarations. Size remains diagnostic and never changes selection.
- Charge the entire new task-2 proposal in the extension cases, with no discount
  for an exact-specialization certificate. Count each subsequent adapter too.
- Compare each future task with its matching cold control only when both pass
  the private evaluation. Failed or missing solutions have no complexity value;
  never enter them as zero or claim savings against them.
- For a conservative payback diagnostic, sum cold-minus-retained future node
  counts and ask when that saving covers the entire task-2 addition. This is
  repayment relative to the observed cold solutions, not an optimality claim.
- Record the complete before/after interventions. In the two extension cases,
  this removes all task-2 cells, not just a syntactically isolated helper. Use
  execution and individual-helper elision to identify which new helper actually
  carries the effect. Preserve old cells and replay prior validation tasks.
- Freeze every planned trial, source hash, input-library hash and selection.
  Reconstruct all pushout certificates and check request/response provenance.
  Re-enumerate failed adapter searches that claim finite-subset exhaustion.
- After all selections are frozen, test 40 hidden and 13 stress examples per
  admitted task, including 128-leaf trees. Audit every binary shape through seven
  leaves with two fresh token assignments, and 81 concatenated-region cases.
  These are finite behavioral checks, not universal correctness proofs.
- Run only one model client or search worker at a time, with the existing
  640 MiB process-group cap. Do not alter the model, proposer prompt, grammar,
  verifier or historical reports to favor an observed result.

This is a deliberately narrow causal follow-up about the utility of a discovered
interface. It cannot establish arbitrary control-structure discovery or a
repeated sawtooth across unrelated tasks. It can establish that an earlier costly
generalization enables cheaper, genuinely new transfers and that removing it
removes that capability within the specified adapter grammar.

The model uses the existing managed Codex transport documented under
[Codex exec](https://learn.chatgpt.com/docs/developer-commands#codex-exec).
No credential is read or copied by this experiment.

## Reproduction

```sh
python transduction/transfer_value.py --output output/transduction_transfer_value/20261004-v1 --prior output/transduction_frontier/20261004-retries-codex
python transduction/transfer_value.py --output output/transduction_transfer_value/20261004-v1 --replay
python transduction/analyze_transfer_value.py --output output/transduction_transfer_value/20261004-v1 --plot
```

At clean completion, report all outcomes and failure modes, then wait for the
user's reply. No automatic follow-up campaign.

## Results

**The two interface extensions enabled subsequent transfer, and their added
description cost was recovered within the three-task follow-up.** This is a
controlled result about the retained interfaces, not just a suggestive curve.
Every planned outcome is included; the benchmark sources stayed unchanged.

| Starting library | Verified future tasks | Interpretation |
| --- | ---: | --- |
| Two extension lineages, before task 2 | 0/18 | Every permitted adapter was exhausted |
| Same two lineages, after task 2 | 18/18 | Unchanged extended helpers solved all transfers |
| Four earlier general lineages, before task 2 | 36/36 | Their first helper already sufficed |
| Same four lineages, after task 2 | 36/36 | Task 2 was not necessary for these transfers |
| Independent cold controls | 9/9 | Fresh synthesis also solved each task |

The before/after counts are paired on identical training and evaluation data.
The failed before arms exhausted the finite adapter subset, rather than hitting
their time or candidate caps. Fresh replay independently repeated that exhaustion.
This is not a claim that the old libraries could not support newly synthesized
glue: those arms deliberately tested direct adapters only.

Both extended helpers kept their hashes and were executed on all 40 hidden
examples of each corresponding task. Disabling `P6` or `L1` changed correctness
from 40/40 to 0/40 in all 18 extension transfers: **720 necessary-use cases**.
No model was called to perform those transfers. The host enumerated bindings,
attached the selected call root, and verified it. For example, keeping the first
token of each leaf pair produced these actual roots:

```text
T4 = Call(P6, false, swap_pair, keep_first_pair)
T4 = Call(L1, keep_first_pair, swap_pair, keep_first_pair)
```

These are readable renderings of the persisted typed calls, not replacement
programs. `swap_pair` is the original `F004` fragment and `keep_first_pair` is
`C03`. The existing recursive helper supplies the control structure; the new
root only supplies operations. The earlier helpers `P5` and `L0` remain retained
but are not what executes these new transfers. The original strict task-2 reuse
result therefore remains 4/6, not retroactively 6/6.

### The complexity profile

All three fresh data seeds produced the same retained node counts:

| Original lineage | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seed 2, trial 1: extension to `P6` | 31 | 45 | 7 | 7 | 7 |
| Seed 4, trial 2: extension to `L1` | 33 | 47 | 8 | 8 | 8 |
| Seed 2, trial 2: direct reuse | 42 | 8 | 8 | 8 | 8 |
| Seed 3, trial 1: direct reuse | 42 | 8 | 8 | 8 | 8 |
| Seed 3, trial 2: direct reuse | 45 | 7 | 7 | 7 | 7 |
| Seed 4, trial 1: direct reuse | 35 | 6 | 6 | 6 | 6 |

These are new expression nodes per task, not total retained library size or
computed Kolmogorov complexity. The measure is unchanged from the transduction
trial; it is not numerically interchangeable with ARC's historical LOC/literal
ledger. Argument declarations are recorded separately. Old code is not deleted
to manufacture a decrease.

[Complexity plot](../output/transduction_transfer_value/20261004-v1/complexity.png):
the vertical divider separates the frozen first two tasks from the prospective
follow-up. The cold band is the observed minimum and maximum, not a confidence
interval; its mean uses all three successful controls at each task. Two direct
reuse curves overlap exactly. This shows acquisition/generalization followed by
cheap transfers, not several repeated cycles of discovering new control forms.

| Fresh seed | Cold costs, tasks 3–5 | `P6` repays its 45 nodes after | `L1` repays its 47 nodes after |
| --- | --- | --- | --- |
| 21 | 55, 33, 61 | 1 future task | 1 future task |
| 22 | 28, 33, 33 | 2 future tasks | 3 future tasks |
| 23 | 38, 42, 24 | 2 future tasks | 2 future tasks |

For example, on seed 22, `P6` saves `(28−7)+(33−7)=47` nodes over two cold
solutions, covering its entire 45-node task-2 addition. The calculation charges
all the extension code, including the task-2 root, and every subsequent binding.
It does not count the already-solved task 2 as a benefit. Cold solutions cost
24–61 nodes each, compared with 6–8 for retained transfers. These are observed
solution costs, not minima over all valid programs.

### Failure modes and scope

1. **A nominal parameter can be too weak.** `P5` fixed operations internally;
   `L0` exposed an entry argument but overrode operation choices during recursion.
   No binding of those interfaces solved the new tasks. Exposing the internal
   choices in `P6` and `L1` changed what the same adapter search could express.
2. **Fresh synthesis can choose the wrong control structure.** One cold proposal
   propagated an alternating phase through nested regions and scored 4/26; its
   second reply scored 26/26. Another proposed a pair loop without a terminating
   region boundary, overran even the shortest input, and scored 0/26; its second
   reply also scored 26/26. Both failures and their public feedback are retained.
3. **Code costs vary.** The nine independent cold solutions used 24–61 nodes.
   The payback claim is relative to those actual solutions, not a proof that no
   equally short solution exists without the retained interface.
4. **The task family remains fixed.** We varied all three previously unused
   elementary leaf operations, not the recursive framing or the elementary
   basis. The result demonstrates future value within this family; it does not
   establish arbitrary abstraction discovery or universal LLM superiority.
5. **The interventions are specific.** We removed the task-2 cells and tested
   their execution necessity. We did not ablate the soft growth instruction or
   compare it with a hard free-energy selector. This experiment therefore cannot
   identify which part of that instruction caused the interface to be discovered.

The GKM conclusion is now empirical: an expensive attachment can make later
factorizations cheap, and immediate unchanged reuse is not the only productive
form of library growth. The complexity drop and the structural intervention
support each other. Once the interface existed, mechanical binding was enough.
The verifier stayed exact and complexity never selected a winner.

### Verification, effort and completion

- All **117** planned outcomes replayed: 99 admitted task instances and 18
  expected adapter failures. Every admitted instance passed private evaluation.
- All **109 new attachment certificates** reconstructed and verified.
  A second replay in a fresh process reproduced the complete result unchanged.
- Hidden cases: **3,960/3,960**; stress cases: **1,287/1,287**.
- Shape audit: **39,006/39,006**; concatenated regions: **8,019/8,019**.
  These repeated executions share learned programs and are not independent
  abstraction-discovery trials.
- **11 completed model calls**, no incomplete call: 163,081 input tokens
  (40,704 cached) and 39,206 output tokens (37,247 reasoning).
- Model receipt time: 998.49 seconds. Total search time across all three arms:
  1,031.31 seconds. The 54 after-library searches together used 15.02 seconds,
  including worker startup; none called a model.
- Largest sampled model process group: **161.9 MiB**, below the 640 MiB cap.
  No memory kill or model timeout occurred.
- Final test suite: **115 tests and 15 subtests passed**.

Artifacts are under `output/transduction_transfer_value/20261004-v1`, which is
ignored by Git. The protocol, source archive, input snapshots, raw model receipts,
adapter derivations, frozen outcomes, replay summary and accounting are retained.
New code and report changes remain local and uncommitted. No follow-up experiment
starts automatically; this batch ends with a summary and a wait for the user.

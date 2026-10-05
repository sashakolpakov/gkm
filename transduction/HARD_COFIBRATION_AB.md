# Harder recursive glue, with longer proposal budgets

## Fixed protocol, declared before measured runs

This repeats the [direct A/B comparison](COFIBRATION_AB.md), not the earlier
library-retention assay. The two arms independently propose their own attaching
programs. The LLM authors every attachment, including transfer calls; it receives
no host binding proposal. Mechanical search receives no LLM-created program.

### Added difficulty

Each internal region now has payloads before, between and after its two
recursive children. Its middle transformation depends on a summary of the left
child; its trailing transformation depends on a summary of both children. A
fixed traversal or a test of whether the next child is a leaf is insufficient.
This combines and extends the earlier framed and returned-information tasks.

The private reference uses leaf-count parity. It swaps the leading pair, swaps
the middle pair when the left child has an odd leaf count (otherwise keeping
its first token), and keeps the trailing pair's first token when the whole
region has an odd leaf count (otherwise swapping it). This description is not
supplied to either proposer: they receive examples and existing programs only.

Task 1 copies leaves. Task 2 duplicates them, leaving the region rules
unchanged, so a suitable acquired helper should transfer by changing its
binding. Each arm starts from the original nine fragments and three supplied
source programs, independently on fresh seeds **41 and 42**. No learned library
from a previous experiment is imported. Failure of task 1 blocks task 2 for
that arm and seed. There are **eight planned outcomes** in total.

### Budgets and unchanged rules

- **1,200 seconds (20 minutes) per attempted task**, four times the previous
  deadline, for both arms. Allow at most **10 LLM replies**, up from four, and
  **32 million mechanical candidates**, up from eight million. These are
  ceilings, not promised resource use or equivalent compute quantities.
- Preserve the same `gpt-5.6-sol` model, reasoning setting, managed Codex access,
  prompt, typed language, compiler, graph pushout verifier and admission rules.
  The existing invocation remains consistent with the
  [official command documentation](https://learn.chatgpt.com/docs/developer-commands#codex-exec).
  No credential is copied or changed.
- Preserve the mechanical algorithm, ordering and finite template pool. Its
  existing retained-call probe stays within the task budget. Greater elapsed
  time does not turn this heuristic into complete search over the language.
- Give both methods the same **44 public examples and 44 private validation
  cases**. They cover all binary shapes through five leaves, several combs with
  six to eight leaves, and random trees with six, eight and ten leaves. This
  provides evidence against shallow structural shortcuts, rather than relying
  on the poorly distinguishing public samples from the previous hard task.
- Keep hidden cases separate: 40 trees with 12–96 leaves and 13 stress cases,
  including 128-leaf trees. Token pools are disjoint across all four splits.
- Stop on the first training-exact proposal, then apply private validation and
  preservation gates. Private failures stop the sequence; they are never fed
  back to a proposer. Thus an overfit proposal may fail before using its entire
  time allowance. Only public errors and counterexamples drive retries.
- Retain the complete admitted library. Complexity is diagnostic only. No hard
  energy selector, manual repair or solution hint is added.
- Run at most one mechanical worker and one model client concurrently, each
  with the existing sampled 640 MiB process-group limit and timeout cleanup.
  Worst-case allocation is 80 minutes per arm if all four tasks use their full
  budgets. No heavy audits run concurrently with discovery.

The new task adapter uses the existing A/B engine without editing its frozen
source. Both the previous result and the new result remain replayable. Unit
tests use private coverage controls to check that the new task fits the
unchanged mechanical grammar, using three generic effect insertions and a
Boolean return. Those controls are never proposer inputs or discovery successes.
Coverage records for measured seeds are produced after both arms freeze.

### Evaluation and completion

Freeze all eight outcomes before hidden evaluation. Fresh-replay proposal
provenance, independent libraries, graph certificates, admission and blocked
prefixes. For admitted programs, repeat the hidden/stress suite, all binary
shapes through seven leaves with two assignments, and 81 continuation cases.
For claimed transfer, check unchanged helper hashes, runtime use and failure
when the helper is disabled. Distinguish training fit, admission, hidden
generalization and actual recursive reuse.

Changing both the task and the time budget means this is not an isolated
measurement of the effect of extra time on the previous task. It is the
requested harder-task comparison, with more opportunity for both methods.

```sh
python transduction/hard_cofibration_ab.py --output output/transduction_cofibration_ab/20261004-hard-v1 --prepare
python -u transduction/hard_cofibration_ab.py --output output/transduction_cofibration_ab/20261004-hard-v1 --arm mechanical
python -u transduction/hard_cofibration_ab.py --output output/transduction_cofibration_ab/20261004-hard-v1 --arm codex
python transduction/hard_cofibration_ab.py --output output/transduction_cofibration_ab/20261004-hard-v1 --replay
python transduction/analyze_cofibration_ab.py --output output/transduction_cofibration_ab/20261004-hard-v1
```

At clean completion, report results, costs, and failure modes, stop all owned
workers, and wait for a reply. No additional experiment starts automatically.

## Completed results: 4 October 2026

**The LLM solved acquisition and the subsequent task on both seeds. Mechanical
search solved neither acquisition within 20 minutes.** All admitted programs
also passed the independent evaluation. The two LLM transitions were different:
one reused an unchanged recursive helper, and the other extended its interface.

| Seed | LLM acquisition | LLM second task | Added nodes per task | Mechanical acquisition |
| --- | --- | --- | --- | --- |
| 41 | 709.9 s, seven replies | 45.3 s, one reply | 77 → 84 | Timeout; 9,261,328 candidates |
| 42 | 591.7 s, five replies | 70.9 s, two replies | 100 → 12 | Timeout; 10,377,878 candidates |

The LLM completed four of four planned tasks. Search attempted two acquisitions,
failed both, and therefore never attempted the two dependent tasks. Those
blocked tasks are not independent failures or zero-cost solutions.

The winning acquisitions came after the old five-minute deadline and after the
old four-reply ceiling. The recorded public scores were:

- seed 41: **2, 2, 6, 2, 6, 37, 44**, out of 44;
- seed 42: **syntax error, 2, 6, 33, 44**, out of 44.

No winning program was manually repaired. Public compiler errors and failed
executions drove the revisions. Both winning acquisitions then passed 44/44
private validation cases. These trajectories show why the extra allowance
mattered in this run; they are not a separate randomized experiment isolating
time from all the other task changes.

### What was learned and reused

The LLM programs traverse five positions in a region: leading payload, left
child, middle payload, right child and trailing payload. Recursive calls return
a Boolean summary which influences the operations at later positions. This is
new control and data flow, not a list of memorized input/output pairs. The two
seeds discovered different interfaces for this behavior.

**Seed 42: direct recursive reuse.** The first task retained `Walk5Cut` with
function arguments. The second task attached only this root composition:

```text
(seq (call Walk5Cut (ref C02) (ref C02) (ref F004) (ref C03)) unit)
```

`C02` duplicates a pair, `F004` swaps it, and `C03` retains its first token.
The recursive helper's content hash stayed unchanged. It executed on all 40
hidden cases; disabling it reduced correctness from 40/40 to 0/40. The first
transfer reply was incorrectly bound and scored 2/44; model-generated correction
produced the successful call. There was no host binder in this arm.

The initially acquired `FirstPair` helper was retained but was not needed in the
second task, which used the existing equivalent fragment `C03` instead. Thus
not every acquired helper transferred, and the first acquisition was not a
minimal description of the behavior.

**Seed 41: verified interface extension, not direct reuse.** The first helper,
`walk6`, had Boolean arguments but hardwired copying at leaves. The model then
created `walk6f` with an additional function argument, `emit`. Fixing `emit` to
the old copy fragment `C01`, and removing that fixed argument from recursive
calls, recovers the entire original program AST exactly. Every recursive call
preserves the binding. The original Boolean arguments and the order of the
local bindings and effects are unchanged.

This identity was checked after selection, not used to help the proposer or to
promote its program. The existing specialization auditor was conservatively
extended to handle lexical `let`: parameter capture, shadowing and changed
recursive bindings are rejected. Neither proposer nor the admission code was
changed. The proof is content-bound to the old and new helper hashes, and all
**564 behavioral comparisons** across the four Boolean argument settings and
the public, validation, hidden and stress inputs agreed on output, consumption
and returned value.

`walk6` itself executed on **zero** hidden cases in the second task. Its retention
is not counted as direct reuse. The extension proof is also not an extra graph
pushout certificate, and it makes no claim of equal interpreter step cost.

This explains the diagnostic curves: **100 → 12** shows cheap unchanged reuse,
whereas **77 → 84** records the cost of generalizing an initially narrow
interface. Both are successful learning outcomes under exact admission checks,
but only the former exhibits an immediate size drop. Future cheap reuse of
`walk6f` was not tested in this fixed two-task protocol.

### Verification and failure modes

All eight planned outcomes were frozen before the independent evaluation.
Two separate processes reproduced the same complete replay. Across the four
admitted programs, the results were:

- hidden cases: **160/160**;
- stress cases, including 128-leaf trees: **52/52**;
- binary-shape checks: **1,576/1,576**;
- concatenated-region continuation checks: **324/324**;
- promoted graph attachment certificates: **8 verified**.

The coverage controls also passed for both tasks and seeds. They establish that
solutions exist within the declared mechanical grammar; they are not counted
as search discoveries. The whole test suite passes **125 tests and 23 subtests**.

Mechanical search hit the wall-time limit on both seeds, not its candidate or
memory cap. Its best recorded prefix was only one public example in both runs.
That statistic is a sorted-prefix score, not full accuracy. The unchanged
algorithm's template pool, eviction policy and joint binding enumeration remain
substantial limitations. The extra time did not overcome them here. This result
does not rule out a better mechanical synthesis algorithm.

The LLM had several semantic mistakes, including temporary regressions, plus one
malformed program and one initially incorrect transfer binding. More feedback
rounds allowed recovery. The first acquisition on seed 41 still chose an
interface too narrow for direct reuse on the next task; exact admission alone
does not guarantee a useful future interface or immediate compression.

### Resources and reproducibility

Recorded proposal/search time was **1,417.8 seconds** for the LLM's four tasks
and **2,400.6 seconds** for the two mechanical acquisition attempts. Search
tested **19,639,206 synthesis candidates**, plus 24 retained-call candidates.
There were **15 completed model replies and no interrupted replies**.
Receipts report **373,522 input tokens** (47,488 cached) and **55,609 output
tokens** (52,077 reasoning); cached and reasoning counts are subsets, not
additional tokens. No API billing amount is inferred from the managed login.

Peak sampled process-group memory was **44.4 MiB** for mechanical workers and
**160.5 MiB** for the model client, below the 640 MiB limits. Neither arm suffered
a memory or transport failure. All owned discovery workers exited cleanly.

Local ignored artifacts are in
[`output/transduction_cofibration_ab/20261004-hard-v1`](../output/transduction_cofibration_ab/20261004-hard-v1).
They include the frozen plan and discovery sources, all requests and replies,
independent library histories, selection records, certificates, `summary.json`,
`analysis.json`, and `interface-extension.json`. The last file records the exact
old/new hashes, fixed binding, recovered body and behavioral comparisons.
`post-audit-sources.tar.gz` preserves the specialization auditor, accounting
script and new tests separately from the unchanged discovery sources.

The extension diagnostic can be reproduced by calling
`audit_frontier.exact_extension` on the stored `walk6` and `walk6f` definitions
and their library signatures. The returned binding is `emit = (ref C01)`.
The behavioral check compares `Machine.run("walk6", source, args=flags)` with
`Machine.run("walk6f", source, args=(*flags, "C01"))`, for all four Boolean
flag pairs and all 141 cases across the four seed-41 task-1 data splits.
Compare output, cursor and return value, not interpreter step count.

### Conclusion

This is stronger evidence than the earlier short run that **LLM proposals can
find useful recursive glue beyond this mechanical search's practical reach**.
The exact verifier remains common to both. The successful programs were found
from examples, not by giving the LLM the reference rule or a desired interface.

The retention outcome is not uniform: one learned interface supported direct
reuse; the other needed a verified extension. A diagnostic size drop captures
the former but would miss the latter. Neither requires a hard complexity
selector. These are still two seeds of one harder task family, with fixed
primitive and type vocabularies—not proof of universal model superiority or
arbitrary abstraction discovery.

The experiment is complete. No further experiment starts until the user replies.

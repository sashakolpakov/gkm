# LLM proposals versus mechanical attachment search

## Correction and fixed protocol

The [library-value experiment](TRANSFER_VALUE.md) tested an LLM-created library
with mechanical transfer and fresh LLM controls. That is useful evidence about
retention, but **not the requested comparison of LLM and mechanical cofibration
proposals**. Its results remain separate and are not counted as evidence from
this experiment.

This protocol is fixed before any new measured search or model call. There are
two genuinely separate construction processes:

- **LLM arm:** `gpt-5.6-sol` proposes every attachment, including simple reuse
  calls. The host supplies public examples and counterexamples, compiles and
  verifies the proposal, and retains it. It does not enumerate or select a call
  binding for the model. The model's `reuse_probe` field is empty.
- **Mechanical arm:** the existing bounded search first tries retained calls,
  then synthesizes new helpers and glue using generic typed abstraction and up
  to three effect insertions. It receives no LLM-created cell or proposal.

Both start independently from the same nine elementary fragments and three
explicitly supplied concrete source programs. Both may add named helpers with
chosen argument/result interfaces in the same accepted typed language. Both
use exactly the same graph pushout construction, correctness gates and full
retention. Neither has a hard complexity selector. A single body, template or
intermediate answer is never transferred between arms or between data seeds.

### Cumulative tasks

Use fresh data seeds **31 and 32**, one independent run per arm and seed. Each
run starts from the original basis and follows six tasks:

| Task | New requirement | Leaf operation |
| --- | --- | --- |
| 1 | Payload between recursive children | Copy |
| 2 | Same structure, requiring transfer | Duplicate |
| 3 | Payloads before, between and after children | Copy |
| 4 | Same structure, requiring transfer | Duplicate |
| 5 | Middle operation depends on information returned by the left child | Copy |
| 6 | Same returned-information structure, requiring transfer | Duplicate |

Those descriptions and reference programs are private to the benchmark. Each
proposer receives only the public examples, elementary instructions, retained
programs, type language and earlier public task history. The mechanical
implementation uses current examples and retained source; its search policy does
not exploit the serialized prior examples as a learned proposer state.

Failure stops that arm's contiguous sequence for that seed. Subsequent tasks
remain recorded as blocked, not removed or assigned zero complexity. A failure
in one arm does not stop the other. No imported solver is used to restart a
failed arm at a later task.

### Matched rules and budgets

- **300 seconds per attempted task**, including all search/proposal rounds.
  The LLM has at most four replies; mechanical synthesis has an eight-million
  candidate ceiling. These method-specific caps are reported, not equated.
- The mechanical arm includes its existing ten-second retained-call probe in
  that budget. The LLM gets the full budget to propose its own attachment and
  receives no host-generated candidate or binder result.
- Stop on the first training-exact proposal. Apply private validation and
  replay of previous validation tasks before promotion. A gate failure stops
  the sequence; private counterexamples are never returned to a proposer.
- Keep the complete admitted library. Measure new expression nodes after
  promotion, with no bonus for an interface extension and no scalar ranking
  among valid programs. Record exact calls, dependencies, runtime helper use and
  helper elision separately from the size curve.
- The public/validation/hidden/stress split stays 26/26/40/13 cases, with disjoint
  token pools and stress trees reaching 128 leaves.
- At most one model client and one mechanical worker concurrently. Each has the
  existing sampled 640 MiB process-group limit. The local mechanical search and
  remotely served model do not have equal hardware or compute expenditure;
  this is a matched elapsed-time experiment. No heavy audits run during search.
- All **24 planned outcomes** are frozen before evaluating hidden cases.
  Replay request/response provenance, full independent library histories, all
  attachment certificates, every admission and every blocked prefix. Audit all
  binary shapes through seven leaves with two token assignments, plus 81
  concatenated-region cases per admitted task.
- Construct independent language-coverage certificates after both arms freeze.
  Each task has a solution within the mechanical search grammar; none of these
  certificates is a proposer success or a hint supplied during discovery.

The mechanical proposal algorithm is unchanged from the earlier frontier trial.
It is a bounded heuristic, not exhaustive search over the accepted language. In
particular, new synthesis starts from anti-unified concrete nullary source
skeletons; already parameterized helpers are available for calls but are not
all automatically lifted as new control skeletons. Ordering and sketch eviction
are further limits. Report these as possible causes of a bounded search failure,
not as intrinsic inability of mechanical methods to create a cofibration.

The important distinction is **who proposes the attaching program**, not two
different standards of categorical validity. Both arms must pass the same exact
verifier. “Softer” LLM construction means greater proposal flexibility, not a
weaker gate.

The existing Codex transport and model are preserved, following the
[official command documentation](https://learn.chatgpt.com/docs/developer-commands#codex-exec).
No credential is read, copied or changed. Worst-case search allocation is 60
minutes per arm across two seeds; actual early stops and usage will be reported.

## Commands

```sh
python transduction/cofibration_ab.py --output output/transduction_cofibration_ab/20261004-v1 --prepare
python -u transduction/cofibration_ab.py --output output/transduction_cofibration_ab/20261004-v1 --arm mechanical
python -u transduction/cofibration_ab.py --output output/transduction_cofibration_ab/20261004-v1 --arm codex
python transduction/cofibration_ab.py --output output/transduction_cofibration_ab/20261004-v1 --replay
```

The two `--arm` commands can run concurrently in their separate output paths.
Preparation freezes the common manifest; each arm freezes all its outcomes;
replay refuses to evaluate an incomplete pair. At clean completion, report the
comparison, costs and failure modes, then wait for the user's reply. No automatic
follow-up experiment.

## Completed comparison: 4 October 2026

**The LLM reached a further construction frontier on both seeds, but did not
solve the final requirement.** These are independent proposal processes: every
LLM attachment came from the model, and every mechanical attachment came from
search. No successful library was shared between them.

| Method | Seed 31 | Seed 32 | Added expression nodes, on each seed |
| --- | --- | --- | --- |
| Mechanical search | Tasks 1–2 verified; task 3 timed out | Same | 36 → 6 |
| LLM proposals | Tasks 1–4 verified; task 5 timed out | Tasks 1–4 verified; task 5 failed validation | 27 → 6 → 35 → 6 |

Of twelve planned outcomes per arm, mechanical search admitted four, failed two
attempts, and left six blocked. The LLM admitted eight, failed two attempts, and
left two blocked. These are contiguous progress counts, not accuracy estimates
on twelve independent tasks. Blocked tasks have no measured complexity or
search time; they are not zero-cost successes.

### Where the methods separated

The first parser was easier for mechanical search: it constructed the helper in
36.1 and 38.0 seconds, testing 238,192 synthesis candidates on each seed. It then
selected the correct retained call in 0.29 and 0.28 seconds, after 28 bindings.
The LLM needed 67.6 and 109.3 seconds for construction, and 223.6 and 131.5 seconds
for the first transfer. On seed 31 it initially duplicated the middle payload
as well as the leaves; public counterexamples enabled its correction.

The larger parser reversed the result. Mechanical synthesis tested **1,888,660
and 2,005,511 candidates** without finding a solution before its five-minute
limit. The LLM constructed valid helpers in **61.6 and 44.8 seconds**, then wrote
the reuse calls itself in **13.6 and 12.4 seconds**.

This search is not plain breadth-first enumeration. It abstracts retained
concrete source programs, enumerates typed insertions and parameter bindings,
and rotates among a bounded set of edited templates. It records its best public
prefix but does not use that score to adapt the enumeration. Template ordering,
eviction and the limited lifting of acquired parameterized helpers are real
limitations. The private coverage check subsequently verified solutions for all
six tasks inside its grammar. Thus these timeouts show a limitation of this
search policy and budget, not an impossibility for mechanical construction.

### What the cofibrations and reuse actually were

Both proposers supplied typed program definitions. The common compiler built
their control graphs and attached them through retained procedure entry and
return interfaces. The exact graph pushout and immutable dependencies were
checked in the same way for both arms. The LLM did not receive a weaker gate.

For example, the first LLM run introduced `quinaryAlternating` on task 3. Its
task 4 proposal was exactly:

```text
(call quinaryAlternating (ref C02) (ref F004))
```

Here `C02` is the existing pair-duplication fragment and `F004` swaps a pair. The
recursive helper was not rewritten. The new root supplied different arguments
and attached to that helper's existing interface.

Across both seeds, there were **six verified transfers of recursive helpers**:
two mechanical task 1→2 transfers and four LLM transfers, task 1→2 and task 3→4.
For each transfer, the old helper's content hash was unchanged, it executed on
all 40 hidden cases, and disabling it reduced correctness from 40/40 to 0/40.
This checks actual necessary reuse in the selected program, not just the
presence of a helper name or a small root body.

Task 3 nevertheless introduces a **new** recursive helper. It does not execute
the task 1 helper. The experiment establishes discovery and subsequent reuse
within those pairs, not unchanged parser reuse across all changes in format,
and not a merger of several independently learned recursive parsers.

The repeated size drops are therefore supported by execution evidence. They
occurred without a hard complexity score: first admissible proposals were
retained, under the existing reuse and minimal-glue guidance. Expression nodes
are a diagnostic in this syntax, not Kolmogorov complexity in bits. In
particular, the mechanical output contains constant branches that its search
does not simplify. Absolute size differences between proposers should not be
read as a proof of optimal compression.

### Failure modes

- **Mechanical search:** candidate growth at the change in parser structure.
  Both timeouts' best recorded prefix was two examples; this prefix statistic
  is not a full accuracy score. Later tasks were blocked, so this experiment
  did not directly test mechanical discovery on task 5.
- **LLM, seed 31:** the first task 5 proposal used a local shape condition and
  scored 21/26 on public training. The next reply exceeded the remaining budget.
  No attachment was promoted.
- **LLM, seed 32:** the first task 5 proposal had malformed parentheses. The
  model repaired it after compiler feedback and achieved 26/26 training, but
  only **23/26 private validation**. All four prior tasks still passed. This
  was a new-behavior generalization failure, not corruption of the retained
  library. No private counterexample was returned to the model.

The last case is important: a mathematically valid graph attachment and perfect
training fit do not imply the right abstraction. Its two graph certificates
verified, but the behavioral admission gate rejected the program.

### Verification, resources and artifacts

All 24 planned outcomes were frozen before the independent audit. A second
fresh process reproduced the complete replay result, including the rejection
and all blocked prefixes. Across the twelve admitted programs:

- hidden cases: **480/480**;
- stress cases, including trees with 128 leaves: **156/156**;
- every binary shape through seven leaves, with two assignments: **4,728/4,728**;
- concatenated-region continuation checks: **972/972**.

The verifier checked **20 attachment certificates**: 18 for promoted definitions
and two for the rejected task 5 proposal. Private grammar-coverage programs are
separate controls and are not counted as discoveries. The complete test suite
passes **119 tests and 15 subtests**.

Mechanical search used 675.1 seconds over six attempted tasks, 4,370,555 synthesis
candidates and 500 retained-call candidates. The LLM used 1,250.8 seconds over
ten attempted tasks, with twelve completed replies and one interrupted reply.
Completed receipts report 238,746 input tokens (13,568 cached) and 42,292 output
tokens (40,434 reasoning). Those subsets are not added again. The interrupted
reply has no complete usage receipt, so these token totals are incomplete.
The existing managed Codex login was used; no API billing amount is inferred.

Maximum recorded process-group memory was 44.8 MiB for mechanical workers and
162.0 MiB in completed model receipts, against the 640 MiB limits. The only
terminated model reply hit its wall-time limit, not a memory limit. The model
and search do not use equal hardware; budgets match elapsed time per task,
not compute or total time across the differently long successful sequences.

Local, ignored artifacts are in
[`output/transduction_cofibration_ab/20261004-v1`](../output/transduction_cofibration_ab/20261004-v1):
the frozen plan and sources, independent proposals and receipts, both frozen
outcome lists, `summary.json`, `analysis.json`, and the
[complexity plot](../output/transduction_cofibration_ab/20261004-v1/complexity.png).
Post-selection accounting is reproducible with:

```sh
python transduction/analyze_cofibration_ab.py --output output/transduction_cofibration_ab/20261004-v1 --plot
```

### Conclusion

This directly supports **LLM proposals for changes in recursive structure,
and mechanical search for bindings within an acquired interface**, under the
tested budget. It does not establish universal LLM superiority: there are only
two fresh data seeds with the same task mechanics, the mechanical strategy is
bounded, and neither arm completed the sequence. The unresolved LLM frontier
is inferring the correct returned summary rather than a structural shortcut.

Both methods use the same exact categorical construction. The greater
flexibility is in finding the attaching program, not in relaxing validity.
This experiment is complete; no follow-up experiment starts without a reply.

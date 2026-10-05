# Control-structure frontier

## Question and protocol

The corrected interface trial established reusable attachment with both
proposers. It did not establish a practical advantage for LLM proposals. This
experiment changes the control structure needed to consume an input, while
keeping the elementary token instructions, nine fragments, three supplied source
programs, accepted language, exact graph attachment and model unchanged.

The LLM is not given a softer verifier. Its flexibility is the ability to propose
different bodies and interfaces. Both proposers must pass the same replay,
validation, provenance and old-task preservation gates. There is no hard energy
selector. The whole admitted library is retained.

Four families are fixed before calling the model:

1. A control using the existing recursive layout, changing local behavior.
2. An additional payload between two recursive regions.
3. Additional payloads before, between and after the recursive regions.
4. A payload between regions whose treatment depends on a Boolean summary
   computed while processing the preceding region.

The proposer sees only public input/output examples and the retained library,
not these family descriptions, encodings, signatures, insertion sites or reference
programs. The second task changes the leaf operation while keeping the new
structural convention and payload operations fixed. Thus it asks for reuse of
the newly learned control, not foresight about another unseen format change.

Each arm has 180 seconds of search per stage, including the identical bounded
reuse probe. The model has at most two replies, with public training feedback.
Its existing managed `gpt-5.6-sol` transport and prompt are unchanged. Both arms
have a monitored 640 MiB process-group cap. The initial seed is 1. There are 26
training, 26 private validation, 40 hidden and 13 stress cases per task; stress
inputs reach 128 leaves. All selections freeze before hidden evaluation.

## Expanded mechanical baseline

The old search could only lift a retained control skeleton. The expanded search
also inserts up to three effects before or after arbitrary typed expressions.
Effects call procedure arguments, possibly selected using available Boolean
values. An insertion after a value-producing expression uses a lexical binding
to preserve its result. These are generic AST operations, not encodings of the
four task families. Up to four procedure arguments are supported within the
unchanged total argument bound.

The zero-edit streams are never evicted and receive at least one third of search
batches. The edited search widens across signatures, edit counts and positions,
using up to 512 active sketches and a budget of 2,000,000 tested candidates.
A sketch must receive 256 candidates before eviction. Bindings and typed Boolean
fillings use the existing diagonal enumeration. This is bounded heuristic search,
not exhaustive synthesis over the whole accepted language. Timeouts and search
failures must be reported as such, not impossibility proofs.

Importantly, a private certificate constructs a solution to every task **inside
this expanded mechanical grammar**. Tests check its generic derivation and exact
execution; the certificate is withheld from both proposers and published only
after selection freezes. Therefore the old baseline's obvious missing-insertion
capability cannot alone explain failure of this expanded baseline. Search
ordering and finite budgets can still explain failures.

## Outcomes to report

Record solves, actual later use of unchanged acquired helpers, fresh pushout
replay, helper-body elision, hidden/stress accuracy, size diagnostics, time,
mechanical candidate counts by edit count, model usage and peak sampled memory.
Separate the model's helper proposal from a host-discovered later binding.
Retain failed families and partial successes in the report. Do not select only
conditions where the LLM wins.

An LLM advantage here would be a practical proposal/search advantage under the
specified budgets, not a distinction between two kinds of categorical pushout,
not general superiority over mechanical synthesis, and not a replication of ARC.
The frontier still restricts types, memory and available control operations.

## Results

The pilot is complete at `output/transduction_frontier/20261003-control-v1`.
The protocol above was written before its first model call. Frozen sources,
proposal provenance, attachment certificates and executions passed fresh replay.

**The pilot produced one genuine LLM acquisition and transfer beyond the
mechanical search's budget, but a fresh-data repeat did not reproduce it.**
Mechanical search was faster on the two easier families. Both failed the
returned-information family within the budget. The table below reports the
pilot; the failed repeat is reported separately below.

| Family | Mechanical acquisition | LLM acquisition | Subsequent unchanged-helper transfer |
| --- | ---: | ---: | --- |
| Existing recursive layout (`control`) | 0.56 s, 1 candidate | 33.70 s | Both, about 0.28 s |
| Payload between regions (`infix`) | 31.44 s, 238,192 candidates | 72.70 s | Both, about 0.28 s |
| Payloads before, between and after (`framed`) | Timeout, 1,340,914 candidates | 65.34 s | LLM library only, 0.28 s |
| Payload governed by preceding region (`feedback`) | Timeout, 1,184,570 candidates | Failed within 180 s | Neither; acquisition failed |

Thus mechanical search completed two of four acquisition/transfer pairs, and
the LLM arm completed three. Failed acquisitions block their transfer stage;
those tasks remain in the denominator. The LLM's first feedback proposal passed
10/26 public training cases. Its second reply exceeded the remaining time, and
no candidate was promoted. No budget extension or prompt correction was made.

Every admitted task passed all 40 hidden and 13 stress cases: 212/212 cases across
the four mechanically admitted tasks, and 318/318 across the six LLM-arm tasks.
These conditional accuracies do not count the failed tasks as successes. Every
successful transfer visited the acquired helper on all 40 hidden examples;
removing that helper's body reduced accuracy to 0/40. Its content hash was
unchanged. All five transfer bindings were found by the common host reuse probe,
without another LLM call.

### What the winning glue actually does

The model proposed `W5(leaf, odd, even)`, a helper with three procedure arguments.
It interprets an internal region as five successive positions: payload, left
region, payload, right region, payload. It recursively processes all five,
using the `odd` operation at positions 1, 3 and 5 and the `even` operation at
positions 2 and 4. At a nonrecursive pair it calls the current `leaf` operation.
The argument signature and five-call body were proposed by the model, not
supplied as a template.

The two task roots are just:

```text
Acquisition: W5(copy_pair,      swap_pair, copy_pair)
Transfer:    W5(duplicate_pair, swap_pair, duplicate_pair)
```

These readable names denote retained fragments `C01`, `F004` and `C02`.
`W5` itself is byte-for-byte unchanged at transfer. The second root is attached
by gluing its typed dependency boundaries to retained cells. Execution follows
the resulting graph, not a flattened trace outside the certificate. The LLM
invented the helper; the host later discovered the new bindings. These are
distinct contributions.

This is a new recursive topology, not a new primitive. The mechanical grammar
also contains a correct solution, using the old two-child skeleton plus three
generic effect insertions. The LLM's five-child representation need not belong
to that proposal subset. The result shows an advantage of flexible proposal
search over this bounded baseline, not that the target is inexpressible by
mechanical construction.

The `infix` model proposal likewise chose a different representation: a
three-child recursion with a Boolean mode passed between children. Mechanical
search found a simpler insertion into the existing two-child traversal, and
was faster on that task. Flexible proposals do not win every comparison.

### Verification, effort and size

All **15 attachment certificates** passed fresh graph reconstruction and exact
execution replay, including cumulative library continuity and public-input
provenance. The post-selection audit passed **25,040/25,040** cases over all 626
binary shapes up to eight leaves, with four token assignments per shape. Calling
each admitted task twice on concatenated regions passed **810/810** cases,
checking that helpers return at region boundaries rather than relying on end of
input. This is extensive finite testing, not a proof of behavioral correctness
on every possible input. The complete transduction suite passes **101 tests and
15 subtests**, including a check that an audit with no admitted programs cannot
report behavioral success.

For the framed timeout, mechanical search tested 447,684 zero-edit candidates,
149,648 one-edit candidates, 371,850 two-edit candidates and 371,732 three-edit
candidates. It opened 3,907 sketches. The recorded `best.correct` in mechanical
search is the number of consecutive public examples passed before the first
failure, not a full accuracy score. The search can miss a valid solution because
of ordering, sketch eviction and finite resources.

The LLM acquisitions added 19, 44 and 42 expression nodes in the successful
families; their transfers added only 4, 9 and 8. The mechanical acquisitions
added 21 and 36 nodes, with transfer increments of 4 and 6. These are diagnostic
sizes of additions, not decreasing total library size or Kolmogorov complexity.
No hard complexity formula selected a winner or removed a useful helper.

Four completed model receipts record 47,862 input tokens, including 6,784 cached,
and 9,015 output tokens. The timed-out fifth call has no completed usage receipt;
these totals therefore **exclude unknown usage from that call**. Managed Codex
access was used, so no dollar cost is inferred. The largest sampled process-group
RSS in available receipts was 160.7 MiB for a completed model call and 44.2 MiB
for a mechanical worker. No memory kill was recorded; sampling can miss brief
peaks. Search times include the shared reuse probe and process startup, but not
basis construction, admission checks or later audits. Equal wall-time limits do
not imply equal hardware or total compute.

### Replication and scope

After observing the first admitted acquisition and transfer on the three-payload
family, a follow-up was scheduled for that family with data seed 2. It uses the
same frozen code, prompt, budgets and initial library, without carrying the first
run's helper into the second run. This is a replication selected after the pilot,
not an additional preregistered task or a tuned retry. All four pilot families
remain in the report regardless of their outcomes. The follow-up result is
complete at `output/transduction_frontier/20261004-framed-seed2`: **neither arm
acquired a valid helper**, so neither reached transfer.

| Framed trial | Mechanical | LLM |
| --- | --- | --- |
| Pilot, seed 1 | Timeout after 1,340,914 candidates | Acquired in 65.34 s; unchanged-helper transfer in 0.28 s |
| Follow-up, seed 2 | Timeout after 1,298,984 candidates | Failed after two replies in 105.35 s; no transfer |

The repeat used exactly the same source hashes, model, initial library and
limits. The first model reply passed 2/26 public examples; the second passed
4/26. It stopped at the two-reply limit with time remaining, not from a timeout
or memory limit. The first helper used five recursive calls but interchanged
its two operation arguments incorrectly. The second response added a wrapper
to handle a standalone leaf; it left the recursive helper unchanged, so deeper
examples still failed. This was a failed hypothesis and incomplete correction,
not rejection of a correct program by the attachment verifier.

No response was repaired by the host. The repeat's request/response provenance
and frozen failure records passed replay, with **zero admitted attachments**.
It has no admitted programs to audit or score on hidden examples. The private
grammar coverage certificates still pass; the tasks are expressible.

The two completed follow-up calls used 27,496 input tokens, including 13,568
cached, and 3,930 output tokens. Largest recorded model RSS was 159.3 MiB;
the mechanical worker peaked at 43.7 MiB. Taken together, the two framed trials
give one LLM acquisition/transfer success and no mechanical success. That is
evidence of a possible practical advantage, **not a reliable win rate**. Data
sampling and model sampling both changed; two trials do not isolate their
effects or establish statistical superiority.

Both arms use exactly the same graph category and strict attachment checks.
The greater flexibility lies in generating a useful body and its interface,
not in weakening a cofibration. The experiment demonstrates practical discovery
and subsequent reuse under a specified budget; it does not establish that an
LLM is necessary. Better mechanical pruning or search ordering could close this
gap. The unresolved feedback family is the next observed frontier, not a solved
problem omitted from the results. Types, memory, primitives and the accepted
control language are still supplied, and no ARC performance claim follows from
these synthetic tasks.

The subsequent [retry study](FRONTIER_RETRIES.md) varies search allowance and
repeats independent trials without changing these tasks. This pilot's sources
remain in each run's `sources.tar.gz`; use that archive for replay after changing
the harness.

```sh
python transduction/frontier_benchmark.py --output output/transduction_frontier/20261003-control-v1 --seconds 180
python transduction/frontier_benchmark.py --output output/transduction_frontier/20261003-control-v1 --replay
python transduction/audit_frontier.py --output output/transduction_frontier/20261003-control-v1
python transduction/frontier_benchmark.py --output output/transduction_frontier/20261004-framed-seed2 --families framed --seed 2 --seconds 180
python transduction/frontier_benchmark.py --output output/transduction_frontier/20261004-framed-seed2 --replay
python transduction/audit_frontier.py --output output/transduction_frontier/20261004-framed-seed2
python -m pytest transduction/test_frontier_growth.py -q
```

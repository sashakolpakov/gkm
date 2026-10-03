# Return-aware machine growth

This experiment extends the earlier tail-only cofibration pilot. It asks whether
small acquired procedures can grow into a deeper machine, and whether mechanical
search or an LLM is needed to propose the connecting program.

## Results: 2026-10-03

The completed run is
`output/transduction_growth/20261003-returning-ab`. Both arms grew the same
17-procedure library on both seeds: five acquired fragments plus twelve
successive attachments. This establishes a nontrivial form of growth within
the supplied language. It does not yet establish a need for LLM glue.

| Measure | Mechanical search | Codex `gpt-5.6-sol` |
| --- | ---: | ---: |
| Admitted stages, 12 tasks × 2 seeds | 24/24 | 24/24 |
| Hidden examples, lengths through 97 | 1,152/1,152 | 1,152/1,152 |
| Stress examples, lengths 128, 257, 512 | 288/288 | 288/288 |
| Median proposal-stage time | 0.724 s | 17.304 s |
| Total proposal-stage time | 20.628 s | 673.964 s |
| Search effort | 11,390 candidates | 25 model calls |

These times include proposal generation and training checks, not the subsequent
validation, retention, hidden scoring, and independent replay. Both arms share
the grammar and examples; their compute allocations are not identical. Codex
has up to two calls per stage, each with its own 120-second limit. Mechanical
search has one 120-second search window per stage.

Codex succeeded on its first proposal in 23 of 24 stages. Its first seed's
final proposal put pair swapping alone on the false branch, omitting the
earlier run-collapse operation. It passed 46/48 training cases, was rejected,
and was corrected from the two training counterexamples on the allowed retry.
No host-written solution or hidden evaluation feedback was supplied.

### How large a machine actually grew?

The final task branches between two retained pipelines. On equal initial
tokens, it reverses triples, swaps pairs, collapses runs, and retains alternating
positions. Otherwise it collapses runs and swaps pairs. These operations are
calls to previously acquired programs, not an expanded primitive trace.

The final entry reaches **14 procedures including itself**, with up to **seven
procedure levels** observed during execution. One 512-token case executed
10,785 interpreter steps, including 4,434 primitive instructions and 1,462
matched procedure entries/returns. The earlier skills remained correct in the
final library.

All 17 separately inlined entry programs would require 618 term units; the
shared library stores 98. For the final entry alone, its reachable shared
subgraph uses 83 units versus 153 without sharing. These are the explicitly
defined term units below, not byte counts. The unshared execution control
matched outputs, cursor behavior and work counts on all held-out and stress
examples. Sharing saved stored code, not execution work.

For a concrete pushout, the final attachment joins two retained procedure
interfaces. Its boundary has four nodes and six labelled edges; the old
library has 74 nodes and 174 edges; the glue has nine nodes and 19 edges.
Identifying the shared interfaces produces 79 nodes and 187 edges. The old
bodies are not edited or copied, and the resulting graph is what executes.

### Verification and scope

Independent replay rebuilt **58 pushouts**: ten common fragment acquisitions
and 48 later attachments. It reproduced all mechanical selections, checked
Codex proposals and training-only requests against the recorded events,
replayed admitted training/validation executions and final hidden/stress
executions, and repeated the dependency interventions and unshared controls.
Source hashes matched the frozen protocol.

An additional audit checked **every equality pattern through length eight**:
5,296 patterns for each of twelve final entry programs, all exact. All four
frozen libraries have the same hash, so this audit performed 63,552 unique
executions and reused the evidence for identical libraries; it does not claim
four independent exhaustive runs. Longer tests remain sampled, not exhaustive.

The Codex requests used the existing managed login, requested `gpt-5.6-sol`,
and produced zero tool events. Receipts total 283,355 input tokens and 21,235
output tokens; they also report 19,328 reasoning output tokens, which are not
added again to the output count here. Peak observed Codex process-group RSS
was 160.91 MiB, below the 768 MiB stop limit. All experiment processes finished.
The test suite passes 46 tests plus seven subtests.

The present result favors mechanical search for this finite grammar. The next
discriminating benchmark should widen the common grammar and require new
interfaces or stateful control beyond the supplied forms. More examples of
these same compositions alone would not establish general algorithm invention
or an LLM advantage. These results do not replace the original register-
transducer benchmark or the earlier tail-only A/B results.

## Fixed protocol

Two independent curricula use seeds 1 and 11. Each arm starts with the same five
fragments, acquired by breadth-first primitive instruction search on examples:
copy one token, skip one, duplicate one, swap a pair, and reverse a triple. The
learner does not receive these names. Fragments have width 1, 2, or 3, local
registers, and explicit return interfaces. Their instruction words are learned,
not supplied as solutions.

Each arm then grows its own library across twelve tasks:

1. Copy an arbitrary stream.
2. Duplicate every token.
3. Swap complete pairs, preserving a trailing singleton.
4. Reverse complete triples, preserving a remainder.
5. Retain positions 0, 2, 4, and so on.
6. Collapse adjacent runs of equal tokens.
7. Swap pairs, then collapse runs.
8. Collapse runs, then swap pairs.
9. Reverse triples, then retain alternating positions.
10. Reverse triples, swap pairs, then collapse runs.
11. Apply the previous transformation, then retain alternating positions.
12. Branch on equality of the first two input tokens, selecting task 11 if
    equal and task 8 otherwise.

Task names and reference implementations are **not** given to proposers. Both
receive only training input/output pairs and their own retained library. The
shared language permits calls, sequences, progress-checked loops, a next-token
equality branch, and pipelines. Ten fixed expression shapes contain at most
three calls to retained procedures. Call depth grows through previous additions;
new glue does not contain raw primitive instructions. Both arms have exactly
the same shape and binding vocabulary.

Mechanical search enumerates that vocabulary, stopping at the first training
exact proposal, with 30,000 candidates and 120 seconds per stage. Codex uses
`gpt-5.6-sol` through the existing managed login, with two proposals at most and
120 seconds per call. A second proposal sees only training counterexamples.
The data-only Codex transport disables tools and rejects tool events; its
process group is monitored and stopped above 768 MiB. This is local experiment
isolation, not the ARC production container's containment claim.

The first training exact proposal faces fixed validation, old-skill retention,
and dependency ablation checks. It is not replaced using validation feedback.
There is no hard free-energy selector. Mechanical grammar order and the LLM's
small-glue instruction provide proposal preferences; size is recorded as a
diagnostic. Each successful complete library is retained. The two arms do not
receive each other's acquired programs.

There are 48 training and 48 validation examples per task, up to length 12.
All selections across both seeds and arms are frozen before 48 hidden examples
per task (up to length 97) and 12 stress examples (lengths 128, 257, and 512)
are scored. Splits use disjoint token pools. Samples include random streams,
short repeated runs, constant streams, and periodic streams. All older tasks
are evaluated through the final accumulated library, not through old snapshots.

## What is attached

The new objects are labelled **program graphs**, not flat transducers. A
procedure has an entry, a body, and a return port. The boundary of an attachment
contains the entry and return interfaces of its dependencies, including their
type labels. Its two maps embed those interfaces in the retained library and
the new glue. The graph pushout identifies just those interfaces, keeping one
shared copy of each retained body. A call enters that shared body and resumes
its caller when it returns. Nothing edits the retained body's outgoing edges.

The interpreter executes the resulting quotient graph. A content hash binds
each procedure's AST, type, window width, and dependency hashes. Certificates
record the original span, quotient, and four maps. Replay independently rebuilds
the quotient, checks the AST against that graph, reproduces mechanical choices,
checks Codex proposals against their original events and training-only prompts,
and repeats the recorded executions.

Injective graph maps are designated cofibrations for this experiment. The
pushout is exact in that labelled graph category; this does not assert a Quillen
model structure, a homotopy pushout, or semantic correctness from category
theory alone. Execution and retention checks supply separate evidence.

## Controls and limits

For ablation, a referenced procedure's body is replaced by a no-op that returns
without changing the cursor or output. Caller progress and whole-input checks
remain active. A dependency must affect at least one training example. This is
stronger than simply forcing every ablated call to fail.

An unshared execution control copies each syntactic call's complete procedure
subtree. Loops remain loops; executions are not unrolled into traces. Outputs,
cursor positions, success, work counts, and call depth must match the shared
machine on every hidden and stress example. This is **not** a separate search
baseline without a library. Reported term units count constructors, primitive
instructions, and two procedure-interface units. They are a consistent storage
diagnostic, not Kolmogorov complexity in bits or serialized graph bytes.

The host supplies the composition language, local register/window conventions,
call stack, guard, and pipeline buffers. Tasks were designed to be compositional
in that language. Calls are acyclic; input-dependent loops are supported, but
unrestricted recursive program invention is not tested. Per-execution limits
are 50,000 interpreter steps, 4,096 output/buffer tokens, and call depth 64.
Success establishes growth within this language, not a general solution to ARC
or arbitrary program synthesis.

After selections are frozen, `audit_growth.py` additionally enumerates every
token-equality pattern through length eight: 5,296 patterns per task. Because
the language cannot inspect token values except by equality, these represent
all inputs of those lengths up to renaming. Identical frozen libraries share
audit computation; the receipt identifies their hashes rather than claiming
independent repeated evidence. This finite check is not a proof for all lengths.

## Run and verify

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -u transduction/growth_benchmark.py \
  --output output/transduction_growth/new-run
PYTHONDONTWRITEBYTECODE=1 python3 -u transduction/growth_benchmark.py \
  --output output/transduction_growth/new-run --replay
PYTHONDONTWRITEBYTECODE=1 python3 -u transduction/audit_growth.py \
  --output output/transduction_growth/new-run
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest transduction -q -p no:cacheprovider
```

Use a fresh output directory; the runner refuses to overwrite one. Raw experiment
artifacts remain under ignored `output/`. Earlier tail-only pilots and their
reported results are unchanged.

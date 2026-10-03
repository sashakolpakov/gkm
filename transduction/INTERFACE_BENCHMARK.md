# Growing reusable interfaces

## Status

The previous interface trial is discarded. Its weak standalone-solving prompt
did not give the LLM the same library-growth objective embodied in the mechanical
constructor. Its results are not evidence against LLM discovery of reusable legs.
The invalid trial's artifacts have been permanently deleted.
Earlier, separate transduction experiments are unchanged.

The corrected run is complete at
`output/transduction_interface/20261003-growth-v2`. The protocol below was frozen
before its first proposal; source hashes still match after fresh replay.

## Results

**Both methods created reusable helpers and transferred through them in all
three families, without supplied helper signatures.** The corrected experiment
does not support the earlier suggestion that this LLM fails to grow legs.

| Proposer, with shared harness | Task solves | Hidden and stress cases | Actual helper transfers | Search time |
| --- | ---: | ---: | ---: | ---: |
| Mechanical | 6/6 | 318/318 | 3/3 | 26.31 s |
| Codex / gpt-5.6-sol | 6/6 | 318/318 | 3/3 | 182.82 s |

Each method acquired one new parameterized helper per family. On every subsequent
task, the common reuse probe found a new binding of that exact helper. No helper
body changed, no replacement helper was added, and no second-task LLM call was
needed. Each acquired helper was visited on all 40 hidden transfer examples;
eliding its body reduced accuracy to 0/40. This is actual use, not just creation
of a function bearing a reusable-looking name.

The model chose a procedure argument for the local family; a Boolean plus two
procedure arguments for the inherited-context family; and two procedure
arguments plus a Boolean, with a Boolean return, for the returned-information
family. None of those signatures was supplied. The inherited-context acquisition
needed two responses: the first passed 10/26 public training cases, and the
second passed all 26 after receiving two public counterexamples. The other
acquisitions each used one response.

### Concrete transfer in the hardest family

The LLM invented `ALT(action_a, action_b, mode)`, which consumes one nested region
and returns the mode to use next. Within a composite region it passes the left
child's returned mode to the right child. At a leaf it invokes the selected
action and toggles the mode. The retained helper supplies the traversal and
coordination; the two task compositions differ only in action bindings:

```text
Acquisition: ALT(copy_pair,      swap_pair,   false)
Transfer:   ALT(duplicate_pair, keep_second, false)
```

These readable names denote the unchanged acquired fragments `C01`, `F004`,
`C02`, and `C04`. The actual selected calls are in
`returned/codex/stage-1/selection.json` and `stage-2/selection.json` under the run
directory. The second task adds a root with glued calls into the old library,
not another traversal implementation. The LLM proposed the helper; the common
mechanical probe selected its second-task bindings. Both contributions are
recorded separately.

### Effort, size and verification

Mechanical acquisition examined 1, 9,974 and 173,227 synthesis candidates,
respectively, after the shared reuse probe. All three source generalizations
and typed dataflow derivations passed independent reconstruction. Acquisition
times, including the shared probe and worker/model startup, were 0.57, 1.98 and
22.92 seconds for mechanical search; 39.53, 90.50 and 51.95 seconds for the LLM.
Every transfer took about 0.28 seconds. The table sums these stage search times;
it excludes initial basis construction, admission checks and subsequent audits.

LLM acquisitions added 19, 31 and 38 expression nodes, while their transfers
added only 4, 7 and 9. Mechanical acquisitions added 21, 33 and 41 nodes, with
the same transfer increments. This is a diagnostic of amortized reuse, not a
hard energy selector, a measurement of Kolmogorov complexity, or shrinking the
entire retained library.

There were four completed model calls: 44,100 input tokens, including 13,568
cached, and 6,469 output tokens, including 5,810 reasoning tokens. Managed Codex
access was used; no unsupported dollar estimate is assigned. Largest sampled
process-group RSS was 161.2 MiB for a model call and 35.4 MiB for a mechanical
worker. No timeout or memory kill occurred. Sampling may miss a very short peak.

All **18 attachment certificates** passed fresh reconstruction and execution
replay, including input provenance and cumulative library continuity. The
post-freeze audit passed **30,048/30,048** cases over all 626 binary shapes up to
eight leaves, using four token assignments per shape, and **972/972** cases
calling each task root twice on concatenated regions. The transduction suite
passes **92 tests and 7 subtests**.

The conclusion is narrower than LLM superiority: the corrected workflow grows
and reuses legs with either proposer. Mechanical search remains faster on these
source structures. The experiment still does not locate a boundary where an LLM
can discover useful glue beyond the mechanical baseline's coverage. It also does
not isolate which individual workflow correction mattered, or estimate success
probabilities from repeated model trials.

## Shared contract

Both arms grow a library across an acquisition task and a subsequent related
task. They receive the same starting programs, token primitives, public examples,
language, retention rules and resource limits. No useful helper signature,
callback count, context count, return meaning, task-family label or future
example is supplied.

Before either proposer runs, the same bounded host search tries calls to retained
procedures with type-correct bindings. If one passes, no proposer is called.
Otherwise, either mechanical synthesis or `gpt-5.6-sol` proposes new definitions.
The LLM is explicitly asked to grow a leg library, factor common structure in the
visible programs, expose observed differences through arguments, and prefer
minimal new glue. Both arms receive the earlier public task examples as well as
the complete retained library. This is workflow guidance, not a supplied answer.

Both must put input inspection and recursion in helpers. The task root binds and
composes procedures using calls, references, sequencing, lexical binding,
conditionals and Boolean operations. A helper may have any allowed signature,
including none. A specialized helper remains admissible: merely moving a
monolith into a helper does **not** establish reusable abstraction.

This explicit shared layout is an experimental counterpart of ARC's leg/player
discipline, not a claim that the ARC campaign used this precise syntax gate.
No hard complexity objective or additional paid compression phase is added.

The first complete training-exact proposal faces independent validation and old
task retention checks. Failure at that gate ends that stage without revealing
private counterexamples. Success retains the complete proposed tree, including
all helpers. Description size is measured after selection, never used to discard
an otherwise admissible helper.

## Fixed problems and budgets

The nine elementary token fragments and three concrete recursive source programs
are unchanged. The fragments were acquired by the earlier bounded word search;
the three complete source programs are explicitly supplied, not discovered in
this experiment. They have no parameters and return unit. Their differing local
calls provide public evidence from which an abstraction can be formed.

Private families test changing local behavior, inherited enclosing context, and
information passed back from one region to govern a subsequent region. Each has
one acquisition and one transfer task. Proposers see only opaque token examples,
not this description or the reference implementations.

Each task has 26 public training, 26 separate validation, 40 hidden and 13 stress
examples. Hidden/stress evaluation occurs only after all selections are frozen.
Token pools are disjoint. Small tree shapes deliberately overlap between training
and validation, while hidden/stress inputs grow to 128 leaves. Seed remains 1.

Each arm has a search budget of 180 seconds per stage, including the shared reuse probe (at most
10 seconds and 10,000 bindings). Mechanical synthesis has a 200,000 candidate
cap. The LLM has at most two responses, with public training failures or syntax
errors returned between them. Both workers have a sampled 640 MiB process-group
limit with termination on overflow. These are practical bounded comparisons,
not equal-compute experiments. Managed Codex authentication is unchanged; no
API credentials are copied.

## What is searched

The language permits up to four definitions, each with zero to four Boolean or
procedure arguments and a unit or Boolean result. Procedure arguments reference
nullary unit procedures. Helpers may choose their own recursive structure,
conditions, sequence, lexical values and Boolean operations. Calls preserve input
progress and output while restoring lexical scope; recursion must advance input.
There is no subtree instruction, token literal, rewind or task identifier.

The mechanical proposer extracts exact common structure from pairs of complete
source procedures, replacing differing nullary calls with procedure parameters.
It then explores typed transformations of those source structures: Boolean
arguments, Boolean return values, binding returned values, and expressions
connecting them. It enumerates short-index combinations across candidate
signatures in bounded-memory round robin order. It is not BFS over all possible
programs and cannot invent arbitrary new control structures or interior cuts.

The LLM can propose any body in the same accepted language; no host repairs its
answer. The proposal transport records structured requests, replies, usage and
resource receipts with tools disabled. The existing [Codex structured-output
interface](https://learn.chatgpt.com/docs/developer-commands#codex-exec) is retained.

## Attachment and measurements

The compiler builds an executable labelled graph for each definition. Injective
maps identify its dependency entry/return boundaries with retained cells, and
the graph pushout preserves those old cells. Actual execution follows the glued
edges and typed procedure references; the graph is not a decorative certificate
beside a separate handwritten solver. No Quillen model structure is claimed.

Report three different observations: a new helper was created; it exposes
parameters; a later task actually uses that exact retained helper. Only the last
is demonstrated transfer. Transfer additionally requires unchanged content hashes,
correct hidden execution, and failure under helper-body elision. Renamed copies,
specialized replacements and unused helpers do not count.

The provenance record separates the proposer that acquired a helper from the
host probe that subsequently found its bindings. A successful host transfer in
the Codex arm is a result for **harness plus LLM-acquired library**, not a claim
that the LLM selected the second task's bindings.

Fresh verification reconstructs certificates and library lineage, checks original
requests and responses, and repeats training, validation, retention and hidden
execution. A separate audit enumerates all binary shapes up to eight leaves and
tests returning before the end of concatenated regions. These are finite tests,
not a proof of correctness for unbounded inputs.

This remains one pilot seed and one model trajectory per family. It tests the
corrected library-growth workflow, not a general separation between mechanical
search and LLM glue. Both methods build transducers using the same pushouts.

## Commands

```sh
python transduction/interface_benchmark.py --output output/transduction_interface/20261003-growth-v2 --seconds 180
python transduction/interface_benchmark.py --output output/transduction_interface/20261003-growth-v2 --replay
python transduction/audit_interface.py --output output/transduction_interface/20261003-growth-v2
python -m pytest transduction/test_interface_growth.py -q
```

The run freezes implementation hashes and a source archive before any proposal.
Requests, replies, selections, certificates, executions and ablations stay in
the ignored output directory.

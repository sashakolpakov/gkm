# Recursive acquisition and reuse

This experiment asks whether a proposer can discover recursive control, rather
than only arrange complete input transformations in a pipeline. Both arms see
the same examples, acquired machines, context interface and output language.
The LLM arm uses `gpt-5.6-sol` through the existing managed Codex login.

## Results

**Both methods acquired recursive control and then reused it successfully. This
experiment did not establish a need for LLM glue.** The mechanical method was
faster; the LLM's traversal had one fewer control node.

The completed follow-up is in
`output/transduction_recursive/20261003-recursive-followup`. It used the
right-branch parity task and a five-minute budget per method per stage.

| Method | Acquire traversal | Transfer to new leaf behavior | Traversal nodes | Unseen cases |
| --- | ---: | ---: | ---: | ---: |
| Mechanical | 95.88 s | 0.28 s | 10 | 112/112 |
| Codex / gpt-5.6-sol | 156.62 s | 132.21 s | 9 | 112/112 |

Unseen totals comprise 40 hidden plus 16 stress examples at **each** stage.
Both handled nesting depth 120. Times include mechanical worker startup and
model request latency, but exclude later independent audits; they are not
equal-compute measurements or averages over repeated trials.

Both acquired the following behavior (readable pseudocode, not a replacement
for either saved program):

```text
walk(context, ordinary_leaf, alternate_leaf):
    if the next two tokens are equal:
        skip one token; skip one token
        walk(context, ordinary_leaf, alternate_leaf)
        walk(context XOR 1, ordinary_leaf, alternate_leaf)
    else:
        call alternate_leaf if context is 1, otherwise ordinary_leaf
    return
```

The first stage supplied copy-pair and swap-pair as the leaf parameters. Each
method's second-stage program was just a call to **its own unchanged R01**,
binding duplicate-pair and keep-first instead, followed by return. Neither the
tree traversal nor a flat trace was copied into the second program.

Fresh replay verified all four pushouts, original model replies, source hashes,
admission results, old-task preservation and hidden outputs. Both second-stage
programs called R01 on every hidden example; its content hash was unchanged.
Replacing R01 with a no-op made **0/40** hidden examples succeed in either arm.

An independent post-freeze audit checked all 626 binary tree shapes with at most
eight leaves, with four token assignments per shape: **2,504/2,504 per retained
program**, across four programs. It also concatenated two trees and called each
retained procedure twice: **256/256 per program**. This checks that a call returns
at the subtree boundary, not merely when the entire input ends. These finite
audits are not a proof for every token assignment and unbounded input.

The solver tried fresh workers with capacities 2, 4, 8 and 12 nodes. It used
three supplied examples as active constraints at the final bound and checked
every candidate against all 26 training examples. The final worker produced
three complete candidates before success. Second-stage search tried seven
parameter bindings, taking 0.012 s inside the worker (0.28 s including startup).
Peak recorded mechanical worker RSS was about 292 MiB.

Codex needed two proposals at each stage. In both cases the first proposal
emitted correct-looking output but continued beyond the input and failed the
consumption gate. Training counterexamples identified that error; its next
proposal passed. No host repair was applied. The four completed calls recorded
44,769 input tokens and 11,033 output tokens, including 9,977 reasoning tokens.
Peak recorded Codex process-group RSS was about 162 MiB; accepted tool events: 0.
Managed-login usage is recorded, not converted into an unsupported dollar cost.

### Failed pilot and limitations

The preceding pilot at `output/transduction_recursive/20261003-paired-ab` tested
both families with 120 seconds per stage. Both mechanical searches hit the
640 MiB monitor threshold, and both model calls timed out without a complete
answer. Consequently **no transfer stage ran**. Those failures are retained.

The mechanical follow-up shortened the symbolic execution horizon and started
with smaller control graphs in fresh workers. Both arms then received a larger
300-second budget on the simpler family. These changes, plus fresh model calls,
mean this is not an isolated experiment on any one optimization. The harder
two-context family has **not** been rerun with the improved search.

The pilot was replayed before the solver changes. Its exact implementation is
preserved in `sources-v1.tar.gz`; running that artifact's replay now requires
those archived sources. The successful follow-up has `sources-v2.tar.gz` and
matches the current source hashes. No prior experiment source was altered.

The elementary instructions stayed fixed, but the host still supplied the
control language, one context bit, and two width-two callback slots. The result
demonstrates learning recursive glue and rebinding it through a verified
attachment—not inventing arbitrary interfaces. A stronger claim about LLM
advantage needs a harder task and more than this single successful paired trial.

## Fixed data operations, broader control

The six token instructions are unchanged: advance, emit the current token,
store in either of two local registers, and emit either register. The five
previous fragment tasks are acquired again by the original word search. Three
additional width-two fragments—copy a pair, duplicate each member, and retain
the first member—are also acquired mechanically from examples. They are new
small machines, not new primitive instructions. Their searches, bodies and
attachment certificates are saved.

Both arms may propose a forward control graph with equality tests, context-bit
tests, fragment calls, calls through two procedure parameters, recursive calls,
and calls to retained procedures. Recursive calls preserve the advanced input
cursor and output but restore the caller's context on return. They require
strict input progress before descent. No tree parser, subtree operation or
traversal template is built into the interpreter or supplied in the prompt.

The host supplies one or two context bits, two callback slots of width two,
and a maximum of 12 or 15 control nodes. Discovering arbitrary interfaces,
arbitrary memory layouts or arbitrary recursive languages is **not** tested.
Only data operations stay fixed; the control language is deliberately enlarged.

## Tasks and selection

An equal token pair introduces a binary node followed by its two children.
An unequal pair is a leaf. This encoding is used by the private data generator;
the proposers receive flat input/output examples, not this parser or a solution.

1. **Right-branch parity:** emit leaves in order, reversing a leaf pair when its
   path has an odd number of right branches.
2. **Two contexts:** reverse a leaf pair only when both its left-branch count
   and its right-branch count are odd.

Each family has two stages. The first acquires a traversal with copying and
swapping as the default leaf procedures. The second instead duplicates both
leaf tokens in the first context and emits only the first token in the second.
A successful second stage can bind different leaf procedures to the unchanged
first traversal. Reuse is measured, not imposed as an admission requirement.
If the first stage fails, that arm's second stage is reported as blocked;
it does not receive the other arm's traversal or the reference solution.

Training and validation each contain 26 examples with up to five leaves.
Hidden evaluation has 40 examples with 6–64 leaves. Sixteen stress examples
include chains 120 levels deep and random trees with 128 leaves. Token pools
are disjoint across splits. All candidates are frozen before hidden evaluation.
Private executable witnesses establish that the tasks are expressible; they
are never counted as proposer successes.

The pilot used 120 seconds per stage; the completed follow-up used 300 seconds.
Both enforce a monitored 640 MiB process-group limit. The LLM may submit at
most two answers within the shared deadline and
receives only syntax errors or failures on its supplied training examples.
Tools are disabled, responses are schema constrained, and tool events invalidate
the attempt. This is local isolation, not the ARC container's containment claim.

The mechanical method tries simple calls to retained procedures first. It then
uses counterexample-guided constraint solving over unknown control graphs:
solve the current examples, execute the candidate against all training examples,
and add the shortest failing example. Fragment effects are exact summaries of
their unchanged bodies. There is no template fixing the tree traversal.
The SMT encoding bounds execution length and training inputs; it is not a
complete decision procedure for recursive program synthesis. A timeout or memory
limit is evidence about this implementation and budget, not mechanical search
in general.

In the follow-up the symbolic horizon is `4 * (input_tokens + 1)` control
transitions and active training inputs/outputs must contain fewer than 120
tokens. Eight-bit indices are range constrained; token equality is preserved by
renaming opaque identities. The general executable language has larger runtime
limits. Thus the mechanical search explores a bounded subset of that language;
neither an SMT failure nor failure below a node bound is an unrestricted
impossibility result.

The first training-exact proposal faces exact validation and preservation checks.
Complexity is recorded structurally, not used as a hard energy admission rule.

## What the attachment proves

Each new procedure is a finite labelled control graph. Its recursive edges are
internal to the new cell. Its calls to existing procedures identify matching
entry and return ports by the existing graph pushout. The old graph, bodies and
content hashes remain unchanged. Both legs of the attaching boundary are
injective graph morphisms; no Quillen model structure is claimed.

The verifier reconstructs the complete executable graph from the saved cell
derivation. Execution uses the quotient graph, not a separately trusted AST.
For second-stage reuse it checks that the retained traversal is actually called,
that its hash is unchanged, and that disabling it destroys the result. Structural
preservation is exact; correctness on the task family is tested, not inferred
from the categorical construction alone.

## Commands

```sh
python -m pip install -r transduction/requirements-recursive.txt
python transduction/recursive_benchmark.py --output output/transduction_recursive/new-run --families right_parity --seconds 300
python transduction/recursive_benchmark.py --output output/transduction_recursive/20261003-recursive-followup --replay
python transduction/audit_recursive.py --output output/transduction_recursive/20261003-recursive-followup
python -m pytest transduction/test_recursive_growth.py -q
```

The focused recursive tests and the existing transduction suites passed together:
**74 tests and 7 subtests**.

Implementation: `recursive_machine.py`, `recursive_search.py`, and
`recursive_benchmark.py`. Existing frozen experiment sources are not modified.
The new protocol binds its own implementation and dependencies by source hashes.

Codex structured responses follow the existing `--json`/`--output-schema`
interface described in the [official command documentation](https://learn.chatgpt.com/docs/developer-commands#codex-exec).

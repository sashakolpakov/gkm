# Growing transducers by verified graph attachment

This is a new, small experiment, not an update to the original ten-row benchmark.
It compares cold evolutionary synthesis with mechanical glue search and Codex
glue proposals. All three use the same register-transducer interpreter. No
solution was planted in the acquisition or glue search.

## What is actually attached

A machine is a finite directed graph. States are vertices; each transition is
an edge labelled with an observation and a sequence of primitive instructions.
Every graph in an attachment has the same primitive signature. A morphism maps
states and transitions while preserving endpoints and labels. For this
experiment, **cofibrations mean injective graph morphisms**. We do not claim a
Quillen model structure or a homotopy pushout.

The three input objects are:

- `L`: the retained machine, with an entry for each acquired task;
- `A`: the interface states to be shared;
- `B`: new control states and transitions, plus interface ports.

The harness constructs `L +_A B` by taking the disjoint union and identifying
exactly the interface images. Both state and transition maps are recorded. The
implementation supports interfaces containing transitions as well, although
this first search grammar uses interfaces consisting of states alone.

This is the ordinary labelled-graph pushout construction; see
[König et al., A Tutorial on Graph Transformation, Definitions 1–3](https://ris.utwente.nl/ws/portalfiles/portal/247999750/K_nig2018tutorial.pdf).
`Pushout.mediate` constructs the unique map to any supplied agreeing cocone.
Tests exhaust a small family of cocones, check edge identification separately,
and reject forged maps and hashes. These tests complement the quotient
construction; they are not a machine-checked proof of a general theorem.

Executability requires additional checks beyond being a graph pushout:

1. The resulting machine must remain deterministic. Conflicting transitions
   are rejected, not silently overwritten.
2. Glue cannot add outgoing transitions to retained states, including places
   where an absent rule used to halt execution. With the same primitive
   signature, executions from every old entry therefore remain unchanged for
   any input. This is stronger than replay on the saved examples alone.
3. The new task must pass all training and validation examples exactly and halt.
4. Removing retained transitions must lower the new task's training accuracy.
   Merely visiting an old state is insufficient evidence of useful reuse.
5. Proposal bytes reconstruct the same quotient and reproduce the executions.

The current execution grammar supports **tail reuse**: new control can enter
retained control, but retained control cannot return to newly added states.
It does not yet support general subroutine calls, register rebinding, or
arbitrary recursive glue. Extending it requires explicit return interfaces
and a revised preservation contract, not silently changing old halt behavior.

## Protocol

The curriculum acquires copy, then duplication of the first token, swapping
the first pair, and triplication of the first token. All tasks use one register
and the same primitive signature. Copy is itself acquired by enumerating
instruction sequences; it takes 11 candidate evaluations in these runs.

Each task has 12 training and 12 validation examples of lengths 2–5. Hidden
testing uses 48 examples of lengths 2–13. The three splits use disjoint token
pools. Proposers see only training examples; they receive neither the task
name nor validation or hidden labels. Every selection in a run is frozen
before any hidden result is computed.

- **Cold evolution:** the existing evolutionary algorithm, population 32,
  four states, at most eight rules and six instructions per rule. Fitness is
  training edit loss plus `0.002 × (rules + instructions)`. No lambda sweep is
  used in this pilot. Budgets below count candidate fitness evaluations.
- **Mechanical attachment:** enumerate one new transition, instruction words
  in increasing length, and every retained state binding. Stop at the first
  training-exact proposal with necessary reuse, then validate it. The grammar
  searches up to six instructions per transition.
- **Codex attachment:** `gpt-5.6-sol`, medium reasoning, existing ChatGPT login,
  at most three proposals per task. It receives the retained graph and can
  propose multiple new states and transitions. Training failures can be fed
  back; no hidden feedback is allowed. No API key is copied or used.

Both attachment methods promote the first proposal passing their exact gates.
The whole graph is retained. Codex is asked for small glue, but no tournament
over replay-valid programs is imposed. A loss-plus-size score ranks partial
fallbacks when no valid attachment is found; it cannot displace an admitted
exact attachment. Library size and sharing remain separately reported.

These are equal **candidate-evaluation caps**, not equal compute or equal
prior knowledge. Attachment pays for and retains its acquired copy machine;
cold evolution starts again for each task. Codex has pretrained knowledge,
a richer proposal grammar, and model inference cost. Its proposal count is
not comparable to the number of evolutionary fitness evaluations.

## Results

Mean hidden exact accuracy across the three new tasks and fixed seeds
`1, 11, 23` (432 hidden task examples per method and budget):

| Candidate cap per task | Cold evolution | Mechanical attachment |
|---|---:|---:|
| 512 | 5.6% | 68.5% |
| 8,192 | 36.1% | 100% |

At 512 candidates, mechanical search acquires duplication and triplication,
but does not find an exact swap attachment. At 8,192, it finds all three. Its
successful search counts are 2, 5,369, and 5 candidates, plus the initial 11
for acquisition. Failed tasks are not promoted and do not enter the library.

A separate seed-7 comparison with Codex, also using the 8,192 cold/mechanical
cap, produced:

| Method | Duplicate first | Swap first pair | Triple first |
|---|---:|---:|---:|
| Cold evolution | 48/48 | 17/48 | 48/48 |
| Mechanical attachment | 48/48 | 48/48 | 48/48 |
| Codex attachment | 48/48 | 48/48 | 48/48 |

Codex needed one proposal for each task. Cold evolution's seed-7 swap passed
training and validation but failed on many longer hidden inputs. Thus training
success alone was not evidence of length generalization.

Independent process replay rebuilt all six seed-7 attachment squares from
their saved maps and verified their execution records. Both final libraries
also retain 48/48 accuracy for **each** of all four tasks, including copy.
Erasing retained transitions reduces new-task training accuracy from 100% to
0% for duplication/triplication and 25% for swap. The two-token swap is the
legitimate case where there is no remaining suffix to copy.

The final retained instruction/rule counts are 13 for mechanical attachment
and 16 for Codex attachment. These are the substrate's existing size proxy,
not Kolmogorov complexity in bits and not a full encoding cost of interfaces.

### Operational notes

The first trial stopped before hidden evaluation. Its reuse gate incorrectly
required old transitions on every input, and its Codex parser rejected harmless
startup notices. The corrected gate uses the removal test above. Deprecated
configuration switches were removed; only the exact pre-turn notice that Code
Mode is disabled is tolerated. Tool events and other errors remain invalid.
That aborted trial is not included in the accuracy tables.

The successful Codex calls used 25,364 input tokens and 833 output tokens as
reported by the CLI (551 reasoning output tokens are a separately reported
field). The earlier aborted call additionally used 8,396 input and 232 output
tokens. These are account-usage counts, not an API-dollar cost estimate.
Each call was capped at 120 seconds, 768 MiB for its process group, and 2 MiB
per output stream. Observed process-group peaks were below 161 MiB. No model
tools were used. The local CLI isolation is not a claim of the ARC production
container's stronger containment guarantees.

Codex final output follows a JSON Schema, and the harness requires a completed
turn before parsing it. This follows the
[official Codex noninteractive interface](https://developers.openai.com/codex/noninteractive)
and preserves the explicitly requested
[`gpt-5.6-sol` model](https://developers.openai.com/api/docs/models/gpt-5.6-sol).

## Files and reproduction

- `cofibration.py`: graphs, injective maps, pushout, mediating map, attachment gate.
- `run_cofibration_experiment.py`: acquisition, searches, selection, held-out evaluation, replay.
- `codex_glue.py`: bounded Codex CLI proposer with structured output and no model tools.
- `test_cofibration.py`: structural, execution, provenance, and transport tests.

From the repository root:

```bash
python3 -m unittest discover -s transduction -p 'test_*.py' -v
python3 transduction/run_cofibration_experiment.py --seed 7 --budget 8192 \
  --output output/transduction_cofibration/new-run --codex
python3 transduction/run_cofibration_experiment.py \
  --output output/transduction_cofibration/new-run --replay
```

Omit `--codex` for the mechanical/cold comparison. Output directories must be
new. Raw run artifacts are under ignored `output/transduction_cofibration/`:
the successful live run is `20261003-seed7-b`; the six mechanical comparisons
are `20261003-seed{1,11,23}-budget{512,8192}`. They contain candidate bodies,
both sides of each span, all maps, hashes, execution traces, and Codex receipts.
The discarded initial trial is `20261003-seed7`.

## What this establishes, and what it does not

This provides an executable example of genuine shared graph attachment,
retention, and improved hidden accuracy on a small curriculum with a useful
common suffix. It does not establish that the categorical construction itself
causes a general accuracy advantage: the representation, curriculum, proposer,
and search order also matter. The baseline has not been exhaustively tuned.

The original benchmark's missing-memory and missing-comparison cases cannot
be fixed merely by increasing search at an insufficient primitive tier. This
experiment does not change those tiers or the original reported results.
The next substantive extension is return-capable glue, tested on tasks where
one retained machine must run, hand back control, and another must continue.

# Bongard-First Benchmark Plan

Bongard problems are a good next external target because they ask for few-shot
concept induction rather than direct answer memorization. A problem supplies
positive and negative examples; the solver must infer a compact rule and classify
held-out queries.

## Current Focus

Concentrate on three tracks only:

1. **Internal generated Bongard rules**
   - Purpose: controlled science.
   - Data source: generated locally by `experiments/run_bongard_symbolic_baseline.py`.
   - Why: we know the true rule, can generate hard negatives, can separate positive/negative panel design, and can run exact overcapacity ablations.
   - Current output: sparse deterministic automata with train/validation/hidden/exhaustive-probe evaluation.

2. **Bongard-LOGO as the first external benchmark**
   - Repository: https://github.com/NVlabs/Bongard-LOGO
   - Paper: https://arxiv.org/abs/2010.00763
   - Why it fits: synthetic, program-guided, object/shape concepts, many procedurally generated instances.
   - The public repository describes a 12,000-problem dataset and a `bongard` Python library for synthesis. Each image is paired with an action program, which is the important handle for us.
   - Dataset policy: do not vendor the dataset into this repository. The adapter accepts an external dataset path or generator path.

3. **Two Bongard-LOGO modes**
   - **Symbolic mode first:** convert LOGO action programs or structured metadata into internal opaque-object / relational scene encodings. This tests rule induction without perception.
   - **Visual mode later:** render or load images, run a parser/object extractor, then feed the resulting symbolic scene into the same solver. This tests the perception bottleneck separately.

Everything else is deferred. Bongard-OpenWorld and Bongard-HOI are useful later stress tests, but they mix perception, open vocabulary, natural-image recognition, and rule induction. They should not be the next step.

## Research Question

Can free-energy selection discover or select compact deterministic concept
solvers that generalize to hidden objects/examples?

The Bongard version of the objective is:

```text
F_lambda(h) = classification_loss_train(h) + lambda C(h)
```

where `h` is a deterministic rule/hypothesis. Selection happens in two stages:

```text
train examples      -> local free-energy minimization
validation examples -> choose loss/complexity elbow
hidden examples     -> report once after validation selection
```

## Why Start With A Local Procedural Bongard Scaffold

A local generator gives us controlled splits before dealing with image parsing:

```text
train objects      disjoint from validation objects
validation objects disjoint from hidden-test objects
hidden test        generated after rule selection
```

It also lets us vary exactly what primitive tier is needed:

```text
stream          length/position-only concepts
compare         equality concepts
bidirectional   concepts using both boundaries / scan directions
register/stack  memory-dependent concepts
```

This is not a replacement for Bongard-LOGO. It is the harness that lets us test
our free-energy protocol before writing a dataset adapter.

## Implementation Stages

### Stage 1: Symbolic Bongard Harness

File: `experiments/run_bongard_symbolic_baseline.py`

- Generate positive and negative opaque-object sequences from concept rules.
- Use disjoint train/validation/hidden-test object pools.
- Evaluate a small deterministic hypothesis library.
- Sweep `lambda`.
- Select by validation loss/complexity elbow.
- Report hidden-test accuracy.

This validates the evaluation protocol and exposes ambiguity when multiple
hypotheses fit a small training panel.

### Stage 2: Evolved Bongard Classifiers

File: `experiments/run_bongard_sparse_classifier.py`
Report: `experiments/bongard_sparse_classifier_report.md`

Replace the hand-coded hypothesis library with sparse deterministic classifiers:

```text
(state, TOKEN | EOS | BOS | MATCH_REGISTER_MASK) -> (actions, next_state, label?)
```

Current clean-slate results with concept-specific budgets and exhaustive discovery probes:

- `length_even` is discovered in 8/8 clean replicates by a stream classifier.
- `has_adjacent_duplicate` is discovered in 8/8 clean replicates by compare/register classifiers.
- `first_equals_last` initially produced sampled near misses under fixed search, but is discovered by counterexample-archive evolution with late lambda warmup and loss-frontier preservation. The successful selected classifier passes train, validation, hidden, and exhaustive foreign-alphabet probes at `1.00` with 6 evolved rules and complexity `12.0`.

The key lesson is that sampled validation can still be misleading. Counterexample-
rich train/validation/hidden splits reduce shortcut pressure, but discovery is
now credited only when the selected classifier also passes an exhaustive foreign-
alphabet probe. The optimizer must keep enough temporary dimensions to escape
local minima, while the complexity term makes those dimensions costly. The
minimal hand-written basin should not be treated as the expected cold-search
endpoint; like engineering simplification, leaner automata may require a later
trajectory after a working overbuilt machine exists.

A first ablation supports this: `first_equals_last` fails under five-rule and
six-rule caps, but succeeds when the genome can carry twelve rules during
development and then selects a six-rule exact classifier. A broader pilot matrix
shows the effect is not universal: `has_adjacent_duplicate` solves even under a
limited cap, while `length_multiple_of_three` and `first_equals_last` show clear
overcapacity benefit. Two initially missed rules, `first_equals_second` and
`last_two_equal`, are better understood as underconstrained panels: adding hard
negative examples makes both discoverable, and both then show the same pattern
where overcapacity succeeds but a cap equal to the final rule count fails. This
is the right evidence shape: task-by-task discovery rates plus panel-design
checks, not a claim that every Bongard rule needs overcapacity.

### Stage 3: Bongard-LOGO Symbolic Adapter

This is now the first external target with a working symbolic adapter. It begins
with LOGO action programs and structured metadata, not pixels. That keeps the
experiment about rule induction and free-energy solver selection.

Adapter shape:

```text
Bongard-LOGO problem
  -> positive examples
  -> negative examples
  -> action programs / shape metadata
  -> relational scene encoding
  -> train/validation/hidden split
  -> same sparse free-energy solver-selection harness
```

Initial scene features should be deliberately simple:

```text
object count
shape type or shape-family token
stroke/action sequence tokenization
relative position bins
size/orientation bins
equality relations across objects
containment / intersection / symmetry flags when recoverable
```

The first adapter does not solve all Bongard-LOGO categories. It starts with
Basic Shape and Abstract Shape problems generated from the local checkout. The
initial report is `experiments/bongard_logo_report.md`: one-shape Basic is mostly
recoverable from action skeletons, while Abstract Shape exposes the expected
representation bottleneck unless privileged metadata attributes are supplied. A
first predicate-macro mode now derives reusable geometric predicates from action
programs; it improves Abstract action-only performance but does not yet replace
metadata. Freeform shapes can be added after the internal scene encoding is
stable.

Dataset policy:

- Do not commit the dataset.
- Accept `--dataset-dir` pointing to a local Bongard-LOGO checkout/download.
- Accept `--generated-dir` or a generator hook later for custom generated LOGO
  problems.
- Cache only small derived metadata if needed, not rendered images or the full
  external corpus.

### Stage 4: Bongard-LOGO Visual Path

Only after the symbolic adapter works:

```text
image -> parser/object extractor -> symbolic scene -> sparse solver
```

This stage compares symbolic vs visual input while keeping the downstream solver
fixed. The target question is whether failures come from perception or from rule
induction. This matches the representation-bottleneck hypothesis suggested by
recent Bongard-LOGO symbolic-grounding work.

## Initial Success Criteria

1. The local symbolic Bongard harness recovers the intended rule on hidden
   disjoint-object examples.
2. Deliberately underpowered primitive tiers fail in interpretable ways.
3. Validation selection rejects over-complex or overfit hypotheses when a simpler
   rule has the same validation loss.
4. The Bongard-LOGO adapter can run without changing the free-energy selection
   code.

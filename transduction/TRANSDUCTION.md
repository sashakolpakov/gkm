# Transduction: synthesising compact deterministic solvers

This domain asks a sharper version of the GKM question: can a meta-evolutionary
process **synthesise compact deterministic solvers** from observed pattern
transitions, under the same free-energy selection rule? The evolved object is a
sparse deterministic **register transducer**; token identities are opaque, so the
solver must rely on finite control and relational observations rather than
memorising symbols.

- **Common math:** [`../FREE_ENERGY_EXPLANATION.md`](../FREE_ENERGY_EXPLANATION.md).
- **Benchmark report:** [`register_transducer_benchmark.md`](register_transducer_benchmark.md).
- **Manuscript:** [`manuscript/transduction.tex`](manuscript/transduction.tex)
  (build with `make -C manuscript`).

## The task

Foreign-object transduction — the solver never sees the alphabet:

```text
train:       [A B] -> [B A], [C D] -> [D C]
validation:  [X Y] -> [Y X]
hidden test: [p q] -> [q p]
```

The rule key sees only finite control and relational observations, never the token
identity:

```text
(state, TOKEN | EOS | BOS | MATCH_REGISTER_MASK) -> (action_sequence, next_state)
```

## Tiered primitives

Primitive sets are deliberately layered so we can ask *which* set is sufficient for
a task family:

- `stream` — move right, write current token, halt;
- `register` — `stream` plus store / write-register actions;
- `compare` — `register` plus equality observations between the current token and
  stored registers;
- `bidirectional` — `stream` plus `MOVE_LEFT` and a beginning-of-sequence observation;
- `bidirectional_compare` — bidirectional motion plus register-equality observations.

## Free energy and Pareto selection

Local selection minimises the training free energy

```text
F_lambda(solver) = training_loss(solver) + lambda * C(solver)
```

The runner then sweeps `lambda` and selects from the **validation** loss-complexity
Pareto frontier: it finds the best validation loss, then keeps the *simplest* Pareto
solver within a small validation-loss tolerance. The hidden-test transition is
evaluated **only after** this validation selection — no peeking. This is the
loss-complexity / free-energy lens of [arXiv:2507.13543](https://arxiv.org/abs/2507.13543)
applied to program synthesis: parsimony is a selection pressure, not a tie-breaker
applied after the fact.

## Key modules

- [`pattern_fsa.py`](pattern_fsa.py) — the sparse register-transducer substrate and CLI.
- [`run_register_transducer_benchmark.py`](run_register_transducer_benchmark.py) —
  reproduces the benchmark matrix over tasks × primitive sets.
- [`test_pattern_fsa.py`](test_pattern_fsa.py) — the domain tests.

## Run

```bash
python3 transduction/pattern_fsa.py --task swap --primitive-set register \
    --generations 120 --population 220 --lambda-points 4
# reproduce the benchmark matrix:
python3 transduction/run_register_transducer_benchmark.py
```

Outputs land in `output/pattern_fsa/` (`solver.json`, `history.json`,
`lambda_sweep.json`, `summary.json`).

## Tests

```bash
python -m pytest transduction/test_pattern_fsa.py -q
```

## Scope

The goal is **not** a general ARC solver. It is to study a meta-model that produces
compact deterministic solvers when the pattern family is deterministic enough — and
to measure how the sufficient primitive tier and the achieved complexity move with
the task.

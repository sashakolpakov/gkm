# Free Energy for Open-Ended Automata Evolution

This follows the loss-complexity landscape viewpoint of Kolpakov,
"Loss-Complexity Landscape and Model Structure Functions"
([arXiv:2507.13543](https://arxiv.org/abs/2507.13543)): sweep the complexity
regularization parameter, estimate the loss-complexity frontier, and look for
phase-transition-like regions where complexity varies strongly.

The paper establishes a Legendre-Fenchel duality between model structure
functions and a free-energy functional, using computable complexity proxies and
a susceptibility-like variance of model complexity to locate sharp
loss-complexity tradeoffs. This repository does not reproduce the paper's
regression experiments; it uses the same mathematical lens for sparse
automata-evolution experiments.

## Core Functional

Both current substrates use free energy as a local selection principle:

```text
F_lambda(a, E_t) = R(a, E_t) + lambda C(a)
```

Where:

- `a` is an agent, currently a finite-state automaton.
- `E_t` is the current evaluation ecology.
- `R(a, E_t)` is behavioral risk in that ecology.
- `C(a)` is a chosen description-length proxy.
- `lambda` controls parsimony pressure.

In the grid foraging substrate:

```text
R(a, E_t) = missed_food + 0.05 * step_fraction + 0.10 * bump_fraction
```

The implementation reports four complexity proxies:

```text
rule_complexity(r) = number_of_moves_in_r + next_state_pointer_cost
C_active(a) = sum rule_complexity(r) over observed rules
C_table(a)  = sum rule_complexity(r) over every encoded sparse rule
C_pruned(a) = sum rule_complexity(r) over rules reachable from state 0
C_mixed(a)  = C_active(a) + 0.25 * [C_table(a) - C_active(a)]_+
```

These are raw code-length proxies, not normalized fractions. A rule may contain
a sequence of moves, so a long macro-rule costs more than a one-move rule.
Each rule matches `(state, previous_move, relative_food_azimuth)` and emits
`(move_sequence, next_state)`. `C_active` measures expressed behavior. `C_table`
is a historical name for the whole encoded sparse rule set; it charges for
unused rules and unused macro length. `C_pruned` removes states unreachable from
state 0 but still charges for all encoded rules whose source state is reachable.
`C_mixed` is a behavioral metric with an explicit dead-code tax. Undefined
inputs halt the episode rather than invoking a free fallback policy.

The code minimizes free energy directly, and reports fitness as `-F_lambda`.

In the pattern-transduction substrate, the same form is used:

```text
F_lambda(s, T_train) = edit_loss(s, T_train) + lambda C(s)
```

where `s` is a sparse register transducer and `T_train` is the set of observed
foreign-object transitions used for local evolutionary selection. Token
identity is opaque to the rule key; depending on the primitive tier, the
transducer can move through the input, write the current object, store/write
registers, branch on current-token/register equality, or use a bidirectional
read-only input cursor with explicit beginning/end observations. For model
selection, the runner sweeps
`lambda` and chooses an empirical elbow on the validation loss-complexity
frontier by taking the simplest Pareto solver within a small tolerance of the
best validation loss. Hidden test transitions from disjoint object pools are
evaluated only after that validation selection. This separates the meta-model
from the solver: evolution searches for compact deterministic transducers; the
transducer itself produces the pattern output.

## Why This Is Not Yet Open-Ended

A fixed finite set of maps is a closed world. In that setting, evolution should improve quickly and then saturate. Positive complexity pressure will then favor compression, not continued innovation.

That behavior is expected.

The closed-world automata game is therefore a baseline. It tests whether mutation, selection, inheritance, replay, and complexity accounting work in a substrate where behavior is visible.

Open-endedness requires the ecology itself to change.

## Ecological Extension

For open-ended evolution, the environment distribution must become endogenous:

```text
E_{t+1} = G(E_t, Population_t, Archive_t)
```

The generator `G` should preserve tasks near the competence frontier:

```text
interesting(e) =
    archive_disagreement(e)
  + learning_progress(e)
  + partial_solvability(e)
  - triviality(e)
  - impossibility(e)
```

Selection still uses free energy locally:

```text
F_lambda,t(a) = R(a, E_t) + lambda C(a)
```

But the risk term changes as the ecology changes.

## Complexity Ratchet

A more complex mutation `a'` can replace `a` only when the risk reduction pays for the extra description length:

```text
R(a, E_t) - R(a', E_t) > lambda [C(a') - C(a)]
```

This inequality is the complexity ratchet. It prevents complexity from growing merely because mutation can add structure. Complexity must purchase control, prediction, survival, resource acquisition, or transfer.

In a fixed ecology, the ratchet eventually stops. In an expanding ecology, new environments can make new structures useful.

## Lambda Sweeps

For each ecological epoch, sweep `lambda`:

```text
F_t(lambda) = inf_a [R(a, E_t) + lambda C(a)]
```

For the finite optimizer, this is estimated by the best sampled/evolved agents
rather than computed exactly. The sweep approximates a loss-complexity frontier:

```text
C_t^*(R) = minimum complexity needed to achieve risk <= R
```

Open-ended evolution should move this frontier over time. The important signal is not merely higher score. The stronger signal is continued emergence of new Pareto-efficient structures under changing ecological pressure.

The implementation records a simple susceptibility proxy for each lambda:

```text
chi_C(lambda) = Var[C(a)]
```

computed over sampled incumbent automata for that lambda. Peaks in `chi_C`
are candidates for phase-transition regions in the loss-complexity landscape.
They are diagnostics, not proof of a thermodynamic phase transition.

## Empirical Signatures

A convincing run should show:

1. **Closed-world saturation**
   Fixed maps produce rapid improvement followed by stagnation or compression.

2. **Frontier movement**
   Generated environments create new risks that older archive agents cannot solve.

3. **Complexity with payoff**
   Complexity increases only when it reduces frontier risk.

4. **Lineage depth**
   Later agents depend on multiple accumulated innovations.

5. **Archive diversity**
   Different lambda values or niches preserve different useful strategies.

6. **Transfer**
   New agents retain old competence while solving newly generated environments.

## Research Interpretation

The free-energy paradigm does not itself create open-endedness. It supplies a disciplined selection rule.

The open-ended part must come from the agent-environment coupling:

```text
free energy selects;
ecology expands;
archives preserve;
lambda sweeps reveal structure.
```

The research question is whether that loop can sustain nontrivial innovation without hand-authoring every new task.

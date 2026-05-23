# Open-Ended Evolution Under a Free-Energy Selection Principle

## Abstract

This thesis proposes that open-ended artificial evolution is possible under a free-energy paradigm when the object being optimized is not a fixed task score, but a coupled agent-environment system whose niches, validation pressures, and attainable abstractions change over evolutionary time.

The core claim is:

> A fixed finite benchmark produces bounded adaptation. Open-ended evolution requires a generative ecology. Within that ecology, free energy can act as the local selection principle that decides which new structures persist.

## 1. Motivation

The original implementation optimizes:

```text
F_lambda(a) = R(a) + lambda C(a)
```

where `R(a)` is empirical risk and `C(a)` is description length. This is a useful model-selection principle, but by itself it is not open-ended. On a fixed finite benchmark, there is a finite set of relevant behaviors. Once the agent solves the benchmark with acceptable complexity, evolution has no principled reason to continue.

This is not a bug in the optimizer. It is a limitation of the experimental frame.

Open-ended evolution needs:

1. A population of agents, not a single hill-climber.
2. Heritable structure with explicit mutation operators.
3. Behavioral interaction with an environment.
4. A way for solved behavior to create new pressures.
5. A complexity cost, so growth must earn its keep.
6. A record of structural novelty, not just score improvement.

The free-energy paradigm can supply item 5 and a disciplined selection rule. It does not, by itself, supply open-endedness.

## 2. Central Thesis

Let an agent be a program, automaton, neural controller, or hybrid object:

```text
a in A
```

Let an environment be a generative process:

```text
e in E
```

Let the agent-environment interaction produce trajectories:

```text
tau ~ P(tau | a, e)
```

Define an empirical risk over a current ecology:

```text
R_t(a) = E_{e ~ Q_t, tau ~ P(. | a,e)} [loss(tau)]
```

Define agent complexity as a description length:

```text
C(a) ~= K(a)
```

For finite automata this must be measured several ways, because each proxy
answers a different research question:

```text
rule_complexity(r): number of moves in r plus its next-state pointer
C_active(a): sum of complexities for behaviorally expressed rules
C_table(a): sum of complexities for the full encoded sparse transition set
C_pruned(a): sum of complexities after removing unreachable states
C_mixed(a): active rule complexity plus an explicit dead-code tax
```

`C_active` asks how simple the expressed behavior is. `C_table` asks how much
machine was genetically carried, including unused rules and unused macro-rules.
The name `C_table` is historical in the current sparse implementation: it means
the whole encoded rule set, not a dense table. In the sparse FSA substrate, a
rule matches `(state, previous_move, relative_food_azimuth)` and emits
`(move_sequence, next_state)`. `C_pruned` asks how large the reachable machine is
after obvious dead code is removed. These are raw code-length proxies, not
normalized fractions. They should be swept separately; a result that only
appears under `C_active` may be an artifact of hiding unused capacity.

Then local selection minimizes:

```text
F_lambda,t(a) = R_t(a) + lambda C(a)
```

The key difference from ordinary benchmark optimization is that `Q_t`, the environment distribution, is not fixed. It is shaped by the population, the archive, and the unresolved boundary of current competence.

Open-endedness becomes possible when:

```text
Q_{t+1} = G(Q_t, Population_t, Archive_t)
```

where `G` generates new niches from solved or partially solved niches.

In short:

```text
free energy selects locally;
the ecology expands globally.
```

## 3. Why Fixed Benchmarks Cannot Be Open-Ended

For a fixed finite benchmark `B`, each agent induces a finite behavior vector:

```text
b(a) = (outcome_1(a), ..., outcome_n(a))
```

Even if the space of programs is infinite, the benchmark only observes finitely many distinctions. Once evolution finds a minimal or near-minimal agent for the best observable behavior vector, additional structure is penalized by `lambda C(a)` without producing lower risk.

Therefore:

> Under fixed finite evaluation and positive complexity pressure, free-energy minimization converges toward compression, not open-ended innovation.

## 4. How Free Energy Can Support Open-Endedness

The free-energy objective becomes open-ended when risk is evaluated against an evolving frontier rather than a fixed list.

A useful decomposition is:

```text
F_lambda,t(a) =
    exploitation_loss_t(a)
  + alpha frontier_loss_t(a)
  + beta interaction_loss_t(a)
  + lambda C(a)
```

Where:

- `exploitation_loss_t`: performance on known tasks.
- `frontier_loss_t`: performance on tasks generated near the edge of current competence.
- `interaction_loss_t`: performance in worlds containing other evolving agents or artifacts.
- `C(a)`: description length or effective degrees of freedom.

The important part is not adding arbitrary novelty reward. The important part is making the ecology generate new risk terms that are still grounded in survival, control, prediction, resource acquisition, or reproduction.

## 5. Developmental Overcapacity

A free-energy system should not be expected to find the globally minimal program basin directly. The first reachable basin for a nontrivial behavior may be overbuilt: it carries extra states, rules, registers, or scaffolding that make the behavior discoverable by local mutation and selection.

This is not a failure of the complexity term. It is a basin-accessibility fact. The term `lambda C(a)` makes extra machinery costly, but it does not make the shortest possible program easy to discover. In many domains the historical trajectory is: first find a working machine with excess structure, then simplify after the behavioral principle is reachable and selection can compare viable descendants. The Raptor-engine analogy is apt: early working designs can be more complex than later generations because later generations inherit knowledge, constraints, and reachable redesign paths that did not exist at the start.

For the sparse automata experiments this means a hand-written minimal automaton is a representability witness, not the default expectation for cold stochastic search. The empirically important object is the naturally discovered basin: how much overcapacity was needed to escape local minima, whether the solution generalizes, and whether later evolutionary pressure can produce simpler descendants without external pruning.

## 6. The Complexity Ratchet

A free-energy system can increase complexity only when complexity buys enough risk reduction.

A mutation from `a` to `a'` is selected when:

```text
R_t(a') + lambda C(a') < R_t(a) + lambda C(a)
```

Equivalently:

```text
R_t(a) - R_t(a') > lambda [C(a') - C(a)]
```

This inequality is the complexity ratchet.

It says that complexity growth is not inherently good. It is admitted only when it pays for itself. Across a changing ecology, new niches can make previously useless complexity useful. This creates the possibility of directional structural growth without abandoning parsimony.

## 7. Lambda Sweeps as Structure-Function Probes

Following the loss-complexity structure-function viewpoint of
[arXiv:2507.13543](https://arxiv.org/abs/2507.13543), `lambda` is swept rather
than treated as a single tuned hyperparameter. The paper establishes the
Legendre-Fenchel duality between free energy and the model structure function,
and motivates susceptibility-style variance diagnostics; the open-ended ecology
proposed here is an extension of that viewpoint, not a claim made by the paper.

For each ecological time `t`, define the ideal frontier:

```text
F_t(lambda) = inf_a [R_t(a) + lambda C(a)]
```

Sweeping `lambda` exposes the current loss-complexity frontier:

```text
C_t^*(R) = minimum complexity needed to achieve risk <= R
```

Open-endedness should appear as movement of this frontier over evolutionary time.
In the implementation this frontier is sampled by genetic search or Hyperopt,
so observed curves are empirical approximations.
When a substrate admits a held-out split, local selection uses the training
free energy, while the chosen reported model is selected by an elbow on the
validation loss-complexity frontier: choose the simplest Pareto solver within a
small tolerance of the best validation loss. Hidden test environments are used
only after this selection step.

The empirical signature is not simply “best score goes up.” The stronger signature is:

1. New regions of the frontier appear.
2. Different lambda values select qualitatively different agents.
3. Previously over-complex structures become compressed or repurposed.
4. The archive accumulates agents that solve different ecological niches.
5. Validation environments generated later require structures not present earlier.

## 8. Proposed Research Substrate

The first serious substrate should include two controlled regimes: visible
embodied behavior and deterministic pattern transduction.

Embodied starting point:

```text
2D grid ecology
```

Agents are finite-state machines or tiny programs. They perceive local observations and choose actions. They can forage, avoid hazards, manipulate simple objects, leave markers, compete, cooperate, or reproduce.

Agent genome:

```text
(state, previous_move, relative_food_azimuth) -> (move_sequence, next_state)
```

This is simple enough to mutate directly and complex enough to show real behavioral improvement.
Rules are sparse. If no encoded rule matches the current input, the episode
halts; there is no random fallback policy.

Pattern-transduction starting point:

```text
train:      [A B] -> [B A], [C D] -> [D C]
validation: [X Y] -> [Y X]
hidden test: [p q] -> [q p]
```

Here the meta-model evolves a compact deterministic solver rather than directly
predicting the output sequence. The solver is a sparse register transducer over
opaque objects. Its rule key does not expose literal token identity:

```text
(state, TOKEN | EOS | BOS | MATCH_REGISTER_MASK) -> (action_sequence, next_state)
```

Primitive sets should be varied experimentally:

```text
stream primitives -> register primitives -> comparison primitives -> bidirectional tape -> richer object primitives
```

This directly tests which substrate primitives are sufficient for compact
deterministic solver synthesis under the same free-energy objective. Reversal is
especially diagnostic: in a one-way stream it requires internal memory, while a
bidirectional read-only input tape can solve it by scanning to the end and
emitting while moving left.

Minimum environment stages:

1. Static foraging.
2. Foraging with obstacles.
3. Moving resources.
4. Resource depletion and respawn.
5. Multiple agents.
6. Agent-made markers or trails.
7. Coevolving predators, parasites, or competitors.
8. Procedurally generated worlds from the archive frontier.

The first stage demonstrates adaptation. Later stages test open-endedness.

## 9. Environment Generation

The environment generator must avoid two failures:

1. It must not generate arbitrary noise.
2. It must not merely replay solved tasks.

A good generator samples near the competence frontier:

```text
Q_{t+1}(e) proportional to
    uncertainty_t(e)
  + learning_progress_t(e)
  + archive_disagreement_t(e)
  - triviality_t(e)
```

Practical approximations:

- Keep tasks where some archive agents succeed and some fail.
- Mutate worlds that current champions solve too easily.
- Preserve worlds that distinguish different behavioral lineages.
- Discard worlds that no agent can make progress on for many generations.

This makes the environment a coevolving curriculum.

## 10. What Counts as Open-Endedness?

A weak system merely improves score. A stronger system exhibits continuing innovation.

Operational criteria:

### 10.1 Continued Adaptive Novelty

New agents solve validation worlds that no previous archive agent solved.

```text
NovelSolve(a_t) = count{e in V_t : success(a_t,e) and no archived a solved e}
```

### 10.2 Complexity With Payoff

Complexity grows only when accompanied by frontier risk reduction:

```text
Delta C_t > 0 and Delta R_frontier_t < -epsilon
```

### 10.3 Transfer

Agents evolved in earlier niches retain competence while acquiring new competence.

### 10.4 Lineage Depth

Useful behavior depends on multiple accumulated innovations, not one lucky mutation.

### 10.5 Frontier Movement

The empirical structure function changes over time:

```text
C_{t+1}^*(R) != C_t^*(R)
```

### 10.6 Non-Collapse

The archive maintains multiple behavioral strategies under different lambda values or ecological niches.

## 11. Experimental Program

### Experiment 1: Closed-World Baseline

Use a fixed set of foraging maps.

Prediction:

- Performance improves quickly.
- Complexity stabilizes or shrinks.
- Novelty vanishes.

This confirms that fixed benchmarks are not open-ended.

### Experiment 2: Frontier Curriculum

Allow environment mutations near the competence boundary.

Prediction:

- Performance cycles through solve-expand-solve phases.
- The archive grows.
- New validation worlds continue to separate lineages.

### Experiment 3: Lambda Phase Diagram

Run multiple lambda values at each ecological epoch.

Prediction:

- Low lambda agents discover high-complexity behaviors.
- High lambda agents compress solved strategies.
- Mid lambda agents often dominate validation.
- Phase transitions appear when new behavioral modules become worth their complexity.

### Experiment 4: Coevolution

Introduce multiple interacting populations.

Prediction:

- Stationary optima become unstable.
- Strategies create counter-strategies.
- The archive becomes necessary to prevent forgetting and cycling.

### Experiment 5: Exported Automata

Export evolved automata as readable sparse rule sets and minimal executable policies.

Prediction:

- Some lineages produce interpretable subroutines: wall following, trail use, resource sweeping, evasion.
- Compression pressure makes these subroutines smaller over time.

## 12. Proposed Thesis Statement

The thesis can be stated as:

> Open-ended artificial evolution is possible under a free-energy paradigm if free energy is used as a local selection principle over agents embedded in an expanding, archive-driven ecology. Fixed-task free-energy minimization converges to compression; open-endedness requires that solved structures generate new validation pressures. Lambda sweeps then reveal evolving loss-complexity frontiers, and sustained frontier movement provides the empirical signature of open-ended evolution.

## 13. Falsification Criteria

The thesis is wrong, or at least incomplete, if:

1. Frontier-generated environments collapse into noise or triviality.
2. Complexity grows without transferable risk reduction.
3. New agents do not outperform the archive on genuinely new validation worlds.
4. Lambda sweeps do not reveal structural phase transitions.
5. Apparent novelty is just overfitting to the generator.
6. The system cannot maintain multiple lineages without manual intervention.

These failure modes should be treated as research results, not engineering annoyances.

## 14. Immediate Next Step

The next implementation should be a research instrument, not a product:

```text
evolve automata -> evaluate frontier -> mutate environments -> archive lineages -> sweep lambda -> render behavior
```

The output should include:

- replay videos or ASCII traces,
- lineage graph,
- archive coverage,
- loss-complexity frontier plots,
- environment genealogy,
- exported automata,
- novelty and transfer metrics.

The goal is not to make a clever game agent. The goal is to make open-endedness, or its failure, scientifically visible.

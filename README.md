# GKM: Open-Ended Evolution Under Free Energy

This repository is focused on a research thesis and small sparse-FSA evolution
substrates.

The thesis:

> Open-ended artificial evolution is possible under a free-energy paradigm if free energy is used as a local selection principle over agents embedded in an expanding, archive-driven ecology. Fixed-task free-energy minimization converges to compression; open-endedness requires that solved structures generate new validation pressures.

See [OPEN_ENDED_EVOLUTION_THESIS.md](OPEN_ENDED_EVOLUTION_THESIS.md) for the full argument. The Sphinx documentation source is in [docs/](docs/) and can be deployed through the included GitHub Pages workflow.
There is also a research-style LaTeX manuscript in [manuscript/](manuscript/) focused on free-energy selection for predicate encapsulation in sparse deterministic solvers.

## Current Research Substrates

### Grid Foraging

The first experiment is a sparse finite-state automaton ecology:

- Agents are deterministic FSAs with a sparse encoded transition relation.
- Genomes mutate by explicit rule edits, additions, and deletions. A rule may contain a short sequence of moves, not only a single move.
- Agents play a visible grid foraging game.
- Selection uses an explicit free-energy objective:

```text
F_lambda(a) = R(a) + lambda C(a)
```

where:

- `R(a)` is the loss function: missed resources plus small step and collision costs.
- `C(a)` is the selected raw description-length function. By default this is encoded rule-set size: the sum of the complexities of every rule the genome carries, so extra rules and long macro-rules are paid for. The runner can also optimize active, pruned, or mixed complexity for comparison.
- `lambda` controls pressure toward compact policies.

Complexity modes:

- `active`: sum of complexities for rules actually used in observed episodes.
- `table`: sum of complexities for the whole encoded sparse rule set. This is the default. The name is historical; it now means encoded rule-set size, not a dense transition table.
- `pruned`: all transition rules reachable from state 0 under any possible observation.
- `mixed`: active complexity plus a dead-code tax from unused encoded rules.

The FSA input is:

```text
(current_state, previous_move, relative_food_azimuth) -> (move_sequence, next_state)
```

If no encoded rule matches an input, the automaton halts the episode. There is
no random fallback behavior and no free default movement rule.

The runner sweeps `lambda` to trace a loss-complexity landscape, following the structure-function/free-energy viewpoint in [arXiv:2507.13543](https://arxiv.org/abs/2507.13543). The paper supplies the loss-complexity/free-energy lens; this repository applies that lens to a toy evolutionary substrate.

This is not intended as a final open-ended system. It is a controllable instrument for studying the first necessary pieces: heritable structure, visible behavior, complexity pressure, replayable lineages, and eventually frontier-generating environments.

### Pattern Transduction

The second experiment asks a different question:

```text
Can a meta-evolutionary process synthesize compact deterministic solvers from
observed pattern transitions?
```

The task format is foreign-object transduction:

```text
train:      [A B] -> [B A], [C D] -> [D C]
validation: [X Y] -> [Y X]
hidden test: [p q] -> [q p]
```

The evolved object is a sparse deterministic register transducer. Token
identities are opaque: the rule key does not see `A`, `B`, or `X`. It sees only
finite control and relational observations:

```text
(state, TOKEN | EOS | BOS | MATCH_REGISTER_MASK) -> (action_sequence, next_state)
```

Primitive sets are intentionally tiered:

- `stream`: move right, write current token, halt;
- `register`: stream primitives plus store/write register actions;
- `compare`: register primitives plus equality observations between the current token and stored registers;
- `bidirectional`: stream primitives plus `MOVE_LEFT` and a beginning-of-sequence observation;
- `bidirectional_compare`: bidirectional motion plus register equality observations.

This lets us ask which primitive set is sufficient for a task family, while
selection still minimizes:

```text
F_lambda(solver) = training_loss(solver) + lambda C(solver)
```

Evolution uses this training free energy as the local selection rule. The
runner then sweeps `lambda` and selects a solver from the validation
loss-complexity Pareto frontier: it finds the best validation loss, then keeps
the simplest Pareto solver within a small validation-loss tolerance. The hidden
test transition is evaluated only after this validation selection.

The goal is not to build a general ARC solver. The goal is to study a
meta-model that produces compact deterministic solvers when the pattern family
is deterministic enough.

### ARC-AGI-3 Cracking via a Self-Improving GKM Agent

The third substrate (scratch lab in [crack_lab/](crack_lab/)) pushes the same
free-energy/GKM viewpoint onto live ARC-AGI-3 keyboard games, which run locally and
offline. The question here is:

```text
Can an agent figure a game out ON ITS OWN -- discover its perception, its mechanics,
its goal, and a winning strategy -- from the rawest interface, carrying only general
human preconceptions, with free energy as the selection principle?
```

The design enforces one boundary, because it is the only one that transfers to a
*different type* of game: the engine exposes nothing game-specific. The agent gets
only `step(action) -> frame` (a 64x64 grid of colour integers), the reward
(`levels_completed`), and a `clone()` for safe lookahead. Everything else --
finding the avatar, learning the manipulation mechanic, locating the goal region,
modelling other agents, planning -- must be discovered and written by the agent
itself. The agent is an LLM **proposer** that writes a `solve(env)` program, plus:

- a rich **human-preconception** system prompt (objects/space/barriers, agency and
  theory-of-mind, reachability, cooperation, sparse-reward self-objectives,
  affordance discovery);
- **free-energy admission**: a proposed program is kept only if it lowers
  `F = R + lambda C` on the real game (reward vs. program description length), with
  `lambda` small so parsimony tie-breaks rather than stifles novelty, plus
  compression-progress / disagreement as curiosity signals that *steer* novelty
  (selection can price novelty but is never its source -- that is the proposer's
  job), and replay-preservation;
- the simulator as the **ground-truth verifier** (every result is replay-validated
  on a fresh environment).

The proposer is pluggable: a **local** model (offline, eval-legal, but currently too
weak) or the **Claude Code agent** invoked headlessly with tools + a tester, so it
writes a program, runs it on the real game, sees failures, and iterates -- a strong
proposer, but it uses the network/API, so it is a demonstration / upper bound, not
the offline-eval path.

Result so far: on the game `wa30` the strong proposer wrote (and iterated into) an
adaptive `solve(env)` that cracks **Levels 1, 2 and 3**, replay-validated. It
independently rediscovered the non-obvious tricks -- freezing the target region at
level start (delivered objects change colour and vanish from naive detection),
complementing an autonomous helper agent by taking the farthest objects, and
relaying objects across a dividing wall via an asymmetric carry-collision -- with no
hand-coded strategy. It honestly stops at `wa30` Level 4 (a large escalation). Pointed
at a **different** game (`ls20`, a slide-to-match mechanic, not carry), the same agent
cracked its **Levels 1 through 4**, replay-validated -- the generalisation the rawest
substrate is designed for.

This line is documented in full on the **[Self-Improving Agent](docs/self_improving_agent.rst)**
page (deployed at <https://sashakolpakov.github.io/gkm/>), which is the authoritative
narrative; the chronological lab account, including honest negatives (a
system-prompt-only strategist mis-reasons two-sided reachability; the local model is
too weak), is in [crack_lab/FINDINGS.md](crack_lab/FINDINGS.md).

## Run

Run the foraging ecology experiment:

```bash
python3 experiments/run_foraging_ecology.py --generations 80 --population 160 --render
```

Compatibility entry point:

```bash
python agent.py --generations 80 --population 160 --render
```

Run the pattern-transduction substrate:

```bash
python pattern_fsa.py --task swap --primitive-set register --generations 120 --population 220 --lambda-points 4
```

Reproduce the register-transducer benchmark matrix:

```bash
python3 experiments/run_register_transducer_benchmark.py
```

Run the local symbolic Bongard-style concept-induction harness:

```bash
python3 experiments/run_bongard_symbolic_baseline.py
```

Run the evolved sparse Bongard classifier harness with clean-slate random initialization, concept-specific search budgets, counterexample-rich splits, and exhaustive discovery probes:

```bash
python3 experiments/run_bongard_sparse_classifier.py --concept length_even
python3 experiments/run_bongard_sparse_classifier.py --concept has_adjacent_duplicate
python3 experiments/run_bongard_sparse_classifier.py --concept first_equals_last
python3 experiments/run_bongard_sparse_classifier.py --concept length_multiple_of_three
python3 experiments/run_bongard_sparse_classifier.py --concept first_equals_second
python3 experiments/run_bongard_sparse_classifier.py --concept last_two_equal
python3 experiments/run_bongard_sparse_classifier.py --concept second_equals_last
python3 experiments/run_bongard_sparse_classifier.py --concept first_equals_penultimate
python3 experiments/run_bongard_sparse_classifier.py --concept second_equals_penultimate
python3 experiments/run_bongard_sparse_classifier.py --concept palindrome
python3 experiments/run_bongard_sparse_classifier.py --concept contains_duplicate
python3 experiments/run_bongard_sparse_classifier.py --concept all_unique
```

Override a concept budget explicitly:

```bash
python3 experiments/run_bongard_sparse_classifier.py --concept first_equals_last --replicates 1 --population 700 --generations 450 --states 3 --initial-rules 6 --max-rules 12 --max-rule-length 1 --lambda-min 0.0001 --lambda-max 0.0001 --lambda-points 1 --train-count 16 --validation-count 12 --hidden-count 32 --mutation-rate 0.12 --lambda-warmup-fraction 0.9 --archive-training --archive-interval 40 --archive-add-per-interval 32 --stop-after-discovery
```

Run paired overcapacity ablations across a fast Bongard rule matrix:

```bash
python3 -u experiments/run_bongard_overcapacity_ablation.py --replicates 1
```

Run the local Bongard-LOGO symbolic adapter without vendoring external data:

```bash
git clone https://github.com/NVlabs/Bongard-LOGO.git downloads/Bongard-LOGO
.venv/bin/python -m pip install pillow pandas
.venv/bin/python experiments/run_bongard_logo_adapter.py --dataset-dir downloads/Bongard-LOGO --source both --feature-set both --limit 40 --support-count 10 --validation-count 3 --hidden-count 3 --summary-only
.venv/bin/python experiments/run_bongard_logo_adapter.py --dataset-dir downloads/Bongard-LOGO --source abstract --feature-set all --limit 26 --support-count 10 --validation-count 3 --hidden-count 3 --max-rule-atoms 2 --max-candidate-atoms 20 --summary-only
```

Run the internal abstraction-emergence predicate-library scaffold:

```bash
python3 experiments/run_abstraction_emergence.py
python3 experiments/run_abstraction_emergence.py --scenario multi --show-rules
python3 experiments/run_abstraction_emergence.py --scenario or_factor --show-rules
```

Use Hyperopt/TPE instead of the genetic population loop:

```bash
pip install -r requirements.txt
python3 experiments/run_foraging_ecology.py --optimizer hyperopt --hyperopt-evals 300 --lambda-points 5
```

Run a larger ecology with a less cramped episode horizon:

```bash
python3 experiments/run_foraging_ecology.py --width 10 --height 10 --food-count 6 --max-steps 80 --max-rule-length 3 --initial-rules 100 --max-rules 180 --generations 150 --population 260 --lambda-max 0.0012
```

Compare complexity assumptions:

```bash
python3 experiments/run_foraging_ecology.py --complexity-mode active --lambda-points 5
python3 experiments/run_foraging_ecology.py --complexity-mode table --lambda-points 5
python3 experiments/run_foraging_ecology.py --complexity-mode pruned --lambda-points 5
python3 experiments/run_foraging_ecology.py --complexity-mode mixed --lambda-points 5
```

Outputs are written to `output/evo_game/`:

```text
best_automaton.py        exported evolved policy
evolution_history.json   generation metrics
lambda_sweep.json        per-lambda loss/complexity/free-energy records, including all complexity metrics
summary.json             train/validation summary
best_replay.txt          ASCII replay of the final best policy
```

Pattern-transduction outputs are written to `output/pattern_fsa/` by default:

```text
solver.json              selected sparse register transducer
history.json             per-generation training and validation metrics
lambda_sweep.json        per-lambda validation frontier records
summary.json             selected train/validation/hidden-test evaluation
```

Crack a local ARC-AGI-3 game with the self-improving GKM agent (all offline except the
networked strong-proposer demo):

```bash
# discovered-connector cone cracker (cracks wa30 L1/L2 from scratch, no network):
python3 crack_lab/gkm_crack.py wa30 --no-llm
# rawest agent substrate, local-model proposer (offline, eval-legal):
python3 crack_lab/gkm_arena.py --game=wa30 --proposer=ollama --rounds=8
# strong proposer = the Claude Code agent with discovered context + tools + tester
# (networked; demonstration / upper bound, not the offline-eval path):
python3 crack_lab/gkm_solve_agent.py --game=wa30 --minutes=40
```

Local ARC game sources live under `environment_files/` (downloaded once with a key,
then run locally and key-free; gitignored). See [crack_lab/FINDINGS.md](crack_lab/FINDINGS.md)
and [crack_lab/PLAN.md](crack_lab/PLAN.md).

## Research Direction

The intended direction is:

```text
automata -> interaction -> archive -> frontier environments -> lambda sweeps -> lineage analysis
```

The goal is not to make a clever game bot. The goal is to test whether free-energy selection can support continued structural innovation when the ecology itself expands.

## Near-Term Experiments

1. **Closed-world baseline**
   Evolve automata on fixed maps. Prediction: rapid improvement, then stagnation/compression.

2. **Frontier curriculum**
   Generate new maps near the competence boundary. Prediction: solve-expand-solve cycles.

3. **Lambda phase diagram**
   Sweep `lambda` and measure different loss-complexity frontiers. `lambda_sweep.json` includes `complexity_variance`, a susceptibility-style diagnostic for phase-transition candidates.

4. **Archive validation**
   Keep environments that distinguish lineages. Prediction: multiple strategies persist.

5. **Coevolution**
   Add other agents, resource depletion, markers, or adversaries. Prediction: new niches create new selection pressures.

## Files

```text
agent.py                         compatibility entry point for experiments/run_foraging_ecology.py
evo_game.py                      grid-foraging FSA substrate library
experiments/run_foraging_ecology.py
                                 grid-foraging experiment runner
pattern_fsa.py                   sparse register-transducer pattern experiment
experiments/register_transducer_benchmark.md
                                 register-transducer benchmark report
experiments/run_register_transducer_benchmark.py
                                 benchmark reproduction script
experiments/bongard_first_plan.md
                                 Bongard-first benchmark plan
experiments/run_bongard_symbolic_baseline.py
                                 symbolic Bongard-style baseline
experiments/run_bongard_sparse_classifier.py
                                 evolved sparse Bongard classifier
experiments/run_bongard_overcapacity_ablation.py
                                 paired Bongard overcapacity ablations
experiments/run_bongard_logo_adapter.py
                                 local Bongard-LOGO symbolic adapter and selector
experiments/run_abstraction_emergence.py
                                 internal predicate-library abstraction scaffold
experiments/bongard_sparse_classifier_report.md
                                 sparse Bongard classifier report
experiments/bongard_logo_report.md
                                 Bongard-LOGO symbolic adapter report
experiments/abstraction_emergence_report.md
                                 abstraction-emergence predicate-library report
experiments/abstraction_related_work.md
                                 abstraction-emergence related-work note
OPEN_ENDED_EVOLUTION_THESIS.md   thesis and experimental program
FREE_ENERGY_EXPLANATION.md       mathematical background
tests/test_evo_game.py           standard-library tests
tests/test_pattern_fsa.py        pattern-transduction tests
tests/test_bongard_sparse_classifier.py
                                 Bongard harness tests
tests/test_abstraction_emergence.py
                                 abstraction-emergence tests
requirements.txt                 optional Hyperopt/TPE dependency
```

## Documentation

Build the Sphinx documentation locally:

```bash
python3 -m sphinx -W -b html docs docs/_build/html
```

The repository includes `.github/workflows/pages.yml`, which builds `docs/` and deploys `docs/_build/html` through GitHub Pages when the `manuscript` branch is pushed. In the repository settings, set Pages to use GitHub Actions as the source.

## Tests

```bash
python -m unittest
python -m py_compile agent.py evo_game.py pattern_fsa.py experiments/run_foraging_ecology.py experiments/run_register_transducer_benchmark.py experiments/run_bongard_symbolic_baseline.py experiments/run_bongard_sparse_classifier.py experiments/run_bongard_overcapacity_ablation.py experiments/run_bongard_logo_adapter.py experiments/run_abstraction_emergence.py tests/test_evo_game.py tests/test_pattern_fsa.py tests/test_bongard_sparse_classifier.py tests/test_abstraction_emergence.py
```

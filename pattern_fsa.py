"""
Sparse register-transducer synthesis for deterministic pattern-transduction tasks.

The meta-model evolves sparse deterministic transducers over opaque objects.
The rule key never sees a token's literal identity. It sees only finite control
and relational observations such as TOKEN, EOS, or current-token-equals-register.
This makes foreign-alphabet generalization possible and makes literal lookup
tables unavailable as a shortcut.
"""

import argparse
import json
import math
import os
import random
import statistics
import textwrap
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple


Symbol = int
Observation = int
Action = int
RuleKey = Tuple[int, Observation]
SequencePair = Tuple[Tuple[Symbol, ...], Tuple[Symbol, ...]]

MOVE_RIGHT = 0
WRITE_CURRENT = 1
HALT = 2
STORE_REGISTER_BASE = 10
WRITE_REGISTER_BASE = 30

OBS_EOS = 0
OBS_TOKEN = 1
OBS_MATCH_BASE = 2

ACTION_NAMES = {
    MOVE_RIGHT: "MOVE_RIGHT",
    WRITE_CURRENT: "WRITE_CURRENT",
    HALT: "HALT",
}


@dataclass(frozen=True)
class PrimitiveSet:
    name: str
    actions: Tuple[Action, ...]
    register_count: int = 0
    compare_registers: bool = False

    def allows(self, action: Action, alphabet_size: int) -> bool:
        return action in self.actions

    @property
    def observation_count(self) -> int:
        if not self.compare_registers or self.register_count <= 0:
            return 2
        return OBS_MATCH_BASE + (2**self.register_count - 1)

    def observe(self, cursor: int, input_sequence: Sequence[Symbol], registers: Sequence[Optional[Symbol]]) -> Observation:
        if cursor < 0 or cursor >= len(input_sequence):
            return OBS_EOS
        if not self.compare_registers:
            return OBS_TOKEN
        current = input_sequence[cursor]
        mask = 0
        for idx, value in enumerate(registers[: self.register_count]):
            if value is not None and value == current:
                mask |= 1 << idx
        if mask == 0:
            return OBS_TOKEN
        return OBS_MATCH_BASE + mask - 1


def register_store_action(register_index: int) -> Action:
    return STORE_REGISTER_BASE + register_index


def register_write_action(register_index: int) -> Action:
    return WRITE_REGISTER_BASE + register_index


def register_match_observation(register_indices: Sequence[int]) -> Observation:
    mask = 0
    for idx in register_indices:
        mask |= 1 << idx
    if mask == 0:
        return OBS_TOKEN
    return OBS_MATCH_BASE + mask - 1


def stream_primitives(alphabet_size: int, register_count: int = 0) -> PrimitiveSet:
    return PrimitiveSet(
        name="stream",
        actions=(MOVE_RIGHT, WRITE_CURRENT, HALT),
        register_count=0,
        compare_registers=False,
    )


def register_primitives(alphabet_size: int, register_count: int = 2) -> PrimitiveSet:
    register_actions = tuple(
        action
        for idx in range(register_count)
        for action in (register_store_action(idx), register_write_action(idx))
    )
    return PrimitiveSet(
        name="register",
        actions=(MOVE_RIGHT, WRITE_CURRENT, HALT) + register_actions,
        register_count=register_count,
        compare_registers=False,
    )


def compare_primitives(alphabet_size: int, register_count: int = 2) -> PrimitiveSet:
    base = register_primitives(alphabet_size, register_count=register_count)
    return PrimitiveSet(
        name="compare",
        actions=base.actions,
        register_count=register_count,
        compare_registers=True,
    )


PRIMITIVE_SETS = {
    "stream": stream_primitives,
    "register": register_primitives,
    "compare": compare_primitives,
}


@dataclass(frozen=True, init=False)
class PatternRule:
    state: int
    observation: Observation
    actions: Tuple[Action, ...]
    next_state: int

    def __init__(
        self,
        state: int,
        observation: Observation,
        actions: Sequence[Action],
        next_state: int,
    ) -> None:
        if not actions:
            raise ValueError("PatternRule actions cannot be empty")
        object.__setattr__(self, "state", int(state))
        object.__setattr__(self, "observation", int(observation))
        object.__setattr__(self, "actions", tuple(int(action) for action in actions))
        object.__setattr__(self, "next_state", int(next_state))

    @property
    def key(self) -> RuleKey:
        return (self.state, self.observation)


@dataclass
class PatternGenome:
    state_count: int
    alphabet_size: int
    rules: List[PatternRule]

    def valid_rule(self, rule: PatternRule, primitives: PrimitiveSet) -> bool:
        return (
            0 <= rule.state < self.state_count
            and 0 <= rule.next_state < self.state_count
            and 0 <= rule.observation < primitives.observation_count
            and all(primitives.allows(action, self.alphabet_size) for action in rule.actions)
        )

    def rule_map(self, primitives: PrimitiveSet) -> Dict[RuleKey, PatternRule]:
        return {rule.key: rule for rule in self.rules if self.valid_rule(rule, primitives)}

    @classmethod
    def random(
        cls,
        rng: random.Random,
        state_count: int,
        alphabet_size: int,
        primitives: PrimitiveSet,
        initial_rule_count: int,
        max_rule_length: int,
    ) -> "PatternGenome":
        keys = rng.sample(
            all_rule_keys(state_count, primitives.observation_count),
            k=min(initial_rule_count, state_count * primitives.observation_count),
        )
        rules = [
            random_rule(rng, key, state_count, alphabet_size, primitives, max_rule_length)
            for key in keys
        ]
        return cls(state_count=state_count, alphabet_size=alphabet_size, rules=deduplicate_rules(rules))

    def crossover(self, other: "PatternGenome", rng: random.Random, primitives: PrimitiveSet) -> "PatternGenome":
        state_count = max(self.state_count, other.state_count)
        rule_map_a = self.rule_map(primitives)
        rule_map_b = other.rule_map(primitives)
        rules = []
        for key in sorted(set(rule_map_a) | set(rule_map_b)):
            if key in rule_map_a and key in rule_map_b:
                rules.append(rule_map_a[key] if rng.random() < 0.5 else rule_map_b[key])
            elif key in rule_map_a:
                if rng.random() < 0.5:
                    rules.append(rule_map_a[key])
            elif rng.random() < 0.5:
                rules.append(rule_map_b[key])
        return PatternGenome(state_count=state_count, alphabet_size=self.alphabet_size, rules=deduplicate_rules(rules))

    def mutate(
        self,
        rng: random.Random,
        primitives: PrimitiveSet,
        rate: float,
        max_rule_length: int,
        max_rules: int,
        max_states: int,
    ) -> "PatternGenome":
        state_count = self.state_count
        if state_count < max_states and rng.random() < rate * 0.25:
            state_count += 1

        rules: List[Optional[PatternRule]] = [
            rule for rule in self.rules if rule.state < state_count and rule.next_state < state_count
        ]
        for idx, rule in enumerate(list(rules)):
            if rule is None or rng.random() >= rate:
                continue

            actions = list(rule.actions)
            edit = rng.random()
            if edit < 0.45:
                actions[rng.randrange(len(actions))] = rng.choice(primitives.actions)
            elif edit < 0.62 and len(actions) < max_rule_length:
                actions.insert(rng.randrange(len(actions) + 1), rng.choice(primitives.actions))
            elif edit < 0.74 and len(actions) > 1:
                del actions[rng.randrange(len(actions))]
            elif edit < 0.86:
                rules[idx] = PatternRule(rule.state, rule.observation, actions, rng.randrange(state_count))
                continue
            elif edit < 0.94:
                rules[idx] = random_rule(
                    rng,
                    random_rule_key(rng, state_count, primitives.observation_count),
                    state_count,
                    self.alphabet_size,
                    primitives,
                    max_rule_length,
                )
                continue
            else:
                rules[idx] = None
                continue

            rules[idx] = PatternRule(rule.state, rule.observation, actions, rule.next_state)

        kept = deduplicate_rules([rule for rule in rules if rule is not None])
        for _ in range(1 + int(rate * max(1, len(kept)))):
            if len(kept) >= max_rules or rng.random() >= rate * 2.0:
                continue
            existing = {rule.key for rule in kept}
            available = [key for key in all_rule_keys(state_count, primitives.observation_count) if key not in existing]
            if not available:
                break
            kept.append(
                random_rule(
                    rng,
                    rng.choice(available),
                    state_count,
                    self.alphabet_size,
                    primitives,
                    max_rule_length,
                )
            )

        return PatternGenome(
            state_count=state_count,
            alphabet_size=self.alphabet_size,
            rules=deduplicate_rules(kept)[:max_rules],
        )


@dataclass(frozen=True)
class PatternTask:
    name: str
    alphabet_size: int
    train_pairs: Tuple[SequencePair, ...]
    val_pairs: Tuple[SequencePair, ...]
    test_pairs: Tuple[SequencePair, ...]


@dataclass
class PatternRun:
    output: Tuple[Symbol, ...]
    halted: bool
    steps: int
    active_rules: Tuple[RuleKey, ...]


@dataclass
class PatternEvaluation:
    fitness: float
    loss: float
    free_energy: float
    lambda_value: float
    complexity: float
    pair_count: int
    mean_edit_loss: float
    exact_match_rate: float
    active_rules: int
    encoded_rules: int


@dataclass
class PatternGenerationRecord:
    generation: int
    lambda_value: float
    best_loss: float
    best_free_energy: float
    train_exact_match: float
    val_exact_match: float
    complexity: float
    active_rules: int
    encoded_rules: int


@dataclass
class PatternLambdaRecord:
    lambda_value: float
    train_loss: float
    val_loss: float
    train_exact_match: float
    val_exact_match: float
    train_free_energy: float
    val_free_energy: float
    complexity: float
    active_rules: int
    encoded_rules: int
    test_loss: Optional[float] = None
    test_exact_match: Optional[float] = None
    test_free_energy: Optional[float] = None
    elbow_score: Optional[float] = None
    selected: bool = False


def all_rule_keys(state_count: int, observation_count: int) -> List[RuleKey]:
    return [(state, obs) for state in range(state_count) for obs in range(observation_count)]


def random_rule_key(rng: random.Random, state_count: int, observation_count: int) -> RuleKey:
    return rng.randrange(state_count), rng.randrange(observation_count)


def random_rule(
    rng: random.Random,
    key: RuleKey,
    state_count: int,
    alphabet_size: int,
    primitives: PrimitiveSet,
    max_rule_length: int,
) -> PatternRule:
    length = rng.randrange(1, max(1, max_rule_length) + 1)
    return PatternRule(
        state=key[0],
        observation=key[1],
        actions=tuple(rng.choice(primitives.actions) for _ in range(length)),
        next_state=rng.randrange(state_count),
    )


def deduplicate_rules(rules: Sequence[PatternRule]) -> List[PatternRule]:
    by_key = {rule.key: rule for rule in rules}
    return [by_key[key] for key in sorted(by_key)]


def action_name(action: Action, alphabet_size: int) -> str:
    if action in ACTION_NAMES:
        return ACTION_NAMES[action]
    if STORE_REGISTER_BASE <= action < WRITE_REGISTER_BASE:
        return f"STORE_R{action - STORE_REGISTER_BASE}"
    if WRITE_REGISTER_BASE <= action:
        return f"WRITE_R{action - WRITE_REGISTER_BASE}"
    return f"UNKNOWN_{action}"


def observation_name(observation: Observation, register_count: int) -> str:
    if observation == OBS_EOS:
        return "EOS"
    if observation == OBS_TOKEN:
        return "TOKEN"
    mask = observation - OBS_MATCH_BASE + 1
    registers = [f"R{idx}" for idx in range(register_count) if mask & (1 << idx)]
    return "MATCH_" + "_".join(registers) if registers else f"OBS_{observation}"


def rule_complexity(rule: PatternRule) -> float:
    return len(rule.actions) + 1.0


def genome_complexity(genome: PatternGenome, primitives: PrimitiveSet) -> float:
    return sum(rule_complexity(rule) for rule in genome.rule_map(primitives).values())


def run_transducer(
    genome: PatternGenome,
    input_sequence: Sequence[Symbol],
    primitives: PrimitiveSet,
    max_steps: int = 64,
    max_output_length: int = 64,
) -> PatternRun:
    state = 0
    cursor = 0
    output: List[Symbol] = []
    steps = 0
    active_rules: List[RuleKey] = []
    registers: List[Optional[Symbol]] = [None] * primitives.register_count
    rule_map = genome.rule_map(primitives)

    while steps < max_steps and len(output) < max_output_length:
        observation = primitives.observe(cursor, input_sequence, registers)
        key = (state, observation)
        rule = rule_map.get(key)
        if rule is None:
            return PatternRun(tuple(output), halted=True, steps=steps, active_rules=tuple(active_rules))

        active_rules.append(key)
        for action in rule.actions:
            if steps >= max_steps or len(output) >= max_output_length:
                break
            steps += 1

            if action == MOVE_RIGHT:
                cursor = min(cursor + 1, len(input_sequence))
            elif action == WRITE_CURRENT and cursor < len(input_sequence):
                output.append(input_sequence[cursor])
            elif STORE_REGISTER_BASE <= action < STORE_REGISTER_BASE + primitives.register_count:
                if cursor < len(input_sequence):
                    registers[action - STORE_REGISTER_BASE] = input_sequence[cursor]
            elif WRITE_REGISTER_BASE <= action < WRITE_REGISTER_BASE + primitives.register_count:
                value = registers[action - WRITE_REGISTER_BASE]
                if value is not None:
                    output.append(value)
            elif action == HALT:
                return PatternRun(tuple(output), halted=True, steps=steps, active_rules=tuple(active_rules))

        state = rule.next_state

    return PatternRun(tuple(output), halted=False, steps=steps, active_rules=tuple(active_rules))


def evaluate_genome(
    genome: PatternGenome,
    pairs: Sequence[SequencePair],
    primitives: PrimitiveSet,
    lambda_value: float,
    max_steps: int = 64,
) -> PatternEvaluation:
    runs = [run_transducer(genome, source, primitives, max_steps=max_steps) for source, _target in pairs]
    losses = [
        normalized_edit_distance(run.output, target)
        for run, (_source, target) in zip(runs, pairs)
    ]
    exact = [1.0 if run.output == target else 0.0 for run, (_source, target) in zip(runs, pairs)]
    active_rules = {key for run in runs for key in run.active_rules}
    complexity = genome_complexity(genome, primitives)
    loss = statistics.mean(losses) if losses else 1.0
    free_energy = loss + lambda_value * complexity
    return PatternEvaluation(
        fitness=-free_energy,
        loss=loss,
        free_energy=free_energy,
        lambda_value=lambda_value,
        complexity=complexity,
        pair_count=len(pairs),
        mean_edit_loss=loss,
        exact_match_rate=statistics.mean(exact) if exact else 0.0,
        active_rules=len(active_rules),
        encoded_rules=len(genome.rule_map(primitives)),
    )


def evolve_solver(
    task: PatternTask,
    primitives: PrimitiveSet,
    seed: int = 7,
    generations: int = 80,
    population_size: int = 160,
    state_count: int = 4,
    max_states: Optional[int] = None,
    initial_rule_count: int = 12,
    max_rules: int = 64,
    max_rule_length: int = 3,
    mutation_rate: float = 0.08,
    lambda_value: float = 0.002,
    max_steps: int = 64,
    report_every: int = 20,
) -> Tuple[PatternGenome, List[PatternGenerationRecord], PatternEvaluation, PatternEvaluation]:
    rng = random.Random(seed)
    max_states = max_states or state_count
    population = [
        PatternGenome.random(
            rng,
            state_count=state_count,
            alphabet_size=task.alphabet_size,
            primitives=primitives,
            initial_rule_count=initial_rule_count,
            max_rule_length=max_rule_length,
        )
        for _ in range(population_size)
    ]
    history: List[PatternGenerationRecord] = []
    best = population[0]

    for generation in range(generations + 1):
        evaluations = [
            (evaluate_genome(genome, task.train_pairs, primitives, lambda_value, max_steps=max_steps), genome)
            for genome in population
        ]
        evaluations.sort(key=lambda item: item[0].fitness, reverse=True)
        train_eval, best = evaluations[0]
        val_eval = evaluate_genome(best, task.val_pairs, primitives, lambda_value, max_steps=max_steps)
        history.append(
            PatternGenerationRecord(
                generation=generation,
                lambda_value=lambda_value,
                best_loss=train_eval.loss,
                best_free_energy=train_eval.free_energy,
                train_exact_match=train_eval.exact_match_rate,
                val_exact_match=val_eval.exact_match_rate,
                complexity=train_eval.complexity,
                active_rules=train_eval.active_rules,
                encoded_rules=train_eval.encoded_rules,
            )
        )

        if report_every and (generation == 0 or generation == generations or generation % report_every == 0):
            print_pattern_generation(history[-1])

        if generation == generations:
            break

        elite_count = max(1, population_size // 10)
        next_population = [genome for _eval, genome in evaluations[:elite_count]]
        scored = [(evaluation.fitness, genome) for evaluation, genome in evaluations]
        while len(next_population) < population_size:
            parent_a = tournament_select(scored, rng)
            parent_b = tournament_select(scored, rng)
            child = parent_a.crossover(parent_b, rng, primitives).mutate(
                rng,
                primitives,
                rate=mutation_rate,
                max_rule_length=max_rule_length,
                max_rules=max_rules,
                max_states=max_states,
            )
            next_population.append(child)
        population = next_population

    train_eval = evaluate_genome(best, task.train_pairs, primitives, lambda_value, max_steps=max_steps)
    val_eval = evaluate_genome(best, task.val_pairs, primitives, lambda_value, max_steps=max_steps)
    return best, history, train_eval, val_eval


def tournament_select(scored: Sequence[Tuple[float, PatternGenome]], rng: random.Random, size: int = 5) -> PatternGenome:
    contenders = rng.sample(list(scored), k=min(size, len(scored)))
    return max(contenders, key=lambda item: item[0])[1]


def make_lambda_values(lambda_min: float, lambda_max: float, lambda_points: int) -> List[float]:
    if lambda_points <= 1:
        return [lambda_min]
    step = (lambda_max - lambda_min) / (lambda_points - 1)
    return [lambda_min + idx * step for idx in range(lambda_points)]


def pareto_front(records: Sequence[PatternLambdaRecord]) -> List[PatternLambdaRecord]:
    front: List[PatternLambdaRecord] = []
    for record in records:
        dominated = False
        for other in records:
            if other is record:
                continue
            no_worse = other.val_loss <= record.val_loss and other.complexity <= record.complexity
            strictly_better = other.val_loss < record.val_loss or other.complexity < record.complexity
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(record)
    return sorted(front, key=lambda item: (item.complexity, item.val_loss, item.lambda_value))


def assign_validation_elbow_scores(records: Sequence[PatternLambdaRecord]) -> None:
    front = pareto_front(records)
    for record in records:
        record.elbow_score = None
        record.selected = False
    if not front:
        return

    min_loss = min(record.val_loss for record in front)
    max_loss = max(record.val_loss for record in front)
    min_complexity = min(record.complexity for record in front)
    max_complexity = max(record.complexity for record in front)
    loss_span = max(max_loss - min_loss, 1e-12)
    complexity_span = max(max_complexity - min_complexity, 1e-12)

    for record in front:
        normalized_loss = (record.val_loss - min_loss) / loss_span
        normalized_complexity = (record.complexity - min_complexity) / complexity_span
        record.elbow_score = math.hypot(normalized_loss, normalized_complexity)


def validation_elbow(
    records: Sequence[PatternLambdaRecord],
    loss_tolerance: float = 0.075,
) -> PatternLambdaRecord:
    if not records:
        raise ValueError("cannot select a validation elbow from no records")
    assign_validation_elbow_scores(records)
    front = pareto_front(records)
    best_loss = min(record.val_loss for record in front)
    max_allowed_loss = best_loss + max(0.0, loss_tolerance)
    candidates = [record for record in front if record.val_loss <= max_allowed_loss]
    selected = min(
        candidates,
        key=lambda item: (
            item.complexity,
            item.val_loss,
            math.inf if item.elbow_score is None else item.elbow_score,
        ),
    )
    selected.selected = True
    return selected


def lambda_sweep_solver(
    task: PatternTask,
    primitives: PrimitiveSet,
    lambda_values: Sequence[float],
    seed: int = 7,
    generations: int = 80,
    population_size: int = 160,
    state_count: int = 4,
    max_states: Optional[int] = None,
    initial_rule_count: int = 12,
    max_rules: int = 64,
    max_rule_length: int = 3,
    mutation_rate: float = 0.08,
    max_steps: int = 64,
    validation_loss_tolerance: float = 0.075,
    report_every: int = 20,
) -> Tuple[
    PatternGenome,
    List[PatternLambdaRecord],
    List[PatternGenerationRecord],
    PatternEvaluation,
    PatternEvaluation,
    PatternEvaluation,
]:
    records: List[PatternLambdaRecord] = []
    all_history: List[PatternGenerationRecord] = []
    runs: List[
        Tuple[
            PatternLambdaRecord,
            PatternGenome,
            PatternEvaluation,
            PatternEvaluation,
        ]
    ] = []

    for idx, lambda_value in enumerate(lambda_values, 1):
        print(f"\n=== lambda {idx}/{len(lambda_values)}: {lambda_value:.4f} ===")
        genome, history, train_eval, val_eval = evolve_solver(
            task=task,
            primitives=primitives,
            seed=seed + idx * 997,
            generations=generations,
            population_size=population_size,
            state_count=state_count,
            max_states=max_states,
            initial_rule_count=initial_rule_count,
            max_rules=max_rules,
            max_rule_length=max_rule_length,
            mutation_rate=mutation_rate,
            lambda_value=lambda_value,
            max_steps=max_steps,
            report_every=report_every,
        )
        all_history.extend(history)
        record = PatternLambdaRecord(
            lambda_value=lambda_value,
            train_loss=train_eval.loss,
            val_loss=val_eval.loss,
            train_exact_match=train_eval.exact_match_rate,
            val_exact_match=val_eval.exact_match_rate,
            train_free_energy=train_eval.free_energy,
            val_free_energy=val_eval.free_energy,
            complexity=val_eval.complexity,
            active_rules=val_eval.active_rules,
            encoded_rules=val_eval.encoded_rules,
        )
        records.append(record)
        runs.append((record, genome, train_eval, val_eval))
        print_pattern_lambda_record(record)

    selected_record = validation_elbow(records, loss_tolerance=validation_loss_tolerance)
    for record, genome, train_eval, val_eval in runs:
        if record is selected_record:
            test_eval = evaluate_genome(
                genome,
                task.test_pairs,
                primitives,
                record.lambda_value,
                max_steps=max_steps,
            )
            record.test_loss = test_eval.loss
            record.test_exact_match = test_eval.exact_match_rate
            record.test_free_energy = test_eval.free_energy
            print("\nValidation elbow selection")
            print_pattern_lambda_record(record)
            return genome, records, all_history, train_eval, val_eval, test_eval

    raise RuntimeError("validation elbow selected a record without a corresponding solver")


def print_pattern_lambda_record(record: PatternLambdaRecord) -> None:
    marker = "*" if record.selected else " "
    score = "NA" if record.elbow_score is None else f"{record.elbow_score:.4f}"
    test_loss = "NA" if record.test_loss is None else f"{record.test_loss:.4f}"
    test_exact = "NA" if record.test_exact_match is None else f"{record.test_exact_match:.2f}"
    print(
        f"{marker} lambda={record.lambda_value:.4f} "
        f"L_train={record.train_loss:.4f} "
        f"L_val={record.val_loss:.4f} "
        f"L_test={test_loss} "
        f"F_val={record.val_free_energy:.4f} "
        f"C={record.complexity:.1f} "
        f"exact={record.train_exact_match:.2f}/{record.val_exact_match:.2f}/{test_exact} "
        f"rules={record.encoded_rules} active={record.active_rules} "
        f"elbow={score}"
    )


def print_pattern_generation(record: PatternGenerationRecord) -> None:
    print(
        f"gen={record.generation:03d} "
        f"lambda={record.lambda_value:.4f} "
        f"F={record.best_free_energy:.4f} "
        f"L={record.best_loss:.4f} "
        f"train_exact={record.train_exact_match:.2f} "
        f"val_exact={record.val_exact_match:.2f} "
        f"C={record.complexity:.1f} "
        f"rules={record.encoded_rules} active={record.active_rules}"
    )


def normalized_edit_distance(source: Sequence[Symbol], target: Sequence[Symbol]) -> float:
    if not source and not target:
        return 0.0
    rows = len(source) + 1
    cols = len(target) + 1
    dp = [[0] * cols for _ in range(rows)]
    for row in range(rows):
        dp[row][0] = row
    for col in range(cols):
        dp[0][col] = col
    for row in range(1, rows):
        for col in range(1, cols):
            substitution = 0 if source[row - 1] == target[col - 1] else 1
            dp[row][col] = min(
                dp[row - 1][col] + 1,
                dp[row][col - 1] + 1,
                dp[row - 1][col - 1] + substitution,
            )
    return dp[-1][-1] / max(len(source), len(target), 1)


def transform_sequence(name: str, sequence: Tuple[Symbol, ...], alphabet_size: int) -> Tuple[Symbol, ...]:
    if name == "copy":
        return sequence
    if name == "swap":
        if len(sequence) < 2:
            return sequence
        return (sequence[1], sequence[0]) + sequence[2:]
    if name == "reverse":
        return tuple(reversed(sequence))
    if name == "rotate_left":
        return sequence[1:] + sequence[:1] if sequence else sequence
    if name == "rotate_right":
        return sequence[-1:] + sequence[:-1] if sequence else sequence
    if name == "duplicate_first":
        return sequence[:1] + sequence
    if name == "dedupe_pair":
        if len(sequence) >= 2 and sequence[0] == sequence[1]:
            return sequence[:1] + sequence[2:]
        return sequence
    raise ValueError(f"unknown transform: {name}")


def random_object_sequence(
    rng: random.Random,
    pool: Sequence[Symbol],
    length: int,
    force_duplicate_pair: Optional[bool] = None,
) -> Tuple[Symbol, ...]:
    if length <= 0:
        return ()
    if force_duplicate_pair is None or length < 2:
        return tuple(rng.choice(pool) for _ in range(length))
    first = rng.choice(pool)
    if force_duplicate_pair:
        second = first
    else:
        alternatives = [item for item in pool if item != first]
        second = rng.choice(alternatives or pool)
    return (first, second) + tuple(rng.choice(pool) for _ in range(length - 2))


def build_split_pairs(
    rng: random.Random,
    transform_name: str,
    pool: Sequence[Symbol],
    examples: int,
    length: int,
) -> Tuple[SequencePair, ...]:
    pairs: List[SequencePair] = []
    for idx in range(examples):
        duplicate_pair = idx % 2 == 0 if transform_name == "dedupe_pair" else None
        source = random_object_sequence(rng, pool, length, force_duplicate_pair=duplicate_pair)
        pairs.append((source, transform_sequence(transform_name, source, len(pool))))
    return tuple(pairs)


def make_object_task(
    transform_name: str,
    seed: int = 7,
    train_examples: int = 8,
    val_examples: int = 4,
    test_examples: int = 4,
    train_objects: int = 8,
    val_objects: int = 8,
    test_objects: int = 8,
    length: int = 2,
) -> PatternTask:
    rng = random.Random(seed)
    train_pool = tuple(range(train_objects))
    val_pool = tuple(range(train_objects, train_objects + val_objects))
    test_pool = tuple(range(train_objects + val_objects, train_objects + val_objects + test_objects))
    return PatternTask(
        name=f"{transform_name}_objects",
        alphabet_size=train_objects + val_objects + test_objects,
        train_pairs=build_split_pairs(rng, transform_name, train_pool, train_examples, length),
        val_pairs=build_split_pairs(rng, transform_name, val_pool, val_examples, length),
        test_pairs=build_split_pairs(rng, transform_name, test_pool, test_examples, length),
    )


def make_chain_task(
    transform_name: str,
    seed: int = 7,
    alphabet_size: int = 6,
    chain_count: int = 4,
    train_transitions: int = 2,
    val_transitions: int = 1,
    test_transitions: int = 1,
    length: int = 2,
) -> PatternTask:
    return make_object_task(
        transform_name=transform_name,
        seed=seed,
        train_examples=chain_count * train_transitions,
        val_examples=chain_count * val_transitions,
        test_examples=chain_count * test_transitions,
        train_objects=alphabet_size,
        val_objects=alphabet_size,
        test_objects=alphabet_size,
        length=length,
    )


def export_solver(genome: PatternGenome, primitives: PrimitiveSet) -> Dict[str, object]:
    rules = [
        {
            "state": rule.state,
            "observation": rule.observation,
            "observation_name": observation_name(rule.observation, primitives.register_count),
            "actions": [action_name(action, genome.alphabet_size) for action in rule.actions],
            "next_state": rule.next_state,
        }
        for rule in genome.rule_map(primitives).values()
    ]
    return {
        "state_count": genome.state_count,
        "object_count": genome.alphabet_size,
        "primitive_set": primitives.name,
        "register_count": primitives.register_count,
        "compare_registers": primitives.compare_registers,
        "complexity": genome_complexity(genome, primitives),
        "rules": rules,
    }


def save_outputs(
    output_dir: str,
    genome: PatternGenome,
    primitives: PrimitiveSet,
    history: Sequence[PatternGenerationRecord],
    lambda_records: Sequence[PatternLambdaRecord],
    train_eval: PatternEvaluation,
    val_eval: PatternEvaluation,
    test_eval: PatternEvaluation,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "solver.json"), "w") as f:
        json.dump(export_solver(genome, primitives), f, indent=2)
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump([asdict(record) for record in history], f, indent=2)
    with open(os.path.join(output_dir, "lambda_sweep.json"), "w") as f:
        json.dump([asdict(record) for record in lambda_records], f, indent=2)
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "selected_lambda": next((asdict(record) for record in lambda_records if record.selected), None),
                "train": asdict(train_eval),
                "validation": asdict(val_eval),
                "test": asdict(test_eval),
            },
            f,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolve sparse register transducers for opaque-object pattern tasks.")
    parser.add_argument(
        "--task",
        choices=["copy", "swap", "reverse", "rotate_left", "rotate_right", "duplicate_first", "dedupe_pair"],
        default="swap",
    )
    parser.add_argument("--primitive-set", choices=sorted(PRIMITIVE_SETS), default="register")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--length", type=int, default=2)
    parser.add_argument("--train-examples", type=int, default=8)
    parser.add_argument("--val-examples", type=int, default=4)
    parser.add_argument("--test-examples", type=int, default=4)
    parser.add_argument("--train-objects", type=int, default=8)
    parser.add_argument("--val-objects", type=int, default=8)
    parser.add_argument("--test-objects", type=int, default=8)
    parser.add_argument("--registers", type=int, default=2)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=160)
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--max-states", type=int, default=None)
    parser.add_argument("--initial-rules", type=int, default=12)
    parser.add_argument("--max-rules", type=int, default=64)
    parser.add_argument("--max-rule-length", type=int, default=3)
    parser.add_argument("--mutation-rate", type=float, default=0.08)
    parser.add_argument("--lambda-value", type=float, default=None, help="run a single lambda value instead of a sweep")
    parser.add_argument("--lambda-min", type=float, default=0.0)
    parser.add_argument("--lambda-max", type=float, default=0.006)
    parser.add_argument("--lambda-points", type=int, default=5)
    parser.add_argument("--validation-loss-tolerance", type=float, default=0.075)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--report-every", type=int, default=20)
    parser.add_argument("--output-dir", default="./output/pattern_fsa")
    args = parser.parse_args()

    task = make_object_task(
        transform_name=args.task,
        seed=args.seed,
        train_examples=args.train_examples,
        val_examples=args.val_examples,
        test_examples=args.test_examples,
        train_objects=args.train_objects,
        val_objects=args.val_objects,
        test_objects=args.test_objects,
        length=args.length,
    )
    primitives = PRIMITIVE_SETS[args.primitive_set](task.alphabet_size, register_count=args.registers)
    lambda_values = (
        [args.lambda_value]
        if args.lambda_value is not None
        else make_lambda_values(args.lambda_min, args.lambda_max, args.lambda_points)
    )
    genome, lambda_records, history, train_eval, val_eval, test_eval = lambda_sweep_solver(
        task=task,
        primitives=primitives,
        lambda_values=lambda_values,
        seed=args.seed,
        generations=args.generations,
        population_size=args.population,
        state_count=args.states,
        max_states=args.max_states,
        initial_rule_count=args.initial_rules,
        max_rules=args.max_rules,
        max_rule_length=args.max_rule_length,
        mutation_rate=args.mutation_rate,
        max_steps=args.max_steps,
        validation_loss_tolerance=args.validation_loss_tolerance,
        report_every=args.report_every,
    )
    selected_lambda = next(record for record in lambda_records if record.selected)

    print("\nFinal evaluation")
    print(f"  task:        {task.name}")
    print(f"  primitives:  {primitives.name}")
    print(f"  selected lambda: {selected_lambda.lambda_value:.4f}")
    print(f"  train loss:  {train_eval.loss:.4f}")
    print(f"  val loss:    {val_eval.loss:.4f}")
    print(f"  test loss:   {test_eval.loss:.4f}")
    print(f"  train exact: {train_eval.exact_match_rate:.2f}")
    print(f"  val exact:   {val_eval.exact_match_rate:.2f}")
    print(f"  test exact:  {test_eval.exact_match_rate:.2f}")
    print(f"  complexity:  {test_eval.complexity:.1f}")
    print(f"  rules:       {test_eval.encoded_rules}")

    save_outputs(args.output_dir, genome, primitives, history, lambda_records, train_eval, val_eval, test_eval)
    print(textwrap.dedent(f"""
        Saved:
          {os.path.join(args.output_dir, 'solver.json')}
          {os.path.join(args.output_dir, 'history.json')}
          {os.path.join(args.output_dir, 'lambda_sweep.json')}
          {os.path.join(args.output_dir, 'summary.json')}
    """).strip())


if __name__ == "__main__":
    main()

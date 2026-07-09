#!/usr/bin/env python3
"""Evolve sparse deterministic classifiers for procedural Bongard problems.

The examples are opaque-object sequences. Each problem supplies positive and
negative examples, and the evolved classifier must assign a binary label to
hidden examples from a disjoint object pool.

Run from the repository root:

    python3 bongard/run_bongard_sparse_classifier.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import random
import statistics
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for _p in (HERE, REPO_ROOT / "transduction"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pattern_fsa import (  # noqa: E402
    HALT,
    MOVE_LEFT,
    MOVE_RIGHT,
    OBS_BOS,
    OBS_EOS,
    OBS_TOKEN,
    PRIMITIVE_SETS,
    Action,
    Observation,
    PrimitiveSet,
    RuleKey,
    Symbol,
    action_name,
    observation_name,
    register_store_action,
)
from run_bongard_symbolic_baseline import (  # noqa: E402
    CONCEPTS,
    Concept,
    Problem,
    Split,
    iter_labeled,
    make_problem,
)

PREDICT_FALSE = 100
PREDICT_TRUE = 101
CLASSIFIER_ACTIONS = (PREDICT_FALSE, PREDICT_TRUE)


@dataclass(frozen=True)
class ClassifierRule:
    state: int
    observation: Observation
    actions: Tuple[Action, ...]
    next_state: int

    @property
    def key(self) -> RuleKey:
        return self.state, self.observation


@dataclass
class ClassifierGenome:
    state_count: int
    rules: List[ClassifierRule]

    def valid_rule(self, rule: ClassifierRule, primitives: PrimitiveSet) -> bool:
        return (
            0 <= rule.state < self.state_count
            and 0 <= rule.next_state < self.state_count
            and rule.observation in primitives.observations
            and all(action in allowed_actions(primitives) for action in rule.actions)
        )

    def rule_map(self, primitives: PrimitiveSet) -> Dict[RuleKey, ClassifierRule]:
        return {rule.key: rule for rule in self.rules if self.valid_rule(rule, primitives)}

    @classmethod
    def random(
        cls,
        rng: random.Random,
        state_count: int,
        primitives: PrimitiveSet,
        initial_rules: int,
        max_rule_length: int,
    ) -> "ClassifierGenome":
        keys = rng.sample(
            all_rule_keys(state_count, primitives.observations),
            k=min(initial_rules, state_count * primitives.observation_count),
        )
        return cls(
            state_count=state_count,
            rules=deduplicate_rules(
                [random_rule(rng, key, state_count, primitives, max_rule_length) for key in keys]
            ),
        )

    def crossover(self, other: "ClassifierGenome", rng: random.Random, primitives: PrimitiveSet) -> "ClassifierGenome":
        map_a = self.rule_map(primitives)
        map_b = other.rule_map(primitives)
        rules: List[ClassifierRule] = []
        for key in sorted(set(map_a) | set(map_b)):
            if key in map_a and key in map_b:
                rules.append(map_a[key] if rng.random() < 0.5 else map_b[key])
            elif key in map_a:
                if rng.random() < 0.5:
                    rules.append(map_a[key])
            elif rng.random() < 0.5:
                rules.append(map_b[key])
        return ClassifierGenome(self.state_count, deduplicate_rules(rules))

    def mutate(
        self,
        rng: random.Random,
        primitives: PrimitiveSet,
        rate: float,
        max_rule_length: int,
        max_rules: int,
    ) -> "ClassifierGenome":
        rules: List[Optional[ClassifierRule]] = list(self.rules)
        actions = allowed_actions(primitives)
        for idx, rule in enumerate(list(rules)):
            if rule is None or rng.random() >= rate:
                continue
            updated_actions = list(rule.actions)
            edit = rng.random()
            if edit < 0.45:
                updated_actions[rng.randrange(len(updated_actions))] = rng.choice(actions)
            elif edit < 0.62 and len(updated_actions) < max_rule_length:
                updated_actions.insert(rng.randrange(len(updated_actions) + 1), rng.choice(actions))
            elif edit < 0.74 and len(updated_actions) > 1:
                del updated_actions[rng.randrange(len(updated_actions))]
            elif edit < 0.86:
                rules[idx] = ClassifierRule(rule.state, rule.observation, tuple(updated_actions), rng.randrange(self.state_count))
                continue
            elif edit < 0.94:
                rules[idx] = random_rule(
                    rng,
                    random_rule_key(rng, self.state_count, primitives.observations),
                    self.state_count,
                    primitives,
                    max_rule_length,
                )
                continue
            else:
                rules[idx] = None
                continue
            rules[idx] = ClassifierRule(rule.state, rule.observation, tuple(updated_actions), rule.next_state)

        kept = deduplicate_rules([rule for rule in rules if rule is not None])
        for _ in range(1 + int(rate * max(1, len(kept)))):
            if len(kept) >= max_rules or rng.random() >= rate * 2.0:
                continue
            existing = {rule.key for rule in kept}
            available = [key for key in all_rule_keys(self.state_count, primitives.observations) if key not in existing]
            if not available:
                break
            kept.append(random_rule(rng, rng.choice(available), self.state_count, primitives, max_rule_length))
        return ClassifierGenome(self.state_count, deduplicate_rules(kept)[:max_rules])


@dataclass(frozen=True)
class Evaluation:
    loss: float
    accuracy: float
    balanced_accuracy: float
    positive_accuracy: float
    negative_accuracy: float
    free_energy: float
    complexity: float
    encoded_rules: int
    active_rules: int

    @property
    def fitness(self) -> float:
        return -self.free_energy


@dataclass(frozen=True)
class LambdaRecord:
    lambda_value: float
    train_loss: float
    validation_loss: float
    hidden_loss: float
    probe_loss: float
    train_accuracy: float
    validation_accuracy: float
    hidden_accuracy: float
    probe_accuracy: float
    train_balanced_accuracy: float
    validation_balanced_accuracy: float
    hidden_balanced_accuracy: float
    probe_balanced_accuracy: float
    complexity: float
    rules: int
    active_rules: int
    selected: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    concept: str
    primitive: str
    registers: int
    states: int
    generations: int
    population: int
    initial_rules: int
    max_rules: int
    max_rule_length: int
    max_steps: int
    replicates: int
    train_count: int
    validation_count: int
    hidden_count: int
    train_positive_count: Optional[int] = None
    train_negative_count: Optional[int] = None
    validation_positive_count: Optional[int] = None
    validation_negative_count: Optional[int] = None
    hidden_positive_count: Optional[int] = None
    hidden_negative_count: Optional[int] = None
    objects_per_split: int = 10
    min_length: int = 2
    max_length: int = 5
    probe_object_count: int = 3
    probe_max_length: int = 6
    lambda_min: float = 0.0
    lambda_max: float = 0.02
    lambda_points: int = 4
    mutation_rate: float = 0.08
    lambda_warmup_fraction: float = 0.0
    archive_training: bool = False
    archive_object_count: int = 3
    archive_max_length: int = 6
    archive_interval: int = 50
    archive_add_per_interval: int = 24


def allowed_actions(primitives: PrimitiveSet) -> Tuple[Action, ...]:
    actions: List[Action] = []
    for action in primitives.actions:
        if action in (MOVE_RIGHT, MOVE_LEFT, HALT):
            actions.append(action)
        elif primitives.register_count and action in tuple(register_store_action(idx) for idx in range(primitives.register_count)):
            actions.append(action)
    actions.extend(CLASSIFIER_ACTIONS)
    return tuple(dict.fromkeys(actions))


def classifier_action_name(action: Action, primitives: PrimitiveSet) -> str:
    if action == PREDICT_FALSE:
        return "PREDICT_FALSE"
    if action == PREDICT_TRUE:
        return "PREDICT_TRUE"
    return action_name(action, primitives.register_count)


def all_rule_keys(state_count: int, observations: Sequence[Observation]) -> List[RuleKey]:
    return [(state, observation) for state in range(state_count) for observation in observations]


def random_rule_key(rng: random.Random, state_count: int, observations: Sequence[Observation]) -> RuleKey:
    return rng.randrange(state_count), rng.choice(tuple(observations))


def random_rule(
    rng: random.Random,
    key: RuleKey,
    state_count: int,
    primitives: PrimitiveSet,
    max_rule_length: int,
) -> ClassifierRule:
    length = rng.randrange(1, max(1, max_rule_length) + 1)
    actions = allowed_actions(primitives)
    return ClassifierRule(
        state=key[0],
        observation=key[1],
        actions=tuple(rng.choice(actions) for _ in range(length)),
        next_state=rng.randrange(state_count),
    )


def deduplicate_rules(rules: Sequence[ClassifierRule]) -> List[ClassifierRule]:
    by_key = {rule.key: rule for rule in rules}
    return [by_key[key] for key in sorted(by_key)]



def rule_complexity(rule: ClassifierRule) -> float:
    return len(rule.actions) + 1.0


def genome_complexity(genome: ClassifierGenome, primitives: PrimitiveSet) -> float:
    return sum(rule_complexity(rule) for rule in genome.rule_map(primitives).values())


def classify(
    genome: ClassifierGenome,
    example: Sequence[Symbol],
    primitives: PrimitiveSet,
    max_steps: int,
) -> Tuple[bool, Tuple[RuleKey, ...]]:
    state = 0
    cursor = 0
    registers: List[Optional[Symbol]] = [None] * primitives.register_count
    active: List[RuleKey] = []
    rule_map = genome.rule_map(primitives)

    for _step in range(max_steps):
        observation = primitives.observe(cursor, example, registers)
        key = (state, observation)
        rule = rule_map.get(key)
        if rule is None:
            return False, tuple(active)
        active.append(key)
        for action in rule.actions:
            if action == PREDICT_TRUE:
                return True, tuple(active)
            if action == PREDICT_FALSE or action == HALT:
                return False, tuple(active)
            if action == MOVE_RIGHT:
                cursor = min(cursor + 1, len(example))
            elif action == MOVE_LEFT:
                cursor = max(cursor - 1, -1)
            elif primitives.register_count:
                for idx in range(primitives.register_count):
                    if action == register_store_action(idx) and 0 <= cursor < len(example):
                        registers[idx] = example[cursor]
        state = rule.next_state
    return False, tuple(active)


def iter_labeled_examples(split: Split) -> Iterable[Tuple[Sequence[Symbol], bool]]:
    yield from iter_labeled(split)


def exhaustive_split(
    concept: Concept,
    object_pool: Sequence[Symbol],
    min_length: int,
    max_length: int,
) -> Split:
    positives: List[Tuple[Symbol, ...]] = []
    negatives: List[Tuple[Symbol, ...]] = []
    for length in range(min_length, max_length + 1):
        for example in itertools.product(object_pool, repeat=length):
            if concept.predicate(example):
                positives.append(tuple(example))
            else:
                negatives.append(tuple(example))
    return Split(positives=tuple(positives), negatives=tuple(negatives))


def merge_splits(base: Split, additions: Split) -> Split:
    positives = tuple(dict.fromkeys(base.positives + additions.positives))
    negatives = tuple(dict.fromkeys(base.negatives + additions.negatives))
    return Split(positives=positives, negatives=negatives)


def split_size(split: Split) -> int:
    return len(split.positives) + len(split.negatives)


def misclassified_split(
    genome: ClassifierGenome,
    split: Split,
    primitives: PrimitiveSet,
    max_steps: int,
    max_examples: int,
) -> Split:
    positives: List[Tuple[Symbol, ...]] = []
    negatives: List[Tuple[Symbol, ...]] = []
    per_label = max(1, max_examples // 2)
    for example, label in iter_labeled(split):
        prediction, _active = classify(genome, example, primitives, max_steps=max_steps)
        if prediction == label:
            continue
        if label and len(positives) < per_label:
            positives.append(tuple(example))
        elif not label and len(negatives) < per_label:
            negatives.append(tuple(example))
        if len(positives) + len(negatives) >= max_examples:
            break
    return Split(positives=tuple(positives), negatives=tuple(negatives))


def evaluate(
    genome: ClassifierGenome,
    split: Split,
    primitives: PrimitiveSet,
    lambda_value: float,
    max_steps: int,
) -> Evaluation:
    labeled = list(iter_labeled_examples(split))
    correct = 0
    positives = 0
    positive_correct = 0
    negatives = 0
    negative_correct = 0
    active_rules = set()
    for example, label in labeled:
        prediction, active = classify(genome, example, primitives, max_steps=max_steps)
        active_rules.update(active)
        if label:
            positives += 1
            if prediction == label:
                positive_correct += 1
        else:
            negatives += 1
            if prediction == label:
                negative_correct += 1
        if prediction == label:
            correct += 1
    accuracy = correct / len(labeled) if labeled else 0.0
    positive_accuracy = positive_correct / positives if positives else 0.0
    negative_accuracy = negative_correct / negatives if negatives else 0.0
    if positives and negatives:
        balanced_accuracy = 0.5 * (positive_accuracy + negative_accuracy)
    else:
        balanced_accuracy = accuracy
    loss = 1.0 - accuracy
    complexity = genome_complexity(genome, primitives)
    return Evaluation(
        loss=loss,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        positive_accuracy=positive_accuracy,
        negative_accuracy=negative_accuracy,
        free_energy=loss + lambda_value * complexity,
        complexity=complexity,
        encoded_rules=len(genome.rule_map(primitives)),
        active_rules=len(active_rules),
    )


def tournament_select(scored: Sequence[Tuple[float, ClassifierGenome]], rng: random.Random, size: int = 5) -> ClassifierGenome:
    contenders = rng.sample(list(scored), k=min(size, len(scored)))
    return max(contenders, key=lambda item: item[0])[1]


def evolve_classifier(
    problem: Problem,
    primitives: PrimitiveSet,
    seed: int,
    lambda_value: float,
    states: int,
    generations: int,
    population_size: int,
    initial_rules: int,
    max_rules: int,
    max_rule_length: int,
    max_steps: int,
    mutation_rate: float = 0.08,
    lambda_warmup_fraction: float = 0.0,
    archive: Optional[Split] = None,
    archive_interval: int = 0,
    archive_add_per_interval: int = 0,
) -> Tuple[ClassifierGenome, Evaluation, Evaluation]:
    rng = random.Random(seed)
    population = [
        ClassifierGenome.random(rng, states, primitives, initial_rules, max_rule_length)
        for _ in range(population_size)
    ]
    best = population[0]
    train_split = problem.train
    seen_train_size = split_size(train_split)
    warmup_generations = max(0, int(generations * lambda_warmup_fraction))
    for _generation in range(generations + 1):
        if warmup_generations:
            current_lambda = lambda_value * min(1.0, _generation / warmup_generations)
        else:
            current_lambda = lambda_value
        evaluations = [(evaluate(genome, train_split, primitives, current_lambda, max_steps), genome) for genome in population]
        evaluations.sort(key=lambda item: item[0].fitness, reverse=True)
        _train_eval, best = evaluations[0]
        if archive and archive_interval and archive_add_per_interval and _generation % archive_interval == 0:
            additions = misclassified_split(
                best,
                archive,
                primitives,
                max_steps=max_steps,
                max_examples=archive_add_per_interval,
            )
            train_split = merge_splits(train_split, additions)
            current_train_size = split_size(train_split)
            if current_train_size != seen_train_size:
                seen_train_size = current_train_size
                evaluations = [(evaluate(genome, train_split, primitives, current_lambda, max_steps), genome) for genome in population]
                evaluations.sort(key=lambda item: item[0].fitness, reverse=True)
                _train_eval, best = evaluations[0]
                scored = [(evaluation.fitness, genome) for evaluation, genome in evaluations]

        if _generation == generations:
            break
        elite_count = max(1, population_size // 10)
        free_energy_elite_count = max(1, elite_count // 2)
        loss_frontier_elite_count = max(1, elite_count - free_energy_elite_count)
        loss_frontier = sorted(
            evaluations,
            key=lambda item: (item[0].loss, item[0].complexity, item[0].encoded_rules),
        )

        next_population: List[ClassifierGenome] = []
        seen_signatures = set()

        def add_elite(genome: ClassifierGenome) -> None:
            signature = tuple((rule.state, rule.observation, rule.actions, rule.next_state) for rule in genome.rule_map(primitives).values())
            if signature in seen_signatures:
                return
            seen_signatures.add(signature)
            next_population.append(genome)

        for _eval, genome in evaluations[:free_energy_elite_count]:
            add_elite(genome)
        for _eval, genome in loss_frontier[:loss_frontier_elite_count]:
            add_elite(genome)

        scored = [(evaluation.fitness, genome) for evaluation, genome in evaluations]
        loss_scored = [(-evaluation.loss, genome) for evaluation, genome in loss_frontier]
        while len(next_population) < population_size:
            parent_pool = loss_scored if rng.random() < 0.25 else scored
            parent_a = tournament_select(parent_pool, rng)
            parent_pool = loss_scored if rng.random() < 0.25 else scored
            parent_b = tournament_select(parent_pool, rng)
            child = parent_a.crossover(parent_b, rng, primitives).mutate(
                rng,
                primitives,
                rate=mutation_rate,
                max_rule_length=max_rule_length,
                max_rules=max_rules,
            )
            next_population.append(child)
        population = next_population
    return (
        best,
        evaluate(best, problem.train, primitives, lambda_value, max_steps),
        evaluate(best, problem.validation, primitives, lambda_value, max_steps),
    )


def make_lambda_values(lambda_min: float, lambda_max: float, points: int) -> List[float]:
    if points <= 1:
        return [lambda_min]
    step = (lambda_max - lambda_min) / (points - 1)
    return [lambda_min + idx * step for idx in range(points)]


def validation_elbow(records: Sequence[LambdaRecord], tolerance: float = 0.0) -> LambdaRecord:
    best_validation_loss = min(record.validation_loss for record in records)
    allowed = [record for record in records if record.validation_loss <= best_validation_loss + tolerance]
    return min(allowed, key=lambda record: (record.complexity, record.validation_loss, record.train_loss))


def run_config(
    config: ExperimentConfig,
    lambda_values: Sequence[float],
    replicate: int = 0,
) -> Tuple[LambdaRecord, ClassifierGenome, PrimitiveSet]:
    concept = next(item for item in CONCEPTS if item.name == config.concept)
    problem = make_problem(
        concept,
        seed=211 + 17 * replicate,
        train_count=config.train_count,
        validation_count=config.validation_count,
        hidden_count=config.hidden_count,
        train_positive_count=config.train_positive_count,
        train_negative_count=config.train_negative_count,
        validation_positive_count=config.validation_positive_count,
        validation_negative_count=config.validation_negative_count,
        hidden_positive_count=config.hidden_positive_count,
        hidden_negative_count=config.hidden_negative_count,
        objects_per_split=config.objects_per_split,
        min_length=config.min_length,
        max_length=config.max_length,
        counterexample_train=True,
    )
    primitives = PRIMITIVE_SETS[config.primitive](30, register_count=config.registers)
    probe_start = 3 * config.objects_per_split
    probe = exhaustive_split(
        concept,
        object_pool=tuple(range(probe_start, probe_start + config.probe_object_count)),
        min_length=config.min_length,
        max_length=config.probe_max_length,
    )
    archive = exhaustive_split(
        concept,
        object_pool=tuple(range(config.archive_object_count)),
        min_length=config.min_length,
        max_length=config.archive_max_length,
    )
    records: List[LambdaRecord] = []
    runs: List[Tuple[LambdaRecord, ClassifierGenome]] = []
    for idx, lambda_value in enumerate(lambda_values, 1):
        genome, train_eval, validation_eval = evolve_classifier(
            problem=problem,
            primitives=primitives,
            seed=1000 + 100 * replicate + idx,
            lambda_value=lambda_value,
            states=config.states,
            generations=config.generations,
            population_size=config.population,
            initial_rules=config.initial_rules,
            max_rules=config.max_rules,
            max_rule_length=config.max_rule_length,
            max_steps=config.max_steps,
            mutation_rate=config.mutation_rate,
            lambda_warmup_fraction=config.lambda_warmup_fraction,
            archive=archive if config.archive_training else None,
            archive_interval=config.archive_interval,
            archive_add_per_interval=config.archive_add_per_interval,
        )
        hidden_eval = evaluate(genome, problem.hidden_test, primitives, lambda_value, config.max_steps)
        probe_eval = evaluate(genome, probe, primitives, lambda_value, config.max_steps)
        record = LambdaRecord(
            lambda_value=lambda_value,
            train_loss=train_eval.loss,
            validation_loss=validation_eval.loss,
            hidden_loss=hidden_eval.loss,
            probe_loss=probe_eval.loss,
            train_accuracy=train_eval.accuracy,
            validation_accuracy=validation_eval.accuracy,
            hidden_accuracy=hidden_eval.accuracy,
            probe_accuracy=probe_eval.accuracy,
            train_balanced_accuracy=train_eval.balanced_accuracy,
            validation_balanced_accuracy=validation_eval.balanced_accuracy,
            hidden_balanced_accuracy=hidden_eval.balanced_accuracy,
            probe_balanced_accuracy=probe_eval.balanced_accuracy,
            complexity=validation_eval.complexity,
            rules=validation_eval.encoded_rules,
            active_rules=validation_eval.active_rules,
        )
        records.append(record)
        runs.append((record, genome))
    selected = validation_elbow(records)
    for record, genome in runs:
        if record is selected:
            return LambdaRecord(**{**record.__dict__, "selected": True}), genome, primitives
    raise RuntimeError("selected record has no genome")


def export_rules(genome: ClassifierGenome, primitives: PrimitiveSet) -> List[str]:
    lines = []
    for rule in genome.rule_map(primitives).values():
        actions = ",".join(classifier_action_name(action, primitives) for action in rule.actions)
        lines.append(f"s{rule.state}:{observation_name(rule.observation, primitives.register_count)} -> {actions} / s{rule.next_state}")
    return lines


CONFIGS = (
    ExperimentConfig(
        concept="length_even",
        primitive="stream",
        registers=0,
        states=3,
        generations=120,
        population=220,
        initial_rules=6,
        max_rules=10,
        max_rule_length=3,
        max_steps=16,
        replicates=8,
        train_count=12,
        validation_count=8,
        hidden_count=24,
    ),
    ExperimentConfig(
        concept="length_multiple_of_three",
        primitive="stream",
        registers=0,
        states=4,
        generations=160,
        population=260,
        initial_rules=8,
        max_rules=12,
        max_rule_length=2,
        max_steps=18,
        replicates=6,
        train_count=12,
        validation_count=8,
        hidden_count=24,
        max_length=6,
        probe_max_length=6,
        lambda_min=0.0001,
        lambda_max=0.0001,
        lambda_points=1,
        mutation_rate=0.10,
        lambda_warmup_fraction=0.8,
        archive_training=True,
        archive_interval=40,
        archive_add_per_interval=24,
    ),
    ExperimentConfig(
        concept="first_equals_second",
        primitive="compare",
        registers=1,
        states=3,
        generations=220,
        population=360,
        initial_rules=8,
        max_rules=14,
        max_rule_length=1,
        max_steps=18,
        replicates=6,
        train_count=14,
        validation_count=10,
        hidden_count=24,
        min_length=3,
        max_length=6,
        probe_max_length=6,
        lambda_min=0.0001,
        lambda_max=0.0001,
        lambda_points=1,
        mutation_rate=0.12,
        lambda_warmup_fraction=0.85,
        archive_training=True,
        archive_interval=40,
        archive_add_per_interval=32,
    ),
    ExperimentConfig(
        concept="last_two_equal",
        primitive="bidirectional_compare",
        registers=1,
        states=3,
        generations=300,
        population=480,
        initial_rules=8,
        max_rules=14,
        max_rule_length=1,
        max_steps=28,
        replicates=6,
        train_count=14,
        validation_count=10,
        hidden_count=24,
        min_length=3,
        max_length=6,
        probe_max_length=6,
        lambda_min=0.0001,
        lambda_max=0.0001,
        lambda_points=1,
        mutation_rate=0.12,
        lambda_warmup_fraction=0.85,
        archive_training=True,
        archive_interval=40,
        archive_add_per_interval=32,
    ),
    ExperimentConfig(
        concept="has_adjacent_duplicate",
        primitive="compare",
        registers=1,
        states=3,
        generations=180,
        population=260,
        initial_rules=8,
        max_rules=12,
        max_rule_length=3,
        max_steps=18,
        replicates=8,
        train_count=12,
        validation_count=8,
        hidden_count=24,
    ),
    ExperimentConfig(
        concept="first_equals_last",
        primitive="bidirectional_compare",
        registers=1,
        states=3,
        generations=900,
        population=1400,
        initial_rules=6,
        max_rules=12,
        max_rule_length=1,
        max_steps=32,
        replicates=12,
        train_count=16,
        validation_count=12,
        hidden_count=32,
        lambda_min=0.0001,
        lambda_max=0.0001,
        lambda_points=1,
        mutation_rate=0.12,
        lambda_warmup_fraction=0.9,
        archive_training=True,
        archive_interval=40,
        archive_add_per_interval=32,
    ),
    ExperimentConfig(
        concept="first_equals_penultimate",
        primitive="bidirectional_compare",
        registers=1,
        states=4,
        generations=700,
        population=1000,
        initial_rules=8,
        max_rules=18,
        max_rule_length=1,
        max_steps=40,
        replicates=4,
        train_count=18,
        validation_count=12,
        hidden_count=32,
        min_length=3,
        max_length=6,
        probe_max_length=6,
        lambda_min=0.0001,
        lambda_max=0.0001,
        lambda_points=1,
        mutation_rate=0.13,
        lambda_warmup_fraction=0.9,
        archive_training=True,
        archive_interval=40,
        archive_add_per_interval=48,
    ),
    ExperimentConfig(
        concept="second_equals_last",
        primitive="bidirectional_compare",
        registers=1,
        states=4,
        generations=520,
        population=800,
        initial_rules=8,
        max_rules=16,
        max_rule_length=1,
        max_steps=40,
        replicates=4,
        train_count=18,
        validation_count=12,
        hidden_count=32,
        min_length=3,
        max_length=6,
        probe_max_length=6,
        lambda_min=0.0001,
        lambda_max=0.0001,
        lambda_points=1,
        mutation_rate=0.13,
        lambda_warmup_fraction=0.9,
        archive_training=True,
        archive_interval=40,
        archive_add_per_interval=48,
    ),
    ExperimentConfig(
        concept="second_equals_penultimate",
        primitive="bidirectional_compare",
        registers=1,
        states=4,
        generations=560,
        population=850,
        initial_rules=8,
        max_rules=18,
        max_rule_length=1,
        max_steps=44,
        replicates=4,
        train_count=18,
        validation_count=12,
        hidden_count=32,
        min_length=4,
        max_length=6,
        probe_max_length=6,
        lambda_min=0.0001,
        lambda_max=0.0001,
        lambda_points=1,
        mutation_rate=0.13,
        lambda_warmup_fraction=0.9,
        archive_training=True,
        archive_interval=40,
        archive_add_per_interval=48,
    ),
    ExperimentConfig(
        concept="palindrome",
        primitive="bidirectional_compare",
        registers=2,
        states=5,
        generations=360,
        population=520,
        initial_rules=12,
        max_rules=28,
        max_rule_length=2,
        max_steps=56,
        replicates=3,
        train_count=16,
        validation_count=12,
        hidden_count=32,
        lambda_min=0.00005,
        lambda_max=0.00005,
        lambda_points=1,
        mutation_rate=0.14,
        lambda_warmup_fraction=0.9,
        archive_training=True,
        archive_interval=40,
        archive_add_per_interval=64,
    ),
    ExperimentConfig(
        concept="contains_duplicate",
        primitive="compare",
        registers=4,
        states=4,
        generations=320,
        population=480,
        initial_rules=14,
        max_rules=28,
        max_rule_length=2,
        max_steps=32,
        replicates=3,
        train_count=16,
        validation_count=12,
        hidden_count=32,
        lambda_min=0.00005,
        lambda_max=0.00005,
        lambda_points=1,
        mutation_rate=0.14,
        lambda_warmup_fraction=0.9,
        archive_training=True,
        archive_interval=40,
        archive_add_per_interval=64,
    ),
    ExperimentConfig(
        concept="all_unique",
        primitive="compare",
        registers=4,
        states=4,
        generations=320,
        population=480,
        initial_rules=14,
        max_rules=28,
        max_rule_length=2,
        max_steps=32,
        replicates=3,
        train_count=16,
        validation_count=12,
        hidden_count=32,
        lambda_min=0.00005,
        lambda_max=0.00005,
        lambda_points=1,
        mutation_rate=0.14,
        lambda_warmup_fraction=0.9,
        archive_training=True,
        archive_interval=40,
        archive_add_per_interval=64,
    ),
)


def is_exact_discovery(record: LambdaRecord) -> bool:
    return (
        record.train_accuracy == 1.0
        and record.validation_accuracy == 1.0
        and record.hidden_accuracy == 1.0
        and record.probe_accuracy == 1.0
    )


def parse_args() -> argparse.Namespace:
    choices = ["all"] + [config.concept for config in CONFIGS]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept", choices=choices, default="all")
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--states", type=int, default=None)
    parser.add_argument("--initial-rules", type=int, default=None)
    parser.add_argument("--max-rules", type=int, default=None)
    parser.add_argument("--max-rule-length", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--train-count", type=int, default=None, help="positive and negative training examples per concept")
    parser.add_argument("--validation-count", type=int, default=None, help="positive and negative validation examples per concept")
    parser.add_argument("--hidden-count", type=int, default=None, help="positive and negative hidden examples per concept")
    parser.add_argument("--train-positive-count", type=int, default=None)
    parser.add_argument("--train-negative-count", type=int, default=None)
    parser.add_argument("--validation-positive-count", type=int, default=None)
    parser.add_argument("--validation-negative-count", type=int, default=None)
    parser.add_argument("--hidden-positive-count", type=int, default=None)
    parser.add_argument("--hidden-negative-count", type=int, default=None)
    parser.add_argument("--objects-per-split", type=int, default=None)
    parser.add_argument("--min-length", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--probe-object-count", type=int, default=None)
    parser.add_argument("--probe-max-length", type=int, default=None)
    parser.add_argument("--lambda-min", type=float, default=None)
    parser.add_argument("--lambda-max", type=float, default=None)
    parser.add_argument("--lambda-points", type=int, default=None)
    parser.add_argument("--mutation-rate", type=float, default=None)
    parser.add_argument("--lambda-warmup-fraction", type=float, default=None)
    parser.add_argument("--archive-training", action="store_true")
    parser.add_argument("--no-archive-training", action="store_true")
    parser.add_argument("--archive-object-count", type=int, default=None)
    parser.add_argument("--archive-max-length", type=int, default=None)
    parser.add_argument("--archive-interval", type=int, default=None)
    parser.add_argument("--archive-add-per-interval", type=int, default=None)
    parser.add_argument("--stop-after-discovery", action="store_true")
    return parser.parse_args()


def override_config(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    updates = {}
    for field_name in (
        "replicates",
        "generations",
        "population",
        "states",
        "initial_rules",
        "max_rules",
        "max_rule_length",
        "max_steps",
        "train_count",
        "validation_count",
        "hidden_count",
        "train_positive_count",
        "train_negative_count",
        "validation_positive_count",
        "validation_negative_count",
        "hidden_positive_count",
        "hidden_negative_count",
        "objects_per_split",
        "min_length",
        "max_length",
        "probe_object_count",
        "probe_max_length",
        "lambda_min",
        "lambda_max",
        "lambda_points",
        "mutation_rate",
        "lambda_warmup_fraction",
        "archive_object_count",
        "archive_max_length",
        "archive_interval",
        "archive_add_per_interval",
    ):
        value = getattr(args, field_name)
        if value is not None:
            updates[field_name] = value
    if args.archive_training:
        updates["archive_training"] = True
    if args.no_archive_training:
        updates["archive_training"] = False
    return replace(config, **updates)


def main() -> None:
    args = parse_args()
    selected_configs = [config for config in CONFIGS if args.concept in ("all", config.concept)]
    configs = [override_config(config, args) for config in selected_configs]
    print("per_run")
    print("concept,replicate,primitive,lambda,train_acc,val_acc,hidden_acc,probe_acc,train_bal_acc,val_bal_acc,hidden_bal_acc,probe_bal_acc,hidden_loss,probe_loss,complexity,rules,active_rules,exact_discovery")
    selected_rules = []
    records_by_concept: Dict[str, List[LambdaRecord]] = {config.concept: [] for config in configs}
    for config in configs:
        lambda_values = make_lambda_values(config.lambda_min, config.lambda_max, config.lambda_points)
        for replicate in range(config.replicates):
            with contextlib.redirect_stdout(io.StringIO()):
                record, genome, primitives = run_config(config, lambda_values, replicate=replicate)
            records_by_concept[config.concept].append(record)
            print(
                f"{config.concept},{replicate},{config.primitive},{record.lambda_value:.6f},"
                f"{record.train_accuracy:.2f},{record.validation_accuracy:.2f},{record.hidden_accuracy:.2f},"
                f"{record.probe_accuracy:.2f},{record.train_balanced_accuracy:.2f},"
                f"{record.validation_balanced_accuracy:.2f},{record.hidden_balanced_accuracy:.2f},"
                f"{record.probe_balanced_accuracy:.2f},{record.hidden_loss:.4f},{record.probe_loss:.4f},"
                f"{record.complexity:.1f},{record.rules},{record.active_rules},"
                f"{is_exact_discovery(record)}"
            )
            selected_rules.append((config.concept, replicate, config.primitive, export_rules(genome, primitives)))
            if args.stop_after_discovery and is_exact_discovery(record):
                break

    print("\nsummary")
    print("concept,primitive,runs,exact_discoveries,mean_train_acc,mean_val_acc,mean_hidden_acc,mean_probe_acc,mean_probe_bal_acc,mean_complexity")
    for config in configs:
        records = records_by_concept[config.concept]
        print(
            f"{config.concept},{config.primitive},{len(records)},{sum(is_exact_discovery(record) for record in records)},"
            f"{statistics.mean(record.train_accuracy for record in records):.3f},"
            f"{statistics.mean(record.validation_accuracy for record in records):.3f},"
            f"{statistics.mean(record.hidden_accuracy for record in records):.3f},"
            f"{statistics.mean(record.probe_accuracy for record in records):.3f},"
            f"{statistics.mean(record.probe_balanced_accuracy for record in records):.3f},"
            f"{statistics.mean(record.complexity for record in records):.1f}"
        )

    print("\nselected_rules")
    for concept, replicate, primitive, rules in selected_rules:
        print(f"{concept} replicate={replicate} + {primitive}:")
        for rule in rules:
            print(f"  {rule}")


if __name__ == "__main__":
    main()

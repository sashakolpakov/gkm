"""
Evolve a finite-state automaton in a small visible foraging game.

The agent is a table-driven automaton, mutations are explicit genome edits,
and progress can be watched in an ASCII game replay.
"""

import argparse
import json
import os
import random
import statistics
import textwrap
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
except ModuleNotFoundError:
    STATUS_OK = "ok"
    Trials = fmin = hp = tpe = None


Action = int
Observation = int
Position = Tuple[int, int]

ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT", "STAY"]
ACTION_DELTAS: List[Position] = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]

OBS_LABELS = [
    "NW", "N", "NE",
    "W", "HERE", "E",
    "SW", "S", "SE",
]
COMPLEXITY_MODES = ("active", "table", "pruned", "mixed")


@dataclass(frozen=True)
class Rule:
    action: Action
    next_state: int


@dataclass
class Genome:
    state_count: int
    rules: List[Rule]

    @property
    def observation_count(self) -> int:
        return len(OBS_LABELS)

    def index(self, state: int, observation: Observation) -> int:
        return state * self.observation_count + observation

    def rule_for(self, state: int, observation: Observation) -> Rule:
        return self.rules[self.index(state, observation)]

    @classmethod
    def random(cls, rng: random.Random, state_count: int = 4) -> "Genome":
        rules = [
            Rule(
                action=rng.randrange(len(ACTION_NAMES)),
                next_state=rng.randrange(state_count),
            )
            for _ in range(state_count * len(OBS_LABELS))
        ]
        return cls(state_count=state_count, rules=rules)

    def crossover(self, other: "Genome", rng: random.Random) -> "Genome":
        rules = [
            self_rule if rng.random() < 0.5 else other_rule
            for self_rule, other_rule in zip(self.rules, other.rules)
        ]
        return Genome(state_count=self.state_count, rules=rules)

    def mutate(self, rng: random.Random, rate: float = 0.04) -> "Genome":
        rules = list(self.rules)
        for idx, rule in enumerate(rules):
            if rng.random() >= rate:
                continue

            if rng.random() < 0.65:
                rules[idx] = Rule(
                    action=rng.randrange(len(ACTION_NAMES)),
                    next_state=rule.next_state,
                )
            else:
                rules[idx] = Rule(
                    action=rule.action,
                    next_state=rng.randrange(self.state_count),
                )
        return Genome(state_count=self.state_count, rules=rules)


@dataclass(frozen=True)
class Level:
    width: int
    height: int
    start: Position
    food: Tuple[Position, ...]


@dataclass
class Episode:
    fitness: float
    collected: int
    total_food: int
    steps: int
    bumps: int
    path: List[Position]
    states: List[int]
    observations: List[Observation]
    actions: List[Action]
    collected_at: List[Position]


@dataclass
class Evaluation:
    fitness: float
    loss: float
    free_energy: float
    lambda_value: float
    complexity_mode: str
    mean_collected: float
    mean_steps: float
    complexity: float
    raw_complexity: float
    active_complexity: float
    active_raw_complexity: float
    table_complexity: float
    table_raw_complexity: float
    pruned_complexity: float
    pruned_raw_complexity: float
    mixed_complexity: float
    mixed_raw_complexity: float
    active_states: int
    active_rules: int
    reachable_states: int
    reachable_rules: int
    table_states: int
    table_rules: int
    episodes: List[Episode]


@dataclass
class GenerationRecord:
    generation: int
    best_fitness: float
    best_loss: float
    best_free_energy: float
    lambda_value: float
    complexity_mode: str
    mean_fitness: float
    train_food: float
    val_food: float
    complexity: float
    raw_complexity: float
    active_complexity: float
    table_complexity: float
    pruned_complexity: float
    mixed_complexity: float
    active_states: int
    active_rules: int
    reachable_states: int
    reachable_rules: int


@dataclass
class LambdaRecord:
    lambda_value: float
    optimizer: str
    complexity_mode: str
    train_loss: float
    val_loss: float
    train_free_energy: float
    val_free_energy: float
    train_food: float
    val_food: float
    complexity: float
    complexity_variance: float
    raw_complexity: float
    active_complexity: float
    active_raw_complexity: float
    table_complexity: float
    table_raw_complexity: float
    pruned_complexity: float
    pruned_raw_complexity: float
    mixed_complexity: float
    mixed_raw_complexity: float
    active_states: int
    active_rules: int
    reachable_states: int
    reachable_rules: int
    table_states: int
    table_rules: int


@dataclass(frozen=True)
class ComplexityBreakdown:
    mode: str
    complexity: float
    raw_complexity: float
    active_complexity: float
    active_raw_complexity: float
    table_complexity: float
    table_raw_complexity: float
    pruned_complexity: float
    pruned_raw_complexity: float
    mixed_complexity: float
    mixed_raw_complexity: float
    active_states: int
    active_rules: int
    reachable_states: int
    reachable_rules: int
    table_states: int
    table_rules: int


def make_levels(seed: int, count: int, width: int = 7, height: int = 7, food_count: int = 4) -> List[Level]:
    rng = random.Random(seed)
    levels = []
    center = (width // 2, height // 2)
    cells = [
        (x, y)
        for y in range(height)
        for x in range(width)
        if (x, y) != center
    ]

    for _ in range(count):
        food = tuple(rng.sample(cells, food_count))
        levels.append(Level(width=width, height=height, start=center, food=food))

    return levels


def observation(position: Position, remaining_food: Sequence[Position]) -> Observation:
    if not remaining_food:
        return OBS_LABELS.index("HERE")

    x, y = position
    target = min(remaining_food, key=lambda pos: abs(pos[0] - x) + abs(pos[1] - y))
    dx = sign(target[0] - x)
    dy = sign(target[1] - y)
    return (dy + 1) * 3 + (dx + 1)


def sign(value: int) -> int:
    if value < 0:
        return -1
    if value > 0:
        return 1
    return 0


def clamp_move(position: Position, action: Action, level: Level) -> Tuple[Position, bool]:
    dx, dy = ACTION_DELTAS[action]
    x, y = position
    nx = x + dx
    ny = y + dy
    if nx < 0 or nx >= level.width or ny < 0 or ny >= level.height:
        return position, True
    return (nx, ny), False


def run_episode(genome: Genome, level: Level, max_steps: int = 48) -> Episode:
    position = level.start
    state = 0
    remaining = set(level.food)
    collected_at: List[Position] = []
    path = [position]
    states = [state]
    observations: List[Observation] = []
    actions: List[Action] = []
    bumps = 0
    shaping = 0.0

    for step in range(max_steps):
        if not remaining:
            break

        before_distance = nearest_food_distance(position, remaining)
        obs = observation(position, tuple(remaining))
        rule = genome.rule_for(state, obs)
        next_position, bumped = clamp_move(position, rule.action, level)

        if bumped:
            bumps += 1
            shaping -= 0.75

        if next_position in remaining:
            remaining.remove(next_position)
            collected_at.append(next_position)
            shaping += 8.0

        after_distance = nearest_food_distance(next_position, remaining)
        if remaining:
            if after_distance < before_distance:
                shaping += 0.35
            elif after_distance > before_distance:
                shaping -= 0.20

        position = next_position
        state = rule.next_state
        observations.append(obs)
        actions.append(rule.action)
        states.append(state)
        path.append(position)

    collected = len(level.food) - len(remaining)
    steps = len(actions)
    fitness = collected * 100.0 + shaping - steps * 0.08 - bumps * 2.0
    return Episode(
        fitness=fitness,
        collected=collected,
        total_food=len(level.food),
        steps=steps,
        bumps=bumps,
        path=path,
        states=states,
        observations=observations,
        actions=actions,
        collected_at=collected_at,
    )


def nearest_food_distance(position: Position, remaining: Iterable[Position]) -> int:
    foods = list(remaining)
    if not foods:
        return 0
    x, y = position
    return min(abs(fx - x) + abs(fy - y) for fx, fy in foods)


def episode_loss(episode: Episode, max_steps: int = 48) -> float:
    missed_food = 1.0 - (episode.collected / max(1, episode.total_food))
    step_cost = episode.steps / max_steps
    bump_cost = episode.bumps / max_steps
    return missed_food + 0.05 * step_cost + 0.10 * bump_cost


def loss_function(genome: Genome, levels: Sequence[Level], max_steps: int = 48) -> float:
    episodes = [run_episode(genome, level, max_steps=max_steps) for level in levels]
    return statistics.mean(episode_loss(episode, max_steps=max_steps) for episode in episodes)


def max_complexity(genome: Genome) -> float:
    return 1.5 * genome.state_count + genome.state_count * len(OBS_LABELS) + 0.5 * len(ACTION_NAMES)


def normalize_complexity(genome: Genome, raw_complexity: float) -> float:
    return raw_complexity / max_complexity(genome)


def validate_complexity_mode(mode: str) -> None:
    if mode not in COMPLEXITY_MODES:
        choices = ", ".join(COMPLEXITY_MODES)
        raise ValueError(f"unknown complexity mode: {mode!r}; expected one of {choices}")


def complexity_function(
    genome: Genome,
    episodes: Optional[Sequence[Episode]] = None,
    mode: str = "active",
) -> Tuple[float, float, int, int]:
    breakdown = complexity_breakdown(genome, episodes, mode=mode)
    if mode == "table":
        return breakdown.complexity, breakdown.raw_complexity, breakdown.table_states, breakdown.table_rules
    if mode == "pruned":
        return breakdown.complexity, breakdown.raw_complexity, breakdown.reachable_states, breakdown.reachable_rules
    return breakdown.complexity, breakdown.raw_complexity, breakdown.active_states, breakdown.active_rules


def complexity_breakdown(
    genome: Genome,
    episodes: Optional[Sequence[Episode]] = None,
    mode: str = "active",
) -> ComplexityBreakdown:
    validate_complexity_mode(mode)

    active_raw, active_states, active_rules = active_complexity_raw(genome, episodes)
    table_raw = table_complexity_raw(genome)
    pruned_raw, reachable_states, reachable_rules = pruned_complexity_raw(genome)
    mixed_raw = active_raw + 0.25 * max(0.0, table_raw - active_raw)

    active_complexity = normalize_complexity(genome, active_raw)
    table_complexity = normalize_complexity(genome, table_raw)
    pruned_complexity = normalize_complexity(genome, pruned_raw)
    mixed_complexity = normalize_complexity(genome, mixed_raw)

    selected = {
        "active": (active_complexity, active_raw),
        "table": (table_complexity, table_raw),
        "pruned": (pruned_complexity, pruned_raw),
        "mixed": (mixed_complexity, mixed_raw),
    }[mode]

    return ComplexityBreakdown(
        mode=mode,
        complexity=selected[0],
        raw_complexity=selected[1],
        active_complexity=active_complexity,
        active_raw_complexity=active_raw,
        table_complexity=table_complexity,
        table_raw_complexity=table_raw,
        pruned_complexity=pruned_complexity,
        pruned_raw_complexity=pruned_raw,
        mixed_complexity=mixed_complexity,
        mixed_raw_complexity=mixed_raw,
        active_states=active_states,
        active_rules=active_rules,
        reachable_states=reachable_states,
        reachable_rules=reachable_rules,
        table_states=genome.state_count,
        table_rules=len(genome.rules),
    )


def active_complexity_raw(
    genome: Genome,
    episodes: Optional[Sequence[Episode]] = None,
) -> Tuple[float, int, int]:
    if episodes is None:
        used_states = set(range(genome.state_count))
        used_rules = {
            (state, obs)
            for state in range(genome.state_count)
            for obs in range(genome.observation_count)
        }
    else:
        used_states, used_rules = active_automaton_parts(episodes)

    active_states = len(used_states)
    active_rules = len(used_rules)
    distinct_actions = len({genome.rule_for(state, obs).action for state, obs in used_rules}) if used_rules else 0
    complexity = 1.5 * active_states + active_rules + 0.5 * distinct_actions
    return complexity, active_states, active_rules


def table_complexity_raw(genome: Genome) -> float:
    distinct_actions = len({rule.action for rule in genome.rules})
    return 1.5 * genome.state_count + len(genome.rules) + 0.5 * distinct_actions


def pruned_complexity_raw(genome: Genome) -> Tuple[float, int, int]:
    reachable_states = reachable_automaton_states(genome)
    reachable_rules = {
        (state, obs)
        for state in reachable_states
        for obs in range(genome.observation_count)
    }
    distinct_actions = len({genome.rule_for(state, obs).action for state, obs in reachable_rules}) if reachable_rules else 0
    complexity = 1.5 * len(reachable_states) + len(reachable_rules) + 0.5 * distinct_actions
    return complexity, len(reachable_states), len(reachable_rules)


def reachable_automaton_states(genome: Genome) -> Set[int]:
    reachable = {0}
    frontier = [0]
    while frontier:
        state = frontier.pop()
        for obs in range(genome.observation_count):
            next_state = genome.rule_for(state, obs).next_state
            if next_state not in reachable:
                reachable.add(next_state)
                frontier.append(next_state)
    return reachable


def free_energy_function(
    genome: Genome,
    levels: Sequence[Level],
    lambda_value: float,
    complexity_mode: str = "active",
) -> float:
    return evaluate(genome, levels, lambda_value=lambda_value, complexity_mode=complexity_mode).free_energy


def evaluate(
    genome: Genome,
    levels: Sequence[Level],
    lambda_value: float = 0.0,
    complexity_weight: Optional[float] = None,
    complexity_mode: str = "active",
) -> Evaluation:
    if complexity_weight is not None:
        lambda_value = complexity_weight

    episodes = [run_episode(genome, level) for level in levels]
    loss = statistics.mean(episode_loss(episode) for episode in episodes)
    complexity = complexity_breakdown(genome, episodes, mode=complexity_mode)
    free_energy = loss + lambda_value * complexity.complexity
    fitness = -free_energy
    return Evaluation(
        fitness=fitness,
        loss=loss,
        free_energy=free_energy,
        lambda_value=lambda_value,
        complexity_mode=complexity.mode,
        mean_collected=statistics.mean(ep.collected for ep in episodes),
        mean_steps=statistics.mean(ep.steps for ep in episodes),
        complexity=complexity.complexity,
        raw_complexity=complexity.raw_complexity,
        active_complexity=complexity.active_complexity,
        active_raw_complexity=complexity.active_raw_complexity,
        table_complexity=complexity.table_complexity,
        table_raw_complexity=complexity.table_raw_complexity,
        pruned_complexity=complexity.pruned_complexity,
        pruned_raw_complexity=complexity.pruned_raw_complexity,
        mixed_complexity=complexity.mixed_complexity,
        mixed_raw_complexity=complexity.mixed_raw_complexity,
        active_states=complexity.active_states,
        active_rules=complexity.active_rules,
        reachable_states=complexity.reachable_states,
        reachable_rules=complexity.reachable_rules,
        table_states=complexity.table_states,
        table_rules=complexity.table_rules,
        episodes=episodes,
    )


def active_automaton_parts(episodes: Sequence[Episode]) -> Tuple[Set[int], Set[Tuple[int, Observation]]]:
    used_states = set()
    used_rules = set()
    for episode in episodes:
        used_states.update(episode.states)
        for state, obs in zip(episode.states, episode.observations):
            used_rules.add((state, obs))
    return used_states, used_rules


def tournament_select(scored: Sequence[Tuple[float, Genome]], rng: random.Random, size: int = 5) -> Genome:
    contenders = rng.sample(list(scored), size)
    return max(contenders, key=lambda item: item[0])[1]


def evolve(
    seed: int = 7,
    generations: int = 80,
    population_size: int = 160,
    state_count: int = 4,
    mutation_rate: float = 0.04,
    elite_fraction: float = 0.10,
    complexity_weight: float = 0.02,
    lambda_value: Optional[float] = None,
    complexity_mode: str = "active",
    train_levels: Optional[List[Level]] = None,
    val_levels: Optional[List[Level]] = None,
    report_every: int = 10,
) -> Tuple[Genome, Genome, List[GenerationRecord], Evaluation, Evaluation]:
    if lambda_value is None:
        lambda_value = complexity_weight

    rng = random.Random(seed)
    train_levels = train_levels or make_levels(seed + 100, count=12)
    val_levels = val_levels or make_levels(seed + 200, count=8)

    population = [Genome.random(rng, state_count=state_count) for _ in range(population_size)]
    initial_best = population[0]
    best = population[0]
    history: List[GenerationRecord] = []

    elite_count = max(1, int(population_size * elite_fraction))

    for generation in range(generations + 1):
        evaluations = [
            (evaluate(genome, train_levels, lambda_value=lambda_value, complexity_mode=complexity_mode), genome)
            for genome in population
        ]
        evaluations.sort(key=lambda item: item[0].fitness, reverse=True)
        scored = [(ev.fitness, genome) for ev, genome in evaluations]

        best_eval, best = evaluations[0]
        if generation == 0:
            initial_best = best

        val_eval = evaluate(best, val_levels, lambda_value=lambda_value, complexity_mode=complexity_mode)
        mean_fitness = statistics.mean(ev.fitness for ev, _ in evaluations)
        history.append(GenerationRecord(
            generation=generation,
            best_fitness=best_eval.fitness,
            best_loss=best_eval.loss,
            best_free_energy=best_eval.free_energy,
            lambda_value=lambda_value,
            complexity_mode=complexity_mode,
            mean_fitness=mean_fitness,
            train_food=best_eval.mean_collected,
            val_food=val_eval.mean_collected,
            complexity=best_eval.complexity,
            raw_complexity=best_eval.raw_complexity,
            active_complexity=best_eval.active_complexity,
            table_complexity=best_eval.table_complexity,
            pruned_complexity=best_eval.pruned_complexity,
            mixed_complexity=best_eval.mixed_complexity,
            active_states=best_eval.active_states,
            active_rules=best_eval.active_rules,
            reachable_states=best_eval.reachable_states,
            reachable_rules=best_eval.reachable_rules,
        ))

        if report_every and (generation == 0 or generation == generations or generation % report_every == 0):
            print_generation(history[-1])

        if generation == generations:
            break

        next_population = [genome for _, genome in evaluations[:elite_count]]
        while len(next_population) < population_size:
            parent_a = tournament_select(scored, rng)
            parent_b = tournament_select(scored, rng)
            child = parent_a.crossover(parent_b, rng).mutate(rng, mutation_rate)
            next_population.append(child)

        population = next_population

    train_eval = evaluate(best, train_levels, lambda_value=lambda_value, complexity_mode=complexity_mode)
    val_eval = evaluate(best, val_levels, lambda_value=lambda_value, complexity_mode=complexity_mode)
    return initial_best, best, history, train_eval, val_eval


def genome_from_hyperopt_params(params: Dict[str, int], state_count: int) -> Genome:
    rules = [
        Rule(
            action=int(params[f"action_{idx}"]),
            next_state=int(params[f"state_{idx}"]),
        )
        for idx in range(state_count * len(OBS_LABELS))
    ]
    return Genome(state_count=state_count, rules=rules)


def hyperopt_search(
    levels: Sequence[Level],
    lambda_value: float,
    state_count: int = 4,
    max_evals: int = 200,
    complexity_mode: str = "active",
) -> Tuple[Genome, List[GenerationRecord], Evaluation]:
    if fmin is None:
        raise RuntimeError("hyperopt is required for --optimizer hyperopt. Install with: pip install -r requirements.txt")

    rule_count = state_count * len(OBS_LABELS)
    space = {}
    for idx in range(rule_count):
        space[f"action_{idx}"] = hp.choice(f"action_{idx}", list(range(len(ACTION_NAMES))))
        space[f"state_{idx}"] = hp.choice(f"state_{idx}", list(range(state_count)))

    best_genome: Optional[Genome] = None
    best_eval: Optional[Evaluation] = None
    history: List[GenerationRecord] = []

    def objective(params: Dict[str, int]) -> Dict[str, object]:
        nonlocal best_genome, best_eval
        genome = genome_from_hyperopt_params(params, state_count)
        current_eval = evaluate(genome, levels, lambda_value=lambda_value, complexity_mode=complexity_mode)

        if best_eval is None or current_eval.free_energy < best_eval.free_energy:
            best_genome = genome
            best_eval = current_eval

        generation = len(history)
        incumbent = best_eval or current_eval
        history.append(GenerationRecord(
            generation=generation,
            best_fitness=incumbent.fitness,
            best_loss=incumbent.loss,
            best_free_energy=incumbent.free_energy,
            lambda_value=lambda_value,
            complexity_mode=complexity_mode,
            mean_fitness=current_eval.fitness,
            train_food=incumbent.mean_collected,
            val_food=0.0,
            complexity=incumbent.complexity,
            raw_complexity=incumbent.raw_complexity,
            active_complexity=incumbent.active_complexity,
            table_complexity=incumbent.table_complexity,
            pruned_complexity=incumbent.pruned_complexity,
            mixed_complexity=incumbent.mixed_complexity,
            active_states=incumbent.active_states,
            active_rules=incumbent.active_rules,
            reachable_states=incumbent.reachable_states,
            reachable_rules=incumbent.reachable_rules,
        ))

        return {"loss": current_eval.free_energy, "status": STATUS_OK}

    trials = Trials()
    fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        verbose=False,
    )

    if best_genome is None or best_eval is None:
        raise RuntimeError("hyperopt did not evaluate any genomes")
    return best_genome, history, best_eval


def make_lambda_values(lambda_min: float, lambda_max: float, lambda_points: int) -> List[float]:
    if lambda_points <= 1:
        return [lambda_min]
    step = (lambda_max - lambda_min) / (lambda_points - 1)
    return [lambda_min + idx * step for idx in range(lambda_points)]


def lambda_sweep(
    seed: int,
    lambda_values: Sequence[float],
    optimizer: str,
    generations: int,
    population_size: int,
    state_count: int,
    mutation_rate: float,
    complexity_weight: float,
    hyperopt_evals: int,
    train_levels: Sequence[Level],
    val_levels: Sequence[Level],
    report_every: int = 10,
    complexity_mode: str = "active",
) -> Tuple[Genome, List[LambdaRecord], List[GenerationRecord], Evaluation, Evaluation]:
    records: List[LambdaRecord] = []
    all_history: List[GenerationRecord] = []
    best_genome: Optional[Genome] = None
    best_train_eval: Optional[Evaluation] = None
    best_val_eval: Optional[Evaluation] = None

    for idx, lambda_value in enumerate(lambda_values, 1):
        print(f"\n=== lambda {idx}/{len(lambda_values)}: {lambda_value:.4f} ===")
        if optimizer == "hyperopt":
            genome, history, _train_search_eval = hyperopt_search(
                train_levels,
                lambda_value=lambda_value,
                state_count=state_count,
                max_evals=hyperopt_evals,
                complexity_mode=complexity_mode,
            )
        elif optimizer == "genetic":
            _initial, genome, history, _train_search_eval, _val_search_eval = evolve(
                seed=seed + idx * 997,
                generations=generations,
                population_size=population_size,
                state_count=state_count,
                mutation_rate=mutation_rate,
                complexity_weight=complexity_weight,
                lambda_value=lambda_value,
                complexity_mode=complexity_mode,
                train_levels=list(train_levels),
                val_levels=list(val_levels),
                report_every=report_every,
            )
        else:
            raise ValueError(f"unknown optimizer: {optimizer}")

        train_eval = evaluate(genome, train_levels, lambda_value=lambda_value, complexity_mode=complexity_mode)
        val_eval = evaluate(genome, val_levels, lambda_value=lambda_value, complexity_mode=complexity_mode)
        all_history.extend(history)
        record = LambdaRecord(
            lambda_value=lambda_value,
            optimizer=optimizer,
            complexity_mode=complexity_mode,
            train_loss=train_eval.loss,
            val_loss=val_eval.loss,
            train_free_energy=train_eval.free_energy,
            val_free_energy=val_eval.free_energy,
            train_food=train_eval.mean_collected,
            val_food=val_eval.mean_collected,
            complexity=val_eval.complexity,
            complexity_variance=complexity_variance(history),
            raw_complexity=val_eval.raw_complexity,
            active_complexity=val_eval.active_complexity,
            active_raw_complexity=val_eval.active_raw_complexity,
            table_complexity=val_eval.table_complexity,
            table_raw_complexity=val_eval.table_raw_complexity,
            pruned_complexity=val_eval.pruned_complexity,
            pruned_raw_complexity=val_eval.pruned_raw_complexity,
            mixed_complexity=val_eval.mixed_complexity,
            mixed_raw_complexity=val_eval.mixed_raw_complexity,
            active_states=val_eval.active_states,
            active_rules=val_eval.active_rules,
            reachable_states=val_eval.reachable_states,
            reachable_rules=val_eval.reachable_rules,
            table_states=val_eval.table_states,
            table_rules=val_eval.table_rules,
        )
        records.append(record)
        print_lambda_record(record)

        candidate_key = (val_eval.loss, val_eval.complexity, train_eval.loss)
        if best_val_eval is None or candidate_key < (best_val_eval.loss, best_val_eval.complexity, best_train_eval.loss):
            best_genome = genome
            best_train_eval = train_eval
            best_val_eval = val_eval

    if best_genome is None or best_train_eval is None or best_val_eval is None:
        raise RuntimeError("lambda sweep did not evaluate any genomes")
    return best_genome, records, all_history, best_train_eval, best_val_eval


def print_lambda_record(record: LambdaRecord) -> None:
    print(
        f"lambda={record.lambda_value:.4f} "
        f"L_train={record.train_loss:.4f} "
        f"L_val={record.val_loss:.4f} "
        f"F_val={record.val_free_energy:.4f} "
        f"C_{record.complexity_mode}={record.complexity:.3f} "
        f"Ca={record.active_complexity:.3f} "
        f"Cp={record.pruned_complexity:.3f} "
        f"Ct={record.table_complexity:.3f} "
        f"chi_C={record.complexity_variance:.5f} "
        f"val_food={record.val_food:.2f} "
        f"active={record.active_states}/{record.active_rules} "
        f"reachable={record.reachable_states}/{record.reachable_rules}"
    )


def complexity_variance(history: Sequence[GenerationRecord]) -> float:
    if len(history) < 2:
        return 0.0
    return statistics.pvariance(record.complexity for record in history)


def print_generation(record: GenerationRecord) -> None:
    print(
        f"gen={record.generation:03d} "
        f"F={record.best_free_energy:.4f} "
        f"L={record.best_loss:.4f} "
        f"train_food={record.train_food:.2f} "
        f"val_food={record.val_food:.2f} "
        f"C_{record.complexity_mode}={record.complexity:.3f} "
        f"Ca={record.active_complexity:.3f} "
        f"Ct={record.table_complexity:.3f} "
        f"active={record.active_states}/{record.active_rules}"
    )


def render_episode(level: Level, episode: Episode, delayless: bool = True) -> str:
    frames = []
    remaining = set(level.food)
    collected = set()

    for step, position in enumerate(episode.path):
        if position in remaining:
            remaining.remove(position)
            collected.add(position)

        lines = [
            f"step={step:02d} pos={position} state={episode.states[step]} "
            f"food={len(collected)}/{len(level.food)}"
        ]
        for y in range(level.height):
            row = []
            for x in range(level.width):
                pos = (x, y)
                if pos == position:
                    row.append("@")
                elif pos in remaining:
                    row.append("*")
                elif pos in collected:
                    row.append(".")
                elif pos in episode.path[:step]:
                    row.append(":")
                else:
                    row.append(" ")
            lines.append("".join(row))

        if step < len(episode.actions):
            obs = OBS_LABELS[episode.observations[step]]
            action = ACTION_NAMES[episode.actions[step]]
            lines.append(f"obs={obs} action={action}")
        frames.append("\n".join(lines))

    separator = "\n" + ("-" * level.width) + "\n"
    return separator.join(frames) if delayless else "\n\n".join(frames)


def export_policy_code(genome: Genome) -> str:
    table = [
        (rule.action, rule.next_state)
        for rule in genome.rules
    ]
    return f'''# Auto-generated evolved finite-state automaton policy.
ACTION_NAMES = {ACTION_NAMES!r}
OBS_LABELS = {OBS_LABELS!r}
STATE_COUNT = {genome.state_count}
POLICY_TABLE = {table!r}

def act(state, observation):
    """Return (action_name, next_state) for a discrete observation."""
    action, next_state = POLICY_TABLE[state * len(OBS_LABELS) + observation]
    return ACTION_NAMES[action], next_state
'''


def save_outputs(
    output_dir: str,
    best: Genome,
    history: Sequence[GenerationRecord],
    train_eval: Evaluation,
    val_eval: Evaluation,
    replay: str,
    lambda_records: Optional[Sequence[LambdaRecord]] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "best_automaton.py"), "w") as f:
        f.write(export_policy_code(best))
    with open(os.path.join(output_dir, "evolution_history.json"), "w") as f:
        json.dump([asdict(record) for record in history], f, indent=2)
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump({
            "train": evaluation_summary(train_eval),
            "validation": evaluation_summary(val_eval),
        }, f, indent=2)
    if lambda_records is not None:
        with open(os.path.join(output_dir, "lambda_sweep.json"), "w") as f:
            json.dump([asdict(record) for record in lambda_records], f, indent=2)
    with open(os.path.join(output_dir, "best_replay.txt"), "w") as f:
        f.write(replay)


def evaluation_summary(evaluation: Evaluation) -> Dict[str, object]:
    return {
        "fitness": evaluation.fitness,
        "loss": evaluation.loss,
        "free_energy": evaluation.free_energy,
        "lambda": evaluation.lambda_value,
        "complexity_mode": evaluation.complexity_mode,
        "mean_collected": evaluation.mean_collected,
        "mean_steps": evaluation.mean_steps,
        "complexity": evaluation.complexity,
        "raw_complexity": evaluation.raw_complexity,
        "active_complexity": evaluation.active_complexity,
        "active_raw_complexity": evaluation.active_raw_complexity,
        "table_complexity": evaluation.table_complexity,
        "table_raw_complexity": evaluation.table_raw_complexity,
        "pruned_complexity": evaluation.pruned_complexity,
        "pruned_raw_complexity": evaluation.pruned_raw_complexity,
        "mixed_complexity": evaluation.mixed_complexity,
        "mixed_raw_complexity": evaluation.mixed_raw_complexity,
        "active_states": evaluation.active_states,
        "active_rules": evaluation.active_rules,
        "reachable_states": evaluation.reachable_states,
        "reachable_rules": evaluation.reachable_rules,
        "table_states": evaluation.table_states,
        "table_rules": evaluation.table_rules,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolve a finite-state automaton in a visible foraging game.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=160)
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--mutation-rate", type=float, default=0.04)
    parser.add_argument("--complexity-weight", type=float, default=0.02)
    parser.add_argument("--lambda-min", type=float, default=0.0)
    parser.add_argument("--lambda-max", type=float, default=0.20)
    parser.add_argument("--lambda-points", type=int, default=5)
    parser.add_argument(
        "--complexity-mode",
        choices=COMPLEXITY_MODES,
        default="active",
        help="Complexity term optimized inside free energy.",
    )
    parser.add_argument("--optimizer", choices=["genetic", "hyperopt"], default="genetic")
    parser.add_argument("--hyperopt-evals", type=int, default=200)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--render", action="store_true", help="Print before/after ASCII replays.")
    parser.add_argument("--output-dir", default="./output/evo_game")
    args = parser.parse_args()

    train_levels = make_levels(args.seed + 100, count=12)
    val_levels = make_levels(args.seed + 200, count=8)
    lambda_values = make_lambda_values(args.lambda_min, args.lambda_max, args.lambda_points)

    if args.optimizer == "hyperopt" and fmin is None:
        raise RuntimeError("hyperopt is not installed. Install dependencies with: pip install -r requirements.txt")

    initial_best, _initial_selected, _initial_history, _initial_train, _initial_val = evolve(
        seed=args.seed,
        generations=0,
        population_size=max(2, min(args.population, 50)),
        state_count=args.states,
        complexity_mode=args.complexity_mode,
        train_levels=train_levels,
        val_levels=val_levels,
        report_every=0,
    )

    best, lambda_records, history, train_eval, val_eval = lambda_sweep(
        seed=args.seed,
        lambda_values=lambda_values,
        optimizer=args.optimizer,
        generations=args.generations,
        population_size=args.population,
        state_count=args.states,
        mutation_rate=args.mutation_rate,
        complexity_weight=args.complexity_weight,
        complexity_mode=args.complexity_mode,
        hyperopt_evals=args.hyperopt_evals,
        train_levels=train_levels,
        val_levels=val_levels,
        report_every=args.report_every,
    )

    replay_level = val_levels[0]
    initial_episode = run_episode(initial_best, replay_level)
    best_episode = run_episode(best, replay_level)
    best_replay = render_episode(replay_level, best_episode)

    print("\nFinal evaluation")
    print(f"  selected lambda: {val_eval.lambda_value:.4f}")
    print(f"  complexity mode: {val_eval.complexity_mode}")
    print(f"  train loss:      {train_eval.loss:.4f}")
    print(f"  val loss:        {val_eval.loss:.4f}")
    print(f"  val free energy: {val_eval.free_energy:.4f}")
    print(f"  train mean food: {train_eval.mean_collected:.2f}/{len(train_levels[0].food)}")
    print(f"  val mean food:   {val_eval.mean_collected:.2f}/{len(val_levels[0].food)}")
    print(f"  selected C:      {val_eval.complexity:.3f} ({val_eval.raw_complexity:.1f} raw)")
    print(f"  active C:        {val_eval.active_complexity:.3f} ({val_eval.active_states} states, {val_eval.active_rules} rules)")
    print(f"  pruned C:        {val_eval.pruned_complexity:.3f} ({val_eval.reachable_states} states, {val_eval.reachable_rules} rules)")
    print(f"  table C:         {val_eval.table_complexity:.3f} ({val_eval.table_states} states, {val_eval.table_rules} rules)")
    print(f"  mixed C:         {val_eval.mixed_complexity:.3f}")

    if args.render:
        print("\nInitial best replay")
        print(render_episode(replay_level, initial_episode))
        print("\nFinal best replay")
        print(best_replay)

    save_outputs(args.output_dir, best, history, train_eval, val_eval, best_replay, lambda_records)
    print(textwrap.dedent(f"""
        Saved:
          {os.path.join(args.output_dir, 'best_automaton.py')}
          {os.path.join(args.output_dir, 'evolution_history.json')}
          {os.path.join(args.output_dir, 'lambda_sweep.json')}
          {os.path.join(args.output_dir, 'summary.json')}
          {os.path.join(args.output_dir, 'best_replay.txt')}
    """).strip())


if __name__ == "__main__":
    main()

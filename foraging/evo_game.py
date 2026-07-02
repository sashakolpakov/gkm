"""
Evolve a finite-state automaton in a small visible foraging game.

The agent is a sparse finite-state automaton, mutations are explicit rule-set
edits, and progress can be watched in an ASCII game replay.
"""

import json
import os
import random
import statistics
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
RuleKey = Tuple[int, Action, Observation]

ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT", "STAY"]
ACTION_DELTAS: List[Position] = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
STAY_ACTION = ACTION_NAMES.index("STAY")

OBS_LABELS = [
    "NW", "N", "NE",
    "W", "HERE", "E",
    "SW", "S", "SE",
]
COMPLEXITY_MODES = ("active", "table", "pruned", "mixed")


@dataclass(frozen=True, init=False)
class Rule:
    state: int
    previous_action: Action
    observation: Observation
    actions: Tuple[Action, ...]
    next_state: int

    def __init__(
        self,
        state: int = 0,
        previous_action: Action = STAY_ACTION,
        observation: Observation = 0,
        action: Optional[Action] = None,
        next_state: int = 0,
        actions: Optional[Sequence[Action]] = None,
    ) -> None:
        if actions is None:
            if action is None:
                raise ValueError("Rule requires either action or actions")
            actions = (action,)

        action_tuple = tuple(int(item) for item in actions)
        if not action_tuple:
            raise ValueError("Rule actions cannot be empty")
        if any(action < 0 or action >= len(ACTION_NAMES) for action in action_tuple):
            raise ValueError("Rule contains an invalid action")
        if previous_action < 0 or previous_action >= len(ACTION_NAMES):
            raise ValueError("Rule has an invalid previous_action")
        if observation < 0 or observation >= len(OBS_LABELS):
            raise ValueError("Rule has an invalid observation")

        object.__setattr__(self, "state", int(state))
        object.__setattr__(self, "previous_action", int(previous_action))
        object.__setattr__(self, "observation", int(observation))
        object.__setattr__(self, "actions", action_tuple)
        object.__setattr__(self, "next_state", int(next_state))

    @property
    def action(self) -> Action:
        return self.actions[0]

    @property
    def key(self) -> RuleKey:
        return (self.state, self.previous_action, self.observation)

    def with_key(self, state: int, previous_action: Action, observation: Observation) -> "Rule":
        return Rule(
            state=state,
            previous_action=previous_action,
            observation=observation,
            actions=self.actions,
            next_state=self.next_state,
        )


@dataclass
class Genome:
    state_count: int
    rules: List[Rule]

    @property
    def observation_count(self) -> int:
        return len(OBS_LABELS)

    @property
    def input_count(self) -> int:
        return len(ACTION_NAMES) * len(OBS_LABELS)

    @property
    def max_exact_rules(self) -> int:
        return self.state_count * self.input_count

    def rule_map(self) -> Dict[RuleKey, Rule]:
        return {rule.key: rule for rule in self.rules if self.valid_rule(rule)}

    def valid_rule(self, rule: Rule) -> bool:
        return (
            0 <= rule.state < self.state_count
            and 0 <= rule.next_state < self.state_count
            and 0 <= rule.previous_action < len(ACTION_NAMES)
            and 0 <= rule.observation < self.observation_count
        )

    def rule_for(
        self,
        state: int,
        previous_action: Action,
        observation: Observation,
        rule_map: Optional[Dict[RuleKey, Rule]] = None,
    ) -> Optional[Rule]:
        rules = rule_map if rule_map is not None else self.rule_map()
        return rules.get((state, previous_action, observation))

    @classmethod
    def random(
        cls,
        rng: random.Random,
        state_count: int = 4,
        max_rule_length: int = 1,
        initial_rule_count: Optional[int] = None,
    ) -> "Genome":
        input_count = len(ACTION_NAMES) * len(OBS_LABELS)
        max_rules = state_count * input_count
        if initial_rule_count is None:
            initial_rule_count = min(max_rules, state_count * len(OBS_LABELS))

        keys = rng.sample(all_rule_keys(state_count), k=min(initial_rule_count, max_rules))
        rules = [random_rule(rng, key, state_count, max_rule_length) for key in keys]
        return cls(state_count=state_count, rules=deduplicate_rules(rules))

    def crossover(self, other: "Genome", rng: random.Random) -> "Genome":
        state_count = max(self.state_count, other.state_count)
        self_rules = self.rule_map()
        other_rules = other.rule_map()
        rules = []
        for key in sorted(set(self_rules) | set(other_rules)):
            if key in self_rules and key in other_rules:
                rules.append(self_rules[key] if rng.random() < 0.5 else other_rules[key])
            elif key in self_rules:
                if rng.random() < 0.5:
                    rules.append(self_rules[key])
            elif rng.random() < 0.5:
                rules.append(other_rules[key])
        return Genome(state_count=state_count, rules=deduplicate_rules(rules))

    def mutate(
        self,
        rng: random.Random,
        rate: float = 0.04,
        max_rule_length: int = 1,
        max_rules: Optional[int] = None,
        max_states: Optional[int] = None,
    ) -> "Genome":
        state_count = self.state_count
        max_states = max_states or state_count
        if state_count < max_states and rng.random() < rate * 0.25:
            state_count += 1

        rules = [rule for rule in self.rules if rule.state < state_count and rule.next_state < state_count]
        for idx, rule in enumerate(rules):
            if rng.random() >= rate:
                continue

            edit = rng.random()
            actions = list(rule.actions)
            if edit < 0.55:
                actions[rng.randrange(len(actions))] = rng.randrange(len(ACTION_NAMES))
            elif edit < 0.70:
                rules[idx] = Rule(
                    state=rule.state,
                    previous_action=rule.previous_action,
                    observation=rule.observation,
                    actions=actions,
                    next_state=rng.randrange(state_count),
                )
                continue
            elif edit < 0.82 and len(actions) < max_rule_length:
                insert_at = rng.randrange(len(actions) + 1)
                actions.insert(insert_at, rng.randrange(len(ACTION_NAMES)))
            elif edit < 0.90 and len(actions) > 1:
                del actions[rng.randrange(len(actions))]
            elif edit < 0.96:
                rules[idx] = random_rule(rng, random_rule_key(rng, state_count), state_count, max_rule_length)
                continue
            else:
                # Delete a rule by replacing it with a sentinel filtered below.
                rules[idx] = Rule(
                    state=rule.state,
                    previous_action=rule.previous_action,
                    observation=rule.observation,
                    action=STAY_ACTION,
                    next_state=rule.next_state,
                )
                rules[idx] = None  # type: ignore[assignment]
                continue

            rules[idx] = Rule(
                state=rule.state,
                previous_action=rule.previous_action,
                observation=rule.observation,
                actions=actions,
                next_state=rule.next_state,
            )

        rules = [rule for rule in rules if rule is not None]
        rules = deduplicate_rules(rules)
        max_possible = state_count * len(ACTION_NAMES) * len(OBS_LABELS)
        max_rules = min(max_rules or max_possible, max_possible)
        add_attempts = 1 + int(rate * max(1, len(rules)))
        for _ in range(add_attempts):
            if len(rules) >= max_rules or rng.random() >= rate * 2.0:
                continue
            existing = {rule.key for rule in rules}
            available = [key for key in all_rule_keys(state_count) if key not in existing]
            if not available:
                break
            rules.append(random_rule(rng, rng.choice(available), state_count, max_rule_length))

        return Genome(state_count=state_count, rules=deduplicate_rules(rules)[:max_rules])


def random_actions(rng: random.Random, max_rule_length: int) -> Tuple[Action, ...]:
    length = rng.randrange(1, max(1, max_rule_length) + 1)
    return tuple(rng.randrange(len(ACTION_NAMES)) for _ in range(length))


def all_rule_keys(state_count: int) -> List[RuleKey]:
    return [
        (state, previous_action, observation_id)
        for state in range(state_count)
        for previous_action in range(len(ACTION_NAMES))
        for observation_id in range(len(OBS_LABELS))
    ]


def random_rule_key(rng: random.Random, state_count: int) -> RuleKey:
    return (
        rng.randrange(state_count),
        rng.randrange(len(ACTION_NAMES)),
        rng.randrange(len(OBS_LABELS)),
    )


def random_rule(
    rng: random.Random,
    key: RuleKey,
    state_count: int,
    max_rule_length: int,
) -> Rule:
    state, previous_action, observation_id = key
    return Rule(
        state=state,
        previous_action=previous_action,
        observation=observation_id,
        actions=random_actions(rng, max_rule_length),
        next_state=rng.randrange(state_count),
    )


def deduplicate_rules(rules: Sequence[Rule]) -> List[Rule]:
    by_key: Dict[RuleKey, Rule] = {}
    for rule in rules:
        by_key[rule.key] = rule
    return [by_key[key] for key in sorted(by_key)]


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
    previous_actions: List[Action]
    rule_keys: List[Optional[RuleKey]]
    actions: List[Action]
    collected_at: List[Position]


@dataclass
class Evaluation:
    fitness: float
    loss: float
    free_energy: float
    lambda_value: float
    complexity_mode: str
    max_steps: int
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
    previous_action = STAY_ACTION
    rule_map = genome.rule_map()
    remaining = set(level.food)
    collected_at: List[Position] = []
    path = [position]
    states = [state]
    observations: List[Observation] = []
    previous_actions: List[Action] = []
    rule_keys: List[Optional[RuleKey]] = []
    actions: List[Action] = []
    bumps = 0
    shaping = 0.0

    while len(actions) < max_steps:
        if not remaining:
            break

        obs = observation(position, tuple(remaining))
        rule_key = (state, previous_action, obs)
        rule = genome.rule_for(state, previous_action, obs, rule_map=rule_map)
        if rule is None:
            break
        decision_state = state

        for action_index, action in enumerate(rule.actions):
            if not remaining or len(actions) >= max_steps:
                break

            before_distance = nearest_food_distance(position, remaining)
            next_position, bumped = clamp_move(position, action, level)

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
            actions.append(action)
            observations.append(obs)
            previous_actions.append(previous_action)
            rule_keys.append(rule_key)
            previous_action = action
            if action_index == len(rule.actions) - 1:
                state = rule.next_state
            states.append(state if action_index == len(rule.actions) - 1 else decision_state)
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
        previous_actions=previous_actions,
        rule_keys=rule_keys,
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
    return table_complexity_raw(genome)


def rule_complexity(rule: Rule) -> float:
    return len(rule.actions) + 1.0


def validate_complexity_mode(mode: str) -> None:
    if mode not in COMPLEXITY_MODES:
        choices = ", ".join(COMPLEXITY_MODES)
        raise ValueError(f"unknown complexity mode: {mode!r}; expected one of {choices}")


def complexity_function(
    genome: Genome,
    episodes: Optional[Sequence[Episode]] = None,
    mode: str = "table",
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
    mode: str = "table",
) -> ComplexityBreakdown:
    validate_complexity_mode(mode)

    active_raw, active_states, active_rules = active_complexity_raw(genome, episodes)
    table_raw = table_complexity_raw(genome)
    pruned_raw, reachable_states, reachable_rules = pruned_complexity_raw(genome)
    mixed_raw = active_raw + 0.25 * max(0.0, table_raw - active_raw)

    selected = {
        "active": active_raw,
        "table": table_raw,
        "pruned": pruned_raw,
        "mixed": mixed_raw,
    }[mode]

    return ComplexityBreakdown(
        mode=mode,
        complexity=selected,
        raw_complexity=selected,
        active_complexity=active_raw,
        active_raw_complexity=active_raw,
        table_complexity=table_raw,
        table_raw_complexity=table_raw,
        pruned_complexity=pruned_raw,
        pruned_raw_complexity=pruned_raw,
        mixed_complexity=mixed_raw,
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
        used_states = {rule.state for rule in genome.rules} | {rule.next_state for rule in genome.rules}
        used_rules = {rule.key for rule in genome.rules}
    else:
        used_states, used_rules = active_automaton_parts(episodes)

    active_states = len(used_states)
    active_rules = len(used_rules)
    rule_map = genome.rule_map()
    complexity = sum(rule_complexity(rule_map[key]) for key in used_rules if key in rule_map)
    return complexity, active_states, active_rules


def table_complexity_raw(genome: Genome) -> float:
    return sum(rule_complexity(rule) for rule in genome.rules)


def pruned_complexity_raw(genome: Genome) -> Tuple[float, int, int]:
    reachable_states = reachable_automaton_states(genome)
    reachable_rules = {rule.key for rule in genome.rules if rule.state in reachable_states}
    rule_map = genome.rule_map()
    complexity = sum(rule_complexity(rule_map[key]) for key in reachable_rules if key in rule_map)
    return complexity, len(reachable_states), len(reachable_rules)


def reachable_automaton_states(genome: Genome) -> Set[int]:
    reachable = {0}
    frontier = [0]
    rules_by_state: Dict[int, List[Rule]] = {}
    for rule in genome.rules:
        rules_by_state.setdefault(rule.state, []).append(rule)
    while frontier:
        state = frontier.pop()
        for rule in rules_by_state.get(state, []):
            next_state = rule.next_state
            if next_state not in reachable:
                reachable.add(next_state)
                frontier.append(next_state)
    return reachable


def free_energy_function(
    genome: Genome,
    levels: Sequence[Level],
    lambda_value: float,
    complexity_mode: str = "table",
    max_steps: int = 48,
) -> float:
    return evaluate(
        genome,
        levels,
        lambda_value=lambda_value,
        complexity_mode=complexity_mode,
        max_steps=max_steps,
    ).free_energy


def evaluate(
    genome: Genome,
    levels: Sequence[Level],
    lambda_value: float = 0.0,
    complexity_weight: Optional[float] = None,
    complexity_mode: str = "table",
    max_steps: int = 48,
) -> Evaluation:
    if complexity_weight is not None:
        lambda_value = complexity_weight

    episodes = [run_episode(genome, level, max_steps=max_steps) for level in levels]
    loss = statistics.mean(episode_loss(episode, max_steps=max_steps) for episode in episodes)
    complexity = complexity_breakdown(genome, episodes, mode=complexity_mode)
    free_energy = loss + lambda_value * complexity.complexity
    fitness = -free_energy
    return Evaluation(
        fitness=fitness,
        loss=loss,
        free_energy=free_energy,
        lambda_value=lambda_value,
        complexity_mode=complexity.mode,
        max_steps=max_steps,
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


def active_automaton_parts(episodes: Sequence[Episode]) -> Tuple[Set[int], Set[RuleKey]]:
    used_states = set()
    used_rules = set()
    for episode in episodes:
        used_states.update(episode.states)
        for key in episode.rule_keys:
            if key is not None:
                used_rules.add(key)
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
    complexity_mode: str = "table",
    max_steps: int = 48,
    max_rule_length: int = 1,
    initial_rule_count: Optional[int] = None,
    max_rules: Optional[int] = None,
    max_states: Optional[int] = None,
    train_levels: Optional[List[Level]] = None,
    val_levels: Optional[List[Level]] = None,
    report_every: int = 10,
) -> Tuple[Genome, Genome, List[GenerationRecord], Evaluation, Evaluation]:
    if lambda_value is None:
        lambda_value = complexity_weight

    rng = random.Random(seed)
    train_levels = train_levels or make_levels(seed + 100, count=12)
    val_levels = val_levels or make_levels(seed + 200, count=8)
    max_states = max_states or state_count
    max_possible_rules = max_states * len(ACTION_NAMES) * len(OBS_LABELS)
    max_rules = min(max_rules or max_possible_rules, max_possible_rules)

    population = [
        Genome.random(
            rng,
            state_count=state_count,
            max_rule_length=max_rule_length,
            initial_rule_count=initial_rule_count,
        )
        for _ in range(population_size)
    ]
    initial_best = population[0]
    best = population[0]
    history: List[GenerationRecord] = []

    elite_count = max(1, int(population_size * elite_fraction))

    for generation in range(generations + 1):
        evaluations = [
            (
                evaluate(
                    genome,
                    train_levels,
                    lambda_value=lambda_value,
                    complexity_mode=complexity_mode,
                    max_steps=max_steps,
                ),
                genome,
            )
            for genome in population
        ]
        evaluations.sort(key=lambda item: item[0].fitness, reverse=True)
        scored = [(ev.fitness, genome) for ev, genome in evaluations]

        best_eval, best = evaluations[0]
        if generation == 0:
            initial_best = best

        val_eval = evaluate(
            best,
            val_levels,
            lambda_value=lambda_value,
            complexity_mode=complexity_mode,
            max_steps=max_steps,
        )
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

        if generation == generations:
            break

        next_population = [genome for _, genome in evaluations[:elite_count]]
        while len(next_population) < population_size:
            parent_a = tournament_select(scored, rng)
            parent_b = tournament_select(scored, rng)
            child = parent_a.crossover(parent_b, rng).mutate(
                rng,
                mutation_rate,
                max_rule_length=max_rule_length,
                max_rules=max_rules,
                max_states=max_states,
            )
            next_population.append(child)

        population = next_population

    train_eval = evaluate(
        best,
        train_levels,
        lambda_value=lambda_value,
        complexity_mode=complexity_mode,
        max_steps=max_steps,
    )
    val_eval = evaluate(
        best,
        val_levels,
        lambda_value=lambda_value,
        complexity_mode=complexity_mode,
        max_steps=max_steps,
    )
    return initial_best, best, history, train_eval, val_eval


def genome_from_hyperopt_params(params: Dict[str, int], state_count: int, max_rule_length: int = 1) -> Genome:
    rules = [
        Rule(
            state=int(params[f"source_state_{idx}"]),
            previous_action=int(params[f"previous_action_{idx}"]),
            observation=int(params[f"observation_{idx}"]),
            actions=tuple(
                int(params[f"action_{idx}_{action_idx}"])
                for action_idx in range(int(params[f"length_{idx}"]))
            ),
            next_state=int(params[f"state_{idx}"]),
        )
        for idx in range(state_count * len(OBS_LABELS))
    ]
    return Genome(state_count=state_count, rules=deduplicate_rules(rules))


def hyperopt_search(
    levels: Sequence[Level],
    lambda_value: float,
    state_count: int = 4,
    max_evals: int = 200,
    complexity_mode: str = "table",
    max_steps: int = 48,
    max_rule_length: int = 1,
) -> Tuple[Genome, List[GenerationRecord], Evaluation]:
    if fmin is None:
        raise RuntimeError("hyperopt is required for --optimizer hyperopt. Install with: pip install -r requirements.txt")

    rule_count = state_count * len(OBS_LABELS)
    space = {}
    for idx in range(rule_count):
        space[f"source_state_{idx}"] = hp.choice(f"source_state_{idx}", list(range(state_count)))
        space[f"previous_action_{idx}"] = hp.choice(f"previous_action_{idx}", list(range(len(ACTION_NAMES))))
        space[f"observation_{idx}"] = hp.choice(f"observation_{idx}", list(range(len(OBS_LABELS))))
        space[f"length_{idx}"] = hp.choice(f"length_{idx}", list(range(1, max(1, max_rule_length) + 1)))
        for action_idx in range(max(1, max_rule_length)):
            space[f"action_{idx}_{action_idx}"] = hp.choice(
                f"action_{idx}_{action_idx}",
                list(range(len(ACTION_NAMES))),
            )
        space[f"state_{idx}"] = hp.choice(f"state_{idx}", list(range(state_count)))

    best_genome: Optional[Genome] = None
    best_eval: Optional[Evaluation] = None
    history: List[GenerationRecord] = []

    def objective(params: Dict[str, int]) -> Dict[str, object]:
        nonlocal best_genome, best_eval
        genome = genome_from_hyperopt_params(params, state_count, max_rule_length=max_rule_length)
        current_eval = evaluate(
            genome,
            levels,
            lambda_value=lambda_value,
            complexity_mode=complexity_mode,
            max_steps=max_steps,
        )

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
    complexity_mode: str = "table",
    max_steps: int = 48,
    max_rule_length: int = 1,
    initial_rule_count: Optional[int] = None,
    max_rules: Optional[int] = None,
    max_states: Optional[int] = None,
) -> Tuple[Genome, List[LambdaRecord], List[GenerationRecord], Evaluation, Evaluation]:
    records: List[LambdaRecord] = []
    all_history: List[GenerationRecord] = []
    best_genome: Optional[Genome] = None
    best_train_eval: Optional[Evaluation] = None
    best_val_eval: Optional[Evaluation] = None

    for idx, lambda_value in enumerate(lambda_values, 1):
        if optimizer == "hyperopt":
            genome, history, _train_search_eval = hyperopt_search(
                train_levels,
                lambda_value=lambda_value,
                state_count=state_count,
                max_evals=hyperopt_evals,
                complexity_mode=complexity_mode,
                max_steps=max_steps,
                max_rule_length=max_rule_length,
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
                max_steps=max_steps,
                max_rule_length=max_rule_length,
                initial_rule_count=initial_rule_count,
                max_rules=max_rules,
                max_states=max_states,
                train_levels=list(train_levels),
                val_levels=list(val_levels),
                report_every=report_every,
            )
        else:
            raise ValueError(f"unknown optimizer: {optimizer}")

        train_eval = evaluate(
            genome,
            train_levels,
            lambda_value=lambda_value,
            complexity_mode=complexity_mode,
            max_steps=max_steps,
        )
        val_eval = evaluate(
            genome,
            val_levels,
            lambda_value=lambda_value,
            complexity_mode=complexity_mode,
            max_steps=max_steps,
        )
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

        candidate_key = (val_eval.loss, val_eval.complexity, train_eval.loss)
        if best_val_eval is None or candidate_key < (best_val_eval.loss, best_val_eval.complexity, best_train_eval.loss):
            best_genome = genome
            best_train_eval = train_eval
            best_val_eval = val_eval

    if best_genome is None or best_train_eval is None or best_val_eval is None:
        raise RuntimeError("lambda sweep did not evaluate any genomes")
    return best_genome, records, all_history, best_train_eval, best_val_eval



def complexity_variance(history: Sequence[GenerationRecord]) -> float:
    if len(history) < 2:
        return 0.0
    return statistics.pvariance(record.complexity for record in history)


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
            prev = ACTION_NAMES[episode.previous_actions[step]]
            action = ACTION_NAMES[episode.actions[step]]
            rule = "encoded" if episode.rule_keys[step] is not None else "default"
            lines.append(f"prev={prev} obs={obs} action={action} rule={rule}")
        frames.append("\n".join(lines))

    separator = "\n" + ("-" * level.width) + "\n"
    return separator.join(frames) if delayless else "\n\n".join(frames)


def export_policy_code(genome: Genome) -> str:
    table = [
        (rule.state, rule.previous_action, rule.observation, rule.actions, rule.next_state)
        for rule in genome.rules
    ]
    return f'''# Auto-generated evolved finite-state automaton policy.
ACTION_NAMES = {ACTION_NAMES!r}
OBS_LABELS = {OBS_LABELS!r}
STATE_COUNT = {genome.state_count}
RULES = {table!r}
RULE_MAP = {{(state, previous_action, observation): (actions, next_state)
            for state, previous_action, observation, actions, next_state in RULES}}

def act(state, previous_action, observation):
    """Return (action_names, next_state) for a sparse FSA input."""
    if (state, previous_action, observation) not in RULE_MAP:
        return None, state
    actions, next_state = RULE_MAP[(state, previous_action, observation)]
    return [ACTION_NAMES[action] for action in actions], next_state
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
        "max_steps": evaluation.max_steps,
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

"""
Bound multi-channel foraging substrate: colimit-cone v3 with priced bindings.

A sibling of cone_foraging.py, not a modification (matching the
evo_game -> cone_foraging precedent), so v1/v2 results stay reproducible. The
single change is the removal of *free rebinding*:

- Legs are UNCHANGED. A leg is a channel-blind cone_foraging.ConeGenome keyed
  by (substate, azimuth); the caller binds its abstract slot to a channel at
  CALL time. Channel-blindness is what makes legs natural, so it is preserved.

- Inline solvers and controllers lose the mutable focus register and the
  SET_FOCUS action. A top-level rule now NAMES the channel it perceives:

      BoundRule: (state, channel, observation) -> (actions, next_state)

  To act on two channels inline you must write two sets of rules, so
  within-task duplication is real and factoring it out through a slot-based
  leg can finally pay (the true or_factor analogue, untestable in v1/v2).

- The binding supplied by a CALL is explicitly priced (binding_cost per bound
  slot), discharging Section 4 property 3 of COLIMIT_CONE_APPROACH.md.

This is the step toward ARC-AGI-3 reality: "what I do to red cells" and "what
I do to blue cells" are different rules keyed on different channels unless the
agent builds and pays for a parameterized abstraction over the channel slot.

See COLIMIT_CONE_APPROACH.md Section 11.
"""

import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cone_foraging as cf
from cone_foraging import (
    ANY_OBS,
    CHANNEL_COUNT,
    CHANNEL_NAMES,
    FOOD_CHANNEL,
    HAZARD_CHANNEL,
    HERE_OBS,
    HOME_CHANNEL,
    LEG_OVERHEAD,
    OBS_LABELS,
    RULE_OVERHEAD,
    ConeGenome,
    ConeLevel,
    ConeRule,
    Leg,
    TaskSpec,
    TASKS,
    action_name,
    azimuth_to,
    call_action,
    decode_call,
    is_call,
    is_move,
    leg_def_complexity,
    observation_name,
    run_cone_episode,
    witness_seek_leg,
)

Position = Tuple[int, int]
BoundRuleKey = Tuple[int, int, int]  # (state, channel, observation)

DEFAULT_CALL_COST = 0.5
DEFAULT_BINDING_COST = 0.5

CONDITIONS = ("inline", "shared", "no_share", "witness")


def bound_inline_actions() -> Tuple[int, ...]:
    """Moves only. No SET_FOCUS (no focus register) and no CALL (inline)."""
    return tuple(range(cf.MOVE_COUNT))


def bound_controller_actions(library_size: int) -> Tuple[int, ...]:
    """Moves plus CALL into the library. Still no SET_FOCUS."""
    calls = tuple(
        call_action(leg_index, channel)
        for leg_index in range(min(library_size, cf.MAX_LEGS))
        for channel in range(CHANNEL_COUNT)
    )
    return tuple(range(cf.MOVE_COUNT)) + calls


@dataclass(frozen=True)
class BoundRule:
    state: int
    channel: int
    observation: int
    actions: Tuple[int, ...]
    next_state: int

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("BoundRule actions cannot be empty")
        if not (0 <= self.channel < CHANNEL_COUNT):
            raise ValueError("BoundRule has an invalid channel")
        if not (0 <= self.observation <= ANY_OBS):
            raise ValueError("BoundRule has an invalid observation")

    @property
    def key(self) -> BoundRuleKey:
        return (self.state, self.channel, self.observation)

    def describe(self) -> str:
        actions = ",".join(action_name(a) for a in self.actions)
        return (
            f"s{self.state}:{CHANNEL_NAMES[self.channel]}:{observation_name(self.observation)}"
            f" -> {actions} / s{self.next_state}"
        )


@dataclass
class BoundGenome:
    state_count: int
    rules: List[BoundRule]

    def rule_map(self) -> Dict[BoundRuleKey, BoundRule]:
        return {
            rule.key: rule
            for rule in self.rules
            if 0 <= rule.state < self.state_count and 0 <= rule.next_state < self.state_count
        }

    def call_references(self) -> List[Tuple[int, int]]:
        refs = []
        for rule in self.rules:
            for action in rule.actions:
                if is_call(action):
                    refs.append(decode_call(action))
        return refs

    def describe(self) -> List[str]:
        return [rule.describe() for rule in sorted(self.rules, key=lambda r: r.key)]

    @classmethod
    def random(
        cls,
        rng: random.Random,
        allowed_actions: Sequence[int],
        state_count: int = 3,
        initial_rule_count: int = 10,
        max_rule_length: int = 2,
    ) -> "BoundGenome":
        keys = [
            (state, channel, observation)
            for state in range(state_count)
            for channel in range(CHANNEL_COUNT)
            for observation in range(ANY_OBS + 1)
        ]
        count = min(initial_rule_count, len(keys))
        rules = [
            random_bound_rule(rng, key, allowed_actions, state_count, max_rule_length)
            for key in rng.sample(keys, count)
        ]
        return cls(state_count=state_count, rules=deduplicate_bound_rules(rules))

    def crossover(self, other: "BoundGenome", rng: random.Random) -> "BoundGenome":
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
        return BoundGenome(state_count=state_count, rules=deduplicate_bound_rules(rules))

    def mutate(
        self,
        rng: random.Random,
        allowed_actions: Sequence[int],
        rate: float = 0.10,
        max_rule_length: int = 2,
        max_rules: int = 20,
    ) -> "BoundGenome":
        rules: List[Optional[BoundRule]] = list(self.rules)
        for idx, rule in enumerate(rules):
            if rule is None or rng.random() >= rate:
                continue
            edit = rng.random()
            actions = list(rule.actions)
            if edit < 0.40:
                actions[rng.randrange(len(actions))] = rng.choice(allowed_actions)
            elif edit < 0.55:
                rules[idx] = BoundRule(rule.state, rule.channel, rule.observation, rule.actions, rng.randrange(self.state_count))
                continue
            elif edit < 0.65 and len(actions) < max_rule_length:
                actions.insert(rng.randrange(len(actions) + 1), rng.choice(allowed_actions))
            elif edit < 0.73 and len(actions) > 1:
                del actions[rng.randrange(len(actions))]
            elif edit < 0.82:
                # Re-key onto a fresh (state, channel, observation) slot.
                key = (rng.randrange(self.state_count), rng.randrange(CHANNEL_COUNT), rng.randrange(ANY_OBS + 1))
                rules[idx] = random_bound_rule(rng, key, allowed_actions, self.state_count, max_rule_length)
                continue
            elif edit < 0.90:
                # Re-channel only: same state/obs/actions, different perceived channel.
                rules[idx] = BoundRule(rule.state, rng.randrange(CHANNEL_COUNT), rule.observation, tuple(actions), rule.next_state)
                continue
            else:
                rules[idx] = None
                continue
            rules[idx] = BoundRule(rule.state, rule.channel, rule.observation, tuple(actions), rule.next_state)

        kept = deduplicate_bound_rules([r for r in rules if r is not None])
        add_attempts = 1 + int(rate * max(1, len(kept)))
        for _ in range(add_attempts):
            if len(kept) >= max_rules or rng.random() >= rate * 2.0:
                continue
            existing = {rule.key for rule in kept}
            available = [
                (state, channel, observation)
                for state in range(self.state_count)
                for channel in range(CHANNEL_COUNT)
                for observation in range(ANY_OBS + 1)
                if (state, channel, observation) not in existing
            ]
            if not available:
                break
            key = rng.choice(available)
            kept.append(random_bound_rule(rng, key, allowed_actions, self.state_count, max_rule_length))
        return BoundGenome(state_count=self.state_count, rules=deduplicate_bound_rules(kept)[:max_rules])


def random_bound_rule(
    rng: random.Random,
    key: BoundRuleKey,
    allowed_actions: Sequence[int],
    state_count: int,
    max_rule_length: int,
) -> BoundRule:
    length = rng.randrange(1, max(1, max_rule_length) + 1)
    actions = tuple(rng.choice(allowed_actions) for _ in range(length))
    return BoundRule(key[0], key[1], key[2], actions, rng.randrange(state_count))


def deduplicate_bound_rules(rules: Sequence[BoundRule]) -> List[BoundRule]:
    by_key: Dict[BoundRuleKey, BoundRule] = {}
    for rule in rules:
        by_key[rule.key] = rule
    return [by_key[key] for key in sorted(by_key)]


# ---------------------------------------------------------------------------
# Episode execution
# ---------------------------------------------------------------------------

@dataclass
class BoundEpisode:
    steps: int
    bumps: int
    ops: int
    collected: int
    total_food: int
    final_position: Position
    dynamic_calls: int
    halted: bool
    hazard_hits: int = 0


def run_bound_episode(
    genome: BoundGenome,
    library: Sequence[Leg],
    level: ConeLevel,
    task: TaskSpec,
    max_steps: int = 44,
    op_budget: Optional[int] = None,
    trace: Optional[List[Dict[str, object]]] = None,
) -> BoundEpisode:
    if op_budget is None:
        op_budget = 4 * max_steps + 16

    def record(kind: str, pos: Position, detail: str = "", depth: int = 0) -> None:
        if trace is not None:
            trace.append({"kind": kind, "pos": pos, "detail": detail, "depth": depth})

    position = level.start
    state = 0
    remaining: Set[Position] = set(level.food)
    steps = bumps = ops = dynamic_calls = hazard_hits = 0
    halted = False

    genome_map = genome.rule_map()
    leg_maps = [leg.genome.rule_map() for leg in library]

    def observe(channel: int) -> int:
        if channel == FOOD_CHANNEL:
            if not remaining:
                return HERE_OBS
            x, y = position
            target = min(remaining, key=lambda p: abs(p[0] - x) + abs(p[1] - y))
            return azimuth_to(position, target)
        if channel == HAZARD_CHANNEL:
            if not level.hazards:
                return HERE_OBS
            hazard = min(level.hazards, key=lambda p: cf.manhattan(position, p))
            if cf.manhattan(position, hazard) >= cf.SAFE_RADIUS:
                return HERE_OBS
            return azimuth_to(position, hazard)
        return azimuth_to(position, level.home)

    def match_bound(current_state: int) -> Optional[BoundRule]:
        # Deterministic dispatch: scan channels in index order, exact obs
        # before ANY within each channel; first match fires.
        for channel in range(CHANNEL_COUNT):
            obs = observe(channel)
            rule = genome_map.get((current_state, channel, obs))
            if rule is None:
                rule = genome_map.get((current_state, channel, ANY_OBS))
            if rule is not None:
                return rule
        return None

    def match_leg(leg_map: Dict, current_state: int, obs: int) -> Optional[ConeRule]:
        rule = leg_map.get((current_state, obs))
        if rule is None:
            rule = leg_map.get((current_state, ANY_OBS))
        return rule

    def task_done() -> bool:
        if task.requires_food and remaining:
            return False
        if task.requires_home and position != level.home:
            return False
        if task.requires_safe and level.hazards:
            if min(cf.manhattan(position, h) for h in level.hazards) < cf.SAFE_RADIUS:
                return False
        return True

    def execute_move(action: int, depth: int = 0) -> None:
        nonlocal position, steps, bumps, hazard_hits
        steps += 1
        dx, dy = cf.MOVE_DELTAS[action]
        nx, ny = position[0] + dx, position[1] + dy
        if nx < 0 or nx >= level.width or ny < 0 or ny >= level.height:
            bumps += 1
            record("bump", position, cf.MOVE_NAMES[action], depth)
            return
        position = (nx, ny)
        if position in remaining:
            remaining.remove(position)
        if position in level.hazards:
            hazard_hits += 1
        record("move", position, cf.MOVE_NAMES[action], depth)

    def run_leg(leg_index: int, channel: int) -> bool:
        nonlocal ops
        substate = 0
        leg_map = leg_maps[leg_index]
        while steps < max_steps and ops < op_budget and not task_done():
            rule = match_leg(leg_map, substate, observe(channel))
            if rule is None:
                record("leg_halt", position, f"leg{leg_index}", 1)
                return True
            for action in rule.actions:
                ops += 1
                if ops >= op_budget or steps >= max_steps:
                    return False
                if is_move(action):
                    execute_move(action, depth=1)
                elif action == cf.RETURN_ACTION:
                    record("return", position, f"leg{leg_index}", 1)
                    return False
            substate = rule.next_state
        return False

    record("start", position, "")
    while steps < max_steps and ops < op_budget and not halted and not task_done():
        rule = match_bound(state)
        if rule is None:
            record("halt", position, f"s{state}")
            halted = True
            break
        for action in rule.actions:
            ops += 1
            if ops >= op_budget or steps >= max_steps:
                break
            if is_move(action):
                execute_move(action)
            elif is_call(action):
                leg_index, channel = decode_call(action)
                if leg_index >= len(library):
                    continue
                dynamic_calls += 1
                record("call", position, f"leg{leg_index}@{CHANNEL_NAMES[channel]}")
                if run_leg(leg_index, channel):
                    halted = True
                    break
            # RETURN at top level: op-costed no-op
        state = rule.next_state

    return BoundEpisode(
        steps=steps,
        bumps=bumps,
        ops=ops,
        collected=len(level.food) - len(remaining),
        total_food=len(level.food),
        final_position=position,
        dynamic_calls=dynamic_calls,
        halted=halted,
        hazard_hits=hazard_hits,
    )


@dataclass
class BoundTaskEvaluation:
    loss: float
    solved: bool
    mean_steps: float
    mean_calls: float


def evaluate_bound_task(
    genome: BoundGenome,
    library: Sequence[Leg],
    levels: Sequence[ConeLevel],
    task: TaskSpec,
    max_steps: int = 44,
) -> BoundTaskEvaluation:
    episodes = [run_bound_episode(genome, library, level, task, max_steps=max_steps) for level in levels]
    # Reuse the v1/v2 loss and solved predicates: BoundEpisode carries the same
    # fields cf.episode_loss / cf.episode_solved read.
    losses = [cf.episode_loss(ep, level, task, max_steps=max_steps) for ep, level in zip(episodes, levels)]
    solved = all(cf.episode_solved(ep, level, task) for ep, level in zip(episodes, levels))
    return BoundTaskEvaluation(
        loss=statistics.mean(losses),
        solved=solved,
        mean_steps=statistics.mean(ep.steps for ep in episodes),
        mean_calls=statistics.mean(ep.dynamic_calls for ep in episodes),
    )


# ---------------------------------------------------------------------------
# Complexity accounting with priced bindings (Section 11.4)
# ---------------------------------------------------------------------------

def bound_genome_complexity(
    genome: BoundGenome,
    library: Sequence[Leg],
    condition: str,
    call_cost: float = DEFAULT_CALL_COST,
    binding_cost: float = DEFAULT_BINDING_COST,
) -> float:
    total = 0.0
    for rule in genome.rules:
        total += RULE_OVERHEAD
        for action in rule.actions:
            if is_call(action):
                total += call_cost + binding_cost  # the binding is priced (NEW in v3)
                if condition == "no_share":
                    leg_index, _channel = decode_call(action)
                    if leg_index < len(library):
                        total += leg_def_complexity(library[leg_index])
            else:
                total += 1.0
    return total


def bound_legs_used(genomes: Sequence[BoundGenome], library: Sequence[Leg]) -> Set[int]:
    used = set()
    for genome in genomes:
        for leg_index, _channel in genome.call_references():
            if leg_index < len(library):
                used.add(leg_index)
    return used


def bound_cone_complexity(
    genomes: Sequence[BoundGenome],
    library: Sequence[Leg],
    condition: str,
    call_cost: float = DEFAULT_CALL_COST,
    binding_cost: float = DEFAULT_BINDING_COST,
    charge_library: bool = True,
) -> float:
    total = sum(
        bound_genome_complexity(g, library, condition, call_cost, binding_cost) for g in genomes
    )
    if condition in ("shared", "witness") and charge_library:
        for leg_index in bound_legs_used(genomes, library):
            total += leg_def_complexity(library[leg_index])
    return total


# ---------------------------------------------------------------------------
# Witnesses (representability floors, never part of a discovery claim)
# ---------------------------------------------------------------------------

def bound_seek_rules(state: int, channel: int, next_state_here: int, flee: bool = False) -> List[BoundRule]:
    """A channel-keyed motion body: move toward the named channel's target by
    azimuth (or away, if flee=True), advancing to next_state_here when that
    channel reads HERE (target reached, or safe)."""
    rules = []
    for observation in range(len(OBS_LABELS)):
        if observation == HERE_OBS:
            rules.append(BoundRule(state, channel, observation, (cf.MOVE_NAMES.index("STAY"),), next_state_here))
            continue
        dx = observation % 3 - 1
        dy = observation // 3 - 1
        if dy < 0:
            move = cf.MOVE_NAMES.index("DOWN" if flee else "UP")
        elif dy > 0:
            move = cf.MOVE_NAMES.index("UP" if flee else "DOWN")
        elif dx < 0:
            move = cf.MOVE_NAMES.index("RIGHT" if flee else "LEFT")
        else:
            move = cf.MOVE_NAMES.index("LEFT" if flee else "RIGHT")
        rules.append(BoundRule(state, channel, observation, (move,), state))
    return rules


def witness_inline(task: TaskSpec) -> BoundGenome:
    """Hand-written inline solver (no calls). Multi-channel tasks DUPLICATE the
    motion body per channel — there is no free rebinding. Food/home phases seek
    (move toward); the hazard phase flees (move away)."""
    phases: List[Tuple[int, bool]] = []  # (channel, flee)
    if task.requires_food:
        phases.append((FOOD_CHANNEL, False))
    if task.requires_safe:
        phases.append((HAZARD_CHANNEL, True))
    if task.requires_home:
        phases.append((HOME_CHANNEL, False))
    rules: List[BoundRule] = []
    for idx, (channel, flee) in enumerate(phases):
        rules.extend(bound_seek_rules(idx, channel, idx + 1, flee=flee))
    return BoundGenome(state_count=len(phases) + 1, rules=rules)


def witness_bound_gluing(task: TaskSpec, seek_index: int = 0, flee_index: int = 1) -> BoundGenome:
    """Hand-written controller calling a leg once per phase under the
    appropriate binding. Food/home phases call the seek leg (move toward the
    bound channel); the safe phase calls the flee leg (move away). The leg
    bodies are paid once. A food/home-only task needs only the seek leg, so
    flee_index can collapse to it."""
    phases: List[Tuple[int, int]] = []  # (channel, leg_index)
    if task.requires_food:
        phases.append((FOOD_CHANNEL, seek_index))
    if task.requires_safe:
        phases.append((HAZARD_CHANNEL, flee_index))
    if task.requires_home:
        phases.append((HOME_CHANNEL, seek_index))
    rules = [
        # Dispatch unconditionally (ANY on the bound channel) and advance.
        BoundRule(idx, channel, ANY_OBS, (call_action(leg_index, channel),), idx + 1)
        for idx, (channel, leg_index) in enumerate(phases)
    ]
    return BoundGenome(state_count=len(phases) + 1, rules=rules)


# ---------------------------------------------------------------------------
# Evolution
# ---------------------------------------------------------------------------

@dataclass
class BoundEvolutionResult:
    genome: BoundGenome
    train_loss: float
    free_energy: float
    saw_call_champion: bool


def evolve_bound_task(
    task: TaskSpec,
    train_levels: Sequence[ConeLevel],
    allowed_actions: Sequence[int],
    library: Sequence[Leg],
    condition: str,
    lambda_value: float,
    seed: int,
    population_size: int = 120,
    generations: int = 60,
    state_count: int = 3,
    initial_rule_count: int = 12,
    max_rules: int = 24,
    max_rule_length: int = 2,
    mutation_rate: float = 0.10,
    elite_fraction: float = 0.10,
    call_cost: float = DEFAULT_CALL_COST,
    binding_cost: float = DEFAULT_BINDING_COST,
    max_steps: int = 44,
) -> BoundEvolutionResult:
    rng = random.Random(seed)
    actions = tuple(allowed_actions)

    def free_energy(genome: BoundGenome) -> Tuple[float, float]:
        evaluation = evaluate_bound_task(genome, library, train_levels, task, max_steps=max_steps)
        complexity = bound_genome_complexity(genome, library, condition, call_cost, binding_cost)
        return evaluation.loss + lambda_value * complexity, evaluation.loss

    population = [
        BoundGenome.random(rng, actions, state_count=state_count,
                           initial_rule_count=initial_rule_count, max_rule_length=max_rule_length)
        for _ in range(population_size)
    ]
    elite_count = max(1, int(population_size * elite_fraction))
    best = population[0]
    best_f, best_loss = free_energy(best)
    saw_call_champion = False

    for _generation in range(generations):
        scored = []
        for genome in population:
            f_value, loss = free_energy(genome)
            scored.append((f_value, loss, genome))
        scored.sort(key=lambda item: item[0])
        if scored[0][0] < best_f:
            best_f, best_loss, best = scored[0]
        if any(g.call_references() for _f, _l, g in scored[:elite_count]):
            saw_call_champion = True

        next_population = [g for _f, _l, g in scored[:elite_count]]
        fitness_pairs = [(-f_value, g) for f_value, _loss, g in scored]
        while len(next_population) < population_size:
            parent_a = _tournament(fitness_pairs, rng)
            parent_b = _tournament(fitness_pairs, rng)
            child = parent_a.crossover(parent_b, rng).mutate(
                rng, actions, rate=mutation_rate, max_rule_length=max_rule_length, max_rules=max_rules
            )
            next_population.append(child)
        population = next_population

    return BoundEvolutionResult(best, best_loss, best_f, saw_call_champion)


def _tournament(scored: Sequence[Tuple[float, BoundGenome]], rng: random.Random, size: int = 5) -> BoundGenome:
    contenders = rng.sample(list(scored), min(size, len(scored)))
    return max(contenders, key=lambda item: item[0])[1]

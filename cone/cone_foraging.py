"""
Multi-channel foraging substrate for the colimit-cone experiment.

A sibling of evo_game.py, not a modification of it. Machines are sparse FSAs
keyed by (state, observation) where the observation is the 9-way azimuth of
the machine's current channel or the wildcard ANY. Library legs are FSAs that
read a channel bound at call time and terminate only by an explicit RETURN
action. Controllers glue legs into tasks via CALL(leg, channel) actions.

See COLIMIT_CONE_APPROACH.md for the design, accounting, and pitfalls.
"""

import random
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

Position = Tuple[int, int]
ConeRuleKey = Tuple[int, int]

MOVE_NAMES = ["UP", "RIGHT", "DOWN", "LEFT", "STAY"]
MOVE_DELTAS: List[Position] = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
MOVE_COUNT = len(MOVE_NAMES)

CHANNEL_NAMES = ["FOOD", "HOME", "HAZARD"]
FOOD_CHANNEL = 0
HOME_CHANNEL = 1
HAZARD_CHANNEL = 2
CHANNEL_COUNT = len(CHANNEL_NAMES)

# The HAZARD channel reads HERE ("safe") when no hazard is within SAFE_RADIUS,
# so flee legs share the seek interface: act until the channel reads HERE.
SAFE_RADIUS = 3

OBS_LABELS = [
    "NW", "N", "NE",
    "W", "HERE", "E",
    "SW", "S", "SE",
]
HERE_OBS = OBS_LABELS.index("HERE")
ANY_OBS = len(OBS_LABELS)
OBS_COUNT = len(OBS_LABELS) + 1  # azimuths plus ANY

SET_FOCUS_BASE = MOVE_COUNT  # 5, 6
RETURN_ACTION = SET_FOCUS_BASE + CHANNEL_COUNT  # 7
CALL_BASE = RETURN_ACTION + 1  # 8 onward
MAX_LEGS = 4
ACTION_COUNT = CALL_BASE + MAX_LEGS * CHANNEL_COUNT


def is_move(action: int) -> bool:
    return 0 <= action < MOVE_COUNT


def is_set_focus(action: int) -> bool:
    return SET_FOCUS_BASE <= action < SET_FOCUS_BASE + CHANNEL_COUNT


def is_call(action: int) -> bool:
    return CALL_BASE <= action < ACTION_COUNT


def call_action(leg_index: int, channel: int) -> int:
    if not (0 <= leg_index < MAX_LEGS and 0 <= channel < CHANNEL_COUNT):
        raise ValueError("invalid call encoding")
    return CALL_BASE + leg_index * CHANNEL_COUNT + channel


def decode_call(action: int) -> Tuple[int, int]:
    offset = action - CALL_BASE
    return offset // CHANNEL_COUNT, offset % CHANNEL_COUNT


def action_name(action: int) -> str:
    if is_move(action):
        return MOVE_NAMES[action]
    if is_set_focus(action):
        return f"SET_FOCUS_{CHANNEL_NAMES[action - SET_FOCUS_BASE]}"
    if action == RETURN_ACTION:
        return "RETURN"
    leg_index, channel = decode_call(action)
    return f"CALL_{leg_index}_{CHANNEL_NAMES[channel]}"


def observation_name(observation: int) -> str:
    return "ANY" if observation == ANY_OBS else OBS_LABELS[observation]


def inline_actions() -> Tuple[int, ...]:
    return tuple(range(MOVE_COUNT)) + tuple(
        SET_FOCUS_BASE + channel for channel in range(CHANNEL_COUNT)
    )


def leg_actions() -> Tuple[int, ...]:
    return tuple(range(MOVE_COUNT)) + (RETURN_ACTION,)


def controller_actions(library_size: int) -> Tuple[int, ...]:
    calls = tuple(
        call_action(leg_index, channel)
        for leg_index in range(min(library_size, MAX_LEGS))
        for channel in range(CHANNEL_COUNT)
    )
    return inline_actions() + calls


@dataclass(frozen=True)
class ConeRule:
    state: int
    observation: int
    actions: Tuple[int, ...]
    next_state: int

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("ConeRule actions cannot be empty")
        if not (0 <= self.observation < OBS_COUNT):
            raise ValueError("ConeRule has an invalid observation")
        if any(action < 0 or action >= ACTION_COUNT for action in self.actions):
            raise ValueError("ConeRule contains an invalid action")

    @property
    def key(self) -> ConeRuleKey:
        return (self.state, self.observation)

    def describe(self) -> str:
        actions = ",".join(action_name(action) for action in self.actions)
        return f"s{self.state}:{observation_name(self.observation)} -> {actions} / s{self.next_state}"


@dataclass
class ConeGenome:
    state_count: int
    rules: List[ConeRule]

    def rule_map(self) -> Dict[ConeRuleKey, ConeRule]:
        return {
            rule.key: rule
            for rule in self.rules
            if 0 <= rule.state < self.state_count and 0 <= rule.next_state < self.state_count
        }

    def call_references(self) -> List[Tuple[int, int]]:
        """Static CALL occurrences (leg_index, channel) in the encoded rules."""
        references = []
        for rule in self.rules:
            for action in rule.actions:
                if is_call(action):
                    references.append(decode_call(action))
        return references

    def describe(self) -> List[str]:
        return [rule.describe() for rule in sorted(self.rules, key=lambda r: r.key)]

    @classmethod
    def random(
        cls,
        rng: random.Random,
        allowed_actions: Sequence[int],
        state_count: int = 3,
        initial_rule_count: int = 8,
        max_rule_length: int = 2,
    ) -> "ConeGenome":
        keys = [
            (state, observation)
            for state in range(state_count)
            for observation in range(OBS_COUNT)
        ]
        count = min(initial_rule_count, len(keys))
        rules = [
            random_cone_rule(rng, key, allowed_actions, state_count, max_rule_length)
            for key in rng.sample(keys, count)
        ]
        return cls(state_count=state_count, rules=deduplicate_cone_rules(rules))

    def crossover(self, other: "ConeGenome", rng: random.Random) -> "ConeGenome":
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
        return ConeGenome(state_count=state_count, rules=deduplicate_cone_rules(rules))

    def mutate(
        self,
        rng: random.Random,
        allowed_actions: Sequence[int],
        rate: float = 0.10,
        max_rule_length: int = 2,
        max_rules: int = 14,
    ) -> "ConeGenome":
        rules: List[Optional[ConeRule]] = list(self.rules)
        for idx, rule in enumerate(rules):
            if rule is None or rng.random() >= rate:
                continue
            edit = rng.random()
            actions = list(rule.actions)
            if edit < 0.45:
                actions[rng.randrange(len(actions))] = rng.choice(allowed_actions)
            elif edit < 0.60:
                rules[idx] = ConeRule(rule.state, rule.observation, rule.actions, rng.randrange(self.state_count))
                continue
            elif edit < 0.72 and len(actions) < max_rule_length:
                actions.insert(rng.randrange(len(actions) + 1), rng.choice(allowed_actions))
            elif edit < 0.80 and len(actions) > 1:
                del actions[rng.randrange(len(actions))]
            elif edit < 0.92:
                key = (rng.randrange(self.state_count), rng.randrange(OBS_COUNT))
                rules[idx] = random_cone_rule(rng, key, allowed_actions, self.state_count, max_rule_length)
                continue
            else:
                rules[idx] = None
                continue
            rules[idx] = ConeRule(rule.state, rule.observation, tuple(actions), rule.next_state)

        kept = deduplicate_cone_rules([rule for rule in rules if rule is not None])
        add_attempts = 1 + int(rate * max(1, len(kept)))
        for _ in range(add_attempts):
            if len(kept) >= max_rules or rng.random() >= rate * 2.0:
                continue
            existing = {rule.key for rule in kept}
            available = [
                (state, observation)
                for state in range(self.state_count)
                for observation in range(OBS_COUNT)
                if (state, observation) not in existing
            ]
            if not available:
                break
            key = rng.choice(available)
            kept.append(random_cone_rule(rng, key, allowed_actions, self.state_count, max_rule_length))
        return ConeGenome(state_count=self.state_count, rules=deduplicate_cone_rules(kept)[:max_rules])


def random_cone_rule(
    rng: random.Random,
    key: ConeRuleKey,
    allowed_actions: Sequence[int],
    state_count: int,
    max_rule_length: int,
) -> ConeRule:
    length = rng.randrange(1, max(1, max_rule_length) + 1)
    actions = tuple(rng.choice(allowed_actions) for _ in range(length))
    return ConeRule(state=key[0], observation=key[1], actions=actions, next_state=rng.randrange(state_count))


def deduplicate_cone_rules(rules: Sequence[ConeRule]) -> List[ConeRule]:
    by_key: Dict[ConeRuleKey, ConeRule] = {}
    for rule in rules:
        by_key[rule.key] = rule
    return [by_key[key] for key in sorted(by_key)]


@dataclass(frozen=True)
class Leg:
    name: str
    genome: ConeGenome


@dataclass(frozen=True)
class TaskSpec:
    name: str
    requires_food: bool
    requires_home: bool
    requires_safe: bool = False

    @property
    def food_count(self) -> int:
        return 3 if self.requires_food else 0

    @property
    def hazard_count(self) -> int:
        return 1 if self.requires_safe else 0


TASKS: Dict[str, TaskSpec] = {
    "forage": TaskSpec("forage", requires_food=True, requires_home=False),
    "homing": TaskSpec("homing", requires_food=False, requires_home=True),
    "forage_then_home": TaskSpec("forage_then_home", requires_food=True, requires_home=True),
    "flee": TaskSpec("flee", requires_food=False, requires_home=False, requires_safe=True),
    "forage_flee": TaskSpec("forage_flee", requires_food=True, requires_home=False, requires_safe=True),
    "flee_then_home": TaskSpec("flee_then_home", requires_food=False, requires_home=True, requires_safe=True),
}


@dataclass(frozen=True)
class ConeLevel:
    width: int
    height: int
    start: Position
    food: Tuple[Position, ...]
    home: Position
    hazards: Tuple[Position, ...] = ()


@dataclass
class ConeEpisode:
    steps: int
    bumps: int
    ops: int
    collected: int
    total_food: int
    final_position: Position
    dynamic_calls: int
    halted: bool
    hazard_hits: int = 0


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def make_cone_levels(
    seed: int,
    count: int,
    task: TaskSpec,
    width: int = 7,
    height: int = 7,
) -> List[ConeLevel]:
    rng = random.Random(seed)
    center = (width // 2, height // 2)
    cells = [(x, y) for y in range(height) for x in range(width) if (x, y) != center]
    levels = []
    while len(levels) < count:
        chosen = rng.sample(cells, task.food_count + 1)
        home = chosen[0]
        food = tuple(chosen[1:])
        hazards: Tuple[Position, ...] = ()
        if task.hazard_count:
            if task.requires_food:
                # forage_flee: the hazard sits next to a food cell, so seeking
                # food ends unsafe and the task genuinely composes seek + flee.
                anchor = food[0]
            else:
                # flee / flee_then_home: the hazard sits next to the start.
                anchor = center

            def escape_room(neighbor: Position) -> int:
                # Cells available from the anchor in the antipodal flight
                # direction. A minimal flee leg has no wall sense, so levels
                # must leave room to reach SAFE_RADIUS by pure flight; evolved
                # legs may handle harder geometry, witnesses need not.
                dx = anchor[0] - neighbor[0]
                dy = anchor[1] - neighbor[1]
                if dx > 0:
                    return width - 1 - anchor[0]
                if dx < 0:
                    return anchor[0]
                if dy > 0:
                    return height - 1 - anchor[1]
                return anchor[1]

            neighbor_cells = [
                (anchor[0] + dx, anchor[1] + dy)
                for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0))
                if 0 <= anchor[0] + dx < width and 0 <= anchor[1] + dy < height
                and (anchor[0] + dx, anchor[1] + dy) not in food
                and (anchor[0] + dx, anchor[1] + dy) != center
                and escape_room((anchor[0] + dx, anchor[1] + dy)) >= SAFE_RADIUS - 1
            ]
            if not neighbor_cells:
                continue
            hazard = rng.choice(neighbor_cells)
            hazards = (hazard,)
            if task.requires_home and manhattan(home, hazard) < SAFE_RADIUS:
                safe_cells = [
                    cell for cell in cells
                    if manhattan(cell, hazard) >= SAFE_RADIUS and cell not in food and cell != hazard
                ]
                home = rng.choice(safe_cells)
        levels.append(
            ConeLevel(width=width, height=height, start=center, food=food, home=home, hazards=hazards)
        )
    return levels


def sign(value: int) -> int:
    if value < 0:
        return -1
    if value > 0:
        return 1
    return 0


def azimuth_to(position: Position, target: Position) -> int:
    dx = sign(target[0] - position[0])
    dy = sign(target[1] - position[1])
    return (dy + 1) * 3 + (dx + 1)


def run_cone_episode(
    genome: ConeGenome,
    library: Sequence[Leg],
    level: ConeLevel,
    task: TaskSpec,
    max_steps: int = 44,
    op_budget: Optional[int] = None,
    trace: Optional[List[Dict[str, object]]] = None,
) -> ConeEpisode:
    if op_budget is None:
        op_budget = 4 * max_steps + 16

    def record(kind: str, pos: Position, detail: str = "", depth: int = 0) -> None:
        if trace is not None:
            trace.append({"kind": kind, "pos": pos, "detail": detail, "depth": depth})

    position = level.start
    state = 0
    # The initial focus is the task's default sense (reconciliation log R8):
    # food tasks start on the FOOD compass, pure homing starts on HOME, and
    # food-less safety tasks start on HAZARD. This helps inline solvers (no
    # mandatory SET_FOCUS discovery) and therefore cannot bias selection
    # toward cones.
    if task.requires_food:
        focus = FOOD_CHANNEL
    elif task.requires_safe and not task.requires_home:
        focus = HAZARD_CHANNEL
    else:
        focus = HOME_CHANNEL
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
            target = min(remaining, key=lambda pos: abs(pos[0] - x) + abs(pos[1] - y))
            return azimuth_to(position, target)
        if channel == HAZARD_CHANNEL:
            if not level.hazards:
                return HERE_OBS
            hazard = min(level.hazards, key=lambda pos: manhattan(position, pos))
            if manhattan(position, hazard) >= SAFE_RADIUS:
                return HERE_OBS
            return azimuth_to(position, hazard)
        return azimuth_to(position, level.home)

    def match(rule_map: Dict[ConeRuleKey, ConeRule], current_state: int, obs: int) -> Optional[ConeRule]:
        rule = rule_map.get((current_state, obs))
        if rule is None:
            rule = rule_map.get((current_state, ANY_OBS))
        return rule

    def task_done() -> bool:
        if task.requires_food and remaining:
            return False
        if task.requires_home and position != level.home:
            return False
        if task.requires_safe and level.hazards:
            if min(manhattan(position, hazard) for hazard in level.hazards) < SAFE_RADIUS:
                return False
        return True

    def execute_move(action: int, depth: int = 0) -> None:
        nonlocal position, steps, bumps, hazard_hits
        steps += 1
        dx, dy = MOVE_DELTAS[action]
        nx, ny = position[0] + dx, position[1] + dy
        if nx < 0 or nx >= level.width or ny < 0 or ny >= level.height:
            bumps += 1
            record("bump", position, MOVE_NAMES[action], depth)
            return
        position = (nx, ny)
        if position in remaining:
            remaining.remove(position)
        if position in level.hazards:
            hazard_hits += 1
        record("move", position, MOVE_NAMES[action], depth)

    def run_leg(leg_index: int, channel: int) -> bool:
        """Run a bound leg. Returns True when the episode halts inside the leg."""
        nonlocal ops
        substate = 0
        leg_map = leg_maps[leg_index]
        while steps < max_steps and ops < op_budget and not task_done():
            rule = match(leg_map, substate, observe(channel))
            if rule is None:
                record("leg_halt", position, f"leg{leg_index}", 1)
                return True
            for action in rule.actions:
                ops += 1
                if ops >= op_budget or steps >= max_steps:
                    return False
                if is_move(action):
                    execute_move(action, depth=1)
                elif action == RETURN_ACTION:
                    record("return", position, f"leg{leg_index}", 1)
                    return False
                # other actions are illegal inside a leg: op-costed no-op
            substate = rule.next_state
        return False

    record("start", position, f"focus={CHANNEL_NAMES[focus]}")
    while steps < max_steps and ops < op_budget and not halted and not task_done():
        rule = match(genome_map, state, observe(focus))
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
            elif is_set_focus(action):
                focus = action - SET_FOCUS_BASE
                record("set_focus", position, CHANNEL_NAMES[focus])
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

    return ConeEpisode(
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


def final_hazard_distance(episode: ConeEpisode, level: ConeLevel) -> Optional[int]:
    if not level.hazards:
        return None
    return min(manhattan(episode.final_position, hazard) for hazard in level.hazards)


def episode_loss(episode: ConeEpisode, level: ConeLevel, task: TaskSpec, max_steps: int = 44) -> float:
    loss = 0.0
    if task.requires_food:
        loss += 1.0 - episode.collected / max(1, episode.total_food)
    if task.requires_home:
        distance = manhattan(episode.final_position, level.home)
        loss += distance / max(1, level.width + level.height - 2)
    if task.requires_safe:
        distance = final_hazard_distance(episode, level)
        if distance is not None:
            loss += max(0, SAFE_RADIUS - distance) / SAFE_RADIUS
    loss += 0.05 * episode.steps / max_steps + 0.10 * episode.bumps / max_steps
    loss += 0.15 * episode.hazard_hits / max_steps
    return loss


def episode_solved(episode: ConeEpisode, level: ConeLevel, task: TaskSpec) -> bool:
    if task.requires_food and episode.collected < episode.total_food:
        return False
    if task.requires_home and episode.final_position != level.home:
        return False
    if task.requires_safe:
        distance = final_hazard_distance(episode, level)
        if distance is not None and distance < SAFE_RADIUS:
            return False
    return True


@dataclass
class TaskEvaluation:
    loss: float
    solved: bool
    mean_steps: float
    mean_calls: float


def evaluate_cone_task(
    genome: ConeGenome,
    library: Sequence[Leg],
    levels: Sequence[ConeLevel],
    task: TaskSpec,
    max_steps: int = 44,
) -> TaskEvaluation:
    episodes = [run_cone_episode(genome, library, level, task, max_steps=max_steps) for level in levels]
    losses = [episode_loss(episode, level, task, max_steps=max_steps) for episode, level in zip(episodes, levels)]
    solved = all(episode_solved(episode, level, task) for episode, level in zip(episodes, levels))
    return TaskEvaluation(
        loss=statistics.mean(losses),
        solved=solved,
        mean_steps=statistics.mean(ep.steps for ep in episodes),
        mean_calls=statistics.mean(ep.dynamic_calls for ep in episodes),
    )


# ---------------------------------------------------------------------------
# Complexity accounting (COLIMIT_CONE_APPROACH.md, Section 4)
# ---------------------------------------------------------------------------

RULE_OVERHEAD = 1.0
LEG_OVERHEAD = 1.0
DEFAULT_CALL_COST = 0.5

CONDITIONS = ("inline", "shared", "no_share", "witness")


def leg_def_complexity(leg: Leg) -> float:
    total = LEG_OVERHEAD
    for rule in leg.genome.rules:
        total += RULE_OVERHEAD + float(len(rule.actions))
    return total


def genome_complexity(
    genome: ConeGenome,
    library: Sequence[Leg],
    condition: str,
    call_cost: float = DEFAULT_CALL_COST,
) -> float:
    total = 0.0
    for rule in genome.rules:
        total += RULE_OVERHEAD
        for action in rule.actions:
            if is_call(action):
                total += call_cost
                if condition == "no_share":
                    leg_index, _channel = decode_call(action)
                    if leg_index < len(library):
                        total += leg_def_complexity(library[leg_index])
            else:
                total += 1.0
    return total


def legs_used(genomes: Sequence[ConeGenome], library: Sequence[Leg]) -> Set[int]:
    used = set()
    for genome in genomes:
        for leg_index, _channel in genome.call_references():
            if leg_index < len(library):
                used.add(leg_index)
    return used


def cone_complexity(
    genomes: Sequence[ConeGenome],
    library: Sequence[Leg],
    condition: str,
    call_cost: float = DEFAULT_CALL_COST,
    charge_library: bool = True,
    free_legs: frozenset = frozenset(),
) -> float:
    """Joint cone description length.

    free_legs holds library indices whose definitions were already paid for in
    an earlier growth phase; under shared discounting they are not re-charged
    (marginal accounting for a growing library). The no_share control still
    charges every call in full — it is the no-discount accounting by design.
    """
    total = sum(genome_complexity(genome, library, condition, call_cost) for genome in genomes)
    if condition in ("shared", "witness") and charge_library:
        for leg_index in legs_used(genomes, library):
            if leg_index in free_legs:
                continue
            total += leg_def_complexity(library[leg_index])
    return total


# ---------------------------------------------------------------------------
# Leg lifting (stage B: the encapsulation mutation)
# ---------------------------------------------------------------------------

def lift_leg(genome: ConeGenome, name: str) -> Leg:
    """Lift an inline champion into a library leg.

    Keeps only move actions (a leg has no SET_FOCUS or CALL), drops rules left
    empty, and installs an explicit (state, HERE) -> RETURN boundary for every
    surviving source state. The RETURN rules are paid for in the leg's
    definition complexity; nothing returns implicitly (pitfall P3).
    """
    rules: List[ConeRule] = []
    for rule in genome.rules:
        moves = tuple(action for action in rule.actions if is_move(action))
        if not moves or rule.observation == HERE_OBS:
            continue
        rules.append(ConeRule(rule.state, rule.observation, moves, rule.next_state))
    states = sorted({rule.state for rule in rules}) or [0]
    for state in states:
        rules.append(ConeRule(state, HERE_OBS, (RETURN_ACTION,), state))
    return Leg(name=name, genome=ConeGenome(state_count=genome.state_count, rules=deduplicate_cone_rules(rules)))


def genome_has_return(genome: ConeGenome) -> bool:
    return any(action == RETURN_ACTION for rule in genome.rules for action in rule.actions)


def witness_seek_leg() -> Leg:
    """Hand-written minimal seek leg: a representability witness, not a search result."""
    rules = []
    for observation in range(len(OBS_LABELS)):
        if observation == HERE_OBS:
            rules.append(ConeRule(0, observation, (RETURN_ACTION,), 0))
            continue
        dx = observation % 3 - 1
        dy = observation // 3 - 1
        if dy < 0:
            move = MOVE_NAMES.index("UP")
        elif dy > 0:
            move = MOVE_NAMES.index("DOWN")
        elif dx < 0:
            move = MOVE_NAMES.index("LEFT")
        else:
            move = MOVE_NAMES.index("RIGHT")
        rules.append(ConeRule(0, observation, (move,), 0))
    return Leg(name="witness_seek", genome=ConeGenome(state_count=1, rules=rules))


def witness_flee_leg() -> Leg:
    """Hand-written minimal flee leg: move opposite the azimuth until safe."""
    rules = []
    for observation in range(len(OBS_LABELS)):
        if observation == HERE_OBS:
            rules.append(ConeRule(0, observation, (RETURN_ACTION,), 0))
            continue
        dx = observation % 3 - 1
        dy = observation // 3 - 1
        if dy < 0:
            move = MOVE_NAMES.index("DOWN")
        elif dy > 0:
            move = MOVE_NAMES.index("UP")
        elif dx < 0:
            move = MOVE_NAMES.index("RIGHT")
        else:
            move = MOVE_NAMES.index("LEFT")
        rules.append(ConeRule(0, observation, (move,), 0))
    return Leg(name="witness_flee", genome=ConeGenome(state_count=1, rules=rules))


def witness_gluing(task: TaskSpec, seek_index: int = 0, flee_index: int = 1) -> ConeGenome:
    """Hand-written controllers that glue the witness legs into each task."""
    seek = lambda channel: call_action(seek_index, channel)  # noqa: E731
    flee = lambda channel: call_action(flee_index, channel)  # noqa: E731
    sequences: Dict[str, Tuple[int, ...]] = {
        "forage": (seek(FOOD_CHANNEL),),
        "homing": (seek(HOME_CHANNEL),),
        "forage_then_home": (seek(FOOD_CHANNEL), seek(HOME_CHANNEL)),
        "flee": (flee(HAZARD_CHANNEL),),
        "forage_flee": (seek(FOOD_CHANNEL), flee(HAZARD_CHANNEL)),
        "flee_then_home": (flee(HAZARD_CHANNEL), seek(HOME_CHANNEL)),
    }
    calls = sequences[task.name]
    rules = [ConeRule(idx, ANY_OBS, (call,), idx + 1) for idx, call in enumerate(calls)]
    return ConeGenome(state_count=len(calls) + 1, rules=rules)


# ---------------------------------------------------------------------------
# Evolution
# ---------------------------------------------------------------------------

@dataclass
class EvolutionResult:
    genome: ConeGenome
    train_loss: float
    free_energy: float
    saw_call_champion: bool


def evolve_cone_task(
    task: TaskSpec,
    train_levels: Sequence[ConeLevel],
    allowed_actions: Sequence[int],
    library: Sequence[Leg],
    condition: str,
    lambda_value: float,
    seed: int,
    population_size: int = 80,
    generations: int = 45,
    state_count: int = 3,
    initial_rule_count: int = 8,
    max_rules: int = 14,
    max_rule_length: int = 2,
    mutation_rate: float = 0.10,
    elite_fraction: float = 0.10,
    call_cost: float = DEFAULT_CALL_COST,
    max_steps: int = 44,
) -> EvolutionResult:
    """Evolve one task solver under one condition's local accounting.

    Local selection charges each genome its own complexity: CALL costs
    call_cost under shared/witness accounting and call_cost + def(leg) under
    no_share. Shared leg definitions are charged once at the joint selection
    stage (stage D), not here; see reconciliation log R7.
    """
    rng = random.Random(seed)
    actions = tuple(allowed_actions)

    def free_energy(genome: ConeGenome) -> Tuple[float, float]:
        evaluation = evaluate_cone_task(genome, library, train_levels, task, max_steps=max_steps)
        complexity = genome_complexity(genome, library, condition, call_cost)
        return evaluation.loss + lambda_value * complexity, evaluation.loss

    population = [
        ConeGenome.random(
            rng,
            actions,
            state_count=state_count,
            initial_rule_count=initial_rule_count,
            max_rule_length=max_rule_length,
        )
        for _ in range(population_size)
    ]
    elite_count = max(1, int(population_size * elite_fraction))
    best_genome = population[0]
    best_f, best_loss = free_energy(best_genome)
    saw_call_champion = False

    for generation in range(generations):
        scored = []
        for genome in population:
            f_value, loss = free_energy(genome)
            scored.append((f_value, loss, genome))
        scored.sort(key=lambda item: item[0])

        if scored[0][0] < best_f:
            best_f, best_loss, best_genome = scored[0]
        if any(genome.call_references() for _f, _l, genome in scored[:elite_count]):
            saw_call_champion = True

        next_population = [genome for _f, _l, genome in scored[:elite_count]]
        fitness_pairs = [(-f_value, genome) for f_value, _loss, genome in scored]
        while len(next_population) < population_size:
            parent_a = tournament_select(fitness_pairs, rng)
            parent_b = tournament_select(fitness_pairs, rng)
            child = parent_a.crossover(parent_b, rng).mutate(
                rng,
                actions,
                rate=mutation_rate,
                max_rule_length=max_rule_length,
                max_rules=max_rules,
            )
            next_population.append(child)
        population = next_population

    return EvolutionResult(
        genome=best_genome,
        train_loss=best_loss,
        free_energy=best_f,
        saw_call_champion=saw_call_champion,
    )


def tournament_select(
    scored: Sequence[Tuple[float, ConeGenome]],
    rng: random.Random,
    size: int = 5,
) -> ConeGenome:
    contenders = rng.sample(list(scored), min(size, len(scored)))
    return max(contenders, key=lambda item: item[0])[1]


# ---------------------------------------------------------------------------
# Joint cone evolution (cold leg bodies, no lifting)
# ---------------------------------------------------------------------------

@dataclass
class JointCone:
    """One joint individual: evolvable leg bodies plus one controller per task."""

    legs: List[ConeGenome]
    controllers: Dict[str, ConeGenome]


@dataclass
class JointEvolutionResult:
    legs: List[ConeGenome]
    controllers: Dict[str, ConeGenome]
    train_loss: float
    free_energy: float
    saw_call_champion: bool


def evolve_joint_cone(
    task_levels: Dict[str, Tuple[TaskSpec, Sequence[ConeLevel]]],
    condition: str,
    lambda_value: float,
    seed: int,
    population_size: int = 150,
    generations: int = 80,
    state_count: int = 3,
    leg_state_count: int = 2,
    initial_rule_count: int = 8,
    max_rules: int = 14,
    max_rule_length: int = 2,
    mutation_rate: float = 0.10,
    elite_fraction: float = 0.10,
    call_cost: float = DEFAULT_CALL_COST,
    max_steps: int = 44,
    frozen_legs: Sequence[Leg] = (),
    evolved_legs: int = 1,
) -> JointEvolutionResult:
    """Co-evolve leg bodies and per-task gluings from random initialization.

    Unlike evolve_cone_task (one task, library frozen), the joint individual
    contains every support task, so the true cone accounting — each used leg
    definition charged once across all tasks under shared discounting — is
    computable and charged during local selection, not only at stage D.
    Controllers keep the full inline action set, so a genome remains free to
    ignore its legs; calls must be selected, never forced.

    frozen_legs are inherited library entries from an earlier growth phase:
    callable, immutable, and (under shared discounting) already paid for, so
    their definitions are excluded from the charged complexity. Only
    evolved_legs new leg bodies are mutated and charged.
    """
    rng = random.Random(seed)
    library_size = len(frozen_legs) + evolved_legs
    controller_acts = controller_actions(library_size)
    leg_acts = leg_actions()
    names = sorted(task_levels)
    free_leg_indices = frozenset(range(len(frozen_legs)))

    def make_library(individual: JointCone) -> List[Leg]:
        return list(frozen_legs) + [
            Leg(f"joint_{idx}", genome) for idx, genome in enumerate(individual.legs)
        ]

    def random_individual() -> JointCone:
        return JointCone(
            legs=[
                ConeGenome.random(
                    rng, leg_acts, state_count=leg_state_count,
                    initial_rule_count=initial_rule_count, max_rule_length=max_rule_length,
                )
                for _ in range(evolved_legs)
            ],
            controllers={
                name: ConeGenome.random(
                    rng, controller_acts, state_count=state_count,
                    initial_rule_count=initial_rule_count, max_rule_length=max_rule_length,
                )
                for name in names
            },
        )

    def free_energy(individual: JointCone) -> Tuple[float, float]:
        library = make_library(individual)
        loss = 0.0
        for name in names:
            task, levels = task_levels[name]
            loss += evaluate_cone_task(
                individual.controllers[name], library, levels, task, max_steps=max_steps
            ).loss
        complexity = cone_complexity(
            list(individual.controllers.values()), library, condition,
            call_cost=call_cost, free_legs=free_leg_indices,
        )
        return loss + lambda_value * complexity, loss

    def mutate(individual: JointCone) -> JointCone:
        return JointCone(
            legs=[
                genome.mutate(
                    rng, leg_acts, rate=mutation_rate, max_rule_length=max_rule_length, max_rules=max_rules
                )
                for genome in individual.legs
            ],
            controllers={
                name: genome.mutate(
                    rng, controller_acts, rate=mutation_rate,
                    max_rule_length=max_rule_length, max_rules=max_rules,
                )
                for name, genome in individual.controllers.items()
            },
        )

    def crossover(a: JointCone, b: JointCone) -> JointCone:
        return JointCone(
            legs=[
                leg_a.crossover(leg_b, rng) for leg_a, leg_b in zip(a.legs, b.legs)
            ],
            controllers={
                name: a.controllers[name].crossover(b.controllers[name], rng)
                for name in names
            },
        )

    population = [random_individual() for _ in range(population_size)]
    elite_count = max(1, int(population_size * elite_fraction))
    best = population[0]
    best_f, best_loss = free_energy(best)
    saw_call_champion = False

    for _generation in range(generations):
        scored = []
        for individual in population:
            f_value, loss = free_energy(individual)
            scored.append((f_value, loss, individual))
        scored.sort(key=lambda item: item[0])
        if scored[0][0] < best_f:
            best_f, best_loss, best = scored[0]
        if any(
            any(genome.call_references() for genome in individual.controllers.values())
            for _f, _l, individual in scored[:elite_count]
        ):
            saw_call_champion = True

        next_population = [individual for _f, _l, individual in scored[:elite_count]]
        fitness_pairs = [(-f_value, individual) for f_value, _loss, individual in scored]
        while len(next_population) < population_size:
            contenders_a = rng.sample(fitness_pairs, min(5, len(fitness_pairs)))
            contenders_b = rng.sample(fitness_pairs, min(5, len(fitness_pairs)))
            parent_a = max(contenders_a, key=lambda item: item[0])[1]
            parent_b = max(contenders_b, key=lambda item: item[0])[1]
            next_population.append(mutate(crossover(parent_a, parent_b)))
        population = next_population

    return JointEvolutionResult(
        legs=list(best.legs),
        controllers=dict(best.controllers),
        train_loss=best_loss,
        free_energy=best_f,
        saw_call_champion=saw_call_champion,
    )

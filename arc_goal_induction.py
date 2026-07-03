"""
ARC goal induction: the foraging goal-induction loop lifted onto the ARC
connector's scene features and game score.

The foraging loop (cone_goal_induction.py) inferred which hand-defined feature
subset (food/home/safe) a hidden reward tracked, then compiled it into a cone.
This module runs the SAME free-energy induction over ARC scene features — now
parameterized by colour (the ARC analogue of a channel) — using a hidden game
score as the only reward, then compiles the inferred goal into a cone that
issues CALL(leg, colour-slot) actions against the scene.

The correspondence (COLIMIT_CONE_APPROACH.md Sections 11-13):

    foraging                      ARC
    --------                      ---
    channel (FOOD/HOME/HAZARD)    colour slot (an object colour)
    feature food/home/safe        predicate@colour atom (clear@c / avoid@c)
    hidden task loss              hidden game score (scalar)
    seek/flee leg                 same channel-blind legs, bound to a colour

Goal atoms:
    clear@c  collect (remove on contact) all objects of colour c   -> seek leg
    avoid@c  end at least SAFE_RADIUS from every colour-c object    -> flee leg

The agent observes the atoms and the scalar score, never the game's objective
or which colours are collectible. It must infer BOTH the relevant colour and
the relevant relation. Feature DISCOVERY from raw frames remains the standing
open problem; this module demonstrates goal induction over a given atom
vocabulary, exactly as the foraging version did over given features.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cone_foraging as cf
import cone_goal_induction as gi
import arc_agi3_adapter as arc

Cell = Tuple[int, int]
Color = int

SAFE_RADIUS = cf.SAFE_RADIUS
PREDICATES = ("clear", "avoid")
PRED_MODE = {"clear": "reach", "avoid": "avoid"}  # seek to clear, flee to avoid
PRED_LEG = {"clear": "seek", "avoid": "flee"}


def atom(pred: str, color: Color) -> str:
    return f"{pred}@{color}"


def parse_atom(name: str) -> Tuple[str, Color]:
    pred, color = name.split("@")
    return pred, int(color)


# ---------------------------------------------------------------------------
# A small ARC-shaped game with a hidden objective over colour atoms
# ---------------------------------------------------------------------------

@dataclass
class GoalGame:
    """SYNTHETIC, NOT REAL ARC. A reproducible coloured-grid fixture that mimics
    the ARC interface shape so the goal-induction/discovery machinery can be
    validated deterministically and offline. Objects of a `collect` colour
    vanish when the avatar steps on them; `hazard`/distractor colours persist;
    the hidden objective is a subset of colour atoms and the score is their mean
    satisfaction. For real ARC frames use arc_agi3_adapter.ArcEnv (live) — see
    experiments/arc_live_report.md."""

    width: int
    height: int
    avatar_color: Color
    avatar: Cell
    objects: List[Tuple[Cell, Color]]
    collect_colors: frozenset
    initial_counts: Dict[Color, int]
    state: arc.GameState = arc.GameState.IN_PROGRESS
    steps: int = 0
    max_steps: int = 80

    def colors_present(self) -> List[Color]:
        return sorted({c for _cell, c in self.objects})

    def render(self) -> arc.Frame:
        frame = [[arc.BACKGROUND] * self.width for _ in range(self.height)]
        for (x, y), color in self.objects:
            frame[y][x] = color
        ax, ay = self.avatar
        frame[ay][ax] = self.avatar_color
        return frame

    def step(self, action: arc.GameAction) -> None:
        if self.state != arc.GameState.IN_PROGRESS:
            return
        self.steps += 1
        dx, dy = arc.ACTION_TO_DELTA.get(action, (0, 0))
        nx, ny = self.avatar[0] + dx, self.avatar[1] + dy
        if 0 <= nx < self.width and 0 <= ny < self.height:
            self.avatar = (nx, ny)
        # Collect: remove a collectible object the avatar now stands on.
        self.objects = [
            (cell, color) for cell, color in self.objects
            if not (cell == self.avatar and color in self.collect_colors)
        ]
        if self.steps >= self.max_steps:
            self.state = arc.GameState.GAME_OVER

    def remaining(self, color: Color) -> List[Cell]:
        return [cell for cell, c in self.objects if c == color]


def make_goal_game(
    seed: int,
    objective: Sequence[str],
    candidate_colors: Sequence[Color],
    width: int = 14,
    height: int = 14,
) -> GoalGame:
    # Grid is sized so the minimal (wall-blind) witness flee leg has escape
    # room to reach SAFE_RADIUS by pure antipodal motion — the same rationale
    # as the foraging hazard placement. Evolved legs would relax this.
    import random

    rng = random.Random(seed)
    collect = frozenset(parse_atom(a)[1] for a in objective if parse_atom(a)[0] == "clear")
    cells = [(x, y) for y in range(height) for x in range(width)]
    rng.shuffle(cells)
    avatar = cells.pop()
    objects: List[Tuple[Cell, Color]] = []
    counts: Dict[Color, int] = {}
    # Two or three objects per candidate colour, placed at random cells.
    for color in candidate_colors:
        n = 3 if color in collect else 2
        for _ in range(n):
            objects.append((cells.pop(), color))
        counts[color] = n
    return GoalGame(
        width=width, height=height, avatar_color=4, avatar=avatar,
        objects=objects, collect_colors=collect, initial_counts=counts,
    )


# ---------------------------------------------------------------------------
# Observable scene atoms and the hidden score
# ---------------------------------------------------------------------------

def scene_atoms(game: GoalGame, candidate_colors: Sequence[Color]) -> Dict[str, float]:
    """Graded, observable satisfaction of each colour atom from the current
    game state. The agent can compute all of these; none name the objective."""
    feats: Dict[str, float] = {}
    max_dist = max(1, game.width + game.height - 2)
    ax, ay = game.avatar
    for color in candidate_colors:
        remaining = game.remaining(color)
        initial = game.initial_counts.get(color, 0)
        # clear: fraction removed (only meaningful if collectible, else stays low)
        feats[atom("clear", color)] = 1.0 - (len(remaining) / initial if initial else 0.0)
        # avoid: distance to nearest remaining object, normalized to safe radius
        if not remaining:
            feats[atom("avoid", color)] = 1.0
        else:
            d = min(abs(cx - ax) + abs(cy - ay) for cx, cy in remaining)
            feats[atom("avoid", color)] = min(1.0, d / SAFE_RADIUS)
    return feats


def candidate_atoms(candidate_colors: Sequence[Color]) -> List[str]:
    return [atom(pred, color) for color in candidate_colors for pred in PREDICATES]


def hidden_score(game: GoalGame, objective: Sequence[str], candidate_colors: Sequence[Color]) -> float:
    """The scalar the environment exposes: mean satisfaction of the hidden
    objective's atoms. The agent sees this number, not the objective."""
    if not objective:
        return 1.0
    feats = scene_atoms(game, candidate_colors)
    return sum(feats[a] for a in objective) / len(objective)


# ---------------------------------------------------------------------------
# Cones over colour slots and their execution on a game
# ---------------------------------------------------------------------------

Phase = Tuple[str, Color]  # (behavior in {"seek","flee"}, colour)


def goal_to_cone(goal: Sequence[str]) -> List[Phase]:
    """Compile an inferred atom set into ordered cone phases: clear-atoms first
    (seek to collect), then avoid-atoms (flee to safety)."""
    clears = [parse_atom(a) for a in goal if parse_atom(a)[0] == "clear"]
    avoids = [parse_atom(a) for a in goal if parse_atom(a)[0] == "avoid"]
    phases: List[Phase] = [("seek", c) for _p, c in clears]
    phases += [("flee", c) for _p, c in avoids]
    return phases


def _leg(behavior: str) -> cf.Leg:
    return cf.witness_seek_leg() if behavior == "seek" else cf.witness_flee_leg()


def run_cone(game: GoalGame, phases: Sequence[Phase], max_steps: int = 80) -> None:
    """Drive the game through the cone's phases. Each phase runs a channel-blind
    leg bound to a colour slot until it RETURNs (the slot reads HERE) or the
    game ends — the ARC analogue of CALL(leg, colour)."""
    for behavior, color in phases:
        leg = _leg(behavior)
        leg_map = leg.genome.rule_map()
        mode = PRED_MODE["clear" if behavior == "seek" else "avoid"]
        substate = 0
        guard = 0
        while game.state == arc.GameState.IN_PROGRESS and game.steps < max_steps and guard < max_steps * 2:
            guard += 1
            scene = arc.extract_scene(game.render(), avatar_color=game.avatar_color)
            obs = arc.slot_observation(scene, color, mode=mode, safe_radius=SAFE_RADIUS)
            rule = leg_map.get((substate, obs)) or leg_map.get((substate, cf.ANY_OBS))
            if rule is None:
                break
            advanced = False
            for action in rule.actions:
                if cf.is_move(action):
                    game.step(arc.MOVE_TO_ACTION[action])
                    advanced = True
                elif action == cf.RETURN_ACTION:
                    advanced = False
                    break
            else:
                substate = rule.next_state
                if advanced:
                    continue
                break
            break


# ---------------------------------------------------------------------------
# Hidden task: a seeded family of game instances sharing one objective
# ---------------------------------------------------------------------------

@dataclass
class HiddenArcTask:
    _objective: Tuple[str, ...]
    candidate_colors: Tuple[Color, ...]
    seeds: Tuple[int, ...]
    max_steps: int = 80
    # When a discovered vocabulary is supplied, the agent's observable atoms are
    # the discovered ones, computed from raw frames (reset + final), and the
    # avatar colour is the discovered one. The hidden score is always the true
    # objective's mean satisfaction, computed the same frame-based way.
    vocabulary: Optional[object] = None  # arc_scene_atoms.DiscoveredVocabulary

    def _instance(self, index: int) -> GoalGame:
        return make_goal_game(self.seeds[index], self._objective, self.candidate_colors)

    def _frame_score(self, frame0, frame1, avatar_color: Color) -> float:
        import arc_scene_atoms as sa
        if not self._objective:
            return 1.0
        return sum(sa.evaluate_atom(a, frame0, frame1, avatar_color) for a in self._objective) / len(self._objective)

    def evaluate(self, phases: Sequence[Phase], indices: Sequence[int]) -> List[Tuple[Dict[str, float], float]]:
        if self.vocabulary is not None:
            return self._evaluate_discovered(phases, indices)
        out = []
        for i in indices:
            game = self._instance(i)
            run_cone(game, phases, max_steps=self.max_steps)
            atoms = scene_atoms(game, self.candidate_colors)
            score = hidden_score(game, self._objective, self.candidate_colors)
            out.append((atoms, score))
        return out

    def _evaluate_discovered(self, phases: Sequence[Phase], indices: Sequence[int]) -> List[Tuple[Dict[str, float], float]]:
        vocab = self.vocabulary
        out = []
        for i in indices:
            game = self._instance(i)
            frame0 = game.render()
            run_cone(game, phases, max_steps=self.max_steps)
            frame1 = game.render()
            atoms = {a.name: a.evaluate(frame0, frame1, vocab.avatar_color) for a in vocab.atoms}
            score = self._frame_score(frame0, frame1, vocab.avatar_color)
            out.append((atoms, score))
        return out

    def solved_fraction(self, phases: Sequence[Phase], indices: Sequence[int]) -> float:
        solved = 0
        for i in indices:
            game = self._instance(i)
            frame0 = game.render() if self.vocabulary is not None else None
            run_cone(game, phases, max_steps=self.max_steps)
            if self.vocabulary is not None:
                score = self._frame_score(frame0, game.render(), self.vocabulary.avatar_color)
            else:
                score = hidden_score(game, self._objective, self.candidate_colors)
            if score >= 0.999:
                solved += 1
        return solved / max(1, len(indices))


# ---------------------------------------------------------------------------
# Probe cones and the lifted induction loop
# ---------------------------------------------------------------------------

def probe_phase_sets(candidate_colors: Sequence[Color]) -> Dict[str, List[Phase]]:
    """Single-behavior probes per colour, plus a few informative pairs and a
    do-nothing probe. These independently toggle the atoms so the score model
    is identifiable (the ARC analogue of cone_goal_induction.probe_cones)."""
    probes: Dict[str, List[Phase]] = {"noop": []}
    singles: List[Phase] = []
    for color in candidate_colors:
        for behavior in ("seek", "flee"):
            phases = [(behavior, color)]
            probes[f"{behavior}@{color}"] = phases
            singles.append((behavior, color))
    # Pairs: each seek with each flee of a different colour (collect-then-avoid).
    seeks = [p for p in singles if p[0] == "seek"]
    flees = [p for p in singles if p[0] == "flee"]
    for s in seeks:
        for f in flees:
            if s[1] != f[1]:
                probes[f"seek@{s[1]}+flee@{f[1]}"] = [s, f]
    # Pairs of two seeks (clear two colours).
    for a, b in itertools.combinations(seeks, 2):
        probes[f"seek@{a[1]}+seek@{b[1]}"] = [a, b]
    return probes


@dataclass
class ArcInductionResult:
    inferred_goal: Tuple[str, ...]
    free_energy: float
    probe_episodes: int
    rounds: int
    order: List[str] = field(default_factory=list)


def discover_and_induce(
    objective: Sequence[str],
    candidate_colors: Sequence[Color],
    seeds: Sequence[int],
    probe_indices: Sequence[int],
    lam: float = 0.05,
    budget: int = 8,
    max_goal_size: int = 2,
    variance_eps: float = 0.02,
) -> Tuple["ArcInductionResult", object]:
    """Full raw-frame pipeline: discover the avatar and atom vocabulary from
    frames (no hand-given colours or evaluators), then induce the hidden goal
    over the DISCOVERED atoms. Returns (induction result, discovered vocabulary).

    The candidate_colors / objective here configure the hidden environment; the
    agent does not see them — it rediscovers the colours from frames and infers
    the objective from the score."""
    import arc_scene_atoms as sa

    def game_factory(seed: int) -> GoalGame:
        return make_goal_game(seed, objective, candidate_colors)

    explore = list(probe_phase_sets(candidate_colors).values())
    vocab = sa.discover_vocabulary(
        game_factory, lambda g, ph: run_cone(g, ph), explore, list(seeds), variance_eps=variance_eps
    )
    task = HiddenArcTask(
        _objective=tuple(objective), candidate_colors=tuple(vocab.colors),
        seeds=tuple(seeds), vocabulary=vocab,
    )
    result = _induce_over_atoms(task, vocab.atom_names, probe_indices, lam, budget, max_goal_size)
    return result, vocab


def induce_arc_goal(
    task: HiddenArcTask,
    probe_indices: Sequence[int],
    lam: float = 0.05,
    budget: int = 8,
    max_goal_size: int = 2,
    converge_margin: float = 0.02,
) -> ArcInductionResult:
    """Disagreement-driven induction over the colour atoms implied by the task's
    candidate colours (the oracle-vocabulary path)."""
    atoms = candidate_atoms(task.candidate_colors)
    return _induce_over_atoms(task, atoms, probe_indices, lam, budget, max_goal_size, converge_margin)


def _induce_over_atoms(
    task: HiddenArcTask,
    atoms: Sequence[str],
    probe_indices: Sequence[int],
    lam: float = 0.05,
    budget: int = 8,
    max_goal_size: int = 2,
    converge_margin: float = 0.02,
) -> ArcInductionResult:
    """Disagreement-driven induction over an explicit atom set, reusing the
    free-energy core from cone_goal_induction. Shared by the oracle-vocabulary
    path (candidate_atoms) and the discovered-vocabulary path (vocab atoms).
    Each round runs the untested probe whose outcomes most separate the top
    candidate goals; stops when the best goal's free-energy margin over the
    runner-up is comfortable."""
    atoms = list(atoms)
    probes = probe_phase_sets(task.candidate_colors)
    untested = dict(probes)
    observations: List[Tuple[Dict[str, float], float]] = []
    order: List[str] = []
    probe_episodes = 0
    rounds = 0

    while untested and rounds < budget:
        rounds += 1
        ranked = sorted(gi.candidate_goals(atoms, max_goal_size),
                        key=lambda g: gi.goal_free_energy(g, observations, lam))
        top = ranked[: min(3, len(ranked))]
        if not observations:
            # Seed with the first single-seek probe (basic, informative).
            choice = next((n for n in untested if n.startswith("seek@")), next(iter(untested)))
        else:
            scored = []
            for name, phases in untested.items():
                feats = [f for f, _s in task.evaluate(phases, probe_indices)]
                scored.append((gi.discrimination(feats, top), name))
            scored.sort(reverse=True)
            choice = scored[0][1]
        phases = untested.pop(choice)
        order.append(choice)
        results = task.evaluate(phases, probe_indices)
        observations.extend(results)
        probe_episodes += len(results)

        ranked_fe = sorted((gi.goal_free_energy(g, observations, lam), g)
                           for g in gi.candidate_goals(atoms, max_goal_size))
        if len(ranked_fe) >= 2 and ranked_fe[1][0] - ranked_fe[0][0] >= converge_margin:
            break

    goal, fe = gi.induce_goal(observations, lam, feature_keys=atoms, max_size=max_goal_size)
    return ArcInductionResult(goal, fe, probe_episodes, rounds, order)

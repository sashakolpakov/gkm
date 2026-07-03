"""
Goal induction for cone foraging: infer a hidden goal from a scalar reward,
then compile it into a cone.

Every prior experiment in this repository TELLS the agent its task through the
loss function. Real ARC-AGI-3 games hide the goal and expose only a scalar
score; the agent must infer what the reward rewards. This module builds that
missing core (COLIMIT_CONE_APPROACH.md R12 gap 2) in the familiar foraging
context, designed so the same loop lifts to the ARC connector.

The reduction: a hidden task rewards the agent for the mean satisfaction of a
hidden subset of OBSERVABLE outcome features (food collected, at home, safe
from hazard). The agent can see the features and the scalar reward, but not
which features the reward averages. Goal induction is therefore free-energy
model selection over feature subsets:

    F(S) = misfit(predict reward from S across probes) + lambda * |S|

the same MDL feature-selection the Bongard / abstraction experiments run, now
over GOAL features learned from interaction rather than labels. The inferred
goal is compiled to a bound cone (priced bindings, v3): one phase per feature,
seek legs for food/home, a flee leg for safety. Disagreement-driven
exploration picks the next probe cone whose outcome most separates the top
candidate goals, so the goal is identified from few interactions.

The agent NEVER reads the task name or its loss decomposition; it sees only
outcome features and the scalar reward the environment returns.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cone_foraging as cf
import cone_foraging_bound as cb

FEATURES = ("food", "home", "safe")
FEATURE_CHANNEL = {"food": cf.FOOD_CHANNEL, "home": cf.HOME_CHANNEL, "safe": cf.HAZARD_CHANNEL}
# A sensible phase order when several features are required: collect, then get
# safe, then settle home. (Order matters only for the compiled behavior, not
# for induction.)
FEATURE_ORDER = ("food", "safe", "home")


# ---------------------------------------------------------------------------
# Observable outcome features and the hidden scalar reward
# ---------------------------------------------------------------------------

def outcome_features(episode, level: cf.ConeLevel) -> Dict[str, float]:
    """Graded, fully observable features of an episode outcome. The agent can
    compute all of these from the scene it sees; none of them name the task."""
    food = episode.collected / episode.total_food if episode.total_food else 1.0
    max_dist = max(1, level.width + level.height - 2)
    home = 1.0 - cf.manhattan(episode.final_position, level.home) / max_dist
    distance = cf.final_hazard_distance(episode, level)
    safe = 1.0 if distance is None else min(1.0, distance / cf.SAFE_RADIUS)
    return {"food": food, "home": home, "safe": safe}


def task_goal_features(task: cf.TaskSpec) -> Tuple[str, ...]:
    feats = []
    if task.requires_food:
        feats.append("food")
    if task.requires_home:
        feats.append("home")
    if task.requires_safe:
        feats.append("safe")
    return tuple(feats)


def hidden_reward(episode, level: cf.ConeLevel, task: cf.TaskSpec) -> float:
    """The scalar the environment exposes: the mean satisfaction of the hidden
    goal's features. The agent sees this number but not which features it
    averages."""
    feats = task_goal_features(task)
    if not feats:
        return 1.0
    f = outcome_features(episode, level)
    return sum(f[k] for k in feats) / len(feats)


# ---------------------------------------------------------------------------
# Hidden-task environment (exposes features + scalar reward only)
# ---------------------------------------------------------------------------

@dataclass
class HiddenTask:
    """Wraps a task + levels. The agent submits a cone (controller + library)
    and receives, per level, the observable outcome features and the scalar
    reward. The task identity is private (leading underscore, never read by the
    inducer)."""

    _task: cf.TaskSpec
    levels: List[cf.ConeLevel]
    max_steps: int = 44

    def evaluate(
        self, controller: cb.BoundGenome, library: Sequence[cf.Leg], level_indices: Sequence[int]
    ) -> List[Tuple[Dict[str, float], float]]:
        out = []
        for i in level_indices:
            level = self.levels[i]
            episode = cb.run_bound_episode(controller, library, level, self._task, max_steps=self.max_steps)
            out.append((outcome_features(episode, level), hidden_reward(episode, level, self._task)))
        return out

    def solved_fraction(self, controller: cb.BoundGenome, library: Sequence[cf.Leg], level_indices: Sequence[int]) -> float:
        solved = 0
        for i in level_indices:
            level = self.levels[i]
            episode = cb.run_bound_episode(controller, library, level, self._task, max_steps=self.max_steps)
            solved += int(cf.episode_solved(episode, level, self._task))
        return solved / max(1, len(level_indices))


# ---------------------------------------------------------------------------
# Compiling a goal (feature subset) into a bound cone
# ---------------------------------------------------------------------------

def goal_library() -> List[cf.Leg]:
    return [cf.witness_seek_leg(), cf.witness_flee_leg()]


def compile_goal(goal: Sequence[str]) -> Tuple[cb.BoundGenome, List[cf.Leg]]:
    """One phase per required feature: seek the food/home channel, flee the
    hazard channel. Empty goal -> a do-nothing controller."""
    library = goal_library()
    phases = [f for f in FEATURE_ORDER if f in goal]
    rules = []
    for idx, feat in enumerate(phases):
        channel = FEATURE_CHANNEL[feat]
        leg_index = 1 if feat == "safe" else 0  # flee leg for safety, seek leg otherwise
        rules.append(cb.BoundRule(idx, channel, cb.ANY_OBS, (cb.call_action(leg_index, channel),), idx + 1))
    state_count = max(1, len(phases) + 1)
    return cb.BoundGenome(state_count=state_count, rules=rules), library


def probe_cones() -> Dict[str, Tuple[cb.BoundGenome, List[cf.Leg]]]:
    """A diverse probe set: each single feature alone, plus a couple of pairs,
    plus do-nothing. These independently toggle the features so the reward
    model is identifiable."""
    names = [(),
             ("food",), ("home",), ("safe",),
             ("food", "home"), ("safe", "home"), ("food", "safe")]
    return {("+".join(g) or "noop"): compile_goal(g) for g in names}


# ---------------------------------------------------------------------------
# Free-energy goal induction
# ---------------------------------------------------------------------------

Observation = Tuple[Dict[str, float], float]  # (features, reward)


def candidate_goals(
    feature_keys: Sequence[str] = FEATURES, max_size: Optional[int] = None
) -> List[Tuple[str, ...]]:
    """All non-empty feature subsets up to max_size. Feature-key-agnostic so the
    same free-energy core serves foraging (food/home/safe) and the ARC lift
    (predicate@colour atoms)."""
    keys = tuple(feature_keys)
    limit = max_size if max_size is not None else len(keys)
    goals: List[Tuple[str, ...]] = []
    for size in range(1, min(limit, len(keys)) + 1):
        goals.extend(itertools.combinations(keys, size))
    return goals


def goal_free_energy(goal: Sequence[str], observations: Sequence[Observation], lam: float) -> float:
    """Misfit of predicting reward as the mean of the goal's features, plus a
    complexity penalty on the number of features. This is F = R + lambda*C over
    goal models."""
    if not observations:
        return lam * len(goal)
    err = 0.0
    for features, reward in observations:
        predicted = sum(features[k] for k in goal) / len(goal)
        err += (predicted - reward) ** 2
    return err / len(observations) + lam * len(goal)


def induce_goal(
    observations: Sequence[Observation],
    lam: float,
    feature_keys: Sequence[str] = FEATURES,
    max_size: Optional[int] = None,
) -> Tuple[Tuple[str, ...], float]:
    goals = candidate_goals(feature_keys, max_size)
    scored = [(goal_free_energy(goal, observations, lam), len(goal), goal) for goal in goals]
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    best = scored[0]
    return best[2], best[0]


def discrimination(cone_features: Sequence[Dict[str, float]], top_goals: Sequence[Sequence[str]]) -> float:
    """How much would running this cone separate the top candidate goals? The
    spread of predicted rewards across goals, averaged over the cone's probe
    levels. Larger spread => more informative probe (disagreement-driven)."""
    if len(top_goals) < 2:
        return 0.0
    total = 0.0
    for features in cone_features:
        predictions = [sum(features[k] for k in goal) / len(goal) for goal in top_goals]
        total += max(predictions) - min(predictions)
    return total / max(1, len(cone_features))


@dataclass
class InductionResult:
    inferred_goal: Tuple[str, ...]
    free_energy: float
    probe_episodes: int
    rounds: int
    order: List[str] = field(default_factory=list)


def induce_active(
    env: HiddenTask,
    probe_level_indices: Sequence[int],
    lam: float = 0.05,
    budget: int = 6,
    converge_margin: float = 0.02,
) -> InductionResult:
    """Disagreement-driven loop: each round, run the untested probe cone whose
    outcome most separates the current top candidate goals; update the goal
    posterior; stop when the best goal's free-energy margin over the runner-up
    is comfortable. Reports rounds and total probe episodes (sample
    efficiency)."""
    cones = probe_cones()
    untested = dict(cones)
    observations: List[Observation] = []
    order: List[str] = []
    probe_episodes = 0

    # Seed with the cheapest, most basic probe so we always have some signal.
    rounds = 0
    while untested and rounds < budget:
        rounds += 1
        # Rank current candidate goals to know what to discriminate.
        ranked = sorted(candidate_goals(), key=lambda g: goal_free_energy(g, observations, lam))
        top = ranked[: min(3, len(ranked))]
        # Pick the untested probe cone with the highest discrimination; on the
        # first round (no observations) fall back to a fixed informative probe.
        def cone_outcome_features(item) -> List[Dict[str, float]]:
            _name, (controller, library) = item
            return [features for features, _r in env.evaluate(controller, library, probe_level_indices)]

        if not observations:
            choice_name = "food" if "food" in untested else next(iter(untested))
        else:
            scored = []
            for name, (controller, library) in untested.items():
                feats = [outcome for outcome, _r in env.evaluate(controller, library, probe_level_indices)]
                scored.append((discrimination(feats, top), name))
            scored.sort(reverse=True)
            choice_name = scored[0][1]

        controller, library = untested.pop(choice_name)
        order.append(choice_name)
        results = env.evaluate(controller, library, probe_level_indices)
        observations.extend(results)
        probe_episodes += len(results)

        # Convergence check: best goal's free energy margin over runner-up.
        ranked_fe = sorted((goal_free_energy(g, observations, lam), g) for g in candidate_goals())
        if len(ranked_fe) >= 2 and ranked_fe[1][0] - ranked_fe[0][0] >= converge_margin:
            break

    goal, fe = induce_goal(observations, lam)
    return InductionResult(inferred_goal=goal, free_energy=fe, probe_episodes=probe_episodes, rounds=rounds, order=order)

"""
The colimit-cone METHOD, in full generality — separated from any substrate.

A connector provides a substrate by implementing the `Environment` protocol: it
PROVIDES the actions (the morphism primitives the method may compose), the
perception (observation -> named features), and a scalar reward. The method
here never names a substrate primitive — no ARC click, no foraging move, no
colour. It probes with connector-provided actions, induces the hidden goal over
the connector's features by free energy (the same MDL feature selection the
whole repository runs), and composes provided actions into a cone selected by
free energy.

Connectors implementing this contract:
  - cone_foraging / cone_foraging_bound : grid foraging (moves, legs, CALLs)
  - arc_agi3_adapter.ArcEnvironment     : real ARC-AGI-3 (ACTION1-6, click(x,y))
  - the synthetic GoalGame              : a labelled TEST connector

See COLIMIT_CONE_APPROACH.md Section 14 (General Method, Specific Connectors).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import cone_goal_induction as gi  # the substrate-agnostic free-energy core

Action = Hashable          # opaque to the method; the connector defines its meaning
Features = Dict[str, float]
Observation = Tuple[Features, float]  # (features, reward)


@runtime_checkable
class Environment(Protocol):
    """The contract a connector implements. The method consumes only this."""

    def reset(self) -> None: ...
    def actions(self) -> Sequence[Action]: ...   # CONNECTOR PROVIDES THE ACTIONS
    def apply(self, action: Action) -> None: ...
    def features(self) -> Features: ...          # perception (connector-provided)
    def reward(self) -> float: ...
    def done(self) -> bool: ...


# ---------------------------------------------------------------------------
# General drivers: probe, induce, compose — all over the abstract Environment
# ---------------------------------------------------------------------------

def probe(env: Environment, plan: Sequence[Action]) -> Observation:
    """Reset, apply a plan (a sequence of connector-provided actions), and read
    the resulting (features, reward). A plan is the method's unit of behavior;
    its atoms are whatever the connector provides."""
    env.reset()
    for action in plan:
        if env.done():
            break
        env.apply(action)
    return env.features(), env.reward()


def feature_keys(env: Environment) -> List[str]:
    """The feature vocabulary the connector exposes at reset — the keys goal
    induction selects over. The method does not invent these."""
    env.reset()
    return sorted(env.features().keys())


@dataclass
class MethodResult:
    inferred_goal: Tuple[str, ...]
    free_energy: float
    probes: int
    plans_tried: List[Sequence[Action]] = field(default_factory=list)


def induce_goal_over_env(
    env: Environment,
    plans: Sequence[Sequence[Action]],
    repeats: int = 1,
    lam: float = 0.05,
    max_goal_size: int = 2,
) -> MethodResult:
    """Run the method's goal induction over an arbitrary Environment.

    `plans` are candidate behaviors built from connector-provided actions (the
    connector decides what plans are worth trying — e.g. "press each arrow",
    "click each object"). The method probes each plan, collects (features,
    reward), and selects the minimal-free-energy goal over the connector's
    feature vocabulary. Substrate-agnostic: no action is constructed here.
    """
    keys = feature_keys(env)
    observations: List[Observation] = []
    tried: List[Sequence[Action]] = []
    for plan in plans:
        for _ in range(repeats):
            observations.append(probe(env, plan))
        tried.append(plan)
    goal, fe = gi.induce_goal(observations, lam, feature_keys=keys, max_size=max_goal_size)
    return MethodResult(inferred_goal=goal, free_energy=fe, probes=len(observations), plans_tried=tried)


def select_plan_by_free_energy(
    env: Environment,
    plans: Sequence[Sequence[Action]],
    goal: Sequence[str],
    lam: float = 0.05,
    plan_cost=lambda plan: float(len(plan)),
) -> Tuple[Optional[Sequence[Action]], float]:
    """Pick the connector-provided plan that best achieves an inferred goal
    under free energy F = (1 - goal satisfaction) + lambda * description length.
    The method composes provided actions; it never authors them."""
    best_plan: Optional[Sequence[Action]] = None
    best_f = float("inf")
    for plan in plans:
        feats, _reward = probe(env, plan)
        satisfaction = sum(feats.get(k, 0.0) for k in goal) / len(goal) if goal else 0.0
        free_energy = (1.0 - satisfaction) + lam * plan_cost(plan)
        if free_energy < best_f:
            best_f, best_plan = free_energy, plan
    return best_plan, best_f

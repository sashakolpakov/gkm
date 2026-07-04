"""
Foraging connector for the general method (cone_method) — and a demonstration
that a GKM (a cone selected by free energy) drives a working agent through the
same abstract interface the ARC connector uses.

No hand-coded policy. The agent's behavior is produced entirely by:
  1. the connector PROVIDING candidate cones (legs glued into controllers), and
  2. the method INDUCING the hidden goal from reward (free-energy feature
     selection) and SELECTING the cone that achieves it (free-energy plan
     selection).

This is the germane validation: cone construction + selection yields a working
GKM, driven by cone_method, with the substrate behind the Environment contract.
ARC is the same interface; there the open gap is discovering which actions move
the world (the hidden sys_click sprites), not the method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cone_foraging as cf
import cone_foraging_bound as cb
import cone_goal_induction as gi  # outcome_features / task_goal_features / hidden_reward

FEATURE_ORDER = ("food", "safe", "home")
FEATURE_CHANNEL = {"food": cf.FOOD_CHANNEL, "home": cf.HOME_CHANNEL, "safe": cf.HAZARD_CHANNEL}


def goal_library() -> List[cf.Leg]:
    return [cf.witness_seek_leg(), cf.witness_flee_leg()]


def cone_for_features(features: Sequence[str]) -> cb.BoundGenome:
    """A connector-provided cone: one phase per feature, seek the food/home
    channel and flee the hazard channel, composed in a fixed order. Built from
    the leg library — this IS the GKM the method will select among."""
    phases = [f for f in FEATURE_ORDER if f in features]
    rules = []
    for idx, feat in enumerate(phases):
        channel = FEATURE_CHANNEL[feat]
        leg_index = 1 if feat == "safe" else 0  # flee leg for safety, seek otherwise
        rules.append(cb.BoundRule(idx, channel, cb.ANY_OBS, (cb.call_action(leg_index, channel),), idx + 1))
    return cb.BoundGenome(state_count=max(1, len(phases) + 1), rules=rules)


def candidate_cones() -> List[Tuple[Tuple[str, ...], cb.BoundGenome]]:
    """The connector's behavior menu: a cone for every non-empty feature subset
    (up to the three foraging features). The method selects among these."""
    import itertools
    cones = []
    feats = ("food", "home", "safe")
    for size in range(1, len(feats) + 1):
        for subset in itertools.combinations(feats, size):
            cones.append((subset, cone_for_features(subset)))
    return cones


@dataclass
class ForagingEnvironment:
    """Foraging as a cone_method.Environment. The 'actions' it provides are
    whole cones (controllers); applying a cone runs it across the task's levels
    and exposes the mean outcome features and the hidden-goal reward. The method
    never names food/home/safe or seek/flee — it only sees features() and
    reward() and the provided cone-actions."""

    task: cf.TaskSpec
    levels: List[cf.ConeLevel]
    library: List[cf.Leg]
    max_steps: int = 44
    _features: Dict[str, float] = None  # type: ignore[assignment]
    _reward: float = 0.0

    def reset(self) -> None:
        self._features = {f: 0.0 for f in ("food", "home", "safe")}
        self._reward = 0.0

    def actions(self) -> List[Tuple]:
        return [("CONE", subset) for subset, _g in candidate_cones()]

    def _cone(self, subset) -> cb.BoundGenome:
        return cone_for_features(subset)

    def apply(self, action: Tuple) -> None:
        _kind, subset = action
        controller = self._cone(subset)
        feats = {f: 0.0 for f in ("food", "home", "safe")}
        reward = 0.0
        for level in self.levels:
            ep = cb.run_bound_episode(controller, self.library, level, self.task, max_steps=self.max_steps)
            of = gi.outcome_features(ep, level)
            for k in feats:
                feats[k] += of[k]
            reward += gi.hidden_reward(ep, level, self.task)
        n = max(1, len(self.levels))
        self._features = {k: v / n for k, v in feats.items()}
        self._reward = reward / n

    def features(self) -> Dict[str, float]:
        return dict(self._features) if self._features else {f: 0.0 for f in ("food", "home", "safe")}

    def reward(self) -> float:
        return self._reward

    def done(self) -> bool:
        return False

    def solved_fraction(self, subset) -> float:
        controller = self._cone(subset)
        solved = 0
        for level in self.levels:
            ep = cb.run_bound_episode(controller, self.library, level, self.task, max_steps=self.max_steps)
            solved += int(cf.episode_solved(ep, level, self.task))
        return solved / max(1, len(self.levels))

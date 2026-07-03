#!/usr/bin/env python3
"""A GKM driving a working agent through the general cone_method interface.

The agent's behavior is produced entirely by cone construction + free-energy
selection (the GKM), with foraging behind the same Environment contract the ARC
connector implements. No hand-coded policy: the GKM loop induces the hidden goal
from reward and selects the cone that achieves it. Scope: foraging connector.

    python3 experiments/run_method_foraging.py

See COLIMIT_CONE_APPROACH.md Section 14 (General Method, Specific Connectors).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cone_foraging as cf  # noqa: E402
import cone_method as cm  # noqa: E402
import cone_method_foraging as cmf  # noqa: E402

TASKS = ("forage", "homing", "forage_then_home", "forage_flee", "flee_then_home")


def main() -> None:
    print("task,induced_goal,selected_cone,solved")
    for name in TASKS:
        task = cf.TASKS[name]
        levels = cf.make_cone_levels(29, 8, task)
        env = cmf.ForagingEnvironment(task=task, levels=levels, library=cmf.goal_library())
        plans = [[("CONE", subset)] for subset, _g in cmf.candidate_cones()]
        result = cm.induce_goal_over_env(env, plans, lam=0.05, max_goal_size=2)
        chosen, _fe = cm.select_plan_by_free_energy(env, plans, result.inferred_goal, lam=0.02)
        subset = chosen[0][1]
        solved = env.solved_fraction(subset)
        print(f"{name},{'+'.join(result.inferred_goal)},{'+'.join(subset)},{solved:.2f}")


if __name__ == "__main__":
    main()

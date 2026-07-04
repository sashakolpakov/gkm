#!/usr/bin/env python3
"""Goal induction in foraging: infer a hidden goal from scalar reward, compile
it to a cone, verify on held-out levels.

The agent is given only observable outcome features and a scalar reward; it
never reads the task name or its loss. It runs probe cones (built from its leg
library), induces which feature subset the reward tracks by free-energy model
selection, compiles the inferred goal into a bound cone (priced bindings, v3),
and is scored on held-out levels of the hidden task.

    python3 experiments/run_cone_goal_induction.py
    python3 experiments/run_cone_goal_induction.py --lambda-values 0.005,0.02,0.05

See COLIMIT_CONE_APPROACH.md Section 13.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cone_foraging as cf  # noqa: E402
import cone_goal_induction as gi  # noqa: E402

HIDDEN_TASKS = ("forage", "homing", "forage_then_home", "flee", "forage_flee", "flee_then_home")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--lambda-value", type=float, default=0.05)
    parser.add_argument("--lambda-values", default="0.005,0.02,0.05",
                        help="sweep for the confounded-task diagnostic")
    parser.add_argument("--probe-levels", type=int, default=6)
    parser.add_argument("--holdout-levels", type=int, default=6)
    parser.add_argument("--budget", type=int, default=7)
    parser.add_argument("--output-dir", default="output/cone_goal_induction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = args.probe_levels + args.holdout_levels
    probe_idx = list(range(args.probe_levels))
    holdout_idx = list(range(args.probe_levels, total))

    print("hidden_task,true_goal,inferred_goal,match,rounds,probe_episodes,holdout_solved")
    records: List[dict] = []
    for name in HIDDEN_TASKS:
        task = cf.TASKS[name]
        levels = cf.make_cone_levels(args.seed, total, task)
        env = gi.HiddenTask(task, levels)
        result = gi.induce_active(env, probe_idx, lam=args.lambda_value, budget=args.budget)
        controller, library = gi.compile_goal(result.inferred_goal)
        solved = env.solved_fraction(controller, library, holdout_idx)
        true_goal = gi.task_goal_features(task)
        match = set(result.inferred_goal) == set(true_goal)
        print(
            f"{name},{'+'.join(true_goal)},{'+'.join(result.inferred_goal) or 'none'},"
            f"{match},{result.rounds},{result.probe_episodes},{solved:.2f}"
        )
        records.append({
            "hidden_task": name, "true_goal": list(true_goal),
            "inferred_goal": list(result.inferred_goal), "match": match,
            "rounds": result.rounds, "probe_episodes": result.probe_episodes,
            "holdout_solved": solved, "probe_order": result.order,
        })

    # Parsimony diagnostic: the confounded task across a lambda sweep.
    print("\nlambda_sweep (flee_then_home: home implies safe, so goals are near-equivalent)")
    print("lambda,inferred_goal,holdout_solved")
    task = cf.TASKS["flee_then_home"]
    levels = cf.make_cone_levels(args.seed, total, task)
    env = gi.HiddenTask(task, levels)
    observations = []
    for _name, (controller, library) in gi.probe_cones().items():
        observations += env.evaluate(controller, library, probe_idx)
    sweep = []
    for lam in (float(x) for x in args.lambda_values.split(",") if x.strip()):
        goal, _fe = gi.induce_goal(observations, lam)
        controller, library = gi.compile_goal(goal)
        solved = env.solved_fraction(controller, library, holdout_idx)
        print(f"{lam:.3f},{'+'.join(goal) or 'none'},{solved:.2f}")
        sweep.append({"lambda": lam, "inferred_goal": list(goal), "holdout_solved": solved})

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump({"tasks": records, "lambda_sweep": sweep}, handle, indent=2)


if __name__ == "__main__":
    main()

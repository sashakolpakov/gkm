#!/usr/bin/env python3
"""Goal induction lifted onto the ARC connector: infer a hidden game objective
from the scalar score, compile it to a cone over colour slots, verify.

The foraging goal-induction loop (experiments/run_cone_goal_induction.py) runs
unchanged in form here; only the substrate changed: features are now
predicate@colour atoms over scene objects, the reward is a hidden game score,
and the compiled cone issues seek/flee phases bound to colour slots.

    python3 experiments/run_arc_goal_induction.py
    python3 experiments/run_arc_goal_induction.py --lambda-value 0.05

See COLIMIT_CONE_APPROACH.md Section 13.4.
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

import arc_goal_induction as ag  # noqa: E402
import arc_scene_atoms as sa  # noqa: E402

CANDIDATE_COLORS = (2, 3, 5)
HIDDEN_OBJECTIVES = {
    "clear_2": ("clear@2",),
    "avoid_5": ("avoid@5",),
    "clear_2_3": ("clear@2", "clear@3"),
    "clear_2_avoid_5": ("clear@2", "avoid@5"),
    "clear_3_avoid_5": ("clear@3", "avoid@5"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-value", type=float, default=0.05)
    parser.add_argument("--probe-instances", type=int, default=6)
    parser.add_argument("--holdout-instances", type=int, default=6)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--max-goal-size", type=int, default=2)
    parser.add_argument("--oracle-vocabulary", action="store_true",
                        help="use the hand-given clear/avoid atoms instead of discovering them")
    parser.add_argument("--output-dir", default="output/arc_goal_induction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = args.probe_instances + args.holdout_instances
    seeds = tuple(range(total))
    probe_idx = list(range(args.probe_instances))
    holdout_idx = list(range(args.probe_instances, total))

    mode = "oracle vocabulary (hand-given atoms)" if args.oracle_vocabulary \
        else "discovered vocabulary (avatar + colours + atoms from raw frames)"
    print(f"# atom vocabulary: {mode}")
    print("hidden_objective,avatar,discovered_colors,atoms_kept,atoms_pruned,inferred_goal,name_match,cone_match,rounds,probe_episodes,holdout_solved")
    records: List[dict] = []
    for name, objective in HIDDEN_OBJECTIVES.items():
        if args.oracle_vocabulary:
            task = ag.HiddenArcTask(_objective=objective, candidate_colors=CANDIDATE_COLORS, seeds=seeds)
            result = ag.induce_arc_goal(
                task, probe_idx, lam=args.lambda_value, budget=args.budget, max_goal_size=args.max_goal_size
            )
            vocab = None
            avatar = "given"
            colors = list(CANDIDATE_COLORS)
            kept = len(ag.candidate_atoms(CANDIDATE_COLORS))
            pruned = 0
        else:
            result, vocab = ag.discover_and_induce(
                objective, CANDIDATE_COLORS, seeds, probe_idx,
                lam=args.lambda_value, budget=args.budget, max_goal_size=args.max_goal_size,
            )
            task = ag.HiddenArcTask(_objective=objective, candidate_colors=tuple(vocab.colors),
                                    seeds=seeds, vocabulary=vocab)
            avatar = str(vocab.avatar_color)
            colors = vocab.colors
            kept = len(vocab.atoms)
            pruned = len(vocab.pruned)
        phases = ag.goal_to_cone(result.inferred_goal)
        solved = task.solved_fraction(phases, holdout_idx)
        name_match = set(result.inferred_goal) == set(objective)
        cone_match = set(ag.goal_to_cone(objective)) == set(phases)
        print(
            f"{name},{avatar},{'|'.join(map(str, colors))},{kept},{pruned},"
            f"{'+'.join(result.inferred_goal) or 'none'},{name_match},{cone_match},"
            f"{result.rounds},{result.probe_episodes},{solved:.2f}"
        )
        records.append({
            "hidden_objective": name, "true": list(objective),
            "avatar": avatar, "discovered_colors": colors,
            "atoms_kept": kept, "atoms_pruned": pruned,
            "inferred_goal": list(result.inferred_goal),
            "name_match": name_match, "cone_match": cone_match,
            "rounds": result.rounds, "probe_episodes": result.probe_episodes,
            "holdout_solved": solved, "probe_order": result.order,
        })

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(records, handle, indent=2)


if __name__ == "__main__":
    main()

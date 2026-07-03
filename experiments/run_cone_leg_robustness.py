#!/usr/bin/env python3
"""Leg-quality robustness: lifted versus cold-joint versus sequencing-vetted legs.

The replicate sweep in experiments/colimit_cone_foraging_report.md showed that
gluing-search discovery rates collapse when the lifted leg comes from an
overfit inline champion: the cone is only as good as its diagram. This script
tests the proposed fix — vet legs by more tasks during their own evolution —
by comparing, per seed:

    lifted   leg lifted from the inline forage champion (vetted by 1 task)
    joint    leg cold-evolved on {forage, homing} (vetted by 2 tasks)
    seq      leg cold-evolved on {forage, homing, forage_then_home}
             (vetted by 3 tasks including a sequencing consumer; note this
             source has seen the transfer task FAMILY during evolution, so its
             discovery row is in-distribution, unlike lifted and joint)

For each leg: a naturality probe (a one-rule controller calling the leg on
FOOD over forage hidden levels and on HOME over homing hidden levels — the
same leg under both bindings), whether the leg carries a RETURN boundary, and
budget-matched gluing discovery on forage_then_home, against an inline-search
baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cone_foraging as cf  # noqa: E402

CSV_HEADER = (
    "seed,leg_source,def_complexity,has_return,in_distribution,"
    "probe_forage_solved,probe_homing_solved,probe_mean_loss,discovery_successes,discovery_replicates"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="29,101,202,303,404,505")
    parser.add_argument("--lambda-value", type=float, default=0.003)
    parser.add_argument("--population", type=int, default=120)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--joint-population", type=int, default=150)
    parser.add_argument("--joint-generations", type=int, default=80)
    parser.add_argument("--discovery-replicates", type=int, default=5)
    parser.add_argument("--discovery-population", type=int, default=36)
    parser.add_argument("--discovery-generations", type=int, default=14)
    parser.add_argument("--train-count", type=int, default=6)
    parser.add_argument("--validation-count", type=int, default=4)
    parser.add_argument("--hidden-count", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=44)
    parser.add_argument("--output-dir", default="output/colimit_cone")
    return parser.parse_args()


def levels_for(args: argparse.Namespace, seed: int, task_name: str, seed_offset: int):
    task = cf.TASKS[task_name]
    base = seed + 1000 * seed_offset
    return {
        "task": task,
        "train": cf.make_cone_levels(base + 1, args.train_count, task),
        "validation": cf.make_cone_levels(base + 2, args.validation_count, task),
        "hidden": cf.make_cone_levels(base + 3, args.hidden_count, task),
    }


def probe_controller(channel: int) -> cf.ConeGenome:
    return cf.ConeGenome(
        state_count=2,
        rules=[cf.ConeRule(0, cf.ANY_OBS, (cf.call_action(0, channel),), 1)],
    )


def naturality_probe(args: argparse.Namespace, leg: cf.Leg, data: Dict[str, Dict]) -> Tuple[bool, bool, float]:
    forage_eval = cf.evaluate_cone_task(
        probe_controller(cf.FOOD_CHANNEL), [leg],
        data["forage"]["hidden"], data["forage"]["task"], max_steps=args.max_steps,
    )
    homing_eval = cf.evaluate_cone_task(
        probe_controller(cf.HOME_CHANNEL), [leg],
        data["homing"]["hidden"], data["homing"]["task"], max_steps=args.max_steps,
    )
    return forage_eval.solved, homing_eval.solved, (forage_eval.loss + homing_eval.loss) / 2.0


def discovery_rate(
    args: argparse.Namespace,
    seed: int,
    transfer: Dict[str, Dict],
    library: Sequence[cf.Leg],
    allowed: Sequence[int],
    condition: str,
    salt: int,
) -> int:
    successes = 0
    for replicate in range(args.discovery_replicates):
        result = cf.evolve_cone_task(
            transfer["task"],
            transfer["train"],
            allowed,
            library,
            condition,
            args.lambda_value,
            seed=seed + 9000 + 31 * replicate + salt,
            population_size=args.discovery_population,
            generations=args.discovery_generations,
            max_steps=args.max_steps,
        )
        train_eval = cf.evaluate_cone_task(
            result.genome, library, transfer["train"], transfer["task"], max_steps=args.max_steps
        )
        val_eval = cf.evaluate_cone_task(
            result.genome, library, transfer["validation"], transfer["task"], max_steps=args.max_steps
        )
        if train_eval.solved and val_eval.solved:
            successes += 1
    return successes


def main() -> None:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    print(CSV_HEADER)
    records: List[Dict[str, object]] = []

    for seed in seeds:
        data = {
            "forage": levels_for(args, seed, "forage", 1),
            "homing": levels_for(args, seed, "homing", 2),
        }
        seq_extra = levels_for(args, seed, "forage_then_home", 3)
        transfer = levels_for(args, seed, "forage_then_home", 9)

        # Source 1: lift of the inline forage champion.
        inline = cf.evolve_cone_task(
            data["forage"]["task"], data["forage"]["train"],
            cf.inline_actions(), [], "inline", args.lambda_value,
            seed=seed + 13 + 101,
            population_size=args.population, generations=args.generations,
            max_steps=args.max_steps,
        )
        lifted = cf.lift_leg(inline.genome, "lifted")

        # Source 2: cold joint cone on {forage, homing}.
        joint = cf.evolve_joint_cone(
            {name: (data[name]["task"], data[name]["train"]) for name in ("forage", "homing")},
            "shared", args.lambda_value, seed=seed + 31_000,
            population_size=args.joint_population, generations=args.joint_generations,
            max_steps=args.max_steps,
        )
        joint_leg = cf.Leg("joint", joint.legs[0])

        # Source 3: cold joint cone with the sequencing consumer included.
        seq_tasks = {name: (data[name]["task"], data[name]["train"]) for name in ("forage", "homing")}
        seq_tasks["forage_then_home"] = (seq_extra["task"], seq_extra["train"])
        seq = cf.evolve_joint_cone(
            seq_tasks, "shared", args.lambda_value, seed=seed + 32_000,
            population_size=args.joint_population, generations=args.joint_generations,
            max_steps=args.max_steps,
        )
        seq_leg = cf.Leg("seq", seq.legs[0])

        sources: List[Tuple[str, Optional[cf.Leg], bool, int]] = [
            ("lifted", lifted, False, 11),
            ("joint", joint_leg, False, 23),
            ("seq", seq_leg, True, 37),
            ("inline_baseline", None, False, 17),
        ]
        for name, leg, in_distribution, salt in sources:
            if leg is not None:
                probe_forage, probe_homing, probe_loss = naturality_probe(args, leg, data)
                successes = discovery_rate(
                    args, seed, transfer, [leg], cf.controller_actions(1), "shared", salt=salt,
                )
                record = {
                    "seed": seed,
                    "leg_source": name,
                    "def_complexity": cf.leg_def_complexity(leg),
                    "has_return": cf.genome_has_return(leg.genome),
                    "in_distribution": in_distribution,
                    "probe_forage_solved": probe_forage,
                    "probe_homing_solved": probe_homing,
                    "probe_mean_loss": round(probe_loss, 4),
                    "discovery_successes": successes,
                    "discovery_replicates": args.discovery_replicates,
                    "leg_rules": leg.genome.describe(),
                }
            else:
                successes = discovery_rate(
                    args, seed, transfer, [], cf.inline_actions(), "inline", salt=salt,
                )
                record = {
                    "seed": seed,
                    "leg_source": name,
                    "def_complexity": 0.0,
                    "has_return": False,
                    "in_distribution": False,
                    "probe_forage_solved": False,
                    "probe_homing_solved": False,
                    "probe_mean_loss": float("nan"),
                    "discovery_successes": successes,
                    "discovery_replicates": args.discovery_replicates,
                    "leg_rules": [],
                }
            records.append(record)
            print(
                f"{record['seed']},{record['leg_source']},{record['def_complexity']:.1f},"
                f"{record['has_return']},{record['in_distribution']},"
                f"{record['probe_forage_solved']},{record['probe_homing_solved']},"
                f"{record['probe_mean_loss']},{record['discovery_successes']},{record['discovery_replicates']}"
            )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "leg_robustness.json"), "w") as handle:
        json.dump(records, handle, indent=2)


if __name__ == "__main__":
    main()

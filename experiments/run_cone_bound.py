#!/usr/bin/env python3
"""Colimit-cone v3 experiment: priced bindings and within-task factoring.

Tests whether removing free rebinding (channel goes into the rule key; no
SET_FOCUS) makes the true or_factor analogue testable: within a single
two-phase task, does a slot-based leg called twice under priced bindings beat
an inline solver that must duplicate its seek body per channel?

Two modes, mirroring the abstraction-emergence experiment:

  accounting  fix a seek leg (witness or lifted); compare free energy of the
              hand/derived inline solver vs the cone for each task. Isolates
              the accounting from search noise. The headline claim lives here.
  evolved     evolve inline and gluing solutions at matched budgets; report
              selected free energies and the R9 loss-confound honestly.

See COLIMIT_CONE_APPROACH.md Section 11.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cone_foraging as cf  # noqa: E402
import cone_foraging_bound as cb  # noqa: E402

# Single-task scenarios; the within-task factoring question is per task.
SCENARIOS: Dict[str, str] = {
    "single_bound": "forage",               # one channel: control, no factoring
    "two_phase_bound": "forage_then_home",   # SAME leg twice: the or_factor analogue
    "forage_flee_bound": "forage_flee",      # seek + flee: a two-leg library (flee genuinely needed)
}


def witness_library(task: cf.TaskSpec) -> List[cf.Leg]:
    """Seek leg for food/home phases; add a flee leg for hazard phases."""
    if task.requires_safe:
        return [cf.witness_seek_leg(), cf.witness_flee_leg()]
    return [cf.witness_seek_leg()]

ACCOUNTING_HEADER = (
    "mode,scenario,task,condition,binding_cost,train_loss,solved,complexity,free_energy,leg,static_calls"
)


@dataclass
class TaskData:
    task: cf.TaskSpec
    train: List[cf.ConeLevel]
    validation: List[cf.ConeLevel]
    hidden: List[cf.ConeLevel]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS) + ("all",), default="all")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--lambda-values", default="0.001,0.003,0.01")
    parser.add_argument("--population", type=int, default=140)
    parser.add_argument("--generations", type=int, default=70)
    parser.add_argument("--state-count", type=int, default=3)
    parser.add_argument("--max-rules", type=int, default=24)
    parser.add_argument("--max-rule-length", type=int, default=2)
    parser.add_argument("--mutation-rate", type=float, default=0.10)
    parser.add_argument("--call-cost", type=float, default=cb.DEFAULT_CALL_COST)
    parser.add_argument("--binding-cost", type=float, default=cb.DEFAULT_BINDING_COST)
    parser.add_argument("--train-count", type=int, default=6)
    parser.add_argument("--validation-count", type=int, default=4)
    parser.add_argument("--hidden-count", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=44)
    parser.add_argument("--val-tolerance", type=float, default=0.02)
    parser.add_argument("--evolved", action="store_true", help="also run evolved mode")
    parser.add_argument("--binding-free-ablation", action="store_true",
                        help="also report accounting at binding_cost=0 (control)")
    parser.add_argument("--output-dir", default="output/cone_bound")
    parser.add_argument("--show-rules", action="store_true")
    return parser.parse_args()


def make_task_data(args: argparse.Namespace, task_name: str) -> TaskData:
    task = cf.TASKS[task_name]
    base = args.seed
    return TaskData(
        task=task,
        train=cf.make_cone_levels(base + 1, args.train_count, task),
        validation=cf.make_cone_levels(base + 2, args.validation_count, task),
        hidden=cf.make_cone_levels(base + 3, args.hidden_count, task),
    )


@dataclass
class Row:
    mode: str
    scenario: str
    task: str
    condition: str
    binding_cost: float
    train_loss: float
    solved: bool
    complexity: float
    free_energy: float
    leg: str
    static_calls: int


def print_row(row: Row) -> None:
    print(
        f"{row.mode},{row.scenario},{row.task},{row.condition},{row.binding_cost:.2f},"
        f"{row.train_loss:.4f},{row.solved},{row.complexity:.2f},{row.free_energy:.4f},"
        f"{row.leg},{row.static_calls}"
    )


def accounting_rows(
    args: argparse.Namespace,
    scenario: str,
    data: TaskData,
    binding_cost: float,
    lambda_value: float,
) -> List[Row]:
    """Accounting mode: hand-constructed inline vs cone, fixed witness legs."""
    task = data.task
    library = witness_library(task)
    leg_label = "+".join(leg.name for leg in library)
    inline = cb.witness_inline(task)
    cone = cb.witness_bound_gluing(task, seek_index=0, flee_index=1)

    def make_row(condition: str, genome: cb.BoundGenome, lib: Sequence[cf.Leg]) -> Row:
        evaluation = cb.evaluate_bound_task(genome, lib, data.train, task, max_steps=args.max_steps)
        complexity = cb.bound_cone_complexity(
            [genome], list(lib), condition,
            call_cost=args.call_cost, binding_cost=binding_cost,
        )
        return Row(
            mode="accounting", scenario=scenario, task=task.name, condition=condition,
            binding_cost=binding_cost, train_loss=evaluation.loss, solved=evaluation.solved,
            complexity=complexity, free_energy=evaluation.loss + lambda_value * complexity,
            leg=leg_label if condition in ("shared", "no_share") else "none",
            static_calls=len(genome.call_references()),
        )

    return [
        make_row("inline", inline, []),
        make_row("shared", cone, library),
        make_row("no_share", cone, library),
    ]


def evolved_rows(
    args: argparse.Namespace,
    scenario: str,
    data: TaskData,
    lambda_values: Sequence[float],
) -> List[Row]:
    """Evolved mode: evolve inline and gluing solutions, select by validation
    elbow, report selected free energy. The seek leg is the witness (so the
    comparison is gluing-search vs inline-search, leg held fixed)."""
    task = data.task
    library = witness_library(task)
    leg_count = len(library)
    rows: List[Row] = []

    def select(condition: str, allowed, library) -> Row:
        scored: List[Tuple[float, float, float, bool, cb.BoundGenome]] = []
        for lambda_idx, lambda_value in enumerate(lambda_values):
            result = cb.evolve_bound_task(
                task, data.train, allowed, library, condition, lambda_value,
                seed=args.seed + 700 * (lambda_idx + 1) + (97 if condition == "no_share" else 0),
                population_size=args.population, generations=args.generations,
                state_count=args.state_count, max_rules=args.max_rules,
                max_rule_length=args.max_rule_length, mutation_rate=args.mutation_rate,
                call_cost=args.call_cost, binding_cost=args.binding_cost, max_steps=args.max_steps,
            )
            val = cb.evaluate_bound_task(result.genome, library, data.validation, task, max_steps=args.max_steps)
            complexity = cb.bound_cone_complexity(
                [result.genome], list(library), condition,
                call_cost=args.call_cost, binding_cost=args.binding_cost,
            )
            scored.append((val.loss, complexity, result.train_loss, result.genome, val.solved))
        best_val = min(s[0] for s in scored)
        allowed_scored = [s for s in scored if s[0] <= best_val + args.val_tolerance]
        chosen = min(allowed_scored, key=lambda s: (s[1], s[0]))
        val_loss, complexity, train_loss, genome, solved = chosen
        return Row(
            mode="evolved", scenario=scenario, task=task.name, condition=condition,
            binding_cost=args.binding_cost, train_loss=train_loss, solved=solved,
            complexity=complexity, free_energy=train_loss + lambda_values[0] * complexity,
            leg=("+".join(l.name for l in library)
                 if condition in ("shared", "no_share") and genome.call_references() else "none"),
            static_calls=len(genome.call_references()),
        )

    rows.append(select("inline", cb.bound_inline_actions(), []))
    rows.append(select("shared", cb.bound_controller_actions(leg_count), library))
    rows.append(select("no_share", cb.bound_controller_actions(leg_count), library))
    return rows


def main() -> None:
    args = parse_args()
    lambda_values = tuple(float(x) for x in args.lambda_values.split(",") if x.strip())
    scenarios = tuple(SCENARIOS) if args.scenario == "all" else (args.scenario,)
    mid_lambda = lambda_values[len(lambda_values) // 2]

    print(ACCOUNTING_HEADER)
    all_rows: List[Row] = []
    for scenario in scenarios:
        data = make_task_data(args, SCENARIOS[scenario])
        rows = accounting_rows(args, scenario, data, args.binding_cost, mid_lambda)
        if args.binding_free_ablation:
            rows += accounting_rows(args, scenario, data, 0.0, mid_lambda)
        if args.evolved:
            rows += evolved_rows(args, scenario, data, lambda_values)
        all_rows.extend(rows)
        for row in rows:
            print_row(row)

    if args.show_rules:
        task = cf.TASKS["forage_then_home"]
        print("\n# inline witness (duplicated seek body, per channel):")
        for line in cb.witness_inline(task).describe():
            print(f"  {line}")
        print("# cone witness (seek leg called twice under priced bindings):")
        for line in cb.witness_bound_gluing(task).describe():
            print(f"  {line}")
        print("# seek leg (channel-blind, shared):")
        for line in cf.witness_seek_leg().genome.describe():
            print(f"  {line}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump([vars(r) for r in all_rows], handle, indent=2)


if __name__ == "__main__":
    main()

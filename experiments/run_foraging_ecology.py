#!/usr/bin/env python3
"""Run the grid-foraging sparse-FSA ecology experiment.

This file is the experiment runner. The core substrate lives in ``evo_game.py``
and is intentionally importable without printing experiment progress.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evo_game import (  # noqa: E402
    COMPLEXITY_MODES,
    fmin,
    evolve,
    lambda_sweep,
    make_lambda_values,
    make_levels,
    render_episode,
    run_episode,
    save_outputs,
)


def print_lambda_record(record) -> None:
    print(
        f"lambda={record.lambda_value:.4f} "
        f"L_train={record.train_loss:.4f} "
        f"L_val={record.val_loss:.4f} "
        f"F_val={record.val_free_energy:.4f} "
        f"C_{record.complexity_mode}={record.complexity:.1f} "
        f"Ca={record.active_complexity:.1f} "
        f"Cp={record.pruned_complexity:.1f} "
        f"Ct={record.table_complexity:.1f} "
        f"chi_C={record.complexity_variance:.5f} "
        f"val_food={record.val_food:.2f} "
        f"active={record.active_states}/{record.active_rules} "
        f"reachable={record.reachable_states}/{record.reachable_rules}"
    )


def print_generation(record) -> None:
    print(
        f"gen={record.generation:03d} "
        f"F={record.best_free_energy:.4f} "
        f"L={record.best_loss:.4f} "
        f"train_food={record.train_food:.2f} "
        f"val_food={record.val_food:.2f} "
        f"C_{record.complexity_mode}={record.complexity:.1f} "
        f"Ca={record.active_complexity:.1f} "
        f"Ct={record.table_complexity:.1f} "
        f"active={record.active_states}/{record.active_rules}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sparse-FSA grid-foraging ecology experiment.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=160)
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--max-states", type=int, default=None)
    parser.add_argument("--initial-rules", type=int, default=None)
    parser.add_argument("--max-rules", type=int, default=None)
    parser.add_argument("--width", type=int, default=7)
    parser.add_argument("--height", type=int, default=7)
    parser.add_argument("--food-count", type=int, default=4)
    parser.add_argument("--train-levels", type=int, default=12)
    parser.add_argument("--val-levels", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--max-rule-length", type=int, default=1)
    parser.add_argument("--mutation-rate", type=float, default=0.04)
    parser.add_argument("--complexity-weight", type=float, default=0.002)
    parser.add_argument("--lambda-min", type=float, default=0.0)
    parser.add_argument("--lambda-max", type=float, default=0.01)
    parser.add_argument("--lambda-points", type=int, default=5)
    parser.add_argument(
        "--complexity-mode",
        choices=COMPLEXITY_MODES,
        default="table",
        help="Complexity term optimized inside free energy.",
    )
    parser.add_argument("--optimizer", choices=["genetic", "hyperopt"], default="genetic")
    parser.add_argument("--hyperopt-evals", type=int, default=200)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--render", action="store_true", help="Print before/after ASCII replays.")
    parser.add_argument("--output-dir", default="./output/evo_game")
    return parser.parse_args()


def print_progress(history, lambda_records, report_every: int) -> None:
    if report_every:
        for record in history:
            if record.generation == 0 or record.generation % report_every == 0:
                print_generation(record)
    for idx, record in enumerate(lambda_records, 1):
        print(f"\n=== lambda {idx}/{len(lambda_records)}: {record.lambda_value:.4f} ===")
        print_lambda_record(record)


def main() -> None:
    args = parse_args()
    train_levels = make_levels(
        args.seed + 100,
        count=args.train_levels,
        width=args.width,
        height=args.height,
        food_count=args.food_count,
    )
    val_levels = make_levels(
        args.seed + 200,
        count=args.val_levels,
        width=args.width,
        height=args.height,
        food_count=args.food_count,
    )
    lambda_values = make_lambda_values(args.lambda_min, args.lambda_max, args.lambda_points)

    if args.optimizer == "hyperopt" and fmin is None:
        raise RuntimeError("hyperopt is not installed. Install dependencies with: pip install -r requirements.txt")

    initial_best, _initial_selected, _initial_history, _initial_train, _initial_val = evolve(
        seed=args.seed,
        generations=0,
        population_size=max(2, min(args.population, 50)),
        state_count=args.states,
        complexity_mode=args.complexity_mode,
        max_steps=args.max_steps,
        max_rule_length=args.max_rule_length,
        initial_rule_count=args.initial_rules,
        max_rules=args.max_rules,
        max_states=args.max_states,
        train_levels=train_levels,
        val_levels=val_levels,
        report_every=0,
    )

    best, lambda_records, history, train_eval, val_eval = lambda_sweep(
        seed=args.seed,
        lambda_values=lambda_values,
        optimizer=args.optimizer,
        generations=args.generations,
        population_size=args.population,
        state_count=args.states,
        mutation_rate=args.mutation_rate,
        complexity_weight=args.complexity_weight,
        complexity_mode=args.complexity_mode,
        max_steps=args.max_steps,
        max_rule_length=args.max_rule_length,
        initial_rule_count=args.initial_rules,
        max_rules=args.max_rules,
        max_states=args.max_states,
        hyperopt_evals=args.hyperopt_evals,
        train_levels=train_levels,
        val_levels=val_levels,
        report_every=args.report_every,
    )

    print_progress(history, lambda_records, args.report_every)

    replay_level = val_levels[0]
    initial_episode = run_episode(initial_best, replay_level, max_steps=args.max_steps)
    best_episode = run_episode(best, replay_level, max_steps=args.max_steps)
    best_replay = render_episode(replay_level, best_episode)

    print("\nFinal evaluation")
    print(f"  ecology:         {args.width}x{args.height}, food={args.food_count}, max_steps={args.max_steps}")
    print(f"  rule max length: {args.max_rule_length}")
    print(f"  encoded rules:   {val_eval.table_rules}")
    print(f"  selected lambda: {val_eval.lambda_value:.4f}")
    print(f"  complexity mode: {val_eval.complexity_mode}")
    print(f"  train loss:      {train_eval.loss:.4f}")
    print(f"  val loss:        {val_eval.loss:.4f}")
    print(f"  val free energy: {val_eval.free_energy:.4f}")
    print(f"  train mean food: {train_eval.mean_collected:.2f}/{len(train_levels[0].food)}")
    print(f"  val mean food:   {val_eval.mean_collected:.2f}/{len(val_levels[0].food)}")
    print(f"  selected C:      {val_eval.complexity:.1f}")
    print(f"  active C:        {val_eval.active_complexity:.1f} ({val_eval.active_states} states, {val_eval.active_rules} rules)")
    print(f"  pruned C:        {val_eval.pruned_complexity:.1f} ({val_eval.reachable_states} states, {val_eval.reachable_rules} rules)")
    print(f"  table C:         {val_eval.table_complexity:.1f} ({val_eval.table_states} states, {val_eval.table_rules} rules)")
    print(f"  mixed C:         {val_eval.mixed_complexity:.1f}")

    if args.render:
        print("\nInitial best replay")
        print(render_episode(replay_level, initial_episode))
        print("\nFinal best replay")
        print(best_replay)

    save_outputs(args.output_dir, best, history, train_eval, val_eval, best_replay, lambda_records)
    print(textwrap.dedent(f"""
        Saved:
          {os.path.join(args.output_dir, 'best_automaton.py')}
          {os.path.join(args.output_dir, 'evolution_history.json')}
          {os.path.join(args.output_dir, 'lambda_sweep.json')}
          {os.path.join(args.output_dir, 'summary.json')}
          {os.path.join(args.output_dir, 'best_replay.txt')}
    """).strip())


if __name__ == "__main__":
    main()

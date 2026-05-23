#!/usr/bin/env python3
"""Paired developmental-overcapacity ablations for sparse Bongard classifiers.

Each condition keeps the task and optimizer family fixed while changing how much
encoded rule capacity the genome may carry during search. The relevant test is
not whether a limited cap can ever solve a task, but how often it solves compared
with a larger developmental search space that is still charged by lambda C.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_bongard_sparse_classifier import CONFIGS, ExperimentConfig, is_exact_discovery, run_config


@dataclass(frozen=True)
class Condition:
    task: str
    name: str
    updates: Dict[str, int | float | bool]


def base_config(task: str, quick: bool) -> ExperimentConfig:
    config = next(item for item in CONFIGS if item.concept == task)
    if not quick:
        return config
    quick_budgets = {
        "length_even": {"population": 180, "generations": 90, "replicates": 3},
        "length_multiple_of_three": {"population": 220, "generations": 120, "replicates": 3},
        "first_equals_second": {"population": 280, "generations": 160, "replicates": 3},
        "last_two_equal": {"population": 360, "generations": 220, "replicates": 3},
        "has_adjacent_duplicate": {"population": 220, "generations": 140, "replicates": 3},
        "first_equals_last": {"population": 700, "generations": 450, "replicates": 3},
        "second_equals_last": {"population": 800, "generations": 520, "replicates": 3},
        "first_equals_penultimate": {"population": 800, "generations": 520, "replicates": 3},
        "second_equals_penultimate": {"population": 850, "generations": 560, "replicates": 3},
    }
    return replace(config, **quick_budgets.get(task, {"replicates": 3}))


def conditions_for(task: str) -> List[Condition]:
    if task == "length_even":
        return [
            Condition(task, "cap_3_rules", {"states": 3, "initial_rules": 3, "max_rules": 3, "max_rule_length": 2}),
            Condition(task, "overcapacity_10_rules", {"states": 3, "initial_rules": 6, "max_rules": 10, "max_rule_length": 3}),
        ]
    if task == "length_multiple_of_three":
        return [
            Condition(task, "cap_4_rules", {"states": 4, "initial_rules": 4, "max_rules": 4, "max_rule_length": 2}),
            Condition(task, "overcapacity_12_rules", {"states": 4, "initial_rules": 8, "max_rules": 12, "max_rule_length": 2}),
        ]
    if task == "first_equals_second":
        return [
            Condition(task, "cap_4_rules", {"states": 3, "initial_rules": 4, "max_rules": 4, "max_rule_length": 1}),
            Condition(task, "overcapacity_14_rules", {"states": 3, "initial_rules": 8, "max_rules": 14, "max_rule_length": 1}),
        ]
    if task == "last_two_equal":
        return [
            Condition(task, "cap_5_rules", {"states": 3, "initial_rules": 5, "max_rules": 5, "max_rule_length": 1}),
            Condition(task, "overcapacity_14_rules", {"states": 3, "initial_rules": 8, "max_rules": 14, "max_rule_length": 1}),
        ]
    if task == "has_adjacent_duplicate":
        return [
            Condition(task, "cap_4_rules", {"states": 3, "initial_rules": 4, "max_rules": 4, "max_rule_length": 3}),
            Condition(task, "overcapacity_12_rules", {"states": 3, "initial_rules": 8, "max_rules": 12, "max_rule_length": 3}),
        ]
    if task == "first_equals_last":
        return [
            Condition(task, "cap_5_rules", {"states": 3, "initial_rules": 5, "max_rules": 5, "max_rule_length": 1}),
            Condition(task, "cap_6_rules", {"states": 3, "initial_rules": 6, "max_rules": 6, "max_rule_length": 1}),
            Condition(task, "overcapacity_12_rules", {"states": 3, "initial_rules": 6, "max_rules": 12, "max_rule_length": 1}),
        ]
    if task == "second_equals_last":
        return [
            Condition(task, "cap_6_rules", {"states": 4, "initial_rules": 6, "max_rules": 6, "max_rule_length": 1}),
            Condition(task, "overcapacity_16_rules", {"states": 4, "initial_rules": 8, "max_rules": 16, "max_rule_length": 1}),
        ]
    if task == "first_equals_penultimate":
        return [
            Condition(task, "cap_8_rules", {"states": 4, "initial_rules": 8, "max_rules": 8, "max_rule_length": 1}),
            Condition(task, "overcapacity_18_rules", {"states": 4, "initial_rules": 8, "max_rules": 18, "max_rule_length": 1}),
        ]
    if task == "second_equals_penultimate":
        return [
            Condition(task, "cap_8_rules", {"states": 4, "initial_rules": 8, "max_rules": 8, "max_rule_length": 1}),
            Condition(task, "overcapacity_18_rules", {"states": 4, "initial_rules": 8, "max_rules": 18, "max_rule_length": 1}),
        ]
    raise ValueError(f"unsupported overcapacity ablation task: {task}")


def run_condition(condition: Condition, replicates: int, quick: bool, stream: bool) -> List[dict]:
    config = replace(base_config(condition.task, quick), **condition.updates)
    rows = []
    for replicate in range(replicates):
        record, _genome, _primitives = run_config(config, [config.lambda_min], replicate=replicate)
        row = {
                "task": condition.task,
                "condition": condition.name,
                "replicate": replicate,
                "exact": is_exact_discovery(record),
                "train": record.train_accuracy,
                "val": record.validation_accuracy,
                "hidden": record.hidden_accuracy,
                "probe": record.probe_accuracy,
                "probe_bal": record.probe_balanced_accuracy,
                "complexity": record.complexity,
                "rules": record.rules,
            }
        rows.append(row)
        if stream:
            print(format_row(row), flush=True)
    return rows


def fmt_bool(value: bool) -> str:
    return "True" if value else "False"


def format_row(row: dict) -> str:
    return (
        f"{row['task']},{row['condition']},{row['replicate']},{fmt_bool(row['exact'])},"
        f"{row['train']:.2f},{row['val']:.2f},{row['hidden']:.2f},{row['probe']:.2f},"
        f"{row['probe_bal']:.2f},{row['complexity']:.1f},{row['rules']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    task_choices = (
        "length_even",
        "length_multiple_of_three",
        "first_equals_second",
        "last_two_equal",
        "has_adjacent_duplicate",
        "first_equals_last",
        "second_equals_last",
        "first_equals_penultimate",
        "second_equals_penultimate",
    )
    parser.add_argument("--task", choices=task_choices, action="append")
    parser.add_argument("--include-slow", action="store_true", help="include slower boundary-offset tasks in the default matrix")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--full", action="store_true", help="use configured default budgets instead of quick documented budgets")
    parser.add_argument("--no-stream", action="store_true", help="print per-run rows only after all runs finish")
    args = parser.parse_args()

    if args.task:
        tasks = args.task
    elif args.include_slow:
        tasks = list(task_choices)
    else:
        tasks = [
            "length_even",
            "length_multiple_of_three",
            "first_equals_second",
            "last_two_equal",
            "has_adjacent_duplicate",
            "first_equals_last",
        ]
    all_rows: List[dict] = []
    stream = not args.no_stream

    print("per_run")
    print("task,condition,replicate,exact,train,val,hidden,probe,probe_bal,complexity,rules")
    for task in tasks:
        for condition in conditions_for(task):
            all_rows.extend(run_condition(condition, args.replicates, quick=not args.full, stream=stream))

    if not stream:
        for row in all_rows:
            print(format_row(row))

    print("\nsummary")
    print("task,condition,runs,exact_discoveries,mean_probe_bal,mean_complexity,mean_rules")
    keys = sorted({(row["task"], row["condition"]) for row in all_rows})
    for task, condition in keys:
        rows = [row for row in all_rows if row["task"] == task and row["condition"] == condition]
        print(
            f"{task},{condition},{len(rows)},{sum(row['exact'] for row in rows)},"
            f"{statistics.mean(row['probe_bal'] for row in rows):.3f},"
            f"{statistics.mean(row['complexity'] for row in rows):.1f},"
            f"{statistics.mean(row['rules'] for row in rows):.1f}"
        )


if __name__ == "__main__":
    main()

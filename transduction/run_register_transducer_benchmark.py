#!/usr/bin/env python3
"""Reproduce the sparse register-transducer benchmark matrix.

Run from the repository root:

    python3 experiments/run_register_transducer_benchmark.py
"""

from __future__ import annotations

import contextlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pattern_fsa import (  # noqa: E402
    PRIMITIVE_SETS,
    export_solver,
    lambda_sweep_solver,
    make_lambda_values,
    make_object_task,
)


@dataclass(frozen=True)
class BenchmarkConfig:
    task: str
    primitive: str
    length: int
    registers: int
    generations: int
    population: int
    states: int
    max_rules: int
    max_rule_length: int
    lambda_max: float = 0.01


CONFIGS = [
    BenchmarkConfig("copy", "stream", 3, 0, 80, 160, 3, 10, 3),
    BenchmarkConfig("swap", "stream", 2, 0, 80, 160, 3, 8, 3),
    BenchmarkConfig("swap", "register", 2, 1, 120, 220, 3, 10, 3),
    BenchmarkConfig("duplicate_first", "stream", 2, 0, 100, 180, 3, 10, 3),
    BenchmarkConfig("rotate_left", "stream", 3, 0, 100, 180, 4, 12, 3),
    BenchmarkConfig("rotate_left", "register", 3, 1, 450, 500, 4, 16, 4, lambda_max=0.006),
    BenchmarkConfig("reverse", "register", 3, 2, 220, 320, 4, 16, 4),
    BenchmarkConfig("reverse", "bidirectional", 5, 0, 180, 260, 3, 10, 2, lambda_max=0.006),
    BenchmarkConfig("dedupe_pair", "register", 2, 1, 120, 220, 3, 10, 3),
    BenchmarkConfig("dedupe_pair", "compare", 2, 1, 140, 240, 3, 10, 3),
]


def run_config(config: BenchmarkConfig) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    task = make_object_task(
        config.task,
        seed=7,
        train_examples=10,
        val_examples=6,
        test_examples=6,
        train_objects=10,
        val_objects=10,
        test_objects=10,
        length=config.length,
    )
    primitives = PRIMITIVE_SETS[config.primitive](task.alphabet_size, register_count=config.registers)
    lambda_values = make_lambda_values(0.0, config.lambda_max, 4)

    with contextlib.redirect_stdout(io.StringIO()):
        genome, records, _history, train_eval, val_eval, test_eval = lambda_sweep_solver(
            task=task,
            primitives=primitives,
            lambda_values=lambda_values,
            seed=7,
            generations=config.generations,
            population_size=config.population,
            state_count=config.states,
            max_states=config.states,
            initial_rule_count=min(6, config.states * primitives.observation_count),
            max_rules=config.max_rules,
            max_rule_length=config.max_rule_length,
            mutation_rate=0.08,
            max_steps=32,
            validation_loss_tolerance=0.075,
            report_every=0,
        )

    selected = next(record for record in records if record.selected)
    row = {
        "task": config.task,
        "primitive": config.primitive,
        "length": config.length,
        "registers": config.registers,
        "lambda": selected.lambda_value,
        "train_exact": train_eval.exact_match_rate,
        "val_exact": val_eval.exact_match_rate,
        "test_exact": test_eval.exact_match_rate,
        "test_loss": test_eval.loss,
        "complexity": test_eval.complexity,
        "rules": test_eval.encoded_rules,
        "active_rules": test_eval.active_rules,
    }
    return row, export_solver(genome, primitives)["rules"]


def main() -> None:
    rows = []
    selected_rules = {}
    for config in CONFIGS:
        row, rules = run_config(config)
        rows.append(row)
        if row["test_exact"] == 1.0 or config.task in {"swap", "dedupe_pair"}:
            selected_rules[(config.task, config.primitive)] = rules

    print("task,primitive,length,registers,lambda,train_exact,val_exact,test_exact,test_loss,complexity,rules,active_rules")
    for row in rows:
        print(
            f"{row['task']},{row['primitive']},{row['length']},{row['registers']},"
            f"{row['lambda']:.4f},{row['train_exact']:.2f},{row['val_exact']:.2f},"
            f"{row['test_exact']:.2f},{row['test_loss']:.4f},{row['complexity']:.1f},"
            f"{row['rules']},{row['active_rules']}"
        )

    print("\nselected_solver_rules")
    for (task, primitive), rules in selected_rules.items():
        print(f"{task} + {primitive}:")
        for rule in rules:
            print(f"  s{rule['state']}:{rule['observation_name']} -> {rule['actions']} / s{rule['next_state']}")


if __name__ == "__main__":
    main()
